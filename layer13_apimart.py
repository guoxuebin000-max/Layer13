import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO

import numpy as np
import torch
from PIL import Image


DEFAULT_BASE_URL = "https://api.apimart.ai/v1"
DEFAULT_APIMART_PROMPT = (
    "Create or edit the image with natural photographic detail. "
    "If input images are provided, preserve the main subject, identity, composition, and style."
)


def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError("图像必须是 ComfyUI IMAGE 张量。")
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("图像必须是 IMAGE 批量张量 (B,H,W,C)，且至少包含 RGB 三通道。")
    return image


def _tensor_to_pil(sample: torch.Tensor, fmt: str = "PNG", quality: int = 90) -> tuple[bytes, str]:
    data = sample.detach().float().cpu().clamp(0.0, 1.0)
    array = (data.numpy() * 255.0 + 0.5).astype("uint8")

    if array.shape[-1] >= 4:
        image = Image.fromarray(array[..., :4], "RGBA")
    else:
        image = Image.fromarray(array[..., :3], "RGB")

    fmt = str(fmt or "PNG").upper()
    buffer = BytesIO()
    if fmt == "JPEG":
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        image.save(buffer, format="JPEG", quality=max(1, min(100, int(quality))), optimize=True)
        return buffer.getvalue(), "image/jpeg"

    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), "image/png"


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    data = torch.from_numpy(np.asarray(image).copy()).to(dtype=torch.float32)
    return data / 255.0


def _blank_image() -> torch.Tensor:
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def _clean_base_url(base_url: str) -> str:
    value = str(base_url or DEFAULT_BASE_URL).strip()
    if not value:
        value = DEFAULT_BASE_URL
    return value.rstrip("/")


def _json_object(text: str, default):
    raw = str(text or "").strip()
    if not raw:
        return default
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc
    return value


def _config_value(config, name: str, default):
    if isinstance(config, dict):
        return config.get(name, default)
    return default


def _api_key(config: dict) -> str:
    direct_key = str(_config_value(config, "api_key", "") or "").strip()
    if direct_key:
        return direct_key

    env_name = str(_config_value(config, "api_key_env", "APIMART_API_KEY")).strip()
    if not env_name:
        env_name = "APIMART_API_KEY"
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ValueError(
            f"未找到环境变量 {env_name}。请先在启动 ComfyUI 前设置："
            f'export {env_name}="你的 API Mart Key"'
        )
    return value


def _headers(config: dict, content_type: str | None = "application/json") -> dict:
    headers = {
        "Authorization": f"Bearer {_api_key(config)}",
        "Accept": "application/json",
        "User-Agent": "Layer13-ComfyUI-APIMart/1.0",
    }
    if content_type:
        headers["Content-Type"] = content_type

    extra = _config_value(config, "headers", {})
    if isinstance(extra, dict):
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def _http_json(method: str, url: str, headers: dict, body: bytes | None, timeout: int):
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API Mart HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API Mart 连接失败：{exc}") from exc

    text = payload.decode("utf-8", errors="replace")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return value


def _chat_choices_from_chunk(chunk):
    if isinstance(chunk, dict):
        if isinstance(chunk.get("data"), dict):
            choices = chunk["data"].get("choices")
            if isinstance(choices, list):
                return choices
        choices = chunk.get("choices")
        if isinstance(choices, list):
            return choices
    return []


def _post_chat_stream(config: dict, data: dict) -> str:
    base_url = _clean_base_url(_config_value(config, "base_url", DEFAULT_BASE_URL))
    timeout = int(_config_value(config, "timeout", 120))
    payload = dict(data)
    payload["stream"] = True
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        method="POST",
        headers=_headers(config),
    )

    parts = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                event = line[5:].strip()
                if event == "[DONE]":
                    break
                try:
                    chunk = json.loads(event)
                except json.JSONDecodeError:
                    continue
                for choice in _chat_choices_from_chunk(chunk):
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    message = choice.get("message") if isinstance(choice, dict) else None
                    content = ""
                    if isinstance(delta, dict):
                        content = delta.get("content") or ""
                    if not content and isinstance(message, dict):
                        content = message.get("content") or ""
                    if content:
                        parts.append(str(content))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API Mart Chat HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API Mart Chat 连接失败：{exc}") from exc

    return "".join(parts)


def _post_json(config: dict, path: str, data: dict):
    base_url = _clean_base_url(_config_value(config, "base_url", DEFAULT_BASE_URL))
    timeout = int(_config_value(config, "timeout", 120))
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    return _http_json("POST", f"{base_url}{path}", _headers(config), body, timeout)


def _get_json(config: dict, path: str, params: dict | None = None):
    base_url = _clean_base_url(_config_value(config, "base_url", DEFAULT_BASE_URL))
    timeout = int(_config_value(config, "timeout", 120))
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params)
    return _http_json("GET", f"{base_url}{path}{query}", _headers(config, None), None, timeout)


def _post_multipart_file(config: dict, path: str, file_bytes: bytes, content_type: str, filename: str):
    base_url = _clean_base_url(_config_value(config, "base_url", DEFAULT_BASE_URL))
    timeout = int(_config_value(config, "timeout", 120))
    boundary = f"----Layer13APIMart{int(time.time() * 1000)}"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = prefix + file_bytes + suffix
    headers = _headers(config, f"multipart/form-data; boundary={boundary}")
    return _http_json("POST", f"{base_url}{path}", headers, body, timeout)


def _flatten_urls(value) -> list[str]:
    urls = []
    if isinstance(value, str):
        if value.startswith(("http://", "https://", "data:image/")):
            urls.append(value)
    elif isinstance(value, list):
        for item in value:
            urls.extend(_flatten_urls(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"url", "image_url", "output_url", "src"}:
                urls.extend(_flatten_urls(item))
            else:
                urls.extend(_flatten_urls(item))
    return urls


def _unique(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _extract_urls(response) -> list[str]:
    return _unique(_flatten_urls(response))


def _extract_image_urls_from_text(text: str) -> list[str]:
    raw = str(text or "")
    urls = []
    urls.extend(re.findall(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", raw))
    urls.extend(re.findall(r"https?://[^\s<>'\")\]]+", raw))
    cleaned = [url.rstrip(".,;:!?，。；：！？") for url in urls]
    return _unique(cleaned)


def _extract_task_id(response) -> str:
    if isinstance(response, dict):
        for key in ("task_id", "taskId", "id"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in response.values():
            task_id = _extract_task_id(value)
            if task_id:
                return task_id
    if isinstance(response, list):
        for item in response:
            task_id = _extract_task_id(item)
            if task_id:
                return task_id
    return ""


def _extract_status(response) -> str:
    if isinstance(response, dict):
        for key in ("status", "state", "task_status"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        for value in response.values():
            status = _extract_status(value)
            if status:
                return status
    if isinstance(response, list):
        for item in response:
            status = _extract_status(item)
            if status:
                return status
    return ""


def _extract_base64_images(value) -> list[str]:
    images = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"b64_json", "base64", "image_base64"} and isinstance(item, str):
                images.append(item)
            else:
                images.extend(_extract_base64_images(item))
    elif isinstance(value, list):
        for item in value:
            images.extend(_extract_base64_images(item))
    return images


def _load_image_from_url(url: str) -> Image.Image:
    if url.startswith("data:image/"):
        _, encoded = url.split(",", 1)
        raw = base64.b64decode(encoded)
        image = Image.open(BytesIO(raw))
        image.load()
        return image.convert("RGB")

    request = urllib.request.Request(url, headers={"User-Agent": "Layer13-ComfyUI-APIMart/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read()
    image = Image.open(BytesIO(raw))
    image.load()
    return image.convert("RGB")


def _load_image_from_base64(text: str) -> Image.Image:
    value = str(text or "").strip()
    if value.startswith("data:image/"):
        return _load_image_from_url(value)
    raw = base64.b64decode(value)
    image = Image.open(BytesIO(raw))
    image.load()
    return image.convert("RGB")


def _images_from_response(response) -> tuple[torch.Tensor, list[str]]:
    urls = _extract_urls(response)
    images = []
    for url in urls:
        try:
            images.append(_load_image_from_url(url))
        except Exception:
            continue

    for encoded in _extract_base64_images(response):
        try:
            images.append(_load_image_from_base64(encoded))
        except Exception:
            continue

    if not images:
        return _blank_image(), urls

    size = images[0].size
    tensors = []
    for image in images:
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        tensors.append(_pil_to_tensor(image))
    return torch.stack(tensors, dim=0), urls


def _status_is_done(status: str) -> bool:
    return status in {"completed", "complete", "succeeded", "success", "done", "finished"}


def _status_is_failed(status: str) -> bool:
    return status in {"failed", "failure", "error", "cancelled", "canceled", "timeout"}


def _poll_task(config: dict, task_id: str):
    max_wait = float(_config_value(config, "max_wait", 600))
    interval = max(0.5, float(_config_value(config, "poll_interval", 3.0)))
    first_wait = max(0.0, float(_config_value(config, "first_wait", 5.0)))
    language = str(_config_value(config, "language", "zh")).strip() or "zh"

    deadline = time.time() + max_wait
    if first_wait:
        time.sleep(min(first_wait, max_wait))

    last_response = None
    while time.time() <= deadline:
        last_response = _get_json(config, f"/tasks/{task_id}", {"language": language})
        status = _extract_status(last_response)
        if _status_is_done(status) or _extract_urls(last_response) or _extract_base64_images(last_response):
            return last_response
        if _status_is_failed(status):
            raise RuntimeError(
                "API Mart 任务失败："
                + json.dumps(last_response, ensure_ascii=False, indent=2)
            )
        time.sleep(interval)

    raise TimeoutError(
        "API Mart 任务轮询超时，最后响应："
        + json.dumps(last_response, ensure_ascii=False, indent=2)
    )


def _parse_url_list(text: str):
    value = _json_object(text, [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError("参考图URL列表JSON 必须是字符串或数组。")


def _normalise_image_url_payload(items, mode: str):
    output = []
    as_objects = str(mode) == "对象URL列表"
    for item in items:
        url = ""
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, dict):
            url = str(item.get("url") or item.get("image_url") or "").strip()
        if not url:
            continue
        output.append({"url": url} if as_objects else url)
    return output


def _image_tensor_to_data_uris(image: torch.Tensor, max_items: int) -> list[str]:
    uris = []
    batch = _ensure_image_batch(image)
    for sample in batch:
        if len(uris) >= max_items:
            break

        file_bytes, content_type = _tensor_to_pil(sample, "PNG", 92)
        # API Mart supports data URI in image_urls. JPEG fallback keeps large inputs under the API limit.
        if len(file_bytes) > 9_500_000:
            file_bytes, content_type = _tensor_to_pil(sample, "JPEG", 92)

        encoded = base64.b64encode(file_bytes).decode("ascii")
        subtype = "jpeg" if content_type == "image/jpeg" else "png"
        uris.append(f"data:image/{subtype};base64,{encoded}")
    return uris


def _chat_image_generate(config: dict, model: str, prompt: str, ref_items: list[str]):
    content = [{"type": "text", "text": prompt}]
    for item in ref_items:
        url = str(item or "").strip()
        if url:
            content.append({"type": "image_url", "image_url": {"url": url}})

    response_text = _post_chat_stream(
        config,
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        },
    )
    urls = _extract_image_urls_from_text(response_text)
    image_tensor, _ = _images_from_response(urls)
    has_image = urls or image_tensor.shape[0] > 1 or tuple(image_tensor.shape[1:3]) != (1, 1)
    if not has_image:
        raise RuntimeError("Chat 快速路径没有返回可下载图片 URL：" + response_text[-1000:])
    return image_tensor, urls, response_text


def _resolve_model_and_resolution(model: str, image_size: str) -> tuple[str, str]:
    model_key = str(model or "nano-banana-2").strip()
    resolution = str(image_size or "2K").strip()

    if model_key == "nano-banana-2-2k":
        return "gemini-3.1-flash-image-preview", "2K"
    if model_key == "nano-banana-2-4k":
        return "gemini-3.1-flash-image-preview", "4K"
    if model_key == "nano-banana-2-official":
        return "gemini-3.1-flash-image-preview-official", resolution
    if model_key == "gpt-image-2-fast":
        return "gpt-image-2", resolution
    if model_key == "gpt-image-2-task":
        return "gpt-image-2", resolution
    if model_key == "gpt-image-2":
        return "gpt-image-2", resolution
    return "gemini-3.1-flash-image-preview", resolution


class Layer13APIMartConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "基础地址": ("STRING", {"default": DEFAULT_BASE_URL}),
                "API环境变量": ("STRING", {"default": "APIMART_API_KEY"}),
                "超时秒": ("INT", {"default": 120, "min": 5, "max": 600, "step": 1}),
                "首次等待秒": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.5}),
                "轮询间隔秒": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 30.0, "step": 0.5}),
                "最大等待秒": ("INT", {"default": 600, "min": 10, "max": 7200, "step": 10}),
                "任务语言": (["zh", "en"], {"default": "zh"}),
                "附加HeaderJSON": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("APIMART_CONFIG", "STRING")
    RETURN_NAMES = ("配置", "信息")
    FUNCTION = "配置"
    CATEGORY = "Layer13/API"

    def 配置(
        self,
        基础地址=DEFAULT_BASE_URL,
        API环境变量="APIMART_API_KEY",
        超时秒=120,
        首次等待秒=1.0,
        轮询间隔秒=1.0,
        最大等待秒=600,
        任务语言="zh",
        附加HeaderJSON="{}",
    ):
        headers = _json_object(附加HeaderJSON, {})
        if not isinstance(headers, dict):
            raise ValueError("附加HeaderJSON 必须是 JSON 对象。")
        config = {
            "base_url": _clean_base_url(基础地址),
            "api_key_env": str(API环境变量 or "APIMART_API_KEY").strip(),
            "api_key": "",
            "timeout": int(超时秒),
            "first_wait": float(首次等待秒),
            "poll_interval": float(轮询间隔秒),
            "max_wait": int(最大等待秒),
            "language": str(任务语言 or "zh"),
            "headers": headers,
        }
        info = (
            f"base={config['base_url']}\n"
            f"key_env={config['api_key_env']}\n"
            f"poll={config['first_wait']}s + every {config['poll_interval']}s, max {config['max_wait']}s"
        )
        return (config, info)


class Layer13APIMartUploadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "配置": ("APIMART_CONFIG",),
                "图像": ("IMAGE",),
                "格式": (["PNG", "JPEG"], {"default": "PNG"}),
                "JPEG质量": ("INT", {"default": 92, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("图片URL列表JSON", "信息")
    FUNCTION = "上传"
    CATEGORY = "Layer13/API"

    def 上传(self, 配置, 图像, 格式="PNG", JPEG质量=92):
        image = _ensure_image_batch(图像)
        urls = []
        lines = []
        ext = "jpg" if str(格式).upper() == "JPEG" else "png"

        for index, sample in enumerate(image):
            file_bytes, content_type = _tensor_to_pil(sample, 格式, JPEG质量)
            response = _post_multipart_file(
                配置,
                "/uploads/images",
                file_bytes,
                content_type,
                f"layer13_{index}.{ext}",
            )
            extracted = _extract_urls(response)
            if not extracted:
                raise RuntimeError(
                    "上传成功但没有从响应中找到图片 URL："
                    + json.dumps(response, ensure_ascii=False, indent=2)
                )
            urls.append(extracted[0])
            lines.append(f"{index}: {extracted[0]}")

        return (json.dumps(urls, ensure_ascii=False), "\n".join(lines))


class Layer13APIMartImageGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "配置": ("APIMART_CONFIG",),
                "模型": ("STRING", {"default": "gpt-image-2"}),
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "尺寸": ("STRING", {"default": "1:1"}),
                "分辨率档位": ("STRING", {"default": "1k"}),
                "数量": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "参考图URL列表JSON": ("STRING", {"default": "[]", "multiline": True}),
                "参考图字段格式": (["字符串URL列表", "对象URL列表"], {"default": "字符串URL列表"}),
                "额外参数JSON": ("STRING", {"default": "{}", "multiline": True}),
                "参考图上传格式": (["PNG", "JPEG"], {"default": "PNG"}),
                "JPEG质量": ("INT", {"default": 92, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "参考图": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("图像", "响应JSON", "结果URL列表JSON")
    FUNCTION = "生成"
    CATEGORY = "Layer13/API"

    def 生成(
        self,
        配置,
        模型="gpt-image-2",
        提示词="",
        尺寸="1:1",
        分辨率档位="1k",
        数量=1,
        等待完成=True,
        参考图URL列表JSON="[]",
        参考图字段格式="字符串URL列表",
        额外参数JSON="{}",
        参考图上传格式="PNG",
        JPEG质量=92,
        参考图=None,
    ):
        extra = _json_object(额外参数JSON, {})
        if not isinstance(extra, dict):
            raise ValueError("额外参数JSON 必须是 JSON 对象。")

        ref_items = _parse_url_list(参考图URL列表JSON)
        if 参考图 is not None:
            uploader = Layer13APIMartUploadImage()
            uploaded_json, _ = uploader.上传(配置, 参考图, 参考图上传格式, JPEG质量)
            ref_items.extend(json.loads(uploaded_json))

        body = {
            "model": str(模型).strip(),
            "prompt": str(提示词 or ""),
            "n": int(数量),
        }
        if str(尺寸 or "").strip():
            body["size"] = str(尺寸).strip()
        if str(分辨率档位 or "").strip():
            body["resolution"] = str(分辨率档位).strip()

        image_payload = _normalise_image_url_payload(ref_items, 参考图字段格式)
        if image_payload:
            body["image_urls"] = image_payload

        body.update(extra)
        response = _post_json(配置, "/images/generations", body)

        final_response = response
        if bool(等待完成):
            image_tensor, urls = _images_from_response(response)
            if urls or image_tensor.shape[0] > 1 or tuple(image_tensor.shape[1:3]) != (1, 1):
                return (
                    image_tensor,
                    json.dumps(response, ensure_ascii=False, indent=2),
                    json.dumps(urls, ensure_ascii=False),
                )

            task_id = _extract_task_id(response)
            if not task_id:
                raise RuntimeError(
                    "提交成功但没有找到 task_id，也没有直接返回图片："
                    + json.dumps(response, ensure_ascii=False, indent=2)
                )
            final_response = _poll_task(配置, task_id)

        image_tensor, urls = _images_from_response(final_response)
        return (
            image_tensor,
            json.dumps(final_response, ensure_ascii=False, indent=2),
            json.dumps(urls, ensure_ascii=False),
        )


class Layer13APIMartTaskStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "配置": ("APIMART_CONFIG",),
                "任务ID": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("图像", "响应JSON", "结果URL列表JSON")
    FUNCTION = "查询"
    CATEGORY = "Layer13/API"

    def 查询(self, 配置, 任务ID=""):
        task_id = str(任务ID or "").strip()
        if not task_id:
            raise ValueError("任务ID不能为空。")
        language = str(_config_value(配置, "language", "zh")).strip() or "zh"
        response = _get_json(配置, f"/tasks/{task_id}", {"language": language})
        image_tensor, urls = _images_from_response(response)
        return (
            image_tensor,
            json.dumps(response, ensure_ascii=False, indent=2),
            json.dumps(urls, ensure_ascii=False),
        )


class Layer13APIMartOne:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "nano-banana-2",
                        "nano-banana-2-2k",
                        "nano-banana-2-4k",
                        "nano-banana-2-official",
                        "gpt-image-2",
                        "gpt-image-2-fast",
                        "gpt-image-2-task",
                    ],
                    {"default": "nano-banana-2"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "aspect_ratio": (
                    [
                        "auto",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                    {"default": "auto"},
                ),
                "resolution": (["1K", "2K", "4K"], {"default": "2K"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE",),
                "默认提示词": ("STRING", {"default": DEFAULT_APIMART_PROMPT, "multiline": True}),
                "apikey": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "task_id", "response")
    FUNCTION = "生成"
    CATEGORY = "Layer13/API"

    def 生成(
        self,
        prompt="",
        model="nano-banana-2",
        seed=0,
        aspect_ratio="auto",
        resolution="2K",
        n=1,
        **kwargs,
    ):
        apikey = str(kwargs.get("apikey") or "").strip()
        config, _ = Layer13APIMartConfig().配置(
            基础地址=DEFAULT_BASE_URL,
            API环境变量="APIMART_API_KEY",
            超时秒=120,
            首次等待秒=1.0,
            轮询间隔秒=1.0,
            最大等待秒=600,
            任务语言="zh",
            附加HeaderJSON="{}",
        )
        if apikey:
            config["api_key"] = apikey

        ref_items = []
        images = kwargs.get("images")
        if images is not None:
            ref_items.extend(_image_tensor_to_data_uris(images, 14))

        prompt_text = str(prompt or "").strip()
        default_prompt = str(kwargs.get("默认提示词") or DEFAULT_APIMART_PROMPT).strip()
        if not prompt_text:
            prompt_text = default_prompt
        if not prompt_text:
            raise ValueError("prompt 和默认提示词不能同时为空。")

        resolved_model, resolved_resolution = _resolve_model_and_resolution(model, resolution)
        fast_chat_error = ""
        ui_model = str(model or "").strip()
        if ui_model in {"gpt-image-2", "gpt-image-2-fast"}:
            try:
                image_tensor, urls, response_text = _chat_image_generate(
                    config,
                    resolved_model,
                    prompt_text,
                    ref_items,
                )
                first_url = urls[0] if urls else ""
                response_info = {
                    "mode": "chat_stream",
                    "model": resolved_model,
                    "ui_model": model,
                    "input_images": len(ref_items),
                    "result_urls": urls,
                    "raw_text": response_text,
                }
                return (
                    image_tensor,
                    first_url,
                    "",
                    json.dumps(response_info, ensure_ascii=False, indent=2),
                )
            except Exception as exc:
                fast_chat_error = str(exc)

        body = {
            "model": resolved_model,
            "prompt": prompt_text,
            "n": int(n),
        }
        if str(aspect_ratio or "").strip():
            body["size"] = str(aspect_ratio).strip()
        if str(resolved_resolution or "").strip():
            body["resolution"] = str(resolved_resolution).strip()
        if int(seed) > 0:
            body["seed"] = int(seed)

        image_payload = _normalise_image_url_payload(ref_items, "字符串URL列表")
        if image_payload:
            body["image_urls"] = image_payload

        response = _post_json(config, "/images/generations", body)
        task_id = _extract_task_id(response)

        final_response = response
        image_tensor, urls = _images_from_response(response)
        has_direct_image = urls or image_tensor.shape[0] > 1 or tuple(image_tensor.shape[1:3]) != (1, 1)
        if has_direct_image:
            first_url = urls[0] if urls else ""
            return (
                image_tensor,
                first_url,
                task_id,
                json.dumps(response, ensure_ascii=False, indent=2),
            )

        if not task_id:
            raise RuntimeError(
                "提交成功但没有找到 task_id，也没有直接返回图片："
                + json.dumps(response, ensure_ascii=False, indent=2)
            )
        final_response = _poll_task(config, task_id)

        image_tensor, urls = _images_from_response(final_response)
        first_url = urls[0] if urls else ""
        response_info = {
            "task_id": task_id,
            "mode": "img2img" if ref_items else "text2img",
            "model": resolved_model,
            "ui_model": model,
            "aspect_ratio": aspect_ratio,
            "image_size": resolved_resolution,
            "input_images": len(ref_items),
            "result_urls": urls,
            "raw": final_response,
        }
        if fast_chat_error:
            response_info["fast_chat_fallback_error"] = fast_chat_error
        return (
            image_tensor,
            first_url,
            task_id,
            json.dumps(response_info, ensure_ascii=False, indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "Layer13APIMartOne": Layer13APIMartOne,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13APIMartOne": "Layer13 APIMart",
}
