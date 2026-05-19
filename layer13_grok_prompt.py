import base64
import io
import json
import os
from typing import Any, Dict
from urllib import error, request

import numpy as np
import torch
from PIL import Image

XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
XAI_MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MODEL = "grok-4-latest"
DEFAULT_SYSTEM_PROMPT = (
    "You are a senior prompt engineer for image generation workflows. "
    "Analyze the provided image and follow the user's instruction. "
    "Return concise, production-usable prompts for ComfyUI or Stable Diffusion style workflows. "
    "Do not mention camera metadata unless it is visually justified by the image. "
    "Prefer clean, high-signal wording over long prose."
)
PROMPT_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "positive_prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "analysis": {"type": "string"},
    },
    "required": ["title", "positive_prompt", "negative_prompt", "analysis"],
    "additionalProperties": False,
}


def _ensure_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError("image 必须是 ComfyUI IMAGE 张量。")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError("image 形状必须是 [B,H,W,C] 或 [H,W,C]。")
    if image.shape[0] < 1:
        raise ValueError("image batch 为空。")
    return image[:1]


def _tensor_to_pil(image: torch.Tensor, resize_long_side: int) -> Image.Image:
    image = _ensure_image_tensor(image)[0].detach().cpu()
    numpy_image = np.clip(image.numpy() * 255.0, 0, 255).astype(np.uint8)

    if numpy_image.shape[-1] == 1:
        numpy_image = numpy_image[:, :, 0]

    pil_image = Image.fromarray(numpy_image)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    resize_long_side = max(0, int(resize_long_side))
    if resize_long_side > 0:
        width, height = pil_image.size
        longest_side = max(width, height)
        if longest_side > resize_long_side:
            scale = resize_long_side / float(longest_side)
            pil_image = pil_image.resize(
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                Image.LANCZOS,
            )

    return pil_image


def _encode_image_to_data_url(pil_image: Image.Image) -> str:
    working = pil_image
    qualities = (92, 84, 76, 68)

    for _ in range(6):
        for quality in qualities:
            buffer = io.BytesIO()
            working.save(buffer, format="JPEG", quality=quality, optimize=True)
            binary = buffer.getvalue()
            if len(binary) <= XAI_MAX_IMAGE_BYTES:
                encoded = base64.b64encode(binary).decode("ascii")
                return f"data:image/jpeg;base64,{encoded}"

        width, height = working.size
        if max(width, height) <= 768:
            break
        working = working.resize(
            (max(1, int(round(width * 0.85))), max(1, int(round(height * 0.85)))),
            Image.LANCZOS,
        )

    raise ValueError("图像编码后仍超过 xAI 20MiB 限制，请先缩小图片。")


def _read_api_key(api_key: str) -> str:
    key = str(api_key or "").strip()
    if key:
        return key
    key = os.getenv("XAI_API_KEY", "").strip()
    if key:
        return key
    raise ValueError("未提供 xAI API Key。请在节点里填写 api_key，或设置环境变量 XAI_API_KEY。")


def _extract_output_text(payload: Dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output", [])
    if not isinstance(output, list):
        raise ValueError("xAI 返回结构异常：缺少 output。")

    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    return text.strip()

    raise ValueError("xAI 没有返回可解析的文本结果。")


def _call_xai(
    image_data_url: str,
    instruction: str,
    system_prompt: str,
    model: str,
    api_key: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "store": False,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                    {
                        "type": "input_text",
                        "text": instruction,
                    },
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "grok_prompt_result",
                "schema": PROMPT_RESULT_SCHEMA,
                "strict": True,
            }
        },
    }

    req = request.Request(
        XAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=int(timeout_seconds)) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"xAI HTTP {exc.code}: {error_text}") from exc
    except error.URLError as exc:
        raise ValueError(f"xAI 连接失败: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"xAI 返回了非 JSON 内容: {body[:400]}") from exc


def _parse_prompt_result(payload: Dict[str, Any]) -> Dict[str, str]:
    raw_text = _extract_output_text(payload)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"xAI 返回文本不是合法 JSON: {raw_text[:400]}") from exc

    title = str(parsed.get("title", "")).strip()
    positive_prompt = str(parsed.get("positive_prompt", "")).strip()
    negative_prompt = str(parsed.get("negative_prompt", "")).strip()
    analysis = str(parsed.get("analysis", "")).strip()

    if not positive_prompt:
        raise ValueError("xAI 返回结果里缺少 positive_prompt。")

    return {
        "title": title,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "analysis": analysis,
    }


class GrokVisionPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Analyze this image and produce a high-quality positive prompt and a clean negative prompt "
                            "for an image generation workflow. Focus on subject, composition, lighting, style, "
                            "materials, and atmosphere."
                        ),
                    },
                ),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": ("STRING", {"multiline": False, "default": DEFAULT_MODEL}),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT}),
                "resize_long_side": ("INT", {"default": 1536, "min": 512, "max": 4096, "step": 64}),
                "timeout_seconds": ("INT", {"default": 120, "min": 30, "max": 600, "step": 5}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "analysis", "title", "raw_json")
    FUNCTION = "build_prompt"
    CATEGORY = "Layer13/xAI"

    def build_prompt(
        self,
        image,
        instruction,
        api_key="",
        model=DEFAULT_MODEL,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        resize_long_side=1536,
        timeout_seconds=120,
    ):
        instruction = str(instruction or "").strip()
        if not instruction:
            raise ValueError("instruction 不能为空。")

        api_key = _read_api_key(api_key)
        model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        system_prompt = str(system_prompt or DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT

        pil_image = _tensor_to_pil(image, int(resize_long_side))
        image_data_url = _encode_image_to_data_url(pil_image)
        payload = _call_xai(
            image_data_url=image_data_url,
            instruction=instruction,
            system_prompt=system_prompt,
            model=model,
            api_key=api_key,
            timeout_seconds=int(timeout_seconds),
        )
        result = _parse_prompt_result(payload)
        raw_json = json.dumps(result, ensure_ascii=False, indent=2)

        return (
            result["positive_prompt"],
            result["negative_prompt"],
            result["analysis"],
            result["title"],
            raw_json,
        )


NODE_CLASS_MAPPINGS = {
    "GrokVisionPromptBuilder": GrokVisionPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokVisionPromptBuilder": "Layer13 Grok Vision Prompt Builder",
}
