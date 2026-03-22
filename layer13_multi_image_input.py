from __future__ import annotations
import json
import os
from io import BytesIO
from typing import Any
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths
from aiohttp import web

try:
    from server import PromptServer
except Exception:
    PromptServer = None
LAYER13_IMAGE_LIST_TYPE = "L13_IMAGE_LIST"
def _parse_file_list(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw_items = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("文件列表JSON 不是有效的 JSON") from exc
    if not isinstance(raw_items, list):
        raise ValueError("文件列表JSON 必须是数组")
    items: list[dict[str, str]] = []
    for index, raw in enumerate(raw_items):
        if isinstance(raw, str):
            item = {
                "name": raw,
                "subfolder": "",
                "type": "input",
            }
        elif isinstance(raw, dict):
            item = {
                "name": str(raw.get("name", "")).strip(),
                "subfolder": str(raw.get("subfolder", "")).strip(),
                "type": str(raw.get("type", "input")).strip() or "input",
            }
        else:
            raise ValueError(f"第 {index + 1} 项不是有效的图片描述")
        if not item["name"]:
            raise ValueError(f"第 {index + 1} 项缺少文件名")
        item["label"] = item["name"] if not item["subfolder"] else f"{item['subfolder']}/{item['name']}"
        items.append(item)
    return items
def _resolve_input_path(item: dict[str, str]) -> str:
    base_dir = os.path.realpath(folder_paths.get_input_directory())
    candidate = os.path.realpath(os.path.join(base_dir, item["subfolder"], item["name"]))
    if os.path.commonpath([candidate, base_dir]) != base_dir:
        raise ValueError(f"非法路径: {item['label']}")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"图片不存在: {item['label']}")
    return candidate
def _load_image_as_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


if PromptServer is not None:
    @PromptServer.instance.routes.get("/layer13/thumb")
    async def layer13_thumb(request):
        filename = request.rel_url.query.get("filename", "")
        subfolder = request.rel_url.query.get("subfolder", "")
        file_type = request.rel_url.query.get("type", "input")
        size_text = request.rel_url.query.get("size", "192")

        if not filename:
            return web.Response(status=400, text="missing filename")

        try:
            size = max(48, min(512, int(size_text)))
        except Exception:
            size = 192

        try:
            item = {
                "name": filename,
                "subfolder": subfolder,
                "type": file_type,
                "label": filename if not subfolder else f"{subfolder}/{filename}",
            }
            path = _resolve_input_path(item)
        except Exception as exc:
            return web.Response(status=404, text=str(exc))

        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")
                image.thumbnail((size, size), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                image.save(buffer, format="WEBP", quality=82, method=4)
                buffer.seek(0)
                return web.Response(
                    body=buffer.read(),
                    content_type="image/webp",
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "Content-Disposition": f'filename="{os.path.basename(filename)}"',
                    },
                )
        except Exception as exc:
            return web.Response(status=500, text=f"thumb generation failed: {exc}")


class Layer13MultiImageInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件列表JSON": ("STRING", {"default": "[]", "multiline": True}),
            }
        }
    RETURN_TYPES = (LAYER13_IMAGE_LIST_TYPE, "INT", "STRING")
    RETURN_NAMES = ("图像列表", "数量", "文件名列表")
    FUNCTION = "加载"
    CATEGORY = "Layer13"
    def 加载(self, 文件列表JSON="[]"):
        items = _parse_file_list(文件列表JSON)
        if not items:
            raise ValueError("请先在节点里选择至少一张图片")
        output_items: list[dict[str, Any]] = []
        labels: list[str] = []
        for item in items:
            path = _resolve_input_path(item)
            tensor = _load_image_as_tensor(path)
            output_items.append(
                {
                    "image": tensor,
                    "name": item["name"],
                    "subfolder": item["subfolder"],
                    "type": item["type"],
                    "label": item["label"],
                    "path": path,
                }
            )
            labels.append(item["label"])
        return (output_items, len(output_items), "\n".join(labels))
    @classmethod
    def IS_CHANGED(cls, 文件列表JSON="[]"):
        try:
            items = _parse_file_list(文件列表JSON)
        except Exception:
            return 文件列表JSON
        fingerprints = []
        for item in items:
            try:
                path = _resolve_input_path(item)
                fingerprints.append(f"{item['label']}:{os.path.getmtime(path)}:{os.path.getsize(path)}")
            except Exception:
                fingerprints.append(item["label"])
        return "|".join(fingerprints)
    @classmethod
    def VALIDATE_INPUTS(cls, 文件列表JSON="[]"):
        try:
            _parse_file_list(文件列表JSON)
        except Exception as exc:
            return str(exc)
        return True
class Layer13ImageListGetByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像列表": (LAYER13_IMAGE_LIST_TYPE,),
                "编号": ("INT", {"default": 1, "min": 0, "step": 1}),
                "索引模式": (["1基", "0基"], {"default": "1基"}),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("单张图像", "文件名", "当前编号", "总数量")
    FUNCTION = "取图"
    CATEGORY = "Layer13"

    def _resolve_index(self, raw_number: int, total: int, index_mode: str) -> tuple[int, int, int, int]:
        if index_mode == "0基":
            index = raw_number
            valid_min = 0
            valid_max = total - 1
            return index, index + 1, valid_min, valid_max

        index = raw_number - 1
        valid_min = 1
        valid_max = total
        return index, index + 1, valid_min, valid_max

    def 取图(self, 图像列表, 编号=1, 索引模式="1基"):
        if not isinstance(图像列表, list) or not 图像列表:
            raise ValueError("图像列表为空")
        total = len(图像列表)
        raw_number = int(编号)
        index, current_number, valid_min, valid_max = self._resolve_index(raw_number, total, str(索引模式))
        if index < 0 or index >= total:
            raise ValueError(f"编号超出范围: {编号}，有效范围 {valid_min} 到 {valid_max}")
        item = 图像列表[index]
        image = item.get("image")
        if image is None:
            raise ValueError(f"第 {index + 1} 项缺少图像数据")
        return (image, str(item.get("label", item.get("name", ""))), current_number, total)


class Layer13ManualImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件列表JSON": ("STRING", {"default": "[]", "multiline": True}),
                "手动编号": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
            "optional": {
                "循环索引": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("图像", "文件名", "当前编号", "总数量")
    FUNCTION = "加载单张"
    CATEGORY = "Layer13"

    def 加载单张(self, 文件列表JSON="[]", 手动编号=1, 循环索引=None):
        items = _parse_file_list(文件列表JSON)
        if not items:
            raise ValueError("请先在节点里选择至少一张图片")

        total = len(items)
        if 循环索引 is None:
            current_number = int(手动编号)
            if current_number < 1 or current_number > total:
                raise ValueError(f"手动编号超出范围: {current_number}，有效范围 1 到 {total}")
            index = current_number - 1
        else:
            index = int(循环索引) % total
            current_number = index + 1

        item = items[index]
        path = _resolve_input_path(item)
        image = _load_image_as_tensor(path)
        return (image, str(item.get("label", item.get("name", ""))), current_number, total)

    @classmethod
    def IS_CHANGED(cls, 文件列表JSON="[]", 手动编号=1, 循环索引=None):
        try:
            items = _parse_file_list(文件列表JSON)
        except Exception:
            return 文件列表JSON
        if not items:
            return 文件列表JSON
        total = len(items)
        if 循环索引 is None:
            index = max(0, min(total - 1, int(手动编号) - 1))
            marker = f"manual:{int(手动编号)}"
        else:
            index = int(循环索引) % total
            marker = f"loop:{int(循环索引)}"
        item = items[index]
        try:
            path = _resolve_input_path(item)
            st = os.stat(path)
            return f"{marker}|{item['label']}|{st.st_mtime_ns}|{st.st_size}"
        except Exception:
            return f"{marker}|{item.get('label', item.get('name', ''))}"

    @classmethod
    def VALIDATE_INPUTS(cls, 文件列表JSON="[]", 手动编号=1, 循环索引=None):
        try:
            items = _parse_file_list(文件列表JSON)
        except Exception as exc:
            return str(exc)
        if not items:
            return True
        total = len(items)
        if 循环索引 is None:
            number = int(手动编号)
            if number < 1 or number > total:
                return f"手动编号超出范围: {number}，有效范围 1 到 {total}"
        return True


class Layer13ImageListPick:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像列表": (LAYER13_IMAGE_LIST_TYPE,),
                "手动编号": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
            "optional": {
                "循环索引": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("图像", "文件名", "当前编号", "总数量")
    FUNCTION = "取图"
    CATEGORY = "Layer13"

    def 取图(self, 图像列表, 手动编号=1, 循环索引=None):
        if not isinstance(图像列表, list) or not 图像列表:
            raise ValueError("图像列表为空")
        total = len(图像列表)
        if 循环索引 is None:
            current_number = int(手动编号)
            if current_number < 1 or current_number > total:
                raise ValueError(f"手动编号超出范围: {current_number}，有效范围 1 到 {total}")
            index = current_number - 1
        else:
            index = int(循环索引) % total
            current_number = index + 1

        item = 图像列表[index]
        image = item.get("image")
        if image is None:
            raise ValueError(f"第 {index + 1} 项缺少图像数据")
        return (image, str(item.get("label", item.get("name", ""))), current_number, total)


NODE_CLASS_MAPPINGS = {
    "Layer13MultiImageInput": Layer13MultiImageInput,
    "Layer13ImageListGetByIndex": Layer13ImageListGetByIndex,
    "Layer13ManualImageLoader": Layer13ManualImageLoader,
    "Layer13ImageListPick": Layer13ImageListPick,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13MultiImageInput": "Layer13多图导入",
    "Layer13ImageListGetByIndex": "Layer13按索引取图像列表",
    "Layer13ManualImageLoader": "Layer13拖入加载图片",
    "Layer13ImageListPick": "Layer13拖入取图",
}
