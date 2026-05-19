from __future__ import annotations
import json
import os
from io import BytesIO
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
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


def _normalize_image_batch(
    images: list[torch.Tensor],
    size_mode: str,
    fit_mode: str,
    width: int,
    height: int,
    background_color: str,
) -> torch.Tensor:
    if not images:
        raise ValueError("图像批次为空")

    sizes = [(int(image.shape[2]), int(image.shape[1])) for image in images]

    if size_mode == "保持原尺寸":
        if len(set(sizes)) != 1:
            size_text = ", ".join(f"{w}x{h}" for w, h in sizes)
            raise ValueError(f"保持原尺寸要求所有图片同宽同高，当前尺寸: {size_text}")
        target_w, target_h = sizes[0]
    elif size_mode == "最小尺寸":
        target_w = min(w for w, _ in sizes)
        target_h = min(h for _, h in sizes)
    elif size_mode == "最大尺寸":
        target_w = max(w for w, _ in sizes)
        target_h = max(h for _, h in sizes)
    elif size_mode == "自定义尺寸":
        target_w = max(1, int(width))
        target_h = max(1, int(height))
    else:
        target_w, target_h = sizes[0]

    normalized = [_fit_image_tensor(image, target_w, target_h, fit_mode, background_color) for image in images]
    return torch.cat(normalized, dim=0)


def _parse_rgb(color_text: str, fallback=(0, 0, 0)) -> tuple[float, float, float]:
    s = str(color_text or "").strip()
    if s.startswith("#"):
        s = s[1:]
    try:
        if len(s) == 6:
            return (
                int(s[0:2], 16) / 255.0,
                int(s[2:4], 16) / 255.0,
                int(s[4:6], 16) / 255.0,
            )
        if len(s) == 3:
            return (
                int(s[0] * 2, 16) / 255.0,
                int(s[1] * 2, 16) / 255.0,
                int(s[2] * 2, 16) / 255.0,
            )
    except Exception:
        pass
    return tuple(float(v) / 255.0 for v in fallback)


def _fit_image_tensor(image: torch.Tensor, target_w: int, target_h: int, fit_mode: str, background_color: str) -> torch.Tensor:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("输入图像必须是 IMAGE 张量，形状为 (1,H,W,C)")

    src_h = int(image.shape[1])
    src_w = int(image.shape[2])
    if src_w == target_w and src_h == target_h:
        return image

    x = image.permute(0, 3, 1, 2)
    src_ratio = float(src_w) / float(src_h)
    dst_ratio = float(target_w) / float(target_h)

    if fit_mode == "填充":
        out = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
        return out.permute(0, 2, 3, 1).clamp(0.0, 1.0)

    if fit_mode == "裁剪":
        if src_ratio > dst_ratio:
            crop_w = max(1, int(round(src_h * dst_ratio)))
            left = max(0, (src_w - crop_w) // 2)
            x = x[:, :, :, left:left + crop_w]
        else:
            crop_h = max(1, int(round(src_w / dst_ratio)))
            top = max(0, (src_h - crop_h) // 2)
            x = x[:, :, top:top + crop_h, :]
        out = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
        return out.permute(0, 2, 3, 1).clamp(0.0, 1.0)

    if src_ratio > dst_ratio:
        fit_w = target_w
        fit_h = max(1, int(round(target_w / src_ratio)))
    else:
        fit_h = target_h
        fit_w = max(1, int(round(target_h * src_ratio)))

    resized = F.interpolate(x, size=(fit_h, fit_w), mode="bicubic", align_corners=False, antialias=True)
    bg = torch.tensor(_parse_rgb(background_color), dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    canvas = bg.expand(1, 3, target_h, target_w).clone()
    paste_x = max(0, (target_w - fit_w) // 2)
    paste_y = max(0, (target_h - fit_h) // 2)
    canvas[:, :, paste_y:paste_y + fit_h, paste_x:paste_x + fit_w] = resized
    return canvas.permute(0, 2, 3, 1).clamp(0.0, 1.0)


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
                "批次尺寸": (["第一张尺寸", "保持原尺寸", "最小尺寸", "最大尺寸", "自定义尺寸"], {"default": "第一张尺寸"}),
                "适应方式": (["适应", "裁剪", "填充"], {"default": "适应"}),
                "自定义宽度": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "自定义高度": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "背景颜色": ("STRING", {"default": "#000000"}),
            }
        }
    RETURN_TYPES = (LAYER13_IMAGE_LIST_TYPE, "INT", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("图像列表", "数量", "文件名列表", "图像批次", "图像逐张列表")
    OUTPUT_IS_LIST = (False, False, False, False, True)
    FUNCTION = "加载"
    CATEGORY = "Layer13"
    def 加载(self, 文件列表JSON="[]", 批次尺寸="第一张尺寸", 适应方式="适应", 自定义宽度=1024, 自定义高度=1024, 背景颜色="#000000"):
        items = _parse_file_list(文件列表JSON)
        if not items:
            raise ValueError("请先在节点里选择至少一张图片")
        output_items: list[dict[str, Any]] = []
        image_tensors: list[torch.Tensor] = []
        labels: list[str] = []
        for item in items:
            path = _resolve_input_path(item)
            tensor = _load_image_as_tensor(path)
            image_tensors.append(tensor)
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
        batch = _normalize_image_batch(image_tensors, 批次尺寸, 适应方式, 自定义宽度, 自定义高度, 背景颜色)
        return (output_items, len(output_items), "\n".join(labels), batch, image_tensors)
    @classmethod
    def IS_CHANGED(
        cls,
        文件列表JSON="[]",
        批次尺寸="第一张尺寸",
        适应方式="适应",
        自定义宽度=1024,
        自定义高度=1024,
        背景颜色="#000000",
    ):
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
        return "|".join(fingerprints + [str(批次尺寸), str(适应方式), str(自定义宽度), str(自定义高度), str(背景颜色)])
    @classmethod
    def VALIDATE_INPUTS(
        cls,
        文件列表JSON="[]",
        批次尺寸="第一张尺寸",
        适应方式="适应",
        自定义宽度=1024,
        自定义高度=1024,
        背景颜色="#000000",
    ):
        try:
            _parse_file_list(文件列表JSON)
        except Exception as exc:
            return str(exc)
        if 批次尺寸 == "自定义尺寸" and (int(自定义宽度) < 1 or int(自定义高度) < 1):
            return "自定义宽度和自定义高度必须大于 0"
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
    "Layer13MultiImageInput": "layer13 多图导入",
    "Layer13ImageListGetByIndex": "layer13 按索引取图像列表",
    "Layer13ManualImageLoader": "layer13 拖入加载图片",
    "Layer13ImageListPick": "layer13 拖入取图",
}
