import json
import math
import os
from typing import Dict, Tuple

import folder_paths
import torch
import torch.nn.functional as F

CROP_INFO_TYPE = "L13_CROP_INFO"


def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError("image must be a torch.Tensor")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError("image must have shape [B,H,W,C] or [H,W,C]")
    return image


def _ensure_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask must be a torch.Tensor")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError("mask must have shape [B,H,W], [H,W], or [B,H,W,1]")
    return mask


def _first_image(image: torch.Tensor) -> torch.Tensor:
    image = _ensure_image_batch(image)
    if image.shape[0] < 1:
        raise ValueError("image batch is empty")
    return image[:1]


def _parse_aspect_ratio(value: str):
    value = (value or "keep").strip().lower()
    if value in {"keep", "none", "free"}:
        return None
    if ":" not in value:
        raise ValueError(f"invalid aspect ratio: {value}")
    left, right = value.split(":", 1)
    ratio_w = float(left)
    ratio_h = float(right)
    if ratio_w <= 0 or ratio_h <= 0:
        raise ValueError(f"invalid aspect ratio: {value}")
    return ratio_w / ratio_h


def _fit_box_to_aspect(cx: float, cy: float, width: float, height: float, aspect: float):
    current = width / max(height, 1e-6)
    if current < aspect:
        width = height * aspect
    else:
        height = width / aspect
    return cx, cy, width, height


def _clamp_crop(x0: float, y0: float, crop_w: float, crop_h: float, img_w: int, img_h: int):
    crop_w = min(max(crop_w, 1.0), float(img_w))
    crop_h = min(max(crop_h, 1.0), float(img_h))

    x0 = min(max(x0, 0.0), float(img_w) - crop_w)
    y0 = min(max(y0, 0.0), float(img_h) - crop_h)

    x1 = x0 + crop_w
    y1 = y0 + crop_h

    x0_i = max(0, min(img_w - 1, int(round(x0))))
    y0_i = max(0, min(img_h - 1, int(round(y0))))
    x1_i = max(x0_i + 1, min(img_w, int(round(x1))))
    y1_i = max(y0_i + 1, min(img_h, int(round(y1))))
    return x0_i, y0_i, x1_i, y1_i


def _snap_size_to_multiple(value: float, divisible_by: int, limit: int) -> int:
    limit = max(1, int(limit))
    if divisible_by <= 1:
        return max(1, min(limit, int(round(value))))

    divisible_by = int(divisible_by)
    if limit < divisible_by:
        return limit

    lower = max(divisible_by, int(math.floor(value / divisible_by)) * divisible_by)
    upper = min(limit, int(math.ceil(value / divisible_by)) * divisible_by)

    if lower > limit:
        lower = (limit // divisible_by) * divisible_by
    if upper < divisible_by:
        upper = divisible_by

    candidates = [candidate for candidate in (lower, upper) if divisible_by <= candidate <= limit]
    if not candidates:
        fallback = (limit // divisible_by) * divisible_by
        return fallback if fallback >= divisible_by else limit

    return min(candidates, key=lambda candidate: (abs(candidate - value), candidate))


def _make_integer_crop_box(
    cx: float,
    cy: float,
    crop_w: float,
    crop_h: float,
    img_w: int,
    img_h: int,
    divisible_by: int,
):
    crop_w_i = _snap_size_to_multiple(crop_w, divisible_by, img_w)
    crop_h_i = _snap_size_to_multiple(crop_h, divisible_by, img_h)

    x0 = int(round(cx - crop_w_i / 2.0))
    y0 = int(round(cy - crop_h_i / 2.0))
    x0 = max(0, min(img_w - crop_w_i, x0))
    y0 = max(0, min(img_h - crop_h_i, y0))
    x1 = x0 + crop_w_i
    y1 = y0 + crop_h_i
    return x0, y0, x1, y1


def _build_crop_info(x0: int, y0: int, x1: int, y1: int, img_w: int, img_h: int) -> Dict[str, float]:
    crop_w = x1 - x0
    crop_h = y1 - y0
    return {
        "version": 1,
        "mode": "normalized",
        "x": x0 / img_w,
        "y": y0 / img_h,
        "w": crop_w / img_w,
        "h": crop_h / img_h,
        "x_px": x0,
        "y_px": y0,
        "w_px": crop_w,
        "h_px": crop_h,
        "ref_w": img_w,
        "ref_h": img_h,
    }


def _crop_from_info(image: torch.Tensor, crop_info: Dict[str, float]) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    image = _ensure_image_batch(image)
    _, img_h, img_w, _ = image.shape

    x0 = int(round(float(crop_info["x"]) * img_w))
    y0 = int(round(float(crop_info["y"]) * img_h))
    crop_w = max(1, int(round(float(crop_info["w"]) * img_w)))
    crop_h = max(1, int(round(float(crop_info["h"]) * img_h)))
    x1 = min(img_w, x0 + crop_w)
    y1 = min(img_h, y0 + crop_h)
    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    if x1 <= x0:
        x1 = min(img_w, x0 + 1)
    if y1 <= y0:
        y1 = min(img_h, y0 + 1)
    return image[:, y0:y1, x0:x1, :], (x0, y0, x1, y1)


def _resize_batch_to(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    image = _ensure_image_batch(image)
    bchw = image.permute(0, 3, 1, 2)
    resized = F.interpolate(bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def _trim_batch_to(image: torch.Tensor, target_batch: int) -> torch.Tensor:
    image = _ensure_image_batch(image)
    current_batch = image.shape[0]
    if current_batch == target_batch:
        return image
    if current_batch <= 0:
        raise ValueError("image batch is empty")
    return image[:target_batch]


def _make_feather_mask(height: int, width: int, feather_px: int, device, dtype) -> torch.Tensor:
    if feather_px <= 0 or height <= 1 or width <= 1:
        return torch.ones((1, height, width, 1), device=device, dtype=dtype)

    feather = float(feather_px)
    yy = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, width)
    dist_to_edge = torch.minimum(
        torch.minimum(xx, yy),
        torch.minimum((width - 1) - xx, (height - 1) - yy),
    )
    alpha = torch.clamp(dist_to_edge / feather, 0.0, 1.0)
    return alpha.unsqueeze(0).unsqueeze(-1)


def _extract_filename_parts(path_value: str) -> Tuple[str, str, str]:
    value = str(path_value or "").strip()
    value = value.split("?", 1)[0].split("#", 1)[0]
    value = value.replace("\\", "/")
    filename = value.rsplit("/", 1)[-1] if value else ""
    stem, ext = os.path.splitext(filename)
    ext = ext[1:] if ext.startswith(".") else ext
    return filename, stem, ext


def _sanitize_relative_path(value: str) -> str:
    value = str(value or "").strip().replace("\\", "/")
    parts = [part for part in value.split("/") if part not in {"", ".", ".."}]
    return "/".join(parts)


def _crop_info_output_dir(subfolder: str) -> str:
    base_dir = folder_paths.get_output_directory()
    safe_subfolder = _sanitize_relative_path(subfolder) or "layer13_crop_info"
    output_dir = os.path.join(base_dir, safe_subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _resolve_crop_info_file(file_path: str) -> str:
    value = str(file_path or "").strip()
    if not value:
        raise ValueError("file_path 为空")
    if os.path.isabs(value):
        return value
    safe_rel = _sanitize_relative_path(value)
    if not safe_rel:
        raise ValueError("file_path 无效")
    return os.path.join(folder_paths.get_output_directory(), safe_rel)


class Layer13ImageBatchGetFirst:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("first_image", "batch_count")
    FUNCTION = "get_first"
    CATEGORY = "Layer13/CropInfo"

    def get_first(self, images):
        images = _ensure_image_batch(images)
        return (images[:1], int(images.shape[0]))


class Layer13VideoPathGetFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_path": ("STRING", {"multiline": False, "default": ""})}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("filename", "filename_without_ext", "extension")
    FUNCTION = "get_filename"
    CATEGORY = "Layer13/PathTools"

    def get_filename(self, video_path):
        filename, stem, ext = _extract_filename_parts(video_path)
        return (filename, stem, ext)


class Layer13CropInfoFromBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 320, "min": 0, "step": 1}),
                "y": ("INT", {"default": 240, "min": 0, "step": 1}),
                "width": ("INT", {"default": 320, "min": 1, "step": 1}),
                "height": ("INT", {"default": 320, "min": 1, "step": 1}),
                "pad_left": ("FLOAT", {"default": 0.35, "min": 0.0, "step": 0.01}),
                "pad_right": ("FLOAT", {"default": 0.35, "min": 0.0, "step": 0.01}),
                "pad_top": ("FLOAT", {"default": 0.50, "min": 0.0, "step": 0.01}),
                "pad_bottom": ("FLOAT", {"default": 0.35, "min": 0.0, "step": 0.01}),
                "aspect_ratio": (["keep", "1:1", "3:4", "4:3", "9:16", "16:9"], {"default": "3:4"}),
                "divisible_by": (["1", "2", "4", "8", "16", "32", "64"], {"default": "1"}),
            }
        }

    RETURN_TYPES = ("IMAGE", CROP_INFO_TYPE, "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "crop_info", "crop_json", "crop_x", "crop_y", "crop_w", "crop_h")
    FUNCTION = "make_crop"
    CATEGORY = "Layer13/CropInfo"

    def make_crop(self, image, x, y, width, height, pad_left, pad_right, pad_top, pad_bottom, aspect_ratio, divisible_by):
        image = _first_image(image)
        _, img_h, img_w, _ = image.shape

        x = int(x)
        y = int(y)
        width = max(1, int(width))
        height = max(1, int(height))

        cx = x + width / 2.0
        cy = y + height / 2.0
        crop_w = width * (1.0 + float(pad_left) + float(pad_right))
        crop_h = height * (1.0 + float(pad_top) + float(pad_bottom))

        aspect = _parse_aspect_ratio(aspect_ratio)
        if aspect is not None:
            cx, cy, crop_w, crop_h = _fit_box_to_aspect(cx, cy, crop_w, crop_h, aspect)

        divisible_by = int(divisible_by)
        x0_i, y0_i, x1_i, y1_i = _make_integer_crop_box(cx, cy, crop_w, crop_h, img_w, img_h, divisible_by)

        cropped = image[:, y0_i:y1_i, x0_i:x1_i, :]
        crop_info = _build_crop_info(x0_i, y0_i, x1_i, y1_i, img_w, img_h)
        return (
            cropped,
            crop_info,
            json.dumps(crop_info, ensure_ascii=False),
            x0_i,
            y0_i,
            x1_i - x0_i,
            y1_i - y0_i,
        )


class Layer13CropInfoFromMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_left": ("FLOAT", {"default": 0.35, "min": 0.0, "step": 0.01}),
                "pad_right": ("FLOAT", {"default": 0.35, "min": 0.0, "step": 0.01}),
                "pad_top": ("FLOAT", {"default": 0.50, "min": 0.0, "step": 0.01}),
                "pad_bottom": ("FLOAT", {"default": 0.35, "min": 0.0, "step": 0.01}),
                "aspect_ratio": (["keep", "1:1", "3:4", "4:3", "9:16", "16:9"], {"default": "3:4"}),
                "divisible_by": (["1", "2", "4", "8", "16", "32", "64"], {"default": "1"}),
            }
        }

    RETURN_TYPES = ("IMAGE", CROP_INFO_TYPE, "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "crop_info", "crop_json", "crop_x", "crop_y", "crop_w", "crop_h")
    FUNCTION = "make_crop"
    CATEGORY = "Layer13/CropInfo"

    def make_crop(self, image, mask, mask_threshold, pad_left, pad_right, pad_top, pad_bottom, aspect_ratio, divisible_by):
        image = _first_image(image)
        mask = _ensure_mask_batch(mask)
        if mask.shape[0] < 1:
            raise ValueError("mask batch is empty")

        mask0 = mask[0]
        coords = torch.nonzero(mask0 > float(mask_threshold), as_tuple=False)
        if coords.numel() == 0:
            raise ValueError("mask is empty after thresholding")

        y0 = int(coords[:, 0].min().item())
        y1 = int(coords[:, 0].max().item()) + 1
        x0 = int(coords[:, 1].min().item())
        x1 = int(coords[:, 1].max().item()) + 1

        bbox_w = max(1, x1 - x0)
        bbox_h = max(1, y1 - y0)
        _, img_h, img_w, _ = image.shape

        cx = x0 + bbox_w / 2.0
        cy = y0 + bbox_h / 2.0
        crop_w = bbox_w * (1.0 + float(pad_left) + float(pad_right))
        crop_h = bbox_h * (1.0 + float(pad_top) + float(pad_bottom))

        aspect = _parse_aspect_ratio(aspect_ratio)
        if aspect is not None:
            cx, cy, crop_w, crop_h = _fit_box_to_aspect(cx, cy, crop_w, crop_h, aspect)

        divisible_by = int(divisible_by)
        crop_x0, crop_y0, crop_x1, crop_y1 = _make_integer_crop_box(
            cx, cy, crop_w, crop_h, img_w, img_h, divisible_by
        )

        cropped = image[:, crop_y0:crop_y1, crop_x0:crop_x1, :]
        crop_info = _build_crop_info(crop_x0, crop_y0, crop_x1, crop_y1, img_w, img_h)
        return (
            cropped,
            crop_info,
            json.dumps(crop_info, ensure_ascii=False),
            crop_x0,
            crop_y0,
            crop_x1 - crop_x0,
            crop_y1 - crop_y0,
        )


class Layer13CropInfoFromJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_json": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = (CROP_INFO_TYPE, "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("crop_info", "crop_json", "crop_x", "crop_y", "crop_w", "crop_h")
    FUNCTION = "load_crop_info"
    CATEGORY = "Layer13/CropInfo"

    def load_crop_info(self, crop_json):
        text = str(crop_json or "").strip()
        if not text:
            raise ValueError("crop_json 为空")

        crop_info = json.loads(text)
        if not isinstance(crop_info, dict):
            raise ValueError("crop_json 必须是 JSON 对象")

        required_keys = ("x", "y", "w", "h")
        missing = [key for key in required_keys if key not in crop_info]
        if missing:
            raise ValueError(f"crop_json 缺少字段: {', '.join(missing)}")

        crop_x = int(crop_info.get("x_px", 0))
        crop_y = int(crop_info.get("y_px", 0))
        crop_w = int(crop_info.get("w_px", 0))
        crop_h = int(crop_info.get("h_px", 0))

        # 兼容只保存归一化信息的旧格式
        if (crop_w <= 0 or crop_h <= 0) and crop_info.get("ref_w") and crop_info.get("ref_h"):
            ref_w = int(crop_info["ref_w"])
            ref_h = int(crop_info["ref_h"])
            crop_x = int(round(float(crop_info["x"]) * ref_w))
            crop_y = int(round(float(crop_info["y"]) * ref_h))
            crop_w = max(1, int(round(float(crop_info["w"]) * ref_w)))
            crop_h = max(1, int(round(float(crop_info["h"]) * ref_h)))
            crop_info["x_px"] = crop_x
            crop_info["y_px"] = crop_y
            crop_info["w_px"] = crop_w
            crop_info["h_px"] = crop_h

        if crop_w <= 0 or crop_h <= 0:
            raise ValueError("crop_json 中的裁剪宽高无效")

        return (
            crop_info,
            json.dumps(crop_info, ensure_ascii=False),
            crop_x,
            crop_y,
            crop_w,
            crop_h,
        )


class Layer13SaveCropInfoToFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_json": ("STRING", {"multiline": True, "default": ""}),
                "filename": ("STRING", {"multiline": False, "default": "crop_info"}),
                "subfolder": ("STRING", {"multiline": False, "default": "layer13_crop_info"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("file_path", "file_name", "crop_json")
    FUNCTION = "save_crop_json"
    CATEGORY = "Layer13/CropInfo"

    def save_crop_json(self, crop_json, filename, subfolder, overwrite):
        text = str(crop_json or "").strip()
        if not text:
            raise ValueError("crop_json 为空")

        parsed = json.loads(text)
        normalized_json = json.dumps(parsed, ensure_ascii=False, indent=2)

        output_dir = _crop_info_output_dir(subfolder)
        safe_name = _sanitize_relative_path(filename).replace("/", "_") or "crop_info"
        if not safe_name.lower().endswith(".json"):
            safe_name = f"{safe_name}.json"

        file_path = os.path.join(output_dir, safe_name)
        if not overwrite:
            stem, ext = os.path.splitext(safe_name)
            index = 1
            while os.path.exists(file_path):
                file_path = os.path.join(output_dir, f"{stem}_{index:03d}{ext}")
                index += 1

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(normalized_json)

        return (file_path, os.path.basename(file_path), normalized_json)


class Layer13LoadCropInfoFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = (CROP_INFO_TYPE, "STRING", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("crop_info", "crop_json", "file_path", "crop_x", "crop_y", "crop_w", "crop_h")
    FUNCTION = "load_crop_info"
    CATEGORY = "Layer13/CropInfo"

    def load_crop_info(self, file_path):
        resolved_path = _resolve_crop_info_file(file_path)
        if not os.path.exists(resolved_path):
            raise ValueError(f"找不到裁剪信息文件: {resolved_path}")

        with open(resolved_path, "r", encoding="utf-8") as f:
            text = f.read()

        crop_info, crop_json, crop_x, crop_y, crop_w, crop_h = Layer13CropInfoFromJSON().load_crop_info(text)
        return (crop_info, crop_json, resolved_path, crop_x, crop_y, crop_w, crop_h)


class Layer13ApplyCropInfoToBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "crop_info": (CROP_INFO_TYPE,)}}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_images", "crop_x", "crop_y", "crop_w", "crop_h")
    FUNCTION = "apply_crop"
    CATEGORY = "Layer13/CropInfo"

    def apply_crop(self, images, crop_info):
        images = _ensure_image_batch(images)
        cropped, (x0, y0, x1, y1) = _crop_from_info(images, crop_info)
        return (cropped, x0, y0, x1 - x0, y1 - y0)


class Layer13RestoreFromCropInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_images": ("IMAGE",),
                "original_images": ("IMAGE",),
                "crop_info": (CROP_INFO_TYPE,),
                "feather_px": ("INT", {"default": 20, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_images",)
    FUNCTION = "restore"
    CATEGORY = "Layer13/CropInfo"

    def restore(self, processed_images, original_images, crop_info, feather_px):
        processed_images = _ensure_image_batch(processed_images)
        original_images = _ensure_image_batch(original_images)
        target_batch = min(processed_images.shape[0], original_images.shape[0])
        original_images = _trim_batch_to(original_images, target_batch)
        processed_images = _trim_batch_to(processed_images, target_batch)

        out = original_images.clone()
        _, img_h, img_w, _ = original_images.shape

        x0 = int(round(float(crop_info["x"]) * img_w))
        y0 = int(round(float(crop_info["y"]) * img_h))
        crop_w = max(1, int(round(float(crop_info["w"]) * img_w)))
        crop_h = max(1, int(round(float(crop_info["h"]) * img_h)))
        x1 = min(img_w, x0 + crop_w)
        y1 = min(img_h, y0 + crop_h)
        x0 = max(0, min(x0, img_w - 1))
        y0 = max(0, min(y0, img_h - 1))
        crop_w = x1 - x0
        crop_h = y1 - y0

        if processed_images.shape[1] != crop_h or processed_images.shape[2] != crop_w:
            processed_images = _resize_batch_to(processed_images, crop_h, crop_w)

        processed_region = processed_images[:, :crop_h, :crop_w, :]
        original_region = out[:, y0:y1, x0:x1, :]
        feather_mask = _make_feather_mask(
            crop_h,
            crop_w,
            int(feather_px),
            device=processed_region.device,
            dtype=processed_region.dtype,
        )
        out[:, y0:y1, x0:x1, :] = original_region * (1.0 - feather_mask) + processed_region * feather_mask
        return (out,)


NODE_CLASS_MAPPINGS = {
    "Layer13ImageBatchGetFirst": Layer13ImageBatchGetFirst,
    "Layer13VideoPathGetFilename": Layer13VideoPathGetFilename,
    "Layer13CropInfoFromBBox": Layer13CropInfoFromBBox,
    "Layer13CropInfoFromMask": Layer13CropInfoFromMask,
    "Layer13CropInfoFromJSON": Layer13CropInfoFromJSON,
    "Layer13SaveCropInfoToFile": Layer13SaveCropInfoToFile,
    "Layer13LoadCropInfoFromFile": Layer13LoadCropInfoFromFile,
    "Layer13ApplyCropInfoToBatch": Layer13ApplyCropInfoToBatch,
    "Layer13RestoreFromCropInfo": Layer13RestoreFromCropInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13ImageBatchGetFirst": "Layer13批次取首图",
    "Layer13VideoPathGetFilename": "Layer13视频路径提取文件名",
    "Layer13CropInfoFromBBox": "Layer13从框生成裁剪信息",
    "Layer13CropInfoFromMask": "Layer13从遮罩生成裁剪信息",
    "Layer13CropInfoFromJSON": "Layer13从JSON恢复裁剪信息",
    "Layer13SaveCropInfoToFile": "Layer13保存裁剪信息到文件",
    "Layer13LoadCropInfoFromFile": "Layer13从文件读取裁剪信息",
    "Layer13ApplyCropInfoToBatch": "Layer13按裁剪信息批量裁剪",
    "Layer13RestoreFromCropInfo": "Layer13按裁剪信息恢复",
}
