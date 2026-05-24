import json
import math
import os
from typing import Dict, Tuple

import folder_paths
import numpy as np
import torch
import torch.nn.functional as F

CROP_INFO_TYPE = "L13_CROP_INFO"
BOX_TYPE = "BOX"


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


def _resize_mask_to(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    mask = _ensure_mask_batch(mask).to(dtype=torch.float32)
    if mask.shape[-2:] == (target_h, target_w):
        return mask.clamp(0.0, 1.0)
    resized = F.interpolate(mask.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.squeeze(1).clamp(0.0, 1.0)


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


def _snap_size_to_multiple(value: float, divisible_by: int, limit: int = None) -> int:
    if limit is not None:
        limit = max(1, int(limit))
    if divisible_by <= 1:
        snapped = max(1, int(round(value)))
        return min(limit, snapped) if limit is not None else snapped

    divisible_by = int(divisible_by)
    if limit is None:
        return max(divisible_by, int(round(value / divisible_by)) * divisible_by)

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
    crop_w_i = _snap_size_to_multiple(crop_w, divisible_by)
    crop_h_i = _snap_size_to_multiple(crop_h, divisible_by)

    x0 = int(round(cx - crop_w_i / 2.0))
    y0 = int(round(cy - crop_h_i / 2.0))
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


def _crop_box_from_info(crop_info: Dict[str, float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    if all(key in crop_info for key in ("x_px", "y_px", "w_px", "h_px")):
        ref_w = max(1.0, float(crop_info.get("ref_w", img_w)))
        ref_h = max(1.0, float(crop_info.get("ref_h", img_h)))
        scale_x = float(img_w) / ref_w
        scale_y = float(img_h) / ref_h
        x0 = int(round(float(crop_info["x_px"]) * scale_x))
        y0 = int(round(float(crop_info["y_px"]) * scale_y))
        crop_w = max(1, int(round(float(crop_info["w_px"]) * scale_x)))
        crop_h = max(1, int(round(float(crop_info["h_px"]) * scale_y)))
    else:
        x0 = int(round(float(crop_info["x"]) * img_w))
        y0 = int(round(float(crop_info["y"]) * img_h))
        crop_w = max(1, int(round(float(crop_info["w"]) * img_w)))
        crop_h = max(1, int(round(float(crop_info["h"]) * img_h)))
    return x0, y0, x0 + crop_w, y0 + crop_h


def _box_intersection(x0: int, y0: int, x1: int, y1: int, img_w: int, img_h: int):
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(img_w, x1)
    src_y1 = min(img_h, y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return None

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    return src_x0, src_y0, src_x1, src_y1, dst_x0, dst_y0, dst_x1, dst_y1


def _normalize_crop_box(crop_box) -> Tuple[int, int, int, int]:
    if crop_box is None:
        raise ValueError("crop_box is empty")
    if isinstance(crop_box, dict):
        values = [crop_box.get(key) for key in ("x1", "y1", "x2", "y2")]
    else:
        values = list(crop_box)
    if len(values) != 4:
        raise ValueError("crop_box must contain x1, y1, x2, y2")
    x0, y0, x1, y1 = [int(round(float(value))) for value in values]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    return x0, y0, x1, y1


def _round_box_size_up(x0: int, y0: int, x1: int, y1: int, multiple: int) -> Tuple[int, int, int, int]:
    if multiple <= 1:
        return x0, y0, x1, y1
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    rounded_w = int(math.ceil(width / multiple) * multiple)
    rounded_h = int(math.ceil(height / multiple) * multiple)
    x0 = x0 - (rounded_w - width) // 2
    y0 = y0 - (rounded_h - height) // 2
    return x0, y0, x0 + rounded_w, y0 + rounded_h


def _mask_bbox(mask: torch.Tensor, threshold: float) -> Tuple[int, int, int, int]:
    mask = _ensure_mask_batch(mask)
    coords = torch.nonzero(mask[0] > float(threshold), as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("mask is empty after thresholding")
    y0 = int(coords[:, 0].min().item())
    y1 = int(coords[:, 0].max().item()) + 1
    x0 = int(coords[:, 1].min().item())
    x1 = int(coords[:, 1].max().item()) + 1
    return x0, y0, x1, y1


def _largest_component_bbox(mask: torch.Tensor, threshold: float) -> Tuple[int, int, int, int]:
    binary = (mask[0].detach().cpu().numpy() > float(threshold))
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best = None
    best_area = 0

    for start_y, start_x in np.argwhere(binary):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        area = 0
        x0 = x1 = int(start_x)
        y0 = y1 = int(start_y)
        while stack:
            y, x = stack.pop()
            area += 1
            x0 = min(x0, x)
            x1 = max(x1, x)
            y0 = min(y0, y)
            y1 = max(y1, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if area > best_area:
            best_area = area
            best = (x0, y0, x1 + 1, y1 + 1)

    if best is None:
        raise ValueError("mask is empty after thresholding")
    return best


def _max_inscribed_rect_bbox(mask: torch.Tensor, threshold: float) -> Tuple[int, int, int, int]:
    binary = (mask[0].detach().cpu().numpy() > float(threshold))
    height, width = binary.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = None

    for y in range(height):
        heights = (heights + 1) * binary[y]
        stack: list[int] = []
        for x in range(width + 1):
            current_h = int(heights[x]) if x < width else 0
            while stack and current_h < heights[stack[-1]]:
                top_index = stack.pop()
                rect_h = int(heights[top_index])
                left = stack[-1] + 1 if stack else 0
                rect_w = x - left
                area = rect_w * rect_h
                if area > best_area:
                    best_area = area
                    best = (left, y - rect_h + 1, x, y + 1)
            stack.append(x)

    if best is None:
        return _mask_bbox(mask, threshold)
    return best


def _detect_mask_box(mask: torch.Tensor, threshold: float, detect: str) -> Tuple[int, int, int, int]:
    detect = str(detect or "mask_area")
    if detect == "min_bounding_rect":
        return _largest_component_bbox(mask, threshold)
    if detect == "max_inscribed_rect":
        return _max_inscribed_rect_bbox(mask, threshold)
    return _mask_bbox(mask, threshold)


def _apply_box_reserve_and_round(
    box: Tuple[int, int, int, int],
    top_reserve: int,
    bottom_reserve: int,
    left_reserve: int,
    right_reserve: int,
    round_to_multiple,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 -= int(left_reserve)
    y0 -= int(top_reserve)
    x1 += int(right_reserve)
    y1 += int(bottom_reserve)
    if str(round_to_multiple) not in {"None", "none", "", "0"}:
        x0, y0, x1, y1 = _round_box_size_up(x0, y0, x1, y1, int(round_to_multiple))
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    return x0, y0, x1, y1


def _draw_rect_on_image(image: torch.Tensor, box: Tuple[int, int, int, int], color, line_width: int = 2) -> torch.Tensor:
    image = _ensure_image_batch(image).clone()
    _, height, width, channels = image.shape
    if channels < 3:
        return image
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, int(x0)))
    x1 = max(0, min(width, int(x1)))
    y0 = max(0, min(height, int(y0)))
    y1 = max(0, min(height, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return image
    line_width = max(1, int(line_width))
    rgb = torch.tensor(color, device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
    image[:, y0:min(y0 + line_width, y1), x0:x1, :3] = rgb
    image[:, max(y1 - line_width, y0):y1, x0:x1, :3] = rgb
    image[:, y0:y1, x0:min(x0 + line_width, x1), :3] = rgb
    image[:, y0:y1, max(x1 - line_width, x0):x1, :3] = rgb
    return image


def _build_box_preview(image: torch.Tensor, detected_box: Tuple[int, int, int, int], crop_box: Tuple[int, int, int, int]) -> torch.Tensor:
    preview = _first_image(image)
    _, height, width, _ = preview.shape
    line_width = max(1, int((width + height) / 400))
    preview = _draw_rect_on_image(preview, detected_box, (1.0, 0.0, 0.0), line_width)
    preview = _draw_rect_on_image(preview, crop_box, (0.0, 1.0, 0.0), line_width)
    return preview


def _resize_image_and_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    scale_mode: str,
    scale_length: int,
    scale_width: int,
    scale_height: int,
):
    image = _ensure_image_batch(image)
    mask = _ensure_mask_batch(mask)
    _, height, width, _ = image.shape
    scale_mode = str(scale_mode or "不缩放")
    if scale_mode == "不缩放":
        return image, mask

    if scale_mode == "宽高":
        target_w = max(1, int(scale_width))
        target_h = max(1, int(scale_height))
    else:
        length = max(1, int(scale_length))
        long_edge = max(width, height)
        short_edge = max(1, min(width, height))
        if scale_mode == "短边":
            scale = float(length) / float(short_edge)
        else:
            scale = float(length) / float(long_edge)
        target_w = max(1, int(round(width * scale)))
        target_h = max(1, int(round(height * scale)))

    if target_w == width and target_h == height:
        return image, mask

    resized_image = _resize_batch_to(image, target_h, target_w)
    resized_mask = _resize_mask_to(mask, target_h, target_w)
    return resized_image, resized_mask


def _crop_image_with_black(image: torch.Tensor, x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    image = _ensure_image_batch(image)
    batch, img_h, img_w, channels = image.shape
    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))
    out = torch.zeros((batch, crop_h, crop_w, channels), device=image.device, dtype=image.dtype)

    intersection = _box_intersection(x0, y0, x1, y1, img_w, img_h)
    if intersection is None:
        return out

    src_x0, src_y0, src_x1, src_y1, dst_x0, dst_y0, dst_x1, dst_y1 = intersection
    out[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = image[:, src_y0:src_y1, src_x0:src_x1, :]
    return out


def _crop_mask_with_black(mask: torch.Tensor, x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    mask = _ensure_mask_batch(mask)
    batch, img_h, img_w = mask.shape
    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))
    out = torch.zeros((batch, crop_h, crop_w), device=mask.device, dtype=mask.dtype)

    intersection = _box_intersection(x0, y0, x1, y1, img_w, img_h)
    if intersection is None:
        return out

    src_x0, src_y0, src_x1, src_y1, dst_x0, dst_y0, dst_x1, dst_y1 = intersection
    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = mask[:, src_y0:src_y1, src_x0:src_x1]
    return out


def _crop_from_info(image: torch.Tensor, crop_info: Dict[str, float]) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    image = _ensure_image_batch(image)
    _, img_h, img_w, _ = image.shape

    x0, y0, x1, y1 = _crop_box_from_info(crop_info, img_w, img_h)
    return _crop_image_with_black(image, x0, y0, x1, y1), (x0, y0, x1, y1)


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


def _match_mask_batch(mask: torch.Tensor, target_batch: int) -> torch.Tensor:
    mask = _ensure_mask_batch(mask)
    current_batch = mask.shape[0]
    if current_batch == target_batch:
        return mask
    if current_batch == 1 and target_batch > 1:
        return mask.repeat(target_batch, 1, 1)
    if current_batch > target_batch:
        return mask[:target_batch]
    raise ValueError(f"mask batch size {current_batch} cannot match target batch {target_batch}")


def _weighted_mean_std(image: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    mask = mask.clamp(0.0, 1.0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)
    sum_w = mask.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
    mean = (image * mask).sum(dim=(1, 2), keepdim=True) / sum_w
    variance = ((image - mean) ** 2 * mask).sum(dim=(1, 2), keepdim=True) / sum_w
    std = torch.sqrt(variance.clamp_min(eps))
    return mean, std


def _match_processed_to_reference(processed: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor, strength: float) -> torch.Tensor:
    strength = float(max(0.0, min(2.0, strength)))
    if strength <= 0.0:
        return processed

    mask = mask.to(device=processed.device, dtype=processed.dtype).clamp(0.0, 1.0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)

    channels = min(int(processed.shape[-1]), int(reference.shape[-1]), 3)
    processed_rgb = processed[..., :channels]
    reference_rgb = reference[..., :channels].to(device=processed.device, dtype=processed.dtype)

    proc_mean, proc_std = _weighted_mean_std(processed_rgb, mask)
    ref_mean, ref_std = _weighted_mean_std(reference_rgb, mask)
    matched = (processed_rgb - proc_mean) * (ref_std / proc_std.clamp_min(1e-6)) + ref_mean
    matched = processed_rgb + (matched - processed_rgb) * strength

    out = processed.clone()
    out[..., :channels] = matched.clamp(0.0, 1.0)
    return out


def _make_feather_mask(
    height: int,
    width: int,
    feather_px: int,
    device,
    dtype,
    left_px=None,
    right_px=None,
    top_px=None,
    bottom_px=None,
) -> torch.Tensor:
    if feather_px <= 0 or height <= 1 or width <= 1:
        return torch.ones((1, height, width, 1), device=device, dtype=dtype)

    feather = max(0, int(feather_px))
    left_px = feather if left_px is None else max(0, int(left_px))
    right_px = feather if right_px is None else max(0, int(right_px))
    top_px = feather if top_px is None else max(0, int(top_px))
    bottom_px = feather if bottom_px is None else max(0, int(bottom_px))

    yy = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, width)

    alpha = torch.ones((height, width), device=device, dtype=dtype)
    if left_px > 0:
        alpha = torch.minimum(alpha, torch.clamp(xx / float(left_px), 0.0, 1.0))
    if right_px > 0:
        alpha = torch.minimum(alpha, torch.clamp(((width - 1) - xx) / float(right_px), 0.0, 1.0))
    if top_px > 0:
        alpha = torch.minimum(alpha, torch.clamp(yy / float(top_px), 0.0, 1.0))
    if bottom_px > 0:
        alpha = torch.minimum(alpha, torch.clamp(((height - 1) - yy) / float(bottom_px), 0.0, 1.0))

    return alpha.unsqueeze(0).unsqueeze(-1)


def _make_edge_mask(height: int, width: int, border_px: int, device, dtype) -> torch.Tensor:
    border_px = max(1, int(border_px))
    mask = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
    edge_h = min(border_px, height)
    edge_w = min(border_px, width)
    mask[:, :edge_h, :, :] = 1.0
    mask[:, height - edge_h:, :, :] = 1.0
    mask[:, :, :edge_w, :] = 1.0
    mask[:, :, width - edge_w:, :] = 1.0
    return mask


def _make_crop_valid_mask(x0: int, y0: int, x1: int, y1: int, img_w: int, img_h: int, batch: int, device, dtype):
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    mask = torch.zeros((batch, crop_h, crop_w, 1), device=device, dtype=dtype)
    intersection = _box_intersection(x0, y0, x1, y1, img_w, img_h)
    if intersection is None:
        return mask
    _, _, _, _, dst_x0, dst_y0, dst_x1, dst_y1 = intersection
    mask[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = 1.0
    return mask


def _translate_image_with_reference_fill(image: torch.Tensor, reference: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    image = _ensure_image_batch(image)
    reference = _ensure_image_batch(reference).to(device=image.device, dtype=image.dtype)
    batch, height, width, _ = image.shape
    out = reference.clone()

    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(width, width - dx)
    src_y1 = min(height, height - dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 + dx
    dst_y0 = src_y0 + dy
    dst_x1 = src_x1 + dx
    dst_y1 = src_y1 + dy
    out[:batch, dst_y0:dst_y1, dst_x0:dst_x1, :] = image[:, src_y0:src_y1, src_x0:src_x1, :]
    return out


def _shift_alignment_mask(mask: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    mask = _ensure_image_batch(mask)
    batch, height, width, channels = mask.shape
    out = torch.zeros((batch, height, width, channels), device=mask.device, dtype=mask.dtype)

    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(width, width - dx)
    src_y1 = min(height, height - dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 + dx
    dst_y0 = src_y0 + dy
    dst_x1 = src_x1 + dx
    dst_y1 = src_y1 + dy
    out[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = mask[:, src_y0:src_y1, src_x0:src_x1, :]
    return out


def _align_processed_to_reference_edges(
    processed: torch.Tensor,
    reference: torch.Tensor,
    align_mask: torch.Tensor,
    search_radius: int,
) -> torch.Tensor:
    search_radius = max(0, int(search_radius))
    if search_radius <= 0:
        return processed

    processed = _ensure_image_batch(processed)
    reference = _ensure_image_batch(reference).to(device=processed.device, dtype=processed.dtype)
    align_mask = align_mask.to(device=processed.device, dtype=processed.dtype).clamp(0.0, 1.0)
    if align_mask.ndim == 3:
        align_mask = align_mask.unsqueeze(-1)
    if align_mask.shape[0] == 1 and processed.shape[0] > 1:
        align_mask = align_mask.repeat(processed.shape[0], 1, 1, 1)

    channels = min(int(processed.shape[-1]), int(reference.shape[-1]), 3)
    best_score = None
    best_shift = (0, 0)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            shifted = _translate_image_with_reference_fill(processed, reference, dx, dy)
            shifted_mask = _shift_alignment_mask(align_mask, dx, dy)
            weight = (align_mask * shifted_mask).clamp(0.0, 1.0)
            weight_sum = float(weight.sum().item())
            if weight_sum <= 1e-6:
                continue
            diff = (shifted[..., :channels] - reference[..., :channels]) * weight
            score = float((diff * diff).sum().item() / max(weight_sum * channels, 1e-6))
            if best_score is None or score < best_score:
                best_score = score
                best_shift = (dx, dy)

    if best_shift == (0, 0):
        return processed
    return _translate_image_with_reference_fill(processed, reference, best_shift[0], best_shift[1])


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
                "aspect_ratio": (
                    ["keep", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "9:21", "21:9"],
                    {"default": "3:4"},
                ),
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

        cropped = _crop_image_with_black(image, x0_i, y0_i, x1_i, y1_i)
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
                "反转遮罩": ("BOOLEAN", {"default": False}),
                "检测": (["mask_area", "min_bounding_rect", "max_inscribed_rect"], {"default": "mask_area"}),
                "mask_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "左侧扩展比例": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 20.0, "step": 0.01}),
                "右侧扩展比例": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 20.0, "step": 0.01}),
                "上方扩展比例": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 20.0, "step": 0.01}),
                "下方扩展比例": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 20.0, "step": 0.01}),
                "裁切比例": (
                    ["keep", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "9:21", "21:9"],
                    {"default": "keep"},
                ),
                "四舍五入到倍数": (["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"], {"default": "1"}),
                "裁剪后缩放": (["不缩放", "长边", "短边", "宽高"], {"default": "不缩放"}),
                "裁剪后长度": ("INT", {"default": 1024, "min": 1, "max": 100000000, "step": 1}),
                "裁剪后宽度": ("INT", {"default": 1024, "min": 1, "max": 100000000, "step": 1}),
                "裁剪后高度": ("INT", {"default": 1024, "min": 1, "max": 100000000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", CROP_INFO_TYPE, "STRING", "INT", "INT", "INT", "INT", "MASK", BOX_TYPE, "IMAGE")
    RETURN_NAMES = (
        "cropped_image",
        "original_image",
        "crop_info",
        "crop_json",
        "crop_x",
        "crop_y",
        "crop_w",
        "crop_h",
        "cropped_mask",
        "crop_box",
        "box_preview",
    )
    FUNCTION = "make_crop"
    CATEGORY = "Layer13/CropInfo"

    def make_crop(
        self,
        image,
        mask,
        反转遮罩=False,
        检测="mask_area",
        mask_threshold=0.05,
        左侧扩展比例=0.0,
        右侧扩展比例=0.0,
        上方扩展比例=0.0,
        下方扩展比例=0.0,
        裁切比例="keep",
        四舍五入到倍数="1",
        裁剪后缩放="不缩放",
        裁剪后长度=1024,
        裁剪后宽度=1024,
        裁剪后高度=1024,
        **kwargs,
    ):
        image = _first_image(image)
        mask = _ensure_mask_batch(mask)
        if mask.shape[0] < 1:
            raise ValueError("mask batch is empty")

        old_mode = kwargs.get("裁剪模式")
        if old_mode == "遮罩边界":
            左侧扩展比例 = 右侧扩展比例 = 上方扩展比例 = 下方扩展比例 = 0.0
            裁切比例 = "keep"
        else:
            左侧扩展比例 = kwargs.get("pad_left", 左侧扩展比例)
            右侧扩展比例 = kwargs.get("pad_right", 右侧扩展比例)
            上方扩展比例 = kwargs.get("pad_top", 上方扩展比例)
            下方扩展比例 = kwargs.get("pad_bottom", 下方扩展比例)
            裁切比例 = kwargs.get("aspect_ratio", 裁切比例)
        四舍五入到倍数 = kwargs.get("divisible_by", 四舍五入到倍数)

        _, img_h, img_w, _ = image.shape
        mask = _resize_mask_to(mask, img_h, img_w)
        if 反转遮罩:
            mask = 1.0 - mask

        detected_x0, detected_y0, detected_x1, detected_y1 = _detect_mask_box(mask[:1], float(mask_threshold), 检测)
        bbox_w = max(1, detected_x1 - detected_x0)
        bbox_h = max(1, detected_y1 - detected_y0)

        crop_x0_f = float(detected_x0) - bbox_w * float(左侧扩展比例)
        crop_y0_f = float(detected_y0) - bbox_h * float(上方扩展比例)
        crop_x1_f = float(detected_x1) + bbox_w * float(右侧扩展比例)
        crop_y1_f = float(detected_y1) + bbox_h * float(下方扩展比例)

        crop_w = max(1.0, crop_x1_f - crop_x0_f)
        crop_h = max(1.0, crop_y1_f - crop_y0_f)
        cx = crop_x0_f + crop_w / 2.0
        cy = crop_y0_f + crop_h / 2.0

        aspect = _parse_aspect_ratio(裁切比例)
        if aspect is not None:
            cx, cy, crop_w, crop_h = _fit_box_to_aspect(cx, cy, crop_w, crop_h, aspect)

        divisible_by = int(四舍五入到倍数)
        crop_x0, crop_y0, crop_x1, crop_y1 = _make_integer_crop_box(cx, cy, crop_w, crop_h, img_w, img_h, divisible_by)

        cropped = _crop_image_with_black(image, crop_x0, crop_y0, crop_x1, crop_y1)
        cropped_mask = _crop_mask_with_black(mask[:1], crop_x0, crop_y0, crop_x1, crop_y1).clamp(0.0, 1.0)
        crop_info = _build_crop_info(crop_x0, crop_y0, crop_x1, crop_y1, img_w, img_h)
        crop_box = [crop_x0, crop_y0, crop_x1, crop_y1]
        box_preview = _build_box_preview(image, (detected_x0, detected_y0, detected_x1, detected_y1), (crop_x0, crop_y0, crop_x1, crop_y1))
        cropped, cropped_mask = _resize_image_and_mask(
            cropped,
            cropped_mask,
            裁剪后缩放,
            裁剪后长度,
            裁剪后宽度,
            裁剪后高度,
        )
        return (
            cropped,
            image,
            crop_info,
            json.dumps(crop_info, ensure_ascii=False),
            crop_x0,
            crop_y0,
            crop_x1 - crop_x0,
            crop_y1 - crop_y0,
            cropped_mask,
            crop_box,
            box_preview,
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
                "颜色匹配": ("BOOLEAN", {"default": False}),
                "颜色匹配强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "边缘像素对齐": ("BOOLEAN", {"default": False}),
                "对齐搜索半径": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "对齐边缘宽度": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
            },
            "optional": {
                "crop_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_images",)
    FUNCTION = "restore"
    CATEGORY = "Layer13/CropInfo"

    def restore(
        self,
        processed_images,
        original_images,
        crop_info,
        feather_px,
        颜色匹配=False,
        颜色匹配强度=1.0,
        边缘像素对齐=False,
        对齐搜索半径=6,
        对齐边缘宽度=16,
        crop_mask=None,
    ):
        processed_images = _ensure_image_batch(processed_images)
        original_images = _ensure_image_batch(original_images)
        target_batch = min(processed_images.shape[0], original_images.shape[0])
        original_images = _trim_batch_to(original_images, target_batch)
        processed_images = _trim_batch_to(processed_images, target_batch)

        out = original_images.clone()
        _, img_h, img_w, _ = original_images.shape

        x0, y0, x1, y1 = _crop_box_from_info(crop_info, img_w, img_h)
        crop_w = max(1, x1 - x0)
        crop_h = max(1, y1 - y0)

        if processed_images.shape[1] != crop_h or processed_images.shape[2] != crop_w:
            processed_images = _resize_batch_to(processed_images, crop_h, crop_w)

        reference_crop = _crop_image_with_black(original_images, x0, y0, x1, y1)
        if 边缘像素对齐:
            valid_mask = _make_crop_valid_mask(
                x0,
                y0,
                x1,
                y1,
                img_w,
                img_h,
                target_batch,
                device=processed_images.device,
                dtype=processed_images.dtype,
            )
            edge_mask = _make_edge_mask(
                crop_h,
                crop_w,
                int(对齐边缘宽度),
                device=processed_images.device,
                dtype=processed_images.dtype,
            )
            align_mask = valid_mask * edge_mask
            if crop_mask is not None:
                full_crop_mask = _match_mask_batch(crop_mask, target_batch)
                full_crop_mask = _resize_mask_to(full_crop_mask, crop_h, crop_w).to(
                    device=processed_images.device,
                    dtype=processed_images.dtype,
                ).unsqueeze(-1)
                unmasked_align_mask = align_mask * (1.0 - full_crop_mask).clamp(0.0, 1.0)
                if float(unmasked_align_mask.sum().item()) > 1e-6:
                    align_mask = unmasked_align_mask
            if float(align_mask.sum().item()) > 1e-6:
                processed_images = _align_processed_to_reference_edges(
                    processed_images,
                    reference_crop,
                    align_mask,
                    int(对齐搜索半径),
                )

        intersection = _box_intersection(x0, y0, x1, y1, img_w, img_h)
        if intersection is None:
            return (out,)

        src_x0, src_y0, src_x1, src_y1, dst_x0, dst_y0, dst_x1, dst_y1 = intersection
        processed_region = processed_images[:, dst_y0:dst_y1, dst_x0:dst_x1, :]
        original_region = out[:, src_y0:src_y1, src_x0:src_x1, :]
        region_h = src_y1 - src_y0
        region_w = src_x1 - src_x0

        if crop_mask is not None:
            paste_mask = _match_mask_batch(crop_mask, target_batch)
            paste_mask = _resize_mask_to(paste_mask, crop_h, crop_w).to(
                device=processed_region.device,
                dtype=processed_region.dtype,
            )
            paste_mask = paste_mask[:, dst_y0:dst_y1, dst_x0:dst_x1].unsqueeze(-1)
        else:
            paste_mask = torch.ones(
                (1, region_h, region_w, 1),
                device=processed_region.device,
                dtype=processed_region.dtype,
            )

        feather_mask = _make_feather_mask(
            crop_h,
            crop_w,
            int(feather_px),
            device=processed_region.device,
            dtype=processed_region.dtype,
            left_px=min(int(feather_px), max(0, x0)),
            right_px=min(int(feather_px), max(0, img_w - x1)),
            top_px=min(int(feather_px), max(0, y0)),
            bottom_px=min(int(feather_px), max(0, img_h - y1)),
        )
        feather_mask = feather_mask[:, dst_y0:dst_y1, dst_x0:dst_x1, :]
        paste_mask = (paste_mask * feather_mask).clamp(0.0, 1.0)

        if 颜色匹配:
            processed_region = _match_processed_to_reference(
                processed_region,
                original_region,
                paste_mask,
                float(颜色匹配强度),
            )

        out[:, src_y0:src_y1, src_x0:src_x1, :] = original_region * (1.0 - paste_mask) + processed_region * paste_mask
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
