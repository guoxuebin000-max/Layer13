import logging
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


MAX_RESOLUTION = 16384
_MISSING = object()

ADD_NOISE_CHOICES = ["启用", "禁用"]
LEFTOVER_NOISE_CHOICES = ["禁用", "启用"]
BLEND_CHOICES = ["余弦", "线性", "高斯"]
TARGET_SIZE_CHOICES = ["自定义", "4K", "8K"]
REFERENCE_MODE_CHOICES = ["潜空间缩放", "图像重编码"]
TILE_ORDER_CHOICES = ["顺序", "蛇形", "中心向外"]
PREVIEW_MODE_CHOICES = ["每个分块", "每轮", "关闭"]
PROGRESSIVE_MODE_CHOICES = ["关闭", "平衡1024阶梯", "稳定1.5倍", "快速2倍"]
REDRAW_PRESET_CHOICES = ["自定义", "人物稳定", "人物细节", "背景增强", "建筑线条", "极限8K保守"]
ENABLE_CHOICES = ["启用", "禁用"]
COLOR_MATCH_METHOD_CHOICES = ["RGB均值方差", "YCbCr色度"]
TARGET_SIZE_LONG_EDGE = {
    "自定义": None,
    "4K": 4096,
    "8K": 8192,
}

ENABLE_MAP = {
    "启用": True,
    "禁用": False,
    True: True,
    False: False,
}

ADD_NOISE_MAP = {
    "启用": "enable",
    "禁用": "disable",
    "enable": "enable",
    "disable": "disable",
}
LEFTOVER_NOISE_MAP = {
    "启用": "enable",
    "禁用": "disable",
    "enable": "enable",
    "disable": "disable",
}
BLEND_MAP = {
    "余弦": "cosine",
    "线性": "linear",
    "高斯": "gaussian",
    "cosine": "cosine",
    "linear": "linear",
    "gaussian": "gaussian",
}
REFERENCE_MODE_MAP = {
    "潜空间缩放": "latent",
    "图像重编码": "image",
    "latent": "latent",
    "image": "image",
}


def _latent_dim(pixel_dim: int) -> int:
    return max(1, int(pixel_dim) // 8)


def _round_pixel_dim(pixel_dim: int) -> int:
    return max(8, int(pixel_dim) - int(pixel_dim) % 8)


def _round_pixel_dim_nearest(pixel_dim: float) -> int:
    return max(8, int(round(float(pixel_dim) / 8.0)) * 8)


def _param(kwargs: Dict, *names: str, default=_MISSING):
    for name in names:
        if name in kwargs:
            return kwargs[name]
    if default is not _MISSING:
        return default
    raise KeyError(f"Missing node input. Expected one of: {', '.join(names)}")


def _target_pixels_with_long_edge(samples: torch.Tensor, long_edge: int) -> Tuple[int, int]:
    latent_h = max(1, int(samples.shape[-2]))
    latent_w = max(1, int(samples.shape[-1]))
    if latent_w >= latent_h:
        width = long_edge
        height = _round_pixel_dim_nearest(long_edge * latent_h / latent_w)
    else:
        width = _round_pixel_dim_nearest(long_edge * latent_w / latent_h)
        height = long_edge

    return min(width, MAX_RESOLUTION), min(height, MAX_RESOLUTION)


def _target_pixels_from_latent_ratio(samples: torch.Tensor, target_size: str, custom_width: int, custom_height: int) -> Tuple[int, int]:
    long_edge = TARGET_SIZE_LONG_EDGE.get(target_size)
    if long_edge is not None:
        return _target_pixels_with_long_edge(samples, long_edge)

    width = _round_pixel_dim(custom_width)
    height = _round_pixel_dim(custom_height)
    if width == height:
        return _target_pixels_with_long_edge(samples, width)
    return width, height


def _round_to_multiple_nearest(value: float, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _round_to_multiple_floor(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return max(multiple, int(value) - int(value) % multiple)


def _vae_scale(vae) -> int:
    if hasattr(vae, "spacial_compression_encode"):
        scale = vae.spacial_compression_encode()
    else:
        scale = getattr(vae, "downscale_ratio", 8)
        if isinstance(scale, tuple):
            scale = scale[-1]
    if not isinstance(scale, int):
        scale = 8
    return max(1, scale)


def _target_pixels_from_image_ratio(pixels: torch.Tensor, target_size: str, custom_width: int, custom_height: int, scale: int) -> Tuple[int, int]:
    image_h = max(1, int(pixels.shape[1]))
    image_w = max(1, int(pixels.shape[2]))
    long_edge = TARGET_SIZE_LONG_EDGE.get(target_size)
    if long_edge is not None:
        if image_w >= image_h:
            width = _round_to_multiple_nearest(long_edge, scale)
            height = _round_to_multiple_nearest(long_edge * image_h / image_w, scale)
        else:
            width = _round_to_multiple_nearest(long_edge * image_w / image_h, scale)
            height = _round_to_multiple_nearest(long_edge, scale)
        return min(width, MAX_RESOLUTION), min(height, MAX_RESOLUTION)

    width = _round_to_multiple_floor(custom_width, scale)
    height = _round_to_multiple_floor(custom_height, scale)
    if width == height:
        if image_w >= image_h:
            return width, _round_to_multiple_nearest(width * image_h / image_w, scale)
        return _round_to_multiple_nearest(height * image_w / image_h, scale), height
    return width, height


def _scale_pixels(pixels: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    pixels = pixels[:, :, :, :3]
    scaled = comfy.utils.common_upscale(pixels.movedim(-1, 1), width, height, method, "disabled")
    return scaled.movedim(1, -1)


def _downsample_for_color_stats(pixels: torch.Tensor, max_edge: int = 512) -> torch.Tensor:
    height = max(1, int(pixels.shape[1]))
    width = max(1, int(pixels.shape[2]))
    long_edge = max(height, width)
    if long_edge <= max_edge:
        return pixels
    scale = float(max_edge) / float(long_edge)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    return comfy.utils.common_upscale(pixels.movedim(-1, 1), out_w, out_h, "area", "disabled").movedim(1, -1)


def _rgb_to_ycbcr(pixels: torch.Tensor) -> torch.Tensor:
    r = pixels[..., 0]
    g = pixels[..., 1]
    b = pixels[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack((y, cb, cr), dim=-1)


def _ycbcr_to_rgb(pixels: torch.Tensor) -> torch.Tensor:
    y = pixels[..., 0]
    cb = pixels[..., 1] - 0.5
    cr = pixels[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack((r, g, b), dim=-1)


def _color_stats(pixels: torch.Tensor, method: str) -> Tuple[torch.Tensor, torch.Tensor]:
    small = _downsample_for_color_stats(pixels[:, :, :, :3].clamp(0.0, 1.0))
    if method == "YCbCr色度":
        small = _rgb_to_ycbcr(small)
    flat = small.reshape(small.shape[0], -1, 3)
    mean = flat.mean(dim=1).view(flat.shape[0], 1, 1, 3)
    std = flat.std(dim=1, unbiased=False).clamp_min(1e-4).view(flat.shape[0], 1, 1, 3)
    return mean, std


def _align_color_stats(stats: Tuple[torch.Tensor, torch.Tensor], batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = stats
    if mean.shape[0] == batch:
        return mean, std
    mean = mean[:1].expand(batch, -1, -1, -1)
    std = std[:1].expand(batch, -1, -1, -1)
    return mean, std


def _match_image_color(
    image: torch.Tensor,
    reference: torch.Tensor,
    strength: float,
    method: str,
    chunk_rows: int = 512,
) -> torch.Tensor:
    strength = _clamp_float(strength, 0.0, 1.0)
    if strength <= 0:
        return image

    pixels = image[:, :, :, :3].clamp(0.0, 1.0)
    reference = reference[:, :, :, :3].to(device=pixels.device, dtype=pixels.dtype).clamp(0.0, 1.0)
    method = method if method in COLOR_MATCH_METHOD_CHOICES else "RGB均值方差"
    source_mean, source_std = _color_stats(pixels, method)
    target_mean, target_std = _align_color_stats(_color_stats(reference, method), pixels.shape[0])
    ratio = (target_std / source_std).clamp(0.5, 2.0)

    out = torch.empty_like(pixels)
    rows = max(1, int(chunk_rows))
    for y0 in range(0, int(pixels.shape[1]), rows):
        y1 = min(int(pixels.shape[1]), y0 + rows)
        chunk = pixels[:, y0:y1, :, :]
        if method == "YCbCr色度":
            source = _rgb_to_ycbcr(chunk)
            matched = (source - source_mean) * ratio + target_mean
            matched[..., 0] = source[..., 0] + (matched[..., 0] - source[..., 0]) * 0.35
            matched = _ycbcr_to_rgb(matched)
        else:
            matched = (chunk - source_mean) * ratio + target_mean
        out[:, y0:y1, :, :] = torch.lerp(chunk, matched, strength).clamp(0.0, 1.0)

    if image.shape[-1] > 3:
        return torch.cat((out, image[:, :, :, 3:]), dim=-1)
    return out


def _vae_encode_pixels(vae, pixels: torch.Tensor) -> torch.Tensor:
    pixels = pixels[:, :, :, :3]
    if max(int(pixels.shape[1]), int(pixels.shape[2])) >= 2048 and hasattr(vae, "encode_tiled"):
        return vae.encode_tiled(pixels, tile_x=1024, tile_y=1024, overlap=128)
    return vae.encode(pixels)


def _vae_decode_latent(vae, samples: torch.Tensor) -> torch.Tensor:
    pixel_h = int(samples.shape[-2]) * _vae_scale(vae)
    pixel_w = int(samples.shape[-1]) * _vae_scale(vae)
    if max(pixel_h, pixel_w) >= 2048 and hasattr(vae, "decode_tiled"):
        try:
            return vae.decode_tiled(samples, tile_x=1024, tile_y=1024, overlap=128)
        except TypeError:
            return vae.decode_tiled(samples)
    return vae.decode(samples)


def _pixels_from_long_edge(width: int, height: int, long_edge: int, scale: int) -> Tuple[int, int]:
    width = max(1, int(width))
    height = max(1, int(height))
    long_edge = min(MAX_RESOLUTION, max(scale, int(long_edge)))
    if width >= height:
        out_w = _round_to_multiple_nearest(long_edge, scale)
        out_h = _round_to_multiple_nearest(long_edge * height / width, scale)
    else:
        out_w = _round_to_multiple_nearest(long_edge * width / height, scale)
        out_h = _round_to_multiple_nearest(long_edge, scale)
    return min(out_w, MAX_RESOLUTION), min(out_h, MAX_RESOLUTION)


def _progressive_stage_pixels(
    reference_image: torch.Tensor,
    target_size: str,
    target_width: int,
    target_height: int,
    scale: int,
    mode: str,
) -> List[Tuple[int, int]]:
    final_w, final_h = _target_pixels_from_image_ratio(reference_image, target_size, target_width, target_height, scale)
    if mode == "关闭":
        return [(final_w, final_h)]

    src_h = max(1, int(reference_image.shape[1]))
    src_w = max(1, int(reference_image.shape[2]))
    src_long = max(src_w, src_h)
    final_long = max(final_w, final_h)
    if final_long <= src_long:
        return [(final_w, final_h)]

    stages: List[Tuple[int, int]] = []
    current = float(src_long)
    while current < final_long:
        if mode == "快速2倍":
            next_long = current * 2.0
        elif mode == "稳定1.5倍":
            next_long = current * 1.5
        else:
            next_long = current + 1024.0

        next_long = min(float(final_long), max(next_long, current + scale))
        stage_w, stage_h = _pixels_from_long_edge(final_w, final_h, int(round(next_long)), scale)
        if stages and stages[-1] == (stage_w, stage_h):
            break
        stages.append((stage_w, stage_h))
        current = max(float(stage_w), float(stage_h))

    if not stages or stages[-1] != (final_w, final_h):
        stages.append((final_w, final_h))
    return stages


def _effective_sampler_steps(steps: int, start_step: Optional[int], last_step: Optional[int]) -> int:
    steps = max(1, int(steps))
    start = 0 if start_step is None else max(0, min(int(start_step), steps))
    end = steps if last_step is None else max(0, min(int(last_step), steps))
    return max(1, end - start)


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _enabled(value) -> bool:
    return bool(ENABLE_MAP.get(value, value))


def _seed_for_stage(seed: int, stage_index: int, offset: int = 0) -> int:
    return (int(seed) + int(stage_index) * 100003 + int(offset)) & 0xffffffffffffffff


def _apply_redraw_policy(
    preset: str,
    safety_enabled: bool,
    has_subject_mask: bool,
    cfg: float,
    denoise: float,
    detail: float,
    sample_halo: int,
    subject_cap: float,
    background_multiplier: float,
    subject_threshold: float,
    reference_hold: float,
    seam_enabled: bool,
    seam_denoise: float,
    seam_width: int,
) -> Tuple[float, float, float, int, float, float, float, float, bool, float, int]:
    preset = preset if preset in REDRAW_PRESET_CHOICES else "自定义"
    cfg = float(cfg)
    denoise = _clamp_float(denoise, 0.01, 1.0)
    detail = _clamp_float(detail, 0.0, 0.25)
    sample_halo = max(0, int(sample_halo))
    subject_cap = _clamp_float(subject_cap, 0.01, 1.0)
    background_multiplier = _clamp_float(background_multiplier, 0.1, 3.0)
    subject_threshold = _clamp_float(subject_threshold, 0.0, 1.0)
    reference_hold = _clamp_float(reference_hold, 0.0, 0.8)
    seam_denoise = _clamp_float(seam_denoise, 0.01, 0.5)
    seam_width = max(0, int(seam_width))

    if preset == "人物稳定":
        cfg = min(cfg, 4.8)
        denoise = min(denoise, 0.18)
        detail = min(detail, 0.01)
        sample_halo = max(sample_halo, 96)
        subject_cap = min(subject_cap, 0.14)
        background_multiplier = min(max(background_multiplier, 1.0), 1.25)
        subject_threshold = max(subject_threshold, 0.12)
        reference_hold = _clamp_float(reference_hold, 0.04, 0.08)
    elif preset == "人物细节":
        cfg = min(cfg, 5.0)
        denoise = min(denoise, 0.22)
        detail = min(detail, 0.015)
        sample_halo = max(sample_halo, 96)
        subject_cap = min(subject_cap, 0.16)
        background_multiplier = min(max(background_multiplier, 1.1), 1.35)
        subject_threshold = max(subject_threshold, 0.12)
        reference_hold = _clamp_float(reference_hold, 0.03, 0.08)
    elif preset == "背景增强":
        sample_halo = max(sample_halo, 128)
        background_multiplier = max(background_multiplier, 1.35)
        seam_width = max(seam_width, 96)
    elif preset == "建筑线条":
        cfg = min(cfg, 6.0)
        sample_halo = max(sample_halo, 128)
        background_multiplier = max(background_multiplier, 1.2)
        detail = min(detail, 0.02)
        seam_width = max(seam_width, 128)
    elif preset == "极限8K保守":
        cfg = min(cfg, 4.2)
        denoise = min(denoise, 0.14)
        detail = 0.0
        sample_halo = max(sample_halo, 128)
        subject_cap = min(subject_cap, 0.12)
        background_multiplier = min(background_multiplier, 1.1)
        subject_threshold = max(subject_threshold, 0.10)
        reference_hold = _clamp_float(reference_hold, 0.05, 0.10)
        seam_denoise = min(seam_denoise, 0.06)

    if safety_enabled and (preset in ("人物稳定", "人物细节", "极限8K保守") or has_subject_mask):
        cfg = min(cfg, 5.2)
        detail = min(detail, 0.02)
        subject_cap = min(subject_cap, 0.16)
        background_multiplier = min(background_multiplier, 1.4)

    return (
        cfg,
        denoise,
        detail,
        sample_halo,
        subject_cap,
        background_multiplier,
        subject_threshold,
        reference_hold,
        seam_enabled,
        seam_denoise,
        seam_width,
    )


def _ordered_tiles(tiles: List[Tuple[int, int, int, int]], height: int, width: int, mode: str) -> List[Tuple[int, int, int, int]]:
    if mode == "中心向外":
        cy = height / 2.0
        cx = width / 2.0
        return sorted(tiles, key=lambda r: (((r[0] + r[1]) / 2.0 - cy) ** 2 + ((r[2] + r[3]) / 2.0 - cx) ** 2))
    if mode == "蛇形":
        rows: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for tile in tiles:
            rows.setdefault(tile[0], []).append(tile)
        ordered = []
        for i, y in enumerate(sorted(rows)):
            row = sorted(rows[y], key=lambda r: r[2], reverse=bool(i % 2))
            ordered.extend(row)
        return ordered
    return tiles


def _scale_mask(mask: torch.Tensor, width: int, height: int, device, dtype) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(1).to(device=device, dtype=dtype)
    return comfy.utils.common_upscale(mask, width, height, "bilinear", "disabled").clamp(0.0, 1.0)


def _tile_positions(length: int, tile: int, overlap: int) -> List[int]:
    tile = max(1, min(tile, length))
    if tile >= length:
        return [0]
    stride = max(1, tile - overlap)
    positions = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def _tile_grid(height: int, width: int, tile_h: int, tile_w: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    ys = _tile_positions(height, tile_h, overlap)
    xs = _tile_positions(width, tile_w, overlap)
    return [(y, min(y + tile_h, height), x, min(x + tile_w, width)) for y in ys for x in xs]


def _seam_mask_from_tiles(
    tiles: Sequence[Tuple[int, int, int, int]],
    height: int,
    width: int,
    seam: int,
    device,
    dtype,
) -> Optional[torch.Tensor]:
    seam = max(0, int(seam))
    if seam <= 0:
        return None
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    half = max(1, seam // 2)
    for y0, y1, x0, x1 in tiles:
        if y0 > 0:
            a = max(0, y0 - half)
            b = min(height, y0 + half)
            mask[:, :, a:b, x0:x1] = 1.0
        if x0 > 0:
            a = max(0, x0 - half)
            b = min(width, x0 + half)
            mask[:, :, y0:y1, a:b] = 1.0
    if mask.max().item() <= 0:
        return None
    if seam > 2:
        mask = torch.nn.functional.avg_pool2d(mask, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)
    return mask


def _edge_ramp(n: int, mode: str, device, dtype) -> torch.Tensor:
    if n <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    t = torch.linspace(0.0, 1.0, n + 2, device=device, dtype=dtype)[1:-1]
    if mode == "linear":
        return t
    if mode == "gaussian":
        sigma = 0.38
        return torch.exp(-0.5 * ((1.0 - t) / sigma) ** 2).clamp_min(1e-3)
    return 0.5 - 0.5 * torch.cos(t * math.pi)


def _tile_weight(
    tile_h: int,
    tile_w: int,
    full_h: int,
    full_w: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    overlap: int,
    mode: str,
    device,
    dtype,
) -> torch.Tensor:
    wy = torch.ones(tile_h, device=device, dtype=dtype)
    wx = torch.ones(tile_w, device=device, dtype=dtype)
    oy = min(max(0, overlap), max(0, tile_h // 2))
    ox = min(max(0, overlap), max(0, tile_w // 2))
    if oy > 0:
        ramp = _edge_ramp(oy, mode, device, dtype)
        if y0 > 0:
            wy[:oy] = ramp
        if y1 < full_h:
            wy[-oy:] = torch.flip(ramp, dims=[0])
    if ox > 0:
        ramp = _edge_ramp(ox, mode, device, dtype)
        if x0 > 0:
            wx[:ox] = ramp
        if x1 < full_w:
            wx[-ox:] = torch.flip(ramp, dims=[0])
    return (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)


def _normalize_detail_noise(noise: torch.Tensor) -> torch.Tensor:
    if noise.ndim != 4:
        return noise
    if noise.shape[-2] > 2 and noise.shape[-1] > 2:
        low = torch.nn.functional.avg_pool2d(noise, kernel_size=3, stride=1, padding=1)
        noise = noise - low
    reduce_dims = tuple(range(1, noise.ndim))
    std = noise.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    return noise / std


def _match_latent_moments(samples: torch.Tensor, reference: torch.Tensor, strength: float) -> torch.Tensor:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0 or samples.ndim != 4 or reference.ndim != 4:
        return samples
    dims = (-2, -1)
    work = samples.float()
    ref = reference.to(device=samples.device).float()
    src_mean = work.mean(dim=dims, keepdim=True)
    ref_mean = ref.mean(dim=dims, keepdim=True)
    src_std = work.std(dim=dims, keepdim=True).clamp_min(1e-5)
    ref_std = ref.std(dim=dims, keepdim=True).clamp_min(1e-5)
    matched = (work - src_mean) / src_std * ref_std + ref_mean
    return torch.lerp(work, matched, strength).to(dtype=samples.dtype)


def _blend_reference_latent(samples: torch.Tensor, reference: torch.Tensor, strength: float) -> torch.Tensor:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return samples
    ref = reference.to(device=samples.device, dtype=samples.dtype)
    return torch.lerp(samples, ref, strength)


class _CanvasProgress:
    def __init__(self, model, total_steps: int, preview_mode: str, vae=None):
        self.total = max(1, int(total_steps))
        self.completed = 0
        self.tile_start = 0
        self.tile_steps = 1
        self.preview_mode = preview_mode
        self.pbar = comfy.utils.ProgressBar(self.total)
        self.previewer = None
        self.last_preview = None
        if preview_mode != "关闭":
            try:
                self.previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
            except Exception as exc:
                logging.warning("L13 KSampler-style previewer could not be initialized: %s", exc)

    def start_tile(self, expected_steps: int):
        self.tile_start = self.completed
        self.tile_steps = max(1, int(expected_steps))

    def tile_callback(self):
        def callback(step, x0, x, total_steps):
            current = min(self.total, self.tile_start + max(1, int(step) + 1))
            preview = None
            if self.preview_mode == "每个分块" and self.previewer is not None and x0 is not None:
                try:
                    preview = self.previewer.decode_latent_to_preview_image("JPEG", x0)
                    self.last_preview = preview
                except Exception as exc:
                    logging.warning("L13 KSampler preview failed and will be disabled: %s", exc)
                    self.previewer = None
                    self.last_preview = None
            self.pbar.update_absolute(current, self.total, preview)

        return callback

    def capture_preview(self, latent: torch.Tensor):
        if self.preview_mode == "关闭":
            return
        try:
            if self.previewer is not None:
                self.last_preview = self.previewer.decode_latent_to_preview_image("JPEG", latent)
        except Exception as exc:
            logging.warning("L13 latent preview failed and will be disabled: %s", exc)
            self.previewer = None
            self.last_preview = None

    def finish_tile(self, actual_steps: Optional[int] = None, force_preview: bool = False):
        steps = self.tile_steps if actual_steps is None else max(1, int(actual_steps))
        self.completed = min(self.total, self.tile_start + steps)
        preview = self.last_preview if force_preview and self.preview_mode != "关闭" else None
        self.pbar.update_absolute(self.completed, self.total, preview)

    def force_preview(self):
        if self.preview_mode != "关闭" and self.last_preview is not None:
            self.pbar.update_absolute(self.completed, self.total, self.last_preview)


class L13RedrawSettings:
    blend_modes = BLEND_CHOICES
    tile_orders = TILE_ORDER_CHOICES
    image_upscale_methods = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "目标宽度": ("INT", {"default": 8192, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "自定义目标宽度。若宽高相等，例如 8192/8192，会把该值当成长边并保持参考图比例。"}),
                "目标高度": ("INT", {"default": 8192, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "自定义目标高度。若宽高相等，例如 8192/8192，会把该值当成长边并保持参考图比例。"}),
                "递进强度衰减": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "递进模式下每进入下一段时重绘强度的乘数。"}),
                "重绘强度": ("FLOAT", {"default": 0.22, "min": 0.01, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "masked img2img denoise。人物 8K 建议 0.12-0.22，背景可到 0.30-0.35。"}),
                "细节扰动": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.25, "step": 0.005, "round": 0.001, "tooltip": "在中心写回区域给 latent 加入极小高频扰动。人物建议 0-0.01。"}),
                "分块宽度": ("INT", {"default": 1280, "min": 128, "max": MAX_RESOLUTION, "step": 8, "tooltip": "中心写回区域的像素宽度。人物 8K 建议 1024-1536。"}),
                "分块高度": ("INT", {"default": 1280, "min": 128, "max": MAX_RESOLUTION, "step": 8, "tooltip": "中心写回区域的像素高度。人物 8K 建议 1024-1536。"}),
                "重叠像素": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 8, "tooltip": "中心 tile 之间的重叠像素。重叠越大越不容易有接缝，但更慢。"}),
                "上下文像素": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 8, "tooltip": "每个 tile 向外额外读取的上下文区域。context 只参与推理，不写回全图。"}),
                "采样缓冲像素": ("INT", {"default": 96, "min": 0, "max": 2048, "step": 8, "tooltip": "中心写回区外额外允许采样的一圈 halo；参与 denoise 但不写回。"}),
                "融合方式": (cls.blend_modes, {"tooltip": "写回中心 tile 时的 feather 权重。"}),
                "图像缩放算法": (cls.image_upscale_methods, {"tooltip": "把第一段参考图像缩放到目标尺寸时使用的算法。"}),
                "重绘轮数": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "完整 tile pass 次数。人物建议 1。"}),
                "分块顺序": (cls.tile_orders, {"tooltip": "tile 处理顺序。"}),
                "预览频率": (PREVIEW_MODE_CHOICES, {"tooltip": "运行时预览更新频率。"}),
                "最大分块数": ("INT", {"default": 4096, "min": 0, "max": 65536, "tooltip": "安全限制。预计 tile 数超过此值会报错，0 表示不限制。"}),
                "色彩稳定强度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "兼容旧工作流的轻量保色参数；建议保持 0。"}),
                "参考保留强度": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.8, "step": 0.01, "round": 0.001, "tooltip": "采样后把少量参考 latent 混回输出。"}),
                "主体重绘上限": ("FLOAT", {"default": 0.14, "min": 0.01, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "主体遮罩占比较高的 tile 的 denoise 上限。"}),
                "背景重绘倍率": ("FLOAT", {"default": 1.25, "min": 0.1, "max": 3.0, "step": 0.05, "round": 0.001, "tooltip": "主体占比较低的 tile 的重绘强度倍率。"}),
                "主体判断阈值": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "主体遮罩平均值超过该阈值时按主体 tile 处理。"}),
                "接缝修复": (["禁用", "启用"], {"tooltip": "最终阶段额外跑一轮低强度 seam pass。"}),
                "接缝修复强度": ("FLOAT", {"default": 0.06, "min": 0.01, "max": 0.5, "step": 0.01, "round": 0.001, "tooltip": "接缝修复 pass 的 denoise。"}),
                "接缝宽度": ("INT", {"default": 96, "min": 0, "max": 1024, "step": 8, "tooltip": "接缝修复 mask 的像素宽度。"}),
                "主体保护强度": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "主体保护遮罩的强度。"}),
            }
        }

    RETURN_TYPES = ("L13_REDRAW_SETTINGS",)
    RETURN_NAMES = ("高级参数",)
    FUNCTION = "build"
    CATEGORY = "sampling/l13_redraw"
    DESCRIPTION = "External advanced settings bundle for L13 tiled redraw nodes."

    def build(
        self,
        目标宽度,
        目标高度,
        递进强度衰减,
        重绘强度,
        细节扰动,
        分块宽度,
        分块高度,
        重叠像素,
        上下文像素,
        采样缓冲像素,
        融合方式,
        图像缩放算法,
        重绘轮数,
        分块顺序,
        预览频率,
        最大分块数,
        色彩稳定强度,
        参考保留强度,
        主体重绘上限,
        背景重绘倍率,
        主体判断阈值,
        接缝修复,
        接缝修复强度,
        接缝宽度,
        主体保护强度,
    ):
        return ({
            "目标宽度": 目标宽度,
            "目标高度": 目标高度,
            "递进强度衰减": 递进强度衰减,
            "重绘强度": 重绘强度,
            "细节扰动": 细节扰动,
            "分块宽度": 分块宽度,
            "分块高度": 分块高度,
            "重叠像素": 重叠像素,
            "上下文像素": 上下文像素,
            "采样缓冲像素": 采样缓冲像素,
            "融合方式": 融合方式,
            "图像缩放算法": 图像缩放算法,
            "重绘轮数": 重绘轮数,
            "分块顺序": 分块顺序,
            "预览频率": 预览频率,
            "最大分块数": 最大分块数,
            "色彩稳定强度": 色彩稳定强度,
            "参考保留强度": 参考保留强度,
            "主体重绘上限": 主体重绘上限,
            "背景重绘倍率": 背景重绘倍率,
            "主体判断阈值": 主体判断阈值,
            "接缝修复": 接缝修复,
            "接缝修复强度": 接缝修复强度,
            "接缝宽度": 接缝宽度,
            "主体保护强度": 主体保护强度,
        },)


class L13ImageColorMatch:
    methods = COLOR_MATCH_METHOD_CHOICES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "需要修正颜色的最终图像，通常接在 VAE Decode 后面。"}),
                "参考图像": ("IMAGE", {"tooltip": "第一段参考图像。节点会只读取低频颜色统计，不复制纹理。"}),
                "颜色匹配强度": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "0 不改变颜色，1 完全匹配参考图低频颜色。建议 0.20-0.45。"}),
                "匹配方式": (cls.methods, {"default": "RGB均值方差", "tooltip": "RGB均值方差更直接；YCbCr色度主要匹配色度，亮度只弱修正，更适合避免过度改明暗。"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "match"
    CATEGORY = "image/l13"
    DESCRIPTION = "Pixel-space low-frequency color matching for L13 redraw outputs."

    def match(self, 图像, 参考图像, 颜色匹配强度, 匹配方式):
        return (_match_image_color(图像, 参考图像, 颜色匹配强度, 匹配方式),)


def _crop_mask(mask: torch.Tensor, y0: int, y1: int, x0: int, x1: int) -> torch.Tensor:
    if mask.ndim < 2:
        return mask
    return mask[..., y0:y1, x0:x1]


def _intersect_area(area: Sequence[int], y0: int, y1: int, x0: int, x1: int) -> Optional[Tuple[int, int, int, int]]:
    if len(area) < 4:
        return None
    h, w, y, x = int(area[0]), int(area[1]), int(area[2]), int(area[3])
    iy0 = max(y, y0)
    ix0 = max(x, x0)
    iy1 = min(y + h, y1)
    ix1 = min(x + w, x1)
    if iy1 <= iy0 or ix1 <= ix0:
        return None
    return (iy1 - iy0, ix1 - ix0, iy0 - y0, ix0 - x0)


def _tile_conditioning(conds: Optional[List[Dict]], y0: int, y1: int, x0: int, x1: int) -> Optional[List[Dict]]:
    if conds is None:
        return None
    out = []
    for cond in conds:
        modified = cond.copy()
        if "area" in modified:
            local_area = _intersect_area(modified["area"], y0, y1, x0, x1)
            if local_area is None:
                continue
            modified["area"] = local_area
        if "mask" in modified:
            modified["mask"] = _crop_mask(modified["mask"], y0, y1, x0, x1)
        out.append(modified)
    return out


class TiledCondBatch:
    def __init__(
        self,
        tile_h: int,
        tile_w: int,
        overlap: int,
        context: int,
        blend: str,
        max_tiles: int,
        warn_controlnet: bool,
    ):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.overlap = overlap
        self.context = context
        self.blend = blend
        self.max_tiles = max_tiles
        self.warn_controlnet = warn_controlnet
        self._warned_control = False

    def _has_control(self, conds: Iterable[Optional[List[Dict]]]) -> bool:
        for cond_list in conds:
            if not cond_list:
                continue
            for cond in cond_list:
                if "control" in cond:
                    return True
        return False

    def __call__(self, args):
        conds = args["conds"]
        x = args["input"]
        sigma = args["sigma"]
        model = args["model"]
        model_options = args["model_options"]

        if x.ndim != 4:
            return comfy.samplers.calc_cond_batch(model, conds, x, sigma, model_options)

        _, _, height, width = x.shape
        tile_h = max(1, min(self.tile_h, height))
        tile_w = max(1, min(self.tile_w, width))
        overlap = max(0, min(self.overlap, tile_h - 1, tile_w - 1))
        context = max(0, min(self.context, height - 1, width - 1))
        tiles = _tile_grid(height, width, tile_h, tile_w, overlap)

        if self.max_tiles > 0 and len(tiles) > self.max_tiles:
            raise RuntimeError(
                f"Guided Tiled KSampler would run {len(tiles)} tiles per denoise call; "
                f"max_tiles is {self.max_tiles}. Increase max_tiles or use larger tile size."
            )

        if len(tiles) == 1:
            tile_options = model_options.copy()
            tile_options.pop("sampler_calc_cond_batch_function", None)
            return comfy.samplers.calc_cond_batch(model, conds, x, sigma, tile_options)

        if self.warn_controlnet and not self._warned_control and self._has_control(conds):
            logging.warning(
                "Guided Tiled KSampler detected ControlNet conditioning. "
                "The tiled sampler is intended for prompt/latent-guided 8K sampling; "
                "ControlNet hints may not align exactly per tile."
            )
            self._warned_control = True

        tile_options = model_options.copy()
        tile_options.pop("sampler_calc_cond_batch_function", None)

        accum = [torch.zeros_like(x) for _ in conds]
        counts = torch.zeros((1, 1, height, width), device=x.device, dtype=x.dtype)

        for y0, y1, x0, x1 in tiles:
            cy0 = max(0, y0 - context)
            cy1 = min(height, y1 + context)
            cx0 = max(0, x0 - context)
            cx1 = min(width, x1 + context)
            x_tile = x[:, :, cy0:cy1, cx0:cx1]
            tile_conds = [_tile_conditioning(c, cy0, cy1, cx0, cx1) for c in conds]
            tile_out = comfy.samplers.calc_cond_batch(model, tile_conds, x_tile, sigma, tile_options)
            oy0 = y0 - cy0
            oy1 = oy0 + (y1 - y0)
            ox0 = x0 - cx0
            ox1 = ox0 + (x1 - x0)
            weight = _tile_weight(
                y1 - y0,
                x1 - x0,
                height,
                width,
                y0,
                y1,
                x0,
                x1,
                overlap,
                self.blend,
                x.device,
                x.dtype,
            )
            counts[:, :, y0:y1, x0:x1] += weight
            for i, out in enumerate(tile_out):
                accum[i][:, :, y0:y1, x0:x1] += out[:, :, oy0:oy1, ox0:ox1] * weight

        counts = counts.clamp_min(torch.finfo(x.dtype).eps if x.dtype.is_floating_point else 1e-6)
        return [out / counts for out in accum]


class GuidedTiledKSampler8K:
    upscale_methods = ["bislerp", "bicubic", "bilinear", "nearest-exact", "area"]
    blend_modes = BLEND_CHOICES
    add_noise_modes = ADD_NOISE_CHOICES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL", {"tooltip": "用于分块去噪采样的扩散模型。"}),
                "正向条件": ("CONDITIONING", {"tooltip": "正向提示词条件，描述画面中希望出现的内容。"}),
                "负向条件": ("CONDITIONING", {"tooltip": "负向提示词条件，描述需要避免的内容。"}),
                "构图潜空间": ("LATENT", {"tooltip": "第一段低分辨率构图结果。节点会把它缩放到目标尺寸的潜空间后再分块采样。"}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "生成噪声用的种子。相同种子和参数会尽量复现同样结果。"}),
                "总步数": ("INT", {"default": 28, "min": 1, "max": 10000, "tooltip": "第二段 8K 分块采样使用的总去噪步数。"}),
                "CFG引导": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "提示词引导强度。数值越高越贴提示词，过高可能破坏画面。"}),
                "采样器": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "采样算法。建议先保持和第一段构图采样器一致。"}),
                "调度器": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "噪声调度方式。建议先保持和第一段构图调度器一致。"}),
                "参考来源": (REFERENCE_MODE_CHOICES, {"tooltip": "潜空间缩放会直接放大 latent；图像重编码会先用 VAE 解码构图图像、缩放到目标尺寸、再编码回 latent，通常更稳。"}),
                "目标规格": (TARGET_SIZE_CHOICES, {"tooltip": "自定义时使用目标宽高。选择 4K/8K 时，会把原 latent 长边变成 4096/8192，短边按原比例自动计算。"}),
                "目标宽度": ("INT", {"default": 8192, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "自定义目标宽度。若目标宽度和目标高度相等，例如 8192/8192，会把该值当成长边并保持原 latent 比例。"}),
                "目标高度": ("INT", {"default": 8192, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "自定义目标高度。若目标宽度和目标高度相等，例如 8192/8192，会把该值当成长边并保持原 latent 比例。"}),
                "重绘强度": ("FLOAT", {"default": 0.65, "min": 0.01, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "参考构图潜空间的强度。0.45-0.65 通常比较稳，越高越容易改变构图。"}),
                "分块宽度": ("INT", {"default": 1024, "min": 128, "max": MAX_RESOLUTION, "step": 8, "tooltip": "每个采样块的像素宽度。越大一致性越好，但显存占用越高。"}),
                "分块高度": ("INT", {"default": 1024, "min": 128, "max": MAX_RESOLUTION, "step": 8, "tooltip": "每个采样块的像素高度。越大一致性越好，但显存占用越高。"}),
                "重叠像素": ("INT", {"default": 192, "min": 0, "max": 2048, "step": 8, "tooltip": "相邻分块的重叠区域。建议 192-256，用来减少接缝。"}),
                "上下文像素": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 8, "tooltip": "每块采样时额外向四周读取的上下文，只把中心块写回。值越大越能参考周围内容，但显存占用越高。"}),
                "融合方式": (cls.blend_modes, {"tooltip": "重叠区域的融合曲线。余弦通常最稳，线性更直接，高斯边缘更柔。"}),
                "构图缩放算法": (cls.upscale_methods, {"tooltip": "把低分辨率构图潜空间缩放到目标潜空间时使用的算法。bislerp 通常适合 latent。"}),
                "图像缩放算法": (["lanczos", "bicubic", "bilinear", "nearest-exact", "area"], {"tooltip": "参考来源为图像重编码时使用。先缩放第一段图像，再 VAE 编码回目标 latent。"}),
                "加噪": (cls.add_noise_modes, {"tooltip": "是否在本节点开始时加入随机噪声。参考完整构图再重绘时通常启用；接高级分段后半段时通常禁用。"}),
                "最大分块数": ("INT", {"default": 4096, "min": 0, "max": 65536, "tooltip": "安全限制。预计分块数超过此值会报错，0 表示不限制。"}),
            },
            "optional": {
                "VAE": ("VAE", {"tooltip": "参考来源为图像重编码时需要。未连接参考图像时，会用这个 VAE 解码构图潜空间再重编码。"}),
                "参考图像": ("IMAGE", {"tooltip": "可选。若连接，则图像重编码会直接使用这张第一段完成图作为参考，而不是解码构图潜空间。"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "sampling/guided_tiled"
    DESCRIPTION = (
        "Creates a target-size latent from a low-resolution composition latent and samples it by "
        "running every denoise step in overlapping tiles. This is tiled sampling, not image upscaling."
    )

    def _prepare_target_latent(
        self,
        model,
        composition_latent,
        target_size,
        target_width,
        target_height,
        guide_upscale,
        reference_mode,
        image_upscale,
        vae=None,
        reference_image=None,
    ):
        samples = composition_latent["samples"]
        target_width, target_height = _target_pixels_from_latent_ratio(samples, target_size, target_width, target_height)
        latent_w = _latent_dim(target_width)
        latent_h = _latent_dim(target_height)

        latent_format = model.get_model_object("latent_format")
        channels = getattr(latent_format, "latent_channels", samples.shape[1])
        if samples.shape[1] != channels:
            if samples.shape[1] == 1:
                samples = samples.repeat(1, channels, 1, 1)
            else:
                raise RuntimeError(f"Composition latent has {samples.shape[1]} channels; model expects {channels}.")

        if reference_mode == "image":
            if vae is None:
                raise RuntimeError("参考来源为图像重编码时必须连接 VAE。")
            pixels = reference_image
            if pixels is None:
                pixels = vae.decode(samples)
            pixels = _scale_pixels(pixels, target_width, target_height, image_upscale)
            samples = _vae_encode_pixels(vae, pixels)
        elif samples.shape[-2:] != (latent_h, latent_w):
            samples = comfy.utils.common_upscale(samples, latent_w, latent_h, guide_upscale, "disabled")

        out = composition_latent.copy()
        out["samples"] = samples
        return out

    def _sample_tiled(
        self,
        model,
        positive,
        negative,
        input_latent,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        target_size,
        target_width,
        target_height,
        denoise,
        tile_width,
        tile_height,
        overlap,
        context,
        blend,
        guide_upscale,
        reference_mode,
        image_upscale,
        add_noise,
        max_tiles,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        vae=None,
        reference_image=None,
    ):
        target_latent = self._prepare_target_latent(
            model,
            input_latent,
            target_size,
            target_width,
            target_height,
            guide_upscale,
            reference_mode,
            image_upscale,
            vae=vae,
            reference_image=reference_image,
        )
        latent_image = comfy.sample.fix_empty_latent_channels(model, target_latent["samples"])
        target_latent["samples"] = latent_image

        disable_noise = add_noise == "disable"
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = target_latent.get("batch_index")
            noise = comfy.sample.prepare_noise(latent_image, noise_seed, batch_inds)

        noise_mask = target_latent.get("noise_mask")
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        tiled_model = model.clone()
        tiled_model.set_model_sampler_calc_cond_batch_function(
            TiledCondBatch(
                tile_h=_latent_dim(tile_height),
                tile_w=_latent_dim(tile_width),
                overlap=_latent_dim(overlap),
                context=_latent_dim(context),
                blend=blend,
                max_tiles=max_tiles,
                warn_controlnet=True,
            )
        )

        samples = comfy.sample.sample(
            tiled_model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
        )

        out = target_latent.copy()
        out["samples"] = samples
        return (out,)

    def sample(self, **kwargs):
        add_noise = ADD_NOISE_MAP[_param(kwargs, "加噪", "add_noise")]
        blend = BLEND_MAP[_param(kwargs, "融合方式", "blend")]
        reference_mode = REFERENCE_MODE_MAP[_param(kwargs, "参考来源", "reference_mode", default="潜空间缩放")]
        return self._sample_tiled(
            _param(kwargs, "模型", "model"),
            _param(kwargs, "正向条件", "positive"),
            _param(kwargs, "负向条件", "negative"),
            _param(kwargs, "构图潜空间", "composition_latent"),
            _param(kwargs, "随机种子", "seed"),
            _param(kwargs, "总步数", "steps"),
            _param(kwargs, "CFG引导", "cfg"),
            _param(kwargs, "采样器", "sampler_name"),
            _param(kwargs, "调度器", "scheduler"),
            _param(kwargs, "目标规格", "target_size", default="自定义"),
            _param(kwargs, "目标宽度", "target_width"),
            _param(kwargs, "目标高度", "target_height"),
            _param(kwargs, "重绘强度", "denoise"),
            _param(kwargs, "分块宽度", "tile_width"),
            _param(kwargs, "分块高度", "tile_height"),
            _param(kwargs, "重叠像素", "overlap"),
            _param(kwargs, "上下文像素", "context", default=0),
            blend,
            _param(kwargs, "构图缩放算法", "guide_upscale"),
            reference_mode,
            _param(kwargs, "图像缩放算法", "image_upscale", default="lanczos"),
            add_noise,
            _param(kwargs, "最大分块数", "max_tiles"),
            vae=_param(kwargs, "VAE", "vae", default=None),
            reference_image=_param(kwargs, "参考图像", "reference_image", default=None),
        )


class GuidedTiledKSamplerAdvanced8K(GuidedTiledKSampler8K):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL", {"tooltip": "用于分块去噪采样的扩散模型。"}),
                "加噪": (cls.add_noise_modes, {"tooltip": "是否在本节点开始时加入随机噪声。接第一段剩余噪声继续采样时通常设为禁用。"}),
                "噪声种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "生成噪声用的种子。高级分段流程中需和第一段保持一致。"}),
                "总步数": ("INT", {"default": 30, "min": 1, "max": 10000, "tooltip": "完整采样时间线的总步数。第一段和第二段必须填同一个总步数。"}),
                "CFG引导": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "提示词引导强度。建议和第一段保持一致。"}),
                "采样器": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "采样算法。高级分段流程中需和第一段保持一致。"}),
                "调度器": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "噪声调度方式。高级分段流程中需和第一段保持一致。"}),
                "正向条件": ("CONDITIONING", {"tooltip": "正向提示词条件，描述画面中希望出现的内容。"}),
                "负向条件": ("CONDITIONING", {"tooltip": "负向提示词条件，描述需要避免的内容。"}),
                "输入潜空间": ("LATENT", {"tooltip": "上一段输出的潜空间。可以接低分辨率完整构图，也可以接高级采样第一段的 leftover latent。"}),
                "起始步": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "本节点从完整采样时间线的第几步开始。接 0->3 的第一段时这里填 3。"}),
                "结束步": ("INT", {"default": 10000, "min": 0, "max": 10000, "tooltip": "本节点采样到第几步结束。通常填总步数，例如 30。"}),
                "保留剩余噪声": (LEFTOVER_NOISE_CHOICES, {"tooltip": "启用表示输出仍保留未采完的噪声，方便继续分段；最终输出通常设为禁用。"}),
                "参考来源": (REFERENCE_MODE_CHOICES, {"tooltip": "潜空间缩放会直接放大 latent；图像重编码会先用 VAE 解码参考图像、缩放到目标尺寸、再编码回 latent。高级分段接力通常保持潜空间缩放。"}),
                "目标规格": (TARGET_SIZE_CHOICES, {"tooltip": "自定义时使用目标宽高。选择 4K/8K 时，会把原 latent 长边变成 4096/8192，短边按原比例自动计算。"}),
                "目标宽度": ("INT", {"default": 8192, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "自定义目标宽度。若目标宽度和目标高度相等，例如 8192/8192，会把该值当成长边并保持原 latent 比例。"}),
                "目标高度": ("INT", {"default": 8192, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "自定义目标高度。若目标宽度和目标高度相等，例如 8192/8192，会把该值当成长边并保持原 latent 比例。"}),
                "重绘强度": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "高级分段接力通常填 1.0；参考完整构图再重绘时可用 0.45-0.65。"}),
                "分块宽度": ("INT", {"default": 1024, "min": 128, "max": MAX_RESOLUTION, "step": 8, "tooltip": "每个采样块的像素宽度。越大一致性越好，但显存占用越高。"}),
                "分块高度": ("INT", {"default": 1024, "min": 128, "max": MAX_RESOLUTION, "step": 8, "tooltip": "每个采样块的像素高度。越大一致性越好，但显存占用越高。"}),
                "重叠像素": ("INT", {"default": 192, "min": 0, "max": 2048, "step": 8, "tooltip": "相邻分块的重叠区域。建议 192-256，用来减少接缝。"}),
                "上下文像素": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 8, "tooltip": "每块采样时额外向四周读取的上下文，只把中心块写回。值越大越能参考周围内容，但显存占用越高。"}),
                "融合方式": (cls.blend_modes, {"tooltip": "重叠区域的融合曲线。余弦通常最稳，线性更直接，高斯边缘更柔。"}),
                "构图缩放算法": (cls.upscale_methods, {"tooltip": "把输入潜空间缩放到目标潜空间时使用的算法。bislerp 通常适合 latent。"}),
                "图像缩放算法": (["lanczos", "bicubic", "bilinear", "nearest-exact", "area"], {"tooltip": "参考来源为图像重编码时使用。先缩放第一段图像，再 VAE 编码回目标 latent。"}),
                "最大分块数": ("INT", {"default": 4096, "min": 0, "max": 65536, "tooltip": "安全限制。预计分块数超过此值会报错，0 表示不限制。"}),
            },
            "optional": {
                "VAE": ("VAE", {"tooltip": "参考来源为图像重编码时需要。未连接参考图像时，会用这个 VAE 解码输入潜空间再重编码。"}),
                "参考图像": ("IMAGE", {"tooltip": "可选。若连接，则图像重编码会直接使用这张第一段完成图作为参考，而不是解码输入潜空间。"}),
            }
        }

    DESCRIPTION = (
        "KSampler Advanced style tiled sampling. Use start/end steps to resume a low-resolution "
        "composition pass at target size with overlapping tiled denoise calls."
    )

    def sample(self, **kwargs):
        add_noise = ADD_NOISE_MAP[_param(kwargs, "加噪", "add_noise")]
        blend = BLEND_MAP[_param(kwargs, "融合方式", "blend")]
        reference_mode = REFERENCE_MODE_MAP[_param(kwargs, "参考来源", "reference_mode", default="潜空间缩放")]
        leftover_noise = LEFTOVER_NOISE_MAP[_param(kwargs, "保留剩余噪声", "return_with_leftover_noise")]
        force_full_denoise = leftover_noise == "disable"
        return self._sample_tiled(
            _param(kwargs, "模型", "model"),
            _param(kwargs, "正向条件", "positive"),
            _param(kwargs, "负向条件", "negative"),
            _param(kwargs, "输入潜空间", "latent_image"),
            _param(kwargs, "噪声种子", "noise_seed"),
            _param(kwargs, "总步数", "steps"),
            _param(kwargs, "CFG引导", "cfg"),
            _param(kwargs, "采样器", "sampler_name"),
            _param(kwargs, "调度器", "scheduler"),
            _param(kwargs, "目标规格", "target_size", default="自定义"),
            _param(kwargs, "目标宽度", "target_width"),
            _param(kwargs, "目标高度", "target_height"),
            _param(kwargs, "重绘强度", "denoise"),
            _param(kwargs, "分块宽度", "tile_width"),
            _param(kwargs, "分块高度", "tile_height"),
            _param(kwargs, "重叠像素", "overlap"),
            _param(kwargs, "上下文像素", "context", default=0),
            blend,
            _param(kwargs, "构图缩放算法", "guide_upscale"),
            reference_mode,
            _param(kwargs, "图像缩放算法", "image_upscale", default="lanczos"),
            add_noise,
            _param(kwargs, "最大分块数", "max_tiles"),
            start_step=_param(kwargs, "起始步", "start_at_step"),
            last_step=_param(kwargs, "结束步", "end_at_step"),
            force_full_denoise=force_full_denoise,
            vae=_param(kwargs, "VAE", "vae", default=None),
            reference_image=_param(kwargs, "参考图像", "reference_image", default=None),
        )


class L13ContextMaskedRedraw8K:
    blend_modes = BLEND_CHOICES
    tile_orders = TILE_ORDER_CHOICES
    target_sizes = TARGET_SIZE_CHOICES
    progressive_modes = PROGRESSIVE_MODE_CHOICES
    redraw_presets = REDRAW_PRESET_CHOICES
    enable_modes = ENABLE_CHOICES
    image_upscale_methods = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL", {"tooltip": "用于局部重绘的扩散模型。第二段会用同一模型在高分辨率 latent 上做 masked img2img。"}),
                "VAE": ("VAE", {"tooltip": "用于把第一段参考图像编码成高分辨率 latent。8K 会自动使用 tiled VAE encode。"}),
                "参考图像": ("IMAGE", {"tooltip": "第一段完整构图图像。节点会把它按原比例缩放到 4K/8K，再 VAE 编码为高分辨率画布。"}),
                "正向条件": ("CONDITIONING", {"tooltip": "第二段使用的正向提示词。可以和第一段相同，但建议 CFG 和重绘强度更低。"}),
                "负向条件": ("CONDITIONING", {"tooltip": "第二段使用的负向提示词。节点不会改写提示词，建议手动加入 duplicate / collage 等负向词。"}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "生成整张高分辨率统一噪声场的种子。每个 tile 从同一张噪声图裁切，避免 tile 独立随机。"}),
                "总步数": ("INT", {"default": 10, "min": 1, "max": 10000, "tooltip": "每个局部重绘 tile 的采样步数。人物图建议 10-20。"}),
                "CFG引导": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "第二段提示词引导强度。人物图建议低于第一段；低 CFG 更依赖参考图，不容易重复主体。"}),
                "采样器": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "局部重绘使用的采样器。默认 euler，稳定且预览直观。"}),
                "调度器": (comfy.samplers.KSampler.SCHEDULERS, {"default": "ddim_uniform", "tooltip": "局部重绘使用的调度器。默认 ddim_uniform，适合低 CFG 参考图重绘。"}),
                "参数预设": (cls.redraw_presets, {"default": "自定义", "tooltip": "运行时安全预设。自定义完全按参数跑；人物稳定/人物细节会限制高 CFG、高重绘和高扰动；背景/建筑会更偏向纹理和线条。"}),
                "人物安全模式": (cls.enable_modes, {"default": "启用", "tooltip": "启用后，在人物预设或连接主体遮罩时自动限制主体 tile 的重绘强度和细节扰动，降低重复主体、换脸和发灰风险。"}),
                "目标规格": (cls.target_sizes, {"default": "4K", "tooltip": "4K/8K 会保持参考图原比例，把长边设为 4096/8192。自定义时使用目标宽高；宽高相等时也按长边保持比例。"}),
                "递进放大模式": (cls.progressive_modes, {"default": "关闭", "tooltip": "关闭会直接生成目标尺寸。平衡1024阶梯会按长边每次增加约 1024 像素；稳定1.5倍更稳更慢；快速2倍更快但人物一致性风险更高。"}),
            },
            "optional": {
                "高级参数": ("L13_REDRAW_SETTINGS", {"tooltip": "可选。连接 L13 参考重绘放大参数 节点后，会覆盖本节点里折叠的高级参数。"}),
                "主体保护遮罩": ("MASK", {"tooltip": "可选。白色区域会降低中心 noise_mask 更新强度，用于保护人物主体，防止换人或复制主体。"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "sampling/l13_redraw"
    DESCRIPTION = "Reference-anchored context-aware masked img2img redraw pass for 4K/8K latent canvases."

    def _build_canvas_from_pixels(self, vae, pixels, target_width, target_height, image_upscale):
        pixels = _scale_pixels(pixels, target_width, target_height, image_upscale)
        samples = _vae_encode_pixels(vae, pixels)
        return samples

    def _build_canvas(self, vae, reference_image, target_size, target_width, target_height, image_upscale):
        scale = _vae_scale(vae)
        target_width, target_height = _target_pixels_from_image_ratio(reference_image, target_size, target_width, target_height, scale)
        return self._build_canvas_from_pixels(vae, reference_image, target_width, target_height, image_upscale)

    def _sample_tile(
        self,
        model,
        positive,
        negative,
        context_latent,
        context_noise,
        noise_mask,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        disable_pbar,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        callback=None,
    ):
        return comfy.sample.sample(
            model,
            context_noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            context_latent,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

    def _run_redraw(
        self,
        模型,
        VAE,
        参考图像,
        正向条件,
        负向条件,
        噪声种子,
        总步数,
        CFG引导,
        采样器,
        调度器,
        参数预设,
        人物安全模式,
        目标规格,
        目标宽度,
        目标高度,
        递进放大模式,
        递进强度衰减,
        重绘强度,
        细节扰动,
        分块宽度,
        分块高度,
        重叠像素,
        上下文像素,
        采样缓冲像素,
        融合方式,
        图像缩放算法,
        重绘轮数,
        分块顺序,
        最大分块数,
        色彩稳定强度=0.0,
        参考保留强度=0.06,
        主体重绘上限=0.14,
        背景重绘倍率=1.25,
        主体判断阈值=0.18,
        接缝修复="禁用",
        接缝修复强度=0.06,
        接缝宽度=96,
        预览频率="每个分块",
        加噪="启用",
        起始步=None,
        结束步=None,
        保留剩余噪声="禁用",
        主体保护遮罩=None,
        主体保护强度=0.55,
        高级参数=None,
    ):
        if isinstance(高级参数, dict):
            def setting(name, current):
                value = 高级参数.get(name, current)
                return current if value is None else value

            目标宽度 = setting("目标宽度", 目标宽度)
            目标高度 = setting("目标高度", 目标高度)
            递进强度衰减 = setting("递进强度衰减", 递进强度衰减)
            重绘强度 = setting("重绘强度", 重绘强度)
            细节扰动 = setting("细节扰动", 细节扰动)
            分块宽度 = setting("分块宽度", 分块宽度)
            分块高度 = setting("分块高度", 分块高度)
            重叠像素 = setting("重叠像素", 重叠像素)
            上下文像素 = setting("上下文像素", 上下文像素)
            采样缓冲像素 = setting("采样缓冲像素", 采样缓冲像素)
            融合方式 = setting("融合方式", 融合方式)
            图像缩放算法 = setting("图像缩放算法", 图像缩放算法)
            重绘轮数 = setting("重绘轮数", 重绘轮数)
            分块顺序 = setting("分块顺序", 分块顺序)
            最大分块数 = setting("最大分块数", 最大分块数)
            色彩稳定强度 = setting("色彩稳定强度", 色彩稳定强度)
            参考保留强度 = setting("参考保留强度", 参考保留强度)
            主体重绘上限 = setting("主体重绘上限", 主体重绘上限)
            背景重绘倍率 = setting("背景重绘倍率", 背景重绘倍率)
            主体判断阈值 = setting("主体判断阈值", 主体判断阈值)
            接缝修复 = setting("接缝修复", 接缝修复)
            接缝修复强度 = setting("接缝修复强度", 接缝修复强度)
            接缝宽度 = setting("接缝宽度", 接缝宽度)
            预览频率 = setting("预览频率", 预览频率)
            主体保护强度 = setting("主体保护强度", 主体保护强度)

        blend = BLEND_MAP[融合方式]
        add_noise = ADD_NOISE_MAP[加噪]
        leftover_noise = LEFTOVER_NOISE_MAP[保留剩余噪声]
        disable_noise = add_noise == "disable"
        force_full_denoise = leftover_noise == "disable"
        scale = _vae_scale(VAE)
        pass_count = max(1, int(重绘轮数))
        safety_enabled = _enabled(人物安全模式)
        seam_enabled = _enabled(接缝修复)
        (
            CFG引导,
            重绘强度,
            细节扰动,
            采样缓冲像素,
            主体重绘上限,
            背景重绘倍率,
            主体判断阈值,
            参考保留强度,
            seam_enabled,
            接缝修复强度,
            接缝宽度,
        ) = _apply_redraw_policy(
            参数预设,
            safety_enabled,
            主体保护遮罩 is not None,
            CFG引导,
            重绘强度,
            细节扰动,
            采样缓冲像素,
            主体重绘上限,
            背景重绘倍率,
            主体判断阈值,
            参考保留强度,
            seam_enabled,
            接缝修复强度,
            接缝宽度,
        )
        stage_pixels = _progressive_stage_pixels(
            参考图像,
            目标规格,
            目标宽度,
            目标高度,
            scale,
            递进放大模式,
        )
        stage_plans = []
        for stage_width, stage_height in stage_pixels:
            height = max(1, int(stage_height) // scale)
            width = max(1, int(stage_width) // scale)
            tile_h = max(1, min(_round_to_multiple_floor(分块高度, scale) // scale, height))
            tile_w = max(1, min(_round_to_multiple_floor(分块宽度, scale) // scale, width))
            overlap = max(0, min(_round_to_multiple_floor(重叠像素, scale) // scale, tile_h - 1, tile_w - 1))
            context = max(0, min(_round_to_multiple_floor(上下文像素, scale) // scale, height - 1, width - 1))
            sample_halo = 0
            if int(采样缓冲像素) > 0:
                sample_halo = max(0, min(_round_to_multiple_floor(采样缓冲像素, scale) // scale, height - 1, width - 1))
            tiles = _ordered_tiles(_tile_grid(height, width, tile_h, tile_w, overlap), height, width, 分块顺序)
            stage_plans.append((stage_width, stage_height, height, width, tile_h, tile_w, overlap, context, sample_halo, tiles))

        seam_runs = len(stage_plans[-1][-1]) if seam_enabled and 接缝宽度 > 0 and len(stage_plans[-1][-1]) > 1 else 0
        total_tile_runs = sum(len(plan[-1]) * pass_count for plan in stage_plans) + seam_runs
        if 最大分块数 > 0 and total_tile_runs > 最大分块数:
            raise RuntimeError(f"L13 参考重绘放大预计运行 {total_tile_runs} 个 tile，超过最大分块数 {最大分块数}。请增大 tile 或提高最大分块数。")

        sampler_steps = _effective_sampler_steps(总步数, 起始步, 结束步)
        progress = _CanvasProgress(模型, total_tile_runs * sampler_steps, 预览频率, VAE)
        current_pixels = 参考图像
        canvas = None
        stage_count = len(stage_plans)
        decay = float(递进强度衰减)

        for stage_index, (stage_width, stage_height, height, width, tile_h, tile_w, overlap, context, sample_halo, tiles) in enumerate(stage_plans):
            base = self._build_canvas_from_pixels(VAE, current_pixels, stage_width, stage_height, 图像缩放算法)
            base = comfy.sample.fix_empty_latent_channels(模型, base)
            canvas = base.clone()
            stage_denoise = float(重绘强度)
            if stage_count > 1:
                stage_denoise = max(0.01, min(1.0, stage_denoise * (decay ** stage_index)))

            if disable_noise:
                global_noise = torch.zeros(canvas.size(), dtype=canvas.dtype, layout=canvas.layout, device="cpu")
            else:
                global_noise = comfy.sample.prepare_noise(canvas, _seed_for_stage(噪声种子, stage_index), None)
            detail_noise = None
            if float(细节扰动) > 0:
                detail_seed = _seed_for_stage(噪声种子, stage_index, 0x9E3779B97F4A7C15)
                detail_noise = _normalize_detail_noise(comfy.sample.prepare_noise(canvas, detail_seed, None))

            protect_mask = None
            if 主体保护遮罩 is not None:
                protect_mask = _scale_mask(主体保护遮罩, width, height, canvas.device, canvas.dtype)
                protect_mask = protect_mask.clamp(0.0, 1.0)

            for pass_index in range(pass_count):
                accum = torch.zeros_like(canvas)
                weights = torch.zeros((canvas.shape[0], 1, height, width), device=canvas.device, dtype=canvas.dtype)

                for tile_index, (y0, y1, x0, x1) in enumerate(tiles):
                    sy0 = max(0, y0 - sample_halo)
                    sy1 = min(height, y1 + sample_halo)
                    sx0 = max(0, x0 - sample_halo)
                    sx1 = min(width, x1 + sample_halo)
                    cy0 = max(0, sy0 - context)
                    cy1 = min(height, sy1 + context)
                    cx0 = max(0, sx0 - context)
                    cx1 = min(width, sx1 + context)
                    iy0 = y0 - cy0
                    iy1 = iy0 + (y1 - y0)
                    ix0 = x0 - cx0
                    ix1 = ix0 + (x1 - x0)
                    sy0i = sy0 - cy0
                    sy1i = sy1 - cy0
                    sx0i = sx0 - cx0
                    sx1i = sx1 - cx0

                    context_latent = canvas[:, :, cy0:cy1, cx0:cx1].clone()
                    context_noise = global_noise[:, :, cy0:cy1, cx0:cx1].clone()
                    noise_mask = torch.zeros((canvas.shape[0], 1, cy1 - cy0, cx1 - cx0), device=canvas.device, dtype=canvas.dtype)
                    noise_mask[:, :, sy0i:sy1i, sx0i:sx1i] = 1.0
                    effective_denoise = stage_denoise

                    if protect_mask is not None:
                        subject_ratio = float(protect_mask[:, :, y0:y1, x0:x1].mean().item())
                        if subject_ratio >= float(主体判断阈值):
                            effective_denoise = min(effective_denoise, float(主体重绘上限))
                        else:
                            effective_denoise = min(1.0, effective_denoise * float(背景重绘倍率))
                        if 主体保护强度 > 0:
                            local_protect = protect_mask[:, :, cy0:cy1, cx0:cx1]
                            noise_mask = noise_mask * (1.0 - local_protect * float(主体保护强度))

                    if detail_noise is not None:
                        local_detail = detail_noise[:, :, cy0:cy1, cx0:cx1].to(device=context_latent.device, dtype=context_latent.dtype)
                        context_latent = context_latent + local_detail * noise_mask.to(device=context_latent.device, dtype=context_latent.dtype) * float(细节扰动)

                    progress.start_tile(sampler_steps)
                    out_context = self._sample_tile(
                        模型,
                        正向条件,
                        负向条件,
                        context_latent,
                        context_noise,
                        noise_mask,
                        int(噪声种子),
                        总步数,
                        CFG引导,
                        采样器,
                        调度器,
                        effective_denoise,
                        True,
                        disable_noise=disable_noise,
                        start_step=起始步,
                        last_step=结束步,
                        force_full_denoise=force_full_denoise,
                        callback=progress.tile_callback(),
                    )
                    if 预览频率 == "每个分块":
                        progress.capture_preview(out_context)
                    progress.finish_tile(sampler_steps, force_preview=(预览频率 == "每个分块"))
                    out_tile = out_context[:, :, iy0:iy1, ix0:ix1]
                    base_tile = base[:, :, y0:y1, x0:x1]
                    compatibility_hold = max(0.0, min(1.0, float(色彩稳定强度))) * 0.08
                    reference_hold = max(0.0, min(0.8, float(参考保留强度) + compatibility_hold))
                    out_tile = _blend_reference_latent(out_tile, base_tile, reference_hold)
                    weight = _tile_weight(y1 - y0, x1 - x0, height, width, y0, y1, x0, x1, overlap, blend, canvas.device, canvas.dtype)
                    accum[:, :, y0:y1, x0:x1] += out_tile * weight
                    weights[:, :, y0:y1, x0:x1] += weight

                canvas = accum / weights.clamp_min(torch.finfo(canvas.dtype).eps if canvas.dtype.is_floating_point else 1e-6)
                if 预览频率 == "每轮":
                    progress.capture_preview(canvas)
                    progress.force_preview()

            if seam_enabled and stage_index == stage_count - 1 and len(tiles) > 1:
                seam_latent = 0
                if int(接缝宽度) > 0:
                    seam_latent = max(0, _round_to_multiple_floor(接缝宽度, scale) // scale)
                seam_mask = _seam_mask_from_tiles(tiles, height, width, seam_latent, canvas.device, canvas.dtype)
                if seam_mask is not None:
                    seam_accum = torch.zeros_like(canvas)
                    seam_weights = torch.zeros((canvas.shape[0], 1, height, width), device=canvas.device, dtype=canvas.dtype)
                    for tile_index, (y0, y1, x0, x1) in enumerate(tiles):
                        sy0 = max(0, y0 - sample_halo)
                        sy1 = min(height, y1 + sample_halo)
                        sx0 = max(0, x0 - sample_halo)
                        sx1 = min(width, x1 + sample_halo)
                        cy0 = max(0, sy0 - context)
                        cy1 = min(height, sy1 + context)
                        cx0 = max(0, sx0 - context)
                        cx1 = min(width, sx1 + context)
                        iy0 = y0 - cy0
                        iy1 = iy0 + (y1 - y0)
                        ix0 = x0 - cx0
                        ix1 = ix0 + (x1 - x0)

                        context_latent = canvas[:, :, cy0:cy1, cx0:cx1].clone()
                        context_noise = global_noise[:, :, cy0:cy1, cx0:cx1].clone()
                        noise_mask = seam_mask[:, :, cy0:cy1, cx0:cx1].repeat(canvas.shape[0], 1, 1, 1)
                        if protect_mask is not None and 主体保护强度 > 0:
                            local_protect = protect_mask[:, :, cy0:cy1, cx0:cx1]
                            noise_mask = noise_mask * (1.0 - local_protect * float(主体保护强度))
                        if noise_mask.max().item() <= 0:
                            progress.start_tile(sampler_steps)
                            progress.finish_tile(sampler_steps)
                            continue

                        progress.start_tile(sampler_steps)
                        out_context = self._sample_tile(
                            模型,
                            正向条件,
                            负向条件,
                            context_latent,
                            context_noise,
                            noise_mask,
                            int(噪声种子),
                            总步数,
                            CFG引导,
                            采样器,
                            调度器,
                            接缝修复强度,
                            True,
                            disable_noise=disable_noise,
                            start_step=起始步,
                            last_step=结束步,
                            force_full_denoise=force_full_denoise,
                            callback=progress.tile_callback(),
                        )
                        if 预览频率 == "每个分块":
                            progress.capture_preview(out_context)
                        progress.finish_tile(sampler_steps, force_preview=(预览频率 == "每个分块"))
                        out_tile = out_context[:, :, iy0:iy1, ix0:ix1]
                        base_tile = base[:, :, y0:y1, x0:x1]
                        compatibility_hold = max(0.0, min(1.0, float(色彩稳定强度))) * 0.08
                        reference_hold = max(0.0, min(0.8, float(参考保留强度) + compatibility_hold))
                        out_tile = _blend_reference_latent(out_tile, base_tile, reference_hold)
                        seam_weight = seam_mask[:, :, y0:y1, x0:x1].repeat(canvas.shape[0], 1, 1, 1)
                        weight = _tile_weight(y1 - y0, x1 - x0, height, width, y0, y1, x0, x1, overlap, blend, canvas.device, canvas.dtype)
                        weight = weight * seam_weight
                        seam_accum[:, :, y0:y1, x0:x1] += out_tile * weight
                        seam_weights[:, :, y0:y1, x0:x1] += weight

                    eps = torch.finfo(canvas.dtype).eps if canvas.dtype.is_floating_point else 1e-6
                    updated = seam_accum / seam_weights.clamp_min(eps)
                    canvas = torch.where((seam_weights > eps).expand_as(canvas), updated, canvas)
                    if 预览频率 == "每轮":
                        progress.capture_preview(canvas)
                        progress.force_preview()

            if stage_index < stage_count - 1:
                current_pixels = _vae_decode_latent(VAE, canvas)

        return ({"samples": canvas},)

    def sample(
        self,
        模型,
        VAE,
        参考图像,
        正向条件,
        负向条件,
        随机种子,
        总步数,
        CFG引导,
        采样器,
        调度器,
        参数预设,
        人物安全模式,
        目标规格,
        递进放大模式,
        目标宽度=8192,
        目标高度=8192,
        递进强度衰减=0.85,
        重绘强度=0.22,
        细节扰动=0.0,
        分块宽度=1280,
        分块高度=1280,
        重叠像素=256,
        上下文像素=512,
        采样缓冲像素=96,
        融合方式="余弦",
        图像缩放算法="lanczos",
        重绘轮数=1,
        分块顺序="顺序",
        最大分块数=4096,
        色彩稳定强度=0.0,
        参考保留强度=0.06,
        主体重绘上限=0.14,
        背景重绘倍率=1.25,
        主体判断阈值=0.18,
        接缝修复="禁用",
        接缝修复强度=0.06,
        接缝宽度=96,
        预览频率="每个分块",
        高级参数=None,
        主体保护遮罩=None,
        主体保护强度=0.55,
    ):
        return self._run_redraw(
            模型,
            VAE,
            参考图像,
            正向条件,
            负向条件,
            随机种子,
            总步数,
            CFG引导,
            采样器,
            调度器,
            参数预设,
            人物安全模式,
            目标规格,
            目标宽度,
            目标高度,
            递进放大模式,
            递进强度衰减,
            重绘强度,
            细节扰动,
            分块宽度,
            分块高度,
            重叠像素,
            上下文像素,
            采样缓冲像素,
            融合方式,
            图像缩放算法,
            重绘轮数,
            分块顺序,
            最大分块数,
            色彩稳定强度,
            参考保留强度,
            主体重绘上限,
            背景重绘倍率,
            主体判断阈值,
            接缝修复,
            接缝修复强度,
            接缝宽度,
            预览频率=预览频率,
            高级参数=高级参数,
            主体保护遮罩=主体保护遮罩,
            主体保护强度=主体保护强度,
        )


class L13ContextMaskedRedrawAdvanced8K(L13ContextMaskedRedraw8K):
    add_noise_modes = ADD_NOISE_CHOICES
    leftover_noise_modes = LEFTOVER_NOISE_CHOICES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL", {"tooltip": "用于局部重绘的扩散模型。高级版使用 KSampler Advanced 的起止步逻辑。"}),
                "VAE": ("VAE", {"tooltip": "用于把第一段参考图像编码成高分辨率 latent。8K 会自动使用 tiled VAE encode。"}),
                "参考图像": ("IMAGE", {"tooltip": "第一段完整构图图像。高级版仍以参考图像为全局锚点，不做自由分块 txt2img。"}),
                "加噪": (cls.add_noise_modes, {"tooltip": "是否在本段开始时加入噪声。第一段通常启用；承接上一段剩余噪声时通常禁用。"}),
                "噪声种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "生成整张高分辨率统一噪声场的种子。分段采样时各段必须保持一致。"}),
                "总步数": ("INT", {"default": 10, "min": 1, "max": 10000, "tooltip": "完整采样时间线的总步数。高级分段时每一段都填同一个总步数。"}),
                "CFG引导": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "第二段提示词引导强度。人物图建议低于第一段；低 CFG 更依赖参考图，不容易重复主体。"}),
                "采样器": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "局部重绘使用的采样器。分段采样时各段应保持一致。"}),
                "调度器": (comfy.samplers.KSampler.SCHEDULERS, {"default": "ddim_uniform", "tooltip": "局部重绘使用的调度器。分段采样时各段应保持一致。"}),
                "正向条件": ("CONDITIONING", {"tooltip": "第二段使用的正向提示词。可以和第一段相同，但不要用过高 CFG。"}),
                "负向条件": ("CONDITIONING", {"tooltip": "第二段使用的负向提示词。建议加入 duplicate person、collage、split image 等负向词。"}),
                "参数预设": (cls.redraw_presets, {"default": "自定义", "tooltip": "运行时安全预设。自定义完全按参数跑；人物稳定/人物细节会限制高 CFG、高重绘和高扰动；背景/建筑会更偏向纹理和线条。"}),
                "人物安全模式": (cls.enable_modes, {"default": "启用", "tooltip": "启用后，在人物预设或连接主体遮罩时自动限制主体 tile 的重绘强度和细节扰动，降低重复主体、换脸和发灰风险。"}),
                "起始步": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "本段从完整采样时间线的第几步开始。比如先构图 3 步后，这里填 3。"}),
                "结束步": ("INT", {"default": 10000, "min": 0, "max": 10000, "tooltip": "本段采样到第几步结束。最终段通常填总步数。"}),
                "保留剩余噪声": (cls.leftover_noise_modes, {"tooltip": "启用表示输出保留未采完的噪声，方便继续接下一段；最终输出通常设为禁用。"}),
                "目标规格": (cls.target_sizes, {"default": "4K", "tooltip": "4K/8K 会保持参考图原比例，把长边设为 4096/8192。自定义时使用目标宽高；宽高相等时也按长边保持比例。"}),
                "递进放大模式": (cls.progressive_modes, {"default": "关闭", "tooltip": "关闭会直接生成目标尺寸。平衡1024阶梯会按长边每次增加约 1024 像素；稳定1.5倍更稳更慢；快速2倍更快但人物一致性风险更高。"}),
            },
            "optional": {
                "高级参数": ("L13_REDRAW_SETTINGS", {"tooltip": "可选。连接 L13 参考重绘放大参数 节点后，会覆盖本节点里折叠的高级参数。"}),
                "主体保护遮罩": ("MASK", {"tooltip": "可选。白色区域会降低中心 noise_mask 更新强度，用于保护人物主体，防止换人或复制主体。"}),
            }
        }

    DESCRIPTION = "KSampler Advanced style reference-anchored context masked redraw pass for 4K/8K latent canvases."

    def sample(
        self,
        模型,
        VAE,
        参考图像,
        加噪,
        噪声种子,
        总步数,
        CFG引导,
        采样器,
        调度器,
        正向条件,
        负向条件,
        参数预设,
        人物安全模式,
        起始步,
        结束步,
        保留剩余噪声,
        目标规格,
        递进放大模式,
        目标宽度=8192,
        目标高度=8192,
        递进强度衰减=0.85,
        重绘强度=0.22,
        细节扰动=0.0,
        分块宽度=1280,
        分块高度=1280,
        重叠像素=256,
        上下文像素=512,
        采样缓冲像素=96,
        融合方式="余弦",
        图像缩放算法="lanczos",
        重绘轮数=1,
        分块顺序="顺序",
        最大分块数=4096,
        色彩稳定强度=0.0,
        参考保留强度=0.06,
        主体重绘上限=0.14,
        背景重绘倍率=1.25,
        主体判断阈值=0.18,
        接缝修复="禁用",
        接缝修复强度=0.06,
        接缝宽度=96,
        预览频率="每个分块",
        高级参数=None,
        主体保护遮罩=None,
        主体保护强度=0.55,
    ):
        return self._run_redraw(
            模型,
            VAE,
            参考图像,
            正向条件,
            负向条件,
            噪声种子,
            总步数,
            CFG引导,
            采样器,
            调度器,
            参数预设,
            人物安全模式,
            目标规格,
            目标宽度,
            目标高度,
            递进放大模式,
            递进强度衰减,
            重绘强度,
            细节扰动,
            分块宽度,
            分块高度,
            重叠像素,
            上下文像素,
            采样缓冲像素,
            融合方式,
            图像缩放算法,
            重绘轮数,
            分块顺序,
            最大分块数,
            色彩稳定强度,
            参考保留强度,
            主体重绘上限,
            背景重绘倍率,
            主体判断阈值,
            接缝修复,
            接缝修复强度,
            接缝宽度,
            预览频率=预览频率,
            加噪=加噪,
            起始步=起始步,
            结束步=结束步,
            保留剩余噪声=保留剩余噪声,
            高级参数=高级参数,
            主体保护遮罩=主体保护遮罩,
            主体保护强度=主体保护强度,
        )


NODE_CLASS_MAPPINGS = {
    "Layer13GuidedTiledKSampler8K": GuidedTiledKSampler8K,
    "Layer13GuidedTiledKSamplerAdvanced8K": GuidedTiledKSamplerAdvanced8K,
    "Layer13RedrawSettings": L13RedrawSettings,
    "Layer13ImageColorMatch": L13ImageColorMatch,
    "Layer13ContextMaskedRedraw8K": L13ContextMaskedRedraw8K,
    "Layer13ContextMaskedRedrawAdvanced8K": L13ContextMaskedRedrawAdvanced8K,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13GuidedTiledKSampler8K": "Guided Tiled KSampler 8K (Layer13)",
    "Layer13GuidedTiledKSamplerAdvanced8K": "Guided Tiled KSampler Advanced 8K (Layer13)",
    "Layer13RedrawSettings": "L13 参考重绘放大参数",
    "Layer13ImageColorMatch": "L13 图像颜色匹配",
    "Layer13ContextMaskedRedraw8K": "L13 参考重绘放大",
    "Layer13ContextMaskedRedrawAdvanced8K": "L13 参考重绘放大（高级）",
}
