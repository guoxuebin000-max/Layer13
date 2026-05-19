from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageFilter


def _tensor_to_pil(sample: torch.Tensor) -> Image.Image:
    data = sample.detach().float().cpu().clamp(0.0, 1.0).numpy()
    array = (data[..., :3] * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    data = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(data)


def _resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    long_edge = int(long_edge)
    if long_edge <= 0:
        return image

    width, height = image.size
    current = max(width, height)
    if current == long_edge:
        return image

    scale = long_edge / float(current)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _gaussian_blur_array(image: np.ndarray, radius: float) -> np.ndarray:
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB")
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=max(0.1, float(radius))))
    return np.asarray(blurred).astype(np.float32) / 255.0


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    mask = (
        (r > 0.30)
        & (g > 0.18)
        & (b > 0.12)
        & (r > g * 0.95)
        & (r > b * 1.02)
        & ((maxc - minc) > 0.035)
        & (lum > 0.22)
        & (lum < 0.96)
    ).astype(np.float32)

    if mask.max() <= 0:
        return mask
    mask_rgb = np.repeat(mask[..., None], 3, axis=-1)
    mask_rgb = _gaussian_blur_array(mask_rgb, 5.0)[..., 0]
    return np.clip(mask_rgb, 0.0, 1.0)


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    quality = max(1, min(100, int(quality)))
    if quality >= 100:
        return image
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _apply_phone_look(
    image: Image.Image,
    strength: float,
    hdr: float,
    warmth: float,
    beauty: float,
    sharpen: float,
    noise: float,
    jpeg_quality: int,
    seed: int,
) -> Image.Image:
    strength = np.clip(float(strength), 0.0, 1.0)
    hdr = np.clip(float(hdr), 0.0, 1.0) * strength
    warmth = np.clip(float(warmth), -1.0, 1.0) * strength
    beauty = np.clip(float(beauty), 0.0, 1.0) * strength
    sharpen = np.clip(float(sharpen), 0.0, 1.0) * strength
    noise = np.clip(float(noise), 0.0, 1.0) * strength

    base = np.asarray(image).astype(np.float32) / 255.0
    rgb = base.copy()
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    shadow = np.clip((0.68 - lum) / 0.68, 0.0, 1.0)[..., None]
    highlight = np.clip((lum - 0.72) / 0.28, 0.0, 1.0)[..., None]
    rgb = rgb + shadow * (0.20 * hdr)
    rgb = rgb - highlight * (0.06 * hdr)
    rgb = rgb + (rgb * (1.0 - rgb)) * (0.18 * hdr)

    if warmth != 0:
        rgb = rgb + np.array([0.045, 0.018, -0.030], dtype=np.float32) * warmth

    gray = lum[..., None]
    rgb = gray + (rgb - gray) * (1.0 + 0.07 * strength)

    # Phone HDR often has strong micro-contrast and edge sharpening.
    blur = _gaussian_blur_array(rgb, 2.0)
    rgb = rgb + (rgb - blur) * (0.28 * hdr)

    mask = _skin_mask(rgb)
    if beauty > 0 and mask.max() > 0:
        smooth = _gaussian_blur_array(rgb, 2.2)
        blend = (mask * min(0.55, 0.62 * beauty))[..., None]
        rgb = rgb * (1.0 - blend) + smooth * blend
        skin_tint = np.array([0.035, 0.018, 0.025], dtype=np.float32)
        rgb = rgb + (mask[..., None] * skin_tint * beauty)

    if sharpen > 0:
        blur = _gaussian_blur_array(rgb, 0.85)
        rgb = rgb + (rgb - blur) * (0.65 * sharpen)

    if noise > 0:
        rng = np.random.default_rng(int(seed) if int(seed) != 0 else None)
        lum2 = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        shadow_weight = np.clip((0.75 - lum2) / 0.75, 0.25, 1.0)[..., None]
        luma_noise = rng.normal(0.0, 0.012 * noise, rgb.shape[:2] + (1,)).astype(np.float32)
        chroma_noise = rng.normal(0.0, 0.006 * noise, rgb.shape).astype(np.float32)
        rgb = rgb + luma_noise * shadow_weight + chroma_noise

    rgb = np.clip(rgb, 0.0, 1.0)
    processed = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), "RGB")
    return _jpeg_roundtrip(processed, jpeg_quality)


class Layer13PhoneCamera:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "强度": ("FLOAT", {"default": 0.68, "min": 0.0, "max": 1.0, "step": 0.01}),
                "手机HDR": ("FLOAT", {"default": 0.52, "min": 0.0, "max": 1.0, "step": 0.01}),
                "暖色": ("FLOAT", {"default": 0.14, "min": -1.0, "max": 1.0, "step": 0.01}),
                "肤色美颜": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "锐化": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "细噪声": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "JPEG质量": ("INT", {"default": 88, "min": 50, "max": 100, "step": 1}),
                "输出长边": ("INT", {"default": 1700, "min": 0, "max": 8192, "step": 8}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "处理"
    CATEGORY = "Layer13/图像"

    def 处理(
        self,
        图像,
        强度=0.68,
        手机HDR=0.52,
        暖色=0.14,
        肤色美颜=0.22,
        锐化=0.42,
        细噪声=0.10,
        JPEG质量=88,
        输出长边=1700,
        随机种子=0,
    ):
        if not isinstance(图像, torch.Tensor) or 图像.ndim != 4:
            raise ValueError("图像必须是 ComfyUI IMAGE 张量。")

        outputs = []
        for index, sample in enumerate(图像):
            pil = _tensor_to_pil(sample)
            pil = _resize_long_edge(pil, int(输出长边))
            pil = _apply_phone_look(
                pil,
                强度,
                手机HDR,
                暖色,
                肤色美颜,
                锐化,
                细噪声,
                JPEG质量,
                int(随机种子) + index if int(随机种子) else 0,
            )
            outputs.append(_pil_to_tensor(pil))

        first_shape = outputs[0].shape
        normalized = []
        for tensor in outputs:
            if tensor.shape != first_shape:
                image = Image.fromarray((tensor.numpy() * 255.0 + 0.5).astype(np.uint8), "RGB")
                image = image.resize((first_shape[1], first_shape[0]), Image.Resampling.LANCZOS)
                tensor = _pil_to_tensor(image)
            normalized.append(tensor)

        return (torch.stack(normalized, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Layer13PhoneCamera": Layer13PhoneCamera,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13PhoneCamera": "Layer13手机成像",
}
