import math

import torch
import torch.nn.functional as F


def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError("图像必须是 ComfyUI IMAGE 张量。")
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("图像必须是 IMAGE 批量张量 (B,H,W,C)，且至少包含 RGB 三通道。")
    return image


def _ensure_mask_batch(mask: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{name} 必须是 ComfyUI MASK 张量。")
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4 and mask.shape[-1] == 1:
        return mask[..., 0]
    raise ValueError(f"{name} 必须是 MASK 张量，形状应为 (H,W) 或 (B,H,W)。")


def _resize_mask(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    mask_4d = mask.unsqueeze(1).to(dtype=torch.float32)
    if mask_4d.shape[-2:] != (target_h, target_w):
        mask_4d = F.interpolate(mask_4d, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return mask_4d.squeeze(1).clamp(0.0, 1.0)


def _match_batch_count(tensor: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1 and batch > 1:
        return tensor.repeat(batch, *([1] * (tensor.ndim - 1)))
    raise ValueError(f"{name} 批次数量不匹配：需要 1 或 {batch}，实际 {tensor.shape[0]}。")


def _parse_rgb_color(value: str) -> tuple[float, float, float]:
    text = str(value or "").strip()
    if not text:
        return (0.0, 1.0, 0.0)

    if text.startswith("#"):
        text = text[1:]
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        if len(text) == 6:
            try:
                return tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
            except ValueError:
                return (0.0, 1.0, 0.0)

    parts = [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]
    if len(parts) >= 3:
        try:
            values = [float(parts[i]) for i in range(3)]
        except ValueError:
            return (0.0, 1.0, 0.0)
        if max(values) > 1.0:
            values = [v / 255.0 for v in values]
        return tuple(float(max(0.0, min(1.0, v))) for v in values)

    return (0.0, 1.0, 0.0)


def _color_protect_mask(
    rgb: torch.Tensor,
    color: str,
    tolerance: float,
    feather: float,
    strength: float,
) -> torch.Tensor:
    strength = float(max(0.0, min(1.0, strength)))
    tolerance = float(max(0.0, min(1.0, tolerance)))
    if strength <= 0.0 or tolerance <= 0.0:
        return torch.zeros(rgb.shape[:-1], device=rgb.device, dtype=torch.float32)

    target = rgb.new_tensor(_parse_rgb_color(color)).view(1, 1, 1, 3)
    distance = torch.linalg.vector_norm(rgb[..., :3] - target, dim=-1) / 1.7320508075688772
    feather = float(max(0.0, min(1.0, feather)))
    soft_width = max(1e-5, tolerance * max(0.02, feather))
    protect = 1.0 - torch.clamp((distance - tolerance) / soft_width, 0.0, 1.0)
    return (protect * strength).clamp(0.0, 1.0)


def _make_generator(device: torch.device, seed: int):
    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator()
    generator.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return generator


def _randn(shape, device: torch.device, dtype: torch.dtype, generator):
    try:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    except Exception:
        return torch.randn(shape, dtype=dtype, generator=generator).to(device=device)


def _nchw(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 3, 1, 2).contiguous()


def _bhwc(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 2, 3, 1).contiguous()


def _box_blur_rgb(rgb: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return rgb
    kernel = radius * 2 + 1
    x = _nchw(rgb)
    x = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
    return _bhwc(x)


def _soft_skin_mask(rgb: torch.Tensor) -> torch.Tensor:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    skin = torch.exp(-(((cb - 0.36) ** 2) / 0.0072 + ((cr - 0.59) ** 2) / 0.0064))
    skin = skin * torch.clamp((y - 0.12) / 0.18, 0.0, 1.0)
    skin = skin * torch.clamp((0.95 - y) / 0.25, 0.0, 1.0)
    return skin.clamp(0.0, 1.0)


def _simulate_light_compression(rgb: torch.Tensor, amount: float) -> torch.Tensor:
    amount = float(max(0.0, min(1.0, amount)))
    if amount <= 0.0:
        return rgb

    batch, height, width, _ = rgb.shape
    scale = max(0.55, 1.0 - amount * 0.35)
    small_h = max(8, int(round(height * scale)))
    small_w = max(8, int(round(width * scale)))

    x = _nchw(rgb)
    down = F.interpolate(x, size=(small_h, small_w), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(height, width), mode="bilinear", align_corners=False)
    softened = _bhwc(up)

    levels = max(48.0, 255.0 - amount * 150.0)
    quantized = torch.round(softened * levels) / levels
    return rgb * (1.0 - amount * 0.45) + quantized * (amount * 0.45)


def _temporal_randn_bhwc(
    shape,
    device: torch.device,
    dtype: torch.dtype,
    generator,
    state: dict,
    key: str,
    consistency: float,
    drift_pixels: float,
    frame_index: int,
):
    current = _randn(shape, device, dtype, generator)
    previous = state.get(key)
    if previous is not None and tuple(previous.shape) == tuple(current.shape):
        if drift_pixels > 0.0 and current.ndim >= 4:
            drift = int(round(drift_pixels * frame_index))
            if drift:
                previous = torch.roll(previous, shifts=(drift, drift // 2), dims=(2, 1))
        c = float(max(0.0, min(0.995, consistency)))
        fresh_weight = max(0.01, (1.0 - c * c) ** 0.5)
        current = previous * c + current * fresh_weight
    state[key] = current.detach()
    return current


class Layer13RealPixelTexture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "强度": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "暗部噪声": ("FLOAT", {"default": 0.030, "min": 0.0, "max": 0.2, "step": 0.001}),
                "胶片颗粒": ("FLOAT", {"default": 0.050, "min": 0.0, "max": 0.25, "step": 0.001}),
                "颗粒大小": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "压缩感": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "边缘柔化": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "色度扰动": ("FLOAT", {"default": 0.050, "min": 0.0, "max": 0.08, "step": 0.001}),
                "高光保护": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "肤色保护": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "皮肤真实感": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "毛孔纹理": ("FLOAT", {"default": 0.050, "min": 0.0, "max": 0.16, "step": 0.001}),
                "皮肤细节": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "肤色不均": ("FLOAT", {"default": 0.100, "min": 0.0, "max": 0.12, "step": 0.001}),
                "皮肤纹理大小": ("INT", {"default": 2, "min": 1, "max": 12, "step": 1}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "效果遮罩": ("MASK",),
                "保护遮罩": ("MASK",),
                "皮肤遮罩": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "实际作用遮罩")
    FUNCTION = "处理"
    CATEGORY = "Layer13"

    def 处理(
        self,
        图像,
        强度=0.35,
        暗部噪声=0.030,
        胶片颗粒=0.050,
        颗粒大小=2,
        压缩感=0.08,
        边缘柔化=0.25,
        色度扰动=0.050,
        高光保护=0.65,
        肤色保护=0.55,
        皮肤真实感=0.20,
        毛孔纹理=0.050,
        皮肤细节=0.35,
        肤色不均=0.100,
        皮肤纹理大小=2,
        随机种子=0,
        效果遮罩=None,
        保护遮罩=None,
        皮肤遮罩=None,
    ):
        image = _ensure_image_batch(图像)
        device = image.device
        dtype = image.dtype
        image_f = image.to(dtype=torch.float32)

        batch, height, width, channels = image_f.shape
        rgb = image_f[..., :3]
        luma = (rgb * rgb.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=-1)

        if 效果遮罩 is None:
            mask = torch.ones((batch, height, width), device=device, dtype=torch.float32)
        else:
            mask = _ensure_mask_batch(效果遮罩, "效果遮罩").to(device=device)
            mask = _match_batch_count(mask, batch, "效果遮罩")
            mask = _resize_mask(mask, height, width)

        if 保护遮罩 is not None:
            protect = _ensure_mask_batch(保护遮罩, "保护遮罩").to(device=device)
            protect = _match_batch_count(protect, batch, "保护遮罩")
            protect = _resize_mask(protect, height, width)
            mask = mask * (1.0 - protect.clamp(0.0, 1.0))

        strength = float(max(0.0, min(1.0, 强度)))
        highlight_protect = float(max(0.0, min(1.0, 高光保护)))
        skin_protect = float(max(0.0, min(1.0, 肤色保护)))

        highlight = torch.clamp((luma - 0.55) / 0.45, 0.0, 1.0)
        skin = _soft_skin_mask(rgb)
        if 皮肤遮罩 is not None:
            skin_input = _ensure_mask_batch(皮肤遮罩, "皮肤遮罩").to(device=device)
            skin_input = _match_batch_count(skin_input, batch, "皮肤遮罩")
            skin_input = _resize_mask(skin_input, height, width)
            skin = (skin * 0.25 + skin_input * 0.75).clamp(0.0, 1.0)

        effect = mask * strength
        effect = effect * (1.0 - highlight * highlight_protect)
        effect = effect * (1.0 - skin * skin_protect)
        effect_c = effect.unsqueeze(-1).clamp(0.0, 1.0)

        work = rgb

        edge_amount = float(max(0.0, min(1.0, 边缘柔化))) * 0.45
        if edge_amount > 0.0:
            blurred = _box_blur_rgb(work, 1)
            work = work * (1.0 - edge_amount * effect_c) + blurred * (edge_amount * effect_c)

        compression_amount = float(max(0.0, min(1.0, 压缩感)))
        if compression_amount > 0.0:
            compressed = _simulate_light_compression(work, compression_amount)
            mix = compression_amount * 0.55 * effect_c
            work = work * (1.0 - mix) + compressed * mix

        generator = _make_generator(device, int(随机种子))
        dark_weight = torch.clamp((1.0 - luma) ** 1.4, 0.0, 1.0).unsqueeze(-1)

        sensor_amount = float(max(0.0, min(0.2, 暗部噪声)))
        if sensor_amount > 0.0:
            sensor = _randn((batch, height, width, 3), device, torch.float32, generator)
            sensor = sensor * sensor_amount * (0.25 + dark_weight * 1.75)
            work = work + sensor * effect_c

        grain_amount = float(max(0.0, min(0.25, 胶片颗粒)))
        if grain_amount > 0.0:
            grain_size = max(1, int(颗粒大小))
            grain_h = max(1, height // grain_size)
            grain_w = max(1, width // grain_size)
            grain = _randn((batch, 1, grain_h, grain_w), device, torch.float32, generator)
            grain = F.interpolate(grain, size=(height, width), mode="bilinear", align_corners=False)
            grain = _bhwc(grain)
            grain = grain * grain_amount * (0.45 + dark_weight * 0.85)
            work = work + grain.repeat(1, 1, 1, 3) * effect_c

        chroma_amount = float(max(0.0, min(0.08, 色度扰动)))
        if chroma_amount > 0.0:
            chroma_h = max(4, height // 24)
            chroma_w = max(4, width // 24)
            chroma = _randn((batch, 2, chroma_h, chroma_w), device, torch.float32, generator)
            chroma = F.interpolate(chroma, size=(height, width), mode="bilinear", align_corners=False)
            c1 = chroma[:, 0, :, :].unsqueeze(-1)
            c2 = chroma[:, 1, :, :].unsqueeze(-1)
            shift = torch.cat((c1, -(c1 + c2) * 0.25, c2), dim=-1) * chroma_amount
            work = work + shift * effect_c

        skin_real_amount = float(max(0.0, min(1.0, 皮肤真实感)))
        skin_effect = None
        if skin_real_amount > 0.0:
            skin_live = torch.clamp((luma - 0.08) / 0.22, 0.0, 1.0)
            skin_effect = mask * skin * skin_real_amount
            skin_effect = skin_effect * skin_live * (1.0 - highlight * 0.65)
            skin_effect_c = skin_effect.unsqueeze(-1).clamp(0.0, 1.0)
            texture_radius = max(1, int(皮肤纹理大小))

            detail_amount = float(max(0.0, min(1.0, 皮肤细节)))
            if detail_amount > 0.0:
                base_blur = _box_blur_rgb(work, texture_radius)
                detail = (work - base_blur).clamp(-0.25, 0.25)
                work = work + detail * detail_amount * 0.65 * skin_effect_c

            pore_amount = float(max(0.0, min(0.16, 毛孔纹理)))
            if pore_amount > 0.0:
                pore = _randn((batch, height, width, 1), device, torch.float32, generator)
                pore = pore - _box_blur_rgb(pore, texture_radius)
                pore = torch.tanh(pore * 1.8)
                pore = pore * pore_amount * (0.75 + dark_weight * 0.5)
                work = work + pore.repeat(1, 1, 1, 3) * skin_effect_c

            mottle_amount = float(max(0.0, min(0.12, 肤色不均)))
            if mottle_amount > 0.0:
                small_h = max(4, height // 48)
                small_w = max(4, width // 48)
                red_map = _randn((batch, 1, small_h, small_w), device, torch.float32, generator)
                warm_map = _randn((batch, 1, small_h, small_w), device, torch.float32, generator)
                red_map = F.interpolate(red_map, size=(height, width), mode="bicubic", align_corners=False)
                warm_map = F.interpolate(warm_map, size=(height, width), mode="bicubic", align_corners=False)
                red_map = _bhwc(red_map).clamp(-2.0, 2.0)
                warm_map = _bhwc(warm_map).clamp(-2.0, 2.0)
                red_tint = rgb.new_tensor([0.90, 0.22, 0.12]).view(1, 1, 1, 3)
                warm_tint = rgb.new_tensor([0.35, 0.22, -0.16]).view(1, 1, 1, 3)
                mottle = (red_map * red_tint + warm_map * warm_tint) * mottle_amount
                work = work + mottle * skin_effect_c

        output = image_f.clone()
        output[..., :3] = work.clamp(0.0, 1.0)
        if channels > 3:
            output[..., 3:] = image_f[..., 3:]

        if skin_effect is None:
            actual_mask = effect
        else:
            actual_mask = torch.maximum(effect, skin_effect)

        return (output.clamp(0.0, 1.0).to(dtype=dtype), actual_mask.clamp(0.0, 1.0))


class Layer13VideoRealPixelTexture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频帧": ("IMAGE",),
                "强度": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "时间一致性": ("FLOAT", {"default": 0.930, "min": 0.0, "max": 0.995, "step": 0.005}),
                "纹理漂移": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 0.1}),
                "时间自然波动": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.01}),
                "色彩波动": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "色彩稳定": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "对焦呼吸": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "后段细节补偿": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "暗部噪声": ("FLOAT", {"default": 0.012, "min": 0.0, "max": 0.2, "step": 0.001}),
                "胶片颗粒": ("FLOAT", {"default": 0.008, "min": 0.0, "max": 0.25, "step": 0.001}),
                "传感器纹理": ("FLOAT", {"default": 0.018, "min": 0.0, "max": 0.12, "step": 0.001}),
                "行列噪声": ("FLOAT", {"default": 0.010, "min": 0.0, "max": 0.08, "step": 0.001}),
                "颗粒大小": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "压缩感": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "边缘柔化": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "色度扰动": ("FLOAT", {"default": 0.020, "min": 0.0, "max": 0.08, "step": 0.001}),
                "高光保护": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "肤色保护": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "皮肤真实感": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "毛孔纹理": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.16, "step": 0.001}),
                "皮肤细节": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "肤色不均": ("FLOAT", {"default": 0.020, "min": 0.0, "max": 0.12, "step": 0.001}),
                "皮肤纹理大小": ("INT", {"default": 2, "min": 1, "max": 12, "step": 1}),
                "保护颜色": ("STRING", {"default": "#c00020"}),
                "颜色容差": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.001}),
                "颜色羽化": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "颜色保护强度": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "效果遮罩": ("MASK",),
                "保护遮罩": ("MASK",),
                "皮肤遮罩": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("视频帧", "实际作用遮罩", "信息")
    FUNCTION = "处理"
    CATEGORY = "Layer13"

    def 处理(
        self,
        视频帧,
        强度=0.35,
        时间一致性=0.930,
        纹理漂移=0.0,
        时间自然波动=0.06,
        色彩波动=0.0,
        色彩稳定=0.65,
        对焦呼吸=0.05,
        后段细节补偿=0.10,
        暗部噪声=0.012,
        胶片颗粒=0.008,
        传感器纹理=0.018,
        行列噪声=0.010,
        颗粒大小=2,
        压缩感=0.12,
        边缘柔化=0.18,
        色度扰动=0.020,
        高光保护=0.65,
        肤色保护=0.35,
        皮肤真实感=0.18,
        毛孔纹理=0.025,
        皮肤细节=0.30,
        肤色不均=0.020,
        皮肤纹理大小=2,
        保护颜色="#c00020",
        颜色容差=0.08,
        颜色羽化=0.35,
        颜色保护强度=0.70,
        随机种子=0,
        效果遮罩=None,
        保护遮罩=None,
        皮肤遮罩=None,
    ):
        image = _ensure_image_batch(视频帧)
        device = image.device
        dtype = image.dtype
        image_f = image.to(dtype=torch.float32)

        frame_count, height, width, channels = image_f.shape
        if frame_count <= 0:
            raise ValueError("视频帧批次为空。")

        if 效果遮罩 is None:
            effect_mask = torch.ones((frame_count, height, width), device=device, dtype=torch.float32)
        else:
            effect_mask = _ensure_mask_batch(效果遮罩, "效果遮罩").to(device=device)
            effect_mask = _match_batch_count(effect_mask, frame_count, "效果遮罩")
            effect_mask = _resize_mask(effect_mask, height, width)

        if 保护遮罩 is not None:
            protect = _ensure_mask_batch(保护遮罩, "保护遮罩").to(device=device)
            protect = _match_batch_count(protect, frame_count, "保护遮罩")
            protect = _resize_mask(protect, height, width)
            effect_mask = effect_mask * (1.0 - protect.clamp(0.0, 1.0))

        color_protect_strength = float(max(0.0, min(1.0, 颜色保护强度)))
        if color_protect_strength > 0.0:
            color_protect = _color_protect_mask(
                image_f[..., :3],
                保护颜色,
                颜色容差,
                颜色羽化,
                color_protect_strength,
            )
            effect_mask = effect_mask * (1.0 - color_protect)

        if 皮肤遮罩 is not None:
            skin_input = _ensure_mask_batch(皮肤遮罩, "皮肤遮罩").to(device=device)
            skin_input = _match_batch_count(skin_input, frame_count, "皮肤遮罩")
            skin_input = _resize_mask(skin_input, height, width)
        else:
            skin_input = None

        strength = float(max(0.0, min(1.0, 强度)))
        consistency = float(max(0.0, min(0.995, 时间一致性)))
        drift = float(max(0.0, min(8.0, 纹理漂移)))
        temporal_natural = float(max(0.0, min(1.0, 时间自然波动)))
        color_wobble = float(max(0.0, min(1.0, 色彩波动)))
        color_stabilize = float(max(0.0, min(1.0, 色彩稳定)))
        focus_breathe = float(max(0.0, min(1.0, 对焦呼吸)))
        late_detail_boost = float(max(0.0, min(1.0, 后段细节补偿)))
        highlight_protect = float(max(0.0, min(1.0, 高光保护)))
        skin_protect = float(max(0.0, min(1.0, 肤色保护)))
        sensor_amount = float(max(0.0, min(0.2, 暗部噪声)))
        grain_amount = float(max(0.0, min(0.25, 胶片颗粒)))
        sensor_texture_amount = float(max(0.0, min(0.12, 传感器纹理)))
        row_col_amount = float(max(0.0, min(0.08, 行列噪声)))
        chroma_amount = float(max(0.0, min(0.08, 色度扰动)))
        compression_amount = float(max(0.0, min(1.0, 压缩感)))
        edge_amount = float(max(0.0, min(1.0, 边缘柔化))) * 0.45
        skin_real_amount = float(max(0.0, min(1.0, 皮肤真实感)))
        pore_amount = float(max(0.0, min(0.16, 毛孔纹理)))
        detail_amount = float(max(0.0, min(1.0, 皮肤细节)))
        mottle_amount = float(max(0.0, min(0.12, 肤色不均)))
        grain_size = max(1, int(颗粒大小))
        texture_radius = max(1, int(皮肤纹理大小))

        generator = _make_generator(device, int(随机种子))
        temporal_state = {}
        outputs = []
        masks = []
        seed_phase = (int(随机种子) % 1000003) / 1000003.0
        color_reference = None

        for frame_index in range(frame_count):
            frame = image_f[frame_index : frame_index + 1]
            rgb = frame[..., :3]
            luma = (rgb * rgb.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=-1)
            mask = effect_mask[frame_index : frame_index + 1]
            progress = frame_index / max(1, frame_count - 1)
            late_weight = torch.clamp(rgb.new_tensor((progress - 0.45) / 0.55), 0.0, 1.0)

            highlight = torch.clamp((luma - 0.55) / 0.45, 0.0, 1.0)
            skin = _soft_skin_mask(rgb)
            if skin_input is not None:
                skin = (skin * 0.25 + skin_input[frame_index : frame_index + 1] * 0.75).clamp(0.0, 1.0)

            effect = mask * strength
            effect = effect * (1.0 - highlight * highlight_protect)
            effect = effect * (1.0 - skin * skin_protect)
            effect_c = effect.unsqueeze(-1).clamp(0.0, 1.0)
            work = rgb

            if temporal_natural > 0.0:
                t = frame_index / 30.0
                exposure_wave = (
                    math.sin(t * 1.13 + seed_phase * 6.28318530718)
                    + 0.45 * math.sin(t * 0.37 + seed_phase * 11.7)
                )
                wb_wave = math.sin(t * 0.71 + seed_phase * 8.9)
                exposure_gain = 1.0 + exposure_wave * temporal_natural * 0.010
                wb_shift = wb_wave * temporal_natural * color_wobble * 0.006
                wb = rgb.new_tensor([1.0 + wb_shift, 1.0, 1.0 - wb_shift]).view(1, 1, 1, 3)
                work = (work * exposure_gain * wb).clamp(0.0, 1.0)

            if edge_amount > 0.0:
                blurred = _box_blur_rgb(work, 1)
                work = work * (1.0 - edge_amount * effect_c) + blurred * (edge_amount * effect_c)

            if focus_breathe > 0.0:
                focus_wave = math.sin(frame_index / 30.0 * 0.83 + seed_phase * 5.1)
                focus_mix = abs(focus_wave) * focus_breathe * 0.10
                if focus_mix > 0.0:
                    soft = _box_blur_rgb(work, 1)
                    detail = (work - soft).clamp(-0.20, 0.20)
                    if focus_wave >= 0:
                        work = work + detail * focus_mix
                    else:
                        work = work * (1.0 - focus_mix) + soft * focus_mix

            if compression_amount > 0.0:
                compressed = _simulate_light_compression(work, compression_amount)
                mix = compression_amount * 0.55 * effect_c
                work = work * (1.0 - mix) + compressed * mix

            dark_weight = torch.clamp((1.0 - luma) ** 1.4, 0.0, 1.0).unsqueeze(-1)
            late_boost = 1.0 + float(late_weight.detach().cpu()) * late_detail_boost * 0.65

            if sensor_texture_amount > 0.0:
                texture_consistency = min(0.985, max(0.90, consistency))
                texture = _temporal_randn_bhwc(
                    (1, height, width, 1),
                    device,
                    torch.float32,
                    generator,
                    temporal_state,
                    "sensor_texture",
                    texture_consistency,
                    drift * 0.25,
                    frame_index,
                )
                low_texture = _box_blur_rgb(texture, 2)
                high_texture = (texture - low_texture).clamp(-2.0, 2.0)
                texture = low_texture * 0.35 + high_texture * 0.65
                texture = texture * sensor_texture_amount * late_boost * (0.12 + dark_weight * 0.65)
                work = work + texture.repeat(1, 1, 1, 3) * effect_c

            if row_col_amount > 0.0:
                fixed_consistency = min(0.995, max(0.96, consistency))
                row_noise = _temporal_randn_bhwc(
                    (1, height, 1, 1),
                    device,
                    torch.float32,
                    generator,
                    temporal_state,
                    "row_noise",
                    fixed_consistency,
                    0.0,
                    frame_index,
                )
                col_noise = _temporal_randn_bhwc(
                    (1, 1, width, 1),
                    device,
                    torch.float32,
                    generator,
                    temporal_state,
                    "col_noise",
                    fixed_consistency,
                    0.0,
                    frame_index,
                )
                fixed_pattern = row_noise.expand(1, height, width, 1) * 0.65
                fixed_pattern = fixed_pattern + col_noise.expand(1, height, width, 1) * 0.35
                fixed_pattern = fixed_pattern * row_col_amount * late_boost * (0.08 + dark_weight * 0.45)
                work = work + fixed_pattern.repeat(1, 1, 1, 3) * effect_c

            if sensor_amount > 0.0:
                sensor = _temporal_randn_bhwc(
                    (1, height, width, 3),
                    device,
                    torch.float32,
                    generator,
                    temporal_state,
                    "sensor",
                    min(0.75, consistency * 0.75),
                    drift,
                    frame_index,
                )
                sensor = sensor * sensor_amount * late_boost * (0.25 + dark_weight * 1.75)
                work = work + sensor * effect_c

            if grain_amount > 0.0:
                grain_h = max(1, height // grain_size)
                grain_w = max(1, width // grain_size)
                grain = _temporal_randn_bhwc(
                    (1, grain_h, grain_w, 1),
                    device,
                    torch.float32,
                    generator,
                    temporal_state,
                    "grain",
                    consistency,
                    drift / max(1.0, float(grain_size)),
                    frame_index,
                )
                grain = _nchw(grain)
                grain = F.interpolate(grain, size=(height, width), mode="bilinear", align_corners=False)
                grain = _bhwc(grain)
                grain = grain * grain_amount * late_boost * (0.45 + dark_weight * 0.85)
                work = work + grain.repeat(1, 1, 1, 3) * effect_c

            if chroma_amount > 0.0:
                chroma_h = max(4, height // 24)
                chroma_w = max(4, width // 24)
                chroma = _temporal_randn_bhwc(
                    (1, chroma_h, chroma_w, 2),
                    device,
                    torch.float32,
                    generator,
                    temporal_state,
                    "chroma",
                    consistency,
                    drift / 24.0,
                    frame_index,
                )
                chroma = _nchw(chroma)
                chroma = F.interpolate(chroma, size=(height, width), mode="bilinear", align_corners=False)
                chroma = _bhwc(chroma)
                c1 = chroma[..., 0:1]
                c2 = chroma[..., 1:2]
                shift = torch.cat((c1, -(c1 + c2) * 0.25, c2), dim=-1) * chroma_amount * late_boost
                work = work + shift * effect_c

            skin_effect = None
            if skin_real_amount > 0.0:
                skin_live = torch.clamp((luma - 0.08) / 0.22, 0.0, 1.0)
                skin_effect = mask * skin * skin_real_amount * late_boost
                skin_effect = skin_effect * skin_live * (1.0 - highlight * 0.65)
                skin_effect_c = skin_effect.unsqueeze(-1).clamp(0.0, 1.0)

                if detail_amount > 0.0:
                    base_blur = _box_blur_rgb(work, texture_radius)
                    detail = (work - base_blur).clamp(-0.25, 0.25)
                    work = work + detail * detail_amount * 0.65 * late_boost * skin_effect_c

                if pore_amount > 0.0:
                    pore = _temporal_randn_bhwc(
                        (1, height, width, 1),
                        device,
                        torch.float32,
                        generator,
                        temporal_state,
                        "pore",
                        min(0.97, max(consistency, 0.85)),
                        drift,
                        frame_index,
                    )
                    pore = pore - _box_blur_rgb(pore, texture_radius)
                    pore = torch.tanh(pore * 1.8)
                    pore = pore * pore_amount * (0.75 + dark_weight * 0.5)
                    work = work + pore.repeat(1, 1, 1, 3) * skin_effect_c

                if mottle_amount > 0.0:
                    small_h = max(4, height // 48)
                    small_w = max(4, width // 48)
                    red_map = _temporal_randn_bhwc(
                        (1, small_h, small_w, 1),
                        device,
                        torch.float32,
                        generator,
                        temporal_state,
                        "red_mottle",
                        min(0.985, max(consistency, 0.92)),
                        drift / 48.0,
                        frame_index,
                    )
                    warm_map = _temporal_randn_bhwc(
                        (1, small_h, small_w, 1),
                        device,
                        torch.float32,
                        generator,
                        temporal_state,
                        "warm_mottle",
                        min(0.985, max(consistency, 0.92)),
                        drift / 48.0,
                        frame_index,
                    )
                    red_map = _nchw(red_map)
                    warm_map = _nchw(warm_map)
                    red_map = F.interpolate(red_map, size=(height, width), mode="bicubic", align_corners=False)
                    warm_map = F.interpolate(warm_map, size=(height, width), mode="bicubic", align_corners=False)
                    red_map = _bhwc(red_map).clamp(-2.0, 2.0)
                    warm_map = _bhwc(warm_map).clamp(-2.0, 2.0)
                    red_tint = rgb.new_tensor([0.90, 0.22, 0.12]).view(1, 1, 1, 3)
                    warm_tint = rgb.new_tensor([0.35, 0.22, -0.16]).view(1, 1, 1, 3)
                    mottle = (red_map * red_tint + warm_map * warm_tint) * mottle_amount
                    work = work + mottle * skin_effect_c

            if color_stabilize > 0.0:
                frame_mean = work.mean(dim=(0, 1, 2), keepdim=True)
                if color_reference is None:
                    color_reference = frame_mean.detach()
                else:
                    color_reference = color_reference * 0.96 + frame_mean.detach() * 0.04
                correction = (color_reference - frame_mean).clamp(-0.025, 0.025)
                work = (work + correction * color_stabilize * 0.85).clamp(0.0, 1.0)

            output = frame.clone()
            output[..., :3] = work.clamp(0.0, 1.0)
            if channels > 3:
                output[..., 3:] = frame[..., 3:]
            outputs.append(output.clamp(0.0, 1.0).to(dtype=dtype))

            if skin_effect is None:
                actual_mask = effect
            else:
                actual_mask = torch.maximum(effect, skin_effect)
            masks.append(actual_mask.clamp(0.0, 1.0))

        info = (
            f"帧数={frame_count}, 尺寸={width}x{height}, 时间一致性={consistency:.3f}, "
            f"纹理漂移={drift:.2f}, seed={int(随机种子)}"
        )
        return (torch.cat(outputs, dim=0), torch.cat(masks, dim=0), info)


NODE_CLASS_MAPPINGS = {
    "Layer13RealPixelTexture": Layer13RealPixelTexture,
    "Layer13VideoRealPixelTexture": Layer13VideoRealPixelTexture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13RealPixelTexture": "Layer13真实像素+皮肤",
    "Layer13VideoRealPixelTexture": "Layer13视频真实像素+皮肤",
}
