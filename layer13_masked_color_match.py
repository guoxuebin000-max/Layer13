import torch
import torch.nn.functional as F


def _ensure_image_batch(image: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"{name} 必须是 ComfyUI IMAGE 张量。")
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"{name} 必须是 IMAGE 批量张量 (B,H,W,C)。")
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


def _feather_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    blurred = F.avg_pool2d(mask.unsqueeze(1), kernel_size=kernel, stride=1, padding=radius)
    return blurred.squeeze(1).clamp(0.0, 1.0)


def _match_batch_count(tensor: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1 and batch > 1:
        return tensor.repeat(batch, *([1] * (tensor.ndim - 1)))
    raise ValueError(f"{name} 批次数量不匹配：需要 1 或 {batch}，实际 {tensor.shape[0]}")


def _weighted_stats(image: torch.Tensor, mask: torch.Tensor, threshold: float, eps: float):
    weights = mask.clamp(0.0, 1.0)
    if threshold > 0:
        weights = torch.where(weights >= threshold, weights, torch.zeros_like(weights))

    sum_w = weights.sum(dim=(0, 1)).clamp_min(eps)
    if float(sum_w.max().detach().cpu()) <= eps:
        raise ValueError("遮罩区域太小，无法计算颜色统计。")

    weights_c = weights.unsqueeze(-1)
    mean = (image * weights_c).sum(dim=(0, 1)) / sum_w
    variance = ((image - mean.view(1, 1, -1)) ** 2 * weights_c).sum(dim=(0, 1)) / sum_w
    std = torch.sqrt(variance.clamp_min(eps))
    return mean, std


def _luma_coeffs(channels: int, device, dtype) -> torch.Tensor:
    coeffs = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=dtype)[:channels]
    return coeffs / coeffs.sum().clamp_min(1e-6)


def _luma_stats(image: torch.Tensor, mask: torch.Tensor, threshold: float, eps: float):
    channels = min(int(image.shape[-1]), 3)
    coeffs = _luma_coeffs(channels, image.device, image.dtype).view(1, 1, channels)
    luma = (image[..., :channels] * coeffs).sum(dim=-1, keepdim=True)
    mean, std = _weighted_stats(luma, mask, threshold, eps)
    return luma, mean, std


def _match_luma_distribution(
    matched: torch.Tensor,
    reference: torch.Tensor,
    target_mask: torch.Tensor,
    ref_mask: torch.Tensor,
    threshold: float,
    eps: float,
):
    matched_luma, matched_mean, matched_std = _luma_stats(matched, target_mask, threshold, eps)
    _, ref_mean, ref_std = _luma_stats(reference, ref_mask, threshold, eps)
    corrected_luma = (matched_luma - matched_mean.view(1, 1, 1)) * (
        ref_std / matched_std.clamp_min(eps)
    ).view(1, 1, 1) + ref_mean.view(1, 1, 1)
    return matched + (corrected_luma - matched_luma)


class Layer13MaskedColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "参考图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "匹配方式": (["均值+方差", "仅均值", "亮度比例"], {"default": "均值+方差"}),
                "强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "羽化半径": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
                "统计阈值": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "反相遮罩": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "参考遮罩": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("图像", "实际遮罩", "参考实际遮罩")
    FUNCTION = "处理"
    CATEGORY = "Layer13"

    def 处理(
        self,
        图像,
        参考图像,
        遮罩,
        匹配方式="均值+方差",
        强度=1.0,
        羽化半径=8,
        统计阈值=0.05,
        反相遮罩=False,
        参考遮罩=None,
    ):
        image = _ensure_image_batch(图像, "图像")
        reference = _ensure_image_batch(参考图像, "参考图像").to(device=image.device)
        mask = _ensure_mask_batch(遮罩, "遮罩").to(device=image.device)

        batch, height, width, channels = image.shape
        reference = _match_batch_count(reference, batch, "参考图像")
        mask = _match_batch_count(mask, batch, "遮罩")

        target_mask = _resize_mask(mask, height, width)
        if 反相遮罩:
            target_mask = 1.0 - target_mask

        if 参考遮罩 is None:
            ref_mask = _resize_mask(mask, int(reference.shape[1]), int(reference.shape[2]))
            if 反相遮罩:
                ref_mask = 1.0 - ref_mask
        else:
            ref_mask = _ensure_mask_batch(参考遮罩, "参考遮罩").to(device=image.device)
            ref_mask = _match_batch_count(ref_mask, batch, "参考遮罩")
            ref_mask = _resize_mask(ref_mask, int(reference.shape[1]), int(reference.shape[2]))

        blend_mask = _feather_mask(target_mask, int(羽化半径))
        strength = float(max(0.0, min(2.0, 强度)))
        threshold = float(max(0.0, min(1.0, 统计阈值)))
        eps = 1e-6

        image_f = image.to(dtype=torch.float32)
        reference_f = reference.to(dtype=torch.float32)
        output = image_f.clone()
        rgb_channels = min(3, channels)

        for index in range(batch):
            target_rgb = image_f[index, :, :, :rgb_channels]
            reference_rgb = reference_f[index, :, :, :rgb_channels]
            target_mean, target_std = _weighted_stats(target_rgb, target_mask[index], threshold, eps)
            ref_mean, ref_std = _weighted_stats(reference_rgb, ref_mask[index], threshold, eps)

            if 匹配方式 == "仅均值":
                matched = target_rgb + (ref_mean - target_mean).view(1, 1, rgb_channels)
            elif 匹配方式 == "亮度比例":
                target_luma = (target_mean * target_rgb.new_tensor([0.2126, 0.7152, 0.0722])[:rgb_channels]).sum()
                ref_luma = (ref_mean * target_rgb.new_tensor([0.2126, 0.7152, 0.0722])[:rgb_channels]).sum()
                gain = (ref_luma / target_luma.clamp_min(eps)).clamp(0.05, 20.0)
                matched = target_rgb * gain
            else:
                matched = (target_rgb - target_mean.view(1, 1, rgb_channels))
                matched = matched * (ref_std / target_std.clamp_min(eps)).view(1, 1, rgb_channels)
                matched = matched + ref_mean.view(1, 1, rgb_channels)
                matched = _match_luma_distribution(
                    matched,
                    reference_rgb,
                    target_mask[index],
                    ref_mask[index],
                    threshold,
                    eps,
                )

            matched = target_rgb + (matched - target_rgb) * strength
            mask_c = blend_mask[index].unsqueeze(-1)
            output[index, :, :, :rgb_channels] = target_rgb * (1.0 - mask_c) + matched * mask_c

        return (
            output.clamp(0.0, 1.0).to(dtype=image.dtype),
            blend_mask,
            ref_mask,
        )


NODE_CLASS_MAPPINGS = {
    "Layer13MaskedColorMatch": Layer13MaskedColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13MaskedColorMatch": "layer13 遮罩颜色匹配",
}
