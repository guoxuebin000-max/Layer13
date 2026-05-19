import torch
import torch.nn.functional as F


def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError("图像必须是 ComfyUI IMAGE 张量。")
    if image.ndim != 4:
        raise ValueError("图像必须是 IMAGE 批量张量 (B,H,W,C)。")
    return image


def _ensure_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("遮罩必须是 ComfyUI MASK 张量。")
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4 and mask.shape[-1] == 1:
        return mask[..., 0]
    raise ValueError("遮罩必须是 MASK 张量，形状应为 (H,W) 或 (B,H,W)。")


def _resize_mask(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    mask_4d = mask.unsqueeze(1).to(dtype=torch.float32)
    if mask_4d.shape[-2:] != (target_h, target_w):
        mask_4d = F.interpolate(mask_4d, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return mask_4d.squeeze(1).clamp(0.0, 1.0)


def _feather_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    padded = mask.unsqueeze(1)
    blurred = F.avg_pool2d(padded, kernel_size=kernel, stride=1, padding=radius)
    return blurred.squeeze(1).clamp(0.0, 1.0)


class Layer13MaskedExposure:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "曝光EV": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "羽化半径": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "反相遮罩": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "实际遮罩")
    FUNCTION = "处理"
    CATEGORY = "Layer13"

    def 处理(self, 图像, 遮罩, 曝光EV=1.0, 羽化半径=0, 反相遮罩=False):
        image = _ensure_image_batch(图像)
        mask = _ensure_mask_batch(遮罩)

        batch, height, width, _ = image.shape
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.repeat(batch, 1, 1)
        elif mask.shape[0] != batch:
            raise ValueError(f"遮罩批次数量与图像不匹配：图像 {batch}，遮罩 {mask.shape[0]}")

        mask = _resize_mask(mask.to(device=image.device), height, width)
        mask = _feather_mask(mask, int(羽化半径))
        if 反相遮罩:
            mask = 1.0 - mask

        gain = float(2.0 ** float(曝光EV))
        adjusted = image.to(dtype=torch.float32) * gain
        blended_mask = mask.unsqueeze(-1)
        output = image.to(dtype=torch.float32) * (1.0 - blended_mask) + adjusted * blended_mask
        output = output.clamp(0.0, 1.0).to(dtype=image.dtype)

        return (output, mask)


NODE_CLASS_MAPPINGS = {
    "Layer13MaskedExposure": Layer13MaskedExposure,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13MaskedExposure": "layer13 遮罩曝光",
}
