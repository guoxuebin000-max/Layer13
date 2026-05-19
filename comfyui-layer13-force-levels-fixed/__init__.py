from __future__ import annotations

import torch


def _stretch_to_full_range(x: torch.Tensor, black_level: int, white_level: int) -> torch.Tensor:
    # The user may input "black 253 / white 2". We treat them as two fixed clip points
    # and always stretch the enclosed range back to 0..1, never invert.
    a = float(black_level) / 255.0
    b = float(white_level) / 255.0
    low = min(a, b)
    high = max(a, b)
    span = max(high - low, 1e-6)
    return torch.clamp((x - low) / span, 0.0, 1.0)


class Layer13ForceFixedLevels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "黑阶": ("INT", {"default": 253, "min": 0, "max": 255, "step": 1}),
                "白阶": ("INT", {"default": 2, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Layer13/调色"

    def apply(
        self,
        image: torch.Tensor,
        黑阶: int = 253,
        白阶: int = 2,
    ):
        img = image.clone().to(dtype=torch.float32)

        luma = (
            img[..., 0] * 0.2126
            + img[..., 1] * 0.7152
            + img[..., 2] * 0.0722
        )
        new_luma = _stretch_to_full_range(luma, 黑阶, 白阶)

        scale = (new_luma / torch.clamp(luma, min=1e-6)).unsqueeze(-1)
        out = torch.clamp(img * scale, 0.0, 1.0)

        near_zero_luma = (luma <= 1e-6).unsqueeze(-1)
        lifted_floor = new_luma.unsqueeze(-1).expand_as(out)
        out = torch.where(near_zero_luma, lifted_floor, out)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "Layer13ForceFixedLevels": Layer13ForceFixedLevels,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13ForceFixedLevels": "Layer13 强制黑白阶(亮度)",
}
