from __future__ import annotations

import torch


def _levels_curve(x: torch.Tensor, black_point: int, white_point: int, gamma: float) -> torch.Tensor:
    black = float(black_point) / 255.0
    white = float(white_point) / 255.0
    span = max(white - black, 1e-6)

    y = ((x - black) / span).clamp(0.0, 1.0)
    if abs(float(gamma) - 1.0) > 1e-6:
        y = torch.pow(y, 1.0 / float(gamma))
    return y


class Layer13PSLevels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "输入黑场": ("INT", {"default": 0, "min": 0, "max": 254, "step": 1}),
                "输入白场": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1}),
                "中间调": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "输出黑场": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "输出白场": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "模式": (["保持颜色(按亮度)", "分别处理RGB"], {"default": "保持颜色(按亮度)"}),
                "允许反相": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Layer13/调色"

    def apply(
        self,
        image: torch.Tensor,
        输入黑场: int = 0,
        输入白场: int = 255,
        中间调: float = 1.0,
        输出黑场: int = 0,
        输出白场: int = 255,
        模式: str = "保持颜色(按亮度)",
        允许反相: bool = False,
    ):
        black_point = int(输入黑场)
        white_point = int(输入白场)
        gamma = float(中间调)
        output_black = int(输出黑场)
        output_white = int(输出白场)
        mode = str(模式)

        if white_point <= black_point:
            low = min(black_point, white_point)
            high = max(black_point, white_point)
            black_point = low
            white_point = min(255, high if high > low else low + 1)

        if not bool(允许反相) and output_black > output_white:
            output_black, output_white = output_white, output_black

        out_black = float(output_black) / 255.0
        out_white = float(output_white) / 255.0

        img = image.clone().to(dtype=torch.float32)

        if mode == "分别处理RGB":
            out = _levels_curve(img, black_point, white_point, gamma)
            out = out_black + out * (out_white - out_black)
            out = torch.clamp(out, 0.0, 1.0)
            return (out,)

        luma = (
            img[..., 0] * 0.2126
            + img[..., 1] * 0.7152
            + img[..., 2] * 0.0722
        )
        new_luma = _levels_curve(luma, black_point, white_point, gamma)
        new_luma = out_black + new_luma * (out_white - out_black)
        new_luma = torch.clamp(new_luma, 0.0, 1.0)

        scale = (new_luma / torch.clamp(luma, min=1e-6)).unsqueeze(-1)
        out = torch.clamp(img * scale, 0.0, 1.0)

        near_zero_luma = (luma <= 1e-6).unsqueeze(-1)
        lifted_floor = new_luma.unsqueeze(-1).expand_as(out)
        out = torch.where(near_zero_luma, lifted_floor, out)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "Layer13PSLevels": Layer13PSLevels,
    "Layer13ForceBlackLevel": Layer13PSLevels,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13PSLevels": "Layer13 色阶（PS风格）",
    "Layer13ForceBlackLevel": "Layer13 色阶（兼容旧版）",
}
