import numpy as np
import torch


class Layer13HistogramLimit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "黑场目标": ("INT", {"default": 253, "min": 0, "max": 255}),
                "白场目标": ("INT", {"default": 2, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "处理"
    CATEGORY = "Layer13"

    def 处理(self, 图像, 黑场目标=253, 白场目标=2):
        original_dtype = 图像.dtype
        original_is_float = torch.is_floating_point(图像)
        image_np = 图像.detach().cpu().numpy().astype(np.float32, copy=True)

        # ComfyUI IMAGE 通常是 0..1 的 float，这里统一转到 0..255 做直方图端点映射。
        normalize_back = original_is_float and float(image_np.min()) >= 0.0 and float(image_np.max()) <= 1.0
        if normalize_back:
            working = image_np * 255.0
        else:
            working = image_np

        old_min = float(working.min())
        old_max = float(working.max())

        if old_max == old_min:
            return (图像,)

        # 强制保持正常明暗顺序，避免目标值交叉时出现反相效果。
        target_min = float(min(黑场目标, 白场目标))
        target_max = float(max(黑场目标, 白场目标))

        mapped = (working - old_min) / (old_max - old_min)
        mapped = mapped * (target_max - target_min) + target_min
        mapped = np.clip(mapped, 0.0, 255.0).astype(np.float32, copy=False)

        if normalize_back:
            mapped = mapped / 255.0

        output = torch.from_numpy(mapped).to(device=图像.device)
        if output.dtype != original_dtype:
            output = output.to(dtype=original_dtype)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Layer13HistogramLimit": Layer13HistogramLimit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13HistogramLimit": "Layer13直方图限制",
}
