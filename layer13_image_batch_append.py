import torch


class Layer13ImageBatchAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "当前图像": ("IMAGE",),
            },
            "optional": {
                "已有批次": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("图像批次", "数量")
    FUNCTION = "append_batch"
    CATEGORY = "Layer13"

    @staticmethod
    def _to_bhwc(image):
        if image is None:
            return None
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"不支持的图像类型: {type(image)}")
        if image.ndim == 3:
            return image.unsqueeze(0)
        if image.ndim == 4:
            return image
        raise ValueError("图像必须是 IMAGE 张量(B,H,W,C) 或 (H,W,C)。")

    @staticmethod
    def _match_size(batch: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if batch.shape[1:] == image.shape[1:]:
            return image

        # Use bilinear resize on NHWC -> NCHW -> NHWC to keep the node robust
        image_nchw = image.permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            image_nchw,
            size=(int(batch.shape[1]), int(batch.shape[2])),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1).to(dtype=batch.dtype, device=batch.device)

    def append_batch(self, 当前图像, 已有批次=None):
        current = self._to_bhwc(当前图像)
        existing = self._to_bhwc(已有批次)

        if current is None or current.shape[0] == 0:
            raise ValueError("当前图像为空，无法追加到批次。")

        if existing is None or existing.shape[0] == 0:
            out = current
        else:
            current = current.to(device=existing.device, dtype=existing.dtype)
            current = self._match_size(existing, current)
            out = torch.cat([existing, current], dim=0)

        return (out, int(out.shape[0]))


NODE_CLASS_MAPPINGS = {
    "Layer13ImageBatchAppend": Layer13ImageBatchAppend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13ImageBatchAppend": "Layer13图片批次追加",
}
