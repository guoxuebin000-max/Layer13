import torch


class Layer13GridSplitToBatch:
    ORDER_OPTIONS = ["按行(1->N)", "按列(1->N)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "网格图": ("IMAGE",),
                "列数": ("INT", {"default": 3, "min": 1, "max": 256}),
                "行数": ("INT", {"default": 3, "min": 1, "max": 256}),
                "数量": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "顺序": (cls.ORDER_OPTIONS, {"default": "按行(1->N)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("图像批次", "实际数量")
    FUNCTION = "split_grid"
    CATEGORY = "Layer13"

    @staticmethod
    def _to_bhwc(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            return image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("网格图必须是 IMAGE 张量(B,H,W,C)。")
        return image

    def split_grid(
        self,
        网格图: torch.Tensor,
        列数: int = 3,
        行数: int = 3,
        数量: int = 0,
        顺序: str = "按行(1->N)",
    ):
        batch = self._to_bhwc(网格图)
        cols = max(1, int(列数))
        rows = max(1, int(行数))
        max_count = int(数量)

        all_cells = []
        for b in range(batch.shape[0]):
            img = batch[b : b + 1]  # keep batch dim for concat
            h = int(img.shape[1])
            w = int(img.shape[2])

            tile_h = h // rows
            tile_w = w // cols
            if tile_h <= 0 or tile_w <= 0:
                raise ValueError(f"网格尺寸过小: H={h}, W={w}, 行={rows}, 列={cols}")

            cells = []
            if 顺序 == "按列(1->N)":
                for c in range(cols):
                    for r in range(rows):
                        y0 = r * tile_h
                        x0 = c * tile_w
                        cells.append(img[:, y0 : y0 + tile_h, x0 : x0 + tile_w, :])
            else:
                for r in range(rows):
                    for c in range(cols):
                        y0 = r * tile_h
                        x0 = c * tile_w
                        cells.append(img[:, y0 : y0 + tile_h, x0 : x0 + tile_w, :])

            if max_count > 0:
                cells = cells[: max_count]

            all_cells.extend(cells)

        if not all_cells:
            raise ValueError("拆分后无图像输出，请检查行列和数量。")

        out = torch.cat(all_cells, dim=0)
        return (out, int(out.shape[0]))


NODE_CLASS_MAPPINGS = {
    "Layer13GridSplitToBatch": Layer13GridSplitToBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13GridSplitToBatch": "Layer13网格拆分批次",
}
