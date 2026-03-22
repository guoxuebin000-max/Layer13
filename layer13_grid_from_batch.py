import math
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


class Layer13GridFromBatch:
    SIZE_REF_OPTIONS = ["首图", "最小宽高", "最大宽高", "自定义"]
    FIT_OPTIONS = ["适应(补边)", "裁剪填满", "拉伸"]
    METHOD_OPTIONS = ["lanczos", "bicubic", "bilinear", "hamming", "box", "nearest"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像批次": ("IMAGE",),
                "批次数量限制": ("INT", {"default": 0, "min": 0, "max": 65535, "step": 1}),
                "列数": ("INT", {"default": 0, "min": 0, "max": 256}),
                "行数": ("INT", {"default": 0, "min": 0, "max": 256}),
                "尺寸基准": (cls.SIZE_REF_OPTIONS, {"default": "最小宽高"}),
                "宽度": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "高度": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "适应方式": (cls.FIT_OPTIONS, {"default": "裁剪填满"}),
                "缩放方法": (cls.METHOD_OPTIONS, {"default": "lanczos"}),
                "背景颜色": ("STRING", {"default": "#000000"}),
                "边框像素": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "边框颜色": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("网格图", "列数", "行数", "数量")
    FUNCTION = "build_grid"
    CATEGORY = "Layer13"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, False, False, False)

    @staticmethod
    def _to_bhwc(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            return image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("图像批次必须是 IMAGE 张量(B,H,W,C)。")
        return image

    @staticmethod
    def _pil_sampler(method: str):
        if method == "bicubic":
            return Image.BICUBIC
        if method == "bilinear":
            return Image.BILINEAR
        if method == "hamming":
            return Image.HAMMING
        if method == "box":
            return Image.BOX
        if method == "nearest":
            return Image.NEAREST
        return Image.LANCZOS

    @staticmethod
    def _to_pil_rgb(image_tensor: torch.Tensor, normalized_input: bool) -> Image.Image:
        arr = image_tensor.detach().cpu().numpy()
        if normalized_input:
            arr = np.clip(arr * 255.0, 0.0, 255.0)
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8, copy=False)
        return Image.fromarray(arr, mode="RGB")

    @staticmethod
    def _pil_to_tensor(image_pil: Image.Image, dtype: torch.dtype, device: torch.device, normalized_output: bool) -> torch.Tensor:
        arr = np.asarray(image_pil, dtype=np.float32)
        if normalized_output:
            arr = arr / 255.0
        out = torch.from_numpy(arr).to(device=device)
        if out.dtype != dtype:
            out = out.to(dtype=dtype)
        return out

    @staticmethod
    def _fit_resize(image: Image.Image, target_w: int, target_h: int, fit_mode: str, sampler, background_color: str) -> Image.Image:
        image = image.convert("RGB")
        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError("检测到无效图像尺寸。")

        if fit_mode == "拉伸":
            return image.resize((target_w, target_h), sampler)

        src_ratio = float(src_w) / float(src_h)
        dst_ratio = float(target_w) / float(target_h)

        if fit_mode == "裁剪填满":
            if src_ratio > dst_ratio:
                crop_w = max(1, int(round(src_h * dst_ratio)))
                left = (src_w - crop_w) // 2
                cropped = image.crop((left, 0, left + crop_w, src_h))
            else:
                crop_h = max(1, int(round(src_w / dst_ratio)))
                top = (src_h - crop_h) // 2
                cropped = image.crop((0, top, src_w, top + crop_h))
            return cropped.resize((target_w, target_h), sampler)

        if src_ratio > dst_ratio:
            fit_w = target_w
            fit_h = max(1, int(round(target_w / src_ratio)))
        else:
            fit_h = target_h
            fit_w = max(1, int(round(target_h * src_ratio)))
        resized = image.resize((fit_w, fit_h), sampler)
        canvas = Image.new("RGB", (target_w, target_h), color=background_color)
        paste_x = (target_w - fit_w) // 2
        paste_y = (target_h - fit_h) // 2
        canvas.paste(resized, box=(paste_x, paste_y))
        return canvas

    @staticmethod
    def _add_border(image: Image.Image, border_px: int, border_color: str) -> Image.Image:
        if border_px <= 0:
            return image
        return ImageOps.expand(image, border=border_px, fill=border_color)

    @staticmethod
    def _resolve_target_size(images: List[torch.Tensor], size_ref: str, custom_w: int, custom_h: int) -> Tuple[int, int]:
        # images: List[(H,W,C)]
        widths = [int(img.shape[1]) for img in images]
        heights = [int(img.shape[0]) for img in images]
        if size_ref == "首图":
            return widths[0], heights[0]
        if size_ref == "最小宽高":
            return min(widths), min(heights)
        if size_ref == "最大宽高":
            return max(widths), max(heights)
        return int(custom_w), int(custom_h)

    @staticmethod
    def _pick_last(value, default=None):
        if isinstance(value, list):
            if len(value) == 0:
                return default
            return value[-1]
        return value

    @staticmethod
    def _parse_rgb(color_text: str, fallback=(0, 0, 0)):
        s = str(color_text or "").strip()
        if s.startswith("#"):
            s = s[1:]
        try:
            if len(s) == 6:
                return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
            if len(s) == 3:
                return (int(s[0] * 2, 16), int(s[1] * 2, 16), int(s[2] * 2, 16))
        except Exception:
            pass
        return fallback

    @classmethod
    def _make_blank_tensor(cls, width: int, height: int, color_text: str, dtype: torch.dtype, device: torch.device, normalized: bool):
        r, g, b = cls._parse_rgb(color_text, fallback=(0, 0, 0))
        if normalized:
            rv, gv, bv = r / 255.0, g / 255.0, b / 255.0
        else:
            rv, gv, bv = float(r), float(g), float(b)
        blank = torch.empty((height, width, 3), dtype=dtype, device=device)
        blank[..., 0] = rv
        blank[..., 1] = gv
        blank[..., 2] = bv
        return blank

    @classmethod
    def _collect_images(cls, images_input) -> List[torch.Tensor]:
        # Flatten all list/batch forms into List[(H,W,C)]
        flat: List[torch.Tensor] = []

        def visit(item):
            if isinstance(item, torch.Tensor):
                batch = cls._to_bhwc(item)
                for i in range(int(batch.shape[0])):
                    flat.append(batch[i])
                return
            if isinstance(item, (list, tuple)):
                for sub in item:
                    visit(sub)
                return
            if item is None:
                return
            raise ValueError(f"不支持的图像输入类型: {type(item)}")

        visit(images_input)
        if len(flat) == 0:
            raise ValueError("图像批次为空，无法拼接网格。")
        return flat

    @staticmethod
    def _calc_layout(count: int, cols: int, rows: int) -> Tuple[int, int]:
        if count <= 0:
            return 1, 1

        c = int(cols or 0)
        r = int(rows or 0)

        if c > 0 and r > 0:
            if c * r < count:
                r = int(math.ceil(count / c))
            return c, r
        if c > 0:
            r = int(math.ceil(count / c))
            return c, r
        if r > 0:
            c = int(math.ceil(count / r))
            return c, r

        c = int(math.ceil(math.sqrt(count)))
        r = int(math.ceil(count / c))
        return c, r

    def build_grid(
        self,
        图像批次,
        批次数量限制: int = 0,
        列数: int = 0,
        行数: int = 0,
        尺寸基准: str = "最小宽高",
        宽度: int = 1024,
        高度: int = 1024,
        适应方式: str = "裁剪填满",
        缩放方法: str = "lanczos",
        背景颜色: str = "#000000",
        边框像素: int = 0,
        边框颜色: str = "#000000",
    ):
        batch_limit = int(self._pick_last(批次数量限制, 0) or 0)
        cols_in = int(self._pick_last(列数, 0) or 0)
        rows_in = int(self._pick_last(行数, 0) or 0)
        size_ref = str(self._pick_last(尺寸基准, "最小宽高"))
        custom_w = int(self._pick_last(宽度, 1024) or 1024)
        custom_h = int(self._pick_last(高度, 1024) or 1024)
        fit_mode = str(self._pick_last(适应方式, "裁剪填满"))
        resize_method = str(self._pick_last(缩放方法, "lanczos"))
        bg_color = str(self._pick_last(背景颜色, "#000000"))
        border_px = int(self._pick_last(边框像素, 0) or 0)
        border_color = str(self._pick_last(边框颜色, "#000000"))

        images = self._collect_images(图像批次)
        if batch_limit > 0:
            images = images[:batch_limit]
        count = len(images)
        if count <= 0:
            raise ValueError("图像批次为空（或被批次数量限制截断为0），无法拼接网格。")
        dtype = images[0].dtype
        device = images[0].device

        cols, rows = self._calc_layout(count, cols_in, rows_in)

        target_w, target_h = self._resolve_target_size(images, size_ref, custom_w, custom_h)

        sampler = self._pil_sampler(resize_method)
        if torch.is_floating_point(images[0]):
            first = images[0]
            first_min = float(torch.min(first).item())
            first_max = float(torch.max(first).item())
            normalized_input = (first_min >= 0.0) and (first_max <= 1.0)
        else:
            normalized_input = False

        resized = []
        for image in images:
            if image.dtype != dtype or image.device != device:
                image = image.to(device=device, dtype=dtype)
            # 已满足目标尺寸且无边框时，直接复用，避免 PIL 往返转换。
            if int(image.shape[1]) == target_w and int(image.shape[0]) == target_h and border_px <= 0:
                resized.append(image)
                continue
            pil_img = self._to_pil_rgb(image, normalized_input)
            out_pil = self._fit_resize(pil_img, target_w, target_h, fit_mode, sampler, bg_color)
            out_pil = self._add_border(out_pil, border_px, border_color)
            resized.append(self._pil_to_tensor(out_pil, dtype, device, normalized_input))

        cell_w = target_w + border_px * 2
        cell_h = target_h + border_px * 2
        blank_color = border_color if border_px > 0 else bg_color
        blank = self._make_blank_tensor(cell_w, cell_h, blank_color, dtype, device, normalized_input)

        total = cols * rows
        while len(resized) < total:
            resized.append(blank.clone())

        row_tensors = []
        for r in range(rows):
            start = r * cols
            row_imgs = resized[start : start + cols]
            row = torch.cat([img.unsqueeze(0) for img in row_imgs], dim=2)
            row_tensors.append(row)

        grid = torch.cat(row_tensors, dim=1)
        # Safety: keep output as a single IMAGE (B=1) for Preview/SaveImage.
        if grid.ndim == 3:
            grid = grid.unsqueeze(0)
        if grid.shape[0] > 1:
            grid = grid[:1]
        return (grid, int(cols), int(rows), int(count))


NODE_CLASS_MAPPINGS = {
    "Layer13GridFromBatch": Layer13GridFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13GridFromBatch": "Layer13批量图自动网格",
}
