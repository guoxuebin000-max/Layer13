import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class Layer13ScaleByLongShortEdge:
    # 长短比：始终表示 长边:短边
    RATIO_OPTIONS = ["原图", "自定义", "1:1", "3:2", "4:3", "16:9", "21:9"]
    FIT_OPTIONS = ["适应", "裁剪", "填充"]
    METHOD_OPTIONS = ["lanczos", "bicubic", "hamming", "bilinear", "box", "nearest"]
    ROUND_OPTIONS = ["8", "16", "32", "64", "128", "256", "512", "None"]
    SCALE_TO_OPTIONS = ["不缩放", "长边", "短边", "宽度", "高度", "总像素(kpx)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "长短比": (cls.RATIO_OPTIONS, {"default": "原图"}),
                "比例长边": ("INT", {"default": 1, "min": 1, "max": 100000000, "step": 1}),
                "比例短边": ("INT", {"default": 1, "min": 1, "max": 100000000, "step": 1}),
                "适应": (cls.FIT_OPTIONS, {"default": "适应"}),
                "方法": (cls.METHOD_OPTIONS, {"default": "lanczos"}),
                "四舍五入到倍数": (cls.ROUND_OPTIONS, {"default": "8"}),
                "缩放到": (cls.SCALE_TO_OPTIONS, {"default": "长边"}),
                "缩放长度": ("INT", {"default": 1024, "min": 4, "max": 100000000, "step": 1}),
                "背景颜色": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("图像", "宽", "高")
    FUNCTION = "处理"
    CATEGORY = "Layer13"

    @staticmethod
    def _round_to_nearest_multiple(value: int, multiple: int) -> int:
        if multiple <= 1:
            return max(1, int(value))
        rounded = int(round(float(value) / float(multiple))) * multiple
        return max(multiple, rounded)

    @staticmethod
    def _parse_long_short_ratio(ratio_text: str, prop_long: int, prop_short: int, orig_w: int, orig_h: int) -> float:
        if ratio_text == "原图":
            src_long = max(orig_w, orig_h)
            src_short = max(1, min(orig_w, orig_h))
            return float(src_long) / float(src_short)

        if ratio_text == "自定义":
            a = float(prop_long)
            b = float(prop_short)
        else:
            if ":" not in ratio_text:
                raise ValueError(f"长短比格式无效: {ratio_text}")
            a_text, b_text = ratio_text.split(":", 1)
            a = float(a_text)
            b = float(b_text)

        if a <= 0 or b <= 0:
            raise ValueError(f"长短比必须大于 0: {ratio_text}")
        return float(max(a, b)) / float(min(a, b))

    @staticmethod
    def _calc_target_size(
        orig_w: int,
        orig_h: int,
        ratio_long_short: float,
        scale_to: str,
        scale_length: int,
    ):
        src_is_landscape = orig_w >= orig_h
        src_long = max(orig_w, orig_h)
        src_short = min(orig_w, orig_h)
        ratio = max(1e-8, float(ratio_long_short))

        if scale_to == "长边":
            long_edge = max(1, int(scale_length))
            short_edge = max(1, int(round(float(long_edge) / ratio)))
        elif scale_to == "短边":
            short_edge = max(1, int(scale_length))
            long_edge = max(1, int(round(float(short_edge) * ratio)))
        elif scale_to == "宽度":
            width = max(1, int(scale_length))
            if src_is_landscape:
                long_edge = width
                short_edge = max(1, int(round(float(long_edge) / ratio)))
            else:
                short_edge = width
                long_edge = max(1, int(round(float(short_edge) * ratio)))
        elif scale_to == "高度":
            height = max(1, int(scale_length))
            if src_is_landscape:
                short_edge = height
                long_edge = max(1, int(round(float(short_edge) * ratio)))
            else:
                long_edge = height
                short_edge = max(1, int(round(float(long_edge) / ratio)))
        elif scale_to == "总像素(kpx)":
            area = max(1, int(scale_length)) * 1000.0
            long_edge = max(1, int(round(math.sqrt(area * ratio))))
            short_edge = max(1, int(round(float(long_edge) / ratio)))
        else:
            long_edge = max(1, int(src_long))
            short_edge = max(1, int(round(float(long_edge) / ratio)))

        if src_is_landscape:
            return int(long_edge), int(short_edge)
        return int(short_edge), int(long_edge)

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
    def _pil_sampler(method: str):
        if method == "bicubic":
            return Image.BICUBIC
        if method == "hamming":
            return Image.HAMMING
        if method == "bilinear":
            return Image.BILINEAR
        if method == "box":
            return Image.BOX
        if method == "nearest":
            return Image.NEAREST
        return Image.LANCZOS

    @staticmethod
    def _fit_resize_image(image: Image.Image, target_w: int, target_h: int, fit_mode: str, sampler, background_color: str) -> Image.Image:
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        if orig_w <= 0 or orig_h <= 0:
            raise ValueError("输入图像尺寸无效。")

        src_ratio = float(orig_w) / float(orig_h)
        dst_ratio = float(target_w) / float(target_h)

        if fit_mode == "填充":
            return image.resize((target_w, target_h), sampler)

        if fit_mode == "裁剪":
            if src_ratio > dst_ratio:
                crop_w = max(1, int(round(orig_h * dst_ratio)))
                left = (orig_w - crop_w) // 2
                crop_box = (left, 0, left + crop_w, orig_h)
            else:
                crop_h = max(1, int(round(orig_w / dst_ratio)))
                top = (orig_h - crop_h) // 2
                crop_box = (0, top, orig_w, top + crop_h)
            return image.crop(crop_box).resize((target_w, target_h), sampler)

        # 适应（letterbox）
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
    def _interp_mode(method: str) -> str:
        if method == "nearest":
            return "nearest"
        if method in {"bilinear", "hamming"}:
            return "bilinear"
        if method == "box":
            return "area"
        # lanczos / bicubic 使用 bicubic 快路径
        return "bicubic"

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
    def _interp_tensor(cls, x_nchw: torch.Tensor, out_h: int, out_w: int, method: str) -> torch.Tensor:
        mode = cls._interp_mode(method)
        if mode in {"bilinear", "bicubic"}:
            return F.interpolate(x_nchw, size=(out_h, out_w), mode=mode, align_corners=False, antialias=True)
        return F.interpolate(x_nchw, size=(out_h, out_w), mode=mode)

    def 处理(
        self,
        图像: torch.Tensor,
        长短比: str = "原图",
        比例长边: int = 1,
        比例短边: int = 1,
        适应: str = "适应",
        方法: str = "lanczos",
        四舍五入到倍数: str = "8",
        缩放到: str = "长边",
        缩放长度: int = 1024,
        背景颜色: str = "#000000",
    ):
        if 图像.ndim != 4:
            raise ValueError("输入图像必须是 IMAGE 批量张量 (B,H,W,C)。")

        batch_size, orig_h, orig_w, _ = 图像.shape
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError("输入图像尺寸无效。")

        image_min = float(torch.min(图像).item())
        image_max = float(torch.max(图像).item())
        normalized_input = torch.is_floating_point(图像) and image_min >= 0.0 and image_max <= 1.0

        ratio_long_short = self._parse_long_short_ratio(长短比, 比例长边, 比例短边, orig_w, orig_h)
        target_w, target_h = self._calc_target_size(orig_w, orig_h, ratio_long_short, 缩放到, 缩放长度)

        if 四舍五入到倍数 != "None":
            multiple = int(四舍五入到倍数)
            target_w = self._round_to_nearest_multiple(target_w, multiple)
            target_h = self._round_to_nearest_multiple(target_h, multiple)

        if target_w == orig_w and target_h == orig_h:
            return (图像, int(orig_w), int(orig_h))

        # 统一到 0..1 的 float32 NCHW，使用 torch 快路径批量缩放。
        if normalized_input:
            work = 图像.to(dtype=torch.float32)
        else:
            work = torch.clamp(图像.to(dtype=torch.float32), 0.0, 255.0) / 255.0
        x = work.permute(0, 3, 1, 2).contiguous()

        src_ratio = float(orig_w) / float(orig_h)
        dst_ratio = float(target_w) / float(target_h)

        if 适应 == "填充":
            y = self._interp_tensor(x, target_h, target_w, 方法)
        elif 适应 == "裁剪":
            if src_ratio > dst_ratio:
                crop_w = max(1, int(round(orig_h * dst_ratio)))
                x0 = (orig_w - crop_w) // 2
                cropped = x[:, :, :, x0 : x0 + crop_w]
            else:
                crop_h = max(1, int(round(orig_w / dst_ratio)))
                y0 = (orig_h - crop_h) // 2
                cropped = x[:, :, y0 : y0 + crop_h, :]
            y = self._interp_tensor(cropped, target_h, target_w, 方法)
        else:
            # 适应（letterbox）
            if src_ratio > dst_ratio:
                fit_w = target_w
                fit_h = max(1, int(round(target_w / src_ratio)))
            else:
                fit_h = target_h
                fit_w = max(1, int(round(target_h * src_ratio)))
            resized = self._interp_tensor(x, fit_h, fit_w, 方法)

            r, g, b = self._parse_rgb(背景颜色, fallback=(0, 0, 0))
            canvas = torch.empty((batch_size, 3, target_h, target_w), dtype=resized.dtype, device=resized.device)
            canvas[:, 0, :, :] = r / 255.0
            canvas[:, 1, :, :] = g / 255.0
            canvas[:, 2, :, :] = b / 255.0
            paste_x = (target_w - fit_w) // 2
            paste_y = (target_h - fit_h) // 2
            canvas[:, :, paste_y : paste_y + fit_h, paste_x : paste_x + fit_w] = resized
            y = canvas

        output = y.permute(0, 2, 3, 1).contiguous()
        if not normalized_input:
            output = torch.clamp(output * 255.0, 0.0, 255.0)
        if output.dtype != 图像.dtype:
            output = output.to(dtype=图像.dtype)
        return (output, int(target_w), int(target_h))


NODE_CLASS_MAPPINGS = {
    "Layer13ScaleByLongShortEdge": Layer13ScaleByLongShortEdge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13ScaleByLongShortEdge": "Layer13长短边缩放",
}
