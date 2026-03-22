import fnmatch
import os
import random
import threading
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Layer13RandomLoadNImages:
    MODE_OPTIONS = ["每次随机", "固定随机(种子+索引)"]
    SIZE_REF_OPTIONS = ["首图", "最小宽高", "最大宽高", "自定义"]
    FIT_OPTIONS = ["适应(补边)", "裁剪填满", "拉伸"]
    METHOD_OPTIONS = ["lanczos", "bicubic", "bilinear", "hamming", "box", "nearest"]
    _SCAN_CACHE_LOCK = threading.Lock()
    _SCAN_CACHE: Dict[Tuple[str, str, bool, float], Tuple[str, ...]] = {}
    _SCAN_CACHE_MAX = 64
    _IMAGE_CACHE_LOCK = threading.Lock()
    _IMAGE_CACHE: Dict[str, Tuple[float, torch.Tensor]] = {}
    _IMAGE_CACHE_MAX = 256

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        mode = (kwargs.get("随机模式") or "每次随机").strip()
        if mode == "每次随机":
            return os.urandom(16).hex()
        seed = int(kwargs.get("随机种子", 0) or 0)
        loop_index = int(kwargs.get("循环索引", 0) or 0)
        count = int(kwargs.get("数量", 1) or 1)
        folder = kwargs.get("路径", "")
        pattern = kwargs.get("模式", "*.*")
        recursive = bool(kwargs.get("递归", False))
        allow_repeat = bool(kwargs.get("允许重复", False))
        return f"{folder}|{pattern}|{recursive}|{seed}|{loop_index}|{count}|{allow_repeat}"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "路径": ("STRING", {"default": ""}),
                "模式": ("STRING", {"default": "*.*"}),
                "递归": ("BOOLEAN", {"default": False}),
                "数量": ("INT", {"default": 9, "min": 1, "max": 512}),
                "随机模式": (cls.MODE_OPTIONS, {"default": "每次随机"}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "循环索引": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "允许重复": ("BOOLEAN", {"default": False}),
                "尺寸基准": (cls.SIZE_REF_OPTIONS, {"default": "首图"}),
                "宽度": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "高度": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "适应方式": (cls.FIT_OPTIONS, {"default": "适应(补边)"}),
                "缩放方法": (cls.METHOD_OPTIONS, {"default": "lanczos"}),
                "背景颜色": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("图像批次", "文件名列表", "实际数量")
    FUNCTION = "apply"
    CATEGORY = "Layer13"

    @classmethod
    def _scan_images(cls, folder: str, pattern: str, recursive: bool) -> List[str]:
        if not folder:
            return []
        folder = os.path.expanduser(folder)
        if not os.path.isdir(folder):
            return []
        patt = pattern.strip() if pattern else "*"

        abs_folder = os.path.abspath(folder)
        try:
            mtime = float(os.path.getmtime(abs_folder))
        except OSError:
            mtime = 0.0
        cache_key = (abs_folder, patt, bool(recursive), mtime)
        with cls._SCAN_CACHE_LOCK:
            cached = cls._SCAN_CACHE.get(cache_key)
            if cached is not None:
                return list(cached)

        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        files = []
        if recursive:
            for root, _, names in os.walk(folder):
                for name in names:
                    ext = os.path.splitext(name)[1].lower()
                    if ext not in exts:
                        continue
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, folder)
                    if fnmatch.fnmatch(name, patt) or fnmatch.fnmatch(rel, patt):
                        files.append(full)
        else:
            for name in os.listdir(folder):
                full = os.path.join(folder, name)
                if not os.path.isfile(full):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext not in exts:
                    continue
                if fnmatch.fnmatch(name, patt):
                    files.append(full)
        files.sort()

        with cls._SCAN_CACHE_LOCK:
            cls._SCAN_CACHE[cache_key] = tuple(files)
            if len(cls._SCAN_CACHE) > cls._SCAN_CACHE_MAX:
                # 简单清理，避免缓存无限增长。
                for idx, key in enumerate(list(cls._SCAN_CACHE.keys())):
                    if idx >= len(cls._SCAN_CACHE) - cls._SCAN_CACHE_MAX:
                        break
                    cls._SCAN_CACHE.pop(key, None)
        return files

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
    def _interp_mode(method: str) -> str:
        if method == "nearest":
            return "nearest"
        if method in {"bilinear", "hamming"}:
            return "bilinear"
        if method == "box":
            return "area"
        # lanczos / bicubic 都映射为 bicubic（GPU/torch 快路径）
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
    def _cache_put_image(cls, path: str, mtime: float, tensor_u8_hwc: torch.Tensor):
        with cls._IMAGE_CACHE_LOCK:
            cls._IMAGE_CACHE[path] = (mtime, tensor_u8_hwc)
            if len(cls._IMAGE_CACHE) > cls._IMAGE_CACHE_MAX:
                for _ in range(len(cls._IMAGE_CACHE) - cls._IMAGE_CACHE_MAX):
                    old_key = next(iter(cls._IMAGE_CACHE.keys()))
                    cls._IMAGE_CACHE.pop(old_key, None)

    @classmethod
    def _load_image_u8_cached(cls, path: str) -> torch.Tensor:
        try:
            mtime = float(os.path.getmtime(path))
        except OSError:
            mtime = 0.0

        with cls._IMAGE_CACHE_LOCK:
            cached = cls._IMAGE_CACHE.get(path)
            if cached is not None and float(cached[0]) == mtime:
                return cached[1]

        with Image.open(path) as img:
            if img.mode != "RGB":
                rgb = img.convert("RGB")
            else:
                rgb = img.copy()
        arr = np.asarray(rgb, dtype=np.uint8)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        tensor_u8 = torch.from_numpy(arr.copy())
        cls._cache_put_image(path, mtime, tensor_u8)
        return tensor_u8

    @classmethod
    def _resize_tensor_fit(
        cls,
        image_u8_hwc: torch.Tensor,
        target_w: int,
        target_h: int,
        fit_mode: str,
        method: str,
        bg_color_text: str,
    ) -> torch.Tensor:
        # 输入: uint8 HWC；输出: float32 HWC(0..1)
        src_h = int(image_u8_hwc.shape[0])
        src_w = int(image_u8_hwc.shape[1])
        if src_w <= 0 or src_h <= 0:
            raise ValueError("检测到无效图像尺寸。")

        x = image_u8_hwc.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        if src_w == target_w and src_h == target_h:
            return x.squeeze(0).permute(1, 2, 0).contiguous()

        mode = cls._interp_mode(method)

        def _interp(inp: torch.Tensor, out_h: int, out_w: int):
            if mode in {"bilinear", "bicubic"}:
                return F.interpolate(inp, size=(out_h, out_w), mode=mode, align_corners=False, antialias=True)
            return F.interpolate(inp, size=(out_h, out_w), mode=mode)

        if fit_mode == "拉伸":
            out = _interp(x, target_h, target_w)
            return out.squeeze(0).permute(1, 2, 0).contiguous()

        src_ratio = float(src_w) / float(src_h)
        dst_ratio = float(target_w) / float(target_h)

        if fit_mode == "裁剪填满":
            if src_ratio > dst_ratio:
                crop_w = max(1, int(round(src_h * dst_ratio)))
                x0 = (src_w - crop_w) // 2
                cropped = x[:, :, :, x0 : x0 + crop_w]
            else:
                crop_h = max(1, int(round(src_w / dst_ratio)))
                y0 = (src_h - crop_h) // 2
                cropped = x[:, :, y0 : y0 + crop_h, :]
            out = _interp(cropped, target_h, target_w)
            return out.squeeze(0).permute(1, 2, 0).contiguous()

        # 适应(补边)
        if src_ratio > dst_ratio:
            fit_w = target_w
            fit_h = max(1, int(round(target_w / src_ratio)))
        else:
            fit_h = target_h
            fit_w = max(1, int(round(target_h * src_ratio)))
        resized = _interp(x, fit_h, fit_w)

        r, g, b = cls._parse_rgb(bg_color_text, fallback=(0, 0, 0))
        canvas = torch.empty((1, 3, target_h, target_w), dtype=resized.dtype, device=resized.device)
        canvas[:, 0, :, :] = r / 255.0
        canvas[:, 1, :, :] = g / 255.0
        canvas[:, 2, :, :] = b / 255.0
        px = (target_w - fit_w) // 2
        py = (target_h - fit_h) // 2
        canvas[:, :, py : py + fit_h, px : px + fit_w] = resized
        return canvas.squeeze(0).permute(1, 2, 0).contiguous()

    @staticmethod
    def _resolve_target_size_tensors(images_u8: List[torch.Tensor], size_ref: str, custom_w: int, custom_h: int):
        widths = [int(img.shape[1]) for img in images_u8]
        heights = [int(img.shape[0]) for img in images_u8]
        if size_ref == "首图":
            return widths[0], heights[0]
        if size_ref == "最小宽高":
            return min(widths), min(heights)
        if size_ref == "最大宽高":
            return max(widths), max(heights)
        return int(custom_w), int(custom_h)

    def _pick_indices(self, total: int, count: int, mode: str, seed: int, loop_index: int, allow_repeat: bool):
        if total <= 0:
            return []
        if not allow_repeat and count > total:
            raise ValueError(f"可用图片仅 {total} 张，数量={count} 且未允许重复。")

        if mode == "每次随机":
            rng = random.SystemRandom()
            if allow_repeat:
                return [rng.randrange(total) for _ in range(count)]
            return rng.sample(range(total), count)

        local = random.Random((seed or 0) + (loop_index or 0))
        if allow_repeat:
            return [local.randrange(total) for _ in range(count)]
        return local.sample(range(total), count)

    def apply(
        self,
        路径: str = "",
        模式: str = "*.*",
        递归: bool = False,
        数量: int = 9,
        随机模式: str = "每次随机",
        随机种子: int = 0,
        循环索引: int = 0,
        允许重复: bool = False,
        尺寸基准: str = "首图",
        宽度: int = 1024,
        高度: int = 1024,
        适应方式: str = "适应(补边)",
        缩放方法: str = "lanczos",
        背景颜色: str = "#000000",
    ):
        files = self._scan_images(路径, 模式, 递归)
        if not files:
            raise ValueError(f"未找到图片: 路径={路径}, 模式={模式}")

        count = int(数量)
        indices = self._pick_indices(len(files), count, 随机模式, int(随机种子), int(循环索引), bool(允许重复))
        picked = [files[i] for i in indices]

        image_tensors_u8 = []
        names = []
        for path in picked:
            try:
                image_tensors_u8.append(self._load_image_u8_cached(path))
                names.append(os.path.splitext(os.path.basename(path))[0])
            except (UnidentifiedImageError, OSError):
                continue

        if not image_tensors_u8:
            raise ValueError("所选图片均无法读取。")

        target_w, target_h = self._resolve_target_size_tensors(image_tensors_u8, 尺寸基准, int(宽度), int(高度))

        resized = []
        for img_u8 in image_tensors_u8:
            resized.append(
                self._resize_tensor_fit(
                    img_u8,
                    target_w=target_w,
                    target_h=target_h,
                    fit_mode=适应方式,
                    method=缩放方法,
                    bg_color_text=背景颜色,
                )
            )

        batch = torch.stack(resized, dim=0)
        name_text = "\n".join(names)
        return (batch, name_text, int(batch.shape[0]))


NODE_CLASS_MAPPINGS = {
    "Layer13RandomLoadNImages": Layer13RandomLoadNImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13RandomLoadNImages": "Layer13随机加载N图",
}
