import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import folder_paths


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _tensor_to_pil(sample: torch.Tensor) -> Image.Image:
    data = sample.detach().float().cpu().clamp(0.0, 1.0)
    array = (data.numpy() * 255.0 + 0.5).astype(np.uint8)
    if array.shape[-1] >= 4:
        return Image.fromarray(array[..., :4], "RGBA")
    return Image.fromarray(array[..., :3], "RGB")


def _normalize_format(fmt: str) -> tuple[str, str]:
    value = str(fmt or "PNG").upper()
    if value == "JPEG":
        return "JPEG", ".jpg"
    if value == "WEBP":
        return "WEBP", ".webp"
    return "PNG", ".png"


def _save_clean_image(img: Image.Image, path: Path, fmt: str, quality: int, png_compress: int):
    fmt, _ = _normalize_format(fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "PNG":
        img.save(path, format="PNG", compress_level=max(0, min(9, int(png_compress))))
        return

    if img.mode in {"RGBA", "LA"}:
        base = Image.new("RGB", img.size, (255, 255, 255))
        base.paste(img, mask=img.getchannel("A"))
        img = base
    else:
        img = img.convert("RGB")

    if fmt == "WEBP":
        img.save(path, format="WEBP", quality=max(1, min(100, int(quality))), method=4)
    else:
        img.save(
            path,
            format="JPEG",
            quality=max(1, min(100, int(quality))),
            subsampling=2,
            optimize=True,
        )


def _normalize_vhs_filenames(filenames):
    if filenames is None:
        return []
    if isinstance(filenames, tuple) and len(filenames) == 2:
        return list(filenames[1] or [])
    if isinstance(filenames, list):
        return list(filenames)
    return []


def _resolve_path(path_text: str) -> Path:
    path = Path(str(path_text or "").strip()).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path(folder_paths.get_output_directory()) / path,
        Path(folder_paths.get_input_directory()) / path,
        Path(folder_paths.get_temp_directory()) / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _target_path(source: Path, suffix: str, overwrite: bool) -> Path:
    if overwrite:
        return source
    suffix = str(suffix or "_clean")
    return source.with_name(f"{source.stem}{suffix}{source.suffix}")


def _clean_image_file(source: Path, target: Path, quality: int, png_compress: int) -> str:
    with Image.open(source) as img:
        img.load()
        fmt = "JPEG" if target.suffix.lower() in {".jpg", ".jpeg"} else target.suffix.lower().lstrip(".").upper()
        if fmt not in {"PNG", "JPEG", "WEBP"}:
            fmt = "PNG"
        if target == source:
            temp = source.with_name(f"{source.stem}.layer13_tmp{source.suffix}")
            _save_clean_image(img.copy(), temp, fmt, quality, png_compress)
            temp.replace(source)
        else:
            _save_clean_image(img.copy(), target, fmt, quality, png_compress)
    return str(target)


def _clean_video_file(source: Path, target: Path, overwrite: bool) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg，无法清理视频容器元数据。")

    actual_target = target
    if overwrite:
        actual_target = source.with_name(f"{source.stem}.layer13_tmp{source.suffix}")

    actual_target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-c",
        "copy",
        str(actual_target),
    ]
    subprocess.run(cmd, check=True)
    if overwrite:
        actual_target.replace(source)
        return str(source)
    return str(target)


class Layer13CleanMetadataSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "文件名前缀": ("STRING", {"default": "Layer13_clean"}),
                "格式": (["PNG", "JPEG", "WEBP"], {"default": "PNG"}),
                "质量": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "PNG压缩": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "保存路径")
    FUNCTION = "save"
    CATEGORY = "Layer13"
    OUTPUT_NODE = True

    def save(self, 图像, 文件名前缀="Layer13_clean", 格式="PNG", 质量=95, PNG压缩=4):
        if not isinstance(图像, torch.Tensor) or 图像.ndim != 4:
            raise ValueError("图像必须是 ComfyUI IMAGE 批量张量。")

        fmt, ext = _normalize_format(格式)
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            文件名前缀,
            output_dir,
            图像[0].shape[1],
            图像[0].shape[0],
        )

        results = []
        paths = []
        for batch_number, sample in enumerate(图像):
            img = _tensor_to_pil(sample)
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_{ext}"
            path = Path(full_output_folder) / file
            _save_clean_image(img, path, fmt, 质量, PNG压缩)
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            paths.append(str(path))
            counter += 1

        return {"ui": {"images": results}, "result": (图像, "\n".join(paths))}


class Layer13CleanFileMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件路径": ("STRING", {"default": ""}),
                "输出后缀": ("STRING", {"default": "_clean"}),
                "覆盖原文件": ("BOOLEAN", {"default": False}),
                "图片质量": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "PNG压缩": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1}),
            },
            "optional": {
                "文件列表": ("VHS_FILENAMES",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("清理后路径", "信息")
    FUNCTION = "clean"
    CATEGORY = "Layer13"

    def clean(self, 文件路径="", 输出后缀="_clean", 覆盖原文件=False, 图片质量=95, PNG压缩=4, 文件列表=None):
        paths = []
        if str(文件路径 or "").strip():
            paths.append(str(文件路径).strip())
        paths.extend(_normalize_vhs_filenames(文件列表))
        if not paths:
            raise ValueError("请输入文件路径，或连接 VHS 文件列表。")

        cleaned = []
        notes = []
        for item in paths:
            source = _resolve_path(item)
            if not source.exists():
                notes.append(f"跳过，不存在：{source}")
                continue

            suffix = source.suffix.lower()
            target = _target_path(source, 输出后缀, bool(覆盖原文件))
            if suffix in IMAGE_EXTENSIONS:
                cleaned_path = _clean_image_file(source, target, 图片质量, PNG压缩)
                cleaned.append(cleaned_path)
                notes.append(f"图片已清理：{cleaned_path}")
            elif suffix in VIDEO_EXTENSIONS:
                cleaned_path = _clean_video_file(source, target, bool(覆盖原文件))
                cleaned.append(cleaned_path)
                notes.append(f"视频已清理：{cleaned_path}")
            else:
                notes.append(f"跳过，不支持的格式：{source}")

        return ("\n".join(cleaned), "\n".join(notes))


NODE_CLASS_MAPPINGS = {
    "Layer13CleanMetadataSaveImage": Layer13CleanMetadataSaveImage,
    "Layer13CleanFileMetadata": Layer13CleanFileMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13CleanMetadataSaveImage": "Layer13元数据清理保存",
    "Layer13CleanFileMetadata": "Layer13清理文件元数据",
}
