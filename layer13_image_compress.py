from io import BytesIO

import numpy as np
import torch
from PIL import Image


def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError("图像必须是 ComfyUI IMAGE 张量。")
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("图像必须是 IMAGE 批量张量 (B,H,W,C)，且至少包含 RGB 三通道。")
    return image


def _parse_color(value: str) -> tuple[int, int, int]:
    text = str(value or "#ffffff").strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        return (255, 255, 255)
    try:
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (255, 255, 255)


def _tensor_to_pil(sample: torch.Tensor, background: tuple[int, int, int]) -> Image.Image:
    data = sample.detach().float().cpu().clamp(0.0, 1.0)
    array = (data.numpy() * 255.0 + 0.5).astype("uint8")

    if array.shape[-1] >= 4:
        rgba = Image.fromarray(array[..., :4], "RGBA")
        base = Image.new("RGBA", rgba.size, background + (255,))
        return Image.alpha_composite(base, rgba).convert("RGB")

    return Image.fromarray(array[..., :3], "RGB")


def _pil_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = image.convert("RGB")
    data = torch.from_numpy(np.asarray(image).copy()).to(device=device, dtype=torch.float32)
    data = data / 255.0
    return data.to(dtype=dtype)


def _jpeg_subsampling(value: str):
    if value == "4:4:4":
        return 0
    if value == "4:2:0":
        return 2
    return -1


def _encode_decode_once(
    image: Image.Image,
    fmt: str,
    quality: int,
    subsampling: str,
    keep_metadata: bool,
) -> tuple[Image.Image, int]:
    buffer = BytesIO()
    fmt = fmt.upper()
    save_kwargs = {}

    if fmt == "JPEG":
        save_kwargs.update(
            {
                "quality": int(quality),
                "subsampling": _jpeg_subsampling(subsampling),
                "optimize": True,
            }
        )
    elif fmt == "WEBP":
        save_kwargs.update({"quality": int(quality), "method": 4})
    elif fmt == "PNG":
        compress_level = round((100 - int(quality)) / 100 * 9)
        save_kwargs.update({"compress_level": max(0, min(9, compress_level))})
    else:
        raise ValueError(f"不支持的压缩格式：{fmt}")

    if keep_metadata:
        exif = image.getexif()
        if exif:
            save_kwargs["exif"] = exif.tobytes()

    try:
        image.save(buffer, format=fmt, **save_kwargs)
    except OSError:
        save_kwargs.pop("optimize", None)
        image.save(buffer, format=fmt, **save_kwargs)

    size = buffer.tell()
    buffer.seek(0)
    decoded = Image.open(buffer)
    decoded.load()
    return decoded.convert("RGB"), size


class Layer13ImageCompress:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "格式": (["JPEG", "WEBP", "PNG"], {"default": "JPEG"}),
                "质量": ("INT", {"default": 82, "min": 1, "max": 100, "step": 1}),
                "压缩次数": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "缩放比例": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
                "保持原尺寸": ("BOOLEAN", {"default": True}),
                "JPEG色度采样": (["自动", "4:4:4", "4:2:0"], {"default": "4:2:0"}),
                "透明背景": ("STRING", {"default": "#ffffff"}),
                "保留元数据": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "压缩信息")
    FUNCTION = "处理"
    CATEGORY = "Layer13"

    def 处理(
        self,
        图像,
        格式="JPEG",
        质量=82,
        压缩次数=1,
        缩放比例=1.0,
        保持原尺寸=True,
        JPEG色度采样="4:2:0",
        透明背景="#ffffff",
        保留元数据=False,
    ):
        image = _ensure_image_batch(图像)
        device = image.device
        dtype = image.dtype
        background = _parse_color(透明背景)

        outputs = []
        info = []
        quality = max(1, min(100, int(质量)))
        repeats = max(1, min(10, int(压缩次数)))
        scale = max(0.1, min(1.0, float(缩放比例)))

        for index, sample in enumerate(image):
            pil = _tensor_to_pil(sample, background)
            original_size = pil.size
            work = pil

            if scale < 0.999:
                small_size = (
                    max(1, int(round(original_size[0] * scale))),
                    max(1, int(round(original_size[1] * scale))),
                )
                work = work.resize(small_size, Image.Resampling.LANCZOS)

            last_size = 0
            for _ in range(repeats):
                work, last_size = _encode_decode_once(
                    work,
                    str(格式),
                    quality,
                    str(JPEG色度采样),
                    bool(保留元数据),
                )

            if bool(保持原尺寸) and work.size != original_size:
                work = work.resize(original_size, Image.Resampling.BICUBIC)

            outputs.append(_pil_to_tensor(work, device, dtype))
            info.append(
                f"{index}: {str(格式).upper()} q={quality} 次数={repeats} "
                f"尺寸={work.size[0]}x{work.size[1]} 估算={last_size / 1024:.1f}KB"
            )

        return (torch.stack(outputs, dim=0), "\n".join(info))


NODE_CLASS_MAPPINGS = {
    "Layer13ImageCompress": Layer13ImageCompress,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13ImageCompress": "Layer13图片压缩",
}
