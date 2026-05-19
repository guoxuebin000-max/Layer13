import math

import numpy as np
import torch
from PIL import Image, ImageChops


def _image_tensor_to_pil(tensor):
    array = tensor.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

    if array.shape[-1] == 4:
        return Image.fromarray(array, "RGBA")
    return Image.fromarray(array[..., :3], "RGB").convert("RGBA")


def _mask_tensor_to_pil(mask, width, height, mode):
    array = mask.detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)

    if mode == "Comfy透明遮罩":
        array = 1.0 - array
    elif mode == "黑色可见":
        array = 1.0 - array

    alpha = (array * 255.0).astype(np.uint8)
    alpha_image = Image.fromarray(alpha, "L")
    if alpha_image.size != (width, height):
        alpha_image = alpha_image.resize((width, height), Image.Resampling.LANCZOS)
    return alpha_image


def _pil_to_image_tensor(image):
    rgb = image.convert("RGB")
    array = np.asarray(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(array)


def _resize_watermark(watermark, tile_width):
    if tile_width <= 0:
        return watermark

    ratio = tile_width / max(1, watermark.width)
    tile_height = max(1, int(round(watermark.height * ratio)))
    return watermark.resize((tile_width, tile_height), Image.Resampling.LANCZOS)


def _remove_white_background(watermark, threshold):
    rgb = watermark.convert("RGB")
    alpha = watermark.getchannel("A")

    arr = np.asarray(rgb)
    whiteish = np.min(arr, axis=2) >= threshold
    alpha_arr = np.asarray(alpha).copy()
    alpha_arr[whiteish] = 0

    out = watermark.copy()
    out.putalpha(Image.fromarray(alpha_arr, "L"))
    return out


def _apply_blend(base, overlay, blend_mode):
    if blend_mode == "正常":
        return Image.alpha_composite(base, overlay)

    base_rgb = base.convert("RGB")
    overlay_rgb = overlay.convert("RGB")
    overlay_alpha = overlay.getchannel("A")

    if blend_mode == "正片叠底":
        blended_rgb = ImageChops.multiply(base_rgb, overlay_rgb)
    elif blend_mode == "滤色":
        blended_rgb = ImageChops.screen(base_rgb, overlay_rgb)
    elif blend_mode == "叠加":
        blended_rgb = ImageChops.overlay(base_rgb, overlay_rgb)
    elif blend_mode == "柔光":
        blended_rgb = ImageChops.soft_light(base_rgb, overlay_rgb)
    else:
        blended_rgb = overlay_rgb

    blended = blended_rgb.convert("RGBA")
    blended.putalpha(overlay_alpha)
    return Image.alpha_composite(base, blended)


class Layer13FullscreenWatermark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "水印图": ("IMAGE",),
                "透明度": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "单个宽度": ("INT", {"default": 160, "min": 0, "max": 4096, "step": 1}),
                "横向间距": ("INT", {"default": 80, "min": -4096, "max": 4096, "step": 1}),
                "纵向间距": ("INT", {"default": 80, "min": -4096, "max": 4096, "step": 1}),
                "横向偏移": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "纵向偏移": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "旋转角度": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "错行排列": ("BOOLEAN", {"default": True}),
                "抠除白底": ("BOOLEAN", {"default": False}),
                "白底阈值": ("INT", {"default": 245, "min": 0, "max": 255, "step": 1}),
                "遮罩模式": (["Comfy透明遮罩", "白色可见", "黑色可见"], {"default": "Comfy透明遮罩"}),
                "混合模式": (["正常", "正片叠底", "滤色", "叠加", "柔光"], {"default": "正常"}),
            },
            "optional": {
                "水印遮罩": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "处理"
    CATEGORY = "Layer13"

    def 处理(
        self,
        图像,
        水印图,
        透明度,
        单个宽度,
        横向间距,
        纵向间距,
        横向偏移,
        纵向偏移,
        旋转角度,
        错行排列,
        抠除白底,
        白底阈值,
        遮罩模式,
        混合模式,
        水印遮罩=None,
    ):
        output_images = []
        batch_size = 图像.shape[0]
        watermark_batch = 水印图.shape[0]
        mask_batch = 水印遮罩.shape[0] if 水印遮罩 is not None else 0

        for index in range(batch_size):
            base = _image_tensor_to_pil(图像[index])

            wm_index = index % watermark_batch
            tile = _image_tensor_to_pil(水印图[wm_index])

            if 水印遮罩 is not None:
                mask_index = index % mask_batch
                alpha = _mask_tensor_to_pil(水印遮罩[mask_index], tile.width, tile.height, 遮罩模式)
                tile.putalpha(alpha)

            if 抠除白底:
                tile = _remove_white_background(tile, 白底阈值)

            tile = _resize_watermark(tile, 单个宽度)

            if abs(旋转角度) > 0.001:
                tile = tile.rotate(旋转角度, expand=True, resample=Image.Resampling.BICUBIC)

            if 透明度 < 1.0:
                alpha = tile.getchannel("A")
                alpha = alpha.point(lambda value: int(value * 透明度))
                tile.putalpha(alpha)

            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            step_x = max(1, tile.width + 横向间距)
            step_y = max(1, tile.height + 纵向间距)

            start_x = 横向偏移 - step_x
            start_y = 纵向偏移 - step_y
            rows = int(math.ceil((base.height + abs(start_y) + step_y * 2) / step_y)) + 2
            cols = int(math.ceil((base.width + abs(start_x) + step_x * 2) / step_x)) + 2

            for row in range(rows):
                row_offset = step_x // 2 if 错行排列 and row % 2 == 1 else 0
                y = start_y + row * step_y
                for col in range(cols):
                    x = start_x + col * step_x + row_offset
                    overlay.alpha_composite(tile, (x, y))

            result = _apply_blend(base, overlay, 混合模式)
            output_images.append(_pil_to_image_tensor(result))

        return (torch.stack(output_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Layer13FullscreenWatermark": Layer13FullscreenWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13FullscreenWatermark": "layer13 满屏水印",
}
