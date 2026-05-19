import os
import random
import numpy as np
from PIL import Image

import torch
import folder_paths


def _save_temp_images(images: torch.Tensor, prefix: str = "Layer13Preview"):
    output_dir = folder_paths.get_temp_directory()
    prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
    filename_prefix = prefix + prefix_append

    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
    )

    results = []
    for batch_number, image in enumerate(images):
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_with_batch_num}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file), compress_level=1)
        results.append({"filename": file, "subfolder": subfolder, "type": "temp"})
        counter += 1

    return results


class Layer13PreviewImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "preview"
    CATEGORY = "Layer13"
    OUTPUT_NODE = True

    def preview(self, 图像, prompt=None, extra_pnginfo=None):
        results = _save_temp_images(图像)
        return {"ui": {"images": results}, "result": (图像,)}


NODE_CLASS_MAPPINGS = {
    "Layer13PreviewImage": Layer13PreviewImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13PreviewImage": "Layer13预览图像(可输出)",
}
