import numpy as np
from PIL import Image
import torch
import folder_paths
import os
import random
from datetime import datetime, timedelta


class Layer13SaveJPG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "输入图像，保存为 JPG。"}),
                "质量": ("INT", {"default": 95, "min": 1, "max": 100}),
                "渐进式": ("BOOLEAN", {"default": False}),
                "文件名前缀": ("STRING", {"default": "Layer13"}),
                "子目录": ("STRING", {"default": ""}),
                "输出图像": ("BOOLEAN", {"default": False}),
                "清除元数据": ("BOOLEAN", {"default": True}),
                "注入相机数据": ("BOOLEAN", {"default": False}),
                "机型选择": (["随机混合", "iPhone", "Sony", "Canon", "Nikon", "Fujifilm"], {"default": "随机混合"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "save_jpg"
    CATEGORY = "Layer13"
    DESCRIPTION = "保存为 JPG 到 ComfyUI 输出目录。"

    def save_jpg(
        self,
        图像,
        质量=95,
        渐进式=False,
        文件名前缀="Layer13",
        子目录="",
        输出图像=False,
        清除元数据=True,
        注入相机数据=False,
        机型选择="随机混合",
    ):
        out_images = []
        results = []
        subdir = str(子目录 or "").strip()
        subdir = subdir.lstrip("/\\")
        subdir = os.path.normpath(subdir)
        if subdir in (".", "") or subdir.startswith(".."):
            subdir = ""
        save_root = os.path.join(self.output_dir, subdir) if subdir else self.output_dir
        os.makedirs(save_root, exist_ok=True)
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                文件名前缀,
                save_root,
                图像[0].shape[1],
                图像[0].shape[0],
            )
        )
        os.makedirs(full_output_folder, exist_ok=True)

        for batch_number, image in enumerate(图像):
            img = image
            i = 255.0 * img.cpu().numpy()
            arr = np.clip(i, 0, 255).astype(np.uint8)
            if arr.shape[-1] == 4:
                rgba = Image.fromarray(arr, "RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[3])
                pil_img = background
            else:
                pil_img = Image.fromarray(arr, "RGB")

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"
            save_kwargs = {"quality": int(质量), "optimize": True, "progressive": bool(渐进式)}
            exif = None
            if 注入相机数据:
                exif = _build_camera_exif(机型选择)
                save_kwargs["exif"] = exif
            elif not 清除元数据:
                exif = pil_img.getexif()
                if exif:
                    save_kwargs["exif"] = exif
            pil_img.save(os.path.join(full_output_folder, file), format="JPEG", **save_kwargs)

            if 输出图像:
                out = torch.from_numpy(np.array(pil_img)).to(image.device).float() / 255.0
                out_images.append(out)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        out_batch = 图像 if not 输出图像 else torch.stack(out_images, dim=0)
        return {"ui": {"images": results}, "result": (out_batch,)}


def _build_camera_exif(机型选择):
    presets = {
        "iPhone": [
            ("Apple", "iPhone 15 Pro", 24.0, 1.6),
            ("Apple", "iPhone 14 Pro", 24.0, 1.8),
            ("Apple", "iPhone 13 Pro", 26.0, 1.5),
        ],
        "Sony": [
            ("Sony", "ILCE-7M4", 35.0, 1.8),
            ("Sony", "ILCE-7RM5", 50.0, 1.8),
            ("Sony", "ILCE-6700", 35.0, 2.8),
        ],
        "Canon": [
            ("Canon", "EOS R5", 50.0, 1.8),
            ("Canon", "EOS R6", 35.0, 1.8),
            ("Canon", "EOS R8", 35.0, 2.0),
        ],
        "Nikon": [
            ("Nikon", "Z 7_2", 50.0, 1.8),
            ("Nikon", "Z 6_2", 35.0, 1.8),
            ("Nikon", "Z f", 40.0, 2.0),
        ],
        "Fujifilm": [
            ("FUJIFILM", "X-T5", 35.0, 1.8),
            ("FUJIFILM", "X-S20", 33.0, 1.4),
            ("FUJIFILM", "X100V", 23.0, 2.0),
        ],
    }

    if 机型选择 == "随机混合":
        brand = random.choice(list(presets.keys()))
    else:
        brand = 机型选择
    make, model, focal_length, f_number = random.choice(presets[brand])

    iso = random.choice([100, 160, 200, 320, 400, 640, 800, 1250, 1600])
    shutter = random.choice([1 / 30, 1 / 60, 1 / 80, 1 / 100, 1 / 125, 1 / 160, 1 / 200])
    shutter_num = int(round(shutter * 10000))
    shutter_den = 10000

    now = datetime.now() - timedelta(days=random.randint(0, 365))
    dt = now.strftime("%Y:%m:%d %H:%M:%S")

    exif = Image.Exif()
    exif[271] = make  # Make
    exif[272] = model  # Model
    exif[305] = "Layer13"  # Software
    exif[306] = dt  # DateTime
    exif[36867] = dt  # DateTimeOriginal
    exif[36868] = dt  # DateTimeDigitized
    exif[34855] = iso  # ISOSpeedRatings
    exif[33434] = (shutter_num, shutter_den)  # ExposureTime
    exif[33437] = (int(f_number * 100), 100)  # FNumber
    exif[34850] = random.choice([1, 3])  # ExposureProgram
    exif[37383] = 5  # MeteringMode: Multi-segment
    exif[37384] = 0  # LightSource: Unknown
    exif[37385] = 0  # Flash: No flash
    exif[37386] = (int(focal_length * 10), 10)  # FocalLength
    exif[40961] = 1  # ColorSpace: sRGB
    exif[41986] = 0  # ExposureMode: Auto
    exif[41987] = 0  # WhiteBalance: Auto
    return exif.tobytes()


NODE_CLASS_MAPPINGS = {
    "Layer13SaveJPG": Layer13SaveJPG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13SaveJPG": "Layer13保存JPG",
}
