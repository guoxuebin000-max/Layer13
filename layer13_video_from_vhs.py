from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".gif",
    ".webp",
}


class Layer13VideoPathFromVHS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件列表": ("VHS_FILENAMES",),
            },
            "optional": {
                "图像批次": ("IMAGE",),
                "音频": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES", "STRING", "STRING", "IMAGE", "AUDIO", "BOOLEAN", "INT")
    RETURN_NAMES = (
        "文件列表",
        "视频路径",
        "首帧预览图路径",
        "透传图像批次",
        "透传音频",
        "是否保存到输出目录",
        "文件数量",
    )
    FUNCTION = "extract"
    CATEGORY = "Layer13"

    @staticmethod
    def _normalize_filenames(filenames):
        if filenames is None:
            return False, []
        if isinstance(filenames, tuple) and len(filenames) == 2:
            save_output, paths = filenames
            return bool(save_output), list(paths or [])
        if isinstance(filenames, list):
            return False, list(filenames)
        return False, []

    @staticmethod
    def _pick_video_and_preview(paths):
        video_path = ""
        preview_path = ""
        for item in paths:
            suffix = Path(item).suffix.lower()
            if suffix == ".png" and preview_path == "":
                preview_path = str(item)
            if suffix in VIDEO_EXTENSIONS and video_path == "":
                video_path = str(item)
        if video_path == "" and paths:
            video_path = str(paths[-1])
        return video_path, preview_path

    def extract(self, 文件列表, 图像批次=None, 音频=None):
        save_output, paths = self._normalize_filenames(文件列表)
        video_path, preview_path = self._pick_video_and_preview(paths)
        return (
            文件列表,
            video_path,
            preview_path,
            图像批次,
            音频,
            save_output,
            len(paths),
        )


class Layer13ExtractFirstLastFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像批次": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("首帧", "中间帧", "尾帧", "帧数", "中间帧索引")
    FUNCTION = "extract_frames"
    CATEGORY = "Layer13"

    def extract_frames(self, 图像批次):
        if 图像批次 is None:
            raise ValueError("图像批次不能为空")

        try:
            frame_count = int(图像批次.shape[0])
        except Exception as exc:
            raise ValueError("输入必须是 IMAGE 批次") from exc

        if frame_count <= 0:
            raise ValueError("图像批次里没有帧")

        first_frame = 图像批次[:1]
        middle_index = frame_count // 2
        middle_frame = 图像批次[middle_index:middle_index + 1]
        last_frame = 图像批次[-1:]
        return (first_frame, middle_frame, last_frame, frame_count, middle_index)


NODE_CLASS_MAPPINGS = {
    "Layer13VideoPathFromVHS": Layer13VideoPathFromVHS,
    "Layer13ExtractFirstLastFrame": Layer13ExtractFirstLastFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13VideoPathFromVHS": "Layer13提取VHS视频路径",
    "Layer13ExtractFirstLastFrame": "Layer13提取首中尾帧",
}
