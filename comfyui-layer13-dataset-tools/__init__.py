from .dataset_tools import (
    Layer13MergeImages,
    Layer13FaceFilterDedup,
)

NODE_CLASS_MAPPINGS = {
    "Layer13MergeImages": Layer13MergeImages,
    "Layer13FaceFilterDedup": Layer13FaceFilterDedup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13MergeImages": "Layer13合并图片批次",
    "Layer13FaceFilterDedup": "Layer13人脸相似度去重",
}

