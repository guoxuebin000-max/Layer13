import os

import folder_paths
import nodes


DEFAULT_IMAGE_NAME = "layer13_default_image.png"


def _load_image_input_types(cls):
    input_dir = folder_paths.get_input_directory()
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files = sorted(folder_paths.filter_files_content_types(files, ["image"]))
    options = {"image_upload": True}
    if DEFAULT_IMAGE_NAME in files:
        options["default"] = DEFAULT_IMAGE_NAME
    return {"required": {"image": (files, options)}}


nodes.LoadImage.INPUT_TYPES = classmethod(_load_image_input_types)

