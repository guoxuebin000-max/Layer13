from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import ComfyExtension, InputImpl, io


class Layer13UploadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])

        return io.Schema(
            node_id="Layer13UploadVideo",
            display_name="Layer13 加载视频(上传)",
            category="Layer13/视频",
            inputs=[
                io.Combo.Input("视频文件", options=sorted(files), upload=io.UploadType.video),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
            ],
        )

    @classmethod
    def execute(cls, 视频文件) -> io.NodeOutput:
        video_path = folder_paths.get_annotated_filepath(视频文件)
        return io.NodeOutput(InputImpl.VideoFromFile(video_path))

    @classmethod
    def validate_inputs(cls, 视频文件):
        if not folder_paths.exists_annotated_filepath(视频文件):
            return f"无效视频文件: {视频文件}"
        return True

    @classmethod
    def fingerprint_inputs(cls, 视频文件):
        video_path = folder_paths.get_annotated_filepath(视频文件)
        return os.path.getmtime(video_path)


class Layer13VideoUploadExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Layer13UploadVideo]


def comfy_entrypoint():
    return Layer13VideoUploadExtension()
