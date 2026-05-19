from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
}


def _get_input_directory() -> Path:
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.get_input_directory())
    except Exception:
        return Path.cwd()


def _list_input_videos() -> list[str]:
    input_dir = _get_input_directory()
    if not input_dir.exists():
        return ["没有可用视频"]

    videos = [
        p.name
        for p in sorted(input_dir.iterdir(), key=lambda item: item.name.lower())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return videos or ["没有可用视频"]


def _resolve_video_path(video_name: str) -> Path:
    raw = (video_name or "").strip().strip('"').strip("'")
    if not raw or raw == "没有可用视频":
        raise ValueError("ComfyUI 的 input 目录里没有可用视频。")

    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()

    candidate = (_get_input_directory() / raw).resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"找不到视频文件: {raw}")


def _count_frames_fallback(capture: cv2.VideoCapture) -> int:
    current_pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        count += 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    return count


def _read_frame_rgb(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频: {path}")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"无法读取第 {frame_index} 帧: {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频: {path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = _count_frames_fallback(capture)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    if total_frames <= 0:
        raise ValueError(f"视频没有可读取的帧: {path}")

    return {
        "path": str(path),
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _resolve_video_input(video_input) -> dict:
    if video_input is None:
        raise ValueError("视频输入为空。")

    source = None
    if hasattr(video_input, "get_stream_source"):
        source = video_input.get_stream_source()

    path: Path | None = None
    if isinstance(source, (str, os.PathLike)):
        candidate = Path(source)
        if candidate.exists():
            path = candidate.resolve()

    if path is None:
        fd, temp_path = tempfile.mkstemp(suffix=".mp4", prefix="layer13_video_")
        os.close(fd)
        if hasattr(video_input, "save_to"):
            video_input.save_to(temp_path)
            path = Path(temp_path).resolve()
        else:
            raise ValueError("无法从 VIDEO 输入中提取视频文件。")

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"不支持的视频格式: {path.suffix}")

    info = _inspect_video(path)
    if hasattr(video_input, "get_frame_count"):
        try:
            info["total_frames"] = int(video_input.get_frame_count())
        except Exception:
            pass
    if hasattr(video_input, "get_frame_rate"):
        try:
            fps = float(video_input.get_frame_rate())
            if fps > 0:
                info["fps"] = fps
        except Exception:
            pass
    if hasattr(video_input, "get_dimensions"):
        try:
            width, height = video_input.get_dimensions()
            info["width"] = int(width)
            info["height"] = int(height)
        except Exception:
            pass

    return info


class Layer13LoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频文件": (tuple(_list_input_videos()),),
            },
            "optional": {
                "视频": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("LAYER13_VIDEO", "STRING", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("video", "video_path", "total_frames", "fps", "width", "height")
    FUNCTION = "load"
    CATEGORY = "Layer13/视频"

    def load(self, 视频文件: str, 视频=None):
        if 视频 is not None:
            info = _resolve_video_input(视频)
        else:
            path = _resolve_video_path(视频文件)
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"不支持的视频格式: {path.suffix}")
            info = _inspect_video(path)
        return (
            info,
            info["path"],
            info["total_frames"],
            info["fps"],
            info["width"],
            info["height"],
        )


class Layer13RandomVideoFrame:
    _seed_state: dict[str, int] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频": ("VIDEO",),
                "随机种子": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "生成后控制": (["随机", "固定", "递增", "递减"], {"default": "随机"}),
                "循环索引": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999999,
                        "control_after_generate": True,
                    },
                ),
                "起始帧": ("INT", {"default": 0, "min": 0, "max": 999999999}),
                "结束帧": ("INT", {"default": -1, "min": -1, "max": 999999999}),
            },
            "optional": {
                "视频对象": ("LAYER13_VIDEO",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "INT")
    RETURN_NAMES = ("image", "frame_index", "video_path", "实际种子")
    FUNCTION = "extract"
    CATEGORY = "Layer13/视频"

    @classmethod
    def _next_seed(cls, unique_id, seed: int, mode: str) -> int:
        key = str(unique_id or "__default__")
        current_seed = max(0, int(seed))
        last_seed = cls._seed_state.get(key)

        if mode == "固定":
            next_seed = current_seed
        elif mode == "递增":
            next_seed = current_seed if last_seed is None else min(last_seed + 1, 0xFFFFFFFFFFFFFFFF)
        elif mode == "递减":
            next_seed = current_seed if last_seed is None else max(last_seed - 1, 0)
        else:
            next_seed = random.SystemRandom().randint(0, 0xFFFFFFFFFFFFFFFF)

        cls._seed_state[key] = next_seed
        return next_seed

    def extract(
        self,
        视频,
        随机种子: int = 0,
        生成后控制: str = "随机",
        循环索引: int = 0,
        起始帧: int = 0,
        结束帧: int = -1,
        视频对象=None,
        视频文件: str | None = None,
        unique_id=None,
    ):
        if 视频 is not None:
            info = _resolve_video_input(视频)
            path = Path(info["path"]).resolve()
            total_frames = int(info["total_frames"])
        elif 视频对象 is not None:
            path = Path(视频对象["path"]).resolve()
            total_frames = int(视频对象["total_frames"])
        elif 视频文件 is not None:
            path = _resolve_video_path(视频文件)
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"不支持的视频格式: {path.suffix}")
            total_frames = int(_inspect_video(path)["total_frames"])
        else:
            raise ValueError("请连接 VIDEO 输入。")

        start_frame = max(0, min(int(起始帧), total_frames - 1))
        if int(结束帧) < 0:
            end_frame = total_frames - 1
        else:
            end_frame = max(0, min(int(结束帧), total_frames - 1))

        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        candidate_count = (end_frame - start_frame) + 1
        effective_seed = self._next_seed(unique_id, 随机种子, 生成后控制)
        rng = random.Random(int(effective_seed))
        base_offset = rng.randint(0, candidate_count - 1)
        loop_offset = int(循环索引) % candidate_count
        frame_index = start_frame + ((base_offset + loop_offset) % candidate_count)
        rgb_frame = _read_frame_rgb(path, frame_index)
        image = torch.from_numpy(rgb_frame.astype(np.float32) / 255.0).unsqueeze(0)

        return (image, frame_index, str(path), int(effective_seed))


NODE_CLASS_MAPPINGS = {
    "Layer13LoadVideo": Layer13LoadVideo,
    "Layer13RandomVideoFrame": Layer13RandomVideoFrame,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13LoadVideo": "Layer13 加载视频",
    "Layer13RandomVideoFrame": "Layer13 随机截取视频帧",
}
