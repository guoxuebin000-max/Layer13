import fnmatch
import os
import random
import time
from collections import deque
from typing import List

import numpy as np
from PIL import Image, UnidentifiedImageError, ImageFile
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

_LIST_CACHE_TTL_SEC = 2.0
_PRECHECK_SIZE = 96
_list_cache = {}
_bad_image_cache = {}
_precheck_cache = {}


def _safe_file_signature(path: str):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _is_known_bad(path: str) -> bool:
    sig = _safe_file_signature(path)
    if sig is None:
        return False
    return _bad_image_cache.get(path) == sig


def _mark_bad(path: str):
    sig = _safe_file_signature(path)
    if sig is not None:
        _bad_image_cache[path] = sig


def _clear_bad(path: str):
    _bad_image_cache.pop(path, None)


def _thumbnail_precheck(path: str) -> bool:
    sig = _safe_file_signature(path)
    if sig is None:
        return False
    cached = _precheck_cache.get(path)
    if cached and cached["sig"] == sig:
        return bool(cached["ok"])
    try:
        with Image.open(path) as img:
            try:
                img.draft("RGB", (_PRECHECK_SIZE, _PRECHECK_SIZE))
            except Exception:
                pass
            try:
                img.thumbnail((_PRECHECK_SIZE, _PRECHECK_SIZE))
            except Exception:
                pass
            img.load()
        _precheck_cache[path] = {"sig": sig, "ok": True}
        return True
    except (UnidentifiedImageError, OSError):
        _precheck_cache[path] = {"sig": sig, "ok": False}
        _mark_bad(path)
        return False


def _scan_images(folder: str, pattern: str, recursive: bool) -> List[str]:
    if not folder:
        return []
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        return []
    patt = pattern.strip() if pattern else "*"
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = []
    if recursive:
        pending = [folder]
        while pending:
            current = pending.pop()
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext not in exts:
                            continue
                        rel = os.path.relpath(entry.path, folder)
                        if fnmatch.fnmatch(entry.name, patt) or fnmatch.fnmatch(rel, patt):
                            files.append(entry.path)
            except OSError:
                continue
    else:
        try:
            with os.scandir(folder) as it:
                for entry in it:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in exts:
                        continue
                    if fnmatch.fnmatch(entry.name, patt):
                        files.append(entry.path)
        except OSError:
            return []
    files.sort()
    return files


def _list_images(folder: str, pattern: str, recursive: bool) -> List[str]:
    if not folder:
        return []
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        return []
    key = (folder, pattern or "*", bool(recursive))
    now = time.monotonic()
    root_sig = _safe_file_signature(folder)
    cached = _list_cache.get(key)
    if cached and cached["root_sig"] == root_sig and now < cached["expires_at"]:
        files = cached["files"]
    else:
        files = _scan_images(folder, pattern, recursive)
        _list_cache[key] = {
            "root_sig": root_sig,
            "expires_at": now + _LIST_CACHE_TTL_SEC,
            "files": files,
        }
    usable = [f for f in files if not _is_known_bad(f)]
    return usable or files


def _pick_index(count: int, mode: str, seed: int, loop_index: int) -> int:
    if count <= 0:
        return -1
    if mode == "每次随机":
        rng = random.SystemRandom()
        return rng.randrange(count)
    if mode == "增量":
        return (loop_index or 0) % count
    # 固定随机(种子+索引)
    local = random.Random((seed or 0) + (loop_index or 0))
    return local.randrange(count)


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    # [H,W,3] -> [1,H,W,3]
    return torch.from_numpy(arr)[None, ...]


def _empty_image() -> torch.Tensor:
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def _resolve_mode(kwargs) -> str:
    # Button priority: 增量 > 洗牌 > 固定随机 > 下拉/文本 > 每次随机
    if bool(kwargs.get("增量模式", False)):
        return "增量"
    if bool(kwargs.get("洗牌模式", False)):
        return "每轮洗牌(不重复)"
    if bool(kwargs.get("固定随机模式", False)):
        return "固定随机(种子+索引)"
    mode_text = (kwargs.get("随机模式") or "").strip()
    if mode_text:
        # 兼容旧字符串
        if mode_text in ("固定随机", "固定随机模式"):
            return "固定随机(种子+索引)"
        if mode_text in ("洗牌", "每轮洗牌", "不重复"):
            return "每轮洗牌(不重复)"
        if mode_text in ("递增", "增量模式"):
            return "增量"
        return mode_text
    return "每次随机"


class Layer13RandomImageLoader:
    _session_tokens = {}
    _history_cache = {}
    _last_index_cache = {}
    _auto_index_cache = {}
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        auto_mode = (kwargs.get("运行后操作") or "不变").strip()
        if auto_mode and auto_mode != "不变":
            # 自动递增/随机需要每次执行
            return os.urandom(16).hex()
        mode = _resolve_mode(kwargs)
        min_gap = int(kwargs.get("最小重复间隔", 0) or 0)
        dedupe_by_name = bool(kwargs.get("按文件名去重", False))
        folder = kwargs.get("路径", "")
        pattern = kwargs.get("模式", "*.png")
        recursive = kwargs.get("递归", False)
        seed = int(kwargs.get("随机种子", 0) or 0)
        key = (folder, pattern, bool(recursive), mode, seed, min_gap, dedupe_by_name)
        if mode in ("每次随机", "每轮洗牌(不重复)") or min_gap > 0:
            # Force re-run each execution to avoid caching
            token = os.urandom(16).hex()
            cls._session_tokens[key] = token
            return token
        # Deterministic mode can be cached by seed/index
        loop_index = int(kwargs.get("循环索引", 0) or 0)
        return f"{seed}-{loop_index}"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "路径": ("STRING", {"default": ""}),
                "模式": ("STRING", {"default": "*.*"}),
                "递归": ("BOOLEAN", {"default": False}),
                "随机模式": (["每次随机", "增量", "固定随机(种子+索引)", "每轮洗牌(不重复)"], {"default": "每次随机"}),
                "增量模式": ("BOOLEAN", {"default": False}),
                "洗牌模式": ("BOOLEAN", {"default": False}),
                "固定随机模式": ("BOOLEAN", {"default": False}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "循环索引": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "最小重复间隔": ("INT", {"default": 0, "min": 0, "max": 999}),
                "按文件名去重": ("BOOLEAN", {"default": False}),
                "缩略预检查": ("BOOLEAN", {"default": True}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "文件名文本")
    FUNCTION = "apply"
    CATEGORY = "Layer13"

    def apply(self, **kwargs):
        folder = kwargs.get("路径", "")
        pattern = kwargs.get("模式", "*.png")
        recursive = kwargs.get("递归", False)
        mode = _resolve_mode(kwargs)
        seed = int(kwargs.get("随机种子", 0) or 0)
        loop_index = int(kwargs.get("循环索引", 0) or 0)
        min_gap = int(kwargs.get("最小重复间隔", 0) or 0)
        dedupe_by_name = bool(kwargs.get("按文件名去重", False))
        thumbnail_precheck = bool(kwargs.get("缩略预检查", True))

        files = _list_images(folder, pattern, recursive)
        if not files:
            raise ValueError(f"未找到图片: 路径={folder}, 模式={pattern}")

        idx = -1
        key = (folder, pattern, bool(recursive), mode, seed, min_gap, dedupe_by_name)
        token = self._session_tokens.get(key, "default")

        if mode == "每轮洗牌(不重复)":
            count = len(files)
            cycle = 0 if count == 0 else (loop_index // count)
            base = (seed or 0) + (abs(hash(token)) % 1_000_000_007)
            local = random.Random(base + cycle)
            order = list(range(count))
            local.shuffle(order)
            idx = order[loop_index % count] if count else -1
        else:
            idx = _pick_index(len(files), mode, seed, loop_index)

        if idx < 0:
            raise ValueError("随机选择失败")

        def _item_key(i: int):
            if dedupe_by_name:
                return os.path.splitext(os.path.basename(files[i]))[0]
            return i

        history = None
        selected_key = _item_key(idx)
        if min_gap > 0:
            # 历史键保持稳定，避免每次执行都重置最小重复间隔状态
            hist_key = (folder, pattern, bool(recursive), mode, seed, min_gap, dedupe_by_name)
            history = self._history_cache.get(hist_key)
            if history is None:
                history = deque(maxlen=min_gap)
                self._history_cache[hist_key] = history

            blocked = set(history)
            if selected_key in blocked:
                available = [i for i in range(len(files)) if _item_key(i) not in blocked]
                if available:
                    if mode in ("增量", "每轮洗牌(不重复)"):
                        for step in range(1, len(files) + 1):
                            cand = (idx + step) % len(files)
                            if _item_key(cand) not in blocked:
                                idx = cand
                                break
                    elif mode == "每次随机":
                        idx = random.SystemRandom().choice(available)
                    else:
                        local = random.Random((seed or 0) + (loop_index or 0) + len(history))
                        idx = available[local.randrange(len(available))]
                selected_key = _item_key(idx)

        # 尝试加载图片，遇到损坏文件则自动跳过
        attempts = 0
        bad = set()
        total = len(files)
        while attempts < total:
            path = files[idx]
            try:
                if thumbnail_precheck and not _thumbnail_precheck(path):
                    raise UnidentifiedImageError(f"缩略预检查失败: {path}")
                with Image.open(path) as img:
                    tensor = _image_to_tensor(img)
                _clear_bad(path)
                sig = _safe_file_signature(path)
                if sig is not None:
                    _precheck_cache[path] = {"sig": sig, "ok": True}
                filename_text = os.path.splitext(os.path.basename(path))[0]
                if min_gap > 0 and history is not None:
                    history.append(selected_key)
                # 记录当前索引，供 V2 使用
                self._last_index_cache[key] = idx
                return (tensor, filename_text)
            except (UnidentifiedImageError, OSError):
                _mark_bad(path)
                bad.add(idx)
                attempts += 1
                if attempts >= total:
                    break
                # 选择下一个索引（尽量保持当前模式且跳过坏图）
                if mode in ("增量", "每轮洗牌(不重复)"):
                    next_idx = (idx + 1) % total
                    while next_idx in bad and len(bad) < total:
                        next_idx = (next_idx + 1) % total
                    idx = next_idx
                else:
                    remaining = [i for i in range(total) if i not in bad]
                    if not remaining:
                        break
                    if min_gap > 0 and history is not None:
                        blocked = set(history)
                        remaining_no_repeat = [i for i in remaining if _item_key(i) not in blocked]
                        if remaining_no_repeat:
                            remaining = remaining_no_repeat
                    if mode == "每次随机":
                        rng = random.SystemRandom()
                        idx = rng.choice(remaining)
                    else:
                        local = random.Random((seed or 0) + (loop_index or 0) + attempts)
                        idx = remaining[local.randrange(len(remaining))]
                selected_key = _item_key(idx)

        bad_sample = files[next(iter(bad))] if bad else ""
        raise ValueError(f"图片无法识别或已损坏，已跳过 {len(bad)} 张，无法找到可用图片。示例: {bad_sample}")


class Layer13RandomImageLoaderV2_旧(Layer13RandomImageLoader):
    pass


class Layer13RandomImageLoaderV2(Layer13RandomImageLoader):
    @classmethod
    def INPUT_TYPES(cls):
        # 仅恢复“显示编号”相关功能（不改变其它行为）
        base = super().INPUT_TYPES()
        # 改为“起始编号”避免空字符串导致的校验失败
        base["required"]["起始编号"] = ("INT", {"default": 1, "min": 0, "max": 2147483647})
        base["required"]["显示编号"] = ("BOOLEAN", {"default": True})
        return base

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("图像", "文件名文本", "当前编号")
    FUNCTION = "apply_v2"

    def apply_v2(self, **kwargs):
        def _safe_int(v, default=0):
            try:
                if v is None:
                    return default
                if isinstance(v, str) and v.strip() == "":
                    return default
                return int(v)
            except Exception:
                return default

        # 兼容旧参数“当前编号”，优先使用“起始编号”
        start_number = _safe_int(kwargs.get("起始编号", None), default=None)
        if start_number is None:
            start_number = _safe_int(kwargs.get("当前编号", 1), default=1)
        loop_index = _safe_int(kwargs.get("循环索引", 0), default=0)
        start_offset = max(start_number - 1, 0)

        kwargs = dict(kwargs)
        kwargs["循环索引"] = loop_index + start_offset

        img, name = super().apply(**kwargs)

        # 读取上次索引（0-based），转为 1-based 方便显示
        folder = kwargs.get("路径", "")
        pattern = kwargs.get("模式", "*.png")
        recursive = kwargs.get("递归", False)
        mode = _resolve_mode(kwargs)
        seed = int(kwargs.get("随机种子", 0) or 0)
        min_gap = int(kwargs.get("最小重复间隔", 0) or 0)
        dedupe_by_name = bool(kwargs.get("按文件名去重", False))
        key = (folder, pattern, bool(recursive), mode, seed, min_gap, dedupe_by_name)
        idx = self._last_index_cache.get(key, -1)
        display_idx = (idx + 1) if (idx >= 0) else 0

        ui = {}
        if bool(kwargs.get("显示编号", True)):
            ui = {"text": [f"当前编号: {display_idx}"]}
        return {"ui": ui, "result": (img, name, display_idx)}


class Layer13FolderMonitorLoader:
    _state = {}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 每次执行都扫描
        return os.urandom(16).hex()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "路径": ("STRING", {"default": ""}),
                "模式": ("STRING", {"default": "*.*"}),
                "递归": ("BOOLEAN", {"default": False}),
                "初始化为已读取": ("BOOLEAN", {"default": True}),
                "重置": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "排序": (["名称", "修改时间"], {"default": "名称"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("图像", "文件名文本", "是否有新文件", "当前索引", "总数量")
    FUNCTION = "apply"
    CATEGORY = "Layer13"

    def apply(self, 路径, 模式="*.*", 递归=False, 初始化为已读取=True, 重置=False, 排序="名称"):
        key = (路径, 模式, bool(递归), 排序)
        if 重置 or key not in self._state:
            self._state[key] = {
                "seen": set(),
                "index": 0,
                "last_image": None,
                "last_name": "",
            }

        state = self._state[key]

        files = _list_images(路径, 模式, 递归)
        if 排序 == "修改时间":
            files.sort(key=lambda p: os.path.getmtime(p))

        total = len(files)
        if total == 0:
            state["index"] = 0
            return (_empty_image(), "", 0, 0, 0)

        # 初始化：把已有文件视为已读取
        if not state["seen"]:
            if 初始化为已读取:
                state["seen"] = set(files)
                state["index"] = len(files)
            else:
                state["seen"] = set()
                state["index"] = 0

        # 如果有新增或尚未处理的文件，逐个加载
        if state["index"] < len(files):
            idx = state["index"]
            path = files[idx]
            with Image.open(path) as img:
                tensor = _image_to_tensor(img)
            filename_text = os.path.splitext(os.path.basename(path))[0]
            state["last_image"] = tensor
            state["last_name"] = filename_text
            state["seen"].add(path)
            state["index"] += 1
            return (tensor, filename_text, 1, state["index"], total)

        # 没有新文件，返回上一张（避免 None）
        if state["last_image"] is None:
            return (_empty_image(), "", 0, state["index"], total)
        return (state["last_image"], state["last_name"], 0, state["index"], total)


class Layer13ConditionCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "条件值": ("INT", {"default": 0, "min": 0, "max": 1}),
                "判断": (["等于1", "大于0"], {"default": "等于1"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT")
    RETURN_NAMES = ("是否执行", "条件值")
    FUNCTION = "apply"
    CATEGORY = "Layer13"

    def apply(self, 条件值, 判断="等于1"):
        if 判断 == "大于0":
            ok = int(条件值) > 0
        else:
            ok = int(条件值) == 1
        return (bool(ok), int(条件值))


class Layer13FolderImageCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "路径": ("STRING", {"default": ""}),
                "模式": ("STRING", {"default": "*.*"}),
                "递归": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("数量", "文件名文本")
    FUNCTION = "count"
    CATEGORY = "Layer13"

    def count(self, 路径, 模式="*.*", 递归=False):
        files = _list_images(路径, 模式, 递归)
        # 输出文件名文本（按行）
        names = "\n".join([os.path.basename(p) for p in files])
        return (len(files), names)


NODE_CLASS_MAPPINGS = {
    "Layer13RandomImageLoader": Layer13RandomImageLoader,
    "Layer13RandomImageLoaderV2_旧": Layer13RandomImageLoaderV2_旧,
    "Layer13RandomImageLoaderV2": Layer13RandomImageLoaderV2,
    "Layer13FolderMonitorLoader": Layer13FolderMonitorLoader,
    "Layer13ConditionCheck": Layer13ConditionCheck,
    "Layer13FolderImageCount": Layer13FolderImageCount,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13RandomImageLoader": "Layer13随机加载图片(循环)",
    "Layer13RandomImageLoaderV2_旧": "Layer13随机加载图片(循环)V2(旧版)",
    "Layer13RandomImageLoaderV2": "Layer13随机加载图片(循环)V2",
    "Layer13FolderMonitorLoader": "Layer13文件夹监测加载",
    "Layer13ConditionCheck": "Layer13条件判断",
    "Layer13FolderImageCount": "Layer13文件夹图片数量",
}
