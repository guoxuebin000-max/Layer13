import fnmatch
import json
import os
import random
import threading
from pathlib import Path
from typing import Dict, List, Tuple


class Layer13TextIncrementRandomLoader:
    MODE_OPTIONS = ["递增(记住位置)", "每次随机", "固定随机(种子+索引)", "指定索引", "循环索引"]
    SPLIT_OPTIONS = ["按行", "空行分段", "整文件"]
    OVERFLOW_OPTIONS = ["循环", "停止报错", "夹紧"]
    _state_file = Path(__file__).with_name("state").joinpath("text_loader.json")
    _lock = threading.Lock()
    _runtime_cache: Dict[Tuple, Tuple] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "路径": ("STRING", {"default": ""}),
                "模式": ("STRING", {"default": "*.txt;*.md;*.csv;*.json"}),
                "递归": ("BOOLEAN", {"default": False}),
                "拆分方式": (cls.SPLIT_OPTIONS, {"default": "按行"}),
                "选择模式": (cls.MODE_OPTIONS, {"default": "递增(记住位置)"}),
                "计数器名称": ("STRING", {"default": "default", "multiline": False}),
                "起始索引": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "索引": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "数量": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
                "拼接分隔符": ("STRING", {"default": "\n"}),
                "越界处理": (cls.OVERFLOW_OPTIONS, {"default": "循环"}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "忽略空条目": ("BOOLEAN", {"default": True}),
                "清理首尾空白": ("BOOLEAN", {"default": True}),
                "重置递增": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "循环索引": ("INT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("文本", "文件名", "当前索引", "下个索引", "总条目", "元数据")
    FUNCTION = "load_text"
    CATEGORY = "Layer13"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        mode = str(kwargs.get("选择模式") or "").strip()
        if mode in {"递增(记住位置)", "每次随机"}:
            return float("NaN")
        return json.dumps(
            {
                "path": kwargs.get("路径", ""),
                "pattern": kwargs.get("模式", ""),
                "recursive": bool(kwargs.get("递归", False)),
                "split": kwargs.get("拆分方式", ""),
                "mode": mode,
                "index": int(kwargs.get("索引", 0) or 0),
                "loop_index": kwargs.get("循环索引", None),
                "count": int(kwargs.get("数量", 1) or 1),
                "seed": int(kwargs.get("随机种子", 0) or 0),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _get_execution_context_key():
        try:
            from comfy_execution.utils import get_executing_context
        except Exception:
            return None
        context = get_executing_context()
        if context is None:
            return None
        return (context.prompt_id, context.node_id, context.list_index)

    @classmethod
    def _load_state(cls):
        if not cls._state_file.exists():
            return {}
        try:
            return json.loads(cls._state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @classmethod
    def _save_state(cls, state):
        cls._state_file.parent.mkdir(parents=True, exist_ok=True)
        cls._state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _patterns(pattern_text: str) -> List[str]:
        parts = []
        for raw in str(pattern_text or "*").replace(",", ";").split(";"):
            item = raw.strip()
            if item:
                parts.append(item)
        return parts or ["*"]

    @classmethod
    def _scan_files(cls, path_text: str, pattern_text: str, recursive: bool) -> List[str]:
        path = os.path.expanduser(str(path_text or "").strip())
        if not path:
            return []
        if os.path.isfile(path):
            return [path]
        if not os.path.isdir(path):
            return []

        patterns = cls._patterns(pattern_text)
        files = []
        if recursive:
            for root, _, names in os.walk(path):
                for name in names:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, path)
                    if any(fnmatch.fnmatch(name, patt) or fnmatch.fnmatch(rel, patt) for patt in patterns):
                        files.append(full)
        else:
            for name in os.listdir(path):
                full = os.path.join(path, name)
                if not os.path.isfile(full):
                    continue
                if any(fnmatch.fnmatch(name, patt) for patt in patterns):
                    files.append(full)
        files.sort()
        return files

    @staticmethod
    def _read_text_file(path: str) -> str:
        data = Path(path).read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _flatten_json(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        if isinstance(value, list):
            out = []
            for item in value:
                out.extend(Layer13TextIncrementRandomLoader._flatten_json(item))
            return out
        if isinstance(value, dict):
            for key in ("prompt", "text", "caption", "content", "value"):
                if key in value:
                    return Layer13TextIncrementRandomLoader._flatten_json(value[key])
            return [json.dumps(value, ensure_ascii=False)]
        return [str(value)]

    @classmethod
    def _split_entries(cls, path: str, split_mode: str, strip_text: bool, ignore_empty: bool):
        text = cls._read_text_file(path)
        if path.lower().endswith(".json"):
            try:
                raw_items = cls._flatten_json(json.loads(text))
            except Exception:
                raw_items = [text]
        elif split_mode == "整文件":
            raw_items = [text]
        elif split_mode == "空行分段":
            raw_items = []
            chunk = []
            for line in text.splitlines():
                if line.strip() == "":
                    if chunk:
                        raw_items.append("\n".join(chunk))
                        chunk = []
                else:
                    chunk.append(line)
            if chunk:
                raw_items.append("\n".join(chunk))
        else:
            raw_items = text.splitlines()

        entries = []
        for local_index, item in enumerate(raw_items):
            cur = str(item)
            if strip_text:
                cur = cur.strip()
            if ignore_empty and cur == "":
                continue
            entries.append(
                {
                    "text": cur,
                    "path": path,
                    "file": os.path.basename(path),
                    "entry_index": local_index,
                }
            )
        return entries

    @classmethod
    def _load_entries(
        cls,
        path_text: str,
        pattern_text: str,
        recursive: bool,
        split_mode: str,
        strip_text: bool,
        ignore_empty: bool,
    ):
        entries = []
        for path in cls._scan_files(path_text, pattern_text, recursive):
            entries.extend(cls._split_entries(path, split_mode, strip_text, ignore_empty))
        return entries

    @staticmethod
    def _normalize_index(index: int, total: int, overflow_mode: str) -> int:
        if total <= 0:
            raise ValueError("没有可用文本条目。")
        idx = int(index)
        if 0 <= idx < total:
            return idx
        if overflow_mode == "循环":
            return idx % total
        if overflow_mode == "夹紧":
            return max(0, min(idx, total - 1))
        raise IndexError(f"索引越界: {idx}, 总条目={total}")

    @classmethod
    def _next_increment_index(cls, name: str, start: int, count: int, reset: bool):
        state = cls._load_state()
        if reset or name not in state:
            current = int(start)
        else:
            entry = state.get(name)
            if isinstance(entry, dict):
                current = int(entry.get("next_index", start))
            else:
                current = int(entry)
        next_index = current + int(count)
        state[name] = {"next_index": next_index}
        cls._save_state(state)
        return current, next_index

    def _select_indices(
        self,
        total: int,
        mode: str,
        name: str,
        start_index: int,
        manual_index: int,
        count: int,
        seed: int,
        overflow_mode: str,
        reset_increment: bool,
        loop_index=None,
    ):
        count = max(1, int(count))
        if mode == "每次随机":
            rng = random.SystemRandom()
            indices = [rng.randrange(total) for _ in range(count)]
            return indices, indices[0], -1

        if mode == "固定随机(种子+索引)":
            base = int(loop_index) if loop_index is not None else int(manual_index)
            rng = random.Random(int(seed) + base)
            indices = [rng.randrange(total) for _ in range(count)]
            return indices, indices[0], -1

        if mode == "指定索引":
            current = int(manual_index)
            indices = [self._normalize_index(current + i, total, overflow_mode) for i in range(count)]
            return indices, current, current + count

        if mode == "循环索引":
            current = int(loop_index) if loop_index is not None else int(manual_index)
            indices = [self._normalize_index(current + i, total, overflow_mode) for i in range(count)]
            return indices, current, current + count

        current, next_index = self._next_increment_index(name, start_index, count, reset_increment)
        indices = [self._normalize_index(current + i, total, overflow_mode) for i in range(count)]
        return indices, current, next_index

    def load_text(
        self,
        路径: str = "",
        模式: str = "*.txt;*.md;*.csv;*.json",
        递归: bool = False,
        拆分方式: str = "按行",
        选择模式: str = "递增(记住位置)",
        计数器名称: str = "default",
        起始索引: int = 0,
        索引: int = 0,
        数量: int = 1,
        拼接分隔符: str = "\n",
        越界处理: str = "循环",
        随机种子: int = 0,
        忽略空条目: bool = True,
        清理首尾空白: bool = True,
        重置递增: bool = False,
        循环索引=None,
    ):
        mode = str(选择模式 or "递增(记住位置)")
        name = str(计数器名称 or "default").strip() or "default"
        context_key = self._get_execution_context_key()
        runtime_key = None
        if mode == "递增(记住位置)" and context_key is not None and not 重置递增:
            runtime_key = ("text_loader", name, context_key)
            cached = self._runtime_cache.get(runtime_key)
            if cached is not None:
                return cached

        entries = self._load_entries(
            路径,
            模式,
            bool(递归),
            拆分方式,
            bool(清理首尾空白),
            bool(忽略空条目),
        )
        if not entries:
            raise ValueError(f"未找到可用文本条目: 路径={路径}, 模式={模式}")

        with self._lock:
            indices, current_index, next_index = self._select_indices(
                len(entries),
                mode,
                name,
                int(起始索引),
                int(索引),
                int(数量),
                int(随机种子),
                str(越界处理 or "循环"),
                bool(重置递增),
                循环索引,
            )

        picked = [entries[i] for i in indices]
        text = str(拼接分隔符).join(item["text"] for item in picked)
        filenames = "\n".join(item["file"] for item in picked)
        metadata = {
            "mode": mode,
            "requested_index": int(current_index),
            "selected_indices": indices,
            "next_index": int(next_index),
            "total": len(entries),
            "items": [
                {
                    "selected_index": indices[pos],
                    "file": item["file"],
                    "path": item["path"],
                    "entry_index": item["entry_index"],
                }
                for pos, item in enumerate(picked)
            ],
        }
        output = (
            text,
            filenames,
            int(current_index),
            int(next_index),
            int(len(entries)),
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )
        if runtime_key is not None:
            self._runtime_cache[runtime_key] = output
            if len(self._runtime_cache) > 4096:
                self._runtime_cache.clear()
        return output


NODE_CLASS_MAPPINGS = {
    "Layer13TextIncrementRandomLoader": Layer13TextIncrementRandomLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13TextIncrementRandomLoader": "Layer13增量/随机加载文本",
}

