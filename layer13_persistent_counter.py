import json
import threading
from pathlib import Path


class Layer13PersistentCounter:
    _lock = threading.Lock()
    _state_file = Path(__file__).with_name("state").joinpath("persistent_counter.json")
    _runtime_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "计数器名称": ("STRING", {"default": "default", "multiline": False}),
                "起始值": ("INT", {"default": 0, "step": 1}),
                "步长": ("INT", {"default": 1, "step": 1}),
                "重置": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "循环索引": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("上一个值", "当前值", "下一个值")
    FUNCTION = "next_value"
    CATEGORY = "Layer13"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

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
        cls._state_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def reset_counter(cls, name: str):
        safe_name = str(name).strip() or "default"
        with cls._lock:
            state = cls._load_state()
            existed = safe_name in state
            if existed:
                state.pop(safe_name, None)
                cls._save_state(state)
            cls._runtime_cache = {
                key: value
                for key, value in cls._runtime_cache.items()
                if not (isinstance(key, tuple) and len(key) > 1 and key[1] == safe_name)
            }
        return safe_name, existed

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
    def _remember_runtime_output(cls, key, output):
        if key is None:
            return
        cls._runtime_cache[key] = output
        if len(cls._runtime_cache) > 4096:
            cls._runtime_cache.clear()

    @staticmethod
    def _parse_entry(entry, start, step):
        if isinstance(entry, dict):
            current = int(entry.get("current", start))
            previous = int(entry.get("previous", current - step))
            if "loop_index" in entry:
                loop_index = int(entry["loop_index"])
            elif step != 0 and (current - start) % step == 0:
                loop_index = int((current - start) // step)
            else:
                loop_index = 0
            return previous, current, loop_index

        # 兼容旧版本：之前只保存“下一次要输出的当前值”
        current = int(entry)
        previous = current - step
        if step != 0 and (current - start) % step == 0:
            loop_index = int((current - start) // step)
        else:
            loop_index = 0
        return previous, current, loop_index

    @staticmethod
    def _parse_external_entry(entry, start):
        if isinstance(entry, dict):
            if "run_base" in entry:
                run_base = int(entry.get("run_base", start))
            else:
                # 兼容旧状态：current 表示“下一次应该输出的当前值”，可作为新一轮基准。
                run_base = int(entry.get("current", start))
            if "last_loop_index" in entry:
                last_loop_index = int(entry.get("last_loop_index", -1))
            elif "loop_index" in entry:
                last_loop_index = int(entry.get("loop_index", 0)) - 1
            else:
                last_loop_index = -1
            return run_base, last_loop_index

        # 兼容更老的整型状态：该值视为下一轮起点。
        return int(entry), -1

    def next_value(self, 计数器名称="default", 起始值=0, 步长=1, 重置=False, 循环索引=None):
        name = str(计数器名称).strip() or "default"
        start = int(起始值)
        step = int(步长)
        has_external_loop_index = 循环索引 is not None
        loop_index_input = int(循环索引) if has_external_loop_index else None
        context_key = self._get_execution_context_key()
        runtime_key = (
            "counter",
            name,
            context_key,
            "external" if has_external_loop_index else "internal",
            loop_index_input,
        ) if context_key is not None and not 重置 else None

        with self._lock:
            if runtime_key is not None and runtime_key in self._runtime_cache:
                return self._runtime_cache[runtime_key]

            state = self._load_state()
            if has_external_loop_index:
                if 重置 or name not in state:
                    run_base = start
                    last_loop_index = -1
                else:
                    run_base, last_loop_index = self._parse_external_entry(state[name], start)

                # Only a strict rollback starts a new run. Repeated evaluation of the
                # same loop index can happen inside ComfyUI and must stay idempotent.
                if loop_index_input < last_loop_index:
                    run_base = run_base + (last_loop_index + 1) * step
                    last_loop_index = -1

                current = run_base + loop_index_input * step
                previous = current - step
                next_value = current + step
                state[name] = {
                    "mode": "external",
                    "run_base": run_base,
                    "last_loop_index": loop_index_input,
                    "previous": current,
                    "current": next_value,
                    "loop_index": loop_index_input + 1,
                }
            else:
                if 重置 or name not in state:
                    current = start
                    previous = current - step
                    next_loop_index = 1
                else:
                    previous, current, saved_loop_index = self._parse_entry(state[name], start, step)
                    next_loop_index = saved_loop_index + 1
                next_value = current + step
                state[name] = {
                    "mode": "internal",
                    "previous": current,
                    "current": next_value,
                    "loop_index": next_loop_index,
                }
            self._save_state(state)

            output = (previous, current, next_value)
            self._remember_runtime_output(runtime_key, output)

        return output

NODE_CLASS_MAPPINGS = {
    "Layer13PersistentCounter": Layer13PersistentCounter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13PersistentCounter": "Layer13持久化计数器",
}
