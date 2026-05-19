from __future__ import annotations
import inspect


class _AnyType(str):
    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


ANY_TYPE = _AnyType("*")
MAX_INPUTS = 20
WEB_DIRECTORY = "web"


class Layer13AnyIndexSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        dyn_inputs = {
            "input1": (ANY_TYPE, {"lazy": True, "tooltip": "连接后会自动增加下一个输入接口。"}),
        }

        stack = inspect.stack()
        if len(stack) > 2 and stack[2].function == "get_input_info":
            class AllContainer:
                def __contains__(self, _item):
                    return True

                def __getitem__(self, _key):
                    return (ANY_TYPE, {"lazy": True})

            dyn_inputs = AllContainer()

        return {
            "required": {
                "选择": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1}),
            },
            "optional": dyn_inputs,
        }

    RETURN_TYPES = (ANY_TYPE, "INT")
    RETURN_NAMES = ("value", "selected_index")
    FUNCTION = "select"
    CATEGORY = "Layer13/逻辑"

    def check_lazy_status(self, 选择, **kwargs):
        key = f"input{int(选择)}"
        if key in kwargs:
            return [key]
        return []

    def select(self, 选择, **kwargs):
        selected_index = max(1, int(选择))
        key = f"input{selected_index}"
        if key in kwargs:
            return (kwargs[key], selected_index)
        return (None, selected_index)


class Layer13InputCountControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "值": ("INT", {"default": 1, "min": -0x7FFFFFFFFFFFFFFF, "max": 0x7FFFFFFFFFFFFFFF, "step": 1}),
                "最小值": ("INT", {"default": 1, "min": -0x7FFFFFFFFFFFFFFF, "max": 0x7FFFFFFFFFFFFFFF, "step": 1}),
                "最大值": ("INT", {"default": 10, "min": -0x7FFFFFFFFFFFFFFF, "max": 0x7FFFFFFFFFFFFFFF, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("整数",)
    FUNCTION = "output_int"
    CATEGORY = "Layer13/逻辑"

    @staticmethod
    def _parse_int(value, fallback):
        try:
            text = str(value).strip()
            if not text:
                return int(fallback)
            return int(float(text))
        except Exception:
            return int(fallback)

    def output_int(self, 值, 最小值, 最大值):
        current_value = int(值)
        lower = self._parse_int(最小值, 1)
        upper = self._parse_int(最大值, lower)
        if lower > upper:
            lower, upper = upper, lower
        current_value = max(lower, min(upper, current_value))
        return (current_value,)


class Layer13AnyBoolSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
            },
            "optional": {
                "true_value": (ANY_TYPE, {"lazy": True}),
                "false_value": (ANY_TYPE, {"lazy": True}),
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)
    FUNCTION = "select"
    CATEGORY = "Layer13/逻辑"

    def check_lazy_status(self, condition, true_value=..., false_value=...):
        # Keep the other side runnable if only one branch is connected.
        if true_value is ... and false_value is ...:
            return []
        if true_value is ...:
            return ["false_value"]
        if false_value is ...:
            return ["true_value"]

        if condition and true_value is None:
            return ["true_value"]
        if not condition and false_value is None:
            return ["false_value"]
        return []

    def select(self, condition, true_value=..., false_value=...):
        if true_value is ... and false_value is ...:
            raise ValueError("布尔任意切换节点没有连接任何输入。")
        if true_value is ...:
            return (false_value,)
        if false_value is ...:
            return (true_value,)
        return (true_value if condition else false_value,)


class Layer13AnyRouteSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "条件值": ("INT", {"default": 1, "min": -0x7FFFFFFFFFFFFFFF, "max": 0x7FFFFFFFFFFFFFFF}),
                "A条件值": ("INT", {"default": 1, "min": -0x7FFFFFFFFFFFFFFF, "max": 0x7FFFFFFFFFFFFFFF}),
                "B条件值": ("INT", {"default": 2, "min": -0x7FFFFFFFFFFFFFFF, "max": 0x7FFFFFFFFFFFFFFF}),
                "value": (ANY_TYPE,),
            },
        }

    RETURN_TYPES = (ANY_TYPE, ANY_TYPE)
    RETURN_NAMES = ("a_value", "b_value")
    FUNCTION = "route"
    CATEGORY = "Layer13/逻辑"

    @staticmethod
    def _block():
        from comfy_execution.graph import ExecutionBlocker

        return ExecutionBlocker(None)

    def route(self, 条件值, A条件值, B条件值, value=None):
        current_value = int(条件值)
        a_condition = int(A条件值)
        b_condition = int(B条件值)

        if current_value == a_condition:
            return (
                value,
                self._block(),
            )
        if current_value == b_condition:
            return (
                self._block(),
                value,
            )

        return (
            self._block(),
            self._block(),
        )


class Layer13ConditionalAnySwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "条件值": ("INT", {"default": 1, "min": -0x7FFFFFFFFFFFFFFF, "max": 0x7FFFFFFFFFFFFFFF}),
                "A条件值": ("INT", {"default": 1, "min": -0x7FFFFFFFFFFFFFFF, "max": 0x7FFFFFFFFFFFFFFF}),
                "B条件值": ("INT", {"default": 2, "min": -0x7FFFFFFFFFFFFFFF, "max": 0x7FFFFFFFFFFFFFFF}),
            },
            "optional": {
                "a_value": (ANY_TYPE, {"lazy": True}),
                "b_value": (ANY_TYPE, {"lazy": True}),
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)
    FUNCTION = "select"
    CATEGORY = "Layer13/逻辑"

    @staticmethod
    def _choose_branch(current_value: int, a_condition: int, b_condition: int) -> str | None:
        if current_value == a_condition:
            return "a_value"
        if current_value == b_condition:
            return "b_value"
        return None

    def check_lazy_status(self, 条件值, A条件值, B条件值, a_value=..., b_value=...):
        if a_value is ... and b_value is ...:
            return []
        if a_value is ...:
            return ["b_value"]
        if b_value is ...:
            return ["a_value"]

        branch = self._choose_branch(int(条件值), int(A条件值), int(B条件值))
        if branch == "a_value" and a_value is None:
            return ["a_value"]
        if branch == "b_value" and b_value is None:
            return ["b_value"]
        return []

    def select(self, 条件值, A条件值, B条件值, a_value=..., b_value=...):
        if a_value is ... and b_value is ...:
            raise ValueError("条件单输出切换节点没有连接任何输入。")
        if a_value is ...:
            return (b_value,)
        if b_value is ...:
            return (a_value,)

        branch = self._choose_branch(int(条件值), int(A条件值), int(B条件值))
        if branch == "a_value":
            return (a_value,)
        if branch == "b_value":
            return (b_value,)

        from comfy_execution.graph import ExecutionBlocker

        return (ExecutionBlocker(None),)


NODE_CLASS_MAPPINGS = {
    "Layer13AnyIndexSwitch": Layer13AnyIndexSwitch,
    "Layer13InputCountControl": Layer13InputCountControl,
    "Layer13AnyBoolSwitch": Layer13AnyBoolSwitch,
    "Layer13AnyRouteSwitch": Layer13AnyRouteSwitch,
    "Layer13ConditionalAnySwitch": Layer13ConditionalAnySwitch,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13AnyIndexSwitch": "Layer13 任意编号切换",
    "Layer13InputCountControl": "Layer13 整数",
    "Layer13AnyBoolSwitch": "Layer13 布尔任意切换",
    "Layer13AnyRouteSwitch": "Layer13 任意路由开关",
    "Layer13ConditionalAnySwitch": "Layer13 条件单输出切换",
}
