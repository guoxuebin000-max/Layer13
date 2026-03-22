from typing import Any, List


class Layer13TextJoinN:
    MAX_INPUTS = 32

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {}
        for i in range(1, cls.MAX_INPUTS + 1):
            optional_inputs[f"文本{i}"] = ("STRING", {"forceInput": True})

        return {
            "required": {
                "N": ("INT", {"default": 9, "min": 1, "max": cls.MAX_INPUTS}),
                "分隔符": ("STRING", {"default": ", "}),
                "忽略空文本": ("BOOLEAN", {"default": True}),
                "清理首尾空白": ("BOOLEAN", {"default": True}),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("文本", "实际数量")
    FUNCTION = "join_texts"
    CATEGORY = "Layer13"

    @staticmethod
    def _flatten(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            out: List[str] = []
            for item in value:
                out.extend(Layer13TextJoinN._flatten(item))
            return out
        return [str(value)]

    def join_texts(
        self,
        N: int = 9,
        分隔符: str = ", ",
        忽略空文本: bool = True,
        清理首尾空白: bool = True,
        **kwargs,
    ):
        max_count = max(1, min(int(N), self.MAX_INPUTS))
        chunks: List[str] = []

        for i in range(1, max_count + 1):
            key = f"文本{i}"
            if key not in kwargs:
                continue
            parts = self._flatten(kwargs.get(key))
            for part in parts:
                text = part.strip() if 清理首尾空白 else part
                if 忽略空文本 and text == "":
                    continue
                chunks.append(text)

        return (str(分隔符).join(chunks), len(chunks))


class Layer13PrefixInjectLines:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "前缀文本": ("STRING", {"multiline": True, "default": ""}),
                "逐行文本": ("STRING", {"multiline": True, "default": ""}),
                "前缀与行连接符": ("STRING", {"default": ", "}),
                "行分隔符": ("STRING", {"default": "\n"}),
                "输出拼接分隔符": ("STRING", {"default": "||"}),
                "忽略空行": ("BOOLEAN", {"default": True}),
                "清理首尾空白": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("逐行结果", "拼接结果", "行数")
    FUNCTION = "inject"
    CATEGORY = "Layer13"

    @staticmethod
    def _norm_text(value):
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            if not value:
                return ""
            return str(value[0])
        return str(value)

    def inject(
        self,
        前缀文本: str = "",
        逐行文本: str = "",
        前缀与行连接符: str = ", ",
        行分隔符: str = "\n",
        输出拼接分隔符: str = "||",
        忽略空行: bool = True,
        清理首尾空白: bool = True,
    ):
        prefix = self._norm_text(前缀文本)
        text = self._norm_text(逐行文本)

        if 清理首尾空白:
            prefix = prefix.strip()

        delimiter = 行分隔符 if 行分隔符 != "" else "\n"
        raw_lines = text.split(delimiter)

        items = []
        for line in raw_lines:
            cur = line.strip() if 清理首尾空白 else line
            if 忽略空行 and cur == "":
                continue
            if prefix == "":
                out = cur
            elif cur == "":
                out = prefix
            else:
                out = f"{prefix}{前缀与行连接符}{cur}"
            items.append(out)

        line_join = "\n".join(items)
        packed = str(输出拼接分隔符).join(items)
        return (line_join, packed, len(items))


NODE_CLASS_MAPPINGS = {
    "Layer13TextJoinN": Layer13TextJoinN,
    "Layer13PrefixInjectLines": Layer13PrefixInjectLines,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13TextJoinN": "Layer13联结N文本",
    "Layer13PrefixInjectLines": "Layer13前缀注入每行",
}
