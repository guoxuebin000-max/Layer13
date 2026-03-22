from .layer13_multi_image_input import LAYER13_IMAGE_LIST_TYPE


class Layer13ForLoopIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "起始编号": ("INT", {"default": 0, "min": 0}),
                "步长": ("INT", {"default": 1, "min": 1}),
                "数量": ("INT", {"default": 1, "min": 1, "max": 100000}),
            },
            "optional": {
                "图像批次": ("IMAGE",),
                "图像列表": (LAYER13_IMAGE_LIST_TYPE,),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("循环编号", "总数量")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "生成"
    CATEGORY = "Layer13"

    def 生成(self, 起始编号=1, 步长=1, 数量=1, 图像批次=None, 图像列表=None):
        if 图像列表 is not None:
            if not isinstance(图像列表, list):
                raise ValueError("图像列表必须是有效的图像列表")
            total = len(图像列表)
        elif 图像批次 is not None:
            try:
                total = int(图像批次.shape[0])
            except Exception as exc:
                raise ValueError("图像批次必须是有效的 IMAGE 批次") from exc
        else:
            total = int(数量)

        if total <= 0:
            raise ValueError("数量必须大于 0")

        index_list = [int(起始编号 + i * 步长) for i in range(total)]
        return (index_list, total)


class Layer13ImageBatchGetByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像批次": ("IMAGE",),
                "编号": ("INT", {"default": 1, "min": 0, "step": 1}),
                "索引模式": (["1基", "0基"], {"default": "1基"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("单张图像", "当前编号", "总数量")
    FUNCTION = "取图"
    CATEGORY = "Layer13"

    def _resolve_index(self, raw_number: int, total: int, index_mode: str) -> tuple[int, int, int, int]:
        if index_mode == "0基":
            index = raw_number
            valid_min = 0
            valid_max = total - 1
            return index, index + 1, valid_min, valid_max

        index = raw_number - 1
        valid_min = 1
        valid_max = total
        return index, index + 1, valid_min, valid_max

    def 取图(self, 图像批次, 编号=1, 索引模式="1基"):
        try:
            total = int(图像批次.shape[0])
        except Exception as exc:
            raise ValueError("图像批次必须是有效的 IMAGE 批次") from exc

        if total <= 0:
            raise ValueError("图像批次为空")

        raw_number = int(编号)
        index, current_number, valid_min, valid_max = self._resolve_index(raw_number, total, str(索引模式))
        if index < 0 or index >= total:
            raise ValueError(f"编号超出范围: {编号}，有效范围 {valid_min} 到 {valid_max}")

        return (图像批次[index:index + 1], current_number, total)


NODE_CLASS_MAPPINGS = {
    "Layer13ForLoopIndex": Layer13ForLoopIndex,
    "Layer13ImageBatchGetByIndex": Layer13ImageBatchGetByIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13ForLoopIndex": "Layer13循环索引",
    "Layer13ImageBatchGetByIndex": "Layer13按索引取批次图像",
}
