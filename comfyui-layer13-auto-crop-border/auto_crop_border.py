import numpy as np
import torch


def _to_numpy(img: torch.Tensor) -> np.ndarray:
    # img: [H,W,3] float 0..1
    if img.is_cuda:
        img = img.cpu()
    arr = img.detach().numpy()
    return arr


def _from_numpy(arr: np.ndarray) -> torch.Tensor:
    # arr: [H,W,3] float 0..1
    return torch.from_numpy(arr.astype(np.float32))


def _scan_edge(arr: np.ndarray, ref: np.ndarray, tol: float, edge_ratio: float, texture_tol: float, bright_th: float, dark_th: float):
    # Returns top, bottom, left, right crop indices (inclusive) based on edge scan
    h, w, _ = arr.shape
    dist = np.abs(arr - ref).mean(axis=2)
    within = dist <= tol
    # Row/col mean & texture (std) for robust detection
    row_mean = arr.mean(axis=1)
    row_std = arr.std(axis=1).mean(axis=1)
    col_mean = arr.mean(axis=0)
    col_std = arr.std(axis=0).mean(axis=1)
    # Luma for bright/dark border detection
    luma = (arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722)
    row_luma = luma.mean(axis=1)
    row_luma_std = luma.std(axis=1)
    col_luma = luma.mean(axis=0)
    col_luma_std = luma.std(axis=0)

    def is_uniform_bright_or_dark_row(y: int) -> bool:
        return (row_luma_std[y] <= texture_tol) and (row_luma[y] >= bright_th or row_luma[y] <= dark_th)

    def is_uniform_bright_or_dark_col(x: int) -> bool:
        return (col_luma_std[x] <= texture_tol) and (col_luma[x] >= bright_th or col_luma[x] <= dark_th)

    def scan_top():
        t = 0
        for y in range(h):
            mean_dist = np.abs(row_mean[y] - ref).mean()
            if within[y].mean() >= edge_ratio or (mean_dist <= tol and row_std[y] <= texture_tol) or is_uniform_bright_or_dark_row(y):
                t = y + 1
            else:
                break
        return t

    def scan_bottom():
        b = h - 1
        for y in range(h - 1, -1, -1):
            mean_dist = np.abs(row_mean[y] - ref).mean()
            if within[y].mean() >= edge_ratio or (mean_dist <= tol and row_std[y] <= texture_tol) or is_uniform_bright_or_dark_row(y):
                b = y - 1
            else:
                break
        return b

    def scan_left():
        l = 0
        for x in range(w):
            mean_dist = np.abs(col_mean[x] - ref).mean()
            if within[:, x].mean() >= edge_ratio or (mean_dist <= tol and col_std[x] <= texture_tol) or is_uniform_bright_or_dark_col(x):
                l = x + 1
            else:
                break
        return l

    def scan_right():
        r = w - 1
        for x in range(w - 1, -1, -1):
            mean_dist = np.abs(col_mean[x] - ref).mean()
            if within[:, x].mean() >= edge_ratio or (mean_dist <= tol and col_std[x] <= texture_tol) or is_uniform_bright_or_dark_col(x):
                r = x - 1
            else:
                break
        return r

    top = scan_top()
    bottom = scan_bottom()
    left = scan_left()
    right = scan_right()
    return top, bottom, left, right


def _auto_ref_color(arr: np.ndarray) -> np.ndarray:
    h, w, _ = arr.shape
    strip = max(2, int(min(h, w) * 0.03))
    candidates = []
    # top, bottom, left, right strips
    strips = [
        arr[:strip, :, :],
        arr[-strip:, :, :],
        arr[:, :strip, :],
        arr[:, -strip:, :],
    ]
    for s in strips:
        mean = s.mean(axis=(0, 1))
        std = s.std(axis=(0, 1)).mean()
        candidates.append((std, mean))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _detect_border_bbox(arr: np.ndarray, mode: str, tol: float, min_border: int, protect: int, edge_ratio: float, texture_tol: float, bright_th: float, dark_th: float):
    h, w, _ = arr.shape
    # Compute reference color
    if mode == "白边":
        ref = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    elif mode == "黑边":
        ref = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        # Auto: use most uniform edge strip
        ref = _auto_ref_color(arr)

    # Edge scan by ratio
    top, bottom, left, right = _scan_edge(arr, ref, tol, edge_ratio, texture_tol, bright_th, dark_th)
    # Fallback if scan eats everything
    if top > bottom or left > right:
        return (0, 0, w, h)

    # Expand by protect margin
    top = max(0, top - protect)
    left = max(0, left - protect)
    bottom = min(h - 1, bottom + protect)
    right = min(w - 1, right + protect)

    # Enforce min border (only if border exists)
    if top <= min_border:
        top = 0
    if left <= min_border:
        left = 0
    if (h - 1 - bottom) <= min_border:
        bottom = h - 1
    if (w - 1 - right) <= min_border:
        right = w - 1

    # Convert to bbox (x1,y1,x2,y2) exclusive on right/bottom
    return (left, top, right + 1, bottom + 1)


class Layer13AutoCropBorder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "模式": ("COMBO", {"default": "自动", "values": ["自动", "白边", "黑边"]}),
                "颜色容差": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.3, "step": 0.005}),
                "边缘占比阈值": ("FLOAT", {"default": 0.98, "min": 0.7, "max": 1.0, "step": 0.01}),
                "纹理阈值": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "亮度阈值": ("FLOAT", {"default": 0.92, "min": 0.5, "max": 1.0, "step": 0.01}),
                "暗度阈值": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "最小边框": ("INT", {"default": 2, "min": 0, "max": 200}),
                "保护边距": ("INT", {"default": 2, "min": 0, "max": 200}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "裁切框")
    FUNCTION = "apply"
    CATEGORY = "Layer13"

    def apply(self, 图像, 模式, 颜色容差, 边缘占比阈值, 纹理阈值, 亮度阈值, 暗度阈值, 最小边框, 保护边距):
        # 图像: [B,H,W,3]
        images = 图像
        if images.dim() != 4 or images.shape[-1] != 3:
            raise ValueError("输入必须是 IMAGE (B,H,W,3)")

        b, h, w, _ = images.shape
        out_list = []
        bbox_list = []

        for i in range(b):
            arr = _to_numpy(images[i])
            bbox = _detect_border_bbox(
                arr,
                mode=模式,
                tol=float(颜色容差),
                min_border=int(最小边框),
                protect=int(保护边距),
                edge_ratio=float(边缘占比阈值),
                texture_tol=float(纹理阈值),
                bright_th=float(亮度阈值),
                dark_th=float(暗度阈值),
            )
            x1, y1, x2, y2 = bbox
            cropped = arr[y1:y2, x1:x2, :]
            out_list.append(_from_numpy(cropped))
            bbox_list.append(f"{x1},{y1},{x2},{y2}")

        # Pad to same size? ComfyUI expects consistent batch sizes.
        # If sizes differ, keep only first bbox and return batch with same size by padding.
        # Here we enforce same size by padding to max H/W in batch.
        max_h = max(t.shape[0] for t in out_list)
        max_w = max(t.shape[1] for t in out_list)
        padded = []
        for t in out_list:
            ph = max_h - t.shape[0]
            pw = max_w - t.shape[1]
            if ph or pw:
                pad = torch.zeros((max_h, max_w, 3), dtype=t.dtype)
                pad[: t.shape[0], : t.shape[1], :] = t
                padded.append(pad)
            else:
                padded.append(t)

        out = torch.stack(padded, dim=0).clamp(0.0, 1.0)
        bbox_text = "|".join(bbox_list)
        return (out, bbox_text)


NODE_CLASS_MAPPINGS = {
    "Layer13AutoCropBorder": Layer13AutoCropBorder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer13AutoCropBorder": "Layer13自动裁边(白/黑边)",
}
