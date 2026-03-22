import math

import torch
import torch.nn.functional as F


def _smoothstep(edge0, edge1, x):
    t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gaussian_kernel1d(sigma, device, dtype):
    if sigma <= 0:
        return None
    radius = max(1, int(math.ceil(sigma * 3.0)))
    size = radius * 2 + 1
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma + 1e-8))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    b, c, h, w = img.shape
    kernel = _gaussian_kernel1d(sigma, img.device, img.dtype)
    if kernel is None:
        return img
    kernel = kernel.view(1, 1, -1)
    # Horizontal
    pad = kernel.shape[-1] // 2
    img = F.pad(img, (pad, pad, 0, 0), mode="reflect")
    img = F.conv2d(img, kernel.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    # Vertical
    img = F.pad(img, (0, 0, pad, pad), mode="reflect")
    img = F.conv2d(img, kernel.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return img


def _sobel_edges(luma):
    # luma: (B,1,H,W)
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=luma.device,
        dtype=luma.dtype,
    )
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=luma.device,
        dtype=luma.dtype,
    )
    sobel_x = sobel_x.unsqueeze(0)
    sobel_y = sobel_y.unsqueeze(0)
    gx = F.conv2d(luma, sobel_x, padding=1)
    gy = F.conv2d(luma, sobel_y, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return mag


def _rgb_to_ycbcr(img):
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y) + 0.5
    cr = 0.713 * (r - y) + 0.5
    return y, cb, cr


def _skin_mask(img):
    y, cb, cr = _rgb_to_ycbcr(img)
    cb_mask = _smoothstep(0.33, 0.55, cb)
    cr_mask = _smoothstep(0.40, 0.68, cr)
    y_mask = _smoothstep(0.10, 0.95, y)
    mask = cb_mask * cr_mask * y_mask
    return mask.clamp(0.0, 1.0)


def _apply_clarity(img, amount, skin_mask, skin_protect, strength):
    if abs(amount) <= 1e-6:
        return img
    blurred = _gaussian_blur(img, 1.0)
    highpass = img - blurred
    protect = 1.0 - skin_protect * skin_mask
    highpass = highpass * protect
    return img + highpass * amount * strength


def _apply_highlight_rolloff(img, rolloff):
    if rolloff <= 1e-6:
        return img
    threshold = 0.7
    high = torch.clamp(img - threshold, min=0.0)
    comp = rolloff * (high * high) / max(1e-6, (1.0 - threshold))
    return img - comp


def _apply_bloom(img, bloom, highlight_rolloff, skin_mask, skin_protect, strength):
    if bloom <= 1e-6:
        return img
    luma, _, _ = _rgb_to_ycbcr(img)
    thr = 0.7 - 0.3 * highlight_rolloff
    hl = _smoothstep(thr, 1.0, luma)
    hl = hl * (1.0 - skin_protect * skin_mask)
    bloom_src = img * hl
    bloom_blur = _gaussian_blur(bloom_src, 6.0 + 10.0 * bloom)
    amount = bloom * strength
    # Screen blend
    out = 1.0 - (1.0 - img) * (1.0 - bloom_blur * amount)
    return out


def _apply_halation(img, halation, highlight_rolloff, skin_mask, skin_protect, strength):
    if halation <= 1e-6:
        return img
    luma, _, _ = _rgb_to_ycbcr(img)
    thr = 0.7 - 0.3 * highlight_rolloff
    hl = _smoothstep(thr, 1.0, luma)
    edges = _sobel_edges(luma)
    edge_mask = _smoothstep(0.02, 0.15, edges)
    mask = hl * edge_mask
    mask = mask * (1.0 - skin_protect * skin_mask)
    mask = _gaussian_blur(mask, 8.0 + 12.0 * halation)
    color = torch.tensor([1.0, 0.35, 0.15], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    hal = mask * color
    return img + hal * (halation * strength)


def _apply_chroma_aberration(img, amount):
    if amount <= 1e-6:
        return img
    b, c, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=img.device, dtype=img.dtype),
        torch.linspace(-1.0, 1.0, w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    r = torch.sqrt(xs * xs + ys * ys).clamp(0.0, 1.0)
    shift = (0.004 * amount) * r
    # Normalize direction
    norm = torch.sqrt(xs * xs + ys * ys + 1e-8)
    dx = xs / norm
    dy = ys / norm
    base_grid = torch.stack((xs, ys), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    shift_grid = torch.stack((dx * shift, dy * shift), dim=-1).unsqueeze(0)
    r_grid = base_grid - shift_grid
    b_grid = base_grid + shift_grid

    def sample(channel, grid):
        return F.grid_sample(channel, grid, align_corners=True, padding_mode="border")

    rch = sample(img[:, 0:1], r_grid)
    gch = img[:, 1:2]
    bch = sample(img[:, 2:3], b_grid)
    return torch.cat((rch, gch, bch), dim=1)


def _apply_vignette(img, vignette):
    if vignette <= 1e-6:
        return img
    b, c, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=img.device, dtype=img.dtype),
        torch.linspace(-1.0, 1.0, w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    r2 = (xs * xs + ys * ys).clamp(0.0, 1.0)
    mask = 1.0 - vignette * _smoothstep(0.0, 1.0, r2)
    return img * mask.unsqueeze(0).unsqueeze(0)


def _apply_film_grain(img, film_grain, grain_size, skin_mask, skin_protect, strength):
    if film_grain <= 1e-6:
        return img
    b, c, h, w = img.shape
    noise = torch.randn((b, 1, h, w), device=img.device, dtype=img.dtype)
    noise = _gaussian_blur(noise, float(grain_size))
    luma, _, _ = _rgb_to_ycbcr(img)
    weight = (1.0 - luma).clamp(0.0, 1.0) ** 1.5
    protect = 1.0 - skin_protect * skin_mask
    weight = weight * protect
    grain = noise * weight * (film_grain * strength * 0.06)
    return (img + grain.repeat(1, 3, 1, 1))


class PhotoRealismEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "强度": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
                "胶片颗粒": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "颗粒大小": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 2.0, "step": 0.01}),
                "红晕": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "高光泛光": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "色差": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "暗角": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "高光压缩": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "清晰度": ("FLOAT", {"default": 0.10, "min": -1.0, "max": 1.0, "step": 0.01}),
                "肤色保护": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Layer13"

    def apply(
        self,
        图像,
        强度=0.6,
        胶片颗粒=0.25,
        颗粒大小=0.45,
        红晕=0.25,
        高光泛光=0.30,
        色差=0.50,
        暗角=0.12,
        高光压缩=0.20,
        清晰度=0.10,
        肤色保护=0.55,
    ):
        # image: (B,H,W,3) in 0..1
        img = 图像
        if img.dim() != 4 or img.shape[-1] != 3:
            raise ValueError("Input IMAGE must be (B,H,W,3)")

        img = img.clamp(0.0, 1.0)
        img = img.permute(0, 3, 1, 2).contiguous()

        skin_mask = _skin_mask(img)

        img = _apply_highlight_rolloff(img, float(高光压缩) * float(强度))
        img = _apply_clarity(img, float(清晰度), skin_mask, float(肤色保护), float(强度))
        img = _apply_chroma_aberration(img, float(色差) * float(强度))
        img = _apply_bloom(img, float(高光泛光), float(高光压缩), skin_mask, float(肤色保护), float(强度))
        img = _apply_halation(img, float(红晕), float(高光压缩), skin_mask, float(肤色保护), float(强度))
        img = _apply_vignette(img, float(暗角) * float(强度))
        img = _apply_film_grain(
            img,
            float(胶片颗粒),
            float(颗粒大小),
            skin_mask,
            float(肤色保护),
            float(强度),
        )

        img = img.clamp(0.0, 1.0)
        img = img.permute(0, 2, 3, 1).contiguous()
        return (img,)
