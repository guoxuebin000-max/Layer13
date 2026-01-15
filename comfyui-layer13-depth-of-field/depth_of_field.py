import torch
import torch.nn.functional as F


def _smoothstep(edge0, edge1, x):
    t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _to_luma(img):
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _depth_from_input(depth):
    # Accept: (B,H,W,3), (B,H,W,1), (B,H,W), (B,1,H,W)
    if depth.dim() == 4 and depth.shape[-1] in (1, 3):
        if depth.shape[-1] == 3:
            d = depth.permute(0, 3, 1, 2)
            d = _to_luma(d)
        else:
            d = depth[..., 0].unsqueeze(1)
    elif depth.dim() == 4 and depth.shape[1] in (1, 3):
        if depth.shape[1] == 3:
            d = _to_luma(depth)
        else:
            d = depth
    elif depth.dim() == 3:
        d = depth.unsqueeze(1)
    else:
        raise ValueError("深度图需为 IMAGE (B,H,W,3)/(B,H,W,1)/(B,H,W)/(B,1,H,W)")

    d = d.float()
    if d.max() > 1.0:
        d = d / 255.0
    return d.clamp(0.0, 1.0)


def _gaussian_kernel1d(sigma, device, dtype):
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = int(max(1, round(sigma * 3.0)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    b, c, h, w = img.shape
    kernel = _gaussian_kernel1d(sigma, img.device, img.dtype)
    k = kernel.numel()
    pad = k // 2
    kernel_x = kernel.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    kernel_y = kernel.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    out = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(out, kernel_x, groups=c)
    out = F.conv2d(out, kernel_y, groups=c)
    return out


def _edge_aware_depth_smooth(depth, guide_luma):
    # Simple edge-aware smoothing: blur depth then restore across image edges
    depth_blur = _gaussian_blur(depth, 2.0)
    dx = F.pad(guide_luma[:, :, :, 1:] - guide_luma[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(guide_luma[:, :, 1:, :] - guide_luma[:, :, :-1, :], (0, 0, 0, 1))
    edge = torch.sqrt(dx * dx + dy * dy + 1e-8)
    w = torch.exp(-edge * 50.0).clamp(0.0, 1.0)
    return (depth_blur * w + depth * (1.0 - w)).clamp(0.0, 1.0)


def _multi_scale_blur(img, sigma_levels):
    return [img if s <= 0 else _gaussian_blur(img, s) for s in sigma_levels]


def _blend_blur_levels(levels, blur_sigma, max_sigma):
    b, c, h, w = levels[0].shape
    stack = torch.stack(levels, dim=1)  # [B,L,C,H,W]
    L = stack.shape[1]
    t = torch.clamp(blur_sigma / max(max_sigma, 1e-6) * (L - 1), 0.0, L - 1 - 1e-6)
    idx0 = t.floor().long()
    idx1 = (idx0 + 1).clamp(max=L - 1)
    w1 = (t - idx0.float()).clamp(0.0, 1.0)
    idx0 = idx0.unsqueeze(2).expand(-1, 1, c, -1, -1)
    idx1 = idx1.unsqueeze(2).expand(-1, 1, c, -1, -1)
    lvl0 = torch.gather(stack, 1, idx0).squeeze(1)
    lvl1 = torch.gather(stack, 1, idx1).squeeze(1)
    w1 = w1.expand(-1, c, -1, -1)
    return lvl0 * (1.0 - w1) + lvl1 * w1


class Layer13DepthOfFieldRealistic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "深度图": ("IMAGE",),
                "对焦点X": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "对焦点Y": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "焦外过渡宽度": ("FLOAT", {"default": 0.50, "min": 0.01, "max": 0.5, "step": 0.01}),
                "最大模糊半径": ("INT", {"default": 1, "min": 0, "max": 40}),
                "景深曲线": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 3.0, "step": 0.05}),
                "深度白=近": ("BOOLEAN", {"default": True}),
                "前景保护": ("BOOLEAN", {"default": True}),
                "边缘修复强度": ("INT", {"default": 2, "min": 0, "max": 12}),
                "遮罩羽化": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "亮部光斑增强": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("图像", "CoC遮罩", "前景遮罩")
    FUNCTION = "apply"
    CATEGORY = "Layer13"

    def apply(self, **kwargs):
        def pick(cn, en, default):
            if cn in kwargs:
                return kwargs[cn]
            if en in kwargs:
                return kwargs[en]
            return default

        图像 = pick("图像", "image", None)
        深度图 = pick("深度图", "depth", None)
        对焦点X = float(pick("对焦点X", "focus_x", 0.5))
        对焦点Y = float(pick("对焦点Y", "focus_y", 0.55))
        焦外过渡宽度 = float(pick("焦外过渡宽度", "focus_falloff", 0.50))
        最大模糊半径 = int(pick("最大模糊半径", "max_blur_radius", 1))
        景深曲线 = float(pick("景深曲线", "dof_curve", 3.0))
        深度白近 = bool(pick("深度白=近", "depth_white_near", True))
        前景保护 = bool(pick("前景保护", "foreground_protect", True))
        边缘修复强度 = int(pick("边缘修复强度", "edge_fix_strength", 2))
        遮罩羽化 = float(pick("遮罩羽化", "mask_feather", 2.0))
        亮部光斑增强 = float(pick("亮部光斑增强", "highlight_bokeh", 0.0))

        if 图像 is None or 深度图 is None:
            raise ValueError("图像/深度图 不能为空")

        img = 图像
        if img.dim() != 4 or img.shape[-1] != 3:
            raise ValueError("图像需为 IMAGE (B,H,W,3)")
        img = img.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()

        depth = _depth_from_input(深度图)
        if not 深度白近:
            depth = 1.0 - depth

        # Depth pre-process: small blur + edge-aware smooth
        depth = _gaussian_blur(depth, 1.0)
        guide_luma = _to_luma(img)
        depth = _edge_aware_depth_smooth(depth, guide_luma)

        # Focus depth from focus point
        _, _, h, w = depth.shape
        fx = int(torch.clamp(torch.tensor(对焦点X), 0.0, 1.0).item() * (w - 1))
        fy = int(torch.clamp(torch.tensor(对焦点Y), 0.0, 1.0).item() * (h - 1))
        focus_depth = depth[:, :, fy:fy + 1, fx:fx + 1]

        # CoC map
        coc = torch.abs(depth - focus_depth) / max(焦外过渡宽度, 1e-4)
        coc = torch.clamp(coc, 0.0, 1.0)
        coc = torch.pow(coc, 1.0 / max(景深曲线, 1e-3))
        coc = coc.clamp(0.0, 1.0)

        # Multi-scale Gaussian blur pyramid
        sigma_levels = [0.0, 1.5, 3.0, 6.0, 10.0, 16.0]
        max_sigma = float(max(最大模糊半径, 0))
        blur_sigma = coc * max_sigma
        blur_levels = _multi_scale_blur(img, sigma_levels)
        blurred = _blend_blur_levels(blur_levels, blur_sigma, max(sigma_levels))

        # Highlight bokeh boost
        if 亮部光斑增强 > 1e-6:
            luma = _to_luma(img)
            hl = _smoothstep(0.7, 1.0, luma) * coc
            hl_blur = _gaussian_blur(img * hl, 3.0)
            blurred = blurred + hl_blur * 亮部光斑增强

        # Foreground alpha with repair chain
        alpha = 1.0 - _smoothstep(0.0, 1.0, coc)
        if 遮罩羽化 > 0:
            alpha = _gaussian_blur(alpha, max(0.5, 遮罩羽化))
        if 边缘修复强度 > 0:
            k = int(边缘修复强度) * 2 + 1
            alpha = F.max_pool2d(alpha, kernel_size=k, stride=1, padding=边缘修复强度)
        if 遮罩羽化 > 0:
            alpha = _gaussian_blur(alpha, max(0.5, 遮罩羽化 * 0.6))
        alpha = alpha.clamp(0.0, 1.0)

        if 前景保护:
            out = alpha * img + (1.0 - alpha) * blurred
        else:
            out = torch.lerp(img, blurred, coc)

        out = out.clamp(0.0, 1.0).permute(0, 2, 3, 1).contiguous()
        coc_img = coc.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).contiguous()
        alpha_img = alpha.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).contiguous()
        return (out, coc_img, alpha_img)
