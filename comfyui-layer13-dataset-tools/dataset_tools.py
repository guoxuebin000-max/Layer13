import math
from typing import List, Optional

import torch
import torch.nn.functional as F

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except Exception as exc:
    MTCNN = None
    InceptionResnetV1 = None
    _FACENET_IMPORT_ERROR = exc
else:
    _FACENET_IMPORT_ERROR = None


_FACE_MODELS = {}


def _get_face_models():
    if _FACENET_IMPORT_ERROR is not None:
        raise RuntimeError(
            "缺少 facenet_pytorch 依赖，请先安装：pip install facenet-pytorch"
        ) from _FACENET_IMPORT_ERROR
    # MTCNN 不支持 MPS，统一放 CPU，嵌入模型用 CPU 更稳
    device = torch.device("cpu")
    key = str(device)
    if key not in _FACE_MODELS:
        mtcnn = MTCNN(keep_all=False, device=device)
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        _FACE_MODELS[key] = (mtcnn, resnet)
    return _FACE_MODELS[key]


def _image_to_pil(image_tensor):
    # image_tensor: [H,W,3] float 0..1
    import numpy as np
    from PIL import Image

    img = image_tensor.clamp(0, 1).mul(255).byte().cpu().numpy()
    return Image.fromarray(img)


def _center_crop_tensor(image_tensor, size=160):
    # image_tensor: [H,W,3], float 0..1
    h, w, _ = image_tensor.shape
    side = min(h, w)
    y0 = max((h - side) // 2, 0)
    x0 = max((w - side) // 2, 0)
    cropped = image_tensor[y0 : y0 + side, x0 : x0 + side]
    cropped = cropped.permute(2, 0, 1).unsqueeze(0)
    cropped = F.interpolate(cropped, (size, size), mode="bilinear", align_corners=False)
    return cropped


def _extract_face_embedding(image_tensor):
    # image_tensor: [H,W,3] float 0..1
    mtcnn, resnet = _get_face_models()
    pil_img = _image_to_pil(image_tensor)
    face = mtcnn(pil_img)
    if face is None:
        face = _center_crop_tensor(image_tensor, size=160)
    else:
        face = face.unsqueeze(0)
    with torch.no_grad():
        emb = resnet(face)
    emb = F.normalize(emb, dim=1)
    return emb.squeeze(0)


def _cosine(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


class Layer13MergeImages:
    @classmethod
    def INPUT_TYPES(cls):
        required = {"image1": ("IMAGE",)}
        optional = {f"image{i}": ("IMAGE",) for i in range(2, 31)}
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "Layer13"

    def merge(self, **kwargs):
        images = []
        for key in [f"image{i}" for i in range(1, 31)]:
            img = kwargs.get(key)
            if img is None:
                continue
            images.append(img)
        if not images:
            raise RuntimeError("未收到任何图片输入")
        batch = torch.cat(images, dim=0)
        return (batch,)


class Layer13FaceFilterDedup:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "images": ("IMAGE",),
            "相似度阈值": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            "去重阈值": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
        }
        optional = {
            "参考图": ("IMAGE",),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("筛选结果", "保留索引", "相似度列表")
    FUNCTION = "filter"
    CATEGORY = "Layer13"

    def filter(self, **kwargs):
        images = kwargs["images"]
        ref_images = kwargs.get("参考图")
        ref_threshold = float(kwargs["相似度阈值"])
        dedup_threshold = float(kwargs["去重阈值"])

        batch = images
        b, h, w, c = batch.shape

        if ref_images is not None:
            ref = ref_images[0]
        else:
            ref = batch[0]

        ref_emb = _extract_face_embedding(ref)

        embeddings: List[torch.Tensor] = []
        sims: List[float] = []
        for i in range(b):
            emb = _extract_face_embedding(batch[i])
            embeddings.append(emb)
            sims.append(_cosine(emb, ref_emb))

        # 先过参考图相似度阈值
        candidates = [i for i, s in enumerate(sims) if s >= ref_threshold]

        kept = []
        kept_embs = []
        for idx in candidates:
            emb = embeddings[idx]
            if not kept_embs:
                kept.append(idx)
                kept_embs.append(emb)
                continue
            max_sim = max(_cosine(emb, k) for k in kept_embs)
            if max_sim < dedup_threshold:
                kept.append(idx)
                kept_embs.append(emb)

        if not kept:
            kept = [int(torch.argmax(torch.tensor(sims)).item())]

        out = batch[kept]
        out = out.clamp(0.0, 1.0)
        kept_str = ",".join(str(i) for i in kept)
        sim_str = ",".join(f"{s:.3f}" for s in sims)
        return (out, kept_str, sim_str)

