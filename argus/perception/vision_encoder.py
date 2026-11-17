"""Frozen vision encoder (DINOv2-small) that keeps SPATIAL patch features.

A single global embedding cannot say *where* a button is. So we keep the whole
patch grid: at 448x280 input and patch size 14 we get a 20 x 32 grid of 384-d
vectors (the width, 32, matches the 32 columns of the action grid).
"""
from __future__ import annotations

import hashlib
from typing import List, Tuple

import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VisionEncoder:
    def __init__(self, name: str = "facebook/dinov2-small", size: Tuple[int, int] = (448, 280),
                 device: str | None = None):
        from transformers import AutoModel

        self.size = size                      # (width, height), both multiples of 14
        self.gw, self.gh = size[0] // 14, size[1] // 14
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.dim = self.model.config.hidden_size
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        arr = []
        for im in images:
            im = im.convert("RGB").resize(self.size, Image.BILINEAR)
            t = torch.from_numpy(__import__("numpy").asarray(im).copy()).permute(2, 0, 1).float() / 255.0
            arr.append(t)
        x = torch.stack(arr)
        return (x - self._mean) / self._std

    @torch.no_grad()
    def encode(self, images: List[Image.Image]):
        """Returns (patch [B, gh, gw, C] fp16 cpu, cls [B, C] fp16 cpu)."""
        x = self.preprocess(images).to(self.device)
        if self.device == "cuda":
            x = x.half()
        h = self.model(pixel_values=x).last_hidden_state          # [B, 1 + gh*gw, C]
        cls = h[:, 0]
        patch = h[:, 1:].reshape(h.shape[0], self.gh, self.gw, -1)
        return patch.half().cpu(), cls.half().cpu()


class RandomVisionEncoder:
    """TEST ONLY: deterministic pseudo-features, no model download.

    Features depend on the image content (hash-seeded), so identical frames give
    identical features. Lets us test the pipeline without network access.
    """

    def __init__(self, dim: int = 384, gh: int = 20, gw: int = 32):
        self.dim, self.gh, self.gw = dim, gh, gw

    def encode(self, images: List[Image.Image]):
        P, C = [], []
        for im in images:
            seed = int(hashlib.md5(im.convert("RGB").resize((16, 16)).tobytes()).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed)
            P.append(torch.randn(self.gh, self.gw, self.dim, generator=g))
            C.append(torch.randn(self.dim, generator=g))
        return torch.stack(P).half(), torch.stack(C).half()
