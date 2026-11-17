"""Frozen instruction encoder (all-MiniLM-L6-v2, 384-d, mean pooled)."""
from __future__ import annotations

import hashlib
from typing import List

import torch


class InstructionEncoder:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        b = self.tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
        h = self.model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(e, dim=-1).cpu()


class RandomInstructionEncoder:
    """TEST ONLY: hash-seeded vectors."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str]) -> torch.Tensor:
        out = []
        for t in texts:
            g = torch.Generator().manual_seed(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
            out.append(torch.nn.functional.normalize(torch.randn(self.dim, generator=g), dim=0))
        return torch.stack(out)
