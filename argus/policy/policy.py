"""Argus policy: frozen features -> causal temporal transformer -> action heads.

Input per sample
  current sub-goal id from the planner (1 token)
  last T steps: frame CLS + previous action   (T tokens, causal)
  current frame patch grid           (used by the location heatmap head)
Output
  type / key / text logits, and two 32x32 heatmaps (click-or-drag-start, drag-end)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from argus.actions.action_space import ACTION_TYPES, GRID
from argus.core.planner import N_GOALS

N_TYPES = len(ACTION_TYPES)


class ArgusPolicy(nn.Module):
    def __init__(self, n_keys: int, n_texts: int, vis_dim: int = 384, n_goals: int = N_GOALS,
                 d: int = 256, layers: int = 4, heads: int = 4, T: int = 6, grid: int = GRID,
                 gh: int = 20, gw: int = 32, dropout: float = 0.1, frame_drop: float = 0.5):
        super().__init__()
        self.cfg = dict(n_keys=n_keys, n_texts=n_texts, vis_dim=vis_dim, n_goals=n_goals, d=d,
                        layers=layers, heads=heads, T=T, grid=grid, gh=gh, gw=gw, dropout=dropout, frame_drop=frame_drop)
        self.grid, self.T, self.frame_drop = grid, T, frame_drop
        self.cls_proj = nn.Linear(vis_dim, d)
        self.goal_emb = nn.Embedding(n_goals, d)               # which button / step is next
        self.type_emb = nn.Embedding(N_TYPES + 1, d)          # index 0 = "no previous action"
        self.key_emb = nn.Embedding(n_keys + 1, d)
        self.text_emb = nn.Embedding(n_texts + 1, d)
        self.cell_emb = nn.Embedding(grid * grid + 1, d)
        self.end_emb = nn.Embedding(grid * grid + 1, d)        # end cell of a previous DRAG
        self.pos = nn.Parameter(torch.randn(T + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.type_head = nn.Linear(d, N_TYPES)
        self.key_head = nn.Linear(d, max(n_keys, 1))
        self.text_head = nn.Linear(d, max(n_texts, 1))
        # location heatmap head
        self.patch_proj = nn.Linear(vis_dim, d)
        self.patch_pos = nn.Parameter(torch.randn(gh, gw, d) * 0.02)
        self.film = nn.Linear(d, 2 * d)
        # "button X is at cell Y" as a pure lookup keyed by the goal (no pixels, no history). With frames
        # that differ in every demo (the calculator display) a net can otherwise memorise frame->cell and
        # ignore the goal; this table plus frame dropout makes the goal the easy, dominant route.
        self.goal_table = nn.Embedding(n_goals, 2 * grid * grid)
        nn.init.zeros_(self.goal_table.weight)
        self.loc_conv = nn.Sequential(
            nn.Conv2d(d, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 2, 1),
        )
        self.register_buffer("causal", torch.triu(torch.ones(T + 1, T + 1, dtype=torch.bool), 1), persistent=False)

    def forward(self, b: dict) -> dict:
        B = b["cls"].shape[0]
        if self.training and self.frame_drop > 0:             # blank the screen for a random half of samples
            keep = (torch.rand(B, device=b["cls"].device) >= self.frame_drop).float()
            b = dict(b, cls=b["cls"] * keep.view(B, 1, 1), patch=b["patch"] * keep.view(B, 1, 1, 1))
        obs = (self.cls_proj(b["cls"]) + self.type_emb(b["ptype"]) + self.key_emb(b["pkey"])
               + self.text_emb(b["ptext"]) + self.cell_emb(b["pcell"]) + self.end_emb(b["pend"]))
        x = torch.cat([self.goal_emb(b["goal"]).unsqueeze(1), obs], 1) + self.pos
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), b["mask"]], 1)
        h = self.norm(self.encoder(x, mask=self.causal, src_key_padding_mask=~valid)[:, -1])

        p = self.patch_proj(b["patch"]) + self.patch_pos                    # [B, gh, gw, d]
        g, bt = self.film(h).chunk(2, dim=-1)
        p = p * (1 + g[:, None, None]) + bt[:, None, None]
        l = self.loc_conv(p.permute(0, 3, 1, 2))                            # [B, 2, gh, gw]
        l = F.interpolate(l, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        lookup = self.goal_table(b["goal"]).view(B, 2, -1)                  # [B, 2, grid*grid]
        return {"type": self.type_head(h), "key": self.key_head(h), "text": self.text_head(h),
                "loc": l.flatten(2) + lookup, "loc_goal": lookup}


def soft_targets(cells: torch.Tensor, grid: int, sigma: float) -> torch.Tensor:
    """Gaussian-smoothed one-hot over the grid: a neighbouring cell is nearly as good."""
    r, c = cells // grid, cells % grid
    rr = torch.arange(grid, device=cells.device).view(1, grid, 1)
    cc = torch.arange(grid, device=cells.device).view(1, 1, grid)
    d2 = (rr - r.view(-1, 1, 1)) ** 2 + (cc - c.view(-1, 1, 1)) ** 2
    t = torch.exp(-d2.float() / (2 * sigma ** 2)).flatten(1)
    return t / t.sum(1, keepdim=True)


def bc_loss(out: dict, b: dict, grid: int = GRID, sigma: float = 1.0):
    zero = out["type"].sum() * 0
    parts = {"type": F.cross_entropy(out["type"], b["t_type"])}
    for name, ch, key in (("loc", 0, "t_cell"), ("end", 1, "t_end")):
        m = b[key] >= 0
        if m.any():
            soft = soft_targets(b[key][m], grid, sigma)
            parts[name] = F.cross_entropy(out["loc"][:, ch][m], soft)
            # the goal lookup must solve the task ON ITS OWN, otherwise the frame path memorises first
            parts[name + "_goal"] = F.cross_entropy(out["loc_goal"][:, ch][m], soft)
        else:
            parts[name] = parts[name + "_goal"] = zero
    for name, key in (("key", "t_key"), ("text", "t_text")):
        m = b[key] >= 0
        parts[name] = F.cross_entropy(out[name][m], b[key][m]) if m.any() else zero
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}
