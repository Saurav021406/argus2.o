"""Pre-compute frozen features once, so training never touches the vision model.

Only the frames that an action points at are encoded (about 8 per episode).
Episodes that fail the strict audit are skipped, so bad demos never reach training.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import torch
from PIL import Image

from argus.data.audit import audit_episode
from argus.data.episode import Episode


def cache_features(raw_root: str, out_dir: str, vision, text_enc=None, strict: bool = True, batch: int = 8) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    instr_path = os.path.join(out_dir, "instructions.pt")
    instr = torch.load(instr_path) if os.path.exists(instr_path) else {}
    stats = {"cached": 0, "skipped_bad": 0, "already": 0}

    for meta_path in sorted(glob.glob(os.path.join(raw_root, "*", "meta.json"))):
        ep = Episode.load(os.path.dirname(meta_path))
        eid = ep.meta["id"]
        out = os.path.join(out_dir, f"{eid}.pt")
        if os.path.exists(out):
            stats["already"] += 1
            continue
        issues = audit_episode(ep)
        if strict and any(l == "BAD" for l, _ in issues):
            print(f"skip {eid} ({ep.meta.get('instruction')}): {[m for l, m in issues if l == 'BAD'][0]}")
            stats["skipped_bad"] += 1
            continue

        rows = sorted({s.frame for s in ep.steps})
        patches, clss = [], []
        for i in range(0, len(rows), batch):
            imgs = [Image.open(ep.frame_path(r)) for r in rows[i:i + batch]]
            p, c = vision.encode(imgs)
            patches.append(p)
            clss.append(c)
        text = ep.meta["instruction"]
        if text_enc is not None and text not in instr:      # optional; the policy no longer uses it
            instr[text] = text_enc.encode([text])[0]
        torch.save({
            "id": eid, "instruction": text, "task": ep.meta.get("task", ""), "app": ep.meta.get("app", ""),
            "rows": rows, "patch": torch.cat(patches), "cls": torch.cat(clss),
            "steps": [{"frame": s.frame, "action": s.action.to_dict()} for s in ep.steps],
            "source": ep.meta.get("source", "human"),
            "exec_steps": [{"frame": s.frame, "action": s.action.to_dict()} for s in ep.exec_steps],
        }, out)
        stats["cached"] += 1
        print(f"cached {eid}  {text!r}  ({len(rows)} frames)")

    if instr:
        torch.save(instr, instr_path)
    print(stats)
    return stats
