"""Replay an episode on the live desktop, and render overlays for visual QA."""
from __future__ import annotations

import os
import time

from argus.actions.action_space import ActionType, cell_to_xy
from argus.actions.executor import Executor
from argus.data.episode import Episode


def replay(ep: Episode, executor: Executor, countdown: int = 3, max_gap: float = 2.0) -> int:
    """Re-execute recorded actions. Returns number of failed actions."""
    for n in range(countdown, 0, -1):
        print(f"Replay starts in {n}... (move mouse to a screen corner to abort)")
        time.sleep(1)
    failures, prev_t = 0, None
    for s in ep.steps:
        if prev_t is not None:
            time.sleep(min(max(s.t - prev_t, 0.2), max_gap))
        prev_t = s.t
        res = executor.execute(s.action)
        print(f"step {s.step:02d} {s.action.type.value:6s} -> {res.reason}")
        if not res.ok:
            failures += 1
    return failures


def render_overlays(ep: Episode, out_dir: str) -> int:
    """Draw each action on the frame the human saw. Open the PNGs to check labels."""
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    fw, fh, g = ep.meta["frame_w"], ep.meta["frame_h"], ep.meta["grid"]
    for s in ep.steps:
        img = Image.open(ep.frame_path(s.frame)).convert("RGB")
        d = ImageDraw.Draw(img)
        a = s.action
        label = a.type.value
        if a.cell is not None:
            x, y = cell_to_xy(a.cell, fw, fh, g)
            d.ellipse([x - 12, y - 12, x + 12, y + 12], outline=(255, 0, 0), width=3)
        if a.type == ActionType.DRAG and a.end_cell is not None:
            x1, y1 = cell_to_xy(a.end_cell, fw, fh, g)
            d.line([x, y, x1, y1], fill=(255, 0, 0), width=3)
        if a.text:
            label += f" '{a.text}'"
        if a.key:
            label += f" [{a.key}]"
        d.rectangle([0, 0, fw, 22], fill=(0, 0, 0))
        d.text((6, 5), f"step {s.step} | {ep.meta['instruction']} | {label}", fill=(255, 255, 255))
        img.save(os.path.join(out_dir, f"step_{s.step:03d}.png"))
    return len(ep.steps)
