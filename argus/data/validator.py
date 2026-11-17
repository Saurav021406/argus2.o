"""Sanity-check an episode before it goes into a training set."""
from __future__ import annotations

import os
from typing import List

from argus.actions.action_space import ActionType
from argus.data.episode import Episode

MIN_FRAMES = 3


def validate_episode(ep: Episode) -> List[str]:
    issues: List[str] = []
    m = ep.meta
    for k in ("instruction", "screen_w", "screen_h", "grid", "fps"):
        if not m.get(k):
            issues.append(f"meta missing {k}")
    if len(ep.frames) < MIN_FRAMES:
        issues.append(f"only {len(ep.frames)} frames")
    for f in ep.frames:
        if not os.path.exists(os.path.join(ep.path, f["file"])):
            issues.append(f"missing frame file {f['file']}")
            break
    ts = [f["t"] for f in ep.frames]
    if any(b < a for a, b in zip(ts, ts[1:])):
        issues.append("frame timestamps not monotonic")
    if not ep.steps:
        issues.append("no actions")
        return issues
    n = m.get("grid", 32) ** 2
    last_frame = -1
    for s in ep.steps:
        if not 0 <= s.frame < len(ep.frames):
            issues.append(f"step {s.step}: frame {s.frame} out of range")
        if s.frame < last_frame:
            issues.append(f"step {s.step}: frame index goes backwards")
        last_frame = max(last_frame, s.frame)
        for c in (s.action.cell, s.action.end_cell):
            if c is not None and not 0 <= c < n:
                issues.append(f"step {s.step}: cell {c} outside grid")
    if ep.steps[-1].action.type != ActionType.DONE:
        issues.append("last action is not DONE")
    if len(ep.steps) < 2:
        issues.append("episode has no real actions (only DONE)")
    return issues
