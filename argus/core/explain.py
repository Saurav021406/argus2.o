"""Explain a failed run: for every calculator click, which key did the predicted cell actually land on?

Compares the cell the agent clicked with where YOUR demonstrations clicked each key (centroid of the
demonstrated cells). A key is about 6 cells wide and 3 tall, so distances are measured in key sizes.

    python -m argus.core explain runs\\<run folder> --instruction "7 squared"
"""
from __future__ import annotations

import json
import math
import os
import statistics
from typing import Dict, List, Tuple

from argus.actions.action_space import GRID
from argus.core.planner import parse_instruction
from argus.data.coverage import collect
from argus.data.strokes import stroke_centres

KEY_H, KEY_W = 3.2, 7.0          # approximate key size in grid cells (measured on a maximised calculator)
HALF = 0.5                       # farther than half a key from the centre is OFF the key


def key_centres(raw_root: str) -> Dict[str, Tuple[float, float]]:
    out = {}
    for b, items in collect(raw_root).items():
        if items:
            out[b] = (statistics.mean(c // GRID for _, c in items), statistics.mean(c % GRID for _, c in items))
    return out


def _dist(row: int, col: int, centre: Tuple[float, float]) -> float:
    return math.hypot((row - centre[0]) / KEY_H, (col - centre[1]) / KEY_W)


STROKE_FAR = 6          # cells (about 360 px wide / 220 px tall): farther than this from the demos is an outlier


def _explain_stroke(step, goal, a, strokes) -> List[str]:
    if a["type"] != "DRAG":
        return [f"step {step} {goal}: the model chose {a['type']} instead of drawing  <-- WRONG ACTION"]
    s_row, s_col = divmod(a["cell"], GRID)
    e_row, e_col = divmod(a["end_cell"], GRID)
    head = f"step {step} {goal}: from ({s_row},{s_col}) to ({e_row},{e_col})"
    if goal not in strokes:
        return [head + "; no demos of this stroke to compare with"]
    d_s, d_e = divmod(strokes[goal][0], GRID), divmod(strokes[goal][1], GRID)
    off_s = max(abs(s_row - d_s[0]), abs(s_col - d_s[1]))
    off_e = max(abs(e_row - d_e[0]), abs(e_col - d_e[1]))
    note = f"; demos: from {d_s} to {d_e}; start {off_s} cells off, end {off_e} cells off"
    if max(off_s, off_e) > STROKE_FAR:
        return [head + note + "  <-- FAR FROM THE DEMOS"]
    return [head + note + "  ok"]


def explain_run(trace_dir: str, instruction: str, raw_root: str = "datasets/raw") -> List[str]:
    plan = parse_instruction(instruction)
    centres = key_centres(raw_root)
    strokes = stroke_centres(raw_root)
    lines: List[str] = []
    with open(os.path.join(trace_dir, "trace.jsonl"), encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            step, a = r["step"], r["action"]
            goal = plan.goals[step] if step < len(plan.goals) else "DONE"
            if goal.startswith("DRAW_"):
                lines += _explain_stroke(step, goal, a, strokes)
                continue
            if not goal.startswith("CLICK_"):
                continue
            want = goal[6:]
            if a["type"] != "CLICK":
                lines.append(f"step {step} {goal}: the model chose {a['type']} instead of clicking  <-- WRONG ACTION")
                continue
            row, col = divmod(a["cell"], GRID)
            if want not in centres:
                lines.append(f"step {step} {goal}: cell ({row},{col}); no demonstrations of '{want}' to compare with")
                continue
            near = min(centres, key=lambda b: _dist(row, col, centres[b]))
            own = _dist(row, col, centres[want])
            if near == want and own <= HALF:
                lines.append(f"step {step} {goal}: cell ({row},{col}) is on '{want}'  ok ({own:.1f} key-sizes from its centre)")
            elif near == want:
                lines.append(f"step {step} {goal}: cell ({row},{col}) is nearest to '{want}' but {own:.1f} key-sizes from its centre"
                             f"  <-- EDGE (probably outside the key: it may press a neighbour or nothing)")
            else:
                cr, cc = centres[want]
                lines.append(f"step {step} {goal}: cell ({row},{col}) is nearest to key '{near}', not '{want}'"
                             f"  <-- WRONG KEY ('{want}' was clicked at ({cr:.1f},{cc:.1f}) in your demos, {own:.1f} key-sizes away)")
    return lines
