"""What do YOUR Paint demos look like, stroke by stroke?

A learned policy can only reproduce a drawing if the demos agree on where it goes. This prints, for every
stroke goal (DRAW_line, DRAW_zigzag_1..3), where the demos started and ended it and how much they differ.
    python -m argus.data strokes
"""
from __future__ import annotations

import glob
import os
import statistics
from typing import Dict, List, Tuple

from argus.actions.action_space import GRID, ActionType
from argus.core.planner import PlanError, align_goals, parse_instruction
from argus.data.audit import audit_episode
from argus.data.episode import Episode

SPREAD_OK = 4          # cells; demos that differ by more than this do not agree on one place


def collect_strokes(root: str) -> Dict[str, List[Tuple[str, int, int]]]:
    """goal -> [(episode id, start cell, end cell)] for every audit-valid Paint demo."""
    found: Dict[str, List[Tuple[str, int, int]]] = {}
    for meta in sorted(glob.glob(os.path.join(root, "*", "meta.json"))):
        ep = Episode.load(os.path.dirname(meta))
        if any(l == "BAD" for l, _ in audit_episode(ep)):
            continue
        try:
            plan = parse_instruction(ep.meta["instruction"])
            align_goals(plan, [s.action for s in ep.steps])
        except PlanError:
            continue
        for g, s in zip(plan.goals, ep.steps):
            if g.startswith("DRAW_") and s.action.type == ActionType.DRAG and s.action.end_cell is not None:
                found.setdefault(g, []).append((ep.meta["id"], s.action.cell, s.action.end_cell))
    return found


def _mean_rc(cells: List[int]) -> Tuple[float, float]:
    return statistics.mean(c // GRID for c in cells), statistics.mean(c % GRID for c in cells)


def _std_rc(cells: List[int]) -> Tuple[float, float]:
    if len(cells) < 2:
        return 0.0, 0.0
    return statistics.pstdev(c // GRID for c in cells), statistics.pstdev(c % GRID for c in cells)


def stroke_centres(root: str) -> Dict[str, Tuple[int, int]]:
    """goal -> (start cell, end cell) at the mean of the demos."""
    out = {}
    for g, items in collect_strokes(root).items():
        (sr, sc), (er, ec) = _mean_rc([s for _, s, _ in items]), _mean_rc([e for _, _, e in items])
        out[g] = (int(round(sr)) * GRID + int(round(sc)), int(round(er)) * GRID + int(round(ec)))
    return out


def report(root: str) -> Dict[str, dict]:
    found = collect_strokes(root)
    info: Dict[str, dict] = {}
    if not found:
        print("no Paint demos with strokes found")
        return info
    print(f"{'goal':15s}{'n':>3s}   {'start mean (row,col)':22s}{'+-':>8s}   {'end mean (row,col)':20s}{'+-':>8s}   verdict")
    for g in sorted(found):
        starts, ends = [s for _, s, _ in found[g]], [e for _, _, e in found[g]]
        (sr, sc), (er, ec) = _mean_rc(starts), _mean_rc(ends)
        (ssr, ssc), (ser, sec) = _std_rc(starts), _std_rc(ends)
        spread = max(ssr, ssc, ser, sec)
        verdict = "demos agree" if spread <= SPREAD_OK else "SPREAD: demos draw it in different places"
        info[g] = {"n": len(starts), "start": (sr, sc), "end": (er, ec), "spread": spread}
        print(f"{g:15s}{len(starts):3d}   ({sr:5.1f},{sc:5.1f}){'':8s}{max(ssr, ssc):6.1f}   ({er:5.1f},{ec:5.1f}){'':6s}{max(ser, sec):6.1f}   {verdict}")
    zig = [f"DRAW_zigzag_{k}" for k in (1, 2, 3)]
    if all(g in info for g in zig):
        print("\nzigzag: does each stroke start where the previous one ended?")
        for a, b in zip(zig, zig[1:]):
            gap = max(abs(info[a]["end"][0] - info[b]["start"][0]), abs(info[a]["end"][1] - info[b]["start"][1]))
            print(f"  end of {a[-1]} -> start of {b[-1]}: {gap:4.1f} cells apart  ({'chained' if gap <= SPREAD_OK else 'NOT chained'})")
    bad = [g for g, v in info.items() if v["spread"] > SPREAD_OK]
    print("\n" + (f"Demos disagree for: {' '.join(bad)}. A policy cannot learn ONE place from them; "
                  "record the strokes in the same place each time." if bad else
                  "All stroke goals are consistent across your demos."))
    return info
