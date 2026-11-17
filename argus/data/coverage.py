"""Which calculator buttons have enough demonstrations, and are the clicks consistent?

The policy learns "press button X" from the clicks in your demos. If a button has no (or
inconsistent) clicks, expressions using it will fail. This shows exactly what to record next.
"""
from __future__ import annotations

import glob
import os
import statistics
from typing import Dict, List, Tuple

from argus.actions.action_space import GRID
from argus.core.planner import BUTTONS, CORE_BUTTONS, EXTRA_BUTTONS, PlanError, align_goals, parse_instruction
from argus.data.audit import audit_episode
from argus.data.episode import Episode

MIN_EXAMPLES = 4     # clicks per button before we trust it
# A calculator key is about 6 cells wide and 3 tall on a 32x32 grid, so two correct clicks on the
# SAME key can legitimately differ by 5-6 cells. Beyond that they cannot be the same key.
MAX_SPREAD = 6


def _cheb(a: int, b: int, grid: int = GRID) -> int:
    return max(abs(a // grid - b // grid), abs(a % grid - b % grid))


def collect(root: str) -> Dict[str, List[Tuple[str, int]]]:
    """button -> [(episode id, clicked cell)] over every episode that passes the audit."""
    found: Dict[str, List[Tuple[str, int]]] = {b: [] for b in BUTTONS}
    for meta in sorted(glob.glob(os.path.join(root, "*", "meta.json"))):
        ep = Episode.load(os.path.dirname(meta))
        if any(l == "BAD" for l, _ in audit_episode(ep)):
            continue
        try:
            plan = parse_instruction(ep.meta["instruction"])
            align_goals(plan, [s.action for s in ep.steps])
        except PlanError:
            continue
        if plan.task != "calculator":
            continue
        for g, s in zip(plan.goals, ep.steps):
            if g.startswith("CLICK_"):
                found[g[6:]].append((ep.meta["id"], s.action.cell))
    return found


def _farthest(items: List[Tuple[str, int]]):
    return max(((a, b) for a in items for b in items), key=lambda p: _cheb(p[0][1], p[1][1]))


def _table(found, buttons, title):
    need, bad = [], []
    print(f"\n{title}")
    print(f"{'button':7s}{'clicks':>7s}{'spread':>8s}   status")
    for b in buttons:
        items = found[b]
        cells = [c for _, c in items]
        n = len(cells)
        if n == 0:
            print(f"{b:7s}{0:7d}{'-':>8s}   MISSING")
            need.append(b)
            continue
        spread = max(_cheb(x, y) for x in cells for y in cells)
        status = []
        if n < MIN_EXAMPLES:
            status.append(f"need {MIN_EXAMPLES - n} more")
            need.append(b)
        if spread > MAX_SPREAD:
            status.append("INCONSISTENT")
            bad.append(b)
        print(f"{b:7s}{n:7d}{spread:8d}   {', '.join(status) or 'ok'}")
        if spread > MAX_SPREAD:
            (e1, c1), (e2, c2) = _farthest(items)
            print(f"          farthest pair: {e1} cell {c1}  vs  {e2} cell {c2}")
            print(f"          check with: python -m argus.data overlay datasets\\raw\\{e1}   (and {e2})")
    return need, bad


def report(root: str) -> bool:
    """Prints both tables. Returns True only if every CORE button is covered and consistent."""
    found = collect(root)
    need, bad = _table(found, CORE_BUTTONS, "CORE buttons  0-9 + - * / =   (required)")
    if need:
        print(f"\nNot enough clicks for: {' '.join(need)}")
        print("   python -m argus.data session --plan press     (6 long episodes, every core button in each)")
    if bad:
        print(f"\nInconsistent clicks for: {' '.join(bad)}  (window moved/resized, or a different button was clicked)")
    if not need and not bad:
        print("\nAll core calculator buttons (0-9 + - * / =) have enough consistent demonstrations.")

    e_need, e_bad = _table(found, EXTRA_BUTTONS, "EXTRA buttons  . pct(%) neg(+/-) sqrt sq(x2) inv(1/x) C CE back   (optional)")
    if e_need or e_bad:
        print("\nExtra buttons are optional. To enable them:  python -m argus.data session --plan extras")
    else:
        print("\nAll extra buttons have enough consistent demonstrations too.")
    return not need and not bad
