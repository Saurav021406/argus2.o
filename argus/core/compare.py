"""Compare checkpoints honestly: success counts with confidence intervals and an exact significance test.

    python -m argus.core compare --levels 1,2,5,6 --aim model --base bc_v5.pt

Repeated runs of the same question are collapsed (the agent is deterministic, so a repeat adds nothing).
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Exact p-value that two checkpoints have the same success rate."""
    total_k, total_n = k1 + k2, n1 + n2

    def prob(a):
        return math.comb(n1, a) * math.comb(n2, total_k - a) / math.comb(total_n, total_k)

    p0 = prob(k1)
    lo, hi = max(0, total_k - n2), min(n1, total_k)
    return min(1.0, sum(prob(a) for a in range(lo, hi + 1) if prob(a) <= p0 * (1 + 1e-9)))


def load_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def summarise(rows: List[dict], levels: Optional[set] = None, aim: Optional[str] = None) -> Dict[str, List[int]]:
    """ckpt name -> [successes, distinct questions]; the latest run of a question wins."""
    latest: Dict[tuple, dict] = {}
    for r in rows:
        if r.get("success") is None or r.get("level") is None:
            continue
        if levels and r["level"] not in levels:
            continue
        if aim and r.get("aim", "?") != aim:
            continue
        latest[(os.path.basename(r.get("ckpt", "?")), r["instruction"])] = r
    out: Dict[str, List[int]] = {}
    for (ck, _), r in latest.items():
        c = out.setdefault(ck, [0, 0])
        c[0] += bool(r["success"])
        c[1] += 1
    return out


def table(summary: Dict[str, List[int]], base: Optional[str] = None) -> List[str]:
    lines = [f"{'checkpoint':16s}{'passed':>9s}{'rate':>7s}   95% CI" + (f"        p vs {base}" if base else "")]
    for ck, (k, n) in sorted(summary.items()):
        lo, hi = wilson(k, n)
        line = f"{ck:16s}{f'{k}/{n}':>9s}{k / n:>7.0%}   [{lo:.0%}, {hi:.0%}]"
        if base and base in summary and ck != base:
            line += f"      p = {fisher_two_sided(k, n, *summary[base]):.3f}"
        lines.append(line)
    return lines
