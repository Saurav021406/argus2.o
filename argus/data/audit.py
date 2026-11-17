"""Strict per-task audit of recorded episodes.

validate_episode() checks that files are consistent. audit_episode() checks that
the *content* looks like the task it claims to be (e.g. a calculator episode must
contain clicks, and must not contain stray caps_lock / alt+tab).
"""
from __future__ import annotations

import glob
import os
from typing import List, Tuple

from argus.actions.action_space import ActionType
from argus.core.planner import PlanError, align_goals, parse_instruction
from argus.data.episode import Episode
from argus.data.validator import validate_episode

ALLOWED_KEYS = {"win", "enter"}
APP_WORD = {"notepad": "notepad", "calculator": "calc", "paint": "paint"}
MAX_SECONDS = 40.0

Issue = Tuple[str, str]   # ("BAD" | "WARN", message)


def audit_episode(ep: Episode) -> List[Issue]:
    out: List[Issue] = [("BAD", m) for m in validate_episode(ep)]
    steps = ep.steps
    if len(steps) < 4:
        return out
    task, app = ep.meta.get("task", ""), ep.meta.get("app", "")
    A = [s.action for s in steps]

    stray = [a.key for a in A if a.type == ActionType.KEY and a.key not in ALLOWED_KEYS]
    if stray:
        out.append(("BAD", f"unexpected keys {stray}"))

    ok_start = (A[0].type == ActionType.KEY and A[0].key == "win"
                and A[1].type == ActionType.TYPE
                and A[2].type == ActionType.KEY and A[2].key == "enter")
    if not ok_start:
        out.append(("BAD", "does not start with win -> TYPE -> enter"))
    elif app in APP_WORD and APP_WORD[app] not in (A[1].text or "").lower():
        out.append(("BAD", f"typed {A[1].text!r} but app is {app!r}"))

    try:
        align_goals(parse_instruction(ep.meta.get("instruction", "")), A)
    except PlanError as e:
        out.append(("BAD", f"does not match its plan: {e}"))
    if sum(a.type == ActionType.TYPE for a in A) > 1:
        out.append(("BAD", "more than one TYPE action"))
    if any(a.type == ActionType.SCROLL for a in A):
        out.append(("BAD", "unexpected SCROLL"))
    if task == "paint" and any(a.type == ActionType.CLICK for a in A):
        out.append(("WARN", "extra click(s) in paint episode"))

    if steps[-1].t > MAX_SECONDS:
        out.append(("WARN", f"episode took {steps[-1].t:.0f}s"))
    return out


def audit_root(root: str) -> int:
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "meta.json")))
    per_task, bad_dirs = {}, []
    for d in dirs:
        ep = Episode.load(d)
        issues = audit_episode(ep)
        t = ep.meta.get("task") or "?"
        c = per_task.setdefault(t, {"n": 0, "good": 0, "steps": 0})
        c["n"] += 1
        c["steps"] += len(ep.steps)
        bad = any(l == "BAD" for l, _ in issues)
        c["good"] += (not bad)
        if issues:
            print(f"{os.path.basename(d)}  [{ep.meta.get('instruction')}]")
            for lvl, msg in issues:
                print(f"    {lvl}: {msg}")
        if bad:
            bad_dirs.append(d)
    print(f"\n{len(dirs)} episodes")
    for t, c in sorted(per_task.items()):
        print(f"  {t:12s} total={c['n']:3d}  good={c['good']:3d}  avg_steps={c['steps'] / c['n']:.1f}")
    if bad_dirs:
        print("\nBAD episodes. Delete with (nothing is deleted automatically):")
        for d in bad_dirs:
            print(f"  Remove-Item -Recurse {d}")
    else:
        print("\nNo BAD episodes.")
    return len(bad_dirs)
