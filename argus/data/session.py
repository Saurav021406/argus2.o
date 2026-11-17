"""Guided batch recording: shows one instruction at a time and records it.

    python -m argus.data session --out datasets/raw            # whole first batch
    python -m argus.data session --out datasets/raw --start 12 # resume at item 12
"""
from __future__ import annotations

import shutil
from typing import List, Tuple

from argus.data.validator import validate_episode

# (task, app, instruction, hint, min_steps)  -- every episode starts from a clean desktop
PLAN_FIRST_BATCH: List[Tuple[str, str, str, str, int]] = []

_OPEN = "Win key -> type the app name -> Enter -> wait until it is open -> F10"
for app, phrasings in {
    "notepad": ["Open Notepad", "Launch Notepad", "Start Notepad"],
    "calculator": ["Open Calculator", "Launch Calculator", "Start Calculator"],
    "paint": ["Open Paint", "Launch Paint", "Start Paint"],
}.items():
    for p in phrasings:
        PLAN_FIRST_BATCH.append(("open_app", app, p, _OPEN, 4))

_CALC = "Win -> type calculator -> Enter -> wait -> click the buttons WITH THE MOUSE -> F10"
for expr, clicks in [
    ("Calculate 2 times 4", 4), ("Calculate 7 plus 3", 4), ("Calculate 9 minus 5", 4),
    ("Calculate 6 divided by 2", 4), ("Calculate 8 plus 5", 4), ("Calculate 3 times 3", 4),
    ("Calculate 5 plus 6", 4), ("Calculate 9 times 2", 4), ("Calculate 8 minus 3", 4),
    ("Calculate 4 plus 4", 4),
]:
    PLAN_FIRST_BATCH.append(("calculator", "calculator", expr, _CALC, 3 + clicks))

_PAINT_LINE = "Win -> type paint -> Enter -> wait -> ONE drag across the white canvas -> F10"
_PAINT_ZIG = "Win -> type paint -> Enter -> wait -> THREE drags on the canvas making a zigzag -> F10"
for i in range(5):
    PLAN_FIRST_BATCH.append(("paint", "paint", "Open Paint and draw a line", _PAINT_LINE + " (vary where you draw)", 5))
for i in range(5):
    PLAN_FIRST_BATCH.append(("paint", "paint", "Open Paint and draw a zigzag", _PAINT_ZIG, 7))


def _calc_plan(n: int = 30):
    """30 single-digit expressions that put every digit and every operator in several demos."""
    ops = [("plus", "+"), ("minus", "-"), ("times", "*"), ("divided by", "/")]
    items = []
    for i in range(n):
        a, b, (word, _) = i % 10, (7 * i + 3) % 10, ops[i % 4]
        if word == "divided by" and b == 0:
            b = 1
        if word == "minus" and a < b:
            a, b = b, a
        items.append(("calculator", "calculator", f"Calculate {a} {word} {b}", _CALC, 8))
    return items


PLAN_CALC = _calc_plan()


def _press_plan(n: int = 6, seed: int = 100):
    """n long episodes; each presses all 14 buttons once in a NEW random order, then '='.

    Random order matters: in a fixed order the previous click would predict the next one and the
    model would learn that shortcut instead of reading the target button.
    """
    import random

    items, firsts = [], set()
    for k in range(n):
        rng = random.Random(seed + k)
        while True:
            btns = list("0123456789+-*/")
            rng.shuffle(btns)
            if btns[0] not in firsts or len(firsts) >= 14:
                break
        firsts.add(btns[0])
        seq = btns + ["="]
        hint = ("Win -> calculator -> Enter -> wait -> click EXACTLY these 15 buttons in this order, "
                "then F10:  " + "  ".join(seq))
        items.append(("calculator", "calculator", "Press " + " ".join(seq), hint, 3 + 15 + 1))
    return items


PLAN_PRESS = _press_plan()


def _extras_plan(n: int = 6, seed: int = 500):
    """n long episodes that press every extra key (. % +/- sqrt x2 1/x C CE backspace) in a fresh order.

    Each mini-calculation is followed by C so the Calculator never gets stuck in an error state
    (no sqrt of a negative, no 1/0). Segment order is shuffled per episode.
    """
    import random

    items = []
    for k in range(n):
        r = random.Random(seed + k)
        d = lambda lo=1: str(r.randint(lo, 9))                        # noqa: E731
        segs = [[d(), "sqrt"], [d(), "sq"], [d(), "inv"], [d(), "neg"],
                [d(), ".", d(), "+", d(), "="], [d(), d(), "*", d(), d(), "pct"],
                [d(), d(), d(), "back", "back"], [d(), d(), "CE"]]
        r.shuffle(segs)
        seq = []
        for sgm in segs:
            seq += sgm + ["C"]
        hint = ("Win -> calculator -> Enter -> wait -> click these buttons in order, then F10.  "
                "(sqrt = the square-root key, sq = x-squared, inv = 1/x, neg = +/-, pct = %, back = backspace, "
                "CE, C):  " + "  ".join(seq))
        items.append(("calculator", "calculator", "Press " + " ".join(seq), hint, 3 + len(seq) + 1))
    return items


PLAN_EXTRAS = _extras_plan()
PLAN_PRESS2 = _press_plan(4, seed=300)        # 4 more long episodes for 0-9 + - * / = (new random orders)
PLAN_EXTRAS2 = _extras_plan(4, seed=900)      # 4 more long episodes for . % +/- sqrt x2 1/x C CE back


def run_session(out_root: str, plan=None, start: int = 0, countdown: int = 3) -> None:
    from argus.data.recorder import Recorder

    plan = plan or PLAN_FIRST_BATCH
    i = start
    while i < len(plan):
        task, app, instr, hint, min_steps = plan[i]
        print(f"\n=== [{i + 1}/{len(plan)}] {instr}  (task={task})")
        print(f"    do: {hint}")
        cmd = input("Close the app so the desktop is clean. Enter=record, s=skip, q=quit > ").strip().lower()
        if cmd == "q":
            print(f"Stopped. Resume later with --start {i}")
            return
        if cmd == "s":
            i += 1
            continue
        ep = Recorder(out_root, instr, task, app).run(countdown)
        issues = validate_episode(ep)
        n = len(ep.steps)
        status = "OK" if not issues and n >= min_steps else "CHECK"
        print(f"    -> {len(ep.frames)} frames, {n} actions (expected about {min_steps}+)  [{status}]")
        for s in ep.steps:
            print(f"       {s.step}: {s.action.to_dict()}")
        for msg in issues:
            print(f"       ! {msg}")
        ans = input("Keep? Enter=keep, d=delete and redo this one > ").strip().lower()
        if ans == "d":
            shutil.rmtree(ep.path, ignore_errors=True)
            continue          # redo same item
        i += 1
    print("\nBatch finished.")
