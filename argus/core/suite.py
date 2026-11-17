"""Automatic calculator test suite: runs the agent on unseen questions and checks the answer.

    python -m argus.core suite --ckpt checkpoints/bc_v4.pt            # levels 1 and 2
    python -m argus.core suite --ckpt checkpoints/bc_v4.pt --levels 1,2,3,4 --repeat 2

HANDS OFF the mouse and keyboard while it runs. Slam the mouse into a screen corner to abort.
Do NOT record demonstrations for these questions, or they stop being unseen.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from argus.core.planner import PlanError, parse_instruction

# (level, instruction). Level 4 must be REFUSED (no mouse movement). Level 5 = keys beyond 0-9 + - * / =. Level 6 = --heldout. Level 7 = Paint.
SUITE: List[Tuple[int, str]] = [
    (1, "Calculate 8*6"), (1, "7 x 9"), (1, "Calculate 5+8"), (1, "Calculate 9 minus 4"), (1, "6*3"),
    (1, "Calculate 8 divided by 2"), (1, "9 - 3"), (1, "Calculate 7 plus 6"),
    (2, "Calculate 0 plus 7"), (2, "Calculate 1 times 9"), (2, "7*0"), (2, "Calculate 6 divided by 3"),
    (2, "what is 4 times 7"), (2, "eight times six"), (2, "9 into 3"), (2, "calc 5 plus 5"),
    (2, "Press 3 + 4 ="),
    (3, "Calculate 2+3*4"), (3, "Calculate 12 + 7"), (3, "Calculate 25 minus 8"), (3, "15 divided by 3"),
    (3, "34 + 66"), (3, "Calculate 1+2+3"),
    (4, "make me a sandwich"), (4, "open the calendar"), (4, "Calculate 5 +"), (4, "Calculate 3.5.2 + 1"),
    # level 5: the other keys. They need demonstrations first (python -m argus.data session --plan extras).
    (5, "square root of 81"), (5, "7 squared"), (5, "reciprocal of 4"), (5, "negate 5"),
    (5, "Calculate 3.5 + 2"), (5, "9.5 - 4"), (5, "Press 1 2 3 back ="), (5, "Press 5 + 3 = C"),
    (5, "10% of 50"),                       # the % rule is not verified yet -> you judge it (y/n)
    (7, "Open Paint and draw a line"), (7, "Open Paint and draw a zigzag"),     # judged by the ink check
]


def parse_number(text: str) -> Optional[float]:
    t = text.replace("\u2212", "-").replace("\u200e", "").replace("\u00a0", "").replace(" ", "")
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", t)
    return float(m.group(0).replace(",", "")) if m else None


def _ps(cmd: str):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=15)


def read_calculator_result(sleep: Callable[[float], None] = time.sleep) -> Optional[float]:
    """Ctrl+C in the Calculator copies the displayed value; read it back from the clipboard."""
    import pyautogui

    _ps("Set-Clipboard -Value 'ARGUS_NONE'")          # sentinel so a stale clipboard cannot pass
    sleep(0.5)
    pyautogui.hotkey("ctrl", "c")
    sleep(0.5)
    return parse_number(_ps("Get-Clipboard").stdout)


def calculator_running() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq CalculatorApp.exe"], capture_output=True, text=True).stdout
    return "CalculatorApp.exe" in out


def close_calculator() -> None:
    subprocess.run(["taskkill", "/F", "/IM", "CalculatorApp.exe"], capture_output=True)


def close_apps() -> None:
    """Close the apps the suite opens, so every run starts from a clean desktop."""
    close_calculator()
    subprocess.run(["taskkill", "/F", "/IM", "mspaint.exe"], capture_output=True)


def run_suite(agent, items, read_result: Callable[[], Optional[float]], reset: Callable[[], None],
              ask: Callable[[str], str] = input, out_dir: str = "runs", ckpt: str = "", repeat: int = 1,
              manual: bool = False, countdown: int = 2, sleep: Callable[[float], None] = time.sleep,
              explain: Optional[Callable[[str, str], List[str]]] = None, aim: str = "") -> dict:
    rows = []
    todo = [(lv, t) for lv, t in items for _ in range(repeat)]
    os.makedirs(out_dir, exist_ok=True)
    for i, (level, text) in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] L{level}: {text!r}")
        row = {"level": level, "instruction": text, "task": "calculator", "ckpt": ckpt, "aim": aim, "expected": None,
               "got": None, "stopped": "", "success": False, "trace": ""}
        if level == 4:                                            # must be refused, mouse never moves
            try:
                agent.runner.reset(text)
                row["stopped"] = "NOT refused"
            except PlanError as e:
                row.update(success=True, stopped="refused")
                print(f"    refused as expected: {str(e)[:70]}")
        else:
            plan = parse_instruction(text)
            expected = plan.expected()
            row["task"], row["expected"] = plan.task, expected
            if expected is None:
                manual_item = True              # answer unknown (e.g. the % key): you judge it
            else:
                manual_item = manual
            reset()
            try:
                res = agent.run(text, countdown=countdown)
                row["stopped"], row["trace"] = res["stopped"], res["trace"]
                if res["stopped"] == "done" and plan.task == "paint":
                    from argus.core.paintcheck import ink_stats, judge_paint

                    sleep(0.5)
                    frame = agent.capture.grab()
                    ok, why = judge_paint(ink_stats(frame), "zigzag" if "zigzag" in text.lower() else "line")
                    row["success"], row["got"] = ok, why
                    if os.path.isdir(res["trace"]):               # evidence for the report: what was actually drawn
                        frame.save(os.path.join(res["trace"], "final_canvas.png"))
                        print(f"    final canvas saved: {os.path.join(res['trace'], 'final_canvas.png')}")
                elif res["stopped"] == "done":
                    sleep(0.5)
                    if manual_item:
                        shown = f"{expected:g}" if expected is not None else "the right answer"
                        row["success"] = ask(f"Does the Calculator show {shown}? [y/n] > ").strip().lower() == "y"
                    else:
                        row["got"] = read_result()
                        row["success"] = (row["got"] is not None and expected is not None
                                          and math.isclose(row["got"], expected, rel_tol=1e-9, abs_tol=1e-9))
            except PlanError as e:
                row["stopped"] = f"refused: {str(e)[:60]}"
            print(f"    expected={expected}  got={row['got']}  stopped={row['stopped']}  -> "
                  f"{'PASS' if row['success'] else 'FAIL'}")
            reset()
            sleep(1.0)
        rows.append(row)
        with open(os.path.join(out_dir, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({**row, "time": datetime.now().isoformat(timespec="seconds")}) + "\n")

    by = {}
    for r in rows:
        by.setdefault(r["level"], [0, 0])
        by[r["level"]][0] += r["success"]
        by[r["level"]][1] += 1
    print("\n==== RESULT ====")
    for lv, (ok, n) in sorted(by.items()):
        print(f"  level {lv}: {ok}/{n} = {ok / n:.0%}")
    fails = [r for r in rows if not r["success"]]
    if fails:
        print("  failed:")
        for r in fails:
            print(f"    L{r['level']} {r['instruction']!r}: expected {r['expected']}, got {r['got']}, stopped {r['stopped']}"
                  + (f"   trace {r['trace']}" if r["trace"] else ""))
            if explain and r["trace"]:
                try:
                    for line in explain(r["trace"], r["instruction"]):
                        print("        " + line)
                except Exception as e:                    # never let a diagnostic break the suite
                    print(f"        (could not explain: {e})")
    return {"rows": rows, "by_level": by}
