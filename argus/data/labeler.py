"""Turn raw input events into discrete, frame-aligned action labels.

Pure function (no OS access) so it can be unit-tested with synthetic events.

Raw event kinds (all have "t" seconds since recording start):
  {"kind": "press",   "x", "y", "button"}
  {"kind": "release", "x", "y", "button"}
  {"kind": "scroll",  "x", "y", "dy"}
  {"kind": "key_down", "key"}   # chars as themselves, specials as 'enter','ctrl',...
  {"kind": "key_up",   "key"}
"""
from __future__ import annotations

import bisect
import math
from typing import Dict, List, Optional

from argus.actions.action_space import Action, ActionType, GRID, xy_to_cell
from argus.data.episode import Step

MODIFIERS = {"ctrl", "alt", "shift", "win"}
DRAG_PX = 20            # movement between press and release that counts as a drag
DOUBLE_CLICK_S = 0.4
SCROLL_MERGE_S = 0.3
TYPE_GAP_S = 1.0


def _frame_before(frame_ts: List[float], t: float) -> int:
    """Index of the last frame captured at or before t (the state the human saw)."""
    i = bisect.bisect_right(frame_ts, t + 1e-9) - 1
    return max(i, 0)


def build_steps(
    events: List[dict],
    frame_ts: List[float],
    screen_w: int,
    screen_h: int,
    grid: int = GRID,
    add_done: bool = True,
) -> List[Step]:
    if not frame_ts:
        raise ValueError("no frames recorded")
    ev = sorted(events, key=lambda e: e["t"])
    out: List[Step] = []

    def emit(t: float, action: Action):
        out.append(Step(len(out), t, _frame_before(frame_ts, t), action))

    held = set()
    solo_mod: Dict[str, bool] = {}        # modifier pressed and nothing else yet
    press: Optional[dict] = None
    buf: List[str] = []
    buf_t0 = 0.0
    buf_last = 0.0
    scroll_acc: Optional[dict] = None

    def cell(x, y):
        return xy_to_cell(x, y, screen_w, screen_h, grid)

    def flush_text():
        nonlocal buf
        if buf:
            emit(buf_t0, Action(ActionType.TYPE, text="".join(buf)))
            buf = []

    def flush_scroll():
        nonlocal scroll_acc
        if scroll_acc and scroll_acc["dy"] != 0:
            emit(scroll_acc["t"], Action(ActionType.SCROLL, cell=cell(scroll_acc["x"], scroll_acc["y"]),
                                         scroll=int(scroll_acc["dy"])))
        scroll_acc = None

    for e in ev:
        k, t = e["kind"], e["t"]
        if k != "scroll":
            flush_scroll()

        if k == "press":
            flush_text()
            press = e
        elif k == "release":
            if press is None:
                continue
            dist = math.hypot(e["x"] - press["x"], e["y"] - press["y"])
            if dist > DRAG_PX:
                emit(press["t"], Action(ActionType.DRAG, cell=cell(press["x"], press["y"]),
                                        end_cell=cell(e["x"], e["y"])))
            else:
                c = cell(press["x"], press["y"])
                prev = out[-1] if out else None
                if (prev and prev.action.type == ActionType.CLICK and prev.action.cell == c
                        and prev.action.button == press["button"] and not prev.action.double
                        and press["t"] - prev.t < DOUBLE_CLICK_S):
                    prev.action.double = True
                else:
                    emit(press["t"], Action(ActionType.CLICK, cell=c, button=press["button"]))
            press = None
        elif k == "scroll":
            flush_text()
            if scroll_acc and t - scroll_acc["last"] < SCROLL_MERGE_S:
                scroll_acc["dy"] += e["dy"]
                scroll_acc["last"] = t
            else:
                flush_scroll()
                scroll_acc = {"t": t, "last": t, "x": e["x"], "y": e["y"], "dy": e["dy"]}
        elif k == "key_down":
            key = e["key"]
            if key in MODIFIERS:
                if key not in held:
                    # a new modifier press cancels "solo" status of others
                    for m in solo_mod:
                        solo_mod[m] = False
                    solo_mod[key] = True
                held.add(key)
                continue
            for m in solo_mod:
                solo_mod[m] = False
            combo_mods = held - {"shift"} if len(key) == 1 else set(held)
            if combo_mods:
                flush_text()
                order = [m for m in ("ctrl", "alt", "shift", "win") if m in held]
                emit(t, Action(ActionType.KEY, key="+".join(order + [key])))
            elif len(key) == 1:
                if buf and t - buf_last > TYPE_GAP_S:
                    flush_text()
                if not buf:
                    buf_t0 = t
                buf.append(key)
                buf_last = t
            elif key == "backspace" and buf:
                buf.pop()
                buf_last = t
            elif key == "space":
                if buf and t - buf_last > TYPE_GAP_S:
                    flush_text()
                if not buf:
                    buf_t0 = t
                buf.append(" ")
                buf_last = t
            else:
                flush_text()
                emit(t, Action(ActionType.KEY, key=key))
        elif k == "key_up":
            key = e["key"]
            if key in MODIFIERS:
                if solo_mod.get(key) and key == "win":
                    flush_text()
                    emit(t, Action(ActionType.KEY, key="win"))
                solo_mod.pop(key, None)
                held.discard(key)

    flush_text()
    flush_scroll()
    if add_done:
        t_end = max(ev[-1]["t"] if ev else 0.0, frame_ts[-1])
        out.append(Step(len(out), t_end, len(frame_ts) - 1, Action(ActionType.DONE)))
    return out
