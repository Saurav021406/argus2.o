"""Action executor: turns a predicted Action into real mouse/keyboard input.

pyautogui is imported lazily so the rest of the package (training, tests,
dataset tools) works on machines without a display.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from argus.actions.action_space import Action, ActionType, cell_to_xy, GRID
from argus.safety.validator import SafetyValidator


@dataclass
class ExecResult:
    ok: bool
    reason: str = "ok"
    executed: bool = False


class Executor:
    def __init__(
        self,
        validator: Optional[SafetyValidator] = None,
        dry_run: bool = False,
        pause: float = 0.1,
        screen_size: Optional[Tuple[int, int]] = None,
        grid: int = GRID,
    ):
        self.validator = validator or SafetyValidator(grid=grid)
        self.dry_run = dry_run
        self.pause = pause
        self.grid = grid
        self._size = screen_size
        self._pag = None
        self.log: list = []

    def _backend(self):
        if self._pag is None:
            from argus.actions.platform import ensure_dpi_aware

            ensure_dpi_aware()
            import pyautogui

            pyautogui.FAILSAFE = True  # slam mouse into a corner to abort
            # By default only the TOP-LEFT pixel aborts. Make all four corners abort.
            if hasattr(pyautogui, "FAILSAFE_POINTS"):
                w, h = (int(v) for v in pyautogui.size())
                pyautogui.FAILSAFE_POINTS = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
            pyautogui.PAUSE = self.pause
            self._pag = pyautogui
        return self._pag

    def screen_size(self) -> Tuple[int, int]:
        if self._size is None:
            if self.dry_run:
                self._size = (1920, 1080)
            else:
                s = self._backend().size()
                self._size = (int(s[0]), int(s[1]))
        return self._size

    def execute(self, a: Action) -> ExecResult:
        ok, reason = self.validator.validate(a)
        if not ok:
            res = ExecResult(False, reason, False)
            self.log.append((a, res))
            return res
        if self.dry_run:
            res = ExecResult(True, "dry_run", False)
            self.log.append((a, res))
            return res
        try:
            self._run(a)
            res = ExecResult(True, "ok", True)
        except Exception as e:  # FailSafeException included
            res = ExecResult(False, f"{type(e).__name__}: {e}", False)
        self.log.append((a, res))
        return res

    def _run(self, a: Action) -> None:
        pag = self._backend()
        w, h = self.screen_size()
        t = a.type
        if t == ActionType.DONE:
            return
        if t == ActionType.MOVE:
            x, y = cell_to_xy(a.cell, w, h, self.grid)
            pag.moveTo(x, y, duration=0.15)
        elif t == ActionType.CLICK:
            x, y = cell_to_xy(a.cell, w, h, self.grid)
            pag.click(x, y, clicks=2 if a.double else 1, button=a.button)
        elif t == ActionType.DRAG:
            x0, y0 = cell_to_xy(a.cell, w, h, self.grid)
            x1, y1 = cell_to_xy(a.end_cell, w, h, self.grid)
            pag.moveTo(x0, y0, duration=0.1)
            pag.dragTo(x1, y1, duration=0.4, button="left")
        elif t == ActionType.TYPE:
            pag.write(a.text, interval=0.02)  # note: ASCII only
        elif t == ActionType.KEY:
            pag.hotkey(*a.key.lower().split("+"))
        elif t == ActionType.SCROLL:
            x, y = cell_to_xy(a.cell, w, h, self.grid)
            pag.moveTo(x, y, duration=0.1)
            pag.scroll(a.scroll)
        time.sleep(self.pause)
