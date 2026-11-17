"""Execution safeguard: blocks actions the agent should never run unattended.

This is a guard rail for a real desktop, not a safety-research contribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set, Tuple

from argus.actions.action_space import Action, ActionType, GRID

DEFAULT_BLOCKED_KEYS = {
    "win+l", "ctrl+alt+delete", "ctrl+shift+delete", "alt+f4", "win+r", "win+x",
}


def normalise_key(key: str) -> str:
    parts = [p.strip().lower() for p in key.split("+")]
    order = {"ctrl": 0, "alt": 1, "shift": 2, "win": 3}
    mods = sorted([p for p in parts[:-1]], key=lambda m: order.get(m, 9))
    return "+".join(mods + [parts[-1]])


@dataclass
class SafetyValidator:
    blocked_keys: Set[str] = field(default_factory=lambda: set(DEFAULT_BLOCKED_KEYS))
    max_text_len: int = 200
    max_scroll: int = 20
    grid: int = GRID

    def validate(self, a: Action) -> Tuple[bool, str]:
        n = self.grid * self.grid
        for name in ("cell", "end_cell"):
            v = getattr(a, name)
            if v is not None and not 0 <= v < n:
                return False, f"{name} {v} outside grid"
        if a.type in (ActionType.CLICK, ActionType.MOVE, ActionType.DRAG, ActionType.SCROLL) and a.cell is None:
            return False, f"{a.type.value} needs a cell"
        if a.type == ActionType.DRAG and a.end_cell is None:
            return False, "DRAG needs end_cell"
        if a.type == ActionType.TYPE:
            if not a.text:
                return False, "TYPE needs text"
            if len(a.text) > self.max_text_len:
                return False, f"text longer than {self.max_text_len}"
        if a.type == ActionType.KEY:
            if not a.key:
                return False, "KEY needs key"
            if normalise_key(a.key) in self.blocked_keys:
                return False, f"key combo {a.key!r} is blocked"
        if a.type == ActionType.SCROLL and abs(a.scroll) > self.max_scroll:
            return False, "scroll too large"
        if a.type == ActionType.CLICK and a.button not in ("left", "right", "middle"):
            return False, f"bad button {a.button!r}"
        return True, "ok"
