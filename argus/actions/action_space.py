"""Action space for the Argus policy.

Locations are discretised to a GRID x GRID grid (default 32x32 = 1024 cells), so
the location head is a classification problem, not a regression problem.
A cell id is row * GRID + col.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Tuple

GRID = 32


class ActionType(str, Enum):
    MOVE = "MOVE"
    CLICK = "CLICK"
    DRAG = "DRAG"
    TYPE = "TYPE"
    KEY = "KEY"        # single key or combo, e.g. "enter", "ctrl+s", "win"
    SCROLL = "SCROLL"
    DONE = "DONE"


ACTION_TYPES = [a.value for a in ActionType]  # stable order -> class index


@dataclass
class Action:
    type: ActionType
    cell: Optional[int] = None        # MOVE / CLICK / DRAG(start) / SCROLL
    end_cell: Optional[int] = None    # DRAG end
    text: Optional[str] = None        # TYPE
    key: Optional[str] = None         # KEY
    button: str = "left"              # CLICK
    double: bool = False              # CLICK
    scroll: int = 0                   # SCROLL, + = up

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        if d.get("button") == "left":
            d.pop("button")          # default; keeps the JSON clean
        return {k: v for k, v in d.items() if v not in (None, "", 0, False) or k == "type"}

    @staticmethod
    def from_dict(d: dict) -> "Action":
        d = dict(d)
        d["type"] = ActionType(d["type"])
        return Action(**{k: v for k, v in d.items() if k in Action.__dataclass_fields__})


def xy_to_cell(x: float, y: float, width: int, height: int, grid: int = GRID) -> int:
    col = min(max(int(x / width * grid), 0), grid - 1)
    row = min(max(int(y / height * grid), 0), grid - 1)
    return row * grid + col


def cell_to_xy(cell: int, width: int, height: int, grid: int = GRID) -> Tuple[int, int]:
    """Centre pixel of a cell."""
    if not 0 <= cell < grid * grid:
        raise ValueError(f"cell {cell} out of range for grid {grid}")
    row, col = divmod(cell, grid)
    return (int((col + 0.5) * width / grid), int((row + 0.5) * height / grid))
