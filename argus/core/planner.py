"""Task planner: natural-language instruction -> one sub-goal per primitive action.

The policy no longer reads the sentence. It is told WHICH button/step is next, so it
learns "press the labelled button", which works for any number it has seen the buttons for.

    "Calculate 8*6"        ->  OPEN_calculator x3, CLICK_8, CLICK_*, CLICK_6, CLICK_=, DONE
    "square root of 81"    ->  OPEN_calculator x3, CLICK_8, CLICK_1, CLICK_sqrt, DONE

Goal ids 0..20 never change (old checkpoints keep working); new buttons are appended after DONE.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

from argus.actions.action_space import Action, ActionType

APPS = ["notepad", "calculator", "paint"]
CORE_BUTTONS = list("0123456789") + ["+", "-", "*", "/", "="]
# ".": decimal point, pct: %, neg: +/-, sqrt: square root, sq: x squared, inv: 1/x, C: clear, CE: clear entry,
# back: backspace.  (Memory buttons MC MR M+ M- MS are not supported yet.)
EXTRA_BUTTONS = [".", "pct", "neg", "sqrt", "sq", "inv", "C", "CE", "back"]
BUTTONS = CORE_BUTTONS + EXTRA_BUTTONS
GOALS = ([f"OPEN_{a}" for a in APPS] + [f"CLICK_{b}" for b in CORE_BUTTONS]
         + ["DRAW_line", "DRAW_zigzag", "DONE"] + [f"CLICK_{b}" for b in EXTRA_BUTTONS]
         # a zigzag is three strokes; each gets its OWN goal so each learns its own start and end. (With one shared
         # goal all three strokes started at the same cell and drew a 41x2 px mark. "DRAW_zigzag" stays only so
         # that the ids of everything before it never change.)
         + ["DRAW_zigzag_1", "DRAW_zigzag_2", "DRAW_zigzag_3"])
GOAL_ID = {g: i for i, g in enumerate(GOALS)}
N_GOALS = len(GOALS)

NUM = r"\d+(?:\.\d+)?"
_ALIASES = {"notepad": "notepad", "calculator": "calculator", "calc": "calculator",
            "paint": "paint", "mspaint": "paint"}
_NUMWORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_NUMWORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
                  "eighty": "80", "ninety": "90", "hundred": "100"})
_TOKEN = {"c": "C", "clear": "C", "ce": "CE", "back": "back", "backspace": "back", "del": "back",
          "delete": "back", "sqrt": "sqrt", "\u221a": "sqrt", "sq": "sq", "square": "sq", "x\u00b2": "sq",
          "inv": "inv", "recip": "inv", "reciprocal": "inv", "1/x": "inv", "neg": "neg", "+/-": "neg",
          "\u00b1": "neg", "negate": "neg", "%": "pct", "pct": "pct", "percent": "pct", ".": "."}
_ALLOWED = {"OPEN": {ActionType.KEY, ActionType.TYPE}, "CLICK": {ActionType.CLICK},
            "DRAW": {ActionType.DRAG}, "DONE": {ActionType.DONE}}


class PlanError(ValueError):
    pass


def _fmt(x: float) -> str:
    return str(int(x)) if x == int(x) and abs(x) < 1e15 else repr(x)


def simulate(buttons: List[str]) -> Optional[float]:
    """Value the Windows Calculator (Standard mode, immediate execution) shows after these clicks.

    Returns None when unsure: the % key (its rule differs between operators), divide by zero, sqrt of a negative.
    """
    acc, pending, entry, fresh = None, None, "0", True

    def val():
        return float(entry)

    def do(a, op, b):
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        return None if b == 0 else a / b

    for b in buttons:
        if b.isdigit():
            entry = b if (fresh or entry == "0") else entry + b
            fresh = False
        elif b == ".":
            if fresh:
                entry, fresh = "0.", False
            elif "." not in entry:
                entry += "."
        elif b in ("+", "-", "*", "/"):
            if pending and not fresh:
                acc = do(acc, pending, val())
                if acc is None:
                    return None
                entry = _fmt(acc)
            elif pending is None:
                acc = val()
            pending, fresh = b, True
        elif b == "=":
            if pending:
                acc = do(acc, pending, val())
                if acc is None:
                    return None
                entry, pending = _fmt(acc), None
            fresh = True
        elif b == "sqrt":
            if val() < 0:
                return None
            entry, fresh = _fmt(math.sqrt(val())), True
        elif b == "sq":
            entry, fresh = _fmt(val() ** 2), True
        elif b == "inv":
            if val() == 0:
                return None
            entry, fresh = _fmt(1 / val()), True
        elif b == "neg":
            if val() != 0:
                entry = entry[1:] if entry.startswith("-") else "-" + entry
        elif b == "back":
            if not fresh:
                entry = entry[:-1] if len(entry) > 1 else "0"
                if entry in ("", "-"):
                    entry = "0"
        elif b == "CE":
            entry, fresh = "0", True
        elif b == "C":
            acc, pending, entry, fresh = None, None, "0", True
        elif b == "pct":
            return None
    return float(entry)


@dataclass
class Plan:
    task: str                       # open_app | calculator | paint
    goals: List[str]                # one per primitive action, last is DONE
    expr: Optional[str] = None
    app: str = ""
    ids: List[int] = field(default_factory=list)
    answer: Optional[float] = None  # what the Calculator should display at the end (None = unknown)

    def __post_init__(self):
        self.ids = [GOAL_ID[g] for g in self.goals]

    def expected(self) -> Optional[float]:
        return self.answer


def _calc_plan(buttons: List[str], expr: str) -> Plan:
    goals = ["OPEN_calculator"] * 3 + [f"CLICK_{b}" for b in buttons] + ["DONE"]
    return Plan("calculator", goals, expr=expr, app="calculator", answer=simulate(buttons))


def _normalise(text: str) -> str:
    t = text.lower()
    for a, b in (("\u00d7", "*"), ("\u00f7", "/"), ("\u2212", "-"), ("\u2013", "-"), ("\u2014", "-")):
        t = t.replace(a, b)
    t = re.sub(r"\bmultiplied\s+by\b", "*", t)
    t = re.sub(r"\bdivided\s+by\b", "/", t)
    for w, s in (("times", "*"), ("into", "*"), ("plus", "+"), ("minus", "-"), ("over", "/")):
        t = re.sub(rf"\b{w}\b", s, t)
    t = re.sub(r"\b(" + "|".join(_NUMWORDS) + r")\b", lambda m: _NUMWORDS[m.group(1)], t)
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", t)
    return t


def _chars(num: str) -> List[str]:
    return list(num)                                  # digits and '.'


def _press_buttons(rest: str) -> List[str]:
    out: List[str] = []
    for tok in re.split(r"[\s,]+", rest.strip()):
        if not tok:
            continue
        if tok in _TOKEN:
            out.append(_TOKEN[tok])
        elif tok == "x":
            out.append("*")
        elif re.fullmatch(r"[0-9+\-*/=.%]+", tok):
            out += ["pct" if c == "%" else c for c in tok]
        else:
            raise PlanError(f"unknown calculator button {tok!r} after 'Press'")
    return out


_UNARY = [
    ("sqrt", rf"(?:square\s*root|sqrt|\u221a)\s*(?:of\s*)?({NUM})"),
    ("sq", rf"(?:square\s+of\s+({NUM}))|(?:({NUM})\s*(?:squared|\^\s*2|\u00b2))"),
    ("inv", rf"(?:reciprocal|inverse)\s*(?:of\s*)?({NUM})"),
    ("neg", rf"(?:negate|negative\s+of|(?:change|flip)\s+(?:the\s+)?sign\s+of)\s*({NUM})"),
]


def parse_instruction(text: str) -> Plan:
    t = _normalise(text)
    if re.search(r"\d*\.\d*\.", t):
        raise PlanError("malformed number (two decimal points)")

    pm = re.match(r"\s*(?:press|click|tap)\b(.*)$", t)
    if pm:                                                   # explicit button list, e.g. "Press 4 7 + 9 ="
        buttons = _press_buttons(pm.group(1))
        if not buttons:
            raise PlanError("after 'Press' list calculator buttons, e.g. 'Press 4 7 + 9 ='")
        return _calc_plan(buttons, " ".join(buttons))

    for op, pat in _UNARY:                                   # one-button operations
        m = re.search(pat, t)
        if m:
            if re.search(r"\d", t[:m.start()] + t[m.end():]):
                raise PlanError("one calculator operation at a time, please")
            num = next(g for g in m.groups() if g)
            return _calc_plan(_chars(num) + [op], f"{op}({num})")

    m = re.search(rf"({NUM})\s*(?:%|percent)\s*of\s*({NUM})", t)     # "10% of 50" -> 50 x 10 %
    if m:
        p, n = m.group(1), m.group(2)
        return _calc_plan(_chars(n) + ["*"] + _chars(p) + ["pct"], f"{p}% of {n}")

    m = re.search(rf"{NUM}(?:\s*[-+*/]\s*{NUM})+", t)
    app = next((_ALIASES[w] for w in re.findall(r"[a-z]+", t) if w in _ALIASES), "")
    if m:                                                    # any arithmetic -> calculator
        expr = re.sub(r"\s+", "", m.group(0))
        return _calc_plan(list(expr) + ["="], expr)
    if app == "paint" and "draw" in t:
        if re.search(r"zig\s*-?\s*zag", t):
            return Plan("paint", ["OPEN_paint"] * 3 + ["DRAW_zigzag_1", "DRAW_zigzag_2", "DRAW_zigzag_3"] + ["DONE"],
                        app="paint")
        if "line" in t:
            return Plan("paint", ["OPEN_paint"] * 3 + ["DRAW_line", "DONE"], app="paint")
        raise PlanError("for Paint say 'draw a line' or 'draw a zigzag'")
    if app:
        return Plan("open_app", [f"OPEN_{app}"] * 3 + ["DONE"], app=app)
    raise PlanError(f"cannot understand {text!r}. Try: 'Open Notepad', 'Calculate 8*6', "
                    f"'square root of 81', 'Open Paint and draw a line'")


def align_goals(plan: Plan, actions: List[Action]) -> List[str]:
    """Check a recorded demonstration really follows the plan, step by step."""
    if len(actions) != len(plan.goals):
        raise PlanError(f"{len(actions)} recorded actions but the plan needs {len(plan.goals)}")
    for i, (g, a) in enumerate(zip(plan.goals, actions)):
        if a.type not in _ALLOWED[g.split("_")[0]]:
            raise PlanError(f"step {i}: goal {g} but the recorded action is {a.type.value}")
    return plan.goals
