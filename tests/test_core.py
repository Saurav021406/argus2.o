import os
from PIL import Image

from argus.actions.action_space import Action, ActionType, xy_to_cell, cell_to_xy, GRID
from argus.actions.executor import Executor
from argus.safety.validator import SafetyValidator
from argus.data.labeler import build_steps
from argus.data.episode import Episode, Step
from argus.data.validator import validate_episode
from argus.data.replay import render_overlays

W, H = 1920, 1080
FRAMES = [i * 0.2 for i in range(50)]   # 5 fps for 10 s


def click(t, x, y, b="left", hold=0.05):
    return [{"t": t, "kind": "press", "x": x, "y": y, "button": b},
            {"t": t + hold, "kind": "release", "x": x, "y": y, "button": b}]


def typed(t, s, gap=0.05):
    ev = []
    for i, ch in enumerate(s):
        ev.append({"t": t + i * gap, "kind": "key_down", "key": ch})
        ev.append({"t": t + i * gap + 0.01, "kind": "key_up", "key": ch})
    return ev


def test_grid_roundtrip():
    for c in (0, 31, 32, 500, 1023):
        x, y = cell_to_xy(c, W, H)
        assert xy_to_cell(x, y, W, H) == c
    assert xy_to_cell(W + 50, H + 50, W, H) == GRID * GRID - 1   # clamped
    assert xy_to_cell(-5, -5, W, H) == 0


def test_click_type_enter_done():
    ev = click(1.0, 960, 540) + typed(2.0, "hello") + [
        {"t": 3.0, "kind": "key_down", "key": "enter"}, {"t": 3.02, "kind": "key_up", "key": "enter"}]
    steps = build_steps(ev, FRAMES, W, H)
    kinds = [s.action.type for s in steps]
    assert kinds == [ActionType.CLICK, ActionType.TYPE, ActionType.KEY, ActionType.DONE]
    assert steps[1].action.text == "hello"
    assert steps[2].action.key == "enter"
    assert steps[0].action.cell == xy_to_cell(960, 540, W, H)
    # frame is the last one at or before the action time
    assert steps[0].frame == 5 and steps[1].frame == 10


def test_double_click_and_right_click():
    ev = click(1.0, 100, 100) + click(1.2, 100, 100) + click(3.0, 400, 400, "right")
    steps = build_steps(ev, FRAMES, W, H)
    assert steps[0].action.double is True
    assert steps[1].action.button == "right"
    assert [s.action.type for s in steps].count(ActionType.CLICK) == 2


def test_drag():
    ev = [{"t": 1.0, "kind": "press", "x": 100, "y": 100, "button": "left"},
          {"t": 1.5, "kind": "release", "x": 700, "y": 500, "button": "left"}]
    s = build_steps(ev, FRAMES, W, H)[0]
    assert s.action.type == ActionType.DRAG and s.action.cell != s.action.end_cell


def test_scroll_merge():
    ev = [{"t": 1.0 + i * 0.05, "kind": "scroll", "x": 500, "y": 500, "dy": -1} for i in range(5)]
    s = build_steps(ev, FRAMES, W, H)
    assert s[0].action.type == ActionType.SCROLL and s[0].action.scroll == -5


def test_win_key_alone_and_hotkey():
    ev = [{"t": 1.0, "kind": "key_down", "key": "win"}, {"t": 1.05, "kind": "key_up", "key": "win"}]
    ev += [{"t": 2.0, "kind": "key_down", "key": "ctrl"}, {"t": 2.05, "kind": "key_down", "key": "s"},
           {"t": 2.1, "kind": "key_up", "key": "s"}, {"t": 2.15, "kind": "key_up", "key": "ctrl"}]
    steps = build_steps(ev, FRAMES, W, H)
    assert steps[0].action.key == "win"
    assert steps[1].action.key == "ctrl+s"
    assert len(steps) == 3  # + DONE, ctrl alone did not emit


def test_win_as_modifier_is_not_solo():
    ev = [{"t": 1.0, "kind": "key_down", "key": "win"}, {"t": 1.02, "kind": "key_down", "key": "d"},
          {"t": 1.04, "kind": "key_up", "key": "d"}, {"t": 1.06, "kind": "key_up", "key": "win"}]
    steps = build_steps(ev, FRAMES, W, H)
    assert [s.action.key for s in steps[:-1]] == ["win+d"]


def test_shifted_char_and_backspace_and_space():
    ev = [{"t": 1.0, "kind": "key_down", "key": "shift"}, {"t": 1.01, "kind": "key_down", "key": "H"},
          {"t": 1.02, "kind": "key_up", "key": "H"}, {"t": 1.03, "kind": "key_up", "key": "shift"}]
    ev += typed(1.2, "ix") + [{"t": 1.4, "kind": "key_down", "key": "backspace"},
                              {"t": 1.45, "kind": "key_down", "key": "space"}]
    ev += typed(1.5, "ok")
    s = build_steps(ev, FRAMES, W, H)
    assert s[0].action.text == "Hi ok"


def test_typing_pause_splits():
    ev = typed(1.0, "ab") + typed(4.0, "cd")
    s = build_steps(ev, FRAMES, W, H)
    assert [x.action.text for x in s[:-1]] == ["ab", "cd"]


def test_safety_validator():
    v = SafetyValidator()
    assert v.validate(Action(ActionType.CLICK, cell=5))[0]
    assert not v.validate(Action(ActionType.CLICK))[0]
    assert not v.validate(Action(ActionType.CLICK, cell=5000))[0]
    assert not v.validate(Action(ActionType.KEY, key="Alt+F4"))[0]
    assert not v.validate(Action(ActionType.KEY, key="win+L"))[0]
    assert not v.validate(Action(ActionType.TYPE, text="x" * 500))[0]
    assert v.validate(Action(ActionType.KEY, key="ctrl+s"))[0]


def test_executor_dry_run_blocks_and_passes():
    ex = Executor(dry_run=True)
    assert ex.execute(Action(ActionType.CLICK, cell=10)).reason == "dry_run"
    r = ex.execute(Action(ActionType.KEY, key="alt+f4"))
    assert not r.ok and "blocked" in r.reason


def test_episode_roundtrip_validate_overlay(tmp_path):
    d = str(tmp_path / "ep")
    os.makedirs(d + "/frames")
    frames = []
    for i in range(6):
        Image.new("RGB", (640, 360), (i * 30, 40, 40)).save(f"{d}/frames/{i:06d}.jpg")
        frames.append({"i": i, "t": i * 0.2, "file": f"frames/{i:06d}.jpg"})
    meta = {"instruction": "Open Notepad", "screen_w": W, "screen_h": H, "frame_w": 640, "frame_h": 360,
            "grid": 32, "fps": 5}
    steps = [Step(0, 0.3, 1, Action(ActionType.CLICK, cell=100)),
             Step(1, 0.6, 3, Action(ActionType.TYPE, text="notepad")),
             Step(2, 1.0, 5, Action(ActionType.DONE))]
    Episode(d, meta, frames, steps).save_labels()
    ep = Episode.load(d)
    assert ep.steps[1].action.text == "notepad" and ep.meta["n_actions"] == 3
    assert validate_episode(ep) == []
    assert render_overlays(ep, d + "/ov") == 3 and os.path.exists(d + "/ov/step_000.png")
    ep.steps[1].frame = 99
    assert any("out of range" in i for i in validate_episode(ep))
