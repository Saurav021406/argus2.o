from argus.actions.action_space import Action, ActionType as T
from argus.data.audit import audit_episode
from argus.data.episode import Episode, Step


def mk(task, app, instr, acts):
    n = max(len(acts), 4)
    frames = [{"i": i, "t": i * 0.2, "file": f"frames/{i:06d}.jpg"} for i in range(n + 3)]
    steps = [Step(i, i * 1.0, min(i, len(frames) - 1), a) for i, a in enumerate(acts)]
    meta = {"instruction": instr, "task": task, "app": app, "screen_w": 1920, "screen_h": 1200,
            "grid": 32, "fps": 5}
    return Episode("/nonexistent", meta, frames, steps)


def base(app):
    return [Action(T.KEY, key="win"), Action(T.TYPE, text=app), Action(T.KEY, key="enter")]


def bad(ep):
    # ignore the "missing frame file" validator complaint; this episode lives in memory only
    return [m for l, m in audit_episode(ep) if l == "BAD" and "missing frame" not in m]


def test_good_open_app():
    assert bad(mk("open_app", "notepad", "Open Notepad", base("notepad") + [Action(T.DONE)])) == []


def test_stray_keys_flagged():
    acts = base("calculator") + [Action(T.CLICK, cell=1)] * 4 + [Action(T.KEY, key="caps_lock"),
                                                                Action(T.KEY, key="alt+tab"), Action(T.DONE)]
    assert any("unexpected keys" in m for m in bad(mk("calculator", "calculator", "Calculate 2 plus 3", acts)))


def test_calculator_without_clicks_flagged():
    acts = base("calculator") + [Action(T.DONE)]
    assert any("plan needs 8" in m for m in bad(mk("calculator", "calculator", "Calculate 2 plus 3", acts)))


def test_calculator_matching_its_plan_is_good():
    acts = base("calculator") + [Action(T.CLICK, cell=i) for i in (5, 6, 7, 8)] + [Action(T.DONE)]
    assert bad(mk("calculator", "calculator", "Calculate 8*6", acts)) == []
    assert any("plan needs 9" in m for m in bad(mk("calculator", "calculator", "Calculate 12 plus 6", acts)))


def test_paint_drag_counts():
    one = base("paint") + [Action(T.DRAG, cell=1, end_cell=9), Action(T.DONE)]
    assert bad(mk("paint", "paint", "Open Paint and draw a line", one)) == []
    assert any("plan needs 7" in m for m in bad(mk("paint", "paint", "Open Paint and draw a zigzag", one)))


def test_wrong_app_typed():
    acts = base("notepad") + [Action(T.DONE)]
    assert any("typed" in m for m in bad(mk("open_app", "paint", "Open Paint", acts)))
