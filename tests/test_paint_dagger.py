import json
import os
import random

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from argus.actions.action_space import Action, ActionType as T  # noqa: E402
from argus.core.dagger import Expert, agrees, paint_prompts, rollout  # noqa: E402
from argus.core.explain import explain_run  # noqa: E402
from argus.core.planner import parse_instruction  # noqa: E402
from argus.data.audit import audit_episode  # noqa: E402
from argus.data.episode import Episode, Step  # noqa: E402
from argus.data.strokes import collect_strokes, report, stroke_centres  # noqa: E402
from argus.data.validator import validate_episode  # noqa: E402
from tests.test_agent import ckpt  # noqa: E402,F401
from tests.test_bc import feats  # noqa: E402,F401  (fixture the ckpt fixture needs)
from tests.test_dagger import _agent  # noqa: E402
from tests.test_strokes import G, STROKES, _cell  # noqa: E402


def write_paint(root, eid, text, strokes):
    plan = parse_instruction(text)
    acts = [Action(T.KEY, key="win"), Action(T.TYPE, text="paint"), Action(T.KEY, key="enter")]
    acts += [Action(T.DRAG, cell=a, end_cell=b) for a, b in strokes] + [Action(T.DONE)]
    assert len(acts) == len(plan.goals)
    d = os.path.join(root, eid)
    os.makedirs(d + "/frames")
    frames, steps = [], []
    for s, a in enumerate(acts):
        Image.new("RGB", (64, 40), (s * 20, 10, 10)).save(f"{d}/frames/{s:06d}.jpg")
        frames.append({"i": s, "t": s * 0.5, "file": f"frames/{s:06d}.jpg"})
        steps.append(Step(s, s * 0.5, s, a))
    meta = {"id": eid, "instruction": text, "task": "paint", "app": "paint", "screen_w": 1920, "screen_h": 1200,
            "frame_w": 64, "frame_h": 40, "grid": 32, "fps": 5}
    Episode(d, meta, frames, steps).save_labels()


def zigzag_demos(root, n=6, jitter=1):
    rng = random.Random(3)
    for i in range(n):
        dr, dc = rng.randint(-jitter, jitter), rng.randint(-jitter, jitter)
        strokes = [(_cell((a[0] + dr, a[1] + dc)), _cell((b[0] + dr, b[1] + dc))) for a, b in STROKES]
        write_paint(root, f"z{i}", "Open Paint and draw a zigzag", strokes)


def test_stroke_centres_recover_the_three_strokes(tmp_path):
    zigzag_demos(str(tmp_path))
    found = collect_strokes(str(tmp_path))
    assert set(found) == {"DRAW_zigzag_1", "DRAW_zigzag_2", "DRAW_zigzag_3"} and all(len(v) == 6 for v in found.values())
    cen = stroke_centres(str(tmp_path))
    for k, (a, b) in enumerate(STROKES, start=1):
        s, e = cen[f"DRAW_zigzag_{k}"]
        assert abs(s // G - a[0]) <= 1 and abs(s % G - a[1]) <= 1 and abs(e // G - b[0]) <= 1 and abs(e % G - b[1]) <= 1


def test_report_calls_consistent_demos_consistent_and_scattered_ones_scattered(tmp_path, capsys):
    zigzag_demos(str(tmp_path / "good"))
    info = report(str(tmp_path / "good"))
    out = capsys.readouterr().out
    assert "demos agree" in out and "chained" in out and "All stroke goals are consistent" in out
    assert all(v["spread"] <= 4 for v in info.values())

    rng = random.Random(1)                                           # a line drawn somewhere different every time
    for i in range(6):
        a, b = rng.randint(100, 900), rng.randint(100, 900)
        write_paint(str(tmp_path / "bad"), f"l{i}", "Open Paint and draw a line", [(a, b)])
    report(str(tmp_path / "bad"))
    out = capsys.readouterr().out
    assert "SPREAD" in out and "Demos disagree for: DRAW_line" in out


def _trace(tmp_path, strokes):
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    rows = [{"step": i, "action": {"type": "KEY", "key": "win"}} for i in range(3)]
    for i, (a, b) in enumerate(strokes, start=3):
        rows.append({"step": i, "action": {"type": "DRAG", "cell": a, "end_cell": b}})
    with open(d / "trace.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(d)


def test_explain_flags_a_stroke_that_ends_in_the_screen_corner(tmp_path):
    zigzag_demos(str(tmp_path / "raw"))
    good = [(_cell(a), _cell(b)) for a, b in STROKES]
    lines = explain_run(_trace(tmp_path, good), "Open Paint and draw a zigzag", str(tmp_path / "raw"))
    assert len(lines) == 3 and all(l.endswith("ok") for l in lines)
    bad = list(good)
    bad[0] = (good[0][0], 31)                                        # the live failure: end at row 0, col 31
    lines = explain_run(_trace(tmp_path, bad), "Open Paint and draw a zigzag", str(tmp_path / "raw"))
    far = [l for l in lines if "FAR FROM THE DEMOS" in l]
    assert len(far) == 1 and "DRAW_zigzag_1" in far[0] and "to (0,31)" in far[0]


def test_expert_strokes_and_drag_agreement():
    strokes = {"DRAW_zigzag_1": (_cell((10, 5)), _cell((16, 11)))}
    e = Expert({}, strokes=strokes)
    plan = parse_instruction("Open Paint and draw a zigzag")
    a = e.action(plan, 3)
    assert a.type == T.DRAG and a.cell == _cell((10, 5)) and a.end_cell == _cell((16, 11))
    assert e.action(plan, 4) is None                                  # no demos for stroke 2 -> no expert
    near = Action(T.DRAG, cell=_cell((12, 8)), end_cell=_cell((19, 14)))
    far = Action(T.DRAG, cell=_cell((12, 8)), end_cell=_cell((0, 31)))
    assert agrees(near, a) and not agrees(far, a)
    assert not agrees(Action(T.CLICK, cell=_cell((10, 5))), a)


def test_paint_prompts_alternate():
    ps = paint_prompts(6)
    assert ps == ["Open Paint and draw a zigzag", "Open Paint and draw a line"] * 3


def test_a_paint_rollout_is_a_valid_episode_labelled_with_the_demo_strokes(ckpt, tmp_path):
    strokes = {f"DRAW_zigzag_{k}": (_cell(a), _cell(b)) for k, (a, b) in enumerate(STROKES, start=1)}
    res = rollout(_agent(ckpt, tmp_path), Expert({}, strokes=strokes), "Open Paint and draw a zigzag",
                  str(tmp_path / "dag"), countdown=0, sleep=lambda s: None)
    ep = Episode.load(res["dir"])
    assert len(ep.steps) == len(ep.exec_steps) == 7
    assert [s.action.type for s in ep.steps[3:6]] == [T.DRAG] * 3
    assert [s.action.cell for s in ep.steps[3:6]] == [_cell(a) for a, _ in STROKES]
    assert validate_episode(ep) == [] and not [m for lv, m in audit_episode(ep) if lv == "BAD"]
