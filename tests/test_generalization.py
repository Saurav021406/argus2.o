"""The point of the planner: perform on expressions that were NEVER in the training set.

Frames are identical for every episode of a step (the screen carries NO information about
which expression is being solved), so the only way to click the right button is to use the
sub-goal. Validation expressions are all unseen combinations.
"""
import os
import random

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from argus.actions.action_space import Action, ActionType as T  # noqa: E402
from argus.core.planner import parse_instruction  # noqa: E402
from argus.data.episode import Episode, Step  # noqa: E402
from argus.perception.vision_encoder import RandomVisionEncoder  # noqa: E402
from argus.training.behavioral_cloning import train  # noqa: E402
from argus.training.cache_features import cache_features  # noqa: E402

CELL = {b: 40 + 37 * i for i, b in enumerate("0123456789+-*/=")}      # fixed keypad layout


def make_calc_raw(root, n=24, seed=1):
    rng = random.Random(seed)
    seen = set()
    while len(seen) < n:
        seen.add((rng.randrange(10), rng.choice("+-*/"), rng.randrange(1, 10)))
    for j, (a, op, b) in enumerate(sorted(seen)):
        text = rng.choice([f"Calculate {a}{op}{b}", f"{a} {op} {b}", f"Calculate {a} {op} {b}"])
        plan = parse_instruction(text)
        acts = [Action(T.KEY, key="win"), Action(T.TYPE, text="calculator"), Action(T.KEY, key="enter")]
        acts += [Action(T.CLICK, cell=CELL[g[6:]]) for g in plan.goals if g.startswith("CLICK_")]
        acts.append(Action(T.DONE))
        d = os.path.join(root, f"c{j:03d}")
        os.makedirs(d + "/frames")
        frames, steps = [], []
        for s, act in enumerate(acts):
            Image.new("RGB", (64, 40), (30, 20 * s + 5, 90)).save(f"{d}/frames/{s:06d}.jpg")   # step-only colour
            frames.append({"i": s, "t": s * 0.5, "file": f"frames/{s:06d}.jpg"})
            steps.append(Step(s, s * 0.5, s, act))
        meta = {"id": f"c{j:03d}", "instruction": text, "task": "calculator", "app": "calculator",
                "screen_w": 1920, "screen_h": 1200, "frame_w": 64, "frame_h": 40, "grid": 32, "fps": 5}
        Episode(d, meta, frames, steps).save_labels()


def test_generalises_to_unseen_expressions(tmp_path):
    raw, feat = str(tmp_path / "raw"), str(tmp_path / "feat")
    make_calc_raw(raw)
    cache_features(raw, feat, RandomVisionEncoder(dim=32, gh=10, gw=16))
    r = train(feat, str(tmp_path / "ck.pt"), epochs=40, bs=16, lr=2e-3, eval_every=40, verbose=False,
              device="cpu", d=64, layers=2, val_frac=0.3, min_steps=0)
    m, base = r["final"]["model"], r["final"]["baseline"]
    assert m["n"] >= 40                                    # 7 held-out expressions, never trained on
    assert m["step_acc"] >= 0.9, m                         # clicks the right buttons for unseen numbers
    assert m["step_acc"] > base["step_acc"] + 0.15         # and it is not just step-index statistics


def make_press_raw(root, n=6, seed=100):
    from argus.data.session import _press_plan

    for j, (_, _, text, _, _) in enumerate(_press_plan(n, seed)):
        _write(root, f"p{j:03d}", text)


def _write(root, eid, text, realistic=False):
    """realistic=True mimics real recordings: every frame is different (the calculator display changes
    in each demo) and human clicks scatter +-1 cell inside the key. A net can then MEMORISE
    frame -> cell and ignore the goal, which is what failed on '8*6' in the live test."""
    import random as _r
    plan = parse_instruction(text)
    acts = [Action(T.KEY, key="win"), Action(T.TYPE, text="calculator"), Action(T.KEY, key="enter")]
    acts += [Action(T.CLICK, cell=CELL[g[6:]] + (_r.Random(f"{eid}{i}").choice([-1, 0, 0, 1]) if realistic else 0))
             for i, g in enumerate(plan.goals) if g.startswith("CLICK_")]
    acts.append(Action(T.DONE))
    d = os.path.join(root, eid)
    os.makedirs(d + "/frames")
    frames, steps = [], []
    for s, act in enumerate(acts):
        if realistic:
            rr = _r.Random(f"{eid}-{s}")
            color = (rr.randrange(256), rr.randrange(256), rr.randrange(256))
        else:
            color = (30, 20 * min(s, 9) + 5, 90)
        Image.new("RGB", (64, 40), color).save(f"{d}/frames/{s:06d}.jpg")
        frames.append({"i": s, "t": s * 0.5, "file": f"frames/{s:06d}.jpg"})
        steps.append(Step(s, s * 0.5, s, act))
    meta = {"id": eid, "instruction": text, "task": "calculator", "app": "calculator", "screen_w": 1920,
            "screen_h": 1200, "frame_w": 64, "frame_h": 40, "grid": 32, "fps": 5}
    Episode(d, meta, frames, steps).save_labels()


def test_press_plan_covers_every_button_in_fresh_orders():
    from argus.data.session import PLAN_PRESS

    seqs = [parse_instruction(t).goals[3:-1] for _, _, t, _, _ in PLAN_PRESS]
    assert all(sorted(s) == sorted([f"CLICK_{b}" for b in "0123456789+-*/="]) for s in seqs)
    assert len({tuple(s) for s in seqs}) == len(seqs)              # every episode a different order
    assert len({s[0] for s in seqs}) == len(seqs)                   # and a different first button
    assert all(s[-1] == "CLICK_=" for s in seqs)


def test_press_episodes_teach_unseen_real_expressions(tmp_path):
    """Train on 6 long 'press every button' episodes + 3 short real ones, test on 8 unseen expressions."""
    raw, feat = str(tmp_path / "raw"), str(tmp_path / "feat")
    make_press_raw(raw)
    for j, t in enumerate(["Calculate 2 plus 3", "Calculate 9 minus 4", "Calculate 6 times 7"]):
        _write(raw, f"s{j}", t)
    held = ["Calculate 8*6", "7 x 9", "Calculate 5+8", "Calculate 0 plus 1", "Calculate 1 - 0",
            "Calculate 4 / 2", "9*9", "Calculate 3 plus 6"]
    for j, t in enumerate(held):
        _write(raw, f"v{j}", t)
    cache_features(raw, feat, RandomVisionEncoder(dim=32, gh=10, gw=16))
    r = train(feat, str(tmp_path / "ck.pt"), epochs=40, bs=16, lr=2e-3, eval_every=40, verbose=False,
              device="cpu", d=64, layers=2, val_ids={f"v{j}" for j in range(len(held))}, min_steps=0)
    m, base = r["final"]["model"], r["final"]["baseline"]
    assert m["n"] == 8 * 8
    assert m["step_acc"] >= 0.9, m
    assert m["step_acc"] > base["step_acc"] + 0.15


SHORT = [f"Calculate {a} {w} {b}" for a, w, b in [(2, "plus", 3), (9, "minus", 4), (6, "times", 7), (1, "plus", 5),
                                                  (4, "minus", 2), (7, "plus", 8), (3, "times", 9), (5, "divided by", 5),
                                                  (0, "plus", 9), (8, "minus", 1)]]
HELD = ["Calculate 8*6", "7 x 9", "Calculate 5+8", "Calculate 0 plus 1", "Calculate 1 - 0", "Calculate 4 / 2",
        "9*9", "Calculate 3 plus 6"]


def build_realistic(raw):
    from argus.data.session import _press_plan

    for j, (_, _, t, _, _) in enumerate(_press_plan(6)):
        _write(raw, f"p{j}", t, realistic=True)
    for j, t in enumerate(SHORT):
        _write(raw, f"s{j}", t, realistic=True)
    for j, t in enumerate(HELD):
        _write(raw, f"v{j}", t, realistic=True)
    return {f"v{j}" for j in range(len(HELD))}


def test_does_not_memorise_frames_when_every_frame_is_different(tmp_path):
    raw, feat = str(tmp_path / "raw"), str(tmp_path / "feat")
    val_ids = build_realistic(raw)
    cache_features(raw, feat, RandomVisionEncoder(dim=32, gh=10, gw=16))
    r = train(feat, str(tmp_path / "ck.pt"), eval_every=100, verbose=False, device="cpu", d=64, layers=2,
              val_ids=val_ids)                                   # DEFAULT optimiser settings on purpose
    m, base, dg = r["final"]["model"], r["final"]["baseline"], r["final"]["diag"]
    assert m["step_acc"] >= 0.85, m
    assert m["step_acc"] > base["step_acc"] + 0.2
    assert dg["goal_swap_changes"] >= 0.8 and dg["lookup_within1"] >= 0.7, dg   # it really reads the goal


def test_extras_plan_presses_every_extra_key_each_episode_in_new_orders():
    from argus.core.planner import EXTRA_BUTTONS, align_goals
    from argus.data.session import PLAN_EXTRAS

    seqs = []
    for _, _, text, _, n in PLAN_EXTRAS:
        plan = parse_instruction(text)
        assert len(plan.goals) == n
        seq = [g[6:] for g in plan.goals if g.startswith("CLICK_")]
        assert set(EXTRA_BUTTONS) <= set(seq)                       # every extra key appears
        assert seq[-1] == "C"                                       # leaves the calculator clean
        seqs.append(tuple(seq))
        acts = [Action(T.KEY, key="win"), Action(T.TYPE, text="calculator"), Action(T.KEY, key="enter")]
        acts += [Action(T.CLICK, cell=1) for _ in seq] + [Action(T.DONE)]
        align_goals(plan, acts)
    assert len(set(seqs)) == len(seqs)


def test_second_recording_batches_use_new_orders():
    from argus.data.session import PLAN_EXTRAS, PLAN_EXTRAS2, PLAN_PRESS, PLAN_PRESS2

    assert len(PLAN_PRESS2) == len(PLAN_EXTRAS2) == 4
    assert not {t for _, _, t, _, _ in PLAN_PRESS} & {t for _, _, t, _, _ in PLAN_PRESS2}
    assert not {t for _, _, t, _, _ in PLAN_EXTRAS} & {t for _, _, t, _, _ in PLAN_EXTRAS2}
    for _, _, t, _, n in PLAN_PRESS2 + PLAN_EXTRAS2:
        assert len(parse_instruction(t).goals) == n
