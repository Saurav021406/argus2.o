import json
import os
import random
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
from argus.actions.action_space import Action, ActionType as T  # noqa: E402
from argus.actions.executor import Executor  # noqa: E402
from argus.core.agent import Agent, PolicyRunner  # noqa: E402
from argus.core.dagger import (Expert, _suite_signatures, agrees, dagger_prompts, heldout_prompts, rollout,
                               run_dagger)  # noqa: E402
from argus.core.planner import PlanError, parse_instruction  # noqa: E402
from argus.data.audit import audit_episode  # noqa: E402
from argus.data.episode import Episode  # noqa: E402
from argus.data.validator import validate_episode  # noqa: E402
from argus.perception.vision_encoder import RandomVisionEncoder  # noqa: E402
from argus.training.behavioral_cloning import train  # noqa: E402
from argus.training.cache_features import cache_features  # noqa: E402
from argus.training.dataset import StepDataset, build_vocab, load_episodes, split_episodes  # noqa: E402
from tests.test_agent import FakeCapture, ckpt  # noqa: E402,F401
from tests.test_bc import feats  # noqa: E402,F401

CENTRES = {"1": (25.4, 3.4), "2": (25.2, 10.0), "+": (24.8, 21.1), "=": (28.0, 21.0), "8": (18.6, 10.5)}
G = 32


def _agent(ckpt, tmp_path):
    r = PolicyRunner(ckpt, RandomVisionEncoder(), device="cpu")
    r.seen_goals = set()                                   # the tiny test checkpoint saw only a few buttons
    return Agent(r, FakeCapture(), Executor(dry_run=True), out_dir=str(tmp_path / "runs"), sleep=lambda s: None)


def test_expert_follows_the_plan():
    e, plan = Expert(CENTRES), parse_instruction("Calculate 1 plus 2")
    acts = [e.action(plan, k) for k in range(len(plan.goals))]
    assert [a.type for a in acts[:3]] == [T.KEY, T.TYPE, T.KEY]
    assert acts[0].key == "win" and acts[1].text == "calculator" and acts[2].key == "enter"
    assert [a.cell for a in acts[3:7]] == [25 * G + 3, 25 * G + 21, 25 * G + 10, 28 * G + 21]   # 1 + 2 =
    assert acts[7].type == T.DONE
    assert Expert(CENTRES).action(parse_instruction("Calculate 5 plus 5"), 3) is None          # no centre for '5'
    assert Expert(CENTRES).action(parse_instruction("Open Paint and draw a line"), 3) is None   # no expert for strokes


def test_agreement_uses_key_sized_tolerance():
    exp = Action(T.CLICK, cell=20 * G + 10)
    assert agrees(Action(T.CLICK, cell=21 * G + 12), exp)           # 1 row, 2 columns: still the same key
    assert not agrees(Action(T.CLICK, cell=23 * G + 10), exp)       # 3 rows away: the key below
    assert not agrees(Action(T.KEY, key="win"), exp)
    assert agrees(Action(T.KEY, key="enter"), Action(T.KEY, key="enter"))
    assert not agrees(Action(T.TYPE, text="paint"), Action(T.TYPE, text="calculator"))


def test_prompts_are_valid_distinct_and_never_a_test_question():
    ps = dagger_prompts(30, seed=5)
    sigs = [tuple(parse_instruction(p).goals) for p in ps]
    assert len(ps) == 30 and len(set(sigs)) == 30
    assert not set(sigs) & _suite_signatures()
    assert ps == dagger_prompts(30, seed=5) and ps != dagger_prompts(30, seed=6)


def test_rollout_saves_a_valid_episode_labelled_by_the_expert(ckpt, tmp_path):
    out = str(tmp_path / "dag")
    res = rollout(_agent(ckpt, tmp_path), Expert(CENTRES), "Calculate 1 plus 2", out, iteration=3,
                  countdown=0, sleep=lambda s: None)
    ep = Episode.load(res["dir"])
    plan = parse_instruction("Calculate 1 plus 2")
    assert len(ep.steps) == len(ep.exec_steps) == len(ep.frames) == len(plan.goals) == res["steps"]
    assert ep.meta["source"] == "dagger" and ep.meta["iteration"] == 3 and 0 <= ep.meta["agreement"] <= 1
    assert [s.action.cell for s in ep.steps[3:7]] == [25 * G + 3, 25 * G + 21, 25 * G + 10, 28 * G + 21]
    assert validate_episode(ep) == [] and not [m for lv, m in audit_episode(ep) if lv == "BAD"]
    assert os.path.exists(os.path.join(res["dir"], "executed.jsonl"))


def test_history_comes_from_what_the_learner_did_but_targets_from_the_expert(ckpt, tmp_path):
    out = str(tmp_path / "dag")
    res = rollout(_agent(ckpt, tmp_path), Expert(CENTRES), "Calculate 1 plus 2", out, countdown=0, sleep=lambda s: None)
    p = os.path.join(res["dir"], "executed.jsonl")
    rows = [json.loads(l) for l in open(p)]
    rows[3]["action"] = {"type": "CLICK", "cell": 7}                  # the learner pressed the wrong place at step 3
    open(p, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
    feat = str(tmp_path / "feat")
    cache_features(out, feat, RandomVisionEncoder())
    eps = load_episodes(feat)
    assert len(eps) == 1 and eps[0].source == "dagger"
    ds = StepDataset(eps, build_vocab(eps), T=6)
    assert int(ds[3]["t_cell"]) == 25 * G + 3                         # target at step 3 = the expert's cell
    assert int(ds[4]["pcell"][-1]) == 7 + 1                           # history at step 4 = what was really clicked


def test_dagger_episodes_are_never_validation(feats, ckpt, tmp_path):
    base = load_episodes(feats)
    dag = [replace(e, id=f"dag{i}", source="dagger") for i, e in enumerate(base)]
    tr, va = split_episodes(base + dag, val_frac=0.4, seed=1)
    assert all(e.source != "dagger" for e in va) and sum(e.source == "dagger" for e in tr) == len(dag)


def test_training_accepts_human_and_dagger_feature_folders(feats, ckpt, tmp_path):
    out = str(tmp_path / "dag")
    rollout(_agent(ckpt, tmp_path), Expert(CENTRES), "Calculate 1 plus 2", out, countdown=0, sleep=lambda s: None)
    dfeat = str(tmp_path / "dfeat")
    cache_features(out, dfeat, RandomVisionEncoder())
    r = train([feats, dfeat], str(tmp_path / "m.pt"), epochs=2, eval_every=2, verbose=False, device="cpu",
              d=32, layers=1, min_steps=0)
    assert os.path.exists(str(tmp_path / "m.pt")) and "model" in r["final"]


def test_run_dagger_skips_questions_the_model_cannot_do(ckpt, tmp_path, capsys):
    ag = _agent(ckpt, tmp_path)
    ag.runner.seen_goals = {"OPEN_notepad", "DONE"}                   # now 'Calculate ...' needs untrained buttons
    out = run_dagger(ag, Expert(CENTRES), ["Calculate 1 plus 2"], str(tmp_path / "dag"), sleep=lambda s: None,
                     countdown=0)
    assert out["skipped"] == ["Calculate 1 plus 2"] and out["rollouts"] == []


def test_heldout_prompts_are_in_no_dagger_round_and_no_suite_question():
    held = heldout_prompts(40)
    sigs = [tuple(parse_instruction(p).goals) for p in held]
    assert len(held) == 40 and len(set(sigs)) == 40
    assert not set(sigs) & _suite_signatures()
    for it in range(1, 11):                                          # every DAgger round the CLI can run
        used = {tuple(parse_instruction(p).goals) for p in dagger_prompts(60, 100 + it)}
        assert not set(sigs) & used, f"overlaps DAgger iteration {it}"
    assert held == heldout_prompts(40) and held != heldout_prompts(40, seed=1)


def test_the_first_24_prompts_do_not_depend_on_how_many_are_asked_for():
    # a user who ran `--n 24` must be covered by the 60 we exclude
    assert dagger_prompts(60, 101)[:24] == dagger_prompts(24, 101)
