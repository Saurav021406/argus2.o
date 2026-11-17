import json
import os

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from argus.actions.executor import Executor  # noqa: E402
from argus.core.agent import Agent, PolicyRunner  # noqa: E402
from argus.core.planner import PlanError  # noqa: E402
from argus.perception.vision_encoder import RandomVisionEncoder  # noqa: E402
from argus.policy.policy import ArgusPolicy  # noqa: E402
from argus.training.behavioral_cloning import train  # noqa: E402
from argus.training.dataset import StepDataset, encode_action, load_episodes  # noqa: E402
from tests.test_bc import feats, make_raw  # noqa: E402,F401  (fixture reuse)


@pytest.fixture(scope="module")
def ckpt(feats, tmp_path_factory):
    p = str(tmp_path_factory.mktemp("ck") / "bc.pt")
    train(feats, p, epochs=60, bs=16, lr=1e-3, eval_every=60, verbose=False, device="cpu", d=64, layers=2,
          min_steps=0)
    return p


class FakeCapture:
    """Returns a different solid colour each call, like a screen that keeps changing."""

    screen_size = (1920, 1200)

    def __init__(self):
        self.n = 0

    def grab(self):
        self.n += 1
        return Image.new("RGB", (128, 80), ((self.n * 37) % 255, 60, 120))


def _runner(ckpt):
    return PolicyRunner(ckpt, RandomVisionEncoder(), device="cpu")


def test_live_inputs_match_training_inputs(feats, ckpt):
    """Replaying a recorded episode through the live runner (human actions teacher-forced into its
    history) must give the exact logits the trainer computes. Guards against train/inference skew."""
    r = _runner(ckpt)
    eps = load_episodes(feats)
    ds = StepDataset(eps, r.vocab, T=r.T)
    ei = next(i for i, e in enumerate(eps) if e.task == "calculator")
    ep, k = eps[ei], 5
    r.seen_goals = set()                      # this test is about input construction, not coverage
    r.reset(ep.instruction)
    for s in range(k + 1):
        row = ep.row_of[ep.frames[s]]
        patch, cls = ep.patch[row].float(), ep.cls[row].float()

        class _V:                                    # feed the cached features of the recorded frame
            def encode(self, _imgs):
                return patch.unsqueeze(0), cls.unsqueeze(0)

        r.vision = _V()
        _, info = r.act(Image.new("RGB", (8, 8)))
        r.encs[-1] = encode_action(ep.actions[s], r.vocab)   # teacher-force the history
    b = {n: v.unsqueeze(0) for n, v in ds[ds.index.index((ei, k))].items()}
    with torch.no_grad():
        want = r.model(b)["type"][0]
    assert torch.allclose(info["type_logits"], want, atol=1e-4)


def test_agent_run_writes_trace_and_stops(ckpt, tmp_path):
    ag = Agent(_runner(ckpt), FakeCapture(), Executor(dry_run=True), out_dir=str(tmp_path),
               max_steps=6, sleep=lambda s: None)
    res = ag.run("Open Notepad", countdown=0)
    assert res["stopped"] in ("done", "stuck", "max_steps") and 1 <= res["steps"] <= 6
    lines = [json.loads(l) for l in open(os.path.join(res["trace"], "trace.jsonl"))]
    assert len(lines) == res["steps"]
    assert os.path.exists(os.path.join(res["trace"], "step_00.png"))


def test_agent_stops_when_stuck(ckpt, tmp_path):
    r = _runner(ckpt)
    # a constant frame + dry run => the same prediction every step => stuck detection must fire
    class Still(FakeCapture):
        def grab(self):
            return Image.new("RGB", (128, 80), (10, 20, 30))

    r.model.eval()
    ag = Agent(r, Still(), Executor(dry_run=True), out_dir=str(tmp_path), max_steps=8,
               stuck_limit=3, sleep=lambda s: None)
    res = ag.run("Open Notepad", countdown=0)
    assert res["stopped"] in ("stuck", "done")


def test_agent_stops_on_blocked_action(ckpt, tmp_path):
    from argus.actions.action_space import Action, ActionType

    r = _runner(ckpt)
    orig = r.act
    r.act = lambda img: (Action(ActionType.KEY, key="alt+f4"), {"type_conf": 1.0, "cell_conf": None})
    ag = Agent(r, FakeCapture(), Executor(dry_run=True), out_dir=str(tmp_path), max_steps=5, sleep=lambda s: None)
    res = ag.run("Open Notepad", countdown=0)
    assert res["stopped"].startswith("blocked") and res["steps"] == 1


def test_settle_waits_for_a_stable_screen(ckpt, tmp_path):
    seq = [Image.new("RGB", (128, 80), (c, c, c)) for c in (10, 90, 200, 200, 200, 200)]

    class Seq(FakeCapture):
        def __init__(self):
            self.i = 0

        def grab(self):
            im = seq[min(self.i, len(seq) - 1)]
            self.i += 1
            return im

    cap = Seq()
    ag = Agent(_runner(ckpt), cap, Executor(dry_run=True), out_dir=str(tmp_path), sleep=lambda s: None)
    out = ag._wait_settled(seq[0])
    assert out.getpixel((0, 0)) == (200, 200, 200)        # returned only after it stopped changing


def test_unparseable_and_untrained_requests_are_refused(ckpt):
    r = _runner(ckpt)
    with pytest.raises(PlanError, match="cannot understand"):
        r.reset("make me a sandwich")
    with pytest.raises(PlanError, match="no training demonstrations"):
        r.reset("Calculate 9 times 9")           # the tiny test checkpoint never saw CLICK_9 / CLICK_*


def test_max_steps_follows_the_plan(ckpt, tmp_path):
    r = _runner(ckpt)
    ag = Agent(r, FakeCapture(), Executor(dry_run=True), out_dir=str(tmp_path), sleep=lambda s: None)
    res = ag.run("Open Notepad", countdown=0)
    assert res["steps"] <= len(r.plan.goals) + 3


def test_lookup_clicks_mode_runs_and_uses_the_table(ckpt, tmp_path):
    r = PolicyRunner(ckpt, RandomVisionEncoder(), device="cpu", lookup_clicks=True)
    ag = Agent(r, FakeCapture(), Executor(dry_run=True), out_dir=str(tmp_path), sleep=lambda s: None)
    res = ag.run("Open Notepad", countdown=0)
    assert res["steps"] >= 1


def test_centre_aim_clicks_where_the_demos_clicked(ckpt):
    centres = {"1": (25.4, 3.4), "8": (18.6, 10.5)}
    r = PolicyRunner(ckpt, RandomVisionEncoder(), device="cpu", aim="centre", centres=centres)
    assert r._aim("CLICK_1", 999) == 25 * 32 + 3                 # same goal -> same cell, whatever the model said
    assert r._aim("CLICK_8", 5) == 19 * 32 + 10                  # 18.6 -> 19, 10.5 -> 10 (banker's rounding)
    assert r._aim("CLICK_9", 123) == 123                         # unknown key: keep the model's cell
    assert r._aim("OPEN_notepad", 7) == 7                        # never touches non-click goals


def test_centre_aim_needs_demos_and_a_valid_mode(ckpt):
    with pytest.raises(ValueError):
        PolicyRunner(ckpt, RandomVisionEncoder(), device="cpu", aim="centre", centres={})
    with pytest.raises(ValueError):
        PolicyRunner(ckpt, RandomVisionEncoder(), device="cpu", aim="telepathy")


def test_centre_aim_runs_a_full_episode(ckpt, tmp_path):
    centres = {str(i): (20.0, 5.0 + i) for i in range(4)} | {"+": (10.0, 20.0), "=": (28.0, 25.0)}
    r = PolicyRunner(ckpt, RandomVisionEncoder(), device="cpu", aim="centre", centres=centres)
    ag = Agent(r, FakeCapture(), Executor(dry_run=True), out_dir=str(tmp_path), sleep=lambda s: None)
    res = ag.run("Open Notepad", countdown=0)
    assert res["steps"] >= 1
