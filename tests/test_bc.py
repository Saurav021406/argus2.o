import os

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from argus.actions.action_space import Action, ActionType as T  # noqa: E402
from argus.data.episode import Episode, Step  # noqa: E402
from argus.language.instruction_encoder import RandomInstructionEncoder  # noqa: E402
from argus.perception.vision_encoder import RandomVisionEncoder  # noqa: E402
from argus.policy.policy import ArgusPolicy, bc_loss, soft_targets  # noqa: E402
from argus.training.behavioral_cloning import train  # noqa: E402
from argus.training.cache_features import cache_features  # noqa: E402
from argus.training.dataset import StepDataset, build_vocab, load_episodes  # noqa: E402

TASKS = {"open_app": 0, "calculator": 1, "paint": 2}


def make_raw(root, n_per_task=4):
    """Fake demos shaped like real recordings. Frames are solid colours per (task, step)."""
    made = 0
    for task, ti in TASKS.items():
        for j in range(n_per_task):
            app = ["notepad", "calculator", "paint"][ti if task != "open_app" else j % 3]
            acts = [Action(T.KEY, key="win"), Action(T.TYPE, text=app), Action(T.KEY, key="enter")]
            if task == "calculator":
                acts += [Action(T.CLICK, cell=c) for c in (700, 710, 720, 730)]
                instr = f"Calculate {j} plus {j}"
            elif task == "paint":
                acts += [Action(T.DRAG, cell=300, end_cell=600)]
                instr = "Open Paint and draw a line"
            else:
                instr = f"Open {app.capitalize()}"
            acts.append(Action(T.DONE))
            d = os.path.join(root, f"ep_{task}_{j}")
            os.makedirs(d + "/frames")
            frames, steps = [], []
            for s, a in enumerate(acts):
                color = (40 * ti + 10, 25 * s + 5, 90)
                Image.new("RGB", (64, 40), color).save(f"{d}/frames/{s:06d}.jpg")
                frames.append({"i": s, "t": s * 0.5, "file": f"frames/{s:06d}.jpg"})
                steps.append(Step(s, s * 0.5, s, a))
            meta = {"id": f"ep_{task}_{j}", "instruction": instr, "task": task, "app": app,
                    "screen_w": 1920, "screen_h": 1200, "frame_w": 64, "frame_h": 40, "grid": 32, "fps": 5}
            Episode(d, meta, frames, steps).save_labels()
            made += 1
    return made


@pytest.fixture(scope="module")
def feats(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("feat")
    n = make_raw(str(root))
    stats = cache_features(str(root), str(out), RandomVisionEncoder(), RandomInstructionEncoder())
    assert stats["cached"] == n and stats["skipped_bad"] == 0
    return str(out)


def test_soft_targets_sum_to_one_and_peak():
    t = soft_targets(torch.tensor([0, 500, 1023]), 32, 1.0)
    assert torch.allclose(t.sum(1), torch.ones(3), atol=1e-5)
    assert t.argmax(1).tolist() == [0, 500, 1023]


def test_cache_skips_bad_episodes(tmp_path):
    root, out = tmp_path / "raw", tmp_path / "f"
    make_raw(str(root), n_per_task=1)
    # break one: add a stray caps_lock key to the calculator demo
    ep = Episode.load(str(root / "ep_calculator_0"))
    ep.steps.insert(3, Step(3, 1.0, 3, Action(T.KEY, key="caps_lock")))
    for i, s in enumerate(ep.steps):
        s.step = i
    ep.save_labels()
    st = cache_features(str(root), str(out), RandomVisionEncoder(), RandomInstructionEncoder())
    assert st["skipped_bad"] == 1 and st["cached"] == 2


def test_dataset_shapes_and_history(feats):
    eps = load_episodes(feats)
    vocab = build_vocab(eps)
    assert set(vocab.keys) == {"enter", "win"} and "notepad" in vocab.texts
    ds = StepDataset(eps, vocab, T=6)
    b = ds[0]                                   # first step of an episode: no history, no previous action
    assert b["cls"].shape == (6, 384) and b["patch"].shape == (20, 32, 384)
    assert int(b["mask"].sum()) == 1 and int(b["ptype"].sum()) == 0
    k = next(i for i, (e, s) in enumerate(ds.index) if s == 3)
    b = ds[k]                                   # step 3: three history frames + current
    assert int(b["mask"].sum()) == 4 and int(b["ptype"][-1]) > 0 and bool(b["mask"][-4:].all())


def test_forward_shapes_and_causality(feats):
    eps = load_episodes(feats)
    vocab = build_vocab(eps)
    ds = StepDataset(eps, vocab)
    m = ArgusPolicy(len(vocab.keys), len(vocab.texts)).eval()
    b = {k: v.unsqueeze(0) for k, v in ds[5].items()}
    o = m(b)
    assert o["type"].shape == (1, 7) and o["loc"].shape == (1, 2, 1024)
    loss, parts = bc_loss(o, b)
    assert loss.requires_grad and set(parts) == {"type", "loc", "end", "key", "text", "loc_goal", "end_goal"}
    # the output must not depend on future tokens: perturb nothing after the last position -> trivially
    # true by construction, so instead check padded slots are ignored:
    b2 = {k: v.clone() for k, v in b.items()}
    b2["cls"][~b2["mask"]] += 5.0               # garbage in padded history slots
    assert torch.allclose(m(b2)["type"], o["type"], atol=1e-5)


def test_training_overfits_tiny_set(feats, tmp_path):
    out = str(tmp_path / "bc.pt")
    r = train(feats, out, epochs=80, bs=16, lr=1e-3, eval_every=40, verbose=False, device="cpu", d=64, layers=2,
              min_steps=0)
    assert os.path.exists(out)
    # loss cannot reach 0: four location terms use Gaussian-smoothed targets (sigma=2), an entropy floor of
    # roughly 4 nats each. So only require a clear drop; the accuracy asserts below are the real check.
    assert r["history"][-1]["loss"] < r["history"][0]["loss"] * 0.7
    assert r["final"]["train"]["step_acc"] > 0.9
    assert r["final"]["model"]["step_acc"] >= r["final"]["baseline"]["step_acc"]
    ck = torch.load(out)
    assert ck["vocab"]["keys"] and ck["cfg"]["T"] == 6
