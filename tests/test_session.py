import os
from PIL import Image

from argus.actions.action_space import Action, ActionType
from argus.data import session
from argus.data.episode import Episode, Step


def test_plan_shape():
    p = session.PLAN_FIRST_BATCH
    assert len(p) == 29
    assert {t for t, *_ in p} == {"open_app", "calculator", "paint"}


def test_action_json_has_no_default_button():
    assert "button" not in Action(ActionType.KEY, key="win").to_dict()
    assert Action(ActionType.CLICK, cell=3, button="right").to_dict()["button"] == "right"
    assert Action.from_dict(Action(ActionType.CLICK, cell=3).to_dict()).button == "left"


def test_session_flow_with_fake_recorder(tmp_path, monkeypatch):
    made = []

    class FakeRecorder:
        def __init__(self, out_root, instr, task, app):
            self.out_root, self.instr, self.task, self.app = out_root, instr, task, app

        def run(self, countdown):
            d = os.path.join(self.out_root, f"ep{len(made)}")
            os.makedirs(d + "/frames")
            fr = []
            for i in range(4):
                Image.new("RGB", (64, 36)).save(f"{d}/frames/{i:06d}.jpg")
                fr.append({"i": i, "t": i * 0.2, "file": f"frames/{i:06d}.jpg"})
            meta = {"instruction": self.instr, "screen_w": 1920, "screen_h": 1200, "frame_w": 64,
                    "frame_h": 36, "grid": 32, "fps": 5, "task": self.task}
            steps = [Step(0, 0.1, 0, Action(ActionType.KEY, key="win")), Step(1, 0.5, 3, Action(ActionType.DONE))]
            ep = Episode(d, meta, fr, steps)
            ep.save_labels()
            made.append(d)
            return ep

    import argus.data.recorder as rec
    monkeypatch.setattr(rec, "Recorder", FakeRecorder)
    answers = iter(["", "d", "", "", "s", "q"])      # record, delete+redo, record again, keep, skip, quit
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    plan = session.PLAN_FIRST_BATCH[:3]
    session.run_session(str(tmp_path), plan)
    assert len(made) == 2 and not os.path.exists(made[0]) and os.path.exists(made[1])
