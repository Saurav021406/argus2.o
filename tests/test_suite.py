import json
import types

import pytest

from argus.core.planner import PlanError, parse_instruction
from argus.core.suite import SUITE, parse_number, run_suite


def test_expected_answers():
    exp = lambda t: parse_instruction(t).expected()          # noqa: E731
    assert exp("Calculate 8*6") == 48 and exp("9 into 3") == 27 and exp("Calculate 8 divided by 2") == 4
    assert exp("Calculate 2+3*4") == 20                      # Standard mode is left to right
    assert exp("34 + 66") == 100 and exp("Press 3 + 4 =") == 7
    assert exp("Calculate 5 divided by 0") is None and exp("Open Notepad") is None
    assert exp("square root of 81") == 9 and exp("Calculate 3.5 + 2") == 5.5 and exp("10% of 50") is None


@pytest.mark.parametrize("text,val", [("48", 48), ("1,234", 1234), ("\u22125", -5), ("12.5", 12.5), (" 0 ", 0),
                                      ("\u200e19\r\n", 19), ("ARGUS_NONE", None), ("", None)])
def test_parse_number(text, val):
    assert parse_number(text) == val


def test_suite_questions_are_what_they_claim():
    assert len(SUITE) == len(set(SUITE))
    for level, text in SUITE:
        if level == 4:
            with pytest.raises(PlanError):
                parse_instruction(text)
        elif level == 7:
            assert parse_instruction(text).task == "paint"                  # judged by the ink check, not an answer
        elif "%" in text:
            assert parse_instruction(text).expected() is None, text         # judged by you (y/n)
        else:
            assert parse_instruction(text).expected() is not None, text


class FakeAgent:
    def __init__(self, wrong=(), stop=None):
        self.wrong, self.stop = set(wrong), stop
        self.runner = types.SimpleNamespace(reset=parse_instruction)
        self.ran, self.value = [], None

    def run(self, text, countdown=0):
        self.ran.append(text)
        exp = parse_instruction(text).expected()
        self.value = (exp + 1) if text in self.wrong else exp
        return {"stopped": self.stop or "done", "steps": 8, "trace": "runs/x"}


def _run(agent, items, tmp_path, **kw):
    return run_suite(agent, items, lambda: agent.value, lambda: None, ask=lambda _: "y", out_dir=str(tmp_path),
                     sleep=lambda s: None, **kw)


def test_scoring_per_level(tmp_path):
    items = [(1, "Calculate 8*6"), (1, "7 x 9"), (2, "Calculate 0 plus 7"), (4, "make me a sandwich")]
    ag = FakeAgent(wrong={"7 x 9"})
    out = _run(ag, items, tmp_path)
    assert out["by_level"] == {1: [1, 2], 2: [1, 1], 4: [1, 1]}
    assert "make me a sandwich" not in ag.ran                  # refusal test never moves the mouse


def test_agent_that_does_not_finish_fails(tmp_path):
    out = _run(FakeAgent(stop="stuck"), [(1, "Calculate 8*6")], tmp_path)
    assert out["by_level"] == {1: [0, 1]} and out["rows"][0]["stopped"] == "stuck"


def test_repeat_and_results_file(tmp_path):
    out = _run(FakeAgent(), [(1, "Calculate 8*6")], tmp_path, repeat=3)
    assert out["by_level"] == {1: [3, 3]}
    assert len((tmp_path / "results.jsonl").read_text().strip().splitlines()) == 3


def test_manual_mode_uses_the_answer_you_give(tmp_path):
    out = run_suite(FakeAgent(), [(1, "Calculate 8*6")], lambda: None, lambda: None, ask=lambda _: "n",
                    out_dir=str(tmp_path), sleep=lambda s: None, manual=True)
    assert out["by_level"] == {1: [0, 1]}


class PaintAgent:
    """Fake agent whose screen shows what a run drew."""

    def __init__(self, draw):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (1280, 800), (32, 32, 32))
        d = ImageDraw.Draw(img)
        d.rectangle([120, 170, 1200, 740], fill=(255, 255, 255))
        draw(d)
        self.capture = types.SimpleNamespace(grab=lambda: img)
        self.runner = types.SimpleNamespace(reset=parse_instruction)

    def run(self, text, countdown=0):
        return {"stopped": "done", "steps": 5, "trace": "runs/p"}


def test_paint_runs_are_judged_by_the_ink_not_by_you(tmp_path):
    line = PaintAgent(lambda d: d.line([300, 300, 760, 450], fill=(0, 0, 0), width=3))
    out = run_suite(line, [(7, "Open Paint and draw a line"), (7, "Open Paint and draw a zigzag")],
                    lambda: None, lambda: None, ask=lambda _: "y", out_dir=str(tmp_path), sleep=lambda s: None)
    rows = {r["instruction"]: r for r in out["rows"]}
    assert rows["Open Paint and draw a line"]["success"] is True and rows["Open Paint and draw a line"]["task"] == "paint"
    assert rows["Open Paint and draw a zigzag"]["success"] is False       # one straight stroke is not a zigzag


def test_results_record_which_aim_was_used(tmp_path):
    run_suite(FakeAgent(), [(1, "Calculate 8*6")], lambda: 48.0, lambda: None, ask=lambda _: "y",
              out_dir=str(tmp_path), sleep=lambda s: None, aim="model")
    row = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[0])
    assert row["aim"] == "model" and row["level"] == 1


def test_paint_run_saves_the_final_canvas_for_the_report(tmp_path):
    trace = tmp_path / "trace"
    trace.mkdir()
    agent = PaintAgent(lambda d: d.line([300, 300, 760, 450], fill=(0, 0, 0), width=3))
    agent.run = lambda text, countdown=0: {"stopped": "done", "steps": 5, "trace": str(trace)}
    run_suite(agent, [(7, "Open Paint and draw a line")], lambda: None, lambda: None, ask=lambda _: "y",
              out_dir=str(tmp_path), sleep=lambda s: None)
    assert (trace / "final_canvas.png").exists()
