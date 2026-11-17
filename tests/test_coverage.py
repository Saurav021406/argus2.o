import pytest

torch = pytest.importorskip("torch")
from argus.data.coverage import MAX_SPREAD, collect, report  # noqa: E402
from argus.data.episode import Episode  # noqa: E402
from tests.test_generalization import CELL, make_press_raw  # noqa: E402


def test_all_ok_when_every_button_is_covered_and_consistent(tmp_path, capsys):
    make_press_raw(str(tmp_path))
    assert report(str(tmp_path)) is True
    out = capsys.readouterr().out
    assert "All core calculator buttons" in out and "EXTRA buttons" in out


def test_far_click_is_flagged_and_never_reported_as_fine(tmp_path, capsys):
    make_press_raw(str(tmp_path))
    ep = Episode.load(str(tmp_path / "p000"))
    goals_idx = next(i for i, s in enumerate(ep.steps) if s.action.cell == CELL["5"])
    ep.steps[goals_idx].action.cell = CELL["5"] + 32 * (MAX_SPREAD + 3)        # many rows away
    ep.save_labels()
    assert report(str(tmp_path)) is False
    out = capsys.readouterr().out
    assert "INCONSISTENT" in out and "farthest pair" in out and "p000" in out
    assert "All core calculator buttons" not in out


def test_missing_button_is_reported(tmp_path, capsys):
    make_press_raw(str(tmp_path), n=1)                     # one episode: every button has 1 click only
    assert report(str(tmp_path)) is False
    assert "Not enough clicks" in capsys.readouterr().out


def test_normal_within_key_scatter_is_not_flagged(tmp_path):
    make_press_raw(str(tmp_path))
    ep = Episode.load(str(tmp_path / "p001"))
    i = next(i for i, s in enumerate(ep.steps) if s.action.cell == CELL["0"])
    ep.steps[i].action.cell = CELL["0"] + 4                                    # 4 cells off: same wide key
    ep.save_labels()
    assert report(str(tmp_path)) is True
