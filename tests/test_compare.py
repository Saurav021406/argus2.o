import json

import pytest

from argus.core.compare import fisher_two_sided, load_rows, summarise, table, wilson


def test_wilson_interval_matches_known_values():
    lo, hi = wilson(26, 26)
    assert 0.86 < lo < 0.88 and hi == pytest.approx(1.0, abs=1e-9)        # perfect score still has a floor
    lo, hi = wilson(18, 26)
    assert 0.49 < lo < 0.51 and 0.82 < hi < 0.84
    assert wilson(0, 0) == (0.0, 0.0)


def test_fisher_exact_known_values():
    assert fisher_two_sided(26, 26, 18, 26) == pytest.approx(0.0042, abs=2e-4)    # the real BC vs DAgger-1 counts
    assert fisher_two_sided(10, 20, 10, 20) == pytest.approx(1.0)
    assert fisher_two_sided(20, 20, 0, 20) < 1e-6


def _rows(tmp_path, rows):
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return load_rows(str(p))


def test_repeats_collapse_and_latest_run_wins(tmp_path):
    rows = _rows(tmp_path, [
        {"level": 1, "instruction": "a", "ckpt": "runs/bc_v5.pt", "aim": "model", "success": False},
        {"level": 1, "instruction": "a", "ckpt": "runs/bc_v5.pt", "aim": "model", "success": False},     # repeat
        {"level": 1, "instruction": "b", "ckpt": "runs/bc_v5.pt", "aim": "model", "success": True},
        {"level": 1, "instruction": "a", "ckpt": "bc_v5.pt", "aim": "model", "success": True},           # later fix
    ])
    assert summarise(rows) == {"bc_v5.pt": [2, 2]}


def test_levels_and_aim_filters_and_unjudged_rows(tmp_path):
    rows = _rows(tmp_path, [
        {"level": 1, "instruction": "a", "ckpt": "x.pt", "aim": "model", "success": True},
        {"level": 2, "instruction": "b", "ckpt": "x.pt", "aim": "model", "success": False},
        {"level": 1, "instruction": "c", "ckpt": "x.pt", "aim": "centre", "success": True},
        {"level": 1, "instruction": "d", "ckpt": "x.pt", "aim": "model", "success": None},
        {"instruction": "no level (an old single run)", "ckpt": "x.pt", "success": True},
    ])
    assert summarise(rows, {1}, "model") == {"x.pt": [1, 1]}
    assert summarise(rows, {1, 2}, "model") == {"x.pt": [1, 2]}
    assert summarise(rows, {1}, "centre") == {"x.pt": [1, 1]}


def test_table_reports_p_value_against_the_base():
    lines = table({"bc_v5.pt": [18, 26], "bc_d1.pt": [26, 26]}, base="bc_v5.pt")
    d1 = next(l for l in lines if l.startswith("bc_d1.pt"))
    assert "26/26" in d1 and "100%" in d1 and "p = 0.004" in d1
    assert "p =" not in next(l for l in lines if l.startswith("bc_v5.pt"))
