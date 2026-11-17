import json
import os

import pytest

torch = pytest.importorskip("torch")
from argus.core.explain import explain_run  # noqa: E402
from argus.data.session import _press_plan  # noqa: E402
from tests.test_generalization import CELL, _write  # noqa: E402


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("raw"))
    for j, (_, _, t, _, _) in enumerate(_press_plan(6)):
        _write(d, f"p{j}", t)
    return d


def _trace(tmp_path, cells):
    d = tmp_path / "run"
    d.mkdir()
    with open(d / "trace.jsonl", "w") as f:
        for i, c in enumerate(cells):
            a = {"type": "KEY", "key": "win"} if c is None else {"type": "CLICK", "cell": c}
            f.write(json.dumps({"step": i, "action": a, "ok": True}) + "\n")
    return str(d)


def test_correct_clicks_are_reported_ok(raw, tmp_path):
    cells = [None, None, None] + [CELL[b] for b in "8*6="] + [None]
    lines = explain_run(_trace(tmp_path, cells), "Calculate 8*6", raw)
    assert len(lines) == 4 and all("ok" in l and "WRONG" not in l for l in lines)


def test_a_click_on_the_neighbouring_key_is_named(raw, tmp_path):
    cells = [None, None, None, CELL["8"], CELL["*"], CELL["7"], CELL["="], None]   # '6' became '7'
    lines = explain_run(_trace(tmp_path, cells), "Calculate 8*6", raw)
    bad = [l for l in lines if "WRONG KEY" in l]
    assert len(bad) == 1 and "CLICK_6" in bad[0] and "nearest to key '7'" in bad[0]


def test_wrong_action_type_is_reported(raw, tmp_path):
    cells = [None, None, None, None, CELL["*"], CELL["6"], CELL["="], None]        # step 3 is not a click
    lines = explain_run(_trace(tmp_path, cells), "Calculate 8*6", raw)
    assert any("WRONG ACTION" in l and "CLICK_8" in l for l in lines)


def test_a_click_near_the_key_edge_is_flagged_not_called_ok(raw, tmp_path):
    # 2 rows below the '8' centre = 0.63 key-sizes: still nearest to '8', but probably outside the key
    cells = [None, None, None, CELL["8"] + 64, CELL["*"], CELL["6"], CELL["="], None]
    lines = explain_run(_trace(tmp_path, cells), "Calculate 8*6", raw)
    edge = [l for l in lines if "EDGE" in l]
    assert len(edge) == 1 and "CLICK_8" in edge[0] and "WRONG" not in edge[0]
