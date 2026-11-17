import pytest

torch = pytest.importorskip("torch")
from argus.actions.action_space import Action, ActionType as T  # noqa: E402
from argus.core.planner import GOAL_ID, align_goals, parse_instruction  # noqa: E402
from argus.policy.policy import ArgusPolicy  # noqa: E402
from argus.training.behavioral_cloning import init_goal_table  # noqa: E402
from argus.training.dataset import EpisodeData  # noqa: E402

G = 32
# three strokes of a zigzag: down-right, up-right, down-right (row, col)
STROKES = [((10, 5), (16, 11)), ((16, 11), (10, 17)), ((10, 17), (16, 23))]


def _cell(rc):
    return rc[0] * G + rc[1]


def _episode(i, jitter):
    text = "Open Paint and draw a zigzag"
    plan = parse_instruction(text)
    acts = [Action(T.KEY, key="win"), Action(T.TYPE, text="paint"), Action(T.KEY, key="enter")]
    for (a, b) in STROKES:
        acts.append(Action(T.DRAG, cell=_cell((a[0] + jitter[0], a[1] + jitter[1])),
                           end_cell=_cell((b[0] + jitter[0], b[1] + jitter[1]))))
    acts.append(Action(T.DONE))
    align_goals(plan, acts)
    return EpisodeData(id=f"z{i}", task="paint", app="paint", instruction=text, patch=torch.zeros(1, 2, 2, 4),
                       cls=torch.zeros(1, 4), row_of={0: 0}, frames=[0] * len(acts), actions=acts, goals=plan.ids)


def test_each_stroke_of_a_zigzag_has_its_own_goal():
    plan = parse_instruction("Open Paint and draw a zigzag")
    assert plan.goals[3:6] == ["DRAW_zigzag_1", "DRAW_zigzag_2", "DRAW_zigzag_3"]
    assert len({GOAL_ID[g] for g in plan.goals[3:6]}) == 3


def test_strokes_start_and_end_in_different_places_not_all_at_one_cell():
    eps = [_episode(i, j) for i, j in enumerate([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)])]
    model = ArgusPolicy(n_keys=2, n_texts=1, vis_dim=4, gh=2, gw=2, d=16, layers=1, heads=2)
    assert init_goal_table(model, eps, sigma=2.0) >= 3
    w = model.goal_table.weight.detach()
    for k, (a, b) in enumerate(STROKES, start=1):
        row = w[GOAL_ID[f"DRAW_zigzag_{k}"]]
        start, end = int(row[:G * G].argmax()), int(row[G * G:].argmax())
        assert abs(start // G - a[0]) <= 1 and abs(start % G - a[1]) <= 1, f"stroke {k} start"
        assert abs(end // G - b[0]) <= 1 and abs(end % G - b[1]) <= 1, f"stroke {k} end"
    starts = {int(w[GOAL_ID[f"DRAW_zigzag_{k}"]][:G * G].argmax()) for k in (1, 2, 3)}
    assert len(starts) == 3                                         # the bug: they used to be one and the same cell
