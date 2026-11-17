import pytest

from argus.actions.action_space import Action, ActionType as T
from argus.core.planner import GOALS, N_GOALS, PlanError, align_goals, parse_instruction


def clicks(p):
    return [g[6:] for g in p.goals if g.startswith("CLICK_")]


@pytest.mark.parametrize("text,expr", [
    ("Calculate 8 times 6", "8*6"), ("8*6", "8*6"), ("calculate 8*6", "8*6"), ("8 x 6", "8*6"),
    ("8 X 6", "8*6"), ("8 × 6", "8*6"), ("what is 8 multiplied by 6", "8*6"), ("8 into 6", "8*6"),
    ("eight times six", "8*6"), ("Calculate 7 plus 3", "7+3"), ("9 - 5", "9-5"), ("12+7", "12+7"),
    ("Calculate 6 divided by 2", "6/2"), ("6 ÷ 2", "6/2"), ("2+3*4", "2+3*4"),
])
def test_calculator_phrasings(text, expr):
    p = parse_instruction(text)
    assert p.task == "calculator" and p.expr == expr
    assert clicks(p) == list(expr) + ["="]
    assert p.goals[:3] == ["OPEN_calculator"] * 3 and p.goals[-1] == "DONE"


def test_multi_digit_clicks_each_digit():
    assert clicks(parse_instruction("Calculate 12 plus 7")) == ["1", "2", "+", "7", "="]


@pytest.mark.parametrize("text,app", [("Open Notepad", "notepad"), ("launch calculator", "calculator"),
                                      ("Start Paint", "paint")])
def test_open_app(text, app):
    p = parse_instruction(text)
    assert p.task == "open_app" and p.goals == [f"OPEN_{app}"] * 3 + ["DONE"]


def test_paint():
    assert parse_instruction("Open Paint and draw a line").goals[-2:] == ["DRAW_line", "DONE"]
    z = parse_instruction("Open Paint and draw a zigzag")
    assert z.goals[3:6] == ["DRAW_zigzag_1", "DRAW_zigzag_2", "DRAW_zigzag_3"]      # each stroke has its own goal
    with pytest.raises(PlanError):
        parse_instruction("Open Paint and draw a cat")


@pytest.mark.parametrize("bad", ["hello world", "", "send an email", "Calculate 5 +", "3.5.2 + 1",
                                 "square root of 9 + 1"])
def test_unparseable(bad):
    with pytest.raises(PlanError):
        parse_instruction(bad)


def test_goal_vocab_is_fixed():
    # ids 0..20 never change, so older checkpoints keep working; new buttons are appended after DONE
    assert N_GOALS == len(GOALS) == 33 and GOALS[20] == "DONE" and GOALS[21] == "CLICK_."
    assert GOALS[30:] == ["DRAW_zigzag_1", "DRAW_zigzag_2", "DRAW_zigzag_3"]


def test_alignment():
    p = parse_instruction("Calculate 2 plus 3")
    ok = ([Action(T.KEY, key="win"), Action(T.TYPE, text="calculator"), Action(T.KEY, key="enter")]
          + [Action(T.CLICK, cell=1)] * 4 + [Action(T.DONE)])
    assert align_goals(p, ok) == p.goals
    with pytest.raises(PlanError, match="plan needs"):
        align_goals(p, ok[:-2] + ok[-1:])                    # one click missing
    bad = list(ok)
    bad[4] = Action(T.DRAG, cell=1, end_cell=2)
    with pytest.raises(PlanError, match="CLICK"):
        align_goals(p, bad)


@pytest.mark.parametrize("text,keys", [("Press 4 7 + 9 =", "47+9="), ("press 1,2,x,3,=", "12*3="),
                                       ("Click 0 - / * =", "0-/*=")])
def test_press_mode_lists_exact_buttons(text, keys):
    p = parse_instruction(text)
    assert p.task == "calculator" and clicks(p) == list(keys)
    assert p.goals[:3] == ["OPEN_calculator"] * 3 and p.goals[-1] == "DONE"


@pytest.mark.parametrize("bad", ["Press", "Press the red button", "Press 3 banana 5 ="])
def test_press_mode_rejects_junk(bad):
    with pytest.raises(PlanError):
        parse_instruction(bad)


def test_decimals_and_number_words():
    p = parse_instruction("Calculate 3.5 + 2")
    assert clicks(p) == list("3.5+2=") and p.expected() == 5.5
    assert parse_instruction("what is ten percent of fifty").expected() is None      # % rule unverified


@pytest.mark.parametrize("text,btns,answer", [
    ("square root of 81", ["8", "1", "sqrt"], 9), ("sqrt 16", ["1", "6", "sqrt"], 4),
    ("7 squared", ["7", "sq"], 49), ("square of 7", ["7", "sq"], 49),
    ("reciprocal of 4", ["4", "inv"], 0.25), ("negate 5", ["5", "neg"], -5),
    ("change the sign of 8", ["8", "neg"], -8),
])
def test_one_button_operations(text, btns, answer):
    p = parse_instruction(text)
    assert p.task == "calculator" and clicks(p) == btns and p.expected() == answer
    assert p.goals[-1] == "DONE" and "CLICK_=" not in p.goals           # unary keys act immediately


def test_percent_template_uses_the_percent_key():
    p = parse_instruction("10% of 50")
    assert clicks(p) == ["5", "0", "*", "1", "0", "pct"] and p.expected() is None


def test_press_mode_accepts_named_buttons():
    p = parse_instruction("Press 9 sqrt C 7 sq 1/x neg % back CE")
    assert clicks(p) == ["9", "sqrt", "C", "7", "sq", "inv", "neg", "pct", "back", "CE"]


@pytest.mark.parametrize("btns,answer", [
    ("5 + =", 10), ("2 + 3 * 4 =", 20), ("1 2 3 back =", 12), ("5 + 3 = C", 0), ("9 CE", 0),
    ("5 neg", -5), ("5 neg neg", 5), ("3 . 5 + 2 =", 5.5), ("8 + 2 = * 3 =", 30), ("9 sqrt + 1 =", 4),
    ("6 / 3 =", 2), ("0 . 5 * 4 =", 2),
])
def test_calculator_simulator(btns, answer):
    from argus.core.planner import simulate
    assert simulate(btns.split()) == pytest.approx(answer)


@pytest.mark.parametrize("btns", ["5 / 0 =", "0 inv", "5 neg sqrt", "5 0 * 1 0 pct"])
def test_simulator_says_unknown_instead_of_guessing(btns):
    from argus.core.planner import simulate
    assert simulate(btns.split()) is None
