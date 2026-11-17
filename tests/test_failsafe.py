import sys
import types

from argus.actions.executor import Executor


def test_all_four_corners_abort_not_only_the_top_left(monkeypatch):
    fake = types.SimpleNamespace(FAILSAFE=False, PAUSE=0.0, FAILSAFE_POINTS=[(0, 0)], size=lambda: (1920, 1200))
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    Executor()._backend()
    assert fake.FAILSAFE is True
    assert set(fake.FAILSAFE_POINTS) == {(0, 0), (1919, 0), (0, 1199), (1919, 1199)}


def test_old_pyautogui_without_failsafe_points_still_works(monkeypatch):
    fake = types.SimpleNamespace(FAILSAFE=False, PAUSE=0.0, size=lambda: (1920, 1200))
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    Executor()._backend()
    assert fake.FAILSAFE is True and not hasattr(fake, "FAILSAFE_POINTS")
