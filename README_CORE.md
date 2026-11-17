# Argus core v0 — executor + demo recorder

Drop these folders into the repo root (D:\argus2.o). `argus/__init__.py` REPLACES the old one
(the old one imported argus_server, i.e. MySQL + 442 tools, on every `import argus`).

    pip install -r requirements-core.txt
    python -m pytest -q tests                                   # 12 tests, should pass
    python -m argus.data record --instruction "Open Notepad" --task open_app --app notepad
    #   3s countdown -> do the task -> press F10
    python -m argus.data validate datasets/raw/<episode>
    python -m argus.data overlay  datasets/raw/<episode>         # open overlay/*.png, check red circles
    python -m argus.data replay   datasets/raw/<episode> --dry-run
    python -m argus.data replay   datasets/raw/<episode>         # live; mouse to corner aborts
    python -m argus.data stats

Known limits: TYPE is ASCII only (pyautogui.write); MOVE is never produced by the recorder
(clicks carry the location); multi-monitor uses monitor 1 only.
