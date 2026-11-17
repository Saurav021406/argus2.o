"""CLI:  python -m argus.data <record|validate|overlay|replay|stats> ..."""
import argparse
import glob
import os

from argus.data.episode import Episode
from argus.data.validator import validate_episode


def main():
    p = argparse.ArgumentParser(prog="argus.data")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record")
    r.add_argument("--instruction", required=True)
    r.add_argument("--task", default="")
    r.add_argument("--app", default="")
    r.add_argument("--out", default="datasets/raw")
    r.add_argument("--fps", type=float, default=5.0)
    r.add_argument("--countdown", type=int, default=3)

    for name in ("validate", "overlay", "replay"):
        s = sub.add_parser(name)
        s.add_argument("episode")
        if name == "overlay":
            s.add_argument("--out", default=None)
        if name == "replay":
            s.add_argument("--dry-run", action="store_true")

    se = sub.add_parser("session")
    se.add_argument("--out", default="datasets/raw")
    se.add_argument("--start", type=int, default=0)
    se.add_argument("--plan", choices=["first", "calc", "press", "extras", "press2", "extras2"], default="first",
                    help="first = 29 mixed, calc = 30 short calculator episodes, press = 6 long episodes covering 0-9 + - * / =, extras = 6 long episodes for . % +/- sqrt x2 1/x C CE backspace")

    au = sub.add_parser("audit")
    au.add_argument("root", nargs="?", default="datasets/raw")

    sk = sub.add_parser("strokes", help="where did your Paint demos start and end each stroke?")
    sk.add_argument("root", nargs="?", default="datasets/raw")

    cv = sub.add_parser("coverage")
    cv.add_argument("root", nargs="?", default="datasets/raw")

    st = sub.add_parser("stats")
    st.add_argument("root", nargs="?", default="datasets/raw")

    a = p.parse_args()

    if a.cmd == "record":
        from argus.data.recorder import Recorder

        os.makedirs(a.out, exist_ok=True)
        ep = Recorder(a.out, a.instruction, a.task, a.app, fps=a.fps).run(a.countdown)
        issues = validate_episode(ep)
        print("VALID" if not issues else "ISSUES:\n  " + "\n  ".join(issues))
    elif a.cmd == "validate":
        issues = validate_episode(Episode.load(a.episode))
        print("VALID" if not issues else "ISSUES:\n  " + "\n  ".join(issues))
    elif a.cmd == "overlay":
        from argus.data.replay import render_overlays

        ep = Episode.load(a.episode)
        out = a.out or os.path.join(a.episode, "overlay")
        print(f"wrote {render_overlays(ep, out)} images to {out}")
    elif a.cmd == "replay":
        from argus.actions.executor import Executor
        from argus.data.replay import replay

        ep = Episode.load(a.episode)
        fails = replay(Executor(dry_run=a.dry_run), ep)
        print(f"done, {fails} failed actions")
    elif a.cmd == "session":
        from argus.data.session import run_session

        os.makedirs(a.out, exist_ok=True)
        from argus.data.session import (PLAN_CALC, PLAN_EXTRAS, PLAN_EXTRAS2, PLAN_FIRST_BATCH, PLAN_PRESS,
                                        PLAN_PRESS2)

        plan = {"first": PLAN_FIRST_BATCH, "calc": PLAN_CALC, "press": PLAN_PRESS, "extras": PLAN_EXTRAS,
                "press2": PLAN_PRESS2, "extras2": PLAN_EXTRAS2}[a.plan]
        run_session(a.out, plan, start=a.start)
    elif a.cmd == "audit":
        from argus.data.audit import audit_root

        audit_root(a.root)
    elif a.cmd == "strokes":
        from argus.data.strokes import report as stroke_report

        stroke_report(a.root)
    elif a.cmd == "coverage":
        from argus.data.coverage import report

        report(a.root)
    elif a.cmd == "stats":
        by_task, total = {}, 0
        for d in sorted(glob.glob(os.path.join(a.root, "*", "meta.json"))):
            ep = Episode.load(os.path.dirname(d))
            bad = bool(validate_episode(ep))
            t = ep.meta.get("task") or "?"
            c = by_task.setdefault(t, [0, 0, 0])
            c[0] += 1
            c[1] += len(ep.steps)
            c[2] += bad
            total += 1
        print(f"{total} episodes")
        for t, (n, steps, bad) in sorted(by_task.items()):
            print(f"  {t:15s} episodes={n:3d} steps={steps:4d} invalid={bad}")


if __name__ == "__main__":
    main()
