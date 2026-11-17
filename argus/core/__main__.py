"""CLI:
  python -m argus.core run --ckpt checkpoints/bc_v1.pt --instruction "Open Notepad" --task open_app
  python -m argus.core run ... --dry-run          # predict on the live screen but do NOT move the mouse
  python -m argus.core summary                    # success rate per task from runs/results.jsonl
"""
import argparse
import collections
import json
import os
from datetime import datetime


def main():
    p = argparse.ArgumentParser(prog="argus.core")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--instruction", required=True)
    r.add_argument("--task", default="")
    r.add_argument("--max-steps", type=int, default=None, help="default: plan length + 3")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--no-ask", action="store_true", help="do not ask whether it succeeded")
    r.add_argument("--lookup-clicks", action="store_true",
                   help="click calculator buttons using only the goal->cell table (ignore pixels)")
    r.add_argument("--aim", choices=["model", "lookup", "centre"], default="model",
                   help="centre = click where YOUR demos clicked each key (needs --raw)")
    r.add_argument("--raw", default="datasets/raw")
    r.add_argument("--out", default="runs")
    u = sub.add_parser("suite", help="run the automatic calculator test suite")
    u.add_argument("--ckpt", required=True)
    u.add_argument("--levels", default="1,2", help="comma list from 1,2,3,4,5,7 (7 = Paint)")
    u.add_argument("--heldout", type=int, default=0, help="instead: N fresh questions no DAgger round has seen")
    u.add_argument("--heldout-seed", type=int, default=9001)
    u.add_argument("--repeat", type=int, default=1)
    u.add_argument("--only", default="", help="only questions containing this text")
    u.add_argument("--manual", action="store_true", help="ask y/n instead of reading the Calculator display")
    u.add_argument("--lookup-clicks", action="store_true")
    u.add_argument("--aim", choices=["model", "lookup", "centre"], default="model",
                   help="centre = click where YOUR demos clicked each key")
    u.add_argument("--out", default="runs")
    u.add_argument("--raw", default="datasets/raw", help="your demonstrations, used to explain failures")
    g = sub.add_parser("dagger", help="DAgger round: the learner drives, the expert labels, the frames are saved")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--raw", default="datasets/raw", help="your human demos (the expert clicks where they did)")
    g.add_argument("--out", default=None, help="default datasets/dagger/iter<N>")
    g.add_argument("--iteration", type=int, default=1)
    g.add_argument("--n", type=int, default=24, help="number of questions to roll out")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--beta", type=float, default=0.0, help="share of steps where the EXPERT action is executed")
    g.add_argument("--task", choices=["calc", "paint"], default="calc", help="paint: strokes labelled by the demos' mean")
    x = sub.add_parser("explain", help="which key did each click of a run land on?")
    x.add_argument("run", help="a runs\\<folder>")
    x.add_argument("--instruction", required=True)
    x.add_argument("--raw", default="datasets/raw")
    k = sub.add_parser("compare", help="success rates with confidence intervals, checkpoint vs checkpoint")
    k.add_argument("--file", default="runs/results.jsonl")
    k.add_argument("--levels", default="1,2,5,6")
    k.add_argument("--aim", default="model")
    k.add_argument("--base", default=None, help="checkpoint file name to test the others against, e.g. bc_v5.pt")
    ik = sub.add_parser("inkcheck", help="show what the Paint ink check sees on your screen right now")
    ik.add_argument("--delay", type=int, default=10, help="seconds to wait so you can bring Paint to the front")
    ik.add_argument("--white", type=int, default=235, help="how bright a pixel must be to count as canvas (try 200)")
    ik.add_argument("--out", default="runs/inkcheck.png", help="the screenshot it analysed is saved here")
    s = sub.add_parser("summary")
    s.add_argument("--file", default="runs/results.jsonl")
    a = p.parse_args()

    if a.cmd == "summary":
        rows = [json.loads(l) for l in open(a.file, encoding="utf-8")] if os.path.exists(a.file) else []
        rows = [x for x in rows if x.get("success") is not None]
        by = collections.defaultdict(lambda: [0, 0])
        for x in rows:
            k = (x.get("ckpt", "?"), (x.get("task") or "?") + (f" L{x['level']}" if x.get("level") else ""))
            by[k][0] += bool(x["success"])
            by[k][1] += 1
        for (ck, t), (ok, n) in sorted(by.items()):
            print(f"{os.path.basename(ck):14s} {t:12s} success {ok}/{n} = {ok / n:.0%}")
        if not rows:
            print("no judged runs yet")
        return

    if a.cmd == "dagger":
        from argus.actions.executor import Executor
        from argus.core.agent import Agent, PolicyRunner
        from argus.core.dagger import Expert, dagger_prompts, paint_prompts, run_dagger
        from argus.core.explain import key_centres
        from argus.core.suite import close_apps
        from argus.data.strokes import stroke_centres
        from argus.perception.capture import ScreenCapture
        from argus.perception.vision_encoder import VisionEncoder

        centres = key_centres(a.raw)
        out = a.out or f"datasets/dagger/iter{a.iteration}"
        seed = a.seed if a.seed is not None else 100 + a.iteration
        prompts = paint_prompts(a.n) if a.task == "paint" else dagger_prompts(a.n, seed)
        print(f"{len(prompts)} questions ({'Paint' if a.task == 'paint' else 'none of them is a test-suite question'}). Loading models...")
        runner = PolicyRunner(a.ckpt, VisionEncoder(), aim="model")       # the LEARNER drives, in its own way
        cap = ScreenCapture()
        agent = Agent(runner, cap, Executor(screen_size=cap.screen_size), out_dir="runs")
        input("Hands off the mouse and keyboard while it runs (mouse to a screen corner aborts). Enter to start > ")
        run_dagger(agent, Expert(centres, strokes=stroke_centres(a.raw)), prompts, out, a.iteration, a.beta, seed,
                   reset=close_apps)
        feats = f"datasets/features_d{a.iteration}"
        print("\nNext:")
        print(f"  python -m argus.training cache --root {out} --out {feats}")
        print(f"  python -m argus.training train --features datasets/features {feats} --out checkpoints/bc_d{a.iteration}.pt")
        return

    if a.cmd == "compare":
        from argus.core.compare import load_rows, summarise, table

        summ = summarise(load_rows(a.file), {int(x) for x in a.levels.split(",")}, a.aim)
        print(f"levels {a.levels}, aim={a.aim}, distinct questions per checkpoint")
        print("\n".join(table(summ, a.base)) if summ else "no matching runs yet")
        return

    if a.cmd == "inkcheck":
        from argus.core.paintcheck import ink_stats, judge_paint
        from argus.perception.capture import ScreenCapture

        import time

        cap = ScreenCapture()
        for n in range(a.delay, 0, -1):
            print(f"looking at the screen in {n}s... bring the Paint window (with your drawing) to the front", flush=True)
            time.sleep(1)
        frame = cap.grab()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        frame.save(a.out)
        print(f"saved what it saw to {a.out}  (open it: is Paint visible, on the main monitor?)")
        st = ink_stats(frame, white=a.white)
        print(st)
        print(f"  biggest white region = {st['bright_pct']}% of the screen (a Paint canvas is normally 15-40%)")
        for kind in ("line", "zigzag"):
            print(f"  as a {kind}: {judge_paint(st, kind)}")
        return

    if a.cmd == "explain":
        from argus.core.explain import explain_run

        for line in explain_run(a.run, a.instruction, a.raw):
            print(line)
        return

    if a.cmd == "suite":
        from argus.actions.executor import Executor
        from argus.core.agent import Agent, PolicyRunner
        from argus.core.suite import SUITE, close_apps, read_calculator_result, run_suite
        from argus.perception.capture import ScreenCapture
        from argus.perception.vision_encoder import VisionEncoder

        lv = {int(x) for x in a.levels.split(",")}
        items = [(l, t) for l, t in SUITE if l in lv and a.only.lower() in t.lower()]
        if a.heldout:
            from argus.core.dagger import heldout_prompts

            items = [(6, t) for t in heldout_prompts(a.heldout, a.heldout_seed)]
        print(f"{len(items)} questions x {a.repeat}. Loading models...")
        centres = None
        if a.aim == "centre":
            from argus.core.explain import key_centres

            centres = key_centres(a.raw)
            print(f"aiming at the centre of your demonstrated clicks for {len(centres)} keys")
        runner = PolicyRunner(a.ckpt, VisionEncoder(), lookup_clicks=a.lookup_clicks, aim=a.aim, centres=centres)
        cap = ScreenCapture()
        agent = Agent(runner, cap, Executor(screen_size=cap.screen_size), out_dir=a.out)
        input("Hands off the mouse and keyboard while it runs (mouse to a screen corner aborts). Enter to start > ")
        run_suite(agent, items, read_calculator_result, close_apps, out_dir=a.out, ckpt=a.ckpt,
                  repeat=a.repeat, manual=a.manual, aim=a.aim,
                  explain=lambda trace, text: __import__("argus.core.explain", fromlist=["x"]).explain_run(trace, text, a.raw))
        return

    from argus.actions.executor import Executor
    from argus.core.agent import Agent, PolicyRunner
    from argus.perception.capture import ScreenCapture
    from argus.perception.vision_encoder import VisionEncoder

    print("loading models...")
    centres = None
    if a.aim == "centre":
        from argus.core.explain import key_centres

        centres = key_centres(a.raw)
    runner = PolicyRunner(a.ckpt, VisionEncoder(), lookup_clicks=a.lookup_clicks, aim=a.aim, centres=centres)
    cap = ScreenCapture()
    ex = Executor(dry_run=a.dry_run, screen_size=cap.screen_size)
    try:
        runner.reset(a.instruction)
    except Exception as e:                      # PlanError: cannot parse / untrained button
        raise SystemExit(f"\nCannot run this instruction: {e}")
    print(f"plan: {runner.plan.goals}")
    result = Agent(runner, cap, ex, out_dir=a.out, max_steps=a.max_steps).run(a.instruction)
    print(f"\nstopped: {result['stopped']}  after {result['steps']} steps   trace: {result['trace']}")
    if a.dry_run or a.no_ask:
        return
    ans = input("Did it complete the task? [y/n/skip] > ").strip().lower()
    if ans in ("y", "n"):
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({**result, "task": a.task, "ckpt": a.ckpt, "success": ans == "y",
                                "time": datetime.now().isoformat(timespec="seconds")}) + "\n")


if __name__ == "__main__":
    main()
