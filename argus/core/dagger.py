"""DAgger: the learner drives, an expert labels every step it sees, the labelled frames join the training set.

Why it helps. Behavioural cloning only ever sees the screens a HUMAN produced. When the learner clicks a
wrong key, the calculator shows a screen no demo contained, and the click head drifts further (that is the
2-rows-off clicks you saw in model mode). DAgger records the learner's OWN screens and attaches the correct
action, so the next training round covers the states the learner really gets into.

The expert here is not a person. For a fixed-layout calculator the correct action at every plan step is known:
Win / type the app name / Enter, then the centre of where YOUR demos clicked the goal key. So DAgger teaches the
learner to reproduce that expert from live screens. It does not add new knowledge beyond your demonstrations.

    python -m argus.core dagger --ckpt checkpoints/bc_v5.pt --iteration 1 --n 24
"""
from __future__ import annotations

import os
import random
import time
import zlib
from datetime import datetime
from typing import Callable, Dict, List, Optional

from argus.actions.action_space import GRID, Action, ActionType
from argus.core.planner import PlanError, parse_instruction
from argus.core.suite import SUITE
from argus.data.episode import Episode, Step
from argus.training.dataset import encode_action


class Expert:
    """What a demonstrator would do at plan step k (None when there is no expert, e.g. Paint strokes)."""

    def __init__(self, centres: Dict[str, tuple], grid: int = GRID, strokes: Optional[Dict[str, tuple]] = None):
        self.centres, self.grid, self.strokes = centres, grid, strokes or {}

    def action(self, plan, k: int) -> Optional[Action]:
        goal = plan.goals[k] if k < len(plan.goals) else "DONE"
        if goal.startswith("OPEN_"):
            seq = [Action(ActionType.KEY, key="win"), Action(ActionType.TYPE, text=goal[5:]),
                   Action(ActionType.KEY, key="enter")]
            return seq[k] if k < 3 else None
        if goal.startswith("CLICK_"):
            key = goal[6:]
            if key not in self.centres:
                return None
            r, c = self.centres[key]
            g = self.grid
            return Action(ActionType.CLICK, cell=min(max(int(round(r)), 0), g - 1) * g + min(max(int(round(c)), 0), g - 1))
        if goal.startswith("DRAW_") and goal in self.strokes:        # a stroke at the mean of the demos
            start, end = self.strokes[goal]
            return Action(ActionType.DRAG, cell=start, end_cell=end)
        if goal == "DONE":
            return Action(ActionType.DONE)
        return None


def agrees(learner: Action, expert: Action, grid: int = GRID) -> bool:
    """Did the learner do what the expert would? A click counts if it is within 1 row and 2 columns of the
    expert's cell (a calculator key is about 3 rows x 7 columns)."""
    if learner.type != expert.type:
        return False
    if expert.type == ActionType.CLICK:
        if learner.cell is None:
            return False
        return abs(learner.cell // grid - expert.cell // grid) <= 1 and abs(learner.cell % grid - expert.cell % grid) <= 2
    if expert.type == ActionType.DRAG:                                # strokes are loose: 4 rows x 6 columns
        if learner.cell is None or learner.end_cell is None:
            return False
        return all(abs(a // grid - b // grid) <= 4 and abs(a % grid - b % grid) <= 6
                   for a, b in ((learner.cell, expert.cell), (learner.end_cell, expert.end_cell)))
    if expert.type == ActionType.KEY:
        return learner.key == expert.key
    if expert.type == ActionType.TYPE:
        return learner.text == expert.text
    return True


def _suite_signatures() -> set:
    sigs = set()
    for level, text in SUITE:
        if level == 4:
            continue
        try:
            sigs.add(tuple(parse_instruction(text).goals))
        except PlanError:
            pass
    return sigs


def dagger_prompts(n: int = 24, seed: int = 101, exclude: Optional[set] = None) -> List[str]:
    """Calculator questions for the rollouts. Never the same click sequence as a test-suite question,
    so the suite stays an honest held-out test."""
    rng = random.Random(seed)
    seen = _suite_signatures() | set(exclude or ())
    words = [("plus", "+"), ("minus", "-"), ("times", "*"), ("divided by", "/")]

    def candidate() -> str:
        x = rng.random()
        if x < 0.5:
            a, b = rng.randint(0, 9), rng.randint(0, 9)
            word, sym = rng.choice(words)
            if word == "divided by" and b == 0:
                b = 1
            if word == "minus" and a < b:
                a, b = b, a
            return rng.choice([f"Calculate {a} {word} {b}", f"{a} {sym} {b}", f"what is {a} {word} {b}"])
        if x < 0.7:
            d = rng.randint(1, 9)
            return rng.choice([f"square root of {d}", f"{d} squared", f"reciprocal of {d}", f"negate {d}"])
        if x < 0.85:
            return f"Calculate {rng.randint(1, 9)}.{rng.randint(1, 9)} {rng.choice(['+', '-', '*'])} {rng.randint(1, 9)}"
        return f"Calculate {rng.randint(10, 99)} {rng.choice(['+', '-'])} {rng.randint(1, 9)}"

    out: List[str] = []
    for _ in range(20000):
        if len(out) >= n:
            break
        text = candidate()
        try:
            sig = tuple(parse_instruction(text).goals)
        except PlanError:
            continue
        if sig in seen:
            continue
        seen.add(sig)
        out.append(text)
    return out


def paint_prompts(n: int = 12) -> List[str]:
    """Paint questions for DAgger. The learner is deterministic on a blank canvas, so use --beta > 0 to get
    different screens out of repeated prompts."""
    return ["Open Paint and draw a zigzag" if i % 2 == 0 else "Open Paint and draw a line" for i in range(n)]


def heldout_prompts(n: int = 40, seed: int = 9001) -> List[str]:
    """A bigger test set. By construction its click sequences are in NEITHER the fixed suite NOR any DAgger round
    (iterations 1..10 use seeds 101..110), so BC and every DAgger checkpoint are compared on questions none of
    them was trained on."""
    blocked = set()
    for it in range(1, 11):
        blocked |= {tuple(parse_instruction(p).goals) for p in dagger_prompts(60, 100 + it)}
    return dagger_prompts(n, seed, exclude=blocked)


def rollout(agent, expert: Expert, text: str, out_root: str, iteration: int = 1, beta: float = 0.0,
            rng: Optional[random.Random] = None, countdown: int = 2,
            sleep: Callable[[float], None] = time.sleep) -> dict:
    """Run the learner on one question, saving every screen with the expert's action as the label."""
    runner = agent.runner
    runner.reset(text)                                    # PlanError -> caller skips this question
    plan = runner.plan
    rng = rng or random.Random(0)
    eid = f"{datetime.now():%Y%m%d_%H%M%S}_d{iteration}_{zlib.crc32(text.encode()) % 10000:04d}"
    ep_dir = os.path.join(out_root, eid)
    os.makedirs(os.path.join(ep_dir, "frames"), exist_ok=True)
    for n in range(countdown, 0, -1):
        print(f"  rollout starts in {n}...", flush=True)
        sleep(1)

    img = agent.capture.grab()
    t0 = time.time()
    frames, expert_steps, exec_steps, agree = [], [], [], []
    last = len(plan.goals) - 1
    for k in range(len(plan.goals)):
        learner, _ = runner.act(img)
        exp = expert.action(plan, k)
        if exp is None:
            break
        executed = exp if rng.random() < beta else learner
        runner.encs[-1] = encode_action(executed, runner.vocab)      # history = what really happened
        rel = f"frames/{k:06d}.jpg"
        img.save(os.path.join(ep_dir, rel), quality=90)
        t = time.time() - t0
        frames.append({"i": k, "t": round(t, 3), "file": rel})
        expert_steps.append(Step(k, t, k, exp))
        exec_steps.append(Step(k, t, k, executed))
        agree.append(agrees(learner, exp))
        if k == last:
            break                                                    # DONE: nothing to execute
        if executed.type != ActionType.DONE:
            res = agent.executor.execute(executed)
            if not res.ok:
                print(f"    step {k}: {executed.type.value} not executed ({res.reason})")
            img = img if agent.executor.dry_run else agent._wait_settled(img)
        else:
            img = agent.capture.grab()

    meta = {"id": eid, "instruction": text, "task": plan.task, "app": plan.app,
            "screen_w": agent.capture.screen_size[0], "screen_h": agent.capture.screen_size[1],
            "frame_w": img.width, "frame_h": img.height, "grid": GRID, "fps": 5, "source": "dagger",
            "iteration": iteration, "beta": beta, "created": datetime.now().isoformat(timespec="seconds"),
            "agreement": round(sum(agree) / len(agree), 3) if agree else 0.0, "success": None}
    Episode(ep_dir, meta, frames, expert_steps, exec_steps).save_labels()
    return {"dir": ep_dir, "agree": meta["agreement"], "steps": len(frames), "n_agree": sum(agree)}


def run_dagger(agent, expert: Expert, prompts: List[str], out_root: str, iteration: int = 1, beta: float = 0.0,
               seed: int = 0, reset: Callable[[], None] = lambda: None, countdown: int = 2,
               sleep: Callable[[float], None] = time.sleep) -> dict:
    rng = random.Random(seed)
    done, skipped, agree_steps, total_steps = [], [], 0, 0
    for i, text in enumerate(prompts, 1):
        print(f"\n=== [{i}/{len(prompts)}] {text!r}")
        reset()
        try:
            res = rollout(agent, expert, text, out_root, iteration, beta, rng, countdown, sleep)
        except PlanError as e:
            print(f"    skipped: {str(e)[:90]}")
            skipped.append(text)
            continue
        reset()
        sleep(1.0)
        done.append(res)
        agree_steps += res["n_agree"]
        total_steps += res["steps"]
        print(f"    learner agreed with the expert on {res['n_agree']}/{res['steps']} steps -> {res['dir']}")
    rate = agree_steps / total_steps if total_steps else 0.0
    print(f"\n==== DAGGER ITERATION {iteration} ====")
    print(f"  {len(done)} rollouts saved to {out_root}  ({len(skipped)} skipped)")
    print(f"  learner/expert agreement on its own screens: {agree_steps}/{total_steps} = {rate:.0%}")
    return {"rollouts": done, "skipped": skipped, "agreement": rate}
