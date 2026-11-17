"""Closed-loop agent: observe -> predict -> act -> wait for the UI -> observe again."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Callable, Optional

import torch
from PIL import Image, ImageChops, ImageDraw, ImageStat

from argus.actions.action_space import ACTION_TYPES, Action, ActionType, GRID, cell_to_xy
from argus.actions.executor import Executor
from argus.core.planner import GOAL_ID, PlanError, parse_instruction
from argus.policy.policy import ArgusPolicy
from argus.training.dataset import Vocab, encode_action, make_inputs

POINT_TYPES = {"CLICK", "DRAG", "MOVE", "SCROLL"}


class PolicyRunner:
    """Holds the rolling history and turns a screenshot into the next Action."""

    def __init__(self, ckpt_path: str, vision, device: Optional[str] = None, lookup_clicks: bool = False,
                 aim: str = "model", centres: Optional[dict] = None):
        ck = torch.load(ckpt_path, map_location="cpu")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ArgusPolicy(**ck["cfg"])
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.vocab, self.T, self.grid = Vocab.from_dict(ck["vocab"]), ck["T"], ck["grid"]
        self.vision = vision
        # how to choose the cell for a CLICK_ goal on a fixed keypad:
        #   model  = the learned heatmap (can drift with the frame/history)
        #   lookup = the goal->cell table alone (same goal -> same cell)
        #   centre = the centre of YOUR demonstrated clicks for that key (same goal -> same cell, no drift)
        if aim not in ("model", "lookup", "centre"):
            raise ValueError("aim must be model, lookup or centre")
        self.aim = "lookup" if lookup_clicks else aim
        self.lookup_clicks = self.aim == "lookup"
        self.centres = centres or {}
        if self.aim == "centre" and not self.centres:
            raise ValueError("aim='centre' needs the centres of your demonstrated clicks (raw demos not found?)")
        self.seen_goals = set(ck.get("seen_goals") or [])
        seen = ck.get("seen_types") or list(range(len(ACTION_TYPES)))
        self.type_mask = torch.full((len(ACTION_TYPES),), float("-inf"))
        self.type_mask[seen] = 0.0          # never predict an action type the data never contained
        self.reset("")

    def reset(self, instruction: str):
        """Plan the instruction. Raises PlanError if it cannot be understood or needs untrained buttons."""
        self.instruction = instruction
        self.plan = parse_instruction(instruction) if instruction else None
        if self.plan and self.seen_goals:
            missing = sorted({g for g in self.plan.goals if g not in self.seen_goals})
            if missing:
                raise PlanError(f"no training demonstrations for {missing}. Record episodes that use them "
                                f"(python -m argus.data session --plan press) and retrain.")
        self.cls, self.encs = [], []

    def _aim(self, goal_name: str, cell: int) -> int:
        """aim='centre': click where the demos clicked this key (rounded to a cell); otherwise keep the model's cell."""
        if self.aim == "centre" and goal_name.startswith("CLICK_") and goal_name[6:] in self.centres:
            r, c = self.centres[goal_name[6:]]
            g = self.grid
            return min(max(int(round(r)), 0), g - 1) * g + min(max(int(round(c)), 0), g - 1)
        return cell

    @torch.no_grad()
    def act(self, image: Image.Image):
        patch, cls = self.vision.encode([image])
        self.cls.append(cls[0].float())
        k = len(self.cls) - 1
        lo = max(0, k - self.T + 1)
        prev = [self.encs[s - 1] if s > 0 else None for s in range(lo, k + 1)]
        goal = GOAL_ID[self.plan.goals[k]] if k < len(self.plan.goals) else GOAL_ID["DONE"]
        inp = make_inputs(self.cls[lo:k + 1], prev, patch[0].float(), goal, self.T)
        out = self.model({n: v.unsqueeze(0).to(self.device) for n, v in inp.items()})

        tprob = torch.softmax(out["type"][0].cpu() + self.type_mask, -1)
        ti = int(tprob.argmax())
        atype = ActionType(ACTION_TYPES[ti])
        goal_name = self.plan.goals[min(k, len(self.plan.goals) - 1)]
        use_table = self.lookup_clicks and goal_name.startswith("CLICK_")
        heat = torch.softmax(out["loc_goal" if use_table else "loc"][0].cpu(), -1)   # [2, grid*grid]
        a = Action(atype)
        if atype.value in POINT_TYPES:
            a.cell = int(heat[0].argmax())
        if atype == ActionType.CLICK:
            a.cell = self._aim(goal_name, a.cell)
        if atype == ActionType.DRAG:
            a.end_cell = int(heat[1].argmax())
        if atype == ActionType.KEY and self.vocab.keys:
            a.key = self.vocab.keys[int(out["key"][0].argmax())]
        if atype == ActionType.TYPE and self.vocab.texts:
            a.text = self.vocab.texts[int(out["text"][0].argmax())]
        self.encs.append(encode_action(a, self.vocab))
        info = {"type_conf": round(float(tprob[ti]), 3),
                "cell_conf": round(float(heat[0].max()), 3) if a.cell is not None and self.aim != "centre" else None,
                "type_logits": out["type"][0].cpu()}
        return a, info


def _draw(img: Image.Image, a: Action, grid: int) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    if a.cell is not None:
        x, y = cell_to_xy(a.cell, w, h, grid)
        d.ellipse([x - 12, y - 12, x + 12, y + 12], outline=(255, 0, 0), width=3)
        if a.end_cell is not None:
            x1, y1 = cell_to_xy(a.end_cell, w, h, grid)
            d.line([x, y, x1, y1], fill=(255, 0, 0), width=3)
    d.rectangle([0, 0, w, 20], fill=(0, 0, 0))
    d.text((6, 4), json.dumps(a.to_dict()), fill=(255, 255, 255))
    return img


def _same(a: Action, b: Action) -> bool:
    return (a.type, a.cell, a.end_cell, a.key, a.text) == (b.type, b.cell, b.end_cell, b.key, b.text)


class Agent:
    def __init__(self, runner: PolicyRunner, capture, executor: Executor, out_dir: str = "runs",
                 max_steps: Optional[int] = None, stuck_limit: int = 3, settle_min: float = 0.6,
                 settle_max: float = 4.0, settle_poll: float = 0.3, settle_diff: float = 1.5,
                 sleep: Callable[[float], None] = time.sleep):
        self.runner, self.capture, self.executor = runner, capture, executor
        self.out_dir, self.max_steps, self.stuck_limit = out_dir, max_steps, stuck_limit
        self.settle_min, self.settle_max, self.settle_poll, self.settle_diff = settle_min, settle_max, settle_poll, settle_diff
        self.sleep = sleep

    def _wait_settled(self, before: Image.Image) -> Image.Image:
        """Wait until the screen stops changing (animations, windows opening), then return the new frame."""
        self.sleep(self.settle_min)
        prev, waited = self.capture.grab(), self.settle_min
        while waited < self.settle_max:
            self.sleep(self.settle_poll)
            waited += self.settle_poll
            cur = self.capture.grab()
            diff = ImageChops.difference(prev.convert("L").resize((64, 40)), cur.convert("L").resize((64, 40)))
            if ImageStat.Stat(diff).mean[0] < self.settle_diff:
                return cur
            prev = cur
        return prev

    def run(self, instruction: str, countdown: int = 3) -> dict:
        slug = re.sub(r"[^a-z0-9]+", "_", instruction.lower()).strip("_")[:40]
        run_dir = os.path.join(self.out_dir, datetime.now().strftime("%Y%m%d_%H%M%S_") + slug)
        os.makedirs(run_dir, exist_ok=True)
        self.runner.reset(instruction)
        max_steps = self.max_steps or len(self.runner.plan.goals) + 3
        for n in range(countdown, 0, -1):
            print(f"Agent starts in {n}... (mouse to a screen corner aborts)", flush=True)
            self.sleep(1)

        history, stopped, img = [], "max_steps", self.capture.grab()
        with open(os.path.join(run_dir, "trace.jsonl"), "w", encoding="utf-8") as trace:
            for step in range(max_steps):
                action, info = self.runner.act(img)
                img_path = os.path.join(run_dir, f"step_{step:02d}.png")
                _draw(img, action, self.runner.grid).save(img_path)
                res = self.executor.execute(action)
                trace.write(json.dumps({"step": step, "action": action.to_dict(), "ok": res.ok,
                                        "reason": res.reason, "type_conf": info["type_conf"],
                                        "cell_conf": info["cell_conf"]}) + "\n")
                trace.flush()
                print(f"step {step:02d}  {json.dumps(action.to_dict())}  conf={info['type_conf']}  [{res.reason}]")
                history.append(action)
                if not res.ok:
                    stopped = f"blocked: {res.reason}"
                    break
                if action.type == ActionType.DONE:
                    stopped = "done"
                    break
                if len(history) >= self.stuck_limit and all(_same(history[-1], h) for h in history[-self.stuck_limit:]):
                    stopped = "stuck"
                    break
                img = img if self.executor.dry_run else self._wait_settled(img)
        return {"instruction": instruction, "steps": len(history), "stopped": stopped, "trace": run_dir}
