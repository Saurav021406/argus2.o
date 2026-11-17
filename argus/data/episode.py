"""On-disk episode format.

episode_dir/
  meta.json      instruction, screen/frame size, fps, grid, source ("human"|"dagger")
  frames.json    [{"i": 0, "t": 0.00, "file": "frames/000000.jpg"}, ...]
  frames/        JPEGs (possibly downscaled; coordinates are stored as grid cells)
  events.jsonl   raw input events (kept so labels can be re-built later)
  actions.jsonl  one action per line: {"step", "t", "frame", "action": {...}}   (the TARGET: what should be done)
  executed.jsonl optional, DAgger only: what the learner actually did at each step (history is built from it)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List

from argus.actions.action_space import Action


@dataclass
class Step:
    step: int
    t: float
    frame: int          # index of the frame the human saw when acting
    action: Action

    def to_dict(self):
        return {"step": self.step, "t": round(self.t, 3), "frame": self.frame,
                "action": self.action.to_dict()}

    @staticmethod
    def from_dict(d):
        return Step(d["step"], d["t"], d["frame"], Action.from_dict(d["action"]))


@dataclass
class Episode:
    path: str
    meta: dict
    frames: List[dict] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    exec_steps: List[Step] = field(default_factory=list)      # DAgger: executed actions; empty for human demos

    def frame_path(self, i: int) -> str:
        return os.path.join(self.path, self.frames[i]["file"])

    def save_labels(self):
        with open(os.path.join(self.path, "actions.jsonl"), "w", encoding="utf-8") as f:
            for s in self.steps:
                f.write(json.dumps(s.to_dict()) + "\n")
        if self.exec_steps:
            with open(os.path.join(self.path, "executed.jsonl"), "w", encoding="utf-8") as f:
                for s in self.exec_steps:
                    f.write(json.dumps(s.to_dict()) + "\n")
        self.meta["n_actions"] = len(self.steps)
        self.meta["n_frames"] = len(self.frames)
        with open(os.path.join(self.path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
        with open(os.path.join(self.path, "frames.json"), "w", encoding="utf-8") as f:
            json.dump(self.frames, f)

    @staticmethod
    def load(path: str) -> "Episode":
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(path, "frames.json"), encoding="utf-8") as f:
            frames = json.load(f)
        steps = []
        p = os.path.join(path, "actions.jsonl")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                steps = [Step.from_dict(json.loads(l)) for l in f if l.strip()]
        exec_steps = []
        q = os.path.join(path, "executed.jsonl")
        if os.path.exists(q):
            with open(q, encoding="utf-8") as f:
                exec_steps = [Step.from_dict(json.loads(l)) for l in f if l.strip()]
        return Episode(path, meta, frames, steps, exec_steps)
