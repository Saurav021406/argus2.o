"""Turn cached features into (history, current frame, instruction) -> action samples."""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from argus.actions.action_space import ACTION_TYPES, Action, ActionType, GRID
from argus.core.planner import GOALS, PlanError, align_goals, parse_instruction

N_TYPES = len(ACTION_TYPES)
TYPE_IDX = {t: i for i, t in enumerate(ACTION_TYPES)}
KEY_T, TEXT_T = TYPE_IDX["KEY"], TYPE_IDX["TYPE"]


@dataclass
class Vocab:
    keys: List[str]
    texts: List[str]

    def key_id(self, k: Optional[str]) -> int:
        return self.keys.index(k) if k in self.keys else -1

    def text_id(self, t: Optional[str]) -> int:
        return self.texts.index(t) if t in self.texts else -1

    def to_dict(self):
        return {"keys": self.keys, "texts": self.texts}

    @staticmethod
    def from_dict(d):
        return Vocab(list(d["keys"]), list(d["texts"]))


@dataclass
class EpisodeData:
    id: str
    task: str
    app: str
    instruction: str
    patch: torch.Tensor          # [n, gh, gw, C] fp16
    cls: torch.Tensor            # [n, C] fp16
    row_of: Dict[int, int]       # frame index -> row in patch/cls
    frames: List[int]            # frame index of each step
    actions: List[Action]
    goals: List[int]             # planner sub-goal id for every step
    source: str = "human"        # human | dagger
    exec_actions: Optional[List[Action]] = None   # DAgger: what the learner really did (history input)


def load_episodes(feat_dirs, verbose: bool = True) -> List[EpisodeData]:
    """Load cached episodes from one directory or a list of them (e.g. human demos + DAgger rounds)."""
    dirs = [feat_dirs] if isinstance(feat_dirs, str) else list(feat_dirs)
    out, skipped = [], []
    for feat_dir in dirs:
        for p in sorted(glob.glob(os.path.join(feat_dir, "*.pt"))):
            if os.path.basename(p) == "instructions.pt":
                continue
            d = torch.load(p)
            actions = [Action.from_dict(s["action"]) for s in d["steps"]]
            try:
                plan = parse_instruction(d["instruction"])
                align_goals(plan, actions)
            except PlanError as e:
                skipped.append((d["id"], d["instruction"], str(e)))
                continue
            ex = [Action.from_dict(s["action"]) for s in d.get("exec_steps", [])] or None
            out.append(EpisodeData(
                id=d["id"], task=d["task"], app=d["app"], instruction=d["instruction"],
                patch=d["patch"], cls=d["cls"], row_of={r: i for i, r in enumerate(d["rows"])},
                frames=[s["frame"] for s in d["steps"]], actions=actions, goals=plan.ids,
                source=d.get("source", "human"), exec_actions=ex,
            ))
    if verbose:
        for eid, ins, why in skipped:
            print(f"  skipped {eid} [{ins}]: {why}")
    return out


def build_vocab(eps: List[EpisodeData]) -> Vocab:
    keys = sorted({a.key for e in eps for a in e.actions if a.type == ActionType.KEY and a.key})
    texts = sorted({a.text for e in eps for a in e.actions if a.type == ActionType.TYPE and a.text})
    return Vocab(keys, texts)


def split_episodes(eps: List[EpisodeData], val_frac: float = 0.25, seed: int = 0,
                   holdout_task: Optional[str] = None,
                   val_ids: Optional[set] = None) -> Tuple[List[EpisodeData], List[EpisodeData]]:
    if val_ids:
        return [e for e in eps if e.id not in val_ids], [e for e in eps if e.id in val_ids]
    if holdout_task:
        return [e for e in eps if e.task != holdout_task], [e for e in eps if e.task == holdout_task]
    dagger = [e for e in eps if e.source == "dagger"]            # DAgger rollouts only ever train, never validate
    eps = [e for e in eps if e.source != "dagger"]
    rng = random.Random(seed)
    tr, va = [], []
    by_task: Dict[str, List[EpisodeData]] = {}
    for e in eps:
        by_task.setdefault(e.task, []).append(e)
    for task, items in sorted(by_task.items()):
        items = sorted(items, key=lambda e: e.id)
        rng.shuffle(items)
        n_val = int(round(len(items) * val_frac)) if len(items) >= 3 else 0
        va += items[:n_val]
        tr += items[n_val:]
    return tr + dagger, va


def encode_action(a: Action, vocab: Vocab) -> dict:
    return {
        "type": TYPE_IDX[a.type.value],
        "key": vocab.key_id(a.key) if a.type == ActionType.KEY else -1,
        "text": vocab.text_id(a.text) if a.type == ActionType.TYPE else -1,
        "cell": a.cell if a.cell is not None else -1,
        "end": a.end_cell if a.end_cell is not None else -1,
    }


def make_inputs(cls_rows: List[torch.Tensor], prev_encs: List[Optional[dict]], patch: torch.Tensor,
                goal: int, T: int) -> dict:
    """Build one model input. Used by BOTH training (StepDataset) and the live agent.

    cls_rows : frame CLS vectors for the last <= T steps, oldest first, current step last
    prev_encs: for each of those steps, the encoded action taken just BEFORE it (None for step 0)
    """
    n = len(cls_rows)
    pad = T - n
    C = cls_rows[0].shape[0]
    cls = torch.zeros(T, C)
    mask = torch.zeros(T, dtype=torch.bool)
    ptype, pkey, ptext, pcell, pend = (torch.zeros(T, dtype=torch.long) for _ in range(5))
    for j in range(n):
        pos = pad + j
        cls[pos] = cls_rows[j]
        mask[pos] = True
        pe = prev_encs[j]
        if pe is not None:
            ptype[pos] = pe["type"] + 1
            pkey[pos] = pe["key"] + 1
            ptext[pos] = pe["text"] + 1
            pcell[pos] = pe["cell"] + 1
            pend[pos] = pe["end"] + 1
    return {"cls": cls, "mask": mask, "ptype": ptype, "pkey": pkey, "ptext": ptext, "pcell": pcell,
            "pend": pend, "patch": patch, "goal": torch.tensor(goal)}


class StepDataset(Dataset):
    """One sample per demonstrated step. Token for step s = frame_s + action_{s-1}."""

    def __init__(self, eps: List[EpisodeData], vocab: Vocab, T: int = 6):
        self.eps, self.vocab, self.T = eps, vocab, T
        self.tasks = sorted({e.task for e in eps})
        self.index = [(i, k) for i, e in enumerate(eps) for k in range(len(e.actions))]
        self.enc = [[encode_action(a, vocab) for a in e.actions] for e in eps]               # targets
        self.henc = [[encode_action(a, vocab) for a in (e.exec_actions or e.actions)] for e in eps]   # history

    def __len__(self):
        return len(self.index)

    def targets(self) -> Dict[str, torch.Tensor]:
        """All targets as tensors (no feature loading), aligned with dataset order."""
        cols = {k: [] for k in ("type", "key", "text", "cell", "end", "task", "step")}
        for i, k in self.index:
            e = self.enc[i][k]
            for name in ("type", "key", "text", "cell", "end"):
                cols[name].append(e[name])
            cols["task"].append(self.tasks.index(self.eps[i].task))
            cols["step"].append(k)
        return {k: torch.tensor(v) for k, v in cols.items()}

    def __getitem__(self, idx):
        i, k = self.index[idx]
        ep, T = self.eps[i], self.T
        lo = max(0, k - T + 1)
        rows = [ep.cls[ep.row_of[ep.frames[s]]].float() for s in range(lo, k + 1)]
        prev = [self.henc[i][s - 1] if s > 0 else None for s in range(lo, k + 1)]
        patch = ep.patch[ep.row_of[ep.frames[k]]].float()
        b = make_inputs(rows, prev, patch, ep.goals[k], T)
        t = self.enc[i][k]
        b.update({"t_type": torch.tensor(t["type"]), "t_key": torch.tensor(t["key"]),
                  "t_text": torch.tensor(t["text"]), "t_cell": torch.tensor(t["cell"]),
                  "t_end": torch.tensor(t["end"])})
        return b
