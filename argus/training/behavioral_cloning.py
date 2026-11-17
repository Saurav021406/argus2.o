"""Behavioral cloning: train the policy on demonstrated steps and report OFFLINE accuracy.

Offline = teacher-forced (the history always contains the human's real actions).
It is a sanity check, not the final metric. Task success in a live loop comes later.
"""
from __future__ import annotations

import collections
import os
import random
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from argus.actions.action_space import GRID
from argus.core.planner import GOALS
from argus.policy.policy import ArgusPolicy, bc_loss, soft_targets
from argus.training.dataset import (KEY_T, TEXT_T, StepDataset, Vocab, build_vocab, load_episodes,
                                    split_episodes)

TOL = 1   # a predicted cell counts as correct if within TOL cells (Chebyshev) of the human's


def _to(b, device):
    return {k: v.to(device) for k, v in b.items()}


@torch.no_grad()
def collect_preds(model, ds, device, bs: int = 64) -> Dict[str, torch.Tensor]:
    model.eval()
    out = {k: [] for k in ("type", "key", "text", "cell", "end")}
    for b in DataLoader(ds, batch_size=bs):
        o = model(_to(b, device))
        out["type"].append(o["type"].argmax(-1).cpu())
        out["key"].append(o["key"].argmax(-1).cpu())
        out["text"].append(o["text"].argmax(-1).cpu())
        out["cell"].append(o["loc"][:, 0].argmax(-1).cpu())
        out["end"].append(o["loc"][:, 1].argmax(-1).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def _cheb(a, b, grid):
    return torch.maximum((a // grid - b // grid).abs(), (a % grid - b % grid).abs())


def score(pred: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor], tasks, grid: int = GRID, tol: int = TOL):
    type_ok = pred["type"] == tgt["type"]
    has_cell, has_end = tgt["cell"] >= 0, tgt["end"] >= 0
    is_key, is_text = tgt["type"] == KEY_T, tgt["type"] == TEXT_T
    d_cell, d_end = _cheb(pred["cell"], tgt["cell"].clamp(min=0), grid), _cheb(pred["end"], tgt["end"].clamp(min=0), grid)
    cell_ok, end_ok = d_cell <= tol, d_end <= tol
    key_ok = (pred["key"] == tgt["key"]) & (tgt["key"] >= 0)
    text_ok = (pred["text"] == tgt["text"]) & (tgt["text"] >= 0)
    step_ok = type_ok & (~has_cell | cell_ok) & (~has_end | end_ok) & (~is_key | key_ok) & (~is_text | text_ok)

    def mean(x, m):
        return float(x[m].float().mean()) if m.any() else float("nan")

    allm = torch.ones_like(type_ok)
    res = {
        "n": int(len(type_ok)),
        "type_acc": mean(type_ok, allm),
        "cell_exact": mean(d_cell == 0, has_cell),
        f"cell_within{tol}": mean(cell_ok, has_cell),
        f"end_within{tol}": mean(end_ok, has_end),
        "key_acc": mean(key_ok, is_key),
        "text_acc": mean(text_ok, is_text),
        "step_acc": mean(step_ok, allm),
        "per_task": {},
    }
    for i, t in enumerate(tasks):
        m = tgt["task"] == i
        res["per_task"][t] = {"n": int(m.sum()), "step_acc": mean(step_ok, m)}
    return res


@torch.no_grad()
def init_goal_table(model, eps, sigma: float, grid: int = GRID, floor: float = 1e-3) -> int:
    """Start the goal->cell lookup AT the demonstrated clicks instead of at zero.

    A rare button (5 demos) gets only a handful of gradient updates, so a table that starts at zero can
    end up a row away from where the human clicked, and the agent presses the neighbouring key. Starting
    from the (smoothed) histogram of the demos puts every row where the demos were; training then refines it.
    Returns the number of goal rows initialised.
    """
    G2 = grid * grid
    n_goals = model.goal_table.weight.shape[0]
    hist = torch.zeros(n_goals, 2, G2)
    cnt = torch.zeros(n_goals, 2)
    for e in eps:
        for k, a in enumerate(e.actions):
            for ch, c in ((0, a.cell), (1, a.end_cell)):
                if c is not None:
                    hist[e.goals[k], ch] += soft_targets(torch.tensor([c]), grid, sigma)[0]
                    cnt[e.goals[k], ch] += 1
    done = 0
    for g in range(n_goals):
        for ch in range(2):
            if cnt[g, ch] > 0:
                p = hist[g, ch] / cnt[g, ch]
                model.goal_table.weight[g, ch * G2:(ch + 1) * G2] = torch.log(p + floor)
                done += ch == 0
    return done


@torch.no_grad()
def lookup_by_goal(model, ds, device, grid: int = GRID, tol: int = TOL) -> Dict[int, list]:
    """goal id -> [hits, n]: how often the goal->cell lookup ALONE lands within tol cells of the human click."""
    model.eval()
    out: Dict[int, list] = {}
    for b in DataLoader(ds, batch_size=64):
        b = _to(b, device)
        idx = torch.nonzero(b["t_cell"] >= 0).flatten()
        if not len(idx):
            continue
        look = model(b)["loc_goal"][idx, 0].argmax(-1).cpu()
        ok = _cheb(look, b["t_cell"][idx].cpu(), grid) <= tol
        for g, hit in zip(b["goal"][idx].tolist(), ok.tolist()):
            r = out.setdefault(g, [0, 0])
            r[0] += int(hit)
            r[1] += 1
    return out


@torch.no_grad()
def diagnose(model, ds, device, grid: int = GRID, tol: int = TOL) -> Dict[str, float]:
    """Does the model really read the goal? (the live '8*6 -> 888' failure was a model that did not)

    goal_swap_changes : swap the target button between click samples; share of predicted clicks that move
                        (want ~1.0; ~0 means the model ignores the goal and just recognises the frame)
    lookup_within1    : click accuracy of the goal->cell lookup ALONE, no pixels at all (want ~1.0)
    """
    model.eval()
    changed = swapped = hit = n = 0
    for b in DataLoader(ds, batch_size=64):
        b = _to(b, device)
        idx = torch.nonzero(b["t_cell"] >= 0).flatten()
        if len(idx) < 2:
            continue
        o = model(b)
        g = b["goal"].clone()
        g[idx] = b["goal"][idx.roll(1)]
        o2 = model(dict(b, goal=g))
        differs = (b["goal"][idx] != g[idx]).cpu()
        moved = (o["loc"][idx, 0].argmax(-1) != o2["loc"][idx, 0].argmax(-1)).cpu()
        changed += int(moved[differs].sum())
        swapped += int(differs.sum())
        look = o["loc_goal"][idx, 0].argmax(-1).cpu()
        hit += int((_cheb(look, b["t_cell"][idx].cpu(), grid) <= tol).sum())
        n += len(idx)
    return {"goal_swap_changes": changed / swapped if swapped else float("nan"),
            "lookup_within1": hit / n if n else float("nan")}


def majority_baseline(train_ds: StepDataset, val_ds: StepDataset) -> Dict[str, torch.Tensor]:
    """No-vision baseline: for (task, step index) predict the most common training action."""
    tt = train_ds.targets()
    tr_tasks = train_ds.tasks
    table = collections.defaultdict(collections.Counter)
    glob = collections.Counter()
    for j in range(len(tt["type"])):
        a = tuple(int(tt[k][j]) for k in ("type", "key", "text", "cell", "end"))
        table[(tr_tasks[int(tt["task"][j])], int(tt["step"][j]))][a] += 1
        glob[a] += 1
    vt = val_ds.targets()
    out = {k: [] for k in ("type", "key", "text", "cell", "end")}
    for j in range(len(vt["type"])):
        c = table.get((val_ds.tasks[int(vt["task"][j])], int(vt["step"][j])), glob)
        a = (c or glob).most_common(1)[0][0]
        for k, v in zip(("type", "key", "text", "cell", "end"), a):
            out[k].append(max(v, 0))
    return {k: torch.tensor(v) for k, v in out.items()}


def fmt(res: dict) -> str:
    keys = [k for k in res if k not in ("n", "per_task")]
    return "  ".join(f"{k}={res[k]:.2f}" if res[k] == res[k] else f"{k}=n/a" for k in keys)


def train(feat_dir, out_path: str, epochs: int = 60, bs: int = 16, lr: float = 1e-3, T: int = 6,
          seed: int = 0, val_frac: float = 0.25, holdout_task: Optional[str] = None,
          device: Optional[str] = None, d: int = 256, layers: int = 4, sigma: float = 2.0,
          eval_every: int = 5, verbose: bool = True, val_ids: Optional[set] = None,
          min_steps: int = 1200, table_lr: float = 2e-2, init_table: bool = False) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    eps = load_episodes(feat_dir)
    if not eps:
        raise SystemExit(f"no cached episodes in {feat_dir}; run the cache step first")
    if not isinstance(feat_dir, str):
        print(f"training on {len(eps)} episodes from {len(feat_dir)} feature folders")
    tr_eps, va_eps = split_episodes(eps, val_frac, seed, holdout_task, val_ids)
    vocab = build_vocab(tr_eps)
    tr, va = StepDataset(tr_eps, vocab, T), StepDataset(va_eps, vocab, T)
    gh, gw, C = tr_eps[0].patch.shape[1:]
    print(f"device={device}  train: {len(tr_eps)} episodes / {len(tr)} steps   "
          f"val: {len(va_eps)} episodes / {len(va)} steps   vocab: keys={vocab.keys} texts={vocab.texts}")

    model = ArgusPolicy(len(vocab.keys), len(vocab.texts), vis_dim=C, d=d, layers=layers, T=T,
                        gh=gh, gw=gw).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    if init_table:
        print(f"goal lookup initialised from the demos for {init_goal_table(model, tr_eps, sigma)} goals")
    # The goal->cell lookup table is a sparse embedding (each row is touched only when its goal appears),
    # so with the base lr it barely moves in a short run and the frame path wins. Give it a bigger lr.
    others = [p for n, p in model.named_parameters() if n != "goal_table.weight"]
    opt = torch.optim.AdamW([{"params": others},
                             {"params": [model.goal_table.weight], "lr": table_lr, "weight_decay": 0.0}],
                            lr=lr, weight_decay=1e-2)
    loader = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=False)
    # tiny datasets give few updates per epoch; make sure the run has at least min_steps of them
    epochs = max(epochs, -(-min_steps // max(1, len(loader))))
    print(f"epochs={epochs}  updates={epochs * len(loader)}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(loader)))

    best, best_state, history = -1.0, None, []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for b in loader:
            b = _to(b, device)
            loss, parts = bc_loss(model(b), b, sigma=sigma)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b["t_type"])
        tot /= len(tr)
        rec = {"epoch": ep, "loss": tot}
        if len(va) and (ep % eval_every == 0 or ep == epochs):
            res = score(collect_preds(model, va, device), va.targets(), va.tasks)
            rec["val"] = res
            if res["step_acc"] > best:
                best = res["step_acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"epoch {ep:3d}  loss={tot:.3f}  VAL {fmt(res)}")
        elif verbose and ep % eval_every == 0:
            print(f"epoch {ep:3d}  loss={tot:.3f}")
        history.append(rec)

    state = best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"model": state, "cfg": model.cfg, "vocab": vocab.to_dict(), "T": T, "grid": GRID,
                "holdout_task": holdout_task,
                "seen_types": sorted(set(int(t) for t in tr.targets()["type"].tolist())),
                "seen_goals": sorted({GOALS[g] for e in tr_eps for g in e.goals})}, out_path)
    print(f"saved {out_path}")

    final = {}
    if len(va):
        model.load_state_dict(state)
        final["model"] = score(collect_preds(model, va, device), va.targets(), va.tasks)
        final["baseline"] = score(majority_baseline(tr, va), va.targets(), va.tasks)
        final["train"] = score(collect_preds(model, tr, device), tr.targets(), tr.tasks)
        final["diag"] = diagnose(model, va, device)
        merged: Dict[int, list] = {}
        for part in (lookup_by_goal(model, tr, device), lookup_by_goal(model, va, device)):
            for g, (h, n) in part.items():
                m = merged.setdefault(g, [0, 0])
                m[0] += h
                m[1] += n
        final["lookup_by_goal"] = {GOALS[g]: (h / n, n) for g, (h, n) in merged.items()}
        print("\n== FINAL (best val checkpoint) ==")
        print("train   :", fmt(final["train"]))
        print("model   :", fmt(final["model"]))
        print("baseline:", fmt(final["baseline"]), " (no vision: majority action per task+step)")
        dg = final["diag"]
        print(f"goal check: changing the target button moves the click in {dg['goal_swap_changes']:.0%} of cases "
              f"(want ~100%);  goal-only lookup within1 = {dg['lookup_within1']:.0%}  (want ~100%)")
        weak = sorted((v[0], v[1], k) for k, v in final["lookup_by_goal"].items() if v[0] < 0.9)
        print("lookup is off for: " + (", ".join(f"{k} {a:.0%} (n={n})" for a, n, k in weak) if weak else "no goal (all >= 90%)"))
        for t, r in final["model"]["per_task"].items():
            print(f"   {t:12s} n={r['n']:3d}  model step_acc={r['step_acc']:.2f}  "
                  f"baseline={final['baseline']['per_task'][t]['step_acc']:.2f}")
    return {"history": history, "final": final}
