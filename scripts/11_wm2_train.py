"""Stage 11: train and evaluate world model v2 (the 64 x 64 spatio-temporal models).

    .venv/bin/python scripts/11_wm2_train.py --build                  # pack the rasters into stores
    .venv/bin/python scripts/11_wm2_train.py --model st --steps 20000 --tag st_small
    .venv/bin/python scripts/11_wm2_train.py --model cnn --steps 6000 --tag cnn
    .venv/bin/python scripts/11_wm2_train.py --finetune st_small --rollout 6 --steps 4000 --tag st_ro
                                              # continue training on the model's own rollouts
    .venv/bin/python scripts/11_wm2_train.py --eval st_small cnn      # test days, all baselines

One model is trained on both cities at once (the grid, channels and static layers are the
same; the static layers tell it which city it is in). Split by days, chronological:
train / validation / test = first 75% / next 10% / last 15% of each city's days.
Checkpoints go to data/wm2_models/<tag>.pt, results to outputs/wm2/<outdir>/ (one folder per step).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from world import cities
from world.wm2 import data as WD
from world.wm2 import models as WM

CITIES = ["sf_hd", "pune_hd"]
MODELS = Path(__file__).resolve().parents[1] / "data" / "wm2_models"
OUT = cities.OUTPUTS / "wm2" / "step3"                # set by --outdir: one folder per step
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_IN = len(WD.OBS) + len(WD.ACT) + 12


def load(keys):
    stores = [WD.CityStore(cities.get(k), i) for i, k in enumerate(keys)]
    splits = {s.key: s.split() for s in stores}
    return stores, splits


def road_mask(S):
    return (S[:, -1] > 0).float()                       # has_road


def climatology(stores, splits):
    """(city id, weekend) -> (T, N_OUT, G, G) predictions from the training days."""
    tabs = {}
    for s in stores:
        tr = splits[s.key][0]
        for wk in (0, 1):
            days = [d for d in tr if (s.meta[d]["weekday"] >= 5) == bool(wk)] or list(tr)
            acc = np.zeros((s.T,) + s.X.shape[2:], np.float64)
            sp_s = np.zeros((s.T,) + s.X.shape[3:]); sp_q = np.zeros_like(sp_s); sp_n = np.zeros_like(sp_s)
            for d in days:
                x = s.X[d].astype(np.float64)
                acc += x
                m = x[:, WD.SAMPLES] > 0
                sp = x[:, WD.SPEED]
                sp_s += sp * m; sp_q += sp ** 2 * m; sp_n += m
            acc /= len(days)
            out = np.zeros((s.T, WM.N_OUT) + s.X.shape[3:], np.float32)
            out[:, :WM.N_COUNT] = np.log(acc[:, WD.COUNT] + 0.02)
            out[:, WM.N_COUNT:WM.N_COUNT + WM.N_SURGE] = acc[:, WD.SURGE] / 100.0
            mu = np.where(sp_n > 0, sp_s / np.maximum(sp_n, 1), 30.0)
            var = np.where(sp_n > 1, sp_q / np.maximum(sp_n, 1) - mu ** 2, 64.0)
            out[:, -2] = mu / 60.0
            out[:, -1] = np.log(np.clip(var, 4.0, 2000.0))
            tabs[(s.id, wk)] = torch.tensor(out)
    return tabs


def make_model(kind, K):
    if kind == "st":
        return WM.STTransformer(C_IN, K=K)
    if kind == "st_large":
        return WM.STTransformer(C_IN, d=256, layers=8, heads=8, K=K)
    if kind == "cnn":
        return WM.LocalCNN(C_IN)
    raise SystemExit(kind)


def total_loss(ls):
    return ls["poisson"].mean() + ls["surge"] * 10 + ls["speed"] * 0.05


def fixed_windows(stores, splits, which, n, K, seed):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        s = stores[rng.integers(len(stores))]
        d = rng.choice(splits[s.key][which])
        items.append((s, int(d), int(rng.integers(0, s.T - K))))
    return items


@torch.no_grad()
def evaluate(model, batcher, items, clim=None, bs=8):
    model.eval()
    agg, n = {}, 0
    for i in range(0, len(items), bs):
        x, c, S, _, keys = batcher.gather(items[i: i + bs])
        inp, ctx, tgt = WD.make_io(x, c, S)
        with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
            if isinstance(model, WM.Climatology):
                out = model(inp, ctx, None, keys).to(DEV)
            else:
                out = model(inp, ctx, x[:, :-1].float())
        ls = WM.losses(out.float(), tgt, road_mask(S))
        for k, v in ls.items():
            agg[k] = agg.get(k, 0) + v.double().cpu().numpy() * len(keys)
        n += len(keys)
    return {k: v / n for k, v in agg.items()}


def to_frame(out):
    """Model output (B, N_OUT, G, G) -> expected observation (raw units) on the OBS channels."""
    lr, sg, mu, _ = WM.split_out(out.float())
    f = torch.zeros(out.shape[0], len(WD.CH), *out.shape[-2:], device=out.device)
    f[:, WD.COUNT] = lr.clamp(-12, 8).exp()
    f[:, WD.SURGE] = (sg * 100).clamp(min=0)
    f[:, WD.SPEED] = mu.clamp(0, 120)
    return f


def rollout_loss(model, x, c, S, K, R, backward=False):
    """Unroll R steps: each new frame the model predicts (detached, as an expected value)
    replaces the true one in its input; every step is scored on the true frames. This
    teaches the model to stay on track when it reads its own predictions (exposure bias).
    The fed-back frames are detached, so with backward=True each step's gradient is
    accumulated right away and only one step's graph is held in GPU memory."""
    buf = x.float().clone()
    total = 0.0
    for r in range(R + 1):
        inp, ctx, tgt = WD.make_io(buf[:, r: r + K + 1], c[:, r: r + K + 1], S)
        with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
            out = model(inp, ctx)
        loss = total_loss(WM.losses(out.float(), tgt, road_mask(S))) / (R + 1)
        if backward:
            loss.backward()
        total += loss.item()
        if r < R:
            with torch.no_grad():
                buf[:, r + K, WD.OBS] = to_frame(out[:, -1].detach())[:, WD.OBS]
    return total


def train(args, stores, splits):
    R = args.rollout
    batcher = WD.Batcher(stores, {k: v[0] for k, v in splits.items()}, args.K + R, DEV, seed=args.seed)
    torch.manual_seed(args.seed)
    if args.finetune:
        ck = torch.load(MODELS / f"{args.finetune}.pt", map_location=DEV)
        args.model = ck["kind"]
        model = make_model(args.model, args.K).to(DEV)
        model.load_state_dict(ck["state"])
    else:
        model = make_model(args.model, args.K).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    warm = 100 if args.finetune else 500
    sched = lambda s: min(1.0, s / warm) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0)))  # noqa: E731
    lr_s = torch.optim.lr_scheduler.LambdaLR(opt, sched)
    val = fixed_windows(stores, splits, 1, 192, args.K, seed=7)
    vbatch = WD.Batcher(stores, {}, args.K, DEV)
    if R:                                             # rollout runs are chosen by the rollout loss
        val_ro = fixed_windows(stores, splits, 1, 96, args.K + R, seed=8)
        vbatch_ro = WD.Batcher(stores, {}, args.K + R, DEV)
    best, log = float("inf"), []
    t0 = time.time()
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"== {args.tag}: {args.model}, {n_par / 1e6:.2f} M parameters, K={args.K}, batch {args.batch}, {args.steps} steps on {DEV}"
          + (f", fine-tuning {args.finetune} on {R}-step rollouts" if R or args.finetune else ""), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        x, c, S, _, _ = batcher.sample(args.batch)
        opt.zero_grad(set_to_none=True)
        if R:
            loss = torch.tensor(rollout_loss(model, x, c, S, args.K, R, backward=True))
        else:
            inp, ctx, tgt = WD.make_io(x, c, S)
            with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
                out = model(inp, ctx)
            loss = total_loss(WM.losses(out.float(), tgt, road_mask(S)))
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); lr_s.step()
        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate(model, vbatch, val)
            vl = float(v["poisson"].mean() + v["surge"] * 10 + v["speed"] * 0.05)
            log.append({"step": step, "train": loss.item(), "val": vl, "val_poisson": float(v["poisson"].mean()),
                        "val_speed": float(v["speed"]), "val_surge": float(v["surge"]), "min": (time.time() - t0) / 60})
            if R:
                model.eval()
                with torch.no_grad():
                    ro = [rollout_loss(model, *vbatch_ro.gather(val_ro[i: i + 8])[:3], args.K, R) for i in range(0, len(val_ro), 8)]
                vl = log[-1]["val_rollout"] = float(np.mean(ro))
            flag = ""
            if vl < best:
                best = vl
                torch.save({"state": model.state_dict(), "kind": args.model, "K": args.K, "cities": args.cities, "rollout": R,
                            "from": args.finetune}, MODELS / f"{args.tag}.pt")
                flag = " *"
            print(f"  step {step:6d}  train {loss.item():.4f}  val {vl:.4f}  (poisson {log[-1]['val_poisson']:.4f}, "
                  f"speed {log[-1]['val_speed']:.3f}, surge {log[-1]['val_surge']:.5f})"
                  + (f"  rollout {vl:.4f}" if R else "") + f"  {log[-1]['min']:.1f} min{flag}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(OUT / f"train_{args.tag}.csv", index=False)


def run_eval(args, stores, splits):
    batcher = WD.Batcher(stores, {}, args.K, DEV)
    items = fixed_windows(stores, splits, 2, args.n_test, args.K, seed=11)
    res = {}
    clim = WM.Climatology(climatology(stores, splits))
    models = {"persistence": WM.Persistence(), "climatology": clim}
    for tag in args.eval:
        ck = torch.load(MODELS / f"{tag}.pt", map_location=DEV)
        m = make_model(ck["kind"], ck["K"]).to(DEV)
        m.load_state_dict(ck["state"])
        models[tag] = m
    for name, m in models.items():
        per_city = {}
        for s in stores:
            it = [x for x in items if x[0] is s]
            r = evaluate(m, batcher, it)
            per_city[s.key] = {"poisson_mean": float(r["poisson"].mean()), "speed_nll": float(r["speed"]),
                               "surge_mse": float(r["surge"]),
                               **{f"nll_{c}": float(v) for c, v in zip(WD.COUNT_NAMES, r["poisson"])}}
        res[name] = per_city
        print(f"  {name:16s} " + "  ".join(f"{k}: poisson {v['poisson_mean']:.4f} speed {v['speed_nll']:.3f} surge {v['surge_mse']:.5f}"
                                           for k, v in per_city.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    rows = [{"model": m, "city": c, **v} for m, d in res.items() for c, v in d.items()]
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--cities", nargs="+", default=CITIES)
    ap.add_argument("--model", default="st", choices=["st", "st_large", "cnn"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", nargs="*", default=None)
    ap.add_argument("--finetune", default=None, help="start from this checkpoint tag")
    ap.add_argument("--rollout", type=int, default=0, help="train on R-step rollouts of the model's own predictions")
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--outdir", default="step3", help="results go to outputs/wm2/<outdir>/")
    args = ap.parse_args()
    global OUT
    OUT = cities.OUTPUTS / "wm2" / args.outdir
    if args.build:
        for k in args.cities:
            WD.build_store(cities.get(k))
        return
    stores, splits = load(args.cities)
    if args.eval is not None:
        run_eval(args, stores, splits)
    else:
        args.tag = args.tag or args.model
        train(args, stores, splits)


if __name__ == "__main__":
    main()
