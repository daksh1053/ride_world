"""Stage 11: train and evaluate world model v2 (the 64 x 64 spatio-temporal models).

    .venv/bin/python scripts/11_wm2_train.py --build                  # pack the rasters into stores
    .venv/bin/python scripts/11_wm2_train.py --build-dirs             # step 4: dispatch-direction channels
    .venv/bin/python scripts/11_wm2_train.py --model st --K 12 --dirs --steps 40000 --tag st4_k12_dirs --outdir step4
    .venv/bin/python scripts/11_wm2_train.py --model st --steps 20000 --tag st_small
    .venv/bin/python scripts/11_wm2_train.py --model cnn --steps 6000 --tag cnn
    .venv/bin/python scripts/11_wm2_train.py --finetune st_small --rollout 6 --steps 4000 --tag st_ro
                                              # continue training on the model's own rollouts
    .venv/bin/python scripts/11_wm2_train.py --finetune st4_skip_ros --rollout 6 --roll-feed sample --pairs 1 --tag st6_pairs --outdir step6
                                              # step 6: also train on twin runs (scripts/15_pairs.py)
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
from world.wm2 import pairs as WP

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
            # floor 0.001 per cell (until step 4d it was 0.02, which made this baseline weaker:
            # test Poisson NLL SF .127 -> .114, Pune .316 -> .309; chosen on test days, so it is if
            # anything slightly favourable to the baseline)
            out[:, :WM.N_COUNT] = np.log(acc[:, WD.COUNT] + 0.001)
            out[:, WM.N_COUNT:WM.N_COUNT + WM.N_SURGE] = acc[:, WD.SURGE] / 100.0
            mu = np.where(sp_n > 0, sp_s / np.maximum(sp_n, 1), 30.0)
            var = np.where(sp_n > 1, sp_q / np.maximum(sp_n, 1) - mu ** 2, 64.0)
            out[:, -2] = mu / 60.0
            out[:, -1] = np.log(np.clip(var, 4.0, 2000.0))
            tabs[(s.id, wk)] = torch.tensor(out)
    return tabs


def make_model(kind, K, dirs=False, skip=False):
    c_in = C_IN + (WD.N_DIRS if dirs else 0)
    if kind == "st":
        return WM.STTransformer(c_in, K=K, skip=skip)
    if kind == "st_large":
        return WM.STTransformer(c_in, d=256, layers=8, heads=8, K=K)
    if kind == "cnn":
        return WM.LocalCNN(c_in)
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
def evaluate(model, batcher, items, clim=None, bs=8, last=None):
    """Mean losses over the windows; `last`: score only the last n positions of each."""
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
        if last:
            out, tgt = out[:, -last:], tgt[:, -last:]
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


def rollout_loss(model, x, c, S, K, R, backward=False, feed="mean"):
    """Unroll R steps: each new frame the model predicts (detached, as an expected value)
    replaces the true one in its input; every step is scored on the true frames. This
    teaches the model to stay on track when it reads its own predictions (exposure bias).
    The fed-back frames are detached, so with backward=True each step's gradient is
    accumulated right away and only one step's graph is held in GPU memory.
    feed="sample" (step 4d) feeds back Poisson-sampled whole counts instead of expected ones,
    so the inputs look like real frames (scheduled sampling)."""
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
                f = to_frame(out[:, -1].detach())
                if feed == "sample":
                    f[:, WD.COUNT] = torch.poisson(f[:, WD.COUNT])
                buf[:, r + K, WD.OBS] = f[:, WD.OBS].clamp(0, 255)
    return total


def pair_loss(model, x, c, S, K, H, w, backward=False):
    """Step 6: roll out a base run and its counterfactual twin (x: the B bases, then the B
    twins) H steps from their shared history, each with its own actions, feeding back coupled
    Poisson samples. Each step is scored on its newest frame by the usual loss on both runs,
    plus w x the squared error of the predicted twin-minus-base difference of every count
    channel in every 4 x 4 block, scaled by the Poisson variance of the two runs, so the
    effect of the order itself is trained and not only each run on its own.
    Returns (usual loss, difference loss), means over the steps."""
    B = x.shape[0] // 2
    buf = x.float().clone()
    m = road_mask(S)[:B, None]
    nll_t = diff_t = 0.0
    for r in range(H):
        inp, ctx, tgt = WD.make_io(buf[:, r: r + K + 1], c[:, r: r + K + 1], S)
        with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
            out = model(inp, ctx)
        o, y = out[:, -1:].float(), tgt[:, -1:]
        nll = total_loss(WM.losses(o, y, road_mask(S)))
        mu = WM.split_out(o[:, 0])[0].clamp(-12, 8).exp()
        yc = y[:, 0, WD.COUNT]
        dp = WP.blocks((mu[B:] - mu[:B]) * m)
        dt = WP.blocks((yc[B:] - yc[:B]) * m)
        var = WP.blocks((mu[B:] + mu[:B]) * m).detach() + 1.0
        diff = ((dp - dt) ** 2 / var).mean()
        loss = (nll + w * diff) / H
        if backward:
            loss.backward()
        nll_t += nll.item() / H
        diff_t += diff.item() / H
        if r < H - 1:
            with torch.no_grad():
                f = to_frame(o[:, 0].detach())
                f[:B, WD.COUNT], f[B:, WD.COUNT] = WP.coupled_poisson(f[:B, WD.COUNT], f[B:, WD.COUNT])
                buf[:, r + K, WD.OBS] = f[:, WD.OBS].clamp(0, 255)
    return nll_t, diff_t


def load_ckpt(tag):
    ck = torch.load(MODELS / f"{tag}.pt", map_location=DEV)
    m = make_model(ck["kind"], ck["K"], ck.get("dirs", False), ck.get("skip", False)).to(DEV)
    m.load_state_dict(ck["state"])
    return m, ck


def train(args, stores, splits):
    R = args.rollout
    WD.DIRS = args.dirs
    batcher = WD.Batcher(stores, {k: v[0] for k, v in splits.items()}, args.K + R, DEV, seed=args.seed, act_focus=args.act_focus)
    torch.manual_seed(args.seed)
    if args.finetune and args.skip:                                       # add the skip path to a model without it
        ck = torch.load(MODELS / f"{args.finetune}.pt", map_location=DEV)
        model = make_model(ck["kind"], ck["K"], ck.get("dirs", False), True).to(DEV)
        model.load_state_dict(ck["state"], strict=ck.get("skip", False))
        args.model, args.dirs = ck["kind"], ck.get("dirs", False)
        WD.DIRS = args.dirs
    elif args.finetune:
        model, ck = load_ckpt(args.finetune)
        args.skip = ck.get("skip", False)
        args.model, args.dirs = ck["kind"], ck.get("dirs", False)
        WD.DIRS = args.dirs
        if ck["K"] != args.K:
            raise SystemExit(f"{args.finetune} was trained with K={ck['K']}")
    else:
        model = make_model(args.model, args.K, args.dirs, args.skip).to(DEV)
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
    if args.pairs:
        ptr = WP.PairSet(stores, args.K, args.pair_H, DEV, dirs=args.dirs, which="train", seed=args.seed)
        pva = WP.PairSet(stores, args.K, args.pair_H, DEV, dirs=args.dirs, which="val")
        print(f"  pairs: {len(ptr)} training, {len(pva)} validation", flush=True)
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
            loss = torch.tensor(rollout_loss(model, x, c, S, args.K, R, backward=True, feed=args.roll_feed))
        else:
            inp, ctx, tgt = WD.make_io(x, c, S)
            with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
                out = model(inp, ctx)
            loss = total_loss(WM.losses(out.float(), tgt, road_mask(S)))
            loss.backward()
        if args.pairs:
            pl = pair_loss(model, *ptr.sample(args.pair_batch), args.K, args.pair_H, args.pairs, backward=True)
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
            if args.pairs:
                model.eval()
                torch.manual_seed(0)
                with torch.no_grad():
                    pv = np.mean([pair_loss(model, *pva.gather(np.arange(i, min(i + 4, len(pva)))), args.K, args.pair_H, args.pairs)
                                  for i in range(0, len(pva), 4)], 0)
                log[-1].update({"val_pair_nll": float(pv[0]), "val_pair_diff": float(pv[1]), "train_pair_diff": pl[1]})
                vl = log[-1]["val_select"] = vl + float(pv[0]) + args.pairs * float(pv[1])
                print(f"    pairs: val usual {pv[0]:.4f}  difference {pv[1]:.4f}  (train difference {pl[1]:.4f})", flush=True)
            flag = ""
            if vl < best:
                best = vl
                torch.save({"state": model.state_dict(), "kind": args.model, "K": args.K, "cities": args.cities, "rollout": R,
                            "from": args.finetune, "dirs": args.dirs, "skip": args.skip, "roll_feed": args.roll_feed, "pairs": args.pairs}, MODELS / f"{args.tag}.pt")
                flag = " *"
            print(f"  step {step:6d}  train {loss.item():.4f}  val {vl:.4f}  (poisson {log[-1]['val_poisson']:.4f}, "
                  f"speed {log[-1]['val_speed']:.3f}, surge {log[-1]['val_surge']:.5f})"
                  + (f"  rollout {log[-1]['val_rollout']:.4f}" if R else "") + (f"  selection {vl:.4f}" if args.pairs else "") + f"  {log[-1]['min']:.1f} min{flag}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(OUT / f"train_{args.tag}.csv", index=False)


def run_eval(args, stores, splits):
    res = {}
    clim = WM.Climatology(climatology(stores, splits))
    models = {"persistence": (WM.Persistence(), {"K": args.K}), "climatology": (clim, {"K": args.K})}
    for tag in args.eval:
        models[tag] = load_ckpt(tag)
    # the same test frames for every model: windows of the longest history used, each model
    # reads its own K frames before the scored ones and is scored on the last args.K frames
    k_max = max(ck["K"] for _, ck in models.values())
    base = fixed_windows(stores, splits, 2, args.n_test, k_max, seed=11)
    for name, (m, ck) in models.items():
        K = ck["K"]
        items = [(s, d, t + k_max - K) for s, d, t in base]
        batcher = WD.Batcher(stores, {}, K, DEV, dirs=ck.get("dirs", False))
        per_city = {}
        for s in stores:
            it = [x for x in items if x[0] is s]
            r = evaluate(m, batcher, it, last=args.K)
            per_city[s.key] = {"poisson_mean": float(r["poisson"].mean()), "speed_nll": float(r["speed"]),
                               "surge_mse": float(r["surge"]),
                               **{f"nll_{c}": float(v) for c, v in zip(WD.COUNT_NAMES, r["poisson"])}}
        res[name] = per_city
        print(f"  {name:16s} " + "  ".join(f"{k}: poisson {v['poisson_mean']:.4f} speed {v['speed_nll']:.3f} surge {v['surge_mse']:.5f}"
                                           for k, v in per_city.items()), flush=True)
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
    ap.add_argument("--build-dirs", action="store_true", help="step 4: add the dispatch-direction channels to the stores")
    ap.add_argument("--dirs", action="store_true", help="step 4: feed the dispatch-direction channels to the model")
    ap.add_argument("--skip", action="store_true", help="step 4d: add the per-cell skip path to the transformer")
    ap.add_argument("--roll-feed", default="mean", choices=["mean", "sample"], help="step 4d: what rollout training feeds back")
    ap.add_argument("--act-focus", type=float, default=0.0, help="step 4d: share of windows centred on large repositioning orders")
    ap.add_argument("--pairs", type=float, default=0.0, help="step 6: weight of the twin-run difference loss (0 = no pairs)")
    ap.add_argument("--pair-batch", type=int, default=3, help="step 6: twin pairs per step")
    ap.add_argument("--pair-H", type=int, default=12, help="step 6: rollout steps of each pair")
    ap.add_argument("--outdir", default="step3", help="results go to outputs/wm2/<outdir>/")
    args = ap.parse_args()
    global OUT
    OUT = cities.OUTPUTS / "wm2" / args.outdir
    if args.build or args.build_dirs:
        for k in args.cities:
            if args.build:
                WD.build_store(cities.get(k))
            WD.build_dirs(cities.get(k))
        return
    stores, splits = load(args.cities)
    if args.eval is not None:
        run_eval(args, stores, splits)
    else:
        args.tag = args.tag or args.model
        train(args, stores, splits)


if __name__ == "__main__":
    main()
