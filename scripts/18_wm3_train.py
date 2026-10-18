"""World model 3, stage 2b: train the event model on the snapshots of scripts/17_wm3_dataset.py.

    .venv/bin/python scripts/18_wm3_train.py --steps 30000 --tag wm3_base

Each step draws B snapshots of one city (alternating at random) from a rotating pool of training
days and minimises the summed per-head negative log-likelihood (world/wm3/model.py). Validation
uses fixed snapshots of the validation days. Checkpoints: data/wm3_models/<tag>.pt;
logs: outputs/wm3/stage2/train_<tag>.csv.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from world import cities
from world.wm3 import data as WD
from world.wm3 import model as WM

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = Path(__file__).resolve().parents[1] / "data" / "wm3_models"
OUT = cities.OUTPUTS / "wm3" / "stage2"


def tables():
    return {k: WD.CityTables(k, DEV) for k in WD.CITY_KEYS}


def make_model(tabs, ck=None, **kw):
    m = WM.WorldModel3(tabs, **(ck["arch"] if ck else kw)).to(DEV)
    if ck:
        m.load_state_dict(ck["state"])
    return m


def load(tag, tabs=None):
    ck = torch.load(MODELS / f"{tag}.pt", map_location=DEV)
    return make_model(tabs or tables(), ck), ck


def fixed_snaps(split, n_per_city, seed):
    out = []
    for k in WD.CITY_KEYS:
        pool = WD.DayPool(split, n_days=6, swap=10 ** 9, seed=seed, keys=[k])
        rng = np.random.default_rng(seed)
        for _ in range(n_per_city):
            day = pool.pool[rng.integers(len(pool.pool))]
            out.append(WD.snapshot(day, int(rng.integers(len(day["epoch"])))))
    return out


@torch.no_grad()
def evaluate(model, snaps, bs=4):
    model.eval()
    agg, cnt = {}, {}
    by_city = {k: [s for s in snaps if s["key"] == k] for k in WD.CITY_KEYS}
    for k, ss in by_city.items():
        for i in range(0, len(ss), bs):
            b = WD.collate(ss[i: i + bs], DEV)
            with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
                out = model(b)
            out = {kk: (v.float() if torch.is_tensor(v) else v) for kk, v in out.items()}
            for h, v in WM.losses(model, b, out).items():
                agg[(k, h)] = agg.get((k, h), 0.0) + float(v)
                cnt[(k, h)] = cnt.get((k, h), 0) + 1
    model.train()
    return {f"{k}/{h}": agg[(k, h)] / cnt[(k, h)] for k, h in agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--tag", default="wm3_base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="start from this checkpoint tag (fine-tuning)")
    ap.add_argument("--tf", action="store_true", help="time bumps into the driver heads (world/wm3/model.py)")
    ap.add_argument("--exo", action="store_true", help="demand heads see no supply (world/wm3/model.py)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    tabs = tables()
    if args.init:
        ck0 = torch.load(MODELS / f"{args.init}.pt", map_location=DEV)
        arch = {**ck0["arch"], **({"tf": True} if args.tf else {}), **({"exo": True} if args.exo else {})}
        model = make_model(tabs, **arch)
        own = model.state_dict()
        keep = {k: v for k, v in ck0["state"].items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"  from {args.init}: {len(keep)}/{len(own)} tensors copied, new: "
              f"{sorted({k.rsplit('.', 2)[0] for k in own if k not in keep})}", flush=True)
    else:
        arch = {"d": args.d, "layers": args.layers, "heads": args.heads, **({"tf": True} if args.tf else {}),
                **({"exo": True} if args.exo else {})}
        model = make_model(tabs, **arch)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))
    pools = {k: WD.DayPool("train", n_days=8, swap=60, seed=args.seed + i, keys=[k]) for i, k in enumerate(WD.CITY_KEYS)}
    val = fixed_snaps("val", 48, seed=1)
    rng = np.random.default_rng(args.seed)
    print(f"== {args.tag}: {n_par / 1e6:.2f} M parameters ({sum(p.numel() for n, p in model.named_parameters() if n.startswith('node')) / 1e6:.2f} M in node tables), "
          f"{len(pools['sf_hd'].files)} + {len(pools['pune_hd'].files)} training days, {len(val)} validation snapshots, on {DEV}", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    best, log, t0, run = float("inf"), [], time.time(), {}
    for step in range(1, args.steps + 1):
        key = WD.CITY_KEYS[rng.integers(2)]
        snaps = []
        for _ in range(args.batch):
            day, i = pools[key].sample()
            snaps.append(WD.snapshot(day, i))
        b = WD.collate(snaps, DEV)
        with torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda"):
            out = model(b)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        ls = WM.losses(model, b, out)
        loss = sum(ls.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        for k, v in ls.items():
            run[k] = 0.98 * run.get(k, float(v)) + 0.02 * float(v)
        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate(model, val)
            total = sum(v.values()) / 2
            log.append({"step": step, "min": (time.time() - t0) / 60, "val_total": total, **{f"train/{k}": x for k, x in run.items()}, **v})
            flag = ""
            if total < best:
                best = total
                torch.save({"state": model.state_dict(), "arch": arch, "step": step, "val": v}, MODELS / f"{args.tag}.pt")
                flag = " *"
            heads = sorted({k.split("/")[1] for k in v})
            print(f"  step {step:6d}  {log[-1]['min']:.1f} min  val total {total:.4f}{flag}\n    " +
                  "  ".join(f"{h} {v.get('sf_hd/' + h, float('nan')):.4f}/{v.get('pune_hd/' + h, float('nan')):.4f}" for h in heads), flush=True)
            pd.DataFrame(log).to_csv(OUT / f"train_{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
