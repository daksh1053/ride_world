"""Stage 15 (step 6): paired training data, the same morning with and without an extra order.

    .venv/bin/python scripts/15_pairs.py --city sf_hd --seed0 300 --n-seeds 10      # one worker
    (several workers with different --seed0 run in parallel; each writes its own records)

Step 4d found that the models put the effect of a repositioning order in the right places but
make it several times too small. In ordinary data the effect of an order is hidden among
everything else that happens, and about 30% of ordered drivers decline (a per-driver latent
the 64 x 64 frames do not show). Twin runs show the effect directly: for each seed one base
morning and ARMS counterfactual mornings, each with one random extra order, all from identical
starting states with the same random numbers until the order. Their difference is the effect of
that order (plus the chance divergence that follows), which step 6 trains the model to predict.

Every order is random, to cover the space the planner chooses from:
    time    a policy epoch in 06:30-09:00, never on a 5-minute snapshot (as the training data)
    size    N_RANGE idle drivers
    source  one of the 8 zones with most idle drivers, plus its nearest zones until N drivers
            (at most 15 min away; early in the morning fewer than N may be moved)
    target  any zone 3-30 min away (half the time one of the 10 with the largest deficit)
Only training-split dates are used (the benchmarks of steps 4d and 5 are on test dates).

Writes data/<city>/wm2_pairs/seed<k>_{base,cf<j>}.npz (store units: X uint8 incl. the
direction channels, ctx) and rec_seed<k>.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from world import cities
from world.hd import raster as R
from world.hd.run import Season
from world.hd.sim import HDDispatch
from world.wm2 import data as WD

BASE = dict(reposition_every=10, max_moves=60, explore=0.05)       # as stages 13 and 14
N_RANGE = {"sf_hd": (10, 70), "pune_hd": (20, 140)}
ARMS = 3
HORIZON_H = 10.0
FOLDER = "wm2_pairs"


class RandomOrder(HDDispatch):
    """HDDispatch plus one random extra order at time `t_order` (none if n == 0)."""

    def __init__(self, n=0, t_order=0.0, pick_seed=0, **kw):
        super().__init__(**kw)
        self.n, self.t_order, self.done, self.record = n, t_order, False, {}
        self.pick = np.random.default_rng(pick_seed)        # separate from the simulator's numbers

    def act(self, obs):
        act = super().act(obs)
        if not self.n or self.done or obs.t < self.t_order:
            return act
        self.done = True
        g = self.pick
        used = {d for d, _ in act.offers} | {d for d, _ in act.dispatches}
        free = obs.idle[~obs.idle.driver_id.isin(used)]
        Z = len(obs.recent_demand)
        cnt = np.bincount(free.zone, minlength=Z)
        src = int(g.choice(np.argsort(-cnt)[:8]))
        near = np.argsort(obs.zone_tt[src])
        k = int(np.searchsorted(np.cumsum(cnt[near]), self.n)) + 1
        take = near[: min(k, max(1, int((obs.zone_tt[src] <= 900).sum())))]      # at most 15 min around the source
        need = obs.recent_demand / obs.recent_window_s * 600 + np.bincount(obs.pending.origin_zone, minlength=Z)
        deficit = need - (np.bincount(obs.idle.zone, minlength=Z) + obs.repositioning_to)
        tt = obs.zone_tt[src]
        reach = np.flatnonzero((tt >= 180) & (tt <= 1800) & ~np.isin(np.arange(Z), take))
        if not len(reach):
            reach = np.flatnonzero(~np.isin(np.arange(Z), take))
        if g.random() < 0.5:
            dst = int(g.choice(reach[np.argsort(-deficit[reach])][:10]))
        else:
            dst = int(g.choice(reach))
        movers = free[free.zone.isin(take)].sort_values("idle_since").driver_id.to_numpy()[: self.n]
        act.dispatches = act.dispatches + [(int(d), dst) for d in movers]
        self.record = {"t": float(obs.t), "src_zone": src, "src_zones": [int(z) for z in take], "dst_zone": dst,
                       "n": self.n, "n_moved": int(len(movers)), "idle_in_src": int(cnt[take].sum()),
                       "deficit_dst": float(deficit[dst]), "travel_s": float(tt[dst])}
        return act


def save_arm(path, cr, day_dir, seed):
    """One run in store units: X (T, 22 + 8 direction channels, G, G) uint8 and ctx."""
    X, C, _, _ = cr.day(day_dir, seed=seed)
    X[:, WD.SURGE] *= 100
    T = int(HORIZON_H * 12)                                      # frames up to the end of the run
    X = np.clip(np.round(X[:T]), 0, 255).astype(np.uint8)
    Dd = cr.dispatch_dirs(day_dir)[:T].astype(np.uint8)
    np.savez_compressed(path, X=np.concatenate([X, Dd], 1), ctx=C[:T].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["sf_hd", "pune_hd"])
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()
    city = cities.get(args.city)
    out = city.data_dir / FOLDER
    out.mkdir(exist_ok=True)
    info = json.loads((city.data_dir / "wm2" / "store" / "days.json").read_text())
    val_from = json.loads((city.data_dir / "wm2" / "split.json").read_text())["val_from"]
    train_dates = [d for d in info["dates"] if d < val_from]
    cr = R.CityRaster(city)
    lo, hi = N_RANGE[args.city]
    for seed in range(args.seed0, args.seed0 + args.n_seeds):
        rf = out / f"rec_seed{seed}.json"
        if rf.exists():
            continue
        t0 = time.time()
        g = np.random.default_rng(seed)
        date = str(g.choice(train_dates))
        tmp = out / f"runs{seed}"
        arms = [("base", 0, 0.0)]
        for j in range(ARMS):
            t_order = 6.5 * 3600 + 30 * int(g.integers(0, 300))          # 06:30-08:59:30, every policy epoch
            if t_order % 300 == 0:
                t_order += 30 * int(g.integers(1, 10))                   # never on a snapshot
            arms.append((f"cf{j}", int(g.integers(lo, hi + 1)), float(t_order)))
        rec = {"seed": seed, "date": date, **BASE, "arms": []}
        for arm, n, t_order in arms:
            pol = RandomOrder(n=n, t_order=t_order, pick_seed=seed * 10 + len(rec["arms"]), seed=seed, **BASE)
            season = Season(args.city, seed=seed, policy_factory=lambda: pol, log=lambda *a: None)
            season.run_day(date, out_dir=tmp / arm, horizon_h=HORIZON_H)
            save_arm(out / f"seed{seed}_{arm}.npz", cr, tmp / arm / date, seed)
            rec["arms"].append({"arm": arm, "t_order": t_order, **pol.record})
        shutil.rmtree(tmp)
        rf.write_text(json.dumps(rec, indent=1))
        print(f"  seed {seed} {date}: " + "; ".join(
            f"{a['arm']} {a['n_moved']}/{a['n']} at {a['t'] / 3600:.2f} h, {a['travel_s'] / 60:.0f} min" for a in rec["arms"][1:])
              + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
