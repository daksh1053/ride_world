"""World model 3, stage 4a: twin simulator runs, the same morning with and without a large order.

    .venv/bin/python scripts/22_wm3_twins.py --city sf_hd --seed0 500 --n-seeds 5

For each seed a fresh season (new drivers and riders, never seen in training) runs WARMUP days so
the drivers have a history the platform can read, then one test-split day three times:
    base      the platform's usual dispatch (HDDispatch, as in the training seasons)
    cf<n>     the same, plus one extra repositioning order of n idle drivers at t_order
All three runs are identical until t_order (same random numbers); both orders use the same time,
source zone and target zone (BigOrder), only the number of drivers differs.
The runs keep the full logs, so world/wm3/state.py rebuilds every driver, rider and road.

Writes data/<city>/wm3_twins/seed<k>/<date>/ (warm-up days and the base run),
data/<city>/wm3_twins/seed<k>_cf<n>/<date>/ and data/<city>/wm3_twins/rec_seed<k>.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from world import cities
from world.hd.run import Season

_spec = importlib.util.spec_from_file_location("pairs15", Path(__file__).with_name("15_pairs.py"))
P15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P15)

FOLDER = "wm3_twins"


class BigOrder(P15.RandomOrder.__mro__[1]):
    """HDDispatch plus one extra order at t_order: the n idle drivers nearest to a busy zone go to
    one zone 10-25 min away (half the time the one most short of drivers among those)."""

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
        src = int(g.choice(np.argsort(-cnt)[:3]))
        tt = obs.zone_tt[src]
        need = obs.recent_demand / obs.recent_window_s * 600 + np.bincount(obs.pending.origin_zone, minlength=Z)
        deficit = need - (np.bincount(obs.idle.zone, minlength=Z) + obs.repositioning_to)
        reach = np.flatnonzero((tt >= 600) & (tt <= 1500))
        if not len(reach):
            reach = np.argsort(-tt)[:10]
        dst = int(reach[np.argmax(deficit[reach])]) if g.random() < 0.5 else int(g.choice(reach))
        f = free.assign(d=obs.zone_tt[free.zone.to_numpy(), src])
        f = f[f.zone != dst].sort_values(["d", "idle_since"])
        movers = f.driver_id.to_numpy()[: self.n]
        act.dispatches = act.dispatches + [(int(d), dst) for d in movers]
        zs = sorted({int(z) for z in f.zone.to_numpy()[: self.n]})
        self.record = {"t": float(obs.t), "src_zone": src, "src_zones": zs, "dst_zone": dst, "n": self.n,
                       "n_moved": int(len(movers)), "idle_free": int(len(free)), "deficit_dst": float(deficit[dst]),
                       "travel_s": float(tt[dst])}
        return act


WARMUP = 3
SIZES = {"sf_hd": (30, 60), "pune_hd": (100, 200)}       # midday idle: ~200 SF, ~800-950 Pune
HORIZON_H = 15.0
BASE = P15.BASE


def run_seed(key, seed):
    city = cities.get(key)
    out = city.data_dir / FOLDER
    out.mkdir(exist_ok=True)
    rf = out / f"rec_seed{seed}.json"
    if rf.exists():
        return json.loads(rf.read_text())
    t0 = time.time()
    g = np.random.default_rng(seed)
    split = json.loads((city.data_dir / "wm2" / "split.json").read_text())
    test = pd.date_range(split["test_from"], split["test_until"])
    date = pd.Timestamp(g.choice(test))
    t_order = 11 * 3600 + 30 * int(g.integers(0, 240))            # 11:00-12:59:30, a policy epoch
    if t_order % 300 == 0:
        t_order += 30 * int(g.integers(1, 10))
    pick_seed = seed * 10 + 1
    # warm-up days and the base day share one season (its people evolve between days)
    pol_holder = {}

    def factory():
        pol_holder["p"] = BigOrder(n=0, seed=seed, **BASE)
        return pol_holder["p"]

    season = Season(key, seed=seed, policy_factory=factory, log=lambda *a: None)
    for k in range(WARMUP, 0, -1):
        season.run_day(date - pd.Timedelta(days=k), out_dir=out / f"seed{seed}")
    # snapshot the season so every arm starts its day from the same people
    import copy
    state = copy.deepcopy(season)
    season.run_day(date, out_dir=out / f"seed{seed}", horizon_h=HORIZON_H)
    rec = {"seed": seed, "date": str(date.date()), "t_order": t_order, "warmup": WARMUP, **BASE, "arms": []}
    for n in SIZES[key]:
        s2 = copy.deepcopy(state)
        pol = BigOrder(n=n, t_order=t_order, pick_seed=pick_seed, seed=seed, **BASE)
        s2.policy_factory = lambda pol=pol: pol
        s2.run_day(date, out_dir=out / f"seed{seed}_cf{n}", horizon_h=HORIZON_H)
        rec["arms"].append({"arm": f"cf{n}", **pol.record})
    rf.write_text(json.dumps(rec, indent=1))
    print(f"  {key} seed {seed} {rec['date']} order at {t_order / 3600:.2f} h: " + "; ".join(
        f"{a['arm']} moved {a['n_moved']}/{a['n']} zones {a['src_zones'][:3]}.. -> {a['dst_zone']} ({a['travel_s'] / 60:.0f} min)"
        for a in rec["arms"]) + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["sf_hd", "pune_hd"])
    ap.add_argument("--seed0", type=int, default=500)
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()
    for seed in range(args.seed0, args.seed0 + args.n_seeds):
        run_seed(args.city, seed)


if __name__ == "__main__":
    main()
