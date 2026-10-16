"""Stage 13: ground-truth counterfactuals from the simulator, to test the world model's
understanding of the platform's actions.

    .venv/bin/python scripts/13_cf_truth.py --city sf_hd --pairs 8
    .venv/bin/python scripts/13_cf_truth.py --city pune_hd --pairs 8

For each seed the same morning is simulated twice from identical starting states:
    base  the usual dispatcher (rebalancing every 10 epochs, 60 moves, exploration 0.05)
    cf    the same, plus one extra order at 08:00: N idle drivers from the zone with the most
          idle drivers and its nearest zones are all sent to the most under-supplied zone within 20 minutes
Both runs stop at 10:00. The two days are rasterised exactly like the training data, so
the model can be given either action sequence and its predicted difference compared with
the true difference (averaged over seeds, since the runs also diverge by chance).
Writes data/<city>/wm2_cf/seed<k>_{base,cf}.npz (+ _dirs.npy, the step-4 direction channels)
and pairs.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from world import cities
from world.hd import raster as R
from world.hd.run import Season
from world.hd.sim import HDDispatch

N_MOVE = {"sf_hd": 40, "pune_hd": 80}
T_ORDER = 8 * 3600.0
BASE = dict(reposition_every=10, max_moves=60, explore=0.05)


class Intervene(HDDispatch):
    """HDDispatch plus one extra repositioning order at T_ORDER (only if `n` > 0)."""

    def __init__(self, n=0, **kw):
        super().__init__(**kw)
        self.n, self.done, self.record = n, False, {}

    def act(self, obs):
        act = super().act(obs)
        if self.n and not self.done and obs.t >= T_ORDER:     # module global, set by --t-order
            self.done = True
            used = {d for d, _ in act.offers} | {d for d, _ in act.dispatches}
            free = obs.idle[~obs.idle.driver_id.isin(used)]
            Z = len(obs.recent_demand)
            cnt = np.bincount(free.zone, minlength=Z)
            src = int(cnt.argmax())
            # source area: the zone with most idle drivers plus its nearest zones until n drivers
            near = np.argsort(obs.zone_tt[src])
            take = near[: int(np.searchsorted(np.cumsum(cnt[near]), self.n)) + 1]
            need = obs.recent_demand / obs.recent_window_s * 600 + np.bincount(obs.pending.origin_zone, minlength=Z)
            deficit = need - (np.bincount(obs.idle.zone, minlength=Z) + obs.repositioning_to)
            reach = np.flatnonzero((obs.zone_tt[src] <= 1200) & ~np.isin(np.arange(Z), take))
            if not len(reach):          # the source area already covers every zone within 20 min
                reach = np.flatnonzero((obs.zone_tt[src] <= 1800) & ~np.isin(np.arange(Z), take))
            if not len(reach):
                reach = np.flatnonzero(~np.isin(np.arange(Z), take))
            dst = int(reach[np.argmax(deficit[reach])])
            movers = free[free.zone.isin(take)].sort_values("idle_since").driver_id.to_numpy()[: self.n]
            act.dispatches = act.dispatches + [(int(d), dst) for d in movers]
            self.record = {"t": float(obs.t), "src_zone": src, "src_zones": [int(z) for z in take], "dst_zone": dst, "n_moved": int(len(movers)),
                           "idle_in_src": int(cnt[take].sum()), "deficit_dst": float(deficit[dst]),
                           "travel_s": float(obs.zone_tt[src, dst])}
        return act


def main():
    global T_ORDER
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["sf_hd", "pune_hd"])
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--date", default=None, help="default: a test-period weekday of the dataset")
    ap.add_argument("--t-order", type=float, default=T_ORDER,
                    help="seconds after midnight; step 4d uses 08:02:30 (mid-frame, like the orders in the training data)")
    ap.add_argument("--out", default="wm2_cf", help="folder under data/<city>/")
    args = ap.parse_args()
    city = cities.get(args.city)
    date = args.date or {"sf_hd": "2025-04-30", "pune_hd": "2025-08-06"}[args.city]
    T_ORDER = args.t_order
    out = city.data_dir / args.out
    tmp = out / "runs"
    out.mkdir(exist_ok=True)
    cr = R.CityRaster(city)
    pairs_f = out / "pairs.json"
    pairs = json.loads(pairs_f.read_text()) if pairs_f.exists() else []
    done = {p["seed"] for p in pairs}
    for seed in range(100, 100 + args.pairs):
        if seed in done:
            continue
        rec = {"seed": seed, "date": date, "n": N_MOVE[args.city], **BASE}
        for arm, n in (("base", 0), ("cf", N_MOVE[args.city])):
            pol = Intervene(n=n, seed=seed, **BASE)
            season = Season(args.city, seed=seed, policy_factory=lambda: pol, log=lambda *a: None)
            season.run_day(date, out_dir=tmp / arm, horizon_h=10.0)
            X, C, Y, meta = cr.day(tmp / arm / date, seed=seed)
            R.save_day(out / f"seed{seed}_{arm}.npz", X, C, Y, {**meta, **BASE, "arm": arm})
            np.save(out / f"seed{seed}_{arm}_dirs.npy", cr.dispatch_dirs(tmp / arm / date))
            if arm == "cf":
                rec.update(pol.record)
        shutil.rmtree(tmp)
        pairs.append(rec)
        pairs_f.write_text(json.dumps(pairs, indent=1))
        print(f"  seed {seed}: moved {rec['n_moved']} drivers from zone {rec['src_zone']} ({rec['idle_in_src']} idle) "
              f"to zone {rec['dst_zone']} (deficit {rec['deficit_dst']:.1f}, {rec['travel_s'] / 60:.0f} min away)", flush=True)


if __name__ == "__main__":
    main()
