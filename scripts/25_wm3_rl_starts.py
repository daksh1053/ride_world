"""World model 3, stage 5a: a bank of real start states for training the agent in imagination.

    .venv/bin/python scripts/25_wm3_rl_starts.py

For each city, days of the training seasons (data/<city>/hd_runs/wm2_seed*/) are rebuilt
(world/wm3/state.py) and the entity-level world is taken at random decision epochs between
06:00 and 22:00, with each driver's records from the earlier days of the season (FT.Prior).
Each start is an imagined world (world/wm3/agent.py) pickled without the shared parts.
Training starts come from training days, validation starts from validation days.
Writes data/<city>/wm3_rl/starts/{train,val}_<date>_<epoch>.pkl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from world import cities
from world.wm3 import agent as AG
from world.wm3 import features as FT
from world.wm3 import rollout as RO
from world.wm3 import state as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["sf_hd", "pune_hd"])
    ap.add_argument("--train-days", type=int, default=40)
    ap.add_argument("--val-days", type=int, default=8)
    ap.add_argument("--per-day", type=int, default=3)
    args = ap.parse_args()
    for key in args.cities:
        city = cities.get(key)
        out = city.data_dir / "wm3_rl" / "starts"
        out.mkdir(parents=True, exist_ok=True)
        split = json.loads((city.data_dir / "wm2" / "split.json").read_text())
        g = S.CityGraph(city)
        rules = RO.CityRules(g)
        days = []
        for season in sorted((city.data_dir / "hd_runs").glob("wm2_seed*")):
            for d in sorted(p for p in season.iterdir() if p.is_dir() and p.name.startswith("20")):
                days.append(d)
        split_of = lambda d: "train" if d.name <= split["train_until"] else ("val" if d.name <= split["val_until"] else "test")  # noqa: E731
        rng = np.random.default_rng(0)
        want = {}
        for sp, n in (("train", args.train_days), ("val", args.val_days)):
            pool = [d for d in days if split_of(d) == sp]
            for d in rng.choice(len(pool), min(n, len(pool)), replace=False):
                want[pool[d]] = sp
        n_done, t0 = 0, time.time()
        for season in sorted({d.parent for d in want}):
            prior = FT.Prior()
            for d in sorted(p for p in season.iterdir() if p.is_dir() and p.name.startswith("20")):
                if d in want:
                    L = S.DayLog(g, d)
                    for e in sorted(rng.choice(np.arange(6 * 120, 22 * 120, AG.DECIDE_EVERY), args.per_day, replace=False)):
                        W = AG.AgentWorld(None, rules, L, prior, int(e), None, seed=int(rng.integers(1 << 30)))
                        W.L = None
                        AG.save_start(W, out / f"{want[d]}_{d.name}_{int(e)}.pkl")
                        n_done += 1
                    print(f"  {key} {want[d]} {d.parent.name}/{d.name}: {n_done} starts ({(time.time() - t0) / 60:.1f} min)", flush=True)
                prior.update_light(d, 5000)
        print(f"  {key}: {n_done} starts in {out}", flush=True)


if __name__ == "__main__":
    main()
