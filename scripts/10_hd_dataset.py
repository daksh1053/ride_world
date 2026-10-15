"""Stage 10: world model v2 dataset for the high-detail cities.

    .venv/bin/python scripts/10_hd_dataset.py --city sf_hd --days 120 --start 2025-01-06
    .venv/bin/python scripts/10_hd_dataset.py --city pune_hd --days 100 --start 2025-05-05

Runs one long season (people carried across days) with the dispatcher's settings drawn
afresh each day, so the data shows the effect of different actions:
    reposition_every  0 (never), 4, 10 or 20 epochs of 30 s
    max_moves         20, 60 or 150 instructions per rebalancing
    explore           0, 0.05, 0.15 or 0.3 (random moves of long-idle drivers)
With --blocks (step 4) the platform also issues random concentrated orders during the day,
10 to 60 (SF) or 120 (Pune) drivers from one area to one zone, about every 40 minutes.
Each day's tables go to data/<city>/hd_runs/<name>/<date>/ and its raster (world/hd/raster.py)
to data/<city>/wm2/<date>.npz. Also writes data/<city>/wm2/static.npz (static layers, grid)
and data/<city>/wm2/days_seed<seed>.csv (date, policy settings, day summary). Several seasons
(different seeds and dates) can run in parallel into the same dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from world import cities
from world.hd import raster as R
from world.hd.run import Season
from world.hd.sim import HDDispatch, HDDispatchBlocks

GRID = {"reposition_every": [0, 4, 10, 20], "max_moves": [20, 60, 150], "explore": [0.0, 0.05, 0.15, 0.3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["sf_hd", "pune_hd"])
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--start", default="2025-01-06")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--blocks", action="store_true",
                    help="step 4: add random concentrated orders (HDDispatchBlocks) for action coverage")
    args = ap.parse_args()
    city = cities.get(args.city)
    name = f"wm2_seed{args.seed}"
    run_dir = city.data_dir / "hd_runs" / name
    out = city.data_dir / "wm2"
    out.mkdir(parents=True, exist_ok=True)
    cr = R.CityRaster(city)
    np.savez_compressed(out / "static.npz", S=cr.S, cell_zone=cr.cell_zone.reshape(R.G, R.G),
                        grid=np.array([cr.grid.x0, cr.grid.y0, cr.grid.cell_m]), static=np.array(R.STATIC),
                        channels=np.array(R.CHANNELS), context=np.array(R.CONTEXT))
    rng = np.random.default_rng(args.seed)
    params = {}

    def factory():
        p = {k: v[rng.integers(len(v))] for k, v in GRID.items()}
        params.update(p)
        if args.blocks:
            params["blocks"] = 1.0
            return HDDispatchBlocks(n_max={"sf_hd": 60, "pune_hd": 120}[args.city], seed=int(rng.integers(1 << 30)), **p)
        return HDDispatch(seed=int(rng.integers(1 << 30)), **p)

    season = Season(args.city, seed=args.seed, policy_factory=factory)
    rows = []
    for date in pd.date_range(args.start, periods=args.days):
        season.run_day(date, out_dir=run_dir)
        X, C, Y, meta = cr.day(run_dir / str(date.date()), seed=args.seed)
        meta.update({k: float(v) for k, v in params.items()})
        R.save_day(out / f"{date.date()}.npz", X, C, Y, meta)
        for f in ("ledger", "ratings", "trips"):          # not needed for v2; saves disk
            (run_dir / str(date.date()) / f"{f}.parquet").unlink(missing_ok=True)
        rows.append({**season.summary[-1], **params})
        pd.DataFrame(rows).to_csv(out / f"days_seed{args.seed}.csv", index=False)
        print(f"    raster {date.date()}  policy {json.dumps({k: float(v) for k, v in params.items()})}", flush=True)


if __name__ == "__main__":
    main()
