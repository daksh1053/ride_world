"""Stage 5: generate world-model training data from the simulated world.

    .venv/bin/python scripts/05_wm_generate.py --city all --workers 12

For every city: 3 training days x 5 dispatch policies and 2 test days x 6 policies
(the 6th, P2 ILP, never appears in training: an unseen-policy test). Different
policies on the same date and seed see the same drivers, riders, requests and traffic,
so their outcomes are a true counterfactual pair.

Writes data/<city>/wm/<split>_<date>_<policy>.npz with obs (289, Z, 6), act (288, Z, 3),
ctx (289, 5), and data/<city>/wm/static.npz (zone features, neighbours).
Only observed tables are used; the latent (hidden) tables are never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from world import cities, metrics, wm_data
from world.paper_policies import policy_factories
from world.policies import NearestDispatch
from world.setup import CityData, build_world
from world.sim import SimConfig

TRAIN_DAYS = [("2025-02-11", 11), ("2025-06-14", 12), ("2025-08-20", 13)]   # Tue, Sat, Wed (monsoon)
TEST_DAYS = [("2025-10-08", 21), ("2025-11-15", 22)]                        # Wed, Sat
TRAIN_POLICIES = ["nearest", "nearest_norepo", "p2_nm", "p1_greedy", "p2_wdf"]
TEST_POLICIES = TRAIN_POLICIES + ["p2_ilp"]


def run_day(key: str, split: str, date: str, seed: int) -> list[dict]:
    city = cities.get(key)
    data = CityData(city)
    out = city.data_dir / "wm"
    out.mkdir(exist_ok=True)
    cfg = SimConfig(date=date, seed=seed)
    fac, _, _ = policy_factories(data, cfg)
    fac["nearest_norepo"] = lambda: NearestDispatch(reposition_every=10 ** 9)
    weekend = pd.Timestamp(date).weekday() >= 5
    rows = []
    for pol in (TRAIN_POLICIES if split == "train" else TEST_POLICIES):
        f = out / f"{split}_{date}_{pol}.npz"
        if f.exists():
            continue
        t0 = time.time()
        w = build_world(data, cfg, fac[pol](), log=lambda *a: None)
        w.run(log=lambda *a: None)
        tb = w.tables()
        d = wm_data.build_steps(tb, len(data.zones), weekend)
        np.savez_compressed(f, **d)
        s = metrics.service_summary(tb)
        rows.append({"city": key, "split": split, "date": date, "policy": pol,
                     "completion": s["completion_rate"], "requests": s["requests"],
                     "seconds": round(time.time() - t0, 1)})
    st = out / "static.npz"
    if not st.exists():
        np.savez(st, **wm_data.zone_static(city, data.zones))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=["all"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    keys = list(cities.CITIES) if args.city == ["all"] else args.city
    jobs = [(k, "train", d, s) for k in keys for d, s in TRAIN_DAYS] + \
           [(k, "test", d, s) for k in keys for d, s in TEST_DAYS]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_day, *j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"  FAILED {j}: {type(e).__name__}: {e}", flush=True)
                continue
            rows += r
            print(f"  {j[0]:>11} {j[1]:5} {j[2]}  {len(r)} runs  "
                  f"completion {np.mean([x['completion'] for x in r]) if r else float('nan'):.3f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    log = cities.DATA / "wm_runs.csv"
    old = pd.read_csv(log) if log.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(log, index=False)
    print(json.dumps({"runs": len(rows), "minutes": round((time.time() - t0) / 60, 1)}))


if __name__ == "__main__":
    main()
