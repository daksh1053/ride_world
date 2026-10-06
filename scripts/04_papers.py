"""Stage 4: run the reproduced papers' methods as the ride company's dispatch policy.

    .venv/bin/python scripts/04_papers.py --city sf pune               # all methods
    .venv/bin/python scripts/04_papers.py --city sf --methods nearest p2_vfdcfmvd --hours 6

Protocol (same for every method, per city):
  * learning methods first run `--warmup` training days (different dates and seeds,
    learning on; MOMAQL/DQN exploration on), then
  * every method is evaluated on the same test day (`--date`, `--seed`): same
    drivers, customers, requests and traffic draw (common random numbers), with
    learning in each paper's evaluation mode (paper 1: frozen; paper 2: V(s) keeps
    learning online, DQN greedy).

Outputs
    data/<city>/sim/stage4/<date>_seed<k>_<method>/   all tables + metrics.json
    outputs/<city>/04_replay_<method>.html              animated replay per method (--replay window)
    outputs/<city>/04_results.csv, outputs/04_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import geopandas as gpd

from world import cities, metrics, replay, roadnet
from world.cities import OUTPUTS
from world.paper_policies import LEARNERS, policy_factories
from world.setup import CityData, build_world
from world.sim import SimConfig

WARMUP_DATES = ["2025-02-25", "2025-02-26", "2025-02-27"]
PAPER = {"nearest": "baseline"}


def run_job(city_key: str, method: str, date: str, seed: int, warmup: int, hours: float,
            replay_h=(17.0, 20.0)) -> dict:
    city = cities.get(city_key)
    data = CityData(city)
    quiet = lambda *a, **k: None  # noqa: E731
    cfg = SimConfig(date=date, seed=seed, horizon_s=hours * 3600)
    fac, xi_hour, v_avg = policy_factories(data, cfg)
    policy = fac[method]()
    t0 = time.time()
    if method in LEARNERS:
        policy.set_mode(True)
        for k in range(warmup):
            if hasattr(policy, "new_day"):
                policy.new_day()
            w = build_world(data, SimConfig(date=WARMUP_DATES[k % len(WARMUP_DATES)], seed=101 + k,
                                            horizon_s=hours * 3600), policy, log=quiet)
            w.run(log=quiet)
        policy.set_mode(False)
    t_train = time.time() - t0
    if hasattr(policy, "new_day"):
        policy.new_day()
    world = build_world(data, cfg, policy, log=quiet)
    spent = [0.0]
    act = policy.act

    def timed(obs):
        s = time.perf_counter()
        out = act(obs)
        spent[0] += time.perf_counter() - s
        return out
    policy.act = timed
    t1 = time.time()
    world.run(log=quiet)
    t_test = time.time() - t1

    tb = world.tables()
    # stage-4 runs live in their own folder so they never overwrite stage-3 runs
    run_dir = city.data_dir / "sim" / "stage4" / f"{date}_seed{seed}_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, df in {**tb, "drivers": world.drv}.items():
        df.to_parquet(run_dir / f"{name}.parquet", index=False)
    dout = metrics.driver_outcomes(tb)
    cohort = dout.driver_id.to_numpy()
    res = {"city": city_key, "method": method,
           **metrics.service_summary(tb), **metrics.fairness(dout),
           **metrics.paper1_metrics(dout), **metrics.paper2_metrics(tb, cohort, xi_hour),
           "policy_seconds": spent[0], "sim_seconds": t_test, "train_seconds": t_train,
           "warmup_days": warmup if method in LEARNERS else 0, "v_avg_kmh": v_avg}
    res.pop("cancel_reasons", None)
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    zone_geoms = gpd.read_file(city.data_dir / "zone_geoms.geojson")
    replay.build_replay(city, data.nodes, zone_geoms, tb, world.drv, replay_h[0], replay_h[1],
                        city.out_dir / f"04_replay_{method}.html",
                        res, roadnet.edge_shapes(city, data.nodes),
                        title=f"{city.name}: {method} dispatch replay")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", nargs="+", default=["sf", "pune"], choices=sorted(cities.CITIES))
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--date", default="2025-03-04")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--hours", type=float, default=24.0, help="simulated hours per day (debugging)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--replay", nargs=2, type=float, default=[17.0, 20.0], metavar=("FROM_H", "TO_H"))
    args = ap.parse_args()
    all_methods = ["nearest", "p1_greedy", "p1_reassign", "p1_laf", "p1_brp", "p1_momaql",
                   "p2_nm", "p2_wdf", "p2_laf", "p2_ilp", "p2_sid", "p2_vfdcfmvd"]
    methods = args.methods or all_methods
    jobs = [(c, m) for c in args.city for m in methods]
    # learners first: they take longest
    jobs.sort(key=lambda j: j[1] not in LEARNERS)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_job, c, m, args.date, args.seed, args.warmup, args.hours, tuple(args.replay)): (c, m)
                for c, m in jobs}
        for fut in as_completed(futs):
            c, m = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # keep the other jobs going; report the failure
                print(f"  FAILED {c:>4} {m:<12} {type(e).__name__}: {e}")
                continue
            rows.append(r)
            print(f"  {c:>4} {m:<12} done  completion {r['completion_rate']:.3f}  wait {r['mean_wait_min']:.2f} min  "
                  f"gini {r['gini_net_earnings']:.3f}  p1 F_hat {r['p1_F_hat']:.3f}  p2 unfair/driver "
                  f"{r['p2_unfairness_per_driver']:.3f}  policy {r['policy_seconds']:.0f}s  [{time.time() - t0:.0f}s]")
    if not rows:
        sys.exit("no job finished")
    res = pd.DataFrame(rows)
    out = OUTPUTS / "04_results.csv"
    if out.exists():                           # merge: keep rows for other cities / methods
        old = pd.read_csv(out)
        keep = ~old.set_index(["city", "method"]).index.isin(res.set_index(["city", "method"]).index)
        res = pd.concat([old[keep], res], ignore_index=True)
    res.to_csv(out, index=False)
    for c in args.city:
        res[res.city == c].to_csv(cities.get(c).out_dir / "04_results.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
