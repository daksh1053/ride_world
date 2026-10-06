"""Reproduce Table III (module ablation) and Table IV (low-income threshold sweep).

Both are run "under the medium-scale vehicle setting with 2500 vehicles" (Sec. VI-D).
The paper reports mean +- std over repeated runs, so each configuration is repeated
across several random seeds here too.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from vfd.config import CONFIG, RESULTS
from vfd.data import active_time_weights, load, sample_day
from vfd.experiment import build_ablations, evaluate_algorithm, train
from vfd.graph import ZoneGraph
from vfd.methods import VFDCFMVD

PAPER_TABLE3 = {
    "VFDCFMVD":                     dict(unfairness=3902.63, order_service_rate=0.7740, idle_driver_rate=0.0594, total_income=173157.30),
    "w/o dynamic clustering":       dict(unfairness=4852.91, order_service_rate=0.7486, idle_driver_rate=0.0883, total_income=166842.57),
    "w/o driver-order matching":    dict(unfairness=5887.34, order_service_rate=0.6974, idle_driver_rate=0.1264, total_income=144728.36),
    "w/o idle vehicle dispatching": dict(unfairness=5426.76, order_service_rate=0.7158, idle_driver_rate=0.1827, total_income=150436.82),
}

PAPER_TABLE4 = {
    0.10: dict(unfairness=4032.24, total_income=176268.45),
    0.20: dict(unfairness=3956.83, total_income=175584.91),
    0.30: dict(unfairness=3902.63, total_income=173157.30),
    0.40: dict(unfairness=3872.34, total_income=172632.22),
    0.50: dict(unfairness=3853.85, total_income=172264.85),
}


def _repeat(make, graph, day, xi, n_drivers, cfg, seeds) -> dict:
    """Run one configuration across seeds; return mean and std per metric."""
    rows = []
    for seed in seeds:
        algorithm = make()
        train(algorithm, graph, day, n_drivers, cfg, seed)
        rows.append(evaluate_algorithm(algorithm, graph, day, xi, n_drivers, cfg, seed).as_dict())
    frame = pd.DataFrame(rows)
    return {
        "mean": frame.mean(numeric_only=True).to_dict(),
        "std": frame.std(numeric_only=True, ddof=0).to_dict(),
    }


def _fmt(stats: dict, metric: str, pct: bool = False) -> str:
    m, s = stats["mean"][metric], stats["std"][metric]
    if pct:
        return f"{m * 100:.2f} +- {s * 100:.2f}"
    return f"{m:,.2f} +- {s:,.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order-scale", type=float, default=None)
    ap.add_argument("--vehicles", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-threshold", action="store_true")
    args = ap.parse_args()

    cfg = CONFIG
    if args.order_scale is not None:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, order_scale=args.order_scale))
    n_drivers = args.vehicles or cfg.platform.default_vehicles
    seeds = list(range(args.seeds))

    bundle = load()
    graph = ZoneGraph.from_processed(bundle)
    day = sample_day(bundle, cfg.data.seed, cfg.data.order_scale)
    xi = active_time_weights(bundle, cfg.platform.n_slots)
    print(f"[abl] {len(day):,} orders, {n_drivers} vehicles, {len(seeds)} seeds")

    payload: dict = {"n_orders": len(day), "n_drivers": n_drivers, "seeds": seeds}

    # ------------------------------------------------------------- Table III
    print("\n================ Table III: module ablation ================")
    table3 = {}
    for name, prototype in build_ablations(graph, cfg).items():
        kwargs = dict(
            use_clustering=prototype.use_clustering,
            use_value_matching=prototype.use_value_matching,
            use_dispatching=prototype.use_dispatching,
        )
        stats = _repeat(lambda: VFDCFMVD(graph.n, cfg.algorithm, **kwargs),
                        graph, day, xi, n_drivers, cfg, seeds)
        table3[name] = stats
        print(f"  {name:30s} unfair={_fmt(stats,'unfairness'):>22s}"
              f"  service={_fmt(stats,'order_service_rate',True):>16s}"
              f"  idle={_fmt(stats,'idle_driver_rate',True):>16s}"
              f"  income={_fmt(stats,'total_income'):>24s}")
    payload["table3"] = table3

    print("\n--- reproduction ---")
    print(pd.DataFrame({
        n: {
            "Unfairness": _fmt(s, "unfairness"),
            "Order Service Rate": _fmt(s, "order_service_rate", True),
            "Idle Driver Rate": _fmt(s, "idle_driver_rate", True),
            "Drivers' Total Income": _fmt(s, "total_income"),
        } for n, s in table3.items()
    }).T.to_string())
    print("\n--- paper (Table III) ---")
    print(pd.DataFrame(PAPER_TABLE3).T.to_string(float_format=lambda x: f"{x:,.4f}"))

    # -------------------------------------------------------------- Table IV
    if not args.skip_threshold:
        print("\n================ Table IV: low-income threshold ================")
        table4 = {}
        for threshold in PAPER_TABLE4:
            stats = _repeat(
                lambda t=threshold: VFDCFMVD(graph.n, cfg.algorithm, low_income_threshold=t),
                graph, day, xi, n_drivers, cfg, seeds,
            )
            table4[threshold] = stats
            print(f"  threshold={threshold:.0%}  unfair={_fmt(stats,'unfairness'):>22s}"
                  f"  income={_fmt(stats,'total_income'):>24s}")
        payload["table4"] = {str(k): v for k, v in table4.items()}

        print("\n--- reproduction ---")
        print(pd.DataFrame({
            f"{k:.0%}": {"Unfairness": _fmt(v, "unfairness"),
                         "Drivers' Total Income": _fmt(v, "total_income")}
            for k, v in table4.items()
        }).T.to_string())
        print("\n--- paper (Table IV) ---")
        print(pd.DataFrame(PAPER_TABLE4).T.to_string(float_format=lambda x: f"{x:,.2f}"))

    (RESULTS / "ablation.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[abl] wrote {RESULTS / 'ablation.json'}")


if __name__ == "__main__":
    main()
