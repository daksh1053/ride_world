"""Reproduce Fig. 5 — the five metrics over the vehicle-count sweep (Sec. VI-C)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from vfd.config import CONFIG, RESULTS
from vfd.data import active_time_weights, load, sample_day
from vfd.experiment import run_sweep
from vfd.graph import ZoneGraph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order-scale", type=float, default=None,
                    help="multiplier on the daily order count (see NOTES.md)")
    ap.add_argument("--vehicles", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="fig5_sweep.json")
    args = ap.parse_args()

    cfg = CONFIG
    if args.order_scale is not None:
        cfg = dataclasses.replace(
            cfg, data=dataclasses.replace(cfg.data, order_scale=args.order_scale)
        )
    counts = tuple(args.vehicles) if args.vehicles else cfg.platform.vehicle_counts

    bundle = load()
    graph = ZoneGraph.from_processed(bundle)
    day = sample_day(bundle, cfg.data.seed, cfg.data.order_scale)
    xi = active_time_weights(bundle, cfg.platform.n_slots)

    print(f"[exp] |L|={graph.n} zones, {len(day):,} orders "
          f"(scale {cfg.data.order_scale}), vehicles {counts}")

    results = run_sweep(graph, day, xi, counts, cfg, cfg.data.seed)

    payload = {
        "order_scale": cfg.data.order_scale,
        "n_orders": len(day),
        "vehicle_counts": list(counts),
        "results": results,
    }
    (RESULTS / args.out).write_text(json.dumps(payload, indent=2))

    for metric in ["unfairness", "total_income", "order_service_rate",
                   "idle_driver_rate", "running_time"]:
        table = pd.DataFrame(
            {name: {n: v[metric] for n, v in per_n.items()} for name, per_n in results.items()}
        )
        print(f"\n--- {metric} ---")
        print(table.to_string(float_format=lambda x: f"{x:,.4f}"))
        table.to_csv(RESULTS / f"fig5_{metric}.csv")

    print(f"\n[exp] wrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
