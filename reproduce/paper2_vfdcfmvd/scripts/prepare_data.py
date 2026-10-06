"""Download and preprocess the NYC TLC order data (paper Sec. VI-A)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from vfd.config import CONFIG
from vfd.data import active_time_weights, prepare, sample_day
from vfd.graph import ZoneGraph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    bundle = prepare(CONFIG.data, force=args.force)
    graph = ZoneGraph.from_processed(bundle)
    day = sample_day(bundle, CONFIG.data.seed)
    xi = active_time_weights(bundle, CONFIG.platform.n_slots)

    print("\n=== dataset summary ===")
    print(f"zones |L|                : {graph.n}")
    print(f"observed directed edges  : {int(np.isfinite(bundle['weights']).sum())} / {graph.n ** 2}")
    print(f"mean zone-to-zone dist   : {graph.matrix.mean():.2f} km")
    print(f"orders in pool           : {len(bundle['orders']):,}")
    print(f"mean orders per day      : {bundle['mean_orders_per_day']:,.0f}")
    print(f"sampled day              : {len(day):,} orders")
    print(f"mean fare                : ${day.fare_amount.mean():.2f}")
    print(f"mean trip distance       : {day.distance_km.mean():.2f} km")
    print(f"unit costs c_d ($/km)    : {[round(c, 4) for c in CONFIG.platform.unit_costs]}")
    print(f"average speed            : {CONFIG.platform.v_avg_kmh:.2f} km/h")
    print(f"active-time weights xi^t : min {xi.min():.3f}  mean {xi.mean():.3f}  max {xi.max():.3f}")


if __name__ == "__main__":
    main()
