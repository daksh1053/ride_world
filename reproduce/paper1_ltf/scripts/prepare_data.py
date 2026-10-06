"""Download and preprocess the NYC TLC data (paper Sec. 5.1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ltf.config import CONFIG
from ltf.data import prepare, split_days
from ltf.graph import RoadGraph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    bundle = prepare(CONFIG.data, force=args.force)
    graph = RoadGraph.from_processed(bundle)
    train_days, test_days = split_days(bundle["requests"])

    observed = np.isfinite(bundle["weights"]).sum()
    print("\n=== dataset summary ===")
    print(f"nodes |L| (Manhattan zones): {graph.n}")
    print(f"observed directed edges  : {observed} / {graph.n ** 2}")
    print(f"mean shortest distance   : {graph.matrix.mean():.3f}")
    print(f"peak-window requests     : {len(bundle['requests']):,}")
    print(f"training days            : {len(train_days)}")
    print(f"test days                : {len(test_days)}")
    if test_days:
        print(f"requests per test day    : {[len(d) for d in test_days]}")
    print(f"hourly OD series shape   : {bundle['hourly_counts'].shape}")


if __name__ == "__main__":
    main()
