"""Reproduce Table 1 (baseline comparison) and Table 2 (ablation).

Also stores the per-horizon sweeps consumed by `make_figures.py` for Fig. 4 / Fig. 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ltf.config import CONFIG, DATA_PROC, RESULTS
from ltf.data import load, split_days
from ltf.experiment import build_ablations, build_methods, run_all
from ltf.graph import RoadGraph
from ltf.synthetic import predicted_to_days

# Table 1 as printed in the paper, for side-by-side reporting.
PAPER_TABLE1 = {
    "Greedy":               dict(total_utility=-1514736.24, fairness=1696.95,   normalised_fairness=-0.0005, min_utility=-75803.61, mean_utility=-75736.81, max_utility=-75653.78),
    "REASSIGN":             dict(total_utility=76536.23,    fairness=493637.57, normalised_fairness=0.18,    min_utility=2218.72,   mean_utility=3826.81,   max_utility=4760.17),
    "LAF":                  dict(total_utility=80606.49,    fairness=107789.96, normalised_fairness=0.0814,  min_utility=3001.74,   mean_utility=4030.3245, max_utility=4491.26),
    "Balance Ride-Pooling": dict(total_utility=85923.68,    fairness=100254.73, normalised_fairness=0.074,   min_utility=3451.47,   mean_utility=4296.18,   max_utility=4674.44),
    "Proposed Method":      dict(total_utility=95823.79,    fairness=85194.48,  normalised_fairness=0.061,   min_utility=4565.56,   mean_utility=4791.19,   max_utility=5931.93),
}

PAPER_TABLE2 = {
    "Our Method":                 dict(total_utility=95823.79,   fairness=85193.62),
    "Our Method w/o Prediction":  dict(total_utility=56873.21,   fairness=153697.27),
    "Our Method w/o Fairness":    dict(total_utility=2194901.19, fairness=2677473902.27),
}


def _load_predicted_days() -> list | None:
    """Materialise the forecast into request days for MOMAQL's action space."""
    path = DATA_PROC / "predicted_requests.npz"
    if not path.exists():
        print("[exp] no forecast found — run scripts/train_forecast.py first")
        return None
    predicted = np.load(path)["predicted"]
    days = predicted_to_days(predicted, pd.Timestamp(CONFIG.data.test_start_date))
    print(f"[exp] forecast materialised into {len(days)} synthetic request days "
          f"({sum(len(d) for d in days):,} requests)")
    return days


def _table(results: dict, cols: list[str]) -> pd.DataFrame:
    rows = {name: {c: r["final"][c] for c in cols} for name, r in results.items()}
    return pd.DataFrame(rows).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    bundle = load()
    graph = RoadGraph.from_processed(bundle)
    train_days, test_days = split_days(bundle["requests"])
    predicted = _load_predicted_days()
    print(f"[exp] {len(train_days)} training days, {len(test_days)} test days, "
          f"{CONFIG.sim.n_drivers} drivers, |L|={graph.n}")

    cols = ["total_utility", "fairness", "normalised_fairness",
            "min_utility", "mean_utility", "max_utility"]

    print("\n================ Table 1: comparison with baselines ================")
    main_results = run_all(graph, train_days, test_days, build_methods(graph), predicted)
    t1 = _table(main_results, cols)
    print("\n--- reproduction ---")
    print(t1.to_string(float_format=lambda x: f"{x:,.2f}"))
    print("\n--- paper (Table 1) ---")
    print(pd.DataFrame(PAPER_TABLE1).T[cols].to_string(float_format=lambda x: f"{x:,.2f}"))
    t1.to_csv(RESULTS / "table1.csv")

    all_results = {"table1": main_results}

    if not args.skip_ablation:
        print("\n================ Table 2: ablation study ================")
        abl_results = run_all(graph, train_days, test_days, build_ablations(graph), predicted)
        t2 = _table(abl_results, ["total_utility", "fairness"])
        print("\n--- reproduction ---")
        print(t2.to_string(float_format=lambda x: f"{x:,.2f}"))
        print("\n--- paper (Table 2) ---")
        print(pd.DataFrame(PAPER_TABLE2).T.to_string(float_format=lambda x: f"{x:,.2f}"))
        t2.to_csv(RESULTS / "table2.csv")
        all_results["table2"] = abl_results

    (RESULTS / "results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n[exp] wrote {RESULTS / 'results.json'}")


if __name__ == "__main__":
    main()
