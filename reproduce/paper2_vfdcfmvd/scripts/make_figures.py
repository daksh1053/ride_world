"""Reproduce Fig. 5 (a)-(e): the five metrics against the number of vehicles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from vfd.config import FIGURES, RESULTS
from vfd.plotting import SERIES_STYLE, finish, new_axes

PANELS = [
    ("unfairness", "Fig. 5(a) — Unfairness", "Unfairness  F", 1.0, "fig5a_unfairness.png"),
    ("order_service_rate", "Fig. 5(b) — Order service rate", "Order service rate (%)", 100.0, "fig5b_service_rate.png"),
    ("idle_driver_rate", "Fig. 5(c) — Idle driver rate", "Idle driver rate (%)", 100.0, "fig5c_idle_rate.png"),
    ("total_income", "Fig. 5(d) — Drivers' total income", "Drivers' total income ($)", 1.0, "fig5d_total_income.png"),
    ("running_time", "Fig. 5(e) — Computational time", "Running time (s)", 1.0, "fig5e_running_time.png"),
]


def main() -> None:
    path = RESULTS / "fig5_sweep.json"
    if not path.exists():
        raise SystemExit("run scripts/run_experiments.py first")
    payload = json.loads(path.read_text())
    results = payload["results"]
    counts = payload["vehicle_counts"]

    for metric, title, ylabel, scale, filename in PANELS:
        fig, ax = new_axes()
        for name, per_n in results.items():
            colour, marker = SERIES_STYLE.get(name, ("#52514e", "o"))
            xs = [n for n in counts if str(n) in per_n or n in per_n]
            ys = [per_n.get(str(n), per_n.get(n))[metric] * scale for n in xs]
            ax.plot(xs, ys, color=colour, marker=marker, markersize=6, linewidth=2.0,
                    markeredgecolor="white", markeredgewidth=0.8, label=name, zorder=3)
        ax.set_xticks(counts)
        finish(fig, ax, title, "Number of vehicles  |D|", ylabel, FIGURES / filename)

    print(f"[fig] {len(PANELS)} panels written to {FIGURES}")


if __name__ == "__main__":
    main()
