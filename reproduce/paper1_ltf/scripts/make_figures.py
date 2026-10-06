"""Reproduce Fig. 4 (baseline fairness vs. time horizon) and Fig. 5 (ablation).

Both figures plot the long-term fairness `F(M) = Var(o_v)` on a log y-axis against
the length of the time horizon in days; a larger value means the method is unfairer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ltf.config import FIGURES, RESULTS
from ltf.plotting import SERIES_STYLE, finish, new_axes


def _scatter(results: dict, title: str, out_png: Path, out_csv: Path, metric: str = "fairness"):
    fig, ax = new_axes()
    table = {}
    for name, res in results.items():
        values = [h[metric] for h in res["horizon"]]
        days = np.arange(1, len(values) + 1)
        colour, marker = SERIES_STYLE.get(name, ("#52514e", "o"))
        # Fairness can be plotted on a log axis only where it is positive.
        vals = np.array(values, dtype=float)
        ax.scatter(days, np.abs(vals), s=42, color=colour, marker=marker,
                   edgecolor="white", linewidth=0.8, label=name, zorder=3)
        table[name] = dict(zip(days.tolist(), vals.tolist()))

    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, max(len(r["horizon"]) for r in results.values()) + 1))
    finish(fig, ax, title, "Length of Time Horizon (days)", "Fairness  |Var(o_v)|", out_png)

    # Table view — the relief required by the contrast check, and the numbers behind
    # every marker.
    pd.DataFrame(table).T.to_csv(out_csv)
    print(f"[fig] wrote {out_csv}")


def main() -> None:
    results = json.loads((RESULTS / "results.json").read_text())

    _scatter(
        results["table1"],
        "Fig. 4 — Fairness vs. length of time horizon",
        FIGURES / "fig4_fairness_vs_horizon.png",
        RESULTS / "fig4_fairness_vs_horizon.csv",
    )

    if "table2" in results:
        _scatter(
            results["table2"],
            "Fig. 5 — Ablation: fairness vs. length of time horizon",
            FIGURES / "fig5_ablation_fairness.png",
            RESULTS / "fig5_ablation_fairness.csv",
        )

    # Supplementary: the efficiency/fairness trade-off behind Table 1.
    fig, ax = new_axes()
    for name, res in results["table1"].items():
        colour, marker = SERIES_STYLE.get(name, ("#52514e", "o"))
        final = res["final"]
        ax.scatter(final["total_utility"], abs(final["fairness"]), s=90, color=colour,
                   marker=marker, edgecolor="white", linewidth=1.0, label=name, zorder=3)
    ax.set_yscale("log")
    ax.set_xscale("symlog")
    finish(
        fig, ax,
        "Efficiency vs. fairness (full seven-day horizon)",
        "Total utility  pi(M)", "Fairness  |Var(o_v)|",
        FIGURES / "tradeoff_utility_vs_fairness.png",
    )


if __name__ == "__main__":
    main()
