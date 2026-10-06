"""Stage 6c: figures for the world-model experiments (reads outputs/wm/*.csv).

    .venv/bin/python scripts/06c_wm_figures.py

Writes outputs/wm/fig_*.png and outputs/wm/results.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from world import cities, viz

OUT = cities.OUTPUTS / "wm"
COL = {"persistence": "#898781", "seasonal": "#c3c2b7", "glm": "#eda100", "mlp": "#e87ba4",
       "gru": "#2a78d6", "gru_os": "#1baf7a", "gru_cons": "#4a3aa7", "gru_cons_os": "#eb6834"}
LAB = {"persistence": "persistence", "seasonal": "time-of-day average", "glm": "Poisson GLM",
       "mlp": "MLP (Markov)", "gru": "GRU (recurrent state)", "gru_os": "GRU + overshooting",
       "gru_cons": "fleet-conserving GRU", "gru_cons_os": "fleet-conserving GRU + overshooting"}
TIER_COL = {"core": "#0b0b0b", "metro": "#4a3aa7", "medium": "#1baf7a", "small": "#eda100"}


def style(ax):
    ax.grid(color=viz.HAIRLINE, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_rollouts():
    d = pd.read_csv(OUT / "e2_rollouts.csv")
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), layout="constrained")
    for m, g in d.groupby("model"):
        g = g.sort_values("h")
        kw = dict(color=COL[m], lw=2.6 if m == "gru_cons_os" else 1.6,
                  ls="--" if m in ("persistence", "seasonal") else "-", label=LAB[m])
        axs[0].plot(g.h * 5, g.nll, **kw)
        axs[1].plot(g.h * 5, g.mae_idle, **kw)
    axs[0].set_title("Prediction error of zone counts, by rollout horizon")
    axs[0].set_ylabel("Poisson NLL per zone-count (lower is better)")
    axs[1].set_title("Error in idle cabs per zone, by horizon (GLM off the chart)")
    axs[1].set_ylim(0, float(d[~d.model.isin(["glm"])].mae_idle.max()) * 1.15)
    axs[1].set_ylabel("mean absolute error (cabs)")
    for a in axs[:2]:
        a.set_xlabel("minutes into the imagined future")
        a.set_xticks([5, 15, 30, 45, 60])
        style(a)
    glm_nll = d[(d.model == "glm") & (d.h == 12)].nll
    top = max(1.6, float(d[(d.model != "glm")].nll.max()) * 1.1)
    axs[0].set_ylim(0, top)
    if len(glm_nll) and float(glm_nll.iloc[0]) > top:
        axs[0].text(60, top * 0.97, f"GLM off the chart ({float(glm_nll.iloc[0]):.1f})", ha="right", va="top",
                    fontsize=9, color=COL["glm"])
    div = d.drop_duplicates("model").set_index("model").diverged.reindex(list(COL)).dropna()
    axs[2].barh([LAB[m] for m in div.index], div.values * 100, color=[COL[m] for m in div.index])
    for i, v in enumerate(div.values):
        axs[2].text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=9)
    axs[2].set_title("Rollouts whose fleet drifts beyond 2x or 0.5x")
    axs[2].set_xlabel("share of 1-hour rollouts (%)")
    axs[2].invert_yaxis()
    style(axs[2])
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Persistence: how far ahead can each model imagine? (held-out days, 15 cities, 21 start hours)",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, OUT / "fig_rollouts.png")


def fig_cities():
    d = pd.read_csv(OUT / "e1_one_step_nll.csv")
    piv = d.groupby(["city", "model"]).nll.mean().unstack()
    order = [k for t in cities.TIERS.values() for k in t if k in piv.index]
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    y = np.arange(len(order))
    for m in ["persistence", "seasonal", "mlp", "gru", "gru_cons_os"]:
        if m in piv:
            ax.scatter(piv.loc[order, m], y, color=COL[m], s=60 if m == "gru_cons_os" else 36, label=LAB[m],
                       zorder=3, edgecolor=viz.SURFACE, lw=0.8)
    ax.set_yticks(y, [f"{cities.CITIES[k].name}  ({cities.CITIES[k].tier})" for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("one-step Poisson NLL per zone-count, held-out days (lower is better)")
    ax.set_title("Accuracy per city: one world model serves all 15 cities", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    viz.save(fig, OUT / "fig_cities.png")


def fig_interactability():
    d = pd.read_csv(OUT / "e3_policy_gaps.csv")
    s = json.loads((OUT / "summary.json").read_text())["e3"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), layout="constrained", sharex=True, sharey=True)
    lim = np.percentile(np.abs(d.actual), 99) * 1.1
    for ax, col, title, c in [(axs[0], "predicted", f"World model with actions: r = {s['policy_gap_corr']:.2f}", COL["gru_cons_os"]),
                              (axs[1], "noact_control", f"Same model without action inputs: r = {s['policy_gap_corr_noact_control']:.2f}", COL["persistence"])]:
        ax.scatter(d.actual, d[col], s=6, alpha=0.35, color=c, lw=0)
        ax.plot([-lim, lim], [-lim, lim], color=viz.INK, lw=0.8, ls="--")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(title, loc="left")
        ax.set_xlabel("actual difference between two policies (city pickups or cancellations, next hour)")
        style(ax)
    axs[0].set_ylabel("predicted difference")
    fig.suptitle("Interactability: does the model know what different dispatch decisions do? "
                 f"({s['policy_gap_n']:,} policy pairs x hours, same day, held out)", x=0.01, ha="left", fontweight="bold")
    viz.save(fig, OUT / "fig_interactability.png")


def fig_fairness():
    d = pd.read_csv(OUT / "e6_fairness_obs.csv")
    d = d[d.model == "gru_cons_os"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for _, r in d.iterrows():
        c = TIER_COL[cities.CITIES[r.city].tier]
        axs[0].scatter(r.city_s_actual, r.city_s_pred, color=c, s=24, alpha=0.8, lw=0)
        axs[1].scatter(r.s_std_actual, r.s_std_pred, color=c, s=24, alpha=0.8, lw=0)
    for ax, lo, hi in [(axs[0], 0.4, 1.0), (axs[1], 0, max(d.s_std_actual.max(), d.s_std_pred.max()) * 1.05)]:
        ax.plot([lo, hi], [lo, hi], color=viz.INK, lw=0.8, ls="--")
        style(ax)
    axs[0].set_title("City completion rate, 17:00-20:00", loc="left")
    axs[0].set_xlabel("actual"); axs[0].set_ylabel("imagined by the world model (3-h rollout)")
    axs[1].set_title("Inequality between zones: spread (std) of zone completion s_z", loc="left")
    axs[1].set_xlabel("actual spread across zones"); axs[1].set_ylabel("imagined spread")
    handles = [Line2D([], [], marker="o", ls="", color=v, label=f"{k} cities") for k, v in TIER_COL.items()]
    axs[1].legend(handles=handles, frameon=False, fontsize=9)
    fig.suptitle("Fairness observables Phi computed on imagined trajectories (eq. 25, 32), one dot per held-out day x policy",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, OUT / "fig_fairness.png")


def results_md():
    s = json.loads((OUT / "summary.json").read_text())
    e1 = pd.read_csv(OUT / "e1_one_step_nll.csv").groupby("model").nll.mean().sort_values()
    e2 = pd.read_csv(OUT / "e2_rollouts.csv")
    e4 = pd.read_csv(OUT / "e4_generalization.csv")
    e5 = pd.read_csv(OUT / "e5_plausibility.csv")
    e6 = pd.read_csv(OUT / "e6_fairness_obs.csv").groupby("model")[["s_mae", "s_corr", "s_std_actual", "s_std_pred"]].mean()
    t2 = e2[e2.h.isin([1, 6, 12])].pivot_table(index="model", columns="h", values="nll").round(3)
    t2["diverged"] = e2.drop_duplicates("model").set_index("model").diverged
    md = ["# World-model experiments\n", f"{s['n_train_runs']} training days, {s['n_test_runs']} held-out days, "
          f"{len(s['cities'])} cities.\n", "\n## E1 one-step NLL (held-out, seen policies)\n", e1.round(4).to_markdown(),
          "\n\n## E2 rollout NLL at 5 / 30 / 60 min, and share of diverging rollouts\n", t2.to_markdown(),
          "\n\n## E3 interactability\n", "```json\n" + json.dumps(s["e3"], indent=1) + "\n```",
          "\n\n## E4 generalization to unseen cities\n", e4.round(4).to_markdown(index=False),
          "\n\n## E5 plausibility (3-h evening rollouts)\n", e5.round(4).to_markdown(index=False),
          "\n\n## E6 fairness observable: zone completion s_z over 3-h rollouts\n", e6.round(4).to_markdown(), "\n"]
    (OUT / "results.md").write_text("\n".join(md))


if __name__ == "__main__":
    fig_rollouts()
    fig_cities()
    fig_interactability()
    fig_fairness()
    results_md()
