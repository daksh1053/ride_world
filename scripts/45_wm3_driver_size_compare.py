"""Stage 7b: driver policies of 0.6M, 30M and 60M weights, side by side.

Reads outputs/wm3/stage7b/eval_<tag>.csv (scripts/42) and plots the gain in net pay per hour of the
25% of drivers who use the agent, against the same drivers when everyone takes the first ride
(mean +- s.e. over 12 held-out starts), at rounds 0/10/20/30. Writes 45_size_compare.png and .csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "wm3" / "stage7b"
RUNS = [("drv2_d3", "0.6M (lr 3e-3)", "#2a6fdb"), ("drv2_30m", "30M (lr 3e-4 to round 10, then 1e-4)", "#e8590c"),
        ("drv2_60m", "60M (lr 7e-5 to round 10, then 2e-4)", "#2f9e44")]
CITY = {"sf_hd": "San Francisco", "pune_hd": "Pune"}

rows = []
for tag, name, _ in RUNS:
    d = pd.read_csv(OUT / f"eval_{tag}.csv")
    d = d[(d["mode"] == "mixed") & d.iter.isin([0, 10, 20, 30])]
    for (it, k), g in d.groupby(["iter", "key"]):
        x = g.focal_net_gain_per_h
        rows.append({"model": tag, "label": name, "city": k, "round": it, "gain": x.mean(), "se": x.std() / len(x) ** 0.5,
                     "wins": int((x > 0).sum()), "n": len(x), "others": g.others_net_gain_per_h.mean()})
T = pd.DataFrame(rows)
T.round(4).to_csv(OUT / "45_size_compare.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, k in zip(axes, CITY):
    for tag, name, c in RUNS:
        s = T[(T.model == tag) & (T.city == k)].sort_values("round")
        ax.errorbar(s["round"], s.gain, yerr=s.se, color=c, lw=2, marker="o", ms=6, capsize=3, label=name)
    # end labels, nudged apart when two runs finish close together
    ends = sorted(((T[(T.model == tag) & (T.city == k) & (T["round"] == 30)].gain.iloc[0], c) for tag, _, c in RUNS))
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    ys = [ends[0][0]]
    for g, _ in ends[1:]:
        ys.append(max(g, ys[-1] + 0.06 * span))
    for (g, c), y in zip(ends, ys):
        ax.annotate(f"{g:+.3f}", (30, g), xytext=(31, y), textcoords="data", va="center", fontsize=9, color="#333",
                    arrowprops=None)
    ax.axhline(0, color="#999", lw=1)
    ax.set_title(CITY[k])
    ax.set_xlabel("training round")
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlim(-1, 34)
    ax.grid(axis="y", color="#e5e5e5")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("net pay per hour gained vs first ride\n(25% of drivers on the agent)")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Driver policy size: 0.6M vs 30M vs 60M weights (mean ± s.e., 12 held-out starts)")
fig.tight_layout()
fig.savefig(OUT / "45_size_compare.png", dpi=130)
print(T.round(3).to_string(index=False))
