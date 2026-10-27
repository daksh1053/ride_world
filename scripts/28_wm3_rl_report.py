"""World model 3, stage 5d: figures and tables for the agents trained with GRPO.

    .venv/bin/python scripts/28_wm3_rl_report.py --tags rl_pre rl_scratch

Reads outputs/wm3/stage5/train_<tag>.csv, eval_<tag>.csv (scripts/26_wm3_grpo.py) and
data/<city>/wm3_rl/exec/rec_seed*.json (scripts/27_wm3_rl_execute.py). Writes to outputs/wm3/stage5/:
    28_training.png     during training, on held-out validation states, in imagination: reward,
                        riders picked up, drivers' welfare and drivers moved per hour, agent minus no
                        repositioning, next to the platform's rule; and the world model's own
                        prediction loss (does the agent stay a world model?)
    28_versions.png     every version of the training method (outputs/wm3/stage5/v*/) and the probe of
                        how noisy one driver's effect is at each horizon (scripts/29_wm3_rl_probe.py)
    28_execute.png      in the simulator (fresh seasons, test days): each platform minus no
                        repositioning over the 2-hour window, real (bars) and imagined (dots)
    28_scatter.png      every seed and platform: the world model's predicted effect vs the real one
    28_execute.csv      per seed and platform, real and imagined numbers
    28_summary.json     the means and standard errors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from world import cities

OUT = cities.OUTPUTS / "wm3" / "stage5"
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"none": "#9a9893", "rule": "#2a78d6", "rl_pre": "#eb6834", "rl_scratch": "#1baf7a", "untrained": "#8a5cd6",
       "bc_pre": "#f2a57c", "bc_scratch": "#86d6b4", "rl_bc_pre": "#b8410f", "rl_bc_scratch": "#0d7a52"}
LABEL = {"none": "no repositioning", "rule": "platform rule", "rl_pre": "RL on pretrained world model",
         "rl_scratch": "RL from scratch", "untrained": "agent before RL",
         "bc_pre": "imitation, pretrained", "bc_scratch": "imitation, scratch",
         "rl_bc_pre": "imitation + RL, pretrained", "rl_bc_scratch": "imitation + RL, scratch"}
SHORT = {"rule": "rule", "rl_scratch": "RL-s", "rl_pre": "RL-p", "bc_scratch": "IM-s", "bc_pre": "IM-p",
         "rl_bc_scratch": "IM+RL-s", "rl_bc_pre": "IM+RL-p", "untrained": "untrained", "none": "none"}
QTY = [("reward", "reward (pickups + welfare)"), ("pickups", "riders picked up"), ("welfare", "drivers' welfare gain"),
       ("wait_min", "rider wait (min)"), ("cancels", "cancellations"), ("moved", "drivers ordered to move"),
       ("cost", "drivers' driving cost"), ("gini", "income Gini (window)")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)


def training(tags):
    ev = []
    for t in tags:
        f = OUT / f"eval_{t}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["tag"] = t
            ev.append(d)
    if not ev:
        return {}
    ev = pd.concat(ev)
    keys = [k for k in cities_present(ev)]
    qs = [("reward", "reward"), ("pickups", "riders picked up"), ("welfare", "drivers' welfare gain"), ("moved", "drivers moved")]
    fig, axs = plt.subplots(len(keys), len(qs) + 1, figsize=(3.3 * (len(qs) + 1), 3.0 * len(keys)), squeeze=False)
    out = {}
    for i, k in enumerate(keys):
        for j, (q, lab) in enumerate(qs):
            ax = axs[i, j]
            rule_done = False
            for t in tags:
                d = ev[(ev.tag == t) & (ev.key == k)]
                if not len(d):
                    continue
                piv = d.pivot_table(index=["iter", "start"], columns="mode", values=q).reset_index()
                if q == "moved":
                    a = piv.groupby("iter").agent.mean()
                    r = piv.groupby("iter").rule.mean()
                else:
                    a = (piv.agent - piv.none).groupby(piv.iter).mean()
                    r = (piv.rule - piv.none).groupby(piv.iter).mean()
                ax.plot(a.index, a.values, color=COL[t], lw=1.8, marker="o", ms=3, label=LABEL[t])
                if not rule_done:
                    ax.axhline(r.mean(), color=COL["rule"], lw=1.4, ls="--", label=LABEL["rule"])
                    rule_done = True
                out[f"{k}/{t}/{q}"] = {"first": float(a.iloc[0]), "last": float(a.iloc[-1]), "rule": float(r.mean())}
            if q != "moved":
                ax.axhline(0, color=INK2, lw=0.8)
            style(ax)
            ax.set_title(f"{CITY_NAME[k]}: {lab}" + (" (minus none)" if q != "moved" else " per hour"), fontsize=9, color=INK)
            if i == len(keys) - 1:
                ax.set_xlabel("GRPO iteration", fontsize=8, color=INK2)
        ax = axs[i, -1]
        for t in tags:
            d = ev[(ev.tag == t) & (ev.key == k)].groupby("iter").wm_val_nll.first()
            if len(d):
                ax.plot(d.index, d.values, color=COL[t], lw=1.8, marker="o", ms=3)
        style(ax)
        ax.set_title("world-model validation NLL", fontsize=9, color=INK)
    axs[0, 0].legend(fontsize=7, frameon=False)
    nst = int(ev.groupby(["tag", "key", "iter"]).start.nunique().max())
    fig.suptitle(f"Training in imagination: held-out validation states, 60 imagined minutes (mean of {nst} states per city)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "28_training.png", dpi=115)
    plt.close(fig)
    return out


VERSIONS = [("v0_group16", "v0: 16 drivers varied per copy"), ("v1_pairwise", "v1: one-driver difference, 30 min"),
            ("v2_driver_lr3e-4", "v2: per-driver group, 20 min"), ("v3_collapse", "v3: + fast stay bias"),
            (".", "v4: + destination credit (final)")]


def versions():
    """Held-out imagined reward (agent minus none) during training, for every version of the method,
    and the probe of how noisy one driver's effect is at each horizon."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    vcol = ["#c9c7c1", "#9a9893", "#6b6a66", "#3a3936", COL["rl_pre"]]
    out = {}
    for j, k in enumerate(("sf_hd", "pune_hd")):
        ax = axs[j]
        rule = []
        for (d, lab), c in zip(VERSIONS, vcol):
            f = OUT / d / "eval_rl_pre.csv"
            if not f.exists():
                continue
            ev = pd.read_csv(f)
            ev = ev[ev.key == k]
            piv = ev.pivot_table(index=["iter", "start"], columns="mode", values="reward").reset_index()
            a = (piv.agent - piv.none).groupby(piv.iter).mean()
            rule.append(float((piv.rule - piv.none).mean()))
            ax.plot(a.index, a.values, color=c, lw=2.2 if d == "." else 1.4, marker="o", ms=3, label=lab)
            out[f"{k}/{d}"] = {"first": float(a.iloc[0]), "last": float(a.iloc[-1]), "iters": int(a.index.max())}
        if rule:
            ax.axhline(np.mean(rule), color=COL["rule"], ls="--", lw=1.4, label=LABEL["rule"])
        ax.axhline(0, color=INK2, lw=0.8)
        style(ax)
        ax.set_title(f"{CITY_NAME[k]}: reward minus no repositioning (RL on pretrained)", fontsize=9, color=INK)
        ax.set_xlabel("GRPO iteration", fontsize=8, color=INK2)
    axs[0].legend(fontsize=7, frameon=False)
    ax = axs[2]
    for k, mk in (("sf_hd", "o"), ("pune_hd", "s")):
        f = OUT / "probe" / f"probe_{k}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["dr"] = d.dp + d.dw
            g = d.groupby("H").dr.agg(["mean", "std"])
            ax.plot(g.index / 2, g["std"], marker=mk, color=INK, lw=1.6, label=f"{CITY_NAME[k]}: s.d.")
            ax.plot(g.index / 2, g["mean"].abs(), marker=mk, color=INK2, lw=1.0, ls=":", label=f"{CITY_NAME[k]}: |mean|")
            out[f"probe/{k}"] = g.round(3).to_dict()
    ax.axvline(20, color=COL["rl_pre"], lw=1.2, ls="--")
    ax.set_yscale("log")
    style(ax)
    ax.set_xlabel("imagined minutes after the choice", fontsize=8, color=INK2)
    ax.set_title("one driver's move: effect on reward vs its noise", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("How the RL method was arrived at: held-out imagined reward per version, and why credit stops at 20 minutes", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "28_versions.png", dpi=115)
    plt.close(fig)
    return out


def cities_present(ev):
    return [k for k in ("sf_hd", "pune_hd") if k in set(ev.key)]


def execute():
    rows = []
    for k in ("sf_hd", "pune_hd"):
        for f in sorted((cities.get(k).data_dir / "wm3_rl" / "exec").glob("rec_seed*.json")):
            rec = json.loads(f.read_text())
            for arm, m in rec["arms"].items():
                r = {"key": k, "seed": rec["seed"], "date": rec["date"], "t0_h": rec["t0"] / 3600, "arm": arm}
                r.update({f"real_{q}": m.get(q, np.nan) for q, _ in QTY})
                im = rec.get("imagined", {}).get(arm)
                if im:
                    d = pd.DataFrame(im)
                    r.update({f"imag_{q}": float(d[q].mean()) if q in d else np.nan for q, _ in QTY})
                rows.append(r)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    # differences with no repositioning, per seed
    for src in ("real", "imag"):
        for q, _ in QTY:
            c = f"{src}_{q}"
            if c not in df:
                continue
            base = df[df.arm == "none"].set_index(["key", "seed"])[c]
            df[f"d_{c}"] = df[c] - base.reindex(pd.MultiIndex.from_arrays([df.key, df.seed])).to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "28_execute.csv", index=False)
    arms = [a for a in ("rule", "rl_scratch", "rl_pre", "bc_scratch", "bc_pre", "rl_bc_scratch", "rl_bc_pre") if a in set(df.arm)]
    summ = {}
    for k in ("sf_hd", "pune_hd", "both"):
        d = df if k == "both" else df[df.key == k]
        for a in arms:
            x = d[d.arm == a]
            for q, _ in QTY:
                for src in ("real", "imag"):
                    c = f"d_{src}_{q}"
                    if c in x and x[c].notna().any():
                        v = x[c].dropna()
                        summ[f"{k}/{a}/{src}/{q}"] = {"mean": float(v.mean()), "se": float(v.std() / np.sqrt(max(len(v), 1))), "n": int(len(v))}
    # bars: real effect vs none, imagined as dots
    qs = [q for q in QTY if q[0] in ("reward", "pickups", "welfare", "wait_min", "moved", "cost")]
    keys = [k for k in ("sf_hd", "pune_hd") if k in set(df.key)]
    fig, axs = plt.subplots(len(keys), len(qs), figsize=(3.0 * len(qs), 3.2 * len(keys)), squeeze=False)
    for i, k in enumerate(keys):
        for j, (q, lab) in enumerate(qs):
            ax = axs[i, j]
            for x, a in enumerate(arms):
                d = df[(df.key == k) & (df.arm == a)]
                v = d[f"d_real_{q}"].dropna()
                if not len(v):
                    continue
                ax.bar(x, v.mean(), 0.62, color=COL[a], yerr=v.std() / np.sqrt(len(v)), ecolor=INK2, capsize=3,
                       label=f"{SHORT[a]}: {LABEL[a]}" if (i == 0 and j == 0) else None)
                ax.scatter(np.full(len(v), x) + np.linspace(-0.18, 0.18, len(v)), v, s=9, color=INK, zorder=3, lw=0)
                if f"d_imag_{q}" in d and d[f"d_imag_{q}"].notna().any():
                    ax.scatter([x + 0.38], [d[f"d_imag_{q}"].mean()], marker="D", s=30, color="white", edgecolor=COL[a], lw=1.6, zorder=4,
                               label="imagined (world model)" if (i == 0 and j == 0 and x == 0) else None)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([SHORT[a] for a in arms], fontsize=7, rotation=45, ha="right")
            style(ax)
            ax.set_title(f"{CITY_NAME[k]}: {lab}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("minus no repositioning, 2 h", fontsize=8, color=INK2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 8), fontsize=8, frameon=False)
    fig.suptitle("Choices executed in the simulator (fresh seasons, test days, 2-hour windows): bars = real mean ± s.e., "
                 "dots = seeds, diamonds = what the world model imagined", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "28_execute.png", dpi=115)
    plt.close(fig)
    # imagined vs real effect
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, (q, lab) in zip(axs, [QTY[0], QTY[1], QTY[2]]):
        d = df[(df.arm != "none") & df[f"d_imag_{q}"].notna()] if f"d_imag_{q}" in df else df.iloc[:0]
        for a in arms:
            x = d[d.arm == a]
            for k, mk in (("sf_hd", "o"), ("pune_hd", "s")):
                xx = x[x.key == k]
                ax.scatter(xx[f"d_imag_{q}"], xx[f"d_real_{q}"], color=COL[a], marker=mk, s=28,
                           label=f"{LABEL[a]}, {CITY_NAME[k]}" if q == "reward" else None)
        if len(d) > 2:
            lim = [min(d[f"d_imag_{q}"].min(), d[f"d_real_{q}"].min()), max(d[f"d_imag_{q}"].max(), d[f"d_real_{q}"].max())]
            ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
            r = np.corrcoef(d[f"d_imag_{q}"], d[f"d_real_{q}"])[0, 1]
            ax.set_title(f"{lab}: r = {r:.2f}", fontsize=9, color=INK)
            summ[f"imag_vs_real_r/{q}"] = float(r)
        style(ax)
        ax.set_xlabel("imagined effect (minus no repositioning)", fontsize=8, color=INK2)
        ax.set_ylabel("real effect", fontsize=8, color=INK2)
    axs[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Does the world model predict what each platform's choices really do? (one point per seed and platform)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "28_scatter.png", dpi=115)
    plt.close(fig)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["rl_pre", "rl_scratch"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = {"training": training(args.tags), "versions": versions(), "execute": execute()}
    (OUT / "28_summary.json").write_text(json.dumps(s, indent=1))
    for k, v in s["execute"].items():
        if "/reward" in k or "/pickups" in k or "/welfare" in k:
            print(f"  {k}: {v}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
