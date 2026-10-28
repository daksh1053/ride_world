"""World model 3, stages 6-7: figures and tables.

    .venv/bin/python scripts/34_wm3_match_report.py

Stage 6 (the agent also matches riders to drivers), outputs/wm3/stage6/:
    34_imitation.png    imitating the platform's matching (scripts/31_wm3_match_bc.py): held-out
                        NLL, exact-driver accuracy, pickup minutes of the chosen driver vs the rule's
    34_training.png     GRPO on matching (scripts/32_wm3_match_grpo.py), held-out imagined 60 min:
                        reward, riders picked up, welfare minus no repositioning, for the agent
                        matching + repositioning, next to the stage-5 agent (rule matching) and the rule
    34_execute.png      in the simulator (scripts/27_wm3_rl_execute.py, fresh seasons, test days,
                        2-h windows), minus the platform's rule matching with no repositioning
    34_execute.csv, 34_summary.json, 34_paired.txt
Stage 7 (drivers as earning-maximising agents choosing from menus), outputs/wm3/stage7/:
    34_drivers_training.png   held-out imagined 60 min during GRPO (scripts/33_wm3_driver_grpo.py)
    34_execute.png            in the simulator: single offers answered by the simulator's drivers vs
                              menus with the starting and the trained driver policy
    34_execute.csv, 34_summary.json, 34_paired.txt
"""

from __future__ import annotations

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

OUT6 = cities.OUTPUTS / "wm3" / "stage6"
OUT7 = cities.OUTPUTS / "wm3" / "stage7"
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
KEYS = ("sf_hd", "pune_hd")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PRE, SCR = "#b8410f", "#0d7a52"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)


# ---------------------------------------------------------------------- stage 6
def imitation():
    runs = {t: OUT6 / f"31_{t}.csv" for t in ("mbc_pre", "mbc_scratch")}
    runs = {t: pd.read_csv(f) for t, f in runs.items() if f.exists()}
    if not runs:
        return {}
    qs = [("nll", "held-out NLL of the rule's choice"), ("acc", "exact-driver accuracy"), ("eta", "pickup min of chosen driver"),
          ("repo_kl", "repositioning KL to the stage-5 agent")]
    fig, axs = plt.subplots(2, len(qs), figsize=(3.3 * len(qs), 6.0))
    out = {}
    for i, k in enumerate(KEYS):
        for j, (q, lab) in enumerate(qs):
            ax = axs[i, j]
            for t, df in runs.items():
                c = PRE if t.endswith("pre") else SCR
                name = "pretrained" if t.endswith("pre") else "scratch"
                if q == "eta":
                    ax.plot(df.epoch, df[f"{k}/eta_agent"], color=c, lw=2, label=f"agent ({name})")
                    ax.plot(df.epoch, df[f"{k}/eta_rule"], color=INK2, lw=1, ls="--", label="the rule" if t.endswith("pre") else None)
                else:
                    ax.plot(df.epoch, df[f"{k}/{q}"], color=c, lw=2, label=name)
                out[f"{k}/{t}/{q}"] = float(df[f"{k}/{q}"].iloc[-1]) if q != "eta" else float(df[f"{k}/eta_agent"].iloc[-1])
            style(ax)
            ax.set_title(f"{CITY_NAME[k]}: {lab}", fontsize=9, color=INK)
            ax.set_xlabel("epoch", fontsize=8, color=INK2)
            if i == 0 and j in (0, 2):
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Stage 6a: the agent imitates the platform's matching (each pending request picks a driver or holds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT6 / "34_imitation.png", dpi=115)
    plt.close(fig)
    return out


def grpo_training():
    runs = {t: OUT6 / f"eval_{t}.csv" for t in ("mrl_pre", "mrl_scratch")}
    runs = {t: pd.read_csv(f) for t, f in runs.items() if f.exists()}
    if not runs:
        return {}
    qs = [("reward", "reward (pickups + welfare)"), ("pickups", "riders picked up"), ("welfare", "drivers' welfare gain"),
          ("held", "requests held (agent)")]
    fig, axs = plt.subplots(2, len(qs), figsize=(3.3 * len(qs), 6.0))
    out = {}
    for i, k in enumerate(KEYS):
        for j, (q, lab) in enumerate(qs):
            ax = axs[i, j]
            for t, df in runs.items():
                d = df[df.key == k]
                if not len(d):
                    continue
                piv = d.pivot_table(index=["iter", "start"], columns="mode", values=q).reset_index()
                c = PRE if t.endswith("pre") else SCR
                name = "pretrained" if t.endswith("pre") else "scratch"
                if q == "held":
                    g = piv.groupby("iter")["agent"].mean()
                    ax.plot(g.index, g.values, color=c, lw=2, marker="o", ms=3, label=name)
                    continue
                for arm, ls, lw, lab2 in (("agent", "-", 2, f"agent matches + repositions ({name})"),
                                          ("repo_agent", ":", 1.4, f"stage-5 agent: rule matches ({name})")):
                    g = (piv[arm] - piv["none"]).groupby(piv["iter"])
                    ax.plot(g.mean().index, g.mean().values, color=c, ls=ls, lw=lw, marker="o" if ls == "-" else None, ms=3, label=lab2)
                if t.endswith("pre"):
                    g = (piv["rule"] - piv["none"]).groupby(piv["iter"]).mean()
                    ax.plot(g.index, g.values, color="#2a78d6", lw=1.2, ls="--", label="platform rule")
                g = (piv["agent"] - piv["none"]).groupby(piv["iter"]).mean()
                out[f"{k}/{t}/{q}/first"], out[f"{k}/{t}/{q}/last"] = float(g.iloc[0]), float(g.iloc[-1])
            ax.axhline(0, color=INK2, lw=0.6)
            style(ax)
            ax.set_title(f"{CITY_NAME[k]}: {lab}", fontsize=9, color=INK)
            ax.set_xlabel("GRPO iteration", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel("minus rule matching, no repositioning", fontsize=8, color=INK2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Stage 6b: GRPO on matching, imagined 60 min from held-out states", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT6 / "34_training.png", dpi=115)
    plt.close(fig)
    return out


def load_exec():
    rows = []
    for k in KEYS:
        for f in sorted((cities.get(k).data_dir / "wm3_rl" / "exec").glob("rec_seed*.json")):
            rec = json.loads(f.read_text())
            for arm, m in rec["arms"].items():
                r = {"key": k, "seed": rec["seed"], "arm": arm, **{f"real_{q}": v for q, v in m.items() if isinstance(v, (int, float))}}
                im = rec.get("imagined", {}).get(arm)
                if im:
                    d = pd.DataFrame(im)
                    r.update({f"imag_{q}": float(d[q].mean()) for q in d.columns if q != "draw" and pd.api.types.is_numeric_dtype(d[q])})
                rows.append(r)
    return pd.DataFrame(rows)


def exec_figure(df, arms, base, qs, out, title, fname, labels, colors):
    """Bars: real effect (arm minus base) per seed, mean +- s.e.; diamonds: imagined."""
    df = df[df.arm.isin(arms + [base])].copy()
    if df.empty or not set(arms) & set(df.arm):
        return {}
    cols = [c for c in df.columns if c.startswith(("real_", "imag_"))]
    b = df[df.arm == base].set_index(["key", "seed"])[cols]
    for c in cols:
        df[f"d_{c}"] = df[c] - b[c].reindex(pd.MultiIndex.from_arrays([df.key, df.seed])).to_numpy()
    # only seeds where every arm ran (paired)
    have = df.groupby(["key", "seed"]).arm.nunique()
    keep = have[have == len(set(arms + [base]) & set(df.arm))].index
    df = df.set_index(["key", "seed"]).loc[keep].reset_index()
    arms = [a for a in arms if a in set(df.arm)]
    df.to_csv(out / fname.replace(".png", ".csv"), index=False)
    keys = [k for k in KEYS if k in set(df.key)]
    summ, paired = {}, []
    fig, axs = plt.subplots(len(keys), len(qs), figsize=(2.9 * len(qs), 3.3 * len(keys)), squeeze=False)
    for i, k in enumerate(keys):
        for j, (q, lab) in enumerate(qs):
            ax = axs[i, j]
            for x, a in enumerate(arms):
                d = df[(df.key == k) & (df.arm == a)]
                if f"d_real_{q}" not in d:
                    continue
                v = d[f"d_real_{q}"].dropna()
                if not len(v):
                    continue
                summ[f"{k}/{a}/real/{q}"] = {"mean": float(v.mean()), "se": float(v.std() / np.sqrt(len(v))), "n": int(len(v)),
                                             "abs": float(d[f"real_{q}"].mean())}
                ax.bar(x, v.mean(), 0.62, color=colors[a], yerr=v.std() / np.sqrt(len(v)), ecolor=INK2, capsize=3,
                       label=labels[a] if (i == 0 and j == 0) else None)
                ax.scatter(np.full(len(v), x) + np.linspace(-0.18, 0.18, len(v)), v, s=8, color=INK, zorder=3, lw=0)
                if f"d_imag_{q}" in d and d[f"d_imag_{q}"].notna().any():
                    iv = d[f"d_imag_{q}"].dropna()
                    summ[f"{k}/{a}/imag/{q}"] = {"mean": float(iv.mean()), "se": float(iv.std() / np.sqrt(len(iv))), "n": int(len(iv))}
                    ax.scatter([x + 0.38], [iv.mean()], marker="D", s=30, color="white", edgecolor=colors[a], lw=1.6, zorder=4,
                               label="imagined (world model)" if (i == 0 and j == 0 and x == 0) else None)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([labels[a].split(":")[0] for a in arms], fontsize=7, rotation=40, ha="right")
            style(ax)
            ax.set_title(f"{CITY_NAME[k]}: {lab}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"minus {labels.get(base, base).split(':')[0]}, 2 h", fontsize=8, color=INK2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / fname, dpi=115)
    plt.close(fig)
    # paired differences between arms
    for k in keys:
        for q, _ in qs:
            if f"real_{q}" not in df:
                continue
            piv = df[df.key == k].pivot_table(index="seed", columns="arm", values=f"real_{q}")
            for x in range(len(arms)):
                for y in range(x + 1, len(arms)):
                    a1, a2 = arms[x], arms[y]
                    if a1 in piv and a2 in piv:
                        dd = (piv[a2] - piv[a1]).dropna()
                        if len(dd) > 1:
                            paired.append(f"{CITY_NAME[k]:14s} {q:14s} {a2:>22s} - {a1:<22s} {dd.mean():+8.3f} +- {dd.std() / np.sqrt(len(dd)):.3f}"
                                          f"  (n {len(dd)}, {int((dd > 0).sum())} up)")
    (out / fname.replace(".png", "_paired.txt")).write_text("\n".join(paired) + "\n")
    return summ


def stage6_exec(df):
    arms = ["rule", "rl_bc_pre", "mbc_pre+m", "mbc_scratch+m", "mrl_pre+m"]
    labels = {"none": "none: rule matching, no repositioning", "rule": "rule: platform matching + rebalancing",
              "rl_bc_pre": "stage 5: rule matching, agent repositions", "mbc_pre+m": "IM-p: imitation, pretrained: agent matches + repositions",
              "mbc_scratch+m": "IM-s: imitation, scratch: agent matches + repositions",
              "mrl_pre+m": "RL-p: IM + GRPO pretrained, matches + repositions", "mrl_scratch+m": "RL-s: IM + GRPO scratch, matches + repositions",
              "mrl_pre+m-none": "RL-p match only: agent matches, no repositioning"}
    colors = {"rule": "#2a78d6", "rl_bc_pre": "#f2a57c", "mbc_pre+m": PRE, "mbc_scratch+m": SCR, "mrl_pre+m": "#8a5cd6", "mrl_scratch+m": SCR,
              "mrl_pre+m-none": "#d9a400"}
    qs = [("reward", "reward (pickups + welfare)"), ("pickups", "riders picked up"), ("welfare", "drivers' welfare gain"),
          ("rejected", "offers rejected"), ("offer_eta_min", "pickup min of offers"), ("wait_min", "rider wait (min)")]
    return exec_figure(df, arms, "none", qs, OUT6, "Stage 6: the agent matches riders to drivers in the simulator "
                       "(fresh seasons, test days, 2-hour windows): bars = real mean ± s.e., dots = seeds, diamonds = imagined",
                       "34_execute.png", labels, colors)


# ---------------------------------------------------------------------- stage 7
def drivers_training():
    fs = sorted(OUT7.glob("eval_drv_*.csv"))
    if not fs:
        return {}
    qs = [("net_per_h", "drivers' net per online hour (median fares)"), ("net_per_h_p10", "10th percentile driver, per hour"),
          ("pickups", "riders picked up"), ("welfare", "drivers' welfare gain")]
    col = {"single": "#2a78d6", "menu_start": "#9a9893", "menu_agents": PRE}
    lab = {"single": "single offers, simulated people answer", "menu_start": "menus, drivers take the platform's first option",
           "menu_agents": "menus, earning-maximising driver agents"}
    out = {}
    fig, axs = plt.subplots(2, len(qs), figsize=(3.3 * len(qs), 6.0))
    for f in fs:
        df = pd.read_csv(f)
        for i, k in enumerate(KEYS):
            for j, (q, ql) in enumerate(qs):
                ax = axs[i, j]
                g = df[df.key == k].groupby(["iter", "mode"])[q].mean().unstack()
                for m in ("single", "menu_start", "menu_agents"):
                    if m in g:
                        ax.plot(g.index, g[m], color=col[m], lw=2 if m == "menu_agents" else 1.2, ls="-" if m == "menu_agents" else "--",
                                marker="o" if m == "menu_agents" else None, ms=3, label=lab[m])
                        out[f"{k}/{m}/{q}/first"], out[f"{k}/{m}/{q}/last"] = float(g[m].iloc[0]), float(g[m].iloc[-1])
                style(ax)
                ax.set_title(f"{CITY_NAME[k]}: {ql}", fontsize=9, color=INK)
                ax.set_xlabel("GRPO iteration", fontsize=8, color=INK2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Stage 7: drivers learn to choose from the platform's menus to maximise their own earnings (imagined 60 min, held-out states)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT7 / "34_drivers_training.png", dpi=115)
    plt.close(fig)
    return out


def stage7_exec(df, platform, drv):
    single = f"{platform}+m"
    arms = [f"{platform}+m@start", f"{platform}+m@{drv}"]
    labels = {single: "single: one offer per driver, simulated people accept or reject",
              f"{platform}+m@start": "menus-start: drivers take the platform's first option",
              f"{platform}+m@{drv}": "menus-agents: earning-maximising driver agents choose"}
    colors = {arms[0]: "#9a9893", arms[1]: PRE}
    qs = [("net_per_h", "drivers' net per online hour"), ("net_per_h_p10", "10th-percentile driver per hour"),
          ("pickups", "riders picked up"), ("wait_min", "rider wait (min)"), ("welfare", "drivers' welfare gain"), ("gini", "income Gini")]
    return exec_figure(df, arms, single, qs, OUT7, "Stage 7 in the simulator: menus with driver agents, minus single offers answered by the "
                       "simulated people (same platform, fresh seasons, test days, 2-h windows)", "34_execute.png", labels, colors)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="mrl_pre")
    ap.add_argument("--drivers", default="drv_pre")
    args = ap.parse_args()
    OUT6.mkdir(parents=True, exist_ok=True)
    OUT7.mkdir(parents=True, exist_ok=True)
    df = load_exec()
    s6 = {"imitation": imitation(), "training": grpo_training(), "execute": stage6_exec(df) if len(df) else {}}
    (OUT6 / "34_summary.json").write_text(json.dumps(s6, indent=1))
    s7 = {"training": drivers_training(), "execute": stage7_exec(df, args.platform, args.drivers) if len(df) else {}}
    (OUT7 / "34_summary.json").write_text(json.dumps(s7, indent=1))
    for name, s in (("stage 6", s6), ("stage 7", s7)):
        for k, v in s["execute"].items():
            if any(f"/{q}" in k for q in ("reward", "pickups", "net_per_h", "rejected")) and "/real/" in k:
                print(f"  {name} {k}: {v['mean']:+.3f} +- {v['se']:.3f} (n {v['n']})")
    print(f"  wrote {OUT6} and {OUT7}")


if __name__ == "__main__":
    main()
