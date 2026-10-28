"""World model 3, stage 8c: is it a world model? A scorecard against the usual definition.

    .venv/bin/python scripts/37_wm3_scorecard.py

A world model, in the sense of Ha & Schmidhuber (2018), Dreamer (Hafner et al.), MuZero and LeCun
(2022), is a learned model of an environment that
  1  represents the state of the world (here: every driver, rider and road)
  2  predicts what happens next, as a distribution, from that state
  3  can be rolled forward on its own predictions for a long time without drifting away
  4  spreads its imagined futures as widely as the real world varies (calibrated uncertainty)
  5  responds to actions as the world does (action-conditioned)
  6  answers "what if" for interventions outside its training data (counterfactual, causal)
  7  supports decisions: policies chosen or trained inside it rank and work the same in the world
  8  holds knowledge that transfers (its representation helps learn new tasks)
  9  generalises beyond the days and people it was trained on
 10  learns the dynamics rather than being handed them (how much is learned vs. coded rules)
This script gathers the evidence of stages 1-8 for each, adds the policy-ranking test (8c: every
platform / agent executed in the simulator in stages 5-7 vs. the same policy imagined from the same
start) and writes outputs/wm3/stage8/37_scorecard.png, 37_ranking.png, 37_scorecard.json.
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

OUT = cities.OUTPUTS / "wm3" / "stage8"
W3 = cities.OUTPUTS / "wm3"
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
INK, INK2, GRID, BLUE, ORANGE, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#9a9893"
GOOD, WARN, BAD = "#2f8f4e", "#c98a12", "#c0392b"
ARM_NAME = {"rule": "platform rule", "rl_pre": "RL only, pretrained", "rl_scratch": "RL only, scratch",
            "bc_pre": "imitation, pretrained", "bc_scratch": "imitation, scratch", "rl_bc_pre": "imitation + RL, pretrained",
            "rl_bc_scratch": "imitation + RL, scratch", "mbc_pre+m": "agent matches, pretrained",
            "mbc_scratch+m": "agent matches, scratch", "mbc_pre+m@start": "menus, first option",
            "mbc_pre+m@drv_pre": "menus, trained drivers"}


def spearman(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def ranking(key):
    """Every arm executed in the simulator with an imagined counterpart: effect vs. 'none'."""
    rows = []
    for f in sorted((cities.get(key).data_dir / "wm3_rl" / "exec").glob("rec_seed*.json")):
        rec = json.loads(f.read_text())
        im, ar = rec.get("imagined", {}), rec["arms"]
        if "none" not in im or "none" not in ar:
            continue
        i0 = pd.DataFrame(im["none"])[["pickups", "reward"]].mean()
        for arm in ARM_NAME:
            if arm in im and arm in ar:
                ia = pd.DataFrame(im[arm])[["pickups", "reward"]].mean()
                rows.append({"key": key, "seed": rec["seed"], "arm": arm,
                             "real_pickups": ar[arm]["pickups"] - ar["none"]["pickups"], "real_reward": ar[arm]["reward"] - ar["none"]["reward"],
                             "imag_pickups": ia.pickups - i0.pickups, "imag_reward": ia.reward - i0.reward})
    return pd.DataFrame(rows)


def rank_summary(df):
    out = {}
    for key, d in df.groupby("key"):
        m = d.groupby("arm")[["real_reward", "imag_reward", "real_pickups", "imag_pickups"]].mean()
        se = d.groupby("arm")[["real_reward", "imag_reward"]].sem()
        n = d.groupby("arm").size()
        arms = list(m.index)
        # pairs of arms whose real difference is clear (paired over seeds, |t| > 2): same order imagined?
        agree, clear = 0, 0
        for a in range(len(arms)):
            for b in range(a + 1, len(arms)):
                x = d[d.arm == arms[a]].set_index("seed")
                y = d[d.arm == arms[b]].set_index("seed")
                s = x.index.intersection(y.index)
                if len(s) < 4:
                    continue
                dr = (x.loc[s, "real_reward"] - y.loc[s, "real_reward"]).to_numpy()
                di = (x.loc[s, "imag_reward"] - y.loc[s, "imag_reward"]).to_numpy()
                if abs(dr.mean()) > 2 * dr.std(ddof=1) / np.sqrt(len(s)):
                    clear += 1
                    agree += int(np.sign(dr.mean()) == np.sign(di.mean()))
        out[key] = {"arms": {a: {"real": float(m.loc[a, "real_reward"]), "real_se": float(se.loc[a, "real_reward"]),
                                 "imag": float(m.loc[a, "imag_reward"]), "imag_se": float(se.loc[a, "imag_reward"]), "n": int(n[a])} for a in arms},
                    "spearman_arms": spearman(m.real_reward, m.imag_reward),
                    "r_arm_means": float(np.corrcoef(m.real_reward, m.imag_reward)[0, 1]),
                    "r_seed_arm": float(np.corrcoef(d.real_reward, d.imag_reward)[0, 1]),
                    "clear_pairs": clear, "clear_pairs_same_order": agree,
                    "best_real": str(m.real_reward.idxmax()), "best_imag": str(m.imag_reward.idxmax())}
    return out


def plot_ranking(rs, path):
    keys = list(rs)
    fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 4.6), squeeze=False)
    for a, key in enumerate(keys):
        ax = axes[0, a]
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        r = rs[key]
        for arm, v in r["arms"].items():
            ax.errorbar(v["real"], v["imag"], xerr=v["real_se"], yerr=v["imag_se"], fmt="o", ms=6, color=BLUE, ecolor=GREY, lw=0.8)
            ax.annotate(ARM_NAME[arm], (v["real"], v["imag"]), fontsize=7, color=INK2, xytext=(4, 3), textcoords="offset points")
        lo = min(min(v["real"], v["imag"]) for v in r["arms"].values()) - 3
        hi = max(max(v["real"], v["imag"]) for v in r["arms"].values()) + 3
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
        ax.axhline(0, color=GREY, lw=0.5)
        ax.axvline(0, color=GREY, lw=0.5)
        ax.set_xlabel("effect in the simulator (reward vs. no repositioning, 2 h)", fontsize=8, color=INK2)
        ax.set_ylabel("effect imagined by the world model", fontsize=8, color=INK2)
        ax.set_title(f"{CITY_NAME[key]}: rank correlation {r['spearman_arms']:.2f}; clear pairs in the same order "
                     f"{r['clear_pairs_same_order']}/{r['clear_pairs']}", fontsize=9, color=INK)
    fig.suptitle("Policies judged in imagination vs. in the world (every platform / agent of stages 5-7; mean +- s.e. over seeds)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def criteria(rs):
    s1, s2 = jl(W3 / "stage1/16_stage1.json"), jl(W3 / "stage2/19_eval.json")
    s4, l8, k8 = jl(W3 / "stage4/23_counterfactual.json"), jl(OUT / "35_long.json"), jl(OUT / "36_shock.json")
    C = []

    def add(name, question, verdict, sf, pune, note=""):
        C.append({"criterion": name, "question": question, "verdict": verdict, "sf_hd": sf, "pune_hd": pune, "note": note})

    add("1 State", "Does it represent the whole world state?", "met",
        f"every driver, rider, road; zone counts match the simulator {100 * s1['sf_hd']['busy_zone_match']:.2f}%",
        f"{100 * s1['pune_hd']['busy_zone_match']:.2f}%", "hidden traits are not given; the model infers them from history")
    heads = [h for h in s2["sf_hd"] if isinstance(s2["sf_hd"][h], dict) and "model" in s2["sf_hd"][h]]
    better = {k: sum(s2[k][h]["model"] < s2[k][h]["baseline"] for h in heads) for k in ("sf_hd", "pune_hd")}
    add("2 Next step", "Does it predict the next 30 s as a calibrated distribution?", "met",
        f"beats historical rates on {better['sf_hd']}/{len(heads)} event heads", f"{better['pune_hd']}/{len(heads)}",
        "calibration close to the diagonal on every binary head (stage 2)")
    if l8:
        v = {}
        for k in ("sf_hd", "pune_hd"):
            r = l8.get(k)
            if not r:
                v[k] = "-"
                continue
            kp = r["kpi"]
            win = sum(kp[x]["model"][-1] < kp[x]["history"][-1] for x in kp)
            winp = sum(kp[x]["model"][-1] < kp[x]["persistence"][-1] for x in kp)
            v[k] = (f"8 h error: pickups {kp['pickups']['model'][-1]:.0%}, online {kp['online']['model'][-1]:.0%}; "
                    f"beats persistence {winp}/{len(kp)}, historical average {win}/{len(kp)}")
        add("3 Long rollouts", "Does it stay on track over hours of its own predictions?", "partly", v["sf_hd"], v["pune_hd"],
            "stage 8a; drifts high after ~3 h (requests +10%, cancellations +26-52%)")
        cal = {k: (f"truth inside the draws' range {np.mean([l8[k]['kpi'][x]['inside_range'] for x in l8[k]['kpi']]):.0%} "
                   f"(ideal {l8[k]['ideal_inside']:.0%})" if l8.get(k) else "-") for k in ("sf_hd", "pune_hd")}
        add("4 Uncertainty", "Do its imagined futures spread as much as reality?", "not met", cal["sf_hd"], cal["pune_hd"],
            "draws too narrow: no day-level demand factor, and the drift above")
    add("5 Actions", "Does it respond to the platform's actions as the world does?", "met",
        f"repositioning orders: idle drivers at the target r {s4['sf_hd']['tgt_idle@30']['r']:.2f}",
        f"r {s4['pune_hd']['tgt_idle@30']['r']:.2f}", "stage 4, fresh seasons; city-wide effects ~0 in both")
    if k8:
        def sh(k):
            out = []
            for arm in ("supply60", "price15"):
                s = k8.get(f"{k}/{arm}")
                if s:
                    out.append(f"{arm}: pickups {s['pickups']['real']:+.0f} real vs {s['pickups']['model']:+.0f} model")
            return "; ".join(out) or "-"
        add("6 What-if", "Does it predict shocks it never saw?", "partly", sh("sf_hd"), sh("pune_hd"),
            "stage 8b; supply cuts right in size, price rise right in sign but ~1/3 the size")
    if rs:
        rk = {k: (f"rank corr. {rs[k]['spearman_arms']:.2f} over {len(rs[k]['arms'])} policies; clear pairs same order "
                  f"{rs[k]['clear_pairs_same_order']}/{rs[k]['clear_pairs']}" if k in rs else "-") for k in ("sf_hd", "pune_hd")}
        add("7 Decisions", "Do policies built in imagination work, and rank, the same in the world?", "met", rk["sf_hd"], rk["pune_hd"],
            "agents trained in imagination match the rule in the simulator; imagination overrates the stage-7 menus")
    add("8 Knowledge", "Does its representation help learn new tasks?", "met", "pretrained vs scratch agent +10 to +11 reward",
        "within noise", "stages 5-6; it also stays a predictor under RL (NLL 14.89 -> 15.12)")
    add("9 Generalise", "Does it work on new days and new people?", "partly", "test days, fresh seasons: yes",
        "yes", "one model for both cities; not tried: a new city, new road network, a different fleet size from the start")
    add("10 Learned", "How much of the dynamics is learned?", "partly", "learned: every human decision and all demand",
        "same", "coded rules: driving on the road graph, routes, fares, surge, request lifecycle; traffic is the typical profile")
    return C


def plot_scorecard(C, path):
    col = {"met": GOOD, "partly": WARN, "not met": BAD}
    icon = {"met": "✔ met", "partly": "◐ partly", "not met": "✘ not met"}
    n = len(C)
    import textwrap
    wr = lambda t, w: "\n".join(textwrap.wrap(t, w))
    fig = plt.figure(figsize=(17, 0.8 * n + 1.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(n + 1.2, 0)
    xs = [0.01, 0.11, 0.33, 0.405, 0.73]
    for x, h in zip(xs, ["criterion", "question", "verdict", "San Francisco", "Pune"]):
        ax.text(x, 0.55, h, fontsize=9, color=INK2, weight="bold", va="center")
    for i, c in enumerate(C):
        y = i + 1.3
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - 0.5), 1, 1, color="#f4f3f0", lw=0))
        ax.text(xs[0], y, c["criterion"], fontsize=9, color=INK, va="center", weight="bold")
        ax.text(xs[1], y, wr(c["question"], 40), fontsize=8, color=INK, va="center")
        ax.text(xs[2], y, icon[c["verdict"]], fontsize=9, color=col[c["verdict"]], va="center", weight="bold")
        ax.text(xs[3], y - 0.15, wr(c["sf_hd"], 62), fontsize=7.5, color=INK, va="center")
        ax.text(xs[4], y - 0.15, wr(c["pune_hd"], 52), fontsize=7.5, color=INK, va="center")
        if c["note"]:
            ax.text(xs[3], y + 0.3, c["note"], fontsize=6.8, color=INK2, va="center", style="italic")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([ranking(k) for k in ("sf_hd", "pune_hd")], ignore_index=True)
    df.to_csv(OUT / "37_ranking.csv", index=False)
    rs = rank_summary(df)
    plot_ranking(rs, OUT / "37_ranking.png")
    C = criteria(rs)
    plot_scorecard(C, OUT / "37_scorecard.png")
    (OUT / "37_scorecard.json").write_text(json.dumps({"ranking": rs, "criteria": C}, indent=1, default=float))
    for k, r in rs.items():
        print(k, {x: r[x] for x in r if x != "arms"})
        for a, v in r["arms"].items():
            print(f"   {a:22s} real {v['real']:+6.1f} +- {v['real_se']:.1f}  imagined {v['imag']:+6.1f} +- {v['imag_se']:.1f}  n {v['n']}")
    for c in C:
        print(f"{c['criterion']:16s} {c['verdict']:8s} | {c['sf_hd']} | {c['pune_hd']}")


if __name__ == "__main__":
    main()
