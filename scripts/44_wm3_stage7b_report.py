"""Stage 7b figures and report: making 'take the first ride at once' no longer a driver's best choice.

    .venv/bin/python scripts/44_wm3_stage7b_report.py

Reads outputs/wm3/stage7b/41_tune.csv (market probe), eval_drv2_big.csv / eval_drv2_d3.csv
(training in the world model) and data/<city>/wm3_rl/exec7b/rec_seed*_drv2_d3_it20.json
(simulator test). Writes outputs/wm3/stage7b/44_market.png, 44_training.png, 44_simulator.png and
stage7b_report.html (figures + a replay of one simulator day: all drivers take the first ride vs
all drivers use the trained agent, same day, same riders).
"""

from __future__ import annotations

import glob
import importlib.util
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

ROOT = Path(__file__).resolve().parents[1]
OUT = cities.OUTPUTS / "wm3" / "stage7b"
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
KEYS = ("sf_hd", "pune_hd")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEW, OLD, ALT = "#b8410f", "#7a7975", "#0d7a52"
AGENT = "drv2_d3_it20"
SETTINGS = [("exclusive", 0.0, 1.0, "own\nmenus"), ("exclusive", 0.5, 1.0, "+ half idle\ndrivers off"),
            ("exclusiveb4", 0.5, 1.0, "+ menus\nevery 2 min"), ("exclusiveb4", 0.5, 2.0, "+ demand\nx2"),
            ("exclusiveb4", 0.5, 3.0, "+ demand x3\n(chosen)")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6, axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def zero(ax):
    ax.axhline(0, color=INK, lw=0.8)


# ---------------------------------------------------------------------- 1. the market
def market_fig():
    d = pd.read_csv(OUT / "41_tune.csv")
    d = d[d.scheme == "per_trip"]
    fig, axs = plt.subplots(3, 2, figsize=(10, 8.2), sharex=True)
    x = np.arange(len(SETTINGS))
    out = {}
    for j, k in enumerate(KEYS):
        rows = []
        for menu, s, dm, _ in SETTINGS:
            r = d[(d.city == k) & (d.menu == menu) & (d.supply_cut == s) & (d.demand == dm)]
            rows.append(r.iloc[0] if len(r) else None)
        get = lambda c: np.array([r[c] if r is not None else np.nan for r in rows])  # noqa: E731
        ax = axs[0, j]
        ax.bar(x, get("mean_options"), color=NEW, width=0.6)
        ax.axhline(1, color=INK2, lw=0.8, ls="--")
        ax.set_title(CITY_NAME[k], fontsize=11, color=INK, loc="left")
        ax.set_ylabel("rides per menu", fontsize=9, color=INK2)
        ax = axs[1, j]
        ax.bar(x, 100 * get("first_is_best"), color=NEW, width=0.6)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of choices where the\nfirst ride is the best", fontsize=9, color=INK2)
        ax = axs[2, j]
        w = 0.36
        for o, (c, col, lab) in enumerate([("rate_1.0", NEW, "simple rule: most pay per minute"),
                                           ("learned", ALT, "value model fitted on other choices")]):
            ax.bar(x + (o - 0.5) * w, get(f"gain_{c}"), width=w - 0.04, color=col, label=lab,
                   yerr=get(f"se_{c}"), error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        zero(ax)
        ax.set_ylabel("gain over taking the first\nride (median fares / choice)", fontsize=9, color=INK2)
        ax.set_xticks(x, [s[3] for s in SETTINGS], fontsize=8)
        if j == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        for a in axs[:, j]:
            style(a)
        out[k] = {"rides_per_menu": get("mean_options").tolist(), "first_best": get("first_is_best").tolist(),
                  "gain_rate": get("gain_rate_1.0").tolist(), "gain_learned": get("gain_learned").tolist()}
    fig.suptitle("Only a demand surge gives drivers a real choice (per-trip pay, batched exclusive menus)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "44_market.png", dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------- 2. training
def training_fig():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    out = {}
    for j, k in enumerate(KEYS):
        ax = axs[j]
        for tag, col, lab in [("drv2_big", OLD, "old market (about 1 ride per menu)"), ("drv2_d3", NEW, "demand x3 (about 3 rides per menu)")]:
            f = OUT / f"eval_{tag}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f).dropna(subset=["focal_net_gain_per_h"])
            g = d[d.key == k].groupby("iter").focal_net_gain_per_h
            m, se = g.mean(), g.std() / np.sqrt(g.count())
            ax.errorbar(m.index, m.values, yerr=se.values, color=col, lw=2, marker="o", ms=5, capsize=3, label=lab)
            out[f"{k}/{tag}"] = {int(i): [float(a), float(b)] for i, a, b in zip(m.index, m.values, se.values)}
        zero(ax)
        style(ax)
        ax.set_title(CITY_NAME[k], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("training round", fontsize=9, color=INK2)
        if j == 0:
            ax.set_ylabel("agent drivers' net pay vs taking\nthe first ride (median fares / h)", fontsize=9, color=INK2)
            ax.legend(fontsize=8, frameon=False, loc="center right")
    fig.suptitle("Training in the world model: 25% of drivers on the agent vs the same drivers taking the first ride",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "44_training.png", dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------- 3. simulator
def sim_records(k):
    R = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(cities.get(k).data_dir / "wm3_rl" / "exec7b" / f"rec_seed*_{AGENT}.json")))]
    return [r for r in R if "focal_net_gain_per_h" in r["arms"].get("mixed", {})]


def sim_fig():
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    wm = pd.read_csv(OUT / "eval_drv2_d3.csv").dropna(subset=["focal_net_gain_per_h"])
    out = {}
    for j, k in enumerate(KEYS):
        R = sim_records(k)
        if not R:
            continue
        ax = axs[0, j]
        rows = [("focal_net_gain_per_h", "agent drivers"), ("others_net_gain_per_h", "everyone else")]
        sim = {c: np.array([r["arms"]["mixed"][c] for r in R]) for c, _ in rows}
        w = wm[(wm.key == k) & (wm.iter == 20)]
        for i, (c, lab) in enumerate(rows):
            v = sim[c]
            ax.bar(i - 0.18, v.mean(), width=0.34, color=NEW, yerr=v.std(ddof=1) / np.sqrt(len(v)),
                   error_kw={"ecolor": INK2, "lw": 1, "capsize": 3}, label="simulator (6 unseen days)" if i == 0 else None)
            ax.scatter(np.full(len(v), i - 0.18) + np.linspace(-0.08, 0.08, len(v)), v, s=14, color=INK, zorder=3,
                       label="one day" if i == 0 else None)
            u = w[c]
            ax.bar(i + 0.18, u.mean(), width=0.34, color=OLD, yerr=u.std() / np.sqrt(len(u)),
                   error_kw={"ecolor": INK2, "lw": 1, "capsize": 3}, label="world model's prediction" if i == 0 else None)
        zero(ax)
        style(ax)
        ax.set_xticks([0, 1], [lab for _, lab in rows], fontsize=9)
        ax.set_title(CITY_NAME[k], fontsize=11, color=INK, loc="left")
        if j == 0:
            ax.set_ylabel("net pay vs all taking the first\nride (median fares / h)", fontsize=9, color=INK2)
            ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax = axs[1, j]
        A = {a: {m: np.mean([r["arms"][a][m] for r in R]) for m in ("pickups", "wait_min", "pay_net_per_h")} for a in ("single", "asap", "agents")}
        mets = [("pickups", "riders picked up"), ("wait_min", "rider wait"), ("pay_net_per_h", "drivers' net pay / h")]
        xs = np.arange(len(mets))
        for o, (a, col, lab) in enumerate([("single", ALT, "platform offers one ride at a time"), ("agents", NEW, "every driver uses the agent")]):
            ch = [100 * (A[a][m] / A["asap"][m] - 1) for m, _ in mets]
            ax.bar(xs + (o - 0.5) * 0.36, ch, width=0.32, color=col, label=lab)
        zero(ax)
        style(ax)
        ax.set_xticks(xs, [lab for _, lab in mets], fontsize=9)
        if j == 0:
            ax.set_ylabel("% change vs every driver\ntaking the first ride", fontsize=9, color=INK2)
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=2)
        out[k] = {"days": len(R), **{c: [float(sim[c].mean()), float(sim[c].std(ddof=1) / np.sqrt(len(R)))] for c, _ in rows}, "arms": A}
    fig.suptitle("Simulator test of the round-20 agent: half of idle drivers off and demand x3 from 12:00, 2 hours",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "44_simulator.png", dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------- report page
def report(M, T, S):
    spec = importlib.util.spec_from_file_location("r38", ROOT / "scripts" / "38_wm3_reports.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    spec = importlib.util.spec_from_file_location("t38", ROOT / "scripts" / "38_wm3_reports_text.py")
    X = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(X)
    cs = []
    for k in KEYS:
        rec = json.loads((cities.get(k).data_dir / "wm3_rl" / "exec7b" / f"rec_seed700_{AGENT}.json").read_text())
        t0 = float(rec["t0"])
        geo = R.Geo(k)
        base = cities.get(k).data_dir / "wm3_rl" / "exec7b"
        A, _ = R.real_run(geo, base / f"seed700_asap_{AGENT}" / rec["date"], t0, "Every driver takes the first ride",
                          "menus of up to 6 rides; each driver takes the platform's first option at once")
        B, _ = R.real_run(geo, base / f"seed700_agents_{AGENT}" / rec["date"], t0, "Every driver uses the trained agent",
                          "same menus; the agent picks an option, declines or moves (trained only in the world model)")
        for r in (A, B):
            r["banner"] = f"12:00: {rec['arms']['asap']['removed']} idle drivers logged off; requests x3 from now on"
            r["bannerUntil"] = 20
        cur, dec = R.CUR[k]
        cs.append({"key": k, "name": CITY_NAME[k], "cur": cur, "dec": dec, **geo.pack(),
                   "pairs": [{"label": "take the first ride vs the agent", "A": A, "B": B,
                              "note": "Same unseen day, same riders, same drivers off the road. Watch the riders picked up: with the agent, "
                                      "drivers pass on rides that pay little per minute, so more short rides are served."}]})
        print(f"  {k} packed", flush=True)
    replay = {"layers": [["offers", "rides taken"], ["moves", "move orders"], ["riders", "waiting riders"], ["events", "declined prices"]],
              "legend": X.LEGEND_BASE + X.LEG_OFFER, "howto": X.HOWTO,
              "mini": [{"key": "pickups", "label": "riders picked up"}, {"key": "waiting", "label": "riders waiting now"},
                       {"key": "cancels", "label": "riders who gave up"}, {"key": "online", "label": "drivers online"}]}
    data = {"dt": 30, "unit": 2.0, "cities": cs, **replay}
    f = lambda v: f"{v[0]:+.2f} &plusmn; {v[1]:.2f}"  # noqa: E731
    sf, pu = S["sf_hd"], S["pune_hd"]
    tr = {k: T.get(f"{k}/drv2_d3", {}) for k in KEYS}
    intro = f"""
<h1>Stage 7b: when is "take the first ride" not a driver's best choice?</h1>
<p class="sub">World model 3 · ride_world · San Francisco and Pune</p>
<p>In stage 7 every trained driver agent ended up doing what a plain driver does: take the first ride offered, at once. Stage 7b asks
what the market must look like for a smarter driver to earn more, and gives the drivers richer minds: each has its own income target,
shift length, pull towards home and cost per km, and can take any of up to 6 rides on its menu, decline, or drive itself to another area.
The policy is a 591,000-weight attention network trained only inside the world model; the result is then checked in the simulator.</p>
<div class="short"><b>In short</b><ul>
<li><span class="good">Done: the agent beats taking the first ride in the simulator</span>, on 6 of 6 unseen days in each city.
 Its drivers earn {f(sf['focal_net_gain_per_h'])} median fares per online hour more in San Francisco and {f(pu['focal_net_gain_per_h'])} in Pune.</li>
<li>The key was <b>more riders than drivers</b>. Pay schemes, private or batched menus and taking half the idle drivers off the road
 still left about one ride per menu, and with one ride there is nothing to choose. With demand x3 a San Francisco menu holds about 3 rides
 and the first is the best only 40% of the time.</li>
<li>When every driver uses the agent, more riders are picked up (San Francisco +{100 * (sf['arms']['agents']['pickups'] / sf['arms']['asap']['pickups'] - 1):.0f}%,
 Pune +{100 * (pu['arms']['agents']['pickups'] / pu['arms']['asap']['pickups'] - 1):.0f}%), but riders wait a little longer
 (+{sf['arms']['agents']['wait_min'] - sf['arms']['asap']['wait_min']:.1f} and +{pu['arms']['agents']['wait_min'] - pu['arms']['asap']['wait_min']:.1f} min).</li>
<li>Drivers who keep taking the first ride lose a little ({f(sf['others_net_gain_per_h'])} and {f(pu['others_net_gain_per_h'])} per hour).
 The world model got the direction of every effect right, but overstated the San Francisco gain (it predicted {f(tr['sf_hd'].get(20, [np.nan, np.nan]))}).</li>
</ul></div>
<h3>How it was done</h3>
{X.flow([("1. Separate pay", "driver pay apart from rider fares, 8 schemes, all paying drivers the same total"),
         ("2. Probe", "branch one driver's choice in the world model: decline or each ride, 45 imagined minutes each"),
         ("3. Pick a market", "the setting where choosing beats the first ride: half the idle drivers off, demand x3, per-trip pay"),
         ("4. Train", "GRPO in the world model, 25% of drivers on the agent, 40 rounds (stopped at 33)"),
         ("5. Test", "round-20 agent in the simulator, 6 unseen days per city, 12:00-14:00")])}
"""
    body = f"""
<h2>The figures, explained</h2>
{X.fig(R.png(OUT / "44_market.png"), "1. The market: when do drivers have a real choice?",
       ["Each column is a market design, adding one change at a time from left to right (probe in the world model, 60 SF / 40 Pune branched choices each).",
        "Top: rides on a driver's menu. Middle: how often taking the first ride was the best branch. Bottom: what a driver gains per choice by choosing instead."],
       ["Until demand is raised, menus hold about one ride and choosing gains nothing beyond noise.",
        "With demand x3 in San Francisco the simple rule 'take the ride that pays most per minute' gains about 0.3 fares per choice."])}
{X.fig(R.png(OUT / "44_training.png"), "2. Training in the world model",
       ["Grey: the old market (half the idle drivers off, no surge). Orange: demand x3.",
        "Each point: the gain of the 25% of drivers on the agent over the same drivers taking the first ride, on 12 held-out starts (&plusmn; s.e.)."],
       ["In the old market training never beats zero. With demand x3 it does by round 10 and levels off by round 20.",
        "Rounds after 11 continue from saved weights after a reboot; training stopped at round 33 (GPU memory)."])}
{X.fig(R.png(OUT / "44_simulator.png"), "3. The simulator test",
       ["Top: the gain of the agent's drivers and the loss of everyone else, per day (dots) and on average, next to the world model's prediction.",
        "Bottom: riders and drivers when the platform offers single rides or every driver uses the agent, against every driver taking the first ride."],
       ["The agent wins on every unseen day in both cities.",
        "The world model is right about direction, too optimistic about the size of the gain in San Francisco, and too mild about the cost to others."])}
<h2>Replay: one simulator day</h2>
<p>The same 12:00 state, the same riders and the same drivers off the road. Left: every driver takes the first ride. Right: every driver uses the agent.</p>
"""
    R.page(data, "Stage 7b - Smarter drivers", intro, body, OUT / "stage7b_report.html")


def main():
    M = market_fig()
    T = training_fig()
    S = sim_fig()
    (OUT / "44_summary.json").write_text(json.dumps({"market": M, "training": T, "simulator": S}, indent=1, default=float))
    print("wrote figures", flush=True)
    report(M, T, S)


if __name__ == "__main__":
    main()
