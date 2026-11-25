"""Stage 7b, part 2: does a bigger driver policy earn more? 0.6M vs 30M vs 60M weights.

    .venv/bin/python scripts/46_wm3_driver_size_report.py

Reads outputs/wm3/stage7b/{train,eval}_<tag>.csv written by scripts/42 for drv2_d3 (0.6M), drv2_30m,
drv2_60m and the two 30M runs that collapsed (drv2_30m_collapsed, drv2_30m_collapsed2). Writes
    46_gain.png       gain of the agent's drivers over taking the first ride, by round, per model
    46_training.png   how the policies changed while training (first-ride share, distance from start)
    46_collapse.png   the three 30M runs: why the first two collapsed and what fixed it
    46_outcomes.png   round 30, every driver on the agent: riders served, waits, moves, other drivers
    46_summary.csv    the numbers behind the figures
    stage7b_sizes.html  the figures with explanations (self-contained)
"""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wm3" / "stage7b"
CITY = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# colour + marker per model, so identity never rests on colour alone
MODELS = [("drv2_d3", "0.6M", "#2a6fdb", "o"), ("drv2_30m", "30M", "#c2410c", "s"), ("drv2_60m", "60M", "#15803d", "^")]
RUNS30 = [("drv2_30m_collapsed", "run 1: plain PPO clip, lr 3e-4", "#9ca3af", "x"),
          ("drv2_30m_collapsed2", "run 2: + bounded surrogate, lr 3e-4", "#7c3aed", "D"),
          ("drv2_30m", "run 3: + lr 1e-4 and per-round KL limit", "#c2410c", "s")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6, axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def ev(tag):
    return pd.read_csv(OUT / f"eval_{tag}.csv")


def tr(tag):
    return pd.read_csv(OUT / f"train_{tag}.csv")


def gains(tag):
    d = ev(tag)
    d = d[d["mode"] == "mixed"]
    rows = []
    for (it, k), g in d.groupby(["iter", "key"]):
        x = g.focal_net_gain_per_h
        rows.append({"model": tag, "city": k, "round": int(it), "gain": x.mean(), "se": x.std() / len(x) ** 0.5,
                     "wins": int((x > 0).sum()), "n": len(x), "others": g.others_net_gain_per_h.mean(),
                     "others_se": g.others_net_gain_per_h.std() / len(g) ** 0.5})
    return pd.DataFrame(rows)


def label_ends(ax, ends, span_frac=0.07):
    """ends: [(x, y, text)]; nudge labels apart vertically so they never overlap."""
    ends = sorted(ends, key=lambda e: e[1])
    lo, hi = ax.get_ylim()
    ys = []
    for x, y, t in ends:
        ys.append(y if not ys else max(y, ys[-1] + span_frac * (hi - lo)))
    for (x, y, t), yy in zip(ends, ys):
        ax.annotate(t, (x, y), xytext=(x + 1.2, yy), textcoords="data", va="center", fontsize=8, color=INK)


# ---------------------------------------------------------------------- 1. gain by round
def gain_fig(G):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k in zip(axes, CITY):
        ends = []
        for tag, name, c, m in MODELS:
            s = G[(G.model == tag) & (G.city == k) & G["round"].isin([0, 10, 20, 30])].sort_values("round")
            ax.errorbar(s["round"], s.gain, yerr=s.se, color=c, lw=2, marker=m, ms=6, capsize=3, label=name)
            last = s.iloc[-1]
            ends.append((30, last.gain, f"{name}  {last.gain:+.3f}"))
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlim(-1, 39)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_xlabel("training round", fontsize=9)
        ax.set_title(CITY[k], fontsize=10, loc="left")
        style(ax)
        label_ends(ax, ends)
    axes[0].set_ylabel("net pay per online hour gained\nvs taking the first ride (median fares)", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", title="driver policy", title_fontsize=8)
    fig.suptitle("The gain of the 25% of drivers on the agent (mean ± s.e., 12 held-out starts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "46_gain.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- 2. training dynamics
def training_fig():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for tag, name, c, m in MODELS:
        t = tr(tag)
        for ax, k in zip(axes[:2], CITY):
            y = t[f"{k}/p_first"].rolling(3, min_periods=1).mean()
            ax.plot(t.iter, y, color=c, lw=2, label=name)
            ax.plot(t.iter.iloc[::5], y.iloc[::5], color=c, lw=0, marker=m, ms=5)
        axes[2].plot(t.iter, t.kl.clip(lower=1e-4), color=c, lw=2, label=name)
        axes[2].plot(t.iter.iloc[::5], t.kl.clip(lower=1e-4).iloc[::5], color=c, lw=0, marker=m, ms=5)
    for ax, k in zip(axes[:2], CITY):
        ax.set_ylim(0, 1)
        ax.set_title(f"{CITY[k]}: share of menus where the\npolicy's top choice is the first ride", fontsize=9, loc="left")
    axes[2].set_yscale("log")
    axes[2].set_title("distance from the starting policy\n(KL, log scale)", fontsize=9, loc="left")
    for ax in axes:
        ax.axvline(10, color=GRID, lw=1, ls="--")
        ax.set_xlabel("training round", fontsize=9)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[2].text(10.4, axes[2].get_ylim()[0] * 1.5, "60M: lr raised\nafter round 10", fontsize=7, color=INK2)
    fig.suptitle("How the policies changed while training (first-ride share: 3-round moving average)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "46_training.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- 3. the 30M collapses
def collapse_fig():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for tag, name, c, m in RUNS30:
        t = tr(tag)
        axes[0].plot(t.iter, t.surr.abs().clip(lower=1e-3), color=c, lw=2, marker=m, ms=3, label=name)
        axes[1].plot(t.iter, t.kl, color=c, lw=2, marker=m, ms=3, label=name)
        g = gains(tag)
        g = g[g.city == "sf_hd"].sort_values("round")
        axes[2].errorbar(g["round"], g.gain, yerr=g.se, color=c, lw=2, marker=m, ms=6, capsize=3, label=name)
    axes[0].set_yscale("log")
    axes[0].set_title("size of the training objective\n(|surrogate|, log scale)", fontsize=9, loc="left")
    axes[1].set_title("distance from the starting policy (KL)", fontsize=9, loc="left")
    axes[2].set_title("San Francisco: gain of the agent's drivers\n(held-out evaluation)", fontsize=9, loc="left")
    axes[2].axhline(0, color=INK, lw=0.8)
    axes[2].set_xticks([0, 10, 20, 30])
    for ax in axes:
        ax.set_xlabel("training round", fontsize=9)
        style(ax)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("30M policy: two collapses and the fix (all three runs are identical up to round 10)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "46_collapse.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- 4. round-30 outcomes
def outcomes(tag):
    d = ev(tag)
    last = d.iter.max()
    d = d[d.iter == last]
    rows = []
    for k, g in d.groupby("key"):
        a = g[g["mode"] == "asap"].set_index("start")
        b = g[g["mode"] == "agents"].set_index("start")
        mx = g[g["mode"] == "mixed"]
        rows.append({"model": tag, "city": k, "round": int(last),
                     "pickups_pct": 100 * (b.pickups / a.pickups - 1).mean(),
                     "pickups_pct_se": 100 * (b.pickups / a.pickups - 1).std() / len(a) ** 0.5,
                     "wait_dmin": (b.wait_min - a.wait_min).mean(), "wait_dmin_se": (b.wait_min - a.wait_min).std() / len(a) ** 0.5,
                     "moves": b.self_moves.mean(), "declined": b.declined.mean(),
                     "others": mx.others_net_gain_per_h.mean(), "others_se": mx.others_net_gain_per_h.std() / len(mx) ** 0.5})
    return rows


def outcomes_fig(O):
    cols = [("pickups_pct", "riders picked up\n(% vs everyone takes the first ride)", "%+.1f%%"),
            ("wait_dmin", "rider wait for pickup\n(min, vs everyone takes the first ride)", "%+.2f"),
            ("others", "drivers still taking the first ride\n(net pay per hour, 25% on the agent)", "%+.3f")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    w = 0.26
    for ax, (col, title, fmt) in zip(axes, cols):
        for j, (tag, name, c, m) in enumerate(MODELS):
            s = O[O.model == tag].set_index("city").loc[list(CITY)]
            x = np.arange(len(CITY)) + (j - 1) * w
            bars = ax.bar(x, s[col], w - 0.03, yerr=s[col + "_se"], color=c, label=name, capsize=2,
                          error_kw={"lw": 0.8, "ecolor": INK2})
            ax.bar_label(bars, labels=[fmt % v for v in s[col]], fontsize=7, padding=2, color=INK)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(CITY)))
        ax.set_xticklabels(CITY.values(), fontsize=9)
        ax.set_title(title, fontsize=9, loc="left")
        ax.margins(y=0.18)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", title="every driver on", title_fontsize=8)
    fig.suptitle("Round 30: what the policies do to the city (mean ± s.e., 12 held-out starts, 60 imagined minutes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "46_outcomes.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- page
def img(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def page(G, O):
    g30 = G[G["round"] == 30].set_index(["model", "city"])
    f = lambda tag, k: f"{g30.loc[(tag, k), 'gain']:+.3f} &plusmn; {g30.loc[(tag, k), 'se']:.3f}"  # noqa: E731
    w = lambda tag, k: f"{g30.loc[(tag, k), 'wins']}/{g30.loc[(tag, k), 'n']}"  # noqa: E731
    rows = "".join(f"<tr><td>{name}</td><td>{f(tag, 'sf_hd')} <span class=m>({w(tag, 'sf_hd')})</span></td>"
                   f"<td>{f(tag, 'pune_hd')} <span class=m>({w(tag, 'pune_hd')})</span></td></tr>" for tag, name, _, _ in MODELS)

    def fig(name, title, what, read):
        return (f"<section><h2>{title}</h2><img src='{img(OUT / name)}' alt='{title}'>"
                f"<div class=two><div><h4>What it shows</h4><ul>{''.join(f'<li>{x}</li>' for x in what)}</ul></div>"
                f"<div><h4>How to read it</h4><ul>{''.join(f'<li>{x}</li>' for x in read)}</ul></div></div></section>")

    html = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Driver policy size</title><style>
:root{{--bg:#fbfaf7;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#fff;--good:#15803d}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--ink:#eceae4;--ink2:#a8a6a0;--line:#33322f;--card:#1e1e1c;--good:#4ade80}}
 img{{background:#fff;border-radius:6px}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:28px 16px 60px}}
h1{{font-size:26px;margin:0 0 4px}} .sub{{color:var(--ink2);margin:0 0 18px}}
h2{{font-size:18px;margin:34px 0 10px}} h4{{margin:6px 0;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2)}}
img{{width:100%;height:auto;display:block}} ul{{margin:4px 0;padding-left:20px}}
.short{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 18px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:20px}} @media (max-width:720px){{.two{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;margin:10px 0;font-variant-numeric:tabular-nums}} td,th{{border-bottom:1px solid var(--line);padding:6px 14px 6px 0;text-align:left}}
.m{{color:var(--ink2);font-size:13px}} .good{{color:var(--good);font-weight:600}} code{{font-size:13px}}
</style></head><body><main>
<h1>Does a bigger driver policy earn more?</h1>
<p class=sub>Stage 7b, part 2 · world model 3 · ride_world · San Francisco and Pune · same market as stage 7b (half the idle drivers off, demand &times;3, menus of up to 6 rides every 2 min)</p>
<div class=short><b>In short</b><ul>
<li><b>No.</b> The 60M-weight policy ties the 0.6M one in San Francisco and is worse in Pune; the 30M one is below both.</li>
<li>Round 30, net pay per online hour of the 25% of drivers on the agent, over the same drivers taking the first ride (wins out of 12 held-out starts):
<table><tr><th>policy</th><th>San Francisco</th><th>Pune</th></tr>{rows}</table></li>
<li>The large policies only train safely with small steps. The first two 30M runs <b>collapsed</b> into driving around empty
 (hundreds of self-moves) until the training objective was bounded, the learning rate lowered and each round's change capped.</li>
<li>Every policy serves more riders than taking the first ride in San Francisco; the 60M one serves fewer in Pune.
 Drivers still taking the first ride lose a little with every policy.</li>
<li><span class=good>The 0.6M policy stays the default</span>: as good or better, 50&ndash;100&times; cheaper per decision. The limit is the market and the reward, not the size of the network.</li>
</ul></div>
<h2>The policies</h2>
<table><tr><th>policy</th><th>weights</th><th>shape</th><th>learning rate</th><th>time per round</th></tr>
<tr><td>0.6M</td><td>591k</td><td>h 128, 4 layers, 8 heads</td><td>3e-3</td><td>~50 min (policy on the CPU)</td></tr>
<tr><td>30M</td><td>30.2M</td><td>h 768, 6 layers, 12 heads</td><td>3e-4 to round 10, then 1e-4</td><td>~27 min</td></tr>
<tr><td>60M</td><td>63.3M</td><td>h 768, 13 layers, 12 heads</td><td>7e-5 to round 10, then 2e-4</td><td>~32 min</td></tr></table>
<p class=m>All trained with GRPO inside the world model, every driver on the current policy, 24 SF + 16 Pune branched choices per round; the large ones decide on the GPU in half precision.
After the fixes, every run uses a bounded (dual-clip) surrogate and stops a round's epochs once it moved the policy more than KL 0.03.</p>
{fig("46_gain.png", "1. The gain, round by round",
     ["Each point: 12 held-out starts, 60 imagined minutes, 25% of drivers on the policy and the rest taking the first ride, compared with everyone taking the first ride (&plusmn; s.e.).",
      "Colour and marker shape both mark the policy."],
     ["Round 0 is the starting policy, close to taking the first ride, so the gain is about zero.",
      "The 60M line is flat until round 10 because its learning rate was too small; after raising it, it catches up in San Francisco.",
      "In Pune drivers have fewer rides to choose from (about 1.5 per menu), so every gain is about ten times smaller."])}
{fig("46_training.png", "2. How the policies changed while training",
     ["Left and middle: how often the first ride is the policy's most likely choice, from the training rollouts.",
      "Right: how far the policy has moved from where it started (KL divergence, log scale)."],
     ["Learning means moving away from the first ride: the lines fall as the policies learn to pick better-paying rides.",
      "The bigger the policy, the smaller the steps it can take safely, so it moves away later."])}
{fig("46_collapse.png", "3. The 30M collapses and the fix",
     ["Three runs of the same 30M policy with the same random numbers, which is why they agree up to round 10.",
      "Left: the size of the training objective. Middle: distance from the starting policy. Right: the held-out gain in San Francisco."],
     ["Run 1: the objective reached 10<sup>27</sup>. Every action is tried, and for one the policy had almost never picked, PPO's term is unbounded; it took over the update and the policy collapsed by round 30.",
      "Run 2: with the objective bounded it still collapsed, at round 20, because each round moved the big policy too far.",
      "Run 3: a lower learning rate and a cap on each round's change keep it healthy to round 30."])}
{fig("46_outcomes.png", "4. What the policies do to the city",
     ["Round 30. The first two panels put every driver on the policy; the third has 25% on the policy and shows the other 75%.",
      "All against the same start with every driver taking the first ride."],
     ["Choosing better rides serves more riders in San Francisco (+14 to +18%) at the cost of slightly longer waits.",
      "In Pune the 60M policy turns down so many rides (about 1,900 in the 60 imagined minutes, against about 270 for the 0.6M policy) that fewer riders are picked up.",
      "Drivers who keep taking the first ride lose a little: the agent takes the better rides first."])}
<h2>Files</h2><ul class=m>
<li>Figures: <code>outputs/wm3/stage7b/46_*.png</code>, numbers: <code>46_summary.csv</code>, this page: <code>stage7b_sizes.html</code> (made by <code>scripts/46_wm3_driver_size_report.py</code>)</li>
<li>Training and evaluation logs: <code>outputs/wm3/stage7b/{{train,eval}}_drv2_{{d3,30m,60m}}.csv</code>, collapsed runs: <code>*_drv2_30m_collapsed*.csv</code></li>
<li>Weights: <code>data/wm3_models/drv2_{{d3,30m,60m}}_it{{10,20,30}}.pt</code></li></ul>
</main></body></html>"""
    (OUT / "stage7b_sizes.html").write_text(html)


def main():
    G = pd.concat([gains(t) for t, *_ in MODELS] + [gains(t) for t, *_ in RUNS30[:2]])
    O = pd.DataFrame([r for t, *_ in MODELS for r in outcomes(t)])
    gain_fig(G)
    training_fig()
    collapse_fig()
    outcomes_fig(O)
    G.merge(O, on=["model", "city", "round"], how="left", suffixes=("", "_all")).round(4).to_csv(OUT / "46_summary.csv", index=False)
    page(G, O)
    print(O.round(3).to_string(index=False))
    print(f"wrote {OUT}/46_*.png and stage7b_sizes.html")


if __name__ == "__main__":
    main()
