"""Stage 4b: compare the dispatch methods run by 04_papers.py.

    .venv/bin/python scripts/04b_compare.py

Reads outputs/04_results.csv and writes
    outputs/04_tradeoff.png    each method as a point on three trade-offs, per city
    outputs/04_dashboard.html  interactive: choose axes and city, hover, sortable table,
                               links to every method's replay
    outputs/04_results.md      full tables per city
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D

from world import viz
from world import cities
from world.cities import OUTPUTS

LABEL = {
    "nearest": "Baseline (nearest)",
    "p1_greedy": "P1 Greedy", "p1_reassign": "P1 REASSIGN", "p1_laf": "P1 LAF",
    "p1_brp": "P1 Bal. Ride-Pooling", "p1_momaql": "P1 MOMAQL ★",
    "p2_nm": "P2 NM", "p2_wdf": "P2 WDF", "p2_laf": "P2 LAF", "p2_ilp": "P2 ILP", "p2_sid": "P2 SID",
    "p2_vfdcfmvd": "P2 VFDCFMVD ★",
}
ORDER = list(LABEL)
PROPOSED = {"p1_momaql", "p2_vfdcfmvd"}
FAMILY_COL = {"baseline": "#52514e", "paper 1 (Kang et al.)": "#2a78d6", "paper 2 (Shi et al.)": "#eb6834"}
CITY_NAME = {k: cities.CITIES[k].name for t in cities.TIERS.values() for k in t}
TIER_COL = {"core": "#0b0b0b", "metro": "#4a3aa7", "medium": "#1baf7a", "small": "#eda100"}


def family(m):
    return "baseline" if m == "nearest" else ("paper 1 (Kang et al.)" if m.startswith("p1") else "paper 2 (Shi et al.)")


# (x column, label, higher is better?), (y ...), title
PANELS = [
    (("completion_rate", "riders served (completion rate)", True),
     ("gini_net_earnings", "driver earnings inequality (Gini)", False),
     "Overall: serve riders vs treat drivers fairly"),
    (("p1_total_utility_km", "P1 total utility π (km)", True),
     ("p1_F_hat", "P1 unfairness F̂ = σ/μ of utility", False),
     "Paper 1's own trade-off"),
    (("p2_order_service_rate", "P2 order service rate", True),
     ("p2_unfairness_per_driver", "P2 unfairness per driver", False),
     "Paper 2's own trade-off"),
]


def tradeoff_png(res: pd.DataFrame):
    cities = [c for c in ("sf", "pune") if c in set(res.city)]
    fig, axs = plt.subplots(len(cities), 3, figsize=(20, 6.2 * len(cities)), layout="constrained", squeeze=False)
    for r, c in enumerate(cities):
        g = res[res.city == c]
        for k, ((xc, xl, _), (yc, yl, _), title) in enumerate(PANELS):
            ax = axs[r, k]
            base = g[g.method == "nearest"]
            if len(base):
                ax.axvline(base[xc].iloc[0], color=viz.MUTED, lw=0.8, ls="--", zorder=1)
                ax.axhline(base[yc].iloc[0], color=viz.MUTED, lw=0.8, ls="--", zorder=1)
            texts = []
            for _, row in g.iterrows():
                col = FAMILY_COL[family(row.method)]
                big = row.method in PROPOSED
                ax.scatter(row[xc], row[yc], s=150 if big else 60, color=col, zorder=3,
                           edgecolor=viz.INK if big else viz.SURFACE, lw=1.6 if big else 1)
                texts.append(ax.text(row[xc], row[yc], LABEL[row.method], fontsize=8.5, color=viz.INK_2,
                                     fontweight="bold" if big else "normal"))
            ax.invert_yaxis()          # lower y is better: flip so better = up and to the right
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=viz.MUTED, lw=0.5))
            ax.set_xlabel(f"{xl}   (higher is better →)")
            ax.set_ylabel(f"{yl}   (lower is better ↑)")
            ax.grid(color=viz.HAIRLINE, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_title(f"{CITY_NAME[c]} · {title}   (better ↗)")
    handles = [Line2D([], [], marker="o", ls="", color=v, markersize=9, label=k) for k, v in FAMILY_COL.items()]
    handles.append(Line2D([], [], marker="o", ls="", color="white", markeredgecolor=viz.INK, markersize=12,
                          label="★ the paper's proposed method"))
    handles.append(Line2D([], [], ls="--", color=viz.MUTED, label="baseline's level"))
    fig.legend(handles=handles, loc="lower center", ncols=5, frameon=False, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Which dispatcher is best? Each dot is one method on the same test day. "
                 "Up and to the right is better in every panel.", x=0.01, ha="left", fontweight="bold")
    viz.save(fig, OUTPUTS / "04_tradeoff.png")


METRIC_INFO = [
    # key, label, higher-is-better, format, group
    ("completion_rate", "riders served (completion rate)", True, "pct", "service"),
    ("mean_wait_min", "mean pickup wait (min)", False, "f2", "service"),
    ("cancel_rate", "cancellation rate", False, "pct", "service"),
    ("offer_acceptance", "driver offer acceptance", True, "pct", "service"),
    ("mean_hourly", "driver net earnings per online hour", True, "f1", "drivers"),
    ("gini_net_earnings", "driver earnings inequality (Gini)", False, "f3", "drivers"),
    ("p10_hourly", "10th-percentile driver net per hour", True, "f1", "drivers"),
    ("p1_total_utility_km", "P1 total utility π (km)", True, "f0", "paper 1"),
    ("p1_F_hat", "P1 unfairness F̂ = σ/μ", False, "f3", "paper 1"),
    ("p1_F_var", "P1 unfairness F = Var", False, "f0", "paper 1"),
    ("p2_order_service_rate", "P2 order service rate", True, "pct", "paper 2"),
    ("p2_unfairness_per_driver", "P2 unfairness per driver", False, "f3", "paper 2"),
    ("p2_total_income", "P2 total driver income", True, "f0", "paper 2"),
    ("p2_idle_driver_rate", "P2 idle driver rate", False, "pct", "paper 2"),
    ("policy_seconds", "policy compute for the day (s)", False, "f1", "cost"),
]


def dashboard(res: pd.DataFrame):
    rows = []
    for _, r in res.iterrows():
        rows.append({"city": r.city, "method": r.method, "label": LABEL[r.method], "family": family(r.method),
                     "proposed": r.method in PROPOSED,
                     **{k: (None if pd.isna(r[k]) else float(r[k])) for k, *_ in METRIC_INFO}})
    data = {"rows": rows, "metrics": [dict(zip(("key", "label", "up", "fmt", "group"), m)) for m in METRIC_INFO],
            "colours": FAMILY_COL, "cities": CITY_NAME, "order": ORDER}
    (OUTPUTS / "04_dashboard.html").write_text(DASH.replace("__DATA__", json.dumps(data)))
    print("  wrote outputs/04_dashboard.html")


def allcities_png(res: pd.DataFrame):
    """Each method vs the baseline in every city: change in riders served (x) and in
    driver earnings Gini (y, down = fairer)."""
    base = res[res.method == "nearest"].set_index("city")
    meths = [m for m in ORDER if m != "nearest" and m in set(res.method)]
    fig, axs = plt.subplots(3, 4, figsize=(18, 13), layout="constrained", sharex=True, sharey=True)
    for ax, m in zip(axs.flat, meths):
        g = res[res.method == m].set_index("city")
        dx = (g.completion_rate - base.completion_rate.reindex(g.index)) * 100
        dy = g.gini_net_earnings - base.gini_net_earnings.reindex(g.index)
        ax.axhline(0, color=viz.MUTED, lw=0.8, ls="--"); ax.axvline(0, color=viz.MUTED, lw=0.8, ls="--")
        for c in g.index:
            ax.scatter(dx[c], dy[c], s=40, color=TIER_COL[cities.CITIES[c].tier], edgecolor=viz.SURFACE, lw=0.8, zorder=3)
        ax.set_title(f"{LABEL[m]}: median {dx.median():+.1f} pts served, Gini {dy.median():+.3f}", fontsize=10)
        ax.grid(color=viz.HAIRLINE, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("riders served vs baseline (percentage points)")
    for ax in axs[:, 0]:
        ax.set_ylabel("earnings Gini vs baseline (down = fairer)")
    handles = [Line2D([], [], marker="o", ls="", color=v, markersize=8, label=f"{k} cities") for k, v in TIER_COL.items()]
    fig.legend(handles=handles, loc="lower center", ncols=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Every method against the nearest-cab baseline in {res.city.nunique()} cities "
                 "(one dot per city; bottom right would beat the baseline on both)", x=0.01, ha="left",
                 fontweight="bold")
    viz.save(fig, OUTPUTS / "04_allcities.png")


def main():
    res = pd.read_csv(OUTPUTS / "04_results.csv")
    res = res[res.method.isin(ORDER)]
    tradeoff_png(res)
    allcities_png(res)
    dashboard(res)
    lines = ["# Stage 4 results\n"]
    for c in [c for c in CITY_NAME if c in set(res.city)]:
        g = res[res.city == c].set_index("method")
        g = g.loc[[m for m in ORDER if m in g.index]]
        t = pd.DataFrame({"method": [LABEL[m] for m in g.index]})
        for k, lab, up, fmt, _ in METRIC_INFO:
            t[f"{lab} {'↑' if up else '↓'}"] = [f"{x:.1%}" if fmt == "pct" else f"{x:,.{fmt[1]}f}"
                                                for x in g[k].to_numpy()]
        lines.append(f"\n## {CITY_NAME[c]}\n\n" + t.to_markdown(index=False) + "\n")
    (OUTPUTS / "04_results.md").write_text("\n".join(lines))
    old = OUTPUTS / "04_comparison.png"
    if old.exists():
        old.unlink()        # replaced by 04_tradeoff.png and the dashboard
    print("  wrote outputs/04_results.md")


DASH = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dispatch methods compared</title>
<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--good:#006300}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2);margin:26px 0 8px}
.muted{color:var(--muted)} .row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button,select{font:inherit;padding:5px 10px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink);cursor:pointer}
button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.card{border:1px solid var(--line);border-radius:10px;padding:14px;background:#fff}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
svg text{font:12px system-ui,sans-serif;fill:var(--ink2)}
.tip{position:fixed;pointer-events:none;background:#0b0b0b;color:#fff;padding:8px 10px;border-radius:6px;font-size:12px;display:none;z-index:9;max-width:320px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{cursor:pointer;color:var(--ink2);font-weight:600;vertical-align:bottom;white-space:normal;min-width:84px}
th:first-child,td:first-child{text-align:left}
td.best{color:var(--good);font-weight:700} tr:hover td{background:#f5f4f0}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.tablewrap{overflow-x:auto}
a{color:#2a78d6}
.legend div{margin:4px 0}
.insight li{margin:5px 0}
</style></head><body><main>
<h1>Which dispatcher is best for the ride company?</h1>
<div class="muted">The baseline and 11 methods from the two reproduced papers each dispatched the same simulated
test day (Tue 4 Mar 2025): same drivers, riders, requests and traffic. Learning methods first trained on 3 other days.</div>

<h2>City</h2>
<div class="row" id="cities"></div>

<h2>Trade-off</h2>
<div class="grid">
  <div class="card">
    <div class="row" style="margin-bottom:8px">
      x <select id="xsel"></select> y <select id="ysel"></select>
    </div>
    <svg id="plot" width="100%" height="470"></svg>
  </div>
  <div class="card">
    <div class="legend" id="legend"></div>
    <h2 style="margin-top:14px">Reading this chart</h2>
    <ul class="insight" style="padding-left:18px;margin:0" id="read"></ul>
  </div>
</div>

<h2>All metrics <span class="muted" style="text-transform:none;letter-spacing:0">(click a column to sort; green = best in column)</span></h2>
<div class="card tablewrap"><table id="tbl"></table></div>
</main>
<div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const M = Object.fromEntries(D.metrics.map(m => [m.key, m]));
let city = 'sf', xk = 'completion_rate', yk = 'gini_net_earnings', sortK = null, sortUp = false;
const fmt = (m, v) => v == null ? '–' : m.fmt === 'pct' ? (100 * v).toFixed(1) + '%' : v.toLocaleString(undefined, {maximumFractionDigits: +m.fmt[1], minimumFractionDigits: +m.fmt[1]});
const tip = document.getElementById('tip');
const rows = () => D.rows.filter(r => r.city === city).sort((a, b) => D.order.indexOf(a.method) - D.order.indexOf(b.method));

function selects() {
  for (const [id, cur] of [['xsel', xk], ['ysel', yk]]) {
    const s = document.getElementById(id); s.innerHTML = '';
    let grp = null, og = null;
    D.metrics.forEach(m => { if (m.group !== grp) { og = document.createElement('optgroup'); og.label = m.group; s.appendChild(og); grp = m.group; }
      const o = new Option(`${m.label} ${m.up ? '↑' : '↓'}`, m.key); o.selected = m.key === cur; og.appendChild(o); });
  }
}
document.getElementById('xsel').onchange = e => { xk = e.target.value; draw(); };
document.getElementById('ysel').onchange = e => { yk = e.target.value; draw(); };
document.getElementById('cities').innerHTML = Object.entries(D.cities).map(([k, v]) => `<button data-c="${k}" class="${k === city ? 'on' : ''}">${v}</button>`).join('');
document.querySelectorAll('#cities button').forEach(b => b.onclick = () => { city = b.dataset.c; document.querySelectorAll('#cities button').forEach(x => x.classList.toggle('on', x === b)); draw(); });
document.getElementById('legend').innerHTML = Object.entries(D.colours).map(([k, c]) => `<div><span class="sw" style="background:${c}"></span>${k}</div>`).join('') +
  '<div><span class="sw" style="background:#fff;border:2px solid #0b0b0b"></span>★ the paper\'s proposed method</div>' +
  '<div><span style="display:inline-block;width:14px;border-top:1.5px dashed #898781;margin-right:6px;vertical-align:middle"></span>baseline\'s level</div>';

function draw() {
  const R = rows(), svg = document.getElementById('plot');
  const W = svg.clientWidth, H = 470, L = 74, Rt = 130, T = 18, B = 50;
  const mx = M[xk], my = M[yk];
  const xs = R.map(r => r[xk]), ys = R.map(r => r[yk]);
  const pad = (a, b) => { const d = (b - a) || Math.abs(a) || 1; return [a - 0.08 * d, b + 0.08 * d]; };
  const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
  // better = right and up: an axis is flipped when lower is better
  const X = v => L + (W - L - Rt) * (mx.up ? (v - x0) : (x1 - v)) / (x1 - x0);
  const Y = v => T + (H - T - B) * (my.up ? (y1 - v) : (v - y0)) / (y1 - y0);
  let g = '';
  for (let k = 0; k <= 4; k++) {
    const vx = x0 + (x1 - x0) * k / 4, vy = y0 + (y1 - y0) * k / 4;
    g += `<line x1="${X(vx)}" x2="${X(vx)}" y1="${T}" y2="${H - B}" stroke="#e1e0d9"/><text x="${X(vx)}" y="${H - B + 16}" text-anchor="middle">${fmt(mx, vx)}</text>`;
    g += `<line x1="${L}" x2="${W - Rt}" y1="${Y(vy)}" y2="${Y(vy)}" stroke="#e1e0d9"/><text x="${L - 6}" y="${Y(vy) + 4}" text-anchor="end">${fmt(my, vy)}</text>`;
  }
  g += `<text x="${(L + W - Rt) / 2}" y="${H - 8}" text-anchor="middle" style="fill:#0b0b0b">${mx.label} — ${mx.up ? 'higher' : 'lower'} is better →</text>`;
  g += `<text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle" style="fill:#0b0b0b">${my.label} — ${my.up ? 'higher' : 'lower'} is better ↑</text>`;
  g += `<text x="${W - Rt - 4}" y="${T + 12}" text-anchor="end" style="fill:#006300;font-weight:700">better ↗</text>`;
  const base = R.find(r => r.method === 'nearest');
  if (base) {
    g += `<rect x="${X(base[xk])}" y="${T}" width="${Math.max(0, W - Rt - X(base[xk]))}" height="${Math.max(0, Y(base[yk]) - T)}" fill="#006300" opacity=".05"/>`;
    g += `<line x1="${X(base[xk])}" x2="${X(base[xk])}" y1="${T}" y2="${H - B}" stroke="#898781" stroke-dasharray="4 4"/><line x1="${L}" x2="${W - Rt}" y1="${Y(base[yk])}" y2="${Y(base[yk])}" stroke="#898781" stroke-dasharray="4 4"/>`;
  }
  const pts = R.map(r => ({r, x: X(r[xk]), y: Y(r[yk])})).sort((a, b) => a.y - b.y);
  const placed = [];
  pts.forEach(p => { let ly = p.y + 4; for (const q of placed) if (Math.abs(q.lx - (p.x + 11)) < 120 && Math.abs(q.ly - ly) < 14) ly = q.ly + 14; p.ly = ly; p.lx = p.x + 11; placed.push(p); });
  pts.forEach(p => {
    const c = D.colours[p.r.family], big = p.r.proposed;
    if (Math.abs(p.ly - p.y - 4) > 2) g += `<line x1="${p.x}" y1="${p.y}" x2="${p.lx - 2}" y2="${p.ly - 4}" stroke="#c3c2b7"/>`;
    g += `<circle cx="${p.x}" cy="${p.y}" r="${big ? 9 : 6.5}" fill="${c}" stroke="${big ? '#0b0b0b' : '#fff'}" stroke-width="${big ? 2 : 1.5}" data-m="${p.r.method}" style="cursor:pointer"/>`;
    g += `<text x="${p.lx}" y="${p.ly}" style="fill:#0b0b0b;${big ? 'font-weight:700' : ''}">${p.r.label}</text>`;
  });
  svg.innerHTML = g;
  svg.querySelectorAll('circle[data-m]').forEach(el => {
    const r = R.find(x => x.method === el.dataset.m);
    el.onmousemove = ev => { tip.style.display = 'block'; tip.style.left = ev.clientX + 14 + 'px'; tip.style.top = ev.clientY + 14 + 'px';
      tip.innerHTML = `<b>${r.label}</b><br>` + D.metrics.slice(0, 7).map(m => `${m.label}: ${fmt(m, r[m.key])}`).join('<br>') + '<br><i>click to watch its replay</i>'; };
    el.onmouseleave = () => tip.style.display = 'none';
    el.onclick = () => window.open(`${city}/04_replay_${r.method}.html`, '_blank');
  });
  read(R); table(R);
}

function read(R) {
  const base = R.find(r => r.method === 'nearest'), mx = M[xk], my = M[yk];
  const better = (m, a, b) => m.up ? a > b : a < b;
  const dom = R.filter(r => r.method !== 'nearest' && better(mx, r[xk], base[xk]) && better(my, r[yk], base[yk])).map(r => r.label);
  const bx = R.reduce((a, b) => better(mx, b[xk], a[xk]) ? b : a), by = R.reduce((a, b) => better(my, b[yk], a[yk]) ? b : a);
  document.getElementById('read').innerHTML = [
    `Each dot is one dispatcher. The dashed lines mark the <b>baseline</b> (nearest free cab). The green box is where a method beats it on <i>both</i> axes.`,
    dom.length ? `In the green box: <b>${dom.join(', ')}</b>.` : `The green box is empty: every method that improves one axis loses on the other.`,
    `Best on x: <b>${bx.label}</b> (${fmt(mx, bx[xk])}).<br>Best on y: <b>${by.label}</b> (${fmt(my, by[yk])}).`,
    `Click a dot or "watch" to see that method's cabs move (17:00–20:00).`,
  ].map(s => `<li>${s}</li>`).join('');
}

function table(R) {
  const ms = D.metrics;
  const rs = [...R];
  if (sortK) rs.sort((a, b) => sortUp ? a[sortK] - b[sortK] : b[sortK] - a[sortK]);
  const best = Object.fromEntries(ms.map(m => [m.key, (m.up ? Math.max : Math.min)(...R.map(r => r[m.key]))]));
  let h = '<tr><th>method</th>' + ms.map(m => `<th data-k="${m.key}">${m.label} ${m.up ? '↑' : '↓'}</th>`).join('') + '<th>replay</th></tr>';
  rs.forEach(r => { h += `<tr><td><span class="sw" style="background:${D.colours[r.family]}"></span>${r.label}</td>` +
    ms.map(m => `<td class="${r[m.key] === best[m.key] ? 'best' : ''}">${fmt(m, r[m.key])}</td>`).join('') +
    `<td><a href="${city}/04_replay_${r.method}.html" target="_blank">▶ watch</a></td></tr>`; });
  const t = document.getElementById('tbl'); t.innerHTML = h;
  t.querySelectorAll('th[data-k]').forEach(th => th.onclick = () => { sortUp = sortK === th.dataset.k ? !sortUp : !M[th.dataset.k].up; sortK = th.dataset.k; table(rows()); });
}
selects(); draw(); window.addEventListener('resize', draw);
</script></body></html>
"""

if __name__ == "__main__":
    main()
