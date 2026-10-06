"""Build outputs/index.html: one page to watch the cabs in every city.

    .venv/bin/python scripts/08_index.py

Pick a city and a dispatcher; the page shows that city's 17:00-20:00 replay (every cab a
dot with a trail) inline, with the city's key numbers and links to its map, traffic and
comparison explorers. The replays themselves stay as separate files and load one at a time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from world import cities

METHODS = {
    "baseline": ("Baseline: nearest cab + repositioning", "03_replay.html"),
    "p1_greedy": ("Paper 1 · Greedy", "04_replay_p1_greedy.html"),
    "p1_reassign": ("Paper 1 · REASSIGN", "04_replay_p1_reassign.html"),
    "p1_laf": ("Paper 1 · LAF", "04_replay_p1_laf.html"),
    "p1_brp": ("Paper 1 · Balance Ride-Pooling", "04_replay_p1_brp.html"),
    "p1_momaql": ("Paper 1 · MOMAQL (proposed)", "04_replay_p1_momaql.html"),
    "p2_nm": ("Paper 2 · NM", "04_replay_p2_nm.html"),
    "p2_wdf": ("Paper 2 · WDF", "04_replay_p2_wdf.html"),
    "p2_laf": ("Paper 2 · LAF", "04_replay_p2_laf.html"),
    "p2_ilp": ("Paper 2 · ILP", "04_replay_p2_ilp.html"),
    "p2_sid": ("Paper 2 · SID", "04_replay_p2_sid.html"),
    "p2_vfdcfmvd": ("Paper 2 · VFDCFMVD (proposed)", "04_replay_p2_vfdcfmvd.html"),
}


def main():
    out = cities.OUTPUTS
    summ = pd.read_csv(out / "cities_summary.csv").set_index("key")
    res = pd.read_csv(out / "04_results.csv")
    rows = []
    for tier, keys in cities.TIERS.items():
        for k in keys:
            if not (out / k / "03_replay.html").exists():
                continue
            s = summ.loc[k]
            per = {}
            for m, (_, f) in METHODS.items():
                r = res[(res.city == k) & (res.method == ("nearest" if m == "baseline" else m))]
                if (out / k / f).exists() and len(r):
                    r = r.iloc[0]
                    per[m] = {"completion": round(float(r.completion_rate), 3), "wait": round(float(r.mean_wait_min), 1),
                              "gini": round(float(r.gini_net_earnings), 3), "hourly": round(float(r.mean_hourly), 1)}
            rows.append({"key": k, "name": s["city"], "tier": tier, "country": s["country"],
                         "population": int(s["population"]), "area": int(s["area_km2"]), "road_km": float(s["road_km"]),
                         "zones": int(s["zones"]), "drivers": int(s["drivers"]), "requests": int(s["requests_day"]),
                         "currency": s["currency"], "peak_tti": float(s["peak_tti_target"]),
                         "extent": s["extent"], "methods": per})
    data = {"cities": rows, "methods": {m: v[0] for m, v in METHODS.items()},
            "files": {m: v[1] for m, v in METHODS.items()}}
    (out / "index.html").write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out / 'index.html'} with {len(rows)} cities")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ride world: all cities</title>
<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--surface);color:var(--ink);font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
#top{padding:10px 16px 8px;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 0 2px}
.muted{color:var(--muted)}
.tiers{display:flex;flex-wrap:wrap;gap:14px;margin-top:8px}
.tier .lab{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink2);margin-bottom:3px}
.row{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
button,select{font:inherit;padding:4px 9px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink);cursor:pointer}
button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.bar{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin-top:8px}
.stats{display:flex;flex-wrap:wrap;gap:4px 16px;font-variant-numeric:tabular-nums}
.stats b{font-weight:600}
a{color:#2a78d6;text-decoration:none} a:hover{text-decoration:underline}
#frame{position:absolute;left:0;right:0;bottom:0;border:0;width:100%}
</style></head><body>
<div id="top">
  <h1>Ride world: watch the cabs in 15 cities</h1>
  <div class="muted">Each dot is a cab with a short trail, 17:00–20:00 on a simulated Tuesday (4 Mar 2025). Grey idle, blue to pickup, orange with passenger, green repositioning; rings are waiting requests.</div>
  <div class="tiers" id="tiers"></div>
  <div class="bar">
    <label class="row">dispatcher <select id="method"></select></label>
    <div class="stats" id="stats"></div>
    <div class="row" id="links"></div>
  </div>
</div>
<iframe id="frame" title="cab replay"></iframe>
<script>
const D = __DATA__;
const TIERS = {core: 'core cities', metro: 'metro siblings', medium: 'medium cities', small: 'small cities'};
const sel = document.getElementById('method');
Object.entries(D.methods).forEach(([k, v]) => sel.add(new Option(v, k)));
const q = new URLSearchParams(location.hash.slice(1));
let city = q.get('city') || D.cities[0].key, method = q.get('m') || 'baseline';
document.getElementById('tiers').innerHTML = Object.entries(TIERS).map(([t, lab]) =>
  `<div class="tier"><div class="lab">${lab}</div><div class="row">` +
  D.cities.filter(c => c.tier === t).map(c => `<button data-c="${c.key}">${c.name}</button>`).join('') + `</div></div>`).join('');
document.querySelectorAll('#tiers button').forEach(b => b.onclick = () => { city = b.dataset.c; show(); });
sel.onchange = () => { method = sel.value; show(); };
const fmt = (v, d) => v.toLocaleString(undefined, {maximumFractionDigits: d});
function show() {
  const c = D.cities.find(x => x.key === city);
  if (!c.methods[method]) method = 'baseline';
  sel.value = method;
  document.querySelectorAll('#tiers button').forEach(b => b.classList.toggle('on', b.dataset.c === city));
  const m = c.methods[method], b = c.methods.baseline;
  const delta = (v, base, d, pct) => method === 'baseline' ? '' : ` <span class="muted">(${v - base >= 0 ? '+' : ''}${pct ? fmt(100 * (v - base), 1) + ' pts' : fmt(v - base, d)})</span>`;
  document.getElementById('stats').innerHTML = [
    `<span>${c.country === 'IN' ? 'India' : 'USA'} · pop. ${fmt(c.population / 1e6, 2)} M · ${c.area} km² (${c.extent}) · ${fmt(c.road_km, 0)} km road · ${c.zones} zones</span>`,
    `<span><b>${fmt(c.requests, 0)}</b> requests · <b>${c.drivers}</b> drivers</span>`,
    m ? `<span>served <b>${fmt(100 * m.completion, 1)}%</b>${delta(m.completion, b.completion, 1, true)}</span>` : '',
    m ? `<span>wait <b>${m.wait} min</b>${delta(m.wait, b.wait, 1)}</span>` : '',
    m ? `<span>driver net <b>${fmt(m.hourly, 1)} ${c.currency}/h</b>${delta(m.hourly, b.hourly, 1)}</span>` : '',
    m ? `<span>earnings Gini <b>${m.gini.toFixed(3)}</b>${delta(m.gini, b.gini, 3)}</span>` : '',
  ].join('');
  const f = `${c.key}/${D.files[method]}`;
  document.getElementById('links').innerHTML = `<a href="${f}" target="_blank">open replay ↗</a> · <a href="${c.key}/01_explorer.html" target="_blank">map</a> · <a href="${c.key}/02_explorer.html" target="_blank">traffic</a> · <a href="04_dashboard.html" target="_blank">method comparison</a> · <a href="wm/06_wm_viewer.html" target="_blank">world model</a>`;
  const fr = document.getElementById('frame');
  if (fr.dataset.src !== f) { fr.src = f; fr.dataset.src = f; }
  history.replaceState(null, '', `#city=${city}&m=${method}`);
  layout();
}
function layout() { const h = document.getElementById('top').offsetHeight; const fr = document.getElementById('frame'); fr.style.top = h + 'px'; fr.style.height = (innerHeight - h) + 'px'; }
addEventListener('resize', layout);
show();
</script></body></html>
"""

if __name__ == "__main__":
    main()
