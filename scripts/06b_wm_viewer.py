"""Stage 6b: interactive viewer of the trained world model's imagined futures.

    .venv/bin/python scripts/06b_wm_viewer.py

For each city's held-out Wednesday under the baseline policy and under the unseen
policy (P2 ILP), the main model (gru_cons_os) and plain gru imagine every 1-hour
future from 02:00 to 22:00, fed only the logged actions. The page shows them against
what the world actually did, as city totals and as a zone map.
Writes outputs/wm/06_wm_viewer.html.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import torch

from world import cities, wm_data
from world import wm_models as M

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = cities.DATA / "wm_models"
STARTS = list(range(24, 276, 12))
H = 12
DATE = "2025-10-08"


def load(name):
    ck = torch.load(MODELS / f"{name}.pt", map_location=DEV)
    base = ck["kind"]
    m = M.Net(base[:-3] if base.endswith("_os") else base, ck["use_actions"]).to(DEV)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def main():
    models = {"main": load("gru_cons_os"), "gru": load("gru")}
    keys = [k for t in cities.TIERS.values() for k in t]
    data = {"cities": [], "targets": wm_data.OBS, "starts": STARTS, "H": H}
    for k in keys:
        city = cities.get(k)
        runs = [r for r in M.load_runs([k], "test", DEV, policies=["nearest", "p2_ilp"]) if r.date == DATE]
        if not runs:
            continue
        geoms = gpd.read_file(city.data_dir / "zone_geoms.geojson")[["zone", "geometry"]]
        geoms["geometry"] = geoms.geometry.simplify(0.0002)
        entry = {"key": k, "name": city.name, "tier": city.tier, "country": city.country,
                 "zones": json.loads(geoms.to_json()), "runs": {}}
        for r in runs:
            actual = r.obs.sum(1).cpu().numpy()                              # (K+1, 6) city totals
            roll = {}
            for name, m in models.items():
                tot, zone_idle = [], []
                for k0 in STARTS:
                    p = M.rollout(m, r, k0, H)
                    tot.append(p.sum(1).cpu().numpy().round(2).tolist())
                    zone_idle.append(p[-1, :, 3].cpu().numpy().round(2).tolist())
                roll[name] = {"tot": tot, "zone_idle": zone_idle}
            entry["runs"][r.policy] = {
                "actual": actual.round(2).tolist(),
                "actual_zone_idle": [r.obs[k0 + H, :, 3].cpu().numpy().round(1).tolist() for k0 in STARTS],
                "roll": roll}
        data["cities"].append(entry)
        print(f"  {k}: {len(runs)} runs", flush=True)
    out = cities.OUTPUTS / "wm" / "06_wm_viewer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>World model rollouts</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:18px 16px 50px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2);margin:20px 0 8px}
.muted{color:var(--muted)} .row{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
button,select{font:inherit;padding:4px 9px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink);cursor:pointer}
button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:560px){.grid{grid-template-columns:1fr}}
.card{border:1px solid var(--line);border-radius:10px;padding:10px;background:#fff}
svg text{font:11px system-ui,sans-serif;fill:var(--ink2)}
.maps{display:grid;grid-template-columns:1fr 1fr;gap:10px} @media(max-width:700px){.maps{grid-template-columns:1fr}}
.map{height:340px;border-radius:10px;border:1px solid var(--line)}
.key span{display:inline-flex;align-items:center;gap:5px;margin-right:14px}
.sw{display:inline-block;width:16px;border-top:3px solid}
</style></head><body><main>
<h1>What the world model imagines vs what the world did</h1>
<div class="muted">Held-out Wednesday (8 Oct 2025). From the chosen hour, each model imagines the next hour from its own predictions, fed only the dispatcher's logged actions. Lines show city totals per 5 minutes.</div>
<h2>City</h2><div class="row" id="cities"></div>
<h2>Dispatcher on this day</h2><div class="row" id="pols"></div>
<h2>Rollout starts at <b id="hh"></b></h2>
<input type="range" id="start" min="0" step="1" style="width:100%">
<div class="key" style="margin:8px 0"><span><i class="sw" style="border-color:#0b0b0b"></i>actual</span><span><i class="sw" style="border-color:#eb6834"></i>world model (fleet-conserving GRU, overshoot)</span><span><i class="sw" style="border-color:#2a78d6;border-top-style:dashed"></i>plain GRU</span><span><i class="sw" style="border-color:#898781;border-top-style:dotted"></i>persistence</span></div>
<div class="grid" id="charts"></div>
<h2>Idle cabs per zone, one hour after the start</h2>
<div class="maps"><div><div class="muted">actual</div><div id="mapA" class="map"></div></div><div><div class="muted">world model's imagination</div><div id="mapP" class="map"></div></div></div>
</main>
<script>
const D = __DATA__;
const NAMES = {new: 'new requests', pickups: 'pickups', cancels: 'cancellations', idle: 'idle cabs', busy: 'busy cabs', pending: 'waiting requests'};
let ci = 0, pol = 'nearest', si = 15;
const fmtH = k => { const m = k * 5, h = Math.floor(m / 60) % 24; return `${String(h).padStart(2,'0')}:${String(m % 60).padStart(2,'0')}`; };
document.getElementById('cities').innerHTML = D.cities.map((c, i) => `<button data-i="${i}">${c.name}</button>`).join('');
document.querySelectorAll('#cities button').forEach(b => b.onclick = () => { ci = +b.dataset.i; draw(); });
const sl = document.getElementById('start'); sl.max = D.starts.length - 1; sl.value = si; sl.oninput = () => { si = +sl.value; draw(); };

function chart(el, title, act, k0, preds) {
  const W = el.clientWidth - 20, Hh = 150, L = 36, T = 18, B = 18, x0 = Math.max(0, k0 - 24), x1 = Math.min(act.length - 1, k0 + D.H + 12);
  const vals = act.slice(x0, x1 + 1).concat(...preds.map(p => p.v));
  const ymax = Math.max(1, ...vals) * 1.1;
  const X = k => L + (W - L - 6) * (k - x0) / (x1 - x0), Y = v => T + (Hh - T - B) * (1 - v / ymax);
  let g = `<text x="0" y="12" style="fill:#0b0b0b;font-weight:600">${title}</text>`;
  for (let t = 0; t <= 2; t++) { const v = ymax * t / 2; g += `<line x1="${L}" x2="${W - 6}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e1e0d9"/><text x="${L - 4}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(v < 10 ? 1 : 0)}</text>`; }
  g += `<rect x="${X(k0)}" y="${T}" width="${X(k0 + D.H) - X(k0)}" height="${Hh - T - B}" fill="#f3f2ee"/>`;
  [x0, k0, k0 + D.H, x1].forEach(k => g += `<text x="${X(k)}" y="${Hh - 4}" text-anchor="middle">${fmtH(k)}</text>`);
  g += `<polyline fill="none" stroke="#0b0b0b" stroke-width="1.6" points="${act.slice(x0, x1 + 1).map((v, i) => `${X(x0 + i)},${Y(v)}`).join(' ')}"/>`;
  preds.forEach(p => g += `<polyline fill="none" stroke="${p.c}" stroke-width="2" stroke-dasharray="${p.d}" points="${p.v.map((v, i) => `${X(k0 + 1 + i)},${Y(v)}`).join(' ')}"/>`);
  el.innerHTML = `<svg width="${W}" height="${Hh}">${g}</svg>`;
}

let mA, mP, lA, lP;
const ramp = ['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'];
function drawMaps(c, run) {
  if (!mA) { mA = L.map('mapA', {zoomControl: false, attributionControl: false}); mP = L.map('mapP', {zoomControl: false, attributionControl: false});
    mA.on('move', () => mP.setView(mA.getCenter(), mA.getZoom(), {animate: false})); }
  [lA, lP].forEach((l, i) => l && [mA, mP][i].removeLayer(l));
  const a = run.actual_zone_idle[si], p = run.roll.main.zone_idle[si], vmax = Math.max(1, ...a, ...p);
  const style = vals => f => ({color: '#fcfcfb', weight: .6, fillOpacity: .85, fillColor: ramp[Math.min(6, Math.floor(vals[f.properties.zone] / vmax * 7))]});
  const tip = vals => (f, l) => l.bindTooltip(`zone ${f.properties.zone}: ${vals[f.properties.zone].toFixed(1)} idle`, {sticky: true});
  lA = L.geoJSON(c.zones, {style: style(a), onEachFeature: tip(a)}).addTo(mA);
  lP = L.geoJSON(c.zones, {style: style(p), onEachFeature: tip(p)}).addTo(mP);
  if (!drawMaps.last || drawMaps.last !== c.key) { mA.fitBounds(lA.getBounds()); mP.fitBounds(lA.getBounds()); drawMaps.last = c.key; }
}

function draw() {
  const c = D.cities[ci];
  document.querySelectorAll('#cities button').forEach((b, i) => b.classList.toggle('on', i === ci));
  const pols = Object.keys(c.runs); if (!pols.includes(pol)) pol = pols[0];
  document.getElementById('pols').innerHTML = pols.map(p => `<button data-p="${p}" class="${p === pol ? 'on' : ''}">${p === 'nearest' ? 'baseline (nearest)' : p === 'p2_ilp' ? 'P2 ILP (never seen in training)' : p}</button>`).join('');
  document.querySelectorAll('#pols button').forEach(b => b.onclick = () => { pol = b.dataset.p; draw(); });
  const run = c.runs[pol], k0 = D.starts[si];
  document.getElementById('hh').textContent = fmtH(k0) + '–' + fmtH(k0 + D.H);
  const box = document.getElementById('charts'); box.innerHTML = D.targets.map((t, j) => `<div class="card" id="c${j}"></div>`).join('');
  D.targets.forEach((t, j) => {
    const act = run.actual.map(r => r[j]);
    chart(document.getElementById('c' + j), NAMES[t], act, k0, [
      {v: run.roll.main.tot[si].map(r => r[j]), c: '#eb6834', d: ''},
      {v: run.roll.gru.tot[si].map(r => r[j]), c: '#2a78d6', d: '5 4'},
      {v: Array(D.H).fill(act[k0]), c: '#898781', d: '2 3'}]);
  });
  drawMaps(c, run);
}
draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
