"""Standalone interactive HTML explorers, one per stage.

Each explorer is one self-contained HTML file: data is embedded, Leaflet comes from
cdnjs, basemap tiles from Esri. It opens straight from disk. All of them share the
graph and an in-browser Dijkstra over the *same* processed graph the simulator uses,
so anything clicked in a browser can be checked against the Python side.

* stage 1 `build`: road classes, zone choropleth, free-flow route checker.
* stage 2 `build_traffic`: congestion or traffic volume by time of day (slider + play), network
  TTI chart, time-dependent routes with a 24 h travel-time profile, and zone-to-zone
  travel times from a clicked zone.

Append ``#demo`` to the URL to auto-run a random route (used by headless smoke tests).
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd

from .cities import City
from .viz import ROAD_STYLE

CLASS_ORDER = list(ROAD_STYLE)
VOL_UNIT = 20  # veh/h per uint8 step in the explorer's volume layer (saturates at 5,100)


def _edge_coords(G, edges: pd.DataFrame, nodes: pd.DataFrame, tol_deg=2e-5):
    """Edge polylines (simplified OSM geometry or straight u->v), rounded to ~1 m."""
    pos = nodes.set_index("node_id")[["lon", "lat"]]
    out = []
    for u, v, k in zip(edges.u, edges.v, edges.key):
        geom = G.edges[u, v, k].get("geometry")
        if geom is not None:
            pts = np.asarray(geom.simplify(tol_deg).coords)
        else:
            pts = np.array([pos.loc[u].to_numpy(), pos.loc[v].to_numpy()])
        out.append(np.round(pts[:, ::-1], 5).tolist())  # Leaflet wants [lat, lon]
    return out


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def graph_data(city: City, G, nodes, edges, zone_geoms) -> dict:
    idx = pd.Series(np.arange(len(nodes)), index=nodes.node_id)
    return {
        "city": city.name,
        "currency": city.currency,
        "nodes": np.round(nodes[["lat", "lon"]].to_numpy(), 5).tolist(),
        "node_zone": nodes.zone.astype(int).tolist(),
        "eu": idx.loc[edges.u].astype(int).tolist(),
        "ev": idx.loc[edges.v].astype(int).tolist(),
        "elen": np.round(edges.length_m.to_numpy(), 1).tolist(),
        "etime": np.round(edges.free_flow_s.to_numpy(), 1).tolist(),
        "ecls": edges.road_class.map(CLASS_ORDER.index).astype(int).tolist(),
        "ecoords": _edge_coords(G, edges, nodes),
        "classes": [{"name": c, "color": s[0], "w": s[1]} for c, s in ROAD_STYLE.items()],
        "zones": json.loads(zone_geoms.drop(columns=[c for c in zone_geoms.columns
                                                     if c in ("h3", "h3_res")]).to_json()),
    }


def _write(path, template, title, data):
    html = (template.replace("__HEAD__", HEAD).replace("__JS_BASE__", JS_BASE).replace("__JS_MAP__", JS_MAP)
            .replace("__TITLE__", title).replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    print(f"  wrote {path.relative_to(path.parents[2])} ({path.stat().st_size / 1e6:.1f} MB)")


def build(city: City, G, nodes, edges, zones, zone_geoms, path) -> None:
    data = graph_data(city, G, nodes, edges, zone_geoms)
    data["zone_metrics"] = ["n_nodes", "n_pois", "road_km", "area_km2"]
    _write(path, STAGE1, f"{city.name} world: stage 1, road network and zones", data)


def build_traffic(city: City, G, nodes, edges, zones, zone_geoms, typical_times, volume, calib,
                  zone_tt, gateways, path) -> None:
    """typical_times: (2, 24, E) s; volume: (2, 24, E) veh/h; zone_tt: (2, 24, Z, Z) s."""
    data = graph_data(city, G, nodes, edges, zone_geoms)
    t0 = edges.free_flow_s.to_numpy()[None, None]
    ratio = np.clip(np.round(typical_times / t0 * 20), 20, 255).astype(np.uint8)  # t/t0 * 20
    data["ratio_b64"] = _b64(ratio)
    data["vol_b64"] = _b64(np.clip(np.round(volume / VOL_UNIT), 0, 255).astype(np.uint8))
    data["vol_unit"] = VOL_UNIT
    data["zone_tt_b64"] = _b64(np.minimum(zone_tt / 10, 65535).round().astype(np.uint16))  # 10 s units
    data["n_zones"] = int(len(zones))
    data["calib"] = {dt: {"tti": g.sort_values("hour").tti.round(3).tolist(),
                          "target": g.sort_values("hour").target.tolist()}
                     for dt, g in calib.groupby("daytype")}
    data["peaks"] = [city.traffic.am_peak_hour, city.traffic.pm_peak_hour]
    data["gateways"] = gateways[["lat", "lon", "weight"]].round(5).to_dict("records")
    _write(path, STAGE2, f"{city.name} world: stage 2, traffic by time of day", data)


# ------------------------------------------------------------------ shared pieces

HEAD = r"""<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#eb6834;--accent2:#2a78d6}
*{box-sizing:border-box}
html,body{margin:0;height:100%;font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--surface)}
#map{position:absolute;inset:0 360px 0 0}
#panel{position:absolute;top:0;right:0;bottom:0;width:360px;overflow-y:auto;overflow-x:hidden;padding:14px 16px;border-left:1px solid var(--line);background:var(--surface)}
h1{font-size:15px;margin:0 0 4px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2);margin:18px 0 6px}
.muted{color:var(--muted)} .row{display:flex;align-items:center;gap:6px;margin:3px 0}
.sw{display:inline-block;width:18px;border-top-style:solid;border-radius:2px}
select,button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink)}
button{cursor:pointer} button:hover{background:#f3f2ee} button.on{background:#0b0b0b;color:#fff;border-color:#0b0b0b}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td{padding:2px 0;border-bottom:1px solid var(--line)} td:last-child{text-align:right}
.legendbar{height:8px;border-radius:4px}
#routeinfo b{font-size:14px}
svg text{font:11px system-ui,sans-serif;fill:var(--muted)}
.tip{position:fixed;pointer-events:none;background:#0b0b0b;color:#fff;padding:4px 7px;border-radius:5px;font-size:12px;display:none;z-index:1000;font-variant-numeric:tabular-nums}
.leaflet-overlay-pane canvas{pointer-events:none}  /* roads/route canvas sits above zones; let hover reach zones */
@media (max-width:760px){#map{inset:0 0 50vh 0}#panel{top:auto;left:0;width:auto;height:50vh;border-left:0;border-top:1px solid var(--line)}}
</style>"""

JS_MAP = r"""
const D = __DATA__;
const map = L.map('map', {preferCanvas: true, zoomSnap: 0.25});
const base = {
  'grey basemap': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    {attribution: 'Tiles &copy; Esri · roads &copy; OpenStreetMap contributors', maxZoom: 16}),
  'OpenStreetMap': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution: '&copy; OpenStreetMap contributors', maxZoom: 19}),
  'no basemap': L.layerGroup(),
};
base['grey basemap'].addTo(map);
L.control.layers(base, null, {position: 'topleft'}).addTo(map);
map.createPane('zones').style.zIndex = 350;  // below roads (overlayPane = 400)
const zoneRenderer = L.canvas({pane: 'zones'});
const roadRenderer = L.canvas({padding: 0.3});
map.fitBounds(L.geoJSON(D.zones).getBounds());
const tip = document.createElement('div'); tip.className = 'tip'; document.body.appendChild(tip);
function showTip(ev, html) { tip.innerHTML = html; tip.style.display = 'block'; tip.style.left = (ev.clientX + 12) + 'px'; tip.style.top = (ev.clientY + 12) + 'px'; }
function hideTip() { tip.style.display = 'none'; }
function b64(s, T) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); }
function fmtT(s) { const m = Math.floor(s / 60); return `${m} min ${Math.round(s - 60 * m)} s`; }
function fmtH(h) { const hh = Math.floor(h) % 24, mm = Math.round((h - Math.floor(h)) * 60); return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`; }

// ---- small line chart helper (single axis, hover crosshair)
function lineChart(svg, series, opts) {
  const W = svg.clientWidth || +svg.getAttribute('width'), H = +svg.getAttribute('height'), L0 = opts.left ?? 34, R = 8, T = 8, B = 20;
  const all = series.flatMap(s => s.y.filter(v => v != null));
  const ymin = opts.ymin ?? Math.min(...all), ymax = Math.max(...all) * 1.05;
  const X = h => L0 + (W - L0 - R) * h / 24, Y = v => T + (H - T - B) * (1 - (v - ymin) / (ymax - ymin));
  let g = '';
  for (let k = 0; k <= 4; k++) { const v = ymin + (ymax - ymin) * k / 4; g += `<line x1="${L0}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e1e0d9" stroke-width="${k ? .6 : 1}"/><text x="${L0 - 4}" y="${Y(v) + 4}" text-anchor="end">${(opts.tick ?? opts.fmt)(v)}</text>`; }
  for (let h = 0; h <= 24; h += 6) g += `<text x="${X(h)}" y="${H - 5}" text-anchor="middle">${String(h).padStart(2, '0')}h</text>`;
  series.forEach(s => {
    const pts = s.y.map((v, i) => v == null ? null : [X(s.x[i]), Y(v)]).filter(Boolean);
    if (s.dots) pts.forEach(([x, y]) => g += `<circle cx="${x}" cy="${y}" r="2.2" fill="${s.color}"/>`);
    else g += `<polyline points="${pts.map(p => p.join(',')).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
  });
  if (opts.marker != null) g += `<line x1="${X(opts.marker)}" x2="${X(opts.marker)}" y1="${T}" y2="${H - B}" stroke="#0b0b0b" stroke-width="1"/>`;
  g += `<line id="xh" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="#898781" stroke-dasharray="3 3" visibility="hidden"/>`;
  g += `<rect x="${L0}" y="${T}" width="${W - L0 - R}" height="${H - T - B}" fill="transparent" style="cursor:crosshair"/>`;
  svg.innerHTML = g;
  const rect = svg.querySelector('rect'), xh = svg.querySelector('#xh');
  rect.onmousemove = ev => { const bb = svg.getBoundingClientRect(), h = Math.max(0, Math.min(24, (ev.clientX - bb.left - L0) / (W - L0 - R) * 24));
    xh.setAttribute('x1', X(h)); xh.setAttribute('x2', X(h)); xh.setAttribute('visibility', 'visible');
    showTip(ev, `<b>${fmtH(h)}</b><br>` + series.map(s => `${s.name}: ${opts.fmt(opts.at(s, h))}`).join('<br>')); };
  rect.onmouseleave = () => { hideTip(); xh.setAttribute('visibility', 'hidden'); };
  if (opts.onclick) rect.onclick = ev => { const bb = svg.getBoundingClientRect(); opts.onclick((ev.clientX - bb.left - L0) / (W - L0 - R) * 24); };
}
"""

JS_GRAPH = r"""
const N = D.nodes.length, M = D.eu.length;
// ---- graph in CSR form + Dijkstra (binary heap) over any edge-weight array
const off = new Int32Array(N + 1), adj = new Int32Array(M);
for (let e = 0; e < M; e++) off[D.eu[e] + 1]++;
for (let i = 0; i < N; i++) off[i + 1] += off[i];
{ const fill = off.slice(0, N); for (let e = 0; e < M; e++) adj[fill[D.eu[e]]++] = e; }
function dijkstra(s, t, w) {
  const dist = new Float64Array(N).fill(Infinity), prevE = new Int32Array(N).fill(-1);
  const heap = [[0, s]]; dist[s] = 0;
  const push = x => { heap.push(x); let i = heap.length - 1; while (i) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2*i+1, r = l+1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
  while (heap.length) {
    const [d, u] = pop(); if (d > dist[u]) continue; if (u === t) break;
    for (let k = off[u]; k < off[u + 1]; k++) { const e = adj[k], v = D.ev[e], nd = d + w[e]; if (nd < dist[v]) { dist[v] = nd; prevE[v] = e; push([nd, v]); } }
  }
  if (dist[t] === Infinity) return null;
  const path = []; for (let v = t; v !== s; v = D.eu[prevE[v]]) path.push(prevE[v]);
  return path.reverse();
}
function nearest(lat, lon) {
  const c = Math.cos(lat * Math.PI / 180); let best = -1, bd = Infinity;
  for (let i = 0; i < N; i++) { const dy = D.nodes[i][0] - lat, dx = (D.nodes[i][1] - lon) * c, d = dx*dx + dy*dy; if (d < bd) { bd = d; best = i; } }
  return best;
}
const routeLayer = L.layerGroup().addTo(map);
function mark(i, color) { L.circleMarker(D.nodes[i], {radius: 7, color: '#fcfcfb', weight: 2, fillColor: color, fillOpacity: 1}).addTo(routeLayer); }
function drawPath(p) {
  L.polyline(p.map(e => D.ecoords[e]), {color: '#fcfcfb', weight: 8, opacity: .9}).addTo(routeLayer);
  L.polyline(p.map(e => D.ecoords[e]), {color: '#0b0b0b', weight: 4}).addTo(routeLayer);
}
function pathStats(p, w) {
  let len = 0, tt = 0, ff = 0; const zs = new Set([D.node_zone[D.eu[p[0]]]]), byCls = {};
  p.forEach(e => { len += D.elen[e]; tt += w[e]; ff += D.etime[e]; zs.add(D.node_zone[D.ev[e]]); const c = D.classes[D.ecls[e]].name; byCls[c] = (byCls[c] || 0) + D.elen[e]; });
  return {len, tt, ff, zones: zs.size, byCls};
}
"""

JS_BASE = JS_MAP + JS_GRAPH


# ------------------------------------------------------------------ stage 1

STAGE1 = r"""<!doctype html>
<html lang="en"><head><title>__TITLE__</title>__HEAD__</head><body>
<div id="map"></div>
<div id="panel">
  <h1 id="title"></h1>
  <div class="muted" id="stats"></div>

  <h2>Route checker</h2>
  <div class="muted">Click the map to set an <span style="color:var(--accent2)">origin</span>, then a <span style="color:var(--accent)">destination</span>. Fastest path by free-flow time on the processed graph.</div>
  <div class="row" style="margin-top:6px"><button id="rand">Random route</button><button id="clear">Clear</button></div>
  <div id="routeinfo" style="margin-top:6px"></div>

  <h2>Zones</h2>
  <div class="row">colour by <select id="metric"></select> <label><input type="checkbox" id="showzones" checked> show</label></div>
  <div class="legendbar" style="background:linear-gradient(90deg,#cde2fb,#6da7ec,#256abf,#0d366b)"></div><div class="row muted" style="justify-content:space-between"><span id="lo"></span><span id="hi"></span></div>
  <div id="zoneinfo" class="muted">Hover a zone.</div>

  <h2>Road classes</h2>
  <div id="classes"></div>
</div>
<script>
__JS_BASE__
document.getElementById('title').textContent = D.city;
const totKm = D.elen.reduce((a, b) => a + b, 0) / 1000;
document.getElementById('stats').textContent =
  `${N.toLocaleString()} nodes · ${M.toLocaleString()} directed edges · ${Math.round(totKm).toLocaleString()} km · ${D.zones.features.length} zones`;

// ---- roads, one canvas polyline group per class
const classLayers = D.classes.map((c, ci) => {
  const lines = [];
  for (let e = 0; e < M; e++) if (D.ecls[e] === ci) lines.push(D.ecoords[e]);
  return {c, n: lines.length, layer: L.polyline(lines, {color: c.color, weight: Math.max(c.w * 2, 1), opacity: .9, renderer: roadRenderer, interactive: false})};
});
const clsDiv = document.getElementById('classes');
classLayers.forEach(({c, n, layer}, i) => {
  if (!n) return;
  const on = i <= 6;
  if (on) layer.addTo(map);
  const row = document.createElement('label'); row.className = 'row';
  row.innerHTML = `<input type="checkbox" ${on ? 'checked' : ''}><span class="sw" style="border-top-width:${Math.max(c.w*3,2)}px;border-color:${c.color}"></span>${c.name} <span class="muted">(${n.toLocaleString()})</span>`;
  row.querySelector('input').onchange = ev => ev.target.checked ? layer.addTo(map) : map.removeLayer(layer);
  clsDiv.appendChild(row);
});

// ---- zones
const ramp = ['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'];
const msel = document.getElementById('metric');
D.zone_metrics.forEach(m => msel.add(new Option(m, m)));
let zoneLayer;
function colorFor(v, lo, hi) { const t = hi > lo ? (v - lo) / (hi - lo) : 0; return ramp[Math.min(ramp.length - 1, Math.floor(t * ramp.length))]; }
function drawZones() {
  if (zoneLayer) map.removeLayer(zoneLayer);
  const m = msel.value, vals = D.zones.features.map(f => f.properties[m]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  document.getElementById('lo').textContent = lo.toLocaleString(); document.getElementById('hi').textContent = hi.toLocaleString();
  zoneLayer = L.geoJSON(D.zones, {
    renderer: zoneRenderer, pane: 'zones',
    style: f => ({color: '#fcfcfb', weight: 1.5, fillColor: colorFor(f.properties[m], lo, hi), fillOpacity: .35}),
    onEachFeature: (f, l) => {
      l.on('mouseover', () => { l.setStyle({weight: 3, color: '#0b0b0b'});
        document.getElementById('zoneinfo').innerHTML = '<table>' + Object.entries(f.properties).map(([k, v]) =>
          `<tr><td>${k}</td><td>${typeof v === 'number' ? v.toLocaleString() : v}</td></tr>`).join('') + '</table>'; });
      l.on('mouseout', () => zoneLayer.resetStyle(l));
      l.bindTooltip(`zone ${f.properties.zone}: ${m} = ${f.properties[m].toLocaleString()}`, {sticky: true});
    }
  });
  if (document.getElementById('showzones').checked) zoneLayer.addTo(map);
}
msel.onchange = drawZones;
document.getElementById('showzones').onchange = e => e.target.checked ? zoneLayer.addTo(map) : map.removeLayer(zoneLayer);
drawZones();

// ---- route checker (free-flow)
let pts = [];
function route(a, b) {
  routeLayer.clearLayers(); mark(a, '#2a78d6'); mark(b, '#eb6834');
  const t0 = performance.now(), p = dijkstra(a, b, D.etime), ms = performance.now() - t0;
  const info = document.getElementById('routeinfo');
  if (!p) { info.textContent = 'No path (should not happen: graph is strongly connected).'; return; }
  drawPath(p); mark(a, '#2a78d6'); mark(b, '#eb6834');
  const s = pathStats(p, D.etime);
  info.innerHTML = `<b>${(s.len / 1000).toFixed(2)} km · ${fmtT(s.tt)}</b> <span class="muted">free-flow</span>
    <table><tr><td>avg speed</td><td>${(s.len / s.tt * 3.6).toFixed(1)} km/h</td></tr>
    <tr><td>edges / zones crossed</td><td>${p.length} / ${s.zones}</td></tr>
    <tr><td>zone ${D.node_zone[a]} → zone ${D.node_zone[b]}</td><td></td></tr>
    ${Object.entries(s.byCls).sort((x, y) => y[1] - x[1]).map(([c, l]) => `<tr><td>on ${c}</td><td>${(l/1000).toFixed(2)} km</td></tr>`).join('')}
    <tr><td class="muted">search time</td><td class="muted">${ms.toFixed(0)} ms</td></tr></table>`;
}
map.on('click', ev => {
  const i = nearest(ev.latlng.lat, ev.latlng.lng);
  if (pts.length === 2) { pts = []; routeLayer.clearLayers(); document.getElementById('routeinfo').innerHTML = ''; }
  pts.push(i);
  if (pts.length === 1) mark(i, '#2a78d6'); else route(pts[0], pts[1]);
});
document.getElementById('rand').onclick = () => { pts = [Math.floor(Math.random() * N), Math.floor(Math.random() * N)]; route(...pts); };
document.getElementById('clear').onclick = () => { pts = []; routeLayer.clearLayers(); document.getElementById('routeinfo').innerHTML = ''; };
if (location.hash === '#demo') document.getElementById('rand').click();
</script></body></html>
"""

# ------------------------------------------------------------------ stage 2

STAGE2 = r"""<!doctype html>
<html lang="en"><head><title>__TITLE__</title>__HEAD__</head><body>
<div id="map"></div>
<div id="panel">
  <h1 id="title"></h1>
  <div class="muted">Typical conditions from the calibrated assignment. Colour = delay factor t / t<sub>free-flow</sub> per edge.</div>

  <h2>Time of day</h2>
  <div class="row"><button id="wd" class="on">Weekday</button><button id="we">Weekend</button><button id="play" style="margin-left:auto">▶ Play</button></div>
  <input id="hour" type="range" min="0" max="23.75" step="0.25" value="8.5" style="width:100%">
  <div class="row"><b id="clock" style="font-size:18px"></b><span class="muted" id="ttinow"></span></div>
  <svg id="tti" width="100%" height="130" style="display:block"></svg>

  <h2>Map shows</h2>
  <div class="row"><button id="v_cong" class="on">Congestion</button><button id="v_vol">Traffic volume</button></div>
  <div id="legend" style="margin-top:6px"></div>
  <label class="row"><input type="checkbox" id="showgw" checked><span style="color:#2a78d6">●</span> external gateways <span class="muted" id="ngw"></span></label>

  <h2>Click mode</h2>
  <div class="row"><button id="m_route" class="on">Route</button><button id="m_zone">Zone travel times</button></div>
  <div id="modehelp" class="muted"></div>
  <div class="row" style="margin-top:6px" id="routebtns"><button id="rand">Random route</button><button id="clear">Clear</button></div>
  <div id="routeinfo" style="margin-top:6px"></div>
  <div id="proftitle" class="muted" style="display:none;margin-top:10px">Fastest travel time for this trip, by departure time</div>
  <svg id="profile" width="100%" height="140" style="display:none"></svg>
</div>
<script>
__JS_BASE__
document.getElementById('title').textContent = D.city + ': traffic';
const Z = D.n_zones;
const RATIO = b64(D.ratio_b64, Uint8Array);        // [daytype][hour][edge], t/t0 * 20
const VOL = b64(D.vol_b64, Uint8Array);            // [daytype][hour][edge], veh/h / vol_unit
const ZTT = b64(D.zone_tt_b64, Uint16Array);        // [daytype][hour][i][j], 10 s units
let dt = 0, mode = 'route', view = 'cong';
const hourEl = document.getElementById('hour');

// ---- time-dependent edge times (linear between hourly slices centred on :30)
function edgeTimes(d, h) {
  const x = ((h - 0.5) % 24 + 24) % 24, h0 = Math.floor(x), h1 = (h0 + 1) % 24, w = x - h0;
  const a = (d * 24 + h0) * M, b = (d * 24 + h1) * M, out = new Float64Array(M);
  for (let e = 0; e < M; e++) out[e] = D.etime[e] * ((1 - w) * RATIO[a + e] + w * RATIO[b + e]) / 20;
  return out;
}
function interpHourly(arr, h) { const x = ((h - 0.5) % 24 + 24) % 24, h0 = Math.floor(x), w = x - h0; return (1 - w) * arr[h0] + w * arr[(h0 + 1) % 24]; }

// ---- map layer: edges binned by delay factor (congestion view) or by volume (volume view)
const VIEWS = {
  cong: {title: 'delay factor t / t<sub>free-flow</sub>', bins: [1.15, 1.4, 1.8, 2.5, 4, Infinity],
         cols: ['#d6d5ce', '#fbc4a4', '#f39063', '#eb6834', '#b8431a', '#6e2308'],
         widths: [1, 2.2, 2.2, 2.2, 2.2, 2.2],
         labels: ['< 1.15 (free-flowing)', '1.15–1.4', '1.4–1.8', '1.8–2.5', '2.5–4', '≥ 4']},
  vol:  {title: 'background traffic, vehicles / hour', bins: [20, 100, 300, 800, 1600, Infinity],
         cols: ['#e1e0d9', '#b7d3f6', '#6da7ec', '#2a78d6', '#184f95', '#0d366b'],
         widths: [0.8, 1.2, 1.8, 2.6, 3.4, 4.2],
         labels: ['< 20 (≈ unused)', '20–100', '100–300', '300–800', '800–1,600', '≥ 1,600']},
};
function drawLegend() {
  const V = VIEWS[view];
  document.getElementById('legend').innerHTML = `<div class="muted">${V.title}</div>` + V.cols.map((c, i) =>
    `<div class="row"><span class="sw" style="border-top-width:${Math.max(2, V.widths[i] + 1)}px;border-color:${c}"></span>${V.labels[i]} <span class="muted" id="bin${i}"></span></div>`).join('');
}
let congLayers = [], curTimes;
function drawCongestion() {
  const h = +hourEl.value; curTimes = edgeTimes(dt, h);
  const V = VIEWS[view];
  const x = ((h - 0.5) % 24 + 24) % 24, h0 = Math.floor(x), h1 = (h0 + 1) % 24, w = x - h0;
  const a = (dt * 24 + h0) * M, b1 = (dt * 24 + h1) * M;
  congLayers.forEach(l => map.removeLayer(l));
  const groups = V.cols.map(() => []), km = V.cols.map(() => 0);
  for (let e = 0; e < M; e++) {
    const val = view === 'cong' ? curTimes[e] / D.etime[e] : ((1 - w) * VOL[a + e] + w * VOL[b1 + e]) * D.vol_unit;
    let b = 0; while (val >= V.bins[b]) b++;
    km[b] += D.elen[e] / 1000;
    if (view === 'cong' && b === 0 && D.ecls[e] > 5) continue;   // free-flowing minor streets: clutter, not information
    groups[b].push(D.ecoords[e]);
  }
  congLayers = groups.map((g, b) => L.polyline(g, {color: V.cols[b], weight: V.widths[b], opacity: .95, renderer: roadRenderer, interactive: false}).addTo(map));
  km.forEach((k, i) => document.getElementById('bin' + i).textContent = `(${Math.round(k).toLocaleString()} km)`);
  document.getElementById('clock').textContent = fmtH(h);
  const c = D.calib[dt ? 'weekend' : 'weekday'];
  document.getElementById('ttinow').textContent = ` network TTI ${interpHourly(c.tti, h).toFixed(2)} (target ${interpHourly(c.target, h).toFixed(2)})`;
  drawTTI(); if (mode === 'route' && pts.length === 2) route(pts[0], pts[1], false); if (mode === 'zone' && zoneSel !== null) drawZoneTT();
}

const HOURS = [...Array(24).keys()].map(h => h + 0.5);
function drawTTI() {
  const c = D.calib[dt ? 'weekend' : 'weekday'];
  lineChart(document.getElementById('tti'), [
    {name: 'model TTI', x: HOURS, y: c.tti, color: '#eb6834'},
    {name: 'target', x: HOURS, y: c.target, color: '#0b0b0b', dots: true},
  ], {ymin: 1, fmt: v => v.toFixed(2), marker: +hourEl.value, at: (s, h) => interpHourly(s.y, h),
      onclick: h => { hourEl.value = Math.round(h * 4) / 4; drawCongestion(); }});
}

// ---- route mode
let pts = [];
function route(a, b, withProfile = true) {
  routeLayer.clearLayers();
  const p = dijkstra(a, b, curTimes), info = document.getElementById('routeinfo');
  if (!p) { info.textContent = 'No path.'; return; }
  drawPath(p); mark(a, '#2a78d6'); mark(b, '#eb6834');
  const s = pathStats(p, curTimes);
  info.innerHTML = `<b>${(s.len / 1000).toFixed(2)} km · ${fmtT(s.tt)}</b> <span class="muted">at ${fmtH(+hourEl.value)}</span>
    <table><tr><td>free-flow time on this path</td><td>${fmtT(s.ff)}</td></tr>
    <tr><td>delay factor</td><td>${(s.tt / s.ff).toFixed(2)}×</td></tr>
    <tr><td>avg speed</td><td>${(s.len / s.tt * 3.6).toFixed(1)} km/h</td></tr>
    <tr><td>zone ${D.node_zone[a]} → zone ${D.node_zone[b]}</td><td>${s.zones} zones</td></tr></table>`;
  if (withProfile) profile(a, b);
}
function profile(a, b) {
  // fastest time for this OD at every half hour, re-routing each time (weekday and weekend)
  const svg = document.getElementById('profile'); svg.style.display = 'block'; document.getElementById('proftitle').style.display = 'block';
  const xs = [...Array(48).keys()].map(k => k / 2), ys = [[], []];
  for (let d = 0; d < 2; d++) xs.forEach(h => { const w = edgeTimes(d, h), p = dijkstra(a, b, w); ys[d].push(p.reduce((acc, e) => acc + w[e], 0) / 60); });
  lineChart(svg, [{name: 'weekday', x: xs, y: ys[0], color: '#eb6834'}, {name: 'weekend', x: xs, y: ys[1], color: '#2a78d6'}],
    {ymin: 0, left: 44, fmt: v => v.toFixed(1) + ' min', tick: v => (v < 10 ? v.toFixed(1) : Math.round(v)) + ' min', marker: +hourEl.value, at: (s, h) => s.y[Math.min(47, Math.round(h * 2))],
     onclick: h => { hourEl.value = Math.round(h * 4) / 4; drawCongestion(); }});
}

// ---- zone travel-time mode
let zoneLayer = null, zoneSel = null;
const ramp = ['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'];
function drawZoneTT() {
  if (zoneLayer) map.removeLayer(zoneLayer);
  if (zoneSel === null) return;
  const x = ((+hourEl.value - 0.5) % 24 + 24) % 24, h0 = Math.floor(x), h1 = (h0 + 1) % 24, w = x - h0;
  const tt = j => ((1 - w) * ZTT[((dt * 24 + h0) * Z + zoneSel) * Z + j] + w * ZTT[((dt * 24 + h1) * Z + zoneSel) * Z + j]) * 10 / 60;
  const step = 5; // minutes per colour band
  zoneLayer = L.geoJSON(D.zones, {renderer: zoneRenderer, pane: 'zones',
    style: f => { const m = tt(f.properties.zone); return {color: f.properties.zone === zoneSel ? '#0b0b0b' : '#fcfcfb', weight: f.properties.zone === zoneSel ? 3 : 1, fillColor: ramp[Math.min(6, Math.floor(m / step))], fillOpacity: .6}; },
    onEachFeature: (f, l) => l.bindTooltip(() => `zone ${zoneSel} → ${f.properties.zone}: ${tt(f.properties.zone).toFixed(1)} min`, {sticky: true})}).addTo(map);
  document.getElementById('routeinfo').innerHTML = `<b>From zone ${zoneSel}</b> at ${fmtH(+hourEl.value)}<div class="legendbar" style="margin-top:6px;background:linear-gradient(90deg,${ramp.join(',')})"></div><div class="row muted" style="justify-content:space-between"><span>0</span><span>${step * 3} min</span><span>≥ ${step * 6} min</span></div>`;
}
function setMode(m) {
  mode = m; routeLayer.clearLayers(); pts = []; zoneSel = null; if (zoneLayer) { map.removeLayer(zoneLayer); zoneLayer = null; }
  document.getElementById('routeinfo').innerHTML = ''; document.getElementById('profile').style.display = 'none'; document.getElementById('proftitle').style.display = 'none';
  document.getElementById('m_route').classList.toggle('on', m === 'route'); document.getElementById('m_zone').classList.toggle('on', m === 'zone');
  document.getElementById('routebtns').style.display = m === 'route' ? 'flex' : 'none';
  document.getElementById('modehelp').innerHTML = m === 'route'
    ? 'Click an origin, then a destination. Fastest path on the travel times at the selected hour; below it, how that trip\'s time changes over the day.'
    : 'Click anywhere to colour every zone by travel time from that zone (between zone centre nodes) at the selected hour.';
}
map.on('click', ev => {
  const i = nearest(ev.latlng.lat, ev.latlng.lng);
  if (mode === 'zone') { zoneSel = D.node_zone[i]; drawZoneTT(); return; }
  if (pts.length === 2) { pts = []; routeLayer.clearLayers(); }
  pts.push(i);
  if (pts.length === 1) mark(i, '#2a78d6'); else route(pts[0], pts[1]);
});
document.getElementById('m_route').onclick = () => setMode('route');
document.getElementById('m_zone').onclick = () => setMode('zone');
document.getElementById('rand').onclick = () => { pts = [Math.floor(Math.random() * N), Math.floor(Math.random() * N)]; route(...pts); };
document.getElementById('clear').onclick = () => setMode('route');
document.getElementById('wd').onclick = () => { dt = 0; wd.classList.add('on'); we.classList.remove('on'); drawCongestion(); if (pts.length === 2) profile(...pts); };
document.getElementById('we').onclick = () => { dt = 1; we.classList.add('on'); wd.classList.remove('on'); drawCongestion(); if (pts.length === 2) profile(...pts); };
hourEl.oninput = drawCongestion;
let timer = null;
document.getElementById('play').onclick = ev => {
  if (timer) { clearInterval(timer); timer = null; ev.target.textContent = '▶ Play'; return; }
  ev.target.textContent = '❚❚ Pause';
  timer = setInterval(() => { hourEl.value = (+hourEl.value + 0.25) % 24; drawCongestion(); }, 350);
};
const gwLayer = L.layerGroup(D.gateways.map((g, i) => L.circleMarker([g.lat, g.lon], {radius: 4 + 30 * g.weight, color: '#fcfcfb', weight: 1.5, fillColor: '#2a78d6', fillOpacity: .9})
  .bindTooltip(`gateway ${i}: ${(100 * g.weight).toFixed(1)}% of external traffic`))).addTo(map);
document.getElementById('ngw').textContent = `(${D.gateways.length})`;
document.getElementById('showgw').onchange = e => e.target.checked ? gwLayer.addTo(map) : map.removeLayer(gwLayer);
function setView(v) { view = v; document.getElementById('v_cong').classList.toggle('on', v === 'cong'); document.getElementById('v_vol').classList.toggle('on', v === 'vol'); drawLegend(); drawCongestion(); }
document.getElementById('v_cong').onclick = () => setView('cong');
document.getElementById('v_vol').onclick = () => setView('vol');
setMode('route');
drawLegend();
drawCongestion();
if (location.hash === '#demo') document.getElementById('rand').click();
</script></body></html>
"""
