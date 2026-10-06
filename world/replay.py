"""Animated replay of a simulated day: every cab as a dot with a short fading trail.

Each driver becomes a keyframe track (time, node, status, move-flag) built from its
driven legs (one keyframe per graph node, move=1: interpolate from the previous
keyframe) and its status changes (move=0: stay put). The browser evaluates every
track at the current sim time, and a few trail samples at earlier times, and paints
them on a canvas above the map. Colours follow driver status.

Data is packed as base64 typed arrays; a 3-hour window is ~3-8 MB of HTML.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .explorer import _b64, _write
from .sim import D_STATUS

LEG_STATUS = {"pickup": 3, "trip": 4, "trip_to_gateway": 4, "reposition": 5}
STATUS_COLOURS = {1: "#52514e", 2: "#52514e", 3: "#2a78d6", 4: "#eb6834", 5: "#1baf7a"}


class _ShapePool:
    """Shape points of edges used in the window, as extra point ids >= n_nodes."""

    def __init__(self, shapes: dict, node_ll: np.ndarray):
        self.shapes, self.node_ll = shapes, node_ll
        self.n = len(node_ll)
        self.coords: list[np.ndarray] = []
        self.index: dict = {}      # (u, v) -> (first id, count)
        self.size = 0

    def interior(self, u: int, v: int):
        """(point ids, fractions of the edge's length) for u->v's interior points."""
        key = (u, v)
        if key not in self.index:
            pts = self.shapes.get(key)
            if pts is None:
                self.index[key] = None
            else:
                self.index[key] = (self.n + self.size, pts)
                self.coords.append(pts)
                self.size += len(pts)
        rec = self.index[key]
        if rec is None:
            return (), ()
        first, pts = rec
        chain = np.vstack([self.node_ll[u], pts, self.node_ll[v]])
        seg = np.hypot(np.diff(chain[:, 0]), np.diff(chain[:, 1]) * np.cos(np.radians(chain[0, 0])))
        cum = np.cumsum(seg)
        return range(first, first + len(pts)), cum[:-1] / max(cum[-1], 1e-12)


def _tracks(tb, w0, w1, n_drivers, pool: _ShapePool):
    st = tb["status"].sort_values(["driver_id", "t"], kind="stable")
    legs = tb["legs"]
    legs = legs[(legs.t_end >= w0) & (legs.t_start <= w1)]
    by_leg = legs.groupby("driver_id")
    by_st = st.groupby("driver_id")
    t_all, n_all, s_all, m_all, off = [], [], [], [], [0]
    for d in range(n_drivers):
        rows = []  # (t, rank, node, status, move)
        if d in by_st.groups:
            g = by_st.get_group(d)
            # State at the window start. Dated a day earlier so it sorts before any
            # leg keyframes that began before w0: a cab mid-trip at w0 must follow its
            # leg, not jump from the node where its status last changed.
            before = g[g.t <= w0]
            if len(before):
                r = before.iloc[-1]
                rows.append((w0 - 86400, 0, int(r.node), int(r.status), 0))
            else:
                rows.append((w0 - 86400, 0, int(g.node.iloc[0]), 0, 0))
            # rank 2: at equal times a status change follows the leg node it happens at
            # (e.g. the drop-off node, then "idle")
            for r in g[(g.t > w0) & (g.t <= w1)].itertuples():
                rows.append((r.t, 2, int(r.node), int(r.status), 0))
        else:
            rows.append((w0 - 86400, 0, 0, 0, 0))
        if d in by_leg.groups:
            for lg in by_leg.get_group(d).itertuples():
                times, nodes = lg.times.astype(float), lg.nodes
                a = max(int(np.searchsorted(times, w0)) - 1, 0)
                b = min(int(np.searchsorted(times, w1, side="right")) + 1, len(times))
                s = LEG_STATUS.get(lg.kind, 5)
                L = len(times)
                for j in range(a, b):
                    if j > a:   # follow the edge's shape from the previous node
                        ids, frac = pool.interior(int(nodes[j - 1]), int(nodes[j]))
                        for pid, f in zip(ids, frac):
                            tt = times[j - 1] + f * (times[j] - times[j - 1])
                            rows.append((tt, 1 + (j - 1 + f) / L, pid, s, 1))
                    rows.append((times[j], 1 + j / L, int(nodes[j]), s, 0 if j == a else 1))
        rows.sort(key=lambda x: (x[0], x[1]))
        arr = np.array([(t, n, s, m) for t, _, n, s, m in rows])
        t_all.append(arr[:, 0] - w0)
        n_all.append(arr[:, 1])
        s_all.append(arr[:, 2])
        m_all.append(arr[:, 3])
        off.append(off[-1] + len(arr))
    return (np.concatenate(t_all).astype(np.float32), np.concatenate(n_all).astype(np.int64),
            np.concatenate(s_all).astype(np.uint8), np.concatenate(m_all).astype(np.uint8),
            np.array(off, np.int32))


def _status_series(tb, n_drivers, step=300.0, end=86400.0):
    st = tb["status"].sort_values("t")
    grid = pd.DataFrame({"t": np.arange(0, end + 1, step)})
    out = {}
    counts = np.zeros((len(grid), 7), int)
    for d, g in st.groupby("driver_id"):
        idx = np.searchsorted(g.t.to_numpy(), grid.t.to_numpy(), side="right") - 1
        s = np.where(idx >= 0, g.status.to_numpy()[np.maximum(idx, 0)], 0)
        np.add.at(counts, (np.arange(len(grid)), s), 1)
    out["t"] = (grid.t / 3600).round(3).tolist()
    for k, name in enumerate(D_STATUS):
        out[name] = counts[:, k].tolist()
    return out


def build_replay(city, nodes, zone_geoms, tb, drivers, w0_h, w1_h, path, summary: dict, shapes: dict,
                 title: str | None = None):
    w0, w1 = w0_h * 3600, w1_h * 3600
    n_drivers = len(drivers)
    node_ll = nodes[["lat", "lon"]].to_numpy()
    pool = _ShapePool(shapes, node_ll)
    kt, kn, ks, km, koff = _tracks(tb, w0, w1, n_drivers, pool)
    all_ll = np.vstack([node_ll] + pool.coords) if pool.coords else node_ll

    r = tb["requests"]
    r = r[(r.t_request >= w0 - 900) & (r.t_request <= w1)]
    end_wait = r.t_pickup.fillna(r.t_cancel).fillna(w1 + 3600)
    outcome = np.select([r.t_pickup.notna(), r.status == "cancelled", r.status == "expired"], [0, 1, 2], 3)

    used = np.unique(np.concatenate([kn, r.origin_node.to_numpy()]))
    remap = pd.Series(np.arange(len(used)), index=used)

    led = tb["ledger"]
    dl = led[(led.party == "driver") & (led.kind != "online_time")].sort_values(["driver_id", "t_event"])
    loff = np.searchsorted(dl.driver_id.to_numpy(), np.arange(n_drivers + 1)).astype(np.int32)
    r_all = tb["requests"]
    done = r_all[r_all.status == "completed"]
    per_driver = pd.DataFrame({
        "trips": done.groupby("driver_id").size().reindex(range(n_drivers)).fillna(0).astype(int),
    })
    prof = drivers[["vehicle_type", "tenure_days", "rating_sum", "rating_count"]].copy()
    prof["rating"] = (prof.rating_sum / prof.rating_count.replace(0, np.nan)).round(2)

    rq = r_all[r_all.status != "not_raised"]

    def per15(t):  # counts per 15 min within the day (drain-hour events after 24:00 excluded)
        t = t.dropna()
        return np.bincount((t[t < 86400] // 900).astype(int), minlength=96)[:96].tolist()
    kpi = {"x": ((np.arange(96) + 0.5) / 4).tolist(), "requests": per15(rq.t_request),
           "completed": per15(done.t_dropoff), "cancelled": per15(rq.t_cancel)}
    data = {
        "city": city.name, "currency": city.currency, "w0": w0, "w1": w1,
        "zones": json.loads(zone_geoms[["zone", "geometry"]].to_json()),
        "coords": np.round(all_ll[used], 5).ravel().tolist(),
        "kt": _b64(kt), "kn": _b64(remap.loc[kn].to_numpy().astype(np.int32)), "ks": _b64(ks),
        "km": _b64(km), "koff": _b64(koff),
        "req_t": _b64((r.t_request.to_numpy() - w0).astype(np.float32)),
        "req_end": _b64((end_wait.to_numpy() - w0).astype(np.float32)),
        "req_node": _b64(remap.loc[r.origin_node.to_numpy()].to_numpy().astype(np.int32)),
        "req_out": _b64(outcome.astype(np.uint8)),
        "led_t": _b64(dl.t_event.to_numpy().astype(np.float32)),
        "led_a": _b64(dl.amount.to_numpy().astype(np.float32)), "led_off": _b64(loff),
        "profile": prof[["vehicle_type", "tenure_days", "rating"]].to_dict("list"),
        "trips": per_driver.trips.tolist(),
        "status_series": _status_series(tb, n_drivers),
        "kpi": kpi, "summary": summary,
        "colours": {str(k): v for k, v in STATUS_COLOURS.items()},
    }
    data["title"] = title or f"{city.name}: ride company replay"
    _write(path, STAGE3, title or f"{city.name} world: stage 3, ride company replay", data)


STAGE3 = r"""<!doctype html>
<html lang="en"><head><title>__TITLE__</title>__HEAD__
<style>
#fx{position:absolute;inset:0 360px 0 0;pointer-events:none;z-index:450}
.stat{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.kv td:first-child{color:var(--ink2)}
@media (max-width:760px){#fx{inset:0 0 50vh 0}}
</style></head><body>
<div id="map"></div><canvas id="fx"></canvas>
<div id="panel">
  <h1 id="title"></h1>
  <div class="muted" id="sub"></div>

  <h2>Playback</h2>
  <div class="row"><button id="play">❚❚ Pause</button>
    <select id="speed"><option value="10">10×</option><option value="30">30×</option><option value="60" selected>60×</option><option value="120">120×</option><option value="300">300×</option></select>
    <label class="row" style="margin-left:auto"><input type="checkbox" id="trails" checked>trails</label></div>
  <input id="time" type="range" style="width:100%" step="1">
  <div class="row"><b id="clock" style="font-size:20px"></b><span class="muted" id="range"></span></div>

  <h2>Cabs now</h2>
  <div id="legend"></div>
  <div class="stat" style="margin-top:4px"><span><span class="dot" style="border:1.5px solid #0b0b0b;background:transparent"></span>waiting requests</span><b id="n_wait"></b></div>
  <div class="stat"><span><span style="color:#d03b3b;font-weight:bold;margin-right:6px">×</span>cancelled (last 2 min)</span><b id="n_canc"></b></div>

  <h2>Drivers by status, whole day</h2>
  <svg id="c_status" width="100%" height="120" style="display:block"></svg>
  <h2>Requests per 15 min</h2>
  <svg id="c_req" width="100%" height="120" style="display:block"></svg>

  <h2>Selected driver</h2>
  <div id="driver" class="muted">Click a cab on the map.</div>
  <label class="row"><input type="checkbox" id="follow">follow selected cab</label>

  <h2>Day summary</h2>
  <table class="kv" id="summary"></table>
</div>
<script>
__JS_MAP__
document.getElementById('title').textContent = D.title;
const C = D.coords, KT = b64(D.kt, Float32Array), KN = b64(D.kn, Int32Array), KS = b64(D.ks, Uint8Array),
      KM = b64(D.km, Uint8Array), KOFF = b64(D.koff, Int32Array);
const RT = b64(D.req_t, Float32Array), RE = b64(D.req_end, Float32Array), RN = b64(D.req_node, Int32Array), RO = b64(D.req_out, Uint8Array);
const LT = b64(D.led_t, Float32Array), LA = b64(D.led_a, Float32Array), LOFF = b64(D.led_off, Int32Array);
const ND = KOFF.length - 1, SPAN = D.w1 - D.w0;
const NAMES = {1: 'idle', 3: 'to pickup', 4: 'with passenger', 5: 'repositioning'};
const COL = D.colours;
document.getElementById('sub').textContent = `${ND.toLocaleString()} drivers · window ${fmtH(D.w0 / 3600)}–${fmtH(D.w1 / 3600)} · positions from the simulated legs`;

// ---- evaluate a track at time t (seconds since w0): [lat, lon, status] or null
function lastKf(d, t) {
  let lo = KOFF[d], hi = KOFF[d + 1] - 1;
  if (t < KT[lo]) return lo;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (KT[mid] <= t) lo = mid; else hi = mid - 1; }
  return lo;
}
function at(d, t) {
  const k = lastKf(d, t), s = KS[k];
  if (s === 0 || s === 6) return null;
  const a = KN[k];
  let lat = C[2 * a], lon = C[2 * a + 1];
  if (k + 1 < KOFF[d + 1] && KM[k + 1] === 1) {
    const b = KN[k + 1], t0 = KT[k], t1 = KT[k + 1], f = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 1;
    lat += (C[2 * b] - lat) * f; lon += (C[2 * b + 1] - lon) * f;
  }
  return [lat, lon, s === 2 ? 1 : s];
}

// ---- canvas overlay
const fx = document.getElementById('fx'), ctx = fx.getContext('2d');
function resize() { const r = map.getContainer().getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  fx.width = r.width * dpr; fx.height = r.height * dpr; fx.style.width = r.width + 'px'; fx.style.height = r.height + 'px'; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
window.addEventListener('resize', resize); resize();

let simT = Math.min(SPAN * 0.5, 3600), playing = true, last = performance.now(), sel = -1;
const timeEl = document.getElementById('time'); timeEl.max = SPAN; timeEl.value = simT;
timeEl.oninput = () => { simT = +timeEl.value; };
document.getElementById('play').onclick = ev => { playing = !playing; ev.target.textContent = playing ? '❚❚ Pause' : '▶ Play'; };
let counts = {};

// Trail: walk back along the cab's keyframes (graph nodes and edge shape points) from
// its current position to where it was `span` seconds ago, so the trail lies on the
// road it drove; segments fade with age. Stops at the start of the current leg.
function drawTrail(d, p, col, span, R) {
  const t0 = simT - span, lo = KOFF[d];
  let k = lastKf(d, simT), prev = p, tPrev = simT;
  ctx.strokeStyle = col; ctx.lineWidth = R * 0.9; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  while (k >= lo) {
    const s = KS[k]; if (s === 0 || s === 6) break;
    let tk = KT[k], a = KN[k], lat = C[2 * a], lon = C[2 * a + 1];
    if (tk < t0) {                         // cut the last segment at t0
      if (k + 1 >= KOFF[d + 1] || KM[k + 1] !== 1) break;
      const b = KN[k + 1], t1 = KT[k + 1], f = t1 > tk ? (t0 - tk) / (t1 - tk) : 1;
      lat += (C[2 * b] - lat) * f; lon += (C[2 * b + 1] - lon) * f; tk = t0;
    }
    const q = map.latLngToContainerPoint([lat, lon]);
    ctx.globalAlpha = 0.6 * (1 - (simT - (tk + tPrev) / 2) / span);
    ctx.beginPath(); ctx.moveTo(prev.x, prev.y); ctx.lineTo(q.x, q.y); ctx.stroke();
    prev = q; tPrev = tk;
    if (tk <= t0 || KM[k] !== 1) break;     // reached t0, or the start of this leg
    k--;
  }
  ctx.globalAlpha = 1;
}

function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  const speed = +document.getElementById('speed').value;
  if (playing) { simT += dt * speed; if (simT > SPAN) simT = 0; }
  if (document.activeElement !== timeEl) timeEl.value = simT;
  // trail = the last `trailSpan` sim-seconds of the cab's own path (short at any speed)
  const trails = document.getElementById('trails').checked, trailSpan = Math.min(90, Math.max(20, speed));
  ctx.clearRect(0, 0, fx.width, fx.height);
  counts = {1: 0, 3: 0, 4: 0, 5: 0, out: 0};
  const z = map.getZoom(), R = z >= 15 ? 4.5 : z >= 13 ? 3.2 : 2.4;
  // waiting requests: hollow rings; recent cancellations: red crosses
  let nWait = 0, nCanc = 0;
  ctx.lineWidth = 1.2;
  for (let i = 0; i < RT.length; i++) {
    if (RT[i] <= simT && simT < RE[i]) {
      nWait++; const p = map.latLngToContainerPoint([C[2 * RN[i]], C[2 * RN[i] + 1]]);
      ctx.strokeStyle = 'rgba(11,11,11,.75)'; ctx.beginPath(); ctx.arc(p.x, p.y, R + 2.5, 0, 6.283); ctx.stroke();
    } else if (RO[i] === 1 && RE[i] <= simT && simT < RE[i] + 120) {
      nCanc++; const p = map.latLngToContainerPoint([C[2 * RN[i]], C[2 * RN[i] + 1]]), q = R + 1.5;
      ctx.strokeStyle = '#d03b3b'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(p.x - q, p.y - q); ctx.lineTo(p.x + q, p.y + q); ctx.moveTo(p.x + q, p.y - q); ctx.lineTo(p.x - q, p.y + q); ctx.stroke(); ctx.lineWidth = 1.2;
    }
  }
  let selPos = null;
  for (let d = 0; d < ND; d++) {
    const cur = at(d, simT);
    if (!cur) { const k = lastKf(d, simT); if (KS[k] === 6) counts.out++; continue; }
    counts[cur[2]]++;
    const p = map.latLngToContainerPoint([cur[0], cur[1]]), col = COL[cur[2]];
    if (trails && cur[2] !== 1) drawTrail(d, p, col, trailSpan, R);
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 6.283); ctx.fill();
    if (d === sel) selPos = [p, cur];
  }
  if (selPos) {
    const [p, cur] = selPos;
    ctx.strokeStyle = '#0b0b0b'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p.x, p.y, R + 5, 0, 6.283); ctx.stroke();
    if (document.getElementById('follow').checked) map.panTo([cur[0], cur[1]], {animate: false});
  }
  document.getElementById('clock').textContent = fmtH((D.w0 + simT) / 3600);
  document.getElementById('n_wait').textContent = nWait; document.getElementById('n_canc').textContent = nCanc;
  if (now - lastPanel > 250) { lastPanel = now; updatePanel(); }
  requestAnimationFrame(frame);
}

function updatePanel() {
  document.getElementById('legend').innerHTML = [1, 3, 4, 5].map(s =>
    `<div class="stat"><span><span class="dot" style="background:${COL[s]}"></span>${NAMES[s]}</span><b>${counts[s]}</b></div>`).join('') +
    `<div class="stat muted"><span><span class="dot" style="border:1.5px dashed #898781"></span>outside the city (intercity)</span><b>${counts.out}</b></div>`;
  if (sel >= 0) showDriver(sel);
  const h = (D.w0 + simT) / 3600;
  if (Math.abs(h - lastMarker) > 0.05) { lastMarker = h; drawCharts(h); }
}
let lastMarker = -1, lastPanel = 0;

function drawCharts(h) {
  const S = D.status_series;
  lineChart(document.getElementById('c_status'), [
    {name: 'idle', x: S.t, y: S.idle.map((v, i) => v + S.offered[i]), color: COL[1]},
    {name: 'to pickup', x: S.t, y: S.pickup, color: COL[3]},
    {name: 'with passenger', x: S.t, y: S.occupied, color: COL[4]},
    {name: 'repositioning', x: S.t, y: S.repositioning, color: COL[5]},
  ], {ymin: 0, fmt: v => Math.round(v), marker: h, at: (s, hh) => s.y[Math.min(s.y.length - 1, Math.round(hh * 12))],
      onclick: hh => { simT = Math.max(0, Math.min(SPAN, hh * 3600 - D.w0)); }});
  const K = D.kpi;
  lineChart(document.getElementById('c_req'), [
    {name: 'requests', x: K.x, y: K.requests, color: '#0b0b0b'},
    {name: 'completed', x: K.x, y: K.completed, color: COL[4]},
    {name: 'cancelled', x: K.x, y: K.cancelled, color: '#d03b3b'},
  ], {ymin: 0, fmt: v => Math.round(v), marker: h, at: (s, hh) => s.y[Math.min(95, Math.floor(hh * 4))],
      onclick: hh => { simT = Math.max(0, Math.min(SPAN, hh * 3600 - D.w0)); }});
}

function showDriver(d) {
  const t = D.w0 + simT, cur = at(d, simT), k = lastKf(d, simT);
  let earned = 0; for (let j = LOFF[d]; j < LOFF[d + 1] && LT[j] <= t; j++) earned += LA[j];
  const P = D.profile, st = cur ? NAMES[cur[2]] : (KS[k] === 6 ? 'outside the city' : 'offline');
  document.getElementById('driver').innerHTML = `<table class="kv">
    <tr><td>driver</td><td><b>#${d}</b></td></tr><tr><td>status</td><td>${st}</td></tr>
    <tr><td>vehicle</td><td>${P.vehicle_type[d]}</td></tr><tr><td>rating (before today)</td><td>${P.rating[d] ?? '–'}</td></tr>
    <tr><td>tenure</td><td>${P.tenure_days[d]} days</td></tr>
    <tr><td>net earnings so far</td><td>${earned.toFixed(2)} ${D.currency}</td></tr>
    <tr><td>trips today (whole day)</td><td>${D.trips[d]}</td></tr></table>`;
}
map.on('click', ev => {
  const c = map.latLngToContainerPoint(ev.latlng); let best = -1, bd = 144;
  for (let d = 0; d < ND; d++) { const cur = at(d, simT); if (!cur) continue; const p = map.latLngToContainerPoint([cur[0], cur[1]]), dd = (p.x - c.x) ** 2 + (p.y - c.y) ** 2; if (dd < bd) { bd = dd; best = d; } }
  sel = best; if (best >= 0) showDriver(best); else document.getElementById('driver').textContent = 'Click a cab on the map.';
});

const S0 = D.summary;
document.getElementById('summary').innerHTML = [
  ['requests', S0.requests.toLocaleString()], ['completed', (100 * S0.completion_rate).toFixed(1) + '%'],
  ['cancelled', (100 * S0.cancel_rate).toFixed(1) + '%'], ['mean pickup wait', S0.mean_wait_min.toFixed(1) + ' min'],
  ['p90 pickup wait', S0.p90_wait_min.toFixed(1) + ' min'], ['mean fare', S0.mean_fare.toFixed(2) + ' ' + D.currency],
  ['offer acceptance', (100 * S0.offer_acceptance).toFixed(1) + '%'], ['intercity trips', S0.intercity_trips],
  ['driver net / online hour (mean)', S0.mean_hourly.toFixed(2) + ' ' + D.currency],
  ['driver earnings Gini', S0.gini_net_earnings.toFixed(3)],
].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');

if (location.hash === '#demo') { playing = false; simT = SPAN * 0.5; document.getElementById('play').textContent = '▶ Play'; }
if (location.hash === '#start') { playing = false; simT = 60; document.getElementById('play').textContent = '▶ Play'; }  // headless test
if (location.hash === '#play300') { document.getElementById('speed').value = '300'; simT = SPAN * 0.45; }  // headless test of playback
drawCharts((D.w0 + simT) / 3600);
requestAnimationFrame(frame);
</script></body></html>
"""
