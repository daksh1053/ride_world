"""Stage 10b: an interactive page showing exactly what world model v2 sees for one day.

    .venv/bin/python scripts/10b_wm2_viewer.py --city sf_hd [--date 2025-01-08]

Writes outputs/<city>/10_wm2_viewer.html: every input channel as a 64 x 64 map over the
day (5-minute steps, play / scrub), all channels at once as small multiples, the static
layers, the context, and the hidden congestion that the model never sees (kept to test
later whether it learned traffic from GPS speeds alone). The day is embedded gzip-packed.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from world import cities
from world.hd import raster as R


def pack(a: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a, np.uint8).tobytes(), 6)).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["sf_hd", "pune_hd"])
    ap.add_argument("--date", default=None, help="default: the third day of the dataset")
    args = ap.parse_args()
    city = cities.get(args.city)
    d = city.data_dir / "wm2"
    days = pd.concat([pd.read_csv(f) for f in sorted(d.glob("days*.csv"))]).sort_values("date").reset_index(drop=True)
    date = args.date or days.date.iat[min(2, len(days) - 1)]
    X, C, Y, meta = R.load_day(d / f"{date}.npz")
    st = np.load(d / "static.npz")
    ch = list(R.CHANNELS)
    # quantise to bytes: counts as is; surge (m - 1) x 100; speed km/h; congestion (ratio - 1) x 100
    Q = X.copy()
    scale = {}
    for i, c in enumerate(ch):
        if c.startswith("surge"):
            Q[:, i] *= 100
            scale[c] = 0.01
        else:
            scale[c] = 1.0
    Qy = np.clip((Y - 1) * 100, 0, 255)
    mask = st["S"][R.STATIC.index("has_road")] > 0
    Qy[:, ~mask] = 0
    used = [c for i, c in enumerate(ch) if X[:, i].max() > 0]
    row = days[days.date == date].iloc[0].to_dict()
    data = {
        "city": city.name, "date": date, "weekday": pd.Timestamp(date).day_name(), "G": R.G,
        "T": int(X.shape[0]), "cell_m": float(st["grid"][2]), "channels": ch, "used": used, "scale": scale,
        "static_names": list(R.STATIC), "context_names": list(R.CONTEXT),
        "X": pack(np.clip(np.round(Q), 0, 255)), "Y": pack(np.round(Qy)),
        "S": pack(np.round(st["S"] * 255)), "C": np.round(C, 3).tolist(),
        "policy": {k: row.get(k) for k in ("reposition_every", "max_moves", "explore")},
        "summary": {k: row.get(k) for k in ("requests", "completion", "wait_min", "mean_surge", "rain", "drivers_online")},
        "meta": meta, "n_days": int(len(days)),
    }
    out = city.out_dir / "10_wm2_viewer.html"
    out.write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"), default=float)))
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>What the world model sees</title>
<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6;--hidden:#eb6834}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#141413;--ink:#f1f0ec;--ink2:#c3c2b7;--muted:#898781;--line:#34332f}}
:root[data-theme="dark"]{--surface:#141413;--ink:#f1f0ec;--ink2:#c3c2b7;--muted:#898781;--line:#34332f}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1320px;margin:0 auto;padding:18px 16px 40px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:14px;margin:22px 0 8px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2)}
.muted{color:var(--muted)} .row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
.card{border:1px solid var(--line);border-radius:10px;padding:12px}
canvas{image-rendering:pixelated;display:block;border-radius:4px;background:#0000}
#main{width:100%;aspect-ratio:1}
.ctrl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}
button,select{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);cursor:pointer}
input[type=range]{flex:1;min-width:180px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(118px,1fr));gap:10px}
.cell{cursor:pointer;border:1px solid var(--line);border-radius:8px;padding:6px}
.cell.on{outline:2px solid var(--accent)} .cell canvas{width:100%;aspect-ratio:1}
.cell div{font-size:11px;color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kv{display:grid;grid-template-columns:auto auto;gap:2px 14px;font-size:13px} .kv b{font-weight:600;text-align:right}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:9px;border:1px solid var(--hidden);color:var(--hidden)}
.legend{height:10px;border-radius:3px;margin-top:6px}
#side{flex:1;min-width:280px}
</style></head><body><main>
<h1 id="title"></h1>
<div class="muted" id="sub"></div>

<h2>One channel over the day</h2>
<div class="row">
  <div style="flex:0 1 512px;min-width:260px"><canvas id="main" width="64" height="64"></canvas>
    <div class="legend" id="leg"></div><div class="muted" style="display:flex;justify-content:space-between;font-size:12px"><span>0</span><span id="legmax"></span></div></div>
  <div id="side">
    <div class="ctrl"><select id="chan"></select>
      <label><input type="checkbox" id="roads" checked> roads underneath</label>
      <label><input type="checkbox" id="logs" checked> log colour</label></div>
    <div class="ctrl"><button id="play">▶ play</button><input type="range" id="t" min="0" value="0"><b id="clock" style="min-width:48px"></b></div>
    <div class="card"><div class="kv" id="now"></div></div>
    <p class="muted" style="font-size:13px">Every 5 minutes the model receives all the channels below as a stack of 64 × 64 maps
      (<span id="nnum"></span> numbers per step, against 6 numbers for each of ~150 zones in world model v1), plus the static layers and
      the context. Counts are over the last 5 minutes for flows, or at the moment for the fleet. The two <i>dispatch</i> channels are the platform's
      actions: where it told drivers to move from and to.</p>
  </div>
</div>

<h2>Everything the model sees at this moment <span class="muted" style="text-transform:none;letter-spacing:0">(click to enlarge)</span></h2>
<div class="grid" id="all"></div>

<h2>Hidden from the model <span class="tag">truth, never an input</span></h2>
<div class="row">
  <div class="card"><canvas id="hid" width="64" height="64" style="width:300px;aspect-ratio:1"></canvas>
    <div style="font-size:12px;margin-top:6px">true congestion: travel time ÷ free-flow time on the cell's roads (white = free flow, dark = 2.5× or more)</div></div>
  <div class="card"><canvas id="gps" width="64" height="64" style="width:300px;aspect-ratio:1"></canvas>
    <div style="font-size:12px;margin-top:6px">what the model gets instead: GPS speed of moving cabs (km/h), only where cabs happen to drive</div></div>
  <div style="flex:1;min-width:260px" class="muted"><p style="font-size:13px">A world model that learned real world features should be able to recover the left map
    from the right one and from how its cabs' trips slow down. Step 4 tests this with a probe, along with the hidden rain effect, zone types
    and each driver's acceptance tendency.</p></div>
</div>

<h2>Static layers <span class="muted" style="text-transform:none;letter-spacing:0">(same every step)</span></h2>
<div class="grid" id="stat"></div>

<h2>Context per step</h2>
<canvas id="ctx" width="1200" height="120" style="width:100%;height:120px"></canvas>
<div class="muted" style="font-size:12px" id="ctxleg"></div>
</main>
<script>
const D = __DATA__;
const G = D.G, GG = G * G, NC = D.channels.length;
async function unpack(b64) {
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const ds = new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(ds).arrayBuffer());
}
const RAMP = [[247,251,255],[198,219,239],[107,174,214],[33,113,181],[8,48,107]];
const HOT = [[255,255,255],[254,224,144],[253,174,97],[244,109,67],[165,0,38]];
function col(v, ramp) { v = Math.max(0, Math.min(1, v)) * (ramp.length - 1); const i = Math.min(Math.floor(v), ramp.length - 2), f = v - i;
  return ramp[i].map((a, k) => Math.round(a + (ramp[i + 1][k] - a) * f)); }
let X, Y, S, t = 0, chan = D.used.includes('with_rider') ? D.channels.indexOf('with_rider') : 0, timer = null;
const maxOf = {};
function frame(c, tt) { const o = (tt * NC + c) * GG; return X.subarray(o, o + GG); }
function draw(cv, vals, vmax, opt = {}) {
  const ctx = cv.getContext('2d'), im = ctx.createImageData(G, G), roads = S.subarray(0, GG);
  const lg = opt.log ?? document.getElementById('logs').checked, under = opt.roads ?? document.getElementById('roads').checked;
  for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
    const src = r * G + c, dst = ((G - 1 - r) * G + c) * 4;   // row 0 = south
    const v = vals[src]; let rgb;
    if (v > 0) { const f = lg ? Math.log1p(v) / Math.log1p(vmax) : v / vmax;
      rgb = col(opt.ramp ? f : 0.3 + 0.7 * f, opt.ramp || RAMP); }            // any non-zero count is clearly blue
    else if (under && roads[src] > 0) { const g = 255 - roads[src] * 0.12; rgb = [g, g, g - 2]; }
    else rgb = opt.bg || [252, 252, 251];
    im.data.set([...rgb, 255], dst);
  }
  ctx.putImageData(im, 0, 0);
}
function fmtT(k) { const m = (k + 1) * 5; return `${String(Math.floor(m / 60) % 24).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`; }
function sum(a) { let s = 0; for (const v of a) s += v; return s; }
function render() {
  const name = D.channels[chan], sc = D.scale[name];
  draw(document.getElementById('main'), frame(chan, t), maxOf[chan]);
  document.getElementById('leg').style.background = `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map(v => `rgb(${col(v, RAMP)})`).join(',')})`;
  document.getElementById('legmax').textContent = (maxOf[chan] * sc).toFixed(sc < 1 ? 2 : 0) + (name.startsWith('surge') ? ' (surge − 1)' : '');
  document.getElementById('clock').textContent = fmtT(t);
  const tot = n => sum(frame(D.channels.indexOf(n), t));
  const ctx = D.C[t];
  document.getElementById('now').innerHTML = [
    ['idle vehicles', tot('idle_bike') + tot('idle_auto') + tot('idle_cab')], ['to pickup / with rider', `${tot('to_pickup')} / ${tot('with_rider')}`],
    ['new requests (5 min)', tot('new_requests_bike') + tot('new_requests_auto') + tot('new_requests_cab')], ['unmatched riders', tot('unmatched')],
    ['dispatches (5 min)', tot('dispatch_from')], ['rain (context)', ctx[5].toFixed(2)],
    [`${name} total`, (sum(frame(chan, t)) * sc).toFixed(sc < 1 ? 1 : 0)]].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join('');
  document.querySelectorAll('#all canvas').forEach(cv => draw(cv, frame(+cv.dataset.c, t), maxOf[+cv.dataset.c]));
  document.querySelectorAll('#all .cell').forEach(e => e.classList.toggle('on', +e.dataset.c === chan));
  draw(document.getElementById('hid'), Y.subarray(t * GG, (t + 1) * GG), 150, {log: false, ramp: HOT, roads: true});
  const sp = D.channels.indexOf('gps_speed_kmh');
  draw(document.getElementById('gps'), frame(sp, t), 60, {log: false});
  drawCtx();
}
function drawCtx() {
  const cv = document.getElementById('ctx'), c = cv.getContext('2d'), W = cv.width, H = cv.height;
  c.clearRect(0, 0, W, H);
  const cols = ['#2a78d6', '#1baf7a', '#eda100', '#8a5bd6', '#c3c2b7', '#eb6834'];
  D.context_names.forEach((n, j) => { c.strokeStyle = cols[j]; c.lineWidth = 2; c.beginPath();
    D.C.forEach((r, k) => { const x = k / (D.T - 1) * W, y = H - 6 - (r[j] + 1) / 2 * (H - 12); k ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke(); });
  c.strokeStyle = '#898781'; c.beginPath(); const x = t / (D.T - 1) * W; c.moveTo(x, 0); c.lineTo(x, H); c.stroke();
  document.getElementById('ctxleg').innerHTML = D.context_names.map((n, j) => `<span style="color:${cols[j]}">■</span> ${n}`).join(' &nbsp; ') + ' &nbsp; (all scaled to −1..1)';
}
(async () => {
  [X, Y, S] = await Promise.all([unpack(D.X), unpack(D.Y), unpack(D.S)]);
  for (let c = 0; c < NC; c++) { let m = 0; for (let k = 0; k < D.T; k++) { const f = frame(c, k); for (const v of f) if (v > m) m = v; } maxOf[c] = Math.max(m, 1); }
  document.getElementById('title').textContent = `${D.city}: what world model v2 sees on ${D.weekday} ${D.date}`;
  const p = D.policy, s = D.summary;
  document.getElementById('sub').textContent = `64 × 64 grid, cells of ${Math.round(D.cell_m)} m, 288 steps of 5 min · dataset of ${D.n_days} days · ` +
    `this day: ${s.requests} requests, ${(100 * s.completion).toFixed(1)}% completed, platform setting: ${p.reposition_every > 0 ? `rebalance every ${p.reposition_every / 2} min` : 'no rebalancing'}, ` +
    `up to ${p.max_moves} moves, exploration ${p.explore}`;
  document.getElementById('nnum').textContent = (D.used.length * GG).toLocaleString();
  const sel = document.getElementById('chan');
  D.channels.forEach((n, i) => { if (D.used.includes(n)) sel.add(new Option(n, i)); });
  sel.value = chan; sel.onchange = () => { chan = +sel.value; render(); };
  document.getElementById('all').innerHTML = D.channels.map((n, i) => D.used.includes(n) ?
    `<div class="cell" data-c="${i}"><canvas width="64" height="64" data-c="${i}"></canvas><div>${n}</div></div>` : '').join('');
  document.querySelectorAll('#all .cell').forEach(e => e.onclick = () => { chan = +e.dataset.c; sel.value = chan; render(); });
  document.getElementById('stat').innerHTML = D.static_names.map((n, i) =>
    `<div class="cell"><canvas width="64" height="64" data-s="${i}"></canvas><div>${n}</div></div>`).join('');
  document.querySelectorAll('#stat canvas').forEach(cv => draw(cv, S.subarray(+cv.dataset.s * GG, (+cv.dataset.s + 1) * GG), 255, {log: false, roads: false}));
  const tr = document.getElementById('t'); tr.max = D.T - 1; t = 17 * 12; tr.value = t;
  tr.oninput = () => { t = +tr.value; render(); };
  document.getElementById('play').onclick = e => { if (timer) { clearInterval(timer); timer = null; e.target.textContent = '▶ play'; }
    else { timer = setInterval(() => { t = (t + 1) % D.T; tr.value = t; render(); }, 150); e.target.textContent = '❚❚ pause'; } };
  ['roads', 'logs'].forEach(id => document.getElementById(id).onchange = render);
  render();
})();
</script></body></html>
"""

if __name__ == "__main__":
    main()
