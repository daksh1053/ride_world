"""Stage 12: what world model v2 learned. Run after 11_wm2_train.py (training and --eval).

    .venv/bin/python scripts/12_wm2_analysis.py --tag st_small

Writes to outputs/wm2/<outdir>/ (12_results.png) and outputs/wm2/<outdir>/<tag>/ (the rest):
    12_results.png          one-step test scores of every model, per city and channel
    12_rollout.png/.json    feeding the model its own predictions for up to 1 hour ahead,
                            against persistence and climatology at each horizon
    12_probes.png/.json     linear probes on the frozen token states: hidden congestion
                            (never an input) and the day's platform policy, against the
                            same probes on the raw inputs and on an untrained transformer
    12_counterfactual.png   the model's answer to "send 40 extra drivers into this area"
and outputs/<city>/12_wm2_predictions_<tag>.html: a test day, truth next to the one-step
prediction of the model and of climatology, any channel, play / scrub.
"""

from __future__ import annotations

import argparse
import base64
import gzip
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
import torch

from world import cities
from world.hd import raster as R
from world.wm2 import data as WD
from world.wm2 import models as WM

_spec = importlib.util.spec_from_file_location("tr", Path(__file__).with_name("11_wm2_train.py"))
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)
DEV = TR.DEV
OUT = TR.OUT
CH = list(R.CHANNELS)
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}


def load_model(tag):
    ck = torch.load(TR.MODELS / f"{tag}.pt", map_location=DEV)
    m = TR.make_model(ck["kind"], ck["K"]).to(DEV).eval()
    m.load_state_dict(ck["state"])
    return m, ck["K"]


def amp():
    return torch.autocast("cuda", torch.bfloat16, enabled=DEV.type == "cuda")


to_frame = TR.to_frame


def score(out, tgt, mask):
    """Losses of single frames: out (B, N_OUT, G, G), tgt (B, C, G, G)."""
    return WM.losses(out[:, None].float(), tgt[:, None].float(), mask)


# ----------------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(model, K, stores, splits, clim, H=12, n=192):
    res = {}
    for s in stores:
        rng = np.random.default_rng(5)
        te = splits[s.key][2]
        items = [(s, int(rng.choice(te)), int(rng.integers(0, s.T - K - H - 1))) for _ in range(n)]
        bat = WD.Batcher(stores, {}, K + H, DEV)
        acc = {m: np.zeros((H, 3)) for m in ("model", "persistence", "climatology")}
        cnt = 0
        for i in range(0, n, 8):
            x, c, S, _, keys = bat.gather(items[i: i + 8])
            mask = TR.road_mask(S)
            xf = x.float()
            buf = xf.clone()
            last = xf[:, K - 1]
            p_out = torch.zeros(len(keys), WM.N_OUT, *x.shape[-2:], device=DEV)
            p_out[:, :WM.N_COUNT] = torch.log(last[:, WD.COUNT] + 0.1)
            p_out[:, WM.N_COUNT:WM.N_COUNT + WM.N_SURGE] = last[:, WD.SURGE] / 100
            sp = last[:, WD.SPEED]
            p_out[:, -2] = torch.where(sp > 0, sp, torch.full_like(sp, 30.0)) / 60
            p_out[:, -1] = np.log(64.0)
            for h in range(H):
                win = buf[:, h: h + K + 1]
                inp, cn, _ = WD.make_io(win, c[:, h: h + K + 1], S)
                with amp():
                    o = model(inp, cn)[:, -1].float()
                tgt = xf[:, h + K]
                pred = to_frame(o)
                buf[:, h + K, WD.OBS] = pred[:, WD.OBS]
                cl = torch.stack([clim[(cid, int(c[j, h + K, 4] > 0.5))][t + h + K] for j, (cid, _, t) in enumerate(keys)]).to(DEV)
                for name, out in (("model", o), ("persistence", p_out), ("climatology", cl)):
                    ls = score(out, tgt, mask)
                    acc[name][h] += np.array([ls["poisson"].mean().item(), ls["speed"].item(), ls["surge"].item()]) * len(keys)
            cnt += len(keys)
        res[s.key] = {m: (a / cnt).tolist() for m, a in acc.items()}
        print(f"  rollout {s.key}: poisson at 5 / 30 / 60 min  " + "  ".join(
            f"{m} {a[0, 0] / cnt:.4f} {a[5, 0] / cnt:.4f} {a[-1, 0] / cnt:.4f}" for m, a in acc.items()))
    return res


def plot_rollout(res, path):
    fig, axes = plt.subplots(1, 2 * len(res), figsize=(5.2 * 2 * len(res), 4), constrained_layout=True)
    col = {"model": "#2a78d6", "persistence": "#898781", "climatology": "#eb6834"}
    for j, (city, r) in enumerate(res.items()):
        for k, (idx, lab) in enumerate(((0, "count channels, Poisson NLL"), (1, "GPS speed, Gaussian NLL"))):
            ax = axes[2 * j + k]
            for m, a in r.items():
                a = np.array(a)
                ax.plot(np.arange(1, len(a) + 1) * 5, a[:, idx], "o-", ms=3, color=col[m],
                        label={"model": "transformer, own predictions fed back"}.get(m, m))
            ax.set_title(f"{CITY_NAME[city]}: {lab}", fontsize=10)
            ax.set_xlabel("minutes ahead"); ax.grid(alpha=.3)
            if j == 0 and k == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Predicting further ahead (test days, lower is better)", fontsize=12)
    fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- probes
def ridge_fit(Xtr, ytr, Xte, yte, alphas=(0.1, 1, 10, 100, 1000)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean(0)
    nv = len(A) // 5
    best = None
    for a in alphas:                                     # pick alpha on a held-out fifth of train
        w = np.linalg.solve(A[nv:].T @ A[nv:] + a * np.eye(A.shape[1]), A[nv:].T @ (ytr[nv:] - ym))
        e = ((A[:nv] @ w + ym - ytr[:nv]) ** 2).mean()
        if best is None or e < best[0]:
            best = (e, a)
    w = np.linalg.solve(A.T @ A + best[1] * np.eye(A.shape[1]), A.T @ (ytr - ym))
    p = B @ w + ym
    return p, best[1]


def r2(p, y):
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


@torch.no_grad()
def collect(models, K, stores, splits, which, n, seed):
    """Token states at the last position, patch-level hidden congestion, raw input patches."""
    items = TR.fixed_windows(stores, splits, which, n, K, seed)
    bat = WD.Batcher(stores, {}, K, DEV)
    feats = {k: [] for k in models}
    raw, cong, nogps, city, pol = [], [], [], [], []
    for i in range(0, n, 8):
        x, c, S, y, keys = bat.gather(items[i: i + 8])
        inp, cn, _ = WD.make_io(x, c, S)
        for k, m in models.items():
            with amp():
                z = m.encode(inp, cn)[:, -1].float()                    # (B, P, d)
            feats[k].append(z.cpu().numpy())
        B, p = len(keys), 4
        last = inp[:, -1]                                                # (B, Cin, G, G)
        rp = last.unfold(2, p, p).unfold(3, p, p)                        # (B, C, 16, 16, p, p)
        raw.append(rp.permute(0, 2, 3, 1, 4, 5).reshape(B, 256, -1).cpu().numpy())
        road = (S[:, -1] > 0).float()
        yc = y[:, K - 1].float() / 100                                   # congestion now (ratio - 1)
        num = (yc * road).unfold(1, p, p).unfold(2, p, p).sum((-1, -2))
        den = road.unfold(1, p, p).unfold(2, p, p).sum((-1, -2))
        cong.append(torch.where(den > 0, num / den.clamp(min=1), torch.full_like(num, np.nan)).reshape(B, 256).cpu().numpy())
        gs = x[:, K - 1, WD.SAMPLES].float().unfold(1, p, p).unfold(2, p, p).sum((-1, -2))
        nogps.append((gs == 0).reshape(B, 256).cpu().numpy())
        for (cid, d, _) in keys:
            st = stores[cid]
            mt = st.meta[d]
            city.append(cid)
            pol.append([mt["explore"], float(mt["reposition_every"] > 0), mt["max_moves"]])
    return ({k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(raw), np.concatenate(cong),
            np.concatenate(nogps), np.array(city), np.array(pol, float))


def run_probes(model, K, stores, splits):
    rand = TR.make_model("st", K).to(DEV).eval()                       # same architecture, untrained
    models = {"trained": model, "untrained": rand}
    tr = collect(models, K, stores, splits, 0, 400, 21)
    te = collect(models, K, stores, splits, 2, 200, 22)
    res = {"congestion": {}, "policy": {}}
    for cid, s in enumerate(stores):
        rc = {}
        sel_tr, sel_te = tr[4] == cid, te[4] == cid
        ytr, yte = tr[2][sel_tr].ravel(), te[2][sel_te].ravel()
        mtr, mte = ~np.isnan(ytr), ~np.isnan(yte)
        ng = te[3][sel_te].ravel()[mte]
        sources = {"tokens (trained)": (tr[0]["trained"], te[0]["trained"]),
                   "tokens (untrained)": (tr[0]["untrained"], te[0]["untrained"]),
                   "raw input patch": (tr[1], te[1])}
        for name, (a, b) in sources.items():
            A = a[sel_tr].reshape(-1, a.shape[-1])[mtr]
            Bm = b[sel_te].reshape(-1, b.shape[-1])[mte]
            sub = np.random.default_rng(0).permutation(len(A))[:40000]
            p, alpha = ridge_fit(A[sub], ytr[mtr][sub], Bm, yte[mte])
            rc[name] = {"r2_all": r2(p, yte[mte]), "r2_no_gps_now": r2(p[ng], yte[mte][ng]), "alpha": alpha,
                        "share_no_gps": float(ng.mean())}
        res["congestion"][s.key] = rc
        print(f"  congestion probe {s.key}: " + "  ".join(f"{k} R2 {v['r2_all']:.3f} (no GPS {v['r2_no_gps_now']:.3f})" for k, v in rc.items()))
        rp = {}
        for name, key in (("tokens (trained)", "trained"), ("tokens (untrained)", "untrained"), ("raw input patch", None)):
            a = tr[0][key] if key else tr[1]
            b = te[0][key] if key else te[1]
            A, Bm = a[sel_tr].mean(1), b[sel_te].mean(1)
            p, _ = ridge_fit(A, tr[5][sel_tr], Bm, te[5][sel_te], alphas=(1, 10, 100, 1000, 10000))
            yt = te[5][sel_te]
            rp[name] = {"explore_r2": r2(p[:, 0], yt[:, 0]),
                        "rebalancing_on_acc": float(((p[:, 1] > 0.5) == (yt[:, 1] > 0.5)).mean()),
                        "rebalancing_on_base": float(max(yt[:, 1].mean(), 1 - yt[:, 1].mean())),
                        "max_moves_r2": r2(p[:, 2], yt[:, 2])}
        res["policy"][s.key] = rp
        print(f"  policy probe {s.key}: " + "  ".join(f"{k} explore R2 {v['explore_r2']:.2f}, rebal acc {v['rebalancing_on_acc']:.2f} (base {v['rebalancing_on_base']:.2f})" for k, v in rp.items()))
    return res


def plot_probes(res, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    src = ["tokens (trained)", "tokens (untrained)", "raw input patch"]
    col = ["#2a78d6", "#c3c2b7", "#898781"]
    cities_ = list(res["congestion"])
    xs = np.arange(len(cities_))
    for ax, key, title in ((axes[0], "r2_all", "hidden congestion, all patches"),
                           (axes[1], "r2_no_gps_now", "hidden congestion, patches with no GPS this step")):
        for j, s in enumerate(src):
            v = [res["congestion"][c][s][key] for c in cities_]
            ax.bar(xs + (j - 1) * 0.27, v, 0.27, color=col[j], label=s)
            for x_, v_ in zip(xs, v):
                ax.text(x_ + (j - 1) * 0.27, max(v_, 0) + 0.01, f"{v_:.2f}", ha="center", fontsize=8)
        ax.set_xticks(xs, [CITY_NAME[c] for c in cities_]); ax.set_ylabel("R² on test days"); ax.set_title(title, fontsize=10)
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1)
    axes[0].legend(fontsize=8)
    ax = axes[2]
    for j, s in enumerate(src):
        v = [res["policy"][c][s]["explore_r2"] for c in cities_]
        ax.bar(xs + (j - 1) * 0.27, v, 0.27, color=col[j])
        for x_, v_ in zip(xs, v):
            ax.text(x_ + (j - 1) * 0.27, max(v_, 0) + 0.01, f"{v_:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs, [CITY_NAME[c] for c in cities_]); ax.set_title("the day's exploration setting (from one 30-min window)", fontsize=10)
    ax.set_ylim(min(-0.2, ax.get_ylim()[0]), 1)
    fig.suptitle("Linear probes: can a straight line read hidden facts out of the model's state?", fontsize=12)
    fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- counterfactual
@torch.no_grad()
def counterfactual(model, K, stores, splits, path, n_extra=40, H=6):
    """Add n_extra repositioning moves into one 4 x 4 patch (taken from elsewhere) at the
    first step, roll out H steps, and compare with the unchanged rollout."""
    fig, axes = plt.subplots(len(stores), 4, figsize=(16, 4.1 * len(stores)), constrained_layout=True)
    out = {}
    for j, s in enumerate(stores):
        d = int(splits[s.key][2][0]); t = 8 * 12 - K                    # 08:00 on the first test day
        bat = WD.Batcher(stores, {}, K + H, DEV)
        x, c, S, _, keys = bat.gather([(s, d, t)])
        xf = x.float()
        road = s.S[-1] > 0
        idle = s.X[splits[s.key][0][:10], 90:110][:, :, [CH.index(f"idle_{m}") for m in R.MODES3]].astype(float).sum((0, 1, 2))
        dem = s.X[splits[s.key][0][:10], 90:110][:, :, [CH.index(f"new_requests_{m}") for m in R.MODES3]].astype(float).sum((0, 1, 2))
        pat = lambda a: a.reshape(16, 4, 16, 4).sum((1, 3))  # noqa: E731
        ratio = pat(dem) / (pat(idle) + 1)
        ratio[pat(road.astype(float)) < 8] = -1
        to = np.unravel_index(np.argmax(ratio), ratio.shape)            # busiest patch: most requests per idle driver
        src_ = pat(idle); src_[abs(np.arange(16)[:, None] - to[0]) + abs(np.arange(16)[None] - to[1]) < 5] = -1
        fr = np.unravel_index(np.argmax(src_), src_.shape)               # most idle drivers, away from it
        xc = xf.clone()
        a, b = CH.index("dispatch_to"), CH.index("dispatch_from")

        def spread(arr, ij, ch):
            cells = [(4 * ij[0] + u, 4 * ij[1] + v) for u in range(4) for v in range(4) if road[4 * ij[0] + u, 4 * ij[1] + v]]
            for (u, v) in cells:
                arr[0, K, ch, u, v] += n_extra / len(cells)
        spread(xc, to, a); spread(xc, fr, b)
        idle_ch = [CH.index(f"idle_{m}") for m in R.MODES3]
        rep_ch = CH.index("repositioning")
        res = {}
        for name, xx in (("base", xf), ("cf", xc)):
            buf = xx.clone()
            fr_ = []
            for h in range(H):
                inp, cn, _ = WD.make_io(buf[:, h: h + K + 1], c[:, h: h + K + 1], S)
                with amp():
                    o = model(inp, cn)[:, -1].float()
                pred = to_frame(o)
                buf[:, h + K, WD.OBS] = pred[:, WD.OBS]
                fr_.append(pred[0].cpu().numpy())
            res[name] = np.stack(fr_)
        dlt = res["cf"] - res["base"]
        inpatch = lambda arr, ij: arr[..., 4 * ij[0]: 4 * ij[0] + 4, 4 * ij[1]: 4 * ij[1] + 4].sum((-1, -2))  # noqa: E731
        series = {
            "idle drivers in target": inpatch(dlt[:, idle_ch].sum(1), to).tolist(),
            "repositioning drivers (city)": dlt[:, rep_ch].sum((-1, -2)).tolist(),
            "idle drivers in source": inpatch(dlt[:, idle_ch].sum(1), fr).tolist(),
            "unmatched riders in target": inpatch(dlt[:, CH.index("unmatched")], to).tolist(),
        }
        out[s.key] = {"target_patch": [int(v) for v in to], "source_patch": [int(v) for v in fr], "series": series,
                      "date": s.dates[d], "start": "08:00"}
        print(f"  counterfactual {s.key}: " + "; ".join(f"{k} {np.round(v, 2).tolist()}" for k, v in series.items()))
        ax = axes[j, 0]
        ax.imshow(np.where(road, 0.15, 0), cmap="Greys", vmin=0, vmax=1)
        for ij, colr, lab in ((to, "#eb6834", "to (busy)"), (fr, "#2a78d6", "from")):
            ax.add_patch(plt.Rectangle((4 * ij[1] - .5, 4 * ij[0] - .5), 4, 4, fill=False, ec=colr, lw=2))
            ax.text(4 * ij[1] + 2, 4 * ij[0] - 1.5, lab, color=colr, ha="center", fontsize=8)
        ax.set_title(f"{CITY_NAME[s.key]}: move {n_extra} drivers at 08:00", fontsize=10); ax.axis("off")
        for k, h in enumerate((1, H - 1)):
            ax = axes[j, 1 + k]
            m = dlt[h, idle_ch].sum(0)
            v = max(abs(m).max(), 1e-3)
            ax.imshow(np.where(road, m, np.nan), cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(f"change in predicted idle drivers, +{5 * (h + 1)} min", fontsize=10); ax.axis("off")
        ax = axes[j, 3]
        for k_, v_ in series.items():
            ax.plot(np.arange(1, H + 1) * 5, v_, "o-", ms=3, label=k_)
        ax.axhline(0, color="#898781", lw=.8); ax.set_xlabel("minutes after the order"); ax.legend(fontsize=7)
        ax.set_title("difference to the unchanged rollout", fontsize=10); ax.grid(alpha=.3)
    fig.savefig(path, dpi=120); plt.close(fig)
    return out


# ----------------------------------------------------------------------------- prediction viewer
def pack(a):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a, np.uint8).tobytes(), 6)).decode()


@torch.no_grad()
def prediction_page(model, K, s, splits, clim, tag):
    d = int(splits[s.key][2][len(splits[s.key][2]) // 2])
    bat = WD.Batcher([s], {}, K, DEV)
    T = s.T
    show = [CH.index(c) for c in ("idle_cab", "idle_bike", "idle_auto", "with_rider", "new_requests_cab", "new_requests_bike",
                                  "new_requests_auto", "unmatched", "pickups", "dropoffs", "gps_speed_kmh", "surge_cab")]
    true = np.zeros((T, len(show), R.G, R.G), np.float32)
    pm, pc = np.zeros_like(true), np.zeros_like(true)
    wk = int(s.meta[d]["weekday"] >= 5)
    for t0 in range(0, T - K, 8):
        items = [(s, d, t) for t in range(t0, min(t0 + 8, T - K))]
        x, c, S, _, _ = bat.gather(items)
        inp, cn, _ = WD.make_io(x, c, S)
        with amp():
            o = model(inp, cn).float()
        for j, (_, _, t) in enumerate(items):
            for k in range(K):                       # every position predicts frame t + k + 1
                if t == 0 or k == K - 1:
                    f = to_frame(o[j, k][None])[0].cpu().numpy()
                    pm[t + k + 1] = f[show]
    for t in range(T):
        true[t] = s.X[d, t][show].astype(np.float32)
        pc[t] = to_frame(clim[(s.id, wk)][t][None].to(DEV))[0].cpu().numpy()[show]
    scale = 20.0                                     # bytes = value x 20 for counts (fractional rates)
    q = lambda a, sc: np.clip(np.round(a * sc), 0, 255)  # noqa: E731
    sc = np.array([1.0 if CH[i] in ("gps_speed_kmh",) or CH[i].startswith("surge") else scale for i in show])
    Q = lambda a: q(a, sc[None, :, None, None])  # noqa: E731
    pm[0] = true[0]
    data = {"city": CITY_NAME[s.key], "date": s.dates[d], "T": T, "G": R.G, "names": [CH[i] for i in show],
            "scale": sc.tolist(), "road": pack((s.S[-1] > 0).astype(np.uint8)), "tag": tag,
            "truth": pack(Q(true)), "model": pack(Q(pm)), "clim": pack(Q(pc)),
            "policy": {k: s.meta[d][k] for k in ("reposition_every", "max_moves", "explore")}}
    out = cities.get(s.key).out_dir / f"12_wm2_predictions_{tag}.html"
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"), default=float)))
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>World model predictions</title>
<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6;--warm:#eb6834}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#141413;--ink:#f1f0ec;--ink2:#c3c2b7;--muted:#898781;--line:#34332f}}
:root[data-theme="dark"]{--surface:#141413;--ink:#f1f0ec;--ink2:#c3c2b7;--muted:#898781;--line:#34332f}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:20px 16px 40px}h1{font-size:21px;margin:0 0 4px}.sub{color:var(--ink2);margin:0 0 14px}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:8px 0 14px}select,button{font:inherit;padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink)}
input[type=range]{flex:1;min-width:180px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.card{border:1px solid var(--line);border-radius:8px;padding:10px}.card h3{margin:0 0 6px;font-size:13px;font-weight:600}.card .v{color:var(--ink2);font-size:12px}
canvas{width:100%;image-rendering:pixelated;display:block}#chart{height:190px}.note{color:var(--ink2);font-size:13px;max-width:900px}
</style></head><body><main>
<h1 id="title"></h1><p class="sub" id="sub"></p>
<div class="bar"><select id="ch"></select><button id="play">▶ play</button><input type="range" id="t" min="0" max="287" value="100"><b id="time"></b>
<label><input type="checkbox" id="log" checked> log colour</label><label><input type="checkbox" id="blk" checked> sum over 4 × 4 blocks</label></div>
<div class="grid">
<div class="card"><h3>truth (the simulator)</h3><canvas id="c0" width="64" height="64"></canvas><div class="v" id="v0"></div></div>
<div class="card"><h3 id="mname"></h3><canvas id="c1" width="64" height="64"></canvas><div class="v" id="v1"></div></div>
<div class="card"><h3>climatology (training-day average)</h3><canvas id="c2" width="64" height="64"></canvas><div class="v" id="v2"></div></div>
<div class="card"><h3>model error: prediction minus truth</h3><canvas id="c3" width="64" height="64"></canvas><div class="v" id="v3">blue = predicted too low, red = too high</div></div>
</div>
<h3 style="margin:18px 0 6px;font-size:14px">city total over the day</h3>
<canvas id="chart"></canvas>
<p class="note">The model sees the true frames up to 5 minutes before each shown moment, plus the platform's dispatch orders and the clock and rain, and predicts the next frame. Count channels show expected values, so a prediction is smooth where the truth is a few whole vehicles: the model gives the rate, the simulator draws the actual events. GPS speed is shown only in cells where cabs drove in the truth. With "sum over 4 × 4 blocks" on, each block shows its total, which makes the smooth predictions and the grainy truth easier to compare by eye.</p>
</main>
<script>
const D=__DATA__;
async function unpack(b){const bin=Uint8Array.from(atob(b),c=>c.charCodeAt(0));const s=new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));return new Uint8Array(await new Response(s).arrayBuffer());}
(async()=>{
const G=D.G,T=D.T,NC=D.names.length;
const [tru,mod,cli,road]=await Promise.all([unpack(D.truth),unpack(D.model),unpack(D.clim),unpack(D.road)]);
document.getElementById('title').textContent=`${D.city}: the world model's next-5-minute predictions on a test day (${D.date})`;
document.getElementById('sub').textContent=`A day the model never trained on. Platform setting that day: rebalancing every ${D.policy.reposition_every||'never'} min, up to ${D.policy.max_moves} moves, exploration ${D.policy.explore}. Model: ${D.tag}.`;
document.getElementById('mname').textContent='world model prediction ('+D.tag+')';
const sel=document.getElementById('ch');D.names.forEach((n,i)=>{const o=document.createElement('option');o.value=i;o.textContent=n;sel.appendChild(o)});
sel.value=D.names.indexOf('new_requests_cab');
const raw=(a,t,c,p)=>a[((t*NC+c)*G*G)+p]/D.scale[c];
const blk=()=>document.getElementById('blk').checked;
let fcache={};
function frame(a,t,c){const key=[a===tru?0:a===mod?1:2,t,c,blk()].join();if(fcache[key])return fcache[key];const f=new Float32Array(G*G);
if(!blk()||isSpeed(c)){for(let p=0;p<G*G;p++)f[p]=raw(a,t,c,p)}else{for(let bi=0;bi<G;bi+=4)for(let bj=0;bj<G;bj+=4){let s=0;for(let u=0;u<4;u++)for(let v=0;v<4;v++)s+=raw(a,t,c,(bi+u)*G+bj+v);for(let u=0;u<4;u++)for(let v=0;v<4;v++)f[(bi+u)*G+bj+v]=s}}
if(Object.keys(fcache).length>400)fcache={};return fcache[key]=f}
const val=(a,t,c,p)=>frame(a,t,c)[p];
const isSpeed=c=>D.names[c]==='gps_speed_kmh';
function maxOf(c){let m=0;for(let t=0;t<T;t+=3){const a=frame(tru,t,c),b=frame(mod,t,c);for(let p=0;p<G*G;p++){m=Math.max(m,a[p],b[p])}}fcache={};return m||1}
let cmax={};
function col(v,m,lg){let x=lg?Math.log1p(v)/Math.log1p(m):v/m;x=Math.max(0,Math.min(1,x));return [Math.round(245-x*215),Math.round(245-x*125),Math.round(245-x*30)]}
function draw(id,arr,t,c){const cv=document.getElementById(id),ctx=cv.getContext('2d'),im=ctx.createImageData(G,G);const ck=c+'_'+blk(),m=cmax[ck]||(cmax[ck]=maxOf(c)),lg=document.getElementById('log').checked&&!isSpeed(c);let tot=0;
for(let p=0;p<G*G;p++){let v=val(arr,t,c,p);const r=road[p];let rgb=[r?236:250,r?236:250,r?232:250];
if(isSpeed(c)){if(raw(tru,t,c,p)>0&&r){rgb=col(v,m,false)}else v=0}else if(v>0.02)rgb=col(v,m,lg);
tot+=v;im.data.set([...rgb,255],p*4)}ctx.putImageData(im,0,0);return tot}
function drawErr(t,c){const cv=document.getElementById('c3'),ctx=cv.getContext('2d'),im=ctx.createImageData(G,G);const m=(cmax[c+'_'+blk()]||1)/3;
for(let p=0;p<G*G;p++){let e=val(mod,t,c,p)-val(tru,t,c,p);if(isSpeed(c)&&raw(tru,t,c,p)==0)e=0;const x=Math.max(-1,Math.min(1,e/m));
const rgb=x>0?[245,Math.round(245-x*170),Math.round(245-x*190)]:[Math.round(245+x*200),Math.round(245+x*130),245];im.data.set([...rgb,255],p*4)}ctx.putImageData(im,0,0)}
const tot=(a,t,c)=>{let s=0,n=0;for(let p=0;p<G*G;p++){const tv=raw(tru,t,c,p);if(isSpeed(c)){if(tv>0){s+=raw(a,t,c,p);n++}}else s+=raw(a,t,c,p)}return isSpeed(c)?(n?s/n:0):s};
function chart(c,t){const cv=document.getElementById('chart'),W=cv.clientWidth,H=190;cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio;const x=cv.getContext('2d');x.scale(devicePixelRatio,devicePixelRatio);
const cs=getComputedStyle(document.documentElement);const ser=[[tru,cs.getPropertyValue('--ink')],[mod,cs.getPropertyValue('--accent')],[cli,cs.getPropertyValue('--warm')]].map(([a,k])=>[Array.from({length:T},(_,i)=>tot(a,i,c)),k]);
const mx=Math.max(...ser.flatMap(s=>s[0]))||1;x.strokeStyle=cs.getPropertyValue('--line');x.strokeRect(40,5,W-50,H-30);
x.fillStyle=cs.getPropertyValue('--muted');x.font='11px system-ui';x.fillText(mx.toFixed(isSpeed(c)?0:0),4,14);x.fillText('0',28,H-25);[0,6,12,18,24].forEach(h=>x.fillText(h+':00',40+(W-50)*h*12/T-12,H-8));
ser.forEach(([s,k],j)=>{x.strokeStyle=k;x.lineWidth=j==0?2:1.5;x.beginPath();s.forEach((v,i)=>{const px=40+(W-50)*i/T,py=5+(H-30)*(1-v/mx);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()});
x.strokeStyle=cs.getPropertyValue('--muted');x.beginPath();const px=40+(W-50)*t/T;x.moveTo(px,5);x.lineTo(px,H-25);x.stroke();
[['truth',0],['model',1],['climatology',2]].forEach(([n,j],i)=>{x.fillStyle=ser[j][1];x.fillText('— '+n,W-230+i*75,16)})}
function render(){const t=+document.getElementById('t').value,c=+sel.value;const hh=String(Math.floor(t/12)).padStart(2,'0'),mm=String(t%12*5).padStart(2,'0');document.getElementById('time').textContent=hh+':'+mm;
const unit=isSpeed(c)?' km/h mean':' in the city';const f=v=>isSpeed(c)?v.toFixed(1):v.toFixed(0);
document.getElementById('v0').textContent='total '+f(tot(tru,t,c))+unit;draw('c0',tru,t,c);
draw('c1',mod,t,c);document.getElementById('v1').textContent='total '+f(tot(mod,t,c))+unit;
draw('c2',cli,t,c);document.getElementById('v2').textContent='total '+f(tot(cli,t,c))+unit;drawErr(t,c);chart(c,t)}
let timer=null;document.getElementById('play').onclick=e=>{if(timer){clearInterval(timer);timer=null;e.target.textContent='▶ play'}else{timer=setInterval(()=>{const s=document.getElementById('t');s.value=(+s.value+1)%T;render()},120);e.target.textContent='❚❚ pause'}};
['t','ch','log','blk'].forEach(id=>document.getElementById(id).addEventListener('input',render));window.addEventListener('resize',render);render();
})();
</script></body></html>"""


# ----------------------------------------------------------------------------- results plot
def plot_results(path):
    r = json.loads((OUT / "results.json").read_text())
    models = list(r)
    cities_ = list(next(iter(r.values())))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    pal = ["#898781", "#eb6834", "#c3c2b7", "#2a78d6", "#1b4f8f", "#7aa7e0"]
    xs = np.arange(len(cities_))
    w = 0.8 / len(models)
    for ax, key, title in ((axes[0], "poisson_mean", "count channels, Poisson NLL"), (axes[1], "speed_nll", "GPS speed, Gaussian NLL")):
        for j, m in enumerate(models):
            v = [r[m][c][key] for c in cities_]
            ax.bar(xs + (j - (len(models) - 1) / 2) * w, v, w, color=pal[j % len(pal)], label=m)
            for x_, v_ in zip(xs, v):
                ax.text(x_ + (j - (len(models) - 1) / 2) * w, v_, f"{v_:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(xs, [CITY_NAME[c] for c in cities_]); ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    ax = axes[2]
    names = WD.COUNT_NAMES
    best = [m for m in models if m not in ("persistence", "climatology")]
    for j, m in enumerate(["climatology"] + best[:2]):
        gain = [np.mean([r["persistence"][c][f"nll_{n}"] - r[m][c][f"nll_{n}"] for c in cities_]) for n in names]
        ax.barh(np.arange(len(names)) + (j - 1) * 0.27, gain, 0.27, color=pal[(models.index(m)) % len(pal)], label=m)
    ax.set_yticks(np.arange(len(names)), names, fontsize=7); ax.invert_yaxis()
    ax.set_title("improvement over persistence per channel (both cities)", fontsize=10); ax.legend(fontsize=8)
    fig.suptitle("Next-5-minute prediction on held-out test days (lower NLL is better)", fontsize=12)
    fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="st_small")
    ap.add_argument("--only", nargs="*", default=["results", "rollout", "probes", "cf", "page"])
    ap.add_argument("--outdir", default="step3", help="outputs/wm2/<outdir>/; per-model files in <outdir>/<tag>/")
    args = ap.parse_args()
    global OUT
    OUT = cities.OUTPUTS / "wm2" / args.outdir
    TR.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    MO = OUT / args.tag
    MO.mkdir(exist_ok=True)
    model, K = load_model(args.tag)
    stores, splits = TR.load(TR.CITIES)
    clim = None
    if {"rollout", "page"} & set(args.only):
        clim = TR.climatology(stores, splits)
    if "results" in args.only and (OUT / "results.json").exists():
        plot_results(OUT / "12_results.png")
    if "rollout" in args.only:
        res = rollout(model, K, stores, splits, clim)
        (MO / "12_rollout.json").write_text(json.dumps(res, indent=1))
        plot_rollout(res, MO / "12_rollout.png")
    if "probes" in args.only:
        res = run_probes(model, K, stores, splits)
        (MO / "12_probes.json").write_text(json.dumps(res, indent=1))
        plot_probes(res, MO / "12_probes.png")
    if "cf" in args.only:
        res = counterfactual(model, K, stores, splits, MO / "12_counterfactual.png")
        (MO / "12_counterfactual.json").write_text(json.dumps(res, indent=1))
    if "page" in args.only:
        for s in stores:
            prediction_page(model, K, s, splits, clim, args.tag)


if __name__ == "__main__":
    main()
