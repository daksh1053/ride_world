"""World model 3, stage 1: the entity-level world state and what the model must predict.

    .venv/bin/python scripts/16_wm3_world.py                 # both cities

For each city it rebuilds, from the simulator's logs, the full world at every 30-second
decision epoch (world/wm3/state.py: every driver on the road graph, every open request,
every road's travel time, the platform's actions and the events of the next 30 s), checks
the rebuild against the simulator's own per-zone counts, and writes to outputs/wm3/stage1/:

    16_verify.png             rebuilt vs simulator counts per zone over a whole day
    16_snapshot_<city>.png    the world at 08:00 on the street map, with a zoomed area
    16_driver_day.png         two drivers' days: statuses, legs, offers, orders, income
    16_targets.png            the events the model must predict per 30-second step
    16_hidden.png             hidden driver traits vs what their history shows
    <city>_world.html         interactive viewer 07:00-10:00 (pan, zoom, play, click a
                              driver or request to see its exact record and next events)
    16_stage1.json            the numbers
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from world import cities
from world.wm3 import state as S

OUT = cities.OUTPUTS / "wm3" / "stage1"
CITY_NAME = {"sf_hd": "San Francisco", "pune_hd": "Pune"}
W0, W1 = 7 * 3600, 10 * 3600                       # viewer window
# status -> (label, colour, filled). Three hues + neutral; hollow marks the rarer twin states.
LOOK = {1: ("idle", "#898781", True), 2: ("offered", "#2a78d6", False), 3: ("to pickup", "#2a78d6", True),
        4: ("with rider", "#eb6834", True), 5: ("repositioning", "#1baf7a", True), 6: ("outside city", "#eb6834", False),
        7: ("break", "#898781", False)}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONG = plt.matplotlib.colors.LinearSegmentedColormap.from_list("cong", ["#d9d4f5", "#6250d6", "#2c1f80"])


def day_dirs(key):
    val_from = json.loads((cities.get(key).data_dir / "wm2" / "split.json").read_text())["val_from"]
    return [d for d in sorted((cities.get(key).data_dir / "hd_runs").glob("wm2_seed*/20*")) if d.name < val_from]


def pick_day(key):
    """The first training-split Wednesday of March 2025 on which the platform issued at least
    1,000 repositioning orders (the policy varies between training days)."""
    ds = day_dirs(key)
    wed = [d for d in ds if pd.Timestamp(d.name).weekday() == 2 and d.name.startswith("2025-03")]
    for d in wed:
        if pd.read_parquet(d / "actions.parquet", columns=["n_dispatches"]).n_dispatches.sum() >= 1000:
            return d
    return (wed or ds)[0]


# ------------------------------------------------------------------ verification
def verify(L, g):
    c = L.context
    Z = int(c.zone.max()) + 1
    rows = []
    for t in range(300, 86400, 300):
        e = int(t / S.DT)
        cz = c[c.t == t].sort_values("zone")
        if not len(cz):
            continue
        st, z = L.status[:, e], g.zone[L.stop[:, e]]
        idle = np.bincount(z[st == 1], minlength=Z)
        busy = np.bincount(z[(st != 0) & (st != 1) & (st != 6) & (st != 7)], minlength=Z)
        r = L.open_requests(e)
        pend = np.bincount(g.zone[r.origin[r.status <= 1]], minlength=Z)
        rows.append({"t": t, **{f"{k}_{w}": v for k, a, b in (("idle", idle, cz.idle_drivers), ("busy", busy, cz.busy_drivers),
                                                               ("pending", pend, cz.pending))
                                for w, v in (("rebuilt", int(a.sum())), ("sim", int(b.sum())),
                                             ("cell_err", int(np.abs(a - b.to_numpy()).sum())))}})
    return pd.DataFrame(rows)


def plot_verify(vs, path):
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.4), constrained_layout=True)
    for i, (key, v) in enumerate(vs.items()):
        for j, k in enumerate(["idle", "busy", "pending"]):
            ax = axs[i, j]
            h = v.t / 3600
            ax.plot(h, v[f"{k}_sim"], color=INK, lw=2, label="simulator's own count")
            ax.plot(h, v[f"{k}_rebuilt"], "o", ms=2.2, color="#2a78d6", label="rebuilt from the logs")
            share = v[f"{k}_cell_err"].sum() / max(v[f"{k}_sim"].sum(), 1)
            ax.set_title(f"{CITY_NAME[key]}: {k} {'drivers' if k != 'pending' else 'requests'} — "
                         f"{100 * (1 - share):.2f}% placed in the right zone", fontsize=9.5)
            ax.set_xlabel("hour of day", fontsize=8); ax.tick_params(labelsize=7.5); ax.set_xlim(0, 24)
            ax.grid(color=GRID, lw=.6); ax.spines[["top", "right"]].set_visible(False)
        axs[i, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Check: the entity-level world rebuilt every 30 s from the logs reproduces the simulator's counts "
                 "(totals and zone by zone, every 5 min of the day)", fontsize=11)
    fig.savefig(path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------- map snapshot
def draw_world(ax, L, g, e, box=None, lw=0.35, ms=9):
    rr = L.road_ratio(e)
    seg = np.stack([np.c_[g.x[g.eu], g.y[g.eu]], np.c_[g.x[g.ev], g.y[g.ev]]], 1)
    if box is not None:
        x0, x1, y0, y1 = box
        keep = (seg[:, :, 0].max(1) > x0) & (seg[:, :, 0].min(1) < x1) & (seg[:, :, 1].max(1) > y0) & (seg[:, :, 1].min(1) < y1)
    else:
        keep = np.ones(len(seg), bool)
    jam = keep & (rr >= 1.15)
    ax.add_collection(LineCollection(seg[keep & ~jam], colors="#cfcdc6", linewidths=lw, zorder=1))
    lc = LineCollection(seg[jam], array=np.clip(rr[jam], 1, 3), cmap=CONG, norm=plt.Normalize(1, 3),
                        linewidths=lw * 3.2, zorder=2)
    ax.add_collection(lc)
    D = L.drivers_at(e)
    R = L.open_requests(e)
    pos = D.set_index("driver_id")[["x", "y"]]
    # assignment lines: drivers heading to a rider
    a = D[(D.status == 3) & (D.request_id >= 0)]
    if len(a):
        o = R.set_index("request_id").reindex(a.request_id).origin.to_numpy()
        ok = ~np.isnan(o)
        ls = np.stack([np.c_[a.x[ok], a.y[ok]], np.c_[g.x[o[ok].astype(int)], g.y[o[ok].astype(int)]]], 1)
        ax.add_collection(LineCollection(ls, colors="#2a78d6", linewidths=lw * 2, alpha=.7, zorder=3))
    for s_, (lab, col, fill) in LOOK.items():
        m = D.status == s_
        if s_ == 6:
            continue
        ax.scatter(D.x[m], D.y[m], s=ms, c=col if fill else "none", edgecolors=col, linewidths=.8 if not fill else 0,
                   zorder=5, label=f"{lab} ({int(m.sum())})")
    pend = R[R.status <= 1]
    ax.scatter(g.x[pend.origin], g.y[pend.origin], s=ms * 1.6, marker="x", c=INK, linewidths=.9, zorder=6,
               label=f"waiting rider, no driver yet ({len(pend)})")
    if box is not None:
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    else:
        ax.autoscale_view()
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    return lc, D, R


def busiest_box(D, half):
    xy = np.unique(D[D.status.isin([3, 4, 5])][["x", "y"]].round().to_numpy(), axis=0)   # moving drivers, one per spot
    best, bc = -1, None
    for c in xy[:: max(1, len(xy) // 300)]:
        n = (np.abs(xy - c) < half).all(1).sum()
        if n > best:
            best, bc = n, c
    return bc[0] - half, bc[0] + half, bc[1] - half, bc[1] + half


def plot_snapshot(key, L, g, path):
    e = int(8 * 3600 / S.DT)
    fig = plt.figure(figsize=(16, 8.6), constrained_layout=True)
    a0, a1 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
    lc, D, R = draw_world(a0, L, g, e, lw=.25, ms=5)
    half = 900 if key == "sf_hd" else 1400
    box = busiest_box(D, half)
    a0.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec=INK, lw=1.2, zorder=7))
    a0.set_title(f"{CITY_NAME[key]}, {L.date.date()} 08:00 — whole city: {len(D)} drivers online, "
                 f"{len(R)} open requests, {g.E:,} road edges", fontsize=10)
    draw_world(a1, L, g, e, box=box, lw=1.1, ms=34)
    a1.set_title(f"zoom: {2 * half / 1000:.1f} km x {2 * half / 1000:.1f} km (black box) — every driver on its road, "
                 "blue lines join a driver to the rider it is driving to", fontsize=10)
    a1.legend(loc="lower left", fontsize=8, framealpha=.92)
    cb = fig.colorbar(lc, ax=a1, shrink=.5, pad=.01)
    cb.set_label("road travel time / free-flow time (shown from 1.15)", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=125); plt.close(fig)


# ------------------------------------------------------------ one driver's day
def plot_driver_days(sets, path):
    fig, axs = plt.subplots(4, 1, figsize=(15, 9.5), constrained_layout=True)
    k = 0
    for key, (L, g) in sets.items():
        dd = L.status != 0
        busy = (np.isin(L.status, [3, 4])).sum(1)
        cands = np.argsort(-busy)[:40]
        for d in cands[[0, 25]]:
            ax = axs[k]; k += 1
            st = L.status[d]
            h = np.arange(len(st)) * S.DT / 3600
            for s_, (lab, col, fill) in LOOK.items():
                m = st == s_
                if m.any():
                    ax.fill_between(h, 0, 1, where=m, step="post", color=col, alpha=1 if fill else .35, lw=0, label=lab)
            ev = L.ev[L.ev.driver_id == d]
            for typ, y, mk, col in (("offer_sent", 1.35, "|", INK2), ("offer_accepted", 1.2, "v", "#2a78d6"),
                                    ("offer_rejected", 1.2, "x", INK), ("offer_timeout", 1.2, "1", INK),
                                    ("dispatch_sent", 1.6, "|", INK2), ("dispatch_accepted", 1.48, "v", "#1baf7a"),
                                    ("dispatch_declined", 1.48, "x", INK)):
                t = ev[ev.type == typ].t_event / 3600
                ax.scatter(t, np.full(len(t), y), marker=mk, s=22, color=col, lw=1, zorder=4)
            ax2 = ax.twinx()
            ax2.plot(h, L.income[d], color=INK, lw=1.2)
            ax2.set_ylabel(f"income today ({'USD' if key == 'sf_hd' else 'INR'})", fontsize=8); ax2.tick_params(labelsize=7)
            ax.set_yticks([0.5, 1.2, 1.35, 1.48, 1.6])
            ax.set_yticklabels(["status", "offer answer", "offers", "order answer", "repositioning orders"], fontsize=7.5)
            ax.set_ylim(0, 1.75); ax.set_xlim(0, 24); ax.tick_params(axis="x", labelsize=7.5)
            lat = L.latent.iloc[d]
            ax.set_title(f"{CITY_NAME[key]} driver {d} on {L.date.date()}: {int(L.count['trips'][d, -1])} trips, "
                         f"{int(L.count['offers'][d, -1])} offers ({int(L.count['offers_rejected'][d, -1])} rejected, "
                         f"{int(L.count['offers_timeout'][d, -1])} unanswered), {int(L.count['orders'][d, -1])} repositioning orders "
                         f"({int(L.count['orders_declined'][d, -1])} declined).  Hidden traits (never shown to the model): "
                         f"accept bias {lat.accept_bias:+.2f}, order compliance {lat.reposition_compliance:.2f}", fontsize=9)
            ax.spines[["top"]].set_visible(False)
        axs[k - 2].legend(ncol=7, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.08), frameon=False)
    axs[-1].set_xlabel("hour of day", fontsize=8)
    fig.suptitle("Two drivers' days per city, as the entity-level state records them (ticks: v accepted, x rejected / declined, "
                 "1 no answer; line: income)", fontsize=11)
    fig.savefig(path, dpi=115); plt.close(fig)


# ------------------------------------------------------- prediction targets
def target_stats(L, g):
    """Per-step targets over 06:00-23:00 of one day."""
    e0, e1 = int(6 * 3600 / S.DT), int(23 * 3600 / S.DT)
    a, b = L.status[:, e0:e1], L.status[:, e0 + 1:e1 + 1]
    on = (a != 0) | (b != 0)
    trans = np.zeros((8, 8), np.int64)
    np.add.at(trans, (a[on], b[on]), 1)
    o = L.offers
    ev = L.ev
    resp = o.response.value_counts().to_dict()
    d_acc = int((ev.type == "dispatch_accepted").sum()); d_dec = int((ev.type == "dispatch_declined").sum())
    t = ev.t_event.to_numpy()
    step = np.floor(t[(t >= e0 * S.DT) & (t < e1 * S.DT)] / S.DT).astype(int)
    per_step = np.bincount(step - e0, minlength=e1 - e0)
    new = ev[ev.type == "request_created"].t_event.to_numpy()
    new_h = np.bincount((new // 3600).astype(int), minlength=24)
    # movement: metres driven in 30 s by moving drivers
    xy0, xy1 = L.xy(int(8 * 3600 / S.DT)), L.xy(int(8 * 3600 / S.DT) + 1)
    mv = L.node_to[:, int(8 * 3600 / S.DT)] >= 0
    dist = np.hypot(*(xy1 - xy0)[mv].T)
    return {"trans": trans, "resp": resp, "orders": (d_acc, d_dec), "per_step": per_step, "new_h": new_h, "dist": dist}


def plot_targets(stats, path):
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.6), constrained_layout=True)
    lab = [LOOK.get(i, ("offline",))[0] for i in range(8)]
    for i, (key, st) in enumerate(stats.items()):
        tr = st["trans"].astype(float)
        p = tr / np.maximum(tr.sum(1, keepdims=True), 1)
        ax = axs[i, 0]
        im = ax.imshow(np.where(tr > 0, p, np.nan), cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
            "seq", ["#e3eefb", "#2a78d6", "#0d3b75"]), vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(p):
            if tr[r, c] > 0 and r != c:
                ax.text(c, r, f"{100 * v:.1f}" if v >= .001 else "·", ha="center", va="center", fontsize=6.3,
                        color="white" if v > .5 else INK)
        ax.set_xticks(range(8)); ax.set_yticks(range(8)); ax.set_xticklabels(lab, rotation=60, fontsize=7); ax.set_yticklabels(lab, fontsize=7)
        ax.set_xlabel("status 30 s later", fontsize=8); ax.set_ylabel("status now", fontsize=8)
        ax.set_title(f"{CITY_NAME[key]}: status 30 s later (% of changes\nfrom each status; dark diagonal = no change)", fontsize=8.5)
        ax = axs[i, 1]
        r = st["resp"]; da, dd = st["orders"]
        names = ["offer accepted", "offer rejected", "offer unanswered", "order followed", "order declined"]
        vals = [r.get("accepted", 0), r.get("rejected", 0), r.get("timeout", 0), da, dd]
        tot = [sum(vals[:3])] * 3 + [da + dd] * 2
        ax.barh(names[::-1], [100 * v / max(t, 1) for v, t in zip(vals, tot)][::-1], color="#2a78d6", height=.6)
        for y, (v, t) in enumerate(zip(vals[::-1], tot[::-1])):
            ax.text(100 * v / max(t, 1) + 1, y, f"{100 * v / max(t, 1):.0f}%  ({v:,})", va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(0, 115); ax.tick_params(labelsize=7.5); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("driver responses to the platform's actions\n(one day)", fontsize=8.5)
        ax = axs[i, 2]
        h = 6 + np.arange(len(st["per_step"])) * S.DT / 3600
        ax.plot(h, st["per_step"], color=INK2, lw=.6)
        ax.plot(np.arange(24) + .5, st["new_h"] / 120, color="#eb6834", lw=2, label="new requests per step (hourly mean)")
        ax.set_title("events per 30-s step (all types)\nand new requests", fontsize=8.5)
        ax.set_xlabel("hour of day", fontsize=8); ax.tick_params(labelsize=7.5); ax.legend(fontsize=7.5, frameon=False)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(color=GRID, lw=.5)
        ax = axs[i, 3]
        ax.hist(st["dist"], bins=40, color="#1baf7a")
        ax.set_title(f"metres driven in one step by moving drivers\nat 08:00 (median {np.median(st['dist']):.0f} m)", fontsize=8.5)
        ax.tick_params(labelsize=7.5); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("What the model must predict every 30 seconds, per driver and per request (one training day per city)", fontsize=11)
    fig.savefig(path, dpi=115); plt.close(fig)


def plot_hidden(key_days, path):
    """Hidden traits vs the history the state carries, over several days of one season."""
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    for i, (key, rows) in enumerate(key_days.items()):
        df = pd.DataFrame(rows)
        g = df.groupby("driver_id")[["offers", "accepted", "orders", "followed"]].sum()
        lat = df.groupby("driver_id")[["accept_bias", "reposition_compliance"]].first()
        g = g.join(lat)
        m = g.offers >= 20
        ax = axs[2 * i]
        ax.scatter(g.accept_bias[m], (g.accepted / g.offers)[m], s=8, color="#2a78d6", alpha=.6)
        r = np.corrcoef(g.accept_bias[m], (g.accepted / g.offers)[m])[0, 1]
        ax.set_xlabel("hidden accept bias", fontsize=8); ax.set_ylabel("share of offers accepted", fontsize=8)
        ax.set_title(f"{CITY_NAME[key]}: drivers with 20+ offers (r = {r:.2f})", fontsize=9)
        m = g.orders >= 5
        ax = axs[2 * i + 1]
        ax.scatter(g.reposition_compliance[m], (g.followed / g.orders)[m], s=8, color="#1baf7a", alpha=.6)
        r = np.corrcoef(g.reposition_compliance[m], (g.followed / g.orders)[m])[0, 1]
        ax.set_xlabel("hidden order compliance", fontsize=8); ax.set_ylabel("share of orders followed", fontsize=8)
        ax.set_title(f"{CITY_NAME[key]}: drivers with 5+ orders (r = {r:.2f})", fontsize=9)
        for a in axs[2 * i: 2 * i + 2]:
            a.tick_params(labelsize=7.5); a.grid(color=GRID, lw=.5); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Hidden traits (Z in the formulation) are never inputs, but each driver's history of answers, which the state "
                 "carries, reveals them — so the model can learn who will accept or follow an order", fontsize=10.5)
    fig.savefig(path, dpi=115); plt.close(fig)


def hidden_rows(L):
    ev = L.ev
    cnt = lambda typ: ev[ev.type == typ].groupby("driver_id").size()  # noqa: E731
    df = pd.DataFrame({"offers": cnt("offer_sent"), "accepted": cnt("offer_accepted"), "orders": cnt("dispatch_sent"),
                       "followed": cnt("dispatch_accepted")}).fillna(0)
    df = df.join(L.latent.set_index("driver_id")[["accept_bias", "reposition_compliance"]])
    return df.reset_index().rename(columns={"index": "driver_id"}).to_dict("records")


# ------------------------------------------------------------------ viewer
def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def viewer_data(key, L, g):
    e0, e1 = int(W0 / S.DT), int(W1 / S.DT)
    ne = e1 - e0 + 1
    unit = 2.0                                                        # metres per position unit
    x0, y0 = g.x.min(), g.y.min()
    q = lambda v: np.clip(np.round(v / unit), 0, 65534).astype(np.uint16)  # noqa: E731
    on = (L.status[:, e0:e1 + 1] != 0).any(1)
    ids = np.flatnonzero(on)
    n = len(ids)
    px = np.full((n, ne), 65535, np.uint16); py = px.copy()
    for k, e in enumerate(range(e0, e1 + 1)):
        xy = L.xy(e)[ids]
        m = L.status[ids, e] != 0
        px[m, k] = q(xy[m, 0] - x0); py[m, k] = q(xy[m, 1] - y0)
    stt = L.status[ids, e0:e1 + 1].astype(np.int8)
    leg = L.leg[ids, e0:e1 + 1]
    left = np.clip(np.round(L.left_m[ids, e0:e1 + 1] / 2), 0, 65535).astype(np.uint16)
    used = np.unique(leg[leg >= 0])
    leg_ix = np.full(len(L.legs) + 1, -1, np.int32); leg_ix[used] = np.arange(len(used))
    legw = np.where(leg >= 0, leg_ix[np.maximum(leg, 0)], -1).astype(np.int32)
    lg = L.legs.set_index("leg_id").loc[used]
    # per-driver day logs for the inspector (counters, status history), whole day up to the window end
    ev = L.ev[(L.ev.t_event <= W1 + S.DT) & L.ev.driver_id.isin(ids)]
    types = sorted(set(L.ev.type))
    pos = pd.Series(np.arange(n), index=ids)
    evd = pd.DataFrame({"t_event": ev.t_event.to_numpy(np.float32), "t_av": ev.t_available.to_numpy(np.float32),
                        "type": ev.type.map({t: i for i, t in enumerate(types)}).to_numpy(np.int8),
                        "drv": pos.loc[ev.driver_id].to_numpy(np.int32), "req": ev.request_id.to_numpy(np.int32),
                        "zone": ev.zone.to_numpy(np.int16), "value": np.nan_to_num(ev.value.to_numpy(np.float32), nan=-1)})
    stl = L.status_log[L.status_log.driver_id.isin(ids) & (L.status_log.t <= W1)]
    p = L.drivers.iloc[ids]
    # requests open at any time in the window
    end = np.fmin(L.r_pickup, L.r_cancel)
    rq = np.flatnonzero((L.r_created <= W1) & ~(end <= W0))
    oq = L.offers[L.offers.request_id.isin(rq)]
    # roads
    rr = [L.road_ratio(e) for e in range(e0, e1 + 1, 10)]                 # one per 5-min slot
    cong = []
    for r in rr:
        j = np.flatnonzero(r >= 1.15)
        cong.append({"e": b64(j.astype(np.uint32)), "r": b64(np.clip(np.round((r[j] - 1) * 50), 0, 255).astype(np.uint8))})
    cls = pd.Series(g.road_class).map({"motorway": 3, "trunk": 3, "primary": 2, "secondary": 1}).fillna(0).to_numpy(np.uint8)
    # self-check records computed by state.py, for the page to compare with its own
    rng = np.random.default_rng(0)
    checks = []
    for e in rng.integers(e0, e1, 12):
        D = L.drivers_at(int(e))
        for _, r in D.sample(min(4, len(D)), random_state=int(e)).iterrows():
            checks.append({"e": int(e - e0), "d": int(pos[int(r.driver_id)]),
                           **{k: (None if pd.isna(r[k]) else float(r[k])) for k in
                              ["status", "left_m", "in_status_s", "online_s", "income", *S.COUNTERS]}})
    return {
        "city": CITY_NAME[key], "date": str(L.date.date()), "w0": W0, "dt": S.DT, "ne": ne, "unit": unit,
        "currency": "USD" if key == "sf_hd" else "INR", "types": types, "legKinds": S.LEG_KINDS, "modes": S.MODES,
        "nodes": {"x": b64(q(g.x - x0)), "y": b64(q(g.y - y0)), "lon": b64(g.lon.astype(np.float32)), "lat": b64(g.lat.astype(np.float32)),
                  "zone": b64(g.zone.astype(np.int16))},
        "edges": {"u": b64(g.eu.astype(np.uint32)), "v": b64(g.ev.astype(np.uint32)), "cls": b64(cls)},
        "cong": cong,
        "drivers": {"ids": b64(ids.astype(np.int32)), "x": b64(px), "y": b64(py), "st": b64(stt), "leg": b64(legw), "left": b64(left),
                    "mode": b64(p["mode"].map({m: i for i, m in enumerate(S.MODES)}).to_numpy(np.int8)),
                    "cost": b64(p.cost_per_km.to_numpy(np.float32)), "tenure": b64(p.tenure_days.to_numpy(np.int32)),
                    "rsum": b64(p.rating_sum.to_numpy(np.float32)), "rn": b64(p.rating_count.to_numpy(np.int32)),
                    "home": b64(p.home_node.to_numpy(np.int32)), "shift": b64(p.usual_shift_start_h.to_numpy(np.float32))},
        "legs": {"kind": b64(lg.kind.map({k: i for i, k in enumerate(S.LEG_KINDS)}).to_numpy(np.int8)),
                 "dest": b64(np.array([nn[-1] for nn in lg.nodes], np.int32)), "req": b64(lg.request_id.to_numpy(np.int32)),
                 "len": b64(lg.length_m.to_numpy(np.float32))},
        "events": {k: b64(v.to_numpy()) for k, v in evd.items()},
        "status": {"drv": b64(pos.loc[stl.driver_id].to_numpy(np.int32)), "t": b64(stl.t.to_numpy(np.float32)),
                   "s": b64(stl.status.to_numpy(np.int8))},
        "requests": {"id": b64(rq.astype(np.int32)), "origin": b64(L.r_origin[rq].astype(np.int32)),
                     "dest": b64(L.r_dest[rq].astype(np.int32)), "ext": b64(L.r_external[rq].astype(np.int16)),
                     "mode": b64(L.r_mode[rq].astype(np.int8)), "quote": b64(L.r_quote[rq].astype(np.float32)),
                     **{k: b64(np.nan_to_num(getattr(L, f"r_{k}")[rq], nan=1e9).astype(np.float32))
                        for k in ("created", "accept", "arrive", "pickup", "cancel")}},
        "offers": {"req": b64(oq.request_id.to_numpy(np.int32)), "drv": b64(oq.driver_id.to_numpy(np.int32)),
                   "t": b64(oq.t_sent.to_numpy(np.float32)), "tr": b64(np.nan_to_num(oq.t_response.to_numpy(np.float64), nan=1e9).astype(np.float32)),
                   "resp": b64(oq.response.map({"accepted": 0, "rejected": 1, "timeout": 2}).fillna(3).to_numpy(np.int8))},
        "checks": checks,
        "stats": {"n_drivers": int(n), "n_edges": int(g.E), "n_nodes": int(g.N), "n_requests": int(len(rq)), "n_events": int(len(ev))},
    }


def write_viewer(key, L, g, path):
    data = viewer_data(key, L, g)
    tpl = (Path(__file__).resolve().parents[1] / "world" / "wm3" / "viewer.html").read_text()
    html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    path.write_text(html)
    return path.stat().st_size / 1e6


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res, vs, sets, stats, hid = {}, {}, {}, {}, {}
    for key in ["sf_hd", "pune_hd"]:
        city = cities.get(key)
        g = S.CityGraph(city)
        d = pick_day(key)
        L = S.DayLog(g, d)
        print(f"  {key}: {d.parent.name}/{d.name}: {L.nD} drivers, {len(L.req)} requests, {len(L.ev)} events, "
              f"{g.N} nodes, {g.E} edges", flush=True)
        v = verify(L, g)
        vs[key] = v
        res[key] = {"day": f"{d.parent.name}/{d.name}",
                    **{f"{k}_zone_match": float(1 - v[f"{k}_cell_err"].sum() / max(v[f"{k}_sim"].sum(), 1))
                       for k in ("idle", "busy", "pending")},
                    **{f"{k}_total_abs_diff": int(np.abs(v[f"{k}_rebuilt"] - v[f"{k}_sim"]).sum()) for k in ("idle", "busy", "pending")}}
        print("   ", {k: round(x, 5) if isinstance(x, float) else x for k, x in res[key].items()}, flush=True)
        sets[key] = (L, g)
        stats[key] = target_stats(L, g)
        plot_snapshot(key, L, g, OUT / f"16_snapshot_{key}.png")
        mb = write_viewer(key, L, g, OUT / f"{key}_world.html")
        print(f"    viewer {mb:.1f} MB", flush=True)
        # several days of the same season for the hidden-trait plot
        season = [x for x in day_dirs(key) if x.parent == d.parent][:8]
        rows = []
        for x in season:
            rows += hidden_rows(L if x == d else S.DayLog(g, x))
        hid[key] = rows
        st = stats[key]
        res[key].update({"offer_responses": st["resp"], "orders_followed_declined": list(st["orders"]),
                         "median_step_m_moving": float(np.median(st["dist"])), "events_per_step_mean": float(st["per_step"].mean())})
    plot_verify(vs, OUT / "16_verify.png")
    plot_driver_days(sets, OUT / "16_driver_day.png")
    plot_targets(stats, OUT / "16_targets.png")
    plot_hidden(hid, OUT / "16_hidden.png")
    (OUT / "16_stage1.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
