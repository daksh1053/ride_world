"""Stage 2: time-varying traffic on the stage-1 road graph.

    .venv/bin/python scripts/02_traffic.py --city sf pune           # calibrate + render
    .venv/bin/python scripts/02_traffic.py --city sf --render-only  # reuse saved arrays

Outputs
    data/<city>/traffic/   bg_volume.npy (2,24,E) veh/h, typical_times.npy (2,24,E) s,
                           zone_tt_s.npy / zone_dist_m.npy (2,24,Z,Z), free-flow versions,
                           edge_params.parquet, zone_weights.parquet, calibration.csv, meta.json
    outputs/<city>/        02_tti_profile.png, 02_congestion.png, 02_zone_tt.png,
                           02_sample_week.png, 02_volume.png, 02_explorer.html
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from world import cities, explorer, roadnet, traffic as T, viz

# Delay-factor bins and an orange sequential ramp (the second sequential context
# after blue, per the palette rules), with free-flow edges receding to grey.
DELAY_BINS = [1.0, 1.15, 1.4, 1.8, 2.5, 4.0, 99]
DELAY_COLS = ["#d6d5ce", "#fbc4a4", "#f39063", "#eb6834", "#b8431a", "#6e2308"]
DELAY_LABELS = ["< 1.15 (free)", "1.15–1.4", "1.4–1.8", "1.8–2.5", "2.5–4", "≥ 4"]
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style_axes(ax):
    ax.grid(axis="y", color=viz.HAIRLINE, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_profile(city, calib, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    for dt, col in (("weekday", ORANGE), ("weekend", BLUE)):
        g = calib[calib.daytype == dt].sort_values("hour")
        a1.plot(g.hour + 0.5, g.tti, color=col, lw=2, label=f"{dt}: model")
        a1.scatter(g.hour + 0.5, g.target, s=14, color=col, edgecolor=viz.SURFACE, lw=0.8, zorder=3,
                   label=f"{dt}: target")
        a2.plot(g.hour + 0.5, g.total_vph / 1000, color=col, lw=2, label=dt)
    a1.set_title("Network travel-time index by hour")
    a1.set_ylabel("TTI  (congested / free-flow time)")
    a1.legend(frameon=False, fontsize=9, ncols=2)
    a2.set_title("Calibrated background demand")
    a2.set_ylabel("vehicle trips per hour (thousands)")
    a2.legend(frameon=False, fontsize=9)
    for a in (a1, a2):
        a.set_xlim(0, 24), a.set_xticks(range(0, 25, 3)), a.set_xlabel("hour of day")
        style_axes(a)
    a1.set_ylim(bottom=1)
    a2.set_ylim(bottom=0)
    err = (calib.tti - calib.target).abs().max()
    fig.suptitle(f"{city.name}: traffic calibration (max |TTI error| {err:.3f}). "
                 f"Targets: {city.traffic.source}", x=0.01, ha="left", fontsize=10, color=viz.INK_2)
    viz.save(fig, out / "02_tti_profile.png")


def plot_congestion(city, nodes, edges, times, gateways, out):
    t0 = edges.free_flow_s.to_numpy()
    tgt = city.traffic
    hours = [(0, 4, "04:00 weekday"), (0, int(tgt.am_peak_hour), f"{int(tgt.am_peak_hour):02d}:30 weekday (AM peak)"),
             (0, int(tgt.pm_peak_hour), f"{int(tgt.pm_peak_hour):02d}:30 weekday (PM peak)"),
             (1, 13, "13:30 weekend")]
    cmap = ListedColormap(DELAY_COLS)
    norm = BoundaryNorm(DELAY_BINS, cmap.N)
    segs = viz.segments(nodes, edges)
    minor = edges.road_class.isin(["residential", "unclassified", "living_street", "service", "other"]).to_numpy()
    fig, axs = plt.subplots(1, 4, figsize=(24, 7), layout="constrained")
    for ax, (d, h, lab) in zip(axs, hours):
        r = times[d, h] / t0
        free = r < DELAY_BINS[1]
        keep = ~(free & minor)
        order = np.argsort(r[keep])
        lw = np.where(free[keep], 0.3, 1.1)[order]
        ax.add_collection(LineCollection(segs[keep][order], array=r[keep][order], cmap=cmap, norm=norm,
                                         linewidths=lw, capstyle="round"))
        ax.scatter(gateways.lon, gateways.lat, s=40 + 200 * gateways.weight, marker="^", color=BLUE,
                   edgecolor=viz.SURFACE, lw=0.8, zorder=5)
        viz.map_axes(ax, nodes)
        ax.set_title(lab)
    handles = [Line2D([], [], color=c, lw=3, label=l) for c, l in zip(DELAY_COLS, DELAY_LABELS)]
    handles.append(Line2D([], [], marker="^", ls="", color=BLUE, markersize=8,
                          label=f"external gateway ({len(gateways)}; size = capacity share)"))
    fig.legend(handles=handles, loc="lower center", ncols=7, frameon=False, title="delay factor  t / t_free-flow")
    fig.suptitle(f"{city.name}: typical congestion by time of day (free-flowing residential streets hidden)",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, out / "02_congestion.png")


VOL_BINS = [0, 20, 100, 300, 800, 1600, 1e9]
VOL_COLS = ["#e1e0d9", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
VOL_WIDTHS = [0.15, 0.4, 0.7, 1.1, 1.6, 2.2]
VOL_LABELS = ["< 20 (≈ unused)", "20–100", "100–300", "300–800", "800–1,600", "≥ 1,600"]


def plot_volume(city, nodes, edges, volume, gateways, out):
    """Where the background traffic actually goes: every edge, width and colour by veh/h."""
    segs = viz.segments(nodes, edges)
    hours = [(0, int(city.traffic.am_peak_hour), f"{int(city.traffic.am_peak_hour):02d}:30 weekday (AM peak)"),
             (1, 13, "13:30 weekend")]
    fig, axs = plt.subplots(1, 2, figsize=(18, 9), layout="constrained")
    for ax, (d, h, lab) in zip(axs, hours):
        v = volume[d, h]
        b = np.digitize(v, VOL_BINS[1:-1])
        for k in range(len(VOL_COLS)):
            m = b == k
            ax.add_collection(LineCollection(segs[m], colors=VOL_COLS[k], linewidths=VOL_WIDTHS[k],
                                             capstyle="round", zorder=k))
        ax.scatter(gateways.lon, gateways.lat, s=40 + 200 * gateways.weight, marker="^", color=ORANGE,
                   edgecolor=viz.SURFACE, lw=0.8, zorder=10)
        viz.map_axes(ax, nodes)
        km = edges.length_m.groupby(b).sum() / edges.length_m.sum()
        ax.set_title(f"{lab}: {1 - km.get(0, 0):.0%} of road length carries ≥ 20 veh/h")
    handles = [Line2D([], [], color=c, lw=max(w * 2.5, 1.5), label=l)
               for c, w, l in zip(VOL_COLS, VOL_WIDTHS, VOL_LABELS)]
    handles.append(Line2D([], [], marker="^", ls="", color=ORANGE, markersize=8, label="external gateway"))
    fig.legend(handles=handles, loc="lower center", ncols=7, frameon=False,
               title="background traffic, vehicles / hour")
    fig.suptitle(f"{city.name}: assigned background traffic volume (all roads)", x=0.01, ha="left",
                 fontweight="bold")
    viz.save(fig, out / "02_volume.png")


def plot_zone_tt(city, zones, zone_geoms, nodes, tt, ff_tt, out):
    z0 = int(zones.sort_values("n_pois", ascending=False).zone.iloc[0])  # busiest activity zone
    tgt = city.traffic
    panels = [(ff_tt[z0] / 60, "free-flow"),
              (tt[0, 4, z0] / 60, "04:30 weekday"),
              (tt[0, int(tgt.pm_peak_hour), z0] / 60, f"{int(tgt.pm_peak_hour):02d}:30 weekday (PM peak)")]
    vmax = float(np.percentile(panels[-1][0], 98))
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), layout="constrained")
    for ax, (vals, lab) in zip(axs, panels):
        g = zone_geoms.assign(minutes=vals[zone_geoms.zone.to_numpy()])
        g.plot(ax=ax, column="minutes", cmap=viz.SEQ, vmin=0, vmax=vmax, edgecolor=viz.SURFACE, lw=0.6)
        g[g.zone == z0].boundary.plot(ax=ax, color=viz.INK, lw=2)
        viz.map_axes(ax, nodes)
        ax.set_title(f"{lab}: median {np.median(vals):.1f} min, max {vals.max():.1f} min")
    sm = plt.cm.ScalarMappable(cmap=viz.SEQ, norm=plt.Normalize(0, vmax))
    viz.colorbar(fig, sm, axs, "minutes from the outlined zone")
    fig.suptitle(f"{city.name}: fastest travel time from zone {z0} (most POIs) to every zone centre",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, out / "02_zone_tt.png")


def plot_sample_week(city, model, out):
    """One simulated week: typical vs realised network TTI every 15 minutes."""
    rng = np.random.default_rng(7)
    month = int(np.argmax(city.traffic.rain_prob_by_month)) + 1   # the wettest month: shows rain
    start = pd.Timestamp(2025, month, 1)
    start += pd.Timedelta(days=(7 - start.weekday()) % 7)          # a Monday
    fig, ax = plt.subplots(figsize=(15, 4.6), layout="constrained")
    rows = []
    for k in range(7):
        day = model.sample_day(start + pd.Timedelta(days=k), rng)
        typical = T.DayConditions(day.date, day.daytype, 1.0, None, day.incidents.iloc[:0])
        ts = np.arange(0, 24 * 3600, 900)
        real = [model.network_tti(day, t) for t in ts]
        typ = [model.network_tti(typical, t) for t in ts]
        x = k + ts / 86400
        ax.plot(x, typ, color=viz.MUTED, lw=1.2, ls="--", label="typical" if k == 0 else None)
        ax.plot(x, real, color=ORANGE, lw=2, label="realised" if k == 0 else None)
        if day.rain:
            a, b, inten = day.rain
            ax.axvspan(k + a / 86400, k + min(b, 86400) / 86400, color=BLUE, alpha=0.06 + 0.12 * inten, lw=0)
            ax.text(k + (a + min(b, 86400)) / 2 / 86400, 1.0, f"rain {inten:.1f}", ha="center",
                    va="bottom", color=BLUE, fontsize=8)
        ax.scatter(k + day.incidents.start_s / 86400, np.full(len(day.incidents), 1.0), marker="|",
                   s=60, color=viz.INK_2, label="incident start" if k == 0 else None)
        rows.append(day.summary())
    ax.set_xticks(np.arange(7) + 0.5, [f"{(start + pd.Timedelta(days=k)):%a %d %b}" for k in range(7)])
    ax.set_xlim(0, 7), ax.set_ylim(bottom=0.95)
    for k in range(1, 7):
        ax.axvline(k, color=viz.HAIRLINE, lw=0.8)
    ax.set_ylabel("network TTI")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", ncols=3, fontsize=9)
    ax.set_title(f"{city.name}: one simulated week of realised traffic "
                 f"(day-to-day demand ±{T.DEMAND_SIGMA:.0%}, rain spells shaded by intensity, incidents)")
    viz.save(fig, out / "02_sample_week.png")
    return pd.DataFrame(rows)


def run(key: str, render_only: bool):
    city = cities.get(key)
    print(f"== {city.name}")
    d = city.data_dir
    nodes = pd.read_parquet(d / "nodes.parquet")
    edges = pd.read_parquet(d / "edges.parquet")
    zones = pd.read_parquet(d / "zones.parquet")
    zone_geoms = gpd.read_file(d / "zone_geoms.geojson")
    tdir = d / "traffic"

    if not render_only:
        t = time.time()
        res = T.build_typical(city, nodes, edges, zones, roadnet.boundary(city))
        T.save_typical(city, edges, zones, res)
        print(f"  calibration took {time.time() - t:.0f}s")

    times = np.load(tdir / "typical_times.npy")
    calib = pd.read_csv(tdir / "calibration.csv")
    tt = np.load(tdir / "zone_tt_s.npy")
    ff_tt = np.load(tdir / "zone_tt_freeflow_s.npy")
    out = city.out_dir
    plot_profile(city, calib, out)
    gateways = pd.read_parquet(tdir / "gateways.parquet")
    plot_congestion(city, nodes, edges, times, gateways, out)
    plot_zone_tt(city, zones, zone_geoms, nodes, tt, ff_tt, out)
    week = plot_sample_week(city, T.TrafficModel(city), out)
    print(week.to_string(index=False))
    G = ox.load_graphml(d / "graph.graphml")
    volume = np.load(tdir / "bg_volume.npy")
    plot_volume(city, nodes, edges, volume, gateways, out)
    explorer.build_traffic(city, G, nodes, edges, zones, zone_geoms, times, volume, calib, tt, gateways,
                           out / "02_explorer.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", nargs="+", default=["sf", "pune"], choices=sorted(cities.CITIES))
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()
    for k in args.city:
        run(k, args.render_only)
