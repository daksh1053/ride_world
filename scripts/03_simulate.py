"""Stage 3: simulate one day of the ride company on the stage-1 graph and stage-2 traffic.

    .venv/bin/python scripts/03_simulate.py --city sf pune
    .venv/bin/python scripts/03_simulate.py --city pune --date 2025-07-08 --seed 3 --replay 8 11

Outputs
    data/<city>/sim/<run>/   drivers(+_latent), customers(+_latent), requests, trips, offers,
                             actions, events, ledger, ratings, context, legs, status (parquet),
                             metrics.json, config.json
    outputs/<city>/          03_kpis.png, 03_drivers.png, 03_zones.png, 03_snapshot.png,
                             03_replay.html (animated cabs for the --replay window)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from world import cities, metrics, replay, roadnet, viz
from world.policies import POLICIES
from world.setup import CityData, build_world
from world.sim import SimConfig

STATUS_COL = {"idle": "#52514e", "to pickup": "#2a78d6", "with passenger": "#eb6834",
              "repositioning": "#1baf7a"}
RED = "#d03b3b"


def style_axes(ax):
    ax.grid(axis="y", color=viz.HAIRLINE, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_kpis(city, tb, out):
    r = tb["requests"]
    r = r[r.status != "not_raised"]
    fig, axs = plt.subplots(1, 3, figsize=(19, 4.8), layout="constrained")
    x = (np.arange(96) + 0.5) / 4
    def b(s):  # counts per 15 min within the day; drain-hour events after 24:00 excluded
        s = s.dropna()
        return np.bincount((s[s < 86400] // 900).astype(int), minlength=96)[:96]
    ax = axs[0]
    ax.plot(x, b(r.t_request), color=viz.INK, lw=2, label="requests")
    ax.plot(x, b(r.t_dropoff), color=STATUS_COL["with passenger"], lw=2, label="completed")
    ax.plot(x, b(r.t_cancel), color=RED, lw=2, label="cancelled / expired")
    ax.set_title("Requests per 15 min")
    ax.legend(frameon=False, fontsize=9)

    st = tb["status"].sort_values("t")
    grid = np.arange(0, 86401, 300)
    counts = np.zeros((len(grid), 7))
    for _, g in st.groupby("driver_id"):
        idx = np.searchsorted(g.t.to_numpy(), grid, side="right") - 1
        s = np.where(idx >= 0, g.status.to_numpy()[np.maximum(idx, 0)], 0)
        np.add.at(counts, (np.arange(len(grid)), s), 1)
    ax = axs[1]
    layers = [("with passenger", counts[:, 4] + counts[:, 6]), ("to pickup", counts[:, 3]),
              ("repositioning", counts[:, 5]), ("idle", counts[:, 1] + counts[:, 2])]
    ax.stackplot(grid / 3600, *[v for _, v in layers], colors=[STATUS_COL[k] for k, _ in layers],
                 labels=[("with passenger (incl. intercity)" if k == "with passenger" else k) for k, _ in layers],
                 edgecolor=viz.SURFACE, lw=0.3)
    ax.set_title("Online drivers by status")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axs[2]
    served = r[r.t_pickup.notna()]
    h = (served.t_request // 3600).astype(int)
    wait = served.wait_s / 60
    med = wait.groupby(h).median().reindex(range(24))
    p90 = wait.groupby(h).quantile(0.9).reindex(range(24))
    ax.fill_between(np.arange(24) + 0.5, 0, p90, color="#b7d3f6", lw=0, label="90th percentile")
    ax.plot(np.arange(24) + 0.5, med, color="#2a78d6", lw=2, label="median")
    ax.set_title("Pickup wait of served requests (min)")
    ax.legend(frameon=False, fontsize=9)
    for a in axs:
        a.set_xlim(0, 24), a.set_xticks(range(0, 25, 3)), a.set_xlabel("hour of day")
        style_axes(a)
        a.set_ylim(bottom=0)
    fig.suptitle(f"{city.name}: one simulated day of the ride company", x=0.01, ha="left", fontweight="bold")
    viz.save(fig, out / "03_kpis.png")


def plot_drivers(city, dout, fair, out):
    cur = city.currency
    fig, axs = plt.subplots(1, 3, figsize=(19, 4.8), layout="constrained")
    ax = axs[0]
    ax.hist(dout.hourly, bins=40, color="#2a78d6", edgecolor=viz.SURFACE, lw=0.8)
    ax.axvline(dout.hourly.mean(), color=viz.INK, lw=1.2, ls="--")
    ax.text(dout.hourly.mean(), ax.get_ylim()[1] * 0.95, f" mean {dout.hourly.mean():.1f}", va="top", fontsize=9)
    ax.set_title(f"Net earnings per online hour ({cur})")
    ax.set_xlabel(f"{cur} / h"), ax.set_ylabel("drivers")
    ax = axs[1]
    s = np.sort(dout.net_earnings.clip(lower=0).to_numpy())
    lor = np.r_[0, np.cumsum(s) / s.sum()]
    xs = np.linspace(0, 1, len(lor))
    ax.plot(xs, xs, color=viz.MUTED, lw=1, ls="--", label="equal earnings")
    ax.plot(xs, lor, color="#eb6834", lw=2, label=f"drivers (Gini {fair['gini_net_earnings']:.3f})")
    ax.set_title("Lorenz curve of daily net earnings")
    ax.set_xlabel("share of drivers (poorest first)"), ax.set_ylabel("share of total earnings")
    ax.legend(frameon=False, fontsize=9)
    ax = axs[2]
    ax.scatter(dout.online_h, dout.net_earnings, s=8, color="#2a78d6", alpha=0.5, lw=0)
    ax.set_title(f"Online hours vs daily net earnings ({cur})")
    ax.set_xlabel("online hours"), ax.set_ylabel(cur)
    for a in axs:
        style_axes(a)
    fig.suptitle(f"{city.name}: driver outcomes (cohort {fair['cohort_size']} drivers; "
                 f"F_var {fair['F_var']:,.0f}, F_min {fair['F_min']:,.1f} {cur})",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, out / "03_drivers.png")


def plot_zones(city, zs, zone_geoms, nodes, out):
    g = zone_geoms.merge(zs, on="zone", how="left")
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.5), layout="constrained")
    for ax, col, lab, cmap in [(axs[0], "completion_rate", "completion rate s_z (eq. 25)", viz.SEQ),
                               (axs[1], "mean_wait_min", "mean pickup wait w_z of served requests (min)", viz.SEQ)]:
        g.plot(ax=ax, column=col, cmap=cmap, edgecolor=viz.SURFACE, lw=0.6, legend=True,
               legend_kwds={"shrink": 0.6}, missing_kwds={"color": "#eeeeea"})
        viz.map_axes(ax, nodes)
        ax.set_title(lab)
    fig.suptitle(f"{city.name}: customer service by origin zone", x=0.01, ha="left", fontweight="bold")
    viz.save(fig, out / "03_zones.png")


def plot_snapshot(city, tb, nodes, edges, t_h, out):
    t = t_h * 3600
    st = tb["status"].sort_values("t")
    last = st[st.t <= t].groupby("driver_id").tail(1)
    legs = tb["legs"]
    active = legs[(legs.t_start <= t) & (legs.t_end > t)]
    pos = {}
    for lg in active.itertuples():
        j = int(np.searchsorted(lg.times, t, side="right")) - 1
        pos[lg.driver_id] = int(lg.nodes[max(j, 0)])
    fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
    viz.draw_roads(ax, nodes, edges)
    name = {1: "idle", 2: "idle", 3: "to pickup", 4: "with passenger", 5: "repositioning"}
    for s, lab in name.items():
        dd = last[last.status == s]
        if not len(dd):
            continue
        nd = [pos.get(d, n) for d, n in zip(dd.driver_id, dd.node)]
        ax.scatter(nodes.lon.to_numpy()[nd], nodes.lat.to_numpy()[nd], s=14, color=STATUS_COL[lab],
                   edgecolor=viz.SURFACE, lw=0.5, zorder=20)
    r = tb["requests"]
    wait = r[(r.t_request <= t) & (r.t_pickup.fillna(r.t_cancel).fillna(1e9) > t)]
    ax.scatter(nodes.lon.to_numpy()[wait.origin_node], nodes.lat.to_numpy()[wait.origin_node], s=30,
               facecolor="none", edgecolor=viz.INK, lw=1, zorder=19)
    viz.map_axes(ax, nodes)
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=k) for k, c in STATUS_COL.items()]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", color=viz.INK, markersize=8,
                          label=f"waiting request ({len(wait)})"))
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    ax.set_title(f"{city.name}: cabs at {int(t_h):02d}:{int(t_h % 1 * 60):02d}")
    viz.save(fig, out / "03_snapshot.png")


def run(key, args):
    city = cities.get(key)
    print(f"== {city.name}")
    cfg = SimConfig(date=args.date, seed=args.seed)
    t0 = time.time()
    data = CityData(city)
    world = build_world(data, cfg, POLICIES[args.policy]())
    nodes, edges, zones = data.nodes, data.edges, data.zones
    world.run()
    tb = world.tables()
    run_dir = city.data_dir / "sim" / f"{args.date}_seed{args.seed}_{args.policy}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, df in {**tb, "drivers": world.drv, "drivers_latent": world.dlat,
                     "customers": world.cust, "customers_latent": world.clat}.items():
        df.to_parquet(run_dir / f"{name}.parquet", index=False)

    dout = metrics.driver_outcomes(tb)
    fair = metrics.fairness(dout)
    serv = metrics.service_summary(tb)
    zs = metrics.zone_service(tb, len(zones))
    dout.to_parquet(run_dir / "driver_outcomes.parquet", index=False)
    zs.to_parquet(run_dir / "zone_service.parquet", index=False)
    summary = {**serv, **fair, "day": world.day.summary()}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    (run_dir / "config.json").write_text(json.dumps({**cfg.__dict__, "policy": args.policy,
                                                      "city": city.key}, indent=2))
    print(f"  saved {len(tb['events']):,} events, {len(tb['ledger']):,} ledger entries to "
          f"{run_dir.relative_to(run_dir.parents[3])}")
    print("  " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()
                             if k not in ("cancel_reasons", "day")}))

    out = city.out_dir
    zone_geoms = gpd.read_file(city.data_dir / "zone_geoms.geojson")
    plot_kpis(city, tb, out)
    plot_drivers(city, dout, fair, out)
    plot_zones(city, zs, zone_geoms, nodes, out)
    plot_snapshot(city, tb, nodes, edges, args.replay[0] + 0.5, out)
    replay.build_replay(city, nodes, zone_geoms, tb, world.drv, args.replay[0], args.replay[1],
                        out / "03_replay.html", summary, roadnet.edge_shapes(city, nodes))
    print(f"  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", nargs="+", default=["sf", "pune"], choices=sorted(cities.CITIES))
    ap.add_argument("--date", default="2025-03-04", help="simulated date (sets weekday/weekend, season)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="nearest", choices=sorted(POLICIES))
    ap.add_argument("--replay", nargs=2, type=float, default=[17.0, 20.0], metavar=("FROM_H", "TO_H"),
                    help="hours of the day to include in the animated replay")
    args = ap.parse_args()
    for k in args.city:
        run(k, args)
