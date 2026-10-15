"""Stage 9b: validate a high-detail season against real data and draw its layers.

    .venv/bin/python scripts/09b_hd_validate.py --city sf_hd
    .venv/bin/python scripts/09b_hd_validate.py --city pune_hd

Reads the latest season in data/<city>/hd_runs/. Writes outputs/<city>/:
    09_layers.png       real inputs: residents (GHSL), work and nightlife weights, plus where
                        simulated pickups happen and where surge appears
    09_validation.png   SF: simulated vs SFCTA pickups by zone, drop-offs by zone, by hour and by weekday.
                        Pune: trips by purpose and hour, mode shares, trip durations
    09_validation.json  the numbers behind the figure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from world import cities, viz
from world.hd import demand as DM


def load_season(city):
    runs = sorted((city.data_dir / "hd_runs").glob("*/season.csv"), key=lambda p: p.stat().st_mtime)
    d = runs[-1].parent
    days = sorted(p for p in d.iterdir() if p.is_dir())
    req = pd.concat([pd.read_parquet(x / "requests.parquet").assign(date=x.name) for x in days], ignore_index=True)
    ctx = pd.concat([pd.read_parquet(x / "context.parquet").assign(date=x.name) for x in days], ignore_index=True)
    return d, req[req.status != "not_raised"], ctx, pd.read_csv(d / "season.csv")


def style(ax):
    ax.grid(color=viz.HAIRLINE, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def layers(city, geo, req, ctx, out):
    zg = gpd.read_file(city.data_dir / "zone_geoms.geojson")
    W = geo.W
    z = geo.node_zone
    per_zone = lambda col: np.bincount(z, weights=W[col].to_numpy(), minlength=geo.Z)  # noqa: E731
    pick = np.bincount(req.origin_zone[req.status == "completed"], minlength=geo.Z) / req.date.nunique()
    surge = ctx.groupby("zone").surge.mean().reindex(range(geo.Z)).fillna(1).to_numpy()
    area = zg.set_index("zone").to_crs(city.utm_crs).area.reindex(range(geo.Z)).to_numpy() / 1e6
    panels = [(per_zone("population") / area, "residents per km² (GHSL 2025)"), (per_zone("work"), "work weight (offices, commercial, industrial)"),
              (per_zone("food_night"), "nightlife and food weight"), (pick / area, "simulated pickups per km² per day"),
              (surge, "mean surge multiplier")]
    fig, axs = plt.subplots(1, 5, figsize=(27, 6), layout="constrained")
    for ax, (v, t) in zip(axs, panels):
        zg.assign(v=v[zg.zone.to_numpy()]).plot(ax=ax, column="v", cmap=viz.SEQ, edgecolor=viz.SURFACE, lw=0.3,
                                                legend=True, legend_kwds={"shrink": 0.6})
        viz.map_axes(ax, geo.nodes, pad=0.003)
        ax.set_title(t, fontsize=10)
    fig.suptitle(f"{city.name}: real input layers (left three) and what the simulated company does with them",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, out / "09_layers.png")


def validate_sf(city, geo, req, season, out):
    sf = DM.SFDemand(geo)
    ndays = req.date.nunique()
    done = req[req.status == "completed"]
    res = {}
    fig, axs = plt.subplots(1, 4, figsize=(24, 5.5), layout="constrained")
    # by TAZ: simulated requests vs SFCTA pickups (both per average day), and drop-offs
    taz_o = sf.node_taz[req.origin_node.to_numpy()]
    internal = req[req.dest_node >= 0]
    taz_d = sf.node_taz[internal.dest_node.to_numpy()]
    T = len(sf.taz_ids)
    sim_o = np.bincount(taz_o, minlength=T) / ndays
    sim_d = np.bincount(taz_d, minlength=T) / ndays
    dows = pd.to_datetime(req.date.unique()).weekday
    real_o = sf.P[dows].sum((0, 1)) / len(dows)
    real_d = sf.D[dows].sum((0, 1)) / len(dows)
    for ax, s, r, name in [(axs[0], sim_o, real_o, "pickups"), (axs[1], sim_d, real_d, "drop-offs")]:
        k = (r > 0) & (s > 0)
        ax.scatter(r[k], s[k], s=8, alpha=0.5, color="#2a78d6", lw=0)
        lim = [min(r[k].min(), s[k].min()) * 0.8, max(r[k].max(), s[k].max()) * 1.2]
        share = s.sum() / r.sum()
        ax.plot(lim, [lim[0] * share, lim[1] * share], color=viz.INK, lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        c = float(np.corrcoef(np.log1p(r), np.log1p(s))[0, 1])
        res[f"taz_{name}_log_corr"] = c
        ax.set_title(f"{name} per zone (981 SFCTA zones): log r = {c:.2f}", fontsize=10)
        ax.set_xlabel("SFCTA, all Uber + Lyft per day"); ax.set_ylabel("simulated company per day")
        style(ax)
    res["company_share_of_market"] = float(sim_o.sum() / real_o.sum())
    # hour of day, weekdays
    wk = req[pd.to_datetime(req.date).dt.weekday < 5]
    h_sim = np.bincount((wk.t_request // 3600).astype(int).clip(0, 23), minlength=24)
    h_real = sf.P[:5].sum((0, 2))
    axs[2].plot(range(24), h_real / h_real.sum(), color=viz.INK, lw=2, label="SFCTA weekdays")
    axs[2].plot(range(24), h_sim / h_sim.sum(), color="#eb6834", lw=2, ls="--", label="simulated requests")
    res["hour_profile_corr"] = float(np.corrcoef(h_real, h_sim)[0, 1])
    axs[2].set_title(f"share of the day's trips by hour (r = {res['hour_profile_corr']:.3f})", fontsize=10)
    axs[2].legend(frameon=False, fontsize=9); style(axs[2])
    # weekday totals
    real_dow = sf.P.sum((1, 2))
    sim = season.assign(dow=pd.to_datetime(season.date).dt.weekday).groupby("dow").requests.mean()
    axs[3].bar(np.arange(7) - 0.2, real_dow / real_dow.mean(), width=0.4, color="#c3c2b7", label="SFCTA")
    axs[3].bar(sim.index + 0.2, sim / sim.mean(), width=0.4, color="#2a78d6", label="simulated")
    axs[3].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axs[3].set_title("requests by weekday, relative to the week's mean", fontsize=10)
    axs[3].legend(frameon=False, fontsize=9); style(axs[3])
    fig.suptitle("San Francisco: simulated demand against SFCTA TNCs Today (Uber + Lyft, Nov-Dec 2016)",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, out / "09_validation.png")
    res["trip_km_median"] = float(done.distance_km.median())
    res["trip_min_median"] = float(((done.t_dropoff - done.t_pickup) / 60).median())
    return res


def validate_pune(city, geo, req, season, out):
    done = req[req.status == "completed"]
    res = {}
    fig, axs = plt.subplots(1, 3, figsize=(21, 5.5), layout="constrained")
    wk = req[pd.to_datetime(req.date).dt.weekday < 5]
    top = wk.purpose.value_counts().index[:8]
    hrs = np.arange(24)
    bottom = np.zeros(24)
    cols = viz.BLUE_RAMP[1:] + ["#eda100", "#eb6834"]
    for c, p in zip(cols, top):
        v = np.bincount((wk[wk.purpose == p].t_request // 3600).astype(int).clip(0, 23), minlength=24) / wk.date.nunique()
        axs[0].bar(hrs, v, bottom=bottom, color=c, label=p.replace(">", " → "), width=0.9)
        bottom += v
    axs[0].set_title("weekday requests per hour by trip purpose (top 8)", fontsize=10)
    axs[0].legend(frameon=False, fontsize=7.5, ncols=2); style(axs[0])
    sh = req["mode"].value_counts(normalize=True)
    comp = req.groupby("mode").apply(lambda g: (g.status == "completed").mean())
    dec = req.groupby("mode").apply(lambda g: (g.status == "declined").mean())
    names = ["bike", "auto", "cab"]
    x = np.arange(3)
    axs[1].bar(x - 0.27, [sh.get(n, 0) for n in names], 0.27, color="#2a78d6", label="share of requests")
    axs[1].bar(x, [comp.get(n, 0) for n in names], 0.27, color="#1baf7a", label="completed")
    axs[1].bar(x + 0.27, [dec.get(n, 0) for n in names], 0.27, color="#eb6834", label="declined the quote")
    axs[1].set_xticks(x, ["bike taxi", "auto-rickshaw", "cab"])
    axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axs[1].set_title("by product", fontsize=10); axs[1].legend(frameon=False, fontsize=9); style(axs[1])
    internal = done[done.external_dest < 0]               # intercity trips continue outside the map
    dur = (internal.t_dropoff - internal.t_pickup) / 60
    done = internal
    for n, c in zip(names, ["#1baf7a", "#eda100", "#eb6834"]):
        axs[2].hist(dur[done["mode"] == n].clip(upper=90), bins=45, histtype="step", lw=2, color=c, label=n, density=True)
    cab_mean = float(dur[done["mode"] == "cab"].mean())
    axs[2].axvline(33, color=viz.INK, lw=1, ls="--")
    axs[2].text(34, axs[2].get_ylim()[1] * 0.9, "Uber Pune: typical Go\nride ~33 min", fontsize=8)
    axs[2].set_title(f"trip duration within the metro (minutes); simulated cab mean {cab_mean:.0f} min", fontsize=10)
    axs[2].set_xlabel("minutes"); axs[2].legend(frameon=False, fontsize=9); style(axs[2])
    fig.suptitle("Pune metro: demand by purpose, products and trip durations (no public trip data; checks against what is known)",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, out / "09_validation.png")
    res.update({f"share_{n}": float(sh.get(n, 0)) for n in names})
    res.update({f"completion_{n}": float(comp.get(n, 0)) for n in names})
    res["cab_trip_min_mean"] = cab_mean
    res["note"] = "trip durations exclude intercity trips"
    res["airport_trips_per_day"] = float((done.purpose.str.contains("airport")).sum() / done.date.nunique())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["sf_hd", "pune_hd"])
    args = ap.parse_args()
    city = cities.get(args.city)
    geo = DM.Geography(city)
    run_dir, req, ctx, season = load_season(city)
    out = city.out_dir
    layers(city, geo, req, ctx, out)
    res = (validate_sf if args.city == "sf_hd" else validate_pune)(city, geo, req, season, out)
    res["season"] = run_dir.name
    res["days"] = int(req.date.nunique())
    (out / "09_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
