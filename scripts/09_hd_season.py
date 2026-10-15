"""Stage 9: run a season of a high-detail city and draw its days.

    .venv/bin/python scripts/09_hd_season.py --city sf_hd --days 14
    .venv/bin/python scripts/09_hd_season.py --city pune_hd --days 14 --start 2025-07-07

Runs consecutive days with people carried over (world/hd/run.py), saves every day to
data/<city>/hd_runs/<season>/<date>/, and writes
    outputs/<city>/09_season.png      daily demand, service, surge, earnings, churn, loyalty
    outputs/<city>/09_replay.html     cab replay of the --replay-day (default: 3rd day), 17-20h
    outputs/<city>/09_season.csv      one row per day
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from world import cities, replay, roadnet, viz
from world.hd.run import Season


def plot(city, df, out):
    x = range(len(df))
    lab = [f"{r.weekday} {r.date[5:]}" + (" ☂" if r.rain > 0.05 else "") for r in df.itertuples()]
    panels = [("requests", "requests per day", None), ("completion", "share of requests completed", "pct"),
              ("declined", "share of riders declining the quote", "pct"), ("mean_surge", "mean surge multiplier", None),
              ("net_per_hour", f"driver net per online hour ({city.currency})", None), ("gini", "driver earnings Gini", None),
              ("drivers_pool", "driver pool (after quits and joins)", None), ("mean_loyalty", "mean rider loyalty", None)]
    if "share_bike" in df:
        panels.insert(3, ("share_bike", "share of requests by product", "pct"))
    nc = (len(panels) + 1) // 2
    fig, axs = plt.subplots(2, nc, figsize=(5.5 * nc, 8), layout="constrained")
    for ax in axs.flat[len(panels):]:
        ax.axis("off")
    for ax, (col, title, fmt) in zip(axs.flat, panels):
        ax.plot(x, df[col], color="#2a78d6", lw=2, marker="o", ms=4)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(list(x), lab, rotation=60, fontsize=7.5)
        ax.grid(color=viz.HAIRLINE, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if fmt == "pct":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    if "share_bike" in df:
        ax = axs.flat[3]
        ax.lines[0].remove()
        for m, c in [("bike", "#1baf7a"), ("auto", "#eda100"), ("cab", "#eb6834")]:
            ax.plot(x, df[f"share_{m}"], color=c, lw=2, marker="o", ms=4, label=m)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{city.name}: {len(df)} consecutive simulated days (☂ = rain)", x=0.01, ha="left", fontweight="bold")
    viz.save(fig, out / "09_season.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["sf_hd", "pune_hd"])
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--start", default="2025-03-03")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-day", type=int, default=2)
    ap.add_argument("--name", default=None)
    ap.add_argument("--replay-only", action="store_true", help="redraw from the saved season, no simulation")
    args = ap.parse_args()
    city = cities.get(args.city)
    name = args.name or f"season_{args.start}_seed{args.seed}"
    run_dir = city.data_dir / "hd_runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    replay_tb = None
    dates = pd.date_range(args.start, periods=args.days)
    if args.replay_only:
        df = pd.read_csv(run_dir / "season.csv")
        d = run_dir / str(dates[args.replay_day].date())
        replay_tb = {p.stem: pd.read_parquet(p) for p in d.glob("*.parquet")}
        replay_drv = season_drivers_before(run_dir, dates[args.replay_day])
    else:
        season = Season(args.city, seed=args.seed)
        print(f"== {city.name}: {len(season.drv)} drivers, {len(season.cust):,} riders, airport zone {season.airport_zone}")
        for k, date in enumerate(dates):
            tb = season.run_day(date, out_dir=run_dir)
            if k == args.replay_day:
                replay_tb, replay_drv = tb, season_drivers_before(run_dir, date)
        df = pd.DataFrame(season.summary)
    out = city.out_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "09_season.csv", index=False)
    df.to_csv(run_dir / "season.csv", index=False)
    plot(city, df, out)
    if replay_tb is not None:
        nodes = pd.read_parquet(city.data_dir / "nodes.parquet")
        zg = gpd.read_file(city.data_dir / "zone_geoms.geojson")
        r = replay_tb["requests"]
        r = r[r.status != "not_raised"]
        w = r.wait_s[r.t_pickup.notna()] / 60
        o = replay_tb["offers"]
        summ = {**df.iloc[args.replay_day].to_dict(), "completion_rate": float((r.status == "completed").mean()),
                "cancel_rate": float((r.status == "cancelled").mean()), "mean_wait_min": float(w.mean()),
                "p90_wait_min": float(w.quantile(0.9)), "offer_acceptance": float((o.response == "accepted").mean()),
                "intercity_trips": int(((r.external_dest >= 0) & (r.status == "completed")).sum()),
                "mean_hourly": df.net_per_hour.iat[args.replay_day], "gini_net_earnings": df.gini.iat[args.replay_day]}
        replay.build_replay(city, nodes, zg, replay_tb, replay_drv, 17.0, 20.0, out / "09_replay.html", summ,
                            roadnet.edge_shapes(city, nodes),
                            title=f"{city.name}: {df.weekday.iat[args.replay_day]} {df.date.iat[args.replay_day]}")


def season_drivers_before(run_dir, date):
    """The driver table as it stood during that day (saved after the day; ids are positions)."""
    return pd.read_parquet(run_dir / str(pd.Timestamp(date).date()) / "drivers.parquet")


if __name__ == "__main__":
    main()
