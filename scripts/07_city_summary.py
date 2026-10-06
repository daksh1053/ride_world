"""Cross-city summary of stages 1-3: one row per city, and a gallery of road maps.

    .venv/bin/python scripts/07_city_summary.py

Writes outputs/cities_summary.csv / .md and outputs/cities_gallery.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from world import cities, roadnet, viz


def row(k):
    c = cities.get(k)
    d = c.data_dir
    meta = json.loads((d / "meta.json").read_text())
    tmeta = json.loads((d / "traffic" / "meta.json").read_text())
    calib = pd.read_csv(d / "traffic" / "calibration.csv")
    run = next(iter(sorted((d / "sim").glob("2025-03-04_seed0_nearest"))), None)
    m = json.loads((run / "metrics.json").read_text()) if run else {}
    area = float(roadnet.boundary(c).to_crs(c.utm_crs).area.iloc[0]) / 1e6
    return {
        "city": c.name, "key": k, "tier": c.tier, "country": c.country, "population": c.population,
        "extent": "circle" if c.circle else "admin boundary", "area_km2": round(area),
        "road_km": meta["road_km"], "nodes": meta["n_nodes"], "zones": meta["n_zones"],
        "gateways": tmeta["n_gateways"], "peak_tti_target": max(c.traffic.tti_weekday),
        "calib_max_err": round(float((calib.tti - calib.target).abs().max()), 3),
        "requests_day": m.get("requests"), "drivers": c.market.n_drivers,
        "completion": m.get("completion_rate"), "wait_min": m.get("mean_wait_min"),
        "net_per_hour": m.get("mean_hourly"), "currency": c.currency, "gini": m.get("gini_net_earnings"),
        "traffic_source": c.traffic.source,
    }


def gallery(keys):
    n = len(keys)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4.3 * rows), layout="constrained")
    for ax, k in zip(axs.flat, keys):
        c = cities.get(k)
        nodes = pd.read_parquet(c.data_dir / "nodes.parquet")
        edges = pd.read_parquet(c.data_dir / "edges.parquet")
        viz.draw_roads(ax, nodes, edges)
        viz.map_axes(ax, nodes, pad=0.003)
        ax.set_title(f"{c.name}  ·  {c.tier}", fontsize=11)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"The {n} cities of the ride world (drivable OSM networks; each panel at its own scale)",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, cities.OUTPUTS / "cities_gallery.png")


def main():
    keys = [k for t in cities.TIERS.values() for k in t if (cities.DATA / k / "traffic" / "meta.json").exists()]
    df = pd.DataFrame([row(k) for k in keys])
    df.to_csv(cities.OUTPUTS / "cities_summary.csv", index=False)
    show = df.drop(columns=["key", "traffic_source"]).copy()
    for col in ("completion", "gini"):
        show[col] = show[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    for col in ("wait_min", "net_per_hour"):
        show[col] = show[col].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
    (cities.OUTPUTS / "cities_summary.md").write_text("# Cities in the ride world\n\n" + show.to_markdown(index=False) + "\n")
    print(show.to_string(index=False))
    gallery(keys)


if __name__ == "__main__":
    main()
