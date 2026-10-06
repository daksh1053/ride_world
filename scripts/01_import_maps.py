"""Stage 1: import a city's road network from OpenStreetMap, zone it, save it, render it.

    .venv/bin/python scripts/01_import_maps.py --city sf pune         # all steps
    .venv/bin/python scripts/01_import_maps.py --city sf --redownload  # force a fresh OSM pull

Outputs
    data/<city>/      graph.graphml, nodes/edges/zones parquet, zone_geoms.geojson, meta.json
    outputs/<city>/   01_roads.png, 01_speeds.png, 01_zones.png, 01_explorer.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from world import cities, explorer, roadnet, viz, zones as zmod


def render(city, nodes, edges, zones, zone_geoms, boundary, pois):
    out = city.out_dir
    if not city.show_boundary:
        boundary = None

    # 1) road network by class
    fig, ax = plt.subplots(figsize=(10, 10))
    viz.draw_boundary(ax, boundary)
    viz.draw_roads(ax, nodes, edges)
    viz.map_axes(ax, nodes)
    viz.road_legend(ax, edges)
    ax.set_title(f"{city.name}: drivable road network (OSM)")
    ax.text(0, -0.02, f"{len(nodes):,} nodes · {len(edges):,} directed edges · "
            f"{edges.length_m.sum() / 1000:,.0f} km · largest strongly connected component",
            transform=ax.transAxes, color=viz.INK_2, fontsize=9)
    viz.save(fig, out / "01_roads.png")

    # 2) free-flow speeds: map + length-weighted distribution
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [2.2, 1]})
    viz.draw_boundary(ax, boundary)
    lc = viz.draw_roads(ax, nodes, edges, values=edges.speed_kph.to_numpy(), lw=0.6,
                        vmin=10, vmax=float(np.percentile(edges.speed_kph, 99)))
    viz.map_axes(ax, nodes)
    viz.colorbar(fig, lc, ax, "free-flow speed (km/h)")
    ax.set_title(f"{city.name}: free-flow speed per edge")
    bins = np.arange(0, edges.speed_kph.max() + 5, 5)
    for src, col, lab in [("osm", "#256abf", "OSM maxspeed tag"), ("default", "#9ec5f4", "class default")]:
        m = edges.speed_src == src
        axh.hist(edges.speed_kph[m], bins=bins, weights=edges.length_m[m] / 1000, color=col,
                 label=f"{lab} ({m.mean():.0%} of edges)", edgecolor=viz.SURFACE, linewidth=1)
    axh.set_xlabel("free-flow speed (km/h)"), axh.set_ylabel("road length (km)")
    axh.legend(frameon=False, fontsize=9)
    axh.grid(axis="y", color=viz.HAIRLINE, lw=0.6), axh.set_axisbelow(True)
    for s in ("top", "right"):
        axh.spines[s].set_visible(False)
    axh.set_title("length-weighted distribution")
    viz.save(fig, out / "01_speeds.png")

    # 3) zones: node density and POI density
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")
    for ax, col, lab in [(axs[0], "n_nodes", "road nodes per zone"), (axs[1], "n_pois", "points of interest per zone")]:
        zone_geoms.plot(ax=ax, column=col, cmap=viz.SEQ, edgecolor=viz.SURFACE, linewidth=0.8,
                        legend=True, legend_kwds={"shrink": 0.6, "label": lab})
        viz.draw_boundary(ax, boundary)
        viz.map_axes(ax, nodes)
        if len(zones) <= 200:
            for z in zones.itertuples():
                ax.text(z.lon, z.lat, str(z.zone), fontsize=5.5, ha="center", va="center",
                        color=viz.INK, path_effects=[pe.withStroke(linewidth=1.6, foreground=viz.SURFACE)])
        ax.set_title(lab)
    fig.suptitle(f"{city.name}: zone partition Z — {len(zones)} zones "
                 f"(H3 res {int(zone_geoms.h3_res.iloc[0])}, cells with <{zmod.MIN_NODES} nodes merged)",
                 x=0.01, ha="left", fontweight="bold")
    viz.save(fig, out / "01_zones.png")


def run(key: str, redownload: bool):
    city = cities.get(key)
    print(f"== {city.name}")
    raw = city.data_dir / "raw"
    if redownload or not ((raw / "graph.graphml").exists() and (raw / "pois.parquet").exists()):
        roadnet.download(city)
    G = roadnet.process(city)
    nodes, edges = roadnet.tables(city, G)
    boundary = roadnet.boundary(city)
    pois = pd.read_parquet(city.data_dir / "raw" / "pois.parquet")

    nodes, zones, zone_geoms = zmod.build(city, nodes, edges, pois, boundary)
    d = city.data_dir
    nodes.to_parquet(d / "nodes.parquet", index=False)
    edges.to_parquet(d / "edges.parquet", index=False)
    zones.to_parquet(d / "zones.parquet", index=False)
    zone_geoms.to_file(d / "zone_geoms.geojson", driver="GeoJSON")
    meta = roadnet.write_meta(city, G, nodes, edges, {
        "n_zones": int(len(zones)), "h3_res": int(zone_geoms.h3_res.iloc[0]), "n_pois": int(len(pois)),
        "zone_nodes_min_median_max": [int(zones.n_nodes.min()), int(zones.n_nodes.median()),
                                      int(zones.n_nodes.max())],
    })
    print(json.dumps({k: meta[k] for k in ("n_nodes", "n_edges", "road_km", "n_zones",
                                            "n_pois", "speed_from_osm_tag_frac")}))

    render(city, nodes, edges, zones, zone_geoms, boundary, pois)
    explorer.build(city, G, nodes, edges, zones, zone_geoms, city.out_dir / "01_explorer.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", nargs="+", default=["sf", "pune"], choices=sorted(cities.CITIES))
    ap.add_argument("--redownload", action="store_true")
    args = ap.parse_args()
    for k in args.city:
        run(k, args.redownload)
