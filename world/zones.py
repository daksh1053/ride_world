"""Zone partition Z = {0, ..., Z-1} built from H3 hexagons over the road nodes.

Each road node gets the H3 cell that contains it at the city's resolution. Cells
holding fewer than ``MIN_NODES`` nodes, which are mostly slivers at the city
edge, are merged into the neighbouring cell with the most nodes, so every zone has
a usable interior. A zone may therefore cover more than one hexagon. ``zone_h3``
records the mapping.

Each zone also gets a *centre node*: the road node nearest the zone's
node-weighted centroid. Zone-to-zone travel times (stage 2) run between centre
nodes.
"""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
import geopandas as gpd

from .cities import City

MIN_NODES = 15


HEX_KM2 = {7: 5.161, 8: 0.737, 9: 0.105}


def auto_res(city: City, boundary) -> int:
    """H3 resolution whose hex count over the city area is closest to ~100 zones."""
    area = float(boundary.to_crs(city.utm_crs).area.iloc[0]) / 1e6
    return min(HEX_KM2, key=lambda r: abs(np.log(area / HEX_KM2[r] / 100)))


def hex_polygon(cell: str) -> Polygon:
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def build(city: City, nodes: pd.DataFrame, edges: pd.DataFrame, pois: pd.DataFrame,
          boundary: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    res = city.h3_res or auto_res(city, boundary)
    cells = np.array([h3.latlng_to_cell(la, lo, res) for la, lo in zip(nodes.lat, nodes.lon)])
    counts = pd.Series(cells).value_counts()

    # Merge small cells into their largest neighbour, repeating until stable.
    parent = {c: c for c in counts.index}
    size = counts.to_dict()  # nodes under each root

    def root(c):
        while parent[c] != c:
            c = parent[c]
        return c

    for c in counts.index[::-1]:  # smallest first
        rc = root(c)
        if size[rc] >= MIN_NODES:
            continue
        nbrs = {root(n) for n in h3.grid_disk(c, 1) if n in parent} - {rc}
        if nbrs:
            best = max(nbrs, key=size.get)
            parent[rc] = best
            size[best] += size.pop(rc)
    roots = pd.Series({c: root(c) for c in counts.index})
    order = counts.groupby(roots).sum().sort_values(ascending=False).index
    zone_of_root = {r: i for i, r in enumerate(order)}  # zone 0 = most nodes

    nodes = nodes.copy()
    nodes["h3"] = cells
    nodes["zone"] = roots.loc[cells].map(zone_of_root).values.astype(np.int32)
    for r in (7, 8, 9):
        nodes[f"h3_r{r}"] = [h3.latlng_to_cell(la, lo, r) for la, lo in zip(nodes.lat, nodes.lon)]

    zone_h3 = pd.DataFrame({"h3": roots.index, "zone": roots.map(zone_of_root).values})

    # Zone geometry clipped to the city boundary, for areas and plots.
    geoms = gpd.GeoDataFrame(zone_h3, geometry=[hex_polygon(c) for c in zone_h3.h3], crs="EPSG:4326")
    geoms = geoms.dissolve(by="zone").reset_index()
    clipped = gpd.clip(geoms, boundary.to_crs("EPSG:4326"))
    area = clipped.set_index("zone").to_crs(city.utm_crs).area / 1e6

    # Road length per zone, attributing each edge to the zone of its start node.
    node_zone = nodes.set_index("node_id").zone
    road_km = edges.length_m.groupby(edges.u.map(node_zone)).sum() / 1000

    # POIs by category per zone, attributed through the nearest road node.
    tree = cKDTree(np.c_[nodes.lon * np.cos(np.radians(nodes.lat.mean())), nodes.lat])
    _, idx = tree.query(np.c_[pois.lon * np.cos(np.radians(nodes.lat.mean())), pois.lat])
    pois = pois.assign(zone=nodes.zone.values[idx])
    poi_counts = pois.pivot_table(index="zone", columns="category", values="lon",
                                  aggfunc="size", fill_value=0).add_prefix("poi_")

    g = nodes.groupby("zone")
    zones = pd.DataFrame({
        "n_nodes": g.size(),
        "lon": g.lon.mean(), "lat": g.lat.mean(),
        "x_m": g.x_m.mean(), "y_m": g.y_m.mean(),
    })
    zones["area_km2"] = area.reindex(zones.index).fillna(0).round(3)
    zones["road_km"] = road_km.reindex(zones.index).fillna(0).round(2)
    zones = zones.join(poi_counts).fillna(0)
    zones["n_pois"] = zones.filter(like="poi_").sum(axis=1).astype(int)
    zones["n_hex"] = zone_h3.groupby("zone").size()

    # Centre node: road node nearest the node-weighted centroid of the zone.
    centre = []
    for z, grp in g:
        d2 = (grp.x_m - zones.at[z, "x_m"]) ** 2 + (grp.y_m - zones.at[z, "y_m"]) ** 2
        centre.append(grp.node_id.iloc[int(np.argmin(d2.values))])
    zones["centre_node"] = np.array(centre, dtype=np.int64)
    zones.index.name = "zone"
    zones = zones.reset_index()

    geoms = geoms.merge(zones[["zone", "n_nodes", "n_pois", "area_km2", "road_km"]], on="zone")
    return nodes, zones, geoms.assign(h3_res=res)


def poi_zone_counts(zones: pd.DataFrame) -> pd.DataFrame:
    return zones.filter(like="poi_")
