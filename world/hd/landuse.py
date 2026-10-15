"""Where people live and where they go, per road node.

Every trip purpose gets a weight per road node, built from real data:

    home        GHSL 2025 resident population (100 m grid), each cell to its nearest node
    work        OSM commercial + industrial + office land use (area) and office POIs
    shop        OSM retail land use (area) and shop / mall POIs
    food_night  restaurant, cafe, bar, pub, nightclub, fast food POIs
    education   university, college, school land and POIs
    health      hospital and clinic land and POIs
    hub         railway and bus stations, ferry terminals (airport handled separately)
    leisure     parks, stadiums, cinemas, theatres, tourist attractions, hotels
    airport     airport terminal nodes (aeroway=terminal / aerodrome inside the extent)

Polygons are spread over the road nodes inside them (or the nearest node to their
centroid when none is inside), in proportion to area. POIs go to their nearest node.
Output: data/<city>/hd/node_weights.parquet, one column per purpose, each summing to 1.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
import rasterio.mask
from scipy.spatial import cKDTree

from .. import roadnet
from ..cities import City, DATA

GHSL = DATA / "_external" / "ghsl"
PURPOSES = ["home", "work", "shop", "food_night", "education", "health", "hub", "leisure", "airport"]

LANDUSE_TAGS = {
    "landuse": ["residential", "commercial", "retail", "industrial"],
    "amenity": ["university", "college", "school", "hospital", "clinic"],
    "aeroway": ["terminal", "aerodrome"],
    "leisure": ["park", "stadium"],
    "building": ["office", "commercial", "retail"],
}

# polygon tag -> (purpose, weight per km^2)
POLY_RULES = {
    ("landuse", "commercial"): ("work", 1.0), ("landuse", "industrial"): ("work", 0.5),
    ("building", "office"): ("work", 4.0), ("building", "commercial"): ("work", 2.0),
    ("landuse", "retail"): ("shop", 1.0), ("building", "retail"): ("shop", 3.0),
    ("amenity", "university"): ("education", 1.0), ("amenity", "college"): ("education", 1.0),
    ("amenity", "school"): ("education", 0.5),
    ("amenity", "hospital"): ("health", 2.0), ("amenity", "clinic"): ("health", 1.0),
    ("leisure", "park"): ("leisure", 0.2), ("leisure", "stadium"): ("leisure", 2.0),
}
# POI (category, kind) -> (purpose, weight per POI)
POI_RULES = {
    "office": ("work", 1.0),
    "shop": ("shop", 0.5),
    ("amenity", "restaurant"): ("food_night", 1.0), ("amenity", "cafe"): ("food_night", 0.5),
    ("amenity", "fast_food"): ("food_night", 0.5), ("amenity", "bar"): ("food_night", 1.5),
    ("amenity", "pub"): ("food_night", 1.5), ("amenity", "nightclub"): ("food_night", 3.0),
    ("amenity", "university"): ("education", 5.0), ("amenity", "college"): ("education", 3.0),
    ("amenity", "school"): ("education", 1.0),
    ("amenity", "hospital"): ("health", 5.0), ("amenity", "clinic"): ("health", 1.0),
    ("amenity", "cinema"): ("leisure", 2.0), ("amenity", "theatre"): ("leisure", 2.0),
    ("amenity", "bus_station"): ("hub", 3.0), ("amenity", "ferry_terminal"): ("hub", 3.0),
    ("railway", "station"): ("hub", 5.0), ("public_transport", "station"): ("hub", 2.0),
    ("tourism", "hotel"): ("leisure", 1.0), ("tourism", "attraction"): ("leisure", 2.0),
    ("tourism", "museum"): ("leisure", 1.5), ("leisure", "stadium"): ("leisure", 3.0),
}


def _tile_for(lon, lat):
    import pyproj
    x, y = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True).transform(lon, lat)
    return f"R{int((9000000 - y) // 1e6) + 1}_C{int((x + 18041000) // 1e6) + 1}"


def population_to_nodes(city: City, nodes: pd.DataFrame, boundary) -> np.ndarray:
    """Resident population per road node from the GHSL 100 m grid."""
    lon, lat = float(nodes.lon.mean()), float(nodes.lat.mean())
    tif = GHSL / f"GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_{_tile_for(lon, lat)}.tif"
    with rasterio.open(tif) as src:
        arr, tr = rasterio.mask.mask(src, boundary.to_crs("ESRI:54009").geometry, crop=True, nodata=0)
    arr = np.where(arr[0] > 0, arr[0], 0)
    rows, cols = np.nonzero(arr)
    xs, ys = rasterio.transform.xy(tr, rows, cols)
    cells = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs="ESRI:54009").to_crs(city.utm_crs)
    tree = cKDTree(nodes[["x_m", "y_m"]].to_numpy())
    _, idx = tree.query(np.c_[cells.x, cells.y])
    pop = np.bincount(idx, weights=arr[rows, cols], minlength=len(nodes))
    return pop


def download_landuse(city: City, boundary) -> gpd.GeoDataFrame:
    path = city.data_dir / "hd" / "landuse.gpkg"
    if path.exists():
        return gpd.read_file(path)
    poly = boundary.geometry.iloc[0].simplify(0.0005, preserve_topology=True)
    g = roadnet._overpass(ox.features_from_polygon, poly, LANDUSE_TAGS)
    g = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    keep = [c for c in LANDUSE_TAGS if c in g]
    g = g[keep + ["geometry"]].reset_index(drop=True)
    for c in keep:
        g[c] = g[c].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(path, driver="GPKG")
    return g


def build(city: City, log=print) -> pd.DataFrame:
    d = city.data_dir
    nodes = pd.read_parquet(d / "nodes.parquet")
    boundary = roadnet.boundary(city)
    W = pd.DataFrame(0.0, index=range(len(nodes)), columns=PURPOSES)

    W["home"] = population_to_nodes(city, nodes, boundary)
    log(f"  population: {W.home.sum() / 1e6:.2f} M residents on {int((W.home > 0).sum()):,} nodes")

    lu = download_landuse(city, boundary).to_crs(city.utm_crs)
    tree = cKDTree(nodes[["x_m", "y_m"]].to_numpy())
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(nodes.x_m, nodes.y_m), crs=city.utm_crs)
    counts = {}
    for (tag, val), (purpose, w) in POLY_RULES.items():
        if tag not in lu:
            continue
        sub = lu[lu[tag] == val]
        if not len(sub):
            continue
        inside = gpd.sjoin(pts, sub[["geometry"]].assign(area=sub.area / 1e6), predicate="within")
        per_poly = inside.groupby("index_right").size()
        hit = inside.index_right.map(per_poly)
        np.add.at(W[purpose].values, inside.index.to_numpy(), w * inside.area.to_numpy() / hit.to_numpy())
        missing = sub.index.difference(per_poly.index)       # small polygons: nearest node to centroid
        if len(missing):
            cent = sub.loc[missing].geometry.centroid
            _, idx = tree.query(np.c_[cent.x, cent.y])
            np.add.at(W[purpose].values, idx, w * (sub.loc[missing].area / 1e6).to_numpy())
        counts[f"{tag}={val}"] = len(sub)
    term = lu[(lu.get("aeroway", "") == "terminal")] if "aeroway" in lu else lu.iloc[:0]
    if len(term):
        _, idx = tree.query(np.c_[term.geometry.centroid.x, term.geometry.centroid.y])
        np.add.at(W["airport"].values, idx, 1.0)
    log(f"  land-use polygons: {json.dumps(counts)}; airport terminals in extent: {len(term)}")

    pois = pd.read_parquet(d / "raw" / "pois.parquet")
    pg = gpd.GeoSeries(gpd.points_from_xy(pois.lon, pois.lat), crs="EPSG:4326").to_crs(city.utm_crs)
    _, pidx = tree.query(np.c_[pg.x, pg.y])
    for rule, (purpose, w) in POI_RULES.items():
        m = (pois.category == rule) if isinstance(rule, str) else ((pois.category == rule[0]) & (pois.kind == rule[1]))
        np.add.at(W[purpose].values, pidx[m.to_numpy()], w)

    # every purpose gets a 2% uniform floor over nodes with population or any activity, so
    # no mapped neighbourhood is unreachable; then each column is normalised to sum 1
    live = (W.sum(axis=1) > 0).to_numpy()
    for p in PURPOSES:
        if W[p].sum() == 0:
            continue
        col = W[p].to_numpy()
        col = col / col.sum()
        if p != "airport":
            col = 0.98 * col + 0.02 * live / live.sum()
        W[p] = col
    W.insert(0, "node_id", nodes.node_id.to_numpy())
    W["population"] = population_to_nodes(city, nodes, boundary)
    out = d / "hd"
    out.mkdir(exist_ok=True)
    W.to_parquet(out / "node_weights.parquet", index=False)
    log(f"  wrote {out / 'node_weights.parquet'}")
    return W
