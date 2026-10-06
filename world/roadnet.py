"""Road graph G = (V, E_G): download from OpenStreetMap, clean, attribute, save, load.

On disk, per city (``data/<city>/``):

* ``raw/graph.graphml``   exactly what osmnx returned (drive network, simplified),
                          kept so processing can be re-run offline.
* ``raw/boundary.gpkg``   the administrative boundary polygon.
* ``raw/pois.parquet``    points of interest, used later to shape demand.
* ``graph.graphml``       processed graph: largest strongly connected component,
                          with speed / travel-time / class attributes.
* ``nodes.parquet``       node table  (node_id, x, y, lon, lat, zone, h3 cells).
* ``edges.parquet``       edge table  (u, v, key, length_m, road_class, lanes,
                          speed_kph, free_flow_s, oneway, name).
* ``meta.json``           counts, bounds and provenance.

Only the largest strongly connected component is kept so that every node can reach
every other node; otherwise a trip or reposition could target an unreachable node,
which the formulation's feasibility masks f, g would have to special-case forever.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .cities import ROAD_CLASSES, City

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_timeout = 600

POI_TAGS = {
    "amenity": True,
    "shop": True,
    "office": True,
    "tourism": True,
    "leisure": ["park", "stadium", "sports_centre"],
    "railway": ["station"],
    "aeroway": ["terminal", "aerodrome"],
    "public_transport": ["station"],
}


def _first(v):
    """OSM attributes merged during simplification can be lists; take the first."""
    if isinstance(v, list):
        return v[0] if v else None
    return v


def road_class(hwy) -> str:
    h = _first(hwy) or "other"
    base = h.removesuffix("_link")
    return base if base in ROAD_CLASSES else "other"


def _lanes(v) -> float:
    vals = v if isinstance(v, list) else [v]
    out = []
    for x in vals:
        try:
            out.append(float(str(x).split(";")[0]))
        except (TypeError, ValueError):
            pass
    return max(out) if out else np.nan


# ASSUMPTION: lanes per direction when OSM has no `lanes` tag.
DEFAULT_LANES = {
    "motorway": 3, "trunk": 2, "primary": 2, "secondary": 1.5, "tertiary": 1,
    "residential": 1, "unclassified": 1, "living_street": 1, "service": 1, "other": 1,
}


OVERPASS_URLS = ["https://lz4.overpass-api.de/api", "https://maps.mail.ru/osm/tools/overpass/api",
                 "https://overpass-api.de/api"]


def _overpass(fn, *args, **kw):
    """Call an osmnx Overpass query, retrying and switching to a mirror on refusal."""
    import requests
    for attempt in range(10):
        ox.settings.overpass_url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            return fn(*args, **kw)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.JSONDecodeError, ox._errors.ResponseStatusCodeError) as e:
            print(f"    overpass {ox.settings.overpass_url} failed ({type(e).__name__}); retrying", flush=True)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("all Overpass endpoints failed")


def download(city: City) -> None:
    raw = city.data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if city.circle:
        # No city-sized admin polygon in OSM: a circle of the municipal area's size.
        lat, lon, r_km = city.circle
        centre = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326")
        disc = centre.to_crs(city.utm_crs).buffer(r_km * 1000, resolution=32).to_crs("EPSG:4326")
        boundary = gpd.GeoDataFrame({"display_name": [f"{city.name}: circle r={r_km} km"]},
                                    geometry=disc.values, crs="EPSG:4326")
    else:
        by_id = city.osm_query[:1] in "RWN" and city.osm_query[1:].isdigit()
        boundary = ox.geocode_to_gdf(city.osm_query, by_osmid=by_id)
    boundary.to_file(raw / "boundary.gpkg", driver="GPKG")
    # ~50 m simplification keeps Overpass requests small (detailed polygons draw HTTP 413)
    poly = boundary.geometry.iloc[0].simplify(0.0005, preserve_topology=True)
    print(f"[{city.key}] boundary: {boundary.display_name.iloc[0]}")

    print(f"[{city.key}] downloading drive network ...")
    G = _overpass(ox.graph_from_polygon, poly, network_type="drive", simplify=True, retain_all=True)
    ox.save_graphml(G, raw / "graph.graphml")
    print(f"[{city.key}]   {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({time.time() - t0:.0f}s)")

    print(f"[{city.key}] downloading points of interest ...")
    pois = _overpass(ox.features_from_polygon, poly, POI_TAGS)
    pois = pois.to_crs(city.utm_crs)
    pois["geometry"] = pois.geometry.centroid
    pois = pois.to_crs("EPSG:4326")
    cat = pd.Series("other", index=pois.index)
    for tag in POI_TAGS:  # first matching tag wins, in POI_TAGS order
        if tag in pois:
            cat = cat.where(~((cat == "other") & pois[tag].notna()), tag)
    sub = pd.Series(None, index=pois.index, dtype=object)
    for tag in POI_TAGS:
        if tag in pois:
            sub = sub.where(cat != tag, pois[tag])
    out = pd.DataFrame({
        "lon": pois.geometry.x.values, "lat": pois.geometry.y.values,
        "category": cat.values, "kind": sub.astype(str).values,
        "name": pois["name"].astype(str).values if "name" in pois else None,
    })
    out.to_parquet(raw / "pois.parquet", index=False)
    print(f"[{city.key}]   {len(out):,} POIs ({time.time() - t0:.0f}s total)")


def process(city: City) -> nx.MultiDiGraph:
    raw = city.data_dir / "raw"
    G = ox.load_graphml(raw / "graph.graphml")
    n_raw = G.number_of_nodes()
    G = ox.truncate.largest_component(G, strongly=True)

    # Free-flow speed: OSM maxspeed when tagged, class default otherwise.
    # osmnx parses mph/km/h strings and falls back to hwy_speeds for untagged edges.
    G = ox.routing.add_edge_speeds(G, hwy_speeds=city.hwy_speeds_kph)
    G = ox.routing.add_edge_travel_times(G)

    for u, v, k, d in G.edges(keys=True, data=True):
        cls = road_class(d.get("highway"))
        lanes = _lanes(d.get("lanes"))
        oneway = bool(_first(d.get("oneway", False)))
        if np.isnan(lanes) or lanes <= 0:      # untagged, or tagged lanes=0 (seen in Chandigarh)
            lanes = DEFAULT_LANES[cls]
        elif not oneway:
            lanes = max(1.0, lanes / 2)  # OSM `lanes` counts both directions
        d["road_class"] = cls
        d["lanes_dir"] = float(lanes)
        d["speed_src"] = "osm" if d.get("maxspeed") is not None else "default"

    ox.save_graphml(G, city.data_dir / "graph.graphml")
    print(f"[{city.key}] largest SCC: {G.number_of_nodes():,}/{n_raw:,} nodes kept, "
          f"{G.number_of_edges():,} edges")
    return G


def tables(city: City, G: nx.MultiDiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    nodes_m = nodes_gdf.to_crs(city.utm_crs)
    nodes = pd.DataFrame({
        "node_id": nodes_gdf.index.astype(np.int64),
        "lon": nodes_gdf.x.values, "lat": nodes_gdf.y.values,
        "x_m": nodes_m.geometry.x.values, "y_m": nodes_m.geometry.y.values,
        "street_count": nodes_gdf.get("street_count", pd.Series(np.nan, index=nodes_gdf.index)).values,
    })
    e = edges_gdf.reset_index()
    edges = pd.DataFrame({
        "u": e["u"].astype(np.int64), "v": e["v"].astype(np.int64), "key": e["key"].astype(np.int64),
        "length_m": e["length"].astype(float),
        "road_class": e["road_class"].astype(str),
        "lanes_dir": e["lanes_dir"].astype(float),
        "speed_kph": e["speed_kph"].astype(float),
        "speed_src": e["speed_src"].astype(str),
        "free_flow_s": e["travel_time"].astype(float),
        "oneway": e["oneway"].map(lambda v: bool(_first(v))),
        "name": e.get("name", pd.Series(None, index=e.index)).map(lambda v: str(_first(v)) if v is not None else ""),
    })
    return nodes, edges


def write_meta(city: City, G, nodes: pd.DataFrame, edges: pd.DataFrame, extra: dict) -> dict:
    meta = {
        "city": city.name, "osm_query": city.osm_query, "crs_metric": city.utm_crs,
        "currency": city.currency, "timezone": city.timezone,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "osmnx_version": ox.__version__,
        "n_nodes": int(len(nodes)), "n_edges": int(len(edges)),
        "road_km": round(float(edges.length_m.sum()) / 1000, 1),
        "bounds_lonlat": [float(nodes.lon.min()), float(nodes.lat.min()),
                          float(nodes.lon.max()), float(nodes.lat.max())],
        "speed_from_osm_tag_frac": round(float((edges.speed_src == "osm").mean()), 3),
        "road_km_by_class": edges.groupby("road_class").length_m.sum().div(1000).round(1).to_dict(),
        **extra,
    }
    (city.data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def edge_shapes(city: City, nodes: pd.DataFrame, tol_deg: float = 1e-5) -> dict:
    """Interior shape points of every edge, keyed by (u index, v index) into `nodes`.

    The graph is simplified, so a single edge can be a kilometres-long curve (a
    freeway, a bridge approach). Anything drawn or animated between a path's nodes
    must follow these points instead of a straight line. Cached in
    `edge_shapes.parquet`; parallel edges keep the first shape seen.
    """
    path = city.data_dir / "edge_shapes.parquet"
    if not path.exists():
        G = ox.load_graphml(city.data_dir / "graph.graphml")
        rows = []
        for u, v, d in G.edges(data=True):
            geom = d.get("geometry")
            if geom is None:
                continue
            pts = np.asarray(geom.simplify(tol_deg).coords)[1:-1]
            if len(pts):
                rows.append((u, v, pts[:, 1].round(6).tolist(), pts[:, 0].round(6).tolist()))
        pd.DataFrame(rows, columns=["u", "v", "lat", "lon"]).drop_duplicates(["u", "v"]) \
            .to_parquet(path, index=False)
    df = pd.read_parquet(path)
    pos = pd.Series(np.arange(len(nodes)), index=nodes.node_id)
    ui, vi = pos.reindex(df.u).to_numpy(), pos.reindex(df.v).to_numpy()
    return {(int(a), int(b)): np.c_[la, lo] for a, b, la, lo in zip(ui, vi, df.lat, df.lon)
            if not (np.isnan(a) or np.isnan(b))}


def load(city: City) -> tuple[nx.MultiDiGraph, pd.DataFrame, pd.DataFrame]:
    d = city.data_dir
    G = ox.load_graphml(d / "graph.graphml")
    return G, pd.read_parquet(d / "nodes.parquet"), pd.read_parquet(d / "edges.parquet")


def boundary(city: City) -> gpd.GeoDataFrame:
    return gpd.read_file(city.data_dir / "raw" / "boundary.gpkg")
