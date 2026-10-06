"""NYC TLC extraction and graph construction (paper Sec. 5.1).

A note on the source data
-------------------------
Sec. 5.1 says "each request's pickup and drop-off locations are recorded in
longitude and latitude coordinates" and that trips are extracted by a Manhattan
geographic filter, with "multiple locations ... merged together as a node in
constructed graph". The TLC has since re-issued the 2016 archives in Parquet with
the raw coordinates replaced by `PULocationID` / `DOLocationID` taxi-zone IDs, so
the original per-trip coordinates are no longer publicly downloadable.

That replacement is exactly the merge step the paper performs: a taxi zone *is* a
set of coordinates merged into one location. We therefore take the node set `L` to
be the Manhattan taxi zones and replace the bounding-box filter with the equivalent
`Borough == "Manhattan"` filter. Edge weights are still the mean observed travel
distance per ordered node pair, as the paper specifies.

Pipeline
--------
1. Download `yellow_tripdata_2016-03.parquet` and `taxi_zone_lookup.csv`.
2. Keep trips whose pickup *and* dropoff zones are in Manhattan, with pickup time in
   [2016-03-01, 2016-04-01).
3. Build the directed graph: edge `e_{i,j}` weight = mean travel distance i -> j.
4. Emit the hourly OD count series (forecasting) and the peak-window request stream
   `(t_r, s_r, d_r)` (allocation experiments).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, DATA_PROC, DATA_RAW, DataConfig

_RAW_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
]


def _download(url: str, name: str) -> Path:
    dest = DATA_RAW / name
    if not dest.exists():
        print(f"[data] downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def manhattan_zones(cfg: DataConfig = CONFIG.data) -> pd.DataFrame:
    """The node set L: Manhattan taxi zones, ordered by LocationID."""
    zones = pd.read_csv(_download(cfg.zone_url, cfg.zone_name))
    zones = zones.loc[zones.Borough == "Manhattan"].sort_values("LocationID")
    return zones.reset_index(drop=True)


def _load_and_filter(cfg: DataConfig, zone_ids: np.ndarray) -> pd.DataFrame:
    frames = []
    for month in cfg.months:
        name = f"yellow_tripdata_{month}.parquet"
        path = _download(f"{cfg.base_url}/{name}", name)
        frames.append(pd.read_parquet(path, columns=_RAW_COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            "tpep_pickup_datetime": "pickup_time",
            "tpep_dropoff_datetime": "dropoff_time",
            "PULocationID": "pu",
            "DOLocationID": "do",
        }
    )

    in_manhattan = df.pu.isin(zone_ids) & df["do"].isin(zone_ids)
    in_period = df.pickup_time.between(
        pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date), inclusive="left"
    )
    df = df.loc[in_manhattan & in_period].copy()

    # Drop degenerate records: implausible durations or distances.
    duration = (df.dropoff_time - df.pickup_time).dt.total_seconds()
    df = df.loc[(duration > 60) & (duration < 3 * 3600)]
    df = df.loc[df.trip_distance.between(0.1, 30.0)]
    return df.reset_index(drop=True)


def _build_graph_weights(df: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """Mean observed travel distance for each ordered node pair; NaN if unobserved.

    "the shortest travel time from a certain pickup to a certain drop-off location is
    re-calculated as the mean travel time across the time period" (Sec. 5.1).
    """
    agg = df.groupby(["s", "d"], observed=True)["trip_distance"].mean().reset_index()
    w = np.full((n_nodes, n_nodes), np.nan)
    w[agg.s.to_numpy(), agg.d.to_numpy()] = agg.trip_distance.to_numpy()
    np.fill_diagonal(w, 0.0)
    return w


def _stratified_sample(df: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Stratified sample by (date, hour, origin node) — Sec. 5.2."""
    if rate >= 1.0:
        return df
    strata = [df.pickup_time.dt.date, df.pickup_time.dt.hour, df.s]
    return (
        df.groupby(strata, observed=True, group_keys=False)[df.columns.tolist()]
        .apply(lambda g: g.sample(frac=rate, random_state=seed))
        .reset_index(drop=True)
    )


def prepare(cfg: DataConfig = CONFIG.data, force: bool = False) -> dict:
    """Run the full pipeline and cache artefacts under `data/processed`."""
    paths = [DATA_PROC / n for n in ("requests.parquet", "graph.npz", "hourly_od_counts.npz")]
    if all(p.exists() for p in paths) and not force:
        return load()

    zones = manhattan_zones(cfg)
    zone_ids = zones.LocationID.to_numpy()
    n_nodes = len(zone_ids)
    remap = {z: i for i, z in enumerate(zone_ids)}
    print(f"[data] |L| = {n_nodes} Manhattan taxi zones")

    df = _load_and_filter(cfg, zone_ids)
    df["s"] = df.pu.map(remap).astype(np.int16)
    df["d"] = df["do"].map(remap).astype(np.int16)
    print(f"[data] {len(df):,} Manhattan trips in period")

    weights = _build_graph_weights(df, n_nodes)

    # Hourly OD counts over the whole month drive the forecasting module (Sec. 4.2).
    hours = pd.date_range(cfg.start_date, cfg.end_date, freq="h", inclusive="left")
    hour_idx = (
        (df.pickup_time - pd.Timestamp(cfg.start_date)) // pd.Timedelta("1h")
    ).astype(int)
    counts = np.zeros((len(hours), n_nodes, n_nodes), dtype=np.float32)
    np.add.at(counts, (hour_idx.to_numpy(), df.s.to_numpy(), df.d.to_numpy()), 1.0)

    # Peak-window request stream used by the allocation experiments (Sec. 5.2).
    peak = df.loc[
        df.pickup_time.dt.hour.between(
            cfg.peak_start_hour, cfg.peak_start_hour + cfg.peak_hours - 1
        )
    ].copy()
    # Sampling is applied per-split in `split_days`, not here.
    peak = peak.sort_values("pickup_time").reset_index(drop=True)
    peak = peak[["pickup_time", "s", "d", "trip_distance"]]
    print(f"[data] {len(peak):,} sampled peak-window requests")

    peak.to_parquet(paths[0])
    np.savez_compressed(paths[1], weights=weights, zone_ids=zone_ids)
    np.savez_compressed(paths[2], counts=counts)
    zones.to_csv(DATA_PROC / "nodes.csv", index=False)
    return load()


def load() -> dict:
    """Load cached artefacts."""
    g = np.load(DATA_PROC / "graph.npz")
    h = np.load(DATA_PROC / "hourly_od_counts.npz")
    return {
        "requests": pd.read_parquet(DATA_PROC / "requests.parquet"),
        "weights": g["weights"],
        "zone_ids": g["zone_ids"],
        "hourly_counts": h["counts"],
    }


def split_days(
    requests: pd.DataFrame, cfg: DataConfig = CONFIG.data
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Split the request stream into training days and the seven test days.

    The 0.05 stratified sampling of Sec. 5.2 is applied to the training days (it is
    there to cut training time); the test horizon keeps the full peak demand.
    """
    dates = sorted(requests.pickup_time.dt.normalize().unique())
    test_start = pd.Timestamp(cfg.test_start_date)
    by_day = {d: requests.loc[requests.pickup_time.dt.normalize() == d] for d in dates}

    train = [
        _stratified_sample(by_day[d], cfg.sample_rate, cfg.seed)
        for d in dates if d < test_start
    ]
    test = [
        _stratified_sample(by_day[d], cfg.test_sample_rate, cfg.seed)
        for d in dates if d >= test_start
    ]
    return train, test[: cfg.n_test_days]
