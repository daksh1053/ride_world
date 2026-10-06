"""Order data and the Manhattan zone map (paper Sec. VI-A, "Dataset").

1. *Order Data* — NYC TLC yellow-taxi records on Manhattan Island. "Each taxi order
   data contains departure and destination locations, order starting time, trip fare,
   and trip mileage."
2. *Map Data* — "the Manhattan taxi zone map provided by TLC ... We index each zone
   to denote the pick-up and drop-off zone in the order data."
3. *Order Data Processing* — "We remove some orders with noise (i.e., orders with
   invalid fare, zero trip mileage, and so on). We also eliminate the order data in
   those isolated zones."
4. *Shortest Path Cache* — "we prebuilt the cache of the shortest path matrix and the
   shortest path distance matrix."

Sec. VI-A also fixes the sampling rule for the experiments: "We now compute the
average number of order data from 18:00 to 22:00 on weekdays in these 20 days, and
randomly choose the average number of 1 day as the order input."
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, DATA_PROC, DATA_RAW, MILES_TO_KM, DataConfig

_RAW_COLUMNS = [
    "tpep_pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
]


def _download(url: str, name: str) -> Path:
    dest = DATA_RAW / name
    if not dest.exists():
        print(f"[data] downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def manhattan_zones(cfg: DataConfig = CONFIG.data) -> pd.DataFrame:
    """The zone map: Manhattan taxi zones, indexed 0..K-1 by LocationID order."""
    zones = pd.read_csv(_download(cfg.zone_url, cfg.zone_name))
    zones = zones.loc[zones.Borough == cfg.borough].sort_values("LocationID")
    return zones.reset_index(drop=True)


def _load_orders(cfg: DataConfig, zone_ids: np.ndarray) -> pd.DataFrame:
    name = f"yellow_tripdata_{cfg.month}.parquet"
    df = pd.read_parquet(_download(f"{cfg.base_url}/{name}", name), columns=_RAW_COLUMNS)
    df = df.rename(
        columns={"tpep_pickup_datetime": "time", "PULocationID": "pu", "DOLocationID": "do"}
    )

    df = df.loc[df.pu.isin(zone_ids) & df["do"].isin(zone_ids)]
    # "weekdays", "18:00 to 22:00"
    df = df.loc[df.time.dt.dayofweek < 5]
    df = df.loc[df.time.dt.hour.between(cfg.start_hour, cfg.end_hour - 1)]
    # Noise removal: invalid fare, zero mileage.
    df = df.loc[(df.fare_amount > 0) & (df.fare_amount < 500)]
    df = df.loc[df.trip_distance.between(0.05, 40.0)]
    return df.reset_index(drop=True)


def _zone_distance_matrix(df: pd.DataFrame, n_zones: int) -> np.ndarray:
    """Mean observed trip distance (km) per ordered zone pair; NaN if unobserved."""
    agg = df.groupby(["pu_idx", "do_idx"], observed=True)["distance_km"].mean().reset_index()
    w = np.full((n_zones, n_zones), np.nan)
    w[agg.pu_idx.to_numpy(), agg.do_idx.to_numpy()] = agg.distance_km.to_numpy()
    np.fill_diagonal(w, 0.0)
    return w


def prepare(cfg: DataConfig = CONFIG.data, force: bool = False) -> dict:
    """Build and cache the order pool, the zone graph and the active-time weights."""
    order_path = DATA_PROC / "orders.parquet"
    graph_path = DATA_PROC / "zone_graph.npz"
    if order_path.exists() and graph_path.exists() and not force:
        return load()

    zones = manhattan_zones(cfg)
    zone_ids = zones.LocationID.to_numpy()
    remap = {z: i for i, z in enumerate(zone_ids)}
    n_zones = len(zone_ids)

    df = _load_orders(cfg, zone_ids)
    df["pu_idx"] = df.pu.map(remap).astype(np.int16)
    df["do_idx"] = df["do"].map(remap).astype(np.int16)
    df["distance_km"] = df.trip_distance * MILES_TO_KM
    print(f"[data] {len(df):,} Manhattan weekday orders {cfg.start_hour}:00-{cfg.end_hour}:00")

    # "eliminate the order data in those isolated zones": a zone that never appears
    # as both an origin and a destination cannot be connected in the graph.
    origins = set(df.pu_idx.unique().tolist())
    dests = set(df.do_idx.unique().tolist())
    isolated = sorted(set(range(n_zones)) - (origins & dests))
    if isolated:
        print(f"[data] dropping {len(isolated)} isolated zone(s): {isolated}")
        keep = ~(df.pu_idx.isin(isolated) | df.do_idx.isin(isolated))
        df = df.loc[keep].reset_index(drop=True)
        active = sorted((origins & dests))
        reindex = {z: i for i, z in enumerate(active)}
        df["pu_idx"] = df.pu_idx.map(reindex).astype(np.int16)
        df["do_idx"] = df.do_idx.map(reindex).astype(np.int16)
        zones = zones.iloc[active].reset_index(drop=True)
        n_zones = len(active)

    weights = _zone_distance_matrix(df, n_zones)

    # Per-day order counts, for the "average number of orders of 1 day" input size.
    dates = sorted(df.time.dt.normalize().unique())[: cfg.n_days]
    df = df.loc[df.time.dt.normalize().isin(dates)].reset_index(drop=True)
    per_day = df.groupby(df.time.dt.normalize()).size()
    print(f"[data] {len(dates)} weekdays, mean {per_day.mean():,.0f} orders/day")

    # Slot index inside the 18:00-22:00 window, used for the active-time weights.
    df["slot"] = (
        (df.time.dt.hour - cfg.start_hour) * 60 + df.time.dt.minute
    ).astype(np.int16)

    df = df[["time", "slot", "pu_idx", "do_idx", "distance_km", "fare_amount"]]
    df.to_parquet(order_path)
    np.savez_compressed(
        graph_path,
        weights=weights,
        zone_ids=zones.LocationID.to_numpy(),
        mean_orders_per_day=np.array([per_day.mean()]),
    )
    zones.to_csv(DATA_PROC / "zones.csv", index=False)
    return load()


def load() -> dict:
    g = np.load(DATA_PROC / "zone_graph.npz")
    return {
        "orders": pd.read_parquet(DATA_PROC / "orders.parquet"),
        "weights": g["weights"],
        "zone_ids": g["zone_ids"],
        "mean_orders_per_day": float(g["mean_orders_per_day"][0]),
    }


def sample_day(bundle: dict, seed: int = 0, scale: float | None = None) -> pd.DataFrame:
    """"randomly choose the average number of 1 day as the order input".

    `scale` multiplies the daily order count; see `DataConfig.order_scale`.
    """
    orders = bundle["orders"]
    scale = CONFIG.data.order_scale if scale is None else scale
    n = int(round(bundle["mean_orders_per_day"] * scale))
    n = min(max(n, 1), len(orders))
    day = orders.sample(n=n, random_state=seed).sort_values("slot").reset_index(drop=True)
    return day


def active_time_weights(bundle: dict, n_slots: int) -> np.ndarray:
    """xi^t, the weight for active time in Definition 6.

    Sec. III: "both order volume and the potential income fluctuate throughout the
    day. Forcing income to be proportional to working hours can lead to income
    unfairness among drivers working at different time periods. As a remedy, we
    assign different weights to different time periods."

    ASSUMPTION: the paper (following [11]) does not give a formula. We take xi^t to be
    the slot's mean earning potential — the total fare raised in slot t, normalised so
    that the mean weight over the horizon is 1. Dividing a slot's income by xi^t then
    expresses it relative to what was earnable in that slot.
    """
    orders = bundle["orders"]
    totals = np.zeros(n_slots, dtype=np.float64)
    grouped = orders.groupby("slot")["fare_amount"].sum()
    for slot, value in grouped.items():
        if 0 <= slot < n_slots:
            totals[slot] = value
    positive = totals[totals > 0]
    if positive.size == 0:
        return np.ones(n_slots)
    weights = totals / positive.mean()
    return np.where(weights > 0, weights, 1.0)
