"""Build a ready-to-run `World` for a city, date and seed (shared by stages 3 and 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import market as MK, traffic as T
from .sim import SimConfig, World


class CityData:
    """Static inputs loaded once per city: graph tables, traffic, zone weights, gateways."""

    def __init__(self, city):
        d = city.data_dir
        self.city = city
        self.nodes = pd.read_parquet(d / "nodes.parquet")
        self.edges = pd.read_parquet(d / "edges.parquet")
        self.zones = pd.read_parquet(d / "zones.parquet")
        self.net = T.Net.from_tables(self.nodes, self.edges)
        self.traffic = T.TrafficModel(city)
        zw = pd.read_parquet(d / "traffic" / "zone_weights.parquet")
        self.P, self.A = zw.home_weight.to_numpy(), zw.activity_weight.to_numpy()
        gw = pd.read_parquet(d / "traffic" / "gateways.parquet")
        pos = pd.Series(np.arange(len(self.nodes)), index=self.nodes.node_id)
        self.gw_pos = {}
        for k, e in enumerate(city.market.external):
            if e.gateway_lonlat is None:      # gateways are sorted by road capacity
                j = min(e.gateway_rank, len(gw) - 1)
            else:
                j = int(np.argmin((gw.lon - e.gateway_lonlat[0]) ** 2 + (gw.lat - e.gateway_lonlat[1]) ** 2))
            self.gw_pos[k] = int(pos.loc[gw.node_id.iloc[j]])
        self.zone_nodes = MK.zone_node_sampler(self.nodes, self.edges)


def build_world(data: CityData, cfg: SimConfig, policy, log=print) -> World:
    """People are drawn from `cfg.seed`, so the same seed gives the same drivers,
    customers and requests whatever the policy (common random numbers)."""
    city = data.city
    rng = np.random.default_rng(cfg.seed)
    drivers, dlat = MK.make_drivers(city, data.zones, data.P, rng)
    customers, clat = MK.make_customers(city, data.zones, data.P, rng)
    req = MK.make_requests(city, cfg.date, data.zones, customers, clat, data.zone_nodes,
                           data.P, data.A, data.gw_pos, rng)
    log(f"  {len(drivers):,} drivers, {len(customers):,} customers, {len(req):,} requests on {cfg.date}")
    return World(city, data.net, data.nodes, data.zones, data.traffic, drivers, dlat, customers,
                 clat, req, data.gw_pos, policy, cfg)
