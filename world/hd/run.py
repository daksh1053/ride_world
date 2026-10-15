"""Run a season (consecutive days) of a high-detail city, carrying people across days.

Between days:
  drivers   ratings and tenure accumulate; each driver's online-hour net earnings feed a
            7-day rolling mean; a driver earning below 60% of their hidden reservation
            wage for their last 3 work days quits with probability 0.05 per day; new
            drivers join to replace quitters (and a little more when earnings are good).
  riders    a rider whose request was declined, cancelled or waited > 10 min loses 8% of
            loyalty; a good trip regains 3% (capped at 1). Their riding frequency is
            activity x loyalty, and total demand scales with mean loyalty.
Each day is saved to data/<city>/hd_runs/<season>/<date>/ (all tables) and a season
summary to .../season.csv.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .. import cities, metrics, traffic as T
from ..sim import SimConfig
from . import demand as DM
from . import people as PP
from .sim import HDDispatch, HDWorld

HD = {"sf_hd": {"requests": 12000, "drivers": 1300, "customers": 80000},
      "pune_hd": {"requests": 20000, "drivers": 4000, "customers": 140000}}


class Season:
    def __init__(self, key: str, seed: int = 0, policy_factory=HDDispatch, n_drivers: int | None = None,
                 log=print):
        cfg = HD[key]
        city = cities.get(key)
        self.city = replace(city, market=replace(city.market, requests_per_day=cfg["requests"],
                                                 n_drivers=n_drivers or cfg["drivers"],
                                                 n_customers=cfg["customers"]))
        self.seed, self.log, self.policy_factory = seed, log, policy_factory
        self.geo = DM.Geography(self.city)
        self.sf = DM.SFDemand(self.geo) if key == "sf_hd" else None
        self.data = _CityData(self.city, self.geo)
        rng = np.random.default_rng(seed)
        pop = self.geo.W.population.to_numpy()
        self.drv, self.dlat = PP.make_drivers(self.city, self.city.market.n_drivers, pop, self.geo.node_zone, rng)
        self.cust, self.clat = PP.make_customers(self.city, self.city.market.n_customers, pop, self.geo.node_zone, rng)
        self.history = {}                         # driver_id -> list of daily hourly net
        self.rng = rng
        air = np.flatnonzero(self.geo.W.airport.to_numpy() > 0)
        self.airport_zone = int(np.bincount(self.geo.node_zone[air]).argmax()) if len(air) else -1
        self.summary = []

    def run_day(self, date, out_dir=None, horizon_h=24.0):
        t0 = time.time()
        date = pd.Timestamp(date)
        day_seed = self.seed * 1000 + date.dayofyear
        cfg = SimConfig(date=str(date.date()), seed=day_seed, horizon_s=horizon_h * 3600)
        cond = self.data.traffic.sample_day(cfg.date, np.random.default_rng(day_seed + 7))
        rain = 0.0 if cond.rain is None else cond.rain[2] * min(1.0, (cond.rain[1] - cond.rain[0]) / 43200)
        # demand: loyalty-weighted riders, total scaled by mean loyalty, rain adds up to 15%
        clat = self.clat.copy()
        clat["activity"] = clat.activity * clat.loyalty
        scale = float(self.clat.loyalty.mean()) * (1 + 0.15 * rain)
        rng = np.random.default_rng(day_seed)
        req = DM.generate(self.city, self.geo, date, self.cust, clat, rng, sf=self.sf, rain=rain, scale=scale)
        req = req[req.t_request < cfg.horizon_s].reset_index(drop=True)
        req["request_id"] = np.arange(len(req))
        # who works today: weekly days scaled by the weekday's popularity
        dlat = self.dlat.reset_index(drop=True).copy()
        dlat["days_per_week"] = np.clip(dlat.days_per_week * PP.DAY_WEIGHT[date.weekday()], 0, 7)
        drv_today = self.drv.reset_index(drop=True)
        dlat_today = self.dlat.reset_index(drop=True)
        world = HDWorld(self.city, self.data.net, self.data.nodes, self.data.zones, self.data.traffic,
                        drv_today, dlat, self.cust, self.clat, req,
                        self.data.gw_pos, self.policy_factory(), cfg, airport_zone=self.airport_zone)
        world.run(log=lambda *a: None)
        tb = world.tables()
        self._update_people(tb)
        s = self._day_summary(date, tb, world, rain, time.time() - t0)
        self.summary.append(s)
        if out_dir is not None:
            d = out_dir / str(date.date())
            d.mkdir(parents=True, exist_ok=True)
            for name, df in tb.items():
                df.to_parquet(d / f"{name}.parquet", index=False)
            drv_today.to_parquet(d / "drivers.parquet", index=False)          # as during the day
            dlat_today.to_parquet(d / "drivers_latent.parquet", index=False)
            self.clat[["customer_id", "loyalty"]].to_parquet(d / "customers_loyalty.parquet", index=False)
            (d / "summary.json").write_text(json.dumps(s, indent=2, default=float))
        self.log("  " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()
                                  if k in ("date", "requests", "declined", "completion", "wait_min", "mean_surge",
                                           "drivers_online", "net_per_hour", "gini", "quit", "joined", "seconds")))
        return tb

    # ------------------------------------------------------------ between days
    def _update_people(self, tb):
        rat = tb["ratings"]
        to_d = rat[rat.target_kind == "driver"].groupby("target_id").value.agg(["sum", "count"])
        idx = self.drv.driver_id.isin(to_d.index)
        self.drv.loc[idx, "rating_sum"] += self.drv.loc[idx, "driver_id"].map(to_d["sum"]).to_numpy()
        self.drv.loc[idx, "rating_count"] += self.drv.loc[idx, "driver_id"].map(to_d["count"]).to_numpy()
        self.drv["tenure_days"] += 1
        out = metrics.driver_outcomes(tb)
        pos = pd.Series(np.arange(len(self.drv)), index=self.drv.driver_id)
        quit_ids = []
        for did, hourly in zip(out.driver_id.astype(int), out.hourly):    # driver_id == position
            hist = self.history.setdefault(did, [])
            hist.append(float(hourly))
            del hist[:-7]
            res = float(self.dlat.reservation_hourly.iat[pos[did]])
            if len(hist) >= 3 and np.mean(hist[-3:]) < 0.6 * res and self.rng.random() < 0.05:
                quit_ids.append(did)
        self._quit = len(quit_ids)
        keep = ~self.drv.driver_id.isin(quit_ids)
        self.drv, self.dlat = self.drv[keep].reset_index(drop=True), self.dlat[keep].reset_index(drop=True)
        good = out.hourly.median() > PP.RESERVATION_HOURLY[self.city.key] if len(out) else False
        n_new = int(round(len(quit_ids) * self.rng.uniform(0.8, 1.2) + (0.005 * len(self.drv) if good else 0)))
        self._joined = n_new
        if n_new:
            start = int(max(self.drv.driver_id.max(), max(self.history, default=0)) + 1)
            d2, l2 = PP.make_drivers(self.city, n_new, self.geo.W.population.to_numpy(), self.geo.node_zone,
                                     self.rng, start_id=start)
            d2["tenure_days"] = 0
            d2["rating_sum"] = 0
            d2["rating_count"] = 0
            self.drv = pd.concat([self.drv, d2], ignore_index=True)
            self.dlat = pd.concat([self.dlat, l2], ignore_index=True)
        # the World indexes drivers by position: keep driver_id == position for the next day
        remap = dict(zip(self.drv.driver_id, range(len(self.drv))))
        self.history = {remap[k]: v for k, v in self.history.items() if k in remap}
        self.drv["driver_id"] = np.arange(len(self.drv))
        self.dlat["driver_id"] = np.arange(len(self.dlat))
        # riders: loyalty
        r = tb["requests"]
        r = r[r.status != "not_raised"]
        bad = r[(r.status.isin(["declined", "cancelled", "expired"])) | (r.wait_s > 600)].customer_id
        goodc = r[(r.status == "completed") & (r.wait_s <= 600)].customer_id
        loy = self.clat.loyalty.to_numpy()
        np.multiply.at(loy, bad.to_numpy(), 0.92)
        np.multiply.at(loy, goodc.to_numpy(), 1.03)
        self.clat["loyalty"] = np.clip(loy, 0.2, 1.0)

    def _day_summary(self, date, tb, world, rain, secs):
        r = tb["requests"]
        r = r[r.status != "not_raised"]
        served = r.t_pickup.notna()
        out = metrics.driver_outcomes(tb)
        fair = metrics.fairness(out) if len(out) else {}
        s = {"date": str(date.date()), "weekday": date.day_name()[:3], "rain": round(rain, 2),
             "requests": int(len(r)), "declined": float((r.status == "declined").mean()),
             "completion": float((r.status == "completed").mean()),
             "completion_of_booked": float((r.status == "completed").sum() / max((r.status != "declined").sum(), 1)),
             "wait_min": float(r.wait_s[served].mean() / 60) if served.any() else np.nan,
             "trip_min": float(((r.t_dropoff - r.t_pickup) / 60).mean()),
             "trip_km": float(r.distance_km.mean()), "mean_fare": float(r.fare.mean()),
             "mean_surge": float(r.surge.mean()), "surged_share": float((r.surge > 1.0).mean()),
             "drivers_online": int(len(out)), "drivers_pool": int(len(self.drv)),
             "net_per_hour": float(fair.get("mean_hourly", np.nan)), "gini": float(fair.get("gini_net_earnings", np.nan)),
             "quit": int(getattr(self, "_quit", 0)), "joined": int(getattr(self, "_joined", 0)),
             "mean_loyalty": float(self.clat.loyalty.mean()), "seconds": round(secs, 1)}
        if "mode" in r:
            for mname, g in r.groupby("mode"):
                s[f"share_{mname}"] = float(len(g) / len(r))
                s[f"completion_{mname}"] = float((g.status == "completed").mean())
        return s


class _CityData:
    """Graph, traffic and gateways for a high-detail city (like world.setup.CityData)."""

    def __init__(self, city, geo):
        d = city.data_dir
        self.city = city
        self.nodes, self.zones = geo.nodes, geo.zones
        self.edges = pd.read_parquet(d / "edges.parquet")
        self.net = T.Net.from_tables(self.nodes, self.edges)
        self.traffic = T.TrafficModel(city)
        self.gw_pos = {k: geo.gw_pos[min(e.gateway_rank, len(geo.gw_pos) - 1)] if e.gateway_lonlat is None
                       else _nearest_gw(geo, e.gateway_lonlat) for k, e in enumerate(city.market.external)}


def _nearest_gw(geo, lonlat):
    gw = pd.read_parquet(geo.city.data_dir / "traffic" / "gateways.parquet")
    j = int(np.argmin((gw.lon - lonlat[0]) ** 2 + (gw.lat - lonlat[1]) ** 2))
    return geo.gw_pos[j]
