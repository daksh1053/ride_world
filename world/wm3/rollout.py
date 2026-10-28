"""World model 3, stage 3: imagined rollouts (eq. 30) with the learned event model.

An ImaginedWorld starts from a real state (rebuilt from the logs at some epoch) and steps forward
30 s at a time, without the simulator:

  1. the platform policy (the same code the simulator runs, or any other) sees an Observation
     built from the imagined state and issues offers and repositioning orders;
  2. the state is turned into model inputs exactly as the training data was (same columns);
  3. the event model gives, per driver, offline driver, request and zone, the distribution of
     what happens in the next 30 s, and one outcome is sampled;
  4. the fixed rules F (eq. 13) apply the outcomes in time order: routes are planned on the road
     graph, cars move along them, arrivals, pickups, drop-offs, fares and ledger, expiry of
     unmatched requests, surge every 5 minutes.

What is learned (sampled from the model): offer answers, order answers, drivers' own decisions
(offline, break, head home, chase demand + which zone, end a break, come back into the city),
offline drivers coming online, riders cancelling and boarding, new requests (how many per zone
and mode, origin node, destination, whether the quote is declined). What is rule-based: the rest.
Future traffic is the typical profile (no rain or incidents are known in advance).
"""

from __future__ import annotations

import heapq
import json

import numpy as np
import pandas as pd
import torch

from .. import cities
from .. import traffic as T
from ..hd.demand import MODES as HD_MODES
from ..hd.demand import mode_fare
from ..hd.sim import DETOUR, SURGE_CAP, SURGE_RULE, SURGE_SMOOTH, HDDispatch, HDDispatchBlocks
from ..market import BEHAVIOUR
from ..policies import Observation
from ..routing import Leg, LiveRouter
from . import data as WD
from . import features as FT
from . import state as S
from .model import NEW

IDLE, OFFERED, PICKUP, OCC, REPO, OUTSIDE, BREAK, OFF = 1, 2, 3, 4, 5, 6, 7, 0
LK = {k: i for i, k in enumerate(S.LEG_KINDS)}


class CityRules:
    """Static things the rules need for one city (built once, shared by rollouts)."""

    def __init__(self, g: S.CityGraph):
        city = g.city
        self.g, self.city, self.key = g, city, city.key
        d = city.data_dir
        self.zones = pd.read_parquet(d / "zones.parquet")
        self.Z = len(self.zones)
        self.zone_tt = np.load(d / "traffic" / "zone_tt_s.npy")
        self.zone_dist = np.load(d / "traffic" / "zone_dist_m.npy")
        tt = self.zone_tt.copy()
        idx = np.arange(self.Z)
        tt[..., idx, idx] = np.inf
        self.intra = 0.5 * tt.min(-1) + 60.0
        self.zone_centre = pd.Series(np.arange(g.N), index=pd.read_parquet(d / "nodes.parquet").node_id).loc[
            self.zones.centre_node].to_numpy()
        self.modes = HD_MODES[self.key]
        self.mode_list = [self.modes.get(m) for m in S.MODES]            # None where the city lacks a mode
        self.speed = np.array([m.speed_factor if m else 1.0 for m in self.mode_list])
        near = (self.zone_dist < 6000) & ~np.eye(self.Z, dtype=bool)
        self.v = np.array([[np.median(self.zone_dist[a, h][near[a, h]] / self.zone_tt[a, h][near[a, h]]) for h in range(24)]
                           for a in range(self.zone_dist.shape[0])])
        w = pd.read_parquet(d / "hd" / "node_weights.parquet")
        air = np.flatnonzero(w.airport.to_numpy() > 0)
        self.airport_zone = int(np.bincount(g.zone[air]).argmax()) if len(air) else -1
        m = city.market
        self.market = m
        self.ext = m.external
        gws = pd.read_parquet(d / "traffic" / "gateways.parquet")
        self.gw_pos = self._gateways(gws)
        self.zone_nodes = [np.flatnonzero(g.zone == z) for z in range(self.Z)]

    def _gateways(self, gws):
        """External destination -> gateway node position (as world/hd/run.py)."""
        from ..hd.run import _nearest_gw
        from ..hd import demand as DM
        geo = DM.Geography(self.city)
        return {k: geo.gw_pos[min(e.gateway_rank, len(geo.gw_pos) - 1)] if e.gateway_lonlat is None
                else _nearest_gw(geo, e.gateway_lonlat) for k, e in enumerate(self.ext)}

    def quote(self, mode, oz, dz, ext, h, daytype):
        """Pre-surge quote (world/hd/sim.py _quotes)."""
        mo = self.mode_list[mode]
        if ext >= 0:
            e = self.ext[ext]
            gz = self.g.zone[self.gw_pos[ext]]
            km = self.zone_dist[daytype, h, oz, gz] / 1000
            mins = self.zone_tt[daytype, h, oz, gz] / 60 * mo.speed_factor
            return float(mode_fare(mo, km, mins)) + self.market.intercity_per_km * e.outside_km
        km = max(self.zone_dist[daytype, h, oz, dz], 800) / 1000
        mins = max(self.zone_tt[daytype, h, oz, dz], 120) / 60 * mo.speed_factor
        return float(mode_fare(mo, km, mins))

    def eta(self, a, b, t, mode, daytype):
        g = self.g
        d = np.hypot(g.x[a] - g.x[b], g.y[a] - g.y[b])
        return (30.0 + DETOUR * d / self.v[daytype, int(t // 3600) % 24]) * self.speed[mode]


def policy_for(L: S.DayLog, seed=0):
    """The policy the simulator ran that day, rebuilt from the action log."""
    a = L.dir / "actions.parquet"
    act = pd.read_parquet(a)
    info = json.loads(act["info"].iloc[min(5, len(act) - 1)] or "{}")
    p = {k: info[k] for k in ("reposition_every", "max_moves", "explore") if k in info}
    if act.policy.iloc[0] == "hd_blocks":
        return HDDispatchBlocks(n_max={"sf_hd": 60, "pune_hd": 120}[L.g.city.key], seed=seed, **p)
    return HDDispatch(seed=seed, **p)


_M1, _M2, _G = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB), np.uint64(0x9E3779B97F4A7C15)
KIND = {k: i + 1 for i, k in enumerate(["offer", "offer_t", "order", "order_t", "ev", "ev_t", "repo", "online", "online_t",
                                         "board", "cancel", "rq_t", "n_new", "origin", "dest", "dnode", "decline", "req_t"])}


def _mix(x):
    """splitmix64 finaliser on a uint64 array."""
    with np.errstate(over="ignore"):
        x = x + _G
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
        return x ^ (x >> np.uint64(31))


def keyed_uniform(seed, e, kind, ids):
    """Uniform numbers fixed by (seed, step, kind, entity): two imagined worlds that differ only in
    an action draw the same luck for the same driver or rider (common random numbers)."""
    ids = np.atleast_1d(np.asarray(ids, dtype=np.int64)).astype(np.uint64)
    x = _mix(np.full(ids.shape, seed, np.uint64))
    x = _mix(x ^ np.uint64(e))
    x = _mix(x ^ np.uint64(KIND[kind]))
    x = _mix(x ^ ids)
    return (x >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def pick(p, u):
    """Inverse-CDF choice: rows of probabilities p, one uniform per row."""
    p = np.atleast_2d(p)
    c = p.cumsum(1)
    return np.minimum((c < np.atleast_1d(u)[:, None] * c[:, -1:]).sum(1), p.shape[1] - 1)


def poisson_inv(lam, u):
    """Poisson draws by inversion (same u and nearly the same rate -> the same count)."""
    from scipy.stats import poisson
    return poisson.ppf(u, lam).astype(np.int64)


class ImaginedWorld:
    def __init__(self, model, rules: CityRules, L: S.DayLog, prior: FT.Prior, e0: int, policy, seed=0,
                 traffic="typical"):
        self.model, self.R, self.L, self.g = model, rules, L, rules.g
        self.key = rules.key
        self.city_id = WD.CITY_KEYS.index(self.key)
        self.rng = np.random.default_rng(seed)
        self.torch_gen = torch.Generator(device="cpu").manual_seed(seed)
        self.seed = seed
        self.order_p = {}
        self.policy = policy
        self.t = e0 * S.DT
        self.e = e0
        self.daytype = int(L.date.weekday() >= 5)
        self.date = L.date
        g = self.g
        self.cx, self.cy = float(np.median(g.x)), float(np.median(g.y))
        self.money = float(np.median(L.req.quote))
        self.rain0 = L.day.rain_at(self.t % 86400)
        # traffic: the typical day (no rain or incidents known in advance)
        tm = g.traffic
        self.day = T.DayConditions(L.date, "weekend" if self.daytype else "weekday", 1.0, None,
                                   pd.DataFrame(columns=["edge", "start_s", "end_s", "cap_factor"]))
        self.router = LiveRouter(g.net, tm)
        self.router.set_day(self.day)
        self.road_cache = {}
        nD = L.nD
        self.nD = nD
        p = L.drivers
        self.d_mode = p["mode"].map({m: i for i, m in enumerate(S.MODES)}).to_numpy()
        self.d_home = p.home_node.to_numpy()
        self.home_zone = p.home_zone.to_numpy()
        self.d_shift = p.usual_shift_start_h.to_numpy()
        self.d_static = np.stack([p.cost_per_km / np.median(p.cost_per_km), FT._log(p.tenure_days) / 7,
                                  np.where(p.rating_count > 0, p.rating_sum / p.rating_count.clip(lower=1) - 4.5, 0.0),
                                  FT._log(p.rating_count) / 7], 1)
        self.kappa = p.cost_per_km.to_numpy()
        self.prior_d = prior.features(np.arange(nD))
        # ---- driver state from the logs at e0
        e = e0
        self.st = L.status[:, e].astype(int).copy()
        self.node = L.stop[:, e].astype(np.int64).copy()
        self.status_since = L.status_since[:, e].astype(float).copy()
        self.online_since = L.online_since[:, e].astype(float).copy()
        self.idle_since = np.where(self.st == IDLE, self.status_since, self.t)
        self.count = {c: L.count[c][:, e].astype(float).copy() for c in S.COUNTERS}
        self.income = L.income[:, e].astype(float).copy()
        self.online_s = (L.status[:, :e + 1] != 0).sum(1) * S.DT
        self.want_off = np.zeros(nD, bool)
        self.going_home = np.zeros(nD, bool)
        self.repo_zone = np.full(nD, -1)
        self.d_req = np.full(nD, -1)
        self.leg: list[Leg | None] = [None] * nD
        self.leg_kind = np.full(nD, -1)
        self.leg_req = np.full(nD, -1)
        self.leg_cum = [None] * nD
        self.out_drop = {}                      # driver -> (time, request) of an outside drop-off
        legs = L.legs.set_index("leg_id")
        for d in np.flatnonzero(L.leg[:, e] >= 0):
            r = legs.loc[int(L.leg[d, e])]
            self._set_leg(d, Leg(np.asarray(r.nodes, np.int64), np.asarray(r.times, np.float64), float(r.length_m)),
                          LK[r.kind], int(r.request_id))
            if r.kind in ("reposition", "self_reposition", "go_home"):
                self.repo_zone[d] = g.zone[int(r.nodes[-1])]
                self.going_home[d] = r.kind == "go_home"
        # drivers outside the city: their drop-off may still be pending (unknown: assume done)
        # ---- requests open at e0
        self.req = {}
        R0 = L.open_requests(e)
        st_map = {0: "pending", 1: "offered", 2: "accepted", 3: "waiting"}
        for r in R0.itertuples():
            k = int(r.request_id)
            self.req[k] = {"origin": int(L.r_origin[k]), "dest": int(L.r_dest[k]), "ext": int(L.r_external[k]),
                           "mode": int(L.r_mode[k]), "created": float(L.r_created[k]), "quote": float(L.r_quote[k]),
                           "surge": float(L.req.set_index("request_id").surge.get(k, 1.0)), "status": st_map[r.status],
                           "n_offers": int(r.n_offers), "excluded": set(), "driver": -1, "arrive": float(np.nan_to_num(L.r_arrive[k], nan=np.nan))}
        # requests already in a car (their drop-off pays the fare) and intercity trips outside the city
        rq = L.req.set_index("request_id")
        def add_trip(k, d):
            r = rq.loc[k]
            self.req[k] = {"origin": int(r.origin_node), "dest": int(r.dest_node), "ext": int(r.external_dest),
                           "mode": S.MODES.index(r["mode"]), "created": float(r.t_request), "quote": float(r.quote),
                           "surge": float(r.surge), "status": "in_trip", "n_offers": int(r.n_offers), "excluded": set(),
                           "driver": int(d), "arrive": float(r.t_arrive), "pickup_t": float(r.t_pickup)}
        for d in np.flatnonzero(np.isin(self.leg_kind, [LK["trip"], LK["trip_to_gateway"]])):
            add_trip(int(self.leg_req[d]), d)
        outside = rq[(rq.t_exit <= self.t) & ~(rq.t_dropoff <= self.t) & (rq.driver_id >= 0)]
        for k, r in outside.iterrows():
            d = int(r.driver_id)
            if self.st[d] != OUTSIDE:
                continue
            add_trip(int(k), d)
            e_ = rules.ext[int(r.external_dest)]
            leg_km = rq.loc[k].distance_km - e_.outside_km
            self.req[int(k)]["km_in"] = float(max(leg_km, 0.0))
            out_t = r.t_exit + e_.outside_min * 60 * self.rng.uniform(0.85, 1.3)
            self.out_drop[d] = (max(out_t, self.t + 1.0), int(k))
        for d in range(nD):
            k = self.leg_req[d]
            if k in self.req:
                self.req[k]["driver"] = d
                self.d_req[d] = k
        for k, q in self.req.items():                 # a driver waiting at the pickup has no leg
            if q["status"] == "waiting" and q["driver"] < 0:
                cand = np.flatnonzero((self.st == PICKUP) & (self.leg_kind < 0) & (self.node == q["origin"]))
                if len(cand):
                    q["driver"] = int(cand[0]); self.d_req[cand[0]] = k
        self.next_req = int(L.req.request_id.max()) + 1 + 10 ** 6
        # request history for zone features and the policy (all requests incl. declined quotes)
        q = L.req[L.req.t_request <= self.t]
        self.hist = [(float(t), int(z), int(m), not dec) for t, z, m, dec in
                     zip(q.t_request, q.origin_zone, q["mode"].map({m: i for i, m in enumerate(S.MODES)}), q.status == "declined")]
        # surge as of the last 5-minute update
        sg = L.surge_tab if getattr(L, "surge_tab", None) is not None else pd.read_parquet(L.dir / "surge.parquet")
        self.surge = np.ones((self.R.Z, 3))
        last = sg[(sg.t // 300).astype(int) == int(self.t // 300)]          # the update of this 5-min slot
        for r in last.itertuples():
            self.surge[r.zone, S.MODES.index(r.mode)] = r.multiplier
        self.offers_now, self.orders_now = [], []
        self.log = []                                 # (t, type, driver, request, zone, value)
        self.kpi = []

    def _net(self):
        return self.g.net

    # ------------------------------------------------------------------ legs
    def _set_leg(self, d, leg, kind, req=-1):
        g = self.g
        self.leg[d], self.leg_kind[d], self.leg_req[d] = leg, kind, req
        seg = g.length[np.maximum(g.edge_of(leg.nodes[:-1], leg.nodes[1:]), 0)]
        self.leg_cum[d] = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]

    def _start_leg(self, d, target, t, kind, req=-1):
        leg = self.router.route(int(self.node[d]), int(target), t)
        f = self.R.speed[self.d_mode[d]]
        if f != 1.0:
            leg.times = leg.times[0] + (leg.times - leg.times[0]) * f
        self._set_leg(d, leg, LK[kind], req)
        return leg

    def _end_leg(self, d, t, truncate=False):
        leg = self.leg[d]
        if leg is None:
            return
        if truncate and t < leg.end:
            leg = leg.truncate(t)
        self.node[d] = int(leg.nodes[-1])
        self.leg[d], self.leg_kind[d], self.leg_req[d], self.leg_cum[d] = None, -1, -1, None

    def _status(self, d, s, t):
        self.st[d] = s
        self.status_since[d] = t

    def _idle(self, d, t):
        self.d_req[d] = -1
        if self.want_off[d] or self.going_home[d]:
            self._offline(d, t)
            return
        self.idle_since[d] = t
        self._status(d, IDLE, t)

    def _offline(self, d, t):
        if self.leg[d] is not None:
            self._end_leg(d, t, truncate=True)
        self._status(d, OFF, t)
        self.want_off[d] = self.going_home[d] = False
        self.log.append((t, "driver_offline", d, -1, -1, 0.0))

    # ------------------------------------------------------------------ positions
    def positions(self, t):
        """(nD, 2) metres, edge (u, v, frac), metres left and minutes left on the leg."""
        g = self.g
        x, y = g.x[self.node].copy(), g.y[self.node].copy()
        u, v = self.node.copy(), np.full(self.nD, -1)
        frac, left, eta = np.zeros(self.nD), np.zeros(self.nD), np.zeros(self.nD)
        for d in np.flatnonzero(self.leg_kind >= 0):
            leg = self.leg[d]
            tm, nd = leg.times, leg.nodes
            if len(nd) < 2:
                continue
            j = int(np.clip(np.searchsorted(tm, t, "right") - 1, 0, len(nd) - 2))
            f = float(np.clip((t - tm[j]) / max(tm[j + 1] - tm[j], 1e-6), 0, 1))
            a, b = nd[j], nd[j + 1]
            x[d] = g.x[a] + f * (g.x[b] - g.x[a]); y[d] = g.y[a] + f * (g.y[b] - g.y[a])
            u[d], v[d], frac[d] = a, b, f
            cum = self.leg_cum[d]
            left[d] = cum[j + 1] + (1 - f) * (cum[j] - cum[j + 1])
            eta[d] = (tm[-1] - t) / 60
        return np.c_[x, y], u, v, frac, left, eta

    # ------------------------------------------------------------------ roads
    def _road(self):
        slot = int(self.t // 300)
        if slot not in self.road_cache:
            r = self.g.traffic.edge_times(self.day, (slot + 0.5) * 300 % 86400) / self.g.t0
            w = self.g.length
            ez = self.g.zone[self.g.eu]
            num = np.bincount(ez, weights=r * w, minlength=self.R.Z)
            den = np.bincount(ez, weights=w, minlength=self.R.Z)
            self.road_cache = {slot: (num / np.maximum(den, 1), float((r * w).sum() / w.sum()))}
        return self.road_cache[slot]

    # ------------------------------------------------------------------ policy
    def observe(self):
        t, g, R = self.t, self.g, self.R
        h = int(t // 3600) % 24
        idle_ix = np.flatnonzero(self.st == IDLE)
        idle = pd.DataFrame({"driver_id": idle_ix, "node": self.node[idle_ix], "zone": g.zone[self.node[idle_ix]],
                             "idle_since": self.idle_since[idle_ix], "mode": self.d_mode[idle_ix]})
        pk = [k for k, q in self.req.items() if q["status"] == "pending"]
        rq = [self.req[k] for k in pk]
        ext_zone = lambda q: g.zone[R.gw_pos[q["ext"]]] if q["ext"] >= 0 else g.zone[q["dest"]]  # noqa: E731
        pending = pd.DataFrame({"request_id": pk, "origin_node": [q["origin"] for q in rq],
                                "origin_zone": [int(g.zone[q["origin"]]) for q in rq],
                                "dest_zone": [int(ext_zone(q)) for q in rq], "external": [q["ext"] >= 0 for q in rq],
                                "t_request": [q["created"] for q in rq], "quote": [q["quote"] for q in rq],
                                "excluded": [q["excluded"] for q in rq], "mode": [q["mode"] for q in rq]},
                               columns=["request_id", "origin_node", "origin_zone", "dest_zone", "external", "t_request",
                                        "quote", "excluded", "mode"])
        recent = np.bincount([z for tt, z, m, ok in self.hist if ok and tt >= t - 1800], minlength=R.Z).astype(float)
        repo_to = np.bincount(self.repo_zone[(self.st == REPO) & (self.repo_zone >= 0)], minlength=R.Z).astype(float)
        fleet = {"status": self.st, "zone": g.zone[self.node], "income": self.income, "airport_zone": R.airport_zone,
                 "surge": self.surge, "mode_speed": R.speed, "epoch": int(round(t / S.DT)), "epoch_s": S.DT,
                 "day_type": self.daytype, "zone_dist": R.zone_dist[self.daytype, h], "cost_per_km": self.kappa,
                 "eta": lambda a, b, j, t=t: R.eta(a, b, t, j, self.daytype)}
        return Observation(t=t, hour=h, idle=idle, pending=pending, zone_tt=R.zone_tt[self.daytype, h],
                           recent_demand=recent, recent_window_s=1800.0, intra_zone_s=R.intra[self.daytype, h],
                           max_pickup_s=60 * R.market.max_pickup_min, repositioning_to=repo_to, fleet=fleet)

    def apply_action(self, act):
        """Offers and orders issued at tau (feasibility as in world/sim.py _apply)."""
        t, g, R = self.t, self.g, self.R
        h = int(t // 3600) % 24
        seen_d, seen_r = set(), set()
        self.offers_now, self.orders_now = [], []
        for i, k in act.offers:
            q = self.req.get(k)
            if (q is None or i in seen_d or k in seen_r or self.st[i] != IDLE or q["status"] != "pending" or i in q["excluded"]
                    or self.d_mode[i] != q["mode"]):          # world/hd/sim.py drops cross-mode offers
                continue
            seen_d.add(i); seen_r.add(k)
            # the GPS-style ETA the driver is shown (world/hd/sim.py _apply), not the zone table
            eta = float(R.eta(self.node[i], q["origin"], t, q["mode"], self.daytype))
            self._status(i, OFFERED, t)
            self.d_req[i] = k
            q["status"] = "offered"
            q["n_offers"] += 1
            self.count["offers"][i] += 1
            self.offers_now.append((i, k, float(eta)))
            self.log.append((t, "offer_sent", i, k, -1, float(eta)))
        for i, z in act.dispatches:
            if i in seen_d or self.st[i] != IDLE or not (0 <= z < R.Z):
                continue
            seen_d.add(i)
            self._status(i, REPO, t)
            self.repo_zone[i] = z
            self.count["orders"][i] += 1
            self.orders_now.append((i, int(z)))
            self.log.append((t, "dispatch_sent", i, -1, int(z), 0.0))

    # ------------------------------------------------------------------ features (as features.py)
    def features(self):
        t, g, R = self.t, self.g, self.R
        h = t / 3600
        rz, cong = self._road()
        dow = np.eye(7)[self.date.weekday()]
        G = np.r_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(4 * np.pi * h / 24),
                  np.cos(4 * np.pi * h / 24), dow, self.rain0, cong - 1, self.city_id].astype(np.float32)
        pos, u, v, frac, left, eta_left = self.positions(t)
        on = np.flatnonzero(self.st != OFF)
        off = np.flatnonzero(self.st == OFF)
        xy = pos[on]
        lk = self.leg_kind[on]
        mv = lk >= 0
        dest = np.array([int(self.leg[d].nodes[-1]) if self.leg[d] is not None else self.node[d] for d in on], np.int64)
        rel = lambda nodes: np.c_[(g.x[nodes] - xy[:, 0]) / FT.SCALE, (g.y[nodes] - xy[:, 1]) / FT.SCALE]  # noqa: E731
        cnt = {c: self.count[c][on] for c in S.COUNTERS}
        in_st = t - self.status_since[on]
        online_h = np.nan_to_num((t - self.online_since[on]) / 3600)
        req_of = self.leg_req[on]
        has_req = req_of >= 0
        req_wait = np.array([(t - self.req[k]["created"]) / 60 if (k in self.req and lk[j] == 0) else 0.0
                             for j, k in enumerate(req_of)])
        home = rel(self.d_home[on])
        sh = (h - self.d_shift[on] + 12) % 24 - 12
        dz = g.zone[self.node[on]]
        mode = self.d_mode[on]
        pos_on = {d: j for j, d in enumerate(on)}
        off_feat = np.zeros((len(on), 8))
        for i, k, eta in self.offers_now:
            q = self.req[k]
            o = q["origin"]
            dn = q["dest"] if q["dest"] >= 0 else o
            km = np.hypot(g.x[dn] - g.x[o], g.y[dn] - g.y[o]) / 1000 if q["dest"] >= 0 else 0.0
            off_feat[pos_on[i]] = [1, eta / 60, km, q["quote"] / self.money, (t - q["created"]) / 60,
                                   (g.x[dn] - g.x[o]) / FT.SCALE, (g.y[dn] - g.y[o]) / FT.SCALE, q["surge"]]
        ord_feat = np.zeros((len(on), 5))
        hh = int(h) % 24
        for i, z in self.orders_now:
            j = pos_on[i]
            ord_feat[j] = [1, (R.zones.x_m.iat[z] - xy[j, 0]) / FT.SCALE, (R.zones.y_m.iat[z] - xy[j, 1]) / FT.SCALE,
                           np.hypot(R.zones.x_m.iat[z] - xy[j, 0], R.zones.y_m.iat[z] - xy[j, 1]) / 1000,
                           R.zone_tt[self.daytype, hh, dz[j], z] / 60]
        D = np.column_stack([
            np.eye(8)[self.st[on]], (xy[:, 0] - self.cx) / FT.SCALE, (xy[:, 1] - self.cy) / FT.SCALE, mv,
            (lk[:, None] == np.arange(len(S.LEG_KINDS))[None]), rel(dest), left[on] / 1000, eta_left[on],
            FT._log(in_st / 60), online_h, *[FT._log(cnt[c]) for c in S.COUNTERS],
            self.income[on] / self.money / 10, (cnt["offers"] - cnt["offers_rejected"] - cnt["offers_timeout"] + 1) / (cnt["offers"] + 1.25),
            (cnt["orders"] - cnt["orders_declined"] + 1) / (cnt["orders"] + 1.35),
            self.prior_d[on], np.eye(3)[mode], self.d_static[on], home, sh, has_req, req_wait,
            off_feat, ord_feat, self.surge[dz, mode]]).astype(np.float32)
        ever = ~np.isnan(self.online_since[off])
        hrs = np.where(ever, (t - self.status_since[off]) / 3600, h) / 24
        sh_f = (h - self.d_shift[off] + 12) % 24 - 12
        Fm = np.column_stack([~ever, hrs, self.online_s[off] / 3600 / 8, FT._log(self.count["trips"][off]), sh_f / 12,
                              (g.x[self.d_home[off]] - self.cx) / FT.SCALE, (g.y[self.d_home[off]] - self.cy) / FT.SCALE,
                              np.eye(3)[self.d_mode[off]], self.d_static[off, 1], self.prior_d[off]]).astype(np.float32)
        # open requests
        rs = {"pending": 0, "offered": 1, "accepted": 2, "waiting": 3}
        ks = [k for k, q in self.req.items() if q["status"] in rs]          # riders already in a car are not open
        drv_eta = {self.leg_req[d]: eta_left[d] for d in np.flatnonzero((self.leg_kind == 0) & (self.leg_req >= 0))}
        rows = []
        for k in ks:
            q = self.req[k]
            o = q["origin"]
            dn = q["dest"] if q["dest"] >= 0 else o
            km = np.hypot(g.x[dn] - g.x[o], g.y[dn] - g.y[o]) / 1000 if q["dest"] >= 0 else 0.0
            rows.append(np.r_[np.eye(4)[rs[q["status"]]], (g.x[o] - self.cx) / FT.SCALE, (g.y[o] - self.cy) / FT.SCALE,
                              (g.x[dn] - g.x[o]) / FT.SCALE, (g.y[dn] - g.y[o]) / FT.SCALE, q["dest"] < 0, km,
                              np.eye(3)[q["mode"]], (t - q["created"]) / 60, q["n_offers"], q["quote"] / self.money, q["surge"],
                              drv_eta.get(k, 0.0), (t - q["arrive"]) / 60 if q["status"] == "waiting" else 0.0])
        Rm = np.array(rows, np.float32).reshape(-1, len(FT.R_COLS))
        # zones
        Z = R.Z
        H = np.array(self.hist) if self.hist else np.zeros((0, 4))
        ht, hz, hm = H[:, 0], H[:, 1].astype(int), H[:, 2].astype(int)
        win = lambda m: np.bincount(hz[ht > t - m * 60], minlength=Z)  # noqa: E731
        r15 = np.stack([np.bincount(hz[(ht > t - 900) & (hm == j)], minlength=Z) for j in range(3)], 1)
        pend = np.bincount([g.zone[self.req[k]["origin"]] for k in ks if self.req[k]["status"] in ("pending", "offered")], minlength=Z)
        stop_z = g.zone[self.node]
        idle = np.stack([np.bincount(stop_z[(self.st == IDLE) & (self.d_mode == j)], minlength=Z) for j in range(3)], 1)
        busy = np.bincount(stop_z[np.isin(self.st, [2, 3, 4, 5])], minlength=Z)
        rp = (self.leg_kind == 3) | (self.leg_kind == 4)
        repo = np.bincount(g.zone[[int(self.leg[d].nodes[-1]) for d in np.flatnonzero(rp)]] if rp.any() else np.zeros(0, int), minlength=Z)
        zs = FT._log(R.zones[FT.Z_STATIC].to_numpy(float))
        Zm = np.column_stack([(R.zones.x_m - self.cx) / FT.SCALE, (R.zones.y_m - self.cy) / FT.SCALE, zs,
                              FT._log(win(5)), FT._log(win(15)), FT._log(win(60)), FT._log(r15), FT._log(pend), FT._log(idle),
                              FT._log(busy), FT._log(repo), self.surge, rz - 1]).astype(np.float32)
        return {"G": G, "D": D, "D_id": on, "F": Fm, "F_id": off, "R": Rm, "R_id": np.array(ks, np.int64), "Z": Zm}

    # ------------------------------------------------------------------ one step
    @torch.no_grad()
    def step(self, act=None):
        """Advance 30 s. `act`: an Action to use instead of asking the policy."""
        if act is None:
            act = self.policy.act(self.observe())
        self.apply_action(act)
        return self.advance()

    @torch.no_grad()
    def advance(self):
        """Predict and apply the events of the next 30 s, with this step's actions already applied."""
        f = self.features()
        dev = next(self.model.parameters()).device
        snap = {"key": self.key, **{k: f[k] for k in ("G", "D", "F", "R", "Z")},
                "D_ev": np.zeros(len(f["D"]), np.int8), "D_repo": np.full(len(f["D"]), -1), "D_offer": np.full(len(f["D"]), -1),
                "D_order": np.full(len(f["D"]), -1), "F_on": np.zeros(len(f["F"]), np.int8), "R_cancel": np.zeros(len(f["R"]), np.int8),
                "R_board": np.zeros(len(f["R"]), np.int8), "Z_new": np.zeros((len(f["Z"]), 3), np.int16),
                "NEW": np.zeros((0, len(FT.NEW_COLS)), np.float32)}
        b = WD.collate([snap], dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
            out = self.model(b)
        P = {k: out[k][0].float().cpu() for k in ("ev", "offer", "order", "repo", "online", "cancel", "board", "rate")}
        hZ = out["h"][2][0].float()
        self.hZd = out.get("hZd", out["h"][2])[0].float()                # demand heads' zone states
        self._sample_and_apply(f, P, hZ)
        return f

    def _sample_and_apply(self, f, P, hZ):
        t0, t1 = self.t, self.t + S.DT
        rng, g, R = self.rng, self.g, self.R
        on, off = f["D_id"], f["F_id"]
        ev = []                                               # (time, seq, kind, args)

        def push(t, kind, *a):
            heapq.heappush(ev, (t, len(ev), kind, a))

        ku = lambda kind, ids: keyed_uniform(self.seed, self.e, kind, ids)    # noqa: E731
        pos_on = {d: j for j, d in enumerate(on)}
        pre = getattr(self, "preaccept", None) or ()
        for i, k, eta in self.offers_now:
            p = P["offer"][pos_on[i]].softmax(-1).numpy()
            a = int(pick(p, ku("offer", [i]))[0])
            if (i, k) in pre:                         # a ride the driver chose from its menu (stage 7)
                a = 0
            push(t0 + (15.0 if a == 2 else 4 + 11 * ku("offer_t", [i])[0]), "offer_answer", i, k, a)
        for i, z in self.orders_now:
            p = P["order"][pos_on[i]].softmax(-1).numpy()
            self.order_p.setdefault(i, float(p[0]))                       # chance to follow the first order, for the record
            a = int(pick(p, ku("order", [i]))[0])
            push(t0 + 2 + 8 * ku("order_t", [i])[0], "order_answer", i, z, a)
        pe = P["ev"].softmax(-1).numpy()
        cls = pick(pe, ku("ev", on)) if len(on) else np.zeros(0, int)
        ut = ku("ev_t", on)
        for j in np.flatnonzero(cls > 0):
            d = int(on[j])
            c = int(min(cls[j], len(FT.EV) - 1))
            z = -1
            if c == 4:
                pz = P["repo"][j].softmax(-1).numpy()
                z = int(pick(pz, ku("repo", [d]))[0])
            push(t0 + S.DT * ut[j], "driver_event", d, c, z)
        po = torch.sigmoid(P["online"]).numpy()
        uo, uot = ku("online", off), ku("online_t", off)
        for j in np.flatnonzero(uo < po):
            push(t0 + S.DT * uot[j], "online", int(off[j]))
        ks = f["R_id"]
        pc = torch.sigmoid(P["cancel"]).numpy()
        pb = torch.sigmoid(P["board"]).numpy()
        if len(ks):
            ub, uc, urt = ku("board", ks), ku("cancel", ks), ku("rq_t", ks)
        for j, k in enumerate(ks):
            q = self.req[int(k)]
            if q["status"] == "waiting" and ub[j] < pb[j]:
                push(t0 + S.DT * urt[j], "board", int(k))
            elif uc[j] < pc[j]:
                push(t0 + S.DT * urt[j], "cancel", int(k))
        # new requests: counts per (zone, mode) by inversion
        lam = P["rate"].clamp(-12, 6).exp().numpy()                    # (Z, 3)
        n_new = poisson_inv(lam, ku("n_new", np.arange(lam.size)).reshape(lam.shape))
        for m in range(3):
            if R.mode_list[m] is None:
                n_new[:, m] = 0
        zs, ms = np.nonzero(n_new)
        zone = np.repeat(zs, n_new[zs, ms]); mode = np.repeat(ms, n_new[zs, ms])
        if len(zone):
            self._new_requests(zone, mode, hZ, push)
        # legs that end within the step
        for d in np.flatnonzero(self.leg_kind >= 0):
            if self.leg[d].end <= t1:
                push(self.leg[d].end, "arrive", int(d), self.leg[d])
        for d, (tt, k) in list(self.out_drop.items()):
            if tt <= t1:
                push(tt, "outside_dropoff", d, k)
        while ev:
            t, _, kind, a = heapq.heappop(ev)
            getattr(self, "_on_" + kind)(t, *a, push=push, t1=t1)
        # platform rules at the end of the step
        for k in [k for k, q in self.req.items() if q["status"] in ("pending", "offered") and t1 - q["created"] >= BEHAVIOUR.platform_max_wait_s]:
            d = self._driver_of(k)
            self.req.pop(k)
            self.log.append((t1, "request_expired", -1, k, -1, 0.0))
            if d >= 0 and self.st[d] == OFFERED:
                self._idle(d, t1)
        self.t, self.e = t1, self.e + 1
        self.online_s += S.DT * (self.st != OFF)          # time online today, as features.py counts it
        if int(t1) % 300 == 0:
            self._surge()
        self.offers_now, self.orders_now = [], []

    def _driver_of(self, k):
        q = self.req.get(k)
        return q["driver"] if q else -1

    def _new_requests(self, zone, mode, hZ, push):
        g, R = self.g, self.R
        idx = np.zeros(len(zone), np.int64)
        for i in range(1, len(zone)):
            idx[i] = idx[i - 1] + 1 if (zone[i], mode[i]) == (zone[i - 1], mode[i - 1]) else 0
        kid = 10 ** 9 + self.e * 10 ** 6 + (zone * 3 + mode) * 1000 + idx        # same request, same id in every arm
        ku = lambda kind: keyed_uniform(self.seed, self.e, kind, kid)           # noqa: E731
        zt = torch.tensor(zone, device=hZ.device)
        mt = torch.tensor(mode, device=hZ.device)
        with torch.no_grad():
            hZd = self.hZd
            ol, cand, dl, _, _ = self.model.new_request_logits(self.key, hZd, zt, mt)
            po = ol.float().softmax(-1).cpu().numpy()
            o_idx = pick(po, ku("origin"))
            origin = cand.cpu().numpy()[np.arange(len(zone)), o_idx]
            ol, cand, dl, dnl, dn = self.model.new_request_logits(self.key, hZd, zt, mt, torch.tensor(origin, device=hZ.device),
                                                                  None)
            pd_ = dl.float().softmax(-1).cpu().numpy()
            dsel = pick(pd_, ku("dest"))
            ext = np.where(dsel >= R.Z, dsel - R.Z, -1)
            dz = np.where(dsel < R.Z, dsel, -1)
            _, _, _, dnl, dn = self.model.new_request_logits(self.key, hZd, zt, mt, torch.tensor(origin, device=hZ.device),
                                                             torch.tensor(np.maximum(dz, 0), device=hZ.device))
            pn = dnl.float().softmax(-1).cpu().numpy()
            dnode = dn.cpu().numpy()[np.arange(len(zone)), pick(pn, ku("dnode"))]
        dnode = np.where(ext >= 0, -1, dnode)
        h = int(self.t // 3600) % 24
        ut = ku("req_t")
        for i in range(len(zone)):
            push(self.t + S.DT * ut[i], "request", int(zone[i]), int(mode[i]), int(origin[i]), int(dz[i]), int(dnode[i]), int(ext[i]),
                 hZ, int(kid[i]))

    # ------------------------------------------------------------------ event handlers
    def _on_offer_answer(self, t, i, k, a, push, t1):
        q = self.req.get(k)
        key = ["offer_accepted", "offer_rejected", "offer_timeout"][a]
        self.log.append((t, key, i, k, -1, 0.0))
        if a == 1:
            self.count["offers_rejected"][i] += 1
        if a == 2:
            self.count["offers_timeout"][i] += 1
        if q is None or q["status"] != "offered":
            if self.st[i] == OFFERED:
                self._idle(i, t)
            return
        if a != 0:
            q["status"] = "pending"
            q["excluded"].add(i)
            self._idle(i, t)
            return
        q["status"], q["driver"] = "accepted", i
        self._status(i, PICKUP, t)
        self.d_req[i] = k
        leg = self._start_leg(i, q["origin"], t, "pickup", k)
        if leg.end <= t1:
            push(leg.end, "arrive", i, leg)

    def _on_order_answer(self, t, i, z, a, push, t1):
        if self.st[i] != REPO or self.leg[i] is not None:
            return
        self.log.append((t, "dispatch_declined" if a else "dispatch_accepted", i, -1, z, 0.0))
        if a == 1:
            self.count["orders_declined"][i] += 1
        if a == 1 or self.g.zone[self.node[i]] == z:
            if self.want_off[i]:
                self._offline(i, t)
            else:
                self._status(i, IDLE, t)
            return
        leg = self._start_leg(i, self.R.zone_centre[z], t, "reposition")
        if leg.end <= t1:
            push(leg.end, "arrive", i, leg)

    def _on_driver_event(self, t, d, c, z, push, t1):
        st = self.st[d]
        name = FT.EV[c]
        if name == "offline":
            if st in (IDLE, BREAK):
                self._offline(d, t)
            elif st != OFF:
                self.want_off[d] = True
        elif name == "break" and st == IDLE:
            self._status(d, BREAK, t); self.count["breaks"][d] += 1
            self.log.append((t, "driver_break", d, -1, -1, 0.0))
        elif name == "heads_home" and st == IDLE:
            self.going_home[d] = True
            self._status(d, REPO, t)
            self.repo_zone[d] = self.g.zone[self.d_home[d]]
            leg = self._start_leg(d, self.d_home[d], t, "go_home")
            self.log.append((t, "driver_heads_home", d, -1, -1, 0.0))
            if leg.end <= t1:
                push(leg.end, "arrive", d, leg)
        elif name == "self_reposition" and st == IDLE and z >= 0 and z != self.g.zone[self.node[d]]:
            self._status(d, REPO, t)
            self.repo_zone[d] = z
            leg = self._start_leg(d, self.R.zone_centre[z], t, "self_reposition")
            self.log.append((t, "driver_self_reposition", d, -1, z, 0.0))
            if leg.end <= t1:
                push(leg.end, "arrive", d, leg)
        elif name == "break_end" and st == BREAK:
            self.log.append((t, "driver_break_end", d, -1, -1, 0.0))
            self._idle(d, t)
        elif name == "reenter" and st == OUTSIDE and d not in self.out_drop:
            self.log.append((t, "reenter_city", d, -1, -1, 0.0))
            self._idle(d, t)
        elif name == "abandon_reposition" and st == REPO and self.leg_kind[d] in (3, 4):
            self.log.append((t, "reposition_abandoned", d, -1, -1, 0.0))
            self._offline(d, t)

    def _on_online(self, t, d, push, t1):
        if self.st[d] != OFF:
            return
        cand = self.R.zone_nodes[int(self.home_zone[d])]
        self.node[d] = int(self.rng.choice(cand))
        self.online_since[d] = t
        self.idle_since[d] = t
        self._status(d, IDLE, t)
        self.log.append((t, "driver_online", d, -1, -1, 0.0))

    def _on_cancel(self, t, k, push, t1):
        q = self.req.get(k)
        if q is None:
            return
        d = q["driver"]
        self.req.pop(k)
        self.log.append((t, "customer_cancelled", d, k, -1, 0.0))
        if d >= 0 and q["status"] in ("accepted", "waiting"):
            if self.leg[d] is not None:
                self._end_leg(d, t, truncate=True)
            self._idle(d, t)
        elif d >= 0 and q["status"] == "offered" and self.st[d] == OFFERED:
            pass                                          # the driver's answer comes back to nothing

    def _on_board(self, t, k, push, t1):
        q = self.req.get(k)
        if q is None or q["status"] != "waiting":
            return
        d = q["driver"]
        q["status"] = "in_trip"
        q["pickup_t"] = t
        self._status(d, OCC, t)
        self.log.append((t, "pickup", d, k, -1, 0.0))
        if q["ext"] >= 0:
            leg = self._start_leg(d, self.R.gw_pos[q["ext"]], t, "trip_to_gateway", k)
        else:
            leg = self._start_leg(d, q["dest"], t, "trip", k)
        if leg.end <= t1:
            push(leg.end, "arrive", d, leg)

    def _on_arrive(self, t, d, leg, push, t1):
        if self.leg[d] is not leg:
            return
        kind = S.LEG_KINDS[self.leg_kind[d]]
        k = self.leg_req[d]
        km = leg.length_m / 1000
        self._end_leg(d, t)
        self.income[d] -= 0.0                                    # operating costs are not in the income counter
        if kind == "pickup":
            q = self.req.get(k)
            if q is None:
                self._idle(d, t)
                return
            q["status"], q["arrive"] = "waiting", t
            q["pick_km"] = km
            self.log.append((t, "driver_arrived", d, k, -1, 0.0))
        elif kind == "trip":
            self._dropoff(t, d, k, km, 0.0)
            self._idle(d, t)
        elif kind == "trip_to_gateway":
            q = self.req.get(k)
            e = self.R.ext[q["ext"]]
            q["km_in"] = km
            self._status(d, OUTSIDE, t)
            self.log.append((t, "exit_city", d, k, -1, 0.0))
            self.out_drop[d] = (t + e.outside_min * 60 * self.rng.uniform(0.85, 1.3), k)
            if self.out_drop[d][0] <= t1:
                push(self.out_drop[d][0], "outside_dropoff", d, k)
        elif kind in ("reposition", "self_reposition"):
            self.log.append((t, "reposition_arrived", d, -1, -1, 0.0))
            self._idle(d, t)
        elif kind == "go_home":
            self.log.append((t, "driver_home", d, -1, -1, 0.0))
            self._offline(d, t)

    def _on_outside_dropoff(self, t, d, k, push, t1):
        if self.out_drop.get(d, (None, None))[1] != k:
            return
        self.out_drop.pop(d)
        q = self.req.get(k)
        if q is None:
            return
        self._dropoff(t, d, k, q.get("km_in", 0.0), self.R.ext[q["ext"]].outside_km)

    def _dropoff(self, t, d, k, km_in, outside_km):
        q = self.req.pop(k, None)
        if q is None:
            return
        mo = self.R.mode_list[q["mode"]]
        minutes = (t - q.get("pickup_t", t)) / 60
        fare = (float(mode_fare(mo, km_in, minutes)) + self.R.market.intercity_per_km * outside_km) * q["surge"]
        net = fare * (1 - self.R.market.commission)
        self.income[d] += net
        self.count["trips"][d] += 1
        self.log.append((t, "dropoff", d, k, -1, fare))
        self.log.append((t, "settlement", d, k, -1, net))

    def _on_request(self, t, z, m, origin, dz, dnode, ext, hZ, k, push, t1):
        g, R = self.g, self.R
        h = int(t // 3600) % 24
        dzone = int(g.zone[dnode]) if dnode >= 0 else int(g.zone[R.gw_pos[ext]])
        base = R.quote(m, z, dzone, ext, h, self.daytype)
        s = float(self.surge[z, m])
        idle = np.flatnonzero((self.st == IDLE) & (self.d_mode == m))
        eta = float(R.eta(self.node[idle], origin, t, m, self.daytype).min()) if len(idle) else 900.0
        km = np.hypot(g.x[dnode] - g.x[origin], g.y[dnode] - g.y[origin]) / 1000 if dnode >= 0 else -1.0
        feats = torch.tensor([[base / self.money, s - 1, eta / 60 / 10, max(km, 0) / 10, float(ext >= 0), 1.0]],
                             dtype=torch.float32, device=hZ.device)
        with torch.no_grad():
            pdcl = torch.sigmoid(self.model.decline_logit(hZ, torch.tensor([z], device=hZ.device),
                                                          torch.tensor([m], device=hZ.device), feats)).item()
        declined = keyed_uniform(self.seed, self.e, "decline", [k])[0] < pdcl
        self.hist.append((t, z, m, not declined))
        if declined:
            self.log.append((t, "quote_declined", -1, k, z, base * s))
            return
        self.req[k] = {"origin": origin, "dest": dnode, "ext": ext, "mode": m, "created": t, "quote": base * s, "surge": s,
                       "status": "pending", "n_offers": 0, "excluded": set(), "driver": -1, "arrive": np.nan}
        self.log.append((t, "request_created", -1, k, z, base * s))

    # ------------------------------------------------------------------ surge rule
    def _surge(self):
        t, g, R = self.t, self.g, self.R
        H = np.array(self.hist) if self.hist else np.zeros((0, 4))
        rec = (H[:, 0] >= t - 900) & (H[:, 0] < t) & (H[:, 3] > 0)
        cap = SURGE_CAP[self.key]
        gain, thr = SURGE_RULE[self.key]
        wait = [(g.zone[q["origin"]], q["mode"]) for q in self.req.values() if q["status"] in ("pending", "offered")]
        stop_z = g.zone[self.node]
        for j in range(3):
            dem = np.bincount(H[rec & (H[:, 2] == j), 1].astype(int), minlength=R.Z) * (300 / 900) \
                + np.bincount([z for z, m in wait if m == j], minlength=R.Z)
            sup = np.bincount(stop_z[(self.st == IDLE) & (self.d_mode == j)], minlength=R.Z)
            raw = np.clip(1.0 + gain * (dem / (sup + 1.0) - thr), 1.0, cap)
            self.surge[:, j] = SURGE_SMOOTH * self.surge[:, j] + (1 - SURGE_SMOOTH) * raw
        self.surge = np.round(self.surge * 10) / 10

    # ------------------------------------------------------------------ summaries
    def kpis(self):
        st = self.st
        open_ = [q["status"] for q in self.req.values()]
        return {"t": self.t, "online": int((st != OFF).sum()), "idle": int((st == IDLE).sum()),
                "to_pickup": int(((st == PICKUP) | (st == OFFERED)).sum()), "with_rider": int(((st == OCC) | (st == OUTSIDE)).sum()),
                "repositioning": int((st == REPO).sum()), "break": int((st == BREAK).sum()),
                "waiting_riders": int(sum(s in ("pending", "offered") for s in open_))}
