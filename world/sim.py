"""The ride-service world: an event-driven simulation of one day of a ride company.

State, actions and events follow `formulations/01_all_inputs_formulation.md`:

* Driver status in {offline, idle, offered, pickup, occupied, repositioning}, plus
  `outside` (occupied, beyond the city on an intercity trip or driving back).
* Request status in {pending, offered, accepted, pickup, in_trip, completed,
  cancelled, expired}.
* A_t = (offers, dispatches) comes from a `Policy` every `epoch_s` seconds (eq. 6-8).
* Everything else is a timed world event (eq. 10): request arrival, driver
  online/offline, offer response, reposition response, movement legs, arrival,
  pickup, drop-off, city exit/re-entry, cancellation, expiry, settlement, tip,
  rating, rain and incidents.
* Each event carries `t_event` and `t_available`, the time the platform learns of it
  (eq. 5), and a `recorded` flag: the observation model drops a small share of
  movement events, as real telemetry does.
* Ledger entries implement eq. 15-18: fare, commission, tip, cancellation fee,
  booking fee, operating cost (kappa_i x distance) and online working time.

Output tables mirror the database relations of `02_database_formulation.md`:
Driver, Customer, Request, Trip, Action, Offer, Event, Ledger, Rating, Context, plus
the driven legs (paths) and the driver status timeline used by the replay viewer.
"""

from __future__ import annotations

import heapq
import json
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cities import City
from .market import BEHAVIOUR, fare
from .policies import Observation, Policy
from .routing import Leg, LiveRouter
from .traffic import DAYTYPES, Net, TrafficModel

D_STATUS = ["offline", "idle", "offered", "pickup", "occupied", "repositioning", "outside"]
R_STATUS = ["pending", "offered", "accepted", "pickup", "in_trip", "completed", "cancelled", "expired"]
OFF, IDLE, OFFERED, PICKUP, OCCUPIED, REPO, OUTSIDE = range(7)
PENDING, R_OFFERED, ACCEPTED, R_PICKUP, IN_TRIP, COMPLETED, CANCELLED, EXPIRED = range(8)

# ASSUMPTION: share of movement telemetry events that never reach the platform.
DROP_PROB = {"driver_arrived": 0.02, "leg_start": 0.01}


@dataclass
class SimConfig:
    date: str = "2025-03-04"          # a Tuesday; traffic & rain are drawn for this date
    seed: int = 0
    epoch_s: float = 30.0
    horizon_s: float = 86400.0        # requests are generated for [0, 24h)
    drain_s: float = 3600.0           # keep running after midnight to finish trips
    context_every_s: float = 300.0
    recent_window_s: float = 1800.0


class World:
    def __init__(self, city: City, net: Net, nodes: pd.DataFrame, zones: pd.DataFrame,
                 traffic: TrafficModel, drivers: pd.DataFrame, drivers_latent: pd.DataFrame,
                 customers: pd.DataFrame, customers_latent: pd.DataFrame, requests: pd.DataFrame,
                 gateways_pos: dict, policy: Policy, cfg: SimConfig):
        self.city, self.net, self.cfg, self.policy = city, net, cfg, policy
        self.m, self.b = city.market, BEHAVIOUR
        self.rng = np.random.default_rng(cfg.seed + 101)
        self.node_zone = nodes.zone.to_numpy()
        self.zone_centre = pd.Series(np.arange(len(nodes)), index=nodes.node_id).loc[zones.centre_node].to_numpy()
        self.Z = len(zones)
        tdir = city.data_dir / "traffic"
        self.zone_tt = np.load(tdir / "zone_tt_s.npy")
        self.zone_dist = np.load(tdir / "zone_dist_m.npy")
        # Typical pickup time inside a zone, per day type and hour: half the time to the
        # nearest other zone centre, plus a minute of local driving.
        tt = self.zone_tt.copy()
        idx = np.arange(tt.shape[-1])
        tt[..., idx, idx] = np.inf
        self.intra = 0.5 * tt.min(-1) + 60.0          # (2, 24, Z)
        self.traffic = traffic
        self.day = traffic.sample_day(cfg.date, np.random.default_rng(cfg.seed + 7))
        self.daytype = DAYTYPES.index(self.day.daytype)
        self.router = LiveRouter(net, traffic)
        self.router.set_day(self.day)
        self.gw_pos = gateways_pos      # external dest index -> gateway node position
        self.gw_zone = np.array([self.node_zone[gateways_pos[k]] for k in range(len(gateways_pos))])

        self.drv, self.dlat = drivers, drivers_latent
        self.cust, self.clat = customers, customers_latent
        self.req = requests.reset_index(drop=True)
        n, R = len(drivers), len(self.req)
        self.d_status = np.zeros(n, np.int8)
        self.d_node = np.zeros(n, np.int64)
        self.d_leg: list[Leg | None] = [None] * n
        self.d_leg_id = np.full(n, -1)
        self.d_req = np.full(n, -1)
        self.d_idle_since = np.zeros(n)
        self.d_online_since = np.zeros(n)
        self.d_want_off = np.zeros(n, bool)
        self.d_repo_zone = np.full(n, -1)
        self.kappa = drivers.cost_per_km.to_numpy()
        # platform-known running records per driver (the V_long summaries): net income
        # today, distance utility U_dist (eq. 20), busy seconds, ever online today
        self.d_income = np.zeros(n)
        self.d_udist = np.zeros(n)
        self.d_busy_s = np.zeros(n)
        self.d_status_since = np.zeros(n)
        self.d_ever = np.zeros(n, bool)
        self.r_status = np.full(R, -1, np.int8)      # -1: not yet raised
        self.r_driver = np.full(R, -1)
        self.r_excluded: list[set] = [set() for _ in range(R)]
        self.r_times = {k: np.full(R, np.nan) for k in
                        ("first_offer", "accept", "arrive", "pickup", "dropoff", "cancel", "exit")}
        self.r_n_offers = np.zeros(R, np.int32)
        self.r_fare = np.full(R, np.nan)
        self.r_tip = np.zeros(R)
        self.r_km = np.full(R, np.nan)
        self.r_pick_km = np.full(R, np.nan)
        self.r_outcome_by = np.full(R, "", object)
        self.r_quote = self._quotes()
        self.recent = deque()

        self.heap: list = []
        self._seq = 0
        self.events, self.ledger, self.ratings, self.offers, self.actions = [], [], [], [], []
        self.legs, self.status_log, self.context = [], [], []
        self._eid = 0

    # ------------------------------------------------------------ plumbing

    def push(self, t, kind, *args):
        self._seq += 1
        heapq.heappush(self.heap, (t, self._seq, kind, args))

    def log(self, t, etype, driver=-1, customer=-1, request=-1, node=-1, value=np.nan, info="",
            delay=None):
        """Append a world event. `delay` (s) sets t_available - t_event; by default
        driver/app telemetry arrives after a short random delay."""
        if delay is None:
            delay = self.rng.uniform(*self.b.ping_delay_s) if driver >= 0 else 0.0
        recorded = self.rng.random() >= DROP_PROB.get(etype, 0.0)
        zone = int(self.node_zone[node]) if node >= 0 else -1
        self.events.append((self._eid, t, t + delay, etype, int(driver), int(customer), int(request),
                            zone, int(node), float(value), info, recorded))
        self._eid += 1

    def book(self, t, party, kind, amount, driver=-1, customer=-1, request=-1, quantity=np.nan,
             unit="", delay=0.0):
        if party == "driver" and kind != "online_time":
            self.d_income[driver] += amount
        self.ledger.append((len(self.ledger), t, t + delay, party, kind, round(float(amount), 4),
                            self.city.currency, int(driver), int(customer), int(request),
                            float(quantity), unit))

    def set_status(self, i, s, t):
        if self.d_status[i] in (PICKUP, OCCUPIED, REPO, OUTSIDE):
            self.d_busy_s[i] += t - self.d_status_since[i]
        self.d_status_since[i] = t
        if s != OFF:
            self.d_ever[i] = True
        self.d_status[i] = s
        self.status_log.append((t, int(i), int(s), int(self.d_node[i])))

    def start_leg(self, i, target, t, kind, request=-1) -> Leg:
        leg = self.router.route(self.d_node[i], target, t)
        self.d_leg[i] = leg
        self.d_leg_id[i] = len(self.legs)
        self.legs.append([len(self.legs), int(i), int(request), kind, leg])
        self.log(t, "leg_start", driver=i, request=request, node=int(self.d_node[i]),
                 value=leg.duration, info=kind)
        return leg

    def end_leg(self, i, t, truncate=False):
        """Close the driver's current leg (optionally cut short at t) and book its cost."""
        leg = self.d_leg[i]
        if leg is None:
            return
        if truncate and t < leg.end:
            leg = leg.truncate(t)
            self.legs[self.d_leg_id[i]][4] = leg
        self.d_node[i] = int(leg.nodes[-1])
        km = leg.length_m / 1000
        self.book(t, "driver", "operating_cost", -self.kappa[i] * km, driver=i,
                  request=self.legs[self.d_leg_id[i]][2], quantity=km, unit="km")
        self.d_leg[i] = None
        self.d_leg_id[i] = -1

    # ------------------------------------------------------------ setup

    def _quotes(self):
        """Upfront fare quote from typical zone travel time and distance at request time."""
        q = np.zeros(len(self.req))
        for k, r in enumerate(self.req.itertuples()):
            h = int(r.t_request // 3600) % 24
            oz = r.origin_zone
            if r.external_dest >= 0:
                e = self.m.external[r.external_dest]
                dz = self.node_zone[self.gw_pos[r.external_dest]]
                km = self.zone_dist[self.daytype, h, oz, dz] / 1000 + e.outside_km
                mins = self.zone_tt[self.daytype, h, oz, dz] / 60 + e.outside_min
                q[k] = fare(self.city, km - e.outside_km, mins, e.outside_km)
            else:
                dz = r.dest_zone
                km = max(self.zone_dist[self.daytype, h, oz, dz], 800) / 1000
                mins = max(self.zone_tt[self.daytype, h, oz, dz], 120) / 60
                q[k] = fare(self.city, km, mins)
        return q

    def _schedule_shifts(self):
        rng = np.random.default_rng(self.cfg.seed + 3)
        n = len(self.drv)
        works = rng.random(n) < self.dlat.days_per_week.to_numpy() / 7
        start = (self.drv.usual_shift_start_h.to_numpy() + rng.normal(0, 0.75, n)) * 3600
        length = self.dlat.shift_len_median_h.to_numpy() * np.exp(0.2 * rng.standard_normal(n)) * 3600
        # yesterday's shift may still be running at midnight
        works_y = rng.random(n) < self.dlat.days_per_week.to_numpy() / 7
        start_y = (self.drv.usual_shift_start_h.to_numpy() + rng.normal(0, 0.75, n)) * 3600 - 86400
        end_y = start_y + length * np.exp(0.2 * rng.standard_normal(n))
        for i in range(n):
            if works_y[i] and end_y[i] > 0:
                self.push(0.0, "online", i, end_y[i])
            if works[i]:
                s = max(start[i], 0.0)
                if works_y[i] and end_y[i] > s:      # overlapping shifts: just extend
                    continue
                self.push(s, "online", i, start[i] + length[i])

    def _home_node(self, i, rng):
        z = int(self.drv.home_zone.iloc[i])
        cand = np.flatnonzero(self.node_zone == z)
        return int(rng.choice(cand))

    # ------------------------------------------------------------ run

    def run(self, log=print):
        cfg = self.cfg
        t0 = time.time()
        self._schedule_shifts()
        for k, r in enumerate(self.req.itertuples()):
            self.push(r.t_request, "request", k)
        end = cfg.horizon_s + cfg.drain_s
        t = 0.0
        while t < end:
            self.push(t, "epoch")
            t += cfg.epoch_s
        t = 0.0
        while t < end:
            self.push(t, "context")
            t += cfg.context_every_s
        if self.day.rain:
            a, b_, k = self.day.rain
            self.push(a, "env", "rain_start", k)
            self.push(min(b_, 86400.0), "env", "rain_end", k)
        for inc in self.day.incidents.itertuples():
            self.push(inc.start_s, "env", "incident_start", inc.edge, inc.cap_factor)
            self.push(inc.end_s, "env", "incident_end", inc.edge, inc.cap_factor)

        handlers = {k[3:]: getattr(self, k) for k in dir(self) if k.startswith("on_")}
        next_report = 3 * 3600
        while self.heap:
            t, _, kind, args = heapq.heappop(self.heap)
            if t > end:
                break
            handlers[kind](t, *args)
            if t >= next_report:
                done = int((self.r_status == COMPLETED).sum())
                log(f"    {t / 3600:5.1f}h  requests {int((self.r_status >= 0).sum()):,}  "
                    f"completed {done:,}  online {int((self.d_status != OFF).sum())}  "
                    f"[{time.time() - t0:.0f}s]")
                next_report += 3 * 3600
        # close the day: drivers still online are logged off at the end
        for i in np.flatnonzero(self.d_status != OFF):
            if self.d_leg[i] is not None:
                self.end_leg(i, end, truncate=True)
            self._go_offline(i, end)
        log(f"    done in {time.time() - t0:.0f}s, {self.router.n_routes:,} routed legs")

    # ------------------------------------------------------------ handlers

    def on_env(self, t, what, *args):
        if what.startswith("rain"):
            self.log(t, what, value=args[0], info="intensity")
        else:
            edge, factor = args
            self.log(t, what, node=int(self.net.u[int(edge)]), value=factor, info=f"edge={int(edge)}")

    def on_online(self, t, i, shift_end):
        if self.d_status[i] != OFF:
            return
        self.d_node[i] = self._home_node(i, self.rng)
        self.d_idle_since[i] = t
        self.d_online_since[i] = t
        self.d_want_off[i] = False
        self.set_status(i, IDLE, t)
        self.log(t, "driver_online", driver=i, node=int(self.d_node[i]))
        self.push(max(shift_end, t + 1800), "shift_end", i)

    def on_shift_end(self, t, i):
        if self.d_status[i] == IDLE:
            self._go_offline(i, t)
        elif self.d_status[i] == REPO:
            self.end_leg(i, t, truncate=True)
            self.log(t, "reposition_abandoned", driver=i, node=int(self.d_node[i]))
            self._go_offline(i, t)
        elif self.d_status[i] != OFF:
            self.d_want_off[i] = True

    def _go_offline(self, i, t):
        self.set_status(i, OFF, t)
        self.log(t, "driver_offline", driver=i, node=int(self.d_node[i]))
        self.book(t, "driver", "online_time", 0.0, driver=i,
                  quantity=t - self.d_online_since[i], unit="s")

    def _become_idle(self, i, t):
        self.d_req[i] = -1
        if self.d_want_off[i]:
            self._go_offline(i, t)
            return
        self.d_idle_since[i] = t
        self.set_status(i, IDLE, t)

    def on_request(self, t, k):
        r = self.req.iloc[k]
        self.r_status[k] = PENDING
        self.recent.append((t, int(r.origin_zone)))
        self.log(t, "request_created", customer=r.customer_id, request=k, node=int(r.origin_node),
                 value=self.r_quote[k], info="intercity" if r.external_dest >= 0 else "",
                 delay=0.0)
        self.push(t + r.patience_pre_s, "patience", k)
        self.push(t + self.b.platform_max_wait_s, "expire", k)

    def on_patience(self, t, k):
        if self.r_status[k] in (PENDING, R_OFFERED):
            self._cancel(t, k, by="customer_pre_match")

    def on_expire(self, t, k):
        if self.r_status[k] in (PENDING, R_OFFERED):
            self.r_status[k] = EXPIRED
            self.r_times["cancel"][k] = t
            self.r_outcome_by[k] = "platform_expired"
            self.log(t, "request_expired", customer=self.req.customer_id.iloc[k], request=k, delay=0.0)

    def _cancel(self, t, k, by):
        r = self.req.iloc[k]
        i = self.r_driver[k]
        prev = self.r_status[k]
        self.r_status[k] = CANCELLED
        self.r_times["cancel"][k] = t
        self.r_outcome_by[k] = by
        self.log(t, "customer_cancelled", driver=i if prev >= ACCEPTED else -1, customer=r.customer_id,
                 request=k, value=float(prev), info=by, delay=0.0)
        if prev in (ACCEPTED, R_PICKUP) and i >= 0:
            if self.d_leg[i] is not None:
                self.end_leg(i, t, truncate=True)
            if t - self.r_times["accept"][k] > self.b.cancel_fee_grace_s:
                fee = self.m.cancel_fee
                self.book(t, "customer", "cancel_fee", -fee, customer=r.customer_id, request=k)
                self.book(t, "driver", "cancel_fee", fee * (1 - self.m.commission), driver=i, request=k)
                self.book(t, "platform", "commission", fee * self.m.commission, request=k)
            self._become_idle(i, t)

    # --- decisions

    def on_epoch(self, t):
        obs = self._observe(t)
        if len(obs.idle):
            self._apply(t, self.policy.act(obs), obs)

    def _observe(self, t) -> Observation:
        while self.recent and self.recent[0][0] < t - self.cfg.recent_window_s:
            self.recent.popleft()
        recent = np.bincount([z for _, z in self.recent], minlength=self.Z).astype(float)
        idle_ix = np.flatnonzero(self.d_status == IDLE)
        idle = pd.DataFrame({
            "driver_id": idle_ix, "node": self.d_node[idle_ix],
            "zone": self.node_zone[self.d_node[idle_ix]], "idle_since": self.d_idle_since[idle_ix],
        })
        pk = np.flatnonzero(self.r_status == PENDING)
        rq = self.req.iloc[pk]
        pending = pd.DataFrame({
            "request_id": pk, "origin_node": rq.origin_node.to_numpy(),
            "origin_zone": rq.origin_zone.to_numpy(),
            "dest_zone": np.where(rq.external_dest.to_numpy() >= 0, self.gw_zone[rq.external_dest.to_numpy().clip(0)],
                                  rq.dest_zone.to_numpy()),
            "external": rq.external_dest.to_numpy() >= 0,
            "t_request": rq.t_request.to_numpy(), "quote": self.r_quote[pk],
            "excluded": [self.r_excluded[k] for k in pk],
        })
        h = int(t // 3600) % 24
        return Observation(t=t, hour=h, idle=idle, pending=pending,
                           zone_tt=self.zone_tt[self.daytype, h], recent_demand=recent,
                           recent_window_s=self.cfg.recent_window_s,
                           intra_zone_s=self.intra[self.daytype, h], max_pickup_s=60 * self.m.max_pickup_min,
                           repositioning_to=np.bincount(self.d_repo_zone[self.d_status == REPO],
                                                        minlength=self.Z).astype(float),
                           fleet={"status": self.d_status, "zone": self.node_zone[self.d_node],
                                  "income": self.d_income, "udist": self.d_udist,
                                  "busy_s": self.d_busy_s + np.where(
                                      np.isin(self.d_status, (PICKUP, OCCUPIED, REPO, OUTSIDE)),
                                      t - self.d_status_since, 0.0),
                                  "ever_online": self.d_ever, "cost_per_km": self.kappa,
                                  "epoch": int(round(t / self.cfg.epoch_s)), "epoch_s": self.cfg.epoch_s,
                                  "day_type": self.daytype,
                                  "zone_dist": self.zone_dist[self.daytype, h]})

    def _apply(self, t, act, obs):
        aid = len(self.actions)
        seen_d, seen_r, n_off, n_disp = set(), set(), 0, 0
        eta = {}
        if act.offers:
            od = pd.DataFrame(act.offers, columns=["driver_id", "request_id"])
            dz = self.node_zone[self.d_node[od.driver_id]]
            rz = self.req.origin_zone.to_numpy()[od.request_id]
            e = self.zone_tt[self.daytype, obs.hour][dz, rz]
            e = np.where(dz == rz, self.intra[self.daytype, obs.hour][dz], e)
            eta = dict(zip(zip(od.driver_id, od.request_id), e))
        for i, k in act.offers:
            if (i in seen_d or k in seen_r or self.d_status[i] != IDLE or self.r_status[k] != PENDING
                    or i in self.r_excluded[k]):
                continue                     # infeasible under eq. 7-8: dropped
            seen_d.add(i), seen_r.add(k)
            n_off += 1
            self.set_status(i, OFFERED, t)
            self.d_req[i] = k
            self.r_status[k] = R_OFFERED
            self.r_n_offers[k] += 1
            if np.isnan(self.r_times["first_offer"][k]):
                self.r_times["first_offer"][k] = t
            oid = len(self.offers)
            self.offers.append([oid, aid, t, int(i), int(k), float(eta.get((i, k), np.nan)), "", np.nan])
            self.log(t, "offer_sent", driver=i, customer=self.req.customer_id.iloc[k], request=k,
                     node=int(self.d_node[i]), value=float(eta.get((i, k), np.nan)), delay=0.0)
            if self.rng.random() < 0.04:      # ASSUMPTION: 4% of offers go unanswered
                self.push(t + self.b.offer_timeout_s, "offer_response", oid, "timeout")
            else:
                self.push(t + self.rng.uniform(*self.b.offer_response_s), "offer_response", oid, "")
        for i, z in act.dispatches:
            if i in seen_d or self.d_status[i] != IDLE or not (0 <= z < self.Z):
                continue
            seen_d.add(i)
            n_disp += 1
            self.set_status(i, REPO, t)       # instruction issued; compliance decided below
            self.d_repo_zone[i] = z
            self.log(t, "dispatch_sent", driver=i, node=int(self.d_node[i]), value=z, delay=0.0)
            self.push(t + self.rng.uniform(2, 10), "dispatch_response", i, z)
        self.actions.append((aid, t, self.policy.name, self.policy.version, len(obs.idle),
                             len(obs.pending), n_off, n_disp, json.dumps(act.info)))

    def on_offer_response(self, t, oid, kind):
        _, aid, t_sent, i, k, eta, _, _ = self.offers[oid]
        r = self.req.iloc[k]
        if kind == "timeout":
            accept = None
        else:
            lat = self.dlat.iloc[i]
            h = (t / 3600) % 24
            z = (lat.accept_bias
                 + self.b.accept_eta_per_min * (eta / 60 if np.isfinite(eta) else 10)
                 + self.b.accept_fare_per_unit * self.r_quote[k] / self.m.min_fare
                 + (self.b.accept_intercity if r.external_dest >= 0 else 0.0)
                 + (self.b.accept_night if h < 5 else 0.0))
            accept = bool(self.rng.random() < 1 / (1 + np.exp(-z)))
        result = "timeout" if accept is None else ("accepted" if accept else "rejected")
        self.offers[oid][6], self.offers[oid][7] = result, t
        self.log(t, f"offer_{result}", driver=i, customer=r.customer_id, request=k,
                 node=int(self.d_node[i]))
        if self.r_status[k] != R_OFFERED:     # the customer cancelled meanwhile
            if self.d_status[i] == OFFERED:
                self._become_idle(i, t)
            return
        if not accept:
            self.r_status[k] = PENDING
            self.r_excluded[k].add(i)
            self._become_idle(i, t)
            return
        self.r_status[k] = ACCEPTED
        self.r_driver[k] = i
        self.r_times["accept"][k] = t
        self.set_status(i, PICKUP, t)
        leg = self.start_leg(i, int(r.origin_node), t, "pickup", request=k)
        self.push(leg.end, "arrive", i, k)
        eta_real = leg.duration
        if eta_real > r.patience_post_s:
            self.push(t + self.rng.uniform(15, 60), "cust_cancel", k, "customer_long_eta")
        elif self.rng.random() < self.b.random_cancel_prob:
            self.push(t + self.rng.uniform(0, max(eta_real, 30)), "cust_cancel", k, "customer_changed_mind")

    def on_cust_cancel(self, t, k, why):
        if self.r_status[k] in (ACCEPTED, R_PICKUP):
            self._cancel(t, k, by=why)

    def on_dispatch_response(self, t, i, z):
        if self.d_status[i] != REPO or self.d_leg[i] is not None:
            return
        comply = self.rng.random() < self.dlat.reposition_compliance.iloc[i]
        self.log(t, "dispatch_accepted" if comply else "dispatch_declined", driver=i,
                 node=int(self.d_node[i]), value=z)
        if not comply or self.node_zone[self.d_node[i]] == z:
            self._become_idle_keep_since(i, t)
            return
        leg = self.start_leg(i, int(self.zone_centre[z]), t, "reposition")
        self.push(leg.end, "reposition_end", i, self.d_leg_id[i])

    def _become_idle_keep_since(self, i, t):
        if self.d_want_off[i]:
            self._go_offline(i, t)
        else:
            self.set_status(i, IDLE, t)

    def on_reposition_end(self, t, i, leg_id):
        if self.d_status[i] != REPO or self.d_leg_id[i] != leg_id:
            return
        self.end_leg(i, t)
        self.log(t, "reposition_arrived", driver=i, node=int(self.d_node[i]))
        self._become_idle(i, t)

    # --- trip

    def on_arrive(self, t, i, k):
        if self.r_status[k] != ACCEPTED or self.r_driver[k] != i:
            return
        self.r_pick_km[k] = self.d_leg[i].length_m / 1000
        self.end_leg(i, t)
        self.r_status[k] = R_PICKUP
        self.r_times["arrive"][k] = t
        self.log(t, "driver_arrived", driver=i, customer=self.req.customer_id.iloc[k], request=k,
                 node=int(self.d_node[i]))
        self.push(t + self.rng.uniform(*self.b.boarding_s), "board", i, k)

    def on_board(self, t, i, k):
        if self.r_status[k] != R_PICKUP or self.r_driver[k] != i:
            return
        r = self.req.iloc[k]
        self.r_status[k] = IN_TRIP
        self.r_times["pickup"][k] = t
        self.set_status(i, OCCUPIED, t)
        self.log(t, "pickup", driver=i, customer=r.customer_id, request=k, node=int(self.d_node[i]))
        if r.external_dest >= 0:
            leg = self.start_leg(i, int(self.gw_pos[int(r.external_dest)]), t, "trip_to_gateway", request=k)
            self.push(leg.end, "exit_city", i, k)
        else:
            leg = self.start_leg(i, int(r.dest_node), t, "trip", request=k)
            self.push(leg.end, "dropoff", i, k, 0.0)

    def on_exit_city(self, t, i, k):
        e = self.m.external[self.req.external_dest.iloc[k]]
        self.r_km[k] = self.d_leg[i].length_m / 1000
        self.end_leg(i, t)
        self.r_times["exit"][k] = t
        self.set_status(i, OUTSIDE, t)
        self.log(t, "exit_city", driver=i, request=k, node=int(self.d_node[i]), info=e.name)
        out = e.outside_min * 60 * self.rng.uniform(0.85, 1.3)
        self.push(t + out, "dropoff", i, k, e.outside_km)

    def on_dropoff(self, t, i, k, outside_km):
        r = self.req.iloc[k]
        if outside_km:
            km_in = self.r_km[k]
        else:
            km_in = self.d_leg[i].length_m / 1000
            self.end_leg(i, t)
        minutes = (t - self.r_times["pickup"][k]) / 60
        f = fare(self.city, km_in, minutes, outside_km)
        self.r_status[k] = COMPLETED
        self.r_times["dropoff"][k] = t
        self.r_fare[k] = f
        self.r_km[k] = km_in + outside_km
        self.r_outcome_by[k] = "completed"
        self.d_udist[i] += km_in - self.r_pick_km[k]
        cust = int(r.customer_id)
        node = int(self.d_node[i]) if not outside_km else -1
        self.log(t, "dropoff", driver=i, customer=cust, request=k, node=node, value=f,
                 info="outside" if outside_km else "")
        m = self.m
        self.book(t, "customer", "customer_charge", -(f + m.booking_fee), customer=cust, request=k)
        self.book(t, "driver", "fare", f, driver=i, request=k, quantity=self.r_km[k], unit="km")
        self.book(t, "driver", "commission", -f * m.commission, driver=i, request=k)
        self.book(t, "platform", "commission", f * m.commission, request=k)
        self.book(t, "platform", "booking_fee", m.booking_fee, request=k)
        self.log(t, "settlement", driver=i, customer=cust, request=k, value=f * (1 - m.commission), delay=0.0)
        self._after_trip(t, i, k, cust)
        if outside_km:
            back = self.m.external[int(r.external_dest)].outside_min * 60 * self.rng.uniform(0.9, 1.3)
            self.book(t, "driver", "operating_cost", -self.kappa[i] * 2 * outside_km, driver=i,
                      request=k, quantity=2 * outside_km, unit="km")
            self.push(t + back, "reenter_city", i)
        else:
            self._become_idle(i, t)

    def on_reenter_city(self, t, i):
        self.log(t, "reenter_city", driver=i, node=int(self.d_node[i]))
        self._become_idle(i, t)

    def _after_trip(self, t, i, k, cust):
        """Tips and ratings: stochastic, delayed, observed later than they happen."""
        b, m = self.b, self.m
        cl = self.clat.iloc[cust]
        wait_min = (self.r_times["arrive"][k] - self.req.t_request.iloc[k]) / 60
        tip_p = m.tip_prob * cl.tip_propensity / 0.3
        if self.rng.random() < min(tip_p, 0.95):
            tip = round(self.r_fare[k] * m.tip_frac_mean * np.exp(0.4 * self.rng.standard_normal()), 2)
            self.push(t + self.rng.exponential(b.tip_delay_mean_s), "tip", i, k, cust, tip)
        if self.rng.random() < b.rate_driver_prob:
            q = self.dlat.service_quality.iloc[i] + cl.rating_leniency - 0.08 * max(0.0, wait_min - 5)
            v = int(np.clip(np.round(q + 0.35 * self.rng.standard_normal()), 1, 5))
            self.push(t + self.rng.exponential(b.rating_delay_mean_s), "rating", "customer", cust, "driver", i, k, v)
        if self.rng.random() < b.rate_customer_prob:
            v = int(np.clip(np.round(cl.behaviour_score + 0.3 * self.rng.standard_normal()), 1, 5))
            self.push(t + self.rng.exponential(b.rating_delay_mean_s / 4), "rating", "driver", i, "customer", cust, k, v)

    def on_tip(self, t, i, k, cust, tip):
        self.r_tip[k] = tip
        self.book(t, "customer", "tip", -tip, customer=cust, request=k)
        self.book(t, "driver", "tip", tip, driver=i, request=k)
        self.log(t, "tip", driver=i, customer=cust, request=k, value=tip, delay=0.0)

    def on_rating(self, t, src_kind, src, tgt_kind, tgt, k, v):
        delay = self.rng.uniform(1, 30)
        self.ratings.append((len(self.ratings), src_kind, int(src), tgt_kind, int(tgt), int(k), v, t, t + delay))
        self.log(t, f"rating_{tgt_kind}", driver=src if src_kind == "driver" else tgt,
                 customer=src if src_kind == "customer" else tgt, request=k, value=v, delay=delay)

    # --- context (W_t summaries)

    def on_context(self, t):
        w = self.cfg.context_every_s
        on = self.d_status != OFF
        zone_of_driver = self.node_zone[self.d_node]
        idle_z = np.bincount(zone_of_driver[self.d_status == IDLE], minlength=self.Z)
        busy_z = np.bincount(zone_of_driver[on & (self.d_status != IDLE) & (self.d_status != OUTSIDE)],
                             minlength=self.Z)
        pend = np.flatnonzero((self.r_status == PENDING) | (self.r_status == R_OFFERED))
        pend_z = np.bincount(self.req.origin_zone.to_numpy()[pend], minlength=self.Z)
        t_req = self.req.t_request.to_numpy()
        new = (t_req >= t - w) & (t_req < t)
        new_z = np.bincount(self.req.origin_zone.to_numpy()[new], minlength=self.Z)
        tti = self.traffic.network_tti(self.day, t % 86400)
        rain = self.day.rain_at(t % 86400)
        for z in range(self.Z):
            self.context.append((t, z, int(new_z[z]), int(pend_z[z]), int(idle_z[z]), int(busy_z[z]),
                                 tti, rain))

    # ------------------------------------------------------------ outputs

    def tables(self) -> dict[str, pd.DataFrame]:
        ev = pd.DataFrame(self.events, columns=["event_id", "t_event", "t_available", "type", "driver_id",
                                                "customer_id", "request_id", "zone", "node", "value",
                                                "info", "recorded"])
        ledger = pd.DataFrame(self.ledger, columns=["entry_id", "t_event", "t_available", "party", "kind",
                                                    "amount", "currency", "driver_id", "customer_id",
                                                    "request_id", "quantity", "unit"])
        ratings = pd.DataFrame(self.ratings, columns=["rating_id", "source_kind", "source_id", "target_kind",
                                                      "target_id", "request_id", "value", "t_event",
                                                      "t_available"])
        offers = pd.DataFrame(self.offers, columns=["offer_id", "action_id", "t_sent", "driver_id",
                                                    "request_id", "eta_est_s", "response", "t_response"])
        actions = pd.DataFrame(self.actions, columns=["action_id", "t", "policy", "policy_version",
                                                      "n_idle", "n_pending", "n_offers", "n_dispatches", "info"])
        context = pd.DataFrame(self.context, columns=["t", "zone", "new_requests", "pending", "idle_drivers",
                                                      "busy_drivers", "network_tti", "rain"])
        status = pd.DataFrame(self.status_log, columns=["t", "driver_id", "status", "node"])
        legs = pd.DataFrame({
            "leg_id": [lg[0] for lg in self.legs], "driver_id": [lg[1] for lg in self.legs],
            "request_id": [lg[2] for lg in self.legs], "kind": [lg[3] for lg in self.legs],
            "t_start": [lg[4].start for lg in self.legs], "t_end": [lg[4].end for lg in self.legs],
            "length_m": [lg[4].length_m for lg in self.legs],
            "nodes": [lg[4].nodes.astype(np.int32) for lg in self.legs],
            "times": [lg[4].times.astype(np.float32) for lg in self.legs],
        })
        req = self.req.copy()
        req["quote"] = self.r_quote
        req["status"] = [R_STATUS[s] if s >= 0 else "not_raised" for s in self.r_status]
        req["outcome"] = self.r_outcome_by
        req["driver_id"] = self.r_driver
        req["n_offers"] = self.r_n_offers
        for k, v in self.r_times.items():
            req[f"t_{k}"] = v
        req["fare"] = self.r_fare
        req["tip"] = self.r_tip
        req["distance_km"] = self.r_km
        req["wait_s"] = req.t_arrive - req.t_request
        trips = req[req.driver_id >= 0].copy()
        return {"events": ev, "ledger": ledger, "ratings": ratings, "offers": offers, "actions": actions,
                "context": context, "status": status, "legs": legs, "requests": req, "trips": trips}
