"""World model 3, stage 1: the entity-level world, rebuilt from a simulated day's logs.

Follows formulations/01_all_inputs_formulation.md. Time steps are the platform's decision
epochs, DT = 30 s. At each boundary tau_t:

  drivers   D_t  every driver: status, exact road position (edge u -> v and the fraction
                 driven, or standing at a node), current leg (kind, destination, metres
                 left), assigned request, time in status, time online, today's counters
                 (trips, income, offers received / rejected / timed out, repositioning
                 orders received / declined, breaks) and profile (mode, vehicle, cost per km,
                 tenure, rating)
  requests  R_t  every open request: origin and destination node, mode, time waited,
                 lifecycle status, offers so far, quote
  roads     W_t  travel time / free-flow time of every road edge (the simulator's router
                 refreshes travel times every 5 minutes), rain
  actions   A_t  the offers (driver -> request) and repositioning orders (driver -> zone)
                 issued at tau_t
  events  E_t+1  everything that happens in (tau_t, tau_t + DT]: offer responses,
                 order responses, movement, pickups, drop-offs, cancellations, breaks,
                 going online / offline, new requests

Counters use only events the platform has received by tau_t (t_available <= tau_t, eq. 5).
Positions and statuses are the true ones at tau_t; the telemetry delay of a few seconds
and the ~2% of dropped movement messages are ignored at this stage.
The drivers' hidden traits (drivers_latent: accept bias, repositioning compliance, ...)
are the unobserved Z_t of eq. 2. They are never part of the state; they are loaded only
so that later stages can check what the model infers about them.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import traffic as T

DT = 30.0
N_EPOCH = int(86400 / DT)
STATUS = ["offline", "idle", "offered", "pickup", "occupied", "repositioning", "outside", "break"]
LEG_KINDS = ["pickup", "trip", "trip_to_gateway", "reposition", "self_reposition", "go_home"]
R_OPEN = ["pending", "offered", "accepted", "driver_waiting"]
MODES = ["bike", "auto", "cab"]
# today's counters per driver: name -> event types counted
COUNTERS = {
    "trips": ["dropoff"],
    "offers": ["offer_sent"],
    "offers_rejected": ["offer_rejected"],
    "offers_timeout": ["offer_timeout"],
    "orders": ["dispatch_sent"],
    "orders_declined": ["dispatch_declined"],
    "breaks": ["driver_break"],
}


class CityGraph:
    """The directed road graph of a city, with node positions and the traffic model."""

    def __init__(self, city):
        d = city.data_dir
        self.city = city
        nodes = pd.read_parquet(d / "nodes.parquet")
        edges = pd.read_parquet(d / "edges.parquet")
        self.net = T.Net.from_tables(nodes, edges)
        self.N, self.E = len(nodes), len(edges)
        self.x, self.y = nodes.x_m.to_numpy(), nodes.y_m.to_numpy()
        self.lon, self.lat = nodes.lon.to_numpy(), nodes.lat.to_numpy()
        self.zone = nodes.zone.to_numpy()
        self.eu, self.ev = self.net.u.astype(np.int64), self.net.v.astype(np.int64)
        self.length, self.t0 = self.net.length, self.net.t0
        self.road_class = edges.road_class.to_numpy()
        self.name = edges.name.astype(str).to_numpy()
        # (u, v) -> edge: the fastest one when there are parallel edges (what the router takes)
        key = self.eu * self.N + self.ev
        order = np.lexsort((self.t0, key))
        k_sorted = key[order]
        first = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
        self._keys, self._edge = k_sorted[first], order[first]
        self.traffic = T.TrafficModel(city)

    def edge_of(self, u, v):
        k = np.asarray(u, np.int64) * self.N + np.asarray(v, np.int64)
        i = np.searchsorted(self._keys, k)
        i = np.clip(i, 0, len(self._keys) - 1)
        return np.where(self._keys[i] == k, self._edge[i], -1)


def _season_seed(day_dir: Path) -> int:
    m = re.search(r"seed(\d+)", day_dir.parent.name)
    return int(m.group(1)) if m else 0


class DayLog:
    """One simulated day: the logs plus per-epoch arrays of every driver's state."""

    def __init__(self, graph: CityGraph, day_dir: Path):
        self.g, self.dir = graph, Path(day_dir)
        self.date = pd.Timestamp(self.dir.name)
        rd = lambda n: pd.read_parquet(self.dir / f"{n}.parquet")  # noqa: E731
        self.drivers, self.latent = rd("drivers"), rd("drivers_latent")
        self.ev, self.legs, self.offers = rd("events"), rd("legs"), rd("offers")
        self.req, self.status_log, self.context = rd("requests"), rd("status"), rd("context")
        self.nD = len(self.drivers)
        assert (self.drivers.driver_id.to_numpy() == np.arange(self.nD)).all()
        # the day's traffic, exactly as the simulator drew it (world/hd/run.py)
        seed = _season_seed(self.dir) * 1000 + self.date.dayofyear
        self.day = graph.traffic.sample_day(self.date, np.random.default_rng(seed + 7))
        self._tracks()
        self._counters()
        self._requests()

    # ------------------------------------------------------------------ drivers
    def _tracks(self):
        g, nD, nE = self.g, self.nD, N_EPOCH + 1
        tau = np.arange(nE) * DT
        self.status = np.zeros((nD, nE), np.int8)
        self.status_since = np.zeros((nD, nE), np.float32)
        self.node = np.zeros((nD, nE), np.int32)          # standing node, or the start of the edge driven
        self.stop = np.zeros((nD, nE), np.int32)          # last node stood at (leg start while driving)
        self.node_to = np.full((nD, nE), -1, np.int32)    # end of the edge being driven (-1: standing)
        self.frac = np.zeros((nD, nE), np.float16)        # share of that edge already driven
        self.leg = np.full((nD, nE), -1, np.int32)        # leg being driven
        self.left_m = np.zeros((nD, nE), np.float32)      # metres left on the leg
        st = self.status_log.sort_values(["driver_id", "t"], kind="stable")
        lg = self.legs.sort_values(["driver_id", "t_start"], kind="stable")
        ends = pd.DataFrame({"driver_id": lg.driver_id, "t": lg.t_end, "node": [n[-1] for n in lg.nodes]})
        keys = pd.concat([st[["driver_id", "t", "node"]], ends]).sort_values(["driver_id", "t"], kind="stable")
        for d, s in st.groupby("driver_id"):
            i = np.searchsorted(s.t.to_numpy(), tau, "right") - 1
            ok = i >= 0
            self.status[d, ok] = s.status.to_numpy()[i[ok]]
            self.status_since[d, ok] = s.t.to_numpy()[i[ok]]
        for d, k in keys.groupby("driver_id"):
            i = np.searchsorted(k.t.to_numpy(), tau, "right") - 1
            ok = i >= 0
            self.node[d, ok] = k.node.to_numpy()[i[ok]]
            self.node[d, ~ok] = k.node.to_numpy()[0]
        self.stop[:] = self.node
        self.leg_kind = lg.set_index("leg_id").kind.reindex(np.arange(len(self.legs))).map(
            {k: i for i, k in enumerate(LEG_KINDS)}).fillna(-1).astype(int).to_numpy()
        self.leg_dest = np.array([n[-1] for n in self.legs.sort_values("leg_id").nodes], np.int64)
        self.leg_req = self.legs.sort_values("leg_id").request_id.to_numpy()
        for lid, d, nodes, times in zip(lg.leg_id.to_numpy(), lg.driver_id.to_numpy(), lg.nodes, lg.times):
            times = times.astype(np.float64)
            e0, e1 = int(np.ceil(times[0] / DT)), int(np.ceil(times[-1] / DT))
            if e1 <= e0 or len(nodes) < 2:
                continue
            ee = np.arange(e0, min(e1, nE))
            j = np.clip(np.searchsorted(times, tau[ee], "right") - 1, 0, len(nodes) - 2)
            u, v = nodes[j], nodes[j + 1]
            f = (tau[ee] - times[j]) / np.maximum(times[j + 1] - times[j], 1e-6)
            seg = g.length[np.maximum(g.edge_of(nodes[:-1], nodes[1:]), 0)]
            rem = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]           # metres from node j to the end
            self.node[d, ee], self.node_to[d, ee] = u, v
            self.frac[d, ee] = np.clip(f, 0, 1)
            self.leg[d, ee] = lid
            self.left_m[d, ee] = rem[j + 1] + (1 - np.clip(f, 0, 1)) * seg[j]

    def xy(self, e):
        """(nD, 2) metres: exact positions at epoch e (interpolated along the edge driven)."""
        g = self.g
        u, v, f = self.node[:, e], self.node_to[:, e], self.frac[:, e].astype(np.float64)
        mv = v >= 0
        x, y = g.x[u].copy(), g.y[u].copy()
        x[mv] += f[mv] * (g.x[v[mv]] - g.x[u[mv]])
        y[mv] += f[mv] * (g.y[v[mv]] - g.y[u[mv]])
        return np.c_[x, y]

    def _counters(self):
        """(nD, nE) running counts and income of today, from events received by each epoch."""
        ev = self.ev[self.ev.driver_id >= 0]
        e_av = np.minimum(np.ceil(ev.t_available.to_numpy() / DT).astype(int), N_EPOCH)
        self.count = {}
        for name, types in COUNTERS.items():
            m = ev.type.isin(types).to_numpy()
            a = np.zeros((self.nD, N_EPOCH + 1), np.int16)
            np.add.at(a, (ev.driver_id.to_numpy()[m], e_av[m]), 1)
            self.count[name] = np.cumsum(a, 1, dtype=np.int16)
        m = (ev.type == "settlement").to_numpy()
        a = np.zeros((self.nD, N_EPOCH + 1), np.float32)
        np.add.at(a, (ev.driver_id.to_numpy()[m], e_av[m]), ev.value.to_numpy()[m])
        self.income = np.cumsum(a, 1)
        on = ev[ev.type == "driver_online"]
        self.online_since = np.full((self.nD, N_EPOCH + 1), np.nan, np.float32)
        for d, s in on.groupby("driver_id"):
            t = s.t_event.to_numpy()
            i = np.searchsorted(t, np.arange(N_EPOCH + 1) * DT, "right") - 1
            self.online_since[d, i >= 0] = t[i[i >= 0]]

    # ----------------------------------------------------------------- requests
    def _requests(self):
        q = self.req.set_index("request_id").sort_index()
        R = len(q)
        ev = self.ev[self.ev.request_id >= 0]
        first = lambda typ: ev[ev.type == typ].groupby("request_id").t_event.min().reindex(np.arange(R)).to_numpy()  # noqa: E731
        self.r_created = first("request_created")             # NaN: the rider declined the quote
        self.r_accept, self.r_arrive = first("offer_accepted"), first("driver_arrived")
        self.r_pickup = first("pickup")
        self.r_cancel = np.fmin(first("customer_cancelled"), first("request_expired"))   # the rider gives up, or the platform does
        self.r_t = q.t_request.to_numpy()
        self.r_origin, self.r_dest = q.origin_node.to_numpy(), q.dest_node.to_numpy()
        self.r_mode = q["mode"].map({m: i for i, m in enumerate(MODES)}).to_numpy()
        self.r_quote, self.r_external = q.quote.to_numpy(), q.external_dest.to_numpy()
        o = self.offers.sort_values("t_sent")
        self.o_req, self.o_t, self.o_tr = o.request_id.to_numpy(), o.t_sent.to_numpy(), o.t_response.to_numpy()
        self.o_drv, self.o_resp = o.driver_id.to_numpy(), o.response.to_numpy()

    def open_requests(self, e):
        """DataFrame of the requests open at epoch e (created, not yet picked up or cancelled)."""
        t = e * DT
        end = np.fmin(self.r_pickup, self.r_cancel)
        idx = np.flatnonzero((self.r_created <= t) & ~(end <= t))
        st = np.zeros(len(idx), int)                                       # pending
        outstanding = (self.o_t <= t) & ~(self.o_tr <= t)
        st[np.isin(idx, self.o_req[outstanding])] = 1                      # offered, no answer yet
        st[self.r_accept[idx] <= t] = 2                                    # accepted, driver on the way
        st[self.r_arrive[idx] <= t] = 3                                    # driver waiting at the origin
        n_off = np.searchsorted(self.o_t, t, "right")
        n_off = np.bincount(self.o_req[:n_off], minlength=len(self.r_t))[idx]
        return pd.DataFrame({"request_id": idx, "status": st, "origin": self.r_origin[idx], "dest": self.r_dest[idx],
                             "external": self.r_external[idx], "mode": self.r_mode[idx], "waited_s": t - self.r_created[idx],
                             "n_offers": n_off, "quote": self.r_quote[idx]})

    # -------------------------------------------------------------------- roads
    def road_ratio(self, e):
        """Travel time / free-flow time of every edge, as the router used them at epoch e."""
        slot = int(e * DT // 300)
        return self.g.traffic.edge_times(self.day, (slot + 0.5) * 300 % 86400) / self.g.t0

    # ------------------------------------------------------------------ a state
    def drivers_at(self, e):
        """DataFrame of every driver not offline at epoch e: the D_t records."""
        t = e * DT
        on = np.flatnonzero(self.status[:, e] != 0)
        xy = self.xy(e)[on]
        leg = self.leg[on, e]
        mv = leg >= 0
        p = self.drivers.iloc[on]
        out = pd.DataFrame({
            "driver_id": on, "status": self.status[on, e], "x": xy[:, 0], "y": xy[:, 1],
            "node": self.node[on, e], "node_to": self.node_to[on, e], "frac": self.frac[on, e].astype(np.float32),
            "leg_kind": np.where(mv, self.leg_kind[np.maximum(leg, 0)], -1),
            "leg_dest": np.where(mv, self.leg_dest[np.maximum(leg, 0)], -1),
            "left_m": self.left_m[on, e],
            "request_id": np.where(mv, self.leg_req[np.maximum(leg, 0)], -1),
            "in_status_s": t - self.status_since[on, e], "online_s": t - self.online_since[on, e],
            **{c: self.count[c][on, e] for c in COUNTERS}, "income": self.income[on, e],
            "mode": p["mode"].map({m: i for i, m in enumerate(MODES)}).to_numpy(),
            "cost_per_km": p.cost_per_km.to_numpy(), "tenure_days": p.tenure_days.to_numpy(),
            "rating": (p.rating_sum / p.rating_count.clip(lower=1)).to_numpy(), "rating_n": p.rating_count.to_numpy(),
            "home_node": p.home_node.to_numpy(), "shift_start_h": p.usual_shift_start_h.to_numpy(),
        })
        return out

    def actions_at(self, e):
        t = e * DT
        ev = self.ev
        off = ev[(ev.type == "offer_sent") & (ev.t_event == t)][["driver_id", "request_id"]]
        dis = ev[(ev.type == "dispatch_sent") & (ev.t_event == t)][["driver_id", "zone"]]
        return off.reset_index(drop=True), dis.reset_index(drop=True)

    def events_in(self, e):
        """The events of (tau_e, tau_e + DT]."""
        t = e * DT
        ev = self.ev
        return ev[(ev.t_event > t) & (ev.t_event <= t + DT)].reset_index(drop=True)
