"""The high-detail ride world: World (world/sim.py) extended with

  * products / modes   bike taxi, auto-rickshaw, cab (Pune) or one product (SF). Each
                       driver has one mode; a request asks for one mode; only same-mode
                       offers are feasible. Bikes and autos move faster in congestion
                       (Mode.speed_factor on the leg's travel time).
  * surge pricing      every SURGE_EVERY_S, per zone and mode: multiplier from recent
                       requests + waiting vs idle supply, smoothed, capped (SURGE_CAP).
  * rider choice       at request time the rider sees a quote (mode tariff x surge) and an
                       ETA, and may decline (logit with hidden price / ETA sensitivity).
  * driver autonomy    an idle driver not given an instruction acts on their own hidden
                       strategy: wait, chase surge / demand nearby, take a break after
                       some hours, or head home near the end of the shift.
  * offer acceptance   adds surge (more likely) and trip direction near shift end
                       (less likely when the trip leads away from home).
  * airport queue      requests from the airport zone go to the longest-waiting idle
                       driver of that mode in the zone (first come, first served).

New statuses: driver `break` (online, not available); request `declined` (rider
declined the quote). Everything is logged as events, as before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .. import sim as S
from ..policies import Action, Observation, Policy
from ..routing import Leg
from .demand import BETA_COST, MODES, mode_fare

if "break" not in S.D_STATUS:
    S.D_STATUS.append("break")
if "declined" not in S.R_STATUS:
    S.R_STATUS.append("declined")
BREAK = S.D_STATUS.index("break")
DECLINED = S.R_STATUS.index("declined")
IDLE, OFF, REPO, PICKUP, OCCUPIED, OUTSIDE = S.IDLE, S.OFF, S.REPO, S.PICKUP, S.OCCUPIED, S.OUTSIDE
PENDING, R_OFFERED = S.PENDING, S.R_OFFERED

SURGE_EVERY_S = 300.0
DETOUR = 1.4          # road distance / straight line, fitted on simulated legs (1.37 SF, 1.43 Pune)
SURGE_CAP = {"pune_hd": 1.5, "sf_hd": 2.5}       # ASSUMPTION; Indian aggregator rules cap surge
SURGE_SMOOTH = 0.5
# ASSUMPTION: surge = 1 + gain x (demand / (idle + 1) - threshold); SF zones are smaller
SURGE_RULE = {"pune_hd": (0.4, 0.7), "sf_hd": (0.25, 1.0)}


class HDWorld(S.World):
    def __init__(self, *a, airport_zone: int = -1, **kw):
        self._modes_ready = False
        super().__init__(*a, **kw)
        self.modes = MODES[self.city.key]
        self.mode_names = list(self.modes)
        self.mode_idx = {m: j for j, m in enumerate(self.mode_names)}
        self.d_mode = self.drv["mode"].map(self.mode_idx).to_numpy()
        self.r_mode = self.req["mode"].map(self.mode_idx).to_numpy()
        self.d_shift_end = np.full(len(self.drv), np.inf)
        self.d_next_break = np.full(len(self.drv), np.inf)
        self.d_going_home = np.zeros(len(self.drv), bool)
        self.home_node = self.drv["home_node"].to_numpy()
        self.surge = np.ones((self.Z, len(self.mode_names)))
        self.r_surge = np.ones(len(self.req))
        self.r_eta_est = np.full(len(self.req), np.nan)
        self.airport_zone = airport_zone
        # GPS-style ETA: straight-line distance x detour / the hour's typical speed (from the
        # platform's own zone tables, pairs under 6 km), +30 s. Fits simulated legs within ~20%.
        nodes = a[2] if len(a) > 2 else kw["nodes"]           # World(city, net, nodes, ...)
        self.node_xy = nodes[["x_m", "y_m"]].to_numpy()
        di, tt = self.zone_dist, self.zone_tt
        near = (di < 6000) & ~np.eye(self.Z, dtype=bool)
        self.speed = np.array([[np.median(di[d, h][near[d, h]] / tt[d, h][near[d, h]]) for h in range(24)]
                               for d in range(di.shape[0])])
        self.surge_log = []
        self.context_surge = []
        self._modes_ready = True
        self.r_quote = self._quotes()

    # ---------------------------------------------------------------- pricing
    def _quotes(self):
        """Base (pre-surge) quote per request from its mode's tariff and typical times."""
        if not getattr(self, "_modes_ready", False):
            return np.zeros(len(self.req))
        q = np.zeros(len(self.req))
        for k, r in enumerate(self.req.itertuples()):
            mo = self.modes[r.mode]
            h = int(r.t_request // 3600) % 24
            oz = r.origin_zone
            if r.external_dest >= 0:
                e = self.m.external[r.external_dest]
                dz = self.node_zone[self.gw_pos[r.external_dest]]
                km = self.zone_dist[self.daytype, h, oz, dz] / 1000
                mins = self.zone_tt[self.daytype, h, oz, dz] / 60 * mo.speed_factor
                q[k] = float(mode_fare(mo, km, mins)) + self.m.intercity_per_km * e.outside_km
            else:
                dz = r.dest_zone
                km = max(self.zone_dist[self.daytype, h, oz, dz], 800) / 1000
                mins = max(self.zone_tt[self.daytype, h, oz, dz], 120) / 60 * mo.speed_factor
                q[k] = float(mode_fare(mo, km, mins))
        return q

    def on_surge(self, t):
        """Recompute surge per zone and mode from the last 15 min of requests, the current
        waiting requests and idle drivers (demand / supply ratio, smoothed and capped)."""
        win = 900.0
        t_req = self.req.t_request.to_numpy()
        recent = (t_req >= t - win) & (t_req < t) & (self.r_status >= 0)
        waiting = (self.r_status == PENDING) | (self.r_status == R_OFFERED)
        cap = SURGE_CAP[self.city.key]
        oz = self.req.origin_zone.to_numpy()
        dz_nodes = self.node_zone[self.d_node]
        for j in range(len(self.mode_names)):
            dem = np.bincount(oz[recent & (self.r_mode == j)], minlength=self.Z) * (300 / win) \
                + np.bincount(oz[waiting & (self.r_mode == j)], minlength=self.Z)
            sup = np.bincount(dz_nodes[(self.d_status == IDLE) & (self.d_mode == j)], minlength=self.Z)
            ratio = dem / (sup + 1.0)
            gain, thr = SURGE_RULE[self.city.key]
            raw = np.clip(1.0 + gain * (ratio - thr), 1.0, cap)
            self.surge[:, j] = SURGE_SMOOTH * self.surge[:, j] + (1 - SURGE_SMOOTH) * raw
        self.surge = np.round(self.surge * 10) / 10                          # shown in 0.1 steps
        self.surge_log.append((t, self.surge.copy()))

    def eta(self, from_nodes, to_nodes, t, mode_j):
        """Platform ETA (s) from node(s) to node(s), broadcasting like numpy."""
        a, b = self.node_xy[from_nodes], self.node_xy[to_nodes]
        d = np.sqrt(((a - b) ** 2).sum(-1))
        v = self.speed[self.daytype, int(t // 3600) % 24]
        return (30.0 + DETOUR * d / v) * self.modes[self.mode_names[mode_j]].speed_factor

    def _eta_estimate(self, k, t):
        """Platform's ETA shown to the rider: nearest idle same-mode driver."""
        j = self.r_mode[k]
        idle = np.flatnonzero((self.d_status == IDLE) & (self.d_mode == j))
        if not len(idle):
            return 15 * 60.0
        return float(self.eta(self.d_node[idle], int(self.req.origin_node.iat[k]), t, j).min())

    # ---------------------------------------------------------------- requests
    def on_request(self, t, k):
        r = self.req.iloc[k]
        j = self.r_mode[k]
        s = float(self.surge[r.origin_zone, j])
        self.r_surge[k] = s
        eta = self._eta_estimate(k, t)
        self.r_eta_est[k] = eta
        cl = self.clat.iloc[r.customer_id]
        # ASSUMPTION: rider accepts the quote with a logit in surge price and shown ETA
        extra = self.r_quote[k] * (s - 1.0)
        u = 3.0 + BETA_COST[self.city.key] * cl.price_sensitivity * extra \
            - 0.15 * cl.eta_sensitivity * max(eta / 60 - 4, 0)
        if self.rng.random() > 1 / (1 + np.exp(-u)):
            self.r_status[k] = DECLINED
            self.r_outcome_by[k] = "declined_quote"
            self.log(t, "quote_declined", customer=r.customer_id, request=k, node=int(r.origin_node),
                     value=self.r_quote[k] * s, info=f"surge={s:.1f};eta={eta / 60:.1f}", delay=0.0)
            return
        self.r_quote[k] *= s
        # a rider who accepted a quote with ETA x keeps waiting up to ~1.3 x (ASSUMPTION)
        self.req.at[k, "patience_post_s"] = max(r.patience_post_s, 1.3 * eta)
        super().on_request(t, k)

    # ---------------------------------------------------------------- movement
    def start_leg(self, i, target, t, kind, request=-1) -> Leg:
        leg = super().start_leg(i, target, t, kind, request)
        f = self.modes[self.mode_names[self.d_mode[i]]].speed_factor
        if f != 1.0:
            leg.times = leg.times[0] + (leg.times - leg.times[0]) * f
            e = self.events[-1]                    # the leg_start event: log the real duration
            if e[3] == "leg_start" and e[4] == i:
                self.events[-1] = e[:9] + (leg.duration,) + e[10:]
        return leg

    # ---------------------------------------------------------------- shifts, autonomy
    def on_online(self, t, i, shift_end):
        if self.d_status[i] != OFF:
            return
        super().on_online(t, i, shift_end)
        self.d_shift_end[i] = max(shift_end, t + 1800)
        self.d_next_break[i] = t + self.dlat.break_every_h.iat[i] * 3600
        self.d_going_home[i] = False
        self._schedule_autonomy(i, t)

    def _schedule_autonomy(self, i, t):
        self.push(t + self.rng.uniform(120, 420), "autonomy", i, float(self.d_idle_since[i]))

    def _become_idle(self, i, t):
        super()._become_idle(i, t)
        if self.d_status[i] == IDLE:
            if self.d_going_home[i]:
                self._go_offline(i, t)
                return
            self._schedule_autonomy(i, t)

    def on_autonomy(self, t, i, since):
        """An idle driver without an instruction decides what to do on their own."""
        if self.d_status[i] != IDLE or self.d_idle_since[i] != since:
            return
        lat = self.dlat.iloc[i]
        if t >= self.d_next_break[i]:
            self.set_status(i, BREAK, t)
            self.log(t, "driver_break", driver=i, node=int(self.d_node[i]))
            self.push(t + self.rng.uniform(900, 2100), "break_end", i)
            return
        u = self.rng.random()
        if t > self.d_shift_end[i] - 2700 and u < lat.p_home + 0.3:
            self.d_going_home[i] = True
            self.set_status(i, REPO, t)
            self.d_repo_zone[i] = self.node_zone[self.home_node[i]]
            self.log(t, "driver_heads_home", driver=i, node=int(self.d_node[i]))
            leg = self.start_leg(i, int(self.home_node[i]), t, "go_home")
            self.push(leg.end, "reposition_end", i, self.d_leg_id[i])
            return
        if u < lat.p_chase:
            z = self._hotspot(i, t)
            if z is not None:
                self.set_status(i, REPO, t)
                self.d_repo_zone[i] = z
                self.log(t, "driver_self_reposition", driver=i, node=int(self.d_node[i]), value=z)
                leg = self.start_leg(i, int(self.zone_centre[z]), t, "self_reposition")
                self.push(leg.end, "reposition_end", i, self.d_leg_id[i])
                return
        self.push(t + self.rng.uniform(300, 900), "autonomy", i, since)     # keep waiting

    def _hotspot(self, i, t):
        """Zone within 12 min with the best surge-weighted attraction (what a driver sees
        in the app's heat map), if it beats staying."""
        h = int(t // 3600) % 24
        z0 = self.node_zone[self.d_node[i]]
        j = self.d_mode[i]
        tt = self.zone_tt[self.daytype, h][z0] / 60
        near = np.flatnonzero(tt <= 12)
        if not len(near):
            return None
        score = self.surge[near, j] * self.dlat.surge_responsiveness.iat[i] - 0.03 * tt[near]
        best = near[int(np.argmax(score))]
        return int(best) if best != z0 and self.surge[best, j] > self.surge[z0, j] + 0.15 else None

    def on_break_end(self, t, i):
        if self.d_status[i] != BREAK:
            return
        self.d_next_break[i] = t + self.dlat.break_every_h.iat[i] * 3600
        self.log(t, "driver_break_end", driver=i, node=int(self.d_node[i]))
        if t >= self.d_shift_end[i]:
            self._go_offline(i, t)
        else:
            self._become_idle(i, t)

    def on_shift_end(self, t, i):
        if self.d_status[i] == BREAK:
            self._go_offline(i, t)
            return
        super().on_shift_end(t, i)

    def on_reposition_end(self, t, i, leg_id):
        if self.d_status[i] != REPO or self.d_leg_id[i] != leg_id:
            return
        if self.d_going_home[i]:
            self.end_leg(i, t)
            self.log(t, "driver_home", driver=i, node=int(self.d_node[i]))
            self._go_offline(i, t)
            return
        super().on_reposition_end(t, i, leg_id)

    # ---------------------------------------------------------------- offers
    def on_offer_response(self, t, oid, kind):
        """Base acceptance logit plus surge (more likely) and, near shift end, direction."""
        _, aid, t_sent, i, k, eta, _, _ = self.offers[oid]
        if kind != "timeout":
            s = self.r_surge[k]
            away = 0.0
            if t > self.d_shift_end[i] - 3600 and self.req.dest_node.iat[k] >= 0:
                h = int(t // 3600) % 24
                zh = self.node_zone[self.home_node[i]]
                tt = self.zone_tt[self.daytype, h]
                d_now = tt[self.node_zone[self.d_node[i]], zh]
                d_after = tt[int(self.req.dest_zone.iat[k]), zh]
                away = max(0.0, (d_after - d_now) / 600)               # per 10 min farther from home
            bias = self.dlat.accept_bias.iat[i]
            self.dlat.at[i, "accept_bias"] = bias + 1.2 * np.log(max(s, 1.0)) - 0.8 * away
            try:
                super().on_offer_response(t, oid, kind)
            finally:
                self.dlat.at[i, "accept_bias"] = bias
            return
        super().on_offer_response(t, oid, kind)

    # ---------------------------------------------------------------- dropoff fare
    def on_dropoff(self, t, i, k, outside_km):
        """Charge the mode's tariff for the realised trip, times the surge locked at request."""
        r = self.req.iloc[k]
        mo = self.modes[r["mode"]]
        km_in = self.r_km[k] if outside_km else self.d_leg[i].length_m / 1000
        minutes = (t - self.r_times["pickup"][k]) / 60
        realised = float(mode_fare(mo, km_in, minutes)) + self.m.intercity_per_km * outside_km
        self._fare_override = realised * self.r_surge[k]
        super().on_dropoff(t, i, k, outside_km)

    # ---------------------------------------------------------------- observation
    def _observe(self, t) -> Observation:
        obs = super()._observe(t)
        obs.idle["mode"] = self.d_mode[obs.idle.driver_id.to_numpy()]
        obs.pending["mode"] = self.r_mode[obs.pending.request_id.to_numpy()]
        obs.fleet["surge"] = self.surge
        obs.fleet["airport_zone"] = self.airport_zone
        obs.fleet["mode_speed"] = np.array([self.modes[m].speed_factor for m in self.mode_names])
        obs.fleet["eta"] = lambda a, b, j, t=t: self.eta(a, b, t, j)
        return obs

    def _apply(self, t, act, obs):
        act.offers = [(i, k) for i, k in act.offers if self.d_mode[i] == self.r_mode[k]]
        n0, e0 = len(self.offers), len(self.events)
        super()._apply(t, act, obs)
        eta = {}
        for o in self.offers[n0:]:                 # record the GPS-style ETA the driver is shown
            i, k = o[3], o[4]
            o[5] = eta[(i, k)] = float(self.eta(self.d_node[i], int(self.req.origin_node.iat[k]), t, self.d_mode[i]))
        for x in range(e0, len(self.events)):
            e = self.events[x]
            if e[3] == "offer_sent":
                self.events[x] = e[:9] + (eta.get((e[4], e[6]), e[9]),) + e[10:]

    def on_context(self, t):
        n_before = len(self.context)
        super().on_context(t)
        # break counts as online but neither idle nor busy
        zone_of = self.node_zone[self.d_node]
        brk = np.bincount(zone_of[self.d_status == BREAK], minlength=self.Z)
        for z in range(self.Z):
            row = list(self.context[n_before + z])
            row[5] = max(0, row[5] - int(brk[z]))
            self.context[n_before + z] = tuple(row)
            self.context_surge.append(float(self.surge[z].mean()))

    # ---------------------------------------------------------------- run / tables
    def run(self, log=print):
        t = 0.0
        while t < self.cfg.horizon_s + self.cfg.drain_s:
            self.push(t, "surge")
            t += SURGE_EVERY_S
        super().run(log=log)

    def tables(self):
        tb = super().tables()
        tb["context"]["surge"] = self.context_surge
        req = tb["requests"]
        req["surge"] = self.r_surge
        req["eta_shown_s"] = self.r_eta_est
        tb["requests"] = req
        tb["trips"] = req[req.driver_id >= 0].copy()
        rows = [(t, z, m, float(s[z, j])) for t, s in self.surge_log for z in range(self.Z)
                for j, m in enumerate(self.mode_names) if s[z, j] > 1.0]
        tb["surge"] = pd.DataFrame(rows, columns=["t", "zone", "mode", "multiplier"])
        return tb


class HDDispatch(Policy):
    """Baseline platform for the detailed cities: per-mode min-ETA batch matching, airport
    first-come-first-served queue, and periodic rebalancing per mode."""

    name = "hd_nearest"
    version = "1"

    def __init__(self, reposition_every: int = 10, max_moves: int = 60, explore: float = 0.0, seed: int = 0):
        """`explore`: share of drivers idle 10+ min sent to a random zone within 15 min, so the
        data shows what repositioning does beyond what the rule would choose."""
        self.reposition_every, self.max_moves, self.explore = reposition_every, max_moves, explore
        self.rng = np.random.default_rng(seed)
        self._n = 0

    def act(self, obs: Observation) -> Action:
        self._n += 1
        act = Action()
        idle, pend = obs.idle, obs.pending
        used = set()
        az = obs.fleet.get("airport_zone", -1)
        for j in np.unique(pend["mode"]) if len(pend) else []:
            I = idle[idle["mode"] == j]
            P = pend[pend["mode"] == j]
            if not len(I) or not len(P):
                continue
            # airport queue: longest-waiting idle driver in the airport zone takes it
            if az >= 0:
                qd = I[I.zone == az].sort_values("idle_since")
                qr = P[P.origin_zone == az].sort_values("t_request")
                for (d, _), (k, _) in zip(qd[["driver_id", "zone"]].values, qr[["request_id", "origin_zone"]].values):
                    act.offers.append((int(d), int(k)))
                    used.add(int(d))
                I = I[~I.driver_id.isin(used)]
                P = P[~P.request_id.isin([k for _, k in act.offers])]
                if not len(I) or not len(P):
                    continue
            # GPS positions: ETA from each idle driver's node to each pickup node
            eta = obs.fleet["eta"](I.node.to_numpy()[:, None], P.origin_node.to_numpy()[None, :], j)
            drv = I.driver_id.to_numpy()
            for c, ex in enumerate(P.excluded):
                if ex:
                    eta[np.isin(drv, list(ex)), c] = np.inf
            ok = eta <= obs.max_pickup_s
            if ok.any():
                r, c = linear_sum_assignment(np.where(ok, eta, 1e9))
                for a, b in zip(r, c):
                    if ok[a, b]:
                        act.offers.append((int(drv[a]), int(P.request_id.iloc[b])))
                        used.add(int(drv[a]))
        free = idle[~idle.driver_id.isin(used)]
        if self.reposition_every and self._n % self.reposition_every == 0 and len(free):
            act.dispatches = self._rebalance(obs, free)
        if self.explore and len(free):
            moved = {d for d, _ in act.dispatches}
            cand = free[(obs.t - free.idle_since >= 600) & ~free.driver_id.isin(moved)]
            pick = cand[self.rng.random(len(cand)) < self.explore / 12]      # 30 s epochs: ~explore per 6 min
            for d, z in zip(pick.driver_id, pick.zone):
                reach = np.flatnonzero(obs.zone_tt[z] <= 900)
                act.dispatches.append((int(d), int(self.rng.choice(reach))))
        act.info = {"reposition_every": self.reposition_every, "max_moves": self.max_moves, "explore": self.explore}
        return act

    def _rebalance(self, obs, idle):
        out = []
        Z = len(obs.recent_demand)
        rate = obs.recent_demand / obs.recent_window_s * 600
        need = rate + np.bincount(obs.pending.origin_zone, minlength=Z)
        supply = np.bincount(idle.zone, minlength=Z) + obs.repositioning_to
        deficit = need - supply
        long_idle = idle[obs.t - idle.idle_since >= 900]
        for z in np.argsort(deficit):
            if deficit[z] >= 0 or len(out) >= self.max_moves:
                break
            for d in long_idle[long_idle.zone == z].driver_id:
                reach = np.flatnonzero((obs.zone_tt[z] <= 900) & (deficit > 0.5))
                if not len(reach) or deficit[z] >= 0:
                    break
                tgt = int(reach[np.argmax(deficit[reach] - obs.zone_tt[z, reach] / 600)])
                out.append((int(d), tgt))
                deficit[tgt] -= 1
                deficit[z] += 1
        return out


def block_move(obs, act, n: int, rng=None, max_tt: float = 1200.0):
    """Add one concentrated repositioning order to `act`: n free idle drivers from a crowded
    area (the zone with most of them, or a random zone weighted by them when `rng` is given,
    plus its nearest zones until n drivers) all sent to one zone within max_tt seconds (the
    most under-supplied one, or a random one when `rng` is given). Returns a record."""
    used = {d for d, _ in act.offers} | {d for d, _ in act.dispatches}
    free = obs.idle[~obs.idle.driver_id.isin(used)]
    Z = len(obs.recent_demand)
    cnt = np.bincount(free.zone, minlength=Z)
    if cnt.sum() == 0:
        return {}
    src = int(cnt.argmax()) if rng is None else int(rng.choice(Z, p=cnt / cnt.sum()))
    near = np.argsort(obs.zone_tt[src])
    take = near[: int(np.searchsorted(np.cumsum(cnt[near]), n)) + 1]
    need = obs.recent_demand / obs.recent_window_s * 600 + np.bincount(obs.pending.origin_zone, minlength=Z)
    deficit = need - (np.bincount(obs.idle.zone, minlength=Z) + obs.repositioning_to)
    reach = np.flatnonzero((obs.zone_tt[src] <= max_tt) & ~np.isin(np.arange(Z), take))
    if not len(reach):
        return {}
    dst = int(reach[np.argmax(deficit[reach])]) if rng is None else int(rng.choice(reach))
    movers = free[free.zone.isin(take)].sort_values("idle_since").driver_id.to_numpy()[:n]
    act.dispatches = act.dispatches + [(int(d), dst) for d in movers]
    return {"t": float(obs.t), "src_zone": src, "src_zones": [int(z) for z in take], "dst_zone": dst,
            "n_moved": int(len(movers)), "idle_in_src": int(cnt[take].sum()), "deficit_dst": float(deficit[dst]),
            "travel_s": float(obs.zone_tt[src, dst])}


class HDDispatchBlocks(HDDispatch):
    """HDDispatch plus random concentrated orders between 05:00 and 24:00 (on average one
    every `every_s` seconds, each moving 10..n_max drivers to a random zone within 20 min).
    Used for training data that covers strong actions, which the rule above rarely takes."""

    name = "hd_blocks"

    def __init__(self, n_max: int = 60, every_s: float = 2400.0, **kw):
        super().__init__(**kw)
        self.n_max, self.every_s = n_max, every_s
        self.next_t = 5 * 3600 + self.rng.exponential(every_s)
        self.blocks = []

    def act(self, obs):
        act = super().act(obs)
        if obs.t >= self.next_t:
            self.next_t = obs.t + self.rng.exponential(self.every_s)
            rec = block_move(obs, act, int(self.rng.integers(10, self.n_max + 1)), rng=self.rng)
            if rec:
                self.blocks.append(rec)
            act.info["block"] = rec.get("n_moved", 0)
        return act
