"""World model 3, stage 2: model inputs and targets for one decision epoch.

At boundary tau_t every entity becomes one row of numbers (what the platform knows) and the
events of (tau_t, tau_t + 30 s] become its targets:

  online drivers   D_COLS; targets: own decision (EV: none / go offline / break / head home /
                   reposition on own initiative / end break / re-enter the city / abandon a
                   repositioning), the zone chosen when repositioning on own initiative, the
                   answer to an offer made at tau_t (accept / reject / no answer), the answer to
                   a repositioning order made at tau_t (follow / decline)
  offline drivers  F_COLS; target: comes online
  open requests    R_COLS; targets: the rider cancels; the rider boards (driver waiting)
  zones            Z_COLS; targets: new requests per mode, and each new request's origin
                   node, destination (zone + node, or a gateway out of the city) and whether
                   the rider declined the quote
  global           G_COLS: time of day, weekday, rain, city-wide congestion

Movement along a planned route, arrivals, fares, surge and request expiry are platform or
physical rules (the F of eq. 13) and are not targets. Hidden driver traits are never inputs;
a driver's history on earlier days (PRIOR, the platform's long-term records) is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import state as S

SCALE = 10_000.0                 # metres per position unit
EV = ["none", "offline", "break", "heads_home", "self_reposition", "break_end", "reenter", "abandon_reposition"]
EV_TYPES = {"driver_offline": 1, "driver_break": 2, "driver_heads_home": 3, "driver_self_reposition": 4,
            "driver_break_end": 5, "reenter_city": 6, "reposition_abandoned": 7}
OFFER_RESP = ["accepted", "rejected", "timeout"]
PRIOR = ["days_online_log", "online_share", "offers_log", "accept_rate", "orders_log", "follow_rate",
         "hours_per_day", "breaks_per_day", "trips_per_day"]
D_COLS = ([f"st_{s}" for s in S.STATUS] + ["x", "y", "moving"] + [f"leg_{k}" for k in S.LEG_KINDS]
          + ["dest_dx", "dest_dy", "left_km", "eta_left_min", "in_status_log", "online_h"]
          + [f"n_{c}_log" for c in S.COUNTERS] + ["income", "acc_rate_today", "follow_rate_today"]
          + [f"prior_{p}" for p in PRIOR] + [f"mode_{m}" for m in S.MODES]
          + ["cost", "tenure_log", "rating", "rating_n_log", "home_dx", "home_dy", "shift_offset_h"]
          + ["has_req", "req_waited_min"]
          + ["offer", "offer_eta_min", "offer_trip_km", "offer_quote", "offer_waited_min", "offer_dest_dx", "offer_dest_dy", "offer_surge"]
          + ["order", "order_dx", "order_dy", "order_km", "order_tt_min"] + ["surge_here"])
F_COLS = (["never_online_today", "hours_since_off", "online_h_today", "trips_today_log", "shift_offset_h", "x", "y"]
          + [f"mode_{m}" for m in S.MODES] + ["tenure_log"] + [f"prior_{p}" for p in PRIOR])
R_COLS = ([f"rs_{s}" for s in S.R_OPEN] + ["x", "y", "dest_dx", "dest_dy", "external", "trip_km"]
          + [f"mode_{m}" for m in S.MODES] + ["waited_min", "n_offers", "quote", "surge", "drv_eta_min", "at_pickup_min"])
Z_STATIC = ["area_km2", "road_km", "n_pois", "poi_amenity", "poi_office", "poi_shop", "poi_public_transport", "poi_tourism"]
Z_COLS = (["x", "y"] + [f"s_{c}" for c in Z_STATIC] + ["req_5m", "req_15m", "req_60m"] + [f"req15_{m}" for m in S.MODES]
          + ["pending"] + [f"idle_{m}" for m in S.MODES] + ["busy", "repo_to"] + [f"surge_{m}" for m in S.MODES] + ["road_ratio"])
G_COLS = ["h_sin", "h_cos", "h2_sin", "h2_cos"] + [f"dow_{i}" for i in range(7)] + ["rain", "congestion", "city"]
NEW_COLS = ["zone", "origin", "dest_zone", "dest_node", "external", "mode", "declined", "quote", "surge", "eta_min", "trip_km"]


def _log(x):
    return np.log1p(np.maximum(x, 0))


class Prior:
    """Per-driver records from the earlier days of a season (the platform's long-term summaries)."""

    KEYS = ["days_seen", "days_online", "offers", "accepted", "orders", "followed", "hours", "breaks", "trips"]

    def __init__(self, n=0):
        self.a = {k: np.zeros(n) for k in self.KEYS}

    def _grow(self, n):
        for k, v in self.a.items():
            if len(v) < n:
                self.a[k] = np.r_[v, np.zeros(n - len(v))]

    def features(self, ids):
        self._grow(int(ids.max()) + 1 if len(ids) else 0)
        a = {k: v[ids] for k, v in self.a.items()}
        on = np.maximum(a["days_online"], 1)
        return np.stack([_log(a["days_online"]), a["days_online"] / np.maximum(a["days_seen"], 1), _log(a["offers"]),
                         (a["accepted"] + 1) / (a["offers"] + 1.25), _log(a["orders"]), (a["followed"] + 1) / (a["orders"] + 1.35),
                         a["hours"] / on / 8, a["breaks"] / on, a["trips"] / on / 10], 1)

    def update_light(self, day_dir, nD):
        """Same as update() from the event and status tables only (no DayLog needed)."""
        import pandas as pd
        self._grow(nD)
        ev = pd.read_parquet(day_dir / "events.parquet", columns=["type", "driver_id"])
        ev = ev[ev.driver_id >= 0]
        cnt = lambda typ: np.bincount(ev.driver_id[ev.type == typ], minlength=nD)[:nD]  # noqa: E731
        self.a["days_seen"][:nD] += 1
        self.a["days_online"][:nD] += cnt("driver_online") > 0
        for k, typ in (("offers", "offer_sent"), ("accepted", "offer_accepted"), ("orders", "dispatch_sent"),
                       ("followed", "dispatch_accepted"), ("breaks", "driver_break"), ("trips", "dropoff")):
            self.a[k][:nD] += cnt(typ)
        st = pd.read_parquet(day_dir / "status.parquet").sort_values(["driver_id", "t"], kind="stable")
        nxt = st.groupby("driver_id").t.shift(-1).fillna(86400.0).clip(upper=86400.0)
        on = st.status != 0
        self.a["hours"][:nD] += np.bincount(st.driver_id[on], weights=(nxt - st.t)[on], minlength=nD)[:nD] / 3600

    def update(self, L):
        """Add one finished day."""
        self._grow(L.nD)
        ev = L.ev[L.ev.driver_id >= 0]
        cnt = lambda typ: np.bincount(ev.driver_id[ev.type == typ], minlength=L.nD)[:L.nD]  # noqa: E731
        self.a["days_seen"][:L.nD] += 1
        self.a["days_online"][:L.nD] += cnt("driver_online") > 0
        for k, typ in (("offers", "offer_sent"), ("accepted", "offer_accepted"), ("orders", "dispatch_sent"),
                       ("followed", "dispatch_accepted"), ("breaks", "driver_break"), ("trips", "dropoff")):
            self.a[k][:L.nD] += cnt(typ)
        self.a["hours"][:L.nD] += (L.status != 0).sum(1) * S.DT / 3600


class Featurizer:
    """Snapshots of one day (a state.DayLog) as arrays, with the targets of the next 30 s."""

    def __init__(self, L: S.DayLog, prior: Prior, city_id: int, zones: pd.DataFrame, zone_tt: np.ndarray):
        g = L.g
        self.L, self.g, self.prior, self.city_id = L, g, prior, city_id
        self.Z = len(zones)
        self.cx, self.cy = float(np.median(g.x)), float(np.median(g.y))
        self.zx, self.zy = zones.x_m.to_numpy(), zones.y_m.to_numpy()
        zs = zones[Z_STATIC].to_numpy(float)
        self.z_static = _log(zs)
        self.zone_tt = zone_tt                                     # (2, 24, Z, Z) typical seconds
        self.daytype = int(L.date.weekday() >= 5)
        self.money = float(np.median(L.req.quote))                 # currency unit: a typical fare
        legs = L.legs.sort_values("leg_id")
        self.leg_tend = legs.t_end.to_numpy()
        self.d_mode = L.drivers["mode"].map({m: i for i, m in enumerate(S.MODES)}).to_numpy()
        self.d_home = L.drivers.home_node.to_numpy()
        p = L.drivers
        self.d_static = np.stack([p.cost_per_km / np.median(p.cost_per_km), _log(p.tenure_days) / 7,
                                  np.where(p.rating_count > 0, p.rating_sum / p.rating_count.clip(lower=1) - 4.5, 0.0), _log(p.rating_count) / 7], 1)
        self.d_shift = p.usual_shift_start_h.to_numpy()
        self.prior_d = prior.features(np.arange(L.nD))             # fixed for the day (earlier days only)
        # surge per 5-min slot (the log holds multipliers above 1)
        sg = pd.read_parquet(L.dir / "surge.parquet")
        n_slot = int(86400 / 300) + 13
        self.surge = np.ones((n_slot, self.Z, 3), np.float32)
        if len(sg):
            sl = (sg.t.to_numpy() // 300).astype(int)
            mo = sg["mode"].map({m: i for i, m in enumerate(S.MODES)}).to_numpy()
            ok = sl < n_slot
            self.surge[sl[ok], sg.zone.to_numpy()[ok], mo[ok]] = sg.multiplier.to_numpy()[ok]
        # requests (all of them, incl. declined quotes): cumulative counts per zone and mode
        q = L.req.sort_values("request_id")
        self.q_t, self.q_zone = q.t_request.to_numpy(), q.origin_zone.to_numpy()
        self.q_mode = q["mode"].map({m: i for i, m in enumerate(S.MODES)}).to_numpy()
        self.q_dzone = q.dest_zone.to_numpy()
        self.q_surge, self.q_eta = q.surge.to_numpy(), q.eta_shown_s.to_numpy()
        self.q_declined = (q.status == "declined").to_numpy()
        self.q_quote = q.quote.to_numpy()
        # the stored quote includes surge only when the rider accepted it (world/hd/sim.py on_request):
        # the pre-surge quote, the same for all, so the column cannot give the answer away
        self.q_base = np.where(self.q_declined, self.q_quote, self.q_quote / np.maximum(self.q_surge, 1.0))
        self.q_epoch = np.ceil(self.q_t / S.DT).astype(int) - 1         # the step (e, e+1] it falls in
        ox, oy = g.x[L.r_origin], g.y[L.r_origin]
        dn = np.where(L.r_dest >= 0, L.r_dest, L.r_origin)
        self.q_km = np.where(L.r_dest >= 0, np.hypot(g.x[dn] - ox, g.y[dn] - oy) / 1000, -1)
        # events by step
        ev = L.ev
        self.ev_epoch = np.ceil(ev.t_event.to_numpy() / S.DT).astype(int) - 1
        home = ev[ev.type == "driver_home"][["driver_id", "t_event"]]
        drop = ev.type.eq("driver_offline") & pd.MultiIndex.from_frame(ev[["driver_id", "t_event"]]).isin(
            pd.MultiIndex.from_frame(home))                              # arriving home: a rule, not a choice
        self.ev_type = ev.type.to_numpy()
        self.ev_drv, self.ev_req = ev.driver_id.to_numpy(), ev.request_id.to_numpy()
        self.ev_zone, self.ev_val = ev.zone.to_numpy(), ev.value.to_numpy()
        self.ev_t = ev.t_event.to_numpy()
        self.ev_skip = drop.to_numpy()
        self.by_epoch = pd.Series(np.arange(len(ev))).groupby(self.ev_epoch).apply(np.array).to_dict()
        o = L.offers
        self.o_epoch = np.round(o.t_sent.to_numpy() / S.DT).astype(int)
        self.o_by_epoch = pd.Series(np.arange(len(o))).groupby(self.o_epoch).apply(np.array).to_dict()
        self.o_drv, self.o_req = o.driver_id.to_numpy(), o.request_id.to_numpy()
        self.o_resp = o.response.map({r: i for i, r in enumerate(OFFER_RESP)}).fillna(2).to_numpy().astype(int)
        self.o_eta = o.eta_est_s.to_numpy()
        # roads: length-weighted mean travel-time ratio per zone and city-wide, per 5-min slot
        self.road_zone = {}
        self.ezone = g.zone[g.eu]

    def _road(self, e):
        slot = int(e * S.DT // 300)
        if slot not in self.road_zone:
            r = self.L.road_ratio(e)
            w = self.g.length
            num = np.bincount(self.ezone, weights=r * w, minlength=self.Z)
            den = np.bincount(self.ezone, weights=w, minlength=self.Z)
            self.road_zone[slot] = (num / np.maximum(den, 1), float((r * w).sum() / w.sum()))
        return self.road_zone[slot]

    # ------------------------------------------------------------------ one epoch
    def snapshot(self, e):
        L, g = self.L, self.g
        t = e * S.DT
        h = t / 3600
        slot = min(int(t // 300), len(self.surge) - 1)
        surge = self.surge[slot]
        rz, cong = self._road(e)
        dow = np.eye(7)[L.date.weekday()]
        rain = self.L.day.rain_at(t % 86400)
        glob = np.r_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(4 * np.pi * h / 24),
                     np.cos(4 * np.pi * h / 24), dow, rain, cong - 1, self.city_id].astype(np.float32)
        evi = self.by_epoch.get(e, np.zeros(0, int))
        st = L.status[:, e]
        on = np.flatnonzero(st != 0)
        off = np.flatnonzero(st == 0)

        # ---- online drivers
        xy = L.xy(e)[on]
        leg = L.leg[on, e]
        mv = leg >= 0
        lk = np.where(mv, L.leg_kind[np.maximum(leg, 0)], -1)
        dest = np.where(mv, L.leg_dest[np.maximum(leg, 0)], L.stop[on, e])
        rel = lambda nodes: np.c_[(g.x[nodes] - xy[:, 0]) / SCALE, (g.y[nodes] - xy[:, 1]) / SCALE]  # noqa: E731
        eta_left = np.where(mv, (self.leg_tend[np.maximum(leg, 0)] - t) / 60, 0)
        cnt = {c: L.count[c][on, e].astype(float) for c in S.COUNTERS}
        in_st = t - L.status_since[on, e]
        online_h = np.nan_to_num((t - L.online_since[on, e]) / 3600)
        req_of = np.where(mv, L.leg_req[np.maximum(leg, 0)], -1)
        has_req = req_of >= 0
        req_wait = np.where(has_req & (lk == 0), (t - np.nan_to_num(L.r_created[np.maximum(req_of, 0)], nan=t)) / 60, 0)   # rider waiting for this pickup
        home = rel(self.d_home[on])
        sh = (h - self.d_shift[on] + 12) % 24 - 12
        dz = g.zone[L.stop[on, e]]
        mode = self.d_mode[on]
        # actions at tau: offers and repositioning orders
        pos = pd.Series(np.arange(len(on)), index=on)
        off_feat = np.zeros((len(on), 8))
        offer_lab = np.full(len(on), -1)
        oi = self.o_by_epoch.get(e, np.zeros(0, int))
        if len(oi):
            k = pos.reindex(self.o_drv[oi]).to_numpy()
            ok = ~np.isnan(k)
            k, oi = k[ok].astype(int), oi[ok]
            r = self.o_req[oi]
            o_x, o_y = g.x[L.r_origin[r]], g.y[L.r_origin[r]]
            dn = np.where(L.r_dest[r] >= 0, L.r_dest[r], L.r_origin[r])
            off_feat[k] = np.c_[np.ones(len(k)), self.o_eta[oi] / 60, np.maximum(self.q_km[r], 0), L.r_quote[r] / self.money,
                                (t - L.r_created[r]) / 60, (g.x[dn] - o_x) / SCALE, (g.y[dn] - o_y) / SCALE,
                                self.q_surge[r]]
            offer_lab[k] = self.o_resp[oi]
        ord_feat = np.zeros((len(on), 5))
        order_lab = np.full(len(on), -1)
        # orders are issued exactly at tau (an event at t = tau belongs to the previous step's list)
        ds = np.flatnonzero((self.ev_type == "dispatch_sent") & (self.ev_t == t))
        if len(ds):
            k = pos.reindex(self.ev_drv[ds]).to_numpy()
            ok = ~np.isnan(k)
            k, ds = k[ok].astype(int), ds[ok]
            z = self.ev_zone[ds]
            hh = int(h) % 24
            ord_feat[k] = np.c_[np.ones(len(k)), (self.zx[z] - xy[k, 0]) / SCALE, (self.zy[z] - xy[k, 1]) / SCALE,
                                np.hypot(self.zx[z] - xy[k, 0], self.zy[z] - xy[k, 1]) / 1000,
                                self.zone_tt[self.daytype, hh, dz[k], z] / 60]
            resp = {}
            for i in evi:
                if self.ev_type[i] in ("dispatch_accepted", "dispatch_declined"):
                    resp[self.ev_drv[i]] = int(self.ev_type[i] == "dispatch_declined")
            order_lab[k] = [resp.get(d, -1) for d in self.ev_drv[ds]]
        D = np.column_stack([
            np.eye(8)[st[on]], (xy[:, 0] - self.cx) / SCALE, (xy[:, 1] - self.cy) / SCALE, mv,
            (lk[:, None] == np.arange(len(S.LEG_KINDS))[None]), rel(dest), L.left_m[on, e] / 1000, eta_left,
            _log(in_st / 60), online_h, *[_log(cnt[c]) for c in S.COUNTERS],
            L.income[on, e] / self.money / 10, (cnt["offers"] - cnt["offers_rejected"] - cnt["offers_timeout"] + 1) / (cnt["offers"] + 1.25),
            (cnt["orders"] - cnt["orders_declined"] + 1) / (cnt["orders"] + 1.35),
            self.prior_d[on], np.eye(3)[mode], self.d_static[on], home, sh, has_req, req_wait,
            off_feat, ord_feat, surge[dz, mode]]).astype(np.float32)
        # ---- driver targets
        ev_lab = np.zeros(len(on), np.int8)
        repo_zone = np.full(len(on), -1, np.int16)
        on_lab = np.zeros(len(off), np.int8)
        posf = pd.Series(np.arange(len(off)), index=off)
        for i in evi:
            ty, d = self.ev_type[i], self.ev_drv[i]
            if ty in EV_TYPES and not self.ev_skip[i] and d in pos.index:
                k = pos[d]
                if ev_lab[k] == 0:
                    ev_lab[k] = EV_TYPES[ty]
                    if ty == "driver_self_reposition":
                        repo_zone[k] = int(self.ev_val[i])
            elif ty == "driver_online" and d in posf.index:
                on_lab[posf[d]] = 1

        # ---- offline drivers
        ever = ~np.isnan(L.online_since[off, e])
        last_off = np.full(len(off), np.nan)
        stl = L.status_since[off, e]
        last_off[ever] = stl[ever]
        hrs = np.where(ever, (t - np.nan_to_num(last_off)) / 3600, h) / 24
        sh_f = (h - self.d_shift[off] + 12) % 24 - 12
        hx = g.x[self.d_home[off]]
        hy = g.y[self.d_home[off]]
        Fm = np.column_stack([~ever, hrs, (L.status[off, :e + 1] != 0).sum(1) * S.DT / 3600 / 8,
                              _log(L.count["trips"][off, e]), sh_f / 12, (hx - self.cx) / SCALE, (hy - self.cy) / SCALE,
                              np.eye(3)[self.d_mode[off]], self.d_static[off, 1], self.prior_d[off]]).astype(np.float32)

        # ---- open requests
        R = L.open_requests(e)
        ri = R.request_id.to_numpy()
        o = L.r_origin[ri]
        dn = np.where(L.r_dest[ri] >= 0, L.r_dest[ri], o)
        drv_eta = np.zeros(len(ri))
        at_pick = np.where(R.status.to_numpy() == 3, (t - np.nan_to_num(L.r_arrive[ri], nan=t)) / 60, 0)
        if len(ri) and has_req.any():
            m = pd.Series(eta_left[has_req], index=req_of[has_req])
            m = m[~m.index.duplicated()]
            drv_eta = m.reindex(ri).fillna(0).to_numpy()
        Rm = np.column_stack([np.eye(4)[R.status.to_numpy()], (g.x[o] - self.cx) / SCALE, (g.y[o] - self.cy) / SCALE,
                              (g.x[dn] - g.x[o]) / SCALE, (g.y[dn] - g.y[o]) / SCALE, L.r_dest[ri] < 0,
                              np.maximum(self.q_km[ri], 0), np.eye(3)[L.r_mode[ri]], R.waited_s.to_numpy() / 60,
                              R.n_offers.to_numpy(), L.r_quote[ri] / self.money, self.q_surge[ri], drv_eta, at_pick]).astype(np.float32)
        r_cancel = np.zeros(len(ri), np.int8)
        r_board = np.zeros(len(ri), np.int8)
        rpos = pd.Series(np.arange(len(ri)), index=ri)
        for i in evi:
            ty, q = self.ev_type[i], self.ev_req[i]
            if q in rpos.index:
                if ty == "customer_cancelled":
                    r_cancel[rpos[q]] = 1
                elif ty == "pickup":
                    r_board[rpos[q]] = 1

        # ---- zones
        Z = self.Z
        past = (self.q_t <= t)
        win = lambda m: np.bincount(self.q_zone[past & (self.q_t > t - m * 60)], minlength=Z)  # noqa: E731
        r15 = np.stack([np.bincount(self.q_zone[past & (self.q_t > t - 900) & (self.q_mode == j)], minlength=Z) for j in range(3)], 1)
        pend = np.bincount(g.zone[o[R.status.to_numpy() <= 1]], minlength=Z)
        stop_z = g.zone[L.stop[:, e]]
        idle = np.stack([np.bincount(stop_z[(st == 1) & (self.d_mode == j)], minlength=Z) for j in range(3)], 1)
        busy = np.bincount(stop_z[np.isin(st, [2, 3, 4, 5])], minlength=Z)
        repo = np.bincount(g.zone[dest[(lk == 3) | (lk == 4)]], minlength=Z)
        Zm = np.column_stack([(self.zx - self.cx) / SCALE, (self.zy - self.cy) / SCALE, self.z_static,
                              _log(win(5)), _log(win(15)), _log(win(60)), _log(r15), _log(pend), _log(idle), _log(busy), _log(repo),
                              surge, rz - 1]).astype(np.float32)
        # new requests in (tau, tau + 30 s]
        nq = np.flatnonzero(self.q_epoch == e)
        new = np.column_stack([self.q_zone[nq], L.r_origin[nq], self.q_dzone[nq], L.r_dest[nq], L.r_external[nq],
                               self.q_mode[nq], self.q_declined[nq], self.q_base[nq] / self.money, self.q_surge[nq],
                               self.q_eta[nq] / 60, self.q_km[nq]]).astype(np.float32)
        z_new = np.stack([np.bincount(self.q_zone[nq][self.q_mode[nq] == j], minlength=Z) for j in range(3)], 1).astype(np.int16)
        return {"G": glob, "D": D, "D_id": on.astype(np.int32), "D_ev": ev_lab, "D_repo": repo_zone,
                "D_offer": offer_lab.astype(np.int8), "D_order": order_lab.astype(np.int8),
                "F": Fm, "F_id": off.astype(np.int32), "F_on": on_lab, "R": Rm, "R_id": ri.astype(np.int32),
                "R_cancel": r_cancel, "R_board": r_board, "Z": Zm, "Z_new": z_new, "NEW": new}


def pack(snaps, epochs):
    """List of snapshot dicts -> flat arrays with per-snapshot offsets (one file per day)."""
    out = {"epoch": np.asarray(epochs, np.int32)}
    for k in snaps[0]:
        arrs = [s[k] for s in snaps]
        if k in ("G", "Z", "Z_new"):
            out[k] = np.stack(arrs)
        else:
            out[k] = np.concatenate(arrs)
            out[f"{k}_off"] = np.r_[0, np.cumsum([len(a) for a in arrs])].astype(np.int64)
    for k in list(out):
        if out[k].dtype == np.float32 and k not in ("G", "NEW"):          # NEW holds node ids: keep float32
            out[k] = out[k].astype(np.float16)
    return out
