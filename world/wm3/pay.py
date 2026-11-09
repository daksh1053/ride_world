"""What the platform pays a driver for a ride (stage 7b), separate from what the rider pays.

Riders keep paying the city tariff times surge (world/hd/demand.py), so the riders, and the world
model trained on them, are unchanged. Only the driver's side of the split changes, as with upfront
driver pay on real platforms. The baseline scheme is the simulator's: the rider's fare minus the
commission.

A scheme pays, per ride, in the mode's own money (mo = the mode's tariff):

    scale * [ share * fare_before_surge * (1 + surge_mult * (surge - 1))
              + per_trip * mo.min_fare
              + per_km * mo.per_km * km  +  per_min * mo.per_min * minutes
              + pickup_per_min * mo.per_min * max(0, pickup_min - pickup_free_min) ]
    + surge_flat * (surge - 1) * mo.min_fare

km and minutes are the trip's own (inside the city plus the intercity part paid per km), pickup_min
runs from the driver's accept to the rider boarding. 'scale' makes a scheme budget-neutral: the
platform pays drivers in total what the baseline would for the same rides (pay.calibrate).

Everything here works on trip records, from an imagined world (rollout.trip_record) or from the
simulator's logs (pay.sim_trips), so the same scheme prices both.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..hd.demand import MODES as HD_MODES
from ..hd.demand import mode_fare
from . import state as S


@dataclass(frozen=True)
class PayScheme:
    name: str
    share: float = 0.0
    surge_mult: float = 1.0
    per_trip: float = 0.0
    per_km: float = 0.0
    per_min: float = 0.0
    pickup_per_min: float = 0.0
    pickup_free_min: float = 5.0
    surge_flat: float = 0.0
    scale: float = 1.0


def baseline(market):
    return PayScheme("fare_share", share=1 - market.commission)


def _modes(key):
    ms = [HD_MODES[key].get(m) for m in S.MODES]
    pick = lambda f: np.array([getattr(m, f) if m else 1.0 for m in ms])  # noqa: E731
    return {f: pick(f) for f in ("base", "per_km", "per_min", "min_fare")}, ms


def pay(scheme: PayScheme, key, market, mode, km_in, outside_km, minutes, surge, pickup_min):
    """Vectorised pay per ride (numpy arrays or scalars) in the city's money."""
    T, ms = _modes(key)
    mode = np.asarray(mode, int)
    km_in, outside_km, minutes = np.asarray(km_in, float), np.asarray(outside_km, float), np.asarray(minutes, float)
    surge, pickup_min = np.asarray(surge, float), np.asarray(pickup_min, float)
    base = np.maximum(T["min_fare"][mode], T["base"][mode] + T["per_km"][mode] * km_in + T["per_min"][mode] * minutes) \
        + market.intercity_per_km * outside_km
    km = km_in + outside_km
    p = (scheme.share * base * (1 + scheme.surge_mult * (surge - 1))
         + scheme.per_trip * T["min_fare"][mode]
         + scheme.per_km * T["per_km"][mode] * km + scheme.per_min * T["per_min"][mode] * minutes
         + scheme.pickup_per_min * T["per_min"][mode] * np.maximum(0.0, pickup_min - scheme.pickup_free_min))
    return scheme.scale * p + scheme.surge_flat * (surge - 1) * T["min_fare"][mode]


def trips_pay(scheme, key, market, trips: pd.DataFrame):
    if not len(trips):
        return np.zeros(0)
    return pay(scheme, key, market, trips["mode"], trips.km_in, trips.outside_km, trips.minutes, trips.surge,
               (trips.pickup_t - trips.accept_t) / 60)


def calibrate(scheme, key, market, trips: pd.DataFrame):
    """The scheme with 'scale' set so that it pays drivers in total what the baseline does for
    these rides (the flat surge boost is kept as it is and counted in the total)."""
    base = trips_pay(baseline(market), key, market, trips).sum()
    s0 = replace(scheme, scale=1.0, surge_flat=0.0)
    flat = trips_pay(replace(scheme, share=0, per_trip=0, per_km=0, per_min=0, pickup_per_min=0, scale=0.0), key, market,
                     trips).sum()
    raw = trips_pay(s0, key, market, trips).sum()
    return replace(scheme, scale=float((base - flat) / max(raw, 1e-9)))


def in_progress(W, d):
    """The ride driver d is on (accepted, not yet dropped off) in imagined world W, as a trip
    record with its expected km and minutes and 'frac', the share of its engaged time (accept to
    drop-off) already behind the driver; None if the driver has no ride."""
    from . import rollout as RO
    k = int(W.d_req[d])
    if W.st[d] not in (RO.PICKUP, RO.OCC, RO.OUTSIDE) or k not in W.req:
        return None
    q, R, g, t = W.req[k], W.R, W.g, W.t
    h = int(t // 3600) % 24
    oz = int(g.zone[q["origin"]])
    dz = int(g.zone[q["dest"]]) if q["ext"] < 0 else int(g.zone[R.gw_pos[q["ext"]]])
    out_km = R.ext[q["ext"]].outside_km if q["ext"] >= 0 else 0.0
    out_min = R.ext[q["ext"]].outside_min if q["ext"] >= 0 else 0.0
    trip_min = R.zone_tt[W.daytype, h, oz, dz] / 60 * R.speed[q["mode"]] + out_min
    km_in = max(R.zone_dist[W.daytype, h, oz, dz], 800) / 1000
    acc = float(q.get("accept_t", t))
    leg = W.leg[d]
    if W.st[d] == RO.PICKUP:
        arrive = leg.end if leg is not None else t
        end = arrive + 60 + trip_min * 60
        pick_km = leg.length_m / 1000 if leg is not None else float(q.get("pick_km", 0.0))
        pickup_t = arrive + 60
    elif W.st[d] == RO.OCC:
        end = (leg.end if leg is not None else t) + out_min * 60
        pick_km, pickup_t = float(q.get("pick_km", 0.0)), float(q.get("pickup_t", t))
    else:
        end = W.out_drop.get(d, (t, k))[0]
        pick_km, pickup_t = float(q.get("pick_km", 0.0)), float(q.get("pickup_t", t))
    frac = float(np.clip((t - acc) / max(end - acc, 1.0), 0.0, 1.0))
    return {"driver": int(d), "req": k, "mode": int(q["mode"]), "accept_t": acc, "pickup_t": pickup_t, "oz": oz, "dz": dz,
            "ext": int(q["ext"]), "pick_km": pick_km, "km_in": km_in, "outside_km": out_km,
            "minutes": max(end - pickup_t, 60.0) / 60, "surge": float(q["surge"]), "frac": frac}


def focal_record(W, d, t0, km0):
    """What driver d did in imagined world W since t0 (km0 = W.km[d] then): its finished rides,
    the ride in progress (or None), the km driven and the km of the legs of that ride already
    started (counted fully in W.km)."""
    tr = [x for x in getattr(W, "trips", []) if x["driver"] == d and x["t"] > t0]
    cur = in_progress(W, d)
    started = 0.0
    if cur is not None:
        started = cur["pick_km"] if W.st[d] == 3 else cur["pick_km"] + cur["km_in"]       # 3 = PICKUP
    return {"trips": tr, "cur": cur, "km": float(W.km[d] - km0), "km_cur_started": started}


def focal_value(rec, scheme, key, market, kappa):
    """Driver's net earnings in such a record under a pay scheme: finished rides' pay, the ride
    in progress at its pay times the share done, minus the cost of the km, where the ride in
    progress's km also count by the share done (intercity km twice: out and back)."""
    tr = pd.DataFrame(rec["trips"])
    v = float(trips_pay(scheme, key, market, tr).sum()) if len(tr) else 0.0
    km = rec["km"] + (2 * tr.outside_km.sum() if len(tr) else 0.0)      # the way out of town and back (world/sim.py)
    c = rec["cur"]
    if c is not None:
        cdf = pd.DataFrame([c])
        v += c["frac"] * float(trips_pay(scheme, key, market, cdf)[0])
        km += c["frac"] * (c["pick_km"] + c["km_in"] + 2 * c["outside_km"]) - rec["km_cur_started"]
    return v - kappa * km


def fare_before_surge(key, mode, km_in, minutes):
    _, ms = _modes(key)
    return float(mode_fare(ms[mode], km_in, minutes))
