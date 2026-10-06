"""The ride company's people and demand: drivers, customers, requests, prices.

Everything a real platform would *record* about a person goes in the profile tables
(`drivers`, `customers`). The behavioural parameters that drive their choices
(willingness to accept, compliance, patience, rating habits) go in separate *latent*
tables. They are the hidden factors Z_t of the formulation: a world model is trained on
the profiles and events, never on these, so they can be used to check what it
recovered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cities import City
from .traffic import gravity_od, hour_mix

VEHICLES = pd.DataFrame({
    # ASSUMPTION: fleet mix and per-type cost multipliers on the city's cost_per_km.
    "vehicle_type": ["hatchback", "sedan", "suv", "ev"],
    "capacity": [4, 4, 6, 4],
    "cost_mult": [0.9, 1.0, 1.35, 0.45],
    "share": [0.40, 0.35, 0.10, 0.15],
})


def _lognormal(rng, median, sigma, size):
    return median * np.exp(sigma * rng.standard_normal(size))


# ---------------------------------------------------------------- people

def make_drivers(city: City, zones: pd.DataFrame, home_w: np.ndarray, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = city.market
    n = m.n_drivers
    veh = VEHICLES.sample(n, replace=True, weights="share", random_state=rng.integers(1 << 31)).reset_index(drop=True)
    tenure = rng.gamma(1.2, 400, n).round()                   # days on the platform
    quality = np.clip(rng.normal(4.75, 0.15, n), 3.8, 5.0)    # latent service quality
    # Past ratings (prior history) scale with tenure; observed mean tracks quality.
    n_ratings = rng.poisson(tenure * 1.5)
    rating_sum = np.round(n_ratings * np.clip(quality + rng.normal(0, 0.05, n), 1, 5))
    home_zone = rng.choice(len(zones), size=n, p=home_w)
    starts, weights = zip(*m.shift_starts)
    shift_type = rng.choice(len(starts), size=n, p=np.array(weights) / sum(weights))
    profile = pd.DataFrame({
        "driver_id": np.arange(n),
        "vehicle_type": veh.vehicle_type, "capacity": veh.capacity,
        "cost_per_km": np.round(m.cost_per_km * veh.cost_mult, 4),   # kappa_i, known to the platform
        "tenure_days": tenure.astype(int),
        "rating_sum": rating_sum, "rating_count": n_ratings,
        "home_zone": home_zone,
        "usual_shift_start_h": np.array(starts)[shift_type],
    })
    latent = pd.DataFrame({
        "driver_id": np.arange(n),
        # logit intercept of offer acceptance; slopes are shared (see Behaviour)
        "accept_bias": rng.normal(2.2, 0.8, n),
        "reposition_compliance": rng.beta(6, 2, n),
        "service_quality": quality,
        "shift_len_median_h": _lognormal(rng, m.shift_median_h, 0.25, n),
        "days_per_week": rng.choice([4, 5, 6, 7], size=n, p=[0.15, 0.35, 0.35, 0.15]),
    })
    return profile, latent


def make_customers(city: City, zones: pd.DataFrame, home_w: np.ndarray, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = city.market.n_customers
    tenure = rng.gamma(1.0, 500, n).round()
    leniency = rng.normal(0, 0.3, n)
    n_ratings = rng.poisson(tenure * 0.05)
    profile = pd.DataFrame({
        "customer_id": np.arange(n),
        "home_zone": rng.choice(len(zones), size=n, p=home_w),
        "tenure_days": tenure.astype(int),
        "rating_sum": np.round(n_ratings * np.clip(4.85 + 0.2 * leniency, 1, 5)),
        "rating_count": n_ratings,
    })
    latent = pd.DataFrame({
        "customer_id": np.arange(n),
        # heavy-tailed usage: a few customers ride very often
        "activity": rng.pareto(1.5, n) + 0.2,
        "patience_scale": _lognormal(rng, 1.0, 0.35, n),
        "tip_propensity": rng.beta(1.5, 3.5, n),
        "rating_leniency": leniency,
        "behaviour_score": np.clip(rng.normal(4.85, 0.12, n), 3.5, 5.0),  # how drivers rate them
    })
    return profile, latent


# ---------------------------------------------------------------- behaviour constants

@dataclass(frozen=True)
class Behaviour:
    """Shared behavioural slopes (ASSUMPTION). Per-person intercepts live in latents."""
    accept_eta_per_min: float = -0.18     # logit change per minute of pickup ETA
    accept_fare_per_unit: float = 0.08    # per unit of fare / (city min fare)
    accept_intercity: float = -0.6        # long trips out of town are less liked
    accept_night: float = -0.3            # 0-5 h
    offer_response_s: tuple = (4.0, 15.0)   # uniform response delay
    offer_timeout_s: float = 15.0
    patience_pre_min: float = 5.0         # median wait before a match, times patience_scale
    patience_post_min: float = 12.0       # tolerated pickup ETA after acceptance
    random_cancel_prob: float = 0.03      # cancels after acceptance regardless of ETA
    boarding_s: tuple = (20.0, 90.0)
    cancel_fee_grace_s: float = 120.0
    rate_driver_prob: float = 0.65
    rate_customer_prob: float = 0.85
    rating_delay_mean_s: float = 2400.0
    tip_delay_mean_s: float = 1800.0
    platform_max_wait_s: float = 900.0    # a request unmatched this long expires
    ping_delay_s: tuple = (1.0, 6.0)      # GPS / app events reach the platform after this


BEHAVIOUR = Behaviour()


# ---------------------------------------------------------------- tariff

def fare(city: City, km: float, minutes: float, outside_km: float = 0.0) -> float:
    m = city.market
    f = m.base_fare + m.per_km * km + m.per_min * minutes + m.intercity_per_km * outside_km
    return float(round(max(m.min_fare, f), 2))


# ---------------------------------------------------------------- demand

def zone_node_sampler(nodes: pd.DataFrame, edges: pd.DataFrame):
    """For each zone, candidate pickup/drop-off nodes and weights. Freeway nodes are
    excluded; nodes on bigger streets are a little more likely (people walk to them)."""
    w_cls = {"motorway": 0.0, "trunk": 0.5, "primary": 1.5, "secondary": 1.5, "tertiary": 1.3,
             "residential": 1.0, "unclassified": 1.0, "living_street": 0.8, "service": 0.6, "other": 0.5}
    we = edges.road_class.map(w_cls)
    wn = pd.concat([we.groupby(edges.u).max(), we.groupby(edges.v).max()]).groupby(level=0).max()
    w = wn.reindex(nodes.node_id).fillna(0).to_numpy()
    out = {}
    for z, grp in nodes.groupby("zone"):
        idx = grp.index.to_numpy()
        ww = w[idx]
        out[int(z)] = (idx, ww / ww.sum())
    return out


def make_requests(city: City, date, zones, customers, cust_latent, zone_nodes, home_w, act_w,
                  gateways_pos: dict, rng) -> pd.DataFrame:
    """One day of requests (times in seconds since midnight)."""
    m, tgt = city.market, city.traffic
    date = pd.Timestamp(date)
    weekend = date.weekday() >= 5
    prof = np.array(m.demand_weekend if weekend else m.demand_weekday, float)
    total = m.requests_per_day * (m.weekend_factor if weekend else 1.0)
    lam = total * prof / prof.sum()
    daytype = "weekend" if weekend else "weekday"

    am = gravity_od(zones, home_w, act_w, m.ride_gravity_km)
    am /= am.sum()
    pats = {"am": am, "pm": am.T, "off": 0.5 * (am + am.T)}
    Z = len(zones)

    # customers indexed by home zone, weighted by activity
    act = cust_latent.activity.to_numpy()
    by_zone = {z: g.index.to_numpy() for z, g in customers.groupby("home_zone")}
    all_p = act / act.sum()

    rows = []
    for h in range(24):
        n = rng.poisson(lam[h])
        if n == 0:
            continue
        mix = hour_mix(h + 0.5, daytype, tgt)
        od = sum(w * pats[k] for k, w in mix.items())
        flat = od.ravel() / od.sum()
        pairs = rng.choice(Z * Z, size=n, p=flat)
        oz, dz = pairs // Z, pairs % Z
        t = h * 3600 + rng.uniform(0, 3600, n)
        for i in range(n):
            o = int(oz[i])
            idx, p = zone_nodes[o]
            onode = int(rng.choice(idx, p=p))
            if rng.random() < m.intercity_share:
                k = rng.choice(len(m.external), p=[e.share for e in m.external])
                dest_ext, dnode, dzone = int(k), int(gateways_pos[k]), -1
            else:
                d = int(dz[i])
                idx, p = zone_nodes[d]
                dnode = int(rng.choice(idx, p=p))
                while dnode == onode:
                    dnode = int(rng.choice(idx, p=p))
                dest_ext, dzone = -1, d
            # customer: often someone living at one end of the trip
            home = o if rng.random() < 0.5 else (dzone if dzone >= 0 else o)
            pool = by_zone.get(home)
            if pool is not None and rng.random() < 0.7:
                pp = act[pool] / act[pool].sum()
                cust = int(rng.choice(pool, p=pp))
            else:
                cust = int(rng.choice(len(act), p=all_p))
            rows.append((t[i], cust, o, onode, dzone, dnode, dest_ext))
    req = pd.DataFrame(rows, columns=["t_request", "customer_id", "origin_zone", "origin_node",
                                      "dest_zone", "dest_node", "external_dest"])
    req = req.sort_values("t_request").reset_index(drop=True)
    req.insert(0, "request_id", np.arange(len(req)))
    pat = cust_latent.patience_scale.to_numpy()[req.customer_id]
    b = BEHAVIOUR
    req["patience_pre_s"] = 60 * b.patience_pre_min * pat * np.exp(0.3 * rng.standard_normal(len(req)))
    req["patience_post_s"] = 60 * b.patience_post_min * pat * np.exp(0.3 * rng.standard_normal(len(req)))
    return req
