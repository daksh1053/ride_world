"""Drivers and riders of the high-detail cities, with observed profiles and hidden traits.

Observed (what a platform records): id, mode, vehicle, cost per km, tenure, rating
sum / count, home zone, usual shift start.

Hidden (latent Z, never shown to a world model; kept to test what it recovers):
  drivers  acceptance bias, reposition compliance, service quality, shift length,
           weekly work days and which days, strategy (stayer / chaser / homebound mix),
           break interval, reservation hourly earnings (churn threshold)
  riders   riding frequency, patience, tipping, rating leniency, behaviour score,
           price sensitivity, ETA sensitivity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..cities import City
from .demand import MODES

# ASSUMPTION: share of the fleet per product (Pune) and drivers per daily request.
FLEET_SHARE = {"pune_hd": {"bike": 0.30, "auto": 0.35, "cab": 0.35}, "sf_hd": {"cab": 1.0}}
COST_PER_KM = {"bike": 1.5, "auto": 3.0, "cab": 4.5}            # INR/km; SF cab uses market cost
RESERVATION_HOURLY = {"pune_hd": 70.0, "sf_hd": 14.0}          # median, in local currency
# ASSUMPTION: share of drivers who work on each day of the week (Mon..Sun), relative.
DAY_WEIGHT = np.array([0.85, 0.9, 0.95, 1.0, 1.15, 1.15, 0.8])


def _lognormal(rng, median, sigma, size):
    return median * np.exp(sigma * rng.standard_normal(size))


def make_drivers(city: City, n: int, node_home: np.ndarray, node_zone: np.ndarray, rng,
                 start_id: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`node_home`: per-node residential weight (drivers live where people live)."""
    m = city.market
    shares = FLEET_SHARE[city.key]
    modes = rng.choice(list(shares), size=n, p=list(shares.values()))
    home_node = rng.choice(len(node_home), size=n, p=node_home / node_home.sum())
    tenure = rng.gamma(1.2, 400, n).round()
    quality = np.clip(rng.normal(4.75, 0.15, n), 3.8, 5.0)
    n_rat = rng.poisson(tenure * 1.5)
    starts, weights = zip(*m.shift_starts)
    shift = rng.choice(len(starts), size=n, p=np.array(weights) / sum(weights))
    cost = np.array([m.cost_per_km if city.key == "sf_hd" else COST_PER_KM[md] for md in modes])
    ids = np.arange(start_id, start_id + n)
    profile = pd.DataFrame({
        "driver_id": ids, "mode": modes,
        "vehicle_type": [{"bike": "motorbike", "auto": "auto-rickshaw", "cab": "hatchback/sedan"}[md] for md in modes],
        "capacity": [MODES[city.key][md].capacity for md in modes],
        "cost_per_km": np.round(cost * rng.uniform(0.85, 1.15, n), 3),
        "tenure_days": tenure.astype(int),
        "rating_sum": np.round(n_rat * np.clip(quality + rng.normal(0, 0.05, n), 1, 5)),
        "rating_count": n_rat,
        "home_node": home_node, "home_zone": node_zone[home_node],
        "usual_shift_start_h": np.array(starts)[shift],
    })
    # strategy mix: probability of each behaviour when idle
    strat = rng.dirichlet([2.0, 1.2, 0.8], size=n)            # stayer, chaser, homebound
    days = rng.choice([4, 5, 6, 7], size=n, p=[0.15, 0.35, 0.35, 0.15])
    latent = pd.DataFrame({
        "driver_id": ids,
        "accept_bias": rng.normal(2.2, 0.8, n),
        "reposition_compliance": rng.beta(6, 2, n),
        "service_quality": quality,
        "shift_len_median_h": _lognormal(rng, m.shift_median_h, 0.25, n),
        "days_per_week": days,
        "p_stay": strat[:, 0], "p_chase": strat[:, 1], "p_home": strat[:, 2],
        "break_every_h": rng.uniform(2.5, 4.5, n),
        "reservation_hourly": _lognormal(rng, RESERVATION_HOURLY[city.key], 0.3, n),
        "surge_responsiveness": rng.gamma(2.0, 0.5, n),
    })
    return profile, latent


def make_customers(city: City, n: int, node_home: np.ndarray, node_zone: np.ndarray, rng):
    tenure = rng.gamma(1.0, 500, n).round()
    leniency = rng.normal(0, 0.3, n)
    n_rat = rng.poisson(tenure * 0.05)
    home_node = rng.choice(len(node_home), size=n, p=node_home / node_home.sum())
    profile = pd.DataFrame({
        "customer_id": np.arange(n), "home_node": home_node, "home_zone": node_zone[home_node],
        "tenure_days": tenure.astype(int),
        "rating_sum": np.round(n_rat * np.clip(4.85 + 0.2 * leniency, 1, 5)), "rating_count": n_rat,
    })
    latent = pd.DataFrame({
        "customer_id": np.arange(n),
        "activity": rng.pareto(1.5, n) + 0.2,
        "patience_scale": _lognormal(rng, 1.0, 0.35, n),
        "tip_propensity": rng.beta(1.5, 3.5, n),
        "rating_leniency": leniency,
        "behaviour_score": np.clip(rng.normal(4.85, 0.12, n), 3.5, 5.0),
        "price_sensitivity": _lognormal(rng, 1.0, 0.4, n),
        "eta_sensitivity": _lognormal(rng, 1.0, 0.4, n),
        "loyalty": np.ones(n),                      # evolves with experience across days
    })
    return profile, latent
