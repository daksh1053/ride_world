"""Platform policies: the external action source A_t = (offers, dispatches).

A policy sees an `Observation` built only from what the platform knows at decision
time: its records of idle drivers and pending requests, the typical zone-to-zone
travel times for the current hour, and recent demand counts. It never sees realised
traffic, latent behaviour or future requests. It returns offers
(driver -> request) and dispatches (driver -> zone). The simulator enforces
feasibility (eq. 7-8): each request and driver gets at most one instruction, and
only eligible pairs are kept.

The paper methods in ../reproduce plug in by subclassing `Policy`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class Observation:
    t: float                          # seconds since sim start
    hour: int
    idle: pd.DataFrame                # driver_id, node, zone, idle_since, rating
    pending: pd.DataFrame             # request_id, origin_node, origin_zone, dest_zone, t_request, quote, excluded(set)
    zone_tt: np.ndarray               # (Z, Z) typical seconds between zone centres, this hour
    recent_demand: np.ndarray         # requests per zone in the last `recent_window_s`
    recent_window_s: float
    intra_zone_s: np.ndarray          # typical pickup time within a zone
    max_pickup_s: float
    repositioning_to: np.ndarray      # (Z,) cabs currently repositioning towards each zone
    # Platform records for every driver (index = driver_id), read-only: status code,
    # zone, net income today, U_dist today, busy seconds, ever online today, kappa_i;
    # plus epoch index / length, day type and the typical zone distance matrix (m).
    fleet: dict = field(default_factory=dict)


@dataclass
class Action:
    offers: list[tuple[int, int]] = field(default_factory=list)        # (driver_id, request_id)
    dispatches: list[tuple[int, int]] = field(default_factory=list)    # (driver_id, zone)
    info: dict = field(default_factory=dict)


class Policy:
    name = "base"
    version = "0"

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError


def eta_matrix(obs: Observation, drivers: pd.DataFrame, requests: pd.DataFrame) -> np.ndarray:
    """Estimated pickup seconds (drivers x requests) from the typical zone matrix."""
    dz, rz = drivers.zone.to_numpy(), requests.origin_zone.to_numpy()
    eta = obs.zone_tt[dz][:, rz].astype(float)
    same = dz[:, None] == rz[None, :]
    eta[same] = obs.intra_zone_s[dz][:, None].repeat(len(rz), 1)[same]
    return eta


class NearestDispatch(Policy):
    """Baseline: batch-match pending requests to idle drivers by minimum total pickup ETA
    (Hungarian assignment), within the platform's pickup radius; every `reposition_every`
    epochs, send long-idle drivers from zones with surplus to nearby zones short of supply.
    """

    name = "nearest"
    version = "1"

    def __init__(self, reposition_every: int = 10, idle_before_reposition_s: float = 600.0,
                 max_reposition_s: float = 900.0, max_dispatches: int = 60, horizon_s: float = 600.0):
        self.reposition_every = reposition_every
        self.idle_before = idle_before_reposition_s
        self.max_repo_s = max_reposition_s
        self.max_dispatches = max_dispatches
        self.horizon_s = horizon_s
        self._n = 0

    def act(self, obs: Observation) -> Action:
        self._n += 1
        act = Action()
        idle, pend = obs.idle, obs.pending
        used = set()
        if len(idle) and len(pend):
            eta = eta_matrix(obs, idle, pend)
            drv = idle.driver_id.to_numpy()
            for j, ex in enumerate(pend.excluded):
                if ex:
                    eta[np.isin(drv, list(ex)), j] = np.inf
            ok = eta <= obs.max_pickup_s
            if ok.any():
                cost = np.where(ok, eta, 1e9)
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if ok[r, c]:
                        act.offers.append((int(drv[r]), int(pend.request_id.iloc[c])))
                        used.add(int(drv[r]))
        act.info["n_candidates"] = int(len(idle) * len(pend))

        if self._n % self.reposition_every == 0 and len(idle):
            act.dispatches = self._reposition(obs, idle[~idle.driver_id.isin(used)])
        return act

    def _reposition(self, obs: Observation, idle: pd.DataFrame) -> list[tuple[int, int]]:
        Z = len(obs.recent_demand)
        # supply: idle cabs in the zone plus cabs already repositioning towards it
        supply = np.bincount(idle.zone, minlength=Z).astype(float) + obs.repositioning_to
        # need: requests expected in the next `horizon_s` at the recent rate, plus pending
        rate = obs.recent_demand / obs.recent_window_s
        need = rate * self.horizon_s + np.bincount(obs.pending.origin_zone, minlength=Z)
        deficit = need - supply
        long_idle = idle[obs.t - idle.idle_since >= self.idle_before]
        out = []
        for z in np.argsort(-supply):
            if deficit[z] >= 0 or len(out) >= self.max_dispatches:
                continue
            movers = long_idle[long_idle.zone == z].sort_values("idle_since")
            for d in movers.driver_id:
                reach = obs.zone_tt[z] <= self.max_repo_s
                cand = np.flatnonzero(reach & (deficit > 0.5))
                if not len(cand) or deficit[z] >= 0:
                    break
                tgt = int(cand[np.argmax(deficit[cand] - obs.zone_tt[z, cand] / 600.0)])
                out.append((int(d), tgt))
                deficit[tgt] -= 1
                deficit[z] += 1
        return out


POLICIES = {"nearest": NearestDispatch}
