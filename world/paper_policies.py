"""The two reproduced papers' methods as action sources for the ride world.

The method code in ../reproduce is used *unchanged*. Each paper's methods talk to that
paper's own small environment object, so this module provides facade objects that
answer those same calls from the world's `Observation`:

Paper 1 (Kang et al., long-term fairness, `reproduce/paper1_ltf`, package `ltf`)
    Matchers `Greedy, REASSIGN, LAF, Balance Ride-Pooling, MOMAQL` call
    `env.utility_matrix`, `env.feasible_matrix`, `env.available_mask`, `env.fleet.*`.
    - node set L = the city's zones; Geo = typical zone-to-zone road distance (km) of
      the current hour; a driver's utility is U_dist (eq. 20), from platform records;
    - fleet = every driver who has been online today (the cohort so far);
    - one assignment round per 30 s epoch; matching only, no repositioning (paper 1
      has none).

Paper 2 (Shi et al., VFDCFMVD, `reproduce/paper2_vfdcfmvd`, package `vfd`)
    Algorithms `NM, WDF, LAF, ILP, SID, VFDCFMVD` call `platform.*`, `fleet.*`, `graph.*`.
    - a time slot = one 30 s epoch; zones as above; fare p_o = the upfront quote;
      c_d = the driver's known kappa_i; income = net platform-recorded income;
    - `reposition(...)` calls become dispatch instructions (the world decides
      compliance and drives the cab);
    - order waiting limit t_o^w = the platform's 15-min expiry.

What the world adds that neither paper models: drivers may reject offers, customers
cancel, trips take real routed time on congested roads, and income is settled at
drop-off, not at assignment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .policies import Action, Observation, Policy, eta_matrix

# the paper reproductions ship inside this repository (ride_world/reproduce)
REPRO = Path(__file__).resolve().parents[1] / "reproduce"
for sub in ("paper1_ltf", "paper2_vfdcfmvd"):
    if str(REPRO / sub) not in sys.path:
        sys.path.insert(0, str(REPRO / sub))

import ltf.methods as P1          # noqa: E402
from ltf.config import CONFIG as P1_CFG   # noqa: E402
from ltf.simulator import Batch   # noqa: E402
import vfd.methods as P2          # noqa: E402
from vfd.config import CONFIG as P2_CFG   # noqa: E402
from vfd.simulator import OrderBatch      # noqa: E402

IDLE, BUSY = 1, (3, 4, 5, 6)


def zone_km(obs: Observation) -> np.ndarray:
    """Typical zone-centre road distance (km), with an intra-zone distance on the
    diagonal: half the distance to the nearest other zone."""
    d = obs.fleet["zone_dist"] / 1000.0
    d = d.copy()
    off = d + np.diag(np.full(len(d), np.inf))
    np.fill_diagonal(d, 0.5 * off.min(1))
    return d


def feasible(obs: Observation, zones: np.ndarray, pend: pd.DataFrame, drivers: np.ndarray,
             deadline_s: np.ndarray | None = None) -> np.ndarray:
    """(orders x drivers) pickup-ETA and exclusion mask, same rule as the baseline."""
    eta = eta_matrix(obs, pd.DataFrame({"zone": zones}), pend).T
    lim = np.full(len(pend), obs.max_pickup_s)
    if deadline_s is not None:
        lim = np.minimum(lim, deadline_s)
    ok = eta <= lim[:, None]
    for i, ex in enumerate(pend.excluded):
        if ex:
            ok[i, np.isin(drivers, list(ex))] = False
    return ok


# ======================================================================= paper 1

class _P1Fleet:
    def __init__(self, location, utility):
        self.location = location
        self.utility = utility
        self.n = len(utility)
        self.onboard = np.zeros(self.n, dtype=np.int32)
        self.capacity = np.ones(self.n, dtype=np.int32)
        self.n_served = np.zeros(self.n, dtype=np.int64)

    def snapshot(self):
        return {"location": self.location.copy(), "utility": self.utility.copy(),
                "onboard": self.onboard.copy()}


class _P1Env:
    """Answers the `RideHailingEnv` calls a paper-1 matcher makes."""

    def __init__(self, geo_km, fleet, idle, feas):
        self.geo_km, self.fleet, self._idle, self._feas = geo_km, fleet, idle, feas

    def pickup_matrix(self, batch):
        return self.geo_km[np.ix_(self.fleet.location, batch.origin)].T

    def utility_matrix(self, batch):
        profit = self.geo_km[batch.origin, batch.dest]
        return profit[:, None] - self.pickup_matrix(batch)

    def feasible_matrix(self, batch):
        return self._feas

    def available_mask(self):
        return self._idle.copy()


class Paper1Policy(Policy):
    version = "ltf-adapter-1"

    def __init__(self, matcher, name: str):
        self.m, self.name = matcher, name
        self.learn = False

    def set_mode(self, train: bool):
        self.learn = train
        if hasattr(self.m, "training"):
            self.m.training = train

    def act(self, obs: Observation) -> Action:
        f, pend = obs.fleet, obs.pending
        if not len(pend) or not len(obs.idle):
            return Action()
        cohort = np.flatnonzero(f["ever_online"])
        zones = f["zone"][cohort]
        idle = f["status"][cohort] == IDLE
        geo = zone_km(obs)
        feas = feasible(obs, zones, pend, cohort) & idle[None, :]
        fleet = _P1Fleet(zones.copy(), f["udist"][cohort].astype(float).copy())
        env = _P1Env(geo, fleet, idle, feas)
        batch = Batch(time=obs.t, origin=pend.origin_zone.to_numpy().astype(int),
                      dest=pend.dest_zone.to_numpy().astype(int))
        if hasattr(self.m, "n_drivers"):
            self.m.n_drivers = len(cohort)      # MOMAQL sizes its lookahead by the fleet
        decision = self.m.assign_round(env, batch)
        if self.learn:
            # paper 1 books utility at assignment and moves the driver to the drop-off
            prev = fleet.snapshot()
            gained = np.zeros(fleet.n)
            u = env.utility_matrix(batch)
            for i, v in enumerate(decision):
                if v >= 0:
                    gained[v] += u[i, v]
                    fleet.utility[v] += u[i, v]
                    fleet.location[v] = batch.dest[i]
            self.m.observe(env, batch, decision, gained, prev)
        rid = pend.request_id.to_numpy()
        return Action(offers=[(int(cohort[v]), int(rid[i])) for i, v in enumerate(decision) if v >= 0],
                      info={"cohort": int(len(cohort))})


def paper1_policies(n_zones: int) -> dict:
    m = P1_CFG.method
    return {
        "p1_greedy": lambda: Paper1Policy(P1.Greedy(lam=m.lam), "p1_greedy"),
        "p1_reassign": lambda: Paper1Policy(P1.Reassign(slack=m.reassign_utility_slack,
                                                        iters=m.reassign_iters), "p1_reassign"),
        "p1_laf": lambda: Paper1Policy(P1.LAF(lam=m.lam, alpha=m.alpha, gamma=m.gamma), "p1_laf"),
        "p1_brp": lambda: Paper1Policy(P1.BalanceRidePooling(n_zones, lam=m.lam, omega=m.omega,
                                                             gamma=m.gamma, alpha=m.alpha), "p1_brp"),
        "p1_momaql": lambda: Paper1Policy(P1.MOMAQL(1, n_zones, lam=m.lam, omega=m.omega, gamma=m.gamma,
                                                    alpha=m.alpha, epsilon=m.epsilon), "p1_momaql"),
    }


# ======================================================================= paper 2

class _P2Cfg:
    def __init__(self, n_slots, slot_seconds, v_avg_kmh):
        self.n_slots, self.slot_seconds, self.v_avg_kmh = n_slots, slot_seconds, v_avg_kmh


class _P2Graph:
    def __init__(self, km: np.ndarray):
        self.matrix = km
        self._nearby = np.argsort(km + np.diag(np.full(len(km), -1.0)), axis=1)  # self first

    def dis(self, a, b):
        return self.matrix[a, b]

    def nearby(self, zone, k):
        return self._nearby[zone, 1:k + 1]


class _P2Fleet:
    def __init__(self, owner: "Paper2Policy", f: dict, taken: np.ndarray):
        self._o, self._taken = owner, taken
        self.zone = f["zone"].astype(int)
        self.cost = f["cost_per_km"]
        self._idle = f["status"] == IDLE
        self.income_per_slot = owner.income_ps
        self.active_per_slot = owner.active_ps
        self.n_served = owner.n_served

    def available(self, slot):
        return self._idle & ~self._taken

    def dispatchable(self, slot):
        return self._idle & ~self._taken

    def income_upto(self, slot):
        return self._o.cum_income           # == income_per_slot[:, :slot+1].sum(1), kept incrementally

    def income(self):
        return self._o.cum_income


class _P2Platform:
    """Answers the `Platform` calls a paper-2 algorithm makes."""

    def __init__(self, owner: "Paper2Policy", obs: Observation, pend: pd.DataFrame, slot: int):
        f = obs.fleet
        self.cfg = owner.cfg
        self.graph = _P2Graph(zone_km(obs))
        self.taken = np.zeros(len(f["status"]), bool)
        self.fleet = _P2Fleet(owner, f, self.taken)
        self.n_matched = owner.n_matched
        self._obs, self._pend = obs, pend
        self.dispatches: list[tuple[int, int]] = []

    def travel_slots(self, km):
        minutes = np.asarray(km) / self.cfg.v_avg_kmh * 60.0
        return np.maximum(np.ceil(minutes * 60.0 / self.cfg.slot_seconds), 1).astype(int)

    def pickup_distance(self, batch, drivers):
        return self.graph.matrix[np.ix_(self.fleet.zone[drivers], batch.pickup)].T

    def trip_distance(self, batch):
        return self.graph.matrix[batch.pickup, batch.dropoff]

    def cost_matrix(self, batch, drivers):
        total = self.pickup_distance(batch, drivers) + self.trip_distance(batch)[:, None]
        return total * self.fleet.cost[drivers][None, :]

    def profit_matrix(self, batch, drivers):
        return batch.fare[:, None] - self.cost_matrix(batch, drivers)

    def feasible_matrix(self, batch, drivers):
        left = (batch.deadline - batch.slot) * self.cfg.slot_seconds
        return feasible(self._obs, self.fleet.zone[drivers], self._pend, drivers, deadline_s=left)

    def reposition(self, drivers, zones, slot):
        self.dispatches.extend((int(d), int(z)) for d, z in zip(drivers, zones))


class Paper2Policy(Policy):
    version = "vfd-adapter-1"

    def __init__(self, algo, name: str, n_drivers: int, n_slots: int, slot_s: float,
                 v_avg_kmh: float, max_wait_s: float):
        self.a, self.name = algo, name
        self.cfg = _P2Cfg(n_slots, slot_s, v_avg_kmh)
        self.max_wait_slots = int(np.ceil(max_wait_s / slot_s))
        self.n_drivers = n_drivers
        self.new_day()

    def new_day(self):
        n, T = self.n_drivers, self.cfg.n_slots
        self.income_ps = np.zeros((n, T))
        self.active_ps = np.zeros((n, T))
        self.cum_income = np.zeros(n)
        self.n_served = np.zeros(n, dtype=np.int64)
        self.n_matched = 0

    def set_mode(self, train: bool):
        if hasattr(self.a, "training"):
            self.a.training = train
        if hasattr(self.a, "agent"):
            self.a.agent.training = train

    def act(self, obs: Observation) -> Action:
        f, pend = obs.fleet, obs.pending
        slot = min(int(f["epoch"]), self.cfg.n_slots - 1)
        # book the platform-recorded income change and busy flags of the elapsed slot
        self.income_ps[:, slot] += f["income"] - self.cum_income
        self.cum_income = f["income"].copy()
        self.active_ps[:, slot] = np.isin(f["status"], BUSY)

        plat = _P2Platform(self, obs, pend, slot)
        raised = (pend.t_request.to_numpy() // self.cfg.slot_seconds).astype(int)
        batch = OrderBatch(slot=slot, index=np.arange(len(pend)),
                           pickup=pend.origin_zone.to_numpy().astype(int),
                           dropoff=pend.dest_zone.to_numpy().astype(int),
                           fare=pend.quote.to_numpy().astype(float), raised=raised,
                           deadline=raised + self.max_wait_slots)
        has_dqn = hasattr(self.a, "agent")
        if has_dqn:
            self.a._state = None
        orders, drivers = self.a.match(plat, batch) if len(pend) else (np.empty(0, int), np.empty(0, int))
        plat.taken[drivers] = True
        self.a.dispatch(plat, batch)
        # VFDCFMVD's DQN observes its state inside match(), which returns early on an
        # empty order pool; in such a slot there is no clustering decision to learn
        # from, so only the value function is updated.
        skip_dqn = has_dqn and self.a._state is None and self.a.training
        if skip_dqn:
            self.a.training = False
        self.a.update(plat, batch, orders, drivers)
        if skip_dqn:
            self.a.training = True
        self.n_matched += len(orders)
        self.n_served[drivers] += 1
        rid = pend.request_id.to_numpy()
        return Action(offers=[(int(d), int(rid[o])) for o, d in zip(orders, drivers)],
                      dispatches=plat.dispatches,
                      info={"n_clusters": int(getattr(self.a, "_n_clusters", 0) or 0)})


def paper2_policies(n_zones: int, n_drivers: int, n_slots: int, slot_s: float,
                    v_avg_kmh: float, max_wait_s: float, xi: np.ndarray) -> dict:
    cfg = P2_CFG.algorithm

    def wrap(algo, name):
        return Paper2Policy(algo, name, n_drivers, n_slots, slot_s, v_avg_kmh, max_wait_s)
    return {
        "p2_nm": lambda: wrap(P2.NearestMatching(), "p2_nm"),
        "p2_wdf": lambda: wrap(P2.WorstDriverFirst(xi), "p2_wdf"),
        "p2_laf": lambda: wrap(P2.LAF(n_zones, xi, cfg), "p2_laf"),
        "p2_ilp": lambda: wrap(P2.ILP(), "p2_ilp"),
        "p2_sid": lambda: wrap(P2.SID(n_zones, cfg), "p2_sid"),
        "p2_vfdcfmvd": lambda: wrap(P2.VFDCFMVD(n_zones, cfg), "p2_vfdcfmvd"),
    }


LEARNERS = {"p1_laf", "p1_brp", "p1_momaql", "p2_laf", "p2_sid", "p2_vfdcfmvd"}


def policy_factories(data, cfg) -> tuple[dict, np.ndarray, float]:
    """All dispatch policies for a city: name -> zero-argument constructor; plus the
    hourly income weights xi and the typical zone-to-zone speed used by paper 2."""
    from .policies import POLICIES
    city = data.city
    n_slots = int((cfg.horizon_s + cfg.drain_s) / cfg.epoch_s)
    prof = np.array(city.market.demand_weekday, float)
    xi_hour = prof / prof.mean()
    xi = xi_hour[(np.arange(n_slots) * cfg.epoch_s // 3600).astype(int) % 24]   # per-slot weight
    tt = np.load(city.data_dir / "traffic" / "zone_tt_s.npy")[0]
    dd = np.load(city.data_dir / "traffic" / "zone_dist_m.npy")[0]
    off = ~np.eye(tt.shape[-1], dtype=bool)[None]
    v_avg = float(np.median((dd / np.maximum(tt, 1))[np.broadcast_to(off, tt.shape)]) * 3.6)
    f = {"nearest": POLICIES["nearest"]}
    f.update(paper1_policies(len(data.zones)))
    f.update(paper2_policies(len(data.zones), city.market.n_drivers, n_slots, cfg.epoch_s, v_avg,
                             900.0, xi))
    return f, xi_hour, v_avg
