"""The five benchmark algorithms of Sec. VI-B.

1. **NM** — Nearest-Matching. "commonly adopted in practical ride-hailing systems such
   as Uber, assigns each order to its nearest driver."
2. **LAF** — Learning-to-Assign-with-Fairness (Shi et al. [11]). "LAF consists of
   learning and planning two phases. In the learning phase, LAF updates the value
   function through online learning. In the planning phase, LAF combines the value
   function with a revised KM algorithm to make matching decisions, enhancing fairness
   by removing unfair driver-to-order pairs, and LAF also designs an idle vehicle
   dispatching algorithm based on the value function."
3. **WDF** — Worst-Driver-First. "In each time slot, it ranks all available drivers
   based on weighted unit time income, and prioritizes matching the worst fairness
   drivers with the orders with the highest value. The WDF algorithm represents a
   special case in which the number of clusters per time slot in the VFDCFMVD
   algorithm equals the total number of drivers."
4. **ILP** — Integer-Linear-Programming [10]. "The evaluation in this article focuses
   on the driver side ... the experiments adopt an ILP variant that only retains the
   driver fairness objective."
5. **SID** — Simple-Incentive-for-Drivers [26]. "utilizes integer linear programming
   to optimize the matching process, and combines the fairness incentive score with
   state value function to adjust the matching score, thereby ensuring a more fair
   income among drivers."

The assignment polytope is totally unimodular, so the ILP relaxations in (4) and (5)
are solved exactly by the same Kuhn-Munkres routine used everywhere else — no
branch-and-bound is needed, and the objective is what distinguishes them.
"""

from __future__ import annotations

import numpy as np

from ..config import CONFIG, AlgorithmConfig
from ..value_function import StateValueFunction
from .base import EMPTY, Algorithm, km_match
from .vfdcfmvd import VFDCFMVD


class NearestMatching(Algorithm):
    """NM — each order goes to its nearest available driver; no dispatching."""

    name = "NM"

    def match(self, platform, batch):
        if len(batch) == 0:
            return EMPTY
        available = np.flatnonzero(platform.fleet.available(batch.slot))
        if available.size == 0:
            return EMPTY

        pickup = platform.pickup_distance(batch, available)
        feasible = platform.feasible_matrix(batch, available)
        taken = np.zeros(available.size, dtype=bool)
        orders, drivers = [], []
        # Orders are served in arrival order; each takes the closest free driver.
        for i in np.argsort(batch.raised, kind="stable"):
            options = np.where(feasible[i] & ~taken, pickup[i], np.inf)
            if not np.isfinite(options).any():
                continue
            j = int(np.argmin(options))
            taken[j] = True
            orders.append(i)
            drivers.append(available[j])
            if taken.all():
                break
        return np.array(orders, dtype=int), np.array(drivers, dtype=int)


class WorstDriverFirst(Algorithm):
    """WDF — poorest driver first, highest-value order to each."""

    name = "WDF"

    def __init__(self, xi: np.ndarray):
        self.xi = xi

    def _weighted_unit_income(self, platform, slot: int) -> np.ndarray:
        """F_d restricted to the elapsed horizon (Definition 6)."""
        income = platform.fleet.income_per_slot[:, : slot + 1]
        active = platform.fleet.active_per_slot[:, : slot + 1].sum(axis=1)
        weighted = (income / self.xi[None, : slot + 1]).sum(axis=1)
        return np.divide(weighted, active, out=np.zeros_like(weighted), where=active > 0)

    def match(self, platform, batch):
        if len(batch) == 0:
            return EMPTY
        available = np.flatnonzero(platform.fleet.available(batch.slot))
        if available.size == 0:
            return EMPTY

        profit = platform.profit_matrix(batch, available)
        feasible = platform.feasible_matrix(batch, available)
        rank = np.argsort(self._weighted_unit_income(platform, batch.slot)[available])

        used = np.zeros(len(batch), dtype=bool)
        orders, drivers = [], []
        for j in rank:
            options = np.where(feasible[:, j] & ~used, profit[:, j], -np.inf)
            if not np.isfinite(options).any():
                continue
            i = int(np.argmax(options))
            used[i] = True
            orders.append(i)
            drivers.append(available[j])
            if used.all():
                break
        return np.array(orders, dtype=int), np.array(drivers, dtype=int)


class LAF(Algorithm):
    """LAF — value-guided KM with unfair pairs removed, plus idle dispatching."""

    name = "LAF"

    def __init__(self, n_zones: int, xi: np.ndarray, cfg: AlgorithmConfig = CONFIG.algorithm):
        self.cfg = cfg
        self.n_zones = n_zones
        self.xi = xi
        self.value = StateValueFunction(n_zones, cfg.value_lr, cfg.value_gamma)
        self._pre_zone = None

    def reset(self) -> None:
        self.value = StateValueFunction(self.n_zones, self.cfg.value_lr, self.cfg.value_gamma)
        self._pre_zone = None

    def match(self, platform, batch):
        if len(batch) == 0:
            return EMPTY
        available = np.flatnonzero(platform.fleet.available(batch.slot))
        if available.size == 0:
            return EMPTY

        pickup_km = platform.pickup_distance(batch, available)
        trip_km = platform.trip_distance(batch)[:, None]
        profit = batch.fare[:, None] - (pickup_km + trip_km) * platform.fleet.cost[available][None, :]
        duration = platform.travel_slots(pickup_km + trip_km)
        from_zone = np.broadcast_to(platform.fleet.zone[available][None, :], profit.shape)
        to_zone = np.broadcast_to(batch.dropoff[:, None], profit.shape)
        weights = self.value.advantage(profit, duration, from_zone, to_zone)

        # "enhancing fairness by removing unfair driver-to-order pairs": LAF cuts
        # every pair that would send an already above-average driver even further
        # ahead. This is the aggressive pruning Sec. VI-C blames for LAF's high idle
        # rate ("its strategy of rigidly removing unfair driver-order matching
        # severely degrades overall matching efficiency").
        income = platform.fleet.income_upto(batch.slot)[available]
        rich = income > np.median(income)
        fair = ~np.broadcast_to(rich[None, :], profit.shape) | (profit <= 0)

        admissible = (weights > 0) & fair & platform.feasible_matrix(batch, available)
        r, c = km_match(weights, admissible)
        if r.size == 0:
            self._pre_zone = None
            return EMPTY
        drivers = available[c]
        self._pre_zone = platform.fleet.zone[drivers].copy()
        return r, drivers

    def dispatch(self, platform, batch) -> None:
        """LAF's value-function-based idle dispatching."""
        idle = np.flatnonzero(platform.fleet.dispatchable(batch.slot))
        if idle.size == 0:
            return
        zones = platform.fleet.zone[idle]
        near = np.stack([platform.graph.nearby(z, self.cfg.n_nearby_zones) for z in zones])
        km = platform.graph.matrix[zones[:, None], near]
        duration = platform.travel_slots(km)
        profit = -km * platform.fleet.cost[idle][:, None]
        delta = self.value.reposition_advantage(profit, duration, zones[:, None], near)
        best = np.argmax(delta, axis=1)
        rows = np.arange(idle.size)
        move = delta[rows, best] > 0
        if move.any():
            platform.reposition(idle[move], near[rows[move], best[move]], batch.slot)

    def update(self, platform, batch, matched_orders, matched_drivers) -> None:
        if len(matched_orders) and self._pre_zone is not None:
            pickup = batch.pickup[matched_orders]
            dropoff = batch.dropoff[matched_orders]
            km = platform.graph.dis(self._pre_zone, pickup) + platform.graph.dis(pickup, dropoff)
            profit = batch.fare[matched_orders] - km * platform.fleet.cost[matched_drivers]
            self.value.update(profit, platform.travel_slots(km), self._pre_zone, dropoff)
        # Unmatched drivers contribute the r = 0, s' = s transition (Sec. V-B).
        still_idle = np.flatnonzero(platform.fleet.available(batch.slot))
        if still_idle.size:
            self.value.update_idle(platform.fleet.zone[still_idle])


class ILP(Algorithm):
    """ILP — driver-fairness-only integer program, no idle dispatching.

    Objective: maximise total profit plus a fairness term that rewards assigning to
    drivers whose income is below the fleet average. Sec. VI-C: "the ILP algorithm
    ignores idle vehicle dispatching and future matching impacts."
    """

    name = "ILP"

    def __init__(self, fairness_weight: float = 1.0):
        self.fairness_weight = fairness_weight

    def match(self, platform, batch):
        if len(batch) == 0:
            return EMPTY
        available = np.flatnonzero(platform.fleet.available(batch.slot))
        if available.size == 0:
            return EMPTY

        profit = platform.profit_matrix(batch, available)
        income = platform.fleet.income_upto(batch.slot)[available]
        spread = income.std()
        deficit = (income.mean() - income) / spread if spread > 1e-9 else np.zeros_like(income)
        weights = profit + self.fairness_weight * np.abs(profit) * deficit[None, :]

        admissible = (profit > 0) & platform.feasible_matrix(batch, available)
        r, c = km_match(weights, admissible)
        return r, available[c]


class SID(Algorithm):
    """SID — ILP matching whose score combines a fairness incentive with V(s).

    Sec. VI-C: "The SID algorithm accounts for the effect of current matching on
    future matching but neglects the influence of fairness enhancement operations on
    order matching" — so it uses the value function, but applies the same fairness
    incentive to every driver rather than clustering them.
    """

    name = "SID"

    def __init__(self, n_zones: int, cfg: AlgorithmConfig = CONFIG.algorithm,
                 incentive: float = 1.0):
        self.cfg = cfg
        self.n_zones = n_zones
        self.incentive = incentive
        self.value = StateValueFunction(n_zones, cfg.value_lr, cfg.value_gamma)
        self._pre_zone = None

    def reset(self) -> None:
        self.value = StateValueFunction(self.n_zones, self.cfg.value_lr, self.cfg.value_gamma)
        self._pre_zone = None

    def match(self, platform, batch):
        if len(batch) == 0:
            return EMPTY
        available = np.flatnonzero(platform.fleet.available(batch.slot))
        if available.size == 0:
            return EMPTY

        pickup_km = platform.pickup_distance(batch, available)
        trip_km = platform.trip_distance(batch)[:, None]
        profit = batch.fare[:, None] - (pickup_km + trip_km) * platform.fleet.cost[available][None, :]
        duration = platform.travel_slots(pickup_km + trip_km)
        from_zone = np.broadcast_to(platform.fleet.zone[available][None, :], profit.shape)
        to_zone = np.broadcast_to(batch.dropoff[:, None], profit.shape)
        value_score = self.value.advantage(profit, duration, from_zone, to_zone)

        income = platform.fleet.income_upto(batch.slot)[available]
        spread = income.std()
        incentive = (income.mean() - income) / spread if spread > 1e-9 else np.zeros_like(income)
        weights = value_score + self.incentive * np.abs(value_score) * incentive[None, :]

        admissible = (profit > 0) & platform.feasible_matrix(batch, available)
        r, c = km_match(weights, admissible)
        if r.size == 0:
            self._pre_zone = None
            return EMPTY
        drivers = available[c]
        self._pre_zone = platform.fleet.zone[drivers].copy()
        return r, drivers

    def update(self, platform, batch, matched_orders, matched_drivers) -> None:
        if len(matched_orders) and self._pre_zone is not None:
            pickup = batch.pickup[matched_orders]
            dropoff = batch.dropoff[matched_orders]
            km = platform.graph.dis(self._pre_zone, pickup) + platform.graph.dis(pickup, dropoff)
            profit = batch.fare[matched_orders] - km * platform.fleet.cost[matched_drivers]
            self.value.update(profit, platform.travel_slots(km), self._pre_zone, dropoff)
        # Unmatched drivers contribute the r = 0, s' = s transition (Sec. V-B).
        still_idle = np.flatnonzero(platform.fleet.available(batch.slot))
        if still_idle.size:
            self.value.update_idle(platform.fleet.zone[still_idle])
