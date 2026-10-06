"""The ride-hailing platform simulator (paper Sec. III, "Basic Settings").

"the online ride-hailing platform oversees all drivers, and the total time horizon is
divided into T time slots ... In each time slot, passengers submit orders to the
platform, and the platform matches orders with available drivers and computes the
payment. The platform also needs to provide dispatching suggestions for idle
vehicles."

State carried per driver `d` (Definition 2): current zone `l_d` and unit travel cost
`c_d`. Per order `o` (Definition 3): `(l_o^p, l_o^d, t_o^r, t_o^w, p_o)`.

Two constraints from Sec. IV are enforced here rather than in any single algorithm:
  * `dis(l_d, l_o^p) / V_avg <= t_o^w` — a driver too far away cannot be matched
    before the passenger cancels.
  * a driver occupied with an order is unavailable until it completes, for
    `Dt_{o,d} = ceil((dis(l_d,l_o^p) + dis(l_o^p,l_o^d)) / V_avg * 60)` slots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG, PlatformConfig
from .graph import ZoneGraph


@dataclass
class OrderBatch:
    """Orders raised in one time slot, plus those still inside their waiting window."""

    slot: int
    index: np.ndarray        # row ids into the day's order table
    pickup: np.ndarray       # l_o^p
    dropoff: np.ndarray      # l_o^d
    fare: np.ndarray         # p_o
    raised: np.ndarray       # t_o^r
    deadline: np.ndarray     # t_o^r + t_o^w, in slots

    def __len__(self) -> int:
        return len(self.index)

    def subset(self, sel: np.ndarray) -> "OrderBatch":
        return OrderBatch(
            self.slot, self.index[sel], self.pickup[sel], self.dropoff[sel],
            self.fare[sel], self.raised[sel], self.deadline[sel],
        )


class Fleet:
    """Driver state (Definition 2) and per-slot income/active-time bookkeeping."""

    def __init__(self, n_drivers: int, graph: ZoneGraph, cfg: PlatformConfig, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n = n_drivers
        self.cfg = cfg
        # "The driver's starting location is randomly assigned within the Manhattan
        # taxi zone map."
        self.zone = rng.integers(0, graph.n, size=n_drivers)             # l_d
        # "For each driver, the unit travel cost is randomly selected from {6,8,10}x..."
        self.cost = rng.choice(np.array(cfg.unit_costs), size=n_drivers)  # c_d
        self.busy_until = np.zeros(n_drivers, dtype=np.int32)
        # A repositioning vehicle is still matchable — it can be diverted to an
        # order en route — but is not dispatched again until it has arrived.
        self.repositioning_until = np.zeros(n_drivers, dtype=np.int32)
        self.income_per_slot = np.zeros((n_drivers, cfg.n_slots), dtype=np.float64)
        self.active_per_slot = np.zeros((n_drivers, cfg.n_slots), dtype=np.float64)
        self.n_served = np.zeros(n_drivers, dtype=np.int64)

    # ------------------------------------------------------------------- queries
    def available(self, slot: int) -> np.ndarray:
        """`select_available_drivers(D, t)` of Algorithm 2."""
        return self.busy_until <= slot

    def dispatchable(self, slot: int) -> np.ndarray:
        """`select_idle_drivers(D, t)`: free *and* not already repositioning."""
        return (self.busy_until <= slot) & (self.repositioning_until <= slot)

    def income(self) -> np.ndarray:
        """u_d^T, cumulative income (Eq. 1)."""
        return self.income_per_slot.sum(axis=1)

    def income_upto(self, slot: int) -> np.ndarray:
        return self.income_per_slot[:, : slot + 1].sum(axis=1)


class Platform:
    """Drives one day of order matching and idle-vehicle dispatching."""

    def __init__(
        self,
        graph: ZoneGraph,
        cfg: PlatformConfig = CONFIG.platform,
        n_drivers: int | None = None,
        seed: int = 0,
    ):
        self.graph = graph
        self.cfg = cfg
        self.n_drivers = n_drivers or cfg.default_vehicles
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.fleet = Fleet(self.n_drivers, self.graph, self.cfg, self.seed)
        self.n_matched = 0

    # ---------------------------------------------------------------- quantities
    def travel_slots(self, distance_km: np.ndarray | float) -> np.ndarray | int:
        """Dt_{o,d}: trip duration in time slots, at the average speed."""
        minutes = np.asarray(distance_km) / self.cfg.v_avg_kmh * 60.0
        slots = np.ceil(minutes * 60.0 / self.cfg.slot_seconds)
        return np.maximum(slots, 1).astype(int)

    def pickup_distance(self, batch: OrderBatch, drivers: np.ndarray) -> np.ndarray:
        """`(m_orders, n_drivers)` of dis(l_d, l_o^p)."""
        return self.graph.matrix[np.ix_(self.fleet.zone[drivers], batch.pickup)].T

    def trip_distance(self, batch: OrderBatch) -> np.ndarray:
        """dis(l_o^p, l_o^d) per order."""
        return self.graph.dis(batch.pickup, batch.dropoff)

    def cost_matrix(self, batch: OrderBatch, drivers: np.ndarray) -> np.ndarray:
        """C_d^o of Eq. 2: (dis(l_d,l_o^p) + dis(l_o^p,l_o^d)) * c_d."""
        total = self.pickup_distance(batch, drivers) + self.trip_distance(batch)[:, None]
        return total * self.fleet.cost[drivers][None, :]

    def profit_matrix(self, batch: OrderBatch, drivers: np.ndarray) -> np.ndarray:
        """r = p_o - C_d^o, the immediate reward of Eq. 11."""
        return batch.fare[:, None] - self.cost_matrix(batch, drivers)

    def feasible_matrix(self, batch: OrderBatch, drivers: np.ndarray) -> np.ndarray:
        """The waiting-time constraint of Sec. IV: dis(l_d,l_o^p)/V_avg <= t_o^w."""
        max_km = (batch.deadline - batch.slot) * (
            self.cfg.slot_seconds / 3600.0
        ) * self.cfg.v_avg_kmh
        return self.pickup_distance(batch, drivers) <= max_km[:, None]

    # ------------------------------------------------------------------ dynamics
    def serve(self, batch: OrderBatch, order_idx: np.ndarray, drivers: np.ndarray) -> None:
        """Commit matched (order, driver) pairs.

        The driver collects `p_o - C_d^o`, becomes busy for the trip duration, and
        ends at the order's drop-off zone.
        """
        if len(order_idx) == 0:
            return
        slot = batch.slot
        pickup_km = self.graph.dis(self.fleet.zone[drivers], batch.pickup[order_idx])
        trip_km = self.graph.dis(batch.pickup[order_idx], batch.dropoff[order_idx])
        profit = batch.fare[order_idx] - (pickup_km + trip_km) * self.fleet.cost[drivers]
        duration = self.travel_slots(pickup_km + trip_km)

        np.add.at(self.fleet.income_per_slot, (drivers, slot), profit)
        for d, dur in zip(drivers, duration):
            end = min(slot + dur, self.cfg.n_slots)
            self.fleet.active_per_slot[d, slot:end] += 1.0
        self.fleet.busy_until[drivers] = slot + duration
        self.fleet.zone[drivers] = batch.dropoff[order_idx]
        self.fleet.n_served[drivers] += 1
        self.n_matched += len(order_idx)

    def reposition(self, drivers: np.ndarray, zones: np.ndarray, slot: int) -> None:
        """Idle-vehicle dispatching: "assigning a virtual order o, in which the
        payment p_o is set to zero, the order originates from zone g, and its
        destination corresponds to one of the adjacent zones of g" (Sec. V-C).

        The driver pays the travel cost of repositioning, and the time it spends
        moving counts as active time. It stays *matchable* while it repositions —
        Sec. V-C describes dispatching as a way to "increase the matching
        opportunities" of idle drivers, so a vehicle heading toward a high-value zone
        must remain available for an order that appears on the way. It is simply not
        re-dispatched until it arrives.
        """
        if len(drivers) == 0:
            return
        km = self.graph.dis(self.fleet.zone[drivers], zones)
        cost = km * self.fleet.cost[drivers]
        duration = self.travel_slots(km)
        np.add.at(self.fleet.income_per_slot, (drivers, slot), -cost)
        for d, dur in zip(drivers, duration):
            end = min(slot + dur, self.cfg.n_slots)
            self.fleet.active_per_slot[d, slot:end] += 1.0
        self.fleet.repositioning_until[drivers] = slot + duration
        self.fleet.zone[drivers] = zones

    # ------------------------------------------------------------------- batches
    def run_day(self, day: pd.DataFrame, algorithm, waiting_slots: np.ndarray) -> int:
        """Run one full day (T time slots) through an algorithm.

        Algorithm 2 line 3: `O_t <- {o in O | t_o^r <= t < t_o^r + t_o^w}` — an order
        joins the pool when it is raised and leaves it when matched, or when the
        passenger's maximum waiting time expires and the request is cancelled.
        """
        raised = day["slot"].to_numpy().astype(int)
        pickup = day["pu_idx"].to_numpy().astype(int)
        dropoff = day["do_idx"].to_numpy().astype(int)
        fare = day["fare_amount"].to_numpy().astype(float)
        deadline = raised + np.asarray(waiting_slots, dtype=int)

        arrivals: list[list[int]] = [[] for _ in range(self.cfg.n_slots)]
        for i, t in enumerate(raised):
            if 0 <= t < self.cfg.n_slots:
                arrivals[t].append(i)

        pending = np.empty(0, dtype=int)
        for slot in range(self.cfg.n_slots):
            pending = np.concatenate([pending, np.array(arrivals[slot], dtype=int)])
            pending = pending[deadline[pending] > slot]

            batch = OrderBatch(
                slot=slot,
                index=pending,
                pickup=pickup[pending],
                dropoff=dropoff[pending],
                fare=fare[pending],
                raised=raised[pending],
                deadline=deadline[pending],
            )

            matched_orders, matched_drivers = algorithm.match(self, batch)
            self.serve(batch, matched_orders, matched_drivers)
            algorithm.dispatch(self, batch)
            algorithm.update(self, batch, matched_orders, matched_drivers)

            if len(matched_orders):
                keep = np.ones(len(pending), dtype=bool)
                keep[matched_orders] = False
                pending = pending[keep]

        return len(day)
