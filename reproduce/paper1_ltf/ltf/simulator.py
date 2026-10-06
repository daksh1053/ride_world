"""Ride-hailing environment: drivers, requests, batching and utility accounting.

Follows the formulation in Sec. 3.1:

* A rider request is `r = (t_r, s_r, d_r)`.
* A driver state is `v^t = (c_v, m_v^t, g_v^t, o_v^t)` — capacity, riders on board,
  current node, accumulated utility.
* The utility of assigning `r` to `v` is the trip distance minus the pickup distance,
  `Geo(s_r, d_r) - Geo(g_v^t, s_r)`, evaluated at the moment of assignment.

Assignment happens in batches; within a batch the controller may run several rounds
so that a driver can be given more than one request (Sec. 4.3 explicitly allows an
agent to accept multiple requests concurrently), bounded by the vehicle capacity.
The environment is deliberately method-agnostic: a matcher only sees the batch and
the driver state, and returns request -> driver decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG, SimConfig
from .graph import RoadGraph


@dataclass
class Batch:
    """A set of requests raised in the same decision epoch."""

    time: pd.Timestamp
    origin: np.ndarray       # s_r for each request
    dest: np.ndarray         # d_r for each request

    def __len__(self) -> int:
        return len(self.origin)


class DriverFleet:
    """State of the `n` drivers (Sec. 3.1)."""

    def __init__(
        self,
        n_drivers: int,
        n_nodes: int,
        capacity: int,
        seed: int = 0,
        location_pmf: np.ndarray | None = None,
    ):
        rng = np.random.default_rng(seed)
        self.n = n_drivers
        self.capacity = np.full(n_drivers, capacity, dtype=np.int32)   # c_v
        self.onboard = np.zeros(n_drivers, dtype=np.int32)             # m_v^t
        # Sec. 4 stresses that "the varying initial locations of drivers in the
        # real-world significantly influence their ability to serve ride requests".
        # ASSUMPTION: initial positions are drawn in proportion to pickup demand
        # (drivers wait where riders are), falling back to uniform over L. Drawing
        # uniformly would park some drivers in zones that raise no requests, where
        # a pickup-radius constraint would strand them for the whole horizon.
        if location_pmf is None:
            self.location = rng.integers(0, n_nodes, size=n_drivers)   # g_v^t
        else:
            self.location = rng.choice(n_nodes, size=n_drivers, p=location_pmf)
        self.utility = np.zeros(n_drivers, dtype=np.float64)           # o_v^t
        self.n_served = np.zeros(n_drivers, dtype=np.int64)

    def reset_positions(self, n_nodes: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.location = rng.integers(0, n_nodes, size=self.n)

    def snapshot(self) -> dict:
        return {
            "location": self.location.copy(),
            "utility": self.utility.copy(),
            "onboard": self.onboard.copy(),
        }


class RideHailingEnv:
    """Batched simulator driving any `Matcher`."""

    def __init__(
        self,
        graph: RoadGraph,
        cfg: SimConfig = CONFIG.sim,
        location_pmf: np.ndarray | None = None,
    ):
        self.graph = graph
        self.cfg = cfg
        self.location_pmf = location_pmf
        self.reset()

    # ---------------------------------------------------------------- utilities
    def reset(self, seed: int | None = None) -> None:
        self.fleet = DriverFleet(
            self.cfg.n_drivers,
            self.graph.n,
            self.cfg.capacity,
            self.cfg.seed if seed is None else seed,
            self.location_pmf,
        )

    def utility_matrix(self, batch: Batch) -> np.ndarray:
        """`U[i, v]` = utility of giving request i to driver v, at current state.

        Note: Eq. 6 in the paper prints `Geo(d_a, s_a)` for the profit term while the
        surrounding text defines profit as "the distance from the start to end
        location for a request (s_r to d_r)". We follow the text.
        """
        profit = self.graph.geo(batch.origin, batch.dest)               # (m,)
        return profit[:, None] - self.pickup_matrix(batch)              # (m, n)

    def pickup_matrix(self, batch: Batch) -> np.ndarray:
        """`P[i, v]` = distance driver v must travel to reach request i's origin."""
        return self.graph.matrix[np.ix_(self.fleet.location, batch.origin)].T

    def feasible_matrix(self, batch: Batch) -> np.ndarray:
        """`(m, n)` mask of (request, driver) pairs inside the pickup radius.

        See `SimConfig.max_pickup_distance` for why this constraint exists.
        """
        if self.cfg.max_pickup_distance is None or self.cfg.max_pickup_distance <= 0:
            return np.ones((len(batch), self.fleet.n), dtype=bool)
        return self.pickup_matrix(batch) <= self.cfg.max_pickup_distance

    def available_mask(self) -> np.ndarray:
        """Drivers that may accept another request (capacity not exhausted)."""
        return self.fleet.onboard < self.fleet.capacity

    # ------------------------------------------------------------------ dynamics
    def apply(self, batch: Batch, assignment: np.ndarray) -> np.ndarray:
        """Commit one assignment round. `assignment[i]` is a driver index or -1.

        A round gives each driver at most one request, so the utilities booked here
        are exactly the ones the matcher saw in `utility_matrix`. Returns the
        per-driver utility gained in this round.
        """
        gained = np.zeros(self.fleet.n, dtype=np.float64)
        for i, v in enumerate(assignment):
            if v < 0:
                continue
            u = float(
                self.graph.geo(batch.origin[i], batch.dest[i])
                - self.graph.geo(self.fleet.location[v], batch.origin[i])
            )
            gained[v] += u
            self.fleet.utility[v] += u
            self.fleet.n_served[v] += 1
            self.fleet.onboard[v] += 1
            # The driver ends the trip at the request destination.
            self.fleet.location[v] = batch.dest[i]
        return gained

    def release(self) -> None:
        """Riders are dropped off at the end of each batch.

        ASSUMPTION: the paper keeps trip execution implicit (utility is booked at
        assignment time), so occupancy is cleared once a batch has been resolved.
        """
        self.fleet.onboard[:] = 0

    # ------------------------------------------------------------------- batches
    def iter_batches(self, day: pd.DataFrame):
        """Yield `Batch` objects for one day of requests."""
        if len(day) == 0:
            return
        freq = f"{self.cfg.batch_minutes}min"
        for stamp, grp in day.groupby(pd.Grouper(key="pickup_time", freq=freq)):
            if len(grp) == 0:
                continue
            yield Batch(
                time=stamp,
                origin=grp["s"].to_numpy().astype(int),
                dest=grp["d"].to_numpy().astype(int),
            )

    def run_batch(self, batch: Batch, matcher, learn: bool = False) -> int:
        """Resolve one batch through repeated single-assignment rounds."""
        pending = np.arange(len(batch))
        assigned = 0
        for _ in range(self.cfg.max_rounds_per_batch):
            if len(pending) == 0 or not self.available_mask().any():
                break
            sub = Batch(batch.time, batch.origin[pending], batch.dest[pending])
            decision = matcher.assign_round(self, sub)
            taken = decision >= 0
            if not taken.any():
                break
            prev = self.fleet.snapshot()
            gained = self.apply(sub, decision)
            if learn:
                matcher.observe(self, sub, decision, gained, prev)
            assigned += int(taken.sum())
            pending = pending[~taken]
        return assigned

    def run_day(self, day: pd.DataFrame, matcher, learn: bool = False) -> dict:
        """Run one day through a matcher. Returns per-day bookkeeping."""
        n_req = 0
        n_assigned = 0
        for batch in self.iter_batches(day):
            n_req += len(batch)
            n_assigned += self.run_batch(batch, matcher, learn=learn)
            self.release()
        return {"n_requests": n_req, "n_assigned": n_assigned}
