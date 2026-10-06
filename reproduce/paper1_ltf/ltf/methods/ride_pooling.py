"""Balance Ride-Pooling (Raman, Shah & Dickerson, "Data-Driven Methods for Balancing
Fairness and Efficiency in Ride-Pooling", 2021) — the RL baseline of Sec. 5.2.

The method solves an MDP that maximises the number of serviced requests while
keeping earnings balanced: a state-value function over driver locations (learned in
the NeurADP style) scores the future worth of ending a trip at a node, and the
matching objective adds an income-redistribution term. Sec. 5.2 keeps this baseline
"unchanged" because its fairness definition already matches Eq. 2.

The distinguishing point of the paper we reproduce is that this baseline conditions
only on *historical and current* requests — it has no forecasting module — which is
exactly what `Proposed(use_prediction=False)` isolates in the ablation.
"""

from __future__ import annotations

import numpy as np

from .base import Matcher, hungarian_round, scaled_fairness_penalty


class BalanceRidePooling(Matcher):
    name = "Balance Ride-Pooling"

    def __init__(
        self,
        n_nodes: int,
        lam: float = 1.0,
        omega: float = 0.6,
        gamma: float = 0.9,
        alpha: float = 0.1,
    ):
        self.n_nodes = n_nodes
        self.lam = lam
        self.omega = omega
        self.gamma = gamma
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self.value = np.zeros(self.n_nodes, dtype=np.float64)   # V(location)

    def _fair_penalty(self, utilities: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """Income-redistribution term, scaled to the utility range."""
        return scaled_fairness_penalty(utilities, gains, self.lam, self.omega)

    def assign_round(self, env, batch) -> np.ndarray:
        m = len(batch)
        if m == 0:
            return np.full(0, -1, dtype=int)

        util = env.utility_matrix(batch)                          # (m, n)
        # Future value of ending at the request destination, relative to staying put.
        future = self.gamma * self.value[batch.dest][:, None] - self.value[env.fleet.location][None, :]
        score = util + future - self._fair_penalty(env.fleet.utility, util)
        return hungarian_round(score, env.available_mask(), accept=env.feasible_matrix(batch))

    def observe(self, env, batch, decision, gained, prev) -> None:
        taken = np.flatnonzero(decision >= 0)
        for i in taken:
            v = int(decision[i])
            s, s_next = int(prev["location"][v]), int(batch.dest[i])
            reward = gained[v]
            target = reward + self.gamma * self.value[s_next]
            self.value[s] += self.alpha * (target - self.value[s])
