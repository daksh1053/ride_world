"""Greedy baseline.

Sec. 5.2: "We implement Greedy with the objective of balancing efficiency and
fairness according to Eq. 3", i.e. each request is handed to whichever driver
maximises the myopic marginal objective `u(r,v) - lambda * dVar`. Because the
variance term dominates once utilities have spread out, greedy keeps feeding the
poorest driver regardless of pickup cost — the mechanism behind the negative total
utility reported in Table 1.
"""

from __future__ import annotations

import numpy as np

from .base import Matcher, variance_delta_matrix


class Greedy(Matcher):
    name = "Greedy"

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def assign_round(self, env, batch) -> np.ndarray:
        m = len(batch)
        decision = np.full(m, -1, dtype=int)
        if m == 0:
            return decision

        util = env.utility_matrix(batch)                       # (m, n)
        feasible = env.feasible_matrix(batch)
        available = env.available_mask().copy()
        utilities = env.fleet.utility.copy()

        # Requests are treated one at a time, in arrival order, as greedy does,
        # with the running utilities updated after each decision.
        for i in range(m):
            score = util[i] - self.lam * variance_delta_matrix(utilities, util[i : i + 1])[0]
            score = np.where(available & feasible[i], score, -np.inf)
            if not np.isfinite(score).any():
                continue          # no driver within the pickup radius for this rider
            v = int(np.argmax(score))
            decision[i] = v
            utilities[v] += util[i, v]
            available[v] = False        # one request per driver per round
            if not available.any():
                break
        return decision
