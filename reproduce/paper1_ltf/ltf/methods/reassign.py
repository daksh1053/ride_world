"""REASSIGN (Lesmana, Zhang & Bei, "Balancing efficiency and fairness in on-demand
ridesourcing", NeurIPS 2019) — traditional optimisation baseline.

The original algorithm computes the utility-maximising matching and then *reassigns*
requests to a fairer matching while bounding the loss of total utility. Sec. 5.2 of
the paper we reproduce notes that REASSIGN "can be applied with various fairness
definitions", and that the fairness definition is replaced by Eq. 2 (variance of
accumulated driver utility).

Implementation: max-utility Hungarian matching, then a local-search reassignment
phase that swaps request->driver decisions whenever this lowers the variance of
accumulated utilities and keeps the round's total utility above
`(1 - slack) * optimum`.
"""

from __future__ import annotations

import numpy as np

from .base import Matcher, hungarian_round


class Reassign(Matcher):
    name = "REASSIGN"

    def __init__(self, slack: float = 0.2, iters: int = 200, seed: int = 0):
        self.slack = slack
        self.iters = iters
        self.rng = np.random.default_rng(seed)

    def assign_round(self, env, batch) -> np.ndarray:
        m = len(batch)
        if m == 0:
            return np.full(0, -1, dtype=int)

        util = env.utility_matrix(batch)                  # (m, n)
        available = env.available_mask()
        feasible = env.feasible_matrix(batch)

        # Phase 1: efficiency-optimal matching.
        decision = hungarian_round(util, available, accept=feasible)
        opt = self._round_utility(util, decision)
        floor = opt - abs(opt) * self.slack

        # Phase 2: fairness-improving reassignment under the utility floor.
        free = set(np.flatnonzero(available).tolist())
        used = {v for v in decision if v >= 0}
        decision = self._reassign(util, decision, env.fleet.utility, floor,
                                  free - used, feasible)
        return decision

    @staticmethod
    def _round_utility(util: np.ndarray, decision: np.ndarray) -> float:
        idx = np.flatnonzero(decision >= 0)
        return float(util[idx, decision[idx]].sum()) if idx.size else 0.0

    def _reassign(
        self,
        util: np.ndarray,
        decision: np.ndarray,
        accumulated: np.ndarray,
        floor: float,
        idle: set[int],
        feasible: np.ndarray,
    ) -> np.ndarray:
        matched = np.flatnonzero(decision >= 0)
        if matched.size == 0:
            return decision

        decision = decision.copy()

        for _ in range(self.iters):
            projected = accumulated.copy()
            idx = np.flatnonzero(decision >= 0)
            np.add.at(projected, decision[idx], util[idx, decision[idx]])
            var_before = projected.var()

            # Candidate move: hand the request of the richest matched driver to the
            # poorest currently-idle driver (or swap two matched drivers).
            i = int(self.rng.choice(idx))
            v_old = decision[i]
            pool = list(idle) + [int(v) for v in decision[idx] if v != v_old]
            if not pool:
                break
            pool = [v for v in pool if feasible[i, v]]
            if not pool:
                continue
            v_new = min(pool, key=lambda v: projected[v])
            if v_new == v_old:
                continue

            trial = decision.copy()
            conflict = np.flatnonzero(trial == v_new)
            trial[i] = v_new
            for j in conflict:                 # keep it a valid matching
                trial[j] = v_old if feasible[j, v_old] else -1
            trial_util = self._round_utility(util, trial)
            if trial_util < floor:
                continue

            projected_t = accumulated.copy()
            idx_t = np.flatnonzero(trial >= 0)
            np.add.at(projected_t, trial[idx_t], util[idx_t, trial[idx_t]])
            if projected_t.var() < var_before:
                decision = trial
                idle.discard(v_new)
                if v_old not in trial:
                    idle.add(v_old)
        return decision
