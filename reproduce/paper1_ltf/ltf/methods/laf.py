"""LAF (Shi et al., "Learning to Assign: Towards Fair Task Assignment in Large-Scale
Ride Hailing", KDD 2021).

Sec. 5.2: "LAF exploits a Markov Decision Process as a re-weighting module to refine
the weight for each edge to promote fairness. LAF then utilises the Hungarian
algorithm to optimise total utility and output the final allocation plan." The
fairness definition is replaced by Eq. 2.

Implementation: a tabular state-value function over the driver's *income level*
(discretised accumulated utility relative to the fleet mean) is learned online with
TD(0) against a fairness-shaped reward. The value is added as an edge re-weighting
term before running Hungarian on total utility.
"""

from __future__ import annotations

import numpy as np

from .base import Matcher, hungarian_round


class LAF(Matcher):
    name = "LAF"

    N_LEVELS = 21   # income levels: z-score of accumulated utility, clipped to +-2

    def __init__(self, lam: float = 1.0, alpha: float = 0.1, gamma: float = 0.9, seed: int = 0):
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.reset()

    def reset(self) -> None:
        self.value = np.zeros(self.N_LEVELS, dtype=np.float64)

    def _levels(self, utilities: np.ndarray) -> np.ndarray:
        std = utilities.std()
        z = np.zeros_like(utilities) if std < 1e-9 else (utilities - utilities.mean()) / std
        idx = np.clip(np.round((z + 2.0) / 4.0 * (self.N_LEVELS - 1)), 0, self.N_LEVELS - 1)
        return idx.astype(int)

    def assign_round(self, env, batch) -> np.ndarray:
        m = len(batch)
        if m == 0:
            return np.full(0, -1, dtype=int)

        util = env.utility_matrix(batch)                     # (m, n)
        levels = self._levels(env.fleet.utility)
        # Re-weighting module: an edge to a driver whose income level is valued
        # highly (i.e. currently under-served) gets a bonus before matching.
        reweighted = util + self.lam * self.value[levels][None, :]
        return hungarian_round(reweighted, env.available_mask(), accept=env.feasible_matrix(batch))

    def observe(self, env, batch, decision, gained, prev) -> None:
        """TD(0) update of the income-level value function."""
        utility_before = prev["utility"]
        levels_before = self._levels(utility_before)
        levels_after = self._levels(env.fleet.utility)
        mean_before = utility_before.mean()
        for v in np.unique(decision[decision >= 0]):
            # Fairness-shaped reward: progress towards the fleet mean.
            reward = -abs(env.fleet.utility[v] - env.fleet.utility.mean()) + abs(
                utility_before[v] - mean_before
            )
            s, s2 = levels_before[v], levels_after[v]
            target = reward + self.gamma * self.value[s2]
            self.value[s] += self.alpha * (target - self.value[s])
