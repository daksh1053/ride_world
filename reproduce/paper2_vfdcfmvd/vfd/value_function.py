"""Driver state value function V(s) (paper Sec. V-B, Fig. 3).

The driver's state is its zone: "the state of each driver is defined as s = (g) in S,
where g in G represents the index of the zone occupied by the driver". The value is
learned with TD(0) (Eq. 13):

    V(s) <- V(s) + alpha [ R_gamma + gamma^{Dt_{o,d}} V(s') - V(s) ]

with the duration-aware discounted reward of Eq. 12:

    R_gamma = sum_{t=0}^{T-1} gamma^t * r / T ,     r = p_o - C_d^o   (Eq. 11)

for an order spanning `T` time slots. "For high efficiency, referring to existing work
[34], we choose the tabular approach instead of the neural network to store and update
the value function."

The matching-pair weight of Eq. 14 is the *value difference*

    dV = R_gamma + gamma^{Dt_{o,d}} V(s') - V(s)

and Sec. V-C keeps only pairs with `dV > 0`: "we need to remove the matching pair
which will decrease the drivers' future potential income".
"""

from __future__ import annotations

import numpy as np


class StateValueFunction:
    """Tabular V(s) over zones, updated online with TD(0)."""

    def __init__(self, n_zones: int, alpha: float = 0.025, gamma: float = 0.9):
        self.n_zones = n_zones
        self.alpha = alpha
        self.gamma = gamma
        self.value = np.zeros(n_zones, dtype=np.float64)

    # ------------------------------------------------------------------ Eq. 12
    def discounted_reward(self, profit: np.ndarray, duration: np.ndarray) -> np.ndarray:
        """R_gamma = sum_{t<T} gamma^t r / T — the trip's reward spread over its slots.

        The geometric sum is closed-form: `(1 - gamma^T) / (1 - gamma) * r / T`.
        """
        duration = np.maximum(np.asarray(duration, dtype=float), 1.0)
        factor = (1.0 - self.gamma ** duration) / (1.0 - self.gamma) / duration
        return profit * factor

    # ------------------------------------------------------------------ Eq. 14
    def advantage(
        self,
        profit: np.ndarray,
        duration: np.ndarray,
        from_zone: np.ndarray,
        to_zone: np.ndarray,
    ) -> np.ndarray:
        """dV = R_gamma + gamma^{Dt} V(s') - V(s), the weight of a matching pair."""
        r_gamma = self.discounted_reward(profit, duration)
        return r_gamma + (self.gamma ** duration) * self.value[to_zone] - self.value[from_zone]

    # ------------------------------------------------------------------ Eq. 13
    def update(
        self,
        profit: np.ndarray,
        duration: np.ndarray,
        from_zone: np.ndarray,
        to_zone: np.ndarray,
    ) -> None:
        """TD(0) update over a batch of realised transitions."""
        if np.size(from_zone) == 0:
            return
        delta = self.advantage(profit, duration, from_zone, to_zone)
        # Several drivers can leave the same zone in one slot; accumulate their
        # corrections before applying, so the order inside a slot does not matter.
        correction = np.zeros(self.n_zones, dtype=np.float64)
        counts = np.zeros(self.n_zones, dtype=np.float64)
        np.add.at(correction, from_zone, delta)
        np.add.at(counts, from_zone, 1.0)
        touched = counts > 0
        self.value[touched] += self.alpha * correction[touched] / counts[touched]

    def reposition_advantage(
        self, cost: np.ndarray, duration: np.ndarray, from_zone: np.ndarray, to_zone: np.ndarray
    ) -> np.ndarray:
        """dV' for the idle-dispatching decision of Sec. V-C.

        This is *not* the matching weight of Eq. 14. There, the alternative to
        serving an order is the driver's present value V(s). Here the alternative is
        to keep waiting in the same zone, and Sec. V-B fixes what waiting is worth:
        "When the driver remains idle, their location stays the same, i.e. s' = (g),
        and the agent receives an immediate reward of 0" — so waiting `Dt` slots is
        worth `gamma^Dt V(s)`, not `V(s)`. Comparing like with like:

            dV' = R_gamma + gamma^Dt ( V(g') - V(g) )

        Using `V(g)` undiscounted instead charges the move `(1 - gamma^Dt) V(g)` for
        time that the driver would have spent idling anyway. With a roughly flat
        V that term alone exceeds any plausible gain, and the dispatching module
        never fires at all — repositioning drops to literally zero events per day.
        """
        r_gamma = self.discounted_reward(cost, duration)
        return r_gamma + (self.gamma ** duration) * (self.value[to_zone] - self.value[from_zone])

    def update_idle(self, zones: np.ndarray) -> None:
        """A driver that stays idle receives r = 0 and s' = s (Sec. V-B).

        With `R_gamma = 0` and `s' = s` the TD error is `(gamma - 1) V(s)`, i.e.
        idling decays the value of a zone toward zero.
        """
        if np.size(zones) == 0:
            return
        counts = np.zeros(self.n_zones, dtype=np.float64)
        np.add.at(counts, zones, 1.0)
        touched = counts > 0
        self.value[touched] += self.alpha * (self.gamma - 1.0) * self.value[touched]
