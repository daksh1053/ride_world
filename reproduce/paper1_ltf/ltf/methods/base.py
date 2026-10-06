"""Common interface for allocation methods.

Every method resolves an *assignment round*: given the environment state and the
requests still pending in the current batch, return `decision[i] = v` (driver index)
or `-1` for "no action at t" — Sec. 4.3 explicitly keeps the no-action option so a
driver may go unassigned in a period.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import linear_sum_assignment


class Matcher(ABC):
    name = "matcher"

    @abstractmethod
    def assign_round(self, env, batch) -> np.ndarray:
        """Return an array of length `len(batch)` with driver indices or -1."""

    def observe(self, env, batch, decision, gained, prev) -> None:
        """Learning hook; no-op for non-learning methods.

        `prev` is the fleet snapshot taken immediately before the round was
        committed (`location`, `utility`, `onboard`).
        """

    def reset(self) -> None:
        """Called before a fresh evaluation run."""


def hungarian_round(score: np.ndarray, feasible: np.ndarray, accept: np.ndarray | None = None) -> np.ndarray:
    """Max-weight bipartite matching of requests to drivers.

    `score` is `(m_requests, n_drivers)`; `feasible` is a boolean driver mask.
    `accept` optionally masks (request, driver) pairs that must not be matched.
    Requests left unmatched get -1.
    """
    m, n = score.shape
    decision = np.full(m, -1, dtype=int)
    cols = np.flatnonzero(feasible)
    if m == 0 or cols.size == 0:
        return decision

    sub = score[:, cols].astype(np.float64)
    allowed = np.ones_like(sub, dtype=bool) if accept is None else accept[:, cols]
    if not allowed.any():
        return decision

    # Forbidden pairs get a cost so large that Hungarian never picks them unless
    # forced; rows with no allowed driver are dropped up front.
    keep = allowed.any(axis=1)
    rows = np.flatnonzero(keep)
    sub, allowed = sub[keep], allowed[keep]
    penalty = np.abs(sub).max() * 1e6 + 1.0
    cost = np.where(allowed, -sub, penalty)

    r, c = linear_sum_assignment(cost)
    for ri, ci in zip(r, c):
        if allowed[ri, ci]:
            decision[rows[ri]] = cols[ci]
    return decision


def variance_delta(utilities: np.ndarray, driver: int, gain: float) -> float:
    """Change in population variance of driver utilities if `driver` gains `gain`.

    Used by the fairness-aware objective (Eq. 3): `max pi(M) - lambda F(M)`.
    """
    n = utilities.size
    mean = utilities.mean()
    # Var_new - Var_old for a single-coordinate increment, in closed form.
    return (gain * gain) / n + (2.0 * gain * (utilities[driver] - mean)) / n - (gain * gain) / (n * n)


def variance_delta_matrix(utilities: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Vectorised `variance_delta` over a `(m_requests, n_drivers)` gain matrix."""
    n = utilities.size
    centred = utilities - utilities.mean()
    return (
        (gains ** 2) / n
        + 2.0 * gains * centred[None, :] / n
        - (gains ** 2) / (n * n)
    )


def scaled_fairness_penalty(
    utilities: np.ndarray, gains: np.ndarray, lam: float, omega: float
) -> np.ndarray:
    """`lambda * omega * dVar`, re-expressed on the utility scale.

    Sec. 4.4 introduces omega "to adjust fairness into the same range as utility" and
    "to avoid fairness getting a larger weight due to the unavoidable increase of
    variance while the time horizon gradually increases". The dominant term of the
    variance delta for a gain `g` to driver `v` is `2 g (o_v - mean) / n`, whose
    natural scale is `2 sigma(o) / n`; dividing by it leaves

        penalty = lambda * omega * g * z_v ,   z_v = (o_v - mean) / sigma(o)

    which is dimensionally comparable to utility and does not grow with the horizon.

    The second-order term `g^2/n` of the exact variance delta is dropped on purpose:
    it does not vanish when the fleet is perfectly equal, so keeping it while
    dividing by `sigma(o) -> 0` makes the penalty diverge at the start of the horizon
    and freezes every driver out of every assignment. With the z-score form, a
    perfectly equal fleet correctly yields a zero fairness penalty.
    """
    std = float(utilities.std())
    if std < 1e-9:
        return np.zeros_like(gains)
    z = (utilities - utilities.mean()) / std
    return lam * omega * gains * z[None, :]
