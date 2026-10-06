"""Efficiency and long-term fairness measures (paper Sec. 3.2, 3.3, 5.3)."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Evaluation:
    total_utility: float     # pi(M), Eq. 1
    fairness: float          # F(M) = Var(o_v), Eq. 2
    normalised_fairness: float  # \hat F(M) = sigma(U) / mean(U), Eq. 8
    min_utility: float
    mean_utility: float
    max_utility: float
    objective: float         # pi(M) - lambda * F(M), Eq. 3
    n_assigned: int = 0
    n_requests: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def total_utility(utilities: np.ndarray) -> float:
    """pi(M) = sum_v o_v (Eq. 1)."""
    return float(np.sum(utilities))


def fairness(utilities: np.ndarray) -> float:
    """F(M) = Var(o_v) across drivers (Eq. 2). Population variance."""
    return float(np.var(utilities))


def normalised_fairness(utilities: np.ndarray) -> float:
    """\\hat F(M) = sigma(U) / mean(U) (Eq. 8).

    Sec. 5.3 uses this because the raw variance scales with the attained utility.
    """
    mean = float(np.mean(utilities))
    if mean == 0.0:
        return float("nan")
    return float(np.std(utilities) / mean)


def evaluate(
    utilities: np.ndarray,
    lam: float = 1.0,
    n_assigned: int = 0,
    n_requests: int = 0,
) -> Evaluation:
    u = np.asarray(utilities, dtype=np.float64)
    pi = total_utility(u)
    f = fairness(u)
    return Evaluation(
        total_utility=pi,
        fairness=f,
        normalised_fairness=normalised_fairness(u),
        min_utility=float(np.min(u)),
        mean_utility=float(np.mean(u)),
        max_utility=float(np.max(u)),
        objective=pi - lam * f,
        n_assigned=n_assigned,
        n_requests=n_requests,
    )
