"""Income, fairness and the five evaluation metrics (paper Sec. III, VI-B)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# A driver whose weighted unit time income is <= 0 (they lost money, or never became
# active) would send log(F_d / max F) to -inf. We floor the ratio at this value, so a
# fully unserved driver contributes -log(FLOOR) instead of infinity.
RATIO_FLOOR = 1e-3


def weighted_unit_time_income(
    income_per_slot: np.ndarray, active_per_slot: np.ndarray, xi: np.ndarray
) -> np.ndarray:
    """F_d of Definition 6:  sum_t (u_d^t / xi^t) / sum_t a_d^t.

    `income_per_slot` and `active_per_slot` are `(n_drivers, n_slots)`.
    """
    weighted = (income_per_slot / xi[None, :]).sum(axis=1)
    active = active_per_slot.sum(axis=1)
    return np.divide(weighted, active, out=np.zeros_like(weighted), where=active > 0)


def temporal_earnings_fairness(f_d: np.ndarray) -> float:
    """Unfairness F of Definition 7 / Eq. 5:  -sum_d log(F_d / max_d' F_d').

    "A larger F means that the weighted unit time income among drivers is more
    dispersed, i.e., less driver fairness in the system." Minimised by Eq. 6.
    """
    f_max = float(np.max(f_d))
    if f_max <= 0:
        return float("nan")
    ratio = np.clip(f_d / f_max, RATIO_FLOOR, 1.0)
    return float(-np.sum(np.log(ratio)))


def jain_index(values: np.ndarray) -> float:
    """Jain's fairness index — part of the DQN state `jain_i` (Sec. V-A)."""
    v = np.asarray(values, dtype=np.float64)
    denom = v.size * float(np.sum(v * v))
    if denom <= 0:
        return 1.0
    return float(np.sum(v) ** 2 / denom)


@dataclass
class Evaluation:
    """The five metrics of Sec. VI-B."""

    unfairness: float          # Definition 7
    total_income: float        # U_total, Eq. 3
    idle_driver_rate: float    # share of drivers that never served an order
    order_service_rate: float  # matched orders / total orders
    running_time: float        # seconds
    n_orders: int = 0
    n_matched: int = 0
    n_drivers: int = 0
    mean_income: float = 0.0
    min_income: float = 0.0
    max_income: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(
    income_per_slot: np.ndarray,
    active_per_slot: np.ndarray,
    xi: np.ndarray,
    n_orders: int,
    n_matched: int,
    running_time: float,
) -> Evaluation:
    income = income_per_slot.sum(axis=1)
    served = active_per_slot.sum(axis=1) > 0
    f_d = weighted_unit_time_income(income_per_slot, active_per_slot, xi)
    return Evaluation(
        unfairness=temporal_earnings_fairness(f_d),
        total_income=float(income.sum()),
        idle_driver_rate=float(np.mean(~served)),
        order_service_rate=float(n_matched / n_orders) if n_orders else 0.0,
        running_time=running_time,
        n_orders=n_orders,
        n_matched=n_matched,
        n_drivers=int(income.size),
        mean_income=float(income.mean()),
        min_income=float(income.min()),
        max_income=float(income.max()),
    )
