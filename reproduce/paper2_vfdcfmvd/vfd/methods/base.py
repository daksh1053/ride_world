"""Common interface for order-matching / idle-dispatching algorithms.

Each algorithm is called once per time slot with the current order pool:

    match(platform, batch)   -> (order positions in the batch, driver ids)
    dispatch(platform, batch)-> reposition idle vehicles (may be a no-op)
    update(platform, ...)    -> learning hook (may be a no-op)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import linear_sum_assignment

EMPTY = (np.empty(0, dtype=int), np.empty(0, dtype=int))


class Algorithm(ABC):
    name = "algorithm"

    @abstractmethod
    def match(self, platform, batch) -> tuple[np.ndarray, np.ndarray]:
        """Return matched (order positions within `batch`, driver ids)."""

    def dispatch(self, platform, batch) -> None:
        """Idle-vehicle dispatching. Baselines that ignore it leave this empty."""

    def update(self, platform, batch, matched_orders, matched_drivers) -> None:
        """Learning hook."""

    def reset(self) -> None:
        """Called before each evaluation run."""


# Each order keeps only its best `TOP_K` admissible drivers before the Hungarian
# solve. The paper's own complexity analysis (Eq. 15) is cubic in the bipartite
# graph size, and Sec. VI-C attributes the running-time differences to exactly that
# size, so bounding an order's candidate list is the standard way to keep KM
# tractable on city-scale pools. Raising it toward the driver count recovers the
# exact maximum-weight matching.
TOP_K = 12


def _sparsify(allowed: np.ndarray, weights: np.ndarray, top_k: int) -> np.ndarray:
    """Keep each row's `top_k` highest-weight admissible entries."""
    if allowed.shape[1] <= top_k:
        return allowed
    masked = np.where(allowed, weights, -np.inf)
    idx = np.argpartition(-masked, top_k - 1, axis=1)[:, :top_k]
    kept = np.zeros_like(allowed)
    np.put_along_axis(kept, idx, True, axis=1)
    return kept & allowed


def km_match(
    weights: np.ndarray, allowed: np.ndarray, top_k: int = TOP_K
) -> tuple[np.ndarray, np.ndarray]:
    """Kuhn-Munkres maximum-weight bipartite matching (the paper's KM algorithm).

    `weights` is `(m_orders, n_drivers)`; `allowed` masks pairs that must not be
    matched. Returns the matched row/column positions.
    """
    m, n = weights.shape
    if m == 0 or n == 0 or not allowed.any():
        return EMPTY

    allowed = _sparsify(allowed, weights, top_k)

    # Drop rows and columns with no admissible pair — this is what keeps the
    # Hungarian cost matrix small (Sec. VI-C notes the bipartite graph size drives
    # the running time).
    rows = np.flatnonzero(allowed.any(axis=1))
    cols = np.flatnonzero(allowed.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return EMPTY

    sub = weights[np.ix_(rows, cols)].astype(np.float64)
    sub_allowed = allowed[np.ix_(rows, cols)]
    penalty = np.abs(sub).max() * 1e6 + 1.0
    cost = np.where(sub_allowed, -sub, penalty)

    r, c = linear_sum_assignment(cost)
    keep = sub_allowed[r, c]
    return rows[r[keep]], cols[c[keep]]


def greedy_match(weights: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy maximum-weight matching — used by the `w/o driver-order matching`
    ablation, which "substitutes the value-function-guided matching with a greedy
    approach" (Sec. VI-D)."""
    m, n = weights.shape
    if m == 0 or n == 0 or not allowed.any():
        return EMPTY
    scores = np.where(allowed, weights, -np.inf)
    order = np.argsort(scores, axis=None)[::-1]
    used_r, used_c = set(), set()
    rows, cols = [], []
    for flat in order:
        i, j = divmod(int(flat), n)
        if not np.isfinite(scores[i, j]):
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        rows.append(i)
        cols.append(j)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)
