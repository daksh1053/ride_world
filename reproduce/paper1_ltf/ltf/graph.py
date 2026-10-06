"""Directed road graph and the `Geo(.,.)` shortest-distance operator (Sec. 3.1).

The paper models the city as a directed graph `(L, E)` where the weight of edge
`e_{i,j}` is the travel distance from location `l_i` to `l_j`, explicitly allowing
`e_{i,j} != e_{j,i}`. `Geo(a, b)` is the shortest geographical distance from `a` to
`b`, and the study "assumes that the driver will always choose the shortest path".
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall


class RoadGraph:
    """Directed graph over the merged node set `L` with all-pairs shortest paths."""

    def __init__(self, weights: np.ndarray, zone_ids: np.ndarray | None = None):
        self.n = weights.shape[0]
        self.zone_ids = zone_ids
        self.edge_weights = weights
        self._geo = self._all_pairs(weights)

    @staticmethod
    def _all_pairs(weights: np.ndarray) -> np.ndarray:
        adj = np.where(np.isnan(weights), 0.0, weights)
        dist = floyd_warshall(csr_matrix(adj), directed=True)
        # Unreachable pairs (disconnected in the observed data) fall back to the
        # largest finite distance so that the utility stays well defined.
        finite = dist[np.isfinite(dist)]
        dist = np.where(np.isfinite(dist), dist, finite.max() if finite.size else 0.0)
        return np.asarray(dist, dtype=np.float64)

    def geo(self, a: int | np.ndarray, b: int | np.ndarray) -> float | np.ndarray:
        """Shortest travel distance from node `a` to node `b`."""
        return self._geo[a, b]

    @property
    def matrix(self) -> np.ndarray:
        return self._geo

    @classmethod
    def from_processed(cls, bundle: dict) -> "RoadGraph":
        return cls(bundle["weights"], bundle.get("zone_ids"))
