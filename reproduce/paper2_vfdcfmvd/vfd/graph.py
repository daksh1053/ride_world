"""Road network and the shortest-path cache (paper Definition 1, Sec. VI-A).

"The road network is modeled as a weighted graph G = (L, E) ... The function
dis(l_i, l_j) indicates both the shortest path between nodes l_i and l_j and the
weight of the edge <l_i, l_j>."

Sec. VI-A: "To ensure that the distance between any two nodes can be quickly queried
during experiments, we prebuilt the cache of the shortest path matrix and the shortest
path distance matrix." Both are built once here and held in memory.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall


class ZoneGraph:
    """Manhattan taxi zones with an all-pairs shortest-distance cache (km)."""

    def __init__(self, weights: np.ndarray, zone_ids: np.ndarray | None = None):
        self.n = weights.shape[0]
        self.zone_ids = zone_ids
        self.edge_weights = weights
        self._dist, self._pred = self._build_cache(weights)
        self._nearby = np.argsort(self._dist, axis=1)

    @staticmethod
    def _build_cache(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        adj = np.where(np.isnan(weights), 0.0, weights)
        dist, pred = floyd_warshall(csr_matrix(adj), directed=True, return_predecessors=True)
        finite = dist[np.isfinite(dist)]
        fallback = finite.max() if finite.size else 0.0
        dist = np.where(np.isfinite(dist), dist, fallback)
        return np.asarray(dist, dtype=np.float64), pred

    def dis(self, a, b):
        """Shortest travel distance (km) from zone `a` to zone `b`."""
        return self._dist[a, b]

    @property
    def matrix(self) -> np.ndarray:
        return self._dist

    def nearby(self, zone: int, k: int) -> np.ndarray:
        """The `k` closest zones to `zone`, excluding itself — the `g_near` of Alg. 2."""
        return self._nearby[zone, 1 : k + 1]

    def path(self, a: int, b: int) -> list[int]:
        """Shortest path from `a` to `b` (the cached predecessor matrix)."""
        if a == b:
            return [a]
        out = [b]
        while out[-1] != a:
            nxt = self._pred[a, out[-1]]
            if nxt < 0:
                return [a, b]          # disconnected: fall back to the direct hop
            out.append(int(nxt))
        return out[::-1]

    @classmethod
    def from_processed(cls, bundle: dict) -> "ZoneGraph":
        return cls(bundle["weights"], bundle.get("zone_ids"))
