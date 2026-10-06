"""Point-to-point routing for vehicles on the live (realised) traffic.

Every leg a cab drives (to a pickup, with a passenger, repositioning, to a gateway) is
routed on the edge times in force when the leg starts, and those times are frozen
for the leg. Edge times are refreshed from `TrafficModel` every `refresh_s` seconds of
simulated time.

The search is a numba Dijkstra with early exit at the target on a CSR adjacency, so
one leg costs about a millisecond even on Pune's 33k-node graph. That is fast enough
to route every leg of a simulated day exactly instead of approximating with zone
matrices.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numba
import numpy as np

from .traffic import DayConditions, Net, TrafficModel


@numba.njit(cache=True)
def _dijkstra_pt(indptr, adj_edge, eu, ev, w, s, t, dist, prev_e, touched):
    """Shortest path s -> t on edge weights w.

    `dist` (inf) and `prev_e` (-1) are reusable buffers; every node this call touches
    is restored before returning. Returns (cost, edge ids along the path).
    """
    n_touched = 1
    touched[0] = s
    dist[s] = 0.0
    heap = [(0.0, s)]
    found = s == t
    while len(heap) > 0 and not found:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == t:
            found = True
            break
        for k in range(indptr[u], indptr[u + 1]):
            e = adj_edge[k]
            v = ev[e]
            nd = d + w[e]
            if nd < dist[v]:
                if dist[v] == np.inf:
                    touched[n_touched] = v
                    n_touched += 1
                dist[v] = nd
                prev_e[v] = e
                heapq.heappush(heap, (nd, v))
    cost = dist[t] if found else np.inf
    n = 0
    if found:
        v = t
        while v != s:
            n += 1
            v = eu[prev_e[v]]
    path = np.empty(n, np.int64)
    if found:
        v = t
        i = n - 1
        while v != s:
            path[i] = prev_e[v]
            v = eu[prev_e[v]]
            i -= 1
    for i in range(n_touched):
        dist[touched[i]] = np.inf
        prev_e[touched[i]] = -1
    return cost, path


@dataclass
class Leg:
    """A driven path: node sequence and the absolute time the cab reaches each node."""
    nodes: np.ndarray      # node positions, len = n_edges + 1
    times: np.ndarray      # seconds since sim start, same length
    length_m: float

    @property
    def start(self) -> float:
        return float(self.times[0])

    @property
    def end(self) -> float:
        return float(self.times[-1])

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    def position_at(self, t: float) -> int:
        """Last node reached by time t (the cab's node for re-routing / matching)."""
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        return int(self.nodes[min(max(i, 0), len(self.nodes) - 1)])

    def truncate(self, t: float) -> "Leg":
        """The part of the leg already driven by time t (the cab stops at the last node
        reached), used when a repositioning cab is diverted to a pickup."""
        i = max(int(np.searchsorted(self.times, t, side="right")), 1)
        frac = i / max(len(self.nodes) - 1, 1)
        return Leg(self.nodes[:i].copy(), self.times[:i].copy(), self.length_m * min(frac, 1.0))


class LiveRouter:
    def __init__(self, net: Net, traffic: TrafficModel, refresh_s: float = 300.0):
        self.net = net
        self.traffic = traffic
        self.refresh_s = refresh_s
        N = net.n_nodes
        order = np.argsort(net.u, kind="stable")
        self.adj_edge = order.astype(np.int64)
        self.indptr = np.searchsorted(net.u[order], np.arange(N + 1)).astype(np.int64)
        self.eu = net.u.astype(np.int64)
        self.ev = net.v.astype(np.int64)
        self.length = net.length
        self._dist = np.full(N, np.inf)
        self._prev = np.full(N, -1, np.int64)
        self._touched = np.empty(N, np.int64)
        self.w = net.t0.copy()
        self._slot = None
        self.day: DayConditions | None = None
        self.n_routes = 0

    def set_day(self, day: DayConditions):
        self.day = day
        self._slot = None

    def update(self, t: float):
        """Refresh edge times when the sim clock enters a new refresh slot."""
        slot = int(t // self.refresh_s)
        if slot != self._slot:
            self._slot = slot
            # times for the middle of the slot
            self.w = self.traffic.edge_times(self.day, (slot + 0.5) * self.refresh_s % 86400)

    def route(self, s: int, t: int, t_start: float) -> Leg:
        self.update(t_start)
        self.n_routes += 1
        cost, path = _dijkstra_pt(self.indptr, self.adj_edge, self.eu, self.ev, self.w,
                                  int(s), int(t), self._dist, self._prev, self._touched)
        if not np.isfinite(cost):
            raise RuntimeError(f"no path {s} -> {t}; the graph should be strongly connected")
        nodes = np.empty(len(path) + 1, np.int64)
        nodes[0] = s
        nodes[1:] = self.ev[path]
        times = np.empty(len(path) + 1)
        times[0] = t_start
        times[1:] = t_start + np.cumsum(self.w[path])
        return Leg(nodes, times, float(self.length[path].sum()))
