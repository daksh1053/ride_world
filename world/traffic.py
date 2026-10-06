"""Traffic: time-varying edge travel times, the traffic part of W_t in the formulation.

Two layers:

1. **Typical conditions (offline, `build_typical`).** For each day type (weekday,
   weekend) and hour, a background origin-destination (OD) matrix of private traffic
   is assigned to the road graph with the method of successive averages (MSA) under
   BPR link delays, t = t0 * (1 + a (v/c)^b) with a steeper curve on freeways.
   This gives an approximate user
   equilibrium. The hour's total demand is solved for (secant steps in log-log space) until the
   flow-weighted network travel-time index, TTI = sum v t / sum v t0, hits that hour's
   target from `TrafficTargets`, which is calibrated to TomTom's city figures.
   Stored: background volume v[daytype, hour, edge] and the resulting times.

2. **Realised conditions (runtime, `TrafficModel`).** A simulated day draws a demand
   factor, rain and incidents. Edge times at any second of the day come from BPR on
   the interpolated background volume, those draws, and an optional extra volume from
   the ride fleet itself. That is the feedback hook for stage 3.

The background OD is a doubly shaped gravity model between zones:
    T_ij ∝ P_i A_j exp(-d_ij / L)
where P is residential (home) weight, A is activity weight and L is `gravity_km`.
The morning pattern runs P->A, the evening pattern A->P and the off-peak pattern is
their average. Each hour mixes the three with Gaussian weights around the peak hours.
Every zone loads and unloads at `K_LOAD` nodes redrawn in each MSA iteration from all
its nodes (weighted towards arterials), and each iteration routes on noisy link times
(probit route choice). So traffic starts on side streets too and spreads over
parallel routes instead of piling onto one shortest path between a few points.

External traffic enters and leaves through *gateways*: motorway, trunk and primary
nodes near the city boundary, weighted by the capacity of their roads. A share
`external_share` of all trips has at least one end outside the city. In the morning
it mostly enters (gateway -> activity zones), in the evening it leaves, and a
fraction `through_share` of it crosses the city from gateway to gateway. Without this,
the city's freeways would carry only intra-city trips and never congest.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .cities import City

# ASSUMPTION: per-lane capacity (vehicles/hour) by road class. Standard HCM-order
# values for urban roads with signals; what matters for the model is their ratio,
# because the absolute demand level is calibrated to the TTI targets.
LANE_CAPACITY = {
    "motorway": 1900, "trunk": 1500, "primary": 1100, "secondary": 900,
    "tertiary": 700, "residential": 450, "unclassified": 450,
    "living_street": 250, "service": 250, "other": 300,
}
# BPR delay curve per road class: t = t0 (1 + alpha (v/c)^beta). Classic BPR
# (0.15, 4) adds only 15% delay at capacity, so calibrated freeways looked free-flowing
# while arterials jammed. Freeways use the updated curve from NCHRP Report 365
# (alpha ~0.83, beta ~5.5), which breaks down sharply near capacity.
BPR_DEFAULT = (0.15, 4.0)
BPR_BY_CLASS = {"motorway": (0.83, 5.5), "trunk": (0.83, 5.5)}
K_LOAD = 10                # loading nodes per zone per MSA iteration (redrawn each iteration)
MSA_ITERS = 8              # MSA iterations per assignment
ROUTE_SIGMA = 0.25         # lognormal perception noise on link times (probit route choice)
PERTURB_SEED = 1234
AON_BATCH = 256            # Dijkstra sources per batch (memory bound)
MAX_EVALS = 10             # assignments per hour while solving for the demand level
TTI_TOL = 0.01
GATEWAY_CLASSES = ("motorway", "trunk", "primary")
GATEWAY_SPACING_M = 2000   # keep one gateway per cluster of nearby candidates
DAYTYPES = ("weekday", "weekend")


# ---------------------------------------------------------------- network arrays

@dataclass
class Net:
    n_nodes: int
    u: np.ndarray          # int32 node index
    v: np.ndarray
    t0: np.ndarray         # free-flow seconds
    cap: np.ndarray        # vehicles / hour
    alpha: np.ndarray      # BPR parameters per edge
    beta: np.ndarray
    length: np.ndarray     # metres
    node_ids: np.ndarray   # int64 OSM ids, position = node index

    @classmethod
    def from_tables(cls, nodes: pd.DataFrame, edges: pd.DataFrame) -> "Net":
        idx = pd.Series(np.arange(len(nodes), dtype=np.int32), index=nodes.node_id.to_numpy())
        lanes = edges.lanes_dir.to_numpy()
        lanes = np.where(lanes > 0, lanes, 1.0)                  # guard: never zero capacity
        cap = lanes * edges.road_class.map(LANE_CAPACITY).to_numpy()
        return cls(
            n_nodes=len(nodes),
            u=idx.loc[edges.u].to_numpy(), v=idx.loc[edges.v].to_numpy(),
            t0=np.maximum(edges.free_flow_s.to_numpy(float), 0.1),
            cap=cap.astype(float),
            alpha=edges.road_class.map(lambda c: BPR_BY_CLASS.get(c, BPR_DEFAULT)[0]).to_numpy(float),
            beta=edges.road_class.map(lambda c: BPR_BY_CLASS.get(c, BPR_DEFAULT)[1]).to_numpy(float),
            length=edges.length_m.to_numpy(float),
            node_ids=nodes.node_id.to_numpy(),
        )


def bpr(t0, vol, cap, alpha, beta):
    return t0 * (1.0 + alpha * (vol / cap) ** beta)


def tti(vol, times, t0) -> float:
    w = vol.sum()
    return float((vol * times).sum() / (vol * t0).sum()) if w > 0 else 1.0


class Router:
    """Shortest-path trees on the current edge times.

    Parallel edges (same u, v) collapse to the fastest one; the CSR used to look up
    an edge from a (pred, node) pair only contains those chosen edges.
    """

    def __init__(self, net: Net, times: np.ndarray):
        order = np.lexsort((times, net.v, net.u))
        u, v = net.u[order], net.v[order]
        first = np.ones(len(order), bool)
        first[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
        self.edge = order[first].astype(np.int64)   # chosen edge ids, sorted by (u, v)
        cu, cv = net.u[self.edge], net.v[self.edge]
        self.graph = csr_matrix((times[self.edge], (cu, cv)), shape=(net.n_nodes, net.n_nodes))
        self.indptr = np.searchsorted(cu, np.arange(net.n_nodes + 1)).astype(np.int64)
        self.targets = cv.astype(np.int32)

    def trees(self, sources: np.ndarray):
        dist, pred = dijkstra(self.graph, directed=True, indices=sources, return_predecessors=True)
        return dist, pred.astype(np.int32)


@numba.njit(cache=True)
def _find_edge(indptr, targets, edge, p, n):
    for k in range(indptr[p], indptr[p + 1]):
        if targets[k] == n:
            return edge[k]
    return -1


@numba.njit(cache=True)
def _load_trees(dist, pred, dest_nodes, demand, indptr, targets, edge, n_edges):
    """All-or-nothing loading: push each source's demand up its shortest-path tree."""
    S, N = dist.shape
    flow = np.zeros(n_edges)
    node_flow = np.zeros(N)
    for s in range(S):
        node_flow[:] = 0.0
        for k in range(dest_nodes.shape[0]):
            node_flow[dest_nodes[k]] += demand[s, k]
        order = np.argsort(-dist[s])
        for i in range(N):
            n = order[i]
            f = node_flow[n]
            p = pred[s, n]
            if f == 0.0 or p < 0:
                continue
            node_flow[p] += f
            flow[_find_edge(indptr, targets, edge, p, n)] += f
    return flow


@numba.njit(cache=True)
def _tree_lengths(dist, pred, indptr, targets, edge, length):
    """Length (m) of every shortest-time path: accumulate edge lengths down each tree."""
    S, N = dist.shape
    out = np.full((S, N), np.inf)
    for s in range(S):
        order = np.argsort(dist[s])
        for i in range(N):
            n = order[i]
            if not np.isfinite(dist[s, n]):
                break
            p = pred[s, n]
            if p < 0:
                out[s, n] = 0.0
            else:
                out[s, n] = out[s, p] + length[_find_edge(indptr, targets, edge, p, n)]
    return out


# ---------------------------------------------------------------- demand

def zone_weights(zones: pd.DataFrame, edges: pd.DataFrame, nodes: pd.DataFrame):
    """Home (P) and activity (A) weights per zone.

    ASSUMPTION: without a population layer, residential road length stands in for
    where people live, and POIs plus major road length for where they go. Both are
    normalised to sum 1, and each gets a 10% uniform floor so no zone is dead.
    """
    zone_of = nodes.set_index("node_id").zone
    ez = edges.u.map(zone_of)
    res_km = edges.length_m.where(edges.road_class.isin(["residential", "living_street"]), 0) \
        .groupby(ez).sum().reindex(zones.zone).fillna(0).to_numpy() / 1000
    major_km = edges.length_m.where(edges.road_class.isin(["trunk", "primary", "secondary"]), 0) \
        .groupby(ez).sum().reindex(zones.zone).fillna(0).to_numpy() / 1000
    poi = zones.n_pois.to_numpy(float)

    def norm(x):
        x = x / x.sum()
        return 0.9 * x + 0.1 / len(x)

    return norm(res_km), norm(norm(poi) + 0.5 * norm(major_km))


def gravity_od(zones, P, A, decay_km) -> np.ndarray:
    """Home->activity gravity matrix T_ij ∝ P_i A_j exp(-d_ij / L), unnormalised.
    Intra-zone distance is half the zone's characteristic size."""
    xy = zones[["x_m", "y_m"]].to_numpy() / 1000
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    d[np.diag_indices_from(d)] = 0.5 * np.sqrt(zones.area_km2.clip(lower=0.1).to_numpy())
    return P[:, None] * A[None, :] * np.exp(-d / decay_km)


def od_patterns(zones, P, A, E, tgt):
    """AM / PM / off-peak OD shapes over Z zones + G gateways, each summing to 1.

    E is the gateway weight vector (sums to 1). Rows/columns Z.. are gateways.
    """
    Z, G = len(zones), len(E)
    internal = gravity_od(zones, P, A, tgt.gravity_km)
    phi, theta = tgt.external_share, tgt.through_share
    am = np.zeros((Z + G, Z + G))
    am[:Z, :Z] = (1 - phi) * internal / internal.sum()
    # ASSUMPTION: 70% of non-through external trips in the morning are inbound.
    am[Z:, :Z] = phi * (1 - theta) * 0.7 * np.outer(E, A)
    am[:Z, Z:] = phi * (1 - theta) * 0.3 * np.outer(P, E)
    thr = np.outer(E, E)
    np.fill_diagonal(thr, 0)
    am[Z:, Z:] = phi * theta * thr / thr.sum()
    am /= am.sum()
    pm = am.T.copy()
    return {"am": am, "pm": pm, "off": 0.5 * (am + pm)}


def find_gateways(nodes: pd.DataFrame, edges: pd.DataFrame, boundary, crs, max_dist_m) -> pd.DataFrame:
    """Major-road nodes near the city boundary, one per cluster, weighted by capacity."""
    import geopandas as gpd
    cap = edges.lanes_dir * edges.road_class.map(LANE_CAPACITY)
    major = edges.road_class.isin(GATEWAY_CLASSES)
    score = pd.concat([cap[major].groupby(edges.u[major]).sum(),
                       cap[major].groupby(edges.v[major]).sum()]).groupby(level=0).sum()
    cand = nodes[nodes.node_id.isin(score.index)].copy()
    cand["score"] = score.loc[cand.node_id].to_numpy()
    line = boundary.to_crs(crs).geometry.iloc[0].boundary
    pts = gpd.GeoSeries(gpd.points_from_xy(cand.x_m, cand.y_m), crs=crs)
    cand["dist_m"] = pts.distance(line).to_numpy()
    near = cand[cand.dist_m < max_dist_m]
    if len(near) < 3:
        # few major roads reach the boundary (small cities): the 6 major-road nodes
        # closest to it, so external traffic still has somewhere to enter
        near = cand.nsmallest(6, "dist_m")
    cand = near.sort_values("score", ascending=False)
    keep = []
    for r in cand.itertuples():
        if all((r.x_m - k.x_m) ** 2 + (r.y_m - k.y_m) ** 2 > GATEWAY_SPACING_M ** 2 for k in keep):
            keep.append(r)
    g = pd.DataFrame(keep)[["node_id", "lon", "lat", "x_m", "y_m", "score", "dist_m"]]
    g["weight"] = g.score / g.score.sum()
    return g.reset_index(drop=True)


def hour_mix(hour: float, daytype: str, tgt) -> dict[str, float]:
    if daytype == "weekend":
        return {"am": 0.0, "pm": 0.0, "off": 1.0}
    g = lambda c: float(np.exp(-0.5 * ((hour - c) / 1.5) ** 2))  # noqa: E731
    w_am, w_pm = 0.8 * g(tgt.am_peak_hour), 0.8 * g(tgt.pm_peak_hour)
    return {"am": w_am, "pm": w_pm, "off": max(0.0, 1.0 - w_am - w_pm)}


# ---------------------------------------------------------------- assignment

_W: dict = {}  # worker globals (inherited through fork)


def _setup(net: Net, load_sets: list[np.ndarray], load_zone: np.ndarray):
    _W.update(net=net, load_sets=load_sets, load_zone=load_zone)


def aon(net: Net, times: np.ndarray, load_nodes, demand) -> np.ndarray:
    """All-or-nothing load of a node OD. Sources run in batches of AON_BATCH so the
    (sources x nodes) Dijkstra arrays stay small."""
    r = Router(net, times)
    flow = np.zeros(len(net.u))
    for b in range(0, len(load_nodes), AON_BATCH):
        dist, pred = r.trees(load_nodes[b:b + AON_BATCH])
        flow += _load_trees(dist, pred, load_nodes, demand[b:b + AON_BATCH],
                            r.indptr, r.targets, r.edge, len(net.u))
    return flow


def msa(net: Net, load_sets, demand, iters, seed=PERTURB_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Method of successive averages with two sources of spread.

    * Iteration k loads trips at node set `load_sets[k]`: the same zones, different
      nodes. Over the iterations every zone's traffic enters the network across much
      of its street grid, not at a handful of points.
    * Each all-or-nothing step routes on travel times multiplied by independent
      lognormal noise (sigma ROUTE_SIGMA): Monte-Carlo probit route choice. Near-equal
      parallel streets share the load instead of one taking all of it.

    The noise uses a fixed seed per iteration (common random numbers), so the result
    is a deterministic function of the demand, as the calibration requires.
    """
    rng = np.random.default_rng(seed)
    vol = None
    for k in range(iters):
        base = net.t0 if vol is None else bpr(net.t0, vol, net.cap, net.alpha, net.beta)
        noise = np.exp(ROUTE_SIGMA * rng.standard_normal(len(base)) - ROUTE_SIGMA ** 2 / 2)
        y = aon(net, base * noise, load_sets[k % len(load_sets)], demand)
        vol = y if vol is None else vol + (y - vol) / (k + 1)
    return vol, bpr(net.t0, vol, net.cap, net.alpha, net.beta)


def _node_demand(od_zone: np.ndarray, load_zone: np.ndarray) -> np.ndarray:
    """Zone OD (veh/h) -> loading-node OD, split evenly over the K x K node pairs."""
    k = np.bincount(load_zone)
    return od_zone[load_zone][:, load_zone] / (k[load_zone][:, None] * k[load_zone][None, :])


def calibrate_hour(args) -> dict:
    daytype, hour, target, od_shape = args
    net, load_sets, load_zone = _W["net"], _W["load_sets"], _W["load_zone"]
    shape = _node_demand(od_shape, load_zone)
    np.fill_diagonal(shape, 0.0)

    best = None  # (|tti - target|, total, vol, times, tti)
    tried = []   # (log demand, log(tti - 1))

    def consider(total):
        nonlocal best
        vol, times = msa(net, load_sets, shape * total, MSA_ITERS)
        x = tti(vol, times, net.t0)
        if best is None or abs(x - target) < best[0]:
            best = (abs(x - target), total, vol, times, x)
        tried.append((np.log(total), np.log(max(x - 1, 1e-5))))
        return x

    # BPR makes the excess delay TTI-1 roughly a power of demand, so solve
    # log(TTI-1) = log(target-1) by secant steps in log-log space. The first step
    # assumes exponent 3. Every evaluation uses the same MSA iteration count, so
    # the returned point is exactly one that was scored.
    goal = np.log(max(target - 1, 1e-4))
    total = 150000.0
    for _ in range(MAX_EVALS):
        consider(total)
        if best[0] < TTI_TOL:
            break
        (x1, y1) = tried[-1]
        if len(tried) >= 2 and abs(y1 - tried[-2][1]) > 1e-6:
            x0, y0 = tried[-2]
            slope = np.clip((y1 - y0) / (x1 - x0), 0.5, 8.0)
        else:
            slope = 3.0
        total = float(np.exp(x1 + np.clip((goal - y1) / slope, -1.5, 1.5)))
    _, total, vol, times, x = best
    return {"daytype": daytype, "hour": hour, "target": target, "total_vph": total,
            "tti": x, "vol": vol.astype(np.float32),
            "times": times.astype(np.float32)}


# ASSUMPTION: relative chance that a trip starts/ends at a node, by the best road
# class touching it. Trips can begin on any street, but more of them join at arterials.
LOAD_NODE_WEIGHT = {"motorway": 0.0, "trunk": 2.0, "primary": 4.0, "secondary": 4.0,
                    "tertiary": 2.5, "residential": 1.0, "unclassified": 1.0,
                    "living_street": 0.5, "service": 0.5, "other": 0.5}


def pick_load_sets(nodes: pd.DataFrame, edges: pd.DataFrame, n_sets: int, seed=0):
    """`n_sets` draws of K_LOAD nodes per zone, in the same zone order every time.

    Returns (list of node-position arrays, zone of each position). Motorway nodes are
    excluded because a trip cannot start on a freeway.
    """
    rng = np.random.default_rng(seed)
    w_edge = edges.road_class.map(LOAD_NODE_WEIGHT)
    w_node = pd.concat([w_edge.groupby(edges.u).max(), w_edge.groupby(edges.v).max()]) \
        .groupby(level=0).max()
    w = w_node.reindex(nodes.node_id).fillna(0).to_numpy()
    groups = [(z, grp.index.to_numpy()) for z, grp in nodes.groupby("zone")]
    sets = []
    for _ in range(n_sets):
        pos = []
        for z, idx in groups:
            p = w[idx] / w[idx].sum() if w[idx].sum() > 0 else None
            pos.extend(rng.choice(idx, size=K_LOAD, replace=len(idx) < K_LOAD, p=p))
        sets.append(np.array(pos, np.int64))
    zone = np.repeat([z for z, _ in groups], K_LOAD).astype(np.int64)
    return sets, zone


def build_typical(city: City, nodes, edges, zones, boundary, workers=None, log=print) -> dict:
    tgt = city.traffic
    net = Net.from_tables(nodes, edges)
    load_sets, load_zone = pick_load_sets(nodes, edges, MSA_ITERS)
    gw = find_gateways(nodes, edges, boundary, city.utm_crs, tgt.gateway_max_dist_m)
    node_pos = pd.Series(np.arange(len(nodes)), index=nodes.node_id)
    gw_pos = node_pos.loc[gw.node_id].to_numpy()
    load_sets = [np.r_[s, gw_pos] for s in load_sets]
    load_zone = np.r_[load_zone, len(zones) + np.arange(len(gw))]
    n_distinct = len(np.unique(np.concatenate(load_sets)))
    log(f"  {len(zones)} zones x {K_LOAD} load nodes x {MSA_ITERS} iterations "
        f"({n_distinct:,} distinct nodes, {n_distinct / len(nodes):.0%} of all) + {len(gw)} gateways "
        f"(external share {tgt.external_share:.0%}, through {tgt.through_share:.0%} of that)")
    P, A = zone_weights(zones, edges, nodes)
    pats = od_patterns(zones, P, A, gw.weight.to_numpy(), tgt)

    jobs = []
    for dt, prof in (("weekday", tgt.tti_weekday), ("weekend", tgt.tti_weekend)):
        for h in range(24):
            mix = hour_mix(h + 0.5, dt, tgt)
            jobs.append((dt, h, prof[h], sum(w * pats[k] for k, w in mix.items())))

    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    _setup(net, load_sets, load_zone)
    results = []
    with ProcessPoolExecutor(workers, initializer=_setup, initargs=(net, load_sets, load_zone)) as ex:
        for r in ex.map(calibrate_hour, jobs):
            results.append(r)
            log(f"  {r['daytype']:>7} {r['hour']:02d}h  target TTI {r['target']:.2f}  "
                f"got {r['tti']:.3f}  demand {r['total_vph']:,.0f} veh/h")

    E = len(net.u)
    vol = np.zeros((2, 24, E), np.float32)
    times = np.zeros((2, 24, E), np.float32)
    for r in results:
        d = DAYTYPES.index(r["daytype"])
        vol[d, r["hour"]], times[d, r["hour"]] = r["vol"], r["times"]
    calib = pd.DataFrame([{k: r[k] for k in ("daytype", "hour", "target", "tti", "total_vph")}
                          for r in results])
    return {"net": net, "vol": vol, "times": times, "calib": calib,
            "load_sets": load_sets, "P": P, "A": A, "patterns": pats, "gateways": gw}


def zone_matrices(net: Net, times: np.ndarray, centre_idx: np.ndarray):
    """Fastest-path travel time (s) and that path's length (m) between zone centres."""
    r = Router(net, times)
    dist, pred = r.trees(centre_idx)
    lens = _tree_lengths(dist, pred, r.indptr, r.targets, r.edge, net.length)
    return dist[:, centre_idx].astype(np.float32), lens[:, centre_idx].astype(np.float32)


# ---------------------------------------------------------------- runtime model

@dataclass
class DayConditions:
    date: pd.Timestamp
    daytype: str
    demand_factor: float
    rain: tuple[float, float, float] | None   # (start_s, end_s, intensity 0..1) or None
    incidents: pd.DataFrame                    # edge, start_s, end_s, cap_factor

    def rain_at(self, t_s: float) -> float:
        """Rain intensity at a second of the day: 0 when dry, ramping in and out over
        RAIN_RAMP_S at the ends of the spell."""
        if self.rain is None:
            return 0.0
        a, b, k = self.rain
        return k * float(np.clip(min(t_s - a, b - t_s) / RAIN_RAMP_S, 0.0, 1.0))

    def summary(self) -> dict:
        r = self.rain
        return {"date": str(self.date.date()), "daytype": self.daytype,
                "demand_factor": round(self.demand_factor, 3),
                "rain": "-" if r is None else f"{r[0] / 3600:04.1f}h-{min(r[1], 86400) / 3600:04.1f}h @ {r[2]:.2f}",
                "n_incidents": int(len(self.incidents))}


# ASSUMPTION: at full intensity rain lowers capacity by 8%, free-flow speed by 6% and
# raises car demand by 3% (a shift from two-wheelers / walking). Effects scale with
# intensity. That gives roughly +10-25% peak travel time in a spell, the range
# reported for urban rain. The steep BPR curve amplifies larger values unrealistically.
RAIN_CAP, RAIN_SPEED, RAIN_DEMAND = 0.08, 0.06, 0.03
RAIN_MEDIAN_HOURS = 5.0
RAIN_RAMP_S = 1800.0
# ASSUMPTION: with no re-routing at runtime, part of an incident link's traffic diverts:
# its volume scales by cap_factor ** INCIDENT_DIVERSION. The link's v/c then rises by
# cap_factor ** -0.3 instead of 1 / cap_factor, which keeps a single blocked link from
# dominating the city-wide TTI.
INCIDENT_DIVERSION = 0.7
DEMAND_SIGMA = 0.06   # day-to-day lognormal demand variation


class TrafficModel:
    """Edge travel times for any simulated moment, drawn around the typical profile."""

    def __init__(self, city: City):
        d = city.data_dir / "traffic"
        self.city = city
        self.vol = np.load(d / "bg_volume.npy", mmap_mode="r")     # (2, 24, E) veh/h
        ep = pd.read_parquet(d / "edge_params.parquet")
        self.t0 = ep.free_flow_s.to_numpy()
        self.cap = ep.capacity_vph.to_numpy()
        self.alpha = ep.bpr_alpha.to_numpy()
        self.beta = ep.bpr_beta.to_numpy()
        self.major = ep.road_class.isin(["motorway", "trunk", "primary"]).to_numpy()
        self.length = ep.length_m.to_numpy()

    def sample_day(self, date, rng: np.random.Generator) -> DayConditions:
        date = pd.Timestamp(date)
        tgt = self.city.traffic
        daytype = "weekend" if date.weekday() >= 5 else "weekday"
        rain = None
        if rng.random() < tgt.rain_prob_by_month[date.month - 1]:
            start = rng.uniform(0, 24 * 3600)
            dur = np.exp(rng.normal(np.log(RAIN_MEDIAN_HOURS * 3600), 0.6))
            rain = (float(start), float(start + dur), float(rng.uniform(0.3, 1.0)))
        demand = float(np.exp(rng.normal(0, DEMAND_SIGMA)))
        major_km = self.length[self.major].sum() / 1000
        rate = tgt.incidents_per_100km_day * major_km / 100 * (1.3 if rain else 1.0)
        n = rng.poisson(rate)
        edges = rng.choice(np.flatnonzero(self.major), size=n, replace=True)
        start = rng.uniform(6 * 3600, 23 * 3600, size=n)
        dur = np.exp(rng.normal(np.log(45 * 60), 0.5, size=n))
        inc = pd.DataFrame({"edge": edges, "start_s": start, "end_s": start + dur,
                            "cap_factor": rng.uniform(0.3, 0.6, size=n)})
        return DayConditions(date, daytype, demand, rain, inc)

    def background_volume(self, day: DayConditions, t_s: float) -> np.ndarray:
        """Linear interpolation between hourly slices, centred on the half hour."""
        d = DAYTYPES.index(day.daytype)
        x = (t_s / 3600.0 - 0.5) % 24
        h0 = int(np.floor(x)); w = x - h0
        v = (1 - w) * self.vol[d, h0] + w * self.vol[d, (h0 + 1) % 24]
        return v * day.demand_factor * (1 + RAIN_DEMAND * day.rain_at(t_s))

    def edge_times(self, day: DayConditions, t_s: float, extra_volume=None) -> np.ndarray:
        vol = self.background_volume(day, t_s)
        if extra_volume is not None:
            vol = vol + extra_volume
        k = day.rain_at(t_s)
        cap = self.cap * (1 - RAIN_CAP * k)
        active = day.incidents[(day.incidents.start_s <= t_s) & (day.incidents.end_s > t_s)]
        if len(active):
            e, f = active.edge.to_numpy(), active.cap_factor.to_numpy()
            np.multiply.at(cap, e, f)   # cap is already a fresh array here
            vol = np.array(vol, copy=True)
            np.multiply.at(vol, e, f ** INCIDENT_DIVERSION)
        t0 = self.t0 / (1 - RAIN_SPEED * k)
        return bpr(t0, vol, cap, self.alpha, self.beta)

    def network_tti(self, day: DayConditions, t_s: float) -> float:
        vol = self.background_volume(day, t_s)
        return tti(vol, self.edge_times(day, t_s), self.t0)


def save_typical(city: City, edges: pd.DataFrame, zones: pd.DataFrame, res: dict, log=print):
    d = city.data_dir / "traffic"
    d.mkdir(parents=True, exist_ok=True)
    net = res["net"]
    np.save(d / "bg_volume.npy", res["vol"])
    np.save(d / "typical_times.npy", res["times"])
    pd.DataFrame({
        "u": edges.u, "v": edges.v, "key": edges.key, "road_class": edges.road_class,
        "length_m": edges.length_m, "free_flow_s": net.t0, "capacity_vph": net.cap,
        "bpr_alpha": net.alpha, "bpr_beta": net.beta,
    }).to_parquet(d / "edge_params.parquet", index=False)
    res["calib"].to_csv(d / "calibration.csv", index=False)
    res["gateways"].to_parquet(d / "gateways.parquet", index=False)

    idx = pd.Series(np.arange(len(net.node_ids)), index=net.node_ids)
    centre = idx.loc[zones.centre_node].to_numpy()
    Z = len(zones)
    tt = np.zeros((2, 24, Z, Z), np.float32)
    dist = np.zeros((2, 24, Z, Z), np.float32)
    for di in range(2):
        for h in range(24):
            tt[di, h], dist[di, h] = zone_matrices(net, res["times"][di, h].astype(float), centre)
    np.save(d / "zone_tt_s.npy", tt)
    np.save(d / "zone_dist_m.npy", dist)
    ff_tt, ff_dist = zone_matrices(net, net.t0, centre)
    np.save(d / "zone_tt_freeflow_s.npy", ff_tt)
    np.save(d / "zone_dist_freeflow_m.npy", ff_dist)
    pd.DataFrame({"zone": zones.zone, "home_weight": res["P"], "activity_weight": res["A"]}) \
        .to_parquet(d / "zone_weights.parquet", index=False)
    meta = {
        "bpr_default": BPR_DEFAULT, "bpr_by_class": BPR_BY_CLASS, "lane_capacity_vph": LANE_CAPACITY,
        "k_load_nodes_per_zone_per_iter": K_LOAD, "msa_iters": MSA_ITERS,
        "route_sigma": ROUTE_SIGMA, "load_node_weight": LOAD_NODE_WEIGHT,
        "n_gateways": int(len(res["gateways"])),
        "external_share": city.traffic.external_share, "through_share": city.traffic.through_share,
        "targets_source": city.traffic.source,
        "daily_background_vehicle_trips": {
            dt: float(res["calib"].query("daytype == @dt").total_vph.sum()) for dt in DAYTYPES},
        "max_abs_tti_error": float((res["calib"].tti - res["calib"].target).abs().max()),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"  saved traffic arrays to {d.relative_to(d.parents[2])}")
    return tt, dist
