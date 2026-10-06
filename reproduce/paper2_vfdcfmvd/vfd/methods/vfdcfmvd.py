"""VFDCFMVD — the proposed algorithm (paper Sec. V, Algorithm 2).

Structure (Fig. 1): Clustering -> Matching -> Dispatching -> Update.

1. **Dynamic clustering** (Sec. V-A). A DQN observes
   `s = (ser_r, idle_r, inc_a, jain_i, t)` and outputs `n`, the number of driver
   clusters for this slot; K-Means then groups drivers by income into `n` clusters.
   "we prioritize fairness enhancement for low-income drivers and emphasize total
   income maximization within similar-income groups."
2. **Matching** (Sec. V-C). Clusters are processed lowest-income first. For each
   cluster a bipartite graph is built over `(O_t, D')`, edge weights are the value
   differences `dV = R_gamma + gamma^{Dt} V(s') - V(s)` of Eq. 14, edges with
   `dV <= 0` are dropped ("remove the matching pair which will decrease the drivers'
   future potential income"), and KM solves the maximum-weight matching.
3. **Dispatching** (Sec. V-C). Each still-idle driver is sent to the nearby zone with
   the largest positive state difference — "assigning a virtual order o, in which the
   payment p_o is set to zero".
4. **Update** (Sec. V-B). V(s) is updated by TD(0), Eq. 13.

The two ablation switches of Table III (`use_clustering`, `use_value_matching`,
`use_dispatching`) toggle modules 1, 2 and 3 respectively.
"""

from __future__ import annotations

import numpy as np

from ..config import CONFIG, AlgorithmConfig
from ..dqn import DQNAgent
from ..metrics import jain_index
from ..value_function import StateValueFunction
from .base import EMPTY, Algorithm, greedy_match, km_match

N_STATE = 5   # (ser_r, idle_r, inc_a, jain_i, t)


def kmeans_1d(values: np.ndarray, k: int, seed: int = 0, iters: int = 20) -> np.ndarray:
    """K-Means over driver incomes (Sec. V-A: "the driver population is grouped
    using the K-means algorithm ... by associating each sample with its nearest
    centroid and recalculating the centroids").

    Returns cluster labels ordered by ascending centroid, so label 0 is always the
    lowest-income group.
    """
    n = values.size
    k = int(np.clip(k, 1, n))
    if k == 1 or n == 0:
        return np.zeros(n, dtype=int)

    # Quantile initialisation: deterministic and well spread for 1-D data.
    centroids = np.quantile(values, np.linspace(0, 1, k))
    for _ in range(iters):
        labels = np.argmin(np.abs(values[:, None] - centroids[None, :]), axis=1)
        moved = False
        for j in range(k):
            sel = labels == j
            if sel.any():
                new = values[sel].mean()
                moved |= not np.isclose(new, centroids[j])
                centroids[j] = new
        if not moved:
            break
    order = np.argsort(centroids)
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)
    return rank[np.argmin(np.abs(values[:, None] - centroids[None, :]), axis=1)]


class VFDCFMVD(Algorithm):
    name = "VFDCFMVD"

    def __init__(
        self,
        n_zones: int,
        cfg: AlgorithmConfig = CONFIG.algorithm,
        use_clustering: bool = True,
        use_value_matching: bool = True,
        use_dispatching: bool = True,
        low_income_threshold: float | None = None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.n_zones = n_zones
        self.use_clustering = use_clustering
        self.use_value_matching = use_value_matching
        self.use_dispatching = use_dispatching
        self.threshold = cfg.low_income_threshold if low_income_threshold is None else low_income_threshold
        self.seed = seed
        self.value = StateValueFunction(n_zones, cfg.value_lr, cfg.value_gamma)
        self.agent = DQNAgent(
            N_STATE, cfg.cluster_choices, cfg.dqn_lr, cfg.dqn_gamma, cfg.dqn_memory,
            cfg.dqn_batch, cfg.dqn_target_update, cfg.dqn_hidden,
            cfg.dqn_epsilon_start, cfg.dqn_epsilon_end, cfg.dqn_epsilon_decay, seed,
        )
        self.training = False
        self._pre_zone: np.ndarray | None = None

    def reset(self) -> None:
        self.value = StateValueFunction(self.n_zones, self.cfg.value_lr, self.cfg.value_gamma)
        self._pre_zone = None

    # ------------------------------------------------------------------- state
    def observe_state(self, platform, batch) -> np.ndarray:
        """`s = (ser_r, idle_r, inc_a, jain_i, t)` of Sec. V-A."""
        cfg = platform.cfg
        n_orders = max(len(batch), 1)
        income = platform.fleet.income_upto(batch.slot)
        served = platform.fleet.n_served > 0
        return np.array(
            [
                platform.n_matched / max(platform.n_matched + n_orders, 1),  # ser_r
                float(np.mean(~served)),                                     # idle_r
                float(income.mean()),                                        # inc_a
                jain_index(np.clip(income, 0.0, None)),                      # jain_i
                batch.slot / cfg.n_slots,                                    # t
            ],
            dtype=np.float64,
        )

    # ---------------------------------------------------------------- matching
    def _edge_weights(self, platform, batch, drivers):
        """Eq. 14 weights and the admissibility mask for a driver subset.

        Computed once per time slot over *all* available drivers; the per-cluster
        bipartite graphs of Algorithm 2 are then slices of these matrices, since the
        clusters partition the driver set.
        """
        pickup_km = platform.pickup_distance(batch, drivers)
        trip_km = platform.trip_distance(batch)[:, None]
        profit = batch.fare[:, None] - (pickup_km + trip_km) * platform.fleet.cost[drivers][None, :]
        duration = platform.travel_slots(pickup_km + trip_km)

        if self.use_value_matching:
            from_zone = np.broadcast_to(platform.fleet.zone[drivers][None, :], profit.shape)
            to_zone = np.broadcast_to(batch.dropoff[:, None], profit.shape)
            weights = self.value.advantage(profit, duration, from_zone, to_zone)
            # "if dV > 0 then assign the weights of edges ... and insert them into
            # the bipartite graph" (Alg. 2, lines 11-12).
            admissible = weights > 0
        else:
            # Ablation "w/o driver-order matching": greedy on immediate profit.
            weights = profit
            admissible = profit > 0

        feasible = pickup_km <= self._max_pickup_km(platform, batch)
        return weights, admissible & feasible

    @staticmethod
    def _max_pickup_km(platform, batch) -> np.ndarray:
        """The waiting-time constraint, as a per-order pickup-distance cap."""
        slots_left = batch.deadline - batch.slot
        return (slots_left * platform.cfg.slot_seconds / 3600.0 * platform.cfg.v_avg_kmh)[:, None]

    def match(self, platform, batch):
        if len(batch) == 0:
            return EMPTY

        available = np.flatnonzero(platform.fleet.available(batch.slot))
        if available.size == 0:
            return EMPTY

        self._state = self.observe_state(platform, batch)
        income = platform.fleet.income_upto(batch.slot)[available]

        if self.use_clustering:
            action = self.agent.act(self._state)
            n_clusters = self.cfg.cluster_choices[action]
            self._action = action
        else:
            # Ablation "w/o dynamic clustering": one undifferentiated group.
            n_clusters = 1
            self._action = 0
        self._n_clusters = n_clusters

        # D_min, "the set of drivers with the lowest 30% of income in the current
        # time slot" (Sec. V-A). Only these receive the sequential, cluster-by-cluster
        # fairness priority; everyone else is matched in a single income-maximising
        # pass, which is what "emphasize total income maximization within
        # similar-income groups" asks for. Table IV sweeps the threshold: widening it
        # brings more drivers into the compensation range at the cost of letting
        # fairness intervention disturb more of the matching.
        k_low = int(round(self.threshold * available.size))
        low_income = np.zeros(available.size, dtype=bool)
        if k_low > 0:
            low_income[np.argpartition(income, min(k_low, available.size) - 1)[:k_low]] = True

        labels = np.full(available.size, -1, dtype=int)
        if low_income.any():
            labels[low_income] = kmeans_1d(income[low_income], n_clusters, self.seed)

        # One pass over the (orders x available drivers) geometry; the bipartite
        # graphs below are slices of it.
        weights, admissible = self._edge_weights(platform, batch, available)

        matcher = km_match if self.use_value_matching else greedy_match
        remaining = np.ones(len(batch), dtype=bool)
        all_orders: list[np.ndarray] = []
        all_drivers: list[np.ndarray] = []

        def run_pass(cols: np.ndarray) -> None:
            if cols.size == 0 or not remaining.any():
                return
            pos = np.flatnonzero(remaining)
            r, c = matcher(weights[np.ix_(pos, cols)], admissible[np.ix_(pos, cols)])
            if r.size == 0:
                return
            all_orders.append(pos[r])
            all_drivers.append(available[cols[c]])
            remaining[pos[r]] = False

        # "we choose the driver group that earns the least" first, then the next —
        # Algorithm 2 iterates the clusters, lowest income group prioritised.
        for label in range(n_clusters):
            run_pass(np.flatnonzero(labels == label))
        # Everyone above the threshold competes for what is left, in one pass.
        run_pass(np.flatnonzero(labels < 0))

        if not all_orders:
            self._pre_zone = None
            return EMPTY
        orders = np.concatenate(all_orders)
        drivers = np.concatenate(all_drivers)
        # V(s) is keyed on the driver's zone *before* the trip; the fleet moves
        # during `serve`, so capture it here.
        self._pre_zone = platform.fleet.zone[drivers].copy()
        return orders, drivers

    # -------------------------------------------------------------- dispatching
    def dispatch(self, platform, batch) -> None:
        """Send still-idle vehicles to the best nearby zone (Sec. V-C, Alg. 2 18-27)."""
        if not self.use_dispatching:
            return
        idle = np.flatnonzero(platform.fleet.dispatchable(batch.slot))
        if idle.size == 0:
            return

        zones = platform.fleet.zone[idle]
        near = np.stack([platform.graph.nearby(z, self.cfg.n_nearby_zones) for z in zones])
        km = platform.graph.matrix[zones[:, None], near]
        duration = platform.travel_slots(km)

        # A virtual order with p_o = 0: the reward is minus the repositioning cost.
        profit = -km * platform.fleet.cost[idle][:, None]
        delta = self.value.reposition_advantage(profit, duration, zones[:, None], near)

        best = np.argmax(delta, axis=1)
        rows = np.arange(idle.size)
        # "Find the largest state difference and the largest nearby zone g where
        # dV' > 0" — a driver with no improving zone stays put.
        move = delta[rows, best] > 0
        if not move.any():
            return
        platform.reposition(idle[move], near[rows[move], best[move]], batch.slot)

    # -------------------------------------------------------------------- update
    def update(self, platform, batch, matched_orders, matched_drivers) -> None:
        """Update V(s) (Eq. 13) and, when training, the clustering DQN."""
        if len(matched_orders) and self._pre_zone is not None:
            drivers = matched_drivers
            pickup = batch.pickup[matched_orders]
            dropoff = batch.dropoff[matched_orders]
            km = platform.graph.dis(self._pre_zone, pickup) + platform.graph.dis(pickup, dropoff)
            profit = batch.fare[matched_orders] - km * platform.fleet.cost[drivers]
            self.value.update(profit, platform.travel_slots(km), self._pre_zone, dropoff)

        # Sec. V-B: "The platform collects the state transition information for the
        # ongoing time slot depending on whether the driver has been matched or not
        # and subsequently updates the state value function V(s) using (13)." A
        # driver that stayed idle contributes r = 0 with s' = s, which decays the
        # value of the zone it is waiting in. Without this half of the transitions
        # are missing, V drifts upward, and the dV > 0 rule of Sec. V-C then prunes
        # most of the bipartite graph.
        still_idle = np.flatnonzero(platform.fleet.available(batch.slot))
        if still_idle.size:
            self.value.update_idle(platform.fleet.zone[still_idle])

        if self.training and self.use_clustering:
            reward = self._reward(platform, batch)
            state2 = self.observe_state(platform, batch)
            done = batch.slot == platform.cfg.n_slots - 1
            self.agent.remember(self._state, self._action, reward, state2, done)
            self.agent.learn()

    def _reward(self, platform, batch) -> float:
        """Eq. 8: `r = sum_{d in D_min} u_d^t + sum_{d in D'} u_d^t`.

        `D_min` is "the set of drivers with the lowest 30% of income in the current
        time slot", so the poorest drivers' income is counted twice — the
        maximum-minimum fairness weighting of Sec. V-A.
        """
        income = platform.fleet.income_per_slot[:, batch.slot]
        cumulative = platform.fleet.income_upto(batch.slot)
        k = max(int(round(self.threshold * cumulative.size)), 1)
        poorest = np.argpartition(cumulative, k - 1)[:k]
        return float(income[poorest].sum() + income.sum())
