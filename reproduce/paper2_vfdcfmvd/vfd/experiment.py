"""Experiment driver: builds algorithms, trains them, and evaluates one day."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import CONFIG, Config
from .graph import ZoneGraph
from .methods import ILP, LAF, SID, VFDCFMVD, NearestMatching, WorstDriverFirst
from .metrics import Evaluation, evaluate
from .simulator import Platform


def sample_waiting_slots(n_orders: int, cfg: Config = CONFIG, seed: int = 0) -> np.ndarray:
    """t_o^w — "The maximum waiting time of passengers is chosen randomly from
    {3, 4, 5, 6, 7, 8 min}" (Table II), converted to time slots."""
    rng = np.random.default_rng(seed)
    minutes = rng.choice(np.array(cfg.platform.waiting_minutes), size=n_orders)
    return (minutes * 60 // cfg.platform.slot_seconds).astype(int)


def build_algorithms(graph: ZoneGraph, xi: np.ndarray, cfg: Config = CONFIG) -> dict:
    """The proposed algorithm and the five benchmarks of Sec. VI-B."""
    return {
        "VFDCFMVD": VFDCFMVD(graph.n, cfg.algorithm),
        "LAF": LAF(graph.n, xi, cfg.algorithm),
        "SID": SID(graph.n, cfg.algorithm),
        "ILP": ILP(),
        "WDF": WorstDriverFirst(xi),
        "NM": NearestMatching(),
    }


def build_ablations(graph: ZoneGraph, cfg: Config = CONFIG) -> dict:
    """Table III — the full algorithm and its three single-module removals."""
    def make(**kw):
        return VFDCFMVD(graph.n, cfg.algorithm, **kw)
    return {
        "VFDCFMVD": make(),
        "w/o dynamic clustering": make(use_clustering=False),
        "w/o driver-order matching": make(use_value_matching=False),
        "w/o idle vehicle dispatching": make(use_dispatching=False),
    }


def _needs_training(algorithm) -> bool:
    return hasattr(algorithm, "value")


def train(
    algorithm,
    graph: ZoneGraph,
    day: pd.DataFrame,
    n_drivers: int,
    cfg: Config = CONFIG,
    seed: int = 0,
) -> None:
    """Pre-train the clustering DQN (Algorithm 1).

    The two learners have different lifetimes and must not be conflated:

    * The **clustering policy** is what Algorithm 1 trains, over `Max` epochs, and
      returns as a strategy `pi`; Algorithm 2 then consumes it. So it is pre-trained
      here and frozen for the measured run.
    * The **state value function** is learned *online*: Algorithm 2 takes `V` as an
      input and line 30 updates it every time slot — "the platform collects the state
      transition information of the current time slot and uses the online TD(0)
      learning algorithm in each time slot to update the value function". It is
      therefore reset before evaluation.

    Carrying a converged `V` into the evaluation run instead drives the TD error to
    zero on average, and the `dV > 0` rule of Sec. V-C then prunes roughly every edge
    of the bipartite graph — the order service rate collapses from ~50% to ~13%.
    """
    if not _needs_training(algorithm):
        return

    if isinstance(algorithm, VFDCFMVD) and algorithm.use_clustering:
        algorithm.training = True
        for ep in range(cfg.algorithm.dqn_train_episodes):
            platform = Platform(graph, cfg.platform, n_drivers, seed + ep)
            platform.run_day(day, algorithm, sample_waiting_slots(len(day), cfg, seed + ep))
        algorithm.training = False
        algorithm.agent.training = False

    # V(s) starts fresh for the measured run and is learned online during it.
    algorithm.reset()


def evaluate_algorithm(
    algorithm,
    graph: ZoneGraph,
    day: pd.DataFrame,
    xi: np.ndarray,
    n_drivers: int,
    cfg: Config = CONFIG,
    seed: int = 0,
) -> Evaluation:
    """One measured day. `running_time` is the metric of Sec. VI-B(5)."""
    platform = Platform(graph, cfg.platform, n_drivers, seed)
    waiting = sample_waiting_slots(len(day), cfg, seed)

    start = time.perf_counter()
    n_orders = platform.run_day(day, algorithm, waiting)
    elapsed = time.perf_counter() - start

    return evaluate(
        platform.fleet.income_per_slot,
        platform.fleet.active_per_slot,
        xi,
        n_orders,
        platform.n_matched,
        elapsed,
    )


def run_sweep(
    graph: ZoneGraph,
    day: pd.DataFrame,
    xi: np.ndarray,
    vehicle_counts: tuple[int, ...],
    cfg: Config = CONFIG,
    seed: int = 0,
) -> dict:
    """Fig. 5 — every algorithm over the vehicle-count sweep."""
    results: dict[str, dict[int, dict]] = {}
    for n_drivers in vehicle_counts:
        for name, algorithm in build_algorithms(graph, xi, cfg).items():
            train(algorithm, graph, day, n_drivers, cfg, seed)
            ev = evaluate_algorithm(algorithm, graph, day, xi, n_drivers, cfg, seed)
            results.setdefault(name, {})[n_drivers] = ev.as_dict()
            print(
                f"  |D|={n_drivers:5d}  {name:10s}  unfair={ev.unfairness:9.1f}"
                f"  income={ev.total_income:11.1f}  service={ev.order_service_rate:6.2%}"
                f"  idle={ev.idle_driver_rate:6.2%}  time={ev.running_time:6.2f}s"
            )
    return results
