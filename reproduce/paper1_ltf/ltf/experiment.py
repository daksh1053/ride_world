"""Experiment driver: trains each method, evaluates over the 7-day test horizon,
and produces the horizon sweep used by Fig. 4 and Fig. 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, Config
from .forecasting import MLPRequestPredictor
from .graph import RoadGraph
from .methods import LAF, MOMAQL, BalanceRidePooling, Greedy, Reassign
from .metrics import Evaluation, evaluate
from .simulator import RideHailingEnv


def demand_pmf(train_days: list[pd.DataFrame], n_nodes: int) -> np.ndarray:
    """Pickup-demand distribution over L, used to seed initial driver positions."""
    counts = np.zeros(n_nodes, dtype=np.float64)
    for day in train_days:
        np.add.at(counts, day["s"].to_numpy().astype(int), 1.0)
    if counts.sum() <= 0:
        return np.full(n_nodes, 1.0 / n_nodes)
    return counts / counts.sum()


def build_methods(graph: RoadGraph, cfg: Config = CONFIG) -> dict:
    """All five methods of Table 1, under identical settings (Sec. 5.3)."""
    m = cfg.method
    return {
        "Greedy": Greedy(lam=m.lam),
        "REASSIGN": Reassign(slack=m.reassign_utility_slack, iters=m.reassign_iters),
        "LAF": LAF(lam=m.lam, alpha=m.alpha, gamma=m.gamma),
        "Balance Ride-Pooling": BalanceRidePooling(
            graph.n, lam=m.lam, omega=m.omega, gamma=m.gamma, alpha=m.alpha
        ),
        "Proposed Method": MOMAQL(
            cfg.sim.n_drivers, graph.n, lam=m.lam, omega=m.omega,
            gamma=m.gamma, alpha=m.alpha, epsilon=m.epsilon,
        ),
    }


def build_ablations(graph: RoadGraph, cfg: Config = CONFIG) -> dict:
    """Table 2 / Fig. 5 variants of the proposed method."""
    m = cfg.method
    def make(**kw):
        return MOMAQL(
            cfg.sim.n_drivers, graph.n, lam=m.lam, omega=m.omega,
            gamma=m.gamma, alpha=m.alpha, epsilon=m.epsilon, **kw
        )
    return {
        "Our Method": make(),
        "Our Method w/o Prediction": make(use_prediction=False),
        "Our Method w/o Fairness": make(use_fairness=False),
    }


def train(
    method,
    graph: RoadGraph,
    train_days: list[pd.DataFrame],
    predicted_days: list[pd.DataFrame] | None,
    cfg: Config = CONFIG,
    location_pmf: np.ndarray | None = None,
) -> None:
    """Warm up a learning method on the training days.

    Non-learning methods (Greedy, REASSIGN) simply ignore this.

    For MOMAQL the training stream is "historical requests + predicted requests"
    (Fig. 3): each epoch replays the real training days and then the forecast days,
    both through the same assignment path, so the predicted requests are part of the
    action space rather than a separate offline correction.
    """
    if not hasattr(method, "q_util") and not hasattr(method, "value"):
        return

    env = RideHailingEnv(graph, cfg.sim, location_pmf)
    is_momaql = isinstance(method, MOMAQL)
    if is_momaql:
        method.training = True

    stream = list(train_days)
    if is_momaql and method.wants_prediction and predicted_days:
        stream = stream + list(predicted_days)

    for _ in range(cfg.method.train_epochs):
        env.reset()
        for day in stream:
            env.run_day(day, method, learn=True)

    if is_momaql:
        method.training = False


def evaluate_method(
    method,
    graph: RoadGraph,
    test_days: list[pd.DataFrame],
    cfg: Config = CONFIG,
    location_pmf: np.ndarray | None = None,
) -> tuple[Evaluation, list[Evaluation]]:
    """Run the full test horizon; also return the cumulative result after each day.

    The per-day cumulative results give the "length of time horizon" sweep of
    Fig. 4 and Fig. 5, where the horizon is increased by a number of days.
    """
    env = RideHailingEnv(graph, cfg.sim, location_pmf)
    env.reset()
    per_horizon: list[Evaluation] = []
    n_req = n_asg = 0
    for day in test_days:
        stats = env.run_day(day, method, learn=False)
        n_req += stats["n_requests"]
        n_asg += stats["n_assigned"]
        per_horizon.append(evaluate(env.fleet.utility, cfg.method.lam, n_asg, n_req))
    return per_horizon[-1], per_horizon


def run_all(
    graph: RoadGraph,
    train_days: list[pd.DataFrame],
    test_days: list[pd.DataFrame],
    methods: dict,
    predicted_days: list[pd.DataFrame] | None = None,
    cfg: Config = CONFIG,
) -> dict:
    """Train + evaluate every method, returning final and per-horizon results."""
    results: dict[str, dict] = {}
    pmf = demand_pmf(train_days, graph.n)
    for name, method in methods.items():
        print(f"[exp] {name}: training")
        train(method, graph, train_days, predicted_days, cfg, pmf)
        print(f"[exp] {name}: evaluating")
        final, horizon = evaluate_method(method, graph, test_days, cfg, pmf)
        results[name] = {
            "final": final.as_dict(),
            "horizon": [h.as_dict() for h in horizon],
        }
        print(
            f"       utility={final.total_utility:12.2f}  fairness={final.fairness:14.2f}"
            f"  norm={final.normalised_fairness:7.4f}"
            f"  min/mean/max={final.min_utility:9.2f}/{final.mean_utility:9.2f}/{final.max_utility:9.2f}"
        )
    return results
