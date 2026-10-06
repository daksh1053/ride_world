"""Unit tests for the pieces the reproduction's conclusions rest on.

Run with:  python -m pytest tests/ -q     (or)     python tests/test_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ltf.config import CONFIG, SimConfig
from ltf.graph import RoadGraph
from ltf.methods import Greedy, LAF, MOMAQL, BalanceRidePooling, Reassign
from ltf.methods.base import hungarian_round, scaled_fairness_penalty, variance_delta_matrix
from ltf.metrics import evaluate, fairness, normalised_fairness, total_utility
from ltf.simulator import Batch, RideHailingEnv


def toy_graph(n: int = 6) -> RoadGraph:
    rng = np.random.default_rng(0)
    w = rng.uniform(1.0, 4.0, size=(n, n))
    np.fill_diagonal(w, 0.0)
    return RoadGraph(w)


def toy_day(n_requests: int = 40, n_nodes: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    t0 = pd.Timestamp("2016-03-26 18:00")
    return pd.DataFrame(
        {
            "pickup_time": t0 + pd.to_timedelta(np.sort(rng.uniform(0, 60, n_requests)), unit="m"),
            "s": rng.integers(0, n_nodes, n_requests).astype(np.int16),
            "d": rng.integers(0, n_nodes, n_requests).astype(np.int16),
        }
    )


# ------------------------------------------------------------------------ graph
def test_shortest_path_obeys_triangle_inequality():
    g = toy_graph()
    d = g.matrix
    assert np.all(d <= d[:, :, None].transpose(0, 2, 1) + d[None, :, :] + 1e-9), \
        "Geo(a,c) must never exceed Geo(a,b) + Geo(b,c)"


def test_shortest_path_never_exceeds_direct_edge():
    g = toy_graph()
    assert np.all(g.matrix <= g.edge_weights + 1e-9)


# ---------------------------------------------------------------------- metrics
def test_metrics_match_definitions():
    u = np.array([1.0, 3.0, 5.0, 7.0])
    assert total_utility(u) == 16.0
    assert np.isclose(fairness(u), np.var(u))
    assert np.isclose(normalised_fairness(u), np.std(u) / np.mean(u))
    ev = evaluate(u, lam=1.0)
    assert np.isclose(ev.objective, total_utility(u) - fairness(u))


def test_variance_delta_matches_recomputed_variance():
    """The closed-form delta must equal an explicit recomputation."""
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 50, size=8)
    gains = rng.uniform(-3, 9, size=(5, 8))
    delta = variance_delta_matrix(u, gains)
    for i in range(gains.shape[0]):
        for v in range(u.size):
            after = u.copy()
            after[v] += gains[i, v]
            assert np.isclose(delta[i, v], np.var(after) - np.var(u))


def test_fairness_penalty_is_zero_for_an_equal_fleet():
    """Regression test: the naive variance-normalised penalty diverges here."""
    u = np.full(20, 137.0)
    gains = np.full((3, 20), 5.0)
    pen = scaled_fairness_penalty(u, gains, lam=1.0, omega=0.6)
    assert np.all(np.isfinite(pen))
    assert np.allclose(pen, 0.0)


def test_fairness_penalty_favours_the_poorest_driver():
    u = np.array([0.0, 10.0, 20.0, 30.0])
    gains = np.ones((1, 4)) * 4.0
    pen = scaled_fairness_penalty(u, gains, lam=1.0, omega=0.6)[0]
    assert pen[0] < pen[1] < pen[2] < pen[3]
    assert pen[0] < 0 < pen[3]     # rewarding the poorest, penalising the richest


def test_fairness_penalty_does_not_grow_with_the_horizon():
    """omega's stated purpose: the weight must not drift as utilities accumulate."""
    base = np.array([0.0, 1.0, 2.0, 3.0])
    gains = np.ones((1, 4)) * 2.0
    early = scaled_fairness_penalty(base, gains, 1.0, 0.6)
    late = scaled_fairness_penalty(base * 1000, gains, 1.0, 0.6)
    assert np.allclose(early, late)


# --------------------------------------------------------------------- matching
def test_hungarian_maximises_score():
    score = np.array([[5.0, 1.0], [1.0, 6.0], [0.0, 0.0]])
    d = hungarian_round(score, np.ones(2, dtype=bool))
    assert d[0] == 0 and d[1] == 1


def test_hungarian_never_returns_a_forbidden_pair():
    rng = np.random.default_rng(3)
    score = rng.uniform(-5, 5, size=(12, 4))
    accept = rng.random((12, 4)) > 0.6
    d = hungarian_round(score, np.ones(4, dtype=bool), accept=accept)
    for i, v in enumerate(d):
        if v >= 0:
            assert accept[i, v]


def test_hungarian_assigns_each_driver_at_most_once():
    rng = np.random.default_rng(4)
    score = rng.uniform(0, 5, size=(20, 5))
    d = hungarian_round(score, np.ones(5, dtype=bool))
    taken = [v for v in d if v >= 0]
    assert len(taken) == len(set(taken))


# -------------------------------------------------------------------- simulator
def test_utility_matrix_matches_the_paper_definition():
    g = toy_graph()
    env = RideHailingEnv(g, SimConfig(n_drivers=3, max_pickup_distance=0.0))
    batch = Batch(pd.Timestamp("2016-03-26 18:00"), np.array([0, 2]), np.array([3, 4]))
    u = env.utility_matrix(batch)
    for i, (s, d) in enumerate(zip(batch.origin, batch.dest)):
        for v in range(3):
            expected = g.geo(s, d) - g.geo(env.fleet.location[v], s)
            assert np.isclose(u[i, v], expected)


def test_booked_utility_equals_what_the_matcher_saw():
    g = toy_graph()
    env = RideHailingEnv(g, SimConfig(n_drivers=3, max_pickup_distance=0.0))
    batch = Batch(pd.Timestamp("2016-03-26 18:00"), np.array([0, 2, 4]), np.array([3, 4, 1]))
    u = env.utility_matrix(batch)
    decision = np.array([0, 1, 2])
    gained = env.apply(batch, decision)
    for i, v in enumerate(decision):
        assert np.isclose(gained[v], u[i, v])


def test_pickup_radius_excludes_distant_drivers():
    g = toy_graph()
    env = RideHailingEnv(g, SimConfig(n_drivers=4, max_pickup_distance=1.0))
    batch = Batch(pd.Timestamp("2016-03-26 18:00"), np.array([0, 1]), np.array([2, 3]))
    feasible = env.feasible_matrix(batch)
    assert np.array_equal(feasible, env.pickup_matrix(batch) <= 1.0)


def test_driver_moves_to_the_request_destination():
    g = toy_graph()
    env = RideHailingEnv(g, SimConfig(n_drivers=2, max_pickup_distance=0.0))
    batch = Batch(pd.Timestamp("2016-03-26 18:00"), np.array([0]), np.array([5]))
    env.apply(batch, np.array([1]))
    assert env.fleet.location[1] == 5


def test_capacity_caps_assignments_per_batch():
    g = toy_graph()
    cfg = SimConfig(n_drivers=2, capacity=2, max_pickup_distance=0.0, batch_minutes=1440)
    env = RideHailingEnv(g, cfg)
    day = toy_day(50)
    stats = env.run_day(day, Greedy(lam=0.0))
    assert stats["n_assigned"] <= cfg.n_drivers * cfg.capacity


# ---------------------------------------------------------------------- methods
def _run(method, cfg=None):
    g = toy_graph()
    cfg = cfg or SimConfig(n_drivers=4, max_pickup_distance=0.0)
    env = RideHailingEnv(g, cfg)
    stats = env.run_day(toy_day(60), method, learn=True)
    return env, stats


def test_every_method_produces_a_valid_assignment():
    g = toy_graph()
    for method in [
        Greedy(lam=1.0),
        Reassign(iters=20),
        LAF(),
        BalanceRidePooling(g.n),
        MOMAQL(4, g.n),
    ]:
        env, stats = _run(method)
        assert stats["n_assigned"] <= stats["n_requests"]
        assert env.fleet.n_served.sum() == stats["n_assigned"]


def test_fairness_weight_reduces_earnings_spread():
    """A larger lambda must not make the fleet *less* equal."""
    g = toy_graph()
    cfg = SimConfig(n_drivers=6, max_pickup_distance=0.0)
    spreads = []
    for lam in (0.0, 4.0):
        env = RideHailingEnv(g, cfg)
        env.run_day(toy_day(300), BalanceRidePooling(g.n, lam=lam), learn=True)
        spreads.append(normalised_fairness(env.fleet.utility))
    assert spreads[1] <= spreads[0] + 1e-9


def test_momaql_fairness_ablation_changes_behaviour():
    g = toy_graph()
    cfg = SimConfig(n_drivers=6, max_pickup_distance=0.0)
    out = {}
    for use_fairness in (True, False):
        env = RideHailingEnv(g, cfg)
        m = MOMAQL(6, g.n, use_fairness=use_fairness)
        m.training = True
        env.run_day(toy_day(300), m, learn=True)
        out[use_fairness] = normalised_fairness(env.fleet.utility)
    assert out[True] <= out[False] + 1e-9


def test_momaql_scalarisation_is_utility_minus_fairness():
    m = MOMAQL(3, 5)
    u = np.array([[2.0, 3.0]])
    f = np.array([[0.5, 1.0]])
    assert np.allclose(m.scalarise(u, f), u - f)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print("\nall passed" if not failures else f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
