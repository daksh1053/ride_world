"""Unit tests for the VFDCFMVD reproduction.

Run with:  python -m pytest tests/ -q     (or)     python tests/test_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from vfd.config import CONFIG, PlatformConfig
from vfd.graph import ZoneGraph
from vfd.methods import VFDCFMVD, ILP, LAF, SID, NearestMatching, WorstDriverFirst, kmeans_1d
from vfd.methods.base import _sparsify, greedy_match, km_match
from vfd.metrics import (
    RATIO_FLOOR,
    evaluate,
    jain_index,
    temporal_earnings_fairness,
    weighted_unit_time_income,
)
from vfd.simulator import OrderBatch, Platform
from vfd.value_function import StateValueFunction


def toy_graph(n: int = 8) -> ZoneGraph:
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 3.0, size=(n, n))
    np.fill_diagonal(w, 0.0)
    return ZoneGraph(w)


def toy_cfg(n_drivers: int = 6, n_slots: int = 30) -> PlatformConfig:
    return PlatformConfig(n_slots=n_slots, default_vehicles=n_drivers)


def toy_day(n_orders: int = 120, n_zones: int = 8, n_slots: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "slot": rng.integers(0, n_slots, n_orders),
            "pu_idx": rng.integers(0, n_zones, n_orders),
            "do_idx": rng.integers(0, n_zones, n_orders),
            "fare_amount": rng.uniform(8.0, 40.0, n_orders),
        }
    )


# ------------------------------------------------------------------------ graph
def test_shortest_path_obeys_triangle_inequality():
    g = toy_graph()
    d = g.matrix
    for i in range(g.n):
        for k in range(g.n):
            assert np.all(d[i, k] <= d[i, :] + d[:, k] + 1e-9)


def test_shortest_path_never_exceeds_direct_edge():
    g = toy_graph()
    assert np.all(g.matrix <= g.edge_weights + 1e-9)


def test_nearby_returns_closest_zones_in_order():
    g = toy_graph()
    near = g.nearby(0, 4)
    assert len(near) == 4
    assert 0 not in near
    d = g.matrix[0, near]
    assert np.all(np.diff(d) >= -1e-9)


def test_cached_path_endpoints_are_consistent():
    g = toy_graph()
    for a in range(g.n):
        for b in range(g.n):
            p = g.path(a, b)
            assert p[0] == a and p[-1] == b


# ---------------------------------------------------------------------- metrics
def test_weighted_unit_time_income_matches_definition_6():
    income = np.array([[3.0, 0.0, 6.0], [1.0, 1.0, 1.0]])
    active = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    xi = np.array([1.0, 2.0, 3.0])
    f = weighted_unit_time_income(income, active, xi)
    assert np.isclose(f[0], (3 / 1 + 0 / 2 + 6 / 3) / 2)
    assert np.isclose(f[1], (1 / 1 + 1 / 2 + 1 / 3) / 3)


def test_unfairness_is_zero_when_all_incomes_are_equal():
    """F = -sum log(F_d / max F_d') is 0 exactly when every driver matches the max."""
    assert np.isclose(temporal_earnings_fairness(np.full(10, 4.0)), 0.0)


def test_unfairness_grows_as_incomes_disperse():
    tight = np.array([9.0, 10.0, 11.0])
    loose = np.array([1.0, 10.0, 30.0])
    assert temporal_earnings_fairness(tight) < temporal_earnings_fairness(loose)


def test_unfairness_is_finite_for_a_driver_that_never_earned():
    f = np.array([0.0, 5.0, 10.0])
    value = temporal_earnings_fairness(f)
    assert np.isfinite(value)
    assert np.isclose(value, -np.log(RATIO_FLOOR) - np.log(0.5))


def test_jain_index_bounds():
    assert np.isclose(jain_index(np.full(7, 3.0)), 1.0)
    lopsided = jain_index(np.array([10.0, 0.0, 0.0, 0.0]))
    assert np.isclose(lopsided, 0.25)


def test_evaluate_reports_idle_and_service_rates():
    income = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 0.0]])
    active = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    ev = evaluate(income, active, np.ones(2), n_orders=10, n_matched=4, running_time=0.5)
    assert np.isclose(ev.total_income, 6.0)
    assert np.isclose(ev.idle_driver_rate, 1 / 3)
    assert np.isclose(ev.order_service_rate, 0.4)


# --------------------------------------------------------------- value function
def test_discounted_reward_matches_the_explicit_sum():
    v = StateValueFunction(4, gamma=0.9)
    for T in (1, 3, 12):
        r = 7.0
        expected = sum(0.9 ** t * r / T for t in range(T))
        assert np.isclose(v.discounted_reward(np.array([r]), np.array([T]))[0], expected)


def test_advantage_is_reward_plus_discounted_future_minus_present():
    v = StateValueFunction(4, gamma=0.9)
    v.value[:] = [1.0, 2.0, 3.0, 4.0]
    profit, dur = np.array([5.0]), np.array([2])
    adv = v.advantage(profit, dur, np.array([0]), np.array([3]))
    expected = v.discounted_reward(profit, dur)[0] + 0.9 ** 2 * 4.0 - 1.0
    assert np.isclose(adv[0], expected)


def test_value_update_moves_toward_the_td_target():
    v = StateValueFunction(4, alpha=0.5, gamma=0.9)
    before = v.value[0]
    delta = v.advantage(np.array([10.0]), np.array([1]), np.array([0]), np.array([1]))[0]
    v.update(np.array([10.0]), np.array([1]), np.array([0]), np.array([1]))
    assert np.isclose(v.value[0], before + 0.5 * delta)


def test_idle_update_decays_the_zone_value():
    v = StateValueFunction(4, alpha=0.5, gamma=0.9)
    v.value[2] = 10.0
    v.update_idle(np.array([2]))
    assert v.value[2] < 10.0
    assert np.isclose(v.value[2], 10.0 + 0.5 * (0.9 - 1.0) * 10.0)


def test_idle_update_leaves_untouched_zones_alone():
    v = StateValueFunction(4, alpha=0.5, gamma=0.9)
    v.value[:] = [1.0, 2.0, 3.0, 4.0]
    v.update_idle(np.array([1]))
    assert np.allclose(v.value[[0, 2, 3]], [1.0, 3.0, 4.0])


# --------------------------------------------------------------------- matching
def test_km_finds_the_maximum_weight_matching():
    w = np.array([[9.0, 1.0], [1.0, 8.0]])
    r, c = km_match(w, np.ones_like(w, dtype=bool))
    pairs = dict(zip(r.tolist(), c.tolist()))
    assert pairs == {0: 0, 1: 1}


def test_km_never_returns_a_forbidden_pair():
    rng = np.random.default_rng(5)
    w = rng.uniform(0, 10, size=(15, 6))
    allowed = rng.random((15, 6)) > 0.5
    r, c = km_match(w, allowed)
    assert all(allowed[i, j] for i, j in zip(r, c))


def test_km_matches_each_driver_and_order_at_most_once():
    rng = np.random.default_rng(6)
    w = rng.uniform(0, 10, size=(20, 7))
    r, c = km_match(w, np.ones_like(w, dtype=bool))
    assert len(set(r.tolist())) == len(r)
    assert len(set(c.tolist())) == len(c)


def test_sparsify_keeps_only_the_best_k_per_row():
    w = np.array([[1.0, 5.0, 3.0, 9.0]])
    allowed = np.ones_like(w, dtype=bool)
    kept = _sparsify(allowed, w, 2)
    assert kept.tolist() == [[False, True, False, True]]


def test_sparsify_never_admits_a_forbidden_pair():
    w = np.array([[1.0, 5.0, 3.0, 9.0]])
    allowed = np.array([[True, False, True, False]])
    kept = _sparsify(allowed, w, 2)
    assert not (kept & ~allowed).any()


def test_greedy_match_is_a_valid_matching():
    rng = np.random.default_rng(7)
    w = rng.uniform(0, 10, size=(12, 5))
    r, c = greedy_match(w, np.ones_like(w, dtype=bool))
    assert len(set(c.tolist())) == len(c)


# -------------------------------------------------------------------- k-means
def test_kmeans_labels_are_ordered_by_income():
    values = np.array([1.0, 2.0, 50.0, 51.0, 100.0])
    labels = kmeans_1d(values, 3)
    # Label 0 must be the poorest group, and labels never decrease with income.
    assert labels[0] == 0
    assert np.all(np.diff(labels[np.argsort(values)]) >= 0)


def test_kmeans_handles_degenerate_input():
    assert np.all(kmeans_1d(np.full(5, 3.0), 4) >= 0)
    assert kmeans_1d(np.array([]), 3).size == 0


# ------------------------------------------------------------------- simulator
def test_travel_slots_uses_the_average_speed():
    p = Platform(toy_graph(), toy_cfg())
    km = p.cfg.v_avg_kmh          # exactly one hour of travel
    assert p.travel_slots(np.array([km]))[0] == 60


def test_profit_matches_equation_2():
    g = toy_graph()
    p = Platform(g, toy_cfg())
    batch = OrderBatch(0, np.array([0]), np.array([1]), np.array([5]),
                       np.array([25.0]), np.array([0]), np.array([6]))
    drivers = np.arange(p.fleet.n)
    profit = p.profit_matrix(batch, drivers)
    for v in drivers:
        cost = (g.dis(p.fleet.zone[v], 1) + g.dis(1, 5)) * p.fleet.cost[v]
        assert np.isclose(profit[0, v], 25.0 - cost)


def test_waiting_time_constraint_is_enforced():
    g = toy_graph()
    p = Platform(g, toy_cfg())
    batch = OrderBatch(0, np.array([0]), np.array([1]), np.array([5]),
                       np.array([25.0]), np.array([0]), np.array([3]))
    reach = 3 * p.cfg.slot_seconds / 3600.0 * p.cfg.v_avg_kmh
    assert np.array_equal(
        p.feasible_matrix(batch, np.arange(p.fleet.n))[0],
        p.pickup_distance(batch, np.arange(p.fleet.n))[0] <= reach,
    )


def test_serving_moves_the_driver_and_books_the_profit():
    g = toy_graph()
    p = Platform(g, toy_cfg())
    batch = OrderBatch(0, np.array([0]), np.array([1]), np.array([5]),
                       np.array([25.0]), np.array([0]), np.array([6]))
    expected = p.profit_matrix(batch, np.array([2]))[0, 0]
    p.serve(batch, np.array([0]), np.array([2]))
    assert p.fleet.zone[2] == 5
    assert np.isclose(p.fleet.income_per_slot[2, 0], expected)
    assert p.fleet.busy_until[2] > 0
    assert p.n_matched == 1


def test_busy_driver_is_unavailable_until_the_trip_ends():
    g = toy_graph()
    p = Platform(g, toy_cfg())
    batch = OrderBatch(0, np.array([0]), np.array([1]), np.array([5]),
                       np.array([25.0]), np.array([0]), np.array([6]))
    p.serve(batch, np.array([0]), np.array([2]))
    end = int(p.fleet.busy_until[2])
    assert not p.fleet.available(end - 1)[2]
    assert p.fleet.available(end)[2]


def test_repositioning_driver_stays_matchable_but_not_redispatchable():
    g = toy_graph()
    p = Platform(g, toy_cfg())
    p.reposition(np.array([1]), np.array([4]), slot=0)
    assert p.fleet.zone[1] == 4
    assert p.fleet.available(1)[1]          # still matchable en route
    assert not p.fleet.dispatchable(1)[1]   # but not dispatched again
    assert p.fleet.income_per_slot[1, 0] < 0   # it paid the travel cost


def test_an_order_is_never_served_twice():
    g = toy_graph()
    cfg = toy_cfg(n_drivers=6)
    p = Platform(g, cfg)
    day = toy_day(n_slots=cfg.n_slots)
    waiting = np.full(len(day), 5)
    n = p.run_day(day, NearestMatching(), waiting)
    assert n == len(day)
    assert p.n_matched <= len(day)


# --------------------------------------------------------------------- methods
def _run(algorithm, n_drivers: int = 8):
    g = toy_graph()
    cfg = toy_cfg(n_drivers=n_drivers)
    p = Platform(g, cfg, n_drivers)
    day = toy_day(n_slots=cfg.n_slots)
    p.run_day(day, algorithm, np.full(len(day), 5))
    return p


def test_every_algorithm_produces_a_consistent_run():
    g = toy_graph()
    xi = np.ones(toy_cfg().n_slots)
    for algorithm in [
        VFDCFMVD(g.n), NearestMatching(), WorstDriverFirst(xi),
        LAF(g.n, xi), ILP(), SID(g.n),
    ]:
        p = _run(algorithm)
        assert p.n_matched == int(p.fleet.n_served.sum())
        assert p.n_matched <= 120


def test_ablation_switches_change_behaviour():
    g = toy_graph()
    full = _run(VFDCFMVD(g.n))
    no_dispatch = _run(VFDCFMVD(g.n, use_dispatching=False))
    # Without dispatching no driver ever pays a repositioning cost.
    assert (no_dispatch.fleet.repositioning_until == 0).all()
    assert (full.fleet.repositioning_until >= 0).all()


def test_clustering_switch_collapses_to_a_single_group():
    g = toy_graph()
    algorithm = VFDCFMVD(g.n, use_clustering=False)
    _run(algorithm)
    assert algorithm._n_clusters == 1


def test_value_matching_prunes_non_improving_pairs():
    """With value matching on, every accepted pair must have dV > 0."""
    g = toy_graph()
    cfg = toy_cfg(n_drivers=8)
    p = Platform(g, cfg, 8)
    algorithm = VFDCFMVD(g.n)
    algorithm.value.value[:] = np.linspace(0.0, 5.0, g.n)
    batch = OrderBatch(0, np.arange(4), np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]),
                       np.array([30.0, 30.0, 30.0, 30.0]), np.zeros(4, int), np.full(4, 8))
    weights, admissible = algorithm._edge_weights(p, batch, np.arange(8))
    assert np.all(weights[admissible] > 0)


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
