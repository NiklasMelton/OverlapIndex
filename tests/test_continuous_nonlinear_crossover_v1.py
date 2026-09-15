"""Focused tests for the continuous nonlinear crossover."""

from __future__ import annotations

from experiments.continuous_nonlinear_crossover_v1 import run


def test_protocol_closes_grid_and_counts():
    protocol = run.load_protocol()
    assert len(run.panel_keys(protocol)) == 24
    assert tuple(protocol["ranking_methods"]) == run.METHODS
    assert protocol["execution"]["expected_selector_rows"] == 1056
    assert protocol["execution"]["expected_reference_rows"] == 192


def test_scenario_reverses_analytic_winner_with_exact_recovery():
    protocol = run.load_protocol()
    for scenario, expected in (("linear_wins", "axis_candidate"), ("nonlinear_wins", "signed_candidate")):
        panel = run.generate_panel(protocol, scenario, 16, 300, 1103)
        truth = {row["candidate_id"]: row["analytic_r2"] for row in panel["candidates"]}
        assert max(truth, key=truth.get) == expected
        assert truth[expected] - max(value for name, value in truth.items() if name != expected) > 0.2
        assert all(row["X_train"].shape == (300, 16) for row in panel["candidates"])


def test_generation_is_deterministic():
    protocol = run.load_protocol()
    first = run.generate_panel(protocol, "nonlinear_wins", 512, 300, 1301)
    second = run.generate_panel(protocol, "nonlinear_wins", 512, 300, 1301)
    for left, right in zip(first["candidates"], second["candidates"]):
        assert (left["X_train"] == right["X_train"]).all()

