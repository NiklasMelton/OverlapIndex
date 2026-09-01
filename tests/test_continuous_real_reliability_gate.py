from __future__ import annotations

import copy

import numpy as np

from experiments.continuous_real_reliability_gate import run


def _panels():
    return run.parent.load_panels(run.parent._load_protocol())


def test_protocol_and_feature_schema_are_closed() -> None:
    protocol = run._load_protocol()
    assert tuple(protocol["feature_schema"]) == run.FEATURE_NAMES
    assert tuple(protocol["methods"]) == run.METHODS


def test_features_ignore_all_head_and_envelope_outcomes() -> None:
    panel = _panels()[0]
    baseline = run.feature_mapping(panel)
    changed = copy.deepcopy(panel)
    for candidate in changed["candidates"]:
        candidate["best_head_envelope_r2"] += 1000.0
        for head in candidate["heads"].values():
            head["test_r2"] -= 1000.0
    assert run.feature_mapping(changed) == baseline


def test_ridge_gate_is_deterministic_and_finite() -> None:
    X = np.asarray([[0.0, 0.2, 0.4, 0.6], [1.0, 0.7, 0.5, 0.3], [0.5, 0.4, 0.3, 0.2]])
    y = np.asarray([-0.1, 0.2, 0.05])
    first = run.fit_ridge_gate(X, y)
    second = run.fit_ridge_gate(X, y)
    assert first == second
    assert np.isfinite(run.predict_benefit(first, X[0]))


def test_held_out_outcome_mutation_cannot_change_held_out_model_or_features() -> None:
    panels = _panels()
    fixed = [
        run.parent.panel_metric(panel, method)
        for panel in panels
        for method in ("broad_rff", "bounded_2k", "pool_borda_a050", "linear_probe")
    ]
    baseline = run.fit_lodo_models(panels, fixed)
    changed = copy.deepcopy(fixed)
    for row in changed:
        if row["dataset_group"] == "oxford_pet":
            row["top_5_mean_utility_regret_r2"] += 1000.0
    updated = run.fit_lodo_models(panels, changed)
    assert updated["oxford_pet"] == baseline["oxford_pet"]


def test_summary_has_complete_gate_grid() -> None:
    protocol = run._load_protocol()
    summary = run.summarize(_panels(), protocol)
    assert len(summary["gate_predictions"]) == 15
    assert len(summary["panel_method_rows"]) == 90
    assert {row["method"] for row in summary["panel_method_rows"]} == set(run.METHODS)
