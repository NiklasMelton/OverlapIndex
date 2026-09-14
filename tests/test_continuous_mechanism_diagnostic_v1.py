"""Focused invariants for the synthetic continuous-OI mechanism diagnostic."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from experiments.continuous_mechanism_diagnostic_v1 import run


def test_protocol_closes_mechanisms_and_uses_prototype_free_broad_score():
    protocol = run.load_protocol()
    assert tuple(protocol["data"]["families"]) == run.FAMILY_IDS
    assert tuple(row["method_id"] for row in protocol["selector"]["methods"]) == run.METHOD_IDS
    broad = next(row for row in protocol["selector"]["methods"] if row["method_id"] == "broad_rff")
    assert broad["geometry"] == "rff_mean"
    assert "prototype-free" in broad["description"]
    assert protocol["selector"]["null_permutations"] == 32
    assert protocol["gates"]["frozen_before_outcomes"] is True


def test_analytic_truth_is_monotone_for_predictive_families_and_zero_for_controls():
    protocol = run.load_protocol()
    for family in run.PREDICTIVE_FAMILIES:
        panel = run.generate_panel(protocol, family, 1001)
        values = [float(candidate["truth"]["analytic_expected_r2"]) for candidate in panel["candidates"]]
        empirical = [float(candidate["truth"]["evaluation_conditional_mean_r2"]) for candidate in panel["candidates"]]
        assert values[0] > values[1] > values[2]
        assert np.max(np.abs(np.asarray(empirical) - np.asarray(values))) < 0.015
    for family in run.CONTROL_FAMILIES:
        panel = run.generate_panel(protocol, family, 1001)
        assert all(candidate["truth"]["analytic_expected_r2"] == 0.0 for candidate in panel["candidates"])
        assert all(abs(float(candidate["truth"]["evaluation_conditional_mean_r2"])) < 0.05 for candidate in panel["candidates"])


def test_matched_panel_reuses_targets_but_changes_representation_quality():
    protocol = run.load_protocol()
    panel = run.generate_panel(protocol, "quadratic_mean", 1001)
    assert np.array_equal(panel["y_train"], run.generate_panel(protocol, "quadratic_mean", 1001)["y_train"])
    assert np.array_equal(panel["y_evaluation"], run.generate_panel(protocol, "quadratic_mean", 1001)["y_evaluation"])
    assert not np.array_equal(panel["candidates"][0]["X_train"], panel["candidates"][1]["X_train"])
    assert [candidate["quality"] for candidate in panel["candidates"]] == [0.95, 0.65, 0.35]


def test_broad_score_is_deterministic_and_records_prototype_elision():
    protocol = run.load_protocol()
    panel = run.generate_panel(protocol, "linear_mean", 1001)
    candidate = panel["candidates"][0]
    folds = list(
        KFold(n_splits=3, shuffle=True, random_state=protocol["selector"]["fold_seed"])
        .split(np.arange(len(panel["y_train"])))
    )
    first = run._evaluate_method("broad_rff", candidate["X_train"], panel["y_train"], 101, protocol, folds)
    second = run._evaluate_method("broad_rff", candidate["X_train"], panel["y_train"], 101, protocol, folds)
    assert first["score"] == second["score"]
    assert all(row["prototype_fit_elided"] is True for row in first["folds"])
    assert all(row["kernel_alignment"]["feature_bandwidths"]["4.0"]["score"] is not None for row in first["folds"])


def test_raw_validator_fails_closed_on_incomplete_artifact():
    protocol = run.load_protocol()
    with pytest.raises(ValueError, match="incomplete"):
        run.validate_raw(protocol, {"artifact_status": "running", "rows": []})
    with pytest.raises(ValueError, match="row count"):
        run.validate_raw(protocol, {"artifact_status": "completed", "rows": []})
