"""Tests for the outcome-free continuous practical-value gate."""

from __future__ import annotations

import copy

import pytest

from experiments.continuous_practical_value_gate_v1 import run


def _step2(**overrides):
    record = {
        "status": "passed",
        "method_id": "candidate-a",
        "method_config_frozen_before_outcomes": True,
        "outcomes_used_for_tuning": False,
        "locked_method_config_sha256": "a" * 64,
        "formulation_changes_after_mechanism": 0,
    }
    record.update(overrides)
    return record


def _reference(**overrides):
    record = {
        "selector": "linear_cv_training_only",
        "best_reference_r2_point": 0.30,
        "best_reference_r2_lower95": 0.18,
        "nonlinear_gain_lower95_r2": 0.03,
        "training_only_linear_cv_top1_regret_lower95_r2": 0.02,
        "training_only_linear_cv_material_pair_reversal_count": 2,
        "top3_boundary_identified": True,
    }
    record.update(overrides)
    return record


def _plan(**overrides):
    record = {
        "dataset_groups": [
            {"group_id": "new_nonlinear_a", "source_id": "source-nonlinear-a", "is_new": True, "role": "nonlinear_opportunity"},
            {"group_id": "new_nonlinear_b", "source_id": "source-nonlinear-b", "is_new": True, "role": "nonlinear_opportunity"},
            {"group_id": "new_linear_a", "source_id": "source-linear-a", "is_new": True, "role": "linear_control"},
            {"group_id": "new_linear_b", "source_id": "source-linear-b", "is_new": True, "role": "linear_control"},
        ],
        "scout_backbones": ["scout-a", "scout-b"],
        "reserved_backbones": ["reserved-a", "reserved-b"],
        "embedding_dimensions": [384, 768, 1536],
        "native_dimensions": {"scout-a": 384, "scout-b": 768, "reserved-a": 1536, "reserved-b": 384},
        "sample_sizes": [300, 600, 1200],
        "independent_evaluation": True,
        "method_parameters_locked": True,
        "outcome_free": True,
        "runtime": {
            "includes": ["loaded_embeddings", "selector"],
            "excludes": ["embedding_extraction"],
            "warmup_runs_excluded": 1,
            "timed_repeats": 5,
            "thread_control": "one-thread",
            "complete_grid": True,
        },
        "scorer_runtime_includes_all": True,
    }
    record.update(overrides)
    return record


def _outcomes(**overrides):
    record = {
        "linear_control": {
            "upper95_excess_top1_regret_r2": 0.004,
            "upper95_excess_top3_opportunity_regret_r2": 0.004,
            "upper95_material_pair_accuracy_loss": 0.01,
        },
        "nonlinear_opportunity": {
            "upper95_method_minus_linear_top1_regret_r2": -0.011,
            "upper95_method_minus_linear_top3_opportunity_regret_r2": 0.004,
        },
        "broad_panel": {
            "upper95_excess_top1_regret_r2": 0.004,
            "upper95_excess_top3_opportunity_regret_r2": 0.004,
        },
        "runtime": {
            "geometric_mean_ratio": 0.8,
            "p90_ratio": 0.95,
            "complete_grid": True,
            "peak_memory_recorded": True,
        },
    }
    record.update(overrides)
    return record


def test_protocol_is_frozen_and_hash_sidecar_matches():
    protocol = run.load_protocol()
    assert protocol["scope"]["outcome_comparison_executed"] is False
    assert protocol["estimand"]["primary_metric"] == "top1_selection_regret_r2"
    assert protocol["capacity_grid"]["sample_sizes"] == [300, 600, 1200]
    assert protocol["capacity_grid"]["embedding_dimensions"] == [384, 768, 1536]
    assert protocol["capacity_grid"]["representative_synthetic_dimensions"] == [128, 512, 1024]
    assert run.sha256_path(run.PROTOCOL_PATH) == run.PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()


def test_admission_requires_step2_survival_fresh_groups_reserved_backbones_and_headroom():
    decision = run.admission_decision(
        "candidate-a", _step2(), _reference(), _plan()
    )
    assert decision["admitted"] is True
    assert decision["next_action"] == "independent_evaluation"
    assert decision["checks"]["reference"]["lp_headroom"] is True

    no_step2 = run.admission_decision(
        "candidate-a", _step2(status="failed"), _reference(), _plan()
    )
    assert no_step2["admitted"] is False
    assert no_step2["next_action"] == "stop"

    no_reserved = copy.deepcopy(_plan(reserved_backbones=["scout-a", "scout-b"]))
    rejected = run.admission_decision("candidate-a", _step2(), _reference(), no_reserved)
    assert rejected["admitted"] is False
    assert "reserved backbones" in rejected["failure"] or "overlap" in rejected["failure"]


def test_low_lp_headroom_blocks_nonlinear_claim_even_with_large_head_gain():
    checks = run.validate_reference_headroom(
        _reference(training_only_linear_cv_top1_regret_lower95_r2=0.009, nonlinear_gain_lower95_r2=0.20),
        run.load_protocol(),
    )
    assert checks["nonlinear_gain"] is True
    assert checks["lp_headroom"] is False
    assert checks["nonlinear_opportunity"] is False


def test_more_than_one_post_mechanism_change_is_fail_closed():
    with pytest.raises(ValueError, match="formulation-change limit"):
        run.validate_step2_record(
            _step2(formulation_changes_after_mechanism=2),
            "candidate-a",
            run.load_protocol(),
        )


def test_single_post_mechanism_change_requires_predeclared_mapping_and_new_hash():
    record = _step2(
        formulation_changes_after_mechanism=1,
        diagnosis_to_change_mapping={
            "diagnosis_id": "predeclared_mechanism_diagnosis",
            "change_id": "one_predeclared_formulation_change",
            "declared_before_step2_outcomes": True,
        },
        post_change_outcomes_used_for_tuning=False,
        post_change_locked_method_config_sha256="b" * 64,
    )
    validated = run.validate_step2_record(record, "candidate-a", run.load_protocol())
    assert validated["formulation_changes_after_mechanism"] == 1

    record["diagnosis_to_change_mapping"]["declared_before_step2_outcomes"] = False
    with pytest.raises(ValueError, match="declared too late"):
        run.validate_step2_record(record, "candidate-a", run.load_protocol())


def test_heldout_linear_head_disagreement_cannot_supply_lp_headroom():
    reference = _reference(selector="heldout_linear_head")
    with pytest.raises(ValueError, match="training-only Linear CV"):
        run.validate_reference_headroom(reference, run.load_protocol())


def test_outcome_gate_keeps_top1_top3_pairs_and_runtime_separate():
    result = run.outcome_decision(_outcomes())
    assert result["passed"] is True
    assert result["checks"]["linear_control_top1"] is True
    assert result["checks"]["linear_control_top3"] is True
    assert result["checks"]["linear_control_pairs"] is True
    assert result["checks"]["nonlinear_top1"] is True
    assert result["preferred_speed"] is False

    failed = run.outcome_decision(
        _outcomes(
            nonlinear_opportunity={
                "upper95_method_minus_linear_top1_regret_r2": -0.009,
                "upper95_method_minus_linear_top3_opportunity_regret_r2": 0.004,
            }
        )
    )
    assert failed["passed"] is False
    assert failed["next_action"] == "stop"
    assert failed["checks"]["nonlinear_top1"] is False


def test_missing_uncertainty_or_runtime_fields_cannot_pass():
    missing = _outcomes()
    del missing["linear_control"]["upper95_excess_top3_opportunity_regret_r2"]
    result = run.outcome_decision(missing)
    assert result["passed"] is False
    assert result["next_action"] == "stop"
    assert "missing" in result["failure"]
