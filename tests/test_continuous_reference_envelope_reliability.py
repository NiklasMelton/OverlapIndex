"""Focused checks for the frozen reference-envelope reliability audit."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.continuous_reference_envelope_reliability import run


def test_protocol_sidecar_and_frozen_grid():
    protocol = run.load_protocol()

    assert run.sha256_path(run.PROTOCOL_PATH) == run.PROTOCOL_SHA256_PATH.read_text(
        encoding="utf-8"
    ).strip()
    assert protocol["scope"]["method_promotion_allowed"] is False
    assert protocol["backbones"]["reserved_evaluated"] == []
    assert protocol["expected_grid"]["archive_panel_count"] == 60
    assert protocol["expected_grid"]["budget_new_sklearn_estimator_fit_count"] == 18880
    assert protocol["evaluation_bootstrap"]["archive_100_status"] == (
        "unavailable_archived_predictions_not_stored"
    )
    assert protocol["budget_curve"]["selection_status"].startswith("source-selected")


def test_source_chain_and_archive_audit_are_complete_without_refits():
    protocol = run.load_protocol()
    raw, summary, manifest, cohort = run.validate_sources(protocol)
    archive = run.archive_audit(raw, protocol)

    assert summary["status"] == manifest["status"] == raw["status"]
    assert len(cohort["datasets"]) == 5
    assert archive["panel_count"] == 60
    assert archive["candidate_count"] == 300
    assert archive["evaluation_predictions_available"] is False
    assert len(archive["target_level_seed_stability"]) == 20
    assert len(archive["target_weighted_group_summaries"]) == 4
    assert set(archive["head_confusion"]) == set(run.ARCHIVE_GROUPS)
    assert archive["head_confusion"]["all"]["total"] == 300


def test_archive_cluster_intervals_use_target_and_dataset_units():
    protocol = run.load_protocol()
    raw, _summary, _manifest, _cohort = run.validate_sources(protocol)
    archive = run.archive_audit(raw, protocol)
    intervals = run._archive_cluster_intervals(archive, protocol)

    assert intervals["all"]["target_count"] == 20
    assert intervals["all"]["dataset_count"] == 5
    assert intervals["all"]["bootstrap_unit"].startswith("target block")
    assert set(intervals["all"]["target_block_intervals"]) >= {
        "spearman",
        "top1_regret_r2",
    }
    assert set(intervals["all"]["dataset_cluster_intervals"]) >= {
        "spearman",
        "top1_regret_r2",
    }


def test_budget_row_order_and_sizes_are_frozen():
    protocol = run.load_protocol()
    _raw, _summary, _manifest, cohort = run.validate_sources(protocol)

    for budget in protocol["budget_curve"]["budgets"]:
        target = protocol["budget_curve"]["targets"][0]
        train, evaluation, y_train, y_eval, train_ids, eval_ids = run._budget_rows(
            cohort, target, budget["training_replicate_seeds"]
        )
        assert len(train) == budget["rows"]
        assert len(evaluation) == 200
        assert y_train.shape == (budget["rows"],)
        assert y_eval.shape == (200,)
        assert len(train_ids) == len(set(train_ids))
        assert len(eval_ids) == len(set(eval_ids))
        assert not set(train_ids).intersection(eval_ids)
        target_rows = cohort["datasets"][target.split("::", 1)[0]]["target_cohorts"][target.split("::", 1)[1]]
        expected_ids = [
            str(row["sample_id"])
            for seed in run.REPLICATE_SEEDS
            if seed in tuple(int(value) for value in budget["training_replicate_seeds"])
            for row in target_rows
            if row["split"] == "training" and int(row["replicate_seed"]) == seed
        ]
        assert train_ids == expected_ids


def test_ranking_metric_ties_and_material_pairs_are_deterministic():
    truth = np.asarray([0.9, 0.8, 0.7, 0.4, 0.1])
    exact = run.ranking_metrics(truth, truth)
    reverse = run.ranking_metrics(truth, -truth)

    assert exact["spearman"] == pytest.approx(1.0)
    assert exact["top1_regret_r2"] == pytest.approx(0.0)
    assert exact["top3_opportunity_regret_r2"] == pytest.approx(0.0)
    assert exact["material_pair_accuracy"] == pytest.approx(1.0)
    assert reverse["spearman"] == pytest.approx(-1.0)
    assert reverse["exact_best"] is False


def test_new_budget_head_evaluation_retains_predictions_after_all_fits():
    rng = np.random.default_rng(9137)
    X_train = rng.normal(size=(72, 8)).astype(np.float32)
    y_train = (X_train[:, 0] * X_train[:, 1] + 0.2 * X_train[:, 2]).astype(np.float64)
    X_eval = rng.normal(size=(40, 8)).astype(np.float32)
    y_eval = (X_eval[:, 0] * X_eval[:, 1] + 0.2 * X_eval[:, 2]).astype(np.float64)

    heads = run._evaluate_heads_with_predictions(X_train, y_train, X_eval, y_eval, 7821)

    assert tuple(heads) == run.HEADS
    for head in run.HEADS:
        assert len(heads[head]["test_predictions"]) == len(y_eval)
        assert np.isfinite(heads[head]["test_predictions"]).all()
        assert np.isfinite(heads[head]["test_r2"])


def test_protocol_is_json_serializable_and_unknown_budget_fails():
    protocol = run.load_protocol()
    json.dumps(protocol)
    with pytest.raises(ValueError, match="budget is unknown"):
        run._budget_spec(protocol, "unknown")
