from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.fused3_confirmation_v2 import statistics as stats


def _rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selectors: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    for dataset in stats.DATASET_IDS:
        for backbone_index, backbone in enumerate(stats.BACKBONES):
            for replicate, replicate_seed in enumerate(stats.REPLICATE_SEEDS):
                for budget in stats.BUDGETS:
                    selectors.extend(
                        [
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate": replicate,
                                "replicate_seed": replicate_seed,
                                "budget": budget,
                                "candidate_id": "FUSED3",
                                "score": float(backbone_index),
                                "status": "ok",
                            },
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate": replicate,
                                "replicate_seed": replicate_seed,
                                "budget": budget,
                                "candidate_id": "LP-FULL",
                                "score": float(-backbone_index),
                                "status": "ok",
                            },
                        ]
                    )
                    for head in stats.HEADS:
                        step = 0.0005 if head == "linear" else 0.01
                        references.append(
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate": replicate,
                                "replicate_seed": replicate_seed,
                                "budget": budget,
                                "head": head,
                                "test_accuracy": 0.5 + step * backbone_index,
                                "status": "ok",
                            }
                        )
    return selectors, references


def _contrasts(
    values_by_head: dict[str, dict[str, float] | float]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in stats.DATASET_IDS:
        for replicate_seed in stats.REPLICATE_SEEDS:
            for budget in stats.BUDGETS:
                for head in stats.HEADS:
                    value = values_by_head[head]
                    contrast = value[dataset] if isinstance(value, dict) else value
                    rows.append(
                        {
                            "dataset_id": dataset,
                            "replicate_seed": replicate_seed,
                            "budget": budget,
                            "head": head,
                            "candidate_id": "FUSED3",
                            "comparator_id": "LP-FULL",
                            "contrast_pp": float(contrast),
                        }
                    )
    return rows


def test_exact_complete_grid_and_iterators_are_supported() -> None:
    selectors, references = _rows()
    materialized = stats.validate_confirmation_inputs(
        (row for row in selectors), (row for row in references)
    )
    assert len(materialized[0]) == 1000
    assert len(materialized[1]) == 2000

    with pytest.raises(ValueError, match="exact confirmation grid"):
        stats.validate_confirmation_inputs(selectors[:-1], references)
    duplicated = selectors + [copy.deepcopy(selectors[0])]
    with pytest.raises(ValueError, match="duplicate selector identity"):
        stats.validate_confirmation_inputs(duplicated, references)
    malformed = copy.deepcopy(references)
    malformed[0]["test_accuracy"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        stats.validate_confirmation_inputs(selectors, malformed)


def test_panel_metrics_use_archived_tie_averaging() -> None:
    selectors, references = _rows()
    target = next(
        row
        for row in selectors
        if row["dataset_id"] == stats.DATASET_IDS[0]
        and row["replicate_seed"] == stats.REPLICATE_SEEDS[0]
        and row["budget"] == 32
        and row["candidate_id"] == "FUSED3"
        and row["backbone"] == stats.BACKBONES[-2]
    )
    target["score"] = 9.0 - 5.0e-13
    rows = stats.panel_metrics(selectors, references)
    metric = next(
        row
        for row in rows
        if row["dataset_id"] == stats.DATASET_IDS[0]
        and row["replicate_seed"] == stats.REPLICATE_SEEDS[0]
        and row["budget"] == 32
        and row["head"] == "quadratic"
        and row["candidate_id"] == "FUSED3"
    )
    assert metric["selected_count"] == 2
    assert metric["selected_backbones"] == list(stats.BACKBONES[-2:])
    assert np.isclose(metric["selected_accuracy"], 0.585)
    assert np.isclose(metric["regret_pp"], 0.5)
    assert metric["exact_best"] == 0.5


def test_constant_spearman_is_null_and_rank_auc_is_supporting_only() -> None:
    selectors, references = _rows()
    for row in selectors:
        if row["candidate_id"] == "FUSED3":
            row["score"] = 1.0
    metrics = stats.panel_metrics(selectors, references)
    fused = [row for row in metrics if row["candidate_id"] == "FUSED3"]
    assert all(row["spearman"] is None for row in fused)
    assert {row["spearman_status"] for row in fused} == {"undefined_constant"}
    auc = stats.rank_auc_rows(metrics)
    fused_auc = [row for row in auc if row["candidate_id"] == "FUSED3"]
    assert all(row["rank_auc"] is None for row in fused_auc)
    assert all(row["supporting_only"] and not row["promotion_veto"] for row in fused_auc)
    summary = stats.summarize(selectors, references)
    assert summary["rank_support"]["status"] == "incomplete_undefined"
    assert summary["decision"]["status"] == "pass_confirmed"


def test_two_budget_log2_auc_and_complete_curve_requirement() -> None:
    selectors, references = _rows()
    metrics = stats.panel_metrics(selectors, references)
    auc = stats.rank_auc_rows(metrics)
    assert len(auc) == 5 * 5 * 4 * 2
    assert all(
        np.isclose(row["rank_auc"], 1.0)
        for row in auc
        if row["candidate_id"] == "FUSED3"
    )
    assert all(
        np.isclose(row["rank_auc"], -1.0)
        for row in auc
        if row["candidate_id"] == "LP-FULL"
    )
    with pytest.raises(ValueError, match="both frozen budgets"):
        stats.rank_auc_rows(
            row
            for row in metrics
            if not (
                row["dataset_id"] == stats.DATASET_IDS[0]
                and row["replicate_seed"] == stats.REPLICATE_SEEDS[0]
                and row["head"] == "linear"
                and row["candidate_id"] == "FUSED3"
                and row["budget"] == 32
            )
        )


def test_hierarchical_bootstrap_is_deterministic_and_equally_weighted() -> None:
    values = {
        dataset: float(position)
        for position, dataset in enumerate(stats.DATASET_IDS)
    }
    rows = _contrasts({head: values for head in stats.HEADS})
    linear = [row for row in rows if row["head"] == "linear"]
    first = stats.hierarchical_interval(linear, n_resamples=512)
    second = stats.hierarchical_interval((row for row in linear), n_resamples=512)
    assert first == second
    assert first["estimate"] == 2.0
    assert first["per_dataset"] == values
    assert first["bootstrap_seed"] == 2026082702
    assert first["aggregation"] == "equal_dataset_equal_replicate_equal_budget"


def test_frozen_gate_boundaries_and_no_reselection() -> None:
    passing = stats.confirmation_gates(
        _contrasts(
            {
                "linear": 1.0,
                "quadratic": -0.1,
                "knn": -0.1,
                "rbf": -0.1,
            }
        ),
        n_resamples=256,
    )
    assert passing["status"] == "pass_confirmed"
    assert passing["gates"]["linear"]["upper_95"] == 1.0
    assert passing["runner_up_allowed"] is False
    assert passing["reselection_performed"] is False

    failing = stats.confirmation_gates(
        _contrasts(
            {
                "linear": 1.00001,
                "quadratic": 0.0,
                "knn": -0.1,
                "rbf": -0.1,
            }
        ),
        n_resamples=256,
    )
    assert failing["status"] == "fail_locked_candidate"
    assert failing["gates"]["linear"]["status"] == "fail"
    assert failing["gates"]["quadratic"]["status"] == "fail"


def test_nonlinear_requires_four_of_five_favorable_datasets() -> None:
    four = {
        dataset: (-100.0 if position < 4 else 0.01)
        for position, dataset in enumerate(stats.DATASET_IDS)
    }
    three = {
        dataset: (-100.0 if position < 3 else 0.01)
        for position, dataset in enumerate(stats.DATASET_IDS)
    }
    passed = stats.confirmation_gates(
        _contrasts(
            {
                "linear": 0.0,
                "quadratic": four,
                "knn": four,
                "rbf": four,
            }
        ),
        n_resamples=2048,
    )
    assert passed["status"] == "pass_confirmed"
    assert passed["gates"]["quadratic"]["favorable_dataset_count"] == 4

    failed = stats.confirmation_gates(
        _contrasts(
            {
                "linear": 0.0,
                "quadratic": three,
                "knn": four,
                "rbf": four,
            }
        ),
        n_resamples=2048,
    )
    assert failed["status"] == "fail_locked_candidate"
    assert failed["gates"]["quadratic"]["favorable_dataset_count"] == 3


def test_summary_emits_closed_decision_and_descriptive_runtime_only() -> None:
    selectors, references = _rows()
    summary = stats.summarize(selectors, references)
    assert summary["decision"]["status"] == "pass_confirmed"
    assert summary["decision"]["selected_candidate"] == "FUSED3"
    assert summary["decision"]["runner_up_allowed"] is False
    assert summary["runtime"] == {
        "gate_source": "hash_bound_passed_fused3_food_resource_evidence",
        "new_dataset_timings": "descriptive_only",
        "promotion_veto": False,
    }
    assert summary["rank_support"]["supporting_only"] is True
