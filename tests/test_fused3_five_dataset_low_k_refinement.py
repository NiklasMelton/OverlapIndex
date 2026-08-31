from __future__ import annotations

import json

import pytest

from experiments.fused3_confirmation_v2 import statistics
from experiments.fused3_envelope_reanalysis import (
    five_dataset_low_k_refinement as five,
)


def _synthetic_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selectors: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    for dataset in statistics.DATASET_IDS:
        for seed in statistics.REPLICATE_SEEDS:
            for budget in statistics.BUDGETS:
                for index, backbone in enumerate(statistics.BACKBONES):
                    head_values = {
                        "linear": 0.50 + 0.01 * index,
                        "quadratic": 0.55 + 0.01 * index,
                        "knn": 0.53 + 0.01 * index,
                        "rbf": 0.54 + 0.01 * index,
                    }
                    for head, accuracy in head_values.items():
                        references.append(
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate_seed": seed,
                                "budget": budget,
                                "head": head,
                                "test_accuracy": accuracy,
                            }
                        )
                    for method in five.METHODS:
                        selected_index = 9 if method == five.LOW_REFINED else 8
                        selectors.append(
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate_seed": seed,
                                "budget": budget,
                                "candidate_id": method,
                                "score": 1.0 if index == selected_index else 0.01 * index,
                                "total_wall_seconds": 1.0,
                            }
                        )
    return selectors, references


def test_envelope_metrics_use_best_head_and_tie_safe_selection() -> None:
    selectors, references = _synthetic_rows()
    metrics = five.envelope_panel_metrics(selectors, references)
    assert len(metrics) == 200
    refined = [row for row in metrics if row["candidate_id"] == five.LOW_REFINED]
    frozen = [row for row in metrics if row["candidate_id"] == five.FROZEN]
    assert all(row["regret_pp"] == 0.0 for row in refined)
    assert all(row["regret_pp"] == pytest.approx(1.0) for row in frozen)
    assert all(json.loads(row["selected_backbones"]) == [statistics.BACKBONES[-1]] for row in refined)


def test_aggregate_and_contrast_preserve_equal_dataset_weighting() -> None:
    selectors, references = _synthetic_rows()
    metrics = five.envelope_panel_metrics(selectors, references)
    datasets, overall = five.aggregate_envelope(metrics)
    assert len(datasets) == 20
    assert len(overall) == 4
    refined = next(row for row in overall if row["candidate_id"] == five.LOW_REFINED)
    assert refined["equal_dataset_mean_regret_pp"] == 0.0
    contrast = five.contrast_interval(
        metrics,
        candidate_id=five.LOW_REFINED,
        comparator_id=five.FROZEN,
    )
    assert contrast["estimate"] == pytest.approx(-1.0)
    assert contrast["per_dataset"] == {
        dataset: pytest.approx(-1.0) for dataset in statistics.DATASET_IDS
    }


def test_head_metrics_and_runtime_are_complete() -> None:
    selectors, references = _synthetic_rows()
    heads = five.head_panel_metrics(selectors, references)
    assert len(heads) == 800
    aggregates = five.aggregate_heads(heads)
    assert len(aggregates) == 16
    runtime = five.runtime_summary(selectors)
    assert len(runtime) == 40
    assert all(row["median_call_ratio_to_lp"] == 1.0 for row in runtime)


def test_comparison_validation_rejects_missing_or_duplicate_rows() -> None:
    selectors, _references = _synthetic_rows()
    with pytest.raises(ValueError, match="incomplete"):
        five._validate_comparison_rows(selectors[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        five._validate_comparison_rows([*selectors, selectors[0]])
