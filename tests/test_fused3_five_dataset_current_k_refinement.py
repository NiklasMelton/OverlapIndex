from __future__ import annotations

import json

import pytest

from experiments.fused3_confirmation_v2 import statistics
from experiments.fused3_envelope_reanalysis import (
    five_dataset_current_k_refinement as current,
)
from experiments.fused3_envelope_reanalysis import (
    five_dataset_low_k_refinement as shared,
)


def _rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selectors: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    for dataset in statistics.DATASET_IDS:
        for seed in statistics.REPLICATE_SEEDS:
            for budget in statistics.BUDGETS:
                for index, backbone in enumerate(statistics.BACKBONES):
                    for head, offset in zip(statistics.HEADS, (0.0, 0.05, 0.03, 0.04)):
                        references.append(
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate_seed": seed,
                                "budget": budget,
                                "head": head,
                                "test_accuracy": 0.5 + offset + 0.01 * index,
                            }
                        )
                    for method in current.METHODS:
                        selected = 9 if method == current.CURRENT_REFINED else 8
                        selectors.append(
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate_seed": seed,
                                "budget": budget,
                                "candidate_id": method,
                                "score": 1.0 if index == selected else 0.01 * index,
                                "total_wall_seconds": 1.0,
                            }
                        )
    return selectors, references


def test_current_k_comparison_uses_closed_three_method_surface() -> None:
    selectors, references = _rows()
    metrics = shared.envelope_panel_metrics(
        selectors, references, methods=current.METHODS
    )
    assert len(metrics) == 150
    datasets, overall = shared.aggregate_envelope(
        metrics, methods=current.METHODS
    )
    assert len(datasets) == 15
    assert len(overall) == 3
    refined = next(
        row for row in overall if row["candidate_id"] == current.CURRENT_REFINED
    )
    assert refined["equal_dataset_mean_regret_pp"] == 0.0


def test_current_k_contrast_and_report_sign_are_explicit() -> None:
    selectors, references = _rows()
    metrics = shared.envelope_panel_metrics(
        selectors, references, methods=current.METHODS
    )
    contrast = shared.contrast_interval(
        metrics,
        candidate_id=current.CURRENT_REFINED,
        comparator_id=current.FROZEN,
    )
    assert contrast["estimate"] == pytest.approx(-1.0)
    summary = {
        "overall_aggregate": shared.aggregate_envelope(
            metrics, methods=current.METHODS
        )[1],
        "dataset_aggregate": shared.aggregate_envelope(
            metrics, methods=current.METHODS
        )[0],
        "regret_contrasts": [contrast],
        "head_aggregate": shared.aggregate_heads(
            shared.head_panel_metrics(
                selectors, references, methods=current.METHODS
            ),
            methods=current.METHODS,
        ),
        "runtime_descriptive": shared.runtime_summary(
            selectors,
            methods=current.METHODS,
            executed_methods=(current.CURRENT_REFINED,),
        ),
    }
    report = current.render_report(summary)
    assert "Negative values favor frozen-K refined FUSED3" in report
    assert current.CURRENT_REFINED in report


def test_frozen_score_lookup_requires_exact_complete_grid() -> None:
    selectors, _references = _rows()
    frozen = [row for row in selectors if row["candidate_id"] == current.FROZEN]
    lookup = current._frozen_score_lookup(frozen)
    assert len(lookup) == 500
    with pytest.raises(ValueError, match="incomplete"):
        current._frozen_score_lookup(frozen[:-1])
