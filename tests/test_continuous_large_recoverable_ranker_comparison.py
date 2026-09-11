"""Focused tests for the frozen recoverable-target ranker comparison."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_large_recoverable_ranker_comparison import run
from experiments.continuous_large_structured_opportunity_benchmark import extract
from experiments.continuous_large_structured_opportunity_benchmark import run as source_run


def test_protocol_locks_methods_targets_metrics_and_runtime_gate():
    protocol = run.load_protocol()

    assert tuple(protocol["rankers"]["methods"]) == run.METHODS
    assert tuple(protocol["rankers"]["candidate_methods"]) == run.CANDIDATE_METHODS
    assert protocol["evaluation"]["topk"] == 3
    assert protocol["evaluation"]["material_pair_difference_r2"] == 0.01
    assert protocol["success"]["maximum_aggregate_runtime_ratio_to_linear_cv"] == 3.0
    assert {
        target
        for targets in protocol["strata"].values()
        for target in targets
    } == {
        "lsp_2000_pose_geometry::ankle_span_over_torso_length",
        "lsp_2000_pose_geometry::wrist_span_over_torso_length",
        "pascal_voc_2007_largest_box_geometry::largest_box_area_fraction",
        "pascal_voc_2007_largest_box_geometry::largest_box_log_aspect_ratio",
        "coco_2017_person_pose_geometry::wrist_span_over_torso_length",
        "cub_200_2011_part_geometry::leg_span_over_bbox_diagonal",
        "cub_200_2011_part_geometry::wing_axis_cosine",
        "oxford_iiit_pet_mask_geometry::foreground_compactness",
    }


def test_source_selection_is_complete_and_uses_no_reserved_backbones():
    protocol = run.load_protocol()
    raw, _summary = run.validate_source(protocol)
    selected = run.selected_source_panels(raw, protocol)

    assert len(selected) == 24
    assert sum(len(panel["candidates"]) for panel in selected) == 120
    assert {
        panel["stratum"] for panel in selected
    } == set(protocol["strata"])
    assert all(
        tuple(candidate["candidate_id"] for candidate in panel["candidates"])
        == extract.BACKBONES
        for panel in selected
    )
    assert protocol["backbones"]["reserved_evaluated"] == []


def test_selector_seed_uses_original_protocol_order_not_sorted_artifact_order():
    protocol = {
        **run.load_protocol(),
        "source_protocol": source_run.load_protocol(),
    }

    assert tuple(protocol["source_protocol"]["datasets"]).index(
        "lsp_2000_pose_geometry"
    ) == 1
    assert run._selector_seed(
        protocol,
        "lsp_2000_pose_geometry",
        "ankle_span_over_torso_length",
        4103,
    ) == 4223


def test_ranking_metrics_cover_ranking_shortlist_and_material_pairs():
    truth = [0.9, 0.7, 0.8, 0.4, 0.6]
    exact = run.ranking_metrics(truth, truth)
    reversed_result = run.ranking_metrics(truth, [-value for value in truth])

    assert exact["spearman"] == pytest.approx(1.0)
    assert exact["top1_regret_r2"] == 0.0
    assert exact["exact_best"] is True
    assert exact["top3_opportunity_regret_r2"] == 0.0
    assert exact["top3_mean_utility_regret_r2"] == 0.0
    assert exact["top3_set_recall"] == 1.0
    assert exact["material_pair_accuracy"] == 1.0
    assert reversed_result["spearman"] == pytest.approx(-1.0)
    assert reversed_result["top1_regret_r2"] == pytest.approx(0.5)
    assert reversed_result["exact_best"] is False


def test_save_only_optimization_is_score_exact_to_shared_evaluator():
    rng = np.random.default_rng(2718)
    X = rng.normal(size=(160, 12)).astype(np.float32)
    y = X[:, 0] * X[:, 1] + 0.3 * X[:, 2] ** 2 + 0.05 * rng.normal(size=160)
    primitive = np.zeros((len(X), 2), dtype=np.float32)

    canonical = shared.evaluate_shared_pair(X, primitive, y, 91, 2)
    optimized = run.evaluate_save_only(X, y, 91, 2)

    assert optimized["outer_fit_count"] == 4
    assert optimized["score"] == canonical["save_rff_current"]["score"]
    assert [row["score"] for row in optimized["folds"]] == [
        row["save_rff_current"]["score"] for row in canonical["folds"]
    ]
    assert [row["prototype_count"] for row in optimized["folds"]] == [
        row["prototype_count"] for row in canonical["folds"]
    ]


def _summary_row(group: str, method: str, spearman: float, top1: float, top3: float):
    return {
        "group": group,
        "method": method,
        "mean_spearman": spearman,
        "mean_top1_regret_r2": top1,
        "mean_top3_opportunity_regret_r2": top3,
    }


def test_decision_requires_one_candidate_to_pass_every_frozen_check():
    protocol = run.load_protocol()
    summaries = []
    for group in run.GROUPS:
        summaries.extend(
            [
                _summary_row(group, run.LINEAR, 0.70, 0.020, 0.002),
                _summary_row(group, run.BROAD, 0.76, 0.010, 0.003),
                _summary_row(group, run.SAVE, 0.71, 0.018, 0.002),
                _summary_row(group, run.CV_MAX, 0.80, 0.005, 0.000),
            ]
        )
    runtimes = [
        {"method": run.LINEAR, "relative_aggregate_to_linear_cv": 1.0},
        {"method": run.BROAD, "relative_aggregate_to_linear_cv": 2.5},
        {"method": run.SAVE, "relative_aggregate_to_linear_cv": 1.5},
        {"method": run.CV_MAX, "relative_aggregate_to_linear_cv": 8.0},
    ]

    result = run.decision(summaries, runtimes, protocol)

    assert result["any_candidate_passed"] is True
    assert result["passing_methods"] == [run.BROAD]
    assert result["method_promotion"] is None


def test_protocol_hash_sidecar_matches_exact_bytes():
    assert run.sha256_path(run.PROTOCOL_PATH) == run.PROTOCOL_SHA256_PATH.read_text(
        encoding="utf-8"
    ).strip()
    json.loads(run.PROTOCOL_PATH.read_text(encoding="utf-8"))
