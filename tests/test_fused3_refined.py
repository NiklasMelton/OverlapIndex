from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.fused3_confirmation_v2 import statistics
from experiments.fused3_envelope_reanalysis import refined_fused3 as refined
from experiments.scalable_relevance_kmeans import fused_food_screen


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(83)
    values = np.vstack(
        [
            rng.normal(loc=location, scale=0.35, size=(20, 8))
            for location in (0.0, 2.0, 4.0)
        ]
    ).astype(np.float32)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 20)
    return values, labels


def test_metric_backend_matches_weighted_fused3_scores() -> None:
    values, labels = _toy_data()
    selector = fused_food_screen._build_selector(
        "FUSED3", {"a": 3, "b": 3, "c": 3}, 17
    ).fit(values, labels)
    scale = np.asarray(np.sqrt(selector.weights_), dtype=np.float32)
    backend = refined._MetricCentroidBackend(selector.centers_ * scale, selector.owners_)
    observed = backend.score_block_prepared(values * scale)
    from experiments.scalable_relevance_kmeans.scalable import _score_kernel

    expected = _score_kernel(values, selector.centers_, selector.weights_)
    assert np.array_equal(observed, expected)


def test_balanced_median_refinement_applies_in_weighted_metric_deterministically() -> None:
    values = np.asarray(
        [[-1.2], [-0.9], [0.9], [1.2], [-0.1], [0.1], [8.0], [8.2]],
        dtype=np.float32,
    )
    labels = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object)
    centers = np.asarray([[-1.0], [1.0], [0.0], [8.1]], dtype=np.float32)
    centers_before = centers.copy()
    owners = np.asarray(["a", "a", "b", "b"], dtype=object)
    first = refined.refine_fused3_state(
        values,
        labels,
        centers=centers,
        owners=owners,
        weights=np.asarray([2.0]),
    )
    second = refined.refine_fused3_state(
        values,
        labels,
        centers=centers,
        owners=owners,
        weights=np.asarray([2.0]),
    )
    assert first["summary"]["applied_count"] >= 1
    assert first["summary"] == second["summary"]
    assert first["state_sha256"] == second["state_sha256"]
    assert np.array_equal(first["centers"], second["centers"])
    assert np.array_equal(centers, centers_before)


def test_crossfit_preserves_frozen_baseline_and_is_repeatable() -> None:
    values, labels = _toy_data()
    frozen = fused_food_screen.cross_fitted_candidate(
        values, labels, candidate_id="FUSED3", seed=143
    )
    first = refined.crossfit_fused3_refinement(
        values, labels, seed=143, collect_own_win_rates=True
    )
    second = refined.crossfit_fused3_refinement(
        values, labels, seed=143, collect_own_win_rates=True
    )
    assert first["scores"][refined.BASELINE] == frozen["score"]
    assert second["scores"] == first["scores"]
    assert len(first["folds"]) == 5
    assert all(row["fitted_fused3_state_unchanged"] for row in first["folds"])
    assert [row["refined_state_sha256"] for row in first["folds"]] == [
        row["refined_state_sha256"] for row in second["folds"]
    ]
    assert all(row["prototype_count_after"] >= row["prototype_count_before"] for row in first["folds"])


def test_summary_requires_complete_panels_and_tracks_a_refinement_flip() -> None:
    rows: list[dict[str, object]] = []
    dino = "dinov2-small"
    clip = "openclip-vit-b-32"
    for budget in statistics.BUDGETS:
        for seed in statistics.REPLICATE_SEEDS:
            for method in refined.METHODS:
                for index, backbone in enumerate(statistics.BACKBONES):
                    outcome = 0.5 + 0.01 * index
                    if backbone == dino:
                        outcome = 0.8
                    elif backbone == clip:
                        outcome = 0.7
                    score = 0.01 * index
                    if method == refined.BASELINE and backbone == clip:
                        score = 1.0
                    if method == refined.REFINED and backbone == dino:
                        score = 1.0
                    rows.append(
                        {
                            "budget": budget,
                            "replicate_seed": seed,
                            "candidate_id": method,
                            "backbone": backbone,
                            "score": score,
                            "best_head_envelope_accuracy": outcome,
                            "total_wall_seconds": 1.0 if method == refined.BASELINE else 1.2,
                            "prototype_count": 10 if method == refined.BASELINE else 12,
                            "applied_count": 0 if method == refined.BASELINE else 2,
                        }
                    )
    ranking, aggregate, focal = refined.summarize_refinement(rows)
    assert len(ranking) == 20
    assert len(aggregate) == 4
    assert len(focal) == 4
    baseline = next(
        row for row in aggregate if row["budget"] == 32 and row["candidate_id"] == refined.BASELINE
    )
    changed = next(
        row for row in aggregate if row["budget"] == 32 and row["candidate_id"] == refined.REFINED
    )
    assert json.loads(baseline["selection_counts"]) == {clip: 5}
    assert json.loads(changed["selection_counts"]) == {dino: 5}
    assert baseline["mean_regret_pp"] == pytest.approx(10.0)
    assert changed["mean_regret_pp"] == 0.0


def test_summary_rejects_incomplete_backbone_panel() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        refined.summarize_refinement(
            [
                {
                    "budget": 32,
                    "replicate_seed": statistics.REPLICATE_SEEDS[0],
                    "candidate_id": refined.BASELINE,
                    "backbone": statistics.BACKBONES[0],
                    "score": 0.5,
                    "best_head_envelope_accuracy": 0.5,
                    "total_wall_seconds": 1.0,
                    "prototype_count": 10,
                    "applied_count": 0,
                }
            ]
        )
