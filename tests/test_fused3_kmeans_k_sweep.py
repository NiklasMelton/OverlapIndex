from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.fused3_confirmation_v2 import statistics
from experiments.fused3_envelope_reanalysis import kmeans_k_sweep as sweep
from experiments.scalable_relevance_kmeans import fused_food_screen


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(71)
    values = np.vstack(
        [
            rng.normal(loc=location, scale=0.45, size=(20, 10))
            for location in (0.0, 2.0, 4.0)
        ]
    ).astype(np.float32)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 20)
    return values, labels


def test_constant_k_and_closed_grid_fail_closed() -> None:
    _values, labels = _toy_data()
    train = np.concatenate(
        [np.arange(0, 16), np.arange(20, 36), np.arange(40, 56)]
    )
    mapping = sweep.constant_k_per_class(labels, train, k=8)
    assert mapping == {"a": 8, "b": 8, "c": 8}
    with pytest.raises(ValueError, match="smallest training-fold class"):
        sweep.constant_k_per_class(labels, train, k=17)
    with pytest.raises(ValueError, match="sorted unique"):
        sweep.validate_k_grid({32: (5, 3), 64: (10,)})
    with pytest.raises(ValueError, match="frozen recipe"):
        sweep.validate_k_grid({32: (2, 3), 64: (2, 10)})


def test_current_k_reproduces_frozen_fused3_recipe_and_is_deterministic() -> None:
    values, labels = _toy_data()
    # Five-fold training leaves 16 rows/class, so the frozen rule resolves K=3.
    frozen = fused_food_screen.cross_fitted_candidate(
        values, labels, candidate_id="FUSED3", seed=143
    )
    first = sweep.crossfit_fused3_at_k(
        values,
        labels,
        seed=143,
        k=3,
        collect_own_win_rates=True,
    )
    second = sweep.crossfit_fused3_at_k(
        values,
        labels,
        seed=143,
        k=3,
        collect_own_win_rates=True,
    )
    assert first["score"] == frozen["score"]
    assert second["score"] == first["score"]
    assert first["best_own_win_rate"] == second["best_own_win_rate"]
    assert first["second_own_win_rate"] == second["second_own_win_rate"]
    assert len(first["folds"]) == 5
    assert all(row["k_per_class"] == 3 for row in first["folds"])
    assert all(row["prototype_count"] == 9 for row in first["folds"])
    assert all(row["state_unchanged_after_score_fixed"] for row in first["folds"])


def test_summary_preserves_complete_panels_and_exposes_selection_changes() -> None:
    state_rows: list[dict[str, object]] = []
    clip = "openclip-vit-b-32"
    dino = "dinov2-small"
    for budget in statistics.BUDGETS:
        for seed in statistics.REPLICATE_SEEDS:
            for k in sweep.K_GRID_BY_BUDGET[budget]:
                for index, backbone in enumerate(statistics.BACKBONES):
                    outcome = 0.5 + 0.01 * index
                    if backbone == dino:
                        outcome = 0.80
                    elif backbone == clip:
                        outcome = 0.70
                    score = 0.01 * index
                    if k == 2:
                        score = 1.0 if backbone == dino else score
                    else:
                        score = 1.0 if backbone == clip else score
                    state_rows.append(
                        {
                            "budget": budget,
                            "replicate_seed": seed,
                            "k_per_class": k,
                            "backbone": backbone,
                            "score": score,
                            "best_head_envelope_accuracy": outcome,
                            "best_own_win_rate": 0.8 if backbone == dino else 0.9,
                            "second_own_win_rate": 0.6 if backbone == dino else 0.7,
                            "total_wall_seconds": float(k),
                        }
                    )

    ranking, aggregate, focal = sweep.summarize_k_sweep(state_rows)
    assert len(ranking) == sum(
        len(sweep.K_GRID_BY_BUDGET[budget]) * len(statistics.REPLICATE_SEEDS)
        for budget in statistics.BUDGETS
    )
    assert len(aggregate) == sum(map(len, sweep.K_GRID_BY_BUDGET.values()))
    assert len(focal) == len(aggregate)
    k2 = next(row for row in aggregate if row["budget"] == 32 and row["k_per_class"] == 2)
    k3 = next(row for row in aggregate if row["budget"] == 32 and row["k_per_class"] == 3)
    assert json.loads(k2["selection_counts"]) == {dino: 5}
    assert json.loads(k3["selection_counts"]) == {clip: 5}
    assert k2["mean_regret_pp"] == 0.0
    assert k3["mean_regret_pp"] == pytest.approx(10.0)


def test_own_win_rates_isolate_second_prototype_failure() -> None:
    values = np.asarray([[-2.0], [2.0]], dtype=np.float32)
    labels = np.asarray([0, 1])
    result = sweep.own_prototype_win_rates(
        values,
        labels,
        centers=np.asarray([[-2.0], [-10.0], [2.0], [10.0]], dtype=np.float32),
        owners=np.asarray([0, 0, 1, 1]),
        weights=np.asarray([1.0]),
    )
    assert result == {"best_own_win_rate": 1.0, "second_own_win_rate": 0.0}
