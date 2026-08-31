from __future__ import annotations

import numpy as np

from experiments.fused3_envelope_reanalysis.score_components import (
    COMPONENTS,
    component_geometry,
    crossfit_component_state,
)
from experiments.scalable_relevance_kmeans.scalable_v2 import fast_tiled_oi_score


def test_component_geometry_is_one_on_perfect_redundant_prototypes() -> None:
    values = np.asarray([[-2.5], [2.5]], dtype=np.float32)
    labels = np.asarray([0, 1])
    result = component_geometry(
        values,
        labels,
        centers=np.asarray([[-2.0], [-3.0], [2.0], [3.0]], dtype=np.float32),
        owners=np.asarray([0, 0, 1, 1]),
        weights=np.asarray([1.0]),
    )
    assert result["best_own_win_rate"] == 1.0
    assert result["second_own_win_rate"] == 1.0
    assert result["best_own_worst_rival_oi"] == 1.0
    assert result["mean_pair_second_own_oi"] == 1.0
    assert result["exact_oi"] == 1.0
    assert result["best_own_relative_margin"] > 0.0
    assert result["second_own_relative_margin"] > 0.0


def test_component_geometry_isolates_second_prototype_penalty() -> None:
    values = np.asarray([[-2.0], [2.0]], dtype=np.float32)
    labels = np.asarray([0, 1])
    result = component_geometry(
        values,
        labels,
        centers=np.asarray([[-2.0], [-10.0], [2.0], [10.0]], dtype=np.float32),
        owners=np.asarray([0, 0, 1, 1]),
        weights=np.asarray([1.0]),
    )
    assert result["best_own_win_rate"] == 1.0
    assert result["best_own_worst_rival_oi"] == 1.0
    assert result["second_own_win_rate"] == 0.0
    assert result["mean_pair_second_own_oi"] == 0.0
    assert result["exact_oi"] == 0.0
    assert result["best_own_relative_margin"] > 0.0
    assert result["second_own_relative_margin"] < 0.0


def test_component_exact_oi_matches_tiled_scorer_without_state_mutation() -> None:
    rng = np.random.default_rng(41)
    values = rng.normal(size=(36, 6)).astype(np.float32)
    labels = np.repeat(np.arange(3), 12)
    centers = rng.normal(size=(9, 6)).astype(np.float32)
    owners = np.repeat(np.arange(3), 3)
    weights = rng.uniform(0.2, 2.0, size=6)
    state = (centers.copy(), owners.copy(), weights.copy())

    result = component_geometry(
        values, labels, centers=centers, owners=owners, weights=weights
    )
    exact, _diagnostics = fast_tiled_oi_score(
        values,
        labels,
        centers=centers,
        owners=owners,
        weights=weights,
        memory_budget_mb=64,
    )

    assert result["exact_oi"] == exact
    assert set(result) == set(COMPONENTS)
    np.testing.assert_array_equal(centers, state[0])
    np.testing.assert_array_equal(owners, state[1])
    np.testing.assert_array_equal(weights, state[2])


def test_crossfit_component_states_are_deterministic_and_nested() -> None:
    rng = np.random.default_rng(43)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 20)
    values = np.vstack(
        [
            rng.normal(loc=location, scale=0.5, size=(20, 12))
            for location in (0.0, 2.0, 4.0)
        ]
    ).astype(np.float32)

    first = crossfit_component_state(
        values, labels, seed=143, square_feature_count=5
    )
    second = crossfit_component_state(
        values, labels, seed=143, square_feature_count=5
    )

    for component in COMPONENTS:
        assert first[component] == second[component]
    assert len(first["folds"]) == 5
    for left, right in zip(first["folds"], second["folds"]):
        for component in COMPONENTS:
            assert left[component] == right[component]
        assert left["lift_state_sha256"] == right["lift_state_sha256"]
        assert left["selected_square_feature_count"] == 5
