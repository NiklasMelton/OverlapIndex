from __future__ import annotations

import numpy as np

from experiments.fused3_envelope_reanalysis import structure_heads as heads


def _toy():
    rng = np.random.default_rng(23)
    train = np.vstack([
        rng.normal((-2.0, 0.0, 0.0), 0.2, size=(18, 3)),
        rng.normal((2.0, 0.0, 0.0), 0.2, size=(18, 3)),
    ]).astype(np.float32)
    labels = np.repeat(np.array([0, 1]), 18)
    test = np.array([[-2.0, 0.1, 0.0], [2.0, -0.1, 0.0]], dtype=np.float32)
    expected = np.array([0, 1])
    return train, labels, test, expected


def test_nearest_subspace_predicts_and_supports_centroid_control() -> None:
    train, labels, test, expected = _toy()
    predictions = heads.nearest_subspace_predictions(
        train, labels, test, ranks=(0, 1)
    )
    assert np.array_equal(predictions[0], expected)
    assert np.array_equal(predictions[1], expected)


def test_structure_head_selection_is_independent_of_evaluation_labels() -> None:
    train, labels, test, expected = _toy()
    original_ranks = heads.SUBSPACE_RANK_GRID
    try:
        heads.SUBSPACE_RANK_GRID = (0, 1)
        first = heads.fit_evaluate_heads(
            train, labels, test, expected, seed=31,
            methods=("NEAREST_CLASS_SUBSPACE",),
        )[0]
        changed = heads.fit_evaluate_heads(
            train, labels, test, expected[::-1], seed=31,
            methods=("NEAREST_CLASS_SUBSPACE",),
        )[0]
    finally:
        heads.SUBSPACE_RANK_GRID = original_ranks
    assert first["selected_config"] == changed["selected_config"]
    assert first["inner_cv_accuracy"] == changed["inner_cv_accuracy"]
    assert first["test_accuracy"] != changed["test_accuracy"]


def test_diagonal_gaussian_returns_closed_finite_result() -> None:
    train, labels, test, expected = _toy()
    row = heads.fit_evaluate_heads(
        train, labels, test, expected, seed=37,
        methods=("DIAGONAL_GAUSSIAN",),
    )[0]
    assert row["test_accuracy"] == 1.0
    assert row["selected_config"]["var_smoothing"] in heads.VAR_SMOOTHING_GRID
