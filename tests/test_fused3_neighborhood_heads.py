from __future__ import annotations

import numpy as np

from experiments.fused3_envelope_reanalysis import neighborhood_heads as heads


def _toy() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    train = np.vstack(
        [rng.normal((-2.0, 0.0), 0.25, size=(18, 2)), rng.normal((2.0, 0.0), 0.25, size=(18, 2))]
    ).astype(np.float32)
    labels = np.repeat(np.array(["left", "right"], dtype=object), 18)
    test = np.array([[-2.1, 0.1], [2.1, -0.1]], dtype=np.float32)
    expected = np.array(["left", "right"], dtype=object)
    return train, labels, test, expected


def test_soft_cosine_parzen_is_deterministic_and_local() -> None:
    train, labels, test, expected = _toy()
    first = heads.soft_cosine_predictions(
        train, labels, test, beta_grid=(2.0, 8.0)
    )
    second = heads.soft_cosine_predictions(
        train, labels, test, beta_grid=(2.0, 8.0)
    )
    assert np.array_equal(first[8.0], expected)
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_inner_selection_does_not_use_evaluation_labels() -> None:
    train, labels, test, expected = _toy()
    methods = ("SOFT_COSINE_PARZEN",)
    first = heads.fit_evaluate_heads(
        train, labels, test, expected, seed=11, methods=methods
    )[0]
    changed = heads.fit_evaluate_heads(
        train, labels, test, expected[::-1], seed=11, methods=methods
    )[0]
    assert first["selected_config"] == changed["selected_config"]
    assert first["inner_cv_accuracy"] == changed["inner_cv_accuracy"]
    assert first["test_accuracy"] != changed["test_accuracy"]


def test_centroid_and_pca_subspace_heads_fit_and_predict() -> None:
    train, labels, test, expected = _toy()
    original_dims = heads.SUBSPACE_DIM_GRID
    original_ks = heads.SUBSPACE_K_GRID
    try:
        heads.SUBSPACE_DIM_GRID = (1,)
        heads.SUBSPACE_K_GRID = (1, 3)
        rows = heads.fit_evaluate_heads(
            train,
            labels,
            test,
            expected,
            seed=13,
            methods=("CENTROID_SUBSPACE_KNN", "PCA_WHITENED_KNN"),
        )
    finally:
        heads.SUBSPACE_DIM_GRID = original_dims
        heads.SUBSPACE_K_GRID = original_ks
    assert {row["method"] for row in rows} == {
        "CENTROID_SUBSPACE_KNN",
        "PCA_WHITENED_KNN",
    }
    assert all(row["test_accuracy"] == 1.0 for row in rows)


def test_rbf_tuning_uses_closed_grid_and_returns_finite_scores() -> None:
    train, labels, test, expected = _toy()
    selected, scores = heads.tune_rbf_svc(
        train,
        labels,
        seed=17,
        grid=((1.0, 1.0), (3.0, 1.0)),
    )
    assert selected in scores
    assert all(np.isfinite(value) for value in scores.values())
    original_grid = heads.RBF_GRID
    try:
        heads.RBF_GRID = ((1.0, 1.0),)
        row = heads.fit_evaluate_heads(
            train,
            labels,
            test,
            expected,
            seed=17,
            methods=("RBF_SVC_TUNED",),
        )[0]
    finally:
        heads.RBF_GRID = original_grid
    assert row["test_accuracy"] == 1.0
    assert row["selected_config"]["C"] == 1.0
