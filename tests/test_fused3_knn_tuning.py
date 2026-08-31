from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from experiments.fused3_envelope_reanalysis import knn_tuning


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    train = rng.normal(size=(60, 7)).astype(np.float32)
    labels = np.repeat(np.arange(3), 20)
    train += np.eye(3, 7, dtype=np.float32)[labels] * 0.7
    query = rng.normal(size=(17, 7)).astype(np.float32)
    return train, labels, query


def test_one_search_predictions_match_sklearn_for_every_k() -> None:
    train, labels, query = _fixture()
    grid = (1, 3, 5, 11, 15)
    actual = knn_tuning.predict_k_grid(train, labels, query, k_grid=grid)
    normalized_train = normalize(train, copy=True)
    normalized_query = normalize(query, copy=True)
    for k in grid:
        expected = KNeighborsClassifier(
            n_neighbors=k, weights="distance", metric="cosine", n_jobs=1
        ).fit(normalized_train, labels).predict(normalized_query)
        assert np.array_equal(actual[k], expected)


def test_inner_cv_selection_is_deterministic_and_training_only() -> None:
    train, labels, _query = _fixture()
    first = knn_tuning.tune_k_inner_cv(
        train, labels, seed=17, k_grid=(1, 3, 5, 9)
    )
    second = knn_tuning.tune_k_inner_cv(
        train, labels, seed=17, k_grid=(1, 3, 5, 9)
    )
    assert first == second
    assert first[0] in {1, 3, 5, 9}
    assert set(first[1]) == {1, 3, 5, 9}


def test_k_grid_rejects_ambiguous_or_impossible_values() -> None:
    train, labels, query = _fixture()
    for grid in ((3, 1), (1, 1), (0, 1), (1, 61)):
        try:
            knn_tuning.predict_k_grid(train, labels, query, k_grid=grid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid grid was accepted: {grid!r}")
