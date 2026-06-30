"""Behavior tests for ContinuousOverlapIndex."""

import numpy as np
import pytest

from overlapindex import ContinuousOverlapIndex


def _separated_regression_data(n=80):
    rng = np.random.default_rng(0)
    x_left = rng.normal(loc=-2.0, scale=0.12, size=(n // 2, 2))
    x_right = rng.normal(loc=2.0, scale=0.12, size=(n // 2, 2))
    X = np.vstack([x_left, x_right])
    y = np.concatenate([
        rng.normal(loc=-2.0, scale=0.08, size=n // 2),
        rng.normal(loc=2.0, scale=0.08, size=n // 2),
    ])
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def _overlapping_pathological_data(n=80):
    rng = np.random.default_rng(1)
    X = rng.normal(loc=0.0, scale=0.05, size=(n, 2))
    y = np.tile([-2.0, 2.0], n // 2) + rng.normal(scale=0.03, size=n)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def _model(**overrides):
    params = dict(
        model_type="KMeans",
        kmeans_k=2,
        kmeans_kwargs={"random_state": 0, "n_init": 10},
        n_target_cells=2,
        n_null_permutations=12,
        random_state=0,
    )
    params.update(overrides)
    return ContinuousOverlapIndex(**params)


def test_import_and_basic_api_returns_expected_types():
    X, y = _separated_regression_data()
    model = _model()

    assert model.fit(X, y) is model
    assert isinstance(model.score(), float)
    assert np.isclose(model.add_batch(X, y), model.index, atol=0.0, rtol=0.0)
    assert model.partial_fit(X, y) is model

    pred = model.predict(X[:5])
    assert pred.shape == (5,)
    assert np.issubdtype(pred.dtype, np.integer)
    assert 0.0 <= model.index <= 1.0
    assert 0.0 <= model.weighted_index <= 1.0


def test_univariate_auto_defaults_to_quantile_and_wasserstein():
    X, y = _separated_regression_data()
    model = _model(target_cover="auto", target_distance="auto")

    model.fit(X, y)

    assert model.target_cover_ == "quantile"
    assert model.target_distance_ == "wasserstein"


def test_multivariate_auto_defaults_to_kmeans_and_sliced_wasserstein():
    X, y = _separated_regression_data()
    Y = np.column_stack([y, y ** 2])
    model = _model(
        target_cover="auto",
        target_distance="auto",
        target_cover_kwargs={"n_init": 10},
        n_projections=8,
    )

    model.fit(X, Y)

    assert model.target_cover_ == "kmeans"
    assert model.target_distance_ == "sliced_wasserstein"
    assert np.isfinite(model.index)


def test_separated_regression_scores_above_null():
    X, y = _separated_regression_data()
    model = _model()

    model.fit(X, y)

    assert model.index > 0.55
    assert model.actual_loss_ < model.null_loss_


def test_random_target_assignment_scores_near_null():
    X, y = _separated_regression_data()
    y_random = np.random.default_rng(2).permutation(y)
    model = _model(n_null_permutations=20)

    model.fit(X, y_random)

    assert 0.25 <= model.index <= 0.75


def test_pathological_overlap_scores_below_null():
    X, y = _overlapping_pathological_data()
    model = _model(kmeans_k=2)

    model.fit(X, y)

    assert model.index < 0.5
    assert model.actual_loss_ > model.null_loss_


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_supported_offline_backends(model_type):
    X, y = _separated_regression_data()
    kwargs = {
        "model_type": model_type,
        "n_target_cells": 2,
        "n_null_permutations": 4,
        "random_state": 0,
    }
    if model_type == "KMeans":
        kwargs.update(kmeans_k=1, kmeans_kwargs={"random_state": 0, "n_init": 10})
    elif model_type == "MiniBatchKMeans":
        kwargs.update(
            kmeans_k=1,
            kmeans_kwargs={"random_state": 0, "n_init": 1, "batch_size": 16},
        )
    else:
        kwargs.update(
            ballcover_k=1,
            ballcover_radius="auto",
            ballcover_kwargs={"metric": "euclidean", "random_state": 0},
        )

    model = ContinuousOverlapIndex(**kwargs)

    model.fit(X, y)

    assert np.isfinite(model.index)
    assert 0.0 <= model.index <= 1.0


def test_random_state_makes_result_reproducible():
    X, y = _separated_regression_data()
    a = _model(random_state=4).fit(X, y)
    b = _model(random_state=4).fit(X, y)

    assert np.isclose(a.index, b.index, atol=0.0, rtol=0.0)
    assert np.isclose(a.null_loss_, b.null_loss_, atol=0.0, rtol=0.0)


def test_validation_errors_are_clear():
    X, y = _separated_regression_data()

    with pytest.raises(ValueError, match="X must be a 2D array"):
        _model().fit(X[:, 0], y)
    with pytest.raises(ValueError, match="Y must be numeric"):
        _model().fit(X, ["a"] * X.shape[0])
    with pytest.raises(ValueError, match="same number of rows"):
        _model().fit(X, y[:-1])
    with pytest.raises(NotImplementedError, match="offline backends"):
        _model(model_type="Fuzzy").fit(X, y)
    with pytest.raises(NotImplementedError, match="hard_top1"):
        _model(adjacency_mode="soft_topk").fit(X, y)
