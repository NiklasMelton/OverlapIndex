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


def _worse_than_null_data(n=80):
    rng = np.random.default_rng(1)
    X = rng.normal(loc=0.0, scale=0.05, size=(n, 2))
    y = np.tile([-2.0, 2.0], n // 2) + rng.normal(scale=0.03, size=n)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X, y


def _partially_overlapping_data():
    X, y = _separated_regression_data()
    y = y + np.random.default_rng(12).normal(scale=1.5, size=y.shape[0])
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
    assert 0.0 <= model.macro_index_ <= 1.0
    assert 0.0 <= model.weighted_index <= 1.0
    assert all(0.0 <= score <= 1.0 for score in model.prototype_index_.values())
    assert not hasattr(model, "raw_index_")


def test_removed_clip_parameter_is_not_exposed_or_accepted():
    assert "clip" not in ContinuousOverlapIndex().get_params()

    with pytest.raises(TypeError, match="unexpected keyword argument 'clip'"):
        ContinuousOverlapIndex(clip=False)


@pytest.mark.parametrize(
    ("actual_loss", "null_loss", "expected"),
    [
        (0.0, 2.0, 1.0),
        (2.0, 2.0, 0.0),
        (3.0, 2.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
)
def test_loss_calibration_has_bounded_zero_and_one_anchors(
    actual_loss,
    null_loss,
    expected,
):
    assert ContinuousOverlapIndex._index_from_losses(actual_loss, null_loss) == expected


def test_default_adjacency_mode_is_soft_topk():
    assert ContinuousOverlapIndex().adjacency_mode == "soft_topk"


def test_default_null_mode_is_auto():
    model = ContinuousOverlapIndex()

    assert model.null_mode == "auto"
    assert model.auto_null_work_threshold == 100_000


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
    assert model.index == pytest.approx(1.0 - model.loss_ratio_)


def test_separated_targets_score_above_partially_overlapping_targets():
    X_separated, y_separated = _separated_regression_data()
    X_partial, y_partial = _partially_overlapping_data()

    separated = _model(n_null_permutations=20).fit(X_separated, y_separated)
    partial = _model(n_null_permutations=20).fit(X_partial, y_partial)

    assert 0.0 < partial.index < separated.index < 1.0


def test_random_target_assignment_scores_near_zero():
    X, y = _separated_regression_data()
    y_random = np.random.default_rng(2).permutation(y)
    model = _model(n_null_permutations=20)

    model.fit(X, y_random)

    assert 0.0 <= model.index <= 0.1
    assert model.loss_ratio_ == pytest.approx(1.0, abs=0.1)


def test_worse_than_null_overlap_stays_at_bounded_zero():
    X, y = _worse_than_null_data()
    model = _model(kmeans_k=2, adjacency_mode="hard_top1")

    model.fit(X, y)

    assert model.index == 0.0
    assert model.actual_loss_ > model.null_loss_
    assert model.loss_ratio_ > 1.0
    assert 0.0 <= model.macro_index_ <= 1.0
    assert all(0.0 <= score <= 1.0 for score in model.prototype_index_.values())


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


def test_auto_null_mode_defaults_to_refit_on_small_data():
    X, y = _separated_regression_data()
    auto = _model(null_mode="auto").fit(X, y)
    refit = _model(null_mode="refit_permutation").fit(X, y)

    assert auto.null_mode_ == "refit_permutation"
    assert auto.auto_null_work_ == X.shape[0] * auto.n_null_permutations
    assert np.isclose(auto.null_loss_, refit.null_loss_, atol=0.0, rtol=0.0)
    assert np.allclose(auto.null_loss_samples_, refit.null_loss_samples_, atol=0.0, rtol=0.0)


def test_auto_null_mode_can_switch_to_fixed_structure():
    X, y = _separated_regression_data()
    model = _model(null_mode="auto", auto_null_work_threshold=1).fit(X, y)

    assert model.null_mode_ == "fixed_structure_permutation"
    assert model.auto_null_work_ == X.shape[0] * model.n_null_permutations
    assert np.isfinite(model.null_loss_)
    assert np.isfinite(model.index)
    assert len(model.null_loss_samples_) == model.n_null_permutations


def test_fixed_structure_null_mode_supports_multivariate_targets():
    X, y = _separated_regression_data()
    Y = np.column_stack([y, y ** 2])
    model = _model(
        null_mode="fixed_structure_permutation",
        target_cover="auto",
        target_distance="auto",
        target_cover_kwargs={"n_init": 10},
        n_projections=8,
    )

    model.fit(X, Y)

    assert model.null_mode_ == "fixed_structure_permutation"
    assert np.isfinite(model.null_loss_)
    assert len(model.null_loss_samples_) == model.n_null_permutations
    assert all(np.isfinite(loss) for loss in model.null_loss_samples_)


def test_soft_topk_adjacency_normalizes_outgoing_weights():
    X, y = _separated_regression_data()
    model = _model(adjacency_mode="soft_topk", top_k=2, feature_temperature=0.5)

    model.fit(X, y)

    outgoing = {}
    for (p, _), value in model.prototype_adjacency_normalized_.items():
        outgoing[p] = outgoing.get(p, 0.0) + float(value)
        assert np.isfinite(value)
        assert value >= 0.0

    assert outgoing
    assert any(
        sum(1 for (src, _dst) in model.prototype_adjacency_normalized_ if src == p) > 1
        for p in outgoing
    )
    for total in outgoing.values():
        assert np.isclose(total, 1.0, atol=1e-12, rtol=0.0)


def test_soft_topk_low_temperature_concentrates_mass_on_dominant_competitor():
    X, y = _separated_regression_data()
    model = _model(adjacency_mode="soft_topk", top_k=2, feature_temperature=1e-6)

    model.fit(X, y)

    by_source = {}
    for (p, q), value in model.prototype_adjacency_normalized_.items():
        by_source.setdefault(p, []).append((q, float(value)))

    assert by_source
    assert any(max(weight for _, weight in edges) > 0.999 for edges in by_source.values())


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
    with pytest.raises(ValueError, match="null_mode must be one of"):
        _model(null_mode="bad").fit(X, y)
    with pytest.raises(ValueError, match="auto_null_work_threshold must be a positive integer"):
        _model(auto_null_work_threshold=0).fit(X, y)
    _model(adjacency_mode="soft_topk").fit(X, y)
