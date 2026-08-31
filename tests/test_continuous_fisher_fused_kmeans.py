"""Focused tests for opt-in continuous Fisher/FusedKMeans geometry."""

from copy import deepcopy
import importlib

import numpy as np
import pytest

from overlapindex import ContinuousOverlapIndex
from overlapindex._fused_kmeans import fit_fused_kmeans


continuous_module = importlib.import_module("overlapindex.ContinuousOverlapIndex")


def _continuous_data(n=120, seed=9):
    rng = np.random.default_rng(seed)
    target = np.linspace(-2.5, 2.5, n)
    nuisance = rng.normal(size=(n, 5))
    nuisance[:, 0] *= 5.0
    X = np.column_stack(
        [
            target,
            np.square(target),
            np.sin(target),
            nuisance,
        ]
    ).astype(np.float32)
    return X, target


def _recommended(**overrides):
    params = {
        "model_type": "FusedKMeans",
        "kmeans_k": 2,
        "fused_kmeans_kwargs": {
            "random_state": 3,
            "n_iter": 3,
            "min_samples_per_prototype": 5,
            "relevance_weighting": True,
            "margin_rows_per_class": 24,
            "weight_floor": 0.05,
        },
        "feature_normalization": "l2",
        "feature_transform": "shared_fisher",
        "fisher_rank": 2,
        "fisher_random_state": 3,
        "n_target_cells": 4,
        "n_null_permutations": 2,
        "null_mode": "refit_permutation",
        "random_state": 3,
    }
    params.update(overrides)
    return ContinuousOverlapIndex(**params)


def test_new_defaults_preserve_legacy_continuous_behavior_exactly():
    X, y = _continuous_data(n=80)
    shared = {
        "model_type": "KMeans",
        "kmeans_k": 2,
        "kmeans_kwargs": {"random_state": 4, "n_init": 2},
        "n_target_cells": 4,
        "n_null_permutations": 2,
        "random_state": 4,
    }
    default = ContinuousOverlapIndex(**shared).fit(X, y)
    explicit = ContinuousOverlapIndex(
        **shared,
        fused_kmeans_kwargs=None,
        feature_normalization=None,
        feature_transform=None,
        fisher_rank=32,
        fisher_random_state=None,
    ).fit(X, y)

    assert default.index == explicit.index
    assert default.null_loss_samples_ == explicit.null_loss_samples_
    np.testing.assert_array_equal(default._model.centers, explicit._model.centers)


def test_target_distance_weighting_changes_relevance_geometry():
    rng = np.random.default_rng(0)
    centers = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    X = np.vstack(
        [rng.normal(center, 0.5, size=(100, 2)) for center in centers]
    ).astype(np.float32)
    labels = np.repeat([0, 1, 2], 100)
    kwargs = {
        "k": 1,
        "random_state": 0,
        "iterations": 3,
        "min_samples_per_prototype": 5,
        "relevance_weighting": True,
        "margin_rows_per_class": 100,
        "weight_floor": 0.05,
    }
    categorical = fit_fused_kmeans(X, labels, **kwargs)
    target_distances = {
        (0, 1): 0.05,
        (1, 0): 0.05,
        (0, 2): 3.0,
        (2, 0): 3.0,
        (1, 2): 2.95,
        (2, 1): 2.95,
    }
    continuous = fit_fused_kmeans(
        X,
        labels,
        label_distances=target_distances,
        **kwargs,
    )

    assert "relevance_mode" not in categorical.diagnostics
    assert continuous.diagnostics["relevance_mode"] == (
        "target_distance_weighted"
    )
    assert continuous.feature_weights[0] > categorical.feature_weights[0]
    assert continuous.feature_weights[1] < categorical.feature_weights[1]


def test_target_distance_mapping_is_closed_and_symmetric():
    X = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1])
    kwargs = {
        "k": 1,
        "random_state": 0,
        "iterations": 1,
        "min_samples_per_prototype": 1,
        "relevance_weighting": True,
        "margin_rows_per_class": 2,
        "weight_floor": 0.05,
    }
    with pytest.raises(ValueError, match="every directed non-diagonal"):
        fit_fused_kmeans(X, labels, label_distances={(0, 1): 1.0}, **kwargs)
    with pytest.raises(ValueError, match="symmetric"):
        fit_fused_kmeans(
            X,
            labels,
            label_distances={(0, 1): 1.0, (1, 0): 2.0},
            **kwargs,
        )


def test_recommended_continuous_stack_records_resolved_geometry():
    X, y = _continuous_data()
    model = _recommended().fit(X, y)

    assert model.n_features_in_ == X.shape[1]
    assert model.n_features_out_ == 3
    assert model.feature_transform_.classes == (0, 1, 2, 3)
    assert model.fisher_diagnostics_["target_semantics"] == "target_cells"
    assert model.fused_kmeans_diagnostics_["relevance_mode"] == (
        "target_distance_weighted"
    )
    assert len(model.target_cell_distances_) == 12
    assert all(value >= 0.0 for value in model.target_cell_distances_.values())
    assert model.feature_weights_.shape == (model.n_features_out_,)
    assert not model.feature_weights_.flags.writeable
    assert np.isfinite(model.index)


def test_shared_fisher_is_independent_of_the_fused_backend():
    X, y = _continuous_data(n=80)
    model = ContinuousOverlapIndex(
        model_type="KMeans",
        kmeans_k=2,
        kmeans_kwargs={"random_state": 2, "n_init": 2},
        feature_normalization="l2",
        feature_transform="shared_fisher",
        fisher_rank=0,
        n_target_cells=4,
        n_null_permutations=1,
        null_mode="refit_permutation",
        random_state=2,
    ).fit(X, y)

    assert model.fisher_diagnostics_["status"] == "applied"
    assert model.feature_transform_.effective_nuisance_rank == 0
    assert model.fused_kmeans_diagnostics_ == {"status": "not_applied"}
    assert model.feature_weights_ is None
    assert np.isfinite(model.index)


def test_multivariate_target_stack_uses_fitted_cover_and_sliced_distance():
    X, y = _continuous_data(n=100)
    targets = np.column_stack([y, np.sin(y)])
    model = _recommended(
        target_cover="kmeans",
        target_cover_kwargs={"random_state": 3, "n_init": 2},
        target_distance="sliced_wasserstein",
        n_projections=8,
        n_null_permutations=1,
    ).fit(X[:75], targets[:75])

    score = model.score_fixed(X[75:], targets[75:])

    assert model.target_cover_model_ is not None
    assert model.target_distance_ == "sliced_wasserstein"
    assert model.fused_kmeans_diagnostics_["relevance_mode"] == (
        "target_distance_weighted"
    )
    assert np.isfinite(score)


def test_auto_refits_target_learned_geometry_and_fixed_structure_is_rejected():
    X, y = _continuous_data(n=80)
    auto = _recommended(
        null_mode="auto",
        auto_null_work_threshold=1,
        n_null_permutations=1,
    ).fit(X, y)
    assert auto.null_mode_ == "refit_permutation"

    with pytest.raises(ValueError, match="cannot calibrate target-learned"):
        _recommended(null_mode="fixed_structure_permutation").fit(X, y)


def test_refit_null_relearns_shared_fisher_for_every_permutation(monkeypatch):
    X, y = _continuous_data(n=80)
    calls = []
    original = continuous_module.fit_shared_fisher

    def recording_fit(values, labels, **kwargs):
        calls.append((len(values), np.asarray(labels).copy()))
        return original(values, labels, **kwargs)

    monkeypatch.setattr(continuous_module, "fit_shared_fisher", recording_fit)
    _recommended(n_null_permutations=3).fit(X, y)

    assert len(calls) == 4
    assert all(row_count == len(X) for row_count, _labels in calls)
    assert any(not np.array_equal(calls[0][1], labels) for _, labels in calls[1:])


def test_score_fixed_uses_fitted_geometry_without_refitting(monkeypatch):
    X, y = _continuous_data()
    model = _recommended().fit(X[:80], y[:80])
    centers = model._model.centers.copy()
    weights = model.feature_weights_.copy()
    transform = deepcopy(model.feature_transform_)
    target_distances = dict(model.target_cell_distances_)

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("score_fixed must not refit the backend")

    monkeypatch.setattr(model._model, "fit_offline", forbidden_fit)
    score = model.score_fixed(X[80:], y[80:])

    assert np.isfinite(score)
    assert model.evaluation_null_mode_ == "fixed_model_permutation"
    np.testing.assert_array_equal(model._model.centers, centers)
    np.testing.assert_array_equal(model.feature_weights_, weights)
    np.testing.assert_array_equal(
        model.feature_transform_.center,
        transform.center,
    )
    np.testing.assert_array_equal(
        model.feature_transform_.discriminant_basis,
        transform.discriminant_basis,
    )
    assert model.target_cell_distances_ == target_distances


def test_predict_applies_fitted_feature_geometry():
    X, y = _continuous_data(n=80)
    model = _recommended(n_null_permutations=1).fit(X, y)
    observed = model.predict(X[:8])
    prepared = model._prep_X(X[:8])
    expected = np.asarray(
        [model._model.topk(row, k=1)[0][0] for row in prepared],
        dtype=int,
    )
    np.testing.assert_array_equal(observed, expected)


def test_cross_fit_is_deterministic_and_leaves_estimator_unfit():
    X, y = _continuous_data(n=96)
    first = _recommended(n_null_permutations=1)
    second = _recommended(n_null_permutations=1)

    score_a = first.cross_fit_score(X, y, n_splits=4, random_state=12)
    score_b = second.cross_fit_score(X, y, n_splits=4, random_state=12)

    assert score_a == score_b
    np.testing.assert_array_equal(first.cross_fit_scores_, second.cross_fit_scores_)
    assert first._model is None
    assert second._model is None
    assert first.feature_transform_ is None


def test_cross_fit_does_not_compute_discarded_training_nulls(monkeypatch):
    X, y = _continuous_data(n=80)

    def forbidden_training_null(*_args, **_kwargs):
        raise AssertionError("cross-fit must not calibrate discarded training scores")

    monkeypatch.setattr(
        ContinuousOverlapIndex,
        "_loss_for_permuted_dataset",
        forbidden_training_null,
    )
    score = _recommended(n_null_permutations=2).cross_fit_score(
        X,
        y,
        n_splits=4,
        random_state=5,
    )
    assert np.isfinite(score)


def test_cross_fitted_stack_orders_clean_noisy_and_random_representations():
    rng = np.random.default_rng(42)
    n_rows = 160
    target = rng.uniform(-2.0, 2.0, n_rows)
    nuisance = rng.normal(size=(n_rows, 12))
    clean = np.column_stack(
        [target, 0.5 * np.square(target), np.sin(2.0 * target), 0.2 * nuisance]
    ).astype(np.float32)
    noisy = np.column_stack(
        [
            target + rng.normal(0.0, 0.8, n_rows),
            rng.normal(size=n_rows),
            nuisance,
        ]
    ).astype(np.float32)
    random = rng.normal(size=clean.shape).astype(np.float32)

    scores = [
        _recommended(
            n_target_cells=5,
            n_null_permutations=2,
        ).cross_fit_score(values, target, n_splits=4, random_state=5)
        for values in (clean, noisy, random)
    ]

    assert scores[0] > scores[1] > scores[2]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"feature_normalization": "standard"}, ValueError, "feature_normalization"),
        ({"feature_transform": "pca"}, ValueError, "feature_transform"),
        ({"fisher_rank": -1}, ValueError, "fisher_rank"),
        ({"fisher_random_state": -1}, ValueError, "fisher_random_state"),
        ({"fused_kmeans_kwargs": "bad"}, TypeError, "fused_kmeans_kwargs"),
        (
            {"model_type": "BallCover", "feature_transform": "shared_fisher"},
            ValueError,
            "shared_fisher",
        ),
    ],
)
def test_continuous_feature_geometry_validation(kwargs, error, message):
    X, y = _continuous_data(n=40)
    with pytest.raises(error, match=message):
        ContinuousOverlapIndex(
            n_target_cells=2,
            n_null_permutations=1,
            **kwargs,
        ).fit(X, y)


def test_continuous_fused_and_shared_fisher_reject_sparse_input():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    X, y = _continuous_data(n=40)
    with pytest.raises(TypeError, match="FusedKMeans"):
        _recommended(n_null_permutations=1).fit(
            scipy_sparse.csr_matrix(X),
            y,
        )
