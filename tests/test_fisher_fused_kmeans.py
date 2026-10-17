"""Focused tests for the opt-in shared-Fisher/FusedKMeans path."""

from copy import deepcopy

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

from overlapindex import OverlapIndex


def _fixture(seed=17, classes=4, rows_per_class=30, features=48):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(classes), rows_per_class)
    X = rng.normal(size=(len(y), features)).astype(np.float32)
    X[:, :classes] += np.eye(classes, dtype=np.float32)[y] * 1.2
    return X, y


def _recommended(rank=16, seed=23, k=6):
    return OverlapIndex(
        model_type="FusedKMeans",
        kmeans_k=k,
        feature_normalization="l2",
        feature_transform="shared_fisher",
        fisher_rank=rank,
        fisher_random_state=seed,
        fused_kmeans_kwargs={"random_state": seed},
    )


def _looped_centers(X, y, *, k, seed, iterations):
    blocks = []
    for position, label in enumerate(dict.fromkeys(y.tolist())):
        rows = np.asarray(X[y == label], dtype=np.float32)
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, position, len(rows), k])
        )
        centers = rows[rng.choice(len(rows), size=k, replace=False)].copy()
        for _ in range(iterations):
            scores = rows @ centers.T
            scores -= np.einsum("ij,ij->i", centers, centers)[None, :] * np.float32(0.5)
            assignments = np.argmax(scores, axis=1)
            for prototype in range(k):
                selected = rows[assignments == prototype]
                if len(selected):
                    centers[prototype] = np.mean(selected, axis=0, dtype=np.float32)
        blocks.append(centers)
    return np.vstack(blocks)


def test_new_defaults_preserve_legacy_behavior_exactly():
    X, y = load_iris(return_X_y=True)
    kwargs = {"random_state": 7}
    legacy = OverlapIndex(kmeans_kwargs=kwargs).fit(X, y)
    explicit = OverlapIndex(
        kmeans_kwargs=kwargs,
        feature_normalization=None,
        feature_transform=None,
        fisher_rank=32,
        fisher_random_state=0,
        fused_kmeans_kwargs=None,
    ).fit(X, y)
    assert explicit.index == legacy.index
    assert np.array_equal(explicit._model.centers, legacy._model.centers)
    assert explicit.fisher_diagnostics_ == {"status": "not_applied"}
    assert explicit.fused_kmeans_diagnostics_ == {"status": "not_applied"}


def test_fused_backend_matches_independent_looped_fixed_lloyd_reference():
    X, y = _fixture(classes=3, rows_per_class=20, features=7)
    model = OverlapIndex(
        model_type="FusedKMeans",
        kmeans_k=4,
        fused_kmeans_kwargs={
            "random_state": 11,
            "n_iter": 3,
            "min_samples_per_prototype": 5,
            "relevance_weighting": False,
        },
    ).fit(X, y)
    expected = _looped_centers(X, y, k=4, seed=11, iterations=3)
    assert np.array_equal(model._model.centers, expected)
    assert np.array_equal(model.feature_weights_, np.ones(X.shape[1]))
    assert model.fused_kmeans_diagnostics_["resolved_k"] == {0: 4, 1: 4, 2: 4}


def test_internal_l2_normalization_matches_external_preprocessing():
    X, y = _fixture(classes=3, rows_per_class=20, features=12)
    normalized = X / np.maximum(
        np.linalg.norm(X, axis=1, keepdims=True),
        np.finfo(np.float32).eps,
    )
    external = OverlapIndex(
        model_type="KMeans",
        kmeans_k=4,
        kmeans_kwargs={"random_state": 5, "n_init": 1},
    ).fit(normalized, y)
    internal = OverlapIndex(
        model_type="KMeans",
        kmeans_k=4,
        kmeans_kwargs={"random_state": 5, "n_init": 1},
        feature_normalization="l2",
    ).fit(X, y)
    assert internal.index == external.index
    assert np.array_equal(internal._normalize_features(X), normalized)
    assert np.allclose(internal._model.centers, external._model.centers, atol=1.0e-7)


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_shared_fisher_is_an_independent_option_for_existing_centroid_backends(
    model_type,
):
    X, y = _fixture(classes=3, rows_per_class=20, features=12)
    kwargs = {"random_state": 4, "n_init": 1}
    model = OverlapIndex(
        model_type=model_type,
        kmeans_k=4,
        kmeans_kwargs=kwargs,
        feature_normalization="l2",
        feature_transform="shared_fisher",
        fisher_rank=0,
    ).fit(X, y)
    assert model.n_features_out_ == 2
    assert model.fisher_diagnostics_["effective_nuisance_rank"] == 0
    assert np.isfinite(model.score_fixed(X, y))


@pytest.mark.parametrize("rank", [0, 1, 8, 16, 32, 100])
def test_integer_fisher_rank_is_configurable_and_capped_safely(rank):
    X, y = _fixture(features=36)
    model = _recommended(rank=rank).fit(X, y)
    expected_effective = min(rank, X.shape[1], len(X) - len(np.unique(y)))
    assert model.feature_transform_.effective_nuisance_rank == expected_effective
    assert model.n_features_in_ == X.shape[1]
    assert model.n_features_out_ == len(np.unique(y)) - 1
    assert np.isfinite(model.index)


def test_full_fisher_is_supported_as_the_configurable_ceiling():
    X, y = _fixture(features=18)
    model = _recommended(rank="full").fit(X, y)
    transform = model.feature_transform_
    assert transform.requested_rank == "full"
    assert transform.output_dimension == len(np.unique(y)) - 1
    assert transform.nuisance_vectors.shape == (X.shape[1], 0)
    assert np.isfinite(transform.transform(model._normalize_features(X))).all()


def test_recommended_port_matches_frozen_rank16_numerical_state():
    X, y = _fixture()
    model = _recommended(rank=16).fit(X, y)
    transform = model.feature_transform_
    normalized = model._normalize_features(X)
    transformed = transform.transform(normalized)

    # Frozen from the experiment-local rank-16 implementation on this fixture.
    assert np.allclose(
        transformed[:2],
        np.asarray(
            [
                [-1.3561574, -2.0508132, -0.41881993],
                [-1.3841604, -2.649764, 1.1783457],
            ],
            dtype=np.float32,
        ),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert model._model.centers.shape == (24, 3)
    assert np.allclose(
        model._model.centers[:2],
        np.asarray(
            [
                [-1.1915044, -0.6825749, -0.543912],
                [-0.09210487, 0.09837444, -0.07450906],
            ],
            dtype=np.float32,
        ),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert np.allclose(
        model.feature_weights_,
        np.asarray([0.84368572, 0.86048585, 1.29582843]),
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    assert model.index == pytest.approx(0.5833333333333333)


def test_score_fixed_and_predict_reuse_learned_state_without_refitting():
    X, y = _fixture()
    train = np.concatenate([np.flatnonzero(y == label)[:20] for label in range(4)])
    holdout = np.concatenate([np.flatnonzero(y == label)[20:] for label in range(4)])
    model = _recommended().fit(X[train], y[train])
    center_before = model._model.centers.copy()
    weights_before = model.feature_weights_.copy()
    transform_before = deepcopy(model.feature_transform_)
    score = model.score_fixed(X[holdout], y[holdout])
    prediction = model.predict(X[holdout])
    assert np.isfinite(score)
    assert prediction.shape == (len(holdout),)
    assert np.array_equal(model._model.centers, center_before)
    assert np.array_equal(model.feature_weights_, weights_before)
    assert np.array_equal(model.feature_transform_.center, transform_before.center)
    assert np.array_equal(
        model.feature_transform_.discriminant_basis,
        transform_before.discriminant_basis,
    )


def test_fused_score_is_memory_budget_invariant():
    X, y = _fixture(features=24)
    small = _recommended(rank=8).set_params(offline_memory_budget_mb=1).fit(X, y)
    large = _recommended(rank=8).set_params(offline_memory_budget_mb=256).fit(X, y)
    assert small.index == large.index
    assert np.array_equal(small._model.centers, large._model.centers)
    assert np.array_equal(small.feature_weights_, large.feature_weights_)


def test_cross_fit_score_matches_explicit_fold_execution():
    X, y = _fixture(rows_per_class=15, features=20)
    model = _recommended(rank=8, seed=31, k=3)
    observed = model.cross_fit_score(X, y, n_splits=5, random_state=31)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=31)
    expected = []
    for fold, (train, holdout) in enumerate(splitter.split(X, y)):
        estimator = clone(model).set_params(
            fisher_random_state=31 + fold,
            fused_kmeans_kwargs={"random_state": 31 + fold},
        )
        estimator.fit(X[train], y[train])
        expected.append(estimator.score_fixed(X[holdout], y[holdout]))
    assert observed == pytest.approx(np.mean(expected))
    assert np.array_equal(model.cross_fit_scores_, expected)
    assert not hasattr(model, "n_features_in_")


def test_cross_fit_preserves_original_string_labels():
    X, y_int = _fixture(classes=3, rows_per_class=18, features=12)
    y = np.asarray([f"class-{value}" for value in y_int], dtype=object)
    score = _recommended(rank=8, k=2).cross_fit_score(
        X,
        y,
        n_splits=3,
        random_state=9,
    )
    assert np.isfinite(score)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"feature_normalization": "standard"}, ValueError, "feature_normalization"),
        ({"feature_transform": "pca"}, ValueError, "feature_transform"),
        ({"fisher_rank": -1}, ValueError, "fisher_rank"),
        ({"fisher_random_state": 1.5}, ValueError, "fisher_random_state"),
        (
            {"model_type": "BallCover", "feature_transform": "shared_fisher"},
            ValueError,
            "shared_fisher",
        ),
        (
            {"model_type": "FusedKMeans", "prototype_refinement": True},
            ValueError,
            "prototype_refinement",
        ),
        (
            {
                "model_type": "FusedKMeans",
                "fused_kmeans_kwargs": {"unknown": 1},
            },
            ValueError,
            "Unknown FusedKMeans options",
        ),
        (
            {
                "model_type": "FusedKMeans",
                "fused_kmeans_kwargs": {"random_state": -1},
            },
            ValueError,
            "nonnegative int",
        ),
    ],
)
def test_new_options_fail_closed(kwargs, error, message):
    with pytest.raises(error, match=message):
        OverlapIndex(**kwargs)


def test_shared_fisher_rejects_multilabel_and_sparse_inputs():
    X, y = _fixture(classes=3, rows_per_class=8, features=10)
    model = _recommended(rank=4, k=2)
    multilabel = [(int(label), int((label + 1) % 3)) for label in y]
    with pytest.raises(ValueError, match="multi-label"):
        model.fit(X, multilabel)

    scipy_sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(TypeError, match="requires dense X"):
        model.fit(scipy_sparse.csr_matrix(X), y)


def test_set_params_clears_fitted_transform_and_backend_state():
    X, y = _fixture(features=20)
    model = _recommended(rank=8).fit(X, y)
    model.set_params(fisher_rank=16)
    assert model.feature_transform_ is None
    assert model.feature_weights_ is None
    assert model.fisher_diagnostics_ == {"status": "not_applied"}
    with pytest.raises(ValueError, match="not fit yet"):
        model.predict(X[:2])


def test_rejected_set_params_preserves_previous_fitted_state():
    X, y = _fixture(features=20)
    model = _recommended(rank=8).fit(X, y)
    centers = model._model.centers.copy()
    prediction = model.predict(X[:5])
    with pytest.raises(ValueError, match="shared_fisher"):
        model.set_params(model_type="BallCover")
    assert model.model_type == "FusedKMeans"
    assert model.fisher_rank == 8
    assert np.array_equal(model._model.centers, centers)
    assert np.array_equal(model.predict(X[:5]), prediction)
