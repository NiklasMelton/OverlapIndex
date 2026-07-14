"""Sparse feature-matrix coverage for centroid-backed estimators."""

from importlib.util import find_spec

import numpy as np
import pytest
from scipy import sparse

from overlapindex import ContinuousOverlapIndex, OverlapIndex
from overlapindex.OverlapIndex import _expand_multilabel_for_backend
from overlapindex.clustering import _KMeansManyToOne, _MiniBatchKMeansManyToOne
from overlapindex.utils import _validate_feature_matrix


def _classification_data():
    X = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
        ],
        dtype=float,
    )
    return X, np.asarray([0, 0, 1, 1, 2, 2])


def _overlap_model(model_type):
    kwargs = {"random_state": 0, "n_init": 1}
    if model_type == "MiniBatchKMeans":
        kwargs.update({"batch_size": 6, "max_iter": 20})
    return OverlapIndex(
        model_type=model_type,
        kmeans_k=1,
        kmeans_kwargs=kwargs,
    )


def _continuous_model(model_type, null_mode):
    kwargs = {"random_state": 0, "n_init": 1}
    if model_type == "MiniBatchKMeans":
        kwargs.update({"batch_size": 6, "max_iter": 20})
    return ContinuousOverlapIndex(
        model_type=model_type,
        kmeans_k=1,
        kmeans_kwargs=kwargs,
        n_target_cells=3,
        n_null_permutations=2,
        null_mode=null_mode,
        random_state=0,
    )


def _assert_float_mapping_close(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], abs=1e-7)


@pytest.mark.parametrize("constructor", [sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array])
def test_sparse_validation_normalizes_to_floating_csr_matrix(constructor):
    X, _ = _classification_data()

    validated = _validate_feature_matrix(constructor((X * 10).astype(int)))

    assert sparse.isspmatrix_csr(validated)
    assert np.issubdtype(validated.dtype, np.floating)
    np.testing.assert_array_equal(validated.toarray(), X * 10)


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_integer_sparse_features_fit_without_densifying(model_type):
    X, y = _classification_data()
    X_integer = (X * 10).astype(int)

    dense = _overlap_model(model_type).fit(X_integer, y)
    sparse_fit = _overlap_model(model_type).fit(sparse.csr_matrix(X_integer), y)

    assert sparse_fit.index == pytest.approx(dense.index, abs=1e-7)
    np.testing.assert_array_equal(
        sparse_fit.predict(sparse.csr_matrix(X_integer)), dense.predict(X_integer)
    )


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_overlap_index_sparse_api_matches_dense(model_type):
    X, y = _classification_data()
    X_csr = sparse.csr_matrix(X)

    dense = _overlap_model(model_type).fit(X, y)
    sparse_fit = _overlap_model(model_type).fit(X_csr, y)

    assert sparse_fit.index == pytest.approx(dense.index, abs=1e-7)
    assert sparse_fit.weighted_index == pytest.approx(dense.weighted_index, abs=1e-7)
    _assert_float_mapping_close(sparse_fit.singleton_index, dense.singleton_index)
    np.testing.assert_array_equal(sparse_fit.predict(sparse.csc_matrix(X)), dense.predict(X))

    scored = _overlap_model(model_type)
    assert scored.score(X_csr, y) == pytest.approx(dense.index, abs=1e-7)
    assert scored.partial_fit(X_csr, y) is scored
    np.testing.assert_array_equal(
        _overlap_model(model_type).fit_predict(X_csr, y),
        dense.predict(X),
    )


@pytest.mark.parametrize(
    "backend_type",
    [_KMeansManyToOne, _MiniBatchKMeansManyToOne],
)
def test_centroid_backend_sparse_scoring_matches_dense(backend_type):
    X, y = _classification_data()
    kwargs = {"random_state": 0, "n_init": 1}
    if backend_type is _MiniBatchKMeansManyToOne:
        kwargs["batch_size"] = 6
    model = backend_type(k=1, kmeans_kwargs=kwargs)
    model.fit_offline(sparse.csr_matrix(X), y)

    ids = np.arange(model.n_clusters_total)
    dense_scores = model._scores_matrix(X, ids)
    sparse_scores = model._scores_matrix(sparse.csr_matrix(X), ids)

    np.testing.assert_allclose(sparse_scores, dense_scores, atol=1e-7, rtol=0.0)
    np.testing.assert_array_equal(
        model.bmu_for_class_batch(sparse.csr_matrix(X), y),
        model.bmu_for_class_batch(X, y),
    )
    sparse_ids, sparse_values = model.topk(sparse.csr_matrix(X[:1]), k=2)
    dense_ids, dense_values = model.topk(X[0], k=2)
    np.testing.assert_array_equal(sparse_ids, dense_ids)
    np.testing.assert_allclose(sparse_values, dense_values, atol=1e-7, rtol=0.0)


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
@pytest.mark.parametrize("target_format", ["sequence", "indicator"])
def test_sparse_multilabel_features_match_dense(model_type, target_format):
    X, _ = _classification_data()
    label_sets = [{0}, {0, 1}, {1}, {1, 2}, {2}, {0, 2}]
    if target_format == "indicator":
        Y = sparse.csr_matrix(
            np.asarray([[int(j in labels) for j in range(3)] for labels in label_sets])
        )
    else:
        Y = label_sets

    dense = _overlap_model(model_type).fit(X, Y)
    sparse_fit = _overlap_model(model_type).fit(sparse.csr_matrix(X), Y)

    assert sparse_fit.index == pytest.approx(dense.index, abs=1e-7)
    _assert_float_mapping_close(sparse_fit.pairwise_index, dense.pairwise_index)
    assert dict(sparse_fit.pairwise_cardinality) == dict(dense.pairwise_cardinality)

    expanded, expanded_y = _expand_multilabel_for_backend(
        sparse.csr_matrix(X), label_sets
    )
    assert sparse.isspmatrix_csr(expanded)
    assert expanded.shape[0] == expanded_y.shape[0] == sum(map(len, label_sets))


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
@pytest.mark.parametrize(
    "null_mode", ["fixed_structure_permutation", "refit_permutation"]
)
def test_continuous_sparse_features_match_dense(model_type, null_mode):
    X, _ = _classification_data()
    y = np.asarray([-2.0, -1.8, 0.0, 0.2, 2.0, 2.2])

    dense = _continuous_model(model_type, null_mode).fit(X, y)
    sparse_fit = _continuous_model(model_type, null_mode).fit(
        sparse.csr_matrix(X), y
    )

    assert sparse_fit.index == pytest.approx(dense.index, abs=1e-7)
    assert sparse_fit.actual_loss_ == pytest.approx(dense.actual_loss_, abs=1e-7)
    assert sparse_fit.null_loss_ == pytest.approx(dense.null_loss_, abs=1e-7)
    np.testing.assert_array_equal(
        sparse_fit.predict(sparse.csc_matrix(X)), dense.predict(X)
    )


def test_continuous_refit_permutations_keep_feature_matrix_sparse(monkeypatch):
    X, _ = _classification_data()
    y = np.asarray([-2.0, -1.8, 0.0, 0.2, 2.0, 2.2])
    observed = []
    original = _KMeansManyToOne.fit_offline

    def checked_fit(self, X_fit, Y_fit):
        observed.append(sparse.isspmatrix_csr(X_fit))
        return original(self, X_fit, Y_fit)

    monkeypatch.setattr(_KMeansManyToOne, "fit_offline", checked_fit)
    _continuous_model("KMeans", "refit_permutation").fit(sparse.csc_matrix(X), y)

    assert len(observed) == 3
    assert all(observed)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_sparse_non_finite_features_are_rejected(bad_value):
    X, y = _classification_data()
    X_bad = sparse.csr_matrix(X)
    X_bad.data[0] = bad_value

    with pytest.raises(ValueError, match="X contains NaN or infinite values"):
        _overlap_model("KMeans").fit(X_bad, y)
    with pytest.raises(ValueError, match="X contains NaN or infinite values"):
        _continuous_model("KMeans", "fixed_structure_permutation").fit(X_bad, y)


def test_empty_sparse_features_preserve_empty_input_behavior():
    X = sparse.csr_matrix((0, 4), dtype=float)

    with pytest.warns(RuntimeWarning, match="empty X/Y"):
        assert _overlap_model("KMeans").fit_offline(X, np.asarray([])) == 1.0
    with pytest.warns(RuntimeWarning, match="empty X/Y"):
        assert (
            _continuous_model("KMeans", "fixed_structure_permutation").fit_offline(
                X, np.asarray([])
            )
            == 1.0
        )


def test_ballcover_rejects_sparse_features_explicitly():
    X, y = _classification_data()
    X_csr = sparse.csr_matrix(X)
    message = "Sparse X is supported only for model_type='KMeans' and 'MiniBatchKMeans'"

    with pytest.raises(TypeError, match=message):
        OverlapIndex(model_type="BallCover").fit(X_csr, y)
    with pytest.raises(TypeError, match=message):
        ContinuousOverlapIndex(model_type="BallCover").fit(X_csr, y.astype(float))


@pytest.mark.parametrize("model_type", ["Fuzzy", "Hypersphere"])
def test_continuous_artmap_modes_reject_sparse_features_before_model_validation(
    model_type,
):
    X, y = _classification_data()

    with pytest.raises(
        TypeError,
        match="Sparse X is supported only for model_type='KMeans' and 'MiniBatchKMeans'",
    ):
        ContinuousOverlapIndex(model_type=model_type).fit(
            sparse.csr_matrix(X), y.astype(float)
        )


@pytest.mark.skipif(find_spec("artlib") is None, reason="artlib extra is not installed")
@pytest.mark.parametrize("model_type", ["Fuzzy", "Hypersphere"])
def test_artmap_rejects_sparse_features_explicitly(model_type):
    X, y = _classification_data()
    model = OverlapIndex(model_type=model_type)

    with pytest.raises(
        TypeError,
        match="Sparse X is supported only for model_type='KMeans' and 'MiniBatchKMeans'",
    ):
        model.fit(sparse.csr_matrix(X), y)
