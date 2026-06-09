"""
Behavior-regression tests for OverlapIndex.
"""

from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import overlapindex.clustering as clustering

try:
    from overlapindex.OverlapIndex import OverlapIndex
except ImportError:  # pragma: no cover - useful when running the file directly from the module folder
    from OverlapIndex import OverlapIndex


ATOL = 1e-12
ARTLIB_AVAILABLE = find_spec("artlib") is not None
ARTLIB_REQUIRED = pytest.mark.skipif(
    not ARTLIB_AVAILABLE,
    reason="artlib extra is not installed",
)

ADD_SAMPLE_IDX = 20

# Placeholder values. Replace these after the first pytest run.
EXPECTED_ADD_BATCH_INDEX = {
    "Fuzzy": 0.9266666666666667,
    "Hypersphere": 0.9333333333333332,
    "KMeans": 0.9266666666666666,
    "MiniBatchKMeans": 0.9133333333333332,
    "BallCover": 0.86,
}

# KMeans.add_sample is expected to raise, so only online ARTMAP-style backends
# are included here.
EXPECTED_ADD_SAMPLE_AFTER_BATCH_INDEX = {
    "Fuzzy": 0.9155555555555556,
    "Hypersphere": 0.9155555555555556,
}


def _iris_data():
    """Return a deterministic, complement-coding-safe iris dataset."""
    X, y = load_iris(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)
    return X.astype(float), y.astype(int)



def _make_model(model_type):
    if model_type == "KMeans":
        return OverlapIndex(
            model_type="KMeans",
            kmeans_k=10,
            kmeans_kwargs={"random_state": 0, "n_init": 10},
        )
    if model_type == "MiniBatchKMeans":
        return OverlapIndex(
            model_type="MiniBatchKMeans",
            kmeans_k=10,
            kmeans_kwargs={
                "random_state": 0,
                "n_init": 1,
                "batch_size": 32,
                "max_iter": 100,
            },
        )
    if model_type == "BallCover":
        return OverlapIndex(
            model_type="BallCover",
            ballcover_k=20,
            ballcover_radius="auto",
            ballcover_kwargs={
                "metric": "euclidean",
                "cover_fraction": 0.95,
                "random_state": 0,
            },
        )
    return OverlapIndex(
        model_type=model_type,
        rho=0.95,
        r_hat=0.1,
    )


def _assert_index_close(received, expected, context):
    received = float(received)
    expected = float(expected)
    assert np.isclose(received, expected, atol=ATOL, rtol=0.0), (
        f"{context} index regression mismatch\n"
        f"expected = {expected:.17g}\n"
        f"received = {received:.17g}\n"
        f"absolute difference = {abs(received - expected):.17g}"
    )


def _assert_return_matches_self_index(model, returned, context):
    returned = float(returned)
    current = float(model.index)
    assert np.isclose(returned, current, atol=0.0, rtol=0.0), (
        f"{context} return value does not match model.index\n"
        f"returned = {returned:.17g}\n"
        f"model.index = {current:.17g}"
    )


@ARTLIB_REQUIRED
@pytest.mark.parametrize("model_type", ["Fuzzy", "Hypersphere"])
def test_art_backends_add_batch_index_regression(model_type):
    X, y = _iris_data()

    model = _make_model(model_type)
    returned = model.add_batch(X, y)

    _assert_return_matches_self_index(model, returned, f"{model_type}.add_batch")
    _assert_index_close(
        model.index,
        EXPECTED_ADD_BATCH_INDEX[model_type],
        f"{model_type}.add_batch",
    )


@pytest.mark.parametrize("model_type", ["KMeans", "BallCover"])
def test_add_batch_index_regression(model_type):
    X, y = _iris_data()

    model = _make_model(model_type)
    returned = model.add_batch(X, y)

    _assert_return_matches_self_index(model, returned, f"{model_type}.add_batch")
    _assert_index_close(
        model.index,
        EXPECTED_ADD_BATCH_INDEX[model_type],
        f"{model_type}.add_batch",
    )


def test_minibatch_kmeans_add_batch_index_regression():
    X, y = _iris_data()

    model = _make_model("MiniBatchKMeans")
    returned = model.add_batch(X, y)

    _assert_return_matches_self_index(model, returned, "MiniBatchKMeans.add_batch")
    _assert_index_close(
        model.index,
        EXPECTED_ADD_BATCH_INDEX["MiniBatchKMeans"],
        "MiniBatchKMeans.add_batch",
    )


@ARTLIB_REQUIRED
@pytest.mark.parametrize("model_type", ["Fuzzy", "Hypersphere"])
def test_add_sample_after_batch_index_regression(model_type):
    X, y = _iris_data()

    model = _make_model(model_type)
    model.add_batch(X[:-ADD_SAMPLE_IDX], y[:-ADD_SAMPLE_IDX])
    returned = model.add_sample(X[ADD_SAMPLE_IDX], int(y[ADD_SAMPLE_IDX]))

    _assert_return_matches_self_index(model, returned, f"{model_type}.add_sample")
    _assert_index_close(
        model.index,
        EXPECTED_ADD_SAMPLE_AFTER_BATCH_INDEX[model_type],
        f"{model_type}.add_sample after add_batch",
    )


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_offline_backends_add_sample_raises_not_implemented(model_type):
    X, y = _iris_data()
    model = _make_model(model_type)

    with pytest.raises(NotImplementedError, match="offline-only"):
        model.add_sample(X[0], int(y[0]))


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_offline_backends_module_a_accessor_raises_attribute_error(model_type):
    model = _make_model(model_type)

    with pytest.raises(AttributeError, match="ARTMAP backends"):
        _ = model.module_a


def test_offline_backend_does_not_require_artlib(monkeypatch):
    def _boom():
        raise AssertionError("ART loader should not be called for offline backends")

    monkeypatch.setattr(clustering, "_load_artmap_classes", _boom)
    model = OverlapIndex(model_type="MiniBatchKMeans")
    assert model.model_type == "MiniBatchKMeans"


def test_art_backend_raises_helpful_error_without_artlib(monkeypatch):
    real_import_module = import_module

    def _missing(name, package=None):
        if name == "artlib":
            raise ImportError("No module named 'artlib'")
        return real_import_module(name, package)

    monkeypatch.setattr(clustering, "import_module", _missing)
    with pytest.raises(ImportError, match=r"overlapindex\[art\]"):
        OverlapIndex(model_type="Fuzzy")
