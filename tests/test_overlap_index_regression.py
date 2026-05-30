"""
Behavior-regression tests for OverlapIndex.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

try:
    from overlapindex.OverlapIndex import OverlapIndex
except ImportError:  # pragma: no cover - useful when running the file directly from the module folder
    from OverlapIndex import OverlapIndex


ATOL = 1e-12

# Placeholder values. Replace these after the first pytest run.
EXPECTED_ADD_BATCH_INDEX = {
    "Fuzzy": 0.82,
    "Hypersphere": 0.80,
    "KMeans": 0.84,
}

# KMeans.add_sample is expected to raise, so only online ARTMAP-style backends
# are included here.
EXPECTED_ADD_SAMPLE_AFTER_BATCH_INDEX = {
    "Fuzzy": 0.81,
    "Hypersphere": 0.79,
}


def _iris_data():
    """Return a deterministic, complement-coding-safe iris dataset."""
    X, y = load_iris(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)
    return X.astype(float), y.astype(int)


def _balanced_indices(y, per_class, offset=0):
    """Select the same number of samples from each class."""
    selected = []
    for cls in np.unique(y):
        cls_indices = np.flatnonzero(y == cls)
        selected.extend(cls_indices[offset : offset + per_class])
    return np.asarray(selected, dtype=int)


def _make_model(model_type):
    if model_type == "KMeans":
        return OverlapIndex(
            model_type="KMeans",
            kmeans_k=10,
            kmeans_kwargs={"random_state": 0, "n_init": 10},
        )

    return OverlapIndex(
        model_type=model_type,
        rho=0.75,
        r_hat=np.inf,
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


@pytest.mark.parametrize("model_type", ["Fuzzy", "Hypersphere", "KMeans"])
def test_add_batch_index_regression(model_type):
    X, y = _iris_data()
    batch_idx = _balanced_indices(y, per_class=12, offset=0)

    model = _make_model(model_type)
    returned = model.add_batch(X[batch_idx], y[batch_idx])

    _assert_return_matches_self_index(model, returned, f"{model_type}.add_batch")
    _assert_index_close(
        model.index,
        EXPECTED_ADD_BATCH_INDEX[model_type],
        f"{model_type}.add_batch",
    )


@pytest.mark.parametrize("model_type", ["Fuzzy", "Hypersphere"])
def test_add_sample_after_batch_index_regression(model_type):
    X, y = _iris_data()
    batch_idx = _balanced_indices(y, per_class=10, offset=0)
    sample_idx = int(_balanced_indices(y, per_class=1, offset=10)[0])

    model = _make_model(model_type)
    model.add_batch(X[batch_idx], y[batch_idx])
    returned = model.add_sample(X[sample_idx], int(y[sample_idx]))

    _assert_return_matches_self_index(model, returned, f"{model_type}.add_sample")
    _assert_index_close(
        model.index,
        EXPECTED_ADD_SAMPLE_AFTER_BATCH_INDEX[model_type],
        f"{model_type}.add_sample after add_batch",
    )


def test_kmeans_add_sample_raises_not_implemented():
    X, y = _iris_data()
    model = _make_model("KMeans")

    with pytest.raises(NotImplementedError, match="offline-only"):
        model.add_sample(X[0], int(y[0]))


def test_kmeans_module_a_accessor_raises_attribute_error():
    model = _make_model("KMeans")

    with pytest.raises(AttributeError, match="ARTMAP backends"):
        _ = model.module_a
