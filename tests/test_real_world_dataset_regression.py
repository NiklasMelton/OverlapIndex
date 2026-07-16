"""Regression tests on small real-world datasets."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from overlapindex import OverlapIndex


EXPECTED_REAL_WORLD_INDEX = {
    "yeast": 0.45851927198966064,
}
YEAST_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
YEAST_DATA_SHA256 = "7cf61776fc04f527f93bf57a327b863893a1225d82df02d457e8950173218258"


def _yeast_data():
    path = Path(__file__).with_name("data_yeast.data")
    if not path.exists():
        pytest.fail(
            "The committed UCI yeast fixture is missing; source: "
            f"{YEAST_DATA_URL}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != YEAST_DATA_SHA256:
        pytest.fail("The committed UCI yeast fixture failed its SHA-256 check.")
    rows = []
    labels = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        rows.append([float(value) for value in parts[1:-1]])
        labels.append(parts[-1])

    X = MinMaxScaler().fit_transform(np.asarray(rows, dtype=float))
    y = np.asarray(labels, dtype=object)
    return X, y


def _make_model():
    return OverlapIndex(
        model_type="MiniBatchKMeans",
        kmeans_k=10,
        kmeans_kwargs={
            "random_state": 0,
            "n_init": 1,
            "batch_size": 64,
            "max_iter": 100,
        },
    )


@pytest.mark.parametrize(
    ("dataset_name", "loader"),
    [("yeast", _yeast_data)],
)
def test_real_world_dataset_index_regression(dataset_name, loader):
    X, y = loader()
    model = _make_model()

    returned = model.add_batch(X, y)

    assert np.isclose(returned, model.index, atol=0.0, rtol=0.0)
    assert np.isclose(
        model.index,
        EXPECTED_REAL_WORLD_INDEX[dataset_name],
        atol=1e-2,
        rtol=0.0,
    ), (
        f"{dataset_name} regression mismatch\n"
        f"expected = {EXPECTED_REAL_WORLD_INDEX[dataset_name]:.17g}\n"
        f"received = {float(model.index):.17g}"
    )
