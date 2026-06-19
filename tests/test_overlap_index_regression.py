"""
Behavior-regression tests for OverlapIndex.
"""

from collections.abc import Iterable
from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import overlapindex.clustering as clustering
overlap_index_module = import_module("overlapindex.OverlapIndex")

try:
    from overlapindex.OverlapIndex import OverlapIndex
except ImportError:  # pragma: no cover - useful when running the file directly from the module folder
    from OverlapIndex import OverlapIndex


ATOL = 1e-2
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



def _make_model(model_type, **overrides):
    if model_type == "KMeans":
        params = dict(
            model_type="KMeans",
            kmeans_k=10,
            kmeans_kwargs={"random_state": 0, "n_init": 10},
        )
        params.update(overrides)
        return OverlapIndex(**params)
    if model_type == "MiniBatchKMeans":
        params = dict(
            model_type="MiniBatchKMeans",
            kmeans_k=10,
            kmeans_kwargs={
                "random_state": 0,
                "n_init": 1,
                "batch_size": 32,
                "max_iter": 100,
            },
        )
        params.update(overrides)
        return OverlapIndex(**params)
    if model_type == "BallCover":
        params = dict(
            model_type="BallCover",
            ballcover_k=20,
            ballcover_radius="auto",
            ballcover_kwargs={
                "metric": "euclidean",
                "cover_fraction": 0.95,
                "random_state": 0,
            },
        )
        params.update(overrides)
        return OverlapIndex(**params)
    params = dict(
        model_type=model_type,
        rho=0.95,
        r_hat=0.1,
    )
    params.update(overrides)
    return OverlapIndex(**params)


def _normalize_excluded_for_test(excluded):
    if excluded is None:
        return set()
    if isinstance(excluded, (str, bytes)) or not isinstance(excluded, Iterable):
        return {excluded}
    return set(excluded)


def _manual_global_scores(model, excluded):
    excluded_set = _normalize_excluded_for_test(excluded)
    included_labels = [
        label for label in model.singleton_index if label not in excluded_set
    ]
    if not included_labels:
        return 1.0, 1.0

    macro = float(
        np.mean([model.singleton_index[label] for label in included_labels])
    )
    total_support = sum(model.cluster_cardinality[label] for label in included_labels)
    if total_support <= 0:
        return macro, macro

    weighted = sum(
        model.singleton_index[label] * model.cluster_cardinality[label]
        for label in included_labels
    ) / total_support
    return macro, float(weighted)


def _assert_float_mapping_close(received, expected):
    assert set(received) == set(expected)
    for key, expected_value in expected.items():
        assert np.isclose(received[key], expected_value, atol=0.0, rtol=0.0)


def _assert_array_mapping_equal(received, expected):
    assert set(received) == set(expected)
    for key, expected_value in expected.items():
        assert np.array_equal(received[key], expected_value)


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


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (np.zeros((2, 2)), np.array([0]), "same number of rows"),
        (np.zeros((2,)), np.array([0, 1]), "X must be a 2D array"),
        (
            np.zeros((2, 2)),
            np.zeros((2, 1, 1)),
            "Y must be a 1D label vector, a 1D sequence of label collections, or a 2D binary indicator matrix.",
        ),
    ],
)
def test_fit_validates_input_shapes(X, y, message):
    model = _make_model("MiniBatchKMeans")

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_fit_rejects_non_finite_values():
    X = np.array([[0.0, 0.0], [np.nan, 1.0]])
    y = np.array([0, 1])
    model = _make_model("MiniBatchKMeans")

    with pytest.raises(ValueError, match="NaN or infinite"):
        model.fit(X, y)


def test_empty_data_warns_and_leaves_default_index():
    X = np.empty((0, 2), dtype=float)
    y = np.array([], dtype=int)
    model = _make_model("MiniBatchKMeans")

    with pytest.warns(RuntimeWarning, match="empty X/Y"):
        returned = model.add_batch(X, y)

    assert returned == 1.0
    assert model.index == 1.0
    assert model.weighted_index == 1.0
    assert dict(model.rev_map) == {}


def test_weighted_index_returns_default_before_fit():
    model = _make_model("MiniBatchKMeans")

    assert model.weighted_index == 1.0


def test_weighted_index_uses_class_supports():
    model = _make_model("MiniBatchKMeans")
    model.singleton_index.update({"minority": 0.0, "majority": 1.0})
    model.cluster_cardinality.update({"minority": 1, "majority": 9})
    model.index = float(np.mean(list(model.singleton_index.values())))

    assert model.index == 0.5
    assert model.weighted_index == 0.9


def test_weighted_index_matches_fitted_support_weighted_formula():
    X, y = _iris_data()
    X = X[y != 2]
    y = y[y != 2]
    X = np.concatenate([X[y == 0][:10], X[y == 1][:40]], axis=0)
    y = np.concatenate([y[y == 0][:10], y[y == 1][:40]], axis=0)
    model = OverlapIndex(
        model_type="MiniBatchKMeans",
        kmeans_k=3,
        kmeans_kwargs={
            "random_state": 0,
            "n_init": 1,
            "batch_size": 16,
            "max_iter": 100,
        },
    )

    model.add_batch(X, y)
    total_support = sum(model.cluster_cardinality[y] for y in model.singleton_index)
    expected = sum(
        model.singleton_index[y] * model.cluster_cardinality[y]
        for y in model.singleton_index
    ) / total_support

    assert total_support == 50
    assert np.isclose(model.weighted_index, expected, atol=0.0, rtol=0.0)


def test_single_class_warns_and_returns_default_index():
    X = np.array([[0.0, 0.0], [0.2, 0.2], [0.4, 0.4]])
    y = np.array([0, 0, 0])
    model = _make_model("MiniBatchKMeans")

    with pytest.warns(RuntimeWarning, match="single class"):
        returned = model.add_batch(X, y)

    assert returned == 1.0
    assert model.index == 1.0


def test_art_backend_still_requires_unit_interval_inputs():
    X = np.array([[0.0, 2.0], [0.5, 0.5]])
    y = np.array([0, 1])
    model = OverlapIndex(model_type="Hypersphere", rho=0.95, r_hat=0.1)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.fit(X, y)


def test_offline_backends_do_not_use_complement_coding(monkeypatch):
    def _boom(X):
        raise AssertionError("complement_code should not be called for offline backends")

    monkeypatch.setattr(overlap_index_module, "complement_code", _boom)
    X, y = _iris_data()
    model = _make_model("MiniBatchKMeans")
    model.fit(X, y)

    assert model._model.centers.shape[1] == X.shape[1]


def test_sklearn_style_score_predict_and_fit_predict():
    X, y = _iris_data()
    model = _make_model("MiniBatchKMeans")

    fit_predict_ids = model.fit_predict(X, y)
    score = model.score()
    predict_ids = model.predict(X[:5])

    assert fit_predict_ids.shape == (X.shape[0],)
    assert predict_ids.shape == (5,)
    assert np.isclose(score, model.index, atol=0.0, rtol=0.0)


def test_predict_raises_before_fit():
    model = _make_model("MiniBatchKMeans")

    with pytest.raises(ValueError, match="not fit yet"):
        model.predict(np.zeros((1, 2), dtype=float))


def test_score_with_data_refits_and_matches_index():
    X, y = _iris_data()
    model = _make_model("MiniBatchKMeans")

    returned = model.score(X, y)

    assert np.isclose(returned, model.index, atol=0.0, rtol=0.0)


def test_get_params_and_set_params_follow_sklearn_conventions():
    model = OverlapIndex(model_type="MiniBatchKMeans", kmeans_k=6)

    params = model.get_params()
    assert params["kmeans_k"] == 6
    assert params["multilabel_pair_mode"] == "all"

    model.set_params(kmeans_k=4, offline_chunk_size=2048)
    assert model.kmeans_k == 4
    assert model.offline_chunk_size == 2048


def test_exclude_classes_get_params_and_set_params_follow_sklearn_conventions():
    model = OverlapIndex(
        model_type="MiniBatchKMeans",
        exclude_classes=[0, "unlabeled"],
    )

    params = model.get_params()
    assert params["exclude_classes"] == [0, "unlabeled"]

    model.set_params(exclude_classes="background")
    assert model.exclude_classes == "background"


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_exclude_classes_only_changes_global_aggregation_single_label(model_type):
    X, y = _iris_data()

    baseline = _make_model(model_type)
    excluded = _make_model(model_type, exclude_classes=0)

    baseline.fit(X, y)
    excluded.fit(X, y)

    _assert_float_mapping_close(
        dict(excluded.singleton_index),
        dict(baseline.singleton_index),
    )
    _assert_float_mapping_close(
        dict(excluded.pairwise_index),
        dict(baseline.pairwise_index),
    )
    assert dict(excluded.cluster_cardinality) == dict(baseline.cluster_cardinality)

    expected_index, expected_weighted = _manual_global_scores(baseline, 0)
    assert np.isclose(excluded.index, expected_index, atol=0.0, rtol=0.0)
    assert np.isclose(excluded.weighted_index, expected_weighted, atol=0.0, rtol=0.0)


def test_empty_and_absent_exclusions_preserve_current_behavior():
    X, y = _iris_data()

    baseline = _make_model("MiniBatchKMeans")
    empty = _make_model("MiniBatchKMeans", exclude_classes=[])
    absent = _make_model(
        "MiniBatchKMeans",
        exclude_classes=[999, "missing"],
    )

    baseline.fit(X, y)
    empty.fit(X, y)
    absent.fit(X, y)

    assert np.isclose(empty.index, baseline.index, atol=0.0, rtol=0.0)
    assert np.isclose(absent.index, baseline.index, atol=0.0, rtol=0.0)
    assert np.isclose(empty.weighted_index, baseline.weighted_index, atol=0.0, rtol=0.0)
    assert np.isclose(absent.weighted_index, baseline.weighted_index, atol=0.0, rtol=0.0)


def test_all_observed_classes_excluded_warns_and_leaves_default_scores():
    X, y = _iris_data()
    model = _make_model("MiniBatchKMeans", exclude_classes=[0, 1, 2])

    with pytest.warns(
        RuntimeWarning,
        match="All observed classes were excluded from global aggregation",
    ):
        returned = model.add_batch(X, y)

    assert returned == 1.0
    assert model.index == 1.0
    assert model.weighted_index == 1.0
    assert set(model.singleton_index) == {0, 1, 2}
    assert dict(model.cluster_cardinality) == {0: 50, 1: 50, 2: 50}


def test_multilabel_sequence_of_same_length_label_lists_is_supported():
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
        ],
        dtype=float,
    )
    y = [["A", "B"], ["A", "C"], ["B", "C"], ["A", "B"]]
    model = OverlapIndex(
        model_type="KMeans",
        kmeans_k=1,
        kmeans_kwargs={"random_state": 0, "n_init": 10},
    )

    returned = model.add_batch(X, y)

    assert np.isclose(returned, model.index, atol=0.0, rtol=0.0)
    assert model.pairwise_cardinality[("A", "B")] == 1
    assert model.pairwise_cardinality[("A", "C")] == 2
    assert model.cluster_cardinality["A"] == 3


def test_multilabel_binary_indicator_matrix_uses_column_labels():
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
        ],
        dtype=float,
    )
    y = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=int,
    )
    model = OverlapIndex(
        model_type="MiniBatchKMeans",
        kmeans_k=1,
        kmeans_kwargs={"random_state": 0, "n_init": 1, "batch_size": 4},
    )

    model.fit(X, y)

    assert set(model.label_to_index_) == {0, 1, 2}
    assert model.pairwise_cardinality[(0, 1)] == 1
    assert model.pairwise_cardinality[(0, 2)] == 2
    assert model.cluster_cardinality[0] == 3


def test_multilabel_top_m_limits_competitors():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ],
        dtype=float,
    )
    y = [
        ["A", "B"],
        ["A"],
        ["B"],
        ["B", "C"],
        ["C"],
        ["A", "C"],
    ]
    model = OverlapIndex(
        model_type="KMeans",
        kmeans_k=1,
        kmeans_kwargs={"random_state": 0, "n_init": 10},
        multilabel_pair_mode="top_m",
        top_m=1,
    )

    model.fit(X, y)

    assert set(model.competitors_) == {"A", "B", "C"}
    for label, competitors in model.competitors_.items():
        assert len(competitors) <= 1
        assert label not in set(competitors)


def test_multilabel_sequence_exclusions_change_only_global_aggregation():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ],
        dtype=float,
    )
    y = [
        ["A", "B"],
        ["A"],
        ["B"],
        ["B", "C"],
        ["C"],
        ["A", "C"],
    ]
    baseline = OverlapIndex(
        model_type="KMeans",
        kmeans_k=1,
        kmeans_kwargs={"random_state": 0, "n_init": 10},
        multilabel_pair_mode="top_m",
        top_m=1,
    )
    excluded = OverlapIndex(
        model_type="KMeans",
        kmeans_k=1,
        kmeans_kwargs={"random_state": 0, "n_init": 10},
        multilabel_pair_mode="top_m",
        top_m=1,
        exclude_classes="B",
    )

    baseline.fit(X, y)
    excluded.fit(X, y)

    _assert_array_mapping_equal(excluded.competitors_, baseline.competitors_)
    _assert_float_mapping_close(
        dict(excluded.singleton_index),
        dict(baseline.singleton_index),
    )
    _assert_float_mapping_close(
        dict(excluded.pairwise_index),
        dict(baseline.pairwise_index),
    )
    assert dict(excluded.pairwise_cardinality) == dict(baseline.pairwise_cardinality)
    assert dict(excluded.cluster_cardinality) == dict(baseline.cluster_cardinality)

    expected_index, expected_weighted = _manual_global_scores(baseline, "B")
    assert np.isclose(excluded.index, expected_index, atol=0.0, rtol=0.0)
    assert np.isclose(excluded.weighted_index, expected_weighted, atol=0.0, rtol=0.0)


def test_multilabel_indicator_exclusions_change_only_global_aggregation():
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
        ],
        dtype=float,
    )
    y = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=int,
    )
    baseline = OverlapIndex(
        model_type="MiniBatchKMeans",
        kmeans_k=1,
        kmeans_kwargs={"random_state": 0, "n_init": 1, "batch_size": 4},
    )
    excluded = OverlapIndex(
        model_type="MiniBatchKMeans",
        kmeans_k=1,
        kmeans_kwargs={"random_state": 0, "n_init": 1, "batch_size": 4},
        exclude_classes=[1, "absent"],
    )

    baseline.fit(X, y)
    excluded.fit(X, y)

    assert excluded.label_to_index_ == baseline.label_to_index_
    _assert_float_mapping_close(
        dict(excluded.singleton_index),
        dict(baseline.singleton_index),
    )
    _assert_float_mapping_close(
        dict(excluded.pairwise_index),
        dict(baseline.pairwise_index),
    )
    assert dict(excluded.pairwise_cardinality) == dict(baseline.pairwise_cardinality)
    assert dict(excluded.cluster_cardinality) == dict(baseline.cluster_cardinality)

    expected_index, expected_weighted = _manual_global_scores(baseline, [1, "absent"])
    assert np.isclose(excluded.index, expected_index, atol=0.0, rtol=0.0)
    assert np.isclose(excluded.weighted_index, expected_weighted, atol=0.0, rtol=0.0)


def test_multilabel_top_m_requires_positive_integer():
    with pytest.raises(ValueError, match="top_m must be a positive integer"):
        OverlapIndex(multilabel_pair_mode="top_m", top_m=0)


@ARTLIB_REQUIRED
@pytest.mark.parametrize("model_type", ["Fuzzy", "Hypersphere"])
def test_art_backends_exclude_classes_only_changes_global_aggregation_after_add_batch(model_type):
    X, y = _iris_data()

    baseline = _make_model(model_type)
    excluded = _make_model(model_type, exclude_classes=0)

    baseline.add_batch(X, y)
    excluded.add_batch(X, y)

    _assert_float_mapping_close(
        dict(excluded.singleton_index),
        dict(baseline.singleton_index),
    )
    _assert_float_mapping_close(
        dict(excluded.pairwise_index),
        dict(baseline.pairwise_index),
    )
    assert dict(excluded.cluster_cardinality) == dict(baseline.cluster_cardinality)

    expected_index, expected_weighted = _manual_global_scores(baseline, 0)
    assert np.isclose(excluded.index, expected_index, atol=0.0, rtol=0.0)
    assert np.isclose(excluded.weighted_index, expected_weighted, atol=0.0, rtol=0.0)


@ARTLIB_REQUIRED
@pytest.mark.parametrize("model_type", ["Fuzzy", "Hypersphere"])
def test_art_backends_exclude_classes_only_changes_global_aggregation_after_add_sample(model_type):
    X, y = _iris_data()

    baseline = _make_model(model_type)
    excluded = _make_model(model_type, exclude_classes=0)

    baseline.add_batch(X[:-ADD_SAMPLE_IDX], y[:-ADD_SAMPLE_IDX])
    excluded.add_batch(X[:-ADD_SAMPLE_IDX], y[:-ADD_SAMPLE_IDX])

    baseline.add_sample(X[ADD_SAMPLE_IDX], int(y[ADD_SAMPLE_IDX]))
    excluded.add_sample(X[ADD_SAMPLE_IDX], int(y[ADD_SAMPLE_IDX]))

    _assert_float_mapping_close(
        dict(excluded.singleton_index),
        dict(baseline.singleton_index),
    )
    _assert_float_mapping_close(
        dict(excluded.pairwise_index),
        dict(baseline.pairwise_index),
    )
    assert dict(excluded.cluster_cardinality) == dict(baseline.cluster_cardinality)

    expected_index, expected_weighted = _manual_global_scores(baseline, 0)
    assert np.isclose(excluded.index, expected_index, atol=0.0, rtol=0.0)
    assert np.isclose(excluded.weighted_index, expected_weighted, atol=0.0, rtol=0.0)
