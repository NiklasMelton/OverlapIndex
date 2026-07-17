"""Correctness and robustness regression coverage."""

import pickle

import joblib
import numpy as np
import pytest
from scipy import sparse

from overlapindex import ContinuousOverlapIndex, OverlapIndex
from overlapindex.BallCover import BallCoverManyToOne


KMEANS_KWARGS = {"random_state": 0, "n_init": 10}


def _classification_data():
    X = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=float)
    y = np.asarray([0, 0, 1, 1])
    return X, y


def test_overlap_index_pickle_and_joblib_round_trip(tmp_path):
    unfitted = OverlapIndex(model_type="KMeans", kmeans_k=2, kmeans_kwargs=KMEANS_KWARGS)
    restored_unfitted = pickle.loads(pickle.dumps(unfitted))
    assert restored_unfitted.get_params() == unfitted.get_params()

    X, y = _classification_data()
    fitted = unfitted.fit(X, y)
    path = tmp_path / "overlap.joblib"
    joblib.dump(fitted, path)
    restored = joblib.load(path)

    assert restored.index == fitted.index
    assert restored.under_prototyped_labels_ == fitted.under_prototyped_labels_
    np.testing.assert_array_equal(restored.predict(X), fitted.predict(X))


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_offline_overlap_backends_reject_reset_state_false(model_type):
    X, y = _classification_data()
    model = OverlapIndex(model_type=model_type)
    model.fit(X, y)
    backend = model._model
    score = model.index
    prediction = model.predict(X)

    with pytest.raises(ValueError, match="reset_state=False is supported only for ARTMAP"):
        model.fit_offline(X, y, reset_state=False)

    assert model._model is backend
    assert model.index == score
    np.testing.assert_array_equal(model.predict(X), prediction)


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_offline_full_fit_and_score_rebuild_the_backend(model_type):
    X, y = _classification_data()
    model = OverlapIndex(model_type=model_type)

    initial = model._model
    model.fit(X, y)
    after_fit = model._model
    model.score(X, y)

    assert after_fit is not initial
    assert model._model is not after_fit


def test_continuous_rejects_reset_state_false():
    X, _ = _classification_data()
    y = np.arange(X.shape[0], dtype=float)
    model = ContinuousOverlapIndex(n_null_permutations=1, random_state=0).fit(X, y)
    backend = model._model
    score = model.index
    prediction = model.predict(X)

    with pytest.raises(ValueError, match="reset_state=False is not supported"):
        model.fit_offline(X, y, reset_state=False)

    assert model._model is backend
    assert model.index == score
    np.testing.assert_array_equal(model.predict(X), prediction)


@pytest.mark.parametrize("pair_mode", ["all", "top_m"])
@pytest.mark.parametrize("target_format", ["sequence", "dense", "sparse"])
def test_ballcover_supports_all_multilabel_target_formats(pair_mode, target_format):
    X, _ = _classification_data()
    label_sets = [{0, 1}, {0}, {1}, {0, 1}]
    indicator = np.asarray([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=int)
    if target_format == "sequence":
        Y = label_sets
    elif target_format == "dense":
        Y = indicator
    else:
        Y = sparse.csr_matrix(indicator)

    model = OverlapIndex(
        model_type="BallCover",
        ballcover_k=2,
        ballcover_radius="auto",
        ballcover_kwargs={"metric": "euclidean", "random_state": 0},
        multilabel_pair_mode=pair_mode,
        top_m=1 if pair_mode == "top_m" else None,
    ).fit(X, Y)

    assert np.isfinite(model.index)
    assert model.pairwise_cardinality[(0, 1)] == 1
    assert model.pairwise_cardinality[(1, 0)] == 1


@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.95, 1.0])
def test_ballcover_auto_radius_meets_requested_empirical_coverage(fraction):
    X = np.arange(10, dtype=float).reshape(-1, 1)
    model = BallCoverManyToOne(
        k=1,
        radius="auto",
        cover_fraction=fraction,
        metric="euclidean",
    )
    model.fit_offline(X, np.zeros(X.shape[0], dtype=int))

    assert model.class_diagnostics[0]["covered_fraction"] >= fraction


def test_ballcover_seeded_tie_breaking_is_repeatable_across_fits():
    X = np.asarray([[-1.0], [1.0], [-1.0], [1.0]])
    y = np.zeros(X.shape[0], dtype=int)
    model = BallCoverManyToOne(k=1, radius="auto", random_state=7)

    model.fit_offline(X, y)
    first = model.centers.copy()
    model.fit_offline(X, y)

    np.testing.assert_array_equal(model.centers, first)


def test_ballcover_unseeded_ties_select_the_first_observed_row():
    X = np.asarray([[-1.0], [1.0]])
    model = BallCoverManyToOne(k=1, radius="auto", random_state=None)

    model.fit_offline(X, np.zeros(X.shape[0], dtype=int))

    np.testing.assert_array_equal(model.centers[0], X[0])


@pytest.mark.parametrize(
    "X",
    [
        np.asarray([[0.0]]),
        np.asarray([[0.0], [0.0], [1.0], [1.0]]),
    ],
)
def test_ballcover_auto_radius_coverage_handles_small_samples_and_ties(X):
    model = BallCoverManyToOne(k=1, radius="auto", cover_fraction=0.75)

    model.fit_offline(X, np.zeros(X.shape[0], dtype=int))

    assert model.class_diagnostics[0]["covered_fraction"] >= 0.75


def test_ballcover_raw_cosine_score_matrix_matches_per_row_scoring():
    X = np.asarray([[3.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-2.0, 0.0]])
    y = np.asarray([0, 0, 1, 1])
    model = BallCoverManyToOne(
        k=2,
        radius="auto",
        metric="cosine",
        random_state=0,
    )
    model.fit_offline(X, y)
    query = np.asarray([[10.0, 0.0], [0.0, -5.0], [2.0, 2.0]])

    matrix = model._scores_matrix(query)
    rowwise = np.vstack([model.scores_all(row) for row in query])

    np.testing.assert_allclose(matrix, rowwise)


@pytest.mark.parametrize(
    ("kwargs", "expected_mode"),
    [
        ({"k": "auto", "radius": 0.6}, "auto_k_fixed_radius"),
        ({"k": 2, "radius": "auto"}, "fixed_k_auto_radius"),
        ({"k": 2, "radius": 0.6}, "fixed_k_fixed_radius"),
    ],
)
def test_ballcover_supports_fixed_radius_and_fixed_k_modes(kwargs, expected_mode):
    X = np.asarray([[0.0], [0.5], [1.0], [1.5]])
    model = BallCoverManyToOne(metric="euclidean", **kwargs)

    model.fit_offline(X, np.zeros(X.shape[0], dtype=int))

    assert model.class_diagnostics[0]["mode"] == expected_mode


def test_multilabel_unevaluable_labels_are_nan_and_excluded():
    X = np.asarray([[0.0], [0.1], [1.0], [1.1]])
    Y = [{"A", "B", "C"}, {"A", "B", "C"}, {"B"}, {"C"}]
    model = OverlapIndex(
        model_type="KMeans",
        kmeans_k=2,
        kmeans_kwargs=KMEANS_KWARGS,
    )

    with pytest.warns(RuntimeWarning, match="no evaluable selected competitor"):
        model.fit(X, Y)

    assert model.unevaluable_labels_ == ("A",)
    assert set(model.unevaluable_pairs_) == {("A", "B"), ("A", "C")}
    assert np.isnan(model.singleton_index["A"])
    assert model.pairwise_cardinality[("A", "B")] == 0
    assert np.isnan(model.pairwise_index[("A", "B")])
    assert np.isfinite(model.index)
    assert np.isfinite(model.weighted_index)


def test_multilabel_all_unevaluable_labels_raise():
    X = np.asarray([[0.0], [0.1]], dtype=float)
    Y = [{"A", "B"}, {"A", "B"}]
    model = OverlapIndex(
        model_type="KMeans",
        kmeans_k=2,
        kmeans_kwargs=KMEANS_KWARGS,
    )

    with pytest.warns(RuntimeWarning, match="no evaluable selected competitor"):
        with pytest.raises(ValueError, match="No non-excluded multi-label source labels"):
            model.fit(X, Y)


def test_under_prototyped_labels_warn_but_remain_scored():
    X, y = _classification_data()
    model = OverlapIndex(
        model_type="KMeans",
        kmeans_k=1,
        kmeans_kwargs=KMEANS_KWARGS,
    )

    with pytest.warns(RuntimeWarning, match="fewer than two prototypes") as recorded:
        model.fit(X, y)

    assert len(recorded) == 1
    assert model.under_prototyped_labels_ == (0, 1)
    assert np.isfinite(model.index)


def test_under_prototyped_labels_follow_list_collection_observation_order():
    X = np.asarray([[0.0], [0.1], [1.0]], dtype=float)
    Y = [["z", "a", "z"], ["z"], ["a"]]
    model = OverlapIndex(
        model_type="KMeans",
        kmeans_k=1,
        kmeans_kwargs=KMEANS_KWARGS,
    )

    with pytest.warns(RuntimeWarning, match="fewer than two prototypes"):
        model.fit(X, Y)

    assert model.under_prototyped_labels_ == ("z", "a")
    assert model.cluster_cardinality == {"z": 2, "a": 2}


def test_hand_calculated_balanced_overlap_is_one_half():
    class Backend:
        class_center_id_arrays = {
            0: np.asarray([0, 1]),
            1: np.asarray([2, 3]),
        }

        def bmu_for_class_batch(self, X, Y):
            return np.asarray([0, 1, 2, 3])

        def _scores_matrix(self, X, ids=None):
            scores = np.asarray(
                [
                    [4.0, 3.0, 2.0, 1.0],
                    [2.0, 4.0, 3.0, 1.0],
                    [2.0, 1.0, 4.0, 3.0],
                    [3.0, 1.0, 2.0, 4.0],
                ]
            )
            rows = X[:, 0].astype(int)
            return scores[rows][:, np.asarray(ids, dtype=int)]

    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.asarray([0, 0, 1, 1])
    model = OverlapIndex(model_type="KMeans", kmeans_k=2)
    model._model = Backend()
    model.rev_map.update({0: {0, 1}, 1: {2, 3}})
    model.cluster_cardinality.update({0: 2, 1: 2})

    score = model._fit_offline_centroid_optimized(X, y, np.asarray([0, 1]))

    assert score == 0.5
    assert model.singleton_index[0] == 0.5
    assert model.singleton_index[1] == 0.5


@pytest.mark.parametrize("bad", [1.5, True, "2"])
def test_top_m_rejects_lossy_integer_coercions(bad):
    with pytest.raises(ValueError, match="top_m must be a positive integer"):
        OverlapIndex(multilabel_pair_mode="top_m", top_m=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_kmeans_k_rejects_invalid_cluster_counts(bad):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        OverlapIndex(model_type="KMeans", kmeans_k=bad)


def test_missing_class_specific_k_has_precise_error():
    X, y = _classification_data()
    model = OverlapIndex(
        model_type="KMeans",
        kmeans_k={0: 1},
        kmeans_kwargs=KMEANS_KWARGS,
    )

    with pytest.raises(ValueError, match="Missing class-specific k for class"):
        model.fit(X, y)


@pytest.mark.parametrize(
    "bad_y",
    [
        np.asarray([0.0, 0.0, np.nan, np.nan]),
        np.asarray([0, None, 1, 1], dtype=object),
    ],
)
def test_missing_labels_are_rejected_before_backend_fit(bad_y):
    X, _ = _classification_data()

    with pytest.raises(ValueError, match="Labels must not contain None or NaN"):
        OverlapIndex(model_type="KMeans", kmeans_k=1).fit(X, bad_y)


def test_mixed_hashable_labels_do_not_need_to_sort():
    X, _ = _classification_data()
    y = np.asarray([0, 0, "one", "one"], dtype=object)

    with pytest.warns(RuntimeWarning, match="fewer than two prototypes"):
        model = OverlapIndex(
            model_type="KMeans",
            kmeans_k=1,
            kmeans_kwargs=KMEANS_KWARGS,
        ).fit(X, y)

    assert set(model.singleton_index) == {0, "one"}


def test_rectangular_binary_python_sequence_warns_about_semantics():
    X, _ = _classification_data()
    Y = [[1, 0], [0, 1], [1, 0], [0, 1]]

    # The warning is emitted during normalization; fitting then fails because
    # every row contains both collection labels and no directional pair is evaluable.
    with pytest.warns(UserWarning, match="indicator-matrix semantics"):
        with pytest.warns(RuntimeWarning, match="no evaluable selected competitor"):
            with pytest.raises(ValueError, match="No non-excluded multi-label"):
                OverlapIndex(model_type="KMeans", kmeans_k=2).fit(X, Y)


@pytest.mark.parametrize("constructor", [OverlapIndex, ContinuousOverlapIndex])
def test_zero_feature_inputs_are_rejected(constructor):
    X = np.empty((2, 0))
    y = np.asarray([0, 1])

    with pytest.raises(ValueError, match="at least one feature column"):
        constructor().fit(X, y)


def test_predict_validates_feature_count_for_both_estimators():
    X, y = _classification_data()
    overlap = OverlapIndex(
        model_type="KMeans", kmeans_k=2, kmeans_kwargs=KMEANS_KWARGS
    ).fit(X, y)
    continuous = ContinuousOverlapIndex(
        model_type="KMeans",
        kmeans_k=1,
        kmeans_kwargs=KMEANS_KWARGS,
        n_target_cells=2,
        n_null_permutations=1,
        random_state=0,
    ).fit(X, np.arange(X.shape[0], dtype=float))

    assert overlap.n_features_in_ == continuous.n_features_in_ == 1
    with pytest.raises(ValueError, match="was fit with 1 features"):
        overlap.predict(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="was fit with 1 features"):
        continuous.predict(np.zeros((1, 2)))


def test_continuous_soft_topk_uses_exactly_requested_competitors():
    class Model:
        n_clusters_total = 4

        def topk(self, x, k):
            assert k == 3
            return np.asarray([1, 2, 3]), np.asarray([3.0, 2.0, 1.0])

    model = ContinuousOverlapIndex(top_k=2, adjacency_mode="soft_topk")
    counts, _ = model._adjacency_for_model(
        np.asarray([[0.0]]), np.asarray([0]), Model()
    )

    assert set(counts) == {(0, 1), (0, 2)}


def test_continuous_top_level_random_state_seeds_default_backend():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 4))
    y = X[:, 0] + rng.normal(size=80)
    params = dict(
        n_target_cells=3,
        n_null_permutations=2,
        null_mode="fixed_structure_permutation",
        random_state=7,
    )

    a = ContinuousOverlapIndex(**params).fit(X, y)
    b = ContinuousOverlapIndex(**params).fit(X, y)

    assert a.index == b.index
    np.testing.assert_array_equal(a._model.centers, b._model.centers)


def test_continuous_explicit_backend_seed_takes_precedence():
    model = ContinuousOverlapIndex(
        model_type="KMeans",
        random_state=7,
        kmeans_kwargs={"random_state": 11, "n_init": 1},
    )

    assert model._build_model()._model_kwargs["random_state"] == 11


def test_continuous_single_target_cell_still_supports_prediction():
    X, _ = _classification_data()
    y = np.ones(X.shape[0])
    model = ContinuousOverlapIndex(
        model_type="KMeans",
        kmeans_k=2,
        kmeans_kwargs=KMEANS_KWARGS,
        n_target_cells=2,
        n_null_permutations=1,
    )

    with pytest.warns(RuntimeWarning, match="single target cell"):
        model.fit(X, y)

    assert model.index == 1.0
    assert model.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 1.5}, "top_k must be a positive integer"),
        ({"n_null_permutations": True}, "n_null_permutations must be a positive integer"),
        ({"n_projections": "64"}, "n_projections must be a positive integer"),
        (
            {"auto_null_work_threshold": 1.5},
            "auto_null_work_threshold must be a positive integer",
        ),
        ({"n_target_cells": True}, "n_target_cells must be a positive integer"),
        ({"offline_chunk_size": 2.5}, "offline_chunk_size must be a positive integer"),
        ({"feature_temperature": np.nan}, "feature_temperature must be a finite positive number"),
    ],
)
def test_continuous_strict_parameter_validation(kwargs, message):
    X, _ = _classification_data()

    with pytest.raises(ValueError, match=message):
        ContinuousOverlapIndex(**kwargs).fit(X, np.arange(X.shape[0], dtype=float))


def test_continuous_zero_target_columns_are_rejected():
    X, _ = _classification_data()

    with pytest.raises(ValueError, match="at least one target column"):
        ContinuousOverlapIndex().fit(X, np.empty((X.shape[0], 0)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": np.nan},
        {"k": 1.5, "radius": "auto"},
        {"dtype": np.int32},
        {"chunk_size": 1.5},
    ],
)
def test_ballcover_strict_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        BallCoverManyToOne(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"high_dim_threshold": True},
        {"max_balls": 2.5},
        {"cover_fraction": np.inf},
        {"k": {0: True}, "radius": "auto"},
        {"radius": {0: np.nan}},
    ],
)
def test_ballcover_strict_parameter_validation_additional_cases(kwargs):
    with pytest.raises(ValueError):
        BallCoverManyToOne(**kwargs)


@pytest.mark.parametrize(
    ("k", "radius", "message"),
    [
        ({0: 1}, "auto", "Missing class-specific k for class 1"),
        ("auto", {0: 0.5}, "Missing class-specific radius for class 1"),
    ],
)
def test_ballcover_missing_class_specific_entries_name_the_label(k, radius, message):
    X, y = _classification_data()
    model = BallCoverManyToOne(k=k, radius=radius)

    with pytest.raises(ValueError, match=message):
        model.fit_offline(X, y)


def test_empty_label_collection_is_rejected_precisely():
    X, _ = _classification_data()
    Y = [[0], [], [1], [1]]

    with pytest.raises(ValueError, match="Every sample must have at least one label"):
        OverlapIndex(model_type="KMeans", kmeans_k=1).fit(X, Y)
