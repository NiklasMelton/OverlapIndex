"""Regression coverage for the backend-neutral offline scorer.

The scorer is intentionally exercised through the public ``OverlapIndex`` API.
The tests compare an unconstrained reference run with small row and scratch
budgets so that changes to tiling do not change any public score or diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
import pickle

import joblib
import numpy as np
import pytest
from scipy import sparse

from overlapindex import OverlapIndex
import overlapindex.clustering as clustering


def _single_label_data():
    """Three compact classes with enough rows for two prototypes each."""
    X = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [2.0, 2.0],
            [2.1, 2.0],
            [2.0, 2.1],
            [0.0, 2.0],
            [0.1, 2.0],
            [0.0, 2.1],
        ],
        dtype=float,
    )
    y = np.repeat(np.arange(3), 3)
    return X, y


def _multilabel_data():
    """Feature rows and indicator labels with every directional pair evaluable."""
    X, _ = _single_label_data()
    indicator = np.asarray(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.int8,
    )
    sequence = [
        tuple(np.flatnonzero(row).tolist())
        for row in indicator
    ]
    return X, sequence, indicator


def _paired_multilabel_data(n_groups):
    """Return separated two-row groups, each carrying a label pair.

    Every label owns two nearby samples, while the two labels in one group
    always co-occur.  Groups are far enough apart that no non-cooccurring
    target can beat the source's second prototype.
    """
    X = []
    labels = []
    for group in range(int(n_groups)):
        primary = f"p{group}"
        auxiliary = f"q{group}"
        base = float(group * 10)
        for offset in (0.0, 0.1):
            X.append([base + offset])
            labels.append({primary, auxiliary})
    return np.asarray(X, dtype=float), labels


def _model_kwargs(model_type, **overrides):
    if model_type == "KMeans":
        params = {
            "model_type": model_type,
            "kmeans_k": 2,
            "kmeans_kwargs": {"random_state": 0, "n_init": 1},
        }
    elif model_type == "MiniBatchKMeans":
        params = {
            "model_type": model_type,
            "kmeans_k": 2,
            "kmeans_kwargs": {
                "random_state": 0,
                "n_init": 1,
                "batch_size": 8,
                "max_iter": 100,
            },
        }
    elif model_type == "BallCover":
        params = {
            "model_type": model_type,
            "ballcover_k": 2,
            "ballcover_radius": "auto",
            "ballcover_kwargs": {
                "metric": "euclidean",
                "cover_fraction": 0.95,
                "random_state": 0,
            },
        }
    else:  # pragma: no cover - kept local to make failures obvious
        raise AssertionError(model_type)
    params.update(overrides)
    return params


def _assert_mapping_equal(left, right):
    """Compare scalar mappings while treating paired NaNs as equal."""
    assert set(left) == set(right)
    for key in left:
        lhs = left[key]
        rhs = right[key]
        if isinstance(lhs, (float, np.floating)) and isinstance(rhs, (float, np.floating)):
            if np.isnan(lhs) and np.isnan(rhs):
                continue
        assert lhs == rhs, (key, lhs, rhs)


def _assert_fit_outputs_equal(reference, tiled):
    assert tiled.index == reference.index
    assert tiled.weighted_index == reference.weighted_index
    _assert_mapping_equal(tiled.singleton_index, reference.singleton_index)
    _assert_mapping_equal(tiled.pairwise_index, reference.pairwise_index)
    _assert_mapping_equal(tiled.pairwise_cardinality, reference.pairwise_cardinality)
    _assert_mapping_equal(tiled.cluster_cardinality, reference.cluster_cardinality)
    assert tiled.under_prototyped_labels_ == reference.under_prototyped_labels_
    assert tiled.unevaluable_labels_ == reference.unevaluable_labels_
    assert tiled.unevaluable_pairs_ == reference.unevaluable_pairs_
    assert set(tiled.competitors_) == set(reference.competitors_)
    for label in reference.competitors_:
        np.testing.assert_array_equal(
            tiled.competitors_[label], reference.competitors_[label]
        )


def _bruteforce_pair_reference(model, X, label_sets, *, top_m=False):
    """Compute directional overlap by explicit per-row prototype ranking.

    This intentionally uses the backend's scalar ``scores_all`` and
    ``bmu_for_class`` hooks rather than the universal scorer's block methods.
    A target overlaps only when its best score is strictly greater than the
    source's second-best score.  Equality at that threshold is source-owned.
    """
    classes = list(model.rev_map)
    normalized_sets = [set(labels) for labels in label_sets]
    expected = {}
    expected_cardinality = {}

    for source in classes:
        own_ids = set(int(value) for value in model.rev_map[source])
        selected_competitors = (
            model.competitors_.get(source, [])
            if top_m
            else [other for other in classes if other != source]
        )
        for competitor in selected_competitors:
            competitor_ids = set(int(value) for value in model.rev_map[competitor])
            rows = [
                row
                for row, labels in enumerate(normalized_sets)
                if source in labels and competitor not in labels
            ]
            expected_cardinality[(source, competitor)] = len(rows)
            if not rows:
                expected[(source, competitor)] = np.nan
                continue

            overlap_count = 0
            for row in rows:
                x = np.asarray(X[row : row + 1])
                scores = np.asarray(model._model.scores_all(x[0]), dtype=float)
                source_scores = np.asarray(
                    [scores[prototype_id] for prototype_id in sorted(own_ids)],
                    dtype=float,
                )
                if source_scores.size <= 1:
                    second_source = -np.inf
                else:
                    second_source = float(np.partition(source_scores, -2)[-2])
                target_best = max(scores[prototype_id] for prototype_id in competitor_ids)
                overlap_count += int(target_best > second_source)
            expected[(source, competitor)] = 1.0 - (
                float(overlap_count) / float(len(rows))
            )

    return expected, expected_cardinality


def _fit_reference_and_tiled(model_type, X, y, **kwargs):
    reference = OverlapIndex(
        **_model_kwargs(
            model_type,
            offline_memory_budget_mb=256,
            offline_chunk_size=None,
            **kwargs,
        )
    ).fit(X, y)
    tiled = OverlapIndex(
        **_model_kwargs(
            model_type,
            offline_memory_budget_mb=1,
            offline_chunk_size=1,
            **kwargs,
        )
    ).fit(X, y)
    return reference, tiled


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "256", None])
def test_offline_memory_budget_requires_a_positive_genuine_integer(bad):
    with pytest.raises(ValueError, match="offline_memory_budget_mb must be a positive integer"):
        OverlapIndex(offline_memory_budget_mb=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_offline_chunk_size_retains_positive_integer_validation(bad):
    with pytest.raises(ValueError, match="offline_chunk_size must be a positive integer"):
        OverlapIndex(offline_chunk_size=bad)


def test_offline_memory_budget_is_exposed_by_sklearn_parameter_api():
    model = OverlapIndex(offline_memory_budget_mb=64)
    assert model.get_params()["offline_memory_budget_mb"] == 64

    model.set_params(offline_memory_budget_mb=17)
    assert model.offline_memory_budget_mb == 17
    assert model.get_params()["offline_memory_budget_mb"] == 17


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_offline_chunk_size_caps_rows_seen_by_score_tiles(monkeypatch, model_type):
    observed_rows = []

    def _record_rows(original):
        def wrapped(self, X_prepared, ids=None):
            observed_rows.append(int(X_prepared.shape[0]))
            return original(self, X_prepared, ids)

        return wrapped

    monkeypatch.setattr(
        clustering._BaseCentroidManyToOne,
        "score_block_prepared",
        _record_rows(clustering._BaseCentroidManyToOne.score_block_prepared),
    )
    monkeypatch.setattr(
        clustering._BallCoverManyToOne,
        "score_block_prepared",
        _record_rows(clustering._BallCoverManyToOne.score_block_prepared),
    )

    X, y = _single_label_data()
    model = OverlapIndex(
        **_model_kwargs(
            model_type,
            offline_chunk_size=2,
            offline_memory_budget_mb=1,
        )
    )
    model.fit(X, y)

    assert observed_rows
    assert max(observed_rows) <= 2


def test_generous_budget_packs_all_target_classes_into_one_score_call(monkeypatch):
    X, labels = _paired_multilabel_data(4)
    calls = []
    original = clustering._BaseCentroidManyToOne.score_block_prepared

    def _record_call(self, X_prepared, ids=None):
        ids_array = None if ids is None else np.asarray(ids, dtype=int)
        calls.append(
            (
                int(X_prepared.shape[0]),
                None if ids_array is None else ids_array.copy(),
            )
        )
        return original(self, X_prepared, ids)

    monkeypatch.setattr(
        clustering._BaseCentroidManyToOne,
        "score_block_prepared",
        _record_call,
    )
    model = OverlapIndex(
        model_type="KMeans",
        kmeans_k=2,
        kmeans_kwargs={"random_state": 0, "n_init": 1},
        offline_memory_budget_mb=256,
        offline_chunk_size=None,
    ).fit(X, labels)

    total_prototypes = int(model._model.n_clusters_total)
    packed = [ids for _rows, ids in calls if ids is not None and ids.size == total_prototypes]
    assert packed, "target classes should share at least one packed score call"
    np.testing.assert_array_equal(
        np.sort(packed[0]),
        np.arange(total_prototypes, dtype=int),
    )


def test_all_mode_high_class_diagnostics_remain_lazy_and_sparse():
    # 240 labels would imply 57,360 directional entries if all competitors
    # were eagerly materialized.  The fitted state stores only O(C) class
    # metadata and no default pair scores/cardinalities.
    X, labels = _paired_multilabel_data(120)
    model = OverlapIndex(
        model_type="KMeans",
        kmeans_k=2,
        kmeans_kwargs={"random_state": 0, "n_init": 1},
        offline_memory_budget_mb=256,
        offline_chunk_size=None,
        multilabel_pair_mode="all",
    ).fit(X, labels)

    n_classes = 240
    assert len(model.competitors_) == n_classes
    assert isinstance(model.competitors_, Mapping)
    first_label = "p0"
    assert len(model.competitors_[first_label]) == n_classes - 1
    assert len(model.pairwise_index) == 0
    assert len(model.pairwise_cardinality) == 0
    assert len(model._pairwise_hits) == 0
    assert model.index == 1.0


def test_high_class_unevaluable_pairs_view_is_exact_and_pickle_safe():
    # Each pair of labels always co-occurs, so exactly the partner direction
    # for each label has a zero denominator.  This exceeds the old eager
    # diagnostic threshold without requiring a quadratic test fixture.
    X, labels = _paired_multilabel_data(40)
    model = OverlapIndex(
        model_type="KMeans",
        kmeans_k=2,
        kmeans_kwargs={"random_state": 0, "n_init": 1},
        multilabel_pair_mode="all",
    ).fit(X, labels)

    expected = {
        pair
        for group in range(40)
        for pair in ((f"p{group}", f"q{group}"), (f"q{group}", f"p{group}"))
    }
    diagnostics = model.unevaluable_pairs_
    assert not isinstance(diagnostics, tuple)
    assert len(diagnostics) == len(expected)
    assert set(diagnostics) == expected
    assert ("p0", "q0") in diagnostics
    assert ("q0", "p0") in diagnostics
    assert ("p0", "p1") not in diagnostics
    assert len(model.pairwise_index) == 0
    assert len(model.pairwise_cardinality) == 0

    restored = pickle.loads(pickle.dumps(model))
    assert restored.unevaluable_pairs_ == diagnostics
    assert set(restored.unevaluable_pairs_) == expected
    assert ("p0", "q0") in restored.unevaluable_pairs_


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_single_label_tiling_and_scratch_budget_are_exactly_equivalent(model_type):
    X, y = _single_label_data()
    reference, tiled = _fit_reference_and_tiled(model_type, X, y)
    _assert_fit_outputs_equal(reference, tiled)
    np.testing.assert_array_equal(tiled.predict(X), reference.predict(X))


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
@pytest.mark.parametrize("target_format", ["sequence", "dense", "sparse"])
def test_multilabel_tiling_is_equivalent_for_supported_targets(model_type, target_format):
    X, sequence, indicator = _multilabel_data()
    if target_format == "sequence":
        target = sequence
    elif target_format == "dense":
        target = indicator
    else:
        target = sparse.csr_matrix(indicator)

    reference, tiled = _fit_reference_and_tiled(model_type, X, target)
    _assert_fit_outputs_equal(reference, tiled)


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_supported_sparse_feature_inputs_match_dense_tiled_scores(model_type):
    X, y = _single_label_data()
    dense, tiled_dense = _fit_reference_and_tiled(model_type, X, y)
    sparse_reference, sparse_tiled = _fit_reference_and_tiled(
        model_type,
        sparse.csr_matrix(X),
        y,
    )

    _assert_fit_outputs_equal(dense, tiled_dense)
    _assert_fit_outputs_equal(sparse_reference, sparse_tiled)
    np.testing.assert_allclose(sparse_reference.index, dense.index, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(
        sparse_reference.weighted_index,
        dense.weighted_index,
        atol=1e-7,
        rtol=0.0,
    )


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_single_label_scores_match_independent_scalar_pair_reference(model_type):
    X, y = _single_label_data()
    model = OverlapIndex(
        **_model_kwargs(
            model_type,
            offline_memory_budget_mb=1,
            offline_chunk_size=1,
        )
    ).fit(X, y)
    expected, expected_cardinality = _bruteforce_pair_reference(
        model,
        X,
        [{label} for label in y],
    )

    for pair, value in expected.items():
        received = model.pairwise_index[pair]
        if np.isnan(value):
            assert np.isnan(received)
        else:
            assert received == value
        assert model.pairwise_cardinality[pair] == expected_cardinality[pair]
    for source in model.rev_map:
        source_scores = [
            expected[(source, competitor)]
            for competitor in model.rev_map
            if competitor != source
        ]
        assert model.singleton_index[source] == min(source_scores)


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
@pytest.mark.parametrize("pair_mode", ["all", "top_m"])
def test_multilabel_scores_match_independent_scalar_pair_reference(model_type, pair_mode):
    X, sequence, _ = _multilabel_data()
    kwargs = {
        "multilabel_pair_mode": pair_mode,
        "top_m": 1 if pair_mode == "top_m" else None,
        "offline_memory_budget_mb": 1,
        "offline_chunk_size": 1,
    }
    model = OverlapIndex(**_model_kwargs(model_type, **kwargs)).fit(X, sequence)
    expected, expected_cardinality = _bruteforce_pair_reference(
        model,
        X,
        sequence,
        top_m=pair_mode == "top_m",
    )

    for pair, value in expected.items():
        received = model.pairwise_index[pair]
        if np.isnan(value):
            assert np.isnan(received)
        else:
            assert received == value
        assert model.pairwise_cardinality[pair] == expected_cardinality[pair]
    for source in model.rev_map:
        source_scores = [
            value
            for (label, _), value in expected.items()
            if label == source and not np.isnan(value)
        ]
        if source_scores:
            assert model.singleton_index[source] == min(source_scores)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_ballcover_unequal_k_and_radii_are_budget_invariant(metric):
    X = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.8, 0.6],
            [0.0, 1.0],
            [0.1, 0.99],
            [0.6, 0.8],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 0, 1, 1, 1])
    kwargs = {
        "ballcover_k": {0: 2, 1: 1},
        "ballcover_radius": {0: 0.35, 1: 0.8},
        "ballcover_kwargs": {"metric": metric, "random_state": 0},
    }
    reference, tiled = _fit_reference_and_tiled("BallCover", X, y, **kwargs)
    _assert_fit_outputs_equal(reference, tiled)


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_source_favoring_ties_with_two_prototypes_remain_separated(model_type):
    # The co-occurring x=2 row supplies each label's second prototype but is
    # excluded from both directional denominators.  For the valid x=0/x=4
    # rows, each best target score equals the second source score exactly;
    # strict ``target > second_source`` therefore yields no overlap.
    X = np.asarray([[0.0], [2.0], [4.0]])
    labels = [{"A"}, {"A", "B"}, {"B"}]
    kwargs = (
        {"kmeans_k": 2}
        if model_type != "BallCover"
        else {"ballcover_k": 2, "ballcover_radius": 1.0}
    )
    model = OverlapIndex(**_model_kwargs(model_type, **kwargs))
    model.fit(X, labels)

    assert model.index == 1.0
    assert model.pairwise_index[("A", "B")] == 1.0
    assert model.pairwise_index[("B", "A")] == 1.0


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans", "BallCover"])
def test_one_prototype_per_source_retains_documented_degenerate_zero_score(model_type):
    X = np.asarray([[0.0], [1.0], [0.0], [1.0]])
    y = np.asarray([0, 0, 1, 1])
    kwargs = (
        {"kmeans_k": 1}
        if model_type != "BallCover"
        else {"ballcover_k": 1, "ballcover_radius": 1.0}
    )
    model = OverlapIndex(**_model_kwargs(model_type, **kwargs))
    with pytest.warns(RuntimeWarning, match="fewer than two prototypes"):
        model.fit(X, y)

    assert model.index == 0.0
    assert model.pairwise_index[(0, 1)] == 0.0
    assert model.pairwise_index[(1, 0)] == 0.0


def test_multilabel_top_m_and_exclusions_are_budget_invariant():
    X, sequence, _ = _multilabel_data()
    kwargs = {
        "multilabel_pair_mode": "top_m",
        "top_m": 1,
        "exclude_classes": 1,
    }
    for model_type in ["KMeans", "MiniBatchKMeans", "BallCover"]:
        reference, tiled = _fit_reference_and_tiled(model_type, X, sequence, **kwargs)
        _assert_fit_outputs_equal(reference, tiled)


def test_unevaluable_sources_remain_nan_and_are_excluded_after_tiling():
    X = np.asarray([[0.0], [0.1], [1.0], [1.1], [2.0], [2.1]])
    labels = [
        {"A", "B", "C"},
        {"A", "B", "C"},
        {"B"},
        {"B"},
        {"C"},
        {"C"},
    ]
    reference, tiled = _fit_reference_and_tiled(
        "KMeans",
        X,
        labels,
        kmeans_k=2,
    )
    _assert_fit_outputs_equal(reference, tiled)
    assert "A" in tiled.unevaluable_labels_
    assert np.isnan(tiled.singleton_index["A"])
    assert np.isfinite(tiled.index)
    assert tiled.pairwise_cardinality[("A", "B")] == 0


def test_unmaterialized_zero_overlap_pairs_have_lazy_default_lookup():
    # The first directional pair has positive denominator and no overlap.  It
    # should remain absent from iteration while direct lookup returns 1.0.
    X = np.asarray(
        [[0.0], [0.1], [100.0], [110.0], [110.1], [200.0], [200.1], [210.0], [10.0], [10.1]],
        dtype=float,
    )
    labels = [
        {"A"},
        {"A"},
        {"A", "B"},
        {"B"},
        {"B"},
        {"B", "C"},
        {"B", "C"},
        {"C"},
        {"A", "C"},
        {"A", "C"},
    ]
    model = OverlapIndex(
        **_model_kwargs(
            "KMeans",
            kmeans_k=3,
            offline_memory_budget_mb=1,
            offline_chunk_size=1,
        )
    ).fit(X, labels)

    pair = ("A", "B")
    assert model.pairwise_cardinality[pair] > 0
    materialized = dict(model.pairwise_index)
    assert pair not in materialized
    assert model.pairwise_index[pair] == 1.0


def test_pickle_and_joblib_preserve_budget_and_lazy_pair_lookups(tmp_path):
    X, sequence, _ = _multilabel_data()
    model = OverlapIndex(
        **_model_kwargs(
            "MiniBatchKMeans",
            offline_memory_budget_mb=1,
            offline_chunk_size=1,
        )
    ).fit(X, sequence)

    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params() == model.get_params()
    _assert_fit_outputs_equal(model, restored)

    path = tmp_path / "overlap-universal.joblib"
    joblib.dump(model, path)
    restored_joblib = joblib.load(path)
    assert restored_joblib.index == model.index
    np.testing.assert_array_equal(restored_joblib.predict(X), model.predict(X))
