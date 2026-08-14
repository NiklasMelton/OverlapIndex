"""Public contracts for the optional balanced-median prototype refinement."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone

from overlapindex import OverlapIndex


BACKENDS = ("KMeans", "MiniBatchKMeans")


def _kmeans_kwargs(model_type: str) -> dict[str, object]:
    kwargs: dict[str, object] = {"random_state": 0, "n_init": 10}
    if model_type == "MiniBatchKMeans":
        kwargs.update({"batch_size": 8, "max_iter": 100})
    return kwargs


def _model(
    model_type: str,
    *,
    refinement: bool = False,
    k: int = 1,
    **kwargs: object,
) -> OverlapIndex:
    params: dict[str, object] = {
        "model_type": model_type,
        "kmeans_k": k,
        "kmeans_kwargs": _kmeans_kwargs(model_type),
        "prototype_refinement": refinement,
    }
    params.update(kwargs)
    return OverlapIndex(**params)


def _balanced_data() -> tuple[np.ndarray, np.ndarray]:
    """One-dimensional classes whose median halves have known observations."""

    X = np.asarray(
        [[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]],
        dtype=float,
    )
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def _multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [10.0],
            [11.0],
            [12.0],
            [13.0],
            [20.0],
            [21.0],
            [22.0],
            [23.0],
        ],
        dtype=float,
    )
    # Non-contiguous labels catch accidental use of labels as global IDs.
    y = np.asarray([10] * 4 + [20] * 4 + [30] * 4)
    return X, y


def _trap_data() -> tuple[np.ndarray, np.ndarray]:
    """Two prototypes per class with every parent eligible for a split."""

    X = np.asarray(
        [
            [0.0],
            [0.1],
            [10.0],
            [9.9],
            [3.0],
            [3.1],
            [7.0],
            [6.9],
        ],
        dtype=float,
    )
    y = np.asarray([0] * 4 + [1] * 4)
    return X, y


def _far_prototypes_data() -> tuple[np.ndarray, np.ndarray]:
    """Own runner-up scores dominate distant competitors, so no parent splits."""

    X = np.asarray(
        [
            [0.0],
            [0.1],
            [10.0],
            [10.1],
            [100.0],
            [100.1],
            [110.0],
            [110.1],
        ],
        dtype=float,
    )
    y = np.asarray([0] * 4 + [1] * 4)
    return X, y


def _duplicate_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray([[0.0], [0.0], [0.0], [0.0], [10.0], [10.0], [10.0], [10.0]])
    y = np.asarray([0] * 4 + [1] * 4)
    return X, y


def _summary(model: OverlapIndex) -> Mapping[str, object]:
    value = getattr(model, "prototype_refinement_", None)
    assert isinstance(value, Mapping), "fitted prototype_refinement_ summary is required"
    return value


def _records(model: OverlapIndex) -> tuple[Mapping[str, object], ...]:
    records = _summary(model).get("records")
    assert records is not None
    return tuple(records)  # type: ignore[arg-type]


def _backend_centers(model: OverlapIndex) -> np.ndarray:
    return np.asarray(model._model.centers)


def _backend_ids(model: OverlapIndex) -> dict[object, np.ndarray]:
    return {
        label: np.asarray(values, dtype=int)
        for label, values in model._model.class_center_id_arrays.items()
    }


def _assert_mapping_close(left: Mapping[object, object], right: Mapping[object, object]) -> None:
    assert set(left) == set(right)
    for key in left:
        lhs, rhs = left[key], right[key]
        if isinstance(lhs, (float, np.floating)) or isinstance(rhs, (float, np.floating)):
            assert float(lhs) == pytest.approx(float(rhs), abs=1e-7)
        else:
            assert lhs == rhs


@pytest.mark.parametrize("model_type", BACKENDS)
def test_default_off_and_explicit_false_are_backward_parity(model_type: str) -> None:
    X, y = _balanced_data()
    default = OverlapIndex(
        model_type=model_type,
        kmeans_k=1,
        kmeans_kwargs=_kmeans_kwargs(model_type),
    ).fit(X, y)
    explicit = _model(model_type, refinement=False).fit(X, y)

    assert default.prototype_refinement is False
    assert explicit.prototype_refinement is False
    assert default.get_params() == explicit.get_params()
    assert default.index == pytest.approx(explicit.index, abs=0.0)
    assert default.weighted_index == pytest.approx(explicit.weighted_index, abs=0.0)
    _assert_mapping_close(default.singleton_index, explicit.singleton_index)
    assert dict(default.cluster_cardinality) == dict(explicit.cluster_cardinality)
    assert {label: set(ids) for label, ids in default.rev_map.items()} == {
        label: set(ids) for label, ids in explicit.rev_map.items()
    }
    np.testing.assert_array_equal(_backend_centers(default), _backend_centers(explicit))
    np.testing.assert_array_equal(default.predict(X), explicit.predict(X))

    summary = _summary(default)
    assert summary["method"] == "none"
    assert summary["eligible_count"] == 0
    assert summary["attempted_count"] == 0
    assert summary["applied_count"] == 0
    assert summary["skipped_count"] == 0
    assert summary["prototype_count_before"] == summary["prototype_count_after"]
    assert tuple(summary["records"]) == ()


@pytest.mark.parametrize("model_type", BACKENDS)
def test_true_balanced_median_is_deterministic_one_pass_and_observation_based(
    model_type: str,
) -> None:
    X, y = _balanced_data()
    first = _model(model_type, refinement=True).fit(X, y)
    second = _model(model_type, refinement=True).fit(X, y)

    assert first.prototype_refinement is True
    assert second.prototype_refinement is True
    np.testing.assert_array_equal(_backend_centers(first), _backend_centers(second))
    assert _backend_ids(first).keys() == _backend_ids(second).keys()
    for label in _backend_ids(first):
        np.testing.assert_array_equal(_backend_ids(first)[label], _backend_ids(second)[label])
    assert dict(_summary(first)) == dict(_summary(second))

    summary = _summary(first)
    assert summary["method"] == "balanced_median"
    assert summary["prototype_count_before"] == 2
    assert summary["prototype_count_after"] == 4
    assert summary["eligible_count"] == 2
    assert summary["attempted_count"] == 2
    assert summary["applied_count"] == 2
    assert summary["skipped_count"] == 0

    centers = _backend_centers(first)
    records = _records(first)
    assert len(records) == 2
    assert {
        int(record["parent_id"]): tuple(record["selected_observation_indices"])
        for record in records
    } == {0: (0, 2), 1: (4, 6)}
    for record in records:
        assert record["status"] == "applied"
        assert int(record["support"]) >= 2
        children = np.asarray(record["child_ids"], dtype=int)
        assert children.shape == (2,)
        supports = np.asarray(record["child_supports"], dtype=int)
        assert supports.shape == (2,)
        assert np.all(supports > 0)
        assert int(supports.sum()) == int(record["support"])
        selected = np.asarray(record["selected_observation_indices"], dtype=int)
        assert selected.shape == (2,)
        # Every child center is an actual fit observation, never a synthetic
        # coordinate median; the integer fixture also makes this exact.
        np.testing.assert_array_equal(centers[children], X[selected])

    # One pass: children are not recursively eligible in the same fit.
    assert summary["prototype_count_after"] == summary["prototype_count_before"] + summary["applied_count"]


@pytest.mark.parametrize("model_type", BACKENDS)
def test_eligibility_uses_support_threshold_and_outgoing_runner_up(model_type: str) -> None:
    # Support exactly two is eligible with one prototype per class.
    X_two = np.asarray([[0.0], [1.0], [10.0], [11.0]])
    y_two = np.asarray([0, 0, 1, 1])
    two = _model(model_type, refinement=True).fit(X_two, y_two)
    assert _summary(two)["eligible_count"] == 2
    assert _summary(two)["applied_count"] == 2
    assert _summary(two)["prototype_count_after"] == 4

    # A one-row parent is below the fixed support >=2 eligibility threshold;
    # the other class still refines normally.
    X_one = np.asarray([[0.0], [10.0], [11.0], [12.0], [13.0]])
    y_one = np.asarray([0, 1, 1, 1, 1])
    one = _model(model_type, refinement=True).fit(X_one, y_one)
    assert _summary(one)["eligible_count"] == 1
    assert _summary(one)["applied_count"] == 1
    assert _backend_ids(one)[0].size == 1
    assert _backend_ids(one)[1].size == 2

    # With well-separated two-prototype classes, own runner-up evidence is
    # present for every parent; support alone must not make them eligible.
    X_far, y_far = _far_prototypes_data()
    far = _model(model_type, refinement=True, k=2).fit(X_far, y_far)
    assert _summary(far)["eligible_count"] == 0
    assert _summary(far)["applied_count"] == 0
    assert _summary(far)["prototype_count_before"] == 4
    assert _summary(far)["prototype_count_after"] == 4


@pytest.mark.parametrize("model_type", BACKENDS)
def test_multiple_parents_preserve_class_ownership_and_contiguous_global_ids(
    model_type: str,
) -> None:
    X, y = _trap_data()
    model = _model(model_type, refinement=True, k=2).fit(X, y)
    summary = _summary(model)
    ids = _backend_ids(model)

    assert summary["prototype_count_before"] == 4
    assert summary["prototype_count_after"] == 8
    assert summary["applied_count"] == 4
    all_ids = sorted(int(pid) for values in ids.values() for pid in values)
    assert all_ids == list(range(8))
    # Original IDs stay in place; newly created children are appended to the
    # global ID range rather than interleaved into each class's old block.
    prototype_count_before = int(summary["prototype_count_before"])
    records = _records(model)
    for record in records:
        assert record["child_ids"][0] == record["parent_id"]
        assert int(record["child_ids"][1]) >= prototype_count_before
    appended_ids = sorted(
        int(record["child_ids"][1])
        for record in records
        if record["status"] == "applied"
    )
    assert appended_ids == list(range(prototype_count_before, int(summary["prototype_count_after"])))
    cluster_to_class = np.asarray(model._model.cluster_to_class, dtype=object)
    assert cluster_to_class.shape == (8,)
    for label, values in ids.items():
        assert set(cluster_to_class[values].tolist()) == {label}


@pytest.mark.parametrize("model_type", BACKENDS)
def test_duplicate_observation_representatives_are_skipped_without_id_gaps(
    model_type: str,
) -> None:
    X, y = _duplicate_data()
    model = _model(model_type, refinement=True).fit(X, y)
    summary = _summary(model)

    assert summary["eligible_count"] == 2
    assert summary["attempted_count"] == 2
    assert summary["applied_count"] == 0
    assert summary["skipped_count"] == 2
    assert summary["prototype_count_before"] == summary["prototype_count_after"] == 2
    assert all(record["status"] == "skipped" for record in _records(model))
    assert all(
        str(record["reason"])
        in {"duplicate_points", "duplicate_representatives", "no_progress", "invalid_child"}
        for record in _records(model)
    )
    ids = _backend_ids(model)
    assert sorted(int(pid) for values in ids.values() for pid in values) == [0, 1]


@pytest.mark.parametrize("model_type", BACKENDS)
def test_multiclass_noncontiguous_labels_keep_class_owned_children(model_type: str) -> None:
    X, y = _multiclass_data()
    model = _model(model_type, refinement=True).fit(X, y)
    ids = _backend_ids(model)
    assert set(ids) == {10, 20, 30}
    assert sorted(int(pid) for values in ids.values() for pid in values) == list(range(6))
    for label, values in ids.items():
        assert values.size == 2
        assert set(np.asarray(model._model.cluster_to_class, dtype=object)[values].tolist()) == {label}


@pytest.mark.parametrize("model_type", BACKENDS)
def test_score_fixed_is_immutable_and_does_not_run_a_second_refinement(model_type: str) -> None:
    X, y = _balanced_data()
    model = _model(model_type, refinement=True).fit(X, y)
    centers_before = _backend_centers(model).copy()
    ids_before = {label: values.copy() for label, values in _backend_ids(model).items()}
    summary_before = dict(_summary(model))
    eval_X = np.asarray([[0.5], [2.5], [10.5], [12.5]])
    eval_y = np.asarray([0, 0, 1, 1])

    score = model.score_fixed(eval_X, eval_y)

    assert np.isfinite(score)
    np.testing.assert_array_equal(_backend_centers(model), centers_before)
    for label, values in ids_before.items():
        np.testing.assert_array_equal(_backend_ids(model)[label], values)
    assert dict(_summary(model)) == summary_before


@pytest.mark.parametrize("model_type", BACKENDS)
def test_repeated_fit_and_score_rebuild_one_refinement_pass(model_type: str) -> None:
    X, y = _balanced_data()
    model = _model(model_type, refinement=True)
    model.fit(X, y)
    first_centers = _backend_centers(model).copy()
    first_summary = dict(_summary(model))

    model.fit(X, y)
    np.testing.assert_array_equal(_backend_centers(model), first_centers)
    assert dict(_summary(model)) == first_summary
    model.score(X, y)
    np.testing.assert_array_equal(_backend_centers(model), first_centers)
    assert dict(_summary(model)) == first_summary

    with pytest.raises(ValueError, match="reset_state=False is supported only for ARTMAP"):
        model.fit_offline(X, y, reset_state=False)
    np.testing.assert_array_equal(_backend_centers(model), first_centers)
    assert dict(_summary(model)) == first_summary


def test_get_set_params_and_clone_expose_only_the_public_refinement_switch() -> None:
    model = _model("KMeans", refinement=True)
    params = model.get_params()
    assert params["prototype_refinement"] is True
    assert "splitter" not in params
    assert "init" not in params
    assert "min_support" not in params

    copied = clone(model)
    assert copied is not model
    assert copied.prototype_refinement is True
    assert copied.get_params() == params
    copied.fit(*_balanced_data())
    assert _summary(copied)["method"] == "balanced_median"

    model.set_params(prototype_refinement=False)
    assert model.prototype_refinement is False
    assert model.get_params()["prototype_refinement"] is False


def test_invalid_refinement_values_and_unsupported_backends_fail_clearly() -> None:
    for invalid in (
        "none",
        "balanced_median",
        "unknown",
        None,
        [],
        0,
        1,
        np.bool_(True),
    ):
        with pytest.raises(ValueError, match="prototype_refinement"):
            OverlapIndex(prototype_refinement=invalid)

    # The switch is accepted on every backend while disabled; only enabling
    # refinement on an unsupported backend should fail.
    disabled = OverlapIndex(model_type="BallCover", prototype_refinement=False).fit(
        *_balanced_data()
    )
    assert disabled.prototype_refinement is False

    with pytest.raises((ValueError, NotImplementedError), match="KMeans|MiniBatchKMeans|prototype_refinement"):
        bad = OverlapIndex(model_type="BallCover", prototype_refinement=True)
        bad.fit(*_balanced_data())

    model = _model("KMeans")
    with pytest.raises(ValueError, match="prototype_refinement"):
        model.set_params(prototype_refinement="unknown")
    assert model.prototype_refinement is False

    with pytest.raises(ValueError, match="KMeans|MiniBatchKMeans|prototype_refinement"):
        model.set_params(model_type="BallCover", prototype_refinement=True)
    assert model.model_type == "KMeans"
    assert model.prototype_refinement is False


@pytest.mark.parametrize("model_type", BACKENDS)
def test_true_balanced_multilabel_rejected_but_false_preserves_existing_behavior(model_type: str) -> None:
    X = np.asarray([[0.0], [0.1], [1.0], [1.1], [2.0], [2.1]])
    labels = [{"A", "B"}, {"A"}, {"B"}, {"B", "C"}, {"C"}, {"A", "C"}]

    default = OverlapIndex(
        model_type=model_type,
        kmeans_k=1,
        kmeans_kwargs=_kmeans_kwargs(model_type),
    ).fit(X, labels)
    explicit = _model(model_type, refinement=False).fit(X, labels)
    assert default.index == pytest.approx(explicit.index, abs=0.0)
    _assert_mapping_close(default.singleton_index, explicit.singleton_index)
    assert dict(default.cluster_cardinality) == dict(explicit.cluster_cardinality)
    np.testing.assert_array_equal(_backend_centers(default), _backend_centers(explicit))

    with pytest.raises((ValueError, NotImplementedError), match="multi-label|multilabel|single-label"):
        _model(model_type, refinement=True).fit(X, labels)


@pytest.mark.parametrize("model_type", BACKENDS)
def test_balanced_score_fixed_rejects_multilabel_evaluation(model_type: str) -> None:
    X, y = _balanced_data()
    model = _model(model_type, refinement=True).fit(X, y)
    eval_X = np.asarray([[0.5], [2.5], [10.5], [12.5]])
    eval_labels = [{0}, {0, 1}, {1}, {1}]

    with pytest.raises((ValueError, NotImplementedError), match="multi-label|multilabel|single-label"):
        model.score_fixed(eval_X, eval_labels)


@pytest.mark.parametrize("model_type", BACKENDS)
def test_true_balanced_median_sparse_fit_matches_dense_contract(model_type: str) -> None:
    X, y = _balanced_data()
    dense = _model(model_type, refinement=True).fit(X, y)
    sparse_model = _model(model_type, refinement=True).fit(
        sparse.csr_matrix(X), y
    )

    assert sparse_model.index == pytest.approx(dense.index, abs=1e-6)
    assert sparse_model.weighted_index == pytest.approx(dense.weighted_index, abs=1e-6)
    np.testing.assert_allclose(_backend_centers(sparse_model), _backend_centers(dense), atol=1e-6, rtol=0.0)
    for label in _backend_ids(dense):
        np.testing.assert_array_equal(_backend_ids(sparse_model)[label], _backend_ids(dense)[label])
    assert dict(_summary(sparse_model)) == dict(_summary(dense))
    np.testing.assert_array_equal(
        sparse_model.predict(sparse.csr_matrix(X)),
        dense.predict(X),
    )
