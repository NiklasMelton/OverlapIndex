"""Focused tests for the read-only heteroscedastic diagnostics."""

from __future__ import annotations

from collections import defaultdict
import json

import numpy as np
import pytest

from experiments.heteroscedastic_distance_conditioning.diagnostics import (
    assignment_diagnostics_from_selector,
    assert_state_unchanged,
    call_and_assert_state_unchanged,
    canonical_json,
    directional_pair_evidence,
    fold_scale_stability,
    json_safe,
    metric_space_view,
    neighbor_diagnostics,
    pair_distance_diagnostics,
    prototype_assignment_diagnostics,
    refinement_stage_diagnostics,
    snapshot_fitted_state,
)


class _Transform:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=float)
        self.calls = 0

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.asarray(values) @ self.matrix


def test_metric_space_view_detaches_raw_and_uses_fitted_transform_only() -> None:
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    transform = _Transform(np.diag([2.0, 3.0]))
    view = metric_space_view(raw, transform=transform)

    assert transform.calls == 1
    np.testing.assert_allclose(view.raw, raw)
    np.testing.assert_allclose(view.conditioned, [[2.0, 6.0], [6.0, 12.0]])
    assert view["raw"] is view.raw
    assert view["conditioned"] is view.conditioned
    assert not view.raw.flags.writeable
    assert not view.conditioned.flags.writeable
    raw[0, 0] = 999.0
    assert view.raw[0, 0] == 1.0
    with pytest.raises(ValueError):
        metric_space_view(raw, conditioned=raw, transform=transform)


def test_neighbor_diagnostics_is_separated_only_and_hand_counted() -> None:
    metric = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    oracle = np.asarray([[0.0], [1.0], [10.0], [11.0]])
    labels = np.asarray([0, 0, 1, 1])

    result = neighbor_diagnostics(metric, oracle, labels, 1)
    assert result["oracle_neighbor_jaccard"] == pytest.approx(0.75)
    assert result["cross_class_neighbor_impurity"] == pytest.approx(0.25)
    assert result["n_rows_evaluated"] == 4
    assert result["separated_only"] is False

    result = neighbor_diagnostics(
        metric,
        oracle,
        labels,
        1,
        separated_mask=np.asarray([False, True, True, False]),
    )
    assert result["oracle_neighbor_jaccard"] == pytest.approx(0.5)
    assert result["cross_class_neighbor_impurity"] == pytest.approx(0.5)
    assert result["separated_only"] is True


def test_pair_distance_spearman_preserves_supplied_identity_and_ties() -> None:
    metric = np.asarray([[0.0], [1.0], [2.0], [4.0]])
    oracle = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
    pairs = [(0, 1), (0, 2), (0, 3), (1, 3)]
    result = pair_distance_diagnostics(metric, oracle, pairs)

    assert result["pair_distance_spearman"] == pytest.approx(1.0)
    assert result["pair_indices"] == [list(pair) for pair in pairs]
    assert result["n_pairs"] == len(pairs)
    assert result["metric_n_features"] == 1
    assert result["oracle_n_features"] == 2
    assert isinstance(result["pair_identity_sha256"], str)

    tied = pair_distance_diagnostics(
        np.asarray([[0.0], [1.0], [2.0]]),
        np.asarray([[0.0], [2.0], [4.0]]),
        [(0, 1), (0, 2)],
    )
    assert tied["pair_distance_spearman"] == pytest.approx(1.0)
    one = pair_distance_diagnostics(metric, oracle, [(0, 1)])
    assert one["pair_distance_spearman"] is None

    with pytest.raises(ValueError, match="off-diagonal"):
        pair_distance_diagnostics(metric, oracle, [(0, 0)])
    with pytest.raises(ValueError, match="duplicate"):
        pair_distance_diagnostics(metric, oracle, [(0, 1), (0, 1)])


def test_neighbor_diagnostics_requires_boolean_aligned_separation_mask() -> None:
    metric = np.asarray([[0.0], [1.0], [2.0]])
    oracle = np.asarray([[0.0], [1.0], [2.0]])
    labels = np.asarray([0, 1, 1])
    with pytest.raises(ValueError, match="dtype=bool"):
        neighbor_diagnostics(metric, oracle, labels, 1, separated_mask=[0, 1, 1])
    with pytest.raises(ValueError, match="align"):
        neighbor_diagnostics(
            metric,
            oracle,
            labels,
            1,
            separated_mask=np.asarray([True, False], dtype=bool),
        )


class _RecordingConditioner:
    records: list[np.ndarray] = []

    def __init__(self) -> None:
        self.transform_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RecordingConditioner":
        self.__class__.records.append(np.asarray(y).copy())
        # A fitted transform seam sufficient for the stability diagnostic; the
        # rows and labels are still passed through the real fold-local fit call.
        self.transform_ = np.diag(np.asarray([2.0, 3.0]))
        return self


def test_fold_scale_stability_uses_original_heterogeneous_labels() -> None:
    _RecordingConditioner.records = []
    X = np.arange(20, dtype=float).reshape(10, 2)
    labels = np.asarray([1, "one"] * 5, dtype=object)
    result = fold_scale_stability(
        X,
        labels,
        _RecordingConditioner,
        seed=19,
        n_splits=5,
    )

    assert result["fold_count"] == 5
    assert result["fold_pair_count"] == 10
    assert result["fold_scale_spearman"] == pytest.approx(1.0)
    assert result["median_log_scale_difference"] == pytest.approx(0.0)
    assert result["max_log_scale_difference"] == pytest.approx(0.0)
    assert all(set(record.tolist()) == {1, "one"} for record in _RecordingConditioner.records)
    # The split encoding is a diagnostic trace only; it is not what the
    # fold-local conditioner receives.
    assert set(result["label_encoding_for_split"]) == {0, 1}
    assert result == fold_scale_stability(
        X,
        labels,
        _RecordingConditioner,
        seed=19,
        n_splits=5,
    )


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([1, 2] * 5, dtype=object),
        np.asarray(["one", "two"] * 5, dtype=object),
        np.asarray(["one", "two"] * 5),
    ],
)
def test_fold_scale_stability_accepts_object_integer_and_string_labels(labels: np.ndarray) -> None:
    _RecordingConditioner.records = []
    X = np.arange(20, dtype=float).reshape(10, 2)
    result = fold_scale_stability(
        X,
        labels,
        _RecordingConditioner,
        seed=23,
    )

    assert result["fold_count"] == 5
    assert all(np.array_equal(np.unique(record), np.unique(labels)) for record in _RecordingConditioner.records)
    assert all(record.dtype == labels.dtype for record in _RecordingConditioner.records)


def test_prototype_assignment_metrics_are_hand_counted_and_normalized() -> None:
    assignments = np.asarray([0, 0, 1, 2, 2])
    labels = np.asarray(["a", "a", "b", "b", "a"], dtype=object)
    owners = ["a", "b", "c", "d"]
    result = prototype_assignment_diagnostics(
        assignments,
        labels,
        owners,
        n_prototypes=4,
        class_labels=["a", "b", "c", "d"],
    )

    assert result["occupied_prototype_count"] == 3
    assert result["empty_prototype_count"] == 1
    assert result["empty_rate_all"] == pytest.approx(0.25)
    assert result["occupied_normalized_label_entropy_macro"] == pytest.approx(1.0 / 6.0)
    assert result["occupied_normalized_label_entropy_weighted"] == pytest.approx(0.2)
    assert result["mixed_rate_occupied"] == pytest.approx(1.0 / 3.0)
    assert result["weighted_majority_assignment_purity"] == pytest.approx(0.8)
    assert result["owner_label_agreement"] == pytest.approx(0.6)
    assert result["under_prototyped"] is True


class _Selector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        np.testing.assert_allclose(X, [[1.0], [2.0], [3.0]])
        return np.asarray([0, 1, 1])


def test_assignment_selector_uses_public_predict_and_one_prototype_flag() -> None:
    selector = _Selector()
    result = assignment_diagnostics_from_selector(
        selector,
        np.asarray([[1.0], [2.0], [3.0]]),
        np.asarray(["x", "y", "y"], dtype=object),
        owner_labels=["x", "y", "z"],
        n_prototypes=3,
    )
    assert selector.calls == 1
    assert result["occupied_prototype_count"] == 2
    assert result["empty_rate_all"] == pytest.approx(1.0 / 3.0)
    assert result["under_prototyped"] is True
    assert result["under_prototyped_labels"][0]["prototype_count"] == 1


def test_refinement_metrics_use_pre_refinement_denominator_and_changed_parents() -> None:
    result = refinement_stage_diagnostics(
        {
            "prototype_count_before": 10,
            "prototype_count_after": 12,
            "eligible_count": 4,
            "applied_count": 2,
            "skipped_count": 2,
            "applied_parent_ids": (7, 2),
        },
        unrefined_score=0.5,
        refined_score=0.7,
    )
    assert result["eligible_rate_all_pre_refinement_prototypes"] == pytest.approx(0.4)
    assert result["applied_rate_all_pre_refinement_prototypes"] == pytest.approx(0.2)
    assert result["skipped_rate_all_pre_refinement_prototypes"] == pytest.approx(0.2)
    assert result["refinement_activity_all_pre_refinement_prototypes"] == pytest.approx(0.2)
    assert result["changed_parent_ids"] == [2, 7]
    assert result["paired_total_refined_vs_unrefined_score_movement"] == pytest.approx(0.2)


class _PairState:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.pairwise_cardinality = {(0, 1): 4, (1, 0): 4}
        self._pairwise_hits = {(0, 1): 1, (1, 0): 2}
        self.sparse_adj = {(0, 1): 1, (1, 0): 2}
        if mismatch:
            self.sparse_adj[(1, 0)] = 1
        self.pairwise_index = {
            (0, 1): 0.75,
            (1, 0): 0.5,
        }
        self.under_prototyped_labels_ = ()
        self._score_classes = (0, 1)


def test_directional_pair_evidence_validates_both_oi_representations() -> None:
    model = _PairState()
    result = directional_pair_evidence(model, np.asarray([0, 0, 1, 1]))
    assert result["exact_state_match"] is True
    assert result["pair_count"] == 2
    assert result["directional_pair_support_hits_evidence"][0]["evidence"] == pytest.approx(0.25)

    with pytest.raises(AssertionError, match="state mismatch"):
        directional_pair_evidence(_PairState(mismatch=True), np.asarray([0, 0, 1, 1]))


def test_directional_pair_trace_does_not_insert_into_defaultdicts() -> None:
    model = _PairState()
    model.pairwise_cardinality = defaultdict(int, model.pairwise_cardinality)
    model._pairwise_hits = defaultdict(int, model._pairwise_hits)
    model.sparse_adj = defaultdict(int, model.sparse_adj)
    before = (set(model.pairwise_cardinality), set(model._pairwise_hits), set(model.sparse_adj))
    directional_pair_evidence(model, np.asarray([0, 0, 1, 1]))
    after = (set(model.pairwise_cardinality), set(model._pairwise_hits), set(model.sparse_adj))
    assert before == after


class _Backend:
    def __init__(self) -> None:
        self.centers = np.asarray([[1.0, 2.0], [3.0, 4.0]])


class _FittedModel:
    def __init__(self) -> None:
        self._model = _Backend()
        self.index = 0.25
        self._conditioning_diagnostics = {
            "state_sha256": "abc",
            "mode": "none",
            "condition_after": None,
        }
        self.prototype_refinement_ = {
            "prototype_count_before": 2,
            "applied_parent_ids": (),
        }

    @property
    def conditioning_diagnostics_(self) -> dict[str, object]:
        return dict(self._conditioning_diagnostics)


def test_state_snapshot_and_read_only_operation_are_structurally_invariant() -> None:
    model = _FittedModel()
    before = snapshot_fitted_state(model)
    returned = model.conditioning_diagnostics_
    returned["state_sha256"] = "changed"
    assert snapshot_fitted_state(model) == before
    call_and_assert_state_unchanged(model, lambda: model.index)
    assert_state_unchanged(before, snapshot_fitted_state(model))


def test_state_snapshot_allows_score_index_but_rejects_structural_mutation() -> None:
    model = _FittedModel()
    before = snapshot_fitted_state(model)

    # score_fixed is allowed to update the aggregate index and its holdout
    # bookkeeping; the static no-refit snapshot must remain unchanged.
    model.index = 0.91
    call_and_assert_state_unchanged(model, lambda: model.index)
    assert_state_unchanged(before, snapshot_fitted_state(model))

    centers_changed = _FittedModel()
    centers_before = snapshot_fitted_state(centers_changed)
    centers_changed._model.centers[0, 0] += 1.0
    with pytest.raises(AssertionError, match="state changed"):
        assert_state_unchanged(centers_before, snapshot_fitted_state(centers_changed))

    conditioning_changed = _FittedModel()
    conditioning_before = snapshot_fitted_state(conditioning_changed)
    conditioning_changed._conditioning_diagnostics["state_sha256"] = "changed"
    with pytest.raises(AssertionError, match="state changed"):
        assert_state_unchanged(
            conditioning_before,
            snapshot_fitted_state(conditioning_changed),
        )

    refinement_changed = _FittedModel()
    refinement_before = snapshot_fitted_state(refinement_changed)
    refinement_changed.prototype_refinement_["applied_parent_ids"] = (0,)
    with pytest.raises(AssertionError, match="state changed"):
        assert_state_unchanged(refinement_before, snapshot_fitted_state(refinement_changed))


def test_json_safe_diagnostics_are_deterministic_and_allow_nan_free() -> None:
    payload = {
        "z": np.float64(np.inf),
        "a": np.asarray([np.int64(2), np.nan]),
        "nested": (np.bool_(True),),
    }
    safe = json_safe(payload)
    assert safe == {"z": None, "a": [2, None], "nested": [True]}
    assert json_safe({"b", "a"}) == ["a", "b"]
    serialized = canonical_json(payload)
    assert serialized == canonical_json(payload)
    json.loads(serialized)
    json.dumps(safe, allow_nan=False)
