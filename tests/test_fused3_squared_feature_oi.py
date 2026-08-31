from __future__ import annotations

import numpy as np

from experiments.fused3_envelope_reanalysis.squared_feature_oi import (
    _csv_bytes,
    crossfit_squared_fused3,
    fit_squared_feature_lift,
)
from experiments.m50_backbone_ranking import food101


def _nearest_centroid_accuracy(values: np.ndarray, labels: np.ndarray) -> float:
    classes = list(dict.fromkeys(labels.tolist()))
    centers = np.vstack([np.mean(values[labels == label], axis=0) for label in classes])
    distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    predicted = np.asarray(classes)[np.argmin(distances, axis=1)]
    return float(np.mean(predicted == labels))


def test_square_lift_exposes_sign_symmetric_axis_separation() -> None:
    class_zero = np.tile(
        np.asarray([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]),
        (30, 1),
    )
    class_one = np.tile(
        np.asarray([[0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]),
        (30, 1),
    )
    values = np.asarray(np.vstack((class_zero, class_one)), dtype=np.float32)
    labels = np.asarray(["axis_x"] * len(class_zero) + ["axis_y"] * len(class_one))
    normalized = food101._row_l2(values)

    lift = fit_squared_feature_lift(values, labels, max_square_features=1)
    transformed = lift.transform(values)

    assert lift.selected_indices.tolist() == [0]
    assert _nearest_centroid_accuracy(normalized, labels) == 0.5
    assert _nearest_centroid_accuracy(transformed, labels) == 1.0
    assert transformed.shape == (len(values), values.shape[1] + 1)


def test_square_relevance_is_zero_for_paired_linear_sign_separation() -> None:
    rng = np.random.default_rng(19)
    nuisance = rng.normal(size=(40, 5))
    negative = np.column_stack((-np.full(40, 2.0), nuisance))
    positive = np.column_stack((np.full(40, 2.0), nuisance))
    values = np.asarray(np.vstack((negative, positive)), dtype=np.float32)
    labels = np.asarray([0] * 40 + [1] * 40)

    lift = fit_squared_feature_lift(values, labels, max_square_features=None)

    assert float(np.max(lift.selected_relevance)) <= 1.0e-14
    assert lift.selected_indices.tolist() == list(range(values.shape[1]))


def test_lift_is_train_only_immutable_and_deterministic() -> None:
    rng = np.random.default_rng(23)
    values = rng.normal(size=(90, 12)).astype(np.float32)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 30)
    first = fit_squared_feature_lift(values, labels, max_square_features=5)
    second = fit_squared_feature_lift(values, labels, max_square_features=5)
    state_before = (
        first.state_sha256,
        first.selected_indices.copy(),
        first.square_means.copy(),
        first.square_scales.copy(),
    )

    transformed = first.transform(values[:11] * np.float32(1.1))

    assert transformed.shape == (11, 17)
    assert first.state_sha256 == second.state_sha256
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    np.testing.assert_array_equal(first.square_means, second.square_means)
    np.testing.assert_array_equal(first.square_scales, second.square_scales)
    assert first.state_sha256 == state_before[0]
    np.testing.assert_array_equal(first.selected_indices, state_before[1])
    np.testing.assert_array_equal(first.square_means, state_before[2])
    np.testing.assert_array_equal(first.square_scales, state_before[3])
    assert not first.selected_indices.flags.writeable
    assert not first.square_means.flags.writeable
    assert not first.square_scales.flags.writeable


def test_crossfit_squared_fused3_is_repeatable_with_original_scalar_labels() -> None:
    rng = np.random.default_rng(31)
    labels = np.repeat(np.asarray(["red", "green", "blue"], dtype=object), 20)
    values = np.vstack(
        [
            rng.normal(loc=label, scale=0.5, size=(20, 10))
            for label in (0.0, 2.0, 4.0)
        ]
    ).astype(np.float32)

    first = crossfit_squared_fused3(
        values, labels, seed=143, max_square_features=4
    )
    second = crossfit_squared_fused3(
        values, labels, seed=143, max_square_features=4
    )

    assert first["score"] == second["score"]
    assert len(first["folds"]) == 5
    for left, right in zip(first["folds"], second["folds"]):
        assert left["score"] == right["score"]
        assert left["state_sha256"] == right["state_sha256"]
        assert left["selected_indices"] == right["selected_indices"]
        assert left["output_feature_count"] == 14


def test_csv_schema_uses_closed_union_for_baseline_and_lift_rows() -> None:
    payload = _csv_bytes(
        [
            {"candidate_id": "FUSED3", "baseline_only": 1},
            {"candidate_id": "FUSED3-SQ32", "lift_only": 2},
        ]
    ).decode("utf-8")
    header, baseline, lifted = payload.splitlines()
    assert header == "candidate_id,baseline_only,lift_only"
    assert baseline == "FUSED3,1,"
    assert lifted == "FUSED3-SQ32,,2"
