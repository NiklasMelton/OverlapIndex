from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from experiments.fused3_envelope_reanalysis.w1_quadratic import (
    crossfit_w1,
    decomposed_polynomial_kernel,
    prototype_geometry,
    weighted_lloyd_one_update,
)
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans.scalable import _score_kernel


def test_weighted_lloyd_is_one_fixed_weight_same_class_update() -> None:
    values = np.asarray(
        [
            [-2.0, 0.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [8.0, 0.0],
            [9.0, 1.0],
            [11.0, 1.0],
            [12.0, 0.0],
        ],
        dtype=np.float32,
    )
    target = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    centers = np.asarray([[-2.0, 0.0], [2.0, 0.0], [8.0, 0.0], [12.0, 0.0]])
    owners = np.asarray([0, 0, 1, 1])
    weights = np.asarray([3.0, 0.5])
    centers_before = centers.copy()
    owners_before = owners.copy()
    weights_before = weights.copy()

    updated, diagnostics = weighted_lloyd_one_update(
        values,
        target,
        centers=centers,
        owners=owners,
        weights=weights,
    )

    expected = centers.astype(np.float64, copy=True)
    for label in (0, 1):
        rows = values[target == label]
        ids = np.flatnonzero(owners == label)
        assigned = ids[np.argmax(_score_kernel(rows, centers[ids], weights), axis=1)]
        for prototype_id in ids:
            selected = rows[assigned == prototype_id]
            if len(selected):
                expected[prototype_id] = np.mean(selected, axis=0)
    np.testing.assert_array_equal(updated, np.asarray(expected, dtype=np.float32))
    np.testing.assert_array_equal(centers, centers_before)
    np.testing.assert_array_equal(owners, owners_before)
    np.testing.assert_array_equal(weights, weights_before)
    assert not updated.flags.writeable
    assert diagnostics["weighted_lloyd_update_count"] == 1
    assert diagnostics["weights_relearned"] is False
    assert diagnostics["prototype_owners_changed"] is False


def test_prototype_geometry_reproduces_exact_oi() -> None:
    values = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1])
    result = prototype_geometry(
        values,
        labels,
        centers=values,
        owners=labels,
        weights=np.asarray([1.0, 2.0]),
    )
    assert result["oi_score"] == 1.0
    assert result["nearest_prototype_accuracy"] == 1.0
    assert result["all_rivals_second_own_pass_rate"] == 1.0


def test_crossfit_w1_preserves_frozen_baseline_and_is_deterministic() -> None:
    rng = np.random.default_rng(20260830)
    labels = np.repeat(np.arange(3), 20)
    values = np.vstack(
        [
            rng.normal(loc=4.0 * label, scale=0.4, size=(20, 8))
            for label in range(3)
        ]
    ).astype(np.float32)
    first = crossfit_w1(values, labels, seed=143)
    second = crossfit_w1(values, labels, seed=143)
    assert first["baseline_score"] == second["baseline_score"]
    assert first["w1_score"] == second["w1_score"]
    assert len(first["folds"]) == 5
    for left, right in zip(first["folds"], second["folds"]):
        assert left["baseline"] == right["baseline"]
        assert left["w1"] == right["w1"]
        assert left["w1_update_diagnostics"] == right["w1_update_diagnostics"]


def test_quadratic_kernels_equal_their_explicit_feature_maps() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(size=(9, 5)).astype(np.float32)
    right = rng.normal(size=(6, 5)).astype(np.float32)
    gamma = 0.37

    full = decomposed_polynomial_kernel(
        left, right, gamma=gamma, include_cross_terms=True
    )
    expected_full = np.square(gamma * (left @ right.T) + 1.0)
    np.testing.assert_allclose(full, expected_full, rtol=2e-6, atol=2e-6)

    diagonal = decomposed_polynomial_kernel(
        left, right, gamma=gamma, include_cross_terms=False
    )
    phi_left = np.concatenate(
        [
            np.ones((len(left), 1)),
            np.sqrt(2.0 * gamma) * left,
            gamma * np.square(left),
        ],
        axis=1,
    )
    phi_right = np.concatenate(
        [
            np.ones((len(right), 1)),
            np.sqrt(2.0 * gamma) * right,
            gamma * np.square(right),
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        diagonal, phi_left @ phi_right.T, rtol=2e-6, atol=2e-6
    )


def test_full_precomputed_kernel_matches_frozen_degree_two_svc() -> None:
    rng = np.random.default_rng(11)
    labels = np.repeat(np.arange(3), 30)
    values = np.vstack(
        [
            rng.normal(loc=label * 0.25, scale=1.0, size=(30, 7))
            for label in range(3)
        ]
    ).astype(np.float32)
    train = food101._row_l2(values[:72])
    train_y = labels[:72]
    test = food101._row_l2(values[72:])
    gamma = float(1.0 / (train.shape[1] * float(np.var(train))))

    direct = SVC(kernel="poly", degree=2, C=1.0, gamma="scale", coef0=1.0).fit(
        train, train_y
    )
    precomputed = SVC(kernel="precomputed", C=1.0).fit(
        decomposed_polynomial_kernel(
            train, train, gamma=gamma, include_cross_terms=True
        ),
        train_y,
    )
    np.testing.assert_array_equal(
        precomputed.predict(
            decomposed_polynomial_kernel(
                test, train, gamma=gamma, include_cross_terms=True
            )
        ),
        direct.predict(test),
    )
