from __future__ import annotations

import numpy as np

from experiments.fused3_envelope_reanalysis.redundancy_prototypes import (
    EPOCHS,
    crossfit_redundancy_state,
    redundancy_lvq_epoch,
)


def test_oi_lvq_moves_second_own_toward_and_rival_away() -> None:
    values = np.asarray([[-2.0], [2.0]], dtype=np.float32)
    labels = np.asarray([0, 1])
    centers = np.asarray([[-2.0], [-10.0], [2.0], [10.0]], dtype=np.float32)
    owners = np.asarray([0, 0, 1, 1])
    weights = np.asarray([1.0])

    updated, diagnostics = redundancy_lvq_epoch(
        values,
        labels,
        centers=centers,
        anchor_centers=centers,
        owners=owners,
        weights=weights,
        learning_rate=0.1,
        anchor_strength=0.0,
    )

    assert updated[1, 0] > centers[1, 0]
    assert updated[3, 0] < centers[3, 0]
    assert updated[2, 0] > centers[2, 0]
    assert updated[0, 0] < centers[0, 0]
    assert diagnostics["active_violation_count"] == 2
    assert diagnostics["second_own_prototype_attracted"] is True
    assert diagnostics["nearest_rival_prototype_repelled"] is True
    assert diagnostics["best_own_prototype_attracted"] is False


def test_oi_lvq_is_noop_when_second_own_already_beats_rival() -> None:
    values = np.asarray([[-2.5], [2.5]], dtype=np.float32)
    labels = np.asarray([0, 1])
    centers = np.asarray([[-2.0], [-3.0], [2.0], [3.0]], dtype=np.float32)
    owners = np.asarray([0, 0, 1, 1])
    weights = np.asarray([1.0])

    updated, diagnostics = redundancy_lvq_epoch(
        values,
        labels,
        centers=centers,
        anchor_centers=centers,
        owners=owners,
        weights=weights,
    )

    np.testing.assert_array_equal(updated, centers)
    assert diagnostics["active_violation_count"] == 0
    assert diagnostics["changed_prototype_count"] == 0


def test_anchor_reduces_first_epoch_displacement_and_inputs_are_immutable() -> None:
    values = np.asarray([[-2.0], [2.0]], dtype=np.float32)
    labels = np.asarray([0, 1])
    centers = np.asarray([[-2.0], [-10.0], [2.0], [10.0]], dtype=np.float32)
    owners = np.asarray([0, 0, 1, 1])
    weights = np.asarray([1.0])
    state = tuple(array.copy() for array in (values, labels, centers, owners, weights))

    unanchored, _ = redundancy_lvq_epoch(
        values,
        labels,
        centers=centers,
        anchor_centers=centers,
        owners=owners,
        weights=weights,
        anchor_strength=0.0,
    )
    anchored, _ = redundancy_lvq_epoch(
        values,
        labels,
        centers=centers,
        anchor_centers=centers,
        owners=owners,
        weights=weights,
        anchor_strength=0.5,
    )

    unanchored_step = np.linalg.norm(unanchored - centers)
    anchored_step = np.linalg.norm(anchored - centers)
    np.testing.assert_allclose(
        anchored_step, 0.5 * unanchored_step, rtol=2.0e-6, atol=1.0e-7
    )
    for observed, expected in zip((values, labels, centers, owners, weights), state):
        np.testing.assert_array_equal(observed, expected)
    assert not anchored.flags.writeable
    assert not np.shares_memory(anchored, centers)


def test_crossfit_oi_lvq_is_deterministic_with_original_labels() -> None:
    rng = np.random.default_rng(53)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 20)
    values = np.vstack(
        [
            rng.normal(loc=location, scale=0.7, size=(20, 10))
            for location in (0.0, 1.0, 2.0)
        ]
    ).astype(np.float32)

    first = crossfit_redundancy_state(
        values, labels, seed=143, max_square_features=5
    )
    second = crossfit_redundancy_state(
        values, labels, seed=143, max_square_features=5
    )

    assert set(first) == set(EPOCHS)
    for epoch in EPOCHS:
        assert first[epoch]["exact_oi"] == second[epoch]["exact_oi"]
        assert first[epoch]["second_own_win_rate"] == second[epoch][
            "second_own_win_rate"
        ]
        for left, right in zip(first[epoch]["folds"], second[epoch]["folds"]):
            assert left["lift_state_sha256"] == right["lift_state_sha256"]
            assert left["exact_oi"] == right["exact_oi"]
            assert left["last_update_diagnostics"] == right[
                "last_update_diagnostics"
            ]
