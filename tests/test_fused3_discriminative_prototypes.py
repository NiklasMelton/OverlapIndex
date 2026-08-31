from __future__ import annotations

import numpy as np

from experiments.fused3_envelope_reanalysis.discriminative_prototypes import (
    EPOCHS,
    contrastive_lvq_epoch,
    crossfit_discriminative_state,
)
from experiments.scalable_relevance_kmeans.scalable import _score_kernel


def _nearest_margin(
    values: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    owners: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    scores = _score_kernel(values, centers, weights)
    output = np.empty(len(values), dtype=np.float32)
    for position, label in enumerate(labels):
        own = np.flatnonzero(owners == label)
        wrong = np.flatnonzero(owners != label)
        output[position] = np.max(scores[position, own]) - np.max(
            scores[position, wrong]
        )
    return output


def test_lvq_moves_own_toward_and_rival_away_on_violations() -> None:
    values = np.asarray([[0.0], [2.0]], dtype=np.float32)
    labels = np.asarray([0, 1])
    centers = np.asarray([[1.0], [0.0]], dtype=np.float32)
    owners = np.asarray([0, 1])
    weights = np.asarray([1.0])
    before = _nearest_margin(values, labels, centers, owners, weights)

    updated, diagnostics = contrastive_lvq_epoch(
        values,
        labels,
        centers=centers,
        owners=owners,
        weights=weights,
        learning_rate=0.1,
    )
    after = _nearest_margin(values, labels, updated, owners, weights)

    assert np.all(before < 0.0)
    assert np.all(after > before)
    assert updated[0, 0] < centers[0, 0]
    assert updated[1, 0] > centers[1, 0]
    assert diagnostics["active_violation_count"] == 2
    assert diagnostics["changed_prototype_count"] == 2
    assert diagnostics["weights_relearned"] is False
    assert diagnostics["prototype_owners_changed"] is False


def test_lvq_is_exact_noop_when_every_row_is_correct() -> None:
    values = np.asarray([[-1.0], [1.0]], dtype=np.float32)
    labels = np.asarray([0, 1])
    centers = values.copy()
    owners = labels.copy()
    weights = np.asarray([2.0])

    updated, diagnostics = contrastive_lvq_epoch(
        values,
        labels,
        centers=centers,
        owners=owners,
        weights=weights,
        learning_rate=0.1,
    )

    np.testing.assert_array_equal(updated, centers)
    assert diagnostics["active_violation_count"] == 0
    assert diagnostics["changed_prototype_count"] == 0


def test_lvq_returns_detached_state_and_preserves_all_inputs() -> None:
    values = np.asarray([[0.0], [2.0]], dtype=np.float32)
    labels = np.asarray([0, 1])
    centers = np.asarray([[1.0], [0.0]], dtype=np.float32)
    owners = np.asarray([0, 1])
    weights = np.asarray([1.0])
    state = tuple(array.copy() for array in (values, labels, centers, owners, weights))

    updated, _diagnostics = contrastive_lvq_epoch(
        values,
        labels,
        centers=centers,
        owners=owners,
        weights=weights,
        learning_rate=0.1,
    )

    for observed, expected in zip((values, labels, centers, owners, weights), state):
        np.testing.assert_array_equal(observed, expected)
    assert not updated.flags.writeable
    assert not np.shares_memory(updated, centers)


def test_crossfit_lvq_epochs_share_initial_state_and_are_deterministic() -> None:
    rng = np.random.default_rng(47)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 20)
    values = np.vstack(
        [
            rng.normal(loc=location, scale=0.7, size=(20, 10))
            for location in (0.0, 1.0, 2.0)
        ]
    ).astype(np.float32)

    first = crossfit_discriminative_state(
        values, labels, seed=143, max_square_features=5
    )
    second = crossfit_discriminative_state(
        values, labels, seed=143, max_square_features=5
    )

    assert set(first) == set(EPOCHS)
    for epoch in EPOCHS:
        assert first[epoch]["exact_oi"] == second[epoch]["exact_oi"]
        assert first[epoch]["best_own_win_rate"] == second[epoch][
            "best_own_win_rate"
        ]
        assert len(first[epoch]["folds"]) == 5
        for left, right in zip(first[epoch]["folds"], second[epoch]["folds"]):
            assert left["lift_state_sha256"] == right["lift_state_sha256"]
            assert left["exact_oi"] == right["exact_oi"]
            assert left["last_update_diagnostics"] == right[
                "last_update_diagnostics"
            ]
    assert all(
        row["last_update_diagnostics"] is None for row in first[0]["folds"]
    )
    assert all(
        row["last_update_diagnostics"] is not None for row in first[1]["folds"]
    )
