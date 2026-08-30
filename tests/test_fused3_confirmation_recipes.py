from __future__ import annotations

import numpy as np
import pytest

from experiments.fused3_confirmation_v2 import recipes


def _toy(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), 20)
    centers = np.eye(3, 6, dtype=np.float32) * np.float32(3.0)
    train = centers[labels] + rng.normal(0.0, 0.2, size=(len(labels), 6)).astype(
        np.float32
    )
    test_labels = np.repeat(np.arange(3), 5)
    test = centers[test_labels] + rng.normal(
        0.0, 0.2, size=(len(test_labels), 6)
    ).astype(np.float32)
    return train, labels, test, test_labels


def test_closed_recipe_surface_and_hashes() -> None:
    assert recipes.DATASET_IDS == (
        "torchvision_cifar10",
        "torchvision_stl10",
        "torchvision_gtsrb",
        "torchvision_fgvc_aircraft",
        "torchvision_dtd",
    )
    assert recipes.BUDGETS == (32, 64)
    assert recipes.REPLICATE_SEEDS == tuple(range(2026082710, 2026082715))
    assert recipes.EVALUATION_SEED == 2026082709
    assert recipes.HEADS == ("linear", "quadratic", "knn", "rbf")
    for candidate in recipes.SELECTORS:
        recipe = recipes.candidate_recipe(candidate)
        assert recipe["candidate_id"] == candidate
        assert len(recipes.payload_sha256(recipe)) == 64
    for head in recipes.HEADS:
        recipe = recipes.head_recipe(head)
        assert recipe["head"] == head
        assert len(recipes.payload_sha256(recipe)) == 64
    with pytest.raises(ValueError, match="must be one of"):
        recipes.candidate_recipe("unknown")
    with pytest.raises(ValueError, match="must be one of"):
        recipes.head_recipe("unknown")


def test_method_order_is_exactly_balanced_for_frozen_grid() -> None:
    counts = {
        candidate: [0, 0]
        for candidate in recipes.SELECTORS
    }
    for panel in range(500):
        order = recipes.method_order(panel)
        assert set(order) == set(recipes.SELECTORS)
        for position, candidate in enumerate(order):
            counts[candidate][position] += 1
    assert set(tuple(value) for value in counts.values()) == {(250, 250)}


def test_reference_heads_use_training_rows_and_separate_evaluation_rows() -> None:
    train, labels, test, test_labels = _toy()
    for index, head in enumerate(recipes.HEADS):
        result = recipes.execute_reference_head(
            head,
            train,
            labels,
            test,
            test_labels,
            seed=100 + index,
        )
        assert result["head"] == head
        assert 0.0 <= result["test_accuracy"] <= 1.0
        assert result["training_row_count"] == len(train)
        assert result["evaluation_row_count"] == len(test)
        assert result["recipe"] == recipes.head_recipe(head)


def test_fused3_and_probe_adapters_are_deterministic_and_closed() -> None:
    train, labels, _test, _test_labels = _toy()
    for candidate in recipes.SELECTORS:
        first = recipes.execute_selector(candidate, train, labels, seed=901)
        second = recipes.execute_selector(candidate, train, labels, seed=901)
        assert first["candidate_id"] == candidate
        assert first["score"] == second["score"]
        assert first["candidate_recipe"] == recipes.candidate_recipe(candidate)
        assert first["candidate_recipe_sha256"] == recipes.payload_sha256(
            recipes.candidate_recipe(candidate)
        )
        if candidate == "FUSED3":
            assert all(
                fold["state_unchanged_after_score_fixed"] is True
                for fold in first["folds"]
            )


def test_invalid_or_nonfinite_inputs_fail_closed() -> None:
    train, labels, test, test_labels = _toy()
    broken = train.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        recipes.execute_selector("FUSED3", broken, labels, seed=1)
    with pytest.raises(ValueError, match="matching feature"):
        recipes.execute_reference_head(
            "linear", train, labels, test[:, :-1], test_labels, seed=1
        )
    with pytest.raises(TypeError, match="seed"):
        recipes.execute_selector("FUSED3", train, labels, seed=True)
