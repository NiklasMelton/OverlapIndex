from __future__ import annotations

import numpy as np

from experiments.scalable_relevance_kmeans import fused_toy


def test_fused_toy_protocol_matches_implementation() -> None:
    assert len(fused_toy._validate_protocol()) == 64


def test_looped_and_fused_crossfit_scores_are_exact() -> None:
    candidate = fused_toy.base.PANEL_CANDIDATES[0]
    X, y = fused_toy.base.generate_embedding(
        scenario="nonlinear_nuisance",
        seed=fused_toy.SEED,
        split="selector",
        n_per_class=32,
        candidate=candidate,
    )
    looped, _ = fused_toy._crossfit(
        X, y, method="looped_fixed3", seed=fused_toy.SEED
    )
    fused, _ = fused_toy._crossfit(
        X, y, method="fused_fixed3", seed=fused_toy.SEED
    )
    assert fused == looped


def test_small_food_shaped_profile_is_complete(monkeypatch) -> None:
    monkeypatch.setattr(fused_toy, "PROFILE_FEATURES", (32,))
    monkeypatch.setattr(fused_toy, "PROFILE_REPEATS", 2)
    rows = fused_toy._prototype_benchmark()
    assert len(rows) == 1
    row = rows[0]
    assert row["feature_count"] == 32
    assert row["max_center_abs_difference"] == 0.0
    assert set(row["median_wall_seconds"]) == {
        "current_sklearn",
        "looped_fixed3",
        "fused_fixed3",
    }
    assert all(np.isfinite(value) and value > 0.0 for value in row["median_wall_seconds"].values())
