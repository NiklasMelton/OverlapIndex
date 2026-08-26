"""Focused tests for the private Food-101 conditioned replay plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.nuisance_conditioned_distance import food101


def test_oi_kwargs_are_fresh_nested_mappings_with_paired_seed() -> None:
    first = food101._oi_kwargs({0: 2, 1: 2}, 41)
    second = food101._oi_kwargs({0: 2, 1: 2}, 42)

    assert first is not second
    assert first["kmeans_kwargs"] is not second["kmeans_kwargs"]
    assert first["kmeans_k"] is not second["kmeans_k"]
    assert first["kmeans_kwargs"]["random_state"] == 41
    assert second["kmeans_kwargs"]["random_state"] == 42
    first["kmeans_kwargs"]["random_state"] = -1
    assert second["kmeans_kwargs"]["random_state"] == 42


def test_stratified_cap_is_deterministic_and_balanced() -> None:
    labels = np.repeat(np.arange(4), 100)
    first = food101._stratified_cap(labels, 203, seed=17)
    second = food101._stratified_cap(labels, 203, seed=17)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 203
    counts = np.unique(labels[first], return_counts=True)[1]
    assert int(np.max(counts) - np.min(counts)) <= 1


def test_repository_code_identity_ignores_generated_output(tmp_path) -> None:
    before = food101._repository_provenance()["code_identity_sha256"]
    output = tmp_path / "results" / "food101"
    output.mkdir(parents=True)
    (output / "raw_results.json").write_text('{"generated": true}\n', encoding="utf-8")
    (output / "checkpoint.json").write_text('{"resume": true}\n', encoding="utf-8")
    after = food101._repository_provenance()["code_identity_sha256"]
    assert after == before


def test_guardrail_trigger_is_panel_wide_and_uses_capped_probe() -> None:
    selector_rows = []
    probe_rows = []
    for model, b_score, e_score, refinement_rate, probe_score in (
        ("model-a", 0.90, 0.88, 0.10, 0.72),
        ("model-b", 0.80, 0.73, 0.20, 0.81),
    ):
        for candidate, score in (("B", b_score), ("C", e_score + 0.01), ("E", e_score)):
            selector_rows.append(
                {
                    "status": "ok",
                    "candidate_id": candidate,
                    "model": model,
                    "replicate": 0,
                    "arm": "nuisance_full",
                    "budget": 64,
                    "score": score,
                    "refinement": {"applied_rate": refinement_rate},
                    "outer_wall_seconds": 1.0,
                    "outer_cpu_seconds": 0.5,
                }
            )
        probe_rows.append(
            {
                "model": model,
                "replicate": 0,
                "arm": "nuisance_full",
                "budget": 64,
                "score": probe_score,
                "wall_seconds": 2.0,
                "cpu_seconds": 1.0,
            }
        )

    rows = food101._guardrail_policy_rows(
        selector_rows, probe_rows, promoted_candidate="C"
    )
    assert len(rows) == 2
    assert all(row["panel_triggered"] is True for row in rows)
    assert all(row["selected_score_source"] == "capped_linear_probe" for row in rows)
    assert {row["score"] for row in rows} == {0.72, 0.81}
    assert all(row["policy_wall_seconds"] == 4.0 for row in rows)


def test_guardrail_untriggered_panel_uses_promoted_oi() -> None:
    selector_rows = [
        {
            "status": "ok",
            "candidate_id": candidate,
            "model": "model-a",
            "replicate": 1,
            "arm": "baseline",
            "budget": 80,
            "score": score,
            "refinement": {"applied_rate": 0.10},
            "outer_wall_seconds": 1.0,
            "outer_cpu_seconds": 0.5,
        }
        for candidate, score in (("B", 0.90), ("D", 0.89), ("E", 0.88))
    ]
    probe_rows = [
        {
            "model": "model-a",
            "replicate": 1,
            "arm": "baseline",
            "budget": 80,
            "score": 0.75,
            "wall_seconds": 2.0,
            "cpu_seconds": 1.0,
        }
    ]
    rows = food101._guardrail_policy_rows(
        selector_rows, probe_rows, promoted_candidate="D"
    )
    assert rows[0]["panel_triggered"] is False
    assert rows[0]["selected_score_source"] == "D"
    assert rows[0]["score"] == 0.89
    assert rows[0]["policy_wall_seconds"] == 3.0


def test_candidate_refinement_is_strict_bool() -> None:
    generator = np.random.default_rng(2)
    matrix = np.vstack(
        [generator.normal(-1.0, 0.1, size=(15, 3)), generator.normal(1.0, 0.1, size=(15, 3))]
    ).astype(np.float32)
    labels = np.repeat([0, 1], 15)
    bad = ("C", "bad", "global_isotropy", np.bool_(True), False)

    with pytest.raises(TypeError, match="strict Python bool"):
        food101._cross_fitted_score(matrix, labels, candidate=bad, seed=3)


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0] * 10 + [1] * 10, dtype=object),
        np.asarray(["class-b"] * 10 + ["class-a"] * 10, dtype=object),
    ],
)
def test_object_labels_use_deterministic_integer_stratification_only(
    labels: np.ndarray,
) -> None:
    encoded, classes = food101._stratification_encoding(labels)
    assert encoded.dtype == np.int64
    assert classes == (labels[0], labels[10])

    first = food101._stratified_folds(labels, n_splits=5, seed=19)
    second = food101._stratified_folds(labels.copy(), n_splits=5, seed=19)
    assert len(first) == len(second) == 5
    for (train_a, holdout_a), (train_b, holdout_b) in zip(first, second):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(holdout_a, holdout_b)
        assert set(encoded[holdout_a].tolist()) == {0, 1}


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0] * 20 + [1] * 20, dtype=object),
        np.asarray(["first"] * 20 + ["second"] * 20, dtype=object),
    ],
)
def test_direct_cross_fit_accepts_archived_object_label_forms(
    labels: np.ndarray,
) -> None:
    generator = np.random.default_rng(13)
    matrix = np.vstack(
        [
            generator.normal(-1.0, 0.2, size=(20, 4)),
            generator.normal(1.0, 0.2, size=(20, 4)),
        ]
    ).astype(np.float32)

    first = food101._cross_fitted_score(
        matrix, labels, candidate=food101.CANDIDATES[0], seed=23
    )
    second = food101._cross_fitted_score(
        matrix, labels.copy(), candidate=food101.CANDIDATES[0], seed=23
    )
    assert first["score"] == second["score"]
    assert [fold["seed"] for fold in first["folds"]] == [23, 24, 25, 26, 27]


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0] * 20 + [1] * 20, dtype=object),
        np.asarray(["first"] * 20 + ["second"] * 20, dtype=object),
    ],
)
def test_capped_probe_accepts_object_label_forms(labels: np.ndarray) -> None:
    generator = np.random.default_rng(31)
    matrix = np.vstack(
        [
            generator.normal(-1.0, 0.25, size=(20, 4)),
            generator.normal(1.0, 0.25, size=(20, 4)),
        ]
    ).astype(np.float32)
    first = food101._capped_probe(matrix, labels, seed=37)
    second = food101._capped_probe(matrix, labels.copy(), seed=37)
    assert first["score"] == second["score"]
    assert first["n_rows"] == second["n_rows"] == 40


def test_direct_a_cross_fit_is_deterministic() -> None:
    generator = np.random.default_rng(3)
    matrix = np.vstack(
        [generator.normal(-1.0, 0.2, size=(20, 4)), generator.normal(1.0, 0.2, size=(20, 4))]
    ).astype(np.float32)
    labels = np.repeat([0, 1], 20)
    candidate = food101.CANDIDATES[0]

    first = food101._cross_fitted_score(matrix, labels, candidate=candidate, seed=7)
    second = food101._cross_fitted_score(matrix, labels, candidate=candidate, seed=7)

    assert first["score"] == second["score"]
    assert first["refinement"] == second["refinement"]
    assert first["candidate_id"] == "A"
    assert first["prototype_refinement"] is False


def test_food_determinism_signature_excludes_clocks_but_not_state() -> None:
    base = {
        "candidate_id": "C",
        "candidate_name": "conditioned",
        "mode": "global_isotropy",
        "prototype_refinement": True,
        "score": 0.8,
        "refinement": {"applied_count": 2},
        "conditioning": [{"state_sha256": "abc"}],
        "fit_wall_seconds": 1.0,
        "folds": [
            {
                "fold": 0,
                "seed": 42,
                "train_size": 8,
                "holdout_size": 2,
                "score": 0.75,
                "fit_wall_seconds": 0.5,
                "score_fixed_cpu_seconds": 0.1,
                "refinement": {"applied_count": 1},
                "conditioning": {"state_sha256": "fold"},
                "conditioning_runtime": {
                    "conditioning_fit_wall_seconds": 0.2
                },
            }
        ],
    }
    repeated = {
        **base,
        "fit_wall_seconds": 9.0,
        "folds": [{**base["folds"][0], "fit_wall_seconds": 8.0}],
    }
    comparison = food101._cross_fit_determinism_comparison(base, repeated)
    assert comparison["exact"] is True
    assert comparison["runtime_fields_excluded"] is True

    changed = {**repeated, "score": 0.7}
    assert food101._cross_fit_determinism_comparison(base, changed)["exact"] is False
