import json

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from experiments.fused3_confirmation_v2 import statistics
from experiments.fused3_envelope_reanalysis import (
    fisher_metric_aggregation_factorial as factorial,
)
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen


def _toy(seed=7):
    rng = np.random.default_rng(seed)
    values = np.vstack(
        [
            rng.normal((-2.0, 0.0, 0.0, 0.0), 0.45, size=(20, 4)),
            rng.normal((0.0, 2.0, 0.0, 0.0), 0.50, size=(20, 4)),
            rng.normal((2.0, 0.0, 0.0, 0.0), 0.55, size=(20, 4)),
        ]
    ).astype(np.float32)
    labels = np.asarray(["left"] * 20 + ["top"] * 20 + ["right"] * 20)
    return values, labels


def test_closed_factorial_surface_and_protocol():
    assert factorial.FACTORIAL_METHODS == (
        "FUSED3-RAW-OI-U",
        "FUSED3-RAW-ACC-U",
        "FUSED3-RAW-OI-R",
        "FUSED3-RAW-ACC-R",
        "FUSED3-FISHER-OI-U",
        "FUSED3-FISHER-ACC-U",
        "FUSED3-FISHER-OI-R",
        "FUSED3-FISHER-ACC-R",
    )
    digest, protocol = factorial.validate_design_protocol()
    assert len(digest) == 64
    assert protocol["factorial"]["candidate_ids"] == list(
        factorial.FACTORIAL_METHODS
    )
    assert protocol["scope"]["promotion_or_reselection_allowed"] is False


def test_shared_fisher_matches_sklearn_svd_and_is_query_independent():
    values, labels = _toy()
    fitted = factorial.fit_shared_fisher(values, labels)
    classes, encoded = factorial._first_observed_encoding(labels)
    reference = LinearDiscriminantAnalysis(
        solver="svd", n_components=None, tol=1.0e-4
    ).fit(values, encoded)
    observed = fitted.transform(values)
    expected = np.asarray(reference.transform(values), dtype=np.float32)
    assert fitted.classes == classes
    assert observed.shape == expected.shape == (60, 2)
    assert np.allclose(observed, expected, rtol=2e-5, atol=2e-5)
    state = fitted.state_sha256
    shifted_query = values + np.float32(100.0)
    fitted.transform(shifted_query)
    assert fitted.state_sha256 == state
    assert not fitted.center.flags.writeable
    assert not fitted.scaling.flags.writeable


def test_fisher_label_encoding_does_not_rewrite_original_labels():
    values, _labels = _toy()
    labels = np.asarray([1] * 20 + ["two"] * 20 + [3.5] * 20, dtype=object)
    before = labels.copy()
    fitted = factorial.fit_shared_fisher(values, labels)
    assert np.array_equal(labels, before)
    assert fitted.classes == (1, "two", 3.5)
    assert fitted.transform(values).shape == (60, 2)


def test_prototype_accuracy_is_hand_counted_and_rejects_bad_weights():
    values = np.asarray([[-1.0, 0.0], [0.8, 0.0], [2.2, 0.0]])
    target = np.asarray(["a", "a", "b"])
    centers = np.asarray([[-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    owners = np.asarray(["a", "a", "b"])
    assert factorial.prototype_accuracy(
        values,
        target,
        centers=centers,
        owners=owners,
        weights=np.ones(2),
    ) == 1.0
    with pytest.raises(ValueError, match="positive"):
        factorial.prototype_accuracy(
            values,
            target,
            centers=centers,
            owners=owners,
            weights=np.asarray([1.0, 0.0]),
        )


def test_crossfit_executes_all_eight_cells_and_raw_control_is_exact():
    values, labels = _toy()
    result = factorial.crossfit_factorial(values, labels, seed=31)
    assert set(result["scores"]) == set(factorial.FACTORIAL_METHODS)
    assert set(result["method_wall_seconds"]) == set(factorial.FACTORIAL_METHODS)
    assert len(result["folds"]) == 5
    assert all(np.isfinite(value) for value in result["scores"].values())
    assert all(value >= 0.0 for value in result["method_wall_seconds"].values())
    assert all(row["raw_fitted_state_unchanged"] for row in result["folds"])
    assert all(row["fisher_fitted_state_unchanged"] for row in result["folds"])
    assert all(row["fisher_output_dimension"] == 2 for row in result["folds"])

    normalized = food101._row_l2(values)
    direct = []
    for fold, (train, holdout) in enumerate(
        food101._stratified_folds(labels, n_splits=5, seed=31)
    ):
        k_per_class = fused_food_screen._k_per_class(labels, train)
        selector = fused_food_screen._build_selector("FUSED3", k_per_class, 31 + fold)
        selector.fit(normalized[train], labels[train])
        direct.append(selector.score_fixed(normalized[holdout], labels[holdout]))
    assert result["scores"]["FUSED3-RAW-OI-U"] == float(np.mean(direct))


def test_factorial_effects_recover_additive_regret_and_zero_interactions():
    rows = []
    for dataset_position, dataset in enumerate(statistics.DATASET_IDS):
        for seed in statistics.REPLICATE_SEEDS:
            for budget in statistics.BUDGETS:
                base = float(dataset_position) + (seed % 3) * 0.01 + budget * 0.001
                for method, factors in factorial.FACTOR_BY_METHOD.items():
                    regret = base
                    regret += 1.0 if factors["metric"] == "fisher" else 0.0
                    regret += 2.0 if factors["aggregation"] == "prototype_accuracy" else 0.0
                    regret += 3.0 if factors["refined"] else 0.0
                    rows.append(
                        {
                            "dataset_id": dataset,
                            "replicate_seed": seed,
                            "budget": budget,
                            "candidate_id": method,
                            "regret_pp": regret,
                        }
                    )
    effects = factorial._factorial_effects(rows)
    simple = {
        row["effect_type"]: [] for row in effects if "simple_effect" in row["effect_type"]
    }
    for row in effects:
        if row["effect_type"] in simple:
            simple[row["effect_type"]].append(row["estimate"])
    assert simple["metric_simple_effect"] == pytest.approx([1.0] * 4)
    assert simple["aggregation_simple_effect"] == pytest.approx([2.0] * 4)
    assert simple["refinement_simple_effect"] == pytest.approx([3.0] * 4)
    interactions = [
        row["estimate"]
        for row in effects
        if "simple_effect" not in row["effect_type"]
    ]
    assert interactions == pytest.approx([0.0, 0.0, 0.0])


def test_report_names_factor_cells_and_retrospective_scope():
    summary = {
        "overall_aggregate": [
            {
                "candidate_id": "FUSED3-FISHER-ACC-R",
                "equal_dataset_mean_regret_pp": 0.5,
                "equal_dataset_exact_best_rate": 0.7,
                "equal_dataset_within_one_pp_rate": 0.8,
                "equal_dataset_mean_spearman": 0.9,
            }
        ],
        "factorial_effects": [],
        "dataset_aggregate": [],
        "aircraft_focus": [],
        "runtime_descriptive": [],
    }
    report = factorial.render_report(summary)
    assert "FUSED3-FISHER-ACC-R" in report
    assert "post-outcome development diagnostic only" in report
    assert "does not promote" in report
