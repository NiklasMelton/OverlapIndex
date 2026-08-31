"""Focused tests for the frozen continuous-OI null diagnostic."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from experiments.continuous_null_diagnostic import run
from experiments.continuous_prototype_capacity import run as capacity
from experiments.continuous_save_ranking import run as save


def test_protocol_closes_geometry_fold_and_nested_null_policies():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    assert list(run._method_specs(protocol)) == [
        "save8_k4",
        "rff64_mean8_k4",
        "save_rffmean_k4",
        "save_rffmean_residual_k4",
    ]
    assert run.POLICY_COUNTS == {
        "candidate4": 4,
        "common4": 4,
        "common16": 16,
        "common32": 32,
    }
    assert protocol["fold_policy"]["random_state"] == "panel seed"
    assert protocol["null_policies"]["nested_common_bank"] is True
    assert protocol["continuous_oi"]["prototypes_per_cell"] == 4


def test_parent_artifact_and_regenerated_candidate_identity_match():
    protocol = run._load_protocol()
    raw = run._load_parent_raw(protocol)
    parents = run._parent_panel_map(raw)
    panel = capacity.generate_panel(protocol, 1009, "triangle_wave")
    parent = parents[(1009, "triangle_wave")]

    assert [row["candidate_id"] for row in panel["candidates"]] == [
        row["candidate_id"] for row in parent["candidates"]
    ]
    for left, right in zip(panel["candidates"], parent["candidates"]):
        assert left["quality"] == right["quality"]
        assert left["nuisance_scale"] == right["nuisance_scale"]
        assert left["effective_nuisance_scale"] == right["effective_nuisance_scale"]


def test_common_bank_is_shared_nested_and_candidate_bank_is_distinct():
    common_a = run._permutation_bank(31, 32, run._common_bank_seed(1009, 2))
    common_b = run._permutation_bank(31, 32, run._common_bank_seed(1009, 2))
    candidate = run._permutation_bank(
        31,
        4,
        run._candidate_fit_seed(1009, 3, 2),
    )

    for left, right in zip(common_a, common_b):
        np.testing.assert_array_equal(left, right)
    assert not all(
        np.array_equal(left, right)
        for left, right in zip(common_a[:4], candidate)
    )
    assert len(common_a[:4]) == 4
    assert len(common_a[:16]) == 16
    assert len(common_a) == 32


def _fitted_first_fold():
    protocol = run._load_protocol()
    panel = capacity.generate_panel(protocol, 1009, "triangle_wave")
    candidate = panel["candidates"][0]
    train, holdout = next(
        iter(KFold(n_splits=4, shuffle=True, random_state=1009).split(candidate["X_train"]))
    )
    fit_seed = run._candidate_fit_seed(1009, 0, 0)
    transformed_train, transformed_holdout, _ = run.multiview._prepare_fold_geometry(
        run._method_specs(protocol)["save8_k4"],
        candidate["X_train"][train],
        candidate["X_train"][holdout],
        candidate["primitive_train"][train],
        candidate["primitive_train"][holdout],
        panel["y_train"][train],
        fit_seed,
    )
    estimator = clone(save._base_selector(fit_seed))
    estimator._fit_geometry_only(transformed_train, panel["y_train"][train])
    return estimator, transformed_holdout, panel["y_train"][holdout], fit_seed


def test_explicit_candidate_bank_is_bit_exact_with_public_score_fixed():
    estimator, X, y, fit_seed = _fitted_first_fold()
    bank = run._permutation_bank(len(y), 4, fit_seed)
    explicit = run._fixed_model_components(estimator, X, y, bank)

    score = estimator.score_fixed(X, y)

    assert explicit["actual_loss"] == estimator.actual_loss_
    assert explicit["null_loss_samples"] == estimator.null_loss_samples_
    derived = run._score_from_losses(
        explicit["actual_loss"],
        float(np.mean(explicit["null_loss_samples"])),
    )
    assert derived["score"] == score
    assert derived["loss_ratio"] == estimator.loss_ratio_


def test_explicit_null_diagnostic_does_not_mutate_fitted_or_score_state():
    estimator, X, y, fit_seed = _fitted_first_fold()
    centers_before = np.array(estimator._model.centers, copy=True)
    index_before = estimator.index
    actual_before = estimator.actual_loss_
    null_before = estimator.null_loss_

    run._fixed_model_components(
        estimator,
        X,
        y,
        run._permutation_bank(len(y), 32, fit_seed + 99),
    )

    np.testing.assert_array_equal(estimator._model.centers, centers_before)
    assert estimator.index == index_before
    assert estimator.actual_loss_ == actual_before
    assert estimator.null_loss_ == null_before


def test_selector_holds_actual_loss_fixed_across_all_null_policies():
    protocol = run._load_protocol()
    panel = capacity.generate_panel(protocol, 1103, "sinusoid")
    candidate = panel["candidates"][0]
    result = run.evaluate_selector(
        run._method_specs(protocol)["save8_k4"],
        candidate["X_train"],
        candidate["primitive_train"],
        panel["y_train"],
        1103,
        0,
    )

    assert len(result["folds"]) == 4
    for fold in result["folds"]:
        assert set(fold["policies"]) == set(run.POLICY_COUNTS)
        for policy, count in run.POLICY_COUNTS.items():
            row = fold["policies"][policy]
            assert row["permutation_count"] == count
            assert len(row["null_loss_samples"]) == count
            assert np.isfinite(row["null_loss"])
            assert np.isfinite(row["score"])


def test_decision_distinguishes_variance_repair_from_estimand_mismatch():
    protocol = run._load_protocol()

    def metric(periodic, regret=0.10, preserved=0.60):
        by_family = {
            family: {
                "mean_spearman": periodic if family in {"sinusoid", "triangle_wave"} else preserved
            }
            for family in protocol["families"]
        }
        return {"mean_regret_r2": regret, "by_family": by_family}

    repairing = {
        method_id: {
            "candidate4_oi": metric(0.20),
            "common32_oi": metric(0.40),
            "negative_actual_loss": metric(0.45),
        }
        for method_id in run._method_specs(protocol)
    }
    repaired = run._decision(protocol, repairing)
    assert repaired["conclusion"] == "null_variance_repair_supported"

    mismatch = {
        method_id: {
            "candidate4_oi": metric(0.20),
            "common32_oi": metric(0.22),
            "negative_actual_loss": metric(0.55),
        }
        for method_id in run._method_specs(protocol)
    }
    mismatched = run._decision(protocol, mismatch)
    assert mismatched["conclusion"] == "null_estimand_mismatch_supported"
    assert set(mismatched["estimand_mismatch_methods"]) == set(
        run._method_specs(protocol)
    )
