import numpy as np
import pytest

from experiments.fused3_envelope_reanalysis import metric_aggregation_2x2 as diag


def test_residual_metric_shared_control_matches_identical_class_residuals():
    centers = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    owners = np.asarray([0, 1])
    values = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [11.0, 10.0],
            [9.0, 10.0],
            [10.0, 11.0],
            [10.0, 9.0],
        ],
        dtype=np.float32,
    )
    target = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    result = diag.estimate_residual_metrics(
        values,
        target,
        centers=centers,
        owners=owners,
        weights=np.ones(2, dtype=np.float64),
    )
    assert np.array_equal(result["class_counts"], np.asarray([4, 4]))
    assert np.allclose(
        result["class_variance"],
        np.broadcast_to(result["shared_variance"], (2, 2)),
        rtol=0.0,
        atol=0.0,
    )
    assert result["prior_rows"] == 32.0


def test_gaussian_prototype_scores_match_manual_energy():
    values = np.asarray([[1.0, -2.0], [0.5, 0.25]], dtype=np.float32)
    centers = np.asarray([[0.25, -1.0], [2.0, 1.0]], dtype=np.float32)
    owners = np.asarray(["a", "b"])
    weights = np.asarray([1.5, 0.5], dtype=np.float64)
    variance = np.asarray([[2.0, 0.25]], dtype=np.float64)
    observed = diag.gaussian_prototype_scores(
        values,
        centers=centers,
        owners=owners,
        weights=weights,
        classes=("a", "b"),
        variances=variance,
    )
    expected = np.empty_like(observed)
    for row, value in enumerate(values.astype(np.float64)):
        for column, center in enumerate(centers.astype(np.float64)):
            effective_variance = variance[0] / weights
            expected[row, column] = -0.5 * (
                np.sum((value - center) ** 2 / effective_variance)
                + np.sum(np.log(effective_variance))
            )
    assert np.allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_exact_oi_and_discriminant_are_hand_counted_and_strict():
    owners = np.asarray([0, 0, 1, 1])
    target = np.asarray([0, 1])
    scores = np.asarray([[5.0, 4.0, 3.0, 2.0], [3.0, 2.0, 5.0, 4.0]])
    oi, details = diag.exact_oi_from_scores(scores, target, owners=owners)
    discriminant = diag.discriminant_summary_from_scores(
        scores, target, owners=owners
    )
    assert oi == 1.0
    assert details["pair_hit_total"] == 0
    assert discriminant["accuracy"] == 1.0
    assert discriminant["mean_correct_minus_best_wrong_margin"] == 2.0

    tied_rival = scores.copy()
    tied_rival[0, 2] = 4.0
    tied_oi, tied_details = diag.exact_oi_from_scores(
        tied_rival, target, owners=owners
    )
    assert tied_oi == 1.0
    assert tied_details["pair_hit_total"] == 0


def test_class_conditional_metric_can_help_accuracy_while_oi_is_unchanged():
    values = np.asarray([[0.1], [-0.1], [3.0], [-3.0]], dtype=np.float32)
    target = np.asarray([0, 0, 1, 1])
    centers = np.zeros((4, 1), dtype=np.float32)
    owners = np.asarray([0, 0, 1, 1])
    weights = np.ones(1, dtype=np.float64)
    shared = diag.gaussian_prototype_scores(
        values,
        centers=centers,
        owners=owners,
        weights=weights,
        classes=(0, 1),
        variances=np.asarray([[2.0]]),
    )
    conditioned = diag.gaussian_prototype_scores(
        values,
        centers=centers,
        owners=owners,
        weights=weights,
        classes=(0, 1),
        variances=np.asarray([[0.1], [4.0]]),
    )
    shared_accuracy = diag.discriminant_summary_from_scores(
        shared, target, owners=owners
    )["accuracy"]
    class_accuracy = diag.discriminant_summary_from_scores(
        conditioned, target, owners=owners
    )["accuracy"]
    shared_oi = diag.exact_oi_from_scores(shared, target, owners=owners)[0]
    class_oi = diag.exact_oi_from_scores(conditioned, target, owners=owners)[0]
    assert shared_accuracy == 0.5
    assert class_accuracy == 1.0
    assert shared_oi == class_oi == 1.0


def test_crossfit_emits_complete_2x2_and_protects_fitted_state():
    rng = np.random.default_rng(7)
    values = np.vstack(
        [
            rng.normal(loc=(-1.0, 0.0, 0.0, 0.0), scale=0.2, size=(25, 4)),
            rng.normal(loc=(1.0, 0.0, 0.0, 0.0), scale=0.7, size=(25, 4)),
        ]
    ).astype(np.float32)
    target = np.asarray(["left"] * 25 + ["right"] * 25)
    result = diag.crossfit_metric_aggregation_2x2(values, target, seed=19)
    assert set(result["scores"]) == {
        diag.FROZEN,
        diag.SHARED_OI,
        diag.CLASS_OI,
        diag.SHARED_ACC,
        diag.CLASS_ACC,
    }
    assert len(result["folds"]) == 5
    assert all(row["state_unchanged_after_all_diagnostics"] for row in result["folds"])
    assert all(row["variance_prior_rows"] == 32.0 for row in result["folds"])
    assert all(np.isfinite(value) for value in result["scores"].values())


def test_validation_rejects_bad_variance_scope_and_unknown_labels():
    with pytest.raises(ValueError, match="shared or aligned"):
        diag.gaussian_prototype_scores(
            np.ones((2, 2)),
            centers=np.ones((2, 2)),
            owners=np.asarray([0, 1]),
            weights=np.ones(2),
            classes=(0, 1),
            variances=np.ones((3, 2)),
        )
    with pytest.raises(ValueError, match="class sets must match"):
        diag.exact_oi_from_scores(
            np.ones((2, 2)),
            np.asarray([0, 2]),
            owners=np.asarray([0, 1]),
        )


def test_closed_design_surface():
    assert diag.DIAGNOSTIC_METHODS == (
        "FUSED3-SHARED-DIAG-OI",
        "FUSED3-CLASS-DIAG-OI",
        "FUSED3-SHARED-DIAG-ACC",
        "FUSED3-CLASS-DIAG-ACC",
    )
    assert diag.VARIANCE_PRIOR_ROWS == 32.0
    assert diag.EXPECTED_K_BY_BUDGET == {32: 5, 64: 10}


def test_report_renders_candidate_id_instead_of_template_literal():
    summary = {
        "overall_aggregate": [
            {
                "candidate_id": diag.CLASS_ACC,
                "equal_dataset_mean_regret_pp": 1.0,
                "equal_dataset_exact_best_rate": 0.5,
                "equal_dataset_within_one_pp_rate": 0.7,
                "equal_dataset_mean_spearman": 0.8,
            }
        ],
        "dataset_aggregate": [],
        "aircraft_focus": [],
        "regret_contrasts": [],
        "head_aggregate": [],
        "runtime_descriptive": [],
    }
    report = diag.render_report(summary)
    assert diag.CLASS_ACC in report
    assert "{candidate_id}" not in report
