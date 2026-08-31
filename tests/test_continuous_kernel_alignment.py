"""Focused tests for characteristic-kernel continuous ranking."""

import numpy as np

from experiments.continuous_kernel_alignment import run


def test_protocol_closes_new_seeds_geometries_bandwidths_and_gate():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    assert protocol["seeds"] == [1601, 1709, 1811]
    assert run.BANDWIDTH_MULTIPLIERS == (0.25, 1.0, 4.0)
    assert list(run._method_specs(protocol)) == [
        "rff64_mean8_k4",
        "save_rffmean_k4",
    ]
    assert protocol["kernel_alignment"]["eligible_estimand"] == (
        "null_adjusted_multibandwidth_cka"
    )


def test_motivation_ancestry_hashes_are_exact():
    run._validate_motivation_ancestry(run._load_protocol())


def test_squared_distances_and_median_positive_distance_are_exact():
    X = np.asarray([[0.0], [1.0], [3.0]])
    expected = np.asarray([[0.0, 1.0, 9.0], [1.0, 0.0, 4.0], [9.0, 4.0, 0.0]])

    np.testing.assert_array_equal(run._squared_distances(X), expected)
    assert run._median_positive_squared_distance(X) == 4.0
    assert run._median_positive_squared_distance(np.zeros((3, 2))) > 0.0


def test_rbf_grams_match_frozen_formula_and_have_unit_diagonal():
    X = np.asarray([[0.0], [2.0]])
    grams = run._rbf_grams(X, base_squared_distance=4.0)

    for multiplier in run.BANDWIDTH_MULTIPLIERS:
        expected = np.exp(-4.0 / (2.0 * multiplier * 4.0))
        assert grams[str(multiplier)][0, 1] == expected
        np.testing.assert_array_equal(np.diag(grams[str(multiplier)]), np.ones(2))


def test_double_centering_has_zero_rows_columns_and_is_idempotent():
    gram = np.asarray([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    centered = run._center_gram(gram)

    np.testing.assert_allclose(np.sum(centered, axis=0), 0.0, atol=3.0e-16)
    np.testing.assert_allclose(np.sum(centered, axis=1), 0.0, atol=3.0e-16)
    np.testing.assert_allclose(run._center_gram(centered), centered, atol=3.0e-16)


def test_centered_alignment_is_one_for_identical_nonconstant_gram():
    gram = run._center_gram(np.asarray([[1.0, 0.2], [0.2, 1.0]]))

    assert run._centered_alignment(gram, gram) == 1.0
    assert run._centered_alignment(np.zeros((2, 2)), gram) is None


def test_null_adjustment_has_frozen_boundaries_and_fails_closed():
    assert run._adjust_alignment(0.7, 0.2) == 0.6249999999999999
    assert run._adjust_alignment(0.1, 0.2) == 0.0
    assert run._adjust_alignment(1.0, 0.2) == 1.0
    assert run._adjust_alignment(None, 0.2) is None
    assert run._adjust_alignment(0.2, 1.0) is None


def test_kernel_score_detects_nonlinear_dependence_with_correct_null_direction():
    train_x = np.linspace(-1.0, 1.0, 80)[:, None]
    holdout_x = np.linspace(-0.95, 0.95, 60)[:, None]
    train_y = np.square(train_x)
    holdout_y = np.square(holdout_x)
    bank = run.null_parent._permutation_bank(60, 32, 107)

    result = run._kernel_fold_components(
        train_x, holdout_x, train_y, holdout_y, bank
    )

    assert result["raw_alignment"] > result["null_alignment"]
    assert result["score"] > 0.2
    assert len(result["null_alignment_samples"]) == 32
    assert set(result["feature_bandwidths"]) == {"0.25", "1.0", "4.0"}


def test_kernel_bandwidths_are_fit_from_training_rows_only():
    rng = np.random.default_rng(13)
    train_x = rng.normal(size=(80, 3))
    train_y = rng.normal(size=(80, 1))
    holdout_x = rng.normal(size=(60, 3))
    holdout_y = rng.normal(size=(60, 1))
    bank = run.null_parent._permutation_bank(60, 32, 109)

    first = run._kernel_fold_components(
        train_x, holdout_x, train_y, holdout_y, bank
    )
    second = run._kernel_fold_components(
        train_x, holdout_x * 100.0, train_y, holdout_y * 100.0, bank
    )

    assert first["feature_base_squared_distance"] == second[
        "feature_base_squared_distance"
    ]
    assert first["target_base_squared_distance"] == second[
        "target_base_squared_distance"
    ]


def test_kernel_fold_components_are_deterministic_and_input_nonmutating():
    rng = np.random.default_rng(19)
    train_x = rng.normal(size=(80, 4))
    holdout_x = rng.normal(size=(60, 4))
    train_y = rng.normal(size=(80, 1))
    holdout_y = rng.normal(size=(60, 1))
    copies = [np.array(value, copy=True) for value in (train_x, holdout_x, train_y, holdout_y)]
    bank = run.null_parent._permutation_bank(60, 32, 113)

    first = run._kernel_fold_components(train_x, holdout_x, train_y, holdout_y, bank)
    second = run._kernel_fold_components(train_x, holdout_x, train_y, holdout_y, bank)

    assert first == second
    for observed, expected in zip((train_x, holdout_x, train_y, holdout_y), copies):
        np.testing.assert_array_equal(observed, expected)


def _metric(periodic, overall, regret, exact=0.70, near=0.80, preserved=0.70):
    protocol = run._load_protocol()
    return {
        "panel_count": 18,
        "defined_spearman_panel_count": 18,
        "mean_spearman": overall,
        "mean_regret_r2": regret,
        "exact_best_rate": exact,
        "within_0_01_rate": near,
        "by_family": {
            family: {
                "mean_spearman": (
                    periodic
                    if family in {"sinusoid", "triangle_wave"}
                    else preserved
                )
            }
            for family in protocol["families"]
        },
    }


def test_decision_requires_complete_universal_absolute_and_relative_gates():
    protocol = run._load_protocol()
    metrics = {"linear_probe": {"linear_probe": _metric(0.30, 0.50, 0.08)}}
    for method_id in run._method_specs(protocol):
        metrics[method_id] = {
            "current_common32_oi": _metric(0.30, 0.61, 0.08),
            "kernel_alignment_oi": _metric(0.45, 0.65, 0.07),
        }

    decision = run._decision(protocol, metrics)

    assert decision["status"] == "pass"
    assert set(decision["passing_candidates"]) == {
        f"{method_id}:{run.ELIGIBLE_ESTIMAND}"
        for method_id in run._method_specs(protocol)
    }


def test_decision_fails_closed_on_undefined_panel_correlation():
    protocol = run._load_protocol()
    metrics = {"linear_probe": {"linear_probe": _metric(0.30, 0.50, 0.08)}}
    for method_id in run._method_specs(protocol):
        candidate = _metric(0.50, 0.70, 0.05)
        candidate["defined_spearman_panel_count"] = 17
        metrics[method_id] = {
            "current_common32_oi": _metric(0.30, 0.61, 0.08),
            "kernel_alignment_oi": candidate,
        }

    decision = run._decision(protocol, metrics)

    assert decision["status"] == "fail"
    assert decision["passing_candidates"] == []
