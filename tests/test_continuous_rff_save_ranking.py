"""Focused tests for the frozen continuous RFF-SAVE experiment."""

import numpy as np

from experiments.continuous_rff_save_ranking import run
from experiments.continuous_rff_save_ranking.geometry import (
    fit_random_fourier_transform,
)
from experiments.continuous_save_ranking import run as save


def test_random_fourier_transform_is_deterministic_and_detached():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(300, 16)).astype(np.float32)

    first = fit_random_fourier_transform(
        X,
        n_components=64,
        random_state=13,
    )
    second = fit_random_fourier_transform(
        X,
        n_components=64,
        random_state=13,
    )

    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.offsets, second.offsets)
    np.testing.assert_array_equal(first.transform(X), second.transform(X))
    assert first.output_dimension == 64
    assert first.gamma > 0.0
    assert first.median_squared_distance > 0.0
    assert not first.weights.flags.writeable
    assert not first.offsets.flags.writeable


def test_rff_bandwidth_and_training_projection_ignore_holdout_rows():
    rng = np.random.default_rng(9)
    X_train = rng.normal(size=(240, 12)).astype(np.float32)
    X_holdout = rng.normal(size=(80, 12)).astype(np.float32)

    first = fit_random_fourier_transform(
        X_train,
        n_components=128,
        random_state=17,
    )
    changed_holdout = X_holdout + 1000.0
    second = fit_random_fourier_transform(
        X_train,
        n_components=128,
        random_state=17,
    )

    np.testing.assert_array_equal(first.transform(X_train), second.transform(X_train))
    assert first.gamma == second.gamma
    assert first.transform(X_holdout).shape == first.transform(changed_holdout).shape
    assert np.all(np.isfinite(first.transform(changed_holdout)))


def test_protocol_closes_exact_method_and_success_gate():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    assert list(run._method_specs(protocol)) == [
        "observed_k4",
        "residual8_k4",
        "save8_k4",
        "rff64_k4",
        "rff64_save8_k4",
        "rff128_save8_k4",
        "oracle_quadratic_lift_k4",
    ]
    assert protocol["random_fourier_geometry"]["dimensions"] == [64, 128]
    assert set(protocol["success_gate"]["minimum_periodic_improvement"]) == {
        "sinusoid",
        "triangle_wave",
    }


def test_generator_is_bit_exact_with_save_experiment():
    protocol = run._load_protocol()
    current = run.generate_panel(protocol, 1009, "sinusoid")
    baseline = save.generate_panel(protocol, 1009, "sinusoid")

    np.testing.assert_array_equal(current["y_train"], baseline["y_train"])
    np.testing.assert_array_equal(current["y_test"], baseline["y_test"])
    assert len(current["candidates"]) == 16
    for left, right in zip(current["candidates"], baseline["candidates"]):
        assert left["candidate_id"] == right["candidate_id"]
        np.testing.assert_array_equal(left["X_train"], right["X_train"])
        np.testing.assert_array_equal(left["X_test"], right["X_test"])
        np.testing.assert_array_equal(
            left["primitive_train"], right["primitive_train"]
        )


def test_rff_save_fold_geometry_is_train_only_and_records_state():
    protocol = run._load_protocol()
    spec = run._method_specs(protocol)["rff64_save8_k4"]
    panel = run.generate_panel(protocol, 1009, "triangle_wave")
    candidate = panel["candidates"][0]
    X_train = candidate["X_train"][:240]
    X_holdout = candidate["X_train"][240:320]
    primitive_train = candidate["primitive_train"][:240]
    primitive_holdout = candidate["primitive_train"][240:320]
    y_train = panel["y_train"][:240]

    first_train, _first_holdout, first_diagnostics = run._prepare_fold_geometry(
        spec,
        X_train,
        X_holdout,
        primitive_train,
        primitive_holdout,
        y_train,
        5,
    )
    second_train, _second_holdout, second_diagnostics = run._prepare_fold_geometry(
        spec,
        X_train,
        X_holdout + 1000.0,
        primitive_train,
        primitive_holdout,
        y_train,
        5,
    )

    np.testing.assert_array_equal(first_train, second_train)
    assert first_diagnostics == second_diagnostics
    assert first_train.shape == (240, 8)
    assert first_diagnostics["geometry"] == "rff_save"
    assert first_diagnostics["rff_components"] == 64
    assert first_diagnostics["rff_gamma"] > 0.0
    assert first_diagnostics["save_trace"] >= 0.0


def test_selector_records_rff_and_loss_components():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 1009, "quadratic_product")
    candidate = panel["candidates"][0]
    result = run.evaluate_selector(
        "rff64_save8_k4",
        candidate["X_train"][:128],
        candidate["primitive_train"][:128],
        panel["y_train"][:128],
        19,
    )

    assert len(result["folds"]) == 4
    assert np.isfinite(result["score"])
    assert np.isfinite(result["mean_actual_loss"])
    assert np.isfinite(result["mean_null_loss"])
    for row in result["folds"]:
        diagnostics = row["geometry_diagnostics"]
        assert diagnostics["geometry"] == "rff_save"
        assert diagnostics["rff_components"] == 64
        assert diagnostics["output_dimension"] <= 8
        assert diagnostics["rff_gamma"] > 0.0


def test_success_gate_requires_preservation_periodic_gain_and_regret():
    protocol = run._load_protocol()
    families = protocol["families"]

    def method(overall, regret, family_values):
        return {
            "mean_spearman": overall,
            "mean_regret_r2": regret,
            "by_family": {
                family: {"mean_spearman": family_values[family]}
                for family in families
            },
        }

    baseline_values = {family: 0.60 for family in families}
    passing_values = dict(baseline_values)
    passing_values["sinusoid"] += 0.20
    passing_values["triangle_wave"] += 0.15
    failing_values = dict(passing_values)
    failing_values["linear"] -= 0.11
    summary = {
        "methods": {
            "save8_k4": method(0.60, 0.15, baseline_values),
            "rff64_save8_k4": method(0.61, 0.16, passing_values),
            "rff128_save8_k4": method(0.62, 0.16, failing_values),
        }
    }

    gate = run._success_gate(protocol, summary)

    assert gate["status"] == "pass"
    assert gate["candidates"]["rff64_save8_k4"]["status"] == "pass"
    assert gate["candidates"]["rff128_save8_k4"]["status"] == "fail"
