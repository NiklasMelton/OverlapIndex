"""Focused tests for the frozen continuous kernel-mean experiment."""

import numpy as np

from experiments.continuous_rff_mean_ranking import run
from experiments.continuous_save_ranking import run as save


def test_protocol_closes_kernel_mean_methods_and_gate():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    assert list(run._method_specs(protocol)) == [
        "observed_k4",
        "residual8_k4",
        "save8_k4",
        "rff64_k4",
        "rff64_mean8_k4",
        "rff64_mean_save8_k4",
        "oracle_quadratic_lift_k4",
    ]
    assert protocol["random_fourier_geometry"]["components"] == 64
    assert protocol["success_gate"]["candidate_method_ids"] == [
        "rff64_mean8_k4",
        "rff64_mean_save8_k4",
    ]


def test_generator_remains_bit_exact_with_save_experiment():
    protocol = run._load_protocol()
    current = run.generate_panel(protocol, 1103, "absolute_value")
    baseline = save.generate_panel(protocol, 1103, "absolute_value")

    np.testing.assert_array_equal(current["y_train"], baseline["y_train"])
    np.testing.assert_array_equal(current["y_test"], baseline["y_test"])
    for left, right in zip(current["candidates"], baseline["candidates"]):
        assert left["candidate_id"] == right["candidate_id"]
        np.testing.assert_array_equal(left["X_train"], right["X_train"])
        np.testing.assert_array_equal(left["X_test"], right["X_test"])
        np.testing.assert_array_equal(
            left["primitive_train"], right["primitive_train"]
        )


def test_rff_mean_geometry_is_train_only_and_records_moments():
    protocol = run._load_protocol()
    spec = run._method_specs(protocol)["rff64_mean8_k4"]
    panel = run.generate_panel(protocol, 1009, "sinusoid")
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
        7,
    )
    second_train, _second_holdout, second_diagnostics = run._prepare_fold_geometry(
        spec,
        X_train,
        X_holdout + 1000.0,
        primitive_train,
        primitive_holdout,
        y_train,
        7,
    )

    np.testing.assert_array_equal(first_train, second_train)
    assert first_diagnostics == second_diagnostics
    assert first_train.shape[0] == 240
    assert first_train.shape[1] <= 8
    assert first_diagnostics["geometry"] == "rff_mean"
    assert first_diagnostics["moment_mode"] == "mean"
    assert first_diagnostics["rff_components"] == 64
    assert first_diagnostics["mean_trace"] > 0.0
    assert first_diagnostics["save_trace"] >= 0.0


def test_fixed_rff_mean_save_uses_both_trace_normalized_moments():
    protocol = run._load_protocol()
    spec = run._method_specs(protocol)["rff64_mean_save8_k4"]
    panel = run.generate_panel(protocol, 1009, "quadratic_product")
    candidate = panel["candidates"][0]

    transformed, holdout, diagnostics = run._prepare_fold_geometry(
        spec,
        candidate["X_train"][:240],
        candidate["X_train"][240:320],
        candidate["primitive_train"][:240],
        candidate["primitive_train"][240:320],
        panel["y_train"][:240],
        11,
    )

    assert transformed.shape[0] == 240
    assert holdout.shape[0] == 80
    assert diagnostics["moment_mode"] == "mean_save"
    assert diagnostics["mean_trace"] > 0.0
    assert diagnostics["save_trace"] > 0.0
    assert np.all(np.isfinite(transformed))
    assert np.all(np.isfinite(holdout))


def test_selector_records_kernel_mean_and_loss_components():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 1201, "linear")
    candidate = panel["candidates"][0]
    result = run.evaluate_selector(
        "rff64_mean8_k4",
        candidate["X_train"][:128],
        candidate["primitive_train"][:128],
        panel["y_train"][:128],
        23,
    )

    assert len(result["folds"]) == 4
    assert np.isfinite(result["score"])
    assert np.isfinite(result["mean_actual_loss"])
    assert np.isfinite(result["mean_null_loss"])
    for row in result["folds"]:
        diagnostics = row["geometry_diagnostics"]
        assert diagnostics["geometry"] == "rff_mean"
        assert diagnostics["moment_mode"] == "mean"
        assert diagnostics["output_dimension"] <= 8
        assert diagnostics["rff_gamma"] > 0.0


def test_success_gate_is_reused_without_weakening():
    protocol = run._load_protocol()
    families = protocol["families"]

    def method(overall, regret, values):
        return {
            "mean_spearman": overall,
            "mean_regret_r2": regret,
            "by_family": {
                family: {"mean_spearman": values[family]}
                for family in families
            },
        }

    baseline_values = {family: 0.60 for family in families}
    passing_values = dict(baseline_values)
    passing_values["sinusoid"] = 0.80
    passing_values["triangle_wave"] = 0.75
    failing_values = dict(passing_values)
    failing_values["radial_quadratic"] = 0.49
    summary = {
        "methods": {
            "save8_k4": method(0.60, 0.15, baseline_values),
            "rff64_mean8_k4": method(0.61, 0.16, passing_values),
            "rff64_mean_save8_k4": method(0.62, 0.16, failing_values),
        }
    }

    gate = run.rff_save._success_gate(protocol, summary)

    assert gate["status"] == "pass"
    assert gate["candidates"]["rff64_mean8_k4"]["status"] == "pass"
    assert gate["candidates"]["rff64_mean_save8_k4"]["status"] == "fail"
