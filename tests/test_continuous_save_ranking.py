"""Focused tests for the frozen continuous SAVE experiment."""

import numpy as np

from experiments.continuous_save_ranking import run
from experiments.continuous_save_ranking.geometry import fit_target_moment_transform


def _symmetric_covariance_data(repeats=20):
    cell_a = np.asarray([[2.0, 0.0], [-2.0, 0.0], [0.0, 0.5], [0.0, -0.5]])
    cell_b = np.asarray([[0.5, 0.0], [-0.5, 0.0], [0.0, 2.0], [0.0, -2.0]])
    X = np.vstack([np.tile(cell_a, (repeats, 1)), np.tile(cell_b, (repeats, 1))])
    labels = np.repeat([0, 1], 4 * repeats)
    return X.astype(np.float32), labels


def test_save_detects_conditional_covariance_when_cell_means_are_zero():
    X, labels = _symmetric_covariance_data()

    transform = fit_target_moment_transform(
        X,
        labels,
        mode="save",
        residual_rank=0,
        projection_rank=2,
        random_state=3,
    )

    assert transform.mean_trace < 1.0e-12
    assert transform.save_trace > 0.0
    assert transform.output_dimension >= 1
    assert np.all(transform.eigenvalues > 0.0)
    assert np.all(np.isfinite(transform.transform(X)))


def test_target_moment_transform_is_deterministic_and_detached():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 8)).astype(np.float32)
    labels = np.repeat(np.arange(6), 20)

    first = fit_target_moment_transform(
        X,
        labels,
        mode="mean_save",
        residual_rank=4,
        projection_rank=5,
        random_state=7,
    )
    second = fit_target_moment_transform(
        X,
        labels,
        mode="mean_save",
        residual_rank=4,
        projection_rank=5,
        random_state=7,
    )

    np.testing.assert_array_equal(first.basis, second.basis)
    np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)
    np.testing.assert_array_equal(first.transform(X), second.transform(X))
    assert not first.basis.flags.writeable
    assert not first.whitening.flags.writeable


def test_protocol_closes_method_grid_and_reference_gate():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    specs = run._method_specs(protocol)
    assert len(specs) == 7
    assert {row["geometry"] for row in specs.values()} == {
        "observed",
        "residual8",
        "fisher_full",
        "mean",
        "save",
        "mean_save",
        "oracle_quadratic_lift",
    }
    assert set(protocol["reference_performance_gate"]["minimum_panel_best_r2"]) == set(
        protocol["families"]
    )


def test_generator_is_deterministic_and_applies_local_nuisance_multiplier():
    protocol = run._load_protocol()
    first = run.generate_panel(protocol, 1009, "sinusoid")
    second = run.generate_panel(protocol, 1009, "sinusoid")

    assert len(first["candidates"]) == 16
    for left, right in zip(first["candidates"], second["candidates"]):
        np.testing.assert_array_equal(left["X_train"], right["X_train"])
        np.testing.assert_array_equal(left["primitive_train"], right["primitive_train"])
        assert left["effective_nuisance_scale"] == 0.20 * left["nuisance_scale"]
        assert left["X_train"].shape == (500, 16)
        assert left["primitive_train"].shape == (500, 2)


def test_fold_geometry_fits_training_rows_only():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 1009, "radial_quadratic")
    candidate = panel["candidates"][0]
    X_train = candidate["X_train"][:240]
    X_holdout = candidate["X_train"][240:320]
    primitive_train = candidate["primitive_train"][:240]
    primitive_holdout = candidate["primitive_train"][240:320]
    y_train = panel["y_train"][:240]

    first_train, _first_holdout, first_diagnostics = run._prepare_fold_geometry(
        "mean_save",
        X_train,
        X_holdout,
        primitive_train,
        primitive_holdout,
        y_train,
        5,
    )
    changed_holdout = X_holdout + 1000.0
    second_train, _second_holdout, second_diagnostics = run._prepare_fold_geometry(
        "mean_save",
        X_train,
        changed_holdout,
        primitive_train,
        primitive_holdout,
        y_train,
        5,
    )

    np.testing.assert_array_equal(first_train, second_train)
    assert first_diagnostics == second_diagnostics


def test_selector_records_geometry_and_loss_components():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 1009, "linear")
    candidate = panel["candidates"][0]
    result = run.evaluate_selector(
        "mean_save8_k4",
        candidate["X_train"][:128],
        candidate["primitive_train"][:128],
        panel["y_train"][:128],
        13,
    )

    assert len(result["folds"]) == 4
    assert np.isfinite(result["score"])
    assert np.isfinite(result["mean_actual_loss"])
    assert np.isfinite(result["mean_null_loss"])
    for row in result["folds"]:
        diagnostics = row["geometry_diagnostics"]
        assert diagnostics["geometry"] == "mean_save"
        assert diagnostics["output_dimension"] <= 8
        assert diagnostics["mean_trace"] >= 0.0
        assert diagnostics["save_trace"] >= 0.0

