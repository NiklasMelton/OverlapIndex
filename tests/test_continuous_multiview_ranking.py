"""Focused tests for the frozen continuous multi-view experiment."""

import numpy as np

from experiments.continuous_multiview_ranking import run
from experiments.continuous_save_ranking import run as save


def test_protocol_closes_block_order_and_success_gate():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)
    specs = run._method_specs(protocol)

    assert list(specs) == [
        "observed_k4",
        "residual8_k4",
        "save8_k4",
        "rff64_mean8_k4",
        "save_rffmean_k4",
        "save_rffmean_observed_k4",
        "save_rffmean_residual_k4",
        "oracle_quadratic_lift_k4",
    ]
    assert specs["save_rffmean_k4"]["blocks"] == ["save", "rff_mean"]
    assert specs["save_rffmean_observed_k4"]["blocks"] == [
        "save",
        "rff_mean",
        "observed",
    ]
    assert specs["save_rffmean_residual_k4"]["blocks"] == [
        "save",
        "rff_mean",
        "residual",
    ]


def test_generator_is_bit_exact_with_parent_save_panel():
    protocol = run._load_protocol()
    current = run.generate_panel(protocol, 1201, "triangle_wave")
    parent = save.generate_panel(protocol, 1201, "triangle_wave")

    np.testing.assert_array_equal(current["y_train"], parent["y_train"])
    np.testing.assert_array_equal(current["y_test"], parent["y_test"])
    for left, right in zip(current["candidates"], parent["candidates"]):
        assert left["candidate_id"] == right["candidate_id"]
        np.testing.assert_array_equal(left["X_train"], right["X_train"])
        np.testing.assert_array_equal(left["primitive_train"], right["primitive_train"])


def test_equal_block_concatenation_gives_each_nonzero_block_equal_norm():
    rng = np.random.default_rng(5)
    train_blocks = [
        rng.normal(size=(20, 3)).astype(np.float32),
        rng.normal(size=(20, 7)).astype(np.float32),
        rng.normal(size=(20, 11)).astype(np.float32),
    ]
    holdout_blocks = [block[:8].copy() for block in train_blocks]

    train, holdout = run._equal_block_concatenation(train_blocks, holdout_blocks)

    assert train.shape == (20, 21)
    assert holdout.shape == (8, 21)
    expected = 1.0 / np.sqrt(3.0)
    for start, stop in ((0, 3), (3, 10), (10, 21)):
        np.testing.assert_allclose(
            np.linalg.norm(train[:, start:stop], axis=1),
            expected,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
    np.testing.assert_allclose(
        np.linalg.norm(train, axis=1),
        1.0,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_multiview_fold_geometry_is_train_only_and_records_blocks():
    protocol = run._load_protocol()
    spec = run._method_specs(protocol)["save_rffmean_observed_k4"]
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
        13,
    )
    second_train, _second_holdout, second_diagnostics = run._prepare_fold_geometry(
        spec,
        X_train,
        X_holdout + 1000.0,
        primitive_train,
        primitive_holdout,
        y_train,
        13,
    )

    np.testing.assert_array_equal(first_train, second_train)
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["geometry"] == "multiview"
    assert first_diagnostics["blocks"] == ["save", "rff_mean", "observed"]
    assert first_diagnostics["block_count"] == 3
    assert [row["block"] for row in first_diagnostics["block_diagnostics"]] == [
        "save",
        "rff_mean",
        "observed",
    ]
    assert first_train.shape[1] == first_diagnostics["output_dimension"]
    np.testing.assert_allclose(
        np.linalg.norm(first_train, axis=1),
        1.0,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_two_and_three_block_dimensions_match_declared_components():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 1103, "quadratic_product")
    candidate = panel["candidates"][0]
    values = (
        candidate["X_train"][:160],
        candidate["X_train"][160:220],
        candidate["primitive_train"][:160],
        candidate["primitive_train"][160:220],
        panel["y_train"][:160],
        17,
    )

    two_train, _two_holdout, two_diagnostics = run._prepare_fold_geometry(
        run._method_specs(protocol)["save_rffmean_k4"],
        *values,
    )
    three_train, _three_holdout, three_diagnostics = run._prepare_fold_geometry(
        run._method_specs(protocol)["save_rffmean_residual_k4"],
        *values,
    )

    assert two_train.shape[1] == sum(
        row["dimension"] for row in two_diagnostics["block_diagnostics"]
    )
    assert three_train.shape[1] == sum(
        row["dimension"] for row in three_diagnostics["block_diagnostics"]
    )
    assert three_train.shape[1] > two_train.shape[1]


def test_selector_records_multiview_state_and_loss_components():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 1009, "linear")
    candidate = panel["candidates"][0]
    result = run.evaluate_selector(
        "save_rffmean_k4",
        candidate["X_train"][:128],
        candidate["primitive_train"][:128],
        panel["y_train"][:128],
        29,
    )

    assert len(result["folds"]) == 4
    assert np.isfinite(result["score"])
    assert np.isfinite(result["mean_actual_loss"])
    assert np.isfinite(result["mean_null_loss"])
    for row in result["folds"]:
        diagnostics = row["geometry_diagnostics"]
        assert diagnostics["geometry"] == "multiview"
        assert diagnostics["blocks"] == ["save", "rff_mean"]
        assert diagnostics["block_weight"] == 1.0 / np.sqrt(2.0)


def test_success_gate_remains_exactly_the_parent_gate():
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
    failing_values["absolute_value"] = 0.49
    summary = {
        "methods": {
            "save8_k4": method(0.60, 0.15, baseline_values),
            "save_rffmean_k4": method(0.61, 0.16, passing_values),
            "save_rffmean_observed_k4": method(0.62, 0.16, failing_values),
            "save_rffmean_residual_k4": method(0.62, 0.18, passing_values),
        }
    }

    gate = run.rff_save._success_gate(protocol, summary)

    assert gate["status"] == "pass"
    assert gate["candidates"]["save_rffmean_k4"]["status"] == "pass"
    assert gate["candidates"]["save_rffmean_observed_k4"]["status"] == "fail"
    assert gate["candidates"]["save_rffmean_residual_k4"]["status"] == "fail"
