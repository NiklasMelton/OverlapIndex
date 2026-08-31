"""Contract tests for the frozen continuous OI oracle diagnostic."""

import numpy as np

from experiments.continuous_oracle_diagnostic import run


def test_protocol_closes_oracle_views_and_factorial():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    specs = run._method_specs(protocol)
    assert len(specs) == 7
    assert {row["view"] for row in specs.values()} == {
        "observed",
        "oracle_signal",
        "oracle_sufficient",
        "oracle_quadratic_lift",
    }
    assert len(protocol["candidate_grid"]["quality"]) == 4
    assert len(protocol["candidate_grid"]["nuisance_scale"]) == 4


def test_candidate_grid_varies_quality_and_nuisance_independently():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 701, "quadratic_product")

    assert len(panel["candidates"]) == 16
    assert len({row["candidate_id"] for row in panel["candidates"]}) == 16
    assert {
        (row["quality"], row["nuisance_scale"])
        for row in panel["candidates"]
    } == {
        (quality, nuisance)
        for quality in protocol["candidate_grid"]["quality"]
        for nuisance in protocol["candidate_grid"]["nuisance_scale"]
    }


def test_oracle_views_remove_only_the_declared_information():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 701, "quadratic_product")
    first = panel["candidates"][0]
    same_quality_more_nuisance = panel["candidates"][1]

    assert first["views_train"]["observed"].shape == (400, 16)
    assert first["views_train"]["oracle_signal"].shape == (400, 4)
    assert first["views_train"]["oracle_sufficient"].shape == (400, 2)
    assert first["views_train"]["oracle_quadratic_lift"].shape == (400, 5)
    assert not np.array_equal(
        first["views_train"]["observed"],
        same_quality_more_nuisance["views_train"]["observed"],
    )
    for view in ("oracle_signal", "oracle_sufficient", "oracle_quadratic_lift"):
        np.testing.assert_array_equal(
            first["views_train"][view],
            same_quality_more_nuisance["views_train"][view],
        )


def test_manual_cross_fit_records_score_calibration_components():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 701, "linear")
    candidate = panel["candidates"][0]
    result = run.evaluate_selector(
        "oracle_sufficient_k4",
        candidate["views_train"]["oracle_sufficient"][:96],
        panel["y_train"][:96],
        11,
    )

    assert len(result["folds"]) == 4
    assert np.isfinite(result["score"])
    assert np.isfinite(result["mean_actual_loss"])
    assert np.isfinite(result["mean_null_loss"])
    assert result["mean_loss_ratio"] is None or np.isfinite(result["mean_loss_ratio"])
    for fold, row in enumerate(result["folds"]):
        assert row["fold"] == fold
        assert row["prototype_count"] > 0
        assert row["actual_loss"] >= 0.0
        assert row["null_loss"] >= 0.0


def test_summary_reports_raw_loss_and_normalization_correlations():
    protocol = run._load_protocol()
    method_ids = list(run._method_specs(protocol))
    candidates = []
    for position in range(4):
        heads = {
            head: {"test_r2": 0.1, "wall_seconds": 0.1}
            for head in protocol["reference_heads"]["families"]
        }
        heads["linear_ridge"]["test_r2"] = 0.4 + 0.1 * position
        selectors = {}
        for method_id in method_ids:
            selectors[method_id] = {
                "score": 0.1 * position,
                "mean_actual_loss": 1.0 - 0.1 * position,
                "mean_null_loss": 1.0,
                "mean_loss_ratio": 1.0 - 0.1 * position,
                "wall_seconds": 0.1,
            }
        candidates.append(
            {
                "candidate_id": f"c{position}",
                "quality": float(position // 2),
                "nuisance_scale": float(position % 2),
                "best_head_envelope_r2": 0.4 + 0.1 * position,
                "best_head_ids": ["linear_ridge"],
                "heads": heads,
                "selectors": selectors,
            }
        )
    panel = {
        "seed": 1,
        "family": "linear",
        "candidate_count": 4,
        "candidates": candidates,
    }

    summary = run.summarize(protocol, [panel])

    assert len(summary["component_metrics"]) == len(method_ids)
    assert len(summary["axis_metrics"]) == len(method_ids) * 4
    for method_id in method_ids:
        row = summary["component_summary"][method_id]
        assert np.isclose(row["negative_actual_loss_spearman"], 1.0)
        assert np.isclose(row["negative_loss_ratio_spearman"], 1.0)
        assert row["null_loss_spearman"] is None
        quality = summary["axis_summary"][method_id][
            "quality_at_fixed_nuisance"
        ]
        assert np.isclose(quality["mean_spearman"], 1.0)
        assert np.isclose(quality["mean_regret_r2"], 0.0)
