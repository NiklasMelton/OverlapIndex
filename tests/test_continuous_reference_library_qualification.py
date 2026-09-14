import copy
import json

import numpy as np
import pytest
from sklearn.metrics import r2_score

from experiments.continuous_reference_library_qualification import evaluator
from experiments.continuous_reference_library_qualification import run
from experiments.continuous_ranking_benchmark_v2 import run as v2


def _small_protocol():
    protocol = run.load_protocol()
    protocol = copy.deepcopy(protocol)
    protocol["heads"]["linear_ridge"]["mean_loss_lambda"] = [0.001, 0.1]
    protocol["heads"]["quadratic_ridge"]["mean_loss_lambda"] = [0.001, 0.1]
    protocol["heads"]["knn"]["n_neighbors"] = [3, 9]
    protocol["heads"]["knn"]["weights"] = ["distance"]
    protocol["heads"]["rbf_svr"]["C_standardized_target_units"] = [0.1, 1.0]
    protocol["heads"]["rbf_svr"]["epsilon_standardized_target_units"] = [0.05]
    protocol["heads"]["rbf_svr"]["bandwidth_factors"] = [0.5, 2.0]
    return protocol


def test_protocol_hash_and_scope_are_frozen():
    protocol = run.load_protocol()
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["reserved_backbones_evaluated"] is False
    assert protocol["expected_grid"] == {
        "real_target_count": 3,
        "real_candidate_draw_count": 25,
        "synthetic_panel_count": 9,
        "synthetic_candidate_count": 45,
        "overlapindex_fit_count": 0,
        "reserved_backbone_count": 0,
    }


def test_dimensionless_ridge_alpha_scales_with_training_rows():
    params = {"mean_loss_lambda": 0.1}
    assert evaluator._ridge("linear_ridge", params, 80).alpha == pytest.approx(8.0)
    assert evaluator._ridge("quadratic_ridge", params, 120).alpha == pytest.approx(12.0)


def test_rbf_target_standardization_round_trip_and_distance_scale():
    y = np.asarray([10.0, 12.0, 14.0, 16.0])
    standardized, mean, scale = evaluator._standardize_target(y)
    np.testing.assert_allclose(standardized * scale + mean, y)
    assert np.mean(standardized) == pytest.approx(0.0)
    assert np.std(standardized) == pytest.approx(1.0)
    X = np.asarray([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    assert evaluator._median_nonzero_distance(X) == pytest.approx(5.0)


def test_rbf_predictions_are_equivariant_to_target_affine_scale():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(80, 4))
    y = np.sin(X[:, 0]) + 0.1 * X[:, 1]
    params = {"C_standardized": 1.0, "epsilon_standardized": 0.05, "bandwidth_factor": 1.0}
    base = evaluator._fit_predict("rbf_svr", params, X[:60], y[:60], X[60:])
    shifted = evaluator._fit_predict("rbf_svr", params, X[:60], 7.0 + 3.0 * y[:60], X[60:])
    np.testing.assert_allclose(shifted, 7.0 + 3.0 * base, rtol=1e-10, atol=1e-10)


def test_endpoint_diagnostic_requires_material_advantage():
    protocol = _small_protocol()
    rows = [
        {"params": {"mean_loss_lambda": 0.001}, "mean_r2": 0.52},
        {"params": {"mean_loss_lambda": 0.1}, "mean_r2": 0.50},
    ]
    diagnostic = evaluator.endpoint_diagnostics(
        "linear_ridge", rows[0]["params"], rows, protocol
    )
    assert diagnostic == [
        {
            "parameter": "mean_loss_lambda",
            "endpoint": "minimum",
            "selected_value": 0.001,
            "inner_cv_advantage_r2": pytest.approx(0.02),
            "material": True,
        }
    ]


def test_corrected_evaluator_uses_unique_disjoint_fold_identities():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(100, 6))
    y = X[:, 0] - 0.3 * X[:, 1] + 0.05 * rng.normal(size=100)
    result = evaluator.evaluate_heads(
        X,
        y,
        v2._target_bins(y),
        [f"row/{index}" for index in range(100)],
        X[:20] + 0.01,
        y[:20],
        split_seed=41,
        protocol=_small_protocol(),
    )
    assert len(result["fold_audit"]) == 5
    assert all(row["identity_overlap_count"] == 0 for row in result["fold_audit"])
    assert all(row["preprocessing"]["fit_rows"] == 80 for row in result["fold_audit"])
    assert result["full_preprocessing"]["fit_rows"] == 100
    assert result["heads"]["linear_ridge"]["evaluation_r2"] > 0.9


def test_duplicate_training_identities_fail_closed():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(100, 5))
    y = rng.normal(size=100)
    ids = [f"row/{index}" for index in range(100)]
    ids[-1] = ids[0]
    with pytest.raises(ValueError, match="duplicate sample identities"):
        evaluator.evaluate_heads(
            X, y, v2._target_bins(y), ids, X[:20], y[:20],
            split_seed=7, protocol=_small_protocol()
        )


def test_synthetic_generation_and_panel_are_deterministic():
    protocol = _small_protocol()
    left = v2.synthetic_data("quadratic", 2026093401, protocol)
    right = v2.synthetic_data("quadratic", 2026093401, protocol)
    np.testing.assert_array_equal(left["y"], right["y"])
    for candidate in evaluator.v2.SYNTHETIC_CANDIDATES:
        np.testing.assert_array_equal(left["matrices"][candidate], right["matrices"][candidate])


def test_validate_grid_rejects_incomplete_execution():
    protocol = run.load_protocol()
    checkpoint = {"real_panels": [], "synthetic_panels": []}
    with pytest.raises(ValueError, match="executed grid changed"):
        run.validate_grid(protocol, checkpoint)


def test_experiment_package_does_not_import_overlapindex():
    for path in run.PACKAGE_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "import overlapindex" not in source
        assert "from overlapindex" not in source

