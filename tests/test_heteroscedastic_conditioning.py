"""Focused tests for the private heteroscedastic conditioning adapter."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse
from sklearn.covariance import oas

from overlapindex import OverlapIndex
from experiments.heteroscedastic_distance_conditioning.conditioning_adapter import (
    CONDITIONING_ESTIMATORS,
    CONDITION_NUMBER_CAP,
    RELATIVE_EIGENVALUE_FLOOR,
    ConditionedOverlapIndex,
    _weighted_median,
)
from experiments.nuisance_conditioned_distance.conditioning_adapter import (
    _fit_conditioner as _fit_legacy_conditioner,
)


def _data(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.35, size=(12, 4)),
            rng.normal(loc=0.0, scale=0.35, size=(12, 4)),
            rng.normal(loc=1.0, scale=0.35, size=(12, 4)),
        ]
    )
    y = np.repeat(np.asarray([0, 1, 2], dtype=object), 12)
    return X, y


def _oi_kwargs(seed: int = 5, k: int = 2) -> dict:
    return {
        "model_type": "KMeans",
        "kmeans_k": k,
        "kmeans_kwargs": {"n_init": 1, "random_state": seed},
    }


@pytest.mark.parametrize("estimator", CONDITIONING_ESTIMATORS)
def test_closed_estimators_fit_with_json_safe_diagnostics(estimator: str) -> None:
    X, y = _data()
    weighting = None if estimator == "none" else "sample_weighted_rows"
    gamma = 0.0 if estimator == "none" else 0.5
    adapter = ConditionedOverlapIndex(
        estimator,
        weighting=weighting,
        gamma=gamma,
        overlap_index_kwargs=_oi_kwargs(),
        conditioning_kwargs={
            "condition_number_cap": CONDITION_NUMBER_CAP,
            "relative_eigenvalue_floor": RELATIVE_EIGENVALUE_FLOOR,
        },
    ).fit(X, y)

    diagnostics = adapter.conditioning_diagnostics_
    json.dumps(diagnostics, sort_keys=True)
    assert diagnostics["estimator"] == (None if estimator == "none" else estimator)
    assert diagnostics["residual_weighting"] == weighting
    assert diagnostics["n_rows_fit"] == X.shape[0]
    assert diagnostics["n_features_fit"] == X.shape[1]
    assert diagnostics["n_classes_fit"] == 3
    assert isinstance(diagnostics["state_sha256"], str)
    assert "conditioning_fit_wall_seconds" not in diagnostics
    assert "oi_fit_wall_seconds" not in diagnostics
    if estimator == "none":
        assert diagnostics["relative_eigenvalue_floor"] is None
        assert diagnostics["condition_number_cap"] is None
        assert diagnostics["cap_or_floor_active"] is False
    else:
        assert diagnostics["relative_eigenvalue_floor"] == RELATIVE_EIGENVALUE_FLOOR
        assert diagnostics["condition_number_cap"] == CONDITION_NUMBER_CAP
        assert diagnostics["covariance_scope"] == "pooled_within_class"
        assert diagnostics["structure"] == "diagonal"
    runtime = adapter.runtime_diagnostics_
    assert set(runtime) == {
        "conditioning_fit_wall_seconds",
        "conditioning_fit_cpu_seconds",
        "oi_fit_wall_seconds",
        "oi_fit_cpu_seconds",
    }
    assert all(np.isfinite(value) and value >= 0.0 for value in runtime.values())


def test_exact_none_identity_matches_direct_oi_float32_and_integer_targets() -> None:
    X_float, _ = _data(seed=17)
    X = X_float.astype(np.float32)
    y = np.asarray([10] * 12 + [20] * 12 + [30] * 12, dtype=np.int64)
    kwargs = _oi_kwargs(seed=23)
    direct = OverlapIndex(prototype_refinement=True, **kwargs).fit(X, y)
    adapter = ConditionedOverlapIndex(
        "none",
        prototype_refinement=True,
        overlap_index_kwargs=kwargs,
    ).fit(X, y)

    assert adapter.index == direct.index
    assert adapter.transform(X) is X
    np.testing.assert_array_equal(adapter.transform(X), X)
    np.testing.assert_array_equal(adapter.estimator_._model.centers, direct._model.centers)
    np.testing.assert_array_equal(adapter.predict(X), direct.predict(X))
    assert adapter.score_fixed(X, y) == direct.score_fixed(X, y)


def test_gamma_zero_is_an_exact_identity_seam_for_conditioned_estimator() -> None:
    X, y = _data(seed=19)
    kwargs = _oi_kwargs(seed=29)
    direct = OverlapIndex(prototype_refinement=False, **kwargs).fit(X, y)
    adapter = ConditionedOverlapIndex(
        "oas_diagonal",
        weighting="sample_weighted_rows",
        gamma=0.0,
        overlap_index_kwargs=kwargs,
    ).fit(X, y)
    assert adapter.transform(X) is X
    np.testing.assert_array_equal(adapter.estimator_._model.centers, direct._model.centers)
    assert adapter.index == direct.index
    assert adapter.score_fixed(X, y) == direct.score_fixed(X, y)
    np.testing.assert_array_equal(adapter.predict(X), direct.predict(X))


def test_oas_sample_weighted_gamma_one_matches_legacy_diagonal_transform() -> None:
    X, y = _data(seed=31)
    kwargs = _oi_kwargs(seed=41)
    adapter = ConditionedOverlapIndex(
        "oas_diagonal",
        weighting="sample_weighted_rows",
        gamma=1.0,
        overlap_index_kwargs=kwargs,
    ).fit(X, y)

    residuals = []
    for label in (0, 1, 2):
        rows = X[np.asarray(y == label)]
        residuals.append(rows - np.mean(rows, axis=0))
    legacy_residuals = np.vstack(residuals).astype(np.float64, copy=False)
    legacy_covariance, _ = oas(legacy_residuals, assume_centered=True)
    legacy_covariance = 0.5 * (legacy_covariance + legacy_covariance.T)
    legacy_variance = np.diag(legacy_covariance)
    largest = float(np.max(np.maximum(legacy_variance, 0.0)))
    floor = max(largest * 1e-8, largest / 1e4)
    legacy_regularized = np.maximum(np.maximum(legacy_variance, 0.0), floor)
    legacy_transform = np.diag(1.0 / np.sqrt(legacy_regularized))
    legacy_center = np.mean(X, axis=0)
    expected = (X - legacy_center) @ legacy_transform
    np.testing.assert_allclose(adapter.transform(X), expected, rtol=0.0, atol=0.0)

    # L is not merely formula-compatible: its learned affine state must be
    # bit-identical to the archived prior-D implementation used by the
    # mandatory regression panel.
    archived = _fit_legacy_conditioner(
        np.asarray(X, dtype=np.float64),
        np.asarray(y),
        "pooled_diagonal",
    )
    current = adapter.conditioner_
    np.testing.assert_array_equal(current.mean_, archived.mean_)
    np.testing.assert_array_equal(current.covariance_, archived.covariance_)
    np.testing.assert_array_equal(current.transform_, archived.transform_)
    np.testing.assert_array_equal(adapter.transform(X), archived.transform(X))


def test_weighted_median_is_stable_at_ties() -> None:
    values = np.asarray([[2.0], [1.0], [1.0], [3.0]])
    weights = np.asarray([0.2, 0.15, 0.15, 0.5])
    # The 2.0 row is reached exactly at the 0.5 cumulative threshold; stable
    # sorting keeps the original row order among equal values.
    np.testing.assert_array_equal(_weighted_median(values, weights), [2.0])


def test_winsor_sw_and_cb_share_thresholds_and_alpha_but_not_imbalanced_moments() -> None:
    rng = np.random.default_rng(43)
    X = np.vstack(
        [
            rng.normal(0.0, 0.2, (8, 3)),
            rng.normal(0.0, 1.0, (16, 3)),
            rng.normal(0.0, 2.0, (24, 3)),
        ]
    )
    y = np.asarray([0] * 8 + [1] * 16 + [2] * 24, dtype=object)
    kwargs = _oi_kwargs(seed=47, k=1)
    sw = ConditionedOverlapIndex(
        "winsorized_oas_diagonal",
        weighting="sample_weighted_rows",
        gamma=0.5,
        overlap_index_kwargs=kwargs,
    ).fit(X, y)
    cb = ConditionedOverlapIndex(
        "winsorized_oas_diagonal",
        weighting="class_balanced",
        gamma=0.5,
        overlap_index_kwargs=kwargs,
    ).fit(X, y)
    sw_diag = sw.conditioning_diagnostics_
    cb_diag = cb.conditioning_diagnostics_
    assert sw_diag["clip_count_per_feature"] == cb_diag["clip_count_per_feature"]
    assert sw_diag["clipped_row_fraction"] == cb_diag["clipped_row_fraction"]
    assert sw_diag["shrinkage"] == cb_diag["shrinkage"]
    assert not np.array_equal(sw.conditioner_.covariance_, cb.conditioner_.covariance_)


def test_class_balanced_moment_formula_and_equal_count_bit_identity() -> None:
    rng = np.random.default_rng(53)
    X = np.vstack(
        [
            rng.normal(-1, 0.4, (5, 3)),
            rng.normal(0, 0.8, (7, 3)),
            rng.normal(1, 1.3, (9, 3)),
        ]
    )
    y = np.asarray([0] * 5 + [1] * 7 + [2] * 9, dtype=object)
    kwargs = _oi_kwargs(seed=59, k=1)
    cb = ConditionedOverlapIndex(
        "oas_diagonal",
        weighting="class_balanced",
        gamma=0.5,
        overlap_index_kwargs=kwargs,
    ).fit(X, y)
    residuals = np.vstack(
        [X[y == label] - np.mean(X[y == label], axis=0) for label in (0, 1, 2)]
    )
    weights = np.concatenate(
        [np.full(count, 1.0 / (3.0 * count)) for count in (5, 7, 9)]
    )
    covariance_sw, alpha = oas(residuals, assume_centered=True)
    covariance_sw = 0.5 * (covariance_sw + covariance_sw.T)
    moments = np.sum(weights[:, None] * residuals * residuals, axis=0)
    target = np.mean(moments)
    expected_variance = (1.0 - alpha) * moments + alpha * target
    np.testing.assert_allclose(
        np.diag(cb.conditioner_.covariance_), expected_variance, rtol=0.0, atol=0.0
    )

    X_equal = np.vstack([rng.normal(-1, 0.4, (8, 3)), rng.normal(1, 1.2, (8, 3))])
    y_equal = np.asarray([0] * 8 + [1] * 8, dtype=object)
    sw_equal = ConditionedOverlapIndex(
        "oas_diagonal",
        weighting="sample_weighted_rows",
        gamma=0.5,
        overlap_index_kwargs=kwargs,
    ).fit(X_equal, y_equal)
    cb_equal = ConditionedOverlapIndex(
        "oas_diagonal",
        weighting="class_balanced",
        gamma=0.5,
        overlap_index_kwargs=kwargs,
    ).fit(X_equal, y_equal)
    np.testing.assert_array_equal(sw_equal.transform(X_equal), cb_equal.transform(X_equal))
    sw_equal_diag = sw_equal.conditioning_diagnostics_
    cb_equal_diag = cb_equal.conditioning_diagnostics_
    assert sw_equal_diag["state_sha256"] == cb_equal_diag["state_sha256"]
    np.testing.assert_array_equal(
        sw_equal.conditioner_.covariance_, cb_equal.conditioner_.covariance_
    )


@pytest.mark.parametrize("estimator", CONDITIONING_ESTIMATORS)
def test_train_only_state_predict_and_score_fixed_do_not_refit(
    estimator: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _data(seed=61)
    weighting = None if estimator == "none" else "sample_weighted_rows"
    gamma = 0.0 if estimator == "none" else 0.5
    train_indices = np.concatenate(
        [np.arange(0, 8), np.arange(12, 20), np.arange(24, 32)]
    )
    holdout_indices = np.setdiff1d(np.arange(X.shape[0]), train_indices)
    train_X, train_y = X[train_indices], y[train_indices]
    holdout_X, holdout_y = X[holdout_indices], y[holdout_indices]
    adapter = ConditionedOverlapIndex(
        estimator,
        weighting=weighting,
        gamma=gamma,
        overlap_index_kwargs=_oi_kwargs(seed=67),
    ).fit(train_X, train_y)
    before_diag = adapter.conditioning_diagnostics_
    before_runtime = adapter.runtime_diagnostics_
    before_centers = adapter.estimator_._model.centers.copy()

    monkeypatch.setattr(adapter.estimator_, "fit", lambda *args, **kwargs: pytest.fail("refit"))
    # The held-out fold must contain every fitted class for upstream score_fixed.
    adapter.predict(holdout_X)
    adapter.score_fixed(holdout_X, holdout_y)
    assert adapter.conditioning_diagnostics_ == before_diag
    assert adapter.runtime_diagnostics_ == before_runtime
    np.testing.assert_array_equal(adapter.estimator_._model.centers, before_centers)


def test_conditioned_predict_delegates_transformed_rows_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _data(seed=71)
    adapter = ConditionedOverlapIndex(
        "pooled_mad",
        weighting="sample_weighted_rows",
        gamma=0.5,
        overlap_index_kwargs=_oi_kwargs(seed=73),
    ).fit(X, y)
    transformed = adapter.transform(X)
    captured: list[np.ndarray] = []
    original_predict = adapter.estimator_.predict

    def spy(values: np.ndarray) -> np.ndarray:
        captured.append(np.asarray(values).copy())
        return original_predict(values)

    monkeypatch.setattr(adapter.estimator_, "predict", spy)
    before = adapter.conditioning_diagnostics_
    adapter.predict(X)
    np.testing.assert_array_equal(captured[0], transformed)
    assert adapter.conditioning_diagnostics_ == before


def test_external_diagnostic_and_conditioner_mutation_is_detached() -> None:
    X, y = _data(seed=79)
    adapter = ConditionedOverlapIndex(
        "oas_diagonal",
        weighting="sample_weighted_rows",
        gamma=0.5,
        overlap_index_kwargs=_oi_kwargs(seed=83),
    ).fit(X, y)
    expected = adapter.transform(X).copy()
    diagnostics = adapter.conditioning_diagnostics_
    diagnostics["state_sha256"] = "tampered"
    conditioner = adapter.conditioner_
    with pytest.raises(ValueError):
        conditioner.transform_[...] = 0.0
    conditioner.diagnostics_["state_sha256"] = "tampered"
    np.testing.assert_array_equal(adapter.transform(X), expected)
    assert adapter.conditioning_diagnostics_["state_sha256"] != "tampered"


def test_strict_invalid_configuration_and_input_validation() -> None:
    X, y = _data()
    invalid = [
        ("bad", None, 0.0),
        ("oas_diagonal", None, 0.5),
        ("oas_diagonal", "not-a-weight", 0.5),
        ("oas_diagonal", "sample_weighted_rows", -0.1),
        ("oas_diagonal", "sample_weighted_rows", 1.1),
        ("oas_diagonal", "sample_weighted_rows", True),
    ]
    for estimator, weighting, gamma in invalid:
        with pytest.raises(ValueError):
            ConditionedOverlapIndex(estimator, weighting=weighting, gamma=gamma)
    with pytest.raises(ValueError, match="prototype_refinement"):
        ConditionedOverlapIndex(prototype_refinement=1)
    with pytest.raises(ValueError, match="prototype_refinement"):
        ConditionedOverlapIndex(overlap_index_kwargs={"prototype_refinement": False})
    with pytest.raises(ValueError, match="unknown controls"):
        ConditionedOverlapIndex(conditioning_kwargs={"condition_cap": 1.0})
    with pytest.raises(ValueError, match="frozen"):
        ConditionedOverlapIndex(conditioning_kwargs={"condition_number_cap": 2.0})

    adapter = ConditionedOverlapIndex(
        "oas_diagonal",
        weighting="sample_weighted_rows",
        gamma=0.5,
        overlap_index_kwargs=_oi_kwargs(),
    )
    with pytest.raises(ValueError, match="not fit"):
        adapter.score_fixed(X, y)
    with pytest.raises(TypeError, match="dense"):
        adapter.fit(sparse.csr_matrix(X), y)
    with pytest.raises(ValueError, match="2D"):
        adapter.fit(X[:, 0], y)
    with pytest.raises(ValueError, match="NaN"):
        adapter.fit(X, [*y[:-1], np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        adapter.fit(X, np.column_stack([y, y]))
    with pytest.raises(ValueError, match="scalar"):
        adapter.fit(X, [[label] for label in y])
    with pytest.raises(ValueError, match="At least two"):
        adapter.fit(X, np.zeros(X.shape[0], dtype=int))
