"""Focused correctness tests for the private nuisance-conditioning adapter."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from overlapindex import OverlapIndex
from experiments.nuisance_conditioned_distance.conditioning_adapter import (
    CONDITION_NUMBER_CAP,
    RELATIVE_EIGENVALUE_FLOOR,
    ConditionedOverlapIndex,
)


def _data(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.35, size=(14, 4)),
            rng.normal(loc=0.0, scale=0.35, size=(14, 4)),
            rng.normal(loc=1.0, scale=0.35, size=(14, 4)),
        ]
    )
    y = np.repeat(np.asarray([0, 1, 2], dtype=object), 14)
    return X, y


def _overlap_kwargs(seed: int = 5) -> dict:
    return {
        "model_type": "KMeans",
        "kmeans_k": 2,
        "kmeans_kwargs": {"n_init": 1, "random_state": seed},
    }


@pytest.mark.parametrize("mode", ["none", "global_isotropy", "pooled_diagonal", "pooled_full"])
def test_modes_fit_with_json_safe_diagnostics(mode: str) -> None:
    X, y = _data()
    adapter = ConditionedOverlapIndex(
        mode,
        overlap_index_kwargs=_overlap_kwargs(),
        conditioning_kwargs={
            "condition_number_cap": CONDITION_NUMBER_CAP,
            "relative_eigenvalue_floor": RELATIVE_EIGENVALUE_FLOOR,
        },
    )

    adapter.fit(X, y)
    diagnostics = adapter.conditioning_diagnostics_
    assert diagnostics["mode"] == mode
    assert diagnostics["n_rows_fit"] == X.shape[0]
    assert diagnostics["n_features_fit"] == X.shape[1]
    assert diagnostics["n_classes_fit"] == 3
    assert isinstance(diagnostics["state_sha256"], str)
    assert "conditioning_fit_wall_seconds" not in diagnostics
    assert "conditioning_fit_cpu_seconds" not in diagnostics
    assert "fitted_state_sha256" not in diagnostics
    assert "state_hash" not in diagnostics
    if mode == "none":
        assert diagnostics["covariance_scope"] is None
        assert diagnostics["structure"] == "identity"
        assert diagnostics["estimator"] is None
        assert diagnostics["cap_or_floor_active"] is False
    else:
        assert diagnostics["covariance_scope"] in {"global", "pooled_within_class"}
        assert diagnostics["structure"] in {"diagonal", "full"}
        assert diagnostics["estimator"] == "oas"
    # The report writer can serialize diagnostics without custom encoders.
    json.dumps(diagnostics, sort_keys=True)
    runtime = adapter.runtime_diagnostics_
    assert runtime["conditioning_fit_wall_seconds"] >= 0.0
    assert runtime["conditioning_fit_cpu_seconds"] >= 0.0


def test_strict_validation_and_fitted_state() -> None:
    X, y = _data()
    with pytest.raises(ValueError, match="mode must be one of"):
        ConditionedOverlapIndex("unknown")
    with pytest.raises(ValueError, match="prototype_refinement"):
        ConditionedOverlapIndex(prototype_refinement=1)
    with pytest.raises(ValueError, match="prototype_refinement"):
        ConditionedOverlapIndex(
            prototype_refinement=False,
            overlap_index_kwargs={"prototype_refinement": False},
        )
    with pytest.raises(ValueError, match="unknown controls"):
        ConditionedOverlapIndex(conditioning_kwargs={"pca_components": 2})
    with pytest.raises(ValueError, match="unknown controls"):
        ConditionedOverlapIndex(conditioning_kwargs={"condition_cap": 10.0})
    with pytest.raises(ValueError, match="frozen"):
        ConditionedOverlapIndex(
            conditioning_kwargs={"condition_number_cap": 10.0}
        )

    adapter = ConditionedOverlapIndex(
        "pooled_full", overlap_index_kwargs=_overlap_kwargs()
    )
    with pytest.raises(ValueError, match="not fit"):
        adapter.score_fixed(X, y)
    with pytest.raises(TypeError, match="dense"):
        adapter.fit(sparse.csr_matrix(X), y)
    with pytest.raises(ValueError, match="2D"):
        adapter.fit(X[:, 0], y)
    with pytest.raises(ValueError, match="NaN"):
        adapter.fit(X.copy().astype(float).astype(float), [*y[:-1], np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        adapter.fit(X, np.column_stack([y, y]))
    with pytest.raises(ValueError, match="scalar"):
        adapter.fit(X, [[label] for label in y])


@pytest.mark.parametrize("prototype_refinement", [False, True])
def test_none_matches_direct_overlap_index_exactly(prototype_refinement: bool) -> None:
    X, y = _data(seed=17)
    kwargs = _overlap_kwargs(seed=23)
    direct = OverlapIndex(
        prototype_refinement=prototype_refinement,
        **kwargs,
    ).fit(X, y)
    adapter = ConditionedOverlapIndex(
        "none",
        prototype_refinement=prototype_refinement,
        overlap_index_kwargs=kwargs,
    ).fit(X, y)

    assert adapter.index == direct.index
    np.testing.assert_array_equal(adapter.transform(X), X)
    np.testing.assert_array_equal(adapter.estimator_._model.centers, direct._model.centers)
    assert adapter.score_fixed(X, y) == direct.score_fixed(X, y)
    np.testing.assert_array_equal(adapter.predict(X), direct.predict(X))
    assert adapter.conditioner_.transform(X) is X


@pytest.mark.parametrize("prototype_refinement", [False, True])
@pytest.mark.parametrize("feature_dtype", [np.float32, np.int64])
def test_none_preserves_dense_dtype_values_and_integer_target_semantics(
    prototype_refinement: bool,
    feature_dtype: np.dtype,
) -> None:
    X, y = _data(seed=29)
    X = (
        X.astype(feature_dtype)
        if feature_dtype == np.float32
        else np.rint(4.0 * X).astype(feature_dtype)
    )
    y = np.asarray(y, dtype=np.int64)
    kwargs = _overlap_kwargs(seed=37)
    direct = OverlapIndex(
        prototype_refinement=prototype_refinement,
        **kwargs,
    ).fit(X, y)
    adapter = ConditionedOverlapIndex(
        "none",
        prototype_refinement=prototype_refinement,
        overlap_index_kwargs=kwargs,
    ).fit(X, y)

    transformed = adapter.transform(X)
    assert transformed is X
    assert transformed.dtype == X.dtype
    np.testing.assert_array_equal(transformed, X)
    np.testing.assert_array_equal(adapter.estimator_._model.centers, direct._model.centers)
    assert [type(key) for key in adapter.estimator_.label_to_index_] == [
        type(key) for key in direct.label_to_index_
    ]
    assert adapter.index == direct.index
    assert adapter.score_fixed(X, y) == direct.score_fixed(X, y)
    np.testing.assert_array_equal(adapter.predict(X), direct.predict(X))


def test_all_modes_reject_multilabel_targets() -> None:
    X, _ = _data(seed=43)
    y = ([[0, 1], [1], [2], [2, 3]] * 11)[: X.shape[0]]
    kwargs = {
        "model_type": "KMeans",
        "kmeans_k": 1,
        "kmeans_kwargs": {"n_init": 1, "random_state": 3},
    }
    for mode in ("none", "global_isotropy", "pooled_diagonal", "pooled_full"):
        with pytest.raises(ValueError, match="scalar"):
            ConditionedOverlapIndex(mode, overlap_index_kwargs=kwargs).fit(X, y)


def test_external_diagnostic_mutation_cannot_change_fitted_transform_or_score() -> None:
    X, y = _data(seed=53)
    holdout_X, holdout_y = X.copy(), y.copy()
    adapter = ConditionedOverlapIndex(
        "pooled_full", overlap_index_kwargs=_overlap_kwargs(seed=59)
    ).fit(X, y)
    baseline_transform = adapter.transform(holdout_X).copy()
    baseline_score = adapter.score_fixed(holdout_X, holdout_y)
    diagnostics = adapter.conditioning_diagnostics_
    diagnostics["state_sha256"] = "tampered"
    runtime = adapter.runtime_diagnostics_
    runtime["conditioning_fit_wall_seconds"] = 999.0
    conditioner = adapter.conditioner_
    conditioner.diagnostics_["state_sha256"] = "tampered"
    conditioner.transform_ = np.ones_like(conditioner.transform_)

    np.testing.assert_array_equal(adapter.transform(holdout_X), baseline_transform)
    assert adapter.score_fixed(holdout_X, holdout_y) == baseline_score
    assert adapter.conditioning_diagnostics_["state_sha256"] != "tampered"


def test_predict_uses_fitted_conditioned_geometry_without_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _data(seed=61)
    adapter = ConditionedOverlapIndex(
        "pooled_full", overlap_index_kwargs=_overlap_kwargs(seed=67)
    ).fit(X, y)
    before_conditioning = adapter.conditioning_diagnostics_
    before_refinement = adapter.prototype_refinement_
    before_hash = before_conditioning["state_sha256"]
    before_centers = adapter.estimator_._model.centers.copy()
    transformed = adapter.transform(X)
    captured: list[np.ndarray] = []
    original_predict = adapter.estimator_.predict

    def spy_predict(values: np.ndarray) -> np.ndarray:
        captured.append(np.asarray(values).copy())
        return original_predict(values)

    monkeypatch.setattr(adapter.estimator_, "predict", spy_predict)
    predictions = adapter.predict(X)

    assert predictions.shape[0] == X.shape[0]
    np.testing.assert_array_equal(captured[0], transformed)
    np.testing.assert_array_equal(adapter.estimator_._model.centers, before_centers)
    assert adapter.conditioning_diagnostics_ == before_conditioning
    assert adapter.prototype_refinement_ == before_refinement
    assert adapter.conditioning_diagnostics_["state_sha256"] == before_hash


def test_conditioning_state_is_deterministic_and_nested_options_are_copied() -> None:
    X, y = _data(seed=31)
    kwargs = _overlap_kwargs(seed=41)
    adapter_a = ConditionedOverlapIndex(
        "pooled_full", overlap_index_kwargs=kwargs
    )
    # Mutating the caller's nested options after construction must not alter
    # the adapter's eventual estimator configuration.
    kwargs["kmeans_kwargs"]["random_state"] = 999
    adapter_b = ConditionedOverlapIndex(
        "pooled_full", overlap_index_kwargs=_overlap_kwargs(seed=41)
    )
    adapter_a.fit(X, y)
    adapter_b.fit(X, y)

    assert adapter_a.conditioning_diagnostics_ == adapter_b.conditioning_diagnostics_
    np.testing.assert_array_equal(adapter_a.transform(X), adapter_b.transform(X))
    np.testing.assert_array_equal(
        adapter_a.estimator_._model.centers,
        adapter_b.estimator_._model.centers,
    )


def test_score_fixed_reuses_transform_and_prototypes_after_holdout_perturbation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _data(seed=7)
    # Keep all fitted classes represented in the small training fold.
    train_indices = np.concatenate(
        [np.arange(0, 10), np.arange(14, 24), np.arange(28, 38)]
    )
    holdout_indices = np.setdiff1d(np.arange(X.shape[0]), train_indices)
    train_X = X[train_indices]
    train_y = y[train_indices]
    holdout_X = X[holdout_indices]
    holdout_y = y[holdout_indices]
    adapter = ConditionedOverlapIndex(
        "pooled_diagonal", overlap_index_kwargs=_overlap_kwargs(seed=13)
    ).fit(train_X, train_y)

    before_hash = adapter.conditioning_diagnostics_["state_sha256"]
    before_mean = adapter.conditioner_.mean_.copy()
    before_centers = adapter.estimator_._model.centers.copy()

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("score_fixed must not refit the OI estimator")

    monkeypatch.setattr(adapter.estimator_, "fit", fail_if_refit)
    score_a = adapter.score_fixed(holdout_X, holdout_y)
    perturbed = holdout_X.copy()
    perturbed[0, 0] += 100.0
    score_b = adapter.score_fixed(perturbed, holdout_y)

    assert np.isfinite(score_a)
    assert np.isfinite(score_b)
    assert adapter.conditioning_diagnostics_["state_sha256"] == before_hash
    np.testing.assert_array_equal(adapter.conditioner_.mean_, before_mean)
    np.testing.assert_array_equal(adapter.estimator_._model.centers, before_centers)


def test_heldout_rows_are_transformed_with_training_state_only() -> None:
    X, y = _data(seed=101)
    train_X, train_y = X[:30], y[:30]
    holdout_X, holdout_y = X[30:], y[30:]
    adapter = ConditionedOverlapIndex(
        "global_isotropy", overlap_index_kwargs=_overlap_kwargs()
    ).fit(train_X, train_y)

    before = adapter.conditioning_diagnostics_["state_sha256"]
    transformed_a = adapter.transform(holdout_X)
    holdout_X[0, 0] += 3.0
    transformed_b = adapter.transform(holdout_X)

    assert adapter.conditioning_diagnostics_["state_sha256"] == before
    np.testing.assert_array_equal(
        adapter.transform(train_X),
        adapter.transform(train_X.copy()),
    )
    assert not np.array_equal(transformed_a, transformed_b)
    # The held-out perturbation can move the transformed held-out point, but it
    # cannot alter the fitted translation or scaling state.
    np.testing.assert_array_equal(adapter.conditioner_.mean_, np.mean(train_X, axis=0))
