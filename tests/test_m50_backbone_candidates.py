"""Focused correctness tests for the private M50 candidate slice."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from experiments.heteroscedastic_distance_conditioning.conditioning_adapter import (
    ConditionedOverlapIndex as SourceConditionedOverlapIndex,
)
from experiments.m50_backbone_ranking import conditioning_adapter as local_adapter
from experiments.m50_backbone_ranking import candidates
from experiments.m50_backbone_ranking.conditioning_adapter import (
    ConditionedOverlapIndex,
)
from overlapindex import OverlapIndex


SOURCE_ADAPTER_SHA256 = (
    "e4bde79e64973d8ec22882d279a8fa0faf5c374112d2968555a68e515329d9f6"
)


def test_candidate_table_uses_only_frozen_method_ids_and_promotion_roles() -> None:
    assert candidates.CANDIDATE_IDS == (
        "A",
        "B",
        "M0-SW",
        "M1-SW",
        "M0-CB",
        "M1-CB",
    )
    assert tuple(
        spec.candidate_id for spec in candidates.CANDIDATE_SPECS if spec.promotable
    ) == ("M0-SW", "M1-SW")
    assert tuple(
        spec.candidate_id
        for spec in candidates.CANDIDATE_SPECS
        if spec.direct_control
    ) == ("A", "B")
    assert all(
        spec.diagnostic_only for spec in candidates.CANDIDATE_SPECS if spec.candidate_id.endswith("CB")
    )


def _oi_kwargs(seed: int = 17) -> dict[str, object]:
    """Return the same complete backend mapping used by every tiny panel."""

    return {
        "model_type": "MiniBatchKMeans",
        "kmeans_k": 2,
        "kmeans_kwargs": {
            "batch_size": 4,
            "max_no_improvement": 5,
            "compute_labels": False,
            "n_init": 1,
            "init": "random",
            "random_state": seed,
        },
    }


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(
        [
            [-3.0, -1.0],
            [-2.0, -1.0],
            [-2.0, 0.0],
            [-3.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [2.0, 2.0],
            [2.5, 1.5],
        ],
        dtype=np.float64,
    )
    y = np.asarray(["left"] * 4 + ["right"] * 5, dtype=object)
    X_eval = np.asarray(
        [
            [-2.5, -0.5],
            [2.25, 1.25],
            [-2.75, -0.25],
            [2.75, 1.75],
        ],
        dtype=np.float64,
    )
    y_eval = np.asarray(["left", "right", "left", "right"], dtype=object)
    return X, y, X_eval, y_eval


def _fit_local(
    candidate_id: str,
    *,
    seed: int = 17,
) -> OverlapIndex | ConditionedOverlapIndex:
    X, y, _, _ = _data()
    return candidates.build_candidate(candidate_id, _oi_kwargs(seed)).fit(X, y)


def test_vendored_adapter_records_source_provenance() -> None:
    local_module = import_module(ConditionedOverlapIndex.__module__)
    source_module = import_module(SourceConditionedOverlapIndex.__module__)
    assert local_module.VENDORED_FROM_SOURCE_SHA256 == SOURCE_ADAPTER_SHA256
    assert source_module.__file__ is not None
    assert hashlib.sha256(Path(source_module.__file__).read_bytes()).hexdigest() == SOURCE_ADAPTER_SHA256


@pytest.mark.parametrize(
    "candidate_id, source_refinement",
    [
        ("M0-SW", False),
        ("M1-SW", True),
        ("M0-CB", False),
        ("M1-CB", True),
    ],
)
def test_conditioned_candidates_match_vendored_source_wrapper(
    candidate_id: str,
    source_refinement: bool,
) -> None:
    X, y, X_eval, y_eval = _data()
    spec = candidates.CANDIDATE_BY_ID[candidate_id]
    local = candidates.build_candidate(candidate_id, _oi_kwargs()).fit(X, y)
    source = SourceConditionedOverlapIndex(
        estimator=spec.estimator,
        weighting=spec.weighting,
        gamma=spec.gamma,
        prototype_refinement=source_refinement,
        conditioning_kwargs=dict(candidates.CONDITIONING_KWARGS),
        overlap_index_kwargs=_oi_kwargs(),
    ).fit(X, y)

    assert isinstance(local, ConditionedOverlapIndex)
    assert local.prototype_refinement is source_refinement
    assert local.estimator_ is not None and source.estimator_ is not None
    np.testing.assert_array_equal(local.conditioner_.mean_, source.conditioner_.mean_)
    np.testing.assert_array_equal(
        local.conditioner_.covariance_, source.conditioner_.covariance_
    )
    np.testing.assert_array_equal(
        local.conditioner_.transform_, source.conditioner_.transform_
    )
    assert local.conditioning_diagnostics_ == source.conditioning_diagnostics_
    assert local.conditioning_diagnostics_["state_sha256"] == source.conditioning_diagnostics_[
        "state_sha256"
    ]
    np.testing.assert_array_equal(local.estimator_._model.centers, source.estimator_._model.centers)
    assert local.index == source.index
    np.testing.assert_array_equal(local.predict(X_eval), source.predict(X_eval))
    assert local.score_fixed(X_eval, y_eval) == source.score_fixed(X_eval, y_eval)


@pytest.mark.parametrize("candidate_id, refinement", [("A", False), ("B", True)])
def test_raw_candidates_are_direct_upstream_controls_and_match_direct_fit(
    candidate_id: str,
    refinement: bool,
) -> None:
    X, y, X_eval, y_eval = _data()
    actual = candidates.build_candidate(candidate_id, _oi_kwargs())
    expected = OverlapIndex(prototype_refinement=refinement, **_oi_kwargs())
    assert isinstance(actual, OverlapIndex)
    assert type(actual) is type(expected)
    assert actual.prototype_refinement is refinement
    actual.fit(X, y)
    expected.fit(X, y)
    np.testing.assert_array_equal(actual._model.centers, expected._model.centers)
    assert actual.index == expected.index
    np.testing.assert_array_equal(actual.predict(X_eval), expected.predict(X_eval))
    assert actual.score_fixed(X_eval, y_eval) == expected.score_fixed(X_eval, y_eval)


def test_candidate_pairs_share_conditioning_state_but_keep_refinement_explicit() -> None:
    for unrefined_id, refined_id in (
        ("M0-SW", "M1-SW"),
        ("M0-CB", "M1-CB"),
    ):
        unrefined = _fit_local(unrefined_id)
        refined = _fit_local(refined_id)
        assert isinstance(unrefined, ConditionedOverlapIndex)
        assert isinstance(refined, ConditionedOverlapIndex)
        np.testing.assert_array_equal(unrefined.transform(_data()[0]), refined.transform(_data()[0]))
        assert unrefined.conditioning_diagnostics_ == refined.conditioning_diagnostics_
        assert (
            unrefined.conditioning_diagnostics_["state_sha256"]
            == refined.conditioning_diagnostics_["state_sha256"]
        )
        assert unrefined.prototype_refinement is False
        assert refined.prototype_refinement is True


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_diagonal_elementwise_transform_is_used_and_exactly_matches_source(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.floating],
) -> None:
    X, y, X_eval, y_eval = _data()
    X = X.astype(dtype)
    X_eval = X_eval.astype(dtype)
    local = candidates.build_candidate("M1-SW", _oi_kwargs()).fit(X, y)
    source = SourceConditionedOverlapIndex(
        estimator="pooled_mad",
        weighting="sample_weighted_rows",
        gamma=0.5,
        prototype_refinement=True,
        conditioning_kwargs=dict(candidates.CONDITIONING_KWARGS),
        overlap_index_kwargs=_oi_kwargs(),
    ).fit(X, y)
    assert isinstance(local, ConditionedOverlapIndex)
    assert local.estimator_ is not None and source.estimator_ is not None

    calls: list[int] = []
    original = local_adapter._apply_diagonal_transform

    def spy(centered: np.ndarray, diagonal_scale: np.ndarray) -> np.ndarray:
        calls.append(1)
        return original(centered, diagonal_scale)

    monkeypatch.setattr(local_adapter, "_apply_diagonal_transform", spy)
    np.testing.assert_array_equal(local.transform(X_eval), source.transform(X_eval))
    assert calls
    assert local.score_fixed(X_eval, y_eval) == source.score_fixed(X_eval, y_eval)
    np.testing.assert_array_equal(local.predict(X_eval), source.predict(X_eval))
    np.testing.assert_array_equal(local.estimator_._model.centers, source.estimator_._model.centers)
    assert local.index == source.index
    assert local.conditioning_diagnostics_["state_sha256"] == source.conditioning_diagnostics_[
        "state_sha256"
    ]


def test_factory_deep_copies_nested_kwargs_and_rejects_duplicate_refinement() -> None:
    kwargs = _oi_kwargs()
    first = candidates.build_candidate("A", kwargs)
    second = candidates.build_candidate("A", kwargs)
    assert first.kmeans_kwargs is not second.kmeans_kwargs
    first.kmeans_kwargs["random_state"] = 999
    assert second.kmeans_kwargs["random_state"] == 17
    kwargs["kmeans_kwargs"]["random_state"] = 123
    assert second.kmeans_kwargs["random_state"] == 17
    with pytest.raises(ValueError, match="prototype_refinement"):
        candidates.build_candidate(
            "M1-SW", {**_oi_kwargs(), "prototype_refinement": True}
        )


def test_factory_rejects_unknown_ids_and_non_strict_refinement() -> None:
    with pytest.raises(ValueError, match="unknown candidate_id"):
        candidates.build_candidate("not-a-candidate", _oi_kwargs())
    invalid = replace(candidates.CANDIDATE_BY_ID["A"], prototype_refinement=1)
    with pytest.raises(TypeError, match="strict Python bool"):
        candidates.build_candidate(invalid, _oi_kwargs())


def test_candidate_config_identity_contains_full_backend_configuration() -> None:
    identity = candidates.candidate_config_identity("M1-SW", _oi_kwargs(17))
    assert identity["estimator"] == "pooled_mad"
    assert identity["weighting"] == "sample_weighted_rows"
    assert identity["gamma"] == 0.5
    assert identity["prototype_refinement"] is True
    assert identity["overlap_index_kwargs"] == _oi_kwargs(17)
    assert identity["conditioning_kwargs"] == candidates.CONDITIONING_KWARGS
    assert candidates.candidate_config_sha256("M1-SW", _oi_kwargs(17)) != candidates.candidate_config_sha256(
        "M1-SW", _oi_kwargs(18)
    )
