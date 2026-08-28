"""Frozen candidate definitions for the M50 backbone-ranking experiment.

This module is deliberately private to the experiment package.  The two RAW
arms are constructed directly from the upstream :class:`OverlapIndex`; the
four M50 arms use the vendored train-fold-only conditioning adapter.  A caller
must supply the complete per-panel/per-fold upstream keyword mapping for each
construction.  The factory deep-copies that mapping so a candidate cannot
share mutable nested backend configuration with another candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from overlapindex import OverlapIndex

from .conditioning_adapter import ConditionedOverlapIndex


CONDITIONING_KWARGS = {
    "condition_number_cap": 10_000.0,
    "relative_eigenvalue_floor": 1e-8,
}
"""The frozen M50 conditioning caps, copied for every wrapper construction."""


@dataclass(frozen=True)
class CandidateSpec:
    """One frozen candidate arm and its complete conditioning definition."""

    candidate_id: str
    name: str
    estimator: str
    weighting: str | None
    gamma: float
    prototype_refinement: bool
    role: str
    promotable: bool
    direct_control: bool = False
    diagnostic_only: bool = False


CANDIDATE_SPECS = (
    CandidateSpec(
        candidate_id="A",
        name="raw_unrefined",
        estimator="none",
        weighting=None,
        gamma=0.0,
        prototype_refinement=False,
        role="exact_control",
        promotable=False,
        direct_control=True,
    ),
    CandidateSpec(
        candidate_id="B",
        name="raw_refined",
        estimator="none",
        weighting=None,
        gamma=0.0,
        prototype_refinement=True,
        role="exact_refined_control",
        promotable=False,
        direct_control=True,
    ),
    CandidateSpec(
        candidate_id="M0-SW",
        name="m50_sample_weighted_unrefined",
        estimator="pooled_mad",
        weighting="sample_weighted_rows",
        gamma=0.5,
        prototype_refinement=False,
        role="primary_factorial_candidate",
        promotable=True,
    ),
    CandidateSpec(
        candidate_id="M1-SW",
        name="m50_sample_weighted_refined",
        estimator="pooled_mad",
        weighting="sample_weighted_rows",
        gamma=0.5,
        prototype_refinement=True,
        role="prior_primary_candidate",
        promotable=True,
    ),
    CandidateSpec(
        candidate_id="M0-CB",
        name="m50_class_balanced_unrefined",
        estimator="pooled_mad",
        weighting="class_balanced",
        gamma=0.5,
        prototype_refinement=False,
        role="balanced_factorial_diagnostic",
        promotable=False,
        diagnostic_only=True,
    ),
    CandidateSpec(
        candidate_id="M1-CB",
        name="m50_class_balanced_refined",
        estimator="pooled_mad",
        weighting="class_balanced",
        gamma=0.5,
        prototype_refinement=True,
        role="balanced_factorial_diagnostic",
        promotable=False,
        diagnostic_only=True,
    ),
)
"""Closed candidate table; order is the frozen execution/report order."""

CANDIDATE_BY_ID = {spec.candidate_id: spec for spec in CANDIDATE_SPECS}
CANDIDATE_IDS = tuple(spec.candidate_id for spec in CANDIDATE_SPECS)
PROMOTABLE_CANDIDATES = tuple(
    spec.candidate_id for spec in CANDIDATE_SPECS if spec.promotable
)
DIAGNOSTIC_ONLY_CANDIDATES = tuple(
    spec.candidate_id for spec in CANDIDATE_SPECS if spec.diagnostic_only
)


def _spec(candidate: CandidateSpec | str) -> CandidateSpec:
    if isinstance(candidate, CandidateSpec):
        if candidate.candidate_id not in CANDIDATE_BY_ID:
            raise ValueError(
                f"unknown candidate_id {candidate.candidate_id!r}; "
                f"expected one of {CANDIDATE_IDS!r}."
            )
        return candidate
    try:
        return CANDIDATE_BY_ID[str(candidate)]
    except KeyError as exc:
        raise ValueError(
            f"unknown candidate_id {candidate!r}; expected one of {CANDIDATE_IDS!r}."
        ) from exc


def _copy_oi_kwargs(oi_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and deeply copy one complete panel/fold OI mapping."""

    if not isinstance(oi_kwargs, Mapping):
        raise TypeError("oi_kwargs must be a mapping of upstream OverlapIndex keywords.")
    if "prototype_refinement" in oi_kwargs:
        raise ValueError(
            "prototype_refinement must be supplied by the candidate spec, not oi_kwargs."
        )
    return deepcopy(dict(oi_kwargs))


def build_candidate(
    candidate_id: CandidateSpec | str,
    oi_kwargs: Mapping[str, Any],
) -> OverlapIndex | ConditionedOverlapIndex:
    """Construct one candidate from a fresh panel/fold backend mapping.

    ``oi_kwargs`` is intentionally positional in this private helper so call
    sites have to make the per-row mapping explicit.  The mapping is copied
    before either constructor sees it, including all nested dictionaries.
    """

    spec = _spec(candidate_id)
    if type(spec.prototype_refinement) is not bool:
        raise TypeError("candidate prototype_refinement must be a strict Python bool")
    copied_oi_kwargs = _copy_oi_kwargs(oi_kwargs)
    if spec.direct_control:
        return OverlapIndex(
            prototype_refinement=spec.prototype_refinement,
            **copied_oi_kwargs,
        )
    return ConditionedOverlapIndex(
        estimator=spec.estimator,
        weighting=spec.weighting,
        gamma=spec.gamma,
        prototype_refinement=spec.prototype_refinement,
        conditioning_kwargs=deepcopy(CONDITIONING_KWARGS),
        overlap_index_kwargs=copied_oi_kwargs,
    )


def candidate_config_identity(
    candidate_id: CandidateSpec | str,
    oi_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete JSON-safe identity for one candidate configuration.

    Backend keyword arguments are part of identity, not merely implementation
    details: changing ``kmeans_k``, the backend, or a nested random seed must
    not collide with an earlier artifact.  Learned conditioning state is a
    fit output and is intentionally not used here.
    """

    spec = _spec(candidate_id)
    copied_oi_kwargs = _copy_oi_kwargs(oi_kwargs)
    return {
        "candidate_id": spec.candidate_id,
        "candidate_name": spec.name,
        "estimator": spec.estimator,
        "weighting": spec.weighting,
        "gamma": float(spec.gamma),
        "prototype_refinement": spec.prototype_refinement,
        "role": spec.role,
        "promotable": spec.promotable,
        "direct_control": spec.direct_control,
        "diagnostic_only": spec.diagnostic_only,
        "conditioning_kwargs": deepcopy(CONDITIONING_KWARGS),
        "overlap_index_kwargs": copied_oi_kwargs,
    }


def candidate_config_sha256(
    candidate_id: CandidateSpec | str,
    oi_kwargs: Mapping[str, Any],
) -> str:
    """Hash :func:`candidate_config_identity` with canonical JSON encoding."""

    payload = json.dumps(
        candidate_config_identity(candidate_id, oi_kwargs),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CANDIDATE_BY_ID",
    "CANDIDATE_IDS",
    "CANDIDATE_SPECS",
    "CONDITIONING_KWARGS",
    "CandidateSpec",
    "DIAGNOSTIC_ONLY_CANDIDATES",
    "PROMOTABLE_CANDIDATES",
    "build_candidate",
    "candidate_config_identity",
    "candidate_config_sha256",
]
