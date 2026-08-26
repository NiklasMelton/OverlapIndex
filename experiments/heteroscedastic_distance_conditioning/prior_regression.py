"""Bounded regression of one locked follow-up candidate on the prior screen.

This module is deliberately separate from the follow-up development runner.  The
screen in :mod:`experiments.nuisance_conditioned_distance` is an already observed
panel, so it is used here only after its immutable source evidence has been
verified.  The module never looks at the old screen outcomes to choose a method:
the only method selected by this stage is the single candidate in a hashed
development ``promotion_decision``.

The public entry point is :func:`run_prior_regression`; ``main`` exposes the same
operation as a small, explicitly authorized CLI.  The implementation keeps the
following boundaries intentionally visible for review and tests:

* source hashes and old structural metadata are checked before a split is
  generated;
* A and B are constructed directly from the upstream ``OverlapIndex``;
* L and the locked candidate are constructed through the new experiment-local
  conditioning adapter;
* each case is an atomic four-method block, and checkpoints include every source,
  lock, code, and protocol identity needed to reject an unsafe resume; and
* an incomplete block or failed historical gate creates a stopped/rejected
  artifact and never opens a runner-up.

No public ``overlapindex`` API is changed here.  Long-running execution is not
performed at import time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CURRENT_PROTOCOL_PATH = ROOT / "experiments/heteroscedastic_distance_conditioning/protocol.json"
PRIOR_RAW_RESULTS_PATH = ROOT / "artifacts/nuisance_conditioned_distance/screen/raw_results.json"
PRIOR_MANIFEST_PATH = ROOT / "artifacts/nuisance_conditioned_distance/screen/manifest.json"
PRIOR_PROTOCOL_PATH = ROOT / "experiments/nuisance_conditioned_distance/protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/heteroscedastic_distance_conditioning/prior_regression"

# The hashes below are source evidence, not candidate outcomes.  They are
# duplicated as readable constants so a reviewer can see the historical
# contract without opening the large raw result object.
PRIOR_RAW_RESULTS_SHA256 = "9e88c48369ff3dea28d119391cf66120d740b5e3a19c6963be5ec334238c23e3"
PRIOR_MANIFEST_SHA256 = "9a289e261236e82bd94d188f44833971737267928240806be8ec98e06341137a"
PRIOR_PROTOCOL_SHA256 = "1ec47e521b67ce39db5c887004a68f1f2319a0ed5ee075067d0818b0823df0ea"
PRIOR_REPORT_SHA256 = "3dbfd2da109821841d671d96d70ca5bd1c444a581f18bae49b33f7a774f42c8c"

PRIOR_SCREEN_STAGE = "screen"
PRIOR_SCREEN_CASE_COUNT = 720
PRIOR_SCREEN_METHOD_ROW_COUNT = 4_320
PRIOR_SCREEN_METHODS = ("A", "B", "C", "D", "E", "F")
PRIOR_REGRESSION_CONTROL_METHODS = ("A", "B", "L")
PRIOR_FIT_SPLIT = "train"
PRIOR_EVALUATION_SPLIT = "evaluation"
PRIOR_BACKEND = "MiniBatchKMeans"
PRIOR_SCHEDULE_SEED = 2_026_08_12

# Relevant old source identities.  The old analysis/reporting files are not
# needed to regenerate a split and are intentionally not treated as a recipe
# dependency here.  The three files below are exactly the generator, adapter,
# and runner portions used to define the prior screen.
PRIOR_IMPLEMENTATION_SOURCE_HASHES = {
    "experiments/nuisance_conditioned_distance/fixtures.py": "c27a1b6b5b5454edfd62f5b57b17683699503d2e78f0944538c0167b6ac3f49e",
    "experiments/nuisance_conditioned_distance/conditioning_adapter.py": "b985c1696ef31cb62f85d6152f2ee5f4e7c9362d86569eb63cf2d17b151467b7",
    "experiments/nuisance_conditioned_distance/runner.py": "4f4e620e048035d871dbba922ab48f6f69e425848a29e964837edb5dd47876e8",
}

PROMOTABLE_CANDIDATES = (
    "P25",
    "P50-SW",
    "P50-CB",
    "W50-SW",
    "W50-CB",
    "M50-SW",
    "M50-CB",
)
_CANDIDATE_NAMES = {
    "A": "oi_unrefined_raw",
    "B": "oi_refined_raw",
    "L": "legacy_pooled_diagonal_full_whitening",
    "P25": "oas_partial_025_sample_weighted",
    "P50-SW": "oas_partial_050_sample_weighted",
    "P50-CB": "oas_partial_050_class_balanced",
    "W50-SW": "winsorized_oas_partial_050_sample_weighted",
    "W50-CB": "winsorized_oas_partial_050_class_balanced",
    "M50-SW": "mad_partial_050_sample_weighted",
    "M50-CB": "mad_partial_050_class_balanced",
}


class PriorRegressionError(RuntimeError):
    """Base error for an invalid or stopped prior regression."""


class PriorEvidenceError(PriorRegressionError, ValueError):
    """The immutable prior source evidence cannot be verified."""


class PriorAuthorizationError(PermissionError):
    """A lock or stage authorization is absent or does not match identity."""


class PriorIdentityError(PriorRegressionError, ValueError):
    """A recovered case, checkpoint, or output identity is inconsistent."""


class PriorStructuralError(PriorRegressionError):
    """A complete paired case could not be produced safely."""


class HistoricalGateFailure(PriorRegressionError):
    """The unchanged historical gate surface rejected the locked candidate."""


@dataclass(frozen=True)
class PriorSourceEvidence:
    """Verified hashes and paths for the old screen and current protocol."""

    current_protocol_sha256: str
    prior_raw_results_sha256: str
    prior_manifest_sha256: str
    prior_protocol_sha256: str
    prior_report_sha256: str
    paths: Mapping[str, str]
    implementation_source_hashes: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_protocol_sha256": self.current_protocol_sha256,
            "prior_raw_results_sha256": self.prior_raw_results_sha256,
            "prior_manifest_sha256": self.prior_manifest_sha256,
            "prior_protocol_sha256": self.prior_protocol_sha256,
            "prior_report_sha256": self.prior_report_sha256,
            "paths": dict(self.paths),
            "implementation_source_hashes": dict(self.implementation_source_hashes),
        }


def _validate_supplied_evidence(
    evidence: PriorSourceEvidence,
    *,
    root: os.PathLike[str] | str,
    current_protocol_path: os.PathLike[str] | str,
    verify_implementation_sources: bool | None,
) -> PriorSourceEvidence:
    """Revalidate an injected evidence token before reading old artifacts.

    ``PriorSourceEvidence`` is intentionally a small immutable value object,
    rather than a capability object.  A caller therefore cannot make a forged
    token authoritative merely by constructing the dataclass and passing it to
    the runner.  Re-reading the source-evidence list here also catches a
    current-protocol replacement between authorization and execution.
    """

    checked = verify_prior_source_evidence(
        root=root,
        current_protocol_path=current_protocol_path,
        raw_results_path=evidence.paths.get("prior_raw_results"),
        manifest_path=evidence.paths.get("prior_manifest"),
        prior_protocol_path=evidence.paths.get("prior_protocol"),
        report_path=evidence.paths.get("prior_report"),
        verify_implementation_sources=verify_implementation_sources,
    )
    if checked.as_dict() != evidence.as_dict():
        raise PriorEvidenceError("supplied prior source evidence does not match verified bytes")
    return checked


@dataclass(frozen=True)
class PriorAuthorization:
    """Validated, immutable lock token consumed by prior regression."""

    locked_candidate: str
    promotion_decision_sha256: str
    protocol_sha256: str
    code_identity_sha256: str
    decision: Mapping[str, Any]

    def __str__(self) -> str:  # convenient for runner integration
        return self.locked_candidate

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.locked_candidate == other
        if isinstance(other, PriorAuthorization):
            return (
                self.locked_candidate == other.locked_candidate
                and self.promotion_decision_sha256 == other.promotion_decision_sha256
                and self.protocol_sha256 == other.protocol_sha256
                and self.code_identity_sha256 == other.code_identity_sha256
            )
        return NotImplemented


@dataclass(frozen=True)
class PriorPanel:
    """Recovered prior case identities and their verified structural metadata."""

    cases: tuple[Any, ...]
    source_evidence: PriorSourceEvidence
    manifest: Mapping[str, Any]
    raw_metadata: Mapping[str, Any]
    case_grid_sha256: str
    generator_path: str
    generator_sha256: str
    backend: str
    fit_split: str
    evaluation_split: str
    expected_methods: tuple[str, ...] = PRIOR_SCREEN_METHODS

    def __iter__(self):
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> Any:
        return self.cases[index]


@dataclass(frozen=True)
class PriorRegressionResult:
    """Small result payload used by programmatic callers."""

    status: str
    locked_candidate: str
    case_count: int
    method_row_count: int
    paths: Mapping[str, str]
    historical_gates: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "locked_candidate": self.locked_candidate,
            "case_count": self.case_count,
            "method_row_count": self.method_row_count,
            "paths": dict(self.paths),
            "historical_gates": _json_safe(self.historical_gates),
        }


def _json_safe(value: Any) -> Any:
    """Return recursively JSON-safe values without changing finite numbers."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return repr(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decision_sha256(decision: Mapping[str, Any] | os.PathLike[str] | str) -> str:
    """Hash a decision exactly as the canonical runner does."""

    if isinstance(decision, Mapping):
        return sha256_bytes((canonical_json(decision) + "\n").encode("utf-8"))
    return sha256_path(decision)


def _read_object(value: Mapping[str, Any] | os.PathLike[str] | str, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PriorEvidenceError(f"cannot read {name}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PriorEvidenceError(f"{name} must be a JSON object")
    return dict(payload)


def _resolve_path(root: Path, value: os.PathLike[str] | str) -> Path:
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return path
        # Frozen evidence paths are repository-relative in the new protocol;
        # this fallback also handles a copied test repository with an old
        # absolute path embedded in the source-evidence record.
        text = str(path)
        for marker in ("artifacts/", "experiments/"):
            if marker in text:
                candidate = root / text[text.index(marker) :]
                if candidate.exists():
                    return candidate
        return path
    return root / path


def _source_entry(protocol: Mapping[str, Any], suffixes: Sequence[str]) -> Mapping[str, Any]:
    entries = protocol.get("source_evidence", ())
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise PriorEvidenceError("current protocol has no source_evidence list")
    normalized = tuple(str(item) for item in suffixes)
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        path = str(entry.get("path", ""))
        if any(path == suffix or path.endswith("/" + suffix) for suffix in normalized):
            return entry
    raise PriorEvidenceError(f"current protocol source_evidence lacks {normalized!r}")


def _verify_one(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise PriorEvidenceError(f"missing {label}: {path}")
    actual = sha256_path(path)
    if actual != str(expected):
        raise PriorEvidenceError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def verify_prior_source_evidence(
    *,
    root: os.PathLike[str] | str = ROOT,
    current_protocol_path: os.PathLike[str] | str = CURRENT_PROTOCOL_PATH,
    raw_results_path: os.PathLike[str] | str | None = None,
    manifest_path: os.PathLike[str] | str | None = None,
    prior_protocol_path: os.PathLike[str] | str | None = None,
    report_path: os.PathLike[str] | str | None = None,
    verify_implementation_sources: bool | None = None,
    require_report: bool = True,
) -> PriorSourceEvidence:
    """Verify all required prior hashes from the *new* protocol evidence.

    The old raw object is opened only after its byte hash is verified by this
    function.  Callers may pass paths for a copied/fake repository in tests; the
    expected hashes still come from the current protocol's source-evidence list.
    """

    root_path = Path(root).resolve()
    current_path = _resolve_path(root_path, current_protocol_path)
    if not current_path.is_file():
        raise PriorEvidenceError(f"missing current protocol: {current_path}")
    current_payload = _read_object(current_path, "current protocol")
    current_actual = sha256_path(current_path)
    try:
        from . import manifest as current_manifest

        expected_current = str(current_manifest.PROTOCOL_SHA256)
    except (ImportError, AttributeError):
        expected_current = current_actual
    if current_path.resolve() == CURRENT_PROTOCOL_PATH.resolve() and current_actual != expected_current:
        raise PriorEvidenceError(
            f"current protocol hash mismatch: expected {expected_current}, got {current_actual}"
        )

    raw_entry = _source_entry(
        current_payload,
        ("artifacts/nuisance_conditioned_distance/screen/raw_results.json", "screen/raw_results.json"),
    )
    manifest_entry = _source_entry(
        current_payload,
        ("artifacts/nuisance_conditioned_distance/screen/manifest.json", "screen/manifest.json"),
    )
    protocol_entry = _source_entry(
        current_payload,
        ("experiments/nuisance_conditioned_distance/protocol.json", "nuisance_conditioned_distance/protocol.json"),
    )
    report_entry = _source_entry(
        current_payload,
        ("artifacts/nuisance_conditioned_distance/screen/analysis_final_v2/report.md",),
    )

    raw_path = _resolve_path(
        root_path,
        raw_results_path
        or str(raw_entry.get("path", "artifacts/nuisance_conditioned_distance/screen/raw_results.json")),
    )
    old_manifest_path = _resolve_path(
        root_path,
        manifest_path
        or str(manifest_entry.get("path", "artifacts/nuisance_conditioned_distance/screen/manifest.json")),
    )
    old_protocol = _resolve_path(
        root_path,
        prior_protocol_path
        or str(protocol_entry.get("path", "experiments/nuisance_conditioned_distance/protocol.json")),
    )
    old_report_path = _resolve_path(
        root_path,
        report_path
        or str(report_entry.get("path", "artifacts/nuisance_conditioned_distance/screen/analysis_final_v2/report.md")),
    )

    raw_expected = str(raw_entry.get("sha256", ""))
    manifest_expected = str(manifest_entry.get("sha256", ""))
    protocol_expected = str(protocol_entry.get("sha256", ""))
    report_expected = str(report_entry.get("sha256", ""))
    if not raw_expected or not manifest_expected or not protocol_expected or not report_expected:
        raise PriorEvidenceError("prior source evidence is missing one or more SHA-256 values")
    # The new protocol is the authority for this stage, but it must point at
    # the exact frozen prior artifacts recorded when the follow-up was drafted.
    # Accepting a self-consistent replacement artifact would silently turn the
    # regression into a new, outcome-dependent panel.
    frozen_expectations = {
        "prior raw_results": (raw_expected, PRIOR_RAW_RESULTS_SHA256),
        "prior manifest": (manifest_expected, PRIOR_MANIFEST_SHA256),
        "prior protocol": (protocol_expected, PRIOR_PROTOCOL_SHA256),
        "prior final report": (report_expected, PRIOR_REPORT_SHA256),
    }
    for label, (observed, frozen) in frozen_expectations.items():
        if observed != frozen:
            raise PriorEvidenceError(
                f"{label} source-evidence SHA-256 is not the frozen value: "
                f"expected {frozen}, got {observed}"
            )
    raw_actual = _verify_one(raw_path, raw_expected, "prior raw_results")
    manifest_actual = _verify_one(old_manifest_path, manifest_expected, "prior manifest")
    protocol_actual = _verify_one(old_protocol, protocol_expected, "prior protocol")
    if require_report:
        report_actual = _verify_one(old_report_path, report_expected, "prior final report")
    elif old_report_path.is_file():
        report_actual = sha256_path(old_report_path)
    else:
        report_actual = ""

    if verify_implementation_sources is None:
        verify_implementation_sources = root_path == ROOT.resolve()
    implementation_actual: dict[str, str] = {}
    if verify_implementation_sources:
        for relative, expected in PRIOR_IMPLEMENTATION_SOURCE_HASHES.items():
            implementation_actual[relative] = _verify_one(
                root_path / relative, expected, f"prior implementation source {relative}"
            )

    return PriorSourceEvidence(
        current_protocol_sha256=current_actual,
        prior_raw_results_sha256=raw_actual,
        prior_manifest_sha256=manifest_actual,
        prior_protocol_sha256=protocol_actual,
        prior_report_sha256=report_actual,
        paths={
            "current_protocol": str(current_path),
            "prior_raw_results": str(raw_path),
            "prior_manifest": str(old_manifest_path),
            "prior_protocol": str(old_protocol),
            "prior_report": str(old_report_path),
        },
        implementation_source_hashes=implementation_actual,
    )


# Friendly aliases used by review wrappers.
verify_prior_hashes = verify_prior_source_evidence
verify_source_evidence = verify_prior_source_evidence


def current_code_identity(
    *, root: os.PathLike[str] | str = ROOT, source_hashes: Mapping[str, Optional[str]] | None = None
) -> str:
    """Compute the current follow-up identity using the declared source list."""

    try:
        from . import manifest as current_manifest

        hashes = dict(
            source_hashes
            if source_hashes is not None
            else current_manifest.source_hashes(root, require_existing=True)
        )
        return str(current_manifest.code_identity_sha256(root, hashes, require_existing=True))
    except (ImportError, AttributeError, TypeError, OSError, RuntimeError, ValueError) as exc:
        raise PriorAuthorizationError("cannot compute current follow-up code identity") from exc


def _hash_field(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    for container_name in ("provenance", "identity", "manifest", "input_hashes"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            for name in names:
                if name in container and container[name] is not None:
                    return container[name]
    return None


def validate_development_promotion_decision(
    decision: Mapping[str, Any] | os.PathLike[str] | str,
    *,
    protocol_sha256: str | None = None,
    code_identity_sha256: str | None = None,
    promotable_candidates: Sequence[str] = PROMOTABLE_CANDIDATES,
    require_development_stage: bool = True,
) -> PriorAuthorization:
    """Validate one hashed development lock for prior regression.

    A mapping is accepted as a test seam, but its canonical JSON plus newline is
    still hashed exactly like a decision file.  The lock itself must be singular;
    no list of runner-ups is ever consumed by this module.
    """

    payload = _read_object(decision, "promotion decision")
    if protocol_sha256 is None:
        protocol_sha256 = sha256_path(CURRENT_PROTOCOL_PATH)
    if code_identity_sha256 is None:
        code_identity_sha256 = current_code_identity()
    observed_protocol = _hash_field(payload, "protocol_sha256", "protocol_hash")
    if observed_protocol != str(protocol_sha256):
        raise PriorAuthorizationError(
            f"promotion decision protocol hash mismatch: expected {protocol_sha256}, got {observed_protocol}"
        )
    observed_code = _hash_field(payload, "code_identity_sha256", "code_identity_hash")
    if observed_code != str(code_identity_sha256):
        raise PriorAuthorizationError(
            f"promotion decision code identity mismatch: expected {code_identity_sha256}, got {observed_code}"
        )
    status = str(payload.get("status", "")).lower()
    if status not in {"locked", "pass", "passed", "eligible"}:
        raise PriorAuthorizationError("promotion decision must contain a locked development status")
    if require_development_stage:
        stage = str(payload.get("decision_stage", payload.get("stage", "development"))).lower()
        if stage not in {"development", "screen", "development_lock"}:
            raise PriorAuthorizationError(
                f"promotion decision has unexpected stage {stage!r}"
            )

    locked = payload.get("locked_candidate")
    if locked is None:
        locked = payload.get("selected_candidate", payload.get("candidate"))
    if isinstance(locked, (list, tuple)):
        if len(locked) != 1:
            raise PriorAuthorizationError("promotion decision must lock exactly one candidate")
        locked = locked[0]
    if not isinstance(locked, str) or locked not in tuple(str(v) for v in promotable_candidates):
        raise PriorAuthorizationError(f"invalid or non-promotable locked candidate: {locked!r}")
    selected = payload.get("selected_candidate")
    if selected is not None:
        if isinstance(selected, (list, tuple)) or str(selected) != locked:
            raise PriorAuthorizationError("selected and locked candidates disagree")
    # A separate plural lock field is never interpreted as a ranking surface.
    plural = payload.get("locked_candidates")
    if plural is not None:
        if not isinstance(plural, (list, tuple)) or len(plural) != 1 or plural[0] != locked:
            raise PriorAuthorizationError("promotion decision contains more than one lock")

    return PriorAuthorization(
        locked_candidate=locked,
        promotion_decision_sha256=decision_sha256(decision),
        protocol_sha256=str(protocol_sha256),
        code_identity_sha256=str(code_identity_sha256),
        decision=payload,
    )


validate_promotion_decision = validate_development_promotion_decision
authorize_lock = validate_development_promotion_decision


def authorize_prior_regression(
    promotion_decision: Mapping[str, Any] | os.PathLike[str] | str,
    *,
    protocol_sha256: str | None = None,
    code_identity_sha256: str | None = None,
    **kwargs: Any,
) -> PriorAuthorization:
    """Authorize only the prior-regression stage; no prior data are generated."""

    return validate_development_promotion_decision(
        promotion_decision,
        protocol_sha256=protocol_sha256,
        code_identity_sha256=code_identity_sha256,
        **kwargs,
    )


def prior_regression_methods(lock: str | PriorAuthorization) -> tuple[str, ...]:
    """Return exactly A, B, L, and the one locked candidate."""

    candidate = lock.locked_candidate if isinstance(lock, PriorAuthorization) else str(lock)
    if candidate not in PROMOTABLE_CANDIDATES:
        raise PriorAuthorizationError(f"invalid prior-regression lock {candidate!r}")
    methods = (*PRIOR_REGRESSION_CONTROL_METHODS, candidate)
    if len(set(methods)) != 4:
        raise PriorAuthorizationError("prior-regression method set is not exactly four methods")
    return methods


methods_for_prior_regression = prior_regression_methods


def _case_id(case: Any) -> str:
    value = getattr(case, "case_id", None)
    if value is not None:
        return str(value)
    if isinstance(case, Mapping):
        value = case.get("case_id")
        if value is not None:
            return str(value)
    raise PriorIdentityError("prior case has no case_id")


def _case_value(case: Any, name: str, default: Any = None) -> Any:
    if isinstance(case, Mapping):
        return case.get(name, default)
    return getattr(case, name, default)


def _case_config(case: Any) -> Any:
    return _case_value(case, "config", None)


def _old_backend_kwargs(k: int, seed: int) -> dict[str, Any]:
    try:
        from experiments.nuisance_conditioned_distance import runner as old_runner

        value = old_runner.oi_kwargs(int(k), int(seed))
    except (ImportError, AttributeError):
        value = {
            "model_type": PRIOR_BACKEND,
            "kmeans_k": int(k),
            "kmeans_kwargs": {
                "batch_size": 256,
                "compute_labels": False,
                "init": "random",
                "max_no_improvement": 5,
                "n_init": 1,
                "random_state": int(seed),
            },
        }
    return copy.deepcopy(dict(value))


def prior_backend_kwargs(k: int, seed: int) -> dict[str, Any]:
    """Return a fresh copy of the exact archived backend kwargs."""

    return _old_backend_kwargs(k, seed)


def _case_identity(case: Any) -> dict[str, Any]:
    config = _case_config(case)
    identity: dict[str, Any] = {
        "case_id": _case_id(case),
        "stage": _case_value(case, "stage", PRIOR_SCREEN_STAGE),
        "family": _case_value(case, "family"),
        "condition": _case_value(case, "condition"),
        "seed": _case_value(case, "seed"),
        "k": _case_value(case, "k"),
        "nuisance_strength": _case_value(case, "nuisance_strength"),
        "overlap_severity": _case_value(case, "overlap_severity"),
        "balance": _case_value(case, "balance"),
        "nuisance_shift": _case_value(case, "nuisance_shift"),
        "fit_split": PRIOR_FIT_SPLIT,
        "evaluation_split": PRIOR_EVALUATION_SPLIT,
        "backend": PRIOR_BACKEND,
        "overlap_index_kwargs": prior_backend_kwargs(
            int(_case_value(case, "k", 0)), int(_case_value(case, "seed", 0))
        ),
    }
    if config is not None:
        identity["config"] = _json_safe(config)
        for name in (
            "n_per_class",
            "n_classes",
            "signal_dim",
            "nuisance_dim",
            "train_seed",
            "eval_seed",
            "signal_design",
            "overlap_pairs",
        ):
            if hasattr(config, name):
                identity[name] = _json_safe(getattr(config, name))
    execution_offset = _case_value(case, "execution_offset", None)
    if execution_offset is not None:
        identity["execution_offset"] = int(execution_offset)
    return _json_safe(identity)


def prior_case_identity(case: Any) -> dict[str, Any]:
    return _case_identity(case)


def prior_case_grid_sha256(cases: Sequence[Any]) -> str:
    rows = [_case_identity(case) for case in cases]
    return sha256_bytes((canonical_json(rows) + "\n").encode("utf-8"))


case_grid_sha256 = prior_case_grid_sha256


def _structural_row_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a historical row without reading outcome fields."""

    allowed = (
        "case_id",
        "stage",
        "family",
        "condition",
        "seed",
        "k",
        "nuisance_strength",
        "overlap_severity",
        "balance",
        "nuisance_shift",
        "candidate_id",
        "candidate_name",
        "candidate_kind",
        "mode",
        "prototype_refinement",
        "diagnostic_only",
        "fit_split",
        "evaluation_split",
        "conditioning_fit_split",
        "n_train",
        "n_evaluation",
        "n_features",
        "overlap_index_kwargs",
        "execution_order",
        "fixture_metadata",
    )
    return _json_safe({key: row.get(key) for key in allowed if key in row})


def _expected_old_case_metadata(case: Any) -> dict[str, Any]:
    config = _case_config(case)
    n_per_class = int(getattr(config, "n_per_class", 12)) if config is not None else 12
    n_classes = int(getattr(config, "n_classes", 4)) if config is not None else 4
    n_train = n_per_class * n_classes
    balance = str(_case_value(case, "balance", "balanced"))
    if balance == "imbalanced":
        n_train += n_per_class
    signal_dim = int(getattr(config, "signal_dim", 2)) if config is not None else 2
    nuisance_dim = int(getattr(config, "nuisance_dim", 4)) if config is not None else 4
    seed = int(_case_value(case, "seed", 0))
    k = int(_case_value(case, "k", 0))
    return {
        "case_id": _case_id(case),
        "stage": PRIOR_SCREEN_STAGE,
        "family": _case_value(case, "family"),
        "condition": _case_value(case, "condition"),
        "seed": seed,
        "k": k,
        "nuisance_strength": float(_case_value(case, "nuisance_strength", 0.0)),
        "overlap_severity": float(_case_value(case, "overlap_severity", 0.0)),
        "balance": balance,
        "nuisance_shift": bool(_case_value(case, "nuisance_shift", False)),
        "fit_split": PRIOR_FIT_SPLIT,
        "evaluation_split": PRIOR_EVALUATION_SPLIT,
        "n_train": n_train,
        "n_evaluation": n_train,
        "n_features": signal_dim + nuisance_dim,
        "overlap_index_kwargs": prior_backend_kwargs(k, seed),
    }


def _compare_identity_metadata(case: Any, row: Mapping[str, Any]) -> list[str]:
    expected = _expected_old_case_metadata(case)
    failures: list[str] = []
    for key, value in expected.items():
        observed = row.get(key)
        if key == "overlap_index_kwargs":
            if _json_safe(observed) != _json_safe(value):
                failures.append(f"{key} mismatch")
        elif observed != value:
            # JSON number 0 and 0.0 compare equal, which is intentional here.
            failures.append(f"{key}={observed!r} != {value!r}")
    metadata = row.get("fixture_metadata")
    if isinstance(metadata, Mapping):
        config = _case_config(case)
        expected_meta = {
            "archived_generator_sha256": PRIOR_IMPLEMENTATION_SOURCE_HASHES["experiments/nuisance_conditioned_distance/fixtures.py"],
        }
        # The archived fixture stores its own generator hash, which is distinct
        # from the fixture.py hash.  Compare it only when the row provides it;
        # the exact value is recovered from the archived fixture constants.
        try:
            from experiments.nuisance_conditioned_distance.fixtures import ARCHIVED_STAGE2_GENERATOR_SHA256

            expected_meta["archived_generator_sha256"] = ARCHIVED_STAGE2_GENERATOR_SHA256
        except (ImportError, AttributeError):
            pass
        for key, value in expected_meta.items():
            if key in metadata and metadata.get(key) != value:
                failures.append(f"fixture_metadata.{key} mismatch")
        if config is not None:
            for key, value in (
                ("train_seed", getattr(config, "train_seed", None)),
                ("eval_seed", getattr(config, "eval_seed", None)),
                ("n_features", int(getattr(config, "signal_dim", 2)) + int(getattr(config, "nuisance_dim", 4))),
            ):
                if key in metadata and metadata.get(key) != value:
                    failures.append(f"fixture_metadata.{key} mismatch")
    return failures


def _old_execution_order(case: Any) -> tuple[str, ...]:
    try:
        from experiments.nuisance_conditioned_distance import runner as old_runner

        value = old_runner._execution_order(case)
        return tuple(str(item) for item in value)
    except (ImportError, AttributeError, TypeError, ValueError):
        return PRIOR_SCREEN_METHODS[:5]


def _check_prior_raw_structure(
    raw_payload: Mapping[str, Any],
    old_manifest: Mapping[str, Any],
    cases: Sequence[Any],
) -> dict[str, Any]:
    """Validate old row identities only; never inspect old outcomes."""

    failures: list[str] = []
    if str(raw_payload.get("stage", "")).lower() != PRIOR_SCREEN_STAGE:
        failures.append("prior raw stage is not screen")
    if str(old_manifest.get("stage", "")).lower() != PRIOR_SCREEN_STAGE:
        failures.append("prior manifest stage is not screen")
    if old_manifest.get("artifact_status") != "completed":
        failures.append("prior manifest is not completed")
    if old_manifest.get("backend") != PRIOR_BACKEND:
        failures.append("prior backend is not MiniBatchKMeans")
    if old_manifest.get("fit_split") != PRIOR_FIT_SPLIT:
        failures.append("prior fit split mismatch")
    if old_manifest.get("evaluation_split") != PRIOR_EVALUATION_SPLIT:
        failures.append("prior evaluation split mismatch")
    if int(old_manifest.get("stage_case_count", -1)) != PRIOR_SCREEN_CASE_COUNT:
        failures.append("prior stage case count mismatch")
    rows = raw_payload.get("rows")
    if not isinstance(rows, list):
        failures.append("prior raw rows are missing")
        rows = []
    if len(rows) != PRIOR_SCREEN_METHOD_ROW_COUNT:
        failures.append(f"prior raw row count {len(rows)} != {PRIOR_SCREEN_METHOD_ROW_COUNT}")
    expected_ids = [_case_id(case) for case in cases]
    expected_id_set = set(expected_ids)
    seen_case_order: list[str] = []
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    identity_seen: set[tuple[str, str]] = set()
    for value in rows:
        if not isinstance(value, Mapping):
            failures.append("prior raw row is not an object")
            continue
        # Only structural fields are copied.  In particular candidate_score,
        # pairwise outcomes, and diagnostics are deliberately not accessed.
        row = _structural_row_projection(value)
        identifier = row.get("case_id")
        candidate = row.get("candidate_id")
        if identifier not in by_case:
            seen_case_order.append(str(identifier))
            by_case[str(identifier)] = []
        by_case[str(identifier)].append(row)
        key = (str(identifier), str(candidate))
        if key in identity_seen:
            failures.append(f"duplicate prior row identity {key!r}")
        identity_seen.add(key)
    if seen_case_order != expected_ids:
        failures.append("prior case identity/order does not match recovered screen grid")
    if set(by_case) != expected_id_set:
        failures.append("prior case id set does not match recovered screen grid")
    case_by_id = {_case_id(case): case for case in cases}
    candidate_specs: dict[str, Mapping[str, Any]] = {}
    manifest_candidates = old_manifest.get("candidate_table", ())
    if isinstance(manifest_candidates, Sequence) and not isinstance(manifest_candidates, (str, bytes)):
        for item in manifest_candidates:
            if isinstance(item, Mapping) and item.get("candidate_id") is not None:
                candidate_specs[str(item["candidate_id"])] = item
    expected_structural_fields = (
        "candidate_name",
        "candidate_kind",
        "mode",
        "prototype_refinement",
        "diagnostic_only",
    )
    identity_rows: dict[str, Any] = {}
    for identifier in expected_ids:
        case_rows = by_case.get(identifier, [])
        if len(case_rows) != len(PRIOR_SCREEN_METHODS):
            failures.append(
                f"prior case {identifier!r} has {len(case_rows)} rows, expected {len(PRIOR_SCREEN_METHODS)}"
            )
            continue
        methods = tuple(str(row.get("candidate_id")) for row in case_rows)
        if set(methods) != set(PRIOR_SCREEN_METHODS):
            failures.append(f"prior case {identifier!r} candidate set mismatch")
        case = case_by_id.get(identifier)
        if case is None:
            continue
        for row in case_rows:
            candidate_spec = candidate_specs.get(str(row.get("candidate_id")))
            if candidate_spec is not None:
                for field in expected_structural_fields:
                    manifest_field = "kind" if field == "candidate_kind" else field
                    if manifest_field in candidate_spec and row.get(field) != candidate_spec.get(manifest_field):
                        failures.append(
                            f"{identifier}: {field}={row.get(field)!r} != "
                            f"{candidate_spec.get(manifest_field)!r}"
                        )
            failures.extend(
                f"{identifier}: {item}" for item in _compare_identity_metadata(case, row)
            )
            old_order = _old_execution_order(case)
            if tuple(row.get("execution_order", ())) != old_order:
                failures.append(f"{identifier}: archived execution order mismatch")
        identity_rows[identifier] = {
            "candidate_ids": list(methods),
            "execution_order": list(case_rows[0].get("execution_order", ())),
        }
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "raw_row_count": len(rows),
        "case_count": len(by_case),
        "identity_rows": identity_rows,
    }


def recover_prior_panel(
    *,
    root: os.PathLike[str] | str = ROOT,
    evidence: PriorSourceEvidence | None = None,
    cases: Sequence[Any] | None = None,
    case_factory: Callable[[], Sequence[Any]] | None = None,
    current_protocol_path: os.PathLike[str] | str = CURRENT_PROTOCOL_PATH,
    raw_results_path: os.PathLike[str] | str | None = None,
    manifest_path: os.PathLike[str] | str | None = None,
    prior_protocol_path: os.PathLike[str] | str | None = None,
    verify_implementation_sources: bool | None = None,
) -> PriorPanel:
    """Recover exact old screen identities and generator/backend metadata."""

    if evidence is None:
        evidence = verify_prior_source_evidence(
            root=root,
            current_protocol_path=current_protocol_path,
            raw_results_path=raw_results_path,
            manifest_path=manifest_path,
            prior_protocol_path=prior_protocol_path,
            verify_implementation_sources=verify_implementation_sources,
        )
    else:
        evidence = _validate_supplied_evidence(
            evidence,
            root=root,
            current_protocol_path=current_protocol_path,
            verify_implementation_sources=verify_implementation_sources,
        )
    root_path = Path(root).resolve()
    old_manifest_path = Path(evidence.paths["prior_manifest"])
    old_raw_path = Path(evidence.paths["prior_raw_results"])
    old_protocol_path = Path(evidence.paths["prior_protocol"])
    old_manifest = _read_object(old_manifest_path, "prior manifest")
    old_raw = _read_object(old_raw_path, "prior raw_results")
    old_protocol = _read_object(old_protocol_path, "prior protocol")

    if cases is None:
        if case_factory is not None:
            recovered = case_factory()
        else:
            try:
                from experiments.nuisance_conditioned_distance import runner as old_runner

                recovered = old_runner.stage_cases(PRIOR_SCREEN_STAGE)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                raise PriorEvidenceError("cannot recover the archived prior screen case grid") from exc
        cases = tuple(recovered)
    else:
        cases = tuple(cases)
    if len(cases) != PRIOR_SCREEN_CASE_COUNT:
        raise PriorIdentityError(
            f"recovered prior case count {len(cases)} != {PRIOR_SCREEN_CASE_COUNT}"
        )
    identifiers = [_case_id(case) for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise PriorIdentityError("recovered prior case ids are not unique")
    structural = _check_prior_raw_structure(old_raw, old_manifest, cases)
    if structural["status"] != "pass":
        raise PriorIdentityError(
            "prior screen structural identity mismatch: "
            + "; ".join(structural["failures"][:8])
        )
    # Validate the old protocol's own frozen candidate/backend declarations.
    if old_protocol.get("repository", {}).get("starting_commit") != old_manifest.get("starting_commit"):
        raise PriorIdentityError("prior protocol/manifest starting commit mismatch")
    old_candidates = old_manifest.get("candidate_table", ())
    if not isinstance(old_candidates, Sequence) or isinstance(old_candidates, (str, bytes)):
        raise PriorIdentityError("prior manifest candidate table is missing")
    else:
        old_ids = tuple(
            str(item.get("candidate_id"))
            for item in old_candidates
            if isinstance(item, Mapping)
        )
        if old_ids != PRIOR_SCREEN_METHODS:
            raise PriorIdentityError("prior manifest candidate table is not exactly A-F")

    generator_path = str(
        old_manifest.get("stage2_constants", {}).get(
            "generator_path", "experiments/nuisance_conditioned_distance/fixtures.py"
        )
    )
    try:
        from experiments.nuisance_conditioned_distance.fixtures import ARCHIVED_STAGE2_GENERATOR_SHA256

        generator_sha = str(ARCHIVED_STAGE2_GENERATOR_SHA256)
    except (ImportError, AttributeError):
        generator_sha = str(old_manifest.get("stage2_constants", {}).get("generator_sha256", ""))
    return PriorPanel(
        cases=tuple(cases),
        source_evidence=evidence,
        manifest=_json_safe(old_manifest),
        raw_metadata={
            "stage": old_raw.get("stage"),
            "schema_version": old_raw.get("schema_version"),
            "case_count": structural["case_count"],
            "row_count": structural["raw_row_count"],
            "identity_rows": structural["identity_rows"],
        },
        case_grid_sha256=prior_case_grid_sha256(cases),
        generator_path=generator_path,
        generator_sha256=generator_sha,
        backend=str(old_manifest.get("backend", PRIOR_BACKEND)),
        fit_split=str(old_manifest.get("fit_split", PRIOR_FIT_SPLIT)),
        evaluation_split=str(old_manifest.get("evaluation_split", PRIOR_EVALUATION_SPLIT)),
    )


recover_prior_cases = recover_prior_panel
prior_screen_panel = recover_prior_panel


def _call_factory(factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a seam while respecting positional-only and named parameters.

    The experiment's real factories use explicit keyword names, while tests and
    runner integrations often expose a compact positional ``backend_kwargs``
    parameter.  Building the call from the inspected names keeps both forms
    faithful without catching a ``TypeError`` raised *inside* the factory.
    """

    try:
        parameters = tuple(inspect.signature(factory).parameters.values())
    except (TypeError, ValueError):
        return factory(*args, **kwargs)

    aliases: dict[str, Any] = {name: value for name, value in zip(
        ("candidate_id", "k", "seed"), args[:3]
    )}
    aliases.update(kwargs)
    if "kwargs" in kwargs:
        aliases.setdefault("backend_kwargs", kwargs["kwargs"])
        aliases.setdefault("oi_kwargs", kwargs["kwargs"])
        aliases.setdefault("overlap_index_kwargs", kwargs["kwargs"])
    if "backend_kwargs" in kwargs:
        aliases.setdefault("kwargs", kwargs["backend_kwargs"])
        aliases.setdefault("oi_kwargs", kwargs["backend_kwargs"])
        aliases.setdefault("overlap_index_kwargs", kwargs["backend_kwargs"])

    positional: list[Any] = []
    selected_kwargs: dict[str, Any] = {}
    positional_index = 0
    has_var_keyword = False
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            positional.extend(args[positional_index:])
            positional_index = len(args)
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if parameter.name in aliases:
                value = aliases[parameter.name]
                # The first three positional values are supplied by semantic
                # name, so advance the fallback cursor when they coincide.
                if positional_index < len(args) and value is args[positional_index]:
                    positional_index += 1
                positional.append(value)
            elif positional_index < len(args):
                positional.append(args[positional_index])
                positional_index += 1
            elif parameter.default is not inspect.Parameter.empty:
                # Let the callable apply its own default.
                continue
            else:
                raise TypeError(f"factory is missing required parameter {parameter.name!r}")
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            if parameter.name in aliases:
                selected_kwargs[parameter.name] = aliases[parameter.name]
            elif parameter.default is inspect.Parameter.empty:
                raise TypeError(f"factory is missing required keyword-only parameter {parameter.name!r}")

    if has_var_keyword:
        # Preserve only caller-supplied keyword names (plus the explicit
        # backend aliases above); synthetic aliases such as ``candidate_id``
        # must not leak into a compact ``**kwargs`` capture.
        for name, value in kwargs.items():
            if name not in selected_kwargs and not any(
                parameter.name == name
                and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for parameter in parameters
            ):
                selected_kwargs[name] = value
        if "kwargs" in kwargs:
            for name in ("backend_kwargs", "oi_kwargs", "overlap_index_kwargs"):
                selected_kwargs.setdefault(name, kwargs["kwargs"])
    return factory(*positional, **selected_kwargs)


def _conditioned_spec(candidate_id: str) -> Any:
    try:
        from . import runner as current_runner

        return current_runner.CANDIDATE_BY_ID[candidate_id]
    except (ImportError, AttributeError, KeyError):
        # Keep the adapter seam usable in a copied repository or a focused
        # test that supplies its own constructor.  The values mirror the
        # frozen candidate table; no observed score is consulted here.
        fallback = {
            "P25": ("oas_diagonal", "sample_weighted_rows", 0.25),
            "P50-SW": ("oas_diagonal", "sample_weighted_rows", 0.50),
            "P50-CB": ("oas_diagonal", "class_balanced", 0.50),
            "W50-SW": ("winsorized_oas_diagonal", "sample_weighted_rows", 0.50),
            "W50-CB": ("winsorized_oas_diagonal", "class_balanced", 0.50),
            "M50-SW": ("pooled_mad", "sample_weighted_rows", 0.50),
            "M50-CB": ("pooled_mad", "class_balanced", 0.50),
        }
        values = fallback.get(str(candidate_id))
        if values is None:
            return None
        return {
            "candidate_id": str(candidate_id),
            "estimator": values[0],
            "weighting": values[1],
            "gamma": values[2],
            "prototype_refinement": True,
        }


def build_prior_estimator(
    candidate_id: str,
    k: int,
    seed: int,
    *,
    estimator_factory: Callable[..., Any] | None = None,
    adapter_factory: Callable[..., Any] | None = None,
    upstream_factory: Callable[..., Any] | None = None,
) -> Any:
    """Construct one prior-regression candidate with exact method semantics."""

    candidate_id = str(candidate_id)
    if candidate_id not in {"A", "B", "L", *PROMOTABLE_CANDIDATES}:
        raise PriorAuthorizationError(f"unknown prior-regression candidate {candidate_id!r}")
    if isinstance(k, bool) or int(k) <= 0:
        raise ValueError("k must be a positive integer")
    kwargs = prior_backend_kwargs(int(k), int(seed))
    if estimator_factory is not None:
        return _call_factory(
            estimator_factory,
            candidate_id,
            int(k),
            int(seed),
            kwargs=copy.deepcopy(kwargs),
        )
    if candidate_id in {"A", "B"}:
        if upstream_factory is not None:
            return _call_factory(
                upstream_factory,
                candidate_id,
                int(k),
                int(seed),
                kwargs=copy.deepcopy(kwargs),
                prototype_refinement=(candidate_id == "B"),
            )
        from overlapindex import OverlapIndex

        # Direct construction is an invariant, not a mode alias.
        return OverlapIndex(
            **copy.deepcopy(kwargs),
            prototype_refinement=(candidate_id == "B"),
        )

    if adapter_factory is None:
        try:
            from .conditioning_adapter import ConditionedOverlapIndex as adapter_factory
        except (ImportError, AttributeError) as exc:
            raise PriorStructuralError("new conditioning adapter is unavailable") from exc

    if candidate_id == "L":
        return adapter_factory(
            estimator="oas_diagonal",
            weighting="sample_weighted_rows",
            gamma=1.0,
            prototype_refinement=True,
            conditioning_kwargs={
                "condition_number_cap": 10_000.0,
                "relative_eigenvalue_floor": 1e-8,
            },
            overlap_index_kwargs=copy.deepcopy(kwargs),
        )
    spec = _conditioned_spec(candidate_id)
    if spec is None:
        raise PriorStructuralError(f"current candidate specification is unavailable for {candidate_id}")
    try:
        from . import runner as current_runner

        constructed = current_runner.estimator_for(spec, int(k), int(seed))
    except (ImportError, AttributeError, KeyError, TypeError, ValueError) as exc:
        # A caller-provided adapter factory remains a useful lightweight seam
        # for unit tests and for a runner integration that owns construction.
        if adapter_factory is None:
            raise PriorStructuralError(f"cannot construct locked candidate {candidate_id}") from exc
        constructed = adapter_factory(
            estimator=str(spec.estimator),
            weighting=spec.weighting,
            gamma=float(spec.gamma),
            prototype_refinement=bool(spec.prototype_refinement),
            conditioning_kwargs={
                "condition_number_cap": 10_000.0,
                "relative_eigenvalue_floor": 1e-8,
            },
            overlap_index_kwargs=copy.deepcopy(kwargs),
        )
    return constructed


estimator_for = build_prior_estimator


def assert_direct_control_construction(
    *,
    k: int,
    seed: int,
    estimator_factory: Callable[..., Any] | None = None,
) -> bool:
    """Check the A/B construction seam without fitting either estimator."""

    a = build_prior_estimator("A", k, seed, estimator_factory=estimator_factory)
    b = build_prior_estimator("B", k, seed, estimator_factory=estimator_factory)
    a_refinement = getattr(a, "prototype_refinement", None)
    if isinstance(a, Mapping):
        a_refinement = a.get("prototype_refinement", a.get("prototype_refinement_"))
    if a_refinement is not False:
        raise PriorStructuralError("A is not the direct unrefined upstream control")
    b_refinement = getattr(b, "prototype_refinement", None)
    if isinstance(b, Mapping):
        b_refinement = b.get("prototype_refinement", b.get("prototype_refinement_"))
    if b_refinement is not True:
        raise PriorStructuralError("B is not the direct refined upstream control")
    return True


def prior_execution_order(case: Any, locked_candidate: str | PriorAuthorization) -> tuple[str, ...]:
    """Map the archived A-E schedule to A/B/L/locked deterministically."""

    lock = locked_candidate.locked_candidate if isinstance(locked_candidate, PriorAuthorization) else str(locked_candidate)
    methods = prior_regression_methods(lock)
    old = _old_execution_order(case)
    mapped: list[str] = []
    for method in old:
        replacement = {"D": "L", "E": lock}.get(method, method)
        if replacement in methods and replacement not in mapped:
            mapped.append(replacement)
    for method in methods:
        if method not in mapped:
            mapped.append(method)
    if tuple(mapped) != tuple(method for method in mapped if method in methods) or set(mapped) != set(methods):
        raise PriorIdentityError("prior execution order cannot cover exactly four methods")
    return tuple(mapped)


execution_order_for_case = prior_execution_order


def _dataset_parts(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    def split(name: str) -> Any:
        if isinstance(dataset, Mapping):
            return dataset[name]
        return getattr(dataset, name)

    train = split("train")
    evaluation = split("evaluation") if not isinstance(dataset, Mapping) or "evaluation" in dataset else dataset.get("eval")
    if evaluation is None:
        evaluation = split("eval")

    def value(container: Any, *names: str) -> Any:
        for name in names:
            if isinstance(container, Mapping) and name in container:
                return container[name]
            if hasattr(container, name):
                return getattr(container, name)
        raise PriorStructuralError(f"prior dataset split is missing {names[0]}")

    train_x = np.asarray(value(train, "X", "features"))
    train_y = np.asarray(value(train, "y", "Y", "labels"), dtype=object)
    eval_x = np.asarray(value(evaluation, "X", "features"))
    eval_y = np.asarray(value(evaluation, "y", "Y", "labels"), dtype=object)
    metadata = {}
    if isinstance(dataset, Mapping):
        if isinstance(dataset.get("metadata"), Mapping):
            metadata = dataset["metadata"]
    elif isinstance(getattr(dataset, "metadata", None), Mapping):
        metadata = getattr(dataset, "metadata")
    if train_x.ndim != 2 or eval_x.ndim != 2:
        raise PriorStructuralError("prior train/evaluation features must be two-dimensional")
    if train_x.shape[1] != eval_x.shape[1]:
        raise PriorStructuralError("prior train/evaluation feature counts differ")
    if train_x.shape[0] != train_y.shape[0] or eval_x.shape[0] != eval_y.shape[0]:
        raise PriorStructuralError("prior split rows and labels differ")
    if not np.all(np.isfinite(np.asarray(train_x, dtype=float))) or not np.all(np.isfinite(np.asarray(eval_x, dtype=float))):
        raise PriorStructuralError("prior split contains non-finite features")
    return train_x, train_y, eval_x, eval_y, metadata


def _dataset_truth(dataset: Any) -> Mapping[str, Any]:
    """Return detached generator truth, never a historical outcome field."""

    value: Any = None
    if isinstance(dataset, Mapping):
        value = dataset.get("ground_truth", dataset.get("truth"))
    else:
        value = getattr(dataset, "ground_truth", None)
        if value is None:
            value = getattr(dataset, "truth", None)
    return value if isinstance(value, Mapping) else {}


def _model_inner(model: Any) -> Any:
    nested = getattr(model, "estimator_", None)
    return model if nested is None else nested


def _centers(model: Any) -> Any:
    inner = _model_inner(model)
    backend = getattr(inner, "_model", None)
    for owner in (backend, inner):
        if owner is None:
            continue
        for name in ("centers", "_centers"):
            value = getattr(owner, name, None)
            if value is not None:
                return np.asarray(value)
    return None


def _immutable_snapshot(model: Any) -> dict[str, Any]:
    """Capture only state that must not change during score_fixed."""

    result: dict[str, Any] = {}
    centers = _centers(model)
    if centers is not None:
        centers_array = np.ascontiguousarray(np.asarray(centers))
        result["centers_sha256"] = sha256_bytes(centers_array.tobytes())
        result["centers_shape"] = [int(value) for value in centers_array.shape]
        result["centers_dtype"] = str(centers_array.dtype)
        result["prototype_count"] = int(centers_array.shape[0])
    else:
        result["centers_sha256"] = None
        result["centers_shape"] = None
        result["centers_dtype"] = None
        result["prototype_count"] = None
    for attribute in ("conditioning_diagnostics_", "prototype_refinement_"):
        try:
            value = getattr(model, attribute)
        except (AttributeError, RuntimeError, ValueError):
            value = {}
        result[attribute] = _json_safe(value if isinstance(value, Mapping) else {})
    return result


def _determinism_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "fit_wall_seconds",
        "fit_cpu_seconds",
        "score_fixed_wall_seconds",
        "score_fixed_cpu_seconds",
        "total_wall_seconds",
        "total_cpu_seconds",
        "conditioning_fit_wall_seconds",
        "conditioning_fit_cpu_seconds",
        "oi_fit_refinement_wall_seconds",
        "oi_fit_refinement_cpu_seconds",
        "execution_position",
        "execution_order",
    }
    return _json_safe({key: value for key, value in row.items() if key not in excluded})


def determinism_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    return _determinism_signature(row)


def _fit_score(model: Any) -> float | None:
    value = getattr(model, "index", None)
    if value is None:
        value = getattr(model, "fit_score_", None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _pairwise_payload(model: Any, labels: np.ndarray) -> dict[str, Any]:
    inner = _model_inner(model)
    mapping = getattr(model, "pairwise_index", None)
    if mapping is None:
        mapping = getattr(inner, "pairwise_index", None)
    if mapping is None:
        return {"directional_pair_support_hits_evidence": [], "legacy": {}}
    classes: list[Any] = []
    for label in labels.tolist():
        if label not in classes:
            classes.append(label)
    entries: list[dict[str, Any]] = []
    legacy: dict[str, Any] = {}
    cardinality = getattr(inner, "pairwise_cardinality", None)
    hits_mapping = getattr(inner, "_pairwise_hits", None)
    sparse_mapping = getattr(inner, "sparse_adj", None)
    for source in classes:
        for target in classes:
            if source == target:
                continue
            try:
                pair_value = float(mapping[(source, target)])
            except (KeyError, TypeError, ValueError):
                continue
            support = None
            hits = None
            sparse_hits = None
            try:
                support = int(cardinality[(source, target)]) if cardinality is not None else None
            except (KeyError, TypeError, ValueError):
                support = None
            try:
                hits = int(hits_mapping.get((source, target), 0)) if hits_mapping is not None else None
            except (AttributeError, TypeError, ValueError):
                hits = None
            try:
                sparse_hits = int(sparse_mapping.get((source, target), 0)) if sparse_mapping is not None else None
            except (AttributeError, TypeError, ValueError):
                sparse_hits = None
            evidence = 1.0 - pair_value if math.isfinite(pair_value) else None
            item = {
                "source_label": _json_safe(source),
                "target_label": _json_safe(target),
                "support": support,
                "hits": hits,
                "sparse_adj_hits": sparse_hits,
                "evidence": evidence,
                "overlap_evidence": evidence,
                "pairwise_index": pair_value if math.isfinite(pair_value) else None,
                "exact_state_match": bool(
                    sparse_hits is not None
                    and hits is not None
                    and sparse_hits == hits
                    and support is not None
                    and support >= 0
                    and math.isfinite(pair_value)
                    and math.isclose(
                        pair_value,
                        1.0 - (hits / support) if support else 1.0,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ),
            }
            entries.append(item)
            legacy[f"{source}->{target}"] = {
                "pairwise_index": item["pairwise_index"],
                "overlap_evidence": evidence,
            }
    return {
        "directional_pair_support_hits_evidence": entries,
        "legacy": legacy,
        "pair_count": len(entries),
    }


def _metadata_projection(metadata: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "family",
        "signal_design",
        "n_classes",
        "n_features",
        "class_counts",
        "balance",
        "nuisance_shift",
        "nuisance_strength",
        "nuisance_label_independent",
        "train_eval_independent",
        "train_seed",
        "eval_seed",
        "archived_generator_path",
        "archived_generator_sha256",
        "archived_config_path",
        "archived_config_sha256",
        "archived_seed_role",
        "nested_selection",
        "shared_panel_row_order",
    )
    return _json_safe({key: metadata.get(key) for key in allowed if key in metadata})


def _candidate_row(
    candidate_id: str,
    case: Any,
    dataset: Any,
    order: Sequence[str],
    *,
    source_evidence: PriorSourceEvidence | None = None,
) -> dict[str, Any]:
    identity = _case_identity(case)
    _, train_y, _, eval_y, metadata = _dataset_parts(dataset)
    truth = _dataset_truth(dataset)
    spec = _conditioned_spec(candidate_id)
    mode = _case_value(spec, "mode") if spec is not None else None
    if candidate_id == "L":
        mode = "legacy_pooled_diagonal"
    elif candidate_id in {"A", "B"}:
        mode = "none"
    return {
        **identity,
        "case_id": _case_id(case),
        "candidate_id": candidate_id,
        "candidate_name": _CANDIDATE_NAMES.get(candidate_id, candidate_id),
        "candidate_kind": "oi",
        "candidate_role": "direct_control" if candidate_id in {"A", "B"} else "legacy_reference" if candidate_id == "L" else "locked_candidate",
        "candidate_promotable": candidate_id not in {"A", "B", "L"},
        "direct_control": candidate_id in {"A", "B"},
        "diagnostic_only": False,
        "mode": mode,
        "estimator": None if candidate_id in {"A", "B"} else "oas_diagonal" if candidate_id == "L" else _case_value(spec, "estimator"),
        "residual_weighting": None if candidate_id in {"A", "B"} else "sample_weighted_rows" if candidate_id == "L" else _case_value(spec, "weighting"),
        "gamma": 0.0 if candidate_id in {"A", "B"} else 1.0 if candidate_id == "L" else _case_value(spec, "gamma"),
        "prototype_refinement": candidate_id != "A",
        "fit_split": PRIOR_FIT_SPLIT,
        "evaluation_split": PRIOR_EVALUATION_SPLIT,
        "conditioning_fit_split": PRIOR_FIT_SPLIT,
        "execution_order": list(order),
        "overlap_index_kwargs": prior_backend_kwargs(int(_case_value(case, "k")), int(_case_value(case, "seed"))),
        "n_train": int(train_y.shape[0]),
        "n_evaluation": int(eval_y.shape[0]),
        "n_features": int(_dataset_parts(dataset)[0].shape[1]),
        "fixture_metadata": _metadata_projection(metadata),
        "truth": _json_safe(truth),
        "prior_source_hashes": source_evidence.as_dict() if source_evidence is not None else None,
        # Geometry/prototype diagnostics are attached after fitting.  Keeping
        # these keys in every candidate row makes the normalized tables a
        # drop-in child artifact for the current statistics loader even when a
        # caller supplies a lightweight outcome-free fake model.
        "geometry": {"status": "not_computed_prior_regression"},
        "prototype": {"status": "not_computed_prior_regression"},
        "status": "ok",
        "error": None,
        "stop_reason": None,
    }


def run_prior_case(
    case: Any,
    locked_candidate: str | PriorAuthorization,
    *,
    dataset_factory: Callable[..., Any] | None = None,
    estimator_factory: Callable[..., Any] | None = None,
    adapter_factory: Callable[..., Any] | None = None,
    upstream_factory: Callable[..., Any] | None = None,
    source_evidence: PriorSourceEvidence | None = None,
    execution_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one complete, paired A/B/L/locked case block."""

    lock = locked_candidate.locked_candidate if isinstance(locked_candidate, PriorAuthorization) else str(locked_candidate)
    methods = prior_regression_methods(lock)
    if dataset_factory is None:
        try:
            from experiments.nuisance_conditioned_distance.fixtures import generate_confirm_dataset

            config = _case_config(case)
            if config is None:
                raise PriorStructuralError("archived case lacks its ConfirmConfig")
            dataset_factory = lambda _case: generate_confirm_dataset(config)
        except ImportError as exc:
            raise PriorStructuralError("archived prior generator is unavailable") from exc
    dataset = _call_factory(dataset_factory, case)
    train_x, train_y, eval_x, eval_y, _metadata = _dataset_parts(dataset)
    order = tuple(execution_order or prior_execution_order(case, lock))
    if set(order) != set(methods) or len(order) != len(methods):
        raise PriorStructuralError("prior case execution order does not contain exactly A/B/L/locked")

    rows_by_id: dict[str, dict[str, Any]] = {}
    for candidate_id in order:
        row = _candidate_row(candidate_id, case, dataset, order, source_evidence=source_evidence)
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        try:
            model = build_prior_estimator(
                candidate_id,
                int(_case_value(case, "k")),
                int(_case_value(case, "seed")),
                estimator_factory=estimator_factory,
                adapter_factory=adapter_factory,
                upstream_factory=upstream_factory,
            )
            fit_wall_start = time.perf_counter()
            fit_cpu_start = time.process_time()
            model.fit(train_x, train_y)
            fit_wall = max(0.0, time.perf_counter() - fit_wall_start)
            fit_cpu = max(0.0, time.process_time() - fit_cpu_start)
            immutable_before = _immutable_snapshot(model)
            row.update(
                {
                    "fit_wall_seconds": fit_wall,
                    "fit_cpu_seconds": fit_cpu,
                    "conditioning_fit_wall_seconds": _json_safe(getattr(model, "runtime_diagnostics_", {}).get("conditioning_fit_wall_seconds", 0.0)) if isinstance(getattr(model, "runtime_diagnostics_", {}), Mapping) else 0.0,
                    "conditioning_fit_cpu_seconds": _json_safe(getattr(model, "runtime_diagnostics_", {}).get("conditioning_fit_cpu_seconds", 0.0)) if isinstance(getattr(model, "runtime_diagnostics_", {}), Mapping) else 0.0,
                    "fit_score": _fit_score(model),
                    "fit_state": immutable_before,
                    "conditioning": _json_safe(getattr(model, "conditioning_diagnostics_", {})) if isinstance(getattr(model, "conditioning_diagnostics_", {}), Mapping) else {},
                    "refinement": _json_safe(getattr(model, "prototype_refinement_", {})) if isinstance(getattr(model, "prototype_refinement_", {}), Mapping) else {},
                }
            )
            score_wall_start = time.perf_counter()
            score_cpu_start = time.process_time()
            score = float(model.score_fixed(eval_x, eval_y))
            score_wall = max(0.0, time.perf_counter() - score_wall_start)
            score_cpu = max(0.0, time.process_time() - score_cpu_start)
            immutable_after = _immutable_snapshot(model)
            if immutable_before != immutable_after:
                raise PriorStructuralError("score_fixed changed fitted state; possible refit")
            # The pairwise state is populated by score_fixed's held-out
            # adjacency traversal.  Capture it only after that call; reading
            # the training-time mapping here would silently label training
            # geometry as evaluation evidence in the historical gates.
            pairwise = _pairwise_payload(model, eval_y)
            predictor = getattr(model, "predict", None)
            if not callable(predictor):
                raise PriorStructuralError("prior candidate does not expose predict(X)")
            predictions = np.asarray(predictor(eval_x))
            if predictions.ndim != 1 or predictions.shape[0] != eval_y.shape[0]:
                raise PriorStructuralError("prior candidate predict(X) returned wrong shape")
            immutable_after_predict = _immutable_snapshot(model)
            if immutable_after != immutable_after_predict:
                raise PriorStructuralError("predict changed fitted state; possible refit")
            runtime = getattr(model, "runtime_diagnostics_", {})
            after_conditioning = getattr(model, "conditioning_diagnostics_", {})
            after_refinement = getattr(model, "prototype_refinement_", {})
            row.update(
                {
                    "score_fixed_wall_seconds": score_wall,
                    "score_fixed_cpu_seconds": score_cpu,
                    "total_wall_seconds": max(0.0, time.perf_counter() - started_wall),
                    "total_cpu_seconds": max(0.0, time.process_time() - started_cpu),
                    "candidate_score": score,
                    "evaluation_score": score,
                    # OI predictions are prototype identifiers, not class
                    # labels.  Record an opaque assignment digest rather than
                    # mislabelling prototype IDs as semantic accuracy.
                    "prediction_signature": {
                        "sha256": sha256_bytes(np.ascontiguousarray(predictions).tobytes()),
                        "dtype": str(predictions.dtype),
                        "shape": [int(value) for value in predictions.shape],
                    },
                    "prediction_count": int(predictions.shape[0]),
                    "score_fixed_state": immutable_after,
                    "conditioning_after_score_fixed": _json_safe(after_conditioning) if isinstance(after_conditioning, Mapping) else {},
                    "refinement_after_score_fixed": _json_safe(after_refinement) if isinstance(after_refinement, Mapping) else {},
                    "conditioning_runtime": _json_safe(runtime) if isinstance(runtime, Mapping) else {},
                    "pairwise": pairwise,
                    # The legacy analysis helpers consume a direct mapping of
                    # serialized directions.  Keep the richer directional
                    # list above for the current statistics schema, and this
                    # lossless compatibility view for the frozen gate code.
                    "pairwise_index": pairwise.get("legacy", {}),
                }
            )
            row["historical_gate_status"] = "not_evaluated_until_complete_panel"
        except Exception as exc:
            raise PriorStructuralError(
                f"candidate {candidate_id} failed in prior case {_case_id(case)}: {type(exc).__name__}: {exc}"
            ) from exc
        rows_by_id[candidate_id] = _json_safe(row)
    if set(rows_by_id) != set(methods) or len(rows_by_id) != len(methods):
        raise PriorStructuralError("prior case did not produce exactly four candidate rows")
    rows = [rows_by_id[method] for method in methods]
    return {
        "case": _case_identity(case),
        "case_id": _case_id(case),
        "rows": rows,
        "status": "ok",
        "methods": list(methods),
        "locked_candidate": lock,
    }


run_case = run_prior_case


def validate_paired_block(
    block: Mapping[str, Any],
    *,
    methods: Sequence[str],
    case_id: str | None = None,
) -> bool:
    """Validate one atomic block's identities and exact method cardinality."""

    expected = tuple(str(method) for method in methods)
    observed_case_id = str(block.get("case_id", ""))
    if case_id is not None and observed_case_id != str(case_id):
        raise PriorStructuralError(
            f"paired block case id {observed_case_id!r} != expected {case_id!r}"
        )
    rows = block.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise PriorStructuralError("paired block rows are not a sequence")
    candidate_ids = tuple(str(row.get("candidate_id")) for row in rows if isinstance(row, Mapping))
    if len(rows) != len(expected) or candidate_ids != expected or len(set(candidate_ids)) != len(expected):
        raise PriorStructuralError(
            f"paired block methods {candidate_ids!r} != exact expected {expected!r}"
        )
    if str(block.get("status", "ok")).lower() not in {"ok", "completed", "pass"}:
        raise PriorStructuralError("paired block is not complete")
    return True


def verify_parity_rows(
    observed: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = ("A", "B", "L"),
) -> bool:
    """Compare exact structural parity for supplied control/reference rows.

    This helper is deliberately opt-in: the production stage never imports old
    outcome rows.  A runner can pass a separately generated control surface,
    and the comparison excludes only runtime counters and execution ordering.
    """

    selected = set(str(method) for method in methods)
    left = {
        str(row.get("candidate_id")): _determinism_signature(row)
        for row in observed
        if str(row.get("candidate_id")) in selected
    }
    right = {
        str(row.get("candidate_id")): _determinism_signature(row)
        for row in reference
        if str(row.get("candidate_id")) in selected
    }
    if set(left) != selected or set(right) != selected or canonical_json(left) != canonical_json(right):
        raise PriorStructuralError("control/reference parity mismatch")
    return True


assert_exact_parity = verify_parity_rows
verify_parity = verify_parity_rows


def verify_deterministic_rows(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> bool:
    left = [_determinism_signature(row) for row in first]
    right = [_determinism_signature(row) for row in second]
    if canonical_json(left) != canonical_json(right):
        raise PriorStructuralError("prior regression deterministic signatures differ")
    return True


def _safe_checkpoint_path(output: Path, case_id: str) -> Path:
    if not case_id or "/" in case_id or "\\" in case_id or ".." in case_id:
        raise PriorIdentityError(f"unsafe prior case checkpoint id {case_id!r}")
    return output / "checkpoints" / f"{case_id}.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".partial-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_write_immutable(path: Path, payload: bytes) -> None:
    """Publish an immutable artifact without exposing a partial file.

    A prior-regression decision is a prerequisite for the one-shot
    confirmation stage.  Existing bytes are therefore never replaced—even
    when a caller asks to write the same value again.  The temporary file is
    hard-linked into place, making publication atomic while preserving the
    fail-closed ``FileExistsError`` contract under a concurrent writer.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable prior decision artifact already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=".partial-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"immutable prior decision artifact already exists: {path}")
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def checkpoint_identity(
    case: Any,
    *,
    authorization: PriorAuthorization,
    source_evidence: PriorSourceEvidence,
    code_identity_sha256: str,
    protocol_sha256: str,
    methods: Sequence[str],
) -> dict[str, Any]:
    """Build the complete resume identity for one case."""

    return {
        "stage": "prior_regression",
        "case": _case_identity(case),
        "case_id": _case_id(case),
        "methods": list(methods),
        "promotion_decision_sha256": authorization.promotion_decision_sha256,
        "locked_candidate": authorization.locked_candidate,
        "protocol_sha256": protocol_sha256,
        "code_identity_sha256": code_identity_sha256,
        "prior_source_hashes": source_evidence.as_dict(),
    }


def write_checkpoint(
    output: os.PathLike[str] | str,
    payload: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> Path:
    output_path = Path(output)
    safe_payload = _json_safe(payload)
    if not isinstance(safe_payload, Mapping):
        raise TypeError("prior checkpoint payload must be a mapping")
    record = {
        "schema_version": 1,
        "identity_sha256": sha256_bytes(canonical_json(identity).encode("utf-8")),
        "identity": _json_safe(identity),
        # Identity protects the source/lock/code/protocol binding; this second
        # digest protects the complete row block from an in-place edit that
        # leaves the identity unchanged.  Hash the canonical JSON-safe value
        # so the bytes are stable across NumPy/scalar/tuple adapters.
        "payload_sha256": sha256_bytes(canonical_json(safe_payload).encode("utf-8")),
        "payload": safe_payload,
    }
    path = _safe_checkpoint_path(output_path, str(identity["case_id"]))
    _atomic_write(path, (canonical_json(record) + "\n").encode("utf-8"))
    return path


def _validate_reusable_checkpoint_payload(
    payload: Mapping[str, Any],
    identity: Mapping[str, Any],
    case_identifier: str,
) -> None:
    """Reject anything other than a complete, successful paired case.

    Checkpoint reuse is intentionally stricter than the in-memory block
    validator.  The latter is useful while producing a block, whereas this
    function is the resume trust boundary: a partial/error/stopped payload
    must never be mistaken for a completed A/B/L/locked case merely because
    its outer identity still matches.
    """

    expected_case_id = str(identity.get("case_id", ""))
    if not expected_case_id or expected_case_id != str(case_identifier):
        raise PriorIdentityError(
            f"prior checkpoint identity case id mismatch for {case_identifier}"
        )
    if payload.get("case_id") != str(case_identifier):
        raise PriorIdentityError(
            f"prior checkpoint payload case_id mismatch for {case_identifier}"
        )

    locked_candidate = identity.get("locked_candidate")
    try:
        expected_methods = prior_regression_methods(str(locked_candidate))
    except PriorAuthorizationError as exc:
        raise PriorIdentityError(
            f"prior checkpoint identity has invalid locked candidate for {case_identifier}"
        ) from exc
    identity_methods = identity.get("methods")
    if (
        isinstance(identity_methods, (str, bytes))
        or not isinstance(identity_methods, Sequence)
        or tuple(str(method) for method in identity_methods) != expected_methods
    ):
        raise PriorIdentityError(
            f"prior checkpoint identity methods are not exact A/B/L/locked order for {case_identifier}"
        )

    # A reusable outer payload is itself a successful block.  Keep the marker
    # checks aligned with the current runner contract: null/empty diagnostic
    # fields emitted by serializers are harmless, but any substantive marker
    # rejects resume.
    if payload.get("status") != "ok":
        raise PriorStructuralError(
            f"prior checkpoint payload status is not reusable for {case_identifier}"
        )
    for marker in ("error", "stop_reason", "stopped"):
        value = payload.get(marker)
        if value not in (None, "", False, []):
            raise PriorStructuralError(
                f"prior checkpoint payload contains {marker} for {case_identifier}"
            )

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise PriorStructuralError(
            f"prior checkpoint payload rows must be a list for {case_identifier}"
        )
    observed_methods = [row.get("candidate_id") for row in rows if isinstance(row, Mapping)]
    if len(rows) != len(expected_methods) or observed_methods != list(expected_methods):
        raise PriorStructuralError(
            f"prior checkpoint payload rows do not exactly match A/B/L/locked order for {case_identifier}"
        )
    if any(not isinstance(row, Mapping) for row in rows):
        raise PriorStructuralError(
            f"prior checkpoint payload rows must be objects for {case_identifier}"
        )
    for method, row in zip(expected_methods, rows):
        if row.get("status") != "ok":
            raise PriorStructuralError(
                f"prior checkpoint candidate {method} is not reusable for {case_identifier}"
            )
        for marker in ("error", "stop_reason", "stopped"):
            value = row.get(marker)
            if value not in (None, "", False, []):
                raise PriorStructuralError(
                    f"prior checkpoint candidate {method} contains {marker} for {case_identifier}"
                )


def read_checkpoint(
    output: os.PathLike[str] | str,
    case_id: str,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _safe_checkpoint_path(Path(output), str(case_id))
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PriorIdentityError(f"invalid prior checkpoint {path}") from exc
    if not isinstance(record, Mapping):
        raise PriorIdentityError(f"prior checkpoint record is not an object for {case_id}")
    if record.get("schema_version") != 1:
        raise PriorIdentityError(f"prior checkpoint schema mismatch for {case_id}")
    expected_identity = sha256_bytes(canonical_json(identity).encode("utf-8"))
    if (
        record.get("identity_sha256") != expected_identity
        or canonical_json(record.get("identity")) != canonical_json(_json_safe(identity))
    ):
        raise PriorIdentityError(f"prior checkpoint identity mismatch for {case_id}")
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise PriorIdentityError(f"prior checkpoint payload is not an object for {case_id}")
    payload_hash = record.get("payload_sha256")
    expected_payload = sha256_bytes(canonical_json(payload).encode("utf-8"))
    if not isinstance(payload_hash, str) or payload_hash != expected_payload:
        raise PriorIdentityError(f"prior checkpoint payload hash mismatch for {case_id}")
    _validate_reusable_checkpoint_payload(payload, identity, str(case_id))
    return dict(payload)


_write_checkpoint = write_checkpoint
_read_checkpoint = read_checkpoint


def _ordered_rows(checkpoints: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> list[dict[str, Any]]:
    by_case: dict[str, Mapping[str, Any]] = {}
    for item in checkpoints:
        case_id = str(item.get("case_id", ""))
        if case_id in by_case:
            raise PriorStructuralError(f"duplicate prior checkpoint case id {case_id!r}")
        validate_paired_block(item, methods=methods, case_id=case_id)
        by_case[case_id] = item
    rows: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        checkpoint = by_case[case_id]
        row_map = {str(row.get("candidate_id")): row for row in checkpoint.get("rows", ())}
        rows.extend(dict(row_map[method]) for method in methods)
    return rows


def _table_rows(rows: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    for row in rows:
        base = {
            key: value
            for key, value in row.items()
            if key not in {"geometry", "prototype", "pairwise", "truth", "predictions"}
        }
        case_rows.append(dict(base))
        geometry = row.get("geometry")
        if not isinstance(geometry, Mapping):
            geometry = {"status": "not_computed_prior_regression"}
        geometry_rows.append({**base, "geometry": dict(geometry)})
        prototype = row.get("prototype")
        if not isinstance(prototype, Mapping):
            prototype = {"status": "not_computed_prior_regression"}
        prototype_rows.append({
            **base,
            "prototype": dict(prototype),
            "refinement": row.get("refinement", {}),
        })
        selector_rows.append(
            {
                **{key: value for key, value in base.items() if key not in {"candidate_score", "evaluation_score"}},
                "candidate_score": row.get("candidate_score"),
                "evaluation_score": row.get("evaluation_score"),
                "reference_accuracies": row.get("reference_accuracies", row.get("references", {"status": "not_run_prior_regression"})),
                "references": row.get("references", {}),
                "linear_probe_score": row.get("linear_probe_score"),
                "linear_probe": row.get("linear_probe", {"status": "not_run_prior_regression"}),
                "outcomes_redacted": False,
            }
        )
        pairwise = row.get("pairwise", {})
        if isinstance(pairwise, Mapping):
            for pair in pairwise.get("directional_pair_support_hits_evidence", ()):
                if isinstance(pair, Mapping):
                    pair_rows.append({**base, **dict(pair)})
            legacy = pairwise.get("legacy", {})
            if not pairwise.get("directional_pair_support_hits_evidence") and isinstance(legacy, Mapping):
                for direction, value in sorted(legacy.items()):
                    source, _, target = str(direction).partition("->")
                    pair_rows.append(
                        {
                            **base,
                            "source_label": source,
                            "target_label": target,
                            "support": (dict(value).get("support") if isinstance(value, Mapping) else None),
                            "hits": (dict(value).get("hits") if isinstance(value, Mapping) else None),
                            "sparse_adj_hits": (dict(value).get("sparse_adj_hits") if isinstance(value, Mapping) else None),
                            "pairwise_index": (dict(value).get("pairwise_index") if isinstance(value, Mapping) else None),
                            "exact_state_match": (dict(value).get("exact_state_match", False) if isinstance(value, Mapping) else False),
                            "evidence": (
                                dict(value).get("evidence", dict(value).get("overlap_evidence"))
                                if isinstance(value, Mapping)
                                else value
                            ),
                            **(dict(value) if isinstance(value, Mapping) else {"evidence": value}),
                        }
                    )
    return {
        "case_rows.jsonl": sorted(case_rows, key=lambda item: (str(item.get("case_id")), str(item.get("candidate_id")))),
        "geometry_rows.jsonl": sorted(geometry_rows, key=lambda item: (str(item.get("case_id")), str(item.get("candidate_id")))),
        "prototype_rows.jsonl": sorted(prototype_rows, key=lambda item: (str(item.get("case_id")), str(item.get("candidate_id")))),
        "pair_rows.jsonl": sorted(pair_rows, key=lambda item: (str(item.get("case_id")), str(item.get("candidate_id")), str(item.get("source_label")), str(item.get("target_label")))),
        "selector_panels.jsonl": sorted(selector_rows, key=lambda item: (str(item.get("case_id")), str(item.get("candidate_id")))),
        "resource_rows.jsonl": [],
    }


def write_normalized_tables(
    output: os.PathLike[str] | str,
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> dict[str, Any]:
    output_path = Path(output)
    tables = _table_rows(rows, methods)
    result: dict[str, Any] = {}
    for name, values in tables.items():
        payload = b"".join((canonical_json(value) + "\n").encode("utf-8") for value in values)
        path = output_path / "tables" / name
        _atomic_write(path, payload)
        result[name] = {"schema_version": 1, "row_count": len(values), "sha256": sha256_bytes(payload)}
    return result


assemble_tables = write_normalized_tables


def _status_from_gate_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        status = value.get("status", value.get("gate_status"))
        if status is not None and str(status).lower() in {"pass", "fail", "inconclusive"}:
            return str(status).lower()
        statuses = [_status_from_gate_value(item) for item in value.values()]
        statuses = [item for item in statuses if item is not None]
        if "fail" in statuses:
            return "fail"
        if "inconclusive" in statuses:
            return "inconclusive"
        if statuses and all(item == "pass" for item in statuses):
            return "pass"
    elif isinstance(value, str) and value.lower() in {"pass", "fail", "inconclusive"}:
        return value.lower()
    elif value is True:
        return "pass"
    elif value is False:
        return "fail"
    return None


def evaluate_prior_historical_gates(
    rows: Sequence[Mapping[str, Any]],
    *,
    locked_candidate: str,
    panel: PriorPanel | None = None,
    evaluator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Evaluate explicit unchanged gate evidence without selecting methods.

    A runner may provide its full frozen historical statistic implementation via
    ``evaluator``.  With a verified ``panel`` the default path invokes the
    versioned prior gate implementation on the newly generated rows.  A compact
    caller without a panel may supply explicit gate evidence, but raw scores
    alone remain inconclusive and cannot authorize confirmation.
    """

    if evaluator is not None:
        result = _call_factory(
            evaluator,
            rows,
            locked_candidate=locked_candidate,
            panel=panel,
        )
        if not isinstance(result, Mapping):
            raise HistoricalGateFailure("historical gate evaluator did not return an object")
        normalized = _json_safe(dict(result))
        status = _status_from_gate_value(normalized) or "inconclusive"
        return {
            **normalized,
            "status": status,
            "locked_candidate": locked_candidate,
            "runner_up_after_lock": False,
        }

    if panel is not None:
        # Production calls always provide the verified panel, so the default
        # path executes the versioned unchanged gate surface.  It never reads
        # the archived raw outcomes.
        return _evaluate_verified_historical_gate_surface(
            rows,
            locked_candidate=locked_candidate,
            panel=panel,
        )

    candidate_rows = [row for row in rows if str(row.get("candidate_id")) == str(locked_candidate)]
    if not candidate_rows:
        return {
            "status": "inconclusive",
            "locked_candidate": locked_candidate,
            "reason": "locked candidate has no complete rows",
            "runner_up_after_lock": False,
        }
    explicit_values: list[Any] = []
    for row in candidate_rows:
        for key in ("historical_gates", "gates", "historical_gate_status"):
            if key in row:
                explicit_values.append(row[key])
    statuses = [_status_from_gate_value(value) for value in explicit_values]
    statuses = [status for status in statuses if status is not None]
    if "fail" in statuses:
        status = "fail"
    elif "inconclusive" in statuses or not statuses:
        status = "inconclusive"
    else:
        status = "pass"
    return {
        "status": status,
        "locked_candidate": locked_candidate,
        "gate_count": len(explicit_values),
        "reason": "explicit gate evidence consumed; no fallback outcome inference" if explicit_values else "historical gate evidence was not supplied",
        "runner_up_after_lock": False,
    }


evaluate_historical_gates = evaluate_prior_historical_gates


def _historical_gate_records(
    rows: Sequence[Mapping[str, Any]],
    locked_candidate: str,
) -> list[dict[str, Any]]:
    """Adapt the four-arm regression rows to the archived gate vocabulary.

    The prior analysis names its pooled-diagonal reference ``D`` and its
    promoted arm ``E``.  L is the exact old-D construction; the newly locked
    arm is represented as E solely to reuse the *gate formulas*, never to
    claim that it was selected by the old screen.  All observed values in the
    returned records come from the newly executed regression rows.
    """

    result: list[dict[str, Any]] = []
    for row in rows:
        candidate = str(row.get("candidate_id", ""))
        if candidate == "L":
            mapped = "D"
        elif candidate == str(locked_candidate):
            mapped = "E"
        else:
            mapped = candidate
        record = dict(row)
        record["candidate_id"] = mapped
        record["candidate"] = mapped
        record["method_id"] = mapped
        pairwise_index = record.get("pairwise_index")
        if not isinstance(pairwise_index, Mapping):
            pairwise = record.get("pairwise")
            if isinstance(pairwise, Mapping):
                pairwise_index = pairwise.get("legacy", {})
        if isinstance(pairwise_index, Mapping):
            record["pairwise_index"] = _json_safe(pairwise_index)
            # The old analysis reads ``pairwise`` and ``pairwise_index`` as a
            # direction-to-index mapping.  Keep this compatibility projection
            # local; the normalized current table retains its richer list.
            record["pairwise"] = _json_safe(pairwise_index)
        result.append(_json_safe(record))
    return result


def _evaluate_verified_historical_gate_surface(
    rows: Sequence[Mapping[str, Any]],
    *,
    locked_candidate: str,
    panel: PriorPanel,
) -> dict[str, Any]:
    """Apply the unchanged prior gate formulas to the new paired rows.

    The archived analysis module is a pure summary/gate implementation.  The
    old raw outcomes are never passed to it: only rows just generated by this
    stage are adapted from L/E to its historical D/E labels.  This keeps the
    prior panel's gate definitions (bootstrap seed, thresholds, and all
    family/shift/stratum checks) versioned by the old protocol while avoiding
    any old outcome as a selection input.
    """

    try:
        from experiments.nuisance_conditioned_distance import analysis as old_analysis
    except (ImportError, AttributeError) as exc:
        return {
            "status": "inconclusive",
            "locked_candidate": locked_candidate,
            "reason": "verified historical gate implementation is unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "runner_up_after_lock": False,
        }
    try:
        old_protocol = _read_object(panel.source_evidence.paths["prior_protocol"], "prior protocol")
        records = _historical_gate_records(rows, locked_candidate)
        # Preserve the old stage/grid metadata for the unchanged completeness
        # predicates.  No old result rows are included in this summary input.
        old_manifest = dict(panel.manifest)
        summary = old_analysis.build_analysis_summary(
            records,
            manifest=old_manifest,
            protocol=old_protocol,
            source_hashes_map=dict(panel.source_evidence.implementation_source_hashes),
        )
        surface = old_analysis.evaluate_historical_gates(
            summary,
            protocol=old_protocol,
            candidate_ids=("E",),
        )
        gate = surface.get("E") if isinstance(surface, Mapping) else None
        if not isinstance(gate, Mapping):
            return {
                "status": "inconclusive",
                "locked_candidate": locked_candidate,
                "reason": "historical gate implementation returned no locked-candidate surface",
                "runner_up_after_lock": False,
            }
        result = _json_safe(dict(gate))
        result["status"] = _status_from_gate_value(result) or "inconclusive"
        result["locked_candidate"] = locked_candidate
        result["gate_surface"] = "prior_nuisance_conditioned_distance.analysis.evaluate_historical_gates.v1"
        result["mapped_legacy_reference"] = "L->D"
        result["mapped_locked_candidate"] = f"{locked_candidate}->E"
        result["runner_up_after_lock"] = False
        return result
    except Exception as exc:
        # Missing metrics or a changed old implementation are evidence gaps,
        # not a reason to guess a pass.  A caller may still provide an
        # explicitly reviewed evaluator in a controlled test/integration seam.
        return {
            "status": "inconclusive",
            "locked_candidate": locked_candidate,
            "reason": "historical gate surface could not be evaluated",
            "error": f"{type(exc).__name__}: {exc}",
            "gate_surface": "prior_nuisance_conditioned_distance.analysis.evaluate_historical_gates.v1",
            "runner_up_after_lock": False,
        }


def _stage_manifest(
    *,
    authorization: PriorAuthorization,
    evidence: PriorSourceEvidence,
    panel: PriorPanel,
    methods: Sequence[str],
    code_identity_sha256: str,
    protocol_sha256: str,
    status: str,
    observed_case_count: int,
    observed_method_row_count: int,
    table_manifest: Mapping[str, Any] | None = None,
    determinism: Mapping[str, Any] | None = None,
    stopped: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "prior_regression",
        "artifact_status": status,
        "case_count": PRIOR_SCREEN_CASE_COUNT,
        "method_ids": list(methods),
        "method_row_count": PRIOR_SCREEN_CASE_COUNT * len(methods),
        "observed_case_count": int(observed_case_count),
        "observed_method_row_count": int(observed_method_row_count),
        "fit_split": PRIOR_FIT_SPLIT,
        "evaluation_split": PRIOR_EVALUATION_SPLIT,
        "backend": PRIOR_BACKEND,
        "case_grid_sha256": panel.case_grid_sha256,
        "current_protocol_sha256": protocol_sha256,
        "protocol_sha256": protocol_sha256,
        "code_identity_sha256": code_identity_sha256,
        "promotion_decision_sha256": authorization.promotion_decision_sha256,
        "locked_candidate": authorization.locked_candidate,
        "prior_source_evidence": evidence.as_dict(),
        "prior_case_source": {
            "stage": PRIOR_SCREEN_STAGE,
            "case_count": PRIOR_SCREEN_CASE_COUNT,
            "generator_path": panel.generator_path,
            "generator_sha256": panel.generator_sha256,
            "fit_split": panel.fit_split,
            "evaluation_split": panel.evaluation_split,
            "backend": panel.backend,
        },
        "table_manifest": _json_safe(table_manifest or {}),
        "determinism": _json_safe(determinism or {"status": "inconclusive"}),
        "stopped": _json_safe(stopped),
        "runner_up_after_lock": False,
        "output_root_excluded_from_code_identity": True,
    }


def _read_existing_output_object(path: Path, label: str) -> dict[str, Any]:
    """Read an existing output object without ever replacing it."""

    if not path.is_file():
        raise PriorIdentityError(f"existing {label} is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PriorIdentityError(f"existing {label} is malformed: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PriorIdentityError(f"existing {label} is not an object: {path}")
    return dict(payload)


def _require_existing_identity(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    """Fail closed when an existing output has a different run identity."""

    for key, value in expected.items():
        observed = payload.get(key)
        if canonical_json(observed) != canonical_json(value):
            raise PriorIdentityError(
                f"existing {label} {key} does not match current prior-regression identity"
            )


def _preflight_existing_output(
    output: Path,
    *,
    authorization: PriorAuthorization,
    evidence: PriorSourceEvidence,
    panel: PriorPanel,
    methods: Sequence[str],
    code_identity_sha256: str,
    protocol_sha256: str,
) -> None:
    """Validate an existing output before any new bytes are written.

    A completed decision is an immutable prerequisite.  A stale or malformed
    decision/manifest must not be repaired in place by a new lock, protocol,
    source panel, or code identity.  Matching partial manifests remain
    resumable; matching final decisions deliberately fail with
    ``FileExistsError`` so the existing artifact remains byte-for-byte intact.
    """

    if output.exists() and not output.is_dir():
        raise PriorIdentityError(f"prior-regression output is not a directory: {output}")
    decision_path = output / "prior_regression_decision.json"
    if decision_path.exists():
        decision = _read_existing_output_object(decision_path, "prior-regression decision")
        decision_bytes = decision_path.read_bytes()
        canonical_bytes = (canonical_json(decision) + "\n").encode("utf-8")
        if decision_bytes != canonical_bytes:
            raise PriorIdentityError(
                f"existing prior-regression decision is not canonical: {decision_path}"
            )
        _require_existing_identity(
            decision,
            expected={
                "schema_version": 1,
                "stage": "prior_regression",
                "decision_stage": "prior_regression",
                "locked_candidate": authorization.locked_candidate,
                "promotion_decision_sha256": authorization.promotion_decision_sha256,
                "protocol_sha256": protocol_sha256,
                "code_identity_sha256": code_identity_sha256,
                "prior_source_evidence": evidence.as_dict(),
                "case_grid_sha256": panel.case_grid_sha256,
                "method_ids": list(methods),
                "runner_up_after_lock": False,
            },
            label="prior-regression decision",
        )
        status = str(decision.get("status", "")).lower()
        if status not in {"pass", "passed", "go", "rejected", "fail", "failed", "stopped", "inconclusive"}:
            raise PriorIdentityError(
                "existing prior-regression decision has an unsupported status"
            )
        selected = decision.get("selected_candidate")
        if status in {"pass", "passed", "go"}:
            if selected != authorization.locked_candidate:
                raise PriorIdentityError(
                    "existing prior-regression decision selects a different candidate"
                )
        elif selected is not None:
            raise PriorIdentityError(
                "existing rejected prior-regression decision promotes a candidate"
            )
        # Do not overwrite an existing final decision, even when its identity
        # matches.  This preserves the one-shot artifact contract.
        raise FileExistsError(
            f"immutable prior decision artifact already exists: {decision_path}"
        )

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = _read_existing_output_object(manifest_path, "prior-regression manifest")
        _require_existing_identity(
            manifest,
            expected={
                "schema_version": 1,
                "stage": "prior_regression",
                "locked_candidate": authorization.locked_candidate,
                "promotion_decision_sha256": authorization.promotion_decision_sha256,
                "protocol_sha256": protocol_sha256,
                "current_protocol_sha256": protocol_sha256,
                "code_identity_sha256": code_identity_sha256,
                "prior_source_evidence": evidence.as_dict(),
                "case_grid_sha256": panel.case_grid_sha256,
                "method_ids": list(methods),
                "runner_up_after_lock": False,
            },
            label="prior-regression manifest",
        )


def run_prior_regression(
    output: os.PathLike[str] | str = DEFAULT_OUTPUT_ROOT,
    promotion_decision: Mapping[str, Any] | os.PathLike[str] | str | None = None,
    *,
    root: os.PathLike[str] | str = ROOT,
    current_protocol_path: os.PathLike[str] | str = CURRENT_PROTOCOL_PATH,
    raw_results_path: os.PathLike[str] | str | None = None,
    manifest_path: os.PathLike[str] | str | None = None,
    prior_protocol_path: os.PathLike[str] | str | None = None,
    verify_implementation_sources: bool | None = None,
    code_identity_sha256: str | None = None,
    protocol_sha256: str | None = None,
    evidence: PriorSourceEvidence | None = None,
    panel: PriorPanel | None = None,
    cases: Sequence[Any] | None = None,
    case_factory: Callable[[], Sequence[Any]] | None = None,
    dataset_factory: Callable[..., Any] | None = None,
    estimator_factory: Callable[..., Any] | None = None,
    adapter_factory: Callable[..., Any] | None = None,
    upstream_factory: Callable[..., Any] | None = None,
    historical_gate_evaluator: Callable[..., Any] | None = None,
    resume: bool = True,
    warmup: bool = True,
    raise_on_stop: bool = False,
) -> dict[str, Path]:
    """Execute exactly A/B/L/locked on the immutable prior screen panel."""

    if promotion_decision is None:
        raise PriorAuthorizationError("prior regression requires a promotion decision")
    # Authorization is intentionally first: an unauthorized caller cannot cause
    # old source/artifact reads or any generator call.
    try:
        actual_protocol_sha256 = sha256_path(
            _resolve_path(Path(root).resolve(), current_protocol_path)
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PriorAuthorizationError("cannot hash current protocol identity") from exc
    if protocol_sha256 is None:
        protocol_sha256 = actual_protocol_sha256
    elif str(protocol_sha256) != actual_protocol_sha256:
        raise PriorAuthorizationError(
            "supplied protocol identity does not match current protocol bytes"
        )
    if code_identity_sha256 is None:
        code_identity_sha256 = current_code_identity(root=root)
    elif Path(root).resolve() == ROOT.resolve():
        actual_code_identity = current_code_identity(root=root)
        if str(code_identity_sha256) != actual_code_identity:
            raise PriorAuthorizationError(
                "supplied code identity does not match current declared sources"
            )
    authorization = authorize_prior_regression(
        promotion_decision,
        protocol_sha256=protocol_sha256,
        code_identity_sha256=code_identity_sha256,
    )
    if protocol_sha256 is None:
        protocol_sha256 = authorization.protocol_sha256
    if code_identity_sha256 is None:
        code_identity_sha256 = authorization.code_identity_sha256
    if evidence is None:
        if panel is not None:
            evidence = _validate_supplied_evidence(
                panel.source_evidence,
                root=root,
                current_protocol_path=current_protocol_path,
                verify_implementation_sources=verify_implementation_sources,
            )
        else:
            evidence = verify_prior_source_evidence(
                root=root,
                current_protocol_path=current_protocol_path,
                raw_results_path=raw_results_path,
                manifest_path=manifest_path,
                prior_protocol_path=prior_protocol_path,
                verify_implementation_sources=verify_implementation_sources,
            )
    if panel is None:
        panel = recover_prior_panel(
            root=root,
            evidence=evidence,
            cases=cases,
            case_factory=case_factory,
            current_protocol_path=current_protocol_path,
            raw_results_path=raw_results_path,
            manifest_path=manifest_path,
            prior_protocol_path=prior_protocol_path,
            verify_implementation_sources=verify_implementation_sources,
        )
    elif panel.source_evidence.as_dict() != evidence.as_dict():
        raise PriorIdentityError("supplied prior panel and evidence identities disagree")
    else:
        # A caller-supplied panel is a convenience for integration/tests, not
        # an authorization bypass.  Recheck its evidence immediately before
        # any dataset generation or checkpoint resume.
        evidence = _validate_supplied_evidence(
            evidence,
            root=root,
            current_protocol_path=current_protocol_path,
            verify_implementation_sources=verify_implementation_sources,
        )
    if cases is not None and tuple(_case_id(case) for case in cases) != tuple(_case_id(case) for case in panel.cases):
        raise PriorIdentityError("explicit prior case list differs from recovered panel")
    methods = prior_regression_methods(authorization)
    output_path = Path(output)
    _preflight_existing_output(
        output_path,
        authorization=authorization,
        evidence=evidence,
        panel=panel,
        methods=methods,
        code_identity_sha256=str(code_identity_sha256),
        protocol_sha256=str(protocol_sha256),
    )
    output_path.mkdir(parents=True, exist_ok=True)
    stage_meta = _stage_manifest(
        authorization=authorization,
        evidence=evidence,
        panel=panel,
        methods=methods,
        code_identity_sha256=str(code_identity_sha256),
        protocol_sha256=str(protocol_sha256),
        status="partial",
        observed_case_count=0,
        observed_method_row_count=0,
    )
    _atomic_write(output_path / "manifest.json", (canonical_json(stage_meta) + "\n").encode("utf-8"))

    checkpoints: list[dict[str, Any]] = []
    stopped: dict[str, Any] | None = None
    determinism: dict[str, Any] = {"status": "inconclusive", "warmup_excluded": True}
    if not panel.cases:
        stopped = {
            "status": "stopped",
            "scope": "whole_stage",
            "reason": "incomplete_paired_block",
            "error": "prior panel is empty",
            "completed_case_count": 0,
            "runner_up_after_lock": False,
        }
    if stopped is not None:
        # Keep the existing control flow and make the stop visible in the
        # final atomic manifest/decision below without indexing an empty panel.
        first_case = None
    else:
        first_case = panel.cases[0]
    warmup_payload: dict[str, Any] | None = None
    if warmup and first_case is not None:
        try:
            warmup_payload = run_prior_case(
                first_case,
                authorization,
                dataset_factory=dataset_factory,
                estimator_factory=estimator_factory,
                adapter_factory=adapter_factory,
                upstream_factory=upstream_factory,
                source_evidence=evidence,
                execution_order=prior_execution_order(first_case, authorization),
            )
        except Exception as exc:
            stopped = {
                "status": "stopped",
                "scope": "whole_stage",
                "case_id": _case_id(first_case),
                "reason": "nondeterminism_or_case_failure",
                "error": f"{type(exc).__name__}: {exc}",
                "runner_up_after_lock": False,
            }

    # A failed excluded warm-up is itself a whole-stage stop; do not start a
    # measured block after a construction/generator failure.
    measured_cases = () if stopped is not None else panel.cases
    for position, case in enumerate(measured_cases):
        identity = checkpoint_identity(
            case,
            authorization=authorization,
            source_evidence=evidence,
            code_identity_sha256=str(code_identity_sha256),
            protocol_sha256=str(protocol_sha256),
            methods=methods,
        )
        try:
            existing = read_checkpoint(output_path, _case_id(case), identity) if resume else None
            if existing is not None:
                checkpoint = existing
            else:
                checkpoint = run_prior_case(
                    case,
                    authorization,
                    dataset_factory=dataset_factory,
                    estimator_factory=estimator_factory,
                    adapter_factory=adapter_factory,
                    upstream_factory=upstream_factory,
                    source_evidence=evidence,
                    execution_order=prior_execution_order(case, authorization),
                )
                validate_paired_block(checkpoint, methods=methods, case_id=_case_id(case))
                write_checkpoint(output_path, checkpoint, identity)
            validate_paired_block(checkpoint, methods=methods, case_id=_case_id(case))
            checkpoints.append(checkpoint)
            if position == 0 and warmup_payload is not None:
                verify_deterministic_rows(warmup_payload.get("rows", ()), checkpoint.get("rows", ()))
                determinism = {
                    "status": "pass",
                    "exact": True,
                    "warmup_excluded": True,
                    "candidate_signatures": {
                        str(row.get("candidate_id")): sha256_bytes(canonical_json(_determinism_signature(row)).encode("utf-8"))
                        for row in checkpoint.get("rows", ())
                    },
                }
        except Exception as exc:
            stopped = {
                "status": "stopped",
                "scope": "whole_stage",
                "case_id": _case_id(case),
                "reason": "nondeterminism" if "determin" in str(exc).lower() else "incomplete_paired_block",
                "error": f"{type(exc).__name__}: {exc}",
                "completed_case_count": len(checkpoints),
                "runner_up_after_lock": False,
            }
            error_payload = {
                "case": _case_identity(case),
                "case_id": _case_id(case),
                "rows": [],
                "status": "error",
                "error": stopped["error"],
                "stop_reason": stopped["reason"],
            }
            write_checkpoint(output_path, error_payload, identity)
            checkpoints.append(error_payload)
            break
        if stopped is not None:
            break

    complete = stopped is None and len(checkpoints) == len(panel.cases)
    rows = _ordered_rows([item for item in checkpoints if item.get("status") == "ok"], methods)
    observed_rows = len(rows)
    if complete and observed_rows != PRIOR_SCREEN_CASE_COUNT * len(methods):
        complete = False
        stopped = {
            "status": "stopped",
            "scope": "whole_stage",
            "reason": "incomplete_paired_block",
            "completed_case_count": len(checkpoints),
            "runner_up_after_lock": False,
        }
    table_manifest = write_normalized_tables(output_path, rows, methods)
    artifact_status = "completed" if complete else "stopped"
    stage_meta = _stage_manifest(
        authorization=authorization,
        evidence=evidence,
        panel=panel,
        methods=methods,
        code_identity_sha256=str(code_identity_sha256),
        protocol_sha256=str(protocol_sha256),
        status=artifact_status,
        observed_case_count=len([item for item in checkpoints if item.get("status") == "ok"]),
        observed_method_row_count=observed_rows,
        table_manifest=table_manifest,
        determinism=determinism,
        stopped=stopped,
    )
    _atomic_write(output_path / "manifest.json", (canonical_json(stage_meta) + "\n").encode("utf-8"))

    gate_result: dict[str, Any]
    if complete:
        gate_result = evaluate_prior_historical_gates(
            rows,
            locked_candidate=authorization.locked_candidate,
            panel=panel,
            evaluator=historical_gate_evaluator,
        )
    else:
        gate_result = {
            "status": "inconclusive",
            "locked_candidate": authorization.locked_candidate,
            "reason": "historical gates are not evaluated on an incomplete panel",
        }
    gate_status = str(gate_result.get("status", "inconclusive")).lower()
    lock_pass = complete and gate_status == "pass"
    stage_meta = {
        **stage_meta,
        "historical_gates": _json_safe(gate_result),
    }
    if complete and not lock_pass:
        stopped = {
            "status": "stopped",
            "scope": "post_lock",
            "reason": "historical_gate_failure" if gate_status == "fail" else "historical_gate_inconclusive",
            "locked_candidate": authorization.locked_candidate,
            "runner_up_after_lock": False,
        }
        stage_meta = {
            **stage_meta,
            "stopped": stopped,
            "artifact_status": "completed",
        }
    _atomic_write(output_path / "manifest.json", (canonical_json(stage_meta) + "\n").encode("utf-8"))

    raw_payload = {
        "schema_version": 1,
        "stage": "prior_regression",
        "artifact_status": artifact_status,
        "manifest": stage_meta,
        "table_manifest": table_manifest,
        "provenance": {
            "protocol_sha256": str(protocol_sha256),
            "code_identity_sha256": str(code_identity_sha256),
            "promotion_decision_sha256": authorization.promotion_decision_sha256,
            "prior_source_evidence": evidence.as_dict(),
        },
        "expected_case_count": PRIOR_SCREEN_CASE_COUNT,
        "observed_case_count": len([item for item in checkpoints if item.get("status") == "ok"]),
        "expected_method_row_count": PRIOR_SCREEN_CASE_COUNT * len(methods),
        "observed_method_row_count": observed_rows,
        "checkpoint_count": len(checkpoints),
        "warmup": {
            "enabled": bool(warmup),
            "excluded_from_tables": True,
            "case_id": _case_id(first_case) if first_case is not None else None,
        },
        "stopped": stopped,
        "runner_up_after_lock": False,
    }
    raw_path = output_path / "raw_results.json"
    _atomic_write(raw_path, (canonical_json(raw_payload) + "\n").encode("utf-8"))
    summary_payload = {
        "schema_version": 1,
        "stage": "prior_regression",
        "status": "pass" if lock_pass else "rejected" if complete else "stopped",
        "artifact_status": artifact_status,
        "locked_candidate": authorization.locked_candidate,
        "method_ids": list(methods),
        "case_count": len([item for item in checkpoints if item.get("status") == "ok"]),
        "method_row_count": observed_rows,
        "historical_gates": _json_safe(gate_result),
        "determinism": _json_safe(determinism),
        "runner_up_after_lock": False,
    }
    summary_path = output_path / "summary.json"
    _atomic_write(summary_path, (canonical_json(summary_payload) + "\n").encode("utf-8"))
    stopped_path = output_path / "stopped.json"
    if stopped is not None:
        _atomic_write(stopped_path, (canonical_json(stopped) + "\n").encode("utf-8"))

    # Publish the immutable prerequisite last.  If any normalized artifact
    # write above fails, no final decision can advertise a seemingly complete
    # prior regression; if a decision already exists, preflight has already
    # prevented all writes to this output root.
    decision_payload = {
        "schema_version": 1,
        "stage": "prior_regression",
        "decision_stage": "prior_regression",
        "status": "pass" if lock_pass else "rejected",
        "locked_candidate": authorization.locked_candidate,
        # A failed/inconclusive historical gate retains the authorization
        # lock for auditability but never turns it into a promotion.  Only a
        # complete pass may populate selected_candidate.
        "selected_candidate": authorization.locked_candidate if lock_pass else None,
        "promotion_decision_sha256": authorization.promotion_decision_sha256,
        "protocol_sha256": str(protocol_sha256),
        "code_identity_sha256": str(code_identity_sha256),
        "prior_source_evidence": evidence.as_dict(),
        "case_grid_sha256": panel.case_grid_sha256,
        "method_ids": list(methods),
        "observed_case_count": len([item for item in checkpoints if item.get("status") == "ok"]),
        "observed_method_row_count": observed_rows,
        "historical_gates": _json_safe(gate_result),
        "artifact_status": artifact_status,
        "runner_up_after_lock": False,
        "reason": "all unchanged historical gates passed" if lock_pass else str(gate_result.get("reason", "locked candidate rejected")),
    }
    decision_path = output_path / "prior_regression_decision.json"
    _atomic_write_immutable(
        decision_path,
        (canonical_json(decision_payload) + "\n").encode("utf-8"),
    )

    paths: dict[str, Path] = {
        "manifest": output_path / "manifest.json",
        "raw": raw_path,
        "summary": summary_path,
        "decision": decision_path,
        "tables": output_path / "tables",
    }
    if stopped is not None:
        paths["stopped"] = stopped_path
    if raise_on_stop and not lock_pass:
        raise HistoricalGateFailure(
            str(stopped.get("reason") if stopped is not None else "prior regression stopped")
        )
    return paths


run_stage = run_prior_regression
run = run_prior_regression


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-decision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--raw-results", type=Path)
    parser.add_argument("--prior-manifest", type=Path)
    parser.add_argument("--prior-protocol", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)
    result = run_prior_regression(
        args.output,
        args.promotion_decision,
        root=args.root,
        raw_results_path=args.raw_results,
        manifest_path=args.prior_manifest,
        prior_protocol_path=args.prior_protocol,
        resume=not args.no_resume,
        warmup=not args.no_warmup,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - explicit CLI only
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "CURRENT_PROTOCOL_PATH",
    "PRIOR_BACKEND",
    "PRIOR_EVALUATION_SPLIT",
    "PRIOR_FIT_SPLIT",
    "PRIOR_IMPLEMENTATION_SOURCE_HASHES",
    "PRIOR_MANIFEST_PATH",
    "PRIOR_PROTOCOL_PATH",
    "PRIOR_RAW_RESULTS_PATH",
    "PRIOR_RAW_RESULTS_SHA256",
    "PRIOR_MANIFEST_SHA256",
    "PRIOR_PROTOCOL_SHA256",
    "PRIOR_REPORT_SHA256",
    "PRIOR_REGRESSION_CONTROL_METHODS",
    "PRIOR_SCREEN_CASE_COUNT",
    "PRIOR_SCREEN_METHODS",
    "PRIOR_SCREEN_METHOD_ROW_COUNT",
    "HistoricalGateFailure",
    "PriorAuthorization",
    "PriorAuthorizationError",
    "PriorEvidenceError",
    "PriorIdentityError",
    "PriorPanel",
    "PriorRegressionError",
    "PriorRegressionResult",
    "PriorSourceEvidence",
    "PriorStructuralError",
    "PROMOTABLE_CANDIDATES",
    "assemble_tables",
    "assert_direct_control_construction",
    "assert_exact_parity",
    "authorize_lock",
    "authorize_prior_regression",
    "build_prior_estimator",
    "canonical_json",
    "case_grid_sha256",
    "checkpoint_identity",
    "current_code_identity",
    "decision_sha256",
    "determinism_signature",
    "estimator_for",
    "evaluate_historical_gates",
    "evaluate_prior_historical_gates",
    "execution_order_for_case",
    "main",
    "methods_for_prior_regression",
    "prior_backend_kwargs",
    "prior_case_grid_sha256",
    "prior_case_identity",
    "prior_execution_order",
    "prior_regression_methods",
    "read_checkpoint",
    "recover_prior_cases",
    "recover_prior_panel",
    "run",
    "run_case",
    "run_prior_case",
    "run_prior_regression",
    "run_stage",
    "sha256_bytes",
    "sha256_path",
    "validate_development_promotion_decision",
    "validate_paired_block",
    "validate_promotion_decision",
    "verify_deterministic_rows",
    "verify_parity",
    "verify_parity_rows",
    "verify_prior_hashes",
    "verify_prior_source_evidence",
    "verify_source_evidence",
    "write_checkpoint",
    "write_normalized_tables",
]
