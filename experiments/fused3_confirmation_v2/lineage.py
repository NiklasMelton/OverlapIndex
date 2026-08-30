"""Hash-bound dual lineage primitives for FUSED3 confirmation V2.

The confirmation stage is allowed to consume only two immutable histories:
the original five-dataset design ancestry and the development decision that
selected FUSED3.  This module deliberately contains no candidate-selection
logic.  A lock is supplied by the caller and is checked against the expected
lineage; this module never chooses a candidate or repairs an incomplete lock.

All hashes in this module are hashes of bytes as written.  JSON payload hashes
use the one canonical compact JSON encoding below.  Output/checkpoint paths
are never recursively included in a code identity by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent

# Immutable v1 planning evidence.  These paths are defaults only; callers may
# pass explicit paths when validating a copied v1 artifact, but the bytes and
# hashes must still be checked.
V1_PROTOCOL_PATH = ROOT / "experiments" / "m50_backbone_ranking" / "protocol.json"
V1_FINAL_LOCK_PATH = (
    ROOT
    / "artifacts"
    / "m50_backbone_ranking"
    / "analysis_final"
    / "development_lock.json"
)
V1_CONFIRMATION_REGISTRY_PATH = (
    ROOT / "experiments" / "m50_backbone_ranking" / "confirmation_registry.json"
)

# The v2 candidate-authority record is immutable planning evidence.  Its
# contents include the exact FUSED3 screen/full hashes and the no-reselection
# rule.  The runtime lock envelope is a separate output artifact and is never
# included in code identity.
FUSED3_AUTHORITY_PATH = PACKAGE_DIR / "candidate_authority.json"

SCHEMA_VERSION = 1
AUTHORITY = "FUSED3"
LINEAGE_STATUS = "frozen_v2_design"
LOCK_STATUS = "locked_for_untouched_confirmation"
V1_PROTOCOL_SHA256 = "8634909b99e6c63d9a1f8d39980658884ec3010cc1fd4701efed5044701c61f8"
V1_FINAL_LOCK_SHA256 = "fade0a84de3d678f2247f486bd956e392be447eaa5e7d955e91a297585d052a3"
V1_CONFIRMATION_REGISTRY_SHA256 = "ed9ec2e54cd4316dd8da5841b1d56d2f7599b5f689bdc545852a38723d77f3b9"
FUSED3_AUTHORITY_SHA256 = "0c17dbdbb949896faa525e8cd6fd6349f3e7b227d749bd0473327151d0cd8c80"

# This is intentionally the complete envelope rather than a permissive
# required-subset check.  Extra fields can silently become a second identity
# surface, so callers must version the schema before adding one.
LINEAGE_KEYS = (
    "schema_version",
    "protocol_sha256",
    "code_identity_sha256",
    "v1_protocol_sha256",
    "v1_final_lock_sha256",
    "v1_confirmation_registry_sha256",
    "authority",
    "status",
)
LOCK_KEYS = (
    "schema_version",
    "lineage",
    "candidate_authority_sha256",
    "selected_candidate",
    "runner_up_reselection_allowed",
    "status",
)


def canonical_json(value: Any) -> str:
    """Return the strict byte-stable JSON representation used for identities."""

    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    # Avoid taking a dependency on numpy merely for metadata.  Numpy scalar
    # values expose item(); an unsupported object must fail rather than being
    # stringified into an ambiguous identity.
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return _json_safe(converted)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("identity JSON cannot contain non-finite numbers")
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_payload(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_path(path: os.PathLike[str] | str) -> str:
    """Hash one file without following directories or recursively hashing outputs."""

    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"identity file is missing or not a file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be one lower-case 64-hex SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be one lower-case 64-hex SHA-256 digest")
    return value


def verify_file_hash(
    path: os.PathLike[str] | str,
    expected_sha256: str,
    *,
    field: str = "file",
) -> str:
    """Verify one explicit file before it is treated as lineage evidence."""

    expected = require_hash(expected_sha256, field=f"{field}_sha256")
    observed = sha256_path(path)
    if observed != expected:
        raise ValueError(
            f"{field} hash mismatch: expected {expected}, observed {observed}"
        )
    return observed


def read_json_object(path: os.PathLike[str] | str) -> dict[str, Any]:
    candidate = Path(path)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON object at {candidate}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON identity at {candidate} must be an object")
    return value


def _validate_lineage_shape(lineage: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(lineage, Mapping):
        raise TypeError("lineage must be a mapping")
    if set(lineage) != set(LINEAGE_KEYS):
        raise ValueError(
            "lineage must contain exactly " + ", ".join(LINEAGE_KEYS)
        )
    if lineage.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("lineage schema_version mismatch")
    for field in (
        "protocol_sha256",
        "code_identity_sha256",
        "v1_protocol_sha256",
        "v1_final_lock_sha256",
        "v1_confirmation_registry_sha256",
    ):
        require_hash(lineage.get(field), field=field)
    if lineage.get("authority") != AUTHORITY:
        raise ValueError(f"lineage authority must be {AUTHORITY!r}")
    status = lineage.get("status")
    if status != LINEAGE_STATUS:
        raise ValueError(f"lineage status must be {LINEAGE_STATUS!r}; got {status!r}")
    return {key: lineage[key] for key in LINEAGE_KEYS}


def make_lineage(
    *,
    protocol_sha256: str,
    code_identity_sha256: str,
    v1_protocol_sha256: str,
    v1_final_lock_sha256: str,
    v1_confirmation_registry_sha256: str,
) -> dict[str, Any]:
    """Create the exact eight-key dual-lineage envelope."""

    return _validate_lineage_shape(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256,
            "code_identity_sha256": code_identity_sha256,
            "v1_protocol_sha256": v1_protocol_sha256,
            "v1_final_lock_sha256": v1_final_lock_sha256,
            "v1_confirmation_registry_sha256": v1_confirmation_registry_sha256,
            "authority": AUTHORITY,
            "status": LINEAGE_STATUS,
        }
    )


def validate_lineage(
    lineage: Mapping[str, Any],
    *,
    expected: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Validate shape and, when supplied, exact equality to a sealed lineage."""

    observed = _validate_lineage_shape(lineage)
    if expected is not None:
        sealed = _validate_lineage_shape(expected)
        if observed != sealed:
            raise ValueError("lineage does not match the sealed v2 identity")
    return observed


def build_lineage_from_files(
    *,
    protocol_path: os.PathLike[str] | str,
    protocol_sidecar_path: os.PathLike[str] | str,
    code_identity_sha256: str,
    v1_protocol_path: os.PathLike[str] | str = V1_PROTOCOL_PATH,
    v1_final_lock_path: os.PathLike[str] | str = V1_FINAL_LOCK_PATH,
    v1_confirmation_registry_path: os.PathLike[str] | str = V1_CONFIRMATION_REGISTRY_PATH,
) -> dict[str, Any]:
    """Verify both histories and create a lineage envelope.

    The protocol sidecar is required to be exactly one lower-case digest line.
    The v1 files are hashed as bytes; no fields from those files are interpreted
    here, preventing a caller from silently substituting a semantically similar
    lock.
    """

    protocol_digest = sha256_path(protocol_path)
    sidecar = Path(protocol_sidecar_path)
    if not sidecar.is_file():
        raise ValueError("v2 protocol SHA-256 sidecar is missing")
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or lines[0] != protocol_digest:
        raise ValueError("v2 protocol SHA-256 sidecar mismatch")
    observed_v1_protocol = sha256_path(v1_protocol_path)
    observed_v1_lock = sha256_path(v1_final_lock_path)
    observed_v1_registry = sha256_path(v1_confirmation_registry_path)
    expected_v1 = {
        "v1_protocol_sha256": V1_PROTOCOL_SHA256,
        "v1_final_lock_sha256": V1_FINAL_LOCK_SHA256,
        "v1_confirmation_registry_sha256": V1_CONFIRMATION_REGISTRY_SHA256,
    }
    observed_v1 = {
        "v1_protocol_sha256": observed_v1_protocol,
        "v1_final_lock_sha256": observed_v1_lock,
        "v1_confirmation_registry_sha256": observed_v1_registry,
    }
    if observed_v1 != expected_v1:
        raise ValueError("v1 ancestry bytes differ from the frozen confirmation design")
    return make_lineage(
        protocol_sha256=protocol_digest,
        code_identity_sha256=require_hash(
            code_identity_sha256, field="code_identity_sha256"
        ),
        v1_protocol_sha256=observed_v1_protocol,
        v1_final_lock_sha256=observed_v1_lock,
        v1_confirmation_registry_sha256=observed_v1_registry,
    )


def make_lock_envelope(
    *,
    lineage: Mapping[str, Any],
    candidate_authority_sha256: str,
    selected_candidate: str = AUTHORITY,
    runner_up_reselection_allowed: bool = False,
    status: str = LOCK_STATUS,
) -> dict[str, Any]:
    """Create a strict no-reselection lock around a validated lineage."""

    validated_lineage = validate_lineage(lineage)
    if selected_candidate != AUTHORITY:
        raise ValueError("the v2 lock is sealed to the FUSED3 candidate")
    if type(runner_up_reselection_allowed) is not bool:
        raise TypeError("runner_up_reselection_allowed must be a strict bool")
    if runner_up_reselection_allowed:
        raise ValueError("runner-up reselection is forbidden")
    if status != LOCK_STATUS:
        raise ValueError(f"lock status must be {LOCK_STATUS!r}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "lineage": validated_lineage,
        "candidate_authority_sha256": require_hash(
            candidate_authority_sha256, field="candidate_authority_sha256"
        ),
        "selected_candidate": AUTHORITY,
        "runner_up_reselection_allowed": False,
        "status": LOCK_STATUS,
    }
    return validate_lock_envelope(payload)


def build_lock_from_files(
    *,
    protocol_path: os.PathLike[str] | str,
    protocol_sidecar_path: os.PathLike[str] | str,
    code_identity_sha256: str,
    candidate_authority_path: os.PathLike[str] | str = FUSED3_AUTHORITY_PATH,
    v1_protocol_path: os.PathLike[str] | str = V1_PROTOCOL_PATH,
    v1_final_lock_path: os.PathLike[str] | str = V1_FINAL_LOCK_PATH,
    v1_confirmation_registry_path: os.PathLike[str] | str = V1_CONFIRMATION_REGISTRY_PATH,
) -> dict[str, Any]:
    """Verify both ancestry histories and the sealed FUSED3 authority bytes."""

    dual_lineage = build_lineage_from_files(
        protocol_path=protocol_path,
        protocol_sidecar_path=protocol_sidecar_path,
        code_identity_sha256=code_identity_sha256,
        v1_protocol_path=v1_protocol_path,
        v1_final_lock_path=v1_final_lock_path,
        v1_confirmation_registry_path=v1_confirmation_registry_path,
    )
    return make_lock_envelope(
        lineage=dual_lineage,
        candidate_authority_sha256=authority_sha256(candidate_authority_path),
    )


def validate_lock_envelope(
    lock: Mapping[str, Any],
    *,
    expected_lineage: Optional[Mapping[str, Any]] = None,
    expected_candidate_authority_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Fail closed on any lock mutation, reselection flag, or hash mismatch."""

    if not isinstance(lock, Mapping):
        raise TypeError("lock must be a mapping")
    if set(lock) != set(LOCK_KEYS):
        raise ValueError("lock contains unknown or missing identity keys")
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("lock schema_version mismatch")
    lineage = validate_lineage(lock.get("lineage"), expected=expected_lineage)
    authority_digest = require_hash(
        lock.get("candidate_authority_sha256"),
        field="candidate_authority_sha256",
    )
    if expected_candidate_authority_sha256 is not None:
        expected = require_hash(
            expected_candidate_authority_sha256,
            field="candidate_authority_sha256",
        )
        if authority_digest != expected:
            raise ValueError("candidate authority hash does not match the sealed lock")
    if lock.get("selected_candidate") != AUTHORITY:
        raise ValueError("lock selected_candidate must be FUSED3")
    if type(lock.get("runner_up_reselection_allowed")) is not bool:
        raise TypeError("runner_up_reselection_allowed must be a strict bool")
    if lock.get("runner_up_reselection_allowed") is not False:
        raise ValueError("runner-up reselection is forbidden")
    if lock.get("status") != LOCK_STATUS:
        raise ValueError(f"lock status must be {LOCK_STATUS!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "lineage": lineage,
        "candidate_authority_sha256": authority_digest,
        "selected_candidate": AUTHORITY,
        "runner_up_reselection_allowed": False,
        "status": LOCK_STATUS,
    }


def authority_sha256(
    path: os.PathLike[str] | str = FUSED3_AUTHORITY_PATH,
) -> str:
    """Hash the exact FUSED3 candidate-authority lock bytes."""

    observed = sha256_path(path)
    if Path(path).resolve() == FUSED3_AUTHORITY_PATH.resolve() and observed != FUSED3_AUTHORITY_SHA256:
        raise ValueError("FUSED3 candidate-authority bytes differ from the frozen authority")
    return observed


def validate_candidate_authority_file(
    path: os.PathLike[str] | str = FUSED3_AUTHORITY_PATH,
) -> dict[str, Any]:
    """Validate the exact historical FUSED3 authority before input access."""

    authority_sha256(path)
    value = read_json_object(path)
    expected_keys = {
        "schema_version",
        "lock_id",
        "status",
        "selected_candidate",
        "selected_candidate_recipe_sha256",
        "runner_up_reselection_allowed",
        "design_ancestry",
        "failed_m50_lock",
        "fused3_screen",
        "fused3_full",
        "confirmation_constraints",
    }
    if set(value) != expected_keys:
        raise ValueError("candidate authority has unknown or missing keys")
    if (
        value.get("schema_version") != 1
        or value.get("lock_id") != "fused3_confirmation_v2_candidate_lock"
        or value.get("status") != LOCK_STATUS
        or value.get("selected_candidate") != AUTHORITY
        or value.get("runner_up_reselection_allowed") is not False
    ):
        raise ValueError("candidate authority does not seal FUSED3 without reselection")
    design = value.get("design_ancestry")
    failed = value.get("failed_m50_lock")
    screen = value.get("fused3_screen")
    full = value.get("fused3_full")
    constraints = value.get("confirmation_constraints")
    if not all(isinstance(item, Mapping) for item in (design, failed, screen, full, constraints)):
        raise ValueError("candidate authority nested evidence is malformed")
    if (
        design.get("protocol_sha256") != V1_PROTOCOL_SHA256
        or design.get("planned_registry_sha256") != V1_CONFIRMATION_REGISTRY_SHA256
        or design.get("candidate_authority") is not False
        or failed.get("development_lock_sha256") != V1_FINAL_LOCK_SHA256
        or failed.get("status") != "stopped_guardrail_failure"
        or failed.get("selected_chain") is not None
        or failed.get("candidate_authority") is not False
        or screen.get("decision_status") != "pass_for_full_food_design"
        or full.get("decision_status") != "pass_full_retrospective"
        or full.get("failed_required_gate_count") != 0
        or full.get("resource_status")
        != "passed_all_per_call_and_full_panel_gates_in_all_three_food_arms"
        or constraints.get("candidate_ids") != ["FUSED3", "LP-FULL"]
        or constraints.get("promotable_candidate_ids") != ["FUSED3"]
        or constraints.get("no_reselection") is not True
        or constraints.get("no_recipe_change") is not True
        or constraints.get("no_gate_change") is not True
        or constraints.get("runtime_inheritance") != "fused3_full"
        or constraints.get("confirmation_runtime") != "descriptive_only"
    ):
        raise ValueError("candidate authority historical decisions/gates are invalid")
    return value


def lock_sha256(lock: Mapping[str, Any]) -> str:
    """Hash the canonical newline-terminated lock bytes used on disk."""

    payload = canonical_json(validate_lock_envelope(lock)) + "\n"
    return sha256_bytes(payload.encode("utf-8"))


def write_lock(path: os.PathLike[str] | str, lock: Mapping[str, Any]) -> str:
    """Atomically write one canonical lock and return its exact file hash."""

    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate.exists():
        raise FileExistsError(f"lineage lock already exists: {candidate}")
    payload = (canonical_json(validate_lock_envelope(lock)) + "\n").encode("utf-8")
    temporary = candidate.with_name(f".{candidate.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(candidate)
    observed = sha256_path(candidate)
    if observed != lock_sha256(lock):
        raise RuntimeError("serialized lineage-lock hash mismatch")
    return observed


__all__ = [
    "AUTHORITY",
    "FUSED3_AUTHORITY_PATH",
    "LINEAGE_KEYS",
    "LINEAGE_STATUS",
    "LOCK_KEYS",
    "LOCK_STATUS",
    "PACKAGE_DIR",
    "ROOT",
    "SCHEMA_VERSION",
    "V1_CONFIRMATION_REGISTRY_PATH",
    "V1_CONFIRMATION_REGISTRY_SHA256",
    "V1_FINAL_LOCK_PATH",
    "V1_FINAL_LOCK_SHA256",
    "V1_PROTOCOL_PATH",
    "V1_PROTOCOL_SHA256",
    "authority_sha256",
    "build_lock_from_files",
    "build_lineage_from_files",
    "canonical_json",
    "lock_sha256",
    "make_lineage",
    "make_lock_envelope",
    "read_json_object",
    "require_hash",
    "sha256_bytes",
    "sha256_path",
    "sha256_payload",
    "validate_lineage",
    "validate_candidate_authority_file",
    "validate_lock_envelope",
    "verify_file_hash",
    "write_lock",
]
