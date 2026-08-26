"""Frozen case planning, provenance, and stage authorization primitives.

This module intentionally contains no candidate fitting or outcome logic.  The
manifest is a pre-outcome description of the data grid and execution schedule;
decision validation only checks that explicitly supplied prerequisite artifacts
match the frozen protocol.  Generated output roots are never discovered by a
recursive source hash.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from . import fixtures
from .fixtures import (
    BALANCE_ORDER,
    CaseSpec,
    CONFIRMATION_BASE_METHODS,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_METHODS,
    DEVELOPMENT_SCALE_RANKS,
    DEVELOPMENT_SEEDS,
    DEVELOPMENT_SIZE_RANKS,
    METHOD_IDS,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    SCENARIO_ORDER,
    SIGNAL_STATE_ORDER,
    SMOKE_METHODS,
    ConfirmationAuthorization,
    _make_confirmation_authorization,
    confirmation_cases,
    development_cases,
    smoke_cases,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "codex/experiment-heteroscedastic-distance-conditioning"
EXPECTED_FOLLOWUP_STARTING_COMMIT = "46aa626f07ab91b34371dc8f2065b42d179fe510"
EXPECTED_ORIGINAL_DEVELOP_BASE = "165fa344653a72dd2abc04a20387d90b891b9acc"
OUTPUT_ROOTS: Tuple[str, ...] = (
    "artifacts/heteroscedastic_distance_conditioning",
    "artifacts/heteroscedastic_distance_conditioning/",
)

# This is deliberately an explicit list.  A generated artifact, checkpoint,
# temporary file, or the manifest recording these hashes cannot change the
# experiment code identity.
DECLARED_SOURCE_PATHS: Tuple[str, ...] = (
    "experiments/heteroscedastic_distance_conditioning/__init__.py",
    "experiments/heteroscedastic_distance_conditioning/fixtures.py",
    "experiments/heteroscedastic_distance_conditioning/manifest.py",
    "experiments/heteroscedastic_distance_conditioning/conditioning_adapter.py",
    "experiments/heteroscedastic_distance_conditioning/runner.py",
    "experiments/heteroscedastic_distance_conditioning/diagnostics.py",
    "experiments/heteroscedastic_distance_conditioning/resource_worker.py",
    "experiments/heteroscedastic_distance_conditioning/resource_benchmark.py",
    "experiments/heteroscedastic_distance_conditioning/prior_regression.py",
    "experiments/heteroscedastic_distance_conditioning/statistics.py",
    "experiments/heteroscedastic_distance_conditioning/reporting.py",
    "experiments/heteroscedastic_distance_conditioning/PLAN.md",
    "experiments/heteroscedastic_distance_conditioning/README.md",
    "experiments/heteroscedastic_distance_conditioning/protocol.json",
    "experiments/heteroscedastic_distance_conditioning/protocol.sha256",
)

EXPECTED_DEVELOPMENT_CASE_COUNT = 5184
EXPECTED_CONFIRMATION_CASE_COUNT = 10368
EXPECTED_SMOKE_CASE_COUNT = 30


def canonical_json(value: Any) -> str:
    """Serialize JSON-safe values with one byte-stable representation."""

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
    # Avoid importing NumPy in this small manifest utility.  Values produced by
    # the fixture dataclasses are ordinary Python scalars; the two duck-typed
    # branches are for callers passing NumPy scalar metadata.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Union[str, os.PathLike[str]]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_sha256(path: Union[str, os.PathLike[str]] = PROTOCOL_PATH) -> str:
    """Hash protocol bytes, not parsed/reformatted JSON."""

    return sha256_path(path)


def verify_protocol_hash(path: Union[str, os.PathLike[str]] = PROTOCOL_PATH) -> str:
    digest = protocol_sha256(path)
    if Path(path).resolve() == PROTOCOL_PATH.resolve() and digest != PROTOCOL_SHA256:
        raise RuntimeError(
            f"frozen protocol hash mismatch: expected {PROTOCOL_SHA256}, got {digest}"
        )
    return digest


def _relative_source_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    # Explicit paths are constrained to the requested repository root.  This
    # prevents a caller from silently adding a source outside the experiment.
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"source path escapes repository root: {relative!r}") from exc
    return candidate


def source_hashes(
    root: Union[str, os.PathLike[str]] = ROOT,
    source_paths: Optional[Sequence[str]] = None,
    *,
    require_existing: bool = False,
) -> Dict[str, Optional[str]]:
    """Hash only declared source files; never walk output/cache directories.

    Missing optional modules are represented as ``None`` until their workers
    create them.  A run may request ``require_existing=True`` at its execution
    gate to fail loudly rather than recording an incomplete identity.
    """

    root_path = Path(root).resolve()
    relatives = tuple(source_paths or DECLARED_SOURCE_PATHS)
    result: Dict[str, Optional[str]] = {}
    for relative in relatives:
        if str(relative).startswith("/"):
            raise ValueError("source_paths must be repository-relative")
        if any(str(relative) == output or str(relative).startswith(output) for output in OUTPUT_ROOTS):
            raise ValueError(f"generated output root cannot be a source: {relative!r}")
        path = _relative_source_path(root_path, str(relative))
        if not path.is_file():
            if require_existing:
                raise RuntimeError(f"missing declared experiment source: {relative}")
            result[str(relative)] = None
        else:
            result[str(relative)] = sha256_path(path)
    return result


def code_identity_sha256(
    root: Union[str, os.PathLike[str]] = ROOT,
    source_hashes_map: Optional[Mapping[str, Optional[str]]] = None,
    *,
    require_existing: bool = False,
) -> str:
    """Hash the declared recipe and individual source hashes only.

    The output root and any file that records this digest are intentionally
    absent from the payload.  Consequently creating checkpoints or resuming a
    run cannot change the identity used for comparison.
    """

    hashes = dict(
        source_hashes_map
        if source_hashes_map is not None
        else source_hashes(root, require_existing=require_existing)
    )
    payload = {
        "schema": "heteroscedastic_declared_sources_v1",
        "starting_commit": EXPECTED_FOLLOWUP_STARTING_COMMIT,
        "original_develop_base": EXPECTED_ORIGINAL_DEVELOP_BASE,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_hashes": dict(sorted(hashes.items())),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def working_tree_hash(
    root: Union[str, os.PathLike[str]] = ROOT,
    source_hashes_map: Optional[Mapping[str, Optional[str]]] = None,
) -> str:
    """Compatibility-free name for the output-independent code identity."""

    return code_identity_sha256(root, source_hashes_map)


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True, stderr=subprocess.PIPE
    ).strip()


def repository_provenance(root: Union[str, os.PathLike[str]] = ROOT) -> Dict[str, Any]:
    """Record repository state while keeping generated output out of identity."""

    root_path = Path(root).resolve()
    try:
        branch = _git_output(root_path, "branch", "--show-current")
        head = _git_output(root_path, "rev-parse", "HEAD")
        develop = _git_output(root_path, "rev-parse", "develop")
        merge_base = _git_output(root_path, "merge-base", "HEAD", "develop")
        followup_merge_base = _git_output(
            root_path,
            "merge-base",
            "HEAD",
            EXPECTED_FOLLOWUP_STARTING_COMMIT,
        )
        status = _git_output(root_path, "status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot collect git provenance") from exc
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"experiment must run on {EXPECTED_BRANCH!r}; found {branch!r}")
    if develop != EXPECTED_ORIGINAL_DEVELOP_BASE or merge_base != EXPECTED_ORIGINAL_DEVELOP_BASE:
        raise RuntimeError(
            "recorded original develop base mismatch: expected "
            f"{EXPECTED_ORIGINAL_DEVELOP_BASE}, develop={develop}, merge-base={merge_base}"
        )
    if followup_merge_base != EXPECTED_FOLLOWUP_STARTING_COMMIT:
        raise RuntimeError(
            "recorded follow-up starting point is not in the experiment history: "
            f"expected {EXPECTED_FOLLOWUP_STARTING_COMMIT}, merge-base={followup_merge_base}"
        )
    hashes = source_hashes(root_path)
    return {
        "branch": branch,
        "starting_commit": EXPECTED_FOLLOWUP_STARTING_COMMIT,
        "followup_starting_commit": EXPECTED_FOLLOWUP_STARTING_COMMIT,
        "original_develop_base": EXPECTED_ORIGINAL_DEVELOP_BASE,
        "develop_commit_at_execution": develop,
        "merge_base_with_develop": merge_base,
        "merge_base_with_followup_start": followup_merge_base,
        "experiment_commit": head,
        "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines() if status else [],
        "git_status_sha256": sha256_bytes(status.encode("utf-8")),
        "source_hashes": hashes,
        "code_identity_sha256": code_identity_sha256(root_path, hashes),
        "output_root_excluded_from_code_identity": True,
    }


# ---------------------------------------------------------------------------
# Rank/case grid validators
# ---------------------------------------------------------------------------


def _as_case(value: Union[CaseSpec, Mapping[str, Any]]) -> CaseSpec:
    if isinstance(value, CaseSpec):
        return value
    data = dict(value)
    # Identity dictionaries use ``distribution`` and omit the rank-derived
    # constructor details.  Callers should normally pass CaseSpec instances;
    # this branch exists for normalized manifest rows.
    required = {
        "stage",
        "seed",
        "scenario",
        "balance",
        "count_level",
        "nuisance_dim",
        "k",
        "signal_state",
        "scale_ranks",
        "size_ranks",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"case row missing identity fields: {sorted(missing)}")
    return CaseSpec(
        str(data["stage"]),
        int(data["seed"]),
        str(data["scenario"]),
        str(data["balance"]),
        str(data["count_level"]),
        int(data["nuisance_dim"]),
        int(data["k"]),
        str(data["signal_state"]),
        tuple(int(v) for v in data["scale_ranks"]),
        tuple(int(v) for v in data["size_ranks"]),
    )


def validate_rank_permutations(
    stage: str = "development",
    *,
    scale_ranks: Optional[Sequence[Sequence[int]]] = None,
    size_ranks: Optional[Sequence[Sequence[int]]] = None,
) -> bool:
    """Validate exact frozen rank marginals and joint counts."""

    stage_value = str(stage).lower()
    if stage_value == "development":
        scales = tuple(tuple(int(v) for v in row) for row in (scale_ranks or DEVELOPMENT_SCALE_RANKS))
        sizes = tuple(tuple(int(v) for v in row) for row in (size_ranks or DEVELOPMENT_SIZE_RANKS))
        expected_rows = len(DEVELOPMENT_SEEDS)
        expected_each = 3
    elif stage_value == "confirmation":
        scales = tuple(tuple(int(v) for v in row) for row in (scale_ranks or fixtures.CONFIRMATION_SCALE_RANKS))
        sizes = tuple(tuple(int(v) for v in row) for row in (size_ranks or tuple(
            tuple((rank + shift) % 4 for rank in scales[index])
            for index, shift in enumerate(fixtures.CONFIRMATION_SIZE_SHIFTS)
        )))
        expected_rows = len(CONFIRMATION_SEEDS)
        expected_each = 6
    else:
        raise ValueError("stage must be development or confirmation")
    if len(scales) != expected_rows or len(sizes) != expected_rows:
        raise ValueError(f"{stage_value} rank table has the wrong number of seed rows")
    for row in scales + sizes:
        if sorted(row) != [0, 1, 2, 3]:
            raise ValueError("each rank assignment must be a permutation of 0..3")
    for class_id in range(4):
        scale_counts = [sum(row[class_id] == rank for row in scales) for rank in range(4)]
        size_counts = [sum(row[class_id] == rank for row in sizes) for rank in range(4)]
        if scale_counts != [expected_each] * 4 or size_counts != [expected_each] * 4:
            raise ValueError(
                f"{stage_value} rank marginal mismatch for class {class_id}: "
                f"scale={scale_counts}, size={size_counts}"
            )
    joint: Dict[Tuple[int, int], int] = {}
    for scale_row, size_row in zip(scales, sizes):
        for class_id in range(4):
            key = (scale_row[class_id], size_row[class_id])
            joint[key] = joint.get(key, 0) + 1
    if set(joint) != set(itertools_product(range(4), range(4))) or any(
        count != expected_each for count in joint.values()
    ):
        raise ValueError(f"{stage_value} joint rank counts mismatch: {joint}")
    return True


def itertools_product(first: Iterable[int], second: Iterable[int]) -> Tuple[Tuple[int, int], ...]:
    return tuple((a, b) for a in first for b in second)


def validate_case_grid(
    cases: Sequence[Union[CaseSpec, Mapping[str, Any]]],
    *,
    stage: Optional[str] = None,
) -> bool:
    normalized = tuple(_as_case(value) for value in cases)
    if not normalized:
        raise ValueError("case grid cannot be empty")
    stage_value = str(stage or normalized[0].stage).lower()
    expected_count = {
        "smoke": 30,
        "development": EXPECTED_DEVELOPMENT_CASE_COUNT,
        "confirmation": EXPECTED_CONFIRMATION_CASE_COUNT,
    }.get(stage_value)
    if expected_count is None:
        raise ValueError("unknown case-grid stage")
    if len(normalized) != expected_count:
        raise ValueError(f"{stage_value} case count {len(normalized)} != {expected_count}")
    if any(case.stage != stage_value for case in normalized):
        raise ValueError("case stage mismatch")
    ids = [case.case_id for case in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError("case ids are not unique")
    # Identity order is part of the frozen manifest, not merely an incidental
    # loop order.  Reconstructing the expected list here catches a same-sized
    # but semantically altered grid before any candidate rows are generated.
    if stage_value == "smoke":
        expected_ids = [case.case_id for case in smoke_cases()]
    elif stage_value == "development":
        expected_ids = [case.case_id for case in development_cases()]
    else:
        expected_ids = [case.case_id for case in confirmation_cases()]
    if ids != expected_ids:
        raise ValueError(f"{stage_value} case identities/order do not match the frozen grid")
    for case in normalized:
        if stage_value in {"smoke", "development"}:
            expected_scale = DEVELOPMENT_SCALE_RANKS[DEVELOPMENT_SEEDS.index(case.seed)]
            expected_size = DEVELOPMENT_SIZE_RANKS[DEVELOPMENT_SEEDS.index(case.seed)]
        else:
            seed_index = CONFIRMATION_SEEDS.index(case.seed)
            expected_scale = fixtures.CONFIRMATION_SCALE_RANKS[seed_index]
            shift = fixtures.CONFIRMATION_SIZE_SHIFTS[seed_index]
            expected_size = tuple((rank + shift) % 4 for rank in expected_scale)
        if tuple(case.scale_ranks) != tuple(expected_scale) or tuple(case.size_ranks) != tuple(expected_size):
            raise ValueError(f"{stage_value} rank assignment mismatch for seed {case.seed}")
    expected_seeds = {
        "smoke": {21000, 21001, 21002},
        "development": set(DEVELOPMENT_SEEDS),
        "confirmation": set(CONFIRMATION_SEEDS),
    }[stage_value]
    if {case.seed for case in normalized} != expected_seeds:
        raise ValueError("case seed set mismatch")
    expected_per_seed = {"smoke": None, "development": 432, "confirmation": 432}[stage_value]
    if expected_per_seed is not None:
        counts: Dict[int, int] = {}
        for case in normalized:
            counts[case.seed] = counts.get(case.seed, 0) + 1
        if any(value != expected_per_seed for value in counts.values()):
            raise ValueError(f"per-seed case count mismatch: {counts}")
    if stage_value == "development":
        validate_rank_permutations(stage=stage_value)
    elif stage_value == "confirmation":
        validate_rank_permutations(stage=stage_value)
    return True


def case_identity_rows(
    stage: str,
    *,
    authorization: Optional[ConfirmationAuthorization] = None,
) -> Tuple[Dict[str, Any], ...]:
    stage_value = str(stage).lower()
    if stage_value == "smoke":
        cases = smoke_cases()
    elif stage_value == "development":
        cases = development_cases()
    elif stage_value == "confirmation":
        cases = confirmation_cases(authorization)
    else:
        raise ValueError("stage must be smoke, development, or confirmation")
    validate_case_grid(cases, stage=stage_value)
    return tuple(case.identity() for case in cases)


def case_grid_sha256(
    cases: Sequence[Union[CaseSpec, Mapping[str, Any]]],
) -> str:
    rows = []
    for value in cases:
        case = _as_case(value)
        rows.append(case.identity())
    return sha256_bytes((canonical_json(rows) + "\n").encode("utf-8"))


def exact_smoke_case_identities() -> Tuple[str, ...]:
    return tuple(case.case_id for case in smoke_cases())


def methods_for_stage(stage: str, locked_candidate: Optional[str] = None) -> Tuple[str, ...]:
    stage_value = str(stage).lower()
    if stage_value in {"smoke", "development"}:
        return tuple(DEVELOPMENT_METHODS)
    if stage_value == "confirmation":
        methods = list(CONFIRMATION_BASE_METHODS)
        if locked_candidate is not None:
            candidate = str(locked_candidate)
            promotable = set(METHOD_IDS) - {"A", "B", "L"}
            if candidate not in promotable:
                raise ValueError("confirmation lock must name one promotable candidate")
            if candidate not in methods:
                methods.append(candidate)
        return tuple(methods)
    raise ValueError("unknown stage")


# ---------------------------------------------------------------------------
# Deterministic counterbalancing
# ---------------------------------------------------------------------------


def _schedule_offset(case_id: str, method_count: int, schedule_seed: int) -> int:
    digest = hashlib.sha256(f"{int(schedule_seed)}\0{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % method_count


def planned_execution_order(
    cases: Sequence[Union[CaseSpec, Mapping[str, Any], str]],
    methods: Sequence[str],
    *,
    schedule_seed: int = 20260812,
) -> Tuple[Dict[str, Any], ...]:
    """Return a byte-stable cyclic method/position schedule.

    Every case receives each method exactly once.  Cases are assigned a
    SHA-256-derived cyclic rank and the canonical method cycle advances by one
    position at each rank.  Thus each method's counts across execution
    positions differ by at most one, including when the number of cases is not
    divisible by the method count.
    """

    method_tuple = tuple(str(method) for method in methods)
    if not method_tuple or len(set(method_tuple)) != len(method_tuple):
        raise ValueError("methods must be a non-empty unique sequence")
    normalized_ids: List[str] = []
    for value in cases:
        if isinstance(value, str):
            normalized_ids.append(value)
        else:
            normalized_ids.append(_as_case(value).case_id)
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("case ids must be unique")
    rows: List[Dict[str, Any]] = []
    ranked_ids = sorted(
        normalized_ids,
        key=lambda case_id: (
            hashlib.sha256(f"{int(schedule_seed)}\0{case_id}".encode("utf-8")).digest(),
            case_id,
        ),
    )
    offsets = {case_id: rank % len(method_tuple) for rank, case_id in enumerate(ranked_ids)}
    for case_id in sorted(normalized_ids):
        offset = offsets[case_id]
        for position in range(len(method_tuple)):
            method = method_tuple[(offset + position) % len(method_tuple)]
            rows.append(
                {
                    "case_id": case_id,
                    "method_id": method,
                    "position": position,
                    "schedule_seed": int(schedule_seed),
                }
            )
    return tuple(rows)


def method_position_counts(schedule: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for row in schedule:
        method = str(row["method_id"])
        position = str(int(row["position"]))
        counts.setdefault(method, {})[position] = counts.setdefault(method, {}).get(position, 0) + 1
    return counts


def validate_execution_order(
    schedule: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
    methods: Sequence[str],
    *,
    schedule_seed: Optional[int] = None,
) -> bool:
    method_tuple = tuple(str(method) for method in methods)
    expected = planned_execution_order(case_ids, method_tuple, schedule_seed=20260812 if schedule_seed is None else schedule_seed)
    normalized = tuple(
        {
            "case_id": str(row["case_id"]),
            "method_id": str(row["method_id"]),
            "position": int(row["position"]),
            "schedule_seed": int(row.get("schedule_seed", 20260812 if schedule_seed is None else schedule_seed)),
        }
        for row in schedule
    )
    if normalized != expected:
        raise ValueError("execution schedule does not match the frozen hash-seeded cyclic order")
    positions = [int(row["position"]) for row in normalized]
    if set(positions) != set(range(len(method_tuple))):
        raise ValueError("execution positions are not a complete method cycle")
    # Method counts in each position have the max-difference-one guarantee.
    position_counts = {position: positions.count(position) for position in range(len(method_tuple))}
    if max(position_counts.values()) - min(position_counts.values()) > 1:
        raise ValueError("execution position counts differ by more than one")
    return True


# ---------------------------------------------------------------------------
# Decision artifacts and explicit confirmation authorization
# ---------------------------------------------------------------------------


def _artifact_payload(value: Union[Mapping[str, Any], str, os.PathLike[str]]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"decision artifact is not a JSON object: {path}")
    return payload


def _hash_field(payload: Mapping[str, Any], *names: str) -> Optional[str]:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return str(value)
    return None


def _require_common_identity(
    payload: Mapping[str, Any],
    *,
    protocol_hash: str,
    code_hash: Optional[str] = None,
) -> None:
    observed_protocol = _hash_field(payload, "protocol_sha256", "protocol_hash")
    if observed_protocol != str(protocol_hash):
        raise PermissionError(
            f"decision protocol hash mismatch: expected {protocol_hash}, got {observed_protocol}"
        )
    if code_hash is not None:
        observed_code = _hash_field(payload, "code_identity_sha256", "code_identity_hash")
        if observed_code != str(code_hash):
            raise PermissionError(
                f"decision code identity mismatch: expected {code_hash}, got {observed_code}"
            )


def validate_implementation_review(
    decision: Union[Mapping[str, Any], str, os.PathLike[str]],
    *,
    protocol_hash: str = PROTOCOL_SHA256,
    code_identity_hash: Optional[str] = None,
) -> bool:
    payload = _artifact_payload(decision)
    status = str(payload.get("status", "")).lower()
    if status != "go":
        raise PermissionError("implementation review must have status 'go'")
    _require_common_identity(payload, protocol_hash=protocol_hash, code_hash=code_identity_hash)
    return True


def validate_smoke_decision(
    decision: Union[Mapping[str, Any], str, os.PathLike[str]],
    *,
    protocol_hash: str = PROTOCOL_SHA256,
    code_identity_hash: Optional[str] = None,
) -> bool:
    payload = _artifact_payload(decision)
    status = str(payload.get("status", payload.get("decision", ""))).lower()
    if status not in {"pass", "passed", "go"}:
        raise PermissionError("smoke decision must be a structural pass")
    if payload.get("structural_pass") is False:
        raise PermissionError("smoke structural_pass is false")
    _require_common_identity(payload, protocol_hash=protocol_hash, code_hash=code_identity_hash)
    return True


def validate_pre_screen_review(
    decision: Union[Mapping[str, Any], str, os.PathLike[str]],
    *,
    protocol_hash: str = PROTOCOL_SHA256,
    code_identity_hash: Optional[str] = None,
) -> bool:
    payload = _artifact_payload(decision)
    if str(payload.get("status", "")).lower() != "go":
        raise PermissionError("pre-screen review must have status 'go'")
    _require_common_identity(payload, protocol_hash=protocol_hash, code_hash=code_identity_hash)
    return True


PROMOTABLE_CANDIDATES: Tuple[str, ...] = (
    "P25",
    "P50-SW",
    "P50-CB",
    "W50-SW",
    "W50-CB",
    "M50-SW",
    "M50-CB",
)


def validate_promotion_decision(
    decision: Union[Mapping[str, Any], str, os.PathLike[str]],
    *,
    protocol_hash: str = PROTOCOL_SHA256,
    code_identity_hash: Optional[str] = None,
) -> str:
    payload = _artifact_payload(decision)
    _require_common_identity(payload, protocol_hash=protocol_hash, code_hash=code_identity_hash)
    status = str(payload.get("status", "")).lower()
    if status not in {"locked", "pass", "passed", "eligible"}:
        raise PermissionError("promotion decision does not contain a valid lock")
    locked = payload.get("locked_candidate", payload.get("selected_candidate", payload.get("candidate")))
    # The frozen protocol uses exactly one locked candidate.  A list is accepted
    # only when it contains one string, never as a way to reopen ranking.
    if isinstance(locked, (list, tuple)):
        if len(locked) != 1:
            raise PermissionError("promotion decision must lock exactly one candidate")
        locked = locked[0]
    if locked not in PROMOTABLE_CANDIDATES:
        raise PermissionError(f"invalid or non-promotable locked candidate: {locked!r}")
    if payload.get("selected_candidate") not in (None, locked):
        raise PermissionError("selected and locked candidates disagree")
    return str(locked)


def validate_prior_regression_decision(
    decision: Union[Mapping[str, Any], str, os.PathLike[str]],
    *,
    locked_candidate: str,
    promotion_decision_sha256: Optional[str] = None,
    protocol_hash: str = PROTOCOL_SHA256,
    code_identity_hash: Optional[str] = None,
) -> bool:
    payload = _artifact_payload(decision)
    _require_common_identity(payload, protocol_hash=protocol_hash, code_hash=code_identity_hash)
    if str(payload.get("status", "")).lower() not in {"pass", "passed", "go"}:
        raise PermissionError("prior-regression decision must have status 'pass'")
    observed_lock = payload.get("locked_candidate", payload.get("candidate"))
    if observed_lock != locked_candidate:
        raise PermissionError("prior-regression candidate does not match the promotion lock")
    if promotion_decision_sha256 is not None:
        observed = _hash_field(payload, "promotion_decision_sha256", "promotion_decision_hash")
        if observed != str(promotion_decision_sha256):
            raise PermissionError("prior-regression promotion decision hash mismatch")
    return True


def authorize_confirmation(
    promotion_decision: Union[Mapping[str, Any], str, os.PathLike[str]],
    prior_regression_decision: Union[Mapping[str, Any], str, os.PathLike[str]],
    *,
    protocol_hash: str = PROTOCOL_SHA256,
    code_identity_hash: Optional[str] = None,
    promotion_decision_sha256: Optional[str] = None,
) -> ConfirmationAuthorization:
    """Validate prerequisite decisions and return the sole bank-generation token."""

    promotion_payload = _artifact_payload(promotion_decision)
    locked = validate_promotion_decision(
        promotion_payload,
        protocol_hash=protocol_hash,
        code_identity_hash=code_identity_hash,
    )
    if promotion_decision_sha256 is None:
        if isinstance(promotion_decision, (str, os.PathLike)):
            promotion_hash = sha256_path(promotion_decision)
        else:
            promotion_hash = sha256_bytes((canonical_json(promotion_payload) + "\n").encode("utf-8"))
    else:
        promotion_hash = str(promotion_decision_sha256)
    validate_prior_regression_decision(
        prior_regression_decision,
        locked_candidate=locked,
        promotion_decision_sha256=promotion_hash,
        protocol_hash=protocol_hash,
        code_identity_hash=code_identity_hash,
    )
    prior_payload = _artifact_payload(prior_regression_decision)
    if isinstance(prior_regression_decision, (str, os.PathLike)):
        prior_hash = sha256_path(prior_regression_decision)
    else:
        prior_hash = sha256_bytes((canonical_json(prior_payload) + "\n").encode("utf-8"))
    return _make_confirmation_authorization(promotion_hash, prior_hash, protocol_hash)


def validate_confirmation_authorization(
    authorization: Any,
    *,
    protocol_hash: str = PROTOCOL_SHA256,
) -> bool:
    if not isinstance(authorization, ConfirmationAuthorization):
        raise PermissionError("confirmation authorization token has the wrong type")
    if authorization.protocol_sha256 != protocol_hash:
        raise PermissionError("confirmation authorization protocol mismatch")
    if not authorization.promotion_decision_sha256 or not authorization.prior_regression_decision_sha256:
        raise PermissionError("confirmation authorization is missing prerequisite hashes")
    # The marker check is intentionally performed by the fixture seam too; this
    # function keeps the validation primitive useful to callers before generation.
    if not hasattr(authorization, "_marker"):
        raise PermissionError("confirmation authorization marker missing")
    return True


def build_manifest(
    stage: str,
    *,
    cases: Optional[Sequence[Union[CaseSpec, Mapping[str, Any]]]] = None,
    locked_candidate: Optional[str] = None,
    schedule_seed: int = 20260812,
    root: Union[str, os.PathLike[str]] = ROOT,
    require_sources: bool = False,
) -> Dict[str, Any]:
    """Build a pre-outcome manifest payload for the requested stage."""

    stage_value = str(stage).lower()
    if cases is None:
        if stage_value == "smoke":
            case_values: Sequence[Union[CaseSpec, Mapping[str, Any]]] = smoke_cases()
        elif stage_value == "development":
            case_values = development_cases()
        else:
            raise PermissionError("confirmation manifest requires an explicit authorization and case list")
    else:
        case_values = cases
    normalized = tuple(_as_case(value) for value in case_values)
    validate_case_grid(normalized, stage=stage_value)
    methods = methods_for_stage(stage_value, locked_candidate)
    schedule = planned_execution_order(normalized, methods, schedule_seed=schedule_seed)
    hashes = source_hashes(root, require_existing=require_sources)
    return {
        "schema_version": 1,
        "stage": stage_value,
        "protocol_sha256": PROTOCOL_SHA256,
        "case_count": len(normalized),
        "method_ids": list(methods),
        "method_row_count": len(normalized) * len(methods),
        "case_grid_sha256": case_grid_sha256(normalized),
        "planned_order": list(schedule),
        "schedule_seed": int(schedule_seed),
        "rank_tables": {
            "development_scale_ranks": [list(row) for row in DEVELOPMENT_SCALE_RANKS],
            "development_size_ranks": [list(row) for row in DEVELOPMENT_SIZE_RANKS],
        },
        "source_hashes": hashes,
        "code_identity_sha256": code_identity_sha256(root, hashes),
        "output_root_excluded_from_code_identity": True,
        "confirmation_status": "authorized" if stage_value == "confirmation" else "not_requested",
    }


__all__ = [
    "DECLARED_SOURCE_PATHS",
    "EXPECTED_BRANCH",
    "EXPECTED_CONFIRMATION_CASE_COUNT",
    "EXPECTED_DEVELOPMENT_CASE_COUNT",
    "EXPECTED_FOLLOWUP_STARTING_COMMIT",
    "EXPECTED_ORIGINAL_DEVELOP_BASE",
    "EXPECTED_SMOKE_CASE_COUNT",
    "PROMOTABLE_CANDIDATES",
    "build_manifest",
    "canonical_json",
    "case_grid_sha256",
    "case_identity_rows",
    "code_identity_sha256",
    "exact_smoke_case_identities",
    "methods_for_stage",
    "method_position_counts",
    "planned_execution_order",
    "protocol_sha256",
    "repository_provenance",
    "sha256_bytes",
    "sha256_path",
    "source_hashes",
    "validate_case_grid",
    "validate_confirmation_authorization",
    "validate_execution_order",
    "validate_implementation_review",
    "validate_promotion_decision",
    "validate_pre_screen_review",
    "validate_prior_regression_decision",
    "validate_rank_permutations",
    "validate_smoke_decision",
    "verify_protocol_hash",
    "working_tree_hash",
    "authorize_confirmation",
]
