"""Frozen statistics, gates, and decision rules for the heteroscedastic study.

The runner writes normalized JSONL tables.  This module is intentionally the
only consumer that turns those tables into outcomes or a lock decision.  The
implementation is conservative at the artifact boundary (duplicate or missing
identities make a surface undefined) and permissive inside a row (the runner
has used a few equivalent spellings for scalar diagnostics over its lifetime).

No function in this module changes :mod:`overlapindex` or runs a candidate.  In
particular, ``smoke`` artifacts are structural only and cannot be used to rank
or lock a candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from . import manifest as frozen_manifest


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_812
BONFERRONI_CLAIMS = 4
BONFERRONI_LEVEL = 1.0 - (1.0 - 0.95) / BONFERRONI_CLAIMS
PRIMARY_INTERVAL_LEVEL = 0.95
WITHIN_ONE_POINT = 0.01
EXACT_TOLERANCE = 1e-12
FAMILY_DRIFT_LIMIT = 0.02
NEIGHBOR_NEGATIVE_CONTROL_LIMIT = 0.01
PROMOTION_TOLERANCE = 0.001
PAIR_EVIDENCE_THRESHOLD = 0.05

CANDIDATE_ORDER = (
    "A", "B", "L", "P25", "P50-SW", "P50-CB", "W50-SW", "W50-CB", "M50-SW", "M50-CB",
)
PROMOTABLE_CANDIDATES = ("P25", "P50-SW", "P50-CB", "W50-SW", "W50-CB", "M50-SW", "M50-CB")
CONTROL_CANDIDATES = ("A", "B", "L")
SCENARIO_ORDER = ("S0", "S1", "S2", "H1", "H2", "T", "C", "X", "ALL")
PRIMARY_NUISANCE_SCENARIOS = ("H1", "H2", "T", "C", "X", "ALL")
HOMOSCEDASTIC_SCENARIOS = ("S0", "S1", "S2")
SEPARATED_SCENARIOS = ("S0", "S1", "S2", "H1", "H2", "T", "C", "X", "ALL")
REFERENCE_HEADS = ("linear", "quadratic", "knn", "rbf")
TABLE_NAMES = (
    "case_rows.jsonl", "geometry_rows.jsonl", "prototype_rows.jsonl",
    "pair_rows.jsonl", "selector_panels.jsonl", "resource_rows.jsonl",
)
RESOURCE_CELLS = ("R0", "R1", "R2", "R3")

# These identities are part of the frozen outcome grid.  Keeping them in one
# place prevents a complete-looking subset (for example, sixteen arbitrary
# cells) from being silently accepted after a row is removed or replaced.
_FROZEN_SEPARATED_CELL_KEYS = frozenset(
    itertools.product(
        (4, 32), ("small", "large"), (2, 8),
        ("linear_separated", "nonlinear_separated"),
    )
)
_FROZEN_STABLE_CELL_KEYS = frozenset(
    itertools.product(
        (4, 32), ("small", "large"), (2, 8),
        ("linear_separated", "nonlinear_separated", "genuine_overlap_half"),
    )
)
_FROZEN_DIRECTIONS = frozenset(
    f"{source}->{target}"
    for source in range(4)
    for target in range(4)
    if source != target
)
_FROZEN_SELECTOR_PANEL_AXES = frozenset(
    itertools.product(
        ("balanced", "imbalanced"), ("small", "large"), (4, 32), (2, 8),
        ("linear_separated", "nonlinear_separated", "genuine_overlap_half"),
    )
)
_FROZEN_SELECTOR_AUC_AXES = frozenset(
    itertools.product(
        (4, 32), (2, 8),
        ("linear_separated", "nonlinear_separated", "genuine_overlap_half"),
    )
)
_FROZEN_SELECTOR_STRATUM_AXES = frozenset(
    itertools.product(
        ("small", "large"), (4, 32), (2, 8),
        ("linear_separated", "nonlinear_separated", "genuine_overlap_half"),
    )
)

_CANDIDATE_ALIASES = {
    "oi_unrefined_raw": "A", "oi_refined_raw": "B", "legacy_pooled_diagonal_full_whitening": "L",
    "oas_partial_025_sample_weighted": "P25", "oas_partial_050_sample_weighted": "P50-SW",
    "oas_partial_050_class_balanced": "P50-CB", "winsorized_oas_partial_050_sample_weighted": "W50-SW",
    "winsorized_oas_partial_050_class_balanced": "W50-CB", "mad_partial_050_sample_weighted": "M50-SW",
    "mad_partial_050_class_balanced": "M50-CB",
}
_CANDIDATE_NAMES = {
    "A": "oi_unrefined_raw", "B": "oi_refined_raw", "L": "legacy_pooled_diagonal_full_whitening",
    "P25": "oas_partial_025_sample_weighted", "P50-SW": "oas_partial_050_sample_weighted",
    "P50-CB": "oas_partial_050_class_balanced", "W50-SW": "winsorized_oas_partial_050_sample_weighted",
    "W50-CB": "winsorized_oas_partial_050_class_balanced", "M50-SW": "mad_partial_050_sample_weighted",
    "M50-CB": "mad_partial_050_class_balanced",
}


# ---------------------------------------------------------------------------
# JSON and identity helpers
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Convert values to strict JSON-compatible scalars recursively."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze(value: Any) -> Any:
    """Hash-seed-independent immutable key for row identities."""

    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return _freeze(value.item())
    try:
        hash(value)
        return value
    except TypeError:
        return canonical_json(value)


def _lookup(row: Mapping[str, Any], *paths: str) -> Any:
    """Look up dotted paths, accepting case-only spelling differences."""

    for path in paths:
        current: Any = row
        ok = True
        for part in str(path).split("."):
            if not isinstance(current, Mapping):
                ok = False
                break
            if part in current:
                current = current[part]
                continue
            by_lower = {str(key).lower(): key for key in current}
            key = by_lower.get(part.lower())
            if key is None:
                ok = False
                break
            current = current[key]
        if ok:
            return current
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _contains_nonfinite(value: Any) -> bool:
    """Return whether a JSON-like row contains NaN/Inf at any depth."""

    if isinstance(value, (float, np.floating)):
        return not math.isfinite(float(value))
    if isinstance(value, np.ndarray):
        return bool(np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)))
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def candidate_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text in CANDIDATE_ORDER:
        return text
    return _CANDIDATE_ALIASES.get(text) or _CANDIDATE_ALIASES.get(text.lower())


def row_identity(row: Mapping[str, Any], index: int = 0, *, table: str = "") -> Any:
    """Get the normalized identity, never using Python's randomized hash."""

    fields = (
        "row_id", "observation_id", "pair_id", "panel_id", "cell_id", "case_id",
        "seed", "candidate_id", "method_id", "candidate", "scenario", "condition",
        "balance", "count_level", "nuisance_dim", "k", "signal_state", "family",
        "resource_cell", "head", "direction", "source_class", "target_class",
        "resource_id", "repeat", "run", "iteration", "replicate", "position",
        "source_label", "target_label",
    )
    values: List[Tuple[str, Any]] = []
    for field in fields:
        value = _lookup(row, field)
        if value is not None:
            if field in {"candidate", "candidate_id", "method_id"}:
                value = candidate_id(value) or str(value)
            values.append((field, _freeze(value)))
    if values:
        return (table, tuple(values))
    return (table, "row", int(index))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(dict(value))
    return rows


def _manifest_tables(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    raw = manifest.get("tables", manifest.get("table_manifest", {}))
    result: Dict[str, Mapping[str, Any]] = {}
    if isinstance(raw, Mapping):
        for name, spec in raw.items():
            if isinstance(spec, Mapping):
                result[str(name)] = spec
            elif isinstance(spec, str):
                result[str(name)] = {"path": spec}
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for spec in raw:
            if isinstance(spec, Mapping):
                name = spec.get("name", spec.get("file", spec.get("path")))
                if name is not None:
                    result[str(name)] = spec
    return result


def _spec_path(root: Path, name: str, spec: Mapping[str, Any]) -> Path:
    value = spec.get("path", spec.get("file", name))
    path = Path(str(value))
    if path.is_absolute():
        return path
    direct = root / path
    if direct.exists():
        return direct
    # The runner keeps normalized tables below ``tables/`` while the manifest
    # records just their canonical file names.  Accept an explicit path first,
    # then this one unambiguous convention.
    nested = root / "tables" / path
    return nested if nested.exists() else direct


def validate_table_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    table: str,
    expected_count: Optional[int] = None,
    expected_identities: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Validate duplicate-free normalized rows and return identity metadata."""

    identities = [row_identity(row, index, table=table) for index, row in enumerate(rows)]
    duplicates: List[Any] = []
    seen: set[Any] = set()
    for identity in identities:
        if identity in seen:
            duplicates.append(identity)
        seen.add(identity)
    failures: List[str] = []
    if duplicates:
        failures.append(f"duplicate identities ({len(duplicates)})")
    if expected_count is not None and len(rows) != int(expected_count):
        failures.append(f"row count {len(rows)} != {int(expected_count)}")
    if expected_identities is not None:
        expected = {_freeze(item) for item in expected_identities}
        observed = {_freeze(item) for item in identities}
        missing = expected - observed
        unexpected = observed - expected
        if missing:
            failures.append(f"missing identities ({len(missing)})")
        if unexpected:
            failures.append(f"unexpected identities ({len(unexpected)})")
    return {
        "table": str(table), "row_count": len(rows), "unique_count": len(seen),
        "duplicates": [json_safe(item) for item in duplicates],
        "status": "pass" if not failures else "fail", "failures": failures,
    }


def validate_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    expected_stage: Optional[str] = None,
    protocol_hash: Optional[str] = None,
    code_identity_hash: Optional[str] = None,
    strict_grid: bool = True,
) -> Dict[str, Any]:
    """Validate immutable provenance fields without trusting the manifest itself."""

    failures: List[str] = []
    stage = str(manifest.get("stage", manifest.get("study", ""))).lower()
    if expected_stage is not None and stage != str(expected_stage).lower():
        failures.append(f"stage {stage!r} != {str(expected_stage).lower()!r}")
    if protocol_hash is not None:
        observed = manifest.get("protocol_sha256", manifest.get("protocol_hash"))
        if observed != protocol_hash:
            failures.append(f"protocol hash {observed!r} != {protocol_hash!r}")
    if code_identity_hash is not None:
        observed = manifest.get("code_identity_sha256", manifest.get("code_identity_hash"))
        if observed is None and isinstance(manifest.get("provenance"), Mapping):
            observed = manifest["provenance"].get("code_identity_sha256", manifest["provenance"].get("code_identity_hash"))
        if observed != code_identity_hash:
            failures.append(f"code identity {observed!r} != {code_identity_hash!r}")
    status = str(manifest.get("artifact_status", "")).lower()
    # A self-consistent partial table is not an analyzable outcome.  Require
    # the runner's explicit completion marker rather than treating an omitted
    # marker as success.
    if status != "completed":
        failures.append(f"artifact status {status!r} is not completed")
    if strict_grid and stage in {"smoke", "development", "confirmation"}:
        expected = {
            "smoke": (30, 300),
            "development": (5184, 51840),
            "confirmation": (10368, None),
        }[stage]
        case_count = manifest.get("case_count", manifest.get("expected_case_count"))
        method_count = manifest.get("method_row_count", manifest.get("expected_method_row_count"))
        if case_count is None:
            failures.append("manifest is missing frozen case_count")
        else:
            try:
                case_count_value = int(case_count)
            except (TypeError, ValueError):
                case_count_value = None
            if case_count_value != expected[0]:
                failures.append(f"case count {case_count!r} != frozen {expected[0]}")
        if expected[1] is not None:
            if method_count is None:
                failures.append("manifest is missing frozen method_row_count")
            else:
                try:
                    method_count_value = int(method_count)
                except (TypeError, ValueError):
                    method_count_value = None
                if method_count_value != expected[1]:
                    failures.append(f"method row count {method_count!r} != frozen {expected[1]}")
        observed_case_count = manifest.get("observed_case_count")
        if observed_case_count is None:
            failures.append("manifest is missing observed_case_count")
        else:
            try:
                observed_case_count_value = int(observed_case_count)
            except (TypeError, ValueError):
                observed_case_count_value = None
            if observed_case_count_value != expected[0]:
                failures.append(f"observed case count {observed_case_count!r} != frozen {expected[0]}")
        observed_method_count = manifest.get("observed_method_row_count")
        expected_observed_method_count = expected[1]
        if expected_observed_method_count is not None:
            if observed_method_count is None:
                failures.append("manifest is missing observed_method_row_count")
            else:
                try:
                    observed_method_count_value = int(observed_method_count)
                except (TypeError, ValueError):
                    observed_method_count_value = None
                if observed_method_count_value != expected_observed_method_count:
                    failures.append(
                        f"observed method row count {observed_method_count!r} != frozen {expected_observed_method_count}"
                    )
        methods = manifest.get("method_ids")
        if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)):
            failures.append("manifest is missing frozen method_ids")
        if stage == "confirmation":
            locked = manifest.get("locked_candidate") or manifest.get("promotion_candidate")
            expected_methods = ["A", "B", "L", "P50-SW", "W50-SW", "W50-CB"]
            if locked is None:
                failures.append("confirmation manifest is missing locked_candidate")
            elif locked not in expected_methods:
                expected_methods.append(str(locked))
            if isinstance(methods, Sequence) and not isinstance(methods, (str, bytes)) and list(methods) != expected_methods:
                failures.append(f"confirmation method set {list(methods)!r} != frozen {expected_methods!r}")
            expected_method_count = expected[0] * len(expected_methods)
            if method_count is None:
                failures.append("manifest is missing frozen method_row_count")
            else:
                try:
                    confirmation_method_count = int(method_count)
                except (TypeError, ValueError):
                    confirmation_method_count = None
                if confirmation_method_count != expected_method_count:
                    failures.append(
                        f"confirmation method row count {method_count!r} != frozen {expected_method_count}"
                    )
            if observed_method_count is None:
                failures.append("manifest is missing observed_method_row_count")
            else:
                try:
                    observed_confirmation_method_count = int(observed_method_count)
                except (TypeError, ValueError):
                    observed_confirmation_method_count = None
                if observed_confirmation_method_count != expected_method_count:
                    failures.append(
                        f"observed confirmation method row count {observed_method_count!r} != frozen {expected_method_count}"
                    )
        elif isinstance(methods, Sequence) and not isinstance(methods, (str, bytes)):
            methods = list(methods)
            if methods != list(CANDIDATE_ORDER):
                failures.append(f"{stage} method set/order is not the frozen ten candidates")
    return {"status": "pass" if not failures else "fail", "stage": stage, "failures": failures}


def load_normalized_artifact(
    input_path: os.PathLike[str] | str,
    *,
    expected_stage: Optional[str] = None,
    protocol_hash: Optional[str] = None,
    code_identity_hash: Optional[str] = None,
    require_tables: bool = True,
) -> Dict[str, Any]:
    """Load manifest-declared JSONL tables and verify bytes/counts/identities."""

    source = Path(input_path)
    root = source if source.is_dir() else source.parent
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing normalized artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("normalized artifact manifest must be a JSON object")
    identity = validate_manifest_identity(
        manifest, expected_stage=expected_stage, protocol_hash=protocol_hash, code_identity_hash=code_identity_hash,
    )
    table_specs = _manifest_tables(manifest)
    failures = list(identity["failures"])
    tables: Dict[str, List[Dict[str, Any]]] = {}
    table_checks: Dict[str, Any] = {}
    names = tuple(table_specs) if table_specs else TABLE_NAMES
    if require_tables and not table_specs:
        failures.append("manifest has no table declarations")
    if require_tables:
        missing_names = sorted(set(TABLE_NAMES) - set(table_specs))
        extra_names = sorted(set(table_specs) - set(TABLE_NAMES))
        if missing_names:
            failures.append(f"manifest missing required tables: {missing_names}")
        if extra_names:
            failures.append(f"manifest has unexpected tables: {extra_names}")
    for name in names:
        spec = table_specs.get(name, {})
        path = _spec_path(root, str(name), spec)
        if not path.exists():
            table_checks[str(name)] = {"table": str(name), "status": "fail", "failures": ["missing file"]}
            failures.append(f"missing table {name}")
            continue
        rows = _read_jsonl(path) if path.suffix.lower() in {".jsonl", ".ndjson"} else json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError(f"table {path} must contain a JSON array/object rows")
        normalized = [dict(row) for row in rows]
        check = validate_table_rows(normalized, table=str(name), expected_count=spec.get("row_count"))
        expected_hash = spec.get("sha256", spec.get("hash"))
        actual_hash = sha256_file(path)
        if expected_hash is not None and str(expected_hash) != actual_hash:
            check["failures"].append(f"sha256 {actual_hash} != {expected_hash}")
            check["status"] = "fail"
        check["sha256"] = actual_hash
        check["path"] = str(path)
        table_checks[str(name)] = check
        tables[str(name)] = normalized
        failures.extend(f"{name}: {item}" for item in check["failures"])
    failures.extend(_loaded_grid_failures(manifest, tables, stage=identity.get("stage", "")))
    return {
        "manifest": dict(manifest), "tables": tables, "table_checks": table_checks,
        "identity": {**identity, "status": "pass" if not failures else "fail", "failures": failures},
        "status": "pass" if not failures else "fail", "root": str(root), "manifest_path": str(manifest_path),
    }


def _loaded_grid_failures(manifest: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]], *, stage: str) -> List[str]:
    """Check the frozen case/method grid after table bytes are verified."""

    stage = str(stage).lower()
    if stage not in {"smoke", "development", "confirmation"}:
        return []
    expected_cases = {"smoke": 30, "development": 5184, "confirmation": 10368}[stage]
    methods_value = manifest.get("method_ids")
    methods = [str(value) for value in methods_value] if isinstance(methods_value, Sequence) and not isinstance(methods_value, (str, bytes)) else []
    expected_methods = list(CANDIDATE_ORDER)
    if stage == "confirmation":
        expected_methods = ["A", "B", "L", "P50-SW", "W50-SW", "W50-CB"]
        locked = manifest.get("locked_candidate") or manifest.get("promotion_candidate")
        if locked and locked not in expected_methods:
            expected_methods.append(str(locked))
    failures: List[str] = []
    if stage == "confirmation" and not (manifest.get("locked_candidate") or manifest.get("promotion_candidate")):
        failures.append("confirmation manifest is missing locked_candidate")
    if methods != expected_methods:
        failures.append(f"method_ids {methods!r} != frozen {expected_methods!r}")
    case_rows = list(tables.get("case_rows.jsonl", ()))
    expected_method_rows = expected_cases * len(expected_methods)
    if len(case_rows) != expected_method_rows:
        failures.append(f"case_rows count {len(case_rows)} != frozen {expected_method_rows}")
    case_ids = {str(_lookup(row, "case_id")) for row in case_rows if _lookup(row, "case_id") is not None}
    if len(case_ids) != expected_cases:
        failures.append(f"distinct case ids {len(case_ids)} != frozen {expected_cases}")
    from . import fixtures
    expected_case_objects = (
        fixtures.smoke_cases()
        if stage == "smoke"
        else fixtures.development_cases()
        if stage == "development"
        else fixtures.confirmation_cases()
    )
    expected_ids = {str(case.case_id) for case in expected_case_objects}
    if case_ids != expected_ids:
        failures.append("case ids do not match the frozen fixture grid")
    methods_observed = {candidate_id(_lookup(row, "candidate_id", "method_id", "candidate")) for row in case_rows}
    if methods_observed != set(expected_methods):
        failures.append(f"observed method ids {sorted(methods_observed, key=str)!r} != frozen")
    expected_observed = manifest.get("observed_method_row_count")
    try:
        observed_method_count = int(expected_observed)
    except (TypeError, ValueError):
        observed_method_count = None
    if observed_method_count != expected_method_rows:
        failures.append(f"observed_method_row_count {expected_observed!r} != frozen {expected_method_rows}")
    expected_observed_cases = manifest.get("observed_case_count")
    try:
        observed_case_count = int(expected_observed_cases)
    except (TypeError, ValueError):
        observed_case_count = None
    if observed_case_count != expected_cases:
        failures.append(f"observed_case_count {expected_observed_cases!r} != frozen {expected_cases}")
    if stage == "development":
        expected_resource_methods = ("B", "L", *PROMOTABLE_CANDIDATES)
    elif stage == "confirmation":
        locked = manifest.get("locked_candidate") or manifest.get("promotion_candidate")
        expected_resource_methods = ("B", str(locked)) if locked else ()
    else:
        expected_resource_methods = ()
    # Child normalized surfaces have fixed multiplicities.  Empty resources
    # are incomplete evidence even when the optional benchmark was disabled.
    expected_table_counts = {
        "case_rows.jsonl": expected_method_rows,
        "geometry_rows.jsonl": expected_method_rows,
        "prototype_rows.jsonl": expected_method_rows,
        "selector_panels.jsonl": 0 if stage == "smoke" else expected_method_rows,
        "pair_rows.jsonl": expected_method_rows * 12,
        "resource_rows.jsonl": 0 if stage == "smoke" else (12 if stage == "development" else 24) * len(RESOURCE_CELLS) * (len(expected_resource_methods) if stage != "smoke" else 0),
    }
    for name, expected_count in expected_table_counts.items():
        if name not in tables:
            failures.append(f"required table {name} is missing")
        elif len(tables[name]) != expected_count:
            failures.append(f"{name} count {len(tables[name])} != frozen {expected_count}")

    # Counts alone permit a self-consistent permutation or repeated case.  All
    # normalized child tables therefore carry the exact case/method identity
    # set.  Pair rows additionally cover every ordered off-diagonal pair of
    # the four frozen scalar classes.  The comparison is deliberately based on
    # JSON-safe frozen values, not Python set/hash iteration order.
    expected_case_method = {
        (_freeze(case_id), candidate)
        for case_id in expected_ids
        for candidate in expected_methods
    }

    def case_method_identities(name: str) -> set[Tuple[Any, str]]:
        observed: set[Tuple[Any, str]] = set()
        for row in tables.get(name, ()):
            case_id = _lookup(row, "case_id")
            candidate = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate"))
            if case_id is not None and candidate is not None:
                observed.add((_freeze(str(case_id)), candidate))
        return observed

    for name in ("case_rows.jsonl", "geometry_rows.jsonl", "prototype_rows.jsonl"):
        observed = case_method_identities(name)
        if observed != expected_case_method:
            failures.append(f"{name} identities do not match the frozen case/method grid")
    if stage != "smoke":
        observed = case_method_identities("selector_panels.jsonl")
        if observed != expected_case_method:
            failures.append("selector_panels identities do not match the frozen case/method grid")
    pair_expected = {
        (_freeze(str(case_id)), candidate, _freeze(source), _freeze(target))
        for case_id in expected_ids
        for candidate in expected_methods
        for source in range(4)
        for target in range(4)
        if source != target
    }
    pair_observed: set[Tuple[Any, str, Any, Any]] = set()
    for row in tables.get("pair_rows.jsonl", ()):
        case_id = _lookup(row, "case_id")
        candidate = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate"))
        source = _lookup(row, "source_label")
        target = _lookup(row, "target_label")
        if case_id is not None and candidate is not None and source is not None and target is not None:
            pair_observed.add((_freeze(str(case_id)), candidate, _freeze(source), _freeze(target)))
    if pair_observed != pair_expected:
        failures.append("pair_rows identities do not match the frozen 4-class directed-pair grid")
    declared_resource_count = manifest.get("resource_row_count")
    try:
        resource_count = int(declared_resource_count)
    except (TypeError, ValueError):
        resource_count = None
    expected_resource_count = expected_table_counts["resource_rows.jsonl"]
    if resource_count != expected_resource_count:
        failures.append(f"resource_row_count {declared_resource_count!r} != frozen {expected_resource_count}")

    # Resource identities and measurements form a mandatory, separately
    # paired surface for development/confirmation.  Check it here rather than
    # trusting the manifest's self-declared row count.
    resource_specs = {
        "R0": ("S0", "balanced", "small", 4, 2, "linear_separated"),
        "R1": ("H2", "balanced", "large", 32, 8, "linear_separated"),
        "R2": ("C", "imbalanced", "large", 32, 8, "genuine_overlap_half"),
        "R3": ("ALL", "imbalanced", "large", 32, 8, "nonlinear_separated"),
    }
    expected_resource_cases: Dict[str, set[str]] = {resource_id: set() for resource_id in resource_specs}
    for case in expected_case_objects:
        for resource_id, spec in resource_specs.items():
            if all(
                getattr(case, field) == value
                for field, value in zip(
                    ("scenario", "balance", "count_level", "nuisance_dim", "k", "signal_state"), spec
                )
            ):
                expected_resource_cases[resource_id].add(str(case.case_id))
    expected_resource_identities = {
        (resource_id, case_id, candidate)
        for resource_id, case_ids_for_cell in expected_resource_cases.items()
        for case_id in case_ids_for_cell
        for candidate in expected_resource_methods
    } if stage != "smoke" else set()
    observed_resource_identities: set[Tuple[str, str, str]] = set()
    required_resource_fields = (
        "total_wall_seconds", "total_cpu_seconds", "score_fixed_wall_seconds",
        "score_fixed_cpu_seconds", "peak_rss_bytes",
    )
    for row in tables.get("resource_rows.jsonl", ()):
        resource_id = str(_lookup(row, "resource_id"))
        case_id = str(_lookup(row, "case_id"))
        candidate = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate"))
        identity = (resource_id, case_id, str(candidate))
        observed_resource_identities.add(identity)
        if resource_id not in resource_specs:
            failures.append(f"unknown resource_id {resource_id!r}")
        if candidate not in expected_resource_methods:
            failures.append(f"unexpected resource candidate {candidate!r}")
        if case_id not in expected_resource_cases.get(resource_id, set()):
            failures.append(f"resource case is not a frozen resource cell: {identity!r}")
        status = str(_lookup(row, "status") or "").lower()
        if status not in {"ok", "pass", "completed"}:
            failures.append(f"resource row status {status!r}")
        for field in required_resource_fields:
            value = _number(_lookup(row, field))
            if value is None or value < 0.0:
                failures.append(f"resource row has missing/nonfinite {field}")
    if stage == "smoke":
        if tables.get("resource_rows.jsonl"):
            failures.append("smoke resource_rows must be empty")
    elif observed_resource_identities != expected_resource_identities:
        failures.append("resource row identities do not match the frozen resource grid")
    for name, rows in tables.items():
        for row in rows:
            status = str(_lookup(row, "status") or "").lower()
            if status not in {"ok", "pass", "completed"}:
                failures.append(f"{name} contains status={status!r}")
                break
            if _contains_nonfinite(row):
                failures.append(f"{name} contains a non-finite value")
                break

    # Required identities/measurements are validated independently of the
    # manifest's self-declared schema.  ``None`` remains a valid JSON null for
    # optional diagnostics, but a field that is required for an outcome table
    # must be present and finite.
    for row in tables.get("case_rows.jsonl", ()):
        if _lookup(row, "case_id") is None or candidate_id(_lookup(row, "candidate_id", "method_id", "candidate")) is None:
            failures.append("case_rows contains a row without case_id/candidate_id")
            break
        if _lookup(row, "error") not in (None, "", False):
            failures.append("case_rows contains an execution error")
            break
    for row in tables.get("geometry_rows.jsonl", ()):
        if _lookup(row, "case_id") is None or candidate_id(_lookup(row, "candidate_id", "method_id", "candidate")) is None:
            failures.append("geometry_rows contains a row without case_id/candidate_id")
            break
        if not isinstance(_lookup(row, "geometry"), Mapping):
            failures.append("geometry_rows contains a row without canonical geometry")
            break
    for row in tables.get("prototype_rows.jsonl", ()):
        if _lookup(row, "case_id") is None or candidate_id(_lookup(row, "candidate_id", "method_id", "candidate")) is None:
            failures.append("prototype_rows contains a row without case_id/candidate_id")
            break
        if not isinstance(_lookup(row, "prototype"), Mapping):
            failures.append("prototype_rows contains a row without canonical prototype diagnostics")
            break
    for row in tables.get("pair_rows.jsonl", ()):
        if stage == "smoke":
            required = ("source_label", "target_label", "exact_state_match", "pair_structure", "outcomes_redacted")
            if any(field not in row for field in required):
                failures.append("smoke pair_rows lacks structural direction fields")
                break
            if _lookup(row, "source_label") == _lookup(row, "target_label"):
                failures.append("smoke pair_rows contains a diagonal direction")
                break
            if _lookup(row, "exact_state_match") is not True:
                failures.append("smoke pair_rows contains a non-exact directional state")
                break
            if "finite" in row:
                failures.append("smoke pair_rows exposes a top-level generic finite flag")
                break
            if row.get("outcomes_redacted") is not True:
                failures.append("smoke pair_rows lacks outcomes_redacted flag")
                break
            structure = row.get("pair_structure")
            if not isinstance(structure, Mapping) or set(structure) != set(_SMOKE_PAIR_STRUCTURAL_FIELDS):
                failures.append("smoke pair_rows has a non-canonical pair_structure")
                break
            if any(not _smoke_structural_descriptor(value) for value in structure.values()):
                failures.append("smoke pair_rows has a non-finite pair descriptor")
                break
        else:
            required = (
                "source_label", "target_label", "support", "hits", "sparse_adj_hits",
                "pairwise_index", "evidence", "truth_overlap", "truth_overlap_label",
                "exact_state_match",
            )
            if any(field not in row for field in required):
                failures.append("pair_rows contains a row without canonical directional state fields")
                break
            if _lookup(row, "source_label") == _lookup(row, "target_label"):
                failures.append("pair_rows contains a diagonal direction")
                break
            if _lookup(row, "exact_state_match") is not True:
                failures.append("pair_rows contains a non-exact directional state")
                break
            for field in (
                "support", "hits", "sparse_adj_hits", "pairwise_index", "evidence",
                "truth_overlap", "truth_overlap_label",
            ):
                if _number(_lookup(row, field)) is None:
                    failures.append(f"pair_rows contains non-finite {field}")
                    break
    if stage != "smoke":
        for row in tables.get("selector_panels.jsonl", ()):
            candidate = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate"))
            if candidate is None:
                failures.append("selector_panels contains a row without candidate_id")
                break
            if _number(_lookup(row, "candidate_score")) is None:
                failures.append("selector_panels is missing finite candidate_score")
                break
            if not isinstance(_lookup(row, "reference_accuracies"), Mapping) or any(
                _number(_lookup(_lookup(row, "reference_accuracies"), head)) is None for head in REFERENCE_HEADS
            ):
                failures.append("selector_panels is missing finite reference_accuracies")
                break
            if _number(_lookup(row, "linear_probe_score")) is None:
                failures.append("selector_panels is missing finite linear_probe_score")
                break
    return failures


def validate_normalized_artifact(
    artifact: Mapping[str, Any], *, expected_stage: Optional[str] = None,
    protocol_hash: Optional[str] = None, code_identity_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate an already-loaded artifact; useful for fake rows in tests."""

    manifest = artifact.get("manifest", {})
    result = validate_manifest_identity(
        manifest if isinstance(manifest, Mapping) else {}, expected_stage=expected_stage,
        protocol_hash=protocol_hash, code_identity_hash=code_identity_hash,
    )
    failures = list(result["failures"])
    tables = artifact.get("tables", {})
    if not isinstance(tables, Mapping):
        failures.append("tables is not a mapping")
    else:
        for name, rows in tables.items():
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                failures.append(f"{name}: rows is not a sequence")
                continue
            check = validate_table_rows(rows, table=str(name))
            failures.extend(f"{name}: {item}" for item in check["failures"])
        failures.extend(_loaded_grid_failures(manifest, tables, stage=result.get("stage", "")))
    return {**result, "status": "pass" if not failures else "fail", "failures": failures}


def flatten_rows(artifact_or_rows: Any, table: Optional[str] = None) -> List[Dict[str, Any]]:
    if isinstance(artifact_or_rows, Mapping):
        tables = artifact_or_rows.get("tables")
        if isinstance(tables, Mapping):
            if table is not None and table in tables:
                value = tables[table]
            else:
                value = [row for rows in tables.values() if isinstance(rows, Sequence) for row in rows]
        elif "rows" in artifact_or_rows:
            value = artifact_or_rows["rows"]
        else:
            value = [artifact_or_rows]
    else:
        value = artifact_or_rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _group_candidate(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate", "method", "arm"))
        if value is not None:
            grouped[value].append(dict(row))
    return dict(grouped)


def _seed(row: Mapping[str, Any], default: Any = None) -> Any:
    value = _lookup(row, "seed", "train_seed", "evaluation_seed")
    return default if value is None else _freeze(value)


def _expected_frozen_seeds(count: Optional[int]) -> Optional[set[Any]]:
    if count is None:
        return None
    from . import fixtures
    if int(count) == 12:
        return {int(value) for value in fixtures.DEVELOPMENT_SEEDS}
    if int(count) == 24:
        return {int(value) for value in fixtures.CONFIRMATION_SEEDS}
    return None


def _scenario(row: Mapping[str, Any]) -> Optional[str]:
    value = _lookup(row, "scenario", "condition", "condition_name", "nuisance_kind", "family")
    if value is None:
        return None
    text = str(value)
    # Some normalized case rows carry a compound condition.
    for item in SCENARIO_ORDER:
        if text == item or text.startswith(item + "__") or text.endswith("_" + item):
            return item
    return text


def _finite_array(values: Iterable[Any]) -> np.ndarray:
    result: List[float] = []
    for value in values:
        number = _number(value)
        if number is not None:
            result.append(number)
    return np.asarray(result, dtype=float)


# ---------------------------------------------------------------------------
# Shared complete-seed block bootstrap
# ---------------------------------------------------------------------------


_DRAW_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def joint_bootstrap_draws(
    n_blocks: int, *, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Return one shared seed-resampling matrix for every paired outcome."""

    n_blocks = int(n_blocks)
    n_resamples = int(n_resamples)
    if n_blocks <= 0:
        return np.empty((0, 0), dtype=np.int64)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    key = (n_blocks, n_resamples, int(seed))
    if key not in _DRAW_CACHE:
        draws = np.random.default_rng(int(seed)).integers(0, n_blocks, size=(n_resamples, n_blocks), dtype=np.int64)
        draws.setflags(write=False)
        _DRAW_CACHE[key] = draws
    return _DRAW_CACHE[key]


bootstrap_indices = joint_bootstrap_draws


def _percentile_bounds(samples: np.ndarray, level: float) -> Tuple[Optional[float], Optional[float]]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        return None, None
    tail = (1.0 - float(level)) * 50.0
    return float(np.percentile(samples, tail)), float(np.percentile(samples, 100.0 - tail))


def interval(
    estimate: Optional[float], samples: Sequence[float], *, level: float = PRIMARY_INTERVAL_LEVEL,
    n: Optional[int] = None, n_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    lower, upper = _percentile_bounds(np.asarray(samples, dtype=float), level)
    return {
        "estimate": None if estimate is None or not math.isfinite(float(estimate)) else float(estimate),
        "lower": lower, "upper": upper, "level": float(level),
        "n": int(n if n is not None else len(samples)), "n_blocks": None if n_blocks is None else int(n_blocks),
        "status": "defined" if lower is not None else "undefined",
    }


def percentile_interval(values: Sequence[float], *, level: float = PRIMARY_INTERVAL_LEVEL) -> Dict[str, Any]:
    array = _finite_array(values)
    if not array.size:
        return interval(None, [], level=level, n=0)
    draws = joint_bootstrap_draws(array.size)
    samples = np.mean(array[draws], axis=1)
    return interval(float(np.mean(array)), samples, level=level, n=int(array.size), n_blocks=int(array.size))


def _normalize_blocks(values: Mapping[Any, Any]) -> Tuple[List[Any], Dict[Any, np.ndarray]]:
    maps: Dict[Any, np.ndarray] = {}
    for key, value in values.items():
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size and np.all(np.isfinite(array)):
            maps[_freeze(key)] = array
    return sorted(maps, key=lambda item: canonical_json(item)), maps


def complete_seed_blocks(
    values_by_candidate: Mapping[str, Mapping[Any, Any]], *, require_same_shape: bool = True,
) -> Tuple[List[Any], Dict[str, Dict[Any, np.ndarray]]]:
    """Keep only complete seed identities, including multiplicity/shape checks."""

    if not values_by_candidate:
        return [], {}
    normalized: Dict[str, Dict[Any, np.ndarray]] = {}
    for name, values in values_by_candidate.items():
        _, maps = _normalize_blocks(values)
        normalized[str(name)] = maps
    common = set.intersection(*(set(values) for values in normalized.values()))
    blocks: List[Any] = []
    for block in sorted(common, key=lambda item: canonical_json(item)):
        arrays = [normalized[name][block] for name in normalized]
        if require_same_shape and len({array.shape for array in arrays}) != 1:
            continue
        blocks.append(block)
    return blocks, normalized


def complete_seed_bootstrap(
    values_by_candidate: Mapping[str, Mapping[Any, Any]], *, statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
    level: float = PRIMARY_INTERVAL_LEVEL,
) -> Dict[str, Dict[str, Any]]:
    """Bootstrap complete seed blocks with one draw matrix shared by all arms."""

    blocks, normalized = complete_seed_blocks(values_by_candidate)
    if not blocks:
        return {str(name): interval(None, [], level=level, n=0, n_blocks=0) for name in values_by_candidate}
    draws = joint_bootstrap_draws(len(blocks), n_resamples=n_resamples, seed=seed)
    result: Dict[str, Dict[str, Any]] = {}
    for name in values_by_candidate:
        block_arrays = normalized[str(name)]
        pooled = np.concatenate([block_arrays[block] for block in blocks])
        estimate = float(statistic(pooled)) if pooled.size else None
        samples: List[float] = []
        for draw in draws:
            picked = np.concatenate([block_arrays[blocks[int(index)]] for index in draw])
            value = _number(statistic(picked)) if picked.size else None
            if value is not None:
                samples.append(value)
        result[str(name)] = interval(estimate, samples, level=level, n=int(pooled.size), n_blocks=len(blocks))
        result[str(name)]["bootstrap_seed"] = int(seed)
        result[str(name)]["bootstrap_resamples"] = int(n_resamples)
    return result


def paired_seed_difference(
    values_by_candidate: Mapping[str, Mapping[Any, Any]], *, baseline: str = "B",
    statistic: Callable[[np.ndarray], float] = np.mean, n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED, level: float = PRIMARY_INTERVAL_LEVEL,
) -> Dict[str, Dict[str, Any]]:
    """Candidate-minus-baseline intervals from the same complete seed draws."""

    if baseline not in values_by_candidate:
        return {str(name): interval(None, [], level=level, n=0, n_blocks=0) for name in values_by_candidate if name != baseline}
    selected = {str(name): values for name, values in values_by_candidate.items()}
    blocks, normalized = complete_seed_blocks(selected)
    names = [str(name) for name in selected if str(name) != baseline]
    if not blocks:
        return {name: interval(None, [], level=level, n=0, n_blocks=0) for name in names}
    draws = joint_bootstrap_draws(len(blocks), n_resamples=n_resamples, seed=seed)
    result: Dict[str, Dict[str, Any]] = {}
    base_map = normalized[baseline]
    base_pooled = np.concatenate([base_map[block] for block in blocks])
    base_point = float(statistic(base_pooled))
    for name in names:
        candidate_map = normalized[name]
        candidate_pooled = np.concatenate([candidate_map[block] for block in blocks])
        point = float(statistic(candidate_pooled)) - base_point
        samples: List[float] = []
        for draw in draws:
            base_values = np.concatenate([base_map[blocks[int(index)]] for index in draw])
            candidate_values = np.concatenate([candidate_map[blocks[int(index)]] for index in draw])
            value = _number(statistic(candidate_values) - statistic(base_values))
            if value is not None:
                samples.append(value)
        result[name] = interval(point, samples, level=level, n=min(int(base_pooled.size), int(candidate_pooled.size)), n_blocks=len(blocks))
        result[name]["baseline"] = baseline
        result[name]["bootstrap_seed"] = int(seed)
        result[name]["bootstrap_resamples"] = int(n_resamples)
    return result


paired_bootstrap_difference = paired_seed_difference


def within_draw_max(
    contrasts_by_name: Mapping[str, Mapping[Any, float]], *, n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED, level: float = BONFERRONI_LEVEL,
) -> Dict[str, Any]:
    """Return a max-of-contrasts interval using the same draw per contrast."""

    if not contrasts_by_name:
        return interval(None, [], level=level, n=0, n_blocks=0)
    normalized = {str(name): {_freeze(k): float(v) for k, v in values.items() if _number(v) is not None} for name, values in contrasts_by_name.items()}
    common = set.intersection(*(set(values) for values in normalized.values()))
    blocks = sorted(common, key=lambda item: canonical_json(item))
    if not blocks:
        return interval(None, [], level=level, n=0, n_blocks=0)
    arrays = {name: np.asarray([normalized[name][block] for block in blocks], dtype=float) for name in normalized}
    draws = joint_bootstrap_draws(len(blocks), n_resamples=n_resamples, seed=seed)
    sampled = np.column_stack([np.mean(values[draws], axis=1) for values in arrays.values()])
    maxima = np.max(sampled, axis=1)
    estimate = float(max(np.mean(values) for values in arrays.values()))
    result = interval(estimate, maxima, level=level, n=len(blocks), n_blocks=len(blocks))
    result.update({"contrasts": sorted(arrays), "within_draw_max": True, "bootstrap_seed": int(seed), "bootstrap_resamples": int(n_resamples)})
    return result


# ---------------------------------------------------------------------------
# Detection and rank primitives
# ---------------------------------------------------------------------------


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def tie_aware_auroc(labels: Sequence[Any], scores: Sequence[Any]) -> Optional[float]:
    labels_array = np.asarray(labels, dtype=float).reshape(-1)
    scores_array = np.asarray(scores, dtype=float).reshape(-1)
    finite = np.isfinite(labels_array) & np.isfinite(scores_array)
    labels_array, scores_array = labels_array[finite], scores_array[finite]
    truth = labels_array > 0.5
    positives = int(np.sum(truth)); negatives = int(truth.size - positives)
    if not positives or not negatives:
        return None
    ranks = _rankdata(scores_array)
    u = float(np.sum(ranks[truth]) - positives * (positives + 1) / 2.0)
    return u / float(positives * negatives)


def archived_pooled_auprc(labels: Sequence[Any], scores: Sequence[Any]) -> Optional[float]:
    """Stable descending AP used by the archived directed-pair analysis."""

    y = np.asarray(labels, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(s)
    y, s = y[finite], s[finite]
    truth = y > 0.5
    positives = int(np.sum(truth))
    if not positives or positives == y.size:
        return None
    order = np.argsort(-s, kind="mergesort")
    ranked = truth[order].astype(float)
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, ranked.size + 1, dtype=float)
    return float(np.sum(precision * ranked) / positives)


stable_auprc = archived_pooled_auprc


def binary_detection_metrics(
    labels: Sequence[Any], evidence: Sequence[Any], *, threshold: float = PAIR_EVIDENCE_THRESHOLD,
    continuous_truth: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=float).reshape(-1)
    score = np.asarray(evidence, dtype=float).reshape(-1)
    truth_for_calibration = y if continuous_truth is None else np.asarray(continuous_truth, dtype=float).reshape(-1)
    size = min(y.size, score.size, truth_for_calibration.size)
    y, score, truth_for_calibration = y[:size], score[:size], truth_for_calibration[:size]
    finite = np.isfinite(y) & np.isfinite(score) & np.isfinite(truth_for_calibration)
    y, score, truth_for_calibration = y[finite], score[finite], truth_for_calibration[finite]
    if not y.size:
        return {"status": "undefined", "n": 0, **{key: None for key in ("fpr", "fnr", "auroc", "auprc", "brier", "ece")}}
    truth = y > 0.5
    predicted = score >= float(threshold)
    negatives, positives = ~truth, truth
    fpr = float(np.mean(predicted[negatives])) if np.any(negatives) else None
    fnr = float(np.mean(~predicted[positives])) if np.any(positives) else None
    clipped = np.clip(score, 0.0, 1.0)
    continuous_truth = np.clip(truth_for_calibration, 0.0, 1.0)
    brier = float(np.mean((clipped - continuous_truth) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        mask = (clipped >= edges[index]) & (clipped <= edges[index + 1] if index == 9 else clipped < edges[index + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(clipped[mask])) - float(np.mean(continuous_truth[mask])))
    return {
        "status": "defined", "n": int(y.size), "fpr": fpr, "fnr": fnr,
        "auroc": tie_aware_auroc(truth, score), "auprc": archived_pooled_auprc(truth, score),
        "brier": brier, "ece": float(ece),
    }


def spearman_rank(values_a: Sequence[Any], values_b: Sequence[Any]) -> Optional[float]:
    a, b = np.asarray(values_a, dtype=float).reshape(-1), np.asarray(values_b, dtype=float).reshape(-1)
    size = min(a.size, b.size)
    a, b = a[:size], b[:size]
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    ra -= np.mean(ra); rb -= np.mean(rb)
    denominator = math.sqrt(float(np.sum(ra * ra) * np.sum(rb * rb)))
    return None if denominator == 0.0 else float(np.sum(ra * rb) / denominator)


spearman_rank_correlation = spearman_rank


def normalized_trapezoid_auc(x: Sequence[Any], y: Sequence[Any]) -> Optional[float]:
    xx, yy = np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[finite], yy[finite]
    if xx.size < 2 or float(np.max(xx) - np.min(xx)) <= 0.0:
        return None
    order = np.argsort(xx, kind="mergesort")
    xx, yy = xx[order], yy[order]
    return float(np.trapz(yy, xx) / (xx[-1] - xx[0]))


def _metric_value(row: Mapping[str, Any], metric: str, default: Optional[float] = None) -> Optional[float]:
    aliases = {
        "score": ("score_fixed", "candidate_score", "evaluation_score", "score", "oi_score", "overlap_index", "index"),
        "neighbor_impurity": ("geometry.cross_class_neighbor_impurity",),
        "oracle_neighbor_jaccard": ("geometry.oracle_neighbor_jaccard",),
        "pair_distance_spearman": ("geometry.pair_distance_spearman",),
        "prototype_activity": ("refinement_activity", "refinement.activity", "prototype_activity"),
        "condition_after": ("condition_after", "conditioning.condition_after"),
        "condition_before": ("condition_before", "conditioning.condition_before"),
        "fpr": ("fpr", "detection.fpr", "genuine_overlap.fpr"),
        "fnr": ("fnr", "detection.fnr", "genuine_overlap.fnr"),
        "auroc": ("auroc", "detection.auroc", "genuine_overlap.auroc"),
        "auprc": ("auprc", "detection.auprc", "genuine_overlap.auprc"),
        "brier": ("brier", "detection.brier", "genuine_overlap.brier"),
        "ece": ("ece", "detection.ece", "genuine_overlap.ece"),
        "clean_mae": ("clean_mae", "clean_error", "evidence_error"),
        "nuisance_drift": ("nuisance_drift", "absolute_nuisance_drift", "drift_error"),
        "timing_fit": ("fit_wall_seconds", "timing.fit", "timing.oi_fit_refinement", "oi_fit_refinement_wall_seconds"),
        "timing_score_fixed": ("score_fixed_wall_seconds", "timing.score_fixed", "score_fixed_seconds"),
        "timing_total": ("total_wall_seconds", "timing.total", "total_seconds", "wall_seconds"),
        "peak_memory": ("peak_memory_mb", "peak_memory", "memory_mb"),
    }
    value = _lookup(row, *(aliases.get(metric, (metric,))))
    if isinstance(value, Mapping):
        value = _lookup(value, "estimate", "value", "median", "mean")
    number = _number(value)
    return default if number is None else number


# ---------------------------------------------------------------------------
# Pair row extraction and pooled overlap outcomes
# ---------------------------------------------------------------------------


def _direction(row: Mapping[str, Any]) -> Optional[str]:
    source = _lookup(row, "direction", "pair_direction")
    if source is not None:
        text = str(source).replace(" ", "").replace("→", "->")
        if "->" in text:
            left, right = text.split("->", 1)
            if left != right:
                return f"{left}->{right}"
    source_class = _lookup(row, "source_label", "source_class", "source", "class_from")
    target_class = _lookup(row, "target_label", "target_class", "target", "class_to")
    if source_class is not None and target_class is not None and source_class != target_class:
        return f"{source_class}->{target_class}"
    return None


def _pair_observations(row: Mapping[str, Any]) -> List[Tuple[str, float, float, float]]:
    """Extract (direction, evidence, binary label, continuous truth) tuples."""

    result: List[Tuple[str, float, float, float]] = []
    pairwise = _lookup(row, "pairwise", "pair_rows", "directed_pairs")
    labels = _lookup(row, "pair_overlap_label", "pair_labels", "truth.pair_overlap_label")
    truths = _lookup(row, "pair_overlap", "truth_pair_overlap", "truth.pair_overlap")
    if isinstance(pairwise, Mapping):
        for direction, payload in pairwise.items():
            if isinstance(payload, Mapping):
                evidence = _number(_lookup(payload, "evidence", "pairwise_index", "score", "value"))
            else:
                evidence = _number(payload)
            if evidence is None:
                continue
            label = _lookup(labels, str(direction)) if isinstance(labels, Mapping) else labels
            truth = _lookup(truths, str(direction)) if isinstance(truths, Mapping) else truths
            label_n = _number(label)
            truth_n = _number(truth)
            if label_n is None:
                label_n = 1.0 if (truth_n is not None and truth_n > 0.0) else 0.0
            if truth_n is None:
                truth_n = label_n
            result.append((str(direction), evidence, label_n, truth_n))
    directions = _lookup(row, "directions")
    evidence = _lookup(row, "evidence", "pairwise_index", "pair_evidence", "score")
    if not result and isinstance(directions, Sequence) and not isinstance(directions, (str, bytes)) and isinstance(evidence, Sequence):
        for index, direction in enumerate(directions):
            if index >= len(evidence):
                break
            value = _number(evidence[index])
            if value is not None:
                result.append((str(direction), value, 0.0, 0.0))
    if not result and _direction(row) is not None:
        value = _number(_lookup(row, "evidence", "pairwise_index", "pair_evidence", "score", "value"))
        label = _number(_lookup(row, "pair_overlap_label", "truth_overlap_label", "label", "truth_label"))
        truth = _number(_lookup(row, "pair_overlap", "truth_overlap", "overlap_truth"))
        if value is not None:
            result.append((_direction(row) or "", value, 0.0 if label is None else label, label if truth is None else truth))
    return result


def _direction_sort_key(value: Any) -> Tuple[Any, ...]:
    """Sort serialized directions by endpoint, independent of dict order."""

    text = str(value)
    if "->" in text:
        source, target = text.split("->", 1)
        try:
            return (0, int(source), int(target), text)
        except ValueError:
            return (1, source, target, text)
    return (2, text)


def _canonical_direction_text(value: Any) -> str:
    text = str(value).replace(" ", "").replace("→", "->")
    if "->" in text:
        source, target = text.split("->", 1)
        return f"{source}->{target}"
    return text


def _pair_case_key(row: Mapping[str, Any], index: int) -> Any:
    case_id_value = _lookup(row, "case_id")
    if case_id_value is not None:
        return _freeze(case_id_value)
    # Compact rows without case_id still receive a deterministic case cell
    # from all frozen axes.  The row index is a last-resort diagnostic key only
    # and is never used for a normalized runner artifact.
    fields = (
        _seed(row, index), _scenario(row), _lookup(row, "balance"),
        _lookup(row, "count_level", "count"), _lookup(row, "nuisance_dim", "dimension"),
        _lookup(row, "k"), _lookup(row, "signal_state", "signal"),
    )
    if any(value is not None for value in fields):
        return _freeze(fields)
    return ("row", int(index))


def _pair_case_records(
    rows: Sequence[Mapping[str, Any]], *, candidates: Sequence[str],
) -> Tuple[Dict[str, Dict[Any, Dict[Any, Dict[str, Any]]]], List[str]]:
    """Normalize one flat directed row into complete twelve-pair case cells."""

    candidate_set = {str(item) for item in candidates}
    records: Dict[str, Dict[Any, Dict[Any, Dict[str, Any]]]] = {
        candidate: defaultdict(dict) for candidate in candidate_set
    }
    malformed: List[str] = []
    for index, row in enumerate(rows):
        candidate = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate", "method"))
        if candidate not in candidate_set:
            continue
        seed_key = _seed(row, _pair_case_key(row, index))
        case_key = _pair_case_key(row, index)
        observations = _pair_observations(row)
        if not observations:
            malformed.append(f"{candidate}:{case_key}:missing pair observation")
            continue
        record = records[candidate][seed_key].setdefault(
            case_key,
            {
                "case_key": case_key,
                "seed": seed_key,
                "scenario": _scenario(row),
                "balance": _lookup(row, "balance"),
                "count_level": _lookup(row, "count_level", "count"),
                "nuisance_dim": _lookup(row, "nuisance_dim", "dimension"),
                "k": _lookup(row, "k"),
                "signal_state": _lookup(row, "signal_state", "signal"),
                "overlap_severity": _number(_lookup(row, "overlap_severity", "truth.overlap_severity")),
                "directions": {},
            },
        )
        directions: Dict[str, Tuple[float, float, float]] = record["directions"]
        for direction, evidence, label, truth in observations:
            canonical_direction = _canonical_direction_text(direction)
            if canonical_direction in directions:
                malformed.append(f"{candidate}:{case_key}:duplicate direction {canonical_direction}")
                continue
            numbers = (_number(evidence), _number(label), _number(truth))
            if any(value is None for value in numbers):
                malformed.append(f"{candidate}:{case_key}:non-finite direction {canonical_direction}")
                continue
            directions[canonical_direction] = (float(numbers[0]), float(numbers[1]), float(numbers[2]))
    # Four scalar classes imply 12 ordered off-diagonal directions.  Require
    # both endpoints and exact cardinality; otherwise the cell is not allowed
    # to enter a pooled estimand.
    for candidate in list(records):
        for seed_key in list(records[candidate]):
            for case_key, record in list(records[candidate][seed_key].items()):
                directions = record["directions"]
                if set(directions) != _FROZEN_DIRECTIONS:
                    malformed.append(f"{candidate}:{case_key}:incomplete directed pair cell")
                    del records[candidate][seed_key][case_key]
    return records, malformed


def _complete_pair_case_blocks(
    rows: Sequence[Mapping[str, Any]], *, candidates: Sequence[str],
) -> Tuple[List[Any], Dict[str, Dict[Any, List[Dict[str, Any]]]], Dict[str, Dict[Any, Dict[Any, Dict[str, Any]]]], List[str]]:
    """Return complete, identically paired seed blocks for all candidates."""

    records, malformed = _pair_case_records(rows, candidates=candidates)
    selected = tuple(str(candidate) for candidate in candidates if str(candidate) in records)
    if not selected:
        return [], {}, records, malformed
    common_seed = set.intersection(*(set(records[candidate]) for candidate in selected))
    blocks: List[Any] = []
    block_rows: Dict[str, Dict[Any, List[Dict[str, Any]]]] = {candidate: {} for candidate in selected}
    for seed_key in sorted(common_seed, key=canonical_json):
        case_sets = [set(records[candidate][seed_key]) for candidate in selected]
        if not case_sets or any(case_set != case_sets[0] for case_set in case_sets[1:]) or not case_sets[0]:
            malformed.append(f"seed {seed_key!r}: candidate case-cell identities do not match")
            continue
        blocks.append(seed_key)
        ordered_cases = sorted(case_sets[0], key=canonical_json)
        for candidate in selected:
            block_rows[candidate][seed_key] = [records[candidate][seed_key][case] for case in ordered_cases]
    return blocks, block_rows, records, malformed


def _flatten_pair_records(records: Sequence[Mapping[str, Any]]) -> List[Tuple[float, float, float]]:
    values: List[Tuple[float, float, float]] = []
    for record in records:
        directions = record.get("directions", {})
        for direction in sorted(directions, key=_direction_sort_key):
            values.append(tuple(float(value) for value in directions[direction]))
    return values


def _pair_scope_records(
    records: Sequence[Mapping[str, Any]], scope: str,
) -> List[Tuple[float, float, float]]:
    values: List[Tuple[float, float, float]] = []
    for record in records:
        directions = record.get("directions", {})
        truth_values = [item[2] for item in directions.values()]
        case_is_overlap = any(value > 0.0 for value in truth_values)
        if scope == "separated" and case_is_overlap:
            continue
        if scope in {"genuine", "overlap"} and not case_is_overlap:
            continue
        if scope == "clean" and str(record.get("scenario")) != "S0":
            continue
        values.extend(_flatten_pair_records([record]))
    return values


def _pair_metric_point(values: Sequence[Tuple[float, float, float]]) -> Dict[str, Any]:
    labels = [item[1] for item in values]
    evidence = [item[0] for item in values]
    truth = [item[2] for item in values]
    return binary_detection_metrics(labels, evidence, continuous_truth=truth)


def _pair_scope_summary(
    block_rows: Mapping[Any, Sequence[Mapping[str, Any]]], blocks: Sequence[Any], scope: str,
    *, draws: np.ndarray, n_resamples: int, seed: int,
) -> Dict[str, Any]:
    point_values = _pair_scope_records([record for block in blocks for record in block_rows.get(block, ())], scope)
    point = _pair_metric_point(point_values)
    sampled: Dict[str, List[float]] = defaultdict(list)
    for draw in draws:
        values = _pair_scope_records(
            [record for index in draw for record in block_rows.get(blocks[int(index)], ())], scope
        )
        metrics = _pair_metric_point(values)
        for name in ("auroc", "auprc", "fpr", "fnr", "brier", "ece"):
            number = _number(metrics.get(name))
            if number is not None:
                sampled[name].append(number)
    return {
        name: interval(_number(point.get(name)), sampled[name], n=len(point_values), n_blocks=len(blocks))
        for name in ("auroc", "auprc", "fpr", "fnr", "brier", "ece")
    } | {
        "status": "defined" if point_values else "undefined",
        "n": len(point_values), "n_blocks": len(blocks),
        "bootstrap_seed": int(seed), "bootstrap_resamples": int(n_resamples),
        "scope": scope,
    }


def _pooled_pair_blocks(rows: Sequence[Mapping[str, Any]], *, candidate: str) -> Dict[Any, List[Tuple[float, float, float]]]:
    blocks, block_rows, _records, _malformed = _complete_pair_case_blocks(
        rows, candidates=(candidate,)
    )
    return {
        seed_key: _flatten_pair_records(block_rows.get(candidate, {}).get(seed_key, ()))
        for seed_key in blocks
    }


def pooled_pair_detection(
    rows: Sequence[Mapping[str, Any]], *, candidates: Sequence[str] = CANDIDATE_ORDER,
    n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
    expected_seed_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute archived pooled directed-pair metrics on complete case cells.

    Each seed draw resamples complete seeds and recomputes the nonlinear
    pooled metric.  Candidate deltas use the identical draw matrix, including
    the separated and genuine-overlap scopes used by the reliability gates.
    """

    present = {
        candidate_id(_lookup(row, "candidate_id", "method_id", "candidate", "method"))
        for row in rows
    }
    active_candidates = tuple(str(candidate) for candidate in candidates if str(candidate) in present)
    if not active_candidates:
        return {}
    blocks, block_rows, records, malformed = _complete_pair_case_blocks(
        rows, candidates=active_candidates
    )
    if expected_seed_count is not None:
        expected_seed_keys = _expected_frozen_seeds(expected_seed_count)
        if len(blocks) != int(expected_seed_count) or (
            expected_seed_keys is not None and set(blocks) != expected_seed_keys
        ):
            blocks = []
        # A complete shared seed must contain every frozen case cell, not
        # merely the same partial subset for every candidate.  There are
        # 9 scenarios x 2 balances x 2 counts x 2 dimensions x 2 k x 3
        # signal states = 432 pair-case records per seed.
        if blocks:
            complete_grid = all(
                all(len(block_rows.get(candidate, {}).get(block, ())) == 432 for candidate in active_candidates)
                for block in blocks
            )
            if not complete_grid:
                blocks = []
    if not blocks:
        return {
            str(candidate): {
                "status": "inconclusive",
                "reason": "no complete identically paired directed-pair seed blocks",
                "malformed_rows": len(malformed),
            }
            for candidate in active_candidates
        }
    draws = joint_bootstrap_draws(len(blocks), n_resamples=n_resamples, seed=seed)
    out: Dict[str, Any] = {}
    scopes = ("all", "separated", "genuine")
    for candidate in active_candidates:
        candidate_blocks = block_rows[candidate]
        scope_summaries = {
            scope: _pair_scope_summary(
                candidate_blocks,
                blocks,
                "all" if scope == "all" else scope,
                draws=draws,
                n_resamples=n_resamples,
                seed=seed,
            )
            for scope in scopes
        }
        # Preserve the archived pooled surface at the candidate root while
        # making the scope used by each gate explicit.
        out[candidate] = {
            **scope_summaries["all"],
            "status": "defined",
            "scopes": scope_summaries,
            "separated": scope_summaries["separated"],
            "genuine": scope_summaries["genuine"],
            "n": scope_summaries["all"].get("n", 0),
            "n_blocks": len(blocks),
            "bootstrap_seed": int(seed),
            "bootstrap_resamples": int(n_resamples),
            "pooled_directed_pair_order": True,
            "malformed_rows": len(malformed),
        }

    metric_names = ("auroc", "auprc", "fpr", "fnr", "brier", "ece")
    # Scope-specific paired deltas are computed from each shared seed draw;
    # this is required for nonlinear pooled AUROC/AUPRC and continuous-truth
    # Brier, not a subtraction of independent marginal intervals.
    for baseline_name in ("B", "A"):
        if baseline_name not in active_candidates:
            continue
        for candidate in active_candidates:
            if candidate == baseline_name:
                continue
            candidate_delta: Dict[str, Any] = {}
            for scope in scopes:
                c_blocks = block_rows[candidate]
                b_blocks = block_rows[baseline_name]
                c_values = _pair_scope_records(
                    [record for block in blocks for record in c_blocks[block]],
                    "all" if scope == "all" else scope,
                )
                b_values = _pair_scope_records(
                    [record for block in blocks for record in b_blocks[block]],
                    "all" if scope == "all" else scope,
                )
                c_point = _pair_metric_point(c_values)
                b_point = _pair_metric_point(b_values)
                sampled: Dict[str, List[float]] = defaultdict(list)
                for draw in draws:
                    c_sample = _pair_scope_records(
                        [record for index in draw for record in c_blocks[blocks[int(index)]]],
                        "all" if scope == "all" else scope,
                    )
                    b_sample = _pair_scope_records(
                        [record for index in draw for record in b_blocks[blocks[int(index)]]],
                        "all" if scope == "all" else scope,
                    )
                    c_metrics = _pair_metric_point(c_sample)
                    b_metrics = _pair_metric_point(b_sample)
                    for name in metric_names:
                        c_value, b_value = _number(c_metrics.get(name)), _number(b_metrics.get(name))
                        if c_value is not None and b_value is not None:
                            sampled[name].append(c_value - b_value)
                candidate_delta[scope] = {
                    name: interval(
                        None
                        if _number(c_point.get(name)) is None or _number(b_point.get(name)) is None
                        else _number(c_point.get(name)) - _number(b_point.get(name)),
                        sampled[name],
                        n=min(len(c_values), len(b_values)),
                        n_blocks=len(blocks),
                    )
                    for name in metric_names
                }
            out[candidate]["delta_vs_" + baseline_name] = candidate_delta
            # Flat aliases are intentionally only generated for the canonical
            # all-scope delta to retain a compact machine-readable surface for
            # callers that do not need scope detail.
            for name in metric_names:
                out[candidate]["delta_vs_" + baseline_name][name] = candidate_delta["all"][name]

    # Derived reliability diagnostics are paired at the seed level below.
    _attach_pair_reliability_diagnostics(
        out, block_rows, blocks, draws, active_candidates, n_resamples=n_resamples, seed=seed
    )
    return out


def _pair_error_mean(records: Sequence[Mapping[str, Any]]) -> Optional[float]:
    values = _flatten_pair_records(records)
    if not values:
        return None
    return float(np.mean([abs(evidence - truth) for evidence, _label, truth in values]))


def _pair_designated_mean(records: Sequence[Mapping[str, Any]]) -> Optional[float]:
    values: List[float] = []
    for record in records:
        directions = record.get("directions", {})
        pair_values = [
            directions[direction][0]
            for direction in ("0->1", "1->0")
            if direction in directions
        ]
        if len(pair_values) == 2:
            values.append(float(np.mean(pair_values)))
    return None if not values else float(np.mean(values))


def _pair_seed_values(
    block_rows: Mapping[Any, Sequence[Mapping[str, Any]]], blocks: Sequence[Any],
    predicate: Callable[[Mapping[str, Any]], bool], reducer: Callable[[Sequence[Mapping[str, Any]]], Optional[float]],
) -> Dict[Any, float]:
    result: Dict[Any, float] = {}
    for block in blocks:
        selected = [record for record in block_rows.get(block, ()) if predicate(record)]
        value = reducer(selected)
        if value is not None:
            result[block] = float(value)
    return result


def _pair_metric_from_records(records: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
    value = _pair_metric_point(_flatten_pair_records(records)).get(metric)
    return _number(value)


def _attach_pair_reliability_diagnostics(
    output: MutableMapping[str, Any],
    block_rows: Mapping[str, Mapping[Any, Sequence[Mapping[str, Any]]]],
    blocks: Sequence[Any], draws: np.ndarray,
    candidates: Sequence[str], *, n_resamples: int, seed: int,
) -> None:
    """Attach paired clean, severity, strata, and false-overlap estimands."""

    del draws  # all paired intervals use the canonical complete-seed engine
    reliability_scenarios = ("S2", "H2", "T", "C", "X", "ALL")
    all_balances = ("balanced", "imbalanced")
    for candidate in candidates:
        candidate = str(candidate)
        candidate_blocks = block_rows.get(candidate, {})
        current = output.get(candidate)
        if not isinstance(current, MutableMapping):
            continue
        # Clean S0 evidence error is a continuous-truth metric, not a scalar
        # copied from a row.  Use complete seed means for the paired interval.
        clean_values = _pair_seed_values(
            candidate_blocks, blocks,
            lambda record: str(record.get("scenario")) == "S0",
            _pair_error_mean,
        )
        clean_summary = complete_seed_bootstrap(
            {candidate: clean_values}, n_resamples=n_resamples, seed=seed
        ).get(candidate, interval(None, [], n=0, n_blocks=0))
        current["clean_mae"] = clean_summary
        if "B" in block_rows:
            baseline_clean_values = _pair_seed_values(
                block_rows["B"], blocks,
                lambda record: str(record.get("scenario")) == "S0",
                _pair_error_mean,
            )
            current["clean_mae_delta"] = paired_seed_difference(
                {candidate: clean_values, "B": baseline_clean_values},
                baseline="B", n_resamples=n_resamples, seed=seed,
            ).get(candidate, interval(None, [], n=0, n_blocks=0))

        # A severity curve is defined over the frozen S0/S1/S2 and S0/H1/H2
        # scenario sequences.  It is descriptive; no causal interpretation is
        # made from the curve.
        severity_curves: Dict[str, Any] = {}
        for curve_name, scenarios in (
            ("homoscedastic", ("S0", "S1", "S2")),
            ("heteroscedastic", ("S0", "H1", "H2")),
        ):
            per_seed: Dict[Any, List[float]] = {}
            for block in blocks:
                means: List[float] = []
                complete = True
                for scenario in scenarios:
                    scenario_records = [
                        record for record in candidate_blocks.get(block, ())
                        if str(record.get("scenario")) == scenario
                    ]
                    value = _pair_error_mean(scenario_records)
                    if value is None:
                        complete = False
                        break
                    means.append(float(value))
                if complete:
                    per_seed[block] = means
            if not per_seed:
                severity_curves[curve_name] = {"status": "inconclusive", "reason": "missing complete severity sequence"}
                continue
            spearman_values: Dict[Any, float] = {}
            ordering_values: Dict[Any, float] = {}
            level_values: Dict[int, Dict[Any, float]] = {level: {} for level in range(3)}
            for block, values in per_seed.items():
                rank = spearman_rank((0.0, 1.0, 2.0), values)
                if rank is not None:
                    spearman_values[block] = rank
                ordering_values[block] = float(
                    sum(values[index + 1] >= values[index] for index in range(2)) / 2.0
                )
                for level, value in enumerate(values):
                    level_values[level][block] = float(value)
            severity_curves[curve_name] = {
                "status": "defined" if spearman_values and ordering_values else "inconclusive",
                "spearman": complete_seed_bootstrap(
                    {candidate: spearman_values}, n_resamples=n_resamples, seed=seed
                ).get(candidate, interval(None, [], n=0, n_blocks=0)),
                "ordering_rate": complete_seed_bootstrap(
                    {candidate: ordering_values}, n_resamples=n_resamples, seed=seed
                ).get(candidate, interval(None, [], n=0, n_blocks=0)),
                "n_blocks": len(per_seed),
                "severity_values": [0.0, 1.0, 2.0],
                "error": [
                    complete_seed_bootstrap(
                        {candidate: values}, n_resamples=n_resamples, seed=seed
                    ).get(candidate, interval(None, [], n=0, n_blocks=0))
                    for values in (level_values[0], level_values[1], level_values[2])
                ],
            }
        current["severity_curves"] = severity_curves

        # Every k-by-balance genuine-overlap stratum is paired to B.  The
        # frozen grid has nine scenarios x two dimensions x two count levels
        # = 36 complete genuine case cells in each such stratum and seed.
        # Check the exact identities, not only cardinality, so an unexpected
        # replacement cell cannot masquerade as a complete stratum.
        strata_values: Dict[str, Any] = {}
        for k_value in (2, 8):
            for balance in all_balances:
                key = f"{k_value}:{balance}"
                candidate_seed: Dict[Any, float] = {}
                baseline_seed: Dict[Any, float] = {}
                baseline_blocks = block_rows.get("B", {})
                for block in blocks:
                    candidate_records = [
                        record for record in candidate_blocks.get(block, ())
                        if record.get("k") == k_value
                        and str(record.get("balance")) == balance
                        and str(record.get("signal_state")) == "genuine_overlap_half"
                    ]
                    baseline_records = [
                        record for record in baseline_blocks.get(block, ())
                        if record.get("k") == k_value
                        and str(record.get("balance")) == balance
                        and str(record.get("signal_state")) == "genuine_overlap_half"
                    ]
                    # Nine scenarios x two nuisance dimensions x two count
                    # levels are the complete genuine-overlap stratum for
                    # each seed (truth is present for the genuine signal in
                    # every scenario).
                    expected_cells = {
                        (scenario_name, dimension, count_level)
                        for scenario_name in SCENARIO_ORDER
                        for dimension in (4, 32)
                        for count_level in ("small", "large")
                    }
                    candidate_cells = {
                        (
                            str(record.get("scenario")),
                            record.get("nuisance_dim"),
                            record.get("count_level"),
                        )
                        for record in candidate_records
                    }
                    baseline_cells = {
                        (
                            str(record.get("scenario")),
                            record.get("nuisance_dim"),
                            record.get("count_level"),
                        )
                        for record in baseline_records
                    }
                    if (
                        len(candidate_records) != 36
                        or len(baseline_records) != 36
                        or candidate_cells != expected_cells
                        or baseline_cells != expected_cells
                    ):
                        continue
                    c_value = _pair_metric_from_records(candidate_records, "fnr")
                    b_value = _pair_metric_from_records(baseline_records, "fnr")
                    if c_value is not None and b_value is not None:
                        candidate_seed[block] = c_value
                        baseline_seed[block] = b_value
                paired = paired_seed_difference(
                    {candidate: candidate_seed, "B": baseline_seed},
                    baseline="B", n_resamples=n_resamples, seed=seed,
                )
                strata_values[key] = paired.get(candidate, interval(None, [], n=0, n_blocks=0))
        current["strata_fnr_upper"] = strata_values

        # The lock's third criterion is the worst separated designated-pair
        # false-overlap delta.  Require all six scenario x two balance blocks;
        # an omitted block remains inconclusive rather than disappearing from a
        # maximum over whatever rows happen to be present.
        false_blocks: Dict[str, Any] = {}
        for scenario in reliability_scenarios:
            for balance in all_balances:
                key = f"{scenario}:{balance}"
                candidate_seed: Dict[Any, float] = {}
                baseline_seed: Dict[Any, float] = {}
                for block in blocks:
                    candidate_records = [
                        record for record in candidate_blocks.get(block, ())
                        if str(record.get("scenario")) == scenario
                        and str(record.get("balance")) == balance
                        and str(record.get("signal_state")) in {"linear_separated", "nonlinear_separated"}
                    ]
                    baseline_records = [
                        record for record in block_rows.get("B", {}).get(block, ())
                        if str(record.get("scenario")) == scenario
                        and str(record.get("balance")) == balance
                        and str(record.get("signal_state")) in {"linear_separated", "nonlinear_separated"}
                    ]
                    expected_cells = set(_FROZEN_SEPARATED_CELL_KEYS)
                    candidate_cells = {
                        (
                            record.get("nuisance_dim"),
                            record.get("count_level"),
                            record.get("k"),
                            record.get("signal_state"),
                        )
                        for record in candidate_records
                    }
                    baseline_cells = {
                        (
                            record.get("nuisance_dim"),
                            record.get("count_level"),
                            record.get("k"),
                            record.get("signal_state"),
                        )
                        for record in baseline_records
                    }
                    if (
                        len(candidate_records) != 16
                        or len(baseline_records) != 16
                        or candidate_cells != expected_cells
                        or baseline_cells != expected_cells
                    ):
                        continue
                    c_value = _pair_designated_mean(candidate_records)
                    b_value = _pair_designated_mean(baseline_records)
                    if c_value is not None and b_value is not None:
                        candidate_seed[block] = c_value
                        baseline_seed[block] = b_value
                paired = paired_seed_difference(
                    {candidate: candidate_seed, "B": baseline_seed},
                    baseline="B", n_resamples=n_resamples, seed=seed,
                )
                false_blocks[key] = paired.get(candidate, interval(None, [], n=0, n_blocks=0))
        current["false_overlap_blocks"] = false_blocks
        finite_upper = [_upper(value) for value in false_blocks.values()]
        finite_upper = [value for value in finite_upper if value is not None]
        current["worst_block_false_overlap_upper"] = (
            max(finite_upper) if len(finite_upper) == len(false_blocks) and false_blocks else None
        )

        if "A" in block_rows:
            candidate_genuine = _pair_seed_values(
                candidate_blocks, blocks,
                lambda record: any(value[2] > 0.0 for value in record.get("directions", {}).values()),
                lambda records: _pair_metric_from_records(records, "fnr"),
            )
            raw_genuine = _pair_seed_values(
                block_rows["A"], blocks,
                lambda record: any(value[2] > 0.0 for value in record.get("directions", {}).values()),
                lambda records: _pair_metric_from_records(records, "fnr"),
            )
            current["refinement_fnr_delta_vs_raw"] = paired_seed_difference(
                {candidate: candidate_genuine, "A": raw_genuine},
                baseline="A", n_resamples=n_resamples, seed=seed,
            ).get(candidate, interval(None, [], n=0, n_blocks=0))


def pair_detection_summary(rows: Sequence[Mapping[str, Any]], *, baseline: str = "B") -> Dict[str, Any]:
    pooled = pooled_pair_detection(rows)
    result: Dict[str, Any] = {"pooled": pooled}
    base = pooled.get(baseline, {})
    for candidate, metrics in pooled.items():
        if candidate == baseline or not isinstance(metrics, Mapping) or not isinstance(base, Mapping):
            continue
        deltas: Dict[str, Any] = {}
        for name in ("auroc", "auprc", "fpr", "fnr", "brier", "ece"):
            candidate_value = metrics.get(name, {}); base_value = base.get(name, {})
            if isinstance(candidate_value, Mapping) and isinstance(base_value, Mapping):
                # Point deltas use pooled points; uncertainty is paired below when possible.
                c_est, b_est = _number(candidate_value.get("estimate")), _number(base_value.get("estimate"))
                deltas[name] = {"estimate": None if c_est is None or b_est is None else c_est - b_est}
            else:
                deltas[name] = {"estimate": None}
        result.setdefault("deltas", {})[candidate] = deltas
    return result


# ---------------------------------------------------------------------------
# Selector panels
# ---------------------------------------------------------------------------


def _panel_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    fields = ("seed", "balance", "count_level", "nuisance_dim", "k", "signal_state")
    return tuple(_freeze(_lookup(row, field)) for field in fields)


def _reference_accuracy(row: Mapping[str, Any], head: str) -> Optional[float]:
    refs = _lookup(row, "reference_accuracies")
    if isinstance(refs, Mapping):
        value = refs.get(head)
    else:
        value = None
    return _number(value)


def _selector_score(row: Mapping[str, Any], selector: str) -> Optional[float]:
    if selector in {"probe", "linear_probe", "linear_probe_oof"}:
        return _number(_lookup(row, "linear_probe_score"))
    return _number(_lookup(row, "candidate_score"))


def _selector_row_groups(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[Any, ...], Dict[str, Dict[str, Mapping[str, Any]]]]:
    groups: Dict[Tuple[Any, ...], Dict[str, Dict[str, Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for index, row in enumerate(rows):
        scenario = _scenario(row)
        candidate = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate", "method"))
        if scenario is None or candidate is None:
            continue
        panel = _panel_key(row)
        if scenario in groups[panel][candidate]:
            # Keep duplicate marker rather than silently choosing an order-dependent row.
            groups[panel][candidate][f"__duplicate_{index}"] = row
        else:
            groups[panel][candidate][scenario] = row
    return groups


def _complete_selector_panels(
    rows: Sequence[Mapping[str, Any]], *, candidates: Sequence[str], items: Sequence[str],
) -> Tuple[Dict[Tuple[Any, ...], Dict[str, Dict[str, Mapping[str, Any]]]], List[str]]:
    # Materialize once at the public helper boundary.  The runner supplies a
    # list, but accepting an iterator here must not make the result depend on
    # how many internal passes happen to be needed.
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = tuple(rows)
    groups = _selector_row_groups(rows)
    failures: List[str] = []
    complete: Dict[Tuple[Any, ...], Dict[str, Dict[str, Mapping[str, Any]]]] = {}
    for panel, by_candidate in groups.items():
        if any(key.startswith("__duplicate_") for candidate in by_candidate.values() for key in candidate):
            failures.append(f"duplicate selector item in panel {canonical_json(panel)}")
            continue
        if any(any(item not in by_candidate.get(candidate, {}) for item in items) for candidate in candidates):
            failures.append(f"incomplete selector panel {canonical_json(panel)}")
            continue
        if any(candidate not in by_candidate for candidate in candidates):
            failures.append(f"missing selector candidate in panel {canonical_json(panel)}")
            continue
        invalid = False
        for candidate in candidates:
            for item in items:
                row = by_candidate[candidate][item]
                # The normalized runner schema has one spelling for each
                # selector outcome.  Reject old nested/alias fields rather
                # than silently choosing whichever spelling happens to be
                # present in a row.
                if any(alias in row for alias in ("score", "oi_score", "probe_score", "references", "linear_probe")):
                    invalid = True
                if _number(_lookup(row, "candidate_score")) is None:
                    invalid = True
                refs = _lookup(row, "reference_accuracies")
                if not isinstance(refs, Mapping) or any(head not in refs or _number(refs.get(head)) is None for head in REFERENCE_HEADS):
                    invalid = True
                if _number(_lookup(row, "linear_probe_score")) is None:
                    invalid = True
        if invalid:
            failures.append(f"non-finite or missing canonical selector fields in panel {canonical_json(panel)}")
            continue
        complete[panel] = {candidate: dict(by_candidate[candidate]) for candidate in candidates}
    return complete, failures


def _pick_scenario(rows: Mapping[str, Mapping[str, Any]], *, selector: str, items: Sequence[str]) -> Optional[str]:
    scored = [(scenario, _selector_score(rows[scenario], selector)) for scenario in items]
    scored = [(scenario, score) for scenario, score in scored if score is not None]
    if not scored:
        return None
    order = {scenario: index for index, scenario in enumerate(SCENARIO_ORDER)}
    scored.sort(key=lambda pair: (-float(pair[1]), order.get(pair[0], len(order)), pair[0]))
    return scored[0][0]


def selector_panel_metrics(
    rows: Sequence[Mapping[str, Any]], *, candidates: Sequence[str] = CANDIDATE_ORDER,
    subpanel: Sequence[str] = PRIMARY_NUISANCE_SCENARIOS, selector: str = "oi",
    baseline: str = "B", n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
    expected_seed_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate complete panels, paired regret, rank Spearman, and rank AUC."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = tuple(rows)
    selected_candidates = tuple(str(item) for item in candidates)
    panels, failures = _complete_selector_panels(rows, candidates=selected_candidates, items=tuple(subpanel))
    if expected_seed_count is not None:
        panel_seeds = {_freeze(panel[0]) for panel in panels}
        expected_seed_keys = _expected_frozen_seeds(expected_seed_count)
        # A selector panel is one (seed, balance, count, dimension, k,
        # signal) cell with one row for every nuisance scenario in the
        # primary subpanel.  The frozen grid has three signal states, hence
        # 2*2*2*2*3 = 48 panels per seed (not 32).
        expected_panel_count = int(expected_seed_count) * 48
        observed_axes_by_seed: Dict[Any, set[Any]] = defaultdict(set)
        for panel in panels:
            observed_axes_by_seed[_freeze(panel[0])].add(tuple(panel[1:]))
        expected_axes = set(_FROZEN_SELECTOR_PANEL_AXES)
        grid_complete = (
            len(panels) == expected_panel_count
            and (expected_seed_keys is None or panel_seeds == expected_seed_keys)
            and all(
                len(observed_axes_by_seed.get(seed_key, set())) == 48
                and observed_axes_by_seed.get(seed_key, set()) == expected_axes
                for seed_key in (expected_seed_keys or panel_seeds)
            )
        )
        if not grid_complete:
            failures.append(
                f"selector panel grid has {len(panels)} panels; expected {expected_panel_count} complete panels"
            )
    per_candidate: Dict[str, Dict[str, Dict[Any, float]]] = {candidate: {head: {} for head in REFERENCE_HEADS} for candidate in selected_candidates}
    panel_records: List[Dict[str, Any]] = []
    for panel, by_candidate in sorted(panels.items(), key=lambda item: canonical_json(item[0])):
        # Every candidate row carries the same fold-local probe score.  Use the
        # canonical baseline row to select the probe item once per panel; the
        # reference accuracies are then taken from that row's panel cells.
        probe_candidate = "B" if "B" in by_candidate else selected_candidates[0]
        probe_selected = _pick_scenario(by_candidate[probe_candidate], selector="probe", items=subpanel)
        for candidate in selected_candidates:
            selected = _pick_scenario(by_candidate[candidate], selector=selector, items=subpanel)
            if selected is None or probe_selected is None:
                continue
            for head in REFERENCE_HEADS:
                accuracies = {item: _reference_accuracy(by_candidate[candidate][item], head) for item in subpanel}
                finite = {item: value for item, value in accuracies.items() if value is not None}
                if len(finite) != len(subpanel):
                    continue
                best = max(finite.values())
                selected_accuracy = finite[selected]
                regret = best - selected_accuracy
                key = (panel[0], panel[1], panel[2], panel[3], panel[4], panel[5])
                per_candidate[candidate][head].setdefault(_seed(by_candidate[candidate][selected], panel[0]), []).append(float(regret))
                panel_records.append({
                    "panel": json_safe(panel), "candidate": candidate, "head": head, "selected": selected,
                    "reference_best": best, "selected_accuracy": selected_accuracy, "regret": regret,
                    "exact_best": regret <= EXACT_TOLERANCE, "within_one_point": regret <= WITHIN_ONE_POINT,
                    "rank_spearman": spearman_rank([_selector_score(by_candidate[candidate][item], selector) for item in subpanel], [finite[item] for item in subpanel]),
                    "probe_selected": probe_selected,
                    "probe_regret": best - finite[probe_selected],
                    "probe_exact_best": best - finite[probe_selected] <= EXACT_TOLERANCE,
                    "probe_within_one_point": best - finite[probe_selected] <= WITHIN_ONE_POINT,
                    "probe_rank_spearman": spearman_rank([_selector_score(by_candidate[probe_candidate][item], "probe") for item in subpanel], [finite[item] for item in subpanel]),
                    "balance": panel[1], "count_level": panel[2], "seed": panel[0], "total_rows": _lookup(by_candidate[candidate][selected], "total_rows") or (sum(_lookup(by_candidate[candidate][selected], "counts") or []) if isinstance(_lookup(by_candidate[candidate][selected], "counts"), Sequence) else None),
                })
    # Aggregate each complete seed and balance equally over available panel cells.
    metrics: Dict[str, Any] = {}
    for candidate in selected_candidates:
        metrics[candidate] = {"heads": {}, "n_panels": sum(1 for row in panel_records if row["candidate"] == candidate), "failures": list(failures)}
        for head in REFERENCE_HEADS:
            head_rows = [row for row in panel_records if row["candidate"] == candidate and row["head"] == head]
            by_balance_seed: Dict[str, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))
            by_balance_seed_rank: Dict[str, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))
            by_balance_seed_exact: Dict[str, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))
            by_balance_seed_within: Dict[str, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))
            for row in head_rows:
                balance = str(row.get("balance")); seed_key = _freeze(row.get("seed"))
                by_balance_seed[balance][seed_key].append(float(row["regret"]))
                if row.get("rank_spearman") is not None: by_balance_seed_rank[balance][seed_key].append(float(row["rank_spearman"]))
                by_balance_seed_exact[balance][seed_key].append(1.0 if row["exact_best"] else 0.0)
                by_balance_seed_within[balance][seed_key].append(1.0 if row["within_one_point"] else 0.0)
            balance_metrics: Dict[str, Any] = {}
            for balance in ("balanced", "imbalanced"):
                if balance not in by_balance_seed:
                    balance_metrics[balance] = {"status": "inconclusive", "reason": "missing balance stratum"}
                    continue
                values = {seed_key: np.asarray(cell_values, dtype=float) for seed_key, cell_values in by_balance_seed[balance].items()}
                # Equal cell means within a seed; complete-seed bootstrap across seeds.
                means = {seed_key: float(np.mean(value)) for seed_key, value in values.items() if value.size}
                regret = complete_seed_bootstrap({candidate: means}, n_resamples=n_resamples, seed=seed)[candidate]
                ranks = {seed_key: float(np.mean(cell_values)) for seed_key, cell_values in by_balance_seed_rank[balance].items() if cell_values}
                exact = {seed_key: float(np.mean(cell_values)) for seed_key, cell_values in by_balance_seed_exact[balance].items() if cell_values}
                within = {seed_key: float(np.mean(cell_values)) for seed_key, cell_values in by_balance_seed_within[balance].items() if cell_values}
                balance_metrics[balance] = {
                    "status": "defined", "regret": regret,
                    "rank_spearman": complete_seed_bootstrap({candidate: ranks}, n_resamples=n_resamples, seed=seed)[candidate] if ranks else interval(None, [], n=0, n_blocks=0),
                    "exact_best": complete_seed_bootstrap({candidate: exact}, n_resamples=n_resamples, seed=seed)[candidate] if exact else interval(None, [], n=0, n_blocks=0),
                    "within_one_point": complete_seed_bootstrap({candidate: within}, n_resamples=n_resamples, seed=seed)[candidate] if within else interval(None, [], n=0, n_blocks=0),
                }
            # Balance aggregation explicitly requires both strata and equal stratum means within seed.
            if all(balance_metrics.get(balance, {}).get("status") == "defined" for balance in ("balanced", "imbalanced")):
                seed_keys = sorted(set(by_balance_seed["balanced"]) & set(by_balance_seed["imbalanced"]), key=canonical_json)
                aggregate = {key: float((np.mean(by_balance_seed["balanced"][key]) + np.mean(by_balance_seed["imbalanced"][key])) / 2.0) for key in seed_keys}
                pooled = complete_seed_bootstrap({candidate: aggregate}, n_resamples=n_resamples, seed=seed)[candidate]
            else:
                pooled = {"status": "inconclusive", "estimate": None, "lower": None, "upper": None, "n": 0, "n_blocks": 0, "reason": "missing balance stratum"}
            metrics[candidate]["heads"][head] = {"pooled": pooled, "balance": balance_metrics}
            for metric_name, source in (
                ("pooled_rank_spearman", by_balance_seed_rank),
                ("pooled_exact_best", by_balance_seed_exact),
                ("pooled_within_one_point", by_balance_seed_within),
            ):
                if all(balance in source for balance in ("balanced", "imbalanced")):
                    seed_keys = sorted(set(source["balanced"]) & set(source["imbalanced"]), key=canonical_json)
                    equal_balance = {
                        key: float((np.mean(source["balanced"][key]) + np.mean(source["imbalanced"][key])) / 2.0)
                        for key in seed_keys
                    }
                    metrics[candidate]["heads"][head][metric_name] = complete_seed_bootstrap(
                        {candidate: equal_balance}, n_resamples=n_resamples, seed=seed
                    ).get(candidate, interval(None, [], n=0, n_blocks=0))
                else:
                    metrics[candidate]["heads"][head][metric_name] = {"status": "inconclusive", "reason": "missing balance stratum"}
            # Paired candidate-minus-probe regret, at the same complete panel.
            probe_by_balance_seed: Dict[str, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))
            candidate_by_balance_seed: Dict[str, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))
            for row in head_rows:
                balance = str(row.get("balance")); seed_key = _freeze(row.get("seed"))
                candidate_by_balance_seed[balance][seed_key].append(float(row["regret"]))
                probe_by_balance_seed[balance][seed_key].append(float(row["probe_regret"]))
            probe_deltas_by_seed_balance: Dict[Tuple[Any, str], float] = {}
            for balance in ("balanced", "imbalanced"):
                for seed_key in set(candidate_by_balance_seed[balance]) & set(probe_by_balance_seed[balance]):
                    # Keep the balanced/imbalanced strata as equal components
                    # of one complete-seed panel, not a row-weighted pool.
                    probe_deltas_by_seed_balance[(seed_key, balance)] = float(np.mean(candidate_by_balance_seed[balance][seed_key]) - np.mean(probe_by_balance_seed[balance][seed_key]))
            probe_deltas: Dict[Any, float] = {}
            for seed_key in {key[0] for key in probe_deltas_by_seed_balance}:
                if (seed_key, "balanced") in probe_deltas_by_seed_balance and (seed_key, "imbalanced") in probe_deltas_by_seed_balance:
                    probe_deltas[seed_key] = (probe_deltas_by_seed_balance[(seed_key, "balanced")] + probe_deltas_by_seed_balance[(seed_key, "imbalanced")]) / 2.0
            metrics[candidate]["heads"][head]["regret_vs_probe"] = complete_seed_bootstrap({candidate: probe_deltas}, n_resamples=n_resamples, seed=seed)[candidate] if probe_deltas else {"status": "inconclusive", "reason": "missing linear-probe selector or balance stratum"}

            # Nonlinear retention is paired to B on the same panel, then
            # aggregated by seed with balanced/imbalanced strata weighted
            # equally.  It is not inferred from candidate means.
            if candidate != "B" and head in {"quadratic", "knn", "rbf"}:
                candidate_panel = {
                    (tuple(row["panel"]), str(row["balance"]), _freeze(row["seed"])): float(row["regret"])
                    for row in head_rows
                }
                base_rows = {
                    (tuple(row["panel"]), str(row["balance"]), _freeze(row["seed"])): float(row["regret"])
                    for row in panel_records
                    if row["candidate"] == "B" and row["head"] == head
                }
                delta_by_seed_balance: Dict[Tuple[Any, str], List[float]] = defaultdict(list)
                for key, value in candidate_panel.items():
                    if key in base_rows:
                        delta_by_seed_balance[(key[2], key[1])].append(value - base_rows[key])
                deltas: Dict[Any, float] = {}
                for seed_key in {key[0] for key in delta_by_seed_balance}:
                    by_balance: Dict[str, List[float]] = {
                        balance: values
                        for (block_seed, balance), values in delta_by_seed_balance.items()
                        if block_seed == seed_key
                    }
                    complete_balance_values: Dict[str, float] = {}
                    for balance, values in by_balance.items():
                        # Each seed/balance must contain the complete 24-panel
                        # frozen selector block (2 d x 2 count x 2 k x 3
                        # signals), not only whichever rows survived a join.
                        panel_axes = {
                            tuple(key[0][2:])
                            for key in candidate_panel
                            if key[1] == balance and key[2] == seed_key and key in base_rows
                        }
                        if len(values) == 24 and panel_axes == set(_FROZEN_SELECTOR_STRATUM_AXES):
                            complete_balance_values[balance] = float(np.mean(values))
                    if set(complete_balance_values) == {"balanced", "imbalanced"}:
                        deltas[seed_key] = float(
                            (complete_balance_values["balanced"] + complete_balance_values["imbalanced"]) / 2.0
                        )
                metrics[candidate]["heads"][head]["regret_vs_B"] = complete_seed_bootstrap({candidate: deltas}, n_resamples=n_resamples, seed=seed)[candidate] if deltas else {"status": "inconclusive", "reason": "missing nonlinear B pairing"}
    # Rank AUC is descriptive and uses small/large count rank Spearman at log2(total rows).
    for candidate in selected_candidates:
        for head in REFERENCE_HEADS:
            records = [row for row in panel_records if row["candidate"] == candidate and row["head"] == head and row.get("rank_spearman") is not None]
            grouped_auc: Dict[Tuple[Any, ...], List[Tuple[str, float, float]]] = defaultdict(list)
            for row in records:
                panel = row["panel"]
                # group all axes except count level; total rows are deterministic 160/640 for balanced and
                # 160/640 for imbalanced in the frozen grid, so use the declared level as a stable x value.
                grouped_auc[(row["seed"], row["balance"], panel[3], panel[4], panel[5])].append((
                    str(row["count_level"]),
                    float(row["total_rows"]) if _number(row.get("total_rows")) is not None else (160.0 if row["count_level"] == "small" else 640.0),
                    float(row["rank_spearman"]),
                ))
            auc_by_seed_balance: Dict[Tuple[Any, Any], List[float]] = defaultdict(list)
            auc_axes_by_seed_balance: Dict[Tuple[Any, Any], set[Any]] = defaultdict(set)
            for group_key, points in grouped_auc.items():
                seed_key, balance = group_key[:2]
                # Exactly the frozen small/large pair is required.  A single
                # point or a duplicate count must not become a synthetic
                # stability curve by virtue of being the only available row.
                count_levels = {point[0] for point in points}
                if len(points) != 2 or count_levels != {"small", "large"}:
                    continue
                points.sort(key=lambda item: item[1])
                auc = normalized_trapezoid_auc([math.log2(item[1]) for item in points], [item[2] for item in points])
                if auc is not None:
                    auc_by_seed_balance[(seed_key, balance)].append(auc)
                    auc_axes_by_seed_balance[(seed_key, balance)].add(
                        (group_key[2], group_key[3], group_key[4])
                    )
            rank_balance_seed: Dict[str, Dict[Any, float]] = defaultdict(dict)
            for (seed_key, balance), values in auc_by_seed_balance.items():
                if (
                    len(values) == len(_FROZEN_SELECTOR_AUC_AXES)
                    and auc_axes_by_seed_balance[(seed_key, balance)] == set(_FROZEN_SELECTOR_AUC_AXES)
                ):
                    rank_balance_seed[str(balance)][seed_key] = float(np.mean(values))
            rank_pooled_values: Dict[Any, float] = {}
            for seed_key in set(rank_balance_seed["balanced"]) & set(rank_balance_seed["imbalanced"]):
                rank_pooled_values[seed_key] = (rank_balance_seed["balanced"][seed_key] + rank_balance_seed["imbalanced"][seed_key]) / 2.0
            if rank_pooled_values:
                metrics[candidate]["heads"][head]["rank_auc"] = complete_seed_bootstrap(
                    {candidate: rank_pooled_values}, n_resamples=n_resamples, seed=seed
                )[candidate]
            else:
                # Keep the interval-shaped surface even when a constant score
                # makes Spearman undefined or one of the required count/
                # balance strata is absent.  An explicit inconclusive status
                # prevents presentation code from treating the missing point
                # as a zero or as a valid stability estimate.
                metrics[candidate]["heads"][head]["rank_auc"] = {
                    **interval(None, [], n=0, n_blocks=0),
                    "status": "inconclusive",
                    "reason": "both count levels and balances required; constant ranks are undefined",
                }
    return {"status": "defined" if panels and not failures else ("inconclusive" if failures else "undefined"), "panels": panel_records, "n_complete_panels": len(panels), "failures": failures, "candidates": metrics, "subpanel": list(subpanel), "selector": selector}


downstream_selection_metrics = selector_panel_metrics


# ---------------------------------------------------------------------------
# Geometry, family drift, stable shifts, and resource summaries
# ---------------------------------------------------------------------------


def _case_cell_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        _seed(row), _scenario(row), _lookup(row, "balance"), _lookup(row, "nuisance_dim", "dimension"),
        _lookup(row, "count_level", "count"), _lookup(row, "k"), _lookup(row, "signal_state", "signal"),
    )


def _case_metric_blocks(
    rows: Sequence[Mapping[str, Any]], *, metric: str, scenario: Optional[str] = None,
    candidates: Sequence[str] = CANDIDATE_ORDER,
) -> Dict[str, Dict[Any, List[float]]]:
    result: Dict[str, Dict[Any, List[float]]] = {str(candidate): defaultdict(list) for candidate in candidates}
    for row in rows:
        candidate = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate", "method"))
        if candidate not in result:
            continue
        if scenario is not None and _scenario(row) != scenario:
            continue
        value = _metric_value(row, metric)
        if value is None:
            continue
        result[candidate][_seed(row, row_identity(row, table=metric))].append(value)
    return result


def _mean_seed_cells(
    rows: Sequence[Mapping[str, Any]], *, candidate: str, metric: str,
    scenario: Optional[str] = None, balance: Optional[str] = None,
    separated_only: bool = False,
    require_frozen_cells: bool = False,
) -> Dict[Any, float]:
    cells: Dict[Any, List[float]] = defaultdict(list)
    identities: Dict[Any, set[Any]] = defaultdict(set)
    for row in rows:
        if candidate_id(_lookup(row, "candidate_id", "method_id", "candidate", "method")) != candidate:
            continue
        if scenario is not None and _scenario(row) != scenario:
            continue
        if balance is not None and str(_lookup(row, "balance")) != balance:
            continue
        if separated_only and str(_lookup(row, "signal_state", "signal")) not in {"linear_separated", "nonlinear_separated"}:
            continue
        value = _metric_value(row, metric)
        if value is None:
            continue
        seed_key = _seed(row, row_identity(row, table=metric))
        # The case axes are the equal-weight unit.  A duplicate identity is
        # never silently converted into extra weight.
        cell = (_lookup(row, "nuisance_dim", "dimension"), _lookup(row, "count_level", "count"), _lookup(row, "k"), _lookup(row, "signal_state", "signal"))
        cell_key = _freeze(cell)
        if cell_key in identities[seed_key]:
            # Preserve one deterministic observation; complete artifact
            # validation reports the duplicate separately.
            continue
        identities[seed_key].add(cell_key)
        cells[seed_key].append(value)
    if require_frozen_cells:
        # Two separated signal states x 2 dimensions x 2 count levels x 2 k.
        # Missing axes or rows make the seed unavailable, never reweighting it.
        cells = {
            seed_key: values
            for seed_key, values in cells.items()
            if identities[seed_key] == set(_FROZEN_SEPARATED_CELL_KEYS)
        }
    return {seed_key: float(np.mean(values)) for seed_key, values in cells.items() if values}


def primary_mechanistic_claims(
    geometry_rows: Sequence[Mapping[str, Any]], *, n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED, expected_seed_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute Q1--Q4 with complete-seed means and Bonferroni intervals."""

    # Reduce a case table to (seed, balance, scenario) means; each seed then
    # averages dimensions/counts/k/signals equally by construction.
    def exact_seeds(values: Dict[Any, float]) -> Dict[Any, float]:
        expected_keys = _expected_frozen_seeds(expected_seed_count)
        if expected_seed_count is None:
            return values
        if len(values) != int(expected_seed_count) or (expected_keys is not None and set(values) != expected_keys):
            return {}
        return values

    def balanced_scenario(candidate: str, scenario: str) -> Dict[Any, float]:
        return exact_seeds(_mean_seed_cells(geometry_rows, candidate=candidate, metric="neighbor_impurity", scenario=scenario, balance="balanced", separated_only=True, require_frozen_cells=True))

    l_h2, l_s2 = balanced_scenario("L", "H2"), balanced_scenario("L", "S2")
    q1_values = {key: l_h2[key] - l_s2[key] for key in set(l_h2) & set(l_s2)}
    q1 = complete_seed_bootstrap({"Q1": q1_values}, n_resamples=n_resamples, seed=seed, level=BONFERRONI_LEVEL).get("Q1", interval(None, [], level=BONFERRONI_LEVEL))

    def penalty(candidate: str, scenario_a: str = "H2", scenario_b: str = "S2") -> Dict[Any, float]:
        first, second = balanced_scenario(candidate, scenario_a), balanced_scenario(candidate, scenario_b)
        return {key: first[key] - second[key] for key in set(first) & set(second)}

    p50_penalty, l_penalty = penalty("P50-SW"), penalty("L")
    q2_values = {key: p50_penalty[key] - l_penalty[key] for key in set(p50_penalty) & set(l_penalty)}
    q2 = complete_seed_bootstrap({"Q2": q2_values}, n_resamples=n_resamples, seed=seed, level=BONFERRONI_LEVEL).get("Q2", interval(None, [], level=BONFERRONI_LEVEL))

    def difference_in_difference(first: str, second: str, a: str, b: str, balance: str = "balanced") -> Dict[Any, float]:
        maps = {candidate: {scenario: exact_seeds(_mean_seed_cells(geometry_rows, candidate=candidate, metric="neighbor_impurity", scenario=scenario, balance=balance, separated_only=True, require_frozen_cells=True)) for scenario in (a, b)} for candidate in (first, second)}
        result: Dict[Any, float] = {}
        for key in set(maps[first][a]) & set(maps[first][b]) & set(maps[second][a]) & set(maps[second][b]):
            result[key] = (maps[first][a][key] - maps[first][b][key]) - (maps[second][a][key] - maps[second][b][key])
        return result

    q3 = within_draw_max({
        "T_minus_H2": difference_in_difference("W50-SW", "P50-SW", "T", "H2"),
        "C_minus_H2": difference_in_difference("W50-SW", "P50-SW", "C", "H2"),
    }, n_resamples=n_resamples, seed=seed, level=BONFERRONI_LEVEL)

    def balance_penalty(candidate: str) -> Dict[Any, float]:
        balanced = exact_seeds(_mean_seed_cells(geometry_rows, candidate=candidate, metric="neighbor_impurity", scenario="H2", balance="balanced", separated_only=True, require_frozen_cells=True))
        imbalanced = exact_seeds(_mean_seed_cells(geometry_rows, candidate=candidate, metric="neighbor_impurity", scenario="H2", balance="imbalanced", separated_only=True, require_frozen_cells=True))
        return {key: imbalanced[key] - balanced[key] for key in set(balanced) & set(imbalanced)}

    cb, sw = balance_penalty("W50-CB"), balance_penalty("W50-SW")
    q4_values = {key: cb[key] - sw[key] for key in set(cb) & set(sw)}
    q4 = complete_seed_bootstrap({"Q4": q4_values}, n_resamples=n_resamples, seed=seed, level=BONFERRONI_LEVEL).get("Q4", interval(None, [], level=BONFERRONI_LEVEL))
    return {
        "q1": {**q1, "claim": "balanced L H2-minus-S2 neighbor impurity", "direction": "greater_than_zero"},
        "q2": {**q2, "claim": "balanced P50-SW penalty-minus-L penalty", "direction": "less_than_zero"},
        "q3": {**q3, "claim": "within-draw max W50-SW versus P50-SW T/C difference-in-differences", "direction": "less_than_zero"},
        "q4": {**q4, "claim": "H2 imbalance penalty W50-CB-minus-W50-SW", "direction": "less_than_zero"},
        "bootstrap": {"resamples": int(n_resamples), "seed": int(seed), "interval_level": BONFERRONI_LEVEL},
    }


def negative_controls(
    geometry_rows: Sequence[Mapping[str, Any]], *, n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED, expected_seed_count: Optional[int] = None,
) -> Dict[str, Any]:
    def exact_seeds(values: Dict[Any, float]) -> Dict[Any, float]:
        if expected_seed_count is None:
            return values
        expected_keys = _expected_frozen_seeds(expected_seed_count)
        return values if len(values) == int(expected_seed_count) and (expected_keys is None or set(values) == expected_keys) else {}

    def delta(candidate: str, scenario: str) -> Dict[Any, float]:
        candidate_values = exact_seeds(_mean_seed_cells(geometry_rows, candidate=candidate, metric="neighbor_impurity", scenario=scenario, balance="balanced", separated_only=True, require_frozen_cells=True))
        baseline_values = exact_seeds(_mean_seed_cells(geometry_rows, candidate="L", metric="neighbor_impurity", scenario=scenario, balance="balanced", separated_only=True, require_frozen_cells=True))
        return {key: candidate_values[key] - baseline_values[key] for key in set(candidate_values) & set(baseline_values)}
    q2 = within_draw_max({scenario: delta("P50-SW", scenario) for scenario in HOMOSCEDASTIC_SCENARIOS}, n_resamples=n_resamples, seed=seed, level=PRIMARY_INTERVAL_LEVEL)
    # Q3's negative control is W50-SW minus P50-SW on balanced H2; using L
    # here would test a different contrast than the frozen protocol.
    def explicit_delta(candidate: str, baseline: str, scenario: str) -> Dict[Any, float]:
        candidate_values = exact_seeds(_mean_seed_cells(geometry_rows, candidate=candidate, metric="neighbor_impurity", scenario=scenario, balance="balanced", separated_only=True, require_frozen_cells=True))
        baseline_values = exact_seeds(_mean_seed_cells(geometry_rows, candidate=baseline, metric="neighbor_impurity", scenario=scenario, balance="balanced", separated_only=True, require_frozen_cells=True))
        return {key: candidate_values[key] - baseline_values[key] for key in set(candidate_values) & set(baseline_values)}
    q3_values = explicit_delta("W50-SW", "P50-SW", "H2")
    q3 = complete_seed_bootstrap({"control": q3_values}, n_resamples=n_resamples, seed=seed, level=PRIMARY_INTERVAL_LEVEL).get("control", interval(None, [], n=0, n_blocks=0))
    cb = exact_seeds(_mean_seed_cells(geometry_rows, candidate="W50-CB", metric="neighbor_impurity", scenario="H2", balance="balanced", separated_only=True, require_frozen_cells=True))
    sw = exact_seeds(_mean_seed_cells(geometry_rows, candidate="W50-SW", metric="neighbor_impurity", scenario="H2", balance="balanced", separated_only=True, require_frozen_cells=True))
    q4_values = {key: cb[key] - sw[key] for key in set(cb) & set(sw)}
    q4 = complete_seed_bootstrap({"control": q4_values}, n_resamples=n_resamples, seed=seed, level=PRIMARY_INTERVAL_LEVEL).get("control", interval(None, [], n=0, n_blocks=0))
    # Keep the control identifiers exactly aligned with protocol.json.  These
    # names are consumed by confirmation and reporting; do not add aliases
    # that could make a missing canonical control look defined.
    controls = {"q2": q2, "q3": q3, "q4": q4}
    for value in controls.values():
        value["limit"] = NEIGHBOR_NEGATIVE_CONTROL_LIMIT
        value["status"] = classify_gate(value.get("upper"), NEIGHBOR_NEGATIVE_CONTROL_LIMIT, direction="le")["status"]
    return controls


def family_drift(
    pair_rows: Sequence[Mapping[str, Any]], *, candidates: Sequence[str] = PROMOTABLE_CANDIDATES,
    baseline: str = "B", scenarios: Sequence[str] = SCENARIO_ORDER,
    n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
    expected_seed_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Equal-48-cell absolute evidence-error means by family and candidate."""

    # First reduce each (seed, family) to equal cell means.  A cell identity is
    # case identity excluding candidate and family.  Every cell must have the
    # complete twelve directed class pairs; this is intentionally strict in
    # the outcome path, not an optional convenience for partial artifacts.
    per_candidate: Dict[str, Dict[str, Dict[Any, float]]] = defaultdict(lambda: defaultdict(dict))
    directed: Dict[Tuple[str, str, Any, Any], Dict[str, Tuple[float, float]]] = defaultdict(dict)
    malformed_cells: set[Tuple[str, str, Any, Any]] = set()
    for row in pair_rows:
        candidate = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate", "method"))
        scenario = _scenario(row)
        if candidate not in set(candidates) | {baseline} or scenario not in scenarios:
            continue
        observations = _pair_observations(row)
        cell = _case_cell_key(row)
        cell_without_seed = tuple(cell[1:])
        seed_key = cell[0]
        for direction, observed, _label, truth in observations:
            cell_key = (candidate, scenario, seed_key, _freeze(cell_without_seed))
            direction_key = _canonical_direction_text(direction)
            if direction_key in directed[cell_key]:
                malformed_cells.add(cell_key)
            else:
                directed[cell_key][direction_key] = (float(observed), float(truth))
    for (candidate, scenario, seed_key, cell), observations in directed.items():
        # Four classes imply 12 ordered off-diagonal directions.  The exact
        # direction set is checked by cardinality and by both endpoints; a
        # duplicated pair is rejected by table validation before this point.
        cell_key = (candidate, scenario, seed_key, cell)
        if cell_key in malformed_cells or set(observations) != _FROZEN_DIRECTIONS:
            continue
        values = list(observations.values())
        per_candidate[candidate][scenario][(seed_key, cell)] = float(np.mean([abs(observed - truth) for observed, truth in values]))
    result: Dict[str, Any] = {}
    for candidate in candidates:
        candidate_summaries: Dict[str, Any] = {}
        for scenario in scenarios:
            cand_cells = per_candidate[candidate][scenario]
            base_cells = per_candidate[baseline][scenario]
            # Require each arm to contain exactly the frozen 48 cells and
            # require those identities to match.  Taking a 48-cell
            # intersection would otherwise allow one missing cell to be
            # replaced by an unexpected extra cell without changing the
            # pooled row count.
            candidate_cells_by_seed: Dict[Any, set[Any]] = defaultdict(set)
            baseline_cells_by_seed: Dict[Any, set[Any]] = defaultdict(set)
            for key in cand_cells:
                candidate_cells_by_seed[key[0]].add(key[1])
            for key in base_cells:
                baseline_cells_by_seed[key[0]].add(key[1])
            valid_seed_candidates = {
                seed_key
                for seed_key, cell_keys in candidate_cells_by_seed.items()
                if len(cell_keys) == 48
                and len(baseline_cells_by_seed.get(seed_key, set())) == 48
                and cell_keys == baseline_cells_by_seed.get(seed_key, set())
            }
            common = {
                key for key in (set(cand_cells) & set(base_cells))
                if key[0] in valid_seed_candidates
            }
            by_seed_c: Dict[Any, List[float]] = defaultdict(list); by_seed_b: Dict[Any, List[float]] = defaultdict(list)
            for key in common:
                seed_key = key[0]
                by_seed_c[seed_key].append(cand_cells[key])
                by_seed_b[seed_key].append(base_cells[key])
            valid_seeds = [key for key in by_seed_c if key in valid_seed_candidates and len(by_seed_c[key]) == 48 and len(by_seed_b[key]) == 48]
            if expected_seed_count is not None:
                expected_seed_keys = _expected_frozen_seeds(expected_seed_count)
                if len(valid_seeds) != int(expected_seed_count) or (
                    expected_seed_keys is not None and set(valid_seeds) != expected_seed_keys
                ):
                    valid_seeds = []
            if not valid_seeds:
                candidate_summaries[scenario] = {"status": "inconclusive", "reason": "missing complete 48-cell family blocks"}
                continue
            values_c = {key: float(np.mean(by_seed_c[key])) for key in valid_seeds}
            values_b = {key: float(np.mean(by_seed_b[key])) for key in valid_seeds}
            summary = paired_seed_difference({candidate: values_c, baseline: values_b}, baseline=baseline, n_resamples=n_resamples, seed=seed)[candidate]
            summary["limit"] = FAMILY_DRIFT_LIMIT
            summary["gate"] = classify_gate(summary.get("upper"), FAMILY_DRIFT_LIMIT, direction="le")
            candidate_summaries[scenario] = summary
        result[candidate] = candidate_summaries
    return result


def stable_shift_gates(
    pair_rows: Sequence[Mapping[str, Any]], *, candidates: Sequence[str] = PROMOTABLE_CANDIDATES,
    baseline: str = "B", n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
    expected_seed_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute complete-cell stable-to-shift transfer contrasts.

    Every target scenario cell is matched to its S0 cell on seed, balance,
    nuisance dimension, count, k, and signal state.  The matching is done for
    all twelve directions before any metric is pooled, preventing a scalar
    value from the final row in a case from replacing the other eleven pairs.
    """

    selected = tuple(str(candidate) for candidate in candidates)
    records, malformed = _pair_case_records(
        pair_rows, candidates=tuple(selected) + (str(baseline),)
    )

    def cell_signature(record: Mapping[str, Any]) -> Any:
        return _freeze((record.get("nuisance_dim"), record.get("count_level"), record.get("k"), record.get("signal_state")))

    def transfer_values(
        candidate: str, block: Any, scenario: str, balance: str,
    ) -> Optional[Dict[str, List[Tuple[float, float, float]]]]:
        candidate_records = [
            record for record in records.get(candidate, {}).get(block, {}).values()
            if str(record.get("balance")) == balance
        ]
        by_scenario: Dict[str, Dict[Any, Mapping[str, Any]]] = defaultdict(dict)
        duplicate_cells = False
        for record in candidate_records:
            scenario_name = str(record.get("scenario"))
            signature = cell_signature(record)
            if signature in by_scenario[scenario_name]:
                duplicate_cells = True
            else:
                by_scenario[scenario_name][signature] = record
        shifted = by_scenario.get(scenario, {})
        clean = by_scenario.get("S0", {})
        if (
            duplicate_cells
            or len(shifted) != 24
            or len(clean) != 24
            or set(shifted) != set(clean)
            or set(shifted) != _FROZEN_STABLE_CELL_KEYS
        ):
            return None
        result: Dict[str, List[Tuple[float, float, float]]] = {
            "shift": [], "stable": [], "drift": []
        }
        # The caller supplies the stable scenario separately.  ``shift`` is
        # populated here; ``stable`` is filled by the outer helper.
        for key in sorted(shifted, key=canonical_json):
            shift_record = shifted[key]
            clean_record = clean[key]
            result["shift"].extend(_flatten_pair_records([shift_record]))
            for direction in sorted(shift_record.get("directions", {}), key=_direction_sort_key):
                if direction in clean_record.get("directions", {}):
                    result["drift"].append((
                        shift_record["directions"][direction][0],
                        clean_record["directions"][direction][0],
                        clean_record["directions"][direction][2],
                    ))
        if len(result["drift"]) != 24 * 12:
            return None
        return result

    def metric_value(values: Sequence[Tuple[float, float, float]], metric: str) -> Optional[float]:
        if metric == "clean_mae":
            return None if not values else float(np.mean([abs(item[0] - item[2]) for item in values]))
        if metric == "nuisance_drift":
            return None if not values else float(np.mean([abs(item[0] - item[1]) for item in values]))
        return _pair_metric_from_records(
            [{"directions": {str(index): value for index, value in enumerate(values)}}], metric
        )

    result: Dict[str, Any] = {}
    for candidate in selected:
        candidate_result: Dict[str, Any] = {}
        for balance in ("balanced", "imbalanced"):
            contrasts: Dict[str, Any] = {}
            for shifted, stable in (("X", "H2"), ("ALL", "C")):
                for metric in ("auroc", "auprc", "brier", "clean_mae", "nuisance_drift"):
                    higher = metric in {"auroc", "auprc"}
                    candidate_effects: Dict[Any, float] = {}
                    baseline_effects: Dict[Any, float] = {}
                    common_blocks = sorted(
                        set(records.get(candidate, {})) & set(records.get(baseline, {})),
                        key=canonical_json,
                    )
                    for block in common_blocks:
                        c_shift = transfer_values(candidate, block, shifted, balance)
                        c_stable = transfer_values(candidate, block, stable, balance)
                        b_shift = transfer_values(str(baseline), block, shifted, balance)
                        b_stable = transfer_values(str(baseline), block, stable, balance)
                        if None in (c_shift, c_stable, b_shift, b_stable):
                            continue
                        c_shift_metric = metric_value(c_shift["shift" if metric not in {"clean_mae", "nuisance_drift"} else "shift"], metric)
                        c_stable_metric = metric_value(c_stable["shift" if metric not in {"clean_mae", "nuisance_drift"} else "shift"], metric)
                        b_shift_metric = metric_value(b_shift["shift" if metric not in {"clean_mae", "nuisance_drift"} else "shift"], metric)
                        b_stable_metric = metric_value(b_stable["shift" if metric not in {"clean_mae", "nuisance_drift"} else "shift"], metric)
                        # For clean MAE use the target scenario's own truth;
                        # for drift use target-vs-S0 matched directions.
                        if metric == "clean_mae":
                            c_shift_metric = metric_value(c_shift["shift"], metric)
                            c_stable_metric = metric_value(c_stable["shift"], metric)
                            b_shift_metric = metric_value(b_shift["shift"], metric)
                            b_stable_metric = metric_value(b_stable["shift"], metric)
                        elif metric == "nuisance_drift":
                            c_shift_metric = metric_value(c_shift["drift"], metric)
                            c_stable_metric = metric_value(c_stable["drift"], metric)
                            b_shift_metric = metric_value(b_shift["drift"], metric)
                            b_stable_metric = metric_value(b_stable["drift"], metric)
                        if any(value is None for value in (c_shift_metric, c_stable_metric, b_shift_metric, b_stable_metric)):
                            continue
                        if higher:
                            candidate_effects[block] = float(c_stable_metric - c_shift_metric)
                            baseline_effects[block] = float(b_stable_metric - b_shift_metric)
                        else:
                            candidate_effects[block] = float(c_shift_metric - c_stable_metric)
                            baseline_effects[block] = float(b_shift_metric - b_stable_metric)
                    if expected_seed_count is not None:
                        expected_seed_keys = _expected_frozen_seeds(expected_seed_count)
                        if len(candidate_effects) != int(expected_seed_count) or (
                            expected_seed_keys is not None and set(candidate_effects) != expected_seed_keys
                        ):
                            candidate_effects = {}
                            baseline_effects = {}
                    if candidate_effects:
                        summary = paired_seed_difference(
                            {candidate: candidate_effects, str(baseline): baseline_effects},
                            baseline=str(baseline), n_resamples=n_resamples, seed=seed,
                        ).get(candidate, interval(None, [], n=0, n_blocks=0))
                        summary["gate"] = classify_gate(summary.get("upper"), 0.0, direction="le")
                    else:
                        summary = {"status": "inconclusive", "reason": "missing complete directed stable/shift blocks"}
                    contrasts[f"{shifted}-{stable}:{metric}"] = summary
            candidate_result[balance] = contrasts
        result[candidate] = candidate_result
    if malformed:
        for candidate in selected:
            result.setdefault(candidate, {})["malformed_pair_rows"] = len(malformed)
    return result


def resource_summaries(
    resource_rows: Sequence[Mapping[str, Any]], *, candidates: Sequence[str] = PROMOTABLE_CANDIDATES,
    baseline: str = "B", n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
    expected_seed_count: Optional[int] = None, stage: Optional[str] = None,
    locked_candidate: Optional[str] = None,
) -> Dict[str, Any]:
    thresholds = {"median_total": 1.10, "p95_total": 1.25, "median_score_fixed": 1.25, "median_peak_memory": 1.15, "p95_peak_memory": 1.25, "individual_peak_memory": 1.50}
    result: Dict[str, Any] = {}
    stage_value = str(stage or "development").lower()
    if stage_value == "confirmation":
        if locked_candidate is None:
            expected_methods = ()
        else:
            expected_methods = (baseline, str(locked_candidate))
    elif stage_value == "development":
        expected_methods = tuple(dict.fromkeys((baseline, "L", *tuple(candidates))))
    else:
        # Keep the low-level helper useful for compact unit fixtures while
        # requiring callers that analyze a frozen artifact to pass its stage.
        expected_methods = tuple(dict.fromkeys((baseline, "L", *tuple(candidates))))
    by_identity: Dict[Any, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    malformed = False
    seen_rows: set[Any] = set()
    for row_index, row in enumerate(resource_rows):
        method = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate", "method"))
        resource_id = _lookup(row, "resource_id")
        case_id = _lookup(row, "case_id")
        if method not in set(expected_methods) or resource_id is None or case_id is None:
            malformed = True
            continue
        # resource_worker intentionally omits a top-level seed.  The case id
        # is the frozen seed-bearing identity, so pairing by resource/case is
        # the only stable key and avoids row-index-dependent false pairs.
        identity = (str(resource_id), _freeze(case_id), method)
        if identity in seen_rows:
            malformed = True
        seen_rows.add(identity)
        by_identity[identity[:2]][method] = dict(row)
        status = str(_lookup(row, "status") or "ok").lower()
        if status not in {"ok", "pass", "completed"}:
            malformed = True
        for field in (
            "total_wall_seconds", "total_cpu_seconds", "score_fixed_wall_seconds",
            "score_fixed_cpu_seconds", "peak_rss_bytes",
        ):
            value = _number(_lookup(row, field))
            if value is None or value < 0.0:
                malformed = True
    if expected_seed_count is not None:
        expected_count = int(expected_seed_count) * len(RESOURCE_CELLS) * len(expected_methods)
        if len(seen_rows) != expected_count:
            malformed = True
        observed_resource_ids = {str(identity[0]) for identity in seen_rows}
        if observed_resource_ids != set(RESOURCE_CELLS):
            malformed = True
        for resource_id in RESOURCE_CELLS:
            for method in expected_methods:
                count = sum(
                    1 for identity in seen_rows
                    if str(identity[0]) == resource_id and identity[2] == method
                )
                if count != int(expected_seed_count):
                    malformed = True
    for candidate in candidates:
        candidate = str(candidate)
        by_identity_candidate = by_identity
        if malformed:
            result[candidate] = {"status": "inconclusive", "reason": "resource rows are incomplete, duplicated, or contain errors", "ratios": {}}
            continue
        ratios: Dict[str, List[float]] = defaultdict(list)
        missing = 0
        for identity, values in sorted(by_identity_candidate.items(), key=lambda item: canonical_json(item[0])):
            if candidate not in values or baseline not in values:
                continue
            base_row, candidate_row = values[baseline], values[candidate]
            fields = (("total", "total_wall_seconds"), ("score_fixed", "score_fixed_wall_seconds"), ("peak_memory", "peak_rss_bytes"))
            for stage_name, field in fields:
                numerator = _number(candidate_row.get(field))
                denominator = _number(base_row.get(field))
                if numerator is None or denominator is None or denominator <= 0.0:
                    if stage_name == "total": missing += 1
                    continue
                ratios[stage_name].append(float(numerator / denominator))
        if missing or any(not ratios.get(stage_name) for stage_name in ("total", "score_fixed", "peak_memory")):
            result[candidate] = {"status": "inconclusive", "reason": "missing paired resource evidence", "ratios": {key: list(values) for key, values in ratios.items()}}
            continue
        total, score, memory = ratios["total"], ratios["score_fixed"], ratios["peak_memory"]
        summaries = {
            "median_total": float(np.median(total)), "p95_total": float(np.quantile(total, 0.95, method="linear")),
            "median_score_fixed": float(np.median(score)), "median_peak_memory": float(np.median(memory)),
            "p95_peak_memory": float(np.quantile(memory, 0.95, method="linear")), "individual_peak_memory": float(np.max(memory)),
        }
        gates = {name: classify_gate(value, thresholds[name], direction="le") for name, value in summaries.items()}
        result[candidate] = {"status": "pass" if all(gate["status"] == "pass" for gate in gates.values()) else "fail", "ratios": {key: list(values) for key, values in ratios.items()}, "summaries": summaries, "gates": gates}
    return result

# ---------------------------------------------------------------------------
# Gates, eligibility, and immutable locks
# ---------------------------------------------------------------------------


def classify_gate(value: Any, threshold: float, *, direction: str = "le", strict: bool = False) -> Dict[str, Any]:
    number = _number(value)
    if number is None:
        return {"status": "inconclusive", "value": None, "threshold": float(threshold), "direction": direction}
    threshold = float(threshold)
    if direction == "le":
        passed = number < threshold if strict else number <= threshold
    elif direction == "ge":
        passed = number > threshold if strict else number >= threshold
    else:
        raise ValueError("direction must be 'le' or 'ge'")
    return {"status": "pass" if passed else "fail", "value": number, "threshold": threshold, "direction": direction, "strict": bool(strict)}


def _gate_status(value: Any) -> str:
    if isinstance(value, Mapping) and value.get("status") in {"pass", "fail", "inconclusive"}:
        return str(value["status"])
    if isinstance(value, Mapping):
        if isinstance(value.get("gate"), Mapping):
            return _gate_status(value["gate"])
        child_statuses = [_gate_status(item) for item in value.values() if isinstance(item, Mapping)]
        if child_statuses:
            return "fail" if "fail" in child_statuses else "inconclusive" if "inconclusive" in child_statuses else "pass"
        return "pass" if _number(value.get("upper")) is not None else "inconclusive"
    return "defined" if _number(value) is not None else "inconclusive"


def _upper(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        return _number(value.get("upper", value.get("high")))
    return _number(value)


def genuine_overlap_gates(
    overlap: Mapping[str, Any], *, baseline: str = "B", raw_baseline: str = "A",
    candidates: Sequence[str] = PROMOTABLE_CANDIDATES,
) -> Dict[str, Any]:
    """Evaluate frozen overlap reliability margins from paired summaries."""

    if not isinstance(overlap, Mapping):
        pooled: Mapping[str, Any] = {}
    elif isinstance(overlap.get("pooled"), Mapping):
        pooled = overlap["pooled"]
    elif isinstance(overlap.get("candidates"), Mapping):
        pooled = overlap["candidates"]
    else:
        pooled = overlap

    def metric_summary(candidate_value: Mapping[str, Any], scope: str, metric: str) -> Any:
        scoped = candidate_value.get(scope)
        if isinstance(scoped, Mapping) and metric in scoped:
            return scoped.get(metric)
        scopes = candidate_value.get("scopes")
        if isinstance(scopes, Mapping) and isinstance(scopes.get(scope), Mapping):
            return scopes[scope].get(metric)
        return candidate_value.get(metric)

    def paired_delta(candidate_value: Mapping[str, Any], candidate_name: str, base_name: str, scope: str, metric: str) -> Any:
        direct = candidate_value.get(f"delta_vs_{base_name}")
        if isinstance(direct, Mapping):
            scoped = direct.get(scope)
            if isinstance(scoped, Mapping) and metric in scoped:
                return scoped.get(metric)
            if metric in direct and isinstance(direct.get(metric), Mapping):
                return direct.get(metric)
        # Compact records may provide an explicit candidate-level delta.
        for key in (f"{metric}_delta", f"pair_{metric}_delta"):
            if key in candidate_value:
                return candidate_value.get(key)
        external = overlap.get("deltas") if isinstance(overlap, Mapping) else None
        if isinstance(external, Mapping):
            candidate_external = external.get(candidate_name)
            if isinstance(candidate_external, Mapping):
                value = candidate_external.get(scope)
                if isinstance(value, Mapping) and metric in value:
                    return value.get(metric)
                return candidate_external.get(metric)
        return None

    result: Dict[str, Any] = {}
    for candidate in candidates:
        current = pooled.get(candidate, {}) if isinstance(pooled, Mapping) else {}
        base = pooled.get(baseline, {}) if isinstance(pooled, Mapping) else {}
        raw = pooled.get(raw_baseline, {}) if isinstance(pooled, Mapping) else {}
        current = current if isinstance(current, Mapping) else {}
        base = base if isinstance(base, Mapping) else {}
        raw = raw if isinstance(raw, Mapping) else {}
        gates: Dict[str, Any] = {}

        base_fpr_summary = metric_summary(base, "separated", "fpr")
        base_fpr = _number(base_fpr_summary.get("estimate")) if isinstance(base_fpr_summary, Mapping) else _number(base_fpr_summary)
        fpr_threshold = 0.0 if base_fpr is not None and base_fpr >= 0.10 else 0.02
        fpr_strict = base_fpr is not None and base_fpr >= 0.10
        for name, scope, direction, threshold in (
            ("fpr", "separated", "le", fpr_threshold),
            ("auroc", "all", "ge", -0.03),
            ("auprc", "all", "ge", -0.03),
            ("fnr", "genuine", "le", 0.05),
            ("brier", "all", "le", 0.05),
        ):
            explicit = paired_delta(current, str(candidate), baseline, scope, name)
            if isinstance(explicit, Mapping):
                bound = _upper(explicit) if direction == "le" else _number(explicit.get("lower"))
                gates[name] = classify_gate(
                    bound, threshold, direction=direction,
                    strict=fpr_strict if name == "fpr" else False,
                )
            else:
                gates[name] = {"status": "inconclusive", "reason": f"missing paired {scope} {name} delta"}

        refinement = current.get("refinement_fnr_delta_vs_raw")
        if refinement is None:
            refinement = paired_delta(current, str(candidate), raw_baseline, "genuine", "fnr")
        gates["refinement_fnr_vs_raw"] = (
            classify_gate(_upper(refinement), 0.002, direction="le")
            if isinstance(refinement, Mapping)
            else {"status": "inconclusive", "reason": "missing refinement FNR delta"}
        )
        clean = current.get("clean_mae_delta")
        if clean is None:
            clean = paired_delta(current, str(candidate), baseline, "clean", "clean_mae")
        gates["clean_mae"] = (
            classify_gate(_upper(clean), 0.05, direction="le")
            if isinstance(clean, Mapping)
            else {"status": "inconclusive", "reason": "missing clean MAE delta"}
        )

        # Severity curves use point estimates for the minimum rank/order
        # checks and explicit candidate-vs-B error losses for each severity.
        severity_curves = current.get("severity_curves")
        baseline_curves = base.get("severity_curves")
        severity_children: Dict[str, Any] = {}
        if isinstance(current.get("severity_gates"), Mapping):
            severity_children = dict(current["severity_gates"])
        elif severity_curves is True:
            severity_children = {"status": "pass"}
        elif severity_curves is False:
            severity_children = {"status": "fail"}
        elif isinstance(severity_curves, Mapping) and isinstance(baseline_curves, Mapping):
            for curve_name in ("homoscedastic", "heteroscedastic"):
                curve = severity_curves.get(curve_name)
                base_curve = baseline_curves.get(curve_name)
                if not isinstance(curve, Mapping) or not isinstance(base_curve, Mapping):
                    severity_children[curve_name] = {"status": "inconclusive", "reason": "missing severity curve"}
                    continue
                c_spearman = curve.get("spearman")
                c_order = curve.get("ordering_rate")
                c_spearman_value = _number(c_spearman.get("estimate")) if isinstance(c_spearman, Mapping) else _number(c_spearman)
                c_order_value = _number(c_order.get("estimate")) if isinstance(c_order, Mapping) else _number(c_order)
                b_spearman = base_curve.get("spearman")
                b_order = base_curve.get("ordering_rate")
                b_spearman_value = _number(b_spearman.get("estimate")) if isinstance(b_spearman, Mapping) else _number(b_spearman)
                b_order_value = _number(b_order.get("estimate")) if isinstance(b_order, Mapping) else _number(b_order)
                block_count_match = (
                    _number(curve.get("n_blocks")) is not None
                    and _number(base_curve.get("n_blocks")) is not None
                    and int(curve.get("n_blocks")) == int(base_curve.get("n_blocks"))
                )
                errors = curve.get("error")
                base_errors = base_curve.get("error")
                error_losses: List[Optional[float]] = []
                if isinstance(errors, Sequence) and isinstance(base_errors, Sequence) and len(errors) == len(base_errors):
                    for c_error, b_error in zip(errors, base_errors):
                        c_est = _number(c_error.get("estimate")) if isinstance(c_error, Mapping) else _number(c_error)
                        b_est = _number(b_error.get("estimate")) if isinstance(b_error, Mapping) else _number(b_error)
                        if c_est is None or b_est is None:
                            error_losses.append(None)
                        else:
                            loss = c_est - b_est
                            error_losses.append(loss)
                statuses = [
                    "inconclusive" if not block_count_match or c_spearman_value is None or b_spearman_value is None else "pass" if c_spearman_value >= 0.9 and c_spearman_value >= b_spearman_value - 0.05 else "fail",
                    "inconclusive" if not block_count_match or c_order_value is None or b_order_value is None else "pass" if c_order_value >= 0.9 and c_order_value >= b_order_value - 0.05 else "fail",
                ]
                severity_children[curve_name] = {
                    "status": "fail" if "fail" in statuses else "inconclusive" if "inconclusive" in statuses else "pass",
                    "spearman": c_spearman_value,
                    "ordering_rate": c_order_value,
                    "error_loss": error_losses,
                }
        else:
            severity_children = {"status": "inconclusive", "reason": "missing severity curves"}
        if isinstance(severity_children, Mapping) and severity_children.get("status") in {"pass", "fail", "inconclusive"}:
            gates["severity"] = dict(severity_children)
        else:
            statuses = [_gate_status(value) for value in severity_children.values()] if isinstance(severity_children, Mapping) else []
            gates["severity"] = {"status": "fail" if "fail" in statuses else "inconclusive" if "inconclusive" in statuses else "pass"}

        strata = current.get("strata_fnr_upper")
        strata_gates: Dict[str, Any] = {}
        expected_strata = {"2:balanced", "2:imbalanced", "8:balanced", "8:imbalanced"}
        if (
            isinstance(strata, Mapping)
            and set(str(key) for key in strata) == expected_strata
            and all(isinstance(value, Mapping) for value in strata.values())
        ):
            for key, value in sorted(strata.items(), key=lambda item: str(item[0])):
                upper = _upper(value)
                lower = _number(value.get("lower")) if isinstance(value, Mapping) else None
                if upper is None or lower is None:
                    status = "inconclusive"
                elif lower > 0.0 or upper > 0.05:
                    status = "fail"
                elif upper <= 0.0:
                    status = "pass"
                else:
                    status = "inconclusive"
                strata_gates[str(key)] = {
                    "status": status, "estimate": _number(value.get("estimate")) if isinstance(value, Mapping) else None,
                    "lower": lower, "upper": upper,
                }
            strata_statuses = [value["status"] for value in strata_gates.values()]
            gates["strata_fnr"] = {
                "status": "fail" if "fail" in strata_statuses else "inconclusive" if "inconclusive" in strata_statuses else "pass",
                "by_stratum": strata_gates,
            }
        else:
            gates["strata_fnr"] = {"status": "inconclusive", "reason": "missing k-by-balance strata"}

        false_blocks = current.get("false_overlap_blocks")
        if (
            isinstance(false_blocks, Mapping)
            and len(false_blocks) == 12
            and set(str(key) for key in false_blocks)
            == {f"{scenario}:{balance}" for scenario in ("S2", "H2", "T", "C", "X", "ALL") for balance in ("balanced", "imbalanced")}
            and all(_upper(value) is not None for value in false_blocks.values())
        ):
            worst = max(float(_upper(value)) for value in false_blocks.values())
            gates["false_overlap"] = {
                **classify_gate(worst, 0.0, direction="le"),
                "n_blocks": 12,
            }
        else:
            gates["false_overlap"] = {
                "status": "inconclusive",
                "reason": "all 12 separated scenario/balance blocks are required",
                "n_blocks": len(false_blocks) if isinstance(false_blocks, Mapping) else 0,
            }
        # ``false_overlap`` is deliberately retained as a complete paired
        # diagnostic and lock criterion, but it is not a genuine-overlap
        # reliability gate.  A positive lock metric therefore must not make
        # an otherwise reliable candidate ineligible; only the reliability
        # surfaces below determine this aggregate status.
        reliability_gate_names = (
            "fpr", "auroc", "auprc", "fnr", "brier",
            "refinement_fnr_vs_raw", "clean_mae", "severity", "strata_fnr",
        )
        statuses = [
            gates[name].get("status")
            for name in reliability_gate_names
            if isinstance(gates.get(name), Mapping)
        ]
        result[candidate] = {
            "status": "fail" if "fail" in statuses else "inconclusive" if "inconclusive" in statuses else "pass",
            "gates": gates,
        }
    return result


def _validated_structural_evidence(manifest: Mapping[str, Any], *, stage: Optional[str] = None) -> Dict[str, Any]:
    """Extract structural gates only when tied to an authorized review hash."""

    raw = manifest.get("structural_gates") if isinstance(manifest, Mapping) else None
    if not isinstance(raw, Mapping):
        determinism = manifest.get("determinism") if isinstance(manifest, Mapping) else None
        deterministic_status = (
            {"status": "pass", "reason": None}
            if isinstance(determinism, Mapping) and str(determinism.get("status", "")).lower() == "pass"
            else {"status": "inconclusive", "reason": "manifest determinism did not pass"}
        )
        return {
            "exact_parity": {"status": "inconclusive", "reason": "missing manifest.structural_gates"},
            "leakage": {"status": "inconclusive", "reason": "missing manifest.structural_gates"},
            "determinism": deterministic_status,
            "review_identity": {"status": "inconclusive", "reason": "missing structural review hash"},
        }

    def normalize(value: Any, name: str) -> Dict[str, Any]:
        if isinstance(value, Mapping) and value.get("status") in {"pass", "fail", "inconclusive"}:
            return {"status": str(value["status"]), "reason": value.get("reason")}
        if isinstance(value, Mapping) and "value" in value:
            value = value.get("value")
        if value is True:
            return {"status": "pass", "reason": None}
        if value is False:
            return {"status": "fail", "reason": f"{name} failed"}
        if isinstance(value, str) and value.lower() in {"pass", "passed", "go", "true"}:
            return {"status": "pass", "reason": None}
        if isinstance(value, str) and value.lower() in {"fail", "failed", "false"}:
            return {"status": "fail", "reason": f"{name} failed"}
        if isinstance(value, Mapping) and value:
            booleans = [item for item in value.values() if isinstance(item, bool)]
            if booleans:
                if all(booleans) and len(booleans) == len(value):
                    return {"status": "pass", "reason": None}
                if any(item is False for item in booleans):
                    return {"status": "fail", "reason": f"{name} failed"}
        return {"status": "inconclusive", "reason": f"missing {name} evidence"}

    authorization = manifest.get("authorization_hashes", {})
    auth_values = {
        str(value)
        for value in authorization.values()
        if isinstance(authorization, Mapping) and isinstance(value, str) and value
    }
    source_decisions = raw.get("source_decisions")
    stage_value = str(stage or "development").lower()
    # Each stage consumes one exact prerequisite set.  In particular, the
    # implementation review is consumed by smoke; development consumes the
    # resulting smoke decision plus pre-screen review; confirmation consumes
    # the immutable development promotion and prior-regression decisions.
    required_sources = (
        ("implementation_review",)
        if stage_value == "smoke"
        else ("smoke_decision", "pre_screen_review")
        if stage_value in {"development", "screen"}
        else ("promotion_decision", "prior_regression_decision")
        if stage_value == "confirmation"
        else ()
    )
    if not isinstance(source_decisions, Mapping) or not source_decisions:
        review_identity = {"status": "inconclusive", "reason": "missing structural source decisions"}
    else:
        source_failures: List[str] = []
        source_statuses: Dict[str, str] = {}
        missing_sources = [name for name in required_sources if name not in source_decisions]
        unexpected_sources = [name for name in source_decisions if name not in required_sources]
        source_failures.extend(f"{name}: missing source decision" for name in missing_sources)
        source_failures.extend(f"{name}: unexpected source decision" for name in unexpected_sources)
        for name, source in source_decisions.items():
            if not isinstance(source, Mapping):
                source_failures.append(f"{name}: malformed source decision")
                continue
            source_hash = source.get("sha256")
            source_status = str(source.get("status", "")).lower()
            source_statuses[str(name)] = source_status
            if not isinstance(source_hash, str) or not source_hash:
                source_failures.append(f"{name}: missing sha256")
            elif not auth_values:
                source_failures.append(f"{name}: manifest has no authorized source hashes")
            elif source_hash not in auth_values:
                source_failures.append(f"{name}: sha256 is not authorized")
            if source_status not in {"go", "pass", "passed", "locked"}:
                source_failures.append(f"{name}: status={source_status!r}")
        if source_failures:
            review_identity = {"status": "fail", "reason": "invalid structural source decisions", "failures": source_failures}
        else:
            review_identity = {"status": "pass", "reason": None, "source_statuses": source_statuses}
    determinism = manifest.get("determinism")
    deterministic_status = (
        {"status": "pass", "reason": None}
        if isinstance(determinism, Mapping) and str(determinism.get("status", "")).lower() == "pass"
        else {"status": "inconclusive", "reason": "manifest determinism did not pass"}
    )
    return {
        "exact_parity": normalize(raw.get("exact_parity"), "exact_parity"),
        "leakage": normalize(raw.get("leakage_no_refit"), "leakage_no_refit"),
        "determinism": deterministic_status,
        "review_identity": review_identity,
    }


def candidate_eligibility(
    summary: Mapping[str, Any], *, candidates: Sequence[str] = PROMOTABLE_CANDIDATES,
) -> Dict[str, Any]:
    """Combine all frozen candidate gates; failures exclude, missing is inconclusive."""

    result: Dict[str, Any] = {}
    artifact_identity = summary.get("artifact_identity", {})
    artifact_blocked = not isinstance(artifact_identity, Mapping) or artifact_identity.get("status") != "pass"

    def bool_gate(value: Any, name: str) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            status = value.get("status")
            if status in {"pass", "fail", "inconclusive"}:
                return {"status": str(status), "reason": None}
            value = value.get("value")
        if value is True:
            return {"status": "pass", "reason": None}
        if value is False:
            return {"status": "fail", "reason": f"{name} failed"}
        return {"status": "inconclusive", "reason": f"missing {name} evidence"}

    def interval_gate(value: Any, threshold: float, *, direction: str = "le") -> Dict[str, Any]:
        if isinstance(value, Mapping):
            bound = _upper(value) if direction == "le" else _number(value.get("lower"))
            return classify_gate(bound, threshold, direction=direction)
        return {"status": "inconclusive", "reason": "missing paired interval"}

    def genuine_status(value: Any) -> Dict[str, Any]:
        # False-overlap is a lock-ranking criterion, not a pre-ranking
        # reliability gate.  Keep its detailed gate in the source surface,
        # but do not let a positive value exclude an otherwise reliable arm.
        required = {
            "fpr", "auroc", "auprc", "fnr", "brier", "refinement_fnr_vs_raw",
            "clean_mae", "severity", "strata_fnr",
        }
        if not isinstance(value, Mapping) or not isinstance(value.get("gates"), Mapping):
            return {"status": "inconclusive", "reason": "missing complete genuine-overlap gate surface"}
        gates = value["gates"]
        if not required.issubset({str(key) for key in gates}):
            return {"status": "inconclusive", "reason": "incomplete genuine-overlap gate surface"}
        statuses = [_gate_status(gates[key]) for key in required]
        if "fail" in statuses:
            return {"status": "fail", "reason": "genuine-overlap gate failed"}
        if "inconclusive" in statuses:
            return {"status": "inconclusive", "reason": "genuine-overlap gate incomplete"}
        return {"status": "pass", "reason": None}

    def family_status(value: Any) -> Dict[str, Any]:
        expected = set(SCENARIO_ORDER)
        if not isinstance(value, Mapping) or set(str(key) for key in value) != expected:
            return {"status": "inconclusive", "reason": "all nine family blocks are required"}
        statuses = []
        for scenario in SCENARIO_ORDER:
            child = value.get(scenario)
            statuses.append(_gate_status(child.get("gate") if isinstance(child, Mapping) else child))
        if "fail" in statuses:
            return {"status": "fail", "reason": "family drift gate failed"}
        if "inconclusive" in statuses:
            return {"status": "inconclusive", "reason": "family drift gate incomplete"}
        return {"status": "pass", "required_blocks": list(SCENARIO_ORDER)}

    def stable_status(value: Any) -> Dict[str, Any]:
        expected_balances = {"balanced", "imbalanced"}
        expected_contrasts = {
            f"{shifted}-{stable}:{metric}"
            for shifted, stable in (("X", "H2"), ("ALL", "C"))
            for metric in ("auroc", "auprc", "brier", "clean_mae", "nuisance_drift")
        }
        if not isinstance(value, Mapping) or set(str(key) for key in value) != expected_balances:
            return {"status": "inconclusive", "reason": "both stable-shift balances are required"}
        statuses: List[str] = []
        for balance in expected_balances:
            child = value.get(balance)
            if not isinstance(child, Mapping) or set(str(key) for key in child) != expected_contrasts:
                return {"status": "inconclusive", "reason": "all stable-shift contrasts are required"}
            statuses.extend(_gate_status(item.get("gate") if isinstance(item, Mapping) else item) for item in child.values())
        if "fail" in statuses:
            return {"status": "fail", "reason": "stable-shift gate failed"}
        if "inconclusive" in statuses:
            return {"status": "inconclusive", "reason": "stable-shift gate incomplete"}
        return {"status": "pass", "required_balances": sorted(expected_balances)}

    def resource_status(value: Any) -> Dict[str, Any]:
        required = {"median_total", "p95_total", "median_score_fixed", "median_peak_memory", "p95_peak_memory", "individual_peak_memory"}
        if not isinstance(value, Mapping) or value.get("status") not in {"pass", "fail", "inconclusive"}:
            return {"status": "inconclusive", "reason": "missing resource gate surface"}
        summaries = value.get("summaries")
        gates = value.get("gates")
        if not isinstance(summaries, Mapping) or not required.issubset(set(summaries)) or not isinstance(gates, Mapping) or not required.issubset(set(gates)):
            return {"status": "inconclusive", "reason": "incomplete resource gate surface"}
        return {"status": str(value["status"]), "reason": value.get("reason")}

    top_genuine = summary.get("genuine_overlap_gates")
    top_stable = summary.get("stable_shift_gates")
    top_resources = summary.get("resource_gates")
    top_structural = summary.get("structural_gates")
    top_family = summary.get("family_drift")

    for candidate in candidates:
        metrics = summary.get("candidates", {}).get(candidate, {}) if isinstance(summary.get("candidates", {}), Mapping) else {}
        genuine_value = top_genuine.get(candidate) if isinstance(top_genuine, Mapping) else None
        stable_value = top_stable.get(candidate) if isinstance(top_stable, Mapping) else None
        resource_value = top_resources.get(candidate) if isinstance(top_resources, Mapping) else None
        gate_sets = [
            ("genuine_overlap", genuine_status(genuine_value)),
            ("stable_shift", stable_status(stable_value)),
            ("resources", resource_status(resource_value)),
        ]
        structural: Dict[str, Any] = {}
        for key in ("exact_parity", "determinism", "leakage", "review_identity"):
            structural[key] = bool_gate(top_structural.get(key) if isinstance(top_structural, Mapping) else None, key)
        gate_sets.append(("structural", structural))
        gate_sets.append(("family_drift", family_status(top_family.get(candidate) if isinstance(top_family, Mapping) else None)))
        gate_sets.append((
            "nuisance_linear_regret_vs_probe",
            interval_gate(metrics.get("nuisance_linear_regret_vs_probe"), 0.01, direction="le"),
        ))
        nonlinear = metrics.get("nonlinear_retention")
        for head in ("quadratic", "knn", "rbf"):
            value = nonlinear.get(head) if isinstance(nonlinear, Mapping) else None
            gate_sets.append((f"nonlinear_{head}", interval_gate(value, 0.01, direction="le")))
        statuses: List[str] = []
        reasons: List[str] = []
        if artifact_blocked:
            gate_sets.append(("artifact_identity", {"status": "inconclusive", "reason": "normalized artifact identity/completeness is not verified"}))
        for name, gates in gate_sets:
            status = _gate_status(gates)
            statuses.append(status)
            if status in {"fail", "inconclusive"}:
                reasons.append(f"{name}:{status}")
        result[candidate] = {"status": "fail" if "fail" in statuses else "inconclusive" if "inconclusive" in statuses else "pass", "eligible": bool(statuses) and all(status == "pass" for status in statuses), "reasons": reasons, "gates": {name: gates for name, gates in gate_sets}}
    return result


def _decision_hash(value: Mapping[str, Any] | os.PathLike[str] | str) -> str:
    if isinstance(value, Mapping):
        return sha256_bytes((canonical_json(value) + "\n").encode("utf-8"))
    return sha256_file(value)


def validate_decision_prerequisite(
    decision: Mapping[str, Any] | os.PathLike[str] | str, *, protocol_hash: Optional[str] = None,
    code_identity_hash: Optional[str] = None, expected_status: Optional[Sequence[str]] = None,
    locked_candidate: Optional[str] = None, promotion_hash: Optional[str] = None,
) -> Dict[str, Any]:
    if isinstance(decision, Mapping):
        payload = dict(decision); digest = _decision_hash(payload)
    else:
        path = Path(decision); payload = json.loads(path.read_text(encoding="utf-8")); digest = _decision_hash(path)
    failures: List[str] = []
    if protocol_hash is not None and payload.get("protocol_sha256", payload.get("protocol_hash")) != protocol_hash:
        failures.append("protocol hash mismatch")
    if code_identity_hash is not None and payload.get("code_identity_sha256", payload.get("code_identity_hash")) != code_identity_hash:
        failures.append("code identity hash mismatch")
    if expected_status is not None and str(payload.get("status", "")).lower() not in {str(item).lower() for item in expected_status}:
        failures.append("status mismatch")
    found_lock = payload.get("locked_candidate", payload.get("selected_candidate"))
    if locked_candidate is not None and found_lock != locked_candidate:
        failures.append("locked candidate mismatch")
    if promotion_hash is not None and payload.get("promotion_decision_sha256", payload.get("promotion_decision_hash")) != promotion_hash:
        failures.append("promotion decision hash mismatch")
    return {"status": "pass" if not failures else "fail", "sha256": digest, "payload": payload, "failures": failures}


def select_lock(
    summary: Mapping[str, Any], *, protocol_hash: Optional[str] = None, code_identity_hash: Optional[str] = None,
    existing_decision: Optional[Mapping[str, Any] | os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Apply the sequential 0.001 rule once and never select a runner-up."""

    if existing_decision is not None:
        prior = validate_decision_prerequisite(
            existing_decision, protocol_hash=protocol_hash,
            code_identity_hash=code_identity_hash,
            expected_status=("locked", "pass"),
        )
        if prior["status"] != "pass":
            raise PermissionError("existing lock identity does not match current analysis")
        prior_candidate = prior["payload"].get("locked_candidate", prior["payload"].get("selected_candidate"))
        if prior_candidate is not None:
            return {**prior["payload"], "immutable_reuse": True}
    stage = str(summary.get("stage", "development")).lower()
    eligibility = candidate_eligibility(summary)
    eligible = [candidate for candidate in PROMOTABLE_CANDIDATES if eligibility.get(candidate, {}).get("eligible")]
    decision: Dict[str, Any] = {
        "schema_version": 1, "stage": stage, "status": "inconclusive", "selected_candidate": None,
        "locked_candidate": None, "eligible": eligible, "excluded": {candidate: eligibility.get(candidate, {}).get("reasons", []) for candidate in PROMOTABLE_CANDIDATES},
        "lock_rule": {"criteria": ["nuisance_linear_regret_vs_probe_upper", "worst_family_drift_upper", "worst_block_false_overlap_upper"], "sequential_tolerance": PROMOTION_TOLERANCE, "runner_up_after_lock": False},
        "protocol_sha256": protocol_hash, "code_identity_sha256": code_identity_hash,
        "input_manifest_sha256": summary.get("input_manifest_sha256"),
        "input_table_hashes": summary.get("input_table_hashes", {}),
        # Carry the validated stage prerequisite chain unchanged; downstream
        # prior-regression/confirmation consumers verify this exact surface.
        "structural_gates": summary.get("structural_gates", {}),
        "pipeline_stop": False,
    }
    lock_criterion_names = (
        "nuisance_linear_regret_vs_probe_upper",
        "worst_family_drift_upper",
        "worst_block_false_overlap_upper",
    )

    def missing_lock_criteria(candidate: str) -> List[str]:
        metrics = summary.get("candidates", {}).get(candidate, {})
        return [
            name for name in lock_criterion_names
            if not isinstance(metrics, Mapping) or _number(metrics.get(name)) is None
        ]

    if stage in {"smoke", "structural"}:
        decision["reason"] = "smoke/structural artifacts cannot rank or lock candidates"
        decision["pipeline_stop"] = True
    elif eligible:
        missing_by_candidate = {
            candidate: missing_lock_criteria(candidate)
            for candidate in eligible
            if missing_lock_criteria(candidate)
        }
        if missing_by_candidate:
            # Do not rank a partial lock surface: an absent criterion could
            # change the winner.  This is separate from reliability
            # eligibility and keeps false-overlap's role as criterion 3.
            decision["status"] = "inconclusive"
            decision["pipeline_stop"] = True
            decision["lock_criteria_missing"] = missing_by_candidate
            decision["reason"] = "required lock criteria are incomplete; no candidate was ranked"
            return decision

        def criterion(candidate: str, key: str, default: float = float("inf")) -> float:
            metrics = summary.get("candidates", {}).get(candidate, {})
            value = metrics.get(key)
            if isinstance(value, Mapping): value = value.get("upper", value.get("estimate"))
            return default if _number(value) is None else float(value)
        remaining = list(eligible)
        for key in lock_criterion_names:
            values = [criterion(candidate, key) for candidate in remaining]
            if not values: break
            best = min(values)
            remaining = [candidate for candidate in remaining if criterion(candidate, key) <= best + PROMOTION_TOLERANCE]
        if len(remaining) > 1:
            order = {candidate: index for index, candidate in enumerate(("P50-SW", "P25", "P50-CB", "W50-SW", "W50-CB", "M50-SW", "M50-CB"))}
            remaining.sort(key=lambda candidate: order.get(candidate, len(order)))
        if remaining:
            decision["selected_candidate"] = decision["locked_candidate"] = remaining[0]
            decision["status"] = "locked"
            decision["reason"] = "exactly one immutable sequential lock"
            decision["eligible_after_tie_break"] = remaining
    else:
        # A no-lock artifact is terminal for this input.  Distinguish a
        # definite exclusion (every candidate failed a gate) from a missing or
        # incomplete surface that is merely inconclusive; neither is allowed
        # to look like the pipeline may proceed with an implicit runner-up.
        decision["pipeline_stop"] = True
        statuses = [eligibility.get(candidate, {}).get("status") for candidate in PROMOTABLE_CANDIDATES]
        if statuses and all(status == "fail" for status in statuses):
            decision["status"] = "stopped_no_eligible"
            decision["reason"] = "all candidates failed at least one frozen gate"
        else:
            decision["status"] = "inconclusive"
            decision["reason"] = "required evidence is incomplete or inconclusive; pipeline stopped"
    return decision


select_promotion = select_lock


def evaluate_confirmation(
    summary: Mapping[str, Any], *, locked_candidate: str, promotion_decision: Mapping[str, Any] | os.PathLike[str] | str,
    prior_regression_decision: Mapping[str, Any] | os.PathLike[str] | str,
    protocol_hash: Optional[str] = None, code_identity_hash: Optional[str] = None,
) -> Dict[str, Any]:
    promotion = validate_decision_prerequisite(promotion_decision, protocol_hash=protocol_hash, code_identity_hash=code_identity_hash, expected_status=("locked", "pass"), locked_candidate=locked_candidate)
    prior = validate_decision_prerequisite(prior_regression_decision, protocol_hash=protocol_hash, code_identity_hash=code_identity_hash, expected_status=("pass", "passed"), locked_candidate=locked_candidate, promotion_hash=promotion.get("sha256"))
    failures = promotion["failures"] + prior["failures"]
    eligibility = candidate_eligibility(summary, candidates=(locked_candidate,))
    algorithm_status = "pass" if not failures and eligibility.get(locked_candidate, {}).get("eligible") else "fail" if failures or eligibility.get(locked_candidate, {}).get("status") == "fail" else "inconclusive"
    mechanisms = summary.get("primary_claims", {}) if isinstance(summary.get("primary_claims", {}), Mapping) else {}
    mechanism_status: Dict[str, str] = {}
    for name in ("q1", "q2", "q3", "q4"):
        claim = mechanisms.get(name, {})
        upper = _upper(claim)
        lower = _number(claim.get("lower")) if isinstance(claim, Mapping) else None
        direction = claim.get("direction") if isinstance(claim, Mapping) else None
        if upper is None or lower is None:
            mechanism_status[name] = "inconclusive"
        elif direction == "greater_than_zero":
            mechanism_status[name] = "pass" if lower > 0 else "fail"
        else:
            mechanism_status[name] = "pass" if upper < 0 else "fail"
    # Every primary claim also requires its predeclared negative-control
    # margin, except Q1 which has no separate control.  Evaluate the paired
    # upper 95% bound directly rather than trusting a self-declared status;
    # missing or undefined controls remain inconclusive and never become a
    # mechanism pass by omission.
    negative_controls = summary.get("negative_controls")
    negative_control_status: Dict[str, Any] = {}
    for name in ("q2", "q3", "q4"):
        control = negative_controls.get(name) if isinstance(negative_controls, Mapping) else None
        upper = _upper(control)
        if upper is None:
            negative_control_status[name] = {
                "status": "inconclusive", "estimate": _number(control.get("estimate")) if isinstance(control, Mapping) else None,
                "lower": _number(control.get("lower")) if isinstance(control, Mapping) else None,
                "upper": None, "threshold": NEIGHBOR_NEGATIVE_CONTROL_LIMIT,
                "reason": "missing or undefined paired negative-control upper bound",
            }
        else:
            negative_control_status[name] = {
                **classify_gate(upper, NEIGHBOR_NEGATIVE_CONTROL_LIMIT, direction="le"),
                "estimate": _number(control.get("estimate")) if isinstance(control, Mapping) else None,
                "lower": _number(control.get("lower")) if isinstance(control, Mapping) else None,
                "upper": upper,
            }
    mechanism_status.update({
        f"negative_control_{name}": value["status"]
        for name, value in negative_control_status.items()
    })
    mechanism_values = list(mechanism_status.values())
    return {
        "status": "pass" if algorithm_status == "pass" and all(value == "pass" for value in mechanism_values) else algorithm_status,
        "schema_version": 1,
        "stage": "confirmation",
        "protocol_sha256": protocol_hash,
        "code_identity_sha256": code_identity_hash,
        "input_manifest_sha256": summary.get("input_manifest_sha256"),
        "input_table_hashes": summary.get("input_table_hashes", {}),
        "locked_candidate": locked_candidate,
        "structural_gates": summary.get("structural_gates", {}),
        "algorithm": {"status": algorithm_status, "locked_candidate": locked_candidate, "prerequisite_failures": failures},
        "mechanism": {
            "status": "pass" if all(value == "pass" for value in mechanism_values) else "fail" if "fail" in mechanism_values else "inconclusive",
            "claims": {name: mechanism_status[name] for name in ("q1", "q2", "q3", "q4")},
            "negative_controls": negative_control_status,
        },
        "runner_up_after_failure": False,
    }


# ---------------------------------------------------------------------------
# Summary assembly and deterministic artifact writers
# ---------------------------------------------------------------------------


def _rows_by_table(artifact_or_rows: Any) -> Dict[str, List[Dict[str, Any]]]:
    if isinstance(artifact_or_rows, Mapping) and isinstance(artifact_or_rows.get("tables"), Mapping):
        tables = {str(key): flatten_rows(value) for key, value in artifact_or_rows["tables"].items()}
        # Runner child tables intentionally carry only the case/candidate
        # identity.  Join the immutable case identity here so every downstream
        # estimand sees seed/scenario/balance/dimension/count/k/signal without
        # duplicating that metadata in each JSONL row.
        case_rows = tables.get("case_rows.jsonl", [])
        identities: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        for row in case_rows:
            case = _lookup(row, "case_id")
            candidate = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate"))
            if case is not None and candidate is not None:
                identities[(_freeze(case), candidate)] = dict(row)
        for table_name, rows in tables.items():
            if table_name == "case_rows.jsonl":
                continue
            enriched: List[Dict[str, Any]] = []
            for row in rows:
                case = _lookup(row, "case_id")
                candidate = candidate_id(_lookup(row, "candidate_id", "method_id", "candidate"))
                base = identities.get((_freeze(case), candidate), {})
                merged = dict(base)
                merged.update(row)
                enriched.append(merged)
            tables[table_name] = enriched
        return tables
    rows = flatten_rows(artifact_or_rows)
    return {"case_rows.jsonl": rows, "geometry_rows.jsonl": rows, "prototype_rows.jsonl": rows, "pair_rows.jsonl": rows, "selector_panels.jsonl": rows, "resource_rows.jsonl": rows}


def _descriptive_diagnostics(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Summarize the fitted diagnostic chain without changing estimands.

    These are report-only, equal-row descriptive summaries.  They make the
    geometry -> prototype -> refinement -> score concordance visible while
    keeping the frozen inferential functions above responsible for every gate.
    No claim of mediation or causal attribution is made from these means.
    """

    def mean_path(rows: Sequence[Mapping[str, Any]], *paths: str) -> Optional[float]:
        values = [_number(_lookup(row, *paths)) for row in rows]
        finite = [value for value in values if value is not None]
        return float(np.mean(finite)) if finite else None

    def mean_bool_path(rows: Sequence[Mapping[str, Any]], *paths: str) -> Optional[float]:
        values = [_lookup(row, *paths) for row in rows]
        booleans = [1.0 if value is True else 0.0 for value in values if isinstance(value, bool)]
        return float(np.mean(booleans)) if booleans else None

    def candidate_rows(name: str, table_name: str) -> List[Mapping[str, Any]]:
        return [
            row for row in tables.get(table_name, ())
            if candidate_id(_lookup(row, "candidate_id", "method_id", "candidate", "method")) == name
        ]

    def keyed_values(rows: Sequence[Mapping[str, Any]], *paths: str) -> Dict[Any, float]:
        values: Dict[Any, float] = {}
        for row in rows:
            case_id = _lookup(row, "case_id")
            value = _number(_lookup(row, *paths))
            if case_id is not None and value is not None:
                # Normalized identity validation rejects duplicate case rows;
                # setdefault keeps this report-only helper deterministic when
                # called directly on a compact fixture.
                values.setdefault(_freeze(case_id), value)
        return values

    def descriptive(values: Sequence[float]) -> Dict[str, Any]:
        finite = [float(value) for value in values if _number(value) is not None]
        if not finite:
            return {"n": 0, "mean": None, "median": None}
        return {"n": len(finite), "mean": float(np.mean(finite)), "median": float(np.median(finite))}

    def linked_spearman(left: Mapping[Any, float], right: Mapping[Any, float]) -> Dict[str, Any]:
        common = sorted(set(left) & set(right), key=canonical_json)
        value = spearman_rank([left[key] for key in common], [right[key] for key in common]) if len(common) >= 2 else None
        return {"n": len(common), "spearman": value}

    names = sorted(
        {
            candidate
            for table_name in ("case_rows.jsonl", "geometry_rows.jsonl", "prototype_rows.jsonl")
            for row in tables.get(table_name, ())
            for candidate in [candidate_id(_lookup(row, "candidate_id", "method_id", "candidate", "method"))]
            if candidate is not None
        },
        key=lambda value: (CANDIDATE_ORDER.index(value) if value in CANDIDATE_ORDER else len(CANDIDATE_ORDER), value),
    )
    result: Dict[str, Any] = {}
    for candidate in names:
        case = candidate_rows(candidate, "case_rows.jsonl")
        geometry = candidate_rows(candidate, "geometry_rows.jsonl")
        prototype = candidate_rows(candidate, "prototype_rows.jsonl")
        score_by_case = keyed_values(case, "candidate_score", "score_fixed", "score")
        baseline_scores = keyed_values(candidate_rows("B", "case_rows.jsonl"), "candidate_score", "score_fixed", "score")
        raw_scores = keyed_values(candidate_rows("A", "case_rows.jsonl"), "candidate_score", "score_fixed", "score")
        if candidate == "B":
            score_delta_vs_b = {"n": 0, "mean": None, "median": None, "not_applicable": True}
            refinement_delta = descriptive([score_by_case[key] - raw_scores[key] for key in set(score_by_case) & set(raw_scores)])
        else:
            score_delta_vs_b = descriptive([score_by_case[key] - baseline_scores[key] for key in set(score_by_case) & set(baseline_scores)])
            refinement_delta = None
        geometry_neighbor = keyed_values(geometry, "geometry.cross_class_neighbor_impurity")
        prototype_purity = keyed_values(prototype, "prototype.weighted_majority_assignment_purity")
        delta_values = {
            key: score_by_case[key] - baseline_scores[key]
            for key in set(score_by_case) & set(baseline_scores)
        } if candidate != "B" else {
            key: score_by_case[key] - raw_scores[key]
            for key in set(score_by_case) & set(raw_scores)
        }
        conditioning = [row for row in case if isinstance(_lookup(row, "conditioning"), Mapping)]
        refinement = [
            row for row in case
            if isinstance(_lookup(row, "refinement_stage", "refinement"), Mapping)
        ]
        result[candidate] = {
            "n_case_rows": len(case),
            "n_geometry_rows": len(geometry),
            "n_prototype_rows": len(prototype),
            "conditioning": {
                "mode": next(
                    (_lookup(row, "conditioning.mode") for row in conditioning if _lookup(row, "conditioning.mode") is not None),
                    None,
                ),
                "condition_before": mean_path(conditioning, "conditioning.condition_before", "condition_before"),
                "condition_after": mean_path(conditioning, "conditioning.condition_after", "condition_after"),
                "regularized_eigenvalue_count": mean_path(
                    conditioning,
                    "conditioning.regularized_eigenvalue_count",
                    "conditioning.regularized_variance_count",
                ),
                "cap_or_floor_active_rate": mean_bool_path(
                    conditioning, "conditioning.cap_or_floor_active", "cap_or_floor_active"
                ),
            },
            "refinement": {
                "activity_rate": mean_path(
                    refinement,
                    "refinement_stage.refinement_activity_all_pre_refinement_prototypes",
                    "refinement.refinement_activity_all_pre_refinement_prototypes",
                    "refinement_activity",
                    "prototype_activity",
                ),
                "applied_count": mean_path(
                    refinement, "refinement_stage.applied_count", "refinement.applied_count"
                ),
                "eligible_count": mean_path(
                    refinement, "refinement_stage.eligible_count", "refinement.eligible_count"
                ),
            },
            "geometry": {
                "neighbor_impurity": mean_path(geometry, "geometry.cross_class_neighbor_impurity"),
                "oracle_neighbor_jaccard": mean_path(geometry, "geometry.oracle_neighbor_jaccard"),
                "pair_distance_spearman": mean_path(geometry, "geometry.pair_distance_spearman"),
            },
            "prototype": {
                "occupied_normalized_label_entropy_macro": mean_path(
                    prototype, "prototype.occupied_normalized_label_entropy_macro"
                ),
                "weighted_majority_assignment_purity": mean_path(
                    prototype, "prototype.weighted_majority_assignment_purity"
                ),
                "owner_label_agreement": mean_path(prototype, "prototype.owner_label_agreement"),
                "mixed_rate_occupied": mean_path(prototype, "prototype.mixed_rate_occupied"),
            },
            "score": {
                **descriptive(list(score_by_case.values())),
                "mean": mean_path(case, "candidate_score", "score_fixed") if score_by_case else None,
            },
            # These are descriptive, case-paired movements only.  They are
            # deliberately not eligibility gates and cannot establish a
            # causal mediation path.
            "score_delta_vs_B": score_delta_vs_b,
            "refinement_B_minus_A": refinement_delta,
            "concordance": {
                "neighbor_impurity_vs_score_movement": linked_spearman(geometry_neighbor, delta_values),
                "prototype_purity_vs_score_movement": linked_spearman(prototype_purity, delta_values),
            },
        }
    return result


def build_summary(
    artifact_or_rows: Any, *, stage: Optional[str] = None, promotion_decision: Optional[Mapping[str, Any] | os.PathLike[str] | str] = None,
    protocol_hash: Optional[str] = None, code_identity_hash: Optional[str] = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES, bootstrap_seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    tables = _rows_by_table(artifact_or_rows)
    manifest = artifact_or_rows.get("manifest", {}) if isinstance(artifact_or_rows, Mapping) else {}
    stage_value = str(stage or manifest.get("stage", "development")).lower()
    identity = validate_normalized_artifact(artifact_or_rows, expected_stage=stage_value, protocol_hash=protocol_hash, code_identity_hash=code_identity_hash) if isinstance(artifact_or_rows, Mapping) and "tables" in artifact_or_rows else {"status": "pass", "failures": []}
    summary: Dict[str, Any] = {
        "schema_version": 1, "stage": stage_value, "n_rows": sum(len(rows) for rows in tables.values()),
        "status": "structural" if stage_value == "smoke" else "defined", "protocol_sha256": protocol_hash,
        "code_identity_sha256": code_identity_hash, "artifact_identity": identity,
        "statistics": {"bootstrap_resamples": int(n_resamples), "bootstrap_seed": int(bootstrap_seed), "interval": "two-sided_percentile_95", "bonferroni_interval": BONFERRONI_LEVEL, "bootstrap_unit": "complete_seed"},
        "candidates": {}, "tables": {name: {"rows": len(rows)} for name, rows in tables.items()},
    }
    summary["structural_gates"] = _validated_structural_evidence(
        manifest if isinstance(manifest, Mapping) else {}, stage=stage_value
    )
    summary["diagnostics"] = _descriptive_diagnostics(tables)
    if isinstance(artifact_or_rows, Mapping):
        checks = artifact_or_rows.get("table_checks", {})
        if isinstance(checks, Mapping):
            summary["input_table_hashes"] = {
                str(name): value.get("sha256")
                for name, value in checks.items()
                if isinstance(value, Mapping) and value.get("sha256") is not None
            }
        summary["input_manifest_sha256"] = (
            sha256_file(artifact_or_rows.get("manifest_path"))
            if artifact_or_rows.get("manifest_path") and Path(str(artifact_or_rows["manifest_path"])).exists()
            else None
        )
    geometry = tables.get("geometry_rows.jsonl", [])
    pairs = tables.get("pair_rows.jsonl", [])
    selectors = tables.get("selector_panels.jsonl", [])
    resources = tables.get("resource_rows.jsonl", [])
    if stage_value != "smoke":
        expected_seed_count = 12 if stage_value in {"development", "screen"} else 24 if stage_value == "confirmation" else None
        declared_methods = manifest.get("method_ids") if isinstance(manifest, Mapping) else None
        if isinstance(declared_methods, Sequence) and not isinstance(declared_methods, (str, bytes)):
            analysis_methods = tuple(
                candidate for candidate in (candidate_id(value) for value in declared_methods)
                if candidate is not None
            )
        else:
            analysis_methods = tuple(CANDIDATE_ORDER)
        # Confirmation is a diagnostic panel containing fixed comparators plus
        # the already locked candidate.  Only the locked method is eligible;
        # the other methods remain useful for paired diagnostics.
        analysis_promotables = tuple(
            candidate for candidate in analysis_methods if candidate in PROMOTABLE_CANDIDATES
        )
        locked_candidate = candidate_id(
            manifest.get("locked_candidate", manifest.get("promotion_candidate"))
        ) if isinstance(manifest, Mapping) else None
        summary["primary_claims"] = primary_mechanistic_claims(
            geometry,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
            expected_seed_count=expected_seed_count,
        )
        summary["negative_controls"] = negative_controls(
            geometry,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
            expected_seed_count=expected_seed_count,
        )
        summary["family_drift"] = family_drift(
            pairs, candidates=analysis_promotables, n_resamples=n_resamples,
            seed=bootstrap_seed, expected_seed_count=expected_seed_count,
        )
        summary["stable_shift_gates"] = stable_shift_gates(
            pairs, candidates=analysis_promotables,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
            expected_seed_count=expected_seed_count,
        )
        summary["genuine_overlap"] = pooled_pair_detection(
            pairs, n_resamples=n_resamples, seed=bootstrap_seed,
            expected_seed_count=expected_seed_count,
        )
        summary["genuine_overlap_gates"] = genuine_overlap_gates(
            summary["genuine_overlap"], candidates=analysis_promotables
        )
        # L is retained as a non-promotable descriptive comparator in the
        # development resource surface.  Confirmation is intentionally only
        # B versus the already locked candidate.
        resource_candidates = (
            ("L", *analysis_promotables)
            if stage_value == "development"
            else analysis_promotables
        )
        summary["resource_gates"] = resource_summaries(
            resources,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
            expected_seed_count=expected_seed_count,
            stage=stage_value,
            candidates=resource_candidates,
            locked_candidate=locked_candidate,
        )
        selection = selector_panel_metrics(
            selectors,
            candidates=analysis_methods,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
            expected_seed_count=expected_seed_count,
        )
        summary["selection"] = selection
        for candidate, metrics in selection.get("candidates", {}).items():
            summary["candidates"].setdefault(candidate, {}).update(metrics)
            heads = metrics.get("heads", {}) if isinstance(metrics, Mapping) else {}
            linear = heads.get("linear", {}) if isinstance(heads, Mapping) else {}
            summary["candidates"][candidate]["nuisance_linear_regret_vs_probe"] = linear.get(
                "regret_vs_probe", {"status": "inconclusive", "reason": "missing paired linear-probe selector"}
            )
            summary["candidates"][candidate]["nuisance_linear_regret_vs_probe_upper"] = _lookup(
                summary["candidates"][candidate]["nuisance_linear_regret_vs_probe"], "upper"
            )
            nonlinear: Dict[str, Any] = {}
            for head in ("quadratic", "knn", "rbf"):
                data = heads.get(head, {}) if isinstance(heads, Mapping) else {}
                nonlinear[head] = data.get(
                    "regret_vs_B", {"status": "inconclusive", "reason": "missing paired nonlinear B comparison"}
                )
            summary["candidates"][candidate]["nonlinear_retention"] = nonlinear
        for candidate, values in summary["family_drift"].items():
            family_uppers = [_upper(values.get(scenario, {})) for scenario in SCENARIO_ORDER] if isinstance(values, Mapping) else []
            family_uppers = [value for value in family_uppers if value is not None]
            summary["candidates"].setdefault(candidate, {})["worst_family_drift_upper"] = (
                max(family_uppers) if len(family_uppers) == len(SCENARIO_ORDER) else None
            )
        for candidate in summary["candidates"]:
            genuine = summary["genuine_overlap"].get(candidate, {}) if isinstance(summary.get("genuine_overlap"), Mapping) else {}
            if isinstance(genuine, Mapping):
                value = genuine.get("worst_block_false_overlap_upper")
                summary["candidates"][candidate]["worst_block_false_overlap_upper"] = _number(value)
        eligibility_candidates = (
            (locked_candidate,) if stage_value == "confirmation" and locked_candidate is not None
            else analysis_promotables
        )
        summary["eligibility"] = candidate_eligibility(summary, candidates=eligibility_candidates)
    if promotion_decision is not None:
        summary["promotion_prerequisite"] = validate_decision_prerequisite(promotion_decision, protocol_hash=protocol_hash, code_identity_hash=code_identity_hash)
    return json_safe(summary)


build_analysis_summary = build_summary


_SMOKE_OUTCOME_FIELDS = frozenset({
    "candidate_score", "evaluation_score", "fit_score", "reference_accuracies",
    "reference_timings_seconds", "linear_probe", "linear_probe_score",
    "fit_wall_seconds", "fit_cpu_seconds", "score_fixed_wall_seconds",
    "score_fixed_cpu_seconds", "total_wall_seconds", "total_cpu_seconds",
})

_SMOKE_PAIR_STRUCTURAL_FIELDS = frozenset({
    "support", "hits", "sparse_adj_hits", "pairwise_index", "evidence",
})


def _smoke_structural_descriptor(value: Any) -> bool:
    """Whether a redacted pair value is only a type/finite descriptor."""

    if not isinstance(value, Mapping):
        return False
    # This is intentionally an exact schema.  A descriptor cannot carry a
    # value, shape, count, or array item from which a smoke outcome could be
    # reconstructed.
    return (
        set(str(key) for key in value) == {"present", "type", "finite"}
        and value.get("present") is True
        and value.get("type") in {"int", "float"}
        and value.get("finite") is True
    )


def _smoke_outcome_leaks(value: Any, *, key: Optional[str] = None) -> bool:
    """Recursively reject recoverable smoke outcomes and raw pair evidence."""

    key_name = str(key or "")
    if key_name in _SMOKE_OUTCOME_FIELDS:
        return True
    if key_name in _SMOKE_PAIR_STRUCTURAL_FIELDS and not _smoke_structural_descriptor(value):
        return True
    if isinstance(value, Mapping):
        return any(_smoke_outcome_leaks(item, key=str(name)) for name, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_smoke_outcome_leaks(item, key=key) for item in value)
    return False


def build_smoke_decision(
    artifact: Mapping[str, Any], *, protocol_hash: Optional[str] = None,
    code_identity_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a redacted smoke artifact without calculating any outcome."""

    manifest = artifact.get("manifest", {}) if isinstance(artifact, Mapping) else {}
    identity = validate_normalized_artifact(
        artifact, expected_stage="smoke", protocol_hash=protocol_hash,
        code_identity_hash=code_identity_hash,
    )
    failures = list(identity.get("failures", []))
    structural_evidence = _validated_structural_evidence(
        manifest if isinstance(manifest, Mapping) else {}, stage="smoke"
    )
    for name, value in structural_evidence.items():
        if not isinstance(value, Mapping) or value.get("status") != "pass":
            failures.append(f"smoke structural gate {name} is not pass")
    tables = artifact.get("tables", {}) if isinstance(artifact, Mapping) else {}
    if not isinstance(tables, Mapping):
        failures.append("smoke tables are missing")
        tables = {}
    for table_name, rows in tables.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if _smoke_outcome_leaks(row):
                failures.append(f"smoke outcome field leaked in {table_name}")
                break
            if str(_lookup(row, "status") or "").lower() not in {"ok", "pass", "completed"}:
                failures.append(f"smoke row status is not ok in {table_name}")
                break
    determinism = manifest.get("determinism", {}) if isinstance(manifest, Mapping) else {}
    if not isinstance(determinism, Mapping) or str(determinism.get("status", "")).lower() != "pass":
        failures.append("smoke deterministic-repeat check did not pass")
    # The pair state is structural in smoke.  Require only directional
    # identity, exact-state agreement, and non-reconstructable presence/type/
    # finite flags.  Numeric support, hits, adjacency counts, index, and
    # evidence are outcomes and must not be required or retained.
    pair_rows = tables.get("pair_rows.jsonl", ()) if isinstance(tables, Mapping) else ()
    for row in pair_rows if isinstance(pair_rows, Sequence) else ():
        for field in ("source_label", "target_label", "exact_state_match"):
            if field not in row:
                failures.append(f"smoke pair structural field missing: {field}")
                break
        if _lookup(row, "exact_state_match") is not True:
            failures.append("smoke pair structural state mismatch")
            break
        if _lookup(row, "source_label") == _lookup(row, "target_label"):
            failures.append("smoke pair structural row contains a diagonal direction")
            break
        structure = row.get("pair_structure")
        expected_structure_keys = set(_SMOKE_PAIR_STRUCTURAL_FIELDS)
        if not isinstance(structure, Mapping) or set(structure) != expected_structure_keys:
            failures.append("smoke pair_structure must contain exactly the five canonical fields")
            break
        if any(not _smoke_structural_descriptor(value) for value in structure.values()):
            failures.append("smoke pair structural flags contain a numeric outcome")
            break
        if "finite" in row:
            failures.append("smoke pair row exposes a top-level generic finite flag")
            break
        if row.get("outcomes_redacted") is not True:
            failures.append("smoke pair row lacks outcomes_redacted structural flag")
            break
        if any(field in row for field in _SMOKE_PAIR_STRUCTURAL_FIELDS):
            failures.append("smoke pair row exposes direct support/hit/evidence fields")
            break
    decision = {
        "schema_version": 1, "stage": "smoke",
        "status": "pass" if not failures else "fail",
        "structural_pass": not failures,
        "ranking_available": False,
        "outcomes_redacted_required": True,
        "failures": failures,
        "protocol_sha256": protocol_hash or manifest.get("protocol_sha256"),
        "code_identity_sha256": code_identity_hash or _lookup(manifest, "code_identity_sha256", "provenance.code_identity_sha256"),
        "input_manifest_sha256": artifact.get("manifest_path") and sha256_file(artifact["manifest_path"]) if isinstance(artifact, Mapping) and artifact.get("manifest_path") and Path(str(artifact["manifest_path"])).exists() else None,
        "input_table_hashes": {name: spec.get("sha256") for name, spec in artifact.get("table_checks", {}).items() if isinstance(spec, Mapping) and spec.get("sha256") is not None} if isinstance(artifact, Mapping) else {},
        # Carry the validated structural chain forward unchanged.  The
        # development runner consumes this exact object from the immutable
        # smoke decision; it must not infer parity/leakage from outcome rows.
        "structural_gates": structural_evidence,
    }
    return json_safe(decision)


def write_smoke_decision(
    input_path: os.PathLike[str] | str, output_path: os.PathLike[str] | str, *,
    protocol_hash: Optional[str] = None, code_identity_hash: Optional[str] = None,
) -> Dict[str, Any]:
    artifact = load_normalized_artifact(input_path, expected_stage="smoke", protocol_hash=protocol_hash, code_identity_hash=code_identity_hash)
    decision = build_smoke_decision(artifact, protocol_hash=protocol_hash, code_identity_hash=code_identity_hash)
    destination = Path(output_path)
    destination.mkdir(parents=True, exist_ok=True)
    write_immutable_json(destination / "smoke_decision.json", decision)
    return decision


def write_json(path: os.PathLike[str] | str, value: Any, *, immutable: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    if immutable and target.exists():
        raise FileExistsError(f"immutable decision artifact already exists: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            # A hard-link create is atomic and refuses an existing target,
            # unlike os.replace, so a concurrent rerun cannot overwrite a
            # decision after the preflight existence check.
            os.link(temporary, target)
            temporary.unlink()
        else:
            os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_immutable_json(path: os.PathLike[str] | str, value: Any) -> None:
    write_json(path, value, immutable=True)


def write_analysis(
    input_path: os.PathLike[str] | str, output_path: os.PathLike[str] | str, *,
    stage: Optional[str] = None, protocol_hash: Optional[str] = None, code_identity_hash: Optional[str] = None,
    promotion_decision: Optional[Mapping[str, Any] | os.PathLike[str] | str] = None,
    prior_regression_decision: Optional[Mapping[str, Any] | os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    artifact = load_normalized_artifact(
        input_path, expected_stage=stage, protocol_hash=protocol_hash,
        code_identity_hash=code_identity_hash,
    )
    manifest = artifact.get("manifest", {}) if isinstance(artifact, Mapping) else {}
    stage_value = str(stage or manifest.get("stage", "development")).lower()
    if stage_value not in {"smoke", "development", "confirmation"}:
        raise ValueError("stage must be one of {'smoke', 'development', 'confirmation'}")
    # Artifact manifests are the source of the resolved identities when the
    # CLI caller does not repeat them.  Decisions always carry concrete
    # strings; a missing manifest identity is left missing and therefore
    # cannot satisfy a later prerequisite validation.
    resolved_protocol = protocol_hash or manifest.get("protocol_sha256", manifest.get("protocol_hash"))
    resolved_code = code_identity_hash or manifest.get("code_identity_sha256", manifest.get("code_identity_hash"))
    if resolved_code is None and isinstance(manifest.get("provenance"), Mapping):
        resolved_code = manifest["provenance"].get("code_identity_sha256", manifest["provenance"].get("code_identity_hash"))

    destination = Path(output_path)
    destination.mkdir(parents=True, exist_ok=True)
    decision_name = {
        "smoke": "smoke_decision.json",
        "development": "promotion_decision.json",
        "confirmation": "confirmation_decision.json",
    }[stage_value]
    decision_path = destination / decision_name
    # Decision artifacts are immutable records.  Preflight before writing the
    # summary/report so a rerun cannot partially replace a locked result.
    if decision_path.exists():
        raise FileExistsError(f"immutable decision artifact already exists: {decision_path}")

    if stage_value == "smoke":
        if promotion_decision is not None or prior_regression_decision is not None:
            raise ValueError("smoke analysis does not consume promotion/prior decisions")
        summary = build_summary(
            artifact, stage="smoke", protocol_hash=resolved_protocol,
            code_identity_hash=resolved_code,
        )
        decision = build_smoke_decision(
            artifact, protocol_hash=resolved_protocol, code_identity_hash=resolved_code,
        )
    elif stage_value == "development":
        if prior_regression_decision is not None:
            raise ValueError("development analysis does not consume a prior-regression decision")
        summary = build_summary(
            artifact, stage="development", promotion_decision=promotion_decision,
            protocol_hash=resolved_protocol, code_identity_hash=resolved_code,
        )
        decision = select_lock(
            summary, protocol_hash=resolved_protocol, code_identity_hash=resolved_code,
            existing_decision=promotion_decision,
        )
    else:
        if promotion_decision is None or prior_regression_decision is None:
            raise PermissionError(
                "confirmation analysis requires promotion_decision and prior_regression_decision"
            )
        promotion_check = validate_decision_prerequisite(
            promotion_decision, protocol_hash=resolved_protocol,
            code_identity_hash=resolved_code, expected_status=("locked", "pass"),
        )
        if promotion_check["status"] != "pass":
            raise PermissionError("promotion prerequisite failed identity validation")
        locked_candidate = candidate_id(
            promotion_check["payload"].get(
                "locked_candidate", promotion_check["payload"].get("selected_candidate")
            )
        )
        if locked_candidate is None:
            raise PermissionError("promotion prerequisite has no locked candidate")
        summary = build_summary(
            artifact, stage="confirmation", protocol_hash=resolved_protocol,
            code_identity_hash=resolved_code,
        )
        decision = evaluate_confirmation(
            summary, locked_candidate=locked_candidate,
            promotion_decision=promotion_decision,
            prior_regression_decision=prior_regression_decision,
            protocol_hash=resolved_protocol, code_identity_hash=resolved_code,
        )
        decision["promotion_decision_sha256"] = promotion_check["sha256"]
        decision["prior_regression_decision_sha256"] = validate_decision_prerequisite(
            prior_regression_decision, protocol_hash=resolved_protocol,
            code_identity_hash=resolved_code, expected_status=("pass", "passed"),
            locked_candidate=locked_candidate, promotion_hash=promotion_check["sha256"],
        )["sha256"]

    # Render all mutable presentation surfaces before committing the immutable
    # decision.  A report failure must not strand a final lock/confirmation
    # artifact that falsely appears complete.
    from .reporting import write_reports
    write_json(destination / "analysis_summary.json", summary)
    write_reports(destination, summary, decision)
    write_immutable_json(decision_path, decision)
    return {
        "summary": summary,
        "decision": decision,
        # This is the exact canonical file identity that downstream stages
        # should record when the decision is written.  It is returned outside
        # the decision payload so no impossible self-referential hash field is
        # embedded in the immutable JSON.
        "decision_sha256": _decision_hash(decision),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("smoke", "development", "confirmation"), default=None)
    parser.add_argument("--protocol-sha256", default=None)
    parser.add_argument("--code-identity-sha256", default=None)
    parser.add_argument("--promotion-decision", default=None)
    parser.add_argument("--prior-regression-decision", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    write_analysis(args.input, args.output, stage=args.stage, protocol_hash=args.protocol_sha256, code_identity_hash=args.code_identity_sha256, promotion_decision=args.promotion_decision, prior_regression_decision=args.prior_regression_decision)
    return 0


__all__ = [
    "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED", "BONFERRONI_LEVEL", "CANDIDATE_ORDER", "PROMOTABLE_CANDIDATES",
    "archived_pooled_auprc", "binary_detection_metrics", "build_analysis_summary", "build_summary", "candidate_eligibility",
    "canonical_json", "classify_gate", "complete_seed_bootstrap", "complete_seed_blocks", "evaluate_confirmation",
    "family_drift", "flatten_rows", "genuine_overlap_gates", "joint_bootstrap_draws", "json_safe", "load_normalized_artifact",
    "negative_controls", "normalized_trapezoid_auc", "pair_detection_summary", "paired_seed_difference", "percentile_interval",
    "pooled_pair_detection", "primary_mechanistic_claims", "resource_summaries", "select_lock", "select_promotion",
    "selector_panel_metrics", "sha256_bytes", "sha256_file", "spearman_rank", "stable_auprc", "stable_shift_gates",
    "tie_aware_auroc", "validate_decision_prerequisite", "validate_manifest_identity", "validate_normalized_artifact",
    "validate_table_rows", "within_draw_max", "write_analysis", "write_json", "write_immutable_json",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
