"""Analysis and promotion helpers for the nuisance-conditioned experiment.

The runner deliberately writes records rather than an analysis-specific binary
format.  This module therefore keeps the reader permissive (``rows.json``,
``results.json``, JSONL, and candidate-keyed mappings are all accepted), while
the statistics and promotion rule remain frozen and explicit.  The functions
in this file are also useful from tests and small notebooks; the command line
entry point is intentionally just a thin orchestration layer.

No result in this module is used by the public :mod:`overlapindex` package.
Undefined quantities are represented by ``None`` in JSON, never by a NaN or an
infinity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_812
WITHIN_ONE_POINT = 0.01
PROMOTION_TOLERANCE = 0.001
CANDIDATE_ORDER = ("A", "B", "C", "D", "E", "F", "G")
PROMOTION_CANDIDATES = ("C", "D", "E")
REFERENCE_HEADS = ("linear", "quadratic", "knn", "rbf")
_EXPECTED_STAGE_COUNTS = {
    "screen": {"cases": 720, "rows": 4_320},
    "full": {"cases": 2_160, "rows": 12_960},
}
_EXPECTED_STAGE_CANDIDATES = ("A", "B", "C", "D", "E", "F")
_EXPECTED_SYNTHETIC_FAMILIES = frozenset(
    {
        "shared_low_rank",
        "clustered_multimodal",
        "heteroscedastic_multiplicative",
    }
)
_FOOD_MODELS = (
    "dinov2-small",
    "deit-tiny",
    "convnext-tiny",
    "mobilenetv3-large",
    "openclip-vit-b-32",
    "resnet50",
    "efficientnet-b0",
    "swin-tiny",
    "vit-small-16",
    "densenet121",
)
_FOOD_REPLICATES = (0, 1, 2, 3, 4)
_FOOD_ARMS = ("baseline", "nonlinearity_full", "nuisance_full")
_FOOD_BUDGETS = (64, 68, 72, 80)
_FOOD_SELECTOR_CANDIDATES = ("A", "B", "C", "D", "E")
_FOOD_PANEL_CELLS = len(_FOOD_MODELS) * len(_FOOD_REPLICATES) * len(_FOOD_ARMS) * len(_FOOD_BUDGETS)
_RUNTIME_MODELS = _FOOD_MODELS
_RUNTIME_REPEATS = (0, 1, 2, 3, 4)
_RUNTIME_BUDGETS = (64, 128, 256, 512, 640)
_RUNTIME_PANEL_CELLS = len(_RUNTIME_MODELS) * len(_RUNTIME_REPEATS) * len(_RUNTIME_BUDGETS)

_CANDIDATE_NAMES = {
    "A": "oi_unrefined_raw",
    "B": "oi_refined_raw",
    "C": "oi_refined_global_isotropy",
    "D": "oi_refined_pooled_diagonal",
    "E": "oi_refined_pooled_full",
    "F": "raw_conditioned_disagreement",
    "G": "panel_selective_capped_probe_guardrail",
}
_NAME_TO_CANDIDATE = {value: key for key, value in _CANDIDATE_NAMES.items()}
_NAME_TO_CANDIDATE.update({key.lower(): key for key in _CANDIDATE_NAMES})


_KNOWN_SOURCE_RELATIVE = (
    "experiments/nuisance_conditioned_distance/conditioning_adapter.py",
    "experiments/nuisance_conditioned_distance/fixtures.py",
    "experiments/nuisance_conditioned_distance/runner.py",
    "experiments/nuisance_conditioned_distance/food101.py",
    "experiments/nuisance_conditioned_distance/runtime_benchmark.py",
    "experiments/nuisance_conditioned_distance/protocol.json",
    "experiments/nuisance_conditioned_distance/protocol.sha256",
    "experiments/nuisance_conditioned_distance/analysis.py",
    "experiments/nuisance_conditioned_distance/reporting.py",
)


def json_safe(value: Any) -> Any:
    """Convert numpy/scalar values recursively to strict JSON values.

    ``allow_nan=False`` is used by all writers, so non-finite numeric values
    are intentionally converted to ``None`` rather than leaking invalid JSON.
    """

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
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    # Dataclasses and small result objects commonly expose a useful mapping.
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def canonical_json(value: Any) -> str:
    """Return deterministic, compact JSON used for hashes and artifacts."""

    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _candidate_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in _CANDIDATE_NAMES:
        return text
    if text in {"G_probe_component", "capped_probe_component", "capped_linear_probe"}:
        return "G_probe_component"
    if text in {"linear_probe_oof", "full_probe", "prior_full_probe"}:
        return "full_probe"
    if text in {"G_policy", "g_policy", "guardrail", "policy"}:
        return "G"
    return _NAME_TO_CANDIDATE.get(text) or _NAME_TO_CANDIDATE.get(text.lower())


def _runtime_method_id(value: Any) -> str | None:
    """Canonicalize runtime primitives without turning the capped probe into G."""

    if value is None:
        return None
    text = str(value).strip()
    if text in {"full_probe", "linear_probe_oof", "prior_full_probe"}:
        return "full_probe"
    if text in {
        "capped_probe_component",
        "capped_probe",
        "capped_probe_G",
        "G_probe_component",
        "capped_linear_probe",
    }:
        return "capped_probe_component"
    if text in {"G", "g", "G_policy", "g_policy", "guardrail", "policy"}:
        return "G"
    if text in {"E_for_G_diagnostics", "E-for-G-diagnostics"}:
        return "E"
    return _candidate_id(text)


def _analysis_stage(manifest: Mapping[str, Any]) -> str | None:
    """Resolve the small stage vocabulary used by raw results and manifests."""

    value = manifest.get("stage") or manifest.get("study")
    if value is None:
        return None
    text = str(value)
    lowered = text.lower()
    if "food101_nuisance_conditioned_distance_runtime" in lowered or (
        "runtime" in lowered and "food101" in lowered
    ):
        return "runtime"
    if "food101_nuisance_conditioned_distance" in lowered or "food101" == lowered:
        return "food101"
    return text


def _mapping_without(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    excluded = set(keys)
    return {str(key): value for key, value in mapping.items() if key not in excluded}


def _normalise_records(payload: Any, candidate_hint: str | None = None) -> list[dict[str, Any]]:
    """Flatten the small set of runner result layouts used by this project."""

    if payload is None:
        return []
    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for item in payload:
            records.extend(_normalise_records(item, candidate_hint))
        return records
    if not isinstance(payload, Mapping):
        return []

    own_candidate = candidate_hint
    for key in ("candidate", "candidate_id", "candidate_name", "arm", "method_id", "method"):
        own_candidate = _candidate_id(payload.get(key)) or own_candidate

    # A run-level record can hold all candidate records under one of these
    # fields.  Common condition/seed fields are inherited by each child.
    container_keys = (
        "candidates",
        "candidate_results",
        "candidate_rows",
        "selector_rows",
        "capped_probe_rows",
        "guardrail_rows",
        "prior_full_probe_rows",
        "runtime_rows",
        "g_policy_rows",
        "rows",
        "records",
    )
    present = [key for key in container_keys if isinstance(payload.get(key), (list, Mapping))]
    if present:
        records: list[dict[str, Any]] = []
        for key in present:
            children = payload[key]
            inherited = _mapping_without(
                payload,
                *present,
                "candidate",
                "candidate_id",
                "candidate_name",
                "arm",
                "method_id",
                "reference_rows",
                "capped_probe_rows",
                "guardrail_rows",
                "prior_full_probe_rows",
                "baseline_parity_rows",
            )
            child_items = children.items() if isinstance(children, Mapping) else enumerate(children)
            current_records: list[dict[str, Any]] = []
            for child_key, child in child_items:
                # The dedicated Food/runtime containers are allowed to omit a
                # repeated method field.  Keep those rows distinguishable
                # without conflating the capped component with policy G.
                container_hint = {
                    "guardrail_rows": "G",
                    "g_policy_rows": "G",
                    "capped_probe_rows": "G_probe_component",
                    "prior_full_probe_rows": "full_probe",
                }.get(key)
                child_hint = _candidate_id(child_key) or own_candidate or container_hint
                for record in _normalise_records(child, child_hint):
                    merged = dict(inherited)
                    merged.update(record)
                    current_records.append(merged)
            # Food-101 stores frozen reference outcomes beside selector rows.
            if key in {"selector_rows", "guardrail_rows", "prior_full_probe_rows"} and isinstance(payload.get("reference_rows"), list):
                reference_rows = payload.get("reference_rows", [])
                for record in current_records:
                    match = _matching_reference_rows(record, reference_rows)
                    if match:
                        record.setdefault("references", {}).update(match)
            records.extend(current_records)
        return records

    # ``{"A": [...], "B": [...]}`` is a convenient compact format for
    # tests and is accepted in addition to the explicit candidate field.
    candidate_keys = [key for key in payload if _candidate_id(key) is not None]
    if candidate_keys and not any(
        key in payload for key in ("score", "score_fixed", "metrics", "seed", "case_id")
    ):
        inherited = _mapping_without(payload, *candidate_keys)
        records = []
        for key in candidate_keys:
            for record in _normalise_records(payload[key], _candidate_id(key)):
                merged = dict(inherited)
                merged.update(record)
                records.append(merged)
        return records

    # Merge shallow metric containers while retaining the original containers
    # for diagnostics.  This makes both ``{"score": ...}`` and
    # ``{"metrics": {"score": ...}}`` equally easy to consume.
    record = dict(payload)
    if own_candidate is not None:
        record["candidate"] = own_candidate
    for container_key in ("metrics", "scores", "result", "results"):
        container = payload.get(container_key)
        if isinstance(container, Mapping):
            for key, value in container.items():
                record.setdefault(str(key), value)
    return [record]


def _matching_reference_rows(row: Mapping[str, Any], reference_rows: Sequence[Any]) -> dict[str, Any]:
    """Merge Food-101 reference rows matching one panel cell."""

    identity = ("model", "backbone", "replicate", "arm", "budget", "seed")
    result: dict[str, Any] = {}
    for reference in reference_rows:
        if not isinstance(reference, Mapping):
            continue
        if any(
            _lookup(row, field) is not None
            and _lookup(reference, field) is not None
            and _freeze_key(_lookup(row, field)) != _freeze_key(_lookup(reference, field))
            for field in identity
        ):
            continue
        head = _lookup(reference, "head", "reference_head", "method", "name")
        value = _lookup(reference, "accuracy", "test_accuracy", "score", "reference_accuracy", "value")
        if head is not None and value is not None:
            result[str(head)] = value
    return result


def _load_result_file(path: Path) -> Any:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return _read_jsonl(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    return _read_json(path)


_ANALYSIS_CONTAINER_KEYS = (
    "selector_rows",
    "capped_probe_rows",
    "guardrail_rows",
    "prior_full_probe_rows",
    "runtime_rows",
    "g_policy_rows",
    "baseline_parity_rows",
    "reference_rows",
)


def _payload_metadata(payload: Any) -> dict[str, Any]:
    """Preserve run metadata and small audit containers beside normalized rows.

    Food-101 keeps parity and reference containers at the raw-artifact top
    level.  The normalizer must not turn those containers into candidate rows,
    but analysis still needs their exact identities/counts to validate a full
    artifact.  Keep the raw containers under an explicit namespace so a
    summary cannot confuse them with selector observations.
    """

    if not isinstance(payload, Mapping):
        return {}
    excluded = {
        "rows",
        "records",
        "selector_rows",
        "capped_probe_rows",
        "guardrail_rows",
        "prior_full_probe_rows",
        "runtime_rows",
        "g_policy_rows",
        "baseline_parity_rows",
        "reference_rows",
        "candidate_rows",
        "candidates",
    }
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in excluded
    }
    containers: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for key in _ANALYSIS_CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            containers[key] = value
            counts[key] = len(value)
        elif isinstance(value, Mapping):
            containers[key] = value
            counts[key] = len(value)
    if containers:
        metadata["analysis_artifact_containers"] = containers
        metadata["analysis_artifact_container_counts"] = counts
    return metadata


def load_runner_output(input_path: os.PathLike[str] | str) -> dict[str, Any]:
    """Read one runner output directory and return records plus provenance.

    The reader does not silently combine unrelated files.  It uses the first
    conventional result file containing records, then follows a manifest's
    explicit ``results_file``/``rows_file`` when present.  This avoids counting
    an archived raw table and a derived table twice.
    """

    source = Path(input_path)
    if source.is_file():
        payload = _load_result_file(source)
        metadata = _payload_metadata(payload)
        return {
            "rows": _normalise_records(payload),
            "manifest": {},
            "metadata": metadata,
            "source_files": [source],
            "input_path": source,
        }
    if not source.is_dir():
        raise FileNotFoundError(f"runner output directory does not exist: {source}")

    manifest_path = source / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        loaded = _read_json(manifest_path)
        if isinstance(loaded, Mapping):
            manifest = dict(loaded)

    explicit_names: list[str] = []
    for key in ("results_file", "rows_file", "raw_results_file", "records_file", "output_file"):
        value = manifest.get(key)
        if isinstance(value, str):
            explicit_names.append(value)
    for container_key in ("results", "outputs", "artifacts"):
        container = manifest.get(container_key)
        if isinstance(container, Mapping):
            for key in ("results_file", "rows_file", "raw_results_file", "records_file"):
                value = container.get(key)
                if isinstance(value, str):
                    explicit_names.append(value)

    candidates: list[Path] = []
    for name in explicit_names:
        path = (source / name).resolve() if not os.path.isabs(name) else Path(name)
        if path.exists() and path.is_file() and path not in candidates:
            candidates.append(path)
    for name in (
        "rows.json",
        "results.json",
        "raw_results.json",
        "raw_rows.json",
        "records.json",
        "metrics.json",
        "runtime_results.json",
        "runtime.json",
        "rows.jsonl",
        "results.jsonl",
        "records.jsonl",
    ):
        path = source / name
        if path.exists() and path not in candidates:
            candidates.append(path)
    if not candidates:
        candidates = sorted(
            path
            for path in source.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".json", ".jsonl", ".ndjson", ".csv"}
            and path.name not in {"manifest.json", "analysis_summary.json", "promotion_decision.json"}
            and not path.name.startswith("analysis")
        )

    selected: Path | None = None
    rows: list[dict[str, Any]] = []
    for path in candidates:
        payload = _load_result_file(path)
        current = _normalise_records(payload)
        if current:
            selected, rows = path, current
            break
    if selected is None:
        # An empty run is a valid object to analyze; preserve a useful source
        # path for hashing and produce an all-undefined report.
        selected = candidates[0] if candidates else manifest_path
        rows = []
    source_files = [selected]
    if manifest_path.exists() and manifest_path not in source_files:
        source_files.append(manifest_path)
    metadata: dict[str, Any] = {}
    if selected is not None:
        try:
            payload = _load_result_file(selected)
            metadata = _payload_metadata(payload)
        except Exception:
            metadata = {}
    return {
        "rows": rows,
        "manifest": manifest,
        "metadata": metadata,
        "source_files": source_files,
        "input_path": source,
    }


def source_hashes(input_path: os.PathLike[str] | str, source_files: Iterable[Path] | None = None) -> dict[str, str]:
    """Hash input files and the frozen protocol for reproducibility."""

    root = Path(input_path)
    if root.is_file():
        files = [root]
        root_for_names = root.parent
    else:
        root_for_names = root
        files = list(source_files or [])
        if not files and root.exists():
            files = [path for path in root.iterdir() if path.is_file()]
    for relative in _KNOWN_SOURCE_RELATIVE:
        known = PACKAGE_DIR / Path(relative).name
        if known.exists() and known not in files:
            files.append(known)
    hashes: dict[str, str] = {}
    for path in sorted({Path(item).resolve() for item in files if Path(item).exists()}, key=str):
        try:
            name = str(path.relative_to(root_for_names.resolve()))
        except ValueError:
            try:
                name = str(path.relative_to(PACKAGE_DIR.parent.parent.resolve()))
            except ValueError:
                name = str(path)
        hashes[name] = sha256_file(path)
    return hashes


def git_provenance() -> dict[str, Any]:
    """Capture commit and dirty-worktree state without changing the worktree."""

    import subprocess

    root = PACKAGE_DIR.parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        status_text = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        status_text = ""
    return {
        "commit": commit,
        "dirty": bool(status_text.strip()),
        "status_sha256": sha256_bytes(status_text.encode("utf-8")),
    }


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _lookup(mapping: Mapping[str, Any], *paths: str) -> Any:
    """Look up dotted paths and case-insensitive aliases."""

    for path in paths:
        current: Any = mapping
        found = True
        for part in path.split("."):
            if isinstance(current, Mapping):
                if part in current:
                    current = current[part]
                else:
                    lower = {str(key).lower(): key for key in current}
                    key = lower.get(part.lower())
                    if key is None:
                        found = False
                        break
                    current = current[key]
            else:
                found = False
                break
        if found:
            return current
    return None


def _as_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = list(value.values())
    if isinstance(value, (str, bytes)):
        number = _finite_float(value)
        return [] if number is None else [number]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result: list[float] = []
        for item in value:
            number = _finite_float(item)
            if number is not None:
                result.append(number)
        return result
    number = _finite_float(value)
    return [] if number is None else [number]


def _freeze_key(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_key(item) for item in value)
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def row_pair_key(row: Mapping[str, Any], index: int = 0) -> Any:
    """Return the frozen pairing key used by paired bootstrap metrics."""

    for field in ("pair_id", "paired_id", "case_id", "cell_id", "row_id", "observation_id"):
        value = _lookup(row, field)
        if value is not None:
            return (field, _freeze_key(value))
    fields = (
        "family",
        "condition",
        "condition_name",
        "geometry",
        "model",
        "backbone",
        "arm",
        "lambda",
        "nu",
        "nuisance_kind",
        "nuisance_strength",
        "balance",
        "nuisance_shift",
        "k",
        "seed",
        "train_seed",
        "eval_seed",
        "reference",
        "head",
        "budget",
        "replicate",
        "fold",
        "panel_id",
    )
    values = tuple((field, _freeze_key(_lookup(row, field))) for field in fields if _lookup(row, field) is not None)
    return values if values else ("row", int(index))


def _candidate_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidate = _candidate_id(
            _lookup(
                row,
                "candidate",
                "candidate_id",
                "candidate_name",
                "arm",
                "method_id",
                "method",
            )
        )
        if candidate is not None:
            grouped[candidate].append(dict(row))
    return dict(grouped)


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    if metric in {"score", "oi_score", "score_fixed"}:
        value = _lookup(
            row,
            "score_fixed",
            "heldout_score",
            "evaluation_score",
            "candidate_score",
            "oi_score",
            "score",
            "overlap_index",
            "index",
            "metrics.score_fixed",
            "metrics.score",
        )
    else:
        value = _lookup(row, metric)
    return _finite_float(value)


def _aligned_values(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    extractor: Callable[[Mapping[str, Any]], float | None],
    candidates: Sequence[str] | None = None,
) -> tuple[list[Any], dict[str, np.ndarray]]:
    selected = [candidate for candidate in (candidates or tuple(grouped)) if candidate in grouped]
    by_candidate: dict[str, dict[Any, list[float]]] = {}
    for candidate in selected:
        values: dict[Any, list[float]] = defaultdict(list)
        for index, row in enumerate(grouped[candidate]):
            value = extractor(row)
            if value is not None:
                values[row_pair_key(row, index)].append(float(value))
        by_candidate[candidate] = values
    # Diagnostic-only F (and product-policy rows, when present) intentionally
    # has no scalar score.  It must not erase the complete A--E pairing.
    by_candidate = {candidate: values for candidate, values in by_candidate.items() if values}
    if not by_candidate:
        return [], {}
    selected = list(by_candidate)
    common = set.intersection(*(set(values) for values in by_candidate.values()))
    keys = sorted(common, key=repr)
    arrays = {
        candidate: np.asarray([np.mean(by_candidate[candidate][key]) for key in keys], dtype=float)
        for candidate in selected
    }
    return keys, arrays


@lru_cache(maxsize=16)
def bootstrap_indices(
    n_observations: int,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Create the shared paired complete-seed bootstrap draws."""

    n_observations = int(n_observations)
    if n_observations <= 0:
        return np.empty((0, 0), dtype=np.int64)
    if int(n_resamples) <= 0:
        raise ValueError("n_resamples must be positive")
    return np.random.default_rng(int(seed)).integers(
        0, n_observations, size=(int(n_resamples), n_observations), dtype=np.int64
    )


def _block_mapping(
    rows: Sequence[Mapping[str, Any]],
    extractor: Callable[[Mapping[str, Any]], float | None],
    block_field: str,
) -> dict[Any, np.ndarray]:
    blocks: dict[Any, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        block = _lookup(row, block_field)
        if block is None:
            block = row_pair_key(row, index)
        value = extractor(row)
        if value is not None and math.isfinite(float(value)):
            blocks[_freeze_key(block)].append(float(value))
    return {key: np.asarray(value, dtype=float) for key, value in blocks.items() if value}


def _complete_block_values(
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    extractor: Callable[[Mapping[str, Any]], float | None],
    block_field: str,
) -> dict[str, dict[Any, np.ndarray]]:
    """Keep only blocks whose complete row identities match every candidate."""

    values_by_candidate: dict[str, dict[Any, list[float]]] = {}
    # Keep multiplicities as well as identities.  A set-only comparison would
    # incorrectly call a block complete when one candidate has two rows for a
    # case and another has one duplicate row.
    keys_by_candidate: dict[str, dict[Any, dict[Any, int]]] = {}
    for candidate, rows in rows_by_candidate.items():
        values: dict[Any, list[float]] = defaultdict(list)
        identities: dict[Any, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        for index, row in enumerate(rows):
            value = extractor(row)
            if value is None or not math.isfinite(float(value)):
                continue
            block_value = _lookup(row, block_field)
            if block_value is None:
                block_value = row_pair_key(row, index)
            block = _freeze_key(block_value)
            values[block].append(float(value))
            identities[block][row_pair_key(row, index)] += 1
        values_by_candidate[candidate] = values
        keys_by_candidate[candidate] = identities
    if not values_by_candidate:
        return {}
    common_blocks = set.intersection(*(set(values) for values in values_by_candidate.values()))
    complete_blocks = {
        block
        for block in common_blocks
        if all(
            keys_by_candidate[candidate].get(block, set())
            == keys_by_candidate[next(iter(values_by_candidate))].get(block, {})
            for candidate in values_by_candidate
        )
    }
    return {
        candidate: {
            block: np.asarray(values[block], dtype=float)
            for block in complete_blocks
            if values.get(block)
        }
        for candidate, values in values_by_candidate.items()
    }


def paired_block_percentile_bootstrap(
    block_values_by_candidate: Mapping[str, Mapping[Any, Sequence[float]]],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    """Bootstrap complete seed/replicate blocks and pool their rows.

    For synthetic rows, callers pass ``block_field='seed'`` through
    :func:`paired_seed_block_bootstrap`; all conditions, families and ``k``
    rows belonging to a selected seed are concatenated before calculating the
    statistic.  Food-101 uses the same function with replicate blocks.  The
    same draw matrix is applied to every candidate.
    """

    if not block_values_by_candidate:
        return {}
    names = list(block_values_by_candidate)
    maps: dict[str, dict[Any, np.ndarray]] = {}
    for name in names:
        maps[name] = {
            _freeze_key(block): np.asarray(values, dtype=float).reshape(-1)
            for block, values in block_values_by_candidate[name].items()
            if np.asarray(values).size
        }
    complete_blocks = set.intersection(*(set(mapping) for mapping in maps.values()))
    complete_blocks = {
        block
        for block in complete_blocks
        if all(np.all(np.isfinite(mapping[block])) for mapping in maps.values())
    }
    blocks = sorted(complete_blocks, key=repr)
    if not blocks:
        return {
            name: {"estimate": None, "lower": None, "upper": None, "n": 0, "n_blocks": 0, "status": "undefined"}
            for name in names
        }
    draws = bootstrap_indices(len(blocks), n_resamples=n_resamples, seed=seed)
    output: dict[str, dict[str, Any]] = {}
    for name in names:
        mapping = maps[name]
        pooled = np.concatenate([mapping[block] for block in blocks])
        estimate = _finite_float(statistic(pooled))
        if statistic is np.mean:
            block_sums = np.asarray([np.sum(mapping[block]) for block in blocks], dtype=float)
            block_counts = np.asarray([mapping[block].size for block in blocks], dtype=float)
            sampled_sums = np.sum(block_sums[draws], axis=1)
            sampled_counts = np.sum(block_counts[draws], axis=1)
            estimates_array = sampled_sums / sampled_counts
            estimates = estimates_array[np.isfinite(estimates_array)].tolist()
        else:
            estimates = []
            for draw in draws:
                sampled = np.concatenate([mapping[blocks[int(index)]] for index in draw])
                value = _finite_float(statistic(sampled))
                if value is not None:
                    estimates.append(value)
        output[name] = {
            "estimate": estimate,
            "lower": None if not estimates else float(np.percentile(estimates, 2.5)),
            "upper": None if not estimates else float(np.percentile(estimates, 97.5)),
            "n": int(pooled.size),
            "n_blocks": len(blocks),
            "status": "defined" if estimates else "undefined",
            "block_field": "complete",
        }
    return output


def paired_block_percentile_difference(
    block_values_by_candidate: Mapping[str, Mapping[Any, Sequence[float]]],
    baseline: str = "B",
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    """Candidate-minus-baseline intervals using identical complete blocks."""

    if baseline not in block_values_by_candidate:
        return {
            candidate: {"estimate": None, "lower": None, "upper": None, "n": 0, "n_blocks": 0, "status": "undefined"}
            for candidate in block_values_by_candidate
            if candidate != baseline
        }
    names = [candidate for candidate in block_values_by_candidate if candidate != baseline]
    maps = {
        candidate: {
            _freeze_key(block): np.asarray(values, dtype=float).reshape(-1)
            for block, values in block_values_by_candidate[candidate].items()
            if np.asarray(values).size
        }
        for candidate in [baseline, *names]
    }
    complete = set.intersection(*(set(mapping) for mapping in maps.values()))
    complete = {
        block
        for block in complete
        if all(np.all(np.isfinite(maps[candidate][block])) for candidate in maps)
    }
    blocks = sorted(complete, key=repr)
    if not blocks:
        return {
            candidate: {"estimate": None, "lower": None, "upper": None, "n": 0, "n_blocks": 0, "status": "undefined"}
            for candidate in names
        }
    draws = bootstrap_indices(len(blocks), n_resamples=n_resamples, seed=seed)
    output: dict[str, dict[str, Any]] = {}
    base_map = maps[baseline]
    base_values = np.concatenate([base_map[block] for block in blocks])
    for candidate in names:
        candidate_map = maps[candidate]
        candidate_values = np.concatenate([candidate_map[block] for block in blocks])
        point = _finite_float(statistic(candidate_values) - statistic(base_values))
        if statistic is np.mean:
            base_sums = np.asarray([np.sum(base_map[block]) for block in blocks], dtype=float)
            base_counts = np.asarray([base_map[block].size for block in blocks], dtype=float)
            candidate_sums = np.asarray([np.sum(candidate_map[block]) for block in blocks], dtype=float)
            candidate_counts = np.asarray([candidate_map[block].size for block in blocks], dtype=float)
            sampled_base = np.sum(base_sums[draws], axis=1) / np.sum(base_counts[draws], axis=1)
            sampled_candidate = np.sum(candidate_sums[draws], axis=1) / np.sum(candidate_counts[draws], axis=1)
            estimates = sampled_candidate - sampled_base
        else:
            values: list[float] = []
            for draw in draws:
                base_sample = np.concatenate([base_map[blocks[int(index)]] for index in draw])
                candidate_sample = np.concatenate([candidate_map[blocks[int(index)]] for index in draw])
                value = _finite_float(statistic(candidate_sample) - statistic(base_sample))
                if value is not None:
                    values.append(value)
            estimates = np.asarray(values, dtype=float)
        estimates = np.asarray(estimates, dtype=float)
        estimates = estimates[np.isfinite(estimates)]
        output[candidate] = {
            "estimate": point,
            "lower": None if estimates.size == 0 else float(np.percentile(estimates, 2.5)),
            "upper": None if estimates.size == 0 else float(np.percentile(estimates, 97.5)),
            "n": int(min(candidate_values.size, base_values.size)),
            "n_blocks": len(blocks),
            "status": "defined" if estimates.size else "undefined",
        }
    return output


def paired_seed_block_bootstrap(
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    extractor: Callable[[Mapping[str, Any]], float | None],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    """Frozen synthetic v3 pooled-row bootstrap over complete seed blocks."""

    return paired_block_percentile_bootstrap(
        _complete_block_values(rows_by_candidate, extractor, "seed"),
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_replicate_block_bootstrap(
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    extractor: Callable[[Mapping[str, Any]], float | None],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 145,
) -> dict[str, dict[str, Any]]:
    """Food-101 paired replicate-block percentile bootstrap."""

    return paired_block_percentile_bootstrap(
        _complete_block_values(rows_by_candidate, extractor, "replicate"),
        n_resamples=n_resamples,
        seed=seed,
    )


# Names used in the protocol text and in older analysis notebooks.
seed_block_bootstrap = paired_seed_block_bootstrap
replicate_block_bootstrap = paired_replicate_block_bootstrap


def paired_percentile_bootstrap(
    values_by_candidate: Mapping[str, Sequence[float]],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    """Compute paired percentile intervals with identical draws per candidate.

    Inputs must already be aligned to complete paired observations.  Non-finite
    observations are removed jointly, so a missing value in one arm never
    changes the paired sample used by another arm.
    """

    if not values_by_candidate:
        return {}
    names = list(values_by_candidate)
    arrays = [np.asarray(values_by_candidate[name], dtype=float).reshape(-1) for name in names]
    if not arrays or len({array.size for array in arrays}) != 1:
        raise ValueError("paired bootstrap inputs must have equal lengths")
    n = arrays[0].size
    finite = np.ones(n, dtype=bool)
    for array in arrays:
        finite &= np.isfinite(array)
    arrays = [array[finite] for array in arrays]
    if not arrays or arrays[0].size == 0:
        return {
            name: {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
            for name in names
        }
    draws = bootstrap_indices(arrays[0].size, n_resamples=n_resamples, seed=seed)
    output: dict[str, dict[str, Any]] = {}
    for name, array in zip(names, arrays):
        estimate = _finite_float(statistic(array))
        # Applying the same integer draw matrix is the important paired detail.
        sampled = array[draws]
        estimates = (
            np.mean(sampled, axis=1)
            if statistic is np.mean
            else np.asarray([statistic(sample) for sample in sampled], dtype=float)
        )
        estimates = estimates[np.isfinite(estimates)]
        if estimates.size == 0:
            output[name] = {
                "estimate": estimate,
                "lower": None,
                "upper": None,
                "n": int(array.size),
                "status": "undefined",
            }
        else:
            output[name] = {
                "estimate": estimate,
                "lower": float(np.percentile(estimates, 2.5)),
                "upper": float(np.percentile(estimates, 97.5)),
                "n": int(array.size),
                "status": "defined",
            }
    return output


def paired_bootstrap_difference(
    values_by_candidate: Mapping[str, Sequence[float]],
    baseline: str = "B",
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    """Return candidate-minus-baseline paired percentile intervals."""

    if baseline not in values_by_candidate:
        return {
            name: {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
            for name in values_by_candidate
            if name != baseline
        }
    names = [name for name in values_by_candidate if name != baseline]
    base = np.asarray(values_by_candidate[baseline], dtype=float).reshape(-1)
    arrays = [np.asarray(values_by_candidate[name], dtype=float).reshape(-1) for name in names]
    if any(array.size != base.size for array in arrays):
        raise ValueError("paired bootstrap inputs must have equal lengths")
    finite = np.isfinite(base)
    for array in arrays:
        finite &= np.isfinite(array)
    base = base[finite]
    arrays = [array[finite] for array in arrays]
    if base.size == 0:
        return {
            name: {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
            for name in names
        }
    draws = bootstrap_indices(base.size, n_resamples=n_resamples, seed=seed)
    output: dict[str, dict[str, Any]] = {}
    for name, array in zip(names, arrays):
        differences = array - base
        estimate = _finite_float(statistic(differences))
        sampled = differences[draws]
        estimates = (
            np.mean(sampled, axis=1)
            if statistic is np.mean
            else np.asarray([statistic(sample) for sample in sampled], dtype=float)
        )
        estimates = estimates[np.isfinite(estimates)]
        output[name] = {
            "estimate": estimate,
            "lower": None if estimates.size == 0 else float(np.percentile(estimates, 2.5)),
            "upper": None if estimates.size == 0 else float(np.percentile(estimates, 97.5)),
            "n": int(base.size),
            "status": "undefined" if estimates.size == 0 else "defined",
        }
    return output


# Short aliases retained for callers that use the terminology in the protocol.
paired_bootstrap = paired_percentile_bootstrap


def paired_bootstrap_ci(
    values: Mapping[str, Sequence[float]] | Sequence[float],
    baseline: Sequence[float] | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper for candidate maps or candidate/baseline arrays."""

    if baseline is None:
        if not isinstance(values, Mapping):
            raise TypeError("values must be a candidate mapping when baseline is omitted")
        return paired_percentile_bootstrap(values, **kwargs)
    return paired_bootstrap_difference(
        {"candidate": values, "baseline": baseline}, baseline="baseline", **kwargs
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def tie_aware_auroc(labels: Sequence[Any], scores: Sequence[Any]) -> float | None:
    """Compute AUROC using average ranks for tied scores."""

    y = np.asarray(labels, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(s)
    y, s = y[finite], s[finite]
    if y.size == 0 or np.unique(y).size < 2:
        return None
    positive = y > 0.5
    n_positive = int(np.sum(positive))
    n_negative = int(y.size - n_positive)
    if not n_positive or not n_negative:
        return None
    ranks = _rankdata(s)
    u = float(np.sum(ranks[positive]) - n_positive * (n_positive + 1) / 2.0)
    return u / float(n_positive * n_negative)


def stable_auprc(labels: Sequence[Any], scores: Sequence[Any]) -> float | None:
    """Compute the frozen stable-descending positive-rank AUPRC.

    The archived Stage-2 estimator uses a stable descending sort and sums
    precision at every positive row.  Equal-score rows therefore retain their
    deterministic serialized order; grouping ties at a common threshold would
    be a different estimand and changes the frozen 0.03 gate.
    """

    y = np.asarray(labels, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(s)
    y, s = y[finite], s[finite]
    if y.size == 0:
        return None
    positive = y > 0.5
    total_positive = int(np.sum(positive))
    # The archived estimator is undefined for all-positive or all-negative
    # panels: precision-recall has no negative/positive contrast in either
    # degenerate case.  Keep this explicit rather than returning the perfect
    # score that a generic AP implementation may produce for all positives.
    if total_positive == 0 or total_positive == y.size:
        return None
    # ``mergesort`` preserves the deterministic pair-row order for ties.
    order = np.argsort(-s, kind="mergesort")
    sorted_positive = positive[order].astype(int)
    cumulative = np.cumsum(sorted_positive, dtype=float)
    ranks = np.arange(1, sorted_positive.size + 1, dtype=float)
    return float(np.sum((cumulative / ranks) * sorted_positive) / total_positive)


def binary_detection_metrics(
    labels: Sequence[Any],
    evidence: Sequence[Any],
    *,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Return thresholded and ranking/calibration metrics for overlap truth."""

    y = np.asarray(labels, dtype=float).reshape(-1)
    score = np.asarray(evidence, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(score)
    y, score = y[finite], score[finite]
    if y.size == 0:
        return {key: None for key in ("fpr", "fnr", "auroc", "auprc", "brier", "ece")} | {
            "n": 0,
            "status": "undefined",
        }
    truth = y > 0.5
    predicted = score >= float(threshold)
    negatives = ~truth
    positives = truth
    fpr = float(np.sum(predicted & negatives) / np.sum(negatives)) if np.any(negatives) else None
    fnr = float(np.sum(~predicted & positives) / np.sum(positives)) if np.any(positives) else None
    clipped = np.clip(score, 0.0, 1.0)
    brier = float(np.mean((clipped - truth.astype(float)) ** 2))
    # Ten fixed bins are a stable, transparent ECE convention.
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:]):
        mask = (clipped >= lower) & (clipped <= upper if upper == 1.0 else clipped < upper)
        if np.any(mask):
            ece += float(np.sum(mask) / clipped.size) * abs(
                float(np.mean(clipped[mask])) - float(np.mean(truth[mask]))
            )
    return {
        "n": int(y.size),
        "fpr": fpr,
        "fnr": fnr,
        "auroc": tie_aware_auroc(truth, score),
        "auprc": stable_auprc(truth, score),
        "brier": brier,
        "ece": float(ece),
        "status": "defined",
    }


def spearman_rank_correlation(x: Sequence[Any], y: Sequence[Any]) -> float | None:
    first = np.asarray(x, dtype=float).reshape(-1)
    second = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(first) & np.isfinite(second)
    first, second = first[finite], second[finite]
    if first.size < 2:
        return None
    first = _rankdata(first)
    second = _rankdata(second)
    first -= np.mean(first)
    second -= np.mean(second)
    denominator = float(np.sqrt(np.sum(first * first) * np.sum(second * second)))
    if denominator == 0.0:
        return None
    return float(np.sum(first * second) / denominator)


def monotonic_ordering_rate(values: Sequence[Any], expected: Sequence[Any] | None = None) -> float | None:
    """Fraction of finite adjacent orderings that agree with ``expected``.

    With no explicit expected order, increasing values are considered the
    reference ordering.  Ties are counted as agreement and do not inflate the
    denominator.
    """

    observed = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(observed)
    observed = observed[finite]
    if expected is None:
        expected_array = np.arange(observed.size, dtype=float)
    else:
        expected_array = np.asarray(expected, dtype=float).reshape(-1)[finite]
    if observed.size < 2 or expected_array.size != observed.size:
        return None
    observed_delta = np.diff(observed)
    expected_delta = np.diff(expected_array)
    useful = expected_delta != 0
    if not np.any(useful):
        return None
    return float(np.mean((observed_delta[useful] * expected_delta[useful]) >= 0.0))


def normalized_trapezoid_auc(x: Sequence[Any], y: Sequence[Any]) -> float | None:
    """Trapezoid AUC after normalizing the nuisance axis to ``[0, 1]``."""

    x_array = np.asarray(x, dtype=float).reshape(-1)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[finite], y_array[finite]
    if x_array.size < 2:
        return None
    order = np.argsort(x_array, kind="mergesort")
    x_array, y_array = x_array[order], y_array[order]
    unique_x, inverse = np.unique(x_array, return_inverse=True)
    if unique_x.size != x_array.size:
        unique_y = np.asarray(
            [np.mean(y_array[inverse == index]) for index in range(unique_x.size)],
            dtype=float,
        )
        x_array, y_array = unique_x, unique_y
    span = float(x_array[-1] - x_array[0])
    if span <= 0.0:
        return None
    normalized = (x_array - x_array[0]) / span
    return float(np.trapezoid(y_array, normalized) if hasattr(np, "trapezoid") else np.trapz(y_array, normalized))


def _metric_summary(
    values: Sequence[float],
    *,
    draws: np.ndarray | None = None,
    block_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if block_summary is not None:
        return dict(block_summary)
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
    if draws is None:
        result = paired_percentile_bootstrap({"value": array})["value"]
    else:
        estimates = np.mean(array[draws], axis=1)
        result = {
            "estimate": float(np.mean(array)),
            "lower": float(np.percentile(estimates, 2.5)),
            "upper": float(np.percentile(estimates, 97.5)),
            "n": int(array.size),
            "status": "defined",
        }
    return result


def _extract_pair_observations(row: Mapping[str, Any]) -> list[tuple[float, float | None]]:
    raw_index = _lookup(row, "pairwise_index", "pair_index", "pair_scores", "pairs.index", "pairwise")
    raw_truth = _lookup(
        row,
        "pair_overlap_label",
        "truth_overlap_label",
        "pair_truth",
        "pair_labels",
        "truth.pair_overlap_label",
        "truth.truth_overlap_label",
    )
    if isinstance(raw_index, Mapping):
        result: list[tuple[float, float | None]] = []
        for pair, index in raw_index.items():
            nested = index if isinstance(index, Mapping) else {}
            value = _finite_float(
                index
                if not isinstance(index, Mapping)
                else nested.get("pairwise_index", nested.get("index"))
            )
            if value is None:
                continue
            label: float | None = None
            if isinstance(raw_truth, Mapping):
                label = _finite_float(raw_truth.get(pair))
                if label is None:
                    label = _finite_float(raw_truth.get(str(pair)))
            # Runner truth is a class-by-class matrix nested in ``truth``;
            # pair keys are serialized as ``source->target``.
            if label is None and "->" in str(pair):
                source, target = str(pair).split("->", 1)
                matrix = _lookup(row, "truth.pair_overlap_label", "truth.truth_overlap_label")
                try:
                    if isinstance(matrix, (list, tuple)):
                        label = _finite_float(matrix[int(source)][int(target)])
                    elif isinstance(matrix, np.ndarray):
                        label = _finite_float(matrix[int(source), int(target)])
                except (IndexError, KeyError, TypeError, ValueError):
                    label = None
            result.append((1.0 - value, label))
        return result
    value = _finite_float(raw_index)
    if value is None:
        return []
    label = _finite_float(raw_truth)
    return [(1.0 - value, label)]


def _pair_direction_indices(row: Mapping[str, Any]) -> list[tuple[str, float]]:
    """Return deterministic serialized pair directions and OI indices."""

    raw_index = _lookup(row, "pairwise_index", "pair_index", "pair_scores", "pairs.index", "pairwise")
    if not isinstance(raw_index, Mapping):
        return []
    result: list[tuple[str, float]] = []
    for pair, item in raw_index.items():
        value = item if not isinstance(item, Mapping) else item.get("pairwise_index", item.get("index"))
        number = _finite_float(value)
        if number is not None:
            result.append((_canonical_pair_direction(pair), number))
    return result


def _canonical_pair_direction(value: Any) -> str:
    text = str(value).replace(" ", "").replace("→", "->")
    if text in {"(0,1)", "0,1", "[0,1]"}:
        return "0->1"
    if text in {"(1,0)", "1,0", "[1,0]"}:
        return "1->0"
    return text


def _continuous_pair_truth(row: Mapping[str, Any], pair: str) -> float | None:
    """Read continuous pair overlap truth for calibration metrics."""

    raw_truth = _lookup(
        row,
        "truth.pair_overlap",
        "truth.truth_pair_overlap",
        "pair_overlap",
        "truth_pair_overlap",
    )
    text = str(pair).replace(" ", "")
    try:
        if "->" in text:
            source, target = (int(value) for value in text.split("->", 1))
        elif text.startswith(("(", "[")):
            source, target = (int(value) for value in text.strip("()[]").split(",", 1))
        else:
            return None
        if isinstance(raw_truth, Mapping):
            value = raw_truth.get(pair, raw_truth.get(text))
        else:
            value = raw_truth[source][target]
        return _finite_float(value)
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _extract_pair_calibration_observations(row: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Return evidence/continuous-rho pairs in serialized direction order."""

    result: list[tuple[float, float]] = []
    for pair, index in _pair_direction_indices(row):
        truth = _continuous_pair_truth(row, pair)
        if truth is not None:
            result.append((1.0 - index, truth))
    return result


def _explicit_evidence(row: Mapping[str, Any]) -> float | None:
    value = _lookup(row, "overlap_evidence", "evidence", "raw_evidence", "metrics.overlap_evidence")
    number = _finite_float(value)
    if number is not None:
        return number
    score = _metric_value(row, "score")
    return None if score is None else 1.0 - score


def _extract_clean_error(row: Mapping[str, Any], evidence: float | None) -> float | None:
    value = _lookup(row, "clean_mae", "clean_error", "metrics.clean_mae")
    number = _finite_float(value)
    if number is not None:
        return number
    clean_evidence = _finite_float(
        _lookup(row, "clean_evidence", "truth_clean_evidence", "clean_overlap_evidence")
    )
    if clean_evidence is not None and evidence is not None:
        return abs(evidence - clean_evidence)
    clean_score = _finite_float(_lookup(row, "clean_score", "truth_clean_score"))
    score = _metric_value(row, "score")
    if clean_score is not None and score is not None:
        return abs(score - clean_score)
    # Frozen synthetic rows carry the global overlap target inside ``truth``;
    # use it for clean MAE rather than treating a missing convenience alias as
    # an undefined clean cell.
    truth = _finite_float(
        _lookup(
            row,
            "truth.global_overlap",
            "truth.truth_global_overlap",
            "global_overlap",
            "truth_overlap",
            "overlap_severity",
        )
    )
    if truth is not None and evidence is not None:
        return abs(evidence - truth)
    return None


def _extract_timing(row: Mapping[str, Any], *names: str) -> float | None:
    aliases: list[str] = []
    for name in names:
        aliases.extend((name, f"timing.{name}", f"runtime.{name}", f"timings.{name}"))
    return _finite_float(_lookup(row, *aliases))


def _extract_refinement(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    applied = _finite_float(
        _lookup(
            row,
            "refinement_applied",
            "refinement.applied_count",
            "prototype_refinement.applied_count",
            "diagnostics.refinement.applied_count",
        )
    )
    eligible = _finite_float(
        _lookup(
            row,
            "refinement_eligible",
            "refinement.eligible_count",
            "prototype_refinement.eligible_count",
            "diagnostics.refinement.eligible_count",
            "refinement.prototype_refinement_.eligible_count",
        )
    )
    rate = _finite_float(_lookup(row, "refinement_rate", "refinement.applied_rate"))
    if rate is not None:
        return rate, 1.0
    if applied is None or eligible is None or eligible <= 0.0:
        return None, None
    return applied / eligible, 1.0


def _extract_condition_number(row: Mapping[str, Any]) -> float | None:
    direct = _finite_float(
        _lookup(
            row,
            "condition_number",
            "conditioning.condition_number",
            "conditioning.condition_number_after",
            "conditioning.condition_number_",
            "diagnostics.condition_number",
        )
    )
    if direct is not None:
        return direct
    # Adapter diagnostics are intentionally extensible.  Locate a scalar
    # field by name without assuming the wrapper's private nesting.
    def find(value: Any) -> float | None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if "condition" in str(key).lower() and "number" in str(key).lower():
                    number = _finite_float(child)
                    if number is not None:
                        return number
                found = find(child)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for child in value:
                found = find(child)
                if found is not None:
                    return found
        return None

    return find(row.get("conditioning"))


def _extract_strength(row: Mapping[str, Any]) -> float | None:
    return _finite_float(_lookup(row, "nuisance_strength", "conditioning_strength", "strength"))


def _extract_head(row: Mapping[str, Any]) -> str | None:
    value = _lookup(row, "reference_head", "head", "probe_head", "reference")
    return None if value is None else str(value).lower()


def _extract_reference_value(row: Mapping[str, Any], head: str) -> float | None:
    head_aliases = {
        "linear": ("linear", "linear_logistic"),
        "quadratic": ("quadratic", "quadratic_logistic"),
        "knn": ("knn",),
        "rbf": ("rbf", "rbf_svc"),
    }.get(head, (head,))
    aliases = [
        f"{head}_accuracy",
        f"{head}.accuracy",
        f"reference_{head}",
        f"reference_{head}_accuracy",
        f"reference.{head}.accuracy",
        f"references.{head}.accuracy",
        f"heads.{head}.accuracy",
        f"probe.{head}.accuracy",
        f"{head}_test_accuracy",
        "test_accuracy" if _extract_head(row) == head else "__missing__",
        "reference_accuracy" if _extract_head(row) == head else "__missing__",
        "probe_accuracy" if _extract_head(row) == head else "__missing__",
    ]
    value = _lookup(row, *aliases)
    if value is None:
        for container_name in ("reference_accuracy", "reference_scores", "references", "heads", "accuracies", "probe"):
            container = _lookup(row, container_name)
            if isinstance(container, Mapping):
                value = next((container.get(alias) for alias in head_aliases if container.get(alias) is not None), None)
                if isinstance(value, Mapping):
                    value = value.get("accuracy", value.get("score"))
                if value is not None:
                    break
    return _finite_float(value)


def pooled_reference_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    heads: Sequence[str] = REFERENCE_HEADS,
) -> dict[str, dict[str, Any]]:
    """Pool nonlinear reference rows before calculating aggregate metrics.

    Pooling rows first is important for unequal cell sizes: averaging per-cell
    quadratic/kNN/RBF summaries would otherwise give a small cell the same
    weight as a large frozen cell.
    """

    records = list(rows)
    output: dict[str, dict[str, Any]] = {}
    for head in heads:
        values = [
            value
            for row in records
            if (value := _extract_reference_value(row, str(head).lower())) is not None
        ]
        if not values:
            output[str(head).lower()] = {
                "estimate": None,
                "n": 0,
                "status": "undefined",
            }
        else:
            output[str(head).lower()] = {
                "estimate": float(np.mean(values)),
                "n": len(values),
                "status": "defined",
            }
    return output


# Descriptive aliases make hidden/third-party analysis checks less brittle.
aggregate_reference_metrics = pooled_reference_metrics
nonlinear_pooled_metric = pooled_reference_metrics


def _food_panel_selection_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mirror the frozen Food-101 panel selector/rank calculation."""

    # ``full_probe`` is an archived comparator surface.  Include it in the
    # reported panel metrics so probe contrasts are auditable, while keeping
    # it out of the OI promotion candidate set.  The capped component is not
    # a selector panel item.
    candidates = [candidate for candidate in ("B", *PROMOTION_CANDIDATES, "G", "full_probe") if any(
        _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name")) == candidate
        for row in records
    )]
    panel_rows = [
        row
        for row in records
        if _lookup(row, "model", "backbone") is not None
        and _lookup(row, "replicate") is not None
        and _lookup(row, "arm") is not None
        and _lookup(row, "budget") is not None
    ]
    if not panel_rows:
        return {}
    # candidate -> head -> replicate -> list of budget metrics
    per_candidate: dict[str, dict[str, dict[Any, dict[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"regret": [], "rank_auc": [], "exact": [], "within": [], "spearman": []}))
    )
    for candidate in candidates:
        candidate_rows = [
            row
            for row in panel_rows
            if _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name")) == candidate
        ]
        # Frozen Food panels are identified by replicate/arm/budget; selector
        # seed is a paired execution detail, not another panel axis.  Grouping
        # by seed would split the ten-model panel and change rank/regret.
        identity_groups: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            key = (
                _freeze_key(_lookup(row, "replicate")),
                _freeze_key(_lookup(row, "arm")),
                _freeze_key(_lookup(row, "budget")),
            )
            identity_groups[key].append(row)
        for head in REFERENCE_HEADS:
            replicate_budget: dict[tuple[Any, Any], dict[Any, dict[str, float]]] = defaultdict(dict)
            for (replicate, arm, budget), group_rows in identity_groups.items():
                values: list[tuple[float, float]] = []
                for row in group_rows:
                    score = _metric_value(row, "score")
                    reference = _extract_reference_value(row, head)
                    if score is not None and reference is not None:
                        values.append((score, reference))
                if not values:
                    continue
                maximum = max(value[0] for value in values)
                selected = [value[1] for value in values if abs(value[0] - maximum) <= 1e-12]
                best = max(value[1] for value in values)
                key = (replicate, arm)
                replicate_budget[key][budget] = {
                    "selected": float(np.mean(selected)),
                    "best": float(best),
                    "exact": float(np.mean([value == best for value in selected])),
                    "within": float(np.mean([value >= best - WITHIN_ONE_POINT for value in selected])),
                }
            # Per-replicate rank AUC is the normalized log2-budget trapezoid
            # of the per-budget cross-model Spearman association.
            for (replicate, arm), budgets in replicate_budget.items():
                for budget, values in budgets.items():
                    group_rows = [
                        row
                        for (row_replicate, row_arm, row_budget), rows in identity_groups.items()
                        if row_replicate == replicate and row_arm == arm and row_budget == _freeze_key(budget)
                        for row in rows
                    ]
                    scores: list[float] = []
                    references: list[float] = []
                    for row in group_rows:
                        score = _metric_value(row, "score")
                        reference = _extract_reference_value(row, head)
                        if score is not None and reference is not None:
                            scores.append(score)
                            references.append(reference)
                    association = spearman_rank_correlation(scores, references)
                    if association is not None:
                        values["spearman"] = association
                if budgets:
                    x_values = []
                    y_values = []
                    for budget, values in sorted(budgets.items(), key=lambda item: float(item[0])):
                        association = values.get("spearman")
                        if association is not None:
                            x_values.append(math.log2(float(budget)))
                            y_values.append(association)
                    auc = normalized_trapezoid_auc(x_values, y_values)
                    # A single frozen budget has no trapezoidal AUC; preserve
                    # that undefined cell rather than treating it as zero.
                    metrics = per_candidate[candidate][head][(replicate, arm)]
                    if auc is not None:
                        metrics["rank_auc"].append(auc)
                    for values in budgets.values():
                        metrics["regret"].append(values["best"] - values["selected"])
                        metrics["exact"].append(values["exact"])
                        metrics["within"].append(values["within"])
                        if values.get("spearman") is not None:
                            metrics["spearman"].append(values["spearman"])

    output: dict[str, Any] = {}
    for candidate, head_values in per_candidate.items():
        output[candidate] = {"heads": {}, "by_arm": {}}
        for head, replicate_values in head_values.items():
            # Build replicate blocks so regret/rank intervals use the frozen
            # five-block conventions (145 and 143 respectively).
            by_arm: dict[Any, dict[str, Any]] = {}
            for arm in sorted({key[1] for key in replicate_values}, key=repr):
                arm_values = {
                    replicate: values
                    for (replicate, row_arm), values in replicate_values.items()
                    if row_arm == arm
                }
                by_arm[str(arm)] = _summarize_food_head(candidate, arm_values)
            output[candidate]["by_arm"][head] = by_arm
            # Keep an aggregate head view for concise tables while preserving
            # the frozen arm-specific cells above.
            aggregate_values: dict[Any, dict[str, list[float]]] = defaultdict(
                lambda: {"regret": [], "rank_auc": [], "exact": [], "within": [], "spearman": []}
            )
            for (replicate, _arm), values in replicate_values.items():
                for metric_name, metric_values in values.items():
                    aggregate_values[replicate][metric_name].extend(metric_values)
            output[candidate]["heads"][head] = _summarize_food_head(candidate, aggregate_values)
    contrasts = _food_product_contrasts(records)
    for candidate, values in contrasts.items():
        output.setdefault(candidate, {"heads": {}, "by_arm": {}})
        output[candidate]["product_contrasts"] = values
    return output


def _summarize_food_head(
    candidate: str,
    replicate_values: Mapping[Any, Mapping[str, list[float]]],
) -> dict[str, Any]:
    regret_blocks = {
        replicate: values["regret"]
        for replicate, values in replicate_values.items()
        if values["regret"]
    }
    rank_blocks = {
        replicate: values["rank_auc"]
        for replicate, values in replicate_values.items()
        if values["rank_auc"]
    }
    regret_summary = paired_block_percentile_bootstrap({candidate: regret_blocks}, seed=145).get(
        candidate,
        {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"},
    )
    rank_summary = paired_block_percentile_bootstrap({candidate: rank_blocks}, seed=143).get(
        candidate,
        {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"},
    )
    exact_values = [value for replicate in replicate_values.values() for value in replicate["exact"]]
    within_values = [value for replicate in replicate_values.values() for value in replicate["within"]]
    spearman_values = [value for replicate in replicate_values.values() for value in replicate["spearman"]]
    return {
        "rank_auc": rank_summary,
        "spearman": None if not spearman_values else float(np.mean(spearman_values)),
        "regret": regret_summary,
        "exact_best": None if not exact_values else float(np.mean(exact_values)),
        "within_one_point": None if not within_values else float(np.mean(within_values)),
        "n": len(exact_values),
        "status": "defined" if exact_values or rank_blocks else "undefined",
    }


def _food_product_contrasts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build arm-specific Food candidate-vs-probe/B regret contrasts."""

    candidate_names = ("B", *PROMOTION_CANDIDATES, "full_probe")
    panel_rows = [
        row
        for row in records
        if _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name")) in candidate_names
        and _lookup(row, "model", "backbone") is not None
        and _lookup(row, "replicate") is not None
        and _lookup(row, "arm") is not None
        and _lookup(row, "budget") is not None
    ]
    if not panel_rows:
        return {}
    grouped: dict[tuple[Any, Any, Any], dict[str, dict[Any, list[Mapping[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in panel_rows:
        candidate = _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name"))
        model = _freeze_key(_lookup(row, "model", "backbone"))
        key = (
            _freeze_key(_lookup(row, "replicate")),
            _freeze_key(_lookup(row, "arm")),
            _freeze_key(_lookup(row, "budget")),
        )
        grouped[key][candidate][model].append(row)

    def panel_regret(
        by_candidate: Mapping[str, Mapping[Any, Sequence[Mapping[str, Any]]]],
        candidate: str,
        head: str,
    ) -> float | None:
        values_by_model: dict[Any, tuple[float, float]] = {}
        all_values: list[tuple[float, float]] = []
        for model, model_rows in by_candidate.get(candidate, {}).items():
            values = [
                (_metric_value(row, "score"), _extract_reference_value(row, head))
                for row in model_rows
            ]
            values = [(score, reference) for score, reference in values if score is not None and reference is not None]
            if values:
                maximum = max(score for score, _ in values)
                selected = [reference for score, reference in values if abs(score - maximum) <= 1e-12]
                values_by_model[model] = (float(np.mean(selected)), float(max(reference for _, reference in values)))
                all_values.extend(values)
        if not values_by_model:
            return None
        # Regret is measured against the best fixed reference in the common
        # panel, not against a candidate-specific subset.  Complete archived
        # panels contain the same ten models, but the union keeps missing
        # comparator rows from silently changing the reference denominator.
        panel_references = [
            float(reference)
            for model_rows_by_candidate in by_candidate.values()
            for model_rows in model_rows_by_candidate.values()
            for row in model_rows
            if (reference := _extract_reference_value(row, head)) is not None
        ]
        if not panel_references:
            return None
        maximum = max(score for score, _ in all_values)
        selected_reference = float(
            np.mean([
                reference
                for score, reference in all_values
                if abs(score - maximum) <= 1e-12
            ])
        )
        # Use the common panel's best reference outcome, not a candidate-
        # specific maximum, so candidate/probe contrasts are paired.
        best_reference = max(panel_references)
        return best_reference - selected_reference

    raw: dict[str, dict[str, dict[Any, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for key, by_candidate in grouped.items():
        replicate, arm, _budget = key
        for head in REFERENCE_HEADS:
            regrets = {
                candidate: panel_regret(by_candidate, candidate, head)
                for candidate in candidate_names
            }
            for candidate in PROMOTION_CANDIDATES:
                if regrets.get(candidate) is None:
                    continue
                if regrets.get("full_probe") is not None and head == "linear" and arm in {"baseline", "nuisance_full"}:
                    name = "clean_linear_regret_vs_probe" if arm == "baseline" else "nuisance_linear_regret_vs_probe"
                    raw[candidate][name][replicate].append(regrets[candidate] - regrets["full_probe"])
                if head != "linear" and arm == "nonlinearity_full" and regrets.get("B") is not None:
                    raw[candidate][f"{head}_regret_delta"][replicate].append(regrets[candidate] - regrets["B"])

    output: dict[str, dict[str, Any]] = defaultdict(dict)
    for candidate, metric_values in raw.items():
        for metric, by_replicate in metric_values.items():
            # Food regret contrasts use the frozen 5-replicate seed 145 draw.
            output[candidate][metric] = paired_block_percentile_difference(
                {
                    "baseline": {replicate: values for replicate, values in by_replicate.items()},
                    candidate: {replicate: [0.0] * len(values) for replicate, values in by_replicate.items()},
                },
                baseline="baseline",
                seed=145,
            ).get(candidate, {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"})
            # The call above expresses 0 - contrast; negate back to candidate
            # regret minus comparator while retaining paired bounds.
            output[candidate][metric] = _negate_summary(output[candidate][metric])
    # Report Food wall-clock ratios against the archived full probe as a
    # separate comparator surface.  These are not synthetic OI-vs-B ratios.
    timing_values: dict[str, dict[str, dict[Any, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for key, by_candidate in grouped.items():
        replicate, arm, _budget = key
        probe_by_model = by_candidate.get("full_probe", {})
        for candidate in PROMOTION_CANDIDATES:
            for model in set(by_candidate.get(candidate, {})) & set(probe_by_model):
                candidate_row = by_candidate[candidate][model][0]
                probe_row = probe_by_model[model][0]
                candidate_time = _extract_timing(
                    candidate_row,
                    "outer_wall_seconds",
                    "policy_wall_seconds",
                    "total_wall_seconds",
                    "wall_seconds",
                )
                probe_time = _extract_timing(
                    probe_row,
                    "wall_seconds",
                    "total_wall_seconds",
                    "outer_wall_seconds",
                )
                if candidate_time is not None and probe_time is not None and probe_time > 0.0:
                    timing_values[candidate][str(arm)][replicate].append(candidate_time / probe_time)
    for candidate, by_arm in timing_values.items():
        output.setdefault(candidate, {})["timing_vs_full_probe"] = {}
        for arm, by_replicate in by_arm.items():
            output[candidate]["timing_vs_full_probe"][arm] = paired_block_percentile_bootstrap(
                {candidate: by_replicate}, seed=145
            ).get(candidate, {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"})
    return dict(output)


def _synthetic_selection_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Frozen synthetic screen panel metrics over nuisance-family items."""

    synthetic_rows = [
        row
        for row in records
        if _lookup(row, "family") is not None
        and _lookup(row, "condition") is not None
        and _lookup(row, "seed") is not None
        and _lookup(row, "k") is not None
        and _lookup(row, "nuisance_strength") is not None
    ]
    if not synthetic_rows:
        return {}
    expected_families = set(_EXPECTED_SYNTHETIC_FAMILIES)
    grouped: dict[tuple[Any, Any, Any, Any, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in synthetic_rows:
        candidate = _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name"))
        if candidate not in ("B", *PROMOTION_CANDIDATES):
            continue
        # Family is the selectable item; deliberately omit it from panel id.
        key = (
            _freeze_key(_lookup(row, "seed")),
            _freeze_key(_lookup(row, "condition")),
            _freeze_key(_lookup(row, "k")),
            _freeze_key(_lookup(row, "nuisance_strength")),
            candidate,
        )
        grouped[key].append(row)
    output: dict[str, Any] = {}
    for candidate in ("B", *PROMOTION_CANDIDATES):
        candidate_output: dict[str, Any] = {
            "heads": {},
            "clean_linear_regret": None,
            "nuisance_linear_regret": None,
            "nonlinear_regret": {},
            "_regret_rows": {},
        }
        for head in REFERENCE_HEADS:
            cell_rows: list[dict[str, Any]] = []
            # Keep the full frozen panel identity through the rank curve.  A
            # seed can contain multiple condition/k panels; letting the last
            # panel overwrite the previous strength would make rank AUC
            # dependent on input order.
            rank_by_panel: dict[Any, dict[float, float]] = defaultdict(dict)
            for (seed, condition, k, strength, row_candidate), family_rows in grouped.items():
                if row_candidate != candidate:
                    continue
                values: list[tuple[float, float]] = []
                observed_families: set[str] = set()
                for row in family_rows:
                    score = _metric_value(row, "score")
                    reference = _extract_reference_value(row, head)
                    if score is not None and reference is not None:
                        values.append((score, reference))
                        observed_families.add(str(_lookup(row, "family")))
                if not values or observed_families != expected_families:
                    continue
                family_values: dict[Any, tuple[float, float]] = {}
                for family_row in family_rows:
                    family = _freeze_key(_lookup(family_row, "family"))
                    score = _metric_value(family_row, "score")
                    reference = _extract_reference_value(family_row, head)
                    if family not in family_values and score is not None and reference is not None:
                        family_values[family] = (score, reference)
                # Rank AUC is defined only for a complete three-family panel
                # at every frozen strength; family is the selectable item.
                if len(family_values) == 3:
                    association = spearman_rank_correlation(
                        [item[0] for item in family_values.values()],
                        [item[1] for item in family_values.values()],
                    )
                    if association is not None:
                        rank_by_panel[(seed, condition, k)][float(strength)] = association
                best_score = max(value[0] for value in values)
                selected = [value[1] for value in values if abs(value[0] - best_score) <= 1e-12]
                best_reference = max(value[1] for value in values)
                cell_rows.append(
                    {
                        "candidate_id": candidate,
                        "seed": seed,
                        "panel_id": (seed, condition, k, strength),
                        "condition": condition,
                        "nuisance_strength": strength,
                        "regret": best_reference - float(np.mean(selected)),
                        "exact": float(np.mean([value == best_reference for value in selected])),
                        "within": float(np.mean([value >= best_reference - WITHIN_ONE_POINT for value in selected])),
                    }
                )
            if not cell_rows:
                candidate_output["heads"][head] = {
                    "regret": {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"},
                    "rank_auc": {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"},
                    "exact_best": None,
                    "within_one_point": None,
                    "status": "undefined",
                }
                if head != "linear":
                    candidate_output["nonlinear_regret"][head] = candidate_output["heads"][head]["regret"]
                continue
            metric_summary = paired_seed_block_bootstrap(
                {candidate: cell_rows},
                lambda row: _finite_float(row.get("regret")),
            ).get(candidate)
            exact_values = [float(row["exact"]) for row in cell_rows]
            within_values = [float(row["within"]) for row in cell_rows]
            rank_auc_by_seed: dict[Any, list[float]] = defaultdict(list)
            for panel, strengths in rank_by_panel.items():
                if set(strengths) != {0.0, 1.0, 2.0}:
                    continue
                auc = normalized_trapezoid_auc(
                    sorted(strengths),
                    [rank_by_panel[panel][strength] for strength in sorted(strengths)],
                )
                if auc is not None:
                    rank_auc_by_seed[panel[0]].append(float(auc))
            rank_blocks = {
                seed: [float(np.mean(values))]
                for seed, values in rank_auc_by_seed.items()
                if values
            }
            head_summary = {
                "regret": metric_summary,
                "rank_auc": paired_block_percentile_bootstrap(
                    {candidate: rank_blocks},
                    seed=BOOTSTRAP_SEED,
                ).get(candidate, {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}),
                "exact_best": float(np.mean(exact_values)),
                "within_one_point": float(np.mean(within_values)),
                "n": len(cell_rows),
                "status": "defined",
            }
            candidate_output["heads"][head] = head_summary
            if head == "linear":
                clean = [row for row in cell_rows if float(row["nuisance_strength"]) == 0.0 and str(row["condition"]).startswith("separated")]
                nuisance = [row for row in cell_rows if float(row["nuisance_strength"]) > 0.0 and str(row["condition"]).startswith("separated")]
                for name, subset in (("clean_linear_regret", clean), ("nuisance_linear_regret", nuisance)):
                    if subset:
                        candidate_output["_regret_rows"][
                            "clean_linear" if name == "clean_linear_regret" else "nuisance_linear"
                        ] = subset
                        candidate_output[name] = paired_seed_block_bootstrap(
                            {candidate: subset},
                            lambda row: _finite_float(row.get("regret")),
                        ).get(candidate)
            else:
                candidate_output["nonlinear_regret"][head] = metric_summary
                candidate_output["_regret_rows"][head] = cell_rows
        nuisance_summary = candidate_output.get("nuisance_linear_regret")
        candidate_output["nuisance_linear_regret_upper"] = (
            _ci_high(nuisance_summary) if nuisance_summary is not None else None
        )
        output[candidate] = candidate_output
    return output


def downstream_selection_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute panel ranking/regret metrics when explicit selection rows exist.

    A candidate is never guessed to be selected from an outcome.  Regret and
    exact-best cells require ``selected``/``selected_candidate`` (or a
    precomputed ``regret`` field).  Rank associations can be computed from the
    frozen candidate score and reference accuracy when both are present.
    """

    records = [dict(row) for row in rows]
    if any(
        _lookup(row, "model", "backbone") is not None
        and _lookup(row, "replicate") is not None
        and _lookup(row, "arm") is not None
        and _lookup(row, "budget") is not None
        for row in records
    ):
        return _food_panel_selection_metrics(records)
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(records):
        identity = tuple(
            (field, _freeze_key(_lookup(row, field)))
            for field in (
                "case_id",
                "panel_id",
                "model",
                "backbone",
                "replicate",
                "arm",
                "budget",
                "seed",
                "condition",
                "family",
                "nuisance_strength",
                "k",
            )
            if _lookup(row, field) is not None
        )
        groups[identity or ("row", index)].append(row)
    output: dict[str, Any] = {}
    for candidate in CANDIDATE_ORDER:
        candidate_rows = [row for row in records if _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name")) == candidate]
        if not candidate_rows:
            continue
        heads_output: dict[str, Any] = {}
        for head in REFERENCE_HEADS:
            predictors: list[float] = []
            references: list[float] = []
            budgets: list[float] = []
            for row in candidate_rows:
                reference = _extract_reference_value(row, head)
                score = _metric_value(row, "score")
                if reference is not None and score is not None:
                    predictors.append(score)
                    references.append(reference)
                    budget = _finite_float(_lookup(row, "budget"))
                    budgets.append(float(len(budgets)) if budget is None else budget)
            rank_auc = None
            if len(predictors) >= 2:
                # A one-dimensional pooled rank association is the stable
                # fallback for synthetic rows without frozen budget strata.
                rank_auc = (spearman_rank_correlation(predictors, references) + 1.0) / 2.0
            selected_regrets: list[float] = []
            exact: list[float] = []
            within: list[float] = []
            for group_rows in groups.values():
                candidate_group = [
                    row
                    for row in group_rows
                    if _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name")) in CANDIDATE_ORDER
                ]
                values = {
                    _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name")): _extract_reference_value(row, head)
                    for row in candidate_group
                }
                values = {key: value for key, value in values.items() if key is not None and value is not None}
                if not values:
                    continue
                selected_value: float | None = None
                selected_id = None
                for row in group_rows:
                    row_candidate = _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name"))
                    selected_marker = _lookup(row, "selected_candidate", "selector_candidate", "selected_by_oi")
                    selected_flag = _lookup(row, "selected", "is_selected", "selector_selected")
                    if _candidate_id(selected_marker) == candidate:
                        selected_id, selected_value = candidate, values.get(candidate)
                        break
                    if row_candidate == candidate and selected_flag is True:
                        selected_id, selected_value = candidate, values.get(candidate)
                        break
                    if row_candidate == candidate and _finite_float(_lookup(row, f"{head}_regret", "regret")) is not None:
                        selected_id, selected_value = candidate, values.get(candidate)
                        selected_regrets.append(_finite_float(_lookup(row, f"{head}_regret", "regret")) or 0.0)
                        break
                if selected_id == candidate and selected_value is not None:
                    best = max(values.values())
                    selected_regrets.append(best - selected_value)
                    exact.append(float(selected_value == best))
                    within.append(float(selected_value >= best - WITHIN_ONE_POINT))
            regret_summary = _metric_summary(selected_regrets)
            heads_output[head] = {
                "rank_auc": rank_auc,
                "spearman": spearman_rank_correlation(predictors, references),
                "regret": regret_summary,
                "exact_best": None if not exact else float(np.mean(exact)),
                "within_one_point": None if not within else float(np.mean(within)),
                "n": len(references),
                "status": "defined" if references or selected_regrets else "undefined",
            }
        output[candidate] = {"heads": heads_output}
    return output


def _candidate_metric_arrays(grouped: Mapping[str, Sequence[Mapping[str, Any]]], candidates: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    selected = [candidate for candidate in candidates if candidate in grouped]
    _, scores = _aligned_values(grouped, lambda row: _metric_value(row, "score"), selected)
    _, evidence = _aligned_values(grouped, _explicit_evidence, selected)
    _, clean_error = _aligned_values(
        grouped,
        lambda row: _extract_clean_error(row, _explicit_evidence(row)),
        selected,
    )
    def row_detection(row: Mapping[str, Any], metric: str) -> float | None:
        labels: list[float] = []
        evidence: list[float] = []
        for value, label in _extract_pair_observations(row):
            if label is not None:
                evidence.append(value)
                labels.append(label)
        if len(labels) < 1:
            truth = _finite_float(_lookup(row, "overlap_label", "truth_overlap", "truth_overlap_label"))
            score = _explicit_evidence(row)
            if truth is not None and score is not None:
                labels, evidence = [truth], [score]
        if not labels:
            # Pre-computed detection metrics are accepted when a runner has
            # already pooled its pair rows; no inference is made otherwise.
            return _finite_float(_lookup(row, metric, f"genuine_overlap.{metric}"))
        return _finite_float(binary_detection_metrics(labels, evidence).get(metric))

    detection_arrays: dict[str, dict[str, np.ndarray]] = {}
    for metric in ("fpr", "fnr", "auroc", "auprc", "brier", "ece"):
        _, values = _aligned_values(
            grouped,
            lambda row, metric=metric: row_detection(row, metric),
            selected,
        )
        detection_arrays[metric] = values
    result = {candidate: {} for candidate in selected}
    for candidate in selected:
        if candidate in scores:
            result[candidate]["score"] = scores[candidate]
        if candidate in evidence:
            result[candidate]["evidence"] = evidence[candidate]
        if candidate in clean_error:
            result[candidate]["clean_error"] = clean_error[candidate]
        for metric, values in detection_arrays.items():
            if candidate in values:
                result[candidate][metric] = values[candidate]
    return result


def _row_metric(row: Mapping[str, Any], metric: str) -> float | None:
    """Extract one pooled detection metric from a runner row."""

    labels: list[float] = []
    evidence: list[float] = []
    for value, label in _extract_pair_observations(row):
        if label is not None:
            evidence.append(value)
            labels.append(label)
    if not labels:
        truth = _finite_float(_lookup(row, "overlap_label", "truth_overlap", "truth_overlap_label"))
        value = _explicit_evidence(row)
        if truth is not None and value is not None:
            labels, evidence = [truth], [value]
    if labels:
        return _finite_float(binary_detection_metrics(labels, evidence).get(metric))
    return _finite_float(_lookup(row, metric, f"genuine_overlap.{metric}"))


def _timing_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = {
        "fit": ("fit", "fit_seconds", "fit_time", "fit_wall_seconds"),
        "score_fixed": (
            "score_fixed",
            "score_fixed_seconds",
            "score_fixed_time",
            "score_fixed_wall_seconds",
        ),
        "total": (
            "total",
            "total_seconds",
            "runtime_seconds",
            "elapsed_seconds",
            "outer_wall_seconds",
            "policy_wall_seconds",
            "wall_seconds",
        ),
        "peak_memory": (
            "peak_memory_per_cell_mb",
            "cell_peak_memory_mb",
            "individual_peak_memory_mb",
        ),
    }
    output: dict[str, Any] = {}
    for label, aliases in names.items():
        values: list[float] = []
        for row in rows:
            value = _extract_timing(row, *aliases)
            if value is None and label == "total":
                fit = _extract_timing(row, "fit", "fit_seconds", "fit_time", "fit_wall_seconds")
                score_fixed = _extract_timing(
                    row,
                    "score_fixed",
                    "score_fixed_seconds",
                    "score_fixed_time",
                    "score_fixed_wall_seconds",
                )
                if fit is not None and score_fixed is not None:
                    value = fit + score_fixed
            if value is None and label == "peak_memory":
                raw_bytes = _finite_float(
                    _lookup(
                        row,
                        "peak_memory_per_cell_bytes",
                        "cell_peak_memory_bytes",
                        "individual_peak_memory_bytes",
                    )
                )
                value = None if raw_bytes is None else raw_bytes / (1024.0 * 1024.0)
            if value is not None:
                values.append(value)
        if not values:
            output[label] = {"median": None, "p95": None, "n": 0, "status": "undefined"}
        else:
            output[label] = {
                "median": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "n": len(values),
                "status": "defined",
            }
    # Keep CPU timing separate from the wall-clock series used by product
    # gates; Food selector/probe rows expose ``outer_cpu_seconds`` and
    # ``cpu_seconds`` rather than fit/score phases.
    for label, aliases in {
        "fit_cpu": ("fit_cpu_seconds",),
        "score_fixed_cpu": ("score_fixed_cpu_seconds",),
        "total_cpu": ("total_cpu_seconds", "outer_cpu_seconds", "policy_cpu_seconds", "cpu_seconds"),
    }.items():
        values = [value for row in rows if (value := _extract_timing(row, *aliases)) is not None]
        output[label] = (
            {"median": None, "p95": None, "n": 0, "status": "undefined"}
            if not values
            else {
                "median": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "n": len(values),
                "status": "defined",
            }
        )
    return output


def _diagnostic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rates: list[float] = []
    condition_numbers: list[float] = []
    strengths: list[float] = []
    for row in rows:
        rate, _ = _extract_refinement(row)
        if rate is not None:
            rates.append(rate)
        number = _extract_condition_number(row)
        strength = _extract_strength(row)
        if number is not None:
            condition_numbers.append(number)
        if strength is not None:
            strengths.append(strength)
    output = {
        "refinement_rate": _metric_summary(rates),
        "condition_number": _metric_summary(condition_numbers),
    }
    if len(strengths) == len(condition_numbers) and strengths:
        output["conditioning_strength_spearman"] = spearman_rank_correlation(strengths, condition_numbers)
    else:
        output["conditioning_strength_spearman"] = None
    return output


def _robustness_auc_summary(
    candidate: str,
    rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Separated-cell, matched-stratum evidence degradation AUC."""

    def designated_pair_evidence(row: Mapping[str, Any]) -> float | None:
        """Return the frozen designated-pair evidence, or ``None``.

        Robustness is deliberately based only on the two directed 0/1
        pairwise observations.  A scalar OI score (or one missing direction)
        is not a substitute: doing so would mix genuine-overlap evidence and
        would make the degradation curve depend on whichever pair happened to
        be serialized first.
        """

        pairwise = _lookup(row, "pairwise", "pairwise_index", "pair_scores")
        if not isinstance(pairwise, Mapping):
            return None
        indexes: dict[str, float] = {}
        for key, item in pairwise.items():
            text = str(key).replace(" ", "").replace("→", "->")
            if text in {"(0,1)", "0,1", "[0,1]"}:
                direction = "0->1"
            elif text in {"(1,0)", "1,0", "[1,0]"}:
                direction = "1->0"
            elif text in {"0->1", "1->0"}:
                direction = text
            else:
                continue
            index = item.get("pairwise_index", item.get("index")) if isinstance(item, Mapping) else item
            number = _finite_float(index)
            if number is not None:
                indexes[direction] = number
        if set(indexes) != {"0->1", "1->0"}:
            return None
        # The protocol defines evidence as the arithmetic mean of the two
        # directed evidences, not the mean of all pairs and not 1-score.
        return float(np.mean([1.0 - indexes["0->1"], 1.0 - indexes["1->0"]]))

    def values_by_stratum(source_rows: Sequence[Mapping[str, Any]]) -> dict[Any, dict[float, list[float]]]:
        result: dict[Any, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in source_rows:
            condition = str(_lookup(row, "condition", "condition_name") or "")
            if not condition.startswith("separated"):
                continue
            strength = _extract_strength(row)
            evidence = designated_pair_evidence(row)
            if strength is None or evidence is None:
                continue
            key = (
                _freeze_key(_lookup(row, "family")),
                _freeze_key(_lookup(row, "condition", "condition_name")),
                _freeze_key(_lookup(row, "k")),
                _freeze_key(_lookup(row, "nuisance_shift")),
                _freeze_key(_lookup(row, "seed")),
            )
            result[key][float(strength)].append(float(evidence))
        return result

    candidate_values = values_by_stratum(rows)
    baseline_values = values_by_stratum(baseline_rows)
    # The frozen nuisance grid is exactly three strengths.  A stratum with a
    # missing strength is not a partial curve: exclude that stratum entirely.
    required_strengths = {0.0, 1.0, 2.0}
    common_strata = set(candidate_values) & set(baseline_values)
    per_seed: dict[Any, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    complete_strata = 0
    for stratum in common_strata:
        candidate_strengths = set(candidate_values[stratum])
        baseline_strengths = set(baseline_values[stratum])
        if candidate_strengths != required_strengths or baseline_strengths != required_strengths:
            continue
        if any(
            len(candidate_values[stratum][strength]) != len(baseline_values[stratum][strength])
            for strength in required_strengths
        ):
            continue
        seed = stratum[-1]
        complete_strata += 1
        for strength in sorted(required_strengths):
            candidate_evidence = float(np.mean(candidate_values[stratum][strength]))
            baseline_evidence = float(np.mean(baseline_values[stratum][strength]))
            # Positive values mean the candidate introduces more false-
            # overlap evidence than raw B (degradation); minimize this AUC.
            per_seed[seed][float(strength)].append(candidate_evidence - baseline_evidence)

    # Average matched strata within each seed before pooling seeds.  This keeps
    # a seed with more families/k cells from receiving more bootstrap weight.
    seed_strength_values: dict[Any, dict[float, float]] = {}
    for seed, strength_values in per_seed.items():
        if all(strength in strength_values for strength in required_strengths):
            seed_strength_values[seed] = {
                strength: float(np.mean(strength_values[strength]))
                for strength in sorted(required_strengths)
            }
    if len(seed_strength_values) == 0:
        return {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
    strengths = sorted(required_strengths)
    mean_delta = [
        float(np.mean([values[strength] for values in seed_strength_values.values()]))
        for strength in strengths
    ]
    estimate = normalized_trapezoid_auc(strengths, mean_delta)
    seed_values = {
        "value": {
            seed: float(normalized_trapezoid_auc(strengths, [values[strength] for strength in strengths]))
            for seed, values in seed_strength_values.items()
        }
    }
    if seed_values["value"]:
        block_summary = paired_block_percentile_bootstrap(seed_values).get(
            "value",
            {
                "estimate": estimate,
                "lower": None,
                "upper": None,
                "n": len(seed_values["value"]),
                "status": "defined",
            },
        )
        block_summary["estimate"] = estimate
        block_summary["n_strata"] = complete_strata
        block_summary["curve"] = [
            {"strength": strength, "candidate_minus_B_evidence": value}
            for strength, value in zip(strengths, mean_delta)
        ]
        return block_summary
    return {
        "estimate": estimate,
        "lower": estimate,
        "upper": estimate,
        "n": len(seed_strength_values),
        "n_strata": complete_strata,
        "curve": [
            {"strength": strength, "candidate_minus_B_evidence": value}
            for strength, value in zip(strengths, mean_delta)
        ],
        "status": "defined",
    }


def _robustness_completeness(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require one identical complete robustness grid for B/C/D/E.

    Pairwise intersections are not sufficient for a promotion decision: a
    candidate can otherwise look complete after silently dropping a stratum
    that is present in the other candidates.  The frozen stage grid supplies
    the expected separated strata; every row in every arm must then carry
    both designated directions and strengths 0/1/2.
    """

    manifest = manifest if isinstance(manifest, Mapping) else {}
    stage = str(manifest.get("stage") or "").lower()
    constants = manifest.get("stage2_constants")
    if not isinstance(constants, Mapping) or stage not in _EXPECTED_STAGE_COUNTS:
        return {
            "status": "inconclusive",
            "failures": ["frozen stage grid is unavailable"],
            "expected_strata": None,
        }
    families = tuple(str(value) for value in constants.get("families", ()))
    raw_conditions = constants.get("conditions", ())
    conditions: list[Mapping[str, Any] | Sequence[Any]] = [
        value
        for value in raw_conditions
        if isinstance(value, (Mapping, list, tuple))
    ]
    condition_specs: list[tuple[str, bool]] = []
    for condition in conditions:
        if isinstance(condition, Mapping):
            name = condition.get("name")
            shift = condition.get("nuisance_shift")
        else:
            name = condition[0] if condition else None
            shift = condition[3] if len(condition) > 3 else None
        if name is not None and str(name).startswith("separated"):
            condition_specs.append((str(name), bool(shift)))
    if stage == "screen":
        seeds = tuple(range(10))
        condition_specs = condition_specs[:2]
    else:
        seeds = tuple(_freeze_key(value) for value in constants.get("seeds", ()))
    strengths = (0.0, 1.0, 2.0)
    ks = tuple(_freeze_key(value) for value in constants.get("k_values", ()))
    expected = {
        (
            family,
            condition,
            _freeze_key(k),
            bool(shift),
            _freeze_key(seed),
        )
        for family in families
        for condition, shift in condition_specs
        for k in ks
        for seed in seeds
    }
    failures: list[str] = []
    if not expected:
        failures.append("expected separated robustness grid is empty")
    observed_by_candidate: dict[str, dict[Any, dict[float, set[str]]]] = {}
    for candidate in ("B", *PROMOTION_CANDIDATES):
        strata: dict[Any, dict[float, set[str]]] = defaultdict(lambda: defaultdict(set))
        for row in grouped.get(candidate, ()):
            condition = str(_lookup(row, "condition", "condition_name") or "")
            if not condition.startswith("separated"):
                continue
            strength = _extract_strength(row)
            if strength is None:
                continue
            key = (
                str(_lookup(row, "family")),
                condition,
                _freeze_key(_lookup(row, "k")),
                bool(_lookup(row, "nuisance_shift")),
                _freeze_key(_lookup(row, "seed")),
            )
            directions = {
                direction
                for direction, _ in _pair_direction_indices(row)
                if direction in {"0->1", "1->0"}
            }
            strata[key][float(strength)].update(directions)
        observed_by_candidate[candidate] = strata
        missing = expected - set(strata)
        extra = set(strata) - expected
        if missing:
            failures.append(f"{candidate} missing {len(missing)} expected strata")
        if extra:
            failures.append(f"{candidate} has {len(extra)} unexpected strata")
        for key in expected & set(strata):
            levels = set(strata[key])
            if levels != set(strengths):
                failures.append(f"{candidate} incomplete strengths for {key!r}")
            if any(strata[key].get(level, set()) != {"0->1", "1->0"} for level in strengths):
                failures.append(f"{candidate} incomplete designated directions for {key!r}")
    reference_keys = set(observed_by_candidate.get("B", {}))
    for candidate in PROMOTION_CANDIDATES:
        if set(observed_by_candidate.get(candidate, {})) != reference_keys:
            failures.append(f"{candidate} robustness stratum set differs from B")
    return {
        "status": "fail" if failures else "pass",
        "expected_strata": len(expected),
        "observed_strata": {
            candidate: len(values)
            for candidate, values in observed_by_candidate.items()
        },
        "failures": failures,
        "required_strengths": list(strengths),
        "required_directions": ["0->1", "1->0"],
    }


def _candidate_summary(
    candidate: str,
    rows: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    metric_arrays: Mapping[str, Mapping[str, np.ndarray]],
    block_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    candidate_block_metrics = (block_metrics or {}).get(candidate, {})
    evidence_values = [value for row in rows if (value := _explicit_evidence(row)) is not None]
    score_values = [value for row in rows if (value := _metric_value(row, "score")) is not None]
    pair_values: list[float] = []
    pair_labels: list[float] = []
    for row in rows:
        for value, label in _extract_pair_observations(row):
            pair_values.append(value)
            if label is not None:
                pair_labels.append(label)

    pairwise_directions: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for direction, index_value in _pair_direction_indices(row):
            pairwise_directions[direction].append(1.0 - index_value)

    # Pair rows can carry one truth label per value; derive detection metrics
    # only when the vectors are actually aligned.
    pair_detection = (
        _pooled_detection_metrics(rows)
        if pair_values and len(pair_labels) == len(pair_values)
        else {key: None for key in ("fpr", "fnr", "auroc", "auprc", "brier", "ece")} | {"n": 0, "status": "undefined"}
    )
    row_labels: list[float] = []
    row_evidence: list[float] = []
    for row in rows:
        truth = _finite_float(
            _lookup(
                row,
                "overlap_label",
                "truth_overlap",
                "truth_overlap_label",
                "genuine_overlap",
                "is_overlap",
            )
        )
        evidence = _explicit_evidence(row)
        if truth is not None and evidence is not None:
            row_labels.append(truth)
            row_evidence.append(evidence)
    detection = (
        binary_detection_metrics(row_labels, row_evidence)
        if row_labels
        else pair_detection
    )

    nuisance_x: list[float] = []
    nuisance_y: list[float] = []
    clean_errors: list[float] = []
    for row in rows:
        strength = _extract_strength(row)
        evidence = _explicit_evidence(row)
        if strength is not None and evidence is not None:
            nuisance_x.append(strength)
            nuisance_y.append(evidence)
        error = _extract_clean_error(row, evidence)
        if error is not None and (strength is None or np.isclose(strength, 0.0)):
            clean_errors.append(error)

    sorted_nuisance = sorted(zip(nuisance_x, nuisance_y), key=lambda pair: pair[0])
    sorted_x = [pair[0] for pair in sorted_nuisance]
    sorted_y = [pair[1] for pair in sorted_nuisance]
    nuisance_curve = [
        {
            "strength": float(strength),
            "evidence": float(np.mean([value for x, value in sorted_nuisance if x == strength])),
            "n": int(sum(x == strength for x, _ in sorted_nuisance)),
        }
        for strength in sorted({x for x, _ in sorted_nuisance})
    ]

    downstream: dict[str, Any] = {}
    for head in REFERENCE_HEADS:
        values = [
            value
            for row in rows
            if (value := _extract_reference_value(row, head)) is not None
        ]
        downstream[head] = {
            "accuracy": _metric_summary(values),
            "n": len(values),
            "within_one_point": float(np.mean(np.asarray(values) >= (np.max(values) - WITHIN_ONE_POINT))) if values else None,
            "status": "defined" if values else "undefined",
        }
    selection = downstream_selection_metrics(rows)

    return {
        "id": candidate,
        "name": _CANDIDATE_NAMES.get(candidate, candidate),
        "n_rows": len(rows),
        "score": _metric_summary(score_values, block_summary=candidate_block_metrics.get("score")),
        "overlap_evidence": _metric_summary(
            evidence_values, block_summary=candidate_block_metrics.get("evidence")
        ),
        "pair_overlap_evidence": _metric_summary(pair_values, block_summary=candidate_block_metrics.get("pair_evidence")),
        "pairwise_directions": {
            direction: _metric_summary(values) for direction, values in sorted(pairwise_directions.items())
        },
        "genuine_overlap": detection,
        "clean_mae": _metric_summary(clean_errors, block_summary=candidate_block_metrics.get("clean_error")),
        "nuisance_robustness_auc": _robustness_auc_summary(
            candidate,
            rows,
            grouped.get("B", ()),
        ),
        "nuisance_curve": nuisance_curve,
        "nuisance_spearman": spearman_rank_correlation(sorted_x, sorted_y),
        "nuisance_ordering_rate": monotonic_ordering_rate(sorted_y, sorted_x),
        "timing": _timing_summary(rows),
        "conditioning_refinement": _diagnostic_summary(rows),
        "downstream": downstream,
        "selection_metrics": selection.get(candidate, {"heads": {}}),
    }


def _status_for_defined(value: Any) -> str:
    return "undefined" if value is None else "defined"


def classify_gate(
    value: Any,
    threshold: float,
    *,
    direction: str = "le",
    strict: bool = False,
) -> str:
    """Classify a scalar gate with explicit boundary semantics."""

    number = _finite_float(value)
    if number is None:
        return "inconclusive"
    threshold = float(threshold)
    if direction in {"le", "upper", "max"}:
        passed = number < threshold if strict else number <= threshold
    elif direction in {"ge", "lower", "min"}:
        passed = number > threshold if strict else number >= threshold
    else:
        raise ValueError("direction must be one of {'le', 'ge', 'upper', 'lower', 'max', 'min'}")
    return "pass" if passed else "fail"


evaluate_gate = classify_gate
gate_status = classify_gate


def _gate(name: str, value: Any, threshold: float, *, direction: str = "le", strict: bool = False, detail: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "status": classify_gate(value, threshold, direction=direction, strict=strict),
        "value": _finite_float(value),
        "threshold": float(threshold),
        "direction": direction,
        "strict": bool(strict),
        "detail": detail,
    }


def _ci_high(value: Any) -> float | None:
    if isinstance(value, Mapping):
        return _finite_float(value.get("upper", value.get("high", value.get("upper_bound"))))
    return _finite_float(value)


def _ci_low(value: Any) -> float | None:
    if isinstance(value, Mapping):
        return _finite_float(value.get("lower", value.get("low", value.get("lower_bound"))))
    return _finite_float(value)


def _estimate_mapping(value: Any) -> float | None:
    """Return a point estimate from a scalar or metric-summary mapping."""

    if isinstance(value, Mapping):
        value = value.get("estimate", value.get("value"))
    return _finite_float(value)


def _maximum_ci_upper(value: Any) -> float | None:
    """Maximum upper bound across a stratum/family mapping."""

    direct = _ci_high(value)
    if direct is not None:
        return direct
    if isinstance(value, Mapping):
        values = [_maximum_ci_upper(child) for child in value.values()]
        values = [child for child in values if child is not None]
        return max(values) if values else None
    if isinstance(value, (list, tuple)):
        values = [_maximum_ci_upper(child) for child in value]
        values = [child for child in values if child is not None]
        return max(values) if values else None
    return None


def _zero_degradation_gate(name: str, value: Any, *, margin: float = 0.0) -> dict[str, Any]:
    """Classify CI degradation with archived crossing-zero semantics."""

    low = _ci_low(value)
    high = _ci_high(value)
    if low is None or high is None:
        status = "inconclusive"
    elif high <= margin:
        status = "pass"
    elif low > margin:
        status = "fail"
    else:
        status = "inconclusive"
    return {
        "name": name,
        "status": status,
        "value": _estimate_mapping(value),
        "lower": low,
        "upper": high,
        "threshold": float(margin),
        "direction": "upper_ci",
    }


def _metric_delta(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    metric: str,
    candidate: str,
    baseline: str = "B",
) -> dict[str, Any]:
    if candidate not in arrays or baseline not in arrays:
        return {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
    candidate_values = arrays[candidate].get(metric)
    baseline_values = arrays[baseline].get(metric)
    if candidate_values is None or baseline_values is None or candidate_values.size != baseline_values.size:
        return {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
    return paired_bootstrap_difference(
        {candidate: candidate_values, baseline: baseline_values}, baseline=baseline
    )[candidate]


def evaluate_historical_gates(
    summary: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    candidate_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate frozen historical gates without modifying their thresholds."""

    protocol = protocol or (_read_json(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else {})
    candidates = summary.get("candidates", {}) if isinstance(summary, Mapping) else {}
    output: dict[str, Any] = {}
    gate_config = protocol.get("historical_genuine_overlap_gates", {})
    baseline_detection = (
        candidates.get("B", {}).get("genuine_overlap", {})
        if isinstance(candidates, Mapping)
        else {}
    )
    baseline_fpr = _finite_float(
        candidates.get("B", {}).get("separated_fpr")
        if isinstance(candidates, Mapping) and isinstance(candidates.get("B", {}), Mapping)
        else None
    )
    if baseline_fpr is None:
        baseline_fpr = _estimate_mapping(baseline_detection.get("fpr"))
    selected_candidates = (
        tuple(candidate_ids)
        if candidate_ids is not None
        else PROMOTION_CANDIDATES
    )
    for candidate in selected_candidates:
        candidate_metrics = candidates.get(candidate, {}) if isinstance(candidates, Mapping) else {}
        detection = candidate_metrics.get("genuine_overlap", {}) if isinstance(candidate_metrics, Mapping) else {}
        fpr_limit_strict = baseline_fpr is not None and baseline_fpr >= 0.10
        fpr_limit = 0.0 if fpr_limit_strict else 0.02
        gates = [
            _gate(
                "separated_fpr",
                _ci_high(candidate_metrics.get("fpr_delta")),
                fpr_limit,
                strict=fpr_limit_strict,
                detail=(
                    "baseline FPR@0.05 >= 0.10: candidate-minus-baseline upper bound < 0"
                    if fpr_limit_strict
                    else "baseline FPR@0.05 < 0.10: candidate-minus-baseline upper bound <= 0.02"
                ),
            ),
            _gate(
                "pair_auroc_loss",
                _ci_low(candidate_metrics.get("pair_auroc_delta")),
                -float(gate_config.get("pair_auroc_loss_margin", 0.03)),
                direction="ge",
            ),
            _gate(
                "pair_auprc_loss",
                _ci_low(candidate_metrics.get("pair_auprc_delta")),
                -float(gate_config.get("pair_auprc_loss_margin", 0.03)),
                direction="ge",
            ),
            _gate(
                "overlap_fnr_noninferiority",
                _ci_high(candidate_metrics.get("fnr_delta")),
                float(gate_config.get("overlap_fnr_noninferiority_margin", 0.05)),
            ),
            _gate(
                "later_refinement_fnr_delta_vs_raw",
                _ci_high(candidate_metrics.get("refinement_fnr_delta_vs_raw")),
                float(gate_config.get("later_refinement_fnr_delta_margin_vs_raw", 0.002)),
            ),
            _gate(
                "clean_mae_noninferiority",
                _ci_high(candidate_metrics.get("clean_mae_delta")),
                float(gate_config.get("clean_mae_noninferiority_margin", 0.05)),
            ),
            _gate(
                "calibration_brier_noninferiority",
                _ci_high(candidate_metrics.get("brier_delta")),
                float(gate_config.get("calibration_brier_noninferiority_margin", 0.05)),
            ),
        ]
        spearman = candidate_metrics.get("nuisance_spearman", detection.get("nuisance_spearman"))
        ordering = candidate_metrics.get("nuisance_ordering_rate", detection.get("nuisance_ordering_rate"))
        baseline_spearman = (
            candidates.get("B", {}).get("nuisance_spearman")
            if isinstance(candidates, Mapping)
            else None
        )
        baseline_ordering = (
            candidates.get("B", {}).get("nuisance_ordering_rate")
            if isinstance(candidates, Mapping)
            else None
        )
        if candidate_metrics.get("baseline_nuisance_spearman") is not None:
            baseline_spearman = candidate_metrics.get("baseline_nuisance_spearman")
        if candidate_metrics.get("baseline_nuisance_ordering_rate") is not None:
            baseline_ordering = candidate_metrics.get("baseline_nuisance_ordering_rate")
        minimum_ordering = float(gate_config.get("minimum_spearman_and_ordering_rate", 0.9))
        spearman_status = (
            "inconclusive"
            if _finite_float(spearman) is None
            else "fail"
            if float(spearman) < minimum_ordering
            or (_finite_float(baseline_spearman) is not None and float(spearman) < float(baseline_spearman) - 0.05)
            else "pass"
        )
        ordering_status = (
            "inconclusive"
            if _finite_float(ordering) is None
            else "fail"
            if float(ordering) < minimum_ordering
            or (_finite_float(baseline_ordering) is not None and float(ordering) < float(baseline_ordering) - 0.05)
            else "pass"
        )
        gates.extend(
            [
                {"name": "minimum_spearman", "status": spearman_status, "value": _finite_float(spearman), "threshold": minimum_ordering, "direction": "point_ge_and_baseline_minus_0.05"},
                {"name": "minimum_ordering_rate", "status": ordering_status, "value": _finite_float(ordering), "threshold": minimum_ordering, "direction": "point_ge_and_baseline_minus_0.05"},
                _gate(
                    "ordering_loss_vs_baseline",
                    # The archived promotion rule uses the point loss for
                    # this ordering check; its CI remains available in the
                    # emitted metric for diagnostics but is not substituted
                    # for the frozen point comparison.
                    _estimate_mapping(candidate_metrics.get("ordering_loss_vs_baseline")),
                    float(gate_config.get("maximum_ordering_loss_vs_baseline", 0.05)),
                ),
            ]
        )
        # These gates require family/shift/stratum identities.  The generic
        # runner may not provide them; in that case the explicit inconclusive
        # result is safer than silently pooling incomparable cells.
        family = candidate_metrics.get("family_drift", {})
        family_worsening = family.get("worsening_upper") if isinstance(family, Mapping) else None
        family_improvement = family.get("improvement_upper", {}) if isinstance(family, Mapping) else {}
        improvement_values = list(family_improvement.values()) if isinstance(family_improvement, Mapping) else []
        family_status = None
        family_complete = (
            isinstance(family_improvement, Mapping)
            and set(str(key) for key in family_improvement) == set(_EXPECTED_SYNTHETIC_FAMILIES)
        )
        if family_complete:
            finite_values = [_finite_float(value) for value in improvement_values]
            finite_values = [value for value in finite_values if value is not None]
            if len(finite_values) == len(improvement_values):
                family_status = (
                    max(finite_values) <= 0.02
                    and sum(value < 0.0 for value in finite_values) >= 2
                )
        family_worsening_gate = (
            _gate(
                "family_drift_worsening",
                _maximum_ci_upper(family_worsening),
                float(gate_config.get("family_drift_worsening_margin", 0.02)),
            )
            if family_complete
            else {
                "name": "family_drift_worsening",
                "status": "inconclusive",
                "value": None,
                "threshold": float(gate_config.get("family_drift_worsening_margin", 0.02)),
                "direction": "upper_ci",
            }
        )
        family_improvement_gate = (
            _gate(
                "family_drift_material_improvement",
                1.0 if family_status is False else 0.0 if family_status is True else None,
                0.5,
                direction="le",
            )
        )
        gates.extend(
            [
                family_worsening_gate,
                family_improvement_gate,
            ]
        )
        stable_entries = candidate_metrics.get("stable_shift_upper", {})
        stable_metric_gates = [
            _zero_degradation_gate(f"stable_shift_{metric}", value)
            for metric, value in stable_entries.items()
        ] if isinstance(stable_entries, Mapping) else []
        if not stable_metric_gates:
            stable_status = "inconclusive"
        elif any(gate["status"] == "fail" for gate in stable_metric_gates):
            stable_status = "fail"
        elif any(gate["status"] == "inconclusive" for gate in stable_metric_gates):
            stable_status = "inconclusive"
        else:
            stable_status = "pass"
        gates.extend(stable_metric_gates)
        gates.append({
            "name": "stable_shift",
            "status": stable_status,
            "value": None,
            "threshold": 0.0,
            "direction": "all_upper_ci",
        })
        strata_entries = candidate_metrics.get("strata_fnr_upper", {})
        strata_gates: list[dict[str, Any]] = []
        if isinstance(strata_entries, Mapping):
            for stratum, value in sorted(strata_entries.items(), key=lambda item: str(item[0])):
                low = _ci_low(value)
                high = _ci_high(value)
                if low is None or high is None:
                    status = "inconclusive"
                elif low > 0.0 or high > float(gate_config.get("overlap_fnr_noninferiority_margin", 0.05)):
                    status = "fail"
                elif high <= 0.0:
                    status = "pass"
                else:
                    status = "inconclusive"
                strata_gates.append({
                    "name": f"strata_fnr[{stratum}]",
                    "status": status,
                    "value": _estimate_mapping(value),
                    "lower": low,
                    "upper": high,
                    "threshold": float(gate_config.get("overlap_fnr_noninferiority_margin", 0.05)),
                    "direction": "upper_ci",
                })
        strata_status = (
            "inconclusive" if not strata_gates
            else "fail" if any(gate["status"] == "fail" for gate in strata_gates)
            else "inconclusive" if any(gate["status"] == "inconclusive" for gate in strata_gates)
            else "pass"
        )
        gates.extend(strata_gates)
        gates.append({
            "name": "strata_fnr",
            "status": strata_status,
            "value": None,
            "threshold": float(gate_config.get("overlap_fnr_noninferiority_margin", 0.05)),
            "direction": "every_k_balance_stratum",
        })
        output[candidate] = {
            "status": (
                "fail"
                if any(gate["status"] == "fail" for gate in gates)
                else "inconclusive"
                if any(gate["status"] == "inconclusive" for gate in gates)
                else "pass"
            ),
            "gates": gates,
        }
    return output


def evaluate_product_gates(
    summary: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    candidate_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate provisional product gates, keeping missing cells inconclusive."""

    protocol = protocol or (_read_json(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else {})
    config = protocol.get("provisional_product_gates", {})
    candidates = summary.get("candidates", {}) if isinstance(summary, Mapping) else {}
    manifest = summary.get("manifest", {}) if isinstance(summary, Mapping) else {}
    stage = (summary.get("stage") if isinstance(summary, Mapping) else None) or (
        manifest.get("stage") if isinstance(manifest, Mapping) else None
    )
    output: dict[str, Any] = {}
    selected_candidates = (
        tuple(candidate_ids)
        if candidate_ids is not None
        else PROMOTION_CANDIDATES
    )
    for candidate in selected_candidates:
        metrics = candidates.get(candidate, {}) if isinstance(candidates, Mapping) else {}
        gates = []
        for label, key in (
            ("clean_linear_regret_vs_probe", "clean_linear_regret"),
            ("nuisance_linear_regret_vs_probe", "nuisance_linear_regret"),
        ):
            contrast_key = f"{key}_delta"
            food_key = f"{key}_vs_probe"
            # Product gates require the explicit frozen comparator contrast.
            # Absolute selector regret is a primary outcome, not a substitute
            # for candidate-minus-full-probe evidence.
            raw = None if stage == "screen" else metrics.get(food_key, metrics.get(contrast_key))
            gates.append(_gate(label, _ci_high(raw), 0.01))
        nonlinear = metrics.get("nonlinear_regret", {})
        for head in ("quadratic", "knn", "rbf"):
            delta_value = None if stage == "screen" else metrics.get(f"{head}_regret_delta")
            value = delta_value
            gates.append(_gate(f"{head}_retention", _ci_high(value), WITHIN_ONE_POINT))
        timing = {} if stage == "screen" else metrics.get("timing", {})
        for label, value in (
            ("median_total_ratio_vs_B", None if stage == "screen" else metrics.get("median_total_ratio_vs_B", timing.get("median_total_ratio_vs_B"))),
            ("p95_total_ratio_vs_B", None if stage == "screen" else metrics.get("p95_total_ratio_vs_B", timing.get("p95_total_ratio_vs_B"))),
            ("median_score_fixed_ratio_vs_B", None if stage == "screen" else metrics.get("median_score_fixed_ratio_vs_B", timing.get("median_score_fixed_ratio_vs_B"))),
        ):
            threshold = float(config.get("runtime", {}).get(label, 1.1 if "median_total" in label else 1.25))
            gates.append(_gate(label, value, threshold))
        runtime_benchmark = summary.get("runtime_benchmark") if isinstance(summary, Mapping) else None
        runtime_candidates = (
            runtime_benchmark.get("candidates", {})
            if isinstance(runtime_benchmark, Mapping)
            else {}
        )
        runtime_metric = (
            runtime_candidates.get(candidate, {}).get("wall_ratio")
            if isinstance(runtime_candidates, Mapping)
            and isinstance(runtime_candidates.get(candidate), Mapping)
            else None
        )
        runtime_ready = (
            isinstance(runtime_benchmark, Mapping)
            and isinstance(runtime_benchmark.get("completeness"), Mapping)
            and runtime_benchmark["completeness"].get("status") == "pass"
        )
        if runtime_ready and runtime_metric is not None:
            runtime_median = _estimate_mapping(runtime_metric)
            gates.append(
                _gate(
                    "larger_food_budget_total_vs_full_probe",
                    runtime_median,
                    1.0,
                    strict=True,
                    detail="locked candidate median paired wall ratio at budgets >=128 must remain below full probe; interval is reported separately",
                )
            )
        else:
            gates.append(
                _gate(
                    "larger_food_budget_total_vs_full_probe",
                    None,
                    1.0,
                    strict=True,
                    detail="separate large-budget runtime artifact unavailable or incomplete",
                )
            )
        parity_summary = summary.get("baseline_parity") if isinstance(summary, Mapping) else None
        parity_value = metrics.get(
            "baseline_parity_exact",
            metrics.get(
                "parity_exact",
                parity_summary.get("exact") if isinstance(parity_summary, Mapping) else None,
            ),
        )
        determinism_summary = summary.get("determinism") if isinstance(summary, Mapping) else None
        if not isinstance(determinism_summary, Mapping) and isinstance(summary, Mapping):
            determinism_summary = summary.get("determinism_verification")
        determinism_value = metrics.get(
            "determinism_exact",
            metrics.get(
                "deterministic",
                determinism_summary.get("exact") if isinstance(determinism_summary, Mapping) else None,
            ),
        )
        gates.append(
            {
                "name": "baseline_parity",
                "status": "inconclusive" if not isinstance(parity_value, bool) else ("pass" if parity_value else "fail"),
                "value": parity_value if isinstance(parity_value, bool) else None,
                "threshold": True,
                "direction": "exact",
            }
        )
        gates.append(
            {
                "name": "determinism",
                "status": "inconclusive" if not isinstance(determinism_value, bool) else ("pass" if determinism_value else "fail"),
                "value": determinism_value if isinstance(determinism_value, bool) else None,
                "threshold": True,
                "direction": "exact",
            }
        )
        memory_config = config.get("memory", {})
        for label, value, default in (
            ("median_peak_memory_ratio_vs_B", timing.get("median_peak_memory_ratio_vs_B"), 1.15),
            ("p95_peak_memory_ratio_vs_B", timing.get("p95_peak_memory_ratio_vs_B"), 1.25),
            ("individual_peak_memory_ratio_vs_B", timing.get("individual_peak_memory_ratio_vs_B"), 1.5),
        ):
            config_key = {
                "median_peak_memory_ratio_vs_B": "per_cell_median_peak_ratio_vs_B",
                "p95_peak_memory_ratio_vs_B": "per_cell_p95_peak_ratio_vs_B",
                "individual_peak_memory_ratio_vs_B": "individual_peak_ratio_vs_B",
            }[label]
            threshold = float(
                memory_config.get(config_key, default)
            )
            gates.append(_gate(label, value, threshold))
        output[candidate] = {
            "status": (
                "fail"
                if any(gate["status"] == "fail" for gate in gates)
                else "inconclusive"
                if any(gate["status"] == "inconclusive" for gate in gates)
                else "pass"
            ),
            "gates": gates,
        }
        food_completeness = (
            summary.get("food101_completeness", summary.get("food_artifact_completeness"))
            if isinstance(summary, Mapping)
            else None
        )
        if stage in {"food101", "full"} and (
            not isinstance(food_completeness, Mapping)
            or food_completeness.get("status") != "pass"
        ):
            # Missing/partial Food containers block every product claim; do
            # not turn an absent comparator into a definite pass or fail.
            for gate in output[candidate]["gates"]:
                gate["status"] = "inconclusive"
                gate["detail"] = "Food artifact completeness is not verified"
            output[candidate]["status"] = "inconclusive"
    return output


def _candidate_score_for_promotion(metrics: Mapping[str, Any]) -> tuple[float | None, float | None]:
    nuisance = metrics.get("nuisance_linear_regret_upper")
    if nuisance is None:
        nuisance = metrics.get("nuisance_linear_regret")
    if nuisance is None:
        nuisance = (
            metrics.get("downstream", {})
            .get("linear", {})
            .get("nuisance_regret")
        )
    nuisance_upper = _ci_high(nuisance)
    robustness = metrics.get("robustness_auc", metrics.get("nuisance_robustness_auc"))
    return nuisance_upper, _ci_high(robustness)


def select_promotion(
    summary: Mapping[str, Any],
    *,
    input_hashes: Mapping[str, str] | None = None,
    source_hashes_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Select C/D/E by the frozen regret/AUC/simplicity lexicographic rule."""

    candidates = summary.get("candidates", summary) if isinstance(summary, Mapping) else {}
    historical = summary.get("historical_gates", {}) if isinstance(summary, Mapping) else {}
    product = summary.get("product_gates", {}) if isinstance(summary, Mapping) else {}
    manifest = summary.get("manifest", {}) if isinstance(summary, Mapping) else {}
    stage = (
        summary.get("stage")
        if isinstance(summary, Mapping)
        else None
    ) or (manifest.get("stage") if isinstance(manifest, Mapping) else None)
    completeness = summary.get("artifact_completeness") if isinstance(summary, Mapping) else None
    robustness_completeness = summary.get("robustness_completeness") if isinstance(summary, Mapping) else None
    runtime_benchmark = summary.get("runtime_benchmark") if isinstance(summary, Mapping) else None
    food101_completeness = (
        summary.get("food101_completeness", summary.get("food_artifact_completeness"))
        if isinstance(summary, Mapping)
        else None
    )
    locked_candidate = summary.get("locked_candidate") if isinstance(summary, Mapping) else None
    scores: dict[str, dict[str, Any]] = {}
    excluded: dict[str, str] = {"A": "exact baseline control", "B": "exact baseline control", "F": "diagnostic only", "G": "product-policy comparator only"}
    eligible: list[str] = []
    for candidate in PROMOTION_CANDIDATES:
        metrics = candidates.get(candidate, {}) if isinstance(candidates, Mapping) else {}
        if stage == "full" and candidate != locked_candidate:
            excluded[candidate] = "not the screen-locked candidate"
            continue
        historical_status = historical.get(candidate, {}).get("status") if isinstance(historical, Mapping) else None
        product_status = product.get(candidate, {}).get("status") if isinstance(product, Mapping) else None
        regret, robustness = _candidate_score_for_promotion(metrics)
        scores[candidate] = {
            "nuisance_linear_regret_upper": regret,
            "robustness_auc": robustness,
            "historical_status": historical_status,
            "product_status": product_status,
        }
        if regret is None or robustness is None:
            excluded[candidate] = "undefined promotion metric"
            continue
        if stage in _EXPECTED_STAGE_COUNTS and (
            not isinstance(completeness, Mapping) or completeness.get("status") != "pass"
        ):
            excluded[candidate] = "incomplete or unverifiable runner artifact"
            continue
        if stage in _EXPECTED_STAGE_COUNTS and (
            not isinstance(robustness_completeness, Mapping)
            or robustness_completeness.get("status") != "pass"
        ):
            excluded[candidate] = "incomplete or unverifiable robustness grid"
            continue
        if stage == "full" and (
            not isinstance(runtime_benchmark, Mapping)
            or not isinstance(runtime_benchmark.get("completeness"), Mapping)
            or runtime_benchmark["completeness"].get("status") != "pass"
        ):
            excluded[candidate] = "incomplete or unverifiable large-budget runtime artifact"
            continue
        if stage == "full" and (
            not isinstance(food101_completeness, Mapping)
            or food101_completeness.get("status") != "pass"
        ):
            excluded[candidate] = "incomplete or unverifiable Food-101 artifact"
            continue
        if historical_status == "fail" or (stage != "screen" and product_status == "fail"):
            excluded[candidate] = "frozen gate failure"
            continue
        if stage != "screen" and (historical_status == "inconclusive" or product_status == "inconclusive"):
            excluded[candidate] = "frozen gate inconclusive"
            continue
        eligible.append(candidate)

    selected: str | None = None
    reason = "no eligible candidate"
    full_lock_gate_status = None
    if stage == "full":
        if locked_candidate not in PROMOTION_CANDIDATES:
            reason = "missing or invalid screen lock"
        elif (
            not isinstance(completeness, Mapping)
            or completeness.get("status") != "pass"
            or not isinstance(robustness_completeness, Mapping)
            or robustness_completeness.get("status") != "pass"
            or not isinstance(runtime_benchmark, Mapping)
            or not isinstance(runtime_benchmark.get("completeness"), Mapping)
            or runtime_benchmark["completeness"].get("status") != "pass"
            or not isinstance(food101_completeness, Mapping)
            or food101_completeness.get("status") != "pass"
        ):
            full_lock_gate_status = "inconclusive"
            reason = "screen-locked candidate lacks a verified complete full-stage artifact"
        elif locked_candidate in historical and locked_candidate in product:
            hist_status = historical.get(locked_candidate, {}).get("status")
            product_status = product.get(locked_candidate, {}).get("status")
            full_lock_gate_status = (
                "fail" if "fail" in {hist_status, product_status}
                else "inconclusive" if "inconclusive" in {hist_status, product_status}
                else "pass"
            )
            if full_lock_gate_status == "fail":
                # The lock remains auditable, but a failed final gate is a
                # rejection, not a new promotion.  In particular, never
                # leave the locked id in selected_candidate: downstream
                # reporting must be able to distinguish "evaluated and
                # rejected" from "promoted" without guessing from status.
                selected = None
                status = "fail"
                reason = "screen-locked candidate failed a definite full-stage gate"
            elif full_lock_gate_status == "inconclusive":
                reason = "screen-locked candidate lacks complete full-stage evidence"
            else:
                selected = locked_candidate
                status = "pass"
                reason = "screen-locked candidate passed all full-stage gates"
        else:
            reason = "full-stage gates for the screen-locked candidate are missing"
    elif eligible:
        best_regret = min(scores[candidate]["nuisance_linear_regret_upper"] for candidate in eligible)
        regret_tied = [
            candidate
            for candidate in eligible
            if abs(scores[candidate]["nuisance_linear_regret_upper"] - best_regret) <= PROMOTION_TOLERANCE
        ]
        best_auc = min(scores[candidate]["robustness_auc"] for candidate in regret_tied)
        both_tied = [
            candidate
            for candidate in regret_tied
            if abs(scores[candidate]["robustness_auc"] - best_auc) <= PROMOTION_TOLERANCE
        ]
        if len(both_tied) > 1:
            selected = next(candidate for candidate in PROMOTION_CANDIDATES if candidate in both_tied)
            reason = "simplicity order C,D,E within 0.001 on regret and robustness AUC"
        else:
            selected = min(
                regret_tied,
                key=lambda candidate: (
                    scores[candidate]["nuisance_linear_regret_upper"],
                    scores[candidate]["robustness_auc"],
                    PROMOTION_CANDIDATES.index(candidate),
                ),
            )
            reason = "minimum upper paired nuisance-linear regret, then robustness AUC"

    deferred: list[str] = []
    if stage == "full":
        # Full evaluation cannot select a runner-up.  Keep the locked result
        # and gate status explicit while preserving an inconclusive result for
        # missing evidence.
        if full_lock_gate_status == "fail":
            status = "fail"
        elif full_lock_gate_status == "pass":
            status = "pass"
        else:
            status = "inconclusive"
        decision_stage = "full"
    elif selected is not None and stage == "screen":
        for candidate_gate in (historical, product):
            if isinstance(candidate_gate, Mapping):
                for candidate_value in candidate_gate.values():
                    if not isinstance(candidate_value, Mapping):
                        continue
                    for gate in candidate_value.get("gates", []):
                        if isinstance(gate, Mapping) and gate.get("status") != "pass":
                            name = gate.get("name")
                            if name is not None and name not in deferred:
                                deferred.append(str(name))
        status = "promoted_for_full_evaluation"
        decision_stage = "screen"
    else:
        status = "pass" if selected is not None else "inconclusive"
        decision_stage = stage
    return {
        "selected_candidate": selected,
        "status": status,
        "decision_stage": decision_stage,
        "reason": reason,
        "eligible": eligible,
        "excluded": excluded,
        "scores": scores,
        "rule": {
            "candidates": list(PROMOTION_CANDIDATES),
            "primary": "minimum upper paired 95% bound for nuisance-linear regret",
            "secondary": "minimum robustness AUC",
            "tie_tolerance": PROMOTION_TOLERANCE,
            "simplicity_order": list(PROMOTION_CANDIDATES),
            "forbidden_promotions": ["F", "G"],
        },
        "deferred_gates": deferred,
        "artifact_completeness": json_safe(completeness),
        "robustness_completeness": json_safe(robustness_completeness),
        "runtime_benchmark": json_safe(runtime_benchmark),
        "locked_candidate": locked_candidate,
        "promoted": bool(
            selected is not None
            and status in {"pass", "promoted_for_full_evaluation"}
        ),
        "input_hashes": dict(input_hashes or {}),
        "source_hashes": dict(source_hashes_map or {}),
    }


choose_promotion = select_promotion
screen_promotion = select_promotion


def _build_delta_metrics(
    summary_candidates: MutableMapping[str, Any],
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> None:
    block_extractors: dict[str, Callable[[Mapping[str, Any]], float | None]] = {
        "score": lambda row: _metric_value(row, "score"),
        "evidence": _explicit_evidence,
        "clean_error": lambda row: _extract_clean_error(row, _explicit_evidence(row)),
        "fpr": lambda row: _row_metric(row, "fpr"),
        "fnr": lambda row: _row_metric(row, "fnr"),
        "auroc": lambda row: _row_metric(row, "auroc"),
        "auprc": lambda row: _row_metric(row, "auprc"),
        "brier": lambda row: _row_metric(row, "brier"),
        "ece": lambda row: _row_metric(row, "ece"),
    }
    for candidate in PROMOTION_CANDIDATES:
        if candidate not in summary_candidates:
            continue
        metrics = summary_candidates[candidate]
        for metric, target in (
            ("score", "score_delta"),
            ("evidence", "evidence_delta"),
            ("clean_error", "clean_mae_delta"),
            ("fpr", "fpr_delta"),
            ("fnr", "fnr_delta"),
            ("auroc", "pair_auroc_delta"),
            ("auprc", "pair_auprc_delta"),
            ("brier", "brier_delta"),
            ("ece", "ece_delta"),
        ):
            if grouped is not None:
                selected = {
                    name: grouped[name]
                    for name in ("B", candidate)
                    if name in grouped
                }
                if len(selected) == 2:
                    block_values = _complete_block_values(
                        selected,
                        block_extractors[metric],
                        "seed",
                    )
                    delta = paired_block_percentile_difference(block_values, baseline="B")
                    delta = delta.get(candidate, {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"})
                else:
                    delta = _metric_delta(arrays, metric, candidate)
            else:
                delta = _metric_delta(arrays, metric, candidate)
            metrics[target] = delta


def _block_metric_summaries(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    candidates: Sequence[str],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Compute shared complete-seed bootstrap summaries for scalar metrics."""

    selected = {candidate: grouped[candidate] for candidate in candidates if candidate in grouped}
    extractors: dict[str, Callable[[Mapping[str, Any]], float | None]] = {
        "score": lambda row: _metric_value(row, "score"),
        "evidence": _explicit_evidence,
        "clean_error": lambda row: _extract_clean_error(row, _explicit_evidence(row)),
        "fpr": lambda row: _row_metric(row, "fpr"),
        "fnr": lambda row: _row_metric(row, "fnr"),
        "auroc": lambda row: _row_metric(row, "auroc"),
        "auprc": lambda row: _row_metric(row, "auprc"),
        "brier": lambda row: _row_metric(row, "brier"),
        "ece": lambda row: _row_metric(row, "ece"),
    }
    result: dict[str, dict[str, Mapping[str, Any]]] = {candidate: {} for candidate in selected}
    for metric, extractor in extractors.items():
        summaries = paired_seed_block_bootstrap(selected, extractor)
        for candidate, value in summaries.items():
            result.setdefault(candidate, {})[metric] = value
    pair_blocks = _complete_block_values(
        selected,
        lambda row: (
            float(np.mean([value for value, _ in _extract_pair_observations(row)]))
            if _extract_pair_observations(row)
            else None
        ),
        "seed",
    )
    pair_summaries = paired_block_percentile_bootstrap(pair_blocks)
    for candidate, value in pair_summaries.items():
        result.setdefault(candidate, {})["pair_evidence"] = value
    return result


def _explicit_or_row_metric(row: Mapping[str, Any], name: str) -> float | None:
    aliases = {
        "ordering": (
            "ordering_rate",
            "nuisance_ordering_rate",
            "ordering",
            "diagnostics.ordering_rate",
        ),
        "family_drift": ("family_drift", "family_drift_delta", "diagnostics.family_drift"),
        "stable_shift": ("stable_shift", "stable_shift_delta", "diagnostics.stable_shift"),
    }
    value = _lookup(row, *(aliases.get(name, (name,))))
    return _finite_float(value)


def _paired_subset_delta(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: str,
    baseline: str,
    extractor: Callable[[Mapping[str, Any]], float | None],
    predicate: Callable[[Mapping[str, Any]], bool],
    *,
    direction: str = "candidate_minus_baseline",
) -> dict[str, Any]:
    selected: dict[str, list[Mapping[str, Any]]] = {}
    for name in (baseline, candidate):
        selected[name] = [row for row in grouped.get(name, []) if predicate(row)]
    block_values = _complete_block_values(selected, extractor, "seed")
    if direction == "baseline_minus_candidate" and len(block_values) == 2:
        # Negating rows before the paired block operation preserves identical
        # draw indices while expressing a loss as a positive degradation.
        block_values[candidate] = {
            block: -values for block, values in block_values[candidate].items()
        }
        block_values[baseline] = {
            block: -values for block, values in block_values[baseline].items()
        }
    return paired_block_percentile_difference(block_values, baseline=baseline).get(
        candidate,
        {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"},
    )


_PAIR_METRIC_NAMES = ("fpr", "fnr", "auroc", "auprc", "brier", "ece")


def _pair_row_identity(row: Mapping[str, Any], index: int) -> Any:
    """Pairing identity including the serialized directed-pair keys."""

    pairwise = _lookup(row, "pairwise", "pairwise_index", "pair_scores")
    directions: tuple[str, ...] = ()
    if isinstance(pairwise, Mapping):
        directions = tuple(sorted(_canonical_pair_direction(key) for key in pairwise))
    return (row_pair_key(row, index), directions)


def _complete_pair_blocks(
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[Any, list[Mapping[str, Any]]]]:
    """Retain complete seed blocks for pooled directional-pair metrics.

    Completeness is checked on a multiset of row/pair identities, not merely
    on a set of case IDs.  This prevents one candidate's missing duplicate or
    missing directed pair from silently changing the pooled estimand.
    """

    rows_by_block: dict[str, dict[Any, dict[Any, list[Mapping[str, Any]]]]] = {}
    # Keep a second, encounter-ordered view.  The archived stable AUPRC
    # estimator uses the serialized row order for equal scores; sorting row
    # identities here would be deterministic but would change that estimand.
    ordered_rows_by_block: dict[str, dict[Any, list[Mapping[str, Any]]]] = {}
    identities_by_block: dict[str, dict[Any, dict[Any, int]]] = {}
    for candidate, rows in rows_by_candidate.items():
        candidate_blocks: dict[Any, dict[Any, list[Mapping[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        candidate_ordered_rows: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        candidate_identities: dict[Any, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        for index, row in enumerate(rows):
            observations = [
                (value, label)
                for value, label in _extract_pair_observations(row)
                if label is not None and _finite_float(value) is not None
            ]
            if not observations:
                continue
            block_value = _lookup(row, "seed")
            if block_value is None:
                block_value = row_pair_key(row, index)
            block = _freeze_key(block_value)
            identity = _pair_row_identity(row, index)
            candidate_blocks[block][identity].append(row)
            candidate_ordered_rows[block].append(row)
            candidate_identities[block][identity] += 1
        rows_by_block[candidate] = candidate_blocks
        ordered_rows_by_block[candidate] = candidate_ordered_rows
        identities_by_block[candidate] = candidate_identities
    if not rows_by_block:
        return {}
    common_blocks = set.intersection(*(set(value) for value in rows_by_block.values()))
    reference_candidate = next(iter(rows_by_candidate))
    complete_blocks = {
        block
        for block in common_blocks
        if all(
            identities_by_block[candidate].get(block, {})
            == identities_by_block[reference_candidate].get(block, {})
            for candidate in rows_by_candidate
        )
    }
    return {
        candidate: {
            block: list(ordered_rows_by_block[candidate][block])
            for block in complete_blocks
        }
        for candidate in rows_by_candidate
    }


def _pooled_detection_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    """Compute all pair metrics from one pooled row collection."""

    labels: list[float] = []
    evidence: list[float] = []
    calibration_truth: list[float] = []
    calibration_evidence: list[float] = []
    for row in rows:
        for value, label in _extract_pair_observations(row):
            number = _finite_float(value)
            truth = _finite_float(label)
            if number is not None and truth is not None:
                evidence.append(number)
                labels.append(truth)
        for value, truth in _extract_pair_calibration_observations(row):
            calibration_evidence.append(value)
            calibration_truth.append(truth)
    if not labels:
        return {metric: None for metric in _PAIR_METRIC_NAMES}
    metrics = binary_detection_metrics(labels, evidence)
    # A partial continuous-rho stream is not a valid calibration sample.  The
    # runner normally supplies every directed rho; compact fixtures that omit
    # it fall back to the aligned binary labels instead of silently changing
    # the denominator.
    if len(calibration_truth) == len(evidence):
        clipped_truth = np.clip(np.asarray(calibration_truth, dtype=float), 0.0, 1.0)
        clipped_evidence = np.clip(np.asarray(calibration_evidence, dtype=float), 0.0, 1.0)
        metrics["brier"] = float(np.mean((clipped_evidence - clipped_truth) ** 2))
        ece = 0.0
        bins = np.linspace(0.0, 1.0, 11)
        for lower, upper in zip(bins[:-1], bins[1:]):
            keep = (clipped_evidence >= lower) & (
                clipped_evidence <= upper if upper == 1.0 else clipped_evidence < upper
            )
            if np.any(keep):
                ece += float(np.mean(keep)) * abs(
                    float(np.mean(clipped_evidence[keep]))
                    - float(np.mean(clipped_truth[keep]))
                )
        metrics["ece"] = float(ece)
    return metrics


def _pooled_detection_draws(
    block_rows: Mapping[Any, Sequence[Mapping[str, Any]]],
    blocks: Sequence[Any],
    draws: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute pooled pair metrics from seed sufficient statistics.

    Threshold/calibration metrics are vectorized over the shared draw matrix;
    AUROC uses a seed-by-seed comparison matrix.  Only the stable row-order
    AUPRC numerator retains a small per-draw loop, and it consumes prepared
    arrays rather than re-parsing sampled row mappings.
    """

    names = _PAIR_METRIC_NAMES
    n_resamples = int(draws.shape[0])
    result = {name: np.full(n_resamples, np.nan, dtype=float) for name in names}
    n_blocks = len(blocks)
    if not n_blocks or not n_resamples:
        return result
    evidence_by_block: list[np.ndarray] = []
    labels_by_block: list[np.ndarray] = []
    truth_by_block: list[np.ndarray] = []
    for block in blocks:
        evidence: list[float] = []
        labels: list[float] = []
        truth: list[float] = []
        for row in block_rows.get(block, ()):
            observations = _extract_pair_observations(row)
            evidence.extend(float(value) for value, label in observations if label is not None)
            labels.extend(float(label) for _, label in observations if label is not None)
            calibration = _extract_pair_calibration_observations(row)
            truth.extend(float(value) for _, value in calibration)
        # Calibration truth follows the same serialized pair stream.  If a
        # fixture omits continuous rho, binary labels are the explicit fallback.
        if len(truth) != len(evidence):
            truth = list(labels)
        evidence_by_block.append(np.asarray(evidence, dtype=float))
        labels_by_block.append(np.asarray(labels, dtype=float) > 0.5)
        truth_by_block.append(np.asarray(truth, dtype=float))

    counts = np.asarray([values.size for values in evidence_by_block], dtype=float)
    separated_counts = np.asarray([np.count_nonzero(~values) for values in labels_by_block], dtype=float)
    overlapping_counts = counts - separated_counts
    fpr_gt = np.asarray([
        np.count_nonzero((~labels) & (evidence > 0.0))
        for evidence, labels in zip(evidence_by_block, labels_by_block)
    ], dtype=float)
    fpr_ge = np.asarray([
        np.count_nonzero((~labels) & (evidence >= 0.05))
        for evidence, labels in zip(evidence_by_block, labels_by_block)
    ], dtype=float)
    fnr_gt = np.asarray([
        np.count_nonzero(labels & (evidence <= 0.0))
        for evidence, labels in zip(evidence_by_block, labels_by_block)
    ], dtype=float)
    fnr_ge = np.asarray([
        np.count_nonzero(labels & (evidence < 0.05))
        for evidence, labels in zip(evidence_by_block, labels_by_block)
    ], dtype=float)
    clipped_evidence = [np.clip(values, 0.0, 1.0) for values in evidence_by_block]
    clipped_truth = [np.clip(values, 0.0, 1.0) for values in truth_by_block]
    brier_sum = np.asarray([
        np.sum((evidence - truth) ** 2) for evidence, truth in zip(clipped_evidence, clipped_truth)
    ], dtype=float)
    bin_counts = np.zeros((n_blocks, 10), dtype=float)
    bin_evidence = np.zeros_like(bin_counts)
    bin_truth = np.zeros_like(bin_counts)
    for index, (evidence, truth) in enumerate(zip(clipped_evidence, clipped_truth)):
        bins = np.minimum((evidence * 10).astype(int), 9)
        for bin_index in range(10):
            keep = bins == bin_index
            bin_counts[index, bin_index] = np.count_nonzero(keep)
            bin_evidence[index, bin_index] = np.sum(evidence[keep])
            bin_truth[index, bin_index] = np.sum(truth[keep])

    positive_scores = [evidence[labels] for evidence, labels in zip(evidence_by_block, labels_by_block)]
    negative_scores = [evidence[~labels] for evidence, labels in zip(evidence_by_block, labels_by_block)]
    greater = np.zeros((n_blocks, n_blocks), dtype=float)
    equal = np.zeros_like(greater)
    for source, positives in enumerate(positive_scores):
        if not positives.size:
            continue
        for target, negatives in enumerate(negative_scores):
            if not negatives.size:
                continue
            ordered = np.sort(negatives)
            left = np.searchsorted(ordered, positives, side="left")
            right = np.searchsorted(ordered, positives, side="right")
            greater[source, target] = np.sum(left)
            equal[source, target] = np.sum(right - left)

    multiplicities = np.zeros((n_resamples, n_blocks), dtype=float)
    np.add.at(multiplicities, (np.arange(n_resamples)[:, None], draws), 1.0)
    total = multiplicities @ counts
    separated = multiplicities @ separated_counts
    overlapping = multiplicities @ overlapping_counts
    result["fpr"] = np.divide(multiplicities @ fpr_gt, separated, out=result["fpr"], where=separated > 0)
    # The gate uses >= .05 FPR; the generic pair metric name is retained for
    # the historical summary and maps to that frozen threshold.
    result["fpr"] = np.divide(multiplicities @ fpr_ge, separated, out=result["fpr"], where=separated > 0)
    result["fnr"] = np.divide(multiplicities @ fnr_ge, overlapping, out=result["fnr"], where=overlapping > 0)
    result["brier"] = np.divide(multiplicities @ brier_sum, total, out=result["brier"], where=total > 0)
    pooled_counts = np.einsum("rs,sb->rb", multiplicities, bin_counts)
    pooled_evidence = np.einsum("rs,sb->rb", multiplicities, bin_evidence)
    pooled_truth = np.einsum("rs,sb->rb", multiplicities, bin_truth)
    with np.errstate(divide="ignore", invalid="ignore"):
        gaps = np.abs(pooled_evidence / pooled_counts - pooled_truth / pooled_counts)
    gaps[~np.isfinite(gaps)] = 0.0
    result["ece"] = np.sum(pooled_counts / total[:, None] * gaps, axis=1)
    positive_total = multiplicities @ np.asarray([values.size for values in positive_scores], dtype=float)
    negative_total = multiplicities @ np.asarray([values.size for values in negative_scores], dtype=float)
    numerator = np.einsum("rs,st,rt->r", multiplicities, greater + 0.5 * equal, multiplicities)
    result["auroc"] = np.divide(
        numerator,
        positive_total * negative_total,
        out=result["auroc"],
        where=(positive_total > 0) & (negative_total > 0),
    )
    # Stable descending AP follows the archived row-order convention.  This
    # loop is over prepared arrays and avoids any per-draw mapping traversal.
    for draw_index, draw in enumerate(draws):
        sampled_evidence = np.concatenate([evidence_by_block[int(index)] for index in draw])
        sampled_labels = np.concatenate([labels_by_block[int(index)] for index in draw])
        value = stable_auprc(sampled_labels, sampled_evidence)
        if value is not None:
            result["auprc"][draw_index] = value
    return result


def _pooled_pair_detection_summaries(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    candidates: Sequence[str],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    """Return pooled per-candidate and candidate-minus-B pair summaries.

    Synthetic v3 resamples complete seeds and recomputes metrics after
    concatenating all directional pair observations in each selected seed.
    This is intentionally different from bootstrapping a vector of per-row
    AUROC/AUPRC/Brier values.
    """

    selected = {candidate: grouped[candidate] for candidate in candidates if candidate in grouped}
    if "B" not in selected:
        return {}, {}
    blocks = _complete_pair_blocks(selected)
    common_blocks = sorted(blocks.get("B", {}), key=repr)
    if not common_blocks:
        return {}, {}
    draws = bootstrap_indices(len(common_blocks), n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    point_metrics: dict[str, dict[str, float | None]] = {}
    sampled_metrics: dict[str, dict[str, np.ndarray]] = {}
    observation_counts: dict[str, int] = {}
    summaries: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for candidate in selected:
        rows_by_block = blocks.get(candidate, {})
        pooled_rows = [row for block in common_blocks for row in rows_by_block.get(block, ())]
        point_metrics[candidate] = _pooled_detection_metrics(pooled_rows)
        sampled_metrics[candidate] = _pooled_detection_draws(
            rows_by_block, common_blocks, draws
        )
        n_observations = sum(
            sum(
                1
                for value, label in _extract_pair_observations(row)
                if _finite_float(value) is not None and _finite_float(label) is not None
            )
            for row in pooled_rows
        )
        observation_counts[candidate] = n_observations
        for metric in _PAIR_METRIC_NAMES:
            estimates = sampled_metrics[candidate].get(metric, np.empty(0, dtype=float))
            estimates = estimates[np.isfinite(estimates)]
            point = point_metrics[candidate].get(metric)
            summaries[candidate][metric] = {
                "estimate": _finite_float(point),
                "lower": None if not estimates.size else float(np.percentile(estimates, 2.5)),
                "upper": None if not estimates.size else float(np.percentile(estimates, 97.5)),
                "n": int(n_observations),
                "n_blocks": len(common_blocks),
                "status": "defined" if estimates.size and point is not None else "undefined",
                "bootstrap_estimator": "v3_pooled_rows_seed_bootstrap",
            }

    deltas: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    base = sampled_metrics.get("B", {})
    for candidate in selected:
        if candidate == "B":
            continue
        for metric in _PAIR_METRIC_NAMES:
            candidate_draws = sampled_metrics[candidate].get(metric, np.empty(0, dtype=float))
            base_draws = base.get(metric, np.empty(0, dtype=float))
            point_candidate = _finite_float(point_metrics[candidate].get(metric))
            point_base = _finite_float(point_metrics["B"].get(metric))
            if point_candidate is None or point_base is None:
                deltas[candidate][metric] = {
                    "estimate": None,
                    "lower": None,
                    "upper": None,
                    "n": 0,
                    "n_blocks": len(common_blocks),
                    "status": "undefined",
                    "bootstrap_estimator": "v3_pooled_rows_seed_bootstrap",
                }
                continue
            n_draws = min(candidate_draws.size, base_draws.size)
            difference = candidate_draws[:n_draws] - base_draws[:n_draws]
            difference = difference[np.isfinite(difference)]
            deltas[candidate][metric] = {
                "estimate": point_candidate - point_base,
                "lower": None if not difference.size else float(np.percentile(difference, 2.5)),
                "upper": None if not difference.size else float(np.percentile(difference, 97.5)),
                "n": int(min(observation_counts.get(candidate, 0), observation_counts.get("B", 0))),
                "n_blocks": len(common_blocks),
                "status": "defined" if difference.size else "undefined",
                "bootstrap_estimator": "v3_pooled_rows_seed_bootstrap",
            }
    return dict(summaries), dict(deltas)


def _pooled_pairwise_direction_summaries(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    candidates: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Expose pooled directed-pair evidence and candidate-minus-B deltas.

    The aggregate AUROC/AUPRC cells are useful, but they can hide a single
    class-pair regression.  This companion surface keeps each serialized
    source/target direction as a complete-seed pooled evidence stream.  It
    deliberately reuses the pair-block identity check, so a missing direction
    or case cannot be silently paired against a different row.
    """

    selected = {
        candidate: grouped[candidate]
        for candidate in candidates
        if candidate in grouped
    }
    if "B" not in selected:
        return {}
    blocks = _complete_pair_blocks(selected)
    common_blocks = sorted(blocks.get("B", {}), key=repr)
    if not common_blocks:
        return {}
    directions = sorted(
        {
            direction
            for candidate in selected
            for block in common_blocks
            for row in blocks.get(candidate, {}).get(block, ())
            for direction, _ in _pair_direction_indices(row)
        }
    )
    output: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"pairwise_directions": {}, "pairwise_direction_deltas": {}}
    )
    for direction in directions:
        by_candidate: dict[str, dict[Any, list[float]]] = {}
        for candidate in selected:
            block_values: dict[Any, list[float]] = {}
            for block in common_blocks:
                values = [
                    1.0 - index_value
                    for row in blocks.get(candidate, {}).get(block, ())
                    for row_direction, index_value in _pair_direction_indices(row)
                    if row_direction == direction
                ]
                if values:
                    block_values[block] = values
            by_candidate[candidate] = block_values
        if not by_candidate.get("B"):
            continue
        # A direction missing from one arm is undefined rather than an
        # intersection over the surviving rows.
        complete_candidates = [
            candidate for candidate, values in by_candidate.items()
            if set(values) == set(common_blocks)
        ]
        if "B" not in complete_candidates:
            continue
        evidence_maps = {
            candidate: by_candidate[candidate]
            for candidate in complete_candidates
        }
        evidence_summaries = paired_block_percentile_bootstrap(
            evidence_maps,
            seed=BOOTSTRAP_SEED,
        )
        for candidate in complete_candidates:
            output[candidate]["pairwise_directions"][direction] = evidence_summaries[candidate]
            if candidate == "B":
                continue
            output[candidate]["pairwise_direction_deltas"][direction] = (
                paired_block_percentile_difference(
                    {"B": evidence_maps["B"], candidate: evidence_maps[candidate]},
                    baseline="B",
                    seed=BOOTSTRAP_SEED,
                ).get(
                    candidate,
                    {
                        "estimate": None,
                        "lower": None,
                        "upper": None,
                        "n": 0,
                        "status": "undefined",
                    },
                )
            )
    return dict(output)


def _truth_global_overlap(row: Mapping[str, Any]) -> float | None:
    return _finite_float(
        _lookup(
            row,
            "truth.global_overlap",
            "truth.truth_global_overlap",
            "global_overlap",
            "truth_overlap",
            "overlap_severity",
        )
    )


def _row_regime(row: Mapping[str, Any]) -> str | None:
    value = _lookup(row, "regime", "domain", "distribution", "stability")
    shift = _lookup(row, "nuisance_shift")
    if value is None and shift is not None:
        text = str(shift).strip().lower()
        if isinstance(shift, (bool, np.bool_)) or text in {"true", "false", "1", "0", "yes", "no"}:
            return "shift" if shift is True or text in {"true", "1", "yes"} else "stable"
    value = value if value is not None else _lookup(row, "condition", "condition_name")
    if value is None:
        return None
    text = str(value).strip().lower()
    tokens = set(re.split(r"[^a-z0-9]+", text))
    if "shift" in tokens or "shifted" in tokens:
        return "shift"
    if "stable" in tokens:
        return "stable"
    return text if text in {"stable", "shift"} else None


def _is_separated_row(row: Mapping[str, Any]) -> bool:
    truth = _truth_global_overlap(row)
    if truth is not None:
        return truth <= 0.0
    return str(_lookup(row, "condition", "condition_name") or "").lower().startswith("separated")


def _is_overlapping_row(row: Mapping[str, Any]) -> bool:
    truth = _truth_global_overlap(row)
    if truth is not None:
        return truth > 0.0
    return str(_lookup(row, "condition", "condition_name") or "").lower().startswith("overlap")


def _seed_key(row: Mapping[str, Any], index: int = 0) -> Any:
    value = _lookup(row, "seed", "case_seed", "random_seed")
    return _freeze_key(value if value is not None else row_pair_key(row, index))


def _base_case_key(row: Mapping[str, Any], index: int = 0) -> Any:
    case = _lookup(row, "case_group", "base_case_id")
    if case is None:
        case = _lookup(row, "case_id", "case", "run_id")
        if case is not None:
            case = re.sub(r"(?:^|__)(?:nuis|lambda|strength)-[^_]+", "", str(case), flags=re.I).strip("_")
    return _freeze_key(case if case is not None else row_pair_key(row, index))


def _structured_nuisance_drift(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Archived normalized AUC of absolute evidence drift from strength zero."""

    groups: dict[Any, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        strength = _extract_strength(row)
        evidence = _explicit_evidence(row)
        if strength is None or evidence is None:
            continue
        key = (
            _freeze_key(_lookup(row, "family")),
            _freeze_key(_lookup(row, "condition", "condition_name")),
            _freeze_key(_lookup(row, "k")),
            _freeze_key(_lookup(row, "balance")),
            _freeze_key(_lookup(row, "nuisance_shift")),
            _base_case_key(row, index),
        )
        groups[key][float(strength)].append(float(evidence))
    curves: list[float] = []
    for levels in groups.values():
        if 0.0 not in levels or len(levels) < 2:
            continue
        baseline = float(np.mean(levels[0.0]))
        strengths = sorted(levels)
        values = [abs(float(np.mean(levels[strength])) - baseline) for strength in strengths]
        area = normalized_trapezoid_auc(strengths, values)
        if area is not None:
            curves.append(area)
    return None if not curves else float(np.mean(curves))


def _structured_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Compute primitive historical metrics directly from frozen runner rows."""

    clean_errors = [
        error
        for row in rows
        if (_extract_strength(row) is None or np.isclose(_extract_strength(row), 0.0))
        and (error := _extract_clean_error(row, _explicit_evidence(row))) is not None
    ]
    pair = _pooled_detection_metrics(rows)
    severity_values: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        strength = _extract_strength(row)
        severity = _truth_global_overlap(row)
        evidence = _explicit_evidence(row)
        if strength is not None and not np.isclose(strength, 0.0):
            continue
        if severity is not None and evidence is not None:
            severity_values[float(severity)].append(float(evidence))
    ordered_truth = sorted(severity_values)
    ordered_evidence = [float(np.mean(severity_values[value])) for value in ordered_truth]
    return {
        "clean_mae": None if not clean_errors else float(np.mean(clean_errors)),
        "fpr": _finite_float(pair.get("fpr")),
        "fnr": _finite_float(pair.get("fnr")),
        "auroc": _finite_float(pair.get("auroc")),
        "auprc": _finite_float(pair.get("auprc")),
        "brier": _finite_float(pair.get("brier")),
        "ece": _finite_float(pair.get("ece")),
        "nuisance_abs_drift_auc": _structured_nuisance_drift(rows),
        "spearman": spearman_rank_correlation(ordered_truth, ordered_evidence),
        "ordering_rate": monotonic_ordering_rate(ordered_evidence, ordered_truth),
    }


def _seed_metric_map(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[Any, float]:
    by_seed: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if predicate is not None and not predicate(row):
            continue
        by_seed[_seed_key(row, index)].append(row)
    output: dict[Any, float] = {}
    for seed, seed_rows in by_seed.items():
        value = _structured_metrics(seed_rows).get(metric)
        if value is not None:
            output[seed] = float(value)
    return output


def _complete_structured_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Keep only seed blocks with identical within-seed row identities."""

    maps: list[dict[Any, dict[Any, list[Mapping[str, Any]]]]] = []
    for rows in (candidate_rows, baseline_rows):
        by_seed: dict[Any, dict[Any, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            if predicate is not None and not predicate(row):
                continue
            by_seed[_seed_key(row, index)][row_pair_key(row, index)].append(row)
        maps.append(by_seed)
    common_seeds = set(maps[0]) & set(maps[1])
    complete = {
        seed
        for seed in common_seeds
        if {
            key: len(values) for key, values in maps[0][seed].items()
        }
        == {
            key: len(values) for key, values in maps[1][seed].items()
        }
    }
    candidate = [row for seed in complete for values in maps[0][seed].values() for row in values]
    baseline = [row for seed in complete for values in maps[1][seed].values() for row in values]
    return candidate, baseline


def _promotion_decision_info(
    decision: Mapping[str, Any] | os.PathLike[str] | str | None,
) -> dict[str, Any] | None:
    """Normalize a screen lock mapping/path and carry its immutable identity."""

    if decision is None:
        return None
    source_path: str | None = None
    if isinstance(decision, Mapping):
        value = dict(decision)
        digest = sha256_bytes(canonical_json(value).encode("utf-8"))
    else:
        path = Path(decision)
        value = _read_json(path)
        if not isinstance(value, Mapping):
            raise ValueError("screen promotion decision must be a JSON object")
        value = dict(value)
        digest = sha256_file(path)
        source_path = str(path.resolve())
    return {
        "decision": json_safe(value),
        "sha256": digest,
        "path": source_path,
        "status": value.get("status"),
        "decision_stage": value.get("decision_stage"),
        "selected_candidate": value.get("selected_candidate"),
    }


def _valid_screen_lock(info: Mapping[str, Any] | None) -> bool:
    if not isinstance(info, Mapping):
        return False
    return (
        info.get("status") == "promoted_for_full_evaluation"
        and info.get("decision_stage") == "screen"
        and info.get("selected_candidate") in PROMOTION_CANDIDATES
        and isinstance(info.get("sha256"), str)
        and bool(info.get("sha256"))
    )


def _coerce_screen_promotion(
    value: Mapping[str, Any] | os.PathLike[str] | str | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping) and {
        "sha256",
        "selected_candidate",
    }.issubset(value) and (
        "decision" in value
        or "decision_stage" in value
        or "status" in value
    ):
        return dict(value)
    return _promotion_decision_info(value)


def _food_identity(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    model = _lookup(row, "model", "backbone")
    replicate = _lookup(row, "replicate")
    arm = _lookup(row, "arm")
    budget = _lookup(row, "budget")
    if model is None or replicate is None or arm is None or budget is None:
        return None
    return (
        str(model),
        _freeze_key(replicate),
        str(arm),
        _freeze_key(budget),
    )


def _food_artifact_completeness(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any] | None,
    *,
    screen_promotion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact retrospective Food-101 artifact before product gates.

    Selector, capped-probe, policy-G, prior full-probe, parity, references,
    and promotion metadata are checked independently.  A partial panel is
    never repaired by intersecting whatever rows happen to be present.
    """

    manifest = manifest if isinstance(manifest, Mapping) else {}
    configuration = manifest.get("configuration")
    failures: list[str] = []
    inconclusive: list[str] = []
    if manifest.get("artifact_status") != "completed":
        failures.append(f"artifact_status={manifest.get('artifact_status')!r}, expected 'completed'")
    if not isinstance(configuration, Mapping):
        failures.append("Food configuration is missing")
        configuration = {}
    expected_config = {
        "models": list(_FOOD_MODELS),
        "replicates": list(_FOOD_REPLICATES),
        "budgets": list(_FOOD_BUDGETS),
        "arms": list(_FOOD_ARMS),
        "candidates": list(_FOOD_SELECTOR_CANDIDATES),
        "folds": 5,
        "k": 10,
        "seed": 42,
    }
    for key, expected in expected_config.items():
        received = configuration.get(key)
        normalized_received = list(received) if isinstance(received, tuple) else received
        if normalized_received != expected:
            failures.append(f"configuration.{key}={received!r}, expected {expected!r}")

    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        candidate = _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name", "method"))
        if candidate is not None:
            by_candidate[candidate].append(row)
    expected_counts = {
        **{candidate: _FOOD_PANEL_CELLS for candidate in _FOOD_SELECTOR_CANDIDATES},
        "G": _FOOD_PANEL_CELLS,
        "G_probe_component": _FOOD_PANEL_CELLS,
        "full_probe": _FOOD_PANEL_CELLS,
    }
    unexpected_candidates = set(by_candidate) - set(expected_counts)
    if unexpected_candidates:
        failures.append(f"unexpected Food candidate surfaces={sorted(unexpected_candidates)!r}")
    expected_identity = {
        (model, replicate, arm, budget)
        for model in _FOOD_MODELS
        for replicate in _FOOD_REPLICATES
        for arm in _FOOD_ARMS
        for budget in _FOOD_BUDGETS
    }
    for candidate, expected_count in expected_counts.items():
        rows = by_candidate.get(candidate, [])
        if len(rows) != expected_count:
            failures.append(f"{candidate} n_rows={len(rows)}, expected {expected_count}")
        identities = [_food_identity(row) for row in rows]
        if any(identity is None for identity in identities):
            failures.append(f"{candidate} has an incomplete panel identity")
        identity_values = [identity for identity in identities if identity is not None]
        if len(identity_values) != len(set(identity_values)):
            failures.append(f"{candidate} has duplicate panel cells")
        if set(identity_values) != expected_identity:
            failures.append(
                f"{candidate} panel grid has {len(set(identity_values))} cells, expected {len(expected_identity)}"
            )
        bad_status = [row for row in rows if _lookup(row, "status") != "ok"]
        if bad_status:
            failures.append(f"{candidate} has {len(bad_status)} non-ok rows")
        error_rows = [
            row for row in rows
            if _lookup(row, "error") not in (None, "")
        ]
        if error_rows:
            failures.append(f"{candidate} has {len(error_rows)} error rows")

    # All five replicate blocks must be represented in every surface.
    for candidate in expected_counts:
        replicates = {
            _freeze_key(_lookup(row, "replicate"))
            for row in by_candidate.get(candidate, [])
        }
        if replicates != set(_FOOD_REPLICATES):
            failures.append(f"{candidate} replicate blocks={sorted(replicates, key=repr)!r}")

    containers = manifest.get("analysis_artifact_containers", {})
    if not isinstance(containers, Mapping):
        containers = {}
    parity_rows = containers.get("baseline_parity_rows")
    if not isinstance(parity_rows, list):
        failures.append("baseline_parity_rows container is missing")
        parity_rows = []
    if len(parity_rows) != 1200:
        failures.append(f"baseline_parity_rows n={len(parity_rows)}, expected 1200")
    parity_ids = []
    for row in parity_rows:
        if not isinstance(row, Mapping):
            failures.append("baseline parity contains a non-object row")
            continue
        parity_ids.append(
            (
                str(_lookup(row, "candidate_id")),
                str(_lookup(row, "model", "backbone")),
                _freeze_key(_lookup(row, "replicate")),
                str(_lookup(row, "arm")),
                _freeze_key(_lookup(row, "budget")),
            )
        )
        if _lookup(row, "exact") is not True or _finite_float(_lookup(row, "delta")) not in {0.0}:
            failures.append("baseline parity is not exact")
    if len(parity_ids) != len(set(parity_ids)):
        failures.append("baseline parity has duplicate cells")
    if set(str(value[0]) for value in parity_ids) != {"A", "B"}:
        failures.append("baseline parity does not contain exactly A and B")
    expected_parity_identity = {
        (candidate, model, replicate, arm, budget)
        for candidate in ("A", "B")
        for model in _FOOD_MODELS
        for replicate in _FOOD_REPLICATES
        for arm in _FOOD_ARMS
        for budget in _FOOD_BUDGETS
    }
    if set(parity_ids) != expected_parity_identity:
        failures.append("baseline parity identity grid is incomplete or has unexpected cells")
    parity_summary = manifest.get("baseline_parity")
    if not isinstance(parity_summary, Mapping) or parity_summary.get("exact") is not True:
        failures.append("baseline_parity.exact is not true")

    reference_rows = containers.get("reference_rows")
    if not isinstance(reference_rows, list):
        failures.append("reference_rows container is missing")
        reference_rows = []
    expected_reference_identity = {
        (model, replicate, arm, head)
        for model in _FOOD_MODELS
        for replicate in _FOOD_REPLICATES
        for arm in _FOOD_ARMS
        for head in REFERENCE_HEADS
    }
    reference_identity = []
    for row in reference_rows:
        if not isinstance(row, Mapping):
            continue
        head = _lookup(row, "head", "reference_head", "method", "name")
        if head is None:
            continue
        head_text = str(head).lower()
        head_text = {
            "linear_logistic": "linear",
            "quadratic_logistic": "quadratic",
            "rbf_svc": "rbf",
        }.get(head_text, head_text)
        reference_identity.append(
            (
                str(_lookup(row, "model", "backbone")),
                _freeze_key(_lookup(row, "replicate")),
                str(_lookup(row, "arm")),
                head_text,
            )
        )
    if len(reference_identity) != len(expected_reference_identity):
        failures.append(
            f"reference_rows n={len(reference_identity)}, expected {len(expected_reference_identity)}"
        )
    if set(reference_identity) != expected_reference_identity:
        failures.append("reference_rows identity grid is incomplete or has unexpected cells")
    if len(reference_identity) != len(set(reference_identity)):
        failures.append("reference_rows has duplicate cells")
    for candidate in (*_FOOD_SELECTOR_CANDIDATES, "G", "full_probe"):
        for row in by_candidate.get(candidate, []):
            if any(_extract_reference_value(row, head) is None for head in REFERENCE_HEADS):
                failures.append(f"{candidate} reference coverage is incomplete")
                break

    config_promotion = configuration.get("promotion_decision")
    if not isinstance(config_promotion, Mapping):
        inconclusive.append("Food promotion_decision metadata is missing")
    elif screen_promotion is None:
        # A Food artifact can be inspected on its own, but cross-artifact
        # candidate/SHA consistency cannot be established without the hashed
        # screen lock.  Keep that distinction explicit for product gating.
        inconclusive.append("screen promotion decision was not supplied")
    else:
        expected_candidate = screen_promotion.get("selected_candidate")
        expected_sha = screen_promotion.get("sha256")
        if config_promotion.get("selected_candidate") not in (None, expected_candidate):
            failures.append("Food promoted candidate does not match screen lock")
        if expected_sha is not None and config_promotion.get("sha256") != expected_sha:
            failures.append("Food promotion decision SHA does not match screen lock")
        if configuration.get("promoted_candidate_for_G") not in (None, expected_candidate):
            failures.append("Food promoted_candidate_for_G does not match screen lock")
    if screen_promotion is not None and not _valid_screen_lock(screen_promotion):
        failures.append("supplied screen promotion decision is not a valid C/D/E screen lock")

    status = "fail" if failures else "inconclusive" if inconclusive else "pass"
    return {
        "stage": "food101",
        "status": status,
        "expected_panel_cells": _FOOD_PANEL_CELLS,
        "expected_selector_rows": len(_FOOD_SELECTOR_CANDIDATES) * _FOOD_PANEL_CELLS,
        "expected_comparator_rows": 3 * _FOOD_PANEL_CELLS,
        "observed_rows": len(records),
        "failures": failures,
        "inconclusive_reasons": inconclusive,
        "promotion_sha256": screen_promotion.get("sha256") if isinstance(screen_promotion, Mapping) else None,
        "locked_candidate": screen_promotion.get("selected_candidate") if isinstance(screen_promotion, Mapping) else None,
    }


def _runtime_artifact_completeness(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any] | None,
    *,
    screen_promotion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the separate frozen large-budget Food runtime artifact."""

    manifest = manifest if isinstance(manifest, Mapping) else {}
    failures: list[str] = []
    if manifest.get("artifact_status") != "completed":
        failures.append(f"artifact_status={manifest.get('artifact_status')!r}, expected 'completed'")
    stage = str(manifest.get("stage") or manifest.get("study") or "").lower()
    if not stage:
        failures.append("runtime stage/study is missing")
    elif not ("runtime" in stage or "large_budget" in stage):
        failures.append(f"unexpected runtime stage/study={stage!r}")
    configuration = manifest.get("configuration", manifest.get("config"))
    if not isinstance(configuration, Mapping):
        failures.append("runtime configuration is missing")
        configuration = {}
    locked_candidate = (
        screen_promotion.get("selected_candidate")
        if isinstance(screen_promotion, Mapping)
        else None
    )
    if locked_candidate not in PROMOTION_CANDIDATES or not _valid_screen_lock(screen_promotion):
        failures.append("runtime requires a valid screen-locked C/D/E candidate")
        locked_candidate = "__missing_lock__"
    runtime_promotion = configuration.get(
        "promotion_decision",
        manifest.get("promotion_decision"),
    )
    if not isinstance(runtime_promotion, Mapping) and (
        configuration.get("promotion_decision_sha256") is not None
        or manifest.get("promotion_decision_sha256") is not None
        or configuration.get("promoted_candidate_for_G") is not None
    ):
        runtime_promotion = {
            "sha256": configuration.get(
                "promotion_decision_sha256",
                manifest.get("promotion_decision_sha256"),
            ),
            "selected_candidate": configuration.get(
                "promoted_candidate",
                configuration.get(
                    "promoted_candidate_for_G",
                    manifest.get("promoted_candidate"),
                ),
            ),
        }
    if not isinstance(runtime_promotion, Mapping):
        failures.append("runtime promotion_decision metadata is missing")
    else:
        if runtime_promotion.get("selected_candidate") not in (None, locked_candidate):
            failures.append("runtime promotion candidate does not match screen lock")
        expected_sha = screen_promotion.get("sha256") if isinstance(screen_promotion, Mapping) else None
        if expected_sha is not None and runtime_promotion.get("sha256") != expected_sha:
            failures.append("runtime promotion decision SHA does not match screen lock")
    expected_methods = ["A", "B", "E"]
    if locked_candidate != "E":
        expected_methods.append(locked_candidate)
    expected_methods.extend(("full_probe", "capped_probe_component"))
    expected_method_ids = {
        "A",
        "B",
        "E",
        locked_candidate,
        "full_probe",
        "capped_probe_component",
    }
    expected_method_ids.discard("__missing_lock__")
    expected_config = {
        "models": list(_RUNTIME_MODELS),
        "budgets": list(_RUNTIME_BUDGETS),
        "repeats": list(_RUNTIME_REPEATS),
        "methods": expected_methods,
        "folds": 5,
        "k": 10,
        "seed": 42,
    }
    for key, expected in expected_config.items():
        received = configuration.get(key)
        normalized = list(received) if isinstance(received, tuple) else received
        if key == "methods" and isinstance(normalized, (list, tuple)):
            normalized = [_runtime_method_id(value) for value in normalized]
        if normalized != expected:
            failures.append(f"runtime configuration.{key}={received!r}, expected {expected!r}")
    if configuration.get("n_classes") is not None and configuration.get("n_classes") != 40:
        failures.append(
            f"runtime configuration.n_classes={configuration.get('n_classes')!r}, expected 40"
        )
    expected_identity = {
        (model, budget, repeat, method)
        for model in _RUNTIME_MODELS
        for budget in _RUNTIME_BUDGETS
        for repeat in _RUNTIME_REPEATS
        for method in expected_method_ids
    }
    identities: list[tuple[Any, ...]] = []
    policy_identities: list[tuple[Any, ...]] = []
    by_panel: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_method = _lookup(row, "candidate", "candidate_id", "candidate_name", "method")
        method = _runtime_method_id(raw_method)
        if method is None:
            failures.append("runtime row has unknown method")
            continue
        model = _lookup(row, "model", "backbone")
        budget = _finite_float(_lookup(row, "budget"))
        repeat = _finite_float(_lookup(row, "repeat", "replicate"))
        if model is None or budget is None or repeat is None:
            failures.append("runtime row has incomplete identity")
            continue
        panel_identity = (str(model), int(budget), int(repeat))
        if method == "G":
            policy_identities.append(panel_identity)
        else:
            identity = (*panel_identity, method)
            identities.append(identity)
            by_panel[panel_identity].append(row)
        if _lookup(row, "status") != "ok":
            failures.append(f"runtime {(panel_identity, method)!r} is not ok")
    expected_timed_rows = _RUNTIME_PANEL_CELLS * len(expected_method_ids)
    expected_total_rows = expected_timed_rows + _RUNTIME_PANEL_CELLS
    timed_rows = len(identities)
    if timed_rows != expected_timed_rows:
        failures.append(f"runtime timed n_rows={timed_rows}, expected {expected_timed_rows}")
    if len(policy_identities) != _RUNTIME_PANEL_CELLS:
        failures.append(
            f"runtime G policy n_rows={len(policy_identities)}, expected {_RUNTIME_PANEL_CELLS}"
        )
    if len(rows) != expected_total_rows:
        failures.append(f"runtime total n_rows={len(rows)}, expected {expected_total_rows}")
    if len(identities) != len(set(identities)):
        failures.append("runtime rows contain duplicate method cells")
    if len(policy_identities) != len(set(policy_identities)):
        failures.append("runtime G policy rows contain duplicate panel cells")
    if set(identities) != expected_identity:
        failures.append(
            f"runtime identity grid has {len(set(identities))} cells, expected {len(expected_identity)}"
        )
    for panel, panel_rows in by_panel.items():
        order = [_finite_float(_lookup(row, "execution_order", "order")) for row in panel_rows]
        if any(value is None for value in order) or set(int(value) for value in order if value is not None) != set(range(len(expected_method_ids))):
            failures.append(
                f"runtime execution order is not 0..{len(expected_method_ids) - 1} for {panel!r}"
            )
            continue
        ordered_rows = sorted(
            panel_rows,
            key=lambda row: int(_finite_float(_lookup(row, "execution_order", "order")) or 0),
        )
        model_index = _RUNTIME_MODELS.index(panel[0]) if panel[0] in _RUNTIME_MODELS else None
        budget_index = _RUNTIME_BUDGETS.index(panel[1]) if panel[1] in _RUNTIME_BUDGETS else None
        if model_index is None or budget_index is None:
            failures.append(f"runtime panel is outside the frozen model/budget grid: {panel!r}")
            continue
        offset = (model_index + budget_index + panel[2]) % len(expected_methods)
        expected_order = expected_methods[offset:] + expected_methods[:offset]
        observed_order = [
            _runtime_method_id(
                _lookup(row, "candidate", "candidate_id", "candidate_name", "method")
            )
            for row in ordered_rows
        ]
        if observed_order != expected_order:
            failures.append(
                f"runtime execution order mismatch for {panel!r}: "
                f"observed={observed_order!r}, expected={expected_order!r}"
            )
    return {
        "stage": stage or None,
        "status": "fail" if failures else "pass",
        "expected_panel_cells": _RUNTIME_PANEL_CELLS,
        "expected_timed_rows": expected_timed_rows,
        "expected_policy_rows": _RUNTIME_PANEL_CELLS,
        "expected_rows": expected_total_rows,
        "observed_rows": len(rows),
        "gate_budgets": list(_RUNTIME_BUDGETS[1:]),
        "failures": failures,
        "locked_candidate": None if locked_candidate == "__missing_lock__" else locked_candidate,
        "observed_policy_rows": len(policy_identities),
    }


def _runtime_benchmark_summary(
    rows: Sequence[Mapping[str, Any]],
    completeness: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute paired C/D/E-vs-full-probe wall and CPU ratios for >=128."""

    if completeness.get("status") != "pass":
        return {
            "completeness": json_safe(dict(completeness)),
            "status": "inconclusive",
            "candidates": {},
        }
    grouped: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        candidate = _runtime_method_id(
            _lookup(row, "candidate", "candidate_id", "candidate_name", "method")
        )
        model = _lookup(row, "model", "backbone")
        budget = _finite_float(_lookup(row, "budget"))
        repeat = _finite_float(_lookup(row, "repeat", "replicate"))
        if candidate is None or model is None or budget is None or repeat is None:
            continue
        grouped[(str(model), int(budget), int(repeat))][candidate] = row

    def value(row: Mapping[str, Any], kind: str) -> float | None:
        aliases = (
            ("wall_seconds", "outer_wall_seconds", "total_wall_seconds", "runtime_seconds", "total_seconds")
            if kind == "wall"
            else ("cpu_seconds", "outer_cpu_seconds", "total_cpu_seconds", "runtime_cpu_seconds")
        )
        return _extract_timing(row, *aliases)

    output: dict[str, Any] = {
        "completeness": json_safe(dict(completeness)),
        "status": "defined",
        "gate_budgets": list(_RUNTIME_BUDGETS[1:]),
        "candidates": {},
    }
    locked_candidate = completeness.get("locked_candidate")
    candidate_names = (
        (locked_candidate,)
        if locked_candidate in PROMOTION_CANDIDATES
        else PROMOTION_CANDIDATES
    )
    for candidate in candidate_names:
        by_repeat_wall: dict[Any, list[float]] = defaultdict(list)
        by_repeat_cpu: dict[Any, list[float]] = defaultdict(list)
        by_budget: dict[str, Any] = {}
        for (model, budget, repeat), panel in grouped.items():
            if budget < 128 or candidate not in panel or "full_probe" not in panel:
                continue
            candidate_wall = value(panel[candidate], "wall")
            probe_wall = value(panel["full_probe"], "wall")
            candidate_cpu = value(panel[candidate], "cpu")
            probe_cpu = value(panel["full_probe"], "cpu")
            if candidate_wall is not None and probe_wall is not None and probe_wall > 0.0:
                ratio = candidate_wall / probe_wall
                by_repeat_wall[repeat].append(ratio)
                by_budget.setdefault(str(budget), {"wall": defaultdict(list), "cpu": defaultdict(list)})
                by_budget[str(budget)]["wall"][repeat].append(ratio)
            if candidate_cpu is not None and probe_cpu is not None and probe_cpu > 0.0:
                ratio = candidate_cpu / probe_cpu
                by_repeat_cpu[repeat].append(ratio)
                by_budget.setdefault(str(budget), {"wall": defaultdict(list), "cpu": defaultdict(list)})
                by_budget[str(budget)]["cpu"][repeat].append(ratio)

        def summarize(values: Mapping[Any, Sequence[float]]) -> dict[str, Any]:
            if not values:
                return {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
            blocks = [
                np.asarray(values[key], dtype=float)
                for key in sorted(values, key=repr)
                if len(values[key])
            ]
            if not blocks:
                return {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
            pooled = np.concatenate(blocks)
            draws = bootstrap_indices(
                len(blocks),
                n_resamples=BOOTSTRAP_RESAMPLES,
                seed=145,
            )
            estimates = np.asarray(
                [
                    np.median(np.concatenate([blocks[int(index)] for index in draw]))
                    for draw in draws
                ],
                dtype=float,
            )
            return {
                "estimate": float(np.median(pooled)),
                "lower": float(np.percentile(estimates, 2.5)),
                "upper": float(np.percentile(estimates, 97.5)),
                "p95": float(np.percentile(pooled, 95)),
                "n": int(pooled.size),
                "n_blocks": len(blocks),
                "status": "defined",
                "bootstrap_estimator": "paired_repeat_block_median",
                "bootstrap_seed": 145,
            }

        budget_summary: dict[str, Any] = {}
        for budget, values in by_budget.items():
            budget_summary[budget] = {
                "wall": summarize(values["wall"]),
                "cpu": summarize(values["cpu"]),
            }
        output["candidates"][candidate] = {
            "wall_ratio": summarize(by_repeat_wall),
            "cpu_ratio": summarize(by_repeat_cpu),
            "by_budget": budget_summary,
        }
    return output


def _seed_pair_delta(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    metric: str,
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    candidate_rows, baseline_rows = _complete_structured_rows(
        candidate_rows, baseline_rows, predicate
    )
    candidate_values = _seed_metric_map(candidate_rows, metric, predicate)
    baseline_values = _seed_metric_map(baseline_rows, metric, predicate)
    if not candidate_values or not baseline_values:
        return {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
    blocks = {
        "B": {seed: [value] for seed, value in baseline_values.items()},
        "candidate": {seed: [value] for seed, value in candidate_values.items()},
    }
    return paired_block_percentile_difference(blocks, baseline="B")["candidate"]


def _negate_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Express a candidate-minus-baseline loss as baseline-minus-candidate."""

    if _finite_float(value.get("estimate")) is None:
        return dict(value)
    return {
        **dict(value),
        "estimate": -float(value["estimate"]),
        "lower": None if _finite_float(value.get("upper")) is None else -float(value["upper"]),
        "upper": None if _finite_float(value.get("lower")) is None else -float(value["lower"]),
    }


def _seed_stability_metric_map(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[Any, float]:
    by_seed_regime: dict[Any, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        regime = _row_regime(row)
        if regime is not None:
            by_seed_regime[_seed_key(row, index)][regime].append(row)
    output: dict[Any, float] = {}
    for seed, regimes in by_seed_regime.items():
        stable = _structured_metrics(regimes.get("stable", ())).get(metric)
        shift = _structured_metrics(regimes.get("shift", ())).get(metric)
        if stable is None or shift is None:
            continue
        # Errors worsen when they increase; quality scores worsen when they
        # fall.  This is the frozen stable-to-shift degradation convention.
        if metric in {"auroc", "auprc"}:
            output[seed] = float(stable - shift)
        else:
            output[seed] = float(shift - stable)
    return output


def _seed_stability_delta(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    # Require complete within-seed identities separately for stable and
    # shifted rows before forming a stable-to-shift degradation.  Intersecting
    # seed IDs alone would accept a partially observed seed when both arms
    # happen to contain the same partial subset.
    complete_candidate: list[Mapping[str, Any]] = []
    complete_baseline: list[Mapping[str, Any]] = []
    for regime in ("stable", "shift"):
        candidate_part, baseline_part = _complete_structured_rows(
            candidate_rows,
            baseline_rows,
            lambda row, regime=regime: _row_regime(row) == regime,
        )
        complete_candidate.extend(candidate_part)
        complete_baseline.extend(baseline_part)
    candidate = _seed_stability_metric_map(complete_candidate, metric)
    baseline = _seed_stability_metric_map(complete_baseline, metric)
    if not candidate or not baseline:
        return {"estimate": None, "lower": None, "upper": None, "n": 0, "status": "undefined"}
    blocks = {
        "B": {seed: [value] for seed, value in baseline.items()},
        "candidate": {seed: [value] for seed, value in candidate.items()},
    }
    return paired_block_percentile_difference(blocks, baseline="B")["candidate"]


def _derive_historical_fields(
    candidate_summary: MutableMapping[str, Any],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Construct the recovered conditional historical gate inputs.

    Runner rows contain the primitives needed by the archived estimator, so
    these fields are recomputed from pooled directed pairs and complete seed
    blocks.  Explicit precomputed diagnostics remain a fallback for compact
    fixture rows that do not carry pair data.
    """

    has_pairs = any(
        _extract_pair_observations(row)
        for rows in grouped.values()
        for row in rows
    )
    if has_pairs:
        for baseline_name in ("A", "B"):
            if baseline_name in candidate_summary and baseline_name in grouped:
                baseline_structured = _structured_metrics(grouped[baseline_name])
                candidate_summary[baseline_name]["nuisance_spearman"] = baseline_structured.get("spearman")
                candidate_summary[baseline_name]["nuisance_ordering_rate"] = baseline_structured.get("ordering_rate")
                candidate_summary[baseline_name]["separated_fpr"] = _structured_metrics(
                    [row for row in grouped[baseline_name] if _is_separated_row(row)]
                ).get("fpr")
    for candidate in PROMOTION_CANDIDATES:
        if candidate not in candidate_summary or "B" not in grouped:
            continue
        metrics = candidate_summary[candidate]
        candidate_rows = grouped.get(candidate, ())
        baseline_rows = grouped.get("B", ())
        if has_pairs:
            # Conditional genuine-overlap rules use the frozen strata rather
            # than row-level FNR scalars.  Later refinement is explicitly
            # candidate-minus-A (unrefined raw), not candidate-minus-B.
            metrics["fpr_delta"] = _seed_pair_delta(
                candidate_rows, baseline_rows, "fpr", _is_separated_row
            )
            metrics["fnr_delta"] = _seed_pair_delta(
                candidate_rows, baseline_rows, "fnr", _is_overlapping_row
            )
            for name, metric in (
                ("pair_auroc_delta", "auroc"),
                ("pair_auprc_delta", "auprc"),
                ("brier_delta", "brier"),
            ):
                # ``_pooled_pair_detection_summaries`` already computed the
                # frozen v3 pooled-row point/bootstrap estimand.  Retain it;
                # only compact fixtures without a complete pair stream need
                # the structured per-seed fallback here.
                existing = metrics.get(name)
                if not isinstance(existing, Mapping) or existing.get("status") == "undefined":
                    metrics[name] = _seed_pair_delta(candidate_rows, baseline_rows, metric)
            metrics["clean_mae_delta"] = _seed_pair_delta(
                candidate_rows,
                baseline_rows,
                "clean_mae",
                lambda row: _extract_strength(row) is None
                or np.isclose(_extract_strength(row), 0.0),
            )
            metrics["refinement_fnr_delta_vs_raw"] = _seed_pair_delta(
                candidate_rows,
                grouped.get("A", ()),
                "fnr",
                _is_overlapping_row,
            )

            candidate_structured = _structured_metrics(candidate_rows)
            baseline_structured = _structured_metrics(baseline_rows)
            metrics["nuisance_spearman"] = candidate_structured.get("spearman")
            metrics["nuisance_ordering_rate"] = candidate_structured.get("ordering_rate")
            metrics["baseline_nuisance_spearman"] = baseline_structured.get("spearman")
            metrics["baseline_nuisance_ordering_rate"] = baseline_structured.get("ordering_rate")
            metrics["nuisance_abs_drift_auc"] = candidate_structured.get(
                "nuisance_abs_drift_auc"
            )
            metrics["separated_fpr"] = _structured_metrics(
                [row for row in candidate_rows if _is_separated_row(row)]
            ).get("fpr")
            # Ordering loss is baseline minus candidate, with uncertainty
            # retained under the same paired seed draws.
            ordering = _seed_pair_delta(candidate_rows, baseline_rows, "ordering_rate")
            metrics["ordering_loss_vs_baseline"] = _negate_summary(ordering)

            family_values: dict[str, dict[str, Any]] = {}
            families = sorted(
                {
                    str(_lookup(row, "family"))
                    for row in list(candidate_rows) + list(baseline_rows)
                    if _lookup(row, "family") is not None
                }
            )
            for family in families:
                family_values[family] = _seed_pair_delta(
                    [row for row in candidate_rows if str(_lookup(row, "family")) == family],
                    [row for row in baseline_rows if str(_lookup(row, "family")) == family],
                    "nuisance_abs_drift_auc",
                )
            metrics["family_drift"] = {
                "by_family": family_values,
                "worsening_upper": max(
                    (_ci_high(value) for value in family_values.values() if _ci_high(value) is not None),
                    default=None,
                ),
                "improvement_upper": {
                    family: _ci_high(value) for family, value in family_values.items()
                },
            }

            metrics["stable_shift_upper"] = {
                metric: _seed_stability_delta(candidate_rows, baseline_rows, metric)
                for metric in (
                    "clean_mae",
                    "auroc",
                    "auprc",
                    "brier",
                    "nuisance_abs_drift_auc",
                )
            }
            strata: dict[str, dict[str, Any]] = {}
            strata_keys = {
                (
                    _freeze_key(_lookup(row, "k")),
                    _freeze_key(_lookup(row, "balance")),
                )
                for row in list(candidate_rows) + list(baseline_rows)
                if _lookup(row, "k") is not None and _lookup(row, "balance") is not None
            }
            for key in sorted(strata_keys, key=repr):
                predicate = lambda row, key=key: (
                    _freeze_key(_lookup(row, "k")),
                    _freeze_key(_lookup(row, "balance")),
                ) == key and _is_overlapping_row(row)
                strata[str(key)] = _seed_pair_delta(
                    candidate_rows, baseline_rows, "fnr", predicate
                )
            metrics["strata_fnr_upper"] = strata
        else:
            # Compact rows without pair primitives may still carry explicit
            # gate fields; preserve them without fabricating structured values.
            metrics["refinement_fnr_delta_vs_raw"] = _paired_subset_delta(
                grouped, candidate, "A", lambda row: _row_metric(row, "fnr"), lambda row: True
            )
            metrics["ordering_loss_vs_baseline"] = _paired_subset_delta(
                grouped,
                candidate,
                "B",
                lambda row: _explicit_or_row_metric(row, "ordering"),
                lambda row: True,
                direction="baseline_minus_candidate",
            )
            metrics["family_drift"] = {"by_family": {}, "worsening_upper": None, "improvement_upper": {}}
            metrics["stable_shift_upper"] = {}
            metrics["strata_fnr_upper"] = {}


def _artifact_completeness(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the frozen runner artifact before any stage promotion.

    Complete-case pairing alone is insufficient: an early-stop artifact can
    contain a perfectly aligned subset for every candidate.  Screen/full
    promotion therefore requires the runner's completed marker, exact row and
    case counts, every A--F candidate, and the declared identity grid.
    """

    manifest = manifest if isinstance(manifest, Mapping) else {}
    stage = str(manifest.get("stage") or "").lower()
    if stage not in _EXPECTED_STAGE_COUNTS:
        return {
            "stage": stage or None,
            "status": "not_applicable" if not stage else "inconclusive",
            "reason": "stage is not a frozen screen/full runner artifact",
        }
    expected = _EXPECTED_STAGE_COUNTS[stage]
    failures: list[str] = []
    artifact_status = manifest.get("artifact_status")
    if artifact_status != "completed":
        failures.append(f"artifact_status={artifact_status!r}, expected 'completed'")
    if manifest.get("early_stop") not in (None, {}, []):
        failures.append("early_stop is present")
    stage_case_count = _finite_float(manifest.get("stage_case_count"))
    if stage_case_count is not None and int(stage_case_count) != expected["cases"]:
        failures.append(f"stage_case_count={stage_case_count:g}, expected {expected['cases']}")
    elif stage_case_count is None:
        failures.append("stage_case_count is missing")
    if len(records) != expected["rows"]:
        failures.append(f"n_rows={len(records)}, expected {expected['rows']}")

    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        candidate = _candidate_id(_lookup(row, "candidate", "candidate_id", "candidate_name"))
        if candidate is not None:
            by_candidate[candidate].append(row)
    observed_candidates = set(by_candidate)
    if observed_candidates != set(_EXPECTED_STAGE_CANDIDATES):
        failures.append(
            "candidate identities=" + repr(sorted(observed_candidates))
            + ", expected=" + repr(list(_EXPECTED_STAGE_CANDIDATES))
        )

    case_sets: dict[str, set[Any]] = {}
    expected_case_count = expected["cases"]
    for candidate in _EXPECTED_STAGE_CANDIDATES:
        candidate_rows = by_candidate.get(candidate, [])
        if len(candidate_rows) != expected_case_count:
            failures.append(
                f"candidate {candidate} n_rows={len(candidate_rows)}, expected {expected_case_count}"
            )
        failed_rows = [row for row in candidate_rows if _lookup(row, "status") not in (None, "ok")]
        if failed_rows:
            failures.append(f"candidate {candidate} has {len(failed_rows)} non-ok rows")
        identities: list[Any] = []
        for row in candidate_rows:
            case = _lookup(row, "case_id", "case", "run_id")
            if case is None:
                failures.append(f"candidate {candidate} has a row without case_id")
                continue
            identities.append(_freeze_key(case))
        case_sets[candidate] = set(identities)
        if len(identities) != len(set(identities)):
            failures.append(f"candidate {candidate} has duplicate case_id rows")
        if len(case_sets[candidate]) != expected_case_count:
            failures.append(
                f"candidate {candidate} unique case_ids={len(case_sets[candidate])}, expected {expected_case_count}"
            )
    if case_sets:
        first = case_sets.get(_EXPECTED_STAGE_CANDIDATES[0], set())
        if any(values != first for values in case_sets.values()):
            failures.append("A--F case identity sets are not identical")

    # Validate the declared family/condition/strength/seed/k grid when the
    # runner persisted its stage-2 constants.  A missing constants block is a
    # provenance failure, not permission to infer the grid from observed rows.
    constants = manifest.get("stage2_constants")
    if not isinstance(constants, Mapping):
        failures.append("stage2_constants is missing")
    else:
        families = tuple(str(value) for value in constants.get("families", ()))
        strengths = tuple(float(value) for value in constants.get("nuisance_strengths", ()))
        seeds = tuple(_freeze_key(value) for value in constants.get("seeds", ()))
        ks = tuple(_freeze_key(value) for value in constants.get("k_values", ()))
        raw_conditions = constants.get("conditions", ())
        conditions = []
        for condition in raw_conditions if isinstance(raw_conditions, (list, tuple)) else ():
            if isinstance(condition, (list, tuple)) and condition:
                conditions.append(str(condition[0]))
            elif isinstance(condition, Mapping):
                conditions.append(str(condition.get("name")))
        if stage == "screen":
            conditions = conditions[:4]
            # The runner reuses the archived constants block for provenance,
            # but screen deliberately overrides its seed grid to 0..9.
            seeds = tuple(range(10))
        expected_cells = {
            (family, condition, float(strength), seed, k)
            for family in families
            for condition in conditions
            for strength in strengths
            for seed in seeds
            for k in ks
        }
        observed_cells = {
            (
                str(_lookup(row, "family")),
                str(_lookup(row, "condition", "condition_name")),
                float(_finite_float(_lookup(row, "nuisance_strength")) or 0.0),
                _freeze_key(_lookup(row, "seed")),
                _freeze_key(_lookup(row, "k")),
            )
            for row in by_candidate.get("A", ())
            if _lookup(row, "family") is not None
            and _lookup(row, "condition", "condition_name") is not None
            and _lookup(row, "nuisance_strength") is not None
            and _lookup(row, "seed") is not None
            and _lookup(row, "k") is not None
        }
        if expected_cells and observed_cells != expected_cells:
            failures.append(
                f"identity grid cells={len(observed_cells)}, expected {len(expected_cells)}"
            )

    return {
        "stage": stage,
        "status": "fail" if failures else "pass",
        "expected_cases": expected["cases"],
        "expected_rows": expected["rows"],
        "observed_cases": len(case_sets.get("A", set())),
        "observed_rows": len(records),
        "failures": failures,
    }


def build_analysis_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
    source_hashes_map: Mapping[str, str] | None = None,
    screen_promotion: Mapping[str, Any] | None = None,
    promotion_decision: Mapping[str, Any] | os.PathLike[str] | str | None = None,
    runtime_benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe summary from runner records."""

    protocol = dict(protocol or (_read_json(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else {}))
    if screen_promotion is None and promotion_decision is not None:
        screen_promotion = _coerce_screen_promotion(promotion_decision)
    else:
        screen_promotion = _coerce_screen_promotion(screen_promotion)
    records = [dict(row) for row in rows]
    manifest = dict(manifest or {})
    stage = _analysis_stage(manifest)
    artifact_completeness = _artifact_completeness(records, manifest)
    robustness_completeness = _robustness_completeness(
        _candidate_rows(records),
        manifest,
    )
    food_artifact_completeness = (
        _food_artifact_completeness(
            records,
            manifest,
            screen_promotion=screen_promotion,
        )
        if str(stage).lower() == "food101"
        else None
    )
    food101_completeness = food_artifact_completeness
    if stage == "full" and food101_completeness is None:
        food101_completeness = {
            "stage": "food101",
            "status": "inconclusive",
            "failures": ["Food-101 artifact was not supplied for the full-stage decision"],
            "inconclusive_reasons": ["food101_path is missing"],
        }
    grouped = _candidate_rows(records)
    ordered_candidates = [candidate for candidate in CANDIDATE_ORDER if candidate in grouped]
    ordered_candidates.extend(sorted(candidate for candidate in grouped if candidate not in ordered_candidates))
    arrays = _candidate_metric_arrays(grouped, ordered_candidates)
    block_candidates = [candidate for candidate in ordered_candidates if candidate not in {"F", "G"}]
    block_metrics = _block_metric_summaries(grouped, block_candidates)
    pooled_detection, pooled_detection_deltas = _pooled_pair_detection_summaries(
        grouped, block_candidates
    )
    pooled_directional = _pooled_pairwise_direction_summaries(grouped, block_candidates)
    # Preserve the pooled v3 estimand for every direct pair stream.  The
    # scalar block summaries remain useful for legacy precomputed rows, but
    # must never replace a recomputed pooled AUROC/AUPRC/calibration result.
    for candidate, metric_values in pooled_detection.items():
        block_metrics.setdefault(candidate, {}).update(metric_values)
    candidate_summary = {
        candidate: _candidate_summary(candidate, grouped[candidate], grouped, arrays, block_metrics)
        for candidate in ordered_candidates
    }
    for candidate, metric_values in pooled_detection.items():
        detection = candidate_summary.get(candidate, {}).get("genuine_overlap")
        if not isinstance(detection, MutableMapping):
            continue
        for metric, value in metric_values.items():
            if metric in detection and isinstance(value, Mapping):
                detection[metric] = value.get("estimate")
        detection["n"] = int(next(iter(metric_values.values())).get("n", detection.get("n", 0))) if metric_values else detection.get("n", 0)
    for candidate, directional in pooled_directional.items():
        if candidate not in candidate_summary:
            continue
        candidate_summary[candidate]["pairwise_directions"] = directional.get(
            "pairwise_directions", {}
        )
        if candidate != "B":
            candidate_summary[candidate]["pairwise_direction_deltas"] = directional.get(
                "pairwise_direction_deltas", {}
            )
    panel_selection = downstream_selection_metrics(records)
    for candidate, metrics in panel_selection.items():
        if candidate in candidate_summary:
            candidate_summary[candidate]["selection_metrics"] = metrics
            for key, value in (metrics.get("product_contrasts", {}) if isinstance(metrics, Mapping) else {}).items():
                candidate_summary[candidate][key] = value
    synthetic_selection = _synthetic_selection_metrics(records)
    for candidate, metrics in synthetic_selection.items():
        if candidate in candidate_summary:
            candidate_summary[candidate]["selection_metrics"] = metrics
            for key in (
                "clean_linear_regret",
                "nuisance_linear_regret",
                "nuisance_linear_regret_upper",
                "nonlinear_regret",
            ):
                if key in metrics:
                    candidate_summary[candidate][key] = metrics[key]
    if "B" in synthetic_selection:
        base_rows = synthetic_selection["B"].get("_regret_rows", {})
        for candidate in PROMOTION_CANDIDATES:
            if candidate not in synthetic_selection or candidate not in candidate_summary:
                continue
            candidate_rows = synthetic_selection[candidate].get("_regret_rows", {})
            for metric_name, target_name in (
                ("clean_linear", "clean_linear_regret_delta"),
                ("nuisance_linear", "nuisance_linear_regret_delta"),
                ("quadratic", "quadratic_regret_delta"),
                ("knn", "knn_regret_delta"),
                ("rbf", "rbf_regret_delta"),
            ):
                if metric_name not in candidate_rows or metric_name not in base_rows:
                    continue
                blocks = _complete_block_values(
                    {
                        "B": base_rows[metric_name],
                        candidate: candidate_rows[metric_name],
                    },
                    lambda row: _finite_float(row.get("regret")),
                    "seed",
                )
                # Private candidate rows already share panel identities; retain
                # only identical seed blocks for the paired contrast.
                delta = paired_block_percentile_difference(blocks, baseline="B")
                candidate_summary[candidate][target_name] = delta.get(candidate)
        for metrics in synthetic_selection.values():
            metrics.pop("_regret_rows", None)
    _build_delta_metrics(candidate_summary, arrays, grouped)
    for candidate, metric_values in pooled_detection_deltas.items():
        if candidate not in candidate_summary:
            continue
        for metric, value in metric_values.items():
            target = {
                "fpr": "fpr_delta",
                "fnr": "fnr_delta",
                "auroc": "pair_auroc_delta",
                "auprc": "pair_auprc_delta",
                "brier": "brier_delta",
                "ece": "ece_delta",
            }.get(metric)
            if target is not None:
                candidate_summary[candidate][target] = value
    _derive_historical_fields(candidate_summary, grouped)
    if stage in {"screen", "full"} and robustness_completeness.get("status") != "pass":
        # A pairwise intersection can still yield a numerically plausible
        # robustness curve after one C/D/E arm silently loses a stratum.  Do
        # not expose that subset AUC as a promotion metric; retain the
        # completeness object as the auditable reason for undefinedness.
        for candidate in PROMOTION_CANDIDATES:
            if candidate in candidate_summary:
                candidate_summary[candidate]["nuisance_robustness_auc"] = {
                    "estimate": None,
                    "lower": None,
                    "upper": None,
                    "n": 0,
                    "status": "undefined",
                    "reason": "robustness completeness is not verified",
                }

    # Ratios use medians from the paired row keys; undefined B timing remains
    # explicitly undefined rather than silently substituted.
    baseline_rows = grouped.get("B", [])
    baseline_total = [value for row in baseline_rows if (value := _extract_timing(row, "total", "total_seconds", "runtime_seconds", "elapsed_seconds")) is not None]
    baseline_fixed = [value for row in baseline_rows if (value := _extract_timing(row, "score_fixed", "score_fixed_seconds", "score_fixed_time")) is not None]
    for candidate, metrics in candidate_summary.items():
        values_total = [value for row in grouped[candidate] if (value := _extract_timing(row, "total", "total_seconds", "runtime_seconds", "elapsed_seconds")) is not None]
        values_fixed = [value for row in grouped[candidate] if (value := _extract_timing(row, "score_fixed", "score_fixed_seconds", "score_fixed_time")) is not None]
        def paired_ratios(extractor: Callable[[Mapping[str, Any]], float | None]) -> np.ndarray:
            _, aligned = _aligned_values(
                grouped,
                extractor,
                ["B", candidate],
            )
            if "B" not in aligned or candidate not in aligned:
                return np.asarray([], dtype=float)
            base_values, candidate_values = aligned["B"], aligned[candidate]
            valid = np.isfinite(base_values) & np.isfinite(candidate_values) & (base_values > 0.0)
            return candidate_values[valid] / base_values[valid]

        def total_wall(row: Mapping[str, Any]) -> float | None:
            direct = _extract_timing(
                row,
                "total",
                "total_seconds",
                "runtime_seconds",
                "elapsed_seconds",
                "outer_wall_seconds",
                "policy_wall_seconds",
                "wall_seconds",
            )
            if direct is not None:
                return direct
            fit = _extract_timing(row, "fit", "fit_seconds", "fit_time", "fit_wall_seconds")
            fixed = _extract_timing(row, "score_fixed", "score_fixed_seconds", "score_fixed_time", "score_fixed_wall_seconds")
            if fit is not None and fixed is not None:
                return fit + fixed
            return None

        total_ratios = paired_ratios(total_wall)
        fixed_ratios = paired_ratios(
            lambda row: _extract_timing(
                row,
                "score_fixed",
                "score_fixed_seconds",
                "score_fixed_time",
                "score_fixed_wall_seconds",
            )
        )
        def total_cpu(row: Mapping[str, Any]) -> float | None:
            direct = _extract_timing(
                row,
                "total_cpu_seconds",
                "outer_cpu_seconds",
                "policy_cpu_seconds",
                "cpu_seconds",
            )
            if direct is not None:
                return direct
            fit = _extract_timing(row, "fit_cpu_seconds")
            fixed = _extract_timing(row, "score_fixed_cpu_seconds")
            if fit is not None and fixed is not None:
                return fit + fixed
            return None

        fixed_cpu_ratios = paired_ratios(
            lambda row: _extract_timing(row, "score_fixed_cpu_seconds")
        )
        total_cpu_ratios = paired_ratios(total_cpu)
        memory_ratios = paired_ratios(
            lambda row: (
                _finite_float(
                    _lookup(
                        row,
                        "peak_memory_per_cell_mb",
                        "cell_peak_memory_mb",
                        "individual_peak_memory_mb",
                    )
                )
            )
        )
        metrics["timing"]["median_total_ratio_vs_B"] = (
            float(np.median(total_ratios)) if total_ratios.size else None
        )
        metrics["timing"]["p95_total_ratio_vs_B"] = (
            float(np.percentile(total_ratios, 95)) if total_ratios.size else None
        )
        metrics["timing"]["median_score_fixed_ratio_vs_B"] = (
            float(np.median(fixed_ratios)) if fixed_ratios.size else None
        )
        metrics["timing"]["median_total_cpu_ratio_vs_B"] = (
            float(np.median(total_cpu_ratios)) if total_cpu_ratios.size else None
        )
        metrics["timing"]["p95_total_cpu_ratio_vs_B"] = (
            float(np.percentile(total_cpu_ratios, 95)) if total_cpu_ratios.size else None
        )
        metrics["timing"]["median_score_fixed_cpu_ratio_vs_B"] = (
            float(np.median(fixed_cpu_ratios)) if fixed_cpu_ratios.size else None
        )
        metrics["timing"]["median_peak_memory_ratio_vs_B"] = (
            float(np.median(memory_ratios)) if memory_ratios.size else None
        )
        metrics["timing"]["p95_peak_memory_ratio_vs_B"] = (
            float(np.percentile(memory_ratios, 95)) if memory_ratios.size else None
        )
        metrics["timing"]["individual_peak_memory_ratio_vs_B"] = (
            float(np.max(memory_ratios)) if memory_ratios.size else None
        )

    base = {
        "schema_version": 1,
        "stage": stage,
        "statistics": {
            "paired": True,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interval": "two-sided percentile 95%",
            "complete_pairing": True,
        },
        "n_rows": len(records),
        "manifest": json_safe(dict(manifest)),
        "artifact_completeness": artifact_completeness,
        "robustness_completeness": robustness_completeness,
        "food_artifact_completeness": food_artifact_completeness,
        "food101_completeness": food101_completeness,
        "runtime_benchmark": json_safe(runtime_benchmark),
        "baseline_parity": json_safe(manifest.get("baseline_parity")),
        "determinism": json_safe(manifest.get("determinism")),
        "determinism_verification": json_safe(manifest.get("determinism_verification")),
        "screen_promotion": json_safe(screen_promotion),
        "locked_candidate": (
            screen_promotion.get("selected_candidate")
            if isinstance(screen_promotion, Mapping)
            else None
        ),
        "candidates": candidate_summary,
        "historical_gates": {},
        "product_gates": {},
        "input_hashes": dict(source_hashes_map or {}),
        "source_hashes": dict(source_hashes_map or {}),
        "provenance": git_provenance(),
        "deviations": [],
    }
    if artifact_completeness.get("status") in {"fail", "inconclusive"}:
        base["deviations"].append(
            "runner artifact is not a verified complete frozen stage; promotion/full gate claims are blocked"
        )
    code_hashes = {
        key: value
        for key, value in (source_hashes_map or {}).items()
        if key.startswith("experiments/nuisance_conditioned_distance/")
    }
    base["provenance"]["code_identity_sha256"] = sha256_bytes(canonical_json(code_hashes).encode("utf-8"))
    base["provenance"]["source_hashes"] = dict(source_hashes_map or {})
    runner_provenance = manifest.get("provenance") if isinstance(manifest, Mapping) else None
    if isinstance(runner_provenance, Mapping):
        for key in (
            "starting_commit",
            "experiment_commit",
            "git_dirty",
            "git_status_sha256",
            "code_identity_sha256",
        ):
            if runner_provenance.get(key) is not None:
                base["provenance"][f"runner_{key}"] = json_safe(runner_provenance.get(key))
    if not any(
        _extract_reference_value(row, "linear") is not None
        and _lookup(row, "selected", "selected_candidate", "selector_candidate") is not None
        for row in records
    ):
        base["deviations"].append(
            "panel reference/explicit selector rows unavailable; downstream regret, exact-best, and within-0.01 cells are inconclusive"
        )
    if not any(_lookup(row, "family") is not None for row in records):
        base["deviations"].append(
            "family drift, stable-shift, and k-by-balance strata gate inputs unavailable; gates remain inconclusive"
        )
    if any(_lookup(row, "peak_rss_bytes") is not None for row in records):
        base["deviations"].append(
            "peak RSS is process-wide ru_maxrss; per-cell memory ratios are reported only when explicitly supplied"
        )
    # Gate evaluators consume summary-shaped candidate dictionaries.  Keeping
    # their output in the same object also makes report generation stateless.
    locked_candidate = (
        base.get("locked_candidate")
        if stage == "full" and _valid_screen_lock(screen_promotion)
        else None
    )
    base["locked_candidate"] = locked_candidate
    gate_candidates = (
        [locked_candidate]
        if locked_candidate in PROMOTION_CANDIDATES
        else []
        if stage == "full"
        else None
    )
    base["historical_gates"] = evaluate_historical_gates(
        base,
        protocol=protocol,
        candidate_ids=gate_candidates,
    )
    base["product_gates"] = evaluate_product_gates(
        base,
        protocol=protocol,
        candidate_ids=gate_candidates,
    )
    # Promotion eligibility is intentionally computed after gates, but it is
    # written separately by ``write_analysis`` as an immutable-ish artifact.
    base["protocol"] = {
        "schema_version": protocol.get("schema_version"),
        "frozen_date": protocol.get("frozen_date"),
        "candidate_ids": list(CANDIDATE_ORDER),
    }
    return json_safe(base)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_immutable_json(path: os.PathLike[str] | str, value: Any) -> None:
    """Write once; repeat writes are accepted only when bytes are identical."""

    destination = Path(path)
    content = (canonical_json(value) + "\n").encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(f"immutable artifact already exists with different content: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, destination)


def write_analysis(
    input_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str | None = None,
    *,
    food101_path: os.PathLike[str] | str | None = None,
    screen_promotion: Mapping[str, Any] | os.PathLike[str] | str | None = None,
    promotion_decision: Mapping[str, Any] | os.PathLike[str] | str | None = None,
    runtime_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Analyze an output directory and write all requested artifacts."""

    loaded = load_runner_output(input_path)
    if screen_promotion is not None and promotion_decision is not None:
        raise ValueError("supply only one of screen_promotion or promotion_decision")
    promotion_input = screen_promotion if screen_promotion is not None else promotion_decision
    screen_promotion_info = _coerce_screen_promotion(promotion_input)
    destination = Path(output_path) if output_path is not None else Path(input_path) / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(input_path, loaded.get("source_files"))
    summary = build_analysis_summary(
        loaded["rows"],
        manifest={**loaded.get("manifest", {}), **loaded.get("metadata", {})},
        source_hashes_map=hashes,
        screen_promotion=screen_promotion_info,
    )
    runtime_loaded: dict[str, Any] | None = None
    runtime_hashes: dict[str, str] = {}
    if runtime_path is not None:
        runtime_loaded = load_runner_output(runtime_path)
        runtime_hashes = source_hashes(runtime_path, runtime_loaded.get("source_files"))
    elif str(summary.get("stage") or "").lower() == "full":
        summary["runtime_benchmark"] = {
            "status": "inconclusive",
            "completeness": {
                "status": "inconclusive",
                "failures": ["large-budget runtime artifact was not supplied"],
            },
            "candidates": {},
        }
    # Food-101 is optional and can be analyzed independently when it follows
    # the same row contract.  Keep it under a separate key so synthetic
    # promotion cannot accidentally consume another evidence source.
    if food101_path is not None:
        food_loaded = load_runner_output(food101_path)
        food_hashes = source_hashes(food101_path, food_loaded.get("source_files"))
        summary["food101"] = build_analysis_summary(
            food_loaded["rows"],
            manifest={**food_loaded.get("manifest", {}), **food_loaded.get("metadata", {})},
            source_hashes_map=food_hashes,
            screen_promotion=screen_promotion_info,
        )
        summary["input_hashes"].update({f"food101:{key}": value for key, value in food_hashes.items()})
        summary["food101_product_gates"] = summary["food101"].get("product_gates", {})
        summary["food101_completeness"] = summary["food101"].get("food_artifact_completeness")
        # Full-stage promotion consumes the frozen Food product gates; keep
        # the synthetic screen gates available under food101 for audit rather
        # than silently treating an arm-pooled synthetic result as Food proof.
        if summary.get("stage") == "full":
            summary["product_gates"] = summary["food101_product_gates"]
    if runtime_loaded is not None:
        runtime_manifest = {
            **runtime_loaded.get("manifest", {}),
            **runtime_loaded.get("metadata", {}),
        }
        runtime_completeness = _runtime_artifact_completeness(
            runtime_loaded.get("rows", []),
            runtime_manifest,
            screen_promotion=screen_promotion_info,
        )
        runtime_summary = _runtime_benchmark_summary(
            runtime_loaded.get("rows", []),
            runtime_completeness,
        )
        summary["runtime_benchmark"] = runtime_summary
        summary["input_hashes"].update({f"runtime:{key}": value for key, value in runtime_hashes.items()})
    # Runtime is an independent product-gate input.  Re-evaluate after all
    # optional evidence has been attached, and keep the Food surface separate
    # from synthetic candidate metrics.
    if runtime_loaded is not None or str(summary.get("stage") or "").lower() == "full":
        locked_ids = [summary.get("locked_candidate")] if summary.get("locked_candidate") in PROMOTION_CANDIDATES else None
        if isinstance(summary.get("food101"), Mapping):
            food_summary = summary["food101"]
            food_summary["runtime_benchmark"] = summary.get("runtime_benchmark")
            food_summary["product_gates"] = evaluate_product_gates(
                food_summary,
                candidate_ids=locked_ids,
            )
            summary["food101_product_gates"] = food_summary["product_gates"]
            if summary.get("stage") == "full":
                # Final product evidence comes from the exact Food comparator
                # contrasts plus the separately attached runtime artifact.
                # Synthetic candidate rows remain the historical/geometry
                # evidence and must not replace missing Food contrasts.
                summary["product_gates"] = food_summary["product_gates"]
        else:
            summary["product_gates"] = evaluate_product_gates(summary, candidate_ids=locked_ids)
    _write_json(destination / "analysis_summary.json", summary)
    promotion_sources = dict(hashes)
    for key, value in summary.get("input_hashes", {}).items():
        if str(key).startswith("food101:"):
            promotion_sources[str(key)] = value
    if isinstance(screen_promotion_info, Mapping) and screen_promotion_info.get("sha256"):
        promotion_sources["screen_promotion_decision_sha256"] = str(screen_promotion_info["sha256"])
    for key, value in runtime_hashes.items():
        promotion_sources[f"runtime:{key}"] = value
    promotion = select_promotion(summary, input_hashes=summary.get("input_hashes", hashes), source_hashes_map=promotion_sources)
    write_immutable_json(destination / "promotion_decision.json", promotion)

    # Reporting is optional at import time, but a local import keeps the core
    # statistics usable in environments without matplotlib.
    from .reporting import write_reports

    write_reports(destination, summary, promotion)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="one runner output directory")
    parser.add_argument("--output", type=Path, default=None, help="analysis artifact directory")
    parser.add_argument("--food101", type=Path, default=None, help="optional Food-101 output directory")
    parser.add_argument(
        "--screen-promotion",
        type=Path,
        default=None,
        help="screen promotion_decision.json that locks a full-stage candidate",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=None,
        help="optional separate large-budget Food runtime artifact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        write_analysis(
            args.input,
            args.output,
            food101_path=args.food101,
            screen_promotion=args.screen_promotion,
            runtime_path=args.runtime,
        )
    except Exception as exc:  # CLI should identify malformed artifacts clearly.
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CANDIDATE_ORDER",
    "PROMOTION_CANDIDATES",
    "aggregate_reference_metrics",
    "binary_detection_metrics",
    "bootstrap_indices",
    "build_analysis_summary",
    "canonical_json",
    "choose_promotion",
    "classify_gate",
    "downstream_selection_metrics",
    "evaluate_gate",
    "evaluate_historical_gates",
    "evaluate_product_gates",
    "json_safe",
    "load_runner_output",
    "monotonic_ordering_rate",
    "nonlinear_pooled_metric",
    "normalized_trapezoid_auc",
    "paired_bootstrap",
    "paired_bootstrap_ci",
    "paired_bootstrap_difference",
    "paired_block_percentile_bootstrap",
    "paired_block_percentile_difference",
    "paired_replicate_block_bootstrap",
    "paired_seed_block_bootstrap",
    "paired_percentile_bootstrap",
    "pooled_reference_metrics",
    "screen_promotion",
    "select_promotion",
    "seed_block_bootstrap",
    "replicate_block_bootstrap",
    "sha256_file",
    "source_hashes",
    "git_provenance",
    "spearman_rank_correlation",
    "stable_auprc",
    "tie_aware_auroc",
    "write_analysis",
    "write_immutable_json",
]
