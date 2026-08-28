"""Hash-locked Food-101 bridge replay for the M50 ranking experiment.

The bridge and embedding caches are retrospective inputs.  This runner keeps
the six OI methods and the full linear probe on exactly the same rows, folds,
and seeds, while preserving original scalar labels for OI/conditioning.  The
integer label encoding in this file is used only as a scikit-learn split/probe
implementation detail.

The full grid is intentionally expensive and is never part of the default
unit suite.  ``--smoke`` executes only a declared structural subset; its
decision surface omits numerical outcomes.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

# The Food runner is intentionally serial.  The CLI/launcher must provide
# these values before importing NumPy/scikit-learn; run_food101 verifies them
# again so an in-process caller cannot silently report an uncontrolled run.
_THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from . import manifest
from . import candidates


ARCHIVED_ROOT = Path("/Users/niklasmelton/code/vertabrae")
DEFAULT_STEM = "food101_nonlinear_backbone_bridge_food101_k10_89095e3bc0db"
DEFAULT_DRIVER = ARCHIVED_ROOT / "examples/food101_nonlinear_backbone_bridge.py"
DEFAULT_RESULT = ARCHIVED_ROOT / "examples/output" / f"{DEFAULT_STEM}.json"
DEFAULT_COHORT = ARCHIVED_ROOT / "examples/output" / f"{DEFAULT_STEM}.cohorts.json"
DEFAULT_CACHE = ARCHIVED_ROOT / "examples/output/cache"
DEFAULT_PRIOR = Path(
    "/Users/niklasmelton/.codex/worktrees/a9b5/vertabrae/examples/research/"
    "selector_replay/artifacts/full/raw_results.json"
)
DEFAULT_RUNTIME_SOURCE = ARCHIVED_ROOT / "examples/food101_selector_runtime_scaling.py"

SOURCE_CONFIGURATION_HASH = "89095e3bc0db391235a71489fdb86013a8503a113673ce7a757989a27cd05529"
DRIVER_SHA256 = "63f906f12070aaf697d88c193dd11dc2f430bc9fa6b0389756d556636ad85eae"
RESULT_SHA256 = "0826871de7a37d72ad6618e48a39b4895c74bb6ab30d9eb7521a2b767e483b55"
COHORT_SHA256 = "038c5ecff1fa42b4a71bb235a35fb562d2a9542ce52caa336566be56d0f9d88d"
PRIOR_SHA256 = "56d394e8c7f07a83e37cc14e0c33fd6188c2020c4011f3b5d635a2d13934586d"
RUNTIME_SOURCE_SHA256 = "72a10f57681c27229572f885805b06c73ea20b7b0240e5477980dbef6103e7b9"

MODELS = manifest.MODELS
ARMS = (
    ("baseline", 0.0, 0.0),
    ("nonlinearity_full", 1.0, 0.0),
    ("nuisance_full", 0.0, 1.5),
)
BUDGETS = manifest.BUDGETS
REPLICATES = manifest.REPLICATES
FOLDS = manifest.FOLDS
K = manifest.K
SEED = 42
CAP_ROWS = 2048
METHODS = manifest.SELECTOR_METHODS
OI_METHODS = manifest.OI_METHODS
LP_METHOD = "LP-FULL"
SMOKE_MODELS = (MODELS[0],)
SMOKE_REPLICATES = (0,)
SMOKE_ARMS = ("baseline",)
SMOKE_BUDGETS = (64,)

# Terminal tables are published only after their complete staged bundle has
# been assembled.  If a process is interrupted during that publication, these
# exact files are safe to replace on a same-identity resume while the running
# manifest remains the publication marker.  Checkpoints are deliberately not
# included here: they have their own identity-locked validation.
_TERMINAL_OUTPUT_FILES = frozenset(
    {
        "selector_rows.jsonl",
        "reference_rows.jsonl",
        "probe_rows.jsonl",
        "capped_probe_rows.jsonl",
        "baseline_parity_rows.jsonl",
        "selector_rows.csv",
        "reference_rows.csv",
        "probe_rows.csv",
        "capped_probe_rows.csv",
        "baseline_parity_rows.csv",
        "raw_results.json",
        "manifest.json",
    }
)
_FINALIZATION_DIRECTORY = ".finalizing"



def _require_thread_environment() -> dict[str, str | None]:
    observed = {
        name: os.environ.get(name)
        for name in _THREAD_ENVIRONMENT
    }
    mismatches = {
        name: value
        for name, value in observed.items()
        if value != _THREAD_ENVIRONMENT[name]
    }
    if mismatches:
        raise RuntimeError(
            "Food runner requires one-thread environment values before outcomes; "
            f"mismatches={mismatches!r}"
        )
    return observed


def _sha256(path: os.PathLike[str] | str) -> str:
    return manifest.sha256_path(path)


def _json_safe(value: Any) -> Any:
    return manifest._json_safe(value)


def canonical_json(value: Any) -> str:
    return manifest.canonical_json(value)


def _read_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, (canonical_json(payload) + "\n").encode("utf-8"))


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: canonical_json(row.get(key))
                if isinstance(row.get(key), (Mapping, list, tuple))
                else _json_safe(row.get(key))
                for key in fields
            }
        )
    _atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def _bundle_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the exact run configuration used by running/final manifests."""

    return {
        "models": list(
            args.models
            if isinstance(args.models, (list, tuple))
            else _parse_subset(args.models, MODELS)
        ),
        "replicates": list(
            args.replicates
            if isinstance(args.replicates, (list, tuple))
            else _parse_subset(args.replicates, REPLICATES, int)
        ),
        "budgets": list(
            args.budgets
            if isinstance(args.budgets, (list, tuple))
            else _parse_subset(args.budgets, BUDGETS, int)
        ),
        "arms": list(
            args.arms
            if isinstance(args.arms, (list, tuple))
            else _parse_subset(args.arms, tuple(name for name, _, _ in ARMS))
        ),
        "methods": list(
            args.methods
            if isinstance(args.methods, (list, tuple))
            else _parse_subset(args.methods, METHODS)
        ),
        "folds": FOLDS,
        "k": K,
        "seed": SEED,
        "counterbalanced": True,
        "warmup_excluded": True,
        "smoke": bool(args.smoke),
    }


def _source_manifest(
    source_hashes: Mapping[str, tuple[Path, str]],
) -> dict[str, dict[str, str]]:
    return {
        str(name): {"path": str(path), "sha256": str(digest)}
        for name, (path, digest) in source_hashes.items()
    }


def _running_manifest(
    *,
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
    protocol_hash: str | None,
    source_hashes: Mapping[str, tuple[Path, str]],
) -> dict[str, Any]:
    """Build the immutable identity surface written before expensive work."""

    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "artifact_status": "running",
        "study": "food101_m50_backbone_ranking",
        "retrospective": True,
        "configuration": _bundle_configuration(args),
        "environment": _environment(provenance),
        "repository_provenance": dict(provenance),
        "protocol": {"path": str(manifest.PROTOCOL_PATH), "sha256": protocol_hash},
        "sources": _source_manifest(source_hashes),
        # Cache identities are discovered one model at a time.  Checkpoint
        # identities validate the exact matrix hash on resume; the up-front
        # manifest intentionally carries an empty, non-authorizing surface.
        "cache_identity": {},
        "tables": {},
        "warmup_excluded": True,
        "deviations": [
            "Food-101 is retrospective development evidence, not untouched confirmation.",
            "A running manifest is an identity marker only; terminal tables are published later.",
        ],
    }


def _validate_running_manifest(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Reject resume unless the immutable run identity is unchanged."""

    for field in ("schema_version", "artifact_status", "study", "configuration", "protocol", "sources"):
        if existing.get(field) != expected.get(field):
            raise RuntimeError(f"Food running manifest {field!r} does not match current identity")
    if existing.get("environment") != expected.get("environment"):
        raise RuntimeError("Food running manifest environment does not match current identity")
    existing_provenance = existing.get("repository_provenance")
    expected_provenance = expected.get("repository_provenance")
    if not isinstance(existing_provenance, Mapping) or not isinstance(expected_provenance, Mapping):
        raise RuntimeError("Food running manifest provenance is malformed")
    # Git status lines/hash can change as checkpoints and terminal surfaces
    # are created; every execution-relevant provenance field is immutable.
    for field in (
        "starting_commit",
        "original_develop_base",
        "develop_commit_at_execution",
        "merge_base_with_develop",
        "merge_base_with_starting_commit",
        "experiment_commit",
        "source_hashes",
        "code_identity_sha256",
        "output_roots_excluded",
    ):
        if existing_provenance.get(field) != expected_provenance.get(field):
            raise RuntimeError(
                f"Food running manifest provenance {field!r} does not match current identity"
            )


def _clear_transient_bundle_surfaces(output: Path) -> None:
    """Remove only terminal files left by an interrupted finalization."""

    for name in sorted(_TERMINAL_OUTPUT_FILES - {"manifest.json"}):
        path = output / name
        if path.exists():
            if not path.is_file():
                raise RuntimeError(f"Food transient surface is not a file: {path}")
            path.unlink()
    staging = output / _FINALIZATION_DIRECTORY
    if staging.exists():
        if not staging.is_dir():
            raise RuntimeError(f"Food finalization surface is not a directory: {staging}")
        for child in staging.iterdir():
            # _atomic_write_bytes can leave a same-directory temporary file
            # when interrupted.  Accept only names generated for this fixed
            # terminal surface set; unknown files are never recursively removed.
            if child.is_file() and (
                child.name in _TERMINAL_OUTPUT_FILES
                or any(child.name.startswith(f".{name}.") for name in _TERMINAL_OUTPUT_FILES)
            ):
                child.unlink()
            else:
                raise RuntimeError(f"Food finalization surface is unexpected: {child.name!r}")
        staging.rmdir()


def _prepare_finalization_directory(output: Path) -> Path:
    """Create the fixed, narrowly scoped staging directory for a bundle."""

    staging = output / _FINALIZATION_DIRECTORY
    if staging.exists():
        if not staging.is_dir():
            raise RuntimeError(f"Food finalization surface is not a directory: {staging}")
        for child in staging.iterdir():
            if child.is_file() and (
                child.name in _TERMINAL_OUTPUT_FILES
                or any(child.name.startswith(f".{name}.") for name in _TERMINAL_OUTPUT_FILES)
            ):
                child.unlink()
            else:
                raise RuntimeError(f"Food finalization surface is unexpected: {child.name!r}")
        staging.rmdir()
    staging.mkdir(parents=False, exist_ok=False)
    return staging


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _table_manifest(rows: Sequence[Mapping[str, Any]], payload: bytes) -> dict[str, Any]:
    return {
        "row_count": int(len(rows)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "encoding": "canonical_jsonl_utf8",
    }


def _parse_subset(raw: str, allowed: Sequence[Any], cast: Callable[[str], Any] = str) -> tuple[Any, ...]:
    values = tuple(cast(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values or len(set(values)) != len(values) or any(value not in allowed for value in values):
        raise ValueError(f"invalid subset {values!r}; expected values from {tuple(allowed)!r}")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--source-cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--prior-replay", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--runtime-source", type=Path, default=DEFAULT_RUNTIME_SOURCE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    # Keep selector arguments unset until ``_run`` knows whether this is a
    # full run or the documented structural smoke.  A parser default of the
    # full grid would make ``--smoke`` fail its exact-subset validation before
    # the user had an opportunity to specify any selectors.
    parser.add_argument("--models", default=None)
    parser.add_argument("--replicates", default=None)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--arms", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def _load_driver(path: Path, expected_hash: str = DRIVER_SHA256) -> Any:
    observed = _sha256(path)
    if observed != expected_hash:
        raise ValueError(f"Food bridge driver hash mismatch: expected {expected_hash}, got {observed}")
    specification = importlib.util.spec_from_file_location("_m50_frozen_food101_bridge", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load bridge driver at {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _validate_sources(result: Mapping[str, Any], cohort: Mapping[str, Any]) -> None:
    if result.get("artifact_status") != "completed":
        raise ValueError("archived Food source result is not completed")
    if result.get("configuration_hash") != SOURCE_CONFIGURATION_HASH:
        raise ValueError("archived Food source result configuration hash mismatch")
    if cohort.get("configuration_hash") != SOURCE_CONFIGURATION_HASH:
        raise ValueError("archived Food source cohort configuration hash mismatch")
    sample_ids = cohort.get("extracted_sample_ids")
    if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)) or len(sample_ids) != 28_480:
        raise ValueError("archived Food cohort must contain exactly 28,480 sample ids")
    roles = cohort.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("archived Food cohort has no role map")
    if not all(str(rep) in roles for rep in REPLICATES):
        raise ValueError("archived Food cohort is missing one or more replicate roles")
    normalized_ids = [str(value) for value in sample_ids]
    observed_sample_hash = _sample_ids_hash(normalized_ids)
    if observed_sample_hash != SAMPLE_IDS_SHA256:
        raise ValueError(
            "archived Food cohort sample identity hash mismatch: "
            f"expected {SAMPLE_IDS_SHA256}, got {observed_sample_hash}"
        )
    labels = [value.split("/")[2] for value in normalized_ids]
    observed_label_hash = hashlib.sha256(
        json.dumps(labels, default=str).encode("utf-8")
    ).hexdigest()
    if observed_label_hash != LABELS_SHA256:
        raise ValueError(
            "archived Food cohort label identity hash mismatch: "
            f"expected {LABELS_SHA256}, got {observed_label_hash}"
        )
    frozen_inputs = result.get("frozen_inputs", {})
    cache_identity = frozen_inputs.get("cache_identity") if isinstance(frozen_inputs, Mapping) else None
    if not isinstance(cache_identity, Mapping) or not cache_identity:
        raise ValueError("archived replay has no cache identity surface")
    identity_hashes = {
        str(value.get("identity_hash"))
        for value in cache_identity.values()
        if isinstance(value, Mapping)
    }
    if identity_hashes != {COMMON_CACHE_IDENTITY_HASH}:
        raise ValueError(
            "archived replay cache identity mismatch: "
            f"expected {COMMON_CACHE_IDENTITY_HASH}, got {sorted(identity_hashes)!r}"
        )


def _load_archived_inputs(
    driver_path: Path,
    result_path: Path,
    cohort_path: Path,
    prior_path: Path,
    runtime_path: Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    expected = {
        "driver": DRIVER_SHA256,
        "source_result": RESULT_SHA256,
        "source_cohort": COHORT_SHA256,
        "prior_replay": PRIOR_SHA256,
    }
    paths = {
        "driver": driver_path,
        "source_result": result_path,
        "source_cohort": cohort_path,
        "prior_replay": prior_path,
    }
    if runtime_path is not None:
        paths["runtime_source"] = runtime_path
        expected["runtime_source"] = RUNTIME_SOURCE_SHA256
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing archived {name}: {path}")
        hashes[name] = _sha256(path)
        if hashes[name] != expected[name]:
            raise ValueError(f"{name} hash mismatch: expected {expected[name]}, got {hashes[name]}")
    driver = _load_driver(driver_path)
    result = _read_json(result_path)
    cohort = _read_json(cohort_path)
    prior = _read_json(prior_path)
    _validate_sources(result, cohort)
    return driver, result, cohort, prior, hashes


def _sample_ids_hash(sample_ids: Sequence[str]) -> str:
    # Match the archived bridge's identity convention exactly (default
    # json separators, not this package's compact canonical recipe JSON).
    return hashlib.sha256(json.dumps(list(sample_ids)).encode("utf-8")).hexdigest()


SAMPLE_IDS_SHA256 = "d17b3e8cf42d8af02a9761569bfdddb775153484c9b93c1f0eaa8611d7118f37"
LABELS_SHA256 = "abdd53b66b50622113023a9f9b7810faeee667f80b781ec47cca521fc4fe7645"
COMMON_CACHE_IDENTITY_HASH = "9d648b596083e39558f8a2fab0a5fc3a451954dbc81c47a09f65bce45a41c853"


def validate_cache_manifest(
    manifest_path: os.PathLike[str] | str,
    *,
    model: str,
    expected_sample_ids_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one cache only after validating manifest and matrix bytes."""

    path = Path(manifest_path)
    payload = _read_json(path)
    if payload.get("model") != model:
        raise ValueError(f"cache manifest model mismatch for {model}")
    if payload.get("sample_ids_sha256") != expected_sample_ids_sha256:
        raise ValueError(f"cache sample identity mismatch for {model}")
    matrix_name = payload.get("path")
    if not isinstance(matrix_name, str) or not matrix_name:
        raise ValueError(f"cache manifest has no matrix path for {model}")
    matrix_path = Path(matrix_name)
    if not matrix_path.is_absolute():
        local = path.parent / matrix_path
        matrix_path = local if local.exists() else (ARCHIVED_ROOT / matrix_path)
    expected_matrix_hash = payload.get("sha256", payload.get("matrix_sha256"))
    if not isinstance(expected_matrix_hash, str) or len(expected_matrix_hash) != 64:
        raise ValueError(f"cache manifest has no valid matrix hash for {model}")
    if not matrix_path.is_file():
        raise ValueError(f"cache matrix is missing for {model}: {matrix_path}")
    observed_matrix_hash = _sha256(matrix_path)
    if observed_matrix_hash != expected_matrix_hash:
        raise ValueError(f"cache matrix SHA-256 mismatch for {model}")
    identity_hash = payload.get("identity_hash")
    if identity_hash is not None and str(identity_hash) != COMMON_CACHE_IDENTITY_HASH:
        raise ValueError(f"cache identity hash mismatch for {model}")
    labels_hash = payload.get("labels_sha256")
    if labels_hash is not None and str(labels_hash) != LABELS_SHA256:
        raise ValueError(f"cache label identity hash mismatch for {model}")
    sample_hash = payload.get("sample_ids_sha256")
    if sample_hash != SAMPLE_IDS_SHA256:
        raise ValueError(f"cache sample identity hash mismatch for {model}")
    try:
        matrix = np.load(matrix_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load cache matrix for {model}") from exc
    shape = payload.get("shape")
    if not isinstance(shape, Sequence) or list(matrix.shape) != [int(value) for value in shape]:
        raise ValueError(f"cache shape mismatch for {model}")
    if matrix.ndim != 2 or matrix.shape[0] != 28_480:
        raise ValueError(f"cache matrix row count mismatch for {model}")
    if not np.issubdtype(matrix.dtype, np.number) or not np.all(np.isfinite(np.asarray(matrix[: min(2, matrix.shape[0])]))):
        raise ValueError(f"cache matrix is not finite numeric data for {model}")
    return matrix, payload


def _load_cache(cache_dir: Path, model: str, expected_sample_ids_sha256: str) -> tuple[np.ndarray, dict[str, Any]]:
    return validate_cache_manifest(
        cache_dir / f"food101_{model}_final.json",
        model=model,
        expected_sample_ids_sha256=expected_sample_ids_sha256,
    )


def _row_l2(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _stratification_encoding(labels: np.ndarray) -> tuple[np.ndarray, tuple[Any, ...]]:
    """First-observed integer encoding used only by sklearn internals."""

    target = np.asarray(labels)
    if target.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    encoded = np.empty(target.size, dtype=np.int64)
    classes: list[Any] = []
    positions: list[Any] = []
    for index, raw in enumerate(target.tolist()):
        label = raw.item() if isinstance(raw, np.generic) else raw
        if isinstance(label, (list, tuple, set, frozenset, dict, np.ndarray)):
            raise ValueError("labels must contain hashable scalar values")
        try:
            position = positions.index(label)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TypeError):
                raise ValueError("labels must contain hashable scalar values") from exc
            position = len(classes)
            try:
                hash(label)
            except TypeError as hash_exc:
                raise ValueError("labels must contain hashable scalar values") from hash_exc
            positions.append(label)
            classes.append(label)
        encoded[index] = position
    if len(classes) < 2:
        raise ValueError("at least two classes are required")
    return encoded, tuple(classes)


def _stratified_folds(labels: np.ndarray, *, n_splits: int = FOLDS, seed: int = SEED) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    encoded, _ = _stratification_encoding(np.asarray(labels))
    counts = np.bincount(encoded)
    if counts.size == 0 or int(np.min(counts)) < int(n_splits):
        raise ValueError(f"every class must have at least {n_splits} rows")
    placeholder = np.zeros((encoded.size, 1), dtype=np.uint8)
    splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    return tuple(
        (np.asarray(train, dtype=np.int64), np.asarray(holdout, dtype=np.int64))
        for train, holdout in splitter.split(placeholder, encoded)
    )


def _oi_kwargs(k_per_class: Mapping[Any, int], seed: int) -> dict[str, Any]:
    """Fresh archived upstream kwargs for one fold.

    The archived replay intentionally supplied only the backend selector,
    per-class prototype counts, and fold seed.  Adding seemingly harmless
    MiniBatchKMeans defaults changes the constructor recipe and can change
    random initialization/parity, so the exact archived mapping is preserved.
    """

    return {
        "model_type": "MiniBatchKMeans",
        "kmeans_k": deepcopy(dict(k_per_class)),
        "kmeans_kwargs": {"random_state": int(seed)},
    }


def _refinement_summary(model: Any) -> dict[str, Any]:
    value = getattr(model, "prototype_refinement_", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _conditioning_summary(model: Any) -> dict[str, Any]:
    value = getattr(model, "conditioning_diagnostics_", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _conditioning_runtime_summary(model: Any) -> dict[str, Any]:
    value = getattr(model, "runtime_diagnostics_", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _candidate_spec(candidate_id: str) -> tuple[str, str, str, bool, bool]:
    value = str(candidate_id)
    if value == LP_METHOD:
        return (LP_METHOD, "linear_probe_full", "probe", False, False)
    try:
        spec = candidates.CANDIDATE_BY_ID[value]
    except KeyError as exc:
        raise ValueError(f"unknown Food method {candidate_id!r}; expected {METHODS!r}")
    return (
        spec.candidate_id,
        spec.name,
        spec.estimator,
        spec.prototype_refinement,
        spec.direct_control,
    )


def _probe_config_identity(
    candidate_id: str,
    *,
    fold: int,
    seed: int,
    cap_rows: int | None,
) -> dict[str, Any]:
    """Return the closed, resolved configuration for one probe fold."""

    candidate = str(candidate_id)
    if candidate not in {LP_METHOD, "LP-CAPPED-2048"}:
        raise ValueError(f"unknown probe recipe {candidate!r}")
    if candidate == LP_METHOD and cap_rows is not None:
        raise ValueError("LP-FULL cannot carry a cap")
    if candidate == "LP-CAPPED-2048" and cap_rows != CAP_ROWS:
        raise ValueError("LP-CAPPED-2048 requires the frozen 2048-row cap")
    return {
        "candidate_id": candidate,
        "fold": int(fold),
        "split_seed": int(seed),
        "model_random_state": int(seed),
        "selection": {
            "cap_rows": int(cap_rows) if cap_rows is not None else None,
            "strategy": "stratified_first_observed" if cap_rows is not None else "all_rows",
        },
        "normalizer": {
            "class": "sklearn.preprocessing.Normalizer",
            "norm": "l2",
        },
        "logistic_regression": {
            "class": "sklearn.linear_model.LogisticRegression",
            "C": 1.0,
            "max_iter": 2_000,
            "random_state": int(seed),
            "n_jobs": 1,
        },
        "label_encoding": "first_observed_for_sklearn_only",
    }


def _config_sha256(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def _recipe_bindings(
    rows: Sequence[Mapping[str, Any]],
    capped_rows: Sequence[Mapping[str, Any]] = (),
    *,
    seed_rule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind resolved fold recipes and the deterministic seed convention."""

    hashes: dict[str, set[str]] = {}
    for row in tuple(rows) + tuple(capped_rows):
        candidate = str(row.get("candidate_id"))
        folds = row.get("folds", ())
        if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
            continue
        for fold in folds:
            if not isinstance(fold, Mapping):
                continue
            digest = fold.get("candidate_config_sha256")
            if isinstance(digest, str) and len(digest) == 64:
                hashes.setdefault(candidate, set()).add(digest)
    ordered_hashes = {
        candidate: sorted(values)
        for candidate, values in sorted(hashes.items())
    }
    family_hashes = {
        candidate: _config_sha256(
            {"candidate_id": candidate, "candidate_config_sha256": values}
        )
        for candidate, values in ordered_hashes.items()
    }
    return {
        "candidate_config_sha256": ordered_hashes,
        "recipe_family_sha256": family_hashes,
        "deterministic_seed_rule": dict(seed_rule) if seed_rule is not None else {
            "selector_panel_seed": "SEED + replicate",
            "oi_fold_seed": "selector_panel_seed + fold",
            "probe_split_seed": "selector_panel_seed",
            "probe_model_random_state": "selector_panel_seed",
            "capped_subset_seed": "selector_panel_seed",
        },
    }


def _validate_fold_recipe(
    candidate_id: str,
    fold: Mapping[str, Any],
    *,
    error_prefix: str,
) -> None:
    """Recompute and validate one resolved fold recipe digest."""

    identity = fold.get("candidate_config_identity")
    digest = fold.get("candidate_config_sha256")
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"{error_prefix} is missing candidate_config_identity")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise RuntimeError(f"{error_prefix} has malformed candidate_config_sha256")
    candidate = str(candidate_id)
    if candidate in OI_METHODS:
        oi_kwargs = identity.get("overlap_index_kwargs")
        if not isinstance(oi_kwargs, Mapping):
            raise RuntimeError(f"{error_prefix} lacks overlap_index_kwargs")
        expected_identity = candidates.candidate_config_identity(candidate, oi_kwargs)
        expected_digest = candidates.candidate_config_sha256(candidate, oi_kwargs)
    elif candidate in {LP_METHOD, "LP-CAPPED-2048"}:
        try:
            fold_index = int(fold["fold"])
            split_seed = int(fold["split_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{error_prefix} lacks fold/seed identity") from exc
        expected_identity = _probe_config_identity(
            candidate,
            fold=fold_index,
            seed=split_seed,
            cap_rows=CAP_ROWS if candidate == "LP-CAPPED-2048" else None,
        )
        expected_digest = _config_sha256(expected_identity)
    else:
        raise RuntimeError(f"{error_prefix} has unknown candidate {candidate!r}")
    if dict(identity) != expected_identity or digest != expected_digest:
        raise RuntimeError(f"{error_prefix} candidate configuration hash mismatch")


def _validate_row_recipes(
    row: Mapping[str, Any],
    *,
    error_prefix: str,
) -> None:
    """Require all five fold recipes and validate each resolved identity."""

    candidate = str(row.get("candidate_id"))
    folds = row.get("folds")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
        raise RuntimeError(f"{error_prefix} is missing fold recipe rows")
    if len(folds) != FOLDS:
        raise RuntimeError(f"{error_prefix} has an incomplete fold recipe")
    observed_folds: list[int] = []
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise RuntimeError(f"{error_prefix} has a malformed fold recipe")
        try:
            observed_folds.append(int(fold["fold"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{error_prefix} has a malformed fold index") from exc
        _validate_fold_recipe(candidate, fold, error_prefix=error_prefix)
    if sorted(observed_folds) != list(range(FOLDS)):
        raise RuntimeError(f"{error_prefix} has duplicate or missing fold recipes")


def _validate_recipe_bindings(
    bindings: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    capped_rows: Sequence[Mapping[str, Any]],
    *,
    expected_ids: Sequence[str],
    seed_rule: Mapping[str, Any] | None = None,
    error_prefix: str = "recipe bindings",
) -> None:
    """Require top-level recipe hashes to match every executed fold row."""

    if set(bindings) != {
        "candidate_config_sha256",
        "recipe_family_sha256",
        "deterministic_seed_rule",
    }:
        raise RuntimeError(f"{error_prefix} has a noncanonical schema")
    expected = _recipe_bindings(rows, capped_rows, seed_rule=seed_rule)
    if bindings != expected:
        raise RuntimeError(f"{error_prefix} do not match fold recipes")
    candidate_hashes = bindings.get("candidate_config_sha256")
    family_hashes = bindings.get("recipe_family_sha256")
    expected_keys = {str(value) for value in expected_ids}
    if (
        not isinstance(candidate_hashes, Mapping)
        or not isinstance(family_hashes, Mapping)
        or set(candidate_hashes) != expected_keys
        or set(family_hashes) != expected_keys
    ):
        raise RuntimeError(f"{error_prefix} do not cover exact executed IDs")


def _build_selector(candidate_id: str, kwargs: Mapping[str, Any]) -> Any:
    spec = _candidate_spec(candidate_id)
    if spec[0] == LP_METHOD:
        return None
    # The canonical candidate table owns both direct A/B controls and the
    # conditioned experiment-local adapters.  Its factory deep-copies the
    # nested OI mapping for each fold.
    return candidates.build_candidate(spec[0], deepcopy(dict(kwargs)))


def _lp_fold_score(
    values: np.ndarray,
    labels: np.ndarray,
    seed: int,
    *,
    candidate_id: str = LP_METHOD,
    cap_rows: int | None = None,
) -> dict[str, Any]:
    if str(candidate_id) not in {LP_METHOD, "LP-CAPPED-2048"}:
        raise ValueError(f"unknown probe recipe {candidate_id!r}")
    encoded, _ = _stratification_encoding(labels)
    predictions = np.empty(encoded.size, dtype=np.int64)
    folds = _stratified_folds(labels, n_splits=FOLDS, seed=seed)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        model = make_pipeline(
            Normalizer(norm="l2"),
            # The archived bridge used the panel seed for every fold.  Fold
            # indices remain part of the schedule, but must not be folded
            # into the LogisticRegression random_state when exact score
            # parity is required.
            LogisticRegression(C=1.0, max_iter=2_000, random_state=int(seed), n_jobs=1),
        )
        fit_wall, fit_cpu = time.perf_counter(), time.process_time()
        model.fit(values[train], encoded[train])
        fit_wall_elapsed, fit_cpu_elapsed = time.perf_counter() - fit_wall, time.process_time() - fit_cpu
        predict_wall, predict_cpu = time.perf_counter(), time.process_time()
        predictions[holdout] = model.predict(values[holdout])
        pred_wall, pred_cpu = time.perf_counter() - predict_wall, time.process_time() - predict_cpu
        config_identity = _probe_config_identity(
            candidate_id,
            fold=fold,
            seed=int(seed),
            cap_rows=cap_rows,
        )
        fold_rows.append(
            {
                "fold": fold,
                "seed": int(seed),
                "split_seed": int(seed),
                "model_random_state": int(seed),
                "train_size": int(train.size),
                "holdout_size": int(holdout.size),
                "fit_wall_seconds": float(max(0.0, fit_wall_elapsed)),
                "fit_cpu_seconds": float(max(0.0, fit_cpu_elapsed)),
                "predict_wall_seconds": float(max(0.0, pred_wall)),
                "predict_cpu_seconds": float(max(0.0, pred_cpu)),
                "score": float(accuracy_score(encoded[holdout], predictions[holdout])),
                "candidate_config_identity": config_identity,
                "candidate_config_sha256": _config_sha256(config_identity),
            }
        )
    return {
        "candidate_id": str(candidate_id),
        "candidate_name": "linear_probe_full",
        "mode": "probe",
        "prototype_refinement": False,
        "score": float(accuracy_score(encoded, predictions)),
        "fit_wall_seconds": float(sum(row["fit_wall_seconds"] for row in fold_rows)),
        "fit_cpu_seconds": float(sum(row["fit_cpu_seconds"] for row in fold_rows)),
        "score_fixed_wall_seconds": float(sum(row["predict_wall_seconds"] for row in fold_rows)),
        "score_fixed_cpu_seconds": float(sum(row["predict_cpu_seconds"] for row in fold_rows)),
        "folds": fold_rows,
        "refinement": {},
        "conditioning": {},
        "conditioning_runtime": {},
    }


def _cross_fitted_score(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    candidate: str | tuple[Any, ...],
    seed: int,
) -> dict[str, Any]:
    """Fit/score one method on exact paired folds and return structural rows."""

    candidate_id = str(candidate[0]) if isinstance(candidate, tuple) else str(candidate)
    target = np.asarray(labels)
    raw_values = np.asarray(matrix, dtype=np.float32)
    encoded, classes = _stratification_encoding(target)
    if candidate_id == LP_METHOD:
        # The frozen probe recipe owns its Normalizer; do not apply a second
        # external row-L2 pass before the probe.
        return _lp_fold_score(raw_values, target, int(seed))
    values = _row_l2(raw_values)
    folds = _stratified_folds(target, n_splits=FOLDS, seed=int(seed))
    scores: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    refinement_totals = {
        "prototype_count_before": 0,
        "prototype_count_after": 0,
        "eligible_count": 0,
        "attempted_count": 0,
        "applied_count": 0,
        "skipped_count": 0,
    }
    conditioning_rows: list[dict[str, Any]] = []
    conditioning_runtime_totals = {
        "conditioning_fit_wall_seconds": 0.0,
        "conditioning_fit_cpu_seconds": 0.0,
        "oi_fit_wall_seconds": 0.0,
        "oi_fit_cpu_seconds": 0.0,
    }
    spec = _candidate_spec(candidate_id)
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        train_encoded = encoded[train]
        train_counts = {
            label: int(np.count_nonzero(train_encoded == class_index))
            for class_index, label in enumerate(classes)
        }
        k_per_class = {
            label: min(K, max(1, count // 5), count)
            for label, count in train_counts.items()
        }
        kwargs = _oi_kwargs(k_per_class, fold_seed)
        config_identity = candidates.candidate_config_identity(candidate_id, kwargs)
        config_sha256 = candidates.candidate_config_sha256(candidate_id, kwargs)
        selector = _build_selector(candidate_id, kwargs)
        fit_wall, fit_cpu = time.perf_counter(), time.process_time()
        selector.fit(values[train], target[train])
        fit_wall_elapsed, fit_cpu_elapsed = time.perf_counter() - fit_wall, time.process_time() - fit_cpu
        score_wall, score_cpu = time.perf_counter(), time.process_time()
        score = float(selector.score_fixed(values[holdout], target[holdout]))
        score_wall_elapsed, score_cpu_elapsed = time.perf_counter() - score_wall, time.process_time() - score_cpu
        refinement = _refinement_summary(selector)
        conditioning = _conditioning_summary(selector)
        conditioning_runtime = _conditioning_runtime_summary(selector)
        for field in conditioning_runtime_totals:
            value = conditioning_runtime.get(field, 0.0)
            if value is not None:
                conditioning_runtime_totals[field] += max(0.0, float(value))
        for field in refinement_totals:
            refinement_totals[field] += int(refinement.get(field, 0) or 0)
        conditioning_rows.append(conditioning)
        scores.append(score)
        fold_rows.append(
            {
                "fold": fold,
                "seed": fold_seed,
                "train_size": int(train.size),
                "holdout_size": int(holdout.size),
                "score": score,
                "fit_wall_seconds": float(max(0.0, fit_wall_elapsed)),
                "fit_cpu_seconds": float(max(0.0, fit_cpu_elapsed)),
                "score_fixed_wall_seconds": float(max(0.0, score_wall_elapsed)),
                "score_fixed_cpu_seconds": float(max(0.0, score_cpu_elapsed)),
                "refinement": refinement,
                "conditioning": conditioning,
                "conditioning_runtime": conditioning_runtime,
                "candidate_config_identity": config_identity,
                "candidate_config_sha256": config_sha256,
            }
        )
    before = refinement_totals["prototype_count_before"]
    refinement_totals["applied_rate"] = float(refinement_totals["applied_count"] / before) if before else 0.0
    return {
        "candidate_id": candidate_id,
        "candidate_name": spec[1],
        "mode": spec[2],
        "prototype_refinement": spec[3],
        "score": float(np.mean(scores)),
        "fit_wall_seconds": float(sum(row["fit_wall_seconds"] for row in fold_rows)),
        "fit_cpu_seconds": float(sum(row["fit_cpu_seconds"] for row in fold_rows)),
        "score_fixed_wall_seconds": float(sum(row["score_fixed_wall_seconds"] for row in fold_rows)),
        "score_fixed_cpu_seconds": float(sum(row["score_fixed_cpu_seconds"] for row in fold_rows)),
        "refinement": refinement_totals,
        "conditioning": conditioning_rows,
        "conditioning_runtime": conditioning_runtime_totals,
        "folds": fold_rows,
    }


def _determinism_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    def fold_signature(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: row.get(key)
            for key in (
                "fold",
                "seed",
                "train_size",
                "holdout_size",
                "score",
                "refinement",
                "conditioning",
                "candidate_config_sha256",
            )
        }

    return _json_safe(
        {
            "candidate_id": result.get("candidate_id"),
            "candidate_name": result.get("candidate_name"),
            "mode": result.get("mode"),
            "prototype_refinement": result.get("prototype_refinement"),
            "score": result.get("score"),
            "refinement": result.get("refinement"),
            "conditioning": result.get("conditioning"),
            "folds": [fold_signature(row) for row in result.get("folds", ()) if isinstance(row, Mapping)],
        }
    )


def _determinism_comparison(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    first_bytes = canonical_json(_determinism_signature(first)).encode()
    second_bytes = canonical_json(_determinism_signature(second)).encode()
    return {
        "exact": first_bytes == second_bytes,
        "first_signature_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "second_signature_sha256": hashlib.sha256(second_bytes).hexdigest(),
        "runtime_fields_excluded": True,
    }


def _capped_determinism_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the outcome-bearing but clock-free LP-CAPPED surface."""

    folds = result.get("folds", ())
    fold_scores = [
        {
            "fold": row.get("fold"),
            "score": row.get("score"),
            "candidate_config_sha256": row.get("candidate_config_sha256"),
        }
        for row in folds
        if isinstance(row, Mapping)
    ] if isinstance(folds, Sequence) and not isinstance(folds, (str, bytes)) else []
    return {
        "candidate_id": "LP-CAPPED-2048",
        "score": result.get("score"),
        "n_rows": result.get("n_rows"),
        "fold_scores": fold_scores,
    }


def _capped_determinism_comparison(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    first_bytes = canonical_json(_capped_determinism_signature(first)).encode()
    second_bytes = canonical_json(_capped_determinism_signature(second)).encode()
    return {
        "exact": first_bytes == second_bytes,
        "first_signature_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "second_signature_sha256": hashlib.sha256(second_bytes).hexdigest(),
        "runtime_fields_excluded": True,
    }


def _first_determinism_panel_ids(
    panels: Sequence[manifest.FoodPanel],
    schedule: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> dict[str, str]:
    """Return the frozen first eligible panel for each warm-up identity.

    The execution loop is panel-major and then follows each panel's planned
    cyclic method order.  Deriving this map from that same schedule makes the
    warm-up location explicit and prevents a resumed run from moving a
    determinism comparison to a later panel.  The capped probe is not in the
    seven-method schedule, but is eligible alongside every panel and therefore
    uses the first panel in panel-major order.
    """

    if not panels:
        raise ValueError("at least one Food panel is required for determinism")
    panel_order = {panel.panel_id: position for position, panel in enumerate(panels)}
    method_set = {str(method) for method in methods}
    first: dict[str, str] = {}
    ordered_schedule = sorted(
        (row for row in schedule if isinstance(row, Mapping)),
        key=lambda row: (
            panel_order.get(str(row.get("panel_id")), len(panel_order)),
            int(row.get("execution_position", -1)),
        ),
    )
    for row in ordered_schedule:
        method = str(row.get("method_id"))
        panel_id = str(row.get("panel_id"))
        if method in method_set and method not in first and panel_id in panel_order:
            first[method] = panel_id
    missing = method_set.difference(first)
    if missing:
        raise ValueError(f"schedule has no eligible determinism panel for {sorted(missing)!r}")
    first["LP-CAPPED-2048"] = panels[0].panel_id
    return first


def _determinism_record(
    comparison: Mapping[str, Any],
    *,
    candidate_id: str,
    panel: manifest.FoodPanel,
) -> dict[str, Any]:
    """Attach one canonical panel-identity object to a clock-free comparison."""

    return {
        "panel_identity": {
            "model": panel.model,
            "replicate": int(panel.replicate),
            "arm": panel.arm,
            "budget": int(panel.budget),
            "candidate_id": str(candidate_id),
        },
        **dict(comparison),
    }


def _validate_determinism_records(
    deterministic: Mapping[str, Any],
    *,
    expected_ids: Sequence[str],
    expected_panel_ids: Mapping[str, str] | None = None,
    panels: Sequence[manifest.FoodPanel] | None = None,
    error_prefix: str = "determinism",
) -> None:
    """Validate one closed, clock-free record per candidate identity."""

    expected = {str(value) for value in expected_ids}
    if set(deterministic) != expected:
        raise RuntimeError(f"{error_prefix} verification is incomplete")
    panel_by_id = {panel.panel_id: panel for panel in (panels or ())}
    required = {
        "panel_identity",
        "exact",
        "first_signature_sha256",
        "second_signature_sha256",
        "runtime_fields_excluded",
    }
    for candidate_id, value in deterministic.items():
        if not isinstance(value, Mapping) or set(value) != required:
            raise RuntimeError(f"{error_prefix} record is malformed for {candidate_id!r}")
        panel_identity = value.get("panel_identity")
        if (
            not isinstance(panel_identity, Mapping)
            or set(panel_identity) != {"model", "replicate", "arm", "budget", "candidate_id"}
        ):
            raise RuntimeError(f"{error_prefix} panel identity is malformed for {candidate_id!r}")
        if panel_identity.get("candidate_id") != candidate_id:
            raise RuntimeError(f"{error_prefix} candidate identity mismatch for {candidate_id!r}")
        try:
            panel_id = (
                f"{panel_identity['model']}__replicate-{int(panel_identity['replicate'])}__"
                f"{panel_identity['arm']}__budget-{int(panel_identity['budget'])}"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{error_prefix} panel identity is malformed for {candidate_id!r}") from exc
        if expected_panel_ids is not None and panel_id != expected_panel_ids.get(candidate_id):
            raise RuntimeError(f"{error_prefix} panel identity mismatch for {candidate_id!r}")
        panel = panel_by_id.get(panel_id)
        if panel is None:
            # Checkpoint validation may not have the complete global panel map;
            # the canonical panel-id encoding above still has to be coherent.
            if not panel_id:
                raise RuntimeError(f"{error_prefix} panel identity is malformed for {candidate_id!r}")
        else:
            if (
                panel_identity.get("model") != panel.model
                or int(panel_identity.get("replicate", -1)) != int(panel.replicate)
                or panel_identity.get("arm") != panel.arm
                or int(panel_identity.get("budget", -1)) != int(panel.budget)
            ):
                raise RuntimeError(f"{error_prefix} panel fields mismatch for {candidate_id!r}")
        for field in ("first_signature_sha256", "second_signature_sha256"):
            digest = value.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise RuntimeError(f"{error_prefix} signature hash is malformed for {candidate_id!r}")
        if value.get("exact") is not True or value.get("runtime_fields_excluded") is not True:
            raise RuntimeError(f"{error_prefix} record is not an exact clock-free comparison")
        if value.get("first_signature_sha256") != value.get("second_signature_sha256"):
            raise RuntimeError(f"{error_prefix} exact record has unequal signatures")


def _stratified_cap(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    target = np.asarray(labels)
    if target.size <= int(maximum):
        return np.arange(target.size, dtype=np.int64)
    encoded, classes = _stratification_encoding(target)
    base, extra = divmod(int(maximum), len(classes))
    selected: list[np.ndarray] = []
    for class_index, _label in enumerate(classes):
        available = np.flatnonzero(encoded == class_index)
        generator = np.random.default_rng(np.random.SeedSequence([int(seed), class_index, target.size]))
        selected.append(generator.permutation(available)[: base + int(class_index < extra)])
    return np.sort(np.concatenate(selected).astype(np.int64))


def _capped_probe(matrix: np.ndarray, labels: np.ndarray, seed: int = SEED) -> dict[str, Any]:
    selected = _stratified_cap(labels, CAP_ROWS, seed)
    result = _lp_fold_score(
        np.asarray(matrix)[selected],
        np.asarray(labels)[selected],
        int(seed),
        candidate_id="LP-CAPPED-2048",
        cap_rows=CAP_ROWS,
    )
    return {
        "score": result["score"],
        "n_rows": int(selected.size),
        "wall_seconds": result["fit_wall_seconds"] + result["score_fixed_wall_seconds"],
        "cpu_seconds": result["fit_cpu_seconds"] + result["score_fixed_cpu_seconds"],
        "folds": result["folds"],
    }


def _prior_lookup(prior: Mapping[str, Any]) -> dict[tuple[str, int, str, int, str], float]:
    result: dict[tuple[str, int, str, int, str], float] = {}
    for row in prior.get("selector_rows", ()):
        if not isinstance(row, Mapping) or row.get("score") is None:
            continue
        try:
            key = (str(row["backbone"]), int(row["replicate"]), str(row["arm"]), int(row["budget"]), str(row["method"]))
            result[key] = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _baseline_parity_rows(rows: Sequence[Mapping[str, Any]], prior: Mapping[str, Any]) -> list[dict[str, Any]]:
    lookup = _prior_lookup(prior)
    old_methods = {"A": "overlap_unrefined_cross_fitted", "B": "overlap_refined_cross_fitted"}
    output: list[dict[str, Any]] = []
    for row in rows:
        method = old_methods.get(str(row.get("candidate_id")))
        if method is None or row.get("score") is None:
            continue
        key = (str(row.get("backbone", row.get("model"))), int(row["replicate"]), str(row["arm"]), int(row["budget"]), method)
        if key not in lookup:
            continue
        delta = float(row["score"]) - lookup[key]
        output.append({**{field: row.get(field) for field in ("candidate_id", "model", "replicate", "arm", "budget")}, "new_score": row["score"], "prior_score": lookup[key], "delta": delta, "exact": delta == 0.0})
    return output


def _probe_parity_rows(rows: Sequence[Mapping[str, Any]], prior: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare directly executed LP-FULL scores with archived LP rows.

    Archived timing is intentionally never consulted.  The score is the only
    parity surface, and a missing reference is itself a failed parity check.
    """

    lookup = _prior_lookup(prior)
    output: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("candidate_id")) != LP_METHOD or row.get("score") is None:
            continue
        key = (
            str(row.get("backbone", row.get("model"))),
            int(row["replicate"]),
            str(row["arm"]),
            int(row["budget"]),
            "linear_probe_oof",
        )
        if key not in lookup:
            continue
        delta = float(row["score"]) - lookup[key]
        output.append(
            {
                "candidate_id": LP_METHOD,
                "model": row.get("model", row.get("backbone")),
                "replicate": row.get("replicate"),
                "arm": row.get("arm"),
                "budget": row.get("budget"),
                "new_score": row["score"],
                "prior_score": lookup[key],
                "delta": delta,
                "exact": delta == 0.0,
            }
        )
    return output


def _smoke_redact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Retain panel/type/finite structure while removing all numeric outcomes."""

    def descriptor(value: Any) -> dict[str, Any]:
        if value is None:
            return {"present": False, "type": None, "finite": None}
        if isinstance(value, Mapping):
            return {"present": True, "type": "mapping", "finite": True}
        if isinstance(value, (bool, np.bool_)):
            return {"present": True, "type": "bool", "finite": True}
        if isinstance(value, (int, np.integer)):
            return {"present": True, "type": "int", "finite": True}
        if isinstance(value, (float, np.floating)):
            return {"present": True, "type": "float", "finite": bool(np.isfinite(value))}
        return {"present": True, "type": type(value).__name__, "finite": True}

    structural = {
        key: row.get(key)
        for key in (
            "stage", "model", "backbone", "replicate", "arm", "budget",
            "candidate_id", "candidate_name", "mode", "prototype_refinement_enabled",
            "execution_position", "execution_order", "schedule_seed", "status",
            "warmup_excluded", "error",
        )
        if key in row
    }
    structural.update(
        {
            "score": descriptor(row.get("score")),
            "fit_wall_seconds": descriptor(row.get("fit_wall_seconds")),
            "fit_cpu_seconds": descriptor(row.get("fit_cpu_seconds")),
            "score_fixed_wall_seconds": descriptor(row.get("score_fixed_wall_seconds")),
            "score_fixed_cpu_seconds": descriptor(row.get("score_fixed_cpu_seconds")),
            "total_wall_seconds": descriptor(row.get("total_wall_seconds")),
            "total_cpu_seconds": descriptor(row.get("total_cpu_seconds")),
            "prototype_refinement": descriptor(row.get("prototype_refinement")),
            "conditioning_diagnostics": descriptor(row.get("conditioning_diagnostics")),
            "folds": {"present": bool(row.get("folds")), "type": "sequence", "finite": True},
        }
    )
    return structural


def _smoke_redact_reference(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("test_accuracy")
    return {
        key: row.get(key)
        for key in ("model", "backbone", "replicate", "arm", "head")
        if key in row
    } | {
        "test_accuracy": {
            "present": value is not None,
            "type": "float" if value is not None else None,
            "finite": bool(np.isfinite(value)) if value is not None else None,
        }
    }


def _smoke_redact_parity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in ("candidate_id", "model", "replicate", "arm", "budget")
        if key in row
    } | {
        "exact": bool(row.get("exact")),
        "delta": {"present": row.get("delta") is not None, "type": "float", "finite": bool(np.isfinite(row["delta"])) if row.get("delta") is not None else None},
    }


def _validate_completed_surfaces(
    *,
    rows: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    capped_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    deterministic: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    replicates: Sequence[int],
    budgets: Sequence[int],
    arms: Sequence[str],
    methods: Sequence[str],
    smoke: bool,
) -> None:
    """Fail closed if completion would publish an incomplete/duplicate grid."""

    resolved_grid = (
        tuple(str(value) for value in models),
        tuple(int(value) for value in replicates),
        tuple(int(value) for value in budgets),
        tuple(str(value) for value in arms),
        tuple(str(value) for value in methods),
    )
    if smoke:
        expected_grid = (
            SMOKE_MODELS,
            SMOKE_REPLICATES,
            SMOKE_BUDGETS,
            SMOKE_ARMS,
            METHODS,
        )
    else:
        expected_grid = (
            tuple(MODELS),
            tuple(REPLICATES),
            tuple(BUDGETS),
            tuple(name for name, _, _ in ARMS),
            tuple(METHODS),
        )
    if resolved_grid != expected_grid:
        raise RuntimeError("completion surface does not match the frozen Food grid")
    panel_count = len(models) * len(replicates) * len(budgets) * len(arms)
    expected_selector = panel_count * len(methods)
    expected_reference = len(models) * len(replicates) * len(arms) * 4
    expected_parity = panel_count * 3
    expected_capped = panel_count
    if len(rows) != expected_selector:
        raise RuntimeError(f"selector row count {len(rows)} != {expected_selector}")
    identities = {
        (
            row.get("model"), row.get("replicate"), row.get("arm"),
            row.get("budget"), row.get("candidate_id"),
        )
        for row in rows
    }
    if len(identities) != len(rows):
        raise RuntimeError("selector rows contain duplicate panel/method identities")
    expected_selector_ids = {
        (model, int(replicate), arm, int(budget), method)
        for model in models
        for replicate in replicates
        for arm in arms
        for budget in budgets
        for method in methods
    }
    if identities != expected_selector_ids:
        raise RuntimeError("selector rows do not cover the exact configured panel/method grid")
    expected_reference = expected_reference if not smoke else 4
    if len(references) != expected_reference:
        raise RuntimeError(f"reference row count {len(references)} != {expected_reference}")
    reference_keys = {
        (
            row.get("backbone", row.get("model")), row.get("replicate"),
            row.get("arm"), row.get("head"),
        )
        for row in references
    }
    if len(reference_keys) != len(references):
        raise RuntimeError("reference rows contain duplicate identities")
    expected_reference_keys = {
        (model, int(replicate), arm, head)
        for model in models
        for replicate in replicates
        for arm in arms
        for head in ("linear", "quadratic", "knn", "rbf")
    }
    if reference_keys != expected_reference_keys:
        raise RuntimeError("reference rows do not cover the exact configured panel grid")
    if any(
        row.get("test_accuracy") is None
        or not np.isfinite(float(row.get("test_accuracy")))
        for row in references
        if not smoke
    ):
        raise RuntimeError("reference rows contain non-finite test accuracy")
    if len(probe_rows) != panel_count:
        raise RuntimeError(f"LP-FULL row count {len(probe_rows)} != {panel_count}")
    probe_ids = {
        (
            row.get("model", row.get("backbone")), row.get("replicate"),
            row.get("arm"), row.get("budget"), row.get("candidate_id"),
        )
        for row in probe_rows
    }
    expected_probe_ids = {
        (model, int(replicate), arm, int(budget), LP_METHOD)
        for model in models
        for replicate in replicates
        for arm in arms
        for budget in budgets
    }
    if probe_ids != expected_probe_ids:
        raise RuntimeError("LP-FULL rows do not cover the exact configured panel grid")
    if len(capped_rows) != expected_capped:
        raise RuntimeError(f"LP-CAPPED-2048 row count {len(capped_rows)} != {expected_capped}")
    capped_ids = {
        (
            row.get("model", row.get("backbone")), row.get("replicate"),
            row.get("arm"), row.get("budget"), row.get("candidate_id"),
        )
        for row in capped_rows
    }
    expected_capped_ids = {
        (model, int(replicate), arm, int(budget), "LP-CAPPED-2048")
        for model in models
        for replicate in replicates
        for arm in arms
        for budget in budgets
    }
    if capped_ids != expected_capped_ids:
        raise RuntimeError("LP-CAPPED-2048 rows do not cover the exact configured panel grid")
    if len(parity_rows) != expected_parity:
        raise RuntimeError(f"parity row count {len(parity_rows)} != {expected_parity}")
    parity_keys = {
        (
            row.get("candidate_id"), row.get("model"), row.get("replicate"),
            row.get("arm"), row.get("budget"),
        )
        for row in parity_rows
    }
    if len(parity_keys) != len(parity_rows) or any(
        row.get("exact") is not True for row in parity_rows
    ):
        raise RuntimeError("parity surface contains duplicates or failures")
    expected_parity_methods = {"A", "B", LP_METHOD}
    if {str(row.get("candidate_id")) for row in parity_rows} != expected_parity_methods:
        raise RuntimeError("parity surface does not contain exact A/B/LP-FULL controls")
    configured_panels = tuple(
        manifest.FoodPanel(model, int(replicate), arm, int(budget))
        for model in models
        for replicate in replicates
        for arm in arms
        for budget in budgets
    )
    configured_schedule = manifest.planned_execution_order(configured_panels, methods)
    expected_determinism_panel_ids = _first_determinism_panel_ids(
        configured_panels,
        configured_schedule,
        methods,
    )
    _validate_determinism_records(
        deterministic,
        expected_ids=tuple(methods) + ("LP-CAPPED-2048",),
        expected_panel_ids=expected_determinism_panel_ids,
        panels=configured_panels,
        error_prefix="determinism",
    )
    if smoke:
        def require_descriptor(value: Any, *, field: str, expected_type: str | None = None) -> None:
            if not isinstance(value, Mapping):
                raise RuntimeError(f"smoke descriptor {field!r} is missing")
            if set(value) != {"present", "type", "finite"}:
                raise RuntimeError(f"smoke descriptor {field!r} has a noncanonical schema")
            if value.get("present") is not True or value.get("finite") is not True:
                raise RuntimeError(f"smoke descriptor {field!r} is not present and finite")
            if expected_type is not None and value.get("type") != expected_type:
                raise RuntimeError(f"smoke descriptor {field!r} has the wrong type")

        for row in tuple(rows) + tuple(probe_rows) + tuple(capped_rows):
            for field in (
                "score",
                "fit_wall_seconds",
                "fit_cpu_seconds",
                "score_fixed_wall_seconds",
                "score_fixed_cpu_seconds",
                "total_wall_seconds",
                "total_cpu_seconds",
                "prototype_refinement",
                "conditioning_diagnostics",
            ):
                require_descriptor(row.get(field), field=field)
            require_descriptor(row.get("folds"), field="folds", expected_type="sequence")
        for row in references:
            require_descriptor(row.get("test_accuracy"), field="test_accuracy")
        for row in parity_rows:
            require_descriptor(row.get("delta"), field="delta")
            if row.get("exact") is not True:
                raise RuntimeError("smoke parity descriptor is not exact")
        return
    for row in tuple(rows) + tuple(probe_rows) + tuple(capped_rows):
        if row.get("status") != "ok":
            raise RuntimeError("cannot publish a completed artifact with failed rows")
        if type(row.get("prototype_refinement_enabled")) is not bool:
            raise RuntimeError("prototype_refinement_enabled must be a strict bool")
        for key in (
            "score", "total_wall_seconds", "total_cpu_seconds",
            "fit_wall_seconds", "fit_cpu_seconds",
            "score_fixed_wall_seconds", "score_fixed_cpu_seconds",
        ):
            value = row.get(key)
            if value is None or not np.isfinite(float(value)) or float(value) < 0.0:
                raise RuntimeError(f"non-finite required timing/score field {key!r}")
        _validate_row_recipes(
            row,
            error_prefix=f"recipe for {row.get('candidate_id')!r}",
        )


def _panel_identity(
    *,
    code_identity_sha256: str,
    protocol_sha256: str,
    source_hashes: Mapping[str, str],
    cache_sha256: str,
    panel: manifest.FoodPanel,
    methods: Sequence[str],
    schedule: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "code_identity_sha256": str(code_identity_sha256),
        "protocol_sha256": str(protocol_sha256),
        "source_hashes": dict(sorted(source_hashes.items())),
        "cache_matrix_sha256": str(cache_sha256),
        "panel": panel.identity(),
        "methods": list(methods),
        "schedule": [dict(row) for row in schedule],
        "warmup_excluded": True,
    }


def _validate_checkpoint(
    path: Path,
    identity: Mapping[str, Any],
    methods: Sequence[str],
    *,
    expected_determinism_panel_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    try:
        payload = _read_json(path)
    except Exception as exc:
        raise RuntimeError(f"resume_checkpoint_malformed: {path}") from exc
    if payload.get("identity") != dict(identity):
        raise RuntimeError(f"resume_checkpoint_identity_mismatch: {path}")
    if payload.get("artifact_status") != "completed":
        raise RuntimeError(f"resume_checkpoint_not_completed: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RuntimeError(f"resume_checkpoint_incomplete_rows: {path}")
    if (
        {str(row.get("candidate_id")) for row in rows if isinstance(row, Mapping)}
        != set(methods)
        or len(rows) != len(methods)
    ):
        raise RuntimeError(f"resume_checkpoint_incomplete_rows: {path}")
    if any(not isinstance(row, Mapping) or row.get("status") != "ok" for row in rows):
        raise RuntimeError(f"resume_checkpoint_failed_rows: {path}")
    panel = identity.get("panel")
    if not isinstance(panel, Mapping):
        raise RuntimeError(f"resume_checkpoint_missing_panel_identity: {path}")
    schedule = identity.get("schedule")
    if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)):
        raise RuntimeError(f"resume_checkpoint_missing_schedule: {path}")
    expected_order = tuple(
        str(row.get("method_id"))
        for row in sorted(
            (row for row in schedule if isinstance(row, Mapping)),
            key=lambda row: int(row.get("execution_position", -1)),
        )
    )
    method_values = tuple(str(value) for value in methods)
    if len(expected_order) != len(method_values) or set(expected_order) != set(method_values):
        raise RuntimeError(f"resume_checkpoint_schedule_mismatch: {path}")
    positions: list[int] = []
    for row in rows:
        if isinstance(row.get("score"), Mapping):
            raise RuntimeError(f"resume_checkpoint_structural_smoke_not_resumable: {path}")
        try:
            position = int(row.get("execution_position"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"resume_checkpoint_bad_execution_position: {path}") from exc
        positions.append(position)
        if tuple(str(value) for value in row.get("execution_order", ())) != expected_order:
            raise RuntimeError(f"resume_checkpoint_execution_order_mismatch: {path}")
        if type(row.get("prototype_refinement_enabled")) is not bool:
            raise RuntimeError(f"resume_checkpoint_refinement_flag_malformed: {path}")
        for field, expected in (
            ("model", panel.get("model")),
            ("replicate", int(panel.get("replicate", -1))),
            ("arm", panel.get("arm")),
            ("budget", int(panel.get("budget", -1))),
        ):
            if row.get(field) != expected:
                raise RuntimeError(f"resume_checkpoint_panel_identity_mismatch: {path}")
        for field in (
            "score",
            "fit_wall_seconds",
            "fit_cpu_seconds",
            "score_fixed_wall_seconds",
            "score_fixed_cpu_seconds",
            "total_wall_seconds",
            "total_cpu_seconds",
        ):
            value = row.get(field)
            if value is None or isinstance(value, bool) or not np.isfinite(float(value)) or float(value) < 0.0:
                raise RuntimeError(f"resume_checkpoint_nonfinite_{field}: {path}")
        try:
            _validate_row_recipes(row, error_prefix=f"resume_checkpoint_{row.get('candidate_id')}")
        except RuntimeError as exc:
            raise RuntimeError(f"resume_checkpoint_recipe_invalid: {path}") from exc
    if sorted(positions) != list(range(len(methods))):
        raise RuntimeError(f"resume_checkpoint_execution_positions_incomplete: {path}")
    for field, expected_method in (
        ("probe_rows", LP_METHOD),
        ("capped_probe_rows", "LP-CAPPED-2048"),
    ):
        table = payload.get(field)
        if not isinstance(table, Sequence) or isinstance(table, (str, bytes)) or len(table) != 1:
            raise RuntimeError(f"resume_checkpoint_{field}_incomplete: {path}")
        probe = table[0]
        if not isinstance(probe, Mapping) or probe.get("candidate_id") != expected_method:
            raise RuntimeError(f"resume_checkpoint_{field}_identity_mismatch: {path}")
        if probe.get("status") != "ok" or type(probe.get("prototype_refinement_enabled")) is not bool:
            raise RuntimeError(f"resume_checkpoint_{field}_status_mismatch: {path}")
        for field_name, expected in (
            ("model", panel.get("model")),
            ("replicate", int(panel.get("replicate", -1))),
            ("arm", panel.get("arm")),
            ("budget", int(panel.get("budget", -1))),
        ):
            if probe.get(field_name) != expected:
                raise RuntimeError(f"resume_checkpoint_{field}_panel_mismatch: {path}")
        for field_name in (
            "score",
            "fit_wall_seconds",
            "fit_cpu_seconds",
            "score_fixed_wall_seconds",
            "score_fixed_cpu_seconds",
            "total_wall_seconds",
            "total_cpu_seconds",
        ):
            value = probe.get(field_name)
            if value is None or isinstance(value, bool) or not np.isfinite(float(value)) or float(value) < 0.0:
                raise RuntimeError(f"resume_checkpoint_{field}_{field_name}_invalid: {path}")
        try:
            _validate_row_recipes(
                probe,
                error_prefix=f"resume_checkpoint_{expected_method}",
            )
        except RuntimeError as exc:
            raise RuntimeError(f"resume_checkpoint_{field}_recipe_invalid: {path}") from exc
    determinism = payload.get("determinism_verification")
    try:
        _validate_determinism_records(
            determinism if isinstance(determinism, Mapping) else {},
            expected_ids=tuple(methods) + ("LP-CAPPED-2048",),
            expected_panel_ids=expected_determinism_panel_ids,
            error_prefix=f"resume_checkpoint_determinism: {path}",
        )
    except RuntimeError as exc:
        raise RuntimeError(f"resume_checkpoint_incomplete_determinism: {path}") from exc
    return payload


def _environment(provenance: Mapping[str, Any]) -> dict[str, Any]:
    # Keep Food and runtime artifacts on one canonical environment surface.
    # In particular, the local checkout is not necessarily installed as an
    # ``overlapindex`` distribution, so each dependency is recorded
    # independently and the project version comes from the hashed pyproject.
    observed = manifest.environment(provenance)
    observed["thread_environment"] = {
        name: os.environ.get(name)
        for name in _THREAD_ENVIRONMENT
    }
    return observed


def validate_run_selection(
    models: Sequence[str],
    replicates: Sequence[int],
    budgets: Sequence[int],
    arms: Sequence[str],
    methods: Sequence[str],
    *,
    smoke: bool,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    values = (tuple(models), tuple(int(v) for v in replicates), tuple(int(v) for v in budgets), tuple(arms), tuple(methods))
    if smoke:
        if values != (SMOKE_MODELS, SMOKE_REPLICATES, SMOKE_BUDGETS, SMOKE_ARMS, METHODS):
            raise ValueError("smoke selection must use the declared structural subset")
        return values
    if values != (tuple(MODELS), tuple(REPLICATES), tuple(BUDGETS), tuple(name for name, _, _ in ARMS), tuple(METHODS)):
        raise ValueError("full Food development refuses subsets or limits")
    return values


def _smoke_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return structural smoke evidence without exposing numerical outcomes."""

    return {
        "status": "pass" if rows and all(row.get("status") == "ok" for row in rows) else "fail",
        "structural_only": True,
        "row_count": len(rows),
        "methods_observed": sorted({str(row.get("candidate_id")) for row in rows}),
        "numeric_outcomes_redacted": True,
    }


def _write_bundle(
    output: Path,
    *,
    artifact_status: str,
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    cache_identity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    capped_probe_rows: Sequence[Mapping[str, Any]],
    deterministic: Mapping[str, Mapping[str, Any]],
    stop_reason: str | None = None,
    stop_error: str | None = None,
) -> dict[str, Any]:
    protocol_hash = manifest.protocol_sha256(manifest.PROTOCOL_PATH) if manifest.PROTOCOL_PATH.is_file() else None
    output_rows: Sequence[Mapping[str, Any]] = (
        tuple(_smoke_redact_row(row) for row in rows) if args.smoke else rows
    )
    output_references: Sequence[Mapping[str, Any]] = (
        tuple(_smoke_redact_reference(row) for row in reference_rows)
        if args.smoke else reference_rows
    )
    output_probe_rows: Sequence[Mapping[str, Any]] = (
        tuple(_smoke_redact_row(row) for row in probe_rows) if args.smoke else probe_rows
    )
    output_capped_rows: Sequence[Mapping[str, Any]] = (
        tuple(_smoke_redact_row(row) for row in capped_probe_rows)
        if args.smoke else capped_probe_rows
    )
    output_parity_rows: Sequence[Mapping[str, Any]] = (
        tuple(_smoke_redact_parity(row) for row in parity_rows) if args.smoke else parity_rows
    )
    recipe_bindings = _recipe_bindings(rows, capped_probe_rows)
    if artifact_status == "completed" and not args.smoke:
        configured_methods = (
            tuple(str(value) for value in args.methods)
            if isinstance(args.methods, (list, tuple))
            else _parse_subset(args.methods, METHODS)
        )
        _validate_recipe_bindings(
            recipe_bindings,
            rows,
            capped_probe_rows,
            expected_ids=tuple(configured_methods) + ("LP-CAPPED-2048",),
        )
    table_payloads = {
        "selector_rows": _canonical_jsonl(output_rows),
        "reference_rows": _canonical_jsonl(output_references),
        "probe_rows": _canonical_jsonl(output_probe_rows),
        "capped_probe_rows": _canonical_jsonl(output_capped_rows),
        "baseline_parity_rows": _canonical_jsonl(output_parity_rows),
    }
    table_identity = {
        name: _table_manifest(rows_value, table_payloads[name])
        for name, rows_value in (
            ("selector_rows", output_rows),
            ("reference_rows", output_references),
            ("probe_rows", output_probe_rows),
            ("capped_probe_rows", output_capped_rows),
            ("baseline_parity_rows", output_parity_rows),
        )
    }
    raw: dict[str, Any] = {
        "schema_version": manifest.SCHEMA_VERSION,
        "artifact_status": artifact_status,
        "study": "food101_m50_backbone_ranking",
        "retrospective": True,
        "configuration": _bundle_configuration(args),
        "environment": _environment(provenance),
        "repository_provenance": dict(provenance),
        "protocol": {"path": str(manifest.PROTOCOL_PATH), "sha256": protocol_hash},
        "sources": {name: {"path": str(path), "sha256": digest} for name, (path, digest) in source_hashes.items()},
        "cache_identity": dict(cache_identity),
        "tables": table_identity,
        "selector_rows": [dict(row) for row in output_rows],
        "reference_rows": [dict(row) for row in output_references],
        "probe_rows": [dict(row) for row in output_probe_rows],
        "capped_probe_rows": [dict(row) for row in output_capped_rows],
        "baseline_parity_rows": [dict(row) for row in output_parity_rows],
        "recipe_bindings": recipe_bindings,
        "baseline_parity": {
            "n": len(parity_rows),
            "exact": bool(parity_rows) and all(row.get("exact") is True for row in parity_rows),
        },
        "determinism_verification": {str(key): dict(value) for key, value in deterministic.items()},
        "deviations": [
            "Food-101 is retrospective development evidence, not untouched confirmation.",
            "All LP-FULL results and timings are executed in this runner; archived LP rows are parity references only.",
        ],
    }
    if args.smoke:
        raw["smoke_decision"] = _smoke_decision(rows)
    if stop_reason is not None:
        raw["stop_reason"] = str(stop_reason)
        raw["stop_error"] = str(stop_error) if stop_error is not None else None
    output.mkdir(parents=True, exist_ok=True)
    staging = _prepare_finalization_directory(output)
    _atomic_write_bytes(staging / "selector_rows.jsonl", table_payloads["selector_rows"])
    _atomic_write_bytes(staging / "reference_rows.jsonl", table_payloads["reference_rows"])
    _atomic_write_bytes(staging / "probe_rows.jsonl", table_payloads["probe_rows"])
    _atomic_write_bytes(staging / "capped_probe_rows.jsonl", table_payloads["capped_probe_rows"])
    _atomic_write_bytes(staging / "baseline_parity_rows.jsonl", table_payloads["baseline_parity_rows"])
    _atomic_write_json(staging / "raw_results.json", raw)
    raw_identity = {
        "sha256": _sha256(staging / "raw_results.json"),
        "size_bytes": int((staging / "raw_results.json").stat().st_size),
    }
    manifest_payload = {key: raw[key] for key in ("schema_version", "artifact_status", "study", "configuration", "environment", "repository_provenance", "protocol", "sources", "cache_identity", "tables", "recipe_bindings", "baseline_parity", "determinism_verification", "deviations")}
    manifest_payload["raw_results"] = raw_identity
    if stop_reason is not None:
        manifest_payload.update({"stop_reason": raw["stop_reason"], "stop_error": raw["stop_error"]})
    _atomic_write_csv(staging / "selector_rows.csv", output_rows)
    _atomic_write_csv(staging / "reference_rows.csv", output_references)
    _atomic_write_csv(staging / "probe_rows.csv", output_probe_rows)
    _atomic_write_csv(staging / "capped_probe_rows.csv", output_capped_rows)
    _atomic_write_csv(staging / "baseline_parity_rows.csv", output_parity_rows)
    _atomic_write_json(staging / "manifest.json", manifest_payload)
    for name in sorted(_TERMINAL_OUTPUT_FILES - {"manifest.json"}):
        (staging / name).replace(output / name)
    # The manifest is the terminal publication marker.  A process interrupted
    # before this replacement leaves the prior running manifest in place, so
    # resume can safely discard the staged/partial terminal surfaces.
    (staging / "manifest.json").replace(output / "manifest.json")
    staging.rmdir()
    return raw


def run_food101(
    *,
    output: Path,
    driver_path: Path = DEFAULT_DRIVER,
    source_result_path: Path = DEFAULT_RESULT,
    source_cohort_path: Path = DEFAULT_COHORT,
    prior_replay_path: Path = DEFAULT_PRIOR,
    runtime_source_path: Path | None = DEFAULT_RUNTIME_SOURCE,
    cache_dir: Path = DEFAULT_CACHE,
    models: Sequence[str] = MODELS,
    replicates: Sequence[int] = REPLICATES,
    budgets: Sequence[int] = BUDGETS,
    arms: Sequence[str] = tuple(name for name, _, _ in ARMS),
    methods: Sequence[str] = METHODS,
    smoke: bool = False,
    resume: bool = False,
    driver: Any | None = None,
    source_result: Mapping[str, Any] | None = None,
    source_cohort: Mapping[str, Any] | None = None,
    prior_replay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute full or smoke Food replay, writing only atomic stage artifacts."""

    selected = validate_run_selection(models, replicates, budgets, arms, methods, smoke=smoke)
    models_t, replicates_t, budgets_t, arms_t, methods_t = selected
    if smoke and resume:
        raise ValueError(
            "smoke resume is disabled because structural checkpoints cannot reconstruct outcomes"
        )
    output_path = Path(output).resolve()
    existing_manifest = output_path / "manifest.json"
    existing_manifest_payload: dict[str, Any] | None = None
    if existing_manifest.exists():
        try:
            existing_manifest_payload = _read_json(existing_manifest)
            existing_status = existing_manifest_payload.get("artifact_status")
        except Exception as exc:
            raise RuntimeError("terminal artifact manifest is malformed; refuse overwrite/resume") from exc
        if existing_status in {"completed", "stopped"}:
            raise RuntimeError(
                f"terminal Food artifact already exists at {output_path}; refuse overwrite/resume"
            )
        if existing_status != "running" and existing_status is not None:
            raise RuntimeError(
                f"Food artifact has an unknown status {existing_status!r}; refuse overwrite/resume"
            )
        if existing_status == "running" and not resume:
            raise RuntimeError(
                "Food artifact is running; pass resume=True to continue its identity-locked stage"
            )
        if existing_status == "running" and resume:
            unexpected = [
                child.name
                for child in output_path.iterdir()
                if child.name
                not in ({"checkpoints", _FINALIZATION_DIRECTORY} | set(_TERMINAL_OUTPUT_FILES))
            ]
            if unexpected:
                raise RuntimeError(
                    "Food resume contains an unexpected running-stage entry: "
                    f"{sorted(unexpected)!r}"
                )
    elif output_path.exists() and any(output_path.iterdir()):
        if not resume:
            raise RuntimeError(
                "Food output directory is non-empty without a manifest; refuse overwrite/resume"
            )
        unexpected = [
            child.name
            for child in output_path.iterdir()
            if child.name != "checkpoints"
        ]
        if unexpected:
            raise RuntimeError(
                "Food resume requires a checkpoint-only directory when no manifest exists; "
                f"unexpected entries={sorted(unexpected)!r}"
            )
    _require_thread_environment()
    provenance = manifest.repository_provenance(require_sources=True)
    protocol_hash = manifest.verify_protocol_hash() if manifest.PROTOCOL_PATH.is_file() else None
    protocol_evidence = (
        manifest.verify_protocol_source_evidence()
        if manifest.PROTOCOL_PATH.is_file()
        else {}
    )
    source_paths: dict[str, tuple[Path, str]] = {
        "driver": (Path(driver_path).resolve(), DRIVER_SHA256),
        "source_result": (Path(source_result_path).resolve(), RESULT_SHA256),
        "source_cohort": (Path(source_cohort_path).resolve(), COHORT_SHA256),
        "prior_replay": (Path(prior_replay_path).resolve(), PRIOR_SHA256),
    }
    if runtime_source_path is not None:
        source_paths["runtime_source"] = (Path(runtime_source_path).resolve(), RUNTIME_SOURCE_SHA256)
    archived_hashes: dict[str, str] = {
        str(name): str(value["sha256"])
        for name, value in protocol_evidence.items()
        if name in {"food_bridge_driver", "food_source_result", "food_source_cohort", "archived_selector_replay", "food_runtime_driver", "archived_runtime_replay"}
    }
    for name, (path, expected_hash) in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing archived {name}: {path}")
        observed = _sha256(path)
        if observed != expected_hash:
            raise ValueError(f"{name} hash mismatch: expected {expected_hash}, got {observed}")
        archived_hashes[name] = observed
    if driver is None:
        driver = _load_driver(source_paths["driver"][0])
    if source_result is None:
        source_result = _read_json(source_paths["source_result"][0])
    if source_cohort is None:
        source_cohort = _read_json(source_paths["source_cohort"][0])
    if prior_replay is None:
        prior_replay = _read_json(source_paths["prior_replay"][0])
    _validate_sources(source_result, source_cohort)
    run_args = argparse.Namespace(
        models=models_t,
        replicates=replicates_t,
        budgets=budgets_t,
        arms=arms_t,
        methods=methods_t,
        smoke=smoke,
    )
    expected_running_manifest = _running_manifest(
        args=run_args,
        provenance=provenance,
        protocol_hash=protocol_hash,
        source_hashes=source_paths,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    if existing_manifest_payload is not None:
        _validate_running_manifest(existing_manifest_payload, expected_running_manifest)
        _clear_transient_bundle_surfaces(output_path)
    else:
        _atomic_write_json(output_path / "manifest.json", expected_running_manifest)
    sample_ids = [str(value) for value in source_cohort["extracted_sample_ids"]]
    sample_hash = _sample_ids_hash(sample_ids)
    labels = np.asarray([value.split("/")[2] for value in sample_ids], dtype=object)
    roles = {int(key): value for key, value in source_cohort["roles"].items()}
    checkpoints = output_path / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    capped_probe_rows: list[dict[str, Any]] = []
    capped_warmed = False
    reference_rows: list[dict[str, Any]] = [
        dict(row)
        for row in source_result.get("reference_rows", ())
        if isinstance(row, Mapping)
        and str(row.get("backbone")) in models_t
        and int(row.get("replicate", -1)) in replicates_t
        and str(row.get("arm")) in arms_t
    ]
    cache_identity: dict[str, Any] = {}
    deterministic: dict[str, dict[str, Any]] = {}
    warmed: set[str] = set()
    arm_lookup = {name: (lam, nu) for name, lam, nu in ARMS}
    panel_list = tuple(manifest.FoodPanel(model, replicate, arm, budget) for model in models_t for replicate in replicates_t for arm in arms_t for budget in budgets_t)
    if resume and checkpoints.exists():
        expected_checkpoint_names = {
            f"{panel.panel_id}.json" for panel in panel_list
        }
        unexpected_checkpoints = sorted(
            child.name
            for child in checkpoints.iterdir()
            if child.name not in expected_checkpoint_names
        )
        if unexpected_checkpoints:
            raise RuntimeError(
                "Food resume contains unknown checkpoint files: "
                f"{unexpected_checkpoints!r}"
            )
    full_schedule = manifest.planned_execution_order(panel_list, methods_t)
    manifest.validate_execution_order(full_schedule, panel_list, methods_t)
    first_determinism_panel_ids = _first_determinism_panel_ids(
        panel_list,
        full_schedule,
        methods_t,
    )
    schedule_by_panel: dict[str, tuple[dict[str, Any], ...]] = {}
    for panel in panel_list:
        schedule_by_panel[panel.panel_id] = tuple(row for row in full_schedule if row["panel_id"] == panel.panel_id)
    try:
        for model in models_t:
            matrix, cache_manifest = _load_cache(cache_dir, model, sample_hash)
            cache_identity[model] = {"path": cache_manifest.get("path"), "sha256": cache_manifest.get("sha256"), "shape": cache_manifest.get("shape"), "sample_ids_sha256": cache_manifest.get("sample_ids_sha256")}
            for replicate in replicates_t:
                role = roles[int(replicate)]
                selector_indices = np.asarray(role["selector"], dtype=np.int64)
                raw = np.asarray(matrix[selector_indices], dtype=np.float32)
                target = labels[selector_indices]
                banks = driver._paired_split_banks(raw, target, seed=SEED + int(replicate) + 11)
                nested = driver._nested_stratified_indices(target, budgets_t, seed=SEED + int(replicate) + 23)
                for arm_position, arm in enumerate(arms_t):
                    lam, nu = arm_lookup[arm]
                    transformed = driver._bridge_transform(raw, donor=banks["donor"], mode=banks["mode"], nuisance=banks["nuisance"], q=1.0, lam=lam, nu=nu)
                    for budget_position, budget in enumerate(budgets_t):
                        selected_rows = np.asarray(nested[int(budget)], dtype=np.int64)
                        subset = np.asarray(transformed[selected_rows], dtype=np.float32)
                        subset_target = target[selected_rows]
                        panel = manifest.FoodPanel(model, int(replicate), arm, int(budget))
                        panel_schedule = schedule_by_panel[panel.panel_id]
                        identity = _panel_identity(code_identity_sha256=provenance["code_identity_sha256"], protocol_sha256=str(protocol_hash), source_hashes=archived_hashes, cache_sha256=str(cache_manifest.get("sha256")), panel=panel, methods=methods_t, schedule=panel_schedule)
                        checkpoint = checkpoints / f"{panel.panel_id}.json"
                        if resume and checkpoint.exists():
                            cached = _validate_checkpoint(
                                checkpoint,
                                identity,
                                methods_t,
                                expected_determinism_panel_ids=first_determinism_panel_ids,
                            )
                            rows.extend(dict(row) for row in cached["rows"])
                            probe_rows.extend(dict(row) for row in cached.get("probe_rows", ()))
                            capped_probe_rows.extend(
                                dict(row) for row in cached.get("capped_probe_rows", ())
                            )
                            for key, value in cached["determinism_verification"].items():
                                if key in deterministic and deterministic[key] != value:
                                    raise RuntimeError("resume_checkpoint_determinism_mismatch")
                                deterministic[key] = dict(value)
                                if key == "LP-CAPPED-2048":
                                    capped_warmed = True
                                else:
                                    warmed.add(key)
                            continue
                        execution = tuple(
                            str(row["method_id"])
                            for row in sorted(
                                panel_schedule,
                                key=lambda row: int(row["execution_position"]),
                            )
                        )
                        if len(execution) != len(methods_t) or set(execution) != set(methods_t):
                            raise RuntimeError("execution schedule reconstruction mismatch")
                        block: list[dict[str, Any]] = []
                        local_deterministic: dict[str, dict[str, Any]] = {}
                        for order, method in enumerate(execution):
                            warmup_result = None
                            try:
                                if method not in warmed:
                                    if first_determinism_panel_ids.get(method) != panel.panel_id:
                                        raise RuntimeError(
                                            "determinism warm-up reached a later panel before its "
                                            "frozen first eligible panel"
                                        )
                                    warmup_result = _cross_fitted_score(subset, subset_target, candidate=method, seed=SEED + int(replicate))
                                    warmed.add(method)
                                outer_wall, outer_cpu = time.perf_counter(), time.process_time()
                                result = _cross_fitted_score(subset, subset_target, candidate=method, seed=SEED + int(replicate))
                                total_wall_elapsed = float(max(0.0, time.perf_counter() - outer_wall))
                                total_cpu_elapsed = float(max(0.0, time.process_time() - outer_cpu))
                                if warmup_result is not None:
                                    comparison = _determinism_comparison(warmup_result, result)
                                    comparison = _determinism_record(
                                        comparison,
                                        candidate_id=method,
                                        panel=panel,
                                    )
                                    local_deterministic[method] = comparison
                                    deterministic[method] = comparison
                                    if comparison["exact"] is not True:
                                        raise RuntimeError("nondeterministic warmup/measured structural result")
                                elif method in deterministic:
                                    # Every checkpoint carries the complete
                                    # candidate determinism record, even after
                                    # the one independent warm-up for a method
                                    # has already been consumed.
                                    local_deterministic[method] = dict(deterministic[method])
                                block.append({
                                    "stage": "smoke" if smoke else "development",
                                    "model": model,
                                    "backbone": model,
                                    "replicate": int(replicate),
                                    "arm": arm,
                                    "budget": int(budget),
                                    "n_rows": int(subset.shape[0]),
                                    "n_features": int(subset.shape[1]),
                                    "candidate_id": method,
                                    "candidate_name": result.get("candidate_name"),
                                    "mode": result.get("mode"),
                                    "prototype_refinement_enabled": bool(
                                        _candidate_spec(method)[3]
                                    ),
                                    "score": result.get("score"),
                                    "fit_wall_seconds": result.get("fit_wall_seconds"),
                                    "fit_cpu_seconds": result.get("fit_cpu_seconds"),
                                    "score_fixed_wall_seconds": result.get("score_fixed_wall_seconds"),
                                    "score_fixed_cpu_seconds": result.get("score_fixed_cpu_seconds"),
                                    "total_wall_seconds": total_wall_elapsed,
                                    "total_cpu_seconds": total_cpu_elapsed,
                                    "prototype_refinement": result.get("refinement", {}),
                                    "conditioning_diagnostics": result.get("conditioning", {}),
                                    "conditioning_runtime": result.get("conditioning_runtime", {}),
                                    "folds": result.get("folds", []),
                                    "execution_position": int(order),
                                    "execution_order": list(execution),
                                    "schedule_seed": manifest.SCHEDULE_SEED,
                                    "warmup_excluded": True,
                                    "status": "ok",
                                    "error": None,
                                })
                            except Exception as exc:
                                block.append({"stage": "smoke" if smoke else "development", "model": model, "backbone": model, "replicate": int(replicate), "arm": arm, "budget": int(budget), "candidate_id": method, "candidate_name": method, "execution_position": int(order), "execution_order": list(execution), "warmup_excluded": True, "status": "error", "error": f"{type(exc).__name__}: {exc}", "score": None})
                        if any(row.get("status") != "ok" for row in block):
                            checkpoint_rows = (
                                [_smoke_redact_row(row) for row in block]
                                if smoke else block
                            )
                            _atomic_write_json(checkpoint, {"identity": identity, "rows": checkpoint_rows, "probe_rows": [], "capped_probe_rows": [], "determinism_verification": local_deterministic, "artifact_status": "stopped"})
                            rows.extend(block)
                            raise RuntimeError("candidate_failure")
                        rows.extend(block)
                        for row in block:
                            if row["candidate_id"] == LP_METHOD:
                                probe_rows.append(dict(row))
                        capped_warmup_result = None
                        if not capped_warmed:
                            if first_determinism_panel_ids["LP-CAPPED-2048"] != panel.panel_id:
                                raise RuntimeError(
                                    "LP-CAPPED-2048 warm-up reached a later panel before its "
                                    "frozen first eligible panel"
                                )
                            capped_warmup_result = _capped_probe(
                                subset,
                                subset_target,
                                seed=SEED + int(replicate),
                            )
                            capped_warmed = True
                        capped = _capped_probe(
                            subset,
                            subset_target,
                                seed=SEED + int(replicate),
                        )
                        if capped_warmup_result is not None:
                            capped_comparison = _capped_determinism_comparison(
                                capped_warmup_result, capped
                            )
                            capped_comparison = _determinism_record(
                                capped_comparison,
                                candidate_id="LP-CAPPED-2048",
                                panel=panel,
                            )
                            local_deterministic["LP-CAPPED-2048"] = capped_comparison
                            deterministic["LP-CAPPED-2048"] = capped_comparison
                            if capped_comparison["exact"] is not True:
                                raise RuntimeError(
                                    "nondeterministic LP-CAPPED-2048 structural result"
                                )
                        elif "LP-CAPPED-2048" in deterministic:
                            local_deterministic["LP-CAPPED-2048"] = dict(
                                deterministic["LP-CAPPED-2048"]
                            )
                        capped_probe_rows.append(
                            {
                                "stage": "smoke" if smoke else "development",
                                "model": model,
                                "backbone": model,
                                "replicate": int(replicate),
                                "arm": arm,
                                "budget": int(budget),
                                "candidate_id": "LP-CAPPED-2048",
                                "candidate_name": "linear_probe_capped_2048",
                                "mode": "probe",
                                "prototype_refinement_enabled": False,
                                "prototype_refinement": {},
                                "conditioning_diagnostics": {},
                                "score": capped.get("score"),
                                "fit_wall_seconds": sum(
                                    float(row.get("fit_wall_seconds") or 0.0)
                                    for row in capped.get("folds", ())
                                ),
                                "fit_cpu_seconds": sum(
                                    float(row.get("fit_cpu_seconds") or 0.0)
                                    for row in capped.get("folds", ())
                                ),
                                "score_fixed_wall_seconds": sum(
                                    float(row.get("predict_wall_seconds") or 0.0)
                                    for row in capped.get("folds", ())
                                ),
                                "score_fixed_cpu_seconds": sum(
                                    float(row.get("predict_cpu_seconds") or 0.0)
                                    for row in capped.get("folds", ())
                                ),
                                "total_wall_seconds": capped.get("wall_seconds"),
                                "total_cpu_seconds": capped.get("cpu_seconds"),
                                "folds": capped.get("folds", []),
                                "execution_position": None,
                                "warmup_excluded": True,
                                "status": "ok",
                                "error": None,
                            }
                        )
                        # A/B are exact replay controls and are checked before
                        # proceeding to the next paired panel.
                        parity = _baseline_parity_rows(block, prior_replay)
                        probe_parity = _probe_parity_rows(block, prior_replay)
                        expected_parity_count = sum(
                            int(method in {"A", "B", LP_METHOD}) for method in methods_t
                        )
                        if len(parity) + len(probe_parity) != expected_parity_count:
                            raise RuntimeError(f"parity reference missing for {panel.panel_id}")
                        if not all(row["exact"] for row in parity + probe_parity):
                            if not all(row["exact"] for row in parity):
                                raise RuntimeError(f"baseline parity failure for {panel.panel_id}")
                            raise RuntimeError(f"LP-FULL parity failure for {panel.panel_id}")
                        # A checkpoint is terminal only after every required
                        # parity check has passed.  A parity mismatch therefore
                        # cannot leave a resumable "completed" checkpoint.
                        checkpoint_rows = (
                            [_smoke_redact_row(row) for row in block]
                            if smoke else block
                        )
                        checkpoint_probe_rows = (
                            [_smoke_redact_row(row) for row in probe_rows if row.get("model") == model and row.get("replicate") == int(replicate) and row.get("arm") == arm and row.get("budget") == int(budget)]
                            if smoke else [row for row in probe_rows if row.get("model") == model and row.get("replicate") == int(replicate) and row.get("arm") == arm and row.get("budget") == int(budget)]
                        )
                        checkpoint_capped_rows = (
                            [_smoke_redact_row(row) for row in capped_probe_rows if row.get("model") == model and row.get("replicate") == int(replicate) and row.get("arm") == arm and row.get("budget") == int(budget)]
                            if smoke else [row for row in capped_probe_rows if row.get("model") == model and row.get("replicate") == int(replicate) and row.get("arm") == arm and row.get("budget") == int(budget)]
                        )
                        _atomic_write_json(checkpoint, {"identity": identity, "rows": checkpoint_rows, "probe_rows": checkpoint_probe_rows, "capped_probe_rows": checkpoint_capped_rows, "determinism_verification": local_deterministic, "artifact_status": "completed"})
    except Exception as exc:
        message = str(exc).lower()
        if "parity" in message:
            stop_reason = "baseline_or_probe_parity_failure"
        elif "nondetermin" in message:
            stop_reason = "nondeterminism_failure"
        elif "hash" in message or "source" in message or "protocol" in message:
            stop_reason = "provenance_failure"
        else:
            stop_reason = "candidate_failure"
        _write_bundle(output_path, artifact_status="stopped", args=argparse.Namespace(models=models_t, replicates=replicates_t, budgets=budgets_t, arms=arms_t, methods=methods_t, smoke=smoke), provenance=provenance, source_hashes=source_paths, cache_identity=cache_identity, rows=rows, reference_rows=reference_rows, parity_rows=_baseline_parity_rows(rows, prior_replay) + _probe_parity_rows(rows, prior_replay), probe_rows=probe_rows, capped_probe_rows=capped_probe_rows, deterministic=deterministic, stop_reason=stop_reason, stop_error=str(exc))
        raise
    parity_rows = _baseline_parity_rows(rows, prior_replay) + _probe_parity_rows(rows, prior_replay)
    try:
        _validate_completed_surfaces(
            rows=rows,
            references=reference_rows,
            probe_rows=probe_rows,
            capped_rows=capped_probe_rows,
            parity_rows=parity_rows,
            deterministic=deterministic,
            models=models_t,
            replicates=replicates_t,
            budgets=budgets_t,
            arms=arms_t,
            methods=methods_t,
            smoke=smoke,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "parity" in message:
            stop_reason = "baseline_or_probe_parity_failure"
        elif "nondetermin" in message:
            stop_reason = "nondeterminism_failure"
        elif "hash" in message or "source" in message or "protocol" in message:
            stop_reason = "provenance_failure"
        else:
            stop_reason = "incomplete_or_nonfinite_surface"
        _write_bundle(
            output_path,
            artifact_status="stopped",
            args=argparse.Namespace(
                models=models_t,
                replicates=replicates_t,
                budgets=budgets_t,
                arms=arms_t,
                methods=methods_t,
                smoke=smoke,
            ),
            provenance=provenance,
            source_hashes=source_paths,
            cache_identity=cache_identity,
            rows=rows,
            reference_rows=reference_rows,
            parity_rows=parity_rows,
            probe_rows=probe_rows,
            capped_probe_rows=capped_probe_rows,
            deterministic=deterministic,
            stop_reason=stop_reason,
            stop_error=str(exc),
        )
        raise
    return _write_bundle(output_path, artifact_status="completed", args=argparse.Namespace(models=models_t, replicates=replicates_t, budgets=budgets_t, arms=arms_t, methods=methods_t, smoke=smoke), provenance=provenance, source_hashes=source_paths, cache_identity=cache_identity, rows=rows, reference_rows=reference_rows, parity_rows=parity_rows, probe_rows=probe_rows, capped_probe_rows=capped_probe_rows, deterministic=deterministic)


def _run(args: argparse.Namespace) -> int:
    # ``None`` means the user omitted that selector.  Smoke then receives the
    # exact declared structural subset; an explicit selector still goes
    # through the same strict validation in ``run_food101``.
    models_raw = args.models if args.models is not None else ",".join(
        SMOKE_MODELS if args.smoke else MODELS
    )
    replicates_raw = args.replicates if args.replicates is not None else ",".join(
        str(value) for value in (SMOKE_REPLICATES if args.smoke else REPLICATES)
    )
    budgets_raw = args.budgets if args.budgets is not None else ",".join(
        str(value) for value in (SMOKE_BUDGETS if args.smoke else BUDGETS)
    )
    arms_raw = args.arms if args.arms is not None else ",".join(
        SMOKE_ARMS if args.smoke else tuple(name for name, _, _ in ARMS)
    )
    methods_raw = args.methods if args.methods is not None else ",".join(METHODS)
    models = _parse_subset(models_raw, MODELS)
    replicates = _parse_subset(replicates_raw, REPLICATES, int)
    budgets = _parse_subset(budgets_raw, BUDGETS, int)
    arms = _parse_subset(arms_raw, tuple(name for name, _, _ in ARMS))
    methods = _parse_subset(methods_raw, METHODS)
    run_food101(output=args.output, driver_path=args.driver, source_result_path=args.source_result, source_cohort_path=args.source_cohort, prior_replay_path=args.prior_replay, runtime_source_path=args.runtime_source, cache_dir=args.cache_dir, models=models, replicates=replicates, budgets=budgets, arms=arms, methods=methods, smoke=bool(args.smoke), resume=bool(args.resume))
    return 0


def main() -> int:
    return _run(_parser().parse_args())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARMS", "BUDGETS", "CAP_ROWS", "COHORT_SHA256", "DEFAULT_CACHE", "DEFAULT_COHORT", "DEFAULT_DRIVER", "DEFAULT_PRIOR", "DEFAULT_RESULT", "DEFAULT_RUNTIME_SOURCE", "DRIVER_SHA256", "FOLDS", "K", "LP_METHOD", "METHODS", "MODELS", "OI_METHODS", "PRIOR_SHA256", "REPLICATES", "RESULT_SHA256", "RUNTIME_SOURCE_SHA256", "SEED", "SMOKE_ARMS", "SMOKE_BUDGETS", "SMOKE_MODELS", "SMOKE_REPLICATES", "SOURCE_CONFIGURATION_HASH", "_baseline_parity_rows", "_candidate_spec", "_capped_probe", "_config_sha256", "_cross_fitted_score", "_determinism_comparison", "_first_determinism_panel_ids", "_load_cache", "_load_driver", "_oi_kwargs", "_parser", "_probe_config_identity", "_recipe_bindings", "_row_l2", "_run", "_stratification_encoding", "_stratified_cap", "_stratified_folds", "canonical_json", "run_food101", "validate_cache_manifest", "validate_run_selection",
]
