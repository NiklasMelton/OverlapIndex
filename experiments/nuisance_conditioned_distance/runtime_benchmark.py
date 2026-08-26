"""Post-hoc large-budget Food-101 runtime benchmark.

This module is intentionally separate from :mod:`food101`.  It owns only the
resource-scaling surface: hash-locked Food-101 inputs, nested class-balanced
budgets, counterbalanced selector calls, and timing artifacts.  It does not
change the public ``OverlapIndex`` API and it has no import-time experiment
side effects.

The default (full) invocation requires a screen-locked C/D/E promotion
decision.  The locked candidate is timed alongside the immutable A/B controls,
the full uncapped probe, and the capped-probe component used by policy G.  A
``--smoke`` invocation may use a bounded in-memory panel for correctness checks;
its rows are never a full-stage result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Callable

# These controls must be installed before NumPy/scikit-learn/joblib imports.
_THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}
for _name, _value in _THREAD_ENVIRONMENT.items():
    os.environ[_name] = _value

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import Normalizer, StandardScaler  # noqa: E402

from overlapindex import OverlapIndex  # noqa: E402

from . import food101  # noqa: E402
from .conditioning_adapter import ConditionedOverlapIndex  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = food101.DEFAULT_COHORT
DEFAULT_CACHE_DIR = food101.DEFAULT_CACHE
DEFAULT_RUNTIME_DRIVER = Path(
    "/Users/niklasmelton/code/vertabrae/examples/food101_selector_runtime_scaling.py"
)
DEFAULT_RUNTIME_RAW = Path(
    "/Users/niklasmelton/.codex/worktrees/a9b5/vertabrae/examples/research/"
    "selector_replay/artifacts/runtime/raw_results.json"
)
RUNTIME_DRIVER_SHA256 = "72a10f57681c27229572f885805b06c73ea20b7b0240e5477980dbef6103e7b9"
RUNTIME_RAW_SHA256 = "98ec1a8e2a76332478567a2a4ce801cfbf077bb5c8fd7e1f8ce090e2783bd3c3"
COHORT_SHA256 = food101.COHORT_SHA256
RUNTIME_RAW_STUDY = "food101_overlap_refinement_selector_runtime"
STARTING_COMMIT = food101.STARTING_COMMIT
EXPECTED_BRANCH = food101.EXPECTED_BRANCH

MODELS = (
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
BUDGETS = (64, 128, 256, 512, 640)
REPEATS = tuple(range(5))
N_CLASSES = 40
TRAIN_ROWS = 26_400
COHORT_ROWS = 28_480
FOLDS = 5
K = 10
SEED = 42
CAP_ROWS = 2_048

# The six primitive methods are the complete smoke/development surface.  A
# full run narrows this to A/B/E plus the screen-locked candidate, full_probe,
# and the capped component; see ``primitive_method_ids`` below.
METHOD_IDS = ("A", "B", "C", "D", "E", "full_probe")
OI_METHOD_IDS = frozenset({"A", "B", "C", "D", "E"})
PROMOTION_CANDIDATES = frozenset({"C", "D", "E"})


@dataclass(frozen=True)
class RuntimeCandidate:
    candidate_id: str
    name: str
    mode: str | None
    prototype_refinement: bool | None
    direct: bool = False
    kind: str = "oi"


CANDIDATES = {
    "A": RuntimeCandidate("A", "oi_unrefined_raw", "none", False, direct=True),
    "B": RuntimeCandidate("B", "oi_refined_raw", "none", True, direct=True),
    "C": RuntimeCandidate(
        "C", "oi_refined_global_isotropy", "global_isotropy", True
    ),
    "D": RuntimeCandidate(
        "D", "oi_refined_pooled_diagonal", "pooled_diagonal", True
    ),
    "E": RuntimeCandidate("E", "oi_refined_pooled_full", "pooled_full", True),
    "full_probe": RuntimeCandidate(
        "full_probe", "linear_probe_oof", None, None, kind="full_probe"
    ),
    "capped_probe_component": RuntimeCandidate(
        "capped_probe_component", "capped_linear_probe", None, None, kind="probe"
    ),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256_path(path: Path | str) -> str:
    """Return a streaming SHA-256 for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _read_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = sorted({str(key) for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_safe(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else _json_safe(row.get(key))
                    for key in fields
                }
            )
    temporary.replace(path)


def _stratification_encoding(labels: Sequence[Any]) -> tuple[np.ndarray, tuple[Any, ...]]:
    """Encode scalar labels only for deterministic scikit-learn splitting."""

    target = np.asarray(labels)
    if target.ndim != 1:
        raise ValueError("labels must be a one-dimensional scalar-label array")
    encoded = np.empty(target.size, dtype=np.int64)
    classes: list[Any] = []
    positions: dict[Any, int] = {}
    for row, raw in enumerate(target):
        label = raw.item() if isinstance(raw, np.generic) else raw
        if isinstance(label, (list, tuple, set, frozenset, dict, np.ndarray)):
            raise ValueError("labels must contain hashable scalar values")
        try:
            position = positions.get(label)
        except TypeError as exc:
            raise ValueError("labels must contain hashable scalar values") from exc
        if position is None:
            position = len(classes)
            positions[label] = position
            classes.append(label)
        encoded[row] = position
    return encoded, tuple(classes)


def stratified_folds(
    labels: Sequence[Any], *, n_splits: int = FOLDS, seed: int = SEED
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return folds while preserving original labels for model fitting."""

    from sklearn.model_selection import StratifiedKFold

    encoded, _ = _stratification_encoding(labels)
    if encoded.size == 0:
        raise ValueError("labels must not be empty")
    splitter = StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(seed)
    )
    placeholder = np.zeros((encoded.size, 1), dtype=np.uint8)
    return tuple(
        (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
        for train, test in splitter.split(placeholder, encoded)
    )


def _label_seed(seed: int, label: Any) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).hexdigest()[:8]
    return int(seed) + int(digest, 16)


def validate_nested_indices(
    labels: Sequence[Any], budgets: Sequence[int], *, classes: Sequence[Any] | None = None
) -> tuple[Any, ...]:
    values = np.asarray(labels, dtype=object)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    requested = tuple(int(value) for value in budgets)
    if not requested or any(value < 1 for value in requested):
        raise ValueError("budgets must contain positive integers")
    if len(set(requested)) != len(requested):
        raise ValueError("budgets must be unique")
    if classes is not None:
        selected = tuple(classes)
    else:
        # ``np.unique`` sorts through native comparisons and therefore raises
        # for otherwise valid heterogeneous scalar labels (for example ``1``
        # and ``"one"``).  The archived planner's deterministic string sort
        # is label-encoding invariant and handles that case explicitly.
        observed: list[Any] = []
        for raw in values:
            label = raw.item() if isinstance(raw, np.generic) else raw
            if label not in observed:
                observed.append(label)
        selected = tuple(sorted(observed, key=lambda value: str(value)))
    if not selected:
        raise ValueError("at least one class is required")
    counts = {label: int(np.sum(values == label)) for label in selected}
    unknown = [label for label, count in counts.items() if count == 0]
    if unknown:
        raise ValueError(f"requested classes are absent from labels: {unknown!r}")
    missing = [label for label, count in counts.items() if count < max(requested)]
    if missing:
        raise ValueError(
            f"largest budget {max(requested)} exceeds rows for classes {missing!r}"
        )
    return selected


def build_nested_indices(
    labels: Sequence[Any],
    budgets: Sequence[int] = BUDGETS,
    *,
    seed: int = SEED,
    classes: Sequence[Any] | None = None,
) -> dict[int, np.ndarray]:
    """Build the archived per-class hash-seeded nested index panel."""

    values = np.asarray(labels, dtype=object)
    selected = validate_nested_indices(values, budgets, classes=classes)
    permutations = {
        label: np.random.default_rng(_label_seed(int(seed), label)).permutation(
            np.flatnonzero(values == label)
        )
        for label in selected
    }
    return {
        int(budget): np.sort(
            np.concatenate(
                [permutations[label][: int(budget)] for label in selected]
            ).astype(np.int64)
        )
        for budget in sorted(int(value) for value in budgets)
    }


def fresh_oi_kwargs(seed: int) -> dict[str, Any]:
    """Return a fresh exact backend mapping for every fit call."""

    return {
        "model_type": "MiniBatchKMeans",
        "kmeans_k": int(K),
        "kmeans_kwargs": {
            "batch_size": 256,
            "max_no_improvement": 5,
            "compute_labels": False,
            "n_init": 1,
            "init": "random",
            "random_state": int(seed),
        },
    }


def build_oi_candidate(candidate_id: str, *, seed: int) -> Any:
    """Construct one direct or conditioned OI with strict immutable controls."""

    try:
        spec = CANDIDATES[str(candidate_id)]
    except KeyError as exc:
        raise ValueError(f"unknown OI candidate {candidate_id!r}") from exc
    if spec.kind != "oi":
        raise ValueError(f"{candidate_id!r} is not an OI candidate")
    if type(spec.prototype_refinement) is not bool:
        raise TypeError("candidate prototype_refinement must be a strict Python bool")
    kwargs = fresh_oi_kwargs(int(seed))
    if spec.direct:
        return OverlapIndex(
            prototype_refinement=spec.prototype_refinement,
            **kwargs,
        )
    return ConditionedOverlapIndex(
        mode=str(spec.mode),
        prototype_refinement=spec.prototype_refinement,
        conditioning_kwargs={
            "condition_number_cap": 10_000.0,
            "relative_eigenvalue_floor": 1e-8,
        },
        overlap_index_kwargs=kwargs,
    )


def build_full_probe(seed: int) -> Any:
    """Build the archived full probe: row-L2 Normalizer + logistic head."""

    return make_pipeline(
        Normalizer(norm="l2"),
        LogisticRegression(
            C=1.0, max_iter=2_000, n_jobs=1, random_state=int(seed)
        ),
    )


def build_capped_probe(seed: int) -> Any:
    """Build the frozen StandardScaler capped-probe component used by G."""

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, max_iter=2_000, n_jobs=1, random_state=int(seed)
        ),
    )


def primitive_method_ids(
    promoted_candidate: str | None = None, *, all_candidates: bool = False
) -> tuple[str, ...]:
    """Return the measured primitive surface for a stage.

    ``all_candidates`` is used by smoke/development tests.  Full execution
    must provide one screen-locked C/D/E candidate and never retune a runner-up.
    """

    if all_candidates:
        return (*METHOD_IDS, "capped_probe_component")
    if promoted_candidate not in PROMOTION_CANDIDATES:
        raise ValueError(
            f"promoted_candidate must be one of {sorted(PROMOTION_CANDIDATES)!r}"
        )
    result = ["A", "B", "E"]
    if promoted_candidate != "E":
        result.append(str(promoted_candidate))
    result.extend(("full_probe", "capped_probe_component"))
    return tuple(result)


def counterbalanced_method_order(
    method_ids: Sequence[str], *, model_index: int, budget_index: int, repeat: int
) -> tuple[str, ...]:
    """Return a deterministic cyclic Latin order for one paired cell."""

    methods = tuple(str(value) for value in method_ids)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("method_ids must be a non-empty unique sequence")
    offset = (int(model_index) + int(budget_index) + int(repeat)) % len(methods)
    return methods[offset:] + methods[:offset]


def _summary_mapping(model: Any, name: str) -> dict[str, Any]:
    value = getattr(model, name, None)
    return dict(value) if isinstance(value, Mapping) else {}


def _timed_oi(
    matrix: np.ndarray,
    labels: Sequence[Any],
    *,
    candidate_id: str,
    seed: int,
    folds: int = FOLDS,
) -> dict[str, Any]:
    """Fit/score one OI candidate with direct clocks for each stage."""

    # Match the archived OverlapScoringConfig(normalize_embeddings=True)
    # geometry.  The probe keeps its own pipeline Normalizer below.
    values = food101._row_l2(np.asarray(matrix))
    target = np.asarray(labels, dtype=object)
    encoded, classes = _stratification_encoding(target)
    counts = np.bincount(encoded, minlength=len(classes))
    if int(np.min(counts)) < int(folds):
        raise ValueError("Every class must have at least the requested fold count")
    total_wall_start = perf_counter()
    total_cpu_start = process_time()
    fold_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for fold, (train, holdout) in enumerate(
        stratified_folds(target, n_splits=int(folds), seed=int(seed))
    ):
        fold_seed = int(seed) + int(fold)
        estimator = build_oi_candidate(candidate_id, seed=fold_seed)
        fit_wall_start, fit_cpu_start = perf_counter(), process_time()
        # Preserve original scalar labels for OI; encoded labels are only for
        # the fold planner and full-probe implementation.
        estimator.fit(values[train], target[train])
        fit_wall = float(perf_counter() - fit_wall_start)
        fit_cpu = float(process_time() - fit_cpu_start)
        score_wall_start, score_cpu_start = perf_counter(), process_time()
        score = float(estimator.score_fixed(values[holdout], target[holdout]))
        score_wall = float(perf_counter() - score_wall_start)
        score_cpu = float(process_time() - score_cpu_start)
        scores.append(score)
        fold_rows.append(
            {
                "fold": int(fold),
                "seed": fold_seed,
                "train_size": int(train.size),
                "holdout_size": int(holdout.size),
                "score": score,
                "fit_wall_seconds": fit_wall,
                "fit_cpu_seconds": fit_cpu,
                "score_fixed_wall_seconds": score_wall,
                "score_fixed_cpu_seconds": score_cpu,
                "refinement": _summary_mapping(estimator, "prototype_refinement_"),
                "conditioning": _summary_mapping(estimator, "conditioning_diagnostics_"),
            }
        )
    return {
        "candidate_id": str(candidate_id),
        "candidate_name": CANDIDATES[str(candidate_id)].name,
        "score": float(np.mean(scores)),
        "fit_wall_seconds": float(sum(row["fit_wall_seconds"] for row in fold_rows)),
        "fit_cpu_seconds": float(sum(row["fit_cpu_seconds"] for row in fold_rows)),
        "score_fixed_wall_seconds": float(
            sum(row["score_fixed_wall_seconds"] for row in fold_rows)
        ),
        "score_fixed_cpu_seconds": float(
            sum(row["score_fixed_cpu_seconds"] for row in fold_rows)
        ),
        "total_wall_seconds": float(perf_counter() - total_wall_start),
        "total_cpu_seconds": float(process_time() - total_cpu_start),
        "folds": fold_rows,
    }


def _timed_probe(
    matrix: np.ndarray,
    labels: Sequence[Any],
    *,
    seed: int,
    folds: int = FOLDS,
    maximum_rows: int | None = None,
) -> dict[str, Any]:
    """Run the full/capped probe with a direct total clock."""

    target = np.asarray(labels, dtype=object)
    selected = np.arange(target.size, dtype=np.int64)
    if maximum_rows is not None and selected.size > int(maximum_rows):
        # Reuse the archived deterministic per-class cap, preserving strings.
        encoded, classes = _stratification_encoding(target)
        base, extra = divmod(int(maximum_rows), len(classes))
        pieces: list[np.ndarray] = []
        for position, _label in enumerate(classes):
            available = np.flatnonzero(encoded == position)
            generator = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(position), int(target.size)])
            )
            pieces.append(generator.permutation(available)[: base + int(position < extra)])
        selected = np.sort(np.concatenate(pieces).astype(np.int64))
    selected_target = target[selected]
    encoded, classes = _stratification_encoding(selected_target)
    counts = np.bincount(encoded, minlength=len(classes))
    if int(np.min(counts)) < int(folds):
        raise ValueError("Every class must have at least the requested fold count")
    predictions = np.empty(encoded.size, dtype=np.int64)
    total_wall_start, total_cpu_start = perf_counter(), process_time()
    for train, holdout in stratified_folds(selected_target, n_splits=int(folds), seed=int(seed)):
        estimator = (
            build_capped_probe(int(seed))
            if maximum_rows is not None
            else build_full_probe(int(seed))
        )
        estimator.fit(np.asarray(matrix)[selected][train], encoded[train])
        predictions[holdout] = estimator.predict(np.asarray(matrix)[selected][holdout])
    return {
        "score": float(accuracy_score(encoded, predictions)),
        "n_rows": int(selected.size),
        "total_wall_seconds": float(perf_counter() - total_wall_start),
        "total_cpu_seconds": float(process_time() - total_cpu_start),
        "probe_rows_capped": maximum_rows is not None,
    }


def _runtime_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    """Structural result signature with all clock fields excluded."""

    excluded = {
        "fit_wall_seconds",
        "fit_cpu_seconds",
        "score_fixed_wall_seconds",
        "score_fixed_cpu_seconds",
        "total_wall_seconds",
        "total_cpu_seconds",
        "policy_wall_seconds",
        "policy_cpu_seconds",
    }

    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: strip(item) for key, item in value.items() if key not in excluded}
        if isinstance(value, (list, tuple)):
            return [strip(item) for item in value]
        return value

    return _json_safe(strip(result))


def benchmark_matrix(
    matrix: np.ndarray,
    labels: Sequence[Any],
    *,
    model: str = "backbone",
    budgets: Sequence[int] = BUDGETS,
    repeats: Sequence[int] = REPEATS,
    seed: int = SEED,
    folds: int = FOLDS,
    k: int = K,
    method_ids: Sequence[str] = METHOD_IDS,
    warmup: bool = True,
    model_index: int = 0,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Benchmark one already-loaded model matrix.

    This is the testable core.  ``k`` is validated against the frozen runtime
    value and is intentionally not threaded through the candidate constructor;
    all runtime candidates use the canonical ``kmeans_k=10`` mapping.
    """

    if int(k) != K:
        raise ValueError(f"runtime k is frozen at {K}; got {k!r}")
    values = np.asarray(matrix)
    target = np.asarray(labels, dtype=object)
    if values.ndim != 2 or values.shape[0] != target.size:
        raise ValueError("matrix and labels must have matching rows and a 2-D matrix")
    resolved_budgets = tuple(int(value) for value in budgets)
    resolved_repeats = tuple(int(value) for value in repeats)
    classes = validate_nested_indices(target, resolved_budgets)
    nested = build_nested_indices(target, resolved_budgets, seed=int(seed), classes=classes)
    methods = tuple(str(value) for value in method_ids)
    for method in methods:
        if method not in CANDIDATES:
            raise ValueError(f"unknown runtime method {method!r}")
    rows: list[dict[str, Any]] = []
    structural: dict[tuple[int, str], dict[str, Any]] = {}
    determinism_checks: dict[str, dict[str, Any]] = {}
    warmed: set[str] = set()
    warmup_log: list[dict[str, Any]] = []

    def execute(method: str, subset: np.ndarray, subset_target: np.ndarray, call_seed: int) -> dict[str, Any]:
        if method in OI_METHOD_IDS:
            return _timed_oi(subset, subset_target, candidate_id=method, seed=call_seed, folds=folds)
        if method == "full_probe":
            return _timed_probe(subset, subset_target, seed=call_seed, folds=folds)
        if method == "capped_probe_component":
            return _timed_probe(
                subset, subset_target, seed=call_seed, folds=folds, maximum_rows=CAP_ROWS
            )
        raise ValueError(f"unknown runtime method {method!r}")

    # One real warmup per method, performed on the smallest requested budget
    # and omitted from rows and all timing summaries.
    if warmup and methods:
        warm_budget = min(resolved_budgets)
        warm_subset = values[nested[warm_budget]]
        warm_target = target[nested[warm_budget]]
        for method in methods:
            warmup_result = execute(method, warm_subset, warm_target, int(seed) + warm_budget)
            warmed.add(method)
            warmup_log.append(
                {
                    "method": method,
                    "budget": warm_budget,
                    "structural_signature": _runtime_signature(warmup_result),
                }
            )
            if progress is not None:
                progress({"event": "warmup", "method": method, "budget": warm_budget})

    warmup_signatures = {
        str(record["method"]): record["structural_signature"]
        for record in warmup_log
    }

    for repeat in resolved_repeats:
        for budget_index, budget in enumerate(resolved_budgets):
            selected = nested[int(budget)]
            subset = values[selected]
            subset_target = target[selected]
            order = counterbalanced_method_order(
                methods,
                model_index=int(model_index),
                budget_index=int(budget_index),
                repeat=int(repeat),
            )
            block: list[dict[str, Any]] = []
            for order_position, method in enumerate(order):
                call_seed = int(seed) + int(budget)
                started_wall, started_cpu = perf_counter(), process_time()
                try:
                    result = execute(method, subset, subset_target, call_seed)
                    status, error = "ok", None
                except Exception as exc:
                    # The caller writes this row into a partial artifact before
                    # propagating the frozen early-stop condition.
                    result = {
                        "candidate_id": method,
                        "candidate_name": CANDIDATES[method].name,
                        "score": None,
                    }
                    status, error = "error", f"{type(exc).__name__}: {exc}"
                row = {
                    "model": str(model),
                    "backbone": str(model),
                    "budget": int(budget),
                    "samples_per_class": int(budget),
                    "n_samples": int(selected.size),
                    "embedding_dim": int(values.shape[1]),
                    "repeat": int(repeat),
                    "seed": call_seed,
                    "method": method,
                    "candidate_id": method,
                    "candidate_name": CANDIDATES[method].name,
                    "order_position": int(order_position),
                    "execution_order": int(order_position),
                    "status": status,
                    "error": error,
                    "warmup_excluded": True,
                    "outer_wall_seconds": float(perf_counter() - started_wall),
                    "outer_cpu_seconds": float(process_time() - started_cpu),
                    **result,
                }
                block.append(row)
                structural[(int(budget), method)] = _runtime_signature(result)
                if warmup and int(budget) == min(resolved_budgets) and int(repeat) == min(resolved_repeats):
                    warmup_signature = warmup_signatures.get(method)
                    measured_signature = _runtime_signature(result)
                    warmup_bytes = _canonical_json(warmup_signature)
                    measured_bytes = _canonical_json(measured_signature)
                    check = {
                        "exact": bool(warmup_bytes == measured_bytes),
                        "warmup_signature_sha256": hashlib.sha256(warmup_bytes).hexdigest(),
                        "measured_signature_sha256": hashlib.sha256(measured_bytes).hexdigest(),
                        "runtime_fields_excluded": True,
                    }
                    determinism_checks[method] = check
                    if not check["exact"]:
                        # Preserve the measured failure row in the partial
                        # artifact; the block-level frozen early stop below
                        # raises only after every paired row is serialized.
                        row["status"] = "error"
                        row["error"] = (
                            "warmup/measured structural nondeterminism for "
                            f"method {method!r}"
                        )
            rows.extend(block)
            if any(row["status"] != "ok" for row in block):
                raise RuntimeError(
                    "runtime candidate cell failed; frozen early stop: "
                    + "; ".join(
                        f"{row['method']}: {row['error']}"
                        for row in block
                        if row["status"] != "ok"
                    )
                )
            if progress is not None:
                progress({"event": "cell", "model": model, "budget": budget, "repeat": repeat})

    return {
        "rows": rows,
        "nested_indices": {str(key): value for key, value in nested.items()},
        "classes": list(classes),
        "warmup": warmup_log,
        "warmed_methods": sorted(warmed),
        "determinism": {
            "status": (
                "pass"
                if warmup and set(determinism_checks) == set(methods)
                and all(value["exact"] for value in determinism_checks.values())
                else "inconclusive"
            ),
            "exact": (
                True
                if warmup and set(determinism_checks) == set(methods)
                and all(value["exact"] for value in determinism_checks.values())
                else None
            ),
            "checks": determinism_checks,
            "structural_signatures": {
                f"{budget}:{method}": signature
                for (budget, method), signature in sorted(structural.items())
            },
            "runtime_fields_excluded": True,
        },
    }


def _labels_sha256(labels: Sequence[Any]) -> str:
    return hashlib.sha256(json.dumps([_json_safe(v) for v in labels], default=str).encode()).hexdigest()


def _load_cohort(path: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if sha256_path(path) != COHORT_SHA256:
        raise ValueError("Frozen Food-101 cohort SHA-256 mismatch")
    payload = _read_json(path)
    if payload.get("study") != "food101_nonlinear_backbone_bridge":
        raise ValueError("Frozen Food-101 cohort study identity mismatch")
    if payload.get("configuration_hash") != food101.SOURCE_CONFIGURATION_HASH:
        raise ValueError("Frozen Food-101 cohort configuration hash mismatch")
    ids_raw = payload.get("extracted_sample_ids")
    if not isinstance(ids_raw, list) or len(ids_raw) != COHORT_ROWS:
        raise ValueError("Frozen Food-101 cohort must contain exactly 28,480 rows")
    ids = [str(value) for value in ids_raw]
    if len(set(ids)) != len(ids):
        raise ValueError("Frozen Food-101 cohort sample IDs must be unique")
    train_rows = int(payload.get("extracted_train_rows", 0))
    test_rows = int(payload.get("extracted_test_rows", 0))
    if (train_rows, test_rows) != (TRAIN_ROWS, 2_080):
        raise ValueError("Frozen Food-101 cohort train/test counts are not 26,400/2,080")
    labels: list[str] = []
    for row, sample_id in enumerate(ids):
        pieces = sample_id.split("/")
        if len(pieces) < 4 or pieces[0] != "food101" or pieces[1] not in {"train", "test"}:
            raise ValueError(f"Unexpected Food-101 sample ID at row {row}: {sample_id!r}")
        if row < train_rows and pieces[1] != "train":
            raise ValueError("Frozen cohort train rows must precede test rows")
        if row >= train_rows and pieces[1] != "test":
            raise ValueError("Frozen cohort test rows must follow train rows")
        labels.append(pieces[2])
    train_labels = labels[:train_rows]
    class_values = sorted(set(train_labels))
    if len(class_values) != N_CLASSES or any(train_labels.count(value) != 660 for value in class_values):
        raise ValueError("Frozen cohort must contain 40 classes with 660 train rows each")
    return np.asarray(labels, dtype=object), ids, payload


def load_frozen_inputs(
    *,
    cohort_path: Path | str = DEFAULT_COHORT,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    runtime_raw_path: Path | str = DEFAULT_RUNTIME_RAW,
    runtime_driver_path: Path | str = DEFAULT_RUNTIME_DRIVER,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Verify archived runtime/cohort/cache identities and load train rows."""

    raw_path, driver_path, cache_root = Path(runtime_raw_path), Path(runtime_driver_path), Path(cache_dir)
    if sha256_path(raw_path) != RUNTIME_RAW_SHA256:
        raise ValueError("Archived runtime raw-results SHA-256 mismatch")
    if sha256_path(driver_path) != RUNTIME_DRIVER_SHA256:
        raise ValueError("Archived runtime driver SHA-256 mismatch")
    runtime_raw = _read_json(raw_path)
    if runtime_raw.get("artifact_status") != "completed" or runtime_raw.get("study") != RUNTIME_RAW_STUDY:
        raise ValueError("Archived runtime raw results are not the expected completed study")
    labels, sample_ids, cohort = _load_cohort(Path(cohort_path))
    sample_hash = hashlib.sha256(json.dumps(sample_ids).encode()).hexdigest()
    labels_hash = _labels_sha256(labels.tolist())
    expected_cache = runtime_raw.get("frozen_inputs", {}).get("cache_identity", {})
    matrices: dict[str, np.ndarray] = {}
    identities: dict[str, Any] = {}
    for model in MODELS:
        manifest_path = cache_root / f"food101_{model}_final.json"
        manifest = _read_json(manifest_path)
        if manifest.get("model") != model:
            raise ValueError(f"Cache model mismatch for {model}")
        if manifest.get("sample_ids_sha256") != sample_hash:
            raise ValueError(f"Cache sample identity mismatch for {model}")
        if manifest.get("labels_sha256") != labels_hash:
            raise ValueError(f"Cache label identity mismatch for {model}")
        matrix_path = Path(str(manifest["path"]))
        if not matrix_path.is_absolute():
            matrix_path = cache_root / matrix_path.name
        if sha256_path(matrix_path) != str(manifest.get("sha256")):
            raise ValueError(f"Cache matrix SHA-256 mismatch for {model}")
        matrix = np.load(matrix_path, mmap_mode="r")
        expected_shape = [COHORT_ROWS, int(matrix.shape[1])]
        if list(matrix.shape) != list(manifest.get("shape", expected_shape)):
            raise ValueError(f"Cache shape mismatch for {model}")
        if model in expected_cache:
            if str(expected_cache[model].get("sha256")) != str(manifest.get("sha256")):
                raise ValueError(f"Archived runtime cache identity mismatch for {model}")
        matrices[model] = np.asarray(matrix[:TRAIN_ROWS])
        identities[model] = {
            "path": str(matrix_path),
            "sha256": str(manifest.get("sha256")),
            "shape": list(matrix.shape),
            "sample_ids_sha256": sample_hash,
            "labels_sha256": labels_hash,
        }
    return matrices, labels[:TRAIN_ROWS], {
        "runtime_raw": {"path": str(raw_path), "sha256": RUNTIME_RAW_SHA256},
        "runtime_driver": {"path": str(driver_path), "sha256": RUNTIME_DRIVER_SHA256},
        "cohort": {"path": str(Path(cohort_path)), "sha256": COHORT_SHA256},
        "cohort_configuration_hash": cohort.get("configuration_hash"),
        "sample_ids_sha256": sample_hash,
        "labels_sha256": labels_hash,
        "cache_identity": identities,
    }


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE
    ).strip()


def source_hashes() -> dict[str, str]:
    paths = tuple(food101.EXPERIMENT_SOURCE_PATHS) + (
        "experiments/nuisance_conditioned_distance/runtime_benchmark.py",
    )
    result: dict[str, str] = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing experiment source file: {relative}")
        result[relative] = sha256_path(path)
    return result


def code_identity_sha256() -> str:
    """Hash declared source only; generated output/cache roots are excluded."""

    return hashlib.sha256(
        _canonical_json(
            {
                "schema": "declared_runtime_sources_v1",
                "starting_commit": STARTING_COMMIT,
                "source_hashes": source_hashes(),
            }
        )
    ).hexdigest()


def repository_provenance() -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    develop = _git("rev-parse", "develop")
    merge_base = _git("merge-base", "HEAD", "develop")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Experiment must run on {EXPECTED_BRANCH!r}; found {branch!r}")
    if develop != STARTING_COMMIT or merge_base != STARTING_COMMIT:
        raise RuntimeError(
            f"Recorded develop starting point mismatch: expected {STARTING_COMMIT}, "
            f"develop={develop}, merge-base={merge_base}"
        )
    status = _git("status", "--porcelain=v1").splitlines()
    hashes = source_hashes()
    return {
        "branch": branch,
        "starting_commit": STARTING_COMMIT,
        "develop_commit_at_execution": develop,
        "merge_base_with_develop": merge_base,
        "experiment_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_porcelain": status,
        "experiment_source_sha256": hashes,
        "code_identity_sha256": code_identity_sha256(),
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for name in ("numpy", "scipy", "scikit-learn", "overlapindex"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "thread_environment": dict(_THREAD_ENVIRONMENT),
    }


def _policy_rows(rows: Sequence[Mapping[str, Any]], *, promoted_candidate: str) -> list[dict[str, Any]]:
    """Compose actual frozen G policy rows after each repeat/budget panel."""

    result: list[dict[str, Any]] = []
    panels = sorted({(int(row["repeat"]), int(row["budget"])) for row in rows})
    for repeat, budget in panels:
        panel = [row for row in rows if int(row["repeat"]) == repeat and int(row["budget"]) == budget]
        by_model: dict[str, dict[str, Mapping[str, Any]]] = {}
        for row in panel:
            if row.get("status") == "ok":
                by_model.setdefault(str(row["model"]), {})[str(row["method"])] = row
        trigger = False
        for values in by_model.values():
            b = values.get("B")
            e = values.get("E")
            if b is None or e is None:
                continue
            refinement = b.get("folds", [])
            applied_rate = 0.0
            if refinement:
                before = sum(int(f.get("refinement", {}).get("prototype_count_before", 0) or 0) for f in refinement)
                applied = sum(int(f.get("refinement", {}).get("applied_count", 0) or 0) for f in refinement)
                applied_rate = float(applied / before) if before else 0.0
            if applied_rate >= 0.50 or abs(float(b.get("score", 0.0)) - float(e.get("score", 0.0))) >= 0.05:
                trigger = True
        for model, values in sorted(by_model.items()):
            b, e, locked = values.get("B"), values.get("E"), values.get(promoted_candidate)
            if b is None or e is None or locked is None:
                continue
            source = "capped_linear_probe" if trigger else promoted_candidate
            chosen = values.get("capped_probe_component") if trigger else locked
            # B+E are the diagnostic OI execution cost.  The locked candidate
            # is added only when distinct from E, as prescribed by the frozen
            # product policy; capped probe is a separately timed component.
            wall = float(b.get("total_wall_seconds", 0.0)) + float(e.get("total_wall_seconds", 0.0))
            cpu = float(b.get("total_cpu_seconds", 0.0)) + float(e.get("total_cpu_seconds", 0.0))
            if not trigger and promoted_candidate != "E":
                wall += float(locked.get("total_wall_seconds", 0.0))
                cpu += float(locked.get("total_cpu_seconds", 0.0))
            if trigger and chosen is not None:
                wall += float(chosen.get("total_wall_seconds", 0.0))
                cpu += float(chosen.get("total_cpu_seconds", 0.0))
            result.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "budget": budget,
                    "method": "G",
                    "candidate_id": "G",
                    "candidate_name": "panel_selective_capped_probe_guardrail",
                    "panel_triggered": bool(trigger),
                    "selected_score_source": source,
                    "policy_wall_seconds": wall,
                    "policy_cpu_seconds": cpu,
                    "status": "ok",
                    "composed_from": (
                        ["B", "E"]
                        + ([promoted_candidate] if not trigger and promoted_candidate != "E" else [])
                        + (["capped_probe_component"] if trigger else [])
                    ),
                }
            )
    return result


def _write_artifacts(output: Path, payload: Mapping[str, Any]) -> None:
    rows = list(payload.get("primitive_rows", ()))
    _atomic_json(output / "raw_results.json", payload)
    manifest = {
        key: payload[key]
        for key in (
            "schema_version",
            "artifact_status",
            "study",
            "configuration",
            "environment",
            "repository_provenance",
            "protocol",
            "sources",
            "cache_identity",
            "determinism",
            "grid_counts",
            "deviations",
        )
        if key in payload
    }
    _atomic_json(output / "manifest.json", manifest)
    _atomic_csv(output / "primitive_rows.csv", rows)
    _atomic_csv(output / "policy_rows.csv", list(payload.get("policy_rows", ())))
    _atomic_csv(output / "runtime_rows.csv", list(payload.get("runtime_rows", rows)))


def run_runtime_benchmark(
    *,
    output: Path | str,
    promotion_decision: Path | str | None = None,
    smoke: bool = False,
    models: Sequence[str] = MODELS,
    budgets: Sequence[int] = BUDGETS,
    repeats: Sequence[int] = REPEATS,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    cohort_path: Path | str = DEFAULT_COHORT,
    runtime_raw_path: Path | str = DEFAULT_RUNTIME_RAW,
    runtime_driver_path: Path | str = DEFAULT_RUNTIME_DRIVER,
    matrices: Mapping[str, np.ndarray] | None = None,
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run and atomically write a complete or explicitly partial artifact."""

    resolved_models = tuple(str(value) for value in models)
    resolved_budgets = tuple(int(value) for value in budgets)
    resolved_repeats = tuple(int(value) for value in repeats)
    promoted: str | None = None
    promotion_sha: str | None = None
    if promotion_decision is not None:
        promotion_path = Path(promotion_decision)
        promotion_payload = _read_json(promotion_path)
        promotion_sha = sha256_path(promotion_path)
        promoted = str(promotion_payload.get("selected_candidate"))
        if (
            promotion_payload.get("status") != "promoted_for_full_evaluation"
            or promotion_payload.get("decision_stage") != "screen"
            or promoted not in PROMOTION_CANDIDATES
        ):
            raise ValueError("promotion decision must be a valid screen-locked C/D/E decision")
    if not smoke and promoted is None:
        raise ValueError("full runtime requires a hashed screen promotion decision")
    if not smoke and (
        resolved_models != MODELS
        or resolved_budgets != BUDGETS
        or resolved_repeats != REPEATS
    ):
        raise ValueError("full runtime forbids model/budget/repeat subsets; use --smoke")
    method_ids = (
        primitive_method_ids(promoted) if not smoke else primitive_method_ids(promoted, all_candidates=True)
        if promoted is not None
        else primitive_method_ids(None, all_candidates=True)
    )
    provenance = repository_provenance()
    if matrices is None or labels is None:
        matrices, labels, sources = load_frozen_inputs(
            cohort_path=cohort_path,
            cache_dir=cache_dir,
            runtime_raw_path=runtime_raw_path,
            runtime_driver_path=runtime_driver_path,
        )
    else:
        matrices, labels, sources = dict(matrices), np.asarray(labels, dtype=object), {
            "in_memory_test_inputs": True
        }
    if not smoke:
        target = np.asarray(labels, dtype=object)
        if target.size != TRAIN_ROWS:
            raise ValueError(
                f"full runtime requires the {TRAIN_ROWS}-row Food-101 train cohort"
            )
        _, target_classes = _stratification_encoding(target)
        if len(target_classes) != N_CLASSES:
            raise ValueError(
                f"full runtime requires exactly {N_CLASSES} Food-101 classes"
            )
        for model in resolved_models:
            if model not in matrices or np.asarray(matrices[model]).shape[0] != TRAIN_ROWS:
                raise ValueError(f"full runtime matrix row-count mismatch for {model!r}")
    rows: list[dict[str, Any]] = []
    warmup_records: dict[str, Any] = {}
    determinism: dict[str, Any] = {}
    try:
        for model_index, model in enumerate(resolved_models):
            if model not in matrices:
                raise ValueError(f"missing matrix for model {model!r}")
            result = benchmark_matrix(
                matrices[model],
                labels,
                model=model,
                budgets=resolved_budgets,
                repeats=resolved_repeats,
                method_ids=method_ids,
                model_index=model_index,
            )
            rows.extend(result["rows"])
            warmup_records[model] = result["warmup"]
            determinism[model] = result["determinism"]
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "artifact_status": "stopped_candidate_error",
            "study": "food101_nuisance_conditioned_distance_runtime",
            "configuration": {
                "models": list(resolved_models),
                "budgets": list(resolved_budgets),
                "repeats": list(resolved_repeats),
                "methods": list(method_ids),
                "folds": FOLDS,
                "k": K,
                "seed": SEED,
                "serial_one_thread": True,
                "warmup_excluded": True,
                "smoke": bool(smoke),
                "promotion_decision_sha256": promotion_sha,
                "promoted_candidate": promoted,
                "promotion_decision": (
                    {"sha256": promotion_sha, "selected_candidate": promoted}
                    if promoted is not None
                    else None
                ),
            },
            "grid_counts": {
                "expected_cells": len(resolved_models) * len(resolved_budgets) * len(resolved_repeats),
                "expected_primitive_rows": len(resolved_models) * len(resolved_budgets) * len(resolved_repeats) * len(method_ids),
                "observed_primitive_rows": len(rows),
                "complete": False,
            },
            "environment": _environment(),
            "repository_provenance": provenance,
            "protocol": {"path": str(food101.PROTOCOL_PATH), "sha256": sha256_path(food101.PROTOCOL_PATH)},
            "sources": sources,
            "primitive_rows": rows,
            "policy_rows": [],
            "runtime_rows": rows,
            "determinism": determinism,
            "deviations": ["candidate error caused frozen early stop", f"{type(exc).__name__}: {exc}"],
        }
        _write_artifacts(Path(output).resolve(), payload)
        raise
    policy_rows = _policy_rows(rows, promoted_candidate=promoted) if promoted is not None else []
    # Keep the capped component before the full probe in the analysis-facing
    # flattened view.  The analysis reader intentionally maps the component
    # to ``G_probe_component`` and then excludes it from full-probe ratios; the
    # full-probe row must therefore be the final value for that grouped key.
    analysis_rows = [
        row for row in rows if row.get("method") == "capped_probe_component"
    ] + [row for row in rows if row.get("method") != "capped_probe_component"] + policy_rows
    payload = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": "food101_nuisance_conditioned_distance_runtime",
        "retrospective": True,
        "configuration": {
            "models": list(resolved_models),
            "budgets": list(resolved_budgets),
            "repeats": list(resolved_repeats),
            "methods": list(method_ids),
            "folds": FOLDS,
            "k": K,
            "seed": SEED,
            "n_classes": N_CLASSES,
            "serial_one_thread": True,
            "warmup_excluded": True,
            "counterbalanced": True,
            "promotion_decision_sha256": promotion_sha,
            "promoted_candidate": promoted,
            "promotion_decision": (
                {"sha256": promotion_sha, "selected_candidate": promoted}
                if promoted is not None
                else None
            ),
            "smoke": bool(smoke),
        },
        "grid_counts": {
            "expected_cells": len(resolved_models) * len(resolved_budgets) * len(resolved_repeats),
            "expected_primitive_rows": len(resolved_models) * len(resolved_budgets) * len(resolved_repeats) * len(method_ids),
            "observed_primitive_rows": len(rows),
            "expected_policy_rows": len(resolved_budgets) * len(resolved_repeats) * len(resolved_models)
            if promoted is not None
            else 0,
            "complete": True,
        },
        "environment": _environment(),
        "repository_provenance": provenance,
        "protocol": {"path": str(food101.PROTOCOL_PATH), "sha256": sha256_path(food101.PROTOCOL_PATH)},
        "sources": sources,
        "cache_identity": sources.get("cache_identity", {}),
        "warmup": warmup_records,
        "primitive_rows": rows,
        "policy_rows": policy_rows,
        "runtime_rows": analysis_rows,
        "determinism": determinism,
        "memory": {
            "status": "unavailable",
            "reason": "No fresh one-process-per-cell peak RSS measurement was collected.",
        },
        "deviations": [
            "The resource panel is retrospective post-hoc evidence, not untouched confirmation.",
            "Peak memory is unavailable; process-wide parent ru_maxrss is not reported.",
            "Policy G rows compose direct primitive timings and keep the capped component separate.",
        ],
    }
    _write_artifacts(Path(output).resolve(), payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion-decision", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--runtime-raw", type=Path, default=DEFAULT_RUNTIME_RAW)
    parser.add_argument("--runtime-driver", type=Path, default=DEFAULT_RUNTIME_DRIVER)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--budgets", default=",".join(str(value) for value in BUDGETS))
    parser.add_argument("--repeats", default=",".join(str(value) for value in REPEATS))
    parser.add_argument("--smoke", action="store_true")
    return parser


def _parse_csv(raw: str, cast: Callable[[str], Any]) -> tuple[Any, ...]:
    values = tuple(cast(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError(f"invalid comma-separated subset {raw!r}")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_runtime_benchmark(
            output=args.output,
            promotion_decision=args.promotion_decision,
            smoke=bool(args.smoke),
            models=_parse_csv(args.models, str),
            budgets=_parse_csv(args.budgets, int),
            repeats=_parse_csv(args.repeats, int),
            cache_dir=args.cache_dir,
            cohort_path=args.cohort,
            runtime_raw_path=args.runtime_raw,
            runtime_driver_path=args.runtime_driver,
        )
    except Exception as exc:
        print(f"runtime benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Runtime benchmark complete: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
