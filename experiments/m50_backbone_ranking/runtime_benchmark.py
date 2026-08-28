"""Runtime/resource benchmark for the locked M50 backbone-ranking chain.

Runtime is a separate stage from Food selector accuracy.  It uses a fresh
process for each measured method/cell, excludes an in-process warm-up from all
reported clocks, and records fit, conditioning/refinement, fixed-score, and
peak-memory components independently.  Every provisional runtime stage uses
the exact six primitive methods ``A``, ``B``, ``M0-SW``, ``M1-SW``, ``LP-FULL``,
and ``LP-CAPPED-2048``.  Fusion ``F`` and guardrail ``G`` are derived from
those primitive rows and are never estimator IDs here.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

import numpy as np

from . import candidates, food101, manifest


RUNTIME_BUDGETS = manifest.RUNTIME_BUDGETS
RUNTIME_REPEATS = manifest.RUNTIME_REPEATS
RUNTIME_PRIMITIVE_METHODS = manifest.RUNTIME_PRIMITIVE_METHODS
DEFAULT_METHODS = RUNTIME_PRIMITIVE_METHODS
OPTIONAL_METHODS: tuple[str, ...] = ()
SCHEMA_VERSION = manifest.SCHEMA_VERSION
RUNTIME_COHORT_ROWS = 26_400
RUNTIME_CLASSES = 40
RUNTIME_ROWS_PER_CLASS = 660
RUNTIME_NESTED_INDEX_SEED = food101.SEED
RUNTIME_BRIDGE_BANK_SEED = food101.SEED + 11

_THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}


def canonical_json(value: Any) -> str:
    return manifest.canonical_json(value)


def _json_safe(value: Any) -> Any:
    return manifest._json_safe(value)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, (canonical_json(payload) + "\n").encode("utf-8"))


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _table_manifest(rows: Sequence[Mapping[str, Any]], payload: bytes) -> dict[str, Any]:
    return {
        "row_count": int(len(rows)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "encoding": "canonical_jsonl_utf8",
    }


def _validate_methods(methods: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in methods)
    if values != RUNTIME_PRIMITIVE_METHODS:
        raise ValueError(
            "runtime methods must exactly match the frozen primitive schedule "
            f"{RUNTIME_PRIMITIVE_METHODS!r}; got {values!r}"
        )
    return values


def _peak_memory_mb() -> float:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    bytes_value = value if sys.platform == "darwin" else value * 1024
    return float(max(0, bytes_value) / (1024.0 * 1024.0))


def _candidate_for_runtime(method_id: str, seed: int, k_per_class: Mapping[Any, int]) -> Any:
    kwargs = food101._oi_kwargs(k_per_class, int(seed))
    if method_id in {"A", "B", "M0-SW", "M1-SW"}:
        # The canonical candidate table owns direct A/B construction and the
        # experiment-local conditioned construction.  It deep-copies the
        # already fresh per-fold kwargs before handing them to the estimator.
        return candidates.build_candidate(method_id, deepcopy(kwargs))
    if method_id in {"LP-FULL", "LP-CAPPED-2048"}:
        return None
    raise ValueError(f"runtime method {method_id!r} is not an estimator component")


def _direct_probe(
    values: np.ndarray,
    labels: np.ndarray,
    seed: int,
    *,
    capped: bool = False,
) -> dict[str, Any]:
    """Run the exact LP recipe, capping only the explicit policy comparator."""

    selected = (
        food101._stratified_cap(labels, food101.CAP_ROWS, seed)
        if capped and values.shape[0] > food101.CAP_ROWS
        else np.arange(values.shape[0], dtype=np.int64)
    )
    # LP-FULL uses every supplied row. LP-CAPPED-2048 is the policy component.
    return food101._lp_fold_score(
        values[selected],
        np.asarray(labels)[selected],
        int(seed),
        candidate_id="LP-CAPPED-2048" if capped else "LP-FULL",
        cap_rows=food101.CAP_ROWS if capped else None,
    )


def _fold_k_per_class(labels: np.ndarray) -> dict[Any, int]:
    encoded, classes = food101._stratification_encoding(np.asarray(labels))
    counts = np.bincount(encoded, minlength=len(classes))
    return {
        label: min(food101.K, max(1, int(count) // 5), int(count))
        for label, count in zip(classes, counts)
    }


def _oi_crossfit_runtime(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    method: str,
    seed: int,
    k_per_class: Mapping[Any, int],
    estimator_factory: Optional[Callable[[str, int, Mapping[Any, int]], Any]],
) -> dict[str, Any]:
    """Run the five-fold OI fit/score recipe on already-normalized values.

    The archived selector normalizes the complete panel once, before it
    creates the folds.  Keeping that operation outside this helper prevents
    an accidental normalization pass for every fold and makes the stage
    clocks describe OI fitting/scoring rather than repeated preprocessing.
    """

    folds = food101._stratified_folds(labels, n_splits=food101.FOLDS, seed=int(seed))
    fold_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    conditioning_wall = conditioning_cpu = 0.0
    fit_wall_total = fit_cpu_total = 0.0
    oi_fit_wall_total = oi_fit_cpu_total = 0.0
    score_wall_total = score_cpu_total = 0.0
    refinement_totals: dict[str, int] = {
        "prototype_count_before": 0,
        "prototype_count_after": 0,
        "eligible_count": 0,
        "attempted_count": 0,
        "applied_count": 0,
        "skipped_count": 0,
    }
    conditioning_rows: list[Mapping[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        fold_kwargs = _fold_k_per_class(np.asarray(labels)[train])
        if not fold_kwargs:
            fold_kwargs = deepcopy(dict(k_per_class))
        resolved_oi_kwargs = food101._oi_kwargs(fold_kwargs, fold_seed)
        config_identity = candidates.candidate_config_identity(method, resolved_oi_kwargs)
        config_sha256 = candidates.candidate_config_sha256(method, resolved_oi_kwargs)
        if estimator_factory is not None:
            model = estimator_factory(method, fold_seed, deepcopy(fold_kwargs))
        else:
            model = _candidate_for_runtime(method, fold_seed, fold_kwargs)
        if model is None:
            raise ValueError(f"runtime method {method!r} has no OI estimator")
        fit_wall, fit_cpu = time.perf_counter(), time.process_time()
        model.fit(values[train], labels[train])
        fit_wall_elapsed = max(0.0, time.perf_counter() - fit_wall)
        fit_cpu_elapsed = max(0.0, time.process_time() - fit_cpu)
        runtime = getattr(model, "runtime_diagnostics_", {})
        runtime = runtime if isinstance(runtime, Mapping) else {}
        c_wall = float(runtime.get("conditioning_fit_wall_seconds", 0.0))
        c_cpu = float(runtime.get("conditioning_fit_cpu_seconds", 0.0))
        oi_wall_value = runtime.get("oi_fit_wall_seconds")
        oi_cpu_value = runtime.get("oi_fit_cpu_seconds")
        score_wall, score_cpu = time.perf_counter(), time.process_time()
        score = float(model.score_fixed(values[holdout], labels[holdout]))
        score_wall_elapsed = max(0.0, time.perf_counter() - score_wall)
        score_cpu_elapsed = max(0.0, time.process_time() - score_cpu)
        scores.append(score)
        conditioning_wall += max(0.0, c_wall)
        conditioning_cpu += max(0.0, c_cpu)
        fit_wall_total += fit_wall_elapsed
        fit_cpu_total += fit_cpu_elapsed
        # Conditioned adapters expose the OI-only clock separately from their
        # conditioning clock.  Raw controls have no adapter diagnostic, so
        # their external fit clock is the correct OI fit component.
        oi_fit_wall_total += (
            float(oi_wall_value)
            if oi_wall_value is not None
            else fit_wall_elapsed
        )
        oi_fit_cpu_total += (
            float(oi_cpu_value)
            if oi_cpu_value is not None
            else fit_cpu_elapsed
        )
        score_wall_total += score_wall_elapsed
        score_cpu_total += score_cpu_elapsed
        refinement = getattr(model, "prototype_refinement_", {})
        refinement = dict(refinement) if isinstance(refinement, Mapping) else {}
        for key in refinement_totals:
            value = refinement.get(key, 0)
            if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                refinement_totals[key] += int(value)
        conditioning = getattr(model, "conditioning_diagnostics_", {})
        conditioning = dict(conditioning) if isinstance(conditioning, Mapping) else {}
        conditioning_rows.append(conditioning)
        fold_rows.append(
            {
                "fold": int(fold),
                "fold_seed": fold_seed,
                "train_size": int(train.size),
                "holdout_size": int(holdout.size),
                "score": score,
                "fit_wall_seconds": fit_wall_elapsed,
                "fit_cpu_seconds": fit_cpu_elapsed,
                "conditioning_fit_wall_seconds": max(0.0, c_wall),
                "conditioning_fit_cpu_seconds": max(0.0, c_cpu),
                "oi_fit_wall_seconds": max(
                    0.0,
                    float(oi_wall_value)
                    if oi_wall_value is not None
                    else fit_wall_elapsed,
                ),
                "oi_fit_cpu_seconds": max(
                    0.0,
                    float(oi_cpu_value)
                    if oi_cpu_value is not None
                    else fit_cpu_elapsed,
                ),
                "score_fixed_wall_seconds": score_wall_elapsed,
                "score_fixed_cpu_seconds": score_cpu_elapsed,
                "prototype_refinement": refinement,
                "conditioning_diagnostics": conditioning,
                "candidate_config_identity": config_identity,
                "candidate_config_sha256": config_sha256,
            }
        )
    before = refinement_totals["prototype_count_before"]
    refinement_totals["applied_rate"] = (
        float(refinement_totals["applied_count"] / before) if before else 0.0
    )
    return {
        "fit_wall_seconds": fit_wall_total,
        "fit_cpu_seconds": fit_cpu_total,
        "conditioning_fit_wall_seconds": conditioning_wall,
        "conditioning_fit_cpu_seconds": conditioning_cpu,
        "score_fixed_wall_seconds": score_wall_total,
        "score_fixed_cpu_seconds": score_cpu_total,
        # ``measure_runtime`` replaces these with an outer clock around the
        # complete cross-fit call.  The sums remain component diagnostics, not
        # the primary end-to-end measurement.
        "total_wall_seconds": fit_wall_total + score_wall_total,
        "total_cpu_seconds": fit_cpu_total + score_cpu_total,
        "oi_fit_cpu_seconds": oi_fit_cpu_total,
        "oi_fit_wall_seconds": oi_fit_wall_total,
        "candidate_score": float(np.mean(scores)),
        "prototype_refinement": refinement_totals,
        "conditioning_diagnostics": list(conditioning_rows),
        "folds": fold_rows,
        "peak_memory_mb": float(_peak_memory_mb()),
        "warmup_excluded": True,
    }


def measure_runtime(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    method_id: str,
    seed: int,
    k_per_class: Mapping[Any, int],
    estimator_factory: Optional[Callable[[str, int, Mapping[Any, int]], Any]] = None,
    warmup: bool = True,
) -> dict[str, Any]:
    """Measure one method with independent fit/conditioning/score clocks.

    The optional factory is a focused-test seam.  It receives a freshly copied
    k mapping and may return a fake estimator implementing ``fit`` and
    ``score_fixed``.  Warm-up execution is never included in returned times.
    """

    method = str(method_id)
    if method in {"F", "G"}:
        raise ValueError(
            f"{method} is a derived policy row; measure its primitive components "
            "and aggregate it outside the estimator factory"
        )
    if method not in set(RUNTIME_PRIMITIVE_METHODS):
        raise ValueError(f"unknown runtime method {method!r}")
    # Callers provide pre-normalization frozen subsets.  The archived OI
    # recipe applies one panel-wide row-L2 pass inside the outer measured
    # selector call, before fold construction.  LP applies its one prescribed
    # Normalizer inside its LogisticRegression pipeline.
    values_arr = np.asarray(values, dtype=np.float32)
    labels_arr = np.asarray(labels)
    if values_arr.ndim != 2 or labels_arr.ndim != 1 or values_arr.shape[0] != labels_arr.size:
        raise ValueError("runtime values and labels must be aligned 2D/1D arrays")
    if warmup:
        if method in {"LP-FULL", "LP-CAPPED-2048"}:
            _direct_probe(values_arr, labels_arr, int(seed), capped=method == "LP-CAPPED-2048")
        else:
            normalized = food101._row_l2(values_arr)
            _oi_crossfit_runtime(
                normalized,
                labels_arr,
                method=method,
                seed=int(seed),
                k_per_class=k_per_class,
                estimator_factory=estimator_factory,
            )
    if method in {"LP-FULL", "LP-CAPPED-2048"}:
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        probe = _direct_probe(values_arr, labels_arr, int(seed), capped=method == "LP-CAPPED-2048")
        elapsed_wall, elapsed_cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
        return {
            "candidate_id": method,
            "fit_wall_seconds": probe.get("fit_wall_seconds"),
            "fit_cpu_seconds": probe.get("fit_cpu_seconds"),
            "conditioning_fit_wall_seconds": 0.0,
            "conditioning_fit_cpu_seconds": 0.0,
            "score_fixed_wall_seconds": probe.get("score_fixed_wall_seconds"),
            "score_fixed_cpu_seconds": probe.get("score_fixed_cpu_seconds"),
            "total_wall_seconds": float(max(0.0, elapsed_wall)),
            "total_cpu_seconds": float(max(0.0, elapsed_cpu)),
            "candidate_score": probe.get("score"),
            "folds": probe.get("folds", []),
            "prototype_refinement": {},
            "conditioning_diagnostics": {},
            "peak_memory_mb": float(_peak_memory_mb()),
            "peak_memory_includes_warmup": True,
            "primary_clock": "outer_probe_call",
            "normalization_in_clock": "pipeline_normalizer",
            "warmup_excluded": True,
        }
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    normalized = food101._row_l2(values_arr)
    measured = _oi_crossfit_runtime(
        normalized,
        labels_arr,
        method=method,
        seed=int(seed),
        k_per_class=k_per_class,
        estimator_factory=estimator_factory,
    )
    measured["total_wall_seconds"] = float(
        max(0.0, time.perf_counter() - started_wall)
    )
    measured["total_cpu_seconds"] = float(
        max(0.0, time.process_time() - started_cpu)
    )
    measured["primary_clock"] = "outer_crossfit_call"
    measured["peak_memory_includes_warmup"] = True
    measured["normalization_in_clock"] = "outer_crossfit_before_fold_clocks"
    return {"candidate_id": method, **measured}


def _fresh_process_measure(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    method_id: str,
    seed: int,
    k_per_class: Mapping[Any, int],
    data_path: Path | None = None,
) -> dict[str, Any]:
    """Run warm-up + measured call in a fresh worker process.

    A caller running a complete block may provide one immutable NPZ so the
    six method workers do not rewrite the same feature matrix six times.
    """

    if data_path is None:
        with tempfile.TemporaryDirectory(prefix="m50-runtime-") as temp_dir:
            owned_path = Path(temp_dir) / "data.npz"
            np.savez(
                owned_path,
                values=np.asarray(values, dtype=np.float32),
                labels=np.asarray(labels, dtype=object),
            )
            return _fresh_process_measure(
                values,
                labels,
                method_id=method_id,
                seed=seed,
                k_per_class=k_per_class,
                data_path=owned_path,
            )
    request = {
        "values_path": str(data_path),
        "method_id": str(method_id),
        "seed": int(seed),
        "k_per_class": dict(k_per_class),
    }
    command = [sys.executable, "-m", "experiments.m50_backbone_ranking.runtime_benchmark", "--worker"]
    completed = subprocess.run(
        command,
        input=canonical_json(request) + "\n",
        text=True,
        capture_output=True,
        env={**os.environ, **_THREAD_ENVIRONMENT},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"runtime worker failed: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("runtime worker returned malformed JSON") from exc
    if not isinstance(payload, dict) or payload.get("warmup_excluded") is not True:
        raise RuntimeError("runtime worker response violates warmup contract")
    return payload


def _worker_main() -> int:
    request = json.loads(sys.stdin.read())
    if not isinstance(request, Mapping):
        raise ValueError("runtime worker request must be an object")
    with np.load(str(request["values_path"]), allow_pickle=True) as loaded:
        values = np.asarray(loaded["values"])
        labels = np.asarray(loaded["labels"], dtype=object)
    result = measure_runtime(values, labels, method_id=str(request["method_id"]), seed=int(request["seed"]), k_per_class=dict(request.get("k_per_class", {})), warmup=True)
    sys.stdout.write(canonical_json(result) + "\n")
    return 0


def _runtime_identity(
    *,
    code_identity_sha256: str,
    protocol_sha256: str,
    provisional_decision_sha256: str,
    provisional_analysis_manifest_sha256: str,
    methods: Sequence[str],
    arms: Sequence[str] = manifest.RUNTIME_ARMS,
    provenance: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    source_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "code_identity_sha256": str(code_identity_sha256),
        "protocol_sha256": str(protocol_sha256),
        "provisional_decision_sha256": str(provisional_decision_sha256),
        "provisional_analysis_manifest_sha256": str(provisional_analysis_manifest_sha256),
        "models": list(manifest.MODELS),
        "arms": [str(value) for value in arms],
        "budgets": list(RUNTIME_BUDGETS),
        "repeats": list(RUNTIME_REPEATS),
        "methods": list(methods),
        "warmup_excluded": True,
        "fresh_process_peak_memory": True,
        "dataset_recipe": {
            "cohort_rows": RUNTIME_COHORT_ROWS,
            "class_count": RUNTIME_CLASSES,
            "rows_per_class": RUNTIME_ROWS_PER_CLASS,
            "nested_index_seed": RUNTIME_NESTED_INDEX_SEED,
            "bridge_bank_seed": RUNTIME_BRIDGE_BANK_SEED,
            "nested_budgets": list(RUNTIME_BUDGETS),
            "normalization": "raw_factory_output; one_panel_wide_oi_row_l2_inside_outer_clock; lp_pipeline_normalizer",
        },
        "deterministic_seed_rule": {
            "runtime_call_seed": "SEED + budget",
            "oi_fold_seed": "runtime_call_seed + fold",
            "probe_split_seed": "runtime_call_seed",
            "probe_model_random_state": "runtime_call_seed",
            "capped_subset_seed": "runtime_call_seed",
            "repeat_effect": "order_and_timing_only",
        },
    }
    # Runtime is resumable, so retain the verified provenance and environment
    # in the immutable identity itself.  No caller-supplied hash is trusted;
    # run_runtime derives these values from the current repository before any
    # dataset factory call.
    if provenance is not None:
        identity["repository_provenance"] = dict(provenance)
    if environment is not None:
        identity["environment"] = dict(environment)
    if source_evidence is not None:
        identity["source_evidence"] = dict(source_evidence)
    return identity


def _runtime_block_identity(
    *,
    runtime_identity: Mapping[str, Any],
    model: str,
    arm: str,
    budget: int,
    repeat: int,
    schedule: Sequence[Mapping[str, Any]],
    values: np.ndarray,
    labels: np.ndarray,
    dataset_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the immutable identity carried by one complete runtime block."""

    value_bytes = np.ascontiguousarray(np.asarray(values)).tobytes()
    label_bytes = canonical_json(np.asarray(labels).tolist()).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime_identity": dict(runtime_identity),
        "model": str(model),
        "arm": str(arm),
        "budget": int(budget),
        "repeat": int(repeat),
        "method_order": [
            str(row["method_id"])
            for row in sorted(schedule, key=lambda row: int(row["execution_position"]))
        ],
        "schedule": [dict(row) for row in schedule],
        "call_seed": int(food101.SEED + int(budget)),
        "row_count": int(np.asarray(values).shape[0]),
        "feature_count": int(np.asarray(values).shape[1]),
        "values_sha256": hashlib.sha256(value_bytes).hexdigest(),
        "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "dataset_metadata": dict(dataset_metadata or {}),
        "warmup_excluded": True,
    }


def _validate_runtime_block(
    payload: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate a complete block checkpoint before reusing its rows."""

    if payload.get("artifact_status") != "completed":
        raise RuntimeError("runtime checkpoint is not completed")
    observed_identity = payload.get("identity")
    if observed_identity != dict(identity):
        raise RuntimeError("runtime checkpoint identity mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RuntimeError("runtime checkpoint rows are malformed")
    materialized = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(materialized) != len(rows) or len(materialized) != len(methods):
        raise RuntimeError("runtime checkpoint has incomplete method rows")
    expected_order_value = identity.get("method_order")
    if not isinstance(expected_order_value, Sequence) or isinstance(expected_order_value, (str, bytes)):
        raise RuntimeError("runtime checkpoint identity has no scheduled method order")
    expected_order = tuple(str(value) for value in expected_order_value)
    if len(expected_order) != len(methods) or set(expected_order) != set(str(value) for value in methods):
        raise RuntimeError("runtime checkpoint identity has an invalid method set")
    positions: list[int] = []
    for row in materialized:
        try:
            position = int(row.get("execution_position"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("runtime checkpoint has an invalid execution position") from exc
        positions.append(position)
    if sorted(positions) != list(range(len(methods))):
        raise RuntimeError("runtime checkpoint execution positions are incomplete/duplicated")
    observed_ids = tuple(
        str(row.get("candidate_id"))
        for row in sorted(materialized, key=lambda row: int(row.get("execution_position", -1)))
    )
    if observed_ids != expected_order or len(set(observed_ids)) != len(observed_ids):
        raise RuntimeError("runtime checkpoint method order mismatch")
    expected_call_seed = identity.get("call_seed")
    schedule_rows = identity.get("schedule")
    expected_schedule_seed = None
    if isinstance(schedule_rows, Sequence) and schedule_rows:
        schedule_seeds = {
            row.get("schedule_seed")
            for row in schedule_rows
            if isinstance(row, Mapping)
        }
        if len(schedule_seeds) == 1:
            expected_schedule_seed = next(iter(schedule_seeds))
    for row in materialized:
        if (
            str(row.get("backbone")) != str(identity.get("model"))
            or str(row.get("arm")) != str(identity.get("arm"))
            or int(row.get("budget", -1)) != int(identity.get("budget", -2))
            or int(row.get("repeat", -1)) != int(identity.get("repeat", -2))
        ):
            raise RuntimeError("runtime checkpoint row identity mismatch")
        if tuple(str(value) for value in row.get("execution_order", ())) != expected_order:
            raise RuntimeError("runtime checkpoint execution order mismatch")
        if row.get("warmup_excluded") is not True:
            raise RuntimeError("runtime checkpoint warmup contract mismatch")
        if row.get("status") != "ok":
            raise RuntimeError("runtime checkpoint contains a failed row")
        if expected_call_seed is not None and row.get("call_seed") != expected_call_seed:
            raise RuntimeError("runtime checkpoint call seed mismatch")
        if expected_schedule_seed is not None and row.get("schedule_seed") != expected_schedule_seed:
            raise RuntimeError("runtime checkpoint schedule seed mismatch")
        expected_dataset_identity = {
            "values_sha256": identity.get("values_sha256"),
            "labels_sha256": identity.get("labels_sha256"),
            "row_count": identity.get("row_count"),
            "feature_count": identity.get("feature_count"),
            "metadata": identity.get("dataset_metadata", {}),
        }
        if row.get("dataset_identity") != expected_dataset_identity:
            raise RuntimeError("runtime checkpoint dataset identity mismatch")
        for key in ("total_wall_seconds", "total_cpu_seconds", "peak_memory_mb"):
            value = row.get(key)
            if isinstance(value, bool) or value is None or not np.isfinite(float(value)) or float(value) < 0.0:
                raise RuntimeError(f"runtime checkpoint has invalid {key}")
        score = row.get("candidate_score")
        if (
            isinstance(score, bool)
            or score is None
            or not np.isfinite(float(score))
        ):
            raise RuntimeError("runtime checkpoint has invalid candidate_score")
        try:
            food101._validate_row_recipes(
                row,
                error_prefix=f"runtime checkpoint recipe for {row.get('candidate_id')}",
            )
        except RuntimeError as exc:
            raise RuntimeError("runtime checkpoint candidate recipe mismatch") from exc
    return materialized


def _runtime_repeat_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic runtime surface, excluding clocks/RSS."""

    fold_scores = []
    folds = row.get("folds", ())
    if isinstance(folds, Sequence) and not isinstance(folds, (str, bytes)):
        for fold in folds:
            if isinstance(fold, Mapping):
                fold_scores.append(
                    {
                        "fold": fold.get("fold"),
                        "score": fold.get("score"),
                        "candidate_config_sha256": fold.get("candidate_config_sha256"),
                    }
                )
    return {
        "candidate_id": row.get("candidate_id"),
        "dataset_identity": row.get("dataset_identity"),
        "candidate_score": row.get("candidate_score"),
        "prototype_refinement": row.get("prototype_refinement"),
        "conditioning_diagnostics": row.get("conditioning_diagnostics"),
        "fold_scores": fold_scores,
    }


def _validate_runtime_repeat_determinism(
    rows: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> None:
    """Require exact structural equality over the five timing repeats.

    Runtime repeats use the same subset, bridge bank, call seed, and method
    recipe.  Their clocks and conservative peak RSS are allowed to vary, but
    score and learned diagnostic surfaces must not.  This gate is independent
    of the counterbalanced execution order.
    """

    groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("candidate_id")),
            str(row.get("backbone")),
            str(row.get("arm")),
            int(row.get("budget")),
        )
        groups.setdefault(key, []).append(row)
    expected_repeats = set(int(value) for value in RUNTIME_REPEATS)
    expected_methods = set(str(value) for value in methods)
    for key, values in groups.items():
        if key[0] not in expected_methods:
            raise RuntimeError(f"runtime repeat determinism has unknown method {key[0]!r}")
        observed_repeats = {int(row.get("repeat", -1)) for row in values}
        if observed_repeats != expected_repeats:
            raise RuntimeError(
                f"runtime repeat determinism has incomplete repeats for {key!r}"
            )
        signatures = {
            canonical_json(_runtime_repeat_signature(row))
            for row in values
        }
        if len(signatures) != 1:
            raise RuntimeError(f"runtime repeat determinism mismatch for {key!r}")


def _validate_runtime_rows(rows: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> None:
    """Enforce the exact complete runtime grid before publishing completed."""

    expected_count = len(manifest.runtime_panels()) * len(methods)
    if len(rows) != expected_count:
        raise RuntimeError(f"runtime row count {len(rows)} != {expected_count}")
    expected_blocks = set(manifest.runtime_panels())
    observed_blocks = {
        (
            str(row.get("backbone")),
            str(row.get("arm")),
            int(row.get("budget")),
            int(row.get("repeat")),
        )
        for row in rows
    }
    if observed_blocks != expected_blocks:
        raise RuntimeError("runtime rows do not cover the exact model/budget/repeat grid")
    identities: set[tuple[Any, ...]] = set()
    dataset_identities: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        identity = (
            str(row.get("candidate_id")),
            str(row.get("backbone")),
            str(row.get("arm")),
            int(row.get("budget")),
            int(row.get("repeat")),
        )
        if identity in identities:
            raise RuntimeError(f"duplicate runtime row identity {identity!r}")
        identities.add(identity)
        if str(row.get("candidate_id")) not in methods:
            raise RuntimeError("runtime row contains an unknown method")
        if row.get("warmup_excluded") is not True or row.get("status") != "ok":
            raise RuntimeError("runtime row violates completion contract")
        dataset_identity = row.get("dataset_identity")
        if not isinstance(dataset_identity, Mapping):
            raise RuntimeError("runtime row lacks hash-bound dataset identity")
        dataset_block = (
            str(row.get("backbone")),
            str(row.get("arm")),
            int(row.get("budget")),
        )
        prior_dataset_identity = dataset_identities.get(dataset_block)
        if prior_dataset_identity is None:
            dataset_identities[dataset_block] = dict(dataset_identity)
        elif dict(dataset_identity) != dict(prior_dataset_identity):
            raise RuntimeError(
                "runtime rows use inconsistent dataset identity across methods/repeats"
            )
        for field in ("values_sha256", "labels_sha256"):
            digest = dataset_identity.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise RuntimeError(f"runtime dataset identity has malformed {field}")
        for key in ("total_wall_seconds", "total_cpu_seconds", "peak_memory_mb"):
            value = row.get(key)
            if isinstance(value, bool) or value is None or not np.isfinite(float(value)) or float(value) < 0.0:
                raise RuntimeError(f"runtime row has invalid {key}")
        score = row.get("candidate_score")
        if isinstance(score, bool) or score is None or not np.isfinite(float(score)):
            raise RuntimeError("runtime row has invalid candidate_score")
        try:
            food101._validate_row_recipes(
                row,
                error_prefix=f"runtime recipe for {row.get('candidate_id')}",
            )
        except RuntimeError as exc:
            raise RuntimeError("runtime row candidate recipe mismatch") from exc
    expected_per_block = set(methods)
    for block in expected_blocks:
        observed = {
            str(row.get("candidate_id"))
            for row in rows
            if (
                str(row.get("backbone")),
                str(row.get("arm")),
                int(row.get("budget")),
                int(row.get("repeat")),
            ) == block
        }
        if observed != expected_per_block:
            raise RuntimeError(f"runtime block {block!r} has an incomplete method set")
    _validate_runtime_repeat_determinism(rows, methods)


def _classify_runtime_stop(exc: BaseException) -> str:
    message = str(exc).lower()
    if "hash" in message or "identity" in message or "protocol" in message or "source" in message:
        return "provenance_failure"
    if "checkpoint" in message or "resume" in message:
        return "resume_failure"
    if "nondetermin" in message:
        return "nondeterminism_failure"
    return "runtime_candidate_failure"


def _write_runtime_bundle(
    output: Path,
    *,
    artifact_status: str,
    runtime_identity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    stop_reason: str | None = None,
    stop_error: str | None = None,
) -> dict[str, Any]:
    """Atomically write the canonical runtime JSONL/raw/manifest surfaces."""

    materialized = [dict(row) for row in rows]
    runtime_jsonl = _canonical_jsonl(materialized)
    tables = {"runtime_rows": _table_manifest(materialized, runtime_jsonl)}
    recipe_seed_rule = runtime_identity.get("deterministic_seed_rule")
    if not isinstance(recipe_seed_rule, Mapping):
        raise RuntimeError("runtime identity lacks deterministic seed rule")
    recipe_bindings = food101._recipe_bindings(
        materialized,
        seed_rule=recipe_seed_rule,
    )
    if artifact_status == "completed":
        food101._validate_recipe_bindings(
            recipe_bindings,
            materialized,
            (),
            expected_ids=tuple(methods),
            seed_rule=recipe_seed_rule,
            error_prefix="runtime recipe bindings",
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": str(artifact_status),
        "study": "m50_backbone_ranking_runtime",
        "identity": dict(runtime_identity),
        "environment": dict(runtime_identity.get("environment", {}))
        if isinstance(runtime_identity.get("environment"), Mapping)
        else {},
        "repository_provenance": dict(
            runtime_identity.get("repository_provenance", {})
        )
        if isinstance(runtime_identity.get("repository_provenance"), Mapping)
        else {},
        "source_evidence": dict(runtime_identity.get("source_evidence", {}))
        if isinstance(runtime_identity.get("source_evidence"), Mapping)
        else {},
        "grid": {
            "models": list(manifest.MODELS),
            "arms": list(manifest.RUNTIME_ARMS),
            "budgets": list(RUNTIME_BUDGETS),
            "repeats": list(RUNTIME_REPEATS),
            "methods": list(methods),
        },
        "tables": tables,
        "recipe_bindings": recipe_bindings,
        "runtime_rows": materialized,
        "warmup_excluded": True,
        "fresh_process_peak_memory": True,
        "primary_clock": "outer_complete_method_call",
        "derived_chain_clock": "component_sum_only",
    }
    if stop_reason is not None:
        payload["stop_reason"] = str(stop_reason)
        payload["stop_error"] = str(stop_error) if stop_error is not None else None
    _atomic_write_bytes(output / "runtime_rows.jsonl", runtime_jsonl)
    _atomic_write_json(output / "raw_results.json", payload)
    raw_identity = {
        "sha256": manifest.sha256_path(output / "raw_results.json"),
        "size_bytes": int((output / "raw_results.json").stat().st_size),
    }
    manifest_payload = {
        key: payload[key]
        for key in (
            "schema_version",
            "artifact_status",
            "study",
            "identity",
            "environment",
            "repository_provenance",
            "source_evidence",
            "grid",
            "tables",
            "recipe_bindings",
            "warmup_excluded",
            "fresh_process_peak_memory",
            "primary_clock",
            "derived_chain_clock",
        )
    }
    manifest_payload["raw_results"] = raw_identity
    if stop_reason is not None:
        manifest_payload.update({"stop_reason": payload["stop_reason"], "stop_error": payload["stop_error"]})
    _atomic_write_json(output / "manifest.json", manifest_payload)
    return payload


def run_runtime(
    *,
    output: Path,
    development_decision: Mapping[str, Any] | os.PathLike[str] | str,
    protocol_hash: str,
    code_identity_hash: str,
    dataset_factory: Callable[[str, str, int], tuple[np.ndarray, np.ndarray]],
    methods: Sequence[str] | None = None,
    use_fresh_process: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the exact three-arm runtime grid against a provisional dev decision.

    ``dataset_factory`` must return one deterministic pre-normalization
    dataset for ``(model, arm, budget)`` and expose the verified Food runtime
    metadata created by :func:`food_dataset_factory`.  OI performs the
    archived one-pass row-L2 operation inside the measured outer method call,
    while LP's frozen pipeline performs its own one-pass normalizer.
    """

    verified_protocol_hash = manifest.verify_protocol_hash()
    verified_code_identity_hash = manifest.code_identity_sha256(require_existing=True)
    if str(protocol_hash) != verified_protocol_hash:
        raise PermissionError("runtime protocol hash does not match current verified protocol")
    if str(code_identity_hash) != verified_code_identity_hash:
        raise PermissionError("runtime code identity does not match current verified sources")
    protocol_hash = verified_protocol_hash
    code_identity_hash = verified_code_identity_hash
    # Runtime bridge construction happens in the parent process as well as in
    # fresh workers.  Refuse an uncontrolled parent before creating output or
    # touching the dataset factory; the CLI exports these values before import.
    food101._require_thread_environment()
    # Collect these only after the explicit hash checks and before the first
    # factory call.  The values are persisted in the identity carried by each
    # checkpoint and in the final raw/manifest surfaces, so a resumed run
    # cannot silently change its source or environment contract.
    # Preflight provenance before constructing the Food factory.  The runtime
    # stage repeats this check when it builds the immutable identity because
    # the programmatic entry point must be equally fail-closed.
    provenance = manifest.repository_provenance(require_sources=True)
    source_evidence = manifest.verify_protocol_source_evidence()
    runtime_environment = manifest.environment(provenance)
    runtime_environment["thread_environment"] = {
        name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
    }
    lock_payload, decision_identity = manifest.provisional_runtime_artifact(
        development_decision,
        protocol_hash=protocol_hash,
        code_identity_hash=code_identity_hash,
    )
    locked_methods = lock_payload.get("runtime_candidate_ids")
    if not isinstance(locked_methods, Sequence) or isinstance(locked_methods, (str, bytes)):
        raise PermissionError("provisional runtime decision has no runtime_candidate_ids")
    method_tuple = _validate_methods(tuple(str(value) for value in locked_methods))
    if method_tuple != RUNTIME_PRIMITIVE_METHODS:
        raise PermissionError(
            "runtime must execute the exact frozen primitive method set "
            f"{RUNTIME_PRIMITIVE_METHODS!r}"
        )
    if methods is not None and tuple(str(value) for value in methods) != method_tuple:
        raise PermissionError(
            "runtime methods must exactly match the provisional decision; "
            "do not benchmark an outcome-selected subset"
        )
    decision_hash = str(decision_identity["provisional_decision_sha256"])
    analysis_manifest_hash = str(
        decision_identity["provisional_analysis_manifest_sha256"]
    )
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    existing_manifest = output_path / "manifest.json"
    existing_status: str | None = None
    if existing_manifest.exists():
        try:
            existing_status = json.loads(existing_manifest.read_text(encoding="utf-8")).get("artifact_status")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("terminal runtime manifest is malformed; refuse overwrite/resume") from exc
        if existing_status in {"completed", "stopped"}:
            raise RuntimeError("terminal runtime artifact exists; refuse overwrite/resume")
        if existing_status != "running":
            raise RuntimeError("runtime manifest has an unknown artifact status")
        if not resume:
            raise RuntimeError("runtime artifact is running; pass --resume to continue")
    if output_path.exists() and not existing_manifest.exists():
        contents = tuple(output_path.iterdir())
        if contents:
            raise RuntimeError(
                "runtime output directory is non-empty without a manifest; "
                "refuse overwrite/resume"
            )
    runtime_identity = _runtime_identity(
        code_identity_sha256=code_identity_hash,
        protocol_sha256=protocol_hash,
        provisional_decision_sha256=decision_hash,
        provisional_analysis_manifest_sha256=analysis_manifest_hash,
        methods=method_tuple,
        arms=manifest.RUNTIME_ARMS,
        provenance=provenance,
        environment=runtime_environment,
        source_evidence=source_evidence,
    )
    if existing_status == "running":
        try:
            running_payload = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("running runtime manifest is malformed") from exc
        if running_payload.get("identity") != runtime_identity:
            raise RuntimeError("running runtime manifest identity mismatch")
    else:
        _atomic_write_json(
            existing_manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_status": "running",
                "study": "m50_backbone_ranking_runtime",
                "identity": runtime_identity,
                "environment": dict(runtime_identity.get("environment", {})),
                "repository_provenance": dict(
                    runtime_identity.get("repository_provenance", {})
                ),
                "source_evidence": dict(runtime_identity.get("source_evidence", {})),
                "grid": {
                    "models": list(manifest.MODELS),
                    "arms": list(manifest.RUNTIME_ARMS),
                    "budgets": list(RUNTIME_BUDGETS),
                    "repeats": list(RUNTIME_REPEATS),
                    "methods": list(method_tuple),
                },
            },
        )
    rows: list[dict[str, Any]] = []
    checkpoints = output_path / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    schedule = manifest.runtime_execution_order(method_tuple)
    schedule_by_block: dict[tuple[str, str, int, int], tuple[Mapping[str, Any], ...]] = {}
    for model, arm, budget, repeat in manifest.runtime_panels():
        block_rows = tuple(
            row
            for row in schedule
            if row["model"] == model
            and row["arm"] == arm
            and int(row["budget"]) == int(budget)
            and int(row["repeat"]) == int(repeat)
        )
        ordered = tuple(
            sorted(block_rows, key=lambda row: int(row["execution_position"]))
        )
        scheduled_ids = tuple(str(row["method_id"]) for row in ordered)
        if len(scheduled_ids) != len(method_tuple) or set(scheduled_ids) != set(method_tuple):
            raise RuntimeError(
                f"runtime schedule method set mismatch for {(model, arm, budget, repeat)!r}"
            )
        schedule_by_block[(model, arm, int(budget), int(repeat))] = ordered

    try:
        for model, arm, budget, repeat in manifest.runtime_panels():
            block_key = (str(model), str(arm), int(budget), int(repeat))
            values, labels = dataset_factory(model, arm, int(budget))
            values = np.asarray(values)
            labels = np.asarray(labels, dtype=object)
            if values.ndim != 2 or labels.ndim != 1 or values.shape[0] != labels.size:
                raise ValueError(f"runtime dataset factory returned misaligned data for {block_key!r}")
            expected_rows = RUNTIME_CLASSES * int(budget)
            if values.shape[0] != expected_rows:
                raise ValueError(
                    f"runtime dataset {block_key!r} must contain {expected_rows} rows; "
                    f"got {values.shape[0]}"
                )
            encoded, classes = food101._stratification_encoding(labels)
            counts = np.bincount(encoded, minlength=len(classes))
            if len(classes) != RUNTIME_CLASSES or any(
                int(count) != int(budget) for count in counts
            ):
                raise ValueError(
                    f"runtime dataset {block_key!r} is not "
                    f"{RUNTIME_CLASSES}-class balanced"
                )
            k_per_class = {
                label: min(food101.K, max(1, int(count) // 5), int(count))
                for label, count in zip(classes, counts)
            }
            factory_metadata = getattr(dataset_factory, "_m50_runtime_metadata", {})
            if not isinstance(factory_metadata, Mapping):
                raise RuntimeError(
                    "runtime dataset factory lacks verified Food provenance metadata"
                )
            dataset_metadata: dict[str, Any] = {}
            if isinstance(factory_metadata, Mapping):
                model_metadata = factory_metadata.get("models", {})
                if isinstance(model_metadata, Mapping):
                    raw_model_metadata = model_metadata.get(model, {})
                    if isinstance(raw_model_metadata, Mapping):
                        dataset_metadata.update(dict(raw_model_metadata))
            required_metadata = (
                "cache_matrix_sha256",
                "max_panel_values_sha256",
                "max_panel_labels_sha256",
                "max_panel_index_sha256",
                "bridge_bank_seed",
                "max_panel_budget",
                "nested_index_seed",
                "nested_indices_sha256",
            )
            missing_metadata = tuple(
                key
                for key in required_metadata
                if key not in dataset_metadata
            )
            if missing_metadata:
                raise RuntimeError(
                    "runtime dataset factory metadata is incomplete: "
                    f"missing={missing_metadata!r}"
                )
            for hash_key in (
                "cache_matrix_sha256",
                "max_panel_values_sha256",
                "max_panel_labels_sha256",
                "max_panel_index_sha256",
                "nested_indices_sha256",
            ):
                digest = dataset_metadata.get(hash_key)
                if not isinstance(digest, str) or len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest
                ):
                    raise RuntimeError(
                        f"runtime dataset factory metadata has malformed {hash_key}"
                    )
            arm_metadata = {
                name: (float(lam), float(nu))
                for name, lam, nu in food101.ARMS
            }
            if arm not in arm_metadata:
                raise ValueError(f"runtime bridge arm lacks frozen parameters: {arm!r}")
            dataset_metadata.update(
                {
                    "bridge_arm": str(arm),
                    "bridge_lambda": arm_metadata[arm][0],
                    "bridge_nuisance_strength": arm_metadata[arm][1],
                }
            )
            block_schedule = schedule_by_block[block_key]
            identity = _runtime_block_identity(
                runtime_identity=runtime_identity,
                model=model,
                arm=arm,
                budget=int(budget),
                repeat=int(repeat),
                schedule=block_schedule,
                values=values,
                labels=labels,
                dataset_metadata=dataset_metadata,
            )
            checkpoint = checkpoints / (
                f"{model}__arm-{arm}__budget-{int(budget)}__repeat-{int(repeat)}.json"
            )
            if checkpoint.exists():
                if not resume:
                    raise RuntimeError(
                        f"runtime checkpoint exists; pass --resume to reuse {checkpoint}"
                    )
                try:
                    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"runtime checkpoint malformed: {checkpoint}") from exc
                rows.extend(
                    _validate_runtime_block(
                        checkpoint_payload,
                        identity=identity,
                        methods=method_tuple,
                    )
                )
                continue

            block_results: list[dict[str, Any]] = []
            with (
                tempfile.TemporaryDirectory(prefix="m50-runtime-block-")
                if use_fresh_process
                else nullcontext()
            ) as temp_dir:
                block_data_path: Path | None = None
                if use_fresh_process:
                    block_data_path = Path(str(temp_dir)) / "data.npz"
                    np.savez(
                        block_data_path,
                        values=np.asarray(values, dtype=np.float32),
                        labels=np.asarray(labels, dtype=object),
                    )
                for scheduled in block_schedule:
                    method = str(scheduled["method_id"])
                    call_seed = int(food101.SEED + int(budget))
                    if use_fresh_process:
                        result = _fresh_process_measure(
                            values,
                            labels,
                            method_id=method,
                            seed=call_seed,
                            k_per_class=k_per_class,
                            data_path=block_data_path,
                        )
                    else:
                        result = measure_runtime(
                            values,
                            labels,
                            method_id=method,
                            seed=call_seed,
                            k_per_class=k_per_class,
                        )
                    block_results.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "stage": "runtime",
                            "candidate_id": method,
                            "backbone": str(model),
                            "arm": str(arm),
                            "budget": int(budget),
                            "repeat": int(repeat),
                            "execution_position": int(scheduled["execution_position"]),
                            "execution_order": [str(item["method_id"]) for item in block_schedule],
                            "schedule_seed": manifest.SCHEDULE_SEED,
                            "call_seed": call_seed,
                            "warmup_excluded": True,
                            "fresh_process": bool(use_fresh_process),
                            "dataset_identity": {
                                "values_sha256": identity["values_sha256"],
                                "labels_sha256": identity["labels_sha256"],
                                "row_count": identity["row_count"],
                                "feature_count": identity["feature_count"],
                                "metadata": identity.get("dataset_metadata", {}),
                            },
                            "status": "ok",
                            **result,
                        }
                    )
            _validate_runtime_block(
                {"artifact_status": "completed", "identity": identity, "rows": block_results},
                identity=identity,
                methods=method_tuple,
            )
            _atomic_write_json(
                checkpoint,
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_status": "completed",
                    "identity": identity,
                    "rows": block_results,
                },
            )
            rows.extend(block_results)
    except Exception as exc:
        stop_reason = _classify_runtime_stop(exc)
        _write_runtime_bundle(
            output_path,
            artifact_status="stopped",
            runtime_identity=runtime_identity,
            rows=rows,
            methods=method_tuple,
            stop_reason=stop_reason,
            stop_error=str(exc),
        )
        raise

    _validate_runtime_rows(rows, method_tuple)
    return _write_runtime_bundle(
        output_path,
        artifact_status="completed",
        runtime_identity=runtime_identity,
        rows=rows,
        methods=method_tuple,
    )


def food_dataset_factory(
    cache_dir: Path,
) -> Callable[[str, str, int], tuple[np.ndarray, np.ndarray]]:
    """Build the exact pre-normalization Food runtime subsets.

    Runtime uses the frozen 26,400-row training cohort (40 classes × 660
    rows), not any selector role from the development cohort.  One nested
    permutation is built at seed 42 and shared across all models and timing
    repeats.  The returned subsets remain raw/pre-normalization so the
    measured OI call performs its one panel-wide row-L2 pass and the measured
    LP call performs its frozen pipeline normalization.
    """

    bridge_driver, _result, cohort, _prior, _hashes = food101._load_archived_inputs(
        food101.DEFAULT_DRIVER,
        food101.DEFAULT_RESULT,
        food101.DEFAULT_COHORT,
        food101.DEFAULT_PRIOR,
        food101.DEFAULT_RUNTIME_SOURCE,
    )
    del _result, _prior, _hashes
    runtime_source = Path(food101.DEFAULT_RUNTIME_SOURCE)
    if manifest.sha256_path(runtime_source) != food101.RUNTIME_SOURCE_SHA256:
        raise ValueError("archived runtime driver hash mismatch")
    spec = importlib.util.spec_from_file_location(
        "_m50_frozen_food101_runtime_driver", runtime_source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load runtime driver at {runtime_source}")
    runtime_driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime_driver
    spec.loader.exec_module(runtime_driver)
    sample_ids = [str(value) for value in cohort["extracted_sample_ids"]]
    train_rows = int(cohort.get("extracted_train_rows", 0))
    if train_rows != RUNTIME_COHORT_ROWS:
        raise ValueError(
            f"runtime cohort must contain {RUNTIME_COHORT_ROWS:,} train rows; got {train_rows}"
        )
    labels = np.asarray(
        [value.split("/")[2] for value in sample_ids[:train_rows]], dtype=object
    )
    encoded, classes = food101._stratification_encoding(labels)
    counts = np.bincount(encoded, minlength=len(classes))
    if len(classes) != RUNTIME_CLASSES or any(
        int(count) != RUNTIME_ROWS_PER_CLASS for count in counts
    ):
        raise ValueError(
            "runtime cohort must contain "
            f"{RUNTIME_CLASSES} classes with {RUNTIME_ROWS_PER_CLASS} train rows each"
        )
    build_nested_indices = getattr(runtime_driver, "build_nested_indices", None)
    if not callable(build_nested_indices):
        raise AttributeError("archived runtime driver has no build_nested_indices")
    nested_indices = build_nested_indices(
        labels,
        RUNTIME_BUDGETS,
        seed=RUNTIME_NESTED_INDEX_SEED,
    )
    expected_index_sets = {
        int(budget): np.asarray(nested_indices[int(budget)], dtype=np.int64)
        for budget in RUNTIME_BUDGETS
    }
    if any(
        index.size != RUNTIME_CLASSES * int(budget)
        for budget, index in expected_index_sets.items()
    ):
        raise ValueError("runtime nested indices have an unexpected row count")
    sample_hash = food101._sample_ids_hash(sample_ids)
    max_budget = max(RUNTIME_BUDGETS)
    max_indices = expected_index_sets[max_budget]
    nested_identity = {
        str(budget): [int(value) for value in indices.tolist()]
        for budget, indices in sorted(expected_index_sets.items())
    }
    nested_indices_sha256 = hashlib.sha256(
        canonical_json(nested_identity).encode("utf-8")
    ).hexdigest()
    arm_lookup = {name: (float(lam), float(nu)) for name, lam, nu in food101.ARMS}
    if tuple(arm_lookup) != tuple(manifest.RUNTIME_ARMS):
        raise ValueError("runtime bridge arm order does not match the frozen manifest")
    max_positions = {
        int(global_index): int(position)
        for position, global_index in enumerate(max_indices.tolist())
    }
    # Keep at most one transformed max-panel matrix alive.  The runtime loop
    # is model/arm-major, so this cache is reused across that arm's budgets
    # and repeats and evicted before the next arm/model is loaded.
    cached_key: tuple[str, str] | None = None
    cached_matrix: np.ndarray | None = None
    cache_manifests: dict[str, dict[str, Any]] = {}
    model_metadata: dict[str, dict[str, Any]] = {}

    def factory(model: str, arm: str, budget: int) -> tuple[np.ndarray, np.ndarray]:
        if arm not in arm_lookup:
            raise ValueError(
                f"unknown runtime bridge arm {arm!r}; expected {manifest.RUNTIME_ARMS!r}"
            )
        if int(budget) not in expected_index_sets:
            raise ValueError(
                f"unknown runtime budget {budget!r}; expected {RUNTIME_BUDGETS!r}"
            )
        nonlocal cached_key, cached_matrix
        cache_key = (str(model), str(arm))
        if cached_key != cache_key:
            matrix, cache_manifest = food101._load_cache(Path(cache_dir), model, sample_hash)
            raw_max = np.asarray(matrix[max_indices], dtype=np.float32)
            max_labels = labels[max_indices]
            banks = bridge_driver._paired_split_banks(
                raw_max,
                max_labels,
                seed=RUNTIME_BRIDGE_BANK_SEED,
            )
            lam, nu = arm_lookup[arm]
            if arm == "baseline":
                # The baseline runtime arm is the unmodified cache panel.
                # Keeping this explicit makes its paired identity exact and
                # leaves all row-L2 preprocessing to the measured selector
                # call (OI) or the frozen probe pipeline (LP).
                cached_matrix = np.asarray(raw_max, dtype=np.float32)
            else:
                cached_matrix = np.asarray(
                    bridge_driver._bridge_transform(
                        raw_max,
                        donor=banks["donor"],
                        mode=banks["mode"],
                        nuisance=banks["nuisance"],
                        q=1.0,
                        lam=lam,
                        nu=nu,
                    ),
                    dtype=np.float32,
                )
            cached_key = cache_key
            cache_manifests[model] = dict(cache_manifest)
            model_metadata[model] = {
                "cache_matrix_sha256": str(cache_manifest.get("sha256")),
                "max_panel_values_sha256": hashlib.sha256(raw_max.tobytes()).hexdigest(),
                "max_panel_labels_sha256": hashlib.sha256(
                    canonical_json(max_labels.tolist()).encode("utf-8")
                ).hexdigest(),
                "max_panel_index_sha256": hashlib.sha256(
                    np.ascontiguousarray(max_indices).tobytes()
                ).hexdigest(),
                "bridge_bank_seed": int(RUNTIME_BRIDGE_BANK_SEED),
                "max_panel_budget": int(max_budget),
                "nested_index_seed": int(RUNTIME_NESTED_INDEX_SEED),
                "nested_indices_sha256": nested_indices_sha256,
            }
        if cached_matrix is None or cached_key != cache_key:
            raise RuntimeError("runtime transformed cache state is unavailable")
        matrix = cached_matrix
        if matrix.shape[0] != max_indices.size:
            raise ValueError(f"runtime transformed max panel has an unexpected row count for {model!r}")
        if matrix.ndim != 2 or not np.all(np.isfinite(np.asarray(matrix))):
            raise ValueError(f"runtime transformed max panel is not finite for {model!r}/{arm!r}")
        selected = expected_index_sets[int(budget)]
        try:
            positions = np.asarray([max_positions[int(value)] for value in selected], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("runtime nested budgets are not subsets of the max panel") from exc
        return np.asarray(matrix[positions], dtype=np.float32), labels[selected]

    # The runner copies this structural metadata into each block identity.  It
    # contains no outcomes or timing values and makes the max-panel/bank
    # recipe auditable after checkpoints are deleted.
    factory._m50_runtime_metadata = {
        "nested_indices_sha256": nested_indices_sha256,
        "nested_index_seed": int(RUNTIME_NESTED_INDEX_SEED),
        "max_panel_budget": int(max_budget),
        "bridge_bank_seed": int(RUNTIME_BRIDGE_BANK_SEED),
        "models": model_metadata,
        "cache_manifests": cache_manifests,
    }
    # A private structural seam used by focused tests and diagnostics.  It
    # deliberately exposes only the one retained cache key/size, never the
    # matrix itself, so tests can prove model/arm eviction without retaining
    # another reference to the large arrays.
    factory._m50_runtime_cache_snapshot = lambda: {
        "active_key": cached_key,
        "active_matrix_nbytes": (
            int(cached_matrix.nbytes) if cached_matrix is not None else 0
        ),
        "retained_model_metadata": tuple(sorted(model_metadata)),
        "retained_cache_manifest_metadata": tuple(sorted(cache_manifests)),
    }

    return factory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-decision", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=food101.DEFAULT_CACHE)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--code-identity-sha256")
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        return _worker_main()
    if args.output is None or args.development_decision is None:
        raise SystemExit(
            "runtime benchmark requires --output and --development-decision "
            "(or use --worker)"
        )
    # Keep the CLI preflight explicit; run_runtime repeats it when constructing
    # the immutable identity for direct/programmatic callers.
    manifest.repository_provenance(require_sources=True)
    current_protocol_hash = manifest.verify_protocol_hash()
    current_code_hash = str(
        manifest.code_identity_sha256(require_existing=True)
    )
    if args.protocol_sha256 is not None and args.protocol_sha256 != current_protocol_hash:
        raise SystemExit("--protocol-sha256 does not match the verified current protocol")
    if args.code_identity_sha256 is not None and args.code_identity_sha256 != current_code_hash:
        raise SystemExit("--code-identity-sha256 does not match verified current sources")
    protocol_hash = current_protocol_hash
    code_hash = current_code_hash
    run_runtime(
        output=args.output,
        development_decision=args.development_decision,
        protocol_hash=protocol_hash,
        code_identity_hash=code_hash,
        dataset_factory=food_dataset_factory(args.cache_dir),
        use_fresh_process=not bool(args.in_process),
        resume=bool(args.resume),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_METHODS", "OPTIONAL_METHODS", "RUNTIME_CLASSES", "RUNTIME_COHORT_ROWS",
    "RUNTIME_NESTED_INDEX_SEED", "RUNTIME_BRIDGE_BANK_SEED",
    "RUNTIME_ROWS_PER_CLASS", "RUNTIME_PRIMITIVE_METHODS",
    "RUNTIME_BUDGETS", "RUNTIME_REPEATS", "_fresh_process_measure",
    "_peak_memory_mb", "_worker_main", "food_dataset_factory", "measure_runtime",
    "run_runtime",
]
