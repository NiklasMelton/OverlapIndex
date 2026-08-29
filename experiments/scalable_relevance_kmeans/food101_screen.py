"""Resumable frozen Food-101 screen for scalable one-update relevance KMeans."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn

from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans.scalable import (
    ScalableOneUpdateRelevanceKMeans,
)


METHODS = ("A", "B", "S", "LP-FULL")
MODELS = tuple(food101.MODELS)
REPLICATES = tuple(food101.REPLICATES)
ARMS = tuple(name for name, _lam, _nu in food101.ARMS)
BUDGETS = tuple(food101.BUDGETS)
FOLDS = int(food101.FOLDS)
SEED = int(food101.SEED)
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / "food101_protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "food101_protocol.sha256"
THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}
LOCAL_SOURCE_PATHS = (
    PACKAGE_DIR / "scalable.py",
    PACKAGE_DIR / "food101_screen.py",
    PACKAGE_DIR / "food101_analysis.py",
    PROTOCOL_PATH,
    ROOT / "pyproject.toml",
) + tuple(sorted((ROOT / "overlapindex").glob("*.py"))) + tuple(
    sorted((ROOT / "experiments" / "m50_backbone_ranking").glob("*.py"))
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return value


def _validate_protocol() -> tuple[str, dict[str, Any]]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SIDECAR.is_file():
        raise RuntimeError("Food screen protocol and sidecar must exist before execution")
    observed = sha256_path(PROTOCOL_PATH)
    expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
    if observed != expected or len(expected) != 64:
        raise RuntimeError("Food screen protocol sidecar mismatch")
    payload = _read_json(PROTOCOL_PATH)
    if payload.get("status") != "frozen_before_outcomes":
        raise RuntimeError("Food screen protocol is not frozen before outcomes")
    grid = payload.get("grid", {})
    expected_grid = {
        "models": list(MODELS),
        "replicates": list(REPLICATES),
        "arms": list(ARMS),
        "budgets_per_class": list(BUDGETS),
        "folds": FOLDS,
        "k_per_class_max": int(food101.K),
        "panel_count": 600,
        "method_row_count": 2400,
        "reference_row_count": 600,
    }
    if grid != expected_grid:
        raise RuntimeError("Food screen protocol grid does not match implementation")
    if tuple(payload.get("methods", {})) != METHODS:
        raise RuntimeError("Food screen method order does not match implementation")
    return observed, payload


def _thread_environment() -> dict[str, str]:
    observed = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT}
    if any(observed[key] != value for key, value in THREAD_ENVIRONMENT.items()):
        raise RuntimeError(
            f"one-thread environment mismatch: expected {THREAD_ENVIRONMENT!r}, got {observed!r}"
        )
    return {key: str(value) for key, value in observed.items()}


def _source_identity() -> dict[str, Any]:
    missing = [str(path) for path in LOCAL_SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Food screen source files: {missing!r}")
    local_hashes = {str(path): sha256_path(path) for path in LOCAL_SOURCE_PATHS}
    archived = {
        "driver": (Path(food101.DEFAULT_DRIVER), food101.DRIVER_SHA256),
        "source_result": (Path(food101.DEFAULT_RESULT), food101.RESULT_SHA256),
        "source_cohort": (Path(food101.DEFAULT_COHORT), food101.COHORT_SHA256),
        "prior_replay": (Path(food101.DEFAULT_PRIOR), food101.PRIOR_SHA256),
    }
    archived_hashes: dict[str, str] = {}
    for name, (path, expected) in archived.items():
        observed = sha256_path(path)
        if observed != expected:
            raise RuntimeError(f"archived {name} hash mismatch")
        archived_hashes[name] = observed
    code_identity = hashlib.sha256(canonical_json(local_hashes).encode("utf-8")).hexdigest()
    return {
        "local_source_hashes": local_hashes,
        "code_identity_sha256": code_identity,
        "archived_source_hashes": archived_hashes,
    }


def _panel_id(model: str, replicate: int, arm: str, budget: int) -> str:
    return f"{model}__r{replicate}__{arm}__b{budget}"


def _method_order(panel_index: int) -> tuple[str, ...]:
    offset = int(panel_index) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def _k_per_class(labels: np.ndarray, train: np.ndarray) -> dict[Any, int]:
    target = np.asarray(labels)
    encoded, classes = food101._stratification_encoding(target)
    train_encoded = encoded[np.asarray(train, dtype=np.int64)]
    return {
        label: min(
            int(food101.K),
            max(1, int(np.count_nonzero(train_encoded == position)) // 5),
            int(np.count_nonzero(train_encoded == position)),
        )
        for position, label in enumerate(classes)
    }


def _scalable_config_identity(k_per_class: Mapping[Any, int], fold_seed: int) -> dict[str, Any]:
    return {
        "candidate_id": "S",
        "class": "ScalableOneUpdateRelevanceKMeans",
        "k_per_class": {str(key): int(value) for key, value in k_per_class.items()},
        "kmeans_kwargs": {"random_state": int(fold_seed)},
        "weight_floor": 0.05,
        "memory_budget_mb": 64,
        "row_cap": None,
        "prototype_refinement": False,
    }


def scalable_cross_fitted_score(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    raw_values = np.asarray(matrix, dtype=np.float32)
    target = np.asarray(labels)
    values = food101._row_l2(raw_values)
    folds = food101._stratified_folds(target, n_splits=FOLDS, seed=int(seed))
    scores: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        k = _k_per_class(target, train)
        identity = _scalable_config_identity(k, fold_seed)
        identity_sha = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        selector = ScalableOneUpdateRelevanceKMeans(
            k_per_class=k,
            kmeans_kwargs={"random_state": fold_seed},
            weight_floor=0.05,
            memory_budget_mb=64,
        )
        fit_wall, fit_cpu = time.perf_counter(), time.process_time()
        selector.fit(values[train], target[train])
        fit_wall_elapsed = time.perf_counter() - fit_wall
        fit_cpu_elapsed = time.process_time() - fit_cpu
        score_wall, score_cpu = time.perf_counter(), time.process_time()
        score, score_diagnostics = selector.score_fixed_with_diagnostics(
            values[holdout], target[holdout]
        )
        score_wall_elapsed = time.perf_counter() - score_wall
        score_cpu_elapsed = time.process_time() - score_cpu
        if not np.isfinite(score):
            raise RuntimeError("scalable candidate emitted a non-finite score")
        diagnostics = dict(selector.diagnostics_ or {})
        structural = {
            key: value
            for key, value in diagnostics.items()
            if not key.endswith("_wall_seconds")
        }
        stage_runtime = {
            key: value
            for key, value in diagnostics.items()
            if key.endswith("_wall_seconds")
        }
        scores.append(float(score))
        fold_rows.append(
            {
                "fold": int(fold),
                "seed": fold_seed,
                "train_size": int(train.size),
                "holdout_size": int(holdout.size),
                "score": float(score),
                "fit_wall_seconds": float(fit_wall_elapsed),
                "fit_cpu_seconds": float(fit_cpu_elapsed),
                "score_fixed_wall_seconds": float(score_wall_elapsed),
                "score_fixed_cpu_seconds": float(score_cpu_elapsed),
                "structural_diagnostics": structural,
                "stage_runtime": stage_runtime,
                "score_diagnostics": score_diagnostics,
                "candidate_config_identity": identity,
                "candidate_config_sha256": identity_sha,
            }
        )
    return {
        "candidate_id": "S",
        "candidate_name": "scalable_one_update_relevance_kmeans",
        "mode": "scalable_one_update",
        "prototype_refinement": False,
        "score": float(np.mean(scores)),
        "fit_wall_seconds": float(sum(row["fit_wall_seconds"] for row in fold_rows)),
        "fit_cpu_seconds": float(sum(row["fit_cpu_seconds"] for row in fold_rows)),
        "score_fixed_wall_seconds": float(
            sum(row["score_fixed_wall_seconds"] for row in fold_rows)
        ),
        "score_fixed_cpu_seconds": float(
            sum(row["score_fixed_cpu_seconds"] for row in fold_rows)
        ),
        "folds": fold_rows,
    }


def _execute_method(method: str, values: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, Any]:
    if method == "S":
        return scalable_cross_fitted_score(values, labels, seed=seed)
    if method not in {"A", "B", "LP-FULL"}:
        raise ValueError(f"unknown Food screen method {method!r}")
    return food101._cross_fitted_score(values, labels, candidate=method, seed=seed)


def _signature(result: Mapping[str, Any]) -> dict[str, Any]:
    folds = []
    for row in result.get("folds", ()):
        if not isinstance(row, Mapping):
            continue
        folds.append(
            {
                key: row.get(key)
                for key in (
                    "fold",
                    "seed",
                    "split_seed",
                    "train_size",
                    "holdout_size",
                    "score",
                    "refinement",
                    "conditioning",
                    "structural_diagnostics",
                    "score_diagnostics",
                    "candidate_config_sha256",
                )
                if key in row
            }
        )
    return {
        "candidate_id": result.get("candidate_id"),
        "candidate_name": result.get("candidate_name"),
        "mode": result.get("mode"),
        "prototype_refinement": result.get("prototype_refinement"),
        "score": result.get("score"),
        "refinement": result.get("refinement"),
        "conditioning": result.get("conditioning"),
        "folds": folds,
    }


def _determinism(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    first_sha = hashlib.sha256(canonical_json(_signature(first)).encode("utf-8")).hexdigest()
    second_sha = hashlib.sha256(canonical_json(_signature(second)).encode("utf-8")).hexdigest()
    return {
        "first_signature_sha256": first_sha,
        "second_signature_sha256": second_sha,
        "exact": first_sha == second_sha,
        "runtime_fields_excluded": True,
    }


def _row(
    result: Mapping[str, Any],
    *,
    model: str,
    replicate: int,
    arm: str,
    budget: int,
    position: int,
    order: Sequence[str],
    total_wall: float,
    total_cpu: float,
    stage: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "model": model,
        "backbone": model,
        "replicate": int(replicate),
        "arm": arm,
        "budget": int(budget),
        "candidate_id": str(result["candidate_id"]),
        "candidate_name": result.get("candidate_name"),
        "mode": result.get("mode"),
        "prototype_refinement_enabled": bool(result.get("prototype_refinement", False)),
        "score": float(result["score"]),
        "fit_wall_seconds": float(result.get("fit_wall_seconds", 0.0)),
        "fit_cpu_seconds": float(result.get("fit_cpu_seconds", 0.0)),
        "score_fixed_wall_seconds": float(result.get("score_fixed_wall_seconds", 0.0)),
        "score_fixed_cpu_seconds": float(result.get("score_fixed_cpu_seconds", 0.0)),
        "total_wall_seconds": float(total_wall),
        "total_cpu_seconds": float(total_cpu),
        "refinement": result.get("refinement", {}),
        "conditioning": result.get("conditioning", {}),
        "folds": result.get("folds", []),
        "execution_position": int(position),
        "execution_order": list(order),
        "warmup_excluded": True,
        "status": "ok",
        "error": None,
    }


def _run_identity(protocol_sha: str, sources: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    grid = {
        "models": [MODELS[0]] if smoke else list(MODELS),
        "replicates": [REPLICATES[0]] if smoke else list(REPLICATES),
        "arms": [ARMS[0]] if smoke else list(ARMS),
        "budgets": [BUDGETS[0]] if smoke else list(BUDGETS),
        "methods": list(METHODS),
    }
    return {
        "study": "scalable_one_update_food101_screen",
        "stage": "smoke" if smoke else "development",
        "protocol_sha256": protocol_sha,
        "sources": dict(sources),
        "grid": grid,
        "thread_environment": _thread_environment(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }


def run_screen(output: Path, *, smoke: bool = False, resume: bool = False) -> dict[str, Any]:
    protocol_sha, _protocol = _validate_protocol()
    sources = _source_identity()
    identity = _run_identity(protocol_sha, sources, smoke)
    identity_sha = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.is_file():
            raise RuntimeError("refusing non-empty Food screen output without identity-locked resume")
        existing = _read_json(manifest_path)
        if existing.get("artifact_status") != "running" or existing.get("run_identity") != identity:
            raise RuntimeError("Food screen resume manifest identity mismatch")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "artifact_status": "running",
                "run_identity": identity,
                "run_identity_sha256": identity_sha,
            },
        )
    driver, source_result, cohort, prior, archived_hashes = food101._load_archived_inputs(
        food101.DEFAULT_DRIVER,
        food101.DEFAULT_RESULT,
        food101.DEFAULT_COHORT,
        food101.DEFAULT_PRIOR,
    )
    if archived_hashes != sources["archived_source_hashes"]:
        raise RuntimeError("loaded archived source hashes changed after preflight")
    sample_ids = [str(value) for value in cohort["extracted_sample_ids"]]
    sample_hash = food101._sample_ids_hash(sample_ids)
    food101._validate_all_cache_manifests(food101.DEFAULT_CACHE, sample_hash)
    labels = np.asarray([value.split("/")[2] for value in sample_ids], dtype=object)
    roles = {int(key): value for key, value in cohort["roles"].items()}
    models = (MODELS[0],) if smoke else MODELS
    replicates = (REPLICATES[0],) if smoke else REPLICATES
    arms = (ARMS[0],) if smoke else ARMS
    budgets = (BUDGETS[0],) if smoke else BUDGETS
    panels = [
        (model, replicate, arm, budget)
        for model in models
        for replicate in replicates
        for arm in arms
        for budget in budgets
    ]
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    expected_names = {_panel_id(*panel) + ".json" for panel in panels}
    unexpected = {path.name for path in checkpoint_dir.iterdir()} - expected_names
    if unexpected:
        raise RuntimeError(f"unexpected Food screen checkpoints: {sorted(unexpected)!r}")
    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    deterministic: dict[str, dict[str, Any]] = {}
    arm_lookup = {name: (lam, nu) for name, lam, nu in food101.ARMS}
    panel_index = 0
    for model in models:
        matrix, cache_manifest = food101._load_cache(food101.DEFAULT_CACHE, model, sample_hash)
        for replicate in replicates:
            selector_indices = np.asarray(roles[int(replicate)]["selector"], dtype=np.int64)
            raw = np.asarray(matrix[selector_indices], dtype=np.float32)
            target = labels[selector_indices]
            banks = driver._paired_split_banks(raw, target, seed=SEED + int(replicate) + 11)
            nested = driver._nested_stratified_indices(
                target, budgets, seed=SEED + int(replicate) + 23
            )
            for arm in arms:
                lam, nu = arm_lookup[arm]
                transformed = driver._bridge_transform(
                    raw,
                    donor=banks["donor"],
                    mode=banks["mode"],
                    nuisance=banks["nuisance"],
                    q=1.0,
                    lam=lam,
                    nu=nu,
                )
                for budget in budgets:
                    selected = np.asarray(nested[int(budget)], dtype=np.int64)
                    subset = np.asarray(transformed[selected], dtype=np.float32)
                    subset_target = target[selected]
                    order = _method_order(panel_index)
                    panel_id = _panel_id(model, int(replicate), arm, int(budget))
                    panel_identity = {
                        "run_identity_sha256": identity_sha,
                        "panel_id": panel_id,
                        "model": model,
                        "replicate": int(replicate),
                        "arm": arm,
                        "budget": int(budget),
                        "method_order": list(order),
                        "cache_sha256": str(cache_manifest["sha256"]),
                        "values_sha256": hashlib.sha256(subset.tobytes()).hexdigest(),
                        "labels_sha256": hashlib.sha256(
                            canonical_json(subset_target.tolist()).encode("utf-8")
                        ).hexdigest(),
                    }
                    checkpoint = checkpoint_dir / f"{panel_id}.json"
                    if resume and checkpoint.is_file():
                        cached = _read_json(checkpoint)
                        if cached.get("artifact_status") != "completed" or cached.get("identity") != panel_identity:
                            raise RuntimeError(f"checkpoint identity/status mismatch for {panel_id}")
                        cached_rows = cached.get("rows")
                        cached_parity = cached.get("parity_rows")
                        if not isinstance(cached_rows, list) or len(cached_rows) != len(METHODS):
                            raise RuntimeError(f"checkpoint method block is incomplete for {panel_id}")
                        if not isinstance(cached_parity, list) or len(cached_parity) != 3:
                            raise RuntimeError(f"checkpoint parity block is incomplete for {panel_id}")
                        rows.extend(dict(row) for row in cached_rows)
                        parity_rows.extend(dict(row) for row in cached_parity)
                        for method, record in cached.get("determinism", {}).items():
                            if method in deterministic and deterministic[method] != record:
                                raise RuntimeError("determinism record changed across checkpoints")
                            deterministic[method] = dict(record)
                        panel_index += 1
                        continue
                    block: list[dict[str, Any]] = []
                    local_determinism: dict[str, dict[str, Any]] = {}
                    panel_seed = SEED + int(replicate)
                    for position, method in enumerate(order):
                        warmup = None
                        if method not in deterministic:
                            warmup = _execute_method(method, subset, subset_target, panel_seed)
                        outer_wall, outer_cpu = time.perf_counter(), time.process_time()
                        result = _execute_method(method, subset, subset_target, panel_seed)
                        total_wall = time.perf_counter() - outer_wall
                        total_cpu = time.process_time() - outer_cpu
                        if warmup is not None:
                            record = _determinism(warmup, result)
                            record["panel_id"] = panel_id
                            record["candidate_id"] = method
                            if record["exact"] is not True:
                                raise RuntimeError(f"nondeterministic method {method}")
                            deterministic[method] = record
                            local_determinism[method] = record
                        block.append(
                            _row(
                                result,
                                model=model,
                                replicate=int(replicate),
                                arm=arm,
                                budget=int(budget),
                                position=position,
                                order=order,
                                total_wall=total_wall,
                                total_cpu=total_cpu,
                                stage="smoke" if smoke else "development",
                            )
                        )
                    panel_parity = food101._baseline_parity_rows(block, prior)
                    panel_parity.extend(food101._probe_parity_rows(block, prior))
                    if len(panel_parity) != 3 or any(row.get("exact") is not True for row in panel_parity):
                        _atomic_json(
                            checkpoint,
                            {
                                "artifact_status": "stopped_parity_failure",
                                "identity": panel_identity,
                                "rows": block,
                                "parity_rows": panel_parity,
                                "determinism": local_determinism,
                            },
                        )
                        raise RuntimeError(f"archived A/B/LP parity failure for {panel_id}")
                    _atomic_json(
                        checkpoint,
                        {
                            "artifact_status": "completed",
                            "identity": panel_identity,
                            "rows": block,
                            "parity_rows": panel_parity,
                            "determinism": local_determinism,
                        },
                    )
                    rows.extend(block)
                    parity_rows.extend(panel_parity)
                    panel_index += 1
    expected_rows = len(panels) * len(METHODS)
    if len(rows) != expected_rows or len(parity_rows) != len(panels) * 3:
        raise RuntimeError("Food screen completed with an incomplete paired grid")
    if set(deterministic) != set(METHODS) or any(
        record.get("exact") is not True for record in deterministic.values()
    ):
        raise RuntimeError("Food screen determinism evidence is incomplete")
    reference_rows = [
        dict(row)
        for row in source_result.get("reference_rows", ())
        if isinstance(row, Mapping)
        and str(row.get("backbone")) in models
        and int(row.get("replicate", -1)) in replicates
        and str(row.get("arm")) in arms
    ]
    expected_reference = len(models) * len(replicates) * len(arms) * 4
    if len(reference_rows) != expected_reference:
        raise RuntimeError("Food reference grid is incomplete")
    payload = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": identity["study"],
        "stage": identity["stage"],
        "retrospective_development_only": True,
        "run_identity": identity,
        "run_identity_sha256": identity_sha,
        "configuration": identity["grid"],
        "determinism": deterministic,
        "parity_rows": parity_rows,
        "reference_rows": reference_rows,
        "selector_rows": rows,
    }
    raw_path = output / "raw_results.json"
    _atomic_json(raw_path, payload)
    terminal = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": identity["study"],
        "stage": identity["stage"],
        "run_identity": identity,
        "run_identity_sha256": identity_sha,
        "raw_results_sha256": sha256_path(raw_path),
        "counts": {
            "panels": len(panels),
            "selector_rows": len(rows),
            "parity_rows": len(parity_rows),
            "reference_rows": len(reference_rows),
            "checkpoints": len(list(checkpoint_dir.glob("*.json"))),
        },
    }
    _atomic_json(manifest_path, terminal)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    payload = run_screen(args.output, smoke=args.smoke, resume=args.resume)
    print(canonical_json({"artifact_status": payload["artifact_status"], "stage": payload["stage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
