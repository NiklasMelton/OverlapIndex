"""Versioned speed-focused Food-101 experiment for scalable relevance K-means."""

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
from experiments.scalable_relevance_kmeans.scalable_v2 import (
    FastScalableRelevanceKMeans,
)


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / "speed_protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "speed_protocol.sha256"
V1_ARTIFACT_DIR = ROOT / "artifacts" / "scalable_relevance_kmeans" / "food101_screen_v1"
THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}
MODELS = tuple(food101.MODELS)
REPLICATES = tuple(food101.REPLICATES)
ARMS = tuple(name for name, _lam, _nu in food101.ARMS)
BUDGETS = tuple(food101.BUDGETS)
FOLDS = int(food101.FOLDS)
SEED = int(food101.SEED)
DEVELOPMENT_REPLICATE = int(REPLICATES[0])
DEVELOPMENT_BUDGET = int(BUDGETS[-1])
VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "V1": {"implementation": "v1", "fast_scoring": False, "margin_rows_per_class": None, "lloyd_update": True},
    "F32": {"implementation": "v2", "fast_scoring": False, "margin_rows_per_class": None, "lloyd_update": True},
    "FAST": {"implementation": "v2", "fast_scoring": True, "margin_rows_per_class": None, "lloyd_update": True},
    "CAP32": {"implementation": "v2", "fast_scoring": True, "margin_rows_per_class": 32, "lloyd_update": True},
    "NOLLOYD": {"implementation": "v2", "fast_scoring": True, "margin_rows_per_class": 32, "lloyd_update": False},
}
DEVELOPMENT_METHODS = tuple(VARIANT_SPECS) + ("LP-FULL",)
PROMOTION_PRECEDENCE = ("FAST", "CAP32", "NOLLOYD")


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


def validate_protocol() -> tuple[str, dict[str, Any]]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SIDECAR.is_file():
        raise RuntimeError("speed protocol and sidecar must exist before outcomes")
    observed = sha256_path(PROTOCOL_PATH)
    expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or expected != observed:
        raise RuntimeError("speed protocol sidecar mismatch")
    protocol = _read_json(PROTOCOL_PATH)
    if protocol.get("status") != "frozen_corrective_amendment_before_valid_rerun":
        raise RuntimeError("speed protocol is not the frozen corrective version")
    if protocol.get("variant_specs") != VARIANT_SPECS:
        raise RuntimeError("speed protocol variant table does not match implementation")
    if protocol.get("development", {}).get("methods") != list(DEVELOPMENT_METHODS):
        raise RuntimeError("speed protocol development methods mismatch")
    development = protocol.get("development", {})
    if (
        development.get("models") != list(MODELS)
        or development.get("replicates") != [DEVELOPMENT_REPLICATE]
        or development.get("arms") != list(ARMS)
        or development.get("budgets_per_class") != [DEVELOPMENT_BUDGET]
        or development.get("panel_count") != 30
        or development.get("method_row_count") != 180
    ):
        raise RuntimeError("speed protocol development grid mismatch")
    if protocol.get("promotion_precedence") != list(PROMOTION_PRECEDENCE):
        raise RuntimeError("speed protocol promotion precedence mismatch")
    return observed, protocol


def _thread_environment() -> dict[str, str]:
    observed = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT}
    if observed != THREAD_ENVIRONMENT:
        raise RuntimeError(
            f"one-thread environment mismatch: expected {THREAD_ENVIRONMENT!r}, got {observed!r}"
        )
    return {key: str(value) for key, value in observed.items()}


def _source_paths() -> tuple[Path, ...]:
    local = (
        PACKAGE_DIR / "scalable.py",
        PACKAGE_DIR / "scalable_v2.py",
        PACKAGE_DIR / "speed_experiment.py",
        PACKAGE_DIR / "speed_analysis.py",
        PROTOCOL_PATH,
        ROOT / "pyproject.toml",
    )
    return local + tuple(sorted((ROOT / "overlapindex").glob("*.py"))) + tuple(
        sorted((ROOT / "experiments" / "m50_backbone_ranking").glob("*.py"))
    )


def source_identity() -> dict[str, Any]:
    paths = _source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing speed experiment sources: {missing!r}")
    local_hashes = {str(path): sha256_path(path) for path in paths}
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
    v1_raw = V1_ARTIFACT_DIR / "raw_results.json"
    v1_manifest = V1_ARTIFACT_DIR / "manifest.json"
    v1_manifest_payload = _read_json(v1_manifest)
    v1_raw_sha = sha256_path(v1_raw)
    if v1_manifest_payload.get("raw_results_sha256") != v1_raw_sha:
        raise RuntimeError("v1 scalable Food artifact hash mismatch")
    code_identity = hashlib.sha256(canonical_json(local_hashes).encode("utf-8")).hexdigest()
    return {
        "local_source_hashes": local_hashes,
        "code_identity_sha256": code_identity,
        "archived_source_hashes": archived_hashes,
        "v1_food_raw_sha256": v1_raw_sha,
        "v1_food_manifest_sha256": sha256_path(v1_manifest),
    }


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


def _candidate_identity(candidate_id: str, k: Mapping[Any, int], seed: int) -> dict[str, Any]:
    spec = VARIANT_SPECS[candidate_id]
    return {
        "candidate_id": candidate_id,
        "spec": dict(spec),
        "k_per_class": {str(key): int(value) for key, value in k.items()},
        "kmeans_kwargs": {"random_state": int(seed)},
        "weight_floor": 0.05,
        "memory_budget_mb": 64,
        "row_cap": None,
    }


def _build_selector(candidate_id: str, k: Mapping[Any, int], seed: int) -> Any:
    spec = VARIANT_SPECS[candidate_id]
    common = {
        "k_per_class": dict(k),
        "kmeans_kwargs": {"random_state": int(seed)},
        "weight_floor": 0.05,
        "memory_budget_mb": 64,
    }
    if spec["implementation"] == "v1":
        return ScalableOneUpdateRelevanceKMeans(**common)
    return FastScalableRelevanceKMeans(
        **common,
        fast_scoring=bool(spec["fast_scoring"]),
        margin_rows_per_class=spec["margin_rows_per_class"],
        lloyd_update=bool(spec["lloyd_update"]),
    )


def cross_fitted_variant(
    matrix: np.ndarray, labels: np.ndarray, *, candidate_id: str, seed: int
) -> dict[str, Any]:
    if candidate_id not in VARIANT_SPECS:
        raise ValueError(f"unknown speed variant {candidate_id!r}")
    values = food101._row_l2(np.asarray(matrix, dtype=np.float32))
    target = np.asarray(labels)
    folds = food101._stratified_folds(target, n_splits=FOLDS, seed=int(seed))
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        k = _k_per_class(target, train)
        identity = _candidate_identity(candidate_id, k, fold_seed)
        identity_sha = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        selector = _build_selector(candidate_id, k, fold_seed)
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
            raise RuntimeError(f"{candidate_id} emitted a non-finite score")
        diagnostics = dict(selector.diagnostics_ or {})
        structural = {
            key: value for key, value in diagnostics.items() if not key.endswith("_wall_seconds")
        }
        stage_runtime = {
            key: value for key, value in diagnostics.items() if key.endswith("_wall_seconds")
        }
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
        "candidate_id": candidate_id,
        "candidate_name": candidate_id.lower(),
        "mode": "speed_variant",
        "prototype_refinement": False,
        "score": float(np.mean([row["score"] for row in fold_rows])),
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


def _execute(method: str, values: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, Any]:
    if method in VARIANT_SPECS:
        return cross_fitted_variant(values, labels, candidate_id=method, seed=seed)
    if method == "LP-FULL":
        return food101._cross_fitted_score(values, labels, candidate=method, seed=seed)
    raise ValueError(f"unknown speed experiment method {method!r}")


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
                    "structural_diagnostics",
                    "score_diagnostics",
                    "candidate_config_sha256",
                )
                if key in row
            }
        )
    return {
        "candidate_id": result.get("candidate_id"),
        "score": result.get("score"),
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


def _method_row(
    result: Mapping[str, Any], *, panel: Mapping[str, Any], position: int, order: Sequence[str], wall: float, cpu: float
) -> dict[str, Any]:
    return {
        **dict(panel),
        "candidate_id": str(result["candidate_id"]),
        "candidate_name": result.get("candidate_name"),
        "score": float(result["score"]),
        "fit_wall_seconds": float(result.get("fit_wall_seconds", 0.0)),
        "fit_cpu_seconds": float(result.get("fit_cpu_seconds", 0.0)),
        "score_fixed_wall_seconds": float(result.get("score_fixed_wall_seconds", 0.0)),
        "score_fixed_cpu_seconds": float(result.get("score_fixed_cpu_seconds", 0.0)),
        "total_wall_seconds": float(wall),
        "total_cpu_seconds": float(cpu),
        "folds": result.get("folds", []),
        "execution_position": int(position),
        "execution_order": list(order),
        "warmup_excluded": True,
        "status": "ok",
    }


def _panel_id(model: str, replicate: int, arm: str, budget: int) -> str:
    return f"{model}__r{replicate}__{arm}__b{budget}"


def _ordered(methods: tuple[str, ...], panel_index: int) -> tuple[str, ...]:
    offset = int(panel_index) % len(methods)
    return methods[offset:] + methods[:offset]


def _validate_lock(lock_path: Path, protocol_sha: str, sources: Mapping[str, Any]) -> dict[str, Any]:
    lock = _read_json(lock_path)
    if lock.get("status") != "locked_for_full_evaluation":
        raise RuntimeError("development decision did not authorize full evaluation")
    selected = lock.get("selected_candidate")
    if selected not in PROMOTION_PRECEDENCE:
        raise RuntimeError("development lock selected an invalid candidate")
    if lock.get("protocol_sha256") != protocol_sha or lock.get("source_identity") != sources:
        raise RuntimeError("development lock protocol/source identity mismatch")
    expected_sha = hashlib.sha256(
        canonical_json({key: value for key, value in lock.items() if key != "lock_sha256"}).encode("utf-8")
    ).hexdigest()
    if lock.get("lock_sha256") != expected_sha:
        raise RuntimeError("development lock hash mismatch")
    return lock


def run_stage(
    output: Path,
    *,
    stage: str,
    resume: bool = False,
    development_lock: Path | None = None,
) -> dict[str, Any]:
    if stage not in {"development", "full"}:
        raise ValueError("stage must be 'development' or 'full'")
    protocol_sha, _protocol = validate_protocol()
    sources = source_identity()
    lock = None
    if stage == "development":
        methods = DEVELOPMENT_METHODS
        models, replicates, arms, budgets = MODELS, (DEVELOPMENT_REPLICATE,), ARMS, (DEVELOPMENT_BUDGET,)
    else:
        if development_lock is None:
            raise RuntimeError("full stage requires the immutable development lock")
        lock = _validate_lock(Path(development_lock), protocol_sha, sources)
        methods = (str(lock["selected_candidate"]), "LP-FULL")
        models, replicates, arms, budgets = MODELS, REPLICATES, ARMS, BUDGETS
    identity = {
        "study": "scalable_relevance_speed_food101",
        "stage": stage,
        "protocol_sha256": protocol_sha,
        "source_identity": sources,
        "methods": list(methods),
        "models": list(models),
        "replicates": list(replicates),
        "arms": list(arms),
        "budgets": list(budgets),
        "development_lock_sha256": None if lock is None else lock["lock_sha256"],
        "thread_environment": _thread_environment(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    identity_sha = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.is_file():
            raise RuntimeError("refusing a non-empty speed output without identity-locked resume")
        existing = _read_json(manifest_path)
        if existing.get("artifact_status") != "running" or existing.get("run_identity") != identity:
            raise RuntimeError("speed experiment resume identity mismatch")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            manifest_path,
            {"schema_version": 1, "artifact_status": "running", "run_identity": identity, "run_identity_sha256": identity_sha},
        )

    driver, source_result, cohort, _prior, archived_hashes = food101._load_archived_inputs(
        food101.DEFAULT_DRIVER, food101.DEFAULT_RESULT, food101.DEFAULT_COHORT, food101.DEFAULT_PRIOR
    )
    if archived_hashes != sources["archived_source_hashes"]:
        raise RuntimeError("loaded source hashes changed after preflight")
    sample_ids = [str(value) for value in cohort["extracted_sample_ids"]]
    sample_hash = food101._sample_ids_hash(sample_ids)
    food101._validate_all_cache_manifests(food101.DEFAULT_CACHE, sample_hash)
    labels = np.asarray([value.split("/")[2] for value in sample_ids], dtype=object)
    roles = {int(key): value for key, value in cohort["roles"].items()}
    panels = [
        (model, int(replicate), arm, int(budget))
        for model in models for replicate in replicates for arm in arms for budget in budgets
    ]
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    expected_names = {_panel_id(*panel) + ".json" for panel in panels}
    unexpected = {path.name for path in checkpoints.iterdir()} - expected_names
    if unexpected:
        raise RuntimeError(f"unexpected speed checkpoints: {sorted(unexpected)!r}")
    rows: list[dict[str, Any]] = []
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
            nested = driver._nested_stratified_indices(target, budgets, seed=SEED + int(replicate) + 23)
            for arm in arms:
                lam, nu = arm_lookup[arm]
                transformed = driver._bridge_transform(
                    raw, donor=banks["donor"], mode=banks["mode"], nuisance=banks["nuisance"], q=1.0, lam=lam, nu=nu
                )
                for budget in budgets:
                    selected = np.asarray(nested[int(budget)], dtype=np.int64)
                    subset = np.asarray(transformed[selected], dtype=np.float32)
                    subset_target = target[selected]
                    panel_id = _panel_id(model, int(replicate), arm, int(budget))
                    order = _ordered(methods, panel_index)
                    panel = {"model": model, "backbone": model, "replicate": int(replicate), "arm": arm, "budget": int(budget), "stage": stage}
                    panel_identity = {
                        "run_identity_sha256": identity_sha,
                        "panel_id": panel_id,
                        "method_order": list(order),
                        "cache_sha256": str(cache_manifest["sha256"]),
                        "values_sha256": hashlib.sha256(subset.tobytes()).hexdigest(),
                        "labels_sha256": hashlib.sha256(canonical_json(subset_target.tolist()).encode("utf-8")).hexdigest(),
                    }
                    checkpoint = checkpoints / f"{panel_id}.json"
                    if resume and checkpoint.is_file():
                        cached = _read_json(checkpoint)
                        if cached.get("artifact_status") != "completed" or cached.get("identity") != panel_identity:
                            raise RuntimeError(f"checkpoint identity mismatch for {panel_id}")
                        block = cached.get("rows")
                        if not isinstance(block, list) or len(block) != len(methods):
                            raise RuntimeError(f"checkpoint block is incomplete for {panel_id}")
                        rows.extend(dict(row) for row in block)
                        for method, record in cached.get("determinism", {}).items():
                            if method in deterministic and deterministic[method] != record:
                                raise RuntimeError("determinism evidence changed across checkpoints")
                            deterministic[method] = dict(record)
                        panel_index += 1
                        continue
                    block: list[dict[str, Any]] = []
                    local_determinism: dict[str, dict[str, Any]] = {}
                    panel_seed = SEED + int(replicate)
                    for position, method in enumerate(order):
                        warmup = None
                        if method not in deterministic:
                            warmup = _execute(method, subset, subset_target, panel_seed)
                        outer_wall, outer_cpu = time.perf_counter(), time.process_time()
                        result = _execute(method, subset, subset_target, panel_seed)
                        wall = time.perf_counter() - outer_wall
                        cpu = time.process_time() - outer_cpu
                        if warmup is not None:
                            record = _determinism(warmup, result)
                            record.update({"candidate_id": method, "panel_id": panel_id})
                            if record["exact"] is not True:
                                raise RuntimeError(f"nondeterministic speed method {method}")
                            deterministic[method] = record
                            local_determinism[method] = record
                        block.append(
                            _method_row(result, panel=panel, position=position, order=order, wall=wall, cpu=cpu)
                        )
                    _atomic_json(
                        checkpoint,
                        {"artifact_status": "completed", "identity": panel_identity, "rows": block, "determinism": local_determinism},
                    )
                    rows.extend(block)
                    panel_index += 1
    if len(rows) != len(panels) * len(methods):
        raise RuntimeError("speed experiment method grid is incomplete")
    if set(deterministic) != set(methods) or any(record.get("exact") is not True for record in deterministic.values()):
        raise RuntimeError("speed experiment determinism evidence is incomplete")
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
        raise RuntimeError("speed experiment reference grid is incomplete")
    payload = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": identity["study"],
        "stage": stage,
        "retrospective_development_only": True,
        "run_identity": identity,
        "run_identity_sha256": identity_sha,
        "determinism": deterministic,
        "reference_rows": reference_rows,
        "selector_rows": rows,
    }
    raw_path = output / "raw_results.json"
    _atomic_json(raw_path, payload)
    terminal = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": identity["study"],
        "stage": stage,
        "run_identity": identity,
        "run_identity_sha256": identity_sha,
        "raw_results_sha256": sha256_path(raw_path),
        "counts": {"panels": len(panels), "selector_rows": len(rows), "reference_rows": len(reference_rows), "checkpoints": len(list(checkpoints.glob("*.json")))},
    }
    _atomic_json(manifest_path, terminal)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "full"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--development-lock", type=Path)
    args = parser.parse_args()
    payload = run_stage(
        args.output, stage=args.stage, resume=args.resume, development_lock=args.development_lock
    )
    print(canonical_json({"artifact_status": payload["artifact_status"], "stage": payload["stage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
