"""Frozen Food-101 development screen for fixed-Lloyd class prototypes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn

from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans.fused_prototypes import (
    FusedFastRelevanceKMeans,
)
from experiments.scalable_relevance_kmeans.scalable_v2 import (
    FastScalableRelevanceKMeans,
)


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / "fused_food_protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "fused_food_protocol.sha256"
PRIOR_SPEED_DIR = ROOT / "artifacts" / "scalable_relevance_kmeans" / "speed_development_v2"
FUSED_TOY_DIR = ROOT / "artifacts" / "scalable_relevance_kmeans" / "fused_toy_v1"

MODELS = tuple(food101.MODELS)
ARMS = tuple(name for name, _lam, _nu in food101.ARMS)
REPLICATE = 0
BUDGET = 80
FOLDS = int(food101.FOLDS)
SEED = int(food101.SEED)
METHODS = ("NOLLOYD", "LOOP3", "FUSED3", "LP-FULL")
CANDIDATES = ("FUSED3", "LOOP3")

THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}

PRIOR_SPEED_RAW_SHA256 = "82bb209046d756da79d49ce05c98e442e621e6cdbbde09480a161c78d2e3ce31"
PRIOR_SPEED_MANIFEST_SHA256 = "7c9c859628c6786161e5e660775a38caa62260404acdfaa7df7ef3b7dbce4af2"
FUSED_TOY_RAW_SHA256 = "f52e8ef1a665f9289204c0c6295e666ca78a5f682801667d6bf0d757f6362f4e"
FUSED_TOY_MANIFEST_SHA256 = "0b556faccf6b7ed21898ffa72066f78acde6e56fba755b9eae2deb44ef1ee571"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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


def _candidate_specs() -> dict[str, dict[str, Any]]:
    return {
        "NOLLOYD": {
            "role": "immutable_current_reference",
            "implementation": "FastScalableRelevanceKMeans",
            "prototype_backend": "one_sklearn_minibatch_kmeans_per_class",
            "margin_rows_per_class": 32,
            "lloyd_update": False,
            "weight_floor": 0.05,
            "memory_budget_mb": 64,
        },
        "LOOP3": {
            "role": "promotable_candidate",
            "implementation": "FusedFastRelevanceKMeans",
            "prototype_backend": "looped_fixed_lloyd",
            "prototype_iterations": 3,
            "margin_rows_per_class": 32,
            "lloyd_update": False,
            "weight_floor": 0.05,
            "memory_budget_mb": 64,
        },
        "FUSED3": {
            "role": "promotable_candidate",
            "implementation": "FusedFastRelevanceKMeans",
            "prototype_backend": "fused_fixed_lloyd",
            "prototype_iterations": 3,
            "margin_rows_per_class": 32,
            "lloyd_update": False,
            "weight_floor": 0.05,
            "memory_budget_mb": 64,
        },
        "LP-FULL": {
            "role": "fresh_comparator",
            "implementation": "five_fold_normalizer_logistic_regression",
            "C": 1.0,
            "max_iter": 2000,
            "n_jobs": 1,
        },
    }


def validate_protocol() -> tuple[str, dict[str, Any]]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SIDECAR.is_file():
        raise RuntimeError("fused Food protocol and sidecar must exist before outcomes")
    observed = sha256_path(PROTOCOL_PATH)
    expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or observed != expected:
        raise RuntimeError("fused Food protocol sidecar mismatch")
    protocol = _read_json(PROTOCOL_PATH)
    if protocol.get("status") != "frozen_before_food_outcomes":
        raise RuntimeError("fused Food protocol is not frozen before outcomes")
    grid = protocol.get("development_grid")
    expected_grid = {
        "models": list(MODELS),
        "replicates": [REPLICATE],
        "arms": list(ARMS),
        "budgets_per_class": [BUDGET],
        "folds": FOLDS,
        "k_per_class_max": int(food101.K),
        "panel_count": 30,
        "methods": list(METHODS),
        "method_row_count": 120,
        "reference_row_count": 120,
    }
    if grid != expected_grid:
        raise RuntimeError("fused Food protocol grid does not match implementation")
    if protocol.get("candidate_specs") != _candidate_specs():
        raise RuntimeError("fused Food candidate table does not match implementation")
    if protocol.get("selection_rule", {}).get("eligible_candidates") != list(CANDIDATES):
        raise RuntimeError("fused Food candidate eligibility order mismatch")
    evidence = protocol.get("source_evidence", {})
    expected_evidence = {
        "valid_speed_development_v2_raw_sha256": PRIOR_SPEED_RAW_SHA256,
        "valid_speed_development_v2_manifest_sha256": PRIOR_SPEED_MANIFEST_SHA256,
        "fused_toy_raw_sha256": FUSED_TOY_RAW_SHA256,
        "fused_toy_manifest_sha256": FUSED_TOY_MANIFEST_SHA256,
    }
    if evidence != expected_evidence:
        raise RuntimeError("fused Food frozen source-evidence table mismatch")
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
        PACKAGE_DIR / "fused_prototypes.py",
        PACKAGE_DIR / "fused_food_screen.py",
        PACKAGE_DIR / "fused_food_analysis.py",
        PROTOCOL_PATH,
        ROOT / "pyproject.toml",
    )
    return local + tuple(sorted((ROOT / "overlapindex").glob("*.py"))) + tuple(
        sorted((ROOT / "experiments" / "m50_backbone_ranking").glob("*.py"))
    )


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _external_evidence() -> dict[str, str]:
    paths = {
        "prior_speed_raw": PRIOR_SPEED_DIR / "raw_results.json",
        "prior_speed_manifest": PRIOR_SPEED_DIR / "manifest.json",
        "fused_toy_raw": FUSED_TOY_DIR / "raw_results.json",
        "fused_toy_manifest": FUSED_TOY_DIR / "manifest.json",
    }
    expected = {
        "prior_speed_raw": PRIOR_SPEED_RAW_SHA256,
        "prior_speed_manifest": PRIOR_SPEED_MANIFEST_SHA256,
        "fused_toy_raw": FUSED_TOY_RAW_SHA256,
        "fused_toy_manifest": FUSED_TOY_MANIFEST_SHA256,
    }
    observed = {name: sha256_path(path) for name, path in paths.items()}
    if observed != expected:
        raise RuntimeError(f"fused Food development-evidence hash mismatch: {observed!r}")
    return observed


def _cache_identities() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        matrix, payload = food101._load_cache(
            food101.DEFAULT_CACHE, model, food101.SAMPLE_IDS_SHA256
        )
        output[model] = {
            "manifest_sha256": sha256_path(
                Path(food101.DEFAULT_CACHE) / f"food101_{model}_final.json"
            ),
            "matrix_sha256": str(payload["sha256"]),
            "shape": [int(value) for value in matrix.shape],
            "dtype": str(matrix.dtype),
            "sample_ids_sha256": str(payload["sample_ids_sha256"]),
        }
        del matrix
    return output


def source_identity() -> dict[str, Any]:
    paths = _source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing fused Food sources: {missing!r}")
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
    return {
        "local_source_hashes": local_hashes,
        "code_identity_sha256": _hash_payload(local_hashes),
        "archived_source_hashes": archived_hashes,
        "development_evidence_hashes": _external_evidence(),
        "cache_identities": _cache_identities(),
        "repository": {
            "branch": _git_output("branch", "--show-current"),
            "experiment_commit": _git_output("rev-parse", "HEAD"),
            "develop_merge_base": _git_output("merge-base", "HEAD", "develop"),
            "dirty_status_recorded": _git_output("status", "--short"),
        },
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


def _candidate_identity(
    candidate_id: str, k_per_class: Mapping[Any, int], fold_seed: int
) -> dict[str, Any]:
    spec = _candidate_specs()[candidate_id]
    identity: dict[str, Any] = {
        "candidate_id": candidate_id,
        "spec": dict(spec),
        "k_per_class": {str(key): int(value) for key, value in k_per_class.items()},
        "fold_seed": int(fold_seed),
    }
    if candidate_id == "NOLLOYD":
        identity["kmeans_kwargs"] = {"random_state": int(fold_seed)}
    return identity


def _build_selector(candidate_id: str, k_per_class: Mapping[Any, int], seed: int) -> Any:
    if candidate_id == "NOLLOYD":
        return FastScalableRelevanceKMeans(
            k_per_class=dict(k_per_class),
            kmeans_kwargs={"random_state": int(seed)},
            weight_floor=0.05,
            memory_budget_mb=64,
            fast_scoring=True,
            margin_rows_per_class=32,
            lloyd_update=False,
        )
    if candidate_id not in {"LOOP3", "FUSED3"}:
        raise ValueError(f"unknown fused Food selector {candidate_id!r}")
    return FusedFastRelevanceKMeans(
        k_per_class=dict(k_per_class),
        seed=int(seed),
        prototype_iterations=3,
        prototype_backend="looped" if candidate_id == "LOOP3" else "fused",
        margin_rows_per_class=32,
        weight_floor=0.05,
        memory_budget_mb=64,
    )


def _array_digest(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _numerical_state_sha256(selector: Any) -> str:
    digest = hashlib.sha256()
    _array_digest(digest, "centers", selector.centers_)
    _array_digest(digest, "owners", np.asarray(selector.owners_, dtype=str))
    _array_digest(digest, "weights", selector.weights_)
    return digest.hexdigest()


def _clock_free(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _clock_free(item)
            for key, item in value.items()
            if not str(key).endswith("_wall_seconds")
            and not str(key).endswith("_cpu_seconds")
            and str(key) != "wall_seconds"
        }
    if isinstance(value, (list, tuple)):
        return [_clock_free(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def cross_fitted_candidate(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    candidate_id: str,
    seed: int,
) -> dict[str, Any]:
    if candidate_id not in {"NOLLOYD", "LOOP3", "FUSED3"}:
        raise ValueError(f"unknown candidate {candidate_id!r}")
    values = food101._row_l2(np.asarray(matrix, dtype=np.float32))
    target = np.asarray(labels)
    folds = food101._stratified_folds(target, n_splits=FOLDS, seed=int(seed))
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        k = _k_per_class(target, train)
        identity = _candidate_identity(candidate_id, k, fold_seed)
        selector = _build_selector(candidate_id, k, fold_seed)
        fit_wall, fit_cpu = time.perf_counter(), time.process_time()
        selector.fit(values[train], target[train])
        fit_wall_elapsed = time.perf_counter() - fit_wall
        fit_cpu_elapsed = time.process_time() - fit_cpu
        state_before = _numerical_state_sha256(selector)
        score_wall, score_cpu = time.perf_counter(), time.process_time()
        score, score_diagnostics = selector.score_fixed_with_diagnostics(
            values[holdout], target[holdout]
        )
        score_wall_elapsed = time.perf_counter() - score_wall
        score_cpu_elapsed = time.process_time() - score_cpu
        state_after = _numerical_state_sha256(selector)
        if state_before != state_after:
            raise RuntimeError(f"{candidate_id} score_fixed mutated fitted state")
        if not np.isfinite(score):
            raise RuntimeError(f"{candidate_id} emitted a non-finite score")
        diagnostics = dict(selector.diagnostics_ or {})
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
                "numerical_state_sha256": state_before,
                "state_unchanged_after_score_fixed": True,
                "structural_diagnostics": _clock_free(diagnostics),
                "runtime_diagnostics": {
                    key: item
                    for key, item in diagnostics.items()
                    if str(key).endswith("_wall_seconds")
                },
                "score_diagnostics": _clock_free(score_diagnostics),
                "candidate_config_identity": identity,
                "candidate_config_sha256": _hash_payload(identity),
            }
        )
    return {
        "candidate_id": candidate_id,
        "candidate_name": candidate_id.lower(),
        "mode": "fixed_lloyd_relevance" if candidate_id != "NOLLOYD" else "current_nolloyd",
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


def _execute_method(
    method: str, values: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, Any]:
    if method == "LP-FULL":
        return food101._cross_fitted_score(values, labels, candidate=method, seed=seed)
    return cross_fitted_candidate(values, labels, candidate_id=method, seed=seed)


def _signature(result: Mapping[str, Any]) -> dict[str, Any]:
    return _clock_free(
        {
            "candidate_id": result.get("candidate_id"),
            "candidate_name": result.get("candidate_name"),
            "mode": result.get("mode"),
            "prototype_refinement": result.get("prototype_refinement"),
            "score": result.get("score"),
            "folds": result.get("folds", []),
        }
    )


def _determinism(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    first_sha = _hash_payload(_signature(first))
    second_sha = _hash_payload(_signature(second))
    return {
        "first_signature_sha256": first_sha,
        "second_signature_sha256": second_sha,
        "exact": first_sha == second_sha,
        "runtime_fields_excluded": True,
    }


def _method_order(panel_index: int) -> tuple[str, ...]:
    offset = int(panel_index) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def _panel_id(model: str, arm: str) -> str:
    return f"{model}__r{REPLICATE}__{arm}__b{BUDGET}"


def _method_row(
    result: Mapping[str, Any],
    *,
    model: str,
    arm: str,
    position: int,
    order: Sequence[str],
    wall: float,
    cpu: float,
) -> dict[str, Any]:
    return {
        "stage": "development",
        "model": model,
        "backbone": model,
        "replicate": REPLICATE,
        "arm": arm,
        "budget": BUDGET,
        "candidate_id": str(result["candidate_id"]),
        "candidate_name": result.get("candidate_name"),
        "mode": result.get("mode"),
        "prototype_refinement_enabled": False,
        "score": float(result["score"]),
        "fit_wall_seconds": float(result.get("fit_wall_seconds", 0.0)),
        "fit_cpu_seconds": float(result.get("fit_cpu_seconds", 0.0)),
        "score_fixed_wall_seconds": float(result.get("score_fixed_wall_seconds", 0.0)),
        "score_fixed_cpu_seconds": float(result.get("score_fixed_cpu_seconds", 0.0)),
        "total_wall_seconds": float(wall),
        "total_cpu_seconds": float(cpu),
        "folds": list(result.get("folds", [])),
        "execution_position": int(position),
        "execution_order": list(order),
        "warmup_excluded": True,
        "status": "ok",
        "error": None,
    }


def _load_prior_speed_rows() -> dict[tuple[str, str], float]:
    manifest_path = PRIOR_SPEED_DIR / "manifest.json"
    raw_path = PRIOR_SPEED_DIR / "raw_results.json"
    if sha256_path(manifest_path) != PRIOR_SPEED_MANIFEST_SHA256:
        raise RuntimeError("prior speed terminal manifest hash mismatch")
    if sha256_path(raw_path) != PRIOR_SPEED_RAW_SHA256:
        raise RuntimeError("prior speed raw-results hash mismatch")
    manifest = _read_json(manifest_path)
    raw = _read_json(raw_path)
    if (
        manifest.get("artifact_status") != "completed"
        or raw.get("artifact_status") != "completed"
        or manifest.get("raw_results_sha256") != PRIOR_SPEED_RAW_SHA256
    ):
        raise RuntimeError("prior speed artifact is not the valid completed artifact")
    lookup: dict[tuple[str, str], float] = {}
    for row in raw.get("selector_rows", ()):
        if not isinstance(row, Mapping):
            continue
        if (
            int(row.get("replicate", -1)) == REPLICATE
            and int(row.get("budget", -1)) == BUDGET
            and str(row.get("candidate_id")) in {"NOLLOYD", "LP-FULL"}
        ):
            key = (_panel_id(str(row["model"]), str(row["arm"])), str(row["candidate_id"]))
            if key in lookup:
                raise RuntimeError("prior speed artifact duplicates a parity row")
            lookup[key] = float(row["score"])
    if len(lookup) != 60:
        raise RuntimeError("prior speed artifact lacks exact NOLLOYD/LP parity rows")
    return lookup


def _paired_parity(
    block: Sequence[Mapping[str, Any]],
    *,
    panel_id: str,
    prior: Mapping[tuple[str, str], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_method = {str(row["candidate_id"]): row for row in block}
    prior_rows = []
    for method in ("NOLLOYD", "LP-FULL"):
        current = float(by_method[method]["score"])
        expected = float(prior[(panel_id, method)])
        prior_rows.append(
            {
                "panel_id": panel_id,
                "candidate_id": method,
                "current_score": current,
                "prior_score": expected,
                "delta": current - expected,
                "exact": current == expected,
            }
        )
    looped = by_method["LOOP3"]
    fused = by_method["FUSED3"]
    loop_folds = looped.get("folds", [])
    fused_folds = fused.get("folds", [])
    fold_parity = []
    if len(loop_folds) != FOLDS or len(fused_folds) != FOLDS:
        raise RuntimeError("LOOP3/FUSED3 fold rows are incomplete")
    for left, right in zip(loop_folds, fused_folds):
        exact = (
            left.get("fold") == right.get("fold")
            and left.get("score") == right.get("score")
            and left.get("numerical_state_sha256") == right.get("numerical_state_sha256")
            and left.get("score_diagnostics") == right.get("score_diagnostics")
        )
        fold_parity.append(
            {
                "fold": int(left["fold"]),
                "looped_numerical_state_sha256": left.get("numerical_state_sha256"),
                "fused_numerical_state_sha256": right.get("numerical_state_sha256"),
                "exact": bool(exact),
            }
        )
    implementation = {
        "panel_id": panel_id,
        "exact_panel_score": float(looped["score"]) == float(fused["score"]),
        "exact_folds": all(row["exact"] for row in fold_parity),
        "folds": fold_parity,
    }
    implementation["exact"] = bool(
        implementation["exact_panel_score"] and implementation["exact_folds"]
    )
    return prior_rows, implementation


def _run_identity(protocol_sha: str, sources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "study": "fused_class_prototypes_food101_screen",
        "stage": "development",
        "protocol_sha256": protocol_sha,
        "source_identity": dict(sources),
        "methods": list(METHODS),
        "models": list(MODELS),
        "replicates": [REPLICATE],
        "arms": list(ARMS),
        "budgets": [BUDGET],
        "thread_environment": _thread_environment(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }


def run_screen(output: Path, *, resume: bool = False) -> dict[str, Any]:
    protocol_sha, _protocol = validate_protocol()
    sources = source_identity()
    identity = _run_identity(protocol_sha, sources)
    identity_sha = _hash_payload(identity)
    prior_lookup = _load_prior_speed_rows()
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.is_file():
            raise RuntimeError("refusing non-empty fused Food output without locked resume")
        existing = _read_json(manifest_path)
        if existing.get("artifact_status") != "running" or existing.get("run_identity") != identity:
            raise RuntimeError("fused Food resume manifest identity mismatch")
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

    driver, source_result, cohort, _prior, archived_hashes = food101._load_archived_inputs(
        food101.DEFAULT_DRIVER,
        food101.DEFAULT_RESULT,
        food101.DEFAULT_COHORT,
        food101.DEFAULT_PRIOR,
    )
    if archived_hashes != sources["archived_source_hashes"]:
        raise RuntimeError("loaded archived sources changed after preflight")
    sample_ids = [str(value) for value in cohort["extracted_sample_ids"]]
    sample_hash = food101._sample_ids_hash(sample_ids)
    if sample_hash != food101.SAMPLE_IDS_SHA256:
        raise RuntimeError("Food sample identity changed")
    labels = np.asarray([value.split("/")[2] for value in sample_ids], dtype=object)
    roles = {int(key): value for key, value in cohort["roles"].items()}
    panels = [(model, arm) for model in MODELS for arm in ARMS]
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    expected_names = {_panel_id(model, arm) + ".json" for model, arm in panels}
    unexpected = {path.name for path in checkpoint_dir.iterdir()} - expected_names
    if unexpected:
        raise RuntimeError(f"unexpected fused Food checkpoints: {sorted(unexpected)!r}")

    rows: list[dict[str, Any]] = []
    prior_parity: list[dict[str, Any]] = []
    implementation_parity: list[dict[str, Any]] = []
    deterministic: dict[str, dict[str, Any]] = {}
    arm_lookup = {name: (lam, nu) for name, lam, nu in food101.ARMS}
    panel_index = 0
    for model in MODELS:
        matrix, cache_manifest = food101._load_cache(
            food101.DEFAULT_CACHE, model, sample_hash
        )
        selector_indices = np.asarray(roles[REPLICATE]["selector"], dtype=np.int64)
        raw = np.asarray(matrix[selector_indices], dtype=np.float32)
        target = labels[selector_indices]
        banks = driver._paired_split_banks(raw, target, seed=SEED + REPLICATE + 11)
        nested = driver._nested_stratified_indices(target, [BUDGET], seed=SEED + REPLICATE + 23)
        selected = np.asarray(nested[BUDGET], dtype=np.int64)
        for arm in ARMS:
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
            subset = np.asarray(transformed[selected], dtype=np.float32)
            subset_target = target[selected]
            panel_id = _panel_id(model, arm)
            order = _method_order(panel_index)
            panel_identity = {
                "run_identity_sha256": identity_sha,
                "panel_id": panel_id,
                "model": model,
                "replicate": REPLICATE,
                "arm": arm,
                "budget": BUDGET,
                "method_order": list(order),
                "cache_sha256": str(cache_manifest["sha256"]),
                "values_sha256": hashlib.sha256(subset.tobytes()).hexdigest(),
                "labels_sha256": _hash_payload(subset_target.tolist()),
            }
            checkpoint = checkpoint_dir / f"{panel_id}.json"
            if resume and checkpoint.is_file():
                cached = _read_json(checkpoint)
                if cached.get("artifact_status") != "completed" or cached.get("identity") != panel_identity:
                    raise RuntimeError(f"checkpoint identity/status mismatch for {panel_id}")
                block = cached.get("rows")
                prior_block = cached.get("prior_parity")
                implementation_block = cached.get("implementation_parity")
                if not isinstance(block, list) or len(block) != len(METHODS):
                    raise RuntimeError(f"checkpoint method block incomplete for {panel_id}")
                if not isinstance(prior_block, list) or len(prior_block) != 2:
                    raise RuntimeError(f"checkpoint prior parity incomplete for {panel_id}")
                if not isinstance(implementation_block, Mapping):
                    raise RuntimeError(f"checkpoint implementation parity missing for {panel_id}")
                if any(item.get("exact") is not True for item in prior_block):
                    raise RuntimeError(f"cached prior parity failed for {panel_id}")
                if implementation_block.get("exact") is not True:
                    raise RuntimeError(f"cached LOOP3/FUSED3 parity failed for {panel_id}")
                rows.extend(dict(row) for row in block)
                prior_parity.extend(dict(row) for row in prior_block)
                implementation_parity.append(dict(implementation_block))
                for method, record in cached.get("determinism", {}).items():
                    if method in deterministic and deterministic[method] != record:
                        raise RuntimeError("determinism evidence changed across checkpoints")
                    deterministic[method] = dict(record)
                panel_index += 1
                continue

            block: list[dict[str, Any]] = []
            local_determinism: dict[str, dict[str, Any]] = {}
            for position, method in enumerate(order):
                warmup = None
                if method not in deterministic:
                    warmup = _execute_method(method, subset, subset_target, SEED + REPLICATE)
                started_wall, started_cpu = time.perf_counter(), time.process_time()
                result = _execute_method(method, subset, subset_target, SEED + REPLICATE)
                wall = time.perf_counter() - started_wall
                cpu = time.process_time() - started_cpu
                if warmup is not None:
                    record = _determinism(warmup, result)
                    record.update({"candidate_id": method, "panel_id": panel_id})
                    if record["exact"] is not True:
                        raise RuntimeError(f"nondeterministic fused Food method {method}")
                    deterministic[method] = record
                    local_determinism[method] = record
                block.append(
                    _method_row(
                        result,
                        model=model,
                        arm=arm,
                        position=position,
                        order=order,
                        wall=wall,
                        cpu=cpu,
                    )
                )
            prior_block, implementation_block = _paired_parity(
                block, panel_id=panel_id, prior=prior_lookup
            )
            if any(item["exact"] is not True for item in prior_block):
                _atomic_json(
                    checkpoint,
                    {
                        "artifact_status": "stopped_prior_parity_failure",
                        "identity": panel_identity,
                        "rows": block,
                        "prior_parity": prior_block,
                        "implementation_parity": implementation_block,
                        "determinism": local_determinism,
                    },
                )
                raise RuntimeError(f"NOLLOYD/LP prior parity failed for {panel_id}")
            if implementation_block["exact"] is not True:
                _atomic_json(
                    checkpoint,
                    {
                        "artifact_status": "stopped_implementation_parity_failure",
                        "identity": panel_identity,
                        "rows": block,
                        "prior_parity": prior_block,
                        "implementation_parity": implementation_block,
                        "determinism": local_determinism,
                    },
                )
                raise RuntimeError(f"LOOP3/FUSED3 numerical parity failed for {panel_id}")
            _atomic_json(
                checkpoint,
                {
                    "artifact_status": "completed",
                    "identity": panel_identity,
                    "rows": block,
                    "prior_parity": prior_block,
                    "implementation_parity": implementation_block,
                    "determinism": local_determinism,
                },
            )
            rows.extend(block)
            prior_parity.extend(prior_block)
            implementation_parity.append(implementation_block)
            panel_index += 1

    if len(rows) != 120 or len(prior_parity) != 60 or len(implementation_parity) != 30:
        raise RuntimeError("fused Food screen completed with an incomplete paired grid")
    if set(deterministic) != set(METHODS) or any(
        record.get("exact") is not True for record in deterministic.values()
    ):
        raise RuntimeError("fused Food determinism evidence is incomplete")
    reference_rows = [
        dict(row)
        for row in source_result.get("reference_rows", ())
        if isinstance(row, Mapping)
        and str(row.get("backbone")) in MODELS
        and int(row.get("replicate", -1)) == REPLICATE
        and str(row.get("arm")) in ARMS
    ]
    if len(reference_rows) != 120:
        raise RuntimeError("fused Food reference grid is incomplete")
    payload = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": identity["study"],
        "stage": "development",
        "retrospective_development_only": True,
        "run_identity": identity,
        "run_identity_sha256": identity_sha,
        "determinism": deterministic,
        "prior_parity": prior_parity,
        "implementation_parity": implementation_parity,
        "reference_rows": reference_rows,
        "selector_rows": rows,
    }
    raw_path = output / "raw_results.json"
    _atomic_json(raw_path, payload)
    terminal = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": identity["study"],
        "stage": "development",
        "run_identity": identity,
        "run_identity_sha256": identity_sha,
        "raw_results_sha256": sha256_path(raw_path),
        "counts": {
            "panels": 30,
            "selector_rows": len(rows),
            "prior_parity_rows": len(prior_parity),
            "implementation_parity_rows": len(implementation_parity),
            "reference_rows": len(reference_rows),
            "checkpoints": len(list(checkpoint_dir.glob("*.json"))),
        },
    }
    _atomic_json(manifest_path, terminal)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    payload = run_screen(args.output, resume=args.resume)
    print(canonical_json({"artifact_status": payload["artifact_status"], "stage": payload["stage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
