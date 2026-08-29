"""Hash-locked full Food-101 replay for the screen-selected FUSED3 method."""

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
from experiments.scalable_relevance_kmeans import fused_food_screen as screen


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / "fused_food_full_protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "fused_food_full_protocol.sha256"
SCREEN_RUN_DIR = ROOT / "artifacts" / "scalable_relevance_kmeans" / "fused_food_screen_v2"
SCREEN_ANALYSIS_DIR = (
    ROOT / "artifacts" / "scalable_relevance_kmeans" / "fused_food_analysis_v2_erratum"
)
PRIOR_REPLAY_PATH = Path(food101.DEFAULT_PRIOR)

METHODS = ("FUSED3", "REFINED-OI", "LP-FULL")
MODELS = tuple(food101.MODELS)
REPLICATES = tuple(food101.REPLICATES)
ARMS = tuple(name for name, _lam, _nu in food101.ARMS)
BUDGETS = tuple(food101.BUDGETS)
FOLDS = int(food101.FOLDS)
SEED = int(food101.SEED)

EXPECTED_ENVIRONMENT = {
    "python": "3.12.2",
    "numpy": "2.1.3",
    "sklearn": "1.6.1",
}
THREAD_ENVIRONMENT = dict(screen.THREAD_ENVIRONMENT)
LOCKED_HASHES = {
    "protocol_sha256": "2b4289636157fcdcd552c688cd3212c6311a8fddf128766d48fe126ab03d1069",
    "raw_results_sha256": "aa34175fa1d06ea4721561a0b1011c5a97267e5838b0ce044749922f3524f2a1",
    "run_manifest_sha256": "e3f6d06c4094aa05c5ff0fefd4def96ecebcc32e63b39551aee715daa809a86e",
    "analysis_summary_sha256": "19fff8bb9874a871d0c1c14b483e718dd50d90408b6ace93e699c874c0c451ba",
    "analysis_decision_sha256": "612f120b7ec09a4c00a7450c39a2c8e0466433b09e66ff1d175ec8c09eff6b15",
    "analysis_manifest_sha256": "a2476dcc2b112321be8ca76d6569bd4b25091ca7249a931d0364402833511a02",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_protocol() -> tuple[str, dict[str, Any]]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SIDECAR.is_file():
        raise RuntimeError("full Food protocol and sidecar must exist before outcomes")
    observed = sha256_path(PROTOCOL_PATH)
    expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or observed != expected:
        raise RuntimeError("full Food protocol sidecar mismatch")
    protocol = _read_json(PROTOCOL_PATH)
    if protocol.get("status") != "frozen_before_full_outcomes":
        raise RuntimeError("full Food protocol is not frozen before outcomes")
    grid = protocol.get("grid")
    expected_grid = {
        "models": list(MODELS),
        "replicates": list(REPLICATES),
        "arms": list(ARMS),
        "budgets_per_class": list(BUDGETS),
        "folds": FOLDS,
        "k_per_class_max": int(food101.K),
        "panel_count": 600,
        "executed_methods": list(METHODS),
        "executed_method_row_count": 1800,
        "archived_refined_row_count": 600,
        "reference_row_count": 600,
    }
    if grid != expected_grid:
        raise RuntimeError("full Food protocol grid mismatch")
    locked = protocol.get("locked_screen")
    expected_locked = {
        "selected_candidate": "FUSED3",
        "decision_status": "pass_for_full_food_design",
        **LOCKED_HASHES,
    }
    if locked != expected_locked:
        raise RuntimeError("full Food protocol locked-screen surface mismatch")
    if protocol.get("environment") != {
        **EXPECTED_ENVIRONMENT,
        "thread_controls": THREAD_ENVIRONMENT,
        "reason": "exactly preserve the valid locked FUSED3 screen environment",
    }:
        raise RuntimeError("full Food protocol environment mismatch")
    return observed, protocol


def _thread_environment() -> dict[str, str]:
    observed = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT}
    if observed != THREAD_ENVIRONMENT:
        raise RuntimeError(
            f"one-thread environment mismatch: expected {THREAD_ENVIRONMENT!r}, got {observed!r}"
        )
    return {key: str(value) for key, value in observed.items()}


def _validate_runtime_environment() -> dict[str, str]:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    if observed != EXPECTED_ENVIRONMENT:
        raise RuntimeError(
            f"full Food runtime environment mismatch: expected {EXPECTED_ENVIRONMENT!r}, got {observed!r}"
        )
    return observed


def _validate_screen_lock() -> dict[str, Any]:
    paths = {
        "raw_results_sha256": SCREEN_RUN_DIR / "raw_results.json",
        "run_manifest_sha256": SCREEN_RUN_DIR / "manifest.json",
        "analysis_summary_sha256": SCREEN_ANALYSIS_DIR / "summary.json",
        "analysis_decision_sha256": SCREEN_ANALYSIS_DIR / "decision.json",
        "analysis_manifest_sha256": SCREEN_ANALYSIS_DIR / "manifest.json",
    }
    for key, path in paths.items():
        if sha256_path(path) != LOCKED_HASHES[key]:
            raise RuntimeError(f"locked screen {key} mismatch")
    decision = _read_json(SCREEN_ANALYSIS_DIR / "decision.json")
    if (
        decision.get("status") != "pass_for_full_food_design"
        or decision.get("selected_candidate") != "FUSED3"
        or decision.get("full_replay_status") != "not_run_requires_separate_frozen_protocol"
    ):
        raise RuntimeError("locked screen decision does not authorize FUSED3 full design")
    screen_protocol = screen.sha256_path(screen.PROTOCOL_PATH)
    if screen_protocol != LOCKED_HASHES["protocol_sha256"]:
        raise RuntimeError("locked screen protocol bytes changed")
    return decision


def _source_paths() -> tuple[Path, ...]:
    local = (
        PACKAGE_DIR / "scalable.py",
        PACKAGE_DIR / "scalable_v2.py",
        PACKAGE_DIR / "fused_prototypes.py",
        PACKAGE_DIR / "fused_food_screen.py",
        PACKAGE_DIR / "fused_food_analysis.py",
        PACKAGE_DIR / "fused_food_analysis_erratum.py",
        PACKAGE_DIR / "fused_food_full.py",
        PACKAGE_DIR / "fused_food_full_analysis.py",
        screen.PROTOCOL_PATH,
        PROTOCOL_PATH,
        ROOT / "pyproject.toml",
    )
    return local + tuple(sorted((ROOT / "overlapindex").glob("*.py"))) + tuple(
        sorted((ROOT / "experiments" / "m50_backbone_ranking").glob("*.py"))
    )


def _git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _cache_identities() -> dict[str, dict[str, Any]]:
    output = {}
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
        raise FileNotFoundError(f"missing full Food sources: {missing!r}")
    local_hashes = {str(path): sha256_path(path) for path in paths}
    archived = {
        "driver": (Path(food101.DEFAULT_DRIVER), food101.DRIVER_SHA256),
        "source_result": (Path(food101.DEFAULT_RESULT), food101.RESULT_SHA256),
        "source_cohort": (Path(food101.DEFAULT_COHORT), food101.COHORT_SHA256),
        "prior_replay": (Path(food101.DEFAULT_PRIOR), food101.PRIOR_SHA256),
    }
    archived_hashes = {}
    for name, (path, expected) in archived.items():
        observed = sha256_path(path)
        if observed != expected:
            raise RuntimeError(f"archived {name} hash mismatch")
        archived_hashes[name] = observed
    return {
        "local_source_hashes": local_hashes,
        "code_identity_sha256": _hash_payload(local_hashes),
        "archived_source_hashes": archived_hashes,
        "locked_screen_hashes": dict(LOCKED_HASHES),
        "cache_identities": _cache_identities(),
        "repository": {
            "branch": _git_output("branch", "--show-current"),
            "experiment_commit": _git_output("rev-parse", "HEAD"),
            "develop_merge_base": _git_output("merge-base", "HEAD", "develop"),
            "dirty_status_recorded": _git_output("status", "--short"),
        },
    }


def _panel_id(model: str, replicate: int, arm: str, budget: int) -> str:
    return f"{model}__r{replicate}__{arm}__b{budget}"


def _method_order(panel_index: int) -> tuple[str, ...]:
    offset = int(panel_index) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def _execute_method(
    method: str, values: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, Any]:
    if method == "FUSED3":
        return screen.cross_fitted_candidate(
            values, labels, candidate_id="FUSED3", seed=int(seed)
        )
    if method == "LP-FULL":
        return food101._cross_fitted_score(
            values, labels, candidate="LP-FULL", seed=int(seed)
        )
    if method == "REFINED-OI":
        result = dict(
            food101._cross_fitted_score(values, labels, candidate="B", seed=int(seed))
        )
        result.update(
            {
                "candidate_id": "REFINED-OI",
                "candidate_name": "balanced_median_refined_overlap_index",
                "mode": "direct_refined_oi",
                "prototype_refinement": True,
            }
        )
        return result
    raise ValueError(f"unknown full Food method {method!r}")


def _signature(result: Mapping[str, Any]) -> dict[str, Any]:
    return screen._clock_free(
        {
            "candidate_id": result.get("candidate_id"),
            "candidate_name": result.get("candidate_name"),
            "mode": result.get("mode"),
            "prototype_refinement": result.get("prototype_refinement"),
            "score": result.get("score"),
            "refinement": result.get("refinement", {}),
            "conditioning": result.get("conditioning", {}),
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


def _method_row(
    result: Mapping[str, Any],
    *,
    model: str,
    replicate: int,
    arm: str,
    budget: int,
    position: int,
    order: Sequence[str],
    wall: float,
    cpu: float,
) -> dict[str, Any]:
    return {
        "stage": "full",
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
        "total_wall_seconds": float(wall),
        "total_cpu_seconds": float(cpu),
        "refinement": result.get("refinement", {}),
        "folds": list(result.get("folds", [])),
        "execution_position": int(position),
        "execution_order": list(order),
        "warmup_excluded": True,
        "status": "ok",
        "error": None,
    }


def _load_frozen_inputs() -> tuple[
    dict[tuple[str, int, str, int, str], float],
    dict[tuple[str, str], float],
    list[dict[str, Any]],
]:
    if sha256_path(PRIOR_REPLAY_PATH) != food101.PRIOR_SHA256:
        raise RuntimeError("archived replay hash mismatch")
    prior = _read_json(PRIOR_REPLAY_PATH)
    prior_lookup = {}
    archived_refined = []
    method_names = {
        "overlap_refined_cross_fitted": "REFINED-OI-ARCHIVED",
        "linear_probe_oof": "LP-FULL",
    }
    for row in prior.get("selector_rows", ()):
        if not isinstance(row, Mapping) or row.get("method") not in method_names:
            continue
        method = method_names[str(row["method"])]
        key = (
            str(row["backbone"]),
            int(row["replicate"]),
            str(row["arm"]),
            int(row["budget"]),
            method,
        )
        if key in prior_lookup:
            raise RuntimeError("archived replay duplicates a selector row")
        prior_lookup[key] = float(row["score"])
        if method == "REFINED-OI-ARCHIVED":
            archived_refined.append(
                {
                    "model": key[0],
                    "backbone": key[0],
                    "replicate": key[1],
                    "arm": key[2],
                    "budget": key[3],
                    "candidate_id": method,
                    "score": float(row["score"]),
                    "source": "hash_frozen_prior_replay",
                }
            )
    if len(prior_lookup) != 1200 or len(archived_refined) != 600:
        raise RuntimeError("archived replay comparator grid is incomplete")
    screen_raw = _read_json(SCREEN_RUN_DIR / "raw_results.json")
    screen_lookup = {
        (_panel_id(str(row["model"]), int(row["replicate"]), str(row["arm"]), int(row["budget"])), "FUSED3"): float(row["score"])
        for row in screen_raw.get("selector_rows", ())
        if isinstance(row, Mapping) and row.get("candidate_id") == "FUSED3"
    }
    if len(screen_lookup) != 30:
        raise RuntimeError("locked screen FUSED3 overlap grid is incomplete")
    return prior_lookup, screen_lookup, archived_refined


def _run_identity(protocol_sha: str, sources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "study": "fused_class_prototypes_food101_full_replay",
        "stage": "full",
        "protocol_sha256": protocol_sha,
        "source_identity": dict(sources),
        "locked_screen_decision_sha256": LOCKED_HASHES["analysis_decision_sha256"],
        "methods": list(METHODS),
        "models": list(MODELS),
        "replicates": list(REPLICATES),
        "arms": list(ARMS),
        "budgets": list(BUDGETS),
        "thread_environment": _thread_environment(),
        "environment": _validate_runtime_environment(),
    }


def run_full(output: Path, *, resume: bool = False) -> dict[str, Any]:
    protocol_sha, _protocol = validate_protocol()
    _validate_screen_lock()
    sources = source_identity()
    identity = _run_identity(protocol_sha, sources)
    identity_sha = _hash_payload(identity)
    prior_lookup, screen_lookup, archived_refined = _load_frozen_inputs()
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.is_file():
            raise RuntimeError("refusing non-empty full Food output without locked resume")
        existing = _read_json(manifest_path)
        if existing.get("artifact_status") != "running" or existing.get("run_identity") != identity:
            raise RuntimeError("full Food resume manifest identity mismatch")
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
    panels = [
        (model, int(replicate), arm, int(budget))
        for model in MODELS
        for replicate in REPLICATES
        for arm in ARMS
        for budget in BUDGETS
    ]
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    expected_names = {_panel_id(*panel) + ".json" for panel in panels}
    unexpected = {path.name for path in checkpoint_dir.iterdir()} - expected_names
    if unexpected:
        raise RuntimeError(f"unexpected full Food checkpoints: {sorted(unexpected)!r}")

    rows: list[dict[str, Any]] = []
    lp_parity: list[dict[str, Any]] = []
    refined_deltas: list[dict[str, Any]] = []
    screen_parity: list[dict[str, Any]] = []
    deterministic: dict[str, dict[str, Any]] = {}
    arm_lookup = {name: (lam, nu) for name, lam, nu in food101.ARMS}
    panel_index = 0
    for model in MODELS:
        matrix, cache_manifest = food101._load_cache(
            food101.DEFAULT_CACHE, model, sample_hash
        )
        for replicate in REPLICATES:
            selector_indices = np.asarray(roles[int(replicate)]["selector"], dtype=np.int64)
            raw = np.asarray(matrix[selector_indices], dtype=np.float32)
            target = labels[selector_indices]
            banks = driver._paired_split_banks(raw, target, seed=SEED + int(replicate) + 11)
            nested = driver._nested_stratified_indices(
                target, BUDGETS, seed=SEED + int(replicate) + 23
            )
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
                for budget in BUDGETS:
                    selected = np.asarray(nested[int(budget)], dtype=np.int64)
                    subset = np.asarray(transformed[selected], dtype=np.float32)
                    subset_target = target[selected]
                    panel_id = _panel_id(model, int(replicate), arm, int(budget))
                    order = _method_order(panel_index)
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
                        "labels_sha256": _hash_payload(subset_target.tolist()),
                    }
                    checkpoint = checkpoint_dir / f"{panel_id}.json"
                    if resume and checkpoint.is_file():
                        cached = _read_json(checkpoint)
                        if cached.get("artifact_status") != "completed" or cached.get("identity") != panel_identity:
                            raise RuntimeError(f"checkpoint identity/status mismatch for {panel_id}")
                        block = cached.get("rows")
                        if not isinstance(block, list) or len(block) != len(METHODS):
                            raise RuntimeError(f"checkpoint method block incomplete for {panel_id}")
                        if cached.get("lp_parity", {}).get("exact") is not True:
                            raise RuntimeError(f"cached LP parity failed for {panel_id}")
                        screen_record = cached.get("screen_parity")
                        if screen_record is not None and screen_record.get("exact") is not True:
                            raise RuntimeError(f"cached screen parity failed for {panel_id}")
                        rows.extend(dict(row) for row in block)
                        lp_parity.append(dict(cached["lp_parity"]))
                        refined_deltas.append(dict(cached["refined_delta"]))
                        if screen_record is not None:
                            screen_parity.append(dict(screen_record))
                        for method, record in cached.get("determinism", {}).items():
                            if method in deterministic and deterministic[method] != record:
                                raise RuntimeError("determinism evidence changed across checkpoints")
                            deterministic[method] = dict(record)
                        panel_index += 1
                        continue

                    block = []
                    local_determinism = {}
                    panel_seed = SEED + int(replicate)
                    for position, method in enumerate(order):
                        warmup = None
                        if method not in deterministic:
                            warmup = _execute_method(method, subset, subset_target, panel_seed)
                        started_wall, started_cpu = time.perf_counter(), time.process_time()
                        result = _execute_method(method, subset, subset_target, panel_seed)
                        wall = time.perf_counter() - started_wall
                        cpu = time.process_time() - started_cpu
                        if warmup is not None:
                            record = _determinism(warmup, result)
                            record.update({"candidate_id": method, "panel_id": panel_id})
                            if record["exact"] is not True:
                                raise RuntimeError(f"nondeterministic full Food method {method}")
                            deterministic[method] = record
                            local_determinism[method] = record
                        block.append(
                            _method_row(
                                result,
                                model=model,
                                replicate=int(replicate),
                                arm=arm,
                                budget=int(budget),
                                position=position,
                                order=order,
                                wall=wall,
                                cpu=cpu,
                            )
                        )
                    by_method = {str(row["candidate_id"]): row for row in block}
                    lp_expected = prior_lookup[(model, int(replicate), arm, int(budget), "LP-FULL")]
                    lp_current = float(by_method["LP-FULL"]["score"])
                    lp_record = {
                        "panel_id": panel_id,
                        "current_score": lp_current,
                        "archived_score": lp_expected,
                        "delta": lp_current - lp_expected,
                        "exact": lp_current == lp_expected,
                    }
                    refined_expected = prior_lookup[
                        (model, int(replicate), arm, int(budget), "REFINED-OI-ARCHIVED")
                    ]
                    refined_current = float(by_method["REFINED-OI"]["score"])
                    refined_record = {
                        "panel_id": panel_id,
                        "current_score": refined_current,
                        "archived_score": refined_expected,
                        "delta": refined_current - refined_expected,
                        "exact": refined_current == refined_expected,
                    }
                    screen_record = None
                    screen_key = (panel_id, "FUSED3")
                    if screen_key in screen_lookup:
                        current = float(by_method["FUSED3"]["score"])
                        expected = float(screen_lookup[screen_key])
                        screen_record = {
                            "panel_id": panel_id,
                            "current_score": current,
                            "screen_score": expected,
                            "delta": current - expected,
                            "exact": current == expected,
                        }
                    status = "completed"
                    if lp_record["exact"] is not True:
                        status = "stopped_lp_parity_failure"
                    elif screen_record is not None and screen_record["exact"] is not True:
                        status = "stopped_screen_parity_failure"
                    _atomic_json(
                        checkpoint,
                        {
                            "artifact_status": status,
                            "identity": panel_identity,
                            "rows": block,
                            "lp_parity": lp_record,
                            "refined_delta": refined_record,
                            "screen_parity": screen_record,
                            "determinism": local_determinism,
                        },
                    )
                    if status != "completed":
                        raise RuntimeError(f"full Food parity stop for {panel_id}: {status}")
                    rows.extend(block)
                    lp_parity.append(lp_record)
                    refined_deltas.append(refined_record)
                    if screen_record is not None:
                        screen_parity.append(screen_record)
                    panel_index += 1

    if (
        len(rows) != 1800
        or len(lp_parity) != 600
        or len(refined_deltas) != 600
        or len(screen_parity) != 30
    ):
        raise RuntimeError("full Food replay completed with an incomplete grid")
    if set(deterministic) != set(METHODS) or any(
        record.get("exact") is not True for record in deterministic.values()
    ):
        raise RuntimeError("full Food determinism evidence is incomplete")
    reference_rows = [dict(row) for row in source_result.get("reference_rows", ())]
    if len(reference_rows) != 600:
        raise RuntimeError("full Food reference grid is incomplete")
    payload = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": identity["study"],
        "stage": "full",
        "retrospective_development_only": True,
        "run_identity": identity,
        "run_identity_sha256": identity_sha,
        "determinism": deterministic,
        "lp_parity": lp_parity,
        "refined_historical_deltas": refined_deltas,
        "screen_parity": screen_parity,
        "reference_rows": reference_rows,
        "archived_refined_rows": archived_refined,
        "selector_rows": rows,
    }
    raw_path = output / "raw_results.json"
    _atomic_json(raw_path, payload)
    terminal = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": identity["study"],
        "stage": "full",
        "run_identity": identity,
        "run_identity_sha256": identity_sha,
        "raw_results_sha256": sha256_path(raw_path),
        "counts": {
            "panels": 600,
            "selector_rows": len(rows),
            "archived_refined_rows": len(archived_refined),
            "reference_rows": len(reference_rows),
            "lp_parity_rows": len(lp_parity),
            "refined_historical_delta_rows": len(refined_deltas),
            "screen_parity_rows": len(screen_parity),
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
    payload = run_full(args.output, resume=args.resume)
    print(canonical_json({"artifact_status": payload["artifact_status"], "stage": "full"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
