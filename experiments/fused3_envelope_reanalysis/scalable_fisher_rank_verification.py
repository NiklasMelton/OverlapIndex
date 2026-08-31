"""Counterbalanced rank/runtime verification for the scalable Fisher metric.

This post-outcome development diagnostic directly remeasures the frozen full
linear probe and rank-8/16/32 variants of the scalable shared-Fisher FUSED3
selector.  It neither changes the Confirmation V2 decision nor promotes a
candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from sklearn.utils.extmath import randomized_svd

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, recipes, statistics
from experiments.fused3_envelope_reanalysis import scalable_fisher_approximation as prior_sf
from experiments.fused3_envelope_reanalysis.five_dataset_low_k_refinement import (
    FROZEN,
    LINEAR_PROBE,
    _reference_lookup,
    _validate_comparison_rows,
    aggregate_envelope,
    aggregate_heads,
    contrast_interval,
    envelope_panel_metrics,
    head_panel_metrics,
)
from experiments.fused3_envelope_reanalysis.kmeans_k_sweep import (
    _canonical_json,
    _csv_bytes,
    _sha256,
)
from experiments.fused3_envelope_reanalysis.metric_aggregation_2x2 import (
    _frozen_score_lookup,
)
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "scalable_fisher_rank_verification_protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "scalable_fisher_rank_verification_protocol.sha256"

STUDY = "fused3_confirmation_v2_scalable_fisher_rank_runtime_verification"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
FULL_FISHER = prior_sf.FULL_FISHER
RANK8 = "FUSED3-LR8-FISHER-OI-U"
RANK16 = prior_sf.LOW_RANK
RANK32 = "FUSED3-LR32-FISHER-OI-U"
RANK_METHODS = (RANK8, RANK16, RANK32)
RANK_BY_METHOD: Mapping[str, int] = {RANK8: 8, RANK16: 16, RANK32: 32}
SCHEDULED_METHODS = (LINEAR_PROBE, *RANK_METHODS)
COMPARISON_METHODS = (FROZEN, LINEAR_PROBE, FULL_FISHER, *RANK_METHODS)
SCHEDULE_SEED = 2026083017
EXPECTED_K_BY_BUDGET: Mapping[int, int] = {32: 5, 64: 10}
ONE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}


def validate_design_protocol() -> tuple[str, dict[str, Any]]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SIDECAR.is_file():
        raise RuntimeError("rank verification protocol and sidecar must exist")
    observed = _sha256(PROTOCOL_PATH)
    expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or observed != expected:
        raise RuntimeError("rank verification protocol sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_post_outcome_before_execution":
        raise RuntimeError("rank verification protocol is not frozen")
    if protocol.get("scheduled_methods") != list(SCHEDULED_METHODS):
        raise RuntimeError("rank verification method table mismatch")
    if protocol.get("rank_candidates") != {
        method: rank for method, rank in RANK_BY_METHOD.items()
    }:
        raise RuntimeError("rank verification rank table mismatch")
    return observed, protocol


def _validate_one_thread_environment() -> dict[str, str]:
    observed = {name: os.environ.get(name) for name in ONE_THREAD_ENVIRONMENT}
    if observed != ONE_THREAD_ENVIRONMENT:
        raise RuntimeError(
            "rank verification requires the frozen one-thread environment; "
            f"observed {observed!r}"
        )
    return dict(ONE_THREAD_ENVIRONMENT)


def canonical_panel_keys() -> tuple[tuple[str, str, int, int, int], ...]:
    return tuple(
        (dataset_id, backbone, replicate, int(seed), int(budget))
        for dataset_id in statistics.DATASET_IDS
        for backbone in statistics.BACKBONES
        for replicate, seed in enumerate(statistics.REPLICATE_SEEDS)
        for budget in statistics.BUDGETS
    )


def execution_order(panel_index: int) -> tuple[str, ...]:
    if type(panel_index) is not int or not 0 <= panel_index < 500:
        raise ValueError("panel_index must be an integer in [0, 500)")
    seed_hash = hashlib.sha256(str(SCHEDULE_SEED).encode("ascii")).digest()
    offset = (int(seed_hash[0]) + panel_index) % len(SCHEDULED_METHODS)
    return tuple(
        SCHEDULED_METHODS[(position + offset) % len(SCHEDULED_METHODS)]
        for position in range(len(SCHEDULED_METHODS))
    )


def execution_position_counts() -> dict[str, list[int]]:
    counts = {method: [0] * len(SCHEDULED_METHODS) for method in SCHEDULED_METHODS}
    for panel_index in range(500):
        for position, method in enumerate(execution_order(panel_index)):
            counts[method][position] += 1
    return counts


def _rank_transform_state_sha256(
    *,
    center: np.ndarray,
    feature_scale: np.ndarray,
    nuisance_vectors: np.ndarray,
    nuisance_gains: np.ndarray,
    discriminant_basis: np.ndarray,
    classes: Sequence[Any],
    requested_rank: int,
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("center", center),
        ("feature_scale", feature_scale),
        ("nuisance_vectors", nuisance_vectors),
        ("nuisance_gains", nuisance_gains),
        ("discriminant_basis", discriminant_basis),
        ("classes", np.asarray([repr(value) for value in classes])),
    ):
        prior_sf._array_digest(digest, name, value)
    digest.update(f"low_rank_{requested_rank}".encode("ascii") + b"\0")
    digest.update(str(requested_rank).encode("ascii") + b"\0")
    digest.update(str(prior_sf.RANDOMIZED_POWER_ITERATIONS).encode("ascii") + b"\0")
    digest.update(str(prior_sf.RANDOMIZED_OVERSAMPLES).encode("ascii"))
    return digest.hexdigest()


def fit_rank_transform(
    values: Any,
    target: Any,
    *,
    seed: int,
    requested_rank: int,
) -> dict[str, Any]:
    """Fit one rank-specific transform using the exact prior rank-16 recipe."""

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must be a non-empty matrix")
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise ValueError("target must be a scalar vector aligned with values")
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    if type(requested_rank) is not int or requested_rank not in set(RANK_BY_METHOD.values()):
        raise ValueError("requested_rank must be one of (8, 16, 32)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite")

    started = time.perf_counter()
    classes, encoded = prior_sf._first_observed_encoding(labels)
    if len(classes) < 2:
        raise ValueError("rank Fisher fitting requires at least two classes")
    output_rank = min(len(classes) - 1, matrix.shape[1])
    values64 = matrix.astype(np.float64)
    counts = np.bincount(encoded, minlength=len(classes)).astype(np.int64)
    if np.any(counts < 2):
        raise ValueError("every class requires at least two training rows")
    class_means = np.vstack(
        [np.mean(values64[encoded == position], axis=0) for position in range(len(classes))]
    )
    global_mean = np.mean(values64, axis=0)
    residual = values64 - class_means[encoded]
    pooled_variance_raw = np.mean(np.square(residual), axis=0)
    positive = pooled_variance_raw[pooled_variance_raw > 0.0]
    scale_reference = float(np.median(positive)) if positive.size else 1.0
    variance_floor = max(
        scale_reference * prior_sf.RELATIVE_VARIANCE_FLOOR,
        prior_sf.ABSOLUTE_VARIANCE_FLOOR,
    )
    pooled_variance = np.maximum(pooled_variance_raw, variance_floor)
    class_weights = np.sqrt(counts.astype(np.float64) / float(len(matrix)))
    contrasts = (class_means - global_mean[None, :]) * class_weights[:, None]
    diagonal_scale = np.reciprocal(np.sqrt(pooled_variance))
    standardized_contrasts = contrasts * diagonal_scale[None, :]
    standardized_residual = np.asarray(residual * diagonal_scale[None, :], dtype=np.float64)
    effective_rank = min(
        requested_rank,
        matrix.shape[1],
        len(matrix) - len(classes),
    )
    _left, singular, right = randomized_svd(
        standardized_residual,
        n_components=effective_rank,
        n_iter=prior_sf.RANDOMIZED_POWER_ITERATIONS,
        n_oversamples=prior_sf.RANDOMIZED_OVERSAMPLES,
        random_state=int(seed),
        flip_sign=True,
    )
    nuisance_vectors = np.asarray(right.T, dtype=np.float64)
    degrees_of_freedom = max(1, len(matrix) - len(classes))
    eigenvalues = np.square(np.asarray(singular, dtype=np.float64)) / float(
        degrees_of_freedom
    )
    nuisance_gains = np.reciprocal(np.sqrt(np.maximum(eigenvalues, 1.0))) - 1.0
    corrected_contrasts = standardized_contrasts + (
        (standardized_contrasts @ nuisance_vectors) * nuisance_gains[None, :]
    ) @ nuisance_vectors.T
    basis = prior_sf._basis(corrected_contrasts, output_rank)
    center_ro = prior_sf._read_only(global_mean, dtype=np.float64)
    scale_ro = prior_sf._read_only(diagonal_scale, dtype=np.float64)
    vectors_ro = prior_sf._read_only(nuisance_vectors, dtype=np.float64)
    gains_ro = prior_sf._read_only(nuisance_gains, dtype=np.float64)
    basis_ro = prior_sf._read_only(basis, dtype=np.float64)
    transform = prior_sf.ScalableFisherTransform(
        center=center_ro,
        feature_scale=scale_ro,
        nuisance_vectors=vectors_ro,
        nuisance_gains=gains_ro,
        discriminant_basis=basis_ro,
        classes=classes,
        approximation=f"low_rank_{requested_rank}",
        variance_floor=float(variance_floor),
        state_sha256=_rank_transform_state_sha256(
            center=center_ro,
            feature_scale=scale_ro,
            nuisance_vectors=vectors_ro,
            nuisance_gains=gains_ro,
            discriminant_basis=basis_ro,
            classes=classes,
            requested_rank=requested_rank,
        ),
    )
    return {
        "transform": transform,
        "fit_wall_seconds": float(time.perf_counter() - started),
        "requested_rank": int(requested_rank),
        "effective_rank": int(effective_rank),
        "output_dimension": int(output_rank),
        "variance_floor": float(variance_floor),
        "variance_floor_count": int(np.count_nonzero(pooled_variance_raw < variance_floor)),
        "nuisance_eigenvalue_min": float(np.min(eigenvalues)),
        "nuisance_eigenvalue_max": float(np.max(eigenvalues)),
    }


def crossfit_rank_candidate(
    values: Any,
    target: Any,
    *,
    seed: int,
    requested_rank: int,
) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    if matrix.ndim != 2 or labels.shape != (len(matrix),):
        raise ValueError("matrix and scalar targets must align")
    normalization_started = time.perf_counter()
    normalized = food101._row_l2(matrix)
    normalization_wall = time.perf_counter() - normalization_started
    folds = food101._stratified_folds(labels, n_splits=5, seed=int(seed))
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        k_per_class = fused_food_screen._k_per_class(labels, train)
        fitted = fit_rank_transform(
            normalized[train],
            labels[train],
            seed=fold_seed,
            requested_rank=requested_rank,
        )
        transform = fitted["transform"]
        transform_started = time.perf_counter()
        transformed_train = transform.transform(normalized[train])
        transformed_holdout = transform.transform(normalized[holdout])
        transform_wall = time.perf_counter() - transform_started
        selector = fused_food_screen._build_selector("FUSED3", k_per_class, fold_seed)
        selector_started = time.perf_counter()
        selector.fit(transformed_train, labels[train])
        selector_fit_wall = time.perf_counter() - selector_started
        state_before = fused_food_screen._numerical_state_sha256(selector)
        score_started = time.perf_counter()
        score = float(selector.score_fixed(transformed_holdout, labels[holdout]))
        score_wall = time.perf_counter() - score_started
        if fused_food_screen._numerical_state_sha256(selector) != state_before:
            raise RuntimeError("rank Fisher scoring mutated fitted FUSED3 state")
        fold_rows.append(
            {
                "fold": int(fold),
                "fold_seed": int(fold_seed),
                "train_size": int(len(train)),
                "holdout_size": int(len(holdout)),
                "k_min": int(min(k_per_class.values())),
                "k_max": int(max(k_per_class.values())),
                "score": score,
                "normalization_wall_seconds": float(normalization_wall if fold == 0 else 0.0),
                "fit_transform_wall_seconds": float(fitted["fit_wall_seconds"] + transform_wall),
                "selector_fit_wall_seconds": float(selector_fit_wall),
                "score_fixed_wall_seconds": float(score_wall),
                "requested_rank": int(requested_rank),
                "effective_rank": int(fitted["effective_rank"]),
                "output_dimension": int(fitted["output_dimension"]),
                "variance_floor": float(fitted["variance_floor"]),
                "variance_floor_count": int(fitted["variance_floor_count"]),
                "nuisance_eigenvalue_min": float(fitted["nuisance_eigenvalue_min"]),
                "nuisance_eigenvalue_max": float(fitted["nuisance_eigenvalue_max"]),
                "transform_state_sha256": transform.state_sha256,
                "selector_state_sha256": state_before,
                "state_unchanged": True,
            }
        )
    return {
        "score": float(np.mean([row["score"] for row in fold_rows])),
        "folds": fold_rows,
        "requested_rank": int(requested_rank),
    }


def _minimal_lp_folds(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in result["folds"]:
        output.append(
            {
                "fold": int(row["fold"]),
                "fold_seed": int(row["seed"]),
                "train_size": int(row["train_size"]),
                "holdout_size": int(row["holdout_size"]),
                "score": float(row["score"]),
                "fit_wall_seconds": float(row["fit_wall_seconds"]),
                "fit_cpu_seconds": float(row["fit_cpu_seconds"]),
                "score_fixed_wall_seconds": float(row["predict_wall_seconds"]),
                "score_fixed_cpu_seconds": float(row["predict_cpu_seconds"]),
                "candidate_config_sha256": str(row["candidate_config_sha256"]),
            }
        )
    return output


def execute_method(
    method: str,
    values: Any,
    labels: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    if method == LINEAR_PROBE:
        raw = recipes.execute_selector(method, values, labels, seed=int(seed))
        score = float(raw["score"])
        folds = _minimal_lp_folds(raw)
        recipe_identity = recipes.candidate_recipe(method)
    elif method in RANK_BY_METHOD:
        raw = crossfit_rank_candidate(
            values,
            labels,
            seed=int(seed),
            requested_rank=int(RANK_BY_METHOD[method]),
        )
        score = float(raw["score"])
        folds = [dict(row) for row in raw["folds"]]
        recipe_identity = {
            "candidate_id": method,
            "base_selector": recipes.candidate_recipe("FUSED3"),
            "requested_nuisance_rank": int(RANK_BY_METHOD[method]),
            "refinement": False,
            "shared_algorithm": "scalable_fisher_rank_verification_protocol.json",
        }
    else:
        raise ValueError(f"method must be one of {SCHEDULED_METHODS!r}; got {method!r}")
    cpu_elapsed = time.process_time() - cpu_started
    wall_elapsed = time.perf_counter() - wall_started
    if not np.isfinite(score):
        raise RuntimeError(f"{method} produced a non-finite score")
    return {
        "candidate_id": method,
        "score": score,
        "outer_wall_seconds": float(max(0.0, wall_elapsed)),
        "outer_cpu_seconds": float(max(0.0, cpu_elapsed)),
        "folds": folds,
        "recipe_identity": recipe_identity,
        "recipe_sha256": hashlib.sha256(
            _canonical_json(recipe_identity).encode("utf-8")
        ).hexdigest(),
    }


def _clock_free_signature(result: Mapping[str, Any]) -> str:
    folds = []
    for row in result["folds"]:
        folds.append(
            {
                key: value
                for key, value in row.items()
                if not key.endswith("_seconds")
            }
        )
    payload = {
        "candidate_id": result["candidate_id"],
        "score": result["score"],
        "recipe_sha256": result["recipe_sha256"],
        "folds": folds,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_hashed_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((root / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_status") != "completed":
        raise ValueError("source diagnostic artifact is not completed")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("source diagnostic artifact has no file table")
    for name, descriptor in files.items():
        path = root / str(name)
        if (
            not path.is_file()
            or _sha256(path) != descriptor.get("sha256")
            or path.stat().st_size != descriptor.get("size_bytes")
        ):
            raise ValueError(f"source diagnostic hash mismatch for {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return manifest, summary


def load_prior_rank16_rows(
    artifact: os.PathLike[str] | str,
    *,
    frozen_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(artifact)
    manifest, summary = _validate_hashed_bundle(root)
    if summary.get("study") != prior_sf.STUDY:
        raise ValueError("unexpected prior scalable-Fisher study")
    if summary.get("input_identity") != dict(frozen_identity):
        raise ValueError("prior scalable-Fisher identity differs from Confirmation V2")
    rows = [
        dict(row)
        for row in summary["state_rows"]
        if row.get("candidate_id") == RANK16
    ]
    expected = {
        (dataset, backbone, int(seed), int(budget))
        for dataset in statistics.DATASET_IDS
        for backbone in statistics.BACKBONES
        for seed in statistics.REPLICATE_SEEDS
        for budget in statistics.BUDGETS
    }
    keys = {
        (str(row["dataset_id"]), str(row["backbone"]), int(row["replicate_seed"]), int(row["budget"]))
        for row in rows
    }
    if len(rows) != 500 or keys != expected:
        raise ValueError("prior rank-16 grid is incomplete")
    return rows, {
        "manifest_sha256": _sha256(root / "diagnostic_manifest.json"),
        "summary_sha256": _sha256(root / "summary.json"),
        "design_protocol_sha256": summary["design_protocol_sha256"],
        "files": manifest["files"],
    }


def _runtime_ratios(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["dataset_id"]),
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        ): row
        for row in rows
    }
    if len(lookup) != 2000:
        raise RuntimeError("measured runtime row grid is incomplete or duplicated")
    output: list[dict[str, Any]] = []
    for dataset_id in statistics.DATASET_IDS:
        for budget in statistics.BUDGETS:
            for method in RANK_METHODS:
                call_wall: list[float] = []
                call_cpu: list[float] = []
                panel_wall: list[float] = []
                panel_cpu: list[float] = []
                for seed in statistics.REPLICATE_SEEDS:
                    candidate_wall = candidate_cpu = probe_wall = probe_cpu = 0.0
                    for backbone in statistics.BACKBONES:
                        candidate = lookup[(dataset_id, backbone, int(seed), int(budget), method)]
                        probe = lookup[(dataset_id, backbone, int(seed), int(budget), LINEAR_PROBE)]
                        cwall = float(candidate["outer_wall_seconds"])
                        pwall = float(probe["outer_wall_seconds"])
                        ccpu = float(candidate["outer_cpu_seconds"])
                        pcpu = float(probe["outer_cpu_seconds"])
                        call_wall.append(cwall / pwall)
                        call_cpu.append(ccpu / pcpu)
                        candidate_wall += cwall
                        probe_wall += pwall
                        candidate_cpu += ccpu
                        probe_cpu += pcpu
                    panel_wall.append(candidate_wall / probe_wall)
                    panel_cpu.append(candidate_cpu / probe_cpu)
                output.append(
                    {
                        "dataset_id": dataset_id,
                        "budget": int(budget),
                        "candidate_id": method,
                        "call_count": len(call_wall),
                        "full_panel_count": len(panel_wall),
                        "median_call_wall_ratio_to_LP_FULL": float(np.median(call_wall)),
                        "median_call_cpu_ratio_to_LP_FULL": float(np.median(call_cpu)),
                        "median_full_panel_wall_ratio_to_LP_FULL": float(np.median(panel_wall)),
                        "median_full_panel_cpu_ratio_to_LP_FULL": float(np.median(panel_cpu)),
                        "candidate_faster_call_count": int(np.count_nonzero(np.asarray(call_wall) < 1.0)),
                        "candidate_faster_panel_count": int(np.count_nonzero(np.asarray(panel_wall) < 1.0)),
                    }
                )
    return output


def _overall_runtime(runtime_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in RANK_METHODS:
        selected = [row for row in runtime_rows if row["candidate_id"] == method]
        output.append(
            {
                "candidate_id": method,
                "median_of_dataset_budget_call_wall_ratios": float(
                    np.median([row["median_call_wall_ratio_to_LP_FULL"] for row in selected])
                ),
                "median_of_dataset_budget_full_panel_wall_ratios": float(
                    np.median([row["median_full_panel_wall_ratio_to_LP_FULL"] for row in selected])
                ),
                "all_dataset_budget_full_panels_faster": bool(
                    all(row["candidate_faster_panel_count"] == row["full_panel_count"] for row in selected)
                ),
            }
        )
    return output


def analyze_rank_verification(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    full_fisher_artifact: os.PathLike[str] | str,
    prior_scalable_artifact: os.PathLike[str] | str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    protocol_sha256, protocol = validate_design_protocol()
    thread_environment = _validate_one_thread_environment()
    if any(counts != [125, 125, 125, 125] for counts in execution_position_counts().values()):
        raise RuntimeError("measured execution schedule is not exactly position-balanced")
    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    source_selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    source_score_lookup = _frozen_score_lookup(source_selectors)
    source_lp_lookup = {
        (str(row["dataset_id"]), str(row["backbone"]), int(row["replicate_seed"]), int(row["budget"])): float(row["score"])
        for row in source_selectors
        if row["candidate_id"] == LINEAR_PROBE
    }
    input_identity = {
        key: verified["raw"][key]
        for key in (
            "protocol_sha256",
            "code_identity_sha256",
            "run_identity_sha256",
            "audited_registry_sha256",
            "lineage_lock_sha256",
        )
    }
    full_fisher_rows, full_fisher_identity = prior_sf.load_full_fisher_rows(
        full_fisher_artifact, frozen_identity=input_identity
    )
    prior_rank16_rows, prior_rank16_identity = load_prior_rank16_rows(
        prior_scalable_artifact, frozen_identity=input_identity
    )
    prior_rank16_lookup = {
        (str(row["dataset_id"]), str(row["backbone"]), int(row["replicate_seed"]), int(row["budget"])): float(row["score"])
        for row in prior_rank16_rows
    }
    root = Path(registry_root)
    registry = json.loads((root / "audited_registry.json").read_text(encoding="utf-8"))
    registry_rows = {str(row["dataset_id"]): row for row in registry["datasets"]}
    if set(registry_rows) != set(statistics.DATASET_IDS):
        raise RuntimeError("audited registry dataset set mismatch")
    _reference_lookup(references)

    measured_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    warmup_rows: list[dict[str, Any]] = []
    warmups: dict[tuple[str, int, str], dict[str, Any]] = {}
    parity_lp: list[dict[str, Any]] = []
    parity_rank16: list[dict[str, Any]] = []
    panel_keys = canonical_panel_keys()
    for panel_index, (dataset_id, backbone, replicate, seed, budget) in enumerate(panel_keys):
        panel = datasets.load_panel(
            registry_rows[dataset_id],
            backbone,
            int(seed),
            int(budget),
            registry_root=root,
        )
        order = execution_order(panel_index)
        is_warmup_panel = backbone == statistics.BACKBONES[0] and replicate == 0
        if is_warmup_panel:
            for method in SCHEDULED_METHODS:
                warmup = execute_method(
                    method,
                    panel["training_values"],
                    panel["training_labels"],
                    seed=int(seed),
                )
                warmups[(dataset_id, int(budget), method)] = warmup
        for position, method in enumerate(order):
            result = execute_method(
                method,
                panel["training_values"],
                panel["training_labels"],
                seed=int(seed),
            )
            key = (dataset_id, backbone, int(seed), int(budget))
            if method == LINEAR_PROBE:
                expected = source_lp_lookup[key]
                if float(result["score"]) != expected:
                    raise RuntimeError(f"LP-FULL exact parity failed at {key!r}")
                parity_lp.append(
                    {
                        "dataset_id": dataset_id,
                        "backbone": backbone,
                        "replicate_seed": int(seed),
                        "budget": int(budget),
                        "source_score": expected,
                        "measured_score": float(result["score"]),
                        "exact_match": True,
                    }
                )
            elif method == RANK16:
                expected = prior_rank16_lookup[key]
                if float(result["score"]) != expected:
                    raise RuntimeError(f"rank-16 exact parity failed at {key!r}")
                parity_rank16.append(
                    {
                        "dataset_id": dataset_id,
                        "backbone": backbone,
                        "replicate_seed": int(seed),
                        "budget": int(budget),
                        "source_score": expected,
                        "measured_score": float(result["score"]),
                        "exact_match": True,
                    }
                )
            if is_warmup_panel:
                warmup = warmups[(dataset_id, int(budget), method)]
                warmup_signature = _clock_free_signature(warmup)
                measured_signature = _clock_free_signature(result)
                if warmup_signature != measured_signature:
                    raise RuntimeError("excluded warmup and measured deterministic result differ")
                warmup_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "budget": int(budget),
                        "candidate_id": method,
                        "backbone": backbone,
                        "replicate_seed": int(seed),
                        "warmup_signature_sha256": warmup_signature,
                        "measured_signature_sha256": measured_signature,
                        "exact_match": True,
                        "excluded_from_summaries": True,
                    }
                )
            measured_rows.append(
                {
                    "dataset_id": dataset_id,
                    "backbone": backbone,
                    "replicate": int(replicate),
                    "replicate_seed": int(seed),
                    "budget": int(budget),
                    "candidate_id": method,
                    "requested_rank": RANK_BY_METHOD.get(method),
                    "score": float(result["score"]),
                    "outer_wall_seconds": float(result["outer_wall_seconds"]),
                    "outer_cpu_seconds": float(result["outer_cpu_seconds"]),
                    "execution_position": int(position),
                    "execution_order": list(order),
                    "recipe_sha256": result["recipe_sha256"],
                    "warmup_excluded": True,
                }
            )
            order_rows.append(
                {
                    "panel_index": int(panel_index),
                    "dataset_id": dataset_id,
                    "backbone": backbone,
                    "replicate_seed": int(seed),
                    "budget": int(budget),
                    "candidate_id": method,
                    "execution_position": int(position),
                }
            )
            fold_rows.extend(
                {
                    "dataset_id": dataset_id,
                    "backbone": backbone,
                    "replicate": int(replicate),
                    "replicate_seed": int(seed),
                    "budget": int(budget),
                    "candidate_id": method,
                    **row,
                }
                for row in result["folds"]
            )
        if progress and (panel_index + 1) % 10 == 0:
            print(f"completed {panel_index + 1}/500 panels", flush=True)

    if (
        len(measured_rows) != 2000
        or len(fold_rows) != 10000
        or len(order_rows) != 2000
        or len(warmup_rows) != 40
        or len(parity_lp) != 500
        or len(parity_rank16) != 500
    ):
        raise RuntimeError("rank verification output grid is incomplete")
    observed_positions = {
        method: [
            sum(
                row["candidate_id"] == method and row["execution_position"] == position
                for row in order_rows
            )
            for position in range(4)
        ]
        for method in SCHEDULED_METHODS
    }
    if any(counts != [125, 125, 125, 125] for counts in observed_positions.values()):
        raise RuntimeError("observed execution positions are not exactly balanced")

    frozen_rows = [
        {
            "dataset_id": str(row["dataset_id"]),
            "backbone": str(row["backbone"]),
            "replicate": int(row["replicate"]),
            "replicate_seed": int(row["replicate_seed"]),
            "budget": int(row["budget"]),
            "candidate_id": str(row["candidate_id"]),
            "score": float(row["score"]),
            "total_wall_seconds": float(row["total_wall_seconds"]),
            "source": "immutable_confirmation_v2",
        }
        for row in source_selectors
        if row["candidate_id"] == FROZEN
    ]
    measured_comparison = [
        {
            **{key: row[key] for key in ("dataset_id", "backbone", "replicate", "replicate_seed", "budget", "candidate_id", "score")},
            "total_wall_seconds": float(row["outer_wall_seconds"]),
            "source": "direct_counterbalanced_measurement",
        }
        for row in measured_rows
    ]
    comparison_rows = [*frozen_rows, *full_fisher_rows, *measured_comparison]
    _validate_comparison_rows(comparison_rows, methods=COMPARISON_METHODS)
    envelope = envelope_panel_metrics(comparison_rows, references, methods=COMPARISON_METHODS)
    heads = head_panel_metrics(comparison_rows, references, methods=COMPARISON_METHODS)
    dataset_aggregate, overall = aggregate_envelope(envelope, methods=COMPARISON_METHODS)
    head_aggregate = aggregate_heads(heads, methods=COMPARISON_METHODS)
    runtime_ratios = _runtime_ratios(measured_rows)
    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "primary_estimand": "frozen_four_head_best_head_envelope_backbone_ranking",
        "head_library": list(statistics.HEADS),
        "comparison_methods": list(COMPARISON_METHODS),
        "scheduled_methods": list(SCHEDULED_METHODS),
        "design_protocol_sha256": protocol_sha256,
        "design": protocol,
        "thread_environment": thread_environment,
        "execution_position_counts": observed_positions,
        "measured_rows": measured_rows,
        "fold_rows": fold_rows,
        "execution_order_rows": order_rows,
        "warmup_verification_rows": warmup_rows,
        "LP_FULL_parity_rows": parity_lp,
        "rank16_parity_rows": parity_rank16,
        "panel_metrics": envelope,
        "dataset_aggregate": dataset_aggregate,
        "overall_aggregate": overall,
        "head_panel_metrics": heads,
        "head_aggregate": head_aggregate,
        "runtime_ratios": runtime_ratios,
        "overall_runtime": _overall_runtime(runtime_ratios),
        "regret_contrasts": [
            contrast_interval(envelope, candidate_id=method, comparator_id=comparator)
            for method in RANK_METHODS
            for comparator in (LINEAR_PROBE, FULL_FISHER)
        ],
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "post_outcome_limit": (
            "All five datasets and outcomes were previously inspected. This is "
            "retrospective rank/runtime verification and cannot confirm a candidate."
        ),
        "input_hashes": dict(verified["input_hashes"]),
        "input_identity": input_identity,
        "full_fisher_source_identity": full_fisher_identity,
        "prior_rank16_source_identity": prior_rank16_identity,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Scalable Fisher rank and counterbalanced-runtime verification",
        "",
        "Status: **post-outcome development diagnostic only**. No promotion or confirmation reinterpretation is made.",
        "",
        "LP-FULL and ranks 8/16/32 were measured directly in one exactly position-balanced schedule. Warmups were excluded.",
        "",
        "## Best-head-envelope ranking",
        "",
        "| Selector | Regret (pp) | Exact best | Within 1 pp | Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["overall_aggregate"]:
        lines.append(
            f"| {row['candidate_id']} | {row['equal_dataset_mean_regret_pp']:.3f} | "
            f"{row['equal_dataset_exact_best_rate']:.3f} | "
            f"{row['equal_dataset_within_one_pp_rate']:.3f} | "
            f"{row['equal_dataset_mean_spearman']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Direct runtime versus LP-FULL",
            "",
            "Ratios below 1 favor the Fisher/OI candidate. Full-panel ratios sum all 10 backbone calls before division.",
            "",
            "| Candidate | Median cell call ratio | Median cell full-panel ratio | Faster in every cell panel? |",
            "|---|---:|---:|---|",
        ]
    )
    for row in summary["overall_runtime"]:
        lines.append(
            f"| {row['candidate_id']} | {row['median_of_dataset_budget_call_wall_ratios']:.3f}x | "
            f"{row['median_of_dataset_budget_full_panel_wall_ratios']:.3f}x | "
            f"{row['all_dataset_budget_full_panels_faster']} |"
        )
    lines.extend(
        [
            "",
            "## Parity and execution checks",
            "",
            f"- LP-FULL exact source parity: {len(summary['LP_FULL_parity_rows'])}/500.",
            f"- Rank-16 exact prior-artifact parity: {len(summary['rank16_parity_rows'])}/500.",
            f"- Excluded warmup checks: {len(summary['warmup_verification_rows'])}/40.",
            f"- Per-method position counts: `{summary['execution_position_counts']}`.",
            "",
            "## Regret contrasts",
            "",
            "Negative estimates favor the rank candidate.",
            "",
            "| Candidate | Comparator | Estimate (pp) | Descriptive 95% interval |",
            "|---|---|---:|---:|",
        ]
    )
    for row in summary["regret_contrasts"]:
        lines.append(
            f"| {row['candidate_id']} | {row['comparator_id']} | {row['estimate']:+.3f} | "
            f"[{row['lower_95']:+.3f}, {row['upper_95']:+.3f}] |"
        )
    lines.extend(["", "The immutable Confirmation V2 rejection remains unchanged.", ""])
    return "\n".join(lines)


def _fold_csv_payloads(rows: Sequence[Mapping[str, Any]]) -> dict[str, bytes]:
    lp_rows = [dict(row) for row in rows if row["candidate_id"] == LINEAR_PROBE]
    rank_rows = [dict(row) for row in rows if row["candidate_id"] in RANK_METHODS]
    if len(lp_rows) + len(rank_rows) != len(rows):
        raise ValueError("fold rows contain an unknown candidate")
    return {
        "LP_FULL_fold_rows.csv": _csv_bytes(lp_rows),
        "rank_fold_rows.csv": _csv_bytes(rank_rows),
    }


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    full_fisher_artifact: os.PathLike[str] | str,
    prior_scalable_artifact: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_rank_verification(
        full_artifact,
        registry_root,
        full_fisher_artifact,
        prior_scalable_artifact,
        progress=progress,
    )
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "measured_rows.csv": _csv_bytes(summary["measured_rows"]),
        "execution_order_rows.csv": _csv_bytes(summary["execution_order_rows"]),
        "warmup_verification_rows.csv": _csv_bytes(summary["warmup_verification_rows"]),
        "LP_FULL_parity_rows.csv": _csv_bytes(summary["LP_FULL_parity_rows"]),
        "rank16_parity_rows.csv": _csv_bytes(summary["rank16_parity_rows"]),
        "panel_metrics.csv": _csv_bytes(summary["panel_metrics"]),
        "dataset_aggregate.csv": _csv_bytes(summary["dataset_aggregate"]),
        "overall_aggregate.csv": _csv_bytes(summary["overall_aggregate"]),
        "head_panel_metrics.csv": _csv_bytes(summary["head_panel_metrics"]),
        "head_aggregate.csv": _csv_bytes(summary["head_aggregate"]),
        "runtime_ratios.csv": _csv_bytes(summary["runtime_ratios"]),
        "overall_runtime.csv": _csv_bytes(summary["overall_runtime"]),
        "regret_contrasts.csv": _csv_bytes(summary["regret_contrasts"]),
        "report.md": render_report(summary).encode("utf-8"),
    }
    payloads.update(_fold_csv_payloads(summary["fold_rows"]))
    manifest = {
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "design_protocol_sha256": summary["design_protocol_sha256"],
        "files": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            for name, content in payloads.items()
        },
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
    }
    payloads["diagnostic_manifest.json"] = (_canonical_json(manifest) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name, content in payloads.items():
            (staging / name).write_bytes(content)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for name, descriptor in manifest["files"].items():
        if _sha256(destination / name) != descriptor["sha256"]:
            raise RuntimeError("written rank verification artifact hash mismatch")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--full-fisher", type=Path, required=True)
    parser.add_argument("--prior-scalable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_bundle(
        args.input,
        args.registry_root,
        args.full_fisher,
        args.prior_scalable,
        args.output,
        progress=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
