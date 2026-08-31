"""Five-dataset FUSED3 metric-by-aggregation diagnostic.

This post-outcome diagnostic keeps the exact frozen FUSED3 prototypes and
global relevance weights inside every training fold.  It crosses two residual
metrics (pooled/shared diagonal versus class-conditional diagonal) with two
held-out aggregations (exact OI events versus prototype-discriminant OOF
accuracy).  It cannot promote, reselect, or alter Confirmation V2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics
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
    runtime_summary,
)
from experiments.fused3_envelope_reanalysis.kmeans_k_sweep import (
    _canonical_json,
    _csv_bytes,
    _sha256,
)
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen
from experiments.scalable_relevance_kmeans.scalable import _class_ids, _score_kernel


STUDY = "fused3_confirmation_v2_metric_aggregation_2x2"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
SHARED_OI = "FUSED3-SHARED-DIAG-OI"
CLASS_OI = "FUSED3-CLASS-DIAG-OI"
SHARED_ACC = "FUSED3-SHARED-DIAG-ACC"
CLASS_ACC = "FUSED3-CLASS-DIAG-ACC"
DIAGNOSTIC_METHODS = (SHARED_OI, CLASS_OI, SHARED_ACC, CLASS_ACC)
METHODS = (FROZEN, LINEAR_PROBE, *DIAGNOSTIC_METHODS)
EXPECTED_K_BY_BUDGET: Mapping[int, int] = {32: 5, 64: 10}
VARIANCE_PRIOR_ROWS = 32.0
RELATIVE_VARIANCE_FLOOR = 1.0e-6
ABSOLUTE_VARIANCE_FLOOR = 1.0e-12


def _validated_metric_inputs(
    values: Any,
    target: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    prototypes = np.asarray(centers, dtype=np.float32)
    owner_array = np.asarray(owners)
    relevance = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must be a non-empty two-dimensional matrix")
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise ValueError("target must be a scalar label vector aligned with values")
    if prototypes.ndim != 2 or prototypes.shape[1] != matrix.shape[1]:
        raise ValueError("centers must match the value feature dimension")
    if owner_array.ndim != 1 or owner_array.shape[0] != prototypes.shape[0]:
        raise ValueError("owners must align with centers")
    if relevance.shape != (matrix.shape[1],) or np.any(relevance <= 0.0):
        raise ValueError("weights must be a positive feature vector")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(prototypes)):
        raise ValueError("values and centers must be finite")
    if not np.all(np.isfinite(relevance)):
        raise ValueError("weights must be finite")
    classes = tuple(dict.fromkeys(owner_array.tolist()))
    if len(classes) < 2 or set(labels.tolist()) != set(classes):
        raise ValueError("target and prototype class sets must match")
    return matrix, labels, prototypes, owner_array, relevance, classes


def estimate_residual_metrics(
    values: Any,
    target: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
    prior_rows: float = VARIANCE_PRIOR_ROWS,
) -> dict[str, Any]:
    """Estimate matched shared and class-conditional residual variances.

    Rows are assigned to their nearest own-class frozen FUSED3 prototype under
    the frozen global relevance metric.  The treatment changes only covariance
    scope.  Each class variance receives ``prior_rows`` pooled pseudo-rows.
    """

    if not isinstance(prior_rows, float) or not np.isfinite(prior_rows) or prior_rows <= 0:
        raise ValueError("prior_rows must be a positive finite float")
    matrix, labels, prototypes, owner_array, relevance, classes = (
        _validated_metric_inputs(
            values,
            target,
            centers=centers,
            owners=owners,
            weights=weights,
        )
    )
    ids_by_class = _class_ids(owner_array)
    base_scores = _score_kernel(matrix, prototypes, relevance)
    pooled_sum = np.zeros(matrix.shape[1], dtype=np.float64)
    class_sums = np.zeros((len(classes), matrix.shape[1]), dtype=np.float64)
    class_counts = np.zeros(len(classes), dtype=np.int64)
    assigned_ids = np.empty(len(matrix), dtype=np.int64)
    for position, label in enumerate(classes):
        rows = np.flatnonzero(labels == label)
        own_ids = ids_by_class[label]
        chosen = own_ids[np.argmax(base_scores[rows][:, own_ids], axis=1)]
        residual = matrix[rows].astype(np.float64) - prototypes[chosen].astype(np.float64)
        squared = np.square(residual)
        class_sums[position] = np.sum(squared, axis=0, dtype=np.float64)
        class_counts[position] = int(len(rows))
        pooled_sum += class_sums[position]
        assigned_ids[rows] = chosen
    pooled_raw = pooled_sum / float(len(matrix))
    positive = pooled_raw[pooled_raw > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    floor = max(scale * RELATIVE_VARIANCE_FLOOR, ABSOLUTE_VARIANCE_FLOOR)
    pooled = np.maximum(pooled_raw, floor)
    class_raw = class_sums / class_counts[:, None]
    alpha = class_counts.astype(np.float64) / (
        class_counts.astype(np.float64) + float(prior_rows)
    )
    class_variance = (
        alpha[:, None] * class_raw + (1.0 - alpha[:, None]) * pooled[None, :]
    )
    class_variance = np.maximum(class_variance, floor)
    pooled_floor_count = int(np.count_nonzero(pooled_raw < floor))
    class_floor_count = int(np.count_nonzero(class_variance <= floor))
    for array in (pooled, class_variance, class_counts, alpha, assigned_ids):
        array.setflags(write=False)
    return {
        "classes": classes,
        "shared_variance": pooled,
        "class_variance": class_variance,
        "class_counts": class_counts,
        "class_shrinkage_data_weight": alpha,
        "assigned_prototype_ids": assigned_ids,
        "variance_floor": float(floor),
        "shared_floor_count": pooled_floor_count,
        "class_floor_count": class_floor_count,
        "prior_rows": float(prior_rows),
        "class_to_position": {label: position for position, label in enumerate(classes)},
    }


def gaussian_prototype_scores(
    values: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
    classes: Sequence[Any],
    variances: Any,
) -> np.ndarray:
    """Return comparable diagonal-Gaussian prototype energies.

    A single variance row is the shared control.  One row per class is the
    class-conditional treatment.  Frozen FUSED3 relevance multiplies each
    Gaussian precision and is identical in both metric scopes.
    """

    matrix = np.asarray(values, dtype=np.float32)
    prototypes = np.asarray(centers, dtype=np.float32)
    owner_array = np.asarray(owners)
    relevance = np.asarray(weights, dtype=np.float64)
    variance = np.asarray(variances, dtype=np.float64)
    class_order = tuple(classes)
    if matrix.ndim != 2 or prototypes.ndim != 2 or matrix.shape[1] != prototypes.shape[1]:
        raise ValueError("values and centers must be aligned matrices")
    if owner_array.shape != (len(prototypes),):
        raise ValueError("owners must align with centers")
    if relevance.shape != (matrix.shape[1],) or np.any(relevance <= 0.0):
        raise ValueError("weights must be a positive feature vector")
    if variance.ndim != 2 or variance.shape[1] != matrix.shape[1]:
        raise ValueError("variances must be a metric-by-feature matrix")
    if variance.shape[0] not in (1, len(class_order)):
        raise ValueError("variances must be shared or aligned one-per-class")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("variances must be finite and positive")
    class_positions = {label: position for position, label in enumerate(class_order)}
    if set(owner_array.tolist()) != set(class_positions):
        raise ValueError("owner classes must match the variance class order")
    prototype_class_positions = np.asarray(
        [class_positions[label] for label in owner_array.tolist()], dtype=np.int64
    )
    if variance.shape[0] == 1:
        prototype_variance = np.broadcast_to(variance, prototypes.shape)
    else:
        prototype_variance = variance[prototype_class_positions]
    precision = np.asarray(relevance[None, :] / prototype_variance, dtype=np.float32)
    effective_variance = prototype_variance / relevance[None, :]
    log_determinant = np.sum(np.log(effective_variance), axis=1, dtype=np.float64)
    linear = np.asarray(matrix @ (prototypes * precision).T, dtype=np.float32)
    if variance.shape[0] == 1:
        quadratic_rows = np.sum(
            np.square(matrix, dtype=np.float32) * precision[0],
            axis=1,
            dtype=np.float32,
        )
        quadratic = quadratic_rows[:, None]
    else:
        quadratic = np.asarray(
            np.square(matrix, dtype=np.float32) @ precision.T,
            dtype=np.float32,
        )
    center_constant = -0.5 * np.sum(
        np.square(prototypes, dtype=np.float32) * precision,
        axis=1,
        dtype=np.float32,
    )
    constant = center_constant.astype(np.float64) - 0.5 * log_determinant
    scores = linear.astype(np.float64) - 0.5 * quadratic.astype(np.float64)
    scores += constant[None, :]
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Gaussian prototype scoring emitted non-finite values")
    return np.asarray(scores, dtype=np.float64)


def exact_oi_from_scores(
    scores: Any,
    target: Any,
    *,
    owners: Any,
) -> tuple[float, dict[str, Any]]:
    """Apply the exact second-own/worst-rival OI event aggregation."""

    matrix = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(target)
    owner_array = np.asarray(owners)
    if matrix.ndim != 2 or labels.shape != (len(matrix),):
        raise ValueError("scores and scalar targets must align")
    if owner_array.shape != (matrix.shape[1],):
        raise ValueError("owners must align with score columns")
    classes = tuple(dict.fromkeys(labels.tolist()))
    ids_by_class = _class_ids(owner_array)
    if set(classes) != set(ids_by_class):
        raise ValueError("target and owner class sets must match")
    pair_hits: dict[tuple[Any, Any], int] = {}
    singleton: list[float] = []
    under_prototyped: list[Any] = []
    for source in classes:
        rows = np.flatnonzero(labels == source)
        own_ids = ids_by_class[source]
        own_scores = matrix[rows][:, own_ids]
        if len(own_ids) < 2:
            threshold = np.full(len(rows), -np.inf, dtype=np.float64)
            under_prototyped.append(source)
        else:
            threshold = np.partition(own_scores, kth=-2, axis=1)[:, -2]
        pair_values: list[float] = []
        for rival in classes:
            if rival == source:
                continue
            best_rival = np.max(matrix[rows][:, ids_by_class[rival]], axis=1)
            hits = int(np.count_nonzero(best_rival > threshold))
            pair_hits[(source, rival)] = hits
            pair_values.append(1.0 - float(hits) / float(len(rows)))
        singleton.append(min(pair_values))
    score = float(np.mean(singleton))
    return score, {
        "class_count": len(classes),
        "directed_pair_count": len(pair_hits),
        "under_prototyped_class_count": len(under_prototyped),
        "pair_hit_total": int(sum(pair_hits.values())),
    }


def discriminant_summary_from_scores(
    scores: Any,
    target: Any,
    *,
    owners: Any,
) -> dict[str, float]:
    """Return OOF nearest-component class accuracy and signed class margin."""

    matrix = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(target)
    owner_array = np.asarray(owners)
    if matrix.ndim != 2 or labels.shape != (len(matrix),):
        raise ValueError("scores and scalar targets must align")
    if owner_array.shape != (matrix.shape[1],):
        raise ValueError("owners must align with score columns")
    classes = tuple(dict.fromkeys(owner_array.tolist()))
    ids_by_class = _class_ids(owner_array)
    if set(labels.tolist()) != set(classes):
        raise ValueError("target and owner class sets must match")
    class_scores = np.column_stack(
        [np.max(matrix[:, ids_by_class[label]], axis=1) for label in classes]
    )
    predicted_positions = np.argmax(class_scores, axis=1)
    predicted = np.asarray([classes[position] for position in predicted_positions])
    correct_positions = np.asarray(
        [{label: position for position, label in enumerate(classes)}[label] for label in labels],
        dtype=np.int64,
    )
    correct_scores = class_scores[np.arange(len(matrix)), correct_positions]
    wrong = class_scores.copy()
    wrong[np.arange(len(matrix)), correct_positions] = -np.inf
    margin = correct_scores - np.max(wrong, axis=1)
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "mean_correct_minus_best_wrong_margin": float(np.mean(margin)),
        "median_correct_minus_best_wrong_margin": float(np.median(margin)),
    }


def crossfit_metric_aggregation_2x2(
    values: Any,
    target: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit exact frozen FUSED3 folds and evaluate the four diagnostics."""

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    if matrix.ndim != 2 or labels.shape != (len(matrix),):
        raise ValueError("values and target must be aligned")
    normalized = food101._row_l2(matrix)
    folds = food101._stratified_folds(labels, n_splits=5, seed=int(seed))
    fold_rows: list[dict[str, Any]] = []
    overall_started = time.perf_counter()
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        k_per_class = fused_food_screen._k_per_class(labels, train)
        selector = fused_food_screen._build_selector("FUSED3", k_per_class, fold_seed)
        fit_started = time.perf_counter()
        selector.fit(normalized[train], labels[train])
        fit_wall = time.perf_counter() - fit_started
        state_before = fused_food_screen._numerical_state_sha256(selector)
        frozen_started = time.perf_counter()
        frozen_score = selector.score_fixed(normalized[holdout], labels[holdout])
        frozen_score_wall = time.perf_counter() - frozen_started
        state_after = fused_food_screen._numerical_state_sha256(selector)
        if state_before != state_after:
            raise RuntimeError("diagnostic scoring mutated frozen FUSED3 state")
        metric_started = time.perf_counter()
        metrics = estimate_residual_metrics(
            normalized[train],
            labels[train],
            centers=selector.centers_,
            owners=selector.owners_,
            weights=selector.weights_,
        )
        metric_wall = time.perf_counter() - metric_started
        shared_started = time.perf_counter()
        shared_scores = gaussian_prototype_scores(
            normalized[holdout],
            centers=selector.centers_,
            owners=selector.owners_,
            weights=selector.weights_,
            classes=metrics["classes"],
            variances=np.asarray(metrics["shared_variance"])[None, :],
        )
        shared_oi, shared_oi_diagnostics = exact_oi_from_scores(
            shared_scores, labels[holdout], owners=selector.owners_
        )
        shared_discriminant = discriminant_summary_from_scores(
            shared_scores, labels[holdout], owners=selector.owners_
        )
        shared_wall = time.perf_counter() - shared_started
        class_started = time.perf_counter()
        class_scores = gaussian_prototype_scores(
            normalized[holdout],
            centers=selector.centers_,
            owners=selector.owners_,
            weights=selector.weights_,
            classes=metrics["classes"],
            variances=metrics["class_variance"],
        )
        class_oi, class_oi_diagnostics = exact_oi_from_scores(
            class_scores, labels[holdout], owners=selector.owners_
        )
        class_discriminant = discriminant_summary_from_scores(
            class_scores, labels[holdout], owners=selector.owners_
        )
        class_wall = time.perf_counter() - class_started
        class_variance = np.asarray(metrics["class_variance"], dtype=np.float64)
        shared_variance = np.asarray(metrics["shared_variance"], dtype=np.float64)
        fold_rows.append(
            {
                "fold": int(fold),
                "fold_seed": int(fold_seed),
                "train_size": int(len(train)),
                "holdout_size": int(len(holdout)),
                "k_min": int(min(k_per_class.values())),
                "k_max": int(max(k_per_class.values())),
                "prototype_count": int(len(selector.centers_)),
                "frozen_fused3_score": float(frozen_score),
                "shared_oi_score": float(shared_oi),
                "class_oi_score": float(class_oi),
                "shared_accuracy": float(shared_discriminant["accuracy"]),
                "class_accuracy": float(class_discriminant["accuracy"]),
                "shared_mean_margin": float(
                    shared_discriminant["mean_correct_minus_best_wrong_margin"]
                ),
                "class_mean_margin": float(
                    class_discriminant["mean_correct_minus_best_wrong_margin"]
                ),
                "fit_wall_seconds": float(fit_wall),
                "frozen_score_wall_seconds": float(frozen_score_wall),
                "metric_fit_wall_seconds": float(metric_wall),
                "shared_score_wall_seconds": float(shared_wall),
                "class_score_wall_seconds": float(class_wall),
                "variance_prior_rows": float(metrics["prior_rows"]),
                "variance_floor": float(metrics["variance_floor"]),
                "shared_floor_count": int(metrics["shared_floor_count"]),
                "class_floor_count": int(metrics["class_floor_count"]),
                "median_class_to_shared_variance_ratio": float(
                    np.median(class_variance / shared_variance[None, :])
                ),
                "p95_abs_log_class_to_shared_variance_ratio": float(
                    np.percentile(
                        np.abs(np.log(class_variance / shared_variance[None, :])), 95
                    )
                ),
                "shared_pair_hit_total": int(shared_oi_diagnostics["pair_hit_total"]),
                "class_pair_hit_total": int(class_oi_diagnostics["pair_hit_total"]),
                "state_sha256": state_before,
                "state_unchanged_after_all_diagnostics": True,
            }
        )
    return {
        "scores": {
            FROZEN: float(np.mean([row["frozen_fused3_score"] for row in fold_rows])),
            SHARED_OI: float(np.mean([row["shared_oi_score"] for row in fold_rows])),
            CLASS_OI: float(np.mean([row["class_oi_score"] for row in fold_rows])),
            SHARED_ACC: float(np.mean([row["shared_accuracy"] for row in fold_rows])),
            CLASS_ACC: float(np.mean([row["class_accuracy"] for row in fold_rows])),
        },
        "method_wall_seconds": {
            SHARED_OI: float(
                sum(
                    row["fit_wall_seconds"]
                    + row["metric_fit_wall_seconds"]
                    + row["shared_score_wall_seconds"]
                    for row in fold_rows
                )
            ),
            SHARED_ACC: float(
                sum(
                    row["fit_wall_seconds"]
                    + row["metric_fit_wall_seconds"]
                    + row["shared_score_wall_seconds"]
                    for row in fold_rows
                )
            ),
            CLASS_OI: float(
                sum(
                    row["fit_wall_seconds"]
                    + row["metric_fit_wall_seconds"]
                    + row["class_score_wall_seconds"]
                    for row in fold_rows
                )
            ),
            CLASS_ACC: float(
                sum(
                    row["fit_wall_seconds"]
                    + row["metric_fit_wall_seconds"]
                    + row["class_score_wall_seconds"]
                    for row in fold_rows
                )
            ),
        },
        "total_wall_seconds": float(time.perf_counter() - overall_started),
        "folds": fold_rows,
    }


def _frozen_score_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, int], float]:
    lookup: dict[tuple[str, str, int, int], float] = {}
    for row in rows:
        if row["candidate_id"] != FROZEN:
            continue
        key = (
            str(row["dataset_id"]),
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
        )
        if key in lookup:
            raise ValueError("duplicate frozen FUSED3 score")
        lookup[key] = float(row["score"])
    expected = {
        (dataset, backbone, seed, budget)
        for dataset in statistics.DATASET_IDS
        for backbone in statistics.BACKBONES
        for seed in statistics.REPLICATE_SEEDS
        for budget in statistics.BUDGETS
    }
    if set(lookup) != expected:
        raise ValueError("frozen FUSED3 score grid is incomplete")
    return lookup


def _aircraft_focus(
    selector_rows: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selector_lookup = _validate_comparison_rows(selector_rows, methods=METHODS)
    outcomes = _reference_lookup(references)
    rows: list[dict[str, Any]] = []
    dataset = "torchvision_fgvc_aircraft"
    for budget in statistics.BUDGETS:
        for method in METHODS:
            selected_counts: dict[str, int] = {}
            dino_minus_openclip: list[float] = []
            regret_values: list[float] = []
            for seed in statistics.REPLICATE_SEEDS:
                scores = {
                    backbone: float(
                        selector_lookup[(dataset, backbone, seed, budget, method)]["score"]
                    )
                    for backbone in statistics.BACKBONES
                }
                maximum = max(scores.values())
                selected = [
                    backbone
                    for backbone, score in scores.items()
                    if np.isclose(
                        score,
                        maximum,
                        atol=statistics.SELECTION_TIE_ATOL,
                        rtol=0.0,
                    )
                ]
                for backbone in selected:
                    selected_counts[backbone] = selected_counts.get(backbone, 0) + 1
                dino_minus_openclip.append(
                    scores["dinov2-small"] - scores["openclip-vit-b-32"]
                )
                envelope = {
                    backbone: max(
                        outcomes[(dataset, backbone, seed, budget, head)]
                        for head in statistics.HEADS
                    )
                    for backbone in statistics.BACKBONES
                }
                regret_values.append(
                    100.0
                    * (
                        max(envelope.values())
                        - float(np.mean([envelope[backbone] for backbone in selected]))
                    )
                )
            rows.append(
                {
                    "budget": int(budget),
                    "candidate_id": method,
                    "mean_regret_pp": float(np.mean(regret_values)),
                    "mean_dino_minus_openclip_score": float(
                        np.mean(dino_minus_openclip)
                    ),
                    "selection_counts": json.dumps(
                        dict(sorted(selected_counts.items())), separators=(",", ":")
                    ),
                }
            )
    return rows


def analyze_metric_aggregation_2x2(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    """Execute the complete fixed 2x2 diagnostic on existing inputs."""

    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    source_selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    frozen_scores = _frozen_score_lookup(source_selectors)
    root = Path(registry_root)
    registry = json.loads((root / "audited_registry.json").read_text(encoding="utf-8"))
    registry_rows = {str(row["dataset_id"]): row for row in registry["datasets"]}
    if set(registry_rows) != set(statistics.DATASET_IDS):
        raise RuntimeError("audited registry dataset set mismatch")
    _reference_lookup(references)

    state_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for dataset_position, dataset_id in enumerate(statistics.DATASET_IDS, start=1):
        dataset_row = registry_rows[dataset_id]
        for backbone in statistics.BACKBONES:
            for replicate, seed in enumerate(statistics.REPLICATE_SEEDS):
                for budget in statistics.BUDGETS:
                    panel = datasets.load_panel(
                        dataset_row,
                        backbone,
                        int(seed),
                        int(budget),
                        registry_root=root,
                    )
                    result = crossfit_metric_aggregation_2x2(
                        panel["training_values"],
                        panel["training_labels"],
                        seed=int(seed),
                    )
                    expected_k = EXPECTED_K_BY_BUDGET[int(budget)]
                    if any(
                        int(row["k_min"]) != expected_k
                        or int(row["k_max"]) != expected_k
                        for row in result["folds"]
                    ):
                        raise RuntimeError("resolved K differs from frozen FUSED3 capacity")
                    key = (dataset_id, backbone, int(seed), int(budget))
                    recomputed = float(result["scores"][FROZEN])
                    source_score = frozen_scores[key]
                    if recomputed != source_score:
                        raise RuntimeError("frozen FUSED3 recomputation parity failed")
                    parity_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "backbone": backbone,
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "source_score": float(source_score),
                            "recomputed_score": float(recomputed),
                            "exact_match": True,
                        }
                    )
                    for method in DIAGNOSTIC_METHODS:
                        state_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "backbone": backbone,
                                "replicate": int(replicate),
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                "candidate_id": method,
                                "metric_scope": (
                                    "shared_global"
                                    if method in (SHARED_OI, SHARED_ACC)
                                    else "class_conditional"
                                ),
                                "aggregation": (
                                    "exact_oi_events"
                                    if method in (SHARED_OI, CLASS_OI)
                                    else "oof_discriminant_accuracy"
                                ),
                                "score": float(result["scores"][method]),
                                "total_wall_seconds": float(
                                    result["method_wall_seconds"][method]
                                ),
                            }
                        )
                    for row in result["folds"]:
                        fold_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "backbone": backbone,
                                "replicate": int(replicate),
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                **row,
                            }
                        )
        if progress:
            print(
                f"completed {dataset_position}/{len(statistics.DATASET_IDS)}: {dataset_id}",
                flush=True,
            )
    if len(state_rows) != 2000 or len(parity_rows) != 500 or len(fold_rows) != 2500:
        raise RuntimeError("metric-by-aggregation diagnostic row count mismatch")

    source_rows = [
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
    ]
    comparison_rows = [*source_rows, *state_rows]
    _validate_comparison_rows(comparison_rows, methods=METHODS)
    envelope = envelope_panel_metrics(comparison_rows, references, methods=METHODS)
    heads = head_panel_metrics(comparison_rows, references, methods=METHODS)
    dataset_aggregate, overall = aggregate_envelope(envelope, methods=METHODS)
    head_aggregate = aggregate_heads(heads, methods=METHODS)
    runtimes = runtime_summary(
        comparison_rows,
        methods=METHODS,
        executed_methods=DIAGNOSTIC_METHODS,
    )
    contrasts = [
        contrast_interval(envelope, candidate_id=method, comparator_id=comparator)
        for method in DIAGNOSTIC_METHODS
        for comparator in (FROZEN, LINEAR_PROBE)
    ]
    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "primary_estimand": "best_head_envelope_backbone_ranking",
        "head_library": list(statistics.HEADS),
        "methods": list(METHODS),
        "executed_methods": list(DIAGNOSTIC_METHODS),
        "design": {
            "prototypes": "exact frozen FUSED3 fitted independently per training fold",
            "assignments": "nearest own prototype under frozen FUSED3 relevance weights",
            "shared_metric": "pooled own-prototype residual diagonal variance",
            "class_metric": "class residual diagonal variance shrunk toward pooled",
            "variance_prior_rows": VARIANCE_PRIOR_ROWS,
            "variance_floor": "max(1e-6*median_positive_pooled_variance,1e-12)",
            "effective_precision": "frozen_FUSED3_relevance_weight/residual_variance",
            "gaussian_log_determinant": True,
            "oi_aggregation": "exact second-own threshold and worst-rival singleton mean",
            "discriminant_aggregation": "five-fold OOF nearest-component class accuracy",
        },
        "state_rows": state_rows,
        "baseline_parity_rows": parity_rows,
        "fold_rows": fold_rows,
        "panel_metrics": envelope,
        "dataset_aggregate": dataset_aggregate,
        "overall_aggregate": overall,
        "head_panel_metrics": heads,
        "head_aggregate": head_aggregate,
        "runtime_descriptive": runtimes,
        "regret_contrasts": contrasts,
        "aircraft_focus": _aircraft_focus(comparison_rows, references),
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "post_outcome_limit": (
            "All five datasets and their outcomes were previously inspected; "
            "this result is development evidence and not confirmation."
        ),
        "input_hashes": dict(verified["input_hashes"]),
        "input_identity": {
            key: verified["raw"][key]
            for key in (
                "protocol_sha256",
                "code_identity_sha256",
                "run_identity_sha256",
                "audited_registry_sha256",
                "lineage_lock_sha256",
            )
        },
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# FUSED3 diagonal metric × aggregation diagnostic",
        "",
        "Status: **post-outcome development diagnostic only**. The frozen confirmation decision is unchanged.",
        "",
        "Every cell starts from the exact frozen FUSED3 prototypes and relevance weights. The shared/class comparison changes only residual-covariance scope; the OI/ACC comparison changes only the held-out aggregation.",
        "",
        "## Overall best-head-envelope ranking",
        "",
        "| Selector | Metric | Aggregation | Mean regret (pp) | Exact best | Within 1 pp | Mean Spearman |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    method_design = {
        FROZEN: ("frozen relevance", "exact OI"),
        LINEAR_PROBE: ("linear probe", "OOF accuracy"),
        SHARED_OI: ("shared diagonal", "exact OI"),
        CLASS_OI: ("class diagonal", "exact OI"),
        SHARED_ACC: ("shared diagonal", "OOF accuracy"),
        CLASS_ACC: ("class diagonal", "OOF accuracy"),
    }
    for row in summary["overall_aggregate"]:
        metric, aggregation = method_design[row["candidate_id"]]
        lines.append(
            f"| {row['candidate_id']} | {metric} | {aggregation} | "
            f"{row['equal_dataset_mean_regret_pp']:.3f} | "
            f"{row['equal_dataset_exact_best_rate']:.3f} | "
            f"{row['equal_dataset_within_one_pp_rate']:.3f} | "
            f"{row['equal_dataset_mean_spearman']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset results",
            "",
            "| Dataset | Selector | Regret (pp) | Exact best | Spearman | Selections |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["dataset_aggregate"]:
        rho = "—" if row["mean_spearman"] is None else f"{row['mean_spearman']:.3f}"
        lines.append(
            f"| {row['dataset_id']} | {row['candidate_id']} | "
            f"{row['mean_regret_pp']:.3f} | {row['exact_best_rate']:.3f} | "
            f"{rho} | `{row['selection_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Aircraft mechanism focus",
            "",
            "Positive DINO − OpenCLIP score means the method prefers DINO, the Aircraft envelope winner.",
            "",
            "| Budget | Selector | Regret (pp) | DINO − OpenCLIP score | Selections |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in summary["aircraft_focus"]:
        lines.append(
            f"| {row['budget']} | {row['candidate_id']} | "
            f"{row['mean_regret_pp']:.3f} | "
            f"{row['mean_dino_minus_openclip_score']:+.6f} | "
            f"`{row['selection_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Paired envelope-regret contrasts",
            "",
            "Negative values favor the diagnostic method.",
            "",
            "| Candidate | Comparator | Estimate (pp) | Descriptive 95% interval |",
            "|---|---|---:|---:|",
        ]
    )
    for row in summary["regret_contrasts"]:
        lines.append(
            f"| {row['candidate_id']} | {row['comparator_id']} | "
            f"{row['estimate']:+.3f} | [{row['lower_95']:+.3f}, {row['upper_95']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Secondary per-head ranking regret",
            "",
            "| Head | Selector | Equal-dataset regret (pp) | Mean Spearman |",
            "|---|---|---:|---:|",
        ]
    )
    for row in summary["head_aggregate"]:
        rho = (
            "—"
            if row["equal_dataset_mean_spearman"] is None
            else f"{row['equal_dataset_mean_spearman']:.3f}"
        )
        lines.append(
            f"| {row['head']} | {row['candidate_id']} | "
            f"{row['equal_dataset_mean_regret_pp']:.3f} | {rho} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "Diagnostic timings are separate-run totals and are descriptive, not counterbalanced product gates.",
            "",
            "| Dataset | Budget | Selector | Call/LP | Full panel/LP |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in summary["runtime_descriptive"]:
        lines.append(
            f"| {row['dataset_id']} | {row['budget']} | {row['candidate_id']} | "
            f"{row['median_call_ratio_to_lp']:.2f}× | "
            f"{row['median_full_panel_ratio_to_lp']:.2f}× |"
        )
    lines.extend(
        [
            "",
            "No promotion, reselection, or confirmation claim is made from these previously inspected datasets.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_metric_aggregation_2x2(
        full_artifact, registry_root, progress=progress
    )
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "state_rows.csv": _csv_bytes(summary["state_rows"]),
        "baseline_parity_rows.csv": _csv_bytes(summary["baseline_parity_rows"]),
        "fold_rows.csv": _csv_bytes(summary["fold_rows"]),
        "panel_metrics.csv": _csv_bytes(summary["panel_metrics"]),
        "dataset_aggregate.csv": _csv_bytes(summary["dataset_aggregate"]),
        "overall_aggregate.csv": _csv_bytes(summary["overall_aggregate"]),
        "head_panel_metrics.csv": _csv_bytes(summary["head_panel_metrics"]),
        "head_aggregate.csv": _csv_bytes(summary["head_aggregate"]),
        "runtime_descriptive.csv": _csv_bytes(summary["runtime_descriptive"]),
        "regret_contrasts.csv": _csv_bytes(summary["regret_contrasts"]),
        "aircraft_focus.csv": _csv_bytes(summary["aircraft_focus"]),
        "report.md": render_report(summary).encode("utf-8"),
    }
    manifest = {
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
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
    payloads["diagnostic_manifest.json"] = (
        _canonical_json(manifest) + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for name, content in payloads.items():
            (staging / name).write_bytes(content)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for name, descriptor in manifest["files"].items():
        if _sha256(destination / name) != descriptor["sha256"]:
            raise RuntimeError("written 2x2 diagnostic artifact hash mismatch")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_bundle(args.input, args.registry_root, args.output, progress=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
