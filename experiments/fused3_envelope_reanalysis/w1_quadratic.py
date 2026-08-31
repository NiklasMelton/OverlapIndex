"""Aircraft diagnosis: one weighted Lloyd update and quadratic decomposition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.svm import SVC

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen
from experiments.scalable_relevance_kmeans.scalable import _class_ids, _score_kernel
from experiments.scalable_relevance_kmeans.scalable_v2 import fast_tiled_oi_score


STUDY = "fused3_confirmation_v2_aircraft_w1_quadratic_diagnostic"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
DATASET_ID = "torchvision_fgvc_aircraft"
METHODS = ("FUSED3", "FUSED3-W1")
FOCAL_BACKBONES = ("dinov2-small", "openclip-vit-b-32")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError("cannot average an empty diagnostic group")
    return float(np.mean([float(row[field]) for row in rows]))


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("Spearman inputs must be aligned vectors")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    value = float(
        np.corrcoef(
            rankdata(x, method="average"), rankdata(y, method="average")
        )[0, 1]
    )
    return value if np.isfinite(value) else None


def weighted_lloyd_one_update(
    values: Any,
    target: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Perform one same-class assignment/mean update under frozen weights."""

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    initial = np.asarray(centers, dtype=np.float32)
    owner_array = np.asarray(owners)
    feature_weights = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or labels.ndim != 1 or len(matrix) != len(labels):
        raise ValueError("values and target must be aligned 2-D/1-D arrays")
    if initial.ndim != 2 or initial.shape[1] != matrix.shape[1]:
        raise ValueError("centers do not match the feature dimension")
    if owner_array.shape != (len(initial),):
        raise ValueError("owners do not align with centers")
    if feature_weights.shape != (matrix.shape[1],) or np.any(feature_weights <= 0):
        raise ValueError("weights must be a positive feature vector")
    ids_by_class = _class_ids(owner_array)
    if set(dict.fromkeys(labels.tolist())) != set(ids_by_class):
        raise ValueError("training and prototype class sets must match")
    sums = np.zeros_like(initial, dtype=np.float64)
    counts = np.zeros(len(initial), dtype=np.int64)
    for label in dict.fromkeys(labels.tolist()):
        class_values = matrix[labels == label]
        own_ids = ids_by_class[label]
        scores = _score_kernel(class_values, initial[own_ids], feature_weights)
        assigned = own_ids[np.argmax(scores, axis=1)]
        for prototype_id in np.unique(assigned):
            selected = class_values[assigned == prototype_id]
            sums[prototype_id] = np.sum(selected, axis=0, dtype=np.float64)
            counts[prototype_id] = len(selected)
    updated = initial.astype(np.float64, copy=True)
    nonempty = counts > 0
    updated[nonempty] = sums[nonempty] / counts[nonempty, None]
    output = np.asarray(updated, dtype=np.float32)
    output.setflags(write=False)
    return output, {
        "weighted_lloyd_update_count": 1,
        "empty_prototype_count_after_update": int(np.count_nonzero(~nonempty)),
        "changed_prototype_count": int(
            np.count_nonzero(np.any(output != initial, axis=1))
        ),
        "prototype_count": int(len(initial)),
        "weights_relearned": False,
        "prototype_owners_changed": False,
        "empty_prototype_policy": "retain_pre_update_center",
    }


def prototype_geometry(
    values: Any,
    target: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
) -> dict[str, Any]:
    """Expose the exact best-own/second-own/rival geometry behind OI."""

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    prototypes = np.asarray(centers, dtype=np.float32)
    owner_array = np.asarray(owners)
    feature_weights = np.asarray(weights, dtype=np.float64)
    scores = _score_kernel(matrix, prototypes, feature_weights)
    ids_by_class = _class_ids(owner_array)
    pair_hits = {
        (source, other): 0
        for source in ids_by_class
        for other in ids_by_class
        if source != other
    }
    supports = {
        source: int(np.count_nonzero(labels == source)) for source in ids_by_class
    }
    nearest_correct: list[np.ndarray] = []
    all_rivals_second_own_pass: list[np.ndarray] = []
    nearest_correct_second_own_fail: list[np.ndarray] = []
    best_gaps: list[np.ndarray] = []
    second_gaps: list[np.ndarray] = []
    for source, own_ids in ids_by_class.items():
        rows = np.flatnonzero(labels == source)
        own_scores = scores[rows][:, own_ids]
        own_best = np.max(own_scores, axis=1)
        if own_ids.size < 2:
            own_second = np.full(len(rows), -np.inf, dtype=np.float32)
        else:
            own_second = np.partition(own_scores, kth=-2, axis=1)[:, -2]
        wrong_ids = np.concatenate(
            [ids for other, ids in ids_by_class.items() if other != source]
        )
        wrong_best = np.max(scores[rows][:, wrong_ids], axis=1)
        correct = own_best > wrong_best
        redundant = own_second >= wrong_best
        nearest_correct.append(correct)
        all_rivals_second_own_pass.append(redundant)
        nearest_correct_second_own_fail.append(correct & ~redundant)
        best_gaps.append(own_best - wrong_best)
        second_gaps.append(own_second - wrong_best)
        for other, other_ids in ids_by_class.items():
            if other == source:
                continue
            target_best = np.max(scores[rows][:, other_ids], axis=1)
            pair_hits[(source, other)] = int(
                np.count_nonzero(target_best > own_second)
            )
    singleton = [
        min(
            1.0 - pair_hits[(source, other)] / supports[source]
            for other in ids_by_class
            if other != source
        )
        for source in ids_by_class
    ]
    oi_score = float(np.mean(singleton))
    exact, _diagnostics = fast_tiled_oi_score(
        matrix,
        labels,
        centers=prototypes,
        owners=owner_array,
        weights=feature_weights,
        memory_budget_mb=64,
    )
    if abs(exact - oi_score) > 1.0e-12:
        raise RuntimeError("prototype geometry did not reproduce exact OI")
    return {
        "oi_score": oi_score,
        "nearest_prototype_accuracy": float(
            np.mean(np.concatenate(nearest_correct))
        ),
        "all_rivals_second_own_pass_rate": float(
            np.mean(np.concatenate(all_rivals_second_own_pass))
        ),
        "nearest_correct_but_second_own_fails_rate": float(
            np.mean(np.concatenate(nearest_correct_second_own_fail))
        ),
        "mean_best_own_minus_best_wrong_score": float(
            np.mean(np.concatenate(best_gaps))
        ),
        "mean_second_own_minus_best_wrong_score": float(
            np.mean(np.concatenate(second_gaps))
        ),
    }


def crossfit_w1(values: Any, target: Any, *, seed: int) -> dict[str, Any]:
    """Reproduce frozen FUSED3 and derive W1 from the identical fitted state."""

    matrix = food101._row_l2(np.asarray(values, dtype=np.float32))
    labels = np.asarray(target)
    folds = food101._stratified_folds(labels, n_splits=5, seed=int(seed))
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        k_per_class = fused_food_screen._k_per_class(labels, train)
        baseline = fused_food_screen._build_selector(
            "FUSED3", k_per_class, fold_seed
        ).fit(matrix[train], labels[train])
        centers_before = np.array(baseline.centers_, copy=True)
        owners_before = np.array(baseline.owners_, copy=True)
        weights_before = np.array(baseline.weights_, copy=True)
        baseline_geometry = prototype_geometry(
            matrix[holdout],
            labels[holdout],
            centers=baseline.centers_,
            owners=baseline.owners_,
            weights=baseline.weights_,
        )
        started = time.perf_counter()
        updated_centers, update_diagnostics = weighted_lloyd_one_update(
            matrix[train],
            labels[train],
            centers=baseline.centers_,
            owners=baseline.owners_,
            weights=baseline.weights_,
        )
        update_wall = time.perf_counter() - started
        if not np.array_equal(baseline.centers_, centers_before):
            raise RuntimeError("W1 mutated frozen baseline centers")
        if not np.array_equal(baseline.owners_, owners_before):
            raise RuntimeError("W1 mutated frozen baseline owners")
        if not np.array_equal(baseline.weights_, weights_before):
            raise RuntimeError("W1 mutated frozen baseline weights")
        w1_geometry = prototype_geometry(
            matrix[holdout],
            labels[holdout],
            centers=updated_centers,
            owners=baseline.owners_,
            weights=baseline.weights_,
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "fold_seed": fold_seed,
                "train_size": int(len(train)),
                "holdout_size": int(len(holdout)),
                "prototype_count": int(len(baseline.centers_)),
                "weight_condition": float(
                    np.max(baseline.weights_) / np.min(baseline.weights_)
                ),
                "baseline": baseline_geometry,
                "w1": w1_geometry,
                "w1_update_wall_seconds": float(update_wall),
                "w1_update_diagnostics": update_diagnostics,
            }
        )
    return {
        "baseline_score": float(
            np.mean([row["baseline"]["oi_score"] for row in fold_rows])
        ),
        "w1_score": float(
            np.mean([row["w1"]["oi_score"] for row in fold_rows])
        ),
        "folds": fold_rows,
    }


def decomposed_polynomial_kernel(
    left: Any,
    right: Any,
    *,
    gamma: float,
    include_cross_terms: bool,
) -> np.ndarray:
    """Return the frozen degree-two kernel or its diagonal-only component."""

    X = np.asarray(left, dtype=np.float32)
    Z = np.asarray(right, dtype=np.float32)
    if X.ndim != 2 or Z.ndim != 2 or X.shape[1] != Z.shape[1]:
        raise ValueError("kernel inputs must be aligned two-dimensional matrices")
    if not np.isfinite(X).all() or not np.isfinite(Z).all():
        raise ValueError("kernel inputs must be finite")
    if not isinstance(gamma, (float, np.floating)) or gamma <= 0:
        raise ValueError("gamma must be a positive float")
    linear_gram = np.asarray(X @ Z.T, dtype=np.float32)
    if include_cross_terms:
        result = np.square(
            linear_gram * np.float32(gamma) + np.float32(1.0),
            dtype=np.float32,
        )
    else:
        result = linear_gram * np.float32(2.0 * gamma)
        result += np.asarray((X * X) @ (Z * Z).T, dtype=np.float32) * np.float32(
            gamma * gamma
        )
        result += np.float32(1.0)
    return np.asarray(result, dtype=np.float32)


def diagonal_quadratic_accuracy(
    training_values: Any,
    training_labels: Any,
    evaluation_values: Any,
    evaluation_labels: Any,
) -> dict[str, Any]:
    """Fit the frozen SVC on linear plus per-coordinate square features only."""

    train = food101._row_l2(np.asarray(training_values, dtype=np.float32))
    test = food101._row_l2(np.asarray(evaluation_values, dtype=np.float32))
    train_y = np.asarray(training_labels)
    test_y = np.asarray(evaluation_labels)
    gamma = float(1.0 / (train.shape[1] * float(np.var(train))))
    started = time.perf_counter()
    train_kernel = decomposed_polynomial_kernel(
        train, train, gamma=gamma, include_cross_terms=False
    )
    estimator = SVC(kernel="precomputed", C=1.0)
    estimator.fit(train_kernel, train_y)
    test_kernel = decomposed_polynomial_kernel(
        test, train, gamma=gamma, include_cross_terms=False
    )
    predicted = estimator.predict(test_kernel)
    elapsed = time.perf_counter() - started
    return {
        "test_accuracy": float(np.mean(predicted == test_y)),
        "gamma_scale": gamma,
        "training_row_count": int(len(train)),
        "evaluation_row_count": int(len(test)),
        "support_vector_count": int(np.sum(estimator.n_support_)),
        "wall_seconds": float(elapsed),
        "kernel": "1 + 2*gamma*(x dot z) + gamma^2*((x squared) dot (z squared))",
        "cross_coordinate_terms_included": False,
    }


def _row_lookups(
    selector_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int, int], float],
    dict[tuple[str, int, int, str], float],
]:
    frozen: dict[tuple[str, int, int], float] = {}
    references: dict[tuple[str, int, int, str], float] = {}
    for row in selector_rows:
        if row["dataset_id"] == DATASET_ID and row["candidate_id"] == "FUSED3":
            key = (
                str(row["backbone"]),
                int(row["replicate_seed"]),
                int(row["budget"]),
            )
            frozen[key] = float(row["score"])
    for row in reference_rows:
        if row["dataset_id"] == DATASET_ID:
            key = (
                str(row["backbone"]),
                int(row["replicate_seed"]),
                int(row["budget"]),
                str(row["head"]),
            )
            references[key] = float(row["test_accuracy"])
    if len(frozen) != 100 or len(references) != 400:
        raise RuntimeError("frozen Aircraft row lookups are incomplete")
    return frozen, references


def analyze_diagnostic(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Run the complete W1 grid and focal quadratic decomposition."""

    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    frozen_scores, reference = _row_lookups(selectors, references)
    root = Path(registry_root)
    registry = json.loads((root / "audited_registry.json").read_text(encoding="utf-8"))
    aircraft = next(
        row for row in registry["datasets"] if row["dataset_id"] == DATASET_ID
    )
    panel_rows: list[dict[str, Any]] = []
    kernel_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for backbone in statistics.BACKBONES:
        for replicate, seed in enumerate(statistics.REPLICATE_SEEDS):
            for budget in statistics.BUDGETS:
                panel = datasets.load_panel(
                    aircraft,
                    backbone,
                    int(seed),
                    int(budget),
                    registry_root=root,
                )
                w1 = crossfit_w1(
                    panel["training_values"], panel["training_labels"], seed=int(seed)
                )
                identity = (backbone, int(seed), int(budget))
                if abs(w1["baseline_score"] - frozen_scores[identity]) > 1.0e-12:
                    raise RuntimeError("recomputed baseline differs from frozen FUSED3")
                heads = {
                    head: reference[identity + (head,)] for head in statistics.HEADS
                }
                envelope = max(heads.values())
                panel_rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "backbone": backbone,
                        "replicate": int(replicate),
                        "replicate_seed": int(seed),
                        "budget": int(budget),
                        "baseline_score": float(w1["baseline_score"]),
                        "w1_score": float(w1["w1_score"]),
                        "w1_minus_baseline_score": float(
                            w1["w1_score"] - w1["baseline_score"]
                        ),
                        "linear_accuracy": heads["linear"],
                        "quadratic_accuracy": heads["quadratic"],
                        "knn_accuracy": heads["knn"],
                        "rbf_accuracy": heads["rbf"],
                        "best_head_envelope_accuracy": envelope,
                        "best_head": statistics.HEADS[
                            int(np.argmax([heads[head] for head in statistics.HEADS]))
                        ],
                        "w1_update_wall_seconds": float(
                            sum(row["w1_update_wall_seconds"] for row in w1["folds"])
                        ),
                        "empty_prototype_count_after_update": int(
                            sum(
                                row["w1_update_diagnostics"][
                                    "empty_prototype_count_after_update"
                                ]
                                for row in w1["folds"]
                            )
                        ),
                    }
                )
                for row in w1["folds"]:
                    fold_rows.append(
                        {
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "fold": int(row["fold"]),
                            "weight_condition": float(row["weight_condition"]),
                            "w1_update_wall_seconds": float(
                                row["w1_update_wall_seconds"]
                            ),
                            "empty_prototype_count_after_update": int(
                                row["w1_update_diagnostics"][
                                    "empty_prototype_count_after_update"
                                ]
                            ),
                            **{
                                f"baseline_{field}": value
                                for field, value in row["baseline"].items()
                            },
                            **{
                                f"w1_{field}": value
                                for field, value in row["w1"].items()
                            },
                        }
                    )
                if backbone in FOCAL_BACKBONES:
                    diagonal = diagonal_quadratic_accuracy(
                        panel["training_values"],
                        panel["training_labels"],
                        panel["evaluation_values"],
                        panel["evaluation_labels"],
                    )
                    kernel_rows.append(
                        {
                            "dataset_id": DATASET_ID,
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "linear_accuracy": heads["linear"],
                            "diagonal_quadratic_accuracy": float(
                                diagonal["test_accuracy"]
                            ),
                            "full_quadratic_accuracy": heads["quadratic"],
                            "diagonal_gain_over_linear_pp": 100.0
                            * (diagonal["test_accuracy"] - heads["linear"]),
                            "cross_term_gain_over_diagonal_pp": 100.0
                            * (heads["quadratic"] - diagonal["test_accuracy"]),
                            "full_gain_over_linear_pp": 100.0
                            * (heads["quadratic"] - heads["linear"]),
                            "gamma_scale": float(diagonal["gamma_scale"]),
                            "support_vector_count": int(
                                diagonal["support_vector_count"]
                            ),
                            "wall_seconds": float(diagonal["wall_seconds"]),
                        }
                    )
    if len(panel_rows) != 100 or len(fold_rows) != 500 or len(kernel_rows) != 20:
        raise RuntimeError("Aircraft W1/quadratic diagnostic grid is incomplete")

    selection_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for seed in statistics.REPLICATE_SEEDS:
            rows = [
                row
                for row in panel_rows
                if row["budget"] == budget and row["replicate_seed"] == seed
            ]
            targets = [float(row["best_head_envelope_accuracy"]) for row in rows]
            oracle = max(targets)
            for method, field in (
                ("FUSED3", "baseline_score"),
                ("FUSED3-W1", "w1_score"),
            ):
                scores = [float(row[field]) for row in rows]
                maximum = max(scores)
                selected = [
                    row
                    for row in rows
                    if abs(float(row[field]) - maximum)
                    <= statistics.SELECTION_TIE_ATOL
                ]
                selected_accuracy = float(
                    np.mean(
                        [row["best_head_envelope_accuracy"] for row in selected]
                    )
                )
                selection_rows.append(
                    {
                        "budget": int(budget),
                        "replicate_seed": int(seed),
                        "candidate_id": method,
                        "selected_backbones": [row["backbone"] for row in selected],
                        "selected_envelope_accuracy": selected_accuracy,
                        "oracle_envelope_accuracy": oracle,
                        "regret_pp": 100.0 * (oracle - selected_accuracy),
                    }
                )
                rank_rows.append(
                    {
                        "budget": int(budget),
                        "replicate_seed": int(seed),
                        "candidate_id": method,
                        "spearman_with_best_head_envelope": _spearman(scores, targets),
                    }
                )

    focal_rows: list[dict[str, Any]] = []
    for backbone in FOCAL_BACKBONES:
        for budget in statistics.BUDGETS:
            panels = [
                row
                for row in panel_rows
                if row["backbone"] == backbone and row["budget"] == budget
            ]
            folds = [
                row
                for row in fold_rows
                if row["backbone"] == backbone and row["budget"] == budget
            ]
            kernels = [
                row
                for row in kernel_rows
                if row["backbone"] == backbone and row["budget"] == budget
            ]
            focal_rows.append(
                {
                    "backbone": backbone,
                    "budget": int(budget),
                    "baseline_score": _mean(panels, "baseline_score"),
                    "w1_score": _mean(panels, "w1_score"),
                    "w1_minus_baseline_score": _mean(
                        panels, "w1_minus_baseline_score"
                    ),
                    "baseline_nearest_prototype_accuracy": _mean(
                        folds, "baseline_nearest_prototype_accuracy"
                    ),
                    "w1_nearest_prototype_accuracy": _mean(
                        folds, "w1_nearest_prototype_accuracy"
                    ),
                    "baseline_all_rivals_second_own_pass_rate": _mean(
                        folds, "baseline_all_rivals_second_own_pass_rate"
                    ),
                    "w1_all_rivals_second_own_pass_rate": _mean(
                        folds, "w1_all_rivals_second_own_pass_rate"
                    ),
                    "baseline_mean_second_own_minus_best_wrong_score": _mean(
                        folds, "baseline_mean_second_own_minus_best_wrong_score"
                    ),
                    "w1_mean_second_own_minus_best_wrong_score": _mean(
                        folds, "w1_mean_second_own_minus_best_wrong_score"
                    ),
                    "linear_accuracy": _mean(panels, "linear_accuracy"),
                    "diagonal_quadratic_accuracy": _mean(
                        kernels, "diagonal_quadratic_accuracy"
                    ),
                    "full_quadratic_accuracy": _mean(
                        kernels, "full_quadratic_accuracy"
                    ),
                    "diagonal_gain_over_linear_pp": _mean(
                        kernels, "diagonal_gain_over_linear_pp"
                    ),
                    "cross_term_gain_over_diagonal_pp": _mean(
                        kernels, "cross_term_gain_over_diagonal_pp"
                    ),
                }
            )

    selection_counts: dict[str, Counter[str]] = {
        method: Counter() for method in METHODS
    }
    for row in selection_rows:
        for backbone in row["selected_backbones"]:
            selection_counts[row["candidate_id"]][backbone] += 1
    return {
        "study": STUDY,
        "stage": STAGE,
        "diagnostic_status": STATUS,
        "dataset_id": DATASET_ID,
        "methods": list(METHODS),
        "w1_contract": {
            "initial_prototypes": "exact frozen FUSED3 fitted state",
            "weights": "exact frozen FUSED3 global diagonal weights held fixed",
            "assignment": "nearest same-class prototype under frozen weighted distance",
            "update": "one arithmetic centroid update",
            "empty_prototype_policy": "retain pre-update center",
            "weight_relearning": False,
            "oi_event_rule_changed": False,
        },
        "quadratic_contract": {
            "linear": "frozen Normalizer(l2)+LogisticRegression reference",
            "diagonal_quadratic": (
                "SVC(C=1,precomputed) with 1+2gamma<x,z>+gamma^2<x^2,z^2>"
            ),
            "full_quadratic": (
                "frozen Normalizer(l2)+SVC(poly degree=2,C=1,gamma=scale,coef0=1)"
            ),
            "scope": "DINO and OpenCLIP, both budgets, all five replicates",
        },
        "panel_rows": panel_rows,
        "fold_rows": fold_rows,
        "kernel_rows": kernel_rows,
        "selection_rows": selection_rows,
        "rank_rows": rank_rows,
        "focal_rows": focal_rows,
        "selection_counts": {
            method: dict(sorted(counts.items()))
            for method, counts in selection_counts.items()
        },
        "aggregate": {
            method: {
                "mean_regret_pp": _mean(
                    [row for row in selection_rows if row["candidate_id"] == method],
                    "regret_pp",
                ),
                "mean_spearman": _mean(
                    [row for row in rank_rows if row["candidate_id"] == method],
                    "spearman_with_best_head_envelope",
                ),
            }
            for method in METHODS
        },
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
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


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty CSV")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: _canonical_json(value)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    return output.getvalue().encode("utf-8")


def _report(summary: Mapping[str, Any]) -> str:
    baseline = summary["aggregate"]["FUSED3"]
    w1 = summary["aggregate"]["FUSED3-W1"]
    focal = {
        (row["backbone"], row["budget"]): row for row in summary["focal_rows"]
    }
    dino = focal[("dinov2-small", 64)]
    clip = focal[("openclip-vit-b-32", 64)]
    return "\n".join(
        [
            "# Aircraft FUSED3-W1 and quadratic-decomposition diagnostic",
            "",
            "Status: **post-outcome descriptive diagnostic only**. The frozen confirmation decision is unchanged.",
            "",
            "## One weighted Lloyd update",
            "",
            f"- Mean Aircraft envelope regret: FUSED3 **{baseline['mean_regret_pp']:.3f} pp**, FUSED3-W1 **{w1['mean_regret_pp']:.3f} pp**",
            f"- Mean rank Spearman: FUSED3 **{baseline['mean_spearman']:.3f}**, FUSED3-W1 **{w1['mean_spearman']:.3f}**",
            f"- Budget-64 DINO OI: **{dino['baseline_score']:.4f} -> {dino['w1_score']:.4f}**",
            f"- Budget-64 OpenCLIP OI: **{clip['baseline_score']:.4f} -> {clip['w1_score']:.4f}**",
            f"- Budget-64 DINO all-rivals second-own pass: **{100*dino['baseline_all_rivals_second_own_pass_rate']:.2f}% -> {100*dino['w1_all_rivals_second_own_pass_rate']:.2f}%**",
            f"- Budget-64 OpenCLIP all-rivals second-own pass: **{100*clip['baseline_all_rivals_second_own_pass_rate']:.2f}% -> {100*clip['w1_all_rivals_second_own_pass_rate']:.2f}%**",
            "",
            "## Quadratic decomposition at budget 64",
            "",
            f"- DINO: linear **{100*dino['linear_accuracy']:.2f}%**, diagonal quadratic **{100*dino['diagonal_quadratic_accuracy']:.2f}%**, full quadratic **{100*dino['full_quadratic_accuracy']:.2f}%**",
            f"- OpenCLIP: linear **{100*clip['linear_accuracy']:.2f}%**, diagonal quadratic **{100*clip['diagonal_quadratic_accuracy']:.2f}%**, full quadratic **{100*clip['full_quadratic_accuracy']:.2f}%**",
            f"- DINO gain: diagonal-square terms **{dino['diagonal_gain_over_linear_pp']:+.2f} pp**, remaining cross terms **{dino['cross_term_gain_over_diagonal_pp']:+.2f} pp**",
            f"- OpenCLIP gain: diagonal-square terms **{clip['diagonal_gain_over_linear_pp']:+.2f} pp**, remaining cross terms **{clip['cross_term_gain_over_diagonal_pp']:+.2f} pp**",
            "",
            "## Interpretation",
            "",
            "W1 isolates whether the learned metric was misaligned with the frozen prototypes. The diagonal/full kernel comparison isolates whether axis-aligned magnitude curvature or cross-coordinate interactions supply the quadratic advantage. Neither surface authorizes promotion or reselection.",
            "",
        ]
    )


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_diagnostic(full_artifact, registry_root)
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "panel_rows.csv": _csv_bytes(summary["panel_rows"]),
        "fold_geometry.csv": _csv_bytes(summary["fold_rows"]),
        "quadratic_decomposition.csv": _csv_bytes(summary["kernel_rows"]),
        "selection_rows.csv": _csv_bytes(summary["selection_rows"]),
        "rank_correlations.csv": _csv_bytes(summary["rank_rows"]),
        "focal_summary.csv": _csv_bytes(summary["focal_rows"]),
        "report.md": _report(summary).encode("utf-8"),
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
            raise RuntimeError("written diagnostic hash mismatch")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_bundle(args.input, args.registry_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
