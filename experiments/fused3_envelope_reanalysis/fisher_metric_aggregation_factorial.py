"""Five-dataset FUSED3 Fisher metric x aggregation x refinement factorial.

This is a post-outcome retrospective diagnostic.  It crosses a raw versus
training-fold Fisher/LDA metric, exact OI versus nearest-prototype OOF
accuracy, and balanced-median refinement off versus on.  It cannot alter the
immutable Confirmation V2 decision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
from experiments.fused3_envelope_reanalysis.metric_aggregation_2x2 import (
    _frozen_score_lookup,
)
from experiments.fused3_envelope_reanalysis.refined_fused3 import (
    refine_fused3_state,
)
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen
from experiments.scalable_relevance_kmeans.scalable import _score_kernel
from experiments.scalable_relevance_kmeans.scalable_v2 import fast_tiled_oi_score


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "fisher_metric_aggregation_factorial_protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "fisher_metric_aggregation_factorial_protocol.sha256"

STUDY = "fused3_confirmation_v2_fisher_metric_aggregation_refinement_factorial"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
METRICS = ("raw", "fisher")
AGGREGATIONS = ("exact_oi", "prototype_accuracy")
REFINEMENTS = (False, True)
FISHER_SOLVER = "svd"
FISHER_TOL = 1.0e-4
EXPECTED_K_BY_BUDGET: Mapping[int, int] = {32: 5, 64: 10}


def candidate_id(metric: str, aggregation: str, refined: bool) -> str:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS!r}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS!r}")
    if type(refined) is not bool:
        raise TypeError("refined must be a bool")
    metric_name = "RAW" if metric == "raw" else "FISHER"
    aggregation_name = "OI" if aggregation == "exact_oi" else "ACC"
    refinement_name = "R" if refined else "U"
    return f"FUSED3-{metric_name}-{aggregation_name}-{refinement_name}"


FACTORIAL_METHODS = tuple(
    candidate_id(metric, aggregation, refined)
    for metric in METRICS
    for refined in REFINEMENTS
    for aggregation in AGGREGATIONS
)
METHODS = (FROZEN, LINEAR_PROBE, *FACTORIAL_METHODS)
FACTOR_BY_METHOD = {
    candidate_id(metric, aggregation, refined): {
        "metric": metric,
        "aggregation": aggregation,
        "refined": refined,
    }
    for metric in METRICS
    for refined in REFINEMENTS
    for aggregation in AGGREGATIONS
}


def _array_digest(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _state_sha256(center: np.ndarray, scaling: np.ndarray, labels: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    _array_digest(digest, "center", center)
    _array_digest(digest, "scaling", scaling)
    _array_digest(digest, "labels", np.asarray([repr(value) for value in labels]))
    digest.update(FISHER_SOLVER.encode("ascii") + b"\0")
    digest.update(repr(FISHER_TOL).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SharedFisherTransform:
    """Detached shared Fisher transform fitted on one training fold."""

    center: np.ndarray
    scaling: np.ndarray
    classes: tuple[Any, ...]
    state_sha256: str

    @property
    def input_dimension(self) -> int:
        return int(self.scaling.shape[0])

    @property
    def output_dimension(self) -> int:
        return int(self.scaling.shape[1])

    def transform(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.input_dimension:
            raise ValueError("Fisher input has the wrong feature dimension")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Fisher input must be finite")
        transformed = np.asarray(
            (matrix.astype(np.float64) - self.center) @ self.scaling,
            dtype=np.float32,
        )
        if transformed.shape[1] != self.output_dimension or not np.all(
            np.isfinite(transformed)
        ):
            raise RuntimeError("Fisher transform emitted invalid coordinates")
        return transformed


def _first_observed_encoding(labels: np.ndarray) -> tuple[tuple[Any, ...], np.ndarray]:
    classes: list[Any] = []
    positions: dict[Any, int] = {}
    encoded = np.empty(len(labels), dtype=np.int64)
    for index, value in enumerate(labels.tolist()):
        try:
            position = positions.get(value)
        except TypeError as error:
            raise TypeError("labels must be hashable scalar values") from error
        if position is None and value not in positions:
            position = len(classes)
            positions[value] = position
            classes.append(value)
        encoded[index] = int(position)
    return tuple(classes), encoded


def fit_shared_fisher(values: Any, target: Any) -> SharedFisherTransform:
    """Fit the fixed, maximum-rank shared Fisher/LDA transform.

    Integer encoding is used only inside sklearn.  Original labels remain the
    labels supplied to FUSED3, refinement, and held-out scoring.
    """

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must be a non-empty two-dimensional matrix")
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise ValueError("target must be a scalar vector aligned with values")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite")
    classes, encoded = _first_observed_encoding(labels)
    if len(classes) < 2:
        raise ValueError("Fisher fitting requires at least two classes")
    estimator = LinearDiscriminantAnalysis(
        solver=FISHER_SOLVER,
        n_components=None,
        tol=FISHER_TOL,
    ).fit(matrix, encoded)
    maximum = min(len(classes) - 1, matrix.shape[1])
    scaling = np.asarray(estimator.scalings_[:, :maximum], dtype=np.float64)
    center = np.asarray(estimator.xbar_, dtype=np.float64)
    if scaling.ndim != 2 or scaling.shape[1] == 0:
        raise RuntimeError("Fisher fitting produced no discriminant dimensions")
    if not np.all(np.isfinite(scaling)) or not np.all(np.isfinite(center)):
        raise RuntimeError("Fisher fitting produced non-finite state")
    center = np.array(center, copy=True)
    scaling = np.array(scaling, copy=True)
    center.setflags(write=False)
    scaling.setflags(write=False)
    return SharedFisherTransform(
        center=center,
        scaling=scaling,
        classes=classes,
        state_sha256=_state_sha256(center, scaling, classes),
    )


def prototype_accuracy(
    values: Any,
    target: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
) -> float:
    """Return nearest-prototype class accuracy under one shared metric."""

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    prototypes = np.asarray(centers, dtype=np.float32)
    owner_array = np.asarray(owners)
    relevance = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or labels.shape != (len(matrix),):
        raise ValueError("prototype-accuracy values and labels must align")
    if prototypes.ndim != 2 or prototypes.shape[1] != matrix.shape[1]:
        raise ValueError("prototype centers have the wrong feature dimension")
    if owner_array.shape != (len(prototypes),):
        raise ValueError("prototype owners must align with centers")
    if relevance.shape != (matrix.shape[1],) or np.any(relevance <= 0.0):
        raise ValueError("prototype weights must be positive and feature-aligned")
    scores = _score_kernel(matrix, prototypes, relevance)
    predicted = owner_array[np.argmax(scores, axis=1)]
    return float(np.mean(predicted == labels))


def _score_fitted_state(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    centers: np.ndarray,
    owners: np.ndarray,
    weights: np.ndarray,
    memory_budget_mb: int,
    row_cap: int | None,
) -> tuple[float, float, float, float]:
    oi_started = time.perf_counter()
    oi, _diagnostics = fast_tiled_oi_score(
        values,
        labels,
        centers=centers,
        owners=owners,
        weights=weights,
        memory_budget_mb=memory_budget_mb,
        row_cap=row_cap,
    )
    oi_wall = time.perf_counter() - oi_started
    accuracy_started = time.perf_counter()
    accuracy = prototype_accuracy(
        values,
        labels,
        centers=centers,
        owners=owners,
        weights=weights,
    )
    accuracy_wall = time.perf_counter() - accuracy_started
    if not np.isfinite(oi) or not np.isfinite(accuracy):
        raise RuntimeError("factorial scoring emitted a non-finite value")
    return float(oi), float(accuracy), float(oi_wall), float(accuracy_wall)


def crossfit_factorial(values: Any, target: Any, *, seed: int) -> dict[str, Any]:
    """Execute all eight cells with shared folds and fixed FUSED3 capacity."""

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
        fisher_started = time.perf_counter()
        fisher = fit_shared_fisher(normalized[train], labels[train])
        fisher_train = fisher.transform(normalized[train])
        fisher_holdout = fisher.transform(normalized[holdout])
        fisher_wall = time.perf_counter() - fisher_started
        metric_values = {
            "raw": (normalized[train], normalized[holdout]),
            "fisher": (fisher_train, fisher_holdout),
        }
        row: dict[str, Any] = {
            "fold": int(fold),
            "fold_seed": int(fold_seed),
            "train_size": int(len(train)),
            "holdout_size": int(len(holdout)),
            "k_min": int(min(k_per_class.values())),
            "k_max": int(max(k_per_class.values())),
            "fisher_input_dimension": int(fisher.input_dimension),
            "fisher_output_dimension": int(fisher.output_dimension),
            "fisher_state_sha256": fisher.state_sha256,
            "fisher_fit_transform_wall_seconds": float(fisher_wall),
        }
        for metric in METRICS:
            train_values, holdout_values = metric_values[metric]
            selector = fused_food_screen._build_selector(
                "FUSED3", k_per_class, fold_seed
            )
            fit_started = time.perf_counter()
            selector.fit(train_values, labels[train])
            fit_wall = time.perf_counter() - fit_started
            state_before = fused_food_screen._numerical_state_sha256(selector)
            oi_u, acc_u, oi_u_wall, acc_u_wall = _score_fitted_state(
                holdout_values,
                labels[holdout],
                centers=selector.centers_,
                owners=selector.owners_,
                weights=selector.weights_,
                memory_budget_mb=int(selector.memory_budget_mb),
                row_cap=selector.row_cap,
            )
            refined = refine_fused3_state(
                train_values,
                labels[train],
                centers=selector.centers_,
                owners=selector.owners_,
                weights=selector.weights_,
                memory_budget_mb=int(selector.memory_budget_mb),
            )
            scale = np.asarray(np.sqrt(selector.weights_), dtype=np.float32)
            refined_holdout = np.asarray(holdout_values * scale, dtype=np.float32)
            oi_r, acc_r, oi_r_wall, acc_r_wall = _score_fitted_state(
                refined_holdout,
                labels[holdout],
                centers=refined["centers"],
                owners=refined["owners"],
                weights=refined["weights"],
                memory_budget_mb=int(selector.memory_budget_mb),
                row_cap=selector.row_cap,
            )
            if fused_food_screen._numerical_state_sha256(selector) != state_before:
                raise RuntimeError("factorial diagnostics mutated fitted FUSED3 state")
            summary = refined["summary"]
            prefix = metric
            row.update(
                {
                    f"{prefix}_unrefined_oi_score": oi_u,
                    f"{prefix}_unrefined_accuracy": acc_u,
                    f"{prefix}_refined_oi_score": oi_r,
                    f"{prefix}_refined_accuracy": acc_r,
                    f"{prefix}_selector_fit_wall_seconds": float(fit_wall),
                    f"{prefix}_unrefined_oi_wall_seconds": oi_u_wall,
                    f"{prefix}_unrefined_accuracy_wall_seconds": acc_u_wall,
                    f"{prefix}_refinement_wall_seconds": float(
                        refined["refinement_wall_seconds"]
                    ),
                    f"{prefix}_refined_oi_wall_seconds": oi_r_wall,
                    f"{prefix}_refined_accuracy_wall_seconds": acc_r_wall,
                    f"{prefix}_prototype_count_before": int(
                        summary["prototype_count_before"]
                    ),
                    f"{prefix}_prototype_count_after": int(
                        summary["prototype_count_after"]
                    ),
                    f"{prefix}_refinement_eligible_count": int(
                        summary["eligible_count"]
                    ),
                    f"{prefix}_refinement_applied_count": int(
                        summary["applied_count"]
                    ),
                    f"{prefix}_selector_state_sha256": state_before,
                    f"{prefix}_refined_state_sha256": str(refined["state_sha256"]),
                    f"{prefix}_fitted_state_unchanged": True,
                }
            )
        fold_rows.append(row)

    scores: dict[str, float] = {}
    method_wall: dict[str, float] = {}
    for method, factors in FACTOR_BY_METHOD.items():
        metric = str(factors["metric"])
        refined = bool(factors["refined"])
        aggregation = str(factors["aggregation"])
        state = "refined" if refined else "unrefined"
        score_field = (
            f"{metric}_{state}_oi_score"
            if aggregation == "exact_oi"
            else f"{metric}_{state}_accuracy"
        )
        score_wall_field = (
            f"{metric}_{state}_oi_wall_seconds"
            if aggregation == "exact_oi"
            else f"{metric}_{state}_accuracy_wall_seconds"
        )
        common = normalization_wall + sum(
            float(row[f"{metric}_selector_fit_wall_seconds"])
            + (float(row["fisher_fit_transform_wall_seconds"]) if metric == "fisher" else 0.0)
            + (
                float(row[f"{metric}_refinement_wall_seconds"])
                if refined
                else 0.0
            )
            for row in fold_rows
        )
        scores[method] = float(np.mean([row[score_field] for row in fold_rows]))
        method_wall[method] = float(
            common + sum(float(row[score_wall_field]) for row in fold_rows)
        )
    return {
        "scores": scores,
        "method_wall_seconds": method_wall,
        "normalization_wall_seconds": float(normalization_wall),
        "folds": fold_rows,
    }


def _factorial_effects(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["dataset_id"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        ): float(row["regret_pp"])
        for row in metrics
    }

    def ident(metric: str, aggregation: str, refined: bool) -> str:
        return candidate_id(metric, aggregation, refined)

    specs: list[tuple[str, str, list[tuple[float, str]]]] = []
    for aggregation in AGGREGATIONS:
        for refined in REFINEMENTS:
            specs.append(
                (
                    "metric_simple_effect",
                    f"fisher_minus_raw|aggregation={aggregation}|refined={refined}",
                    [
                        (1.0, ident("fisher", aggregation, refined)),
                        (-1.0, ident("raw", aggregation, refined)),
                    ],
                )
            )
    for metric in METRICS:
        for refined in REFINEMENTS:
            specs.append(
                (
                    "aggregation_simple_effect",
                    f"accuracy_minus_oi|metric={metric}|refined={refined}",
                    [
                        (1.0, ident(metric, "prototype_accuracy", refined)),
                        (-1.0, ident(metric, "exact_oi", refined)),
                    ],
                )
            )
    for metric in METRICS:
        for aggregation in AGGREGATIONS:
            specs.append(
                (
                    "refinement_simple_effect",
                    f"refined_minus_unrefined|metric={metric}|aggregation={aggregation}",
                    [
                        (1.0, ident(metric, aggregation, True)),
                        (-1.0, ident(metric, aggregation, False)),
                    ],
                )
            )
    for refined in REFINEMENTS:
        specs.append(
            (
                "metric_x_aggregation",
                f"difference_in_differences|refined={refined}",
                [
                    (1.0, ident("fisher", "prototype_accuracy", refined)),
                    (-1.0, ident("fisher", "exact_oi", refined)),
                    (-1.0, ident("raw", "prototype_accuracy", refined)),
                    (1.0, ident("raw", "exact_oi", refined)),
                ],
            )
        )
    specs.append(
        (
            "metric_x_aggregation_x_refinement",
            "three_way_difference_in_differences",
            [
                (
                    (1.0 if metric == "fisher" else -1.0)
                    * (1.0 if aggregation == "prototype_accuracy" else -1.0)
                    * (1.0 if refined else -1.0),
                    ident(metric, aggregation, refined),
                )
                for metric in METRICS
                for aggregation in AGGREGATIONS
                for refined in REFINEMENTS
            ],
        )
    )
    output: list[dict[str, Any]] = []
    for effect_type, contrast_name, terms in specs:
        contrast_rows = []
        for dataset in statistics.DATASET_IDS:
            for seed in statistics.REPLICATE_SEEDS:
                for budget in statistics.BUDGETS:
                    contrast_rows.append(
                        {
                            "dataset_id": dataset,
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "contrast_pp": float(
                                sum(
                                    coefficient
                                    * lookup[(dataset, seed, budget, method)]
                                    for coefficient, method in terms
                                )
                            ),
                        }
                    )
        output.append(
            {
                "effect_type": effect_type,
                "contrast": contrast_name,
                "sign": "negative_favors_first_named_treatment",
                **statistics.hierarchical_interval(contrast_rows),
                "status": "descriptive_post_outcome",
            }
        )
    return output


def _aircraft_focus(
    selector_rows: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selectors = _validate_comparison_rows(selector_rows, methods=METHODS)
    outcomes = _reference_lookup(references)
    dataset = "torchvision_fgvc_aircraft"
    rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for method in METHODS:
            selected_counts: dict[str, int] = {}
            score_gaps: list[float] = []
            regrets: list[float] = []
            for seed in statistics.REPLICATE_SEEDS:
                scores = {
                    backbone: float(
                        selectors[(dataset, backbone, seed, budget, method)]["score"]
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
                score_gaps.append(scores["dinov2-small"] - scores["openclip-vit-b-32"])
                envelope = {
                    backbone: max(
                        outcomes[(dataset, backbone, seed, budget, head)]
                        for head in statistics.HEADS
                    )
                    for backbone in statistics.BACKBONES
                }
                regrets.append(
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
                    "mean_regret_pp": float(np.mean(regrets)),
                    "mean_dino_minus_openclip_score": float(np.mean(score_gaps)),
                    "selection_counts": json.dumps(
                        dict(sorted(selected_counts.items())), separators=(",", ":")
                    ),
                }
            )
    return rows


def validate_design_protocol() -> tuple[str, dict[str, Any]]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SIDECAR.is_file():
        raise RuntimeError("factorial protocol and sidecar must exist")
    observed = _sha256(PROTOCOL_PATH)
    expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or observed != expected:
        raise RuntimeError("factorial protocol sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_post_outcome_before_execution":
        raise RuntimeError("factorial protocol is not frozen")
    if protocol.get("factorial", {}).get("candidate_ids") != list(FACTORIAL_METHODS):
        raise RuntimeError("factorial protocol candidate table mismatch")
    if protocol.get("grid", {}).get("factorial_row_count") != 4000:
        raise RuntimeError("factorial protocol row count mismatch")
    return observed, protocol


def analyze_factorial(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    protocol_sha256, protocol = validate_design_protocol()
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
    completed_panels = 0
    for dataset_id in statistics.DATASET_IDS:
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
                    result = crossfit_factorial(
                        panel["training_values"], panel["training_labels"], seed=int(seed)
                    )
                    expected_k = EXPECTED_K_BY_BUDGET[int(budget)]
                    if any(
                        int(row["k_min"]) != expected_k
                        or int(row["k_max"]) != expected_k
                        for row in result["folds"]
                    ):
                        raise RuntimeError("factorial FUSED3 capacity differs from frozen K")
                    raw_control = candidate_id("raw", "exact_oi", False)
                    source_score = frozen_scores[(dataset_id, backbone, int(seed), int(budget))]
                    recomputed_score = float(result["scores"][raw_control])
                    if source_score != recomputed_score:
                        raise RuntimeError("raw/unrefined/OI control failed exact parity")
                    parity_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "backbone": backbone,
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "source_score": float(source_score),
                            "recomputed_score": float(recomputed_score),
                            "exact_match": True,
                        }
                    )
                    for method in FACTORIAL_METHODS:
                        factors = FACTOR_BY_METHOD[method]
                        state_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "backbone": backbone,
                                "replicate": int(replicate),
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                "candidate_id": method,
                                "metric": factors["metric"],
                                "aggregation": factors["aggregation"],
                                "refined": factors["refined"],
                                "score": float(result["scores"][method]),
                                "total_wall_seconds": float(
                                    result["method_wall_seconds"][method]
                                ),
                            }
                        )
                    fold_rows.extend(
                        {
                            "dataset_id": dataset_id,
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            **row,
                        }
                        for row in result["folds"]
                    )
                    completed_panels += 1
                    if progress and completed_panels % 10 == 0:
                        print(f"completed {completed_panels}/500 panels", flush=True)
    if len(state_rows) != 4000 or len(parity_rows) != 500 or len(fold_rows) != 2500:
        raise RuntimeError("factorial output grid is incomplete")

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
    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "primary_estimand": "frozen_four_head_best_head_envelope_backbone_ranking",
        "head_library": list(statistics.HEADS),
        "methods": list(METHODS),
        "factorial_methods": list(FACTORIAL_METHODS),
        "design_protocol_sha256": protocol_sha256,
        "design": protocol,
        "state_rows": state_rows,
        "baseline_parity_rows": parity_rows,
        "fold_rows": fold_rows,
        "panel_metrics": envelope,
        "dataset_aggregate": dataset_aggregate,
        "overall_aggregate": overall,
        "head_panel_metrics": heads,
        "head_aggregate": head_aggregate,
        "runtime_descriptive": runtime_summary(
            comparison_rows, methods=METHODS, executed_methods=FACTORIAL_METHODS
        ),
        "regret_contrasts": [
            contrast_interval(envelope, candidate_id=method, comparator_id=comparator)
            for method in FACTORIAL_METHODS
            for comparator in (FROZEN, LINEAR_PROBE)
        ],
        "factorial_effects": _factorial_effects(envelope),
        "aircraft_focus": _aircraft_focus(comparison_rows, references),
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "post_outcome_limit": (
            "All five datasets and their outcomes were previously inspected. "
            "The target remains the complete frozen four-head envelope; focal "
            "Aircraft LDA-kNN results are mechanism evidence, not a complete target."
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
        "# FUSED3 Fisher metric x aggregation x refinement factorial",
        "",
        "Status: **post-outcome development diagnostic only**. The frozen confirmation decision is unchanged.",
        "",
        "The primary target is the complete frozen linear/quadratic/kNN/RBF best-head envelope. Negative regret contrasts favor the first-named treatment.",
        "",
        "## Overall ranking",
        "",
        "| Selector | Metric | Aggregation | Refinement | Regret (pp) | Exact best | Within 1 pp | Spearman |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["overall_aggregate"]:
        method = str(row["candidate_id"])
        if method in FACTOR_BY_METHOD:
            factors = FACTOR_BY_METHOD[method]
            metric = str(factors["metric"])
            aggregation = str(factors["aggregation"])
            refinement = "on" if factors["refined"] else "off"
        elif method == FROZEN:
            metric, aggregation, refinement = "frozen raw", "exact_oi", "off"
        else:
            metric, aggregation, refinement = "raw", "linear_probe", "n/a"
        lines.append(
            f"| {method} | {metric} | {aggregation} | {refinement} | "
            f"{row['equal_dataset_mean_regret_pp']:.3f} | "
            f"{row['equal_dataset_exact_best_rate']:.3f} | "
            f"{row['equal_dataset_within_one_pp_rate']:.3f} | "
            f"{row['equal_dataset_mean_spearman']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Factor effects on regret",
            "",
            "| Effect | Contrast | Estimate (pp) | Descriptive 95% interval |",
            "|---|---|---:|---:|",
        ]
    )
    for row in summary["factorial_effects"]:
        lines.append(
            f"| {row['effect_type']} | {row['contrast']} | {row['estimate']:+.3f} | "
            f"[{row['lower_95']:+.3f}, {row['upper_95']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Per-dataset ranking",
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
            "Positive DINO minus OpenCLIP score favors DINO, the frozen-envelope winner.",
            "",
            "| Budget | Selector | Regret (pp) | DINO - OpenCLIP | Selections |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in summary["aircraft_focus"]:
        lines.append(
            f"| {row['budget']} | {row['candidate_id']} | {row['mean_regret_pp']:.3f} | "
            f"{row['mean_dino_minus_openclip_score']:+.6f} | `{row['selection_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Runtime (descriptive separate run)",
            "",
            "| Dataset | Budget | Selector | Call/LP | Full panel/LP |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in summary["runtime_descriptive"]:
        lines.append(
            f"| {row['dataset_id']} | {row['budget']} | {row['candidate_id']} | "
            f"{row['median_call_ratio_to_lp']:.2f}x | "
            f"{row['median_full_panel_ratio_to_lp']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "These results are retrospective development evidence. This diagnostic does not promote a candidate or reinterpret Confirmation V2.",
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
    summary = analyze_factorial(full_artifact, registry_root, progress=progress)
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
        "factorial_effects.csv": _csv_bytes(summary["factorial_effects"]),
        "aircraft_focus.csv": _csv_bytes(summary["aircraft_focus"]),
        "report.md": render_report(summary).encode("utf-8"),
    }
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
    payloads["diagnostic_manifest.json"] = (
        _canonical_json(manifest) + "\n"
    ).encode("utf-8")
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
            raise RuntimeError("written factorial artifact hash mismatch")
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
