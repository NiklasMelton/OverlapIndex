"""Scalable shared-Fisher approximations for FUSED3 backbone ranking.

This retrospective development experiment keeps exact OI, frozen FUSED3
capacity, and refinement off.  It compares mean-only, diagonal-Fisher, and a
rank-16 correlated-nuisance correction with the previously executed full
Fisher/SVD diagnostic, raw FUSED3, and LP-FULL.
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
from sklearn.utils.extmath import randomized_svd

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics
from experiments.fused3_envelope_reanalysis.fisher_metric_aggregation_factorial import (
    _first_observed_encoding,
)
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
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "scalable_fisher_approximation_protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "scalable_fisher_approximation_protocol.sha256"

STUDY = "fused3_confirmation_v2_scalable_shared_fisher_approximation"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
FULL_FISHER = "FUSED3-FISHER-OI-U"
MEAN_ONLY = "FUSED3-MEAN-SUBSPACE-OI-U"
DIAGONAL = "FUSED3-DIAG-FISHER-OI-U"
LOW_RANK = "FUSED3-LR16-FISHER-OI-U"
SCALABLE_METHODS = (MEAN_ONLY, DIAGONAL, LOW_RANK)
METHODS = (FROZEN, LINEAR_PROBE, FULL_FISHER, *SCALABLE_METHODS)
APPROXIMATION_BY_METHOD = {
    MEAN_ONLY: "mean_only",
    DIAGONAL: "diagonal",
    LOW_RANK: "low_rank_16",
}
NUISANCE_RANK = 16
RANDOMIZED_POWER_ITERATIONS = 1
RANDOMIZED_OVERSAMPLES = 8
RELATIVE_VARIANCE_FLOOR = 1.0e-6
ABSOLUTE_VARIANCE_FLOOR = 1.0e-12
EXPECTED_K_BY_BUDGET: Mapping[int, int] = {32: 5, 64: 10}


def _array_digest(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _transform_sha256(
    *,
    center: np.ndarray,
    feature_scale: np.ndarray,
    nuisance_vectors: np.ndarray,
    nuisance_gains: np.ndarray,
    discriminant_basis: np.ndarray,
    classes: Sequence[Any],
    approximation: str,
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
        _array_digest(digest, name, value)
    digest.update(approximation.encode("ascii") + b"\0")
    digest.update(str(NUISANCE_RANK).encode("ascii") + b"\0")
    digest.update(str(RANDOMIZED_POWER_ITERATIONS).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ScalableFisherTransform:
    center: np.ndarray
    feature_scale: np.ndarray
    nuisance_vectors: np.ndarray
    nuisance_gains: np.ndarray
    discriminant_basis: np.ndarray
    classes: tuple[Any, ...]
    approximation: str
    variance_floor: float
    state_sha256: str

    @property
    def input_dimension(self) -> int:
        return int(self.feature_scale.shape[0])

    @property
    def output_dimension(self) -> int:
        return int(self.discriminant_basis.shape[1])

    @property
    def effective_nuisance_rank(self) -> int:
        return int(self.nuisance_vectors.shape[1])

    def transform(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.input_dimension:
            raise ValueError("scalable Fisher input has the wrong feature dimension")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("scalable Fisher input must be finite")
        standardized = np.asarray(
            (matrix.astype(np.float64) - self.center) * self.feature_scale,
            dtype=np.float64,
        )
        if self.effective_nuisance_rank:
            coordinates = standardized @ self.nuisance_vectors
            standardized = standardized + (
                coordinates * self.nuisance_gains[None, :]
            ) @ self.nuisance_vectors.T
        transformed = np.asarray(
            standardized @ self.discriminant_basis, dtype=np.float32
        )
        if transformed.shape != (len(matrix), self.output_dimension):
            raise RuntimeError("scalable Fisher transform emitted the wrong shape")
        if not np.all(np.isfinite(transformed)):
            raise RuntimeError("scalable Fisher transform emitted non-finite values")
        return transformed


def _read_only(value: Any, *, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _basis(contrasts: np.ndarray, rank: int) -> np.ndarray:
    _left, _singular, right = np.linalg.svd(
        np.asarray(contrasts, dtype=np.float64), full_matrices=False
    )
    basis = np.asarray(right[:rank].T, dtype=np.float64)
    if basis.shape != (contrasts.shape[1], rank) or not np.all(np.isfinite(basis)):
        raise RuntimeError("class-contrast SVD produced invalid basis")
    return basis


def fit_scalable_fisher_transforms(
    values: Any,
    target: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit all three shared transforms from the same training-fold moments."""

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must be a non-empty matrix")
    if labels.ndim != 1 or labels.shape[0] != matrix.shape[0]:
        raise ValueError("target must be a scalar vector aligned with values")
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite")
    classes, encoded = _first_observed_encoding(labels)
    if len(classes) < 2:
        raise ValueError("scalable Fisher fitting requires at least two classes")
    output_rank = min(len(classes) - 1, matrix.shape[1])

    moments_started = time.perf_counter()
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
        scale_reference * RELATIVE_VARIANCE_FLOOR, ABSOLUTE_VARIANCE_FLOOR
    )
    pooled_variance = np.maximum(pooled_variance_raw, variance_floor)
    class_weights = np.sqrt(counts.astype(np.float64) / float(len(matrix)))
    contrasts = (class_means - global_mean[None, :]) * class_weights[:, None]
    moments_wall = time.perf_counter() - moments_started

    transforms: dict[str, ScalableFisherTransform] = {}
    fit_wall: dict[str, float] = {}

    mean_started = time.perf_counter()
    mean_scale = np.ones(matrix.shape[1], dtype=np.float64)
    mean_basis = _basis(contrasts, output_rank)
    mean_specific_wall = time.perf_counter() - mean_started
    empty_vectors = np.empty((matrix.shape[1], 0), dtype=np.float64)
    empty_gains = np.empty(0, dtype=np.float64)
    transforms[MEAN_ONLY] = _make_transform(
        center=global_mean,
        feature_scale=mean_scale,
        nuisance_vectors=empty_vectors,
        nuisance_gains=empty_gains,
        discriminant_basis=mean_basis,
        classes=classes,
        approximation="mean_only",
        variance_floor=variance_floor,
    )
    fit_wall[MEAN_ONLY] = float(moments_wall + mean_specific_wall)

    diagonal_started = time.perf_counter()
    diagonal_scale = np.reciprocal(np.sqrt(pooled_variance))
    standardized_contrasts = contrasts * diagonal_scale[None, :]
    diagonal_basis = _basis(standardized_contrasts, output_rank)
    diagonal_specific_wall = time.perf_counter() - diagonal_started
    transforms[DIAGONAL] = _make_transform(
        center=global_mean,
        feature_scale=diagonal_scale,
        nuisance_vectors=empty_vectors,
        nuisance_gains=empty_gains,
        discriminant_basis=diagonal_basis,
        classes=classes,
        approximation="diagonal",
        variance_floor=variance_floor,
    )
    fit_wall[DIAGONAL] = float(moments_wall + diagonal_specific_wall)

    low_rank_started = time.perf_counter()
    standardized_residual = np.asarray(
        residual * diagonal_scale[None, :], dtype=np.float64
    )
    maximum_nuisance_rank = min(
        NUISANCE_RANK,
        matrix.shape[1],
        len(matrix) - len(classes),
    )
    _left, singular, right = randomized_svd(
        standardized_residual,
        n_components=maximum_nuisance_rank,
        n_iter=RANDOMIZED_POWER_ITERATIONS,
        n_oversamples=RANDOMIZED_OVERSAMPLES,
        random_state=int(seed),
        flip_sign=True,
    )
    nuisance_vectors = np.asarray(right.T, dtype=np.float64)
    degrees_of_freedom = max(1, len(matrix) - len(classes))
    eigenvalues = np.square(np.asarray(singular, dtype=np.float64)) / float(
        degrees_of_freedom
    )
    # Only deflate unusually energetic correlated directions.  Low-variance
    # directions are not amplified, which keeps the approximation stable.
    nuisance_gains = np.reciprocal(np.sqrt(np.maximum(eigenvalues, 1.0))) - 1.0
    corrected_contrasts = standardized_contrasts + (
        (standardized_contrasts @ nuisance_vectors)
        * nuisance_gains[None, :]
    ) @ nuisance_vectors.T
    low_rank_basis = _basis(corrected_contrasts, output_rank)
    low_rank_specific_wall = time.perf_counter() - low_rank_started
    transforms[LOW_RANK] = _make_transform(
        center=global_mean,
        feature_scale=diagonal_scale,
        nuisance_vectors=nuisance_vectors,
        nuisance_gains=nuisance_gains,
        discriminant_basis=low_rank_basis,
        classes=classes,
        approximation="low_rank_16",
        variance_floor=variance_floor,
    )
    fit_wall[LOW_RANK] = float(moments_wall + low_rank_specific_wall)

    return {
        "transforms": transforms,
        "fit_wall_seconds": fit_wall,
        "moments_wall_seconds": float(moments_wall),
        "output_rank": int(output_rank),
        "variance_floor": float(variance_floor),
        "variance_floor_count": int(np.count_nonzero(pooled_variance_raw < variance_floor)),
        "nuisance_eigenvalues": np.asarray(eigenvalues, dtype=np.float64),
    }


def _make_transform(
    *,
    center: np.ndarray,
    feature_scale: np.ndarray,
    nuisance_vectors: np.ndarray,
    nuisance_gains: np.ndarray,
    discriminant_basis: np.ndarray,
    classes: tuple[Any, ...],
    approximation: str,
    variance_floor: float,
) -> ScalableFisherTransform:
    center_ro = _read_only(center, dtype=np.float64)
    scale_ro = _read_only(feature_scale, dtype=np.float64)
    vectors_ro = _read_only(nuisance_vectors, dtype=np.float64)
    gains_ro = _read_only(nuisance_gains, dtype=np.float64)
    basis_ro = _read_only(discriminant_basis, dtype=np.float64)
    return ScalableFisherTransform(
        center=center_ro,
        feature_scale=scale_ro,
        nuisance_vectors=vectors_ro,
        nuisance_gains=gains_ro,
        discriminant_basis=basis_ro,
        classes=classes,
        approximation=approximation,
        variance_floor=float(variance_floor),
        state_sha256=_transform_sha256(
            center=center_ro,
            feature_scale=scale_ro,
            nuisance_vectors=vectors_ro,
            nuisance_gains=gains_ro,
            discriminant_basis=basis_ro,
            classes=classes,
            approximation=approximation,
        ),
    )


def crossfit_scalable_fisher(values: Any, target: Any, *, seed: int) -> dict[str, Any]:
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
        raw_selector = fused_food_screen._build_selector("FUSED3", k_per_class, fold_seed)
        raw_selector.fit(normalized[train], labels[train])
        raw_state = fused_food_screen._numerical_state_sha256(raw_selector)
        raw_score = float(raw_selector.score_fixed(normalized[holdout], labels[holdout]))
        if fused_food_screen._numerical_state_sha256(raw_selector) != raw_state:
            raise RuntimeError("raw parity scoring mutated FUSED3 state")

        fitted = fit_scalable_fisher_transforms(
            normalized[train], labels[train], seed=fold_seed
        )
        row: dict[str, Any] = {
            "fold": int(fold),
            "fold_seed": int(fold_seed),
            "train_size": int(len(train)),
            "holdout_size": int(len(holdout)),
            "k_min": int(min(k_per_class.values())),
            "k_max": int(max(k_per_class.values())),
            "raw_score": raw_score,
            "raw_state_sha256": raw_state,
            "moments_wall_seconds": float(fitted["moments_wall_seconds"]),
            "output_dimension": int(fitted["output_rank"]),
            "variance_floor": float(fitted["variance_floor"]),
            "variance_floor_count": int(fitted["variance_floor_count"]),
            "nuisance_eigenvalue_max": float(
                np.max(fitted["nuisance_eigenvalues"])
            ),
            "nuisance_eigenvalue_min": float(
                np.min(fitted["nuisance_eigenvalues"])
            ),
        }
        for method in SCALABLE_METHODS:
            transform = fitted["transforms"][method]
            transform_started = time.perf_counter()
            transformed_train = transform.transform(normalized[train])
            transformed_holdout = transform.transform(normalized[holdout])
            transform_wall = time.perf_counter() - transform_started
            selector = fused_food_screen._build_selector(
                "FUSED3", k_per_class, fold_seed
            )
            selector_started = time.perf_counter()
            selector.fit(transformed_train, labels[train])
            selector_fit_wall = time.perf_counter() - selector_started
            state_before = fused_food_screen._numerical_state_sha256(selector)
            score_started = time.perf_counter()
            score = float(selector.score_fixed(transformed_holdout, labels[holdout]))
            score_wall = time.perf_counter() - score_started
            if fused_food_screen._numerical_state_sha256(selector) != state_before:
                raise RuntimeError("scalable Fisher scoring mutated fitted state")
            prefix = APPROXIMATION_BY_METHOD[method]
            row.update(
                {
                    f"{prefix}_score": score,
                    f"{prefix}_fit_transform_wall_seconds": float(
                        fitted["fit_wall_seconds"][method] + transform_wall
                    ),
                    f"{prefix}_selector_fit_wall_seconds": float(selector_fit_wall),
                    f"{prefix}_score_wall_seconds": float(score_wall),
                    f"{prefix}_total_wall_seconds": float(
                        fitted["fit_wall_seconds"][method]
                        + transform_wall
                        + selector_fit_wall
                        + score_wall
                    ),
                    f"{prefix}_transform_state_sha256": transform.state_sha256,
                    f"{prefix}_selector_state_sha256": state_before,
                    f"{prefix}_effective_nuisance_rank": int(
                        transform.effective_nuisance_rank
                    ),
                    f"{prefix}_state_unchanged": True,
                }
            )
        fold_rows.append(row)
    return {
        "raw_score": float(np.mean([row["raw_score"] for row in fold_rows])),
        "scores": {
            method: float(
                np.mean(
                    [
                        row[f"{APPROXIMATION_BY_METHOD[method]}_score"]
                        for row in fold_rows
                    ]
                )
            )
            for method in SCALABLE_METHODS
        },
        "method_wall_seconds": {
            method: float(
                normalization_wall
                + sum(
                    row[f"{APPROXIMATION_BY_METHOD[method]}_total_wall_seconds"]
                    for row in fold_rows
                )
            )
            for method in SCALABLE_METHODS
        },
        "normalization_wall_seconds": float(normalization_wall),
        "folds": fold_rows,
    }


def _validate_hashed_files(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("full-Fisher artifact manifest has no files")
    for name, descriptor in files.items():
        path = root / str(name)
        if not path.is_file() or _sha256(path) != descriptor.get("sha256"):
            raise ValueError("full-Fisher artifact file hash mismatch")
        if path.stat().st_size != descriptor.get("size_bytes"):
            raise ValueError("full-Fisher artifact file size mismatch")


def load_full_fisher_rows(
    artifact: os.PathLike[str] | str,
    *,
    frozen_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(artifact)
    manifest = json.loads((root / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_status") != "completed":
        raise ValueError("full-Fisher source artifact is not completed")
    _validate_hashed_files(root, manifest)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("study") != "fused3_confirmation_v2_fisher_metric_aggregation_refinement_factorial":
        raise ValueError("unexpected full-Fisher source study")
    if summary.get("input_identity") != dict(frozen_identity):
        raise ValueError("full-Fisher source identity differs from Confirmation V2")
    rows = [
        {
            "dataset_id": str(row["dataset_id"]),
            "backbone": str(row["backbone"]),
            "replicate": int(row["replicate"]),
            "replicate_seed": int(row["replicate_seed"]),
            "budget": int(row["budget"]),
            "candidate_id": FULL_FISHER,
            "score": float(row["score"]),
            "total_wall_seconds": float(row["total_wall_seconds"]),
            "source": "validated_full_fisher_factorial",
        }
        for row in summary["state_rows"]
        if row["candidate_id"] == FULL_FISHER
    ]
    expected = {
        (dataset, backbone, seed, budget)
        for dataset in statistics.DATASET_IDS
        for backbone in statistics.BACKBONES
        for seed in statistics.REPLICATE_SEEDS
        for budget in statistics.BUDGETS
    }
    keys = {
        (row["dataset_id"], row["backbone"], row["replicate_seed"], row["budget"])
        for row in rows
    }
    if len(rows) != 500 or keys != expected:
        raise ValueError("full-Fisher source row grid is incomplete")
    return rows, {
        "manifest_sha256": _sha256(root / "diagnostic_manifest.json"),
        "summary_sha256": _sha256(root / "summary.json"),
        "design_protocol_sha256": summary["design_protocol_sha256"],
    }


def _speed_tradeoff(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = _validate_comparison_rows(rows, methods=METHODS)
    output: list[dict[str, Any]] = []
    for dataset in statistics.DATASET_IDS:
        for budget in statistics.BUDGETS:
            for method in SCALABLE_METHODS:
                call_ratios = []
                panel_ratios = []
                for seed in statistics.REPLICATE_SEEDS:
                    candidate_total = 0.0
                    full_total = 0.0
                    for backbone in statistics.BACKBONES:
                        candidate = float(
                            lookup[(dataset, backbone, seed, budget, method)][
                                "total_wall_seconds"
                            ]
                        )
                        full = float(
                            lookup[(dataset, backbone, seed, budget, FULL_FISHER)][
                                "total_wall_seconds"
                            ]
                        )
                        call_ratios.append(candidate / full)
                        candidate_total += candidate
                        full_total += full
                    panel_ratios.append(candidate_total / full_total)
                output.append(
                    {
                        "dataset_id": dataset,
                        "budget": int(budget),
                        "candidate_id": method,
                        "median_call_ratio_to_full_fisher": float(
                            np.median(call_ratios)
                        ),
                        "median_full_panel_ratio_to_full_fisher": float(
                            np.median(panel_ratios)
                        ),
                    }
                )
    return output


def validate_design_protocol() -> tuple[str, dict[str, Any]]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SIDECAR.is_file():
        raise RuntimeError("scalable Fisher protocol and sidecar must exist")
    observed = _sha256(PROTOCOL_PATH)
    expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or observed != expected:
        raise RuntimeError("scalable Fisher protocol sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_post_outcome_before_execution":
        raise RuntimeError("scalable Fisher protocol is not frozen")
    if protocol.get("candidates") != list(SCALABLE_METHODS):
        raise RuntimeError("scalable Fisher candidate table mismatch")
    return observed, protocol


def analyze_scalable_fisher(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    full_fisher_artifact: os.PathLike[str] | str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    protocol_sha256, protocol = validate_design_protocol()
    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    source_selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    frozen_scores = _frozen_score_lookup(source_selectors)
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
    full_fisher_rows, full_fisher_identity = load_full_fisher_rows(
        full_fisher_artifact, frozen_identity=input_identity
    )
    root = Path(registry_root)
    registry = json.loads((root / "audited_registry.json").read_text(encoding="utf-8"))
    registry_rows = {str(row["dataset_id"]): row for row in registry["datasets"]}
    if set(registry_rows) != set(statistics.DATASET_IDS):
        raise RuntimeError("audited registry dataset set mismatch")
    _reference_lookup(references)

    state_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    completed = 0
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
                    result = crossfit_scalable_fisher(
                        panel["training_values"], panel["training_labels"], seed=int(seed)
                    )
                    expected_k = EXPECTED_K_BY_BUDGET[int(budget)]
                    if any(
                        int(row["k_min"]) != expected_k
                        or int(row["k_max"]) != expected_k
                        for row in result["folds"]
                    ):
                        raise RuntimeError("scalable Fisher capacity differs from frozen K")
                    source_score = frozen_scores[(dataset_id, backbone, int(seed), int(budget))]
                    if float(result["raw_score"]) != source_score:
                        raise RuntimeError("scalable Fisher raw parity control failed")
                    parity_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "backbone": backbone,
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "source_score": float(source_score),
                            "recomputed_score": float(result["raw_score"]),
                            "exact_match": True,
                        }
                    )
                    for method in SCALABLE_METHODS:
                        state_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "backbone": backbone,
                                "replicate": int(replicate),
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                "candidate_id": method,
                                "approximation": APPROXIMATION_BY_METHOD[method],
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
                    completed += 1
                    if progress and completed % 10 == 0:
                        print(f"completed {completed}/500 panels", flush=True)
    if len(state_rows) != 1500 or len(parity_rows) != 500 or len(fold_rows) != 2500:
        raise RuntimeError("scalable Fisher output grid is incomplete")

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
    comparison_rows = [*source_rows, *full_fisher_rows, *state_rows]
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
        "executed_methods": list(SCALABLE_METHODS),
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
            comparison_rows, methods=METHODS, executed_methods=SCALABLE_METHODS
        ),
        "speed_tradeoff": _speed_tradeoff(comparison_rows),
        "regret_contrasts": [
            contrast_interval(envelope, candidate_id=method, comparator_id=comparator)
            for method in SCALABLE_METHODS
            for comparator in (FROZEN, LINEAR_PROBE, FULL_FISHER)
        ],
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "post_outcome_limit": (
            "All five datasets and outcomes were previously inspected. This is "
            "retrospective development evidence and cannot confirm a candidate."
        ),
        "input_hashes": dict(verified["input_hashes"]),
        "input_identity": input_identity,
        "full_fisher_source_identity": full_fisher_identity,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Scalable shared-Fisher approximation diagnostic",
        "",
        "Status: **post-outcome development diagnostic only**. The frozen confirmation decision is unchanged.",
        "",
        "Exact OI and refinement-off are fixed. The scalable candidates differ only in the training-fold shared metric.",
        "",
        "## Overall best-head-envelope ranking",
        "",
        "| Selector | Metric | Regret (pp) | Exact best | Within 1 pp | Spearman |",
        "|---|---|---:|---:|---:|---:|",
    ]
    metric_names = {
        FROZEN: "raw FUSED3",
        LINEAR_PROBE: "linear probe",
        FULL_FISHER: "full SVD Fisher",
        MEAN_ONLY: "class-mean contrast",
        DIAGONAL: "diagonal pooled Fisher",
        LOW_RANK: "diagonal + rank-16 nuisance",
    }
    for row in summary["overall_aggregate"]:
        lines.append(
            f"| {row['candidate_id']} | {metric_names[row['candidate_id']]} | "
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
            "## Regret contrasts",
            "",
            "Negative estimates favor the scalable candidate.",
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
            "## Runtime relative to full Fisher",
            "",
            "Separate-run descriptive timings; lower is faster.",
            "",
            "| Dataset | Budget | Candidate | Call/full Fisher | Panel/full Fisher |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in summary["speed_tradeoff"]:
        lines.append(
            f"| {row['dataset_id']} | {row['budget']} | {row['candidate_id']} | "
            f"{row['median_call_ratio_to_full_fisher']:.2f}x | "
            f"{row['median_full_panel_ratio_to_full_fisher']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "No promotion, reselection, or confirmation reinterpretation is made.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    full_fisher_artifact: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_scalable_fisher(
        full_artifact,
        registry_root,
        full_fisher_artifact,
        progress=progress,
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
        "speed_tradeoff.csv": _csv_bytes(summary["speed_tradeoff"]),
        "regret_contrasts.csv": _csv_bytes(summary["regret_contrasts"]),
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
            raise RuntimeError("written scalable Fisher artifact hash mismatch")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--full-fisher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_bundle(
        args.input,
        args.registry_root,
        args.full_fisher,
        args.output,
        progress=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
