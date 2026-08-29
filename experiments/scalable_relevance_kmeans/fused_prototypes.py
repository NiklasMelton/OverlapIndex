"""Experiment-local fused class-batched prototype fitter.

The implementation preserves class ownership: a row is compared only with
prototypes belonging to its own class.  It replaces one estimator object per
class with padded batched assignment/update operations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from experiments.scalable_relevance_kmeans.scalable import (
    DEFAULT_MEMORY_BUDGET_MB,
    DEFAULT_WEIGHT_FLOOR,
    _class_ids,
    _resolved_k,
    _score_kernel,
    _tile_row_count,
)
from experiments.scalable_relevance_kmeans.scalable_v2 import (
    _margin_indices,
    _state_sha256,
    _validate_X_y_float32,
    fast_tiled_oi_score,
)


@dataclass(frozen=True)
class PrototypeFit:
    """Detached result from a class-owned fixed-Lloyd prototype fit."""

    centers: np.ndarray
    owners: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class _PackedClasses:
    values: np.ndarray
    row_mask: np.ndarray
    center_mask: np.ndarray
    initial_centers: np.ndarray
    labels: tuple[Any, ...]
    resolved_k: tuple[int, ...]


def _validate_iterations(iterations: int) -> int:
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive int")
    return iterations


def _pack_classes(
    values: np.ndarray,
    target: np.ndarray,
    *,
    k_per_class: int | Mapping[Any, int],
    seed: int,
) -> _PackedClasses:
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    labels = tuple(dict.fromkeys(target.tolist()))
    class_rows = [values[target == label] for label in labels]
    resolved_k = tuple(
        _resolved_k(k_per_class, label, rows.shape[0])
        for label, rows in zip(labels, class_rows)
    )
    max_rows = max(rows.shape[0] for rows in class_rows)
    max_k = max(resolved_k)
    feature_count = values.shape[1]
    packed = np.zeros((len(labels), max_rows, feature_count), dtype=np.float32)
    row_mask = np.zeros((len(labels), max_rows), dtype=bool)
    center_mask = np.zeros((len(labels), max_k), dtype=bool)
    centers = np.zeros((len(labels), max_k, feature_count), dtype=np.float32)
    for position, (rows, k) in enumerate(zip(class_rows, resolved_k)):
        packed[position, : rows.shape[0]] = rows
        row_mask[position, : rows.shape[0]] = True
        center_mask[position, :k] = True
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(position), int(rows.shape[0]), int(k)])
        )
        selected = np.asarray(rng.choice(rows.shape[0], size=k, replace=False), dtype=np.int64)
        centers[position, :k] = rows[selected]
    return _PackedClasses(
        values=packed,
        row_mask=row_mask,
        center_mask=center_mask,
        initial_centers=centers,
        labels=labels,
        resolved_k=resolved_k,
    )


def _fixed_lloyd_batch(
    values: np.ndarray,
    row_mask: np.ndarray,
    center_mask: np.ndarray,
    initial_centers: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run fixed full-batch Lloyd updates for one or more packed classes."""

    centers = np.asarray(initial_centers, dtype=np.float32).copy()
    prototype_axis = np.arange(centers.shape[1], dtype=np.int64)
    assignments = np.zeros(row_mask.shape, dtype=np.int64)
    counts = np.zeros(center_mask.shape, dtype=np.int64)
    for _ in range(iterations):
        # The omitted ||x||^2 term is constant across prototypes.  This is the
        # same float32 ranking kernel used by the scalable OI implementation.
        scores = np.matmul(values, np.swapaxes(centers, 1, 2))
        center_norms = np.einsum("ckd,ckd->ck", centers, centers)
        scores -= center_norms[:, None, :] * np.float32(0.5)
        scores = np.where(center_mask[:, None, :], scores, -np.inf)
        assignments = np.argmax(scores, axis=2).astype(np.int64, copy=False)
        one_hot = (
            (assignments[:, :, None] == prototype_axis[None, None, :])
            & row_mask[:, :, None]
            & center_mask[:, None, :]
        )
        counts = np.sum(one_hot, axis=1, dtype=np.int64)
        sums = np.matmul(
            np.swapaxes(one_hot.astype(np.float32), 1, 2), values
        )
        nonempty = counts > 0
        updated = np.divide(
            sums,
            np.maximum(counts, 1)[:, :, None],
            dtype=np.float32,
        )
        centers = np.where(nonempty[:, :, None], updated, centers)
        centers = np.where(center_mask[:, :, None], centers, np.float32(0.0))
    return centers, assignments, counts


def _flatten_fit(
    packed: _PackedClasses,
    centers: np.ndarray,
    counts: np.ndarray,
    *,
    algorithm: str,
    iterations: int,
    wall_seconds: float,
) -> PrototypeFit:
    center_blocks = [
        np.asarray(centers[position, :k], dtype=np.float32)
        for position, k in enumerate(packed.resolved_k)
    ]
    owner_values: list[Any] = []
    for label, k in zip(packed.labels, packed.resolved_k):
        owner_values.extend([label] * k)
    flat_centers = np.vstack(center_blocks).astype(np.float32, copy=False)
    owners = np.asarray(owner_values)
    empty = sum(
        int(np.count_nonzero(counts[position, :k] == 0))
        for position, k in enumerate(packed.resolved_k)
    )
    for array in (flat_centers, owners):
        array.setflags(write=False)
    diagnostics = {
        "algorithm": algorithm,
        "class_count": len(packed.labels),
        "prototype_count": int(flat_centers.shape[0]),
        "feature_count": int(flat_centers.shape[1]),
        "fixed_lloyd_iterations": int(iterations),
        "max_rows_per_class": int(packed.values.shape[1]),
        "max_prototypes_per_class": int(packed.initial_centers.shape[1]),
        "packed_values_bytes": int(packed.values.nbytes),
        "empty_prototype_count": int(empty),
        "wall_seconds": float(wall_seconds),
    }
    return PrototypeFit(flat_centers, owners, diagnostics)


def fit_looped_class_prototypes(
    X: Any,
    y: Any,
    *,
    k_per_class: int | Mapping[Any, int],
    seed: int,
    iterations: int = 3,
) -> PrototypeFit:
    """Reference implementation using one fixed-Lloyd batch per class."""

    steps = _validate_iterations(iterations)
    values, target = _validate_X_y_float32(X, y)
    packed = _pack_classes(values, target, k_per_class=k_per_class, seed=seed)
    started = time.perf_counter()
    centers = np.zeros_like(packed.initial_centers)
    counts = np.zeros_like(packed.center_mask, dtype=np.int64)
    for position in range(len(packed.labels)):
        local_centers, _assignments, local_counts = _fixed_lloyd_batch(
            packed.values[position : position + 1],
            packed.row_mask[position : position + 1],
            packed.center_mask[position : position + 1],
            packed.initial_centers[position : position + 1],
            iterations=steps,
        )
        centers[position] = local_centers[0]
        counts[position] = local_counts[0]
    wall = time.perf_counter() - started
    return _flatten_fit(
        packed,
        centers,
        counts,
        algorithm="looped_fixed_lloyd",
        iterations=steps,
        wall_seconds=wall,
    )


def fit_fused_class_prototypes(
    X: Any,
    y: Any,
    *,
    k_per_class: int | Mapping[Any, int],
    seed: int,
    iterations: int = 3,
) -> PrototypeFit:
    """Fit all class-owned prototypes in one padded batched Lloyd kernel."""

    steps = _validate_iterations(iterations)
    values, target = _validate_X_y_float32(X, y)
    packed = _pack_classes(values, target, k_per_class=k_per_class, seed=seed)
    started = time.perf_counter()
    centers, _assignments, counts = _fixed_lloyd_batch(
        packed.values,
        packed.row_mask,
        packed.center_mask,
        packed.initial_centers,
        iterations=steps,
    )
    wall = time.perf_counter() - started
    return _flatten_fit(
        packed,
        centers,
        counts,
        algorithm="fused_fixed_lloyd",
        iterations=steps,
        wall_seconds=wall,
    )


class FusedFastRelevanceKMeans:
    """Fused prototypes plus capped relevance and the fast fixed OI scorer."""

    def __init__(
        self,
        *,
        k_per_class: int | Mapping[Any, int],
        seed: int,
        prototype_iterations: int = 3,
        prototype_backend: str = "fused",
        margin_rows_per_class: int = 32,
        weight_floor: float = DEFAULT_WEIGHT_FLOOR,
        memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
        row_cap: int | None = None,
    ) -> None:
        if not isinstance(k_per_class, (int, Mapping)) or isinstance(k_per_class, bool):
            raise TypeError("k_per_class must be an int or mapping")
        if type(seed) is not int:
            raise TypeError("seed must be an int")
        _validate_iterations(prototype_iterations)
        if prototype_backend not in {"fused", "looped"}:
            raise ValueError("prototype_backend must be 'fused' or 'looped'")
        if type(margin_rows_per_class) is not int or margin_rows_per_class < 1:
            raise ValueError("margin_rows_per_class must be a positive int")
        if not isinstance(weight_floor, float) or not 0.0 < weight_floor <= 1.0:
            raise ValueError("weight_floor must be a float in (0, 1]")
        _tile_row_count(
            prototype_count=1,
            feature_count=1,
            memory_budget_mb=memory_budget_mb,
            row_cap=row_cap,
        )
        self.k_per_class = k_per_class
        self.seed = seed
        self.prototype_iterations = prototype_iterations
        self.prototype_backend = prototype_backend
        self.margin_rows_per_class = margin_rows_per_class
        self.weight_floor = weight_floor
        self.memory_budget_mb = memory_budget_mb
        self.row_cap = row_cap
        self.centers_: np.ndarray | None = None
        self.owners_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] | None = None

    def fit(self, X: Any, y: Any) -> "FusedFastRelevanceKMeans":
        values, target = _validate_X_y_float32(X, y)
        labels = list(dict.fromkeys(target.tolist()))
        prototype_started = time.perf_counter()
        prototype_fitter = (
            fit_fused_class_prototypes
            if self.prototype_backend == "fused"
            else fit_looped_class_prototypes
        )
        prototype_fit = prototype_fitter(
            values,
            target,
            k_per_class=self.k_per_class,
            seed=self.seed,
            iterations=self.prototype_iterations,
        )
        prototype_wall = time.perf_counter() - prototype_started
        centers = prototype_fit.centers
        owners = prototype_fit.owners
        ids_by_class = _class_ids(owners)
        tile_rows = _tile_row_count(
            prototype_count=centers.shape[0],
            feature_count=values.shape[1],
            memory_budget_mb=self.memory_budget_mb,
            row_cap=self.row_cap,
        )

        margin_started = time.perf_counter()
        chosen = _margin_indices(
            target, labels, self.margin_rows_per_class, self.seed
        )
        margin_values = values[chosen]
        margin_target = target[chosen]
        margin_sum = np.zeros(values.shape[1], dtype=np.float64)
        residual_sum = np.zeros(values.shape[1], dtype=np.float64)
        unit_weights = np.ones(values.shape[1], dtype=np.float64)
        for start in range(0, margin_values.shape[0], tile_rows):
            block = margin_values[start : start + tile_rows]
            block_target = margin_target[start : start + tile_rows]
            scores = _score_kernel(block, centers, unit_weights)
            for label in dict.fromkeys(block_target.tolist()):
                local = np.flatnonzero(block_target == label)
                own_ids = ids_by_class[label]
                wrong_ids = np.flatnonzero(owners != label)
                own_choice = own_ids[np.argmax(scores[local][:, own_ids], axis=1)]
                wrong_choice = wrong_ids[np.argmax(scores[local][:, wrong_ids], axis=1)]
                selected = block[local]
                own_feature = np.square(
                    selected - centers[own_choice], dtype=np.float32
                )
                wrong_feature = np.square(
                    selected - centers[wrong_choice], dtype=np.float32
                )
                margin_sum += np.sum(
                    wrong_feature.astype(np.float64) - own_feature.astype(np.float64),
                    axis=0,
                    dtype=np.float64,
                )
                residual_sum += np.sum(own_feature, axis=0, dtype=np.float64)
        denominator = float(margin_values.shape[0])
        margin = margin_sum / denominator
        residual = residual_sum / denominator
        positive_margin = np.maximum(margin, 0.0)
        epsilon = max(float(np.mean(residual)) * 1.0e-6, 1.0e-12)
        relevance = positive_margin / (positive_margin + residual + epsilon)
        raw_weights = self.weight_floor + (1.0 - self.weight_floor) * relevance
        weights = np.asarray(raw_weights / float(np.mean(raw_weights)), dtype=np.float64)
        margin_wall = time.perf_counter() - margin_started

        config = {
            "algorithm": "fused_fixed_lloyd_capped_relevance_no_final_lloyd",
            "k_per_class": (
                {str(key): int(value) for key, value in self.k_per_class.items()}
                if isinstance(self.k_per_class, Mapping)
                else int(self.k_per_class)
            ),
            "seed": self.seed,
            "prototype_iterations": self.prototype_iterations,
            "prototype_backend": self.prototype_backend,
            "margin_rows_per_class": self.margin_rows_per_class,
            "weight_floor": self.weight_floor,
            "memory_budget_mb": self.memory_budget_mb,
            "row_cap": self.row_cap,
        }
        centers = np.asarray(centers, dtype=np.float32)
        owners = np.asarray(owners)
        for array in (centers, owners, weights):
            array.setflags(write=False)
        self.centers_ = centers
        self.owners_ = owners
        self.weights_ = weights
        self.diagnostics_ = {
            "n_rows_fit": int(values.shape[0]),
            "n_features_fit": int(values.shape[1]),
            "n_classes_fit": len(labels),
            "prototype_count": int(centers.shape[0]),
            "prototype_fit": dict(prototype_fit.diagnostics),
            "prototype_fit_wall_seconds": float(prototype_wall),
            "margin_wall_seconds": float(margin_wall),
            "margin_row_count": int(margin_values.shape[0]),
            "weight_condition": float(np.max(weights) / np.min(weights)),
            "positive_margin_feature_count": int(np.count_nonzero(positive_margin)),
            "weighted_lloyd_update_count": 0,
            "state_sha256": _state_sha256(centers, owners, weights, config),
        }
        return self

    def _require_fit(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.centers_ is None or self.owners_ is None or self.weights_ is None:
            raise ValueError("FusedFastRelevanceKMeans is not fitted")
        return self.centers_, self.owners_, self.weights_

    def score_fixed(self, X: Any, y: Any) -> float:
        score, _ = self.score_fixed_with_diagnostics(X, y)
        return score

    def score_fixed_with_diagnostics(
        self, X: Any, y: Any
    ) -> tuple[float, dict[str, Any]]:
        centers, owners, weights = self._require_fit()
        return fast_tiled_oi_score(
            X,
            y,
            centers=centers,
            owners=owners,
            weights=weights,
            memory_budget_mb=self.memory_budget_mb,
            row_cap=self.row_cap,
        )


__all__ = [
    "FusedFastRelevanceKMeans",
    "PrototypeFit",
    "fit_fused_class_prototypes",
    "fit_looped_class_prototypes",
]
