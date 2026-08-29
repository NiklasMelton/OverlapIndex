"""Tiled one-update relevance-weighted K-means with fixed-center OI scoring."""

from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import MiniBatchKMeans


DEFAULT_MEMORY_BUDGET_MB = 64
DEFAULT_WEIGHT_FLOOR = 0.05


def _validate_X_y(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(X)
    target = np.asarray(y)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("X must be numeric")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values")
    if target.ndim != 1 or target.shape[0] != values.shape[0]:
        raise ValueError("y must be a one-dimensional scalar label vector aligned with X")
    if len(dict.fromkeys(target.tolist())) < 2:
        raise ValueError("at least two classes are required")
    return values, target


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _array_digest(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _state_sha256(
    centers: np.ndarray, owners: np.ndarray, weights: np.ndarray, config: Mapping[str, Any]
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(dict(config)).encode("utf-8") + b"\0")
    _array_digest(digest, "centers", centers)
    _array_digest(digest, "owners", np.asarray(owners, dtype=str))
    _array_digest(digest, "weights", weights)
    return digest.hexdigest()


def _resolved_k(
    k_per_class: int | Mapping[Any, int], label: Any, row_count: int
) -> int:
    raw = k_per_class[label] if isinstance(k_per_class, Mapping) else k_per_class
    if type(raw) is not int or raw < 1:
        raise ValueError(f"k for class {label!r} must be a positive int")
    return min(int(raw), int(row_count))


def _score_kernel(
    X: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return upstream-equivalent ranking scores in float32 arithmetic."""

    scale = np.asarray(np.sqrt(weights), dtype=np.float32)
    X_scaled = np.asarray(X, dtype=np.float32) * scale
    centers_scaled = np.asarray(centers, dtype=np.float32) * scale
    center_norms = np.einsum("ij,ij->i", centers_scaled, centers_scaled)
    scores = np.asarray(X_scaled @ centers_scaled.T)
    scores -= center_norms[None, :] * np.float32(0.5)
    return scores.astype(np.float32, copy=False)


def _tile_row_count(
    *,
    prototype_count: int,
    feature_count: int,
    memory_budget_mb: int,
    row_cap: int | None,
) -> int:
    if type(memory_budget_mb) is not int or memory_budget_mb < 1:
        raise ValueError("memory_budget_mb must be a positive int")
    if row_cap is not None and (type(row_cap) is not int or row_cap < 1):
        raise ValueError("row_cap must be None or a positive int")
    budget_bytes = int(memory_budget_mb) * 1024 * 1024
    # One R x P float32 score tile plus two R x D float64 selected-feature
    # buffers.  The latter are conservative: implementations reuse temporaries
    # but the planner reserves both.
    bytes_per_row = max(1, 4 * int(prototype_count) + 16 * int(feature_count))
    rows = max(1, budget_bytes // bytes_per_row)
    return min(rows, row_cap) if row_cap is not None else rows


def _class_ids(owners: np.ndarray) -> dict[Any, np.ndarray]:
    return {
        label: np.flatnonzero(owners == label).astype(np.int64)
        for label in dict.fromkeys(owners.tolist())
    }


def tiled_oi_score(
    X: Any,
    y: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    row_cap: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score fixed prototypes with strict current OI single-label events."""

    values, target = _validate_X_y(X, y)
    prototype_matrix = np.asarray(centers, dtype=np.float32)
    owner_array = np.asarray(owners)
    feature_weights = np.asarray(weights, dtype=np.float64)
    if prototype_matrix.ndim != 2 or prototype_matrix.shape[1] != values.shape[1]:
        raise ValueError("centers have the wrong shape")
    if owner_array.ndim != 1 or owner_array.size != prototype_matrix.shape[0]:
        raise ValueError("owners must align with centers")
    if feature_weights.shape != (values.shape[1],) or np.any(feature_weights <= 0.0):
        raise ValueError("weights must be a positive feature vector")
    classes = list(dict.fromkeys(target.tolist()))
    ids_by_class = _class_ids(owner_array)
    if set(classes) != set(ids_by_class):
        raise ValueError("evaluation and prototype class sets must match")
    tile_rows = _tile_row_count(
        prototype_count=prototype_matrix.shape[0],
        feature_count=values.shape[1],
        memory_budget_mb=memory_budget_mb,
        row_cap=row_cap,
    )
    pair_hits = {(source, other): 0 for source in classes for other in classes if other != source}
    supports = {source: int(np.count_nonzero(target == source)) for source in classes}
    max_score_tile_rows = 0
    for source in classes:
        source_values = values[target == source]
        own_ids = ids_by_class[source]
        for start in range(0, source_values.shape[0], tile_rows):
            block = source_values[start : start + tile_rows]
            scores = _score_kernel(block, prototype_matrix, feature_weights)
            max_score_tile_rows = max(max_score_tile_rows, int(scores.shape[0]))
            own_scores = scores[:, own_ids]
            if own_ids.size < 2:
                threshold = np.full(block.shape[0], -np.inf, dtype=np.float32)
            else:
                threshold = np.partition(own_scores, kth=-2, axis=1)[:, -2]
            for other in classes:
                if other == source:
                    continue
                target_best = np.max(scores[:, ids_by_class[other]], axis=1)
                pair_hits[(source, other)] += int(np.count_nonzero(target_best > threshold))
    singleton = []
    for source in classes:
        singleton.append(
            min(
                1.0 - float(pair_hits[(source, other)]) / float(supports[source])
                for other in classes
                if other != source
            )
        )
    diagnostics = {
        "tile_row_count": int(tile_rows),
        "max_score_tile_rows": int(max_score_tile_rows),
        "prototype_count": int(prototype_matrix.shape[0]),
        "max_score_tile_bytes": int(
            max_score_tile_rows * prototype_matrix.shape[0] * np.dtype(np.float32).itemsize
        ),
        "memory_budget_mb": int(memory_budget_mb),
        "row_cap": row_cap,
    }
    return float(np.mean(singleton)), diagnostics


class ScalableOneUpdateRelevanceKMeans:
    """One K-means fit, one tiled relevance pass, and one weighted Lloyd update."""

    def __init__(
        self,
        *,
        k_per_class: int | Mapping[Any, int],
        kmeans_kwargs: Mapping[str, Any] | None = None,
        weight_floor: float = DEFAULT_WEIGHT_FLOOR,
        memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
        row_cap: int | None = None,
    ) -> None:
        if not isinstance(k_per_class, (int, Mapping)) or isinstance(k_per_class, bool):
            raise TypeError("k_per_class must be an int or mapping")
        if kmeans_kwargs is not None and not isinstance(kmeans_kwargs, Mapping):
            raise TypeError("kmeans_kwargs must be None or a mapping")
        if not isinstance(weight_floor, float) or not 0.0 < weight_floor <= 1.0:
            raise ValueError("weight_floor must be a float in (0, 1]")
        _tile_row_count(
            prototype_count=1,
            feature_count=1,
            memory_budget_mb=memory_budget_mb,
            row_cap=row_cap,
        )
        self.k_per_class = deepcopy(k_per_class)
        self.kmeans_kwargs = deepcopy(dict(kmeans_kwargs or {}))
        self.weight_floor = weight_floor
        self.memory_budget_mb = memory_budget_mb
        self.row_cap = row_cap
        self.centers_: np.ndarray | None = None
        self.owners_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] | None = None

    def fit(self, X: Any, y: Any) -> "ScalableOneUpdateRelevanceKMeans":
        values, target = _validate_X_y(X, y)
        labels = list(dict.fromkeys(target.tolist()))
        initial_started = time.perf_counter()
        center_blocks: list[np.ndarray] = []
        owners: list[Any] = []
        resolved_k: dict[Any, int] = {}
        defaults = {
            "batch_size": 256,
            "max_no_improvement": 5,
            "compute_labels": False,
            "n_init": 1,
            "init": "random",
        }
        defaults.update(self.kmeans_kwargs)
        base_random_state = defaults.pop("random_state", None)
        for position, label in enumerate(labels):
            rows = values[target == label]
            k = _resolved_k(self.k_per_class, label, rows.shape[0])
            resolved_k[label] = k
            kwargs = deepcopy(defaults)
            if base_random_state is not None:
                # Upstream supplies the same fold random_state to every
                # class-owned estimator.
                kwargs["random_state"] = int(base_random_state)
            estimator = MiniBatchKMeans(n_clusters=k, **kwargs)
            estimator.fit(rows)
            center_blocks.append(np.asarray(estimator.cluster_centers_, dtype=np.float32))
            owners.extend([label] * k)
        initial_centers = np.vstack(center_blocks).astype(np.float32, copy=False)
        owner_array = np.asarray(owners)
        initial_wall = time.perf_counter() - initial_started
        ids_by_class = _class_ids(owner_array)
        tile_rows = _tile_row_count(
            prototype_count=initial_centers.shape[0],
            feature_count=values.shape[1],
            memory_budget_mb=self.memory_budget_mb,
            row_cap=self.row_cap,
        )

        margin_started = time.perf_counter()
        margin_sum = np.zeros(values.shape[1], dtype=np.float64)
        residual_sum = np.zeros(values.shape[1], dtype=np.float64)
        max_score_tile_rows = 0
        for start in range(0, values.shape[0], tile_rows):
            block = values[start : start + tile_rows]
            block_y = target[start : start + tile_rows]
            scores = _score_kernel(
                block,
                initial_centers,
                np.ones(values.shape[1], dtype=np.float64),
            )
            max_score_tile_rows = max(max_score_tile_rows, int(scores.shape[0]))
            for label in dict.fromkeys(block_y.tolist()):
                local_rows = np.flatnonzero(block_y == label)
                own_ids = ids_by_class[label]
                wrong_ids = np.flatnonzero(owner_array != label)
                own_choice = own_ids[np.argmax(scores[local_rows][:, own_ids], axis=1)]
                wrong_choice = wrong_ids[np.argmax(scores[local_rows][:, wrong_ids], axis=1)]
                selected = block[local_rows]
                own_feature = (
                    selected - initial_centers[own_choice].astype(np.float64)
                ) ** 2
                wrong_feature = (
                    selected - initial_centers[wrong_choice].astype(np.float64)
                ) ** 2
                margin_sum += np.sum(wrong_feature - own_feature, axis=0, dtype=np.float64)
                residual_sum += np.sum(own_feature, axis=0, dtype=np.float64)
        margin = margin_sum / float(values.shape[0])
        residual = residual_sum / float(values.shape[0])
        positive_margin = np.maximum(margin, 0.0)
        epsilon = max(float(np.mean(residual)) * 1.0e-6, 1.0e-12)
        relevance = positive_margin / (positive_margin + residual + epsilon)
        raw_weights = self.weight_floor + (1.0 - self.weight_floor) * relevance
        weights = np.asarray(raw_weights / float(np.mean(raw_weights)), dtype=np.float64)
        margin_wall = time.perf_counter() - margin_started

        lloyd_started = time.perf_counter()
        sums = np.zeros_like(initial_centers, dtype=np.float64)
        counts = np.zeros(initial_centers.shape[0], dtype=np.int64)
        for label in labels:
            class_values = values[target == label]
            own_ids = ids_by_class[label]
            class_tile_rows = _tile_row_count(
                prototype_count=own_ids.size,
                feature_count=values.shape[1],
                memory_budget_mb=self.memory_budget_mb,
                row_cap=self.row_cap,
            )
            for start in range(0, class_values.shape[0], class_tile_rows):
                block = class_values[start : start + class_tile_rows]
                scores = _score_kernel(block, initial_centers[own_ids], weights)
                assigned = own_ids[np.argmax(scores, axis=1)]
                for prototype_id in np.unique(assigned):
                    assigned_rows = block[assigned == prototype_id]
                    sums[prototype_id] += np.sum(assigned_rows, axis=0, dtype=np.float64)
                    counts[prototype_id] += int(assigned_rows.shape[0])
        updated = np.asarray(initial_centers, dtype=np.float64)
        nonempty = counts > 0
        updated[nonempty] = sums[nonempty] / counts[nonempty, None]
        updated_centers = np.asarray(updated, dtype=np.float32)
        lloyd_wall = time.perf_counter() - lloyd_started
        for array in (updated_centers, owner_array, weights):
            array.setflags(write=False)
        config = {
            "k_per_class": {str(key): value for key, value in resolved_k.items()},
            "kmeans_kwargs": self.kmeans_kwargs,
            "weight_floor": self.weight_floor,
            "memory_budget_mb": self.memory_budget_mb,
            "row_cap": self.row_cap,
        }
        self.centers_ = updated_centers
        self.owners_ = owner_array
        self.weights_ = weights
        self.diagnostics_ = {
            "n_rows_fit": int(values.shape[0]),
            "n_features_fit": int(values.shape[1]),
            "n_classes_fit": len(labels),
            "prototype_count": int(updated_centers.shape[0]),
            "positive_margin_feature_count": int(np.count_nonzero(positive_margin)),
            "weight_condition": float(np.max(weights) / np.min(weights)),
            "empty_prototype_count_after_update": int(np.count_nonzero(~nonempty)),
            "initial_kmeans_pass_count": 1,
            "class_owned_estimator_fit_count": int(len(labels)),
            "weighted_lloyd_update_count": 1,
            "tile_row_count": int(tile_rows),
            "max_score_tile_rows": int(max_score_tile_rows),
            "max_score_tile_bytes": int(
                max_score_tile_rows * initial_centers.shape[0] * np.dtype(np.float32).itemsize
            ),
            "memory_budget_mb": self.memory_budget_mb,
            "initial_kmeans_wall_seconds": float(initial_wall),
            "margin_wall_seconds": float(margin_wall),
            "weighted_lloyd_wall_seconds": float(lloyd_wall),
            "state_sha256": _state_sha256(updated_centers, owner_array, weights, config),
        }
        return self

    def _require_fit(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.centers_ is None or self.owners_ is None or self.weights_ is None:
            raise ValueError("ScalableOneUpdateRelevanceKMeans is not fitted")
        return self.centers_, self.owners_, self.weights_

    def score_fixed(self, X: Any, y: Any) -> float:
        score, _diagnostics = self.score_fixed_with_diagnostics(X, y)
        return score

    def score_fixed_with_diagnostics(
        self, X: Any, y: Any
    ) -> tuple[float, dict[str, Any]]:
        centers, owners, weights = self._require_fit()
        return tiled_oi_score(
            X,
            y,
            centers=centers,
            owners=owners,
            weights=weights,
            memory_budget_mb=self.memory_budget_mb,
            row_cap=self.row_cap,
        )


__all__ = [
    "ScalableOneUpdateRelevanceKMeans",
    "tiled_oi_score",
]
