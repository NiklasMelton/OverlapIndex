"""Speed-focused variants of the scalable relevance-weighted K-means selector.

This module is experiment-local.  It deliberately leaves ``scalable.py`` and
the public :mod:`overlapindex` package untouched so the v1 Food-101 evidence
remains reproducible.
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from experiments.scalable_relevance_kmeans.scalable import (
    DEFAULT_MEMORY_BUDGET_MB,
    DEFAULT_WEIGHT_FLOOR,
    _class_ids,
    _resolved_k,
    _score_kernel,
    _tile_row_count,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def _validate_X_y_float32(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(X)
    target = np.asarray(y)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("X must be numeric")
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values")
    if target.ndim != 1 or target.shape[0] != values.shape[0]:
        raise ValueError("y must be a one-dimensional scalar label vector aligned with X")
    if len(dict.fromkeys(target.tolist())) < 2:
        raise ValueError("at least two classes are required")
    return values, target


def fast_tiled_oi_score(
    X: Any,
    y: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    row_cap: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Apply the exact OI event rule with one score matmul per row tile.

    V1 groups evaluation rows by source class before scoring.  That repeats the
    matrix-multiplication launch once per class.  Here each aligned row tile is
    scored once, then class masks are applied to that already-computed score
    matrix.  Pair hit counts and strict ``>`` tie behavior are unchanged.
    """

    values, target = _validate_X_y_float32(X, y)
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
    pair_hits = {
        (source, other): 0
        for source in classes
        for other in classes
        if other != source
    }
    supports = {source: int(np.count_nonzero(target == source)) for source in classes}
    max_score_tile_rows = 0
    matmul_count = 0
    for start in range(0, values.shape[0], tile_rows):
        block = values[start : start + tile_rows]
        block_target = target[start : start + tile_rows]
        scores = _score_kernel(block, prototype_matrix, feature_weights)
        matmul_count += 1
        max_score_tile_rows = max(max_score_tile_rows, int(scores.shape[0]))
        for source in dict.fromkeys(block_target.tolist()):
            local = np.flatnonzero(block_target == source)
            own_ids = ids_by_class[source]
            own_scores = scores[local][:, own_ids]
            if own_ids.size < 2:
                threshold = np.full(local.size, -np.inf, dtype=np.float32)
            else:
                threshold = np.partition(own_scores, kth=-2, axis=1)[:, -2]
            for other in classes:
                if other == source:
                    continue
                target_best = np.max(scores[local][:, ids_by_class[other]], axis=1)
                pair_hits[(source, other)] += int(np.count_nonzero(target_best > threshold))
    singleton = [
        min(
            1.0 - float(pair_hits[(source, other)]) / float(supports[source])
            for other in classes
            if other != source
        )
        for source in classes
    ]
    diagnostics = {
        "tile_row_count": int(tile_rows),
        "max_score_tile_rows": int(max_score_tile_rows),
        "prototype_count": int(prototype_matrix.shape[0]),
        "matmul_count": int(matmul_count),
        "max_score_tile_bytes": int(
            max_score_tile_rows * prototype_matrix.shape[0] * np.dtype(np.float32).itemsize
        ),
        "memory_budget_mb": int(memory_budget_mb),
        "row_cap": row_cap,
        "scoring_algorithm": "single_matmul_per_row_tile",
    }
    return float(np.mean(singleton)), diagnostics


def _margin_indices(
    target: np.ndarray,
    labels: list[Any],
    cap: int | None,
    random_state: int | None,
) -> np.ndarray:
    if cap is None:
        return np.arange(target.shape[0], dtype=np.int64)
    chosen: list[np.ndarray] = []
    base_seed = 0 if random_state is None else int(random_state)
    for position, label in enumerate(labels):
        local = np.flatnonzero(target == label).astype(np.int64)
        if local.size <= cap:
            chosen.append(local)
            continue
        rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, int(position), int(local.size), int(cap)])
        )
        selected = np.sort(rng.choice(local, size=cap, replace=False))
        chosen.append(np.asarray(selected, dtype=np.int64))
    return np.sort(np.concatenate(chosen))


class FastScalableRelevanceKMeans:
    """Float32 scalable selector with closed, predeclared speed controls."""

    def __init__(
        self,
        *,
        k_per_class: int | Mapping[Any, int],
        kmeans_kwargs: Mapping[str, Any] | None = None,
        weight_floor: float = DEFAULT_WEIGHT_FLOOR,
        memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
        row_cap: int | None = None,
        fast_scoring: bool = True,
        margin_rows_per_class: int | None = None,
        lloyd_update: bool = True,
    ) -> None:
        if not isinstance(k_per_class, (int, Mapping)) or isinstance(k_per_class, bool):
            raise TypeError("k_per_class must be an int or mapping")
        if kmeans_kwargs is not None and not isinstance(kmeans_kwargs, Mapping):
            raise TypeError("kmeans_kwargs must be None or a mapping")
        if not isinstance(weight_floor, float) or not 0.0 < weight_floor <= 1.0:
            raise ValueError("weight_floor must be a float in (0, 1]")
        if type(fast_scoring) is not bool or type(lloyd_update) is not bool:
            raise TypeError("fast_scoring and lloyd_update must be strict bools")
        if margin_rows_per_class is not None and (
            type(margin_rows_per_class) is not int or margin_rows_per_class < 1
        ):
            raise ValueError("margin_rows_per_class must be None or a positive int")
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
        self.fast_scoring = fast_scoring
        self.margin_rows_per_class = margin_rows_per_class
        self.lloyd_update = lloyd_update
        self.centers_: np.ndarray | None = None
        self.owners_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] | None = None

    def fit(self, X: Any, y: Any) -> "FastScalableRelevanceKMeans":
        values, target = _validate_X_y_float32(X, y)
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
        for label in labels:
            rows = values[target == label]
            k = _resolved_k(self.k_per_class, label, rows.shape[0])
            resolved_k[label] = k
            kwargs = deepcopy(defaults)
            if base_random_state is not None:
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
        selected_indices = _margin_indices(
            target, labels, self.margin_rows_per_class, base_random_state
        )
        margin_values = values[selected_indices]
        margin_target = target[selected_indices]
        margin_sum = np.zeros(values.shape[1], dtype=np.float64)
        residual_sum = np.zeros(values.shape[1], dtype=np.float64)
        max_score_tile_rows = 0
        for start in range(0, margin_values.shape[0], tile_rows):
            block = margin_values[start : start + tile_rows]
            block_y = margin_target[start : start + tile_rows]
            scores = _score_kernel(
                block, initial_centers, np.ones(values.shape[1], dtype=np.float64)
            )
            max_score_tile_rows = max(max_score_tile_rows, int(scores.shape[0]))
            for label in dict.fromkeys(block_y.tolist()):
                local = np.flatnonzero(block_y == label)
                own_ids = ids_by_class[label]
                wrong_ids = np.flatnonzero(owner_array != label)
                own_choice = own_ids[np.argmax(scores[local][:, own_ids], axis=1)]
                wrong_choice = wrong_ids[np.argmax(scores[local][:, wrong_ids], axis=1)]
                selected = block[local]
                own_feature = np.square(
                    selected - initial_centers[own_choice], dtype=np.float32
                )
                wrong_feature = np.square(
                    selected - initial_centers[wrong_choice], dtype=np.float32
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

        lloyd_started = time.perf_counter()
        if self.lloyd_update:
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
                        sums[prototype_id] += np.sum(
                            assigned_rows, axis=0, dtype=np.float64
                        )
                        counts[prototype_id] += int(assigned_rows.shape[0])
            updated = np.asarray(initial_centers, dtype=np.float64)
            nonempty = counts > 0
            updated[nonempty] = sums[nonempty] / counts[nonempty, None]
            updated_centers = np.asarray(updated, dtype=np.float32)
        else:
            nonempty = np.ones(initial_centers.shape[0], dtype=bool)
            updated_centers = initial_centers.copy()
        lloyd_wall = time.perf_counter() - lloyd_started
        center_movement = float(
            np.sqrt(
                np.mean(
                    np.square(
                        updated_centers.astype(np.float64)
                        - initial_centers.astype(np.float64)
                    )
                )
            )
        )
        config = {
            "k_per_class": {str(key): value for key, value in resolved_k.items()},
            "kmeans_kwargs": self.kmeans_kwargs,
            "weight_floor": self.weight_floor,
            "memory_budget_mb": self.memory_budget_mb,
            "row_cap": self.row_cap,
            "fast_scoring": self.fast_scoring,
            "margin_rows_per_class": self.margin_rows_per_class,
            "lloyd_update": self.lloyd_update,
        }
        for array in (updated_centers, owner_array, weights):
            array.setflags(write=False)
        self.centers_ = updated_centers
        self.owners_ = owner_array
        self.weights_ = weights
        self.diagnostics_ = {
            "n_rows_fit": int(values.shape[0]),
            "n_features_fit": int(values.shape[1]),
            "n_classes_fit": len(labels),
            "prototype_count": int(updated_centers.shape[0]),
            "input_dtype": str(values.dtype),
            "margin_row_count": int(margin_values.shape[0]),
            "margin_rows_per_class": self.margin_rows_per_class,
            "positive_margin_feature_count": int(np.count_nonzero(positive_margin)),
            "weight_condition": float(np.max(weights) / np.min(weights)),
            "empty_prototype_count_after_update": int(np.count_nonzero(~nonempty)),
            "initial_kmeans_pass_count": 1,
            "class_owned_estimator_fit_count": int(len(labels)),
            "weighted_lloyd_update_count": int(self.lloyd_update),
            "center_update_rms": center_movement,
            "fast_scoring": self.fast_scoring,
            "tile_row_count": int(tile_rows),
            "max_score_tile_rows": int(max_score_tile_rows),
            "max_score_tile_bytes": int(
                max_score_tile_rows
                * initial_centers.shape[0]
                * np.dtype(np.float32).itemsize
            ),
            "memory_budget_mb": self.memory_budget_mb,
            "initial_kmeans_wall_seconds": float(initial_wall),
            "margin_wall_seconds": float(margin_wall),
            "weighted_lloyd_wall_seconds": float(lloyd_wall),
            "state_sha256": _state_sha256(
                updated_centers, owner_array, weights, config
            ),
        }
        return self

    def _require_fit(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.centers_ is None or self.owners_ is None or self.weights_ is None:
            raise ValueError("FastScalableRelevanceKMeans is not fitted")
        return self.centers_, self.owners_, self.weights_

    def score_fixed(self, X: Any, y: Any) -> float:
        score, _ = self.score_fixed_with_diagnostics(X, y)
        return score

    def score_fixed_with_diagnostics(
        self, X: Any, y: Any
    ) -> tuple[float, dict[str, Any]]:
        centers, owners, weights = self._require_fit()
        if self.fast_scoring:
            return fast_tiled_oi_score(
                X,
                y,
                centers=centers,
                owners=owners,
                weights=weights,
                memory_budget_mb=self.memory_budget_mb,
                row_cap=self.row_cap,
            )
        # Import lazily to make the slow v1-style scorer an explicit diagnostic.
        from experiments.scalable_relevance_kmeans.scalable import tiled_oi_score

        return tiled_oi_score(
            X,
            y,
            centers=centers,
            owners=owners,
            weights=weights,
            memory_budget_mb=self.memory_budget_mb,
            row_cap=self.row_cap,
        )


__all__ = ["FastScalableRelevanceKMeans", "fast_tiled_oi_score"]
