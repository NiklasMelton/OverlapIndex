"""Fixed-iteration class-batched K-means used by the opt-in fused backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import numpy as np

from overlapindex.utils import _group_indices_by_label, _validate_positive_integer


@dataclass(frozen=True)
class FusedKMeansFit:
    centers: np.ndarray
    owners: np.ndarray
    feature_weights: np.ndarray
    resolved_k: dict[Any, int]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _PackedClasses:
    values: np.ndarray
    row_mask: np.ndarray
    center_mask: np.ndarray
    initial_centers: np.ndarray
    labels: tuple[Any, ...]
    resolved_k: tuple[int, ...]


def _resolve_k(
    configured: Union[int, dict[Any, int]],
    label: Any,
    row_count: int,
    min_samples_per_prototype: int,
) -> int:
    raw = configured[label] if isinstance(configured, dict) else configured
    maximum = _validate_positive_integer(raw, f"k for class {label!r}")
    density_limit = max(1, row_count // min_samples_per_prototype)
    return min(maximum, density_limit, row_count)


def _pack_classes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k: Union[int, dict[Any, int]],
    random_state: int,
    min_samples_per_prototype: int,
) -> _PackedClasses:
    rows_by_class = _group_indices_by_label(y)
    labels = tuple(rows_by_class)
    class_rows = [X[rows_by_class[label]] for label in labels]
    resolved = tuple(
        _resolve_k(k, label, len(rows), min_samples_per_prototype)
        for label, rows in zip(labels, class_rows)
    )
    max_rows = max(len(rows) for rows in class_rows)
    max_k = max(resolved)
    packed = np.zeros((len(labels), max_rows, X.shape[1]), dtype=np.float32)
    row_mask = np.zeros((len(labels), max_rows), dtype=bool)
    center_mask = np.zeros((len(labels), max_k), dtype=bool)
    centers = np.zeros((len(labels), max_k, X.shape[1]), dtype=np.float32)
    for position, (rows, class_k) in enumerate(zip(class_rows, resolved)):
        packed[position, : len(rows)] = rows
        row_mask[position, : len(rows)] = True
        center_mask[position, :class_k] = True
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [random_state, position, len(rows), class_k]
            )
        )
        selected = rng.choice(len(rows), size=class_k, replace=False)
        centers[position, :class_k] = rows[np.asarray(selected, dtype=np.int64)]
    return _PackedClasses(
        values=packed,
        row_mask=row_mask,
        center_mask=center_mask,
        initial_centers=centers,
        labels=labels,
        resolved_k=resolved,
    )


def _fixed_lloyd(
    packed: _PackedClasses,
    *,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers = packed.initial_centers.copy()
    prototype_axis = np.arange(centers.shape[1], dtype=np.int64)
    counts = np.zeros(packed.center_mask.shape, dtype=np.int64)
    for _ in range(iterations):
        scores = np.matmul(packed.values, np.swapaxes(centers, 1, 2))
        center_norms = np.einsum("ckd,ckd->ck", centers, centers)
        scores -= center_norms[:, None, :] * np.float32(0.5)
        scores = np.where(packed.center_mask[:, None, :], scores, -np.inf)
        assignments = np.argmax(scores, axis=2).astype(np.int64, copy=False)
        one_hot = (
            (assignments[:, :, None] == prototype_axis[None, None, :])
            & packed.row_mask[:, :, None]
            & packed.center_mask[:, None, :]
        )
        counts = np.sum(one_hot, axis=1, dtype=np.int64)
        sums = np.matmul(np.swapaxes(one_hot.astype(np.float32), 1, 2), packed.values)
        updated = np.divide(
            sums,
            np.maximum(counts, 1)[:, :, None],
            dtype=np.float32,
        )
        centers = np.where((counts > 0)[:, :, None], updated, centers)
        centers = np.where(packed.center_mask[:, :, None], centers, np.float32(0.0))
    return centers, counts


def _flatten(
    packed: _PackedClasses,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    blocks = [
        np.asarray(centers[position, :class_k], dtype=np.float32)
        for position, class_k in enumerate(packed.resolved_k)
    ]
    owners: list[Any] = []
    for label, class_k in zip(packed.labels, packed.resolved_k):
        owners.extend([label] * class_k)
    return np.vstack(blocks).astype(np.float32, copy=False), np.asarray(owners)


def _margin_indices(
    y: np.ndarray,
    labels: tuple[Any, ...],
    *,
    cap: int,
    random_state: int,
) -> np.ndarray:
    rows_by_class = _group_indices_by_label(y)
    selected_blocks: list[np.ndarray] = []
    for position, label in enumerate(labels):
        local = np.asarray(rows_by_class[label], dtype=np.int64)
        if local.size <= cap:
            selected_blocks.append(local)
            continue
        rng = np.random.default_rng(
            np.random.SeedSequence([random_state, position, local.size, cap])
        )
        selected_blocks.append(
            np.sort(rng.choice(local, size=cap, replace=False)).astype(np.int64)
        )
    return np.sort(np.concatenate(selected_blocks))


def _score_kernel(
    X: np.ndarray,
    centers: np.ndarray,
    feature_weights: np.ndarray,
) -> np.ndarray:
    scale = np.asarray(np.sqrt(feature_weights), dtype=np.float32)
    X_scaled = np.asarray(X, dtype=np.float32) * scale
    centers_scaled = np.asarray(centers, dtype=np.float32) * scale
    center_norms = np.einsum("ij,ij->i", centers_scaled, centers_scaled)
    scores = np.asarray(X_scaled @ centers_scaled.T)
    scores -= center_norms[None, :] * np.float32(0.5)
    return scores.astype(np.float32, copy=False)


def _relevance_weights(
    X: np.ndarray,
    y: np.ndarray,
    *,
    centers: np.ndarray,
    owners: np.ndarray,
    labels: tuple[Any, ...],
    margin_rows_per_class: int,
    weight_floor: float,
    random_state: int,
    label_distances: Optional[Mapping[tuple[Any, Any], float]] = None,
) -> tuple[np.ndarray, int, int]:
    if len(labels) < 2:
        return np.ones(X.shape[1], dtype=np.float64), 0, 0
    chosen = _margin_indices(
        y,
        labels,
        cap=margin_rows_per_class,
        random_state=random_state,
    )
    margin_X = X[chosen]
    margin_y = y[chosen]
    scores = _score_kernel(
        margin_X,
        centers,
        np.ones(X.shape[1], dtype=np.float64),
    )
    margin_sum = np.zeros(X.shape[1], dtype=np.float64)
    residual_sum = np.zeros(X.shape[1], dtype=np.float64)
    ids_by_class = {
        label: np.flatnonzero(owners == label).astype(np.int64)
        for label in labels
    }
    for label in labels:
        local = np.flatnonzero(margin_y == label)
        own_ids = ids_by_class[label]
        own_choice = own_ids[np.argmax(scores[local][:, own_ids], axis=1)]
        selected = margin_X[local]
        own_residual = np.square(selected - centers[own_choice], dtype=np.float32)
        if label_distances is None:
            wrong_ids = np.flatnonzero(owners != label)
            wrong_choice = wrong_ids[np.argmax(scores[local][:, wrong_ids], axis=1)]
            wrong_residual = np.square(
                selected - centers[wrong_choice],
                dtype=np.float32,
            )
        else:
            competitors = [other for other in labels if other != label]
            target_weights = np.asarray(
                [label_distances[(label, other)] for other in competitors],
                dtype=np.float64,
            )
            total_weight = float(np.sum(target_weights))
            if total_weight <= np.finfo(float).eps:
                target_weights = np.full(
                    len(competitors),
                    1.0 / float(len(competitors)),
                    dtype=np.float64,
                )
            else:
                target_weights /= total_weight
            wrong_residual = np.zeros_like(own_residual, dtype=np.float64)
            for other, target_weight in zip(competitors, target_weights):
                wrong_ids = ids_by_class[other]
                wrong_choice = wrong_ids[
                    np.argmax(scores[local][:, wrong_ids], axis=1)
                ]
                residual = np.square(
                    selected - centers[wrong_choice],
                    dtype=np.float32,
                )
                wrong_residual += float(target_weight) * residual.astype(np.float64)
        margin_sum += np.sum(
            wrong_residual.astype(np.float64) - own_residual.astype(np.float64),
            axis=0,
            dtype=np.float64,
        )
        residual_sum += np.sum(own_residual, axis=0, dtype=np.float64)
    denominator = float(len(margin_X))
    positive_margin = np.maximum(margin_sum / denominator, 0.0)
    residual = residual_sum / denominator
    epsilon = max(float(np.mean(residual)) * 1.0e-6, 1.0e-12)
    relevance = positive_margin / (positive_margin + residual + epsilon)
    raw = weight_floor + (1.0 - weight_floor) * relevance
    weights = np.asarray(raw / float(np.mean(raw)), dtype=np.float64)
    return weights, int(len(chosen)), int(np.count_nonzero(positive_margin))


def _validate_label_distances(
    labels: tuple[Any, ...],
    label_distances: Optional[Mapping[tuple[Any, Any], float]],
) -> Optional[dict[tuple[Any, Any], float]]:
    """Validate a complete directed target-distance mapping."""
    if label_distances is None:
        return None
    if not isinstance(label_distances, Mapping):
        raise TypeError("FusedKMeans label_distances must be a mapping or None")
    expected = {
        (source, target)
        for source in labels
        for target in labels
        if source != target
    }
    if set(label_distances) != expected:
        raise ValueError(
            "FusedKMeans label_distances must contain every directed "
            "non-diagonal label pair exactly once"
        )
    resolved: dict[tuple[Any, Any], float] = {}
    for pair in expected:
        value = label_distances[pair]
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise TypeError("FusedKMeans label distances must be real scalars")
        distance = float(value)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError(
                "FusedKMeans label distances must be finite and nonnegative"
            )
        reverse = float(label_distances[(pair[1], pair[0])])
        if not np.isclose(distance, reverse, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("FusedKMeans label_distances must be symmetric")
        resolved[pair] = distance
    return resolved


def fit_fused_kmeans(
    X: Any,
    y: Any,
    *,
    k: Union[int, dict[Any, int]],
    random_state: int,
    iterations: int,
    min_samples_per_prototype: int,
    relevance_weighting: bool,
    margin_rows_per_class: int,
    weight_floor: float,
    label_distances: Optional[Mapping[tuple[Any, Any], float]] = None,
) -> FusedKMeansFit:
    """Fit fixed-iteration class-owned prototypes and optional relevance weights."""

    values = np.asarray(X, dtype=np.float32)
    target = np.asarray(y)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("FusedKMeans requires a non-empty two-dimensional X")
    if target.ndim != 1 or target.shape[0] != values.shape[0]:
        raise ValueError("FusedKMeans requires scalar labels aligned with X")
    if not np.all(np.isfinite(values)):
        raise ValueError("FusedKMeans requires finite X")
    if type(random_state) is not int or random_state < 0:
        raise ValueError("FusedKMeans random_state must be a nonnegative int")
    iterations = _validate_positive_integer(iterations, "FusedKMeans iterations")
    min_samples_per_prototype = _validate_positive_integer(
        min_samples_per_prototype,
        "FusedKMeans min_samples_per_prototype",
    )
    margin_rows_per_class = _validate_positive_integer(
        margin_rows_per_class,
        "FusedKMeans margin_rows_per_class",
    )
    if type(relevance_weighting) is not bool:
        raise TypeError("FusedKMeans relevance_weighting must be a bool")
    if type(weight_floor) is not float or not 0.0 < weight_floor <= 1.0:
        raise ValueError("FusedKMeans weight_floor must be a float in (0, 1]")

    packed = _pack_classes(
        values,
        target,
        k=k,
        random_state=random_state,
        min_samples_per_prototype=min_samples_per_prototype,
    )
    resolved_label_distances = _validate_label_distances(
        packed.labels,
        label_distances,
    )
    packed_centers, counts = _fixed_lloyd(packed, iterations=iterations)
    centers, owners = _flatten(packed, packed_centers)
    if relevance_weighting:
        weights, margin_count, positive_count = _relevance_weights(
            values,
            target,
            centers=centers,
            owners=owners,
            labels=packed.labels,
            margin_rows_per_class=margin_rows_per_class,
            weight_floor=weight_floor,
            random_state=random_state,
            label_distances=resolved_label_distances,
        )
    else:
        weights = np.ones(values.shape[1], dtype=np.float64)
        margin_count = 0
        positive_count = 0
    resolved_k = dict(zip(packed.labels, packed.resolved_k))
    diagnostics = {
        "algorithm": "fused_fixed_lloyd_capped_relevance_no_final_lloyd",
        "class_count": len(packed.labels),
        "prototype_count": int(len(centers)),
        "fixed_lloyd_iterations": int(iterations),
        "min_samples_per_prototype": int(min_samples_per_prototype),
        "relevance_weighting": bool(relevance_weighting),
        "margin_row_count": int(margin_count),
        "margin_rows_per_class": int(margin_rows_per_class),
        "weight_floor": float(weight_floor),
        "positive_margin_feature_count": int(positive_count),
        "empty_prototype_count": int(
            sum(
                np.count_nonzero(counts[position, :class_k] == 0)
                for position, class_k in enumerate(packed.resolved_k)
            )
        ),
        "weighted_lloyd_update_count": 0,
    }
    if resolved_label_distances is not None:
        diagnostics["relevance_mode"] = "target_distance_weighted"
    for array in (centers, owners, weights):
        array.setflags(write=False)
    return FusedKMeansFit(
        centers=centers,
        owners=owners,
        feature_weights=weights,
        resolved_k=resolved_k,
        diagnostics=diagnostics,
    )


__all__ = ["FusedKMeansFit", "fit_fused_kmeans"]
