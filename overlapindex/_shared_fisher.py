"""Train-only shared-Fisher feature transforms for offline OI backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.extmath import randomized_svd


FisherRank = Union[int, Literal["full"]]

_RELATIVE_VARIANCE_FLOOR = 1.0e-6
_ABSOLUTE_VARIANCE_FLOOR = 1.0e-12
_RANDOMIZED_POWER_ITERATIONS = 1
_RANDOMIZED_OVERSAMPLES = 8
_FULL_FISHER_TOLERANCE = 1.0e-4


def _first_observed_encoding(labels: np.ndarray) -> tuple[tuple[Any, ...], np.ndarray]:
    classes: list[Any] = []
    positions: dict[Any, int] = {}
    encoded = np.empty(labels.size, dtype=np.int64)
    for index, label in enumerate(labels.tolist()):
        try:
            position = positions.get(label)
        except TypeError as exc:
            raise TypeError("labels must be hashable scalar values") from exc
        if position is None and label not in positions:
            position = len(classes)
            positions[label] = position
            classes.append(label)
        encoded[index] = int(position)
    return tuple(classes), encoded


def _read_only(value: Any, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _contrast_basis(contrasts: np.ndarray, rank: int) -> np.ndarray:
    _left, _singular, right = np.linalg.svd(
        np.asarray(contrasts, dtype=np.float64),
        full_matrices=False,
    )
    basis = np.asarray(right[:rank].T, dtype=np.float64)
    if basis.shape != (contrasts.shape[1], rank) or not np.all(np.isfinite(basis)):
        raise RuntimeError("shared Fisher class-contrast SVD produced invalid state")
    return basis


@dataclass(frozen=True)
class SharedFisherTransform:
    """Detached transform fitted from one labeled training set."""

    center: np.ndarray
    feature_scale: np.ndarray
    nuisance_vectors: np.ndarray
    nuisance_gains: np.ndarray
    discriminant_basis: np.ndarray
    classes: tuple[Any, ...]
    requested_rank: FisherRank
    effective_nuisance_rank: int
    variance_floor: float
    variance_floor_count: int

    @property
    def input_dimension(self) -> int:
        return int(self.feature_scale.shape[0])

    @property
    def output_dimension(self) -> int:
        return int(self.discriminant_basis.shape[1])

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dimension:
            raise ValueError("shared Fisher input has the wrong feature dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("shared Fisher input must contain only finite values")
        standardized = (
            values.astype(np.float64) - self.center
        ) * self.feature_scale
        if self.nuisance_vectors.shape[1]:
            coordinates = standardized @ self.nuisance_vectors
            standardized = standardized + (
                coordinates * self.nuisance_gains[None, :]
            ) @ self.nuisance_vectors.T
        transformed = np.asarray(
            standardized @ self.discriminant_basis,
            dtype=np.float32,
        )
        if transformed.shape != (len(values), self.output_dimension):
            raise RuntimeError("shared Fisher transform emitted the wrong shape")
        if not np.all(np.isfinite(transformed)):
            raise RuntimeError("shared Fisher transform emitted non-finite values")
        return transformed


def _fit_full_fisher(
    values: np.ndarray,
    encoded: np.ndarray,
    classes: tuple[Any, ...],
) -> SharedFisherTransform:
    estimator = LinearDiscriminantAnalysis(
        solver="svd",
        n_components=None,
        tol=_FULL_FISHER_TOLERANCE,
    ).fit(values, encoded)
    output_rank = min(len(classes) - 1, values.shape[1])
    basis = np.asarray(estimator.scalings_[:, :output_rank], dtype=np.float64)
    center = np.asarray(estimator.xbar_, dtype=np.float64)
    if basis.shape != (values.shape[1], output_rank):
        raise RuntimeError("full Fisher fitting produced the wrong shape")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(basis)):
        raise RuntimeError("full Fisher fitting produced non-finite state")
    return SharedFisherTransform(
        center=_read_only(center, np.float64),
        feature_scale=_read_only(np.ones(values.shape[1]), np.float64),
        nuisance_vectors=_read_only(np.empty((values.shape[1], 0)), np.float64),
        nuisance_gains=_read_only(np.empty(0), np.float64),
        discriminant_basis=_read_only(basis, np.float64),
        classes=classes,
        requested_rank="full",
        effective_nuisance_rank=values.shape[1],
        variance_floor=0.0,
        variance_floor_count=0,
    )


def fit_shared_fisher(
    X: Any,
    y: Any,
    *,
    rank: FisherRank,
    random_state: int | None,
) -> SharedFisherTransform:
    """Fit the full or scalable shared-Fisher transform.

    Integer ranks use diagonal within-class scaling plus a randomized low-rank
    correction for energetic correlated residual directions. ``"full"`` uses
    scikit-learn's full SVD Fisher/LDA transform.
    """

    values = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("shared Fisher requires a non-empty two-dimensional X")
    if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
        raise ValueError("shared Fisher requires scalar labels aligned with X")
    if not np.all(np.isfinite(values)):
        raise ValueError("shared Fisher requires finite X")
    if rank != "full" and (type(rank) is not int or rank < 0):
        raise ValueError("fisher_rank must be a nonnegative int or 'full'")
    if random_state is not None and (
        type(random_state) is not int or random_state < 0
    ):
        raise ValueError("fisher_random_state must be None or a nonnegative int")

    classes, encoded = _first_observed_encoding(labels)
    if len(classes) < 2:
        raise ValueError("shared Fisher requires at least two classes")
    counts = np.bincount(encoded, minlength=len(classes)).astype(np.int64)
    if np.any(counts < 2):
        raise ValueError("shared Fisher requires at least two rows per class")
    if rank == "full":
        return _fit_full_fisher(values, encoded, classes)

    values64 = values.astype(np.float64)
    class_means = np.vstack(
        [np.mean(values64[encoded == position], axis=0) for position in range(len(classes))]
    )
    global_mean = np.mean(values64, axis=0)
    residual = values64 - class_means[encoded]
    pooled_variance_raw = np.mean(np.square(residual), axis=0)
    positive = pooled_variance_raw[pooled_variance_raw > 0.0]
    reference = float(np.median(positive)) if positive.size else 1.0
    variance_floor = max(
        reference * _RELATIVE_VARIANCE_FLOOR,
        _ABSOLUTE_VARIANCE_FLOOR,
    )
    feature_scale = np.reciprocal(
        np.sqrt(np.maximum(pooled_variance_raw, variance_floor))
    )
    class_weights = np.sqrt(counts.astype(np.float64) / float(len(values)))
    contrasts = (class_means - global_mean[None, :]) * class_weights[:, None]
    standardized_contrasts = contrasts * feature_scale[None, :]
    standardized_residual = residual * feature_scale[None, :]
    effective_rank = min(
        int(rank),
        values.shape[1],
        len(values) - len(classes),
    )
    if effective_rank:
        _left, singular, right = randomized_svd(
            standardized_residual,
            n_components=effective_rank,
            n_iter=_RANDOMIZED_POWER_ITERATIONS,
            n_oversamples=_RANDOMIZED_OVERSAMPLES,
            random_state=random_state,
            flip_sign=True,
        )
        nuisance_vectors = np.asarray(right.T, dtype=np.float64)
        degrees_of_freedom = max(1, len(values) - len(classes))
        eigenvalues = np.square(np.asarray(singular, dtype=np.float64)) / float(
            degrees_of_freedom
        )
        # Deflate only unusually energetic residual directions. Low-variance
        # directions are deliberately not amplified.
        nuisance_gains = (
            np.reciprocal(np.sqrt(np.maximum(eigenvalues, 1.0))) - 1.0
        )
    else:
        nuisance_vectors = np.empty((values.shape[1], 0), dtype=np.float64)
        nuisance_gains = np.empty(0, dtype=np.float64)
    corrected_contrasts = standardized_contrasts + (
        (standardized_contrasts @ nuisance_vectors) * nuisance_gains[None, :]
    ) @ nuisance_vectors.T
    output_rank = min(len(classes) - 1, values.shape[1])
    basis = _contrast_basis(corrected_contrasts, output_rank)
    return SharedFisherTransform(
        center=_read_only(global_mean, np.float64),
        feature_scale=_read_only(feature_scale, np.float64),
        nuisance_vectors=_read_only(nuisance_vectors, np.float64),
        nuisance_gains=_read_only(nuisance_gains, np.float64),
        discriminant_basis=_read_only(basis, np.float64),
        classes=classes,
        requested_rank=int(rank),
        effective_nuisance_rank=int(effective_rank),
        variance_floor=float(variance_floor),
        variance_floor_count=int(np.count_nonzero(pooled_variance_raw < variance_floor)),
    )


__all__ = ["FisherRank", "SharedFisherTransform", "fit_shared_fisher"]
