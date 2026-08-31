"""Train-only target-moment transforms for the SAVE experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from overlapindex._shared_fisher import SharedFisherTransform, fit_shared_residual


MomentMode = Literal["mean", "save", "mean_save"]

_COVARIANCE_FLOOR_RELATIVE = 1.0e-6
_COVARIANCE_FLOOR_ABSOLUTE = 1.0e-12
_MOMENT_TOLERANCE = 1.0e-10


def _readonly(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


def _canonical_columns(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True)
class TargetMomentTransform:
    """Detached residual-preconditioned target-moment projection."""

    residual_transform: SharedFisherTransform
    projection_center: np.ndarray
    whitening: np.ndarray
    basis: np.ndarray
    mode: MomentMode
    requested_rank: int
    eigenvalues: np.ndarray
    mean_trace: float
    save_trace: float
    covariance_floor: float

    @property
    def input_dimension(self) -> int:
        return self.residual_transform.input_dimension

    @property
    def output_dimension(self) -> int:
        return int(self.basis.shape[1])

    def transform(self, X: Any) -> np.ndarray:
        residual = self.residual_transform.transform(X).astype(np.float64)
        projected = (
            (residual - self.projection_center) @ self.whitening @ self.basis
        )
        result = np.asarray(projected, dtype=np.float32)
        if result.shape != (len(residual), self.output_dimension):
            raise RuntimeError("target-moment transform emitted the wrong shape")
        if not np.all(np.isfinite(result)):
            raise RuntimeError("target-moment transform emitted non-finite values")
        return result


def _first_observed_encoding(labels: np.ndarray) -> tuple[tuple[Any, ...], np.ndarray]:
    classes = []
    positions = {}
    encoded = np.empty(labels.size, dtype=np.int64)
    for index, label in enumerate(labels.tolist()):
        if label not in positions:
            positions[label] = len(classes)
            classes.append(label)
        encoded[index] = positions[label]
    return tuple(classes), encoded


def _normalized(moment: np.ndarray) -> tuple[np.ndarray, float]:
    trace = float(np.trace(moment))
    if trace <= np.finfo(float).eps:
        return np.zeros_like(moment), trace
    return moment / trace, trace


def fit_target_moment_transform(
    X: Any,
    target_cells: Any,
    *,
    mode: MomentMode,
    residual_rank: int,
    projection_rank: int,
    random_state: int,
) -> TargetMomentTransform:
    """Fit a regularized SIR, SAVE, or normalized combined projection.

    Rows are first transformed by the existing pooled within-cell residual
    scaling/deflation. The corrected rows are globally whitened, after which
    target-cell mean and conditional-covariance moment matrices are estimated.
    Mean and SAVE terms are trace-normalized before combination so neither term
    wins merely because of units.
    """

    values = np.asarray(X, dtype=np.float32)
    labels = np.asarray(target_cells)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("target-moment fitting requires a non-empty 2D X")
    if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
        raise ValueError("target cells must be scalar labels aligned with X")
    if mode not in {"mean", "save", "mean_save"}:
        raise ValueError("mode must be 'mean', 'save', or 'mean_save'")
    if type(residual_rank) is not int or residual_rank < 0:
        raise ValueError("residual_rank must be a nonnegative integer")
    if type(projection_rank) is not int or projection_rank <= 0:
        raise ValueError("projection_rank must be a positive integer")
    if type(random_state) is not int or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer")

    classes, encoded = _first_observed_encoding(labels)
    if len(classes) < 2:
        raise ValueError("target-moment fitting requires at least two cells")
    counts = np.bincount(encoded, minlength=len(classes)).astype(np.int64)
    if np.any(counts < 2):
        raise ValueError("every target cell requires at least two rows")

    residual_transform = fit_shared_residual(
        values,
        labels,
        rank=residual_rank,
        random_state=random_state,
    )
    corrected = residual_transform.transform(values).astype(np.float64)
    center = np.mean(corrected, axis=0)
    centered = corrected - center
    covariance = centered.T @ centered / float(len(centered))
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    positive = eigenvalues[eigenvalues > 0.0]
    reference = float(np.median(positive)) if positive.size else 1.0
    covariance_floor = max(
        reference * _COVARIANCE_FLOOR_RELATIVE,
        _COVARIANCE_FLOOR_ABSOLUTE,
    )
    whitening = (
        eigenvectors
        * np.reciprocal(np.sqrt(np.maximum(eigenvalues, covariance_floor)))[None, :]
    ) @ eigenvectors.T
    whitened = centered @ whitening

    dimension = whitened.shape[1]
    mean_moment = np.zeros((dimension, dimension), dtype=np.float64)
    save_moment = np.zeros((dimension, dimension), dtype=np.float64)
    identity = np.eye(dimension, dtype=np.float64)
    for position, count in enumerate(counts.tolist()):
        rows = whitened[encoded == position]
        weight = float(count) / float(len(whitened))
        cell_mean = np.mean(rows, axis=0)
        mean_moment += weight * np.outer(cell_mean, cell_mean)
        cell_centered = rows - cell_mean
        cell_covariance = cell_centered.T @ cell_centered / float(len(rows))
        difference = identity - cell_covariance
        save_moment += weight * (difference @ difference)
    mean_moment = 0.5 * (mean_moment + mean_moment.T)
    save_moment = 0.5 * (save_moment + save_moment.T)
    normalized_mean, mean_trace = _normalized(mean_moment)
    normalized_save, save_trace = _normalized(save_moment)
    if mode == "mean":
        selected_moment = normalized_mean
    elif mode == "save":
        selected_moment = normalized_save
    else:
        selected_moment = normalized_mean + normalized_save

    moment_eigenvalues, moment_eigenvectors = np.linalg.eigh(selected_moment)
    order = np.argsort(moment_eigenvalues)[::-1]
    moment_eigenvalues = np.maximum(moment_eigenvalues[order], 0.0)
    moment_eigenvectors = moment_eigenvectors[:, order]
    maximum = float(moment_eigenvalues[0]) if moment_eigenvalues.size else 0.0
    positive_rank = int(
        np.count_nonzero(moment_eigenvalues > maximum * _MOMENT_TOLERANCE)
    )
    effective_rank = min(projection_rank, max(1, positive_rank), dimension)
    basis = _canonical_columns(moment_eigenvectors[:, :effective_rank])
    return TargetMomentTransform(
        residual_transform=residual_transform,
        projection_center=_readonly(center),
        whitening=_readonly(whitening),
        basis=_readonly(basis),
        mode=mode,
        requested_rank=int(projection_rank),
        eigenvalues=_readonly(moment_eigenvalues[:effective_rank]),
        mean_trace=float(mean_trace),
        save_trace=float(save_trace),
        covariance_floor=float(covariance_floor),
    )


__all__ = ["MomentMode", "TargetMomentTransform", "fit_target_moment_transform"]
