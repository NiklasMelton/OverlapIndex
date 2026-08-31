"""Deterministic train-only random Fourier features for the RFF-SAVE study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_MEDIAN_SAMPLE_LIMIT = 256


def _readonly(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


def _row_l2(X: Any) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("RFF input must be a non-empty 2D matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("RFF input must contain only finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _median_squared_distance(X: np.ndarray) -> float:
    count = min(len(X), _MEDIAN_SAMPLE_LIMIT)
    positions = np.linspace(0, len(X) - 1, count, dtype=np.int64)
    sample = np.asarray(X[positions], dtype=np.float64)
    squared_norms = np.sum(np.square(sample), axis=1)
    squared = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (sample @ sample.T)
    )
    squared = np.maximum(squared, 0.0)
    upper = squared[np.triu_indices(count, k=1)]
    positive = upper[upper > np.finfo(np.float64).eps]
    if positive.size == 0:
        raise ValueError("RFF bandwidth requires at least two distinct training rows")
    median = float(np.median(positive))
    if not np.isfinite(median) or median <= 0.0:
        raise RuntimeError("RFF median squared distance is invalid")
    return median


@dataclass(frozen=True)
class RandomFourierTransform:
    """Detached RBF random-feature map fitted only from training rows."""

    weights: np.ndarray
    offsets: np.ndarray
    median_squared_distance: float
    gamma: float
    random_state: int

    @property
    def input_dimension(self) -> int:
        return int(self.weights.shape[0])

    @property
    def output_dimension(self) -> int:
        return int(self.weights.shape[1])

    def transform(self, X: Any) -> np.ndarray:
        normalized = _row_l2(X)
        if normalized.shape[1] != self.input_dimension:
            raise ValueError(
                "RFF feature mismatch: expected "
                f"{self.input_dimension}, got {normalized.shape[1]}"
            )
        values = np.sqrt(2.0 / self.output_dimension) * np.cos(
            normalized.astype(np.float64) @ self.weights + self.offsets
        )
        result = _row_l2(values)
        if result.shape != (len(normalized), self.output_dimension):
            raise RuntimeError("RFF transform emitted the wrong shape")
        return np.asarray(result, dtype=np.float32)


def fit_random_fourier_transform(
    X: Any,
    *,
    n_components: int,
    random_state: int,
) -> RandomFourierTransform:
    """Fit the frozen median-bandwidth RBF sketch on training rows only."""

    if type(n_components) is not int or n_components <= 0:
        raise ValueError("n_components must be a positive integer")
    if type(random_state) is not int or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer")
    normalized = _row_l2(X)
    median_squared_distance = _median_squared_distance(normalized)
    gamma = 1.0 / (2.0 * median_squared_distance)
    rng = np.random.default_rng(random_state)
    weights = rng.normal(
        scale=np.sqrt(2.0 * gamma),
        size=(normalized.shape[1], n_components),
    )
    offsets = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    return RandomFourierTransform(
        weights=_readonly(weights),
        offsets=_readonly(offsets),
        median_squared_distance=float(median_squared_distance),
        gamma=float(gamma),
        random_state=int(random_state),
    )


__all__ = ["RandomFourierTransform", "fit_random_fourier_transform"]
