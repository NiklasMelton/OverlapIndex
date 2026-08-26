"""Private distance conditioning used by the heteroscedastic experiment.

The adapter in this module is deliberately experiment-local.  It fits a
training-fold-only affine feature transform and delegates all prototype,
adjacency, and overlap calculations to the existing :class:`OverlapIndex`.
No public ``overlapindex`` API is changed here.

The protocol names four estimator families:

``none``
    Exact identity test seam for the raw A/B controls and ``gamma=0``.
``oas_diagonal``
    Pooled within-class diagonal OAS variance, with a configurable partial
    whitening exponent.
``winsorized_oas_diagonal``
    MAD-seeded, feature-wise winsorized diagonal OAS variance.
``pooled_mad``
    Feature-wise pooled weighted MAD variance.

The conditioning state is fit only from rows supplied to :meth:`fit`.  The
held-out methods apply that frozen state and never refit the conditioner or
the wrapped overlap index.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
from numbers import Real
import time
from typing import Any, Optional

import numpy as np
from scipy import sparse
from sklearn.covariance import oas

from overlapindex import OverlapIndex


CONDITIONING_ESTIMATORS = (
    "none",
    "oas_diagonal",
    "winsorized_oas_diagonal",
    "pooled_mad",
)
"""Closed estimator family names in the frozen protocol."""

CONDITIONING_WEIGHTINGS = (
    None,
    "sample_weighted_rows",
    "class_balanced",
)
"""Closed residual weighting values; ``None`` is the JSON null value."""

RELATIVE_EIGENVALUE_FLOOR = 1e-8
"""Frozen relative variance floor."""

CONDITION_NUMBER_CAP = 10_000.0
"""Frozen maximum variance condition number."""

WINSOR_LIMIT_MAD_UNITS = 3.0
"""Frozen winsorization threshold in MAD units."""

MAD_CONSISTENCY = 1.4826
"""Normal-consistency multiplier in the frozen pooled-MAD definition."""


def _json_number(value: Any) -> Optional[float]:
    """Return a finite JSON number, or ``None`` for an undefined value."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    """Recursively copy diagnostics into JSON-compatible Python values."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return _json_number(value)
    return str(value)


def _validate_dense_matrix(
    X: Any,
    name: str,
    *,
    preserve_dtype: bool,
) -> np.ndarray:
    """Validate a non-empty finite dense feature matrix.

    The identity path intentionally returns an ndarray with its original
    numeric dtype.  Conditioned paths cast once to float64 before any
    arithmetic, keeping the raw A/B controls byte-for-byte equivalent to
    direct upstream ``OverlapIndex`` calls for ordinary dense inputs.
    """

    if sparse.issparse(X):
        raise TypeError(f"{name} must be a finite dense 2D array.")
    try:
        raw = np.asarray(X)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite dense 2D array.") from exc
    if raw.ndim != 2:
        raise ValueError(f"{name} must be a finite dense 2D array; got shape {raw.shape}.")
    if raw.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if raw.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature column.")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real-valued features.")

    try:
        if preserve_dtype:
            # Numeric means an ordinary NumPy numeric dtype here.  Object
            # arrays containing numbers are accepted by non-identity casts,
            # but accepting them on the exact path would make dtype parity
            # ambiguous and is unnecessary for the frozen controls.
            if not np.issubdtype(raw.dtype, np.number):
                raise ValueError(f"{name} must contain numeric features.")
            array = raw
        else:
            array = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values.")
    except (TypeError, ValueError) as exc:
        message = str(exc)
        if "NaN" in message or "infinite" in message or "numeric" in message:
            raise
        raise ValueError(f"{name} must be a finite dense 2D array.") from exc
    return array


def _validate_scalar_label(label: Any) -> None:
    """Reject non-scalar, unhashable, or missing target values."""

    if isinstance(label, (list, tuple, set, frozenset, dict)):
        raise ValueError("Labels must be scalar single-label values.")
    if isinstance(label, np.ndarray):
        raise ValueError("Labels must be scalar single-label values.")
    if not isinstance(label, (str, bytes)) and not np.isscalar(label):
        raise ValueError("Labels must be scalar single-label values.")
    try:
        hash(label)
    except TypeError as exc:
        raise TypeError("Labels must be hashable scalar values.") from exc
    if label is None:
        raise ValueError("Labels must not contain None or NaN values.")
    try:
        unequal_to_self = label != label
        missing = bool(unequal_to_self)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Labels must be scalar, non-missing values with unambiguous equality."
        ) from exc
    if missing:
        raise ValueError("Labels must not contain None or NaN values.")


def _validate_labels(Y: Any, n_rows: int) -> np.ndarray:
    """Return an object processing view while retaining the delegate target."""

    if isinstance(Y, (str, bytes)):
        raise ValueError("Y must be a one-dimensional scalar-label sequence.")
    try:
        labels = np.asarray(Y, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("Y must be a one-dimensional scalar-label sequence.") from exc
    if labels.ndim != 1:
        raise ValueError(
            "Y must be a one-dimensional scalar-label sequence; "
            f"got shape {labels.shape}."
        )
    if labels.shape[0] != int(n_rows):
        raise ValueError(
            f"X and Y must have the same number of rows; got {n_rows} and "
            f"{labels.shape[0]}."
        )
    for label in labels.tolist():
        _validate_scalar_label(label)
    classes: list[Any] = []
    for label in labels.tolist():
        if not any(label == previous for previous in classes):
            classes.append(label)
    if len(classes) < 2:
        raise ValueError("At least two scalar classes are required.")
    return labels


def _class_blocks(labels: np.ndarray) -> tuple[list[Any], list[np.ndarray]]:
    """Return first-observed classes and their original row indices."""

    classes: list[Any] = []
    rows: list[list[int]] = []
    for row_index, label in enumerate(labels.tolist()):
        found = None
        for class_index, previous in enumerate(classes):
            if label == previous:
                found = class_index
                break
        if found is None:
            classes.append(label)
            rows.append([int(row_index)])
        else:
            rows[found].append(int(row_index))
    return classes, [np.asarray(indices, dtype=int) for indices in rows]


def _condition_number(values: np.ndarray) -> Optional[float]:
    """Return the finite condition number of positive variance/eigen values."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum <= 0.0 or not np.isfinite(minimum) or not np.isfinite(maximum):
        return None
    return _json_number(maximum / minimum)


def _regularize_variances(values: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
    """Apply the frozen floor/cap rule and report the effective floor."""

    raw = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if raw.size == 0:
        return raw.copy(), None
    largest = float(np.max(raw))
    if not np.isfinite(largest) or largest <= 0.0:
        # Identity variance is the stable fallback for an all-constant pool.
        return np.ones_like(raw), 1.0
    effective_floor = max(
        largest * float(RELATIVE_EIGENVALUE_FLOOR),
        largest / float(CONDITION_NUMBER_CAP),
    )
    return np.maximum(raw, effective_floor), float(effective_floor)


def _canonical_array_bytes(array: np.ndarray) -> bytes:
    """Encode one numeric array with shape and little-endian float64 values."""

    canonical = np.asarray(array, dtype="<f8", order="C")
    shape = np.asarray(canonical.shape, dtype="<i8")
    return shape.tobytes() + canonical.tobytes(order="C")


def _state_hash(
    estimator: str,
    gamma: float,
    mean: np.ndarray,
    covariance: np.ndarray,
    transform: np.ndarray,
) -> str:
    """Hash only deterministic conditioning configuration and numeric state."""

    digest = hashlib.sha256()
    # Weighting is intentionally absent: when class counts are equal, the
    # class-balanced path routes through exactly the sample-weighted numeric
    # state and must have the same structural identity.  Unequal weighting
    # still changes the covariance/transform bytes below and therefore the
    # digest.
    for value in (estimator, repr(float(gamma))):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    digest.update(_canonical_array_bytes(mean))
    digest.update(_canonical_array_bytes(covariance))
    digest.update(_canonical_array_bytes(transform))
    return digest.hexdigest()


def _readonly_float(array: np.ndarray) -> np.ndarray:
    """Copy a fitted array into detached read-only float64 storage."""

    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


class _FittedConditioner:
    """Detached/read-only affine conditioning state."""

    def __init__(
        self,
        *,
        estimator: str,
        weighting: Optional[str],
        gamma: float,
        mean: np.ndarray,
        covariance: np.ndarray,
        transform_matrix: np.ndarray,
        diagnostics: Mapping[str, Any],
        identity: bool,
    ) -> None:
        self.estimator = str(estimator)
        self.weighting = weighting
        self.gamma = float(gamma)
        self._identity = bool(identity)
        self.mean_ = _readonly_float(mean)
        self.covariance_ = _readonly_float(covariance)
        self.transform_ = _readonly_float(transform_matrix)
        self.n_features_in_ = int(self.mean_.size)
        self.diagnostics_ = deepcopy(_json_safe(dict(diagnostics)))
        self.state_sha256_ = str(self.diagnostics_["state_sha256"])

    def copy(self) -> "_FittedConditioner":
        """Return a detached read-only snapshot for inspection."""

        return _FittedConditioner(
            estimator=self.estimator,
            weighting=self.weighting,
            gamma=self.gamma,
            mean=self.mean_,
            covariance=self.covariance_,
            transform_matrix=self.transform_,
            diagnostics=self.diagnostics_,
            identity=self._identity,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted affine map without consulting new rows/labels."""

        if self._identity:
            return X
        centered = np.asarray(X, dtype=np.float64) - self.mean_
        return np.asarray(centered @ self.transform_, dtype=np.float64)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute protocol weighted medians with stable value/index tie order."""

    matrix = np.asarray(values, dtype=np.float64)
    row_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or row_weights.size != matrix.shape[0]:
        raise ValueError("weighted median inputs have incompatible shapes")
    total = float(np.sum(row_weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weighted median requires positive finite weights")
    result = np.empty(matrix.shape[1], dtype=np.float64)
    threshold = 0.5 * total
    for feature in range(matrix.shape[1]):
        order = np.argsort(matrix[:, feature], kind="mergesort")
        cumulative = np.cumsum(row_weights[order])
        selected = int(np.searchsorted(cumulative, threshold, side="left"))
        selected = min(selected, order.size - 1)
        result[feature] = matrix[order[selected], feature]
    return result


def _weights_for_blocks(
    blocks: list[np.ndarray],
    weighting: str,
) -> tuple[np.ndarray, bool]:
    """Build deterministic row weights and route balanced counts through SW."""

    n_rows = int(sum(block.size for block in blocks))
    if weighting == "class_balanced" and len({int(block.size) for block in blocks}) > 1:
        class_count = len(blocks)
        parts = [
            np.full(block.size, 1.0 / (class_count * block.size), dtype=np.float64)
            for block in blocks
        ]
        return np.concatenate(parts), False
    # The equal-count balanced case deliberately uses the exact same array and
    # code path as sample weighting, establishing the frozen bit-identity gate.
    return np.full(n_rows, 1.0 / n_rows, dtype=np.float64), True


def _residual_stack(X: np.ndarray, blocks: list[np.ndarray]) -> np.ndarray:
    """Build class-major within-class residual rows."""

    residual_blocks: list[np.ndarray] = []
    for indices in blocks:
        class_rows = X[indices]
        class_mean = np.mean(class_rows, axis=0)
        residual_blocks.append(class_rows - class_mean)
    return np.vstack(residual_blocks).astype(np.float64, copy=False)


def _oas_diagonal(
    residuals: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the exact sample-weighted OAS diagonal and alpha."""

    covariance, alpha = oas(residuals, assume_centered=True)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    return np.asarray(np.diag(covariance), dtype=np.float64), float(alpha)


def _diagnostic_template(
    *,
    estimator: str,
    weighting: Optional[str],
    gamma: float,
    n_rows: int,
    n_features: int,
    n_classes: int,
) -> dict[str, Any]:
    """Create the stable schema shared by every fitted conditioner."""

    return {
        "estimator": None if estimator == "none" else estimator,
        "residual_weighting": weighting,
        "gamma": float(gamma),
        "n_rows_fit": int(n_rows),
        "n_features_fit": int(n_features),
        "n_classes_fit": int(n_classes),
        "covariance_scope": None if estimator == "none" else "pooled_within_class",
        "structure": "identity" if estimator == "none" else "diagonal",
        "shrinkage": None,
        "variance_min_before": None,
        "variance_max_before": None,
        "variance_min_after": None,
        "variance_max_after": None,
        "condition_before": None,
        "condition_after": None,
        "relative_eigenvalue_floor": None,
        "condition_number_cap": None,
        "effective_eigenvalue_floor": None,
        "regularized_variance_count": 0,
        "cap_or_floor_active": False,
        "clip_count_per_feature": [0] * int(n_features),
        "clipped_row_fraction": 0.0,
    }


def _fit_conditioner(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    estimator: str,
    weighting: Optional[str],
    gamma: float,
) -> _FittedConditioner:
    """Fit one frozen conditioner from training rows only."""

    n_rows, n_features = X.shape
    classes, blocks = _class_blocks(labels)
    diagnostics = _diagnostic_template(
        estimator=estimator,
        weighting=weighting,
        gamma=gamma,
        n_rows=n_rows,
        n_features=n_features,
        n_classes=len(classes),
    )
    identity = estimator == "none" or gamma == 0.0
    identity_mean = np.zeros(n_features, dtype=np.float64)
    identity_covariance = np.eye(n_features, dtype=np.float64)
    if identity:
        diagnostics["state_sha256"] = _state_hash(
            estimator,
            gamma,
            identity_mean,
            identity_covariance,
            identity_covariance,
        )
        return _FittedConditioner(
            estimator=estimator,
            weighting=weighting,
            gamma=gamma,
            mean=identity_mean,
            covariance=identity_covariance,
            transform_matrix=identity_covariance,
            diagnostics=diagnostics,
            identity=True,
        )

    global_mean = np.asarray(np.mean(X, axis=0), dtype=np.float64)
    residuals = _residual_stack(X, blocks)
    weights, uses_sample_path = _weights_for_blocks(blocks, str(weighting))

    clip_counts = np.zeros(n_features, dtype=np.int64)
    clipped_fraction = 0.0
    shrinkage: Optional[float] = None
    if estimator == "oas_diagonal":
        raw_variance, shrinkage = _oas_diagonal(residuals)
        if weighting == "class_balanced" and not uses_sample_path:
            moments = np.sum(weights[:, None] * (residuals * residuals), axis=0)
            target = float(np.mean(moments))
            raw_variance = (1.0 - float(shrinkage)) * moments + float(shrinkage) * target
    elif estimator == "winsorized_oas_diagonal":
        # Both SW and CB use this shared sample-weighted threshold and alpha.
        shared_weights = np.full(residuals.shape[0], 1.0 / residuals.shape[0])
        median = _weighted_median(residuals, shared_weights)
        mad = MAD_CONSISTENCY * _weighted_median(
            np.abs(residuals - median), shared_weights
        )
        limits = WINSOR_LIMIT_MAD_UNITS * mad
        clipped = np.clip(residuals, -limits, limits)
        clip_counts = np.count_nonzero(clipped != residuals, axis=0).astype(np.int64)
        clipped_fraction = float(np.count_nonzero(np.any(clipped != residuals, axis=1))) / float(
            residuals.shape[0]
        )
        sw_variance, shrinkage = _oas_diagonal(clipped)
        if weighting == "class_balanced" and not uses_sample_path:
            moments = np.sum(weights[:, None] * (clipped * clipped), axis=0)
            target = float(np.mean(moments))
            raw_variance = (1.0 - float(shrinkage)) * moments + float(shrinkage) * target
        else:
            raw_variance = sw_variance
    elif estimator == "pooled_mad":
        median = _weighted_median(residuals, weights)
        mad = MAD_CONSISTENCY * _weighted_median(
            np.abs(residuals - median), weights
        )
        raw_variance = mad * mad
    else:  # pragma: no cover - constructor validates this closed set.
        raise ValueError(f"unknown conditioning estimator {estimator!r}")

    raw_variance = np.asarray(raw_variance, dtype=np.float64)
    regularized, effective_floor = _regularize_variances(raw_variance)
    # Keep the legacy full-whitening operation order for the exact L parity
    # path.  The mathematically equivalent power form can differ by a final
    # bit for ``gamma=1`` on some libm/NumPy combinations.
    if float(gamma) == 1.0:
        diagonal_scale = 1.0 / np.sqrt(regularized)
    else:
        diagonal_scale = regularized ** (-float(gamma) / 2.0)
    transform = np.diag(diagonal_scale)
    regularized_changed = regularized != np.maximum(raw_variance, 0.0)
    diagnostics.update(
        {
            "shrinkage": None if shrinkage is None else float(shrinkage),
            "variance_min_before": _json_number(np.min(raw_variance)),
            "variance_max_before": _json_number(np.max(raw_variance)),
            "variance_min_after": _json_number(np.min(regularized)),
            "variance_max_after": _json_number(np.max(regularized)),
            "condition_before": _condition_number(raw_variance),
            "condition_after": _condition_number(regularized),
            "relative_eigenvalue_floor": float(RELATIVE_EIGENVALUE_FLOOR),
            "condition_number_cap": float(CONDITION_NUMBER_CAP),
            "effective_eigenvalue_floor": effective_floor,
            "regularized_variance_count": int(np.count_nonzero(regularized_changed)),
            "cap_or_floor_active": bool(np.any(regularized_changed)),
            "clip_count_per_feature": [int(value) for value in clip_counts],
            "clipped_row_fraction": float(clipped_fraction),
        }
    )
    covariance = np.diag(raw_variance)
    diagnostics["state_sha256"] = _state_hash(
        estimator,
        gamma,
        global_mean,
        covariance,
        transform,
    )
    return _FittedConditioner(
        estimator=estimator,
        weighting=weighting,
        gamma=gamma,
        mean=global_mean,
        covariance=covariance,
        transform_matrix=transform,
        diagnostics=diagnostics,
        identity=False,
    )


class ConditionedOverlapIndex:
    """Research-only train-fold-conditioned wrapper around ``OverlapIndex``.

    Parameters
    ----------
    estimator : str, default="none"
        One of the four frozen estimator names in
        :data:`CONDITIONING_ESTIMATORS`.
    weighting : {None, "sample_weighted_rows", "class_balanced"}, optional
        Residual weighting for conditioned estimators.  ``None`` is required
        for ``estimator="none"``.
    gamma : float, default=0.0
        Partial-whitening exponent in the closed interval ``[0, 1]``.
        ``gamma=0`` is an exact identity test seam.
    prototype_refinement : bool, default=False
        Strict Python boolean forwarded explicitly to the wrapped estimator.
    conditioning_kwargs : mapping, optional
        Frozen controls.  Only ``condition_number_cap`` and
        ``relative_eigenvalue_floor`` are accepted, and both must retain the
        protocol values.
    overlap_index_kwargs : mapping, optional
        Copied keyword arguments for a fresh upstream ``OverlapIndex``.
    """

    def __init__(
        self,
        estimator: str = "none",
        *,
        weighting: Optional[str] = None,
        gamma: float = 0.0,
        prototype_refinement: bool = False,
        conditioning_kwargs: Optional[Mapping[str, Any]] = None,
        overlap_index_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(estimator, str) or estimator not in CONDITIONING_ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {CONDITIONING_ESTIMATORS!r}; got {estimator!r}."
            )
        if weighting is not None and (
            not isinstance(weighting, str) or weighting not in CONDITIONING_WEIGHTINGS
        ):
            raise ValueError(
                f"weighting must be one of {CONDITIONING_WEIGHTINGS!r}; got {weighting!r}."
            )
        if not isinstance(gamma, Real) or isinstance(gamma, (bool, np.bool_)):
            raise ValueError("gamma must be a finite real number in [0, 1].")
        gamma_value = float(gamma)
        if not math.isfinite(gamma_value) or gamma_value < 0.0 or gamma_value > 1.0:
            raise ValueError("gamma must be a finite real number in [0, 1].")
        if type(prototype_refinement) is not bool:
            raise ValueError("prototype_refinement must be a boolean (True or False).")
        if estimator == "none" and (weighting is not None or gamma_value != 0.0):
            raise ValueError("estimator='none' requires weighting=None and gamma=0.")
        if estimator != "none" and weighting is None:
            raise ValueError(
                "conditioned estimators require weighting='sample_weighted_rows' "
                "or weighting='class_balanced'."
            )

        if overlap_index_kwargs is not None and not isinstance(overlap_index_kwargs, Mapping):
            raise TypeError("overlap_index_kwargs must be a mapping or None.")
        copied_oi_kwargs = deepcopy(dict(overlap_index_kwargs or {}))
        if "prototype_refinement" in copied_oi_kwargs:
            raise ValueError(
                "prototype_refinement must be supplied explicitly, not inside "
                "overlap_index_kwargs."
            )

        if conditioning_kwargs is not None and not isinstance(conditioning_kwargs, Mapping):
            raise TypeError("conditioning_kwargs must be a mapping or None.")
        copied_conditioning_kwargs = deepcopy(dict(conditioning_kwargs or {}))
        allowed_controls = {"condition_number_cap", "relative_eigenvalue_floor"}
        unknown_controls = sorted(set(copied_conditioning_kwargs) - allowed_controls)
        if unknown_controls:
            raise ValueError(
                f"conditioning_kwargs contains unknown controls: {unknown_controls!r}."
            )
        for key, frozen in (
            ("condition_number_cap", CONDITION_NUMBER_CAP),
            ("relative_eigenvalue_floor", RELATIVE_EIGENVALUE_FLOOR),
        ):
            if key not in copied_conditioning_kwargs:
                continue
            value = copied_conditioning_kwargs[key]
            if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} is frozen at {frozen!r} for this protocol.")
            if not math.isfinite(float(value)) or float(value) != float(frozen):
                raise ValueError(f"{key} is frozen at {frozen!r} for this protocol.")

        self.estimator = estimator
        self.weighting = weighting
        self.gamma = gamma_value
        self.prototype_refinement = prototype_refinement
        self.conditioning_kwargs = copied_conditioning_kwargs
        self.overlap_index_kwargs = copied_oi_kwargs
        self.estimator_: Optional[OverlapIndex] = None
        self._conditioner: Optional[_FittedConditioner] = None
        self._conditioning_diagnostics: Optional[dict[str, Any]] = None
        self._runtime_diagnostics: Optional[dict[str, Any]] = None
        self._prototype_refinement: Optional[dict[str, Any]] = None
        self.n_features_in_: Optional[int] = None

    def _check_fitted(self) -> None:
        if self._conditioner is None or self.estimator_ is None:
            raise ValueError("This ConditionedOverlapIndex instance is not fit yet.")

    def _new_estimator(self) -> OverlapIndex:
        kwargs = deepcopy(self.overlap_index_kwargs)
        return OverlapIndex(prototype_refinement=self.prototype_refinement, **kwargs)

    def _validate_features(self, X: Any) -> np.ndarray:
        array = _validate_dense_matrix(
            X,
            "X",
            preserve_dtype=self._identity_path,
        )
        if self.n_features_in_ is not None and array.shape[1] != int(self.n_features_in_):
            raise ValueError(
                f"X has {array.shape[1]} features, but this ConditionedOverlapIndex "
                f"instance was fit with {self.n_features_in_} features."
            )
        return array

    @property
    def _identity_path(self) -> bool:
        return self.estimator == "none" or self.gamma == 0.0

    def fit(self, X: Any, Y: Any) -> "ConditionedOverlapIndex":
        """Fit conditioning and wrapped OI state from training rows only."""

        X_train = _validate_dense_matrix(
            X,
            "X",
            preserve_dtype=self._identity_path,
        )
        labels = _validate_labels(Y, X_train.shape[0])
        conditioning_input = (
            np.asarray(X_train, dtype=np.float64)
            if not self._identity_path
            else X_train
        )
        conditioning_wall_start = time.perf_counter()
        conditioning_cpu_start = time.process_time()
        conditioner = _fit_conditioner(
            conditioning_input,
            labels,
            estimator=self.estimator,
            weighting=self.weighting,
            gamma=self.gamma,
        )
        conditioning_fit_wall = time.perf_counter() - conditioning_wall_start
        conditioning_fit_cpu = time.process_time() - conditioning_cpu_start

        transformed = conditioner.transform(X_train)
        estimator = self._new_estimator()
        oi_fit_wall_start = time.perf_counter()
        oi_fit_cpu_start = time.process_time()
        # Pass the validated original target object through unchanged.  In
        # particular, this does not integer-encode heterogeneous scalar labels.
        estimator.fit(transformed, Y)
        oi_fit_wall = time.perf_counter() - oi_fit_wall_start
        oi_fit_cpu = time.process_time() - oi_fit_cpu_start

        # Publish state only after every fit stage succeeds, so a failed refit
        # cannot leave a partially updated wrapper.
        self._conditioner = conditioner
        self.estimator_ = estimator
        self.n_features_in_ = int(X_train.shape[1])
        self._conditioning_diagnostics = deepcopy(conditioner.diagnostics_)
        self._runtime_diagnostics = {
            "conditioning_fit_wall_seconds": float(max(0.0, conditioning_fit_wall)),
            "conditioning_fit_cpu_seconds": float(max(0.0, conditioning_fit_cpu)),
            "oi_fit_wall_seconds": float(max(0.0, oi_fit_wall)),
            "oi_fit_cpu_seconds": float(max(0.0, oi_fit_cpu)),
        }
        self._prototype_refinement = deepcopy(
            _json_safe(getattr(estimator, "prototype_refinement_", {}))
        )
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Apply the already-fitted transform without refitting."""

        self._check_fitted()
        return self._conditioner.transform(self._validate_features(X))

    def score_fixed(self, X: Any, Y: Any) -> float:
        """Score held-out rows against fixed conditioning and prototypes."""

        self._check_fitted()
        X_eval = self._validate_features(X)
        _validate_labels(Y, X_eval.shape[0])
        return float(self.estimator_.score_fixed(self._conditioner.transform(X_eval), Y))

    def predict(self, X: Any) -> np.ndarray:
        """Predict through the same frozen transformed geometry as scoring."""

        self._check_fitted()
        X_eval = self._validate_features(X)
        return np.asarray(self.estimator_.predict(self._conditioner.transform(X_eval)))

    @property
    def index(self) -> float:
        """Return the wrapped fit-time OI index."""

        self._check_fitted()
        return float(self.estimator_.index)

    @property
    def conditioner_(self) -> _FittedConditioner:
        """Return a detached read-only conditioner snapshot."""

        self._check_fitted()
        return self._conditioner.copy()

    @property
    def conditioning_diagnostics_(self) -> dict[str, Any]:
        """Return a detached JSON-safe structural diagnostics mapping."""

        self._check_fitted()
        return deepcopy(self._conditioning_diagnostics or {})

    @property
    def runtime_diagnostics_(self) -> dict[str, Any]:
        """Return detached fit timing diagnostics excluded from state identity."""

        self._check_fitted()
        return deepcopy(self._runtime_diagnostics or {})

    @property
    def prototype_refinement_(self) -> dict[str, Any]:
        """Return detached wrapped prototype-refinement diagnostics."""

        self._check_fitted()
        return deepcopy(self._prototype_refinement or {})
