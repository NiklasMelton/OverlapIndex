"""Private, fitted distance conditioning for the nuisance experiment.

This module intentionally does not add a public :mod:`overlapindex` option.
``ConditionedOverlapIndex`` fits a deterministic affine feature transform on
the rows supplied to :meth:`fit`, applies that fixed transform to both the
fit and hold-out rows, and delegates prototype fitting/scoring to the
existing :class:`overlapindex.OverlapIndex` estimator.

The four modes are frozen for this experiment:

``none``
    Identity transform.  This is the exact unconditioned baseline.
``global_isotropy``
    Global-centred OAS covariance with a full inverse square-root transform.
``pooled_diagonal``
    OAS covariance of residuals around each class training mean, using only
    its diagonal for scaling.
``pooled_full``
    OAS covariance of the same pooled within-class residuals, with a full
    inverse square-root transform.

All non-identity transforms use float64 arithmetic, a relative eigenvalue
floor of ``1e-8``, and a maximum covariance condition number of ``1e4``.
Neither the transform nor the underlying prototypes are refit by
:meth:`score_fixed`.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import time
from typing import Any, Optional

import numpy as np
from scipy import sparse
from sklearn.covariance import oas

from overlapindex import OverlapIndex


CONDITIONING_MODES = (
    "none",
    "global_isotropy",
    "pooled_diagonal",
    "pooled_full",
)
"""Closed set of conditioning modes used by the frozen protocol."""

RELATIVE_EIGENVALUE_FLOOR = 1e-8
"""Relative covariance eigenvalue floor used by all fitted modes."""

CONDITION_NUMBER_CAP = 1e4
"""Maximum condition number of the regularized covariance."""


def _validate_dense_matrix(
    X: Any,
    name: str,
    *,
    preserve_dtype: bool = False,
) -> np.ndarray:
    """Return finite dense rows and reject sparse/non-matrix inputs.

    ``none`` deliberately keeps the caller's ordinary numeric dtype so its
    identity path preserves the same target/data semantics as direct OI.
    Conditioned modes request float64 explicitly.
    """

    if sparse.issparse(X):
        raise TypeError(f"{name} must be a finite dense 2D array.")
    try:
        raw = np.asarray(X)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite dense 2D array.") from exc
    if raw.ndim != 2:
        raise ValueError(
            f"{name} must be a finite dense 2D array; got shape {raw.shape}."
        )
    if raw.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature column.")
    if raw.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real-valued features.")
    try:
        array = raw if preserve_dtype else np.asarray(raw, dtype=np.float64)
        finite = np.isfinite(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite dense 2D array.") from exc
    if not np.all(finite):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _validate_scalar_label(label: Any) -> None:
    """Require one hashable, scalar, non-missing label."""

    if isinstance(label, (list, tuple, set, frozenset, dict)):
        raise ValueError("Labels must be scalar single-label values.")
    if isinstance(label, np.ndarray) and label.ndim != 0:
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
        is_missing = bool(unequal_to_self)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Labels must be scalar, non-missing values with unambiguous equality."
        ) from exc
    if is_missing:
        raise ValueError("Labels must not contain None or NaN values.")


def _validate_labels(y: Any, n_rows: int) -> np.ndarray:
    """Validate scalar labels while preserving their ordinary array dtype."""

    if isinstance(y, (str, bytes)):
        raise ValueError("Y must be a one-dimensional scalar-label sequence.")
    try:
        labels = np.asarray(y)
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
    return labels


def _json_number(value: float) -> Optional[float]:
    """Return a finite JSON number, or ``None`` for undefined extrema."""

    value = float(value)
    return value if np.isfinite(value) else None


def _condition_number(eigenvalues: np.ndarray) -> Optional[float]:
    """Return a JSON-safe covariance condition number."""

    values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    smallest = float(np.min(values))
    largest = float(np.max(values))
    if smallest <= 0.0 or not np.isfinite(smallest) or not np.isfinite(largest):
        return None
    return _json_number(largest / smallest)


def _canonical_bytes(array: np.ndarray) -> bytes:
    """Encode a numeric array deterministically for the fitted-state hash."""

    canonical = np.asarray(array, dtype="<f8", order="C")
    shape = np.asarray(canonical.shape, dtype="<i8")
    return shape.tobytes() + canonical.tobytes(order="C")


def _state_hash(
    mode: str,
    mean: np.ndarray,
    covariance: np.ndarray,
    transform: np.ndarray,
) -> str:
    """Hash the immutable numeric state of a fitted conditioner."""

    digest = hashlib.sha256()
    digest.update(mode.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_bytes(mean))
    digest.update(_canonical_bytes(covariance))
    digest.update(_canonical_bytes(transform))
    return digest.hexdigest()


def _make_readonly(array: np.ndarray) -> np.ndarray:
    """Return a float64 C-contiguous array that cannot be modified in place."""

    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _safe_copy(value: Any) -> Any:
    """Convert fitted estimator diagnostics into JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return [_safe_copy(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _safe_copy(value.item())
    if isinstance(value, Mapping):
        return {str(key): _safe_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_copy(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return _json_number(value)
    return str(value)


class _FittedConditioner:
    """Immutable fitted affine transform used by ``ConditionedOverlapIndex``."""

    def __init__(
        self,
        *,
        mode: str,
        mean: np.ndarray,
        transform: np.ndarray,
        covariance: np.ndarray,
        diagnostics: Mapping[str, Any],
    ) -> None:
        self.mode = str(mode)
        self.mean_ = _make_readonly(mean)
        self.transform_ = _make_readonly(transform)
        self.covariance_ = _make_readonly(covariance)
        self.n_features_in_ = int(self.mean_.size)
        self.diagnostics_ = dict(_safe_copy(diagnostics))
        self.state_sha256_ = str(self.diagnostics_["state_sha256"])

    def copy(self) -> "_FittedConditioner":
        """Return a detached read-only snapshot for external inspection."""

        return _FittedConditioner(
            mode=self.mode,
            mean=self.mean_,
            transform=self.transform_,
            covariance=self.covariance_,
            diagnostics=self.diagnostics_,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply this fitted transform without consulting any new labels/data."""

        if self.mode == "none":
            # The validated input is intentionally returned unchanged.  This
            # keeps the unconditioned candidate an exact identity path.
            return X
        centered = np.asarray(X, dtype=np.float64) - self.mean_
        return np.asarray(centered @ self.transform_, dtype=np.float64)


def _regularized_eigenvalues(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply the frozen relative floor and condition cap to covariance values."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy(), float(RELATIVE_EIGENVALUE_FLOOR)
    values = np.maximum(values, 0.0)
    largest = float(np.max(values))
    if not np.isfinite(largest) or largest <= 0.0:
        # identity is the stable, geometry-preserving transform in this case.
        return np.ones_like(values), 1.0
    relative_floor = largest * float(RELATIVE_EIGENVALUE_FLOOR)
    cap_floor = largest / float(CONDITION_NUMBER_CAP)
    effective_floor = max(relative_floor, cap_floor)
    regularized = np.maximum(values, effective_floor)
    return regularized, float(effective_floor)


def _fit_covariance(
    X: np.ndarray,
    *,
    diagonal: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit OAS and return covariance, shrinkage, and raw eigenvalues."""

    covariance, shrinkage = oas(X, assume_centered=True)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    if diagonal:
        covariance_for_transform = np.diag(np.diag(covariance))
        raw_values = np.asarray(np.diag(covariance_for_transform), dtype=np.float64)
    else:
        covariance_for_transform = covariance
        raw_values = np.linalg.eigvalsh(covariance_for_transform)
    return covariance_for_transform, float(shrinkage), np.asarray(raw_values, dtype=np.float64)


def _fit_conditioner(
    X: np.ndarray,
    labels: Optional[np.ndarray],
    mode: str,
    *,
    class_count: Optional[int] = None,
) -> _FittedConditioner:
    """Fit one of the frozen transforms using training rows only."""

    n_rows, n_features = X.shape
    global_mean = np.asarray(np.mean(X, axis=0), dtype=np.float64)
    if class_count is None:
        if labels is None:
            raise ValueError("labels are required to fit a conditioned mode")
        class_count = int(len(dict.fromkeys(labels.tolist())))
    identity_covariance = np.eye(n_features, dtype=np.float64)

    if mode == "none":
        transform = identity_covariance
        covariance = identity_covariance
        diagnostics = {
            "mode": mode,
            "n_rows_fit": int(n_rows),
            "n_features_fit": int(n_features),
            "n_classes_fit": int(class_count),
            "shrinkage": None,
            "eigen_min_before": None,
            "eigen_max_before": None,
            "eigen_min_after": None,
            "eigen_max_after": None,
            "condition_before": None,
            "condition_after": None,
            "relative_eigenvalue_floor": None,
            "condition_number_cap": None,
            "effective_eigenvalue_floor": None,
            "covariance_scope": None,
            "residual_weighting": None,
            "structure": "identity",
            "estimator": None,
            "regularized_eigenvalue_count": 0,
            "cap_or_floor_active": False,
        }
    elif mode == "global_isotropy":
        centered = np.asarray(X - global_mean, dtype=np.float64)
        covariance, shrinkage, raw_values = _fit_covariance(
            centered,
            diagonal=False,
        )
        if raw_values.size:
            eigvals, eigvecs = np.linalg.eigh(covariance)
            eigvals = np.maximum(eigvals, 0.0)
            regularized, eigen_floor = _regularized_eigenvalues(eigvals)
            transform = eigvecs @ np.diag(1.0 / np.sqrt(regularized)) @ eigvecs.T
            raw_values = eigvals
        else:  # pragma: no cover - X always has at least one feature
            transform = identity_covariance
        diagnostics = {
            "mode": mode,
            "n_rows_fit": int(n_rows),
            "n_features_fit": int(n_features),
            "n_classes_fit": int(class_count),
            "shrinkage": float(shrinkage),
            "eigen_min_before": _json_number(float(np.min(raw_values))),
            "eigen_max_before": _json_number(float(np.max(raw_values))),
            "eigen_min_after": _json_number(float(np.min(regularized))),
            "eigen_max_after": _json_number(float(np.max(regularized))),
            "condition_before": _condition_number(raw_values),
            "condition_after": _condition_number(regularized),
            "relative_eigenvalue_floor": float(RELATIVE_EIGENVALUE_FLOOR),
            "condition_number_cap": float(CONDITION_NUMBER_CAP),
            "effective_eigenvalue_floor": float(eigen_floor),
            "covariance_scope": "global",
            "residual_weighting": None,
            "structure": "full",
            "estimator": "oas",
            "regularized_eigenvalue_count": int(
                np.count_nonzero(regularized != raw_values)
            ),
            "cap_or_floor_active": bool(np.any(regularized != raw_values)),
        }
    elif mode in {"pooled_diagonal", "pooled_full"}:
        if labels is None:
            raise ValueError("labels are required to fit a pooled conditioned mode")
        # Dict insertion order is the validated first-observed class order;
        # it makes the residual stack deterministic for any scalar labels.
        rows_by_class: dict[Any, list[int]] = {}
        for row_idx, label in enumerate(labels.tolist()):
            rows_by_class.setdefault(label, []).append(int(row_idx))
        residual_blocks = []
        for row_indices in rows_by_class.values():
            class_rows = X[np.asarray(row_indices, dtype=int)]
            class_mean = np.mean(class_rows, axis=0)
            residual_blocks.append(class_rows - class_mean)
        residuals = np.vstack(residual_blocks).astype(np.float64, copy=False)
        diagonal = mode == "pooled_diagonal"
        covariance, shrinkage, raw_values = _fit_covariance(
            residuals,
            diagonal=diagonal,
        )
        regularized, eigen_floor = _regularized_eigenvalues(raw_values)
        if diagonal:
            transform = np.diag(1.0 / np.sqrt(regularized))
        else:
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # ``covariance`` is the symmetric OAS output.  Re-clipping here
            # is defensive against tiny negative roundoff in eigh.
            eigvals = np.maximum(eigvals, 0.0)
            regularized_full, _ = _regularized_eigenvalues(eigvals)
            transform = eigvecs @ np.diag(1.0 / np.sqrt(regularized_full)) @ eigvecs.T
            regularized = regularized_full
            raw_values = eigvals
        diagnostics = {
            "mode": mode,
            "n_rows_fit": int(n_rows),
            "n_features_fit": int(n_features),
            "n_classes_fit": int(class_count),
            "shrinkage": float(shrinkage),
            "eigen_min_before": _json_number(float(np.min(raw_values))),
            "eigen_max_before": _json_number(float(np.max(raw_values))),
            "eigen_min_after": _json_number(float(np.min(regularized))),
            "eigen_max_after": _json_number(float(np.max(regularized))),
            "condition_before": _condition_number(raw_values),
            "condition_after": _condition_number(regularized),
            "relative_eigenvalue_floor": float(RELATIVE_EIGENVALUE_FLOOR),
            "condition_number_cap": float(CONDITION_NUMBER_CAP),
            "effective_eigenvalue_floor": float(eigen_floor),
            "covariance_scope": "pooled_within_class",
            "residual_weighting": "sample_weighted_rows",
            "structure": "diagonal" if diagonal else "full",
            "estimator": "oas",
            "regularized_eigenvalue_count": int(
                np.count_nonzero(regularized != raw_values)
            ),
            "cap_or_floor_active": bool(np.any(regularized != raw_values)),
        }
    else:  # pragma: no cover - mode is checked at the public boundary
        raise ValueError(f"Unknown conditioning mode: {mode!r}.")

    state_mean = np.zeros_like(global_mean) if mode == "none" else global_mean
    state_sha256 = _state_hash(mode, state_mean, covariance, transform)
    diagnostics["state_sha256"] = state_sha256
    return _FittedConditioner(
        mode=mode,
        mean=state_mean,
        transform=transform,
        covariance=covariance,
        diagnostics=diagnostics,
    )


class ConditionedOverlapIndex:
    """Research-only fitted distance-conditioning wrapper around OverlapIndex.

    Parameters
    ----------
    mode : str, default="none"
        One of ``none``, ``global_isotropy``, ``pooled_diagonal``, or
        ``pooled_full``.
    prototype_refinement : bool, optional
        Explicit refinement setting.  It is always forwarded to the fresh
        :class:`OverlapIndex` and must be a genuine Python ``bool``.
    conditioning_kwargs : mapping, optional
        Predeclared conditioning controls.  The only accepted keys are
        ``condition_number_cap`` and ``relative_eigenvalue_floor``; both are
        frozen to :data:`CONDITION_NUMBER_CAP` and
        :data:`RELATIVE_EIGENVALUE_FLOOR` respectively.
        This mapping exists to make the protocol manifest explicit without
        permitting evaluation-driven tuning.
    overlap_index_kwargs : mapping, optional
        One copied mapping of keyword arguments for a fresh
        :class:`OverlapIndex`.  ``prototype_refinement`` must not appear in
        this mapping because it is an explicit wrapper argument.

    Notes
    -----
    All modes accept only finite dense two-dimensional features and scalar
    single-label targets.  This intentionally keeps the private experiment's
    target scope narrower than the public estimator's broader label API.
    """

    def __init__(
        self,
        mode: str = "none",
        *,
        prototype_refinement: bool = False,
        conditioning_kwargs: Optional[Mapping[str, Any]] = None,
        overlap_index_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(mode, str) or mode not in CONDITIONING_MODES:
            raise ValueError(
                f"mode must be one of {CONDITIONING_MODES!r}; got {mode!r}."
            )
        if type(prototype_refinement) is not bool:
            raise ValueError("prototype_refinement must be a boolean (True or False).")
        if overlap_index_kwargs is not None and not isinstance(
            overlap_index_kwargs, Mapping
        ):
            raise TypeError("overlap_index_kwargs must be a mapping or None.")
        if overlap_index_kwargs is not None and "prototype_refinement" in overlap_index_kwargs:
            raise ValueError(
                "prototype_refinement must be supplied explicitly, not inside "
                "overlap_index_kwargs."
            )

        if conditioning_kwargs is not None and not isinstance(
            conditioning_kwargs, Mapping
        ):
            raise TypeError("conditioning_kwargs must be a mapping or None.")
        frozen_conditioning_kwargs = dict(conditioning_kwargs or {})
        allowed_conditioning_keys = {
            "condition_number_cap",
            "relative_eigenvalue_floor",
        }
        unknown_conditioning_keys = sorted(
            set(frozen_conditioning_kwargs) - allowed_conditioning_keys
        )
        if unknown_conditioning_keys:
            raise ValueError(
                "conditioning_kwargs contains unknown controls: "
                f"{unknown_conditioning_keys!r}."
            )
        if (
            "condition_number_cap" in frozen_conditioning_kwargs
            and float(frozen_conditioning_kwargs["condition_number_cap"])
            != float(CONDITION_NUMBER_CAP)
        ):
            raise ValueError(
                "condition_number_cap is frozen at "
                f"{CONDITION_NUMBER_CAP!r} for this protocol."
            )
        if (
            "relative_eigenvalue_floor" in frozen_conditioning_kwargs
            and float(frozen_conditioning_kwargs["relative_eigenvalue_floor"])
            != float(RELATIVE_EIGENVALUE_FLOOR)
        ):
            raise ValueError(
                "relative_eigenvalue_floor is frozen at "
                f"{RELATIVE_EIGENVALUE_FLOOR!r} for this protocol."
            )

        self.mode = str(mode)
        self.prototype_refinement = prototype_refinement
        self.conditioning_kwargs = deepcopy(frozen_conditioning_kwargs)
        self.overlap_index_kwargs = deepcopy(dict(overlap_index_kwargs or {}))

        # Fitted attributes are deliberately created only by fit.  This makes
        # accidental score_fixed calls fail loudly instead of using stale state.
        self._conditioner: Optional[_FittedConditioner] = None
        self.estimator_: Optional[OverlapIndex] = None
        self._conditioning_diagnostics: Optional[dict[str, Any]] = None
        self._runtime_diagnostics: Optional[dict[str, Any]] = None
        self._prototype_refinement: Optional[dict[str, Any]] = None
        self.n_features_in_: Optional[int] = None

    def _new_estimator(self) -> OverlapIndex:
        """Build an unfitted estimator from copied options."""

        kwargs = deepcopy(self.overlap_index_kwargs)
        return OverlapIndex(
            prototype_refinement=self.prototype_refinement,
            **kwargs,
        )

    def _check_fitted(self) -> None:
        """Raise a stable error when transform/scoring precedes fit."""

        if self._conditioner is None or self.estimator_ is None:
            raise ValueError("This ConditionedOverlapIndex instance is not fit yet.")

    def fit(self, X: Any, Y: Any) -> "ConditionedOverlapIndex":
        """Fit the conditioner and OI prototypes on the supplied rows."""

        identity_mode = self.mode == "none"
        X_train = _validate_dense_matrix(
            X,
            "X",
            preserve_dtype=identity_mode,
        )
        labels = _validate_labels(Y, X_train.shape[0])
        conditioning_labels = labels
        class_count = int(len(dict.fromkeys(labels.tolist())))
        targets = labels
        conditioning_wall_start = time.perf_counter()
        conditioning_cpu_start = time.process_time()
        conditioner = _fit_conditioner(
            X_train,
            conditioning_labels,
            self.mode,
            class_count=class_count,
        )
        conditioning_wall_seconds = time.perf_counter() - conditioning_wall_start
        conditioning_cpu_seconds = time.process_time() - conditioning_cpu_start
        X_conditioned = conditioner.transform(X_train)
        estimator = self._new_estimator()
        # The explicit refinement switch reaches the current public
        # constructor without altering that API.
        estimator.fit(X_conditioned, targets)

        self._conditioner = conditioner
        self.estimator_ = estimator
        self.n_features_in_ = int(X_train.shape[1])
        self._conditioning_diagnostics = dict(_safe_copy(conditioner.diagnostics_))
        self._runtime_diagnostics = {
            "conditioning_fit_wall_seconds": _json_number(
                max(0.0, conditioning_wall_seconds)
            ),
            "conditioning_fit_cpu_seconds": _json_number(
                max(0.0, conditioning_cpu_seconds)
            ),
        }
        self._prototype_refinement = _safe_copy(
            getattr(estimator, "prototype_refinement_", {})
        )
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Apply the already fitted transform to dense rows."""

        self._check_fitted()
        X_array = _validate_dense_matrix(
            X,
            "X",
            preserve_dtype=self.mode == "none",
        )
        if X_array.shape[1] != int(self.n_features_in_):
            raise ValueError(
                f"X has {X_array.shape[1]} features, but this "
                f"ConditionedOverlapIndex instance was fit with "
                f"{self.n_features_in_} features."
            )
        return self._conditioner.transform(X_array)

    def score_fixed(self, X: Any, Y: Any) -> float:
        """Score hold-out rows with fixed conditioning and fixed prototypes."""

        self._check_fitted()
        X_eval = _validate_dense_matrix(
            X,
            "X",
            preserve_dtype=self.mode == "none",
        )
        targets = _validate_labels(Y, X_eval.shape[0])
        if X_eval.shape[1] != int(self.n_features_in_):
            raise ValueError(
                f"X has {X_eval.shape[1]} features, but this "
                f"ConditionedOverlapIndex instance was fit with "
                f"{self.n_features_in_} features."
            )
        # There is intentionally no fit/refit call in this path.  OI's own
        # score_fixed recomputes hold-out bookkeeping but reuses its centers.
        return float(
            self.estimator_.score_fixed(
                self._conditioner.transform(X_eval),
                targets,
            )
        )

    def predict(self, X: Any) -> np.ndarray:
        """Predict through the same fixed geometry used by ``score_fixed``."""

        self._check_fitted()
        X_eval = _validate_dense_matrix(
            X,
            "X",
            preserve_dtype=self.mode == "none",
        )
        if X_eval.shape[1] != int(self.n_features_in_):
            raise ValueError(
                f"X has {X_eval.shape[1]} features, but this "
                f"ConditionedOverlapIndex instance was fit with "
                f"{self.n_features_in_} features."
            )
        return np.asarray(self.estimator_.predict(self._conditioner.transform(X_eval)))

    @property
    def index(self) -> float:
        """Return the fit-time OI index after checking fitted state."""

        self._check_fitted()
        return float(self.estimator_.index)

    @property
    def conditioner_(self) -> _FittedConditioner:
        """Expose a detached read-only snapshot of the fitted transform."""

        self._check_fitted()
        return self._conditioner.copy()

    @property
    def conditioning_diagnostics_(self) -> dict[str, Any]:
        """Return a detached JSON-safe copy of conditioning diagnostics."""

        self._check_fitted()
        return deepcopy(self._conditioning_diagnostics or {})

    @property
    def runtime_diagnostics_(self) -> dict[str, Any]:
        """Return timing diagnostics kept outside deterministic state."""

        self._check_fitted()
        return deepcopy(self._runtime_diagnostics or {})

    @property
    def prototype_refinement_(self) -> dict[str, Any]:
        """Return a detached copy of fitted prototype-refinement diagnostics."""

        self._check_fitted()
        return deepcopy(self._prototype_refinement or {})
