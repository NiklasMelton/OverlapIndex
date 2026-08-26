"""Mechanistic, read-only diagnostics for the heteroscedastic experiment.

The functions in this module deliberately sit outside both the public
``OverlapIndex`` API and the experiment runner.  They consume already-fitted
objects and caller-supplied arrays, and never fit an estimator or infer a
prototype graph.  In particular, the oracle used by the neighbour diagnostics
is a view of the supplied signal geometry; it is not a model and cannot enter
candidate selection.

The module is intentionally conservative about what it reports.  A prototype
id is an assignment, not an explanation of a prototype's score contribution,
and a backend owner is not a probabilistic assignment.  Consequently this
module reports occupied-prototype label composition and owner agreement, but
does not invent owner entropy, a prototype adjacency graph, or per-prototype
score contributions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Optional

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold


def _validate_dense_matrix(value: Any, name: str) -> np.ndarray:
    """Return a detached finite dense numeric matrix.

    Diagnostic transforms are intentionally dense: this keeps the metric and
    oracle paths identical and prevents sparse input from silently selecting a
    different distance implementation.  The returned copy is owned by the
    diagnostic and is never written back to the caller.
    """

    if sparse.issparse(value):
        raise TypeError(f"{name} must be a finite dense 2D numeric array.")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite dense 2D numeric array.") from exc
    if raw.ndim != 2 or raw.shape[1] <= 0:
        raise ValueError(
            f"{name} must be a finite dense 2D numeric array; got shape {raw.shape}."
        )
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real-valued features.")
    try:
        result = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite dense 2D numeric array.") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return result


def _validate_labels(value: Any, n_rows: int, name: str = "labels") -> np.ndarray:
    """Validate scalar, hashable, non-missing labels without recoding them."""

    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional scalar-label sequence.")
    try:
        labels = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a one-dimensional scalar-label sequence."
        ) from exc
    if labels.ndim != 1 or int(labels.shape[0]) != int(n_rows):
        raise ValueError(
            f"{name} must have shape ({int(n_rows)},); got {labels.shape}."
        )
    # Keep the validated labels' ordinary dtype.  Heterogeneous labels already
    # arrive as ``object``; homogeneous integer/string labels must not be
    # recast merely because the split planner has a separate integer encoding.
    labels = np.array(labels, copy=True)
    for label in labels.tolist():
        _validate_scalar_label(label, name)
    return labels


def _validate_scalar_label(label: Any, name: str = "labels") -> None:
    """Validate one scalar label using the experiment's strict target scope."""

    if isinstance(label, (list, tuple, set, frozenset, dict)):
        raise ValueError(f"{name} must contain scalar single-label values.")
    if isinstance(label, np.ndarray) and label.ndim != 0:
        raise ValueError(f"{name} must contain scalar single-label values.")
    if not isinstance(label, (str, bytes)) and not np.isscalar(label):
        raise ValueError(f"{name} must contain scalar single-label values.")
    try:
        hash(label)
    except TypeError as exc:
        raise TypeError(f"{name} labels must be hashable scalar values.") from exc
    if label is None:
        raise ValueError(f"{name} must not contain None or NaN values.")
    try:
        missing = bool(label != label)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} labels must have unambiguous scalar equality."
        ) from exc
    if missing:
        raise ValueError(f"{name} must not contain None or NaN values.")


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    """Return first-observed unique scalar values."""

    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def json_safe(value: Any) -> Any:
    """Convert a diagnostic value to finite, deterministic JSON data.

    Non-finite floating-point values become ``None``.  This is intentionally a
    public helper so artifact writers and tests can apply exactly the same
    conversion to nested diagnostic payloads.
    """

    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return repr(value)
    if isinstance(value, (set, frozenset)):
        # Sets are not emitted by the canonical diagnostics, but keeping this
        # utility deterministic makes accidental nested set values safe too.
        ordered = sorted(value, key=lambda item: canonical_json(item))
        return [json_safe(item) for item in ordered]
    if isinstance(value, Mapping):
        # Diagnostic mappings use string schema keys.  Stringifying arbitrary
        # keys keeps this helper total without relying on Python's mixed-type
        # ordering rules.
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    return repr(value)


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by diagnostic hashes."""

    return json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class MetricSpaceView(Mapping[str, np.ndarray]):
    """Detached raw and conditioned feature spaces.

    Both arrays are read-only copies.  Mapping access (``view["raw"]`` and
    ``view["conditioned"]``) is provided alongside attributes because the
    runner serializes diagnostics as mappings while tests and reviewers often
    inspect the arrays directly.
    """

    raw: np.ndarray
    conditioned: np.ndarray

    def __post_init__(self) -> None:
        raw = _validate_dense_matrix(self.raw, "raw")
        conditioned = _validate_dense_matrix(self.conditioned, "conditioned")
        if raw.shape != conditioned.shape:
            raise ValueError(
                "raw and conditioned metric spaces must have identical shapes; "
                f"got {raw.shape} and {conditioned.shape}."
            )
        raw.setflags(write=False)
        conditioned.setflags(write=False)
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "conditioned", conditioned)

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "raw":
            return self.raw
        if key == "conditioned":
            return self.conditioned
        raise KeyError(key)

    def __iter__(self):
        return iter(("raw", "conditioned"))

    def __len__(self) -> int:
        return 2


def metric_space_view(
    X: Any,
    *,
    conditioned: Any = None,
    transform: Any = None,
) -> MetricSpaceView:
    """Build a raw/conditioned metric-space view without fitting.

    Parameters
    ----------
    X:
        Raw rows.  If ``X`` is already a :class:`MetricSpaceView`, its raw
        component is used and, when no other source is supplied, its existing
        conditioned component is retained.
    conditioned:
        Optional already-transformed rows.  Supplying this and ``transform``
        together is an error; the caller must make the geometry choice
        explicit.
    transform:
        A fitted object exposing ``transform`` or a callable.  It is called
        exactly once and never fitted.  Its output is validated and detached.
    """

    existing = X if isinstance(X, MetricSpaceView) else None
    raw = _validate_dense_matrix(existing.raw if existing is not None else X, "X")
    if conditioned is not None and transform is not None:
        raise ValueError("conditioned and transform are mutually exclusive.")
    if conditioned is None and transform is None and existing is not None:
        transformed = np.array(existing.conditioned, dtype=np.float64, copy=True)
    elif conditioned is not None:
        transformed = _validate_dense_matrix(conditioned, "conditioned")
    elif transform is None:
        transformed = np.array(raw, dtype=np.float64, copy=True)
    else:
        apply = getattr(transform, "transform", None)
        if apply is None:
            if not callable(transform):
                raise TypeError("transform must expose transform(X) or be callable.")
            apply = transform
        try:
            candidate = apply(np.array(raw, copy=True))
        except (TypeError, ValueError) as exc:
            raise ValueError("fitted transform rejected the diagnostic rows.") from exc
        transformed = _validate_dense_matrix(candidate, "conditioned")
    if raw.shape != transformed.shape:
        raise ValueError(
            "conditioned metric rows must preserve raw shape; "
            f"got {raw.shape} and {transformed.shape}."
        )
    return MetricSpaceView(raw=raw, conditioned=transformed)


def _metric_array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, MetricSpaceView):
        return _validate_dense_matrix(value.conditioned, name)
    return _validate_dense_matrix(value, name)


def _validate_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def top_k_neighbors(X: Any, k: int) -> np.ndarray:
    """Return deterministic Euclidean top-k neighbors, excluding self."""

    values = _metric_array(X, "X")
    k_requested = _validate_positive_integer(k, "k")
    n_rows = int(values.shape[0])
    k_effective = min(k_requested, max(0, n_rows - 1))
    if k_effective == 0:
        return np.empty((n_rows, 0), dtype=int)
    # Squared Euclidean distance has the same ordering as Euclidean distance.
    distances = np.sum(
        (values[:, None, :] - values[None, :, :]) ** 2,
        axis=2,
        dtype=np.float64,
    )
    np.fill_diagonal(distances, np.inf)
    # Mergesort gives row-index tie breaking because columns are in original
    # index order.  This tie rule is part of the diagnostic, not an accidental
    # dependency on NumPy's unstable default quicksort.
    ordered = np.argsort(distances, axis=1, kind="mergesort")
    return np.asarray(ordered[:, :k_effective], dtype=int)


def neighbor_diagnostics(
    metric_X: Any,
    oracle_X: Any,
    labels: Any,
    k: int,
    *,
    separated_mask: Any = None,
) -> dict[str, Any]:
    """Measure oracle-neighbor recovery and separated cross-class impurity.

    ``separated_mask`` is supplied by the caller from fixture truth.  When it
    is present, only those rows contribute to both means.  Genuine-overlap
    rows are never guessed to be separated from their labels.
    """

    metric = _metric_array(metric_X, "metric_X")
    oracle = _metric_array(oracle_X, "oracle_X")
    if metric.shape[0] != oracle.shape[0]:
        raise ValueError("metric_X and oracle_X must have the same row count.")
    labels_array = _validate_labels(labels, metric.shape[0])
    k_requested = _validate_positive_integer(k, "k")
    if separated_mask is None:
        selected = np.ones(metric.shape[0], dtype=bool)
        separated_only = False
    else:
        selected = np.asarray(separated_mask)
        if selected.ndim != 1 or selected.shape[0] != metric.shape[0]:
            raise ValueError("separated_mask must align with metric_X rows.")
        if selected.dtype != np.dtype(bool):
            raise ValueError("separated_mask must have dtype=bool.")
        selected = selected.astype(bool, copy=True)
        separated_only = True

    metric_neighbors = top_k_neighbors(metric, k_requested)
    oracle_neighbors = top_k_neighbors(oracle, k_requested)
    row_indices = np.flatnonzero(selected)
    k_effective = int(metric_neighbors.shape[1])
    if row_indices.size == 0 or k_effective == 0:
        jaccard = None
        impurity = None
    else:
        jaccard_values: list[float] = []
        impurity_values: list[float] = []
        for row in row_indices.tolist():
            metric_set = set(metric_neighbors[row].tolist())
            oracle_set = set(oracle_neighbors[row].tolist())
            union_size = len(metric_set | oracle_set)
            jaccard_values.append(
                float(len(metric_set & oracle_set) / union_size)
                if union_size
                else 1.0
            )
            source_label = labels_array[row]
            impurity_values.append(
                float(
                    np.mean(
                        [
                            bool(labels_array[neighbor] != source_label)
                            for neighbor in metric_neighbors[row].tolist()
                        ]
                    )
                )
            )
        jaccard = float(np.mean(jaccard_values))
        impurity = float(np.mean(impurity_values))
    return json_safe(
        {
            "oracle_neighbor_jaccard": jaccard,
            "cross_class_neighbor_impurity": impurity,
            "n_rows": int(metric.shape[0]),
            "n_rows_evaluated": int(row_indices.size),
            "k_requested": int(k_requested),
            "k_effective": int(k_effective),
            "separated_only": bool(separated_only),
        }
    )


def _validate_pairs(pairs: Any, n_rows: int) -> list[tuple[int, int]]:
    """Validate caller-supplied pair identities without sampling or sorting."""

    try:
        raw = list(pairs)
    except TypeError as exc:
        raise ValueError("pairs must be a sequence of two-index pairs.") from exc
    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in raw:
        try:
            values = list(pair)
        except TypeError as exc:
            raise ValueError("each pair must contain exactly two indices.") from exc
        if len(values) != 2:
            raise ValueError("each pair must contain exactly two indices.")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in values
        ):
            raise ValueError("pair indices must be integers (not booleans or floats).")
        try:
            first = int(values[0])
            second = int(values[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("pair indices must be integers.") from exc
        if first < 0 or second < 0 or first >= n_rows or second >= n_rows:
            raise ValueError(
                f"pair ({first}, {second}) is outside the {n_rows}-row input."
            )
        if first == second:
            raise ValueError("pair identities must be off-diagonal.")
        normalized = (first, second)
        if normalized in seen:
            raise ValueError(f"duplicate pair identity: {normalized!r}.")
        seen.add(normalized)
        result.append(normalized)
    return result


def _pair_distances(X: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.empty(0, dtype=np.float64)
    return np.asarray(
        [float(np.linalg.norm(X[first] - X[second])) for first, second in pairs],
        dtype=np.float64,
    )


def _spearman(values_a: np.ndarray, values_b: np.ndarray) -> Optional[float]:
    """Tie-safe Spearman correlation with explicit degenerate semantics."""

    if values_a.size != values_b.size:
        raise ValueError("Spearman inputs must have equal length.")
    if values_a.size < 2:
        return None
    ranks_a = np.asarray(rankdata(values_a, method="average"), dtype=np.float64)
    ranks_b = np.asarray(rankdata(values_b, method="average"), dtype=np.float64)
    centered_a = ranks_a - float(np.mean(ranks_a))
    centered_b = ranks_b - float(np.mean(ranks_b))
    norm_a = float(np.linalg.norm(centered_a))
    norm_b = float(np.linalg.norm(centered_b))
    if norm_a == 0.0 or norm_b == 0.0:
        # A constant-vs-constant input has identical rank ordering; a
        # constant-vs-varying input has no recoverable rank relationship.
        return (
            1.0
            if norm_a == 0.0
            and norm_b == 0.0
            and np.array_equal(values_a, values_b)
            else 0.0
        )
    value = float(np.dot(centered_a, centered_b) / (norm_a * norm_b))
    return float(np.clip(value, -1.0, 1.0))


def pair_distance_diagnostics(
    metric_X: Any,
    oracle_X: Any,
    pairs: Any,
) -> dict[str, Any]:
    """Compare distances on an exact caller-supplied pair sample.

    No pair sampling, sorting, deduplication, or hidden RNG occurs here.  The
    returned identity digest lets an artifact consumer verify that all methods
    used the same pair rows.
    """

    metric = _metric_array(metric_X, "metric_X")
    oracle = _metric_array(oracle_X, "oracle_X")
    if metric.shape[0] != oracle.shape[0]:
        raise ValueError("metric_X and oracle_X must have the same row count.")
    identities = _validate_pairs(pairs, metric.shape[0])
    metric_distances = _pair_distances(metric, identities)
    oracle_distances = _pair_distances(oracle, identities)
    identity_payload = [[int(first), int(second)] for first, second in identities]
    identity_sha256 = hashlib.sha256(
        canonical_json(identity_payload).encode("utf-8")
    ).hexdigest()
    return json_safe(
        {
            "pair_distance_spearman": _spearman(metric_distances, oracle_distances),
            "n_pairs": int(len(identities)),
            "pair_indices": identity_payload,
            "pair_identity_sha256": identity_sha256,
            "metric_n_features": int(metric.shape[1]),
            "oracle_n_features": int(oracle.shape[1]),
        }
    )


def _first_observed_integer_encoding(labels: np.ndarray) -> np.ndarray:
    """Encode labels for sklearn splitting while preserving original labels."""

    positions: dict[Any, int] = {}
    encoded = np.empty(labels.shape[0], dtype=np.int64)
    for row, label in enumerate(labels.tolist()):
        if label not in positions:
            positions[label] = len(positions)
        encoded[row] = int(positions[label])
    return encoded


def _new_conditioner(factory: Any) -> Any:
    """Create a fresh conditioner from a class, factory, or template object."""

    if isinstance(factory, type):
        return factory()
    if callable(factory):
        return factory()
    return deepcopy(factory)


def _conditioner_scale_vector(conditioner: Any, n_features: int) -> np.ndarray:
    """Extract a deterministic per-output-feature scale from fitted state."""

    candidates = [conditioner]
    fitted = getattr(conditioner, "conditioner_", None)
    if fitted is not None:
        candidates.insert(0, fitted)
    transform = None
    for candidate in candidates:
        transform = getattr(candidate, "transform_", None)
        if transform is not None:
            break
    if transform is None:
        raise ValueError(
            "fitted conditioner must expose conditioner_.transform_ or transform_."
        )
    values = np.asarray(transform, dtype=np.float64)
    if values.ndim == 1 and values.shape[0] == n_features:
        scales = np.abs(values)
    elif values.ndim == 2 and values.shape == (n_features, n_features):
        # For a full transform, the norm of each output column is the
        # coordinate-wise scale exposed to a row vector x @ T.  It reduces to
        # abs(diag(T)) for diagonal transforms.
        scales = np.linalg.norm(values, axis=0)
    else:
        raise ValueError(
            "fitted conditioner transform has an unexpected feature shape: "
            f"{values.shape}; expected {(n_features, n_features)}."
        )
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("fitted conditioner scales must be finite and positive.")
    return np.asarray(scales, dtype=np.float64)


def fold_scale_stability(
    X: Any,
    labels: Any,
    conditioner_factory: Any,
    *,
    seed: int,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Fit fold-local conditioners and compare all ten scale-vector pairs.

    Integer encoding is used only for ``StratifiedKFold.split``.  Each fresh
    conditioner receives the original scalar label values, including
    heterogeneous object labels, unchanged in meaning and order.
    """

    values = _validate_dense_matrix(X, "X")
    labels_array = _validate_labels(labels, values.shape[0])
    folds_requested = _validate_positive_integer(n_splits, "n_splits")
    if folds_requested < 2:
        raise ValueError("n_splits must be at least two.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    encoded = _first_observed_integer_encoding(labels_array)
    splitter = StratifiedKFold(
        n_splits=folds_requested,
        shuffle=True,
        random_state=int(seed),
    )
    try:
        split_rows = list(splitter.split(np.zeros(values.shape[0]), encoded))
    except ValueError as exc:
        raise ValueError("labels cannot support the requested stratified folds.") from exc

    fold_scales: list[np.ndarray] = []
    fold_sizes: list[dict[str, int]] = []
    for train_rows, holdout_rows in split_rows:
        estimator = _new_conditioner(conditioner_factory)
        if not hasattr(estimator, "fit"):
            raise TypeError("conditioner_factory must produce an object with fit(X, y).")
        # This is intentionally the original object-label array, not encoded.
        fitted = estimator.fit(values[train_rows], labels_array[train_rows])
        if fitted is not None:
            estimator = fitted
        fold_scales.append(_conditioner_scale_vector(estimator, values.shape[1]))
        fold_sizes.append(
            {
                "train": int(np.asarray(train_rows).size),
                "heldout": int(np.asarray(holdout_rows).size),
            }
        )

    pair_correlations: list[Optional[float]] = []
    log_differences: list[float] = []
    for first in range(len(fold_scales)):
        for second in range(first + 1, len(fold_scales)):
            pair_correlations.append(_spearman(fold_scales[first], fold_scales[second]))
            log_differences.extend(
                np.abs(np.log(fold_scales[first] / fold_scales[second])).tolist()
            )
    finite_correlations = [value for value in pair_correlations if value is not None]
    return json_safe(
        {
            "fold_count": int(len(fold_scales)),
            "fold_pair_count": int(len(pair_correlations)),
            "fold_sizes": fold_sizes,
            "fold_scale_vectors": [scale.tolist() for scale in fold_scales],
            "fold_scale_spearman": (
                float(np.mean(finite_correlations)) if finite_correlations else None
            ),
            "fold_scale_spearman_pairs": pair_correlations,
            "median_log_scale_difference": (
                float(np.median(log_differences)) if log_differences else None
            ),
            "max_log_scale_difference": (
                float(np.max(log_differences)) if log_differences else None
            ),
            "label_encoding_for_split": encoded.tolist(),
        }
    )


def _label_equal(first: Any, second: Any) -> bool:
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _owner_vector(owner_labels: Any, n_prototypes: int) -> list[Any] | None:
    if owner_labels is None:
        return None
    if isinstance(owner_labels, Mapping):
        result = [None] * n_prototypes
        for key, label in owner_labels.items():
            if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
                raise ValueError("prototype owner ids must be integers.")
            prototype_id = int(key)
            if prototype_id < 0 or prototype_id >= n_prototypes:
                raise ValueError("prototype owner id is outside n_prototypes.")
            result[prototype_id] = label
        for owner in result:
            if owner is not None:
                _validate_scalar_label(owner, "owner_labels")
        return result
    values = list(owner_labels)
    if len(values) != n_prototypes:
        raise ValueError("owner_labels must have one entry per prototype.")
    for owner in values:
        if owner is not None:
            _validate_scalar_label(owner, "owner_labels")
    return values


def prototype_assignment_diagnostics(
    prototype_ids: Any,
    labels: Any,
    owner_labels: Any = None,
    *,
    n_prototypes: int | None = None,
    class_labels: Any = None,
) -> dict[str, Any]:
    """Summarize held-out label composition of assigned prototypes.

    Entropy is normalized by ``log(number_of_classes)`` and is reported only
    over occupied prototypes.  Empty-prototype rate uses all fitted prototype
    ids as its denominator.  ``owner_labels`` is optional; without backend
    ownership, owner agreement is explicitly ``None`` rather than inferred.
    """

    assignments_raw = np.asarray(prototype_ids)
    if assignments_raw.ndim != 1:
        raise ValueError("prototype_ids must be one-dimensional.")
    if assignments_raw.dtype.kind not in "iu":
        raise ValueError("prototype_ids must contain integer ids.")
    assignments = np.asarray(assignments_raw, dtype=np.int64).copy()
    if np.any(assignments < 0):
        raise ValueError("prototype_ids must be non-negative.")
    labels_array = _validate_labels(labels, assignments.shape[0])
    observed_classes = _ordered_unique(labels_array.tolist())
    if class_labels is None:
        all_classes = observed_classes
    else:
        class_array = _validate_labels(class_labels, len(np.asarray(class_labels).reshape(-1)))
        all_classes = _ordered_unique(class_array.tolist())
        for label in observed_classes:
            if label not in all_classes:
                all_classes.append(label)
    inferred_count = int(assignments.max()) + 1 if assignments.size else 0
    if n_prototypes is None:
        prototype_count = inferred_count
    else:
        if (
            isinstance(n_prototypes, bool)
            or int(n_prototypes) != n_prototypes
            or int(n_prototypes) < inferred_count
            or int(n_prototypes) < 0
        ):
            raise ValueError("n_prototypes must cover every assigned prototype id.")
        prototype_count = int(n_prototypes)
    owners = _owner_vector(owner_labels, prototype_count)
    if owners is not None:
        for owner in owners:
            if owner is not None and owner not in all_classes:
                all_classes.append(owner)
    occupied_ids = sorted(set(assignments.tolist()))
    entropy_records: list[dict[str, Any]] = []
    support_total = int(assignments.size)
    weighted_entropy_sum = 0.0
    entropy_values: list[float] = []
    mixed_count = 0
    majority_count = 0
    owner_matches = 0
    owner_support = 0
    for prototype_id in occupied_ids:
        rows = np.flatnonzero(assignments == prototype_id)
        row_labels = labels_array[rows].tolist()
        counts: dict[Any, int] = {}
        for label in row_labels:
            counts[label] = counts.get(label, 0) + 1
        probabilities = np.asarray(list(counts.values()), dtype=np.float64) / float(rows.size)
        entropy = (
            float(-np.sum(probabilities * np.log(probabilities)))
            if probabilities.size
            else 0.0
        )
        denominator = math.log(len(all_classes)) if len(all_classes) > 1 else 1.0
        normalized_entropy = float(entropy / denominator) if denominator > 0.0 else 0.0
        mixed = len(counts) > 1
        if mixed:
            mixed_count += 1
        entropy_values.append(normalized_entropy)
        weighted_entropy_sum += float(rows.size) * normalized_entropy
        majority_count += max(counts.values()) if counts else 0
        if owners is not None and owners[prototype_id] is not None:
            owner_support += int(rows.size)
            owner_matches += sum(
                _label_equal(label, owners[prototype_id]) for label in row_labels
            )
        entropy_records.append(
            {
                "prototype_id": int(prototype_id),
                "support": int(rows.size),
                "normalized_label_entropy": normalized_entropy,
                "mixed": bool(mixed),
            }
        )

    under_records: list[dict[str, Any]] = []
    owner_counts: dict[Any, int] = {}
    if owners is not None:
        for owner in owners:
            if owner is not None:
                owner_counts[owner] = owner_counts.get(owner, 0) + 1
    if owners is not None:
        for label in all_classes:
            count = int(owner_counts.get(label, 0))
            under_records.append(
                {
                    "label": json_safe(label),
                    "prototype_count": count,
                    "under_prototyped": bool(count < 2),
                }
            )
    return json_safe(
        {
            "n_rows": int(assignments.size),
            "n_prototypes": int(prototype_count),
            "occupied_prototype_count": int(len(occupied_ids)),
            "empty_prototype_count": int(max(0, prototype_count - len(occupied_ids))),
            "occupied_normalized_label_entropy_macro": (
                float(np.mean(entropy_values)) if entropy_values else None
            ),
            "occupied_normalized_label_entropy_weighted": (
                float(weighted_entropy_sum / support_total) if support_total else None
            ),
            "mixed_rate_occupied": (
                float(mixed_count / len(occupied_ids)) if occupied_ids else None
            ),
            "empty_rate_all": (
                float((prototype_count - len(occupied_ids)) / prototype_count)
                if prototype_count
                else None
            ),
            "weighted_majority_assignment_purity": (
                float(majority_count / support_total) if support_total else None
            ),
            "owner_label_agreement": (
                float(owner_matches / owner_support) if owner_support else None
            ),
            "prototype_entropy": entropy_records,
            "under_prototyped": (
                bool(any(record["under_prototyped"] for record in under_records))
                if owners is not None
                else None
            ),
            "under_prototyped_labels": under_records,
        }
    )


def assignment_diagnostics_from_selector(
    selector: Any,
    X: Any,
    labels: Any,
    owner_labels: Any = None,
    *,
    n_prototypes: int | None = None,
    class_labels: Any = None,
) -> dict[str, Any]:
    """Obtain assignments through a fitted selector's public ``predict`` only."""

    predict = getattr(selector, "predict", None)
    if not callable(predict):
        raise TypeError("selector must expose fitted predict(X).")
    values = _validate_dense_matrix(X, "X")
    assignments = predict(values)
    return prototype_assignment_diagnostics(
        assignments,
        labels,
        owner_labels,
        n_prototypes=n_prototypes,
        class_labels=class_labels,
    )


def refinement_stage_diagnostics(
    refinement_summary: Mapping[str, Any] | None,
    *,
    unrefined_score: float | None = None,
    refined_score: float | None = None,
) -> dict[str, Any]:
    """Normalize refinement counts against all pre-refinement prototypes."""

    summary = dict(refinement_summary or {})
    before_raw = summary.get("prototype_count_before", summary.get("prototype_count", 0))
    try:
        before = int(before_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("prototype_count_before must be an integer.") from exc
    if before < 0:
        raise ValueError("prototype_count_before must be non-negative.")
    after = summary.get("prototype_count_after")
    after_count = int(after) if after is not None else None
    eligible = int(summary.get("eligible_count", summary.get("attempted_count", 0)))
    applied = int(summary.get("applied_count", summary.get("applied", 0)))
    skipped = int(summary.get("skipped_count", summary.get("skipped", 0)))
    if min(eligible, applied, skipped) < 0:
        raise ValueError("refinement counts must be non-negative.")
    changed = summary.get("changed_parent_ids", summary.get("applied_parent_ids", ()))
    if changed is None:
        changed = ()
    changed_ids = sorted({int(value) for value in changed})
    movement = None
    if unrefined_score is not None and refined_score is not None:
        unrefined = float(unrefined_score)
        refined = float(refined_score)
        if np.isfinite(unrefined) and np.isfinite(refined):
            movement = refined - unrefined
    def rate(value: int) -> Optional[float]:
        return float(value / before) if before else None
    return json_safe(
        {
            "prototype_count_before": before,
            "prototype_count_after": after_count,
            "eligible_count": eligible,
            "applied_count": applied,
            "skipped_count": skipped,
            "eligible_rate_all_pre_refinement_prototypes": rate(eligible),
            "applied_rate_all_pre_refinement_prototypes": rate(applied),
            "skipped_rate_all_pre_refinement_prototypes": rate(skipped),
            "refinement_activity_all_pre_refinement_prototypes": rate(applied),
            "changed_parent_ids": changed_ids,
            "paired_total_refined_vs_unrefined_score_movement": movement,
        }
    )


def _pair_value(mapping: Any, pair: tuple[Any, Any], default: Any = None) -> Any:
    if mapping is None:
        return default
    # Prefer ``get`` so a defaultdict (as used by OI) is not mutated by a
    # diagnostic lookup.  Lazy pairwise mappings also implement ``get``
    # without materializing a missing key.
    try:
        return mapping.get(pair, default)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        return mapping[pair]
    except (KeyError, TypeError, ValueError, IndexError):
        return default


def _unwrap_overlap_index(model: Any) -> Any:
    nested = getattr(model, "estimator_", None)
    return nested if nested is not None else model


def directional_pair_evidence(
    model: Any,
    labels: Any,
    pairs: Any = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Trace exact directional support/hits/evidence from fitted OI state.

    The function reads ``pairwise_cardinality``, ``_pairwise_hits``,
    ``sparse_adj``, and ``pairwise_index`` without fitting or replaying rows.
    With ``strict=True`` every stored representation must agree with the
    arithmetic implied by support and hits; mismatches raise immediately.
    """

    inner = _unwrap_overlap_index(model)
    score_classes = getattr(inner, "_score_classes", ())
    if score_classes:
        classes = list(score_classes)
    else:
        rev_map = getattr(inner, "rev_map", {})
        if rev_map:
            classes = list(rev_map.keys())
        else:
            label_count = len(np.asarray(labels).reshape(-1))
            classes = _ordered_unique(_validate_labels(labels, label_count).tolist())
    if pairs is None:
        pair_list = [
            (source, target)
            for source in classes
            for target in classes
            if source != target
        ]
    else:
        try:
            pair_list = [tuple(pair) for pair in pairs]
        except (TypeError, ValueError) as exc:
            raise ValueError("pairs must contain (source_label, target_label) pairs.") from exc
        for pair in pair_list:
            if len(pair) != 2 or _label_equal(pair[0], pair[1]):
                raise ValueError("directional pairs must be off-diagonal label pairs.")
    for source, target in pair_list:
        if not any(_label_equal(source, known) for known in classes) or not any(
            _label_equal(target, known) for known in classes
        ):
            raise ValueError(
                "directional pair labels must belong to the fitted OI class set; "
                f"got {(source, target)!r}."
            )

    cardinality_mapping = getattr(inner, "pairwise_cardinality", None)
    hits_mapping = getattr(inner, "_pairwise_hits", None)
    sparse_mapping = getattr(inner, "sparse_adj", None)
    index_mapping = getattr(inner, "pairwise_index", None)
    if (
        cardinality_mapping is None
        or hits_mapping is None
        or sparse_mapping is None
        or index_mapping is None
    ):
        raise ValueError("fitted OI state lacks the required pairwise diagnostics.")
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for source, target in pair_list:
        pair = (source, target)
        support_raw = _pair_value(cardinality_mapping, pair, None)
        if support_raw is None:
            cluster_cardinality = getattr(inner, "cluster_cardinality", {})
            support_raw = _pair_value(cluster_cardinality, pair, None)
            if support_raw is None:
                try:
                    support_raw = cluster_cardinality[source]
                except (KeyError, TypeError, ValueError):
                    support_raw = 0
        hits_raw = _pair_value(hits_mapping, pair, 0)
        sparse_hits_raw = _pair_value(sparse_mapping, pair, 0)
        try:
            support = int(support_raw)
            hits = int(hits_raw)
            sparse_hits = int(sparse_hits_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"non-integer pair state for {pair!r}.") from exc
        if support < 0 or hits < 0 or sparse_hits < 0:
            raise ValueError(f"negative pair state for {pair!r}.")
        pairwise_index_raw = _pair_value(index_mapping, pair, None)
        try:
            pairwise_index = float(pairwise_index_raw)
        except (TypeError, ValueError):
            pairwise_index = math.nan
        evidence = 1.0 - pairwise_index if np.isfinite(pairwise_index) else None
        expected_index = 1.0 - (hits / support) if support > 0 else 1.0
        expected_evidence = hits / support if support > 0 else 0.0
        exact = (
            hits == sparse_hits
            and np.isfinite(pairwise_index)
            and bool(np.isclose(pairwise_index, expected_index, rtol=0.0, atol=1e-12))
            and bool(
                evidence is not None
                and np.isclose(
                    evidence,
                    expected_evidence,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        )
        row = {
            "source_label": json_safe(source),
            "target_label": json_safe(target),
            "support": support,
            "hits": hits,
            "sparse_adj_hits": sparse_hits,
            "pairwise_index": pairwise_index if np.isfinite(pairwise_index) else None,
            "evidence": evidence,
            "exact_state_match": bool(exact),
        }
        rows.append(row)
        if not exact:
            mismatches.append(row)
    result = {
        "directional_pair_support_hits_evidence": rows,
        "pair_count": int(len(rows)),
        "exact_state_match": not mismatches,
        "mismatches": mismatches,
        "under_prototyped_labels": json_safe(
            list(getattr(inner, "under_prototyped_labels_", ()))
        ),
    }
    if strict and mismatches:
        raise AssertionError(
            "OI directional pair state mismatch: " + canonical_json(mismatches)
        )
    return json_safe(result)


def _array_digest(value: Any) -> tuple[str, list[int], str] | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 2:
        return None
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest(), [int(item) for item in contiguous.shape], str(contiguous.dtype)


def snapshot_fitted_state(model: Any) -> dict[str, Any]:
    """Capture static fitted state for no-refit/invariance checks.

    Scoring bookkeeping is deliberately excluded: upstream ``score_fixed``
    recomputes its hold-out pair counters by design.  Prototype centers,
    conditioning identity, structural diagnostics, and refinement decisions
    are the state that a read-only diagnostic or ``predict`` must preserve.
    Runtime measurements are also excluded because they are not structural.
    """

    inner = _unwrap_overlap_index(model)
    backend = getattr(inner, "_model", None)
    centers = getattr(backend, "centers", None) if backend is not None else None
    if centers is None and backend is not None:
        centers = getattr(backend, "_centers", None)
    centers_digest = _array_digest(centers)
    try:
        conditioning = getattr(model, "conditioning_diagnostics_")
    except (AttributeError, ValueError):
        conditioning = getattr(inner, "conditioning_diagnostics_", {})
    try:
        refinement = getattr(model, "prototype_refinement_")
    except (AttributeError, ValueError):
        refinement = getattr(inner, "prototype_refinement_", {})
    payload = {
        "centers_sha256": centers_digest[0] if centers_digest else None,
        "centers_shape": centers_digest[1] if centers_digest else None,
        "centers_dtype": centers_digest[2] if centers_digest else None,
        "prototype_count": int(centers_digest[1][0]) if centers_digest else None,
        "conditioning": json_safe(conditioning),
        "refinement": json_safe(refinement),
    }
    state_digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    payload["state_sha256"] = state_digest
    return json_safe(payload)


def assert_state_unchanged(before: Any, after: Any) -> None:
    """Assert two snapshots (or two fitted objects) are structurally equal."""

    first = (
        before
        if isinstance(before, Mapping) and "state_sha256" in before
        else snapshot_fitted_state(before)
    )
    second = (
        after
        if isinstance(after, Mapping) and "state_sha256" in after
        else snapshot_fitted_state(after)
    )
    if canonical_json(first) != canonical_json(second):
        raise AssertionError(
            "fitted state changed unexpectedly: "
            f"before={canonical_json(first)}, after={canonical_json(second)}"
        )


def call_and_assert_state_unchanged(
    model: Any,
    operation: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a read-only operation and fail if fitted structural state changes."""

    before = snapshot_fitted_state(model)
    result = operation(*args, **kwargs)
    after = snapshot_fitted_state(model)
    assert_state_unchanged(before, after)
    return result


__all__ = [
    "MetricSpaceView",
    "assignment_diagnostics_from_selector",
    "assert_state_unchanged",
    "call_and_assert_state_unchanged",
    "canonical_json",
    "directional_pair_evidence",
    "fold_scale_stability",
    "json_safe",
    "metric_space_view",
    "neighbor_diagnostics",
    "pair_distance_diagnostics",
    "prototype_assignment_diagnostics",
    "refinement_stage_diagnostics",
    "snapshot_fitted_state",
    "top_k_neighbors",
]
