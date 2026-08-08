import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Set as AbstractSet
from typing import Literal, Optional, Union, Dict, Any, Tuple

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances

try:
    from sklearn.base import BaseEstimator
except ImportError:  # pragma: no cover - sklearn is a required dependency for offline backends
    class BaseEstimator:  # type: ignore[no-redef]
        """Fallback base class when sklearn is unavailable at import time."""
        pass

from overlapindex.utils import (
    _group_indices_by_label,
    _ordered_unique_1d,
    _validate_feature_matrix,
    _validate_positive_integer,
    complement_code,
    top_two_indices_against_others_from_backend,
)
from overlapindex.clustering import (
    _BaseManyToOneClusteringModel,
    _ARTMAPManyToOne,
    _KMeansManyToOne,
    _MiniBatchKMeansManyToOne,
    _BallCoverManyToOne,
)
from overlapindex._universal_scorer import (
    compute_second_best_source_scores,
    iter_target_class_blocks,
)


def _default_one() -> float:
    """Return the picklable default score used by overlap mappings."""
    return 1.0


def _validate_label(label: Any) -> None:
    """Require one hashable, non-missing label value."""
    try:
        hash(label)
    except TypeError as exc:
        raise TypeError("Labels must be hashable.") from exc

    if label is None:
        raise ValueError("Labels must not contain None or NaN values.")
    try:
        unequal_to_self = label != label
    except Exception:
        unequal_to_self = False
    try:
        is_missing = bool(unequal_to_self)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Labels must be scalar, non-missing values with unambiguous equality."
        ) from exc
    if is_missing:
        raise ValueError("Labels must not contain None or NaN values.")


def _is_multilabel_entry(y: Any) -> bool:
    """Return True when one target entry is a collection of labels."""
    if isinstance(y, (str, bytes)):
        return False
    return isinstance(y, Iterable)


def _label_sets_from_indicator_matrix(Y: Any) -> list[tuple]:
    """Convert a dense or sparse binary indicator matrix to label tuples."""
    if sparse.issparse(Y):
        Y_csr = Y.tocsr(copy=True)
        Y_csr.eliminate_zeros()
        Y_csr.sort_indices()
        if not np.all(Y_csr.data == 1):
            raise ValueError("2D Y must contain only 0/1 indicators.")
        return [
            tuple(Y_csr.indices[Y_csr.indptr[i] : Y_csr.indptr[i + 1]])
            for i in range(Y_csr.shape[0])
        ]

    try:
        Y_num = np.asarray(Y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("2D Y must be a numeric binary indicator matrix.") from exc

    if Y_num.ndim != 2:
        raise ValueError(
            "Y must be a 1D label vector, a 1D sequence of label collections, "
            "or a 2D binary indicator matrix."
        )
    if not np.all(np.isfinite(Y_num)):
        raise ValueError("2D Y contains NaN or infinite values.")
    if not np.all((Y_num == 0) | (Y_num == 1)):
        raise ValueError("2D Y must contain only 0/1 indicators.")

    return [tuple(np.flatnonzero(row)) for row in Y_num]


def _stable_collection_labels(labels: Iterable[Any]) -> tuple:
    """Validate and deduplicate one label collection without losing list order."""
    if isinstance(labels, (set, frozenset)):
        labels = sorted(
            labels,
            key=lambda label: (
                type(label).__module__,
                type(label).__qualname__,
                repr(label),
            ),
        )

    result = []
    seen = set()
    for label in labels:
        _validate_label(label)
        if label not in seen:
            seen.add(label)
            result.append(label)
    return tuple(result)


def _normalize_label_sets(Y: Any) -> list[tuple]:
    """
    Convert supported label formats to per-sample label sets.

    Supported formats are a 1D single-label vector, a 1D sequence of label
    collections, and an explicit dense or sparse 2D binary indicator matrix.
    """
    if sparse.issparse(Y):
        if Y.ndim != 2:
            raise ValueError(
                "Y must be a 1D label vector, a 1D sequence of label collections, "
                "or a 2D binary indicator matrix."
            )
        label_sets = _label_sets_from_indicator_matrix(Y)
    elif isinstance(Y, np.ndarray) and Y.ndim == 2:
        label_sets = _label_sets_from_indicator_matrix(Y)
    else:
        if isinstance(Y, (str, bytes)):
            raise ValueError(
                "Y must be a 1D label vector, a 1D sequence of label collections, "
                "or a 2D binary indicator matrix."
            )
        Y_arr = np.asarray(Y, dtype=object)
        if Y_arr.ndim > 1 and isinstance(Y, np.ndarray):
            raise ValueError(
                "Y must be a 1D label vector, a 1D sequence of label collections, "
                "or a 2D binary indicator matrix."
            )

        if Y_arr.ndim == 2 and not isinstance(Y, np.ndarray):
            try:
                numeric = np.asarray(Y, dtype=float)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None and np.all((numeric == 0) | (numeric == 1)):
                warnings.warn(
                    "A rectangular binary Python sequence is interpreted as a "
                    "sequence of label collections. Convert it to a NumPy array "
                    "or SciPy sparse matrix for indicator-matrix semantics.",
                    UserWarning,
                    stacklevel=2,
                )

        try:
            entries = list(Y)
        except TypeError as exc:
            raise ValueError(
                "Y must be a 1D label vector, a 1D sequence of label collections, "
                "or a 2D binary indicator matrix."
            ) from exc
        label_sets = []
        for y in entries:
            try:
                labels = (
                    _stable_collection_labels(y)
                    if _is_multilabel_entry(y)
                    else _stable_collection_labels((y,))
                )
            except TypeError as exc:
                raise TypeError("Labels must be hashable.") from exc
            label_sets.append(labels)

    if any(len(labels) == 0 for labels in label_sets):
        raise ValueError("Every sample must have at least one label.")

    return label_sets


def _flatten_single_label_sets(Y_sets: list[tuple]) -> np.ndarray:
    """Convert normalized singleton label sets back to a 1D label vector."""
    return np.asarray([labels[0] for labels in Y_sets], dtype=object)


def _expand_multilabel_for_backend(
    X: np.ndarray,
    Y_sets: list[Iterable[Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Duplicate each sample once for every positive label before backend fit."""
    row_indices = []
    Y_expanded = []

    for i, labels in enumerate(Y_sets):
        for label in _stable_collection_labels(labels):
            row_indices.append(i)
            Y_expanded.append(label)

    X_expanded = X[np.asarray(row_indices, dtype=int)]
    if not sparse.issparse(X_expanded):
        X_expanded = np.asarray(X_expanded, dtype=float)
    return X_expanded, np.asarray(Y_expanded, dtype=object)


def _ordered_unique_labels(Y_sets: list[Iterable[Any]]) -> np.ndarray:
    """Return labels in first-observed order across normalized label sets."""
    labels = []
    seen = set()
    for sample_labels in Y_sets:
        for label in _stable_collection_labels(sample_labels):
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return np.asarray(labels, dtype=object)


def _deduplicated_labels(labels: Iterable[Any]) -> list[Any]:
    """Return labels with first-seen order preserved after hashability checks."""
    deduplicated = []
    seen = set()
    for label in labels:
        try:
            _validate_label(label)
        except TypeError as exc:
            raise TypeError("exclude_classes entries must be hashable.") from exc
        if label not in seen:
            seen.add(label)
            deduplicated.append(label)
    return deduplicated


class _LazyPairwiseMapping(dict):
    """Sparse pairwise diagnostics with resolver-backed direct lookups.

    Only explicitly materialized, non-default values are stored in the dict
    payload.  ``mapping[(a, b)]`` still resolves the logical default (or the
    exact directional denominator) through the owning estimator without
    inserting a key, keeping iteration sparse for high-cardinality targets.
    """

    def __init__(self, owner: Any, kind: str) -> None:
        super().__init__()
        self._owner = owner
        self._kind = str(kind)

    def __missing__(self, key: Any) -> Any:
        return self._owner._resolve_lazy_pairwise_value(self._kind, key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError, ValueError):
            return default


class _LazyAllCompetitors(Mapping):
    """Lazy view of all competing labels for multi-label ``all`` mode."""

    def __init__(self, classes: Iterable[Any]) -> None:
        self._classes = tuple(classes)
        self._class_set = set(self._classes)

    def __getitem__(self, source: Any) -> np.ndarray:
        if source not in self._class_set:
            raise KeyError(source)
        return np.asarray(
            [label for label in self._classes if label != source],
            dtype=object,
        )

    def __iter__(self):
        return iter(self._classes)

    def __len__(self) -> int:
        return len(self._classes)


class _LazyUnevaluablePairs(AbstractSet):
    """Set-like view of selected multi-label pairs with zero denominator.

    Multi-label scoring can have a quadratic number of directional pairs whose
    source-positive rows always contain the target label.  The diagnostics
    need to expose those pairs exactly, but storing them during ``fit`` would
    recreate the quadratic state that the scorer avoids.  This view retains
    the observed class order and, for ``top_m`` mode, the selected competitor
    order; iteration and membership resolve each candidate's denominator on
    demand through the fitted estimator.
    """

    def __init__(
        self,
        owner: Any,
        classes: Iterable[Any],
    ) -> None:
        self._owner = owner
        self._classes = tuple(classes)
        self._class_set = set(self._classes)
        self._top_m = owner.multilabel_pair_mode == "top_m"

        # ``all`` mode deliberately stores no target list per source: deriving
        # all competitors from the O(C) class tuple is what keeps this view
        # non-materializing for high-cardinality targets.  ``top_m`` mode is
        # already bounded by the configured number of selected competitors.
        if self._top_m:
            selected = []
            competitors = owner.competitors_
            for source in self._classes:
                targets = competitors.get(source, ())
                if isinstance(targets, np.ndarray):
                    targets = targets.tolist()
                selected.append((source, tuple(targets)))
            self._selected_by_source = tuple(selected)
            self._selected_lookup = {
                source: targets for source, targets in self._selected_by_source
            }
        else:
            self._selected_by_source = None
            self._selected_lookup = None

    def _targets_for(self, source: Any) -> Iterable[Any]:
        if self._selected_by_source is None:
            return (target for target in self._classes if target != source)
        return self._selected_lookup.get(source, ())

    def _selected_pair(self, source: Any, target: Any) -> bool:
        try:
            source_observed = source in self._class_set
        except TypeError:
            source_observed = False
        if not source_observed or source == target:
            return False
        return any(target == candidate for candidate in self._targets_for(source))

    def _is_zero_denominator(self, source: Any, target: Any) -> bool:
        if not self._selected_pair(source, target):
            return False
        try:
            return self._owner._resolve_lazy_pairwise_cardinality(
                (source, target)
            ) <= 0
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _unpack_pair(pair: Any) -> Optional[tuple[Any, Any]]:
        if isinstance(pair, (str, bytes)):
            return None
        try:
            source, target = pair
        except (TypeError, ValueError):
            return None
        return source, target

    def __contains__(self, pair: Any) -> bool:
        unpacked = self._unpack_pair(pair)
        if unpacked is None:
            return False
        source, target = unpacked
        try:
            return self._is_zero_denominator(source, target)
        except (TypeError, ValueError):
            return False

    def __iter__(self):
        for source in self._classes:
            for target in self._targets_for(source):
                if self._is_zero_denominator(source, target):
                    yield source, target

    def __len__(self) -> int:
        # Counting is intentionally a lazy scan: it reports the exact size
        # without retaining the yielded C² candidate pairs.
        return sum(1 for _ in self)

    def __eq__(self, other: Any) -> bool:
        """Compare like a set without eagerly materializing this view."""
        if isinstance(other, (set, frozenset, AbstractSet, _LazyUnevaluablePairs)):
            try:
                return len(self) == len(other) and all(pair in other for pair in self)
            except (TypeError, ValueError):
                return False

        if isinstance(other, (tuple, list)):
            # Historical fits exposed a tuple.  Keep deterministic sequence
            # equality useful for small-data callers while still streaming the
            # lazy side rather than constructing a second collection.
            if len(self) != len(other):
                return False
            return all(
                pair == expected
                for pair, expected in zip(self, other)
            )
        return NotImplemented

# ----------------------------
# OverlapIndex with model_type
# ----------------------------

class OverlapIndex(BaseEstimator):
    """
    Compute an overlap index over class-owned clustering prototypes.

    The class supports centroid-style offline backends by default, along with
    ARTMAP-style online backends when explicitly selected. All samples are
    preprocessed before being passed to the backend.
    The index is updated by comparing each sample's best matching unit against
    competing class-owned clusters.
    """
    def __init__(
        self,
        rho: float = 0.9,
        r_hat: float = np.inf,
        model_type: Literal["Fuzzy", "Hypersphere", "KMeans", "MiniBatchKMeans", "BallCover"] = "MiniBatchKMeans",
        match_tracking: str = "MT+",
        # centroid backend options:
        kmeans_k: Union[int, Dict[Any, int]] = 8,
        kmeans_kwargs: Optional[dict] = None,
        # ball-cover backend options:
        ballcover_k: Union[int, Dict[Any, int], Literal["auto"]] = "auto",
        ballcover_radius: Union[float, Dict[Any, float], Literal["auto"]] = 0.25,
        ballcover_kwargs: Optional[dict] = None,
        offline_chunk_size: Optional[int] = 10_000,
        multilabel_pair_mode: Literal["all", "top_m"] = "all",
        top_m: Optional[int] = None,
        exclude_classes: Optional[Any] = None,
        offline_memory_budget_mb: int = 256,
    ) -> None:
        """
        Initialize the overlap index and its clustering backend.

        Parameters
        ----------
        rho : float, default=0.9
            ARTMAP vigilance parameter used by Fuzzy and Hypersphere backends.
        r_hat : float, default=np.inf
            Hypersphere ARTMAP radius constraint.
        model_type : {"Fuzzy", "Hypersphere", "KMeans", "MiniBatchKMeans", "BallCover"}, default="MiniBatchKMeans"
            Backend family used to create class-owned clusters.
        match_tracking : str, default="MT+"
            Match-tracking mode forwarded to ARTMAP partial-fit calls.
        kmeans_k : int or dict, default=8
            Number of clusters per class for centroid backends. A dictionary may
            specify class-specific values.
        kmeans_kwargs : dict, optional
            Keyword arguments forwarded to the selected centroid backend.
        ballcover_k : int, dict, or "auto", default="auto"
            Number of balls per class, class-specific ball counts, or "auto" to
            greedily add fixed-radius balls until the requested cover fraction is
            reached.
        ballcover_radius : float, dict, or "auto", default=0.25
            Ball radius, class-specific radii, or "auto" to infer the radius
            after selecting a fixed number of balls. Only one of ballcover_k and
            ballcover_radius may be "auto".
        ballcover_kwargs : dict, optional
            Additional keyword arguments forwarded to the BallCover backend, such
            as metric, cover_fraction, chunk_size, max_balls, or random_state.
        offline_chunk_size : int or None, default=10000
            Number of samples per chunk for optimized offline centroid scoring.
            If None, each class block is scored at once.
        multilabel_pair_mode : {"all", "top_m"}, default="all"
            Competitor-pair selection mode used only for multi-label offline
            scoring.
        top_m : int, optional
            Number of nearest competing labels per source label when
            multilabel_pair_mode is "top_m".
        exclude_classes : None, scalar label, or iterable of labels, optional
            Label ids to exclude from the global ``index`` and
            ``weighted_index`` aggregation. Excluded labels remain fully
            involved in fitting, singleton scoring, pairwise scoring, and all
            bookkeeping outputs.
        offline_memory_budget_mb : int, default=256
            Approximate scratch-memory budget, in mebibytes, used by the
            backend-neutral offline scorer.  The scorer tiles rows and
            prototypes so score blocks stay within this budget.  This
            parameter is appended after the historical positional arguments
            to preserve their calling convention.
        """
        self.rho = rho
        self.r_hat = r_hat
        self.model_type = model_type
        self.match_tracking = match_tracking
        self.kmeans_k = kmeans_k
        self.kmeans_kwargs = kmeans_kwargs
        self.ballcover_k = ballcover_k
        self.ballcover_radius = ballcover_radius
        self.ballcover_kwargs = ballcover_kwargs
        self.offline_chunk_size = offline_chunk_size
        self.offline_memory_budget_mb = offline_memory_budget_mb
        self.multilabel_pair_mode = multilabel_pair_mode
        self.top_m = top_m
        self.exclude_classes = exclude_classes
        self._validate_multilabel_params()

        # indices / bookkeeping
        self.sparse_adj = defaultdict(int)
        self.cluster_cardinality = defaultdict(int)
        self.rev_map = defaultdict(set)
        self._pairwise_hits = defaultdict(int)
        self._pairwise_multilabel = False
        self.pairwise_index = _LazyPairwiseMapping(self, "index")
        self.singleton_index = defaultdict(_default_one)
        self.pairwise_cardinality = _LazyPairwiseMapping(self, "cardinality")
        self.competitors_ = {}
        self.under_prototyped_labels_ = ()
        self.unevaluable_pairs_ = ()
        self.unevaluable_labels_ = ()
        self.label_to_index_ = {}
        self.index_to_label_ = {}
        self._label_indicator_csr_ = None
        self._label_indicator_csc_ = None
        self._positive_rows_by_label_index_ = {}
        self._score_classes = ()
        self.index = 1.0

        self._model: _BaseManyToOneClusteringModel = self._build_model()

    def _validate_multilabel_params(self) -> None:
        """Validate multi-label pair-selection parameters."""
        if self.multilabel_pair_mode not in {"all", "top_m"}:
            raise ValueError("multilabel_pair_mode must be one of {'all', 'top_m'}.")
        if self.multilabel_pair_mode == "top_m":
            if self.top_m is None:
                raise ValueError(
                    "top_m must be a positive integer when multilabel_pair_mode='top_m'."
                )
            try:
                _validate_positive_integer(self.top_m, "top_m")
            except ValueError as exc:
                raise ValueError(
                    "top_m must be a positive integer when multilabel_pair_mode='top_m'."
                ) from exc
        if self.offline_chunk_size is not None:
            _validate_positive_integer(self.offline_chunk_size, "offline_chunk_size")
        _validate_positive_integer(
            self.offline_memory_budget_mb,
            "offline_memory_budget_mb",
        )

    def _build_model(self) -> _BaseManyToOneClusteringModel:
        """Construct the backend adapter from the current estimator parameters."""
        if self.model_type in ["Fuzzy", "Hypersphere"]:
            return _ARTMAPManyToOne(
                model_type=self.model_type,
                rho=self.rho,
                r_hat=self.r_hat,
            )
        if self.model_type == "KMeans":
            return _KMeansManyToOne(k=self.kmeans_k, kmeans_kwargs=self.kmeans_kwargs)
        if self.model_type == "MiniBatchKMeans":
            return _MiniBatchKMeansManyToOne(k=self.kmeans_k, kmeans_kwargs=self.kmeans_kwargs)
        if self.model_type == "BallCover":
            kwargs = self.ballcover_kwargs or {}
            return _BallCoverManyToOne(
                k=self.ballcover_k,
                radius=self.ballcover_radius,
                **kwargs,
            )
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def set_params(self, **params: Any) -> "OverlapIndex":
        """Update estimator parameters and rebuild the backend adapter."""
        super().set_params(**params)
        self._validate_multilabel_params()
        self._model = self._build_model()
        self._reset_indices()
        return self

    @property
    def _is_artmap_backend(self) -> bool:
        """Return True when the active backend is ARTMAP-style and online-capable."""
        return self.model_type in ["Fuzzy", "Hypersphere"]

    @property
    def _is_offline_backend(self) -> bool:
        """Return True when the active backend is restricted to offline fitting."""
        return not self._is_artmap_backend

    # ---- preprocessing ----

    def _prep_X(self, X: np.ndarray) -> np.ndarray:
        """Preprocess raw samples before clustering."""
        if self._is_artmap_backend:
            return complement_code(np.asarray(X, dtype=float))
        return X

    def _validate_sparse_backend(self, X: Any) -> None:
        """Reject sparse features for backends that require dense arrays."""
        if sparse.issparse(X) and self.model_type not in {"KMeans", "MiniBatchKMeans"}:
            raise TypeError(
                "Sparse X is supported only for model_type='KMeans' and "
                f"'MiniBatchKMeans'; got model_type={self.model_type!r}."
            )

    def _validate_input_data(
        self,
        X: np.ndarray,
        Y: Any,
    ) -> Tuple[np.ndarray, list[tuple]]:
        """Validate aligned batch inputs before preprocessing."""
        self._validate_sparse_backend(X)
        X_arr = _validate_feature_matrix(X)

        Y_sets = _normalize_label_sets(Y)
        if X_arr.shape[0] != len(Y_sets):
            raise ValueError(
                f"X and Y must have the same number of rows; got {X_arr.shape[0]} and {len(Y_sets)}."
            )
        return X_arr, Y_sets

    @staticmethod
    def _warn_empty_input() -> None:
        """Warn when an operation receives no samples."""
        warnings.warn(
            "Received empty X/Y; leaving OverlapIndex at its default value of 1.0.",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _warn_single_class() -> None:
        """Warn when an operation receives only one unique class."""
        warnings.warn(
            "Received data with a single class; OverlapIndex remains 1.0 until multiple classes are observed.",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _warn_all_observed_classes_excluded() -> None:
        """Warn when exclusions remove every observed class from summaries."""
        warnings.warn(
            "All observed classes were excluded from global aggregation; leaving OverlapIndex scores at 1.0.",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _warn_under_prototyped(labels: Tuple[Any, ...]) -> None:
        """Warn once when top-two scoring has fewer than two owned prototypes."""
        warnings.warn(
            "Top-two overlap scoring is degenerate for labels owning fewer than "
            f"two prototypes: {labels!r}. Scores are still computed.",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _warn_unevaluable_multilabel(labels: Tuple[Any, ...]) -> None:
        """Warn once when multi-label source labels lack comparison rows."""
        warnings.warn(
            "Some multi-label source labels have no evaluable selected competitor "
            f"pairs and were excluded from global aggregation: {labels!r}.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _check_feature_count(self, X: np.ndarray) -> None:
        """Validate raw feature count against a previously fitted estimator."""
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this OverlapIndex instance "
                f"was fit with {self.n_features_in_} features."
            )

    def _refresh_under_prototyped_labels(self) -> None:
        """Refresh deterministic diagnostics and warn only when they change."""
        previous = self.under_prototyped_labels_
        if len(self.rev_map) <= 1:
            current = ()
        else:
            current = tuple(
                label
                for label in self.rev_map
                if len(self.rev_map[label]) < 2
            )
        self.under_prototyped_labels_ = current
        if current and current != previous:
            self._warn_under_prototyped(current)

    def _reset_indices(self) -> None:
        """Reset overlap-index bookkeeping without replacing the clustering backend."""
        self.sparse_adj = defaultdict(int)
        self.cluster_cardinality = defaultdict(int)
        self.rev_map = defaultdict(set)
        self._pairwise_hits = defaultdict(int)
        self._pairwise_multilabel = False
        self.pairwise_index = _LazyPairwiseMapping(self, "index")
        self.singleton_index = defaultdict(_default_one)
        self.pairwise_cardinality = _LazyPairwiseMapping(self, "cardinality")
        self.competitors_ = {}
        self.under_prototyped_labels_ = ()
        self.unevaluable_pairs_ = ()
        self.unevaluable_labels_ = ()
        self.label_to_index_ = {}
        self.index_to_label_ = {}
        self._label_indicator_csr_ = None
        self._label_indicator_csc_ = None
        self._positive_rows_by_label_index_ = {}
        self._score_classes = ()
        self.index = 1.0
        if hasattr(self, "n_features_in_"):
            del self.n_features_in_

    # ---- compatibility accessors (optional) ----

    @property
    def weighted_index(self) -> float:
        """
        Return the support-weighted overlap index across observed classes.

        The default ``index`` is a macro average over per-class scores. This
        property weights each per-class score by that class's positive support.
        """
        included_labels = self._included_singleton_labels()
        if not included_labels:
            return float(self.index)

        total_support = sum(
            int(self.cluster_cardinality.get(y, 0))
            for y in included_labels
        )
        if total_support <= 0:
            return float(self.index)

        weighted_sum = sum(
            float(score) * int(self.cluster_cardinality.get(y, 0))
            for y, score in self.singleton_index.items()
            if y in included_labels
        )
        return float(weighted_sum / float(total_support))

    def _normalized_exclude_classes(self) -> set[Any]:
        """Normalize the configured global-aggregation exclusions."""
        excluded = self.exclude_classes
        if excluded is None:
            return set()

        if isinstance(excluded, (str, bytes)) or not isinstance(excluded, Iterable):
            labels = [excluded]
        else:
            labels = list(excluded)

        return set(_deduplicated_labels(labels))

    def _included_singleton_labels(self) -> list[Any]:
        """Return observed labels that still contribute to global summaries."""
        observed_labels = list(self.singleton_index.keys())
        if not observed_labels:
            return []

        excluded = self._normalized_exclude_classes()
        unevaluable = set(self.unevaluable_labels_)
        return [
            label
            for label in observed_labels
            if label not in excluded
            and label not in unevaluable
            and np.isfinite(self.singleton_index[label])
        ]

    def _recompute_global_index(self) -> float:
        """Recompute the exclusion-aware global macro overlap index."""
        included_labels = self._included_singleton_labels()
        if not included_labels:
            if self.singleton_index:
                self._warn_all_observed_classes_excluded()
            self.index = 1.0
            return self.index

        self.index = float(
            np.mean([self.singleton_index[label] for label in included_labels])
        )
        return self.index

    @property
    def module_a(self) -> Any:
        """Return the underlying ARTMAP module A object for ARTMAP backends."""
        if self._is_artmap_backend:
            return self._model.model.module_a
        raise AttributeError("module_a is only available for ARTMAP backends.")

    @property
    def map(self) -> Optional[Any]:
        """Return the underlying ARTMAP map object when available."""
        if self._is_artmap_backend:
            return self._model.model.map
        return None

    # ---- BMU helpers ----

    def get_top2_bmu(self, x: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """
        Return the first and second global BMU ids for one preprocessed sample.

        Parameters
        ----------
        x : np.ndarray
            A single sample that has already been transformed by _prep_X.

        Returns
        -------
        tuple of int or None
            The best and second-best global cluster ids. Missing entries are None.
        """
        ids, _ = self._model.topk(x, k=2)
        if ids.size == 0:
            return None, None
        b1 = int(ids[0])
        b2 = int(ids[1]) if ids.size > 1 else None
        return b1, b2

    def predict_subset_pairs(self, x: np.ndarray, y: Any) -> Dict[Any, Tuple[int, ...]]:
        """
        Return top-2 candidate cluster ids for comparisons between one class and all others.

        The candidate sets are taken from self.rev_map to preserve the historical
        replay semantics used by add_batch and fit_offline.
        """
        classes = list(self.rev_map.keys())
        return top_two_indices_against_others_from_backend(
            self._model,
            x,
            classes,
            self.rev_map,
            y,
        )

    def add_sample(self, x: np.ndarray, y: Any) -> float:
        """
        Incrementally add one labeled sample and update the overlap index.

        This method is available only for ARTMAP-style backends. Centroid
        backends are offline-only and should use fit_offline or add_batch.
        """
        if self._is_offline_backend:
            raise NotImplementedError(
                f"{self.model_type} backend is offline-only here. Use fit_offline(X, Y)."
            )
        _validate_label(y)
        x_ = np.asarray(x, dtype=float)

        if x_.ndim != 1:
            raise ValueError("x must be a 1D array or list")
        if not np.all(np.isfinite(x_)):
            raise ValueError("x contains NaN or infinite values.")
        if hasattr(self, "n_features_in_") and x_.shape[0] != self.n_features_in_:
            raise ValueError(
                f"x has {x_.shape[0]} features, but this OverlapIndex instance "
                f"was fit with {self.n_features_in_} features."
            )

        x_prep = self._prep_X(x_.reshape(1, -1))
        self._model.partial_fit(
            x_prep, [y], match_tracking=self.match_tracking
        )
        self.n_features_in_ = int(x_.shape[0])

        # ARTMAP path: latest assigned label is BMU1
        bmu1 = int(self._model.model.module_a.labels_[-1])

        # keep rev_map in sync with backend
        self.rev_map[y].add(bmu1)
        self._refresh_under_prototyped_labels()

        self.cluster_cardinality[y] += 1
        top2bmu = self.predict_subset_pairs(x_prep[0], y)

        if y not in self.singleton_index:
            self.singleton_index[y] = 1.0

        for b in self.rev_map.keys():
            bmu2 = int(bmu1)
            if b != y:
                if len(top2bmu[b]) > 1:
                    bmu2_, bmu3_ = top2bmu[b]
                    bmu2 = int(bmu3_ if bmu2_ == bmu1 else bmu2_)
                if bmu2 in self.rev_map[b]:
                    self.sparse_adj[(y, b)] += 1
                self.pairwise_index[(y, b)] = 1.0 - (
                    float(self.sparse_adj[(y, b)]) / float(self.cluster_cardinality[y])
                )

        if len(self.rev_map) > 1:
            self.singleton_index[y] = min(
                [self.pairwise_index[(y, b)] for b in self.rev_map.keys() if b != y]
            )
            self._recompute_global_index()
        else:
            if self._included_singleton_labels():
                self._warn_single_class()
            else:
                self._warn_all_observed_classes_excluded()
        return self.index

    def add_batch(self, X: np.ndarray, Y: Any) -> float:
        """
        Add a labeled batch and update the overlap index.

        ARTMAP backends perform a batch partial-fit followed by historical replay.
        Offline centroid backends delegate to fit_offline with reset_state=True.
        """
        if self._is_offline_backend:
            # For consistency with your original API, treat add_batch as offline-fit+score.
            return self.fit_offline(X, Y, reset_state=True)

        X, Y_sets = self._validate_input_data(X, Y)
        self._check_feature_count(X)
        if X.shape[0] == 0:
            self._warn_empty_input()
            return self.index

        if any(len(labels) > 1 for labels in Y_sets):
            raise NotImplementedError(
                "Multi-label add_batch is currently implemented only for offline backends."
            )

        Y_single = _flatten_single_label_sets(Y_sets)

        if _ordered_unique_1d(Y_single).size <= 1:
            self._warn_single_class()

        X_prep = self._prep_X(X)
        self._model.partial_fit(
            X_prep, Y_single, match_tracking=self.match_tracking
        )
        self.n_features_in_ = int(X.shape[1])

        BMU1 = self._model.model.module_a.labels_[-len(Y_single):]
        for x, y, bmu1 in zip(X_prep, Y_single, BMU1):
            bmu1 = int(bmu1)
            self.rev_map[y].add(bmu1)
            if y not in self.singleton_index:
                self.singleton_index[y] = 1.0

            self.cluster_cardinality[y] += 1
            top2bmu = self.predict_subset_pairs(x, y)

            for b in self.rev_map.keys():
                bmu2 = int(bmu1)
                if b != y:
                    if len(top2bmu[b]) > 1:
                        bmu2_, bmu3_ = top2bmu[b]
                        bmu2 = int(bmu3_ if bmu2_ == bmu1 else bmu2_)
                    if bmu2 in self.rev_map[b]:
                        self.sparse_adj[(y, b)] += 1
                    self.pairwise_index[(y, b)] = 1.0 - (
                        float(self.sparse_adj[(y, b)]) / float(self.cluster_cardinality[y])
                    )

        self._refresh_under_prototyped_labels()

        unique_y = _ordered_unique_1d(Y_single)
        if len(self.rev_map) > 1:
            for y in unique_y:
                self.singleton_index[y] = min(
                    [self.pairwise_index[(y, b)] for b in self.rev_map.keys() if b != y]
                )
            self._recompute_global_index()
        return self.index

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "OverlapIndex":
        """
        Fit the overlap index on a complete labeled dataset.

        This sklearn-style method delegates to fit_offline with reset_state=True
        and returns self. The computed overlap index is available through the
        ``index`` attribute.

        Parameters
        ----------
        X : np.ndarray
            Raw input samples.
        Y : np.ndarray
            Class labels aligned with X.

        Returns
        -------
        OverlapIndex
            The fitted overlap-index instance.
        """
        self.fit_offline(X, Y, reset_state=True)
        return self

    def partial_fit(self, X: np.ndarray, Y: np.ndarray) -> "OverlapIndex":
        """
        Update the overlap index from a labeled batch and return self.

        For ARTMAP backends, this performs an incremental batch update. For
        offline backends, this behaves like add_batch, which refits the backend
        on the provided batch and recomputes the index.

        Parameters
        ----------
        X : np.ndarray
            Raw input samples.
        Y : np.ndarray
            Class labels aligned with X.

        Returns
        -------
        OverlapIndex
            The updated overlap-index instance.
        """
        self.add_batch(X, Y)
        return self

    def score(
        self,
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> float:
        """
        Return the current overlap-index score.

        If X and Y are provided together, refit on that labeled dataset first.
        """
        if X is None and Y is None:
            return float(self.index)
        if X is None or Y is None:
            raise ValueError("score expects both X and Y, or neither.")
        return float(self.fit_offline(X, Y, reset_state=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the highest-scoring global prototype id for each sample.
        """
        self._validate_sparse_backend(X)
        X_arr = _validate_feature_matrix(X)
        if not self.rev_map or self._model.n_clusters_total <= 0:
            raise ValueError("This OverlapIndex instance is not fit yet.")
        self._check_feature_count(X_arr)

        X_prep = self._prep_X(X_arr)
        result = np.empty(X_prep.shape[0], dtype=int)
        for i, x in enumerate(X_prep):
            ids, _ = self._model.topk(x, k=1)
            if ids.size == 0:
                raise ValueError("The backend did not return any prototype ids for prediction.")
            result[i] = int(ids[0])
        return result

    def fit_predict(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit the estimator and return per-sample global prototype ids.
        """
        self.fit(X, Y)
        return self.predict(X)

    # ---- offline (centroid backends primary; also works for ARTMAP if you want) ----

    def _build_label_indicator_matrices(
        self,
        Y_sets: list[tuple],
        classes: np.ndarray,
    ) -> None:
        """Build cached sparse label matrices for pairwise row retrieval."""
        self.label_to_index_ = {label: j for j, label in enumerate(classes)}
        self.index_to_label_ = {j: label for label, j in self.label_to_index_.items()}

        row_ind = []
        col_ind = []
        for i, labels in enumerate(Y_sets):
            for label in labels:
                row_ind.append(i)
                col_ind.append(self.label_to_index_[label])

        data = np.ones(len(row_ind), dtype=np.uint8)
        Y_csr = sparse.csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(len(Y_sets), len(classes)),
            dtype=np.uint8,
        )
        Y_csr.sort_indices()

        self._label_indicator_csr_ = Y_csr
        self._label_indicator_csc_ = Y_csr.tocsc()
        self._label_indicator_csc_.sort_indices()

        self._positive_rows_by_label_index_ = {}
        for j in range(len(classes)):
            start = self._label_indicator_csc_.indptr[j]
            stop = self._label_indicator_csc_.indptr[j + 1]
            rows = self._label_indicator_csc_.indices[start:stop]
            self._positive_rows_by_label_index_[j] = rows

    def _valid_rows_for_pair(self, a: Any, b: Any) -> np.ndarray:
        """Return row indices where label a is present and label b is absent."""
        a_idx = self.label_to_index_[a]
        b_idx = self.label_to_index_[b]

        a_rows = self._positive_rows_by_label_index_[a_idx]
        b_rows = self._positive_rows_by_label_index_[b_idx]

        if a_rows.size == 0 or b_rows.size == 0:
            return a_rows
        return np.setdiff1d(a_rows, b_rows, assume_unique=True)

    def _resolve_lazy_pairwise_cardinality(self, pair: Any) -> int:
        """Resolve one directional denominator without materializing a key."""
        try:
            a, b = pair
        except (TypeError, ValueError):
            return 0
        if a == b:
            return 0
        if self._pairwise_multilabel:
            try:
                if a not in self.label_to_index_ or b not in self.label_to_index_:
                    return 0
                return int(self._valid_rows_for_pair(a, b).size)
            except (KeyError, TypeError, ValueError):
                return 0
        # Single-label rows are all valid evidence for every other observed
        # class.  Unknown labels retain the mapping's zero default.
        if a not in self.cluster_cardinality or b not in self.cluster_cardinality:
            return 0
        return int(self.cluster_cardinality.get(a, 0))

    def _resolve_lazy_pairwise_value(self, kind: str, pair: Any) -> Any:
        """Resolve direct lookup values for sparse pairwise mappings."""
        if kind == "cardinality":
            explicit = dict.get(self.pairwise_cardinality, pair, None)
            if explicit is not None:
                return explicit
            return self._resolve_lazy_pairwise_cardinality(pair)

        explicit = dict.get(self.pairwise_index, pair, None)
        if explicit is not None:
            return explicit
        denominator = self._resolve_lazy_pairwise_cardinality(pair)
        if denominator <= 0:
            return np.nan if self._pairwise_multilabel else 1.0
        hits = int(self._pairwise_hits.get(pair, 0))
        return 1.0 - (float(hits) / float(denominator))

    def _build_top_m_competitors_from_prototypes(
        self,
        classes: np.ndarray,
        top_m: int,
    ) -> Dict[Any, np.ndarray]:
        """
        Select each label's nearest prototype-owning competitors by minimum
        prototype-to-prototype distance.

        Distances are evaluated against global prototype tiles rather than by
        constructing one ``own x target`` matrix for every class pair.  The
        running minimum for each target class is only ``O(C)`` state per source
        label, and the prototype tile is bounded by the configured scratch
        budget.  This retains the exact minimum-distance semantics and stable
        class-order tie breaking of the historical pairwise implementation.
        """
        top_m = _validate_positive_integer(top_m, "top_m")

        try:
            centers = self._model.centers
        except AttributeError as exc:
            raise AttributeError(
                "The backend must expose centers for top_m competitor selection."
            ) from exc

        class_to_ids = self._model.class_center_id_arrays
        classes_list = classes.tolist()
        n_classes = len(classes_list)
        competitors: Dict[Any, np.ndarray] = {}

        # Resolve global prototype ownership once.  The adapters assign global
        # ids into ``centers``; unowned ids are ignored defensively.
        n_prototypes = int(np.asarray(centers).shape[0])
        owner_positions = np.full(n_prototypes, -1, dtype=int)
        normalized_ids: list[np.ndarray] = []
        for position, label in enumerate(classes_list):
            ids = np.asarray(
                class_to_ids.get(label, np.asarray([], dtype=int)),
                dtype=int,
            ).reshape(-1)
            normalized_ids.append(ids)
            valid_ids = ids[(ids >= 0) & (ids < n_prototypes)]
            owner_positions[valid_ids] = int(position)

        # ``pairwise_distances`` materializes a float distance tile.  Reserve a
        # conservative multiplier for the distance output and advanced-indexed
        # center tile, then keep at least one target prototype per iteration.
        try:
            budget_bytes = int(self.offline_memory_budget_mb) * 1024 * 1024
        except (TypeError, ValueError):  # pragma: no cover - constructor validates
            budget_bytes = 1
        center_dtype = np.asarray(centers).dtype
        center_bytes = int(center_dtype.itemsize)
        distance_bytes = int(np.dtype(np.float64).itemsize)

        for source_position, source_label in enumerate(classes_list):
            own_ids = normalized_ids[source_position]
            if own_ids.size == 0 or n_prototypes == 0:
                competitors[source_label] = np.asarray([], dtype=object)
                continue

            # Per-target-class minima are the only cross-tile state.  Sorting
            # this C-length vector below preserves deterministic class-order
            # ties without storing all source/target distances.
            class_best = np.full(n_classes, np.inf, dtype=float)
            own_centers = np.asarray(centers[own_ids])
            n_own = max(1, int(own_centers.shape[0]))
            per_target_bytes = (
                n_own * distance_bytes
                + center_bytes * max(1, int(own_centers.shape[1]))
                + distance_bytes
            )
            tile_size = max(1, int(budget_bytes // max(1, 4 * per_target_bytes)))
            tile_size = min(n_prototypes, tile_size)

            for start in range(0, n_prototypes, tile_size):
                stop = min(start + tile_size, n_prototypes)
                target_ids = np.arange(start, stop, dtype=int)
                target_centers = np.asarray(centers[target_ids])
                # Distances are Euclidean, matching the existing
                # ``pairwise_distances`` call (and cosine BallCover centers,
                # which are normalized at fit time).
                distances = pairwise_distances(own_centers, target_centers)
                target_best = np.min(distances, axis=0)
                target_owners = owner_positions[target_ids]
                valid = target_owners >= 0
                if np.any(valid):
                    np.minimum.at(
                        class_best,
                        target_owners[valid],
                        target_best[valid],
                    )

            # Never select the source label itself.  ``lexsort`` uses class
            # order as the secondary key, exactly matching stable sorting of
            # ``(distance, class)`` records in the former implementation.
            class_best[source_position] = np.inf
            class_order = np.arange(n_classes, dtype=int)
            order = np.lexsort((class_order, class_best))
            order = order[np.isfinite(class_best[order])]
            selected = order[:top_m]
            competitors[source_label] = np.asarray(
                [classes_list[int(position)] for position in selected],
                dtype=object,
            )

        return competitors

    def _build_multilabel_competitors(
        self,
        classes: np.ndarray,
    ) -> Dict[Any, np.ndarray]:
        """Build source-label competitor arrays for multi-label scoring."""
        if self.multilabel_pair_mode == "all":
            return _LazyAllCompetitors(classes)
        if self.multilabel_pair_mode == "top_m":
            return self._build_top_m_competitors_from_prototypes(
                classes,
                int(self.top_m),
            )
        raise ValueError("multilabel_pair_mode must be one of {'all', 'top_m'}.")

    # ---- backend-neutral offline score helpers -------------------------

    def _prepare_offline_score_input(self, X: Any) -> Any:
        """Prepare one offline query block through the backend contract.

        New offline adapters expose ``prepare_score_input`` and
        ``score_block_prepared``.  The fallback keeps compatibility with
        third-party/test adapters that still expose only ``_scores_matrix``.
        """
        prepare = getattr(self._model, "prepare_score_input", None)
        if prepare is None:
            return X
        try:
            return prepare(X)
        except NotImplementedError:
            return X

    def _score_offline_block(self, X_prepared: Any, ids: Any = None) -> np.ndarray:
        """Score one prepared row block against selected global prototypes."""
        score_block = getattr(self._model, "score_block_prepared", None)
        if score_block is not None:
            try:
                scores = score_block(X_prepared, ids=ids)
            except NotImplementedError:
                scores = self._model._scores_matrix(X_prepared, ids)
        else:
            # ``_scores_matrix`` is the historical private adapter hook.
            scores = self._model._scores_matrix(X_prepared, ids)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        if scores.ndim != 2:
            raise ValueError(
                "Offline backend score_block_prepared must return a 2D score matrix."
            )
        return scores

    def _offline_tile_limits(self, n_rows: int, n_prototypes: int) -> Tuple[int, int]:
        """Return row/prototype tile sizes bounded by estimator controls.

        A score matrix is the dominant temporary in all supported offline
        adapters.  Reserve a conservative multiplier for reductions and
        indexing temporaries; this keeps the actual allocation below the
        public budget while retaining at least one row and one prototype.
        """
        n_rows = max(1, int(n_rows))
        n_prototypes = max(1, int(n_prototypes))
        centers = getattr(self._model, "centers", None)
        try:
            itemsize = int(np.asarray(centers).dtype.itemsize)
        except Exception:
            itemsize = int(np.dtype(np.float32).itemsize)
        # Eight score-sized temporaries is deliberately conservative.  The
        # budget is a planning bound rather than a hard allocator guarantee.
        cells = max(
            1,
            int(self.offline_memory_budget_mb) * 1024 * 1024
            // max(1, itemsize * 8),
        )
        proto_tile = min(n_prototypes, max(1, int(np.sqrt(cells))))
        row_tile = max(1, min(n_rows, cells // max(1, proto_tile)))
        if self.offline_chunk_size is not None:
            row_tile = min(row_tile, int(self.offline_chunk_size))
        return int(row_tile), int(proto_tile)

    def _iter_id_tiles(self, ids: np.ndarray, max_prototypes: int):
        """Yield contiguous slices of a prototype-id array."""
        ids = np.asarray(ids, dtype=int).reshape(-1)
        step = max(1, int(max_prototypes))
        for start in range(0, ids.size, step):
            yield ids[start : start + step]

    def _source_threshold_block(
        self,
        X_prepared: Any,
        own_ids: np.ndarray,
        row_tile: int,
        proto_tile: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return each source row's second-best source score threshold.

        With one source prototype there is no second source score.  We retain
        the historical top-two degeneracy by using ``-inf`` as the missing
        threshold, so every finite target activation counts as overlap.
        """
        own_ids = np.asarray(own_ids, dtype=int).reshape(-1)
        n_rows = int(X_prepared.shape[0])
        threshold = np.full(n_rows, -np.inf, dtype=float)
        best_all = np.full(n_rows, -np.inf, dtype=float)
        if own_ids.size <= 1:
            # There is no second source prototype.  Preserve the historical
            # degenerate top-two behavior: every finite target score beats
            # the missing threshold, yielding a zero overlap score.
            return threshold, best_all

        for row_start in range(0, n_rows, row_tile):
            row_stop = min(row_start + row_tile, n_rows)
            X_block = X_prepared[row_start:row_stop]
            best = np.full(row_stop - row_start, -np.inf, dtype=float)
            second = np.full(row_stop - row_start, -np.inf, dtype=float)
            for id_tile in self._iter_id_tiles(own_ids, proto_tile):
                scores = self._score_offline_block(X_block, id_tile)
                if scores.shape[1] == 0:
                    continue
                tile_best = np.max(scores, axis=1)
                if scores.shape[1] > 1:
                    tile_second = np.partition(scores, -2, axis=1)[:, -2]
                else:
                    tile_second = np.full(tile_best.shape, -np.inf, dtype=float)
                # Merge two sorted (best, second) pairs from the existing and
                # current prototype tiles without materializing all scores.
                merged = np.stack((best, second, tile_best, tile_second), axis=1)
                merged = np.partition(merged, -2, axis=1)[:, -2:]
                best = np.max(merged, axis=1)
                second = np.min(merged, axis=1)
            threshold[row_start:row_stop] = second
            best_all[row_start:row_stop] = best
        return threshold, best_all

    def _best_target_scores_block(
        self,
        X_prepared: Any,
        target_ids: np.ndarray,
        row_tile: int,
        proto_tile: int,
    ) -> np.ndarray:
        """Return each row's best score among one target class's prototypes."""
        target_ids = np.asarray(target_ids, dtype=int).reshape(-1)
        n_rows = int(X_prepared.shape[0])
        result = np.full(n_rows, -np.inf, dtype=float)
        if target_ids.size == 0:
            return result
        # Keep both row and prototype tiling here.  The source caller may pass
        # an already row-tiled block, while this loop remains correct for
        # arbitrarily large target prototype sets.
        for row_start in range(0, n_rows, row_tile):
            row_stop = min(row_start + row_tile, n_rows)
            X_block = X_prepared[row_start:row_stop]
            best = np.full(row_stop - row_start, -np.inf, dtype=float)
            for id_tile in self._iter_id_tiles(target_ids, proto_tile):
                scores = self._score_offline_block(X_block, id_tile)
                if scores.shape[1]:
                    best = np.maximum(best, np.max(scores, axis=1))
            result[row_start:row_stop] = best
        return result

    def _score_source_target_pair(
        self,
        X_input: Any,
        source_ids: np.ndarray,
        target_ids: np.ndarray,
    ) -> int:
        """Count rows where target score strictly beats source threshold."""
        n_rows = int(X_input.shape[0])
        if n_rows == 0 or source_ids.size == 0 or target_ids.size == 0:
            return 0
        row_tile, proto_tile = self._offline_tile_limits(
            n_rows,
            max(int(source_ids.size), int(target_ids.size)),
        )
        overlap_count = 0
        for row_start in range(0, n_rows, row_tile):
            row_stop = min(row_start + row_tile, n_rows)
            # Preparation is deliberately scoped to this row tile.  This is
            # essential for BallCover cosine mode, whose preparation may
            # allocate a dense R x D normalization temporary.
            X_block = self._prepare_offline_score_input(
                X_input[row_start:row_stop]
            )
            block_rows = int(row_stop - row_start)
            threshold, _source_best = self._source_threshold_block(
                X_block,
                source_ids,
                block_rows,
                proto_tile,
            )
            target_best = self._best_target_scores_block(
                X_block,
                target_ids,
                block_rows,
                proto_tile,
            )
            # Strict ``>`` is intentional: exact score ties remain
            # source-owned at the second-own threshold.
            overlap_count += int(np.count_nonzero(target_best > threshold))
        return overlap_count

    def _fit_offline_multilabel_event_scored(
        self,
        X_prep: Any,
        Y_sets: list[tuple],
        classes: np.ndarray,
    ) -> float:
        """Score all multi-label source events in row/prototype tiles.

        Source thresholds are computed once per positive-label event.  Target
        classes are scored in packed prototype tiles over the same prepared
        row block, so this path does not loop over every directional class
        pair or allocate a dense C-by-C diagnostic matrix.
        """
        classes_list = classes.tolist()
        label_to_position = {label: i for i, label in enumerate(classes_list)}
        class_to_ids = self._model.class_center_id_arrays
        integer_prototypes = {
            i: np.asarray(class_to_ids.get(label, np.asarray([], dtype=int)), dtype=int)
            for i, label in enumerate(classes_list)
        }

        event_rows: list[int] = []
        event_sources: list[int] = []
        for row, labels in enumerate(Y_sets):
            for label in labels:
                event_rows.append(int(row))
                event_sources.append(int(label_to_position[label]))

        self._build_label_indicator_matrices(Y_sets, classes)
        self.competitors_ = self._build_multilabel_competitors(classes)
        self._pairwise_multilabel = True
        self._score_classes = tuple(classes_list)

        # Determine which source labels have any valid directional evidence
        # without enumerating all class pairs.  A competitor is valid iff it
        # is absent from at least one source-positive row.
        source_rows: dict[Any, list[int]] = {label: [] for label in classes_list}
        for row, labels in enumerate(Y_sets):
            for label in labels:
                source_rows[label].append(int(row))
        source_evaluable_all: dict[Any, bool] = {}
        for source in classes_list:
            rows = source_rows[source]
            source_evaluable_all[source] = any(
                len(Y_sets[row]) < len(classes_list)
                for row in rows
            )

        selected_targets: dict[Any, set[Any]] = {}
        if self.multilabel_pair_mode == "top_m":
            selected_targets = {
                source: set(self.competitors_.get(source, []))
                for source in classes_list
            }

        source_result = compute_second_best_source_scores(
            X_prep,
            event_rows,
            event_sources,
            integer_prototypes,
            self._model,
            memory_budget_mb=self.offline_memory_budget_mb,
            row_cap=self.offline_chunk_size,
        )

        # Encode indicator positives once for vectorized target-absence masks
        # across all packed target row blocks.
        n_classes = len(classes_list)
        positive_rows = np.repeat(
            np.arange(self._label_indicator_csr_.shape[0], dtype=np.int64),
            np.diff(self._label_indicator_csr_.indptr),
        )
        positive_cols = self._label_indicator_csr_.indices.astype(
            np.int64,
            copy=False,
        )
        positive_keys = positive_rows * n_classes + positive_cols
        allowed_codes = None
        if self.multilabel_pair_mode == "top_m":
            allowed_codes = np.asarray(
                [
                    label_to_position[source] * n_classes
                    + label_to_position[target]
                    for source, targets in selected_targets.items()
                    for target in targets
                ],
                dtype=np.int64,
            )

        # The target iterator yields the same event rows in contiguous blocks;
        # retain an event cursor rather than relying on original row numbers,
        # which can repeat for multi-label observations.
        event_offset = 0
        for target_block in iter_target_class_blocks(
            X_prep,
            integer_prototypes,
            self._model,
            memory_budget_mb=self.offline_memory_budget_mb,
            row_cap=self.offline_chunk_size,
            sample_rows=event_rows,
        ):
            n_block = int(target_block.row_indices.size)
            if n_block == 0:
                continue
            thresholds = source_result.second_best_scores[
                event_offset : event_offset + n_block
            ]
            block_sources = source_result.source_class_ids[
                event_offset : event_offset + n_block
            ]
            block_scores = np.asarray(target_block.best_scores)
            # Extract hits one target column at a time.  ``np.nonzero`` on the
            # full R x C comparison can allocate two coordinate arrays as
            # large as the entire dense target block; column-wise masks keep
            # transient storage O(R) while preserving the same strict
            # comparator and pair-count results.
            original_rows = np.asarray(target_block.row_indices, dtype=int)
            for target_position in range(n_classes):
                hit_rows = np.flatnonzero(
                    block_scores[:, target_position] > thresholds
                )
                if hit_rows.size == 0:
                    continue
                source_positions = block_sources[hit_rows].astype(int, copy=False)
                keep = source_positions != target_position

                # A hit is valid only when its target label is absent from the
                # original multi-label row.  Encode indicator positives once
                # and use vectorized key membership for this O(R) column.
                hit_keys = (
                    original_rows[hit_rows].astype(np.int64) * n_classes
                    + np.int64(target_position)
                )
                keep &= ~np.isin(hit_keys, positive_keys)

                if self.multilabel_pair_mode == "top_m":
                    pair_codes = (
                        source_positions.astype(np.int64) * n_classes
                        + np.int64(target_position)
                    )
                    if allowed_codes is not None and allowed_codes.size:
                        keep &= np.isin(pair_codes, allowed_codes)
                    else:
                        keep[:] = False

                pair_codes = (
                    source_positions[keep].astype(np.int64) * n_classes
                    + np.int64(target_position)
                )
                if pair_codes.size:
                    unique_codes, counts = np.unique(pair_codes, return_counts=True)
                    for code, count in zip(unique_codes.tolist(), counts.tolist()):
                        source_position, target_position_ = divmod(
                            int(code),
                            n_classes,
                        )
                        pair = (
                            classes_list[source_position],
                            classes_list[target_position_],
                        )
                        self._pairwise_hits[pair] += int(count)
                        self.sparse_adj[pair] += int(count)
            event_offset += n_block

        # Resolve per-source minima from sparse hit pairs.  Any valid pair
        # with zero hits has the default score one, so it need not be stored.
        # Zero-denominator diagnostics are exposed through a lazy set-like
        # view below; no C² pair list is built during fitting.
        unevaluable_labels: list[Any] = []
        hits_by_source: dict[Any, list[tuple[Any, int]]] = defaultdict(list)
        for (source, target), hits in self._pairwise_hits.items():
            hits_by_source[source].append((target, int(hits)))
        for source in classes_list:
            if self.multilabel_pair_mode == "top_m":
                selected = selected_targets[source]
                valid_cardinality = {
                    target: int(self._valid_rows_for_pair(source, target).size)
                    for target in selected
                    if target != source
                }
                valid = {
                    target
                    for target, denominator in valid_cardinality.items()
                    if denominator > 0
                }
            else:
                selected = None
                valid = None
                valid_cardinality = {}

            source_evaluable = (
                bool(valid)
                if valid is not None
                else source_evaluable_all[source]
            )
            if not source_evaluable:
                unevaluable_labels.append(source)
                self.singleton_index[source] = np.nan
                continue

            scores = []
            for target, hits in hits_by_source.get(source, ()):
                if valid is not None and target not in valid:
                    continue
                if valid is None:
                    denominator = int(
                        self._valid_rows_for_pair(source, target).size
                    )
                else:
                    denominator = valid_cardinality[target]
                if denominator > 0:
                    scores.append(1.0 - float(hits) / float(denominator))
            self.singleton_index[source] = min(scores) if scores else 1.0

        self.unevaluable_pairs_ = _LazyUnevaluablePairs(self, classes_list)
        self.unevaluable_labels_ = tuple(unevaluable_labels)
        if self.unevaluable_labels_:
            self._warn_unevaluable_multilabel(self.unevaluable_labels_)

        excluded = self._normalized_exclude_classes()
        non_excluded = [label for label in classes_list if label not in excluded]
        evaluable = [
            label for label in non_excluded if label not in set(unevaluable_labels)
        ]
        if non_excluded and not evaluable:
            raise ValueError(
                "No non-excluded multi-label source labels have evaluable "
                "selected competitor pairs."
            )
        self._recompute_global_index()
        return self.index

    def _fit_offline_single_event_scored(
        self,
        X_prep: Any,
        Y: np.ndarray,
        classes: np.ndarray,
    ) -> float:
        """Score single-label events with the packed universal target pass."""
        classes_list = classes.tolist()
        label_to_position = {label: i for i, label in enumerate(classes_list)}
        class_to_ids = self._model.class_center_id_arrays
        integer_prototypes = {
            i: np.asarray(class_to_ids.get(label, np.asarray([], dtype=int)), dtype=int)
            for i, label in enumerate(classes_list)
        }
        rows = np.arange(int(np.asarray(Y).size), dtype=int)
        sources = np.asarray(
            [label_to_position[label] for label in np.asarray(Y, dtype=object)],
            dtype=int,
        )
        source_result = compute_second_best_source_scores(
            X_prep,
            rows,
            sources,
            integer_prototypes,
            self._model,
            memory_budget_mb=self.offline_memory_budget_mb,
            row_cap=self.offline_chunk_size,
        )

        event_offset = 0
        for target_block in iter_target_class_blocks(
            X_prep,
            integer_prototypes,
            self._model,
            memory_budget_mb=self.offline_memory_budget_mb,
            row_cap=self.offline_chunk_size,
            sample_rows=rows,
        ):
            n_block = int(target_block.row_indices.size)
            if n_block == 0:
                continue
            thresholds = source_result.second_best_scores[
                event_offset : event_offset + n_block
            ]
            block_sources = source_result.source_class_ids[
                event_offset : event_offset + n_block
            ]
            block_scores = np.asarray(target_block.best_scores)
            n_classes = len(classes_list)
            # As in the multi-label path, avoid materializing a full R x C
            # hit-coordinate list.  Each target column contributes only an
            # O(R) mask and pair-code vector.
            for target_position in range(n_classes):
                hit_rows = np.flatnonzero(
                    block_scores[:, target_position] > thresholds
                )
                if hit_rows.size == 0:
                    continue
                source_positions = block_sources[hit_rows].astype(int, copy=False)
                keep = source_positions != target_position
                pair_codes = (
                    source_positions[keep].astype(np.int64) * n_classes
                    + np.int64(target_position)
                )
                if pair_codes.size:
                    unique_codes, counts = np.unique(pair_codes, return_counts=True)
                    for code, count in zip(unique_codes.tolist(), counts.tolist()):
                        source_position, target_position_ = divmod(
                            int(code),
                            n_classes,
                        )
                        pair = (
                            classes_list[source_position],
                            classes_list[target_position_],
                        )
                        self._pairwise_hits[pair] += int(count)
                        self.sparse_adj[pair] += int(count)
            event_offset += n_block

        hit_scores: dict[Any, list[float]] = defaultdict(list)
        for (source, _target), hits_value in self._pairwise_hits.items():
            support = int(self.cluster_cardinality.get(source, 0))
            if support > 0 and hits_value:
                hit_scores[source].append(
                    1.0 - float(hits_value) / float(support)
                )
        for source in classes_list:
            # Every single-label source has a positive denominator for every
            # observed competitor.  Thus unmaterialized pairs contribute the
            # default score one and need not be enumerated.
            scores = hit_scores.get(source, [])
            self.singleton_index[source] = min(scores) if scores else 1.0
        self._pairwise_multilabel = False
        self._score_classes = tuple(classes_list)
        self._recompute_global_index()
        return self.index

    def _fit_offline_centroid_optimized_multilabel(
        self,
        X_prep: np.ndarray,
        Y_sets: list[tuple],
        classes: np.ndarray,
    ) -> float:
        """
        Compute offline overlap for multi-label data with pairwise denominators.
        """
        return self._fit_offline_multilabel_event_scored(
            X_prep,
            Y_sets,
            classes,
        )

    def _fit_offline_centroid_optimized(
        self,
        X_prep: np.ndarray,
        Y: np.ndarray,
        classes: np.ndarray,
    ) -> float:
        """
        Optimized offline overlap computation for centroid-style backends.

        This avoids materializing scores for all clusters for every sample. Instead,
        for each class pair (y, b), it scores only clusters owned by y or b in one
        vectorized block and updates the overlap counts from the resulting top-2 BMUs.
        """
        if hasattr(self._model, "prepare_score_input") and hasattr(
            self._model, "score_block_prepared"
        ):
            return self._fit_offline_single_event_scored(X_prep, Y, classes)

        class_to_cluster_arrays = self._model.class_center_id_arrays
        rows_by_class = _group_indices_by_label(Y)
        self._pairwise_multilabel = False
        self._score_classes = tuple(classes.tolist())

        for y in classes:
            row_idx = rows_by_class.get(y, np.asarray([], dtype=int))
            own_ids = np.asarray(
                class_to_cluster_arrays.get(y, np.asarray([], dtype=int)),
                dtype=int,
            )
            if row_idx.size == 0 or own_ids.size == 0:
                continue

            X_y = X_prep[row_idx]
            n_y = int(row_idx.size)
            for b in classes:
                if b == y:
                    continue
                other_ids = np.asarray(
                    class_to_cluster_arrays.get(b, np.asarray([], dtype=int)),
                    dtype=int,
                )
                if other_ids.size == 0:
                    continue

                pair = (y, b)
                self.pairwise_cardinality[pair] = n_y
                overlap_count = self._score_source_target_pair(
                    X_y,
                    own_ids,
                    other_ids,
                )
                if overlap_count:
                    self.sparse_adj[pair] += overlap_count
                    self._pairwise_hits[pair] = int(overlap_count)
                    self.pairwise_index[pair] = 1.0 - (
                        float(overlap_count) / float(n_y)
                    )

        if len(self.rev_map) > 1:
            for y in classes:
                self.singleton_index[y] = min(
                    [self.pairwise_index[(y, b)] for b in self.rev_map.keys() if b != y]
                )
            self._recompute_global_index()
        return self.index

    def _fit_offline_replay(
        self,
        X_prep: np.ndarray,
        Y: np.ndarray,
        classes: np.ndarray,
    ) -> float:
        """
        Compatibility fallback for non-centroid backends.
        """
        BMU1 = self._model.bmu_for_class_batch(X_prep, Y)
        for x, y, bmu1 in zip(X_prep, Y, BMU1):
            bmu1 = int(bmu1)
            top2bmu = self.predict_subset_pairs(x, y)

            for b in self.rev_map.keys():
                if b == y:
                    continue
                bmu2 = int(bmu1)
                if len(top2bmu[b]) > 1:
                    bmu2_, bmu3_ = top2bmu[b]
                    bmu2 = int(bmu3_ if bmu2_ == bmu1 else bmu2_)
                if bmu2 in self.rev_map[b]:
                    self.sparse_adj[(y, b)] += 1
                self.pairwise_index[(y, b)] = 1.0 - (
                    float(self.sparse_adj[(y, b)]) / float(self.cluster_cardinality[y])
                )

        if len(self.rev_map) > 1:
            for y in classes:
                self.singleton_index[y] = min(
                    [self.pairwise_index[(y, b)] for b in self.rev_map.keys() if b != y]
                )
            self._recompute_global_index()
        return self.index

    def fit_offline(self, X: np.ndarray, Y: Any, reset_state: bool = True) -> float:
        """
        Fit the backend on a full labeled dataset and compute the overlap index.

        Centroid backends use a chunked vectorized class-pair scoring path. Other
        backends fall back to replaying samples with backend top-k hooks.

        Parameters
        ----------
        X : np.ndarray
            Raw input samples.
        Y : np.ndarray
            Class labels aligned with X.
        reset_state : bool, default=True
            If True, reset overlap-index bookkeeping and construct a fresh
            backend. False is supported only for ARTMAP continuation.

        Returns
        -------
        float
            The current overlap index value.
        """
        if not reset_state and self._is_offline_backend:
            raise ValueError(
                "reset_state=False is supported only for ARTMAP backends; "
                "offline backends must be fit from a complete dataset."
            )

        X, Y_sets = self._validate_input_data(X, Y)
        if reset_state:
            self._reset_indices()
            self._model = self._build_model()
        else:
            self._check_feature_count(X)
        if X.shape[0] == 0:
            self._warn_empty_input()
            return self.index

        classes = _ordered_unique_labels(Y_sets)
        X_prep = self._prep_X(X)
        is_multilabel = any(len(labels) > 1 for labels in Y_sets)

        if is_multilabel:
            if not (
                self._is_offline_backend
                and (
                    hasattr(self._model, "score_block_prepared")
                    or hasattr(self._model, "_scores_matrix")
                )
            ):
                raise NotImplementedError(
                    "Multi-label scoring is currently implemented only for offline "
                    "backends with vectorized prototype scoring."
                )
            X_fit, Y_fit = _expand_multilabel_for_backend(X_prep, Y_sets)
        else:
            X_fit = X_prep
            Y_fit = _flatten_single_label_sets(Y_sets)

        # Full ARTMAP fits use a freshly built backend and must preserve the
        # estimator's configured match-tracking behavior.
        if self._is_artmap_backend:
            self._model.partial_fit(
                X_fit,
                Y_fit,
                match_tracking=self.match_tracking,
            )
        else:
            self._model.fit_offline(X_fit, Y_fit)
        self.n_features_in_ = int(X.shape[1])
        self.rev_map = defaultdict(set, {c: set(s) for c, s in self._model.class_to_clusters.items()})
        self._refresh_under_prototyped_labels()

        # Cardinalities remain per-label positive sample counts.
        for c in classes:
            self.cluster_cardinality[c] += sum(c in labels for labels in Y_sets)
            if c not in self.singleton_index:
                self.singleton_index[c] = 1.0

        if len(classes) <= 1:
            if self._included_singleton_labels():
                self._warn_single_class()
            else:
                self._warn_all_observed_classes_excluded()
            return self.index

        if is_multilabel:
            return self._fit_offline_centroid_optimized_multilabel(
                X_prep,
                Y_sets,
                classes,
            )

        Y_single = _flatten_single_label_sets(Y_sets)

        if self._is_offline_backend and (
            hasattr(self._model, "score_block_prepared")
            or hasattr(self._model, "_scores_matrix")
        ):
            return self._fit_offline_centroid_optimized(X_prep, Y_single, classes)

        return self._fit_offline_replay(X_prep, Y_single, classes)
