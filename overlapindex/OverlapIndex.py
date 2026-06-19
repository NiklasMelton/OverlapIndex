import warnings
from collections import defaultdict
from collections.abc import Iterable
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


def _is_multilabel_entry(y: Any) -> bool:
    """Return True when one target entry is a collection of labels."""
    if isinstance(y, (str, bytes)):
        return False
    return isinstance(y, Iterable)


def _label_sets_from_indicator_matrix(Y: Any) -> list[set]:
    """Convert a dense or sparse binary indicator matrix to label sets."""
    if sparse.issparse(Y):
        Y_csr = Y.tocsr(copy=True)
        Y_csr.eliminate_zeros()
        if not np.all(Y_csr.data == 1):
            raise ValueError("2D Y must contain only 0/1 indicators.")
        return [
            set(Y_csr.indices[Y_csr.indptr[i] : Y_csr.indptr[i + 1]])
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

    return [set(np.flatnonzero(row)) for row in Y_num]


def _normalize_label_sets(Y: Any) -> list[set]:
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

        entries = list(Y)
        label_sets = []
        for y in entries:
            labels = set(y) if _is_multilabel_entry(y) else {y}
            label_sets.append(labels)

    if any(len(labels) == 0 for labels in label_sets):
        raise ValueError("Every sample must have at least one label.")

    return label_sets


def _flatten_single_label_sets(Y_sets: list[set]) -> np.ndarray:
    """Convert normalized singleton label sets back to a 1D label vector."""
    return np.asarray([next(iter(labels)) for labels in Y_sets], dtype=object)


def _expand_multilabel_for_backend(
    X: np.ndarray,
    Y_sets: list[set],
) -> Tuple[np.ndarray, np.ndarray]:
    """Duplicate each sample once for every positive label before backend fit."""
    X_expanded = []
    Y_expanded = []

    for x, labels in zip(X, Y_sets):
        for label in labels:
            X_expanded.append(x)
            Y_expanded.append(label)

    return np.asarray(X_expanded, dtype=float), np.asarray(Y_expanded, dtype=object)


def _ordered_unique_labels(Y_sets: list[set]) -> np.ndarray:
    """Return labels in first-observed order across normalized label sets."""
    labels = []
    seen = set()
    for sample_labels in Y_sets:
        for label in sample_labels:
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
            hash(label)
        except TypeError as exc:
            raise TypeError("exclude_classes entries must be hashable.") from exc
        if label not in seen:
            seen.add(label)
            deduplicated.append(label)
    return deduplicated

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
        self.multilabel_pair_mode = multilabel_pair_mode
        self.top_m = top_m
        self.exclude_classes = exclude_classes
        self._validate_multilabel_params()

        # indices / bookkeeping
        self.sparse_adj = defaultdict(lambda: 0)
        self.cluster_cardinality = defaultdict(int)
        self.rev_map = defaultdict(set)
        self.pairwise_index = defaultdict(lambda: 1.0)
        self.singleton_index = defaultdict(lambda: 1.0)
        self.pairwise_cardinality = defaultdict(int)
        self.competitors_ = {}
        self.label_to_index_ = {}
        self.index_to_label_ = {}
        self._label_indicator_csr_ = None
        self._label_indicator_csc_ = None
        self._positive_rows_by_label_index_ = {}
        self.index = 1.0

        self._model: _BaseManyToOneClusteringModel = self._build_model()

    def _validate_multilabel_params(self) -> None:
        """Validate multi-label pair-selection parameters."""
        if self.multilabel_pair_mode not in {"all", "top_m"}:
            raise ValueError("multilabel_pair_mode must be one of {'all', 'top_m'}.")
        if self.multilabel_pair_mode == "top_m":
            try:
                top_m = int(self.top_m) if self.top_m is not None else None
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "top_m must be a positive integer when multilabel_pair_mode='top_m'."
                ) from exc
            if top_m is None or top_m <= 0:
                raise ValueError(
                    "top_m must be a positive integer when multilabel_pair_mode='top_m'."
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
        X = np.asarray(X, dtype=float)
        if self._is_artmap_backend:
            return complement_code(X)
        return X

    def _validate_input_data(
        self,
        X: np.ndarray,
        Y: Any,
    ) -> Tuple[np.ndarray, list[set]]:
        """Validate aligned batch inputs before preprocessing."""
        X_arr = np.asarray(X, dtype=float)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array; got shape {X_arr.shape}.")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or infinite values.")

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

    def _reset_indices(self) -> None:
        """Reset overlap-index bookkeeping without replacing the clustering backend."""
        self.sparse_adj = defaultdict(lambda: 0)
        self.cluster_cardinality = defaultdict(int)
        self.rev_map = defaultdict(set)
        self.pairwise_index = defaultdict(lambda: 1.0)
        self.singleton_index = defaultdict(lambda: 1.0)
        self.pairwise_cardinality = defaultdict(int)
        self.competitors_ = {}
        self.label_to_index_ = {}
        self.index_to_label_ = {}
        self._label_indicator_csr_ = None
        self._label_indicator_csc_ = None
        self._positive_rows_by_label_index_ = {}
        self.index = 1.0

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
        return [label for label in observed_labels if label not in excluded]

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
        x_ = np.asarray(x, dtype=float)

        if x_.ndim != 1:
            raise ValueError("x must be a 1D array or list")
        if not np.all(np.isfinite(x_)):
            raise ValueError("x contains NaN or infinite values.")

        x_prep = self._prep_X(x_.reshape(1, -1))
        self._model.partial_fit(
            x_prep, [y], match_tracking=self.match_tracking
        )

        # ARTMAP path: latest assigned label is BMU1
        bmu1 = int(self._model.model.module_a.labels_[-1])

        # keep rev_map in sync with backend
        self.rev_map[y].add(bmu1)

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
        if X.shape[0] == 0:
            self._warn_empty_input()
            return self.index

        if any(len(labels) > 1 for labels in Y_sets):
            raise NotImplementedError(
                "Multi-label add_batch is currently implemented only for offline backends."
            )

        Y_single = _flatten_single_label_sets(Y_sets)

        if np.unique(Y_single).size <= 1:
            self._warn_single_class()

        X_prep = self._prep_X(X)
        self._model.partial_fit(
            X_prep, Y_single, match_tracking=self.match_tracking
        )

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

        unique_y = np.unique(Y_single)
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
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array; got shape {X_arr.shape}.")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or infinite values.")
        if not self.rev_map or self._model.n_clusters_total <= 0:
            raise ValueError("This OverlapIndex instance is not fit yet.")

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
        Y_sets: list[set],
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

    def _build_top_m_competitors_from_prototypes(
        self,
        classes: np.ndarray,
        top_m: int,
    ) -> Dict[Any, np.ndarray]:
        """
        Select each label's nearest prototype-owning competitors by minimum
        prototype-to-prototype distance.
        """
        if top_m is None or int(top_m) <= 0:
            raise ValueError("top_m must be a positive integer.")
        top_m = int(top_m)

        try:
            centers = self._model.centers
        except AttributeError as exc:
            raise AttributeError(
                "The backend must expose centers for top_m competitor selection."
            ) from exc

        class_to_ids = self._model.class_center_id_arrays
        competitors = {}

        for a in classes:
            own_ids = class_to_ids.get(a)
            if own_ids is None or own_ids.size == 0:
                competitors[a] = np.asarray([], dtype=object)
                continue

            distances = []
            for b in classes:
                if b == a:
                    continue

                other_ids = class_to_ids.get(b)
                if other_ids is None or other_ids.size == 0:
                    continue

                d_ab = pairwise_distances(centers[own_ids], centers[other_ids]).min()
                distances.append((float(d_ab), b))

            distances.sort(key=lambda t: t[0])
            competitors[a] = np.asarray(
                [b for _, b in distances[:top_m]],
                dtype=object,
            )

        return competitors

    def _build_multilabel_competitors(
        self,
        classes: np.ndarray,
    ) -> Dict[Any, np.ndarray]:
        """Build source-label competitor arrays for multi-label scoring."""
        if self.multilabel_pair_mode == "all":
            return {
                y: np.asarray([b for b in classes if b != y], dtype=object)
                for y in classes
            }
        if self.multilabel_pair_mode == "top_m":
            return self._build_top_m_competitors_from_prototypes(
                classes,
                int(self.top_m),
            )
        raise ValueError("multilabel_pair_mode must be one of {'all', 'top_m'}.")

    def _fit_offline_centroid_optimized_multilabel(
        self,
        X_prep: np.ndarray,
        Y_sets: list[set],
        classes: np.ndarray,
    ) -> float:
        """
        Compute offline overlap for multi-label data with pairwise denominators.
        """
        class_to_cluster_arrays = self._model.class_center_id_arrays

        self._build_label_indicator_matrices(Y_sets, classes)
        self.competitors_ = self._build_multilabel_competitors(classes)

        for y in classes:
            own_ids = class_to_cluster_arrays.get(y)
            if own_ids is None or own_ids.size == 0:
                continue

            for b in self.competitors_.get(y, []):
                if b == y:
                    continue

                other_ids = class_to_cluster_arrays.get(b)
                if other_ids is None or other_ids.size == 0:
                    continue

                valid_rows = self._valid_rows_for_pair(y, b)
                n_valid = int(valid_rows.size)
                if n_valid == 0:
                    continue

                self.pairwise_cardinality[(y, b)] += n_valid
                X_valid = X_prep[valid_rows]
                candidate_ids = np.concatenate((own_ids, other_ids))
                chunk_size = (
                    n_valid
                    if self.offline_chunk_size is None
                    else int(self.offline_chunk_size)
                )
                if chunk_size <= 0:
                    raise ValueError("offline_chunk_size must be a positive integer or None.")

                overlap_count = 0
                for start in range(0, n_valid, chunk_size):
                    stop = min(start + chunk_size, n_valid)
                    X_chunk = X_valid[start:stop]
                    scores = self._model._scores_matrix(X_chunk, candidate_ids)

                    if scores.shape[1] == 0:
                        continue
                    if scores.shape[1] == 1:
                        selected = np.full(stop - start, int(candidate_ids[0]), dtype=int)
                    else:
                        top2_rel = np.argpartition(scores, -2, axis=1)[:, -2:]
                        top2_scores = np.take_along_axis(scores, top2_rel, axis=1)
                        order = np.argsort(top2_scores, axis=1)[:, ::-1]
                        top2_rel_sorted = np.take_along_axis(top2_rel, order, axis=1)
                        top2_ids = candidate_ids[top2_rel_sorted]
                        selected = np.where(
                            np.isin(top2_ids[:, 0], own_ids),
                            top2_ids[:, 1],
                            top2_ids[:, 0],
                        )

                    overlap_count += int(np.isin(selected, other_ids).sum())

                self.sparse_adj[(y, b)] += overlap_count
                self.pairwise_index[(y, b)] = 1.0 - (
                    float(self.sparse_adj[(y, b)])
                    / float(self.pairwise_cardinality[(y, b)])
                )

        if len(classes) > 1:
            for y in classes:
                valid_scores = [
                    self.pairwise_index[(y, b)]
                    for b in self.competitors_.get(y, [])
                    if self.pairwise_cardinality[(y, b)] > 0
                ]
                self.singleton_index[y] = min(valid_scores) if valid_scores else 1.0
            self._recompute_global_index()

        return self.index

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
        BMU1 = self._model.bmu_for_class_batch(X_prep, Y)
        class_to_cluster_arrays = self._model.class_center_id_arrays

        for y in classes:
            row_idx = np.where(Y == y)[0]
            if row_idx.size == 0:
                continue

            X_y = X_prep[row_idx]
            bmu1_y = BMU1[row_idx]
            n_y = row_idx.size
            chunk_size = n_y if self.offline_chunk_size is None else int(self.offline_chunk_size)
            if chunk_size <= 0:
                raise ValueError("offline_chunk_size must be a positive integer or None.")
            own_ids = class_to_cluster_arrays.get(y)
            if own_ids is None or own_ids.size == 0:
                continue

            for b in self.rev_map.keys():
                if b == y:
                    continue

                other_ids = class_to_cluster_arrays.get(b)
                if other_ids is None or other_ids.size == 0:
                    continue

                candidate_ids = np.concatenate((own_ids, other_ids))
                if candidate_ids.size == 0:
                    continue

                overlap_count = 0
                for start in range(0, n_y, chunk_size):
                    stop = min(start + chunk_size, n_y)
                    X_chunk = X_y[start:stop]
                    bmu1_chunk = bmu1_y[start:stop]

                    scores = self._model._scores_matrix(X_chunk, candidate_ids)
                    if scores.shape[1] == 0:
                        continue

                    if scores.shape[1] == 1:
                        selected = np.full(stop - start, int(candidate_ids[0]), dtype=int)
                    else:
                        top2_rel = np.argpartition(scores, -2, axis=1)[:, -2:]
                        top2_scores = np.take_along_axis(scores, top2_rel, axis=1)
                        order = np.argsort(top2_scores, axis=1)[:, ::-1]
                        top2_rel_sorted = np.take_along_axis(top2_rel, order, axis=1)
                        top2_ids = candidate_ids[top2_rel_sorted]
                        selected = np.where(top2_ids[:, 0] == bmu1_chunk, top2_ids[:, 1], top2_ids[:, 0])

                    overlap_count += int(np.isin(selected, other_ids).sum())
                self.sparse_adj[(y, b)] += overlap_count
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
            If True, reset overlap-index bookkeeping before fitting.

        Returns
        -------
        float
            The current overlap index value.
        """
        if reset_state:
            self._reset_indices()

        X, Y_sets = self._validate_input_data(X, Y)
        if X.shape[0] == 0:
            self._warn_empty_input()
            return self.index

        classes = _ordered_unique_labels(Y_sets)
        if classes.size <= 1:
            self._warn_single_class()

        X_prep = self._prep_X(X)
        is_multilabel = any(len(labels) > 1 for labels in Y_sets)

        if is_multilabel:
            if not (self._is_offline_backend and hasattr(self._model, "_scores_matrix")):
                raise NotImplementedError(
                    "Multi-label scoring is currently implemented only for offline "
                    "backends with vectorized prototype scoring."
                )
            X_fit, Y_fit = _expand_multilabel_for_backend(X_prep, Y_sets)
        else:
            X_fit = X_prep
            Y_fit = _flatten_single_label_sets(Y_sets)

        # Fit backend and sync rev_map
        self._model.fit_offline(X_fit, Y_fit)
        self.rev_map = defaultdict(set, {c: set(s) for c, s in self._model.class_to_clusters.items()})

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

        if self._is_offline_backend and hasattr(self._model, "_scores_matrix"):
            return self._fit_offline_centroid_optimized(X_prep, Y_single, classes)

        return self._fit_offline_replay(X_prep, Y_single, classes)
