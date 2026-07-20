"""Continuous-target Overlap Index estimator."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans

try:
    from sklearn.base import BaseEstimator
except ImportError:  # pragma: no cover - scikit-learn is a required dependency
    class BaseEstimator:  # type: ignore[no-redef]
        """Fallback base class when sklearn is unavailable at import time."""

        pass

from overlapindex.clustering import (
    _BallCoverManyToOne,
    _BaseManyToOneClusteringModel,
    _KMeansManyToOne,
    _MiniBatchKMeansManyToOne,
)
from overlapindex.utils import (
    _validate_feature_matrix,
    _validate_finite_positive_real,
    _validate_positive_integer,
)


ModelType = Literal["KMeans", "MiniBatchKMeans", "BallCover", "Fuzzy", "Hypersphere"]
TargetCover = Literal["auto", "quantile", "kmeans"]
TargetDistance = Literal["auto", "wasserstein", "sliced_wasserstein"]
TargetScaling = Literal["standard", "none", "minmax", "robust"]
AdjacencyMode = Literal["hard_top1", "soft_topk"]
Aggregation = Literal["support_weighted", "macro"]
NullMode = Literal["auto", "refit_permutation", "fixed_structure_permutation"]


class ContinuousOverlapIndex(BaseEstimator):
    """
    Compute an overlap index for continuous regression targets.

    The estimator builds target-space cells, fits class-owned feature
    prototypes using those cells as pseudo-labels, and scores feature-space
    prototype overlap by empirical target-distribution disagreement. The score
    is normalized by a permutation null so that 1.0 indicates no observed
    overlap, values between 0.0 and 1.0 indicate partial separation, and 0.0
    indicates complete or permutation-equivalent overlap. Worse-than-null loss
    remains at the 0.0 lower endpoint and is exposed through ``loss_ratio_``.
    """

    def __init__(
        self,
        rho: float = 0.9,
        r_hat: float = np.inf,
        model_type: ModelType = "MiniBatchKMeans",
        match_tracking: str = "MT+",
        kmeans_k: Union[int, Dict[Any, int]] = 8,
        kmeans_kwargs: Optional[dict] = None,
        ballcover_k: Union[int, Dict[Any, int], Literal["auto"]] = "auto",
        ballcover_radius: Union[float, Dict[Any, float], Literal["auto"]] = 0.25,
        ballcover_kwargs: Optional[dict] = None,
        offline_chunk_size: Optional[int] = 10_000,
        target_cover: TargetCover = "auto",
        n_target_cells: Union[int, Literal["auto"]] = "auto",
        target_cover_kwargs: Optional[dict] = None,
        target_distance: TargetDistance = "auto",
        adjacency_mode: AdjacencyMode = "soft_topk",
        top_k: int = 5,
        feature_temperature: float = 1.0,
        normalization: Literal["permutation"] = "permutation",
        null_mode: NullMode = "auto",
        n_null_permutations: int = 20,
        auto_null_work_threshold: int = 100_000,
        aggregation: Aggregation = "support_weighted",
        target_scaling: TargetScaling = "standard",
        n_projections: int = 64,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the continuous-target overlap estimator.

        Parameters
        ----------
        rho : float, default=0.9
            Reserved ARTMAP vigilance parameter. Continuous-target ARTMAP
            backends are not supported in the current offline implementation.
        r_hat : float, default=np.inf
            Reserved Hypersphere ARTMAP radius constraint.
        model_type : {"KMeans", "MiniBatchKMeans", "BallCover"}, default="MiniBatchKMeans"
            Offline backend used to build feature-space prototypes. ``"Fuzzy"``
            and ``"Hypersphere"`` are accepted by the type signature for API
            consistency but raise ``NotImplementedError`` during fitting.
        match_tracking : str, default="MT+"
            Reserved ARTMAP match-tracking setting.
        kmeans_k : int or dict, default=8
            Number of feature prototypes per target cell for KMeans backends,
            or a dictionary keyed by every target-cell id.
        kmeans_kwargs : dict, optional
            Keyword arguments forwarded to the selected scikit-learn centroid
            backend. An explicit ``random_state`` here overrides the top-level
            seed for feature-prototype fitting.
        ballcover_k : int, dict, or "auto", default="auto"
            Number of balls per target cell, cell-specific counts, or ``"auto"``
            for greedy fixed-radius covering.
        ballcover_radius : float, dict, or "auto", default=0.25
            Ball radius, cell-specific radii, or ``"auto"`` when a fixed number
            of balls should determine the radius. Exactly one of
            ``ballcover_k`` and ``ballcover_radius`` may be ``"auto"``.
        ballcover_kwargs : dict, optional
            Additional options forwarded to BallCover. An explicit
            ``random_state`` here overrides the top-level seed for the backend.
        offline_chunk_size : int or None, default=10000
            Maximum row block used for feature-prototype adjacency scoring.
            ``None`` scores each available row block at once.
        target_cover : {"auto", "quantile", "kmeans"}, default="auto"
            Target-space cell construction. Auto selects quantiles for a
            univariate target and KMeans for multivariate targets.
        n_target_cells : int or "auto", default="auto"
            Requested number of target cells. Auto uses a sample-size-dependent
            value between 8 and 64, capped by the sample count.
        target_cover_kwargs : dict, optional
            Extra keyword arguments forwarded to target-space KMeans.
        target_distance : {"auto", "wasserstein", "sliced_wasserstein"}, default="auto"
            Distance between empirical target distributions. Auto selects 1D
            Wasserstein distance for a univariate target and sliced Wasserstein
            distance for multivariate targets.
        adjacency_mode : {"hard_top1", "soft_topk"}, default="soft_topk"
            Rule used to connect each own feature prototype to competing
            prototypes.
        top_k : int, default=5
            Maximum number of non-own competitors receiving adjacency mass in
            ``"soft_topk"`` mode.
        feature_temperature : float, default=1.0
            Positive softmax temperature for soft top-k adjacency. Lower values
            concentrate mass on the strongest competitor.
        normalization : {"permutation"}, default="permutation"
            Continuous-index calibration method. Permutation is currently the
            only supported value.
        null_mode : {"auto", "refit_permutation", "fixed_structure_permutation"}, default="auto"
            Permutation-null strategy. Auto switches from refitting to the
            approximate fixed-structure mode at ``auto_null_work_threshold``.
        n_null_permutations : int, default=20
            Number of shuffled target assignments used to estimate null loss.
        auto_null_work_threshold : int, default=100000
            Auto-mode cutoff applied to
            ``n_samples * n_null_permutations``.
        aggregation : {"support_weighted", "macro"}, default="support_weighted"
            Aggregation used for the public ``index`` value.
        target_scaling : {"standard", "none", "minmax", "robust"}, default="standard"
            Scaling applied to target columns before cell construction and
            target-distribution distances.
        n_projections : int, default=64
            Number of random directions used by sliced Wasserstein distance.
        random_state : int, optional
            Seed for target-cell construction, sliced-Wasserstein projections,
            null permutations, and backend fitting when no backend-specific
            seed is provided.
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
        self.target_cover = target_cover
        self.n_target_cells = n_target_cells
        self.target_cover_kwargs = target_cover_kwargs
        self.target_distance = target_distance
        self.adjacency_mode = adjacency_mode
        self.top_k = top_k
        self.feature_temperature = feature_temperature
        self.normalization = normalization
        self.null_mode = null_mode
        self.n_null_permutations = n_null_permutations
        self.auto_null_work_threshold = auto_null_work_threshold
        self.aggregation = aggregation
        self.target_scaling = target_scaling
        self.n_projections = n_projections
        self.random_state = random_state

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset fitted-state attributes and score diagnostics."""
        self.index = 1.0
        self.macro_index_ = 1.0
        self.actual_loss_ = 0.0
        self.null_loss_ = 0.0
        self.null_loss_samples_ = []
        self.loss_ratio_ = 0.0
        self.null_mode_ = None
        self.auto_null_work_ = 0
        self.prototype_index_ = {}
        self.prototype_loss_ = {}
        self.prototype_null_loss_ = {}
        self.prototype_target_values_ = {}
        self._prototype_target_values_scaled_ = {}
        self.prototype_target_weights_ = {}
        self.prototype_support_ = {}
        self.prototype_target_mean_ = {}
        self.prototype_target_cov_ = {}
        self.prototype_target_radius_ = {}
        self.prototype_adjacency_ = {}
        self.prototype_adjacency_count_ = {}
        self.prototype_adjacency_normalized_ = {}
        self.target_cell_to_prototypes_ = defaultdict(set)
        self.prototype_to_target_cell_ = {}
        self.target_cell_ids_ = None
        self.target_cover_ = None
        self.target_distance_ = None
        self.target_center_ = None
        self.target_scale_ = None
        self.target_directions_ = None
        self.Y_train_ = None
        self.Y_scaled_ = None
        self.own_prototype_ids_ = None
        self._rows_by_prototype_ = {}
        self._model: Optional[_BaseManyToOneClusteringModel] = None
        if hasattr(self, "n_features_in_"):
            del self.n_features_in_

    @property
    def weighted_index(self) -> float:
        """Return the support-weighted continuous overlap index."""
        if self.aggregation == "support_weighted":
            return float(self.index)
        return float(self._aggregate_index("support_weighted"))

    def set_params(self, **params: Any) -> "ContinuousOverlapIndex":
        """Update estimator parameters and clear fitted state."""
        super().set_params(**params)
        self._reset_state()
        return self

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ContinuousOverlapIndex":
        """Fit the estimator and store the current continuous overlap index."""
        self.fit_offline(X, Y, reset_state=True)
        return self

    def partial_fit(self, X: np.ndarray, Y: np.ndarray) -> "ContinuousOverlapIndex":
        """
        Refit the continuous overlap index on the provided batch.

        V1 supports the same refit semantics as offline ``OverlapIndex``
        backends. True incremental continuous-target updates are intentionally
        deferred.
        """
        self.add_batch(X, Y)
        return self

    def add_batch(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Fit on a batch and return the current index."""
        return self.fit_offline(X, Y, reset_state=True)

    def score(
        self,
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> float:
        """Return the current score, or refit and return a score."""
        if X is None and Y is None:
            return float(self.index)
        if X is None or Y is None:
            raise ValueError("score expects both X and Y, or neither.")
        return float(self.fit_offline(X, Y, reset_state=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the highest-scoring global prototype id for each sample."""
        if self._model is None or self._model.n_clusters_total <= 0:
            raise ValueError("This ContinuousOverlapIndex instance is not fit yet.")

        X_arr = self._validate_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but this "
                f"ContinuousOverlapIndex instance was fit with "
                f"{self.n_features_in_} features."
            )
        result = np.empty(X_arr.shape[0], dtype=int)
        for i, x in enumerate(X_arr):
            ids, _ = self._model.topk(x, k=1)
            if ids.size == 0:
                raise ValueError("The backend did not return any prototype ids for prediction.")
            result[i] = int(ids[0])
        return result

    def fit_predict(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fit the estimator and return per-sample global prototype ids."""
        self.fit(X, Y)
        return self.predict(X)

    def fit_offline(self, X: np.ndarray, Y: np.ndarray, reset_state: bool = True) -> float:
        """Fit the backend on a complete regression dataset and compute COI."""
        if not reset_state:
            raise ValueError(
                "reset_state=False is not supported by ContinuousOverlapIndex; "
                "continuous backends must be fit from a complete dataset."
            )
        self._validate_sparse_backend(X)
        self._validate_params()

        X_arr = self._validate_X(X)
        Y_arr = self._validate_Y(Y)
        if X_arr.shape[0] != Y_arr.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of rows; got {X_arr.shape[0]} and {Y_arr.shape[0]}."
            )
        self._reset_state()
        if X_arr.shape[0] == 0:
            self._warn_empty_input()
            return self.index

        self.auto_null_work_ = int(X_arr.shape[0]) * int(self.n_null_permutations)
        self.null_mode_ = self._resolve_null_mode(X_arr.shape[0])
        Y_scaled = self._scale_targets(Y_arr)
        target_cell_ids = self._build_target_cells(Y_scaled)
        unique_cells = np.unique(target_cell_ids)

        self._model = self._build_model()
        self._model.fit_offline(X_arr, target_cell_ids)

        own_proto = self._model.bmu_for_class_batch(X_arr, target_cell_ids)
        self.n_features_in_ = int(X_arr.shape[1])
        self._store_training_targets(Y_arr, Y_scaled, target_cell_ids)
        self.own_prototype_ids_ = own_proto
        self._sync_prototype_bookkeeping(target_cell_ids, own_proto)

        if unique_cells.size <= 1:
            self._warn_single_target_cell()
            return self.index

        if self._model.n_clusters_total <= 1:
            self._warn_single_prototype()
            return self.index

        self._build_prototype_adjacency(X_arr, own_proto)
        self._compute_index_from_current_state(X_arr, Y_scaled)
        return float(self.index)

    def _validate_params(self) -> None:
        """Validate constructor parameters that affect fitting."""
        if self.model_type in {"Fuzzy", "Hypersphere"}:
            raise NotImplementedError(
                "ContinuousOverlapIndex V1 supports only offline backends: "
                "'KMeans', 'MiniBatchKMeans', and 'BallCover'."
            )
        if self.model_type not in {"KMeans", "MiniBatchKMeans", "BallCover"}:
            raise ValueError(
                "model_type must be one of {'KMeans', 'MiniBatchKMeans', 'BallCover', 'Fuzzy', 'Hypersphere'}."
            )
        if self.target_cover not in {"auto", "quantile", "kmeans"}:
            raise ValueError("target_cover must be one of {'auto', 'quantile', 'kmeans'}.")
        if self.target_distance not in {"auto", "wasserstein", "sliced_wasserstein"}:
            raise ValueError(
                "target_distance must be one of {'auto', 'wasserstein', 'sliced_wasserstein'}."
            )
        if self.adjacency_mode not in {"hard_top1", "soft_topk"}:
            raise ValueError("adjacency_mode must be one of {'hard_top1', 'soft_topk'}.")
        if self.normalization != "permutation":
            raise ValueError("normalization must be 'permutation'.")
        if self.null_mode not in {"auto", "refit_permutation", "fixed_structure_permutation"}:
            raise ValueError(
                "null_mode must be one of {'auto', 'refit_permutation', 'fixed_structure_permutation'}."
            )
        if self.aggregation not in {"support_weighted", "macro"}:
            raise ValueError("aggregation must be one of {'support_weighted', 'macro'}.")
        if self.target_scaling not in {"standard", "none", "minmax", "robust"}:
            raise ValueError("target_scaling must be one of {'standard', 'none', 'minmax', 'robust'}.")
        _validate_positive_integer(self.n_null_permutations, "n_null_permutations")
        _validate_positive_integer(self.auto_null_work_threshold, "auto_null_work_threshold")
        _validate_positive_integer(self.n_projections, "n_projections")
        _validate_positive_integer(self.top_k, "top_k")
        _validate_finite_positive_real(self.feature_temperature, "feature_temperature")
        if self.offline_chunk_size is not None:
            _validate_positive_integer(self.offline_chunk_size, "offline_chunk_size")
        if self.n_target_cells != "auto":
            _validate_positive_integer(self.n_target_cells, "n_target_cells")

    def _validate_sparse_backend(self, X: Any) -> None:
        """Reject sparse features for backends that require dense arrays."""
        if sparse.issparse(X) and self.model_type not in {"KMeans", "MiniBatchKMeans"}:
            raise TypeError(
                "Sparse X is supported only for model_type='KMeans' and "
                f"'MiniBatchKMeans'; got model_type={self.model_type!r}."
            )

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        """Validate feature input."""
        self._validate_sparse_backend(X)
        return _validate_feature_matrix(X)

    @staticmethod
    def _validate_Y(Y: np.ndarray) -> np.ndarray:
        """Validate and normalize continuous targets to a 2D float array."""
        try:
            Y_arr = np.asarray(Y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Y must be numeric for ContinuousOverlapIndex.") from exc
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)
        elif Y_arr.ndim != 2:
            raise ValueError(f"Y must be a 1D or 2D numeric array; got shape {Y_arr.shape}.")
        if Y_arr.shape[1] == 0:
            raise ValueError("Y must contain at least one target column.")
        if not np.all(np.isfinite(Y_arr)):
            raise ValueError("Y contains NaN or infinite values.")
        return Y_arr.astype(float, copy=False)

    def _scale_targets(self, Y: np.ndarray) -> np.ndarray:
        """Scale targets for target cover construction and distances."""
        if self.target_scaling == "none":
            center = np.zeros(Y.shape[1], dtype=float)
            scale = np.ones(Y.shape[1], dtype=float)
        elif self.target_scaling == "standard":
            center = np.mean(Y, axis=0)
            scale = np.std(Y, axis=0)
        elif self.target_scaling == "minmax":
            center = np.min(Y, axis=0)
            scale = np.max(Y, axis=0) - center
        else:
            q25, q75 = np.percentile(Y, [25, 75], axis=0)
            center = np.median(Y, axis=0)
            scale = q75 - q25

        scale = np.where(np.abs(scale) <= np.finfo(float).eps, 1.0, scale)
        self.target_center_ = center
        self.target_scale_ = scale
        return (Y - center) / scale

    def _build_target_cells(self, Y_scaled: np.ndarray) -> np.ndarray:
        """Build target-space pseudo-labels."""
        self.is_multivariate_target_ = Y_scaled.shape[1] > 1
        self.n_targets_ = int(Y_scaled.shape[1])

        cover = self.target_cover
        if cover == "auto":
            cover = "quantile" if self.n_targets_ == 1 else "kmeans"
        if cover == "quantile" and self.n_targets_ != 1:
            raise ValueError("target_cover='quantile' is supported only for univariate Y.")
        self.target_cover_ = cover

        n_cells = self._resolve_n_target_cells(Y_scaled.shape[0])
        if cover == "quantile":
            return self._quantile_target_cells(Y_scaled[:, 0], n_cells)
        return self._kmeans_target_cells(Y_scaled, n_cells)

    def _resolve_n_target_cells(self, n_samples: int) -> int:
        """Resolve the requested number of target cells."""
        if self.n_target_cells == "auto":
            return int(min(max(8, int(np.sqrt(n_samples))), 64, n_samples))
        try:
            n_cells = _validate_positive_integer(self.n_target_cells, "n_target_cells")
        except ValueError as exc:
            raise ValueError("n_target_cells must be a positive integer or 'auto'.") from exc
        return int(min(n_cells, n_samples))

    @staticmethod
    def _quantile_target_cells(y: np.ndarray, n_cells: int) -> np.ndarray:
        """Assign univariate targets to quantile cells."""
        if y.size == 0:
            return np.asarray([], dtype=int)
        quantiles = np.linspace(0.0, 1.0, n_cells + 1)
        edges = np.quantile(y, quantiles)
        inner_edges = np.unique(edges[1:-1])
        if inner_edges.size == 0:
            return np.zeros(y.shape[0], dtype=int)
        return np.searchsorted(inner_edges, y, side="right").astype(int, copy=False)

    def _kmeans_target_cells(self, Y_scaled: np.ndarray, n_cells: int) -> np.ndarray:
        """Assign targets to KMeans target cells."""
        kwargs = {
            "n_init": "auto",
            "random_state": self.random_state,
        }
        kwargs.update(self.target_cover_kwargs or {})
        model = KMeans(n_clusters=n_cells, **kwargs)
        return model.fit_predict(Y_scaled).astype(int, copy=False)

    def _build_model(self) -> _BaseManyToOneClusteringModel:
        """Construct the selected offline backend."""
        kmeans_kwargs = dict(self.kmeans_kwargs or {})
        kmeans_kwargs.setdefault("random_state", self.random_state)
        if self.model_type == "KMeans":
            return _KMeansManyToOne(k=self.kmeans_k, kmeans_kwargs=kmeans_kwargs)
        if self.model_type == "MiniBatchKMeans":
            return _MiniBatchKMeansManyToOne(k=self.kmeans_k, kmeans_kwargs=kmeans_kwargs)
        if self.model_type == "BallCover":
            kwargs = dict(self.ballcover_kwargs or {})
            kwargs.setdefault("random_state", self.random_state)
            return _BallCoverManyToOne(
                k=self.ballcover_k,
                radius=self.ballcover_radius,
                **kwargs,
            )
        raise NotImplementedError(
            "ContinuousOverlapIndex V1 supports only offline backends."
        )

    def _store_training_targets(
        self,
        Y: np.ndarray,
        Y_scaled: np.ndarray,
        target_cell_ids: np.ndarray,
    ) -> None:
        """Store target arrays and resolved target-cell ids."""
        self.Y_train_ = Y
        self.Y_scaled_ = Y_scaled
        self.target_cell_ids_ = target_cell_ids
        self.target_distance_ = self._resolve_target_distance(Y.shape[1])
        if self.target_distance_ == "sliced_wasserstein":
            self.target_directions_ = self._make_projection_directions(Y.shape[1])

    def _resolve_target_distance(self, n_targets: int) -> str:
        """Resolve auto target distance."""
        if self.target_distance == "auto":
            return "wasserstein" if n_targets == 1 else "sliced_wasserstein"
        if self.target_distance == "wasserstein" and n_targets != 1:
            raise ValueError("target_distance='wasserstein' is supported only for univariate Y.")
        if self.target_distance == "sliced_wasserstein" and n_targets == 1:
            return "wasserstein"
        return self.target_distance

    def _make_projection_directions(self, n_targets: int) -> np.ndarray:
        """Create random unit directions for sliced Wasserstein distance."""
        rng = np.random.default_rng(self.random_state)
        directions = rng.normal(size=(int(self.n_projections), n_targets))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms <= np.finfo(float).eps, 1.0, norms)
        return directions / norms

    def _sync_prototype_bookkeeping(
        self,
        target_cell_ids: np.ndarray,
        own_proto: np.ndarray,
    ) -> None:
        """Build prototype ownership, support, and target-measure diagnostics."""
        self.target_cell_to_prototypes_ = defaultdict(set)
        self.prototype_to_target_cell_ = {}
        for cell, ids in self._model.class_to_clusters.items():
            self.target_cell_to_prototypes_[int(cell)].update(int(pid) for pid in ids)
            for pid in ids:
                self.prototype_to_target_cell_[int(pid)] = int(cell)

        rows_by_proto: Dict[int, np.ndarray] = {}
        self.prototype_target_values_ = {}
        self._prototype_target_values_scaled_ = {}
        self.prototype_target_weights_ = {}
        self.prototype_support_ = {}
        self.prototype_target_mean_ = {}
        self.prototype_target_cov_ = {}
        self.prototype_target_radius_ = {}

        for pid in np.unique(own_proto):
            rows = np.flatnonzero(own_proto == pid)
            pid_int = int(pid)
            rows_by_proto[pid_int] = rows
            raw_values = self.Y_train_[rows]
            scaled_values = self.Y_scaled_[rows]
            self.prototype_target_values_[pid_int] = raw_values
            self._prototype_target_values_scaled_[pid_int] = scaled_values
            self.prototype_target_weights_[pid_int] = np.ones(rows.size, dtype=float)
            self.prototype_support_[pid_int] = int(rows.size)
            self.prototype_target_mean_[pid_int] = np.mean(raw_values, axis=0)
            if rows.size > 1:
                self.prototype_target_cov_[pid_int] = np.atleast_2d(
                    np.cov(raw_values, rowvar=False)
                )
            else:
                self.prototype_target_cov_[pid_int] = np.zeros((self.n_targets_, self.n_targets_))
            diffs = raw_values - self.prototype_target_mean_[pid_int]
            self.prototype_target_radius_[pid_int] = float(np.max(np.linalg.norm(diffs, axis=1)))

        self._rows_by_prototype_ = rows_by_proto

    def _build_prototype_adjacency(self, X: np.ndarray, own_proto: np.ndarray) -> None:
        """Build prototype adjacency from feature-space competitors."""
        counts, normalized = self._adjacency_for_model(X, own_proto, self._model)
        self.prototype_adjacency_count_ = counts
        self.prototype_adjacency_ = counts
        self.prototype_adjacency_normalized_ = normalized

    def _adjacency_for_model(
        self,
        X: np.ndarray,
        own_proto: np.ndarray,
        model: _BaseManyToOneClusteringModel,
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        """Return prototype adjacency for a fitted backend."""
        counts: Dict[Tuple[int, int], float] = defaultdict(float)
        n_clusters = int(model.n_clusters_total)
        n_competitors = min(int(self.top_k), max(0, n_clusters - 1))
        requested = min(n_competitors + 1, n_clusters)

        for x, p in zip(X, own_proto):
            p_int = int(p)
            ids, scores = model.topk(x, k=requested)
            mask = ids != p_int
            q_ids = ids[mask][:n_competitors]
            q_scores = scores[mask][:n_competitors]
            if q_ids.size == 0:
                continue
            if self.adjacency_mode == "hard_top1":
                counts[(p_int, int(q_ids[0]))] += 1.0
                continue

            weights = self._soft_topk_weights(q_scores)
            for q_id, weight in zip(q_ids, weights):
                counts[(p_int, int(q_id))] += float(weight)

        normalized = {}
        outgoing = defaultdict(float)
        for (p, _), value in counts.items():
            outgoing[p] += float(value)
        for key, value in counts.items():
            p, _ = key
            normalized[key] = float(value) / float(outgoing[p])

        return dict(counts), normalized

    def _soft_topk_weights(self, scores: np.ndarray) -> np.ndarray:
        """Return temperature-scaled softmax weights over competing prototype scores."""
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return np.asarray([], dtype=float)
        scaled = (scores - np.max(scores)) / float(self.feature_temperature)
        weights = np.exp(scaled)
        total = float(np.sum(weights))
        if total <= np.finfo(float).eps:
            return np.full(scores.shape, 1.0 / float(scores.size), dtype=float)
        return weights / total

    def _compute_index_from_current_state(self, X: np.ndarray, Y_scaled: np.ndarray) -> None:
        """Compute actual/null losses and derived COI summaries."""
        actual_loss, proto_loss = self._loss_for_values_by_prototype(
            self._prototype_target_values_scaled_
        )
        null_loss = self._estimate_null_loss(X, Y_scaled)

        self.actual_loss_ = float(actual_loss)
        self.null_loss_ = float(null_loss)
        self.prototype_loss_ = proto_loss
        self.prototype_null_loss_ = {
            pid: self.null_loss_
            for pid in self.prototype_support_
        }

        eps = np.finfo(float).eps
        if self.null_loss_ <= eps:
            if self.actual_loss_ <= eps:
                self.loss_ratio_ = 0.0
            else:
                self.loss_ratio_ = np.inf
        else:
            self.loss_ratio_ = float(self.actual_loss_ / self.null_loss_)

        self.prototype_index_ = {}
        for pid, loss in self.prototype_loss_.items():
            null = self.prototype_null_loss_.get(pid, self.null_loss_)
            self.prototype_index_[pid] = self._index_from_losses(loss, null)

        self.macro_index_ = float(self._aggregate_index("macro"))
        self.index = float(self._aggregate_index(self.aggregation))

    @staticmethod
    def _index_from_losses(actual_loss: float, null_loss: float) -> float:
        """Return the bounded continuous index for actual and null losses."""
        actual = float(actual_loss)
        null = float(null_loss)
        eps = np.finfo(float).eps
        if null <= eps:
            return 1.0 if actual <= eps else 0.0
        return float(np.clip(1.0 - (actual / null), 0.0, 1.0))

    def _resolve_null_mode(self, n_samples: int) -> str:
        """Resolve the null estimation mode for the current fit."""
        if self.null_mode != "auto":
            return str(self.null_mode)
        work = int(n_samples) * int(self.n_null_permutations)
        if work >= int(self.auto_null_work_threshold):
            return "fixed_structure_permutation"
        return "refit_permutation"

    def _estimate_null_loss(self, X: np.ndarray, Y_scaled: np.ndarray) -> float:
        """
        Estimate permutation-null loss for the configured null mode.
        """
        rng = np.random.default_rng(self.random_state)
        losses = []

        for _ in range(int(self.n_null_permutations)):
            permuted = Y_scaled[rng.permutation(Y_scaled.shape[0])]
            if self.null_mode_ == "fixed_structure_permutation":
                loss = self._loss_for_fixed_structure_permutation(permuted)
            else:
                loss = self._loss_for_permuted_dataset(X, permuted)
            losses.append(loss)

        self.null_loss_samples_ = [float(loss) for loss in losses]
        return float(np.mean(losses)) if losses else 0.0

    def _loss_for_fixed_structure_permutation(self, Y_scaled: np.ndarray) -> float:
        """Compute null loss with fitted prototype structure held fixed."""
        if not self.prototype_adjacency_normalized_:
            return 0.0

        values_by_proto = {
            int(pid): Y_scaled[rows]
            for pid, rows in self._rows_by_prototype_.items()
        }
        loss, _ = self._loss_for_components(
            values_by_proto,
            self.prototype_support_,
            self.prototype_adjacency_normalized_,
        )
        return float(loss)

    def _loss_for_permuted_dataset(self, X: np.ndarray, Y_scaled: np.ndarray) -> float:
        """Compute actual-style loss for one permuted target assignment."""
        target_cell_ids = self._target_cells_for_values(Y_scaled)
        if np.unique(target_cell_ids).size <= 1:
            return 0.0

        model = self._build_model()
        model.fit_offline(X, target_cell_ids)
        if model.n_clusters_total <= 1:
            return 0.0

        own_proto = model.bmu_for_class_batch(X, target_cell_ids)
        _, adjacency = self._adjacency_for_model(X, own_proto, model)
        if not adjacency:
            return 0.0

        values_by_proto = {}
        support = {}
        for pid in np.unique(own_proto):
            rows = np.flatnonzero(own_proto == pid)
            pid_int = int(pid)
            values_by_proto[pid_int] = Y_scaled[rows]
            support[pid_int] = int(rows.size)

        loss, _ = self._loss_for_components(values_by_proto, support, adjacency)
        return float(loss)

    def _target_cells_for_values(self, Y_scaled: np.ndarray) -> np.ndarray:
        """Build target cells for a concrete scaled target matrix."""
        n_cells = self._resolve_n_target_cells(Y_scaled.shape[0])
        if self.target_cover_ == "quantile":
            return self._quantile_target_cells(Y_scaled[:, 0], n_cells)
        return self._kmeans_target_cells(Y_scaled, n_cells)

    def _loss_for_values_by_prototype(
        self,
        values_by_proto: Dict[int, np.ndarray],
    ) -> Tuple[float, Dict[int, float]]:
        """Compute weighted adjacency loss for a prototype-target mapping."""
        return self._loss_for_components(
            values_by_proto,
            self.prototype_support_,
            self.prototype_adjacency_normalized_,
        )

    def _loss_for_components(
        self,
        values_by_proto: Dict[int, np.ndarray],
        support: Dict[int, int],
        adjacency: Dict[Tuple[int, int], float],
    ) -> Tuple[float, Dict[int, float]]:
        """Compute weighted adjacency loss from explicit prototype components."""
        local_loss: Dict[int, float] = {}
        total_support = float(sum(support.values()))
        if total_support <= 0:
            return 0.0, local_loss

        for (p, q), alpha in adjacency.items():
            if p not in values_by_proto or q not in values_by_proto:
                continue
            distance = self._target_distance(values_by_proto[p], values_by_proto[q])
            if not np.isfinite(distance):
                raise ValueError("Target distribution distance produced a non-finite value.")
            local_loss[p] = local_loss.get(p, 0.0) + float(alpha) * float(distance)

        for pid in support:
            local_loss.setdefault(pid, 0.0)

        weighted = 0.0
        for pid, loss in local_loss.items():
            weighted += (float(support.get(pid, 0)) / total_support) * float(loss)
        return float(weighted), local_loss

    def _target_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute empirical target-distribution distance between prototypes."""
        if a.shape[0] == 0 or b.shape[0] == 0:
            return 0.0
        if self.target_distance_ == "wasserstein":
            return float(wasserstein_distance(a[:, 0], b[:, 0]))
        if self.target_distance_ == "sliced_wasserstein":
            distances = []
            for direction in self.target_directions_:
                distances.append(
                    wasserstein_distance(a @ direction, b @ direction)
                )
            return float(np.mean(distances))
        raise ValueError(f"Unsupported resolved target distance: {self.target_distance_}")

    def _aggregate_index(self, mode: str) -> float:
        """Return the requested bounded aggregate index."""
        if mode == "macro":
            if not self.prototype_index_:
                value = self._index_from_losses(self.actual_loss_, self.null_loss_)
            else:
                value = float(np.mean(list(self.prototype_index_.values())))
        elif mode == "support_weighted":
            value = self._index_from_losses(self.actual_loss_, self.null_loss_)
        else:
            raise ValueError("aggregation must be one of {'support_weighted', 'macro'}.")
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _warn_empty_input() -> None:
        """Warn when fitting receives no samples."""
        warnings.warn(
            "Received empty X/Y; leaving ContinuousOverlapIndex at its default value of 1.0.",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _warn_single_target_cell() -> None:
        """Warn when target cover cannot form multiple cells."""
        warnings.warn(
            "Received data with a single target cell; ContinuousOverlapIndex remains 1.0.",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _warn_single_prototype() -> None:
        """Warn when overlap cannot be assessed with one prototype."""
        warnings.warn(
            "Received data with a single prototype; ContinuousOverlapIndex remains 1.0.",
            RuntimeWarning,
            stacklevel=2,
        )
