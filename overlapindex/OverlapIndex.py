import numpy as np
from collections import defaultdict
from typing import Literal, Optional, Union, Dict, Any, Tuple
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

# ----------------------------
# OverlapIndex with model_type
# ----------------------------

class OverlapIndex:
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
        """
        self.model_type = model_type
        self.match_tracking = match_tracking
        self.offline_chunk_size = offline_chunk_size

        # indices / bookkeeping
        self.sparse_adj = defaultdict(lambda: 0)
        self.cluster_cardinality = defaultdict(int)
        self.rev_map = defaultdict(set)
        self.pairwise_index = defaultdict(lambda: 1.0)
        self.singleton_index = defaultdict(lambda: 1.0)
        self.index = 1.0

        # swappable backend
        if model_type in ["Fuzzy", "Hypersphere"]:
            self._model: _BaseManyToOneClusteringModel = _ARTMAPManyToOne(
                model_type=model_type, rho=rho, r_hat=r_hat
            )
        elif model_type == "KMeans":
            self._model = _KMeansManyToOne(k=kmeans_k, kmeans_kwargs=kmeans_kwargs)
        elif model_type == "MiniBatchKMeans":
            self._model = _MiniBatchKMeansManyToOne(k=kmeans_k, kmeans_kwargs=kmeans_kwargs)
        elif model_type == "BallCover":
            kwargs = ballcover_kwargs or {}
            self._model = _BallCoverManyToOne(
                k=ballcover_k,
                radius=ballcover_radius,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

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
        return complement_code(X)

    def _reset_indices(self) -> None:
        """Reset overlap-index bookkeeping without replacing the clustering backend."""
        self.sparse_adj = defaultdict(lambda: 0)
        self.cluster_cardinality = defaultdict(int)
        self.rev_map = defaultdict(set)
        self.pairwise_index = defaultdict(lambda: 1.0)
        self.singleton_index = defaultdict(lambda: 1.0)
        self.index = 1.0

    # ---- compatibility accessors (optional) ----

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
            self.index = float(np.mean(list(self.singleton_index.values())))
        return self.index

    def add_batch(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Add a labeled batch and update the overlap index.

        ARTMAP backends perform a batch partial-fit followed by historical replay.
        Offline centroid backends delegate to fit_offline with reset_state=True.
        """
        if self._is_offline_backend:
            # For consistency with your original API, treat add_batch as offline-fit+score.
            return self.fit_offline(X, Y, reset_state=True)

        X_prep = self._prep_X(X)
        self._model.partial_fit(
            X_prep, Y, match_tracking=self.match_tracking
        )

        BMU1 = self._model.model.module_a.labels_[-len(Y):]
        for x, y, bmu1 in zip(X_prep, Y, BMU1):
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

        unique_y = np.unique(Y)
        if len(self.rev_map) > 1:
            for y in unique_y:
                self.singleton_index[y] = min(
                    [self.pairwise_index[(y, b)] for b in self.rev_map.keys() if b != y]
                )
            self.index = float(np.mean(list(self.singleton_index.values())))
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

    # ---- offline (centroid backends primary; also works for ARTMAP if you want) ----

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
            self.index = float(np.mean(list(self.singleton_index.values())))
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
            self.index = float(np.mean(list(self.singleton_index.values())))
        return self.index

    def fit_offline(self, X: np.ndarray, Y: np.ndarray, reset_state: bool = True) -> float:
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

        X_prep = self._prep_X(X)
        Y = np.asarray(Y)
        classes = np.unique(Y)

        # Fit backend and sync rev_map
        self._model.fit_offline(X_prep, Y)
        self.rev_map = defaultdict(set, {c: set(s) for c, s in self._model.class_to_clusters.items()})

        # Cardinalities are per-class sample counts (eq 4 denominator)
        for c in classes:
            self.cluster_cardinality[c] += int(np.sum(Y == c))
            if c not in self.singleton_index:
                self.singleton_index[c] = 1.0

        if len(classes) <= 1:
            return self.index

        if self._is_offline_backend and hasattr(self._model, "_scores_matrix"):
            return self._fit_offline_centroid_optimized(X_prep, Y, classes)

        return self._fit_offline_replay(X_prep, Y, classes)
