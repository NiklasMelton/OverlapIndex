import numpy as np
from artlib import HypersphereARTMAP, FuzzyARTMAP
from collections import defaultdict
from sklearn.cluster import KMeans, MiniBatchKMeans
from overlapindex.BallCover import BallCoverManyToOne
from typing import Literal, Optional, Union, Dict, Any, Sequence, Tuple, Type

# ----------------------------
# Swappable backend interface
# ----------------------------

class _BaseManyToOneClusteringModel:
    """
    A small adapter interface that makes:
      - ARTMAP-style incremental models
      - offline per-class KMeans
    swappable for OverlapIndex.

    Conventions:
      - X passed in should already be preprocessed (e.g., complement-coded).
      - cluster ids returned by this backend are "global ids" (ints) consistent across classes.
      - class_to_clusters maps class_label -> set(global_cluster_ids).
    """
    def fit_offline(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the backend on a complete labeled dataset.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed input samples.
        Y : np.ndarray
            Class labels aligned with X.
        """
        raise NotImplementedError

    def partial_fit(self, X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> None:
        """
        Incrementally fit the backend on a labeled batch.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed input samples.
        Y : np.ndarray
            Class labels aligned with X.
        **kwargs : Any
            Backend-specific keyword arguments.
        """
        raise NotImplementedError

    def bmu_for_class(self, x: np.ndarray, y: Any) -> int:
        """
        Return the best matching global cluster id restricted to one class.

        Parameters
        ----------
        x : np.ndarray
            One preprocessed sample.
        y : Any
            Class label whose clusters define the candidate set.
        """
        raise NotImplementedError

    def scores_all(self, x: np.ndarray) -> np.ndarray:
        """
        Return one score per global cluster for a preprocessed sample.

        Higher scores indicate better matches.
        """
        raise NotImplementedError

    def topk(
        self,
        x: np.ndarray,
        k: int,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the top-k global cluster ids and scores for one sample.

        Parameters
        ----------
        x : np.ndarray
            One preprocessed sample.
        k : int
            Number of clusters to return.
        candidate_ids : sequence of int, optional
            Optional global cluster ids to restrict the search.

        Returns
        -------
        tuple of np.ndarray
            Arrays of global cluster ids and corresponding scores, sorted from
            highest to lowest score.
        """
        scores = self.scores_all(x)
        if scores.size == 0 or k <= 0:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)

        if candidate_ids is None:
            ids = np.arange(scores.size, dtype=int)
            values = scores
        else:
            ids = np.asarray(candidate_ids, dtype=int)
            if ids.size == 0:
                return np.asarray([], dtype=int), np.asarray([], dtype=float)
            values = scores[ids]

        k_eff = int(min(k, values.size))
        if k_eff <= 0:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)

        rel = np.argpartition(values, -k_eff)[-k_eff:]
        rel = rel[np.argsort(values[rel])[::-1]]
        return ids[rel].astype(int, copy=False), values[rel].astype(float, copy=False)

    def bmu_for_class_batch(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Return class-restricted BMUs for a batch of samples.

        The default implementation calls bmu_for_class for each sample.
        """
        return np.asarray([self.bmu_for_class(x, y) for x, y in zip(X, Y)], dtype=int)

    def top2_for_class_pair(
        self,
        x: np.ndarray,
        own_class: Any,
        other_class: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return top-2 cluster ids and scores among two classes' clusters.

        Parameters
        ----------
        x : np.ndarray
            One preprocessed sample.
        own_class : Any
            First class label contributing candidate clusters.
        other_class : Any
            Second class label contributing candidate clusters.
        """
        candidate_ids = list(
            self.class_to_clusters.get(own_class, set())
            | self.class_to_clusters.get(other_class, set())
        )
        return self.topk(x, k=2, candidate_ids=candidate_ids)

    @property
    def class_to_clusters(self) -> Dict[Any, set]:
        """Return a mapping from class labels to owned global cluster ids."""
        raise NotImplementedError

    @property
    def n_clusters_total(self) -> int:
        """Return the total number of global clusters currently known to the backend."""
        raise NotImplementedError




# ---- Centroid/prototype backends ----

class _BaseCentroidManyToOne(_BaseManyToOneClusteringModel):
    """
    Shared implementation for offline centroid/prototype backends.

    Subclasses provide _make_model(...). After fitting, all centers are stored as one
    global center matrix. Scores are negative squared Euclidean distances, so higher
    remains better and no square root is needed for ranking.
    """
    def __init__(
        self,
        k: Union[int, Dict[Any, int]] = 8,
        model_kwargs: Optional[dict] = None,
        dtype: Type[np.floating] = np.float32,
    ) -> None:
        """
        Initialize shared centroid-backend state.

        Parameters
        ----------
        k : int or dict, default=8
            Number of clusters per class, or class-specific cluster counts.
        model_kwargs : dict, optional
            Keyword arguments forwarded to the concrete clustering estimator.
        dtype : numpy floating dtype, default=np.float32
            Floating-point dtype used to store centroid arrays.
        """
        self._k = k
        self._model_kwargs = model_kwargs or {}
        self._dtype = dtype

        self._models: Dict[Any, Any] = {}
        self._centers: Optional[np.ndarray] = None
        self._center_norms: Optional[np.ndarray] = None
        self._class_center_ids: Dict[Any, list] = {}
        self._class_center_id_arrays: Dict[Any, np.ndarray] = {}
        self._class_to_clusters: Dict[Any, set] = defaultdict(set)
        self._cluster_to_class: Optional[np.ndarray] = None

    def _make_model(self, n_clusters: int) -> Any:
        """Create a concrete centroid estimator with the requested cluster count."""
        raise NotImplementedError

    def fit_offline(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit one centroid estimator per class and concatenate their centers.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed input samples.
        Y : np.ndarray
            Class labels aligned with X.
        """
        Y = np.asarray(Y)
        classes = np.unique(Y)

        centers_list = []
        cluster_classes = []
        self._models = {}
        self._class_center_ids = {}
        self._class_center_id_arrays = {}
        self._class_to_clusters = defaultdict(set)

        gid = 0
        for c in classes:
            idx = np.where(Y == c)[0]
            Xc = X[idx]
            nc = Xc.shape[0]
            if nc == 0:
                continue

            k = self._k[c] if isinstance(self._k, dict) and c in self._k else self._k
            k = int(max(1, min(int(k), int(nc))))

            model = self._make_model(k)
            model.fit(Xc)
            self._models[c] = model

            c_centers = np.asarray(model.cluster_centers_, dtype=self._dtype)
            centers_list.append(c_centers)

            ids = list(range(gid, gid + c_centers.shape[0]))
            ids_array = np.asarray(ids, dtype=int)
            self._class_center_ids[c] = ids
            self._class_center_id_arrays[c] = ids_array
            self._class_to_clusters[c].update(ids)
            cluster_classes.extend([c] * c_centers.shape[0])
            gid += c_centers.shape[0]

        self._centers = (
            np.vstack(centers_list).astype(self._dtype, copy=False)
            if len(centers_list)
            else np.zeros((0, X.shape[1]), dtype=self._dtype)
        )
        self._center_norms = np.einsum("ij,ij->i", self._centers, self._centers)
        self._cluster_to_class = np.asarray(cluster_classes, dtype=object)

    def partial_fit(self, X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> None:
        """Raise because centroid backends in this adapter are offline-only."""
        raise NotImplementedError(f"{self.__class__.__name__} is offline-only in this adapter.")

    def _check_fit(self) -> None:
        """Raise if centroid arrays have not been initialized by fit_offline."""
        if self._centers is None or self._center_norms is None:
            raise AssertionError(f"{self.__class__.__name__} backend not fit.")

    def _scores_for_ids(self, x: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Return negative squared distances from one sample to selected centers."""
        self._check_fit()
        ids = np.asarray(ids, dtype=int)
        if ids.size == 0:
            return np.asarray([], dtype=float)
        x = np.asarray(x, dtype=self._centers.dtype)
        x_norm = float(np.dot(x, x))
        d2 = self._center_norms[ids] + x_norm - 2.0 * (self._centers[ids] @ x)
        return -d2

    def bmu_for_class(self, x: np.ndarray, y: Any) -> int:
        """Return the nearest global centroid owned by class y."""
        ids = self._class_center_id_arrays.get(y)
        if ids is None or ids.size == 0:
            raise ValueError(f"No clusters found for class {y}. Did you fit_offline?")
        scores = self._scores_for_ids(x, ids)
        return int(ids[int(np.argmax(scores))])

    def bmu_for_class_batch(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return nearest class-owned centroids for a batch of samples."""
        X = np.asarray(X, dtype=self._centers.dtype)
        Y = np.asarray(Y)
        result = np.empty(X.shape[0], dtype=int)
        for c in np.unique(Y):
            row_idx = np.where(Y == c)[0]
            ids = self._class_center_id_arrays.get(c)
            if ids is None or ids.size == 0:
                raise ValueError(f"No clusters found for class {c}. Did you fit_offline?")
            scores = self._scores_matrix(X[row_idx], ids)
            result[row_idx] = ids[np.argmax(scores, axis=1)]
        return result

    def _scores_matrix(self, X: np.ndarray, ids: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return negative squared distances from a sample matrix to selected centers."""
        self._check_fit()
        X = np.asarray(X, dtype=self._centers.dtype)
        if ids is None:
            centers = self._centers
            center_norms = self._center_norms
        else:
            ids = np.asarray(ids, dtype=int)
            centers = self._centers[ids]
            center_norms = self._center_norms[ids]
        if centers.shape[0] == 0:
            return np.zeros((X.shape[0], 0), dtype=self._centers.dtype)
        X_norms = np.einsum("ij,ij->i", X, X)
        d2 = X_norms[:, None] + center_norms[None, :] - 2.0 * (X @ centers.T)
        return -d2

    def scores_all(self, x: np.ndarray) -> np.ndarray:
        """Return negative squared-distance scores for all global centroids."""
        self._check_fit()
        if self._centers.shape[0] == 0:
            return np.asarray([], dtype=float)
        return self._scores_for_ids(x, np.arange(self._centers.shape[0], dtype=int))

    def topk(
        self,
        x: np.ndarray,
        k: int,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-k centroid ids and scores, optionally restricted to candidate ids."""
        self._check_fit()
        if k <= 0:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)

        if candidate_ids is None:
            ids = np.arange(self._centers.shape[0], dtype=int)
        else:
            ids = np.asarray(candidate_ids, dtype=int)

        if ids.size == 0:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)

        values = self._scores_for_ids(x, ids)
        k_eff = int(min(k, values.size))
        rel = np.argpartition(values, -k_eff)[-k_eff:]
        rel = rel[np.argsort(values[rel])[::-1]]
        return ids[rel].astype(int, copy=False), values[rel].astype(float, copy=False)

    def top2_for_class_pair(
        self,
        x: np.ndarray,
        own_class: Any,
        other_class: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-2 centroid matches among the union of two class-owned centroid sets."""
        own_ids = self._class_center_id_arrays.get(own_class, np.asarray([], dtype=int))
        other_ids = self._class_center_id_arrays.get(other_class, np.asarray([], dtype=int))
        candidate_ids = np.concatenate((own_ids, other_ids))
        return self.topk(x, k=2, candidate_ids=candidate_ids)

    @property
    def centers(self) -> np.ndarray:
        """Return the global centroid matrix."""
        self._check_fit()
        return self._centers

    @property
    def class_center_id_arrays(self) -> Dict[Any, np.ndarray]:
        """Return class-to-global-centroid-id mappings as NumPy arrays."""
        return self._class_center_id_arrays

    @property
    def cluster_to_class(self) -> Optional[np.ndarray]:
        """Return an array mapping global centroid ids to class labels."""
        return self._cluster_to_class

    @property
    def class_to_clusters(self) -> Dict[Any, set]:
        """Return class-to-global-centroid-id mappings as sets."""
        return self._class_to_clusters

    @property
    def n_clusters_total(self) -> int:
        """Return the number of global centroids."""
        return 0 if self._centers is None else int(self._centers.shape[0])


class _KMeansManyToOne(_BaseCentroidManyToOne):
    """
    Offline: fit one KMeans per class.
    Global cluster ids are assigned by concatenating centers in class order.
    scores_all(x) = -squared_euclidean_distance_to_center.
    """
    def __init__(
        self,
        k: Union[int, Dict[Any, int]] = 8,
        kmeans_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize a per-class scikit-learn KMeans backend."""
        if KMeans is None:
            raise ImportError("scikit-learn is required for model_type='KMeans'.")
        super().__init__(k=k, model_kwargs=kmeans_kwargs, dtype=np.float32)

    def _make_model(self, n_clusters: int) -> KMeans:
        """Create a scikit-learn KMeans estimator."""
        return KMeans(
            n_clusters=n_clusters,
            **({"n_init": "auto"} if "n_init" not in self._model_kwargs else {}),
            **self._model_kwargs,
        )


class _MiniBatchKMeansManyToOne(_BaseCentroidManyToOne):
    """
    Offline: fit one MiniBatchKMeans per class.
    This is the preferred centroid backend for very large batch data.
    """
    def __init__(
        self,
        k: Union[int, Dict[Any, int]] = 8,
        kmeans_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize a per-class scikit-learn MiniBatchKMeans backend."""
        if MiniBatchKMeans is None:
            raise ImportError("scikit-learn is required for model_type='MiniBatchKMeans'.")
        super().__init__(k=k, model_kwargs=kmeans_kwargs, dtype=np.float32)


    def _make_model(self, n_clusters: int) -> MiniBatchKMeans:
        """Create a scikit-learn MiniBatchKMeans estimator."""
        kwargs = {
            "batch_size": 8192,
            "n_init": 1,
            "init": "random",
        }
        kwargs.update(self._model_kwargs)
        return MiniBatchKMeans(n_clusters=n_clusters, **kwargs)


# --- BallCover backend ---

class _BallCoverManyToOne(BallCoverManyToOne, _BaseManyToOneClusteringModel):
    """
    Offline: fit one greedy ball cover per class.

    This thin adapter exposes the standalone BallCoverManyToOne implementation
    through the same private backend naming convention used by the other
    clustering backends in this module.
    """
    pass


class _ARTMAPManyToOne(_BaseManyToOneClusteringModel):
    """
    Adapter around your existing FuzzyARTMAP / HypersphereARTMAP.
    Uses the ARTMAP module_a prototypes as global clusters (indices into module_a.W).
    """
    def __init__(
        self,
        model_type: Literal["Fuzzy", "Hypersphere"],
        rho: float,
        r_hat: float,
        alpha: float = 1e-10,
        beta: float = 1.0,
    ) -> None:
        """
        Initialize an ARTMAP adapter.

        Parameters
        ----------
        model_type : {"Fuzzy", "Hypersphere"}
            ARTMAP implementation to wrap.
        rho : float
            Vigilance parameter.
        r_hat : float
            Radius constraint used by HypersphereARTMAP.
        alpha : float, default=1e-10
            ARTMAP alpha parameter.
        beta : float, default=1.0
            ARTMAP learning-rate parameter.
        """
        if model_type == "Fuzzy":
            self._model = FuzzyARTMAP(rho=rho, alpha=alpha, beta=beta)
        else:
            self._model = HypersphereARTMAP(rho=rho, alpha=alpha, beta=beta, r_hat=r_hat)

        self._class_to_clusters: Dict[Any, set] = defaultdict(set)

    def fit_offline(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit ARTMAP using one batch partial-fit call."""
        # "offline" for ARTMAP is just a single partial_fit on the batch.
        self.partial_fit(X, Y)

    def partial_fit(self, X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> None:
        """Incrementally fit ARTMAP and update class-owned cluster mappings."""
        self._model = self._model.partial_fit(X, Y, **kwargs)
        # Update mapping from class -> cluster ids encountered in this update.
        # Note: this assumes labels_ corresponds to module_a cluster assignments.
        # If your library semantics differ, adjust here.
        new_labels = self._model.module_a.labels_[-len(Y):]
        for y, bmu in zip(Y, new_labels):
            self._class_to_clusters[y].add(int(bmu))

    def bmu_for_class(self, x: np.ndarray, y: Any) -> int:
        """Return the highest-scoring ARTMAP cluster currently owned by class y."""
        # For ARTMAP, "BMU for class" is the BMU chosen by the model itself when trained with (x,y).
        # In streaming use, callers should get BMU from the last label_. For batch replay we can compute
        # best matching among clusters owned by y using scores_all().
        ids = list(self._class_to_clusters.get(y, []))
        if len(ids) == 0:
            raise ValueError(f"No clusters found for class {y}. Did you fit/partial_fit?")
        scores = self.scores_all(x)
        best = ids[int(np.argmax(scores[ids]))]
        return int(best)

    def scores_all(self, x: np.ndarray) -> np.ndarray:
        """Return ARTMAP category-choice scores for all module A prototypes."""
        W = self._model.module_a.W
        if len(W) == 0:
            return np.asarray([], dtype=float)
        T, _ = zip(*[
            self._model.module_a.category_choice(x, w, params=self._model.module_a.params)
            for w in W
        ])
        return np.asarray(T, dtype=float)

    def topk(
        self,
        x: np.ndarray,
        k: int,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-k ARTMAP cluster ids and scores, optionally candidate-restricted."""
        scores = self.scores_all(x)
        if scores.size == 0 or k <= 0:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)

        if candidate_ids is None:
            ids = np.arange(scores.size, dtype=int)
            values = scores
        else:
            ids = np.asarray(candidate_ids, dtype=int)
            if ids.size == 0:
                return np.asarray([], dtype=int), np.asarray([], dtype=float)
            values = scores[ids]

        k_eff = int(min(k, values.size))
        rel = np.argpartition(values, -k_eff)[-k_eff:]
        rel = rel[np.argsort(values[rel])[::-1]]
        return ids[rel].astype(int, copy=False), values[rel].astype(float, copy=False)

    @property
    def class_to_clusters(self) -> Dict[Any, set]:
        """Return class-to-ARTMAP-cluster-id mappings."""
        return self._class_to_clusters

    @property
    def n_clusters_total(self) -> int:
        """Return the number of ARTMAP module A prototypes."""
        return len(self._model.module_a.W)

    @property
    def model(self) -> Any:
        """Return the wrapped ARTMAP model."""
        return self._model  # expose if needed (e.g., for module_a/map parity)

