import numpy as np
from artlib import HypersphereARTMAP, FuzzyARTMAP
from collections import defaultdict
from sklearn.cluster import KMeans
from typing import Literal, Optional, Union, Dict, Any

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
        raise NotImplementedError

    def partial_fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    def bmu_for_class(self, x: np.ndarray, y: Any) -> int:
        """Return BMU1 restricted to clusters owned by class y (global cluster id)."""
        raise NotImplementedError

    def scores_all(self, x: np.ndarray) -> np.ndarray:
        """Return a score per global cluster (higher is better)."""
        raise NotImplementedError

    @property
    def class_to_clusters(self) -> Dict[Any, set]:
        raise NotImplementedError

    @property
    def n_clusters_total(self) -> int:
        raise NotImplementedError


class _KMeansManyToOne(_BaseManyToOneClusteringModel):
    """
    Offline: fit one KMeans per class.
    Global cluster ids are assigned by concatenating centers in class order.
    scores_all(x) = -euclidean_distance_to_center.
    """
    def __init__(
        self,
        k: Union[int, Dict[Any, int]] = 8,
        kmeans_kwargs: Optional[dict] = None,
    ):
        if KMeans is None:
            raise ImportError("scikit-learn is required for model_type='KMeans'.")
        self._k = k
        self._kmeans_kwargs = kmeans_kwargs or {}

        self._models: Dict[Any, KMeans] = {}
        self._centers: Optional[np.ndarray] = None
        self._class_center_ids: Dict[Any, list] = {}
        self._class_to_clusters: Dict[Any, set] = defaultdict(set)

    def fit_offline(self, X: np.ndarray, Y: np.ndarray) -> None:
        Y = np.asarray(Y)
        classes = np.unique(Y)

        centers_list = []
        self._models = {}
        self._class_center_ids = {}
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

            km = KMeans(
                n_clusters=k,
                **({"n_init": "auto"} if "n_init" not in self._kmeans_kwargs else {}),
                **self._kmeans_kwargs,
            )
            km.fit(Xc)
            self._models[c] = km

            c_centers = np.asarray(km.cluster_centers_, dtype=float)
            centers_list.append(c_centers)

            ids = list(range(gid, gid + c_centers.shape[0]))
            self._class_center_ids[c] = ids
            self._class_to_clusters[c].update(ids)
            gid += c_centers.shape[0]

        self._centers = (
            np.vstack(centers_list) if len(centers_list) else np.zeros((0, X.shape[1]))
        )

    def partial_fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> None:
        raise NotImplementedError("KMeans backend is offline-only in this adapter.")

    def bmu_for_class(self, x: np.ndarray, y: Any) -> int:
        ids = self._class_center_ids.get(y, [])
        if len(ids) == 0:
            raise ValueError(f"No clusters found for class {y}. Did you fit_offline?")
        d = np.linalg.norm(self._centers[ids] - x, axis=1)
        return int(ids[int(np.argmin(d))])

    def scores_all(self, x: np.ndarray) -> np.ndarray:
        if self._centers is None:
            raise AssertionError("KMeans backend not fit.")
        if self._centers.shape[0] == 0:
            return np.asarray([], dtype=float)
        d = np.linalg.norm(self._centers - x, axis=1)
        return -d  # higher is better

    @property
    def class_to_clusters(self) -> Dict[Any, set]:
        return self._class_to_clusters

    @property
    def n_clusters_total(self) -> int:
        return 0 if self._centers is None else int(self._centers.shape[0])


class _ARTMAPManyToOne(_BaseManyToOneClusteringModel):
    """
    Adapter around your existing FuzzyARTMAP / HypersphereARTMAP.
    Uses the ARTMAP module_a prototypes as global clusters (indices into module_a.W).
    """
    def __init__(self, model_type: Literal["Fuzzy", "Hypersphere"], rho: float, r_hat: float, alpha=1e-10, beta=1.0):
        if model_type == "Fuzzy":
            self._model = FuzzyARTMAP(rho=rho, alpha=alpha, beta=beta)
        else:
            self._model = HypersphereARTMAP(rho=rho, alpha=alpha, beta=beta, r_hat=r_hat)

        self._class_to_clusters: Dict[Any, set] = defaultdict(set)

    def fit_offline(self, X: np.ndarray, Y: np.ndarray) -> None:
        # "offline" for ARTMAP is just a single partial_fit on the batch.
        self.partial_fit(X, Y)

    def partial_fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> None:
        self._model = self._model.partial_fit(X, Y, **kwargs)
        # Update mapping from class -> cluster ids encountered in this update.
        # Note: this assumes labels_ corresponds to module_a cluster assignments.
        # If your library semantics differ, adjust here.
        new_labels = self._model.module_a.labels_[-len(Y):]
        for y, bmu in zip(Y, new_labels):
            self._class_to_clusters[y].add(int(bmu))

    def bmu_for_class(self, x: np.ndarray, y: Any) -> int:
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
        W = self._model.module_a.W
        if len(W) == 0:
            return np.asarray([], dtype=float)
        T, _ = zip(*[
            self._model.module_a.category_choice(x, w, params=self._model.module_a.params)
            for w in W
        ])
        return np.asarray(T, dtype=float)

    @property
    def class_to_clusters(self) -> Dict[Any, set]:
        return self._class_to_clusters

    @property
    def n_clusters_total(self) -> int:
        return len(self._model.module_a.W)

    @property
    def model(self):
        return self._model  # expose if needed (e.g., for module_a/map parity)

