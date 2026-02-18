import numpy as np
from artlib import HypersphereARTMAP, FuzzyARTMAP, complement_code
from collections import defaultdict
from sklearn.cluster import KMeans
from typing import Literal, Optional, Union, Dict, Any

class GrowingArray1D:
    def __init__(self, dtype=int):
        self.array = np.zeros(0, dtype=dtype)

    def _ensure_size(self, i):
        if i >= self.array.size:
            new_size = i + 1
            new_array = np.zeros(new_size, dtype=self.array.dtype)
            new_array[:self.array.size] = self.array
            self.array = new_array

    def __getitem__(self, i):
        self._ensure_size(i)
        return self.array[i]

    def __setitem__(self, i, value):
        self._ensure_size(i)
        self.array[i] = value

    def __iadd__(self, idx_value):
        i, value = idx_value
        self._ensure_size(i)
        self.array[i] += value
        return self

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return repr(self.array)

    def asarray(self):
        return self.array.copy()

    def __iter__(self):
        # iterate over the *current* contents only
        for v in self.array:
            yield v


def top_two_indices_against_others(T, classes, class_to_clusters, a):
    T = np.asarray(T)
    result = {}

    clusters_a = class_to_clusters.get(a, set())

    for b in classes:
        if b == a:
            continue

        clusters_b = class_to_clusters.get(b, set())
        cluster_indices = list(clusters_a | clusters_b)

        if len(cluster_indices) == 0:
            top2 = ()
        elif len(cluster_indices) == 1:
            top2 = (cluster_indices[0],)
        else:
            values = T[cluster_indices]
            top2_rel = np.argpartition(values, -2)[-2:]
            top2_sorted = top2_rel[np.argsort(values[top2_rel])[::-1]]
            top2 = tuple(cluster_indices[i] for i in top2_sorted)

        result[b] = top2

    return result


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


# ----------------------------
# OverlapIndex with model_type
# ----------------------------

class OverlapIndex:
    def __init__(
        self,
        rho: float = 0.9,
        r_hat: float = np.inf,
        model_type: Literal["Fuzzy", "Hypersphere", "KMeans"] = "Fuzzy",
        match_tracking: str = "MT+",
        # KMeans options:
        kmeans_k: Union[int, Dict[Any, int]] = 8,
        kmeans_kwargs: Optional[dict] = None,
    ):
        self.model_type = model_type
        self.match_tracking = match_tracking

        # indices / bookkeeping
        self.sparse_adj = defaultdict(lambda: 0)
        self.cluster_cardinality = GrowingArray1D()
        self.rev_map = defaultdict(set)
        self.pairwise_index = defaultdict(lambda: 1.0)
        self.singleton_index = defaultdict(lambda: 1.0)
        self.index = 1.0

        # swappable backend
        if model_type in ["Fuzzy", "Hypersphere"]:
            self._model: _BaseManyToOneClusteringModel = _ARTMAPManyToOne(
                model_type=model_type, rho=rho, r_hat=r_hat
            )
        else:
            self._model = _KMeansManyToOne(k=kmeans_k, kmeans_kwargs=kmeans_kwargs)

    # ---- preprocessing ----

    def _prep_X(self, X):
        return complement_code(X)

    def _reset_indices(self):
        self.sparse_adj = defaultdict(lambda: 0)
        self.cluster_cardinality = GrowingArray1D()
        self.rev_map = defaultdict(set)
        self.pairwise_index = defaultdict(lambda: 1.0)
        self.singleton_index = defaultdict(lambda: 1.0)
        self.index = 1.0

    # ---- compatibility accessors (optional) ----

    @property
    def module_a(self):
        if self.model_type in ["Fuzzy", "Hypersphere"]:
            return self._model.model.module_a
        raise AttributeError("module_a is only available for ARTMAP backends.")

    @property
    def map(self):
        if self.model_type in ["Fuzzy", "Hypersphere"]:
            return self._model.model.map
        return None

    # ---- BMU helpers ----

    def get_top2_bmu(self, x):
        """
        Return (bmu1, bmu2) as global cluster ids for a single *preprocessed* sample.
        """
        scores = self._model.scores_all(x)
        if scores.size == 0:
            return None, None
        order = np.argsort(scores)[::-1]  # higher is better
        b1 = int(order[0])
        b2 = int(order[1]) if len(order) > 1 else None
        return b1, b2

    def predict_subset_pairs(self, x, y):
        """
        Keep your existing top_two_indices_against_others(...) flow:
        - scores over all clusters
        - rev_map: class -> set(cluster_ids)
        """
        scores = self._model.scores_all(x)
        classes = list(self.rev_map.keys())
        return top_two_indices_against_others(scores, classes, self.rev_map, y)

    # ---- incremental (ARTMAP only) ----

    def add_sample(self, x, y):
        if self.model_type == "KMeans":
            raise NotImplementedError(
                "KMeans backend is offline-only here. Use fit_offline(X, Y)."
            )

        x_prep = self._prep_X([x])
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

    def add_batch(self, X, Y):
        if self.model_type == "KMeans":
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

    # ---- offline (KMeans primary; also works for ARTMAP if you want) ----

    def fit_offline(self, X, Y, reset_state: bool = True):
        """
        Offline fit for KMeans (and permitted for ARTMAP as one-shot batch partial_fit).
        Computes overlap index by replaying samples with the same equations.

        Returns self.index
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

        # Replay to compute overlap stats
        for x, y in zip(X_prep, Y):
            bmu1 = int(self._model.bmu_for_class(x, y))
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
