import numpy as np
from artlib import complement_code
from collections import defaultdict
from typing import Literal, Optional, Union, Dict, Any
from overlapindex.utils import (
    top_two_indices_against_others,
    GrowingArray1D,
)
from overlapindex.clustering import (
    _BaseManyToOneClusteringModel,
    _ARTMAPManyToOne,
    _KMeansManyToOne
)

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
