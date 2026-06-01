import numpy as np
from artlib import complement_code
from collections import defaultdict
from typing import Literal, Optional, Union, Dict, Any
from overlapindex.utils import (
    top_two_indices_against_others_from_backend,
    top_two_indices_against_others
)
from overlapindex.clustering import (
    _BaseManyToOneClusteringModel,
    _ARTMAPManyToOne,
    _KMeansManyToOne,
    _MiniBatchKMeansManyToOne,
)

# ----------------------------
# OverlapIndex with model_type
# ----------------------------

class OverlapIndex:
    def __init__(
        self,
        rho: float = 0.9,
        r_hat: float = np.inf,
        model_type: Literal["Fuzzy", "Hypersphere", "KMeans", "MiniBatchKMeans"] = "Fuzzy",
        match_tracking: str = "MT+",
        # centroid backend options:
        kmeans_k: Union[int, Dict[Any, int]] = 8,
        kmeans_kwargs: Optional[dict] = None,
        offline_chunk_size: Optional[int] = 10_000,
    ):
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
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @property
    def _is_artmap_backend(self):
        return self.model_type in ["Fuzzy", "Hypersphere"]

    @property
    def _is_offline_backend(self):
        return not self._is_artmap_backend

    # ---- preprocessing ----

    def _prep_X(self, X):
        return complement_code(X)

    def _reset_indices(self):
        self.sparse_adj = defaultdict(lambda: 0)
        self.cluster_cardinality = defaultdict(int)
        self.rev_map = defaultdict(set)
        self.pairwise_index = defaultdict(lambda: 1.0)
        self.singleton_index = defaultdict(lambda: 1.0)
        self.index = 1.0

    # ---- compatibility accessors (optional) ----

    @property
    def module_a(self):
        if self._is_artmap_backend:
            return self._model.model.module_a
        raise AttributeError("module_a is only available for ARTMAP backends.")

    @property
    def map(self):
        if self._is_artmap_backend:
            return self._model.model.map
        return None

    # ---- BMU helpers ----

    def get_top2_bmu(self, x):
        """
        Return (bmu1, bmu2) as global cluster ids for a single *preprocessed* sample.
        """
        ids, _ = self._model.topk(x, k=2)
        if ids.size == 0:
            return None, None
        b1 = int(ids[0])
        b2 = int(ids[1]) if ids.size > 1 else None
        return b1, b2

    def predict_subset_pairs(self, x, y):
        classes = list(self.rev_map.keys())
        return top_two_indices_against_others_from_backend(
            self._model,
            x,
            classes,
            self.rev_map,
            y,
        )
    # def predict_subset_pairs(self, x, y):
    #     """
    #     Legacy-compatible subset-pair prediction.
    #
    #     This intentionally uses self.rev_map, not backend.class_to_clusters,
    #     because add_batch builds rev_map incrementally and the historical
    #     OverlapIndex behavior depends on that replay state.
    #     """
    #     scores = self._model.scores_all(x)
    #     classes = list(self.rev_map.keys())
    #     return top_two_indices_against_others(scores, classes, self.rev_map, y)
    # ---- incremental (ARTMAP only) ----

    def add_sample(self, x, y):
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

    def add_batch(self, X, Y):
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

    # ---- offline (centroid backends primary; also works for ARTMAP if you want) ----

    def _fit_offline_centroid_optimized(self, X_prep, Y, classes):
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

    def _fit_offline_replay(self, X_prep, Y, classes):
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

    def fit_offline(self, X, Y, reset_state: bool = True):
        """
        Offline fit for centroid backends and one-shot batch ARTMAP.

        Centroid backends use a chunked vectorized class-pair scoring path. Other
        backends fall back to replaying samples with backend top-k hooks.

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

        if len(classes) <= 1:
            return self.index

        if self._is_offline_backend and hasattr(self._model, "_scores_matrix"):
            return self._fit_offline_centroid_optimized(X_prep, Y, classes)

        return self._fit_offline_replay(X_prep, Y, classes)
