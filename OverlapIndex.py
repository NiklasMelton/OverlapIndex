import numpy as np
from artlib import HypersphereARTMAP, FuzzyARTMAP, complement_code
from typing import Literal
from collections import defaultdict


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


class OverlapIndex:
    def __init__(
            self,
            rho: float = 0.9,
            r_hat: float = np.inf,
            ART: Literal["Fuzzy", "Hypersphere"] = "Hypersphere",
            match_tracking="MT+",
    ):
        assert ART in ["Fuzzy", "Hypersphere"]
        if ART == "Fuzzy":
            self.ARTMAP = FuzzyARTMAP(rho=rho, alpha=1e-10, beta=1.0)
        else:
            self.ARTMAP = HypersphereARTMAP(rho=rho, alpha=1e-10, beta=1.0, r_hat=r_hat)
        self.ART = ART
        self.sparse_adj = defaultdict(lambda: 0)
        self.cluster_cardinality = GrowingArray1D()
        self.rev_map = defaultdict(set)
        self.pairwise_index = defaultdict(lambda: 1.0)
        self.singleton_index = defaultdict(lambda: 1.0)
        self.index = 1.0
        self.match_tracking = match_tracking

    @property
    def module_a(self):
        return self.ARTMAP.module_a

    @property
    def map(self):
        return self.ARTMAP.map

    def predict_subset_pairs(self, x, y):
        assert len(self.module_a.W) >= 0, "ART module is not fit."
        T, _ = zip(*[
            self.module_a.category_choice(x, w, params=self.module_a.params)
            for w in self.module_a.W
        ])
        classes = list(self.rev_map.keys())
        top2bmu = top_two_indices_against_others(T, classes, self.rev_map, y)
        return top2bmu

    def add_sample(self, x, y):
        x_prep = complement_code([x])
        self.ARTMAP = self.ARTMAP.partial_fit(x_prep, [y],
                                              match_tracking=self.match_tracking)
        bmu1 = self.ARTMAP.module_a.labels_[-1]
        self.rev_map[y].add(bmu1)

        self.cluster_cardinality[y] += 1
        top2bmu = self.predict_subset_pairs(x_prep, y)

        if y not in self.singleton_index:
            self.singleton_index[y] = 1.0
        for b in self.rev_map.keys():
            bmu2 = int(bmu1)
            if b != y:
                if len(top2bmu[b]) > 1:
                    bmu2_, bmu3_ = top2bmu[b]
                    if bmu2_ == bmu1:
                        bmu2 = bmu3_
                    else:
                        bmu2 = bmu2_
                if bmu2 in self.rev_map[b]:
                    self.sparse_adj[(y, b)] += 1

                self.pairwise_index[(y, b)] = 1. - (
                        float(self.sparse_adj[(y, b)]) /
                        float(self.cluster_cardinality[y])
                )
        if len(self.rev_map) > 1:
            self.singleton_index[y] = min(
                [self.pairwise_index[(y, b)] for b in self.rev_map.keys() if b != y]
            )
            self.index = np.mean(list(self.singleton_index.values()))
        return self.index

    def add_batch(self, X, Y):
        X_prep = complement_code(X)
        self.ARTMAP = self.ARTMAP.partial_fit(X_prep, Y,
                                              match_tracking=self.match_tracking)
        BMU1 = self.ARTMAP.module_a.labels_[-len(Y):]
        for x, y, bmu1 in zip(X_prep, Y, BMU1):
            self.rev_map[y].add(bmu1)
            if y not in self.singleton_index:
                self.singleton_index[y] = 1.0

            self.cluster_cardinality[y] += 1
            top2bmu = self.predict_subset_pairs(x, y)  # eq 1 & 2

            for b in self.rev_map.keys():
                bmu2 = int(bmu1)
                if b != y:
                    if len(top2bmu[b]) > 1:
                        bmu2_, bmu3_ = top2bmu[b]
                        if bmu2_ == bmu1:
                            bmu2 = bmu3_
                        else:
                            bmu2 = bmu2_
                    if bmu2 in self.rev_map[b]:
                        self.sparse_adj[(y, b)] += 1  # eq 3
                    self.pairwise_index[(y, b)] = 1. - (
                            float(self.sparse_adj[(y, b)]) /
                            float(self.cluster_cardinality[y])
                    )  # eq 4
        unique_y = np.unique(Y)
        if len(self.rev_map) > 1:
            for y in unique_y:
                self.singleton_index[y] = min(
                    [self.pairwise_index[(y, b)] for b in self.rev_map.keys() if b != y]
                )  # eq 5
            self.index = np.mean(list(self.singleton_index.values()))  # eq 6
        return self.index
