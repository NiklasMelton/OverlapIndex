import numpy as np

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


def top_two_indices_against_others_from_backend(model, x, classes, a):
    """
    Return top-2 cluster ids for each class-pair using the backend's optimized
    pairwise top-k API instead of requiring a full scores_all vector.

    Parameters
    ----------
    model : _BaseManyToOneClusteringModel-like
        Backend implementing top2_for_class_pair(x, own_class, other_class).
    x : np.ndarray
        One preprocessed sample.
    classes : iterable
        Class labels to compare against.
    a : Any
        Own/current class label.

    Returns
    -------
    dict
        Maps each class b != a to a tuple of top cluster ids among clusters(a) union clusters(b).
    """
    result = {}
    for b in classes:
        if b == a:
            continue
        ids, _ = model.top2_for_class_pair(x, own_class=a, other_class=b)
        result[b] = tuple(int(i) for i in ids)
    return result


def top_two_indices_against_others(T, classes, class_to_clusters, a):
    T = np.asarray(T)
    result = {}

    clusters_a = class_to_clusters.get(a, set())

    for b in classes:
        if b == a:
            continue

        clusters_b = class_to_clusters.get(b, set())
        cluster_indices = np.fromiter(clusters_a | clusters_b, dtype=int)

        if len(cluster_indices) == 0:
            top2 = ()
        elif len(cluster_indices) == 1:
            top2 = (int(cluster_indices[0]),)
        else:
            values = T[cluster_indices]
            top2_rel = np.argpartition(values, -2)[-2:]
            top2_sorted = top2_rel[np.argsort(values[top2_rel])[::-1]]
            top2 = tuple(int(cluster_indices[i]) for i in top2_sorted)

        result[b] = top2

    return result
