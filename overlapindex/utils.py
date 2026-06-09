import numpy as np

def complement_code(X):
    """
    Return the complement-coded representation of a 2D array.

    Complement coding doubles the feature dimension by concatenating each
    feature vector with its elementwise complement ``1 - x``. Inputs are
    expected to be scaled to the unit interval.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array; got shape {X.shape}.")
    if np.any((X < 0.0) | (X > 1.0)):
        raise ValueError("Complement coding requires inputs scaled to the [0, 1] interval.")
    return np.concatenate((X, 1.0 - X), axis=1)


def top_two_indices_against_others_from_backend(model, x, classes, class_to_clusters, a):
    """
    Return top-2 cluster ids for each class-pair using the backend's optimized
    top-k API while preserving the legacy candidate-set semantics.

    This is behaviorally equivalent to top_two_indices_against_others(...), except
    it avoids materializing full scores_all when the backend can score only the
    requested candidate ids.

    Parameters
    ----------
    model : _BaseManyToOneClusteringModel-like
        Backend implementing topk(x, k, candidate_ids=...).
    x : np.ndarray
        One preprocessed sample.
    classes : iterable
        Class labels to compare against.
    class_to_clusters : mapping
        Mapping from class label to the cluster ids currently visible to the
        caller. For OverlapIndex this should usually be self.rev_map, not
        model.class_to_clusters.
    a : Any
        Own/current class label.

    Returns
    -------
    dict
        Maps each class b != a to a tuple of top cluster ids among
        class_to_clusters[a] union class_to_clusters[b].
    """
    result = {}
    clusters_a = class_to_clusters.get(a, set())

    for b in classes:
        if b == a:
            continue

        clusters_b = class_to_clusters.get(b, set())
        candidate_ids = np.fromiter(clusters_a | clusters_b, dtype=int)

        if candidate_ids.size == 0:
            result[b] = ()
            continue

        ids, _ = model.topk(x, k=2, candidate_ids=candidate_ids)
        result[b] = tuple(int(i) for i in ids)

    return result
