import numpy as np
from scipy import sparse
from numbers import Integral, Real


def _validate_positive_integer(value, name):
    """Return a strictly positive integer without lossy coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_finite_positive_real(value, name):
    """Return a finite positive float without accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number.")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def _ordered_unique_1d(values):
    """Return first-observed unique values without requiring them to sort."""
    result = []
    seen = set()
    for value in np.asarray(values, dtype=object).reshape(-1):
        try:
            is_new = value not in seen
        except TypeError as exc:
            raise TypeError("Labels must be hashable.") from exc
        if is_new:
            seen.add(value)
            result.append(value)
    return np.asarray(result, dtype=object)


def _validate_class_dictionary_coverage(value, labels, name):
    """Reject class-specific dictionaries that omit observed labels."""
    if not isinstance(value, dict):
        return
    missing = [label for label in labels if label not in value]
    if not missing:
        return
    if len(missing) == 1:
        raise ValueError(
            f"Missing class-specific {name} for class {missing[0]!r}."
        )
    raise ValueError(
        f"Missing class-specific {name} for classes {tuple(missing)!r}."
    )


def _validate_feature_matrix(X):
    """Return a finite 2D dense array or floating-point CSR matrix."""
    if sparse.issparse(X):
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array; got shape {X.shape}.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature column.")
        X_csr = sparse.csr_matrix(X, dtype=float, copy=False)
        if not np.all(np.isfinite(X_csr.data)):
            raise ValueError("X contains NaN or infinite values.")
        return X_csr

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be a 2D array; got shape {X_arr.shape}.")
    if X_arr.shape[1] == 0:
        raise ValueError("X must contain at least one feature column.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")
    return X_arr


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
