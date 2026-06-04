"""Greedy ball-cover clustering backend.

This module provides a standalone many-to-one ball-cover backend intended to be
wrapped by clustering modules that expect a centroid/prototype-like interface.

The implementation is offline-only. It supports exactly one automatic structural
parameter at a time:

- ``k='auto', radius=<float>``: greedily add fixed-radius balls until the target
  cover fraction is reached.
- ``k=<int>, radius='auto'``: greedily select ``k`` landmarks, then choose the
  radius needed to cover the requested fraction of samples.

The backend is designed for dense high-dimensional arrays, such as image
embeddings. For high-dimensional data, ``metric='auto'`` resolves to cosine
geometry by internally L2-normalizing samples and using Euclidean chord distance
on the unit sphere.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Type, Union

import numpy as np

Metric = Literal["auto", "euclidean", "cosine"]
Selection = Literal["greedy_uncovered", "farthest"]
Auto = Literal["auto"]


class BallCoverManyToOne:
    """Offline greedy ball-cover backend with many-to-one class ownership.

    Parameters
    ----------
    k : int, dict, or "auto", default="auto"
        Number of balls per class, class-specific ball counts, or ``"auto"``.
        If ``k='auto'``, ``radius`` must be numeric.
    radius : float, dict, or "auto", default=0.25
        Ball radius, class-specific radii, or ``"auto"``. If ``radius='auto'``,
        ``k`` must be an integer or class-specific integer dictionary.
    metric : {"auto", "euclidean", "cosine"}, default="auto"
        Distance geometry. ``"auto"`` resolves to cosine when the feature
        dimension is at least ``high_dim_threshold`` and Euclidean otherwise.
        Cosine mode internally L2-normalizes rows and uses Euclidean chord
        distance on the unit sphere.
    high_dim_threshold : int, default=128
        Feature dimension at which ``metric='auto'`` switches to cosine.
    cover_fraction : float, default=1.0
        Target fraction of each class to cover. Values must be in ``(0, 1]``.
        For fixed-radius mode, greedy selection stops after this fraction is
        covered. For fixed-k mode, radius is chosen as this quantile of nearest
        center distances.
    selection : {"greedy_uncovered", "farthest"}, default="greedy_uncovered"
        Center-selection policy. ``"greedy_uncovered"`` chooses subsequent
        centers among currently uncovered samples. ``"farthest"`` chooses the
        farthest sample overall from the current landmark set.
    chunk_size : int, default=8192
        Number of rows processed at a time in distance computations. The current
        implementation mainly uses matrix-vector distance updates, so this is a
        safeguard for very large arrays and query matrices.
    max_balls : int, optional
        Maximum number of balls allowed per class in ``k='auto'`` mode.
    store_memberships : bool, default=False
        If true, store training-set member indices for each ball. Memberships are
        temporary otherwise and discarded after fitting.
    dtype : numpy floating dtype, default=np.float32
        Floating-point dtype used for stored centers and working arrays.
    random_state : int, optional
        Seed used only for deterministic tie-breaking, if needed.

    Notes
    -----
    Scores follow the convention "higher is better":

    ``score_j(x) = 1 - d(x, center_j)^2 / radius_j^2``.

    Thus, for a given ball, positive scores indicate that ``x`` lies inside the
    ball, zero indicates the boundary, and negative scores indicate that ``x``
    lies outside the ball.
    """

    def __init__(
        self,
        k: Union[int, Dict[Any, int], Auto] = "auto",
        radius: Union[float, Dict[Any, float], Auto] = 0.25,
        metric: Metric = "auto",
        high_dim_threshold: int = 128,
        cover_fraction: float = 1.0,
        selection: Selection = "greedy_uncovered",
        chunk_size: int = 8192,
        max_balls: Optional[int] = None,
        store_memberships: bool = False,
        dtype: Type[np.floating] = np.float32,
        random_state: Optional[int] = None,
    ) -> None:
        self.k = k
        self.radius = radius
        self.metric = metric
        self.high_dim_threshold = int(high_dim_threshold)
        self.cover_fraction = float(cover_fraction)
        self.selection = selection
        self.chunk_size = int(chunk_size)
        self.max_balls = None if max_balls is None else int(max_balls)
        self.store_memberships = bool(store_memberships)
        self.dtype = dtype
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._resolved_metric: Optional[str] = None
        self._eps = np.finfo(dtype).eps if np.issubdtype(dtype, np.floating) else np.finfo(np.float32).eps

        self._centers: Optional[np.ndarray] = None
        self._radii: Optional[np.ndarray] = None
        self._radius2: Optional[np.ndarray] = None
        self._center_norms: Optional[np.ndarray] = None
        self._class_ball_ids: Dict[Any, list] = {}
        self._class_ball_id_arrays: Dict[Any, np.ndarray] = {}
        self._class_to_clusters: Dict[Any, set] = defaultdict(set)
        self._cluster_to_class: Optional[np.ndarray] = None
        self._ball_to_points: Optional[Dict[int, np.ndarray]] = None
        self._diagnostics: Dict[str, Any] = {}
        self._class_diagnostics: Dict[Any, Dict[str, Any]] = {}

        self._validate_init_params()

    # ------------------------------------------------------------------
    # Public fitting API
    # ------------------------------------------------------------------

    def fit_offline(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit one greedy ball cover per class and concatenate global balls."""
        X = self._as_2d_float_array(X)
        Y = np.asarray(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have aligned rows; got {X.shape[0]} and {Y.shape[0]}.")

        self._resolved_metric = self._resolve_metric(X.shape[1])
        X_work = self._prepare_X(X)

        classes = np.unique(Y)
        centers_list = []
        radii_list = []
        cluster_classes = []
        self._class_ball_ids = {}
        self._class_ball_id_arrays = {}
        self._class_to_clusters = defaultdict(set)
        self._ball_to_points = {} if self.store_memberships else None
        self._class_diagnostics = {}

        gid = 0
        for c in classes:
            idx = np.where(Y == c)[0]
            Xc = X_work[idx]
            if Xc.shape[0] == 0:
                continue

            k_c = self._value_for_class(self.k, c)
            radius_c = self._value_for_class(self.radius, c)
            centers_c, radii_c, memberships_c, diag_c = self._fit_cover_for_class(Xc, k_c, radius_c)

            n_c_balls = centers_c.shape[0]
            ids = list(range(gid, gid + n_c_balls))
            ids_array = np.asarray(ids, dtype=int)

            centers_list.append(centers_c)
            radii_list.append(radii_c)
            self._class_ball_ids[c] = ids
            self._class_ball_id_arrays[c] = ids_array
            self._class_to_clusters[c].update(ids)
            cluster_classes.extend([c] * n_c_balls)

            if self.store_memberships and self._ball_to_points is not None:
                for local_j, members in enumerate(memberships_c):
                    # Convert class-local sample positions back to original dataset row indices.
                    self._ball_to_points[gid + local_j] = idx[members].astype(int, copy=False)

            diag_c = dict(diag_c)
            diag_c["class_label"] = c
            diag_c["global_ball_ids"] = ids
            self._class_diagnostics[c] = diag_c
            gid += n_c_balls

        if centers_list:
            self._centers = np.vstack(centers_list).astype(self.dtype, copy=False)
            self._radii = np.concatenate(radii_list).astype(self.dtype, copy=False)
        else:
            self._centers = np.zeros((0, X.shape[1]), dtype=self.dtype)
            self._radii = np.zeros((0,), dtype=self.dtype)

        self._radius2 = np.maximum(self._radii * self._radii, self._eps).astype(self.dtype, copy=False)
        self._center_norms = np.einsum("ij,ij->i", self._centers, self._centers).astype(self.dtype, copy=False)
        self._cluster_to_class = np.asarray(cluster_classes, dtype=object)
        self._diagnostics = self._summarize_diagnostics(X.shape, classes)

    def partial_fit(self, X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> None:
        """Raise because this backend is offline-only."""
        raise NotImplementedError(f"{self.__class__.__name__} is offline-only.")

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def bmu_for_class(self, x: np.ndarray, y: Any) -> int:
        """Return the highest-scoring ball owned by class ``y``."""
        ids = self._class_ball_id_arrays.get(y)
        if ids is None or ids.size == 0:
            raise ValueError(f"No balls found for class {y}. Did you fit_offline?")
        scores = self._scores_for_ids(x, ids)
        return int(ids[int(np.argmax(scores))])

    def bmu_for_class_batch(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return class-restricted BMUs for a batch."""
        self._check_fit()
        X = self._prepare_X(self._as_2d_float_array(X))
        Y = np.asarray(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have aligned rows; got {X.shape[0]} and {Y.shape[0]}.")
        result = np.empty(X.shape[0], dtype=int)
        for c in np.unique(Y):
            row_idx = np.where(Y == c)[0]
            ids = self._class_ball_id_arrays.get(c)
            if ids is None or ids.size == 0:
                raise ValueError(f"No balls found for class {c}. Did you fit_offline?")
            scores = self._scores_matrix_prepared(X[row_idx], ids)
            result[row_idx] = ids[np.argmax(scores, axis=1)]
        return result

    def scores_all(self, x: np.ndarray) -> np.ndarray:
        """Return one score per global ball for a sample."""
        self._check_fit()
        if self._centers.shape[0] == 0:
            return np.asarray([], dtype=float)
        ids = np.arange(self._centers.shape[0], dtype=int)
        return self._scores_for_ids(x, ids)

    def topk(
        self,
        x: np.ndarray,
        k: int,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-k ball ids and scores, optionally restricted to candidates."""
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
        """Return top-2 balls among the union of two class-owned ball sets."""
        own_ids = self._class_ball_id_arrays.get(own_class, np.asarray([], dtype=int))
        other_ids = self._class_ball_id_arrays.get(other_class, np.asarray([], dtype=int))
        return self.topk(x, k=2, candidate_ids=np.concatenate((own_ids, other_ids)))

    # ------------------------------------------------------------------
    # Core cover construction
    # ------------------------------------------------------------------

    def _fit_cover_for_class(
        self,
        Xc: np.ndarray,
        k_c: Union[int, Auto],
        radius_c: Union[float, Auto],
    ) -> Tuple[np.ndarray, np.ndarray, list[np.ndarray], Dict[str, Any]]:
        n, d = Xc.shape
        if n == 0:
            return np.zeros((0, d), dtype=self.dtype), np.zeros((0,), dtype=self.dtype), [], {}

        X_norms = np.einsum("ij,ij->i", Xc, Xc).astype(self.dtype, copy=False)
        first_idx = self._first_center_index(Xc, X_norms)
        nearest_d2 = np.full(n, np.inf, dtype=self.dtype)
        covered = np.zeros(n, dtype=bool)
        centers = []
        memberships: list[np.ndarray] = []

        fixed_radius_mode = k_c == "auto" and radius_c != "auto"
        fixed_k_mode = k_c != "auto" and radius_c == "auto"
        if not fixed_radius_mode and not fixed_k_mode:
            # Also allow fully fixed k and radius, since it is useful and unambiguous.
            if k_c == "auto" and radius_c == "auto":
                raise ValueError("Only one of k or radius may be 'auto'.")
            if k_c == "auto":
                fixed_radius_mode = True
            elif radius_c == "auto":
                fixed_k_mode = True
            else:
                fixed_k_mode = True

        if fixed_radius_mode:
            radius = float(radius_c)  # type: ignore[arg-type]
            if radius <= 0:
                raise ValueError(f"radius must be positive; got {radius}.")
            radius2 = float(radius * radius)
            next_idx = first_idx
            while float(np.mean(covered)) < self.cover_fraction:
                center = Xc[next_idx].copy()
                centers.append(center)
                d2 = self._sq_dists_to_center(Xc, X_norms, center)
                nearest_d2 = np.minimum(nearest_d2, d2)
                members = np.flatnonzero(d2 <= radius2)
                covered[members] = True
                if self.store_memberships:
                    memberships.append(members.astype(int, copy=False))

                if self.max_balls is not None and len(centers) >= self.max_balls:
                    break
                if covered.all():
                    break
                if len(centers) >= n:
                    break
                next_idx = self._next_center_index(nearest_d2, covered)

            centers_arr = np.vstack(centers).astype(self.dtype, copy=False)
            radii_arr = np.full(len(centers), radius, dtype=self.dtype)
            if not self.store_memberships:
                memberships = []
            diag = self._make_class_diag(
                n=n,
                n_balls=len(centers),
                radius=radius,
                covered=covered,
                nearest_d2=nearest_d2,
                mode="auto_k_fixed_radius",
            )
            if self.max_balls is not None and len(centers) >= self.max_balls and float(np.mean(covered)) < self.cover_fraction:
                diag["stopped_by_max_balls"] = True
            return centers_arr, radii_arr, memberships, diag

        # fixed-k mode; if radius is also fixed, choose k landmarks and keep given radius.
        k_int = int(k_c)  # type: ignore[arg-type]
        k_int = int(max(1, min(k_int, n)))
        next_idx = first_idx
        for _ in range(k_int):
            center = Xc[next_idx].copy()
            centers.append(center)
            d2 = self._sq_dists_to_center(Xc, X_norms, center)
            nearest_d2 = np.minimum(nearest_d2, d2)
            if len(centers) >= k_int:
                break
            next_idx = self._next_center_index(nearest_d2, covered=None)

        centers_arr = np.vstack(centers).astype(self.dtype, copy=False)
        if radius_c == "auto":
            # np.quantile with 1.0 returns the max; clip for numerical safety.
            q = float(np.clip(self.cover_fraction, 0.0, 1.0))
            radius = float(np.sqrt(max(np.quantile(nearest_d2, q), 0.0)))
            radius = max(radius, float(np.sqrt(self._eps)))
            mode = "fixed_k_auto_radius"
        else:
            radius = float(radius_c)  # type: ignore[arg-type]
            if radius <= 0:
                raise ValueError(f"radius must be positive; got {radius}.")
            mode = "fixed_k_fixed_radius"

        radii_arr = np.full(len(centers), radius, dtype=self.dtype)
        covered = nearest_d2 <= radius * radius
        if self.store_memberships:
            # Store ball memberships only when requested. This costs O(k * n * d).
            memberships = []
            for center in centers_arr:
                d2 = self._sq_dists_to_center(Xc, X_norms, center)
                memberships.append(np.flatnonzero(d2 <= radius * radius).astype(int, copy=False))

        diag = self._make_class_diag(
            n=n,
            n_balls=len(centers),
            radius=radius,
            covered=covered,
            nearest_d2=nearest_d2,
            mode=mode,
        )
        return centers_arr, radii_arr, memberships, diag

    def _first_center_index(self, X: np.ndarray, X_norms: np.ndarray) -> int:
        """Choose the first center as the point nearest the class mean."""
        mean = np.mean(X, axis=0).astype(self.dtype, copy=False)
        mean_norm = float(np.dot(mean, mean))
        d2 = X_norms + mean_norm - 2.0 * (X @ mean)
        return int(np.argmin(d2))

    def _next_center_index(self, nearest_d2: np.ndarray, covered: Optional[np.ndarray]) -> int:
        """Choose the next center by farthest-current-distance policy."""
        if covered is not None and self.selection == "greedy_uncovered":
            candidates = np.flatnonzero(~covered)
            if candidates.size == 0:
                return int(np.argmax(nearest_d2))
            rel = int(np.argmax(nearest_d2[candidates]))
            return int(candidates[rel])
        return int(np.argmax(nearest_d2))

    # ------------------------------------------------------------------
    # Distance and score helpers
    # ------------------------------------------------------------------

    def _sq_dists_to_center(self, X: np.ndarray, X_norms: np.ndarray, center: np.ndarray) -> np.ndarray:
        center = np.asarray(center, dtype=self.dtype)
        center_norm = float(np.dot(center, center))
        out = np.empty(X.shape[0], dtype=self.dtype)
        for start in range(0, X.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, X.shape[0])
            d2 = X_norms[start:stop] + center_norm - 2.0 * (X[start:stop] @ center)
            out[start:stop] = np.maximum(d2, 0.0)
        return out

    def _scores_for_ids(self, x: np.ndarray, ids: np.ndarray) -> np.ndarray:
        self._check_fit()
        ids = np.asarray(ids, dtype=int)
        if ids.size == 0:
            return np.asarray([], dtype=float)
        x = self._prepare_x(x)
        x_norm = float(np.dot(x, x))
        d2 = self._center_norms[ids] + x_norm - 2.0 * (self._centers[ids] @ x)
        d2 = np.maximum(d2, 0.0)
        return (1.0 - d2 / self._radius2[ids]).astype(float, copy=False)

    def _scores_matrix_prepared(self, X: np.ndarray, ids: Optional[Sequence[int]] = None) -> np.ndarray:
        self._check_fit()
        if ids is None:
            centers = self._centers
            center_norms = self._center_norms
            radius2 = self._radius2
        else:
            ids = np.asarray(ids, dtype=int)
            centers = self._centers[ids]
            center_norms = self._center_norms[ids]
            radius2 = self._radius2[ids]
        if centers.shape[0] == 0:
            return np.zeros((X.shape[0], 0), dtype=self.dtype)
        X_norms = np.einsum("ij,ij->i", X, X).astype(self.dtype, copy=False)
        scores = np.empty((X.shape[0], centers.shape[0]), dtype=self.dtype)
        for start in range(0, X.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, X.shape[0])
            d2 = X_norms[start:stop, None] + center_norms[None, :] - 2.0 * (X[start:stop] @ centers.T)
            d2 = np.maximum(d2, 0.0)
            scores[start:stop] = 1.0 - d2 / radius2[None, :]
        return scores

    # ------------------------------------------------------------------
    # Validation / preparation
    # ------------------------------------------------------------------

    def _validate_init_params(self) -> None:
        if self.metric not in ("auto", "euclidean", "cosine"):
            raise ValueError("metric must be one of {'auto', 'euclidean', 'cosine'}.")
        if self.selection not in ("greedy_uncovered", "farthest"):
            raise ValueError("selection must be one of {'greedy_uncovered', 'farthest'}.")
        if not (0.0 < self.cover_fraction <= 1.0):
            raise ValueError("cover_fraction must be in (0, 1].")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        k_auto = self.k == "auto"
        radius_auto = self.radius == "auto"
        if k_auto and radius_auto:
            raise ValueError("Only one of k or radius may be 'auto'.")
        if not k_auto and not isinstance(self.k, dict):
            if int(self.k) <= 0:  # type: ignore[arg-type]
                raise ValueError("k must be positive, a class-specific dict, or 'auto'.")
        if not radius_auto and not isinstance(self.radius, dict):
            if float(self.radius) <= 0:  # type: ignore[arg-type]
                raise ValueError("radius must be positive, a class-specific dict, or 'auto'.")
        if self.high_dim_threshold <= 0:
            raise ValueError("high_dim_threshold must be positive.")
        if self.max_balls is not None and self.max_balls <= 0:
            raise ValueError("max_balls must be positive when provided.")

    def _resolve_metric(self, n_features: int) -> str:
        if self.metric == "auto":
            return "cosine" if int(n_features) >= self.high_dim_threshold else "euclidean"
        return self.metric

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        if self._resolved_metric == "cosine":
            norms = np.linalg.norm(X, axis=1)
            denom = np.maximum(norms, self._eps)
            X = X / denom[:, None]
        return X.astype(self.dtype, copy=False)

    def _prepare_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=self.dtype).reshape(-1)
        self._check_fit()
        if x.shape[0] != self._centers.shape[1]:
            raise ValueError(f"x has {x.shape[0]} features, expected {self._centers.shape[1]}.")
        if self._resolved_metric == "cosine":
            norm = float(np.linalg.norm(x))
            x = x / max(norm, self._eps)
        return x.astype(self.dtype, copy=False)

    def _as_2d_float_array(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array; got shape {X.shape}.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")
        return X

    @staticmethod
    def _value_for_class(value: Any, c: Any) -> Any:
        if isinstance(value, dict):
            if c not in value:
                raise ValueError(f"Missing class-specific parameter for class {c!r}.")
            return value[c]
        return value

    def _check_fit(self) -> None:
        if self._centers is None or self._radii is None or self._radius2 is None or self._center_norms is None:
            raise AssertionError(f"{self.__class__.__name__} backend not fit.")

    def _make_class_diag(
        self,
        n: int,
        n_balls: int,
        radius: float,
        covered: np.ndarray,
        nearest_d2: np.ndarray,
        mode: str,
    ) -> Dict[str, Any]:
        finite_d2 = nearest_d2[np.isfinite(nearest_d2)]
        nearest_dist = np.sqrt(np.maximum(finite_d2, 0.0)) if finite_d2.size else np.asarray([], dtype=float)
        return {
            "mode": mode,
            "n_samples": int(n),
            "n_balls": int(n_balls),
            "radius": float(radius),
            "metric": self._resolved_metric,
            "cover_fraction_target": float(self.cover_fraction),
            "covered_fraction": float(np.mean(covered)) if covered.size else 0.0,
            "mean_nearest_center_distance": float(np.mean(nearest_dist)) if nearest_dist.size else 0.0,
            "max_nearest_center_distance": float(np.max(nearest_dist)) if nearest_dist.size else 0.0,
        }

    def _summarize_diagnostics(self, x_shape: Tuple[int, int], classes: np.ndarray) -> Dict[str, Any]:
        return {
            "n_samples": int(x_shape[0]),
            "n_features": int(x_shape[1]),
            "n_classes": int(len(classes)),
            "n_balls_total": int(0 if self._centers is None else self._centers.shape[0]),
            "metric": self._resolved_metric,
            "high_dim_threshold": int(self.high_dim_threshold),
            "cover_fraction_target": float(self.cover_fraction),
            "store_memberships": bool(self.store_memberships),
        }

    # ------------------------------------------------------------------
    # Properties matching centroid/prototype backend conventions
    # ------------------------------------------------------------------

    @property
    def centers(self) -> np.ndarray:
        """Return global ball-center matrix."""
        self._check_fit()
        return self._centers

    @property
    def radii(self) -> np.ndarray:
        """Return one radius per global ball."""
        self._check_fit()
        return self._radii

    @property
    def resolved_metric(self) -> Optional[str]:
        """Return the metric chosen during fitting."""
        return self._resolved_metric

    @property
    def class_center_id_arrays(self) -> Dict[Any, np.ndarray]:
        """Alias for class-to-ball id arrays, matching centroid backend naming."""
        return self._class_ball_id_arrays

    @property
    def class_ball_id_arrays(self) -> Dict[Any, np.ndarray]:
        """Return class-to-global-ball-id mappings as NumPy arrays."""
        return self._class_ball_id_arrays

    @property
    def cluster_to_class(self) -> Optional[np.ndarray]:
        """Return array mapping global ball ids to class labels."""
        return self._cluster_to_class

    @property
    def class_to_clusters(self) -> Dict[Any, set]:
        """Return class-to-global-ball-id mappings as sets."""
        return self._class_to_clusters

    @property
    def n_clusters_total(self) -> int:
        """Return total number of global balls."""
        return 0 if self._centers is None else int(self._centers.shape[0])

    @property
    def ball_to_points(self) -> Optional[Dict[int, np.ndarray]]:
        """Return optional training memberships, if stored."""
        return self._ball_to_points

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Return fit-level diagnostics."""
        return self._diagnostics

    @property
    def class_diagnostics(self) -> Dict[Any, Dict[str, Any]]:
        """Return class-level fit diagnostics."""
        return self._class_diagnostics


# Alias with a leading underscore for direct integration beside existing private backends.
_BallCoverManyToOne = BallCoverManyToOne
