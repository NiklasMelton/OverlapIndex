"""Memory-bounded primitives for backend-neutral discrete overlap scoring.

The offline centroid and ball-cover adapters expose two deliberately small
vectorized hooks: :meth:`prepare_score_input` and
:meth:`score_block_prepared`.  This module contains the bookkeeping needed to
use those hooks without constructing an ``n_samples x n_prototypes`` score
matrix.  It does not know anything about public overlap mappings or index
aggregation; callers receive row-aligned score events and are responsible for
aggregating them.

Scores are always interpreted with the same convention as the adapters:
larger values are better.  Running maxima use *strict* ``>`` comparisons.  As
a result, a prototype encountered earlier in its supplied id array wins an
exact tie, which is deterministic and agrees with source-owned tie handling
in the estimator.  A class with one prototype has no second score and its
threshold is represented by ``-inf``.

The planner is intentionally conservative.  Score tiles and masks are
accounted for as float32 and bool storage respectively, while prepared input
blocks reserve the caller-specified number of dense float32 ``R x D`` arrays.
The estimator passes four arrays to cover non-contiguous float64-to-float32
selection plus normalization temporaries; direct planner callers retain the
two-array default.  At least one row and one prototype are planned even when
the caller's budget is smaller than that irreducible tile.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Optional, Union

import numpy as np
from scipy import sparse


DEFAULT_MEMORY_BUDGET_MB = 256
"""Default scratch-memory budget used by :func:`plan_score_tiles`."""

_FLOAT32_BYTES = np.dtype(np.float32).itemsize
_BOOL_BYTES = np.dtype(bool).itemsize


def validate_memory_budget_mb(
    value: Any,
    name: str = "offline_memory_budget_mb",
) -> int:
    """Validate and return a positive, non-boolean integer budget in MiB.

    ``bool`` and floating-point values are rejected instead of silently
    truncating them.  The explicit error wording is shared with the public
    estimator's parameter validation.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def validate_row_cap(
    value: Any,
    name: str = "offline_chunk_size",
    *,
    allow_none: bool = True,
) -> Optional[int]:
    """Validate and return a positive integer row cap.

    ``None`` means that no caller-specified cap is present when
    ``allow_none=True``.  As with :func:`validate_memory_budget_mb`, genuine
    integer types are required and booleans are not accepted.
    """

    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a positive integer{suffix}.")
    result = int(value)
    if result <= 0:
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a positive integer{suffix}.")
    return result


def _validate_positive_count(value: Any, name: str) -> int:
    """Validate a positive genuine integer used by the planner."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _prototype_count(value: Any, name: str) -> int:
    """Return a prototype count from an integer or an id sequence."""

    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return _validate_positive_count(value, name)
    try:
        array = np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise ValueError(f"{name} must be an integer or a sequence of ids.") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return _validate_positive_count(array.size, name)


def _max_prototype_count(prototype_counts: Any) -> int:
    """Resolve the largest class prototype count without materializing ids."""

    if isinstance(prototype_counts, Integral) and not isinstance(
        prototype_counts, (bool, np.bool_)
    ):
        return _prototype_count(prototype_counts, "prototype_counts")
    if isinstance(prototype_counts, Mapping):
        counts = [
            _prototype_count(value, f"prototype_counts[{key!r}]")
            for key, value in prototype_counts.items()
        ]
    else:
        try:
            counts = [
                _prototype_count(value, "prototype_counts entry")
                for value in prototype_counts
            ]
        except TypeError as exc:
            raise ValueError(
                "prototype_counts must be an integer, mapping, or sequence."
            ) from exc
    return max(counts, default=0)


@dataclass(frozen=True)
class ScoreTilePlan:
    """Deterministic row/prototype tile dimensions and byte accounting.

    ``row_tile_size`` and ``prototype_tile_size`` are upper bounds.  A caller
    may use smaller tiles for a class with fewer prototypes or for a final
    partial row block.  ``estimated_bytes`` is the conservative scratch size
    for one full planned tile; it can exceed ``memory_budget_bytes`` only for
    the irreducible one-row/one-prototype tile when the requested budget is
    smaller than its prepared-input overhead.
    """

    row_tile_size: int
    prototype_tile_size: int
    n_rows: int
    n_features: int
    max_prototypes: int
    memory_budget_mb: int
    memory_budget_bytes: int
    estimated_bytes: int
    prepared_arrays: int
    score_tiles: int
    mask_tiles: int
    result_columns: int
    result_arrays: int
    result_id_arrays: int

    @property
    def row_cap(self) -> int:
        """Alias for :attr:`row_tile_size` used by estimator callers."""

        return self.row_tile_size

    @property
    def prototype_cap(self) -> int:
        """Alias for :attr:`prototype_tile_size`."""

        return self.prototype_tile_size

    @property
    def within_budget(self) -> bool:
        """Whether the conservative tile estimate fits the caller budget."""

        return self.estimated_bytes <= self.memory_budget_bytes


# Shorter spelling retained for callers that prefer ``TilePlan``.
TilePlan = ScoreTilePlan


def plan_score_tiles(
    n_rows: int,
    n_features: int,
    prototype_counts: Union[int, Mapping[Any, Any], Sequence[Any]],
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    row_cap: Optional[int] = None,
    *,
    prepared_arrays: int = 2,
    score_tiles: int = 2,
    mask_tiles: int = 1,
    result_columns: int = 0,
    result_arrays: int = 0,
    result_id_arrays: int = 0,
) -> ScoreTilePlan:
    """Plan deterministic row/prototype tiles under a scratch-memory budget.

    Parameters
    ----------
    n_rows, n_features : int
        Number of rows and feature columns available to the scorer.  Counts
        may be zero; the returned tile dimensions remain at least one for
        convenient iteration.
    prototype_counts : int, mapping, or sequence
        Maximum number of prototypes in any class.  A mapping may contain
        prototype-id arrays or integer counts; a sequence is interpreted the
        same way.  Unequal class sizes therefore work without padding.
    memory_budget_mb : int, default=256
        Caller scratch budget in mebibytes.
    row_cap : int or None, optional
        Hard upper bound supplied by the caller (``offline_chunk_size``).
    prepared_arrays, score_tiles, mask_tiles : int, keyword-only
        Conservative numbers of dense prepared ``R x D`` arrays, float32
        score ``R x P`` tiles, and bool ``R x P`` masks retained concurrently.
    result_columns, result_arrays, result_id_arrays : int, keyword-only
        Optional row-by-class result arrays retained while processing a tile.
        ``result_arrays`` use float32 bytes and ``result_id_arrays`` use int64
        bytes.  They are useful when reducing one packed prototype tile into
        an ``R x B`` class-best block.

    Notes
    -----
    The planner first honors the requested row cap and chooses the largest
    prototype tile that fits.  If prepared-input overhead leaves no room for a
    prototype at that row count, rows are reduced deterministically until a
    one-prototype tile fits.  A minimum one-by-one tile is always returned.
    """

    n_rows_i = _validate_positive_count(n_rows, "n_rows")
    n_features_i = _validate_positive_count(n_features, "n_features")
    max_prototypes = _max_prototype_count(prototype_counts)
    budget_mb = validate_memory_budget_mb(memory_budget_mb)
    cap = validate_row_cap(row_cap)
    prepared_n = _validate_positive_count(prepared_arrays, "prepared_arrays")
    score_n = _validate_positive_count(score_tiles, "score_tiles")
    mask_n = _validate_positive_count(mask_tiles, "mask_tiles")
    result_columns_n = _validate_positive_count(result_columns, "result_columns")
    result_n = _validate_positive_count(result_arrays, "result_arrays")
    result_id_n = _validate_positive_count(result_id_arrays, "result_id_arrays")

    budget_bytes = int(budget_mb) * 1024 * 1024
    requested_rows = n_rows_i if cap is None else min(n_rows_i, cap)
    # Keep at least one row for callers that pass an empty event set.
    requested_rows = max(1, requested_rows)

    per_row_prepared = prepared_n * n_features_i * _FLOAT32_BYTES
    per_row_prototype = score_n * _FLOAT32_BYTES + mask_n * _BOOL_BYTES
    per_row_result = (
        result_n * result_columns_n * _FLOAT32_BYTES
        + result_id_n * result_columns_n * np.dtype(np.int64).itemsize
    )
    max_p = max(1, max_prototypes)

    def _prototype_capacity(rows: int) -> int:
        """Largest prototype tile for ``rows`` under the budget."""

        if per_row_prototype <= 0:
            return max_p
        available = budget_bytes - rows * (per_row_prepared + per_row_result)
        return max(0, available // (rows * per_row_prototype))

    # Honor a caller row cap whenever it permits a minimum score tile.  If the
    # cap is too large for the budget, reduce rows to the largest feasible
    # value.  The search is integer arithmetic and therefore deterministic.
    proto_capacity = _prototype_capacity(requested_rows)
    if proto_capacity < 1:
        denominator = per_row_prepared + per_row_result + per_row_prototype
        feasible_rows = budget_bytes // denominator if denominator else requested_rows
        requested_rows = max(1, min(requested_rows, int(feasible_rows)))
        proto_capacity = _prototype_capacity(requested_rows)
    row_size = max(1, min(n_rows_i if n_rows_i else 1, requested_rows))
    proto_size = max(1, min(max_p, proto_capacity if proto_capacity else 1))

    estimated = (
        row_size * per_row_prepared
        + row_size * proto_size * per_row_prototype
        + row_size * per_row_result
    )
    return ScoreTilePlan(
        row_tile_size=int(row_size),
        prototype_tile_size=int(proto_size),
        n_rows=n_rows_i,
        n_features=n_features_i,
        max_prototypes=max_prototypes,
        memory_budget_mb=budget_mb,
        memory_budget_bytes=budget_bytes,
        estimated_bytes=int(estimated),
        prepared_arrays=prepared_n,
        score_tiles=score_n,
        mask_tiles=mask_n,
        result_columns=result_columns_n,
        result_arrays=result_n,
        result_id_arrays=result_id_n,
    )


# Friendly alias for callers that use the shorter planner name.
plan_tiles = plan_score_tiles


def _matrix_shape(X: Any) -> tuple[int, int]:
    """Validate only matrix dimensionality while preserving sparse inputs."""

    if sparse.issparse(X):
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array; got shape {X.shape}.")
        return int(X.shape[0]), int(X.shape[1])
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(f"X must be a 2D array; got shape {array.shape}.")
    return int(array.shape[0]), int(array.shape[1])


def _select_rows(X: Any, rows: np.ndarray) -> Any:
    """Select rows while preserving CSR inputs and supporting dense sequences."""

    if sparse.issparse(X):
        return X[rows]
    array = np.asarray(X)
    # Most single-label class blocks (and many event blocks) are contiguous.
    # Use a view for those ranges so the scorer does not retain an additional
    # dense R x D fancy-indexing copy alongside backend preparation buffers.
    # Duplicate or non-monotone event rows still use advanced indexing, which
    # is handled by the conservative four-prepared-array planner allowance.
    if rows.size and (rows.size == 1 or np.all(np.diff(rows) == 1)):
        start = int(rows[0])
        stop = int(rows[-1]) + 1
        return array[start:stop]
    return array[rows]


def _validate_row_indices(
    rows: Optional[Sequence[int]],
    n_rows: int,
    *,
    name: str = "sample_rows",
) -> np.ndarray:
    """Validate event-row indices and return an integer copy."""

    if rows is None:
        return np.arange(n_rows, dtype=int)
    array = np.asarray(rows)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if isinstance(array.dtype, np.dtype) and np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain integer row indices.")
    if not np.issubdtype(array.dtype, np.integer):
        # Object arrays can still contain genuine integer values; avoid lossy
        # conversion of floats such as 1.5.
        try:
            values = list(array)
        except TypeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"{name} must contain integer row indices.") from exc
        if any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
            for item in values
        ):
            raise ValueError(f"{name} must contain integer row indices.")
    result = np.asarray(array, dtype=int)
    if result.size and (np.any(result < 0) or np.any(result >= n_rows)):
        raise ValueError(f"{name} contains an out-of-range row index.")
    return result


def _as_prototype_ids(value: Any, name: str) -> np.ndarray:
    """Validate one class's global prototype-id sequence."""

    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    if array.dtype.kind == "b" or not np.issubdtype(array.dtype, np.integer):
        if array.dtype.kind == "O":
            values = list(array)
            if any(
                isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
                for item in values
            ):
                raise ValueError(f"{name} must be a one-dimensional integer array.")
        else:
            raise ValueError(f"{name} must be a one-dimensional integer array.")
    ids = np.asarray(array, dtype=int)
    if ids.size and np.any(ids < 0):
        raise ValueError(f"{name} must contain non-negative prototype ids.")
    return ids


def _normalize_class_prototypes(class_prototype_ids: Mapping[Any, Any]) -> dict[Any, np.ndarray]:
    """Copy and validate class-to-global-prototype id mappings."""

    if not isinstance(class_prototype_ids, Mapping):
        raise TypeError("class_prototype_ids must be a mapping from class to prototype ids.")
    return {
        label: _as_prototype_ids(ids, f"prototype ids for class {label!r}")
        for label, ids in class_prototype_ids.items()
    }


def _score_tile(
    backend: Any,
    prepared: Any,
    ids: np.ndarray,
) -> np.ndarray:
    """Call and validate one backend score tile as float32."""

    values = np.asarray(backend.score_block_prepared(prepared, ids), dtype=np.float32)
    if values.ndim != 2 or values.shape != (int(prepared.shape[0]), int(ids.size)):
        raise ValueError(
            "score_block_prepared must return an array shaped "
            f"({prepared.shape[0]}, {ids.size}); got {values.shape}."
        )
    return values


def _update_top_two(
    best_scores: np.ndarray,
    second_scores: np.ndarray,
    best_ids: np.ndarray,
    values: np.ndarray,
    ids: np.ndarray,
) -> None:
    """Update row-wise top two values using strict higher-is-better ties."""

    # Iterating prototype columns keeps exact-tie behavior deterministic and
    # avoids allocating a second R x P ordering matrix.
    for column, prototype_id in enumerate(ids):
        candidate = values[:, column]
        better = candidate > best_scores
        if np.any(better):
            second_scores[better] = best_scores[better]
            best_scores[better] = candidate[better]
            best_ids[better] = int(prototype_id)

        # Equal-to-best candidates become a valid second score, but never
        # replace the earlier best prototype.  Strict ``>`` is intentional.
        between = (~better) & (candidate > second_scores)
        if np.any(between):
            second_scores[between] = candidate[between]


@dataclass(frozen=True)
class SourceScoreResult:
    """Row-aligned source-class top-two scores for overlap event evaluation."""

    row_indices: np.ndarray
    source_class_ids: np.ndarray
    best_scores: np.ndarray
    second_best_scores: np.ndarray
    best_prototype_ids: np.ndarray

    @property
    def threshold(self) -> np.ndarray:
        """Alias exposing the second-best source score as an event threshold."""

        return self.second_best_scores

    @property
    def source_bmu_scores(self) -> np.ndarray:
        """Alias for the best source-owned prototype score."""

        return self.best_scores

    @property
    def source_bmu_ids(self) -> np.ndarray:
        """Alias for the selected source-owned prototype ids."""

        return self.best_prototype_ids


def compute_second_best_source_scores(
    X: Any,
    sample_rows: Sequence[int],
    source_class_ids: Sequence[int],
    class_prototype_ids: Mapping[Any, Any],
    backend: Any,
    *,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    row_cap: Optional[int] = None,
) -> SourceScoreResult:
    """Compute top-one and second-best scores within each source class.

    ``sample_rows`` and ``source_class_ids`` describe source events and may
    contain duplicate row indices (as required by multi-label observations).
    Only the selected event rows are prepared and scored.  Prototype arrays may
    have arbitrary, unequal lengths; each is traversed in deterministic tiles.

    The returned arrays preserve event order.  A source class with no supplied
    prototypes receives ``-inf`` scores and prototype id ``-1``; callers may
    then mark those events unevaluable without constructing a score matrix.
    """

    n_rows, n_features = _matrix_shape(X)
    rows = _validate_row_indices(sample_rows, n_rows)
    labels = np.asarray(source_class_ids)
    if labels.ndim != 1 or labels.size != rows.size:
        raise ValueError("source_class_ids must be a one-dimensional array aligned with sample_rows.")
    if labels.size == 0:
        labels = np.asarray(labels, dtype=int)
    elif labels.dtype.kind == "b" or not np.issubdtype(labels.dtype, np.integer):
        if labels.dtype.kind == "O":
            values = list(labels)
            if any(
                isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
                for item in values
            ):
                raise ValueError("source_class_ids must contain integer class ids.")
            labels = np.asarray(values, dtype=int)
        else:
            raise ValueError("source_class_ids must contain integer class ids.")
    labels = np.asarray(labels, dtype=int)
    prototypes = _normalize_class_prototypes(class_prototype_ids)
    missing = [label for label in np.unique(labels) if label not in prototypes]
    if missing:
        raise ValueError(f"No prototype ids supplied for source class {missing[0]!r}.")

    max_count = max((ids.size for ids in prototypes.values()), default=0)
    max_events_per_class = max(
        (int(np.count_nonzero(labels == class_id)) for class_id in set(labels.tolist())),
        default=0,
    )
    plan = plan_score_tiles(
        max(1, max_events_per_class),
        n_features,
        max_count,
        memory_budget_mb,
        row_cap,
        # Dense input may be float64 while adapters store float32; cosine
        # BallCover preparation can then retain conversion, normalization, and
        # row-selection buffers concurrently.  Four float32-equivalent arrays
        # conservatively cover that path (contiguous rows use a view above).
        prepared_arrays=4,
        score_tiles=1,
        mask_tiles=1,
    )
    n_events = rows.size
    best = np.full(n_events, -np.inf, dtype=np.float32)
    second = np.full(n_events, -np.inf, dtype=np.float32)
    best_ids = np.full(n_events, -1, dtype=int)

    # Group events by source class before preparing rows.  This prevents a
    # mixed event block from being scored against every class's prototypes and
    # keeps each GEMM restricted to one source class.
    unique_classes = list(dict.fromkeys(labels.tolist()))
    for class_id in unique_classes:
        positions = np.flatnonzero(labels == class_id)
        ids = prototypes[class_id]
        if ids.size == 0:
            continue
        for block_start in range(0, positions.size, plan.row_tile_size):
            block_stop = min(block_start + plan.row_tile_size, positions.size)
            block_positions = positions[block_start:block_stop]
            block_rows = rows[block_positions]
            prepared = backend.prepare_score_input(_select_rows(X, block_rows))
            local_best = np.full(int(block_rows.size), -np.inf, dtype=np.float32)
            local_second = np.full(int(block_rows.size), -np.inf, dtype=np.float32)
            local_ids = np.full(int(block_rows.size), -1, dtype=int)
            for proto_start in range(0, ids.size, plan.prototype_tile_size):
                proto_stop = min(proto_start + plan.prototype_tile_size, ids.size)
                tile_ids = ids[proto_start:proto_stop]
                values = _score_tile(backend, prepared, tile_ids)
                _update_top_two(local_best, local_second, local_ids, values, tile_ids)
            best[block_positions] = local_best
            second[block_positions] = local_second
            best_ids[block_positions] = local_ids

    return SourceScoreResult(
        row_indices=rows,
        source_class_ids=labels,
        best_scores=best,
        second_best_scores=second,
        best_prototype_ids=best_ids,
    )


# Name requested by some estimator call sites.
second_best_source_scores = compute_second_best_source_scores
compute_source_event_scores = compute_second_best_source_scores


@dataclass(frozen=True)
class TargetClassBlock:
    """Best target-class scores for one row block and its metadata.

    ``class_ids`` preserves the insertion order of the caller's class mapping.
    ``best_scores`` and ``best_prototype_ids`` are shaped ``(R, B)`` where
    ``R == len(row_indices)`` and ``B == len(class_ids)``.  A class can span
    multiple prototype score tiles; the maxima in these matrices retain the
    result across all of its tiles.
    """

    class_ids: np.ndarray
    row_start: int
    row_stop: int
    row_indices: np.ndarray
    best_scores: np.ndarray
    best_prototype_ids: np.ndarray

    @property
    def class_id(self) -> Any:
        """Return the sole class id for backward-compatible one-class use.

        New callers should use :attr:`class_ids`.  Accessing ``class_id`` on a
        block containing multiple classes is ambiguous and raises
        :class:`AttributeError` rather than silently choosing one.
        """

        if self.class_ids.size != 1:
            raise AttributeError("class_id is only defined for a one-class block")
        return self.class_ids[0]

    @property
    def target_class_id(self) -> Any:
        """Alias for :attr:`class_id` for one-class consumers."""

        return self.class_id

    @property
    def scores(self) -> np.ndarray:
        """Alias for :attr:`best_scores`."""

        return self.best_scores

    @property
    def prototype_ids(self) -> np.ndarray:
        """Alias for :attr:`best_prototype_ids`."""

        return self.best_prototype_ids

    @property
    def scores_by_class(self) -> np.ndarray:
        """Alias exposing the row-by-class best-score matrix."""

        return self.best_scores


def iter_target_class_blocks(
    X: Any,
    class_prototype_ids: Mapping[Any, Any],
    backend: Any,
    *,
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB,
    row_cap: Optional[int] = None,
    sample_rows: Optional[Sequence[int]] = None,
) -> Iterator[TargetClassBlock]:
    """Yield row blocks containing each target class's running best score.

    Raw ``X`` is accepted in dense or CSR form.  Every row block is prepared
    exactly when it is consumed, so callers never need to materialize a full
    prepared copy.  Target prototypes from all classes are packed into each
    score tile, producing one larger GEMM per tile.  A target class's prototype
    ids may split across tiles and row-wise maxima are retained across those
    tiles.  Ties preserve the first supplied prototype id through strict
    higher-is-better updates.

    ``sample_rows`` optionally selects event rows (including duplicates).  Each
    yielded block includes the selected absolute row indices and an enclosing
    ``row_start:row_stop`` range so aggregation can align events cheaply.
    """

    n_rows, n_features = _matrix_shape(X)
    rows = _validate_row_indices(sample_rows, n_rows)
    prototypes = _normalize_class_prototypes(class_prototype_ids)
    class_values = list(prototypes)
    class_ids = np.empty(len(class_values), dtype=object)
    class_ids[:] = class_values
    n_classes = int(class_ids.size)
    # Target tiles pack columns from multiple classes.  Plan against the
    # *total* prototype count so a generous budget can use one GEMM for all
    # classes; per-class segments are still split safely when one class is
    # oversized.
    total_count = int(sum(ids.size for ids in prototypes.values()))
    if n_classes == 0:
        return
    plan = plan_score_tiles(
        max(1, rows.size),
        n_features,
        total_count,
        memory_budget_mb,
        row_cap,
        prepared_arrays=4,
        score_tiles=1,
        mask_tiles=1,
        result_columns=n_classes,
        result_arrays=1,
        result_id_arrays=1,
    )

    # Flatten each class's prototype ids into deterministic segments.  A
    # segment never crosses a class boundary, but a class may have many
    # segments and therefore be reduced across multiple score tiles.
    segments: list[tuple[int, np.ndarray]] = []
    for class_position, class_id in enumerate(class_ids.tolist()):
        ids = prototypes[class_id]
        for proto_start in range(0, ids.size, plan.prototype_tile_size):
            proto_stop = min(proto_start + plan.prototype_tile_size, ids.size)
            segments.append((class_position, ids[proto_start:proto_stop]))

    # Row blocks are outermost so a prepared block is reused across every
    # target class tile.  Empty class mappings produce no yielded blocks.
    for block_start in range(0, rows.size, plan.row_tile_size):
        block_stop = min(block_start + plan.row_tile_size, rows.size)
        block_rows = rows[block_start:block_stop]
        best = np.full((int(block_rows.size), n_classes), -np.inf, dtype=np.float32)
        best_ids = np.full((int(block_rows.size), n_classes), -1, dtype=int)
        prepared = backend.prepare_score_input(_select_rows(X, block_rows))
        segment_cursor = 0
        while segment_cursor < len(segments):
            # Pack as many class segments as fit the planned prototype tile.
            packed_ids: list[np.ndarray] = []
            owners: list[int] = []
            packed_size = 0
            while segment_cursor < len(segments):
                owner, segment_ids = segments[segment_cursor]
                segment_size = int(segment_ids.size)
                if packed_ids and packed_size + segment_size > plan.prototype_tile_size:
                    break
                # A segment is itself no larger than the tile by construction.
                packed_ids.append(segment_ids)
                owners.extend([owner] * segment_size)
                packed_size += segment_size
                segment_cursor += 1
                if packed_size >= plan.prototype_tile_size:
                    break
            if not packed_ids:  # pragma: no cover - defensive cursor guard
                break
            tile_ids = np.concatenate(packed_ids).astype(int, copy=False)
            values = _score_tile(backend, prepared, tile_ids)
            owner_array = np.asarray(owners, dtype=int)
            for column, owner in enumerate(owner_array):
                candidate = values[:, column]
                selected = candidate > best[:, owner]
                if np.any(selected):
                    best[selected, owner] = candidate[selected]
                    best_ids[selected, owner] = int(tile_ids[column])
        if block_rows.size:
            row_start = int(np.min(block_rows))
            row_stop = int(np.max(block_rows)) + 1
        else:
            row_start = row_stop = 0
        yield TargetClassBlock(
            class_ids=class_ids.copy(),
            row_start=row_start,
            row_stop=row_stop,
            row_indices=block_rows.copy(),
            best_scores=best,
            best_prototype_ids=best_ids,
        )


# Alias used by callers that prefer a verb phrase.
target_class_blocks = iter_target_class_blocks
iter_target_score_blocks = iter_target_class_blocks


__all__ = [
    "DEFAULT_MEMORY_BUDGET_MB",
    "ScoreTilePlan",
    "TilePlan",
    "SourceScoreResult",
    "TargetClassBlock",
    "validate_memory_budget_mb",
    "validate_row_cap",
    "plan_score_tiles",
    "plan_tiles",
    "compute_second_best_source_scores",
    "second_best_source_scores",
    "compute_source_event_scores",
    "iter_target_class_blocks",
    "target_class_blocks",
    "iter_target_score_blocks",
]
