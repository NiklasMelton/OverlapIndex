"""Optional one-pass prototype refinement for centroid backends.

The public estimator keeps this feature deliberately small and opt-in.  The
implementation in this module operates on a fitted centroid adapter, computes
outgoing isolation with the universal tiled scorer, and then replaces selected
parent centers with two deterministic observation representatives.  It does
not fit a second sklearn model and never materializes a samples-by-prototypes
distance matrix.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from overlapindex._universal_scorer import (
    compute_second_best_source_scores,
    iter_target_class_blocks,
)


REFINEMENT_METHOD = "balanced_median"
NO_REFINEMENT_METHOD = "none"


def refinement_method(enabled: bool) -> str:
    """Return the internal refinement method name for a public bool flag.

    ``prototype_refinement`` is intentionally a strict Python ``bool`` at the
    estimator boundary.  Keeping this conversion in one place avoids leaking
    the public flag into fitted diagnostics, which continue to expose the
    descriptive method names used by the refinement implementation.
    """

    if type(enabled) is not bool:
        raise ValueError("prototype_refinement must be a boolean (True or False).")
    return REFINEMENT_METHOD if enabled else NO_REFINEMENT_METHOD


def _plain_label(value: Any) -> Any:
    """Convert NumPy scalar labels to ordinary Python values for diagnostics."""

    return value.item() if isinstance(value, np.generic) else value


def empty_refinement_summary(
    method: str = NO_REFINEMENT_METHOD,
    *,
    prototype_count: int = 0,
) -> dict[str, Any]:
    """Return the fitted-diagnostics shape used by the estimator.

    ``records`` is a tuple so callers can retain a stable, read-mostly view of
    per-parent decisions.  Individual records are ordinary dictionaries for
    convenient serialization and backwards-compatible key access.
    """

    count = int(prototype_count)
    return {
        "method": str(method),
        "mode": str(method),
        "splitter": str(method),
        "split_method": str(method),
        "prototype_count_before": count,
        "prototype_count_after": count,
        "eligible_count": 0,
        "attempted_count": 0,
        "applied_count": 0,
        "skipped_count": 0,
        "attempted": 0,
        "applied": 0,
        "skipped": 0,
        "eligible_parent_ids": (),
        "applied_parent_ids": (),
        "skipped_parent_ids": (),
        "records": (),
    }


def _select_rows_dense(X: Any, rows: np.ndarray) -> np.ndarray:
    """Select fit rows and densify only the local parent support block."""

    if sparse.issparse(X):
        values = X[rows].toarray()
    else:
        values = np.asarray(X[rows], dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("prototype refinement requires a two-dimensional feature matrix")
    return values


def _fit_isolation(
    backend: Any,
    X: Any,
    Y: np.ndarray,
    classes: Sequence[Any],
    *,
    memory_budget_mb: int,
    row_cap: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return source best ids, source runner-up scores, support and counts.

    ``runner_count`` is the outgoing isolation count used by the research
    prototype: a source row contributes only when no competing class's best
    prototype strictly beats the source-class second-best score.
    """

    label_to_position = {label: int(position) for position, label in enumerate(classes)}
    source_positions = np.asarray(
        [label_to_position[label] for label in np.asarray(Y, dtype=object)],
        dtype=int,
    )
    class_ids = backend.class_center_id_arrays
    integer_ids = {
        int(position): np.asarray(class_ids.get(label, np.asarray([], dtype=int)), dtype=int).reshape(-1)
        for label, position in label_to_position.items()
    }
    rows = np.arange(int(source_positions.size), dtype=int)
    source = compute_second_best_source_scores(
        X,
        rows,
        source_positions,
        integer_ids,
        backend,
        memory_budget_mb=int(memory_budget_mb),
        row_cap=row_cap,
    )
    best_ids = np.asarray(source.best_prototype_ids, dtype=int)
    own_second = np.asarray(source.second_best_scores, dtype=float)

    n_prototypes = int(backend.centers.shape[0])
    support = np.zeros(n_prototypes, dtype=int)
    valid_best = best_ids[(best_ids >= 0) & (best_ids < n_prototypes)]
    if valid_best.size:
        support += np.bincount(valid_best, minlength=n_prototypes)[:n_prototypes]

    competitor_beats = np.zeros(best_ids.size, dtype=bool)
    for block in iter_target_class_blocks(
        X,
        integer_ids,
        backend,
        memory_budget_mb=int(memory_budget_mb),
        row_cap=row_cap,
        sample_rows=rows,
    ):
        block_rows = np.asarray(block.row_indices, dtype=int)
        block_sources = source_positions[block_rows]
        for column, target_position in enumerate(block.class_ids.tolist()):
            target_position = int(target_position)
            competitor_beats[block_rows] |= (
                (block_sources != target_position)
                & (np.asarray(block.best_scores[:, column]) > own_second[block_rows])
            )

    runner_count = np.zeros(n_prototypes, dtype=int)
    isolated_rows = np.flatnonzero(~competitor_beats)
    isolated_ids = best_ids[isolated_rows]
    isolated_ids = isolated_ids[(isolated_ids >= 0) & (isolated_ids < n_prototypes)]
    if isolated_ids.size:
        runner_count += np.bincount(isolated_ids, minlength=n_prototypes)[:n_prototypes]
    return best_ids, own_second, support, runner_count


def _stable_child_order(children: np.ndarray) -> np.ndarray:
    """Return a deterministic lexicographic ordering of two child centers."""

    return np.asarray(
        np.lexsort(tuple(children[:, column] for column in reversed(range(children.shape[1])))),
        dtype=int,
    )


def _split_parent(
    X: Any,
    rows: np.ndarray,
    parent_center: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], Optional[Tuple[int, int]], str]:
    """Build balanced observation-median children for one parent.

    Returns ``(children, selected_rows, child_supports, reason)``.  A non-empty
    reason denotes a skipped parent; applied splits return an empty reason.
    """

    if rows.size < 2:
        return None, None, None, "support<2"
    points = _select_rows_dense(X, rows)
    if not np.isfinite(points).all():
        return None, None, None, "invalid_points"
    parent_center = np.asarray(parent_center, dtype=float).reshape(-1)
    if parent_center.size != points.shape[1] or not np.isfinite(parent_center).all():
        return None, None, None, "invalid_parent"

    first_distances = np.sum((points - parent_center) ** 2, axis=1)
    if not np.isfinite(first_distances).all():
        return None, None, None, "invalid_points"
    first = int(np.argmax(first_distances))
    second_distances = np.sum((points - points[first]) ** 2, axis=1)
    if not np.isfinite(second_distances).all():
        return None, None, None, "invalid_points"
    second = int(np.argmax(second_distances))
    if second == first or np.array_equal(points[first], points[second]):
        return None, None, None, "duplicate_points"

    direction = np.asarray(points[second] - points[first], dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= 0.0:
        return None, None, None, "invalid_axis"
    projections = np.asarray(points @ (direction / direction_norm), dtype=float)
    if not np.isfinite(projections).all():
        return None, None, None, "invalid_projection"

    # ``rows`` is in original fit-row coordinates and therefore makes exact
    # projection ties deterministic regardless of sparse/dense slicing order.
    ordered = np.lexsort((rows, projections))
    split_at = int(points.shape[0] // 2)
    if split_at <= 0 or split_at >= points.shape[0]:
        return None, None, None, "empty_child"
    halves = (ordered[:split_at], ordered[split_at:])
    representatives = []
    representative_rows = []
    child_supports = []
    for half in halves:
        half_points = points[half]
        half_rows = rows[half]
        coordinate_median = np.median(half_points, axis=0)
        squared_distances = np.sum((half_points - coordinate_median) ** 2, axis=1)
        if not np.isfinite(coordinate_median).all() or not np.isfinite(squared_distances).all():
            return None, None, None, "invalid_child"
        nearest_order = np.lexsort((half_rows, squared_distances))
        nearest = int(nearest_order[0])
        representatives.append(np.asarray(half_points[nearest], dtype=float))
        representative_rows.append(int(half_rows[nearest]))
        child_supports.append(int(half.size))

    children = np.asarray(representatives, dtype=float)
    if (
        children.shape != (2, points.shape[1])
        or not np.isfinite(children).all()
        or np.array_equal(children[0], children[1])
    ):
        return None, None, None, "duplicate_representatives"

    order = _stable_child_order(children)
    children = children[order]
    representative_rows = [representative_rows[int(index)] for index in order.tolist()]
    child_supports = [child_supports[int(index)] for index in order.tolist()]
    return (
        children,
        (int(representative_rows[0]), int(representative_rows[1])),
        (int(child_supports[0]), int(child_supports[1])),
        "",
    )


def apply_balanced_median_refinement(
    backend: Any,
    X: Any,
    Y: Any,
    *,
    memory_budget_mb: int = 256,
    row_cap: Optional[int] = None,
) -> dict[str, Any]:
    """Apply one frozen balanced observation-median pass to a backend.

    The adapter has already been fitted when this function is called.  Parent
    eligibility is measured once against the original center set; appended
    children are never reconsidered during the same pass.
    """

    centers_before = np.asarray(backend.centers)
    prototype_count_before = int(centers_before.shape[0])
    summary = empty_refinement_summary(
        REFINEMENT_METHOD,
        prototype_count=prototype_count_before,
    )
    if prototype_count_before == 0:
        return summary

    Y_array = np.asarray(Y, dtype=object).reshape(-1)
    if Y_array.size == 0:
        return summary
    classes = list(dict.fromkeys(Y_array.tolist()))
    if len(classes) < 2:
        return summary

    best_ids, _own_second, support, runner_count = _fit_isolation(
        backend,
        X,
        Y_array,
        classes,
        memory_budget_mb=int(memory_budget_mb),
        row_cap=row_cap,
    )
    owner_values = np.asarray(backend.cluster_to_class, dtype=object)
    owner_for = {
        int(pid): owner_values[int(pid)]
        for pid in range(min(prototype_count_before, owner_values.size))
    }
    eligible = tuple(
        int(pid)
        for pid in range(prototype_count_before)
        if int(support[pid]) >= 2 and int(runner_count[pid]) == 0
    )

    records = []
    applied_parent_ids = []
    skipped_parent_ids = []
    replacements: Dict[int, np.ndarray] = {}
    appended: list[tuple[int, Any, np.ndarray]] = []
    next_id = prototype_count_before

    original_centers = np.asarray(centers_before, dtype=float)
    for parent in eligible:
        rows = np.flatnonzero(best_ids == parent).astype(int, copy=False)
        label = _plain_label(owner_for.get(parent))
        children, selected_rows, child_supports, reason = _split_parent(
            X,
            rows,
            original_centers[parent],
        )
        base_record: dict[str, Any] = {
            "parent": int(parent),
            "parent_id": int(parent),
            "original_id": int(parent),
            "class": label,
            "class_label": label,
            "support": int(rows.size),
            "support_before": int(rows.size),
            "status": "skipped",
            "reason": str(reason),
            "children": (),
            "child_ids": (),
            "child_supports": (),
            "selected_observation_indices": (),
        }
        if children is None:
            skipped_parent_ids.append(int(parent))
            records.append(base_record)
            continue

        child_id = int(next_id)
        next_id += 1
        replacements[int(parent)] = np.asarray(children[0], dtype=backend._dtype)
        appended.append((child_id, label, np.asarray(children[1], dtype=backend._dtype)))
        applied_parent_ids.append(int(parent))
        base_record.update(
            {
                "status": "applied",
                "reason": "",
                "children": (int(parent), child_id),
                "child_ids": (int(parent), child_id),
                "child_supports": tuple(int(value) for value in child_supports or ()),
                "selected_observation_indices": tuple(int(value) for value in selected_rows or ()),
                "selected_sample_indices": tuple(int(value) for value in selected_rows or ()),
                "new_id": child_id,
            }
        )
        records.append(base_record)

    if appended:
        final_centers = np.array(original_centers, copy=True)
        for parent, replacement in replacements.items():
            final_centers[parent] = replacement
        final_centers = np.vstack(
            [final_centers] + [child for _child_id, _label, child in appended]
        ).astype(backend._dtype, copy=False)
        backend._centers = final_centers
        backend._center_norms = np.einsum("ij,ij->i", final_centers, final_centers)

        class_lists = {
            label: [int(value) for value in np.asarray(ids, dtype=int).reshape(-1).tolist()]
            for label, ids in backend._class_center_ids.items()
        }
        for child_id, label, _child in appended:
            class_lists.setdefault(label, []).append(int(child_id))
        backend._class_center_ids = class_lists
        backend._class_center_id_arrays = {
            label: np.asarray(ids, dtype=int) for label, ids in class_lists.items()
        }
        backend._class_to_clusters = defaultdict(
            set,
            {label: set(ids) for label, ids in class_lists.items()},
        )
        original_owners = np.asarray(backend._cluster_to_class, dtype=object).reshape(-1)
        appended_owners = np.asarray([label for _child_id, label, _child in appended], dtype=object)
        backend._cluster_to_class = np.concatenate((original_owners, appended_owners))

    prototype_count_after = int(backend._centers.shape[0])
    summary.update(
        {
            "prototype_count_after": prototype_count_after,
            "eligible_count": len(eligible),
            "attempted_count": len(eligible),
            "applied_count": len(applied_parent_ids),
            "skipped_count": len(skipped_parent_ids),
            "attempted": len(eligible),
            "applied": len(applied_parent_ids),
            "skipped": len(skipped_parent_ids),
            "eligible_parent_ids": tuple(eligible),
            "applied_parent_ids": tuple(applied_parent_ids),
            "skipped_parent_ids": tuple(skipped_parent_ids),
            "records": tuple(dict(record) for record in records),
        }
    )
    return summary


__all__ = [
    "REFINEMENT_METHOD",
    "NO_REFINEMENT_METHOD",
    "refinement_method",
    "empty_refinement_summary",
    "apply_balanced_median_refinement",
]
