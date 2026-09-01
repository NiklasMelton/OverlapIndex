"""Compute broad-RFF and SAVE+RFF ranking scores from one shared fold pass."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from experiments.continuous_kernel_alignment import run as kernel
from experiments.continuous_multiview_ranking import run as multiview
from experiments.continuous_null_diagnostic import run as null_parent
from experiments.continuous_save_ranking import run as save


def _mean_or_none(key: str, rows: Sequence[Mapping[str, Any]]) -> float | None:
    values = [row[key] for row in rows]
    return float(np.mean(values)) if all(value is not None for value in values) else None


def _aggregate_kernel(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bandwidths = {}
    for multiplier in kernel.BANDWIDTH_MULTIPLIERS:
        selected = [row["kernel_alignment"]["feature_bandwidths"][str(multiplier)] for row in folds]
        bandwidths[str(multiplier)] = {
            "raw_alignment": _mean_or_none("raw_alignment", selected),
            "null_alignment": _mean_or_none("null_alignment", selected),
            "score": _mean_or_none("score", selected),
        }
    selected = [row["kernel_alignment"] for row in folds]
    return {
        "raw_alignment": _mean_or_none("raw_alignment", selected),
        "null_alignment": _mean_or_none("null_alignment", selected),
        "score": _mean_or_none("score", selected),
        "feature_bandwidths": bandwidths,
    }


def evaluate_shared_pair(
    X: np.ndarray,
    primitive: np.ndarray,
    y: np.ndarray,
    panel_seed: int,
    candidate_position: int,
) -> dict[str, Any]:
    """Return exactly the two ranking scores using one fitted selector per fold.

    The RFF-mean block is constructed once and used both as the broad kernel
    view and as one half of the SAVE+RFF multiview geometry. The sole fitted
    ContinuousOverlapIndex model is the SAVE+RFF model. Its target scaling is
    feature-independent and therefore also supplies the exact scaled targets
    required by broad-RFF kernel alignment.
    """

    values = np.asarray(X, dtype=np.float32)
    primitive_values = np.asarray(primitive, dtype=np.float32)
    targets = np.asarray(y, dtype=float)
    if values.ndim != 2 or primitive_values.ndim != 2 or targets.ndim != 1:
        raise ValueError("shared bounded evaluation requires 2D X/primitive and 1D y")
    if len(values) != len(primitive_values) or len(values) != len(targets):
        raise ValueError("shared bounded evaluation rows are not aligned")
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(primitive_values))
        and np.all(np.isfinite(targets))
    ):
        raise ValueError("shared bounded evaluation inputs must be finite")
    if type(panel_seed) is not int or type(candidate_position) is not int:
        raise TypeError("panel_seed and candidate_position must be integers")

    folds = []
    started = time.perf_counter()
    splitter = KFold(n_splits=4, shuffle=True, random_state=panel_seed)
    for fold, (train, holdout) in enumerate(splitter.split(values)):
        fit_seed = null_parent._candidate_fit_seed(
            panel_seed, candidate_position, fold
        )
        save_train, save_holdout, save_diagnostics = multiview._single_block(
            "save",
            values[train],
            values[holdout],
            primitive_values[train],
            primitive_values[holdout],
            targets[train],
            fit_seed,
        )
        rff_train, rff_holdout, rff_diagnostics = multiview._single_block(
            "rff_mean",
            values[train],
            values[holdout],
            primitive_values[train],
            primitive_values[holdout],
            targets[train],
            fit_seed,
        )
        multiview_train, multiview_holdout = multiview._equal_block_concatenation(
            [save_train, rff_train], [save_holdout, rff_holdout]
        )
        estimator = clone(save._base_selector(fit_seed))
        estimator._fit_geometry_only(multiview_train, targets[train])
        bank_seed = null_parent._common_bank_seed(panel_seed, fold)
        bank = null_parent._permutation_bank(len(holdout), 32, bank_seed)
        current_components = null_parent._fixed_model_components(
            estimator, multiview_holdout, targets[holdout], bank
        )
        current_null = float(np.mean(current_components["null_loss_samples"]))
        current = null_parent._score_from_losses(
            float(current_components["actual_loss"]), current_null
        )
        y_train_scaled = estimator._transform_targets(
            estimator._validate_Y(targets[train])
        )
        y_holdout_scaled = estimator._transform_targets(
            estimator._validate_Y(targets[holdout])
        )
        broad_kernel = kernel._kernel_fold_components(
            rff_train,
            rff_holdout,
            y_train_scaled,
            y_holdout_scaled,
            bank,
        )
        folds.append(
            {
                "fold": int(fold),
                "fit_seed": int(fit_seed),
                "common_bank_seed": int(bank_seed),
                "row_count": int(len(holdout)),
                "fitted_selector_count": 1,
                "prototype_count": int(estimator._model.n_clusters_total),
                "broad_rff": {
                    "geometry_diagnostics": rff_diagnostics,
                    "kernel_alignment": broad_kernel,
                    "prototype_fit_elided": True,
                },
                "save_rff_current": {
                    "geometry_diagnostics": {
                        "geometry": "multiview",
                        "blocks": ["save", "rff_mean"],
                        "block_count": 2,
                        "block_weight": 1.0 / np.sqrt(2.0),
                        "block_diagnostics": [
                            {
                                "block": "save",
                                "dimension": int(save_train.shape[1]),
                                "state": save_diagnostics,
                            },
                            {
                                "block": "rff_mean",
                                "dimension": int(rff_train.shape[1]),
                                "state": rff_diagnostics,
                            },
                        ],
                        "output_dimension": int(multiview_train.shape[1]),
                    },
                    "actual_loss": float(current_components["actual_loss"]),
                    "null_loss": current_null,
                    "null_loss_samples": current_components["null_loss_samples"],
                    "loss_ratio": current["loss_ratio"],
                    "score": float(current["score"]),
                },
            }
        )
    current_rows = [row["save_rff_current"] for row in folds]
    ratios = [row["loss_ratio"] for row in current_rows]
    broad_kernel = _aggregate_kernel(
        [
            {"kernel_alignment": row["broad_rff"]["kernel_alignment"]}
            for row in folds
        ]
    )
    return {
        "broad_rff": {
            "score": float(
                broad_kernel["feature_bandwidths"]["4.0"]["score"]
            ),
            "kernel_alignment": broad_kernel,
            "prototype_fit_elided": True,
        },
        "save_rff_current": {
            "score": float(np.mean([row["score"] for row in current_rows])),
            "mean_actual_loss": float(
                np.mean([row["actual_loss"] for row in current_rows])
            ),
            "mean_null_loss": float(
                np.mean([row["null_loss"] for row in current_rows])
            ),
            "mean_loss_ratio": (
                float(np.mean(ratios))
                if all(value is not None for value in ratios)
                else None
            ),
        },
        "folds": folds,
        "fitted_selector_count": 4,
        "wall_seconds": float(time.perf_counter() - started),
    }


__all__ = ["evaluate_shared_pair"]
