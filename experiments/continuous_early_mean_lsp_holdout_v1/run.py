"""Run the frozen early-mean LSP fresh-evaluation ranking benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any
import warnings

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_cached_embedding_information_audit_v1 import run as information
from experiments.continuous_cached_embedding_smooth_head_audit_v1 import run as smooth
from experiments.continuous_early_mean_lsp_holdout_v1 import extract, prepare
from experiments.continuous_early_mean_replication_v1 import oi as prior_oi
from experiments.continuous_intermediate_pooling_audit_v1 import run as head_parent
from experiments.continuous_ranking_benchmark_v2 import run as base


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_early_mean_lsp_holdout_v1" / "results"
HEADS = ("linear_ridge", "quadratic_kernel_ridge", "knn", "rbf_svr")
TIE_ATOL = 1.0e-12
MAX_PANEL_SEED = 4_000_000


def load_protocol() -> dict[str, Any]:
    protocol = extract.load_protocol()
    if (
        tuple(protocol["reference_heads"]["families"]) != HEADS
        or tuple(protocol["continuous_ranking"]["methods"]) != (
            "linear_cv_full", "raw_cv_max_full", "observed_oi_full", "broad_rff_full", "save_rff_full",
        )
        or int(protocol["expected_grid"]["candidate_cell_count"]) != 150
        or int(protocol["expected_grid"]["ranking_panel_count"]) != 30
    ):
        raise ValueError("early-mean LSP holdout execution contract changed")
    return protocol


def _source_hashes() -> dict[str, str]:
    paths = {
        "input_cohort": prepare.DEFAULT_OUTPUT_ROOT / "cohort.json",
        "input_manifest": prepare.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "embedding_manifest": extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "preparer": Path(prepare.__file__).resolve(),
        "extractor": Path(extract.__file__).resolve(),
        "prior_oi_runner": Path(prior_oi.__file__).resolve(),
        "shared_selector": Path(shared.__file__).resolve(),
        "continuous_overlap_index": ROOT / "overlapindex" / "ContinuousOverlapIndex.py",
    }
    return {name: base.sha256_path(path) for name, path in paths.items()}


def _panel_seed(*parts: Any) -> int:
    """Return a deterministic seed safe for the inherited seed arithmetic."""
    return 1 + int(base._seed(*parts)) % MAX_PANEL_SEED


def _draw_rows(
    protocol: Mapping[str, Any], cohort: Mapping[str, Any], draw_seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    target = protocol["target"]
    training = [row for row in cohort["rows"] if row["split"] == "training"]
    evaluation = [row for row in cohort["rows"] if row["split"] == "evaluation"]
    bins = int(target["target_bins"])
    count = int(target["training_rows_per_draw"])
    if count % bins or len(training) != int(target["training_pool_rows"]):
        raise ValueError("LSP holdout training draw contract changed")
    selected = []
    for target_bin in range(bins):
        candidates = [row for row in training if int(row["target_bin"]) == target_bin]
        candidates.sort(
            key=lambda row: base.sha256_payload(
                ["early-mean-lsp-holdout-draw", int(draw_seed), str(row["sample_id"])]
            )
        )
        selected.extend(candidates[: count // bins])
    if (
        len(selected) != count
        or len({str(row["sample_id"]) for row in selected}) != count
        or len(evaluation) != int(target["evaluation_rows"])
        or {str(row["sample_id"]) for row in selected}.intersection(str(row["sample_id"]) for row in evaluation)
    ):
        raise ValueError("LSP holdout draw rows are invalid")
    return selected, evaluation


def _best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    values = np.asarray([float(row["mean_r2"]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("head tuning produced a nonfinite score")
    return rows[int(np.flatnonzero(values >= np.max(values) - TIE_ATOL)[0])]


def _rbf_grid(protocol: Mapping[str, Any]) -> list[dict[str, float]]:
    config = protocol["reference_heads"]["rbf_svr"]
    return [
        {"C_standardized": float(C), "epsilon_standardized": float(epsilon), "bandwidth_factor": float(factor)}
        for C, epsilon, factor in itertools.product(
            config["C_standardized_target_units"],
            config["epsilon_standardized_target_units"],
            config["bandwidth_factors"],
        )
    ]


def _quadratic_kernels(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dimension = int(train.shape[1])
    if dimension <= 0 or other.shape[1] != dimension:
        raise ValueError("quadratic head dimensions are invalid")
    return (
        np.square(1.0 + (train @ train.T) / dimension),
        np.square(1.0 + (other @ train.T) / dimension),
    )


def _kernel_ridge_predictions(
    train_kernel: np.ndarray, other_kernel: np.ndarray, y_train: np.ndarray, lambdas: Sequence[float],
) -> list[np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(train_kernel, dtype=np.float64))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ np.asarray(y_train, dtype=np.float64)
    return [
        np.asarray(other_kernel @ (eigenvectors @ (projected / (eigenvalues + len(y_train) * float(value)))), dtype=np.float64)
        for value in lambdas
    ]


def evaluate_heads(
    protocol: Mapping[str, Any], X_train: np.ndarray, y_train: np.ndarray, bins: np.ndarray,
    X_evaluation: np.ndarray, y_evaluation: np.ndarray, *, split_seed: int,
) -> dict[str, Any]:
    values = np.asarray(X_train, dtype=np.float64)
    evaluation = np.asarray(X_evaluation, dtype=np.float64)
    targets = np.asarray(y_train, dtype=np.float64)
    evaluation_targets = np.asarray(y_evaluation, dtype=np.float64)
    strata = np.asarray(bins, dtype=np.int64)
    if (
        values.ndim != 2 or evaluation.ndim != 2 or values.shape[1] != evaluation.shape[1]
        or targets.shape != (len(values),) or strata.shape != targets.shape
        or evaluation_targets.shape != (len(evaluation),)
        or not all(np.isfinite(array).all() for array in (values, evaluation, targets, evaluation_targets))
    ):
        raise ValueError("four-head evaluation inputs are invalid")
    fold_rows = []
    preprocessing_seconds = 0.0
    splitter = StratifiedKFold(
        n_splits=int(protocol["reference_heads"]["inner_folds"]), shuffle=True, random_state=int(split_seed),
    )
    factors = [float(value) for value in protocol["reference_heads"]["rbf_svr"]["bandwidth_factors"]]
    for fold, (train, holdout) in enumerate(splitter.split(values, strata)):
        preprocessing_started = time.perf_counter()
        Z_train, Z_holdout, diagnostic = information._fit_preprocessor(
            values[train], values[holdout], row_l2=True, pca_rank=None,
            seed=base._seed("early-mean-lsp-head-preprocess", split_seed, fold),
        )
        preprocessing_seconds += float(time.perf_counter() - preprocessing_started)
        rbf_kernels, median = smooth._distance_views(Z_train, Z_holdout, factors)
        quadratic_train, quadratic_holdout = _quadratic_kernels(Z_train, Z_holdout)
        fold_rows.append(
            {
                "Z_train": Z_train, "Z_holdout": Z_holdout,
                "y_train": targets[train], "y_holdout": targets[holdout],
                "rbf_kernels": rbf_kernels,
                "quadratic_train": quadratic_train, "quadratic_holdout": quadratic_holdout,
                "preprocessing": diagnostic, "median_distance": median,
            }
        )
    preprocessing_started = time.perf_counter()
    Z_full, Z_evaluation, full_preprocessing = information._fit_preprocessor(
        values, evaluation, row_l2=True, pca_rank=None,
        seed=base._seed("early-mean-lsp-head-preprocess", split_seed, "full"),
    )
    preprocessing_seconds += float(time.perf_counter() - preprocessing_started)
    full_rbf_kernels, full_median = smooth._distance_views(Z_full, Z_evaluation, factors)
    full_quadratic_train, full_quadratic_evaluation = _quadratic_kernels(Z_full, Z_evaluation)
    results: dict[str, Any] = {}

    config = protocol["reference_heads"]["linear_ridge"]
    started = time.perf_counter()
    rows = []
    for value in config["mean_loss_lambda"]:
        fold_scores = []
        warning_types = set()
        for fold in fold_rows:
            estimator = Ridge(
                alpha=len(fold["Z_train"]) * float(value), solver=str(config["solver"]), tol=float(config["tolerance"]),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(fold["Z_train"], fold["y_train"])
                prediction = estimator.predict(fold["Z_holdout"])
            warning_types.update(type(item.message).__name__ for item in caught)
            fold_scores.append(float(r2_score(fold["y_holdout"], prediction)))
        rows.append({"params": {"mean_loss_lambda": float(value)}, "fold_r2": fold_scores, "mean_r2": float(np.mean(fold_scores)), "warning_types": sorted(warning_types)})
    best = _best(rows)
    estimator = Ridge(
        alpha=len(Z_full) * float(best["params"]["mean_loss_lambda"]), solver=str(config["solver"]), tol=float(config["tolerance"]),
    )
    estimator.fit(Z_full, targets)
    prediction = estimator.predict(Z_evaluation)
    results["linear_ridge"] = {
        "best_params": dict(best["params"]), "inner_best_r2": float(best["mean_r2"]),
        "evaluation_r2": float(r2_score(evaluation_targets, prediction)), "parameter_rows": rows,
        "wall_seconds_excluding_shared_preprocessing": float(time.perf_counter() - started),
    }

    config = protocol["reference_heads"]["quadratic_kernel_ridge"]
    lambdas = [float(value) for value in config["mean_loss_lambda"]]
    started = time.perf_counter()
    fold_predictions = [
        _kernel_ridge_predictions(fold["quadratic_train"], fold["quadratic_holdout"], fold["y_train"], lambdas)
        for fold in fold_rows
    ]
    rows = [
        {
            "params": {"mean_loss_lambda": value},
            "fold_r2": [float(r2_score(fold["y_holdout"], fold_predictions[position][index])) for position, fold in enumerate(fold_rows)],
        }
        for index, value in enumerate(lambdas)
    ]
    for row in rows:
        row["mean_r2"] = float(np.mean(row["fold_r2"]))
    best = _best(rows)
    full_predictions = _kernel_ridge_predictions(
        full_quadratic_train, full_quadratic_evaluation, targets, [float(best["params"]["mean_loss_lambda"])],
    )[0]
    results["quadratic_kernel_ridge"] = {
        "best_params": dict(best["params"]), "inner_best_r2": float(best["mean_r2"]),
        "evaluation_r2": float(r2_score(evaluation_targets, full_predictions)), "parameter_rows": rows,
        "wall_seconds_excluding_shared_preprocessing": float(time.perf_counter() - started),
    }

    config = protocol["reference_heads"]["knn"]
    started = time.perf_counter()
    rows = []
    for neighbors in config["n_neighbors"]:
        scores = []
        for fold in fold_rows:
            estimator = KNeighborsRegressor(n_neighbors=int(neighbors), weights=str(config["weights"]), n_jobs=1)
            estimator.fit(fold["Z_train"], fold["y_train"])
            scores.append(float(r2_score(fold["y_holdout"], estimator.predict(fold["Z_holdout"]))))
        rows.append({"params": {"n_neighbors": int(neighbors)}, "fold_r2": scores, "mean_r2": float(np.mean(scores))})
    best = _best(rows)
    estimator = KNeighborsRegressor(n_neighbors=int(best["params"]["n_neighbors"]), weights=str(config["weights"]), n_jobs=1)
    estimator.fit(Z_full, targets)
    prediction = estimator.predict(Z_evaluation)
    results["knn"] = {
        "best_params": dict(best["params"]), "inner_best_r2": float(best["mean_r2"]),
        "evaluation_r2": float(r2_score(evaluation_targets, prediction)), "parameter_rows": rows,
        "wall_seconds_excluding_shared_preprocessing": float(time.perf_counter() - started),
    }

    started = time.perf_counter()
    rows = []
    for params in _rbf_grid(protocol):
        scores = []
        warning_types = set()
        for fold in fold_rows:
            train_kernel, holdout_kernel = fold["rbf_kernels"][float(params["bandwidth_factor"])]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                prediction = smooth._fit_precomputed(train_kernel, fold["y_train"], holdout_kernel, params)
            warning_types.update(type(item.message).__name__ for item in caught)
            scores.append(float(r2_score(fold["y_holdout"], prediction)))
        rows.append({"params": params, "fold_r2": scores, "mean_r2": float(np.mean(scores)), "warning_types": sorted(warning_types)})
    best = _best(rows)
    train_kernel, evaluation_kernel = full_rbf_kernels[float(best["params"]["bandwidth_factor"])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prediction = smooth._fit_precomputed(train_kernel, targets, evaluation_kernel, best["params"])
    results["rbf_svr"] = {
        "best_params": dict(best["params"]), "inner_best_r2": float(best["mean_r2"]),
        "evaluation_r2": float(r2_score(evaluation_targets, prediction)), "parameter_rows": rows,
        "final_warning_types": sorted({type(item.message).__name__ for item in caught}),
        "wall_seconds_excluding_shared_preprocessing": float(time.perf_counter() - started),
    }
    return {
        "heads": results,
        "fold_preprocessing": [fold["preprocessing"] for fold in fold_rows],
        "full_preprocessing": full_preprocessing,
        "full_median_nonzero_training_distance": full_median,
        "shared_preprocessing_wall_seconds": preprocessing_seconds,
    }


def evaluate_broad_only(
    X: np.ndarray, y: np.ndarray, panel_seed: int, candidate_position: int,
) -> dict[str, Any]:
    values = np.asarray(X, dtype=np.float32)
    targets = np.asarray(y, dtype=np.float64)
    primitive = np.zeros((len(values), 2), dtype=np.float32)
    if values.ndim != 2 or targets.shape != (len(values),) or not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("broad-only inputs are invalid")
    started = time.perf_counter()
    folds = []
    splitter = KFold(n_splits=4, shuffle=True, random_state=int(panel_seed))
    for fold, (train, holdout) in enumerate(splitter.split(values)):
        fit_seed = shared.null_parent._candidate_fit_seed(panel_seed, candidate_position, fold)
        rff_train, rff_holdout, diagnostics = shared.multiview._single_block(
            "rff_mean", values[train], values[holdout], primitive[train], primitive[holdout], targets[train], fit_seed,
        )
        center = float(np.mean(targets[train])); scale = float(np.std(targets[train]))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("broad-only training target is constant")
        bank_seed = shared.null_parent._common_bank_seed(panel_seed, fold)
        bank = shared.null_parent._permutation_bank(len(holdout), 32, bank_seed)
        y_train_scaled = ((targets[train] - center) / scale)[:, None]
        y_holdout_scaled = ((targets[holdout] - center) / scale)[:, None]
        kernel = shared.kernel._kernel_fold_components(
            rff_train, rff_holdout, y_train_scaled, y_holdout_scaled, bank,
        )
        folds.append({"fold": fold, "fit_seed": fit_seed, "bank_seed": bank_seed, "geometry": diagnostics, "kernel_alignment": kernel})
    aggregate = shared._aggregate_kernel(folds)
    return {
        "score": float(aggregate["feature_bandwidths"]["4.0"]["score"]),
        "kernel_alignment": aggregate,
        "folds": folds,
        "fitted_selector_count": 0,
        "wall_seconds": float(time.perf_counter() - started),
    }


def evaluate_cell(
    protocol: Mapping[str, Any], cohort: Mapping[str, Any], matrices: Mapping[str, Mapping[str, np.ndarray]],
    draw_seed: int, view: str, backbone: str, candidate_position: int,
) -> dict[str, Any]:
    draw, evaluation = _draw_rows(protocol, cohort, draw_seed)
    matrix = matrices[backbone][view]
    train_indices = np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)
    evaluation_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    X_train = np.asarray(matrix[train_indices], dtype=np.float32)
    X_evaluation = np.asarray(matrix[evaluation_indices], dtype=np.float32)
    y_train = np.asarray([float(row["target"]) for row in draw], dtype=np.float64)
    y_evaluation = np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64)
    bins = np.asarray([int(row["target_bin"]) for row in draw], dtype=np.int64)
    split_seed = base._seed("early-mean-lsp-holdout-head", draw_seed, backbone, view)
    heads = evaluate_heads(protocol, X_train, y_train, bins, X_evaluation, y_evaluation, split_seed=split_seed)
    evaluation_scores = {head: float(row["evaluation_r2"]) for head, row in heads["heads"].items()}
    envelope = max(evaluation_scores.values())
    panel_seed = _panel_seed("early-mean-lsp-holdout-oi", draw_seed, view)
    primitive = np.zeros((len(X_train), 2), dtype=np.float32)
    observed = shared.save.evaluate_selector("observed_k4", X_train, primitive, y_train, panel_seed)
    broad = evaluate_broad_only(X_train, y_train, panel_seed, candidate_position)
    combined = shared.evaluate_shared_pair(X_train, primitive, y_train, panel_seed, candidate_position)
    if abs(float(broad["score"]) - float(combined["broad_rff"]["score"])) > 1.0e-12:
        raise ValueError("prototype-free broad score differs from the locked shared implementation")
    linear_runtime = float(heads["shared_preprocessing_wall_seconds"]) + float(
        heads["heads"]["linear_ridge"]["wall_seconds_excluding_shared_preprocessing"]
    )
    raw_cv_runtime = float(heads["shared_preprocessing_wall_seconds"]) + sum(
        float(row["wall_seconds_excluding_shared_preprocessing"]) for row in heads["heads"].values()
    )
    return {
        "cell_id": f"{protocol['target']['target_key']}::{draw_seed}::{view}::{backbone}",
        "target_key": str(protocol["target"]["target_key"]),
        "draw_seed": int(draw_seed), "view_id": view, "candidate_id": backbone,
        "training_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in draw]),
        "evaluation_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in evaluation]),
        "truth": {
            "heldout_best_head_envelope_r2": float(envelope),
            "heldout_linear_r2": evaluation_scores["linear_ridge"],
            "head_evaluation_r2": evaluation_scores,
            "best_head_ids": [head for head, score in evaluation_scores.items() if score >= envelope - TIE_ATOL],
        },
        "scores": {
            "linear_cv_full": float(heads["heads"]["linear_ridge"]["inner_best_r2"]),
            "raw_cv_max_full": float(max(row["inner_best_r2"] for row in heads["heads"].values())),
            "observed_oi_full": float(observed["score"]),
            "broad_rff_full": float(broad["score"]),
            "save_rff_full": float(combined["save_rff_current"]["score"]),
        },
        "runtime_seconds": {
            "linear_cv_full": linear_runtime,
            "raw_cv_max_full": raw_cv_runtime,
            "observed_oi_full": float(observed["wall_seconds"]),
            "broad_rff_full": float(broad["wall_seconds"]),
            "save_rff_full": float(combined["wall_seconds"]),
        },
        "heads": heads,
        "continuous": {"observed_oi_full": observed, "broad_rff_full": broad, "save_rff_full": combined},
    }


def _ranking_metrics(candidate_ids: Sequence[str], truth: Sequence[float], scores: Sequence[float]) -> dict[str, Any]:
    return prior_oi.ranking_metrics(candidate_ids, truth, scores)


def _panel_rows(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    order = [str(row["backbone_id"]) for row in protocol["backbones"]]
    output = []
    for draw_seed in protocol["target"]["draw_seeds"]:
        for view in protocol["views"]:
            selected = [row for row in cells if int(row["draw_seed"]) == int(draw_seed) and row["view_id"] == view]
            by_id = {str(row["candidate_id"]): row for row in selected}
            if set(by_id) != set(order):
                raise ValueError("LSP holdout ranking panel is incomplete")
            rows = [by_id[candidate] for candidate in order]
            truth = [float(row["truth"]["heldout_best_head_envelope_r2"]) for row in rows]
            heldout_linear = [float(row["truth"]["heldout_linear_r2"]) for row in rows]
            if len(np.unique(truth)) < 2 or len(np.unique(heldout_linear)) < 2:
                reference_rho = None
            else:
                reference_rho = float(spearmanr(heldout_linear, truth).statistic)
            linear_best = np.flatnonzero(np.asarray(heldout_linear) >= np.max(heldout_linear) - TIE_ATOL)
            reference_regret = float(np.max(truth) - np.max(np.asarray(truth)[linear_best]))
            maximum_gain = float(np.max(np.asarray(truth) - np.asarray(heldout_linear)))
            reference = {
                "linear_to_envelope_spearman": reference_rho,
                "linear_top1_regret_r2": reference_regret,
                "maximum_nonlinear_gain_r2": maximum_gain,
                "material_rank_conflict": bool(
                    reference_regret >= float(protocol["ranking"]["material_linear_top1_regret_r2"])
                    or (reference_rho is not None and reference_rho <= float(protocol["ranking"]["material_linear_to_envelope_spearman_upper"]))
                ),
                "material_nonlinear_gain": maximum_gain >= float(protocol["ranking"]["material_nonlinear_gain_r2"]),
            }
            for method in protocol["continuous_ranking"]["methods"]:
                output.append(
                    {
                        "draw_seed": int(draw_seed), "view_id": view, "method": method,
                        "reference": reference,
                        **_ranking_metrics(order, truth, [float(row["scores"][method]) for row in rows]),
                    }
                )
    return output


def _summaries(protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for view in protocol["views"]:
        for method in protocol["continuous_ranking"]["methods"]:
            rows = [row for row in panels if row["view_id"] == view and row["method"] == method]
            rhos = [float(row["spearman"]) for row in rows if row["spearman"] is not None]
            pairs = [float(row["material_pair_accuracy"]) for row in rows if row["material_pair_accuracy"] is not None]
            output.append(
                {
                    "view_id": view, "method": method, "panel_count": len(rows),
                    "defined_spearman_count": len(rhos),
                    "mean_spearman": float(np.mean(rhos)) if rhos else None,
                    "mean_top1_regret_r2": float(np.mean([row["top1_regret_r2"] for row in rows])),
                    "exact_best_rate": float(np.mean([row["exact_best"] for row in rows])),
                    "mean_material_pair_accuracy": float(np.mean(pairs)) if pairs else None,
                    "mean_top3_utility_regret_r2": float(np.mean([row["top3_mean_utility_regret_r2"] for row in rows])),
                    "mean_top3_set_recall": float(np.mean([row["top3_set_recall"] for row in rows])),
                }
            )
    return output


def summarize(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    panels = _panel_rows(protocol, cells)
    if len(panels) != int(protocol["expected_grid"]["ranking_row_count"]):
        raise ValueError("LSP holdout ranking row count changed")
    summaries = _summaries(protocol, panels)
    primary_view = str(protocol["primary_view"])
    reference_rows = [row for row in panels if row["view_id"] == primary_view and row["method"] == "linear_cv_full"]
    conflicts = sum(bool(row["reference"]["material_rank_conflict"]) for row in reference_rows)
    gains = sum(bool(row["reference"]["material_nonlinear_gain"]) for row in reference_rows)
    opportunity = (
        conflicts >= int(protocol["gates"]["early_mean_opportunity"]["minimum_conflicting_draws"])
        and gains >= int(protocol["gates"]["early_mean_opportunity"]["minimum_material_nonlinear_draws"])
    )
    by = {(row["view_id"], row["method"]): row for row in summaries}
    primary = by[(primary_view, str(protocol["continuous_ranking"]["locked_primary_method"]))]
    linear = by[(primary_view, "linear_cv_full")]
    gate = protocol["gates"]
    competitive = {
        "top1_regret": primary["mean_top1_regret_r2"] <= linear["mean_top1_regret_r2"] + float(gate["primary_method_competitive"]["mean_top1_regret_margin_over_linear_maximum_r2"]),
        "spearman": primary["mean_spearman"] is not None and linear["mean_spearman"] is not None and primary["mean_spearman"] >= linear["mean_spearman"] - float(gate["primary_method_competitive"]["mean_spearman_margin_below_linear_maximum"]),
    }
    runtime_rows = []
    for view in protocol["views"]:
        selected = [row for row in cells if row["view_id"] == view]
        linear_total = float(sum(row["runtime_seconds"]["linear_cv_full"] for row in selected))
        for method in protocol["continuous_ranking"]["methods"]:
            total = float(sum(row["runtime_seconds"][method] for row in selected))
            runtime_rows.append(
                {"view_id": view, "method": method, "aggregate_seconds": total, "relative_to_linear_cv": float(total / linear_total)}
            )
    primary_runtime = next(row for row in runtime_rows if row["view_id"] == primary_view and row["method"] == protocol["continuous_ranking"]["locked_primary_method"])
    competitive["runtime"] = primary_runtime["relative_to_linear_cv"] <= float(gate["primary_method_competitive"]["aggregate_runtime_relative_to_linear_cv_maximum"])
    advantages = {
        "top1_regret": linear["mean_top1_regret_r2"] - primary["mean_top1_regret_r2"] >= float(gate["primary_method_nonlinear_advantage_any_of"]["mean_top1_regret_improvement_over_linear_minimum_r2"]),
        "exact_best_rate": primary["exact_best_rate"] - linear["exact_best_rate"] >= float(gate["primary_method_nonlinear_advantage_any_of"]["exact_best_rate_improvement_over_linear_minimum"]),
    }
    warning_types = sorted(
        {
            warning
            for cell in cells
            for head in cell["heads"]["heads"].values()
            for warning in (
                list(head.get("final_warning_types", []))
                + [item for row in head["parameter_rows"] for item in row.get("warning_types", [])]
            )
        }
    )
    passed = bool(opportunity and all(competitive.values()) and any(advantages.values()))
    return {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_lsp_holdout_v1",
        "ranking_rows": panels,
        "ranking_summaries": summaries,
        "runtime_summaries": runtime_rows,
        "reference_opportunity": {"material_rank_conflict_draws": conflicts, "material_nonlinear_gain_draws": gains, "pass": opportunity},
        "warning_types": warning_types,
        "decision": {
            "locked_primary_method": protocol["continuous_ranking"]["locked_primary_method"],
            "competitive_checks": competitive,
            "nonlinear_advantage_checks": advantages,
            "primary_gate_pass": passed,
            "method_promotion": "none_experiment_only",
            "next_stage": "wholly_new_dataset_replication" if passed else "diagnose_without_holdout_retuning",
        },
        "overlapindex_fit_count": len(cells) * 8,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Early-mean LSP fresh-evaluation benchmark", "",
        "| View | Method | Mean rho | Mean top-1 regret | Exact best | Pair accuracy | Relative time |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    runtimes = {(row["view_id"], row["method"]): row for row in summary["runtime_summaries"]}
    for row in summary["ranking_summaries"]:
        rho = "undefined" if row["mean_spearman"] is None else f"{row['mean_spearman']:.3f}"
        pair = "undefined" if row["mean_material_pair_accuracy"] is None else f"{row['mean_material_pair_accuracy']:.3f}"
        relative = runtimes[(row["view_id"], row["method"])]["relative_to_linear_cv"]
        lines.append(
            f"| {row['view_id']} | {row['method']} | {rho} | {row['mean_top1_regret_r2']:.4f} | "
            f"{row['exact_best_rate']:.3f} | {pair} | {relative:.2f}x |"
        )
    lines.extend(
        [
            "", "## Frozen decision", "",
            f"- Early-mean opportunity: **{str(summary['reference_opportunity']['pass']).lower()}** "
            f"({summary['reference_opportunity']['material_rank_conflict_draws']}/5 conflicts; "
            f"{summary['reference_opportunity']['material_nonlinear_gain_draws']}/5 nonlinear-gain draws).",
            f"- Locked primary method passed: **{str(summary['decision']['primary_gate_pass']).lower()}**.",
            f"- Competitive checks: `{json.dumps(summary['decision']['competitive_checks'], sort_keys=True)}`.",
            f"- Nonlinear-advantage checks: `{json.dumps(summary['decision']['nonlinear_advantage_checks'], sort_keys=True)}`.",
            f"- Warning types: `{json.dumps(summary['warning_types'])}`.",
            "- No method is promoted by this single-target experiment.", "",
        ]
    )
    return "\n".join(lines)


def _validate_grid(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> None:
    expected = int(protocol["expected_grid"]["candidate_cell_count"])
    identities = {str(row["cell_id"]) for row in cells}
    if len(cells) != expected or len(identities) != expected:
        raise ValueError("LSP holdout candidate grid changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    output_root = output_root.resolve()
    protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("LSP holdout outcomes require a clean committed source tree")
    cohort = prepare.load_cohort()
    extract.validate_bank()
    source_hashes = _source_hashes()
    identity = {
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("LSP holdout terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    checkpoint_path = output_root / "checkpoint.json"
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError("LSP holdout checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("LSP holdout checkpoint identity changed")
    else:
        checkpoint = {"schema_version": 1, "identity": identity, "cells": []}
    output_root.mkdir(parents=True, exist_ok=True)
    _manifest, matrices = extract.load_bank()
    complete = {str(row["cell_id"]) for row in checkpoint["cells"]}
    expected = int(protocol["expected_grid"]["candidate_cell_count"])
    for draw_seed in protocol["target"]["draw_seeds"]:
        for view in protocol["views"]:
            for position, backbone_config in enumerate(protocol["backbones"]):
                backbone = str(backbone_config["backbone_id"])
                cell_id = f"{protocol['target']['target_key']}::{draw_seed}::{view}::{backbone}"
                if cell_id in complete:
                    continue
                checkpoint["cells"].append(
                    evaluate_cell(protocol, cohort, matrices, int(draw_seed), view, backbone, position)
                )
                base._finite_json(checkpoint)
                base._atomic_json(checkpoint_path, checkpoint)
                print(f"LSP holdout cell {len(checkpoint['cells'])}/{expected}: {cell_id}", flush=True)
    _validate_grid(protocol, checkpoint["cells"])
    summary = summarize(protocol, checkpoint["cells"])
    raw = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_lsp_holdout_v1",
        "protocol": protocol,
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_environment": {key: os.environ.get(key) for key in base.THREAD_KEYS},
        },
        "cells": checkpoint["cells"],
        "overlapindex_fit_count": summary["overlapindex_fit_count"],
    }
    base._finite_json(raw); base._finite_json(summary)
    base._atomic_json(output_root / "raw_results.json", raw)
    base._atomic_json(output_root / "summary.json", summary)
    base._atomic_bytes(output_root / "report.md", render_report(summary).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_lsp_holdout_v1",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "checkpoint_identity": identity,
        "artifacts": {name: base.sha256_path(output_root / name) for name in ("checkpoint.json", "raw_results.json", "summary.json", "report.md")},
        "counts": {
            "cells": len(checkpoint["cells"]), "ranking_rows": len(summary["ranking_rows"]),
            "score_rows": len(checkpoint["cells"]) * len(protocol["continuous_ranking"]["methods"]),
            "overlapindex_fits": summary["overlapindex_fit_count"],
        },
        "git": {"commit": identity["git_commit"], "dirty": False},
        "decision": summary["decision"],
    }
    base._atomic_json(output_root / "manifest.json", manifest)
    terminal = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_lsp_holdout_v1",
        "manifest_sha256": base.sha256_path(output_root / "manifest.json"),
        "summary_sha256": base.sha256_path(output_root / "summary.json"),
        "decision": summary["decision"],
    }
    base._atomic_json(terminal_path, terminal)
    return validate_artifacts(output_root)


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    protocol = load_protocol()
    terminal = json.loads((output_root / "terminal.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output_root / "checkpoint.json").read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "completed_continuous_early_mean_lsp_holdout_v1"
        or terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json")
        or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json")
        or terminal.get("decision") != summary.get("decision")
        or manifest.get("source_hashes") != _source_hashes()
        or int(summary.get("overlapindex_fit_count", -1)) != int(protocol["expected_grid"]["overlapindex_fit_count"])
    ):
        raise ValueError("LSP holdout terminal identity changed")
    for name, digest in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != digest:
            raise ValueError(f"LSP holdout artifact changed: {name}")
    _validate_grid(protocol, checkpoint["cells"])
    if len(summary["ranking_rows"]) != int(protocol["expected_grid"]["ranking_row_count"]):
        raise ValueError("LSP holdout summary grid changed")
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume)
    print(prepare.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
