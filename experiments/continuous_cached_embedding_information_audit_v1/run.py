"""Run the frozen cached-embedding information audit."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_real_source_qualification_v1 import extract, prepare
from experiments.continuous_real_source_qualification_v1 import run as real_screen


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_cached_embedding_information_audit_v1" / "audit"
SCREEN_ROOT = ROOT / "artifacts" / "continuous_real_source_qualification_v1" / "screen"
HEADS = ("linear_ridge", "knn", "extra_trees")
TIE_ATOL = 1.0e-12


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or base.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("cached-embedding audit protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_cached_embedding_information_audit_v1_before_outcomes":
        raise ValueError("cached-embedding audit protocol is not frozen")
    if tuple(protocol["heads"]["families"]) != HEADS:
        raise ValueError("cached-embedding audit head roster changed")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "real_source_protocol": prepare.PROTOCOL_PATH,
        "real_source_screen_raw_results": SCREEN_ROOT / "raw_results.json",
        "real_source_screen_manifest": SCREEN_ROOT / "manifest.json",
        "real_source_screen_terminal": SCREEN_ROOT / "terminal.json",
        "embedding_manifest": extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "cohort": prepare.DEFAULT_PREPARED_ROOT / "cohort.json",
        "qualified_library_protocol": ROOT / "experiments" / "continuous_reference_library_qualification_v3" / "protocol.json",
    }


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cached-embedding audit input is missing: {missing}")
    observed = {name: base.sha256_path(path) for name, path in paths.items()}
    ancestry = protocol["ancestry"]
    expected = {
        "real_source_protocol": ancestry["real_source_protocol_sha256"],
        "real_source_screen_raw_results": ancestry["real_source_screen_raw_results_sha256"],
        "real_source_screen_manifest": ancestry["real_source_screen_manifest_sha256"],
        "real_source_screen_terminal": ancestry["real_source_screen_terminal_sha256"],
        "embedding_manifest": ancestry["embedding_manifest_sha256"],
        "qualified_library_protocol": ancestry["qualified_library_protocol_sha256"],
    }
    if any(observed[name] != value for name, value in expected.items()):
        raise ValueError("cached-embedding audit ancestry changed")
    real_screen.validate_artifacts(SCREEN_ROOT)
    return observed


def _load_inputs(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    cohort = prepare.load_cohort()
    embedding_manifest = json.loads((extract.DEFAULT_OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    records = {
        (str(row["dataset_id"]), str(row["backbone_id"])): row
        for row in embedding_manifest["records"]
    }
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for target in protocol["targets"]:
        dataset_id = str(target["target_key"]).split("::", 1)[0]
        matrices[dataset_id] = {}
        for backbone in protocol["backbones"]:
            record = records.get((dataset_id, backbone))
            if record is None:
                raise ValueError(f"cached matrix record is missing: {dataset_id}/{backbone}")
            path = extract.DEFAULT_OUTPUT_ROOT / dataset_id / f"{backbone}.npy"
            if not path.is_file() or base.sha256_path(path) != record["matrix_sha256"]:
                raise ValueError(f"cached matrix identity changed: {dataset_id}/{backbone}")
            matrix = np.load(path, mmap_mode="r", allow_pickle=False)
            if tuple(matrix.shape) != (3000, int(record["dimension"])) or matrix.dtype != np.dtype("float32"):
                raise ValueError(f"cached matrix structure changed: {dataset_id}/{backbone}")
            matrices[dataset_id][backbone] = matrix
    return cohort, matrices, embedding_manifest


def _target_rows(cohort: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    dataset_id, target_id = str(config["target_key"]).split("::", 1)
    rows = cohort["datasets"][dataset_id]["target_cohorts"][target_id]
    draw_600 = base.select_unique_draw(
        rows,
        pool_rows=int(config["training_pool_rows"]),
        draw_rows=int(config["archived_training_rows"]),
        seed=int(config["draw_seed"]),
    )
    draw_2000 = base.select_unique_draw(
        rows,
        pool_rows=int(config["training_pool_rows"]),
        draw_rows=int(config["training_pool_rows"]),
        seed=int(config["draw_seed"]),
    )
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    if len(evaluation) != int(config["evaluation_rows"]):
        raise ValueError(f"evaluation count changed: {config['target_key']}")
    ids_600 = {str(row["sample_id"]) for row in draw_600}
    ids_2000 = {str(row["sample_id"]) for row in draw_2000}
    ids_eval = {str(row["sample_id"]) for row in evaluation}
    if not ids_600.issubset(ids_2000) or ids_2000.intersection(ids_eval):
        raise ValueError(f"nested training/evaluation identity contract changed: {config['target_key']}")
    return dataset_id, draw_600, draw_2000, evaluation


def _fit_preprocessor(
    X_train: np.ndarray,
    X_other: np.ndarray,
    *,
    row_l2: bool,
    pca_rank: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = np.asarray(X_train, dtype=np.float64)
    other = np.asarray(X_other, dtype=np.float64)
    if train.ndim != 2 or other.ndim != 2 or train.shape[1] != other.shape[1]:
        raise ValueError("preprocessing requires aligned finite 2-D feature matrices")
    if not np.isfinite(train).all() or not np.isfinite(other).all():
        raise ValueError("preprocessing requires aligned finite 2-D feature matrices")
    input_dimension = int(train.shape[1])
    if row_l2:
        train = normalize(train, norm="l2", axis=1, copy=True)
        other = normalize(other, norm="l2", axis=1, copy=True)
    retained_variance = None
    if pca_rank is not None:
        rank = int(pca_rank)
        if rank <= 0 or rank >= min(train.shape):
            raise ValueError("PCA rank must be positive and below both training dimensions")
        reducer = PCA(
            n_components=rank,
            svd_solver="randomized",
            random_state=int(seed),
            iterated_power=4,
            n_oversamples=10,
        )
        train = reducer.fit_transform(train)
        other = reducer.transform(other)
        retained_variance = float(np.sum(reducer.explained_variance_ratio_))
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    other = scaler.transform(other)
    if not np.isfinite(train).all() or not np.isfinite(other).all():
        raise ValueError("preprocessing produced nonfinite values")
    return train, other, {
        "training_rows": int(len(train)),
        "input_dimension": input_dimension,
        "output_dimension": int(train.shape[1]),
        "row_l2": bool(row_l2),
        "pca_rank": None if pca_rank is None else int(pca_rank),
        "pca_solver": None if pca_rank is None else "randomized",
        "pca_retained_variance_ratio": retained_variance,
        "post_representation_scaling": "StandardScaler",
    }


def _parameter_grid(head: str, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = protocol["heads"][head]
    if head == "linear_ridge":
        return [{"mean_loss_lambda": float(value)} for value in config["mean_loss_lambda"]]
    if head == "knn":
        return [
            {"n_neighbors": int(k), "weights": str(weight)}
            for k, weight in itertools.product(config["n_neighbors"], config["weights"])
        ]
    if head == "extra_trees":
        return [
            {"max_features": value, "min_samples_leaf": int(leaf)}
            for value, leaf in itertools.product(config["max_features"], config["min_samples_leaf"])
        ]
    raise ValueError(f"unknown audit head: {head!r}")


def _distance_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_other: np.ndarray,
    neighbors: Sequence[int],
) -> dict[tuple[int, str], np.ndarray]:
    maximum = max(int(value) for value in neighbors)
    if maximum > len(X_train):
        raise ValueError("k-nearest-neighbor grid exceeds the training fold")
    distances = pairwise_distances(X_other, X_train, metric="euclidean", n_jobs=1)
    order = np.argsort(distances, axis=1, kind="stable")[:, :maximum]
    ordered_distances = np.take_along_axis(distances, order, axis=1)
    ordered_targets = np.asarray(y_train, dtype=np.float64)[order]
    predictions: dict[tuple[int, str], np.ndarray] = {}
    for k in neighbors:
        k = int(k)
        local_distances = ordered_distances[:, :k]
        local_targets = ordered_targets[:, :k]
        predictions[(k, "uniform")] = np.mean(local_targets, axis=1)
        zero = local_distances == 0.0
        weights = np.divide(1.0, local_distances, out=np.zeros_like(local_distances), where=~zero)
        has_zero = np.any(zero, axis=1)
        if np.any(has_zero):
            weights[has_zero] = zero[has_zero].astype(np.float64)
        predictions[(k, "distance")] = np.sum(weights * local_targets, axis=1) / np.sum(weights, axis=1)
    return predictions


def _best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    values = np.asarray([float(row["mean_r2"]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("head tuning produced a nonfinite score")
    return rows[int(np.flatnonzero(values >= np.max(values) - TIE_ATOL)[0])]


def _endpoint_diagnostics(head: str, best: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for parameter, selected in best["params"].items():
        if isinstance(selected, str):
            continue
        values = sorted({float(row["params"][parameter]) for row in rows})
        if len(values) < 2:
            continue
        endpoint = "minimum" if float(selected) == values[0] else "maximum" if float(selected) == values[-1] else None
        if endpoint is not None:
            interior = [float(row["mean_r2"]) for row in rows if float(row["params"][parameter]) != float(selected)]
            output.append({
                "head": head,
                "parameter": parameter,
                "endpoint": endpoint,
                "selected_value": float(selected),
                "inner_cv_advantage_r2": float(float(best["mean_r2"]) - max(interior)),
            })
    return output


def evaluate_cell(
    X_train: np.ndarray,
    y_train: np.ndarray,
    bins: np.ndarray,
    sample_ids: Sequence[str],
    X_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    *,
    view: Mapping[str, Any],
    split_seed: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    X_train = np.asarray(X_train, dtype=np.float64)
    X_evaluation = np.asarray(X_evaluation, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_evaluation = np.asarray(y_evaluation, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.int64)
    ids = tuple(str(value) for value in sample_ids)
    if X_train.shape[0] != len(y_train) or bins.shape != y_train.shape or len(ids) != len(y_train):
        raise ValueError("audit training rows are misaligned")
    if X_evaluation.shape[0] != len(y_evaluation) or len(ids) != len(set(ids)):
        raise ValueError("audit evaluation rows or training identities are invalid")
    splitter = StratifiedKFold(
        n_splits=int(protocol["heads"]["inner_folds"]), shuffle=True, random_state=int(split_seed)
    )
    fold_views = []
    fold_audit = []
    preprocessing_started = time.perf_counter()
    for fold, (train_position, validation_position) in enumerate(splitter.split(X_train, bins)):
        train_ids = {ids[int(position)] for position in train_position}
        validation_ids = {ids[int(position)] for position in validation_position}
        if train_ids.intersection(validation_ids):
            raise ValueError("audit inner fold leaks a sample identity")
        Z_train, Z_validation, diagnostic = _fit_preprocessor(
            X_train[train_position], X_train[validation_position],
            row_l2=bool(view["row_l2"]), pca_rank=view["pca_rank"],
            seed=base._seed("cached-audit-preprocess", split_seed, fold),
        )
        fold_views.append((Z_train, y_train[train_position], Z_validation, y_train[validation_position]))
        fold_audit.append({
            "fold": fold,
            "train_rows": int(len(train_position)),
            "validation_rows": int(len(validation_position)),
            "identity_overlap_count": 0,
            "train_sample_ids_sha256": base.sha256_payload(sorted(train_ids)),
            "validation_sample_ids_sha256": base.sha256_payload(sorted(validation_ids)),
            "preprocessing": diagnostic,
        })
    Z_full, Z_evaluation, full_diagnostic = _fit_preprocessor(
        X_train, X_evaluation, row_l2=bool(view["row_l2"]), pca_rank=view["pca_rank"],
        seed=base._seed("cached-audit-preprocess", split_seed, "full"),
    )
    preprocessing_seconds = float(time.perf_counter() - preprocessing_started)
    heads: dict[str, Any] = {}

    started = time.perf_counter()
    ridge_rows = []
    ridge_config = protocol["heads"]["linear_ridge"]
    for params in _parameter_grid("linear_ridge", protocol):
        scores, warning_types = [], set()
        for Z_train, fold_y_train, Z_validation, fold_y_validation in fold_views:
            estimator = Ridge(
                alpha=len(Z_train) * float(params["mean_loss_lambda"]),
                solver="lsqr", tol=float(ridge_config["tolerance"]),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always"); estimator.fit(Z_train, fold_y_train)
                prediction = estimator.predict(Z_validation)
            warning_types.update(type(item.message).__name__ for item in caught)
            scores.append(float(r2_score(fold_y_validation, prediction)))
        ridge_rows.append({"params": params, "fold_r2": scores, "mean_r2": float(np.mean(scores)), "warning_types": sorted(warning_types)})
    ridge_best = _best_row(ridge_rows)
    ridge = Ridge(
        alpha=len(Z_full) * float(ridge_best["params"]["mean_loss_lambda"]),
        solver="lsqr", tol=float(ridge_config["tolerance"]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always"); ridge.fit(Z_full, y_train); ridge_prediction = ridge.predict(Z_evaluation)
    heads["linear_ridge"] = _head_result("linear_ridge", ridge_best, ridge_rows, ridge_prediction, y_evaluation, caught, time.perf_counter() - started, len(ridge_rows) * len(fold_views) + 1)

    started = time.perf_counter()
    neighbor_values = [int(value) for value in protocol["heads"]["knn"]["n_neighbors"]]
    fold_neighbors = [
        (_distance_predictions(Z_train, fold_y_train, Z_validation, neighbor_values), fold_y_validation)
        for Z_train, fold_y_train, Z_validation, fold_y_validation in fold_views
    ]
    knn_rows = []
    for params in _parameter_grid("knn", protocol):
        key = (int(params["n_neighbors"]), str(params["weights"]))
        scores = [float(r2_score(fold_y, predictions[key])) for predictions, fold_y in fold_neighbors]
        knn_rows.append({"params": params, "fold_r2": scores, "mean_r2": float(np.mean(scores)), "warning_types": []})
    knn_best = _best_row(knn_rows)
    full_neighbors = _distance_predictions(Z_full, y_train, Z_evaluation, neighbor_values)
    knn_key = (int(knn_best["params"]["n_neighbors"]), str(knn_best["params"]["weights"]))
    heads["knn"] = _head_result("knn", knn_best, knn_rows, full_neighbors[knn_key], y_evaluation, [], time.perf_counter() - started, len(fold_views) + 1)

    started = time.perf_counter()
    tree_rows = []
    tree_config = protocol["heads"]["extra_trees"]
    for position, params in enumerate(_parameter_grid("extra_trees", protocol)):
        scores, warning_types = [], set()
        for fold, (Z_train, fold_y_train, Z_validation, fold_y_validation) in enumerate(fold_views):
            estimator = ExtraTreesRegressor(
                n_estimators=int(tree_config["n_estimators"]),
                max_features=params["max_features"], min_samples_leaf=int(params["min_samples_leaf"]),
                bootstrap=False, n_jobs=1,
                random_state=base._seed("cached-audit-trees", split_seed, position, fold),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always"); estimator.fit(Z_train, fold_y_train); prediction = estimator.predict(Z_validation)
            warning_types.update(type(item.message).__name__ for item in caught)
            scores.append(float(r2_score(fold_y_validation, prediction)))
        tree_rows.append({"params": params, "fold_r2": scores, "mean_r2": float(np.mean(scores)), "warning_types": sorted(warning_types)})
    tree_best = _best_row(tree_rows)
    best_position = tree_rows.index(tree_best)
    trees = ExtraTreesRegressor(
        n_estimators=int(tree_config["n_estimators"]),
        max_features=tree_best["params"]["max_features"], min_samples_leaf=int(tree_best["params"]["min_samples_leaf"]),
        bootstrap=False, n_jobs=1,
        random_state=base._seed("cached-audit-trees", split_seed, best_position, "full"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always"); trees.fit(Z_full, y_train); tree_prediction = trees.predict(Z_evaluation)
    heads["extra_trees"] = _head_result("extra_trees", tree_best, tree_rows, tree_prediction, y_evaluation, caught, time.perf_counter() - started, len(tree_rows) * len(fold_views) + 1)

    return {
        "heads": heads,
        "fold_audit": fold_audit,
        "full_preprocessing": full_diagnostic,
        "shared_preprocessing_wall_seconds": preprocessing_seconds,
    }


def _head_result(
    head: str,
    best: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    prediction: np.ndarray,
    y_evaluation: np.ndarray,
    caught: Sequence[Any],
    seconds: float,
    fit_count: int,
) -> dict[str, Any]:
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != y_evaluation.shape or not np.isfinite(values).all():
        raise ValueError(f"{head} prediction is invalid")
    return {
        "best_params": dict(best["params"]),
        "inner_best_r2": float(best["mean_r2"]),
        "parameter_rows": list(rows),
        "endpoint_diagnostics": _endpoint_diagnostics(head, best, rows),
        "evaluation_r2": float(r2_score(y_evaluation, values)),
        "prediction": values.tolist(),
        "prediction_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "final_warning_types": sorted({type(item.message).__name__ for item in caught}),
        "model_fit_count": int(fit_count),
        "head_wall_seconds_excluding_shared_preprocessing": float(seconds),
    }


def _archived_reference(raw: Mapping[str, Any], target_key: str, backbone: str, draw_seed: int) -> dict[str, Any]:
    panel = next(row for row in raw["panels"] if row["target_key"] == target_key)
    candidate = next(row for row in panel["candidates"] if row["candidate_id"] == backbone)
    draw = next(row for row in candidate["draws"] if int(row["draw_seed"]) == int(draw_seed))
    scores = {head: float(result["evaluation_r2"]) for head, result in draw["heads"].items()}
    envelope = max(scores.values())
    return {
        "draw_seed": int(draw_seed),
        "sample_ids_sha256": draw["sample_ids_sha256"],
        "head_evaluation_r2": scores,
        "envelope_r2": float(envelope),
        "best_head_ids": [head for head, score in scores.items() if score >= envelope - TIE_ATOL],
        "linear_prediction_sha256": draw["heads"]["linear_ridge"]["prediction_sha256"],
    }


def build_cell(
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    view: Mapping[str, Any],
    backbone: str,
    cohort: Mapping[str, Any],
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    archived_raw: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_id, draw_600, draw_2000, evaluation = _target_rows(cohort, config)
    draw = draw_600 if int(view["training_rows"]) == 600 else draw_2000
    matrix = matrices[dataset_id][backbone]
    train_indices = np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)
    evaluation_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    result = evaluate_cell(
        matrix[train_indices], np.asarray([float(row["target"]) for row in draw], dtype=np.float64),
        np.asarray([int(row["target_bin"]) for row in draw], dtype=np.int64),
        [str(row["sample_id"]) for row in draw], matrix[evaluation_indices],
        np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64),
        view=view,
        split_seed=base._seed("cached-information-audit", config["target_key"], backbone, view["view_id"]),
        protocol=protocol,
    )
    scores = {head: float(row["evaluation_r2"]) for head, row in result["heads"].items()}
    envelope = max(scores.values())
    archived = _archived_reference(
        archived_raw, str(config["target_key"]), backbone, int(config["draw_seed"])
    )
    training_sha256 = base.sha256_payload([str(row["sample_id"]) for row in draw])
    if (
        int(view["training_rows"]) == int(config["archived_training_rows"])
        and training_sha256 != archived["sample_ids_sha256"]
    ):
        raise ValueError("cached audit 600-row draw no longer matches the archived reference")
    return {
        "cell_id": f"{config['target_key']}::{backbone}::{view['view_id']}",
        "target_key": str(config["target_key"]),
        "dataset_id": dataset_id,
        "candidate_id": backbone,
        "view": dict(view),
        "training_sample_ids_sha256": training_sha256,
        "evaluation_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in evaluation]),
        "head_evaluation_r2": scores,
        "envelope_r2": float(envelope),
        "best_head_ids": [head for head, score in scores.items() if score >= envelope - TIE_ATOL],
        "archived_reference": archived,
        **result,
    }


def _ranking_summary(cells: Sequence[Mapping[str, Any]], target_key: str, view_id: str) -> dict[str, Any]:
    rows = sorted(
        (row for row in cells if row["target_key"] == target_key and row["view"]["view_id"] == view_id),
        key=lambda row: str(row["candidate_id"]),
    )
    if len(rows) < 2:
        raise ValueError("ranking summary requires at least two candidates")
    linear = np.asarray([row["head_evaluation_r2"]["linear_ridge"] for row in rows], dtype=np.float64)
    envelope = np.asarray([row["envelope_r2"] for row in rows], dtype=np.float64)
    if len(np.unique(linear)) < 2 or len(np.unique(envelope)) < 2:
        correlation = None
        correlation_status = "undefined_constant_ranking"
    else:
        correlation = float(spearmanr(linear, envelope).statistic)
        correlation_status = "defined"
    linear_best = np.flatnonzero(linear >= np.max(linear) - TIE_ATOL)
    regret = float(np.max(envelope) - np.max(envelope[linear_best]))
    return {
        "target_key": target_key,
        "view_id": view_id,
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "linear_r2": float(row["head_evaluation_r2"]["linear_ridge"]),
                "envelope_r2": float(row["envelope_r2"]),
                "nonlinear_gain_r2": float(row["envelope_r2"] - row["head_evaluation_r2"]["linear_ridge"]),
                "best_head_ids": list(row["best_head_ids"]),
            }
            for row in rows
        ],
        "linear_to_envelope_spearman": correlation,
        "linear_to_envelope_spearman_status": correlation_status,
        "linear_top1_regret_r2": regret,
        "maximum_nonlinear_gain_r2": float(np.max(envelope - linear)),
        "linear_best_candidate_ids": [rows[int(position)]["candidate_id"] for position in linear_best],
        "envelope_best_candidate_ids": [rows[int(position)]["candidate_id"] for position in np.flatnonzero(envelope >= np.max(envelope) - TIE_ATOL)],
    }


def _contrast_effect(
    summaries: Mapping[tuple[str, str], Mapping[str, Any]],
    target_key: str,
    source_view: str,
    target_view: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    source = summaries[(target_key, source_view)]
    target = summaries[(target_key, target_view)]
    source_candidates = {row["candidate_id"]: row for row in source["candidates"]}
    candidate_deltas = []
    for row in target["candidates"]:
        prior = source_candidates[row["candidate_id"]]
        candidate_deltas.append({
            "candidate_id": row["candidate_id"],
            "linear_delta_r2": float(row["linear_r2"] - prior["linear_r2"]),
            "envelope_delta_r2": float(row["envelope_r2"] - prior["envelope_r2"]),
            "nonlinear_gain_delta_r2": float(row["nonlinear_gain_r2"] - prior["nonlinear_gain_r2"]),
        })
    maximum_gain = max(row["nonlinear_gain_delta_r2"] for row in candidate_deltas)
    maximum_envelope = max(row["envelope_delta_r2"] for row in candidate_deltas)
    regret_increase = float(target["linear_top1_regret_r2"] - source["linear_top1_regret_r2"])
    target_rho = target["linear_to_envelope_spearman"]
    source_rho = source["linear_to_envelope_spearman"]
    rank_disagreement = bool(
        target_rho is not None
        and target_rho <= float(thresholds["material_linear_to_envelope_spearman_upper"])
        and (source_rho is None or target_rho < source_rho)
    )
    material = bool(
        maximum_gain >= float(thresholds["material_nonlinear_gain_change_r2"])
        or (
            target["linear_top1_regret_r2"] >= float(thresholds["material_linear_probe_regret_r2"])
            and regret_increase >= float(thresholds["material_linear_probe_regret_increase_r2"])
        )
        or rank_disagreement
    )
    return {
        "target_key": target_key,
        "source_view": source_view,
        "target_view": target_view,
        "candidate_deltas": candidate_deltas,
        "maximum_nonlinear_gain_increase_r2": float(maximum_gain),
        "maximum_envelope_improvement_r2": float(maximum_envelope),
        "linear_top1_regret_increase_r2": regret_increase,
        "target_linear_top1_regret_r2": float(target["linear_top1_regret_r2"]),
        "rank_disagreement_became_material": rank_disagreement,
        "material": material,
    }


def summarize(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = {
        (str(target["target_key"]), str(view["view_id"])): _ranking_summary(
            cells, str(target["target_key"]), str(view["view_id"])
        )
        for target in protocol["targets"]
        for view in protocol["evaluated_views"]
    }
    effects = {}
    for effect, contrasts in protocol["effects"]["contrasts"].items():
        effects[effect] = [
            _contrast_effect(summaries, str(target["target_key"]), pair[0], pair[1], protocol["effects"])
            for target in protocol["targets"]
            for pair in contrasts
        ]
    calibration = []
    for target in protocol["targets"]:
        for cell in cells:
            if cell["target_key"] == target["target_key"] and cell["view"]["view_id"] == "n600_l2_pca32":
                delta = float(cell["head_evaluation_r2"]["linear_ridge"] - cell["archived_reference"]["head_evaluation_r2"]["linear_ridge"])
                calibration.append({"target_key": cell["target_key"], "candidate_id": cell["candidate_id"], "linear_score_delta_r2": delta})
    max_calibration = max(abs(row["linear_score_delta_r2"]) for row in calibration)
    material_effects = {name: sum(bool(row["material"]) for row in rows) for name, rows in effects.items()}
    any_material = any(value > 0 for value in material_effects.values())
    calibrated = max_calibration <= float(protocol["effects"]["maximum_calibration_linear_score_delta_r2"])
    return {
        "schema_version": 1,
        "status": "completed_continuous_cached_embedding_information_audit_v1",
        "view_summaries": list(summaries.values()),
        "effect_contrasts": effects,
        "calibration": {
            "rows": calibration,
            "maximum_absolute_linear_score_delta_r2": float(max_calibration),
            "within_tolerance": bool(calibrated),
        },
        "decision": {
            "material_effect_counts": material_effects,
            "any_material_cached_embedding_effect": bool(any_material),
            "calibration_within_tolerance": bool(calibrated),
            "next_stage": (
                "repeat_material_effect_on_all_five_backbones_and_multiple_draws"
                if any_material and calibrated
                else "intermediate_layer_and_pooling_audit"
                if calibrated
                else "resolve_pca_solver_calibration_before_interpretation"
            ),
            "method_promotion": "none_reference_only",
        },
        "overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous cached-embedding information audit V1", "",
        "| Target | View | Linear rho | LP regret | Max nonlinear gain | Linear winner | Envelope winner |", 
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in summary["view_summaries"]:
        lines.append(
            f"| {row['target_key']} | {row['view_id']} | "
            f"{('undefined' if row['linear_to_envelope_spearman'] is None else format(row['linear_to_envelope_spearman'], '.3f'))} | "
            f"{row['linear_top1_regret_r2']:.4f} | {row['maximum_nonlinear_gain_r2']:.4f} | "
            f"{', '.join(row['linear_best_candidate_ids'])} | {', '.join(row['envelope_best_candidate_ids'])} |"
        )
    lines.extend(["", "## Predeclared effects", "", "| Effect | Material contrasts | Total contrasts |", "|---|---:|---:|"])
    for name, rows in summary["effect_contrasts"].items():
        lines.append(f"| {name} | {sum(row['material'] for row in rows)} | {len(rows)} |")
    decision = summary["decision"]
    lines.extend([
        "", "## Decision", "",
        f"- PCA solver calibration within tolerance: **{str(decision['calibration_within_tolerance']).lower()}** "
        f"(maximum absolute linear-score delta {summary['calibration']['maximum_absolute_linear_score_delta_r2']:.4f} R²).",
        f"- Any material cached-embedding effect: **{str(decision['any_material_cached_embedding_effect']).lower()}**.",
        f"- Material contrast counts: `{json.dumps(decision['material_effect_counts'], sort_keys=True)}`.",
        f"- Next stage: **{decision['next_stage']}**.",
        "- This diagnostic promotes no OverlapIndex method.", "",
    ])
    return "\n".join(lines)


def _identity(protocol: Mapping[str, Any], source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "runner_sha256": base.sha256_path(Path(__file__)),
        "source_hashes": dict(source_hashes),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def validate_grid(protocol: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    cells = checkpoint["cells"]
    expected = int(protocol["expected_grid"]["cell_count"])
    if len(cells) != expected or len({row["cell_id"] for row in cells}) != expected:
        raise ValueError("cached-embedding audit executed grid changed")
    wanted = {
        f"{target['target_key']}::{backbone}::{view['view_id']}"
        for target in protocol["targets"]
        for backbone in protocol["backbones"]
        for view in protocol["evaluated_views"]
    }
    if {row["cell_id"] for row in cells} != wanted:
        raise ValueError("cached-embedding audit cell identities changed")
    if any(set(row["heads"]) != set(HEADS) for row in cells):
        raise ValueError("cached-embedding audit head results changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    output_root = output_root.resolve()
    protocol = load_protocol()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError("cached-embedding audit execution requires a clean committed source tree")
    source_hashes = _source_hashes(protocol)
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("cached-embedding audit terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    identity = _identity(protocol, source_hashes)
    checkpoint_path = output_root / "checkpoint.json"
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError("cached-embedding audit checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("cached-embedding audit checkpoint identity changed")
    else:
        checkpoint = {"schema_version": 1, "identity": identity, "cells": []}
    output_root.mkdir(parents=True, exist_ok=True)
    cohort, matrices, embedding_manifest = _load_inputs(protocol)
    archived_raw = json.loads((SCREEN_ROOT / "raw_results.json").read_text(encoding="utf-8"))
    complete = {row["cell_id"] for row in checkpoint["cells"]}
    for target in protocol["targets"]:
        for backbone in protocol["backbones"]:
            for view in protocol["evaluated_views"]:
                cell_id = f"{target['target_key']}::{backbone}::{view['view_id']}"
                if cell_id in complete:
                    continue
                cell = build_cell(protocol, target, view, backbone, cohort, matrices, archived_raw)
                checkpoint["cells"].append(cell)
                base._finite_json(checkpoint)
                base._atomic_json(checkpoint_path, checkpoint)
                print(f"cell {len(checkpoint['cells'])}/{protocol['expected_grid']['cell_count']}: {cell_id}", flush=True)
    validate_grid(protocol, checkpoint)
    summary = summarize(protocol, checkpoint["cells"])
    raw = {
        "schema_version": 1,
        "status": "completed_continuous_cached_embedding_information_audit_v1",
        "protocol": protocol,
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "embedding_record_count": len(embedding_manifest["records"]),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_environment": {key: os.environ.get(key) for key in base.THREAD_KEYS},
        },
        "cells": checkpoint["cells"],
        "overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }
    base._finite_json(raw); base._finite_json(summary)
    raw_path = output_root / "raw_results.json"
    summary_path = output_root / "summary.json"
    report_path = output_root / "report.md"
    base._atomic_json(raw_path, raw)
    base._atomic_json(summary_path, summary)
    base._atomic_bytes(report_path, render_report(summary).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_cached_embedding_information_audit_v1",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "checkpoint_identity": identity,
        "artifacts": {name: base.sha256_path(output_root / name) for name in ("checkpoint.json", "raw_results.json", "summary.json", "report.md")},
        "counts": {"cells": len(checkpoint["cells"]), "overlapindex_fits": 0, "reserved_backbones": 0},
        "git": {"commit": identity["git_commit"], "dirty": False},
        "decision": summary["decision"],
    }
    base._atomic_json(output_root / "manifest.json", manifest)
    terminal = {
        "schema_version": 1,
        "status": "completed_continuous_cached_embedding_information_audit_v1",
        "manifest_sha256": base.sha256_path(output_root / "manifest.json"),
        "summary_sha256": base.sha256_path(summary_path),
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
    if terminal.get("status") != "completed_continuous_cached_embedding_information_audit_v1":
        raise ValueError("cached-embedding audit terminal status changed")
    if terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json") or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json"):
        raise ValueError("cached-embedding audit terminal identity changed")
    if terminal.get("decision") != summary.get("decision") or manifest.get("source_hashes") != _source_hashes(protocol):
        raise ValueError("cached-embedding audit summary or source identity changed")
    for name, expected in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != expected:
            raise ValueError(f"cached-embedding audit artifact changed: {name}")
    validate_grid(protocol, checkpoint)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume)
    print(base.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
