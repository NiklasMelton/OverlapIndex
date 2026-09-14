"""Scale-aware four-head evaluator used by the qualification experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import time
from typing import Any
import warnings

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from experiments.continuous_ranking_benchmark_v2 import run as v2


HEADS = ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
TIE_ATOL = 1.0e-12


def _median_nonzero_distance(X: np.ndarray) -> float:
    distances = pdist(np.asarray(X, dtype=np.float64), metric="euclidean")
    positive = distances[distances > 0.0]
    if not len(positive):
        raise ValueError("RBF bandwidth requires at least one nonzero training distance")
    value = float(np.median(positive))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("RBF median training distance is invalid")
    return value


def parameter_grid(head: str, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = protocol["heads"][head]
    if head in ("linear_ridge", "quadratic_ridge"):
        return [{"mean_loss_lambda": float(value)} for value in config["mean_loss_lambda"]]
    if head == "knn":
        return [
            {"n_neighbors": int(k), "weights": str(weight)}
            for k, weight in itertools.product(config["n_neighbors"], config["weights"])
        ]
    if head == "rbf_svr":
        return [
            {
                "C_standardized": float(C),
                "epsilon_standardized": float(epsilon),
                "bandwidth_factor": float(factor),
            }
            for C, epsilon, factor in itertools.product(
                config["C_standardized_target_units"],
                config["epsilon_standardized_target_units"],
                config["bandwidth_factors"],
            )
        ]
    raise ValueError(f"unknown reference head: {head!r}")


def _ridge(head: str, params: Mapping[str, Any], training_rows: int) -> Ridge:
    if head not in ("linear_ridge", "quadratic_ridge"):
        raise ValueError("Ridge factory received a non-Ridge head")
    alpha = float(training_rows) * float(params["mean_loss_lambda"])
    return Ridge(alpha=alpha, solver="svd")


def _standardize_target(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(y, dtype=np.float64)
    mean = float(np.mean(values))
    scale = float(np.std(values, ddof=0))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("RBF target standardization requires nonconstant finite targets")
    return (values - mean) / scale, mean, scale


def _fit_predict(
    head: str,
    params: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_other: np.ndarray,
    *,
    median_distance: float | None = None,
) -> np.ndarray:
    if head in ("linear_ridge", "quadratic_ridge"):
        estimator = _ridge(head, params, len(X_train))
        estimator.fit(X_train, y_train)
        return np.asarray(estimator.predict(X_other), dtype=np.float64)
    if head == "knn":
        estimator = KNeighborsRegressor(
            n_neighbors=int(params["n_neighbors"]), weights=str(params["weights"])
        )
        estimator.fit(X_train, y_train)
        return np.asarray(estimator.predict(X_other), dtype=np.float64)
    if head == "rbf_svr":
        distance = _median_nonzero_distance(X_train) if median_distance is None else median_distance
        bandwidth = float(params["bandwidth_factor"]) * float(distance)
        gamma = 1.0 / (2.0 * bandwidth * bandwidth)
        standardized, mean, scale = _standardize_target(y_train)
        estimator = SVR(
            kernel="rbf",
            C=float(params["C_standardized"]),
            epsilon=float(params["epsilon_standardized"]),
            gamma=gamma,
        )
        estimator.fit(X_train, standardized)
        return np.asarray(estimator.predict(X_other) * scale + mean, dtype=np.float64)
    raise ValueError(f"unknown reference head: {head!r}")


def endpoint_diagnostics(
    head: str,
    best_params: Mapping[str, Any],
    parameter_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    categorical = set(protocol["grid_adequacy"]["categorical_parameters_ignored"])
    threshold = float(protocol["grid_adequacy"]["material_endpoint_advantage_r2"])
    rows = []
    for parameter, selected in best_params.items():
        if parameter in categorical:
            continue
        values = sorted({float(row["params"][parameter]) for row in parameter_rows})
        if len(values) < 2:
            continue
        endpoint = None
        if float(selected) == values[0]:
            endpoint = "minimum"
        elif float(selected) == values[-1]:
            endpoint = "maximum"
        if endpoint is None:
            continue
        interior = [
            float(row["mean_r2"])
            for row in parameter_rows
            if float(row["params"][parameter]) != float(selected)
        ]
        advantage = float(max(row["mean_r2"] for row in parameter_rows) - max(interior))
        rows.append(
            {
                "parameter": parameter,
                "endpoint": endpoint,
                "selected_value": float(selected),
                "inner_cv_advantage_r2": advantage,
                "material": bool(advantage >= threshold),
            }
        )
    return rows


def evaluate_heads(
    X_train: np.ndarray,
    y_train: np.ndarray,
    bins: np.ndarray,
    sample_ids: Sequence[str],
    X_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    *,
    split_seed: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Tune all heads on common folds with fold-local preprocessing."""
    X_train = np.asarray(X_train, dtype=np.float64)
    X_evaluation = np.asarray(X_evaluation, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_evaluation = np.asarray(y_evaluation, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.int64)
    ids = tuple(str(value) for value in sample_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("head tuning received duplicate sample identities")
    if X_train.shape[0] != len(y_train) or bins.shape != y_train.shape or len(ids) != len(y_train):
        raise ValueError("head-tuning rows are misaligned")
    if X_evaluation.shape[0] != len(y_evaluation):
        raise ValueError("evaluation rows are misaligned")

    folds = int(protocol["heads"]["inner_folds"])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(split_seed))
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    views = []
    fold_audit = []
    preprocessing_started = time.perf_counter()
    for fold, (train_position, validation_position) in enumerate(splitter.split(X_train, bins)):
        train_ids = {ids[int(position)] for position in train_position}
        validation_ids = {ids[int(position)] for position in validation_position}
        if train_ids.intersection(validation_ids):
            raise ValueError("inner fold leaks a sample identity")
        Z_train, Z_validation, diagnostic = v2._fit_preprocessor(
            X_train[train_position], X_train[validation_position]
        )
        views.append(
            {
                "linear_ridge": (Z_train, Z_validation),
                "quadratic_ridge": (
                    polynomial.fit_transform(Z_train), polynomial.transform(Z_validation)
                ),
                "knn": (Z_train, Z_validation),
                "rbf_svr": (Z_train, Z_validation),
                "y_train": y_train[train_position],
                "y_validation": y_train[validation_position],
                "rbf_median_distance": _median_nonzero_distance(Z_train),
            }
        )
        fold_audit.append(
            {
                "fold": fold,
                "train_rows": int(len(train_position)),
                "validation_rows": int(len(validation_position)),
                "identity_overlap_count": 0,
                "train_sample_ids_sha256": v2.sha256_payload(sorted(train_ids)),
                "validation_sample_ids_sha256": v2.sha256_payload(sorted(validation_ids)),
                "preprocessing": diagnostic,
            }
        )
    Z_full, Z_evaluation, full_diagnostic = v2._fit_preprocessor(X_train, X_evaluation)
    full_views = {
        "linear_ridge": (Z_full, Z_evaluation),
        "quadratic_ridge": (polynomial.fit_transform(Z_full), polynomial.transform(Z_evaluation)),
        "knn": (Z_full, Z_evaluation),
        "rbf_svr": (Z_full, Z_evaluation),
    }
    full_rbf_median = _median_nonzero_distance(Z_full)
    preprocessing_seconds = float(time.perf_counter() - preprocessing_started)

    results: dict[str, Any] = {}
    for head in HEADS:
        started = time.perf_counter()
        parameter_rows = []
        for params in parameter_grid(head, protocol):
            fold_scores = []
            warning_types: set[str] = set()
            for view in views:
                train_values, validation_values = view[head]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    prediction = _fit_predict(
                        head,
                        params,
                        train_values,
                        view["y_train"],
                        validation_values,
                        median_distance=(
                            float(view["rbf_median_distance"]) if head == "rbf_svr" else None
                        ),
                    )
                warning_types.update(type(item.message).__name__ for item in caught)
                fold_scores.append(float(r2_score(view["y_validation"], prediction)))
            parameter_rows.append(
                {
                    "params": dict(params),
                    "fold_r2": fold_scores,
                    "mean_r2": float(np.mean(fold_scores)),
                    "warning_types": sorted(warning_types),
                }
            )
        means = np.asarray([row["mean_r2"] for row in parameter_rows], dtype=np.float64)
        if not np.isfinite(means).all():
            raise ValueError(f"{head} inner score is nonfinite")
        best_position = int(np.flatnonzero(means >= np.max(means) - TIE_ATOL)[0])
        best = parameter_rows[best_position]
        train_values, evaluation_values = full_views[head]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prediction = _fit_predict(
                head,
                best["params"],
                train_values,
                y_train,
                evaluation_values,
                median_distance=full_rbf_median if head == "rbf_svr" else None,
            )
        if prediction.shape != y_evaluation.shape or not np.isfinite(prediction).all():
            raise ValueError(f"{head} prediction is invalid")
        results[head] = {
            "best_params": best["params"],
            "inner_best_r2": best["mean_r2"],
            "parameter_rows": parameter_rows,
            "endpoint_diagnostics": endpoint_diagnostics(
                head, best["params"], parameter_rows, protocol
            ),
            "evaluation_r2": float(r2_score(y_evaluation, prediction)),
            "prediction": prediction.tolist(),
            "prediction_sha256": hashlib.sha256(prediction.tobytes(order="C")).hexdigest(),
            "final_warning_types": sorted({type(item.message).__name__ for item in caught}),
            "model_fit_count": len(parameter_rows) * folds + 1,
            "head_wall_seconds_excluding_shared_preprocessing": float(
                time.perf_counter() - started
            ),
        }
    return {
        "heads": results,
        "fold_audit": fold_audit,
        "full_preprocessing": full_diagnostic,
        "full_rbf_median_distance": full_rbf_median,
        "shared_preprocessing_wall_seconds": preprocessing_seconds,
    }

