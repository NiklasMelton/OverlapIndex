"""Run the corrected best-head reference-evaluator pilot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any
import warnings

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, normalize
from sklearn.svm import SVR

from experiments.continuous_ranking_reliability_recoverability import extract as v1_extract
from experiments.continuous_ranking_reliability_recoverability import prepare as v1_prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_ranking_benchmark_v2" / "evaluator_pilot"
V1_RESULT_ROOT = ROOT / "artifacts" / "continuous_ranking_reliability_recoverability" / "results"

BACKBONES = tuple(v1_extract.BACKBONES)
HEADS = ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
SYNTHETIC_CANDIDATES = (
    "clean_latent", "mild_latent_noise", "linear_shortcut", "degraded_latent", "noise_only",
)
TARGET_BINS = 10
INNER_FOLDS = 5
PCA_RANK = 32
TIE_ATOL = 1.0e-12
THREAD_KEYS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def _finite_json(value: Any, label: str = "artifact") -> None:
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{label} contains a nonfinite JSON number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_json(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            _finite_json(item, f"{label}[{position}]")


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("continuous ranking benchmark V2 protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_ranking_benchmark_v2_evaluator_pilot_before_outcomes":
        raise ValueError("continuous ranking benchmark V2 evaluator pilot is not frozen")
    if tuple(protocol["backbones"]) != BACKBONES:
        raise ValueError("evaluator-pilot backbone roster changed")
    if tuple(protocol["heads"]["families"]) != HEADS:
        raise ValueError("evaluator-pilot head roster changed")
    if int(protocol["heads"]["inner_folds"]) != INNER_FOLDS:
        raise ValueError("evaluator-pilot inner-fold count changed")
    if int(protocol["preprocessing"]["pca_rank"]) != PCA_RANK:
        raise ValueError("evaluator-pilot PCA rank changed")
    if tuple(protocol["synthetic_controls"]["candidate_ids"]) != SYNTHETIC_CANDIDATES:
        raise ValueError("evaluator-pilot synthetic candidate roster changed")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL_PATH,
        "protocol_sidecar": PROTOCOL_SHA256_PATH,
        "design": PACKAGE_DIR / "DESIGN.md",
        "runner": Path(__file__).resolve(),
        "v1_input_cohort": v1_prepare.DEFAULT_OUTPUT_ROOT / "cohort.json",
        "v1_input_manifest": v1_prepare.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "v1_embedding_manifest": v1_extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "v1_raw_results": V1_RESULT_ROOT / "raw_results.json",
        "v1_manifest": V1_RESULT_ROOT / "manifest.json",
    }


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"evaluator-pilot provenance input is missing: {missing}")
    observed = {name: sha256_path(path) for name, path in paths.items()}
    ancestry = protocol["ancestry"]
    expected = {
        "v1_input_cohort": ancestry["v1_input_cohort_sha256"],
        "v1_input_manifest": ancestry["v1_input_manifest_sha256"],
        "v1_embedding_manifest": ancestry["v1_embedding_manifest_sha256"],
        "v1_raw_results": ancestry["v1_raw_results_sha256"],
        "v1_manifest": ancestry["v1_manifest_sha256"],
    }
    if any(observed[key] != value for key, value in expected.items()):
        raise ValueError("evaluator-pilot V1 ancestry identity changed")
    return observed


def _target_bins(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or len(y) < TARGET_BINS or not np.isfinite(y).all():
        raise ValueError("target bins require a finite one-dimensional target")
    order = np.argsort(y, kind="mergesort")
    bins = np.empty(len(y), dtype=np.int64)
    bins[order] = np.minimum(TARGET_BINS - 1, np.arange(len(y)) * TARGET_BINS // len(y))
    return bins


def select_unique_draw(
    rows: Sequence[Mapping[str, Any]],
    *,
    pool_rows: int,
    draw_rows: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    """Select a target-balanced draw with distinct original sample identities."""
    training = sorted(
        (row for row in rows if row["split"] == "training"),
        key=lambda row: int(row["budget_rank"]),
    )
    pool = training[:pool_rows]
    if len(pool) != pool_rows or pool_rows % TARGET_BINS or draw_rows % TARGET_BINS:
        raise ValueError("evaluator-pilot training pool or draw size changed")
    sample_ids = [str(row["sample_id"]) for row in pool]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("evaluator-pilot training pool contains duplicate sample identities")
    rng = np.random.default_rng(int(seed))
    selected: list[Mapping[str, Any]] = []
    per_bin = draw_rows // TARGET_BINS
    for target_bin in range(TARGET_BINS):
        candidates = [row for row in pool if int(row["target_bin"]) == target_bin]
        if len(candidates) < per_bin:
            raise ValueError("evaluator-pilot target bin cannot support the unique draw")
        order = rng.permutation(len(candidates))[:per_bin]
        selected.extend(candidates[int(position)] for position in order)
    selected.sort(key=lambda row: (int(row["target_bin"]), str(row["sample_id"])))
    selected_ids = [str(row["sample_id"]) for row in selected]
    if len(selected) != draw_rows or len(set(selected_ids)) != draw_rows:
        raise ValueError("evaluator-pilot draw contains duplicate sample identities")
    return selected


def _fit_preprocessor(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    X_train = np.asarray(X_train, dtype=np.float64)
    X_other = np.asarray(X_other, dtype=np.float64)
    if X_train.ndim != 2 or X_other.ndim != 2 or X_train.shape[1] != X_other.shape[1]:
        raise ValueError("evaluator-pilot feature matrices are not aligned")
    if not np.isfinite(X_train).all() or not np.isfinite(X_other).all():
        raise ValueError("evaluator-pilot feature matrices are nonfinite")
    normalized_train = normalize(X_train, norm="l2", copy=True)
    normalized_other = normalize(X_other, norm="l2", copy=True)
    rank = min(PCA_RANK, X_train.shape[1], len(X_train) - 1)
    if rank < 1:
        raise ValueError("evaluator-pilot PCA rank is empty")
    pca = PCA(n_components=rank, svd_solver="full")
    reduced_train = pca.fit_transform(normalized_train)
    reduced_other = pca.transform(normalized_other)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(reduced_train)
    scaled_other = scaler.transform(reduced_other)
    return scaled_train, scaled_other, {
        "rank": rank,
        "input_dimension": int(X_train.shape[1]),
        "fit_rows": int(len(X_train)),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "mean_sha256": hashlib.sha256(np.ascontiguousarray(pca.mean_).tobytes()).hexdigest(),
        "components_sha256": hashlib.sha256(np.ascontiguousarray(pca.components_).tobytes()).hexdigest(),
    }


def _parameter_grid(head: str, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = protocol["heads"][head]
    if head in ("linear_ridge", "quadratic_ridge"):
        return [{"alpha": float(alpha)} for alpha in config["alpha"]]
    if head == "knn":
        return [
            {"n_neighbors": int(k), "weights": str(weight)}
            for k, weight in itertools.product(config["n_neighbors"], config["weights"])
        ]
    if head == "rbf_svr":
        return [
            {"C": float(C), "gamma": gamma, "epsilon": float(epsilon)}
            for C, gamma, epsilon in itertools.product(config["C"], config["gamma"], config["epsilon"])
        ]
    raise ValueError(f"unknown evaluator-pilot head: {head!r}")


def _model(head: str, params: Mapping[str, Any]) -> Any:
    if head in ("linear_ridge", "quadratic_ridge"):
        return Ridge(alpha=float(params["alpha"]), solver="svd")
    if head == "knn":
        return KNeighborsRegressor(
            n_neighbors=int(params["n_neighbors"]), weights=str(params["weights"])
        )
    if head == "rbf_svr":
        return SVR(
            kernel="rbf", C=float(params["C"]), gamma=params["gamma"],
            epsilon=float(params["epsilon"]),
        )
    raise ValueError(f"unknown evaluator-pilot head: {head!r}")


def evaluate_heads_corrected(
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
    """Tune four heads on common folds with every transform fitted fold-locally."""
    y_train = np.asarray(y_train, dtype=np.float64)
    y_evaluation = np.asarray(y_evaluation, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.int64)
    ids = tuple(str(value) for value in sample_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("evaluator-pilot head tuning received duplicate sample identities")
    if X_train.shape[0] != len(y_train) or bins.shape != y_train.shape or len(ids) != len(y_train):
        raise ValueError("evaluator-pilot head-tuning rows are misaligned")
    if X_evaluation.shape[0] != len(y_evaluation):
        raise ValueError("evaluator-pilot evaluation rows are misaligned")

    splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=int(split_seed))
    views = []
    fold_audit = []
    preprocessing_started = time.perf_counter()
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    for fold, (train_position, validation_position) in enumerate(splitter.split(X_train, bins)):
        train_ids = {ids[int(position)] for position in train_position}
        validation_ids = {ids[int(position)] for position in validation_position}
        if train_ids.intersection(validation_ids):
            raise ValueError("evaluator-pilot inner fold leaks a sample identity")
        Z_train, Z_validation, diagnostic = _fit_preprocessor(
            X_train[train_position], X_train[validation_position]
        )
        Q_train = polynomial.fit_transform(Z_train)
        Q_validation = polynomial.transform(Z_validation)
        views.append(
            {
                "Z_train": Z_train,
                "Z_validation": Z_validation,
                "Q_train": Q_train,
                "Q_validation": Q_validation,
                "y_train": y_train[train_position],
                "y_validation": y_train[validation_position],
            }
        )
        fold_audit.append(
            {
                "fold": fold,
                "train_rows": len(train_position),
                "validation_rows": len(validation_position),
                "identity_overlap_count": 0,
                "train_sample_ids_sha256": sha256_payload(sorted(train_ids)),
                "validation_sample_ids_sha256": sha256_payload(sorted(validation_ids)),
                "preprocessing": diagnostic,
            }
        )
    Z_full, Z_evaluation, full_diagnostic = _fit_preprocessor(X_train, X_evaluation)
    Q_full = polynomial.fit_transform(Z_full)
    Q_evaluation = polynomial.transform(Z_evaluation)
    preprocessing_seconds = float(time.perf_counter() - preprocessing_started)

    results: dict[str, Any] = {}
    for head in HEADS:
        started = time.perf_counter()
        parameter_rows = []
        for params in _parameter_grid(head, protocol):
            fold_scores = []
            warning_types: set[str] = set()
            for view in views:
                train_key = "Q_train" if head == "quadratic_ridge" else "Z_train"
                validation_key = "Q_validation" if head == "quadratic_ridge" else "Z_validation"
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator = _model(head, params)
                    estimator.fit(view[train_key], view["y_train"])
                    prediction = estimator.predict(view[validation_key])
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
            raise ValueError(f"evaluator-pilot {head} inner score is nonfinite")
        best_position = int(np.flatnonzero(means >= np.max(means) - TIE_ATOL)[0])
        best = parameter_rows[best_position]
        full_key = "Q_full" if head == "quadratic_ridge" else "Z_full"
        eval_key = "Q_evaluation" if head == "quadratic_ridge" else "Z_evaluation"
        arrays = {"Q_full": Q_full, "Q_evaluation": Q_evaluation, "Z_full": Z_full, "Z_evaluation": Z_evaluation}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator = _model(head, best["params"])
            estimator.fit(arrays[full_key], y_train)
            prediction = np.asarray(estimator.predict(arrays[eval_key]), dtype=np.float64)
        if prediction.shape != y_evaluation.shape or not np.isfinite(prediction).all():
            raise ValueError(f"evaluator-pilot {head} prediction is invalid")
        results[head] = {
            "best_params": best["params"],
            "inner_best_r2": best["mean_r2"],
            "parameter_rows": parameter_rows,
            "evaluation_r2": float(r2_score(y_evaluation, prediction)),
            "prediction": prediction.tolist(),
            "prediction_sha256": hashlib.sha256(prediction.tobytes(order="C")).hexdigest(),
            "final_warning_types": sorted({type(item.message).__name__ for item in caught}),
            "model_fit_count": len(parameter_rows) * INNER_FOLDS + 1,
            "head_wall_seconds_excluding_shared_preprocessing": float(time.perf_counter() - started),
        }
    return {
        "heads": results,
        "fold_audit": fold_audit,
        "full_preprocessing": full_diagnostic,
        "shared_preprocessing_wall_seconds": preprocessing_seconds,
    }


def ranking_metrics(truth: Sequence[float], score: Sequence[float], candidate_ids: Sequence[str]) -> dict[str, Any]:
    truth_values = np.asarray(truth, dtype=np.float64)
    score_values = np.asarray(score, dtype=np.float64)
    if truth_values.shape != score_values.shape or truth_values.shape != (len(candidate_ids),):
        raise ValueError("evaluator-pilot ranking vectors are misaligned")
    if not np.isfinite(truth_values).all() or not np.isfinite(score_values).all():
        raise ValueError("evaluator-pilot ranking vectors are nonfinite")
    correlation = None
    if np.ptp(truth_values) > TIE_ATOL and np.ptp(score_values) > TIE_ATOL:
        value = float(spearmanr(truth_values, score_values).correlation)
        correlation = value if np.isfinite(value) else None
    selected = np.flatnonzero(score_values >= np.max(score_values) - TIE_ATOL)
    truth_best = np.flatnonzero(truth_values >= np.max(truth_values) - TIE_ATOL)
    return {
        "spearman": correlation,
        "top1_regret_r2": float(np.max(truth_values) - np.max(truth_values[selected])),
        "exact_best": bool(set(selected).intersection(truth_best)),
        "selected_candidate_ids": [str(candidate_ids[position]) for position in selected],
        "truth_best_candidate_ids": [str(candidate_ids[position]) for position in truth_best],
    }


def _reference_uncertainty(panel: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    candidates = panel["candidates"]
    candidate_ids = tuple(row["candidate_id"] for row in candidates)
    draw_count = len(candidates[0]["draws"])
    if any(len(row["draws"]) != draw_count for row in candidates):
        raise ValueError("evaluator-pilot candidate draw grids are misaligned")
    evaluation = np.asarray(panel["evaluation_target"], dtype=np.float64)
    predictions = np.asarray(
        [
            [
                [draw["heads"][head]["prediction"] for head in HEADS]
                for draw in candidate["draws"]
            ]
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    point = np.asarray([row["envelope_r2"] for row in candidates], dtype=np.float64)
    rng = np.random.default_rng(_seed(protocol["reference"]["uncertainty_seed"], panel["target_key"]))
    resamples = int(protocol["reference"]["uncertainty_resamples"])
    boot = np.empty((resamples, len(candidates)), dtype=np.float64)
    for sample in range(resamples):
        draw_ids = rng.integers(0, draw_count, size=draw_count)
        row_ids = rng.integers(0, len(evaluation), size=len(evaluation))
        for candidate in range(len(candidates)):
            head_values = []
            for head in range(len(HEADS)):
                values = [
                    r2_score(evaluation[row_ids], predictions[candidate, draw, head, row_ids])
                    for draw in draw_ids
                ]
                head_values.append(float(np.mean(values)))
            boot[sample, candidate] = max(head_values)
    correlations = [
        float(spearmanr(point, row).correlation)
        for row in boot
        if np.ptp(row) > TIE_ATOL and np.isfinite(spearmanr(point, row).correlation)
    ]
    point_best = set(np.flatnonzero(point >= np.max(point) - TIE_ATOL))
    top1_probability = float(np.mean([
        bool(point_best.intersection(np.flatnonzero(row >= np.max(row) - TIE_ATOL)))
        for row in boot
    ]))
    material = float(protocol["reference"]["material_difference_r2"])
    pair_rows = []
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            delta = float(point[left] - point[right])
            if abs(delta) < material:
                continue
            probability = float(np.mean(np.sign(boot[:, left] - boot[:, right]) == np.sign(delta)))
            pair_rows.append({
                "left": candidate_ids[left], "right": candidate_ids[right],
                "point_delta_r2": delta, "sign_probability": probability,
            })
    median_pair = float(np.median([row["sign_probability"] for row in pair_rows])) if pair_rows else None
    median_rho = float(np.median(correlations)) if correlations else None
    config = protocol["reference"]
    stable = bool(
        median_rho is not None
        and median_rho >= float(config["stable_minimum_median_rank_spearman"])
        and top1_probability >= float(config["stable_minimum_top1_probability"])
        and median_pair is not None
        and median_pair >= float(config["stable_minimum_material_pair_sign_probability"])
    )
    return {
        "resamples": resamples,
        "draw_count": draw_count,
        "evaluation_rows": len(evaluation),
        "median_rank_spearman": median_rho,
        "rank_spearman_interval_95": [
            float(np.quantile(correlations, 0.025)), float(np.quantile(correlations, 0.975))
        ] if correlations else None,
        "point_top1_probability": top1_probability,
        "material_pairs": pair_rows,
        "median_material_pair_sign_probability": median_pair,
        "candidate_envelope_intervals_95": {
            candidate_id: [float(np.quantile(boot[:, position], 0.025)), float(np.quantile(boot[:, position], 0.975))]
            for position, candidate_id in enumerate(candidate_ids)
        },
        "stable": stable,
    }


def build_real_panel(
    protocol: Mapping[str, Any],
    target_config: Mapping[str, Any],
    cohort: Mapping[str, Any],
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    v1_raw: Mapping[str, Any],
) -> dict[str, Any]:
    target_key = str(target_config["target_key"])
    dataset_id, target_id = target_key.split("::", 1)
    target_rows = cohort["datasets"][dataset_id]["target_cohorts"][target_id]
    evaluation = [row for row in target_rows if row["split"] == "evaluation"]
    if len(evaluation) != int(target_config["evaluation_rows"]):
        raise ValueError(f"evaluator-pilot evaluation row count changed: {target_key}")
    evaluation_ids = [str(row["sample_id"]) for row in evaluation]
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError(f"evaluator-pilot evaluation identities repeat: {target_key}")
    y_evaluation = np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64)
    evaluation_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    draws = [
        select_unique_draw(
            target_rows,
            pool_rows=int(target_config["training_pool_rows"]),
            draw_rows=int(target_config["training_rows_per_draw"]),
            seed=int(seed),
        )
        for seed in target_config["draw_seeds"]
    ]
    if any(set(evaluation_ids).intersection(str(row["sample_id"]) for row in draw) for draw in draws):
        raise ValueError(f"evaluator-pilot training/evaluation identities overlap: {target_key}")

    candidates = []
    for backbone in BACKBONES:
        candidate_draws = []
        matrix = matrices[dataset_id][backbone]
        for draw_position, (draw_seed, draw) in enumerate(zip(target_config["draw_seeds"], draws)):
            train_indices = np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)
            y_train = np.asarray([float(row["target"]) for row in draw], dtype=np.float64)
            result = evaluate_heads_corrected(
                matrix[train_indices], y_train,
                np.asarray([int(row["target_bin"]) for row in draw], dtype=np.int64),
                [str(row["sample_id"]) for row in draw],
                matrix[evaluation_indices], y_evaluation,
                split_seed=_seed("real-inner", target_key, backbone, draw_seed),
                protocol=protocol,
            )
            candidate_draws.append({
                "draw": draw_position,
                "draw_seed": int(draw_seed),
                "sample_ids": [str(row["sample_id"]) for row in draw],
                "sample_ids_sha256": sha256_payload([str(row["sample_id"]) for row in draw]),
                "unique_training_rows": len({str(row["sample_id"]) for row in draw}),
                **result,
            })
        head_means = {
            head: float(np.mean([draw["heads"][head]["evaluation_r2"] for draw in candidate_draws]))
            for head in HEADS
        }
        envelope = max(head_means.values())
        candidates.append({
            "candidate_id": backbone,
            "draws": candidate_draws,
            "head_mean_evaluation_r2": head_means,
            "envelope_r2": float(envelope),
            "best_head_ids": [head for head in HEADS if head_means[head] >= envelope - TIE_ATOL],
        })

    v1_panel = next(row for row in v1_raw["reference_panels"] if row["target_key"] == target_key)
    panel = {
        "target_key": target_key,
        "training_pool_rows": int(target_config["training_pool_rows"]),
        "training_rows_per_draw": int(target_config["training_rows_per_draw"]),
        "draw_count": len(draws),
        "evaluation_rows": len(evaluation),
        "evaluation_sample_ids": evaluation_ids,
        "evaluation_target": y_evaluation.tolist(),
        "candidates": candidates,
        "v1_descriptive": {
            "candidate_ids": [row["candidate_id"] for row in v1_panel["candidates"]],
            "linear_r2": [row["head_mean_heldout_r2"]["linear_ridge"] for row in v1_panel["candidates"]],
            "envelope_r2": [row["envelope_r2"] for row in v1_panel["candidates"]],
            "best_head_ids": [row["best_head_ids"] for row in v1_panel["candidates"]],
        },
    }
    panel["uncertainty"] = _reference_uncertainty(panel, protocol)
    envelope = [row["envelope_r2"] for row in candidates]
    linear = [row["head_mean_evaluation_r2"]["linear_ridge"] for row in candidates]
    panel["linear_ranking_metrics"] = ranking_metrics(envelope, linear, BACKBONES)
    return panel


def synthetic_data(family: str, seed: int, protocol: Mapping[str, Any]) -> dict[str, Any]:
    config = protocol["synthetic_controls"]
    training_rows = int(config["training_rows"])
    evaluation_rows = int(config["evaluation_rows"])
    rng = np.random.default_rng(_seed("synthetic", family, seed))
    theta = rng.uniform(-np.pi, np.pi, size=training_rows + evaluation_rows)
    if family == "linear":
        signal = np.cos(theta) + 0.5 * np.sin(theta)
    elif family == "quadratic":
        signal = np.sin(2.0 * theta)
    elif family == "smooth_periodic":
        signal = np.sin(3.0 * theta) + 0.4 * np.cos(5.0 * theta)
    else:
        raise ValueError(f"unknown evaluator-pilot synthetic family: {family!r}")
    y = signal + float(config["target_noise_sd"]) * rng.normal(size=len(theta))

    def circle(angle: np.ndarray) -> np.ndarray:
        return np.asarray(np.column_stack((np.cos(angle), np.sin(angle))), dtype=np.float64)

    amplitude = max(float(np.max(np.abs(signal))), 1.0e-12)
    shortcut = np.clip(
        signal / amplitude + float(config["shortcut_noise_sd"]) * rng.normal(size=len(theta)),
        -0.98, 0.98,
    )
    matrices = {
        "clean_latent": circle(theta),
        "mild_latent_noise": circle(theta + float(config["mild_angle_noise_sd"]) * rng.normal(size=len(theta))),
        "linear_shortcut": np.asarray(np.column_stack((shortcut, np.sqrt(1.0 - shortcut**2))), dtype=np.float64),
        "degraded_latent": circle(theta + float(config["degraded_angle_noise_sd"]) * rng.normal(size=len(theta))),
        "noise_only": circle(rng.uniform(-np.pi, np.pi, size=len(theta))),
    }
    return {"y": y, "matrices": matrices}


def build_synthetic_panel(family: str, seed: int, protocol: Mapping[str, Any]) -> dict[str, Any]:
    data = synthetic_data(family, seed, protocol)
    training_rows = int(protocol["synthetic_controls"]["training_rows"])
    y = np.asarray(data["y"], dtype=np.float64)
    y_train, y_evaluation = y[:training_rows], y[training_rows:]
    bins = _target_bins(y_train)
    sample_ids = [f"synthetic/{family}/{seed}/{position}" for position in range(training_rows)]
    candidates = []
    for candidate_id in SYNTHETIC_CANDIDATES:
        matrix = data["matrices"][candidate_id]
        result = evaluate_heads_corrected(
            matrix[:training_rows], y_train, bins, sample_ids,
            matrix[training_rows:], y_evaluation,
            split_seed=_seed("synthetic-inner", family, seed, candidate_id),
            protocol=protocol,
        )
        head_scores = {head: result["heads"][head]["evaluation_r2"] for head in HEADS}
        envelope = max(head_scores.values())
        candidates.append({
            "candidate_id": candidate_id,
            "heads": result["heads"],
            "head_evaluation_r2": head_scores,
            "envelope_r2": float(envelope),
            "best_head_ids": [head for head in HEADS if head_scores[head] >= envelope - TIE_ATOL],
            "fold_audit": result["fold_audit"],
            "full_preprocessing": result["full_preprocessing"],
            "shared_preprocessing_wall_seconds": result["shared_preprocessing_wall_seconds"],
        })
    envelope = [row["envelope_r2"] for row in candidates]
    linear = [row["head_evaluation_r2"]["linear_ridge"] for row in candidates]
    linear_metrics = ranking_metrics(envelope, linear, SYNTHETIC_CANDIDATES)
    sorted_values = sorted(envelope, reverse=True)
    material = float(protocol["synthetic_controls"]["eligibility"]["minimum_material_candidate_gap_r2"])
    gap_count = sum(
        sorted_values[position] - sorted_values[position + 1] >= material
        for position in range(len(sorted_values) - 1)
    )
    return {
        "family": family,
        "seed": int(seed),
        "training_rows": training_rows,
        "unique_training_rows": len(set(sample_ids)),
        "evaluation_rows": len(y_evaluation),
        "candidates": candidates,
        "linear_ranking_metrics": linear_metrics,
        "best_envelope_r2": float(max(envelope)),
        "material_adjacent_gap_count": int(gap_count),
        "clean_latent_is_best": bool(
            envelope[0] >= max(envelope) - TIE_ATOL
        ),
    }


def summarize(protocol: Mapping[str, Any], real_panels: Sequence[Mapping[str, Any]], synthetic_panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    real_rows = []
    for panel in real_panels:
        values = np.asarray([row["envelope_r2"] for row in panel["candidates"]], dtype=np.float64)
        linear = np.asarray([row["head_mean_evaluation_r2"]["linear_ridge"] for row in panel["candidates"]], dtype=np.float64)
        v1 = panel["v1_descriptive"]
        real_rows.append({
            "target_key": panel["target_key"],
            "draw_count": panel["draw_count"],
            "training_rows_per_draw": panel["training_rows_per_draw"],
            "minimum_unique_training_rows": min(
                draw["unique_training_rows"] for candidate in panel["candidates"] for draw in candidate["draws"]
            ),
            "envelope_best_candidate_ids": [
                BACKBONES[position] for position in np.flatnonzero(values >= np.max(values) - TIE_ATOL)
            ],
            "linear_best_candidate_ids": [
                BACKBONES[position] for position in np.flatnonzero(linear >= np.max(linear) - TIE_ATOL)
            ],
            "linear_to_envelope_spearman": panel["linear_ranking_metrics"]["spearman"],
            "linear_top1_regret_r2": panel["linear_ranking_metrics"]["top1_regret_r2"],
            "maximum_nonlinear_gain_r2": float(np.max(values - linear)),
            "stable": panel["uncertainty"]["stable"],
            "top1_probability": panel["uncertainty"]["point_top1_probability"],
            "v1_linear_to_envelope_spearman": ranking_metrics(v1["envelope_r2"], v1["linear_r2"], BACKBONES)["spearman"],
            "v1_linear_top1_regret_r2": ranking_metrics(v1["envelope_r2"], v1["linear_r2"], BACKBONES)["top1_regret_r2"],
        })

    eligibility = protocol["synthetic_controls"]["eligibility"]
    synthetic_rows = []
    synthetic_all_panel_gates = True
    for panel in synthetic_panels:
        eligible = bool(
            panel["best_envelope_r2"] >= float(eligibility["minimum_best_envelope_r2"])
            and panel["material_adjacent_gap_count"] >= int(eligibility["minimum_material_gap_count"])
        )
        synthetic_all_panel_gates &= eligible
        synthetic_rows.append({
            "family": panel["family"],
            "seed": panel["seed"],
            "eligible_learnability_and_gaps": eligible,
            "best_envelope_r2": panel["best_envelope_r2"],
            "material_adjacent_gap_count": panel["material_adjacent_gap_count"],
            "clean_latent_is_best": panel["clean_latent_is_best"],
            "linear_top1_regret_r2": panel["linear_ranking_metrics"]["top1_regret_r2"],
            "linear_to_envelope_spearman": panel["linear_ranking_metrics"]["spearman"],
        })
    family_rows = []
    family_passes = {}
    for family in protocol["synthetic_controls"]["families"]:
        rows = [row for row in synthetic_rows if row["family"] == family]
        clean_probability = float(np.mean([row["clean_latent_is_best"] for row in rows]))
        mean_regret = float(np.mean([row["linear_top1_regret_r2"] for row in rows]))
        if family == "linear":
            passed = bool(mean_regret <= float(eligibility["maximum_linear_control_top1_regret_r2"]))
        else:
            passed = bool(mean_regret >= float(eligibility["minimum_nonlinear_linear_probe_top1_regret_r2"]))
        passed &= clean_probability >= float(eligibility["minimum_clean_latent_top_probability"])
        family_passes[family] = passed
        family_rows.append({
            "family": family,
            "panel_count": len(rows),
            "clean_latent_top_probability": clean_probability,
            "mean_linear_top1_regret_r2": mean_regret,
            "mean_linear_to_envelope_spearman": float(np.mean([
                row["linear_to_envelope_spearman"] for row in rows
                if row["linear_to_envelope_spearman"] is not None
            ])),
            "passed": passed,
        })
    synthetic_pass = bool(synthetic_all_panel_gates and all(family_passes.values()))
    by_target = {row["target_key"]: row for row in real_rows}
    cub_pass = all(
        by_target[key]["linear_top1_regret_r2"] <= 0.01
        for key in (
            "cub_200_2011_part_geometry::leg_span_over_bbox_diagonal",
            "cub_200_2011_part_geometry::wing_axis_cosine",
        )
    )
    lsp = by_target["lsp_2000_pose_geometry::wrist_span_over_torso_length"]
    lsp_reversal = bool(
        set(lsp["envelope_best_candidate_ids"]).isdisjoint(lsp["linear_best_candidate_ids"])
        and lsp["linear_top1_regret_r2"] > 0.01
    )
    return {
        "schema_version": 1,
        "status": "completed_continuous_ranking_benchmark_v2_evaluator_pilot",
        "real_rows": real_rows,
        "synthetic_rows": synthetic_rows,
        "synthetic_family_rows": family_rows,
        "decision": {
            "synthetic_controls_pass": synthetic_pass,
            "cub_linear_controls_preserved": cub_pass,
            "lsp_wrist_nonlinear_reversal_replicated": lsp_reversal,
            "pilot_pass": bool(synthetic_pass and cub_pass and lsp_reversal),
            "method_promotion": "none_evaluator_pilot_only",
        },
        "overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous ranking benchmark V2 evaluator pilot", "", "## Corrected real references", "",
        "| Target | Draws | Unique labels | Envelope best | Linear best | Linear rho | Linear regret | Stable |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in summary["real_rows"]:
        rho = "undefined" if row["linear_to_envelope_spearman"] is None else f"{row['linear_to_envelope_spearman']:.4f}"
        lines.append(
            f"| {row['target_key']} | {row['draw_count']} | {row['minimum_unique_training_rows']} | "
            f"{','.join(row['envelope_best_candidate_ids'])} | {','.join(row['linear_best_candidate_ids'])} | "
            f"{rho} | {row['linear_top1_regret_r2']:.4f} | {str(row['stable']).lower()} |"
        )
    lines.extend([
        "", "## Synthetic controls", "",
        "| Family | Panels | Clean-top probability | Mean linear regret | Mean linear rho | Pass |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary["synthetic_family_rows"]:
        lines.append(
            f"| {row['family']} | {row['panel_count']} | {row['clean_latent_top_probability']:.4f} | "
            f"{row['mean_linear_top1_regret_r2']:.4f} | {row['mean_linear_to_envelope_spearman']:.4f} | "
            f"{str(row['passed']).lower()} |"
        )
    decision = summary["decision"]
    lines.extend([
        "", "## Frozen pilot decision", "",
        f"- Synthetic controls pass: **{str(decision['synthetic_controls_pass']).lower()}**.",
        f"- CUB linear controls preserved: **{str(decision['cub_linear_controls_preserved']).lower()}**.",
        f"- LSP wrist nonlinear reversal replicated: **{str(decision['lsp_wrist_nonlinear_reversal_replicated']).lower()}**.",
        f"- Evaluator pilot pass: **{str(decision['pilot_pass']).lower()}**.",
        "- This pilot promotes no continuous OverlapIndex method.", "",
    ])
    return "\n".join(lines)


def _checkpoint_identity(protocol: Mapping[str, Any], source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "source_hashes": dict(source_hashes),
        "backbones": list(BACKBONES),
        "target_keys": [row["target_key"] for row in protocol["targets"]],
        "synthetic_families": list(protocol["synthetic_controls"]["families"]),
        "synthetic_seeds": list(protocol["synthetic_controls"]["confirmation_seeds"]),
    }


def _load_checkpoint(path: Path, identity: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError("evaluator-pilot checkpoint exists; use --resume")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("identity") != identity:
            raise ValueError("evaluator-pilot checkpoint identity changed")
        return value
    return {
        "schema_version": 1,
        "status": "running_continuous_ranking_benchmark_v2_evaluator_pilot",
        "identity": dict(identity),
        "real_panels": [],
        "synthetic_panels": [],
    }


def validate_grid(protocol: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    expected = protocol["expected_grid"]
    real_panels = checkpoint["real_panels"]
    synthetic_panels = checkpoint["synthetic_panels"]
    if len(real_panels) != int(expected["real_target_count"]):
        raise ValueError("evaluator-pilot real target grid changed")
    if [row["target_key"] for row in real_panels] != [row["target_key"] for row in protocol["targets"]]:
        raise ValueError("evaluator-pilot real target order changed")
    real_draws = sum(len(candidate["draws"]) for panel in real_panels for candidate in panel["candidates"])
    real_heads = sum(
        len(draw["heads"]) for panel in real_panels for candidate in panel["candidates"] for draw in candidate["draws"]
    )
    if real_draws != int(expected["real_candidate_draw_count"]) or real_heads != int(expected["real_final_head_fit_count"]):
        raise ValueError("evaluator-pilot real fit grid changed")
    expected_synthetic = [
        (family, int(seed))
        for family in protocol["synthetic_controls"]["families"]
        for seed in protocol["synthetic_controls"]["confirmation_seeds"]
    ]
    if [(row["family"], int(row["seed"])) for row in synthetic_panels] != expected_synthetic:
        raise ValueError("evaluator-pilot synthetic panel grid changed")
    if len(synthetic_panels) != int(expected["synthetic_panel_count"]):
        raise ValueError("evaluator-pilot synthetic panel count changed")
    synthetic_draws = sum(len(row["candidates"]) for row in synthetic_panels)
    synthetic_heads = sum(len(candidate["heads"]) for row in synthetic_panels for candidate in row["candidates"])
    if synthetic_draws != int(expected["synthetic_candidate_draw_count"]) or synthetic_heads != int(expected["synthetic_final_head_fit_count"]):
        raise ValueError("evaluator-pilot synthetic fit grid changed")
    for panel in real_panels:
        for candidate in panel["candidates"]:
            for draw in candidate["draws"]:
                if draw["unique_training_rows"] != panel["training_rows_per_draw"]:
                    raise ValueError("evaluator-pilot real draw is not unique")
                if any(fold["identity_overlap_count"] != 0 for fold in draw["fold_audit"]):
                    raise ValueError("evaluator-pilot real inner fold leaks identities")
    for panel in synthetic_panels:
        if panel["unique_training_rows"] != panel["training_rows"]:
            raise ValueError("evaluator-pilot synthetic draw is not unique")
        for candidate in panel["candidates"]:
            if any(fold["identity_overlap_count"] != 0 for fold in candidate["fold_audit"]):
                raise ValueError("evaluator-pilot synthetic inner fold leaks identities")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    protocol = load_protocol()
    source_hashes = _source_hashes(protocol)
    cohort, matrices = v1_extract.load_bank()
    v1_raw = json.loads((V1_RESULT_ROOT / "raw_results.json").read_text(encoding="utf-8"))
    output_root = output_root.resolve()
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("evaluator-pilot terminal exists")
        return validate_artifacts(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "checkpoint.json"
    identity = _checkpoint_identity(protocol, source_hashes)
    checkpoint = _load_checkpoint(checkpoint_path, identity, resume=resume)

    completed_real = {row["target_key"] for row in checkpoint["real_panels"]}
    for target_config in protocol["targets"]:
        target_key = target_config["target_key"]
        if target_key in completed_real:
            continue
        checkpoint["real_panels"].append(
            build_real_panel(protocol, target_config, cohort, matrices, v1_raw)
        )
        _finite_json(checkpoint)
        _atomic_json(checkpoint_path, checkpoint)
        print(f"real {len(checkpoint['real_panels'])}/{len(protocol['targets'])}: {target_key}", flush=True)

    completed_synthetic = {(row["family"], int(row["seed"])) for row in checkpoint["synthetic_panels"]}
    total_synthetic = int(protocol["expected_grid"]["synthetic_panel_count"])
    for family in protocol["synthetic_controls"]["families"]:
        for seed in protocol["synthetic_controls"]["confirmation_seeds"]:
            key = (str(family), int(seed))
            if key in completed_synthetic:
                continue
            checkpoint["synthetic_panels"].append(build_synthetic_panel(*key, protocol))
            _finite_json(checkpoint)
            _atomic_json(checkpoint_path, checkpoint)
            print(f"synthetic {len(checkpoint['synthetic_panels'])}/{total_synthetic}: {key[0]}/{key[1]}", flush=True)

    validate_grid(protocol, checkpoint)
    summary = summarize(protocol, checkpoint["real_panels"], checkpoint["synthetic_panels"])
    raw = {
        "schema_version": 1,
        "status": "completed_continuous_ranking_benchmark_v2_evaluator_pilot",
        "protocol": protocol,
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
        },
        "real_panels": checkpoint["real_panels"],
        "synthetic_panels": checkpoint["synthetic_panels"],
        "overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }
    _finite_json(raw)
    _finite_json(summary)
    raw_path = output_root / "raw_results.json"
    summary_path = output_root / "summary.json"
    report_path = output_root / "report.md"
    _atomic_json(raw_path, raw)
    _atomic_json(summary_path, summary)
    _atomic_bytes(report_path, render_report(summary).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_ranking_benchmark_v2_evaluator_pilot",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "checkpoint_identity": identity,
        "artifacts": {
            "checkpoint.json": sha256_path(checkpoint_path),
            "raw_results.json": sha256_path(raw_path),
            "summary.json": sha256_path(summary_path),
            "report.md": sha256_path(report_path),
        },
        "counts": {
            "real_panels": len(checkpoint["real_panels"]),
            "real_candidate_draws": sum(len(candidate["draws"]) for panel in checkpoint["real_panels"] for candidate in panel["candidates"]),
            "synthetic_panels": len(checkpoint["synthetic_panels"]),
            "synthetic_candidate_draws": sum(len(row["candidates"]) for row in checkpoint["synthetic_panels"]),
            "overlapindex_fits": 0,
            "reserved_backbones": 0,
        },
        "git": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        },
        "decision": summary["decision"],
    }
    _atomic_json(output_root / "manifest.json", manifest)
    terminal = {
        "schema_version": 1,
        "status": "completed_continuous_ranking_benchmark_v2_evaluator_pilot",
        "manifest_sha256": sha256_path(output_root / "manifest.json"),
        "summary_sha256": sha256_path(summary_path),
        "decision": summary["decision"],
    }
    _atomic_json(terminal_path, terminal)
    return validate_artifacts(output_root)


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    protocol = load_protocol()
    terminal = json.loads((output_root / "terminal.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output_root / "checkpoint.json").read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "completed_continuous_ranking_benchmark_v2_evaluator_pilot"
        or terminal.get("manifest_sha256") != sha256_path(output_root / "manifest.json")
        or terminal.get("summary_sha256") != sha256_path(output_root / "summary.json")
        or manifest.get("protocol_sha256") != sha256_path(PROTOCOL_PATH)
        or manifest.get("source_hashes") != _source_hashes(protocol)
        or terminal.get("decision") != summary.get("decision")
    ):
        raise ValueError("evaluator-pilot terminal identity changed")
    for name, expected in manifest["artifacts"].items():
        if sha256_path(output_root / name) != expected:
            raise ValueError(f"evaluator-pilot artifact changed: {name}")
    validate_grid(protocol, checkpoint)
    counts = manifest["counts"]
    expected = protocol["expected_grid"]
    if (
        int(counts["real_panels"]) != int(expected["real_target_count"])
        or int(counts["real_candidate_draws"]) != int(expected["real_candidate_draw_count"])
        or int(counts["synthetic_panels"]) != int(expected["synthetic_panel_count"])
        or int(counts["synthetic_candidate_draws"]) != int(expected["synthetic_candidate_draw_count"])
        or int(counts["overlapindex_fits"]) != 0
        or int(counts["reserved_backbones"]) != 0
    ):
        raise ValueError("evaluator-pilot terminal counts changed")
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
