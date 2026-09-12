"""Run the frozen target-reliability and recoverability diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold

from experiments.continuous_residual_ranking import run as head_source
from experiments.continuous_ranking_reliability_recoverability import extract, prepare
from experiments.continuous_structured_head_discovery import run as geometry


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_ranking_reliability_recoverability" / "results"

HEADS = ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
METHODS = ("linear_nested_cv", "four_head_nested_cv_envelope")
BACKBONES = tuple(extract.BACKBONES)
BUDGETS = (300, 600, 1200)
REFERENCE_BUDGET = 600
REFERENCE_BUDGET_OVERRIDES = {
    "lsp_2000_pose_geometry::wrist_span_over_torso_length": 590,
}
REFERENCE_REPEATS = 3
OUTER_FOLDS = 3
EVALUATION_ROWS = 200
UNCERTAINTY_RESAMPLES = 1000
TIE_ATOL = 1.0e-12
MATERIAL_ATOL = 0.01
THREAD_KEYS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: Path) -> str:
    return prepare.sha256_path(path)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _target_key(dataset_id: str, target_id: str) -> str:
    return f"{dataset_id}::{target_id}"


def _target_order(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        target
        for group in ("recoverable_lp_dominated_control", "rank_changing_nonlinear_opportunity")
        for target in protocol["targets"][group]
    )


def _strata(protocol: Mapping[str, Any]) -> dict[str, str]:
    return {
        target: group
        for group, targets in protocol["targets"].items()
        for target in targets
    }


def load_protocol() -> dict[str, Any]:
    protocol = prepare.load_protocol()
    if tuple(protocol["backbones"]) != BACKBONES:
        raise ValueError("diagnostic backbone roster changed")
    if tuple(protocol["heads"]["families"]) != HEADS:
        raise ValueError("diagnostic head roster changed")
    if tuple(protocol["recoverability"]["methods"]) != METHODS:
        raise ValueError("diagnostic method roster changed")
    if tuple(int(value) for value in protocol["recoverability"]["budgets"]) != BUDGETS:
        raise ValueError("diagnostic budget roster changed")
    if (
        int(protocol["ranking_target"]["default_downstream_training_rows"]) != REFERENCE_BUDGET
        or {key: int(value) for key, value in protocol["ranking_target"]["training_row_overrides"].items()}
        != REFERENCE_BUDGET_OVERRIDES
        or int(protocol["ranking_target"]["training_bootstrap_repeats"]) != REFERENCE_REPEATS
        or int(protocol["ranking_target"]["evaluation_rows"]) != EVALUATION_ROWS
        or int(protocol["ranking_target"]["uncertainty_resamples"]) != UNCERTAINTY_RESAMPLES
        or int(protocol["recoverability"]["outer_folds"]) != OUTER_FOLDS
    ):
        raise ValueError("diagnostic evaluation contract changed")
    return protocol


def _source_hashes() -> dict[str, str]:
    paths = {
        "protocol": PROTOCOL_PATH,
        "protocol_sidecar": PROTOCOL_SHA256_PATH,
        "prepare": Path(prepare.__file__).resolve(),
        "extract": Path(extract.__file__).resolve(),
        "runner": Path(__file__).resolve(),
        "input_cohort": prepare.DEFAULT_OUTPUT_ROOT / "cohort.json",
        "input_manifest": prepare.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "embedding_manifest": extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "head_source": Path(head_source.__file__).resolve(),
        "pca_source": Path(geometry.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"diagnostic provenance input is missing: {missing}")
    return {name: sha256_path(path) for name, path in paths.items()}


def _head_estimators() -> dict[str, tuple[Any, Mapping[str, Sequence[Any]]]]:
    estimators = head_source._head_estimators()
    if tuple(estimators) != HEADS:
        raise ValueError("frozen head evaluator order changed")
    return estimators


def evaluate_heads(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    *,
    retain_predictions: bool,
) -> dict[str, dict[str, Any]]:
    results = {}
    for head_id, (estimator, grid) in _head_estimators().items():
        # All families see identical inner folds so cross-family comparisons
        # are not contaminated by a different validation partition.
        inner = KFold(n_splits=3, shuffle=True, random_state=_seed(seed, "inner"))
        search = GridSearchCV(
            clone(estimator),
            param_grid=dict(grid),
            cv=inner,
            scoring="r2",
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        started = time.perf_counter()
        search.fit(X_train, y_train)
        prediction = np.asarray(search.predict(X_test), dtype=np.float64)
        row = {
            "test_r2": float(r2_score(y_test, prediction)),
            "inner_best_r2": float(search.best_score_),
            "best_params": dict(search.best_params_),
            "wall_seconds": float(time.perf_counter() - started),
            "prediction_sha256": hashlib.sha256(prediction.tobytes(order="C")).hexdigest(),
        }
        if retain_predictions:
            row["prediction"] = prediction.tolist()
        results[head_id] = row
    return results


def ranking_metrics(truth: Sequence[float], scores: Sequence[float]) -> dict[str, Any]:
    truth_values = np.asarray(truth, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    if truth_values.shape != (len(BACKBONES),) or score_values.shape != truth_values.shape:
        raise ValueError("diagnostic ranking vectors have the wrong shape")
    if not np.isfinite(truth_values).all() or not np.isfinite(score_values).all():
        raise ValueError("diagnostic ranking vectors are nonfinite")
    correlation = None
    if np.ptp(truth_values) > TIE_ATOL and np.ptp(score_values) > TIE_ATOL:
        observed = float(spearmanr(truth_values, score_values).correlation)
        correlation = observed if np.isfinite(observed) else None
    selected = np.flatnonzero(score_values >= np.max(score_values) - TIE_ATOL)
    truth_best = np.flatnonzero(truth_values >= np.max(truth_values) - TIE_ATOL)
    pair_credit = []
    for left in range(len(truth_values)):
        for right in range(left + 1, len(truth_values)):
            truth_delta = float(truth_values[left] - truth_values[right])
            if abs(truth_delta) < MATERIAL_ATOL:
                continue
            score_delta = float(score_values[left] - score_values[right])
            pair_credit.append(0.5 if abs(score_delta) <= TIE_ATOL else float(np.sign(score_delta) == np.sign(truth_delta)))
    return {
        "spearman": correlation,
        "top1_regret_r2": float(np.max(truth_values) - np.max(truth_values[selected])),
        "exact_best": bool(set(selected).intersection(truth_best)),
        "material_pair_accuracy": float(np.mean(pair_credit)) if pair_credit else None,
        "material_pair_count": len(pair_credit),
        "selected_candidate_ids": [BACKBONES[index] for index in selected],
    }


def _panel_rows(dataset: Mapping[str, Any], target_id: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    cohort = dataset["target_cohorts"][target_id]
    training = sorted(
        (row for row in cohort if row["split"] == "training"),
        key=lambda row: int(row["budget_rank"]),
    )
    evaluation = [row for row in cohort if row["split"] == "evaluation"]
    if not training or len(training) % prepare.TARGET_BINS or len(evaluation) != EVALUATION_ROWS:
        raise ValueError("diagnostic panel cohort is incomplete")
    if [int(row["budget_rank"]) for row in training] != list(range(len(training))):
        raise ValueError("diagnostic nested training order changed")
    return training, evaluation


def _stratified_bootstrap(rows: Sequence[Mapping[str, Any]], seed: int) -> np.ndarray:
    reference_budget = len(rows)
    if not reference_budget or reference_budget % prepare.TARGET_BINS:
        raise ValueError("reference bootstrap requires a nonempty bin-balanced cohort")
    rng = np.random.default_rng(seed)
    selected = []
    for target_bin in range(prepare.TARGET_BINS):
        positions = np.asarray(
            [position for position, row in enumerate(rows) if int(row["target_bin"]) == target_bin],
            dtype=np.int64,
        )
        if len(positions) != reference_budget // prepare.TARGET_BINS:
            raise ValueError("reference target-bin size changed")
        selected.extend(rng.choice(positions, size=len(positions), replace=True).tolist())
    return np.asarray(selected, dtype=np.int64)


def build_reference_panel(
    protocol: Mapping[str, Any],
    dataset_id: str,
    target_id: str,
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    target_key = _target_key(dataset_id, target_id)
    training, evaluation = _panel_rows(dataset, target_id)
    reference_budget = REFERENCE_BUDGET_OVERRIDES.get(target_key, REFERENCE_BUDGET)
    base_training = training[:reference_budget]
    if len(base_training) != reference_budget:
        raise ValueError("reference budget exceeds the target cohort")
    eval_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    y_eval = np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64)
    candidates = []
    frozen_seeds = tuple(int(value) for value in protocol["ranking_target"]["bootstrap_seeds"])
    for backbone_position, backbone in enumerate(BACKBONES):
        repeats = []
        matrix = dataset["matrices"][backbone]
        for repeat, base_seed in enumerate(frozen_seeds):
            positions = _stratified_bootstrap(base_training, _seed(base_seed, target_key))
            sampled = [base_training[int(position)] for position in positions]
            train_indices = np.asarray([int(row["matrix_index"]) for row in sampled], dtype=np.int64)
            y_train = np.asarray([float(row["target"]) for row in sampled], dtype=np.float64)
            pca_seed = _seed("reference-pca", target_key, backbone, repeat)
            X_train, X_eval, pca = geometry.pca_geometry(matrix, train_indices, eval_indices, pca_seed)
            heads = evaluate_heads(
                X_train,
                y_train,
                X_eval,
                y_eval,
                _seed("reference-head", target_key, backbone, repeat),
                retain_predictions=True,
            )
            repeats.append(
                {
                    "repeat": repeat,
                    "bootstrap_seed": base_seed,
                    "training_sample_ids_sha256": sha256_payload([row["sample_id"] for row in sampled]),
                    "unique_training_rows": len({row["sample_id"] for row in sampled}),
                    "pca_seed": pca_seed,
                    "pca": pca,
                    "heads": heads,
                }
            )
        means = {
            head: float(np.mean([row["heads"][head]["test_r2"] for row in repeats]))
            for head in HEADS
        }
        envelope = max(means.values())
        candidates.append(
            {
                "candidate_id": backbone,
                "repeats": repeats,
                "head_mean_heldout_r2": means,
                "envelope_r2": float(envelope),
                "best_head_ids": [head for head in HEADS if means[head] >= envelope - TIE_ATOL],
            }
        )
    panel = {
        "target_key": target_key,
        "dataset_id": dataset_id,
        "target_id": target_id,
        "stratum": _strata(protocol)[target_key],
        "training_rows": reference_budget,
        "evaluation_rows": EVALUATION_ROWS,
        "evaluation_sample_ids": [row["sample_id"] for row in evaluation],
        "evaluation_target": y_eval.tolist(),
        "candidates": candidates,
    }
    panel["uncertainty"] = reference_uncertainty(panel, protocol)
    return panel


def reference_uncertainty(panel: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    y = np.asarray(panel["evaluation_target"], dtype=np.float64)
    predictions = np.asarray(
        [
            [
                [candidate["repeats"][repeat]["heads"][head]["prediction"] for head in HEADS]
                for repeat in range(REFERENCE_REPEATS)
            ]
            for candidate in panel["candidates"]
        ],
        dtype=np.float64,
    )
    point = np.asarray([candidate["envelope_r2"] for candidate in panel["candidates"]], dtype=np.float64)
    rng = np.random.default_rng(_seed(protocol["ranking_target"]["uncertainty_seed"], panel["target_key"]))
    boot = np.empty((UNCERTAINTY_RESAMPLES, len(BACKBONES)), dtype=np.float64)
    for sample in range(UNCERTAINTY_RESAMPLES):
        repeat_ids = rng.integers(0, REFERENCE_REPEATS, size=REFERENCE_REPEATS)
        row_ids = rng.integers(0, len(y), size=len(y))
        for candidate in range(len(BACKBONES)):
            head_values = []
            for head in range(len(HEADS)):
                head_values.append(
                    np.mean(
                        [r2_score(y[row_ids], predictions[candidate, repeat, head, row_ids]) for repeat in repeat_ids]
                    )
                )
            boot[sample, candidate] = max(head_values)
    correlations = [
        float(spearmanr(point, row).correlation)
        for row in boot
        if np.ptp(row) > TIE_ATOL and np.isfinite(spearmanr(point, row).correlation)
    ]
    point_best = np.flatnonzero(point >= np.max(point) - TIE_ATOL)
    top1_probability = float(
        np.mean([
            bool(set(np.flatnonzero(row >= np.max(row) - TIE_ATOL)).intersection(point_best))
            for row in boot
        ])
    )
    pair_rows = []
    for left in range(len(BACKBONES)):
        for right in range(left + 1, len(BACKBONES)):
            delta = float(point[left] - point[right])
            if abs(delta) < MATERIAL_ATOL:
                continue
            probability = float(np.mean(np.sign(boot[:, left] - boot[:, right]) == np.sign(delta)))
            pair_rows.append(
                {"left": BACKBONES[left], "right": BACKBONES[right], "point_delta_r2": delta, "sign_probability": probability}
            )
    median_pair = float(np.median([row["sign_probability"] for row in pair_rows])) if pair_rows else None
    median_rho = float(np.median(correlations)) if correlations else None
    thresholds = protocol["ranking_target"]
    stable = bool(
        median_rho is not None
        and median_rho >= float(thresholds["stable_target_minimum_bootstrap_rank_spearman"])
        and top1_probability >= float(thresholds["stable_target_minimum_top1_probability"])
        and median_pair is not None
        and median_pair >= float(thresholds["stable_target_minimum_material_pair_sign_probability"])
    )
    return {
        "resamples": UNCERTAINTY_RESAMPLES,
        "median_bootstrap_rank_spearman": median_rho,
        "rank_spearman_interval_95": [float(np.quantile(correlations, 0.025)), float(np.quantile(correlations, 0.975))] if correlations else None,
        "point_top1_probability": top1_probability,
        "material_pairs": pair_rows,
        "median_material_pair_sign_probability": median_pair,
        "candidate_envelope_intervals_95": {
            backbone: [float(np.quantile(boot[:, position], 0.025)), float(np.quantile(boot[:, position], 0.975))]
            for position, backbone in enumerate(BACKBONES)
        },
        "stable": stable,
    }


def nested_candidate_scores(
    matrix: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    target_key: str,
    backbone: str,
    budget: int,
) -> dict[str, Any]:
    selected = list(rows[:budget])
    if len(selected) != budget:
        raise ValueError("nested diagnostic budget exceeds selected rows")
    indices = np.asarray([int(row["matrix_index"]) for row in selected], dtype=np.int64)
    y = np.asarray([float(row["target"]) for row in selected], dtype=np.float64)
    outer_seed = _seed("outer", target_key, budget)
    splitter = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=outer_seed)
    predictions = {head: np.empty(budget, dtype=np.float64) for head in HEADS}
    folds = []
    started = time.perf_counter()
    for fold, (train_position, holdout_position) in enumerate(splitter.split(indices)):
        train_indices, holdout_indices = indices[train_position], indices[holdout_position]
        y_train, y_holdout = y[train_position], y[holdout_position]
        pca_seed = _seed("nested-pca", target_key, backbone, budget, fold)
        X_train, X_holdout, pca = geometry.pca_geometry(matrix, train_indices, holdout_indices, pca_seed)
        heads = evaluate_heads(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            _seed("nested-head", target_key, backbone, budget, fold),
            retain_predictions=True,
        )
        for head in HEADS:
            predictions[head][holdout_position] = np.asarray(heads[head].pop("prediction"), dtype=np.float64)
        folds.append(
            {
                "fold": fold,
                "pca_seed": pca_seed,
                "train_sample_ids_sha256": sha256_payload([selected[int(position)]["sample_id"] for position in train_position]),
                "holdout_sample_ids_sha256": sha256_payload([selected[int(position)]["sample_id"] for position in holdout_position]),
                "pca": pca,
                "heads": heads,
            }
        )
    head_scores = {head: float(r2_score(y, prediction)) for head, prediction in predictions.items()}
    return {
        "candidate_id": backbone,
        "budget": budget,
        "head_nested_cv_r2": head_scores,
        "scores": {
            "linear_nested_cv": head_scores["linear_ridge"],
            "four_head_nested_cv_envelope": max(head_scores.values()),
        },
        "head_prediction_sha256": {head: hashlib.sha256(value.tobytes(order="C")).hexdigest() for head, value in predictions.items()},
        "folds": folds,
        "wall_seconds": float(time.perf_counter() - started),
    }


def build_recoverability_panel(
    protocol: Mapping[str, Any],
    dataset_id: str,
    target_id: str,
    dataset: Mapping[str, Any],
    budget: int,
    reference_panel: Mapping[str, Any],
) -> dict[str, Any]:
    target_key = _target_key(dataset_id, target_id)
    training, _evaluation = _panel_rows(dataset, target_id)
    if budget > len(training):
        raise ValueError("recoverability budget is not supported by this target")
    candidates = [
        nested_candidate_scores(dataset["matrices"][backbone], training, target_key, backbone, budget)
        for backbone in BACKBONES
    ]
    truth = [float(row["envelope_r2"]) for row in reference_panel["candidates"]]
    metrics = {
        method: ranking_metrics(truth, [float(row["scores"][method]) for row in candidates])
        for method in METHODS
    }
    return {
        "target_key": target_key,
        "dataset_id": dataset_id,
        "target_id": target_id,
        "stratum": _strata(protocol)[target_key],
        "budget": budget,
        "truth_reference_budget": int(reference_panel["training_rows"]),
        "candidates": candidates,
        "ranking_metrics": metrics,
    }


def synthetic_panel(family: str, seed: int) -> dict[str, Any]:
    if family not in ("linear", "quadratic", "smooth_local"):
        raise ValueError(f"unknown synthetic family: {family!r}")
    rng = np.random.default_rng(_seed(seed, family))
    total = 1200 + 2000
    latent = rng.normal(size=(total, 4))
    target_noise = 0.10 * rng.normal(size=total)
    if family == "linear":
        y = latent[:, 0] + 0.5 * latent[:, 1] + target_noise
    elif family == "quadratic":
        y = latent[:, 0] * latent[:, 1] + 0.35 * latent[:, 2] ** 2 + target_noise
    else:
        y = np.sin(2.5 * latent[:, 0]) + 0.6 * np.cos(2.0 * latent[:, 1]) + target_noise
    scales = (1.20, 0.85, 0.60, 0.38, 0.18)
    candidates = []
    for position, scale in enumerate(scales):
        candidate_rng = np.random.default_rng(_seed(seed, family, position))
        signal = latent + scale * candidate_rng.normal(size=latent.shape)
        nuisance = 0.25 * candidate_rng.normal(size=(total, 28))
        matrix = np.asarray(np.column_stack((signal, nuisance)), dtype=np.float32)
        candidates.append({"candidate_id": BACKBONES[position], "matrix": matrix})
    return {"family": family, "y": y, "candidates": candidates, "known_quality_order": list(reversed(BACKBONES))}


def run_synthetic_controls(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    base_seed = int(protocol["synthetic_controls"]["seed"])
    for family in protocol["synthetic_controls"]["families"]:
        panel = synthetic_panel(str(family), base_seed)
        y = np.asarray(panel["y"], dtype=np.float64)
        reference_values = []
        for position, candidate in enumerate(panel["candidates"]):
            train = np.arange(600, dtype=np.int64)
            evaluation = np.arange(1200, 3200, dtype=np.int64)
            X_train, X_eval, _pca = geometry.pca_geometry(candidate["matrix"], train, evaluation, _seed("synthetic-reference-pca", family, position))
            heads = evaluate_heads(X_train, y[train], X_eval, y[evaluation], _seed("synthetic-reference-head", family, position), retain_predictions=False)
            reference_values.append(max(float(row["test_r2"]) for row in heads.values()))
        for budget in BUDGETS:
            rows = [
                {"sample_id": f"synthetic/{family}/{position}", "matrix_index": position, "target": float(y[position])}
                for position in range(1200)
            ]
            candidates = [
                nested_candidate_scores(candidate["matrix"], rows, f"synthetic::{family}", candidate["candidate_id"], budget)
                for candidate in panel["candidates"]
            ]
            output.append(
                {
                    "family": family,
                    "budget": budget,
                    "reference_envelope_r2": reference_values,
                    "known_quality_order": panel["known_quality_order"],
                    "candidates": candidates,
                    "ranking_metrics": {
                        method: ranking_metrics(reference_values, [row["scores"][method] for row in candidates])
                        for method in METHODS
                    },
                }
            )
    return output


def _aggregate_metric(rows: Sequence[Mapping[str, Any]], method: str, key: str) -> float | None:
    values = [row["ranking_metrics"][method][key] for row in rows]
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def summarize(
    protocol: Mapping[str, Any],
    reference_panels: Sequence[Mapping[str, Any]],
    recoverability_panels: Sequence[Mapping[str, Any]],
    synthetic_panels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_rows = []
    for panel in reference_panels:
        linear = np.asarray(
            [candidate["head_mean_heldout_r2"]["linear_ridge"] for candidate in panel["candidates"]],
            dtype=np.float64,
        )
        envelope = np.asarray([candidate["envelope_r2"] for candidate in panel["candidates"]], dtype=np.float64)
        reference_rows.append(
            {
                "target_key": panel["target_key"],
                "stratum": panel["stratum"],
                "stable": bool(panel["uncertainty"]["stable"]),
                "median_bootstrap_rank_spearman": panel["uncertainty"]["median_bootstrap_rank_spearman"],
                "point_top1_probability": panel["uncertainty"]["point_top1_probability"],
                "median_material_pair_sign_probability": panel["uncertainty"]["median_material_pair_sign_probability"],
                "maximum_nonlinear_gain_r2": float(np.max(envelope - linear)),
                "linear_to_envelope_rank_spearman": ranking_metrics(envelope, linear)["spearman"],
                "linear_top1_regret_r2": ranking_metrics(envelope, linear)["top1_regret_r2"],
            }
        )
    recoverability_rows = []
    for budget in BUDGETS:
        budget_rows = [row for row in recoverability_panels if int(row["budget"]) == budget]
        for method in METHODS:
            recoverability_rows.append(
                {
                    "budget": budget,
                    "method": method,
                    "target_count": len(budget_rows),
                    "mean_spearman": _aggregate_metric(budget_rows, method, "spearman"),
                    "mean_top1_regret_r2": _aggregate_metric(budget_rows, method, "top1_regret_r2"),
                    "mean_material_pair_accuracy": _aggregate_metric(budget_rows, method, "material_pair_accuracy"),
                    "exact_best_rate": _aggregate_metric(budget_rows, method, "exact_best"),
                    "aggregate_runtime_seconds": float(sum(sum(candidate["wall_seconds"] for candidate in row["candidates"]) for row in budget_rows)),
                }
            )
    synthetic_rows = []
    for family in protocol["synthetic_controls"]["families"]:
        for budget in BUDGETS:
            row = next(item for item in synthetic_panels if item["family"] == family and int(item["budget"]) == budget)
            synthetic_rows.append(
                {
                    "family": family,
                    "budget": budget,
                    "linear_spearman": row["ranking_metrics"]["linear_nested_cv"]["spearman"],
                    "four_head_spearman": row["ranking_metrics"]["four_head_nested_cv_envelope"]["spearman"],
                    "linear_top1_regret_r2": row["ranking_metrics"]["linear_nested_cv"]["top1_regret_r2"],
                    "four_head_top1_regret_r2": row["ranking_metrics"]["four_head_nested_cv_envelope"]["top1_regret_r2"],
                }
            )
    stable_count = sum(row["stable"] for row in reference_rows)
    n300 = {row["method"]: row for row in recoverability_rows if row["budget"] == 300}
    real_gain = None
    if n300[METHODS[0]]["mean_spearman"] is not None and n300[METHODS[1]]["mean_spearman"] is not None:
        real_gain = float(n300[METHODS[1]]["mean_spearman"] - n300[METHODS[0]]["mean_spearman"])
    synthetic_n300 = {row["family"]: row for row in synthetic_rows if row["budget"] == 300}
    nonlinear_gains = [
        float(synthetic_n300[family]["four_head_spearman"] - synthetic_n300[family]["linear_spearman"])
        for family in ("quadratic", "smooth_local")
        if synthetic_n300[family]["four_head_spearman"] is not None and synthetic_n300[family]["linear_spearman"] is not None
    ]
    linear_drop = None
    linear_control = synthetic_n300["linear"]
    if linear_control["four_head_spearman"] is not None and linear_control["linear_spearman"] is not None:
        linear_drop = float(linear_control["linear_spearman"] - linear_control["four_head_spearman"])
    synthetic_pass = bool(
        nonlinear_gains
        and float(np.mean(nonlinear_gains)) >= float(protocol["synthetic_controls"]["minimum_nonlinear_family_spearman_gain_at_n300"])
        and linear_drop is not None
        and linear_drop <= float(protocol["synthetic_controls"]["maximum_linear_family_spearman_drop_at_n300"])
    )
    return {
        "schema_version": 1,
        "status": "completed_continuous_ranking_reliability_recoverability",
        "reference_target_rows": reference_rows,
        "reference_stable_target_count": stable_count,
        "reference_target_count": len(reference_rows),
        "recoverability_rows": recoverability_rows,
        "synthetic_rows": synthetic_rows,
        "diagnosis": {
            "synthetic_positive_controls_pass": synthetic_pass,
            "synthetic_mean_nonlinear_spearman_gain_n300": float(np.mean(nonlinear_gains)) if nonlinear_gains else None,
            "synthetic_linear_spearman_drop_n300": linear_drop,
            "real_four_head_minus_linear_mean_spearman_n300": real_gain,
            "target_reliability_sufficient_for_method_search": stable_count >= 6,
            "nonlinear_ranking_signal_recoverable_at_n300": bool(real_gain is not None and real_gain >= float(protocol["recoverability"]["minimum_material_spearman_gain_over_linear"])),
        },
        "reserved_backbones_evaluated": [],
        "overlapindex_fit_count": 0,
        "method_promotion": "none_diagnostic_only",
    }


def report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous ranking reliability and recoverability result",
        "",
        f"Stable reference targets: **{summary['reference_stable_target_count']}/{summary['reference_target_count']}**.",
        "",
        "## Reference reliability",
        "",
        "| Target | Stable | Bootstrap rho | Top-1 probability | Pair-sign probability | Max nonlinear gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["reference_target_rows"]:
        fmt = lambda value: "undefined" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {row['target_key']} | {str(row['stable']).lower()} | {fmt(row['median_bootstrap_rank_spearman'])} | "
            f"{fmt(row['point_top1_probability'])} | {fmt(row['median_material_pair_sign_probability'])} | "
            f"{fmt(row['maximum_nonlinear_gain_r2'])} |"
        )
    lines.extend([
        "", "## Recoverability", "",
        "| Labels | Method | Targets | Mean rho | Top-1 regret | Pair accuracy | Runtime (s) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary["recoverability_rows"]:
        fmt = lambda value: "undefined" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {row['budget']} | {row['method']} | {row['target_count']} | {fmt(row['mean_spearman'])} | "
            f"{fmt(row['mean_top1_regret_r2'])} | {fmt(row['mean_material_pair_accuracy'])} | "
            f"{row['aggregate_runtime_seconds']:.2f} |"
        )
    diagnosis = summary["diagnosis"]
    lines.extend([
        "", "## Frozen diagnostic conclusions", "",
        f"- Synthetic positive controls pass: **{str(diagnosis['synthetic_positive_controls_pass']).lower()}**.",
        f"- Target reliability is sufficient for method search: **{str(diagnosis['target_reliability_sufficient_for_method_search']).lower()}**.",
        f"- Nonlinear ranking signal is recoverable at 300 labels: **{str(diagnosis['nonlinear_ranking_signal_recoverable_at_n300']).lower()}**.",
        "- This run is diagnostic only; it promotes no method and evaluates no reserved backbone.",
        "",
    ])
    return "\n".join(lines)


def _checkpoint_identity(protocol: Mapping[str, Any], source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "source_hashes": dict(source_hashes),
        "target_order": list(_target_order(protocol)),
        "backbones": list(BACKBONES),
        "budgets": list(BUDGETS),
    }


def _load_or_create_checkpoint(
    path: Path,
    identity: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError("diagnostic checkpoint exists; use --resume")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("identity") != identity:
            raise ValueError("diagnostic checkpoint identity changed")
        return value
    return {"schema_version": 1, "status": "running", "identity": dict(identity), "reference_panels": [], "recoverability_panels": [], "synthetic_panels": []}


def validate_completed_grid(
    protocol: Mapping[str, Any],
    reference_panels: Sequence[Mapping[str, Any]],
    recoverability_panels: Sequence[Mapping[str, Any]],
    synthetic_panels: Sequence[Mapping[str, Any]],
) -> None:
    expected = protocol["expected_grid"]
    if len(reference_panels) != int(expected["real_target_count"]):
        raise ValueError("diagnostic reference panel count changed")
    target_order = _target_order(protocol)
    if tuple(row["target_key"] for row in reference_panels) != target_order:
        raise ValueError("diagnostic reference target order changed")
    reference_repeats = 0
    head_evaluations = 0
    for panel in reference_panels:
        if tuple(row["candidate_id"] for row in panel["candidates"]) != BACKBONES:
            raise ValueError("diagnostic reference candidate order changed")
        for candidate in panel["candidates"]:
            if len(candidate["repeats"]) != REFERENCE_REPEATS:
                raise ValueError("diagnostic reference repeat count changed")
            reference_repeats += len(candidate["repeats"])
            for repeat in candidate["repeats"]:
                if tuple(repeat["heads"]) != HEADS:
                    raise ValueError("diagnostic reference head order changed")
                head_evaluations += len(repeat["heads"])
    if reference_repeats != int(expected["ranking_target_candidate_repeat_count"]):
        raise ValueError("diagnostic reference candidate-repeat grid changed")
    if head_evaluations != int(expected["ranking_target_head_evaluation_count"]):
        raise ValueError("diagnostic reference head grid changed")
    expected_pairs = [
        (target, budget)
        for target in target_order
        for budget in BUDGETS
        if budget <= prepare.MAX_TRAIN_BY_TARGET[target]
    ]
    observed_pairs = [(row["target_key"], int(row["budget"])) for row in recoverability_panels]
    if observed_pairs != expected_pairs or len(observed_pairs) != int(expected["recoverability_target_budget_count"]):
        raise ValueError("diagnostic recoverability target-budget grid changed")
    candidate_count = 0
    outer_count = 0
    for panel in recoverability_panels:
        if tuple(row["candidate_id"] for row in panel["candidates"]) != BACKBONES:
            raise ValueError("diagnostic recoverability candidate order changed")
        candidate_count += len(panel["candidates"])
        outer_count += sum(len(row["folds"]) for row in panel["candidates"])
    if candidate_count != int(expected["recoverability_candidate_count"]):
        raise ValueError("diagnostic recoverability candidate grid changed")
    if outer_count != int(expected["recoverability_outer_candidate_fold_count"]):
        raise ValueError("diagnostic recoverability outer-fold grid changed")
    expected_synthetic = [
        (family, budget)
        for family in protocol["synthetic_controls"]["families"]
        for budget in BUDGETS
    ]
    observed_synthetic = [(row["family"], int(row["budget"])) for row in synthetic_panels]
    if observed_synthetic != expected_synthetic or len(observed_synthetic) != int(expected["synthetic_family_budget_count"]):
        raise ValueError("diagnostic synthetic grid changed")
    if sum(len(row["candidates"]) for row in synthetic_panels) != int(expected["synthetic_candidate_count"]):
        raise ValueError("diagnostic synthetic candidate grid changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    protocol = load_protocol()
    cohort, matrices = extract.load_bank()
    datasets = {
        dataset_id: {**cohort["datasets"][dataset_id], "matrices": matrices[dataset_id]}
        for dataset_id in cohort["datasets"]
    }
    source_hashes = _source_hashes()
    output_root = output_root.resolve()
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("diagnostic terminal artifact exists")
        return validate_artifacts(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "checkpoint.json"
    identity = _checkpoint_identity(protocol, source_hashes)
    checkpoint = _load_or_create_checkpoint(checkpoint_path, identity, resume=resume)
    reference_done = {row["target_key"] for row in checkpoint["reference_panels"]}
    for target_key in _target_order(protocol):
        if target_key in reference_done:
            continue
        dataset_id, target_id = target_key.split("::", 1)
        checkpoint["reference_panels"].append(build_reference_panel(protocol, dataset_id, target_id, datasets[dataset_id]))
        _finite_json(checkpoint)
        _atomic_json(checkpoint_path, checkpoint)
        print(f"reference {len(checkpoint['reference_panels'])}/8: {target_key}", flush=True)
    reference_by_target = {row["target_key"]: row for row in checkpoint["reference_panels"]}
    recoverability_done = {(row["target_key"], int(row["budget"])) for row in checkpoint["recoverability_panels"]}
    expected_recoverability = []
    for target_key in _target_order(protocol):
        dataset_id, target_id = target_key.split("::", 1)
        maximum = prepare.MAX_TRAIN_BY_TARGET[target_key]
        for budget in BUDGETS:
            if budget <= maximum:
                expected_recoverability.append((target_key, budget))
                if (target_key, budget) in recoverability_done:
                    continue
                checkpoint["recoverability_panels"].append(
                    build_recoverability_panel(protocol, dataset_id, target_id, datasets[dataset_id], budget, reference_by_target[target_key])
                )
                _finite_json(checkpoint)
                _atomic_json(checkpoint_path, checkpoint)
                total = int(protocol["expected_grid"]["recoverability_target_budget_count"])
                print(f"recoverability {len(checkpoint['recoverability_panels'])}/{total}: {target_key}/n{budget}", flush=True)
    if len(expected_recoverability) != int(protocol["expected_grid"]["recoverability_target_budget_count"]):
        raise RuntimeError("diagnostic recoverability grid changed")
    if not checkpoint["synthetic_panels"]:
        checkpoint["synthetic_panels"] = run_synthetic_controls(protocol)
        _finite_json(checkpoint)
        _atomic_json(checkpoint_path, checkpoint)
        print("synthetic controls 9/9", flush=True)
    validate_completed_grid(
        protocol,
        checkpoint["reference_panels"],
        checkpoint["recoverability_panels"],
        checkpoint["synthetic_panels"],
    )
    summary = summarize(protocol, checkpoint["reference_panels"], checkpoint["recoverability_panels"], checkpoint["synthetic_panels"])
    raw = {
        "schema_version": 1,
        "status": "completed_continuous_ranking_reliability_recoverability",
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
        "reference_panels": checkpoint["reference_panels"],
        "recoverability_panels": checkpoint["recoverability_panels"],
        "synthetic_panels": checkpoint["synthetic_panels"],
        "reserved_backbones_evaluated": [],
        "overlapindex_fit_count": 0,
    }
    _finite_json(raw)
    _finite_json(summary)
    raw_path, summary_path, report_path = output_root / "raw_results.json", output_root / "summary.json", output_root / "report.md"
    _atomic_json(raw_path, raw)
    _atomic_json(summary_path, summary)
    _atomic_bytes(report_path, report(summary).encode("utf-8"))
    counts = {
        "reference_panels": len(checkpoint["reference_panels"]),
        "reference_candidate_repeats": sum(len(candidate["repeats"]) for panel in checkpoint["reference_panels"] for candidate in panel["candidates"]),
        "recoverability_panels": len(checkpoint["recoverability_panels"]),
        "recoverability_candidates": sum(len(panel["candidates"]) for panel in checkpoint["recoverability_panels"]),
        "synthetic_panels": len(checkpoint["synthetic_panels"]),
        "reserved_backbones": 0,
        "overlapindex_fits": 0,
    }
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_ranking_reliability_recoverability",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "checkpoint_identity": identity,
        "artifacts": {
            "raw_results.json": sha256_path(raw_path),
            "summary.json": sha256_path(summary_path),
            "report.md": sha256_path(report_path),
            "checkpoint.json": sha256_path(checkpoint_path),
        },
        "counts": counts,
        "git": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        },
        "method_promotion": "none_diagnostic_only",
    }
    manifest_path = output_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    terminal = {
        "schema_version": 1,
        "status": manifest["status"],
        "manifest_sha256": sha256_path(manifest_path),
        "summary_sha256": sha256_path(summary_path),
        "decision": summary["diagnosis"],
    }
    _atomic_json(terminal_path, terminal)
    return terminal


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path, terminal_path = output_root / "manifest.json", output_root / "terminal.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed_continuous_ranking_reliability_recoverability"
        or terminal.get("status") != manifest["status"]
        or terminal.get("manifest_sha256") != sha256_path(manifest_path)
        or manifest.get("protocol_sha256") != sha256_path(PROTOCOL_PATH)
        or manifest.get("source_hashes") != _source_hashes()
    ):
        raise ValueError("diagnostic terminal identity changed")
    for name, expected in manifest["artifacts"].items():
        if sha256_path(output_root / name) != expected:
            raise ValueError(f"diagnostic artifact hash changed: {name}")
    expected = load_protocol()["expected_grid"]
    counts = manifest["counts"]
    if (
        int(counts["reference_panels"]) != int(expected["real_target_count"])
        or int(counts["reference_candidate_repeats"]) != int(expected["ranking_target_candidate_repeat_count"])
        or int(counts["recoverability_panels"]) != int(expected["recoverability_target_budget_count"])
        or int(counts["recoverability_candidates"]) != int(expected["recoverability_candidate_count"])
        or int(counts["synthetic_panels"]) != int(expected["synthetic_family_budget_count"])
        or int(counts["reserved_backbones"]) != 0
        or int(counts["overlapindex_fits"]) != 0
    ):
        raise ValueError("diagnostic terminal counts changed")
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
