"""Run the frozen smooth nonlinear extension of the cached-embedding audit."""

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
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import sklearn
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR

from experiments.continuous_cached_embedding_information_audit_v1 import run as information
from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_reference_library_qualification import evaluator as qualified
from experiments.continuous_real_source_qualification_v1 import run as real_screen


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_cached_embedding_smooth_head_audit_v1" / "audit"
INFORMATION_ROOT = ROOT / "artifacts" / "continuous_cached_embedding_information_audit_v1" / "audit"
SCREEN_ROOT = ROOT / "artifacts" / "continuous_real_source_qualification_v1" / "screen"
TIE_ATOL = 1.0e-12


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or base.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("smooth-head audit protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_cached_embedding_smooth_head_audit_v1_before_outcomes":
        raise ValueError("smooth-head audit protocol is not frozen")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "information_audit_protocol": information.PROTOCOL_PATH,
        "information_audit_raw_results": INFORMATION_ROOT / "raw_results.json",
        "information_audit_summary": INFORMATION_ROOT / "summary.json",
        "information_audit_manifest": INFORMATION_ROOT / "manifest.json",
        "information_audit_terminal": INFORMATION_ROOT / "terminal.json",
        "archived_real_source_raw_results": SCREEN_ROOT / "raw_results.json",
    }


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"smooth-head audit input is missing: {missing}")
    observed = {name: base.sha256_path(path) for name, path in paths.items()}
    ancestry = protocol["ancestry"]
    if any(observed[name] != ancestry[f"{name}_sha256"] for name in observed):
        raise ValueError("smooth-head audit ancestry changed")
    information.validate_artifacts(INFORMATION_ROOT)
    real_screen.validate_artifacts(SCREEN_ROOT)
    return observed


def _grid(protocol: Mapping[str, Any]) -> list[dict[str, float]]:
    config = protocol["rbf_svr"]
    return [
        {"C_standardized": float(C), "epsilon_standardized": float(epsilon), "bandwidth_factor": float(factor)}
        for C, epsilon, factor in itertools.product(
            config["C_standardized_target_units"],
            config["epsilon_standardized_target_units"],
            config["bandwidth_factors"],
        )
    ]


def _distance_views(X_train: np.ndarray, X_other: np.ndarray, factors: Sequence[float]) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], float]:
    train_squared = euclidean_distances(X_train, X_train, squared=True)
    np.maximum(train_squared, 0.0, out=train_squared)
    direct_distances = pdist(np.asarray(X_train, dtype=np.float64), metric="euclidean")
    positive = direct_distances[direct_distances > 0.0]
    if not len(positive):
        raise ValueError("RBF calibration requires at least one nonzero training distance")
    median_distance = float(np.median(positive))
    other_squared = euclidean_distances(X_other, X_train, squared=True)
    np.maximum(other_squared, 0.0, out=other_squared)
    kernels = {}
    for factor in factors:
        factor = float(factor)
        denominator = 2.0 * (factor * median_distance) ** 2
        kernels[factor] = (np.exp(-train_squared / denominator), np.exp(-other_squared / denominator))
    return kernels, median_distance


def _fit_precomputed(
    train_kernel: np.ndarray,
    y_train: np.ndarray,
    other_kernel: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    standardized, mean, scale = qualified._standardize_target(y_train)
    estimator = SVR(
        kernel="precomputed",
        C=float(params["C_standardized"]),
        epsilon=float(params["epsilon_standardized"]),
    )
    estimator.fit(train_kernel, standardized)
    return np.asarray(estimator.predict(other_kernel) * scale + mean, dtype=np.float64)


def _best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    values = np.asarray([row["mean_r2"] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("smooth-head tuning produced a nonfinite score")
    return rows[int(np.flatnonzero(values >= np.max(values) - TIE_ATOL)[0])]


def evaluate_rbf(
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
    y_train = np.asarray(y_train, dtype=np.float64)
    y_evaluation = np.asarray(y_evaluation, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.int64)
    ids = tuple(str(value) for value in sample_ids)
    if len(ids) != len(set(ids)) or len(ids) != len(y_train) or bins.shape != y_train.shape:
        raise ValueError("smooth-head training rows or identities are invalid")
    factors = [float(value) for value in protocol["rbf_svr"]["bandwidth_factors"]]
    splitter = StratifiedKFold(
        n_splits=int(protocol["rbf_svr"]["inner_folds"]), shuffle=True, random_state=int(split_seed)
    )
    folds = []
    fold_audit = []
    preprocessing_started = time.perf_counter()
    for fold, (train_position, validation_position) in enumerate(splitter.split(X_train, bins)):
        train_ids = {ids[int(position)] for position in train_position}
        validation_ids = {ids[int(position)] for position in validation_position}
        if train_ids.intersection(validation_ids):
            raise ValueError("smooth-head inner fold leaks a sample identity")
        Z_train, Z_validation, diagnostic = information._fit_preprocessor(
            X_train[train_position], X_train[validation_position],
            row_l2=bool(view["row_l2"]), pca_rank=view["pca_rank"],
            seed=base._seed("cached-audit-preprocess", split_seed, fold),
        )
        kernels, median = _distance_views(Z_train, Z_validation, factors)
        folds.append((kernels, y_train[train_position], y_train[validation_position]))
        fold_audit.append({
            "fold": fold, "train_rows": int(len(train_position)), "validation_rows": int(len(validation_position)),
            "identity_overlap_count": 0,
            "train_sample_ids_sha256": base.sha256_payload(sorted(train_ids)),
            "validation_sample_ids_sha256": base.sha256_payload(sorted(validation_ids)),
            "preprocessing": diagnostic, "median_nonzero_training_distance": median,
        })
    Z_full, Z_evaluation, full_diagnostic = information._fit_preprocessor(
        X_train, X_evaluation, row_l2=bool(view["row_l2"]), pca_rank=view["pca_rank"],
        seed=base._seed("cached-audit-preprocess", split_seed, "full"),
    )
    full_kernels, full_median = _distance_views(Z_full, Z_evaluation, factors)
    preprocessing_seconds = float(time.perf_counter() - preprocessing_started)
    started = time.perf_counter()
    parameter_rows = []
    for params in _grid(protocol):
        scores, warning_types = [], set()
        for kernels, fold_y_train, fold_y_validation in folds:
            train_kernel, validation_kernel = kernels[float(params["bandwidth_factor"])]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                prediction = _fit_precomputed(train_kernel, fold_y_train, validation_kernel, params)
            warning_types.update(type(item.message).__name__ for item in caught)
            scores.append(float(r2_score(fold_y_validation, prediction)))
        parameter_rows.append({"params": params, "fold_r2": scores, "mean_r2": float(np.mean(scores)), "warning_types": sorted(warning_types)})
    best = _best_row(parameter_rows)
    train_kernel, evaluation_kernel = full_kernels[float(best["params"]["bandwidth_factor"])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prediction = _fit_precomputed(train_kernel, y_train, evaluation_kernel, best["params"])
    if prediction.shape != y_evaluation.shape or not np.isfinite(prediction).all():
        raise ValueError("smooth-head prediction is invalid")
    return {
        "best_params": dict(best["params"]), "inner_best_r2": float(best["mean_r2"]),
        "parameter_rows": parameter_rows, "evaluation_r2": float(r2_score(y_evaluation, prediction)),
        "prediction": prediction.tolist(), "prediction_sha256": hashlib.sha256(prediction.tobytes(order="C")).hexdigest(),
        "final_warning_types": sorted({type(item.message).__name__ for item in caught}),
        "model_fit_count": len(parameter_rows) * len(folds) + 1,
        "head_wall_seconds_excluding_shared_preprocessing": float(time.perf_counter() - started),
        "shared_kernel_preprocessing_wall_seconds": preprocessing_seconds,
        "full_preprocessing": full_diagnostic, "full_median_nonzero_training_distance": full_median,
        "fold_audit": fold_audit,
    }


def _information_cell(raw: Mapping[str, Any], target_key: str, backbone: str, view_id: str) -> Mapping[str, Any]:
    return next(
        row for row in raw["cells"]
        if row["target_key"] == target_key and row["candidate_id"] == backbone and row["view"]["view_id"] == view_id
    )


def calibration_replay(
    protocol: Mapping[str, Any], target: Mapping[str, Any], backbone: str,
    cohort: Mapping[str, Any], matrices: Mapping[str, Mapping[str, np.ndarray]], archived_raw: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_id, draw, _, evaluation = information._target_rows(cohort, target)
    matrix = matrices[dataset_id][backbone]
    train_indices = np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)
    evaluation_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    y_train = np.asarray([float(row["target"]) for row in draw], dtype=np.float64)
    y_evaluation = np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64)
    Z_train, Z_evaluation, _ = base._fit_preprocessor(matrix[train_indices], matrix[evaluation_indices])
    panel = next(row for row in archived_raw["panels"] if row["target_key"] == target["target_key"])
    candidate = next(row for row in panel["candidates"] if row["candidate_id"] == backbone)
    archived_draw = next(row for row in candidate["draws"] if int(row["draw_seed"]) == int(target["draw_seed"]))
    archived_head = archived_draw["heads"]["rbf_svr"]
    params = archived_head["best_params"]
    direct = qualified._fit_predict("rbf_svr", params, Z_train, y_train, Z_evaluation)
    kernels, _ = _distance_views(Z_train, Z_evaluation, [float(params["bandwidth_factor"])])
    train_kernel, evaluation_kernel = kernels[float(params["bandwidth_factor"])]
    precomputed = _fit_precomputed(train_kernel, y_train, evaluation_kernel, params)
    return {
        "calibration_id": f"{target['target_key']}::{backbone}",
        "target_key": str(target["target_key"]), "candidate_id": backbone,
        "archived_best_params": dict(params),
        "archived_evaluation_r2": float(archived_head["evaluation_r2"]),
        "direct_evaluation_r2": float(r2_score(y_evaluation, direct)),
        "precomputed_evaluation_r2": float(r2_score(y_evaluation, precomputed)),
        "maximum_absolute_direct_precomputed_prediction_delta": float(np.max(np.abs(direct - precomputed))),
        "maximum_absolute_direct_precomputed_prediction_delta_standardized": float(
            np.max(np.abs(direct - precomputed)) / np.std(y_train, ddof=0)
        ),
        "absolute_archived_precomputed_evaluation_r2_delta": float(abs(float(archived_head["evaluation_r2"]) - r2_score(y_evaluation, precomputed))),
    }


def build_cell(
    protocol: Mapping[str, Any], target: Mapping[str, Any], view: Mapping[str, Any], backbone: str,
    cohort: Mapping[str, Any], matrices: Mapping[str, Mapping[str, np.ndarray]], information_raw: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_id, draw_600, draw_2000, evaluation = information._target_rows(cohort, target)
    draw = draw_600 if int(view["training_rows"]) == 600 else draw_2000
    matrix = matrices[dataset_id][backbone]
    train_indices = np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)
    evaluation_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    split_seed = base._seed("cached-information-audit", target["target_key"], backbone, view["view_id"])
    result = evaluate_rbf(
        matrix[train_indices], np.asarray([float(row["target"]) for row in draw], dtype=np.float64),
        np.asarray([int(row["target_bin"]) for row in draw], dtype=np.int64), [str(row["sample_id"]) for row in draw],
        matrix[evaluation_indices], np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64),
        view=view, split_seed=split_seed, protocol=protocol,
    )
    source = _information_cell(information_raw, str(target["target_key"]), backbone, str(view["view_id"]))
    scores = dict(source["head_evaluation_r2"])
    scores["rbf_svr"] = float(result["evaluation_r2"])
    envelope = max(scores.values())
    return {
        "cell_id": f"{target['target_key']}::{backbone}::{view['view_id']}",
        "target_key": str(target["target_key"]), "dataset_id": dataset_id,
        "candidate_id": backbone, "view": dict(view),
        "training_sample_ids_sha256": source["training_sample_ids_sha256"],
        "evaluation_sample_ids_sha256": source["evaluation_sample_ids_sha256"],
        "head_evaluation_r2": scores, "envelope_r2": float(envelope),
        "best_head_ids": [head for head, score in scores.items() if score >= envelope - TIE_ATOL],
        "rbf_svr": result,
    }


def _ranking_summary(cells: Sequence[Mapping[str, Any]], target_key: str, view_id: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    rows = sorted(
        (row for row in cells if row["target_key"] == target_key and row["view"]["view_id"] == view_id),
        key=lambda row: str(row["candidate_id"]),
    )
    linear = np.asarray([row["head_evaluation_r2"]["linear_ridge"] for row in rows], dtype=np.float64)
    envelope = np.asarray([row["envelope_r2"] for row in rows], dtype=np.float64)
    if len(np.unique(linear)) < 2 or len(np.unique(envelope)) < 2:
        rho, rho_status = None, "undefined_constant_ranking"
    else:
        rho, rho_status = float(spearmanr(linear, envelope).statistic), "defined"
    linear_best = np.flatnonzero(linear >= np.max(linear) - TIE_ATOL)
    regret = float(np.max(envelope) - np.max(envelope[linear_best]))
    gains = envelope - linear
    thresholds = protocol["ranking_diagnostics"]
    rank_conflict = bool(
        regret >= float(thresholds["material_linear_top1_regret_r2"])
        or (rho is not None and rho <= float(thresholds["material_linear_to_envelope_spearman_upper"]))
    )
    return {
        "target_key": target_key, "view_id": view_id,
        "candidates": [{
            "candidate_id": row["candidate_id"], "linear_r2": float(row["head_evaluation_r2"]["linear_ridge"]),
            "rbf_svr_r2": float(row["head_evaluation_r2"]["rbf_svr"]), "envelope_r2": float(row["envelope_r2"]),
            "nonlinear_gain_r2": float(row["envelope_r2"] - row["head_evaluation_r2"]["linear_ridge"]),
            "best_head_ids": list(row["best_head_ids"]),
        } for row in rows],
        "linear_to_envelope_spearman": rho, "linear_to_envelope_spearman_status": rho_status,
        "linear_top1_regret_r2": regret, "maximum_nonlinear_gain_r2": float(np.max(gains)),
        "linear_best_candidate_ids": [rows[int(position)]["candidate_id"] for position in linear_best],
        "envelope_best_candidate_ids": [rows[int(position)]["candidate_id"] for position in np.flatnonzero(envelope >= np.max(envelope) - TIE_ATOL)],
        "material_rank_conflict": rank_conflict,
    }


def summarize(protocol: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    calibration = protocol["calibration"]
    calibration_pass = all(
        row["maximum_absolute_direct_precomputed_prediction_delta_standardized"] <= float(calibration["maximum_absolute_standardized_prediction_delta"])
        and row["absolute_archived_precomputed_evaluation_r2_delta"] <= float(calibration["maximum_absolute_evaluation_r2_delta"])
        for row in checkpoint["calibrations"]
    )
    views = [
        _ranking_summary(checkpoint["cells"], str(target["target_key"]), str(view["view_id"]), protocol)
        for target in protocol["targets"] for view in protocol["evaluated_views"]
    ]
    conflicts = [row for row in views if row["material_rank_conflict"]]
    material_gain = [row for row in views if row["maximum_nonlinear_gain_r2"] >= float(protocol["ranking_diagnostics"]["material_nonlinear_gain_r2"])]
    return {
        "schema_version": 1, "status": "completed_continuous_cached_embedding_smooth_head_audit_v1",
        "calibration": {"pass": bool(calibration_pass), "rows": checkpoint["calibrations"]},
        "view_summaries": views,
        "decision": {
            "calibration_pass": bool(calibration_pass),
            "material_rank_conflict_count": len(conflicts),
            "material_nonlinear_gain_view_count": len(material_gain),
            "next_stage": (
                "blocked_calibration" if not calibration_pass else
                "repeat_on_all_five_backbones_and_multiple_draws" if conflicts else
                "intermediate_layer_and_pooling_audit"
            ),
            "method_promotion": "none_reference_only",
        },
        "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous cached-embedding smooth-head audit V1", "",
        "| Target | View | Linear rho | LP regret | Max nonlinear gain | Linear winner | Envelope winner | Conflict |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in summary["view_summaries"]:
        rho = "undefined" if row["linear_to_envelope_spearman"] is None else f"{row['linear_to_envelope_spearman']:.3f}"
        lines.append(
            f"| {row['target_key']} | {row['view_id']} | {rho} | {row['linear_top1_regret_r2']:.4f} | "
            f"{row['maximum_nonlinear_gain_r2']:.4f} | {', '.join(row['linear_best_candidate_ids'])} | "
            f"{', '.join(row['envelope_best_candidate_ids'])} | {str(row['material_rank_conflict']).lower()} |"
        )
    decision = summary["decision"]
    lines.extend([
        "", "## Decision", "",
        f"- Precomputed-kernel calibration passed: **{str(decision['calibration_pass']).lower()}**.",
        f"- Material nonlinear-gain views: **{decision['material_nonlinear_gain_view_count']}**.",
        f"- Material rank-conflict views: **{decision['material_rank_conflict_count']}**.",
        f"- Next stage: **{decision['next_stage']}**.",
        "- This diagnostic promotes no OverlapIndex method.", "",
    ])
    return "\n".join(lines)


def _identity(source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "runner_sha256": base.sha256_path(Path(__file__)),
        "source_hashes": dict(source_hashes),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def validate_grid(protocol: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    if len(checkpoint["calibrations"]) != int(protocol["expected_grid"]["calibration_count"]):
        raise ValueError("smooth-head calibration grid changed")
    if len(checkpoint["cells"]) != int(protocol["expected_grid"]["cell_count"]):
        raise ValueError("smooth-head executed grid changed")
    if len({row["calibration_id"] for row in checkpoint["calibrations"]}) != len(checkpoint["calibrations"]):
        raise ValueError("smooth-head calibration identities repeat")
    if len({row["cell_id"] for row in checkpoint["cells"]}) != len(checkpoint["cells"]):
        raise ValueError("smooth-head cell identities repeat")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    output_root = output_root.resolve(); protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("smooth-head audit execution requires a clean committed source tree")
    source_hashes = _source_hashes(protocol)
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("smooth-head terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    identity = _identity(source_hashes); checkpoint_path = output_root / "checkpoint.json"
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError("smooth-head checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("smooth-head checkpoint identity changed")
    else:
        checkpoint = {"schema_version": 1, "identity": identity, "calibrations": [], "cells": []}
    output_root.mkdir(parents=True, exist_ok=True)
    cohort, matrices, _ = information._load_inputs(protocol)
    information_raw = json.loads((INFORMATION_ROOT / "raw_results.json").read_text(encoding="utf-8"))
    archived_raw = json.loads((SCREEN_ROOT / "raw_results.json").read_text(encoding="utf-8"))
    calibrated = {row["calibration_id"] for row in checkpoint["calibrations"]}
    for target in protocol["targets"]:
        for backbone in protocol["backbones"]:
            calibration_id = f"{target['target_key']}::{backbone}"
            if calibration_id not in calibrated:
                checkpoint["calibrations"].append(calibration_replay(protocol, target, backbone, cohort, matrices, archived_raw))
                base._finite_json(checkpoint); base._atomic_json(checkpoint_path, checkpoint)
                print(f"calibration {len(checkpoint['calibrations'])}/{protocol['expected_grid']['calibration_count']}: {calibration_id}", flush=True)
    complete = {row["cell_id"] for row in checkpoint["cells"]}
    for target in protocol["targets"]:
        for backbone in protocol["backbones"]:
            for view in protocol["evaluated_views"]:
                cell_id = f"{target['target_key']}::{backbone}::{view['view_id']}"
                if cell_id in complete:
                    continue
                checkpoint["cells"].append(build_cell(protocol, target, view, backbone, cohort, matrices, information_raw))
                base._finite_json(checkpoint); base._atomic_json(checkpoint_path, checkpoint)
                print(f"cell {len(checkpoint['cells'])}/{protocol['expected_grid']['cell_count']}: {cell_id}", flush=True)
    validate_grid(protocol, checkpoint)
    summary = summarize(protocol, checkpoint)
    raw = {
        "schema_version": 1, "status": "completed_continuous_cached_embedding_smooth_head_audit_v1",
        "protocol": protocol, "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": source_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "thread_environment": {key: os.environ.get(key) for key in base.THREAD_KEYS}},
        "calibrations": checkpoint["calibrations"], "cells": checkpoint["cells"],
        "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }
    base._finite_json(raw); base._finite_json(summary)
    for name, value in (("raw_results.json", raw), ("summary.json", summary)):
        base._atomic_json(output_root / name, value)
    base._atomic_bytes(output_root / "report.md", render_report(summary).encode("utf-8"))
    manifest = {
        "schema_version": 1, "status": "completed_continuous_cached_embedding_smooth_head_audit_v1",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": source_hashes, "checkpoint_identity": identity,
        "artifacts": {name: base.sha256_path(output_root / name) for name in ("checkpoint.json", "raw_results.json", "summary.json", "report.md")},
        "counts": {"calibrations": len(checkpoint["calibrations"]), "cells": len(checkpoint["cells"]), "overlapindex_fits": 0, "reserved_backbones": 0},
        "git": {"commit": identity["git_commit"], "dirty": False}, "decision": summary["decision"],
    }
    base._atomic_json(output_root / "manifest.json", manifest)
    terminal = {
        "schema_version": 1, "status": "completed_continuous_cached_embedding_smooth_head_audit_v1",
        "manifest_sha256": base.sha256_path(output_root / "manifest.json"), "summary_sha256": base.sha256_path(output_root / "summary.json"),
        "decision": summary["decision"],
    }
    base._atomic_json(terminal_path, terminal)
    return validate_artifacts(output_root)


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve(); protocol = load_protocol()
    terminal = json.loads((output_root / "terminal.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output_root / "checkpoint.json").read_text(encoding="utf-8"))
    if terminal.get("status") != "completed_continuous_cached_embedding_smooth_head_audit_v1":
        raise ValueError("smooth-head terminal status changed")
    if terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json") or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json"):
        raise ValueError("smooth-head terminal identity changed")
    if terminal.get("decision") != summary.get("decision") or manifest.get("source_hashes") != _source_hashes(protocol):
        raise ValueError("smooth-head summary or source identity changed")
    for name, expected in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != expected:
            raise ValueError(f"smooth-head artifact changed: {name}")
    validate_grid(protocol, checkpoint)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume)
    print(base.canonical_json(result)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
