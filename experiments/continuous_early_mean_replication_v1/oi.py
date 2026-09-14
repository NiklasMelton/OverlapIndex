"""Conditionally compare frozen continuous-OI candidates on the replicated benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import normalize

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_cached_embedding_information_audit_v1 import run as information
from experiments.continuous_early_mean_replication_v1 import extract, reference
from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_real_source_qualification_v1 import prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_early_mean_replication_v1" / "oi"
TIE_ATOL = 1.0e-12


def load_protocol() -> dict[str, Any]:
    protocol = extract.load_protocol()
    expected_methods = (
        "linear_cv_full",
        "raw_cv_max_full",
        "observed_oi_full",
        "broad_rff_full",
        "save_rff_full",
        "broad_rff_pca32",
        "save_rff_pca32",
    )
    if (
        tuple(protocol["continuous_oi"]["methods"]) != expected_methods
        or int(protocol["expected_grid"]["oi_score_row_count"]) != 350
        or int(protocol["expected_grid"]["oi_ranking_row_count"]) != 70
    ):
        raise ValueError("early-mean OI protocol differs from code")
    return protocol


def _source_hashes() -> dict[str, str]:
    paths = {
        "embedding_manifest": extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "reference_terminal": reference.DEFAULT_OUTPUT_ROOT / "terminal.json",
        "reference_manifest": reference.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "reference_raw_results": reference.DEFAULT_OUTPUT_ROOT / "raw_results.json",
        "reference_summary": reference.DEFAULT_OUTPUT_ROOT / "summary.json",
        "extractor": Path(extract.__file__).resolve(),
        "reference_runner": Path(reference.__file__).resolve(),
        "shared_selector": Path(shared.__file__).resolve(),
        "continuous_overlap_index": ROOT / "overlapindex" / "ContinuousOverlapIndex.py",
    }
    return {name: base.sha256_path(path) for name, path in paths.items()}


def _rows(
    cohort: Mapping[str, Any], target: Mapping[str, Any], draw_seed: int,
) -> tuple[str, list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    return reference._rows(cohort, target, draw_seed)


def _linear_cv(
    X: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    *,
    split_seed: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    values = np.asarray(X, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    strata = np.asarray(bins, dtype=np.int64)
    config = protocol["reference_heads"]["linear_ridge"]
    started = time.perf_counter()
    folds = []
    splitter = StratifiedKFold(
        n_splits=int(protocol["reference_heads"]["inner_folds"]),
        shuffle=True,
        random_state=int(split_seed),
    )
    for fold, (train, holdout) in enumerate(splitter.split(values, strata)):
        Z_train, Z_holdout, diagnostic = information._fit_preprocessor(
            values[train], values[holdout], row_l2=True, pca_rank=None,
            seed=base._seed("early-mean-linear-preprocess", split_seed, fold),
        )
        folds.append((Z_train, Z_holdout, targets[train], targets[holdout], diagnostic))
    rows = []
    for mean_loss_lambda in config["mean_loss_lambda"]:
        scores = []
        for Z_train, Z_holdout, y_train, y_holdout, _diagnostic in folds:
            estimator = Ridge(
                alpha=len(Z_train) * float(mean_loss_lambda),
                solver="lsqr",
                tol=float(config["tolerance"]),
            )
            estimator.fit(Z_train, y_train)
            scores.append(float(r2_score(y_holdout, estimator.predict(Z_holdout))))
        rows.append({"mean_loss_lambda": float(mean_loss_lambda), "fold_r2": scores, "mean_r2": float(np.mean(scores))})
    best = sorted(rows, key=lambda row: (-float(row["mean_r2"]), float(row["mean_loss_lambda"])))[0]
    return {
        "score": float(best["mean_r2"]),
        "best_mean_loss_lambda": float(best["mean_loss_lambda"]),
        "parameter_rows": rows,
        "fold_preprocessing": [fold[-1] for fold in folds],
        "wall_seconds": float(time.perf_counter() - started),
    }


def _pca32_shared_pair(
    X: np.ndarray, y: np.ndarray, panel_seed: int, candidate_position: int,
) -> dict[str, Any]:
    values = np.asarray(X, dtype=np.float32)
    targets = np.asarray(y, dtype=np.float64)
    primitive = np.zeros((len(values), 2), dtype=np.float32)
    if values.ndim != 2 or targets.shape != (len(values),) or not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("PCA32 shared-selector inputs are invalid")
    folds = []
    started = time.perf_counter()
    splitter = KFold(n_splits=4, shuffle=True, random_state=int(panel_seed))
    for fold, (train, holdout) in enumerate(splitter.split(values)):
        fit_seed = shared.null_parent._candidate_fit_seed(panel_seed, candidate_position, fold)
        normalized_train = normalize(values[train], norm="l2", axis=1)
        normalized_holdout = normalize(values[holdout], norm="l2", axis=1)
        reducer = PCA(
            n_components=32,
            svd_solver="randomized",
            random_state=int(fit_seed),
            iterated_power=4,
            n_oversamples=10,
        )
        reduced_train = np.asarray(reducer.fit_transform(normalized_train), dtype=np.float32)
        reduced_holdout = np.asarray(reducer.transform(normalized_holdout), dtype=np.float32)
        save_train, save_holdout, save_diagnostics = shared.multiview._single_block(
            "save", reduced_train, reduced_holdout, primitive[train], primitive[holdout], targets[train], fit_seed
        )
        rff_train, rff_holdout, rff_diagnostics = shared.multiview._single_block(
            "rff_mean", reduced_train, reduced_holdout, primitive[train], primitive[holdout], targets[train], fit_seed
        )
        multiview_train, multiview_holdout = shared.multiview._equal_block_concatenation(
            [save_train, rff_train], [save_holdout, rff_holdout]
        )
        estimator = clone(shared.save._base_selector(fit_seed))
        estimator._fit_geometry_only(multiview_train, targets[train])
        bank_seed = shared.null_parent._common_bank_seed(panel_seed, fold)
        bank = shared.null_parent._permutation_bank(len(holdout), 32, bank_seed)
        components = shared.null_parent._fixed_model_components(
            estimator, multiview_holdout, targets[holdout], bank
        )
        null_loss = float(np.mean(components["null_loss_samples"]))
        current = shared.null_parent._score_from_losses(float(components["actual_loss"]), null_loss)
        y_train_scaled = estimator._transform_targets(estimator._validate_Y(targets[train]))
        y_holdout_scaled = estimator._transform_targets(estimator._validate_Y(targets[holdout]))
        broad_kernel = shared.kernel._kernel_fold_components(
            rff_train, rff_holdout, y_train_scaled, y_holdout_scaled, bank
        )
        folds.append(
            {
                "fold": int(fold),
                "fit_seed": int(fit_seed),
                "common_bank_seed": int(bank_seed),
                "pca": {
                    "rank": 32,
                    "input_dimension": int(values.shape[1]),
                    "fit_rows": int(len(train)),
                    "explained_variance_ratio_sum": float(np.sum(reducer.explained_variance_ratio_)),
                },
                "broad_rff": {"kernel_alignment": broad_kernel, "geometry_diagnostics": rff_diagnostics},
                "save_rff_current": {
                    "score": float(current["score"]),
                    "actual_loss": float(components["actual_loss"]),
                    "null_loss": null_loss,
                    "loss_ratio": current["loss_ratio"],
                    "geometry_diagnostics": save_diagnostics,
                },
            }
        )
    broad = shared._aggregate_kernel(
        [{"kernel_alignment": row["broad_rff"]["kernel_alignment"]} for row in folds]
    )
    current_rows = [row["save_rff_current"] for row in folds]
    return {
        "broad_rff": {
            "score": float(broad["feature_bandwidths"]["4.0"]["score"]),
            "kernel_alignment": broad,
        },
        "save_rff_current": {
            "score": float(np.mean([row["score"] for row in current_rows])),
            "mean_actual_loss": float(np.mean([row["actual_loss"] for row in current_rows])),
            "mean_null_loss": float(np.mean([row["null_loss"] for row in current_rows])),
        },
        "folds": folds,
        "fitted_selector_count": 4,
        "wall_seconds": float(time.perf_counter() - started),
    }


def evaluate_candidate(
    protocol: Mapping[str, Any], target: Mapping[str, Any], draw_seed: int, backbone: str,
    candidate_position: int, cohort: Mapping[str, Any], matrices: Mapping[str, Mapping[str, np.ndarray]],
    reference_cell: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_id, draw, _evaluation = _rows(cohort, target, draw_seed)
    train_indices = np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)
    X = np.asarray(matrices[dataset_id][backbone][train_indices], dtype=np.float32)
    y = np.asarray([float(row["target"]) for row in draw], dtype=np.float64)
    bins = np.asarray([int(row["target_bin"]) for row in draw], dtype=np.int64)
    sample_hash = base.sha256_payload([str(row["sample_id"]) for row in draw])
    if sample_hash != reference_cell["training_sample_ids_sha256"]:
        raise ValueError("OI training identities differ from the reference stage")
    split_seed = base._seed("early-mean-reference", target["target_key"], draw_seed, backbone)
    linear = _linear_cv(X, y, bins, split_seed=split_seed, protocol=protocol)
    archived_linear = float(reference_cell["head_inner_best_r2"]["linear_ridge"])
    if abs(float(linear["score"]) - archived_linear) > 1.0e-10:
        raise ValueError(f"linear-CV replay changed: {target['target_key']}::{draw_seed}::{backbone}")
    panel_seed = base._seed("early-mean-oi", target["target_key"], draw_seed)
    primitive = np.zeros((len(X), 2), dtype=np.float32)
    observed = shared.save.evaluate_selector("observed_k4", X, primitive, y, panel_seed)
    full = shared.evaluate_shared_pair(X, primitive, y, panel_seed, candidate_position)
    pca32 = _pca32_shared_pair(X, y, panel_seed, candidate_position)
    raw_cv_max = max(float(value) for value in reference_cell["head_inner_best_r2"].values())
    scores = {
        "linear_cv_full": float(linear["score"]),
        "raw_cv_max_full": float(raw_cv_max),
        "observed_oi_full": float(observed["score"]),
        "broad_rff_full": float(full["broad_rff"]["score"]),
        "save_rff_full": float(full["save_rff_current"]["score"]),
        "broad_rff_pca32": float(pca32["broad_rff"]["score"]),
        "save_rff_pca32": float(pca32["save_rff_current"]["score"]),
    }
    raw_cv_runtime = float(reference_cell["shared_preprocessing_wall_seconds"]) + sum(
        float(row["head_wall_seconds_excluding_shared_preprocessing"])
        for row in reference_cell["heads"].values()
    )
    runtimes = {
        "linear_cv_full": float(linear["wall_seconds"]),
        "raw_cv_max_full": raw_cv_runtime,
        "observed_oi_full": float(observed["wall_seconds"]),
        "broad_rff_full": float(full["wall_seconds"]),
        "save_rff_full": float(full["wall_seconds"]),
        "broad_rff_pca32": float(pca32["wall_seconds"]),
        "save_rff_pca32": float(pca32["wall_seconds"]),
    }
    return {
        "candidate_id": backbone,
        "target_key": str(target["target_key"]),
        "draw_seed": int(draw_seed),
        "truth": {
            "heldout_best_head_envelope_r2": float(reference_cell["envelope_r2"]),
            "heldout_linear_r2": float(reference_cell["head_evaluation_r2"]["linear_ridge"]),
            "heldout_rbf_svr_r2": float(reference_cell["head_evaluation_r2"]["rbf_svr"]),
            "best_head_ids": list(reference_cell["best_head_ids"]),
        },
        "scores": scores,
        "runtime_seconds": runtimes,
        "linear_replay": linear,
        "continuous": {"observed_oi_full": observed, "full": full, "pca32": pca32},
    }


def ranking_metrics(candidate_ids: Sequence[str], truth: Sequence[float], scores: Sequence[float]) -> dict[str, Any]:
    truth_values = np.asarray(truth, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    if truth_values.shape != score_values.shape or truth_values.shape != (5,):
        raise ValueError("early-mean ranking vectors must contain five candidates")
    if len(np.unique(score_values)) < 2 or len(np.unique(truth_values)) < 2:
        rho, status = None, "undefined_constant_ranking"
    else:
        rho, status = float(spearmanr(truth_values, score_values).statistic), "defined"
    selected = np.flatnonzero(score_values >= np.max(score_values) - TIE_ATOL)
    truth_best = np.flatnonzero(truth_values >= np.max(truth_values) - TIE_ATOL)
    score_order = np.argsort(-score_values, kind="stable")
    truth_order = np.argsort(-truth_values, kind="stable")
    score_threshold = float(score_values[score_order[2]])
    truth_threshold = float(truth_values[truth_order[2]])
    selected_top3 = np.flatnonzero(score_values >= score_threshold - TIE_ATOL)
    truth_top3 = np.flatnonzero(truth_values >= truth_threshold - TIE_ATOL)
    pair_credit = []
    for left in range(5):
        for right in range(left + 1, 5):
            delta = float(truth_values[left] - truth_values[right])
            if abs(delta) < 0.01 - TIE_ATOL:
                continue
            score_delta = float(score_values[left] - score_values[right])
            pair_credit.append(0.5 if abs(score_delta) <= TIE_ATOL else float(np.sign(delta) == np.sign(score_delta)))
    return {
        "spearman": rho,
        "spearman_status": status,
        "top1_regret_r2": float(np.max(truth_values) - np.max(truth_values[selected])),
        "exact_best": bool(set(selected).intersection(truth_best)),
        "top3_opportunity_regret_r2": float(np.max(truth_values) - np.max(truth_values[selected_top3])),
        "top3_mean_utility_regret_r2": float(np.mean(truth_values[truth_top3]) - np.mean(truth_values[selected_top3])),
        "top3_set_recall": float(len(set(selected_top3).intersection(truth_top3)) / len(set(truth_top3))),
        "material_pair_accuracy": float(np.mean(pair_credit)) if pair_credit else None,
        "material_pair_count": len(pair_credit),
        "selected_candidate_ids": [candidate_ids[int(index)] for index in selected],
        "selected_top3_candidate_ids": [candidate_ids[int(index)] for index in selected_top3],
    }


def _panel_rows(protocol: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    methods = list(protocol["continuous_oi"]["methods"])
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault((str(candidate["target_key"]), int(candidate["draw_seed"])), []).append(candidate)
    output = []
    expected_order = [str(row["backbone_id"]) for row in protocol["backbones"]]
    for (target_key, draw_seed), rows in sorted(grouped.items()):
        by_id = {str(row["candidate_id"]): row for row in rows}
        if set(by_id) != set(expected_order):
            raise ValueError("early-mean OI panel is incomplete")
        ordered = [by_id[candidate] for candidate in expected_order]
        truth = [float(row["truth"]["heldout_best_head_envelope_r2"]) for row in ordered]
        for method in methods:
            output.append(
                {
                    "target_key": target_key,
                    "draw_seed": draw_seed,
                    "method": method,
                    **ranking_metrics(expected_order, truth, [float(row["scores"][method]) for row in ordered]),
                }
            )
    return output


def summarize(protocol: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranking_rows = _panel_rows(protocol, candidates)
    if len(ranking_rows) != int(protocol["expected_grid"]["oi_ranking_row_count"]):
        raise ValueError("early-mean OI ranking row count changed")
    methods = list(protocol["continuous_oi"]["methods"])
    groups = ["all"] + [str(target["target_key"]) for target in protocol["targets"]]
    summaries = []
    for group in groups:
        for method in methods:
            selected = [row for row in ranking_rows if row["method"] == method and (group == "all" or row["target_key"] == group)]
            rhos = [float(row["spearman"]) for row in selected if row["spearman"] is not None]
            pairs = [float(row["material_pair_accuracy"]) for row in selected if row["material_pair_accuracy"] is not None]
            summaries.append(
                {
                    "group": group,
                    "method": method,
                    "panel_count": len(selected),
                    "defined_spearman_count": len(rhos),
                    "mean_spearman": float(np.mean(rhos)) if rhos else None,
                    "mean_top1_regret_r2": float(np.mean([row["top1_regret_r2"] for row in selected])),
                    "exact_best_rate": float(np.mean([row["exact_best"] for row in selected])),
                    "mean_top3_opportunity_regret_r2": float(np.mean([row["top3_opportunity_regret_r2"] for row in selected])),
                    "mean_top3_utility_regret_r2": float(np.mean([row["top3_mean_utility_regret_r2"] for row in selected])),
                    "mean_top3_set_recall": float(np.mean([row["top3_set_recall"] for row in selected])),
                    "mean_material_pair_accuracy": float(np.mean(pairs)) if pairs else None,
                }
            )
    runtime_rows = []
    totals = {
        method: [float(row["runtime_seconds"][method]) for row in candidates]
        for method in methods
    }
    linear_total = float(np.sum(totals["linear_cv_full"]))
    for method in methods:
        total = float(np.sum(totals[method]))
        runtime_rows.append(
            {
                "method": method,
                "candidate_count": len(totals[method]),
                "aggregate_wall_seconds": total,
                "median_candidate_wall_seconds": float(np.median(totals[method])),
                "relative_aggregate_to_linear_cv": float(total / linear_total),
            }
        )
    return {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_oi_comparison_v1",
        "ranking_rows": ranking_rows,
        "ranking_summaries": summaries,
        "runtime_summaries": runtime_rows,
        "decision": {
            "reference_gate_pass": True,
            "method_promotion": "none_development_diagnostic",
            "interpretation": "compare frozen methods against the replicated held-out best-head envelope; no method or input view may be reselected from these outcomes",
        },
        "overlapindex_fit_count": len(candidates) * 12,
        "reserved_backbones_evaluated": [],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous OI on the replicated early-mean benchmark", "",
        "| Group | Method | Spearman | Top-1 regret | Exact best | Top-3 utility regret | Top-3 recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["ranking_summaries"]:
        rho = "undefined" if row["mean_spearman"] is None else f"{row['mean_spearman']:.3f}"
        lines.append(
            f"| {row['group']} | {row['method']} | {rho} | {row['mean_top1_regret_r2']:.4f} | "
            f"{row['exact_best_rate']:.3f} | {row['mean_top3_utility_regret_r2']:.4f} | {row['mean_top3_set_recall']:.3f} |"
        )
    lines.extend(["", "## Runtime", "", "| Method | Aggregate seconds | Median/candidate | Relative to LP |", "|---|---:|---:|---:|"])
    for row in summary["runtime_summaries"]:
        lines.append(
            f"| {row['method']} | {row['aggregate_wall_seconds']:.3f} | {row['median_candidate_wall_seconds']:.4f} | {row['relative_aggregate_to_linear_cv']:.2f}x |"
        )
    lines.extend([
        "", "## Scope", "",
        "- Reference-head outcomes were frozen and validated before this stage ran.",
        "- Full-input methods are eligible development comparisons; PCA-32 and raw-CV-max are diagnostics.",
        "- No method is promoted from this development benchmark.", "",
    ])
    return "\n".join(lines)


def _identity(source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": dict(source_hashes),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def _validate_grid(protocol: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> None:
    expected = int(protocol["expected_grid"]["reference_cell_count"])
    ids = {(row["target_key"], int(row["draw_seed"]), row["candidate_id"]) for row in candidates}
    if len(candidates) != expected or len(ids) != expected:
        raise ValueError("early-mean OI candidate grid changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    output_root = output_root.resolve()
    protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("early-mean OI execution requires a clean committed source tree")
    reference_raw, reference_summary = reference.load_results()
    if not reference_summary.get("decision", {}).get("oi_stage_authorized"):
        raise RuntimeError("early-mean OI stage is blocked because the frozen reference gate did not pass")
    source_hashes = _source_hashes()
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("early-mean OI terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    identity = _identity(source_hashes)
    checkpoint_path = output_root / "checkpoint.json"
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError("early-mean OI checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("early-mean OI checkpoint identity changed")
    else:
        checkpoint = {"schema_version": 1, "identity": identity, "candidates": []}
    output_root.mkdir(parents=True, exist_ok=True)
    cohort = prepare.load_cohort()
    _embedding_manifest, matrices = extract.load_bank()
    reference_cells = {str(row["cell_id"]): row for row in reference_raw["cells"]}
    complete = {
        (str(row["target_key"]), int(row["draw_seed"]), str(row["candidate_id"]))
        for row in checkpoint["candidates"]
    }
    expected = int(protocol["expected_grid"]["reference_cell_count"])
    for target in protocol["targets"]:
        for draw_seed in target["draw_seeds"]:
            for position, backbone_config in enumerate(protocol["backbones"]):
                backbone = str(backbone_config["backbone_id"])
                key = (str(target["target_key"]), int(draw_seed), backbone)
                if key in complete:
                    continue
                cell_id = f"{target['target_key']}::{draw_seed}::{backbone}"
                checkpoint["candidates"].append(
                    evaluate_candidate(
                        protocol, target, int(draw_seed), backbone, position, cohort, matrices, reference_cells[cell_id]
                    )
                )
                base._finite_json(checkpoint)
                base._atomic_json(checkpoint_path, checkpoint)
                print(f"selector {len(checkpoint['candidates'])}/{expected}: {cell_id}", flush=True)
    _validate_grid(protocol, checkpoint["candidates"])
    summary = summarize(protocol, checkpoint["candidates"])
    raw = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_oi_comparison_v1",
        "protocol": protocol,
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_environment": {key: os.environ.get(key) for key in base.THREAD_KEYS},
        },
        "candidates": checkpoint["candidates"],
        "overlapindex_fit_count": summary["overlapindex_fit_count"],
        "reserved_backbones_evaluated": [],
    }
    base._finite_json(raw)
    base._finite_json(summary)
    base._atomic_json(output_root / "raw_results.json", raw)
    base._atomic_json(output_root / "summary.json", summary)
    base._atomic_bytes(output_root / "report.md", render_report(summary).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_oi_comparison_v1",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "checkpoint_identity": identity,
        "artifacts": {
            name: base.sha256_path(output_root / name)
            for name in ("checkpoint.json", "raw_results.json", "summary.json", "report.md")
        },
        "counts": {
            "candidates": len(checkpoint["candidates"]),
            "score_rows": len(checkpoint["candidates"]) * len(protocol["continuous_oi"]["methods"]),
            "ranking_rows": len(summary["ranking_rows"]),
            "overlapindex_fits": summary["overlapindex_fit_count"],
        },
        "git": {"commit": identity["git_commit"], "dirty": False},
        "decision": summary["decision"],
    }
    base._atomic_json(output_root / "manifest.json", manifest)
    terminal = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_oi_comparison_v1",
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
        terminal.get("status") != "completed_continuous_early_mean_oi_comparison_v1"
        or terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json")
        or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json")
        or terminal.get("decision") != summary.get("decision")
        or manifest.get("source_hashes") != _source_hashes()
    ):
        raise ValueError("early-mean OI artifact identity changed")
    for name, expected in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != expected:
            raise ValueError(f"early-mean OI artifact changed: {name}")
    _validate_grid(protocol, checkpoint["candidates"])
    if len(summary["ranking_rows"]) != int(protocol["expected_grid"]["oi_ranking_row_count"]):
        raise ValueError("early-mean OI summary grid changed")
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
