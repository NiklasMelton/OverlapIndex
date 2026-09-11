"""Run the frozen recoverable-target continuous ranker comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import KFold

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_large_structured_opportunity_benchmark import extract as source_extract
from experiments.continuous_large_structured_opportunity_benchmark import prepare as source_prepare
from experiments.continuous_large_structured_opportunity_benchmark import run as source_run
from experiments.continuous_rff_broad_fusion_scaling import run as broad_scaling
from experiments.continuous_structured_head_discovery import run as geometry


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts" / "continuous_large_recoverable_ranker_comparison"
)

LINEAR = "linear_cv"
BROAD = "continuous_broad_rff"
SAVE = "continuous_save_rff"
CV_MAX = "raw_cv_max"
METHODS = (LINEAR, BROAD, SAVE, CV_MAX)
CANDIDATE_METHODS = (BROAD, SAVE)
GROUPS = (
    "all",
    "rank_changing_nonlinear_opportunity",
    "recoverable_lp_dominated_control",
)
HEADS = ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
TIE_ATOL = 1.0e-12


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    source_prepare._atomic_json(path, value)


def _atomic_bytes(path: Path, value: bytes) -> None:
    source_prepare._atomic_bytes(path, value)


def _target_key(dataset_id: str, target_id: str) -> str:
    return f"{dataset_id}::{target_id}"


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("recoverable ranker protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    grid = protocol["expected_grid"]
    evaluation = protocol["evaluation"]
    success = protocol["success"]
    if (
        protocol.get("status") != "frozen_recoverable_ranker_comparison_before_selector_outcomes"
        or tuple(protocol["rankers"]["methods"]) != METHODS
        or tuple(protocol["rankers"]["candidate_methods"]) != CANDIDATE_METHODS
        or tuple(protocol["evaluation"]["groups"]) != GROUPS
        or tuple(protocol["backbones"]["scout"]) != source_extract.BACKBONES
        or protocol["backbones"]["reserved_evaluated"] != []
        or int(protocol["selector"]["folds"]) != 4
        or int(evaluation["topk"]) != 3
        or float(evaluation["material_pair_difference_r2"]) != 0.01
        or float(evaluation["tie_tolerance"]) != TIE_ATOL
        or {
            key: success[key]
            for key in (
                "minimum_opportunity_spearman_gain_over_linear_cv",
                "minimum_opportunity_top1_regret_reduction_r2",
                "maximum_control_spearman_drop_from_linear_cv",
                "maximum_control_top1_regret_increase_r2",
                "maximum_overall_top3_opportunity_regret_increase_r2",
                "maximum_aggregate_runtime_ratio_to_linear_cv",
            )
        }
        != {
            "minimum_opportunity_spearman_gain_over_linear_cv": 0.05,
            "minimum_opportunity_top1_regret_reduction_r2": 0.005,
            "maximum_control_spearman_drop_from_linear_cv": 0.05,
            "maximum_control_top1_regret_increase_r2": 0.005,
            "maximum_overall_top3_opportunity_regret_increase_r2": 0.002,
            "maximum_aggregate_runtime_ratio_to_linear_cv": 3.0,
        }
        or [
            int(grid[key])
            for key in (
                "target_count",
                "targets_per_stratum",
                "replicates_per_target",
                "panel_count",
                "candidates_per_panel",
                "candidate_count",
                "ranking_method_count",
                "ranking_row_count",
                "ranking_summary_count",
                "continuous_oi_candidate_count",
                "continuous_oi_outer_fit_count",
                "reserved_backbone_count",
            )
        ]
        != [8, 4, 3, 24, 5, 120, 4, 96, 12, 120, 480, 0]
    ):
        raise ValueError("recoverable ranker protocol differs from code")
    return protocol


def validate_source(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = protocol["source"]
    result_root = ROOT / str(descriptor["directory"])
    paths = {
        "protocol_sha256": source_prepare.PROTOCOL_PATH,
        "prepare_sha256": Path(source_prepare.__file__).resolve(),
        "extract_sha256": Path(source_extract.__file__).resolve(),
        "runner_sha256": Path(source_run.__file__).resolve(),
        "raw_results_sha256": result_root / "raw_results.json",
        "summary_sha256": result_root / "summary.json",
        "manifest_sha256": result_root / "manifest.json",
    }
    for key, path in paths.items():
        if not path.is_file() or sha256_path(path) != descriptor[key]:
            raise ValueError(f"recoverable ranker source changed: {path}")
    raw = json.loads((result_root / "raw_results.json").read_text(encoding="utf-8"))
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((result_root / "manifest.json").read_text(encoding="utf-8"))
    source_protocol = source_run.load_protocol()
    if (
        raw.get("status") != descriptor["expected_status"]
        or summary.get("status") != descriptor["expected_status"]
        or manifest.get("status") != descriptor["expected_status"]
        or len(raw.get("panels", ())) != int(descriptor["expected_panels"])
        or sum(len(panel.get("candidates", ())) for panel in raw["panels"])
        != int(descriptor["expected_candidates"])
        or manifest.get("artifacts", {}).get("raw_results.json") != descriptor["raw_results_sha256"]
        or manifest.get("artifacts", {}).get("summary.json") != descriptor["summary_sha256"]
        or int(manifest.get("counts", {}).get("reserved_backbones", -1)) != 0
        or int(manifest.get("counts", {}).get("overlapindex_fits", -1)) != 0
        or canonical_json(raw.get("protocol")) != canonical_json(source_protocol)
    ):
        raise ValueError("recoverable ranker source artifact is malformed")
    expected = {
        stratum: set(values) for stratum, values in protocol["strata"].items()
    }
    observed = {stratum: set() for stratum in expected}
    for row in summary["task_summaries"]:
        stratum = str(row["stratum"])
        if stratum in observed:
            observed[stratum].add(_target_key(str(row["dataset_id"]), str(row["target_id"])))
    if observed != expected or set.intersection(*expected.values()):
        raise ValueError("recoverable ranker frozen stratum identities changed")
    return raw, summary


def _source_hashes() -> dict[str, str]:
    paths = {
        "protocol": PROTOCOL_PATH,
        "runner": Path(__file__).resolve(),
        "source_protocol": source_prepare.PROTOCOL_PATH,
        "source_prepare": Path(source_prepare.__file__).resolve(),
        "source_extract": Path(source_extract.__file__).resolve(),
        "source_runner": Path(source_run.__file__).resolve(),
        "broad_evaluator": Path(broad_scaling.__file__).resolve(),
        "shared_evaluator": Path(shared.__file__).resolve(),
        "continuous_overlap_index": ROOT / "overlapindex" / "ContinuousOverlapIndex.py",
    }
    return {key: sha256_path(path) for key, path in paths.items()}


def _stratum_by_target(protocol: Mapping[str, Any]) -> dict[str, str]:
    return {
        target: stratum
        for stratum, targets in protocol["strata"].items()
        for target in targets
    }


def selected_source_panels(
    raw: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    strata = _stratum_by_target(protocol)
    selected = []
    for panel in raw["panels"]:
        key = _target_key(str(panel["dataset_id"]), str(panel["target_id"]))
        if key in strata:
            selected.append({**dict(panel), "stratum": strata[key]})
    expected = {
        (target, seed)
        for target in strata
        for seed in source_prepare.REPLICATE_SEEDS
    }
    observed = {
        (_target_key(str(panel["dataset_id"]), str(panel["target_id"])), int(panel["seed"]))
        for panel in selected
    }
    if (
        observed != expected
        or len(selected) != int(protocol["expected_grid"]["panel_count"])
        or any(tuple(row["candidate_id"] for row in panel["candidates"]) != source_extract.BACKBONES for panel in selected)
    ):
        raise ValueError("recoverable ranker selected source panel grid is incomplete")
    return sorted(selected, key=lambda row: (str(row["stratum"]), str(row["dataset_id"]), str(row["target_id"]), int(row["seed"])))


def _selector_seed(
    protocol: Mapping[str, Any], dataset_id: str, target_id: str, seed: int
) -> int:
    dataset_position = tuple(protocol["source_protocol"]["datasets"]).index(dataset_id)
    target_position = tuple(protocol["source_protocol"]["datasets"][dataset_id]["targets"]).index(target_id)
    value = int(seed) + 100 * dataset_position + 10 * target_position
    if not 0 <= value < 2**32:
        raise ValueError("recoverable ranker selector seed is outside uint32")
    return value


def evaluate_save_only(
    X: np.ndarray,
    y: np.ndarray,
    panel_seed: int,
    candidate_position: int,
) -> dict[str, Any]:
    values = np.asarray(X, dtype=np.float32)
    targets = np.asarray(y, dtype=np.float64)
    if (
        values.ndim != 2
        or targets.ndim != 1
        or len(values) != len(targets)
        or len(values) < 8
        or not np.isfinite(values).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("SAVE-only candidate inputs are invalid")
    folds = []
    started = time.perf_counter()
    splitter = KFold(n_splits=4, shuffle=True, random_state=int(panel_seed))
    primitive = np.zeros((len(values), 2), dtype=np.float32)
    for fold, (train, holdout) in enumerate(splitter.split(values)):
        fit_seed = shared.null_parent._candidate_fit_seed(panel_seed, candidate_position, fold)
        save_train, save_holdout, save_diagnostics = shared.multiview._single_block(
            "save", values[train], values[holdout], primitive[train], primitive[holdout], targets[train], fit_seed
        )
        rff_train, rff_holdout, rff_diagnostics = shared.multiview._single_block(
            "rff_mean", values[train], values[holdout], primitive[train], primitive[holdout], targets[train], fit_seed
        )
        transformed_train, transformed_holdout = shared.multiview._equal_block_concatenation(
            [save_train, rff_train], [save_holdout, rff_holdout]
        )
        estimator = clone(shared.save._base_selector(fit_seed))
        estimator._fit_geometry_only(transformed_train, targets[train])
        bank_seed = shared.null_parent._common_bank_seed(panel_seed, fold)
        bank = shared.null_parent._permutation_bank(len(holdout), 32, bank_seed)
        components = shared.null_parent._fixed_model_components(
            estimator, transformed_holdout, targets[holdout], bank
        )
        null_loss = float(np.mean(components["null_loss_samples"]))
        score = shared.null_parent._score_from_losses(float(components["actual_loss"]), null_loss)
        folds.append(
            {
                "fold": int(fold),
                "fit_seed": int(fit_seed),
                "common_bank_seed": int(bank_seed),
                "score": float(score["score"]),
                "actual_loss": float(components["actual_loss"]),
                "null_loss": null_loss,
                "loss_ratio": score["loss_ratio"],
                "prototype_count": int(estimator._model.n_clusters_total),
                "save_geometry": save_diagnostics,
                "rff_geometry": rff_diagnostics,
            }
        )
    return {
        "configuration": "save_rff_current_without_unused_broad_alignment",
        "score": float(np.mean([row["score"] for row in folds])),
        "outer_fit_count": len(folds),
        "folds": folds,
        "outer_wall_seconds": float(time.perf_counter() - started),
    }


def evaluate_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    source_candidate: Mapping[str, Any],
    selector_seed: int,
    candidate_position: int,
) -> dict[str, Any]:
    broad = broad_scaling.evaluate_broad_only_candidate_geometry(
        X_train, y_train, selector_seed, candidate_position
    )
    save = evaluate_save_only(X_train, y_train, selector_seed, candidate_position)
    heads = source_candidate["heads"]
    inner = {head: float(heads[head]["inner_best_r2"]) for head in HEADS}
    heldout = {head: float(heads[head]["test_r2"]) for head in HEADS}
    envelope = max(heldout.values())
    if abs(envelope - float(source_candidate["best_head_envelope_r2"])) > TIE_ATOL:
        raise ValueError("recoverable ranker heldout envelope does not reconstruct")
    return {
        "candidate_id": str(source_candidate["candidate_id"]),
        "truth": {
            "heldout_best_head_envelope_r2": float(envelope),
            "heldout_linear_r2": heldout["linear_ridge"],
        },
        "scores": {
            LINEAR: inner["linear_ridge"],
            BROAD: float(broad["score"]),
            SAVE: float(save["score"]),
            CV_MAX: float(max(inner.values())),
        },
        "runtime_seconds": {
            LINEAR: float(heads["linear_ridge"]["wall_seconds"]),
            BROAD: float(broad["outer_wall_seconds"]),
            SAVE: float(save["outer_wall_seconds"]),
            CV_MAX: float(sum(float(heads[head]["wall_seconds"]) for head in HEADS)),
        },
        "continuous": {BROAD: broad, SAVE: save},
    }


def run_panel(
    source_panel: Mapping[str, Any],
    protocol: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_protocol = protocol["source_protocol"]
    dataset_id, target_id, seed = (
        str(source_panel["dataset_id"]),
        str(source_panel["target_id"]),
        int(source_panel["seed"]),
    )
    data = datasets[dataset_id]
    train, evaluation, y_train, _y_eval = source_run._panel_data(data, target_id, seed)
    dataset_position = tuple(source_protocol["datasets"]).index(dataset_id)
    target_position = tuple(source_protocol["datasets"][dataset_id]["targets"]).index(target_id)
    selector_seed = _selector_seed(protocol, dataset_id, target_id, seed)
    candidates = []
    for position, source_candidate in enumerate(source_panel["candidates"]):
        backbone = source_extract.BACKBONES[position]
        if source_candidate["candidate_id"] != backbone:
            raise ValueError("recoverable ranker source candidate order changed")
        X_train, _X_eval, pca = geometry.pca_geometry(
            data["matrices"][backbone],
            train,
            evaluation,
            source_run._geometry_seed(seed, dataset_position, target_position, position),
        )
        if canonical_json(pca) != canonical_json(source_candidate["pca"]):
            raise RuntimeError("recoverable ranker PCA geometry does not reproduce source")
        candidates.append(
            evaluate_candidate(
                X_train,
                y_train,
                source_candidate,
                selector_seed,
                position,
            )
        )
    return {
        "dataset_id": dataset_id,
        "target_id": target_id,
        "stratum": str(source_panel["stratum"]),
        "seed": seed,
        "selector_seed": selector_seed,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _top_indices(values: np.ndarray, depth: int) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    threshold = float(values[order[min(depth, len(order)) - 1]])
    return np.flatnonzero(values >= threshold - TIE_ATOL)


def ranking_metrics(truth: Sequence[float], scores: Sequence[float]) -> dict[str, Any]:
    truth_values = np.asarray(truth, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    if (
        truth_values.ndim != 1
        or score_values.shape != truth_values.shape
        or len(truth_values) != len(source_extract.BACKBONES)
        or not np.isfinite(truth_values).all()
        or not np.isfinite(score_values).all()
    ):
        raise ValueError("recoverable ranker metric inputs are invalid")
    selected = np.flatnonzero(score_values >= np.max(score_values) - TIE_ATOL)
    truth_best = np.flatnonzero(truth_values >= np.max(truth_values) - TIE_ATOL)
    selected_top = _top_indices(score_values, 3)
    truth_top = _top_indices(truth_values, 3)
    correlation = spearmanr(truth_values, score_values).correlation
    pair_credit = []
    for left in range(len(truth_values)):
        for right in range(left + 1, len(truth_values)):
            truth_delta = float(truth_values[left] - truth_values[right])
            if abs(truth_delta) < 0.01 - TIE_ATOL:
                continue
            score_delta = float(score_values[left] - score_values[right])
            if abs(score_delta) <= TIE_ATOL:
                pair_credit.append(0.5)
            else:
                pair_credit.append(float(np.sign(score_delta) == np.sign(truth_delta)))
    return {
        "spearman": None if not np.isfinite(correlation) else float(correlation),
        "top1_regret_r2": float(np.max(truth_values) - np.max(truth_values[selected])),
        "exact_best": bool(set(selected).intersection(truth_best)),
        "top3_opportunity_regret_r2": float(
            np.max(truth_values) - np.max(truth_values[selected_top])
        ),
        "top3_mean_utility_regret_r2": float(
            np.mean(truth_values[truth_top]) - np.mean(truth_values[selected_top])
        ),
        "top3_set_recall": float(
            len(set(selected_top).intersection(truth_top)) / len(set(truth_top))
        ),
        "material_pair_accuracy": float(np.mean(pair_credit)) if pair_credit else None,
        "material_pair_count": len(pair_credit),
        "selected_candidate_ids": [source_extract.BACKBONES[index] for index in selected],
        "selected_top3_candidate_ids": [source_extract.BACKBONES[index] for index in selected_top],
    }


def ranking_rows(panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for panel in panels:
        truth = [row["truth"]["heldout_best_head_envelope_r2"] for row in panel["candidates"]]
        for method in METHODS:
            rows.append(
                {
                    "dataset_id": str(panel["dataset_id"]),
                    "target_id": str(panel["target_id"]),
                    "stratum": str(panel["stratum"]),
                    "seed": int(panel["seed"]),
                    "method": method,
                    **ranking_metrics(truth, [row["scores"][method] for row in panel["candidates"]]),
                }
            )
    if len(rows) != 96:
        raise RuntimeError("recoverable ranker ranking row count changed")
    return rows


def _group(rows: Sequence[Mapping[str, Any]], group: str, method: str) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["method"] == method and (group == "all" or row["stratum"] == group)
    ]


def ranking_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group in GROUPS:
        expected = 24 if group == "all" else 12
        for method in METHODS:
            selected = _group(rows, group, method)
            if len(selected) != expected:
                raise ValueError("recoverable ranker ranking summary grid is incomplete")
            spearman = [float(row["spearman"]) for row in selected if row["spearman"] is not None]
            pair = [float(row["material_pair_accuracy"]) for row in selected if row["material_pair_accuracy"] is not None]
            output.append(
                {
                    "group": group,
                    "method": method,
                    "panel_count": len(selected),
                    "defined_spearman_count": len(spearman),
                    "mean_spearman": float(np.mean(spearman)) if spearman else None,
                    "mean_top1_regret_r2": float(np.mean([row["top1_regret_r2"] for row in selected])),
                    "exact_best_rate": float(np.mean([row["exact_best"] for row in selected])),
                    "mean_top3_opportunity_regret_r2": float(
                        np.mean([row["top3_opportunity_regret_r2"] for row in selected])
                    ),
                    "mean_top3_utility_regret_r2": float(
                        np.mean([row["top3_mean_utility_regret_r2"] for row in selected])
                    ),
                    "mean_top3_set_recall": float(np.mean([row["top3_set_recall"] for row in selected])),
                    "mean_material_pair_accuracy": float(np.mean(pair)) if pair else None,
                }
            )
    if len(output) != 12:
        raise RuntimeError("recoverable ranker ranking summary count changed")
    return output


def runtime_summary(panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    totals = {
        method: [float(row["runtime_seconds"][method]) for panel in panels for row in panel["candidates"]]
        for method in METHODS
    }
    linear_total = float(np.sum(totals[LINEAR]))
    return [
        {
            "method": method,
            "candidate_count": len(totals[method]),
            "aggregate_wall_seconds": float(np.sum(totals[method])),
            "median_candidate_wall_seconds": float(np.median(totals[method])),
            "relative_aggregate_to_linear_cv": float(np.sum(totals[method]) / linear_total),
        }
        for method in METHODS
    ]


def _lookup(summaries: Sequence[Mapping[str, Any]], group: str, method: str) -> Mapping[str, Any]:
    selected = [row for row in summaries if row["group"] == group and row["method"] == method]
    if len(selected) != 1:
        raise ValueError("recoverable ranker summary row is not unique")
    return selected[0]


def decision(
    summaries: Sequence[Mapping[str, Any]],
    runtimes: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rule = protocol["success"]
    baseline_opportunity = _lookup(summaries, GROUPS[1], LINEAR)
    baseline_control = _lookup(summaries, GROUPS[2], LINEAR)
    baseline_all = _lookup(summaries, GROUPS[0], LINEAR)
    runtime = {row["method"]: row for row in runtimes}
    evaluations = []
    for method in CANDIDATE_METHODS:
        opportunity = _lookup(summaries, GROUPS[1], method)
        control = _lookup(summaries, GROUPS[2], method)
        overall = _lookup(summaries, GROUPS[0], method)
        checks = {
            "opportunity_spearman_gain": float(opportunity["mean_spearman"]) - float(baseline_opportunity["mean_spearman"])
            >= float(rule["minimum_opportunity_spearman_gain_over_linear_cv"]) - TIE_ATOL,
            "opportunity_top1_regret_reduction": float(baseline_opportunity["mean_top1_regret_r2"])
            - float(opportunity["mean_top1_regret_r2"])
            >= float(rule["minimum_opportunity_top1_regret_reduction_r2"]) - TIE_ATOL,
            "control_spearman_preserved": float(control["mean_spearman"])
            >= float(baseline_control["mean_spearman"])
            - float(rule["maximum_control_spearman_drop_from_linear_cv"])
            - TIE_ATOL,
            "control_top1_regret_preserved": float(control["mean_top1_regret_r2"])
            <= float(baseline_control["mean_top1_regret_r2"])
            + float(rule["maximum_control_top1_regret_increase_r2"])
            + TIE_ATOL,
            "overall_top3_preserved": float(overall["mean_top3_opportunity_regret_r2"])
            <= float(baseline_all["mean_top3_opportunity_regret_r2"])
            + float(rule["maximum_overall_top3_opportunity_regret_increase_r2"])
            + TIE_ATOL,
            "runtime_competitive": float(runtime[method]["relative_aggregate_to_linear_cv"])
            <= float(rule["maximum_aggregate_runtime_ratio_to_linear_cv"]) + TIE_ATOL,
        }
        evaluations.append(
            {
                "method": method,
                "passed": bool(all(checks.values())),
                "checks": checks,
                "contrasts": {
                    "opportunity_spearman_gain": float(opportunity["mean_spearman"])
                    - float(baseline_opportunity["mean_spearman"]),
                    "opportunity_top1_regret_reduction_r2": float(baseline_opportunity["mean_top1_regret_r2"])
                    - float(opportunity["mean_top1_regret_r2"]),
                    "control_spearman_drop": float(baseline_control["mean_spearman"])
                    - float(control["mean_spearman"]),
                    "control_top1_regret_increase_r2": float(control["mean_top1_regret_r2"])
                    - float(baseline_control["mean_top1_regret_r2"]),
                    "overall_top3_opportunity_regret_increase_r2": float(overall["mean_top3_opportunity_regret_r2"])
                    - float(baseline_all["mean_top3_opportunity_regret_r2"]),
                    "runtime_ratio_to_linear_cv": float(runtime[method]["relative_aggregate_to_linear_cv"]),
                },
            }
        )
    passed = [row["method"] for row in evaluations if row["passed"]]
    return {
        "status": rule["result_if_pass"] if passed else rule["result_if_fail"],
        "any_candidate_passed": bool(passed),
        "passing_methods": passed,
        "evaluations": evaluations,
        "method_promotion": None,
    }


def summarize(panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    if len(panels) != 24 or sum(len(panel["candidates"]) for panel in panels) != 120:
        raise ValueError("recoverable ranker completed panel grid is incomplete")
    rows = ranking_rows(panels)
    summaries = ranking_summaries(rows)
    runtimes = runtime_summary(panels)
    return {
        "schema_version": 1,
        "status": "completed_recoverable_ranker_comparison",
        "scope": dict(protocol["scope"]),
        "panel_count": len(panels),
        "candidate_count": sum(len(panel["candidates"]) for panel in panels),
        "ranking_rows": rows,
        "ranking_summaries": summaries,
        "runtime_summary": runtimes,
        "decision": decision(summaries, runtimes, protocol),
        "reserved_backbones_evaluated": [],
        "continuous_oi_outer_fit_count": 480,
    }


def report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Recoverable-target continuous ranker comparison",
        "",
        f"Decision: **{summary['decision']['status']}**.",
        "",
        "This development comparison includes every recoverable target from the frozen 20-target benchmark: four rank-changing nonlinear opportunities and four LP-dominated controls.",
        "",
        "| Group | Method | Spearman | Top-1 regret | Exact best | Top-3 opportunity regret | Top-3 utility regret | Top-3 recall | Pair accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["ranking_summaries"]:
        lines.append(
            f"| {row['group']} | {row['method']} | {row['mean_spearman']:.4f} | "
            f"{row['mean_top1_regret_r2']:.5f} | {row['exact_best_rate']:.3f} | "
            f"{row['mean_top3_opportunity_regret_r2']:.5f} | {row['mean_top3_utility_regret_r2']:.5f} | "
            f"{row['mean_top3_set_recall']:.3f} | {row['mean_material_pair_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| Method | Aggregate seconds | Median candidate seconds | Relative to linear CV |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["runtime_summary"]:
        lines.append(
            f"| {row['method']} | {row['aggregate_wall_seconds']:.3f} | "
            f"{row['median_candidate_wall_seconds']:.4f} | {row['relative_aggregate_to_linear_cv']:.2f}x |"
        )
    lines.extend(["", "## Frozen gate", ""])
    for evaluation in summary["decision"]["evaluations"]:
        lines.append(f"### {evaluation['method']}")
        lines.append("")
        for check, passed in evaluation["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: `{check}`")
        lines.append("")
    lines.extend(
        [
            "## Guardrails",
            "",
            "- No target or method was selected after continuous ranker outcomes.",
            "- Held-out envelope outcomes were used only for final metrics.",
            "- No reserved backbone was evaluated.",
            "- This development result cannot promote a production method.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    protocol = load_protocol()
    raw, _summary = validate_source(protocol)
    # Read the original protocol file because sorted artifact JSON does not preserve
    # the mapping order used to derive the source experiment's positional seeds.
    protocol = {**protocol, "source_protocol": source_run.load_protocol()}
    source_panels = selected_source_panels(raw, protocol)
    datasets = source_run.load_dataset()
    source_hashes = _source_hashes()
    output_root = output_root.resolve()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        raise FileExistsError("recoverable ranker terminal artifact already exists")
    checkpoints = output_root / "checkpoints"
    if checkpoints.exists() and not resume:
        raise FileExistsError("recoverable ranker checkpoints exist; pass --resume")
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    for position, source_panel in enumerate(source_panels, start=1):
        name = f"{source_panel['dataset_id']}__{source_panel['target_id']}__seed_{source_panel['seed']}.json"
        checkpoint = checkpoints / name
        if checkpoint.exists():
            if not resume:
                raise FileExistsError(f"recoverable ranker checkpoint exists: {name}")
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("source_hashes") != source_hashes:
                raise ValueError(f"recoverable ranker checkpoint source changed: {name}")
            panel = payload["panel"]
        else:
            started = time.perf_counter()
            panel = run_panel(source_panel, protocol, datasets)
            _atomic_json(
                checkpoint,
                {
                    "schema_version": 1,
                    "source_hashes": source_hashes,
                    "panel_wall_seconds": float(time.perf_counter() - started),
                    "panel": panel,
                },
            )
        panels.append(panel)
        print(
            f"completed {position}/{len(source_panels)}: {panel['dataset_id']}/{panel['target_id']}/seed={panel['seed']}",
            flush=True,
        )
    summary = summarize(panels, protocol)
    raw_output = {
        "schema_version": 1,
        "status": "completed_recoverable_ranker_comparison",
        "protocol": {key: value for key, value in protocol.items() if key != "source_protocol"},
        "source_hashes": source_hashes,
        "panels": panels,
        "reserved_backbones_evaluated": [],
    }
    raw_path, summary_path, report_path = (
        output_root / "raw_results.json",
        output_root / "summary.json",
        output_root / "report.md",
    )
    _atomic_json(raw_path, raw_output)
    _atomic_json(summary_path, summary)
    _atomic_bytes(report_path, report(summary).encode("utf-8"))
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    manifest = {
        "schema_version": 1,
        "status": summary["status"],
        "decision_status": summary["decision"]["status"],
        "source_hashes": source_hashes,
        "artifacts": {
            "raw_results.json": sha256_path(raw_path),
            "summary.json": sha256_path(summary_path),
            "report.md": sha256_path(report_path),
        },
        "counts": {
            "panels": len(panels),
            "candidates": sum(len(panel["candidates"]) for panel in panels),
            "continuous_oi_outer_fits": 480,
            "reserved_backbones": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "method_promotion": None,
    }
    _atomic_json(terminal, manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(canonical_json(run(args.output_root, resume=args.resume)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
