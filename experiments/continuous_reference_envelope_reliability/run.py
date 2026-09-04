"""Audit reliability of the frozen continuous four-head reference envelope.

The archive part only reads the completed large structured benchmark.  The
budget part refits the same four head grids on predeclared unions of the
source training cohorts and records predictions on the already fixed
evaluation cohort.  This module is diagnostic only; it never fits
OverlapIndex or evaluates a reserved backbone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold

from experiments.continuous_large_structured_opportunity_benchmark import extract as source_extract
from experiments.continuous_large_structured_opportunity_benchmark import prepare as source_prepare
from experiments.continuous_large_structured_opportunity_benchmark import run as source_run
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_structured_head_discovery import run as geometry


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_reference_envelope_reliability"

HEADS = ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
BACKBONES = tuple(source_extract.BACKBONES)
REPLICATE_SEEDS = (4103, 4201, 4303)
TIE_ATOL = 1.0e-12
MATERIAL_ATOL = 0.01
ARCHIVE_GROUPS = (
    "all",
    "recoverable_lp_dominated_control",
    "rank_changing_nonlinear_opportunity",
    "unusable_for_ranking",
)
BUDGET_GROUPS = ("all", "recoverable_lp_dominated_control", "rank_changing_nonlinear_opportunity")
BUDGET_BASE_SEED = 2026091501
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2026091402


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _target_key(dataset_id: str, target_id: str) -> str:
    return f"{dataset_id}::{target_id}"


def _split_target_key(value: str) -> tuple[str, str]:
    pieces = str(value).split("::")
    if len(pieces) != 2 or not all(pieces):
        raise ValueError(f"invalid target key: {value!r}")
    return pieces[0], pieces[1]


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("reference envelope protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_reference_envelope_reliability_before_outcomes":
        raise ValueError("reference envelope protocol is not frozen")
    if tuple(protocol["heads"]["ids"]) != HEADS:
        raise ValueError("reference envelope head set changed")
    if tuple(protocol["backbones"]["scout"]) != BACKBONES:
        raise ValueError("reference envelope scout roster changed")
    if protocol["backbones"].get("reserved_evaluated") != []:
        raise ValueError("reference envelope reserved roster is nonempty")
    cohort = protocol["cohort"]
    if (
        tuple(cohort["replicate_seeds"]) != REPLICATE_SEEDS
        or int(cohort["training_rows_per_replicate"]) != 100
        or int(cohort["evaluation_rows"]) != 200
        or int(cohort["training_rows_total"]) != 300
        or int(cohort["target_rows_total"]) != 500
    ):
        raise ValueError("reference envelope cohort changed")
    strata = protocol["strata"]
    if set(strata) != {"recoverable_lp_dominated_control", "rank_changing_nonlinear_opportunity", "unusable_for_ranking"}:
        raise ValueError("reference envelope strata changed")
    target_keys = [target for values in strata.values() for target in values]
    if len(target_keys) != 20 or len(set(target_keys)) != 20 or set(strata["unusable_for_ranking"]) & set(target_keys[:8]):
        raise ValueError("reference envelope target strata are not disjoint")
    budgets = protocol["budget_curve"]["budgets"]
    expected_budget_ids = (
        "n100_s4103", "n100_s4201", "n100_s4303",
        "n200_u4103_4201", "n200_u4103_4303", "n200_u4201_4303",
        "n300_u4103_4201_4303",
    )
    if tuple(row["id"] for row in budgets) != expected_budget_ids:
        raise ValueError("reference envelope budget order changed")
    if tuple(protocol["budget_curve"]["targets"]) != tuple(
        strata["recoverable_lp_dominated_control"] + strata["rank_changing_nonlinear_opportunity"]
    ):
        raise ValueError("reference envelope budget target order changed")
    grid = protocol["expected_grid"]
    expected_grid = {
        "archive_target_count": 20,
        "archive_panel_count": 60,
        "archive_candidate_count": 300,
        "archive_recoverable_panel_count": 24,
        "budget_target_count": 8,
        "budget_archived_100_panel_count": 24,
        "budget_new_200_panel_count": 24,
        "budget_new_300_panel_count": 8,
        "budget_new_panel_count": 32,
        "budget_new_candidate_count": 160,
        "budget_new_pca_fit_count": 160,
        "budget_new_head_family_call_count": 640,
        "budget_new_sklearn_estimator_fit_count": 18880,
        "reserved_backbone_count": 0,
        "overlapindex_fit_count": 0,
    }
    if {key: int(grid[key]) for key in expected_grid} != expected_grid:
        raise ValueError("reference envelope expected grid changed")
    if int(protocol["evaluation_bootstrap"]["resamples"]) != BOOTSTRAP_RESAMPLES:
        raise ValueError("reference envelope bootstrap count changed")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "protocol": source_prepare.PROTOCOL_PATH,
        "protocol_sidecar": source_prepare.PROTOCOL_SHA256_PATH,
        "prepare": Path(source_prepare.__file__).resolve(),
        "extract": Path(source_extract.__file__).resolve(),
        "runner": Path(source_run.__file__).resolve(),
        "head_evaluator": Path(residual.__file__).resolve(),
        "geometry": Path(geometry.__file__).resolve(),
        "cohort": source_prepare.DEFAULT_OUTPUT_ROOT / "cohort.json",
        "embedding_manifest": source_extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "raw_results": source_run.DEFAULT_OUTPUT_ROOT / "raw_results.json",
        "summary": source_run.DEFAULT_OUTPUT_ROOT / "summary.json",
        "manifest": source_run.DEFAULT_OUTPUT_ROOT / "manifest.json",
    }


def source_hashes() -> dict[str, str]:
    values = {f"source_{key}": sha256_path(path) for key, path in _source_paths().items()}
    values["experiment_protocol"] = sha256_path(PROTOCOL_PATH)
    values["experiment_runner"] = sha256_path(Path(__file__).resolve())
    return values


def _validate_source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    expected = protocol["source"]
    paths = _source_paths()
    aliases = {
        "protocol": "protocol_sha256",
        "protocol_sidecar": "protocol_sidecar_sha256",
        "prepare": "prepare_sha256",
        "extract": "extract_sha256",
        "runner": "runner_sha256",
        "head_evaluator": "head_evaluator_sha256",
        "geometry": "geometry_sha256",
        "cohort": "cohort_sha256",
        "embedding_manifest": "embedding_manifest_sha256",
        "raw_results": "raw_results_sha256",
        "summary": "summary_sha256",
        "manifest": "manifest_sha256",
    }
    values = {}
    for key, path in paths.items():
        if not path.is_file():
            raise ValueError(f"reference envelope source is missing: {path}")
        actual = sha256_path(path)
        values[f"source_{key}"] = actual
        if actual != expected[aliases[key]]:
            raise ValueError(f"reference envelope source hash changed: {path}")
    return values


def validate_sources(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate and return the frozen raw results, summary, manifest, and cohort."""
    hashes = _validate_source_hashes(protocol)
    source_protocol = source_run.load_protocol()
    raw = json.loads(_source_paths()["raw_results"].read_text(encoding="utf-8"))
    summary = json.loads(_source_paths()["summary"].read_text(encoding="utf-8"))
    manifest = json.loads(_source_paths()["manifest"].read_text(encoding="utf-8"))
    cohort = json.loads(_source_paths()["cohort"].read_text(encoding="utf-8"))
    embedding_manifest = json.loads(_source_paths()["embedding_manifest"].read_text(encoding="utf-8"))
    descriptor = protocol["source"]
    if (
        raw.get("status") != descriptor["expected_status"]
        or summary.get("status") != descriptor["expected_status"]
        or manifest.get("status") != descriptor["expected_status"]
        or len(raw.get("panels", ())) != int(descriptor["expected_panel_count"])
        or sum(len(panel.get("candidates", ())) for panel in raw.get("panels", ()))
        != int(descriptor["expected_candidate_count"])
        or canonical_json(raw.get("protocol")) != canonical_json(source_protocol)
        or manifest.get("artifacts", {}).get("raw_results.json") != descriptor["raw_results_sha256"]
        or manifest.get("artifacts", {}).get("summary.json") != descriptor["summary_sha256"]
        or int(manifest.get("counts", {}).get("reserved_backbones", -1)) != 0
        or int(manifest.get("counts", {}).get("overlapindex_fits", -1)) != 0
        or embedding_manifest.get("status") != "completed_large_structured_opportunity_embedding_extraction"
        or embedding_manifest.get("predictive_outcomes_inspected") is not False
    ):
        raise ValueError("reference envelope source artifacts are malformed")
    if tuple(source_protocol["backbones"]["scout"]) != BACKBONES:
        raise ValueError("reference envelope source scout roster changed")
    source_keys = set(source_protocol["datasets"])
    target_keys = {
        _target_key(dataset_id, target_id)
        for dataset_id, dataset in source_protocol["datasets"].items()
        for target_id in dataset["targets"]
    }
    protocol_target_keys = {
        target for values in protocol["strata"].values() for target in values
    }
    if target_keys != protocol_target_keys:
        raise ValueError("reference envelope source target grid differs from frozen strata")
    observed_panels = set()
    for panel in raw["panels"]:
        dataset_id, target_id, seed = str(panel["dataset_id"]), str(panel["target_id"]), int(panel["seed"])
        key = _target_key(dataset_id, target_id)
        if dataset_id not in source_keys or key not in target_keys or seed not in REPLICATE_SEEDS:
            raise ValueError("reference envelope source panel identity changed")
        identity = (key, seed)
        if identity in observed_panels:
            raise ValueError("reference envelope source panel is duplicated")
        observed_panels.add(identity)
        if tuple(row["candidate_id"] for row in panel["candidates"]) != BACKBONES:
            raise ValueError("reference envelope source candidate order changed")
        if panel.get("split") != {"training_rows": 100, "evaluation_rows": 200}:
            raise ValueError("reference envelope source panel split changed")
        for candidate in panel["candidates"]:
            if set(candidate["heads"]) != set(HEADS):
                raise ValueError("reference envelope source head set changed")
            if any("test_predictions" in candidate["heads"][head] for head in HEADS):
                raise ValueError("archive unexpectedly contains evaluation predictions")
    expected_panel_ids = {
        (target, seed) for target in target_keys for seed in REPLICATE_SEEDS
    }
    if observed_panels != expected_panel_ids:
        raise ValueError("reference envelope source panel grid is incomplete")
    if tuple(cohort.get("replicate_seeds", ())) != REPLICATE_SEEDS:
        raise ValueError("reference envelope cohort seeds changed")
    cohort_target_keys = set()
    for dataset_id, dataset in cohort.get("datasets", {}).items():
        for target_id, rows in dataset.get("target_cohorts", {}).items():
            key = _target_key(dataset_id, target_id)
            cohort_target_keys.add(key)
            if len(rows) != 500:
                raise ValueError("reference envelope target cohort does not contain 500 rows")
            training = [row for row in rows if row.get("split") == "training"]
            evaluation = [row for row in rows if row.get("split") == "evaluation"]
            if len(training) != 300 or len(evaluation) != 200:
                raise ValueError("reference envelope cohort split changed")
            if len({row["sample_id"] for row in rows}) != 500:
                raise ValueError("reference envelope cohort sample IDs are duplicated")
            if {row["sample_id"] for row in training} & {row["sample_id"] for row in evaluation}:
                raise ValueError("reference envelope training/evaluation rows overlap")
            if {int(row["replicate_seed"]) for row in training} != set(REPLICATE_SEEDS):
                raise ValueError("reference envelope cohort replicate membership changed")
            if any(sum(int(row["replicate_seed"]) == seed for row in training) != 100 for seed in REPLICATE_SEEDS):
                raise ValueError("reference envelope cohort replicate size changed")
    if cohort_target_keys != target_keys:
        raise ValueError("reference envelope cohort target grid changed")
    return raw, summary, manifest, cohort


def _stratum_by_target(protocol: Mapping[str, Any]) -> dict[str, str]:
    return {
        target: stratum
        for stratum, targets in protocol["strata"].items()
        for target in targets
    }


def _winner_ids(values: Mapping[str, float]) -> list[str]:
    maximum = max(float(value) for value in values.values())
    return [head for head in HEADS if float(values[head]) >= maximum - TIE_ATOL]


def _primary_winner(values: Mapping[str, float]) -> str:
    return _winner_ids(values)[0]


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
        or truth_values.shape != (len(BACKBONES),)
        or not np.isfinite(truth_values).all()
        or not np.isfinite(score_values).all()
    ):
        raise ValueError("reference envelope ranking inputs are invalid")
    selected = np.flatnonzero(score_values >= np.max(score_values) - TIE_ATOL)
    truth_best = np.flatnonzero(truth_values >= np.max(truth_values) - TIE_ATOL)
    selected_top = _top_indices(score_values, 3)
    truth_top = _top_indices(truth_values, 3)
    correlation = None
    if np.ptp(truth_values) > TIE_ATOL and np.ptp(score_values) > TIE_ATOL:
        correlation = spearmanr(truth_values, score_values).correlation
    pair_credit = []
    for left in range(len(truth_values)):
        for right in range(left + 1, len(truth_values)):
            truth_delta = float(truth_values[left] - truth_values[right])
            if abs(truth_delta) < MATERIAL_ATOL - TIE_ATOL:
                continue
            score_delta = float(score_values[left] - score_values[right])
            pair_credit.append(0.5 if abs(score_delta) <= TIE_ATOL else float(np.sign(score_delta) == np.sign(truth_delta)))
    return {
        "spearman": None if correlation is None or not np.isfinite(correlation) else float(correlation),
        "top1_regret_r2": float(np.max(truth_values) - np.max(truth_values[selected])),
        "exact_best": bool(set(selected).intersection(truth_best)),
        "top3_opportunity_regret_r2": float(np.max(truth_values) - np.max(truth_values[selected_top])),
        "top3_mean_utility_regret_r2": float(np.mean(truth_values[truth_top]) - np.mean(truth_values[selected_top])),
        "top3_set_recall": float(len(set(selected_top).intersection(truth_top)) / len(set(truth_top))),
        "material_pair_accuracy": float(np.mean(pair_credit)) if pair_credit else None,
        "material_pair_count": len(pair_credit),
        "selected_candidate_ids": [BACKBONES[index] for index in selected],
        "selected_top3_candidate_ids": [BACKBONES[index] for index in selected_top],
    }


def _candidate_archive_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if str(candidate.get("candidate_id")) not in BACKBONES:
        raise ValueError("reference envelope candidate identity changed")
    heads = candidate.get("heads")
    if not isinstance(heads, Mapping) or set(heads) != set(HEADS):
        raise ValueError("reference envelope candidate head set changed")
    cv_scores = {head: float(heads[head]["inner_best_r2"]) for head in HEADS}
    heldout_scores = {head: float(heads[head]["test_r2"]) for head in HEADS}
    if not np.isfinite(list(cv_scores.values())).all() or not np.isfinite(list(heldout_scores.values())).all():
        raise ValueError("reference envelope candidate scores are nonfinite")
    cv_envelope = max(cv_scores.values())
    heldout_envelope = max(heldout_scores.values())
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "cv_head_scores": cv_scores,
        "heldout_head_scores": heldout_scores,
        "cv_envelope_r2": float(cv_envelope),
        "heldout_envelope_r2": float(heldout_envelope),
        "cv_winner_heads": _winner_ids(cv_scores),
        "heldout_winner_heads": _winner_ids(heldout_scores),
        "cv_primary_winner": _primary_winner(cv_scores),
        "heldout_primary_winner": _primary_winner(heldout_scores),
        "envelope_bias_cv_minus_heldout_r2": float(cv_envelope - heldout_envelope),
        "envelope_absolute_error_r2": float(abs(cv_envelope - heldout_envelope)),
        "evaluation_predictions_available": False,
    }


def _metric_rows_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("reference envelope metric group is empty")
    spearman_values = [float(row["spearman"]) for row in rows if row["spearman"] is not None]
    pair_values = [float(row["material_pair_accuracy"]) for row in rows if row["material_pair_accuracy"] is not None]
    return {
        "panel_count": len(rows),
        "defined_spearman_count": len(spearman_values),
        "mean_spearman": float(np.mean(spearman_values)) if spearman_values else None,
        "mean_top1_regret_r2": float(np.mean([row["top1_regret_r2"] for row in rows])),
        "exact_best_rate": float(np.mean([row["exact_best"] for row in rows])),
        "mean_top3_opportunity_regret_r2": float(np.mean([row["top3_opportunity_regret_r2"] for row in rows])),
        "mean_top3_utility_regret_r2": float(np.mean([row["top3_mean_utility_regret_r2"] for row in rows])),
        "mean_top3_set_recall": float(np.mean([row["top3_set_recall"] for row in rows])),
        "mean_material_pair_accuracy": float(np.mean(pair_values)) if pair_values else None,
        "material_pair_count": int(sum(int(row["material_pair_count"]) for row in rows)),
    }


def _stability_metrics(vectors: Sequence[Sequence[float]], *, candidate_ids: Sequence[str] = BACKBONES) -> dict[str, Any]:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(candidate_ids) or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("reference envelope stability vectors are invalid")
    candidate_ids = tuple(str(value) for value in candidate_ids)
    anchor = np.mean(values, axis=0)
    anchor_pairs = [
        (left, right)
        for left in range(values.shape[1])
        for right in range(left + 1, values.shape[1])
        if abs(float(anchor[left] - anchor[right])) >= MATERIAL_ATOL - TIE_ATOL
    ]
    rank_values, winner_agreement, top3_jaccards, pair_agreement = [], [], [], []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            correlation = None
            if np.ptp(values[left]) > TIE_ATOL and np.ptp(values[right]) > TIE_ATOL:
                correlation = spearmanr(values[left], values[right]).correlation
            if correlation is not None and np.isfinite(correlation):
                rank_values.append(float(correlation))
            left_winners = set(np.flatnonzero(values[left] >= np.max(values[left]) - TIE_ATOL))
            right_winners = set(np.flatnonzero(values[right] >= np.max(values[right]) - TIE_ATOL))
            winner_agreement.append(bool(left_winners.intersection(right_winners)))
            left_top = set(_top_indices(values[left], 3))
            right_top = set(_top_indices(values[right], 3))
            top3_jaccards.append(float(len(left_top & right_top) / len(left_top | right_top)))
            credits = []
            for l_index, r_index in anchor_pairs:
                left_delta = float(values[left, l_index] - values[left, r_index])
                right_delta = float(values[right, l_index] - values[right, r_index])
                credits.append(0.5 if abs(left_delta) <= TIE_ATOL or abs(right_delta) <= TIE_ATOL else float(np.sign(left_delta) == np.sign(right_delta)))
            if credits:
                pair_agreement.append(float(np.mean(credits)))
    winner_counts = {
        candidate: int(
            sum(
                candidate_ids.index(candidate)
                in set(np.flatnonzero(row >= np.max(row) - TIE_ATOL).tolist())
                for row in values
            )
        )
        for candidate in candidate_ids
    }
    top3_counts = {
        candidate: int(
            sum(candidate in [candidate_ids[index] for index in _top_indices(row, 3)] for row in values)
        )
        for candidate in candidate_ids
    }
    return {
        "vector_count": int(len(values)),
        "pair_count": int(len(values) * (len(values) - 1) // 2),
        "mean_pair_spearman": float(np.mean(rank_values)) if rank_values else None,
        "winner_intersection_rate": float(np.mean(winner_agreement)) if winner_agreement else None,
        "top3_jaccard_mean": float(np.mean(top3_jaccards)) if top3_jaccards else None,
        "material_pair_anchor_count": int(len(anchor_pairs)),
        "material_pair_agreement_mean": float(np.mean(pair_agreement)) if pair_agreement else None,
        "winner_counts": winner_counts,
        "top3_counts": top3_counts,
        "candidate_ids": list(candidate_ids),
    }


def _archive_confusion(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for group in ARCHIVE_GROUPS:
        selected = [
            candidate
            for panel in rows
            if group == "all" or panel["stratum"] == group
            for candidate in panel["candidates"]
        ]
        matrix = {cv: {heldout: 0 for heldout in HEADS} for cv in HEADS}
        for row in selected:
            matrix[row["cv_primary_winner"]][row["heldout_primary_winner"]] += 1
        output[group] = {
            "primary_tie_break": "first head in frozen HEADS order",
            "matrix": matrix,
            "total": len(selected),
            "primary_agreement_count": int(sum(row["cv_primary_winner"] == row["heldout_primary_winner"] for row in selected)),
            "winner_set_intersection_count": int(sum(bool(set(row["cv_winner_heads"]) & set(row["heldout_winner_heads"])) for row in selected)),
        }
    return output


def archive_audit(raw: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    strata = _stratum_by_target(protocol)
    panels = []
    observed = set()
    rows = []
    target_vectors: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"cv": [], "heldout": []})
    for source_panel in raw["panels"]:
        dataset_id, target_id, seed = str(source_panel["dataset_id"]), str(source_panel["target_id"]), int(source_panel["seed"])
        target = _target_key(dataset_id, target_id)
        identity = (target, seed)
        if target not in strata or identity in observed or tuple(row["candidate_id"] for row in source_panel["candidates"]) != BACKBONES:
            raise ValueError("reference envelope archive panel grid changed")
        observed.add(identity)
        candidates = [_candidate_archive_summary(candidate) for candidate in source_panel["candidates"]]
        cv = [candidate["cv_envelope_r2"] for candidate in candidates]
        heldout = [candidate["heldout_envelope_r2"] for candidate in candidates]
        metrics = ranking_metrics(heldout, cv)
        metric_row = {
            "dataset_id": dataset_id,
            "target_id": target_id,
            "target": target,
            "seed": seed,
            "stratum": strata[target],
            **metrics,
        }
        rows.append(metric_row)
        target_vectors[target]["cv"].append((seed, cv))
        target_vectors[target]["heldout"].append((seed, heldout))
        panels.append({
            "dataset_id": dataset_id,
            "target_id": target_id,
            "target": target,
            "seed": seed,
            "stratum": strata[target],
            "candidate_count": len(candidates),
            "candidates": candidates,
            "cv_vs_heldout_metrics": metrics,
            "evaluation_predictions_available": False,
        })
    expected = {(target, seed) for target in strata for seed in REPLICATE_SEEDS}
    if observed != expected or len(panels) != 60:
        raise ValueError("reference envelope archive panel count changed")
    target_stability = []
    for target in sorted(target_vectors):
        values = target_vectors[target]
        cv_vectors = [row[1] for row in sorted(values["cv"], key=lambda row: row[0])]
        heldout_vectors = [row[1] for row in sorted(values["heldout"], key=lambda row: row[0])]
        dataset_id, target_id = _split_target_key(target)
        target_stability.append({
            "target": target,
            "dataset_id": dataset_id,
            "target_id": target_id,
            "stratum": strata[target],
            "seed_order": list(REPLICATE_SEEDS),
            "cv_envelope_stability": _stability_metrics(cv_vectors),
            "heldout_envelope_stability": _stability_metrics(heldout_vectors),
        })
    group_summaries = []
    for group in ARCHIVE_GROUPS:
        selected = [row for row in rows if group == "all" or row["stratum"] == group]
        group_summaries.append({"group": group, **_metric_rows_summary(selected)})
    bias = {}
    for group in ARCHIVE_GROUPS:
        selected = [panel for panel in panels if group == "all" or panel["stratum"] == group]
        candidate_rows = [candidate for panel in selected for candidate in panel["candidates"]]
        bias[group] = {
            "candidate_count": len(candidate_rows),
            "envelope_mean_cv_minus_heldout_r2": float(np.mean([row["envelope_bias_cv_minus_heldout_r2"] for row in candidate_rows])),
            "envelope_mean_absolute_error_r2": float(np.mean([row["envelope_absolute_error_r2"] for row in candidate_rows])),
            "per_head": {
                head: {
                    "mean_inner_minus_heldout_r2": float(np.mean([row["cv_head_scores"][head] - row["heldout_head_scores"][head] for row in candidate_rows])),
                    "mean_absolute_error_r2": float(np.mean([abs(row["cv_head_scores"][head] - row["heldout_head_scores"][head]) for row in candidate_rows])),
                }
                for head in HEADS
            },
        }
    target_weighted = []
    for group in ARCHIVE_GROUPS:
        selected = [row for row in rows if group == "all" or row["stratum"] == group]
        by_target = defaultdict(list)
        for row in selected:
            by_target[str(row["target"])].append(row)
        target_rows = []
        for target, target_selected in by_target.items():
            target_rows.append({
                key: float(np.mean([float(row[key]) for row in target_selected]))
                for key in (
                    "top1_regret_r2",
                    "top3_opportunity_regret_r2",
                    "top3_mean_utility_regret_r2",
                    "top3_set_recall",
                )
            } | {
                "spearman": (
                    float(np.mean([float(row["spearman"]) for row in target_selected if row["spearman"] is not None]))
                    if any(row["spearman"] is not None for row in target_selected)
                    else None
                ),
                "exact_best": float(np.mean([float(row["exact_best"]) for row in target_selected])),
                "material_pair_accuracy": (
                    float(np.mean([float(row["material_pair_accuracy"]) for row in target_selected if row["material_pair_accuracy"] is not None]))
                    if any(row["material_pair_accuracy"] is not None for row in target_selected)
                    else None
                ),
                "material_pair_count": int(sum(int(row["material_pair_count"]) for row in target_selected)),
            })
        finite_spearman = [row for row in target_rows if row["spearman"] is not None]
        finite_pair = [row for row in target_rows if row["material_pair_accuracy"] is not None]
        target_weighted.append({
            "group": group,
            "target_count": len(target_rows),
            "weighting": "equal target weight after equal replicate averaging",
            "defined_spearman_count": len(finite_spearman),
            "mean_spearman": float(np.mean([row["spearman"] for row in finite_spearman])) if finite_spearman else None,
            "mean_top1_regret_r2": float(np.mean([row["top1_regret_r2"] for row in target_rows])),
            "exact_best_rate": float(np.mean([row["exact_best"] for row in target_rows])),
            "mean_top3_opportunity_regret_r2": float(np.mean([row["top3_opportunity_regret_r2"] for row in target_rows])),
            "mean_top3_utility_regret_r2": float(np.mean([row["top3_mean_utility_regret_r2"] for row in target_rows])),
            "mean_top3_set_recall": float(np.mean([row["top3_set_recall"] for row in target_rows])),
            "mean_material_pair_accuracy": float(np.mean([row["material_pair_accuracy"] for row in finite_pair])) if finite_pair else None,
            "material_pair_count": int(sum(row["material_pair_count"] for row in target_rows)),
        })
    return {
        "status": "completed_archive_only_audit",
        "panel_count": len(panels),
        "candidate_count": sum(len(panel["candidates"]) for panel in panels),
        "evaluation_predictions_available": False,
        "panels": panels,
        "metric_rows": rows,
        "group_summaries": group_summaries,
        "target_weighted_group_summaries": target_weighted,
        "head_confusion": _archive_confusion(panels),
        "bias": bias,
        "target_level_seed_stability": target_stability,
        "shared_evaluation_dependence": "Each target uses one shared 200-row evaluation cohort across its three archived training replicates; target blocks, rather than panels, are the uncertainty unit.",
    }


def _budget_specs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in protocol["budget_curve"]["budgets"]]


def _budget_spec(protocol: Mapping[str, Any], budget_id: str) -> dict[str, Any]:
    selected = [row for row in _budget_specs(protocol) if row["id"] == budget_id]
    if len(selected) != 1:
        raise ValueError(f"reference envelope budget is unknown: {budget_id!r}")
    return selected[0]


def _target_rows(
    cohort: Mapping[str, Any], target: str
) -> tuple[str, str, list[Mapping[str, Any]]]:
    dataset_id, target_id = _split_target_key(target)
    try:
        rows = cohort["datasets"][dataset_id]["target_cohorts"][target_id]
    except KeyError as exc:
        raise ValueError(f"reference envelope target is absent from cohort: {target!r}") from exc
    return dataset_id, target_id, list(rows)


def _budget_rows(
    cohort: Mapping[str, Any], target: str, training_replicate_seeds: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    dataset_id, target_id, rows = _target_rows(cohort, target)
    expected_order = tuple(int(seed) for seed in REPLICATE_SEEDS)
    requested = tuple(int(seed) for seed in training_replicate_seeds)
    if not requested or any(seed not in expected_order for seed in requested) or len(set(requested)) != len(requested):
        raise ValueError("reference envelope budget replicate set is invalid")
    by_seed = {
        seed: [row for row in rows if row.get("split") == "training" and int(row.get("replicate_seed")) == seed]
        for seed in requested
    }
    if any(len(by_seed[seed]) != 100 for seed in requested):
        raise ValueError("reference envelope budget training replicate is incomplete")
    training = [row for seed in expected_order if seed in requested for row in by_seed[seed]]
    evaluation = [row for row in rows if row.get("split") == "evaluation"]
    if len(training) != 100 * len(requested) or len(evaluation) != 200:
        raise ValueError("reference envelope budget cohort size changed")
    training_ids = [str(row["sample_id"]) for row in training]
    evaluation_ids = [str(row["sample_id"]) for row in evaluation]
    if len(set(training_ids)) != len(training_ids) or len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("reference envelope budget sample IDs are duplicated")
    if set(training_ids) & set(evaluation_ids):
        raise ValueError("reference envelope budget training/evaluation rows overlap")
    train_indices = np.asarray([int(row["matrix_index"]) for row in training], dtype=np.int64)
    eval_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    y_train = np.asarray([float(row["target"]) for row in training], dtype=np.float64)
    y_eval = np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64)
    if not np.isfinite(y_train).all() or not np.isfinite(y_eval).all():
        raise ValueError("reference envelope budget target contains nonfinite values")
    if len(set(train_indices.tolist())) != len(train_indices) or len(set(eval_indices.tolist())) != len(eval_indices):
        raise ValueError("reference envelope budget matrix indices are duplicated")
    return train_indices, eval_indices, y_train, y_eval, training_ids, evaluation_ids


def _budget_seed(
    dataset_position: int, target_position: int, budget_position: int, backbone_position: int
) -> int:
    value = BUDGET_BASE_SEED + int(dataset_position) * 100_000 + int(target_position) * 1_000 + int(budget_position) * 100 + int(backbone_position)
    if not 0 <= value < 2**32:
        raise ValueError("reference envelope budget fit seed is outside uint32")
    return value


def _budget_geometry_seed(
    dataset_position: int, target_position: int, budget_position: int, backbone_position: int
) -> int:
    value = BUDGET_BASE_SEED + 10_000_000 + int(dataset_position) * 100_000 + int(target_position) * 1_000 + int(budget_position) * 100 + int(backbone_position)
    if not 0 <= value < 2**32:
        raise ValueError("reference envelope budget PCA seed is outside uint32")
    return value


def _evaluate_heads_with_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Fit the unchanged grids while delaying every held-out score until fits finish."""
    train_values = np.asarray(X_train, dtype=np.float32)
    targets = np.asarray(y_train, dtype=np.float64)
    eval_values = np.asarray(X_eval, dtype=np.float32)
    eval_targets = np.asarray(y_eval, dtype=np.float64)
    if (
        train_values.ndim != 2
        or eval_values.ndim != 2
        or train_values.shape[1] != eval_values.shape[1]
        or targets.shape != (len(train_values),)
        or eval_targets.shape != (len(eval_values),)
        or len(train_values) < 8
        or not np.isfinite(train_values).all()
        or not np.isfinite(eval_values).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(eval_targets).all()
    ):
        raise ValueError("reference envelope head inputs are invalid")
    inner = KFold(n_splits=3, shuffle=True, random_state=int(seed))
    fitted: dict[str, tuple[float, dict[str, Any], float, np.ndarray]] = {}
    for head_id, (estimator, grid) in residual._head_estimators().items():
        if head_id not in HEADS:
            raise ValueError("reference envelope source head set changed")
        search = GridSearchCV(
            clone(estimator),
            param_grid=grid,
            cv=inner,
            scoring="r2",
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        started = time.perf_counter()
        search.fit(train_values, targets)
        prediction = np.asarray(search.predict(eval_values), dtype=np.float64)
        fitted[head_id] = (
            float(search.best_score_),
            dict(search.best_params_),
            float(time.perf_counter() - started),
            prediction,
        )
    # y_eval is intentionally consumed only after all training/CV searches complete.
    output = {}
    for head_id in HEADS:
        inner_score, params, wall, prediction = fitted[head_id]
        score = float(r2_score(eval_targets, prediction))
        if not np.isfinite(score) or not np.isfinite(prediction).all():
            raise ValueError("reference envelope held-out prediction is nonfinite")
        output[head_id] = {
            "test_r2": score,
            "inner_best_r2": inner_score,
            "best_params": params,
            "wall_seconds": wall,
            "test_predictions": prediction.tolist(),
        }
    return output


def _source_100_budget_panel(
    source_panel: Mapping[str, Any], target: str, seed: int, stratum: str, budget_id: str
) -> dict[str, Any]:
    if (
        str(source_panel["dataset_id"]) != _split_target_key(target)[0]
        or str(source_panel["target_id"]) != _split_target_key(target)[1]
        or int(source_panel["seed"]) != int(seed)
    ):
        raise ValueError("reference envelope archived budget panel identity changed")
    candidates = []
    for source_candidate in source_panel["candidates"]:
        candidate = _candidate_archive_summary(source_candidate)
        heads = {}
        for head in HEADS:
            source_head = source_candidate["heads"][head]
            heads[head] = {
                "test_r2": float(source_head["test_r2"]),
                "inner_best_r2": float(source_head["inner_best_r2"]),
                "best_params": dict(source_head["best_params"]),
                "wall_seconds": float(source_head["wall_seconds"]),
            }
        candidates.append({
            "candidate_id": candidate["candidate_id"],
            "pca": dict(source_candidate["pca"]),
            "heads": heads,
            "cv_envelope_r2": candidate["cv_envelope_r2"],
            "heldout_envelope_r2": candidate["heldout_envelope_r2"],
            "best_head_ids": candidate["heldout_winner_heads"],
            "evaluation_predictions_available": False,
        })
    return {
        "dataset_id": str(source_panel["dataset_id"]),
        "target_id": str(source_panel["target_id"]),
        "target": target,
        "stratum": stratum,
        "budget_id": budget_id,
        "training_replicate_seeds": [int(seed)],
        "training_rows": 100,
        "evaluation_rows": 200,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "evaluation_predictions_available": False,
        "source_archived": True,
        "shared_evaluation_dependence": True,
    }


def _fit_budget_panel(
    protocol: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
    cohort: Mapping[str, Any],
    target: str,
    budget: Mapping[str, Any],
    budget_position: int,
) -> dict[str, Any]:
    dataset_id, target_id = _split_target_key(target)
    dataset_position = tuple(protocol["budget_curve"]["targets"]).index(target) // 4
    source_dataset_order = tuple(source_run.load_protocol()["datasets"])
    dataset_position = source_dataset_order.index(dataset_id)
    target_position = tuple(source_run.load_protocol()["datasets"][dataset_id]["targets"]).index(target_id)
    train_idx, eval_idx, y_train, y_eval, training_ids, evaluation_ids = _budget_rows(
        cohort, target, budget["training_replicate_seeds"]
    )
    data = datasets[dataset_id]
    candidates = []
    started = time.perf_counter()
    for backbone_position, backbone in enumerate(BACKBONES):
        matrix = data["matrices"][backbone]
        if train_idx.max() >= len(matrix) or eval_idx.max() >= len(matrix):
            raise ValueError("reference envelope budget matrix index is outside embedding bank")
        pca_started = time.perf_counter()
        X_train, X_eval, pca = geometry.pca_geometry(
            matrix,
            train_idx,
            eval_idx,
            _budget_geometry_seed(dataset_position, target_position, budget_position, backbone_position),
        )
        pca_wall = float(time.perf_counter() - pca_started)
        fit_seed = _budget_seed(dataset_position, target_position, budget_position, backbone_position)
        heads = _evaluate_heads_with_predictions(X_train, y_train, X_eval, y_eval, fit_seed)
        cv_envelope = max(float(heads[head]["inner_best_r2"]) for head in HEADS)
        heldout_envelope = max(float(heads[head]["test_r2"]) for head in HEADS)
        candidates.append({
            "candidate_id": backbone,
            "pca": pca,
            "pca_seed": _budget_geometry_seed(dataset_position, target_position, budget_position, backbone_position),
            "fit_seed": fit_seed,
            "pca_wall_seconds": pca_wall,
            "heads": heads,
            "cv_envelope_r2": cv_envelope,
            "heldout_envelope_r2": heldout_envelope,
            "best_head_ids": _winner_ids({head: float(heads[head]["test_r2"]) for head in HEADS}),
            "evaluation_predictions_available": True,
        })
    if len(candidates) != len(BACKBONES):
        raise ValueError("reference envelope budget candidate count changed")
    return {
        "dataset_id": dataset_id,
        "target_id": target_id,
        "target": target,
        "stratum": _stratum_by_target(protocol)[target],
        "budget_id": str(budget["id"]),
        "budget_position": int(budget_position),
        "training_replicate_seeds": [int(seed) for seed in budget["training_replicate_seeds"]],
        "training_rows": int(budget["rows"]),
        "evaluation_rows": 200,
        "training_sample_ids": training_ids,
        "evaluation_sample_ids": evaluation_ids,
        "evaluation_targets": y_eval.tolist(),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "evaluation_predictions_available": True,
        "source_archived": False,
        "shared_evaluation_dependence": True,
        "panel_wall_seconds": float(time.perf_counter() - started),
    }


def _budget_metric_row(panel: Mapping[str, Any]) -> dict[str, Any]:
    truth = [float(candidate["heldout_envelope_r2"]) for candidate in panel["candidates"]]
    cv = [float(candidate["cv_envelope_r2"]) for candidate in panel["candidates"]]
    return {
        "target": str(panel["target"]),
        "dataset_id": str(panel["dataset_id"]),
        "target_id": str(panel["target_id"]),
        "stratum": str(panel["stratum"]),
        "budget_id": str(panel["budget_id"]),
        **ranking_metrics(truth, cv),
    }


def _grouped_metric_summaries(rows: Sequence[Mapping[str, Any]], groups: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for budget_id in sorted({str(row["budget_id"]) for row in rows}):
        selected_budget = [row for row in rows if str(row["budget_id"]) == budget_id]
        for group in groups:
            selected = [row for row in selected_budget if group == "all" or row["stratum"] == group]
            if not selected:
                continue
            output.append({
                "budget_id": budget_id,
                "group": group,
                "weighting": "equal target weight; one panel per target within each budget",
                "target_count": len({str(row["target"]) for row in selected}),
                **_metric_rows_summary(selected),
            })
    return output


def _vector_from_panel(panel: Mapping[str, Any], score: str = "heldout_envelope_r2") -> list[float]:
    return [float(candidate[score]) for candidate in panel["candidates"]]


def _target_budget_stability(
    panels: Sequence[Mapping[str, Any]],
    *,
    budget_ids: Sequence[str],
    label: str,
) -> list[dict[str, Any]]:
    selected = [panel for panel in panels if str(panel["budget_id"]) in set(budget_ids)]
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for panel in selected:
        by_target[str(panel["target"])].append(panel)
    output = []
    for target in sorted(by_target):
        target_panels = sorted(by_target[target], key=lambda panel: str(panel["budget_id"]))
        if len(target_panels) < 2:
            continue
        output.append({
            "target": target,
            "dataset_id": str(target_panels[0]["dataset_id"]),
            "target_id": str(target_panels[0]["target_id"]),
            "stratum": str(target_panels[0]["stratum"]),
            "comparison": label,
            "budget_order": [str(panel["budget_id"]) for panel in target_panels],
            "heldout_envelope_stability": _stability_metrics([_vector_from_panel(panel) for panel in target_panels]),
            "cv_envelope_stability": _stability_metrics([_vector_from_panel(panel, "cv_envelope_r2") for panel in target_panels]),
        })
    return output


def _budget_reference_comparisons(
    panels: Sequence[Mapping[str, Any]], reference_budget_id: str
) -> list[dict[str, Any]]:
    reference = {
        str(panel["target"]): panel
        for panel in panels
        if str(panel["budget_id"]) == reference_budget_id
    }
    rows = []
    for panel in panels:
        budget_id = str(panel["budget_id"])
        if budget_id == reference_budget_id or str(panel["target"]) not in reference:
            continue
        ref = reference[str(panel["target"])]
        rows.append({
            "target": str(panel["target"]),
            "dataset_id": str(panel["dataset_id"]),
            "target_id": str(panel["target_id"]),
            "stratum": str(panel["stratum"]),
            "from_budget_id": budget_id,
            "reference_budget_id": reference_budget_id,
            "heldout_envelope_metrics": ranking_metrics(_vector_from_panel(ref), _vector_from_panel(panel)),
            "cv_envelope_metrics": ranking_metrics(_vector_from_panel(ref), _vector_from_panel(panel, "cv_envelope_r2")),
        })
    return rows


def _comparison_summary(rows: Sequence[Mapping[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    output = []
    for budget_id in sorted({str(row["from_budget_id"]) for row in rows}):
        selected_budget = [row for row in rows if str(row["from_budget_id"]) == budget_id]
        for group in BUDGET_GROUPS:
            selected = [row for row in selected_budget if group == "all" or row["stratum"] == group]
            if not selected:
                continue
            metric_rows = [row[metric_key] for row in selected]
            output.append({"from_budget_id": budget_id, "group": group, **_metric_rows_summary(metric_rows)})
    return output


def _bootstrap_metric_values(
    panel: Mapping[str, Any],
    indices: np.ndarray,
) -> dict[str, list[float]]:
    candidate_count = len(panel["candidates"])
    y = np.asarray(panel["evaluation_targets"], dtype=np.float64)
    if y.shape != (200,) or not np.isfinite(y).all():
        raise ValueError("reference envelope evaluation target vector is invalid")
    predictions = np.asarray(
        [
            [candidate["heads"][head]["test_predictions"] for head in HEADS]
            for candidate in panel["candidates"]
        ],
        dtype=np.float64,
    )
    if predictions.shape != (candidate_count, len(HEADS), 200) or not np.isfinite(predictions).all():
        raise ValueError("reference envelope held-out prediction vectors are invalid")
    values = {key: [] for key in ("spearman", "top1_regret_r2", "top3_opportunity_regret_r2", "material_pair_accuracy")}
    for sample in indices:
        y_sample = y[sample]
        denominator = float(np.sum((y_sample - np.mean(y_sample)) ** 2))
        if denominator <= np.finfo(np.float64).eps:
            continue
        scores = np.empty((candidate_count, len(HEADS)), dtype=np.float64)
        for candidate_index in range(candidate_count):
            for head_index in range(len(HEADS)):
                prediction = predictions[candidate_index, head_index, sample]
                scores[candidate_index, head_index] = 1.0 - float(np.sum((y_sample - prediction) ** 2)) / denominator
        heldout_envelope = np.max(scores, axis=1)
        cv_envelope = np.asarray([float(candidate["cv_envelope_r2"]) for candidate in panel["candidates"]], dtype=np.float64)
        metric = ranking_metrics(heldout_envelope, cv_envelope)
        for key in values:
            if metric[key] is not None:
                values[key].append(float(metric[key]))
    if not values["top1_regret_r2"]:
        raise ValueError("all paired evaluation bootstrap samples were degenerate")
    return values


def evaluation_bootstrap(
    panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    new_panels = [panel for panel in panels if not bool(panel.get("source_archived"))]
    if not new_panels:
        return {
            "status": "unavailable_no_new_budget_predictions",
            "resamples": int(protocol["evaluation_bootstrap"]["resamples"]),
            "archive_100_status": "unavailable_archived_predictions_not_stored",
            "conditional_on_fitted_models": True,
        }
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for panel in new_panels:
        if not panel.get("evaluation_predictions_available"):
            raise ValueError("reference envelope new budget predictions are missing")
        by_target[str(panel["target"])].append(panel)
    output = []
    resamples = int(protocol["evaluation_bootstrap"]["resamples"])
    for target_position, target in enumerate(sorted(by_target)):
        target_panels = sorted(by_target[target], key=lambda panel: str(panel["budget_id"]))
        rng = np.random.default_rng(BOOTSTRAP_SEED + target_position)
        indices = rng.integers(0, 200, size=(resamples, 200), dtype=np.int64)
        for panel in target_panels:
            values = _bootstrap_metric_values(panel, indices)
            intervals = {}
            for metric, draws in values.items():
                lower, upper = np.quantile(np.asarray(draws, dtype=np.float64), [0.025, 0.975])
                intervals[metric] = (
                    {
                        "defined_count": len(draws),
                        "lower_95": float(lower),
                        "median": float(np.median(draws)),
                        "upper_95": float(upper),
                    }
                    if draws
                    else {
                        "defined_count": 0,
                        "lower_95": None,
                        "median": None,
                        "upper_95": None,
                        "status": "undefined_nonfinite_metric",
                    }
                )
            output.append({
                "target": target,
                "dataset_id": str(panel["dataset_id"]),
                "target_id": str(panel["target_id"]),
                "stratum": str(panel["stratum"]),
                "budget_id": str(panel["budget_id"]),
                "resamples": resamples,
                "seed": BOOTSTRAP_SEED + target_position,
                "row_count": 200,
                "paired_across_candidates_heads_budgets": True,
                "conditional_on_fitted_models": True,
                "intervals": intervals,
            })
    return {
        "status": "completed_paired_evaluation_row_bootstrap",
        "resamples": resamples,
        "panel_count": len(output),
        "rows": output,
        "archive_100_status": "unavailable_archived_predictions_not_stored",
        "paired_across_candidates_heads_budgets": True,
        "conditional_on_fitted_models": True,
        "nonfinite_handling": "degenerate bootstrap samples with zero target variance are omitted and counted by defined_count",
    }


def _budget_summary(
    archive: Mapping[str, Any],
    budget_panels: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [_budget_metric_row(panel) for panel in budget_panels]
    expected_budgets = [str(row["id"]) for row in _budget_specs(protocol)]
    if {str(row["budget_id"]) for row in rows} != set(expected_budgets):
        raise ValueError("reference envelope budget summary is incomplete")
    target_stability = _target_budget_stability(
        budget_panels,
        budget_ids=("n100_s4103", "n100_s4201", "n100_s4303"),
        label="archived_100_seed_stability",
    )
    union_stability = _target_budget_stability(
        budget_panels,
        budget_ids=("n200_u4103_4201", "n200_u4103_4303", "n200_u4201_4303"),
        label="new_200_union_stability",
    )
    comparisons = _budget_reference_comparisons(budget_panels, "n300_u4103_4201_4303")
    return {
        "status": "completed_budget_curve",
        "panel_count": len(budget_panels),
        "candidate_count": sum(len(panel["candidates"]) for panel in budget_panels),
        "metric_rows": rows,
        "group_summaries": _grouped_metric_summaries(rows, BUDGET_GROUPS),
        "archived_100_seed_stability": target_stability,
        "new_200_union_stability": union_stability,
        "budget_reference_comparisons_to_n300": comparisons,
        "budget_reference_comparison_summaries": {
            "heldout_envelope": _comparison_summary(comparisons, "heldout_envelope_metrics"),
            "cv_envelope": _comparison_summary(comparisons, "cv_envelope_metrics"),
        },
        "evaluation_bootstrap": evaluation_bootstrap(budget_panels, protocol),
        "shared_evaluation_dependence": "Each target reuses one fixed 200-row evaluation cohort across its three 100 cohorts, three 200 unions, and one 300 union. Bootstrap row indices are common to every candidate, head, and budget for that target.",
        "archive_reference": "100-label rows are copied from the source artifact; their missing prediction vectors make their evaluation-row bootstrap unavailable.",
    }


def _cluster_interval(
    values: Sequence[float], *, resamples: int, seed: int
) -> dict[str, Any]:
    observed = np.asarray(values, dtype=np.float64)
    if observed.ndim != 1 or len(observed) < 2 or not np.isfinite(observed).all():
        raise ValueError("reference envelope bootstrap cluster values are invalid")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(observed), size=(int(resamples), len(observed)), dtype=np.int64)
    draws = np.mean(observed[indices], axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(np.mean(observed)),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "cluster_count": int(len(observed)),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def _archive_cluster_intervals(
    archive: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rows = list(archive["metric_rows"])
    resamples = int(protocol["archive_audit"]["bootstrap"]["target_blocks"])
    output = {}
    for group_position, group in enumerate(ARCHIVE_GROUPS):
        selected = [row for row in rows if group == "all" or row["stratum"] == group]
        by_target = defaultdict(list)
        by_dataset = defaultdict(list)
        for row in selected:
            by_target[str(row["target"])].append(row)
        target_means = {}
        for target, target_rows in by_target.items():
            target_means[target] = {
                metric: float(np.mean([float(row[metric]) for row in target_rows]))
                for metric in ("top1_regret_r2", "top3_opportunity_regret_r2", "top3_mean_utility_regret_r2")
            }
            spearman_rows = [float(row["spearman"]) for row in target_rows if row["spearman"] is not None]
            if spearman_rows:
                target_means[target]["spearman"] = float(np.mean(spearman_rows))
            dataset_id = str(target.split("::", 1)[0])
            by_dataset[dataset_id].append(target_means[target])
        target_intervals = {
            metric: _cluster_interval(
                [target_means[target][metric] for target in sorted(target_means) if metric in target_means[target]],
                resamples=resamples,
                seed=2026091401 + group_position * 100 + metric_position,
            )
            for metric_position, metric in enumerate(("spearman", "top1_regret_r2", "top3_opportunity_regret_r2"))
            if any(metric in values for values in target_means.values())
        }
        dataset_means = {}
        for dataset_id, dataset_rows in by_dataset.items():
            dataset_means[dataset_id] = {
                metric: float(np.mean([row[metric] for row in dataset_rows if metric in row]))
                for metric in ("spearman", "top1_regret_r2", "top3_opportunity_regret_r2")
                if any(metric in row for row in dataset_rows)
            }
        dataset_intervals = {
            metric: _cluster_interval(
                [dataset_means[dataset][metric] for dataset in sorted(dataset_means) if metric in dataset_means[dataset]],
                resamples=resamples,
                seed=2026091401 + group_position * 1000 + 50 + metric_position,
            )
            for metric_position, metric in enumerate(("spearman", "top1_regret_r2", "top3_opportunity_regret_r2"))
            if sum(metric in values for values in dataset_means.values()) >= 2
        }
        output[group] = {
            "weighting": "equal target weight for target blocks; equal dataset weight for dataset clusters",
            "target_block_intervals": target_intervals,
            "dataset_cluster_intervals": dataset_intervals,
            "target_count": len(target_means),
            "dataset_count": len(dataset_means),
            "bootstrap_unit": "target block for target intervals; dataset cluster for dataset intervals",
        }
    return output


def _diagnostic_gates(
    archive: Mapping[str, Any], budget: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    archive_lookup = {
        (str(row["group"]), "mean_spearman"): row.get("mean_spearman")
        for row in archive["target_weighted_group_summaries"]
    }
    budget_rows = budget["group_summaries"]
    budget_lookup = {(str(row["budget_id"]), str(row["group"])): row for row in budget_rows}
    gates = protocol["diagnostic_gates"]
    checks = []

    def add(name: str, estimate: Optional[float], minimum: Optional[float] = None, maximum: Optional[float] = None) -> None:
        if estimate is None or not np.isfinite(float(estimate)):
            checks.append({"id": name, "status": "inconclusive_nonfinite_estimate", "estimate": estimate})
            return
        passed = True
        if minimum is not None:
            passed = passed and float(estimate) >= float(minimum)
        if maximum is not None:
            passed = passed and float(estimate) <= float(maximum)
        checks.append({"id": name, "status": "pass" if passed else "fail", "estimate": float(estimate), "minimum": minimum, "maximum": maximum})

    add(
        "archive_all_cv_vs_heldout_spearman",
        archive_lookup[("all", "mean_spearman")],
        minimum=float(gates["archive_all_cv_vs_heldout_spearman_minimum"]),
    )
    add(
        "archive_recoverable_cv_vs_heldout_spearman",
        archive_lookup[("recoverable_lp_dominated_control", "mean_spearman")],
        minimum=float(gates["archive_recoverable_cv_vs_heldout_spearman_minimum"]),
    )
    stability_values = [
        row["heldout_envelope_stability"]["mean_pair_spearman"]
        for row in archive["target_level_seed_stability"]
        if row["heldout_envelope_stability"]["mean_pair_spearman"] is not None
    ]
    add(
        "archive_target_seed_rank_stability",
        float(np.mean(stability_values)) if stability_values else None,
        minimum=float(gates["archive_target_seed_rank_stability_minimum"]),
    )
    n300 = budget_lookup[("n300_u4103_4201_4303", "all")]
    add(
        "budget_n300_cv_vs_heldout_spearman",
        n300.get("mean_spearman"),
        minimum=float(gates["budget_cv_vs_heldout_spearman_minimum"]),
    )
    comparison_summaries = budget["budget_reference_comparison_summaries"]["heldout_envelope"]
    for budget_id, gate_key in (
        ("n100_s4103", "budget_100_vs_300_heldout_spearman_minimum"),
        ("n100_s4201", "budget_100_vs_300_heldout_spearman_minimum"),
        ("n100_s4303", "budget_100_vs_300_heldout_spearman_minimum"),
        ("n200_u4103_4201", "budget_200_vs_300_heldout_spearman_minimum"),
        ("n200_u4103_4303", "budget_200_vs_300_heldout_spearman_minimum"),
        ("n200_u4201_4303", "budget_200_vs_300_heldout_spearman_minimum"),
    ):
        selected = [row for row in comparison_summaries if row["from_budget_id"] == budget_id and row["group"] == "all"]
        add(
            f"{budget_id}_vs_n300_heldout_spearman",
            selected[0].get("mean_spearman") if selected else None,
            minimum=float(gates[gate_key]),
        )
    gate_rows = checks
    return {
        "diagnostic_only": True,
        "weighting": "point gates use equal target weight; intervals use target or dataset cluster units as reported",
        "point_and_bound_policy": "point threshold failures are actionable; bootstrap bounds are reported for review and do not authorize promotion",
        "checks": gate_rows,
        "all_point_checks_passed": bool(gate_rows) and all(row["status"] == "pass" for row in gate_rows),
        "continuation": "Any failed or inconclusive check blocks a promotion/confirmation decision and requires design review; all passing checks still authorize only a prospective diagnostic stage.",
    }


def summarize(
    protocol: Mapping[str, Any],
    archive: Mapping[str, Any],
    budget_panels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(archive["panels"]) != 60 or archive["candidate_count"] != 300:
        raise ValueError("reference envelope archive grid is incomplete")
    if len(budget_panels) != 56 or sum(len(panel["candidates"]) for panel in budget_panels) != 280:
        raise ValueError("reference envelope budget grid is incomplete")
    budget = _budget_summary(archive, budget_panels, protocol)
    archive_result = dict(archive)
    archive_result["cluster_intervals"] = _archive_cluster_intervals(archive, protocol)
    result = {
        "schema_version": 1,
        "status": "completed_reference_envelope_reliability",
        "scope": dict(protocol["scope"]),
        "archive_audit": archive_result,
        "budget_curve": budget,
        "diagnostic_gates": _diagnostic_gates(archive_result, budget, protocol),
        "reserved_backbones_evaluated": [],
        "overlapindex_fit_count": 0,
        "shared_evaluation_dependence": "Within each target, the same 200 evaluation rows are shared by all three archived 100 cohorts, all three new 200 unions, and the new 300 union. Reported target and dataset cluster intervals retain this dependence.",
    }
    return result


def report(summary: Mapping[str, Any]) -> str:
    archive = summary["archive_audit"]
    budget = summary["budget_curve"]
    gates = summary["diagnostic_gates"]
    lines = [
        "# Continuous reference-envelope reliability audit",
        "",
        "This is a diagnostic-only audit. It does not promote a ranker, evaluate reserved backbones, or authorize confirmation.",
        "",
        "## Archive-only 100-label audit",
        "",
        f"The archive contains {archive['panel_count']} panels and {archive['candidate_count']} candidate rows across all 20 targets. Evaluation predictions were not stored, so evaluation-row bootstrap is unavailable for these rows.",
        "",
        "Group summaries below are target-weighted; the JSON also contains panel-weighted summaries and target/dataset cluster intervals.",
        "",
        "| Group | Panels | Target-weighted CV-envelope vs held-out rho | Top-1 regret | Head agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    target_summary = {str(row["group"]): row for row in archive["target_weighted_group_summaries"]}
    for group in ARCHIVE_GROUPS:
        row = target_summary[group]
        selected = [
            panel
            for panel in archive["panels"]
            if group == "all" or panel["stratum"] == group
        ]
        candidates = [candidate for panel in selected for candidate in panel["candidates"]]
        agreement = float(np.mean([
            candidate["cv_primary_winner"] == candidate["heldout_primary_winner"]
            for candidate in candidates
        ]))
        rho = "null" if row["mean_spearman"] is None else f"{row['mean_spearman']:.4f}"
        lines.append(f"| {group} | {row['target_count']} targets | {rho} | {row['mean_top1_regret_r2']:.4f} | {agreement:.3f} |")
    lines.extend([
        "",
        "## Budget curve",
        "",
        "The budget curve uses the eight source-selected frozen recoverable targets. It contains three archived 100-label cohorts, three fixed 200-label unions, and one fixed 300-label union per target. New 200/300 rows store per-head held-out predictions for paired evaluation-row bootstrap.",
        "",
        "| Budget | Panels | Candidates | New sklearn fits |",
        "|---|---:|---:|---:|",
        "| 100 labels (archived) | 24 | 120 | 0 |",
        "| 200 labels (three unions) | 24 | 120 | 14,160 |",
        "| 300 labels (one union) | 8 | 40 | 4,720 |",
        "",
        "The exact new budget total is 160 candidate rows, 640 family calls, 160 PCA fits, and 18,880 sklearn estimator fits. All fitting uses one-thread controls and the unchanged residual-ranking grids.",
        "",
        "## Diagnostic gates",
        "",
        f"All gates are diagnostic-only. Point estimates use equal target weighting. Bootstrap bounds are reported for review; a failed or inconclusive check blocks promotion/confirmation and triggers design review, while a pass only permits a prospective diagnostic stage.",
        "",
        "| Gate | Status | Estimate |",
        "|---|---|---:|",
    ])
    for check in gates["checks"]:
        estimate = "null" if check.get("estimate") is None else f"{float(check['estimate']):.4f}"
        lines.append(f"| {check['id']} | {check['status']} | {estimate} |")
    lines.extend([
        "",
        "Shared evaluation rows are reused within each target across every budget and seed/union; uncertainty must therefore be interpreted conditionally on the fitted models and with target-level dependence retained.",
        "",
    ])
    return "\n".join(lines)


def _checkpoint_name(panel: Mapping[str, Any]) -> str:
    return f"{panel['budget_id']}__{panel['dataset_id']}__{panel['target_id']}.json"


def _validate_budget_checkpoint(
    checkpoint: Mapping[str, Any],
    protocol: Mapping[str, Any],
    target: str,
    budget: Mapping[str, Any],
    source_hash_map: Mapping[str, str],
) -> dict[str, Any]:
    panel = checkpoint.get("panel")
    if checkpoint.get("source_hashes") != dict(source_hash_map) or not isinstance(panel, Mapping):
        raise ValueError("reference envelope budget checkpoint source identity changed")
    expected_position = next(
        index for index, row in enumerate(_budget_specs(protocol)) if str(row["id"]) == str(budget["id"])
    )
    if (
        str(panel.get("target")) != target
        or str(panel.get("budget_id")) != str(budget["id"])
        or int(panel.get("budget_position", -1)) != expected_position
        or int(panel.get("training_rows", -1)) != int(budget["rows"])
        or int(panel.get("evaluation_rows", -1)) != 200
        or tuple(str(value) for value in panel.get("training_replicate_seeds", ()))
        != tuple(str(value) for value in budget["training_replicate_seeds"])
        or tuple(row.get("candidate_id") for row in panel.get("candidates", ())) != BACKBONES
        or int(panel.get("candidate_count", -1)) != len(BACKBONES)
        or not bool(panel.get("evaluation_predictions_available"))
    ):
        raise ValueError("reference envelope budget checkpoint identity changed")
    for candidate in panel["candidates"]:
        if set(candidate.get("heads", ())) != set(HEADS):
            raise ValueError("reference envelope checkpoint head set changed")
        for head in HEADS:
            predictions = candidate["heads"][head].get("test_predictions")
            if not isinstance(predictions, list) or len(predictions) != 200 or not np.isfinite(np.asarray(predictions, dtype=np.float64)).all():
                raise ValueError("reference envelope checkpoint predictions are incomplete")
        if not isinstance(candidate.get("fit_seed"), int) or not isinstance(candidate.get("pca_seed"), int):
            raise ValueError("reference envelope checkpoint seeds are missing")
    return dict(panel)


def _git_identity() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def _validate_existing_json_identity(path: Path, source_hash_map: Mapping[str, str]) -> None:
    if not path.exists():
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"reference envelope existing artifact is not valid JSON: {path}") from exc
    if value.get("source_hashes") != dict(source_hash_map):
        raise ValueError(f"reference envelope existing artifact source identity changed: {path}")


def _validate_terminal_manifest(
    manifest: Mapping[str, Any], output_root: Path, source_hash_map: Mapping[str, str]
) -> dict[str, Any]:
    if manifest.get("status") != "completed_reference_envelope_reliability":
        raise ValueError("reference envelope terminal status changed")
    if manifest.get("source_hashes") != dict(source_hash_map):
        raise ValueError("reference envelope terminal source identity changed")
    artifacts = manifest.get("artifacts", {})
    for name in ("raw_results.json", "summary.json", "report.md"):
        path = output_root / name
        if not path.is_file() or sha256_path(path) != artifacts.get(name):
            raise ValueError(f"reference envelope terminal artifact is missing or changed: {name}")
    counts = manifest.get("counts", {})
    if counts != {
        "archive_panels": 60,
        "archive_candidates": 300,
        "budget_panels": 56,
        "budget_candidates": 280,
        "new_budget_panels": 32,
        "new_budget_candidates": 160,
        "new_pca_fits": 160,
        "new_head_family_calls": 640,
        "new_sklearn_estimator_fits": 18880,
        "reserved_backbones": 0,
        "overlapindex_fits": 0,
    }:
        raise ValueError("reference envelope terminal counts changed")
    return dict(manifest)


def run(
    input_root: Path = source_prepare.DEFAULT_INPUT_ROOT,
    embedding_root: Path = source_extract.DEFAULT_OUTPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    protocol = load_protocol()
    raw_source, source_summary, _source_manifest, cohort = validate_sources(protocol)
    source_hash_map = source_hashes()
    output_root = Path(output_root).resolve()
    terminal_path = output_root / "manifest.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("reference envelope terminal manifest exists")
        return _validate_terminal_manifest(
            json.loads(terminal_path.read_text(encoding="utf-8")), output_root, source_hash_map
        )
    for path in (output_root / "raw_results.json", output_root / "summary.json"):
        _validate_existing_json_identity(path, source_hash_map)
    report_path = output_root / "report.md"
    if report_path.exists() and not resume:
        raise FileExistsError("reference envelope report already exists without terminal manifest")
    archive = archive_audit(raw_source, protocol)
    strata = _stratum_by_target(protocol)
    source_panel_lookup = {
        (_target_key(str(panel["dataset_id"]), str(panel["target_id"])), int(panel["seed"])): panel
        for panel in raw_source["panels"]
    }
    budget_panels = []
    for budget in _budget_specs(protocol):
        budget_id = str(budget["id"])
        if bool(budget.get("archived")):
            for target in protocol["budget_curve"]["targets"]:
                seed = int(budget["training_replicate_seeds"][0])
                source_panel = source_panel_lookup[(str(target), seed)]
                budget_panels.append(
                    _source_100_budget_panel(source_panel, str(target), seed, strata[str(target)], budget_id)
                )
    new_specs = [budget for budget in _budget_specs(protocol) if not bool(budget.get("archived"))]
    datasets = None
    checkpoint_root = output_root / "checkpoints"
    for budget_position, budget in enumerate(new_specs):
        for target in protocol["budget_curve"]["targets"]:
            dataset_id, target_id = _split_target_key(str(target))
            expected_identity = {
                "budget_id": str(budget["id"]),
                "target": str(target),
                "dataset_id": dataset_id,
                "target_id": target_id,
            }
            checkpoint_path = checkpoint_root / _checkpoint_name(expected_identity)
            panel = None
            if checkpoint_path.exists():
                if not resume:
                    raise FileExistsError(f"reference envelope checkpoint exists: {checkpoint_path}")
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                panel = _validate_budget_checkpoint(
                    checkpoint, protocol, str(target), budget, source_hash_map
                )
            else:
                if datasets is None:
                    datasets = source_run.load_dataset(input_root, embedding_root)
                panel = _fit_budget_panel(
                    protocol, datasets, cohort, str(target), budget, budget_position + 3
                )
                _atomic_json(
                    checkpoint_path,
                    {
                        "schema_version": 1,
                        "status": "completed_reference_envelope_budget_checkpoint",
                        "source_hashes": source_hash_map,
                        "panel": panel,
                    },
                )
            budget_panels.append(panel)
    summary = summarize(protocol, archive, budget_panels)
    raw = {
        "schema_version": 1,
        "status": "completed_reference_envelope_reliability",
        "protocol": protocol,
        "source_hashes": source_hash_map,
        "archive_audit": archive,
        "budget_panels": budget_panels,
        "reserved_backbones_evaluated": [],
        "overlapindex_fit_count": 0,
    }
    raw_path = output_root / "raw_results.json"
    summary_path = output_root / "summary.json"
    _atomic_json(raw_path, raw)
    _atomic_json(summary_path, summary)
    _atomic_bytes(report_path, report(summary).encode("utf-8"))
    counts = {
        "archive_panels": 60,
        "archive_candidates": 300,
        "budget_panels": 56,
        "budget_candidates": 280,
        "new_budget_panels": 32,
        "new_budget_candidates": 160,
        "new_pca_fits": 160,
        "new_head_family_calls": 640,
        "new_sklearn_estimator_fits": 18880,
        "reserved_backbones": 0,
        "overlapindex_fits": 0,
    }
    manifest = {
        "schema_version": 1,
        "status": "completed_reference_envelope_reliability",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hash_map,
        "artifacts": {
            "raw_results.json": sha256_path(raw_path),
            "summary.json": sha256_path(summary_path),
            "report.md": sha256_path(report_path),
        },
        "counts": counts,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "scikit_learn": __import__("sklearn").__version__,
            **_git_identity(),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "LOKY_MAX_CPU_COUNT": os.environ.get("LOKY_MAX_CPU_COUNT"),
            "n_jobs": 1,
        },
        "archive_evaluation_predictions_available": False,
        "budget_new_evaluation_predictions_available": True,
        "reserved_backbones_evaluated": [],
        "overlapindex_fit_count": 0,
        "diagnostic_only": True,
    }
    _atomic_json(terminal_path, manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=source_prepare.DEFAULT_INPUT_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=source_extract.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(canonical_json(run(args.input_root, args.embedding_root, args.output_root, resume=args.resume)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
