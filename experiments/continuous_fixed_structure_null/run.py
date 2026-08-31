"""Frozen strict fixed-structure evaluation-null experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import KFold

from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_multiview_ranking import run as multiview
from experiments.continuous_null_diagnostic import run as parent
from experiments.continuous_prototype_capacity import run as capacity
from experiments.continuous_save_ranking import run as save


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_PARENT_RAW = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "continuous_null_diagnostic"
    / "raw_results.json"
)


def _load_protocol() -> Dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_exploratory_before_outcomes":
        raise ValueError("protocol is not frozen for exploratory execution")
    return protocol


def _method_specs(protocol: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row["method_id"]): dict(row)
        for row in protocol["continuous_oi"]["geometries"]
    }


def _validate_protocol_against_code(protocol: Mapping[str, Any]) -> None:
    expected = {
        "save8_k4": ("save", None),
        "rff64_mean8_k4": ("rff_mean", None),
        "save_rffmean_k4": ("multiview", ("save", "rff_mean")),
        "save_rffmean_residual_k4": (
            "multiview",
            ("save", "rff_mean", "residual"),
        ),
    }
    observed = {
        method_id: (
            str(row["geometry"]),
            tuple(row["blocks"]) if "blocks" in row else None,
        )
        for method_id, row in _method_specs(protocol).items()
    }
    if observed != expected:
        raise ValueError("frozen fixed-structure geometry grid does not match code")
    if protocol["seeds"] != [1009, 1103, 1201]:
        raise ValueError("frozen seed set does not match code")
    if len(protocol["families"]) != 6:
        raise ValueError("frozen family set must contain six families")
    shared = protocol["shared_execution"]
    if int(shared["permutations"]) != 32:
        raise ValueError("strict null must use exactly 32 permutations")
    if shared["common_bank_seed"] != "2026083101 + panel seed * 100 + fold":
        raise ValueError("common-bank seed rule does not match code")


def _load_parent_raw(
    protocol: Mapping[str, Any],
    path: Path = DEFAULT_PARENT_RAW,
) -> Dict[str, Any]:
    resolved = path.resolve()
    ancestry = protocol["ancestry"]
    if common._sha256(resolved) != ancestry["parent_raw_results_sha256"]:
        raise ValueError("parent raw-results SHA-256 does not match protocol")
    if common._sha256(parent.PROTOCOL_PATH) != ancestry["parent_protocol_sha256"]:
        raise ValueError("parent protocol bytes do not match frozen ancestry")
    if common._sha256(parent.RUNNER_PATH) != ancestry["parent_runner_sha256"]:
        raise ValueError("parent runner bytes do not match frozen ancestry")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if raw.get("artifact_status") != "completed":
        raise ValueError("parent null diagnostic is not completed")
    expected = len(protocol["seeds"]) * len(protocol["families"])
    if len(raw.get("panels", [])) != expected:
        raise ValueError("parent null diagnostic has an incomplete panel grid")
    return raw


def _parent_panel_map(raw: Mapping[str, Any]) -> Dict[tuple[int, str], Mapping[str, Any]]:
    result = {
        (int(panel["seed"]), str(panel["family"])): panel
        for panel in raw["panels"]
    }
    if len(result) != len(raw["panels"]):
        raise ValueError("parent raw results contain duplicate panels")
    return result


def _fixed_evaluation_structure(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """Build the actual held-out structure once for strict null replay."""
    X_arr = estimator._validate_X(X)
    y_arr = estimator._validate_Y(y)
    X_prepared = estimator._prep_X(X_arr)
    y_scaled = estimator._transform_targets(y_arr)
    target_cells = estimator._assign_target_cells(y_scaled)
    own_proto = estimator._model.bmu_for_class_batch(X_prepared, target_cells)
    _counts, adjacency = estimator._adjacency_for_model(
        X_prepared,
        own_proto,
        estimator._model,
    )
    rows_by_proto = {
        int(pid): np.flatnonzero(own_proto == pid)
        for pid in np.unique(own_proto)
    }
    support = {pid: int(rows.size) for pid, rows in rows_by_proto.items()}
    values = {pid: y_scaled[rows] for pid, rows in rows_by_proto.items()}
    actual_loss, local_loss = estimator._loss_for_components(
        values,
        support,
        adjacency,
    )
    return {
        "y_scaled": y_scaled,
        "target_cells": target_cells,
        "own_proto": own_proto,
        "rows_by_proto": rows_by_proto,
        "support": support,
        "adjacency": dict(adjacency),
        "actual_loss": float(actual_loss),
        "local_loss": dict(local_loss),
    }


def _strict_null_samples(
    estimator: Any,
    structure: Mapping[str, Any],
    permutations: Sequence[np.ndarray],
) -> list[float]:
    """Permute only targets; keep assignments, supports, and adjacency fixed."""
    y_scaled = np.asarray(structure["y_scaled"])
    rows_by_proto = structure["rows_by_proto"]
    support = structure["support"]
    adjacency = structure["adjacency"]
    losses = []
    for permutation in permutations:
        indices = np.asarray(permutation, dtype=np.int64)
        if indices.shape != (len(y_scaled),):
            raise ValueError("permutation length does not match held-out rows")
        permuted = y_scaled[indices]
        values = {
            int(pid): permuted[rows]
            for pid, rows in rows_by_proto.items()
        }
        loss, _local = estimator._loss_for_components(
            values,
            support,
            adjacency,
        )
        losses.append(float(loss))
    return losses


def _aggregate_strict(folds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ratios = [row["strict_loss_ratio"] for row in folds]
    return {
        "mean_actual_loss": float(np.mean([row["actual_loss"] for row in folds])),
        "mean_strict_null_loss": float(
            np.mean([row["strict_null_loss"] for row in folds])
        ),
        "mean_strict_loss_ratio": (
            float(np.mean(ratios))
            if all(value is not None for value in ratios)
            else None
        ),
        "strict_score": float(np.mean([row["strict_score"] for row in folds])),
        "mean_fold_strict_null_standard_deviation": float(
            np.mean([row["strict_null_standard_deviation"] for row in folds])
        ),
        "mean_fold_strict_null_standard_error": float(
            np.mean([row["strict_null_standard_error"] for row in folds])
        ),
        "folds": list(folds),
    }


def evaluate_selector(
    spec: Mapping[str, Any],
    X: np.ndarray,
    primitive: np.ndarray,
    y: np.ndarray,
    panel_seed: int,
    candidate_position: int,
    parent_selector: Mapping[str, Any],
) -> Dict[str, Any]:
    splitter = KFold(n_splits=4, shuffle=True, random_state=int(panel_seed))
    parent_folds = parent_selector["folds"]
    if len(parent_folds) != 4:
        raise ValueError("parent selector does not contain four folds")
    folds = []
    started = time.perf_counter()
    for fold, (train, holdout) in enumerate(splitter.split(X)):
        fit_seed = parent._candidate_fit_seed(panel_seed, candidate_position, fold)
        transformed_train, transformed_holdout, geometry_diagnostics = (
            multiview._prepare_fold_geometry(
                spec,
                X[train],
                X[holdout],
                primitive[train],
                primitive[holdout],
                y[train],
                fit_seed,
            )
        )
        estimator = clone(save._base_selector(fit_seed))
        estimator._fit_geometry_only(transformed_train, y[train])
        structure = _fixed_evaluation_structure(
            estimator,
            transformed_holdout,
            y[holdout],
        )
        expected_fold = parent_folds[fold]
        if int(expected_fold["fold"]) != fold:
            raise ValueError("parent fold order mismatch")
        if int(expected_fold["fit_seed"]) != fit_seed:
            raise ValueError("parent fit seed mismatch")
        if structure["actual_loss"] != float(expected_fold["actual_loss"]):
            raise RuntimeError("replayed actual fold loss does not match parent exactly")
        bank_seed = parent._common_bank_seed(panel_seed, fold)
        if int(expected_fold["common_bank_seed"]) != bank_seed:
            raise ValueError("parent common-bank seed mismatch")
        bank = parent._permutation_bank(len(holdout), 32, bank_seed)
        strict_samples = _strict_null_samples(estimator, structure, bank)
        strict_null = float(np.mean(strict_samples))
        derived = parent._score_from_losses(structure["actual_loss"], strict_null)
        standard_deviation = float(np.std(strict_samples, ddof=1))
        folds.append(
            {
                "fold": int(fold),
                "fit_seed": int(fit_seed),
                "common_bank_seed": int(bank_seed),
                "actual_loss": float(structure["actual_loss"]),
                "strict_null_loss": strict_null,
                "strict_null_loss_samples": strict_samples,
                "strict_loss_ratio": derived["loss_ratio"],
                "strict_score": derived["score"],
                "strict_null_standard_deviation": standard_deviation,
                "strict_null_standard_error": float(
                    standard_deviation / np.sqrt(len(strict_samples))
                ),
                "prototype_count": int(estimator._model.n_clusters_total),
                "occupied_evaluation_prototype_count": len(structure["support"]),
                "evaluation_support_total": int(sum(structure["support"].values())),
                "evaluation_adjacency_edge_count": len(structure["adjacency"]),
                "geometry_diagnostics": geometry_diagnostics,
            }
        )
    result = _aggregate_strict(folds)
    if result["mean_actual_loss"] != float(parent_selector["mean_actual_loss"]):
        raise RuntimeError("replayed mean actual loss does not match parent exactly")
    result["current_common16"] = dict(parent_selector["policies"]["common16"])
    result["current_common32"] = dict(parent_selector["policies"]["common32"])
    result["wall_seconds"] = float(time.perf_counter() - started)
    return result


def _parent_candidate(
    panel: Mapping[str, Any],
    candidate_id: str,
) -> Mapping[str, Any]:
    matches = [row for row in panel["candidates"] if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"parent candidate identity mismatch: {candidate_id}")
    return matches[0]


def run_panel(
    protocol: Mapping[str, Any],
    seed: int,
    family: str,
    parent_panel: Mapping[str, Any],
) -> Dict[str, Any]:
    generated = capacity.generate_panel(protocol, seed, family)
    methods = _method_specs(protocol)
    candidates = []
    for position, candidate in enumerate(generated["candidates"]):
        old = _parent_candidate(parent_panel, candidate["candidate_id"])
        for key in ("quality", "nuisance_scale", "effective_nuisance_scale"):
            if float(candidate[key]) != float(old[key]):
                raise ValueError(f"parent candidate metadata mismatch for {key}")
        selectors = {
            method_id: evaluate_selector(
                spec,
                candidate["X_train"],
                candidate["primitive_train"],
                generated["y_train"],
                seed,
                position,
                old["selectors"][method_id],
            )
            for method_id, spec in methods.items()
        }
        candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "quality": float(candidate["quality"]),
                "nuisance_scale": float(candidate["nuisance_scale"]),
                "effective_nuisance_scale": float(
                    candidate["effective_nuisance_scale"]
                ),
                "best_head_envelope_r2": float(old["best_head_envelope_r2"]),
                "best_head_ids": list(old["best_head_ids"]),
                "heads": old["heads"],
                "parent_candidate4": old["parent_candidate4"],
                "selectors": selectors,
            }
        )
    return {
        "seed": int(seed),
        "family": family,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _defined_spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = [float(value) for value in left]
    y = [float(value) for value in right]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def _selector_values(
    panel: Mapping[str, Any],
    method_id: str,
    estimand: str,
) -> list[float]:
    rows = panel["candidates"]
    if estimand == "parent_bounded_oi":
        return [float(row["parent_candidate4"][method_id]["score"]) for row in rows]
    if estimand == "negative_actual_loss":
        return [-float(row["selectors"][method_id]["mean_actual_loss"]) for row in rows]
    if estimand in {"current_common16_ratio", "current_common32_ratio"}:
        policy = estimand.split("_ratio")[0].replace("current_", "")
        values = [
            row["selectors"][method_id][f"current_{policy}"]["mean_loss_ratio"]
            for row in rows
        ]
        if any(value is None for value in values):
            raise ValueError("current loss-ratio ranking contains undefined values")
        return [-float(value) for value in values]
    if estimand in {"current_common16_oi", "current_common32_oi"}:
        policy = estimand.split("_oi")[0].replace("current_", "")
        return [
            float(row["selectors"][method_id][f"current_{policy}"]["score"])
            for row in rows
        ]
    if estimand == "strict_common32_ratio":
        values = [
            row["selectors"][method_id]["mean_strict_loss_ratio"] for row in rows
        ]
        if any(value is None for value in values):
            raise ValueError("strict loss-ratio ranking contains undefined values")
        return [-float(value) for value in values]
    if estimand == "strict_common32_oi":
        return [
            float(row["selectors"][method_id]["strict_score"]) for row in rows
        ]
    raise ValueError(f"unknown estimand: {estimand!r}")


def _panel_metric(
    panel: Mapping[str, Any],
    method_id: str,
    estimand: str,
) -> Dict[str, Any]:
    candidates = panel["candidates"]
    envelope = np.asarray(
        [row["best_head_envelope_r2"] for row in candidates], dtype=float
    )
    selector = np.asarray(_selector_values(panel, method_id, estimand), dtype=float)
    selected = np.flatnonzero(
        np.isclose(selector, np.max(selector), atol=1.0e-12, rtol=0.0)
    )
    best = float(np.max(envelope))
    regret = float(best - np.mean(envelope[selected]))
    return {
        "seed": int(panel["seed"]),
        "family": panel["family"],
        "method_id": method_id,
        "estimand": estimand,
        "spearman": _defined_spearman(selector, envelope),
        "selection_regret_r2": regret,
        "exact_best": bool(regret <= 1.0e-12),
        "within_0_01_r2": bool(regret <= 0.01 + 1.0e-12),
        "selected_candidate_ids": [
            candidates[index]["candidate_id"] for index in selected.tolist()
        ],
    }


def _mean_defined(values: Iterable[Any]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def _aggregate_metrics(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for method_id in _method_specs(protocol):
        result[method_id] = {}
        estimands = sorted(
            {row["estimand"] for row in rows if row["method_id"] == method_id}
        )
        for estimand in estimands:
            selected = [
                row for row in rows
                if row["method_id"] == method_id and row["estimand"] == estimand
            ]
            by_family = {}
            for family in protocol["families"]:
                family_rows = [row for row in selected if row["family"] == family]
                by_family[family] = {
                    "mean_spearman": _mean_defined(row["spearman"] for row in family_rows),
                    "mean_regret_r2": float(
                        np.mean([row["selection_regret_r2"] for row in family_rows])
                    ),
                    "exact_best_rate": float(np.mean([row["exact_best"] for row in family_rows])),
                    "within_0_01_rate": float(
                        np.mean([row["within_0_01_r2"] for row in family_rows])
                    ),
                }
            result[method_id][estimand] = {
                "panel_count": len(selected),
                "mean_spearman": _mean_defined(row["spearman"] for row in selected),
                "mean_regret_r2": float(
                    np.mean([row["selection_regret_r2"] for row in selected])
                ),
                "max_regret_r2": float(
                    np.max([row["selection_regret_r2"] for row in selected])
                ),
                "exact_best_rate": float(np.mean([row["exact_best"] for row in selected])),
                "within_0_01_rate": float(
                    np.mean([row["within_0_01_r2"] for row in selected])
                ),
                "by_family": by_family,
            }
    return result


def _null_diagnostics(
    protocol: Mapping[str, Any],
    panels: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    result = []
    for panel in panels:
        envelope = [float(row["best_head_envelope_r2"]) for row in panel["candidates"]]
        for method_id in _method_specs(protocol):
            selectors = [row["selectors"][method_id] for row in panel["candidates"]]
            actual = [float(row["mean_actual_loss"]) for row in selectors]
            current = [
                float(row["current_common32"]["mean_null_loss"]) for row in selectors
            ]
            strict = [float(row["mean_strict_null_loss"]) for row in selectors]
            current_scores = [
                float(row["current_common32"]["score"]) for row in selectors
            ]
            strict_scores = [float(row["strict_score"]) for row in selectors]
            result.append(
                {
                    "seed": int(panel["seed"]),
                    "family": panel["family"],
                    "method_id": method_id,
                    "strict_null_actual_spearman": _defined_spearman(strict, actual),
                    "strict_null_envelope_spearman": _defined_spearman(strict, envelope),
                    "current_null_envelope_spearman": _defined_spearman(current, envelope),
                    "current_strict_null_spearman": _defined_spearman(current, strict),
                    "mean_absolute_strict_current_null_difference": float(
                        np.mean(np.abs(np.asarray(strict) - np.asarray(current)))
                    ),
                    "mean_absolute_strict_current_score_difference": float(
                        np.mean(
                            np.abs(np.asarray(strict_scores) - np.asarray(current_scores))
                        )
                    ),
                    "mean_fold_strict_null_standard_error": float(
                        np.mean(
                            [
                                row["mean_fold_strict_null_standard_error"]
                                for row in selectors
                            ]
                        )
                    ),
                }
            )
    return result


def _decision(
    protocol: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    config = protocol["decision_rule"]
    results = {}
    for method_id, method in metrics.items():
        current = method["current_common32_oi"]
        strict = method["strict_common32_oi"]
        periodic_gain = float(
            np.mean(
                [
                    strict["by_family"][family]["mean_spearman"]
                    - current["by_family"][family]["mean_spearman"]
                    for family in config["periodic_families"]
                ]
            )
        )
        checks = [
            {
                "criterion": "mean_periodic_spearman_gain",
                "observed": periodic_gain,
                "minimum": float(config["minimum_mean_periodic_spearman_gain"]),
                "pass": bool(periodic_gain >= config["minimum_mean_periodic_spearman_gain"]),
            },
            {
                "criterion": "mean_regret_preservation",
                "observed": float(strict["mean_regret_r2"]),
                "maximum": float(current["mean_regret_r2"])
                + float(config["maximum_mean_regret_increase"]),
                "pass": bool(
                    strict["mean_regret_r2"]
                    <= current["mean_regret_r2"] + config["maximum_mean_regret_increase"]
                ),
            },
            {
                "criterion": "minimum_overall_spearman",
                "observed": float(strict["mean_spearman"]),
                "minimum": float(config["minimum_overall_spearman"]),
                "pass": bool(strict["mean_spearman"] >= config["minimum_overall_spearman"]),
            },
            {
                "criterion": "maximum_mean_regret",
                "observed": float(strict["mean_regret_r2"]),
                "maximum": float(config["maximum_mean_regret"]),
                "pass": bool(strict["mean_regret_r2"] <= config["maximum_mean_regret"]),
            },
        ]
        for family in config["preserve_families"]:
            observed = float(strict["by_family"][family]["mean_spearman"])
            minimum = float(current["by_family"][family]["mean_spearman"]) - float(
                config["maximum_preservation_spearman_drop"]
            )
            checks.append(
                {
                    "criterion": f"preserve_{family}_spearman",
                    "observed": observed,
                    "minimum": minimum,
                    "pass": bool(observed >= minimum),
                }
            )
        results[method_id] = {
            "status": "pass" if all(row["pass"] for row in checks) else "fail",
            "checks": checks,
        }
    passing = [method_id for method_id, row in results.items() if row["status"] == "pass"]
    return {
        "status": "pass" if passing else "fail",
        "passing_methods": passing,
        "methods": results,
    }


def summarize(
    protocol: Mapping[str, Any],
    panels: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    estimands = [
        "parent_bounded_oi",
        "negative_actual_loss",
        "current_common16_ratio",
        "current_common16_oi",
        "current_common32_ratio",
        "current_common32_oi",
        "strict_common32_ratio",
        "strict_common32_oi",
    ]
    panel_metrics = [
        _panel_metric(panel, method_id, estimand)
        for panel in panels
        for method_id in _method_specs(protocol)
        for estimand in estimands
    ]
    metrics = _aggregate_metrics(protocol, panel_metrics)
    return {
        "schema_version": 1,
        "panel_count": len(panels),
        "candidate_panel_rows": sum(int(panel["candidate_count"]) for panel in panels),
        "selector_rows": sum(
            int(panel["candidate_count"]) * len(_method_specs(protocol))
            for panel in panels
        ),
        "metrics": metrics,
        "panel_metrics": panel_metrics,
        "null_diagnostics": _null_diagnostics(protocol, panels),
        "decision": _decision(protocol, metrics),
    }


def run(output: Path, parent_raw_path: Path = DEFAULT_PARENT_RAW) -> Dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    parent_raw = _load_parent_raw(protocol, parent_raw_path)
    parents = _parent_panel_map(parent_raw)
    source_paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "parent_raw_results": parent_raw_path.resolve(),
        "parent_runner": parent.RUNNER_PATH,
        "capacity_runner": capacity.RUNNER_PATH,
        "multiview_runner": multiview.RUNNER_PATH,
        "save_runner": save.RUNNER_PATH,
        "common_runner": common.RUNNER_PATH,
        "continuous_overlap_index": (
            Path(__file__).resolve().parents[2]
            / "overlapindex"
            / "ContinuousOverlapIndex.py"
        ),
    }
    source_hashes = {key: common._sha256(path) for key, path in source_paths.items()}
    output = output.resolve()
    for terminal in ("raw_results.json", "summary.json", "manifest.json"):
        if (output / terminal).exists():
            raise FileExistsError(f"refusing to overwrite {output / terminal}")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    total = len(protocol["seeds"]) * len(protocol["families"])
    completed = 0
    for seed in protocol["seeds"]:
        for family in protocol["families"]:
            identity = (int(seed), str(family))
            path = checkpoints / f"seed_{seed}__{family}.json"
            if path.exists():
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                if checkpoint.get("source_hashes") != source_hashes:
                    raise ValueError(f"stale source checkpoint: {path}")
                panel = checkpoint["panel"]
            else:
                started = time.perf_counter()
                panel = run_panel(
                    protocol,
                    identity[0],
                    identity[1],
                    parents[identity],
                )
                common._atomic_json(
                    path,
                    {
                        "schema_version": 1,
                        "source_hashes": source_hashes,
                        "panel_wall_seconds": float(time.perf_counter() - started),
                        "panel": panel,
                    },
                )
            panels.append(panel)
            completed += 1
            print(
                f"completed {completed}/{total}: seed={seed} family={family}",
                flush=True,
            )
    raw = {
        "schema_version": 1,
        "artifact_status": "completed",
        "source_hashes": source_hashes,
        "implementation_commit": os.environ.get("CONTINUOUS_OI_COMMIT"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "protocol": protocol,
        "panels": panels,
    }
    summary = summarize(protocol, panels)
    common._atomic_json(output / "raw_results.json", raw)
    common._atomic_json(output / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "source_hashes": source_hashes,
        "files": {
            name: {
                "sha256": common._sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in ("raw_results.json", "summary.json")
        },
        "checkpoint_count": len(panels),
    }
    common._atomic_json(output / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-raw", type=Path, default=DEFAULT_PARENT_RAW)
    args = parser.parse_args()
    summary = run(args.output, args.parent_raw)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "strict_common32": {
                    method_id: {
                        "mean_spearman": row["strict_common32_oi"]["mean_spearman"],
                        "mean_regret_r2": row["strict_common32_oi"]["mean_regret_r2"],
                        "sinusoid_spearman": row["strict_common32_oi"]["by_family"][
                            "sinusoid"
                        ]["mean_spearman"],
                        "triangle_wave_spearman": row["strict_common32_oi"]["by_family"][
                            "triangle_wave"
                        ]["mean_spearman"],
                    }
                    for method_id, row in summary["metrics"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
