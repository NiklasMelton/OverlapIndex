"""Frozen continuous edge-level target-continuity experiment."""

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
from experiments.continuous_fixed_structure_null import run as strict_parent
from experiments.continuous_multiview_ranking import run as multiview
from experiments.continuous_null_diagnostic import run as null_parent
from experiments.continuous_prototype_capacity import run as capacity
from experiments.continuous_save_ranking import run as save


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_PARENT_RAW = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "continuous_fixed_structure_null"
    / "raw_results.json"
)
EDGE_IDS = ("row_knn5", "prototype_cross")


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
        raise ValueError("frozen edge-score geometry grid does not match code")
    if tuple(protocol["edge_scores"]) != EDGE_IDS:
        raise ValueError("frozen edge-score order does not match code")
    if protocol["seeds"] != [1009, 1103, 1201]:
        raise ValueError("frozen seed set does not match code")
    if len(protocol["families"]) != 6:
        raise ValueError("frozen family set must contain six families")
    if int(protocol["shared_execution"]["permutations"]) != 32:
        raise ValueError("edge scores require exactly 32 permutations")


def _load_parent_raw(
    protocol: Mapping[str, Any],
    path: Path = DEFAULT_PARENT_RAW,
) -> Dict[str, Any]:
    resolved = path.resolve()
    ancestry = protocol["ancestry"]
    if common._sha256(resolved) != ancestry["parent_raw_results_sha256"]:
        raise ValueError("parent raw-results SHA-256 does not match protocol")
    if common._sha256(strict_parent.PROTOCOL_PATH) != ancestry[
        "parent_protocol_sha256"
    ]:
        raise ValueError("parent protocol bytes do not match frozen ancestry")
    if common._sha256(strict_parent.RUNNER_PATH) != ancestry[
        "parent_runner_sha256"
    ]:
        raise ValueError("parent runner bytes do not match frozen ancestry")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if raw.get("artifact_status") != "completed":
        raise ValueError("parent strict-null artifact is not completed")
    expected = len(protocol["seeds"]) * len(protocol["families"])
    if len(raw.get("panels", [])) != expected:
        raise ValueError("parent strict-null artifact has an incomplete panel grid")
    return raw


def _parent_panel_map(raw: Mapping[str, Any]) -> Dict[tuple[int, str], Mapping[str, Any]]:
    result = {
        (int(panel["seed"]), str(panel["family"])): panel
        for panel in raw["panels"]
    }
    if len(result) != len(raw["panels"]):
        raise ValueError("parent raw results contain duplicate panels")
    return result


def _row_knn_edges(X: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("row-kNN graph requires a finite two-dimensional matrix")
    n_rows = int(values.shape[0])
    if type(k) is not int or k < 1 or k >= n_rows:
        raise ValueError("row-kNN k must be an integer in [1, n_rows)")
    norms = np.sum(values * values, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * values @ values.T
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :k]
    sources = np.repeat(np.arange(n_rows, dtype=np.int64), k)
    targets = np.asarray(neighbors.reshape(-1), dtype=np.int64)
    if np.any(sources == targets):
        raise RuntimeError("row-kNN graph contains a self edge")
    return sources, targets


def _row_edge_loss(
    y_scaled: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
) -> float:
    values = np.asarray(y_scaled, dtype=float)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("edge experiment requires a univariate target matrix")
    source_ids = np.asarray(sources, dtype=np.int64)
    target_ids = np.asarray(targets, dtype=np.int64)
    if source_ids.shape != target_ids.shape or source_ids.size == 0:
        raise ValueError("edge arrays must be aligned and nonempty")
    return float(np.mean(np.abs(values[source_ids, 0] - values[target_ids, 0])))


def _mean_cross_absolute(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float).reshape(-1)
    right = np.sort(np.asarray(b, dtype=float).reshape(-1))
    if left.size == 0 or right.size == 0:
        return 0.0
    prefix = np.concatenate(([0.0], np.cumsum(right)))
    positions = np.searchsorted(right, left, side="left")
    lower = left * positions - prefix[positions]
    upper = (prefix[-1] - prefix[positions]) - left * (right.size - positions)
    return float(np.sum(lower + upper) / float(left.size * right.size))


def _prototype_cross_loss(
    y_scaled: np.ndarray,
    rows_by_proto: Mapping[int, np.ndarray],
    support: Mapping[int, int],
    adjacency: Mapping[tuple[int, int], float],
) -> float:
    values = np.asarray(y_scaled, dtype=float)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("edge experiment requires a univariate target matrix")
    total = float(sum(int(value) for value in support.values()))
    if total <= 0.0:
        return 0.0
    loss = 0.0
    for (source, target), weight in adjacency.items():
        if source not in rows_by_proto or target not in rows_by_proto:
            continue
        distance = _mean_cross_absolute(
            values[rows_by_proto[source], 0],
            values[rows_by_proto[target], 0],
        )
        loss += (float(support[source]) / total) * float(weight) * distance
    return float(loss)


def _edge_components(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    permutations: Sequence[np.ndarray],
) -> Dict[str, Any]:
    structure = strict_parent._fixed_evaluation_structure(estimator, X, y)
    X_prepared = estimator._prep_X(estimator._validate_X(X))
    y_scaled = np.asarray(structure["y_scaled"])
    row_sources, row_targets = _row_knn_edges(X_prepared, k=5)
    actual = {
        "row_knn5": _row_edge_loss(y_scaled, row_sources, row_targets),
        "prototype_cross": _prototype_cross_loss(
            y_scaled,
            structure["rows_by_proto"],
            structure["support"],
            structure["adjacency"],
        ),
    }
    null_samples = {edge_id: [] for edge_id in EDGE_IDS}
    for permutation in permutations:
        indices = np.asarray(permutation, dtype=np.int64)
        if indices.shape != (len(y_scaled),):
            raise ValueError("permutation length does not match held-out rows")
        permuted = y_scaled[indices]
        null_samples["row_knn5"].append(
            _row_edge_loss(permuted, row_sources, row_targets)
        )
        null_samples["prototype_cross"].append(
            _prototype_cross_loss(
                permuted,
                structure["rows_by_proto"],
                structure["support"],
                structure["adjacency"],
            )
        )
    return {
        "existing_actual_loss": float(structure["actual_loss"]),
        "actual_loss": actual,
        "null_loss_samples": null_samples,
        "row_edge_count": int(len(row_sources)),
        "prototype_edge_count": int(len(structure["adjacency"])),
        "occupied_evaluation_prototype_count": len(structure["support"]),
        "evaluation_support_total": int(sum(structure["support"].values())),
    }


def _aggregate_selector(folds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    edges: Dict[str, Any] = {}
    for edge_id in EDGE_IDS:
        rows = [row["edges"][edge_id] for row in folds]
        ratios = [row["loss_ratio"] for row in rows]
        edges[edge_id] = {
            "mean_actual_loss": float(np.mean([row["actual_loss"] for row in rows])),
            "mean_null_loss": float(np.mean([row["null_loss"] for row in rows])),
            "mean_loss_ratio": (
                float(np.mean(ratios))
                if all(value is not None for value in ratios)
                else None
            ),
            "score": float(np.mean([row["score"] for row in rows])),
            "mean_fold_null_standard_error": float(
                np.mean([row["null_standard_error"] for row in rows])
            ),
        }
    return {"edges": edges, "folds": list(folds)}


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
    old_folds = parent_selector["folds"]
    if len(old_folds) != 4:
        raise ValueError("parent selector does not contain four folds")
    folds = []
    started = time.perf_counter()
    for fold, (train, holdout) in enumerate(splitter.split(X)):
        fit_seed = null_parent._candidate_fit_seed(panel_seed, candidate_position, fold)
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
        bank_seed = null_parent._common_bank_seed(panel_seed, fold)
        bank = null_parent._permutation_bank(len(holdout), 32, bank_seed)
        components = _edge_components(
            estimator,
            transformed_holdout,
            y[holdout],
            bank,
        )
        old_fold = old_folds[fold]
        if int(old_fold["fit_seed"]) != fit_seed:
            raise ValueError("parent fit seed mismatch")
        if int(old_fold["common_bank_seed"]) != bank_seed:
            raise ValueError("parent common-bank seed mismatch")
        if components["existing_actual_loss"] != float(old_fold["actual_loss"]):
            raise RuntimeError("replayed existing actual loss does not match parent")
        edge_rows = {}
        for edge_id in EDGE_IDS:
            samples = components["null_loss_samples"][edge_id]
            null_loss = float(np.mean(samples))
            derived = null_parent._score_from_losses(
                components["actual_loss"][edge_id], null_loss
            )
            standard_deviation = float(np.std(samples, ddof=1))
            edge_rows[edge_id] = {
                "actual_loss": float(components["actual_loss"][edge_id]),
                "null_loss": null_loss,
                "null_loss_samples": samples,
                "loss_ratio": derived["loss_ratio"],
                "score": derived["score"],
                "null_standard_deviation": standard_deviation,
                "null_standard_error": float(standard_deviation / np.sqrt(32)),
            }
        folds.append(
            {
                "fold": int(fold),
                "fit_seed": int(fit_seed),
                "common_bank_seed": int(bank_seed),
                "existing_actual_loss": components["existing_actual_loss"],
                "row_edge_count": components["row_edge_count"],
                "prototype_edge_count": components["prototype_edge_count"],
                "occupied_evaluation_prototype_count": components[
                    "occupied_evaluation_prototype_count"
                ],
                "evaluation_support_total": components["evaluation_support_total"],
                "geometry_diagnostics": geometry_diagnostics,
                "edges": edge_rows,
            }
        )
    result = _aggregate_selector(folds)
    if float(np.mean([row["existing_actual_loss"] for row in folds])) != float(
        parent_selector["mean_actual_loss"]
    ):
        raise RuntimeError("replayed mean existing actual loss does not match parent")
    result["existing_mean_actual_loss"] = float(parent_selector["mean_actual_loss"])
    result["current_common32"] = dict(parent_selector["current_common32"])
    result["wall_seconds"] = float(time.perf_counter() - started)
    return result


def _parent_candidate(
    panel: Mapping[str, Any], candidate_id: str
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
    panel: Mapping[str, Any], method_id: str, estimand: str
) -> list[float]:
    rows = panel["candidates"]
    if estimand == "parent_bounded_oi":
        return [float(row["parent_candidate4"][method_id]["score"]) for row in rows]
    if estimand == "negative_existing_actual_loss":
        return [-float(row["selectors"][method_id]["existing_mean_actual_loss"]) for row in rows]
    if estimand == "current_common32_oi":
        return [
            float(row["selectors"][method_id]["current_common32"]["score"])
            for row in rows
        ]
    if estimand == "current_common32_ratio":
        values = [
            row["selectors"][method_id]["current_common32"]["mean_loss_ratio"]
            for row in rows
        ]
        if any(value is None for value in values):
            raise ValueError("current common32 ratio contains undefined values")
        return [-float(value) for value in values]
    for edge_id in EDGE_IDS:
        if estimand == f"{edge_id}_oi":
            return [
                float(row["selectors"][method_id]["edges"][edge_id]["score"])
                for row in rows
            ]
        if estimand == f"{edge_id}_ratio":
            values = [
                row["selectors"][method_id]["edges"][edge_id]["mean_loss_ratio"]
                for row in rows
            ]
            if any(value is None for value in values):
                raise ValueError(f"{edge_id} ratio contains undefined values")
            return [-float(value) for value in values]
        if estimand == f"negative_{edge_id}_actual_loss":
            return [
                -float(row["selectors"][method_id]["edges"][edge_id]["mean_actual_loss"])
                for row in rows
            ]
    raise ValueError(f"unknown estimand: {estimand!r}")


def _panel_metric(
    panel: Mapping[str, Any], method_id: str, estimand: str
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
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
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
                    "defined_spearman_panel_count": sum(
                        row["spearman"] is not None for row in family_rows
                    ),
                    "expected_spearman_panel_count": len(family_rows),
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
                "defined_spearman_panel_count": sum(
                    row["spearman"] is not None for row in selected
                ),
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


def _edge_agreement(
    protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]
) -> list[Dict[str, Any]]:
    result = []
    for panel in panels:
        envelope = [float(row["best_head_envelope_r2"]) for row in panel["candidates"]]
        for method_id in _method_specs(protocol):
            selectors = [row["selectors"][method_id] for row in panel["candidates"]]
            row_scores = [float(row["edges"]["row_knn5"]["score"]) for row in selectors]
            proto_scores = [
                float(row["edges"]["prototype_cross"]["score"]) for row in selectors
            ]
            result.append(
                {
                    "seed": int(panel["seed"]),
                    "family": panel["family"],
                    "method_id": method_id,
                    "row_prototype_score_spearman": _defined_spearman(
                        row_scores, proto_scores
                    ),
                    "row_score_envelope_spearman": _defined_spearman(
                        row_scores, envelope
                    ),
                    "prototype_score_envelope_spearman": _defined_spearman(
                        proto_scores, envelope
                    ),
                    "mean_absolute_score_difference": float(
                        np.mean(np.abs(np.asarray(row_scores) - np.asarray(proto_scores)))
                    ),
                }
            )
    return result


def _decision(
    protocol: Mapping[str, Any], metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    config = protocol["decision_rule"]
    results: Dict[str, Any] = {}
    passing = []
    for method_id, method in metrics.items():
        baseline = method["current_common32_oi"]
        results[method_id] = {}
        for edge_id in EDGE_IDS:
            candidate = method[f"{edge_id}_oi"]
            periodic_values = [
                (
                    candidate["by_family"][family]["mean_spearman"],
                    baseline["by_family"][family]["mean_spearman"],
                )
                for family in config["periodic_families"]
            ]
            periodic_gain = (
                float(np.mean([left - right for left, right in periodic_values]))
                if all(
                    left is not None and right is not None
                    for left, right in periodic_values
                )
                else None
            )
            checks = [
                {
                    "criterion": "all_panel_spearman_defined",
                    "observed": int(candidate["defined_spearman_panel_count"]),
                    "required": int(candidate["panel_count"]),
                    "pass": bool(
                        candidate["defined_spearman_panel_count"]
                        == candidate["panel_count"]
                    ),
                },
                {
                    "criterion": "mean_periodic_spearman_gain",
                    "observed": periodic_gain,
                    "minimum": float(config["minimum_mean_periodic_spearman_gain"]),
                    "pass": bool(
                        periodic_gain is not None
                        and periodic_gain >= config["minimum_mean_periodic_spearman_gain"]
                    ),
                },
                {
                    "criterion": "mean_regret_preservation",
                    "observed": float(candidate["mean_regret_r2"]),
                    "maximum": float(baseline["mean_regret_r2"])
                    + float(config["maximum_mean_regret_increase"]),
                    "pass": bool(
                        candidate["mean_regret_r2"]
                        <= baseline["mean_regret_r2"]
                        + config["maximum_mean_regret_increase"]
                    ),
                },
                {
                    "criterion": "minimum_overall_spearman",
                    "observed": candidate["mean_spearman"],
                    "minimum": float(config["minimum_overall_spearman"]),
                    "pass": bool(
                        candidate["mean_spearman"] is not None
                        and candidate["mean_spearman"] >= config["minimum_overall_spearman"]
                    ),
                },
                {
                    "criterion": "maximum_mean_regret",
                    "observed": float(candidate["mean_regret_r2"]),
                    "maximum": float(config["maximum_mean_regret"]),
                    "pass": bool(candidate["mean_regret_r2"] <= config["maximum_mean_regret"]),
                },
                {
                    "criterion": "minimum_exact_best_rate",
                    "observed": float(candidate["exact_best_rate"]),
                    "minimum": float(config["minimum_exact_best_rate"]),
                    "pass": bool(candidate["exact_best_rate"] >= config["minimum_exact_best_rate"]),
                },
                {
                    "criterion": "minimum_within_0_01_rate",
                    "observed": float(candidate["within_0_01_rate"]),
                    "minimum": float(config["minimum_within_0_01_rate"]),
                    "pass": bool(candidate["within_0_01_rate"] >= config["minimum_within_0_01_rate"]),
                },
            ]
            for family in config["preserve_families"]:
                observed = candidate["by_family"][family]["mean_spearman"]
                baseline_value = baseline["by_family"][family]["mean_spearman"]
                minimum = (
                    float(baseline_value)
                    - float(config["maximum_preservation_spearman_drop"])
                    if baseline_value is not None
                    else None
                )
                checks.append(
                    {
                        "criterion": f"preserve_{family}_spearman",
                        "observed": observed,
                        "minimum": minimum,
                        "pass": bool(
                            observed is not None
                            and minimum is not None
                            and observed >= minimum
                        ),
                    }
                )
            status = "pass" if all(row["pass"] for row in checks) else "fail"
            candidate_id = f"{method_id}:{edge_id}"
            results[method_id][edge_id] = {"status": status, "checks": checks}
            if status == "pass":
                passing.append(candidate_id)
    return {
        "status": "pass" if passing else "fail",
        "passing_candidates": passing,
        "methods": results,
    }


def summarize(
    protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    estimands = [
        "parent_bounded_oi",
        "negative_existing_actual_loss",
        "current_common32_ratio",
        "current_common32_oi",
        *[
            estimand
            for edge_id in EDGE_IDS
            for estimand in (
                f"negative_{edge_id}_actual_loss",
                f"{edge_id}_ratio",
                f"{edge_id}_oi",
            )
        ],
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
        "edge_agreement": _edge_agreement(protocol, panels),
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
        "strict_parent_runner": strict_parent.RUNNER_PATH,
        "null_parent_runner": null_parent.RUNNER_PATH,
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
                "edge_scores": {
                    method_id: {
                        edge_id: {
                            "mean_spearman": row[f"{edge_id}_oi"]["mean_spearman"],
                            "mean_regret_r2": row[f"{edge_id}_oi"]["mean_regret_r2"],
                            "sinusoid_spearman": row[f"{edge_id}_oi"]["by_family"][
                                "sinusoid"
                            ]["mean_spearman"],
                            "triangle_wave_spearman": row[f"{edge_id}_oi"]["by_family"][
                                "triangle_wave"
                            ]["mean_spearman"],
                        }
                        for edge_id in EDGE_IDS
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
