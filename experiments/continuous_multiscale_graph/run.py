"""Frozen multiscale graph-smoothness experiment for continuous ranking."""

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

from experiments.continuous_edge_target_continuity import run as edge_parent
from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_multiview_ranking import run as multiview
from experiments.continuous_null_diagnostic import run as null_parent
from experiments.continuous_prototype_capacity import run as capacity
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_save_ranking import run as save


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_MOTIVATION_RAW = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "continuous_edge_target_continuity"
    / "raw_results.json"
)
FORMULATION_IDS = ("uniform_multiscale", "self_tuning_multiscale")
SCALE_IDS = (5, 15, 45)


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
        "rff64_mean8_k4": ("rff_mean", None),
        "save_rffmean_k4": ("multiview", ("save", "rff_mean")),
    }
    observed = {
        method_id: (
            str(row["geometry"]),
            tuple(row["blocks"]) if "blocks" in row else None,
        )
        for method_id, row in _method_specs(protocol).items()
    }
    if observed != expected:
        raise ValueError("frozen multiscale geometry grid does not match code")
    if tuple(protocol["graph"]["formulations"]) != FORMULATION_IDS:
        raise ValueError("frozen graph formulations do not match code")
    if tuple(int(value) for value in protocol["graph"]["scales"]) != SCALE_IDS:
        raise ValueError("frozen graph scales do not match code")
    weights = np.asarray(protocol["graph"]["scale_weights"], dtype=float)
    if weights.shape != (3,) or not np.all(weights > 0.0) or not np.isclose(
        np.sum(weights), 1.0, atol=1.0e-15, rtol=0.0
    ):
        raise ValueError("frozen scale weights must be three positive values summing to one")
    if int(protocol["graph"]["self_tuning_local_scale_k"]) != 15:
        raise ValueError("frozen self-tuning local scale must be 15")
    if protocol["seeds"] != [1301, 1409, 1511]:
        raise ValueError("frozen new-seed panel set does not match code")
    if len(protocol["families"]) != 6:
        raise ValueError("frozen family set must contain six families")
    if int(protocol["shared_execution"]["permutations"]) != 32:
        raise ValueError("multiscale score requires exactly 32 permutations")
    if tuple(protocol["reference_heads"]["ids"]) != tuple(
        residual._head_estimators()
    ):
        raise ValueError("frozen reference-head order does not match code")


def _validate_motivation_ancestry(
    protocol: Mapping[str, Any],
    raw_path: Path = DEFAULT_MOTIVATION_RAW,
) -> None:
    ancestry = protocol["ancestry"]
    if common._sha256(raw_path.resolve()) != ancestry["motivation_raw_results_sha256"]:
        raise ValueError("motivation raw-results SHA-256 does not match protocol")
    if common._sha256(edge_parent.PROTOCOL_PATH) != ancestry[
        "motivation_protocol_sha256"
    ]:
        raise ValueError("motivation protocol bytes do not match frozen ancestry")
    if common._sha256(edge_parent.RUNNER_PATH) != ancestry[
        "motivation_runner_sha256"
    ]:
        raise ValueError("motivation runner bytes do not match frozen ancestry")


def _ordered_neighbor_graph(
    X: np.ndarray,
    *,
    maximum_k: int = 45,
    local_scale_k: int = 15,
) -> Dict[str, np.ndarray]:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("multiscale graph requires a finite two-dimensional matrix")
    n_rows = int(values.shape[0])
    if type(maximum_k) is not int or maximum_k < 1 or maximum_k >= n_rows:
        raise ValueError("maximum_k must be an integer in [1, n_rows)")
    if type(local_scale_k) is not int or not 1 <= local_scale_k <= maximum_k:
        raise ValueError("local_scale_k must be in [1, maximum_k]")
    norms = np.sum(values * values, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * values @ values.T
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :maximum_k]
    squared = np.take_along_axis(distances, neighbors, axis=1)
    if np.any(neighbors == np.arange(n_rows)[:, None]):
        raise RuntimeError("multiscale graph contains a self edge")
    local_scale = np.sqrt(squared[:, local_scale_k - 1])
    positive = local_scale[local_scale > 0.0]
    reference = float(np.median(positive)) if positive.size else 1.0
    floor = max(reference * 1.0e-12, np.finfo(float).eps)
    local_scale = np.maximum(local_scale, floor)
    return {
        "neighbors": np.asarray(neighbors, dtype=np.int64),
        "squared_distances": np.asarray(squared, dtype=float),
        "local_scale": np.asarray(local_scale, dtype=float),
    }


def _scale_edges(
    graph: Mapping[str, np.ndarray],
    k: int,
    formulation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbors = np.asarray(graph["neighbors"], dtype=np.int64)
    squared = np.asarray(graph["squared_distances"], dtype=float)
    local_scale = np.asarray(graph["local_scale"], dtype=float)
    if type(k) is not int or not 1 <= k <= neighbors.shape[1]:
        raise ValueError("graph scale is outside the fitted neighbor range")
    sources = np.repeat(np.arange(neighbors.shape[0], dtype=np.int64), k)
    targets = neighbors[:, :k].reshape(-1)
    if formulation == "uniform_multiscale":
        weights = np.ones(len(sources), dtype=float)
    elif formulation == "self_tuning_multiscale":
        denominator = local_scale[sources] * local_scale[targets]
        weights = np.exp(-squared[:, :k].reshape(-1) / denominator)
    else:
        raise ValueError(f"unknown graph formulation: {formulation!r}")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise RuntimeError("graph weights must be finite and positive")
    return sources, targets, weights


def _weighted_edge_loss(
    y_scaled: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> float:
    values = np.asarray(y_scaled, dtype=float)
    source_ids = np.asarray(sources, dtype=np.int64)
    target_ids = np.asarray(targets, dtype=np.int64)
    edge_weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("multiscale graph requires a univariate target matrix")
    if (
        source_ids.shape != target_ids.shape
        or source_ids.shape != edge_weights.shape
        or source_ids.size == 0
    ):
        raise ValueError("edge arrays and weights must be aligned and nonempty")
    differences = np.abs(values[source_ids, 0] - values[target_ids, 0])
    return float(np.average(differences, weights=edge_weights))


def _graph_fold_components(
    X: np.ndarray,
    y_scaled: np.ndarray,
    permutations: Sequence[np.ndarray],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    graph = _ordered_neighbor_graph(
        X,
        maximum_k=max(SCALE_IDS),
        local_scale_k=int(protocol["graph"]["self_tuning_local_scale_k"]),
    )
    scale_weights = np.asarray(protocol["graph"]["scale_weights"], dtype=float)
    formulations: Dict[str, Any] = {}
    for formulation in FORMULATION_IDS:
        scales: Dict[str, Any] = {}
        for k in SCALE_IDS:
            sources, targets, weights = _scale_edges(graph, k, formulation)
            actual = _weighted_edge_loss(y_scaled, sources, targets, weights)
            samples = []
            for permutation in permutations:
                indices = np.asarray(permutation, dtype=np.int64)
                if indices.shape != (len(y_scaled),):
                    raise ValueError("permutation length does not match held-out rows")
                samples.append(
                    _weighted_edge_loss(
                        np.asarray(y_scaled)[indices], sources, targets, weights
                    )
                )
            null_loss = float(np.mean(samples))
            derived = null_parent._score_from_losses(actual, null_loss)
            standard_deviation = float(np.std(samples, ddof=1))
            scales[str(k)] = {
                "edge_count": int(len(sources)),
                "weight_sum": float(np.sum(weights)),
                "actual_loss": float(actual),
                "null_loss": null_loss,
                "null_loss_samples": samples,
                "loss_ratio": derived["loss_ratio"],
                "score": float(derived["score"]),
                "null_standard_deviation": standard_deviation,
                "null_standard_error": float(standard_deviation / np.sqrt(32.0)),
            }
        ratios = [scales[str(k)]["loss_ratio"] for k in SCALE_IDS]
        if any(value is None for value in ratios):
            integrated_ratio = None
        else:
            integrated_ratio = float(np.dot(scale_weights, ratios))
        formulations[formulation] = {
            "score": float(
                np.dot(scale_weights, [scales[str(k)]["score"] for k in SCALE_IDS])
            ),
            "mean_loss_ratio": integrated_ratio,
            "scales": scales,
        }
    return {
        "formulations": formulations,
        "row_count": int(len(y_scaled)),
        "maximum_neighbor_count": int(graph["neighbors"].shape[1]),
        "local_scale_minimum": float(np.min(graph["local_scale"])),
        "local_scale_median": float(np.median(graph["local_scale"])),
        "local_scale_maximum": float(np.max(graph["local_scale"])),
    }


def _aggregate_selector(folds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    current_rows = [row["current_common32"] for row in folds]
    current_ratios = [row["loss_ratio"] for row in current_rows]
    result: Dict[str, Any] = {
        "current_common32": {
            "score": float(np.mean([row["score"] for row in current_rows])),
            "mean_actual_loss": float(
                np.mean([row["actual_loss"] for row in current_rows])
            ),
            "mean_null_loss": float(
                np.mean([row["null_loss"] for row in current_rows])
            ),
            "mean_loss_ratio": (
                float(np.mean(current_ratios))
                if all(value is not None for value in current_ratios)
                else None
            ),
        },
        "formulations": {},
        "folds": list(folds),
    }
    for formulation in FORMULATION_IDS:
        rows = [row["formulations"][formulation] for row in folds]
        ratios = [row["mean_loss_ratio"] for row in rows]
        scale_summary = {}
        for k in SCALE_IDS:
            scale_rows = [row["scales"][str(k)] for row in rows]
            scale_ratios = [row["loss_ratio"] for row in scale_rows]
            scale_summary[str(k)] = {
                "score": float(np.mean([row["score"] for row in scale_rows])),
                "mean_actual_loss": float(
                    np.mean([row["actual_loss"] for row in scale_rows])
                ),
                "mean_null_loss": float(
                    np.mean([row["null_loss"] for row in scale_rows])
                ),
                "mean_loss_ratio": (
                    float(np.mean(scale_ratios))
                    if all(value is not None for value in scale_ratios)
                    else None
                ),
            }
        result["formulations"][formulation] = {
            "score": float(np.mean([row["score"] for row in rows])),
            "mean_loss_ratio": (
                float(np.mean(ratios))
                if all(value is not None for value in ratios)
                else None
            ),
            "scales": scale_summary,
        }
    return result


def evaluate_selector(
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    X: np.ndarray,
    primitive: np.ndarray,
    y: np.ndarray,
    panel_seed: int,
    candidate_position: int,
) -> Dict[str, Any]:
    splitter = KFold(n_splits=4, shuffle=True, random_state=int(panel_seed))
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
        common_components = null_parent._fixed_model_components(
            estimator, transformed_holdout, y[holdout], bank
        )
        current_null = float(np.mean(common_components["null_loss_samples"]))
        current = null_parent._score_from_losses(
            float(common_components["actual_loss"]), current_null
        )
        X_prepared = estimator._prep_X(estimator._validate_X(transformed_holdout))
        y_scaled = estimator._transform_targets(estimator._validate_Y(y[holdout]))
        graph = _graph_fold_components(X_prepared, y_scaled, bank, protocol)
        folds.append(
            {
                "fold": int(fold),
                "fit_seed": int(fit_seed),
                "common_bank_seed": int(bank_seed),
                "geometry_diagnostics": geometry_diagnostics,
                "prototype_count": int(estimator._model.n_clusters_total),
                "row_count": graph["row_count"],
                "maximum_neighbor_count": graph["maximum_neighbor_count"],
                "local_scale_minimum": graph["local_scale_minimum"],
                "local_scale_median": graph["local_scale_median"],
                "local_scale_maximum": graph["local_scale_maximum"],
                "current_common32": {
                    "actual_loss": float(common_components["actual_loss"]),
                    "null_loss": current_null,
                    "null_loss_samples": common_components["null_loss_samples"],
                    "loss_ratio": current["loss_ratio"],
                    "score": float(current["score"]),
                },
                "formulations": graph["formulations"],
            }
        )
    result = _aggregate_selector(folds)
    result["wall_seconds"] = float(time.perf_counter() - started)
    return result


def run_panel(protocol: Mapping[str, Any], seed: int, family: str) -> Dict[str, Any]:
    generated = capacity.generate_panel(protocol, seed, family)
    methods = _method_specs(protocol)
    candidates = []
    for position, candidate in enumerate(generated["candidates"]):
        reference_seed = int(seed) * 100 + int(position)
        heads = residual.evaluate_heads(
            candidate["X_train"],
            generated["y_train"],
            candidate["X_test"],
            generated["y_test"],
            reference_seed,
        )
        head_scores = {
            head_id: float(row["test_r2"]) for head_id, row in heads.items()
        }
        envelope = max(head_scores.values())
        selectors = {
            method_id: evaluate_selector(
                protocol,
                spec,
                candidate["X_train"],
                candidate["primitive_train"],
                generated["y_train"],
                seed,
                position,
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
                "heads": heads,
                "best_head_envelope_r2": float(envelope),
                "best_head_ids": sorted(
                    head_id
                    for head_id, value in head_scores.items()
                    if np.isclose(value, envelope, atol=1.0e-12, rtol=0.0)
                ),
                "selectors": selectors,
            }
        )
    return {
        "seed": int(seed),
        "family": str(family),
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
    candidates = panel["candidates"]
    if estimand == "linear_probe":
        return [float(row["heads"]["linear_ridge"]["test_r2"]) for row in candidates]
    if estimand == "current_common32_oi":
        return [
            float(row["selectors"][method_id]["current_common32"]["score"])
            for row in candidates
        ]
    for formulation in FORMULATION_IDS:
        if estimand == f"{formulation}_oi":
            return [
                float(row["selectors"][method_id]["formulations"][formulation]["score"])
                for row in candidates
            ]
        if estimand == f"{formulation}_ratio":
            ratios = [
                row["selectors"][method_id]["formulations"][formulation][
                    "mean_loss_ratio"
                ]
                for row in candidates
            ]
            if any(value is None for value in ratios):
                raise ValueError(f"{formulation} contains an undefined ratio")
            return [-float(value) for value in ratios]
        for k in SCALE_IDS:
            if estimand == f"{formulation}_k{k}_oi":
                return [
                    float(
                        row["selectors"][method_id]["formulations"][formulation][
                            "scales"
                        ][str(k)]["score"]
                    )
                    for row in candidates
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
    regret = float(np.max(envelope) - np.mean(envelope[selected]))
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
    selected = [float(value) for value in values if value is not None]
    return float(np.mean(selected)) if selected else None


def _aggregate_metrics(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    identities = sorted({(row["method_id"], row["estimand"]) for row in rows})
    for method_id, estimand in identities:
        selected = [
            row
            for row in rows
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
        result.setdefault(method_id, {})[estimand] = {
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


def _decision(
    protocol: Mapping[str, Any], metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    config = protocol["decision_rule"]
    linear_probe = metrics["linear_probe"]["linear_probe"]
    passing = []
    methods: Dict[str, Any] = {}
    for method_id in _method_specs(protocol):
        baseline = metrics[method_id]["current_common32_oi"]
        methods[method_id] = {}
        for formulation in FORMULATION_IDS:
            candidate = metrics[method_id][f"{formulation}_oi"]
            periodic_pairs = [
                (
                    candidate["by_family"][family]["mean_spearman"],
                    baseline["by_family"][family]["mean_spearman"],
                )
                for family in config["periodic_families"]
            ]
            periodic_gain = (
                float(np.mean([left - right for left, right in periodic_pairs]))
                if all(left is not None and right is not None for left, right in periodic_pairs)
                else None
            )
            checks = [
                {
                    "criterion": "all_panel_spearman_defined",
                    "observed": int(candidate["defined_spearman_panel_count"]),
                    "required": int(candidate["panel_count"]),
                    "pass": bool(
                        candidate["defined_spearman_panel_count"] == candidate["panel_count"]
                    ),
                },
                {
                    "criterion": "minimum_mean_periodic_spearman_gain",
                    "observed": periodic_gain,
                    "minimum": float(
                        config["minimum_mean_periodic_spearman_gain_over_matched_current_oi"]
                    ),
                    "pass": bool(
                        periodic_gain is not None
                        and periodic_gain
                        >= config[
                            "minimum_mean_periodic_spearman_gain_over_matched_current_oi"
                        ]
                    ),
                },
                {
                    "criterion": "mean_regret_preservation_vs_current_oi",
                    "observed": float(candidate["mean_regret_r2"]),
                    "maximum": float(baseline["mean_regret_r2"])
                    + float(config["maximum_mean_regret_increase_over_matched_current_oi"]),
                    "pass": bool(
                        candidate["mean_regret_r2"]
                        <= baseline["mean_regret_r2"]
                        + config["maximum_mean_regret_increase_over_matched_current_oi"]
                    ),
                },
                {
                    "criterion": "mean_regret_vs_linear_probe",
                    "observed": float(candidate["mean_regret_r2"]),
                    "maximum": float(linear_probe["mean_regret_r2"])
                    + float(config["maximum_mean_regret_increase_over_linear_probe"]),
                    "pass": bool(
                        candidate["mean_regret_r2"]
                        <= linear_probe["mean_regret_r2"]
                        + config["maximum_mean_regret_increase_over_linear_probe"]
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
                    - float(config["maximum_family_spearman_drop_from_matched_current_oi"])
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
            candidate_id = f"{method_id}:{formulation}"
            methods[method_id][formulation] = {"status": status, "checks": checks}
            if status == "pass":
                passing.append(candidate_id)
    return {
        "status": "pass" if passing else "fail",
        "passing_candidates": passing,
        "methods": methods,
    }


def summarize(
    protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    graph_estimands = [
        estimand
        for formulation in FORMULATION_IDS
        for estimand in (
            f"{formulation}_ratio",
            f"{formulation}_oi",
            *[f"{formulation}_k{k}_oi" for k in SCALE_IDS],
        )
    ]
    rows = [
        _panel_metric(panel, method_id, estimand)
        for panel in panels
        for method_id in _method_specs(protocol)
        for estimand in ("current_common32_oi", *graph_estimands)
    ]
    rows.extend(
        _panel_metric(panel, "linear_probe", "linear_probe") for panel in panels
    )
    metrics = _aggregate_metrics(protocol, rows)
    return {
        "schema_version": 1,
        "panel_count": len(panels),
        "candidate_panel_rows": sum(int(panel["candidate_count"]) for panel in panels),
        "selector_rows": sum(
            int(panel["candidate_count"]) * len(_method_specs(protocol))
            for panel in panels
        ),
        "panel_metrics": rows,
        "metrics": metrics,
        "decision": _decision(protocol, metrics),
    }


def run(
    output: Path,
    motivation_raw_path: Path = DEFAULT_MOTIVATION_RAW,
) -> Dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    _validate_motivation_ancestry(protocol, motivation_raw_path)
    source_paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "motivation_raw_results": motivation_raw_path.resolve(),
        "edge_parent_runner": edge_parent.RUNNER_PATH,
        "null_parent_runner": null_parent.RUNNER_PATH,
        "capacity_runner": capacity.RUNNER_PATH,
        "multiview_runner": multiview.RUNNER_PATH,
        "save_runner": save.RUNNER_PATH,
        "residual_runner": residual.RUNNER_PATH,
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
            path = checkpoints / f"seed_{seed}__{family}.json"
            if path.exists():
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                if checkpoint.get("source_hashes") != source_hashes:
                    raise ValueError(f"stale source checkpoint: {path}")
                panel = checkpoint["panel"]
            else:
                started = time.perf_counter()
                panel = run_panel(protocol, int(seed), str(family))
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
    parser.add_argument("--motivation-raw", type=Path, default=DEFAULT_MOTIVATION_RAW)
    args = parser.parse_args()
    summary = run(args.output, args.motivation_raw)
    print(json.dumps({"decision": summary["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
