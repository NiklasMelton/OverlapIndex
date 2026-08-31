"""Frozen continuous-OI evaluation-null diagnostic."""

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
from experiments.continuous_prototype_capacity import run as capacity
from experiments.continuous_save_ranking import run as save


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_PARENT_RAW = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "continuous_prototype_capacity"
    / "raw_results.json"
)
COMMON_BANK_SEED = 2026083101
POLICY_COUNTS = {
    "candidate4": 4,
    "common4": 4,
    "common16": 16,
    "common32": 32,
}


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
        raise ValueError("frozen null-diagnostic geometry grid does not match code")
    if protocol["seeds"] != [1009, 1103, 1201]:
        raise ValueError("frozen seed set does not match code")
    if len(protocol["families"]) != 6:
        raise ValueError("frozen family set must contain six families")
    if protocol["null_policies"]["common_bank_seed"] != (
        "2026083101 + panel seed * 100 + fold"
    ):
        raise ValueError("frozen common-bank seed rule does not match code")
    observed_counts = {
        key: int(protocol["null_policies"][key]["permutations"])
        for key in POLICY_COUNTS
    }
    if observed_counts != POLICY_COUNTS:
        raise ValueError("frozen null-policy counts do not match code")


def _load_parent_raw(
    protocol: Mapping[str, Any],
    path: Path = DEFAULT_PARENT_RAW,
) -> Dict[str, Any]:
    resolved = path.resolve()
    expected = protocol["ancestry"]["parent_raw_results_sha256"]
    if common._sha256(resolved) != expected:
        raise ValueError("parent raw-results SHA-256 does not match protocol")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if raw.get("artifact_status") != "completed":
        raise ValueError("parent raw results are not completed")
    if common._sha256(capacity.PROTOCOL_PATH) != protocol["ancestry"][
        "parent_protocol_sha256"
    ]:
        raise ValueError("parent protocol bytes do not match frozen ancestry")
    if common._sha256(capacity.RUNNER_PATH) != protocol["ancestry"][
        "parent_runner_sha256"
    ]:
        raise ValueError("parent runner bytes do not match frozen ancestry")
    expected_panels = len(protocol["seeds"]) * len(protocol["families"])
    if len(raw.get("panels", [])) != expected_panels:
        raise ValueError("parent raw results do not contain the exact panel grid")
    return raw


def _parent_panel_map(raw: Mapping[str, Any]) -> Dict[tuple[int, str], Mapping[str, Any]]:
    result = {
        (int(panel["seed"]), str(panel["family"])): panel
        for panel in raw["panels"]
    }
    if len(result) != len(raw["panels"]):
        raise ValueError("parent raw results contain duplicate panels")
    return result


def _permutation_bank(size: int, count: int, seed: int) -> list[np.ndarray]:
    if size < 2 or count < 1:
        raise ValueError("permutation bank requires size >= 2 and count >= 1")
    rng = np.random.default_rng(int(seed))
    bank = [np.asarray(rng.permutation(size), dtype=np.int64) for _ in range(count)]
    expected = np.arange(size, dtype=np.int64)
    if any(not np.array_equal(np.sort(row), expected) for row in bank):
        raise RuntimeError("invalid permutation bank")
    return bank


def _score_from_losses(actual_loss: float, null_loss: float) -> Dict[str, Any]:
    eps = np.finfo(float).eps
    if null_loss <= eps:
        ratio = 0.0 if actual_loss <= eps else None
        score = 1.0 if actual_loss <= eps else 0.0
    else:
        ratio = float(actual_loss / null_loss)
        score = float(np.clip(1.0 - ratio, 0.0, 1.0))
    return {
        "null_loss": float(null_loss),
        "loss_ratio": ratio,
        "score": score,
    }


def _fixed_model_components(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    permutations: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Evaluate explicit held-out permutations without mutating fitted state."""
    X_arr = estimator._validate_X(X)
    y_arr = estimator._validate_Y(y)
    X_prepared = estimator._prep_X(X_arr)
    y_scaled = estimator._transform_targets(y_arr)
    actual_loss, _local, _support = estimator._fixed_model_loss(
        X_prepared,
        y_scaled,
    )
    null_samples = []
    for permutation in permutations:
        indices = np.asarray(permutation, dtype=np.int64)
        if indices.shape != (len(y_scaled),):
            raise ValueError("permutation length does not match held-out rows")
        null_loss, _null_local, _null_support = estimator._fixed_model_loss(
            X_prepared,
            y_scaled[indices],
        )
        null_samples.append(float(null_loss))
    return {
        "actual_loss": float(actual_loss),
        "null_loss_samples": null_samples,
    }


def _candidate_fit_seed(panel_seed: int, position: int, fold: int) -> int:
    return int(panel_seed) * 1000 + int(position) * 10 + int(fold)


def _common_bank_seed(panel_seed: int, fold: int) -> int:
    return COMMON_BANK_SEED + int(panel_seed) * 100 + int(fold)


def _aggregate_selector(folds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    actual = float(np.mean([row["actual_loss"] for row in folds]))
    policies: Dict[str, Any] = {}
    for policy in POLICY_COUNTS:
        rows = [row["policies"][policy] for row in folds]
        ratios = [row["loss_ratio"] for row in rows]
        policies[policy] = {
            "mean_null_loss": float(np.mean([row["null_loss"] for row in rows])),
            "mean_loss_ratio": (
                float(np.mean(ratios))
                if all(value is not None for value in ratios)
                else None
            ),
            "score": float(np.mean([row["score"] for row in rows])),
            "mean_fold_null_standard_deviation": float(
                np.mean([row["null_standard_deviation"] for row in rows])
            ),
            "mean_fold_null_standard_error": float(
                np.mean([row["null_standard_error"] for row in rows])
            ),
        }
    return {
        "mean_actual_loss": actual,
        "policies": policies,
        "folds": list(folds),
    }


def evaluate_selector(
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
        fit_seed = _candidate_fit_seed(panel_seed, candidate_position, fold)
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
        common_bank = _permutation_bank(
            len(holdout),
            32,
            _common_bank_seed(panel_seed, fold),
        )
        candidate_bank = _permutation_bank(len(holdout), 4, fit_seed)
        candidate_components = _fixed_model_components(
            estimator,
            transformed_holdout,
            y[holdout],
            candidate_bank,
        )
        common_components = _fixed_model_components(
            estimator,
            transformed_holdout,
            y[holdout],
            common_bank,
        )
        if candidate_components["actual_loss"] != common_components["actual_loss"]:
            raise RuntimeError("actual loss changed across null policies")
        actual_loss = float(common_components["actual_loss"])
        policies = {}
        for policy, count in POLICY_COUNTS.items():
            samples = (
                candidate_components["null_loss_samples"]
                if policy == "candidate4"
                else common_components["null_loss_samples"][:count]
            )
            null_loss = float(np.mean(samples))
            derived = _score_from_losses(actual_loss, null_loss)
            standard_deviation = float(np.std(samples, ddof=1)) if count > 1 else 0.0
            policies[policy] = {
                **derived,
                "permutation_count": int(count),
                "null_loss_samples": list(samples),
                "null_standard_deviation": standard_deviation,
                "null_standard_error": float(standard_deviation / np.sqrt(count)),
            }
        folds.append(
            {
                "fold": int(fold),
                "fit_seed": int(fit_seed),
                "common_bank_seed": int(_common_bank_seed(panel_seed, fold)),
                "actual_loss": actual_loss,
                "prototype_count": int(estimator._model.n_clusters_total),
                "geometry_diagnostics": geometry_diagnostics,
                "policies": policies,
            }
        )
    result = _aggregate_selector(folds)
    result["wall_seconds"] = float(time.perf_counter() - started)
    return result


def _copy_parent_reference(
    parent_panel: Mapping[str, Any],
    candidate_id: str,
) -> Mapping[str, Any]:
    matches = [
        row for row in parent_panel["candidates"]
        if row["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"parent candidate identity mismatch: {candidate_id}")
    return matches[0]


def run_panel(
    protocol: Mapping[str, Any],
    seed: int,
    family: str,
    parent_panel: Mapping[str, Any],
) -> Dict[str, Any]:
    panel = capacity.generate_panel(protocol, seed, family)
    if int(parent_panel["seed"]) != int(seed) or parent_panel["family"] != family:
        raise ValueError("parent panel identity mismatch")
    methods = _method_specs(protocol)
    candidates = []
    for position, candidate in enumerate(panel["candidates"]):
        parent = _copy_parent_reference(parent_panel, candidate["candidate_id"])
        for key in ("quality", "nuisance_scale", "effective_nuisance_scale"):
            if float(candidate[key]) != float(parent[key]):
                raise ValueError(f"parent candidate metadata mismatch for {key}")
        selectors = {
            method_id: evaluate_selector(
                spec,
                candidate["X_train"],
                candidate["primitive_train"],
                panel["y_train"],
                seed,
                position,
            )
            for method_id, spec in methods.items()
        }
        parent_selectors = {
            method_id: {
                key: parent["selectors"][method_id][key]
                for key in (
                    "score",
                    "mean_actual_loss",
                    "mean_null_loss",
                    "mean_loss_ratio",
                )
            }
            for method_id in methods
        }
        candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "quality": float(candidate["quality"]),
                "nuisance_scale": float(candidate["nuisance_scale"]),
                "effective_nuisance_scale": float(
                    candidate["effective_nuisance_scale"]
                ),
                "best_head_envelope_r2": float(parent["best_head_envelope_r2"]),
                "best_head_ids": list(parent["best_head_ids"]),
                "heads": parent["heads"],
                "parent_candidate4": parent_selectors,
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
    policy, kind = estimand.rsplit("_", 1)
    if policy not in POLICY_COUNTS or kind not in {"ratio", "oi"}:
        raise ValueError(f"unknown estimand: {estimand!r}")
    if kind == "ratio":
        values = [
            row["selectors"][method_id]["policies"][policy]["mean_loss_ratio"]
            for row in rows
        ]
        if any(value is None for value in values):
            raise ValueError("loss-ratio ranking contains undefined values")
        return [-float(value) for value in values]
    return [
        float(row["selectors"][method_id]["policies"][policy]["score"])
        for row in rows
    ]


def _panel_metric(
    panel: Mapping[str, Any],
    method_id: str,
    estimand: str,
) -> Dict[str, Any]:
    candidates = panel["candidates"]
    envelope = np.asarray(
        [row["best_head_envelope_r2"] for row in candidates],
        dtype=float,
    )
    selector = np.asarray(
        _selector_values(panel, method_id, estimand),
        dtype=float,
    )
    selected = np.flatnonzero(
        np.isclose(selector, np.max(selector), atol=1.0e-12, rtol=0.0)
    )
    best_reference = float(np.max(envelope))
    regret = float(best_reference - np.mean(envelope[selected]))
    return {
        "seed": int(panel["seed"]),
        "family": panel["family"],
        "method_id": method_id,
        "estimand": estimand,
        "spearman": _defined_spearman(selector.tolist(), envelope.tolist()),
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
        estimands = sorted({row["estimand"] for row in rows if row["method_id"] == method_id})
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


def _null_diagnostic_rows(
    protocol: Mapping[str, Any],
    panels: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows = []
    for panel in panels:
        envelope = [float(row["best_head_envelope_r2"]) for row in panel["candidates"]]
        for method_id in _method_specs(protocol):
            actual = [
                float(row["selectors"][method_id]["mean_actual_loss"])
                for row in panel["candidates"]
            ]
            common32 = [
                float(row["selectors"][method_id]["policies"]["common32"]["mean_null_loss"])
                for row in panel["candidates"]
            ]
            for policy in POLICY_COUNTS:
                selectors = [row["selectors"][method_id] for row in panel["candidates"]]
                null = [float(row["policies"][policy]["mean_null_loss"]) for row in selectors]
                scores = [float(row["policies"][policy]["score"]) for row in selectors]
                rows.append(
                    {
                        "seed": int(panel["seed"]),
                        "family": panel["family"],
                        "method_id": method_id,
                        "policy": policy,
                        "null_actual_spearman": _defined_spearman(null, actual),
                        "null_envelope_spearman": _defined_spearman(null, envelope),
                        "null_common32_spearman": _defined_spearman(null, common32),
                        "mean_absolute_null_difference_from_common32": float(
                            np.mean(np.abs(np.asarray(null) - np.asarray(common32)))
                        ),
                        "mean_absolute_score_difference_from_common32": float(
                            np.mean(
                                np.abs(
                                    np.asarray(scores)
                                    - np.asarray(
                                        [
                                            float(row["policies"]["common32"]["score"])
                                            for row in selectors
                                        ]
                                    )
                                )
                            )
                        ),
                        "mean_fold_null_standard_deviation": float(
                            np.mean(
                                [
                                    row["policies"][policy]["mean_fold_null_standard_deviation"]
                                    for row in selectors
                                ]
                            )
                        ),
                        "mean_fold_null_standard_error": float(
                            np.mean(
                                [
                                    row["policies"][policy]["mean_fold_null_standard_error"]
                                    for row in selectors
                                ]
                            )
                        ),
                    }
                )
    return rows


def _decision(
    protocol: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    config = protocol["decision_rule"]
    results = {}
    for method_id, method in metrics.items():
        baseline = method["candidate4_oi"]
        candidate = method["common32_oi"]
        actual = method["negative_actual_loss"]
        periodic_gains = []
        for family in config["periodic_families"]:
            left = baseline["by_family"][family]["mean_spearman"]
            right = candidate["by_family"][family]["mean_spearman"]
            periodic_gains.append(float(right - left))
        periodic_gain = float(np.mean(periodic_gains))
        checks = [
            {
                "criterion": "mean_periodic_spearman_gain",
                "observed": periodic_gain,
                "minimum": float(config["variance_repair_minimum_periodic_spearman_gain"]),
                "pass": bool(
                    periodic_gain
                    >= float(config["variance_repair_minimum_periodic_spearman_gain"])
                ),
            },
            {
                "criterion": "mean_regret_preservation",
                "observed": float(candidate["mean_regret_r2"]),
                "maximum": float(baseline["mean_regret_r2"])
                + float(config["maximum_regret_increase"]),
                "pass": bool(
                    candidate["mean_regret_r2"]
                    <= baseline["mean_regret_r2"]
                    + float(config["maximum_regret_increase"])
                ),
            },
        ]
        for family in config["preserve_families"]:
            observed = float(candidate["by_family"][family]["mean_spearman"])
            minimum = float(baseline["by_family"][family]["mean_spearman"]) - float(
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
        repair = all(row["pass"] for row in checks)
        mismatches = {}
        for family in config["periodic_families"]:
            actual_rho = actual["by_family"][family]["mean_spearman"]
            normalized_rho = candidate["by_family"][family]["mean_spearman"]
            difference = float(actual_rho - normalized_rho)
            mismatches[family] = {
                "negative_actual_loss_spearman": actual_rho,
                "common32_bounded_oi_spearman": normalized_rho,
                "difference": difference,
                "meets_threshold": bool(
                    difference
                    >= float(
                        config[
                            "estimand_mismatch_minimum_actual_over_common32_periodic_spearman"
                        ]
                    )
                ),
            }
        results[method_id] = {
            "variance_repair_status": "pass" if repair else "fail",
            "variance_repair_checks": checks,
            "periodic_estimand_mismatch": mismatches,
        }
    repaired = [
        method_id for method_id, row in results.items()
        if row["variance_repair_status"] == "pass"
    ]
    mismatch = [
        method_id for method_id, row in results.items()
        if any(
            family["meets_threshold"]
            for family in row["periodic_estimand_mismatch"].values()
        )
    ]
    if repaired:
        conclusion = "null_variance_repair_supported"
    elif mismatch:
        conclusion = "null_estimand_mismatch_supported"
    else:
        conclusion = "inconclusive"
    return {
        "conclusion": conclusion,
        "variance_repaired_methods": repaired,
        "estimand_mismatch_methods": mismatch,
        "methods": results,
    }


def summarize(
    protocol: Mapping[str, Any],
    panels: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    estimands = [
        "parent_bounded_oi",
        "negative_actual_loss",
        *[
            f"{policy}_{kind}"
            for policy in POLICY_COUNTS
            for kind in ("ratio", "oi")
        ],
    ]
    panel_metrics = [
        _panel_metric(panel, method_id, estimand)
        for panel in panels
        for method_id in _method_specs(protocol)
        for estimand in estimands
    ]
    metrics = _aggregate_metrics(protocol, panel_metrics)
    null_rows = _null_diagnostic_rows(protocol, panels)
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
        "null_diagnostics": null_rows,
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
                "common32": {
                    method_id: {
                        "mean_spearman": row["common32_oi"]["mean_spearman"],
                        "mean_regret_r2": row["common32_oi"]["mean_regret_r2"],
                        "triangle_wave_spearman": row["common32_oi"]["by_family"][
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
