"""Frozen characteristic-kernel alignment experiment."""

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
from experiments.continuous_multiscale_graph import run as motivation
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
    / "continuous_multiscale_graph"
    / "raw_results.json"
)
BANDWIDTH_MULTIPLIERS = (0.25, 1.0, 4.0)
ELIGIBLE_ESTIMAND = "null_adjusted_multibandwidth_cka"


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
        raise ValueError("frozen kernel geometry grid does not match code")
    if tuple(protocol["seeds"]) != (1601, 1709, 1811):
        raise ValueError("frozen new-seed set does not match code")
    if len(protocol["families"]) != 6:
        raise ValueError("frozen family set must contain six families")
    if tuple(protocol["kernel_alignment"]["bandwidth_multipliers"]) != (
        BANDWIDTH_MULTIPLIERS
    ):
        raise ValueError("frozen kernel bandwidths do not match code")
    if protocol["kernel_alignment"]["eligible_estimand"] != ELIGIBLE_ESTIMAND:
        raise ValueError("frozen eligible kernel estimand does not match code")
    if int(protocol["shared_execution"]["permutations"]) != 32:
        raise ValueError("kernel alignment requires exactly 32 permutations")
    if tuple(protocol["reference_heads"]["ids"]) != tuple(
        residual._head_estimators()
    ):
        raise ValueError("frozen reference-head order does not match code")


def _validate_motivation_ancestry(
    protocol: Mapping[str, Any], raw_path: Path = DEFAULT_MOTIVATION_RAW
) -> None:
    ancestry = protocol["ancestry"]
    if common._sha256(raw_path.resolve()) != ancestry["motivation_raw_results_sha256"]:
        raise ValueError("motivation raw-results SHA-256 does not match protocol")
    if common._sha256(motivation.PROTOCOL_PATH) != ancestry[
        "motivation_protocol_sha256"
    ]:
        raise ValueError("motivation protocol bytes do not match frozen ancestry")
    if common._sha256(motivation.RUNNER_PATH) != ancestry[
        "motivation_runner_sha256"
    ]:
        raise ValueError("motivation runner bytes do not match frozen ancestry")


def _squared_distances(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("kernel values must be a finite two-dimensional matrix")
    norms = np.sum(matrix * matrix, axis=1)
    result = norms[:, None] + norms[None, :] - 2.0 * matrix @ matrix.T
    return np.maximum(result, 0.0)


def _median_positive_squared_distance(values: np.ndarray) -> float:
    distances = _squared_distances(values)
    upper = distances[np.triu_indices(len(distances), k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return float(np.finfo(float).eps)
    return max(float(np.median(positive)), float(np.finfo(float).eps))


def _rbf_grams(
    values: np.ndarray,
    base_squared_distance: float,
) -> Dict[str, np.ndarray]:
    if not np.isfinite(base_squared_distance) or base_squared_distance <= 0.0:
        raise ValueError("RBF base squared distance must be finite and positive")
    distances = _squared_distances(values)
    return {
        str(multiplier): np.exp(
            -distances / (2.0 * float(multiplier) * float(base_squared_distance))
        )
        for multiplier in BANDWIDTH_MULTIPLIERS
    }


def _center_gram(gram: np.ndarray) -> np.ndarray:
    values = np.asarray(gram, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Gram matrix must be finite and square")
    return (
        values
        - np.mean(values, axis=0, keepdims=True)
        - np.mean(values, axis=1, keepdims=True)
        + float(np.mean(values))
    )


def _centered_alignment(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("centered Gram matrices must be aligned and square")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= np.finfo(float).eps:
        return None
    value = float(np.sum(a * b) / denominator)
    if not np.isfinite(value):
        return None
    return float(np.clip(value, -1.0, 1.0))


def _adjust_alignment(actual: float | None, null: float | None) -> float | None:
    if actual is None or null is None:
        return None
    denominator = 1.0 - float(null)
    if denominator <= np.finfo(float).eps:
        return None
    return float(np.clip((float(actual) - float(null)) / denominator, 0.0, 1.0))


def _kernel_fold_components(
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    y_train_scaled: np.ndarray,
    y_holdout_scaled: np.ndarray,
    permutations: Sequence[np.ndarray],
) -> Dict[str, Any]:
    feature_base = _median_positive_squared_distance(X_train)
    target_base = _median_positive_squared_distance(y_train_scaled)
    feature_grams = _rbf_grams(X_holdout, feature_base)
    target_grams = _rbf_grams(y_holdout_scaled, target_base)
    centered_features = {
        key: _center_gram(value) for key, value in feature_grams.items()
    }
    centered_target = _center_gram(
        np.mean(np.stack(list(target_grams.values()), axis=0), axis=0)
    )
    mixed_feature = _center_gram(
        np.mean(np.stack(list(feature_grams.values()), axis=0), axis=0)
    )

    def alignment_samples(feature: np.ndarray) -> tuple[float | None, list[float | None]]:
        actual = _centered_alignment(feature, centered_target)
        samples = []
        for permutation in permutations:
            indices = np.asarray(permutation, dtype=np.int64)
            if indices.shape != (len(centered_target),):
                raise ValueError("permutation length does not match held-out rows")
            permuted = centered_target[np.ix_(indices, indices)]
            samples.append(_centered_alignment(feature, permuted))
        return actual, samples

    actual, samples = alignment_samples(mixed_feature)
    if any(value is None for value in samples):
        null_alignment = None
    else:
        null_alignment = float(np.mean(samples))
    score = _adjust_alignment(actual, null_alignment)
    bandwidths: Dict[str, Any] = {}
    for key, feature in centered_features.items():
        bandwidth_actual, bandwidth_samples = alignment_samples(feature)
        bandwidth_null = (
            float(np.mean(bandwidth_samples))
            if all(value is not None for value in bandwidth_samples)
            else None
        )
        bandwidths[key] = {
            "raw_alignment": bandwidth_actual,
            "null_alignment": bandwidth_null,
            "null_alignment_samples": bandwidth_samples,
            "score": _adjust_alignment(bandwidth_actual, bandwidth_null),
        }
    return {
        "feature_base_squared_distance": float(feature_base),
        "target_base_squared_distance": float(target_base),
        "raw_alignment": actual,
        "null_alignment": null_alignment,
        "null_alignment_samples": samples,
        "score": score,
        "feature_bandwidths": bandwidths,
    }


def _aggregate_selector(folds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    current_rows = [row["current_common32"] for row in folds]
    current_ratios = [row["loss_ratio"] for row in current_rows]
    kernel_rows = [row["kernel_alignment"] for row in folds]

    def mean_or_none(key: str, rows: Sequence[Mapping[str, Any]]) -> float | None:
        values = [row[key] for row in rows]
        return float(np.mean(values)) if all(value is not None for value in values) else None

    bandwidths = {}
    for multiplier in BANDWIDTH_MULTIPLIERS:
        selected = [row["feature_bandwidths"][str(multiplier)] for row in kernel_rows]
        bandwidths[str(multiplier)] = {
            "raw_alignment": mean_or_none("raw_alignment", selected),
            "null_alignment": mean_or_none("null_alignment", selected),
            "score": mean_or_none("score", selected),
        }
    return {
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
        "kernel_alignment": {
            "raw_alignment": mean_or_none("raw_alignment", kernel_rows),
            "null_alignment": mean_or_none("null_alignment", kernel_rows),
            "score": mean_or_none("score", kernel_rows),
            "feature_bandwidths": bandwidths,
        },
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
        current_components = null_parent._fixed_model_components(
            estimator, transformed_holdout, y[holdout], bank
        )
        current_null = float(np.mean(current_components["null_loss_samples"]))
        current = null_parent._score_from_losses(
            float(current_components["actual_loss"]), current_null
        )
        X_train_prepared = estimator._prep_X(estimator._validate_X(transformed_train))
        X_holdout_prepared = estimator._prep_X(estimator._validate_X(transformed_holdout))
        y_train_scaled = estimator._transform_targets(estimator._validate_Y(y[train]))
        y_holdout_scaled = estimator._transform_targets(estimator._validate_Y(y[holdout]))
        kernel = _kernel_fold_components(
            X_train_prepared,
            X_holdout_prepared,
            y_train_scaled,
            y_holdout_scaled,
            bank,
        )
        folds.append(
            {
                "fold": int(fold),
                "fit_seed": int(fit_seed),
                "common_bank_seed": int(bank_seed),
                "row_count": int(len(holdout)),
                "prototype_count": int(estimator._model.n_clusters_total),
                "geometry_diagnostics": geometry_diagnostics,
                "current_common32": {
                    "actual_loss": float(current_components["actual_loss"]),
                    "null_loss": current_null,
                    "null_loss_samples": current_components["null_loss_samples"],
                    "loss_ratio": current["loss_ratio"],
                    "score": float(current["score"]),
                },
                "kernel_alignment": kernel,
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
    if estimand == "kernel_alignment_oi":
        values = [
            row["selectors"][method_id]["kernel_alignment"]["score"]
            for row in candidates
        ]
    elif estimand == "kernel_alignment_raw":
        values = [
            row["selectors"][method_id]["kernel_alignment"]["raw_alignment"]
            for row in candidates
        ]
    elif estimand.startswith("feature_bandwidth_") and estimand.endswith("_oi"):
        multiplier = estimand[len("feature_bandwidth_") : -len("_oi")]
        values = [
            row["selectors"][method_id]["kernel_alignment"]["feature_bandwidths"]
            [multiplier]["score"]
            for row in candidates
        ]
    else:
        raise ValueError(f"unknown estimand: {estimand!r}")
    if any(value is None for value in values):
        raise ValueError(f"{estimand} contains an undefined candidate score")
    return [float(value) for value in values]


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


def _decision(
    protocol: Mapping[str, Any], metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    config = protocol["decision_rule"]
    linear_probe = metrics["linear_probe"]["linear_probe"]
    methods: Dict[str, Any] = {}
    passing = []
    for method_id in _method_specs(protocol):
        baseline = metrics[method_id]["current_common32_oi"]
        candidate = metrics[method_id]["kernel_alignment_oi"]
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
                        observed is not None and minimum is not None and observed >= minimum
                    ),
                }
            )
        status = "pass" if all(row["pass"] for row in checks) else "fail"
        methods[method_id] = {"status": status, "checks": checks}
        if status == "pass":
            passing.append(f"{method_id}:{ELIGIBLE_ESTIMAND}")
    return {
        "status": "pass" if passing else "fail",
        "passing_candidates": passing,
        "methods": methods,
    }


def summarize(
    protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    estimands = (
        "current_common32_oi",
        "kernel_alignment_raw",
        "kernel_alignment_oi",
        *[f"feature_bandwidth_{value}_oi" for value in BANDWIDTH_MULTIPLIERS],
    )
    rows = [
        _panel_metric(panel, method_id, estimand)
        for panel in panels
        for method_id in _method_specs(protocol)
        for estimand in estimands
    ]
    rows.extend(
        _panel_metric(panel, "linear_probe", "linear_probe") for panel in panels
    )
    metrics = motivation._aggregate_metrics(protocol, rows)
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
    output: Path, motivation_raw_path: Path = DEFAULT_MOTIVATION_RAW
) -> Dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    _validate_motivation_ancestry(protocol, motivation_raw_path)
    source_paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "motivation_raw_results": motivation_raw_path.resolve(),
        "motivation_runner": motivation.RUNNER_PATH,
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
            print(f"completed {completed}/{total}: seed={seed} family={family}", flush=True)
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
