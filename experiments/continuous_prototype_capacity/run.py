"""Frozen k=4 versus k=8 continuous-OI prototype-capacity experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import scipy
import sklearn
from sklearn.base import clone
from sklearn.model_selection import KFold

from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_multiview_ranking import run as multiview
from experiments.continuous_oracle_diagnostic import run as oracle
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_rff_mean_ranking import run as rff_mean
from experiments.continuous_rff_save_ranking import run as rff_save
from experiments.continuous_save_ranking import run as save


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
RUNNER_PATH = Path(__file__).resolve()


def _load_protocol() -> Dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_exploratory_before_outcomes":
        raise ValueError("protocol is not frozen for exploratory execution")
    return protocol


def _method_specs(protocol: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        row["method_id"]: dict(row)
        for row in protocol["continuous_oi"]["methods"]
    }


def generate_panel(
    protocol: Mapping[str, Any],
    seed: int,
    family: str,
) -> Dict[str, Any]:
    return save.generate_panel(protocol, seed, family)


def _prototype_diagnostics(
    estimator: Any,
    *,
    configured_k: int,
    training_row_count: int,
) -> Dict[str, Any]:
    backend = estimator._model
    fused = dict(estimator.fused_kmeans_diagnostics_)
    resolved = fused.get("resolved_k")
    if not isinstance(resolved, Mapping) or not resolved:
        raise RuntimeError("FusedKMeans did not expose resolved prototype counts")
    resolved_rows = sorted(
        (
            {
                "target_cell": int(cell),
                "prototype_count": int(count),
            }
            for cell, count in resolved.items()
        ),
        key=lambda row: row["target_cell"],
    )
    supports = sorted(int(value) for value in estimator.prototype_support_.values())
    prototype_count = int(backend.n_clusters_total)
    occupied = len(supports)
    if not supports or sum(supports) != int(training_row_count):
        raise RuntimeError("prototype assignment supports do not cover training rows")
    return {
        "configured_prototypes_per_cell": int(configured_k),
        "resolved_prototypes_per_cell": resolved_rows,
        "minimum_resolved_prototypes_per_cell": min(
            row["prototype_count"] for row in resolved_rows
        ),
        "maximum_resolved_prototypes_per_cell": max(
            row["prototype_count"] for row in resolved_rows
        ),
        "prototype_count": prototype_count,
        "occupied_prototype_count": occupied,
        "empty_assignment_prototype_count": prototype_count - occupied,
        "lloyd_empty_prototype_count": int(fused["empty_prototype_count"]),
        "support_minimum": int(min(supports)),
        "support_median": float(np.median(supports)),
        "support_maximum": int(max(supports)),
        "support_mean": float(np.mean(supports)),
        "support_total": int(sum(supports)),
    }


def evaluate_selector(
    method_id: str,
    X: np.ndarray,
    primitive: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    protocol = _load_protocol()
    spec = _method_specs(protocol)[method_id]
    configured_k = int(spec["prototypes_per_cell"])
    splitter = KFold(n_splits=4, shuffle=True, random_state=int(seed))
    folds = []
    started = time.perf_counter()
    for fold, (train, holdout) in enumerate(splitter.split(X)):
        fold_seed = int(seed) + int(fold)
        transformed_train, transformed_holdout, geometry_diagnostics = (
            multiview._prepare_fold_geometry(
                spec,
                X[train],
                X[holdout],
                primitive[train],
                primitive[holdout],
                y[train],
                fold_seed,
            )
        )
        base = save._base_selector(fold_seed)
        base.set_params(kmeans_k=configured_k)
        estimator = clone(base)
        estimator._fit_geometry_only(transformed_train, y[train])
        capacity_diagnostics = _prototype_diagnostics(
            estimator,
            configured_k=configured_k,
            training_row_count=len(train),
        )
        score = estimator.score_fixed(transformed_holdout, y[holdout])
        ratio = float(estimator.loss_ratio_)
        folds.append(
            {
                "fold": int(fold),
                "score": float(score),
                "actual_loss": float(estimator.actual_loss_),
                "null_loss": float(estimator.null_loss_),
                "loss_ratio": ratio if np.isfinite(ratio) else None,
                "prototype_count": int(estimator._model.n_clusters_total),
                "geometry_diagnostics": geometry_diagnostics,
                "prototype_capacity_diagnostics": capacity_diagnostics,
            }
        )
    ratios = [row["loss_ratio"] for row in folds]
    return {
        "score": float(np.mean([row["score"] for row in folds])),
        "mean_actual_loss": float(np.mean([row["actual_loss"] for row in folds])),
        "mean_null_loss": float(np.mean([row["null_loss"] for row in folds])),
        "mean_loss_ratio": (
            float(np.mean(ratios))
            if all(value is not None for value in ratios)
            else None
        ),
        "folds": folds,
        "wall_seconds": float(time.perf_counter() - started),
    }


def run_panel(protocol: Mapping[str, Any], seed: int, family: str) -> Dict[str, Any]:
    panel = generate_panel(protocol, seed, family)
    methods = _method_specs(protocol)
    candidates = []
    for position, candidate in enumerate(panel["candidates"]):
        candidate_seed = int(seed) * 100 + int(position)
        heads = residual.evaluate_heads(
            candidate["X_train"],
            panel["y_train"],
            candidate["X_test"],
            panel["y_test"],
            candidate_seed,
        )
        selectors = {
            method_id: evaluate_selector(
                method_id,
                candidate["X_train"],
                candidate["primitive_train"],
                panel["y_train"],
                candidate_seed,
            )
            for method_id in methods
        }
        head_scores = {
            head_id: float(result["test_r2"])
            for head_id, result in heads.items()
        }
        envelope = max(head_scores.values())
        winners = sorted(
            head_id
            for head_id, value in head_scores.items()
            if np.isclose(value, envelope, atol=1.0e-12, rtol=0.0)
        )
        candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "quality": candidate["quality"],
                "nuisance_scale": candidate["nuisance_scale"],
                "effective_nuisance_scale": candidate[
                    "effective_nuisance_scale"
                ],
                "heads": heads,
                "best_head_envelope_r2": float(envelope),
                "best_head_ids": winners,
                "selectors": selectors,
            }
        )
    return {
        "seed": int(seed),
        "family": family,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _capacity_summary(
    protocol: Mapping[str, Any],
    panels: list[Mapping[str, Any]],
) -> Dict[str, Any]:
    result = {}
    for method_id in _method_specs(protocol):
        folds = [
            fold["prototype_capacity_diagnostics"]
            for panel in panels
            for candidate in panel["candidates"]
            for fold in candidate["selectors"][method_id]["folds"]
        ]
        result[method_id] = {
            "fold_count": len(folds),
            "configured_prototypes_per_cell": int(
                folds[0]["configured_prototypes_per_cell"]
            ),
            "minimum_resolved_prototypes_per_cell": int(
                min(row["minimum_resolved_prototypes_per_cell"] for row in folds)
            ),
            "maximum_resolved_prototypes_per_cell": int(
                max(row["maximum_resolved_prototypes_per_cell"] for row in folds)
            ),
            "minimum_prototype_count": int(
                min(row["prototype_count"] for row in folds)
            ),
            "maximum_prototype_count": int(
                max(row["prototype_count"] for row in folds)
            ),
            "maximum_lloyd_empty_prototype_count": int(
                max(row["lloyd_empty_prototype_count"] for row in folds)
            ),
            "maximum_empty_assignment_prototype_count": int(
                max(row["empty_assignment_prototype_count"] for row in folds)
            ),
            "minimum_support": int(min(row["support_minimum"] for row in folds)),
            "median_support": float(
                np.median([row["support_median"] for row in folds])
            ),
        }
    return result


def _capacity_gate(
    protocol: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    config = protocol["capacity_gate"]
    methods = summary["methods"]
    capacity = summary["prototype_capacity_summary"]
    results = {}
    for pair in config["pairs"]:
        baseline_id = pair["k4_method_id"]
        candidate_id = pair["k8_method_id"]
        baseline = methods[baseline_id]
        candidate = methods[candidate_id]
        triangle_baseline = float(
            baseline["by_family"]["triangle_wave"]["mean_spearman"]
        )
        triangle_candidate = float(
            candidate["by_family"]["triangle_wave"]["mean_spearman"]
        )
        checks = [
            {
                "criterion": "triangle_wave_spearman_improvement",
                "observed": triangle_candidate - triangle_baseline,
                "minimum": float(config["minimum_triangle_spearman_improvement"]),
                "pass": bool(
                    triangle_candidate - triangle_baseline
                    >= float(config["minimum_triangle_spearman_improvement"])
                ),
            },
            {
                "criterion": "overall_mean_spearman_preservation",
                "observed": float(candidate["mean_spearman"]),
                "minimum": float(baseline["mean_spearman"])
                - float(config["maximum_overall_spearman_drop"]),
                "pass": bool(
                    candidate["mean_spearman"]
                    >= baseline["mean_spearman"]
                    - float(config["maximum_overall_spearman_drop"])
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
                    + float(config["maximum_mean_regret_increase"])
                ),
            },
            {
                "criterion": "k8_resolved_in_every_fold",
                "observed": int(
                    capacity[candidate_id]["minimum_resolved_prototypes_per_cell"]
                ),
                "minimum": 8,
                "pass": bool(
                    capacity[candidate_id]["minimum_resolved_prototypes_per_cell"]
                    >= 8
                ),
            },
        ]
        for family in config["preserve_families"]:
            observed = float(candidate["by_family"][family]["mean_spearman"])
            minimum = float(baseline["by_family"][family]["mean_spearman"]) - float(
                config["maximum_family_spearman_drop"]
            )
            checks.append(
                {
                    "criterion": f"preserve_{family}_spearman",
                    "observed": observed,
                    "minimum": minimum,
                    "pass": bool(observed >= minimum),
                }
            )
        results[candidate_id] = {
            "baseline_method_id": baseline_id,
            "status": "pass" if all(row["pass"] for row in checks) else "fail",
            "checks": checks,
        }
    return {
        "status": (
            "pass" if any(row["status"] == "pass" for row in results.values()) else "fail"
        ),
        "candidates": results,
    }


def summarize(protocol: Mapping[str, Any], panels: list[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = save.summarize(protocol, panels)
    summary["prototype_capacity_summary"] = _capacity_summary(protocol, panels)
    summary["capacity_gate"] = _capacity_gate(protocol, summary)
    summary["universal_success_gate"] = rff_save._success_gate(protocol, summary)
    return summary


def _validate_protocol_against_code(protocol: Mapping[str, Any]) -> None:
    specs = _method_specs(protocol)
    expected = {
        "save8_k4": ("save", 4, None),
        "save8_k8": ("save", 8, None),
        "rff64_mean8_k4": ("rff_mean", 4, None),
        "rff64_mean8_k8": ("rff_mean", 8, None),
        "save_rffmean_k4": ("multiview", 4, ("save", "rff_mean")),
        "save_rffmean_k8": ("multiview", 8, ("save", "rff_mean")),
        "save_rffmean_residual_k4": (
            "multiview",
            4,
            ("save", "rff_mean", "residual"),
        ),
        "save_rffmean_residual_k8": (
            "multiview",
            8,
            ("save", "rff_mean", "residual"),
        ),
    }
    observed = {
        method_id: (
            row["geometry"],
            row["prototypes_per_cell"],
            tuple(row["blocks"]) if "blocks" in row else None,
        )
        for method_id, row in specs.items()
    }
    if observed != expected:
        raise ValueError("frozen prototype-capacity method grid does not match code")
    if len(protocol["seeds"]) != 3 or len(protocol["families"]) != 6:
        raise ValueError("frozen panel grid must contain 3 seeds by 6 families")
    expected_pairs = {
        ("save8_k4", "save8_k8"),
        ("rff64_mean8_k4", "rff64_mean8_k8"),
        ("save_rffmean_k4", "save_rffmean_k8"),
        ("save_rffmean_residual_k4", "save_rffmean_residual_k8"),
    }
    observed_pairs = {
        (row["k4_method_id"], row["k8_method_id"])
        for row in protocol["capacity_gate"]["pairs"]
    }
    if observed_pairs != expected_pairs:
        raise ValueError("frozen k=4/k=8 pair grid does not match code")


def run(output: Path) -> Dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    source_paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "multiview_runner": multiview.RUNNER_PATH,
        "rff_mean_runner": rff_mean.RUNNER_PATH,
        "rff_save_runner": rff_save.RUNNER_PATH,
        "rff_geometry": rff_save.GEOMETRY_PATH,
        "save_runner": save.RUNNER_PATH,
        "save_geometry": save.GEOMETRY_PATH,
        "common_runner": common.RUNNER_PATH,
        "oracle_runner": oracle.RUNNER_PATH,
        "residual_runner": residual.RUNNER_PATH,
    }
    source_hashes = {
        key: common._sha256(path)
        for key, path in source_paths.items()
    }
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
                if checkpoint["source_hashes"] != source_hashes:
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
    args = parser.parse_args()
    summary = run(args.output)
    print(
        json.dumps(
            {
                "reference_performance_gate": summary[
                    "reference_performance_gate"
                ]["status"],
                "capacity_gate": summary["capacity_gate"],
                "universal_success_gate": summary["universal_success_gate"],
                "methods": {
                    method_id: {
                        "mean_spearman": row["mean_spearman"],
                        "mean_regret_r2": row["mean_regret_r2"],
                    }
                    for method_id, row in summary["methods"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
