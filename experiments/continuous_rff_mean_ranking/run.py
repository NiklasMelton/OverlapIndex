"""Frozen random-feature mean-embedding experiment for continuous OI."""

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
from experiments.continuous_oracle_diagnostic import run as oracle
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_rff_save_ranking import run as rff_save
from experiments.continuous_rff_save_ranking.geometry import (
    fit_random_fourier_transform,
)
from experiments.continuous_save_ranking import run as save
from experiments.continuous_save_ranking.geometry import (
    fit_target_moment_transform,
)


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


def _prepare_fold_geometry(
    spec: Mapping[str, Any],
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    primitive_train: np.ndarray,
    primitive_holdout: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    geometry = str(spec["geometry"])
    if geometry not in {"rff_mean", "rff_mean_save"}:
        return rff_save._prepare_fold_geometry(
            spec,
            X_train,
            X_holdout,
            primitive_train,
            primitive_holdout,
            y_train,
            seed,
        )

    n_components = int(spec["rff_components"])
    sketch = fit_random_fourier_transform(
        X_train,
        n_components=n_components,
        random_state=seed,
    )
    sketch_train = sketch.transform(X_train)
    sketch_holdout = sketch.transform(X_holdout)
    mode = "mean" if geometry == "rff_mean" else "mean_save"
    cells = save._target_cells(y_train, 8)
    transform = fit_target_moment_transform(
        sketch_train,
        cells,
        mode=mode,
        residual_rank=8,
        projection_rank=8,
        random_state=seed,
    )
    diagnostics = {
        "geometry": geometry,
        "moment_mode": mode,
        "rff_components": n_components,
        "rff_gamma": sketch.gamma,
        "rff_median_squared_distance": sketch.median_squared_distance,
        "rff_input_dimension": sketch.input_dimension,
        "output_dimension": transform.output_dimension,
        "effective_residual_rank": (
            transform.residual_transform.effective_nuisance_rank
        ),
        "mean_trace": transform.mean_trace,
        "save_trace": transform.save_trace,
        "moment_eigenvalues": transform.eigenvalues.tolist(),
    }
    return (
        transform.transform(sketch_train),
        transform.transform(sketch_holdout),
        diagnostics,
    )


def evaluate_selector(
    method_id: str,
    X: np.ndarray,
    primitive: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    protocol = _load_protocol()
    spec = _method_specs(protocol)[method_id]
    splitter = KFold(n_splits=4, shuffle=True, random_state=int(seed))
    folds = []
    started = time.perf_counter()
    for fold, (train, holdout) in enumerate(splitter.split(X)):
        fold_seed = int(seed) + int(fold)
        transformed_train, transformed_holdout, geometry_diagnostics = (
            _prepare_fold_geometry(
                spec,
                X[train],
                X[holdout],
                primitive[train],
                primitive[holdout],
                y[train],
                fold_seed,
            )
        )
        estimator = clone(save._base_selector(fold_seed))
        estimator._fit_geometry_only(transformed_train, y[train])
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


def summarize(protocol: Mapping[str, Any], panels: list[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = save.summarize(protocol, panels)
    summary["rff_mean_success_gate"] = rff_save._success_gate(protocol, summary)
    return summary


def _validate_protocol_against_code(protocol: Mapping[str, Any]) -> None:
    specs = _method_specs(protocol)
    expected = {
        "observed_k4": ("observed", None),
        "residual8_k4": ("residual8", None),
        "save8_k4": ("save", None),
        "rff64_k4": ("rff", 64),
        "rff64_mean8_k4": ("rff_mean", 64),
        "rff64_mean_save8_k4": ("rff_mean_save", 64),
        "oracle_quadratic_lift_k4": ("oracle_quadratic_lift", None),
    }
    observed = {
        method_id: (row["geometry"], row.get("rff_components"))
        for method_id, row in specs.items()
    }
    if observed != expected:
        raise ValueError("frozen RFF-mean method grid does not match code")
    if len(protocol["seeds"]) != 3 or len(protocol["families"]) != 6:
        raise ValueError("frozen panel grid must contain 3 seeds by 6 families")
    if len(protocol["candidate_grid"]["quality"]) != 4 or len(
        protocol["candidate_grid"]["nuisance_scale"]
    ) != 4:
        raise ValueError("candidate grid must be a 4 by 4 factorial")
    if set(protocol["success_gate"]["candidate_method_ids"]) != {
        "rff64_mean8_k4",
        "rff64_mean_save8_k4",
    }:
        raise ValueError("success gate must evaluate both frozen kernel-mean methods")


def run(output: Path) -> Dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    source_paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
        "rff_runner": rff_save.RUNNER_PATH,
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
                "rff_mean_success_gate": summary["rff_mean_success_gate"],
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
