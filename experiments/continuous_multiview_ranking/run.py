"""Frozen equal-block multi-view experiment for continuous OI."""

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


def _single_block(
    block: str,
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    primitive_train: np.ndarray,
    primitive_holdout: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if block == "rff_mean":
        spec = {
            "geometry": "rff_mean",
            "rff_components": 64,
        }
        return rff_mean._prepare_fold_geometry(
            spec,
            X_train,
            X_holdout,
            primitive_train,
            primitive_holdout,
            y_train,
            seed,
        )
    geometry = {
        "save": "save",
        "observed": "observed",
        "residual": "residual8",
    }.get(block)
    if geometry is None:
        raise ValueError(f"unknown multi-view block: {block!r}")
    return save._prepare_fold_geometry(
        geometry,
        X_train,
        X_holdout,
        primitive_train,
        primitive_holdout,
        y_train,
        seed,
    )


def _equal_block_concatenation(
    train_blocks: list[np.ndarray],
    holdout_blocks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(train_blocks) < 2 or len(train_blocks) != len(holdout_blocks):
        raise ValueError("multi-view concatenation requires aligned block lists")
    weight = 1.0 / np.sqrt(float(len(train_blocks)))
    normalized_train = [save._row_l2(block) * weight for block in train_blocks]
    normalized_holdout = [save._row_l2(block) * weight for block in holdout_blocks]
    train = np.asarray(np.concatenate(normalized_train, axis=1), dtype=np.float32)
    holdout = np.asarray(np.concatenate(normalized_holdout, axis=1), dtype=np.float32)
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(holdout)):
        raise RuntimeError("multi-view geometry emitted non-finite values")
    return train, holdout


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
    if geometry != "multiview":
        return rff_mean._prepare_fold_geometry(
            spec,
            X_train,
            X_holdout,
            primitive_train,
            primitive_holdout,
            y_train,
            seed,
        )

    blocks = tuple(str(value) for value in spec["blocks"])
    train_blocks = []
    holdout_blocks = []
    block_diagnostics = []
    for block in blocks:
        train, holdout, diagnostics = _single_block(
            block,
            X_train,
            X_holdout,
            primitive_train,
            primitive_holdout,
            y_train,
            seed,
        )
        train_blocks.append(train)
        holdout_blocks.append(holdout)
        block_diagnostics.append(
            {
                "block": block,
                "dimension": int(train.shape[1]),
                "state": diagnostics,
            }
        )
    transformed_train, transformed_holdout = _equal_block_concatenation(
        train_blocks,
        holdout_blocks,
    )
    return transformed_train, transformed_holdout, {
        "geometry": "multiview",
        "blocks": list(blocks),
        "block_count": len(blocks),
        "block_weight": 1.0 / np.sqrt(float(len(blocks))),
        "block_diagnostics": block_diagnostics,
        "output_dimension": int(transformed_train.shape[1]),
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
    summary["multiview_success_gate"] = rff_save._success_gate(protocol, summary)
    return summary


def _validate_protocol_against_code(protocol: Mapping[str, Any]) -> None:
    specs = _method_specs(protocol)
    expected = {
        "observed_k4": ("observed", None),
        "residual8_k4": ("residual8", None),
        "save8_k4": ("save", None),
        "rff64_mean8_k4": ("rff_mean", None),
        "save_rffmean_k4": ("multiview", ("save", "rff_mean")),
        "save_rffmean_observed_k4": (
            "multiview",
            ("save", "rff_mean", "observed"),
        ),
        "save_rffmean_residual_k4": (
            "multiview",
            ("save", "rff_mean", "residual"),
        ),
        "oracle_quadratic_lift_k4": ("oracle_quadratic_lift", None),
    }
    observed = {
        method_id: (row["geometry"], tuple(row["blocks"]) if "blocks" in row else None)
        for method_id, row in specs.items()
    }
    if observed != expected:
        raise ValueError("frozen multi-view method grid does not match code")
    if len(protocol["seeds"]) != 3 or len(protocol["families"]) != 6:
        raise ValueError("frozen panel grid must contain 3 seeds by 6 families")
    if len(protocol["candidate_grid"]["quality"]) != 4 or len(
        protocol["candidate_grid"]["nuisance_scale"]
    ) != 4:
        raise ValueError("candidate grid must be a 4 by 4 factorial")
    if set(protocol["success_gate"]["candidate_method_ids"]) != {
        "save_rffmean_k4",
        "save_rffmean_observed_k4",
        "save_rffmean_residual_k4",
    }:
        raise ValueError("success gate must evaluate all frozen multi-view methods")


def run(output: Path) -> Dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    source_paths = {
        "protocol": PROTOCOL_PATH,
        "runner": RUNNER_PATH,
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
                "multiview_success_gate": summary["multiview_success_gate"],
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
