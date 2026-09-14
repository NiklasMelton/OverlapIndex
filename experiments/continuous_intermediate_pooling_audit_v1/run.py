"""Evaluate frozen intermediate-layer pooling views with linear and smooth heads."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

from experiments.continuous_cached_embedding_information_audit_v1 import run as information
from experiments.continuous_cached_embedding_smooth_head_audit_v1 import run as smooth
from experiments.continuous_intermediate_pooling_audit_v1 import extract
from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_real_source_qualification_v1 import extract as terminal_source
from experiments.continuous_real_source_qualification_v1 import prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_intermediate_pooling_audit_v1" / "audit"
HEADS = ("linear_ridge", "rbf_svr")
TIE_ATOL = 1.0e-12


def load_protocol() -> dict[str, Any]:
    protocol = extract.load_protocol()
    if tuple(protocol["heads"]["families"]) != HEADS:
        raise ValueError("intermediate-pooling head roster changed")
    expected = (
        len(protocol["targets"]) * len(protocol["backbones"]) * len(protocol["views"])
    )
    if expected != int(protocol["expected_grid"]["candidate_cell_count"]):
        raise ValueError("intermediate-pooling candidate grid changed")
    if len(_rbf_grid(protocol)) != int(protocol["expected_grid"]["rbf_parameter_count"]):
        raise ValueError("intermediate-pooling RBF grid changed")
    return protocol


def _rbf_grid(protocol: Mapping[str, Any]) -> list[dict[str, float]]:
    config = protocol["heads"]["rbf_svr"]
    return [
        {"C_standardized": float(C), "epsilon_standardized": float(epsilon), "bandwidth_factor": float(factor)}
        for C, epsilon, factor in itertools.product(
            config["C_standardized_target_units"],
            config["epsilon_standardized_target_units"],
            config["bandwidth_factors"],
        )
    ]


def _source_hashes() -> dict[str, str]:
    manifest = extract.validate_bank()
    paths = {
        "embedding_manifest": extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "protocol": PROTOCOL_PATH,
        "extractor": Path(extract.__file__).resolve(),
        "runner": Path(__file__).resolve(),
        "terminal_embedding_manifest": terminal_source.DEFAULT_OUTPUT_ROOT / "manifest.json",
    }
    observed = {name: base.sha256_path(path) for name, path in paths.items()}
    if manifest["protocol_sha256"] != observed["protocol"]:
        raise ValueError("intermediate-pooling embedding protocol identity changed")
    return observed


def _best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    values = np.asarray([float(row["mean_r2"]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("intermediate-pooling head tuning produced a nonfinite score")
    return rows[int(np.flatnonzero(values >= np.max(values) - TIE_ATOL)[0])]


def _endpoint_diagnostics(best: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for parameter, selected in best["params"].items():
        values = sorted({float(row["params"][parameter]) for row in rows})
        endpoint = "minimum" if float(selected) == values[0] else "maximum" if float(selected) == values[-1] else None
        if endpoint is not None:
            interior = [float(row["mean_r2"]) for row in rows if float(row["params"][parameter]) != float(selected)]
            output.append({
                "parameter": parameter, "endpoint": endpoint, "selected_value": float(selected),
                "inner_cv_advantage_r2": float(float(best["mean_r2"]) - max(interior)),
            })
    return output


def evaluate_heads(
    X_train: np.ndarray,
    y_train: np.ndarray,
    bins: np.ndarray,
    sample_ids: Sequence[str],
    X_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    *,
    split_seed: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    X_train = np.asarray(X_train, dtype=np.float64); X_evaluation = np.asarray(X_evaluation, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64); y_evaluation = np.asarray(y_evaluation, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.int64); ids = tuple(str(value) for value in sample_ids)
    if X_train.shape[0] != len(y_train) or bins.shape != y_train.shape or len(ids) != len(y_train) or len(ids) != len(set(ids)):
        raise ValueError("intermediate-pooling training rows or identities are invalid")
    if X_evaluation.shape[0] != len(y_evaluation) or X_train.shape[1] != X_evaluation.shape[1]:
        raise ValueError("intermediate-pooling evaluation rows are invalid")
    folds = []
    fold_audit = []
    splitter = StratifiedKFold(n_splits=int(protocol["heads"]["inner_folds"]), shuffle=True, random_state=int(split_seed))
    factors = [float(value) for value in protocol["heads"]["rbf_svr"]["bandwidth_factors"]]
    preprocessing_started = time.perf_counter()
    for fold, (train_position, validation_position) in enumerate(splitter.split(X_train, bins)):
        train_ids = {ids[int(position)] for position in train_position}; validation_ids = {ids[int(position)] for position in validation_position}
        if train_ids.intersection(validation_ids): raise ValueError("intermediate-pooling inner fold leaks a sample identity")
        Z_train, Z_validation, diagnostic = information._fit_preprocessor(
            X_train[train_position], X_train[validation_position], row_l2=True, pca_rank=None,
            seed=base._seed("intermediate-pooling-preprocess", split_seed, fold),
        )
        kernels, median = smooth._distance_views(Z_train, Z_validation, factors)
        folds.append({
            "Z_train": Z_train, "Z_validation": Z_validation,
            "y_train": y_train[train_position], "y_validation": y_train[validation_position], "kernels": kernels,
        })
        fold_audit.append({
            "fold": fold, "train_rows": int(len(train_position)), "validation_rows": int(len(validation_position)),
            "identity_overlap_count": 0, "train_sample_ids_sha256": base.sha256_payload(sorted(train_ids)),
            "validation_sample_ids_sha256": base.sha256_payload(sorted(validation_ids)),
            "preprocessing": diagnostic, "median_nonzero_training_distance": median,
        })
    Z_full, Z_evaluation, full_diagnostic = information._fit_preprocessor(
        X_train, X_evaluation, row_l2=True, pca_rank=None,
        seed=base._seed("intermediate-pooling-preprocess", split_seed, "full"),
    )
    full_kernels, full_median = smooth._distance_views(Z_full, Z_evaluation, factors)
    preprocessing_seconds = float(time.perf_counter() - preprocessing_started)

    ridge_started = time.perf_counter(); ridge_rows = []
    ridge_config = protocol["heads"]["linear_ridge"]
    for mean_loss_lambda in ridge_config["mean_loss_lambda"]:
        params = {"mean_loss_lambda": float(mean_loss_lambda)}; scores = []; warning_types = set()
        for fold in folds:
            estimator = Ridge(alpha=len(fold["Z_train"]) * float(mean_loss_lambda), solver="lsqr", tol=float(ridge_config["tolerance"]))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always"); estimator.fit(fold["Z_train"], fold["y_train"]); prediction = estimator.predict(fold["Z_validation"])
            warning_types.update(type(item.message).__name__ for item in caught)
            scores.append(float(r2_score(fold["y_validation"], prediction)))
        ridge_rows.append({"params": params, "fold_r2": scores, "mean_r2": float(np.mean(scores)), "warning_types": sorted(warning_types)})
    ridge_best = _best_row(ridge_rows)
    ridge = Ridge(alpha=len(Z_full) * float(ridge_best["params"]["mean_loss_lambda"]), solver="lsqr", tol=float(ridge_config["tolerance"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always"); ridge.fit(Z_full, y_train); ridge_prediction = ridge.predict(Z_evaluation)
    ridge_result = {
        "best_params": dict(ridge_best["params"]), "inner_best_r2": float(ridge_best["mean_r2"]),
        "parameter_rows": ridge_rows, "endpoint_diagnostics": _endpoint_diagnostics(ridge_best, ridge_rows),
        "evaluation_r2": float(r2_score(y_evaluation, ridge_prediction)), "prediction": np.asarray(ridge_prediction, dtype=np.float64).tolist(),
        "prediction_sha256": hashlib.sha256(np.asarray(ridge_prediction, dtype=np.float64).tobytes(order="C")).hexdigest(),
        "final_warning_types": sorted({type(item.message).__name__ for item in caught}),
        "model_fit_count": len(ridge_rows) * len(folds) + 1, "head_wall_seconds_excluding_shared_preprocessing": float(time.perf_counter() - ridge_started),
    }

    rbf_started = time.perf_counter(); rbf_rows = []
    for params in _rbf_grid(protocol):
        scores = []; warning_types = set()
        for fold in folds:
            train_kernel, validation_kernel = fold["kernels"][float(params["bandwidth_factor"])]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always"); prediction = smooth._fit_precomputed(train_kernel, fold["y_train"], validation_kernel, params)
            warning_types.update(type(item.message).__name__ for item in caught)
            scores.append(float(r2_score(fold["y_validation"], prediction)))
        rbf_rows.append({"params": params, "fold_r2": scores, "mean_r2": float(np.mean(scores)), "warning_types": sorted(warning_types)})
    rbf_best = _best_row(rbf_rows)
    train_kernel, evaluation_kernel = full_kernels[float(rbf_best["params"]["bandwidth_factor"])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always"); rbf_prediction = smooth._fit_precomputed(train_kernel, y_train, evaluation_kernel, rbf_best["params"])
    rbf_result = {
        "best_params": dict(rbf_best["params"]), "inner_best_r2": float(rbf_best["mean_r2"]),
        "parameter_rows": rbf_rows, "endpoint_diagnostics": _endpoint_diagnostics(rbf_best, rbf_rows),
        "evaluation_r2": float(r2_score(y_evaluation, rbf_prediction)), "prediction": rbf_prediction.tolist(),
        "prediction_sha256": hashlib.sha256(rbf_prediction.tobytes(order="C")).hexdigest(),
        "final_warning_types": sorted({type(item.message).__name__ for item in caught}),
        "model_fit_count": len(rbf_rows) * len(folds) + 1, "head_wall_seconds_excluding_shared_preprocessing": float(time.perf_counter() - rbf_started),
    }
    return {
        "heads": {"linear_ridge": ridge_result, "rbf_svr": rbf_result},
        "fold_audit": fold_audit, "full_preprocessing": full_diagnostic,
        "full_median_nonzero_training_distance": full_median, "shared_preprocessing_wall_seconds": preprocessing_seconds,
    }


def _rows(protocol: Mapping[str, Any], cohort: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    dataset_id, draw, _, evaluation = information._target_rows(cohort, {**target, "archived_training_rows": int(target["training_rows"])})
    return dataset_id, draw, evaluation


def _matrix(
    view_id: str, dataset_id: str, backbone: str, draw: Sequence[Mapping[str, Any]], evaluation: Sequence[Mapping[str, Any]],
    extracted: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]], terminal: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if view_id == "cached_final":
        matrix = terminal[dataset_id][backbone]
        return (
            matrix[np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)],
            matrix[np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)],
        )
    matrix = extracted[dataset_id][backbone][view_id]
    return np.asarray(matrix[: len(draw)]), np.asarray(matrix[len(draw) :])


def build_cell(
    protocol: Mapping[str, Any], target: Mapping[str, Any], backbone: str, view_id: str,
    cohort: Mapping[str, Any], extracted: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]], terminal: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    dataset_id, draw, evaluation = _rows(protocol, cohort, target)
    X_train, X_evaluation = _matrix(view_id, dataset_id, backbone, draw, evaluation, extracted, terminal)
    result = evaluate_heads(
        X_train, np.asarray([float(row["target"]) for row in draw], dtype=np.float64),
        np.asarray([int(row["target_bin"]) for row in draw], dtype=np.int64), [str(row["sample_id"]) for row in draw],
        X_evaluation, np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64),
        split_seed=base._seed("intermediate-pooling-audit", target["target_key"], backbone, view_id), protocol=protocol,
    )
    scores = {head: float(row["evaluation_r2"]) for head, row in result["heads"].items()}; envelope = max(scores.values())
    return {
        "cell_id": f"{target['target_key']}::{backbone}::{view_id}", "target_key": str(target["target_key"]),
        "dataset_id": dataset_id, "candidate_id": backbone, "view": {"view_id": view_id},
        "training_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in draw]),
        "evaluation_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in evaluation]),
        "head_evaluation_r2": scores, "envelope_r2": float(envelope),
        "best_head_ids": [head for head, score in scores.items() if score >= envelope - TIE_ATOL], **result,
    }


def summarize(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    view_summaries = [
        smooth._ranking_summary(cells, str(target["target_key"]), view_id, protocol)
        for target in protocol["targets"] for view_id in protocol["views"]
    ]
    nonterminal_conflicts = [row for row in view_summaries if row["view_id"] != "cached_final" and row["material_rank_conflict"]]
    conflicts_by_view = {view_id: sum(row["view_id"] == view_id for row in nonterminal_conflicts) for view_id in protocol["views"] if view_id != "cached_final"}
    conflicts_by_target = {str(target["target_key"]): sum(row["target_key"] == target["target_key"] for row in nonterminal_conflicts) for target in protocol["targets"]}
    stable = any(value >= 2 for value in conflicts_by_view.values()) or any(value >= 2 for value in conflicts_by_target.values())
    warnings_found = sorted({warning for cell in cells for head in cell["heads"].values() for warning in head["final_warning_types"]})
    return {
        "schema_version": 1, "status": "completed_continuous_intermediate_pooling_audit_v1",
        "view_summaries": view_summaries,
        "conflict_counts": {"by_view": conflicts_by_view, "by_target": conflicts_by_target, "nonterminal_total": len(nonterminal_conflicts)},
        "warning_types": warnings_found,
        "decision": {
            "stable_nonterminal_rank_conflict": bool(stable),
            "material_nonlinear_gain_view_count": sum(row["maximum_nonlinear_gain_r2"] >= float(protocol["ranking_diagnostics"]["material_nonlinear_gain_r2"]) for row in view_summaries),
            "next_stage": "freeze_implicated_view_for_continuous_oi_development" if stable else "mechanism_controlled_synthetic_development_panel",
            "method_promotion": "none_reference_only",
        },
        "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous intermediate-layer and pooling audit V1", "",
        "| Target | View | Linear rho | LP regret | Max nonlinear gain | Linear winner | Envelope winner | Conflict |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in summary["view_summaries"]:
        rho = "undefined" if row["linear_to_envelope_spearman"] is None else f"{row['linear_to_envelope_spearman']:.3f}"
        lines.append(
            f"| {row['target_key']} | {row['view_id']} | {rho} | {row['linear_top1_regret_r2']:.4f} | {row['maximum_nonlinear_gain_r2']:.4f} | "
            f"{', '.join(row['linear_best_candidate_ids'])} | {', '.join(row['envelope_best_candidate_ids'])} | {str(row['material_rank_conflict']).lower()} |"
        )
    decision = summary["decision"]
    lines.extend([
        "", "## Decision", "",
        f"- Stable nonterminal rank conflict: **{str(decision['stable_nonterminal_rank_conflict']).lower()}**.",
        f"- Nonterminal conflict count: **{summary['conflict_counts']['nonterminal_total']}**.",
        f"- Material nonlinear-gain views: **{decision['material_nonlinear_gain_view_count']}**.",
        f"- Model warning types: `{json.dumps(summary['warning_types'])}`.",
        f"- Next stage: **{decision['next_stage']}**.",
        "- This diagnostic promotes no OverlapIndex method.", "",
    ])
    return "\n".join(lines)


def _identity(source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {"protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": dict(source_hashes), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}


def validate_grid(protocol: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    cells = checkpoint["cells"]; expected = int(protocol["expected_grid"]["candidate_cell_count"])
    if len(cells) != expected or len({row["cell_id"] for row in cells}) != expected: raise ValueError("intermediate-pooling executed grid changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    output_root = output_root.resolve(); protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(): raise ValueError("intermediate-pooling outcome execution requires a clean committed source tree")
    source_hashes = _source_hashes(); terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume: raise FileExistsError("intermediate-pooling terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    identity = _identity(source_hashes); checkpoint_path = output_root / "checkpoint.json"
    if checkpoint_path.exists():
        if not resume: raise FileExistsError("intermediate-pooling checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity: raise ValueError("intermediate-pooling checkpoint identity changed")
    else: checkpoint = {"schema_version": 1, "identity": identity, "cells": []}
    output_root.mkdir(parents=True, exist_ok=True)
    cohort = prepare.load_cohort(); _, extracted = extract.load_bank()
    terminal_protocol = {
        "targets": protocol["targets"],
        "backbones": [str(row["backbone_id"]) for row in protocol["backbones"]],
    }
    _, terminal, _ = information._load_inputs(terminal_protocol)
    complete = {row["cell_id"] for row in checkpoint["cells"]}
    for target in protocol["targets"]:
        for backbone_config in protocol["backbones"]:
            backbone = str(backbone_config["backbone_id"])
            for view_id in protocol["views"]:
                cell_id = f"{target['target_key']}::{backbone}::{view_id}"
                if cell_id not in complete:
                    checkpoint["cells"].append(build_cell(protocol, target, backbone, view_id, cohort, extracted, terminal))
                    base._finite_json(checkpoint); base._atomic_json(checkpoint_path, checkpoint)
                    print(f"cell {len(checkpoint['cells'])}/{protocol['expected_grid']['candidate_cell_count']}: {cell_id}", flush=True)
    validate_grid(protocol, checkpoint); summary = summarize(protocol, checkpoint["cells"])
    raw = {
        "schema_version": 1, "status": "completed_continuous_intermediate_pooling_audit_v1", "protocol": protocol,
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": source_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "thread_environment": {key: os.environ.get(key) for key in base.THREAD_KEYS}},
        "cells": checkpoint["cells"], "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }
    base._finite_json(raw); base._finite_json(summary)
    base._atomic_json(output_root / "raw_results.json", raw); base._atomic_json(output_root / "summary.json", summary); base._atomic_bytes(output_root / "report.md", render_report(summary).encode("utf-8"))
    manifest = {
        "schema_version": 1, "status": "completed_continuous_intermediate_pooling_audit_v1", "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes, "checkpoint_identity": identity,
        "artifacts": {name: base.sha256_path(output_root / name) for name in ("checkpoint.json", "raw_results.json", "summary.json", "report.md")},
        "counts": {"cells": len(checkpoint["cells"]), "overlapindex_fits": 0, "reserved_backbones": 0},
        "git": {"commit": identity["git_commit"], "dirty": False}, "decision": summary["decision"],
    }
    base._atomic_json(output_root / "manifest.json", manifest)
    terminal_record = {"schema_version": 1, "status": "completed_continuous_intermediate_pooling_audit_v1", "manifest_sha256": base.sha256_path(output_root / "manifest.json"), "summary_sha256": base.sha256_path(output_root / "summary.json"), "decision": summary["decision"]}
    base._atomic_json(terminal_path, terminal_record); return validate_artifacts(output_root)


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve(); protocol = load_protocol()
    terminal = json.loads((output_root / "terminal.json").read_text(encoding="utf-8")); manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8")); summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8")); checkpoint = json.loads((output_root / "checkpoint.json").read_text(encoding="utf-8"))
    if terminal.get("status") != "completed_continuous_intermediate_pooling_audit_v1": raise ValueError("intermediate-pooling terminal status changed")
    if terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json") or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json"): raise ValueError("intermediate-pooling terminal identity changed")
    if terminal.get("decision") != summary.get("decision") or manifest.get("source_hashes") != _source_hashes(): raise ValueError("intermediate-pooling source or summary identity changed")
    for name, expected in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != expected: raise ValueError(f"intermediate-pooling artifact changed: {name}")
    validate_grid(protocol, checkpoint); return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv); result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume)
    print(base.canonical_json(result)); return 0


if __name__ == "__main__": raise SystemExit(main())
