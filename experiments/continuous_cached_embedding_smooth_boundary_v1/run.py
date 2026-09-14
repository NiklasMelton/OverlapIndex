"""Resolve RBF grid boundaries in the sole smooth-head ranking conflict."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn

from experiments.continuous_cached_embedding_information_audit_v1 import run as information
from experiments.continuous_cached_embedding_smooth_head_audit_v1 import run as smooth
from experiments.continuous_ranking_benchmark_v2 import run as base


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_cached_embedding_smooth_boundary_v1" / "audit"
SMOOTH_ROOT = ROOT / "artifacts" / "continuous_cached_embedding_smooth_head_audit_v1" / "audit"
INFORMATION_ROOT = ROOT / "artifacts" / "continuous_cached_embedding_information_audit_v1" / "audit"


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or base.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("smooth boundary protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_cached_embedding_smooth_boundary_v1_before_outcomes":
        raise ValueError("smooth boundary protocol is not frozen")
    if len(smooth._grid(protocol)) != int(protocol["expected_grid"]["rbf_parameter_count"]):
        raise ValueError("smooth boundary RBF parameter grid changed")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "smooth_protocol": smooth.PROTOCOL_PATH,
        "smooth_raw_results": SMOOTH_ROOT / "raw_results.json",
        "smooth_summary": SMOOTH_ROOT / "summary.json",
        "smooth_manifest": SMOOTH_ROOT / "manifest.json",
        "smooth_terminal": SMOOTH_ROOT / "terminal.json",
        "information_raw_results": INFORMATION_ROOT / "raw_results.json",
    }


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = _source_paths()
    observed = {name: base.sha256_path(path) for name, path in paths.items()}
    if any(observed[name] != protocol["ancestry"][f"{name}_sha256"] for name in observed):
        raise ValueError("smooth boundary ancestry changed")
    terminal = smooth.validate_artifacts(SMOOTH_ROOT)
    if not terminal["decision"]["calibration_pass"] or terminal["decision"]["material_rank_conflict_count"] != 1:
        raise ValueError("smooth boundary requires the calibrated sole-conflict source result")
    information.validate_artifacts(INFORMATION_ROOT)
    return observed


def build_cell(
    protocol: Mapping[str, Any], target: Mapping[str, Any], view: Mapping[str, Any], backbone: str,
    cohort: Mapping[str, Any], matrices: Mapping[str, Mapping[str, np.ndarray]], information_raw: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_id, draw, _, evaluation = information._target_rows(cohort, target)
    matrix = matrices[dataset_id][backbone]
    train_indices = np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)
    evaluation_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    result = smooth.evaluate_rbf(
        matrix[train_indices], np.asarray([float(row["target"]) for row in draw], dtype=np.float64),
        np.asarray([int(row["target_bin"]) for row in draw], dtype=np.int64), [str(row["sample_id"]) for row in draw],
        matrix[evaluation_indices], np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64),
        view=view,
        split_seed=base._seed("cached-information-audit", target["target_key"], backbone, view["view_id"]),
        protocol=protocol,
    )
    source = smooth._information_cell(information_raw, str(target["target_key"]), backbone, str(view["view_id"]))
    scores = dict(source["head_evaluation_r2"]); scores["rbf_svr"] = float(result["evaluation_r2"])
    envelope = max(scores.values())
    endpoints = []
    for parameter, selected in result["best_params"].items():
        values = sorted({float(row["params"][parameter]) for row in result["parameter_rows"]})
        endpoint = "minimum" if float(selected) == values[0] else "maximum" if float(selected) == values[-1] else None
        if endpoint is not None:
            interior = [float(row["mean_r2"]) for row in result["parameter_rows"] if float(row["params"][parameter]) != float(selected)]
            endpoints.append({"parameter": parameter, "endpoint": endpoint, "selected_value": float(selected), "inner_cv_advantage_r2": float(result["inner_best_r2"] - max(interior))})
    return {
        "cell_id": f"{target['target_key']}::{backbone}::{view['view_id']}",
        "target_key": str(target["target_key"]), "dataset_id": dataset_id, "candidate_id": backbone,
        "view": dict(view), "head_evaluation_r2": scores, "envelope_r2": float(envelope),
        "best_head_ids": [head for head, score in scores.items() if score >= envelope - smooth.TIE_ATOL],
        "endpoint_diagnostics": endpoints, "rbf_svr": result,
    }


def summarize(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = str(protocol["targets"][0]["target_key"]); view = str(protocol["evaluated_views"][0]["view_id"])
    ranking = smooth._ranking_summary(cells, target, view, protocol)
    warnings_found = sorted({warning for cell in cells for warning in cell["rbf_svr"]["final_warning_types"]})
    endpoint_rows = [row for cell in cells for row in cell["endpoint_diagnostics"]]
    material = bool(ranking["material_rank_conflict"])
    return {
        "schema_version": 1, "status": "completed_continuous_cached_embedding_smooth_boundary_v1",
        "ranking": ranking, "endpoint_diagnostics": endpoint_rows, "warning_types": warnings_found,
        "decision": {
            "material_rank_conflict": material,
            "linear_top1_regret_r2": float(ranking["linear_top1_regret_r2"]),
            "linear_to_envelope_spearman": ranking["linear_to_envelope_spearman"],
            "next_stage": "repeat_on_all_five_backbones_and_multiple_draws" if material else "intermediate_layer_and_pooling_audit",
            "method_promotion": "none_reference_only",
        },
        "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    ranking = summary["ranking"]; decision = summary["decision"]
    lines = [
        "# Continuous cached-embedding smooth boundary V1", "",
        "| Candidate | Linear R² | RBF R² | Envelope R² | Nonlinear gain | Best head |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in ranking["candidates"]:
        lines.append(f"| {row['candidate_id']} | {row['linear_r2']:.4f} | {row['rbf_svr_r2']:.4f} | {row['envelope_r2']:.4f} | {row['nonlinear_gain_r2']:.4f} | {', '.join(row['best_head_ids'])} |")
    lines.extend([
        "", "## Decision", "",
        f"- Linear-to-envelope Spearman: **{ranking['linear_to_envelope_spearman']:.3f}**.",
        f"- Linear top-1 regret: **{ranking['linear_top1_regret_r2']:.4f} R²**.",
        f"- Remaining numeric endpoint selections: **{len(summary['endpoint_diagnostics'])}**.",
        f"- Model warning types: `{json.dumps(summary['warning_types'])}`.",
        f"- Next stage: **{decision['next_stage']}**.",
        "- This diagnostic promotes no OverlapIndex method.", "",
    ])
    return "\n".join(lines)


def _identity(source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "runner_sha256": base.sha256_path(Path(__file__)),
        "source_hashes": dict(source_hashes),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def validate_grid(protocol: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    cells = checkpoint["cells"]
    if len(cells) != int(protocol["expected_grid"]["cell_count"]) or len({row["cell_id"] for row in cells}) != len(cells):
        raise ValueError("smooth boundary executed grid changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    output_root = output_root.resolve(); protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("smooth boundary execution requires a clean committed source tree")
    source_hashes = _source_hashes(protocol); terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume: raise FileExistsError("smooth boundary terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    identity = _identity(source_hashes); checkpoint_path = output_root / "checkpoint.json"
    if checkpoint_path.exists():
        if not resume: raise FileExistsError("smooth boundary checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity: raise ValueError("smooth boundary checkpoint identity changed")
    else:
        checkpoint = {"schema_version": 1, "identity": identity, "cells": []}
    output_root.mkdir(parents=True, exist_ok=True)
    cohort, matrices, _ = information._load_inputs(protocol)
    information_raw = json.loads((INFORMATION_ROOT / "raw_results.json").read_text(encoding="utf-8"))
    complete = {row["cell_id"] for row in checkpoint["cells"]}
    target, view = protocol["targets"][0], protocol["evaluated_views"][0]
    for backbone in protocol["backbones"]:
        cell_id = f"{target['target_key']}::{backbone}::{view['view_id']}"
        if cell_id not in complete:
            checkpoint["cells"].append(build_cell(protocol, target, view, backbone, cohort, matrices, information_raw))
            base._finite_json(checkpoint); base._atomic_json(checkpoint_path, checkpoint)
            print(f"cell {len(checkpoint['cells'])}/{protocol['expected_grid']['cell_count']}: {cell_id}", flush=True)
    validate_grid(protocol, checkpoint); summary = summarize(protocol, checkpoint["cells"])
    raw = {
        "schema_version": 1, "status": "completed_continuous_cached_embedding_smooth_boundary_v1",
        "protocol": protocol, "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": source_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "thread_environment": {key: os.environ.get(key) for key in base.THREAD_KEYS}},
        "cells": checkpoint["cells"], "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }
    base._finite_json(raw); base._finite_json(summary)
    base._atomic_json(output_root / "raw_results.json", raw); base._atomic_json(output_root / "summary.json", summary)
    base._atomic_bytes(output_root / "report.md", render_report(summary).encode("utf-8"))
    manifest = {
        "schema_version": 1, "status": "completed_continuous_cached_embedding_smooth_boundary_v1",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": source_hashes, "checkpoint_identity": identity,
        "artifacts": {name: base.sha256_path(output_root / name) for name in ("checkpoint.json", "raw_results.json", "summary.json", "report.md")},
        "counts": {"cells": len(checkpoint["cells"]), "overlapindex_fits": 0, "reserved_backbones": 0},
        "git": {"commit": identity["git_commit"], "dirty": False}, "decision": summary["decision"],
    }
    base._atomic_json(output_root / "manifest.json", manifest)
    terminal = {
        "schema_version": 1, "status": "completed_continuous_cached_embedding_smooth_boundary_v1",
        "manifest_sha256": base.sha256_path(output_root / "manifest.json"), "summary_sha256": base.sha256_path(output_root / "summary.json"),
        "decision": summary["decision"],
    }
    base._atomic_json(terminal_path, terminal); return validate_artifacts(output_root)


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve(); protocol = load_protocol()
    terminal = json.loads((output_root / "terminal.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output_root / "checkpoint.json").read_text(encoding="utf-8"))
    if terminal.get("status") != "completed_continuous_cached_embedding_smooth_boundary_v1": raise ValueError("smooth boundary terminal status changed")
    if terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json") or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json"): raise ValueError("smooth boundary terminal identity changed")
    if terminal.get("decision") != summary.get("decision") or manifest.get("source_hashes") != _source_hashes(protocol): raise ValueError("smooth boundary source or summary identity changed")
    for name, expected in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != expected: raise ValueError(f"smooth boundary artifact changed: {name}")
    validate_grid(protocol, checkpoint); return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume)
    print(base.canonical_json(result)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

