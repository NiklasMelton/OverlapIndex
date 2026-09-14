"""Run the frozen reference-head replication before any continuous-OI outcomes."""

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
from scipy.stats import spearmanr

from experiments.continuous_early_mean_replication_v1 import extract
from experiments.continuous_intermediate_pooling_audit_v1 import run as head_parent
from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_real_source_qualification_v1 import prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_early_mean_replication_v1" / "reference"
TIE_ATOL = 1.0e-12


def load_protocol() -> dict[str, Any]:
    protocol = extract.load_protocol()
    expected = protocol["expected_grid"]
    if (
        tuple(protocol["reference_heads"]["families"]) != ("linear_ridge", "rbf_svr")
        or int(expected["reference_cell_count"]) != 50
        or int(expected["reference_panel_count"]) != 10
    ):
        raise ValueError("early-mean reference protocol differs from code")
    return protocol


def _source_hashes() -> dict[str, str]:
    paths = {
        "embedding_manifest": extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "extractor": Path(extract.__file__).resolve(),
        "head_evaluator": Path(head_parent.__file__).resolve(),
        "prepared_cohort": prepare.DEFAULT_PREPARED_ROOT / "cohort.json",
    }
    return {name: base.sha256_path(path) for name, path in paths.items()}


def _rows(
    cohort: Mapping[str, Any], target: Mapping[str, Any], draw_seed: int,
) -> tuple[str, list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    dataset_id, target_id = str(target["target_key"]).split("::", 1)
    rows = cohort["datasets"][dataset_id]["target_cohorts"][target_id]
    draw = base.select_unique_draw(
        rows,
        pool_rows=int(target["training_pool_rows"]),
        draw_rows=int(target["training_rows"]),
        seed=int(draw_seed),
    )
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    if (
        len(draw) != int(target["training_rows"])
        or len(evaluation) != int(target["evaluation_rows"])
        or {str(row["sample_id"]) for row in draw}.intersection(str(row["sample_id"]) for row in evaluation)
    ):
        raise ValueError(f"early-mean draw identity changed: {target['target_key']}::{draw_seed}")
    return dataset_id, list(draw), evaluation


def evaluate_cell(
    protocol: Mapping[str, Any], target: Mapping[str, Any], draw_seed: int, backbone: str,
    cohort: Mapping[str, Any], matrices: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    dataset_id, draw, evaluation = _rows(cohort, target, draw_seed)
    matrix = matrices[dataset_id][backbone]
    train_indices = np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)
    evaluation_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    head_protocol = {**dict(protocol), "heads": protocol["reference_heads"]}
    result = head_parent.evaluate_heads(
        np.asarray(matrix[train_indices]),
        np.asarray([float(row["target"]) for row in draw], dtype=np.float64),
        np.asarray([int(row["target_bin"]) for row in draw], dtype=np.int64),
        [str(row["sample_id"]) for row in draw],
        np.asarray(matrix[evaluation_indices]),
        np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64),
        split_seed=base._seed("early-mean-reference", target["target_key"], draw_seed, backbone),
        protocol=head_protocol,
    )
    scores = {head: float(row["evaluation_r2"]) for head, row in result["heads"].items()}
    envelope = max(scores.values())
    return {
        "cell_id": f"{target['target_key']}::{draw_seed}::{backbone}",
        "target_key": str(target["target_key"]),
        "dataset_id": dataset_id,
        "draw_seed": int(draw_seed),
        "candidate_id": backbone,
        "training_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in draw]),
        "evaluation_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in evaluation]),
        "head_evaluation_r2": scores,
        "head_inner_best_r2": {head: float(row["inner_best_r2"]) for head, row in result["heads"].items()},
        "envelope_r2": float(envelope),
        "best_head_ids": [head for head, score in scores.items() if score >= envelope - TIE_ATOL],
        **result,
    }


def panel_summary(
    protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], target_key: str, draw_seed: int,
) -> dict[str, Any]:
    order = [str(row["backbone_id"]) for row in protocol["backbones"]]
    by_id = {
        str(row["candidate_id"]): row
        for row in cells
        if row["target_key"] == target_key and int(row["draw_seed"]) == int(draw_seed)
    }
    if set(by_id) != set(order):
        raise ValueError("early-mean reference panel is incomplete")
    rows = [by_id[candidate] for candidate in order]
    linear = np.asarray([row["head_evaluation_r2"]["linear_ridge"] for row in rows], dtype=np.float64)
    envelope = np.asarray([row["envelope_r2"] for row in rows], dtype=np.float64)
    if len(np.unique(linear)) < 2 or len(np.unique(envelope)) < 2:
        rho, status = None, "undefined_constant_ranking"
    else:
        rho, status = float(spearmanr(linear, envelope).statistic), "defined"
    linear_best = np.flatnonzero(linear >= np.max(linear) - TIE_ATOL)
    envelope_best = np.flatnonzero(envelope >= np.max(envelope) - TIE_ATOL)
    regret = float(np.max(envelope) - np.max(envelope[linear_best]))
    maximum_gain = float(np.max(envelope - linear))
    gate = protocol["reference_gate"]
    conflict = bool(
        regret >= float(gate["material_linear_top1_regret_r2"])
        or (rho is not None and rho <= float(gate["material_linear_to_envelope_spearman_upper"]))
    )
    return {
        "panel_id": f"{target_key}::{draw_seed}",
        "target_key": target_key,
        "draw_seed": int(draw_seed),
        "candidate_count": len(rows),
        "linear_to_envelope_spearman": rho,
        "linear_to_envelope_spearman_status": status,
        "linear_top1_regret_r2": regret,
        "maximum_nonlinear_gain_r2": maximum_gain,
        "material_rank_conflict": conflict,
        "material_nonlinear_gain": maximum_gain >= float(gate["material_nonlinear_gain_r2"]),
        "linear_best_candidate_ids": [order[int(index)] for index in linear_best],
        "envelope_best_candidate_ids": [order[int(index)] for index in envelope_best],
        "candidates": [
            {
                "candidate_id": order[index],
                "linear_r2": float(linear[index]),
                "rbf_svr_r2": float(rows[index]["head_evaluation_r2"]["rbf_svr"]),
                "envelope_r2": float(envelope[index]),
                "nonlinear_gain_r2": float(envelope[index] - linear[index]),
                "best_head_ids": list(rows[index]["best_head_ids"]),
            }
            for index in range(len(rows))
        ],
    }


def summarize(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    panels = [
        panel_summary(protocol, cells, str(target["target_key"]), int(seed))
        for target in protocol["targets"]
        for seed in target["draw_seeds"]
    ]
    target_rows = []
    required_conflicts = int(protocol["reference_gate"]["minimum_conflicting_draws_per_target"])
    required_gains = int(protocol["reference_gate"]["minimum_material_nonlinear_draws_per_target"])
    for target in protocol["targets"]:
        selected = [row for row in panels if row["target_key"] == target["target_key"]]
        conflicts = sum(bool(row["material_rank_conflict"]) for row in selected)
        gains = sum(bool(row["material_nonlinear_gain"]) for row in selected)
        target_rows.append(
            {
                "target_key": str(target["target_key"]),
                "panel_count": len(selected),
                "material_rank_conflict_draw_count": conflicts,
                "material_nonlinear_gain_draw_count": gains,
                "mean_linear_to_envelope_spearman": float(np.mean([row["linear_to_envelope_spearman"] for row in selected if row["linear_to_envelope_spearman"] is not None])),
                "mean_linear_top1_regret_r2": float(np.mean([row["linear_top1_regret_r2"] for row in selected])),
                "mean_maximum_nonlinear_gain_r2": float(np.mean([row["maximum_nonlinear_gain_r2"] for row in selected])),
                "gate_pass": bool(conflicts >= required_conflicts and gains >= required_gains),
            }
        )
    warning_types = sorted(
        {
            warning
            for cell in cells
            for head in cell["heads"].values()
            for warning in list(head["final_warning_types"])
        }
    )
    passed = bool(all(row["gate_pass"] for row in target_rows))
    return {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_reference_replication_v1",
        "panel_summaries": panels,
        "target_summaries": target_rows,
        "warning_types": warning_types,
        "decision": {
            "reference_gate_pass": passed,
            "oi_stage_authorized": passed,
            "next_stage": "run_frozen_continuous_oi_comparison" if passed else "stop_unstable_reference_benchmark",
            "method_promotion": "none_reference_only",
        },
        "overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Replicated early-mean reference benchmark", "",
        "| Target | Draw | LP rho | LP regret | Max nonlinear gain | LP winner | Envelope winner | Conflict |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in summary["panel_summaries"]:
        rho = "undefined" if row["linear_to_envelope_spearman"] is None else f"{row['linear_to_envelope_spearman']:.3f}"
        lines.append(
            f"| {row['target_key']} | {row['draw_seed']} | {rho} | {row['linear_top1_regret_r2']:.4f} | "
            f"{row['maximum_nonlinear_gain_r2']:.4f} | {', '.join(row['linear_best_candidate_ids'])} | "
            f"{', '.join(row['envelope_best_candidate_ids'])} | {str(row['material_rank_conflict']).lower()} |"
        )
    lines.extend(["", "## Gate", ""])
    for row in summary["target_summaries"]:
        lines.append(
            f"- {row['target_key']}: {row['material_rank_conflict_draw_count']}/5 conflicting draws, "
            f"{row['material_nonlinear_gain_draw_count']}/5 material nonlinear-gain draws; pass={str(row['gate_pass']).lower()}."
        )
    lines.extend([
        f"- OI stage authorized: **{str(summary['decision']['oi_stage_authorized']).lower()}**.",
        f"- Warning types: `{json.dumps(summary['warning_types'])}`.",
        "- No OverlapIndex method was evaluated or promoted in this stage.", "",
    ])
    return "\n".join(lines)


def _identity(source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": dict(source_hashes),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def _validate_grid(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> None:
    expected = int(protocol["expected_grid"]["reference_cell_count"])
    if len(cells) != expected or len({str(row["cell_id"]) for row in cells}) != expected:
        raise ValueError("early-mean reference cell grid changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    output_root = output_root.resolve()
    protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("early-mean reference execution requires a clean committed source tree")
    extract.validate_bank()
    source_hashes = _source_hashes()
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("early-mean reference terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    identity = _identity(source_hashes)
    checkpoint_path = output_root / "checkpoint.json"
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError("early-mean reference checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("early-mean reference checkpoint identity changed")
    else:
        checkpoint = {"schema_version": 1, "identity": identity, "cells": []}
    output_root.mkdir(parents=True, exist_ok=True)
    cohort = prepare.load_cohort()
    _manifest, matrices = extract.load_bank()
    complete = {str(row["cell_id"]) for row in checkpoint["cells"]}
    expected = int(protocol["expected_grid"]["reference_cell_count"])
    for target in protocol["targets"]:
        for draw_seed in target["draw_seeds"]:
            for backbone_config in protocol["backbones"]:
                backbone = str(backbone_config["backbone_id"])
                cell_id = f"{target['target_key']}::{draw_seed}::{backbone}"
                if cell_id in complete:
                    continue
                checkpoint["cells"].append(
                    evaluate_cell(protocol, target, int(draw_seed), backbone, cohort, matrices)
                )
                base._finite_json(checkpoint)
                base._atomic_json(checkpoint_path, checkpoint)
                print(f"reference {len(checkpoint['cells'])}/{expected}: {cell_id}", flush=True)
    _validate_grid(protocol, checkpoint["cells"])
    summary = summarize(protocol, checkpoint["cells"])
    raw = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_reference_replication_v1",
        "protocol": protocol,
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_environment": {key: os.environ.get(key) for key in base.THREAD_KEYS},
        },
        "cells": checkpoint["cells"],
        "overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }
    base._finite_json(raw)
    base._finite_json(summary)
    base._atomic_json(output_root / "raw_results.json", raw)
    base._atomic_json(output_root / "summary.json", summary)
    base._atomic_bytes(output_root / "report.md", render_report(summary).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_reference_replication_v1",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "checkpoint_identity": identity,
        "artifacts": {
            name: base.sha256_path(output_root / name)
            for name in ("checkpoint.json", "raw_results.json", "summary.json", "report.md")
        },
        "counts": {"cells": len(checkpoint["cells"]), "panels": len(summary["panel_summaries"]), "overlapindex_fits": 0},
        "git": {"commit": identity["git_commit"], "dirty": False},
        "decision": summary["decision"],
    }
    base._atomic_json(output_root / "manifest.json", manifest)
    terminal = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_reference_replication_v1",
        "manifest_sha256": base.sha256_path(output_root / "manifest.json"),
        "summary_sha256": base.sha256_path(output_root / "summary.json"),
        "decision": summary["decision"],
    }
    base._atomic_json(terminal_path, terminal)
    return validate_artifacts(output_root)


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    protocol = load_protocol()
    terminal = json.loads((output_root / "terminal.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output_root / "checkpoint.json").read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "completed_continuous_early_mean_reference_replication_v1"
        or terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json")
        or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json")
        or terminal.get("decision") != summary.get("decision")
        or manifest.get("source_hashes") != _source_hashes()
    ):
        raise ValueError("early-mean reference artifact identity changed")
    for name, expected in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != expected:
            raise ValueError(f"early-mean reference artifact changed: {name}")
    _validate_grid(protocol, checkpoint["cells"])
    return terminal


def load_results(output_root: Path = DEFAULT_OUTPUT_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_artifacts(output_root)
    return (
        json.loads((output_root / "raw_results.json").read_text(encoding="utf-8")),
        json.loads((output_root / "summary.json").read_text(encoding="utf-8")),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume)
    print(base.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
