"""Evaluate the frozen five-backbone intermediate-layer by pooling factorial."""

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

from experiments.continuous_early_mean_replication_v1 import reference as early_reference
from experiments.continuous_intermediate_pooling_audit_v1 import run as head_parent
from experiments.continuous_intermediate_pooling_factorial_v2 import extract
from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_real_source_qualification_v1 import prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_intermediate_pooling_factorial_v2" / "audit"
TIE_ATOL = 1.0e-12


def load_protocol() -> dict[str, Any]:
    protocol = extract.load_protocol()
    expected = protocol["expected_grid"]
    if (
        tuple(protocol["reference_heads"]["families"]) != ("linear_ridge", "rbf_svr")
        or tuple(protocol["views"]) != (
            "early_mean", "early_mean_std", "middle_mean", "middle_mean_std", "late_mean", "late_mean_std"
        )
        or int(expected["candidate_cell_count"]) != 300
        or int(expected["ranking_panel_count"]) != 60
    ):
        raise ValueError("intermediate-pooling factorial protocol differs from code")
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
    return early_reference._rows(cohort, target, draw_seed)


def evaluate_cell(
    protocol: Mapping[str, Any], target: Mapping[str, Any], draw_seed: int, backbone: str, view: str,
    cohort: Mapping[str, Any], matrices: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, Any]:
    dataset_id, draw, evaluation = _rows(cohort, target, draw_seed)
    matrix = matrices[dataset_id][backbone][view]
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
        split_seed=base._seed("intermediate-pooling-factorial-v2", target["target_key"], draw_seed, backbone, view),
        protocol=head_protocol,
    )
    scores = {head: float(row["evaluation_r2"]) for head, row in result["heads"].items()}
    envelope = max(scores.values())
    return {
        "cell_id": f"{target['target_key']}::{draw_seed}::{backbone}::{view}",
        "target_key": str(target["target_key"]),
        "dataset_id": dataset_id,
        "draw_seed": int(draw_seed),
        "candidate_id": backbone,
        "view_id": view,
        "training_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in draw]),
        "evaluation_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in evaluation]),
        "head_evaluation_r2": scores,
        "head_inner_best_r2": {head: float(row["inner_best_r2"]) for head, row in result["heads"].items()},
        "envelope_r2": float(envelope),
        "best_head_ids": [head for head, score in scores.items() if score >= envelope - TIE_ATOL],
        **result,
    }


def panel_summary(
    protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], target_key: str, draw_seed: int, view: str,
) -> dict[str, Any]:
    order = [str(row["backbone_id"]) for row in protocol["backbones"]]
    by_id = {
        str(row["candidate_id"]): row
        for row in cells
        if row["target_key"] == target_key and int(row["draw_seed"]) == int(draw_seed) and row["view_id"] == view
    }
    if set(by_id) != set(order):
        raise ValueError("factorial ranking panel is incomplete")
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
    gate = protocol["ranking_diagnostics"]
    conflict = bool(
        regret >= float(gate["material_linear_top1_regret_r2"])
        or (rho is not None and rho <= float(gate["material_linear_to_envelope_spearman_upper"]))
    )
    return {
        "panel_id": f"{target_key}::{draw_seed}::{view}",
        "target_key": target_key,
        "draw_seed": int(draw_seed),
        "view_id": view,
        "depth": view.split("_", 1)[0],
        "pooling": "mean_std" if view.endswith("mean_std") else "mean",
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


def _target_view_summaries(protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    diagnostics = protocol["ranking_diagnostics"]
    for target in protocol["targets"]:
        for view in protocol["views"]:
            selected = [row for row in panels if row["target_key"] == target["target_key"] and row["view_id"] == view]
            conflicts = sum(bool(row["material_rank_conflict"]) for row in selected)
            gains = sum(bool(row["material_nonlinear_gain"]) for row in selected)
            rhos = [float(row["linear_to_envelope_spearman"]) for row in selected if row["linear_to_envelope_spearman"] is not None]
            output.append(
                {
                    "target_key": str(target["target_key"]),
                    "view_id": view,
                    "depth": selected[0]["depth"],
                    "pooling": selected[0]["pooling"],
                    "panel_count": len(selected),
                    "material_rank_conflict_draw_count": conflicts,
                    "material_nonlinear_gain_draw_count": gains,
                    "mean_linear_to_envelope_spearman": float(np.mean(rhos)) if rhos else None,
                    "mean_linear_top1_regret_r2": float(np.mean([row["linear_top1_regret_r2"] for row in selected])),
                    "mean_maximum_nonlinear_gain_r2": float(np.mean([row["maximum_nonlinear_gain_r2"] for row in selected])),
                    "target_view_gate_pass": bool(
                        conflicts >= int(diagnostics["minimum_conflicting_draws_per_target_view"])
                        and gains >= int(diagnostics["minimum_material_nonlinear_draws_per_target_view"])
                    ),
                }
            )
    return output


def _factorial_effects(target_views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pooling = []
    depth = []
    targets = sorted({str(row["target_key"]) for row in target_views})
    for target in targets:
        for depth_id in ("early", "middle", "late"):
            mean = next(row for row in target_views if row["target_key"] == target and row["view_id"] == f"{depth_id}_mean")
            mean_std = next(row for row in target_views if row["target_key"] == target and row["view_id"] == f"{depth_id}_mean_std")
            pooling.append(
                {
                    "target_key": target,
                    "depth": depth_id,
                    "mean_std_minus_mean_spearman": None if mean["mean_linear_to_envelope_spearman"] is None or mean_std["mean_linear_to_envelope_spearman"] is None else float(mean_std["mean_linear_to_envelope_spearman"] - mean["mean_linear_to_envelope_spearman"]),
                    "mean_std_minus_mean_top1_regret_r2": float(mean_std["mean_linear_top1_regret_r2"] - mean["mean_linear_top1_regret_r2"]),
                    "mean_std_minus_mean_nonlinear_gain_r2": float(mean_std["mean_maximum_nonlinear_gain_r2"] - mean["mean_maximum_nonlinear_gain_r2"]),
                    "mean_std_minus_mean_conflicting_draw_count": int(mean_std["material_rank_conflict_draw_count"] - mean["material_rank_conflict_draw_count"]),
                }
            )
        for pooling_id in ("mean", "mean_std"):
            rows = [row for row in target_views if row["target_key"] == target and row["pooling"] == pooling_id]
            depth.append(
                {
                    "target_key": target,
                    "pooling": pooling_id,
                    "by_depth": {
                        row["depth"]: {
                            "mean_spearman": row["mean_linear_to_envelope_spearman"],
                            "mean_top1_regret_r2": row["mean_linear_top1_regret_r2"],
                            "mean_maximum_nonlinear_gain_r2": row["mean_maximum_nonlinear_gain_r2"],
                            "conflicting_draw_count": row["material_rank_conflict_draw_count"],
                        }
                        for row in rows
                    },
                }
            )
    return {"pooling_effects_within_depth": pooling, "depth_profiles_within_pooling": depth}


def _uplift_stability(panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target in sorted({str(row["target_key"]) for row in panels}):
        for view in sorted({str(row["view_id"]) for row in panels}):
            selected = [row for row in panels if row["target_key"] == target and row["view_id"] == view]
            candidates = sorted({str(candidate["candidate_id"]) for row in selected for candidate in row["candidates"]})
            for candidate_id in candidates:
                values = np.asarray(
                    [next(candidate for candidate in row["candidates"] if candidate["candidate_id"] == candidate_id)["nonlinear_gain_r2"] for row in selected],
                    dtype=np.float64,
                )
                output.append(
                    {
                        "target_key": target,
                        "view_id": view,
                        "candidate_id": candidate_id,
                        "draw_count": len(values),
                        "mean_nonlinear_gain_r2": float(np.mean(values)),
                        "standard_deviation_nonlinear_gain_r2": float(np.std(values, ddof=1)),
                        "minimum_nonlinear_gain_r2": float(np.min(values)),
                        "maximum_nonlinear_gain_r2": float(np.max(values)),
                        "material_draw_count": int(np.sum(values >= 0.02)),
                    }
                )
    return output


def summarize(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    panels = [
        panel_summary(protocol, cells, str(target["target_key"]), int(seed), view)
        for target in protocol["targets"]
        for seed in target["draw_seeds"]
        for view in protocol["views"]
    ]
    target_views = _target_view_summaries(protocol, panels)
    nominated = [
        view
        for view in protocol["views"]
        if all(
            next(row for row in target_views if row["target_key"] == target["target_key"] and row["view_id"] == view)["target_view_gate_pass"]
            for target in protocol["targets"]
        )
    ]
    warning_types = sorted(
        {
            warning
            for cell in cells
            for head in cell["heads"].values()
            for warning in list(head["final_warning_types"])
        }
    )
    return {
        "schema_version": 1,
        "status": "completed_continuous_intermediate_pooling_factorial_v2",
        "panel_summaries": panels,
        "target_view_summaries": target_views,
        "factorial_effects": _factorial_effects(target_views),
        "uplift_stability": _uplift_stability(panels),
        "warning_types": warning_types,
        "decision": {
            "cross_target_nominated_views": nominated,
            "cross_target_view_nomination_pass": bool(nominated),
            "next_stage": "fresh_oi_benchmark_for_nominated_view" if nominated else "no_general_layer_pooling_benchmark",
            "same_outcome_oi_authorized": False,
            "method_promotion": "none_reference_only",
        },
        "overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Intermediate-layer × pooling factorial V2", "",
        "| Target | View | Mean LP rho | Mean LP regret | Mean max nonlinear gain | Conflicts | Gains | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["target_view_summaries"]:
        rho = "undefined" if row["mean_linear_to_envelope_spearman"] is None else f"{row['mean_linear_to_envelope_spearman']:.3f}"
        lines.append(
            f"| {row['target_key']} | {row['view_id']} | {rho} | {row['mean_linear_top1_regret_r2']:.4f} | "
            f"{row['mean_maximum_nonlinear_gain_r2']:.4f} | {row['material_rank_conflict_draw_count']}/5 | "
            f"{row['material_nonlinear_gain_draw_count']}/5 | {str(row['target_view_gate_pass']).lower()} |"
        )
    lines.extend([
        "", "## Decision", "",
        f"- Cross-target nominated views: `{json.dumps(summary['decision']['cross_target_nominated_views'])}`.",
        f"- Same-outcome OI authorized: **{str(summary['decision']['same_outcome_oi_authorized']).lower()}**.",
        f"- Warning types: `{json.dumps(summary['warning_types'])}`.",
        "- This reference-only factorial does not promote an OI method.", "",
    ])
    return "\n".join(lines)


def _identity(source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": dict(source_hashes),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def _validate_grid(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> None:
    expected = int(protocol["expected_grid"]["candidate_cell_count"])
    if len(cells) != expected or len({str(row["cell_id"]) for row in cells}) != expected:
        raise ValueError("factorial outcome cell grid changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    output_root = output_root.resolve()
    protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("factorial outcome execution requires a clean committed source tree")
    extract.validate_bank()
    source_hashes = _source_hashes()
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("factorial terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    identity = _identity(source_hashes)
    checkpoint_path = output_root / "checkpoint.json"
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError("factorial checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("factorial checkpoint identity changed")
    else:
        checkpoint = {"schema_version": 1, "identity": identity, "cells": []}
    output_root.mkdir(parents=True, exist_ok=True)
    cohort = prepare.load_cohort()
    _manifest, matrices = extract.load_bank()
    complete = {str(row["cell_id"]) for row in checkpoint["cells"]}
    expected = int(protocol["expected_grid"]["candidate_cell_count"])
    for target in protocol["targets"]:
        for draw_seed in target["draw_seeds"]:
            for view in protocol["views"]:
                for backbone_config in protocol["backbones"]:
                    backbone = str(backbone_config["backbone_id"])
                    cell_id = f"{target['target_key']}::{draw_seed}::{backbone}::{view}"
                    if cell_id in complete:
                        continue
                    checkpoint["cells"].append(
                        evaluate_cell(protocol, target, int(draw_seed), backbone, view, cohort, matrices)
                    )
                    base._finite_json(checkpoint)
                    base._atomic_json(checkpoint_path, checkpoint)
                    print(f"factorial reference {len(checkpoint['cells'])}/{expected}: {cell_id}", flush=True)
    _validate_grid(protocol, checkpoint["cells"])
    summary = summarize(protocol, checkpoint["cells"])
    raw = {
        "schema_version": 1,
        "status": "completed_continuous_intermediate_pooling_factorial_v2",
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
        "status": "completed_continuous_intermediate_pooling_factorial_v2",
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
        "status": "completed_continuous_intermediate_pooling_factorial_v2",
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
        terminal.get("status") != "completed_continuous_intermediate_pooling_factorial_v2"
        or terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json")
        or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json")
        or terminal.get("decision") != summary.get("decision")
        or manifest.get("source_hashes") != _source_hashes()
    ):
        raise ValueError("factorial artifact identity changed")
    for name, expected in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != expected:
            raise ValueError(f"factorial artifact changed: {name}")
    _validate_grid(protocol, checkpoint["cells"])
    if len(summary["panel_summaries"]) != int(protocol["expected_grid"]["ranking_panel_count"]):
        raise ValueError("factorial ranking panel grid changed")
    return terminal


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
