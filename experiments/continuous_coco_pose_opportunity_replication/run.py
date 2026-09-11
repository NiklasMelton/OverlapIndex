"""Run the frozen COCO four-head opportunity replication."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn

from experiments.continuous_coco_pose_opportunity_replication import extract, prepare
from experiments.continuous_fresh_structured_opportunity_discovery import run as source
from experiments.continuous_real_opportunity_stratified_discovery import run as opportunity
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_structured_head_discovery import run as geometry


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts" / "continuous_coco_pose_opportunity_replication" / "results"
)


def load_protocol() -> dict[str, Any]:
    protocol = prepare.load_protocol()
    if tuple(protocol["backbones"]["scout"]) != extract.BACKBONES:
        raise ValueError("COCO pose scout roster differs from extraction code")
    if tuple(protocol["backbones"]["reserved"]) != geometry.RESERVED_BACKBONES:
        raise ValueError("COCO pose reserved roster differs from frozen implementation")
    if tuple(protocol["heads"]["families"]) != source.HEADS:
        raise ValueError("COCO pose head grid changed")
    if int(protocol["preprocessing"]["pca_rank"]) != geometry.PCA_RANK:
        raise ValueError("COCO pose PCA rank changed")
    grid = protocol["expected_grid"]
    if [int(grid[key]) for key in ("target_count", "panel_count", "candidate_rows", "overlapindex_fit_count")] != [3, 9, 45, 0]:
        raise ValueError("COCO pose expected grid changed")
    return protocol


def _indices(
    cohort: Mapping[str, Any], target_id: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = cohort["target_cohorts"][target_id]
    train = [row for row in rows if row["source_split"] == "train" and int(row["replicate_seed"]) == seed]
    evaluation = [row for row in rows if row["source_split"] == "val"]
    if (
        len(train) != prepare.TRAIN_ROWS
        or len(evaluation) != prepare.EVALUATION_ROWS
        or set(row["matrix_index"] for row in train).intersection(
            row["matrix_index"] for row in evaluation
        )
    ):
        raise ValueError("COCO pose panel split is incomplete or overlapping")
    return (
        np.asarray([row["matrix_index"] for row in train], dtype=np.int64),
        np.asarray([row["matrix_index"] for row in evaluation], dtype=np.int64),
        np.asarray([row["target"] for row in train], dtype=np.float64),
        np.asarray([row["target"] for row in evaluation], dtype=np.float64),
    )


def run_panel(
    protocol: Mapping[str, Any],
    cohort: Mapping[str, Any],
    matrices: Mapping[str, np.ndarray],
    target_id: str,
    seed: int,
) -> dict[str, Any]:
    target_position = tuple(protocol["dataset"]["targets"]).index(target_id)
    train, evaluation, y_train, y_evaluation = _indices(cohort, target_id, seed)
    candidates = []
    for backbone_position, backbone in enumerate(extract.BACKBONES):
        X_train, X_evaluation, pca = geometry.pca_geometry(
            matrices[backbone],
            train,
            evaluation,
            source._geometry_seed(seed, 0, target_position, backbone_position),
        )
        candidates.append(
            source.evaluate_candidate(
                backbone,
                X_train,
                X_evaluation,
                y_train,
                y_evaluation,
                source._fit_seed(seed, 0, target_position, backbone_position),
                pca,
            )
        )
    return {
        "dataset_id": prepare.DATASET_ID,
        "target_id": target_id,
        "seed": int(seed),
        "candidate_count": len(candidates),
        "split": {"training_rows": len(train), "evaluation_rows": len(evaluation)},
        "candidates": candidates,
    }


def summarize(
    panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if len(panels) != int(protocol["expected_grid"]["panel_count"]):
        raise ValueError("COCO pose panel count is incomplete")
    expected = {
        (target, seed)
        for target in protocol["dataset"]["targets"]
        for seed in prepare.REPLICATE_SEEDS
    }
    observed = {(row["target_id"], int(row["seed"])) for row in panels}
    if observed != expected or len(observed) != len(panels):
        raise ValueError("COCO pose panel identities are incomplete or duplicated")
    normalized = source._normalized_panels(panels)
    opportunity_rows = opportunity.panel_opportunity_rows(normalized, protocol)
    tasks = source._task_summaries(opportunity_rows, protocol)
    admitted = [
        row["target_id"]
        for row in tasks
        if row["stratum"] == "rank_changing_nonlinear_opportunity"
    ]
    passed = bool(admitted)
    return {
        "schema_version": 1,
        "status": (
            "completed_coco_pose_opportunity_replication_admitted"
            if passed
            else "completed_coco_pose_opportunity_replication_not_admitted"
        ),
        "scope": dict(protocol["scope"]),
        "task_summaries": tasks,
        "admitted_targets": admitted,
        "dataset_gate_passed": passed,
        "combined_two_dataset_gate_passed": passed,
        "prior_admitted_dataset": protocol["prior_evidence"]["admitted_dataset"],
        "reserved_backbones_evaluated": [],
        "overlapindex_fit_count": 0,
        "next_stage": (
            "freeze a separate LSP-plus-COCO ranker comparison before touching reserved backbones"
            if passed
            else "do not run a reserved ranker comparison; the independent replication failed"
        ),
    }


def report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# COCO pose opportunity replication",
        "",
        "This prospectively frozen screen tests whether the prior LSP pose opportunity replicates on independently sourced COCO images. It fits no OverlapIndex method.",
        "",
        "| Target | Stratum | Best-head R2 | Envelope span | Material nonlinear | LP winner change | LP top-1 regret | Material pair reversals | LP top-3 miss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["task_summaries"]:
        lines.append(
            f"| {row['target_id']} | {row['stratum']} | {row['median_candidate_best_head_r2']:.4f} | "
            f"{row['median_panel_envelope_span_r2']:.4f} | {row['material_nonlinear_candidate_fraction']:.3f} | "
            f"{row['linear_winner_change_rate']:.3f} | {row['mean_linear_top1_regret_r2']:.5f} | "
            f"{row['mean_material_pair_reversal_rate']:.3f} | {row['linear_top3_miss_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"COCO dataset gate passed: **{summary['dataset_gate_passed']}**.",
            f"Combined LSP-plus-COCO gate passed: **{summary['combined_two_dataset_gate_passed']}**.",
            f"Next stage: {summary['next_stage']}.",
            "",
            "Reserved backbones were not evaluated and OverlapIndex fit count is zero.",
        ]
    )
    return "\n".join(lines) + "\n"


def _source_hashes() -> dict[str, str]:
    paths = {
        "protocol": prepare.PROTOCOL_PATH,
        "prepare": Path(prepare.__file__).resolve(),
        "extract": Path(extract.__file__).resolve(),
        "runner": Path(__file__).resolve(),
        "head_evaluator": Path(residual.__file__).resolve(),
        "opportunity_analysis": Path(opportunity.__file__).resolve(),
        "prior_summary": prepare.PRIOR_SUMMARY,
    }
    return {name: prepare.sha256_path(path) for name, path in paths.items()}


def run(
    input_root: Path = prepare.DEFAULT_INPUT_ROOT,
    embedding_root: Path = extract.DEFAULT_OUTPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    protocol = load_protocol()
    cohort, matrices = extract.load_bank(input_root, embedding_root)
    source_hashes = _source_hashes()
    output_root = output_root.resolve()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        raise FileExistsError("COCO pose terminal result already exists")
    checkpoints = output_root / "checkpoints"
    if checkpoints.exists() and not resume:
        raise FileExistsError("COCO pose checkpoints exist; pass --resume")
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    completed = 0
    for target_id in protocol["dataset"]["targets"]:
        for seed in prepare.REPLICATE_SEEDS:
            checkpoint = checkpoints / f"{target_id}__seed_{seed}.json"
            if checkpoint.exists():
                if not resume:
                    raise FileExistsError(f"COCO pose checkpoint exists: {checkpoint.name}")
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                if payload.get("source_hashes") != source_hashes:
                    raise ValueError(f"COCO pose stale checkpoint: {checkpoint.name}")
                panel = payload["panel"]
            else:
                started = time.perf_counter()
                panel = run_panel(protocol, cohort, matrices, target_id, seed)
                prepare._atomic_json(
                    checkpoint,
                    {
                        "schema_version": 1,
                        "source_hashes": source_hashes,
                        "panel_wall_seconds": float(time.perf_counter() - started),
                        "panel": panel,
                    },
                )
            panels.append(panel)
            completed += 1
            print(f"completed {completed}/9: {target_id}/seed={seed}", flush=True)
    summary = summarize(panels, protocol)
    raw = {
        "schema_version": 1,
        "status": "completed_coco_pose_opportunity_replication",
        "source_hashes": source_hashes,
        "protocol": protocol,
        "panels": panels,
        "reserved_backbones_evaluated": [],
        "overlapindex_fit_count": 0,
    }
    raw_path, summary_path, report_path = (
        output_root / "raw_results.json",
        output_root / "summary.json",
        output_root / "report.md",
    )
    prepare._atomic_json(raw_path, raw)
    prepare._atomic_json(summary_path, summary)
    prepare._atomic_bytes(report_path, report(summary).encode("utf-8"))
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    manifest = {
        "schema_version": 1,
        "status": summary["status"],
        "source_hashes": source_hashes,
        "artifacts": {
            "raw_results.json": prepare.sha256_path(raw_path),
            "summary.json": prepare.sha256_path(summary_path),
            "report.md": prepare.sha256_path(report_path),
        },
        "counts": {"panels": len(panels), "candidate_rows": 45, "reserved_backbones": 0, "overlapindex_fits": 0},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "combined_two_dataset_gate_passed": summary["combined_two_dataset_gate_passed"],
        "next_stage": summary["next_stage"],
    }
    prepare._atomic_json(terminal, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=prepare.DEFAULT_INPUT_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=extract.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(prepare.canonical_json(run(args.input_root, args.embedding_root, args.output_root, resume=args.resume)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
