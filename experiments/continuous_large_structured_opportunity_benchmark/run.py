"""Run the frozen 20-target, five-domain four-head opportunity benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn

from experiments.continuous_large_structured_opportunity_benchmark import extract, prepare
from experiments.continuous_real_opportunity_stratified_discovery import run as opportunity
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_structured_head_discovery import run as geometry


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts" / "continuous_large_structured_opportunity_benchmark" / "results"
)
HEADS = ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
NONLINEAR_HEADS = HEADS[1:]
TIE_ATOL = 1.0e-12


def load_protocol() -> dict[str, Any]:
    protocol = prepare.load_protocol()
    if tuple(protocol["backbones"]["scout"]) != extract.BACKBONES:
        raise ValueError("large structured scout roster differs from extraction code")
    if tuple(protocol["backbones"]["reserved"]) != geometry.RESERVED_BACKBONES:
        raise ValueError("large structured reserved roster changed")
    if set(extract.BACKBONES).intersection(protocol["backbones"]["reserved"]):
        raise ValueError("large structured scout and reserved rosters overlap")
    if tuple(protocol["heads"]["families"]) != HEADS:
        raise ValueError("large structured head family grid changed")
    if int(protocol["preprocessing"]["pca_rank"]) != geometry.PCA_RANK:
        raise ValueError("large structured PCA rank changed")
    grid = protocol["expected_grid"]
    checks = (
        int(grid["dataset_count"]) == 5,
        int(grid["target_count"]) == 20,
        int(grid["panel_count"]) == 60,
        int(grid["candidate_rows"]) == 300,
        int(grid["overlapindex_fit_count"]) == 0,
    )
    if not all(checks):
        raise ValueError("large structured expected grid changed")
    return protocol


def load_dataset(
    input_root: Path = prepare.DEFAULT_INPUT_ROOT,
    embedding_root: Path = extract.DEFAULT_OUTPUT_ROOT,
) -> dict[str, dict[str, Any]]:
    cohort, matrices = extract.load_bank(input_root.resolve(), embedding_root.resolve())
    return {
        dataset_id: {
            "rows": cohort["datasets"][dataset_id]["rows"],
            "target_cohorts": cohort["datasets"][dataset_id]["target_cohorts"],
            "matrices": matrices[dataset_id],
        }
        for dataset_id in prepare.DATASETS
    }


def _panel_data(
    data: Mapping[str, Any], target_id: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cohort = data["target_cohorts"][target_id]
    training = [
        row
        for row in cohort
        if row["split"] == "training" and int(row["replicate_seed"]) == seed
    ]
    evaluation = [row for row in cohort if row["split"] == "evaluation"]
    if len(training) != prepare.TRAIN_ROWS or len(evaluation) != prepare.EVALUATION_ROWS:
        raise ValueError("large structured panel cohort is incomplete")
    train_indices = np.asarray([row["matrix_index"] for row in training], dtype=np.int64)
    eval_indices = np.asarray([row["matrix_index"] for row in evaluation], dtype=np.int64)
    if len(set(train_indices).intersection(eval_indices)):
        raise ValueError("large structured panel train/evaluation rows overlap")
    y_train = np.asarray([row["target"] for row in training], dtype=np.float64)
    y_eval = np.asarray([row["target"] for row in evaluation], dtype=np.float64)
    if not np.isfinite(y_train).all() or not np.isfinite(y_eval).all():
        raise ValueError("large structured panel target is nonfinite")
    return train_indices, eval_indices, y_train, y_eval


def _fit_seed(seed: int, dataset_position: int, target_position: int, backbone_position: int) -> int:
    value = int(seed) * 100_000 + dataset_position * 10_000 + target_position * 100 + backbone_position
    if not 0 <= value < 2**32:
        raise ValueError("large structured derived fit seed is outside uint32")
    return value


def _geometry_seed(seed: int, dataset_position: int, target_position: int, backbone_position: int) -> int:
    value = int(seed) * 10_000 + dataset_position * 1_000 + target_position * 10 + backbone_position
    if not 0 <= value < 2**32:
        raise ValueError("large structured derived geometry seed is outside uint32")
    return value


def evaluate_candidate(
    backbone: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fit_seed: int,
    pca: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    heads = residual.evaluate_heads(X_train, y_train, X_test, y_test, fit_seed)
    heldout = {head: float(heads[head]["test_r2"]) for head in HEADS}
    envelope = max(heldout.values())
    winners = [head for head in HEADS if heldout[head] >= envelope - TIE_ATOL]
    return {
        "candidate_id": backbone,
        "pca": dict(pca),
        "heads": heads,
        "heldout_linear_r2": heldout["linear_ridge"],
        "best_head_envelope_r2": float(envelope),
        "best_head_ids": winners,
        "nonlinear_minus_linear_r2": float(
            max(heldout[head] for head in NONLINEAR_HEADS) - heldout["linear_ridge"]
        ),
        "head_fit_wall_seconds": float(time.perf_counter() - started),
    }


def run_panel(
    protocol: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
    dataset_id: str,
    target_id: str,
    seed: int,
) -> dict[str, Any]:
    dataset_position = tuple(protocol["datasets"]).index(dataset_id)
    target_position = tuple(protocol["datasets"][dataset_id]["targets"]).index(target_id)
    data = datasets[dataset_id]
    train, evaluation, y_train, y_eval = _panel_data(data, target_id, seed)
    candidates = []
    for backbone_position, backbone in enumerate(extract.BACKBONES):
        matrix = data["matrices"][backbone]
        if train.max() >= len(matrix) or evaluation.max() >= len(matrix):
            raise ValueError(f"large structured matrix index is outside {dataset_id}/{backbone}")
        X_train, X_test, pca = geometry.pca_geometry(
            matrix,
            train,
            evaluation,
            _geometry_seed(seed, dataset_position, target_position, backbone_position),
        )
        candidates.append(
            evaluate_candidate(
                backbone,
                X_train,
                X_test,
                y_train,
                y_eval,
                _fit_seed(seed, dataset_position, target_position, backbone_position),
                pca,
            )
        )
    return {
        "dataset_id": dataset_id,
        "target_id": target_id,
        "seed": int(seed),
        "candidate_count": len(candidates),
        "split": {"training_rows": len(train), "evaluation_rows": len(evaluation)},
        "candidates": candidates,
    }


def _normalized_panels(panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for panel in panels:
        task_id = f"{panel['dataset_id']}::{panel['target_id']}::scout_five_backbones"
        output.append(
            {
                "panel_id": f"{task_id}::s{panel['seed']}",
                "task_id": task_id,
                "source_id": "large_structured_predeclared",
                "dataset_id": panel["dataset_id"],
                "target_id": panel["target_id"],
                "roster": "scout_five_backbones",
                "seed": int(panel["seed"]),
                "candidates": [
                    {
                        "candidate_id": row["candidate_id"],
                        "heldout_linear": float(row["heldout_linear_r2"]),
                        "envelope": float(row["best_head_envelope_r2"]),
                        "heads": row["heads"],
                    }
                    for row in panel["candidates"]
                ],
            }
        )
    return output


def _task_summaries(
    opportunity_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for task_id in sorted({str(row["task_id"]) for row in opportunity_rows}):
        selected = [row for row in opportunity_rows if row["task_id"] == task_id]
        first = selected[0]
        candidate_count = sum(int(row["candidate_count"]) for row in selected)
        winners = {
            head: sum(int(row["unique_heldout_winner_counts"][head]) for row in selected)
            for head in HEADS
        }
        summary = {
            "task_id": task_id,
            "dataset_id": first["dataset_id"],
            "target_id": first["target_id"],
            "panel_count": len(selected),
            "candidate_row_count": candidate_count,
            "median_candidate_best_head_r2": float(
                np.median([value for row in selected for value in row["candidate_best_head_r2_values"]])
            ),
            "median_panel_envelope_span_r2": float(np.median([row["envelope_span_r2"] for row in selected])),
            "material_nonlinear_candidate_fraction": float(
                sum(int(row["material_uplift_candidate_count"]) for row in selected) / candidate_count
            ),
            "positive_nonlinear_candidate_fraction": float(
                sum(int(row["positive_uplift_candidate_count"]) for row in selected) / candidate_count
            ),
            "linear_winner_change_rate": float(np.mean([not bool(row["linear_exact_best"]) for row in selected])),
            "mean_linear_top1_regret_r2": float(np.mean([row["linear_top1_regret_r2"] for row in selected])),
            "linear_top3_miss_rate": float(
                np.mean([row["linear_top3_opportunity_regret_r2"] > TIE_ATOL for row in selected])
            ),
            "mean_linear_top3_opportunity_regret_r2": float(
                np.mean([row["linear_top3_opportunity_regret_r2"] for row in selected])
            ),
            "mean_pair_order_change_rate": float(np.mean([row["pair_order_change_rate"] for row in selected])),
            "mean_material_pair_reversal_rate": float(
                np.mean([row["material_pair_reversal_rate"] for row in selected])
            ),
            "unique_heldout_winner_counts": winners,
        }
        summary.update(opportunity._classify_task(summary, protocol))
        output.append(summary)
    if len(output) != int(protocol["expected_grid"]["target_count"]):
        raise ValueError("large structured task summary grid is incomplete")
    return output


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Wilson interval counts are invalid")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def summarize(panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    if len(panels) != int(protocol["expected_grid"]["panel_count"]):
        raise ValueError("large structured panel count is incomplete")
    identities = {
        (str(row["dataset_id"]), str(row["target_id"]), int(row["seed"]))
        for row in panels
    }
    expected = {
        (dataset_id, target_id, seed)
        for dataset_id, descriptor in protocol["datasets"].items()
        for target_id in descriptor["targets"]
        for seed in prepare.REPLICATE_SEEDS
    }
    if identities != expected or len(identities) != len(panels):
        raise ValueError("large structured panel identities are incomplete or duplicated")
    if any(tuple(row["candidate_id"] for row in panel["candidates"]) != extract.BACKBONES for panel in panels):
        raise ValueError("large structured panel scout roster changed")
    opportunity_rows = opportunity.panel_opportunity_rows(_normalized_panels(panels), protocol)
    tasks = _task_summaries(opportunity_rows, protocol)
    datasets = []
    for dataset_id in protocol["datasets"]:
        rows = [row for row in tasks if row["dataset_id"] == dataset_id]
        admitted = [row["target_id"] for row in rows if row["stratum"] == "rank_changing_nonlinear_opportunity"]
        datasets.append(
            {
                "dataset_id": dataset_id,
                "target_count": len(rows),
                "admitted_target_count": len(admitted),
                "admitted_target_fraction": float(len(admitted) / len(rows)),
                "admitted_targets": admitted,
                "status": "admit" if admitted else "reject",
            }
        )
    admitted_targets = sum(row["stratum"] == "rank_changing_nonlinear_opportunity" for row in tasks)
    admitted_datasets = sum(row["status"] == "admit" for row in datasets)
    return {
        "schema_version": 1,
        "status": "completed_large_structured_opportunity_benchmark",
        "scope": dict(protocol["scope"]),
        "panel_count": len(panels),
        "candidate_rows": sum(len(row["candidates"]) for row in panels),
        "task_summaries": tasks,
        "dataset_summaries": datasets,
        "prevalence": {
            "admitted_targets": admitted_targets,
            "target_count": len(tasks),
            "admitted_target_fraction": float(admitted_targets / len(tasks)),
            "admitted_target_fraction_wilson95": _wilson(admitted_targets, len(tasks)),
            "admitted_datasets": admitted_datasets,
            "dataset_count": len(datasets),
            "admitted_dataset_fraction": float(admitted_datasets / len(datasets)),
            "admitted_dataset_fraction_wilson95": _wilson(admitted_datasets, len(datasets)),
        },
        "reserved_backbones_evaluated": [],
        "overlapindex_fit_count": 0,
        "benchmark_complete": True,
        "next_stage": "interpret the frozen opportunity prevalence; any OverlapIndex ranker comparison requires a separate protocol",
    }


def report(summary: Mapping[str, Any]) -> str:
    prevalence = summary["prevalence"]
    lines = [
        "# Larger structured continuous opportunity benchmark",
        "",
        "This prospectively frozen target-channel benchmark measures how often a four-head best-head envelope changes five-backbone rankings relative to linear ridge. It does not fit or select an OverlapIndex method.",
        "",
        f"Admitted targets: **{prevalence['admitted_targets']} / {prevalence['target_count']}** "
        f"({prevalence['admitted_target_fraction']:.1%}; Wilson 95% CI "
        f"{prevalence['admitted_target_fraction_wilson95'][0]:.1%}–{prevalence['admitted_target_fraction_wilson95'][1]:.1%}).",
        f"Datasets with at least one admitted target: **{prevalence['admitted_datasets']} / {prevalence['dataset_count']}** "
        f"({prevalence['admitted_dataset_fraction']:.1%}; Wilson 95% CI "
        f"{prevalence['admitted_dataset_fraction_wilson95'][0]:.1%}–{prevalence['admitted_dataset_fraction_wilson95'][1]:.1%}).",
        "",
        "| Dataset | Target | Stratum | Best-head R2 | Envelope span | Material nonlinear | LP winner change | LP top-1 regret | Pair reversals | LP top-3 miss |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["task_summaries"]:
        lines.append(
            f"| {row['dataset_id']} | {row['target_id']} | {row['stratum']} | "
            f"{row['median_candidate_best_head_r2']:.4f} | {row['median_panel_envelope_span_r2']:.4f} | "
            f"{row['material_nonlinear_candidate_fraction']:.3f} | {row['linear_winner_change_rate']:.3f} | "
            f"{row['mean_linear_top1_regret_r2']:.5f} | {row['mean_material_pair_reversal_rate']:.3f} | "
            f"{row['linear_top3_miss_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset prevalence",
            "",
            "| Dataset | Admitted targets | Frozen targets |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["dataset_summaries"]:
        lines.append(f"| {row['dataset_id']} | {row['admitted_target_count']} | {row['target_count']} |")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- All 20 frozen targets are reported; none were replaced after outcomes.",
            "- Reserved-backbone outcomes were not computed.",
            "- No OverlapIndex variant was fit or selected.",
            "- This is development evidence, not untouched-dataset confirmation.",
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
    datasets = load_dataset(input_root, embedding_root)
    source_hashes = _source_hashes()
    output_root = output_root.resolve()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        raise FileExistsError("large structured terminal result already exists")
    checkpoints = output_root / "checkpoints"
    if checkpoints.exists() and not resume:
        raise FileExistsError("large structured checkpoints exist; pass --resume")
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    total = int(protocol["expected_grid"]["panel_count"])
    completed = 0
    for dataset_id, descriptor in protocol["datasets"].items():
        for target_id in descriptor["targets"]:
            for seed in prepare.REPLICATE_SEEDS:
                checkpoint = checkpoints / f"{dataset_id}__{target_id}__seed_{seed}.json"
                if checkpoint.exists():
                    if not resume:
                        raise FileExistsError(f"large structured checkpoint exists: {checkpoint.name}")
                    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if payload.get("source_hashes") != source_hashes:
                        raise ValueError(f"large structured stale checkpoint: {checkpoint.name}")
                    panel = payload["panel"]
                else:
                    started = time.perf_counter()
                    panel = run_panel(protocol, datasets, dataset_id, target_id, seed)
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
                print(f"completed {completed}/{total}: {dataset_id}/{target_id}/seed={seed}", flush=True)
    summary = summarize(panels, protocol)
    raw = {
        "schema_version": 1,
        "status": "completed_large_structured_opportunity_benchmark",
        "source_hashes": source_hashes,
        "protocol": protocol,
        "panels": panels,
        "reserved_backbones_evaluated": [],
        "overlapindex_fit_count": 0,
    }
    raw_path = output_root / "raw_results.json"
    summary_path = output_root / "summary.json"
    report_path = output_root / "report.md"
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
        "counts": {
            "panels": len(panels),
            "candidate_rows": sum(len(row["candidates"]) for row in panels),
            "reserved_backbones": 0,
            "overlapindex_fits": 0,
        },
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
        "benchmark_complete": True,
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
