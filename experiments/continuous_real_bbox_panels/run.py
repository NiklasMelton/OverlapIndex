"""Run the frozen GTSRB and Aircraft box-occupancy panels."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import scipy
import sklearn

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_real_age_panel import run as age_parent
from experiments.continuous_real_bbox_panels import prepare
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_shortlist_depth_audit import run as audit_parent
from experiments.continuous_winner_fill_shortlist import run as winner_parent


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_bbox_panels" / "results"
BACKBONES = (
    "dinov2-small",
    "deit-tiny",
    "convnext-tiny",
    "mobilenetv3-large",
    "openclip-vit-b-32",
    "resnet50",
    "efficientnet-b0",
    "swin-tiny",
    "vit-small-16",
    "densenet121",
)
METHODS = ("broad_rff", "save_rff_current", "bounded_2k", "linear_probe")
PCA_RANK = 32
TIE_ATOL = 1.0e-12


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or prepare.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("box-panel protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_independent_bbox_regression_panels_before_outcomes":
        raise ValueError("box-panel protocol is not frozen")
    if tuple(protocol["backbones"]) != BACKBONES:
        raise ValueError("box-panel backbone roster differs from code")
    if tuple(protocol["methods"]) != METHODS:
        raise ValueError("box-panel method set differs from code")
    if tuple(protocol["replicate_seeds"]) != prepare.REPLICATE_SEEDS:
        raise ValueError("box-panel replicate seeds differ from code")
    if int(protocol["preprocessing"]["pca_rank"]) != PCA_RANK:
        raise ValueError("box-panel PCA rank differs from code")
    ancestry = protocol["ancestry"]
    paths = {
        "cohort_sha256": prepare.DEFAULT_OUTPUT_ROOT / "cohort.json",
        "input_manifest_sha256": prepare.DEFAULT_OUTPUT_ROOT / "manifest.json",
    }
    for field, path in paths.items():
        if prepare.sha256_path(path) != ancestry[field]:
            raise ValueError(f"box-panel ancestry changed: {field}")
    if ancestry["registry_sha256"] != prepare.REGISTRY_SHA256:
        raise ValueError("box-panel registry identity differs from code")
    return protocol


def _load_registry(input_root: Path) -> dict[str, Any]:
    path = input_root / "audited_registry.json"
    if prepare.sha256_path(path) != prepare.REGISTRY_SHA256:
        raise ValueError("box-panel registry SHA-256 mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def _record(registry: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    selected = [row for row in registry["datasets"] if row.get("dataset_id") == dataset_id]
    if len(selected) != 1:
        raise ValueError("box-panel dataset registry row is not unique")
    return selected[0]


def _load_bank(record: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    union_ids = tuple(record["cohort_manifest"]["union"]["sample_ids"])
    if len(union_ids) != len(set(union_ids)):
        raise ValueError("box-panel registry union IDs are duplicated")
    bank: dict[str, np.ndarray] = {}
    for backbone in BACKBONES:
        descriptor = record["embeddings"][backbone]
        matrix_path = Path(descriptor["matrix_path"])
        manifest_path = Path(descriptor["manifest_path"])
        if prepare.sha256_path(matrix_path) != descriptor["matrix_sha256"]:
            raise ValueError(f"box-panel matrix hash changed: {backbone}")
        if prepare.sha256_path(manifest_path) != descriptor["manifest_sha256"]:
            raise ValueError(f"box-panel matrix manifest hash changed: {backbone}")
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if matrix.dtype != np.dtype("float32") or matrix.ndim != 2:
            raise ValueError("box-panel matrix shape or dtype changed")
        if tuple(int(value) for value in matrix.shape) != tuple(descriptor["shape"]):
            raise ValueError("box-panel matrix shape differs from registry")
        if len(matrix) != len(union_ids) or not np.isfinite(np.asarray(matrix)).all():
            raise ValueError("box-panel matrix rows are incomplete or nonfinite")
        bank[backbone] = matrix
    return union_ids, bank


def _validate_cohort_protocol(
    cohort: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    if tuple(cohort.get("replicate_seeds", ())) != prepare.REPLICATE_SEEDS:
        raise ValueError("box-panel cohort replicate seeds changed")
    expected_scalars = {
        "target_bins": prepare.TARGET_BINS,
        "training_rows_per_replicate": prepare.TRAIN_ROWS,
        "evaluation_rows": prepare.EVALUATION_ROWS,
        "training_bank_seed": prepare.TRAIN_BANK_SEED,
        "evaluation_seed": prepare.EVALUATION_SEED,
    }
    for field, expected in expected_scalars.items():
        if cohort.get(field) != expected:
            raise ValueError(f"box-panel cohort field changed: {field}")
    rows = cohort.get("datasets")
    if not isinstance(rows, list) or len(rows) != len(prepare.DATASETS):
        raise ValueError("box-panel cohort dataset grid changed")
    by_dataset = {row.get("dataset_id"): row for row in rows if isinstance(row, Mapping)}
    if set(by_dataset) != set(prepare.DATASETS):
        raise ValueError("box-panel cohort dataset identities changed")
    for dataset_id in prepare.DATASETS:
        row = by_dataset[dataset_id]
        expected = protocol["datasets"][dataset_id]
        if row.get("family") != expected["family"]:
            raise ValueError("box-panel target family changed")
        if row.get("target_sha256") != expected["target_sha256"]:
            raise ValueError("box-panel target values changed")
        if row.get("rows_sha256") != expected["cohort_rows_sha256"]:
            raise ValueError("box-panel cohort rows changed")
        if prepare.sha256_payload(row.get("rows")) != row.get("rows_sha256"):
            raise ValueError("box-panel cohort row payload is malformed")
    return by_dataset


def _panel_rows(dataset: Mapping[str, Any], seed: int) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    training = [
        row for row in dataset["rows"]
        if row["split"] == "training" and int(row["replicate_seed"]) == int(seed)
    ]
    evaluation = [row for row in dataset["rows"] if row["split"] == "evaluation"]
    if len(training) != prepare.TRAIN_ROWS or len(evaluation) != prepare.EVALUATION_ROWS:
        raise ValueError("box-panel train/evaluation cohort is incomplete")
    if {row["sample_id"] for row in training}.intersection(row["sample_id"] for row in evaluation):
        raise ValueError("box-panel train/evaluation cohorts overlap")
    return training, evaluation


def _matrix_rows(union_ids: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    positions = {sample_id: index for index, sample_id in enumerate(union_ids)}
    try:
        result = np.asarray([positions[str(row["sample_id"])] for row in rows], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("box-panel cohort sample is absent from the embedding union") from exc
    return result


def evaluate_candidate(
    backbone: str,
    matrix: np.ndarray,
    union_ids: Sequence[str],
    training: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    seed: int,
    position: int,
) -> dict[str, Any]:
    train_rows = _matrix_rows(union_ids, training)
    eval_rows = _matrix_rows(union_ids, evaluation)
    y_train = np.asarray([row["target"] for row in training], dtype=np.float64)
    y_eval = np.asarray([row["target"] for row in evaluation], dtype=np.float64)
    pca_started = time.perf_counter()
    X_train, X_eval, pca = age_parent._pca_geometry(
        matrix[train_rows], matrix[eval_rows], seed * 100 + position
    )
    pca_seconds = float(time.perf_counter() - pca_started)
    heads = residual.evaluate_heads(X_train, y_train, X_eval, y_eval, seed * 100 + position)
    selector = shared.evaluate_shared_pair(
        X_train,
        np.zeros((len(X_train), 2), dtype=np.float32),
        y_train,
        seed,
        position,
    )
    head_scores = {head: float(row["test_r2"]) for head, row in heads.items()}
    envelope = max(head_scores.values())
    return {
        "candidate_id": backbone,
        "pca": pca,
        "pca_wall_seconds": pca_seconds,
        "heads": heads,
        "best_head_envelope_r2": float(envelope),
        "best_head_ids": sorted(
            head for head, score in head_scores.items()
            if abs(score - envelope) <= TIE_ATOL
        ),
        "selectors": {
            "rff64_mean8_k4": {
                "kernel_alignment": selector["broad_rff"]["kernel_alignment"],
                "wall_seconds": float(selector["wall_seconds"]),
            },
            "save_rffmean_k4": {
                "current_common32": selector["save_rff_current"],
                "wall_seconds": 0.0,
            },
        },
    }


def run_panel(
    dataset: Mapping[str, Any],
    union_ids: Sequence[str],
    bank: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    training, evaluation = _panel_rows(dataset, seed)
    return {
        "seed": int(seed),
        "family": str(dataset["family"]),
        "dataset_group": str(dataset["dataset_group"]),
        "candidate_count": len(BACKBONES),
        "candidates": [
            evaluate_candidate(
                backbone, bank[backbone], union_ids, training, evaluation, seed, position
            )
            for position, backbone in enumerate(BACKBONES)
        ],
    }


def ranking_groups(panel: Mapping[str, Any], method: str) -> list[tuple[int, ...]]:
    if method in {"broad_rff", "save_rff_current", "linear_probe"}:
        return winner_parent.ranking_groups(panel, method)
    if method == "bounded_2k":
        return audit_parent.ranking_groups(panel, method, 5)
    raise ValueError(f"unknown box-panel method {method!r}")


def panel_metric(panel: Mapping[str, Any], method: str) -> dict[str, Any]:
    candidates = panel["candidates"]
    reference = np.asarray(
        [candidate["best_head_envelope_r2"] for candidate in candidates], dtype=np.float64
    )
    if reference.shape != (len(BACKBONES),) or not np.isfinite(reference).all():
        raise ValueError("box-panel reference envelope is incomplete")
    groups = ranking_groups(panel, method)
    ordinal = winner_parent._ordinal_scores(groups, len(candidates))
    result: dict[str, Any] = {
        "seed": int(panel["seed"]),
        "family": str(panel["family"]),
        "dataset_group": str(panel["dataset_group"]),
        "panel_id": f"{panel['family']}__s{int(panel['seed'])}",
        "method": method,
        "rank_groups": [[candidates[index]["candidate_id"] for index in group] for group in groups],
        "spearman": winner_parent._spearman(reference, ordinal),
        "material_pair_concordance": winner_parent._material_concordance(reference, ordinal),
        "descriptive_panel_runtime_seconds": (
            winner_parent._runtime_seconds(panel, "broad_rff")
            + winner_parent._runtime_seconds(panel, "save_rff_current")
            if method == "bounded_2k"
            else winner_parent._runtime_seconds(panel, method)
        ),
    }
    for k in (1, 3, 5):
        for key, value in winner_parent._top_k_metrics(reference, groups, k).items():
            result[f"top_{k}_{key}"] = value
    return result


def summarize(panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = [
        (dataset, seed)
        for dataset in prepare.DATASETS
        for seed in prepare.REPLICATE_SEEDS
    ]
    observed = [(panel["dataset_group"], int(panel["seed"])) for panel in panels]
    if observed != expected:
        raise ValueError("box-panel result grid or order changed")
    rows = [panel_metric(panel, method) for panel in panels for method in METHODS]
    summaries = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "panel_count": len(selected),
                "mean_spearman": float(np.mean([row["spearman"] for row in selected])),
                "mean_top_1_opportunity_regret_r2": float(
                    np.mean([row["top_1_opportunity_regret_r2"] for row in selected])
                ),
                "mean_top_5_mean_utility_regret_r2": float(
                    np.mean([row["top_5_mean_utility_regret_r2"] for row in selected])
                ),
                "mean_top_5_material_precision": float(
                    np.mean([row["top_5_material_precision"] for row in selected])
                ),
            }
        )
    return {
        "schema_version": 1,
        "status": "completed_independent_bbox_regression_panels",
        "scope": "two_new_dataset_groups_training_cohort_seed_stability",
        "panel_count": len(panels),
        "panel_metrics": rows,
        "method_summaries": summaries,
        "best_head_counts": {
            head: sum(
                head in candidate["best_head_ids"]
                for panel in panels
                for candidate in panel["candidates"]
            )
            for head in ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
        },
        "protocol": dict(protocol),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Independent box-occupancy continuous panels",
        "",
        "These are descriptive new-dataset panels. No reliability-gate decision is made here.",
        "",
        "| Method | Spearman | Top-1 regret | Top-5 regret | Top-5 precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["method_summaries"]:
        lines.append(
            f"| {row['method']} | {row['mean_spearman']:.4f} | "
            f"{row['mean_top_1_opportunity_regret_r2']:.5f} | "
            f"{row['mean_top_5_mean_utility_regret_r2']:.5f} | "
            f"{row['mean_top_5_material_precision']:.3f} |"
        )
    lines.extend(["", "## Best reference-head counts", ""])
    lines.extend(f"- `{head}`: {count}" for head, count in summary["best_head_counts"].items())
    return "\n".join(lines) + "\n"


def run(
    input_root: Path = prepare.DEFAULT_OUTPUT_ROOT,
    fused_input_root: Path = prepare.DEFAULT_FUSED_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    protocol = _load_protocol()
    cohort = prepare.load_cohort(input_root.resolve())
    registry = _load_registry(fused_input_root.resolve())
    by_dataset = _validate_cohort_protocol(cohort, protocol)
    output_root = output_root.resolve()
    if (output_root / "manifest.json").exists():
        raise FileExistsError("box-panel terminal artifact already exists")
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    for dataset_id in prepare.DATASETS:
        dataset = by_dataset[dataset_id]
        record = _record(registry, dataset_id)
        union_ids, bank = _load_bank(record)
        for seed in prepare.REPLICATE_SEEDS:
            path = checkpoints / f"{dataset_id}__seed_{seed}.json"
            if path.exists():
                if not resume:
                    raise FileExistsError(f"box-panel checkpoint exists: {path.name}")
                panel = json.loads(path.read_text(encoding="utf-8"))
            else:
                panel = run_panel(dataset, union_ids, bank, seed)
                path.write_text(prepare.canonical_json(panel) + "\n", encoding="utf-8")
            panels.append(panel)
    raw = {"schema_version": 1, "status": "completed", "protocol": protocol, "panels": panels}
    summary = summarize(panels, protocol)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, payload in (("raw_results.json", raw), ("summary.json", summary)):
        (output_root / name).write_text(prepare.canonical_json(payload) + "\n", encoding="utf-8")
    (output_root / "report.md").write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed_independent_bbox_regression_panels",
        "implementation_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "protocol_sha256": prepare.sha256_path(PROTOCOL_PATH),
        "registry_sha256": prepare.REGISTRY_SHA256,
        "cohort_sha256": prepare.sha256_path(input_root / "cohort.json"),
        "files": {
            name: {
                "sha256": prepare.sha256_path(output_root / name),
                "bytes": (output_root / name).stat().st_size,
            }
            for name in ("raw_results.json", "summary.json", "report.md")
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    (output_root / "manifest.json").write_text(
        prepare.canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=prepare.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fused-input-root", type=Path, default=prepare.DEFAULT_FUSED_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.input_root, args.fused_input_root, args.output_root, resume=args.resume)
    print(prepare.canonical_json({"status": summary["status"], "panel_count": summary["panel_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
