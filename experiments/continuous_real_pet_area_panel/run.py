"""Run the frozen Oxford-IIIT Pet foreground-occupancy panel."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_real_age_panel import run as parent
from experiments.continuous_real_pet_area_panel import extract, prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_pet_area_panel" / "results"
PCA_RANK = 32


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or prepare.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("pet-area panel protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_natural_real_image_regression_before_outcomes":
        raise ValueError("pet-area panel protocol is not frozen")
    if tuple(protocol["backbones"]) != extract.BACKBONE_IDS:
        raise ValueError("pet-area backbone roster differs from code")
    if tuple(protocol["replicate_seeds"]) != prepare.REPLICATE_SEEDS:
        raise ValueError("pet-area replicate seeds differ from code")
    if int(protocol["preprocessing"]["pca_rank"]) != PCA_RANK:
        raise ValueError("pet-area PCA rank differs from code")
    if protocol["dataset"]["input_manifest_sha256"] != prepare.sha256_path(
        prepare.DEFAULT_INPUT_ROOT / "manifest.json"
    ):
        raise ValueError("pet-area input manifest differs from protocol")
    if protocol["dataset"]["cohort_sha256"] != prepare.sha256_path(
        prepare.DEFAULT_INPUT_ROOT / "cohort.json"
    ):
        raise ValueError("pet-area cohort differs from protocol")
    return protocol


def _load_embedding_bank(
    input_root: Path, embedding_root: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cohort = prepare.load_cohort(input_root)
    terminal = json.loads((embedding_root / "manifest.json").read_text(encoding="utf-8"))
    if terminal.get("status") != "completed_embedding_extraction":
        raise ValueError("pet-area embedding extraction is not complete")
    if tuple(terminal.get("backbone_ids", ())) != extract.BACKBONE_IDS:
        raise ValueError("pet-area embedding backbone roster changed")
    if terminal.get("input_manifest_sha256") != prepare.sha256_path(
        input_root / "manifest.json"
    ):
        raise ValueError("pet-area embedding input identity changed")
    if terminal.get("cohort_sha256") != prepare.sha256_path(input_root / "cohort.json"):
        raise ValueError("pet-area embedding cohort identity changed")
    bank = {}
    for spec in extract.BACKBONE_SPECS:
        extract._validate_cache(embedding_root, spec, cohort)
        bank[str(spec["backbone_id"])] = np.load(
            embedding_root / f"{spec['backbone_id']}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
    return cohort, bank


def _panel_indices(cohort: Mapping[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    train = [
        index
        for index, row in enumerate(cohort["rows"])
        if row["split"] == "training" and row["replicate_seed"] == seed
    ]
    test = [
        index
        for index, row in enumerate(cohort["rows"])
        if row["split"] == "evaluation"
    ]
    if len(train) != 200 or len(test) != 200 or set(train).intersection(test):
        raise ValueError("pet-area panel split is incomplete or overlapping")
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def evaluate_candidate(
    backbone_id: str,
    matrix: np.ndarray,
    cohort: Mapping[str, Any],
    seed: int,
    position: int,
) -> dict[str, Any]:
    train_index, test_index = _panel_indices(cohort, seed)
    targets = np.asarray(
        [row["foreground_fraction"] for row in cohort["rows"]], dtype=np.float64
    )
    started = time.perf_counter()
    X_train, X_test, pca = parent._pca_geometry(
        matrix[train_index], matrix[test_index], seed * 100 + position
    )
    pca_seconds = float(time.perf_counter() - started)
    y_train, y_test = targets[train_index], targets[test_index]
    heads = residual.evaluate_heads(X_train, y_train, X_test, y_test, seed * 100 + position)
    selector = shared.evaluate_shared_pair(
        X_train,
        np.zeros((len(X_train), 2), dtype=np.float32),
        y_train,
        seed,
        position,
    )
    head_scores = {head_id: float(row["test_r2"]) for head_id, row in heads.items()}
    envelope = max(head_scores.values())
    winners = sorted(
        head_id
        for head_id, value in head_scores.items()
        if abs(value - envelope) <= parent.TIE_ATOL
    )
    return {
        "candidate_id": backbone_id,
        "pca": pca,
        "pca_wall_seconds": pca_seconds,
        "heads": heads,
        "best_head_envelope_r2": float(envelope),
        "best_head_ids": winners,
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
    cohort: Mapping[str, Any], bank: Mapping[str, np.ndarray], seed: int
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "family": "oxford_pet_foreground_occupancy",
        "candidate_count": len(extract.BACKBONE_IDS),
        "candidates": [
            evaluate_candidate(backbone_id, bank[backbone_id], cohort, seed, position)
            for position, backbone_id in enumerate(extract.BACKBONE_IDS)
        ],
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Oxford-IIIT Pet foreground-occupancy panel",
        "",
        f"Decision: **{summary['decision']['status'].upper()}**.",
        "",
        "The target is the fraction of official trimap pixels labeled animal foreground or boundary. Negative regret contrasts favor bounded OI.",
        "",
        "| Method | Spearman | Top-5 opportunity regret | Top-5 mean regret | Top-5 material precision | Panel seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["method_summaries"]:
        lines.append(
            f"| {row['method']} | {row['mean_spearman']:.4f} | "
            f"{row['mean_top_5_opportunity_regret_r2']:.5f} | "
            f"{row['mean_top_5_mean_utility_regret_r2']:.5f} | "
            f"{row['mean_top_5_material_precision']:.3f} | "
            f"{row['mean_descriptive_panel_runtime_seconds']:.3f} |"
        )
    lines.extend(["", "## Checks", ""])
    lines.extend(
        f"- {'PASS' if row['pass'] else 'FAIL'}: `{row['criterion']}`"
        for row in summary["decision"]["checks"]
    )
    lines.extend(["", "## Best reference head counts", ""])
    lines.extend(
        f"- `{key}`: {value}" for key, value in summary["best_head_counts"].items()
    )
    lines.extend(
        [
            "",
            "This is one natural real-image target with five disjoint training cohorts and one shared evaluation cohort; it is not an external-dataset generalization claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    input_root: Path = prepare.DEFAULT_INPUT_ROOT,
    embedding_root: Path = extract.DEFAULT_OUTPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    protocol = _load_protocol()
    cohort, bank = _load_embedding_bank(input_root.resolve(), embedding_root.resolve())
    output_root = output_root.resolve()
    if (output_root / "manifest.json").exists():
        raise FileExistsError("pet-area terminal result already exists")
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    for seed in prepare.REPLICATE_SEEDS:
        path = checkpoints / f"seed_{seed}.json"
        if path.exists():
            if not resume:
                raise FileExistsError(f"checkpoint exists for seed {seed}")
            panel = json.loads(path.read_text(encoding="utf-8"))
        else:
            panel = run_panel(cohort, bank, seed)
            path.write_text(prepare.canonical_json(panel) + "\n", encoding="utf-8")
        panels.append(panel)
    summary = parent.summarize(panels, protocol)
    summary["status"] = "completed_natural_real_image_regression_panel"
    summary["scope"] = "single_natural_target_training_seed_stability"
    raw = {"schema_version": 1, "status": "completed", "protocol": protocol, "panels": panels}
    output_root.mkdir(parents=True, exist_ok=True)
    for name, payload in (("raw_results.json", raw), ("summary.json", summary)):
        (output_root / name).write_text(prepare.canonical_json(payload) + "\n", encoding="utf-8")
    (output_root / "report.md").write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed_natural_real_image_regression_panel",
        "implementation_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "protocol_sha256": prepare.sha256_path(PROTOCOL_PATH),
        "input_manifest_sha256": prepare.sha256_path(input_root / "manifest.json"),
        "embedding_manifest_sha256": prepare.sha256_path(embedding_root / "manifest.json"),
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
    parser.add_argument("--input-root", type=Path, default=prepare.DEFAULT_INPUT_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=extract.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    summary = run(
        args.input_root, args.embedding_root, args.output_root, resume=args.resume
    )
    print(prepare.canonical_json(summary["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
