"""Run the frozen V-shaped UTKFace target panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_real_age_panel import extract
from experiments.continuous_real_age_panel import prepare
from experiments.continuous_real_age_panel import run as linear_parent


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_age_nonlinear_panel"
TARGET_CENTER = 44.5
TARGET_SCALE = 44.5


def transformed_target(age: Any) -> float:
    if type(age) is not int or not 0 <= age <= 89:
        raise ValueError("V-shaped age target requires a strict integer age in [0, 89]")
    return float(abs(float(age) - TARGET_CENTER) / TARGET_SCALE)


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or prepare.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("nonlinear real-image protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_controlled_nonlinear_real_image_before_outcomes":
        raise ValueError("nonlinear real-image protocol is not frozen")
    if tuple(protocol["replicate_seeds"]) != prepare.REPLICATE_SEEDS:
        raise ValueError("nonlinear real-image seeds differ from code")
    if float(protocol["target"]["center"]) != TARGET_CENTER or float(protocol["target"]["scale"]) != TARGET_SCALE:
        raise ValueError("nonlinear real-image target differs from code")
    return protocol


def _validate_lineage(protocol: Mapping[str, Any]) -> None:
    ancestry = protocol["ancestry"]
    paths = {
        "embedding_manifest_sha256": extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "linear_panel_summary_sha256": linear_parent.DEFAULT_OUTPUT_ROOT / "summary.json",
        "linear_panel_manifest_sha256": linear_parent.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "linear_panel_protocol_sha256": linear_parent.PROTOCOL_PATH,
    }
    for field, path in paths.items():
        if prepare.sha256_path(path) != ancestry[field]:
            raise ValueError(f"nonlinear real-image ancestry mismatch for {field}")


def evaluate_candidate(
    backbone_id: str,
    matrix: np.ndarray,
    cohort: Mapping[str, Any],
    seed: int,
    position: int,
) -> dict[str, Any]:
    train_index, test_index = linear_parent._panel_indices(cohort, seed)
    targets = np.asarray([transformed_target(row["age"]) for row in cohort["rows"]], dtype=np.float64)
    started = time.perf_counter()
    X_train, X_test, pca = linear_parent._pca_geometry(
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
        head_id for head_id, value in head_scores.items()
        if abs(value - envelope) <= linear_parent.TIE_ATOL
    )
    shared_seconds = float(selector["wall_seconds"])
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
                "wall_seconds": shared_seconds,
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
        "family": "utkface_normalized_absolute_age_deviation",
        "candidate_count": len(extract.BACKBONE_IDS),
        "candidates": [
            evaluate_candidate(backbone_id, bank[backbone_id], cohort, seed, position)
            for position, backbone_id in enumerate(extract.BACKBONE_IDS)
        ],
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Controlled nonlinear real-image panel",
        "",
        f"Decision: **{summary['decision']['status'].upper()}**.",
        "",
        "The target is normalized absolute age deviation from the frozen age-range midpoint: `|age - 44.5| / 44.5`. Negative regret contrasts favor bounded OI.",
        "",
        "| Method | Spearman | Top-5 opportunity regret | Top-5 mean regret | Top-5 material precision | Panel seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["method_summaries"]:
        lines.append(
            f"| {row['method']} | {row['mean_spearman']:.4f} | {row['mean_top_5_opportunity_regret_r2']:.5f} | "
            f"{row['mean_top_5_mean_utility_regret_r2']:.5f} | {row['mean_top_5_material_precision']:.3f} | "
            f"{row['mean_descriptive_panel_runtime_seconds']:.3f} |"
        )
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- {'PASS' if row['pass'] else 'FAIL'}: `{row['criterion']}`" for row in summary["decision"]["checks"])
    lines.extend(["", "## Best reference head counts", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in summary["best_head_counts"].items())
    lines.extend(["", "This is a post-linear-panel developmental stress test on the same real images, not untouched confirmation.", ""])
    return "\n".join(lines)


def run(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False
) -> dict[str, Any]:
    protocol = _load_protocol()
    _validate_lineage(protocol)
    cohort, bank = linear_parent._load_embedding_bank(
        prepare.DEFAULT_INPUT_ROOT, extract.DEFAULT_OUTPUT_ROOT
    )
    output_root = output_root.resolve()
    if (output_root / "manifest.json").exists():
        raise FileExistsError("nonlinear real-image terminal artifact already exists")
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
    summary = linear_parent.summarize(panels, protocol)
    summary["status"] = "completed_controlled_nonlinear_real_image_panel"
    summary["scope"] = "post_linear_panel_developmental_real_image_stress_test"
    raw = {"schema_version": 1, "status": "completed", "protocol": protocol, "panels": panels}
    output_root.mkdir(parents=True, exist_ok=True)
    for name, payload in (("raw_results.json", raw), ("summary.json", summary)):
        (output_root / name).write_text(prepare.canonical_json(payload) + "\n", encoding="utf-8")
    (output_root / "report.md").write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed_controlled_nonlinear_real_image_panel",
        "implementation_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "protocol_sha256": prepare.sha256_path(PROTOCOL_PATH),
        "embedding_manifest_sha256": prepare.sha256_path(extract.DEFAULT_OUTPUT_ROOT / "manifest.json"),
        "files": {
            name: {"sha256": prepare.sha256_path(output_root / name), "bytes": (output_root / name).stat().st_size}
            for name in ("raw_results.json", "summary.json", "report.md")
        },
    }
    (output_root / "manifest.json").write_text(prepare.canonical_json(manifest) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.output_root, resume=args.resume)
    print(prepare.canonical_json(summary["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

