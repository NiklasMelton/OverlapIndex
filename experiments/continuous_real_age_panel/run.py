"""Run the frozen real-image continuous shortlist panel."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_shortlist_depth_audit import run as audit_parent
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_top5_bounded_confirmation import run as bounded_parent
from experiments.continuous_winner_fill_shortlist import run as winner_parent
from experiments.continuous_real_age_panel import extract, prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_age_panel" / "results"
METHODS = ("broad_rff", "save_rff_current", "bounded_2k", "linear_probe")
PCA_RANK = 32
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 2026090102
TIE_ATOL = 1.0e-12


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or prepare.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("real age panel protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_real_image_regression_before_outcomes":
        raise ValueError("real age panel protocol is not frozen")
    if tuple(protocol["backbones"]) != extract.BACKBONE_IDS:
        raise ValueError("real age panel backbone roster differs from code")
    if tuple(protocol["replicate_seeds"]) != prepare.REPLICATE_SEEDS:
        raise ValueError("real age panel replicate seeds differ from code")
    if int(protocol["preprocessing"]["pca_rank"]) != PCA_RANK:
        raise ValueError("real age panel PCA rank differs from code")
    return protocol


def _load_embedding_bank(
    input_root: Path, embedding_root: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cohort = prepare.load_cohort(input_root)
    terminal = json.loads((embedding_root / "manifest.json").read_text(encoding="utf-8"))
    if terminal.get("status") != "completed_embedding_extraction":
        raise ValueError("embedding extraction is not complete")
    if tuple(terminal.get("backbone_ids", ())) != extract.BACKBONE_IDS:
        raise ValueError("embedding manifest backbone roster changed")
    if terminal.get("cohort_sha256") != prepare.sha256_path(input_root / "cohort.json"):
        raise ValueError("embedding bank cohort identity changed")
    bank = {}
    for spec in extract.BACKBONE_SPECS:
        extract._validate_cache(embedding_root, spec, cohort)
        matrix = np.load(embedding_root / f"{spec['backbone_id']}.npy", mmap_mode="r", allow_pickle=False)
        bank[str(spec["backbone_id"])] = matrix
    return cohort, bank


def _panel_indices(cohort: Mapping[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    train = [index for index, row in enumerate(cohort["rows"]) if row["split"] == "training" and row["replicate_seed"] == seed]
    test = [index for index, row in enumerate(cohort["rows"]) if row["split"] == "evaluation"]
    if len(train) != 225 or len(test) != 225 or set(train).intersection(test):
        raise ValueError("real age panel split is incomplete or overlapping")
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def _pca_geometry(
    X_train: np.ndarray, X_test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = normalize(np.asarray(X_train, dtype=np.float32), norm="l2")
    test = normalize(np.asarray(X_test, dtype=np.float32), norm="l2")
    transform = PCA(n_components=PCA_RANK, svd_solver="randomized", random_state=int(seed))
    reduced_train = np.asarray(transform.fit_transform(train), dtype=np.float32)
    reduced_test = np.asarray(transform.transform(test), dtype=np.float32)
    if not np.isfinite(reduced_train).all() or not np.isfinite(reduced_test).all():
        raise ValueError("PCA geometry produced nonfinite values")
    return reduced_train, reduced_test, {
        "rank": PCA_RANK,
        "explained_variance_ratio_sum": float(np.sum(transform.explained_variance_ratio_)),
        "fit_rows": len(train),
        "input_dimension": int(train.shape[1]),
    }


def evaluate_candidate(
    backbone_id: str,
    matrix: np.ndarray,
    cohort: Mapping[str, Any],
    seed: int,
    position: int,
) -> dict[str, Any]:
    train_index, test_index = _panel_indices(cohort, seed)
    labels = np.asarray([row["age"] for row in cohort["rows"]], dtype=np.float64)
    started = time.perf_counter()
    X_train, X_test, pca = _pca_geometry(matrix[train_index], matrix[test_index], seed * 100 + position)
    pca_seconds = float(time.perf_counter() - started)
    y_train, y_test = labels[train_index], labels[test_index]
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
        if abs(value - envelope) <= TIE_ATOL
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
                # The two scores were produced by the same shared call. Put
                # the complete charge on broad and zero here so the bounded
                # parent sum charges shared work exactly once.
                "wall_seconds": 0.0,
            },
        },
    }


def run_panel(
    cohort: Mapping[str, Any], bank: Mapping[str, np.ndarray], seed: int
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "family": "utkface_age",
        "candidate_count": len(extract.BACKBONE_IDS),
        "candidates": [
            evaluate_candidate(backbone_id, bank[backbone_id], cohort, seed, position)
            for position, backbone_id in enumerate(extract.BACKBONE_IDS)
        ],
    }


def panel_metric(panel: Mapping[str, Any], method: str) -> dict[str, Any]:
    return bounded_parent.panel_metric(panel, method)


def _paired_interval(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, metric: str, offset: int
) -> dict[str, Any]:
    lookup = {(row["method"], int(row["seed"])): row for row in rows}
    values = np.asarray(
        [lookup[(left, seed)][metric] - lookup[(right, seed)][metric] for seed in prepare.REPLICATE_SEEDS],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    draws = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "left": left, "right": right, "metric": metric,
        "estimate": float(np.mean(values)), "lower_95": float(lower), "upper_95": float(upper),
        "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED + offset,
        "scope": "training_cohort_seed_stability_with_shared_evaluation_cohort",
    }


def summarize(panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    if tuple(int(row["seed"]) for row in panels) != prepare.REPLICATE_SEEDS:
        raise ValueError("real age result panel order/grid changed")
    rows = [panel_metric(panel, method) for panel in panels for method in METHODS]
    summaries = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summaries.append({
            "method": method,
            "panels": len(selected),
            "mean_spearman": float(np.mean([row["spearman"] for row in selected])),
            "mean_top_5_opportunity_regret_r2": float(np.mean([row["top_5_opportunity_regret_r2"] for row in selected])),
            "mean_top_5_mean_utility_regret_r2": float(np.mean([row["top_5_mean_utility_regret_r2"] for row in selected])),
            "mean_top_5_material_precision": float(np.mean([row["top_5_material_precision"] for row in selected])),
            "mean_descriptive_panel_runtime_seconds": float(np.mean([row["descriptive_panel_runtime_seconds"] for row in selected])),
        })
    contrasts = [
        _paired_interval(rows, "bounded_2k", "linear_probe", "top_5_mean_utility_regret_r2", 0),
        _paired_interval(rows, "bounded_2k", "linear_probe", "top_5_opportunity_regret_r2", 100),
        _paired_interval(rows, "bounded_2k", "broad_rff", "top_5_mean_utility_regret_r2", 200),
    ]
    lookup = {(row["left"], row["right"], row["metric"]): row for row in contrasts}
    primary = lookup[("bounded_2k", "linear_probe", "top_5_mean_utility_regret_r2")]
    opportunity = lookup[("bounded_2k", "linear_probe", "top_5_opportunity_regret_r2")]
    bound_active = all(
        len(
            audit_parent._broad_pool(
                winner_parent.ranking_groups(panel, "broad_rff"), 5, 16
            )
        )
        < 16
        for panel in panels
    )
    broad = {int(row["seed"]): row for row in rows if row["method"] == "broad_rff"}
    top1_difference = max(
        abs(row["top_1_opportunity_regret_r2"] - broad[int(row["seed"])]["top_1_opportunity_regret_r2"])
        for row in rows if row["method"] == "bounded_2k"
    )
    checks = [
        {"criterion": "bounded_top5_mean_regret_competitive_with_linear_probe", "observed_upper_95": primary["upper_95"], "required": "<=0", "pass": bool(primary["upper_95"] <= 0.0)},
        {"criterion": "bounded_top5_opportunity_noninferior_to_linear_probe", "observed_upper_95": opportunity["upper_95"], "maximum": 0.01, "pass": bool(opportunity["upper_95"] <= 0.01)},
        {"criterion": "broad_winner_preserved", "observed_max_absolute_difference": top1_difference, "maximum": TIE_ATOL, "pass": bool(top1_difference <= TIE_ATOL)},
        {"criterion": "top2k_bound_active_in_every_panel", "pass": bool(bound_active)},
        {"criterion": "complete_finite_five_panel_grid", "observed_rows": len(rows), "required_rows": 20, "pass": bool(len(rows) == 20 and all(np.isfinite([row["top_5_mean_utility_regret_r2"], row["top_5_opportunity_regret_r2"]]).all() for row in rows))},
    ]
    return {
        "schema_version": 1,
        "status": "completed_real_image_regression_panel",
        "scope": "single_dataset_training_seed_stability_not_external_generalization",
        "panel_count": len(panels),
        "panel_metrics": rows,
        "method_summaries": summaries,
        "contrasts": contrasts,
        "decision": {"status": "pass" if all(row["pass"] for row in checks) else "fail", "checks": checks},
        "best_head_counts": {
            head: sum(head in candidate["best_head_ids"] for panel in panels for candidate in panel["candidates"])
            for head in ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
        },
        "protocol": dict(protocol),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Real-image continuous shortlist panel", "",
        f"Decision: **{summary['decision']['status'].upper()}**.", "",
        "Negative regret contrasts favor the method on the left. This is one real dataset with five disjoint training cohorts and one shared evaluation cohort; it is not an external-generalization claim.", "",
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
    return "\n".join(lines) + "\n"


def run(
    input_root: Path = prepare.DEFAULT_INPUT_ROOT,
    embedding_root: Path = extract.DEFAULT_OUTPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *, resume: bool = False,
) -> dict[str, Any]:
    protocol = _load_protocol()
    cohort, bank = _load_embedding_bank(input_root.resolve(), embedding_root.resolve())
    output_root = output_root.resolve()
    if (output_root / "manifest.json").exists():
        raise FileExistsError("real age terminal result already exists")
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
    raw = {"schema_version": 1, "status": "completed", "protocol": protocol, "panels": panels}
    summary = summarize(panels, protocol)
    report = _report(summary)
    output_root.mkdir(parents=True, exist_ok=True)
    files = {"raw_results.json": raw, "summary.json": summary}
    for name, payload in files.items():
        (output_root / name).write_text(prepare.canonical_json(payload) + "\n", encoding="utf-8")
    (output_root / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": 1, "status": "completed_real_image_regression_panel",
        "implementation_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "protocol_sha256": prepare.sha256_path(PROTOCOL_PATH),
        "input_manifest_sha256": prepare.sha256_path(input_root / "manifest.json"),
        "embedding_manifest_sha256": prepare.sha256_path(embedding_root / "manifest.json"),
        "files": {name: {"sha256": prepare.sha256_path(output_root / name), "bytes": (output_root / name).stat().st_size} for name in ("raw_results.json", "summary.json", "report.md")},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    }
    (output_root / "manifest.json").write_text(prepare.canonical_json(manifest) + "\n", encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=prepare.DEFAULT_INPUT_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=extract.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run(args.input_root, args.embedding_root, args.output_root, resume=args.resume)
    print(prepare.canonical_json(summary["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
