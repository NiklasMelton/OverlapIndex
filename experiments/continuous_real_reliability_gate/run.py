"""Run the OI-only leave-dataset-out reliability-gate diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from experiments.continuous_real_pet_area_panel import prepare
from experiments.continuous_real_rank_aggregation import run as parent
from experiments.continuous_top5_bounded_confirmation import run as bounded_parent
from experiments.continuous_winner_fill_shortlist import run as winner_parent


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_reliability_gate"
FEATURE_NAMES = (
    "broad_save_spearman",
    "complete_top5_jaccard",
    "normalized_broad_top5_margin",
    "mean_broad_score",
)
METHODS = (
    "broad_rff",
    "bounded_2k",
    "pool_borda_a050",
    "lodo_reliability_gate",
    "oracle_panel_gate",
    "linear_probe",
)
RIDGE_PENALTY = 1.0
TIE_ATOL = 1.0e-12


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or prepare.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("reliability-gate protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_post_outcome_reliability_gate_before_derived_analysis":
        raise ValueError("reliability-gate protocol is not frozen")
    if tuple(protocol["feature_schema"]) != FEATURE_NAMES:
        raise ValueError("reliability-gate features differ from code")
    if tuple(protocol["methods"]) != METHODS:
        raise ValueError("reliability-gate methods differ from code")
    ancestry = protocol["ancestry"]
    paths = {
        "rank_aggregation_protocol_sha256": parent.PROTOCOL_PATH,
        "rank_aggregation_summary_sha256": parent.DEFAULT_OUTPUT_ROOT / "summary.json",
        "rank_aggregation_manifest_sha256": parent.DEFAULT_OUTPUT_ROOT / "manifest.json",
    }
    for field, path in paths.items():
        if prepare.sha256_path(path) != ancestry[field]:
            raise ValueError(f"reliability-gate ancestry changed: {field}")
    return protocol


def _complete_pool(groups: Sequence[Sequence[int]], depth: int) -> set[int]:
    output: set[int] = set()
    for group in groups:
        output.update(int(index) for index in group)
        if len(output) >= int(depth):
            return output
    raise ValueError("rank groups do not contain the requested depth")


def feature_mapping(panel: Mapping[str, Any]) -> dict[str, float]:
    broad = winner_parent._component_values(panel, "broad_rff")
    save = winner_parent._component_values(panel, "save_rff_current")
    broad_groups = winner_parent._score_groups(broad)
    save_groups = winner_parent._score_groups(save)
    broad_pool = _complete_pool(broad_groups, 5)
    save_pool = _complete_pool(save_groups, 5)
    union = broad_pool.union(save_pool)
    jaccard = float(len(broad_pool.intersection(save_pool)) / len(union))
    ordered = np.sort(broad)[::-1]
    span = float(ordered[0] - ordered[-1])
    margin = 0.0 if span <= np.finfo(np.float64).eps else float(
        max(ordered[4] - ordered[5], 0.0) / span
    )
    concordance = winner_parent._spearman(broad, save)
    if concordance is None:
        raise ValueError("reliability-gate broad/SAVE concordance is undefined")
    result = {
        "broad_save_spearman": float(concordance),
        "complete_top5_jaccard": jaccard,
        "normalized_broad_top5_margin": margin,
        "mean_broad_score": float(np.mean(broad)),
    }
    if tuple(result) != FEATURE_NAMES or not np.isfinite(list(result.values())).all():
        raise ValueError("reliability-gate feature vector is incomplete or nonfinite")
    return result


def feature_vector(panel: Mapping[str, Any]) -> np.ndarray:
    mapping = feature_mapping(panel)
    return np.asarray([mapping[name] for name in FEATURE_NAMES], dtype=np.float64)


def fit_ridge_gate(features: np.ndarray, benefits: np.ndarray) -> dict[str, Any]:
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(benefits, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or y.shape != (len(X),):
        raise ValueError("ridge-gate training matrix has the wrong shape")
    if len(X) < 2 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("ridge-gate training data are incomplete or nonfinite")
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale <= np.finfo(np.float64).eps, 1.0, scale)
    standardized = (X - mean) / scale
    design = np.column_stack([np.ones(len(X), dtype=np.float64), standardized])
    penalty = np.diag([0.0, *([RIDGE_PENALTY] * X.shape[1])])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "training_rows": len(X),
        "ridge_penalty": RIDGE_PENALTY,
    }


def predict_benefit(model: Mapping[str, Any], features: np.ndarray) -> float:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if values.shape != mean.shape or mean.shape != scale.shape or scale.shape != coefficients.shape:
        raise ValueError("ridge-gate prediction schema changed")
    return float(model["intercept"] + ((values - mean) / scale) @ coefficients)


def _metric_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    output = {(str(row["method"]), str(row["panel_id"])): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("reliability-gate metric rows are duplicated")
    return output


def fit_lodo_models(
    panels: Sequence[Mapping[str, Any]], fixed_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    lookup = _metric_lookup(fixed_rows)
    groups = {str(panel["dataset_group"]) for panel in panels}
    if groups != {"utkface", "oxford_pet"}:
        raise ValueError("reliability-gate dataset groups changed")
    models = {}
    for held_out in sorted(groups):
        training = [panel for panel in panels if panel["dataset_group"] != held_out]
        X, y = [], []
        for panel in training:
            panel_id = f"{panel['family']}__s{int(panel['seed'])}"
            broad = lookup[("broad_rff", panel_id)]
            blend = lookup[("pool_borda_a050", panel_id)]
            X.append(feature_vector(panel))
            y.append(
                float(broad["top_5_mean_utility_regret_r2"])
                - float(blend["top_5_mean_utility_regret_r2"])
            )
        model = fit_ridge_gate(np.asarray(X), np.asarray(y))
        model["held_out_dataset_group"] = held_out
        model["feature_names"] = list(FEATURE_NAMES)
        models[held_out] = model
    return models


def summarize(panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    fixed_methods = ("broad_rff", "bounded_2k", "pool_borda_a050", "linear_probe")
    fixed_rows = [parent.panel_metric(panel, method) for panel in panels for method in fixed_methods]
    lookup = _metric_lookup(fixed_rows)
    models = fit_lodo_models(panels, fixed_rows)
    gate_rows, oracle_rows, predictions = [], [], []
    for panel in panels:
        panel_id = f"{panel['family']}__s{int(panel['seed'])}"
        features = feature_mapping(panel)
        predicted = predict_benefit(
            models[str(panel["dataset_group"])],
            np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64),
        )
        selected = "pool_borda_a050" if predicted > 0.0 else "broad_rff"
        groups = (
            parent.pool_borda_groups(panel, 0.5)
            if selected == "pool_borda_a050"
            else bounded_parent.ranking_groups(panel, "broad_rff")
        )
        gate = parent._custom_panel_metric(panel, "lodo_reliability_gate", groups)
        gate["selected_method"] = selected
        gate["predicted_blend_benefit"] = predicted
        gate_rows.append(gate)
        broad_regret = float(lookup[("broad_rff", panel_id)]["top_5_mean_utility_regret_r2"])
        blend_regret = float(lookup[("pool_borda_a050", panel_id)]["top_5_mean_utility_regret_r2"])
        oracle_selected = "pool_borda_a050" if blend_regret < broad_regret - TIE_ATOL else "broad_rff"
        oracle_groups = (
            parent.pool_borda_groups(panel, 0.5)
            if oracle_selected == "pool_borda_a050"
            else bounded_parent.ranking_groups(panel, "broad_rff")
        )
        oracle = parent._custom_panel_metric(panel, "oracle_panel_gate", oracle_groups)
        oracle["selected_method"] = oracle_selected
        oracle_rows.append(oracle)
        predictions.append(
            {
                "panel_id": panel_id,
                "family": str(panel["family"]),
                "dataset_group": str(panel["dataset_group"]),
                "features": features,
                "predicted_blend_benefit": predicted,
                "selected_method": selected,
                "oracle_selected_method": oracle_selected,
                "decision_correct": selected == oracle_selected,
            }
        )
    rows = [*fixed_rows, *gate_rows, *oracle_rows]
    rows.sort(key=lambda row: (row["family"], int(row["seed"]), METHODS.index(row["method"])))
    method_summaries = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 15:
            raise ValueError(f"reliability-gate method {method!r} is incomplete")
        method_summaries.append(
            {
                "method": method,
                "panel_count": len(selected),
                "mean_spearman": float(np.mean([row["spearman"] for row in selected])),
                "mean_top_1_opportunity_regret_r2": float(
                    np.mean([row["top_1_opportunity_regret_r2"] for row in selected])
                ),
                "mean_top_1_exact_best_coverage": float(
                    np.mean([row["top_1_exact_best_coverage"] for row in selected])
                ),
                "mean_top_5_opportunity_regret_r2": float(
                    np.mean([row["top_5_opportunity_regret_r2"] for row in selected])
                ),
                "mean_top_5_mean_utility_regret_r2": float(
                    np.mean([row["top_5_mean_utility_regret_r2"] for row in selected])
                ),
                "mean_top_5_material_precision": float(
                    np.mean([row["top_5_material_precision"] for row in selected])
                ),
            }
        )
    contrasts = [
        parent._paired_interval(rows, "lodo_reliability_gate", right, metric, index * 100)
        for index, (right, metric) in enumerate(
            (
                ("broad_rff", "top_5_mean_utility_regret_r2"),
                ("pool_borda_a050", "top_5_mean_utility_regret_r2"),
                ("oracle_panel_gate", "top_5_mean_utility_regret_r2"),
                ("linear_probe", "top_5_mean_utility_regret_r2"),
                ("broad_rff", "top_5_opportunity_regret_r2"),
                ("broad_rff", "top_1_opportunity_regret_r2"),
            )
        )
    ]
    lookup_contrast = {(row["right"], row["metric"]): row for row in contrasts}
    primary = lookup_contrast[("broad_rff", "top_5_mean_utility_regret_r2")]
    top5 = lookup_contrast[("broad_rff", "top_5_opportunity_regret_r2")]
    top1 = lookup_contrast[("broad_rff", "top_1_opportunity_regret_r2")]
    checks = [
        {
            "criterion": "reliability_gate_improves_broad_top5_mean_regret",
            "observed_upper_95": primary["upper_95"],
            "required": "<0",
            "pass": bool(primary["upper_95"] < 0.0),
        },
        {
            "criterion": "gate_top5_opportunity_noninferior_to_broad",
            "observed_upper_95": top5["upper_95"],
            "maximum": 0.01,
            "pass": bool(top5["upper_95"] <= 0.01),
        },
        {
            "criterion": "gate_top1_opportunity_noninferior_to_broad",
            "observed_upper_95": top1["upper_95"],
            "maximum": 0.01,
            "pass": bool(top1["upper_95"] <= 0.01),
        },
        {
            "criterion": "complete_finite_gate_grid",
            "observed_rows": len(rows),
            "observed_predictions": len(predictions),
            "pass": bool(
                len(rows) == 90
                and len(predictions) == 15
                and all(np.isfinite(row["predicted_blend_benefit"]) for row in predictions)
            ),
        },
    ]
    return {
        "schema_version": 1,
        "status": "completed_post_outcome_reliability_gate",
        "scope": "developmental_two_dataset_group_gate_not_confirmation",
        "gate_models": models,
        "gate_predictions": predictions,
        "panel_method_rows": rows,
        "method_summaries": method_summaries,
        "contrasts": contrasts,
        "decision": {
            "status": "pass" if all(row["pass"] for row in checks) else "fail",
            "checks": checks,
        },
        "protocol": dict(protocol),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous-OI reliability gate",
        "",
        f"Decision: **{summary['decision']['status'].upper()}**.",
        "",
        "This is a post-outcome two-dataset-group diagnostic. Negative regret contrasts favor the method on the left.",
        "",
        "| Method | Spearman | Top-1 regret | Top-1 exact | Top-5 regret | Top-5 precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["method_summaries"]:
        lines.append(
            f"| {row['method']} | {row['mean_spearman']:.4f} | "
            f"{row['mean_top_1_opportunity_regret_r2']:.5f} | "
            f"{row['mean_top_1_exact_best_coverage']:.3f} | "
            f"{row['mean_top_5_mean_utility_regret_r2']:.5f} | "
            f"{row['mean_top_5_material_precision']:.3f} |"
        )
    lines.extend(["", "## Gate selections", ""])
    for group in ("utkface", "oxford_pet"):
        selected = [row for row in summary["gate_predictions"] if row["dataset_group"] == group]
        blends = sum(row["selected_method"] == "pool_borda_a050" for row in selected)
        correct = sum(row["decision_correct"] for row in selected)
        lines.append(f"- `{group}`: blend `{blends}/{len(selected)}`; oracle decision agreement `{correct}/{len(selected)}`")
    lines.extend(["", "## Checks", ""])
    lines.extend(
        f"- {'PASS' if row['pass'] else 'FAIL'}: `{row['criterion']}`"
        for row in summary["decision"]["checks"]
    )
    return "\n".join(lines) + "\n"


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = _load_protocol()
    panels = parent.load_panels(parent._load_protocol())
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("reliability-gate output already exists")
    summary = summarize(panels, protocol)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "summary.json").write_text(
        prepare.canonical_json(summary) + "\n", encoding="utf-8"
    )
    (output_root / "report.md").write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed_post_outcome_reliability_gate",
        "implementation_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "protocol_sha256": prepare.sha256_path(PROTOCOL_PATH),
        "files": {
            name: {
                "sha256": prepare.sha256_path(output_root / name),
                "bytes": (output_root / name).stat().st_size,
            }
            for name in ("summary.json", "report.md")
        },
    }
    (output_root / "manifest.json").write_text(
        prepare.canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    summary = run(args.output_root)
    print(prepare.canonical_json(summary["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
