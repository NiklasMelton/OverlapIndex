"""Evaluate winner-preserving reliability-gated Borda filling."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from experiments.continuous_real_bbox_panels import prepare
from experiments.continuous_real_bbox_panels import run as bbox_parent
from experiments.continuous_real_reliability_gate import run as gate_parent
from experiments.continuous_real_reliability_gate_expanded import run as parent
from experiments.continuous_shortlist_depth_audit import run as audit_parent
from experiments.continuous_winner_fill_shortlist import run as winner_parent


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_winner_preserved_gate"
FEATURE_NAMES = gate_parent.FEATURE_NAMES
METHODS = (
    "broad_rff",
    "bounded_2k",
    "winner_preserved_borda_a050",
    "lodo_winner_preserved_gate",
    "oracle_winner_preserved_gate",
    "pool_borda_a050",
    "linear_probe",
)
TIE_ATOL = gate_parent.TIE_ATOL


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or prepare.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("winner-preserved gate protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_winner_preserved_gate_before_derived_analysis":
        raise ValueError("winner-preserved gate protocol is not frozen")
    if tuple(protocol["feature_schema"]) != FEATURE_NAMES:
        raise ValueError("winner-preserved gate feature schema changed")
    if tuple(protocol["methods"]) != METHODS:
        raise ValueError("winner-preserved gate method set changed")
    ancestry = protocol["ancestry"]
    paths = {
        "expanded_protocol_sha256": parent.PROTOCOL_PATH,
        "expanded_summary_sha256": parent.DEFAULT_OUTPUT_ROOT / "summary.json",
        "expanded_manifest_sha256": parent.DEFAULT_OUTPUT_ROOT / "manifest.json",
    }
    for field, path in paths.items():
        if prepare.sha256_path(path) != ancestry[field]:
            raise ValueError(f"winner-preserved gate ancestry changed: {field}")
    return protocol


def winner_preserved_borda_groups(panel: Mapping[str, Any]) -> list[tuple[int, ...]]:
    count = len(panel["candidates"])
    broad = winner_parent._component_values(panel, "broad_rff")
    save = winner_parent._component_values(panel, "save_rff_current")
    broad_groups = winner_parent._score_groups(broad)
    save_groups = winner_parent._score_groups(save)
    winner = tuple(broad_groups[0])
    pool = audit_parent._broad_pool(broad_groups, 5, count)
    pool.difference_update(winner)
    outside = set(range(count)).difference(winner).difference(pool)
    combined = 0.5 * winner_parent._ordinal_scores(broad_groups, count)
    combined += 0.5 * winner_parent._ordinal_scores(save_groups, count)
    groups: list[tuple[int, ...]] = [winner]
    if pool:
        groups.extend(winner_parent._score_groups(combined, pool))
    if outside:
        groups.extend(winner_parent._score_groups(broad, outside))
    if groups[0] != winner or sorted(index for group in groups for index in group) != list(range(count)):
        raise AssertionError("winner-preserved Borda ranking is not a complete partition")
    return groups


def panel_metric(panel: Mapping[str, Any], method: str) -> dict[str, Any]:
    if method in {"broad_rff", "bounded_2k", "pool_borda_a050", "linear_probe"}:
        return parent.panel_metric(panel, method)
    if method == "winner_preserved_borda_a050":
        return parent._custom_panel_metric(panel, method, winner_preserved_borda_groups(panel))
    raise ValueError(f"unknown winner-preserved fixed method {method!r}")


def fit_lodo_models(
    panels: Sequence[Mapping[str, Any]], fixed_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    lookup = parent._lookup(fixed_rows)
    models = {}
    for held_out in parent.GROUPS:
        training = [panel for panel in panels if panel["dataset_group"] != held_out]
        features, benefits = [], []
        for panel in training:
            panel_id = f"{panel['family']}__s{int(panel['seed'])}"
            broad = lookup[("broad_rff", panel_id)]
            alternative = lookup[("winner_preserved_borda_a050", panel_id)]
            features.append(gate_parent.feature_vector(panel))
            benefits.append(
                float(broad["top_5_mean_utility_regret_r2"])
                - float(alternative["top_5_mean_utility_regret_r2"])
            )
        model = gate_parent.fit_ridge_gate(np.asarray(features), np.asarray(benefits))
        model["held_out_dataset_group"] = held_out
        model["feature_names"] = list(FEATURE_NAMES)
        model["training_target"] = "broad_minus_winner_preserved_borda_top5_mean_regret"
        models[held_out] = model
    return models


def summarize(panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    fixed_methods = (
        "broad_rff",
        "bounded_2k",
        "winner_preserved_borda_a050",
        "pool_borda_a050",
        "linear_probe",
    )
    fixed_rows = [panel_metric(panel, method) for panel in panels for method in fixed_methods]
    lookup = parent._lookup(fixed_rows)
    models = fit_lodo_models(panels, fixed_rows)
    gate_rows, oracle_rows, predictions = [], [], []
    for panel in panels:
        panel_id = f"{panel['family']}__s{int(panel['seed'])}"
        features = gate_parent.feature_mapping(panel)
        predicted = gate_parent.predict_benefit(
            models[str(panel["dataset_group"])],
            np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64),
        )
        selected = "winner_preserved_borda_a050" if predicted > 0.0 else "broad_rff"
        groups = (
            winner_preserved_borda_groups(panel)
            if selected == "winner_preserved_borda_a050"
            else bbox_parent.ranking_groups(panel, "broad_rff")
        )
        gate = parent._custom_panel_metric(panel, "lodo_winner_preserved_gate", groups)
        gate["selected_method"] = selected
        gate["predicted_alternative_benefit"] = predicted
        gate_rows.append(gate)
        broad_regret = float(lookup[("broad_rff", panel_id)]["top_5_mean_utility_regret_r2"])
        alternative_regret = float(
            lookup[("winner_preserved_borda_a050", panel_id)]["top_5_mean_utility_regret_r2"]
        )
        oracle_selected = (
            "winner_preserved_borda_a050"
            if alternative_regret < broad_regret - TIE_ATOL
            else "broad_rff"
        )
        oracle_groups = (
            winner_preserved_borda_groups(panel)
            if oracle_selected == "winner_preserved_borda_a050"
            else bbox_parent.ranking_groups(panel, "broad_rff")
        )
        oracle_rows.append(
            parent._custom_panel_metric(panel, "oracle_winner_preserved_gate", oracle_groups)
        )
        predictions.append(
            {
                "panel_id": panel_id,
                "family": str(panel["family"]),
                "dataset_group": str(panel["dataset_group"]),
                "features": features,
                "predicted_alternative_benefit": predicted,
                "selected_method": selected,
                "oracle_selected_method": oracle_selected,
                "decision_correct": selected == oracle_selected,
            }
        )
    rows = [*fixed_rows, *gate_rows, *oracle_rows]
    rows.sort(key=lambda row: (row["family"], int(row["seed"]), METHODS.index(row["method"])))
    summaries = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 25:
            raise ValueError("winner-preserved gate method grid is incomplete")
        summaries.append(
            {
                "method": method,
                "panel_count": len(selected),
                "mean_spearman": float(np.mean([row["spearman"] for row in selected])),
                "mean_top_1_opportunity_regret_r2": float(np.mean([row["top_1_opportunity_regret_r2"] for row in selected])),
                "mean_top_1_exact_best_coverage": float(np.mean([row["top_1_exact_best_coverage"] for row in selected])),
                "mean_top_5_opportunity_regret_r2": float(np.mean([row["top_5_opportunity_regret_r2"] for row in selected])),
                "mean_top_5_mean_utility_regret_r2": float(np.mean([row["top_5_mean_utility_regret_r2"] for row in selected])),
                "mean_top_5_material_precision": float(np.mean([row["top_5_material_precision"] for row in selected])),
            }
        )
    comparisons = (
        ("broad_rff", "top_5_mean_utility_regret_r2"),
        ("bounded_2k", "top_5_mean_utility_regret_r2"),
        ("winner_preserved_borda_a050", "top_5_mean_utility_regret_r2"),
        ("pool_borda_a050", "top_5_mean_utility_regret_r2"),
        ("oracle_winner_preserved_gate", "top_5_mean_utility_regret_r2"),
        ("linear_probe", "top_5_mean_utility_regret_r2"),
        ("broad_rff", "top_5_opportunity_regret_r2"),
        ("broad_rff", "top_1_opportunity_regret_r2"),
    )
    contrasts = [
        parent._paired_interval(
            rows, "lodo_winner_preserved_gate", right, metric, index * 100
        )
        for index, (right, metric) in enumerate(comparisons)
    ]
    contrast_lookup = {(row["right"], row["metric"]): row for row in contrasts}
    primary = contrast_lookup[("broad_rff", "top_5_mean_utility_regret_r2")]
    top5 = contrast_lookup[("broad_rff", "top_5_opportunity_regret_r2")]
    top1 = contrast_lookup[("broad_rff", "top_1_opportunity_regret_r2")]
    winner_exact = all(
        bbox_parent.ranking_groups(panel, "broad_rff")[0]
        == winner_preserved_borda_groups(panel)[0]
        for panel in panels
    )
    checks = [
        {"criterion": "winner_preserved_gate_improves_broad_top5_mean_regret", "observed_upper_95": primary["upper_95"], "required": "<0", "pass": bool(primary["upper_95"] < 0.0)},
        {"criterion": "winner_group_exactly_matches_broad_in_every_panel", "pass": bool(winner_exact)},
        {"criterion": "gate_top1_metrics_exactly_match_broad", "observed_estimate": top1["estimate"], "maximum_absolute": TIE_ATOL, "pass": bool(abs(top1["estimate"]) <= TIE_ATOL and top1["lower_95"] == 0.0 and top1["upper_95"] == 0.0)},
        {"criterion": "gate_top5_opportunity_noninferior_to_broad", "observed_upper_95": top5["upper_95"], "maximum": 0.01, "pass": bool(top5["upper_95"] <= 0.01)},
        {"criterion": "complete_finite_four_group_grid", "observed_rows": len(rows), "observed_predictions": len(predictions), "pass": bool(len(rows) == 175 and len(predictions) == 25 and all(np.isfinite(row["predicted_alternative_benefit"]) for row in predictions))},
    ]
    return {
        "schema_version": 1,
        "status": "completed_winner_preserved_gate",
        "scope": "post_outcome_four_group_developmental_winner_preserved_gate",
        "gate_models": models,
        "gate_predictions": predictions,
        "panel_method_rows": rows,
        "method_summaries": summaries,
        "contrasts": contrasts,
        "decision": {"status": "pass" if all(row["pass"] for row in checks) else "fail", "checks": checks},
        "protocol": dict(protocol),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Winner-preserving continuous-OI reliability gate",
        "",
        f"Decision: **{summary['decision']['status'].upper()}**.",
        "",
        "The broad OI winner tie-group is immutable; the gate can alter only the remaining ranking.",
        "",
        "| Method | Spearman | Top-1 regret | Top-1 exact | Top-5 regret | Top-5 precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["method_summaries"]:
        lines.append(
            f"| {row['method']} | {row['mean_spearman']:.4f} | {row['mean_top_1_opportunity_regret_r2']:.5f} | "
            f"{row['mean_top_1_exact_best_coverage']:.3f} | {row['mean_top_5_mean_utility_regret_r2']:.5f} | "
            f"{row['mean_top_5_material_precision']:.3f} |"
        )
    lines.extend(["", "## Gate selections", ""])
    for group in parent.GROUPS:
        selected = [row for row in summary["gate_predictions"] if row["dataset_group"] == group]
        alternatives = sum(row["selected_method"] == "winner_preserved_borda_a050" for row in selected)
        correct = sum(row["decision_correct"] for row in selected)
        lines.append(f"- `{group}`: alternative `{alternatives}/{len(selected)}`; oracle agreement `{correct}/{len(selected)}`")
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- {'PASS' if row['pass'] else 'FAIL'}: `{row['criterion']}`" for row in summary["decision"]["checks"])
    return "\n".join(lines) + "\n"


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = _load_protocol()
    panels = parent.load_panels(parent._load_protocol())
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("winner-preserved gate output already exists")
    summary = summarize(panels, protocol)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "summary.json").write_text(prepare.canonical_json(summary) + "\n", encoding="utf-8")
    (output_root / "report.md").write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed_winner_preserved_gate",
        "implementation_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "protocol_sha256": prepare.sha256_path(PROTOCOL_PATH),
        "files": {
            name: {"sha256": prepare.sha256_path(output_root / name), "bytes": (output_root / name).stat().st_size}
            for name in ("summary.json", "report.md")
        },
    }
    (output_root / "manifest.json").write_text(prepare.canonical_json(manifest) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(prepare.canonical_json(run(args.output_root)["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

