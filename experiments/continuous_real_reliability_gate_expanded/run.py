"""Evaluate the unchanged OI-only reliability gate on four dataset groups."""

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
from experiments.continuous_real_rank_aggregation import run as rank_parent
from experiments.continuous_real_reliability_gate import run as gate_parent
from experiments.continuous_shortlist_depth_audit import run as audit_parent
from experiments.continuous_winner_fill_shortlist import run as winner_parent


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_reliability_gate_expanded"
FEATURE_NAMES = gate_parent.FEATURE_NAMES
METHODS = gate_parent.METHODS
BACKBONES = bbox_parent.BACKBONES
GROUPS = ("utkface", "oxford_pet", "torchvision_gtsrb", "torchvision_fgvc_aircraft")
REPLICATE_SEEDS = rank_parent.REPLICATE_SEEDS
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026091201
TIE_ATOL = gate_parent.TIE_ATOL


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or prepare.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("expanded reliability-gate protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_expanded_reliability_gate_before_derived_analysis":
        raise ValueError("expanded reliability-gate protocol is not frozen")
    if tuple(protocol["feature_schema"]) != FEATURE_NAMES:
        raise ValueError("expanded reliability-gate features changed")
    if tuple(protocol["methods"]) != METHODS:
        raise ValueError("expanded reliability-gate methods changed")
    if tuple(protocol["dataset_groups"]) != GROUPS:
        raise ValueError("expanded reliability-gate groups changed")
    if tuple(protocol["backbones"]) != BACKBONES:
        raise ValueError("expanded reliability-gate backbone roster changed")
    ancestry = protocol["ancestry"]
    paths = {
        "prior_gate_protocol_sha256": gate_parent.PROTOCOL_PATH,
        "prior_gate_summary_sha256": gate_parent.DEFAULT_OUTPUT_ROOT / "summary.json",
        "prior_gate_manifest_sha256": gate_parent.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "bbox_protocol_sha256": bbox_parent.PROTOCOL_PATH,
        "bbox_raw_results_sha256": bbox_parent.DEFAULT_OUTPUT_ROOT / "raw_results.json",
        "bbox_manifest_sha256": bbox_parent.DEFAULT_OUTPUT_ROOT / "manifest.json",
    }
    for field, path in paths.items():
        if prepare.sha256_path(path) != ancestry[field]:
            raise ValueError(f"expanded reliability-gate ancestry changed: {field}")
    return protocol


def _common_roster_panel(panel: Mapping[str, Any]) -> dict[str, Any]:
    candidates = {row["candidate_id"]: row for row in panel["candidates"]}
    if not set(BACKBONES).issubset(candidates):
        raise ValueError("expanded reliability-gate common roster is incomplete")
    return {
        **panel,
        "candidate_count": len(BACKBONES),
        "candidates": [candidates[backbone] for backbone in BACKBONES],
    }


def load_panels(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    old = [
        _common_roster_panel(panel)
        for panel in rank_parent.load_panels(rank_parent._load_protocol())
    ]
    raw = json.loads((bbox_parent.DEFAULT_OUTPUT_ROOT / "raw_results.json").read_text(encoding="utf-8"))
    new = raw.get("panels", ())
    if len(new) != 10:
        raise ValueError("expanded reliability-gate bbox panel grid is incomplete")
    panels = [*old, *new]
    observed_groups = {str(panel["dataset_group"]) for panel in panels}
    if observed_groups != set(GROUPS):
        raise ValueError("expanded reliability-gate dataset groups are incomplete")
    family_counts: dict[str, int] = {}
    for panel in panels:
        if len(panel.get("candidates", ())) != len(BACKBONES):
            raise ValueError("expanded reliability-gate candidate grid changed")
        if tuple(row["candidate_id"] for row in panel["candidates"]) != BACKBONES:
            raise ValueError("expanded reliability-gate candidate order changed")
        if int(panel["seed"]) not in REPLICATE_SEEDS:
            raise ValueError("expanded reliability-gate replicate seed changed")
        family = str(panel["family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    if len(family_counts) != 5 or set(family_counts.values()) != {5}:
        raise ValueError("expanded reliability-gate target-family grid changed")
    return panels


def pool_borda_groups(panel: Mapping[str, Any], alpha: float = 0.5) -> list[tuple[int, ...]]:
    if float(alpha) != 0.5:
        raise ValueError("expanded reliability gate freezes alpha at 0.5")
    count = len(panel["candidates"])
    broad = winner_parent._component_values(panel, "broad_rff")
    save = winner_parent._component_values(panel, "save_rff_current")
    broad_groups = winner_parent._score_groups(broad)
    save_groups = winner_parent._score_groups(save)
    pool = audit_parent._broad_pool(broad_groups, 5, count)
    outside = set(range(count)).difference(pool)
    combined = 0.5 * winner_parent._ordinal_scores(broad_groups, count)
    combined += 0.5 * winner_parent._ordinal_scores(save_groups, count)
    groups = winner_parent._score_groups(combined, pool)
    if outside:
        groups.extend(winner_parent._score_groups(broad, outside))
    if sorted(index for group in groups for index in group) != list(range(count)):
        raise AssertionError("expanded pool-Borda groups are not a complete partition")
    return groups


def _custom_panel_metric(
    panel: Mapping[str, Any], method: str, groups: Sequence[Sequence[int]]
) -> dict[str, Any]:
    candidates = panel["candidates"]
    reference = np.asarray(
        [candidate["best_head_envelope_r2"] for candidate in candidates], dtype=np.float64
    )
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
        ),
    }
    for k in (1, 3, 5):
        for key, value in winner_parent._top_k_metrics(reference, groups, k).items():
            result[f"top_{k}_{key}"] = value
    return result


def panel_metric(panel: Mapping[str, Any], method: str) -> dict[str, Any]:
    if method in {"broad_rff", "bounded_2k", "linear_probe", "save_rff_current"}:
        return bbox_parent.panel_metric(panel, method)
    if method == "pool_borda_a050":
        return _custom_panel_metric(panel, method, pool_borda_groups(panel))
    raise ValueError(f"unknown expanded reliability fixed method {method!r}")


def _lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    output = {(str(row["method"]), str(row["panel_id"])): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("expanded reliability metric rows are duplicated")
    return output


def fit_lodo_models(
    panels: Sequence[Mapping[str, Any]], fixed_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    lookup = _lookup(fixed_rows)
    models = {}
    for held_out in GROUPS:
        training = [panel for panel in panels if panel["dataset_group"] != held_out]
        X, y = [], []
        for panel in training:
            panel_id = f"{panel['family']}__s{int(panel['seed'])}"
            broad = lookup[("broad_rff", panel_id)]
            blend = lookup[("pool_borda_a050", panel_id)]
            X.append(gate_parent.feature_vector(panel))
            y.append(
                float(broad["top_5_mean_utility_regret_r2"])
                - float(blend["top_5_mean_utility_regret_r2"])
            )
        model = gate_parent.fit_ridge_gate(np.asarray(X), np.asarray(y))
        model["held_out_dataset_group"] = held_out
        model["feature_names"] = list(FEATURE_NAMES)
        models[held_out] = model
    return models


def _paired_interval(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, metric: str, offset: int
) -> dict[str, Any]:
    lookup = _lookup(rows)
    families = sorted({str(row["family"]) for row in rows})
    block_values = []
    for seed in REPLICATE_SEEDS:
        differences = []
        for family in families:
            panel_id = f"{family}__s{seed}"
            differences.append(
                float(lookup[(left, panel_id)][metric])
                - float(lookup[(right, panel_id)][metric])
            )
        block_values.append(float(np.mean(differences)))
    values = np.asarray(block_values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    draws = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "estimate": float(np.mean(values)),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "by_dataset_group": {
            group: float(
                np.mean(
                    [
                        float(row[metric])
                        - float(lookup[(right, str(row["panel_id"]))][metric])
                        for row in rows
                        if row["method"] == left and row["dataset_group"] == group
                    ]
                )
            )
            for group in GROUPS
        },
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED + offset,
        "unit": "joint_replicate_position_across_five_target_families",
    }


def summarize(panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    fixed_methods = ("broad_rff", "bounded_2k", "pool_borda_a050", "linear_probe")
    fixed_rows = [panel_metric(panel, method) for panel in panels for method in fixed_methods]
    lookup = _lookup(fixed_rows)
    models = fit_lodo_models(panels, fixed_rows)
    gate_rows, oracle_rows, predictions = [], [], []
    for panel in panels:
        panel_id = f"{panel['family']}__s{int(panel['seed'])}"
        features = gate_parent.feature_mapping(panel)
        predicted = gate_parent.predict_benefit(
            models[str(panel["dataset_group"])],
            np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64),
        )
        selected = "pool_borda_a050" if predicted > 0.0 else "broad_rff"
        groups = pool_borda_groups(panel) if selected == "pool_borda_a050" else bbox_parent.ranking_groups(panel, "broad_rff")
        gate = _custom_panel_metric(panel, "lodo_reliability_gate", groups)
        gate["selected_method"] = selected
        gate["predicted_blend_benefit"] = predicted
        gate_rows.append(gate)
        broad_regret = float(lookup[("broad_rff", panel_id)]["top_5_mean_utility_regret_r2"])
        blend_regret = float(lookup[("pool_borda_a050", panel_id)]["top_5_mean_utility_regret_r2"])
        oracle_selected = "pool_borda_a050" if blend_regret < broad_regret - TIE_ATOL else "broad_rff"
        oracle_groups = pool_borda_groups(panel) if oracle_selected == "pool_borda_a050" else bbox_parent.ranking_groups(panel, "broad_rff")
        oracle_rows.append(_custom_panel_metric(panel, "oracle_panel_gate", oracle_groups))
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
    summaries = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 25:
            raise ValueError("expanded reliability-gate method grid is incomplete")
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
        ("pool_borda_a050", "top_5_mean_utility_regret_r2"),
        ("oracle_panel_gate", "top_5_mean_utility_regret_r2"),
        ("linear_probe", "top_5_mean_utility_regret_r2"),
        ("broad_rff", "top_5_opportunity_regret_r2"),
        ("broad_rff", "top_1_opportunity_regret_r2"),
    )
    contrasts = [
        _paired_interval(rows, "lodo_reliability_gate", right, metric, index * 100)
        for index, (right, metric) in enumerate(comparisons)
    ]
    contrast_lookup = {(row["right"], row["metric"]): row for row in contrasts}
    primary = contrast_lookup[("broad_rff", "top_5_mean_utility_regret_r2")]
    top5 = contrast_lookup[("broad_rff", "top_5_opportunity_regret_r2")]
    top1 = contrast_lookup[("broad_rff", "top_1_opportunity_regret_r2")]
    checks = [
        {"criterion": "reliability_gate_improves_broad_top5_mean_regret", "observed_upper_95": primary["upper_95"], "required": "<0", "pass": bool(primary["upper_95"] < 0.0)},
        {"criterion": "gate_top5_opportunity_noninferior_to_broad", "observed_upper_95": top5["upper_95"], "maximum": 0.01, "pass": bool(top5["upper_95"] <= 0.01)},
        {"criterion": "gate_top1_opportunity_noninferior_to_broad", "observed_upper_95": top1["upper_95"], "maximum": 0.01, "pass": bool(top1["upper_95"] <= 0.01)},
        {"criterion": "complete_finite_four_group_grid", "observed_rows": len(rows), "observed_predictions": len(predictions), "pass": bool(len(rows) == 150 and len(predictions) == 25 and all(np.isfinite(row["predicted_blend_benefit"]) for row in predictions))},
    ]
    return {
        "schema_version": 1,
        "status": "completed_expanded_reliability_gate",
        "scope": "post_outcome_four_dataset_group_developmental_gate_not_confirmation",
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
        "# Expanded continuous-OI reliability gate",
        "",
        f"Decision: **{summary['decision']['status'].upper()}**.",
        "",
        "Negative regret contrasts favor the method on the left. All groups use the same ten-backbone roster.",
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
    for group in GROUPS:
        selected = [row for row in summary["gate_predictions"] if row["dataset_group"] == group]
        blends = sum(row["selected_method"] == "pool_borda_a050" for row in selected)
        correct = sum(row["decision_correct"] for row in selected)
        lines.append(f"- `{group}`: blend `{blends}/{len(selected)}`; oracle agreement `{correct}/{len(selected)}`")
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- {'PASS' if row['pass'] else 'FAIL'}: `{row['criterion']}`" for row in summary["decision"]["checks"])
    return "\n".join(lines) + "\n"


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = _load_protocol()
    panels = load_panels(protocol)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("expanded reliability-gate output already exists")
    summary = summarize(panels, protocol)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "summary.json").write_text(prepare.canonical_json(summary) + "\n", encoding="utf-8")
    (output_root / "report.md").write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed_expanded_reliability_gate",
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

