"""Run the hash-bound real-panel OI rank-aggregation diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from experiments.continuous_real_pet_area_panel import prepare
from experiments.continuous_shortlist_depth_audit import run as audit_parent
from experiments.continuous_top5_bounded_confirmation import run as bounded_parent
from experiments.continuous_winner_fill_shortlist import run as winner_parent


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_rank_aggregation"
SOURCE_PATHS = {
    "utkface_age": (
        ROOT / "artifacts" / "continuous_real_age_panel" / "results" / "raw_results.json",
        ROOT / "artifacts" / "continuous_real_age_panel" / "results" / "manifest.json",
    ),
    "utkface_normalized_absolute_age_deviation": (
        ROOT / "artifacts" / "continuous_real_age_nonlinear_panel" / "raw_results.json",
        ROOT / "artifacts" / "continuous_real_age_nonlinear_panel" / "manifest.json",
    ),
    "oxford_pet_foreground_occupancy": (
        ROOT / "artifacts" / "continuous_real_pet_area_panel" / "results" / "raw_results.json",
        ROOT / "artifacts" / "continuous_real_pet_area_panel" / "results" / "manifest.json",
    ),
}
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
WEIGHT_PREFERENCE = (0.5, 0.75, 0.25, 1.0, 0.0)
WEIGHT_METHODS = {
    0.0: "pool_borda_a000",
    0.25: "pool_borda_a025",
    0.5: "pool_borda_a050",
    0.75: "pool_borda_a075",
    1.0: "pool_borda_a100",
}
BASE_METHODS = ("broad_rff", "save_rff_current", "bounded_2k")
METHODS = (*BASE_METHODS, *WEIGHT_METHODS.values(), "lodo_pool_borda", "linear_probe")
REPLICATE_SEEDS = (4103, 4201, 4303, 4409, 4507)
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 2026090401
TIE_ATOL = 1.0e-12


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or prepare.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("real rank-aggregation protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_post_outcome_real_rank_aggregation_before_derived_analysis":
        raise ValueError("real rank-aggregation protocol is not frozen")
    if tuple(float(value) for value in protocol["aggregation"]["weights"]) != WEIGHTS:
        raise ValueError("real rank-aggregation weights differ from code")
    if tuple(float(value) for value in protocol["leave_dataset_out_selection"]["tie_preference"]) != WEIGHT_PREFERENCE:
        raise ValueError("real rank-aggregation tie preference differs from code")
    if tuple(protocol["methods"]) != METHODS:
        raise ValueError("real rank-aggregation method set differs from code")
    return protocol


def load_panels(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    panels = []
    sources = protocol["ancestry"]["sources"]
    for family, (raw_path, manifest_path) in SOURCE_PATHS.items():
        descriptor = sources[family]
        if prepare.sha256_path(raw_path) != descriptor["raw_results_sha256"]:
            raise ValueError(f"real rank-aggregation raw source changed: {family}")
        if prepare.sha256_path(manifest_path) != descriptor["manifest_sha256"]:
            raise ValueError(f"real rank-aggregation manifest source changed: {family}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        selected = raw.get("panels", ())
        if len(selected) != 5 or tuple(int(panel["seed"]) for panel in selected) != REPLICATE_SEEDS:
            raise ValueError(f"real rank-aggregation panel grid changed: {family}")
        for panel in selected:
            if panel.get("family") != family or len(panel.get("candidates", ())) != 16:
                raise ValueError(f"real rank-aggregation panel identity changed: {family}")
            panels.append({**panel, "dataset_group": descriptor["dataset_group"]})
    if len(panels) != 15:
        raise ValueError("real rank-aggregation requires exactly 15 panels")
    return panels


def pool_borda_groups(panel: Mapping[str, Any], alpha: float) -> list[tuple[int, ...]]:
    if float(alpha) not in WEIGHTS:
        raise ValueError("pool-Borda alpha is not frozen")
    candidates = panel["candidates"]
    if len(candidates) != 16:
        raise ValueError("pool-Borda requires 16 candidates")
    broad = winner_parent._component_values(panel, "broad_rff")
    save = winner_parent._component_values(panel, "save_rff_current")
    broad_groups = winner_parent._score_groups(broad)
    save_groups = winner_parent._score_groups(save)
    pool = audit_parent._broad_pool(broad_groups, 5, len(candidates))
    outside = set(range(len(candidates))).difference(pool)
    broad_ordinal = winner_parent._ordinal_scores(broad_groups, len(candidates))
    save_ordinal = winner_parent._ordinal_scores(save_groups, len(candidates))
    combined = float(alpha) * broad_ordinal + (1.0 - float(alpha)) * save_ordinal
    groups = winner_parent._score_groups(combined, pool)
    if outside:
        groups.extend(winner_parent._score_groups(broad, outside))
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(len(candidates))):
        raise AssertionError("pool-Borda ranking is not a complete partition")
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
        "rank_groups": [
            [candidates[index]["candidate_id"] for index in group] for group in groups
        ],
        "spearman": winner_parent._spearman(reference, ordinal),
        "material_pair_concordance": winner_parent._material_concordance(reference, ordinal),
        "descriptive_panel_runtime_seconds": bounded_parent._runtime_seconds(panel, "bounded_2k"),
    }
    for k in (1, 3, 5):
        for key, value in winner_parent._top_k_metrics(reference, groups, k).items():
            result[f"top_{k}_{key}"] = value
    return result


def panel_metric(panel: Mapping[str, Any], method: str) -> dict[str, Any]:
    if method in (*BASE_METHODS, "linear_probe"):
        result = bounded_parent.panel_metric(panel, method)
        result["dataset_group"] = str(panel["dataset_group"])
        return result
    alpha = next((weight for weight, name in WEIGHT_METHODS.items() if name == method), None)
    if alpha is None:
        raise ValueError(f"unknown fixed aggregation method {method!r}")
    return _custom_panel_metric(panel, method, pool_borda_groups(panel, alpha))


def select_lodo_weights(
    panels: Sequence[Mapping[str, Any]], fixed_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    groups = {str(panel["dataset_group"]) for panel in panels}
    if groups != {"utkface", "oxford_pet"}:
        raise ValueError("leave-dataset-out groups changed")
    output = {}
    for held_out in sorted(groups):
        training_ids = {
            f"{panel['family']}__s{int(panel['seed'])}"
            for panel in panels
            if panel["dataset_group"] != held_out
        }
        means = {}
        for weight, method in WEIGHT_METHODS.items():
            selected = [
                float(row["top_5_mean_utility_regret_r2"])
                for row in fixed_rows
                if row["method"] == method and row["panel_id"] in training_ids
            ]
            if len(selected) != len(training_ids):
                raise ValueError("leave-dataset-out training rows are incomplete")
            means[weight] = float(np.mean(selected))
        minimum = min(means.values())
        chosen = next(
            weight for weight in WEIGHT_PREFERENCE if means[weight] <= minimum + TIE_ATOL
        )
        output[held_out] = {
            "held_out_dataset_group": held_out,
            "selected_alpha": chosen,
            "training_panel_count": len(training_ids),
            "training_mean_top5_regret_by_alpha": {
                str(weight): means[weight] for weight in WEIGHTS
            },
        }
    return output


def _paired_interval(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, metric: str, offset: int
) -> dict[str, Any]:
    lookup = {(str(row["method"]), str(row["panel_id"])): row for row in rows}
    seed_values = []
    for seed in REPLICATE_SEEDS:
        differences = []
        for family in SOURCE_PATHS:
            panel_id = f"{family}__s{seed}"
            differences.append(
                float(lookup[(left, panel_id)][metric])
                - float(lookup[(right, panel_id)][metric])
            )
        seed_values.append(float(np.mean(differences)))
    values = np.asarray(seed_values, dtype=np.float64)
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
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED + offset,
        "unit": "joint_replicate_seed_across_three_target_families",
    }


def summarize(panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    fixed_rows = [
        panel_metric(panel, method)
        for panel in panels
        for method in (*BASE_METHODS, *WEIGHT_METHODS.values(), "linear_probe")
    ]
    selections = select_lodo_weights(panels, fixed_rows)
    lodo_rows = []
    for panel in panels:
        selection = selections[str(panel["dataset_group"])]
        alpha = float(selection["selected_alpha"])
        row = _custom_panel_metric(
            panel, "lodo_pool_borda", pool_borda_groups(panel, alpha)
        )
        row["selected_alpha"] = alpha
        lodo_rows.append(row)
    rows = [*fixed_rows, *lodo_rows]
    rows.sort(key=lambda row: (row["family"], int(row["seed"]), METHODS.index(row["method"])))
    method_summaries = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 15:
            raise ValueError(f"aggregation method {method!r} is incomplete")
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
        _paired_interval(rows, "lodo_pool_borda", right, metric, position * 100)
        for position, (right, metric) in enumerate(
            (
                ("broad_rff", "top_5_mean_utility_regret_r2"),
                ("bounded_2k", "top_5_mean_utility_regret_r2"),
                ("linear_probe", "top_5_mean_utility_regret_r2"),
                ("broad_rff", "top_5_opportunity_regret_r2"),
                ("broad_rff", "top_1_opportunity_regret_r2"),
            )
        )
    ]
    lookup = {(row["right"], row["metric"]): row for row in contrasts}
    primary = lookup[("broad_rff", "top_5_mean_utility_regret_r2")]
    safety = lookup[("broad_rff", "top_5_opportunity_regret_r2")]
    checks = [
        {
            "criterion": "lodo_aggregation_improves_broad_top5_mean_regret",
            "observed_upper_95": primary["upper_95"],
            "required": "<0",
            "pass": bool(primary["upper_95"] < 0.0),
        },
        {
            "criterion": "lodo_top5_opportunity_noninferior_to_broad",
            "observed_upper_95": safety["upper_95"],
            "maximum": 0.01,
            "pass": bool(safety["upper_95"] <= 0.01),
        },
        {
            "criterion": "complete_finite_aggregation_grid",
            "observed_rows": len(rows),
            "required_rows": 150,
            "pass": bool(
                len(rows) == 150
                and all(
                    np.isfinite(
                        [
                            row["spearman"],
                            row["top_5_mean_utility_regret_r2"],
                            row["top_5_opportunity_regret_r2"],
                        ]
                    ).all()
                    for row in rows
                )
            ),
        },
    ]
    return {
        "schema_version": 1,
        "status": "completed_post_outcome_real_rank_aggregation",
        "scope": "developmental_two_dataset_group_diagnostic_not_confirmation",
        "panel_count": len(panels),
        "panel_method_rows": rows,
        "method_summaries": method_summaries,
        "leave_dataset_out_selections": selections,
        "contrasts": contrasts,
        "decision": {
            "status": "pass" if all(row["pass"] for row in checks) else "fail",
            "checks": checks,
        },
        "protocol": dict(protocol),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Real-panel OI rank aggregation",
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
    lines.extend(["", "## Leave-dataset-out selections", ""])
    for group, selection in summary["leave_dataset_out_selections"].items():
        lines.append(
            f"- `{group}` held out: alpha `{selection['selected_alpha']}` from "
            f"{selection['training_panel_count']} other-group panels"
        )
    lines.extend(["", "## Checks", ""])
    lines.extend(
        f"- {'PASS' if row['pass'] else 'FAIL'}: `{row['criterion']}`"
        for row in summary["decision"]["checks"]
    )
    return "\n".join(lines) + "\n"


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = _load_protocol()
    panels = load_panels(protocol)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("real rank-aggregation output already exists")
    summary = summarize(panels, protocol)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "summary.json").write_text(
        prepare.canonical_json(summary) + "\n", encoding="utf-8"
    )
    (output_root / "report.md").write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed_post_outcome_real_rank_aggregation",
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
