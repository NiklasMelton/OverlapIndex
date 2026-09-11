"""Run the frozen large-panel cheap head-subset generalization test."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
import platform
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import scipy
import sklearn

from experiments.continuous_coi_head_subset_cost_diagnostic import run as prior
from experiments.continuous_large_recoverable_ranker_comparison import run as ranker
from experiments.continuous_large_structured_opportunity_benchmark import extract
from experiments.continuous_large_structured_opportunity_benchmark import prepare
from experiments.continuous_large_structured_opportunity_benchmark import run as source_run


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_large_head_subset_generalization"

HEADS = ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
SUBSETS = tuple(
    subset
    for width in range(1, len(HEADS) + 1)
    for subset in combinations(HEADS, width)
)
GROUPS = (
    "all",
    "rank_changing_nonlinear_opportunity",
    "recoverable_lp_dominated_control",
)
BASELINE = ("linear_ridge",)
PRIMARY = ("quadratic_ridge", "knn")
ACCURACY_CONTROL = ("quadratic_ridge", "rbf_svr")
FULL = HEADS
TIE_ATOL = 1.0e-12


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: Path) -> str:
    return prepare.sha256_path(path)


def subset_id(subset: Sequence[str]) -> str:
    value = tuple(subset)
    if not value or tuple(head for head in HEADS if head in value) != value:
        raise ValueError("large head subset is not in canonical order")
    return "+".join(value)


SUBSET_IDS = tuple(subset_id(subset) for subset in SUBSETS)
BASELINE_ID = subset_id(BASELINE)
PRIMARY_ID = subset_id(PRIMARY)
ACCURACY_CONTROL_ID = subset_id(ACCURACY_CONTROL)
FULL_ID = subset_id(FULL)


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("large head-subset protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    grid = protocol["expected_grid"]
    success = protocol["success"]
    if (
        protocol.get("status")
        != "frozen_large_head_subset_generalization_before_subset_outcomes"
        or tuple(protocol["ablation"]["heads"]) != HEADS
        or tuple(protocol["ablation"]["baseline_subset"]) != BASELINE
        or tuple(protocol["ablation"]["primary_subset"]) != PRIMARY
        or tuple(protocol["ablation"]["accuracy_control_subset"])
        != ACCURACY_CONTROL
        or tuple(protocol["ablation"]["full_reference_subset"]) != FULL
        or tuple(protocol["evaluation"]["groups"]) != GROUPS
        or int(protocol["evaluation"]["topk"]) != 3
        or float(protocol["evaluation"]["material_pair_difference_r2"]) != 0.01
        or float(protocol["ablation"]["tie_tolerance"]) != TIE_ATOL
        or len(SUBSETS) != 15
        or len(set(SUBSET_IDS)) != 15
        or [
            int(grid[key])
            for key in (
                "target_count",
                "panel_count",
                "candidate_count",
                "subset_count",
                "ranking_row_count",
                "ranking_summary_count",
                "cost_row_count",
                "new_model_fit_count",
                "new_overlapindex_fit_count",
                "reserved_backbone_count",
            )
        ]
        != [8, 24, 120, 15, 360, 45, 15, 0, 0, 0]
        or {
            key: success[key]
            for key in (
                "minimum_overall_spearman_gain_over_linear_cv",
                "minimum_opportunity_spearman_gain_over_linear_cv",
                "maximum_control_spearman_drop_from_linear_cv",
                "maximum_overall_top1_regret_increase_r2",
                "minimum_opportunity_top1_regret_reduction_r2",
                "maximum_control_top1_regret_increase_r2",
                "maximum_overall_top3_opportunity_regret_increase_r2",
                "maximum_aggregate_runtime_ratio_to_linear_cv",
            )
        }
        != {
            "minimum_overall_spearman_gain_over_linear_cv": 0.05,
            "minimum_opportunity_spearman_gain_over_linear_cv": 0.05,
            "maximum_control_spearman_drop_from_linear_cv": 0.05,
            "maximum_overall_top1_regret_increase_r2": 0.005,
            "minimum_opportunity_top1_regret_reduction_r2": 0.005,
            "maximum_control_top1_regret_increase_r2": 0.005,
            "maximum_overall_top3_opportunity_regret_increase_r2": 0.002,
            "maximum_aggregate_runtime_ratio_to_linear_cv": 3.0,
        }
    ):
        raise ValueError("large head-subset protocol differs from code")
    return protocol


def _validate_files(descriptor: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    for key, path in paths.items():
        if not path.is_file() or sha256_path(path) != descriptor[key]:
            raise ValueError(f"large head-subset source changed: {path}")


def validate_sources(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = protocol["source"]
    source_root = ROOT / str(source["directory"])
    _validate_files(
        source,
        {
            "protocol_sha256": prepare.PROTOCOL_PATH,
            "runner_sha256": Path(source_run.__file__).resolve(),
            "raw_results_sha256": source_root / "raw_results.json",
            "summary_sha256": source_root / "summary.json",
            "manifest_sha256": source_root / "manifest.json",
        },
    )
    source_raw = json.loads((source_root / "raw_results.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))

    selection = protocol["selection_source"]
    selection_root = ROOT / str(selection["directory"])
    _validate_files(
        selection,
        {
            "protocol_sha256": ranker.PROTOCOL_PATH,
            "runner_sha256": Path(ranker.__file__).resolve(),
            "summary_sha256": selection_root / "summary.json",
            "manifest_sha256": selection_root / "manifest.json",
        },
    )
    selection_summary = json.loads((selection_root / "summary.json").read_text(encoding="utf-8"))
    selection_manifest = json.loads((selection_root / "manifest.json").read_text(encoding="utf-8"))

    basis = protocol["prior_selection_basis"]
    basis_root = ROOT / str(basis["directory"])
    _validate_files(
        basis,
        {
            "protocol_sha256": prior.PROTOCOL_PATH,
            "runner_sha256": Path(prior.__file__).resolve(),
            "summary_sha256": basis_root / "summary.json",
            "manifest_sha256": basis_root / "manifest.json",
        },
    )
    basis_summary = json.loads((basis_root / "summary.json").read_text(encoding="utf-8"))

    ranker_protocol = ranker.load_protocol()
    if (
        source_raw.get("status") != source["expected_status"]
        or source_manifest.get("status") != source["expected_status"]
        or int(source_manifest.get("counts", {}).get("overlapindex_fits", -1)) != 0
        or int(source_manifest.get("counts", {}).get("reserved_backbones", -1)) != 0
        or selection_summary.get("status") != selection["expected_status"]
        or selection_manifest.get("status") != selection["expected_status"]
        or int(selection_summary.get("panel_count", -1)) != 24
        or int(selection_summary.get("candidate_count", -1)) != 120
        or selection_summary.get("reserved_backbones_evaluated") != []
        or basis_summary.get("status") != "completed_head_subset_cost_diagnostic"
        or basis_summary.get("diagnosis", {}).get("status")
        != "rbf_required_but_head_set_reducible"
        or protocol["strata"] != ranker_protocol["strata"]
    ):
        raise ValueError("large head-subset source artifact is malformed")
    return source_raw, selection_summary, basis_summary


def _target_key(panel: Mapping[str, Any]) -> str:
    return f"{panel['dataset_id']}::{panel['target_id']}"


def selected_panels(
    source_raw: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    strata = {
        target: stratum
        for stratum, targets in protocol["strata"].items()
        for target in targets
    }
    panels = [
        {**dict(panel), "stratum": strata[_target_key(panel)]}
        for panel in source_raw["panels"]
        if _target_key(panel) in strata
    ]
    expected = {
        (target, seed)
        for target in strata
        for seed in prepare.REPLICATE_SEEDS
    }
    observed = {(_target_key(panel), int(panel["seed"])) for panel in panels}
    if (
        observed != expected
        or len(panels) != 24
        or any(
            tuple(row["candidate_id"] for row in panel["candidates"])
            != extract.BACKBONES
            for panel in panels
        )
    ):
        raise ValueError("large head-subset selected panel grid is incomplete")
    return sorted(
        panels,
        key=lambda row: (
            str(row["stratum"]),
            str(row["dataset_id"]),
            str(row["target_id"]),
            int(row["seed"]),
        ),
    )


def normalized_panels(panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for panel in panels:
        candidates = []
        for candidate in panel["candidates"]:
            heads = candidate["heads"]
            inner = {head: float(heads[head]["inner_best_r2"]) for head in HEADS}
            heldout = {head: float(heads[head]["test_r2"]) for head in HEADS}
            wall = {head: float(heads[head]["wall_seconds"]) for head in HEADS}
            if (
                not np.isfinite(list(inner.values())).all()
                or not np.isfinite(list(heldout.values())).all()
                or not np.isfinite(list(wall.values())).all()
                or min(wall.values()) < 0.0
                or abs(max(heldout.values()) - float(candidate["best_head_envelope_r2"]))
                > TIE_ATOL
            ):
                raise ValueError("large head-subset candidate record is invalid")
            candidates.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "inner_cv_scores": inner,
                    "head_wall_seconds": wall,
                    "heldout_best_head_envelope_r2": float(
                        candidate["best_head_envelope_r2"]
                    ),
                }
            )
        output.append(
            {
                "dataset_id": str(panel["dataset_id"]),
                "target_id": str(panel["target_id"]),
                "stratum": str(panel["stratum"]),
                "seed": int(panel["seed"]),
                "candidates": candidates,
            }
        )
    if len(output) != 24 or sum(len(row["candidates"]) for row in output) != 120:
        raise ValueError("large head-subset normalized grid is incomplete")
    return output


def subset_scores(panel: Mapping[str, Any], subset: Sequence[str]) -> list[float]:
    value = tuple(subset)
    subset_id(value)
    return [
        max(float(row["inner_cv_scores"][head]) for head in value)
        for row in panel["candidates"]
    ]


def ranking_rows(panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for panel in panels:
        truth = [
            float(row["heldout_best_head_envelope_r2"])
            for row in panel["candidates"]
        ]
        for subset in SUBSETS:
            rows.append(
                {
                    "dataset_id": str(panel["dataset_id"]),
                    "target_id": str(panel["target_id"]),
                    "stratum": str(panel["stratum"]),
                    "seed": int(panel["seed"]),
                    "subset_id": subset_id(subset),
                    "heads": list(subset),
                    **ranker.ranking_metrics(truth, subset_scores(panel, subset)),
                }
            )
    if len(rows) != 360:
        raise RuntimeError("large head-subset ranking row count changed")
    return rows


def _group_rows(
    rows: Sequence[Mapping[str, Any]], group: str, identifier: str
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["subset_id"] == identifier
        and (group == "all" or row["stratum"] == group)
    ]


def ranking_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group in GROUPS:
        expected = 24 if group == "all" else 12
        for subset in SUBSETS:
            identifier = subset_id(subset)
            selected = _group_rows(rows, group, identifier)
            if len(selected) != expected:
                raise ValueError("large head-subset ranking summary grid is incomplete")
            spearman = [
                float(row["spearman"])
                for row in selected
                if row["spearman"] is not None
            ]
            pairs = [
                float(row["material_pair_accuracy"])
                for row in selected
                if row["material_pair_accuracy"] is not None
            ]
            output.append(
                {
                    "group": group,
                    "subset_id": identifier,
                    "heads": list(subset),
                    "panel_count": len(selected),
                    "defined_spearman_count": len(spearman),
                    "mean_spearman": float(np.mean(spearman)) if spearman else None,
                    "mean_top1_regret_r2": float(
                        np.mean([row["top1_regret_r2"] for row in selected])
                    ),
                    "exact_best_rate": float(
                        np.mean([row["exact_best"] for row in selected])
                    ),
                    "mean_top3_opportunity_regret_r2": float(
                        np.mean(
                            [row["top3_opportunity_regret_r2"] for row in selected]
                        )
                    ),
                    "mean_top3_utility_regret_r2": float(
                        np.mean(
                            [row["top3_mean_utility_regret_r2"] for row in selected]
                        )
                    ),
                    "mean_top3_set_recall": float(
                        np.mean([row["top3_set_recall"] for row in selected])
                    ),
                    "mean_material_pair_accuracy": (
                        float(np.mean(pairs)) if pairs else None
                    ),
                }
            )
    if len(output) != 45:
        raise RuntimeError("large head-subset ranking summary count changed")
    return output


def cost_rows(panels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    linear_total = sum(
        float(row["head_wall_seconds"][BASELINE[0]])
        for panel in panels
        for row in panel["candidates"]
    )
    output = []
    for subset in SUBSETS:
        candidate_costs = [
            sum(float(row["head_wall_seconds"][head]) for head in subset)
            for panel in panels
            for row in panel["candidates"]
        ]
        output.append(
            {
                "subset_id": subset_id(subset),
                "heads": list(subset),
                "candidate_count": len(candidate_costs),
                "aggregate_archived_seconds": float(np.sum(candidate_costs)),
                "median_candidate_seconds": float(np.median(candidate_costs)),
                "relative_to_linear_cv": float(np.sum(candidate_costs) / linear_total),
            }
        )
    if len(output) != 15:
        raise RuntimeError("large head-subset cost row count changed")
    return output


def _lookup(
    summaries: Sequence[Mapping[str, Any]], group: str, identifier: str
) -> Mapping[str, Any]:
    selected = [
        row
        for row in summaries
        if row["group"] == group and row["subset_id"] == identifier
    ]
    if len(selected) != 1:
        raise ValueError("large head-subset summary row is not unique")
    return selected[0]


def decision(
    summaries: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rule = protocol["success"]
    baseline = {group: _lookup(summaries, group, BASELINE_ID) for group in GROUPS}
    primary = {group: _lookup(summaries, group, PRIMARY_ID) for group in GROUPS}
    cost = {row["subset_id"]: row for row in costs}[PRIMARY_ID]
    contrasts = {
        "overall_spearman_gain": float(primary["all"]["mean_spearman"])
        - float(baseline["all"]["mean_spearman"]),
        "opportunity_spearman_gain": float(primary[GROUPS[1]]["mean_spearman"])
        - float(baseline[GROUPS[1]]["mean_spearman"]),
        "control_spearman_drop": float(baseline[GROUPS[2]]["mean_spearman"])
        - float(primary[GROUPS[2]]["mean_spearman"]),
        "overall_top1_regret_increase_r2": float(
            primary["all"]["mean_top1_regret_r2"]
        )
        - float(baseline["all"]["mean_top1_regret_r2"]),
        "opportunity_top1_regret_reduction_r2": float(
            baseline[GROUPS[1]]["mean_top1_regret_r2"]
        )
        - float(primary[GROUPS[1]]["mean_top1_regret_r2"]),
        "control_top1_regret_increase_r2": float(
            primary[GROUPS[2]]["mean_top1_regret_r2"]
        )
        - float(baseline[GROUPS[2]]["mean_top1_regret_r2"]),
        "overall_top3_opportunity_regret_increase_r2": float(
            primary["all"]["mean_top3_opportunity_regret_r2"]
        )
        - float(baseline["all"]["mean_top3_opportunity_regret_r2"]),
        "runtime_ratio_to_linear_cv": float(cost["relative_to_linear_cv"]),
    }
    checks = {
        "overall_spearman_gain": contrasts["overall_spearman_gain"]
        >= float(rule["minimum_overall_spearman_gain_over_linear_cv"]) - TIE_ATOL,
        "opportunity_spearman_gain": contrasts["opportunity_spearman_gain"]
        >= float(rule["minimum_opportunity_spearman_gain_over_linear_cv"])
        - TIE_ATOL,
        "control_spearman_preserved": contrasts["control_spearman_drop"]
        <= float(rule["maximum_control_spearman_drop_from_linear_cv"]) + TIE_ATOL,
        "overall_top1_regret_preserved": contrasts[
            "overall_top1_regret_increase_r2"
        ]
        <= float(rule["maximum_overall_top1_regret_increase_r2"]) + TIE_ATOL,
        "opportunity_top1_regret_improved": contrasts[
            "opportunity_top1_regret_reduction_r2"
        ]
        >= float(rule["minimum_opportunity_top1_regret_reduction_r2"])
        - TIE_ATOL,
        "control_top1_regret_preserved": contrasts[
            "control_top1_regret_increase_r2"
        ]
        <= float(rule["maximum_control_top1_regret_increase_r2"]) + TIE_ATOL,
        "overall_top3_preserved": contrasts[
            "overall_top3_opportunity_regret_increase_r2"
        ]
        <= float(rule["maximum_overall_top3_opportunity_regret_increase_r2"])
        + TIE_ATOL,
        "runtime_competitive": contrasts["runtime_ratio_to_linear_cv"]
        <= float(rule["maximum_aggregate_runtime_ratio_to_linear_cv"]) + TIE_ATOL,
    }
    passed = bool(all(checks.values()))
    return {
        "status": rule["result_if_pass"] if passed else rule["result_if_fail"],
        "primary_subset_id": PRIMARY_ID,
        "passed": passed,
        "checks": checks,
        "contrasts": contrasts,
        "promotion": None,
    }


def summarize(
    panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rows = ranking_rows(panels)
    summaries = ranking_summaries(rows)
    costs = cost_rows(panels)
    return {
        "schema_version": 1,
        "status": "completed_large_head_subset_generalization",
        "panel_count": len(panels),
        "candidate_count": sum(len(panel["candidates"]) for panel in panels),
        "ranking_rows": rows,
        "ranking_summaries": summaries,
        "cost_rows": costs,
        "decision": decision(summaries, costs, protocol),
        "new_model_fit_count": 0,
        "new_overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }


def report(summary: Mapping[str, Any]) -> str:
    highlights = {
        BASELINE_ID,
        "quadratic_ridge",
        "knn",
        PRIMARY_ID,
        ACCURACY_CONTROL_ID,
        FULL_ID,
    }
    lines = [
        "# Large-panel cheap head-subset generalization",
        "",
        f"Decision: **{summary['decision']['status']}**.",
        "",
        "| Group | Subset | Spearman | Top-1 regret | Exact best | Top-3 regret | Top-3 utility regret | Top-3 recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["ranking_summaries"]:
        if row["subset_id"] in highlights:
            lines.append(
                f"| {row['group']} | {row['subset_id']} | {row['mean_spearman']:.4f} | "
                f"{row['mean_top1_regret_r2']:.5f} | {row['exact_best_rate']:.3f} | "
                f"{row['mean_top3_opportunity_regret_r2']:.5f} | "
                f"{row['mean_top3_utility_regret_r2']:.5f} | {row['mean_top3_set_recall']:.3f} |"
            )
    lines.extend(["", "## Archived cost", "", "| Subset | Relative to Linear CV |", "|---|---:|"])
    for row in summary["cost_rows"]:
        if row["subset_id"] in highlights:
            lines.append(f"| {row['subset_id']} | {row['relative_to_linear_cv']:.2f}x |")
    lines.extend(["", "## Primary frozen gate", ""])
    for name, passed in summary["decision"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Only quadratic Ridge plus kNN is eligible for the frozen decision.",
            "- All other subsets are diagnostic and cannot be selected post outcome.",
            "- No new model or OverlapIndex fit was run.",
            "- No reserved backbone was evaluated.",
            "- No production method is promoted from this development experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def _source_hashes() -> dict[str, str]:
    paths = {
        "protocol": PROTOCOL_PATH,
        "runner": Path(__file__).resolve(),
        "source_protocol": prepare.PROTOCOL_PATH,
        "source_runner": Path(source_run.__file__).resolve(),
        "source_raw": ROOT / "artifacts/continuous_large_structured_opportunity_benchmark/results/raw_results.json",
        "source_summary": ROOT / "artifacts/continuous_large_structured_opportunity_benchmark/results/summary.json",
        "source_manifest": ROOT / "artifacts/continuous_large_structured_opportunity_benchmark/results/manifest.json",
        "selection_protocol": ranker.PROTOCOL_PATH,
        "selection_runner": Path(ranker.__file__).resolve(),
        "selection_summary": ROOT / "artifacts/continuous_large_recoverable_ranker_comparison/summary.json",
        "selection_manifest": ROOT / "artifacts/continuous_large_recoverable_ranker_comparison/manifest.json",
        "prior_protocol": prior.PROTOCOL_PATH,
        "prior_runner": Path(prior.__file__).resolve(),
        "prior_summary": ROOT / "artifacts/continuous_coi_head_subset_cost_diagnostic/summary.json",
        "prior_manifest": ROOT / "artifacts/continuous_coi_head_subset_cost_diagnostic/manifest.json",
    }
    return {key: sha256_path(path) for key, path in paths.items()}


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    source_raw, _selection, _basis = validate_sources(protocol)
    panels = normalized_panels(selected_panels(source_raw, protocol))
    source_hashes = _source_hashes()
    output_root = output_root.resolve()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        raise FileExistsError("large head-subset terminal artifact already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize(panels, protocol)
    raw = {
        "schema_version": 1,
        "status": summary["status"],
        "protocol": protocol,
        "source_hashes": source_hashes,
        "panels": panels,
        "new_model_fit_count": 0,
        "new_overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }
    raw_path = output_root / "raw_results.json"
    summary_path = output_root / "summary.json"
    report_path = output_root / "report.md"
    prepare._atomic_json(raw_path, raw)
    prepare._atomic_json(summary_path, summary)
    prepare._atomic_bytes(report_path, report(summary).encode("utf-8"))
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    manifest = {
        "schema_version": 1,
        "status": summary["status"],
        "decision_status": summary["decision"]["status"],
        "source_hashes": source_hashes,
        "artifacts": {
            "raw_results.json": sha256_path(raw_path),
            "summary.json": sha256_path(summary_path),
            "report.md": sha256_path(report_path),
        },
        "counts": {
            "panels": 24,
            "candidates": 120,
            "subsets": 15,
            "new_model_fits": 0,
            "new_overlapindex_fits": 0,
            "reserved_backbones": 0,
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
        "promotion": None,
    }
    prepare._atomic_json(terminal, manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(canonical_json(run(args.output_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
