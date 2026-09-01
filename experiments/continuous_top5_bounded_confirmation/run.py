"""Frozen unused-seed confirmation of bounded top-five winner filling."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_kernel_alignment import run as kernel
from experiments.continuous_multiview_ranking import run as multiview
from experiments.continuous_null_diagnostic import run as null_parent
from experiments.continuous_prototype_capacity import run as capacity
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_save_ranking import run as save
from experiments.continuous_shortlist_depth_audit import run as audit_parent
from experiments.continuous_top5_bounded_diagnostic import run as diagnostic_parent
from experiments.continuous_top5_shortlist_confirmation import run as confirmation_parent
from experiments.continuous_winner_fill_shortlist import run as winner_parent


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
RUNNER_PATH = Path(__file__).resolve()
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSTIC_ROOT = ROOT / "artifacts" / "continuous_top5_bounded_diagnostic"
METHODS = (
    "broad_rff",
    "save_rff_current",
    "unbounded_winner_fill",
    "bounded_2k",
    "linear_probe",
)
DEPTHS = (1, 3, 5)
NEW_SEEDS = (3407, 3511, 3607, 3701, 3803)
FAMILIES = confirmation_parent.FAMILIES
TIE_ATOL = 1.0e-12
FAMILY_DEGRADATION_LIMIT = 0.03
BOOTSTRAP_SEED = 2026090601
BOOTSTRAP_RESAMPLES = 10_000


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or common._sha256(PROTOCOL_PATH) != expected:
        raise ValueError("bounded confirmation protocol SHA-256 sidecar does not match")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_unused_seed_confirmation_before_outcomes":
        raise ValueError("bounded confirmation protocol is not frozen")
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if tuple(protocol["seeds"]) != NEW_SEEDS:
        raise ValueError("bounded confirmation seed set does not match code")
    if tuple(protocol["families"]) != FAMILIES:
        raise ValueError("bounded confirmation family set does not match code")
    if set(protocol["methods"]) != set(METHODS):
        raise ValueError("bounded confirmation method set does not match code")
    if tuple(protocol["shortlist_depths"]) != DEPTHS:
        raise ValueError("bounded confirmation shortlist depths do not match code")
    if int(protocol["primary_shortlist_depth"]) != 5:
        raise ValueError("bounded confirmation primary depth does not match code")
    if float(protocol["ties"]["absolute_tolerance"]) != TIE_ATOL:
        raise ValueError("bounded confirmation tie tolerance does not match code")
    if int(protocol["bootstrap"]["seed"]) != BOOTSTRAP_SEED:
        raise ValueError("bounded confirmation bootstrap seed does not match code")
    if int(protocol["bootstrap"]["resamples"]) != BOOTSTRAP_RESAMPLES:
        raise ValueError("bounded confirmation bootstrap count does not match code")
    if tuple(protocol["kernel_alignment"]["bandwidth_multipliers"]) != (
        kernel.BANDWIDTH_MULTIPLIERS
    ):
        raise ValueError("bounded confirmation kernel bandwidths do not match code")
    if tuple(protocol["reference_heads"]["ids"]) != tuple(
        residual._head_estimators()
    ):
        raise ValueError("bounded confirmation reference heads do not match code")
    kernel._validate_protocol_against_code(
        {
            **protocol,
            "seeds": [1601, 1709, 1811],
            "kernel_alignment": {
                **protocol["kernel_alignment"],
                "eligible_estimand": kernel.ELIGIBLE_ESTIMAND,
            },
        }
    )


def _validate_diagnostic(protocol: Mapping[str, Any], root: Path) -> None:
    root = root.resolve()
    ancestry = protocol["ancestry"]
    paths = {
        "diagnostic_summary_sha256": root / "summary.json",
        "diagnostic_report_sha256": root / "report.md",
        "diagnostic_manifest_sha256": root / "manifest.json",
    }
    for key, path in paths.items():
        if common._sha256(path) != ancestry[key]:
            raise ValueError(f"bounded authorizing diagnostic mismatch for {key}")
    if common._sha256(diagnostic_parent.PROTOCOL_PATH) != ancestry[
        "diagnostic_protocol_sha256"
    ]:
        raise ValueError("bounded authorizing diagnostic protocol changed")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if summary.get("decision", {}).get("status") != ancestry[
        "required_diagnostic_status"
    ]:
        raise ValueError("bounded confirmation is not authorized by the diagnostic")
    if manifest.get("implementation_commit") != ancestry[
        "diagnostic_implementation_commit"
    ]:
        raise ValueError("bounded diagnostic implementation commit mismatch")
    for name, descriptor in manifest["files"].items():
        if common._sha256(root / name) != descriptor["sha256"]:
            raise ValueError(f"bounded diagnostic terminal hash mismatch for {name}")
        if (root / name).stat().st_size != int(descriptor["bytes"]):
            raise ValueError(f"bounded diagnostic terminal size mismatch for {name}")


def ranking_groups(panel: Mapping[str, Any], method: str) -> list[tuple[int, ...]]:
    if method in {"broad_rff", "save_rff_current", "linear_probe"}:
        return winner_parent.ranking_groups(panel, method)
    if method == "unbounded_winner_fill":
        return winner_parent.ranking_groups(panel, "winner_fill")
    if method == "bounded_2k":
        return audit_parent.ranking_groups(panel, method, 5)
    raise ValueError(f"unknown bounded confirmation method {method!r}")


def _runtime_seconds(panel: Mapping[str, Any], method: str) -> float:
    if method in {"unbounded_winner_fill", "bounded_2k"}:
        return float(
            winner_parent._runtime_seconds(panel, "broad_rff")
            + winner_parent._runtime_seconds(panel, "save_rff_current")
        )
    return winner_parent._runtime_seconds(panel, method)


def panel_metric(panel: Mapping[str, Any], method: str) -> dict[str, Any]:
    candidates = panel["candidates"]
    reference = np.asarray(
        [candidate["best_head_envelope_r2"] for candidate in candidates],
        dtype=np.float64,
    )
    if reference.shape != (16,) or not np.all(np.isfinite(reference)):
        raise ValueError("bounded confirmation panel requires 16 finite envelope values")
    groups = ranking_groups(panel, method)
    scores = winner_parent._ordinal_scores(groups, len(candidates))
    result: dict[str, Any] = {
        "seed": int(panel["seed"]),
        "family": str(panel["family"]),
        "panel_id": f"{panel['family']}__s{int(panel['seed'])}",
        "method": method,
        "rank_groups": [
            [candidates[index]["candidate_id"] for index in group] for group in groups
        ],
        "spearman": winner_parent._spearman(reference, scores),
        "material_pair_concordance": winner_parent._material_concordance(
            reference, scores
        ),
        "descriptive_panel_runtime_seconds": _runtime_seconds(panel, method),
    }
    for k in DEPTHS:
        for key, value in winner_parent._top_k_metrics(reference, groups, k).items():
            result[f"top_{k}_{key}"] = value
    return result


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    metric: str,
    seed_offset: int,
) -> dict[str, Any]:
    lookup = {(str(row["method"]), str(row["panel_id"])): row for row in rows}
    strata: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != left:
            continue
        other = lookup[(right, str(row["panel_id"]))]
        strata[str(row["family"])].append(float(row[metric]) - float(other[metric]))
    if set(strata) != set(FAMILIES) or any(len(values) != 5 for values in strata.values()):
        raise ValueError("bounded confirmation bootstrap requires five panels per family")
    arrays = [np.asarray(strata[family], dtype=np.float64) for family in FAMILIES]
    estimate = float(np.mean([np.mean(values) for values in arrays]))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    family_draws = []
    for values in arrays:
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
        family_draws.append(np.mean(values[indices], axis=1))
    draws = np.mean(np.stack(family_draws, axis=1), axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "estimate": estimate,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "by_family": {
            family: float(np.mean(strata[family])) for family in FAMILIES
        },
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED + seed_offset,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "spearman",
        "material_pair_concordance",
        "descriptive_panel_runtime_seconds",
        *[
            f"top_{k}_{metric}"
            for k in DEPTHS
            for metric in (
                "opportunity_regret_r2",
                "mean_utility_regret_r2",
                "exact_best_coverage",
                "within_0_01_coverage",
                "material_precision",
            )
        ],
    )
    output = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 30:
            raise ValueError("bounded confirmation method is missing panels")
        output.append(
            {
                "method": method,
                "panel_count": len(selected),
                **{
                    f"mean_{metric}": (
                        float(np.mean([row[metric] for row in selected]))
                        if all(row[metric] is not None for row in selected)
                        else None
                    )
                    for metric in metrics
                },
                "by_family_top5_mean_utility_regret_r2": {
                    family: float(
                        np.mean(
                            [
                                row["top_5_mean_utility_regret_r2"]
                                for row in selected
                                if row["family"] == family
                            ]
                        )
                    )
                    for family in FAMILIES
                },
            }
        )
    return output


def _decision(
    rows: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lookup = {
        (str(row["left"]), str(row["right"]), str(row["metric"])): row
        for row in contrasts
    }
    mean_regret = lookup[("bounded_2k", "broad_rff", "top_5_mean_utility_regret_r2")]
    opportunity = lookup[("bounded_2k", "broad_rff", "top_5_opportunity_regret_r2")]
    precision = lookup[("bounded_2k", "broad_rff", "top_5_material_precision")]
    broad = {str(row["panel_id"]): row for row in rows if row["method"] == "broad_rff"}
    top1 = [
        abs(
            float(row["top_1_opportunity_regret_r2"])
            - float(broad[str(row["panel_id"])]["top_1_opportunity_regret_r2"])
        )
        for row in rows
        if row["method"] == "bounded_2k"
    ]
    complete = len(rows) == 150 and all(
        row["spearman"] is not None
        and row["material_pair_concordance"] is not None
        and np.isfinite(
            [
                row[f"top_{k}_{metric}"]
                for k in DEPTHS
                for metric in (
                    "opportunity_regret_r2",
                    "mean_utility_regret_r2",
                    "material_precision",
                )
            ]
        ).all()
        for row in rows
    )
    family_differences = dict(mean_regret["by_family"])
    checks = [
        {
            "criterion": "top5_mean_regret_improves_broad",
            "observed_upper_95": mean_regret["upper_95"],
            "required": "<0",
            "pass": bool(mean_regret["upper_95"] < 0.0),
        },
        {
            "criterion": "top1_exact_panel_parity_with_broad",
            "observed_max_absolute_difference": max(top1),
            "maximum": TIE_ATOL,
            "pass": bool(max(top1) <= TIE_ATOL),
        },
        {
            "criterion": "top5_opportunity_noninferior_to_broad",
            "observed_upper_95": opportunity["upper_95"],
            "maximum": 0.01,
            "pass": bool(opportunity["upper_95"] <= 0.01),
        },
        {
            "criterion": "top5_material_precision_noninferior_to_broad",
            "observed_lower_95": precision["lower_95"],
            "minimum": -0.02,
            "pass": bool(precision["lower_95"] >= -0.02),
        },
        {
            "criterion": "family_mean_degradation_bounded",
            "observed_by_family": family_differences,
            "observed_maximum": max(family_differences.values()),
            "maximum": FAMILY_DEGRADATION_LIMIT,
            "pass": bool(max(family_differences.values()) <= FAMILY_DEGRADATION_LIMIT),
        },
        {
            "criterion": "all_panel_metrics_complete_and_finite",
            "observed_row_count": len(rows),
            "required_row_count": 150,
            "pass": bool(complete),
        },
    ]
    return {
        "status": "pass" if all(check["pass"] for check in checks) else "fail",
        "candidate": "bounded_2k",
        "utility": "five_backbone_shortlist",
        "checks": checks,
        "scope": "unused_seed_synthetic_confirmation_only",
    }


def summarize(protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(seed, family) for seed in NEW_SEEDS for family in FAMILIES}
    observed = [(int(panel["seed"]), str(panel["family"])) for panel in panels]
    if len(observed) != 30 or len(set(observed)) != 30 or set(observed) != expected:
        raise ValueError("bounded confirmation panel grid is incomplete")
    rows = [panel_metric(panel, method) for panel in panels for method in METHODS]
    specs = (
        ("bounded_2k", "broad_rff", "top_5_mean_utility_regret_r2"),
        ("bounded_2k", "broad_rff", "top_5_opportunity_regret_r2"),
        ("bounded_2k", "broad_rff", "top_5_material_precision"),
        ("bounded_2k", "unbounded_winner_fill", "top_5_mean_utility_regret_r2"),
        ("bounded_2k", "unbounded_winner_fill", "top_5_opportunity_regret_r2"),
        ("bounded_2k", "unbounded_winner_fill", "top_5_material_precision"),
        ("bounded_2k", "save_rff_current", "top_5_mean_utility_regret_r2"),
        ("bounded_2k", "broad_rff", "top_3_mean_utility_regret_r2"),
    )
    contrasts = [
        _paired_bootstrap(rows, left, right, metric, index * 1000)
        for index, (left, right, metric) in enumerate(specs)
    ]
    return {
        "schema_version": 1,
        "status": "completed_unused_seed_bounded_confirmation",
        "panel_count": len(panels),
        "panel_method_row_count": len(rows),
        "method_summaries": _aggregate(rows),
        "panel_metrics": rows,
        "contrasts": contrasts,
        "decision": _decision(rows, contrasts),
        "protocol": dict(protocol),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous bounded top-five confirmation",
        "",
        f"Decision: **{summary['decision']['status'].upper()}**.",
        "",
        "Negative regret contrasts favor bounded filling; positive material-precision contrasts favor it.",
        "",
        "| Method | Top-1 regret | Top-3 mean regret | Top-5 opportunity regret | Top-5 mean regret | Top-5 material precision | Panel seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["method_summaries"]:
        lines.append(
            f"| {row['method']} | {row['mean_top_1_opportunity_regret_r2']:.5f} | "
            f"{row['mean_top_3_mean_utility_regret_r2']:.5f} | "
            f"{row['mean_top_5_opportunity_regret_r2']:.5f} | "
            f"{row['mean_top_5_mean_utility_regret_r2']:.5f} | "
            f"{row['mean_top_5_material_precision']:.3f} | "
            f"{row['mean_descriptive_panel_runtime_seconds']:.3f} |"
        )
    lines.extend(["", "## Frozen checks", ""])
    for check in summary["decision"]["checks"]:
        lines.append(f"- {'PASS' if check['pass'] else 'FAIL'}: `{check['criterion']}`")
    lines.extend(
        [
            "",
            "A pass confirms only the frozen synthetic five-backbone shortlist utility. It supports implementation and real-panel validation but does not alone authorize public integration.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _repository_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_paths(diagnostic_root: Path) -> dict[str, Path]:
    result = {
        "protocol": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256_PATH,
        "runner": RUNNER_PATH,
        "readme": PACKAGE_DIR / "README.md",
        "focused_test": ROOT / "tests" / "test_continuous_top5_bounded_confirmation.py",
        "diagnostic_protocol": diagnostic_parent.PROTOCOL_PATH,
        "diagnostic_runner": diagnostic_parent.RUNNER_PATH,
        "audit_runner": audit_parent.RUNNER_PATH,
        "confirmation_parent_runner": confirmation_parent.RUNNER_PATH,
        "winner_parent_runner": winner_parent.RUNNER_PATH,
        "kernel_runner": kernel.RUNNER_PATH,
        "null_parent_runner": null_parent.RUNNER_PATH,
        "capacity_runner": capacity.RUNNER_PATH,
        "multiview_runner": multiview.RUNNER_PATH,
        "save_runner": save.RUNNER_PATH,
        "residual_runner": residual.RUNNER_PATH,
        "common_runner": common.RUNNER_PATH,
        "project_configuration": ROOT / "pyproject.toml",
    }
    for name in ("summary.json", "report.md", "manifest.json"):
        result[f"diagnostic_{name}"] = diagnostic_root.resolve() / name
    result.update(
        {
            f"overlapindex_{path.name}": path
            for path in sorted((ROOT / "overlapindex").glob("*.py"))
        }
    )
    return result


def run(
    output: Path, diagnostic_root: Path = DEFAULT_DIAGNOSTIC_ROOT
) -> dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol(protocol)
    diagnostic_root = diagnostic_root.resolve()
    _validate_diagnostic(protocol, diagnostic_root)
    implementation_commit = os.environ.get("CONTINUOUS_OI_COMMIT")
    if implementation_commit != _repository_head():
        raise ValueError("CONTINUOUS_OI_COMMIT must equal the frozen Git HEAD")
    source_hashes = {
        key: common._sha256(path)
        for key, path in _source_paths(diagnostic_root).items()
    }
    output = output.resolve()
    for name in ("raw_results.json", "summary.json", "report.md", "manifest.json"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite {output / name}")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    completed = 0
    for seed in NEW_SEEDS:
        for family in FAMILIES:
            path = checkpoints / f"seed_{seed}__{family}.json"
            if path.exists():
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                if checkpoint.get("source_hashes") != source_hashes:
                    raise ValueError(f"stale bounded confirmation checkpoint: {path}")
                panel = checkpoint["panel"]
            else:
                started = time.perf_counter()
                panel = kernel.run_panel(protocol, seed, family)
                common._atomic_json(
                    path,
                    {
                        "schema_version": 1,
                        "source_hashes": source_hashes,
                        "panel_wall_seconds": float(time.perf_counter() - started),
                        "panel": panel,
                    },
                )
            panels.append(panel)
            completed += 1
            print(f"completed {completed}/30: seed={seed} family={family}", flush=True)
    raw = {
        "schema_version": 1,
        "artifact_status": "completed",
        "implementation_commit": implementation_commit,
        "source_hashes": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "protocol": protocol,
        "panels": panels,
    }
    summary = summarize(protocol, panels)
    common._atomic_json(output / "raw_results.json", raw)
    common._atomic_json(output / "summary.json", summary)
    _atomic_text(output / "report.md", _report(summary))
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "implementation_commit": implementation_commit,
        "source_hashes": source_hashes,
        "checkpoint_count": len(panels),
        "files": {
            name: {
                "sha256": common._sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in ("raw_results.json", "summary.json", "report.md")
        },
    }
    common._atomic_json(output / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-root", type=Path, default=DEFAULT_DIAGNOSTIC_ROOT
    )
    args = parser.parse_args()
    summary = run(args.output, args.diagnostic_root)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
