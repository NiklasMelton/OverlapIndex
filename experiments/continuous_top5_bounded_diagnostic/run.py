"""No-refit bounded top-five diagnostic on frozen confirmation panels."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_shortlist_depth_audit import run as audit_parent
from experiments.continuous_top5_shortlist_confirmation import run as confirmation_parent
from experiments.continuous_winner_fill_shortlist import run as winner_parent


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
RUNNER_PATH = Path(__file__).resolve()
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "artifacts" / "continuous_top5_shortlist_confirmation"
METHODS = ("broad_rff", "unbounded_winner_fill", "bounded_2k")
FAMILIES = confirmation_parent.FAMILIES
SEEDS = confirmation_parent.NEW_SEEDS
SHORTLIST_DEPTH = 5
TIE_ATOL = 1.0e-12
FAMILY_DEGRADATION_LIMIT = 0.03
BOOTSTRAP_SEED = 2026090501
BOOTSTRAP_RESAMPLES = 10_000


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or common._sha256(PROTOCOL_PATH) != expected:
        raise ValueError("bounded top-five protocol SHA-256 sidecar does not match")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_post_confirmation_diagnostic_before_metrics":
        raise ValueError("bounded top-five diagnostic protocol is not frozen")
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if tuple(protocol["families"]) != FAMILIES:
        raise ValueError("bounded top-five family set does not match code")
    if tuple(protocol["input"]["seeds"]) != SEEDS:
        raise ValueError("bounded top-five seed set does not match code")
    if set(protocol["methods"]) != set(METHODS):
        raise ValueError("bounded top-five method set does not match code")
    if int(protocol["shortlist_depth"]) != SHORTLIST_DEPTH:
        raise ValueError("bounded top-five shortlist depth does not match code")
    if float(protocol["ties"]["absolute_tolerance"]) != TIE_ATOL:
        raise ValueError("bounded top-five tie tolerance does not match code")
    if int(protocol["bootstrap"]["seed"]) != BOOTSTRAP_SEED:
        raise ValueError("bounded top-five bootstrap seed does not match code")
    if int(protocol["bootstrap"]["resamples"]) != BOOTSTRAP_RESAMPLES:
        raise ValueError("bounded top-five bootstrap count does not match code")
    if float(protocol["decision_rule"]["safety"][3].split("<=")[-1]) != FAMILY_DEGRADATION_LIMIT:
        raise ValueError("bounded top-five family limit does not match code")


def _validate_input(protocol: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    descriptor = protocol["input"]
    paths = {
        "raw_results_sha256": root / "raw_results.json",
        "summary_sha256": root / "summary.json",
        "report_sha256": root / "report.md",
        "manifest_sha256": root / "manifest.json",
    }
    for key, path in paths.items():
        if common._sha256(path) != descriptor[key]:
            raise ValueError(f"bounded top-five input hash mismatch for {key}")
    if common._sha256(confirmation_parent.PROTOCOL_PATH) != descriptor[
        "protocol_sha256"
    ]:
        raise ValueError("bounded top-five input protocol changed")
    if common._sha256(audit_parent.PROTOCOL_PATH) != protocol["method_ancestry"][
        "protocol_sha256"
    ]:
        raise ValueError("bounded top-five method protocol changed")
    raw = json.loads((root / "raw_results.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if raw.get("artifact_status") != descriptor["required_artifact_status"]:
        raise ValueError("bounded top-five input artifact is not completed")
    if manifest.get("artifact_status") != descriptor["required_artifact_status"]:
        raise ValueError("bounded top-five input manifest is not completed")
    if summary.get("decision", {}).get("status") != descriptor[
        "required_decision_status"
    ]:
        raise ValueError("bounded top-five input decision status changed")
    if raw.get("implementation_commit") != descriptor["implementation_commit"]:
        raise ValueError("bounded top-five input implementation commit mismatch")
    if manifest.get("implementation_commit") != descriptor["implementation_commit"]:
        raise ValueError("bounded top-five input manifest commit mismatch")
    if manifest.get("source_hashes") != raw.get("source_hashes"):
        raise ValueError("bounded top-five input source identities disagree")
    for name, file_descriptor in manifest["files"].items():
        path = root / name
        if common._sha256(path) != file_descriptor["sha256"]:
            raise ValueError(f"bounded top-five terminal hash mismatch for {name}")
        if path.stat().st_size != int(file_descriptor["bytes"]):
            raise ValueError(f"bounded top-five terminal size mismatch for {name}")
    panels = raw.get("panels", [])
    expected = {(seed, family) for seed in SEEDS for family in FAMILIES}
    observed = [(int(panel["seed"]), str(panel["family"])) for panel in panels]
    if len(observed) != 30 or len(set(observed)) != 30 or set(observed) != expected:
        raise ValueError("bounded top-five input panel grid is incomplete")
    if any(len(panel.get("candidates", [])) != 16 for panel in panels):
        raise ValueError("bounded top-five input candidate grid is incomplete")
    return panels


def ranking_groups(panel: Mapping[str, Any], method: str) -> list[tuple[int, ...]]:
    if method == "broad_rff":
        return winner_parent.ranking_groups(panel, method)
    if method == "unbounded_winner_fill":
        return winner_parent.ranking_groups(panel, "winner_fill")
    if method == "bounded_2k":
        return audit_parent.ranking_groups(panel, method, SHORTLIST_DEPTH)
    raise ValueError(f"unknown bounded top-five method {method!r}")


def panel_metric(panel: Mapping[str, Any], method: str) -> dict[str, Any]:
    candidates = panel["candidates"]
    reference = np.asarray(
        [candidate["best_head_envelope_r2"] for candidate in candidates],
        dtype=np.float64,
    )
    if reference.shape != (16,) or not np.all(np.isfinite(reference)):
        raise ValueError("bounded top-five panel requires 16 finite envelope values")
    groups = ranking_groups(panel, method)
    metrics = winner_parent._top_k_metrics(reference, groups, SHORTLIST_DEPTH)
    top1 = winner_parent._top_k_metrics(reference, groups, 1)
    return {
        "seed": int(panel["seed"]),
        "family": str(panel["family"]),
        "panel_id": f"{panel['family']}__s{int(panel['seed'])}",
        "method": method,
        "rank_groups": [
            [candidates[index]["candidate_id"] for index in group] for group in groups
        ],
        "top_1_opportunity_regret_r2": float(top1["opportunity_regret_r2"]),
        "top_5_opportunity_regret_r2": float(metrics["opportunity_regret_r2"]),
        "top_5_mean_utility_regret_r2": float(metrics["mean_utility_regret_r2"]),
        "top_5_material_precision": float(metrics["material_precision"]),
        "top_5_exact_best_coverage": float(metrics["exact_best_coverage"]),
        "top_5_within_0_01_coverage": float(metrics["within_0_01_coverage"]),
    }


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
        raise ValueError("bounded top-five bootstrap requires five panels per family")
    arrays = [np.asarray(strata[family], dtype=np.float64) for family in FAMILIES]
    estimate = float(np.mean([np.mean(values) for values in arrays]))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws_by_family = []
    for values in arrays:
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
        draws_by_family.append(np.mean(values[indices], axis=1))
    draws = np.mean(np.stack(draws_by_family, axis=1), axis=1)
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
        "top_1_opportunity_regret_r2",
        "top_5_opportunity_regret_r2",
        "top_5_mean_utility_regret_r2",
        "top_5_material_precision",
        "top_5_exact_best_coverage",
        "top_5_within_0_01_coverage",
    )
    output = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 30:
            raise ValueError("bounded top-five method is missing panels")
        output.append(
            {
                "method": method,
                "panel_count": len(selected),
                **{
                    f"mean_{metric}": float(np.mean([row[metric] for row in selected]))
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
    top1_differences = [
        abs(
            float(row["top_1_opportunity_regret_r2"])
            - float(broad[str(row["panel_id"])]["top_1_opportunity_regret_r2"])
        )
        for row in rows
        if row["method"] == "bounded_2k"
    ]
    complete = len(rows) == 90 and all(
        np.isfinite(
            [
                row["top_1_opportunity_regret_r2"],
                row["top_5_opportunity_regret_r2"],
                row["top_5_mean_utility_regret_r2"],
                row["top_5_material_precision"],
                row["top_5_exact_best_coverage"],
                row["top_5_within_0_01_coverage"],
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
            "observed_max_absolute_difference": max(top1_differences),
            "maximum": TIE_ATOL,
            "pass": bool(max(top1_differences) <= TIE_ATOL),
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
            "required_row_count": 90,
            "pass": bool(complete),
        },
    ]
    return {
        "status": (
            "bounded_top5_confirmation_justified"
            if all(check["pass"] for check in checks)
            else "stop_no_bounded_top5_confirmation"
        ),
        "candidate": "bounded_2k",
        "utility": "five_backbone_shortlist",
        "checks": checks,
        "scope": "post_outcome_diagnostic_only",
    }


def summarize(protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(seed, family) for seed in SEEDS for family in FAMILIES}
    observed = [(int(panel["seed"]), str(panel["family"])) for panel in panels]
    if len(observed) != 30 or len(set(observed)) != 30 or set(observed) != expected:
        raise ValueError("bounded top-five panel grid is incomplete")
    rows = [panel_metric(panel, method) for panel in panels for method in METHODS]
    specs = (
        ("bounded_2k", "broad_rff", "top_5_mean_utility_regret_r2"),
        ("bounded_2k", "broad_rff", "top_5_opportunity_regret_r2"),
        ("bounded_2k", "broad_rff", "top_5_material_precision"),
        ("bounded_2k", "unbounded_winner_fill", "top_5_mean_utility_regret_r2"),
        ("bounded_2k", "unbounded_winner_fill", "top_5_opportunity_regret_r2"),
        ("bounded_2k", "unbounded_winner_fill", "top_5_material_precision"),
    )
    contrasts = [
        _paired_bootstrap(rows, left, right, metric, index * 1000)
        for index, (left, right, metric) in enumerate(specs)
    ]
    return {
        "schema_version": 1,
        "status": "completed_post_confirmation_diagnostic",
        "panel_count": len(panels),
        "panel_method_row_count": len(rows),
        "method_summaries": _aggregate(rows),
        "panel_metrics": rows,
        "contrasts": contrasts,
        "decision": _decision(rows, contrasts),
        "protocol": dict(protocol),
    }


def _report(summary: Mapping[str, Any]) -> str:
    methods = {row["method"]: row for row in summary["method_summaries"]}
    contrasts = {
        (row["left"], row["right"], row["metric"]): row
        for row in summary["contrasts"]
    }
    bounded_broad = contrasts[("bounded_2k", "broad_rff", "top_5_mean_utility_regret_r2")]
    bounded_unbounded = contrasts[("bounded_2k", "unbounded_winner_fill", "top_5_mean_utility_regret_r2")]
    lines = [
        "# Continuous bounded top-five diagnostic",
        "",
        f"Decision: **{summary['decision']['status']}**.",
        "",
        "Negative regret contrasts favor the left-hand method; positive material-precision contrasts favor it.",
        "",
        "| Method | Top-1 regret | Top-5 opportunity regret | Top-5 mean regret | Top-5 material precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = methods[method]
        lines.append(
            f"| {method} | {row['mean_top_1_opportunity_regret_r2']:.5f} | "
            f"{row['mean_top_5_opportunity_regret_r2']:.5f} | "
            f"{row['mean_top_5_mean_utility_regret_r2']:.5f} | "
            f"{row['mean_top_5_material_precision']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Regret contrasts",
            "",
            f"- bounded minus broad: {bounded_broad['estimate']:+.5f} "
            f"[{bounded_broad['lower_95']:+.5f}, {bounded_broad['upper_95']:+.5f}]",
            f"- bounded minus unbounded: {bounded_unbounded['estimate']:+.5f} "
            f"[{bounded_unbounded['lower_95']:+.5f}, {bounded_unbounded['upper_95']:+.5f}]",
            "",
            "## Frozen checks",
            "",
        ]
    )
    for check in summary["decision"]["checks"]:
        lines.append(f"- {'PASS' if check['pass'] else 'FAIL'}: `{check['criterion']}`")
    lines.extend(
        [
            "",
            "This is a post-outcome diagnostic reanalysis. A pass can justify one new unused-seed bounded-top-five confirmation only.",
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


def _source_paths(input_root: Path) -> dict[str, Path]:
    result = {
        "protocol": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256_PATH,
        "runner": RUNNER_PATH,
        "readme": PACKAGE_DIR / "README.md",
        "focused_test": ROOT / "tests" / "test_continuous_top5_bounded_diagnostic.py",
        "confirmation_protocol": confirmation_parent.PROTOCOL_PATH,
        "confirmation_runner": confirmation_parent.RUNNER_PATH,
        "audit_protocol": audit_parent.PROTOCOL_PATH,
        "audit_runner": audit_parent.RUNNER_PATH,
        "winner_parent_runner": winner_parent.RUNNER_PATH,
    }
    for name in ("raw_results.json", "summary.json", "report.md", "manifest.json"):
        result[f"confirmation_{name}"] = input_root.resolve() / name
    return result


def run(output: Path, input_root: Path = DEFAULT_INPUT) -> dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol(protocol)
    panels = _validate_input(protocol, input_root)
    implementation_commit = os.environ.get("CONTINUOUS_OI_COMMIT")
    if implementation_commit != _repository_head():
        raise ValueError("CONTINUOUS_OI_COMMIT must equal the frozen Git HEAD")
    source_hashes = {
        key: common._sha256(path) for key, path in _source_paths(input_root).items()
    }
    output = output.resolve()
    for name in ("summary.json", "report.md", "manifest.json"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite {output / name}")
    summary = summarize(protocol, panels)
    summary["implementation_commit"] = implementation_commit
    summary["source_hashes"] = source_hashes
    common._atomic_json(output / "summary.json", summary)
    _atomic_text(output / "report.md", _report(summary))
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "implementation_commit": implementation_commit,
        "source_hashes": source_hashes,
        "input_panel_count": len(panels),
        "files": {
            name: {
                "sha256": common._sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in ("summary.json", "report.md")
        },
    }
    common._atomic_json(output / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    summary = run(args.output, args.input_root)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
