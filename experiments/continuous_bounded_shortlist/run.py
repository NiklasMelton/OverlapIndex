"""Frozen bounded broad-retrieval continuous shortlist experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import os
import platform
from pathlib import Path
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
from experiments.continuous_winner_fill_shortlist import run as parent


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
MOTIVATION_PATH = PACKAGE_DIR / "motivation.json"
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_MOTIVATION_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "continuous_winner_fill_shortlist"
)
METHODS = (
    "broad_rff",
    "save_rff_current",
    "unbounded_winner_fill",
    "bounded6_rerank",
    "linear_probe",
)
SHORTLIST_SIZES = (1, 2, 3, 5)
NEW_SEEDS = (2411, 2503, 2609, 2711, 2801)
FAMILIES = (
    "linear",
    "quadratic_product",
    "radial_quadratic",
    "absolute_value",
    "sinusoid",
    "triangle_wave",
)
PRIMARY_SHORTLIST_SIZE = 3
RETRIEVAL_DEPTH = 6
TIE_ATOL = 1.0e-12
EQUIVALENCE_BAND = 0.01
FAMILY_DEGRADATION_LIMIT = 0.03
BOOTSTRAP_SEED = 2026090201
BOOTSTRAP_RESAMPLES = 10_000


def _load_protocol() -> dict[str, Any]:
    expected_hash = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected_hash) != 64 or common._sha256(PROTOCOL_PATH) != expected_hash:
        raise ValueError("bounded-shortlist protocol SHA-256 sidecar does not match")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_exploratory_before_outcomes":
        raise ValueError("bounded-shortlist protocol is not frozen")
    return protocol


def _load_motivation() -> dict[str, Any]:
    motivation = json.loads(MOTIVATION_PATH.read_text(encoding="utf-8"))
    if motivation.get("status") != "post_outcome_motivation_only":
        raise ValueError("bounded-shortlist motivation status is invalid")
    return motivation


def _validate_protocol_against_code(protocol: Mapping[str, Any]) -> None:
    if tuple(protocol["seeds"]) != NEW_SEEDS:
        raise ValueError("frozen bounded-shortlist seed set does not match code")
    if tuple(protocol["families"]) != FAMILIES:
        raise ValueError("frozen bounded-shortlist family set does not match code")
    if set(protocol["ranking_methods"]) != set(METHODS):
        raise ValueError("frozen bounded-shortlist method set does not match code")
    if tuple(protocol["shortlist_sizes"]) != SHORTLIST_SIZES:
        raise ValueError("frozen shortlist sizes do not match code")
    if int(protocol["primary_shortlist_size"]) != PRIMARY_SHORTLIST_SIZE:
        raise ValueError("frozen primary shortlist size does not match code")
    if int(protocol["broad_retrieval_depth"]) != RETRIEVAL_DEPTH:
        raise ValueError("frozen retrieval depth does not match code")
    if float(protocol["ranking_ties"]["absolute_tolerance"]) != TIE_ATOL:
        raise ValueError("frozen ranking tolerance does not match code")
    if float(protocol["material_equivalence_band_r2"]) != EQUIVALENCE_BAND:
        raise ValueError("frozen material band does not match code")
    if int(protocol["bootstrap"]["seed"]) != BOOTSTRAP_SEED:
        raise ValueError("frozen bootstrap seed does not match code")
    if int(protocol["bootstrap"]["resamples"]) != BOOTSTRAP_RESAMPLES:
        raise ValueError("frozen bootstrap count does not match code")
    if tuple(protocol["kernel_alignment"]["bandwidth_multipliers"]) != (
        kernel.BANDWIDTH_MULTIPLIERS
    ):
        raise ValueError("frozen kernel bandwidths do not match parent code")
    if tuple(protocol["reference_heads"]["ids"]) != tuple(
        residual._head_estimators()
    ):
        raise ValueError("frozen reference heads do not match parent code")
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


def _validate_ancestry(
    protocol: Mapping[str, Any], root: Path = DEFAULT_MOTIVATION_ROOT
) -> None:
    root = root.resolve()
    ancestry = protocol["ancestry"]
    paths = {
        "motivation_raw_results_sha256": root / "raw_results.json",
        "motivation_summary_sha256": root / "summary.json",
        "motivation_report_sha256": root / "report.md",
    }
    for key, path in paths.items():
        if common._sha256(path) != ancestry[key]:
            raise ValueError(f"bounded-shortlist ancestry mismatch for {key}")
    if common._sha256(parent.PROTOCOL_PATH) != ancestry[
        "motivation_protocol_sha256"
    ]:
        raise ValueError("bounded-shortlist motivation protocol bytes changed")
    raw = json.loads((root / "raw_results.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if raw.get("implementation_commit") != ancestry[
        "motivation_implementation_commit"
    ]:
        raise ValueError("bounded-shortlist motivation commit does not match")
    if summary.get("decision", {}).get("status") != "fail":
        raise ValueError("bounded-shortlist motivation must remain a failed result")
    motivation = _load_motivation()
    for key, path in paths.items():
        source_key = key.replace("motivation_", "source_")
        if motivation.get(source_key) != common._sha256(path):
            raise ValueError(f"motivation record mismatch for {source_key}")


def _broad_pool(groups: Sequence[Sequence[int]]) -> set[int]:
    pool: set[int] = set()
    for group in groups:
        pool.update(int(index) for index in group)
        if len(pool) >= RETRIEVAL_DEPTH:
            break
    if len(pool) < RETRIEVAL_DEPTH:
        raise ValueError("broad ranking does not contain the frozen retrieval depth")
    return pool


def ranking_groups(panel: Mapping[str, Any], method: str) -> list[tuple[int, ...]]:
    if method in {"broad_rff", "save_rff_current", "linear_probe"}:
        return parent.ranking_groups(panel, method)
    if method == "unbounded_winner_fill":
        return parent.ranking_groups(panel, "winner_fill")
    if method != "bounded6_rerank":
        raise ValueError(f"unknown bounded-shortlist method {method!r}")
    broad = parent._component_values(panel, "broad_rff")
    current = parent._component_values(panel, "save_rff_current")
    broad_groups = parent._score_groups(broad)
    winner = broad_groups[0]
    pool = _broad_pool(broad_groups)
    pool.difference_update(winner)
    outside = set(range(len(broad))).difference(winner).difference(pool)
    groups = [winner]
    if pool:
        groups.extend(parent._score_groups(current, pool))
    if outside:
        groups.extend(parent._score_groups(current, outside))
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(len(broad))):
        raise AssertionError("bounded reranker did not produce a complete partition")
    return groups


def _runtime_seconds(panel: Mapping[str, Any], method: str) -> float:
    if method in {"unbounded_winner_fill", "bounded6_rerank"}:
        return float(
            parent._runtime_seconds(panel, "broad_rff")
            + parent._runtime_seconds(panel, "save_rff_current")
        )
    return parent._runtime_seconds(panel, method)


def panel_metric(panel: Mapping[str, Any], method: str) -> dict[str, Any]:
    candidates = panel["candidates"]
    reference = np.asarray(
        [candidate["best_head_envelope_r2"] for candidate in candidates],
        dtype=np.float64,
    )
    if reference.shape != (16,) or not np.all(np.isfinite(reference)):
        raise ValueError("panel envelope must contain 16 finite candidates")
    groups = ranking_groups(panel, method)
    scores = parent._ordinal_scores(groups, len(candidates))
    result: dict[str, Any] = {
        "seed": int(panel["seed"]),
        "family": str(panel["family"]),
        "panel_id": f"{panel['family']}__s{int(panel['seed'])}",
        "method": method,
        "candidate_count": len(candidates),
        "rank_groups": [
            [candidates[index]["candidate_id"] for index in group] for group in groups
        ],
        "spearman": parent._spearman(reference, scores),
        "material_pair_concordance": parent._material_concordance(reference, scores),
        "descriptive_panel_runtime_seconds": _runtime_seconds(panel, method),
    }
    for k in SHORTLIST_SIZES:
        for key, value in parent._top_k_metrics(reference, groups, k).items():
            result[f"top_{k}_{key}"] = value
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    metrics = (
        "spearman",
        "material_pair_concordance",
        "descriptive_panel_runtime_seconds",
        *[
            f"top_{k}_{metric}"
            for k in SHORTLIST_SIZES
            for metric in (
                "opportunity_regret_r2",
                "mean_utility_regret_r2",
                "exact_best_coverage",
                "within_0_01_coverage",
                "material_precision",
            )
        ],
    )
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 30:
            raise ValueError(f"method {method!r} does not contain 30 complete panels")
        summary: dict[str, Any] = {"method": method, "panel_count": len(selected)}
        for key in metrics:
            values = [row[key] for row in selected]
            summary[f"mean_{key}"] = (
                float(np.mean(values))
                if all(value is not None for value in values)
                else None
            )
        summary["by_family"] = {
            family: {
                key: (
                    float(
                        np.mean(
                            [
                                row[key]
                                for row in selected
                                if row["family"] == family
                            ]
                        )
                    )
                    if all(
                        row[key] is not None
                        for row in selected
                        if row["family"] == family
                    )
                    else None
                )
                for key in (
                    "top_1_opportunity_regret_r2",
                    "top_3_opportunity_regret_r2",
                    "top_3_mean_utility_regret_r2",
                    "top_3_material_precision",
                    "material_pair_concordance",
                    "spearman",
                )
            }
            for family in FAMILIES
        }
        output.append(summary)
    return output


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    metric: str,
    seed_offset: int,
) -> dict[str, Any]:
    lookup = {(str(row["method"]), str(row["panel_id"])): row for row in rows}
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != left:
            continue
        other = lookup[(right, str(row["panel_id"]))]
        if row[metric] is None or other[metric] is None:
            raise ValueError("paired bootstrap metric must be defined")
        groups[str(row["family"])].append(float(row[metric]) - float(other[metric]))
    if set(groups) != set(FAMILIES) or any(
        len(values) != len(NEW_SEEDS) for values in groups.values()
    ):
        raise ValueError("paired bootstrap requires five complete panels per family")
    arrays = [np.asarray(groups[family], dtype=np.float64) for family in FAMILIES]
    estimate = float(np.mean([np.mean(values) for values in arrays]))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for draw in range(BOOTSTRAP_RESAMPLES):
        draws[draw] = float(
            np.mean(
                [
                    np.mean(values[rng.integers(0, values.size, size=values.size)])
                    for values in arrays
                ]
            )
        )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "estimate": estimate,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED + seed_offset,
    }


def _family_differences(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, metric: str
) -> dict[str, float]:
    lookup = {(str(row["method"]), str(row["panel_id"])): row for row in rows}
    return {
        family: float(
            np.mean(
                [
                    float(row[metric])
                    - float(lookup[(right, str(row["panel_id"]))][metric])
                    for row in rows
                    if row["method"] == left and row["family"] == family
                ]
            )
        )
        for family in FAMILIES
    }


def _decision(
    rows: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lookup = {
        (str(row["left"]), str(row["right"]), str(row["metric"])): row
        for row in contrasts
    }
    broad_by_panel = {
        str(row["panel_id"]): row for row in rows if row["method"] == "broad_rff"
    }
    top1_differences = [
        abs(
            float(row["top_1_opportunity_regret_r2"])
            - float(broad_by_panel[str(row["panel_id"])]["top_1_opportunity_regret_r2"])
        )
        for row in rows
        if row["method"] == "bounded6_rerank"
    ]
    family_differences = _family_differences(
        rows,
        "bounded6_rerank",
        "broad_rff",
        "top_3_mean_utility_regret_r2",
    )
    broad_mean = lookup[
        ("bounded6_rerank", "broad_rff", "top_3_mean_utility_regret_r2")
    ]
    current_mean = lookup[
        ("bounded6_rerank", "save_rff_current", "top_3_mean_utility_regret_r2")
    ]
    opportunity = lookup[
        ("bounded6_rerank", "broad_rff", "top_3_opportunity_regret_r2")
    ]
    concordance = lookup[
        ("bounded6_rerank", "save_rff_current", "material_pair_concordance")
    ]
    checks = [
        {
            "criterion": "top3_mean_regret_improves_broad",
            "observed_upper_95": broad_mean["upper_95"],
            "required": "<0",
            "pass": bool(broad_mean["upper_95"] < 0.0),
        },
        {
            "criterion": "top3_mean_regret_improves_current",
            "observed_upper_95": current_mean["upper_95"],
            "required": "<0",
            "pass": bool(current_mean["upper_95"] < 0.0),
        },
        {
            "criterion": "top1_exact_panel_parity_with_broad",
            "observed_max_absolute_difference": max(top1_differences),
            "maximum": TIE_ATOL,
            "pass": bool(max(top1_differences) <= TIE_ATOL),
        },
        {
            "criterion": "top3_opportunity_noninferior_to_broad",
            "observed_upper_95": opportunity["upper_95"],
            "maximum": 0.01,
            "pass": bool(opportunity["upper_95"] <= 0.01),
        },
        {
            "criterion": "material_concordance_noninferior_to_current",
            "observed_lower_95": concordance["lower_95"],
            "minimum": -0.02,
            "pass": bool(concordance["lower_95"] >= -0.02),
        },
        {
            "criterion": "family_mean_degradation_bounded",
            "observed_by_family": family_differences,
            "observed_maximum": max(family_differences.values()),
            "maximum": FAMILY_DEGRADATION_LIMIT,
            "pass": bool(
                max(family_differences.values()) <= FAMILY_DEGRADATION_LIMIT
            ),
        },
        {
            "criterion": "all_rank_metrics_defined",
            "observed": sum(
                row["spearman"] is not None
                and row["material_pair_concordance"] is not None
                for row in rows
            ),
            "required": len(rows),
            "pass": bool(
                all(
                    row["spearman"] is not None
                    and row["material_pair_concordance"] is not None
                    for row in rows
                )
            ),
        },
    ]
    return {
        "status": "pass" if all(check["pass"] for check in checks) else "fail",
        "eligible_method": "bounded6_rerank",
        "checks": checks,
        "scope": "synthetic_shortlist_validation_only",
    }


def summarize(
    protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {
        (int(seed), str(family))
        for seed in protocol["seeds"]
        for family in protocol["families"]
    }
    observed = [(int(panel["seed"]), str(panel["family"])) for panel in panels]
    if len(observed) != 30 or len(set(observed)) != len(observed):
        raise ValueError("bounded-shortlist experiment requires 30 unique panels")
    if set(observed) != expected:
        raise ValueError("bounded-shortlist panel identities do not match protocol")
    rows = [panel_metric(panel, method) for panel in panels for method in METHODS]
    specs = (
        ("broad_rff", "top_3_mean_utility_regret_r2"),
        ("save_rff_current", "top_3_mean_utility_regret_r2"),
        ("unbounded_winner_fill", "top_3_mean_utility_regret_r2"),
        ("broad_rff", "top_1_opportunity_regret_r2"),
        ("broad_rff", "top_3_opportunity_regret_r2"),
        ("save_rff_current", "material_pair_concordance"),
        ("save_rff_current", "spearman"),
    )
    contrasts = [
        _paired_bootstrap(rows, "bounded6_rerank", right, metric, index * 1000)
        for index, (right, metric) in enumerate(specs)
    ]
    return {
        "schema_version": 1,
        "status": "post_outcome_synthetic_validation",
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
        "# Continuous bounded shortlist result",
        "",
        f"Decision: **{str(summary['decision']['status']).upper()}** for synthetic shortlist validation.",
        "",
        "Negative regret contrasts and positive concordance contrasts favor bounded reranking.",
        "",
        "| Method | Top-1 regret | Top-3 opportunity regret | Top-3 mean regret | Top-3 material precision | Material concordance | Spearman | Panel seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["method_summaries"]:
        lines.append(
            f"| {row['method']} | {row['mean_top_1_opportunity_regret_r2']:.4f} | "
            f"{row['mean_top_3_opportunity_regret_r2']:.4f} | "
            f"{row['mean_top_3_mean_utility_regret_r2']:.4f} | "
            f"{row['mean_top_3_material_precision']:.3f} | "
            f"{row['mean_material_pair_concordance']:.3f} | "
            f"{row['mean_spearman']:.3f} | "
            f"{row['mean_descriptive_panel_runtime_seconds']:.3f} |"
        )
    lines.extend(["", "## Paired contrasts", "", "| Left | Right | Metric | Estimate | 95% interval |", "|---|---|---|---:|---:|"])
    for row in summary["contrasts"]:
        lines.append(
            f"| {row['left']} | {row['right']} | {row['metric']} | "
            f"{row['estimate']:+.5f} | [{row['lower_95']:+.5f}, {row['upper_95']:+.5f}] |"
        )
    lines.extend(["", "## Frozen checks", ""])
    for check in summary["decision"]["checks"]:
        lines.append(f"- {'PASS' if check['pass'] else 'FAIL'}: `{check['criterion']}`")
    lines.extend(
        [
            "",
            "Runtime is descriptive and conservatively charges both separately evaluated components.",
            "A pass permits shared-computation and real-panel validation only.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _source_paths(motivation_root: Path) -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    result = {
        "protocol": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256_PATH,
        "motivation": MOTIVATION_PATH,
        "readme": PACKAGE_DIR / "README.md",
        "runner": RUNNER_PATH,
        "focused_test": root / "tests" / "test_continuous_bounded_shortlist.py",
        "motivation_raw_results": motivation_root / "raw_results.json",
        "motivation_summary": motivation_root / "summary.json",
        "motivation_report": motivation_root / "report.md",
        "parent_runner": parent.RUNNER_PATH,
        "kernel_runner": kernel.RUNNER_PATH,
        "null_parent_runner": null_parent.RUNNER_PATH,
        "capacity_runner": capacity.RUNNER_PATH,
        "multiview_runner": multiview.RUNNER_PATH,
        "save_runner": save.RUNNER_PATH,
        "residual_runner": residual.RUNNER_PATH,
        "common_runner": common.RUNNER_PATH,
        "project_configuration": root / "pyproject.toml",
    }
    result.update(
        {
            f"overlapindex_{path.name}": path
            for path in sorted((root / "overlapindex").glob("*.py"))
        }
    )
    return result


def run(output: Path, motivation_root: Path = DEFAULT_MOTIVATION_ROOT) -> dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    motivation_root = motivation_root.resolve()
    _validate_ancestry(protocol, motivation_root)
    implementation_commit = os.environ.get("CONTINUOUS_OI_COMMIT")
    if implementation_commit is None or len(implementation_commit) != 40:
        raise ValueError("CONTINUOUS_OI_COMMIT must contain the frozen 40-hex commit")
    try:
        int(implementation_commit, 16)
    except ValueError as error:
        raise ValueError(
            "CONTINUOUS_OI_COMMIT must contain the frozen 40-hex commit"
        ) from error
    source_hashes = {
        key: common._sha256(path) for key, path in _source_paths(motivation_root).items()
    }
    output = output.resolve()
    for terminal in ("raw_results.json", "summary.json", "report.md", "manifest.json"):
        if (output / terminal).exists():
            raise FileExistsError(f"refusing to overwrite {output / terminal}")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    total = len(protocol["seeds"]) * len(protocol["families"])
    completed = 0
    for seed in protocol["seeds"]:
        for family in protocol["families"]:
            path = checkpoints / f"seed_{seed}__{family}.json"
            if path.exists():
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                if checkpoint.get("source_hashes") != source_hashes:
                    raise ValueError(f"stale source checkpoint: {path}")
                panel = checkpoint["panel"]
            else:
                started = time.perf_counter()
                panel = kernel.run_panel(protocol, int(seed), str(family))
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
            print(f"completed {completed}/{total}: seed={seed} family={family}", flush=True)
    raw = {
        "schema_version": 1,
        "artifact_status": "completed",
        "source_hashes": source_hashes,
        "implementation_commit": implementation_commit,
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
        "source_hashes": source_hashes,
        "files": {
            name: {
                "sha256": common._sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in ("raw_results.json", "summary.json", "report.md")
        },
        "checkpoint_count": len(panels),
    }
    common._atomic_json(output / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motivation-root", type=Path, default=DEFAULT_MOTIVATION_ROOT)
    args = parser.parse_args()
    summary = run(args.output, args.motivation_root)
    print(json.dumps({"decision": summary["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
