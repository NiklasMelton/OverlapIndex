"""Frozen new-seed continuous winner-plus-fill shortlist experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import itertools
import json
import os
import platform
from pathlib import Path
import time
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata

from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_kernel_alignment import run as kernel
from experiments.continuous_multiview_ranking import run as multiview
from experiments.continuous_null_diagnostic import run as null_parent
from experiments.continuous_prototype_capacity import run as capacity
from experiments.continuous_residual_ranking import run as residual
from experiments.continuous_save_ranking import run as save


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
MOTIVATION_PATH = PACKAGE_DIR / "motivation.json"
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_MOTIVATION_RAW = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "continuous_kernel_alignment"
    / "raw_results.json"
)
DEFAULT_MOTIVATION_SUMMARY = DEFAULT_MOTIVATION_RAW.with_name("summary.json")
METHODS = (
    "broad_rff",
    "mixed_rff",
    "save_rff_current",
    "winner_fill",
    "linear_probe",
)
SHORTLIST_SIZES = (1, 2, 3, 5)
NEW_SEEDS = (1907, 2011, 2113, 2207, 2311)
TIE_ATOL = 1.0e-12
EQUIVALENCE_BAND = 0.01
BOOTSTRAP_SEED = 2026090101
BOOTSTRAP_RESAMPLES = 10_000


def _load_protocol() -> dict[str, Any]:
    expected_hash = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected_hash) != 64 or common._sha256(PROTOCOL_PATH) != expected_hash:
        raise ValueError("shortlist protocol SHA-256 sidecar does not match")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_exploratory_before_outcomes":
        raise ValueError("shortlist protocol is not frozen")
    return protocol


def _load_motivation() -> dict[str, Any]:
    motivation = json.loads(MOTIVATION_PATH.read_text(encoding="utf-8"))
    if motivation.get("status") != "post_outcome_motivation_only":
        raise ValueError("shortlist motivation status is invalid")
    return motivation


def _validate_protocol_against_code(protocol: Mapping[str, Any]) -> None:
    if tuple(protocol["seeds"]) != NEW_SEEDS:
        raise ValueError("frozen shortlist seed set does not match code")
    if tuple(protocol["families"]) != (
        "linear",
        "quadratic_product",
        "radial_quadratic",
        "absolute_value",
        "sinusoid",
        "triangle_wave",
    ):
        raise ValueError("frozen shortlist family set does not match code")
    if set(protocol["ranking_methods"]) != set(METHODS):
        raise ValueError("frozen shortlist method set does not match code")
    if tuple(protocol["shortlist_sizes"]) != SHORTLIST_SIZES:
        raise ValueError("frozen shortlist sizes do not match code")
    if float(protocol["ranking_ties"]["absolute_tolerance"]) != TIE_ATOL:
        raise ValueError("frozen shortlist tie tolerance does not match code")
    if float(protocol["material_equivalence_band_r2"]) != EQUIVALENCE_BAND:
        raise ValueError("frozen material equivalence band does not match code")
    if int(protocol["bootstrap"]["seed"]) != BOOTSTRAP_SEED:
        raise ValueError("frozen bootstrap seed does not match code")
    if int(protocol["bootstrap"]["resamples"]) != BOOTSTRAP_RESAMPLES:
        raise ValueError("frozen bootstrap count does not match code")
    if tuple(protocol["kernel_alignment"]["bandwidth_multipliers"]) != (
        kernel.BANDWIDTH_MULTIPLIERS
    ):
        raise ValueError("frozen kernel bandwidths do not match parent implementation")
    if tuple(protocol["reference_heads"]["ids"]) != tuple(
        residual._head_estimators()
    ):
        raise ValueError("frozen reference head order does not match code")
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
    protocol: Mapping[str, Any],
    raw_path: Path = DEFAULT_MOTIVATION_RAW,
    summary_path: Path = DEFAULT_MOTIVATION_SUMMARY,
) -> None:
    ancestry = protocol["ancestry"]
    if common._sha256(raw_path.resolve()) != ancestry["motivation_raw_results_sha256"]:
        raise ValueError("motivation raw-results SHA-256 does not match protocol")
    if common._sha256(summary_path.resolve()) != ancestry["motivation_summary_sha256"]:
        raise ValueError("motivation summary SHA-256 does not match protocol")
    if common._sha256(kernel.PROTOCOL_PATH) != ancestry["motivation_protocol_sha256"]:
        raise ValueError("motivation protocol bytes do not match ancestry")
    if common._sha256(kernel.RUNNER_PATH) != ancestry["motivation_runner_sha256"]:
        raise ValueError("motivation runner bytes do not match ancestry")
    motivation = _load_motivation()
    for key in (
        "rank_utility_audit_summary_sha256",
        "rank_utility_audit_report_sha256",
    ):
        if motivation.get(key) != ancestry.get(key):
            raise ValueError(f"rank-utility motivation mismatch for {key}")


def _score_groups(
    values: Sequence[float], indices: Iterable[int] | None = None
) -> list[tuple[int, ...]]:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("ranking scores must be a finite one-dimensional vector")
    unassigned = set(range(scores.size) if indices is None else indices)
    if not unassigned or any(not 0 <= index < scores.size for index in unassigned):
        raise ValueError("ranking indices must be a nonempty aligned subset")
    groups: list[tuple[int, ...]] = []
    while unassigned:
        maximum = max(float(scores[index]) for index in unassigned)
        group = tuple(
            sorted(
                index
                for index in unassigned
                if abs(float(scores[index]) - maximum) <= TIE_ATOL
            )
        )
        unassigned.difference_update(group)
        groups.append(group)
    return groups


def _component_values(panel: Mapping[str, Any], method: str) -> np.ndarray:
    candidates = panel["candidates"]
    if method == "broad_rff":
        values = [
            candidate["selectors"]["rff64_mean8_k4"]["kernel_alignment"][
                "feature_bandwidths"
            ]["4.0"]["score"]
            for candidate in candidates
        ]
    elif method == "mixed_rff":
        values = [
            candidate["selectors"]["rff64_mean8_k4"]["kernel_alignment"]["score"]
            for candidate in candidates
        ]
    elif method == "save_rff_current":
        values = [
            candidate["selectors"]["save_rffmean_k4"]["current_common32"]["score"]
            for candidate in candidates
        ]
    elif method == "linear_probe":
        values = [
            candidate["heads"]["linear_ridge"]["test_r2"]
            for candidate in candidates
        ]
    else:
        raise ValueError(f"unknown component method {method!r}")
    if any(value is None for value in values):
        raise ValueError(f"component method {method!r} contains undefined scores")
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"component method {method!r} contains nonfinite scores")
    return result


def ranking_groups(panel: Mapping[str, Any], method: str) -> list[tuple[int, ...]]:
    if method != "winner_fill":
        return _score_groups(_component_values(panel, method))
    broad = _component_values(panel, "broad_rff")
    current = _component_values(panel, "save_rff_current")
    first = _score_groups(broad)[0]
    remaining = set(range(broad.size)).difference(first)
    groups = [first]
    if remaining:
        groups.extend(_score_groups(current, remaining))
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(broad.size)):
        raise AssertionError("winner-fill ranking is not a complete partition")
    return groups


def _ordinal_scores(groups: Sequence[Sequence[int]], count: int) -> np.ndarray:
    scores = np.empty(count, dtype=np.float64)
    for group_position, group in enumerate(groups):
        scores[np.asarray(group, dtype=np.int64)] = float(len(groups) - group_position)
    return scores


def _top_k_subsets(
    groups: Sequence[Sequence[int]], k: int
) -> list[tuple[int, ...]]:
    fixed: list[int] = []
    for group in groups:
        remaining = k - len(fixed)
        if len(group) <= remaining:
            fixed.extend(group)
            if len(fixed) == k:
                return [tuple(sorted(fixed))]
            continue
        return [
            tuple(sorted([*fixed, *choice]))
            for choice in itertools.combinations(group, remaining)
        ]
    raise AssertionError("top-k groups did not contain enough candidates")


def _top_k_metrics(
    reference: np.ndarray, groups: Sequence[Sequence[int]], k: int
) -> dict[str, float]:
    subsets = _top_k_subsets(groups, k)
    oracle_best = float(np.max(reference))
    oracle_mean = float(np.mean(np.sort(reference)[-k:]))
    reference_kth = float(np.sort(reference)[-k])
    best_values = []
    mean_values = []
    exact = []
    near = []
    precision = []
    for subset in subsets:
        values = reference[np.asarray(subset, dtype=np.int64)]
        best = float(np.max(values))
        best_values.append(best)
        mean_values.append(float(np.mean(values)))
        exact.append(float(abs(best - oracle_best) <= TIE_ATOL))
        near.append(float(best >= oracle_best - EQUIVALENCE_BAND))
        precision.append(
            float(np.mean(values >= reference_kth - EQUIVALENCE_BAND))
        )
    return {
        "opportunity_regret_r2": oracle_best - float(np.mean(best_values)),
        "mean_utility_regret_r2": oracle_mean - float(np.mean(mean_values)),
        "exact_best_coverage": float(np.mean(exact)),
        "within_0_01_coverage": float(np.mean(near)),
        "material_precision": float(np.mean(precision)),
        "tie_averaged_subset_count": len(subsets),
    }


def _spearman(reference: np.ndarray, ranking_scores: np.ndarray) -> float | None:
    if np.all(reference == reference[0]) or np.all(ranking_scores == ranking_scores[0]):
        return None
    value = float(
        np.corrcoef(
            rankdata(reference, method="average"),
            rankdata(ranking_scores, method="average"),
        )[0, 1]
    )
    return value if np.isfinite(value) else None


def _material_concordance(reference: np.ndarray, ranking_scores: np.ndarray) -> float | None:
    terms = []
    for left in range(reference.size):
        for right in range(left + 1, reference.size):
            delta = float(reference[left] - reference[right])
            if abs(delta) <= EQUIVALENCE_BAND:
                continue
            score_delta = float(ranking_scores[left] - ranking_scores[right])
            terms.append(0.5 if score_delta == 0.0 else float(delta * score_delta > 0.0))
    return float(np.mean(terms)) if terms else None


def _runtime_seconds(panel: Mapping[str, Any], method: str) -> float:
    candidates = panel["candidates"]
    rff = sum(
        float(candidate["selectors"]["rff64_mean8_k4"]["wall_seconds"])
        for candidate in candidates
    )
    save_rff = sum(
        float(candidate["selectors"]["save_rffmean_k4"]["wall_seconds"])
        for candidate in candidates
    )
    if method in {"broad_rff", "mixed_rff"}:
        return float(rff)
    if method == "save_rff_current":
        return float(save_rff)
    if method == "winner_fill":
        return float(rff + save_rff)
    if method == "linear_probe":
        return float(
            sum(
                candidate["heads"]["linear_ridge"]["wall_seconds"]
                for candidate in candidates
            )
        )
    raise ValueError(f"unknown runtime method {method!r}")


def panel_metric(panel: Mapping[str, Any], method: str) -> dict[str, Any]:
    candidates = panel["candidates"]
    reference = np.asarray(
        [candidate["best_head_envelope_r2"] for candidate in candidates],
        dtype=np.float64,
    )
    if reference.shape != (16,) or not np.all(np.isfinite(reference)):
        raise ValueError("panel reference envelope must contain 16 finite candidates")
    groups = ranking_groups(panel, method)
    scores = _ordinal_scores(groups, len(candidates))
    result: dict[str, Any] = {
        "seed": int(panel["seed"]),
        "family": str(panel["family"]),
        "panel_id": f"{panel['family']}__s{int(panel['seed'])}",
        "method": method,
        "candidate_count": len(candidates),
        "rank_groups": [
            [candidates[index]["candidate_id"] for index in group] for group in groups
        ],
        "spearman": _spearman(reference, scores),
        "material_pair_concordance": _material_concordance(reference, scores),
        "descriptive_panel_runtime_seconds": _runtime_seconds(panel, method),
    }
    for k in SHORTLIST_SIZES:
        for key, value in _top_k_metrics(reference, groups, k).items():
            result[f"top_{k}_{key}"] = value
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 30:
            raise ValueError(f"method {method!r} does not contain 30 complete panels")
        summary: dict[str, Any] = {"method": method, "panel_count": len(selected)}
        for key in (
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
        ):
            values = [row[key] for row in selected]
            summary[f"mean_{key}"] = (
                float(np.mean(values)) if all(value is not None for value in values) else None
            )
        summary["by_family"] = {
            family: {
                key: (
                    float(np.mean([row[key] for row in selected if row["family"] == family]))
                    if all(row[key] is not None for row in selected if row["family"] == family)
                    else None
                )
                for key in (
                    "top_1_opportunity_regret_r2",
                    "top_3_mean_utility_regret_r2",
                    "top_3_material_precision",
                    "material_pair_concordance",
                    "spearman",
                )
            }
            for family in sorted({row["family"] for row in selected})
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
    lookup = {
        (str(row["method"]), str(row["panel_id"])): row for row in rows
    }
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != left:
            continue
        other = lookup[(right, str(row["panel_id"]))]
        if row[metric] is None or other[metric] is None:
            raise ValueError("paired bootstrap metric must be defined")
        groups[str(row["family"])].append(float(row[metric]) - float(other[metric]))
    if set(groups) != {
        "linear",
        "quadratic_product",
        "radial_quadratic",
        "absolute_value",
        "sinusoid",
        "triangle_wave",
    } or any(len(values) != len(NEW_SEEDS) for values in groups.values()):
        raise ValueError("paired bootstrap requires five complete panels per family")
    estimate = float(np.mean([np.mean(values) for values in groups.values()]))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    arrays = [np.asarray(groups[family], dtype=np.float64) for family in sorted(groups)]
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


def _decision(
    rows: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lookup = {
        (str(row["left"]), str(row["right"]), str(row["metric"])): row
        for row in contrasts
    }
    top1_differences = [
        abs(
            float(row["top_1_opportunity_regret_r2"])
            - float(
                next(
                    other["top_1_opportunity_regret_r2"]
                    for other in rows
                    if other["method"] == "broad_rff"
                    and other["panel_id"] == row["panel_id"]
                )
            )
        )
        for row in rows
        if row["method"] == "winner_fill"
    ]
    checks = [
        {
            "criterion": "top3_mean_regret_improves_broad",
            "observed_upper_95": lookup[("winner_fill", "broad_rff", "top_3_mean_utility_regret_r2")]["upper_95"],
            "required": "<0",
            "pass": bool(lookup[("winner_fill", "broad_rff", "top_3_mean_utility_regret_r2")]["upper_95"] < 0.0),
        },
        {
            "criterion": "top3_mean_regret_improves_current",
            "observed_upper_95": lookup[("winner_fill", "save_rff_current", "top_3_mean_utility_regret_r2")]["upper_95"],
            "required": "<0",
            "pass": bool(lookup[("winner_fill", "save_rff_current", "top_3_mean_utility_regret_r2")]["upper_95"] < 0.0),
        },
        {
            "criterion": "top1_exact_panel_parity_with_broad",
            "observed_max_absolute_difference": max(top1_differences),
            "maximum": TIE_ATOL,
            "pass": bool(max(top1_differences) <= TIE_ATOL),
        },
        {
            "criterion": "top3_opportunity_noninferior_to_broad",
            "observed_upper_95": lookup[("winner_fill", "broad_rff", "top_3_opportunity_regret_r2")]["upper_95"],
            "maximum": 0.01,
            "pass": bool(lookup[("winner_fill", "broad_rff", "top_3_opportunity_regret_r2")]["upper_95"] <= 0.01),
        },
        {
            "criterion": "material_concordance_noninferior_to_current",
            "observed_lower_95": lookup[("winner_fill", "save_rff_current", "material_pair_concordance")]["lower_95"],
            "minimum": -0.02,
            "pass": bool(lookup[("winner_fill", "save_rff_current", "material_pair_concordance")]["lower_95"] >= -0.02),
        },
        {
            "criterion": "all_rank_metrics_defined",
            "observed": sum(
                row["spearman"] is not None and row["material_pair_concordance"] is not None
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
        "eligible_method": "winner_fill",
        "checks": checks,
        "scope": "synthetic_shortlist_validation_only",
    }


def summarize(
    protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(panels) != 30:
        raise ValueError("shortlist experiment requires exactly 30 panels")
    expected_panels = {
        (int(seed), str(family))
        for seed in protocol["seeds"]
        for family in protocol["families"]
    }
    observed_panels = [
        (int(panel["seed"]), str(panel["family"])) for panel in panels
    ]
    if len(set(observed_panels)) != len(observed_panels):
        raise ValueError("shortlist experiment contains duplicate panels")
    if set(observed_panels) != expected_panels:
        raise ValueError("shortlist experiment panel identities do not match protocol")
    rows = [panel_metric(panel, method) for panel in panels for method in METHODS]
    contrast_specs = (
        ("broad_rff", "top_3_mean_utility_regret_r2"),
        ("save_rff_current", "top_3_mean_utility_regret_r2"),
        ("broad_rff", "top_1_opportunity_regret_r2"),
        ("broad_rff", "top_3_opportunity_regret_r2"),
        ("save_rff_current", "material_pair_concordance"),
        ("save_rff_current", "spearman"),
    )
    contrasts = [
        _paired_bootstrap(rows, "winner_fill", comparator, metric, position * 1000)
        for position, (comparator, metric) in enumerate(contrast_specs)
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
        "# Continuous winner-plus-fill shortlist result",
        "",
        f"Decision: **{str(summary['decision']['status']).upper()}** for synthetic shortlist validation.",
        "",
        "Negative regret contrasts favor winner-fill; positive concordance contrasts favor winner-fill.",
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
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "| Left | Right | Metric | Estimate | 95% interval |",
            "|---|---|---|---:|---:|",
        ]
    )
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
            "Runtime is descriptive and conservatively charges winner-fill both separately evaluated components.",
            "A pass permits a shared-computation implementation experiment and real-panel validation only.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def run(
    output: Path,
    motivation_raw_path: Path = DEFAULT_MOTIVATION_RAW,
    motivation_summary_path: Path = DEFAULT_MOTIVATION_SUMMARY,
) -> dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol_against_code(protocol)
    _validate_ancestry(protocol, motivation_raw_path, motivation_summary_path)
    source_paths = {
        "protocol": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256_PATH,
        "motivation": MOTIVATION_PATH,
        "readme": PACKAGE_DIR / "README.md",
        "runner": RUNNER_PATH,
        "focused_test": (
            Path(__file__).resolve().parents[2]
            / "tests"
            / "test_continuous_winner_fill_shortlist.py"
        ),
        "motivation_raw_results": motivation_raw_path.resolve(),
        "motivation_summary": motivation_summary_path.resolve(),
        "kernel_runner": kernel.RUNNER_PATH,
        "null_parent_runner": null_parent.RUNNER_PATH,
        "capacity_runner": capacity.RUNNER_PATH,
        "multiview_runner": multiview.RUNNER_PATH,
        "save_runner": save.RUNNER_PATH,
        "residual_runner": residual.RUNNER_PATH,
        "common_runner": common.RUNNER_PATH,
        "continuous_overlap_index": (
            Path(__file__).resolve().parents[2]
            / "overlapindex"
            / "ContinuousOverlapIndex.py"
        ),
        "project_configuration": Path(__file__).resolve().parents[2] / "pyproject.toml",
    }
    source_paths.update(
        {
            f"overlapindex_{path.name}": path
            for path in sorted(
                (Path(__file__).resolve().parents[2] / "overlapindex").glob("*.py")
            )
            if path != source_paths["continuous_overlap_index"]
        }
    )
    source_hashes = {key: common._sha256(path) for key, path in source_paths.items()}
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
        "implementation_commit": os.environ.get("CONTINUOUS_OI_COMMIT"),
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
    parser.add_argument("--motivation-raw", type=Path, default=DEFAULT_MOTIVATION_RAW)
    parser.add_argument("--motivation-summary", type=Path, default=DEFAULT_MOTIVATION_SUMMARY)
    args = parser.parse_args()
    summary = run(args.output, args.motivation_raw, args.motivation_summary)
    print(json.dumps({"decision": summary["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
