"""Hash-bound cross-cohort continuous shortlist-depth audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.continuous_bounded_shortlist import run as bounded_parent
from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_winner_fill_shortlist import run as winner_parent


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
RUNNER_PATH = Path(__file__).resolve()
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = {
    "winner_fill": ROOT / "artifacts" / "continuous_winner_fill_shortlist",
    "bounded": ROOT / "artifacts" / "continuous_bounded_shortlist",
}
COHORTS = ("winner_fill", "bounded")
FAMILIES = bounded_parent.FAMILIES
DEPTHS = tuple(range(1, 9))
METHODS = (
    "broad_rff",
    "save_rff_current",
    "unbounded_winner_fill",
    "bounded_2k",
    "winner_preserved_pool_oracle",
)
TIE_ATOL = 1.0e-12
BOOTSTRAP_SEED = 2026090301
BOOTSTRAP_RESAMPLES = 10_000


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or common._sha256(PROTOCOL_PATH) != expected:
        raise ValueError("shortlist-depth protocol SHA-256 sidecar does not match")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_post_outcome_audit_before_additional_outcomes":
        raise ValueError("shortlist-depth audit protocol status is invalid")
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if tuple(protocol["families"]) != FAMILIES:
        raise ValueError("shortlist-depth family set does not match code")
    if tuple(protocol["shortlist_depths"]) != DEPTHS:
        raise ValueError("shortlist-depth grid does not match code")
    if set(protocol["methods"]) != set(METHODS):
        raise ValueError("shortlist-depth method set does not match code")
    if float(protocol["ties"]["absolute_tolerance"]) != TIE_ATOL:
        raise ValueError("shortlist-depth tie tolerance does not match code")
    if int(protocol["bootstrap"]["seed"]) != BOOTSTRAP_SEED:
        raise ValueError("shortlist-depth bootstrap seed does not match code")
    if int(protocol["bootstrap"]["resamples"]) != BOOTSTRAP_RESAMPLES:
        raise ValueError("shortlist-depth bootstrap count does not match code")


def _validate_bundle(
    cohort: str, root: Path, descriptor: Mapping[str, Any]
) -> dict[str, Any]:
    root = root.resolve()
    paths = {
        "raw_results_sha256": root / "raw_results.json",
        "summary_sha256": root / "summary.json",
        "manifest_sha256": root / "manifest.json",
    }
    for key, path in paths.items():
        if common._sha256(path) != descriptor[key]:
            raise ValueError(f"{cohort} input hash mismatch for {key}")
    raw = json.loads(paths["raw_results_sha256"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_sha256"].read_text(encoding="utf-8"))
    if raw.get("artifact_status") != "completed" or manifest.get(
        "artifact_status"
    ) != "completed":
        raise ValueError(f"{cohort} input artifact is not completed")
    if raw.get("implementation_commit") != descriptor["implementation_commit"]:
        raise ValueError(f"{cohort} implementation commit mismatch")
    if manifest.get("source_hashes") != raw.get("source_hashes"):
        raise ValueError(f"{cohort} raw/manifest source identity mismatch")
    for name, file_descriptor in manifest["files"].items():
        path = root / name
        if common._sha256(path) != file_descriptor["sha256"]:
            raise ValueError(f"{cohort} terminal file hash mismatch for {name}")
        if path.stat().st_size != int(file_descriptor["bytes"]):
            raise ValueError(f"{cohort} terminal file size mismatch for {name}")
    if len(raw.get("panels", [])) != 30:
        raise ValueError(f"{cohort} must contain exactly 30 panels")
    expected = {
        (int(seed), family) for seed in descriptor["seeds"] for family in FAMILIES
    }
    observed = [
        (int(panel["seed"]), str(panel["family"])) for panel in raw["panels"]
    ]
    if len(set(observed)) != 30 or set(observed) != expected:
        raise ValueError(f"{cohort} panel grid does not match protocol")
    return raw


def load_inputs(
    protocol: Mapping[str, Any], roots: Mapping[str, Path] = DEFAULT_INPUTS
) -> list[dict[str, Any]]:
    if set(roots) != set(COHORTS):
        raise ValueError("shortlist-depth roots must contain exactly two cohorts")
    panels = []
    for cohort in COHORTS:
        raw = _validate_bundle(cohort, roots[cohort], protocol["inputs"][cohort])
        panels.extend({"cohort": cohort, "panel": panel} for panel in raw["panels"])
    if len(panels) != 60:
        raise AssertionError("shortlist-depth audit did not load exactly 60 panels")
    return panels


def _membership_probabilities(
    groups: Sequence[Sequence[int]], k: int, count: int
) -> np.ndarray:
    subsets = winner_parent._top_k_subsets(groups, k)
    result = np.zeros(count, dtype=np.float64)
    for subset in subsets:
        result[np.asarray(subset, dtype=np.int64)] += 1.0 / len(subsets)
    if not np.isclose(np.sum(result), k, atol=1.0e-12, rtol=0.0):
        raise AssertionError("top-k membership probabilities do not sum to k")
    return result


def _inclusion_agreement(left: np.ndarray, right: np.ndarray, k: int) -> float:
    value = 1.0 - float(np.sum(np.abs(left - right))) / (2.0 * float(k))
    return float(np.clip(value, 0.0, 1.0))


def _broad_pool(
    broad_groups: Sequence[Sequence[int]], k: int, count: int
) -> set[int]:
    target = min(2 * int(k), int(count))
    pool: set[int] = set()
    for group in broad_groups:
        pool.update(int(index) for index in group)
        if len(pool) >= target:
            break
    if len(pool) < target:
        raise AssertionError("broad top-2k retrieval pool is incomplete")
    return pool


def ranking_groups(
    panel: Mapping[str, Any], method: str, k: int
) -> list[tuple[int, ...]]:
    if method in {"broad_rff", "save_rff_current"}:
        return winner_parent.ranking_groups(panel, method)
    if method == "unbounded_winner_fill":
        return winner_parent.ranking_groups(panel, "winner_fill")
    if method not in {"bounded_2k", "winner_preserved_pool_oracle"}:
        raise ValueError(f"unknown shortlist-depth method {method!r}")
    broad = winner_parent._component_values(panel, "broad_rff")
    current = winner_parent._component_values(panel, "save_rff_current")
    reference = np.asarray(
        [candidate["best_head_envelope_r2"] for candidate in panel["candidates"]],
        dtype=np.float64,
    )
    broad_groups = winner_parent._score_groups(broad)
    winner = broad_groups[0]
    pool = _broad_pool(broad_groups, k, len(broad))
    pool.difference_update(winner)
    outside = set(range(len(broad))).difference(winner).difference(pool)
    ranking_values = current if method == "bounded_2k" else reference
    groups = [winner]
    if pool:
        groups.extend(winner_parent._score_groups(ranking_values, pool))
    if outside:
        groups.extend(winner_parent._score_groups(ranking_values, outside))
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(len(broad))):
        raise AssertionError("shortlist-depth ranking is not a complete partition")
    return groups


def _normalized_boundary_margin(values: np.ndarray, k: int) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    scale = float(ordered[0] - ordered[-1])
    if k >= len(ordered) or scale <= np.finfo(np.float64).eps:
        return 0.0
    return float(max(ordered[k - 1] - ordered[k], 0.0) / scale)


def _broad_fold_stability(
    panel: Mapping[str, Any], broad_membership: np.ndarray, k: int
) -> float:
    candidates = panel["candidates"]
    fold_count = len(candidates[0]["selectors"]["rff64_mean8_k4"]["folds"])
    values = []
    for fold in range(fold_count):
        scores = np.asarray(
            [
                candidate["selectors"]["rff64_mean8_k4"]["folds"][fold][
                    "kernel_alignment"
                ]["feature_bandwidths"]["4.0"]["score"]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        membership = _membership_probabilities(
            winner_parent._score_groups(scores), k, len(candidates)
        )
        values.append(_inclusion_agreement(broad_membership, membership, k))
    return float(np.mean(values))


def _broad_bandwidth_agreement(
    panel: Mapping[str, Any], broad_membership: np.ndarray, k: int
) -> float:
    candidates = panel["candidates"]
    score_sets = [
        np.asarray(
            [
                candidate["selectors"]["rff64_mean8_k4"]["kernel_alignment"][
                    "feature_bandwidths"
                ][multiplier]["score"]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        for multiplier in ("0.25", "1.0")
    ]
    score_sets.append(
        np.asarray(
            [
                candidate["selectors"]["rff64_mean8_k4"]["kernel_alignment"][
                    "score"
                ]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
    )
    agreements = []
    for scores in score_sets:
        membership = _membership_probabilities(
            winner_parent._score_groups(scores), k, len(candidates)
        )
        agreements.append(_inclusion_agreement(broad_membership, membership, k))
    return float(np.mean(agreements))


def panel_depth_rows(
    cohort: str, panel: Mapping[str, Any], k: int
) -> list[dict[str, Any]]:
    candidates = panel["candidates"]
    count = len(candidates)
    if count != 16:
        raise ValueError("shortlist-depth panel must contain 16 candidates")
    reference = np.asarray(
        [candidate["best_head_envelope_r2"] for candidate in candidates],
        dtype=np.float64,
    )
    broad_values = winner_parent._component_values(panel, "broad_rff")
    broad_groups = ranking_groups(panel, "broad_rff", k)
    broad_membership = _membership_probabilities(broad_groups, k, count)
    current_membership = _membership_probabilities(
        ranking_groups(panel, "save_rff_current", k), k, count
    )
    oracle_metrics = winner_parent._top_k_metrics(
        reference, ranking_groups(panel, "winner_preserved_pool_oracle", k), k
    )
    broad_metrics = winner_parent._top_k_metrics(reference, broad_groups, k)
    oracle_headroom = float(
        broad_metrics["mean_utility_regret_r2"]
        - oracle_metrics["mean_utility_regret_r2"]
    )
    diagnostics = {
        "normalized_broad_boundary_margin": _normalized_boundary_margin(
            broad_values, k
        ),
        "broad_fold_inclusion_stability": _broad_fold_stability(
            panel, broad_membership, k
        ),
        "broad_bandwidth_inclusion_agreement": _broad_bandwidth_agreement(
            panel, broad_membership, k
        ),
        "broad_save_inclusion_agreement": _inclusion_agreement(
            broad_membership, current_membership, k
        ),
        "oracle_pool_headroom": oracle_headroom,
    }
    rows = []
    for method in METHODS:
        groups = ranking_groups(panel, method, k)
        metrics = winner_parent._top_k_metrics(reference, groups, k)
        membership = _membership_probabilities(groups, k, count)
        realized_gain = float(
            broad_metrics["mean_utility_regret_r2"]
            - metrics["mean_utility_regret_r2"]
        )
        capture = (
            float(realized_gain / oracle_headroom)
            if oracle_headroom > TIE_ATOL
            else None
        )
        rows.append(
            {
                "cohort": cohort,
                "seed": int(panel["seed"]),
                "family": str(panel["family"]),
                "panel_id": f"{cohort}__{panel['family']}__s{int(panel['seed'])}",
                "k": int(k),
                "method": method,
                "opportunity_regret_r2": float(metrics["opportunity_regret_r2"]),
                "mean_utility_regret_r2": float(metrics["mean_utility_regret_r2"]),
                "material_precision": float(metrics["material_precision"]),
                "exact_best_coverage": float(metrics["exact_best_coverage"]),
                "within_0_01_coverage": float(metrics["within_0_01_coverage"]),
                "expected_membership_swaps": float(
                    np.sum(np.abs(membership - broad_membership)) / 2.0
                ),
                "realized_mean_utility_gain": realized_gain,
                "oracle_headroom_capture_fraction": capture,
                "swap_effect": (
                    "improved"
                    if realized_gain > TIE_ATOL
                    else "harmed"
                    if realized_gain < -TIE_ATOL
                    else "unchanged"
                ),
                **diagnostics,
            }
        )
    return rows


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    k: int,
    metric: str,
    seed_offset: int,
) -> dict[str, Any]:
    lookup = {
        (str(row["method"]), str(row["panel_id"]), int(row["k"])): row
        for row in rows
    }
    strata: dict[tuple[str, str], list[float]] = defaultdict(list)
    cohort_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != left or int(row["k"]) != int(k):
            continue
        other = lookup[(right, str(row["panel_id"]), int(k))]
        delta = float(row[metric]) - float(other[metric])
        strata[(str(row["cohort"]), str(row["family"]))].append(delta)
        cohort_values[str(row["cohort"])].append(delta)
    expected_strata = {(cohort, family) for cohort in COHORTS for family in FAMILIES}
    if set(strata) != expected_strata or any(len(values) != 5 for values in strata.values()):
        raise ValueError("depth bootstrap requires five panels per cohort-family stratum")
    arrays = [np.asarray(strata[key], dtype=np.float64) for key in sorted(strata)]
    estimate = float(np.mean([np.mean(values) for values in arrays]))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    stratum_draws = []
    for values in arrays:
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
        stratum_draws.append(np.mean(values[indices], axis=1))
    draws = np.mean(np.stack(stratum_draws, axis=1), axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "left": left,
        "right": right,
        "k": int(k),
        "metric": metric,
        "estimate": estimate,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "by_cohort": {
            cohort: float(np.mean(cohort_values[cohort])) for cohort in COHORTS
        },
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED + seed_offset,
    }


def _curve(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method in METHODS:
        for k in DEPTHS:
            selected = [
                row for row in rows if row["method"] == method and row["k"] == k
            ]
            if len(selected) != 60:
                raise ValueError("depth curve requires 60 rows per method-depth")
            result.append(
                {
                    "method": method,
                    "k": k,
                    "panel_count": len(selected),
                    **{
                        f"mean_{metric}": float(
                            np.mean([float(row[metric]) for row in selected])
                        )
                        for metric in (
                            "opportunity_regret_r2",
                            "mean_utility_regret_r2",
                            "material_precision",
                            "expected_membership_swaps",
                            "realized_mean_utility_gain",
                            "oracle_pool_headroom",
                        )
                    },
                    "improved_panel_rate": float(
                        np.mean([row["swap_effect"] == "improved" for row in selected])
                    ),
                    "harmed_panel_rate": float(
                        np.mean([row["swap_effect"] == "harmed" for row in selected])
                    ),
                    "by_cohort_mean_utility_regret_r2": {
                        cohort: float(
                            np.mean(
                                [
                                    row["mean_utility_regret_r2"]
                                    for row in selected
                                    if row["cohort"] == cohort
                                ]
                            )
                        )
                        for cohort in COHORTS
                    },
                }
            )
    return result


def _diagnostic_correlations(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    features = (
        "normalized_broad_boundary_margin",
        "broad_fold_inclusion_stability",
        "broad_bandwidth_inclusion_agreement",
        "broad_save_inclusion_agreement",
        "oracle_pool_headroom",
        "expected_membership_swaps",
    )
    result = []
    for method in ("unbounded_winner_fill", "bounded_2k"):
        for k in DEPTHS:
            selected = [
                row for row in rows if row["method"] == method and row["k"] == k
            ]
            gains = np.asarray(
                [row["realized_mean_utility_gain"] for row in selected],
                dtype=np.float64,
            )
            result.append(
                {
                    "method": method,
                    "k": k,
                    "panel_count": len(selected),
                    "spearman_gain_by_feature": {
                        feature: winner_parent._spearman(
                            gains,
                            np.asarray([row[feature] for row in selected], dtype=np.float64),
                        )
                        for feature in features
                    },
                }
            )
    return result


def _decision(
    rows: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lookup = {
        (str(row["left"]), str(row["right"]), int(row["k"]), str(row["metric"])): row
        for row in contrasts
    }
    mean_regret = lookup[
        ("unbounded_winner_fill", "broad_rff", 5, "mean_utility_regret_r2")
    ]
    opportunity = lookup[
        ("unbounded_winner_fill", "broad_rff", 5, "opportunity_regret_r2")
    ]
    precision = lookup[
        ("unbounded_winner_fill", "broad_rff", 5, "material_precision")
    ]
    complete = len(rows) == 60 * len(DEPTHS) * len(METHODS) and all(
        np.isfinite(
            [
                row["opportunity_regret_r2"],
                row["mean_utility_regret_r2"],
                row["material_precision"],
                row["expected_membership_swaps"],
                row["realized_mean_utility_gain"],
                row["oracle_pool_headroom"],
                row["normalized_broad_boundary_margin"],
                row["broad_fold_inclusion_stability"],
                row["broad_bandwidth_inclusion_agreement"],
                row["broad_save_inclusion_agreement"],
            ]
        ).all()
        for row in rows
    )
    checks = [
        {
            "criterion": "top5_mean_regret_improves_broad",
            "observed_upper_95": mean_regret["upper_95"],
            "required": "<0",
            "pass": bool(mean_regret["upper_95"] < 0.0),
        },
        {
            "criterion": "top5_mean_regret_improves_in_each_cohort",
            "observed_by_cohort": mean_regret["by_cohort"],
            "required": "each <0",
            "pass": bool(all(value < 0.0 for value in mean_regret["by_cohort"].values())),
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
            "criterion": "all_depth_metrics_complete_and_finite",
            "observed_row_count": len(rows),
            "required_row_count": 60 * len(DEPTHS) * len(METHODS),
            "pass": bool(complete),
        },
    ]
    return {
        "status": "top5_confirmation_justified"
        if all(check["pass"] for check in checks)
        else "stop_no_top5_confirmation",
        "candidate": "unbounded_winner_fill",
        "checks": checks,
        "scope": "permits_one_unused_seed_synthetic_top5_confirmation_only",
    }


def summarize(protocol: Mapping[str, Any], loaded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for item in loaded
        for k in DEPTHS
        for row in panel_depth_rows(str(item["cohort"]), item["panel"], k)
    ]
    contrasts = []
    contrast_index = 0
    for method in ("save_rff_current", "unbounded_winner_fill", "bounded_2k"):
        for k in DEPTHS:
            for metric in (
                "mean_utility_regret_r2",
                "opportunity_regret_r2",
                "material_precision",
            ):
                contrasts.append(
                    _paired_bootstrap(
                        rows,
                        method,
                        "broad_rff",
                        k,
                        metric,
                        contrast_index * 1000,
                    )
                )
                contrast_index += 1
    return {
        "schema_version": 1,
        "status": "completed_post_outcome_depth_audit",
        "panel_count": len(loaded),
        "panel_depth_method_row_count": len(rows),
        "curve": _curve(rows),
        "contrasts": contrasts,
        "diagnostic_correlations": _diagnostic_correlations(rows),
        "panel_depth_metrics": rows,
        "decision": _decision(rows, contrasts),
        "protocol": dict(protocol),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous shortlist-depth audit",
        "",
        f"Decision: **{summary['decision']['status']}**.",
        "",
        "Negative regret contrasts favor the structured method; positive material-precision contrasts favor it.",
        "",
        "| k | Broad mean regret | Unbounded mean regret | Bounded 2k mean regret | Oracle-pool mean regret | Unbounded - broad (95% interval) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    curve = {(row["method"], row["k"]): row for row in summary["curve"]}
    contrasts = {
        (row["left"], row["k"], row["metric"]): row
        for row in summary["contrasts"]
        if row["right"] == "broad_rff"
    }
    for k in DEPTHS:
        contrast = contrasts[("unbounded_winner_fill", k, "mean_utility_regret_r2")]
        lines.append(
            f"| {k} | {curve[('broad_rff', k)]['mean_mean_utility_regret_r2']:.5f} | "
            f"{curve[('unbounded_winner_fill', k)]['mean_mean_utility_regret_r2']:.5f} | "
            f"{curve[('bounded_2k', k)]['mean_mean_utility_regret_r2']:.5f} | "
            f"{curve[('winner_preserved_pool_oracle', k)]['mean_mean_utility_regret_r2']:.5f} | "
            f"{contrast['estimate']:+.5f} [{contrast['lower_95']:+.5f}, {contrast['upper_95']:+.5f}] |"
        )
    lines.extend(["", "## Frozen top-five justification checks", ""])
    for check in summary["decision"]["checks"]:
        lines.append(f"- {'PASS' if check['pass'] else 'FAIL'}: `{check['criterion']}`")
    lines.extend(
        [
            "",
            "This is a post-outcome diagnostic reanalysis. A pass justifies one unused-seed top-five confirmation and does not promote a method.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _source_paths(roots: Mapping[str, Path]) -> dict[str, Path]:
    result = {
        "protocol": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256_PATH,
        "runner": RUNNER_PATH,
        "readme": PACKAGE_DIR / "README.md",
        "focused_test": ROOT / "tests" / "test_continuous_shortlist_depth_audit.py",
        "winner_parent_runner": winner_parent.RUNNER_PATH,
        "bounded_parent_runner": bounded_parent.RUNNER_PATH,
    }
    for cohort, root in roots.items():
        for name in ("raw_results.json", "summary.json", "manifest.json"):
            result[f"{cohort}_{name}"] = root.resolve() / name
    return result


def run(output: Path, roots: Mapping[str, Path] = DEFAULT_INPUTS) -> dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol(protocol)
    loaded = load_inputs(protocol, roots)
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
        key: common._sha256(path) for key, path in _source_paths(roots).items()
    }
    output = output.resolve()
    for name in ("summary.json", "report.md", "manifest.json"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite {output / name}")
    summary = summarize(protocol, loaded)
    summary["implementation_commit"] = implementation_commit
    summary["source_hashes"] = source_hashes
    common._atomic_json(output / "summary.json", summary)
    _atomic_text(output / "report.md", _report(summary))
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "implementation_commit": implementation_commit,
        "source_hashes": source_hashes,
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
    parser.add_argument("--winner-fill-root", type=Path, default=DEFAULT_INPUTS["winner_fill"])
    parser.add_argument("--bounded-root", type=Path, default=DEFAULT_INPUTS["bounded"])
    args = parser.parse_args()
    summary = run(
        args.output,
        {"winner_fill": args.winner_fill_root, "bounded": args.bounded_root},
    )
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
