"""Five-dataset low-K FUSED3 refinement reanalysis.

This is a post-outcome development diagnostic over the already-inspected
Confirmation V2 inputs.  It cannot confirm, promote, tune, or reselect a
candidate.  The primary estimand is backbone-ranking regret against the frozen
best-head envelope over linear, quadratic, kNN, and RBF heads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics
from experiments.fused3_envelope_reanalysis.kmeans_k_sweep import (
    _canonical_json,
    _csv_bytes,
    _sha256,
    _spearman,
)
from experiments.fused3_envelope_reanalysis.low_k_refinement import (
    LOW_BASELINE,
    LOW_K_BY_BUDGET,
    LOW_REFINED,
    validate_low_k_design,
)
from experiments.fused3_envelope_reanalysis.refined_fused3 import (
    crossfit_fused3_refinement,
)


STUDY = "fused3_confirmation_v2_five_dataset_low_k_refinement"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
FROZEN = "FUSED3"
LINEAR_PROBE = "LP-FULL"
METHODS = (FROZEN, LINEAR_PROBE, LOW_BASELINE, LOW_REFINED)
EXECUTED_METHODS = (LOW_BASELINE, LOW_REFINED)


def _reference_lookup(
    references: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, int, str], float]:
    lookup: dict[tuple[str, str, int, int, str], float] = {}
    for row in references:
        key = (
            str(row["dataset_id"]),
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["head"]),
        )
        if key in lookup:
            raise ValueError("duplicate reference outcome")
        value = float(row["test_accuracy"])
        if not np.isfinite(value):
            raise ValueError("reference outcome must be finite")
        lookup[key] = value
    expected = {
        (dataset, backbone, seed, budget, head)
        for dataset in statistics.DATASET_IDS
        for backbone in statistics.BACKBONES
        for seed in statistics.REPLICATE_SEEDS
        for budget in statistics.BUDGETS
        for head in statistics.HEADS
    }
    if set(lookup) != expected:
        raise ValueError("reference outcome grid is incomplete")
    return lookup


def _validate_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = METHODS,
) -> dict[tuple[str, str, int, int, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str, int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["dataset_id"]),
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        )
        if key in lookup:
            raise ValueError("duplicate comparison selector row")
        if key[0] not in statistics.DATASET_IDS or key[1] not in statistics.BACKBONES:
            raise ValueError("comparison row has an unknown dataset/backbone")
        if key[2] not in statistics.REPLICATE_SEEDS or key[3] not in statistics.BUDGETS:
            raise ValueError("comparison row has an unknown seed/budget")
        if key[4] not in methods:
            raise ValueError("comparison row has an unknown method")
        score = float(row["score"])
        elapsed = float(row["total_wall_seconds"])
        if not np.isfinite(score) or not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("comparison scores/timings must be finite and nonnegative")
        lookup[key] = row
    expected = {
        (dataset, backbone, seed, budget, method)
        for dataset in statistics.DATASET_IDS
        for backbone in statistics.BACKBONES
        for seed in statistics.REPLICATE_SEEDS
        for budget in statistics.BUDGETS
        for method in methods
    }
    if set(lookup) != expected:
        raise ValueError("comparison selector grid is incomplete")
    return lookup


def envelope_panel_metrics(
    selector_rows: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = METHODS,
) -> list[dict[str, Any]]:
    """Compute tie-safe best-head-envelope metrics on complete panels."""

    selectors = _validate_comparison_rows(selector_rows, methods=methods)
    outcomes = _reference_lookup(references)
    result: list[dict[str, Any]] = []
    for dataset in statistics.DATASET_IDS:
        for seed in statistics.REPLICATE_SEEDS:
            for budget in statistics.BUDGETS:
                envelope = np.asarray(
                    [
                        max(
                            outcomes[(dataset, backbone, seed, budget, head)]
                            for head in statistics.HEADS
                        )
                        for backbone in statistics.BACKBONES
                    ],
                    dtype=np.float64,
                )
                best = float(np.max(envelope))
                for method in methods:
                    scores = np.asarray(
                        [
                            float(
                                selectors[
                                    (dataset, backbone, seed, budget, method)
                                ]["score"]
                            )
                            for backbone in statistics.BACKBONES
                        ],
                        dtype=np.float64,
                    )
                    maximum = float(np.max(scores))
                    selected = np.isclose(
                        scores,
                        maximum,
                        atol=statistics.SELECTION_TIE_ATOL,
                        rtol=0.0,
                    )
                    selected_outcomes = envelope[selected]
                    rho = _spearman(scores, envelope)
                    result.append(
                        {
                            "dataset_id": dataset,
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "candidate_id": method,
                            "selected_count": int(np.count_nonzero(selected)),
                            "selected_backbones": json.dumps(
                                [
                                    backbone
                                    for backbone, chosen in zip(
                                        statistics.BACKBONES, selected.tolist()
                                    )
                                    if chosen
                                ],
                                separators=(",", ":"),
                            ),
                            "best_envelope_accuracy": best,
                            "selected_envelope_accuracy": float(
                                np.mean(selected_outcomes)
                            ),
                            "regret_pp": float(
                                100.0 * (best - float(np.mean(selected_outcomes)))
                            ),
                            "exact_best": float(np.mean(selected_outcomes == best)),
                            "within_one_pp": float(
                                np.mean(selected_outcomes >= best - 0.01)
                            ),
                            "spearman": rho,
                            "spearman_status": (
                                "defined" if rho is not None else "undefined_constant"
                            ),
                            "tie_rule": "maxima_within_absolute_tolerance_1e-12_averaged",
                        }
                    )
    return result


def head_panel_metrics(
    selector_rows: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = METHODS,
) -> list[dict[str, Any]]:
    """Compute secondary per-head regret for the same selector rankings."""

    selectors = _validate_comparison_rows(selector_rows, methods=methods)
    outcomes = _reference_lookup(references)
    result: list[dict[str, Any]] = []
    for dataset in statistics.DATASET_IDS:
        for seed in statistics.REPLICATE_SEEDS:
            for budget in statistics.BUDGETS:
                for head in statistics.HEADS:
                    head_outcomes = np.asarray(
                        [
                            outcomes[(dataset, backbone, seed, budget, head)]
                            for backbone in statistics.BACKBONES
                        ],
                        dtype=np.float64,
                    )
                    best = float(np.max(head_outcomes))
                    for method in methods:
                        scores = np.asarray(
                            [
                                float(
                                    selectors[
                                        (dataset, backbone, seed, budget, method)
                                    ]["score"]
                                )
                                for backbone in statistics.BACKBONES
                            ],
                            dtype=np.float64,
                        )
                        selected = np.isclose(
                            scores,
                            float(np.max(scores)),
                            atol=statistics.SELECTION_TIE_ATOL,
                            rtol=0.0,
                        )
                        selected_accuracy = float(np.mean(head_outcomes[selected]))
                        rho = _spearman(scores, head_outcomes)
                        result.append(
                            {
                                "dataset_id": dataset,
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                "head": head,
                                "candidate_id": method,
                                "regret_pp": float(100.0 * (best - selected_accuracy)),
                                "spearman": rho,
                                "spearman_status": (
                                    "defined" if rho is not None else "undefined_constant"
                                ),
                            }
                        )
    return result


def aggregate_envelope(
    metrics: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = METHODS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_rows: list[dict[str, Any]] = []
    for dataset in statistics.DATASET_IDS:
        for method in methods:
            selected = [
                row
                for row in metrics
                if row["dataset_id"] == dataset and row["candidate_id"] == method
            ]
            if len(selected) != len(statistics.REPLICATE_SEEDS) * len(
                statistics.BUDGETS
            ):
                raise ValueError("envelope metric aggregate is incomplete")
            counts: Counter[str] = Counter()
            for row in selected:
                for backbone in json.loads(str(row["selected_backbones"])):
                    counts[str(backbone)] += 1
            defined = [row["spearman"] for row in selected if row["spearman"] is not None]
            dataset_rows.append(
                {
                    "dataset_id": dataset,
                    "candidate_id": method,
                    "mean_regret_pp": float(np.mean([row["regret_pp"] for row in selected])),
                    "exact_best_rate": float(np.mean([row["exact_best"] for row in selected])),
                    "within_one_pp_rate": float(
                        np.mean([row["within_one_pp"] for row in selected])
                    ),
                    "mean_spearman": (
                        None if not defined else float(np.mean(defined))
                    ),
                    "selection_counts": json.dumps(
                        dict(sorted(counts.items())), separators=(",", ":")
                    ),
                    "panel_count": len(selected),
                }
            )
    overall: list[dict[str, Any]] = []
    for method in methods:
        selected = [row for row in dataset_rows if row["candidate_id"] == method]
        overall.append(
            {
                "candidate_id": method,
                "equal_dataset_mean_regret_pp": float(
                    np.mean([row["mean_regret_pp"] for row in selected])
                ),
                "equal_dataset_exact_best_rate": float(
                    np.mean([row["exact_best_rate"] for row in selected])
                ),
                "equal_dataset_within_one_pp_rate": float(
                    np.mean([row["within_one_pp_rate"] for row in selected])
                ),
                "equal_dataset_mean_spearman": float(
                    np.mean(
                        [row["mean_spearman"] for row in selected if row["mean_spearman"] is not None]
                    )
                ),
            }
        )
    return dataset_rows, overall


def aggregate_heads(
    metrics: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = METHODS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for head in statistics.HEADS:
        for method in methods:
            per_dataset = []
            per_dataset_rho = []
            for dataset in statistics.DATASET_IDS:
                selected = [
                    row
                    for row in metrics
                    if row["dataset_id"] == dataset
                    and row["head"] == head
                    and row["candidate_id"] == method
                ]
                if len(selected) != 10:
                    raise ValueError("head metric aggregate is incomplete")
                per_dataset.append(float(np.mean([row["regret_pp"] for row in selected])))
                defined = [row["spearman"] for row in selected if row["spearman"] is not None]
                if defined:
                    per_dataset_rho.append(float(np.mean(defined)))
            rows.append(
                {
                    "head": head,
                    "candidate_id": method,
                    "equal_dataset_mean_regret_pp": float(np.mean(per_dataset)),
                    "equal_dataset_mean_spearman": (
                        None
                        if not per_dataset_rho
                        else float(np.mean(per_dataset_rho))
                    ),
                }
            )
    return rows


def runtime_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = METHODS,
    executed_methods: Sequence[str] = EXECUTED_METHODS,
) -> list[dict[str, Any]]:
    lookup = _validate_comparison_rows(rows, methods=methods)
    output: list[dict[str, Any]] = []
    for dataset in statistics.DATASET_IDS:
        for budget in statistics.BUDGETS:
            lp_calls = np.asarray(
                [
                    lookup[(dataset, backbone, seed, budget, LINEAR_PROBE)][
                        "total_wall_seconds"
                    ]
                    for backbone in statistics.BACKBONES
                    for seed in statistics.REPLICATE_SEEDS
                ],
                dtype=np.float64,
            )
            for method in methods:
                calls = np.asarray(
                    [
                        lookup[(dataset, backbone, seed, budget, method)][
                            "total_wall_seconds"
                        ]
                        for backbone in statistics.BACKBONES
                        for seed in statistics.REPLICATE_SEEDS
                    ],
                    dtype=np.float64,
                )
                panel_totals = []
                lp_panel_totals = []
                for seed in statistics.REPLICATE_SEEDS:
                    panel_totals.append(
                        float(
                            sum(
                                lookup[(dataset, backbone, seed, budget, method)][
                                    "total_wall_seconds"
                                ]
                                for backbone in statistics.BACKBONES
                            )
                        )
                    )
                    lp_panel_totals.append(
                        float(
                            sum(
                                lookup[
                                    (dataset, backbone, seed, budget, LINEAR_PROBE)
                                ]["total_wall_seconds"]
                                for backbone in statistics.BACKBONES
                            )
                        )
                    )
                output.append(
                    {
                        "dataset_id": dataset,
                        "budget": int(budget),
                        "candidate_id": method,
                        "median_call_wall_seconds": float(np.median(calls)),
                        "median_call_ratio_to_lp": float(
                            np.median(calls) / np.median(lp_calls)
                        ),
                        "median_full_panel_wall_seconds": float(
                            np.median(panel_totals)
                        ),
                        "median_full_panel_ratio_to_lp": float(
                            np.median(
                                np.asarray(panel_totals)
                                / np.asarray(lp_panel_totals)
                            )
                        ),
                        "timing_status": "descriptive_separate_runs"
                        if method in executed_methods
                        else "frozen_counterbalanced_source",
                    }
                )
    return output


def contrast_interval(
    metrics: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    comparator_id: str,
) -> dict[str, Any]:
    lookup = {
        (
            str(row["dataset_id"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        ): float(row["regret_pp"])
        for row in metrics
    }
    contrasts = [
        {
            "dataset_id": dataset,
            "replicate_seed": int(seed),
            "budget": int(budget),
            "contrast_pp": float(
                lookup[(dataset, seed, budget, candidate_id)]
                - lookup[(dataset, seed, budget, comparator_id)]
            ),
        }
        for dataset in statistics.DATASET_IDS
        for seed in statistics.REPLICATE_SEEDS
        for budget in statistics.BUDGETS
    ]
    interval = statistics.hierarchical_interval(contrasts)
    return {
        "candidate_id": candidate_id,
        "comparator_id": comparator_id,
        "sign": "negative_favors_candidate",
        **interval,
        "status": "descriptive_post_outcome",
        "promotion_veto": False,
    }


def analyze_five_datasets(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    """Execute low-K off/on refinement over all five confirmation datasets."""

    low_k = validate_low_k_design()
    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    frozen_selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    root = Path(registry_root)
    registry = json.loads((root / "audited_registry.json").read_text(encoding="utf-8"))
    registry_rows = {str(row["dataset_id"]): row for row in registry["datasets"]}
    if set(registry_rows) != set(statistics.DATASET_IDS):
        raise RuntimeError("audited registry dataset set mismatch")
    reference = _reference_lookup(references)

    state_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for dataset_position, dataset_id in enumerate(statistics.DATASET_IDS, start=1):
        dataset_row = registry_rows[dataset_id]
        for backbone in statistics.BACKBONES:
            for replicate, seed in enumerate(statistics.REPLICATE_SEEDS):
                for budget in statistics.BUDGETS:
                    k = low_k[int(budget)]
                    panel = datasets.load_panel(
                        dataset_row,
                        backbone,
                        int(seed),
                        int(budget),
                        registry_root=root,
                    )
                    result = crossfit_fused3_refinement(
                        panel["training_values"],
                        panel["training_labels"],
                        seed=int(seed),
                        collect_own_win_rates=False,
                        k=int(k),
                    )
                    for method in EXECUTED_METHODS:
                        refined = method == LOW_REFINED
                        state_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "backbone": backbone,
                                "replicate": int(replicate),
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                "k_per_class": int(k),
                                "candidate_id": method,
                                "score": float(
                                    result["scores"][
                                        "FUSED3-REFINED" if refined else "FUSED3"
                                    ]
                                ),
                                "total_wall_seconds": float(
                                    sum(
                                        row[
                                            "refined_total_wall_seconds"
                                            if refined
                                            else "baseline_total_wall_seconds"
                                        ]
                                        for row in result["folds"]
                                    )
                                ),
                                "prototype_count": float(
                                    result[
                                        "prototype_count_after"
                                        if refined
                                        else "prototype_count_before"
                                    ]
                                ),
                                "applied_count": float(
                                    result["applied_count"] if refined else 0.0
                                ),
                            }
                        )
                    for row in result["folds"]:
                        fold_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "backbone": backbone,
                                "replicate": int(replicate),
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                "k_per_class": int(k),
                                **row,
                            }
                        )
        if progress:
            print(
                f"completed {dataset_position}/{len(statistics.DATASET_IDS)}: {dataset_id}",
                flush=True,
            )

    if len(state_rows) != 1000 or len(fold_rows) != 2500:
        raise RuntimeError("five-dataset low-K refinement row count mismatch")
    source_rows = [
        {
            "dataset_id": str(row["dataset_id"]),
            "backbone": str(row["backbone"]),
            "replicate": int(row["replicate"]),
            "replicate_seed": int(row["replicate_seed"]),
            "budget": int(row["budget"]),
            "candidate_id": str(row["candidate_id"]),
            "score": float(row["score"]),
            "total_wall_seconds": float(row["total_wall_seconds"]),
            "source": "immutable_confirmation_v2",
        }
        for row in frozen_selectors
    ]
    comparison_rows = [*source_rows, *state_rows]
    _validate_comparison_rows(comparison_rows)
    envelope = envelope_panel_metrics(comparison_rows, references)
    head_metrics = head_panel_metrics(comparison_rows, references)
    dataset_aggregate, overall = aggregate_envelope(envelope)
    head_aggregate = aggregate_heads(head_metrics)
    runtimes = runtime_summary(comparison_rows)
    contrasts = [
        contrast_interval(
            envelope, candidate_id=LOW_REFINED, comparator_id=comparator
        )
        for comparator in (FROZEN, LINEAR_PROBE, LOW_BASELINE)
    ]
    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "primary_estimand": "best_head_envelope_backbone_ranking",
        "head_library": list(statistics.HEADS),
        "low_k_by_budget": {str(key): value for key, value in low_k.items()},
        "methods": list(METHODS),
        "executed_methods": list(EXECUTED_METHODS),
        "state_rows": state_rows,
        "fold_rows": fold_rows,
        "panel_metrics": envelope,
        "dataset_aggregate": dataset_aggregate,
        "overall_aggregate": overall,
        "head_panel_metrics": head_metrics,
        "head_aggregate": head_aggregate,
        "runtime_descriptive": runtimes,
        "regret_contrasts": contrasts,
        "reference_lookup_count": len(reference),
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "post_outcome_limit": (
            "All five datasets and their outcomes were previously inspected; "
            "this result is development evidence and not confirmation."
        ),
        "input_hashes": dict(verified["input_hashes"]),
        "input_identity": {
            key: verified["raw"][key]
            for key in (
                "protocol_sha256",
                "code_identity_sha256",
                "run_identity_sha256",
                "audited_registry_sha256",
                "lineage_lock_sha256",
            )
        },
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Five-dataset low-K FUSED3 refinement diagnostic",
        "",
        "Status: **post-outcome development diagnostic only**. This is not a new confirmation and the frozen decision is unchanged.",
        "",
        "Primary target: the best held-out accuracy available to each backbone over the frozen linear, quadratic, kNN, and RBF head library.",
        "",
        "## Overall best-head-envelope ranking",
        "",
        "| Selector | Mean regret (pp) | Exact best | Within 1 pp | Mean Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["overall_aggregate"]:
        lines.append(
            "| {candidate_id} | {equal_dataset_mean_regret_pp:.3f} | "
            "{equal_dataset_exact_best_rate:.3f} | "
            "{equal_dataset_within_one_pp_rate:.3f} | "
            "{equal_dataset_mean_spearman:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Dataset results",
            "",
            "| Dataset | Selector | Regret (pp) | Exact best | Spearman | Selections |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["dataset_aggregate"]:
        rho = "—" if row["mean_spearman"] is None else f"{row['mean_spearman']:.3f}"
        lines.append(
            f"| {row['dataset_id']} | {row['candidate_id']} | "
            f"{row['mean_regret_pp']:.3f} | {row['exact_best_rate']:.3f} | "
            f"{rho} | `{row['selection_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Paired envelope-regret contrasts",
            "",
            "Negative values favor low-K refined FUSED3.",
            "",
            "| Comparator | Estimate (pp) | Descriptive 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["regret_contrasts"]:
        lines.append(
            f"| {row['comparator_id']} | {row['estimate']:+.3f} | "
            f"[{row['lower_95']:+.3f}, {row['upper_95']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Secondary per-head ranking regret",
            "",
            "| Head | Selector | Equal-dataset mean regret (pp) | Mean Spearman |",
            "|---|---|---:|---:|",
        ]
    )
    for row in summary["head_aggregate"]:
        rho = (
            "—"
            if row["equal_dataset_mean_spearman"] is None
            else f"{row['equal_dataset_mean_spearman']:.3f}"
        )
        lines.append(
            f"| {row['head']} | {row['candidate_id']} | "
            f"{row['equal_dataset_mean_regret_pp']:.3f} | {rho} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "Low-K timings are from this separate diagnostic; frozen FUSED3/LP timings come from the original counterbalanced run. Ratios are descriptive.",
            "",
            "| Dataset | Budget | Selector | Call/LP | Full panel/LP |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in summary["runtime_descriptive"]:
        if row["candidate_id"] not in (LOW_REFINED, FROZEN, LINEAR_PROBE):
            continue
        lines.append(
            f"| {row['dataset_id']} | {row['budget']} | {row['candidate_id']} | "
            f"{row['median_call_ratio_to_lp']:.2f}× | "
            f"{row['median_full_panel_ratio_to_lp']:.2f}× |"
        )
    lines.extend(
        [
            "",
            "No promotion, reselection, or confirmation claim is made from these previously inspected datasets.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_five_datasets(
        full_artifact, registry_root, progress=progress
    )
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "state_rows.csv": _csv_bytes(summary["state_rows"]),
        "fold_rows.csv": _csv_bytes(summary["fold_rows"]),
        "panel_metrics.csv": _csv_bytes(summary["panel_metrics"]),
        "dataset_aggregate.csv": _csv_bytes(summary["dataset_aggregate"]),
        "overall_aggregate.csv": _csv_bytes(summary["overall_aggregate"]),
        "head_panel_metrics.csv": _csv_bytes(summary["head_panel_metrics"]),
        "head_aggregate.csv": _csv_bytes(summary["head_aggregate"]),
        "runtime_descriptive.csv": _csv_bytes(summary["runtime_descriptive"]),
        "regret_contrasts.csv": _csv_bytes(summary["regret_contrasts"]),
        "report.md": render_report(summary).encode("utf-8"),
    }
    manifest = {
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "files": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            for name, content in payloads.items()
        },
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
    }
    payloads["diagnostic_manifest.json"] = (
        _canonical_json(manifest) + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for name, content in payloads.items():
            (staging / name).write_bytes(content)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for name, descriptor in manifest["files"].items():
        if _sha256(destination / name) != descriptor["sha256"]:
            raise RuntimeError("written five-dataset artifact hash mismatch")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_bundle(args.input, args.registry_root, args.output, progress=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
