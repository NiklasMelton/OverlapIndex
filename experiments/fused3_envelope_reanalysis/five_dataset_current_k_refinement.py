"""Five-dataset frozen-K FUSED3 refinement diagnostic.

The original frozen FUSED3 capacity rule is preserved exactly and every
unrefined recomputation must match the immutable Confirmation V2 score before
the refined score is retained.  The added operation is one balanced-median
pass in FUSED3's learned weighted metric.  This is post-outcome development
evidence only, not confirmation or promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import tempfile
from typing import Any

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics
from experiments.fused3_envelope_reanalysis.five_dataset_low_k_refinement import (
    FROZEN,
    LINEAR_PROBE,
    _reference_lookup,
    _validate_comparison_rows,
    aggregate_envelope,
    aggregate_heads,
    contrast_interval,
    envelope_panel_metrics,
    head_panel_metrics,
    runtime_summary,
)
from experiments.fused3_envelope_reanalysis.kmeans_k_sweep import (
    _canonical_json,
    _csv_bytes,
    _sha256,
)
from experiments.fused3_envelope_reanalysis.refined_fused3 import (
    crossfit_fused3_refinement,
)


STUDY = "fused3_confirmation_v2_five_dataset_current_k_refinement"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
CURRENT_REFINED = "FUSED3-CURRENT-K-REFINED"
METHODS = (FROZEN, LINEAR_PROBE, CURRENT_REFINED)
EXPECTED_K_BY_BUDGET: Mapping[int, int] = {32: 5, 64: 10}


def _frozen_score_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int, int], float]:
    lookup: dict[tuple[str, str, int, int], float] = {}
    for row in rows:
        if row["candidate_id"] != FROZEN:
            continue
        key = (
            str(row["dataset_id"]),
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
        )
        if key in lookup:
            raise ValueError("duplicate frozen FUSED3 score")
        lookup[key] = float(row["score"])
    expected = {
        (dataset, backbone, seed, budget)
        for dataset in statistics.DATASET_IDS
        for backbone in statistics.BACKBONES
        for seed in statistics.REPLICATE_SEEDS
        for budget in statistics.BUDGETS
    }
    if set(lookup) != expected:
        raise ValueError("frozen FUSED3 score grid is incomplete")
    return lookup


def analyze_current_k_refinement(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    """Execute frozen-K refinement across all five existing datasets."""

    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    source_selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    frozen_scores = _frozen_score_lookup(source_selectors)
    root = Path(registry_root)
    registry = json.loads((root / "audited_registry.json").read_text(encoding="utf-8"))
    registry_rows = {str(row["dataset_id"]): row for row in registry["datasets"]}
    if set(registry_rows) != set(statistics.DATASET_IDS):
        raise RuntimeError("audited registry dataset set mismatch")
    _reference_lookup(references)

    state_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for dataset_position, dataset_id in enumerate(statistics.DATASET_IDS, start=1):
        dataset_row = registry_rows[dataset_id]
        for backbone in statistics.BACKBONES:
            for replicate, seed in enumerate(statistics.REPLICATE_SEEDS):
                for budget in statistics.BUDGETS:
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
                    )
                    expected_k = EXPECTED_K_BY_BUDGET[int(budget)]
                    if any(
                        int(row["k_min"]) != expected_k
                        or int(row["k_max"]) != expected_k
                        for row in result["folds"]
                    ):
                        raise RuntimeError("resolved frozen K differs from expected capacity")
                    key = (dataset_id, backbone, int(seed), int(budget))
                    source_score = frozen_scores[key]
                    recomputed_score = float(result["scores"]["FUSED3"])
                    if recomputed_score != source_score:
                        raise RuntimeError("frozen FUSED3 recomputation parity failed")
                    parity_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "backbone": backbone,
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "k_per_class": int(expected_k),
                            "source_score": float(source_score),
                            "recomputed_score": float(recomputed_score),
                            "exact_match": True,
                        }
                    )
                    state_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "k_per_class": int(expected_k),
                            "candidate_id": CURRENT_REFINED,
                            "score": float(result["scores"]["FUSED3-REFINED"]),
                            "total_wall_seconds": float(
                                sum(
                                    row["refined_total_wall_seconds"]
                                    for row in result["folds"]
                                )
                            ),
                            "prototype_count": float(result["prototype_count_after"]),
                            "applied_count": float(result["applied_count"]),
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
                                **row,
                            }
                        )
        if progress:
            print(
                f"completed {dataset_position}/{len(statistics.DATASET_IDS)}: {dataset_id}",
                flush=True,
            )

    if len(state_rows) != 500 or len(parity_rows) != 500 or len(fold_rows) != 2500:
        raise RuntimeError("five-dataset current-K refinement row count mismatch")
    if not all(row["exact_match"] for row in parity_rows):
        raise RuntimeError("frozen baseline parity is not exact")
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
        for row in source_selectors
    ]
    comparison_rows = [*source_rows, *state_rows]
    _validate_comparison_rows(comparison_rows, methods=METHODS)
    envelope = envelope_panel_metrics(comparison_rows, references, methods=METHODS)
    heads = head_panel_metrics(comparison_rows, references, methods=METHODS)
    dataset_aggregate, overall = aggregate_envelope(envelope, methods=METHODS)
    head_aggregate = aggregate_heads(heads, methods=METHODS)
    runtimes = runtime_summary(
        comparison_rows,
        methods=METHODS,
        executed_methods=(CURRENT_REFINED,),
    )
    contrasts = [
        contrast_interval(
            envelope, candidate_id=CURRENT_REFINED, comparator_id=comparator
        )
        for comparator in (FROZEN, LINEAR_PROBE)
    ]
    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "primary_estimand": "best_head_envelope_backbone_ranking",
        "head_library": list(statistics.HEADS),
        "frozen_k_by_budget": {
            str(key): value for key, value in EXPECTED_K_BY_BUDGET.items()
        },
        "methods": list(METHODS),
        "executed_method": CURRENT_REFINED,
        "state_rows": state_rows,
        "baseline_parity_rows": parity_rows,
        "fold_rows": fold_rows,
        "panel_metrics": envelope,
        "dataset_aggregate": dataset_aggregate,
        "overall_aggregate": overall,
        "head_panel_metrics": heads,
        "head_aggregate": head_aggregate,
        "runtime_descriptive": runtimes,
        "regret_contrasts": contrasts,
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
        "# Five-dataset frozen-K FUSED3 refinement diagnostic",
        "",
        "Status: **post-outcome development diagnostic only**. This is not a new confirmation and the frozen decision is unchanged.",
        "",
        "The original FUSED3 capacity is preserved exactly (K=5 at budget 32; K=10 at budget 64), then one balanced-median pass is applied in its learned weighted metric. All 500 unrefined recomputations must match the frozen scores exactly.",
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
            "Negative values favor frozen-K refined FUSED3.",
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
            "Refined timings are from this separate diagnostic; frozen FUSED3/LP timings come from the original counterbalanced run. Ratios are descriptive.",
            "",
            "| Dataset | Budget | Selector | Call/LP | Full panel/LP |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in summary["runtime_descriptive"]:
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
    summary = analyze_current_k_refinement(
        full_artifact, registry_root, progress=progress
    )
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "state_rows.csv": _csv_bytes(summary["state_rows"]),
        "baseline_parity_rows.csv": _csv_bytes(summary["baseline_parity_rows"]),
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
            raise RuntimeError("written current-K refinement artifact hash mismatch")
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
