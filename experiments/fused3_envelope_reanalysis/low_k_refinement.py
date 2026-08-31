"""Aircraft diagnostic for low-K FUSED3 with balanced-median refinement.

This post-outcome diagnostic tests the two low-K settings identified by the
earlier closed capacity sweep: K=3 at budget 32 and K=2 at budget 64.  Each is
evaluated with refinement disabled and enabled.  The frozen confirmation and
all previous diagnostic artifacts remain immutable.
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
from experiments.fused3_envelope_reanalysis.kmeans_k_sweep import (
    DATASET_ID,
    FOCAL_BACKBONES,
    STUDY as K_SWEEP_STUDY,
    _canonical_json,
    _csv_bytes,
    _lookups,
    _sha256,
)
from experiments.fused3_envelope_reanalysis.refined_fused3 import (
    STUDY as CURRENT_REFINEMENT_STUDY,
    crossfit_fused3_refinement,
    summarize_refinement,
)


STUDY = "fused3_confirmation_v2_aircraft_low_k_refinement"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
LOW_K_BY_BUDGET: Mapping[int, int] = {32: 3, 64: 2}
LOW_BASELINE = "FUSED3-KLOW"
LOW_REFINED = "FUSED3-KLOW-REFINED"
LOW_METHODS = (LOW_BASELINE, LOW_REFINED)
SOURCE_CURRENT_METHODS = ("FUSED3", "FUSED3-REFINED")


def validate_low_k_design(
    design: Mapping[int, int] = LOW_K_BY_BUDGET,
) -> dict[int, int]:
    if dict(design) != {32: 3, 64: 2}:
        raise ValueError("low-K refinement design must be exactly {32: 3, 64: 2}")
    return {int(key): int(value) for key, value in design.items()}


def _load_diagnostic_summary(
    path: os.PathLike[str] | str, *, expected_study: str
) -> tuple[dict[str, Any], dict[str, str]]:
    root = Path(path)
    manifest_path = root / "diagnostic_manifest.json"
    summary_path = root / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("study") != expected_study
        or manifest.get("artifact_status") != "completed"
        or manifest.get("diagnostic_status") != STATUS
        or manifest.get("original_confirmation_decision_unchanged") is not True
        or manifest.get("promotion_or_reselection_performed") is not False
    ):
        raise RuntimeError("source diagnostic identity/status mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or "summary.json" not in files:
        raise RuntimeError("source diagnostic file manifest is incomplete")
    hashes: dict[str, str] = {}
    for name, descriptor in files.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise RuntimeError("source diagnostic file descriptor is malformed")
        data = (root / name).read_bytes()
        observed = hashlib.sha256(data).hexdigest()
        if (
            observed != descriptor.get("sha256")
            or len(data) != descriptor.get("size_bytes")
        ):
            raise RuntimeError("source diagnostic file hash/size mismatch")
        hashes[name] = observed
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("study") != expected_study
        or summary.get("artifact_status") != "completed"
        or summary.get("diagnostic_status") != STATUS
        or summary.get("original_confirmation_decision_unchanged") is not True
        or summary.get("promotion_or_reselection_performed") is not False
    ):
        raise RuntimeError("source diagnostic summary mismatch")
    return summary, hashes


def _current_comparison_rows(
    current_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate = [dict(row) for row in current_summary["aggregate_rows"]]
    focal = [dict(row) for row in current_summary["focal_rows"]]
    ranking = [dict(row) for row in current_summary["ranking_rows"]]
    if {
        (int(row["budget"]), str(row["candidate_id"])) for row in aggregate
    } != {
        (budget, method)
        for budget in statistics.BUDGETS
        for method in SOURCE_CURRENT_METHODS
    }:
        raise RuntimeError("current-K comparison aggregate grid is incomplete")
    return aggregate, focal, ranking


def analyze_low_k_refinement(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    k_sweep_artifact: os.PathLike[str] | str,
    current_refinement_artifact: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Run the complete low-K off/on-refinement Aircraft comparison."""

    low_k = validate_low_k_design()
    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    _frozen_scores, reference = _lookups(selectors, references)
    sweep_summary, sweep_hashes = _load_diagnostic_summary(
        k_sweep_artifact, expected_study=K_SWEEP_STUDY
    )
    current_summary, current_hashes = _load_diagnostic_summary(
        current_refinement_artifact, expected_study=CURRENT_REFINEMENT_STUDY
    )
    sweep_lookup = {
        (
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            int(row["k_per_class"]),
        ): float(row["score"])
        for row in sweep_summary["state_rows"]
    }
    current_aggregate, current_focal, current_ranking = _current_comparison_rows(
        current_summary
    )

    root = Path(registry_root)
    registry = json.loads((root / "audited_registry.json").read_text(encoding="utf-8"))
    aircraft = next(
        row for row in registry["datasets"] if row["dataset_id"] == DATASET_ID
    )
    state_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for backbone in statistics.BACKBONES:
        for replicate, seed in enumerate(statistics.REPLICATE_SEEDS):
            for budget in statistics.BUDGETS:
                k = low_k[int(budget)]
                panel = datasets.load_panel(
                    aircraft,
                    backbone,
                    int(seed),
                    int(budget),
                    registry_root=root,
                )
                identity = (backbone, int(seed), int(budget))
                heads = {
                    head: reference[identity + (head,)] for head in statistics.HEADS
                }
                envelope = max(heads.values())
                best_heads = [
                    head
                    for head in statistics.HEADS
                    if abs(heads[head] - envelope) <= statistics.SELECTION_TIE_ATOL
                ]
                result = crossfit_fused3_refinement(
                    panel["training_values"],
                    panel["training_labels"],
                    seed=int(seed),
                    collect_own_win_rates=backbone in FOCAL_BACKBONES,
                    k=int(k),
                )
                expected = sweep_lookup[identity + (int(k),)]
                if abs(float(result["scores"]["FUSED3"]) - expected) > 1.0e-12:
                    raise RuntimeError("low-K baseline differs from the prior K sweep")
                for method in LOW_METHODS:
                    refined = method == LOW_REFINED
                    state_rows.append(
                        {
                            "dataset_id": DATASET_ID,
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
                            "best_own_win_rate": result[
                                f"{'refined' if refined else 'baseline'}_best_own_win_rate"
                            ],
                            "second_own_win_rate": result[
                                f"{'refined' if refined else 'baseline'}_second_own_win_rate"
                            ],
                            "best_head_envelope_accuracy": float(envelope),
                            "best_head_families": json.dumps(
                                best_heads, separators=(",", ":")
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
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "k_per_class": int(k),
                            **row,
                        }
                    )
    if len(state_rows) != 200 or len(fold_rows) != 500:
        raise RuntimeError("completed low-K refinement grid has the wrong row count")

    low_ranking, low_aggregate, low_focal = summarize_refinement(
        state_rows, methods=LOW_METHODS
    )
    baseline_time = {
        int(row["budget"]): float(row["median_panel_wall_seconds"])
        for row in current_aggregate
        if row["candidate_id"] == "FUSED3"
    }
    comparison_aggregate = [*current_aggregate, *low_aggregate]
    for row in comparison_aggregate:
        row["time_ratio_to_current_fused3"] = float(
            row["median_panel_wall_seconds"] / baseline_time[int(row["budget"])]
        )
    comparison_focal = [*current_focal, *low_focal]
    comparison_ranking = [*current_ranking, *low_ranking]
    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "dataset_id": DATASET_ID,
        "low_k_by_budget": {str(key): value for key, value in low_k.items()},
        "methods_executed": list(LOW_METHODS),
        "comparison_methods": [*SOURCE_CURRENT_METHODS, *LOW_METHODS],
        "changed_components": [
            "class_owned_prototypes_per_class",
            "balanced_median_refinement_enabled_or_disabled",
        ],
        "state_rows": state_rows,
        "fold_rows": fold_rows,
        "ranking_rows": low_ranking,
        "aggregate_rows": low_aggregate,
        "focal_rows": low_focal,
        "comparison_ranking_rows": comparison_ranking,
        "comparison_aggregate_rows": comparison_aggregate,
        "comparison_focal_rows": comparison_focal,
        "source_diagnostics": {
            "k_sweep": {
                "path": str(Path(k_sweep_artifact)),
                "study": K_SWEEP_STUDY,
                "file_sha256": sweep_hashes,
            },
            "current_k_refinement": {
                "path": str(Path(current_refinement_artifact)),
                "study": CURRENT_REFINEMENT_STUDY,
                "file_sha256": current_hashes,
            },
        },
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
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
        "# Aircraft low-K FUSED3 refinement diagnostic",
        "",
        "Status: **post-outcome descriptive diagnostic only**. The frozen confirmation decision is unchanged.",
        "",
        "Low K is fixed at 3 for budget 32 and 2 for budget 64. All rows use the same Aircraft cohorts, folds, relevance learner, weighted refinement rule, scorer, and best-head envelope.",
        "",
        "| Budget | Candidate | Mean regret (pp) | Mean Spearman | Mean prototypes | Mean splits | Time/current FUSED3 | Selections |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["comparison_aggregate_rows"]:
        lines.append(
            "| {budget} | {candidate_id} | {mean_regret_pp:.3f} | "
            "{mean_spearman:.3f} | {mean_prototype_count:.1f} | "
            "{mean_applied_count:.1f} | {time_ratio_to_current_fused3:.2f}× | "
            "`{selection_counts}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## DINO versus OpenCLIP",
            "",
            "| Budget | Candidate | OpenCLIP − DINO score |",
            "|---:|---|---:|",
        ]
    )
    for row in summary["comparison_focal_rows"]:
        lines.append(
            "| {budget} | {candidate_id} | {OpenCLIP_minus_DINO_score:+.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Positive OpenCLIP − DINO means the selector still favors OpenCLIP. Timings compare separate post-outcome diagnostic runs and are descriptive, not counterbalanced product gates.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    k_sweep_artifact: os.PathLike[str] | str,
    current_refinement_artifact: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_low_k_refinement(
        full_artifact,
        registry_root,
        k_sweep_artifact,
        current_refinement_artifact,
    )
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "state_rows.csv": _csv_bytes(summary["state_rows"]),
        "fold_rows.csv": _csv_bytes(summary["fold_rows"]),
        "ranking_rows.csv": _csv_bytes(summary["ranking_rows"]),
        "aggregate_rows.csv": _csv_bytes(summary["aggregate_rows"]),
        "focal_rows.csv": _csv_bytes(summary["focal_rows"]),
        "comparison_ranking_rows.csv": _csv_bytes(
            summary["comparison_ranking_rows"]
        ),
        "comparison_aggregate_rows.csv": _csv_bytes(
            summary["comparison_aggregate_rows"]
        ),
        "comparison_focal_rows.csv": _csv_bytes(summary["comparison_focal_rows"]),
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
            raise RuntimeError("written low-K refinement artifact hash mismatch")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--k-sweep-artifact", type=Path, required=True)
    parser.add_argument("--current-refinement-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_bundle(
        args.input,
        args.registry_root,
        args.k_sweep_artifact,
        args.current_refinement_artifact,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
