"""Post-outcome diagnosis of the FUSED3 Aircraft envelope-ranking failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import statistics


STUDY = "fused3_confirmation_v2_aircraft_failure_diagnostic"
STAGE = "post_outcome_exploratory_diagnostic"
DATASET_ID = "torchvision_fgvc_aircraft"
FOCAL_BACKBONES = ("dinov2-small", "openclip-vit-b-32")
STATUS = "descriptive_only_no_promotion"


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("Spearman inputs must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Spearman inputs must be finite")
    if np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = float(
        np.corrcoef(
            rankdata(left, method="average"),
            rankdata(right, method="average"),
        )[0, 1]
    )
    return value if np.isfinite(value) else None


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty diagnostic collection")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def analyze_aircraft(
    selector_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic diagnosis from the exact completed row tables."""

    selectors, references = statistics.validate_confirmation_inputs(
        selector_rows, reference_rows
    )
    aircraft_selectors = [
        row for row in selectors if row["dataset_id"] == DATASET_ID
    ]
    aircraft_references = [
        row for row in references if row["dataset_id"] == DATASET_ID
    ]
    if len(aircraft_selectors) != 200 or len(aircraft_references) != 400:
        raise ValueError("Aircraft diagnostic requires the exact 200/400 row grids")

    selector_groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    reference_groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in aircraft_selectors:
        selector_groups[
            (str(row["backbone"]), int(row["budget"]), str(row["candidate_id"]))
        ].append(float(row["score"]))
    for row in aircraft_references:
        reference_groups[
            (str(row["backbone"]), int(row["budget"]), str(row["head"]))
        ].append(float(row["test_accuracy"]))

    backbone_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for backbone in statistics.BACKBONES:
            selector_scores = {
                method: _mean(selector_groups[(backbone, budget, method)])
                for method in statistics.METHODS
            }
            head_accuracies = {
                head: _mean(reference_groups[(backbone, budget, head)])
                for head in statistics.HEADS
            }
            envelope = max(head_accuracies.values())
            best_heads = [
                head
                for head in statistics.HEADS
                if abs(head_accuracies[head] - envelope)
                <= statistics.SELECTION_TIE_ATOL
            ]
            backbone_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "budget": int(budget),
                    "backbone": backbone,
                    "FUSED3_score": selector_scores["FUSED3"],
                    "LP_FULL_score": selector_scores["LP-FULL"],
                    **{
                        f"{head}_accuracy": head_accuracies[head]
                        for head in statistics.HEADS
                    },
                    "best_head_envelope_accuracy": envelope,
                    "best_head_families": best_heads,
                    "quadratic_gain_over_linear_pp": float(
                        100.0
                        * (
                            head_accuracies["quadratic"]
                            - head_accuracies["linear"]
                        )
                    ),
                }
            )

    correlations: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        rows = [row for row in backbone_rows if row["budget"] == budget]
        for method, score_field in (
            ("FUSED3", "FUSED3_score"),
            ("LP-FULL", "LP_FULL_score"),
        ):
            for target, outcome_field in (
                ("best_head_envelope", "best_head_envelope_accuracy"),
                *((head, f"{head}_accuracy") for head in statistics.HEADS),
            ):
                correlations.append(
                    {
                        "budget": int(budget),
                        "candidate_id": method,
                        "target": target,
                        "spearman": _spearman(
                            [float(row[score_field]) for row in rows],
                            [float(row[outcome_field]) for row in rows],
                        ),
                        "backbone_count": len(rows),
                    }
                )

    selection_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for method in statistics.METHODS:
            counts: Counter[str] = Counter()
            for seed in statistics.REPLICATE_SEEDS:
                rows = [
                    row
                    for row in aircraft_selectors
                    if int(row["budget"]) == budget
                    and str(row["candidate_id"]) == method
                    and int(row["replicate_seed"]) == seed
                ]
                if len(rows) != len(statistics.BACKBONES):
                    raise ValueError("Aircraft selector panel is incomplete")
                maximum = max(float(row["score"]) for row in rows)
                selected = [
                    str(row["backbone"])
                    for row in rows
                    if abs(float(row["score"]) - maximum)
                    <= statistics.SELECTION_TIE_ATOL
                ]
                for backbone in selected:
                    counts[backbone] += 1.0 / len(selected)
            for backbone, count in sorted(counts.items()):
                selection_rows.append(
                    {
                        "budget": int(budget),
                        "candidate_id": method,
                        "backbone": backbone,
                        "selection_count": float(count),
                        "replicate_count": len(statistics.REPLICATE_SEEDS),
                    }
                )

    focal_rows: list[dict[str, Any]] = []
    by_identity = {
        (str(row["backbone"]), int(row["budget"])): row
        for row in backbone_rows
    }
    for budget in statistics.BUDGETS:
        dino = by_identity[(FOCAL_BACKBONES[0], budget)]
        clip = by_identity[(FOCAL_BACKBONES[1], budget)]
        focal_rows.append(
            {
                "budget": int(budget),
                "difference_direction": "OpenCLIP_minus_DINO",
                "FUSED3_score_delta": float(
                    clip["FUSED3_score"] - dino["FUSED3_score"]
                ),
                "LP_FULL_score_delta": float(
                    clip["LP_FULL_score"] - dino["LP_FULL_score"]
                ),
                **{
                    f"{head}_accuracy_delta_pp": float(
                        100.0
                        * (
                            clip[f"{head}_accuracy"]
                            - dino[f"{head}_accuracy"]
                        )
                    )
                    for head in statistics.HEADS
                },
                "envelope_accuracy_delta_pp": float(
                    100.0
                    * (
                        clip["best_head_envelope_accuracy"]
                        - dino["best_head_envelope_accuracy"]
                    )
                ),
                "DINO_quadratic_gain_over_linear_pp": dino[
                    "quadratic_gain_over_linear_pp"
                ],
                "OpenCLIP_quadratic_gain_over_linear_pp": clip[
                    "quadratic_gain_over_linear_pp"
                ],
            }
        )

    structural_rows: list[dict[str, Any]] = []
    for backbone in FOCAL_BACKBONES:
        for budget in statistics.BUDGETS:
            rows = [
                row
                for row in aircraft_selectors
                if row["candidate_id"] == "FUSED3"
                and row["backbone"] == backbone
                and int(row["budget"]) == budget
            ]
            folds = [fold for row in rows for fold in row["folds"]]
            feature_count = int(
                folds[0]["structural_diagnostics"]["n_features_fit"]
            )
            structural_rows.append(
                {
                    "budget": int(budget),
                    "backbone": backbone,
                    "fold_count": len(folds),
                    "feature_count": feature_count,
                    "mean_fold_OI_score": _mean(
                        [float(fold["score"]) for fold in folds]
                    ),
                    "mean_weight_condition": _mean(
                        [
                            float(
                                fold["structural_diagnostics"]["weight_condition"]
                            )
                            for fold in folds
                        ]
                    ),
                    "mean_positive_margin_feature_fraction": _mean(
                        [
                            float(
                                fold["structural_diagnostics"][
                                    "positive_margin_feature_count"
                                ]
                            )
                            / feature_count
                            for fold in folds
                        ]
                    ),
                    "margin_rows_per_fold": int(
                        folds[0]["structural_diagnostics"]["margin_row_count"]
                    ),
                }
            )

    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "dataset_id": DATASET_ID,
        "original_confirmation_decision": "fail_locked_candidate",
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "backbone_rows": backbone_rows,
        "correlations": correlations,
        "selection_rows": selection_rows,
        "focal_contrasts": focal_rows,
        "fused3_structural_diagnostics": structural_rows,
        "interpretation": {
            "primary_observation": "FUSED3 consistently selects OpenCLIP while the best-head envelope is quadratic and favors DINO",
            "budget_observation": "LP-FULL also selects OpenCLIP at budget 32 but switches to DINO at budget 64; FUSED3 does not switch",
            "mechanistic_hypothesis": "FUSED3 measures local prototype separability and therefore follows OpenCLIP's kNN advantage, while DINO's larger global second-order gain is outside the FUSED3 scoring surface",
            "causal_status": "diagnostic_concordance_not_a_causal_intervention",
        },
    }


def render_report(summary: Mapping[str, Any]) -> str:
    focal = {int(row["budget"]): row for row in summary["focal_contrasts"]}
    selection = summary["selection_rows"]
    corr = {
        (int(row["budget"]), str(row["candidate_id"]), str(row["target"])): row[
            "spearman"
        ]
        for row in summary["correlations"]
    }
    structural = {
        (str(row["backbone"]), int(row["budget"])): row
        for row in summary["fused3_structural_diagnostics"]
    }
    lines = [
        "# Why FUSED3 failed and LP succeeded on FGVC Aircraft",
        "",
        "Status: **post-outcome diagnostic only; no promotion or reselection**.",
        "",
        "The original confirmation decision remains `fail_locked_candidate`. This report uses only the existing frozen selector/reference rows and performs no refit or recipe change.",
        "",
        "## Selection path",
        "",
        "| Budget/class | Selector | Selected backbone(s) across five replicates |",
        "|---:|---|---|",
    ]
    for budget in statistics.BUDGETS:
        for method in statistics.METHODS:
            rows = [
                row
                for row in selection
                if row["budget"] == budget and row["candidate_id"] == method
            ]
            label = ", ".join(
                f"{row['backbone']} ({row['selection_count']:.0f}/5)"
                for row in rows
            )
            lines.append(f"| {budget} | {method} | {label} |")
    lines.extend(
        [
            "",
            "At 32 examples/class both selectors choose OpenCLIP. At 64, LP-FULL switches unanimously to DINO, while FUSED3 remains unanimously on OpenCLIP.",
            "",
            "## OpenCLIP minus DINO contrasts",
            "",
            "Positive score deltas mean the selector prefers OpenCLIP. Positive accuracy deltas mean OpenCLIP is better for that head.",
            "",
            "| Budget | FUSED3 score Δ | LP score Δ | Linear Δ pp | Quadratic Δ pp | kNN Δ pp | RBF Δ pp | Envelope Δ pp |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for budget in statistics.BUDGETS:
        row = focal[budget]
        lines.append(
            f"| {budget} | {row['FUSED3_score_delta']:+.4f} | {row['LP_FULL_score_delta']:+.4f} | {row['linear_accuracy_delta_pp']:+.2f} | {row['quadratic_accuracy_delta_pp']:+.2f} | {row['knn_accuracy_delta_pp']:+.2f} | {row['rbf_accuracy_delta_pp']:+.2f} | {row['envelope_accuracy_delta_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "This is the central conflict: OpenCLIP is locally better (kNN), but DINO's quadratic head is the envelope winner. The FUSED3 preference for OpenCLIP grows with budget even as DINO's envelope advantage grows.",
            "",
            "## Across-backbone rank correlations",
            "",
            "| Budget | Selector | Envelope Spearman | Linear | Quadratic | kNN | RBF |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for budget in statistics.BUDGETS:
        for method in statistics.METHODS:
            values = [
                corr[(budget, method, target)]
                for target in (
                    "best_head_envelope",
                    "linear",
                    "quadratic",
                    "knn",
                    "rbf",
                )
            ]
            labels = ["—" if value is None else f"{value:.3f}" for value in values]
            lines.append(
                f"| {budget} | {method} | " + " | ".join(labels) + " |"
            )
    lines.extend(
        [
            "",
            "LP's linear OOF score is a strong proxy for every frozen Aircraft head, not just the linear head. FUSED3's ordering weakens markedly at budget 64.",
            "",
            "## FUSED3 relevance-weight diagnostics",
            "",
            "| Backbone | Budget | Mean OI score | Weight condition | Positive-margin feature fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for backbone in FOCAL_BACKBONES:
        for budget in statistics.BUDGETS:
            row = structural[(backbone, budget)]
            lines.append(
                f"| {backbone} | {budget} | {row['mean_fold_OI_score']:.4f} | {row['mean_weight_condition']:.3f} | {row['mean_positive_margin_feature_fraction']:.3f} |"
            )
    lines.extend(
        [
            "",
            "DINO receives a much more anisotropic relevance metric (about 9.3× max/min weight versus about 2× for OpenCLIP), yet still produces much lower OI scores. This is consistent with useful DINO structure being distributed or jointly expressed rather than captured by FUSED3's global per-feature prototype-margin weighting.",
            "",
            "## Diagnosis",
            "",
            "1. FUSED3 is functioning as designed: it identifies OpenCLIP's stronger local/prototype geometry and kNN performance.",
            "2. The best-head envelope on Aircraft is quadratic for every backbone/budget cell, so local geometry is not the decisive capability.",
            "3. LP succeeds because, with 64 examples/class, its global linear score ranks DINO above OpenCLIP and is highly concordant with quadratic performance across the full backbone panel.",
            "4. The failure is therefore an objective blind spot, not random instability or an implementation error: FUSED3 lacks a global second-order potential term.",
            "",
            "This is diagnostic concordance, not a causal intervention. A follow-up should add a cheap global/second-order companion while leaving FUSED3's local component unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = [dict(row) for row in rows]
    fields = sorted({str(key) for row in ordered for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow(
            {
                field: frozen_analysis.canonical_json(row.get(field))
                if isinstance(row.get(field), (Mapping, list, tuple))
                else row.get(field)
                for field in fields
            }
        )
    return buffer.getvalue().encode("utf-8")


def write_diagnostic_bundle(
    input_dir: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    loaded = frozen_analysis.verify_completed_artifact(input_dir)
    summary = analyze_aircraft(
        loaded["selector_rows"], loaded["reference_rows"]
    )
    summary["input_hashes"] = dict(loaded["input_hashes"])
    summary["lineage"] = {
        key: loaded["raw"][key]
        for key in (
            "protocol_sha256",
            "code_identity_sha256",
            "run_identity_sha256",
            "audited_registry_sha256",
            "lineage_lock_sha256",
        )
    }
    summary["diagnostic_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    payloads = {
        "summary.json": (
            frozen_analysis.canonical_json(summary) + "\n"
        ).encode("utf-8"),
        "backbone_surface.csv": _csv_bytes(summary["backbone_rows"]),
        "rank_correlations.csv": _csv_bytes(summary["correlations"]),
        "focal_contrasts.csv": _csv_bytes(summary["focal_contrasts"]),
        "fused3_structural_diagnostics.csv": _csv_bytes(
            summary["fused3_structural_diagnostics"]
        ),
        "report.md": render_report(summary).encode("utf-8"),
    }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        manifest = {
            "schema_version": 1,
            "study": STUDY,
            "stage": STAGE,
            "artifact_status": "completed",
            "diagnostic_status": STATUS,
            "original_confirmation_decision_unchanged": True,
            "promotion_or_reselection_performed": False,
            "input_hashes": dict(summary["input_hashes"]),
            "lineage": dict(summary["lineage"]),
            "diagnostic_code_sha256": summary["diagnostic_code_sha256"],
            "files": {
                name: {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in sorted(payloads.items())
            },
        }
        (staging / "diagnostic_manifest.json").write_bytes(
            (frozen_analysis.canonical_json(manifest) + "\n").encode("utf-8")
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = write_diagnostic_bundle(args.input, args.output)
    print(frozen_analysis.canonical_json(summary["interpretation"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATASET_ID",
    "FOCAL_BACKBONES",
    "STATUS",
    "STAGE",
    "STUDY",
    "analyze_aircraft",
    "render_report",
    "write_diagnostic_bundle",
]
