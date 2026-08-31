"""Post-outcome best-head-envelope reanalysis of FUSED3 confirmation V2.

This module does not alter the frozen confirmation estimand, gates, or decision.
It evaluates a newly clarified, explicitly exploratory target: each backbone is
valued by the maximum held-out accuracy obtained by any frozen reference head.
Both FUSED3 and LP-FULL are then evaluated as backbone selectors against that
same per-backbone envelope.
"""

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
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import statistics


STUDY = "fused3_confirmation_v2_best_head_envelope_reanalysis"
STAGE = "post_outcome_exploratory_reanalysis"
ORIGINAL_DECISION_STATUS = "fail_locked_candidate"
REANALYSIS_STATUS = "descriptive_only_no_promotion"
ENVELOPE_ID = "best_of_frozen_linear_quadratic_knn_rbf"


def _materialize(
    rows: Iterable[Mapping[str, Any]], *, name: str
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"{name}[{position}] must be a mapping")
        materialized.append(dict(row))
    return materialized


def _spearman(
    scores: Sequence[float], outcomes: Sequence[float]
) -> tuple[float | None, str]:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("Spearman inputs must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Spearman inputs must be finite")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None, "undefined_constant"
    value = float(
        np.corrcoef(
            rankdata(x, method="average"), rankdata(y, method="average")
        )[0, 1]
    )
    if not np.isfinite(value):
        return None, "undefined_nonfinite"
    return value, "defined"


def envelope_panel_metrics(
    selector_rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate both selectors against the per-backbone best-head envelope."""

    selectors, references = statistics.validate_confirmation_inputs(
        selector_rows, reference_rows
    )
    selector_lookup = {
        (
            str(row["dataset_id"]),
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        ): float(row["score"])
        for row in selectors
    }
    reference_lookup = {
        (
            str(row["dataset_id"]),
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["head"]),
        ): float(row["test_accuracy"])
        for row in references
    }

    output: list[dict[str, Any]] = []
    for dataset in statistics.DATASET_IDS:
        for seed in statistics.REPLICATE_SEEDS:
            replicate = statistics.REPLICATE_SEEDS.index(seed)
            for budget in statistics.BUDGETS:
                backbone_envelopes: list[float] = []
                backbone_best_heads: list[list[str]] = []
                for backbone in statistics.BACKBONES:
                    head_values = np.asarray(
                        [
                            reference_lookup[
                                (dataset, backbone, seed, budget, head)
                            ]
                            for head in statistics.HEADS
                        ],
                        dtype=np.float64,
                    )
                    maximum = float(np.max(head_values))
                    best_heads = [
                        head
                        for head, value in zip(statistics.HEADS, head_values)
                        if abs(float(value) - maximum)
                        <= statistics.SELECTION_TIE_ATOL
                    ]
                    backbone_envelopes.append(maximum)
                    backbone_best_heads.append(best_heads)

                outcomes = np.asarray(backbone_envelopes, dtype=np.float64)
                oracle_best = float(np.max(outcomes))
                oracle_mask = np.isclose(
                    outcomes,
                    oracle_best,
                    atol=statistics.SELECTION_TIE_ATOL,
                    rtol=0.0,
                )
                oracle_backbones = [
                    backbone
                    for backbone, chosen in zip(statistics.BACKBONES, oracle_mask)
                    if bool(chosen)
                ]
                oracle_heads = sorted(
                    {
                        head
                        for heads, chosen in zip(backbone_best_heads, oracle_mask)
                        if bool(chosen)
                        for head in heads
                    }
                )

                for method in statistics.METHODS:
                    scores = np.asarray(
                        [
                            selector_lookup[
                                (dataset, backbone, seed, budget, method)
                            ]
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
                    selected_outcomes = outcomes[selected]
                    selected_accuracy = float(np.mean(selected_outcomes))
                    selected_backbones = [
                        backbone
                        for backbone, chosen in zip(statistics.BACKBONES, selected)
                        if bool(chosen)
                    ]
                    selected_best_heads = {
                        backbone: list(heads)
                        for backbone, heads, chosen in zip(
                            statistics.BACKBONES, backbone_best_heads, selected
                        )
                        if bool(chosen)
                    }
                    rho, rho_status = _spearman(scores, outcomes)
                    output.append(
                        {
                            "dataset_id": dataset,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "envelope_id": ENVELOPE_ID,
                            "candidate_id": method,
                            "backbone_count": len(statistics.BACKBONES),
                            "selected_count": int(np.count_nonzero(selected)),
                            "selected_backbones": selected_backbones,
                            "selected_backbone_best_heads": selected_best_heads,
                            "oracle_best_backbones": oracle_backbones,
                            "oracle_best_head_families": oracle_heads,
                            "best_envelope_accuracy": oracle_best,
                            "selected_envelope_accuracy": selected_accuracy,
                            "regret_pp": float(
                                100.0 * (oracle_best - selected_accuracy)
                            ),
                            "exact_best": float(
                                np.mean(
                                    np.isclose(
                                        selected_outcomes,
                                        oracle_best,
                                        atol=statistics.SELECTION_TIE_ATOL,
                                        rtol=0.0,
                                    )
                                )
                            ),
                            "within_one_pp": float(
                                np.mean(selected_outcomes >= oracle_best - 0.01)
                            ),
                            "spearman": rho,
                            "spearman_status": rho_status,
                            "selector_tie_rule": (
                                "maxima_within_absolute_tolerance_1e-12_averaged"
                            ),
                            "head_envelope_tie_rule": (
                                "all_head_maxima_within_absolute_tolerance_1e-12"
                            ),
                        }
                    )
    expected = (
        len(statistics.DATASET_IDS)
        * len(statistics.REPLICATE_SEEDS)
        * len(statistics.BUDGETS)
        * len(statistics.METHODS)
    )
    if len(output) != expected:
        raise AssertionError("best-head envelope metric grid is incomplete")
    return output


def envelope_contrast_rows(
    metrics: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return paired FUSED3-minus-LP envelope-regret contrasts."""

    rows = _materialize(metrics, name="metrics")
    lookup: dict[tuple[str, int, int, str], float] = {}
    for row in rows:
        identity = (
            str(row["dataset_id"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        )
        if identity in lookup:
            raise ValueError(f"duplicate envelope metric identity {identity!r}")
        lookup[identity] = float(row["regret_pp"])
    expected = {
        (dataset, seed, budget, method)
        for dataset in statistics.DATASET_IDS
        for seed in statistics.REPLICATE_SEEDS
        for budget in statistics.BUDGETS
        for method in statistics.METHODS
    }
    if set(lookup) != expected:
        raise ValueError("envelope contrast requires the exact complete metric grid")
    return [
        {
            "dataset_id": dataset,
            "replicate_seed": int(seed),
            "budget": int(budget),
            "envelope_id": ENVELOPE_ID,
            "candidate_id": statistics.LOCKED_CANDIDATE,
            "comparator_id": statistics.COMPARATOR,
            "contrast_pp": float(
                lookup[(dataset, seed, budget, statistics.LOCKED_CANDIDATE)]
                - lookup[(dataset, seed, budget, statistics.COMPARATOR)]
            ),
        }
        for dataset in statistics.DATASET_IDS
        for seed in statistics.REPLICATE_SEEDS
        for budget in statistics.BUDGETS
    ]


def envelope_rank_auc_rows(
    metrics: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Integrate the two-budget envelope Spearman curves descriptively."""

    rows = _materialize(metrics, name="metrics")
    grouped: dict[tuple[str, int, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["dataset_id"]),
            int(row["replicate_seed"]),
            str(row["candidate_id"]),
        )
        budget = int(row["budget"])
        if budget in grouped[key]:
            raise ValueError(f"duplicate envelope rank-AUC budget for {key!r}")
        grouped[key][budget] = row
    expected = {
        (dataset, seed, method)
        for dataset in statistics.DATASET_IDS
        for seed in statistics.REPLICATE_SEEDS
        for method in statistics.METHODS
    }
    if set(grouped) != expected:
        raise ValueError("envelope rank AUC requires the exact curve set")
    x = np.log2(np.asarray(statistics.BUDGETS, dtype=np.float64))
    width = float(x[-1] - x[0])
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        observed = grouped[key]
        if set(observed) != set(statistics.BUDGETS):
            raise ValueError("envelope rank AUC requires both frozen budgets")
        values = [observed[budget].get("spearman") for budget in statistics.BUDGETS]
        statuses = [
            str(observed[budget].get("spearman_status"))
            for budget in statistics.BUDGETS
        ]
        if any(
            value is None or status != "defined"
            for value, status in zip(values, statuses)
        ):
            auc: float | None = None
            status = "undefined_supporting"
        else:
            y = np.asarray([float(value) for value in values], dtype=np.float64)
            auc = float(
                np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)) / width
            )
            status = "defined"
        output.append(
            {
                "dataset_id": key[0],
                "replicate_seed": int(key[1]),
                "candidate_id": key[2],
                "envelope_id": ENVELOPE_ID,
                "rank_auc": auc,
                "status": status,
                "budgets": list(statistics.BUDGETS),
                "supporting_only": True,
                "promotion_veto": False,
            }
        )
    return output


def summarize_envelope(
    selector_rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = statistics.BOOTSTRAP_RESAMPLES,
    seed: int = statistics.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build the complete exploratory envelope summary."""

    selectors = _materialize(selector_rows, name="selector_rows")
    references = _materialize(reference_rows, name="reference_rows")
    metrics = envelope_panel_metrics(selectors, references)
    contrasts = envelope_contrast_rows(metrics)
    interval = statistics.hierarchical_interval(
        contrasts, n_resamples=n_resamples, seed=seed
    )
    rank_auc = envelope_rank_auc_rows(metrics)

    aggregate: list[dict[str, Any]] = []
    for dataset in statistics.DATASET_IDS:
        for method in statistics.METHODS:
            selected = [
                row
                for row in metrics
                if row["dataset_id"] == dataset
                and row["candidate_id"] == method
            ]
            defined_rho = [
                float(row["spearman"])
                for row in selected
                if row["spearman"] is not None
            ]
            aggregate.append(
                {
                    "dataset_id": dataset,
                    "candidate_id": method,
                    "panel_count": len(selected),
                    "mean_regret_pp": float(
                        np.mean([float(row["regret_pp"]) for row in selected])
                    ),
                    "exact_best_rate": float(
                        np.mean([float(row["exact_best"]) for row in selected])
                    ),
                    "within_one_pp_rate": float(
                        np.mean([float(row["within_one_pp"]) for row in selected])
                    ),
                    "mean_spearman": (
                        float(np.mean(defined_rho)) if defined_rho else None
                    ),
                }
            )

    head_counts: Counter[str] = Counter()
    oracle_head_counts: Counter[str] = Counter()
    reference_lookup: dict[tuple[str, int, int, str], dict[str, float]] = defaultdict(dict)
    for row in references:
        reference_lookup[
            (
                str(row["dataset_id"]),
                int(row["replicate_seed"]),
                int(row["budget"]),
                str(row["backbone"]),
            )
        ][str(row["head"])] = float(row["test_accuracy"])
    for values in reference_lookup.values():
        maximum = max(values.values())
        winners = [
            head
            for head, value in values.items()
            if abs(value - maximum) <= statistics.SELECTION_TIE_ATOL
        ]
        for head in winners:
            head_counts[head] += 1
    for row in metrics:
        if row["candidate_id"] != statistics.LOCKED_CANDIDATE:
            continue
        for head in row["oracle_best_head_families"]:
            oracle_head_counts[str(head)] += 1

    overall = []
    for method in statistics.METHODS:
        selected = [row for row in metrics if row["candidate_id"] == method]
        overall.append(
            {
                "candidate_id": method,
                "panel_count": len(selected),
                "mean_regret_pp": float(
                    np.mean([float(row["regret_pp"]) for row in selected])
                ),
                "exact_best_rate": float(
                    np.mean([float(row["exact_best"]) for row in selected])
                ),
                "within_one_pp_rate": float(
                    np.mean([float(row["within_one_pp"]) for row in selected])
                ),
            }
        )

    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "reanalysis_status": REANALYSIS_STATUS,
        "original_confirmation_decision": ORIGINAL_DECISION_STATUS,
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "estimand": {
            "id": ENVELOPE_ID,
            "backbone_value": "maximum held-out test accuracy over the four frozen head families",
            "selector_regret": "best backbone envelope accuracy minus mean envelope accuracy of selector-tied maxima",
            "heads": list(statistics.HEADS),
            "post_outcome": True,
        },
        "panel_metrics": metrics,
        "regret_contrasts": contrasts,
        "comparison": {
            "contrast": "FUSED3_regret_minus_LP_FULL_regret",
            "negative_favors": statistics.LOCKED_CANDIDATE,
            "inferential_role": "descriptive_post_outcome_not_a_frozen_gate",
            **interval,
        },
        "aggregate": aggregate,
        "overall": overall,
        "rank_auc": rank_auc,
        "head_prevalence": {
            "backbone_panel_best_head_memberships": {
                head: int(head_counts.get(head, 0)) for head in statistics.HEADS
            },
            "oracle_best_backbone_head_memberships": {
                head: int(oracle_head_counts.get(head, 0))
                for head in statistics.HEADS
            },
            "tie_note": "memberships include every head tied at the maximum",
        },
    }


def analyze_loaded(
    loaded: Mapping[str, Any],
    *,
    n_resamples: int = statistics.BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    summary = summarize_envelope(
        loaded["selector_rows"],
        loaded["reference_rows"],
        n_resamples=n_resamples,
    )
    raw = loaded["raw"]
    summary["input_hashes"] = dict(loaded["input_hashes"])
    summary["lineage"] = {
        "protocol_sha256": raw["protocol_sha256"],
        "code_identity_sha256": raw["code_identity_sha256"],
        "run_identity_sha256": raw["run_identity_sha256"],
        "audited_registry_sha256": raw["audited_registry_sha256"],
        "lineage_lock_sha256": raw["lineage_lock_sha256"],
    }
    summary["reanalysis_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    overall = {row["candidate_id"]: row for row in summary["overall"]}
    comparison = summary["comparison"]
    lines = [
        "# FUSED3 confirmation V2: best-head-envelope reanalysis",
        "",
        "Status: **descriptive post-outcome reanalysis; no promotion or reselection**.",
        "",
        "The frozen confirmation decision remains **`fail_locked_candidate`**. This versioned analysis answers a newly clarified product question and does not replace, amend, or reinterpret the predeclared head-specific gates.",
        "",
        "## Estimand",
        "",
        "For each backbone, the oracle value is the maximum held-out accuracy over the frozen linear, quadratic, kNN, and RBF heads. FUSED3 and LP-FULL each select a backbone using only their original selector score and are evaluated against this same best-head envelope.",
        "",
        "Regret is `oracle-best envelope accuracy - selected-backbone envelope accuracy` in percentage points. The paired contrast is `FUSED3 regret - LP-FULL regret`; negative values favor FUSED3.",
        "",
        "## Overall result",
        "",
        "| Selector | Mean envelope regret pp | Exact best | Within 1 pp |",
        "|---|---:|---:|---:|",
    ]
    for method in statistics.METHODS:
        row = overall[method]
        lines.append(
            f"| {method} | {row['mean_regret_pp']:.3f} | {row['exact_best_rate']:.3f} | {row['within_one_pp_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Paired FUSED3-minus-LP regret: **{comparison['estimate']:.3f} pp**, descriptive hierarchical 95% interval **[{comparison['lower_95']:.3f}, {comparison['upper_95']:.3f}]**.",
            "",
            "## Dataset results",
            "",
            "| Dataset | FUSED3 regret pp | LP-FULL regret pp | Difference pp |",
            "|---|---:|---:|---:|",
        ]
    )
    aggregate = {
        (row["dataset_id"], row["candidate_id"]): row
        for row in summary["aggregate"]
    }
    for dataset in statistics.DATASET_IDS:
        fused = aggregate[(dataset, statistics.LOCKED_CANDIDATE)]["mean_regret_pp"]
        probe = aggregate[(dataset, statistics.COMPARATOR)]["mean_regret_pp"]
        lines.append(
            f"| {dataset} | {fused:.3f} | {probe:.3f} | {fused - probe:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Best-head prevalence",
            "",
            "The counts below cover all 500 backbone × dataset/replicate/budget cells. A tied head contributes one membership to each tied family.",
            "",
            "| Head | Best-head memberships |",
            "|---|---:|",
        ]
    )
    counts = summary["head_prevalence"]["backbone_panel_best_head_memberships"]
    for head in statistics.HEADS:
        lines.append(f"| {head} | {counts[head]} |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This estimand was specified after confirmation outcomes were available.",
            "- The interval is a descriptive application of the original hierarchical bootstrap, not a frozen confirmation gate.",
            "- Existing confirmation data may now inform future development, but cannot be reused as untouched confirmation for an envelope selector.",
            "- The frozen head library defines the envelope; adding or tuning head families would change the target.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]], *, sort_fields: Sequence[str]
) -> bytes:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: tuple(str(row.get(field, "")) for field in sort_fields),
    )
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


def write_reanalysis_bundle(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    n_resamples: int = statistics.BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Verify the frozen input and publish a non-overwriting reanalysis bundle."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing reanalysis: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    loaded = frozen_analysis.verify_completed_artifact(input_dir)
    summary = analyze_loaded(loaded, n_resamples=n_resamples)
    payloads = {
        "summary.json": (
            frozen_analysis.canonical_json(summary) + "\n"
        ).encode("utf-8"),
        "envelope_panel_metrics.csv": _csv_bytes(
            summary["panel_metrics"],
            sort_fields=(
                "dataset_id",
                "replicate_seed",
                "budget",
                "candidate_id",
            ),
        ),
        "envelope_regret_contrasts.csv": _csv_bytes(
            summary["regret_contrasts"],
            sort_fields=("dataset_id", "replicate_seed", "budget"),
        ),
        "envelope_rank_auc.csv": _csv_bytes(
            summary["rank_auc"],
            sort_fields=("dataset_id", "replicate_seed", "candidate_id"),
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
            "reanalysis_status": REANALYSIS_STATUS,
            "original_confirmation_decision": ORIGINAL_DECISION_STATUS,
            "original_confirmation_decision_unchanged": True,
            "promotion_or_reselection_performed": False,
            "input_hashes": dict(summary["input_hashes"]),
            "lineage": dict(summary["lineage"]),
            "reanalysis_code_sha256": summary["reanalysis_code_sha256"],
            "files": {
                name: {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in sorted(payloads.items())
            },
        }
        (staging / "reanalysis_manifest.json").write_bytes(
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
    summary = write_reanalysis_bundle(args.input, args.output)
    print(frozen_analysis.canonical_json(summary["comparison"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENVELOPE_ID",
    "ORIGINAL_DECISION_STATUS",
    "REANALYSIS_STATUS",
    "STAGE",
    "STUDY",
    "analyze_loaded",
    "envelope_contrast_rows",
    "envelope_panel_metrics",
    "envelope_rank_auc_rows",
    "render_report",
    "summarize_envelope",
    "write_reanalysis_bundle",
]
