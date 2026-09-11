"""Read-only compatibility helpers for archived opportunity artifacts.

The original experiment-local source was removed when its temporary worktree
was cleaned, while downstream frozen experiments still import two pure
reporting helpers from it.  This module restores only those deterministic
helpers; it cannot fit models or write artifacts.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


HEADS = ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")


def _top_indices(values: np.ndarray, depth: int, atol: float) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    threshold = float(values[order[min(depth, len(order)) - 1]])
    return np.flatnonzero(values >= threshold - atol)


def ranking_metrics(
    truth: Sequence[float], score: Sequence[float], *, topk: int = 3,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    truth_array = np.asarray(truth, dtype=np.float64)
    score_array = np.asarray(score, dtype=np.float64)
    if (
        truth_array.ndim != 1
        or score_array.shape != truth_array.shape
        or len(truth_array) < 2
        or not np.all(np.isfinite(truth_array))
        or not np.all(np.isfinite(score_array))
    ):
        raise ValueError("ranking inputs are invalid")
    selected = np.flatnonzero(score_array >= np.max(score_array) - atol)
    truth_best = np.flatnonzero(truth_array >= np.max(truth_array) - atol)
    selected_top = _top_indices(score_array, topk, atol)
    truth_top = _top_indices(truth_array, topk, atol)
    correlation = spearmanr(truth_array, score_array).statistic
    return {
        "spearman": float(correlation),
        "top1_regret_r2": float(np.max(truth_array) - np.max(truth_array[selected])),
        "exact_best": bool(set(selected) & set(truth_best)),
        "topk_opportunity_regret_r2": float(
            np.max(truth_array) - np.max(truth_array[selected_top])
        ),
        "topk_set_recall": float(
            len(set(selected_top) & set(truth_top)) / len(set(truth_top))
        ),
    }


def panel_opportunity_rows(
    panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    definition = protocol["opportunity_definition"]
    material = float(definition["material_uplift_r2"])
    pair_material = float(definition["material_pair_difference_r2"])
    topk = int(definition["topk"])
    atol = float(definition["tie_tolerance"])
    rows: list[dict[str, Any]] = []
    for panel in panels:
        linear = np.asarray(
            [candidate["heldout_linear"] for candidate in panel["candidates"]],
            dtype=np.float64,
        )
        envelope = np.asarray(
            [candidate["envelope"] for candidate in panel["candidates"]],
            dtype=np.float64,
        )
        uplift = envelope - linear
        if np.min(uplift) < -atol:
            raise ValueError("true nonlinear uplift is negative")
        uplift = np.maximum(uplift, 0.0)
        linear_metrics = ranking_metrics(envelope, linear, topk=topk, atol=atol)
        reversals = 0
        material_reversals = 0
        pair_count = 0
        for left in range(len(linear)):
            for right in range(left + 1, len(linear)):
                linear_delta = float(linear[left] - linear[right])
                envelope_delta = float(envelope[left] - envelope[right])
                linear_sign = 0 if abs(linear_delta) <= atol else (1 if linear_delta > 0 else -1)
                envelope_sign = 0 if abs(envelope_delta) <= atol else (1 if envelope_delta > 0 else -1)
                changed = linear_sign != envelope_sign
                reversals += int(changed)
                material_reversals += int(
                    changed
                    and abs(linear_delta) >= pair_material - atol
                    and abs(envelope_delta) >= pair_material - atol
                )
                pair_count += 1
        winner_counts = {head: 0 for head in HEADS}
        for candidate in panel["candidates"]:
            maximum = max(float(candidate["heads"][head]["test_r2"]) for head in HEADS)
            winners = [
                head for head in HEADS
                if float(candidate["heads"][head]["test_r2"]) >= maximum - atol
            ]
            if len(winners) == 1:
                winner_counts[winners[0]] += 1
        rows.append(
            {
                "panel_id": panel["panel_id"],
                "task_id": panel["task_id"],
                "source_id": panel["source_id"],
                "dataset_id": panel["dataset_id"],
                "target_id": panel["target_id"],
                "roster": panel["roster"],
                "seed": panel["seed"],
                "candidate_count": len(linear),
                "candidate_best_head_r2_values": [float(value) for value in envelope],
                "median_candidate_best_head_r2": float(np.median(envelope)),
                "envelope_span_r2": float(np.max(envelope) - np.min(envelope)),
                "material_uplift_candidate_count": int(np.sum(uplift >= material - atol)),
                "positive_uplift_candidate_count": int(np.sum(uplift > atol)),
                "mean_true_uplift_r2": float(np.mean(uplift)),
                "maximum_true_uplift_r2": float(np.max(uplift)),
                "linear_spearman": linear_metrics["spearman"],
                "linear_top1_regret_r2": linear_metrics["top1_regret_r2"],
                "linear_exact_best": linear_metrics["exact_best"],
                "linear_top3_opportunity_regret_r2": linear_metrics[
                    "topk_opportunity_regret_r2"
                ],
                "linear_top3_set_recall": linear_metrics["topk_set_recall"],
                "pair_count": pair_count,
                "pair_order_change_rate": float(reversals / pair_count),
                "material_pair_reversal_rate": float(material_reversals / pair_count),
                "unique_heldout_winner_counts": winner_counts,
            }
        )
    return rows


def _classify_task(
    row: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    admission = protocol["admission"]
    rank_gate = admission["rank_change_any_of"]
    checks = {
        "recoverable": float(row["median_candidate_best_head_r2"])
        >= float(admission["median_candidate_best_head_r2_minimum"]),
        "envelope_spread": float(row["median_panel_envelope_span_r2"])
        >= float(admission["median_panel_envelope_span_r2_minimum"]),
        "material_nonlinear_prevalence": float(row["material_nonlinear_candidate_fraction"])
        >= float(admission["material_nonlinear_candidate_fraction_minimum"]),
    }
    rank_checks = {
        "winner_change": float(row["linear_winner_change_rate"])
        >= float(rank_gate["linear_winner_change_rate_minimum"]),
        "top1_regret": float(row["mean_linear_top1_regret_r2"])
        >= float(rank_gate["mean_linear_top1_regret_r2_minimum"]),
        "material_pair_reversal": float(row["mean_material_pair_reversal_rate"])
        >= float(rank_gate["mean_material_pair_reversal_rate_minimum"]),
        "top3_miss": float(row["linear_top3_miss_rate"])
        >= float(rank_gate["linear_top3_miss_rate_minimum"]),
    }
    checks["rank_change_any"] = bool(any(rank_checks.values()))
    opportunity = bool(all(checks.values()))
    if opportunity:
        stratum = "rank_changing_nonlinear_opportunity"
    elif checks["recoverable"] and checks["envelope_spread"]:
        stratum = "recoverable_lp_dominated_control"
    else:
        stratum = "unusable_for_ranking"
    return {"stratum": stratum, "checks": checks, "rank_change_checks": rank_checks}


__all__ = ["HEADS", "_classify_task", "panel_opportunity_rows", "ranking_metrics"]
