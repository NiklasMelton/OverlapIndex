from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.scalable_relevance_kmeans import fused_food_full as full
from experiments.scalable_relevance_kmeans import fused_food_full_analysis as analysis


def _raw() -> dict:
    selector_rows = []
    archived_rows = []
    reference_rows = []
    refined_deltas = []
    screen_parity = []
    lp_parity = []
    for model_index, model in enumerate(full.MODELS):
        for replicate in full.REPLICATES:
            for arm in full.ARMS:
                for head in analysis.HEADS:
                    reference_rows.append(
                        {
                            "backbone": model,
                            "replicate": replicate,
                            "arm": arm,
                            "head": head,
                            "test_accuracy": 0.50 + 0.01 * model_index,
                        }
                    )
                for budget in full.BUDGETS:
                    panel_id = full._panel_id(model, replicate, arm, budget)
                    for candidate, wall in (
                        ("FUSED3", 0.5),
                        ("REFINED-OI", 0.8),
                        ("LP-FULL", 1.0),
                    ):
                        selector_rows.append(
                            {
                                "model": model,
                                "replicate": replicate,
                                "arm": arm,
                                "budget": budget,
                                "candidate_id": candidate,
                                "score": float(model_index),
                                "total_wall_seconds": wall,
                            }
                        )
                    archived_rows.append(
                        {
                            "model": model,
                            "replicate": replicate,
                            "arm": arm,
                            "budget": budget,
                            "candidate_id": "REFINED-OI-ARCHIVED",
                            "score": float(model_index),
                        }
                    )
                    lp_parity.append({"panel_id": panel_id, "delta": 0.0, "exact": True})
                    refined_deltas.append(
                        {"panel_id": panel_id, "delta": 0.0, "exact": True}
                    )
                    if replicate == 0 and budget == 80:
                        screen_parity.append(
                            {"panel_id": panel_id, "delta": 0.0, "exact": True}
                        )
    return {
        "selector_rows": selector_rows,
        "archived_refined_rows": archived_rows,
        "reference_rows": reference_rows,
        "lp_parity": lp_parity,
        "refined_historical_deltas": refined_deltas,
        "screen_parity": screen_parity,
    }


def test_complete_equal_ranking_fixture_passes_without_reselection() -> None:
    result = analysis.analyze(_raw())
    assert result["decision"]["status"] == "pass_full_retrospective"
    assert result["decision"]["selected_candidate"] == "FUSED3"
    assert result["decision"]["runner_up_reselection_allowed"] is False
    assert result["decision"]["failed_required_gates"] == []
    assert all(row["status"] == "pass" for row in result["gates"])
    assert len(result["rank_auc"]) == 5 * 3 * 4 * 4
    assert all(np.isclose(row["rank_auc"], 1.0) for row in result["rank_auc"])


def test_locked_candidate_failure_does_not_promote_refined_oi() -> None:
    raw = _raw()
    for row in raw["selector_rows"]:
        if row["candidate_id"] == "FUSED3" and row["arm"] == "nuisance_full":
            row["score"] = -float(row["score"])
    result = analysis.analyze(raw)
    assert result["decision"]["status"] == "fail_locked_candidate"
    assert result["decision"]["selected_candidate"] == "FUSED3"
    assert result["decision"]["runner_up_reselection_allowed"] is False
    assert any(
        row["gate"] == "nuisance_linear_vs_probe" and row["status"] == "fail"
        for row in result["gates"]
    )


def test_runtime_contrasts_include_per_call_and_summed_panels() -> None:
    result = analysis.analyze(_raw())
    rows = result["runtime_contrasts"]
    assert len(rows) == 12
    fused_probe = [row for row in rows if row["comparator"] == "LP-FULL"]
    assert {row["kind"] for row in fused_probe} == {"per_call", "full_panel"}
    assert all(np.isclose(row["estimate"], 0.5) for row in fused_probe)
    assert all(row["status"] == "established" for row in result["faster_claims"])


def test_full_panel_gate_catches_one_expensive_backbone() -> None:
    raw = _raw()
    for row in raw["selector_rows"]:
        if row["candidate_id"] == "FUSED3" and row["model"] == full.MODELS[-1]:
            row["total_wall_seconds"] = 20.0
    result = analysis.analyze(raw)
    per_call = [
        row
        for row in result["runtime_contrasts"]
        if row["comparator"] == "LP-FULL" and row["kind"] == "per_call"
    ]
    full_panel = [
        row
        for row in result["runtime_contrasts"]
        if row["comparator"] == "LP-FULL" and row["kind"] == "full_panel"
    ]
    assert all(row["estimate"] == 0.5 for row in per_call)
    assert all(row["estimate"] > 1.0 for row in full_panel)
    assert result["decision"]["status"] == "fail_locked_candidate"


def test_paired_intervals_are_deterministic_and_use_five_blocks() -> None:
    first = analysis._paired_interval(
        [0.1] * 5, key="constant", statistic=lambda values: float(np.mean(values))
    )
    second = analysis._paired_interval(
        [0.1] * 5, key="constant", statistic=lambda values: float(np.mean(values))
    )
    assert first == second
    assert first["replicate_values"] == [0.1] * 5


def test_undefined_budget_rank_correlation_fails_rank_retention_closed() -> None:
    raw = _raw()
    for row in raw["selector_rows"]:
        if (
            row["candidate_id"] == "FUSED3"
            and row["arm"] == "nonlinearity_full"
            and row["budget"] == full.BUDGETS[0]
        ):
            row["score"] = 1.0
    with pytest.warns(Warning, match="constant"):
        result = analysis.analyze(raw)
    failed = [
        row
        for row in result["gates"]
        if row["gate"] == "nonlinear_rank_retention"
    ]
    assert len(failed) == 3
    assert all(row["status"] == "fail" and row["estimate"] is None for row in failed)
    assert result["decision"]["status"] == "fail_locked_candidate"


def test_refined_archive_diagnostic_never_reselects() -> None:
    raw = copy.deepcopy(_raw())
    raw["refined_historical_deltas"][0]["delta"] = 0.25
    result = analysis.analyze(raw)
    assert result["refined_score_parity_diagnostic"]["max_abs_delta"] == 0.25
    assert result["decision"]["selected_candidate"] == "FUSED3"
