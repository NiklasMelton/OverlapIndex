from __future__ import annotations

import numpy as np
import pytest

from experiments.scalable_relevance_kmeans import fused_food_analysis as analysis
from experiments.scalable_relevance_kmeans import fused_food_screen as screen


def _raw(*, loop_wall: float = 0.46, fused_wall: float = 0.50) -> dict:
    selector_rows = []
    reference_rows = []
    for arm in screen.ARMS:
        for model_index, model in enumerate(screen.MODELS):
            for head in analysis.HEADS:
                reference_rows.append(
                    {
                        "backbone": model,
                        "replicate": screen.REPLICATE,
                        "arm": arm,
                        "head": head,
                        "test_accuracy": 0.50 + 0.01 * model_index,
                    }
                )
            for method, wall in (
                ("NOLLOYD", 0.70),
                ("LOOP3", loop_wall),
                ("FUSED3", fused_wall),
                ("LP-FULL", 1.00),
            ):
                selector_rows.append(
                    {
                        "model": model,
                        "replicate": screen.REPLICATE,
                        "arm": arm,
                        "budget": screen.BUDGET,
                        "candidate_id": method,
                        "score": float(model_index),
                        "fit_wall_seconds": wall * 0.8,
                        "score_fixed_wall_seconds": wall * 0.2,
                        "total_wall_seconds": wall,
                    }
                )
    return {"selector_rows": selector_rows, "reference_rows": reference_rows}


def test_analysis_selects_materially_faster_loop_when_both_pass() -> None:
    result = analysis.analyze(_raw(loop_wall=0.46, fused_wall=0.50))
    assert result["decision"]["status"] == "pass_for_full_food_design"
    assert result["decision"]["eligible_candidates"] == ["FUSED3", "LOOP3"]
    assert result["decision"]["selected_candidate"] == "LOOP3"
    assert all(row["status"] == "pass" for row in result["gates"])


def test_analysis_prefers_fused_within_five_percent_tie_band() -> None:
    result = analysis.analyze(_raw(loop_wall=0.48, fused_wall=0.50))
    assert result["decision"]["selected_candidate"] == "FUSED3"


def test_runtime_bootstrap_is_deterministic_and_uses_ten_backbones() -> None:
    first = analysis._bootstrap_runtime([0.5] * 10, key="fixed")
    second = analysis._bootstrap_runtime([0.5] * 10, key="fixed")
    assert first == second
    assert first["estimate"] == 0.5
    assert first["backbone_values"] == [0.5] * 10


def test_rank_retention_gate_fails_closed_on_undefined_correlation() -> None:
    raw = _raw()
    for row in raw["selector_rows"]:
        if row["candidate_id"] == "FUSED3" and row["arm"] == "nonlinearity_full":
            row["score"] = 1.0
    with pytest.warns(Warning, match="constant"):
        result = analysis.analyze(raw)
    failed = [
        row
        for row in result["gates"]
        if row["candidate_id"] == "FUSED3"
        and row["gate"] == "nonlinear_rank_retention"
    ]
    assert len(failed) == 3
    assert all(row["status"] == "fail" and row["estimate"] is None for row in failed)
    assert "FUSED3" not in result["decision"]["eligible_candidates"]


def test_selection_metrics_average_exact_ties() -> None:
    scores = np.asarray([1.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    outcomes = np.asarray([0.9, 0.7, 1.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    result = analysis._selection_metrics(scores, outcomes)
    assert result["selected_backbones"] == list(screen.MODELS[:2])
    assert np.isclose(result["regret_pp"], 20.0)
    assert result["exact_best"] is False
