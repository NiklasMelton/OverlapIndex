from __future__ import annotations

from experiments.scalable_relevance_kmeans import fused_food_analysis_erratum as erratum


def test_erratum_adds_only_lp_fold_timing_aliases() -> None:
    raw = {
        "selector_rows": [
            {
                "candidate_id": "LP-FULL",
                "score": 0.7,
                "folds": [
                    {
                        "score": 0.6,
                        "predict_wall_seconds": 1.25,
                        "predict_cpu_seconds": 1.0,
                    }
                ],
            },
            {
                "candidate_id": "FUSED3",
                "score": 0.8,
                "folds": [{"score_fixed_wall_seconds": 0.2}],
            },
        ]
    }
    corrected = erratum._with_lp_timing_aliases(raw)
    assert corrected["selector_rows"][0]["score"] == 0.7
    assert corrected["selector_rows"][0]["folds"][0]["score"] == 0.6
    assert corrected["selector_rows"][0]["folds"][0]["score_fixed_wall_seconds"] == 1.25
    assert corrected["selector_rows"][0]["folds"][0]["score_fixed_cpu_seconds"] == 1.0
    assert corrected["selector_rows"][1] == raw["selector_rows"][1]
    assert "score_fixed_wall_seconds" not in raw["selector_rows"][0]["folds"][0]
