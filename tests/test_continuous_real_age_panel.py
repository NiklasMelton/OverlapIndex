from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.continuous_real_age_panel import extract, prepare, run


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_closed_roster_match_code() -> None:
    protocol = run._load_protocol()
    assert tuple(protocol["backbones"]) == extract.BACKBONE_IDS
    assert len(extract.BACKBONE_IDS) == 16
    assert len(set(extract.BACKBONE_IDS)) == 16
    assert tuple(protocol["replicate_seeds"]) == prepare.REPLICATE_SEEDS
    assert protocol["continuous_oi"]["shortlist_depth"] == 5


def test_prepared_cohort_is_balanced_disjoint_and_hash_bound() -> None:
    cohort = prepare.load_cohort()
    assert len(cohort["rows"]) == 1350
    ids = [row["sample_id"] for row in cohort["rows"]]
    assert len(ids) == len(set(ids))
    evaluation = [row for row in cohort["rows"] if row["split"] == "evaluation"]
    assert len(evaluation) == 225
    assert {value: sum(row["age_bin"] == value for row in evaluation) for value in range(9)} == {value: 25 for value in range(9)}
    for seed in prepare.REPLICATE_SEEDS:
        selected = [row for row in cohort["rows"] if row["replicate_seed"] == seed]
        assert len(selected) == 225
        assert {value: sum(row["age_bin"] == value for row in selected) for value in range(9)} == {value: 25 for value in range(9)}


def test_all_frozen_model_checkpoints_are_present_and_exact() -> None:
    result = extract.validate_model_inputs()
    assert result["backbone_count"] == 16
    assert {row["backbone_id"] for row in result["records"]} == set(extract.BACKBONE_IDS)


def test_training_only_pca_is_finite_and_deterministic() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(80, 48)).astype(np.float32)
    test = rng.normal(size=(20, 48)).astype(np.float32)
    left_train, left_test, left_diagnostics = run._pca_geometry(train, test, 41)
    right_train, right_test, right_diagnostics = run._pca_geometry(train, test, 41)
    assert left_train.shape == (80, 32)
    assert left_test.shape == (20, 32)
    np.testing.assert_array_equal(left_train, right_train)
    np.testing.assert_array_equal(left_test, right_test)
    assert left_diagnostics == right_diagnostics


def _fake_panel(seed: int) -> dict:
    candidates = []
    for position, backbone in enumerate(extract.BACKBONE_IDS):
        broad = 1.0 - position / 100.0
        structured = 1.0 - abs(position - 5) / 100.0
        envelope = 1.0 - abs(position - 3) / 50.0
        candidates.append(
            {
                "candidate_id": backbone,
                "best_head_envelope_r2": envelope,
                "best_head_ids": ["linear_ridge"],
                "heads": {
                    "linear_ridge": {"test_r2": envelope - 0.01, "wall_seconds": 0.2},
                    "quadratic_ridge": {"test_r2": envelope},
                    "knn": {"test_r2": envelope - 0.02},
                    "rbf_svr": {"test_r2": envelope - 0.03},
                },
                "selectors": {
                    "rff64_mean8_k4": {
                        "kernel_alignment": {
                            "score": broad,
                            "feature_bandwidths": {"4.0": {"score": broad}},
                        },
                        "wall_seconds": 0.3,
                    },
                    "save_rffmean_k4": {
                        "current_common32": {"score": structured},
                        "wall_seconds": 0.0,
                    },
                },
            }
        )
    return {"seed": seed, "family": "utkface_age", "candidate_count": 16, "candidates": candidates}


def test_real_panel_summary_uses_active_bound_and_complete_grid() -> None:
    protocol = run._load_protocol()
    summary = run.summarize([_fake_panel(seed) for seed in prepare.REPLICATE_SEEDS], protocol)
    assert summary["panel_count"] == 5
    assert len(summary["panel_metrics"]) == 20
    active = next(row for row in summary["decision"]["checks"] if row["criterion"] == "top2k_bound_active_in_every_panel")
    assert active["pass"] is True

