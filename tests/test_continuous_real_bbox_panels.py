import json

import numpy as np
import pytest

from experiments.continuous_real_bbox_panels import prepare, run


def test_protocol_and_prepared_cohort_are_frozen():
    protocol = run._load_protocol()
    cohort = prepare.load_cohort()
    assert tuple(protocol["backbones"]) == run.BACKBONES
    assert tuple(protocol["replicate_seeds"]) == prepare.REPLICATE_SEEDS
    assert {row["dataset_id"] for row in cohort["datasets"]} == set(prepare.DATASETS)
    for dataset in cohort["datasets"]:
        assert prepare.sha256_payload(dataset["rows"]) == dataset["rows_sha256"]
        evaluation = [row for row in dataset["rows"] if row["split"] == "evaluation"]
        assert len(evaluation) == prepare.EVALUATION_ROWS
        assert {row["target_bin"] for row in evaluation} == set(range(prepare.TARGET_BINS))
        used = {row["sample_id"] for row in evaluation}
        for seed in prepare.REPLICATE_SEEDS:
            training = [
                row for row in dataset["rows"]
                if row["split"] == "training" and row["replicate_seed"] == seed
            ]
            assert len(training) == prepare.TRAIN_ROWS
            assert {row["target_bin"] for row in training} == set(range(prepare.TARGET_BINS))
            assert used.isdisjoint(row["sample_id"] for row in training)
            used.update(row["sample_id"] for row in training)


def test_box_fraction_is_inclusive_and_validated():
    assert prepare.box_fraction(10, 10, 0, 0, 9, 9) == 1.0
    assert prepare.box_fraction(10, 20, 2, 4, 6, 13) == 0.25
    with pytest.raises(ValueError):
        prepare.box_fraction(10, 10, 0, 0, 10, 9)


def test_target_bin_selection_is_deterministic_and_disjoint():
    rows = [
        {"sample_id": f"train/{index:04d}", "source_split": "train", "target": index / 2000}
        for index in range(1200)
    ] + [
        {"sample_id": f"test/{index:04d}", "source_split": "test", "target": index / 400}
        for index in range(400)
    ]
    first = prepare.select_cohorts(rows)
    second = prepare.select_cohorts(list(reversed(rows)))
    assert first == second
    assert len({row["sample_id"] for row in first}) == len(first) == 1200


def test_generic_panel_metric_supports_common_ten_candidate_roster():
    candidates = []
    for index, backbone in enumerate(run.BACKBONES):
        candidates.append(
            {
                "candidate_id": backbone,
                "best_head_envelope_r2": float(index) / 10,
                "heads": {"linear_ridge": {"test_r2": float(index) / 10}},
                "selectors": {
                    "rff64_mean8_k4": {
                        "kernel_alignment": {
                            "feature_bandwidths": {
                                "4.0": {"score": float(index) / 10}
                            }
                        },
                        "wall_seconds": 1.0,
                    },
                    "save_rffmean_k4": {
                        "current_common32": {"score": float(9 - index) / 10},
                        "wall_seconds": 0.0,
                    },
                },
            }
        )
    panel = {
        "seed": 4103,
        "family": "fixture",
        "dataset_group": "fixture",
        "candidates": candidates,
    }
    metric = run.panel_metric(panel, "broad_rff")
    assert metric["spearman"] == pytest.approx(1.0)
    assert metric["top_5_mean_utility_regret_r2"] == pytest.approx(0.0)
    assert np.isfinite(metric["descriptive_panel_runtime_seconds"])
