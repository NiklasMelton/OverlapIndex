import numpy as np
import pytest

from experiments.continuous_coco_pose_opportunity_replication import extract, prepare, run
from experiments.continuous_fresh_structured_opportunity_discovery import run as source


def test_protocol_is_prospective_head_only_replication_with_locked_prior():
    protocol = run.load_protocol()
    assert protocol["scope"]["coco_outcomes_previously_observed"] is False
    assert protocol["scope"]["overlapindex_methods_evaluated"] is False
    assert protocol["prior_evidence"]["admitted_dataset"] == "lsp_2000_pose"
    assert protocol["prior_evidence"]["admitted_target"] == "torso_axis_sine"
    assert tuple(protocol["backbones"]["scout"]) == extract.BACKBONES
    assert protocol["expected_grid"]["overlapindex_fit_count"] == 0


def test_pose_targets_encode_vertical_torso_and_straight_legs():
    points = np.zeros((17, 3), dtype=np.float64)
    points[:, 2] = 2
    points[15], points[13], points[11] = (0, 2, 2), (0, 1, 2), (0, 0, 2)
    points[12], points[14], points[16] = (2, 0, 2), (2, 1, 2), (2, 2, 2)
    points[5], points[6] = (0, -2, 2), (2, -2, 2)
    values = prepare.pose_targets(points)
    assert values["torso_axis_sine"] == pytest.approx(-1.0)
    assert values["torso_axis_cosine"] == pytest.approx(0.0)
    assert values["bilateral_knee_extension_cosine"] == pytest.approx(1.0)


def test_largest_eligible_person_rank_is_area_then_annotation_id():
    image = {"id": 7, "file_name": "000000000007.jpg", "width": 100, "height": 100}
    keypoints = np.zeros((17, 3), dtype=np.float64)
    keypoints[:, 2] = 2
    keypoints[11, :2], keypoints[12, :2] = (0, 0), (2, 0)
    keypoints[5, :2], keypoints[6, :2] = (0, -2), (2, -2)
    keypoints[15, :2], keypoints[13, :2] = (0, 2), (0, 1)
    keypoints[14, :2], keypoints[16, :2] = (2, 1), (2, 2)
    annotation = {
        "id": 10,
        "image_id": 7,
        "category_id": 1,
        "iscrowd": 0,
        "bbox": [0, 0, 20, 30],
        "keypoints": keypoints.reshape(-1).tolist(),
    }
    row = prepare._eligible_record(annotation, "torso", "train", image)
    assert row["bbox_area"] == 600
    assert row["annotation_id"] == 10
    annotation["iscrowd"] = 1
    assert prepare._eligible_record(annotation, "torso", "train", image) is None


def test_real_planned_cohort_is_exact_and_target_disjoint():
    plan = prepare.load_plan()
    assert plan["protocol_sha256"] == prepare.sha256_path(prepare.PROTOCOL_PATH)
    assert len(plan["rows"]) == 2044
    assert len({row["sample_id"] for row in plan["rows"]}) == 2044
    for target, rows in plan["target_cohorts"].items():
        assert len(rows) == 1100
        assert len({row["matrix_index"] for row in rows}) == 1100
        assert sum(row["source_split"] == "val" for row in rows) == 500
        for seed in prepare.REPLICATE_SEEDS:
            assert sum(row["replicate_seed"] == seed for row in rows) == 200
    assert plan["target_cohorts"]["torso_axis_sine"] != []
    assert [row["matrix_index"] for row in plan["target_cohorts"]["torso_axis_sine"]] == [
        row["matrix_index"] for row in plan["target_cohorts"]["torso_axis_cosine"]
    ]


def _candidate(position):
    linear = 0.2 + 0.1 * position
    values = {
        "linear_ridge": linear,
        "quadratic_ridge": linear - 0.01,
        "knn": linear - 0.02,
        "rbf_svr": 0.8 if position == 0 else linear - 0.03,
    }
    envelope = max(values.values())
    return {
        "candidate_id": extract.BACKBONES[position],
        "heads": {
            head: {"test_r2": value, "inner_best_r2": value - 0.01}
            for head, value in values.items()
        },
        "heldout_linear_r2": linear,
        "best_head_envelope_r2": envelope,
        "best_head_ids": [head for head, value in values.items() if value == envelope],
        "nonlinear_minus_linear_r2": envelope - linear,
    }


def test_summary_opens_combined_gate_only_for_admitted_coco_target():
    protocol = run.load_protocol()
    panels = []
    for target in prepare.TARGETS:
        for seed in prepare.REPLICATE_SEEDS:
            panels.append(
                {
                    "dataset_id": prepare.DATASET_ID,
                    "target_id": target,
                    "seed": seed,
                    "candidates": [_candidate(position) for position in range(5)],
                }
            )
    summary = run.summarize(panels, protocol)
    assert summary["dataset_gate_passed"] is True
    assert summary["combined_two_dataset_gate_passed"] is True
    assert summary["reserved_backbones_evaluated"] == []
    assert summary["overlapindex_fit_count"] == 0
    assert set(summary["admitted_targets"]) == set(prepare.TARGETS)


def test_runner_uses_exact_target_values_and_matrix_indices():
    cohort = {
        "target_cohorts": {
            "x": [
                *[
                    {
                        "source_split": "train",
                        "replicate_seed": seed,
                        "matrix_index": seed % 100 + index,
                        "target": float(index),
                    }
                    for seed in prepare.REPLICATE_SEEDS
                    for index in range(prepare.TRAIN_ROWS)
                ],
                *[
                    {
                        "source_split": "val",
                        "replicate_seed": None,
                        "matrix_index": 1000 + index,
                        "target": float(index),
                    }
                    for index in range(prepare.EVALUATION_ROWS)
                ],
            ]
        }
    }
    train, evaluation, y_train, y_evaluation = run._indices(cohort, "x", 4103)
    assert (len(train), len(evaluation)) == (200, 500)
    assert np.array_equal(y_train, np.arange(200, dtype=float))
    assert np.array_equal(y_evaluation, np.arange(500, dtype=float))
    assert tuple(source.HEADS) == ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
