import numpy as np
import pytest

from experiments.continuous_large_structured_opportunity_benchmark import extract, prepare, run
from experiments.continuous_structured_head_discovery import run as parent


def test_protocol_freezes_twenty_targets_and_head_only_scope():
    protocol = run.load_protocol()
    assert protocol["scope"]["previous_target_outcomes_included"] is False
    assert protocol["scope"]["overlapindex_methods_evaluated"] is False
    assert protocol["scope"]["post_outcome_target_replacement_allowed"] is False
    assert tuple(protocol["datasets"]) == prepare.DATASETS
    assert sum(len(row["targets"]) for row in protocol["datasets"].values()) == 20
    assert protocol["expected_grid"]["panel_count"] == 60
    assert protocol["expected_grid"]["candidate_rows"] == 300
    assert protocol["expected_grid"]["overlapindex_fit_count"] == 0
    assert tuple(protocol["backbones"]["scout"]) == extract.BACKBONES
    assert tuple(protocol["backbones"]["reserved"]) == parent.RESERVED_BACKBONES


def test_target_stratification_is_exact_disjoint_and_deterministic():
    records = []
    for index in range(1000):
        records.append(
            {
                "sample_id": f"x/{index:04d}",
                "source_index": index,
                "source_split": "training" if index < 600 else "evaluation",
                "targets": {"target": float(index)},
            }
        )
    first = prepare._select_target("dataset", "target", records)
    second = prepare._select_target("dataset", "target", records)
    assert first == second
    assert len(first) == 500
    assert len({row["sample_id"] for row in first}) == 500
    assert sum(row["split"] == "evaluation" for row in first) == 200
    for seed in prepare.REPLICATE_SEEDS:
        rows = [row for row in first if row["replicate_seed"] == seed]
        assert len(rows) == 100
        assert {int(row["target"] // 60) for row in rows} == set(range(10))


def test_pose_targets_encode_straight_arms_and_normalized_spans():
    points = np.zeros((17, 2), dtype=np.float64)
    flags = np.full(17, 2, dtype=np.int64)
    points[5], points[6] = (-1, 0), (1, 0)
    points[11], points[12] = (-1, 2), (1, 2)
    points[7], points[8] = (-1, 1), (1, 1)
    points[9], points[10] = (-1, 2), (1, 2)
    points[15], points[16] = (-2, 4), (2, 4)
    values = prepare.pose_targets(points, flags, visible_code=2)
    assert values == pytest.approx(
        {
            "bilateral_elbow_extension_cosine": 1.0,
            "wrist_span_over_torso_length": 1.0,
            "ankle_span_over_torso_length": 2.0,
            "shoulder_to_hip_width_ratio": 1.0,
        }
    )
    flags[9] = 1
    values = prepare.pose_targets(points, flags, visible_code=2)
    assert "bilateral_elbow_extension_cosine" not in values
    assert "wrist_span_over_torso_length" not in values


def test_mask_targets_use_four_neighbor_perimeter():
    mask = np.full((6, 7), 2, dtype=np.uint8)
    mask[1:5, 2:5] = 1
    values = prepare.mask_targets(mask)
    # A 4x3 rectangle has area 12, perimeter 14, and tight-box fill 1.
    assert values["foreground_compactness"] == pytest.approx(4 * np.pi * 12 / 14**2)
    assert values["foreground_bbox_fill"] == pytest.approx(1.0)
    assert values["foreground_boundary_roughness"] == pytest.approx(1.0)
    assert 0.0 <= values["foreground_eccentricity"] <= 1.0


def test_voc_largest_box_excludes_difficult_and_uses_inclusive_coordinates():
    xml = b"""<annotation><size><width>100</width><height>80</height></size>
    <object><name>hard</name><difficult>1</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>100</xmax><ymax>80</ymax></bndbox></object>
    <object><name>cat</name><difficult>0</difficult><bndbox><xmin>11</xmin><ymin>21</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
    <object><name>dog</name><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>
    </annotation>"""
    row = prepare._voc_record(xml, "training", "000001")
    assert row["selected_object_class"] == "cat"
    assert row["targets"]["largest_box_area_fraction"] == pytest.approx(400 / 8000)
    assert row["targets"]["largest_box_log_aspect_ratio"] == pytest.approx(0.0)


def test_panel_data_reads_only_the_requested_seed_and_frozen_evaluation():
    target = "x"
    cohort = []
    for seed in prepare.REPLICATE_SEEDS:
        cohort.extend(
            {
                "split": "training",
                "replicate_seed": seed,
                "matrix_index": 1000 * (seed - 4100) + index,
                "target": float(index),
            }
            for index in range(prepare.TRAIN_ROWS)
        )
    cohort.extend(
        {
            "split": "evaluation",
            "replicate_seed": None,
            "matrix_index": 9000 + index,
            "target": float(index),
        }
        for index in range(prepare.EVALUATION_ROWS)
    )
    train, evaluation, y_train, y_eval = run._panel_data(
        {"target_cohorts": {target: cohort}}, target, 4103
    )
    assert (len(train), len(evaluation)) == (100, 200)
    assert np.array_equal(y_train, np.arange(100, dtype=float))
    assert np.array_equal(y_eval, np.arange(200, dtype=float))


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


def test_summary_reports_prevalence_without_claiming_oi_validation():
    protocol = run.load_protocol()
    panels = []
    for dataset_id, descriptor in protocol["datasets"].items():
        for target_id in descriptor["targets"]:
            for seed in prepare.REPLICATE_SEEDS:
                panels.append(
                    {
                        "dataset_id": dataset_id,
                        "target_id": target_id,
                        "seed": seed,
                        "candidates": [_candidate(position) for position in range(5)],
                    }
                )
    summary = run.summarize(panels, protocol)
    assert summary["benchmark_complete"] is True
    assert summary["prevalence"]["target_count"] == 20
    assert summary["prevalence"]["dataset_count"] == 5
    assert summary["reserved_backbones_evaluated"] == []
    assert summary["overlapindex_fit_count"] == 0
    low, high = summary["prevalence"]["admitted_target_fraction_wilson95"]
    assert 0.0 <= low <= high <= 1.0
