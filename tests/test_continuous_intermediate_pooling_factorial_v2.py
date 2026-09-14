from __future__ import annotations

import numpy as np

from experiments.continuous_intermediate_pooling_factorial_v2 import extract, run


def test_protocol_grid_dimensions_and_ancestry() -> None:
    protocol = extract.load_protocol()
    assert len(protocol["backbones"]) == 5
    assert len(protocol["views"]) == 6
    assert protocol["expected_grid"]["candidate_cell_count"] == 300
    assert set(extract._source_hashes(protocol)) == {
        "early_mean_protocol",
        "early_mean_embedding_manifest",
        "early_mean_reference_raw",
        "early_mean_reference_summary",
        "early_mean_reference_terminal",
        "parent_intermediate_embedding_manifest",
        "prepared_cohort",
        "prepared_manifest",
        "model_spec_source",
    }
    dimensions = {
        row["backbone_id"]: extract._expected_dimensions(protocol, row)
        for row in protocol["backbones"]
    }
    assert dimensions["dinov2-small"]["late_mean_std"] == 768
    assert dimensions["deit-tiny"]["middle_mean"] == 192
    assert dimensions["mobilenetv3-large"]["late_mean_std"] == 1920
    assert dimensions["resnet50"]["late_mean_std"] == 4096
    assert dimensions["convnext-tiny"]["middle_mean_std"] == 768


def test_summary_nominates_only_a_view_passing_both_targets() -> None:
    protocol = run.load_protocol()
    backbones = [row["backbone_id"] for row in protocol["backbones"]]
    cells = []
    for target in protocol["targets"]:
        for seed_position, draw_seed in enumerate(target["draw_seeds"]):
            for view in protocol["views"]:
                conflict = view == "middle_mean_std" and seed_position < 3
                linear = np.linspace(0.1, 0.5, len(backbones))
                envelope = linear[::-1] + 0.1 if conflict else linear + 0.03
                for position, backbone in enumerate(backbones):
                    cell_envelope = float(max(linear[position], envelope[position]))
                    cells.append(
                        {
                            "cell_id": f"{target['target_key']}::{draw_seed}::{backbone}::{view}",
                            "target_key": target["target_key"],
                            "draw_seed": draw_seed,
                            "candidate_id": backbone,
                            "view_id": view,
                            "head_evaluation_r2": {
                                "linear_ridge": float(linear[position]),
                                "rbf_svr": float(envelope[position]),
                            },
                            "envelope_r2": cell_envelope,
                            "best_head_ids": ["rbf_svr"],
                            "heads": {
                                "linear_ridge": {"final_warning_types": []},
                                "rbf_svr": {"final_warning_types": []},
                            },
                        }
                    )
    summary = run.summarize(protocol, cells)
    assert len(summary["panel_summaries"]) == 60
    assert summary["decision"]["cross_target_nominated_views"] == ["middle_mean_std"]
    assert summary["decision"]["same_outcome_oi_authorized"] is False
    assert len(summary["uplift_stability"]) == 60


def test_factorial_effects_preserve_target_depth_and_pooling_identity() -> None:
    rows = []
    for target in ("a", "b"):
        for depth_position, depth in enumerate(("early", "middle", "late")):
            for pooling_position, pooling in enumerate(("mean", "mean_std")):
                rows.append(
                    {
                        "target_key": target,
                        "view_id": f"{depth}_{pooling}",
                        "depth": depth,
                        "pooling": pooling,
                        "mean_linear_to_envelope_spearman": 0.5 + 0.1 * pooling_position,
                        "mean_linear_top1_regret_r2": 0.03 - 0.01 * pooling_position,
                        "mean_maximum_nonlinear_gain_r2": 0.04 + 0.01 * depth_position,
                        "material_rank_conflict_draw_count": 3 - pooling_position,
                    }
                )
    effects = run._factorial_effects(rows)
    assert len(effects["pooling_effects_within_depth"]) == 6
    assert len(effects["depth_profiles_within_pooling"]) == 4
    first = effects["pooling_effects_within_depth"][0]
    assert np.isclose(first["mean_std_minus_mean_spearman"], 0.1)
    assert np.isclose(first["mean_std_minus_mean_top1_regret_r2"], -0.01)


def test_parent_rows_are_reused_instead_of_weakening_parity(monkeypatch, tmp_path) -> None:
    dataset, backbone = "d", "convnext-tiny"
    parent_root = tmp_path / "parent"
    early_root = tmp_path / "early"
    (parent_root / dataset / backbone).mkdir(parents=True)
    (early_root / dataset).mkdir(parents=True)
    views = extract.load_protocol()["views"]
    parent = np.ones((2, 2), dtype=np.float32)
    for view in views:
        np.save(parent_root / dataset / backbone / f"{view}.npy", parent)
    early_values = np.zeros((4, 2), dtype=np.float32)
    early_values[[0, 3]] = 1.0
    np.save(early_root / dataset / f"{backbone}.npy", early_values)
    monkeypatch.setattr(extract, "PARENT_ROOT", parent_root)
    monkeypatch.setattr(extract, "EARLY_ROOT", early_root)
    rows = [{"matrix_index": 0}, {"matrix_index": 3}]
    monkeypatch.setattr(
        extract.information,
        "_target_rows",
        lambda cohort, target: (dataset, rows[:1], rows[:1], rows[1:]),
    )
    matrices = {view: np.zeros((4, 2), dtype=np.float32) for view in views}
    result = extract._parity(
        extract.load_protocol(), {},
        {"target_key": f"{dataset}::t", "draw_seeds": [1], "training_rows": 1},
        backbone, matrices,
    )
    assert result["parent_selected_rows_reused"] is True
    assert result["parent_selected_rows_maximum_absolute_delta_before_reuse_by_view"]["middle_mean_std"] == 1.0
    assert np.array_equal(matrices["middle_mean_std"][[0, 3]], parent)
