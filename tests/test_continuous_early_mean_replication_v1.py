from __future__ import annotations

import numpy as np

from experiments.continuous_early_mean_replication_v1 import extract, oi, reference
from experiments.continuous_real_source_qualification_v1 import prepare


def test_frozen_protocol_and_prepared_row_contract() -> None:
    protocol = extract.load_protocol()
    assert len(protocol["backbones"]) == 5
    assert all(len(target["draw_seeds"]) == 5 for target in protocol["targets"])
    assert protocol["representation"]["view_id"] == "early_mean"
    cohort = prepare.load_cohort()
    for target in protocol["targets"]:
        dataset_id, rows = extract._rows(cohort, target)
        assert dataset_id == target["target_key"].split("::", 1)[0]
        assert len(rows) == 3000
        assert [row["split"] for row in rows[:2000]] == ["training"] * 2000
        assert [row["split"] for row in rows[2000:]] == ["evaluation"] * 1000


def test_reference_gate_requires_replication_on_both_targets() -> None:
    protocol = reference.load_protocol()
    backbones = [row["backbone_id"] for row in protocol["backbones"]]
    cells = []
    for target in protocol["targets"]:
        for seed_position, draw_seed in enumerate(target["draw_seeds"]):
            conflict = seed_position < 3
            linear = np.linspace(0.1, 0.5, len(backbones))
            envelope = linear[::-1] + 0.1 if conflict else linear + 0.03
            for position, backbone in enumerate(backbones):
                cells.append(
                    {
                        "cell_id": f"{target['target_key']}::{draw_seed}::{backbone}",
                        "target_key": target["target_key"],
                        "draw_seed": draw_seed,
                        "candidate_id": backbone,
                        "head_evaluation_r2": {
                            "linear_ridge": float(linear[position]),
                            "rbf_svr": float(envelope[position]),
                        },
                        "envelope_r2": float(max(linear[position], envelope[position])),
                        "best_head_ids": ["rbf_svr"],
                        "heads": {
                            "linear_ridge": {"final_warning_types": []},
                            "rbf_svr": {"final_warning_types": []},
                        },
                    }
                )
    summary = reference.summarize(protocol, cells)
    assert summary["decision"]["oi_stage_authorized"] is True
    assert [row["material_rank_conflict_draw_count"] for row in summary["target_summaries"]] == [3, 3]
    cells = [row for row in cells if not (row["target_key"] == protocol["targets"][1]["target_key"] and row["draw_seed"] == protocol["targets"][1]["draw_seeds"][2])]
    replacement_seed = protocol["targets"][1]["draw_seeds"][2]
    for position, backbone in enumerate(backbones):
        score = 0.1 + 0.1 * position
        cells.append(
            {
                "cell_id": f"{protocol['targets'][1]['target_key']}::{replacement_seed}::{backbone}",
                "target_key": protocol["targets"][1]["target_key"],
                "draw_seed": replacement_seed,
                "candidate_id": backbone,
                "head_evaluation_r2": {"linear_ridge": score, "rbf_svr": score + 0.03},
                "envelope_r2": score + 0.03,
                "best_head_ids": ["rbf_svr"],
                "heads": {
                    "linear_ridge": {"final_warning_types": []},
                    "rbf_svr": {"final_warning_types": []},
                },
            }
        )
    assert reference.summarize(protocol, cells)["decision"]["oi_stage_authorized"] is False


def test_ranking_metrics_separate_ordering_and_selection() -> None:
    ids = ["a", "b", "c", "d", "e"]
    truth = [0.9, 0.8, 0.7, 0.6, 0.5]
    scores = [1.0, 0.1, 0.2, 0.3, 0.4]
    row = oi.ranking_metrics(ids, truth, scores)
    assert row["exact_best"] is True
    assert row["top1_regret_r2"] == 0.0
    assert row["spearman"] == 0.0
    assert row["top3_set_recall"] < 1.0


def test_pca32_shared_pair_is_finite_and_deterministic() -> None:
    rng = np.random.default_rng(2026091501)
    X = rng.normal(size=(96, 40)).astype(np.float32)
    y = (X[:, 0] + 0.2 * X[:, 1] ** 2 + rng.normal(scale=0.2, size=len(X))).astype(np.float64)
    first = oi._pca32_shared_pair(X, y, 3119, 0)
    second = oi._pca32_shared_pair(X, y, 3119, 0)
    assert first["fitted_selector_count"] == 4
    assert first["broad_rff"]["score"] == second["broad_rff"]["score"]
    assert first["save_rff_current"]["score"] == second["save_rff_current"]["score"]
    assert np.isfinite(first["broad_rff"]["score"])
    assert np.isfinite(first["save_rff_current"]["score"])
