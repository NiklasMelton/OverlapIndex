"""Focused tests for the frozen large-panel cheap head-subset test."""

from __future__ import annotations

import numpy as np

from experiments.continuous_large_head_subset_generalization import run


def test_protocol_locks_primary_subset_gate_and_no_fit_scope():
    protocol = run.load_protocol()

    assert tuple(protocol["ablation"]["primary_subset"]) == run.PRIMARY
    assert tuple(protocol["ablation"]["accuracy_control_subset"]) == run.ACCURACY_CONTROL
    assert protocol["success"]["applies_only_to"] == run.PRIMARY_ID
    assert protocol["success"]["maximum_aggregate_runtime_ratio_to_linear_cv"] == 3.0
    assert protocol["scope"]["post_outcome_subset_selection_allowed"] is False
    assert protocol["expected_grid"]["new_model_fit_count"] == 0
    assert protocol["expected_grid"]["new_overlapindex_fit_count"] == 0


def test_all_fifteen_subsets_have_stable_canonical_ids():
    assert len(run.SUBSETS) == 15
    assert len(set(run.SUBSET_IDS)) == 15
    assert run.SUBSET_IDS[:4] == run.HEADS
    assert run.PRIMARY_ID == "quadratic_ridge+knn"
    assert run.FULL_ID == "linear_ridge+quadratic_ridge+knn+rbf_svr"


def test_source_chain_and_selected_panel_grid_are_exact():
    protocol = run.load_protocol()
    raw, selection, basis = run.validate_sources(protocol)
    panels = run.selected_panels(raw, protocol)

    assert len(panels) == 24
    assert sum(len(panel["candidates"]) for panel in panels) == 120
    assert selection["reserved_backbones_evaluated"] == []
    assert basis["diagnosis"]["status"] == "rbf_required_but_head_set_reducible"
    assert {panel["stratum"] for panel in panels} == set(protocol["strata"])


def test_subset_score_is_candidatewise_maximum_of_only_included_heads():
    panel = {
        "candidates": [
            {
                "inner_cv_scores": {
                    "linear_ridge": 0.8,
                    "quadratic_ridge": 0.2,
                    "knn": 0.7,
                    "rbf_svr": 0.9,
                }
            },
            {
                "inner_cv_scores": {
                    "linear_ridge": 0.1,
                    "quadratic_ridge": 0.6,
                    "knn": 0.4,
                    "rbf_svr": 0.3,
                }
            },
        ]
    }

    assert run.subset_scores(panel, run.PRIMARY) == [0.7, 0.6]
    assert run.subset_scores(panel, run.FULL) == [0.9, 0.6]


def _row(group: str, identifier: str, rho: float, top1: float, top3: float):
    return {
        "group": group,
        "subset_id": identifier,
        "mean_spearman": rho,
        "mean_top1_regret_r2": top1,
        "mean_top3_opportunity_regret_r2": top3,
    }


def test_decision_applies_every_check_only_to_preselected_primary():
    protocol = run.load_protocol()
    summaries = []
    for group in run.GROUPS:
        summaries.extend(
            [
                _row(group, run.BASELINE_ID, 0.50, 0.030, 0.005),
                _row(group, run.PRIMARY_ID, 0.57, 0.020, 0.006),
            ]
        )
    costs = [{"subset_id": run.PRIMARY_ID, "relative_to_linear_cv": 2.5}]

    result = run.decision(summaries, costs, protocol)

    assert result["passed"] is True
    assert result["primary_subset_id"] == run.PRIMARY_ID
    assert all(result["checks"].values())
    assert result["promotion"] is None


def test_normalization_rejects_changed_heldout_envelope():
    candidate = {
        "candidate_id": "x",
        "best_head_envelope_r2": 2.0,
        "heads": {
            head: {"inner_best_r2": 0.1, "test_r2": 0.2, "wall_seconds": 0.01}
            for head in run.HEADS
        },
    }
    panel = {
        "dataset_id": "d",
        "target_id": "t",
        "stratum": run.GROUPS[1],
        "seed": 1,
        "candidates": [candidate] * 5,
    }

    try:
        run.normalized_panels([panel] * 24)
    except ValueError as error:
        assert "candidate record" in str(error)
    else:  # pragma: no cover
        raise AssertionError("changed held-out envelope was accepted")


def test_protocol_hash_sidecar_matches_exact_bytes():
    assert run.sha256_path(run.PROTOCOL_PATH) == run.PROTOCOL_SHA256_PATH.read_text(
        encoding="utf-8"
    ).strip()
