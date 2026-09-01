from __future__ import annotations

import copy

from experiments.continuous_real_rank_aggregation import run


def test_protocol_and_source_artifacts_are_exact() -> None:
    protocol = run._load_protocol()
    panels = run.load_panels(protocol)
    assert len(panels) == 15
    assert {panel["dataset_group"] for panel in panels} == {"utkface", "oxford_pet"}
    assert tuple(protocol["methods"]) == run.METHODS


def test_pool_borda_extreme_alpha_recovers_broad_order() -> None:
    panel = run.load_panels(run._load_protocol())[0]
    assert run.pool_borda_groups(panel, 1.0) == run.winner_parent.ranking_groups(
        panel, "broad_rff"
    )


def test_pool_borda_is_complete_and_does_not_force_broad_winner() -> None:
    panel = run.load_panels(run._load_protocol())[-5]
    groups = run.pool_borda_groups(panel, 0.0)
    assert sorted(index for group in groups for index in group) == list(range(16))
    broad_winner = run.winner_parent.ranking_groups(panel, "broad_rff")[0]
    assert groups[0] != broad_winner


def test_lodo_selection_does_not_use_held_out_outcomes() -> None:
    panels = run.load_panels(run._load_protocol())
    fixed = [
        run.panel_metric(panel, method)
        for panel in panels
        for method in run.WEIGHT_METHODS.values()
    ]
    baseline = run.select_lodo_weights(panels, fixed)
    changed = copy.deepcopy(fixed)
    for row in changed:
        if row["dataset_group"] == "oxford_pet":
            row["top_5_mean_utility_regret_r2"] += 1000.0
    updated = run.select_lodo_weights(panels, changed)
    assert updated["oxford_pet"] == baseline["oxford_pet"]


def test_summary_has_closed_complete_grid() -> None:
    protocol = run._load_protocol()
    summary = run.summarize(run.load_panels(protocol), protocol)
    assert summary["panel_count"] == 15
    assert len(summary["panel_method_rows"]) == 150
    assert {row["method"] for row in summary["panel_method_rows"]} == set(run.METHODS)
