import copy

import numpy as np

from experiments.continuous_real_reliability_gate import run as gate_parent
from experiments.continuous_real_reliability_gate_expanded import run


def test_expanded_grid_uses_four_groups_and_common_roster():
    protocol = run._load_protocol()
    panels = run.load_panels(protocol)
    assert len(panels) == 25
    assert {panel["dataset_group"] for panel in panels} == set(run.GROUPS)
    assert {panel["family"] for panel in panels} == set(protocol["target_families"])
    assert all(tuple(row["candidate_id"] for row in panel["candidates"]) == run.BACKBONES for panel in panels)


def test_expanded_gate_inference_features_remain_outcome_free():
    panels = run.load_panels(run._load_protocol())
    baseline = [gate_parent.feature_mapping(panel) for panel in panels]
    mutated = copy.deepcopy(panels)
    for panel in mutated:
        for candidate in panel["candidates"]:
            candidate["best_head_envelope_r2"] += 1000.0
            for head in candidate["heads"].values():
                head["test_r2"] -= 1000.0
    assert [gate_parent.feature_mapping(panel) for panel in mutated] == baseline


def test_lodo_models_exclude_the_held_out_group():
    panels = run.load_panels(run._load_protocol())
    fixed = [run.panel_metric(panel, method) for panel in panels for method in ("broad_rff", "bounded_2k", "pool_borda_a050", "linear_probe")]
    models = run.fit_lodo_models(panels, fixed)
    assert set(models) == set(run.GROUPS)
    for group, model in models.items():
        expected = sum(panel["dataset_group"] != group for panel in panels)
        assert model["training_rows"] == expected
        assert model["feature_names"] == list(run.FEATURE_NAMES)
        assert np.isfinite([model["intercept"], *model["coefficients"]]).all()

