import copy

import numpy as np

from experiments.continuous_real_reliability_gate import run as gate_parent
from experiments.continuous_real_reliability_gate_expanded import run as parent
from experiments.continuous_real_winner_preserved_gate import run
from experiments.continuous_real_bbox_panels import run as bbox_parent


def test_winner_preserved_groups_keep_broad_first_group_exactly():
    panels = parent.load_panels(parent._load_protocol())
    for panel in panels:
        broad = bbox_parent.ranking_groups(panel, "broad_rff")
        alternative = run.winner_preserved_borda_groups(panel)
        assert alternative[0] == broad[0]
        assert sorted(index for group in alternative for index in group) == list(range(10))
        broad_metric = run.panel_metric(panel, "broad_rff")
        alternative_metric = run.panel_metric(panel, "winner_preserved_borda_a050")
        assert alternative_metric["top_1_opportunity_regret_r2"] == broad_metric["top_1_opportunity_regret_r2"]
        assert alternative_metric["top_1_exact_best_coverage"] == broad_metric["top_1_exact_best_coverage"]


def test_gate_features_remain_outcome_free():
    panels = parent.load_panels(parent._load_protocol())
    baseline = [gate_parent.feature_mapping(panel) for panel in panels]
    mutated = copy.deepcopy(panels)
    for panel in mutated:
        for candidate in panel["candidates"]:
            candidate["best_head_envelope_r2"] += 99.0
            for head in candidate["heads"].values():
                head["test_r2"] -= 99.0
    assert [gate_parent.feature_mapping(panel) for panel in mutated] == baseline


def test_lodo_models_use_winner_preserved_training_target():
    panels = parent.load_panels(parent._load_protocol())
    fixed = [
        run.panel_metric(panel, method)
        for panel in panels
        for method in (
            "broad_rff",
            "bounded_2k",
            "winner_preserved_borda_a050",
            "pool_borda_a050",
            "linear_probe",
        )
    ]
    models = run.fit_lodo_models(panels, fixed)
    assert set(models) == set(parent.GROUPS)
    for group, model in models.items():
        assert model["training_target"].startswith("broad_minus_winner_preserved")
        assert model["training_rows"] == sum(panel["dataset_group"] != group for panel in panels)
        assert np.isfinite([model["intercept"], *model["coefficients"]]).all()

