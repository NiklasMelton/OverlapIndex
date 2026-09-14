from experiments.continuous_reference_library_qualification_v3 import run


def _endpoint(**overrides):
    row = {
        "domain": "real",
        "panel": "target",
        "candidate": "dinov2-small",
        "draw_seed": 1,
        "head": "rbf_svr",
        "head_inner_best_r2": 0.10,
        "parameter": "epsilon_standardized",
        "endpoint": "maximum",
        "selected_value": 2.0,
        "inner_cv_advantage_r2": 0.02,
        "material": True,
    }
    row.update(overrides)
    return row


def test_protocol_is_frozen_with_high_epsilon_and_learnability_gate():
    protocol = run.load_protocol()
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["reserved_backbones_evaluated"] is False
    assert max(protocol["heads"]["rbf_svr"]["epsilon_standardized_target_units"]) == 2.0
    assert protocol["grid_adequacy"]["minimum_actionable_inner_cv_r2"] == 0.05


def test_learnable_real_endpoint_remains_actionable():
    protocol = run.load_protocol()
    row = _endpoint(head_inner_best_r2=0.0501)
    actionable, diagnostic = run.classify_material_endpoints([row], protocol)
    assert actionable == [row]
    assert diagnostic == []


def test_unlearnable_real_endpoint_is_reported_diagnostic_only():
    protocol = run.load_protocol()
    row = _endpoint(head_inner_best_r2=0.0499)
    actionable, diagnostic = run.classify_material_endpoints([row], protocol)
    assert actionable == []
    assert diagnostic[0]["non_actionable_reason"] == "predeclared_unlearnable_head_cell"


def test_intended_clean_synthetic_head_is_always_actionable():
    protocol = run.load_protocol()
    row = _endpoint(
        domain="synthetic",
        panel="smooth_periodic/1",
        family="smooth_periodic",
        candidate="clean_latent",
        seed=1,
        head="rbf_svr",
        head_inner_best_r2=-1.0,
    )
    actionable, diagnostic = run.classify_material_endpoints([row], protocol)
    assert actionable == [row]
    assert diagnostic == []


def test_known_null_and_structural_maximum_are_diagnostic_only():
    protocol = run.load_protocol()
    null = _endpoint(
        domain="synthetic", panel="linear/1", family="linear",
        candidate="noise_only", seed=1,
    )
    structural = _endpoint(
        head="knn", parameter="n_neighbors", endpoint="maximum", selected_value=472.0,
    )
    actionable, diagnostic = run.classify_material_endpoints([null, structural], protocol)
    assert actionable == []
    assert [row["non_actionable_reason"] for row in diagnostic] == [
        "predeclared_known_null_candidate", "predeclared_structural_endpoint"
    ]


def test_collect_material_endpoints_retains_inner_score_and_draw_identity():
    endpoint = {
        "parameter": "epsilon_standardized", "endpoint": "maximum",
        "selected_value": 2.0, "inner_cv_advantage_r2": 0.02, "material": True,
    }
    real = [{
        "target_key": "target", "candidates": [{
            "candidate_id": "model", "draws": [{
                "draw_seed": 17, "heads": {"rbf_svr": {
                    "inner_best_r2": 0.2, "endpoint_diagnostics": [endpoint]
                }}
            }]
        }]
    }]
    rows = run.collect_material_endpoints(real, [])
    assert rows[0]["draw_seed"] == 17
    assert rows[0]["head_inner_best_r2"] == 0.2


def test_expected_grid_is_complete_rerun():
    protocol = run.load_protocol()
    assert protocol["expected_grid"]["real_candidate_draw_count"] == 25
    assert protocol["expected_grid"]["synthetic_candidate_count"] == 45
    assert protocol["expected_grid"]["overlapindex_fit_count"] == 0

