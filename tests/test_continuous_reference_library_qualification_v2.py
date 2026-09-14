import copy

from experiments.continuous_reference_library_qualification_v2 import run


def test_protocol_is_frozen_and_widens_failed_ranges():
    protocol = run.load_protocol()
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["reserved_backbones_evaluated"] is False
    assert min(protocol["heads"]["rbf_svr"]["C_standardized_target_units"]) == 0.003
    assert min(protocol["heads"]["rbf_svr"]["bandwidth_factors"]) == 0.0625
    assert max(protocol["heads"]["rbf_svr"]["epsilon_standardized_target_units"]) == 0.5
    assert max(protocol["heads"]["quadratic_ridge"]["mean_loss_lambda"]) == 10000.0
    assert max(protocol["heads"]["knn"]["n_neighbors"]) == 472


def test_material_endpoint_classification_is_fail_closed_except_predeclared_cases():
    protocol = run.load_protocol()
    common = {
        "domain": "real",
        "panel": "target",
        "candidate": "dinov2-small",
        "head": "rbf_svr",
        "parameter": "epsilon_standardized",
        "endpoint": "maximum",
        "selected_value": 0.5,
        "inner_cv_advantage_r2": 0.02,
        "material": True,
    }
    null = dict(common, domain="synthetic", candidate="noise_only")
    structural = dict(
        common,
        head="knn",
        parameter="n_neighbors",
        selected_value=472.0,
    )
    actionable, diagnostic = run.classify_material_endpoints(
        [common, null, structural], protocol
    )
    assert actionable == [common]
    assert [row["non_actionable_reason"] for row in diagnostic] == [
        "predeclared_known_null_candidate",
        "predeclared_structural_endpoint",
    ]


def test_nonmaximum_knn_and_nonnull_synthetic_endpoints_remain_actionable():
    protocol = run.load_protocol()
    rows = [
        {
            "domain": "synthetic",
            "panel": "smooth/1",
            "candidate": "degraded_latent",
            "head": "knn",
            "parameter": "n_neighbors",
            "endpoint": "maximum",
            "selected_value": 255.0,
            "inner_cv_advantage_r2": 0.02,
            "material": True,
        },
        {
            "domain": "synthetic",
            "panel": "smooth/1",
            "candidate": "degraded_latent",
            "head": "rbf_svr",
            "parameter": "C_standardized",
            "endpoint": "minimum",
            "selected_value": 0.003,
            "inner_cv_advantage_r2": 0.02,
            "material": True,
        },
    ]
    actionable, diagnostic = run.classify_material_endpoints(rows, protocol)
    assert actionable == rows
    assert diagnostic == []


def test_summary_recomputes_gate_after_endpoint_classification(monkeypatch):
    protocol = run.load_protocol()
    base_summary = {
        "status": "old",
        "real_rows": [],
        "synthetic_rows": [],
        "synthetic_family_rows": [],
        "endpoint_selection_count": 2,
        "material_endpoint_selection_count": 1,
        "material_endpoint_rows": [
            {
                "domain": "synthetic",
                "panel": "linear/1",
                "candidate": "noise_only",
                "head": "knn",
                "parameter": "n_neighbors",
                "endpoint": "maximum",
                "selected_value": 472.0,
                "inner_cv_advantage_r2": 0.02,
                "material": True,
            }
        ],
        "warning_rows": [],
        "decision": {
            "grid_endpoints_adequate": False,
            "synthetic_controls_pass": True,
            "cub_linear_controls_pass": True,
            "lsp_nonlinear_head_gain_pass": True,
            "warning_gate_pass": True,
            "qualification_pass": False,
            "next_stage": "blocked_new_protocol_required",
            "method_promotion": "none_reference_only",
        },
    }
    monkeypatch.setattr(run.v1, "summarize", lambda *_: copy.deepcopy(base_summary))
    summary = run.summarize(protocol, [], [])
    assert summary["actionable_material_endpoint_selection_count"] == 0
    assert summary["non_actionable_material_endpoint_selection_count"] == 1
    assert summary["decision"]["qualification_pass"] is True
    assert summary["decision"]["next_stage"] == "eight_target_historical_screen"


def test_expected_grid_matches_full_rerun():
    protocol = run.load_protocol()
    assert protocol["expected_grid"] == {
        "real_target_count": 3,
        "real_candidate_draw_count": 25,
        "synthetic_panel_count": 9,
        "synthetic_candidate_count": 45,
        "overlapindex_fit_count": 0,
        "reserved_backbone_count": 0,
    }

