"""Contract tests for the frozen continuous ranking experiment."""

import numpy as np

from experiments.continuous_fisher_ranking import run


def test_protocol_matches_closed_method_and_candidate_grid():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    assert list(run.method_specs()) == protocol["continuous_oi"]["methods"]
    assert len(protocol["candidate_specs"]) == 8
    assert len(protocol["families"]) == 6
    assert len(protocol["seeds"]) == 3


def test_generator_is_deterministic_and_uses_one_target_per_panel():
    protocol = run._load_protocol()
    first = run.generate_panel(protocol, 101, "quadratic")
    second = run.generate_panel(protocol, 101, "quadratic")

    np.testing.assert_array_equal(first["y_train"], second["y_train"])
    np.testing.assert_array_equal(first["y_test"], second["y_test"])
    assert len(first["candidates"]) == 8
    for left, right in zip(first["candidates"], second["candidates"]):
        assert left["candidate_id"] == right["candidate_id"]
        assert left["X_train"].shape == (320, 24)
        assert left["X_test"].shape == (320, 24)
        np.testing.assert_array_equal(left["X_train"], right["X_train"])
        np.testing.assert_array_equal(left["X_test"], right["X_test"])


def test_selector_factorial_changes_only_declared_geometry_options():
    selectors = {
        method_id: run.build_selector(method_id, 7)
        for method_id in run.method_specs()
    }

    assert selectors["legacy_mbk"].model_type == "MiniBatchKMeans"
    assert selectors["fused_raw_no_relevance"].feature_normalization is None
    assert selectors["fused_l2_target_relevance"].feature_transform is None
    assert selectors["fisher0_target_relevance"].fisher_rank == 0
    assert selectors["fisher8_target_relevance"].fisher_rank == 8
    assert selectors["fisher16_target_relevance"].fisher_rank == 16
    assert selectors["fisher_full_target_relevance"].fisher_rank == "full"
    for method_id, selector in selectors.items():
        assert selector.n_target_cells == 8
        assert selector.kmeans_k == 4
        assert selector.n_null_permutations == 6
        if selector.model_type == "FusedKMeans":
            expected = method_id.endswith("target_relevance")
            assert selector.fused_kmeans_kwargs["relevance_weighting"] is expected


def test_panel_metric_uses_tie_averaged_reference_regret():
    panel = {
        "seed": 1,
        "family": "linear",
        "candidates": [
            {
                "candidate_id": "a",
                "best_head_envelope_r2": 0.90,
                "heads": {"linear_ridge": {"test_r2": 0.70, "wall_seconds": 1.0}},
                "selectors": {"method": {"score": 0.5, "wall_seconds": 0.2}},
            },
            {
                "candidate_id": "b",
                "best_head_envelope_r2": 0.80,
                "heads": {"linear_ridge": {"test_r2": 0.80, "wall_seconds": 1.0}},
                "selectors": {"method": {"score": 0.5, "wall_seconds": 0.2}},
            },
            {
                "candidate_id": "c",
                "best_head_envelope_r2": 0.60,
                "heads": {"linear_ridge": {"test_r2": 0.60, "wall_seconds": 1.0}},
                "selectors": {"method": {"score": 0.1, "wall_seconds": 0.2}},
            },
        ],
    }

    metric = run._panel_metric(panel, "method")

    assert metric["selected_candidate_ids"] == ["a", "b"]
    assert np.isclose(metric["selection_regret_r2"], 0.05)
    assert metric["exact_best"] is False
    assert metric["within_0_01_r2"] is False
