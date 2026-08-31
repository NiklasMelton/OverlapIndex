"""Contract tests for the frozen continuous residual-only factorial."""

from copy import deepcopy

import numpy as np

from experiments.continuous_residual_ranking import run


def test_protocol_matches_closed_factorial():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    assert list(run.method_specs()) == protocol["continuous_oi"]["methods"]
    assert len(protocol["seeds"]) == 3
    assert len(protocol["families"]) == 8
    assert len(protocol["candidate_specs"]) == 8
    assert set(protocol["head_diversity_gate"]["minimum_unique_wins"]) == {
        "linear_ridge",
        "quadratic_ridge",
        "knn",
        "rbf_svr",
    }


def test_generator_is_deterministic_and_uses_one_shared_target():
    protocol = run._load_protocol()
    first = run.generate_panel(protocol, 401, "quadratic_product")
    second = run.generate_panel(protocol, 401, "quadratic_product")

    np.testing.assert_array_equal(first["y_train"], second["y_train"])
    np.testing.assert_array_equal(first["y_test"], second["y_test"])
    assert len(first["candidates"]) == 8
    for left, right in zip(first["candidates"], second["candidates"]):
        assert left["candidate_id"] == right["candidate_id"]
        assert left["X_train"].shape == (400, 16)
        assert left["X_test"].shape == (400, 16)
        np.testing.assert_array_equal(left["X_train"], right["X_train"])
        np.testing.assert_array_equal(left["X_test"], right["X_test"])


def test_relationship_families_are_structurally_distinct():
    protocol = run._load_protocol()
    targets = {
        family: run.generate_panel(protocol, 401, family)["y_train"]
        for family in protocol["families"]
    }

    assert not np.array_equal(targets["linear"], targets["quadratic_product"])
    assert not np.array_equal(targets["quadratic_product"], targets["radial_quadratic"])
    assert not np.array_equal(targets["sinusoid"], targets["sawtooth"])


def test_selector_factorial_changes_only_declared_geometry():
    selectors = {
        method_id: run.build_selector(method_id, 7)
        for method_id in run.method_specs()
    }

    assert selectors["legacy_mbk"].model_type == "MiniBatchKMeans"
    assert selectors["fisher_full_target_relevance"].feature_transform == (
        "shared_fisher"
    )
    assert selectors["residual0_target_relevance"].feature_transform == (
        "shared_residual"
    )
    assert selectors["residual0_target_relevance"].fisher_rank == 0
    assert selectors["residual8_target_relevance"].fisher_rank == 8
    assert selectors["residual16_target_relevance"].fisher_rank == 16
    assert selectors["residual_full_target_relevance"].fisher_rank == "full"
    for method_id, selector in selectors.items():
        assert selector.n_target_cells == 8
        assert selector.kmeans_k == 4
        assert selector.n_null_permutations == 4
        if selector.model_type == "FusedKMeans":
            assert selector.fused_kmeans_kwargs["relevance_weighting"] is (
                method_id.endswith("target_relevance")
            )


def _summary_panel(head_id):
    ordered = ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
    position = ordered.index(head_id)
    heads = {
        name: {"test_r2": 0.1, "wall_seconds": 0.1}
        for name in ordered
    }
    envelope = 0.6 + 0.1 * position
    heads[head_id]["test_r2"] = envelope
    selectors = {
        method_id: {"score": float(position), "wall_seconds": 0.1}
        for method_id in run.method_specs()
    }
    return {
        "candidate_id": head_id,
        "best_head_envelope_r2": envelope,
        "best_head_ids": [head_id],
        "heads": heads,
        "selectors": selectors,
    }


def test_diversity_gate_fails_closed_when_one_head_is_underrepresented():
    protocol = deepcopy(run._load_protocol())
    protocol["head_diversity_gate"]["minimum_unique_wins"] = {
        "linear_ridge": 1,
        "quadratic_ridge": 1,
        "knn": 1,
        "rbf_svr": 2,
    }
    candidates = [_summary_panel(name) for name in protocol["reference_heads"]["families"]]
    panel = {
        "seed": 1,
        "family": "linear",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

    summary = run.summarize(protocol, [panel])

    assert summary["head_diversity_gate"]["status"] == "fail"
    assert summary["head_diversity_gate"]["checks"]["rbf_svr"] == {
        "observed": 1,
        "minimum": 2,
        "pass": False,
    }


def test_full_residual_selector_preserves_metric_dimension():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 401, "quadratic_product")
    candidate = panel["candidates"][0]
    selector = run.build_selector("residual_full_target_relevance", 5)

    selector.fit(candidate["X_train"][:160], panel["y_train"][:160])

    assert selector.n_features_in_ == 16
    assert selector.n_features_out_ == 16
    assert selector.fisher_diagnostics_["feature_transform"] == "shared_residual"
