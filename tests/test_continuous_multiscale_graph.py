"""Focused tests for the frozen continuous multiscale graph experiment."""

import numpy as np

from experiments.continuous_multiscale_graph import run


def test_protocol_closes_new_seeds_geometries_scales_and_formulations():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    assert protocol["seeds"] == [1301, 1409, 1511]
    assert run.SCALE_IDS == (5, 15, 45)
    assert run.FORMULATION_IDS == (
        "uniform_multiscale",
        "self_tuning_multiscale",
    )
    assert list(run._method_specs(protocol)) == [
        "rff64_mean8_k4",
        "save_rffmean_k4",
    ]


def test_motivation_ancestry_hashes_are_exact():
    run._validate_motivation_ancestry(run._load_protocol())


def test_ordered_neighbor_graph_has_stable_ties_nested_scales_and_no_self_edges():
    X = np.arange(50, dtype=float)[:, None]
    graph = run._ordered_neighbor_graph(X)

    assert graph["neighbors"].shape == (50, 45)
    assert graph["squared_distances"].shape == (50, 45)
    assert np.all(graph["local_scale"] > 0.0)
    for k in run.SCALE_IDS:
        sources, targets, weights = run._scale_edges(
            graph, k, "uniform_multiscale"
        )
        assert len(sources) == 50 * k
        assert not np.any(sources == targets)
        np.testing.assert_array_equal(weights, np.ones(50 * k))
    # Row 1 is equidistant from rows 0 and 2; stable lower-index order wins.
    assert graph["neighbors"][1, 0] == 0
    assert graph["neighbors"][1, 1] == 2


def test_self_tuning_weights_are_exact_finite_and_density_adaptive():
    X = np.concatenate([np.linspace(0.0, 1.0, 30), np.linspace(3.0, 8.0, 30)])[:, None]
    graph = run._ordered_neighbor_graph(X)
    sources, targets, weights = run._scale_edges(
        graph, 15, "self_tuning_multiscale"
    )
    expected = np.exp(
        -graph["squared_distances"][:, :15].reshape(-1)
        / (graph["local_scale"][sources] * graph["local_scale"][targets])
    )

    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(weights))
    assert np.all((weights > 0.0) & (weights <= 1.0))
    assert not np.allclose(weights, weights[0])


def test_duplicate_rows_receive_positive_deterministic_local_scale():
    X = np.zeros((50, 3), dtype=float)
    first = run._ordered_neighbor_graph(X)
    second = run._ordered_neighbor_graph(X)

    np.testing.assert_array_equal(first["neighbors"], second["neighbors"])
    np.testing.assert_array_equal(first["local_scale"], second["local_scale"])
    assert np.all(first["local_scale"] > 0.0)
    _sources, _targets, weights = run._scale_edges(
        first, 45, "self_tuning_multiscale"
    )
    np.testing.assert_array_equal(weights, np.ones(50 * 45))


def test_weighted_edge_loss_matches_hand_calculation():
    y = np.asarray([[0.0], [1.0], [4.0]])
    sources = np.asarray([0, 1])
    targets = np.asarray([1, 2])
    weights = np.asarray([1.0, 3.0])

    assert run._weighted_edge_loss(y, sources, targets, weights) == 2.5


def test_uniform_k5_exactly_replays_prior_row_edge_definition():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(60, 4))
    y = rng.normal(size=(60, 1))
    graph = run._ordered_neighbor_graph(X)
    sources, targets, weights = run._scale_edges(
        graph, 5, "uniform_multiscale"
    )
    old_sources, old_targets = run.edge_parent._row_knn_edges(X, k=5)

    np.testing.assert_array_equal(sources, old_sources)
    np.testing.assert_array_equal(targets, old_targets)
    assert run._weighted_edge_loss(y, sources, targets, weights) == (
        run.edge_parent._row_edge_loss(y, old_sources, old_targets)
    )


def test_multiscale_null_has_correct_direction_on_ordered_line():
    protocol = run._load_protocol()
    X = np.arange(60, dtype=float)[:, None]
    y = np.arange(60, dtype=float)[:, None]
    bank = run.null_parent._permutation_bank(60, 32, 83)

    result = run._graph_fold_components(X, y, bank, protocol)

    for formulation in run.FORMULATION_IDS:
        assert result["formulations"][formulation]["score"] > 0.5
        for k in run.SCALE_IDS:
            scale = result["formulations"][formulation]["scales"][str(k)]
            assert scale["actual_loss"] < scale["null_loss"]
            assert scale["edge_count"] == 60 * k
            assert len(scale["null_loss_samples"]) == 32


def test_integrated_score_is_fixed_equal_mean_of_scale_scores():
    protocol = run._load_protocol()
    X = np.arange(60, dtype=float)[:, None]
    y = np.sin(np.arange(60, dtype=float) / 10.0)[:, None]
    bank = run.null_parent._permutation_bank(60, 32, 91)

    result = run._graph_fold_components(X, y, bank, protocol)

    for formulation in run.FORMULATION_IDS:
        row = result["formulations"][formulation]
        expected = np.mean([row["scales"][str(k)]["score"] for k in run.SCALE_IDS])
        np.testing.assert_allclose(row["score"], expected, rtol=0.0, atol=2.0e-16)


def _metric(periodic, overall, regret, exact=0.70, near=0.80, preserved=0.70):
    protocol = run._load_protocol()
    return {
        "panel_count": 18,
        "defined_spearman_panel_count": 18,
        "mean_spearman": overall,
        "mean_regret_r2": regret,
        "exact_best_rate": exact,
        "within_0_01_rate": near,
        "by_family": {
            family: {
                "mean_spearman": (
                    periodic
                    if family in {"sinusoid", "triangle_wave"}
                    else preserved
                )
            }
            for family in protocol["families"]
        },
    }


def test_decision_requires_universal_absolute_and_relative_gates():
    protocol = run._load_protocol()
    metrics = {
        "linear_probe": {"linear_probe": _metric(0.30, 0.50, 0.08)},
    }
    for method_id in run._method_specs(protocol):
        metrics[method_id] = {
            "current_common32_oi": _metric(0.30, 0.61, 0.08),
            "uniform_multiscale_oi": _metric(0.45, 0.64, 0.07),
            "self_tuning_multiscale_oi": _metric(
                0.35, 0.59, 0.11, exact=0.50, near=0.60
            ),
        }

    decision = run._decision(protocol, metrics)

    assert decision["status"] == "pass"
    assert set(decision["passing_candidates"]) == {
        f"{method_id}:uniform_multiscale"
        for method_id in run._method_specs(protocol)
    }
    for method in decision["methods"].values():
        assert method["uniform_multiscale"]["status"] == "pass"
        assert method["self_tuning_multiscale"]["status"] == "fail"


def test_decision_fails_closed_on_undefined_periodic_panel():
    protocol = run._load_protocol()
    metrics = {"linear_probe": {"linear_probe": _metric(0.30, 0.50, 0.08)}}
    for method_id in run._method_specs(protocol):
        candidate = _metric(0.50, 0.70, 0.05)
        candidate["defined_spearman_panel_count"] = 17
        metrics[method_id] = {
            "current_common32_oi": _metric(0.30, 0.61, 0.08),
            "uniform_multiscale_oi": candidate,
            "self_tuning_multiscale_oi": candidate,
        }

    decision = run._decision(protocol, metrics)

    assert decision["status"] == "fail"
    assert decision["passing_candidates"] == []


def test_panel_metric_uses_frozen_tie_averaging():
    panel = {
        "seed": 1,
        "family": "linear",
        "candidates": [
            {
                "candidate_id": "a",
                "best_head_envelope_r2": 0.9,
                "heads": {"linear_ridge": {"test_r2": 0.5}},
                "selectors": {},
            },
            {
                "candidate_id": "b",
                "best_head_envelope_r2": 0.7,
                "heads": {"linear_ridge": {"test_r2": 0.5 + 5.0e-13}},
                "selectors": {},
            },
        ],
    }

    metric = run._panel_metric(panel, "linear_probe", "linear_probe")

    assert metric["selected_candidate_ids"] == ["a", "b"]
    np.testing.assert_allclose(metric["selection_regret_r2"], 0.1)
