"""Focused tests for continuous edge-level target continuity."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from experiments.continuous_edge_target_continuity import run
from experiments.continuous_prototype_capacity import run as capacity
from experiments.continuous_save_ranking import run as save


def test_protocol_closes_exact_edge_scores_geometry_and_gates():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    assert run.EDGE_IDS == ("row_knn5", "prototype_cross")
    assert list(run._method_specs(protocol)) == [
        "save8_k4",
        "rff64_mean8_k4",
        "save_rffmean_k4",
        "save_rffmean_residual_k4",
    ]
    assert protocol["shared_execution"]["permutations"] == 32
    assert protocol["decision_rule"]["minimum_overall_spearman"] == 0.60
    assert protocol["decision_rule"]["maximum_mean_regret"] == 0.10


def test_parent_artifact_is_exact_and_complete():
    protocol = run._load_protocol()
    raw = run._load_parent_raw(protocol)

    assert raw["artifact_status"] == "completed"
    assert len(raw["panels"]) == 18
    assert len(run._parent_panel_map(raw)) == 18


def test_row_knn_graph_has_stable_ties_exact_degree_and_no_self_edges():
    X = np.asarray([[0.0], [2.0], [1.0]], dtype=float)

    sources, targets = run._row_knn_edges(X, k=1)

    np.testing.assert_array_equal(sources, [0, 1, 2])
    # Row 2 is equidistant from rows 0 and 1; the lower row index wins.
    np.testing.assert_array_equal(targets, [2, 2, 0])
    assert not np.any(sources == targets)


def test_row_edge_loss_matches_hand_calculation():
    y = np.asarray([[0.0], [2.0], [1.0]])
    sources = np.asarray([0, 1, 2])
    targets = np.asarray([2, 2, 0])

    assert run._row_edge_loss(y, sources, targets) == 1.0


def test_prototype_cross_loss_matches_hand_calculation():
    y = np.asarray([[0.0], [2.0], [1.0], [3.0]])
    rows = {0: np.asarray([0, 1]), 1: np.asarray([2, 3])}
    support = {0: 2, 1: 2}
    adjacency = {(0, 1): 1.0, (1, 0): 1.0}

    assert run._mean_cross_absolute(y[rows[0]], y[rows[1]]) == 1.5
    assert run._prototype_cross_loss(y, rows, support, adjacency) == 1.5


def test_row_edge_permutation_null_has_correct_direction_on_ordered_line():
    X = np.arange(20, dtype=float)[:, None]
    y = np.arange(20, dtype=float)[:, None]
    sources, targets = run._row_knn_edges(X, k=2)
    actual = run._row_edge_loss(y, sources, targets)
    bank = run.null_parent._permutation_bank(20, 32, 73)
    null = np.mean(
        [run._row_edge_loss(y[permutation], sources, targets) for permutation in bank]
    )

    assert actual < null
    assert run.null_parent._score_from_losses(actual, float(null))["score"] > 0.5


def _fitted_first_fold():
    protocol = run._load_protocol()
    panel = capacity.generate_panel(protocol, 1009, "triangle_wave")
    candidate = panel["candidates"][0]
    train, holdout = next(
        iter(KFold(n_splits=4, shuffle=True, random_state=1009).split(candidate["X_train"]))
    )
    fit_seed = run.null_parent._candidate_fit_seed(1009, 0, 0)
    transformed_train, transformed_holdout, _ = run.multiview._prepare_fold_geometry(
        run._method_specs(protocol)["save8_k4"],
        candidate["X_train"][train],
        candidate["X_train"][holdout],
        candidate["primitive_train"][train],
        candidate["primitive_train"][holdout],
        panel["y_train"][train],
        fit_seed,
    )
    estimator = clone(save._base_selector(fit_seed))
    estimator._fit_geometry_only(transformed_train, panel["y_train"][train])
    return estimator, transformed_holdout, panel["y_train"][holdout]


def test_edge_components_are_deterministic_finite_and_nonmutating():
    estimator, X, y = _fitted_first_fold()
    centers = np.array(estimator._model.centers, copy=True)
    index_before = estimator.index
    bank = run.null_parent._permutation_bank(len(y), 32, 101)

    first = run._edge_components(estimator, X, y, bank)
    second = run._edge_components(estimator, X, y, bank)

    assert first["actual_loss"] == second["actual_loss"]
    assert first["null_loss_samples"] == second["null_loss_samples"]
    assert first["row_edge_count"] == len(y) * 5
    assert first["prototype_edge_count"] > 0
    assert first["evaluation_support_total"] == len(y)
    for edge_id in run.EDGE_IDS:
        assert first["actual_loss"][edge_id] >= 0.0
        assert len(first["null_loss_samples"][edge_id]) == 32
        assert all(np.isfinite(value) for value in first["null_loss_samples"][edge_id])
    np.testing.assert_array_equal(estimator._model.centers, centers)
    assert estimator.index == index_before


def test_selector_replays_parent_actual_loss_and_common_bank_exactly():
    protocol = run._load_protocol()
    raw = run._load_parent_raw(protocol)
    old_panel = run._parent_panel_map(raw)[(1103, "sinusoid")]
    generated = capacity.generate_panel(protocol, 1103, "sinusoid")
    candidate = generated["candidates"][0]
    old = old_panel["candidates"][0]
    method_id = "save8_k4"

    result = run.evaluate_selector(
        run._method_specs(protocol)[method_id],
        candidate["X_train"],
        candidate["primitive_train"],
        generated["y_train"],
        1103,
        0,
        old["selectors"][method_id],
    )

    assert result["existing_mean_actual_loss"] == old["selectors"][method_id][
        "mean_actual_loss"
    ]
    assert len(result["folds"]) == 4
    for fold, old_fold in zip(result["folds"], old["selectors"][method_id]["folds"]):
        assert fold["existing_actual_loss"] == old_fold["actual_loss"]
        assert fold["common_bank_seed"] == old_fold["common_bank_seed"]
        assert fold["row_edge_count"] == 625
        for edge_id in run.EDGE_IDS:
            assert len(fold["edges"][edge_id]["null_loss_samples"]) == 32


def test_decision_requires_complete_periodic_repair_and_absolute_quality():
    protocol = run._load_protocol()

    def metric(periodic, overall, regret, exact=0.70, near=0.80, preserved=0.70):
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

    metrics = {}
    for method_id in run._method_specs(protocol):
        metrics[method_id] = {
            "current_common32_oi": metric(0.30, 0.61, 0.08),
            "row_knn5_oi": metric(0.50, 0.66, 0.07),
            "prototype_cross_oi": metric(0.35, 0.59, 0.12, exact=0.50, near=0.60),
        }

    decision = run._decision(protocol, metrics)

    assert decision["status"] == "pass"
    assert set(decision["passing_candidates"]) == {
        f"{method_id}:row_knn5" for method_id in run._method_specs(protocol)
    }
    for method in decision["methods"].values():
        assert method["row_knn5"]["status"] == "pass"
        assert method["prototype_cross"]["status"] == "fail"
