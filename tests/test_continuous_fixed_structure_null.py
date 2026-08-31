"""Focused tests for the strict fixed-structure continuous-OI null."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from experiments.continuous_fixed_structure_null import run
from experiments.continuous_prototype_capacity import run as capacity
from experiments.continuous_save_ranking import run as save


def test_protocol_closes_exact_parent_grid_and_strict_null():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)

    assert list(run._method_specs(protocol)) == [
        "save8_k4",
        "rff64_mean8_k4",
        "save_rffmean_k4",
        "save_rffmean_residual_k4",
    ]
    assert protocol["shared_execution"]["permutations"] == 32
    strict = protocol["null_variants"]["strict_fixed_structure"]
    assert strict["permuted_only"].startswith("scaled held-out target values")
    assert "prototype adjacency" in strict["fixed_from_actual_holdout"]


def test_parent_artifact_and_source_hashes_are_exact():
    protocol = run._load_protocol()
    raw = run._load_parent_raw(protocol)

    assert raw["artifact_status"] == "completed"
    assert len(raw["panels"]) == 18
    assert len(run._parent_panel_map(raw)) == 18


def _fitted_first_fold(method_id="save8_k4"):
    protocol = run._load_protocol()
    panel = capacity.generate_panel(protocol, 1009, "triangle_wave")
    candidate = panel["candidates"][0]
    train, holdout = next(
        iter(KFold(n_splits=4, shuffle=True, random_state=1009).split(candidate["X_train"]))
    )
    fit_seed = run.parent._candidate_fit_seed(1009, 0, 0)
    transformed_train, transformed_holdout, _ = run.multiview._prepare_fold_geometry(
        run._method_specs(protocol)[method_id],
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


def test_fixed_structure_actual_loss_is_bit_exact_with_current_scorer():
    estimator, X, y = _fitted_first_fold()
    structure = run._fixed_evaluation_structure(estimator, X, y)
    X_prepared = estimator._prep_X(estimator._validate_X(X))
    y_scaled = estimator._transform_targets(estimator._validate_Y(y))

    current_actual, current_local, current_support = estimator._fixed_model_loss(
        X_prepared,
        y_scaled,
    )

    assert structure["actual_loss"] == current_actual
    assert structure["local_loss"] == current_local
    assert structure["support"] == current_support
    assert sum(structure["support"].values()) == len(y)
    assert len(structure["rows_by_proto"]) == len(structure["support"])
    assert structure["adjacency"]


def test_strict_replay_never_reassigns_targets_prototypes_or_adjacency(monkeypatch):
    estimator, X, y = _fitted_first_fold()
    structure = run._fixed_evaluation_structure(estimator, X, y)
    bank = run.parent._permutation_bank(len(y), 4, 91)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("strict null attempted to rebuild evaluation structure")

    monkeypatch.setattr(estimator, "_assign_target_cells", forbidden)
    monkeypatch.setattr(estimator._model, "bmu_for_class_batch", forbidden)
    monkeypatch.setattr(estimator, "_adjacency_for_model", forbidden)

    losses = run._strict_null_samples(estimator, structure, bank)

    assert len(losses) == 4
    assert all(np.isfinite(value) and value >= 0.0 for value in losses)


def test_strict_null_is_deterministic_and_does_not_mutate_state():
    estimator, X, y = _fitted_first_fold("rff64_mean8_k4")
    structure = run._fixed_evaluation_structure(estimator, X, y)
    centers = np.array(estimator._model.centers, copy=True)
    index_before = estimator.index
    bank = run.parent._permutation_bank(len(y), 32, 177)

    first = run._strict_null_samples(estimator, structure, bank)
    second = run._strict_null_samples(estimator, structure, bank)

    assert first == second
    np.testing.assert_array_equal(estimator._model.centers, centers)
    assert estimator.index == index_before


def test_selector_replay_matches_parent_actual_loss_and_bank_identity():
    protocol = run._load_protocol()
    raw = run._load_parent_raw(protocol)
    old_panel = run._parent_panel_map(raw)[(1103, "sinusoid")]
    generated = capacity.generate_panel(protocol, 1103, "sinusoid")
    old_candidate = old_panel["candidates"][0]
    candidate = generated["candidates"][0]
    method_id = "save8_k4"

    result = run.evaluate_selector(
        run._method_specs(protocol)[method_id],
        candidate["X_train"],
        candidate["primitive_train"],
        generated["y_train"],
        1103,
        0,
        old_candidate["selectors"][method_id],
    )

    assert result["mean_actual_loss"] == old_candidate["selectors"][method_id][
        "mean_actual_loss"
    ]
    assert len(result["folds"]) == 4
    for fold, old_fold in zip(
        result["folds"], old_candidate["selectors"][method_id]["folds"]
    ):
        assert fold["actual_loss"] == old_fold["actual_loss"]
        assert fold["common_bank_seed"] == old_fold["common_bank_seed"]
        assert len(fold["strict_null_loss_samples"]) == 32
        assert fold["evaluation_support_total"] == 125


def test_decision_requires_periodic_repair_preservation_and_absolute_quality():
    protocol = run._load_protocol()

    def metric(periodic, overall, regret, preserved=0.70):
        return {
            "panel_count": 18,
            "defined_spearman_panel_count": 18,
            "mean_spearman": overall,
            "mean_regret_r2": regret,
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

    passing = {
        method_id: {
            "current_common32_oi": metric(0.30, 0.61, 0.08),
            "strict_common32_oi": metric(0.50, 0.66, 0.07),
        }
        for method_id in run._method_specs(protocol)
    }
    decision = run._decision(protocol, passing)
    assert decision["status"] == "pass"

    failing = {
        method_id: {
            "current_common32_oi": metric(0.30, 0.61, 0.08),
            "strict_common32_oi": metric(0.35, 0.59, 0.12),
        }
        for method_id in run._method_specs(protocol)
    }
    decision = run._decision(protocol, failing)
    assert decision["status"] == "fail"
    assert decision["passing_methods"] == []


def test_decision_fails_closed_when_strict_bounded_score_is_constant():
    protocol = run._load_protocol()

    def metric(value, defined):
        return {
            "panel_count": 18,
            "defined_spearman_panel_count": defined,
            "mean_spearman": value,
            "mean_regret_r2": 0.05,
            "by_family": {
                family: {
                    "mean_spearman": (
                        None if family == "triangle_wave" and defined < 18 else value
                    )
                }
                for family in protocol["families"]
            },
        }

    metrics = {
        method_id: {
            "current_common32_oi": metric(0.70, 18),
            "strict_common32_oi": metric(0.75, 15),
        }
        for method_id in run._method_specs(protocol)
    }

    decision = run._decision(protocol, metrics)

    assert decision["status"] == "fail"
    for row in decision["methods"].values():
        checks = {check["criterion"]: check for check in row["checks"]}
        assert checks["all_panel_spearman_defined"]["pass"] is False
        assert checks["mean_periodic_spearman_gain"]["observed"] is None
        assert checks["mean_periodic_spearman_gain"]["pass"] is False
