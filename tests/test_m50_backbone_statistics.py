"""Focused tests for strict M50 backbone-ranking statistics and policies."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from experiments.m50_backbone_ranking import fusion, statistics


ALL_CANDIDATES = ("A", "B", "M0-SW", "M1-SW", "M0-CB", "M1-CB", "LP-FULL")


def _food_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selectors: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    for replicate in statistics.FOOD_REPLICATES:
        for arm in statistics.FOOD_ARMS:
            for backbone_position, backbone in enumerate(statistics.FOOD_BACKBONES):
                for head in statistics.REFERENCE_HEADS:
                    if arm == "nonlinearity_full" and head != "linear":
                        accuracy = 0.90 if backbone_position == 8 else 0.84 - 0.001 * backbone_position
                    else:
                        accuracy = 0.70 + 0.005 * backbone_position
                    references.append(
                        {
                            "backbone": backbone,
                            "replicate": replicate,
                            "arm": arm,
                            "head": head,
                            "test_accuracy": accuracy,
                        }
                    )
            for budget in statistics.FOOD_BUDGETS:
                for candidate in ALL_CANDIDATES:
                    for backbone_position, backbone in enumerate(statistics.FOOD_BACKBONES):
                        if candidate == "LP-FULL":
                            preferred = 9
                        elif candidate in {"M0-SW", "M1-SW", "M0-CB", "M1-CB"}:
                            preferred = 8 if arm == "nonlinearity_full" else 9
                        else:
                            preferred = 8
                        selectors.append(
                            {
                                "candidate_id": candidate,
                                "backbone": backbone,
                                "replicate": replicate,
                                "arm": arm,
                                "budget": budget,
                                "score": 1.0 if backbone_position == preferred else backbone_position / 100.0,
                            }
                        )
    return selectors, references


def _contrast_metrics(
    *,
    candidate: str = "M0-SW",
    linear: float = 0.0,
    quadratic: float = -1.0,
    knn: float = -1.0,
    rbf: float = -1.0,
) -> list[dict[str, object]]:
    difference = {
        ("baseline", "linear"): linear,
        ("nuisance_full", "linear"): linear,
        ("nonlinearity_full", "quadratic"): quadratic,
        ("nonlinearity_full", "knn"): knn,
        ("nonlinearity_full", "rbf"): rbf,
    }
    rows: list[dict[str, object]] = []
    for replicate in statistics.FOOD_REPLICATES:
        for budget in statistics.FOOD_BUDGETS:
            for (arm, head), value in difference.items():
                rows.extend(
                    [
                        {
                            "candidate_id": "LP-FULL",
                            "replicate": replicate,
                            "arm": arm,
                            "budget": budget,
                            "head": head,
                            "regret_pp": 2.0,
                        },
                        {
                            "candidate_id": candidate,
                            "replicate": replicate,
                            "arm": arm,
                            "budget": budget,
                            "head": head,
                            "regret_pp": 2.0 + value,
                        },
                    ]
                )
    return rows


def _runtime_rows(
    candidate: str = "M0-SW",
    *,
    candidate_probe_ratio: float = 0.9,
    candidate_b_ratio: float = 1.05,
    memory_b_ratio: float = 1.05,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for backbone in statistics.FOOD_BACKBONES:
        for arm in statistics.FOOD_ARMS:
            for budget in statistics.FOOD_RUNTIME_BUDGETS:
                for repeat in statistics.FOOD_REPLICATES:
                    baseline_wall = 1.0
                    candidate_wall = candidate_b_ratio * baseline_wall
                    probe_wall = candidate_wall / candidate_probe_ratio
                    for method, wall, memory in (
                        ("B", baseline_wall, 100.0),
                        (candidate, candidate_wall, memory_b_ratio * 100.0),
                        ("LP-FULL", probe_wall, 110.0),
                    ):
                        rows.append(
                            {
                                "candidate_id": method,
                                "backbone": backbone,
                                "arm": arm,
                                "budget": budget,
                                "repeat": repeat,
                                "total_wall_seconds": wall,
                                "total_cpu_seconds": wall,
                                "fit_wall_seconds": 0.6 * wall,
                                "fit_cpu_seconds": 0.6 * wall,
                                "conditioning_fit_wall_seconds": 0.1 * wall,
                                "conditioning_fit_cpu_seconds": 0.1 * wall,
                                "oi_fit_wall_seconds": 0.5 * wall,
                                "oi_fit_cpu_seconds": 0.5 * wall,
                                "score_fixed_wall_seconds": 0.4 * wall,
                                "score_fixed_cpu_seconds": 0.4 * wall,
                                "peak_memory_mb": memory,
                            }
                        )
    return rows


def test_complete_food_metrics_materialize_generic_iterables_once() -> None:
    selectors, references = _food_rows()
    metrics = statistics.food_panel_metrics(
        (row for row in selectors),
        (row for row in references),
        candidate_ids=ALL_CANDIDATES,
    )
    expected = (
        len(ALL_CANDIDATES)
        * len(statistics.FOOD_REPLICATES)
        * len(statistics.FOOD_ARMS)
        * len(statistics.FOOD_BUDGETS)
        * len(statistics.REFERENCE_HEADS)
    )
    assert len(metrics) == expected
    auc = statistics.rank_auc_rows(metrics)
    assert len(auc) == (
        len(ALL_CANDIDATES)
        * len(statistics.FOOD_REPLICATES)
        * len(statistics.FOOD_ARMS)
        * len(statistics.REFERENCE_HEADS)
    )
    assert all(row["rank_auc_status"] == "defined" for row in auc)


def test_food_panel_validation_rejects_missing_duplicate_and_nonfinite() -> None:
    selectors, references = _food_rows()
    with pytest.raises(ValueError, match="incomplete"):
        statistics.food_panel_metrics(
            selectors[:-1], references, candidate_ids=ALL_CANDIDATES
        )
    with pytest.raises(ValueError, match="duplicate"):
        statistics.food_panel_metrics(
            selectors + [deepcopy(selectors[0])], references, candidate_ids=ALL_CANDIDATES
        )
    invalid = deepcopy(selectors)
    invalid[0]["score"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        statistics.food_panel_metrics(invalid, references, candidate_ids=ALL_CANDIDATES)


def test_archived_ordinary_ties_are_averaged_but_fusion_uses_frozen_tie_order() -> None:
    design = statistics.FoodDesign(
        backbones=("first", "second"),
        replicates=(0,),
        arms=("baseline",),
        budgets=(64,),
        heads=("linear",),
    )
    selectors = [
        {"candidate_id": "A", "backbone": backbone, "replicate": 0, "arm": "baseline", "budget": 64, "score": 0.5}
        for backbone in design.backbones
    ]
    references = [
        {"backbone": "first", "replicate": 0, "arm": "baseline", "head": "linear", "test_accuracy": 0.8},
        {"backbone": "second", "replicate": 0, "arm": "baseline", "head": "linear", "test_accuracy": 1.0},
    ]
    row = statistics.food_panel_metrics(
        selectors, references, candidate_ids=("A",), design=design
    )[0]
    assert row["regret_pp"] == pytest.approx(10.0)
    assert row["exact_best_rate"] == 0.5
    assert row["selected_backbones"] == ["first", "second"]
    assert fusion.select_highest_score(
        {"first": 0.5, "second": 0.5}, item_order=design.backbones
    ) == "first"
    fusion_selectors = [
        {
            **row,
            "candidate_id": "F",
            "selected": row["backbone"] == "first",
        }
        for row in selectors
    ]
    fused = statistics.food_panel_metrics(
        fusion_selectors, references, candidate_ids=("F",), design=design
    )[0]
    assert fused["selected_backbone"] == "first"
    assert fused["regret_pp"] == pytest.approx(20.0)
    assert fused["selection_semantics"] == "derived_frozen_order_argmax"

    guardrail_selectors = [
        {
            **row,
            "candidate_id": "G",
            "score": 1.0 if row["backbone"] == "second" else 1.0 - 5e-13,
            "selected": True,
            "selection_semantics": "archived_atol_1e-12_tie_average",
        }
        for row in selectors
    ]
    guardrail_metric = statistics.food_panel_metrics(
        guardrail_selectors, references, candidate_ids=("G",), design=design
    )[0]
    assert guardrail_metric["selected_backbones"] == ["first", "second"]
    assert guardrail_metric["regret_pp"] == pytest.approx(10.0)
    assert guardrail_metric["selection_semantics"] == "archived_atol_1e-12_tie_average"


def test_bootstrap_is_deterministic_and_uses_frozen_seed() -> None:
    values = {replicate: float(replicate) for replicate in statistics.FOOD_REPLICATES}
    first = statistics.paired_replicate_interval(values, n_resamples=200)
    second = statistics.paired_replicate_interval(values.items(), n_resamples=200)
    assert first == second
    assert statistics.BOOTSTRAP_SEED == 2026082701


def test_factorial_effects_and_provisional_lock_choose_unrefined_speed_trial() -> None:
    selectors, references = _food_rows()
    metrics = statistics.food_panel_metrics(
        selectors, references, candidate_ids=ALL_CANDIDATES
    )
    factorial = statistics.factorial_effects(metrics, n_resamples=200)
    assert factorial["candidate_cells"] == ["A", "B", "M0-SW", "M1-SW"]
    assert factorial["diagnostic_cells"] == ["M0-CB", "M1-CB"]
    assert all(
        payload["interaction"]["estimate"] == pytest.approx(0.0)
        for payload in factorial["effects"].values()
    )
    assert all(
        payload["cb_refinement_diagnostic"]["promotable"] is False
        for payload in factorial["effects"].values()
    )
    decision = statistics.provisional_development_lock(metrics, n_resamples=200)
    assert decision["selected_candidate"] == "M0-SW"
    assert decision["selected_chain"] is None
    assert decision["runtime_candidate_ids"] == list(statistics.RUNTIME_PRIMITIVE_IDS)
    assert decision["unrefined_requires_speed_ratio"] is True
    assert decision["cb_rescue_allowed"] is False


@pytest.mark.parametrize(
    "rbf, expected",
    [(-0.001, "pass"), (0.0, "fusion_required"), (1.0, "fail")],
)
def test_rbf_gate_paths_are_exact(rbf: float, expected: str) -> None:
    gates = statistics.candidate_product_gates(
        _contrast_metrics(rbf=rbf), candidate_id="M0-SW", n_resamples=200
    )
    assert gates["status"] == expected


def test_guardrail_path_requires_nonlinear_pass_and_does_not_weaken_claim() -> None:
    eligible = statistics.candidate_product_gates(
        _contrast_metrics(linear=1.01, rbf=-0.1),
        candidate_id="M0-SW",
        n_resamples=200,
    )
    assert eligible["status"] == "guardrail_eligible"
    assert eligible["nonlinear_gates_pass"] is True
    blocked = statistics.candidate_product_gates(
        _contrast_metrics(linear=1.01, quadratic=0.0, rbf=-0.1),
        candidate_id="M0-SW",
        n_resamples=200,
    )
    assert blocked["status"] == "fail"
    assert blocked["guardrail_eligible"] is False


def test_fractional_midrank_fusion_requires_complete_views() -> None:
    order = ("a", "b", "c")
    assert fusion.fractional_midranks(
        {"a": 1.0, "b": 1.0, "c": 3.0}, item_order=order
    ) == {"a": 0.25, "b": 0.25, "c": 1.0}
    with pytest.raises(ValueError, match="missing"):
        fusion.fuse_panel_scores(
            {"a": 1.0, "b": 2.0},
            {"a": 1.0, "b": 2.0, "c": 3.0},
            item_order=order,
        )


def _policy_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    diagnostics, base, capped = [], [], []
    for position, backbone in enumerate(statistics.FOOD_BACKBONES):
        diagnostics.extend(
            [
                {
                    "candidate_id": "B",
                    "backbone": backbone,
                    "replicate": 0,
                    "arm": "baseline",
                    "budget": 64,
                    "score": position / 10.0,
                    "prototype_refinement": {"applied_rate": 0.50 if position == 0 else 0.0},
                    "total_wall_seconds": 1.0,
                    "total_cpu_seconds": 1.0,
                },
                {
                    "candidate_id": "M1-SW",
                    "backbone": backbone,
                    "replicate": 0,
                    "arm": "baseline",
                    "budget": 64,
                    "score": position / 10.0,
                    "total_wall_seconds": 2.0,
                    "total_cpu_seconds": 2.0,
                },
            ]
        )
        base.append(
            {
                "candidate_id": "F",
                "backbone": backbone,
                "replicate": 0,
                "arm": "baseline",
                "budget": 64,
                "score": position / 10.0,
                "primitive_timings": {
                    "B": {"wall_seconds": 1.0, "cpu_seconds": 1.0},
                    "M1-SW": {"wall_seconds": 2.0, "cpu_seconds": 2.0},
                },
                "total_wall_seconds": 3.0,
                "total_cpu_seconds": 3.0,
                "selected": position == len(statistics.FOOD_BACKBONES) - 1,
            }
        )
        capped.append(
            {
                "candidate_id": "LP-CAPPED-2048",
                "backbone": backbone,
                "replicate": 0,
                "arm": "baseline",
                "budget": 64,
                "score": position / 10.0,
                "total_wall_seconds": 3.0,
                "total_cpu_seconds": 3.0,
            }
        )
    return diagnostics, base, capped


def test_guardrail_trigger_rejects_outcomes_and_unions_primitive_time_once() -> None:
    diagnostics, base, capped = _policy_rows()
    policy = fusion.derive_guardrail_rows(
        diagnostic_rows=diagnostics,
        base_rows=base,
        capped_probe_rows=capped,
        backbones=statistics.FOOD_BACKBONES,
        base_candidate_id="F",
    )
    assert all(row["triggered"] is True for row in policy)
    assert all(row["charged_component_ids"] == ["B", "LP-CAPPED-2048", "M1-SW"] for row in policy)
    assert all(row["total_wall_seconds"] == 6.0 for row in policy)
    untriggered_diagnostics = deepcopy(diagnostics)
    for row in untriggered_diagnostics:
        if row["candidate_id"] == "B":
            row["prototype_refinement"] = {"applied_rate": 0.0}
        if row["backbone"] == statistics.FOOD_BACKBONES[-2]:
            row["score"] = 1.0 - 5e-13
        elif row["backbone"] == statistics.FOOD_BACKBONES[-1]:
            row["score"] = 1.0
    untriggered_base = [
        row for row in untriggered_diagnostics if row["candidate_id"] == "M1-SW"
    ]
    untriggered = fusion.derive_guardrail_rows(
        diagnostic_rows=untriggered_diagnostics,
        base_rows=untriggered_base,
        capped_probe_rows=capped,
        backbones=statistics.FOOD_BACKBONES,
        base_candidate_id="M1-SW",
    )
    assert all(row["triggered"] is False for row in untriggered)
    assert [row["backbone"] for row in untriggered if row["selected"]] == list(
        statistics.FOOD_BACKBONES[-2:]
    )
    assert {
        row["selection_semantics"] for row in untriggered
    } == {"archived_atol_1e-12_tie_average"}
    leaked = deepcopy(diagnostics)
    leaked[0]["test_accuracy"] = 0.9
    with pytest.raises(ValueError, match="must not contain"):
        fusion.guardrail_trigger(leaked, backbones=statistics.FOOD_BACKBONES)

    nested = [
        {
            "candidate_id": "B",
            "backbone": backbone,
            "score": 0.1,
            "prototype_refinement": {
                "applied_rate": 0.0,
                "nested": {"regret_pp": 1.0},
            },
        }
        if position % 2 == 0
        else {
            "candidate_id": "M1-SW",
            "backbone": statistics.FOOD_BACKBONES[position // 2],
            "score": 0.1,
        }
        for position, backbone in enumerate(
            value for item in statistics.FOOD_BACKBONES for value in (item, item)
        )
    ]
    with pytest.raises(ValueError, match="must not contain"):
        fusion.guardrail_trigger(nested, backbones=statistics.FOOD_BACKBONES)

    mixed = deepcopy(diagnostics)
    mixed[0]["budget"] = 68
    with pytest.raises(ValueError, match="one aligned panel"):
        fusion.derive_guardrail_rows(
            diagnostic_rows=mixed,
            base_rows=base,
            capped_probe_rows=capped,
            backbones=statistics.FOOD_BACKBONES,
            base_candidate_id="F",
        )

    wrong_cap = deepcopy(capped)
    wrong_cap[0]["candidate_id"] = "LP-FULL"
    with pytest.raises(ValueError, match="exactly LP-CAPPED-2048"):
        fusion.derive_guardrail_rows(
            diagnostic_rows=diagnostics,
            base_rows=base,
            capped_probe_rows=wrong_cap,
            backbones=statistics.FOOD_BACKBONES,
            base_candidate_id="F",
        )


def test_resource_gate_boundaries_and_no_runtime_runner_up() -> None:
    passing = statistics.resource_gates(
        _runtime_rows(), candidate_id="M0-SW", n_resamples=200
    )
    assert passing["status"] == "pass"
    assert set(passing["arm_gates"]) == set(statistics.FOOD_ARMS)
    assert set(passing["full_panel_arm_gates"]) == set(statistics.FOOD_ARMS)
    assert all(
        payload["status"] == "pass" for payload in passing["arm_gates"].values()
    )
    assert all(
        payload["status"] == "pass"
        for payload in passing["full_panel_arm_gates"].values()
    )
    assert all(
        payload["per_backbone_faster_status"] == "established"
        and payload["full_panel_faster_status"] == "established"
        and payload["faster_claim_status"] == "established"
        for payload in passing["arm_gates"].values()
    )
    assert len(passing["full_panel_ratio_rows"]) == (
        len(statistics.FOOD_ARMS)
        * len(statistics.FOOD_RUNTIME_BUDGETS)
        * len(statistics.FOOD_REPLICATES)
    )
    secondary = passing["secondary_runtime"]
    assert secondary["inferential"] is False
    assert secondary["promotion_gate"] is False
    assert set(secondary["arm_summary"]) == set(statistics.FOOD_ARMS)
    assert len(secondary["scaling_by_arm_budget"]) == (
        len(statistics.FOOD_ARMS) * len(statistics.FOOD_RUNTIME_BUDGETS)
    )
    baseline_secondary = secondary["arm_summary"]["baseline"]
    assert baseline_secondary["per_backbone_cpu_ratios"]["candidate_vs_probe"][
        "median"
    ] == pytest.approx(0.9)
    assert "candidate_oi_fit_wall_seconds" in baseline_secondary["stage_seconds"]
    assert [row["budget"] for row in passing["scaling_by_budget"]] == list(
        statistics.FOOD_RUNTIME_BUDGETS
    )
    failing = statistics.resource_gates(
        _runtime_rows(candidate_probe_ratio=1.11),
        candidate_id="M0-SW",
        n_resamples=200,
    )
    assert failing["status"] == "fail"
    one_arm_failure = _runtime_rows()
    for row in one_arm_failure:
        if row["candidate_id"] == "M0-SW" and row["arm"] == "nuisance_full":
            row["total_wall_seconds"] = 2.0
    separated = statistics.resource_gates(
        one_arm_failure, candidate_id="M0-SW", n_resamples=200
    )
    assert separated["status"] == "fail"
    assert separated["arm_gates"]["nuisance_full"]["status"] == "fail"
    assert separated["arm_gates"]["baseline"]["status"] == "pass"
    competitive_not_faster = statistics.resource_gates(
        _runtime_rows(candidate_probe_ratio=1.0),
        candidate_id="M0-SW",
        n_resamples=200,
    )
    assert competitive_not_faster["status"] == "pass"
    assert all(
        payload["faster_claim_status"] == "not_established"
        and payload["faster_claim"]["promotion_veto"] is False
        for payload in competitive_not_faster["arm_gates"].values()
    )

    # Per-call medians can hide a single expensive backbone when that same
    # call dominates the total product latency. Full-panel ratios aggregate
    # clocks before division and therefore catch the product regression.
    product_failure = _runtime_rows()
    expensive = statistics.FOOD_BACKBONES[0]
    for row in product_failure:
        if (
            row["backbone"] == expensive
            and row["arm"] == "baseline"
            and row["budget"] >= 128
            and row["candidate_id"] in {"M0-SW", "B"}
        ):
            row["total_wall_seconds"] = 100.0
    product_gated = statistics.resource_gates(
        product_failure, candidate_id="M0-SW", n_resamples=200
    )
    assert product_gated["arm_gates"]["baseline"]["status"] == "pass"
    assert product_gated["full_panel_arm_gates"]["baseline"]["status"] == "fail"
    assert product_gated["status"] == "fail"
    provisional = {
        "status": "provisional_runtime_pending",
        "selected_candidate": "M0-SW",
        "selected_chain": None,
        "runtime_pending": True,
        "runner_up_allowed": False,
        "candidate_gates": {"M0-SW": {"requires_fusion": False, "guardrail_eligible": False}},
        "unrefined_requires_speed_ratio": False,
    }
    stopped = statistics.finalize_development_lock(
        provisional,
        resource_summary={"candidate_id": "M0-SW", "status": "fail"},
    )
    assert stopped["status"] == "stopped_resource_failure"
    assert stopped["selected_chain"] is None
    assert stopped["runner_up_allowed"] is False


def test_refinement_speed_choice_requires_all_three_arms() -> None:
    provisional = {
        "status": "provisional_runtime_pending",
        "selected_candidate": "M0-SW",
        "selected_chain": None,
        "runtime_pending": True,
        "runner_up_allowed": False,
        "candidate_gates": {
            candidate: {"requires_fusion": False, "guardrail_eligible": False}
            for candidate in statistics.SW_CANDIDATES
        },
        "unrefined_requires_speed_ratio": True,
    }
    all_fast = {arm: 0.95 for arm in statistics.FOOD_ARMS}
    m0 = statistics.finalize_development_lock(
        provisional,
        resource_summary={"candidate_id": "M0-SW", "status": "pass"},
        unrefined_vs_refined_runtime_ratio_by_arm=all_fast,
    )
    assert m0["selected_candidate"] == "M0-SW"
    assert m0["refinement_runtime_ratio_by_arm"] == all_fast
    one_slow = {**all_fast, "nuisance_full": 0.951}
    m1 = statistics.finalize_development_lock(
        provisional,
        resource_summary={"candidate_id": "M1-SW", "status": "pass"},
        unrefined_vs_refined_runtime_ratio_by_arm=one_slow,
    )
    assert m1["selected_candidate"] == "M1-SW"
    with pytest.raises(ValueError, match="exact per-arm"):
        statistics.finalize_development_lock(
            provisional,
            resource_summary={"candidate_id": "M0-SW", "status": "pass"},
            unrefined_vs_refined_runtime_ratio_by_arm={"baseline": 0.9},
        )
    with pytest.raises(ValueError, match="selected_chain=null"):
        statistics.finalize_development_lock(
            {**provisional, "selected_chain": "M0-SW"},
            resource_summary={"candidate_id": "M0-SW", "status": "pass"},
            unrefined_vs_refined_runtime_ratio_by_arm=all_fast,
        )
    with pytest.raises(ValueError, match="compatibility aliases"):
        statistics.finalize_development_lock(
            {**provisional, "locked_candidate": "M0-SW"},
            resource_summary={"candidate_id": "M0-SW", "status": "pass"},
            unrefined_vs_refined_runtime_ratio_by_arm=all_fast,
        )


def test_confirmation_requires_exact_lock_and_five_complete_datasets() -> None:
    datasets = tuple(f"new-{position}" for position in range(5))
    lock = {
        "status": "locked_for_future_confirmation",
        "selected_candidate": "M0-SW",
        "selected_chain": "M0-SW",
        "chain_kind": "pure_oi",
        "algorithmic_oi_status": "pass",
        "product_policy_status": "not_applicable",
    }
    rows = [
        {
            "candidate_id": "M0-SW",
            "dataset": dataset,
            "replicate": replicate,
            "budget": budget,
            "head": head,
            "contrast_pp": -0.5 if head != "linear" else 0.5,
        }
        for dataset in datasets
        for replicate in statistics.FOOD_REPLICATES
        for budget in (32, 64)
        for head in statistics.REFERENCE_HEADS
    ]
    result = statistics.confirmation_gates(
        rows,
        development_lock=lock,
        expected_datasets=datasets,
        expected_replicates=statistics.FOOD_REPLICATES,
        n_resamples=200,
    )
    assert result["status"] == "pass"
    assert result["reselection_performed"] is False
    with pytest.raises(ValueError, match="exactly five"):
        statistics.confirmation_gates(
            (row for row in rows if row["dataset"] != datasets[-1]),
            development_lock=lock,
            expected_datasets=datasets[:-1],
            expected_replicates=statistics.FOOD_REPLICATES,
            n_resamples=20,
        )
    extra_dataset = "new-5"
    extra_rows = [
        {
            **row,
            "dataset": extra_dataset,
        }
        for row in rows
        if row["dataset"] == datasets[0]
    ]
    with pytest.raises(ValueError, match="exactly five"):
        statistics.confirmation_gates(
            [*rows, *extra_rows],
            development_lock=lock,
            expected_datasets=(*datasets, extra_dataset),
            expected_replicates=statistics.FOOD_REPLICATES,
            n_resamples=20,
        )
    changed = deepcopy(rows)
    changed[0]["candidate_id"] = "M1-SW"
    with pytest.raises(ValueError, match="exactly the locked chain"):
        statistics.confirmation_gates(
            changed,
            development_lock=lock,
            expected_datasets=datasets,
            expected_replicates=statistics.FOOD_REPLICATES,
            n_resamples=20,
        )
    missing = rows[:-1]
    with pytest.raises(ValueError, match="exact complete"):
        statistics.confirmation_gates(
            missing,
            development_lock=lock,
            expected_datasets=datasets,
            expected_replicates=statistics.FOOD_REPLICATES,
            n_resamples=20,
        )
    duplicate = rows + [deepcopy(rows[0])]
    with pytest.raises(ValueError, match="duplicate confirmation identity"):
        statistics.confirmation_gates(
            duplicate,
            development_lock=lock,
            expected_datasets=datasets,
            expected_replicates=statistics.FOOD_REPLICATES,
            n_resamples=20,
        )
