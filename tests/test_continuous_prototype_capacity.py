"""Focused tests for the frozen continuous prototype-capacity experiment."""

import numpy as np

from experiments.continuous_prototype_capacity import run
from experiments.continuous_save_ranking import run as save


def test_protocol_closes_exact_k4_k8_pairs_and_gates():
    protocol = run._load_protocol()
    run._validate_protocol_against_code(protocol)
    specs = run._method_specs(protocol)

    assert list(specs) == [
        "save8_k4",
        "save8_k8",
        "rff64_mean8_k4",
        "rff64_mean8_k8",
        "save_rffmean_k4",
        "save_rffmean_k8",
        "save_rffmean_residual_k4",
        "save_rffmean_residual_k8",
    ]
    assert protocol["continuous_oi"]["prototype_counts"] == [4, 8]
    for pair in protocol["capacity_gate"]["pairs"]:
        left = specs[pair["k4_method_id"]]
        right = specs[pair["k8_method_id"]]
        assert left["prototypes_per_cell"] == 4
        assert right["prototypes_per_cell"] == 8
        assert left["geometry"] == right["geometry"]
        assert left.get("blocks") == right.get("blocks")


def test_generator_remains_bit_exact_with_parent_save_panel():
    protocol = run._load_protocol()
    current = run.generate_panel(protocol, 1009, "triangle_wave")
    parent = save.generate_panel(protocol, 1009, "triangle_wave")

    np.testing.assert_array_equal(current["y_train"], parent["y_train"])
    np.testing.assert_array_equal(current["y_test"], parent["y_test"])
    for left, right in zip(current["candidates"], parent["candidates"]):
        assert left["candidate_id"] == right["candidate_id"]
        np.testing.assert_array_equal(left["X_train"], right["X_train"])
        np.testing.assert_array_equal(left["primitive_train"], right["primitive_train"])


def test_paired_k_methods_use_bit_exact_geometry_before_clustering():
    protocol = run._load_protocol()
    specs = run._method_specs(protocol)
    panel = run.generate_panel(protocol, 1103, "triangle_wave")
    candidate = panel["candidates"][0]
    args = (
        candidate["X_train"][:240],
        candidate["X_train"][240:320],
        candidate["primitive_train"][:240],
        candidate["primitive_train"][240:320],
        panel["y_train"][:240],
        17,
    )

    for k4_id, k8_id in (
        ("save8_k4", "save8_k8"),
        ("rff64_mean8_k4", "rff64_mean8_k8"),
        ("save_rffmean_k4", "save_rffmean_k8"),
        ("save_rffmean_residual_k4", "save_rffmean_residual_k8"),
    ):
        k4_train, k4_holdout, k4_diagnostics = (
            run.multiview._prepare_fold_geometry(specs[k4_id], *args)
        )
        k8_train, k8_holdout, k8_diagnostics = (
            run.multiview._prepare_fold_geometry(specs[k8_id], *args)
        )
        np.testing.assert_array_equal(k4_train, k8_train)
        np.testing.assert_array_equal(k4_holdout, k8_holdout)
        assert k4_diagnostics == k8_diagnostics


def test_k8_resolves_all_prototypes_and_support_covers_training_rows():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 1009, "triangle_wave")
    candidate = panel["candidates"][0]
    result = run.evaluate_selector(
        "save_rffmean_k8",
        candidate["X_train"],
        candidate["primitive_train"],
        panel["y_train"],
        31,
    )

    assert len(result["folds"]) == 4
    for fold in result["folds"]:
        diagnostics = fold["prototype_capacity_diagnostics"]
        assert diagnostics["configured_prototypes_per_cell"] == 8
        assert diagnostics["minimum_resolved_prototypes_per_cell"] == 8
        assert diagnostics["maximum_resolved_prototypes_per_cell"] == 8
        assert diagnostics["prototype_count"] == 64
        assert diagnostics["support_total"] == 375
        assert diagnostics["occupied_prototype_count"] <= 64
        assert diagnostics["empty_assignment_prototype_count"] >= 0
        assert diagnostics["lloyd_empty_prototype_count"] >= 0
        assert diagnostics["support_minimum"] >= 1


def test_k4_and_k8_record_distinct_capacity_but_same_fold_schema():
    protocol = run._load_protocol()
    panel = run.generate_panel(protocol, 1201, "linear")
    candidate = panel["candidates"][0]
    common = (
        candidate["X_train"],
        candidate["primitive_train"],
        panel["y_train"],
        37,
    )

    k4 = run.evaluate_selector("save8_k4", *common)
    k8 = run.evaluate_selector("save8_k8", *common)

    assert len(k4["folds"]) == len(k8["folds"]) == 4
    for left, right in zip(k4["folds"], k8["folds"]):
        assert left["geometry_diagnostics"] == right["geometry_diagnostics"]
        assert left["prototype_capacity_diagnostics"]["prototype_count"] == 32
        assert right["prototype_capacity_diagnostics"]["prototype_count"] == 64
        assert left["prototype_capacity_diagnostics"]["support_total"] == 375
        assert right["prototype_capacity_diagnostics"]["support_total"] == 375


def test_capacity_gate_requires_selective_triangle_gain_and_resolved_k8():
    protocol = run._load_protocol()
    families = protocol["families"]

    def method(overall, regret, values):
        return {
            "mean_spearman": overall,
            "mean_regret_r2": regret,
            "by_family": {
                family: {"mean_spearman": values[family]}
                for family in families
            },
        }

    baseline_values = {family: 0.60 for family in families}
    passing_values = dict(baseline_values)
    passing_values["triangle_wave"] = 0.90
    failing_values = dict(baseline_values)
    methods = {}
    capacity = {}
    for pair in protocol["capacity_gate"]["pairs"]:
        left = pair["k4_method_id"]
        right = pair["k8_method_id"]
        methods[left] = method(0.60, 0.15, baseline_values)
        values = passing_values if right == "save_rffmean_k8" else failing_values
        methods[right] = method(0.60, 0.16, values)
        capacity[left] = {"minimum_resolved_prototypes_per_cell": 4}
        capacity[right] = {"minimum_resolved_prototypes_per_cell": 8}
    summary = {
        "methods": methods,
        "prototype_capacity_summary": capacity,
    }

    gate = run._capacity_gate(protocol, summary)

    assert gate["status"] == "pass"
    assert gate["candidates"]["save_rffmean_k8"]["status"] == "pass"
    assert gate["candidates"]["save8_k8"]["status"] == "fail"


def test_universal_gate_is_not_weakened_for_k8_candidates():
    protocol = run._load_protocol()
    assert protocol["success_gate"]["baseline_method_id"] == "save8_k4"
    assert set(protocol["success_gate"]["candidate_method_ids"]) == {
        "save8_k8",
        "rff64_mean8_k8",
        "save_rffmean_k8",
        "save_rffmean_residual_k8",
    }
    assert protocol["success_gate"]["maximum_preservation_drop"] == 0.10
    assert protocol["success_gate"]["minimum_periodic_improvement"] == {
        "sinusoid": 0.20,
        "triangle_wave": 0.15,
    }
