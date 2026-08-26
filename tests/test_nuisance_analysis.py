"""Focused, cheap tests for nuisance analysis/report plumbing."""

from __future__ import annotations

import json

import numpy as np

from experiments.nuisance_conditioned_distance.analysis import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    binary_detection_metrics,
    build_analysis_summary,
    canonical_json,
    classify_gate,
    downstream_selection_metrics,
    evaluate_product_gates,
    evaluate_historical_gates,
    paired_bootstrap_ci,
    paired_seed_block_bootstrap,
    pooled_reference_metrics,
    select_promotion,
    source_hashes,
    stable_auprc,
    tie_aware_auroc,
)
from experiments.nuisance_conditioned_distance.analysis import _artifact_completeness
from experiments.nuisance_conditioned_distance.analysis import (
    _FOOD_ARMS,
    _FOOD_BUDGETS,
    _FOOD_MODELS,
    _FOOD_REPLICATES,
    _FOOD_SELECTOR_CANDIDATES,
    _RUNTIME_BUDGETS,
    _RUNTIME_MODELS,
    _RUNTIME_REPEATS,
    _food_artifact_completeness,
    _normalise_records,
    _robustness_completeness,
    _runtime_artifact_completeness,
    _runtime_benchmark_summary,
)


def test_seed_block_bootstrap_pools_rows_and_reuses_draws() -> None:
    rows_a = [
        {"candidate_id": "A", "seed": 0, "condition": "x", "value": 0.0},
        {"candidate_id": "A", "seed": 0, "condition": "y", "value": 10.0},
        {"candidate_id": "A", "seed": 1, "condition": "x", "value": 2.0},
        {"candidate_id": "A", "seed": 1, "condition": "y", "value": 4.0},
    ]
    rows_b = [dict(row, candidate_id="B", value=row["value"] + 1.0) for row in rows_a]
    extractor = lambda row: row["value"]
    first = paired_seed_block_bootstrap({"A": rows_a, "B": rows_b}, extractor, n_resamples=64)
    second = paired_seed_block_bootstrap({"A": rows_a, "B": rows_b}, extractor, n_resamples=64)
    assert first == second
    assert first["A"]["n_blocks"] == 2
    assert first["A"]["n"] == 4
    assert first["B"]["estimate"] - first["A"]["estimate"] == 1.0


def test_seed_block_bootstrap_drops_incomplete_seed_blocks() -> None:
    rows_a = [
        {"candidate_id": "A", "seed": 0, "case_id": "a", "value": 1.0},
        {"candidate_id": "A", "seed": 0, "case_id": "b", "value": 2.0},
        {"candidate_id": "A", "seed": 1, "case_id": "a", "value": 3.0},
    ]
    rows_b = [
        {"candidate_id": "B", "seed": 0, "case_id": "a", "value": 2.0},
        {"candidate_id": "B", "seed": 0, "case_id": "b", "value": 3.0},
        {"candidate_id": "B", "seed": 1, "case_id": "a", "value": 4.0},
        {"candidate_id": "B", "seed": 1, "case_id": "b", "value": 5.0},
    ]
    result = paired_seed_block_bootstrap(
        {"A": rows_a, "B": rows_b}, lambda row: row["value"], n_resamples=32
    )
    assert result["A"]["n_blocks"] == 1
    assert result["A"]["n"] == result["B"]["n"] == 2


def test_bootstrap_defaults_are_frozen_and_candidate_baseline_is_paired() -> None:
    result = paired_bootstrap_ci([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], n_resamples=128)
    assert set(result) == {"candidate"}
    assert result["candidate"]["estimate"] == -1.0
    assert BOOTSTRAP_RESAMPLES == 10_000
    assert BOOTSTRAP_SEED == 20_260_812


def test_nonlinear_reference_rows_are_pooled_not_equal_cell_averages() -> None:
    rows = [
        {"reference": "quadratic", "reference_accuracy": 1.0},
        {"reference": "quadratic", "reference_accuracy": 0.0},
        {"reference": "quadratic", "reference_accuracy": 0.5},
    ]
    result = pooled_reference_metrics(rows, heads=("quadratic",))
    assert result["quadratic"]["estimate"] == 0.5
    assert result["quadratic"]["n"] == 3


def test_detection_metrics_are_tie_aware_and_undefined_cells_are_explicit() -> None:
    assert tie_aware_auroc([0, 1], [0.5, 0.5]) == 0.5
    # Frozen archived semantics retain serialized order for equal scores.
    assert stable_auprc([0, 1], [0.5, 0.5]) == 0.5
    assert stable_auprc([1, 0], [0.5, 0.5]) == 1.0
    undefined = binary_detection_metrics([], [])
    assert undefined["status"] == "undefined"
    assert undefined["auroc"] is None


def test_pooled_pair_auprc_is_not_a_mean_of_per_row_auprc() -> None:
    rows = []
    for candidate in ("B", "C"):
        rows.extend(
            [
                {
                    "candidate_id": candidate,
                    "seed": 0,
                    "case_id": "first",
                    "pairwise": {
                        "0->1": {"pairwise_index": 0.1},
                        "1->0": {"pairwise_index": 0.9},
                    },
                    "pair_overlap_label": {"0->1": 1, "1->0": 0},
                },
                {
                    "candidate_id": candidate,
                    "seed": 1,
                    "case_id": "second",
                    "pairwise": {
                        "0->1": {"pairwise_index": 0.2},
                        "1->0": {"pairwise_index": 0.3},
                    },
                    "pair_overlap_label": {"0->1": 0, "1->0": 1},
                },
            ]
        )
    summary = build_analysis_summary(rows)
    pooled = summary["candidates"]["B"]["genuine_overlap"]["auprc"]
    # Per-row APs are 1.0 and 0.5 (mean .75); pooled stable-order AP is
    # (1 + 2/3) / 2 = 5/6.
    assert np.isclose(pooled, 5.0 / 6.0)
    assert summary["candidates"]["C"]["pair_auprc_delta"]["estimate"] == 0.0


def test_calibration_uses_continuous_pair_overlap_truth() -> None:
    rows = []
    for candidate in ("B", "C"):
        rows.append(
            {
                "candidate_id": candidate,
                "seed": 0,
                "case_id": "rho-quarter",
                "pairwise": {
                    "0->1": {"pairwise_index": 0.5},
                    "1->0": {"pairwise_index": 0.5},
                },
                "truth": {
                    "pair_overlap_label": [[0, 1], [1, 0]],
                    "pair_overlap": [[0.0, 0.25], [0.25, 0.0]],
                },
            }
        )
    summary = build_analysis_summary(rows)
    # Evidence is .5 and continuous rho is .25, so the pooled Brier is .0625;
    # a binary-label shortcut would incorrectly report .25.
    assert np.isclose(summary["candidates"]["B"]["genuine_overlap"]["brier"], 0.0625)


def test_gate_boundaries_are_not_tuned() -> None:
    assert classify_gate(0.05, 0.05, direction="le") == "pass"
    assert classify_gate(0.05, 0.05, direction="ge") == "pass"
    assert classify_gate(0.0, 0.0, direction="le", strict=True) == "fail"
    assert classify_gate(None, 0.0) == "inconclusive"


def test_historical_stable_and_strata_crossing_zero_are_inconclusive() -> None:
    summary = {
        "candidates": {
            "B": {"genuine_overlap": {"fpr": 0.0}},
            "C": {
                "genuine_overlap": {"fpr": 0.0},
                "fpr_delta": {"lower": -0.01, "upper": 0.01},
                "pair_auroc_delta": {"lower": 0.0, "upper": 0.0},
                "pair_auprc_delta": {"lower": 0.0, "upper": 0.0},
                "fnr_delta": {"lower": -0.01, "upper": 0.01},
                "refinement_fnr_delta_vs_raw": {"lower": -0.01, "upper": 0.01},
                "clean_mae_delta": {"lower": -0.01, "upper": 0.01},
                "brier_delta": {"lower": -0.01, "upper": 0.01},
                "nuisance_spearman": 1.0,
                "nuisance_ordering_rate": 1.0,
                "ordering_loss_vs_baseline": {"lower": -0.01, "upper": 0.01},
                "family_drift": {"worsening_upper": 0.0, "improvement_upper": {"a": -0.01, "b": -0.01, "c": 0.0}},
                "stable_shift_upper": {"clean_mae": {"lower": -0.01, "upper": 0.01}},
                "strata_fnr_upper": {"(2, 'balanced')": {"lower": -0.01, "upper": 0.01}},
            },
        }
    }
    gates = evaluate_historical_gates(summary)["C"]["gates"]
    statuses = {gate["name"]: gate["status"] for gate in gates}
    assert statuses["stable_shift"] == "inconclusive"
    assert statuses["strata_fnr"] == "inconclusive"


def test_promotion_uses_simplicity_order_and_excludes_f_g() -> None:
    summary = {
        "candidates": {
            "C": {"nuisance_linear_regret_upper": 0.0100, "robustness_auc": 0.2000},
            "D": {"nuisance_linear_regret_upper": 0.0105, "robustness_auc": 0.2005},
            "E": {"nuisance_linear_regret_upper": 0.0110, "robustness_auc": 0.2010},
            "F": {"nuisance_linear_regret_upper": -1.0, "robustness_auc": -1.0},
            "G": {"nuisance_linear_regret_upper": -2.0, "robustness_auc": -2.0},
        }
    }
    decision = select_promotion(summary)
    assert decision["selected_candidate"] == "C"
    assert "F" in decision["excluded"] and "G" in decision["excluded"]


def test_screen_promotion_rejects_partial_or_unverifiable_artifacts() -> None:
    summary = {
        "stage": "screen",
        "artifact_completeness": {
            "stage": "screen",
            "status": "fail",
            "failures": ["artifact_status='partial'"],
        },
        "candidates": {
            candidate: {
                "nuisance_linear_regret_upper": 0.01,
                "robustness_auc": 0.1,
            }
            for candidate in ("C", "D", "E")
        },
    }
    decision = select_promotion(summary)
    assert decision["selected_candidate"] is None
    assert decision["status"] == "inconclusive"
    assert all("incomplete" in decision["excluded"][candidate] for candidate in ("C", "D", "E"))


def test_complete_screen_manifest_uses_screen_seed_override_and_exact_grid() -> None:
    families = ("shared_low_rank", "clustered_multimodal", "heteroscedastic_multiplicative")
    conditions = ("separated_balanced_stable", "separated_balanced_shift", "overlap_half_balanced_stable", "overlap_half_balanced_shift")
    rows = []
    for family in families:
        for condition in conditions:
            for strength in (0.0, 1.0, 2.0):
                for seed in range(10):
                    for k in (2, 8):
                        case_id = f"{family}__{condition}__nuis-{strength:g}__k-{k}__seed-{seed}"
                        for candidate in ("A", "B", "C", "D", "E", "F"):
                            rows.append(
                                {
                                    "case_id": case_id,
                                    "candidate_id": candidate,
                                    "family": family,
                                    "condition": condition,
                                    "nuisance_strength": strength,
                                    "seed": seed,
                                    "k": k,
                                }
                            )
    manifest = {
        "stage": "screen",
        "artifact_status": "completed",
        "early_stop": None,
        "stage_case_count": 720,
        "stage2_constants": {
            "families": list(families),
            "conditions": [[name] for name in (*conditions, "overlap_quarter_imbalanced_stable", "overlap_full_imbalanced_stable")],
            "nuisance_strengths": [0.0, 1.0, 2.0],
            "seeds": list(range(1000, 1020)),
            "k_values": [2, 8],
        },
    }
    result = _artifact_completeness(rows, manifest)
    assert result["status"] == "pass", result


def test_summary_is_json_safe_and_deterministic_for_empty_rows() -> None:
    first = build_analysis_summary([])
    second = build_analysis_summary([])
    assert canonical_json(first) == canonical_json(second)
    assert first["n_rows"] == 0
    assert first["candidates"] == {}
    json.dumps(first, allow_nan=False)


def test_raw_container_normalization_keeps_parity_out_of_candidate_rows() -> None:
    payload = {
        "schema_version": 1,
        "provenance": {"starting_commit": "abc"},
        "rows": [{"candidate_id": "A", "case_id": "x", "score": 0.1}],
        "g_policy_rows": [{"model": "m0", "budget": 64, "repeat": 0}],
        "baseline_parity_rows": [{"candidate_id": "A", "delta": 0.0}],
    }
    records = _normalise_records(payload)
    assert len(records) == 2
    assert {record.get("candidate") for record in records} == {"A", "G"}
    assert all("delta" not in record for record in records)


def test_downstream_requires_explicit_selection_for_regret() -> None:
    rows = [
        {"candidate_id": "A", "case_id": "x", "score": 0.1, "references": {"linear": 0.8}},
        {"candidate_id": "B", "case_id": "x", "score": 0.2, "references": {"linear": 0.9}},
    ]
    result = downstream_selection_metrics(rows)
    assert result["A"]["heads"]["linear"]["regret"]["status"] == "undefined"


def test_synthetic_selection_uses_three_families_and_separates_clean_nuisance() -> None:
    rows = []
    families = ("shared_low_rank", "clustered_multimodal", "heteroscedastic_multiplicative")
    for candidate in ("B", "C"):
        for seed in (0, 1):
            for strength in (0.0, 1.0):
                for family_index, family in enumerate(families):
                    rows.append(
                        {
                            "candidate_id": candidate,
                            "family": family,
                            "condition": "separated_balanced_stable",
                            "seed": seed,
                            "k": 2,
                            "nuisance_strength": strength,
                            "candidate_score": float(family_index + (1 if family_index == 2 else 0)),
                            "references": {"linear_logistic": 0.5 + 0.1 * family_index},
                        }
                    )
    summary = build_analysis_summary(rows, manifest={"stage": "screen"})
    metrics = summary["candidates"]["C"]["selection_metrics"]
    assert metrics["nuisance_linear_regret"]["n_blocks"] == 2
    assert metrics["clean_linear_regret"]["n_blocks"] == 2
    assert metrics["nuisance_linear_regret"]["estimate"] == 0.0
    assert metrics["clean_linear_regret"]["estimate"] == 0.0


def test_synthetic_rank_auc_uses_complete_three_family_strength_curves() -> None:
    rows = []
    families = ("shared_low_rank", "clustered_multimodal", "heteroscedastic_multiplicative")
    references = (0.5, 0.6, 0.7)
    for candidate in ("B", "C"):
        for strength in (0.0, 1.0, 2.0):
            for index, family in enumerate(families):
                score_index = index if candidate == "B" else (2 - index)
                rows.append(
                    {
                        "candidate_id": candidate,
                        "family": family,
                        "condition": "separated_balanced_stable",
                        "seed": 0,
                        "k": 2,
                        "nuisance_strength": strength,
                        "candidate_score": float(score_index),
                        "references": {"linear_logistic": references[index]},
                    }
                )
    summary = build_analysis_summary(rows)
    b_rank = summary["candidates"]["B"]["selection_metrics"]["heads"]["linear"]["rank_auc"]
    c_rank = summary["candidates"]["C"]["selection_metrics"]["heads"]["linear"]["rank_auc"]
    assert np.isclose(b_rank["estimate"], 1.0)
    assert np.isclose(c_rank["estimate"], -1.0)


def test_food_probe_surfaces_stay_distinct_and_product_contrasts_are_paired() -> None:
    rows = []
    for candidate, bonus in (("B", 0.0), ("C", 0.02), ("full_probe", 0.03), ("G", 0.01)):
        for replicate in (0, 1):
            for arm in ("baseline", "nuisance_full", "nonlinearity_full"):
                for budget in (64, 68):
                    for model_index, model in enumerate(("m0", "m1", "m2")):
                        rows.append(
                            {
                                "candidate_id": candidate,
                                "model": model,
                                "backbone": model,
                                "replicate": replicate,
                                "arm": arm,
                                "budget": budget,
                                # Seed is an execution detail; the frozen
                                # panel identity remains replicate/arm/budget
                                # across all ten model rows.
                                "seed": 42 + replicate + model_index,
                                "candidate_score": float(model_index + bonus),
                                "outer_wall_seconds": 2.0 if candidate != "full_probe" else None,
                                "wall_seconds": 1.0 if candidate == "full_probe" else None,
                                "references": {"linear": 0.5 + 0.1 * model_index, "quadratic": 0.4 + 0.1 * model_index, "knn": 0.4 + 0.1 * model_index, "rbf": 0.4 + 0.1 * model_index},
                            }
                        )
    summary = build_analysis_summary(rows, manifest={"stage": "food101"})
    selection = summary["candidates"]["C"]["selection_metrics"]
    assert summary["candidates"]["full_probe"]["selection_metrics"]["heads"]["linear"]["status"] == "defined"
    assert selection["heads"]["linear"]["rank_auc"]["status"] == "defined"
    assert "full_probe" in summary["candidates"]
    assert "G" in summary["candidates"]
    assert "G_probe_component" not in summary["candidates"]
    assert selection["product_contrasts"]["clean_linear_regret_vs_probe"]["status"] == "defined"
    assert summary["candidates"]["C"]["clean_linear_regret_vs_probe"]["status"] == "defined"
    assert summary["candidates"]["C"]["timing_vs_full_probe"]["baseline"]["estimate"] == 2.0


def test_robustness_auc_excludes_genuine_overlap_cells() -> None:
    rows = []
    for candidate in ("B", "C"):
        for strength, evidence in ((0.0, 0.1), (1.0, 0.2), (2.0, 0.3)):
            directed = {
                "0->1": {"pairwise_index": 1.0 - evidence},
                "1->0": {"pairwise_index": 1.0 - evidence},
            }
            if candidate == "C":
                directed = {
                    "0->1": {"pairwise_index": 1.0 - evidence - 0.01},
                    "1->0": {"pairwise_index": 1.0 - evidence - 0.01},
                }
            rows.append(
                {
                    "candidate_id": candidate,
                    "family": "shared_low_rank",
                    "condition": "separated_balanced_stable",
                    "seed": 0,
                    "k": 2,
                    "balance": "balanced",
                    "nuisance_strength": strength,
                    "score": 1.0 - evidence - (0.01 if candidate == "C" else 0.0),
                    "pairwise": directed,
                }
            )
            rows.append(
                {
                    "candidate_id": candidate,
                    "family": "shared_low_rank",
                    "condition": "overlap_full_imbalanced_stable",
                    "seed": 0,
                    "k": 2,
                    "balance": "imbalanced",
                    "nuisance_strength": strength,
                    "score": 0.0,
                }
            )
    summary = build_analysis_summary(rows)
    auc = summary["candidates"]["C"]["nuisance_robustness_auc"]
    assert np.isclose(auc["estimate"], 0.01)


def test_robustness_requires_both_directions_and_all_three_strengths() -> None:
    def row(candidate: str, strength: float, *, missing_direction: bool = False) -> dict:
        pairwise = {"0->1": {"pairwise_index": 0.9}, "1->0": {"pairwise_index": 0.9}}
        if missing_direction:
            pairwise.pop("1->0")
        return {
            "candidate_id": candidate,
            "family": "shared_low_rank",
            "condition": "separated_balanced_stable",
            "seed": 0,
            "k": 2,
            "nuisance_strength": strength,
            "pairwise": pairwise,
        }

    # One missing directed pair makes that cell unavailable, so the whole
    # matched stratum is excluded rather than falling back to score evidence.
    missing = [row(candidate, strength, missing_direction=(strength == 1.0))
               for candidate in ("B", "C") for strength in (0.0, 1.0, 2.0)]
    summary = build_analysis_summary(missing)
    assert summary["candidates"]["C"]["nuisance_robustness_auc"]["status"] == "undefined"

    # A partial nuisance curve is also excluded; exact frozen strengths are
    # required for both candidate and baseline.
    incomplete = [row(candidate, strength)
                  for candidate in ("B", "C") for strength in (0.0, 1.0)]
    summary = build_analysis_summary(incomplete)
    assert summary["candidates"]["C"]["nuisance_robustness_auc"]["status"] == "undefined"


def _screen_lock(candidate: str = "C", sha: str = "screen-sha") -> dict:
    return {
        "status": "promoted_for_full_evaluation",
        "decision_stage": "screen",
        "selected_candidate": candidate,
        "sha256": sha,
    }


def _full_selection_summary(
    *,
    historical_status: str = "pass",
    product_status: str = "pass",
    lock: dict | None = None,
) -> dict:
    lock = _screen_lock() if lock is None else lock
    candidate_metrics = {
        candidate: {
            "nuisance_linear_regret_upper": 0.01 + index * 0.001,
            "robustness_auc": 0.10 + index * 0.01,
        }
        for index, candidate in enumerate(("C", "D", "E"))
    }
    return {
        "stage": "full",
        "locked_candidate": lock.get("selected_candidate"),
        "artifact_completeness": {"status": "pass"},
        "robustness_completeness": {"status": "pass"},
        "runtime_benchmark": {"completeness": {"status": "pass"}},
        "screen_promotion": lock,
        "candidates": candidate_metrics,
        "historical_gates": {
            "C": {"status": historical_status, "gates": []},
            "D": {"status": "pass", "gates": []},
            "E": {"status": "pass", "gates": []},
        },
        "product_gates": {
            "C": {"status": product_status, "gates": []},
            "D": {"status": "pass", "gates": []},
            "E": {"status": "pass", "gates": []},
        },
    }


def test_full_stage_honors_screen_lock_without_runner_up() -> None:
    passing = select_promotion(_full_selection_summary())
    assert passing["selected_candidate"] == "C"
    assert passing["status"] == "pass"
    assert passing["eligible"] == ["C"]
    assert passing["excluded"]["D"] == "not the screen-locked candidate"
    assert passing["excluded"]["E"] == "not the screen-locked candidate"

    failed = select_promotion(_full_selection_summary(historical_status="fail"))
    assert failed["selected_candidate"] == "C"
    assert failed["status"] == "fail"
    assert failed["excluded"]["D"] == "not the screen-locked candidate"

    missing = select_promotion(_full_selection_summary(product_status="inconclusive"))
    assert missing["selected_candidate"] is None
    assert missing["status"] == "inconclusive"
    assert missing["excluded"]["D"] == "not the screen-locked candidate"

    invalid_lock = select_promotion(
        _full_selection_summary(lock={"status": "pass", "decision_stage": "screen", "selected_candidate": "F", "sha256": "x"})
    )
    assert invalid_lock["selected_candidate"] is None
    assert invalid_lock["status"] == "inconclusive"


def test_food_partial_and_sha_mismatch_are_definite_completeness_failures() -> None:
    manifest = {
        "stage": "food101",
        "artifact_status": "completed",
        "configuration": {
            "models": list(_FOOD_MODELS),
            "replicates": list(_FOOD_REPLICATES),
            "arms": list(_FOOD_ARMS),
            "budgets": list(_FOOD_BUDGETS),
            "candidates": list(_FOOD_SELECTOR_CANDIDATES),
            "folds": 5,
            "k": 10,
            "seed": 42,
            "promotion_decision": {"selected_candidate": "C", "sha256": "wrong-sha"},
        },
        "baseline_parity": {"exact": True},
        "analysis_artifact_containers": {
            "baseline_parity_rows": [],
            "reference_rows": [],
        },
    }
    result = _food_artifact_completeness(
        [],
        manifest,
        screen_promotion=_screen_lock(sha="expected-sha"),
    )
    assert result["status"] == "fail"
    assert any("A n_rows" in failure for failure in result["failures"])
    assert any("promotion decision SHA" in failure for failure in result["failures"])


def test_food_reference_grid_is_budget_invariant_600_rows() -> None:
    reference_rows = [
        {
            "backbone": model,
            "replicate": replicate,
            "arm": arm,
            "head": head,
            "test_accuracy": 0.5,
        }
        for model in _FOOD_MODELS
        for replicate in _FOOD_REPLICATES
        for arm in _FOOD_ARMS
        for head in ("linear", "quadratic", "knn", "rbf")
    ]
    assert len(reference_rows) == 600
    manifest = {
        "stage": "food101",
        "artifact_status": "completed",
        "configuration": {
            "models": list(_FOOD_MODELS),
            "replicates": list(_FOOD_REPLICATES),
            "arms": list(_FOOD_ARMS),
            "budgets": list(_FOOD_BUDGETS),
            "candidates": list(_FOOD_SELECTOR_CANDIDATES),
            "folds": 5,
            "k": 10,
            "seed": 42,
            "promotion_decision": {
                "selected_candidate": "C",
                "sha256": "expected-sha",
            },
            "promoted_candidate_for_G": "C",
        },
        "baseline_parity": {"exact": True},
        "analysis_artifact_containers": {
            "baseline_parity_rows": [],
            "reference_rows": reference_rows,
        },
    }
    result = _food_artifact_completeness(
        [], manifest, screen_promotion=_screen_lock(sha="expected-sha")
    )
    assert result["status"] == "fail"  # selector/parity rows intentionally absent
    assert not any("reference_rows" in failure for failure in result["failures"])


def test_food_product_gate_never_falls_back_to_absolute_regret() -> None:
    summary = {
        "stage": "food101",
        "food_artifact_completeness": {"status": "pass"},
        "candidates": {
            "C": {
                "selection_metrics": {
                    "heads": {
                        "linear": {
                            "regret": {"upper": 0.0, "estimate": 0.0},
                        }
                    }
                },
                "timing": {},
            }
        },
    }
    gates = evaluate_product_gates(summary, candidate_ids=("C",))["C"]
    clean = next(gate for gate in gates["gates"] if gate["name"] == "clean_linear_regret_vs_probe")
    nuisance = next(gate for gate in gates["gates"] if gate["name"] == "nuisance_linear_regret_vs_probe")
    assert clean["status"] == "inconclusive"
    assert nuisance["status"] == "inconclusive"


def test_product_gate_consumes_food_determinism_verification_fallback() -> None:
    summary = {
        "stage": "food101",
        "food_artifact_completeness": {"status": "pass"},
        "baseline_parity": {"exact": True},
        "determinism": None,
        "determinism_verification": {"exact": True, "status": "pass"},
        "candidates": {"C": {"timing": {}}},
    }
    gates = evaluate_product_gates(summary, candidate_ids=("C",))["C"]["gates"]
    by_name = {gate["name"]: gate for gate in gates}
    assert by_name["baseline_parity"]["status"] == "pass"
    assert by_name["determinism"]["status"] == "pass"


def _robustness_rows(*, omit: tuple[str, str, int, float] | None = None) -> list[dict]:
    families = ("shared_low_rank", "clustered_multimodal", "heteroscedastic_multiplicative")
    conditions = ("separated_balanced_stable", "separated_balanced_shift")
    rows = []
    for candidate in ("B", "C", "D", "E"):
        for family in families:
            for condition in conditions:
                for seed in range(10):
                    for strength in (0.0, 1.0, 2.0):
                        if omit == (candidate, family, seed, strength):
                            continue
                        rows.append(
                            {
                                "candidate_id": candidate,
                                "family": family,
                                "condition": condition,
                                "nuisance_shift": condition.endswith("shift"),
                                "seed": seed,
                                "k": 2,
                                "nuisance_strength": strength,
                                "pairwise": {
                                    "0->1": {"pairwise_index": 0.9},
                                    "1->0": {"pairwise_index": 0.9},
                                },
                            }
                        )
    return rows


def _robustness_manifest() -> dict:
    return {
        "stage": "screen",
        "stage2_constants": {
            "families": [
                "shared_low_rank",
                "clustered_multimodal",
                "heteroscedastic_multiplicative",
            ],
            "conditions": [
                {"name": "separated_balanced_stable", "nuisance_shift": False},
                {"name": "separated_balanced_shift", "nuisance_shift": True},
            ],
            "k_values": [2],
            "seeds": list(range(1000, 1020)),
        },
    }


def test_robustness_completeness_requires_identical_strata_directions_and_strengths() -> None:
    complete = _robustness_completeness(
        {candidate: [row for row in _robustness_rows() if row["candidate_id"] == candidate] for candidate in ("B", "C", "D", "E")},
        _robustness_manifest(),
    )
    assert complete["status"] == "pass", complete

    missing_direction = _robustness_rows(omit=("C", "shared_low_rank", 0, 1.0))
    # Removing a whole row removes both directions; this is still required to
    # be surfaced as an incomplete strength/direction cell.
    result = _robustness_completeness(
        {candidate: [row for row in missing_direction if row["candidate_id"] == candidate] for candidate in ("B", "C", "D", "E")},
        _robustness_manifest(),
    )
    assert result["status"] == "fail"
    assert any("C incomplete strengths" in failure or "C missing" in failure for failure in result["failures"])

    missing_strength = _robustness_rows(omit=("D", "clustered_multimodal", 1, 2.0))
    result = _robustness_completeness(
        {candidate: [row for row in missing_strength if row["candidate_id"] == candidate] for candidate in ("B", "C", "D", "E")},
        _robustness_manifest(),
    )
    assert result["status"] == "fail"
    assert any("D incomplete strengths" in failure or "D missing" in failure for failure in result["failures"])

    missing_stratum = [
        row
        for row in _robustness_rows()
        if not (row["candidate_id"] == "E" and row["family"] == "heteroscedastic_multiplicative" and row["condition"] == "separated_balanced_shift" and row["seed"] == 9)
    ]
    result = _robustness_completeness(
        {candidate: [row for row in missing_stratum if row["candidate_id"] == candidate] for candidate in ("B", "C", "D", "E")},
        _robustness_manifest(),
    )
    assert result["status"] == "fail"
    assert any("E missing" in failure or "E robustness stratum" in failure for failure in result["failures"])


def test_runtime_contract_separates_capped_component_from_policy_g() -> None:
    lock = _screen_lock("C", "runtime-sha")
    timed_methods = ("A", "B", "E", "C", "full_probe", "capped_probe_component")
    rows = []
    for model_index, model in enumerate(_RUNTIME_MODELS):
        for budget_index, budget in enumerate(_RUNTIME_BUDGETS):
            for repeat in _RUNTIME_REPEATS:
                offset = (model_index + budget_index + repeat) % len(timed_methods)
                execution = timed_methods[offset:] + timed_methods[:offset]
                for order, method in enumerate(execution):
                    rows.append(
                        {
                            "candidate_id": method,
                            "model": model,
                            "budget": budget,
                            "repeat": repeat,
                            "execution_order": order,
                            "status": "ok",
                            "wall_seconds": 2.0 if method == "full_probe" else 1.0,
                            "cpu_seconds": 2.0 if method == "full_probe" else 1.0,
                        }
                    )
                rows.append(
                    {
                        "candidate_id": "G",
                        "model": model,
                        "budget": budget,
                        "repeat": repeat,
                        "status": "ok",
                        "wall_seconds": 3.0,
                    }
                )
    manifest = {
        "stage": "food101_runtime",
        "artifact_status": "completed",
        "configuration": {
            "models": list(_RUNTIME_MODELS),
            "budgets": list(_RUNTIME_BUDGETS),
            "repeats": list(_RUNTIME_REPEATS),
            "methods": list(timed_methods),
            "folds": 5,
            "k": 10,
            "seed": 42,
            "n_classes": 40,
            "promotion_decision": {"selected_candidate": "C", "sha256": "runtime-sha"},
        },
    }
    completeness = _runtime_artifact_completeness(rows, manifest, screen_promotion=lock)
    assert completeness["status"] == "pass", completeness
    assert completeness["expected_timed_rows"] == 1500
    assert completeness["expected_policy_rows"] == 250
    runtime = _runtime_benchmark_summary(rows, completeness)
    ratio = runtime["candidates"]["C"]["wall_ratio"]
    assert ratio["status"] == "defined"
    assert np.isclose(ratio["estimate"], 0.5)

    missing = list(rows)
    missing.pop()
    incomplete = _runtime_artifact_completeness(missing, manifest, screen_promotion=lock)
    assert incomplete["status"] == "fail"
