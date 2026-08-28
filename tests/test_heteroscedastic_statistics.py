"""Small, outcome-free tests for the frozen heteroscedastic statistics layer."""

from __future__ import annotations

import json
import itertools
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.heteroscedastic_distance_conditioning.statistics import (
    BONFERRONI_LEVEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    archived_pooled_auprc,
    binary_detection_metrics,
    build_smoke_decision,
    build_summary,
    candidate_eligibility,
    canonical_json,
    classify_gate,
    complete_seed_bootstrap,
    genuine_overlap_gates,
    family_drift,
    joint_bootstrap_draws,
    pooled_pair_detection,
    primary_mechanistic_claims,
    resource_summaries,
    select_lock,
    selector_panel_metrics,
    stable_shift_gates,
    validate_decision_prerequisite,
    validate_manifest_identity,
    validate_table_rows,
    within_draw_max,
)
from experiments.heteroscedastic_distance_conditioning import statistics as stats
from experiments.heteroscedastic_distance_conditioning import reporting


def test_frozen_bootstrap_draws_are_shared_and_byte_stable() -> None:
    first = joint_bootstrap_draws(4, n_resamples=32, seed=BOOTSTRAP_SEED)
    second = joint_bootstrap_draws(4, n_resamples=32, seed=BOOTSTRAP_SEED)
    assert first is second
    assert first.flags.writeable is False
    assert canonical_json(first.tolist()) == canonical_json(second.tolist())
    assert BOOTSTRAP_RESAMPLES == 10_000
    assert BONFERRONI_LEVEL == 0.9875


def test_complete_seed_bootstrap_drops_missing_seed_and_uses_paired_blocks() -> None:
    values = {"A": {0: [0.0, 2.0], 1: [4.0], 2: [100.0]}, "B": {0: [1.0, 3.0], 1: [5.0]}}
    result = complete_seed_bootstrap(values, n_resamples=64)
    assert result["A"]["n_blocks"] == 2
    assert result["A"]["n"] == result["B"]["n"] == 3
    assert result["B"]["estimate"] - result["A"]["estimate"] == 1.0


def test_within_draw_max_reuses_joint_draws() -> None:
    result = within_draw_max({"a": {0: -1.0, 1: -2.0}, "b": {0: -2.0, 1: -3.0}}, n_resamples=64)
    assert result["within_draw_max"] is True
    assert result["estimate"] == -1.5
    assert result["level"] == BONFERRONI_LEVEL


def test_detection_ties_and_archived_order_are_explicit() -> None:
    assert binary_detection_metrics([0, 1], [0.5, 0.5])["auroc"] == 0.5
    assert archived_pooled_auprc([0, 1], [0.5, 0.5]) == 0.5
    assert archived_pooled_auprc([1, 0], [0.5, 0.5]) == 1.0
    assert binary_detection_metrics([], []) ["status"] == "undefined"


def test_table_and_manifest_identity_reject_duplicates_and_wrong_hashes() -> None:
    rows = [{"case_id": "x", "candidate_id": "A"}, {"case_id": "x", "candidate_id": "A"}]
    check = validate_table_rows(rows, table="case_rows.jsonl")
    assert check["status"] == "fail"
    identity = validate_manifest_identity({"stage": "development", "artifact_status": "partial"}, expected_stage="development")
    assert identity["status"] == "fail"
    assert "artifact status" in identity["failures"][0]


def _selector_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate, scores in {"B": [1.0, 0.0, 0.0], "C": [0.0, 1.0, 0.0], "probe": [0.0, 1.0, 0.0]}.items():
        for scenario, score in zip(("H1", "H2", "T"), scores):
            rows.append({"seed": 0, "balance": "balanced", "count_level": "small", "nuisance_dim": 4, "k": 2, "signal_state": "linear_separated", "scenario": scenario, "candidate_id": candidate, "candidate_score": score, "reference_accuracies": {"linear": 0.9 if scenario == "H2" else 0.5, "quadratic": 0.5, "knn": 0.5, "rbf": 0.5}})
            rows.append({"seed": 0, "balance": "imbalanced", "count_level": "small", "nuisance_dim": 4, "k": 2, "signal_state": "linear_separated", "scenario": scenario, "candidate_id": candidate, "candidate_score": score, "reference_accuracies": {"linear": 0.9 if scenario == "H2" else 0.5, "quadratic": 0.5, "knn": 0.5, "rbf": 0.5}})
    return rows


def test_selector_missing_item_is_inconclusive_and_tie_order_is_fixed() -> None:
    rows = [row for row in _selector_rows() if row["scenario"] != "T"]
    result = selector_panel_metrics(rows, candidates=("B",), subpanel=("H1", "H2", "T"), n_resamples=32)
    assert result["n_complete_panels"] == 0  # only one seed/cell, but missing both count levels is not an outcome panel
    assert result["status"] == "inconclusive"


def test_gate_boundaries_are_strict_only_when_requested() -> None:
    assert classify_gate(0.05, 0.05)["status"] == "pass"
    assert classify_gate(0.05, 0.05, strict=True)["status"] == "fail"
    assert classify_gate(None, 0.0)["status"] == "inconclusive"


def test_lock_does_not_rank_smoke_and_reuses_existing_decision() -> None:
    summary = {"stage": "smoke", "candidates": {candidate: {} for candidate in ("P25", "P50-SW")}}
    decision = select_lock(summary)
    assert decision["selected_candidate"] is None
    assert decision["status"] == "inconclusive"
    existing = {"status": "locked", "selected_candidate": "P25", "locked_candidate": "P25"}
    reused = select_lock(summary, existing_decision=existing)
    assert reused["selected_candidate"] == "P25"
    assert reused["immutable_reuse"] is True


def test_decision_prerequisite_hashes_are_checked() -> None:
    payload = {"status": "locked", "protocol_sha256": "p", "code_identity_sha256": "c", "locked_candidate": "P25"}
    assert validate_decision_prerequisite(payload, protocol_hash="p", code_identity_hash="c", expected_status=("locked",))["status"] == "pass"
    assert validate_decision_prerequisite(payload, protocol_hash="wrong")["status"] == "fail"


def _canonical_pair_row(
    candidate: str,
    case_id: str,
    seed: int,
    scenario: str,
    balance: str,
    count_level: str,
    nuisance_dim: int,
    k: int,
    signal_state: str,
    source: int,
    target: int,
    evidence: float = 0.2,
    truth: float = 0.0,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "candidate_id": candidate,
        "seed": seed,
        "scenario": scenario,
        "balance": balance,
        "count_level": count_level,
        "nuisance_dim": nuisance_dim,
        "k": k,
        "signal_state": signal_state,
        "source_label": source,
        "target_label": target,
        "support": 10,
        "hits": 8,
        "sparse_adj_hits": 8,
        "pairwise_index": 1.0 - evidence,
        "evidence": evidence,
        "truth_overlap": truth,
        "truth_overlap_label": int(truth > 0.0),
        "exact_state_match": True,
        "status": "ok",
    }


def _pair_case(
    candidate: str,
    *,
    seed: int = 0,
    scenario: str = "S0",
    balance: str = "balanced",
    count_level: str = "small",
    nuisance_dim: int = 4,
    k: int = 2,
    signal_state: str = "linear_separated",
    evidence: float = 0.2,
    truth: float = 0.0,
) -> list[dict[str, object]]:
    case_id = f"{seed}:{scenario}:{balance}:{count_level}:{nuisance_dim}:{k}:{signal_state}"
    return [
        _canonical_pair_row(
            candidate, case_id, seed, scenario, balance, count_level,
            nuisance_dim, k, signal_state, source, target, evidence, truth,
        )
        for source in range(4)
        for target in range(4)
        if source != target
    ]


def _geometry_rows(candidate: str, *, seed: int = 0, value: float = 0.2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dimension, count_level, k, signal in itertools.product(
        (4, 32), ("small", "large"), (2, 8), ("linear_separated", "nonlinear_separated")
    ):
        rows.append({
            "candidate_id": candidate,
            "seed": seed,
            "scenario": "H2",
            "balance": "balanced",
            "nuisance_dim": dimension,
            "count_level": count_level,
            "k": k,
            "signal_state": signal,
            "geometry": {"cross_class_neighbor_impurity": value},
        })
    return rows


def test_structural_source_schema_requires_exact_runner_chain() -> None:
    manifest = {
        "determinism": {"status": "pass"},
        "authorization_hashes": {"smoke_decision_sha256": "s", "pre_screen_review_sha256": "p"},
        "structural_gates": {
            "exact_parity": True,
            "leakage_no_refit": True,
            "source_decisions": {
                "smoke_decision": {"sha256": "s", "status": "pass"},
                "pre_screen_review": {"sha256": "p", "status": "pass"},
            },
        },
    }
    result = stats._validated_structural_evidence(manifest, stage="development")
    assert result["exact_parity"]["status"] == "pass"
    assert result["leakage"]["status"] == "pass"
    assert result["review_identity"]["status"] == "pass"

    extra = dict(manifest)
    extra["structural_gates"] = dict(manifest["structural_gates"])
    extra["structural_gates"]["source_decisions"] = {
        **manifest["structural_gates"]["source_decisions"],
        "implementation_review": {"sha256": "s", "status": "pass"},
    }
    assert stats._validated_structural_evidence(extra, stage="development")["review_identity"]["status"] == "fail"


def test_smoke_decision_accepts_only_redacted_pair_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    pair = {
        "case_id": "smoke-case",
        "candidate_id": "A",
        "source_label": 0,
        "target_label": 1,
        "pair_structure": {
            name: {"present": True, "type": "int" if name in {"support", "hits", "sparse_adj_hits"} else "float", "finite": True}
            for name in ("support", "hits", "sparse_adj_hits", "pairwise_index", "evidence")
        },
        "exact_state_match": True,
        "outcomes_redacted": True,
        "status": "ok",
    }
    artifact = {"manifest": {
        "determinism": {"status": "pass"},
        "authorization_hashes": {"implementation_review_sha256": "i"},
        "structural_gates": {
            "exact_parity": True,
            "leakage_no_refit": True,
            "source_decisions": {"implementation_review": {"sha256": "i", "status": "go"}},
        },
    }, "tables": {"pair_rows.jsonl": [pair]}, "table_checks": {}}
    monkeypatch.setattr(stats, "validate_normalized_artifact", lambda *args, **kwargs: {"status": "pass", "failures": []})
    decision = build_smoke_decision(artifact)
    assert decision["status"] == "pass"
    assert "sha256" not in decision

    leaked = dict(pair)
    leaked.pop("pair_structure")
    leaked["evidence"] = 0.4
    artifact["tables"] = {"pair_rows.jsonl": [leaked]}
    assert build_smoke_decision(artifact)["status"] == "fail"
    generic_finite = dict(pair)
    generic_finite["finite"] = True
    artifact["tables"] = {"pair_rows.jsonl": [generic_finite]}
    assert build_smoke_decision(artifact)["status"] == "fail"


def test_pair_parser_keeps_all_12_directed_pairs_and_rejects_duplicate() -> None:
    rows = _pair_case("B")
    records, malformed = stats._pair_case_records(rows, candidates=("B",))
    assert not malformed
    assert len(records["B"][0]) == 1
    assert len(next(iter(records["B"][0].values()))["directions"]) == 12
    duplicate = rows + [dict(rows[0])]
    _records, malformed = stats._pair_case_records(duplicate, candidates=("B",))
    assert any("duplicate direction" in text for text in malformed)


def test_false_overlap_requires_all_12_blocks_and_separated_signal_filter() -> None:
    rows: list[dict[str, object]] = []
    scenarios = ("S2", "H2", "T", "C", "X", "ALL")
    axes = tuple(itertools.product((4, 32), ("small", "large"), (2, 8), ("linear_separated", "nonlinear_separated")))
    for candidate, offset in (("B", 0.0), ("P25", 0.01)):
        for scenario, balance, axis in itertools.product(scenarios, ("balanced", "imbalanced"), axes):
            dimension, count_level, k, signal = axis
            rows.extend(_pair_case(candidate, scenario=scenario, balance=balance, nuisance_dim=dimension, count_level=count_level, k=k, signal_state=signal, evidence=0.2 + offset))
    pooled = pooled_pair_detection(rows, candidates=("B", "P25"), n_resamples=16)
    assert pooled["P25"]["worst_block_false_overlap_upper"] is not None
    gates = genuine_overlap_gates(pooled, candidates=("P25",))
    assert gates["P25"]["gates"]["false_overlap"]["status"] in {"pass", "fail"}
    missing = [row for row in rows if not (row["candidate_id"] == "P25" and row["scenario"] == "ALL" and row["balance"] == "imbalanced")]
    missing_pooled = pooled_pair_detection(missing, candidates=("B", "P25"), n_resamples=16)
    missing_gates = genuine_overlap_gates(missing_pooled, candidates=("P25",))
    assert missing_gates["P25"]["gates"]["false_overlap"]["status"] == "inconclusive"


def test_negative_control_q3_uses_p50_not_legacy_baseline() -> None:
    rows: list[dict[str, object]] = []
    for candidate, value in (("L", 0.5), ("P50-SW", 0.5), ("W50-SW", 0.1), ("W50-CB", 0.2)):
        for scenario in ("S0", "S1", "S2", "H2"):
            rows.extend(_geometry_rows(candidate, value=value))
            for row in rows[-16:]:
                row["scenario"] = scenario
    controls = stats.negative_controls(rows, n_resamples=16)
    assert controls["q3"]["estimate"] == -0.4


def test_stable_shift_matches_exact_cells_and_all_directions() -> None:
    rows: list[dict[str, object]] = []
    for candidate, offset in (("B", 0.0), ("P25", 0.01)):
        for scenario, value in (("S0", 0.2 + offset), ("H2", 0.3 + offset), ("X", 0.4 + offset)):
            for dimension, count_level, k, signal in itertools.product((4, 32), ("small", "large"), (2, 8), ("linear_separated", "nonlinear_separated", "genuine_overlap_half")):
                rows.extend(_pair_case(candidate, scenario=scenario, nuisance_dim=dimension, count_level=count_level, k=k, signal_state=signal, evidence=value, truth=0.2 if signal == "genuine_overlap_half" else 0.0))
    result = stable_shift_gates(rows, candidates=("P25",), n_resamples=16)
    assert result["P25"]["balanced"]["X-H2:auroc"]["status"] == "defined"
    truncated = rows[:-1]
    incomplete = stable_shift_gates(truncated, candidates=("P25",), n_resamples=16)
    assert incomplete["P25"]["balanced"]["X-H2:auroc"]["status"] == "inconclusive"


def test_selector_rank_auc_requires_two_count_levels_and_exact_canonical_fields() -> None:
    rows: list[dict[str, object]] = []
    for seed in (0,):
        for balance, count_level, dimension, k, signal in itertools.product(("balanced", "imbalanced"), ("small", "large"), (4, 32), (2, 8), ("linear_separated", "nonlinear_separated", "genuine_overlap_half")):
            for candidate, selected in (("B", 0.2), ("P25", 0.3)):
                for scenario in ("H1", "H2", "T", "C", "X", "ALL"):
                    rows.append({
                        "seed": seed, "balance": balance, "count_level": count_level,
                        "nuisance_dim": dimension, "k": k, "signal_state": signal,
                        "scenario": scenario, "candidate_id": candidate,
                        "candidate_score": selected + (0.001 * (0 if scenario == "H1" else 1 if scenario == "H2" else 2 if scenario == "T" else 3 if scenario == "C" else 4 if scenario == "X" else 5)), "linear_probe_score": 0.2,
                        "reference_accuracies": {head: 0.8 if scenario == "H2" else 0.5 for head in ("linear", "quadratic", "knn", "rbf")},
                    })
    result = selector_panel_metrics(rows, candidates=("B", "P25"), n_resamples=16)
    assert result["n_complete_panels"] == 48
    assert result["candidates"]["P25"]["heads"]["linear"]["rank_auc"]["status"] == "defined"
    alias = dict(rows[0]); alias.pop("candidate_score"); alias["score"] = 0.2
    assert selector_panel_metrics(rows[:-1] + [alias], candidates=("B", "P25"), n_resamples=16)["status"] == "inconclusive"


def test_selector_rank_auc_does_not_report_partial_expected_seed_blocks() -> None:
    rows: list[dict[str, object]] = []
    for seed in (21000, 21001):
        for balance, count_level, dimension, k, signal in itertools.product(
            ("balanced", "imbalanced"), ("small", "large"), (4, 32), (2, 8),
            ("linear_separated", "nonlinear_separated", "genuine_overlap_half"),
        ):
            for scenario_index, scenario in enumerate(("H1", "H2", "T", "C", "X", "ALL")):
                rows.append({
                    "seed": seed, "balance": balance, "count_level": count_level,
                    "nuisance_dim": dimension, "k": k, "signal_state": signal,
                    "scenario": scenario, "candidate_id": "B",
                    "candidate_score": float(scenario_index) + seed / 100000.0,
                    "linear_probe_score": float(scenario_index),
                    "reference_accuracies": {
                        head: float(scenario_index) for head in ("linear", "quadratic", "knn", "rbf")
                    },
                    "total_rows": 160 if count_level == "small" else 640,
                })
    result = selector_panel_metrics(
        rows, candidates=("B",), n_resamples=16, expected_seed_count=12,
    )
    rank_auc = result["candidates"]["B"]["heads"]["linear"]["rank_auc"]
    assert rank_auc["status"] == "inconclusive"
    assert rank_auc["estimate"] is None
    assert rank_auc["n_blocks"] == 0


def test_selector_enrichment_checks_raw_child_aliases_only() -> None:
    case_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    references = {head: 0.5 for head in ("linear", "quadratic", "knn", "rbf")}
    for scenario, score in (("H1", 0.1), ("H2", 0.2), ("T", 0.3)):
        case_id = f"case-{scenario}"
        case_rows.append({
            "case_id": case_id, "candidate_id": "B", "seed": 0,
            "balance": "balanced", "count_level": "small", "nuisance_dim": 4,
            "k": 2, "signal_state": "linear_separated", "scenario": scenario,
            # This is immutable case metadata, not a selector child field.
            "linear_probe": {"score": 0.9},
        })
        selector_rows.append({
            "case_id": case_id, "candidate_id": "B",
            "candidate_score": score, "linear_probe_score": 0.2,
            "reference_accuracies": references,
        })

    enriched = stats._rows_by_table({
        "tables": {
            "case_rows.jsonl": case_rows,
            "selector_panels.jsonl": selector_rows,
        }
    })["selector_panels.jsonl"]
    assert enriched[0]["linear_probe"] == {"score": 0.9}
    result = selector_panel_metrics(
        enriched, candidates=("B",), subpanel=("H1", "H2", "T"), n_resamples=16,
    )
    assert result["n_complete_panels"] == 1
    assert result["status"] == "defined"

    # An alias present in the raw selector child row remains a hard failure,
    # even though the normalized join is allowed to carry case metadata.
    bad_selector_rows = [dict(row) for row in selector_rows]
    bad_selector_rows[0]["linear_probe"] = {"score": 0.9}
    bad_enriched = stats._rows_by_table({
        "tables": {
            "case_rows.jsonl": case_rows,
            "selector_panels.jsonl": bad_selector_rows,
        }
    })["selector_panels.jsonl"]
    assert selector_panel_metrics(
        bad_enriched, candidates=("B",), subpanel=("H1", "H2", "T"), n_resamples=16,
    )["status"] == "inconclusive"


def test_resource_rows_pair_by_resource_and_case_without_top_level_seed() -> None:
    rows: list[dict[str, object]] = []
    for resource_id in ("R0", "R1", "R2", "R3"):
        for candidate, multiplier in (("B", 1.0), ("P25", 1.1)):
            rows.append({
                "resource_id": resource_id, "case_id": f"seed-21000-{resource_id}",
                "candidate_id": candidate, "status": "ok",
                "total_wall_seconds": multiplier, "total_cpu_seconds": multiplier,
                "score_fixed_wall_seconds": multiplier, "score_fixed_cpu_seconds": multiplier,
                "peak_rss_bytes": 100.0 * multiplier,
            })
    result = resource_summaries(rows, candidates=("P25",), n_resamples=16)
    assert result["P25"]["status"] == "pass"
    assert result["P25"]["summaries"]["median_total"] == 1.1
    duplicate = rows + [dict(rows[0])]
    assert resource_summaries(duplicate, candidates=("P25",), n_resamples=16)["P25"]["status"] == "inconclusive"


def test_candidate_eligibility_requires_every_primary_gate() -> None:
    summary = {
        "artifact_identity": {"status": "pass"},
        "structural_gates": {"exact_parity": True, "determinism": True, "leakage": True, "review_identity": True},
        "genuine_overlap_gates": {"P25": {"status": "pass", "gates": {key: {"status": "pass"} for key in ("fpr", "auroc", "auprc", "fnr", "brier", "refinement_fnr_vs_raw", "clean_mae", "severity", "strata_fnr", "false_overlap")}}},
        "stable_shift_gates": {"P25": {balance: {f"{shifted}-{stable}:{metric}": {"gate": {"status": "pass"}} for shifted, stable in (("X", "H2"), ("ALL", "C")) for metric in ("auroc", "auprc", "brier", "clean_mae", "nuisance_drift")} for balance in ("balanced", "imbalanced")}},
        "resource_gates": {"P25": {"status": "pass", "summaries": {key: 1.0 for key in ("median_total", "p95_total", "median_score_fixed", "median_peak_memory", "p95_peak_memory", "individual_peak_memory")}, "gates": {key: {"status": "pass"} for key in ("median_total", "p95_total", "median_score_fixed", "median_peak_memory", "p95_peak_memory", "individual_peak_memory")}}},
        "family_drift": {"P25": {scenario: {"gate": {"status": "pass"}} for scenario in stats.SCENARIO_ORDER}},
        "candidates": {"P25": {
            "nuisance_linear_regret_vs_probe": {"lower": -0.001, "upper": 0.005},
            "nonlinear_retention": {head: {"lower": -0.001, "upper": 0.005} for head in ("quadratic", "knn", "rbf")},
        }},
    }
    assert candidate_eligibility(summary, candidates=("P25",))["P25"]["status"] == "pass"
    assert not {
        "exact_parity", "determinism", "leakage", "review_identity",
        "genuine_overlap", "stable_shift", "resources", "family_drift",
    } & set(summary["candidates"]["P25"])
    summary["candidates"]["P25"].pop("nonlinear_retention")
    assert candidate_eligibility(summary, candidates=("P25",))["P25"]["status"] == "inconclusive"


def test_false_overlap_is_lock_criterion_not_reliability_exclusion() -> None:
    summary = {
        "artifact_identity": {"status": "pass"},
        "structural_gates": {key: True for key in ("exact_parity", "determinism", "leakage", "review_identity")},
        "genuine_overlap_gates": {"P25": {
            "status": "fail",
            "gates": {
                **{key: {"status": "pass"} for key in ("fpr", "auroc", "auprc", "fnr", "brier", "refinement_fnr_vs_raw", "clean_mae", "severity", "strata_fnr")},
                "false_overlap": {"status": "fail", "upper": 0.25},
            },
        }},
        "stable_shift_gates": {"P25": {balance: {
            f"{shifted}-{stable}:{metric}": {"gate": {"status": "pass"}}
            for shifted, stable in (("X", "H2"), ("ALL", "C"))
            for metric in ("auroc", "auprc", "brier", "clean_mae", "nuisance_drift")
        } for balance in ("balanced", "imbalanced")}},
        "resource_gates": {"P25": {
            "status": "pass",
            "summaries": {key: 1.0 for key in ("median_total", "p95_total", "median_score_fixed", "median_peak_memory", "p95_peak_memory", "individual_peak_memory")},
            "gates": {key: {"status": "pass"} for key in ("median_total", "p95_total", "median_score_fixed", "median_peak_memory", "p95_peak_memory", "individual_peak_memory")},
        }},
        "family_drift": {"P25": {scenario: {"gate": {"status": "pass"}} for scenario in stats.SCENARIO_ORDER}},
        "candidates": {"P25": {
            "nuisance_linear_regret_vs_probe": {"lower": -0.001, "upper": 0.005},
            "nonlinear_retention": {head: {"lower": -0.001, "upper": 0.005} for head in ("quadratic", "knn", "rbf")},
            "nuisance_linear_regret_vs_probe_upper": 0.005,
            "worst_family_drift_upper": 0.01,
            "worst_block_false_overlap_upper": 0.25,
        }},
    }
    eligibility = candidate_eligibility(summary, candidates=("P25",))
    assert eligibility["P25"]["status"] == "pass"
    assert eligibility["P25"]["eligible"] is True
    decision = select_lock(summary)
    assert decision["status"] == "locked"
    assert decision["selected_candidate"] == "P25"

    summary["candidates"]["P25"].pop("worst_block_false_overlap_upper")
    missing_decision = select_lock(summary)
    assert missing_decision["status"] == "inconclusive"
    assert missing_decision["selected_candidate"] is None
    assert missing_decision["pipeline_stop"] is True


def test_genuine_overlap_requires_exact_four_k_balance_strata() -> None:
    strata = {
        key: {"estimate": -0.01, "lower": -0.02, "upper": -0.001}
        for key in ("2:balanced", "2:imbalanced", "8:balanced", "8:imbalanced")
    }
    complete = genuine_overlap_gates(
        {"P25": {"strata_fnr_upper": strata}}, candidates=("P25",)
    )
    assert complete["P25"]["gates"]["strata_fnr"]["status"] == "pass"
    missing = dict(strata)
    missing.pop("8:imbalanced")
    incomplete = genuine_overlap_gates(
        {"P25": {"strata_fnr_upper": missing}}, candidates=("P25",)
    )
    assert incomplete["P25"]["gates"]["strata_fnr"]["status"] == "inconclusive"


def _nonlinear_panel_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    axes = itertools.product(
        ("balanced", "imbalanced"), ("small", "large"), (4, 32), (2, 8),
        ("linear_separated", "nonlinear_separated", "genuine_overlap_half"),
    )
    scenarios = ("H1", "H2", "T", "C", "X", "ALL")
    for balance, count_level, dimension, k, signal in axes:
        # Put one deliberately different panel in the second balance.  The
        # expected candidate-minus-B mean is therefore 1/(2*24), proving
        # that all 24 panels enter rather than only the final assignment.
        one_difference = (
            balance == "imbalanced"
            and (count_level, dimension, k, signal) == ("large", 32, 8, "genuine_overlap_half")
        )
        for candidate in ("B", "P25"):
            for scenario in scenarios:
                score = 1.0 if scenario == "H1" else 0.0
                if candidate == "P25" and one_difference and scenario == "H2":
                    score = 2.0
                rows.append({
                    "seed": 0, "balance": balance, "count_level": count_level,
                    "nuisance_dim": dimension, "k": k, "signal_state": signal,
                    "scenario": scenario, "candidate_id": candidate,
                    "candidate_score": score, "linear_probe_score": 1.0 if scenario == "H1" else 0.0,
                    "reference_accuracies": {
                        "linear": 1.0 if scenario == "H1" else 0.0,
                        "quadratic": 1.0 if scenario == "H1" else 0.0,
                        "knn": 1.0 if scenario == "H1" else 0.0,
                        "rbf": 1.0 if scenario == "H1" else 0.0,
                    },
                    "total_rows": 160 if count_level == "small" else 640,
                })
    return rows


def test_selector_nonlinear_retention_aggregates_all_24_panels_per_balance() -> None:
    result = selector_panel_metrics(
        _nonlinear_panel_rows(), candidates=("B", "P25"), n_resamples=32,
    )
    assert result["n_complete_panels"] == 48
    retention = result["candidates"]["P25"]["heads"]["quadratic"]["regret_vs_B"]
    assert retention["status"] == "defined"
    assert retention["n_blocks"] == 1
    assert retention["estimate"] == pytest.approx(1.0 / 48.0)


def test_selector_constant_score_keeps_rank_auc_inconclusive() -> None:
    rows: list[dict[str, object]] = []
    for candidate in ("B", "P25"):
        for scenario in ("H1", "H2", "T", "C", "X", "ALL"):
            rows.append({
                "seed": 0, "balance": "balanced", "count_level": "small",
                "nuisance_dim": 4, "k": 2, "signal_state": "linear_separated",
                "scenario": scenario, "candidate_id": candidate,
                "candidate_score": 1.0, "linear_probe_score": 1.0,
                "reference_accuracies": {head: float(index) for index, head in enumerate(("linear", "quadratic", "knn", "rbf"))},
            })
    result = selector_panel_metrics(rows, candidates=("B", "P25"), n_resamples=16)
    rank_auc = result["candidates"]["P25"]["heads"]["linear"]["rank_auc"]
    assert rank_auc["status"] == "inconclusive"
    assert rank_auc["estimate"] is None


def test_descriptive_diagnostics_reports_case_paired_score_movements() -> None:
    case_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    prototype_rows: list[dict[str, object]] = []
    for case_id, baseline, raw, candidate, neighbor, purity in (
        ("case-0", 0.5, 0.4, 0.6, 0.2, 0.7),
        ("case-1", 0.7, 0.5, 0.9, 0.4, 0.9),
    ):
        for name, score in (("A", raw), ("B", baseline), ("P25", candidate)):
            case_rows.append({"case_id": case_id, "candidate_id": name, "candidate_score": score})
            geometry_rows.append({"case_id": case_id, "candidate_id": name, "geometry": {"cross_class_neighbor_impurity": neighbor}})
            prototype_rows.append({"case_id": case_id, "candidate_id": name, "prototype": {"weighted_majority_assignment_purity": purity}})
    result = stats._descriptive_diagnostics({
        "case_rows.jsonl": case_rows,
        "geometry_rows.jsonl": geometry_rows,
        "prototype_rows.jsonl": prototype_rows,
    })
    assert result["P25"]["score_delta_vs_B"] == {"n": 2, "mean": pytest.approx(0.15), "median": pytest.approx(0.15)}
    assert result["B"]["refinement_B_minus_A"] == {"n": 2, "mean": pytest.approx(0.15), "median": pytest.approx(0.15)}
    assert result["P25"]["concordance"]["neighbor_impurity_vs_score_movement"]["n"] == 2


def test_development_resources_keep_legacy_l_as_descriptive_only() -> None:
    rows: list[dict[str, object]] = []
    for resource_id in ("R0", "R1", "R2", "R3"):
        for candidate, multiplier in (("B", 1.0), ("L", 1.05), ("P25", 1.08)):
            rows.append({
                "resource_id": resource_id, "case_id": "seed-0-" + resource_id,
                "candidate_id": candidate, "status": "ok",
                "total_wall_seconds": multiplier, "total_cpu_seconds": multiplier,
                "score_fixed_wall_seconds": multiplier, "score_fixed_cpu_seconds": multiplier,
                "peak_rss_bytes": 100.0 * multiplier,
            })
    result = resource_summaries(
        rows, candidates=("L", "P25"), stage="development", expected_seed_count=1,
        n_resamples=16,
    )
    assert result["L"]["status"] == "pass"
    assert result["P25"]["status"] == "pass"
    assert "L" not in candidate_eligibility({"candidates": result})


def _fake_stage_artifact(stage: str) -> dict[str, object]:
    return {
        "manifest": {
            "stage": stage, "protocol_sha256": "p", "code_identity_sha256": "c",
        },
        "tables": {},
    }


def test_write_analysis_is_stage_specific_immutable_and_report_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stats, "load_normalized_artifact",
        lambda *args, **kwargs: _fake_stage_artifact(str(kwargs.get("expected_stage"))),
    )
    monkeypatch.setattr(
        stats, "build_summary",
        lambda artifact, **kwargs: {
            "stage": kwargs["stage"], "artifact_identity": {"status": "pass"},
            "candidates": {}, "structural_gates": {},
        },
    )
    monkeypatch.setattr(
        stats, "build_smoke_decision",
        lambda artifact, **kwargs: {
            "stage": "smoke", "status": "pass", "structural_gates": {},
        },
    )
    monkeypatch.setattr(
        stats, "select_lock",
        lambda summary, **kwargs: {
            "stage": "development", "status": "stopped_no_eligible",
            "selected_candidate": None, "structural_gates": {},
        },
    )
    monkeypatch.setattr(
        stats, "evaluate_confirmation",
        lambda summary, **kwargs: {
            "stage": "confirmation", "status": "pass", "algorithm": {"status": "pass"},
            "locked_candidate": kwargs["locked_candidate"], "structural_gates": {},
        },
    )
    monkeypatch.setattr(reporting, "write_reports", lambda *args, **kwargs: None)

    smoke_out = tmp_path / "smoke"
    smoke = stats.write_analysis(tmp_path / "ignored", smoke_out, stage="smoke")
    assert smoke["decision"]["stage"] == "smoke"
    assert (smoke_out / "smoke_decision.json").exists()
    assert not (smoke_out / "promotion_decision.json").exists()
    with pytest.raises(FileExistsError):
        stats.write_analysis(tmp_path / "ignored", smoke_out, stage="smoke")

    dev_out = tmp_path / "development"
    dev = stats.write_analysis(tmp_path / "ignored", dev_out, stage="development")
    assert dev["decision"]["stage"] == "development"
    assert (dev_out / "promotion_decision.json").exists()
    assert not (dev_out / "confirmation_decision.json").exists()

    promotion = {
        "status": "locked", "protocol_sha256": "p", "code_identity_sha256": "c",
        "locked_candidate": "P25",
    }
    promotion_hash = stats._decision_hash(promotion)
    prior = {
        "status": "pass", "protocol_sha256": "p", "code_identity_sha256": "c",
        "locked_candidate": "P25", "promotion_decision_sha256": promotion_hash,
    }
    confirmation_out = tmp_path / "confirmation"
    confirmation = stats.write_analysis(
        tmp_path / "ignored", confirmation_out, stage="confirmation",
        promotion_decision=promotion, prior_regression_decision=prior,
    )
    assert confirmation["decision"]["stage"] == "confirmation"
    assert (confirmation_out / "confirmation_decision.json").exists()
    assert not (confirmation_out / "promotion_decision.json").exists()


def test_write_analysis_never_commits_decision_after_report_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stats, "load_normalized_artifact",
        lambda *args, **kwargs: _fake_stage_artifact("development"),
    )
    monkeypatch.setattr(
        stats, "build_summary",
        lambda artifact, **kwargs: {
            "stage": "development", "artifact_identity": {"status": "pass"},
            "candidates": {}, "structural_gates": {},
        },
    )
    monkeypatch.setattr(
        stats, "select_lock",
        lambda summary, **kwargs: {"stage": "development", "status": "stopped_no_eligible"},
    )

    def fail_report(*args: object, **kwargs: object) -> None:
        raise RuntimeError("render failed")

    monkeypatch.setattr(reporting, "write_reports", fail_report)
    output = tmp_path / "failed-report"
    with pytest.raises(RuntimeError, match="render failed"):
        stats.write_analysis(tmp_path / "ignored", output, stage="development")
    assert not (output / "promotion_decision.json").exists()
    assert (output / "analysis_summary.json").exists()


def test_decision_file_hash_matches_canonical_mapping_and_prior_consumes_it(
    tmp_path: Path,
) -> None:
    promotion = {
        "status": "locked", "protocol_sha256": "p", "code_identity_sha256": "c",
        "locked_candidate": "P25",
    }
    promotion_path = tmp_path / "promotion_decision.json"
    stats.write_immutable_json(promotion_path, promotion)
    promotion_hash = stats.sha256_file(promotion_path)
    assert promotion_hash == stats._decision_hash(promotion)
    prior = {
        "status": "pass", "protocol_sha256": "p", "code_identity_sha256": "c",
        "locked_candidate": "P25", "promotion_decision_sha256": promotion_hash,
    }
    prior_path = tmp_path / "prior_regression_decision.json"
    stats.write_immutable_json(prior_path, prior)
    assert stats.validate_decision_prerequisite(
        prior_path, protocol_hash="p", code_identity_hash="c",
        expected_status=("pass",), locked_candidate="P25", promotion_hash=promotion_hash,
    )["status"] == "pass"


def _confirmation_summary_with_controls(controls: dict[str, object]) -> dict[str, object]:
    return {
        "stage": "confirmation",
        "artifact_identity": {"status": "pass"},
        "candidates": {"P25": {}},
        "primary_claims": {
            "q1": {"lower": 0.01, "upper": 0.2, "direction": "greater_than_zero"},
            "q2": {"lower": -0.2, "upper": -0.01, "direction": "less_than_zero"},
            "q3": {"lower": -0.2, "upper": -0.01, "direction": "less_than_zero"},
            "q4": {"lower": -0.2, "upper": -0.01, "direction": "less_than_zero"},
        },
        "negative_controls": controls,
    }


def _confirmation_prerequisites() -> tuple[dict[str, object], dict[str, object]]:
    promotion = {
        "status": "locked", "protocol_sha256": "p", "code_identity_sha256": "c",
        "locked_candidate": "P25",
    }
    prior = {
        "status": "pass", "protocol_sha256": "p", "code_identity_sha256": "c",
        "locked_candidate": "P25", "promotion_decision_sha256": stats._decision_hash(promotion),
    }
    return promotion, prior


def test_confirmation_requires_q2_q3_q4_negative_control_margins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stats, "candidate_eligibility",
        lambda summary, candidates: {candidate: {"eligible": True, "status": "pass"} for candidate in candidates},
    )
    controls = {
        name: {"estimate": 0.0, "lower": -0.01, "upper": 0.005}
        for name in ("q2", "q3", "q4")
    }
    promotion, prior = _confirmation_prerequisites()
    passed = stats.evaluate_confirmation(
        _confirmation_summary_with_controls(controls), locked_candidate="P25",
        promotion_decision=promotion, prior_regression_decision=prior,
        protocol_hash="p", code_identity_hash="c",
    )
    assert passed["algorithm"]["status"] == "pass"
    assert passed["mechanism"]["status"] == "pass"
    assert {name: value["status"] for name, value in passed["mechanism"]["negative_controls"].items()} == {"q2": "pass", "q3": "pass", "q4": "pass"}
    assert passed["status"] == "pass"

    violated = dict(controls)
    violated["q3"] = {"estimate": 0.02, "lower": 0.01, "upper": 0.02}
    failed = stats.evaluate_confirmation(
        _confirmation_summary_with_controls(violated), locked_candidate="P25",
        promotion_decision=promotion, prior_regression_decision=prior,
        protocol_hash="p", code_identity_hash="c",
    )
    assert failed["algorithm"]["status"] == "pass"
    assert failed["mechanism"]["negative_controls"]["q3"]["status"] == "fail"
    assert failed["mechanism"]["status"] == "fail"
    # Mechanism failure alone does not reject the independently passing arm.
    assert failed["status"] == "pass"

    missing = dict(controls)
    missing.pop("q4")
    inconclusive = stats.evaluate_confirmation(
        _confirmation_summary_with_controls(missing), locked_candidate="P25",
        promotion_decision=promotion, prior_regression_decision=prior,
        protocol_hash="p", code_identity_hash="c",
    )
    assert inconclusive["algorithm"]["status"] == "pass"
    assert inconclusive["mechanism"]["negative_controls"]["q4"]["status"] == "inconclusive"
    assert inconclusive["mechanism"]["status"] == "inconclusive"
