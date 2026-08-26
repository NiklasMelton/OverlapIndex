"""Deterministic wording/table checks for heteroscedastic reports."""

from __future__ import annotations

from pathlib import Path

from experiments.heteroscedastic_distance_conditioning.reporting import (
    conditioning_rows,
    decision_rows,
    geometry_chain_rows,
    render_report,
    resource_rows,
    selector_rows,
    write_reports,
)


def _summary() -> dict:
    return {
        "stage": "development",
        "candidates": {
            "B": {"heads": {"linear": {"pooled": {"estimate": 0.02}}}},
            "P50-SW": {"heads": {"linear": {"pooled": {"estimate": 0.01}}}},
        },
        "selection": {"candidates": {
            "B": {"heads": {"linear": {
                "pooled": {"estimate": 0.02}, "pooled_rank_spearman": {"estimate": 0.8},
                "rank_auc": {"estimate": 0.7}, "pooled_exact_best": {"estimate": 0.6},
                "pooled_within_one_point": {"estimate": 0.9},
            }}},
            "P50-SW": {"heads": {"linear": {
                "pooled": {"estimate": 0.01}, "pooled_rank_spearman": {"estimate": 0.9},
                "rank_auc": {"estimate": 0.85}, "pooled_exact_best": {"estimate": 0.75},
                "pooled_within_one_point": {"estimate": 0.95},
            }},
        }}},
        "primary_claims": {"q1": {"estimate": 0.1, "lower": 0.05, "upper": 0.2, "status": "defined", "direction": "greater_than_zero"}},
        "genuine_overlap": {"pooled": {
            "B": {"auroc": {"estimate": 0.9}, "auprc": {"estimate": 0.8}, "fpr": {"estimate": 0.1}, "fnr": {"estimate": 0.1}, "brier": {"estimate": 0.1}, "ece": {"estimate": 0.1}},
            "P50-SW": {"auroc": {"estimate": 0.91}, "auprc": {"estimate": 0.81}, "fpr": {"estimate": 0.1}, "fnr": {"estimate": 0.1}, "brier": {"estimate": 0.1}, "ece": {"estimate": 0.1}},
        }},
        "family_drift": {"P50-SW": {"H2": {"estimate": 0.0, "upper": 0.01, "gate": {"status": "pass"}}}},
        "stable_shift_gates": {"P50-SW": {"balanced": {"X-H2:auroc": {"estimate": -0.01, "upper": 0.0, "gate": {"status": "pass"}}}}},
        "resource_gates": {
            "L": {"status": "pass", "summaries": {"median_total": 1.05, "p95_total": 1.1, "median_score_fixed": 1.05, "median_peak_memory": 1.05, "p95_peak_memory": 1.1, "individual_peak_memory": 1.1}},
            "P50-SW": {"status": "pass", "summaries": {"median_total": 1.0, "p95_total": 1.1, "median_score_fixed": 1.0, "median_peak_memory": 1.0, "p95_peak_memory": 1.1, "individual_peak_memory": 1.1}},
        },
        "diagnostics": {
            "B": {
                "conditioning": {"mode": "none", "condition_before": 2.0, "condition_after": 2.0, "regularized_eigenvalue_count": 0, "cap_or_floor_active_rate": 0.0},
                "refinement": {"activity_rate": 0.2, "applied_count": 2, "eligible_count": 10},
                "geometry": {"neighbor_impurity": 0.2, "oracle_neighbor_jaccard": 0.8, "pair_distance_spearman": 0.7},
                "prototype": {"weighted_majority_assignment_purity": 0.75, "owner_label_agreement": 0.7},
                "score": {"mean": 0.6},
                "refinement_B_minus_A": {"n": 2, "mean": 0.1, "median": 0.1},
            },
            "P50-SW": {
                "conditioning": {"mode": "pooled_full", "condition_before": 10.0, "condition_after": 2.0, "regularized_eigenvalue_count": 1, "cap_or_floor_active_rate": 0.5},
                "refinement": {"activity_rate": 0.3, "applied_count": 3, "eligible_count": 10},
                "geometry": {"neighbor_impurity": 0.1, "oracle_neighbor_jaccard": 0.9, "pair_distance_spearman": 0.8},
                "prototype": {"weighted_majority_assignment_purity": 0.85, "owner_label_agreement": 0.8},
                "score": {"mean": 0.65},
                "score_delta_vs_B": {"n": 2, "mean": 0.05, "median": 0.05},
                "concordance": {
                    "neighbor_impurity_vs_score_movement": {"n": 2, "spearman": 0.4},
                    "prototype_purity_vs_score_movement": {"n": 2, "spearman": 0.5},
                },
            },
        },
        "eligibility": {"B": {"status": "pass", "reasons": []}, "P50-SW": {"status": "pass", "reasons": []}},
    }


def test_report_scope_wording_and_determinism() -> None:
    summary = _summary()
    decision = {"status": "locked", "selected_candidate": "P50-SW"}
    first = render_report(summary, decision)
    second = render_report(summary, decision)
    assert first == second
    for phrase in (
        "ordered mechanistic concordance",
        "does not establish formal causal mediation",
        "Food-101 is not authorized",
        "Selector regret and rank AUC",
        "Genuine-overlap detection and calibration",
        "Family drift and stable-shift gates",
        "Runtime and memory",
        "Algorithmic candidate eligibility and mechanism claims are reported separately",
        "inconclusive",
    ):
        assert phrase in first


def test_reports_emit_stable_tables_and_all_required_plot_surfaces(tmp_path: Path) -> None:
    summary = _summary()
    decision = {"status": "locked", "selected_candidate": "P50-SW"}
    write_reports(tmp_path, summary, decision)
    expected = {
        "report.md", "decision_table.csv", "decision_table.json", "geometry_chain.svg", "claims.svg",
        "selector.svg", "overlap.svg", "family_shift.svg", "refinement_conditioning.svg", "runtime_memory.svg", "candidate_decisions.svg",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert "causal" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Runtime and memory" in (tmp_path / "runtime_memory.svg").read_text(encoding="utf-8")
    assert "<polyline" not in (tmp_path / "runtime_memory.svg").read_text(encoding="utf-8")
    first = (tmp_path / "report.md").read_bytes()
    write_reports(tmp_path, summary, decision)
    assert first == (tmp_path / "report.md").read_bytes()


def test_report_consumes_canonical_selector_and_descriptive_surfaces() -> None:
    summary = _summary()
    selector = selector_rows(summary)
    assert selector
    p50_linear = next(row for row in selector if row["candidate"] == "P50-SW" and row["head"] == "linear")
    assert p50_linear["rank_spearman"] == 0.9
    assert p50_linear["exact_best"] == 0.75
    assert {row["candidate"] for row in resource_rows(summary)} == {"L", "P50-SW"}
    conditioning = conditioning_rows(summary)
    assert {row["candidate"] for row in conditioning} == {"B", "P50-SW"}
    chain = geometry_chain_rows(summary)
    p50_chain = next(row for row in chain if row["candidate"] == "P50-SW")
    assert p50_chain["score_delta_vs_B"] == 0.05
    assert p50_chain["neighbor_score_movement_spearman"] == 0.4


def test_confirmation_failure_does_not_mark_locked_candidate_selected() -> None:
    summary = _summary()
    failed = decision_rows(summary, {
        "stage": "confirmation", "status": "fail", "locked_candidate": "P50-SW",
        "algorithm": {"status": "fail"},
    })
    row = next(item for item in failed if item["candidate"] == "P50-SW")
    assert row["selected"] is False
    passed = decision_rows(summary, {
        "stage": "confirmation", "status": "pass", "locked_candidate": "P50-SW",
        "algorithm": {"status": "pass"},
    })
    row = next(item for item in passed if item["candidate"] == "P50-SW")
    assert row["selected"] is True
