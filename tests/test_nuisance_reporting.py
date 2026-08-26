"""Cheap deterministic checks for nuisance report extraction and rendering."""

from __future__ import annotations

from pathlib import Path

from experiments.nuisance_conditioned_distance.reporting import (
    conditioning_rows,
    decision_rows,
    genuine_overlap_rows,
    nuisance_curve_rows,
    probe_rows,
    render_report,
    runtime_rows,
    selector_rows,
    write_reports,
)


def _summary() -> dict:
    candidate = {
        "name": "conditioned",
        "selection_metrics": {
            "heads": {
                "linear": {
                    "regret": {"estimate": 0.01, "lower": 0.0, "upper": 0.02},
                    "rank_auc": {"estimate": 0.8},
                    "exact_best": 0.6,
                    "within_one_point": 0.9,
                }
            }
        },
        "genuine_overlap": {
            "auroc": 0.91,
            "auprc": 0.88,
            "fpr": 0.02,
            "fnr": 0.04,
            "brier": 0.03,
            "ece": 0.01,
        },
        "nuisance_curve": [
            {"strength": 0.0, "evidence": 0.2, "n": 2},
            {"strength": 1.0, "evidence": 0.3, "n": 2},
        ],
        "nuisance_robustness_auc": {"estimate": 0.04, "status": "defined"},
        "timing": {
            "fit": {"median": 1.0, "p95": 1.2},
            "score_fixed": {"median": 0.2, "p95": 0.3},
            "total": {"median": 1.2, "p95": 1.5},
            "peak_memory": {"median": 100.0, "p95": 120.0},
            "median_total_ratio_vs_B": 1.1,
            "p95_total_ratio_vs_B": 1.2,
            "median_peak_memory_ratio_vs_B": 1.05,
        },
        "conditioning_refinement": {
            "refinement_rate": 0.25,
            "condition_number": 12.0,
            "conditioning_strength_spearman": 0.7,
        },
    }
    return {
        "stage": "screen",
        "n_rows": 12,
        "statistics": {"bootstrap_resamples": 32, "bootstrap_seed": 7},
        "artifact_completeness": {"status": "inconclusive"},
        "candidates": {"B": candidate, "C": candidate, "F": {}, "G": {}},
        "historical_gates": {"C": {"status": "pass"}},
        "product_gates": {"C": {"status": "inconclusive"}},
        "source_hashes": {"results.json": "abc123"},
        "provenance": {"commit": "deadbeef", "dirty": False, "code_identity_sha256": "codehash"},
        "manifest": {
            "retrospective": True,
            "deviations": ["probe memory was not collected"],
            "configuration": {"capped_probe_maximum_rows": 2048},
        },
        "capped_probe_rows": [{"status": "ok", "wall_seconds": 2.0, "score": 0.8, "n_rows": 128}],
        "prior_full_probe_rows": [{"status": "ok", "wall_seconds": 5.0, "score": 0.9, "n_rows": 5000}],
        "guardrail_rows": [{"status": "ok", "panel_triggered": True, "policy_wall_seconds": 3.0}],
    }


def test_report_extractors_preserve_real_summary_fields() -> None:
    summary = _summary()

    selector = selector_rows(summary)
    linear = next(row for row in selector if row["candidate"] == "C" and row["head"] == "linear")
    assert linear["regret"] == 0.01
    assert linear["rank_auc"] == 0.8
    assert linear["exact_best"] == 0.6
    assert linear["within_1pp"] == 0.9

    curve = nuisance_curve_rows(summary)
    assert [(row["strength"], row["evidence"]) for row in curve if row["candidate"] == "C"] == [(0.0, 0.2), (1.0, 0.3)]
    assert next(row for row in genuine_overlap_rows(summary) if row["candidate"] == "C")["auroc"] == 0.91
    assert next(row for row in runtime_rows(summary) if row["candidate"] == "C" and row["stage"] == "total")["p95"] == 1.5
    assert next(row for row in conditioning_rows(summary) if row["candidate"] == "C")["condition_number"] == 12.0

    probes = probe_rows(summary)
    assert {row["probe"] for row in probes} == {"capped probe", "full probe (prior)", "G guardrail policy"}
    assert any(row["triggered_panels"] == 1 for row in probes)


def test_report_contains_scope_gates_provenance_and_is_deterministic(tmp_path: Path) -> None:
    summary = _summary()
    summary["food101"] = {
        "statistics": {},
        "candidates": {
            "C": {
                "selection_metrics": {
                    "heads": {
                        "linear": {
                            "regret": {"estimate": 0.02},
                            "rank_auc": {"estimate": 0.75},
                            "exact_best": 0.5,
                            "within_one_point": 0.8,
                        }
                    }
                }
            }
        },
    }
    promotion = {"selected_candidate": "C", "status": "promoted_for_full_evaluation", "reason": "frozen rule"}
    first = render_report(summary, promotion)
    second = render_report(summary, promotion)
    assert first == second
    for phrase in (
        "Selector regret and rank AUC",
        "Genuine-overlap detection and calibration",
        "C–E (C-E) are the algorithmic OI candidates",
        "diagnostic only",
        "product guardrail",
        "Retrospective status and untouched confirmation",
        "results.json",
        "probe memory was not collected",
        "Food-101 retrospective panel (separate evidence)",
        "Synthetic screen ranking locks one C–E candidate",
        "mandatory Food-101 product gates decide the final locked candidate",
        "never trigger reranking",
        "regret **145**; rank AUC **143**",
    ):
        assert phrase in first

    output = tmp_path / "report"
    write_reports(output, summary, promotion)
    assert {path.name for path in output.iterdir()} == {
        "decision_table.csv",
        "decision_table.json",
        "report.md",
        "score_regret.svg",
        "nuisance_curves.svg",
        "genuine_overlap.svg",
        "runtime.svg",
        "conditioning_refinement.svg",
    }
    assert "linear regret" in (output / "score_regret.svg").read_text(encoding="utf-8")
    assert "robustness AUC" in (output / "nuisance_curves.svg").read_text(encoding="utf-8")
    assert "AUROC" in (output / "genuine_overlap.svg").read_text(encoding="utf-8")
    assert "Runtime stages" in (output / "runtime.svg").read_text(encoding="utf-8")
    assert "Conditioning" in (output / "conditioning_refinement.svg").read_text(encoding="utf-8")


def test_decision_rows_keep_non_promotable_f_and_g_explicit() -> None:
    rows = decision_rows(_summary(), {"selected_candidate": "C"})
    by_candidate = {row["candidate"]: row for row in rows}
    assert by_candidate["C"]["decision_status"] == "inconclusive"
    assert by_candidate["F"]["role"] == "diagnostic only"
    assert by_candidate["G"]["role"] == "product guardrail"
    assert by_candidate["F"]["promoted"] is False
    assert by_candidate["G"]["promoted"] is False


def test_locked_candidate_gate_status_never_reranks() -> None:
    summary = _summary()
    summary["candidates"]["D"] = dict(summary["candidates"]["C"])
    for status, expected in (
        ("pass", "Final locked-candidate decision: **pass** for locked candidate **C**"),
        ("fail", "Final locked-candidate decision: **rejected** (status **fail**) for locked candidate **C**"),
        ("inconclusive", "Final locked-candidate decision: **inconclusive** (status **inconclusive**) for locked candidate **C**"),
    ):
        promotion = {
            "locked_candidate": "C",
            "selected_candidate": "D",
            "status": status,
            "reason": "mandatory Food-101 product gates",
        }
        report = render_report(summary, promotion)
        assert expected in report
        assert "No reranking was performed." in report
        rows = decision_rows(summary, promotion)
        assert next(row for row in rows if row["candidate"] == "C")["promoted"] is True
        assert next(row for row in rows if row["candidate"] == "D")["promoted"] is False


def test_food_probe_pseudo_candidates_are_reported_as_probe_artifacts() -> None:
    summary = {
        "candidates": {
            "A": {},
            "G_probe_component": {
                "score": {"estimate": 0.8},
                "n_rows": 4,
                "timing": {"total": {"median": 2.0, "p95": 3.0}},
            },
            "full_probe": {
                "score": {"estimate": 0.9},
                "n_rows": 8,
                "timing": {"total": {"median": 4.0, "p95": 5.0}},
            },
        }
    }
    probes = probe_rows(summary)
    assert {row["probe"] for row in probes} == {"capped probe", "full probe"}
    assert [row["candidate"] for row in decision_rows(summary)] == ["A"]
