"""Focused reporting and artifact-lineage tests for M50 ranking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.m50_backbone_ranking import manifest as experiment_manifest
from experiments.m50_backbone_ranking import reporting


def _summary() -> dict[str, object]:
    return {
        "candidate_gates": {
            "M1-SW": {
                "gates": {
                    "nuisance_full:linear": {
                        "estimate": 0.2,
                        "lower_95": -0.1,
                        "upper_95": 0.6,
                        "status": "pass",
                        "rule": "upper_95 <= 1.0 pp",
                    },
                    "nonlinearity_full:rbf": {
                        "estimate": -0.4,
                        "lower_95": -0.8,
                        "upper_95": -0.1,
                        "status": "pass",
                        "rule": "upper_95 < 0 pp",
                    },
                }
            }
        },
        "factorial": {
            "effects": {
                "baseline:linear": {
                    "raw_refinement_effect": {"estimate": 0.1},
                    "conditioned_refinement_effect": {"estimate": -0.1},
                    "interaction": {"estimate": -0.2},
                    "cb_refinement_diagnostic": {"estimate": -0.3},
                }
            },
        },
        "selection_rates": [
            {
                "candidate_id": "M1-SW",
                "arm": "baseline",
                "head": "linear",
                "mean_regret_pp": 0.2,
                "exact_best_rate": 0.8,
                "within_one_pp_rate": 1.0,
            }
        ],
        "rank_auc_summary": [
            {
                "candidate_id": "M1-SW",
                "arm": "baseline",
                "head": "linear",
                "mean_rank_auc": 0.8,
                "status": "defined",
                "undefined_replicate_count": 0,
            }
        ],
        "fusion_gates": {
            "candidate_id": "F",
            "status": "pass",
            "gates": {
                "nonlinearity_full:rbf": {
                    "estimate": -0.2,
                    "lower_95": -0.4,
                    "upper_95": -0.1,
                    "status": "pass",
                    "rule": "upper_95 < 0 pp",
                }
            },
        },
        "guardrail_gates": None,
        "resources": {
            "arm_gates": {
                arm: {
                    "status": "pass",
                    "per_backbone_faster_status": "established",
                    "full_panel_faster_status": "established",
                    "faster_claim_status": "established",
                    "faster_claim": {
                        "per_backbone_upper_95": 0.95,
                        "full_panel_upper_95": 0.96,
                        "promotion_veto": False,
                    },
                    "gates": {
                        "candidate_vs_probe_wall": {
                            "median": 0.9,
                            "upper_95": 1.0,
                            "status": "pass",
                        }
                    },
                }
                for arm in ("baseline", "nonlinearity_full", "nuisance_full")
            },
            "full_panel_arm_gates": {
                arm: {
                    "status": "pass",
                    "gates": {
                        "candidate_vs_probe_wall": {
                            "median": 0.91,
                            "upper_95": 0.96,
                            "status": "pass",
                        },
                        "candidate_vs_B_wall": {
                            "median": 1.01,
                            "p95": 1.04,
                            "status": "pass",
                        },
                    },
                }
                for arm in ("baseline", "nonlinearity_full", "nuisance_full")
            },
            "full_panel_ratio_rows": [
                {
                    "arm": "baseline",
                    "budget": 128,
                    "repeat": 0,
                    "candidate_probe_wall_ratio": 0.91,
                    "candidate_baseline_wall_ratio": 1.01,
                }
            ],
            "secondary_runtime": {
                "inferential": False,
                "promotion_gate": False,
                "arm_summary": {
                    arm: {
                        "per_backbone_cpu_ratios": {
                            "candidate_vs_probe": {"median": 0.8, "p95": 0.9},
                            "candidate_vs_B": {"median": 1.0, "p95": 1.1},
                        },
                        "full_panel_cpu_ratios": {
                            "candidate_vs_probe": {"median": 0.82, "p95": 0.92},
                            "candidate_vs_B": {"median": 1.01, "p95": 1.11},
                        },
                        "stage_seconds": {
                            "candidate_oi_fit_wall_seconds": {
                                "median": 0.2,
                                "p95": 0.3,
                            }
                        },
                    }
                    for arm in ("baseline", "nonlinearity_full", "nuisance_full")
                },
                "scaling_by_arm_budget": [
                    {
                        "arm": "baseline",
                        "budget": 128,
                        "candidate_probe_cpu_ratio_median": 0.8,
                        "candidate_conditioning_fit_wall_seconds_median": 0.1,
                        "candidate_oi_fit_wall_seconds_median": 0.2,
                        "candidate_score_fixed_wall_seconds_median": 0.3,
                        "probe_fit_wall_seconds_median": 0.4,
                        "probe_score_fixed_wall_seconds_median": 0.5,
                    }
                ],
            },
            "scaling_by_budget": [
                {
                    "budget": 128,
                    "candidate_probe_wall_median": 0.9,
                    "candidate_B_wall_median": 1.0,
                    "candidate_B_memory_median": 1.0,
                }
            ],
        },
    }


def _lock() -> dict[str, object]:
    return {
        "status": "locked_for_future_confirmation",
        "selected_candidate": "M1-SW",
        "selected_chain": "M1-SW",
        "chain_kind": "pure_oi",
        "algorithmic_oi_status": "pass",
        "product_policy_status": "not_applicable",
        "runner_up_allowed": False,
    }


def _lineage() -> dict[str, str]:
    return {
        "stage": "development_provisional",
        "status": "completed",
        "protocol_sha256": "a" * 64,
        "code_identity_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_report_distinguishes_algorithmic_selector_policy_and_retrospective_scope() -> None:
    report = reporting.render_markdown_report(_summary(), _lock())
    assert "retrospective development evidence, not untouched confirmation" in report
    assert "M50 and the fixed 50/50 raw+M50 rank fusion F are algorithmic" in report
    assert "G is a separate selective capped-probe product policy" in report
    assert "never rescues a failed nonlinear superiority claim" in report
    assert "Distribution-shift stress tests" in report
    assert "class-stratified same-distribution sampling" in report
    assert "## Selection rates and average regret" in report
    assert "## Rank AUC" in report
    assert "CB diagnostic" in report
    assert "Descriptive resource scaling by budget (pooled across arms)" in report
    assert "all three arms must pass" in report
    assert "descriptive only" in report
    assert "peak RSS conservatively includes warm-up" in report
    assert "Conditioned OI base candidate" in report
    assert "Faster than LP-FULL claim (non-promotion)" in report
    assert "Full 10-backbone product-panel timing gates" in report
    assert "Per-backbone-call timing and memory gates" in report
    assert "Descriptive CPU and stage clocks (noninferential, nonpromotion)" in report
    assert "No stage clock is derived by subtracting noisy totals" in report
    assert "full 10-backbone panel" in report
    assert "Descriptive CPU/stage scaling by arm and budget" in report
    assert "OI fit/refinement wall" in report
    assert "| baseline | 0.9500 | 0.9600 | established |" in report
    assert "product-only capped-probe guardrail" in report
    assert "no (product only)" in report
    assert "| F | nonlinearity_full:rbf |" in report


def test_bundle_is_non_overwriting_hashed_and_byte_stable(tmp_path: Path) -> None:
    panels = [
        {"candidate_id": "M1-SW", "replicate": 1, "regret_pp": 0.2},
        {"candidate_id": "M1-SW", "replicate": 0, "regret_pp": 0.1},
    ]
    auc = [
        {"candidate_id": "M1-SW", "replicate": 1, "rank_auc": 0.8},
        {"candidate_id": "M1-SW", "replicate": 0, "rank_auc": 0.9},
    ]
    first = reporting.write_report_bundle(
        tmp_path / "first",
        summary=_summary(),
        development_lock=_lock(),
        panel_metrics=panels,
        rank_auc=auc,
        full_panel_resource_rows=_summary()["resources"]["full_panel_ratio_rows"],  # type: ignore[index]
        lineage=_lineage(),
    )
    second = reporting.write_report_bundle(
        tmp_path / "second",
        summary=_summary(),
        development_lock=_lock(),
        panel_metrics=list(reversed(panels)),
        rank_auc=list(reversed(auc)),
        full_panel_resource_rows=list(
            reversed(_summary()["resources"]["full_panel_ratio_rows"])  # type: ignore[index]
        ),
        lineage=_lineage(),
    )
    assert {path.name for path in first.iterdir()} == {path.name for path in second.iterdir()}
    for first_path in first.iterdir():
        second_path = second / first_path.name
        assert first_path.read_bytes() == second_path.read_bytes()

    manifest = json.loads((first / "analysis_manifest.json").read_text())
    assert manifest["protocol_sha256"] == "a" * 64
    assert manifest["code_identity_sha256"] == "b" * 64
    assert manifest["input_manifest_sha256"] == "c" * 64
    assert manifest["development_lock_sha256"] == _sha(first / "development_lock.json")
    assert manifest["files"]["panel_metrics.csv"]["row_count"] == 2
    assert manifest["files"]["full_panel_resource.csv"]["row_count"] == 1
    for name, identity in manifest["files"].items():
        assert identity["sha256"] == _sha(first / name)
    assert b"M1-SW:nuisance_full:linear" in (
        first / "selector_regret.svg"
    ).read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reporting.write_report_bundle(
            first,
            summary=_summary(),
            development_lock=_lock(),
            panel_metrics=panels,
            rank_auc=auc,
            lineage=_lineage(),
        )


def test_bundle_fails_closed_on_incomplete_lineage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lineage must contain exactly"):
        reporting.write_report_bundle(
            tmp_path / "bad",
            summary=_summary(),
            development_lock=_lock(),
            panel_metrics=(),
            rank_auc=(),
            lineage={"stage": "development"},
        )


def test_final_bundle_hash_binds_and_reports_selected_chain_recipe(tmp_path: Path) -> None:
    bindings = {
        "schema_version": 1,
        "selected_candidate": "M1-SW",
        "selected_chain": "F",
        "guardrail_base_id": None,
        "primitive_component_ids": ["B", "M1-SW"],
        "component_recipe_family_sha256": {
            "B": {
                "food_recipe_family_sha256": "a" * 64,
                "runtime_recipe_family_sha256": "b" * 64,
            },
            "M1-SW": {
                "food_recipe_family_sha256": "c" * 64,
                "runtime_recipe_family_sha256": "d" * 64,
            },
        },
        "derived_recipe_identity": {"F": {"candidate_id": "F"}},
        "derived_recipe_sha256": {"F": "e" * 64},
        "deterministic_seed_rule": {"food": {"seed": "food"}, "runtime": {"seed": "runtime"}},
    }
    digest = hashlib.sha256(
        experiment_manifest.canonical_json(bindings).encode("utf-8")
    ).hexdigest()
    lock = {
        **_lock(),
        "selected_chain": "F",
        "selected_chain_recipe_bindings": bindings,
        "selected_chain_recipe_bindings_sha256": digest,
    }
    output = reporting.write_report_bundle(
        tmp_path / "recipe",
        summary=_summary(),
        development_lock=lock,
        panel_metrics=(),
        rank_auc=(),
        selected_chain_recipe_bindings=bindings,
        lineage={**_lineage(), "stage": "development_final"},
    )
    analysis_manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert analysis_manifest["selected_chain_recipe_bindings_sha256"] == digest
    assert analysis_manifest["files"]["selected_chain_recipe_bindings.json"]["sha256"] == _sha(
        output / "selected_chain_recipe_bindings.json"
    )
    report = (output / "report.md").read_text()
    assert "## Hash-bound executable recipe" in report
    assert f"Binding SHA-256: `{digest}`" in report
    assert "future V2 must hash-link these exact bytes" in report

    with pytest.raises(ValueError, match="do not match the final development lock"):
        reporting.write_report_bundle(
            tmp_path / "bad-recipe",
            summary=_summary(),
            development_lock={**lock, "selected_chain_recipe_bindings_sha256": "f" * 64},
            panel_metrics=(),
            rank_auc=(),
            selected_chain_recipe_bindings=bindings,
            lineage={**_lineage(), "stage": "development_final"},
        )


def test_report_renders_confirmation_gate_results() -> None:
    summary = {
        **_summary(),
        "confirmation": {
            "status": "pass",
            "gates": {
                "linear": {
                    "estimate": 0.2,
                    "lower_95": -0.1,
                    "upper_95": 0.7,
                    "status": "pass",
                }
            },
        },
    }
    report = reporting.render_markdown_report(summary, _lock())
    assert "## Future V2 confirmation design evidence" in report
    assert "V1 cannot execute or certify confirmation outcomes" in report
    assert "Design-evidence status: **pass**" in report
    assert "| linear | 0.2000 | [-0.1000, 0.7000] | pass |" in report


def test_final_bundle_preserves_hash_bound_development_surfaces(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    linked = {}
    for name, payload in {
        "panel_metrics.csv": b"candidate_id\nA\n",
        "rank_auc.csv": b"candidate_id\nA\n",
        "report.md": b"development report\n",
    }.items():
        path = source / name
        path.write_bytes(payload)
        linked[name] = path
    output = reporting.write_report_bundle(
        tmp_path / "final",
        summary=_summary(),
        development_lock=_lock(),
        panel_metrics=(),
        rank_auc=(),
        linked_development_files=linked,
        lineage={**_lineage(), "stage": "development_final"},
    )
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    for name, source_path in linked.items():
        copied = output / f"development_{name}"
        assert copied.read_bytes() == source_path.read_bytes()
        assert manifest["files"][copied.name]["sha256"] == _sha(copied)
