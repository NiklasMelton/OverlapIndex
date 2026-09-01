"""Focused tests for the continuous shortlist-depth audit."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from experiments.continuous_shortlist_depth_audit import run


def _candidate(position: int, broad: float, current: float, envelope: float) -> dict:
    folds = [
        {
            "kernel_alignment": {
                "feature_bandwidths": {
                    "0.25": {"score": broad * 0.9},
                    "1.0": {"score": broad * 0.95},
                    "4.0": {"score": broad},
                },
                "score": broad * 0.95,
            }
        }
        for _ in range(4)
    ]
    return {
        "candidate_id": f"candidate_{position:02d}",
        "best_head_envelope_r2": envelope,
        "selectors": {
            "rff64_mean8_k4": {
                "folds": folds,
                "kernel_alignment": {
                    "score": broad * 0.95,
                    "feature_bandwidths": {
                        "0.25": {"score": broad * 0.9},
                        "1.0": {"score": broad * 0.95},
                        "4.0": {"score": broad},
                    },
                },
            },
            "save_rffmean_k4": {"current_common32": {"score": current}},
        },
    }


def _panel(seed: int = 1907, family: str = "linear") -> dict:
    broad = np.linspace(0.95, 0.20, 16)
    current = np.linspace(0.20, 0.95, 16)
    envelope = np.linspace(0.90, 0.30, 16)
    return {
        "seed": seed,
        "family": family,
        "candidate_count": 16,
        "candidates": [
            _candidate(
                position,
                float(broad[position]),
                float(current[position]),
                float(envelope[position]),
            )
            for position in range(16)
        ],
    }


def test_protocol_and_both_completed_input_bundles_are_exact():
    protocol = run._load_protocol()

    run._validate_protocol(protocol)
    loaded = run.load_inputs(protocol)

    assert len(loaded) == 60
    assert {item["cohort"] for item in loaded} == set(run.COHORTS)
    assert tuple(protocol["shortlist_depths"]) == run.DEPTHS


def test_canonically_serialized_protocol_remains_valid():
    protocol = json.loads(json.dumps(run._load_protocol(), sort_keys=True))

    run._validate_protocol(protocol)


def test_dynamic_bounded_pool_is_exactly_top_2k_before_cutoff_ties():
    panel = _panel()

    k3 = run.ranking_groups(panel, "bounded_2k", 3)
    k5 = run.ranking_groups(panel, "bounded_2k", 5)

    assert k3[:3] == [(0,), (5,), (4,)]
    assert k5[:5] == [(0,), (9,), (8,), (7,), (6,)]
    assert all(index < 6 for group in k3[:3] for index in group)
    assert all(index < 10 for group in k5[:5] for index in group)


def test_dynamic_pool_includes_complete_group_crossing_2k_cutoff():
    panel = _panel()
    candidates = panel["candidates"]
    cutoff = candidates[5]["selectors"]["rff64_mean8_k4"]["kernel_alignment"][
        "feature_bandwidths"
    ]["4.0"]["score"]
    candidates[6]["selectors"]["rff64_mean8_k4"]["kernel_alignment"][
        "feature_bandwidths"
    ]["4.0"]["score"] = cutoff
    candidates[6]["selectors"]["save_rffmean_k4"]["current_common32"][
        "score"
    ] = 2.0

    groups = run.ranking_groups(panel, "bounded_2k", 3)

    assert groups[1] == (6,)


def test_membership_probabilities_average_every_tied_cutoff_subset():
    probabilities = run._membership_probabilities([(0,), (1, 2), (3,)], 2, 4)

    np.testing.assert_array_equal(probabilities, [1.0, 0.5, 0.5, 0.0])
    assert run._inclusion_agreement(probabilities, probabilities, 2) == 1.0


def test_oracle_pool_is_a_valid_ceiling_under_winner_and_pool_constraints():
    panel = _panel()
    reference = np.asarray(
        [candidate["best_head_envelope_r2"] for candidate in panel["candidates"]]
    )
    broad = run.winner_parent._top_k_metrics(
        reference, run.ranking_groups(panel, "broad_rff", 3), 3
    )
    oracle = run.winner_parent._top_k_metrics(
        reference,
        run.ranking_groups(panel, "winner_preserved_pool_oracle", 3),
        3,
    )

    assert oracle["mean_utility_regret_r2"] <= broad["mean_utility_regret_r2"]
    assert oracle["opportunity_regret_r2"] <= broad["opportunity_regret_r2"]


def test_outcome_free_diagnostics_do_not_change_when_envelope_changes():
    panel = _panel()
    first = run.panel_depth_rows("winner_fill", panel, 3)[0]
    changed = copy.deepcopy(panel)
    for position, candidate in enumerate(changed["candidates"]):
        candidate["best_head_envelope_r2"] = float(position**2)
    second = run.panel_depth_rows("winner_fill", changed, 3)[0]

    for key in (
        "normalized_broad_boundary_margin",
        "broad_fold_inclusion_stability",
        "broad_bandwidth_inclusion_agreement",
        "broad_save_inclusion_agreement",
    ):
        assert first[key] == second[key]


def test_panel_depth_rows_are_complete_finite_and_partitioned():
    rows = run.panel_depth_rows("winner_fill", _panel(), 5)

    assert {row["method"] for row in rows} == set(run.METHODS)
    assert all(row["k"] == 5 for row in rows)
    assert all(0.0 <= row["material_precision"] <= 1.0 for row in rows)
    assert all(0.0 <= row["broad_fold_inclusion_stability"] <= 1.0 for row in rows)


def _decision_rows() -> list[dict]:
    rows = []
    for cohort in run.COHORTS:
        for family in run.FAMILIES:
            for seed_index in range(5):
                panel_id = f"{cohort}__{family}__s{seed_index}"
                for k in run.DEPTHS:
                    for method in run.METHODS:
                        rows.append(
                            {
                                "cohort": cohort,
                                "family": family,
                                "panel_id": panel_id,
                                "k": k,
                                "method": method,
                                "opportunity_regret_r2": 0.01,
                                "mean_utility_regret_r2": 0.02,
                                "material_precision": 0.8,
                                "expected_membership_swaps": 1.0,
                                "realized_mean_utility_gain": 0.01,
                                "oracle_pool_headroom": 0.02,
                                "normalized_broad_boundary_margin": 0.1,
                                "broad_fold_inclusion_stability": 0.9,
                                "broad_bandwidth_inclusion_agreement": 0.8,
                                "broad_save_inclusion_agreement": 0.7,
                            }
                        )
    return rows


def _decision_contrasts() -> list[dict]:
    return [
        {
            "left": "unbounded_winner_fill",
            "right": "broad_rff",
            "k": 5,
            "metric": metric,
            "estimate": estimate,
            "lower_95": lower,
            "upper_95": upper,
            "by_cohort": {"winner_fill": estimate, "bounded": estimate},
        }
        for metric, estimate, lower, upper in (
            ("mean_utility_regret_r2", -0.03, -0.04, -0.01),
            ("opportunity_regret_r2", 0.0, -0.003, 0.004),
            ("material_precision", 0.02, -0.01, 0.05),
        )
    ]


def test_top5_justification_requires_every_frozen_check():
    passing = run._decision(_decision_rows(), _decision_contrasts())
    failing_contrasts = _decision_contrasts()
    failing_contrasts[0]["by_cohort"]["bounded"] = 0.001
    failing = run._decision(_decision_rows(), failing_contrasts)

    assert passing["status"] == "top5_confirmation_justified"
    assert all(check["pass"] for check in passing["checks"])
    assert failing["status"] == "stop_no_top5_confirmation"


def test_source_identity_closes_both_inputs_and_analysis_code():
    sources = run._source_paths(run.DEFAULT_INPUTS)

    assert {
        "protocol",
        "protocol_sha256",
        "runner",
        "focused_test",
        "winner_fill_raw_results.json",
        "bounded_raw_results.json",
    }.issubset(sources)
    assert all(path.exists() for path in sources.values())


def test_terminal_writer_is_hash_bound_and_nonoverwriting(monkeypatch, tmp_path):
    protocol = run._load_protocol()
    loaded = []
    for cohort in run.COHORTS:
        for family in run.FAMILIES:
            for seed in protocol["inputs"][cohort]["seeds"]:
                loaded.append({"cohort": cohort, "panel": _panel(seed, family)})
    monkeypatch.setattr(run, "load_inputs", lambda protocol, roots: loaded)
    monkeypatch.setenv("CONTINUOUS_OI_COMMIT", "b" * 40)
    output = tmp_path / "audit"

    summary = run.run(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert summary["panel_count"] == 60
    assert summary["panel_depth_method_row_count"] == 2400
    assert manifest["artifact_status"] == "completed"
    assert manifest["implementation_commit"] == "b" * 40
    for name, descriptor in manifest["files"].items():
        assert descriptor["sha256"] == run.common._sha256(output / name)
    with pytest.raises(FileExistsError):
        run.run(output)
