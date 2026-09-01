"""Outcome-blind tests for the frozen winner-plus-fill experiment."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.continuous_winner_fill_shortlist import run


def _candidate(
    position: int,
    *,
    broad: float,
    mixed: float,
    current: float,
    linear: float,
    envelope: float,
) -> dict:
    return {
        "candidate_id": f"candidate_{position:02d}",
        "best_head_envelope_r2": envelope,
        "heads": {
            "linear_ridge": {
                "test_r2": linear,
                "wall_seconds": 0.01 + position / 10_000,
            }
        },
        "selectors": {
            "rff64_mean8_k4": {
                "wall_seconds": 0.02 + position / 10_000,
                "kernel_alignment": {
                    "score": mixed,
                    "feature_bandwidths": {"4.0": {"score": broad}},
                },
            },
            "save_rffmean_k4": {
                "wall_seconds": 0.03 + position / 10_000,
                "current_common32": {"score": current},
            },
        },
    }


def _panel(seed: int = 1907, family: str = "linear") -> dict:
    broad = np.linspace(0.95, 0.20, 16)
    current = np.linspace(0.20, 0.95, 16)
    mixed = (broad + current) / 2.0
    envelope = np.linspace(0.90, 0.30, 16)
    return {
        "seed": seed,
        "family": family,
        "candidate_count": 16,
        "candidates": [
            _candidate(
                position,
                broad=float(broad[position]),
                mixed=float(mixed[position]),
                current=float(current[position]),
                linear=float(envelope[position] - 0.02),
                envelope=float(envelope[position]),
            )
            for position in range(16)
        ],
    }


def test_protocol_closes_new_seeds_methods_shortlists_and_ancestry():
    protocol = run._load_protocol()

    run._validate_protocol_against_code(protocol)
    run._validate_ancestry(protocol)

    assert tuple(protocol["seeds"]) == run.NEW_SEEDS
    assert tuple(protocol["ranking_methods"]) == run.METHODS
    assert tuple(protocol["shortlist_sizes"]) == run.SHORTLIST_SIZES
    assert protocol["decision_rule"]["eligible_method"] == "winner_fill"


def test_score_groups_use_frozen_tolerance_and_partition_once():
    scores = [0.8, 0.8 - 0.5e-12, 0.8 - 1.5e-12, 0.1]

    groups = run._score_groups(scores)

    assert groups == [(0, 1), (2,), (3,)]
    assert sorted(index for group in groups for index in group) == list(range(4))


def test_winner_fill_preserves_broad_top_group_then_uses_current_order():
    panel = _panel()
    panel["candidates"][1]["selectors"]["rff64_mean8_k4"]["kernel_alignment"][
        "feature_bandwidths"
    ]["4.0"]["score"] = 0.95 - 0.5e-12
    panel["candidates"][5]["selectors"]["save_rffmean_k4"]["current_common32"][
        "score"
    ] = 2.0

    groups = run.ranking_groups(panel, "winner_fill")

    assert groups[0] == (0, 1)
    assert groups[1] == (5,)
    assert sorted(index for group in groups for index in group) == list(range(16))


def test_winner_fill_is_invariant_to_candidate_row_order():
    panel = _panel()
    expected = [
        tuple(panel["candidates"][index]["candidate_id"] for index in group)
        for group in run.ranking_groups(panel, "winner_fill")
    ]
    permutation = [7, 3, 15, 0, 9, 1, 11, 2, 13, 5, 14, 4, 10, 8, 12, 6]
    shuffled = copy.deepcopy(panel)
    shuffled["candidates"] = [panel["candidates"][index] for index in permutation]
    observed = [
        tuple(sorted(shuffled["candidates"][index]["candidate_id"] for index in group))
        for group in run.ranking_groups(shuffled, "winner_fill")
    ]

    assert [tuple(sorted(group)) for group in expected] == observed


def test_top_k_metrics_average_every_subset_at_tied_cutoff():
    reference = np.asarray([1.0, 0.8, 0.6, 0.4])
    groups = [(0,), (1, 2), (3,)]

    result = run._top_k_metrics(reference, groups, 2)

    # The admitted shortlists are {0, 1} and {0, 2}.
    assert result["tie_averaged_subset_count"] == 2
    assert result["opportunity_regret_r2"] == 0.0
    assert result["mean_utility_regret_r2"] == pytest.approx(0.05)
    assert result["exact_best_coverage"] == 1.0


def test_panel_metric_retains_exact_broad_top1_and_charges_both_components():
    panel = _panel()

    broad = run.panel_metric(panel, "broad_rff")
    fused = run.panel_metric(panel, "winner_fill")
    current = run.panel_metric(panel, "save_rff_current")

    assert fused["top_1_opportunity_regret_r2"] == broad[
        "top_1_opportunity_regret_r2"
    ]
    assert fused["rank_groups"][0] == broad["rank_groups"][0]
    assert fused["descriptive_panel_runtime_seconds"] == pytest.approx(
        broad["descriptive_panel_runtime_seconds"]
        + current["descriptive_panel_runtime_seconds"]
    )


def _decision_rows() -> list[dict]:
    rows = []
    for method in run.METHODS:
        rows.append(
            {
                "method": method,
                "panel_id": "linear__s1907",
                "family": "linear",
                "top_1_opportunity_regret_r2": 0.01,
                "top_3_opportunity_regret_r2": 0.02,
                "top_3_mean_utility_regret_r2": 0.03,
                "material_pair_concordance": 0.7,
                "spearman": 0.6,
            }
        )
    return rows


def _decision_contrasts() -> list[dict]:
    return [
        {
            "left": "winner_fill",
            "right": right,
            "metric": metric,
            "lower_95": lower,
            "upper_95": upper,
        }
        for right, metric, lower, upper in (
            ("broad_rff", "top_3_mean_utility_regret_r2", -0.04, -0.01),
            ("save_rff_current", "top_3_mean_utility_regret_r2", -0.03, -0.005),
            ("broad_rff", "top_1_opportunity_regret_r2", 0.0, 0.0),
            ("broad_rff", "top_3_opportunity_regret_r2", -0.01, 0.005),
            ("save_rff_current", "material_pair_concordance", -0.01, 0.04),
            ("save_rff_current", "spearman", -0.03, 0.05),
        )
    ]


def test_decision_requires_every_frozen_primary_and_safety_check():
    passing = run._decision(_decision_rows(), _decision_contrasts())
    failing_contrasts = _decision_contrasts()
    failing_contrasts[0]["upper_95"] = 0.001
    failing = run._decision(_decision_rows(), failing_contrasts)

    assert passing["status"] == "pass"
    assert all(check["pass"] for check in passing["checks"])
    assert failing["status"] == "fail"
    assert not failing["checks"][0]["pass"]


def test_summarize_rejects_duplicate_or_wrong_panel_identities_before_outcomes():
    protocol = run._load_protocol()
    duplicated = [_panel(seed, family) for seed in run.NEW_SEEDS for family in protocol["families"]]
    duplicated[-1] = copy.deepcopy(duplicated[0])

    with pytest.raises(ValueError, match="duplicate panels"):
        run.summarize(protocol, duplicated)


def test_atomic_text_never_leaves_temporary_file(tmp_path):
    target = tmp_path / "report.md"

    run._atomic_text(target, "complete\n")

    assert target.read_text(encoding="utf-8") == "complete\n"
    assert not (tmp_path / ".report.md.tmp").exists()
