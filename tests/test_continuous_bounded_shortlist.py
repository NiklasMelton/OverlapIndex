"""Outcome-blind tests for bounded continuous shortlist reranking."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from experiments.continuous_bounded_shortlist import run


def _candidate(
    position: int,
    *,
    broad: float,
    current: float,
    envelope: float,
) -> dict:
    return {
        "candidate_id": f"candidate_{position:02d}",
        "best_head_envelope_r2": envelope,
        "heads": {
            "linear_ridge": {
                "test_r2": envelope - 0.02,
                "wall_seconds": 0.01 + position / 10_000,
            }
        },
        "selectors": {
            "rff64_mean8_k4": {
                "wall_seconds": 0.02 + position / 10_000,
                "kernel_alignment": {
                    "score": broad,
                    "feature_bandwidths": {"4.0": {"score": broad}},
                },
            },
            "save_rffmean_k4": {
                "wall_seconds": 0.03 + position / 10_000,
                "current_common32": {"score": current},
            },
        },
    }


def _panel(seed: int = 2411, family: str = "linear") -> dict:
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
                broad=float(broad[position]),
                current=float(current[position]),
                envelope=float(envelope[position]),
            )
            for position in range(16)
        ],
    }


def test_protocol_freezes_unused_seeds_depth_methods_and_ancestry():
    protocol = run._load_protocol()

    run._validate_protocol_against_code(protocol)
    run._validate_ancestry(protocol)

    assert tuple(protocol["seeds"]) == run.NEW_SEEDS
    assert run.PRIMARY_SHORTLIST_SIZE == 3
    assert run.RETRIEVAL_DEPTH == 6
    assert protocol["decision_rule"]["eligible_method"] == "bounded6_rerank"


def test_canonically_serialized_protocol_remains_valid():
    protocol = json.loads(json.dumps(run._load_protocol(), sort_keys=True))

    run._validate_protocol_against_code(protocol)


def test_bounded_reranker_cannot_promote_candidate_outside_broad_pool():
    panel = _panel()

    bounded = run.ranking_groups(panel, "bounded6_rerank")
    unbounded = run.ranking_groups(panel, "unbounded_winner_fill")

    assert bounded[:3] == [(0,), (5,), (4,)]
    assert unbounded[:3] == [(0,), (15,), (14,)]
    assert all(index in set(range(6)) for group in bounded[:3] for index in group)


def test_broad_pool_includes_complete_rank_group_crossing_cutoff():
    panel = _panel()
    candidates = panel["candidates"]
    cutoff_score = candidates[5]["selectors"]["rff64_mean8_k4"][
        "kernel_alignment"
    ]["feature_bandwidths"]["4.0"]["score"]
    candidates[6]["selectors"]["rff64_mean8_k4"]["kernel_alignment"][
        "feature_bandwidths"
    ]["4.0"]["score"] = cutoff_score
    candidates[6]["selectors"]["save_rffmean_k4"]["current_common32"][
        "score"
    ] = 2.0

    broad_groups = run.parent._score_groups(
        run.parent._component_values(panel, "broad_rff")
    )
    pool = run._broad_pool(broad_groups)
    bounded = run.ranking_groups(panel, "bounded6_rerank")

    assert pool == set(range(7))
    assert bounded[1] == (6,)


def test_bounded_ranking_is_candidate_row_order_invariant():
    panel = _panel()
    expected = [
        tuple(sorted(panel["candidates"][index]["candidate_id"] for index in group))
        for group in run.ranking_groups(panel, "bounded6_rerank")
    ]
    permutation = [7, 3, 15, 0, 9, 1, 11, 2, 13, 5, 14, 4, 10, 8, 12, 6]
    shuffled = copy.deepcopy(panel)
    shuffled["candidates"] = [panel["candidates"][index] for index in permutation]
    observed = [
        tuple(sorted(shuffled["candidates"][index]["candidate_id"] for index in group))
        for group in run.ranking_groups(shuffled, "bounded6_rerank")
    ]

    assert observed == expected


def test_panel_metric_preserves_broad_top1_and_charges_both_components():
    panel = _panel()

    broad = run.panel_metric(panel, "broad_rff")
    current = run.panel_metric(panel, "save_rff_current")
    bounded = run.panel_metric(panel, "bounded6_rerank")

    assert bounded["rank_groups"][0] == broad["rank_groups"][0]
    assert bounded["top_1_opportunity_regret_r2"] == broad[
        "top_1_opportunity_regret_r2"
    ]
    assert bounded["descriptive_panel_runtime_seconds"] == pytest.approx(
        broad["descriptive_panel_runtime_seconds"]
        + current["descriptive_panel_runtime_seconds"]
    )


def _decision_rows() -> list[dict]:
    rows = []
    for family_index, family in enumerate(run.FAMILIES):
        panel_id = f"{family}__s2411"
        for method in run.METHODS:
            rows.append(
                {
                    "method": method,
                    "panel_id": panel_id,
                    "family": family,
                    "top_1_opportunity_regret_r2": 0.01,
                    "top_3_opportunity_regret_r2": 0.02,
                    "top_3_mean_utility_regret_r2": 0.03,
                    "material_pair_concordance": 0.7,
                    "spearman": 0.6,
                    "family_index": family_index,
                }
            )
    return rows


def _decision_contrasts() -> list[dict]:
    return [
        {
            "left": "bounded6_rerank",
            "right": right,
            "metric": metric,
            "lower_95": lower,
            "upper_95": upper,
        }
        for right, metric, lower, upper in (
            ("broad_rff", "top_3_mean_utility_regret_r2", -0.04, -0.01),
            ("save_rff_current", "top_3_mean_utility_regret_r2", -0.03, -0.005),
            ("unbounded_winner_fill", "top_3_mean_utility_regret_r2", -0.02, 0.01),
            ("broad_rff", "top_1_opportunity_regret_r2", 0.0, 0.0),
            ("broad_rff", "top_3_opportunity_regret_r2", -0.01, 0.005),
            ("save_rff_current", "material_pair_concordance", -0.01, 0.04),
            ("save_rff_current", "spearman", -0.03, 0.05),
        )
    ]


def test_decision_requires_primary_safety_and_family_degradation_gates():
    passing = run._decision(_decision_rows(), _decision_contrasts())
    degraded = _decision_rows()
    target = next(
        row
        for row in degraded
        if row["method"] == "bounded6_rerank" and row["family"] == "sinusoid"
    )
    target["top_3_mean_utility_regret_r2"] += 0.031
    failing = run._decision(degraded, _decision_contrasts())

    assert passing["status"] == "pass"
    assert all(check["pass"] for check in passing["checks"])
    assert failing["status"] == "fail"
    assert not next(
        check
        for check in failing["checks"]
        if check["criterion"] == "family_mean_degradation_bounded"
    )["pass"]


def test_decision_fails_when_primary_interval_crosses_zero():
    contrasts = _decision_contrasts()
    contrasts[0]["upper_95"] = 0.001

    decision = run._decision(_decision_rows(), contrasts)

    assert decision["status"] == "fail"
    assert not decision["checks"][0]["pass"]


def test_summarize_rejects_duplicate_panel_grid_before_bootstrap():
    protocol = run._load_protocol()
    panels = [
        _panel(seed, family) for seed in run.NEW_SEEDS for family in run.FAMILIES
    ]
    panels[-1] = copy.deepcopy(panels[0])

    with pytest.raises(ValueError, match="30 unique panels"):
        run.summarize(protocol, panels)


def test_source_identity_includes_protocol_test_parent_and_oi_sources():
    sources = run._source_paths(run.DEFAULT_MOTIVATION_ROOT)

    assert {
        "protocol",
        "protocol_sha256",
        "focused_test",
        "parent_runner",
        "kernel_runner",
        "overlapindex_ContinuousOverlapIndex.py",
        "project_configuration",
    }.issubset(sources)
    assert all(path.exists() for path in sources.values())


def test_terminal_writer_emits_complete_hash_bound_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTINUOUS_OI_COMMIT", "a" * 40)
    monkeypatch.setattr(
        run.kernel,
        "run_panel",
        lambda protocol, seed, family: _panel(int(seed), str(family)),
    )
    output = tmp_path / "artifact"

    summary = run.run(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    raw = json.loads((output / "raw_results.json").read_text(encoding="utf-8"))

    assert summary["panel_count"] == 30
    assert raw["artifact_status"] == "completed"
    assert raw["implementation_commit"] == "a" * 40
    assert manifest["artifact_status"] == "completed"
    assert manifest["checkpoint_count"] == 30
    for name, descriptor in manifest["files"].items():
        assert descriptor["sha256"] == run.common._sha256(output / name)
    with pytest.raises(FileExistsError):
        run.run(output)
