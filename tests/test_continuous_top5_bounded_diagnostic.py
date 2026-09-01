"""Focused tests for the bounded top-five post-confirmation diagnostic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.continuous_top5_bounded_diagnostic import run


def _candidate(position: int, broad: float, current: float, envelope: float) -> dict:
    return {
        "candidate_id": f"candidate_{position:02d}",
        "best_head_envelope_r2": envelope,
        "selectors": {
            "rff64_mean8_k4": {
                "kernel_alignment": {
                    "feature_bandwidths": {"4.0": {"score": broad}}
                }
            },
            "save_rffmean_k4": {"current_common32": {"score": current}},
        },
    }


def _panel(seed: int = 2903, family: str = "linear") -> dict:
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


def test_protocol_and_confirmation_input_are_exact():
    protocol = run._load_protocol()

    run._validate_protocol(protocol)
    panels = run._validate_input(protocol, run.DEFAULT_INPUT)

    assert len(panels) == 30
    assert {(panel["seed"], panel["family"]) for panel in panels} == {
        (seed, family) for seed in run.SEEDS for family in run.FAMILIES
    }


def test_bounded_rule_preserves_winner_and_only_reranks_broad_top_ten():
    panel = _panel()

    groups = run.ranking_groups(panel, "bounded_2k")

    assert groups[0] == (0,)
    assert groups[1:10] == [(9,), (8,), (7,), (6,), (5,), (4,), (3,), (2,), (1,)]
    assert all(index < 10 for group in groups[:10] for index in group)
    assert all(index >= 10 for group in groups[10:] for index in group)


def test_bounded_rule_includes_complete_group_crossing_top_ten_cutoff():
    panel = _panel()
    candidates = panel["candidates"]
    cutoff = candidates[9]["selectors"]["rff64_mean8_k4"]["kernel_alignment"][
        "feature_bandwidths"
    ]["4.0"]["score"]
    candidates[10]["selectors"]["rff64_mean8_k4"]["kernel_alignment"][
        "feature_bandwidths"
    ]["4.0"]["score"] = cutoff
    candidates[10]["selectors"]["save_rffmean_k4"]["current_common32"][
        "score"
    ] = 2.0

    groups = run.ranking_groups(panel, "bounded_2k")

    assert groups[1] == (10,)


def test_panel_metric_uses_best_head_envelope_and_preserves_top_one():
    panel = _panel()
    broad = run.panel_metric(panel, "broad_rff")
    bounded = run.panel_metric(panel, "bounded_2k")

    assert bounded["top_1_opportunity_regret_r2"] == broad[
        "top_1_opportunity_regret_r2"
    ]
    assert bounded["rank_groups"][0] == broad["rank_groups"][0]
    assert len({candidate for group in bounded["rank_groups"] for candidate in group}) == 16


def _decision_rows() -> list[dict]:
    rows = []
    for family in run.FAMILIES:
        for seed_index in range(5):
            panel_id = f"{family}__s{seed_index}"
            for method in run.METHODS:
                rows.append(
                    {
                        "family": family,
                        "panel_id": panel_id,
                        "method": method,
                        "top_1_opportunity_regret_r2": 0.01,
                        "top_5_opportunity_regret_r2": 0.01,
                        "top_5_mean_utility_regret_r2": 0.02,
                        "top_5_material_precision": 0.8,
                        "top_5_exact_best_coverage": 1.0,
                        "top_5_within_0_01_coverage": 1.0,
                    }
                )
    return rows


def _decision_contrasts(family_max: float = -0.01) -> list[dict]:
    by_family = {family: -0.01 for family in run.FAMILIES}
    by_family["triangle_wave"] = family_max
    rows = []
    for right in ("broad_rff", "unbounded_winner_fill"):
        for metric, estimate, lower, upper in (
            ("top_5_mean_utility_regret_r2", -0.02, -0.03, -0.01),
            ("top_5_opportunity_regret_r2", 0.0, -0.003, 0.005),
            ("top_5_material_precision", 0.01, -0.01, 0.03),
        ):
            rows.append(
                {
                    "left": "bounded_2k",
                    "right": right,
                    "metric": metric,
                    "estimate": estimate,
                    "lower_95": lower,
                    "upper_95": upper,
                    "by_family": by_family,
                }
            )
    return rows


def test_decision_requires_primary_and_every_safety_check():
    passing = run._decision(_decision_rows(), _decision_contrasts())
    failing = run._decision(_decision_rows(), _decision_contrasts(family_max=0.031))

    assert passing["status"] == "bounded_top5_confirmation_justified"
    assert all(check["pass"] for check in passing["checks"])
    assert failing["status"] == "stop_no_bounded_top5_confirmation"
    assert not next(
        check
        for check in failing["checks"]
        if check["criterion"] == "family_mean_degradation_bounded"
    )["pass"]


def test_terminal_writer_is_hash_bound_and_nonoverwriting(monkeypatch, tmp_path):
    panels = [
        _panel(seed, family)
        for seed in run.SEEDS
        for family in run.FAMILIES
    ]
    monkeypatch.setattr(run, "_validate_input", lambda protocol, root: panels)
    monkeypatch.setenv("CONTINUOUS_OI_COMMIT", "b" * 40)
    monkeypatch.setattr(run, "_repository_head", lambda: "b" * 40)
    output = tmp_path / "diagnostic"

    summary = run.run(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert summary["panel_count"] == 30
    assert summary["panel_method_row_count"] == 90
    assert manifest["artifact_status"] == "completed"
    for name, descriptor in manifest["files"].items():
        assert descriptor["sha256"] == run.common._sha256(output / name)
    with pytest.raises(FileExistsError):
        run.run(output)


def test_terminal_writer_rejects_commit_argument_that_is_not_head(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "_validate_input", lambda protocol, root: [])
    monkeypatch.setenv("CONTINUOUS_OI_COMMIT", "b" * 40)
    monkeypatch.setattr(run, "_repository_head", lambda: "c" * 40)

    with pytest.raises(ValueError, match="must equal the frozen Git HEAD"):
        run.run(tmp_path / "diagnostic")
