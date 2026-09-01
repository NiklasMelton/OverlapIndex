"""Outcome-blind tests for bounded top-five confirmation."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from experiments.continuous_top5_bounded_confirmation import run


def _candidate(position: int, broad: float, current: float, envelope: float) -> dict:
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


def _panel(seed: int = 3407, family: str = "linear") -> dict:
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


def test_protocol_freezes_new_seeds_rule_and_authorizing_diagnostic():
    protocol = run._load_protocol()

    run._validate_protocol(protocol)
    run._validate_diagnostic(protocol, run.DEFAULT_DIAGNOSTIC_ROOT)

    assert tuple(protocol["seeds"]) == run.NEW_SEEDS
    assert protocol["decision_rule"]["candidate"] == "bounded_2k"
    assert not set(run.NEW_SEEDS).intersection(
        run.confirmation_parent.NEW_SEEDS
    )


def test_canonically_serialized_protocol_remains_valid():
    run._validate_protocol(json.loads(json.dumps(run._load_protocol(), sort_keys=True)))


def test_bounded_rule_preserves_winner_and_reranks_only_top_ten():
    panel = _panel()

    groups = run.ranking_groups(panel, "bounded_2k")

    assert groups[0] == (0,)
    assert groups[1:10] == [(9,), (8,), (7,), (6,), (5,), (4,), (3,), (2,), (1,)]
    assert all(index < 10 for group in groups[:10] for index in group)
    assert all(index >= 10 for group in groups[10:] for index in group)


def test_bounded_metric_preserves_top1_and_charges_both_components():
    panel = _panel()

    broad = run.panel_metric(panel, "broad_rff")
    current = run.panel_metric(panel, "save_rff_current")
    bounded = run.panel_metric(panel, "bounded_2k")

    assert bounded["top_1_opportunity_regret_r2"] == broad[
        "top_1_opportunity_regret_r2"
    ]
    assert bounded["descriptive_panel_runtime_seconds"] == pytest.approx(
        broad["descriptive_panel_runtime_seconds"]
        + current["descriptive_panel_runtime_seconds"]
    )


def _decision_rows() -> list[dict]:
    rows = []
    for family in run.FAMILIES:
        for seed in run.NEW_SEEDS:
            panel_id = f"{family}__s{seed}"
            for method in run.METHODS:
                row = {
                    "family": family,
                    "panel_id": panel_id,
                    "method": method,
                    "spearman": 0.7,
                    "material_pair_concordance": 0.8,
                }
                for k in run.DEPTHS:
                    row.update(
                        {
                            f"top_{k}_opportunity_regret_r2": 0.01,
                            f"top_{k}_mean_utility_regret_r2": 0.02,
                            f"top_{k}_material_precision": 0.8,
                        }
                    )
                rows.append(row)
    return rows


def _decision_contrasts(family_max: float = -0.01) -> list[dict]:
    by_family = {family: -0.01 for family in run.FAMILIES}
    by_family["triangle_wave"] = family_max
    rows = []
    for right in ("broad_rff", "unbounded_winner_fill"):
        for metric, estimate, lower, upper in (
            ("top_5_mean_utility_regret_r2", -0.03, -0.05, -0.01),
            ("top_5_opportunity_regret_r2", 0.0, -0.003, 0.004),
            ("top_5_material_precision", 0.02, -0.01, 0.05),
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


def test_decision_requires_primary_and_every_safety_gate():
    passing = run._decision(_decision_rows(), _decision_contrasts())
    failing = run._decision(_decision_rows(), _decision_contrasts(family_max=0.031))

    assert passing["status"] == "pass"
    assert all(check["pass"] for check in passing["checks"])
    assert failing["status"] == "fail"


def test_summary_rejects_duplicate_panel_grid_before_analysis():
    protocol = run._load_protocol()
    panels = [_panel(seed, family) for seed in run.NEW_SEEDS for family in run.FAMILIES]
    panels[-1] = copy.deepcopy(panels[0])

    with pytest.raises(ValueError, match="panel grid is incomplete"):
        run.summarize(protocol, panels)


def test_terminal_writer_is_hash_bound_and_checks_actual_head(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTINUOUS_OI_COMMIT", "d" * 40)
    monkeypatch.setattr(run, "_repository_head", lambda: "d" * 40)
    monkeypatch.setattr(
        run.kernel,
        "run_panel",
        lambda protocol, seed, family: _panel(int(seed), str(family)),
    )
    output = tmp_path / "confirmation"

    summary = run.run(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert summary["panel_count"] == 30
    assert summary["panel_method_row_count"] == 150
    assert manifest["checkpoint_count"] == 30
    assert manifest["implementation_commit"] == "d" * 40
    for name, descriptor in manifest["files"].items():
        assert descriptor["sha256"] == run.common._sha256(output / name)


def test_terminal_writer_rejects_commit_that_is_not_head(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTINUOUS_OI_COMMIT", "d" * 40)
    monkeypatch.setattr(run, "_repository_head", lambda: "e" * 40)

    with pytest.raises(ValueError, match="must equal the frozen Git HEAD"):
        run.run(tmp_path / "confirmation")
