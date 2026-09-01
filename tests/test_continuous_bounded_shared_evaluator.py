"""Focused tests for exact shared bounded-shortlist computation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.continuous_bounded_shared_evaluator import run, shared


def _data(seed: int = 77, rows: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 16)).astype(np.float32)
    primitive = X[:, :4].copy()
    y = primitive[:, 0] * primitive[:, 1] + 0.1 * rng.normal(size=rows)
    return X, primitive, y


def test_protocol_and_confirmation_ancestry_are_exact():
    protocol = run._load_protocol()

    run._validate_protocol(protocol)
    generation_protocol = run._validate_confirmation(
        protocol, run.DEFAULT_CONFIRMATION_ROOT
    )

    assert generation_protocol["candidate_count_per_panel"] == 16
    assert protocol["speed_gate"]["maximum"] == 0.75


def test_shared_pair_is_fold_exact_to_both_legacy_component_scores():
    X, primitive, y = _data()
    broad = run.kernel.evaluate_selector(
        {"geometry": "rff_mean", "rff_components": 64}, X, primitive, y, 77, 3
    )
    structured = run.kernel.evaluate_selector(
        {"geometry": "multiview", "blocks": ["save", "rff_mean"]},
        X,
        primitive,
        y,
        77,
        3,
    )

    result = shared.evaluate_shared_pair(X, primitive, y, 77, 3)

    assert result["fitted_selector_count"] == 4
    assert result["broad_rff"]["prototype_fit_elided"] is True
    assert result["broad_rff"]["score"] == broad["kernel_alignment"][
        "feature_bandwidths"
    ]["4.0"]["score"]
    assert result["save_rff_current"]["score"] == structured["current_common32"][
        "score"
    ]
    assert [
        fold["broad_rff"]["kernel_alignment"]["feature_bandwidths"]["4.0"][
            "score"
        ]
        for fold in result["folds"]
    ] == [
        fold["kernel_alignment"]["feature_bandwidths"]["4.0"]["score"]
        for fold in broad["folds"]
    ]


def test_shared_pair_fits_exactly_one_selector_per_fold(monkeypatch):
    X, primitive, y = _data(rows=160)
    real_clone = shared.clone
    calls = []

    def counted_clone(estimator):
        calls.append(estimator)
        return real_clone(estimator)

    monkeypatch.setattr(shared, "clone", counted_clone)

    result = shared.evaluate_shared_pair(X, primitive, y, 91, 0)

    assert len(calls) == 4
    assert all(row["fitted_selector_count"] == 1 for row in result["folds"])


def test_shared_pair_rejects_misaligned_or_nonfinite_inputs():
    X, primitive, y = _data(rows=160)

    with pytest.raises(ValueError, match="not aligned"):
        shared.evaluate_shared_pair(X, primitive[:-1], y, 1, 0)
    changed = X.copy()
    changed[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        shared.evaluate_shared_pair(changed, primitive, y, 1, 0)


def _panel(family: str, ratio: float = 0.5, difference: float = 0.0) -> dict:
    rows = [
        {
            "legacy_broad_rff_score": 0.3,
            "shared_broad_rff_score": 0.3 + difference,
            "legacy_save_rff_current_score": 0.4,
            "shared_save_rff_current_score": 0.4 + difference,
            "legacy_wall_seconds": 2.0,
            "shared_wall_seconds": 2.0 * ratio,
        }
        for _ in range(16)
    ]
    return {
        "family": family,
        "candidate_count": 16,
        "candidate_rows": rows,
        "maximum_broad_score_absolute_difference": abs(difference),
        "maximum_save_score_absolute_difference": abs(difference),
        "rank_groups_exact": {
            "broad_rff": difference == 0.0,
            "save_rff_current": difference == 0.0,
            "bounded_2k": difference == 0.0,
        },
        "legacy_panel_wall_seconds": 32.0,
        "shared_panel_wall_seconds": 32.0 * ratio,
        "shared_over_legacy_wall_ratio": ratio,
    }


def test_summary_requires_exactness_completeness_and_speed():
    protocol = run._load_protocol()
    passing = [_panel(family) for family in run.FAMILIES]
    slow = [_panel(family, ratio=0.8) for family in run.FAMILIES]
    changed = [_panel(family) for family in run.FAMILIES]
    changed[-1] = _panel(run.FAMILIES[-1], difference=2.0e-12)

    assert run.summarize(protocol, passing)["decision"]["status"] == "pass"
    assert run.summarize(protocol, slow)["decision"]["status"] == "fail"
    assert run.summarize(protocol, changed)["decision"]["status"] == "fail"


def test_rank_only_panel_adapter_supports_the_inherited_bounded_helper():
    rows = [
        {
            "candidate_id": f"candidate_{position:02d}",
            "legacy_broad_rff_score": float(16 - position),
            "shared_broad_rff_score": float(16 - position),
            "legacy_save_rff_current_score": float(position),
            "shared_save_rff_current_score": float(position),
        }
        for position in range(16)
    ]

    panel = run._score_panel(rows, "shared")
    groups = run.audit_parent.ranking_groups(panel, "bounded_2k", 5)

    assert groups[0] == (0,)
    assert groups[1] == (9,)


def test_terminal_writer_is_hash_bound_and_nonoverwriting(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTINUOUS_OI_COMMIT", "a" * 40)
    monkeypatch.setattr(run, "_repository_head", lambda: "a" * 40)
    monkeypatch.setattr(
        run,
        "evaluate_family",
        lambda protocol, family: _panel(str(family)),
    )
    output = tmp_path / "shared"

    summary = run.run(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert summary["candidate_row_count"] == 96
    assert manifest["checkpoint_count"] == 6
    for name, descriptor in manifest["files"].items():
        assert descriptor["sha256"] == run.common._sha256(output / name)
    with pytest.raises(FileExistsError):
        run.run(output)
