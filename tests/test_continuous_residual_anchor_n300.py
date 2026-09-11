"""Focused pre-outcome tests for the residual-anchor diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.continuous_residual_anchor_n300 import run


def test_protocol_is_frozen_and_has_no_reserved_backbones() -> None:
    protocol = run.load_protocol()
    assert protocol["status"] == "frozen_continuous_residual_anchor_n300_before_outcomes"
    assert tuple(protocol["methods"]) == run.METHODS
    assert protocol["roster"]["reserved_backbones"] == []
    assert protocol["parent_protocol_sha256"] == run._parent_protocol_hash()


def test_selection_split_is_deterministic_disjoint_and_twenty_percent() -> None:
    first = run._deterministic_selection_split(225, 119)
    second = run._deterministic_selection_split(225, 119)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(first[0]) == 180
    assert len(first[1]) == 45
    assert not set(first[0]).intersection(first[1])
    assert set(first[0]).union(first[1]) == set(range(225))


def test_oof_base_predictions_are_reproducible_and_not_in_sample_fit() -> None:
    values = np.arange(120, dtype=np.float64).reshape(40, 3)
    targets = 0.25 * values[:, 0] - 0.5 * values[:, 1] + np.sin(values[:, 2])
    first, first_diag = run._oof_base_predictions(values, targets, 77)
    second, second_diag = run._oof_base_predictions(values, targets, 77)
    assert np.array_equal(first, second)
    assert first_diag == second_diag
    full, _ = run._fit_full_base(values, values, targets, 77)
    assert not np.array_equal(first, full)
    assert np.all(np.isfinite(targets - first))


def test_standardized_residual_targets_are_fit_only_transform() -> None:
    raw = np.asarray([-2.0, 0.5, 3.0, 4.5], dtype=np.float64)
    standardized, diagnostics = run._standardize_residual_targets(raw)
    assert np.isclose(np.mean(standardized), 0.0)
    assert np.isclose(np.std(standardized), 1.0)
    assert diagnostics["center"] == np.mean(raw)
    assert diagnostics["scale"] == np.std(raw)
    with pytest.raises(ValueError):
        run._standardize_residual_targets(np.asarray([0.0, np.nan]))


def test_no_harm_shrinkage_is_explicit_and_wins_ties() -> None:
    base = np.arange(10, dtype=np.float64)
    targets = base.copy()
    correction = np.zeros_like(base)
    fit_rows = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    selection_rows = np.asarray([8, 9], dtype=np.int64)
    selected, evidence = run._select_shrinkage(base, targets, correction, fit_rows, selection_rows)
    assert selected == 0.0
    assert [row["shrinkage"] for row in evidence["grid"]] == list(run.SHRINKAGE_GRID)
    assert evidence["selected_shrinkage"] == 0.0


def test_shrinkage_selection_rejects_overlap() -> None:
    rows = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError):
        run._select_shrinkage(rows, rows, rows, rows[:6], rows[5:])


def test_constant_spearman_is_undefined_not_fabricated() -> None:
    result = run.ranking_metrics(np.arange(5, dtype=float), np.ones(5, dtype=float))
    assert result["spearman"] is None
    assert result["spearman_status"] == "undefined_constant_input"


def test_terminal_validation_is_fail_closed_without_artifact(tmp_path) -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        run.validate_artifact(tmp_path)
