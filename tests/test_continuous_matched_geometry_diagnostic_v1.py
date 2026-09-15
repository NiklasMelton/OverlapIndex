"""Focused tests for the matched-utility continuous geometry diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.continuous_matched_geometry_diagnostic_v1 import run
from experiments.continuous_matched_geometry_diagnostic_v1 import analyze


def test_protocol_closes_grid_and_counts():
    protocol = run.load_protocol()
    assert tuple(protocol["selectors"]["methods"]) == run.METHODS
    assert tuple(protocol["data"]["matched_candidate_ids"]) == run.MATCHED
    assert protocol["artifact_contract"]["panel_count"] == 12
    assert protocol["artifact_contract"]["total_selector_rows"] == 540


@pytest.mark.parametrize("dimension,rows", [(16, 300), (512, 1200)])
def test_geometries_have_matched_analytic_truth_and_exact_recovery(dimension: int, rows: int):
    protocol = run.load_protocol()
    panel = run.generate_panel(protocol, dimension, rows, 401)
    truth = {row["candidate_id"]: row["analytic_r2"] for row in panel["candidates"]}
    assert len({truth[name] for name in run.MATCHED}) == 1
    assert truth["direct_q85"] > truth["signed_q55"] + 0.2
    assert all(abs(row["evaluation_conditional_mean_r2"] - row["analytic_r2"]) < 0.03 for row in panel["candidates"])
    assert all(row["X_train"].shape == (rows, dimension) for row in panel["candidates"])


def test_panel_generation_is_deterministic_and_geometries_differ():
    protocol = run.load_protocol()
    first = run.generate_panel(protocol, 16, 300, 401)
    second = run.generate_panel(protocol, 16, 300, 401)
    for left, right in zip(first["candidates"], second["candidates"]):
        assert np.array_equal(left["X_train"], right["X_train"])
    assert not np.array_equal(first["candidates"][0]["X_train"], first["candidates"][1]["X_train"])


def test_pair_accuracy_counts_ties_as_half_credit():
    scores = {"direct_q85": 1.0, "rotated_q85": 0.5, "anisotropic_q85": 0.0,
              "signed_q85": -1.0, "signed_q55": 0.0}
    assert run._pair_accuracy(scores) == pytest.approx(0.625)


def test_head_envelope_ranking_metrics_use_selected_utility():
    truth = {name: value for name, value in zip(run.CANDIDATES, (0.8, 0.7, 0.6, 0.5, 0.1))}
    scores = {name: value for name, value in zip(run.CANDIDATES, (0.0, 1.0, -1.0, -2.0, -3.0))}
    metrics = analyze.ranking_metrics(scores, truth)
    assert metrics["top1_regret_r2"] == pytest.approx(0.1)
    assert metrics["exact_best"] is False
    assert metrics["selected_candidate_ids"] == ["rotated_q85"]
