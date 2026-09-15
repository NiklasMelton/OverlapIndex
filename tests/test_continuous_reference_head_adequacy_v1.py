"""Focused tests for continuous reference-head adequacy."""

from __future__ import annotations

import numpy as np

from experiments.continuous_reference_head_adequacy_v1 import run


def test_protocol_closes_calibration_grid():
    protocol = run.load_protocol()
    assert tuple(protocol["heads"]) == run.HEADS
    assert protocol["artifact_contract"]["expected_head_rows"] == 40


def test_screened_heads_recover_known_feature_and_pair():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(240, 8))
    linear_y = 2.0 * X[:, 3] + rng.normal(scale=0.05, size=240)
    pair_y = 2.0 * X[:, 2] * X[:, 6] + rng.normal(scale=0.05, size=240)
    linear = run.ScreenedLinearRidge(alpha=0.1).fit(X, linear_y)
    pair = run.ScreenedPairRidge(alpha=0.1).fit(X, pair_y)
    assert linear.selected_feature_ == 3
    assert set(pair.selected_pair_) == {2, 6}
    assert np.corrcoef(linear.predict(X), linear_y)[0, 1] > 0.99
    assert np.corrcoef(pair.predict(X), pair_y)[0, 1] > 0.99

