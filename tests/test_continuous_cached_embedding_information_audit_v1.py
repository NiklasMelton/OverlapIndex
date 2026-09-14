import json

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from experiments.continuous_cached_embedding_information_audit_v1 import run


def _protocol():
    return json.loads(run.PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_protocol_freezes_reference_only_factorial():
    protocol = _protocol()
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["reserved_backbones_evaluated"] is False
    assert len(protocol["targets"]) * len(protocol["backbones"]) * len(protocol["evaluated_views"]) == 48
    assert {view["training_rows"] for view in protocol["evaluated_views"]} == {600, 2000}
    assert {view["pca_rank"] for view in protocol["evaluated_views"]} == {32, 128, None}
    assert {view["row_l2"] for view in protocol["evaluated_views"]} == {True, False}


def test_preprocessor_is_deterministic_and_preserves_full_dimension():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(80, 20))
    Y = rng.normal(size=(12, 20))
    first = run._fit_preprocessor(X, Y, row_l2=False, pca_rank=None, seed=3)
    second = run._fit_preprocessor(X, Y, row_l2=False, pca_rank=None, seed=99)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2]["output_dimension"] == 20
    reduced = run._fit_preprocessor(X, Y, row_l2=True, pca_rank=8, seed=3)
    repeated = run._fit_preprocessor(X, Y, row_l2=True, pca_rank=8, seed=3)
    np.testing.assert_array_equal(reduced[0], repeated[0])
    assert reduced[2]["output_dimension"] == 8


def test_shared_knn_predictions_match_sklearn():
    rng = np.random.default_rng(23)
    X_train = rng.normal(size=(35, 6))
    y_train = rng.normal(size=35)
    X_other = rng.normal(size=(9, 6))
    observed = run._distance_predictions(X_train, y_train, X_other, [3, 7])
    for key, prediction in observed.items():
        estimator = KNeighborsRegressor(n_neighbors=key[0], weights=key[1])
        estimator.fit(X_train, y_train)
        np.testing.assert_allclose(prediction, estimator.predict(X_other), rtol=0.0, atol=1.0e-12)


def test_contrast_effect_detects_new_linear_probe_regret():
    def summary(view, linear, envelope):
        rho = float(np.corrcoef(np.argsort(np.argsort(linear)), np.argsort(np.argsort(envelope)))[0, 1])
        return {
            "target_key": "target", "view_id": view,
            "candidates": [
                {"candidate_id": name, "linear_r2": l, "envelope_r2": e, "nonlinear_gain_r2": e - l, "best_head_ids": []}
                for name, l, e in zip(("a", "b", "c"), linear, envelope)
            ],
            "linear_top1_regret_r2": max(envelope) - envelope[int(np.argmax(linear))],
            "linear_to_envelope_spearman": rho,
        }
    summaries = {
        ("target", "before"): summary("before", [0.7, 0.6, 0.5], [0.7, 0.6, 0.5]),
        ("target", "after"): summary("after", [0.7, 0.6, 0.5], [0.7, 0.74, 0.5]),
    }
    effect = run._contrast_effect(summaries, "target", "before", "after", _protocol()["effects"])
    assert effect["material"] is True
    assert effect["target_linear_top1_regret_r2"] > 0.01
