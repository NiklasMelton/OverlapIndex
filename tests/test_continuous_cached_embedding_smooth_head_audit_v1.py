import json

import numpy as np
from sklearn.metrics import r2_score

from experiments.continuous_cached_embedding_smooth_head_audit_v1 import run
from experiments.continuous_reference_library_qualification import evaluator as qualified


def _protocol():
    return json.loads(run.PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_protocol_freezes_smooth_reference_only_grid():
    protocol = _protocol()
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["reserved_backbones_evaluated"] is False
    assert len(protocol["targets"]) * len(protocol["backbones"]) * len(protocol["evaluated_views"]) == 24
    assert {view["pca_rank"] for view in protocol["evaluated_views"]} == {128, None}


def test_precomputed_rbf_matches_direct_kernel():
    rng = np.random.default_rng(41)
    X_train = rng.normal(size=(45, 7))
    y_train = rng.normal(size=45)
    X_other = rng.normal(size=(13, 7))
    params = {"C_standardized": 3.0, "epsilon_standardized": 0.15, "bandwidth_factor": 1.0}
    direct = qualified._fit_predict("rbf_svr", params, X_train, y_train, X_other)
    kernels, _ = run._distance_views(X_train, X_other, [1.0])
    observed = run._fit_precomputed(kernels[1.0][0], y_train, kernels[1.0][1], params)
    np.testing.assert_allclose(observed, direct, rtol=0.0, atol=1.0e-3)


def test_ranking_summary_marks_material_top1_conflict():
    protocol = _protocol()
    cells = []
    for name, linear, rbf in (("a", 0.70, 0.70), ("b", 0.60, 0.75), ("c", 0.50, 0.50)):
        scores = {"linear_ridge": linear, "knn": linear - 0.1, "extra_trees": linear - 0.05, "rbf_svr": rbf}
        cells.append({"target_key": "target", "candidate_id": name, "view": {"view_id": "view"}, "head_evaluation_r2": scores, "envelope_r2": max(scores.values()), "best_head_ids": []})
    summary = run._ranking_summary(cells, "target", "view", protocol)
    assert summary["material_rank_conflict"] is True
    assert summary["linear_top1_regret_r2"] >= 0.05 - 1.0e-12
