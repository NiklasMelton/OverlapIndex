from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.m50_backbone_ranking import conditioning_adapter as canonical
from experiments.m50_fast_conditioning.benchmark import (
    fit_fast_m50_state,
    project_frozen_runtime,
    uniform_lower_median,
)


@pytest.mark.parametrize("n_rows", [3, 4, 5, 6])
def test_partition_lower_median_exactly_matches_frozen_uniform_weight_path(n_rows: int) -> None:
    values = np.asarray(
        [[float((row * 7 + column * 3) % 5) for column in range(4)] for row in range(n_rows)]
    )
    weights = np.full(n_rows, 1.0 / n_rows, dtype=np.float64)
    expected = canonical._weighted_median(values, weights)
    observed = uniform_lower_median(values, weights)
    assert np.array_equal(observed, expected)


def test_partition_lower_median_rejects_nonuniform_weights() -> None:
    values = np.arange(12, dtype=float).reshape(4, 3)
    with pytest.raises(ValueError, match="uniform"):
        uniform_lower_median(values, np.asarray([0.1, 0.2, 0.3, 0.4]))


def test_fast_vector_state_is_exactly_equal_to_canonical_m50_geometry() -> None:
    rng = np.random.default_rng(20260829)
    labels = np.repeat(np.asarray(["a", "b", "c", "d"], dtype=object), 8)
    values = rng.normal(size=(labels.size, 7)).astype(np.float32)
    canonical_state = canonical._fit_conditioner(
        np.asarray(values, dtype=np.float64),
        labels,
        estimator="pooled_mad",
        weighting="sample_weighted_rows",
        gamma=0.5,
    )
    fast_state = fit_fast_m50_state(values, labels)
    assert np.array_equal(fast_state.mean, canonical_state.mean_)
    assert np.array_equal(fast_state.raw_variance, np.diag(canonical_state.covariance_))
    assert np.array_equal(fast_state.diagonal_scale, np.diag(canonical_state.transform_))
    assert np.array_equal(fast_state.transform(values), canonical_state.transform(values))


def test_runtime_projection_is_closed_and_does_not_claim_frozen_gate() -> None:
    rows = []
    for arm in ("baseline", "nonlinearity_full", "nuisance_full"):
        for model in (
            "dinov2-small", "deit-tiny", "convnext-tiny", "mobilenetv3-large",
            "openclip-vit-b-32", "resnet50", "efficientnet-b0", "swin-tiny",
            "vit-small-16", "densenet121",
        ):
            for budget in (64, 128, 256, 512, 640):
                for repeat in range(5):
                    identity = {
                        "backbone": model,
                        "arm": arm,
                        "budget": budget,
                        "repeat": repeat,
                    }
                    rows.append(
                        {
                            **identity,
                            "candidate_id": "M1-SW",
                            "total_wall_seconds": 3.0,
                            "conditioning_fit_wall_seconds": 2.0,
                        }
                    )
                    rows.append(
                        {
                            **identity,
                            "candidate_id": "LP-FULL",
                            "total_wall_seconds": 2.0,
                            "conditioning_fit_wall_seconds": 0.0,
                        }
                    )
    projected = project_frozen_runtime(rows, conditioning_speedup=4.0)
    assert projected["point_target_met"] is True
    assert projected["conditioning_optimization_alone_can_meet_point_target"] is True
    assert projected["frozen_gate_status"] == "not_evaluated_by_microbenchmark"
    assert json.dumps(projected, sort_keys=True)
