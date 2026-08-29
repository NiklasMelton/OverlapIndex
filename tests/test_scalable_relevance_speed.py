from __future__ import annotations

import numpy as np
import pytest

from experiments.scalable_relevance_kmeans.scalable import tiled_oi_score
from experiments.scalable_relevance_kmeans.scalable_v2 import (
    FastScalableRelevanceKMeans,
    fast_tiled_oi_score,
)


def _fixture(seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 24)
    signal = np.repeat(np.asarray([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]]), 24, axis=0)
    X = np.concatenate(
        [
            signal + rng.normal(scale=0.5, size=(72, 2)),
            rng.normal(scale=3.0, size=(72, 10)),
        ],
        axis=1,
    ).astype(np.float32)
    return X, labels


def _selector(**kwargs: object) -> FastScalableRelevanceKMeans:
    return FastScalableRelevanceKMeans(
        k_per_class=4,
        kmeans_kwargs={"random_state": 23},
        **kwargs,
    )


def test_fast_scorer_is_exactly_equivalent_to_v1_event_rule() -> None:
    X, y = _fixture()
    fitted = _selector().fit(X, y)
    fast, diagnostics = fast_tiled_oi_score(
        X,
        y,
        centers=fitted.centers_,
        owners=fitted.owners_,
        weights=fitted.weights_,
        row_cap=13,
    )
    slow, _ = tiled_oi_score(
        X,
        y,
        centers=fitted.centers_,
        owners=fitted.owners_,
        weights=fitted.weights_,
        row_cap=13,
    )
    assert fast == slow
    assert diagnostics["matmul_count"] == 6
    assert diagnostics["scoring_algorithm"] == "single_matmul_per_row_tile"


def test_float32_path_and_deterministic_state() -> None:
    X, y = _fixture()
    first = _selector().fit(X, y)
    second = _selector().fit(X, y)
    assert first.diagnostics_["input_dtype"] == "float32"
    assert first.diagnostics_["state_sha256"] == second.diagnostics_["state_sha256"]
    np.testing.assert_array_equal(first.centers_, second.centers_)
    np.testing.assert_array_equal(first.weights_, second.weights_)
    assert first.score_fixed(X, y) == second.score_fixed(X, y)


def test_capped_margin_is_deterministic_and_bounded_per_class() -> None:
    X, y = _fixture()
    first = _selector(margin_rows_per_class=7).fit(X, y)
    second = _selector(margin_rows_per_class=7).fit(X, y)
    assert first.diagnostics_["margin_row_count"] == 21
    assert first.diagnostics_["state_sha256"] == second.diagnostics_["state_sha256"]
    np.testing.assert_array_equal(first.weights_, second.weights_)


def test_no_lloyd_variant_records_zero_update() -> None:
    X, y = _fixture()
    fitted = _selector(lloyd_update=False, margin_rows_per_class=7).fit(X, y)
    assert fitted.diagnostics_["weighted_lloyd_update_count"] == 0
    assert fitted.diagnostics_["center_update_rms"] == 0.0
    assert np.isfinite(fitted.score_fixed(X, y))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"fast_scoring": 1}, "strict bools"),
        ({"lloyd_update": 0}, "strict bools"),
        ({"margin_rows_per_class": 0}, "positive int"),
    ],
)
def test_invalid_speed_configuration_fails_loudly(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _selector(**kwargs)


def test_score_fixed_does_not_mutate_fitted_state() -> None:
    X, y = _fixture()
    fitted = _selector().fit(X, y)
    before = fitted.diagnostics_["state_sha256"]
    centers = fitted.centers_.copy()
    weights = fitted.weights_.copy()
    fitted.score_fixed(X[::-1], y[::-1])
    assert fitted.diagnostics_["state_sha256"] == before
    np.testing.assert_array_equal(fitted.centers_, centers)
    np.testing.assert_array_equal(fitted.weights_, weights)
