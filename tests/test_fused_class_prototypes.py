from __future__ import annotations

import numpy as np
import pytest

from experiments.scalable_relevance_kmeans.fused_prototypes import (
    FusedFastRelevanceKMeans,
    fit_fused_class_prototypes,
    fit_looped_class_prototypes,
)


def _fixture(seed: int = 31) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = (17, 23, 14, 19)
    labels = np.concatenate(
        [np.full(count, f"class-{position}", dtype=object) for position, count in enumerate(counts)]
    )
    X = rng.normal(size=(sum(counts), 24)).astype(np.float32)
    X[:, :4] += np.eye(4, dtype=np.float32)[
        np.concatenate([np.full(count, position) for position, count in enumerate(counts)])
    ]
    return X, labels


def test_fused_centers_exactly_match_looped_reference_with_padding() -> None:
    X, y = _fixture()
    k = {"class-0": 3, "class-1": 5, "class-2": 2, "class-3": 4}
    looped = fit_looped_class_prototypes(X, y, k_per_class=k, seed=7, iterations=3)
    fused = fit_fused_class_prototypes(X, y, k_per_class=k, seed=7, iterations=3)
    np.testing.assert_array_equal(fused.owners, looped.owners)
    np.testing.assert_array_equal(fused.centers, looped.centers)
    assert fused.diagnostics["prototype_count"] == 14
    assert fused.diagnostics["max_rows_per_class"] == 23


def test_fused_fit_is_deterministic() -> None:
    X, y = _fixture()
    first = fit_fused_class_prototypes(X, y, k_per_class=4, seed=19, iterations=2)
    second = fit_fused_class_prototypes(X, y, k_per_class=4, seed=19, iterations=2)
    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(first.owners, second.owners)


def test_class_ownership_is_preserved() -> None:
    X, y = _fixture()
    fitted = fit_fused_class_prototypes(X, y, k_per_class=3, seed=5, iterations=2)
    for label in dict.fromkeys(y.tolist()):
        assert np.count_nonzero(fitted.owners == label) == 3
    assert set(fitted.owners.tolist()) == set(y.tolist())


def test_integrated_looped_and_fused_selectors_match() -> None:
    X, y = _fixture()
    looped = FusedFastRelevanceKMeans(
        k_per_class=4,
        seed=13,
        prototype_iterations=3,
        prototype_backend="looped",
        margin_rows_per_class=8,
    ).fit(X, y)
    fused = FusedFastRelevanceKMeans(
        k_per_class=4,
        seed=13,
        prototype_iterations=3,
        prototype_backend="fused",
        margin_rows_per_class=8,
    ).fit(X, y)
    np.testing.assert_array_equal(fused.centers_, looped.centers_)
    np.testing.assert_array_equal(fused.weights_, looped.weights_)
    assert fused.score_fixed(X, y) == looped.score_fixed(X, y)


def test_score_fixed_does_not_mutate_fitted_state() -> None:
    X, y = _fixture()
    selector = FusedFastRelevanceKMeans(k_per_class=3, seed=3).fit(X, y)
    state = selector.diagnostics_["state_sha256"]
    centers = selector.centers_.copy()
    weights = selector.weights_.copy()
    selector.score_fixed(X[::-1], y[::-1])
    assert selector.diagnostics_["state_sha256"] == state
    np.testing.assert_array_equal(selector.centers_, centers)
    np.testing.assert_array_equal(selector.weights_, weights)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"prototype_iterations": 0}, "positive int"),
        ({"prototype_backend": "global"}, "fused.*looped"),
        ({"margin_rows_per_class": 0}, "positive int"),
    ],
)
def test_invalid_configuration_fails_loudly(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        FusedFastRelevanceKMeans(k_per_class=3, seed=1, **kwargs)
