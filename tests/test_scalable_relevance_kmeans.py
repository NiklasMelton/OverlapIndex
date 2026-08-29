import numpy as np

from overlapindex import OverlapIndex

from experiments.diagonal_metric_toy import experiment as toy
from experiments.scalable_relevance_kmeans.scalable import (
    ScalableOneUpdateRelevanceKMeans,
    tiled_oi_score,
)


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    return toy.generate_embedding(
        scenario="nonlinear_nuisance",
        seed=43,
        split="scalable_fixture",
        n_per_class=24,
        candidate=toy.PANEL_CANDIDATES[2],
    )


def test_tiled_fixed_center_score_matches_upstream_oi() -> None:
    X, y = toy.generate_embedding(
        scenario="linear_clean",
        seed=17,
        split="scalable_parity",
        n_per_class=20,
        candidate=toy.PANEL_CANDIDATES[2],
    )
    train = np.concatenate([np.flatnonzero(y == label)[:12] for label in range(toy.CLASS_COUNT)])
    heldout = np.concatenate([np.flatnonzero(y == label)[12:] for label in range(toy.CLASS_COUNT)])
    selector = OverlapIndex(
        prototype_refinement=False,
        **toy._oi_kwargs(31),
    ).fit(X[train], y[train])
    upstream = selector.score_fixed(X[heldout], y[heldout])
    centers = np.asarray(selector._model.centers)
    owners = np.asarray([selector._model.cluster_to_class[index] for index in range(centers.shape[0])])
    local, diagnostics = tiled_oi_score(
        X[heldout],
        y[heldout],
        centers=centers,
        owners=owners,
        weights=np.ones(X.shape[1]),
        row_cap=1,
    )
    assert local == upstream
    assert diagnostics["max_score_tile_rows"] == 1


def test_tiled_scoring_is_row_cap_invariant() -> None:
    X, y = _fixture()
    fitted = ScalableOneUpdateRelevanceKMeans(
        k_per_class=3,
        kmeans_kwargs={"random_state": 5},
        row_cap=7,
    ).fit(X, y)
    first, _ = tiled_oi_score(
        X,
        y,
        centers=fitted.centers_,
        owners=fitted.owners_,
        weights=fitted.weights_,
        row_cap=1,
    )
    second, _ = tiled_oi_score(
        X,
        y,
        centers=fitted.centers_,
        owners=fitted.owners_,
        weights=fitted.weights_,
        row_cap=19,
    )
    assert first == second


def test_scalable_fit_is_tiling_invariant() -> None:
    X, y = _fixture()
    untiled = ScalableOneUpdateRelevanceKMeans(
        k_per_class=toy.K_PER_CLASS,
        kmeans_kwargs={"random_state": 11},
        row_cap=X.shape[0],
    ).fit(X, y)
    tiled = ScalableOneUpdateRelevanceKMeans(
        k_per_class=toy.K_PER_CLASS,
        kmeans_kwargs={"random_state": 11},
        row_cap=9,
    ).fit(X, y)
    assert np.allclose(tiled.weights_, untiled.weights_, rtol=0.0, atol=1.0e-12)
    assert np.array_equal(tiled.owners_, untiled.owners_)
    assert np.allclose(tiled.centers_, untiled.centers_, rtol=0.0, atol=1.0e-7)
    assert tiled.score_fixed(X, y) == untiled.score_fixed(X, y)


def test_fit_is_deterministic_bounded_and_score_fixed_preserves_state() -> None:
    X, y = _fixture()
    kwargs = {
        "k_per_class": 3,
        "kmeans_kwargs": {"random_state": 23},
        "memory_budget_mb": 1,
        "row_cap": 5,
    }
    first = ScalableOneUpdateRelevanceKMeans(**kwargs).fit(X, y)
    second = ScalableOneUpdateRelevanceKMeans(**kwargs).fit(X, y)
    assert first.diagnostics_["state_sha256"] == second.diagnostics_["state_sha256"]
    assert first.diagnostics_["max_score_tile_bytes"] <= 1 * 1024 * 1024
    before = dict(first.diagnostics_)
    assert np.isfinite(first.score_fixed(X, y))
    assert first.diagnostics_ == before
