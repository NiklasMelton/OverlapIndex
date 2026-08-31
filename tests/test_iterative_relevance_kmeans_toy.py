import numpy as np

from overlapindex import OverlapIndex

from experiments.diagonal_metric_toy import experiment as base
from experiments.iterative_relevance_kmeans_toy.experiment import (
    OneUpdateRelevanceKMeans,
    oi_score_from_centers,
)


def _symmetric_axes(seed: int = 9) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), 80)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=y.size)
    signal = signs[:, None] * np.eye(3)[y] * 3.0
    signal += rng.normal(0.0, 0.15, size=signal.shape)
    nuisance = rng.normal(0.0, 2.0, size=(y.size, 6))
    return np.concatenate([signal, nuisance], axis=1), y


def test_fixed_center_scorer_matches_upstream_oi() -> None:
    X, y = base.generate_embedding(
        scenario="linear_clean",
        seed=17,
        split="scorer_parity",
        n_per_class=20,
        candidate=base.PANEL_CANDIDATES[2],
    )
    train = np.concatenate([np.flatnonzero(y == label)[:12] for label in range(base.CLASS_COUNT)])
    heldout = np.concatenate([np.flatnonzero(y == label)[12:] for label in range(base.CLASS_COUNT)])
    selector = OverlapIndex(
        prototype_refinement=False,
        **base._oi_kwargs(31),
    ).fit(X[train], y[train])
    upstream = selector.score_fixed(X[heldout], y[heldout])
    centers = np.asarray(selector._model.centers)
    owners = np.asarray([selector._model.cluster_to_class[index] for index in range(centers.shape[0])])
    local = oi_score_from_centers(
        X[heldout],
        y[heldout],
        centers=centers,
        owners=owners,
        weights=np.ones(X.shape[1]),
    )
    assert local == upstream


def test_one_update_detects_symmetric_nonlinear_signal() -> None:
    X, y = _symmetric_axes()
    selector = OneUpdateRelevanceKMeans(seed=13).fit(X, y)
    assert selector.weights_ is not None
    signal_ratio = float(np.mean(selector.weights_[:3]) / np.mean(selector.weights_[3:]))
    assert signal_ratio > 2.0
    assert selector.diagnostics_["kmeans_fit_count"] == 1
    assert selector.diagnostics_["weighted_lloyd_update_count"] == 1


def test_one_update_is_deterministic_and_score_fixed_preserves_state() -> None:
    X, y = base.generate_embedding(
        scenario="nonlinear_nuisance",
        seed=43,
        split="one_update_unit",
        n_per_class=18,
        candidate=base.PANEL_CANDIDATES[2],
    )
    first = OneUpdateRelevanceKMeans(seed=23).fit(X, y)
    second = OneUpdateRelevanceKMeans(seed=23).fit(X, y)
    assert first.diagnostics_["state_sha256"] == second.diagnostics_["state_sha256"]
    before = dict(first.diagnostics_)
    assert np.isfinite(first.score_fixed(X, y))
    assert first.diagnostics_ == before
