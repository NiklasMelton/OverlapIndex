import numpy as np

from experiments.diagonal_metric_toy import experiment as base
from experiments.prototype_diagonal_metric_toy.experiment import (
    PrototypeDiagonalOverlapIndex,
    _internal_split,
    fit_prototype_margin_metric,
)


def _symmetric_axes(seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), 80)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=labels.size)
    signal = signs[:, None] * np.eye(3)[labels] * 3.0
    signal += rng.normal(0.0, 0.15, size=signal.shape)
    nuisance = rng.normal(0.0, 2.0, size=(labels.size, 6))
    return np.concatenate([signal, nuisance], axis=1), labels


def test_internal_split_is_disjoint_complete_and_deterministic() -> None:
    _X, y = _symmetric_axes()
    first = _internal_split(y, 11)
    second = _internal_split(y, 11)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert np.intersect1d(*first).size == 0
    assert np.array_equal(np.sort(np.concatenate(first)), np.arange(y.size))


def test_prototype_margin_detects_zero_mean_nonlinear_axes() -> None:
    X, y = _symmetric_axes()
    class_metric = base.fit_diagonal_metric(X, y)
    prototype_metric = fit_prototype_margin_metric(X, y, seed=13)
    class_signal_ratio = float(np.mean(class_metric.weights[:3]) / np.mean(class_metric.weights[3:]))
    prototype_signal_ratio = float(
        np.mean(prototype_metric.weights[:3]) / np.mean(prototype_metric.weights[3:])
    )
    assert prototype_signal_ratio > class_signal_ratio
    assert prototype_signal_ratio > 2.0


def test_metric_and_wrapper_are_deterministic_and_score_fixed_does_not_refit() -> None:
    X, y = base.generate_embedding(
        scenario="nonlinear_nuisance",
        seed=43,
        split="unit_prototype",
        n_per_class=16,
        candidate=base.PANEL_CANDIDATES[2],
    )
    first = fit_prototype_margin_metric(X, y, seed=19)
    second = fit_prototype_margin_metric(X, y, seed=19)
    assert first.state_sha256 == second.state_sha256
    selector = PrototypeDiagonalOverlapIndex(
        prototype_refinement=False,
        overlap_index_kwargs={
            "model_type": "MiniBatchKMeans",
            "kmeans_k": 2,
            "kmeans_kwargs": {"random_state": 7, "n_init": 1},
        },
        metric_seed=19,
    ).fit(X, y)
    before = selector.metric_diagnostics_
    assert np.isfinite(selector.score_fixed(X, y))
    assert selector.metric_diagnostics_ == before
    assert before["internal_split_disjoint"] is True
