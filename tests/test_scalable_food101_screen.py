import numpy as np

from experiments.scalable_relevance_kmeans import food101_analysis as analysis
from experiments.scalable_relevance_kmeans import food101_screen as screen


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260829)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 30)
    values = rng.normal(size=(labels.size, 16)).astype(np.float32)
    for index, label in enumerate(("a", "b", "c")):
        values[labels == label, index] += np.float32(2.0)
    return values, labels


def test_scalable_crossfit_is_deterministic_and_complete() -> None:
    values, labels = _fixture()
    first = screen.scalable_cross_fitted_score(values, labels, seed=51)
    second = screen.scalable_cross_fitted_score(values, labels, seed=51)
    assert first["candidate_id"] == "S"
    assert len(first["folds"]) == screen.FOLDS
    assert np.isfinite(first["score"])
    assert screen._determinism(first, second)["exact"] is True
    assert all(len(row["candidate_config_sha256"]) == 64 for row in first["folds"])


def test_full_cyclic_schedule_is_exactly_balanced() -> None:
    counts = {
        method: [0] * len(screen.METHODS)
        for method in screen.METHODS
    }
    for panel_index in range(600):
        order = screen._method_order(panel_index)
        assert set(order) == set(screen.METHODS)
        for position, method in enumerate(order):
            counts[method][position] += 1
    assert all(values == [150, 150, 150, 150] for values in counts.values())


def test_selection_metrics_average_exact_ties() -> None:
    scores = np.asarray([1.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    outcomes = np.asarray([0.9, 0.7, 1.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    result = analysis._selection_metrics(scores, outcomes)
    assert result["selected_backbones"] == list(screen.MODELS[:2])
    assert np.isclose(result["regret_pp"], 20.0)
    assert result["exact_best"] is False


def test_paired_bootstrap_is_deterministic_and_uses_five_blocks() -> None:
    first = analysis._bootstrap([1, 2, 3, 4, 5], key="fixed")
    second = analysis._bootstrap([1, 2, 3, 4, 5], key="fixed")
    assert first == second
    assert first["estimate"] == 3.0
    assert len(first["replicate_values"]) == 5


def test_runtime_contrast_preserves_replicate_blocks() -> None:
    rows = []
    for model in screen.MODELS:
        for replicate in screen.REPLICATES:
            for budget in screen.BUDGETS:
                for candidate, wall in (("S", 0.5), ("LP-FULL", 1.0), ("B", 0.4)):
                    rows.append(
                        {
                            "model": model,
                            "replicate": replicate,
                            "arm": "baseline",
                            "budget": budget,
                            "candidate_id": candidate,
                            "total_wall_seconds": wall,
                        }
                    )
    result = analysis._runtime_contrast(
        {"selector_rows": rows}, arm="baseline", comparator="LP-FULL"
    )
    assert result["estimate"] == 0.5
    assert result["replicate_values"] == [0.5] * 5


def test_complete_analysis_builds_all_primary_gates() -> None:
    selector_rows = []
    reference_rows = []
    for replicate in screen.REPLICATES:
        for arm in screen.ARMS:
            for model_index, model in enumerate(screen.MODELS):
                for head in analysis.HEADS:
                    reference_rows.append(
                        {
                            "backbone": model,
                            "replicate": replicate,
                            "arm": arm,
                            "head": head,
                            "test_accuracy": 0.5 + 0.01 * model_index,
                        }
                    )
                for budget in screen.BUDGETS:
                    for method, wall in (("A", 0.3), ("B", 0.4), ("S", 0.5), ("LP-FULL", 1.0)):
                        selector_rows.append(
                            {
                                "model": model,
                                "replicate": replicate,
                                "arm": arm,
                                "budget": budget,
                                "candidate_id": method,
                                "score": float(model_index),
                                "total_wall_seconds": wall,
                            }
                        )
    result = analysis.analyze(
        {
            "selector_rows": selector_rows,
            "reference_rows": reference_rows,
            "parity_rows": [{"exact": True}] * 1800,
            "determinism": {method: {"exact": True} for method in screen.METHODS},
        }
    )
    assert result["decision"]["status"] == "pass"
    assert len(result["selector_metrics"]) == 960
    assert len(result["rank_auc_rows"]) == 240
    assert len([row for row in result["gates"] if row.get("required")]) == 13
