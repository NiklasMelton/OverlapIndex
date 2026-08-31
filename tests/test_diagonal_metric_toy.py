import numpy as np

from experiments.diagonal_metric_toy.experiment import (
    DiagonalMetricOverlapIndex,
    PANEL_CANDIDATES,
    fit_diagonal_metric,
    generate_embedding,
    summarize,
)


def test_diagonal_metric_upweights_separating_feature() -> None:
    rng = np.random.default_rng(7)
    y = np.repeat([0, 1], 100)
    X = np.column_stack(
        [
            np.where(y == 0, -2.0, 2.0) + rng.normal(0, 0.2, y.size),
            rng.normal(0, 8.0, y.size),
        ]
    )
    metric = fit_diagonal_metric(X, y)
    assert metric.weights[0] > 10.0 * metric.weights[1]
    assert np.isclose(np.mean(metric.weights), 1.0)


def test_score_fixed_does_not_refit_metric() -> None:
    X, y = generate_embedding(
        scenario="linear_nuisance",
        seed=17,
        split="unit",
        n_per_class=12,
        candidate=PANEL_CANDIDATES[2],
    )
    selector = DiagonalMetricOverlapIndex(
        prototype_refinement=False,
        overlap_index_kwargs={
            "model_type": "MiniBatchKMeans",
            "kmeans_k": 2,
            "kmeans_kwargs": {"random_state": 3, "n_init": 1},
        },
    ).fit(X, y)
    before = selector.metric_diagnostics_
    score = selector.score_fixed(X, y)
    assert np.isfinite(score)
    assert selector.metric_diagnostics_ == before


def test_generator_is_deterministic_and_candidate_paired() -> None:
    kwargs = dict(
        scenario="nonlinear_nuisance",
        seed=43,
        split="selector",
        n_per_class=8,
        candidate=PANEL_CANDIDATES[1],
    )
    first_X, first_y = generate_embedding(**kwargs)
    second_X, second_y = generate_embedding(**kwargs)
    assert np.array_equal(first_X, second_X)
    assert np.array_equal(first_y, second_y)
    other_X, other_y = generate_embedding(
        **{**kwargs, "candidate": PANEL_CANDIDATES[3]}
    )
    assert np.array_equal(first_y, other_y)
    assert not np.array_equal(first_X, other_X)


def test_summary_requires_and_aggregates_complete_panels() -> None:
    rows = []
    reference = {row["candidate_id"]: 0.5 + 0.1 * index for index, row in enumerate(PANEL_CANDIDATES)}
    methods = (
        "raw_unrefined",
        "raw_refined",
        "diagonal_unrefined",
        "diagonal_refined",
        "linear_probe",
    )
    for scenario in (
        "linear_clean",
        "linear_nuisance",
        "nonlinear_clean",
        "nonlinear_nuisance",
    ):
        for budget in (64, 256):
            for seed in (17, 43):
                for method_index, method in enumerate(methods):
                    for index, candidate in enumerate(PANEL_CANDIDATES):
                        rows.append(
                            {
                                "scenario": scenario,
                                "budget_per_class": budget,
                                "seed": seed,
                                "method": method,
                                "candidate_id": candidate["candidate_id"],
                                "selector_score": float(index),
                                "reference_accuracy": reference[candidate["candidate_id"]],
                                "wall_seconds": float(method_index + 1),
                            }
                        )
    result = summarize(rows)
    assert len(result["panel_metrics"]) == 4 * 2 * 2 * 5
    assert len(result["aggregate"]) == 4 * 2 * 5
    assert all(row["mean_regret_pp"] == 0.0 for row in result["aggregate"])
