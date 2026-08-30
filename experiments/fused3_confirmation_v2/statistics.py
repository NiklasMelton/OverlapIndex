"""Frozen statistics for the untouched FUSED3 confirmation panel.

This module has no candidate-selection surface.  It compares the single locked
``FUSED3`` candidate with ``LP-FULL`` on the exact five-dataset confirmation
grid and implements the predeclared hierarchical bootstrap and gates.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata


DATASET_IDS = (
    "torchvision_cifar10",
    "torchvision_stl10",
    "torchvision_gtsrb",
    "torchvision_fgvc_aircraft",
    "torchvision_dtd",
)
BACKBONES = (
    "dinov2-small",
    "deit-tiny",
    "convnext-tiny",
    "mobilenetv3-large",
    "openclip-vit-b-32",
    "resnet50",
    "efficientnet-b0",
    "swin-tiny",
    "vit-small-16",
    "densenet121",
)
REPLICATE_SEEDS = (
    2026082710,
    2026082711,
    2026082712,
    2026082713,
    2026082714,
)
BUDGETS = (32, 64)
HEADS = ("linear", "quadratic", "knn", "rbf")
METHODS = ("FUSED3", "LP-FULL")
LOCKED_CANDIDATE = "FUSED3"
COMPARATOR = "LP-FULL"
STUDY = "fused3_backbone_ranking_confirmation_v2"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026082702
SELECTION_TIE_ATOL = 1.0e-12


def _materialize(
    rows: Iterable[Mapping[str, Any]], *, name: str
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"{name}[{position}] must be a mapping")
        materialized.append(dict(row))
    return materialized


def _strict_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a strict int")
    return int(value)


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _closed(value: Any, allowed: Sequence[Any], *, field: str) -> Any:
    if value not in allowed:
        raise ValueError(f"{field} must be one of {tuple(allowed)!r}; got {value!r}")
    return value


def _replicate_seed(row: Mapping[str, Any]) -> int:
    return _closed(
        _strict_int(row.get("replicate_seed"), field="replicate_seed"),
        REPLICATE_SEEDS,
        field="replicate_seed",
    )


def _validate_replicate_pair(row: Mapping[str, Any]) -> tuple[int, int]:
    replicate = _closed(
        _strict_int(row.get("replicate"), field="replicate"),
        tuple(range(len(REPLICATE_SEEDS))),
        field="replicate",
    )
    seed = _replicate_seed(row)
    if seed != REPLICATE_SEEDS[replicate]:
        raise ValueError("replicate and replicate_seed do not match the frozen pairing")
    return replicate, seed


def validate_confirmation_inputs(
    selector_rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize once and require the exact closed confirmation grid."""

    selectors = _materialize(selector_rows, name="selector_rows")
    references = _materialize(reference_rows, name="reference_rows")
    expected_selectors = {
        (dataset, backbone, replicate, seed, budget, method)
        for dataset in DATASET_IDS
        for backbone in BACKBONES
        for replicate, seed in enumerate(REPLICATE_SEEDS)
        for budget in BUDGETS
        for method in METHODS
    }
    expected_references = {
        (dataset, backbone, replicate, seed, budget, head)
        for dataset in DATASET_IDS
        for backbone in BACKBONES
        for replicate, seed in enumerate(REPLICATE_SEEDS)
        for budget in BUDGETS
        for head in HEADS
    }

    observed_selectors: set[tuple[str, str, int, int, int, str]] = set()
    for row in selectors:
        if row.get("status") != "ok":
            raise ValueError("every selector row must have status='ok'")
        replicate, replicate_seed = _validate_replicate_pair(row)
        identity = (
            str(_closed(row.get("dataset_id"), DATASET_IDS, field="dataset_id")),
            str(_closed(row.get("backbone"), BACKBONES, field="backbone")),
            replicate,
            replicate_seed,
            _closed(
                _strict_int(row.get("budget"), field="budget"),
                BUDGETS,
                field="budget",
            ),
            str(_closed(row.get("candidate_id"), METHODS, field="candidate_id")),
        )
        if identity in observed_selectors:
            raise ValueError(f"duplicate selector identity {identity!r}")
        observed_selectors.add(identity)
        _finite_float(row.get("score"), field="selector score")
    if observed_selectors != expected_selectors:
        missing = len(expected_selectors - observed_selectors)
        extra = len(observed_selectors - expected_selectors)
        raise ValueError(
            "selector rows do not match the exact confirmation grid "
            f"(missing={missing}, extra={extra})"
        )

    observed_references: set[tuple[str, str, int, int, int, str]] = set()
    for row in references:
        if row.get("status") != "ok":
            raise ValueError("every reference row must have status='ok'")
        replicate, replicate_seed = _validate_replicate_pair(row)
        identity = (
            str(_closed(row.get("dataset_id"), DATASET_IDS, field="dataset_id")),
            str(_closed(row.get("backbone"), BACKBONES, field="backbone")),
            replicate,
            replicate_seed,
            _closed(
                _strict_int(row.get("budget"), field="budget"),
                BUDGETS,
                field="budget",
            ),
            str(_closed(row.get("head"), HEADS, field="head")),
        )
        if identity in observed_references:
            raise ValueError(f"duplicate reference identity {identity!r}")
        observed_references.add(identity)
        accuracy = _finite_float(row.get("test_accuracy"), field="test_accuracy")
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("test_accuracy must be in [0, 1]")
    if observed_references != expected_references:
        missing = len(expected_references - observed_references)
        extra = len(observed_references - expected_references)
        raise ValueError(
            "reference rows do not match the exact confirmation grid "
            f"(missing={missing}, extra={extra})"
        )
    return selectors, references


def _spearman(
    scores: Sequence[float], outcomes: Sequence[float]
) -> tuple[float | None, str]:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("Spearman inputs must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Spearman inputs must be finite")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None, "undefined_constant"
    value = float(np.corrcoef(rankdata(x, method="average"), rankdata(y, method="average"))[0, 1])
    if not np.isfinite(value):
        return None, "undefined_nonfinite"
    return value, "defined"


def panel_metrics(
    selector_rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute tie-safe selection metrics on complete ten-backbone panels."""

    selectors, references = validate_confirmation_inputs(selector_rows, reference_rows)
    selector_lookup = {
        (
            str(row["dataset_id"]),
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        ): float(row["score"])
        for row in selectors
    }
    reference_lookup = {
        (
            str(row["dataset_id"]),
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["head"]),
        ): float(row["test_accuracy"])
        for row in references
    }
    output: list[dict[str, Any]] = []
    for dataset in DATASET_IDS:
        for seed in REPLICATE_SEEDS:
            for budget in BUDGETS:
                for head in HEADS:
                    outcomes = np.asarray(
                        [
                            reference_lookup[(dataset, backbone, seed, budget, head)]
                            for backbone in BACKBONES
                        ],
                        dtype=np.float64,
                    )
                    best = float(np.max(outcomes))
                    for method in METHODS:
                        scores = np.asarray(
                            [
                                selector_lookup[(dataset, backbone, seed, budget, method)]
                                for backbone in BACKBONES
                            ],
                            dtype=np.float64,
                        )
                        maximum = float(np.max(scores))
                        selected = np.isclose(
                            scores, maximum, atol=SELECTION_TIE_ATOL, rtol=0.0
                        )
                        selected_outcomes = outcomes[selected]
                        rho, rho_status = _spearman(scores, outcomes)
                        output.append(
                            {
                                "dataset_id": dataset,
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                "head": head,
                                "candidate_id": method,
                                "backbone_count": len(BACKBONES),
                                "selected_count": int(np.count_nonzero(selected)),
                                "selected_backbones": [
                                    backbone
                                    for backbone, chosen in zip(BACKBONES, selected)
                                    if bool(chosen)
                                ],
                                "best_accuracy": best,
                                "selected_accuracy": float(np.mean(selected_outcomes)),
                                "regret_pp": float(
                                    100.0 * (best - float(np.mean(selected_outcomes)))
                                ),
                                "exact_best": float(np.mean(selected_outcomes == best)),
                                "within_one_pp": float(
                                    np.mean(selected_outcomes >= best - 0.01)
                                ),
                                "spearman": rho,
                                "spearman_status": rho_status,
                                "tie_rule": "maxima_within_absolute_tolerance_1e-12_averaged",
                            }
                        )
    return output


def rank_auc_rows(metrics: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Integrate complete two-budget Spearman curves in log2-budget space."""

    rows = _materialize(metrics, name="metrics")
    grouped: dict[tuple[str, int, str, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(_closed(row.get("dataset_id"), DATASET_IDS, field="dataset_id")),
            _replicate_seed(row),
            str(_closed(row.get("head"), HEADS, field="head")),
            str(_closed(row.get("candidate_id"), METHODS, field="candidate_id")),
        )
        budget = _closed(
            _strict_int(row.get("budget"), field="budget"), BUDGETS, field="budget"
        )
        if budget in grouped[key]:
            raise ValueError(f"duplicate rank-AUC budget for {key!r}: {budget}")
        grouped[key][budget] = row
    expected_keys = {
        (dataset, seed, head, method)
        for dataset in DATASET_IDS
        for seed in REPLICATE_SEEDS
        for head in HEADS
        for method in METHODS
    }
    if set(grouped) != expected_keys:
        raise ValueError("rank AUC requires the exact complete curve identity set")
    x = np.log2(np.asarray(BUDGETS, dtype=np.float64))
    width = float(x[-1] - x[0])
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        observed = grouped[key]
        if set(observed) != set(BUDGETS):
            raise ValueError(f"rank AUC requires both frozen budgets for {key!r}")
        values = [observed[budget].get("spearman") for budget in BUDGETS]
        statuses = [str(observed[budget].get("spearman_status")) for budget in BUDGETS]
        if any(value is None or status != "defined" for value, status in zip(values, statuses)):
            auc: float | None = None
            status = "undefined_supporting"
        else:
            y = np.asarray([float(value) for value in values], dtype=np.float64)
            auc = float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)) / width)
            status = "defined"
        output.append(
            {
                "dataset_id": key[0],
                "replicate_seed": int(key[1]),
                "head": key[2],
                "candidate_id": key[3],
                "rank_auc": auc,
                "status": status,
                "budgets": list(BUDGETS),
                "normalization": "trapezoid_over_log2_budget_width",
                "supporting_only": True,
                "promotion_veto": False,
            }
        )
    return output


def regret_contrast_rows(
    metrics: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _materialize(metrics, name="metrics")
    lookup: dict[tuple[str, int, int, str, str], float] = {}
    for row in rows:
        identity = (
            str(row.get("dataset_id")),
            _strict_int(row.get("replicate_seed"), field="replicate_seed"),
            _strict_int(row.get("budget"), field="budget"),
            str(row.get("head")),
            str(row.get("candidate_id")),
        )
        if identity in lookup:
            raise ValueError(f"duplicate panel metric identity {identity!r}")
        lookup[identity] = _finite_float(row.get("regret_pp"), field="regret_pp")
    expected = {
        (dataset, seed, budget, head, method)
        for dataset in DATASET_IDS
        for seed in REPLICATE_SEEDS
        for budget in BUDGETS
        for head in HEADS
        for method in METHODS
    }
    if set(lookup) != expected:
        raise ValueError("regret contrast requires the exact complete metric grid")
    return [
        {
            "dataset_id": dataset,
            "replicate_seed": int(seed),
            "budget": int(budget),
            "head": head,
            "candidate_id": LOCKED_CANDIDATE,
            "comparator_id": COMPARATOR,
            "contrast_pp": float(
                lookup[(dataset, seed, budget, head, LOCKED_CANDIDATE)]
                - lookup[(dataset, seed, budget, head, COMPARATOR)]
            ),
        }
        for dataset in DATASET_IDS
        for seed in REPLICATE_SEEDS
        for budget in BUDGETS
        for head in HEADS
    ]


def hierarchical_interval(
    contrast_rows: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Resample datasets, then complete replicate blocks, equally weighted."""

    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError("n_resamples must be a positive int")
    if type(seed) is not int:
        raise TypeError("seed must be a strict int")
    rows = _materialize(contrast_rows, name="contrast_rows")
    grouped: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        dataset = str(_closed(row.get("dataset_id"), DATASET_IDS, field="dataset_id"))
        replicate = _replicate_seed(row)
        budget = _closed(
            _strict_int(row.get("budget"), field="budget"), BUDGETS, field="budget"
        )
        if budget in grouped[dataset][replicate]:
            raise ValueError(
                f"duplicate hierarchical identity {(dataset, replicate, budget)!r}"
            )
        grouped[dataset][replicate][budget] = _finite_float(
            row.get("contrast_pp"), field="contrast_pp"
        )
    for dataset in DATASET_IDS:
        if set(grouped.get(dataset, {})) != set(REPLICATE_SEEDS):
            raise ValueError(f"dataset {dataset!r} has incomplete replicate coverage")
        for replicate in REPLICATE_SEEDS:
            if set(grouped[dataset][replicate]) != set(BUDGETS):
                raise ValueError(
                    f"dataset {dataset!r} replicate {replicate} has incomplete budget coverage"
                )
    if len(rows) != len(DATASET_IDS) * len(REPLICATE_SEEDS) * len(BUDGETS):
        raise ValueError("hierarchical interval received extra rows")

    per_dataset = {
        dataset: float(
            np.mean(
                [
                    np.mean(
                        [grouped[dataset][replicate][budget] for budget in BUDGETS]
                    )
                    for replicate in REPLICATE_SEEDS
                ]
            )
        )
        for dataset in DATASET_IDS
    }
    generator = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=np.float64)
    dataset_array = np.asarray(DATASET_IDS, dtype=object)
    replicate_array = np.asarray(REPLICATE_SEEDS, dtype=np.int64)
    for draw in range(n_resamples):
        sampled_datasets = generator.choice(
            dataset_array, size=len(DATASET_IDS), replace=True
        )
        sampled_dataset_means: list[float] = []
        for sampled_dataset in sampled_datasets:
            dataset = str(sampled_dataset)
            sampled_replicates = generator.choice(
                replicate_array, size=len(REPLICATE_SEEDS), replace=True
            )
            sampled_dataset_means.append(
                float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    grouped[dataset][int(replicate)][budget]
                                    for budget in BUDGETS
                                ]
                            )
                            for replicate in sampled_replicates
                        ]
                    )
                )
            )
        boot[draw] = float(np.mean(sampled_dataset_means))
    return {
        "estimate": float(np.mean([per_dataset[value] for value in DATASET_IDS])),
        "lower_95": float(np.quantile(boot, 0.025)),
        "upper_95": float(np.quantile(boot, 0.975)),
        "per_dataset": per_dataset,
        "n_datasets": len(DATASET_IDS),
        "n_replicates": len(REPLICATE_SEEDS),
        "budgets": list(BUDGETS),
        "n_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": "dataset_then_complete_replicate_block",
        "aggregation": "equal_dataset_equal_replicate_equal_budget",
    }


def confirmation_gates(
    contrasts: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Apply the four frozen gates to the one locked candidate."""

    rows = _materialize(contrasts, name="contrasts")
    expected = {
        (dataset, replicate, budget, head)
        for dataset in DATASET_IDS
        for replicate in REPLICATE_SEEDS
        for budget in BUDGETS
        for head in HEADS
    }
    observed: set[tuple[str, int, int, str]] = set()
    for row in rows:
        if row.get("candidate_id") != LOCKED_CANDIDATE or row.get("comparator_id") != COMPARATOR:
            raise ValueError("confirmation contrasts must be exactly FUSED3 minus LP-FULL")
        identity = (
            str(row.get("dataset_id")),
            _strict_int(row.get("replicate_seed"), field="replicate_seed"),
            _strict_int(row.get("budget"), field="budget"),
            str(row.get("head")),
        )
        if identity in observed:
            raise ValueError(f"duplicate confirmation contrast {identity!r}")
        observed.add(identity)
        _finite_float(row.get("contrast_pp"), field="contrast_pp")
    if observed != expected:
        raise ValueError("confirmation gates require the exact complete contrast grid")

    gates: dict[str, Any] = {}
    for head in HEADS:
        interval = hierarchical_interval(
            (row for row in rows if row["head"] == head),
            n_resamples=n_resamples,
            seed=seed,
        )
        if head == "linear":
            passed = (
                float(interval["upper_95"]) <= 1.0
                and max(float(value) for value in interval["per_dataset"].values()) <= 2.0
            )
            rule = "pooled upper_95 <= 1 pp and maximum dataset point <= 2 pp"
        else:
            favorable = sum(
                float(value) < 0.0 for value in interval["per_dataset"].values()
            )
            interval["favorable_dataset_count"] = int(favorable)
            passed = float(interval["upper_95"]) < 0.0 and favorable >= 4
            rule = "pooled upper_95 < 0 and at least 4 of 5 datasets favor FUSED3"
        gates[head] = {
            **interval,
            "status": "pass" if passed else "fail",
            "rule": rule,
            "required": True,
        }
    status = "pass_confirmed" if all(gate["status"] == "pass" for gate in gates.values()) else "fail_locked_candidate"
    return {
        "candidate_id": LOCKED_CANDIDATE,
        "comparator_id": COMPARATOR,
        "status": status,
        "gates": gates,
        "runner_up_allowed": False,
        "reselection_performed": False,
    }


def summarize(
    selector_rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = panel_metrics(selector_rows, reference_rows)
    rank_auc = rank_auc_rows(metrics)
    contrasts = regret_contrast_rows(metrics)
    gates = confirmation_gates(contrasts)
    aggregate: list[dict[str, Any]] = []
    for dataset in DATASET_IDS:
        for head in HEADS:
            for method in METHODS:
                selected = [
                    row
                    for row in metrics
                    if row["dataset_id"] == dataset
                    and row["head"] == head
                    and row["candidate_id"] == method
                ]
                aggregate.append(
                    {
                        "dataset_id": dataset,
                        "head": head,
                        "candidate_id": method,
                        "mean_regret_pp": float(np.mean([row["regret_pp"] for row in selected])),
                        "exact_best_rate": float(np.mean([row["exact_best"] for row in selected])),
                        "within_one_pp_rate": float(np.mean([row["within_one_pp"] for row in selected])),
                        "mean_spearman": (
                            float(np.mean([row["spearman"] for row in selected if row["spearman"] is not None]))
                            if any(row["spearman"] is not None for row in selected)
                            else None
                        ),
                    }
                )
    undefined_rank = sum(row["rank_auc"] is None for row in rank_auc)
    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": "confirmation",
        "candidate_id": LOCKED_CANDIDATE,
        "comparator_id": COMPARATOR,
        "panel_metrics": metrics,
        "rank_auc": rank_auc,
        "rank_support": {
            "status": "complete_defined" if undefined_rank == 0 else "incomplete_undefined",
            "expected_curve_count": len(DATASET_IDS) * len(REPLICATE_SEEDS) * len(HEADS) * len(METHODS),
            "observed_curve_count": len(rank_auc),
            "defined_curve_count": len(rank_auc) - undefined_rank,
            "undefined_curve_count": undefined_rank,
            "supporting_only": True,
            "promotion_veto": False,
        },
        "regret_contrasts": contrasts,
        "aggregate": aggregate,
        "confirmation": gates,
        "runtime": {
            "gate_source": "hash_bound_passed_fused3_food_resource_evidence",
            "new_dataset_timings": "descriptive_only",
            "promotion_veto": False,
        },
        "decision": {
            "status": gates["status"],
            "selected_candidate": LOCKED_CANDIDATE,
            "comparator_id": COMPARATOR,
            "runner_up_allowed": False,
            "reselection_performed": False,
            "claim_scope": "frozen_class_stratified_same_distribution_confirmation_panel_only",
        },
    }


__all__ = [
    "BACKBONES",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BUDGETS",
    "COMPARATOR",
    "DATASET_IDS",
    "HEADS",
    "LOCKED_CANDIDATE",
    "METHODS",
    "REPLICATE_SEEDS",
    "SELECTION_TIE_ATOL",
    "STUDY",
    "confirmation_gates",
    "hierarchical_interval",
    "panel_metrics",
    "rank_auc_rows",
    "regret_contrast_rows",
    "summarize",
    "validate_confirmation_inputs",
]
