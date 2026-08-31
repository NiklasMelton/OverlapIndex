"""Paired toy comparison of raw OI, supervised diagonal OI, and a linear probe.

This module is deliberately experiment-local.  It does not add a public
OverlapIndex constructor option and it refuses to overwrite an existing result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from overlapindex import OverlapIndex


SCENARIOS = (
    "linear_clean",
    "linear_nuisance",
    "nonlinear_clean",
    "nonlinear_nuisance",
)
BUDGETS_PER_CLASS = (64, 256)
SEEDS = (17, 43)
CLASS_COUNT = 6
FEATURE_COUNT = 96
SIGNAL_FEATURE_COUNT = CLASS_COUNT
REFERENCE_TRAIN_PER_CLASS = 256
REFERENCE_TEST_PER_CLASS = 512
FOLDS = 3
K_PER_CLASS = 4
WEIGHT_FLOOR = 0.05
METHODS = (
    "raw_unrefined",
    "raw_refined",
    "diagonal_unrefined",
    "diagonal_refined",
    "linear_probe",
)
PANEL_CANDIDATES = (
    {"candidate_id": "bb0", "signal": 0.70, "nuisance": 0.70},
    {"candidate_id": "bb1", "signal": 1.00, "nuisance": 3.00},
    {"candidate_id": "bb2", "signal": 1.30, "nuisance": 1.20},
    {"candidate_id": "bb3", "signal": 1.60, "nuisance": 2.50},
    {"candidate_id": "bb4", "signal": 1.90, "nuisance": 1.70},
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_X_y(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(X)
    target = np.asarray(y)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("X must be numeric")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values")
    if target.ndim != 1 or target.shape[0] != values.shape[0]:
        raise ValueError("y must be a one-dimensional scalar label vector aligned with X")
    if len(dict.fromkeys(target.tolist())) < 2:
        raise ValueError("at least two classes are required")
    return values, target


@dataclass(frozen=True)
class FittedDiagonalMetric:
    """Immutable fitted affine diagonal metric."""

    mean: np.ndarray
    weights: np.ndarray
    scale: np.ndarray
    state_sha256: str
    diagnostics: Mapping[str, Any]

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("X has the wrong feature shape for this diagonal metric")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values")
        return np.asarray((values - self.mean) * self.scale, dtype=np.float64)


def fit_diagonal_metric(
    X: Any,
    y: Any,
    *,
    weight_floor: float = WEIGHT_FLOOR,
) -> FittedDiagonalMetric:
    """Fit a bounded supervised diagonal relevance metric on training rows.

    Relevance is ``between / (between + within + epsilon)``.  A nonzero raw
    floor preserves every input direction; mean normalization removes an
    irrelevant global distance multiplier.
    """

    if not isinstance(weight_floor, float) or not 0.0 < weight_floor <= 1.0:
        raise ValueError("weight_floor must be a float in (0, 1]")
    values, target = _validate_X_y(X, y)
    labels = list(dict.fromkeys(target.tolist()))
    mean = np.asarray(np.mean(values, axis=0), dtype=np.float64)
    within_sum = np.zeros(values.shape[1], dtype=np.float64)
    between_sum = np.zeros(values.shape[1], dtype=np.float64)
    for label in labels:
        rows = values[target == label]
        class_mean = np.asarray(np.mean(rows, axis=0), dtype=np.float64)
        within_sum += np.sum((rows - class_mean) ** 2, axis=0)
        between_sum += rows.shape[0] * (class_mean - mean) ** 2
    within = within_sum / float(values.shape[0])
    between = between_sum / float(values.shape[0])
    epsilon = max(float(np.mean(within)) * 1.0e-6, 1.0e-12)
    relevance = between / (between + within + epsilon)
    raw_weights = weight_floor + (1.0 - weight_floor) * relevance
    weights = np.asarray(raw_weights / float(np.mean(raw_weights)), dtype=np.float64)
    scale = np.asarray(np.sqrt(weights), dtype=np.float64)
    for array in (mean, weights, scale):
        array.setflags(write=False)
    state_payload = {
        "weight_floor": weight_floor,
        "mean": mean.tolist(),
        "weights": weights.tolist(),
    }
    state_sha256 = _sha256_bytes(_json_bytes(state_payload))
    diagnostics = {
        "weight_floor": weight_floor,
        "n_rows_fit": int(values.shape[0]),
        "n_features_fit": int(values.shape[1]),
        "n_classes_fit": len(labels),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_condition": float(np.max(weights) / np.min(weights)),
        "state_sha256": state_sha256,
    }
    return FittedDiagonalMetric(mean, weights, scale, state_sha256, diagnostics)


class DiagonalMetricOverlapIndex:
    """Experiment-local OI wrapper using one train-fold-fitted metric."""

    def __init__(
        self,
        *,
        prototype_refinement: bool,
        overlap_index_kwargs: Mapping[str, Any],
        weight_floor: float = WEIGHT_FLOOR,
    ) -> None:
        if type(prototype_refinement) is not bool:
            raise TypeError("prototype_refinement must be a strict bool")
        if not isinstance(overlap_index_kwargs, Mapping):
            raise TypeError("overlap_index_kwargs must be a mapping")
        if "prototype_refinement" in overlap_index_kwargs:
            raise ValueError("prototype_refinement must not be duplicated")
        self.prototype_refinement = prototype_refinement
        self.overlap_index_kwargs = deepcopy(dict(overlap_index_kwargs))
        self.weight_floor = weight_floor
        self.metric_: FittedDiagonalMetric | None = None
        self.estimator_: OverlapIndex | None = None

    def fit(self, X: Any, y: Any) -> "DiagonalMetricOverlapIndex":
        values, target = _validate_X_y(X, y)
        metric = fit_diagonal_metric(values, target, weight_floor=self.weight_floor)
        estimator = OverlapIndex(
            prototype_refinement=self.prototype_refinement,
            **deepcopy(self.overlap_index_kwargs),
        )
        estimator.fit(metric.transform(values), target)
        self.metric_ = metric
        self.estimator_ = estimator
        return self

    def _require_fit(self) -> tuple[FittedDiagonalMetric, OverlapIndex]:
        if self.metric_ is None or self.estimator_ is None:
            raise ValueError("DiagonalMetricOverlapIndex is not fitted")
        return self.metric_, self.estimator_

    @property
    def index(self) -> float:
        _metric, estimator = self._require_fit()
        return float(estimator.index)

    @property
    def metric_diagnostics_(self) -> dict[str, Any]:
        metric, _estimator = self._require_fit()
        return dict(metric.diagnostics)

    @property
    def prototype_refinement_(self) -> dict[str, Any]:
        _metric, estimator = self._require_fit()
        value = getattr(estimator, "prototype_refinement_", {})
        return dict(value) if isinstance(value, Mapping) else {}

    def transform(self, X: Any) -> np.ndarray:
        metric, _estimator = self._require_fit()
        return metric.transform(X)

    def score_fixed(self, X: Any, y: Any) -> float:
        metric, estimator = self._require_fit()
        return float(estimator.score_fixed(metric.transform(X), y))


def _row_l2(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1.0e-12)


def _bank_seed(seed: int, scenario: str, split: str) -> int:
    digest = hashlib.sha256(f"{seed}:{scenario}:{split}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def generate_embedding(
    *,
    scenario: str,
    seed: int,
    split: str,
    n_per_class: int,
    candidate: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one paired synthetic backbone embedding from a shared bank."""

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    rng = np.random.default_rng(_bank_seed(seed, scenario, split))
    labels = np.repeat(np.arange(CLASS_COUNT, dtype=np.int64), n_per_class)
    n_rows = labels.size
    signal_noise = rng.normal(0.0, 0.40, size=(n_rows, SIGNAL_FEATURE_COUNT))
    nuisance_noise = rng.normal(
        0.0, 1.0, size=(n_rows, FEATURE_COUNT - SIGNAL_FEATURE_COUNT)
    )
    signal_strength = float(candidate["signal"])
    nuisance_strength = (
        0.35 if scenario.endswith("clean") else float(candidate["nuisance"])
    )
    class_axes = np.eye(SIGNAL_FEATURE_COUNT, dtype=np.float64)[labels]
    class_axes -= np.mean(np.eye(SIGNAL_FEATURE_COUNT, dtype=np.float64), axis=0)
    if scenario.startswith("linear"):
        signal = signal_strength * class_axes + signal_noise
    else:
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_rows, 1))
        signal = signs * signal_strength * class_axes + signal_noise
    values = np.concatenate(
        [signal, nuisance_strength * nuisance_noise], axis=1
    )
    return _row_l2(values), labels


def _folds(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    planner = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    placeholder = np.zeros(y.shape[0], dtype=np.int8)
    return [(train, heldout) for train, heldout in planner.split(placeholder, y)]


def _oi_kwargs(seed: int) -> dict[str, Any]:
    return {
        "model_type": "MiniBatchKMeans",
        "kmeans_k": K_PER_CLASS,
        "kmeans_kwargs": {
            "batch_size": 256,
            "max_no_improvement": 5,
            "compute_labels": False,
            "n_init": 1,
            "init": "random",
            "random_state": seed,
        },
    }


def _crossfit_selector(
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    scores: list[float] = []
    metric_conditions: list[float] = []
    applied = 0
    prototypes_before = 0
    for fold, (train, heldout) in enumerate(_folds(y, seed)):
        fold_seed = seed + fold
        if method == "linear_probe":
            estimator = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=fold_seed,
            )
            estimator.fit(X[train], y[train])
            scores.append(float(estimator.score(X[heldout], y[heldout])))
            continue
        refined = method.endswith("refined") and not method.endswith("unrefined")
        kwargs = _oi_kwargs(fold_seed)
        if method.startswith("diagonal"):
            selector: Any = DiagonalMetricOverlapIndex(
                prototype_refinement=refined,
                overlap_index_kwargs=kwargs,
            )
        elif method.startswith("raw"):
            selector = OverlapIndex(prototype_refinement=refined, **kwargs)
        else:
            raise ValueError(f"unknown method {method!r}")
        selector.fit(X[train], y[train])
        scores.append(float(selector.score_fixed(X[heldout], y[heldout])))
        if method.startswith("diagonal"):
            metric_conditions.append(
                float(selector.metric_diagnostics_["weight_condition"])
            )
        refinement = getattr(selector, "prototype_refinement_", {})
        if isinstance(refinement, Mapping):
            applied += int(refinement.get("applied_count", 0))
            prototypes_before += int(refinement.get("prototype_count_before", 0))
    return float(np.mean(scores)), {
        "metric_condition_mean": (
            float(np.mean(metric_conditions)) if metric_conditions else None
        ),
        "refinement_applied_rate": (
            float(applied / prototypes_before) if prototypes_before else 0.0
        ),
    }


def _reference_accuracy(
    *,
    scenario: str,
    seed: int,
    candidate: Mapping[str, Any],
) -> float:
    train_X, train_y = generate_embedding(
        scenario=scenario,
        seed=seed,
        split="reference_train",
        n_per_class=REFERENCE_TRAIN_PER_CLASS,
        candidate=candidate,
    )
    test_X, test_y = generate_embedding(
        scenario=scenario,
        seed=seed,
        split="reference_test",
        n_per_class=REFERENCE_TEST_PER_CLASS,
        candidate=candidate,
    )
    if scenario.startswith("linear"):
        model: Any = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
    else:
        model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_X, train_y)
    return float(model.score(test_X, test_y))


def _execution_order(case_index: int) -> tuple[str, ...]:
    shift = case_index % len(METHODS)
    return METHODS[shift:] + METHODS[:shift]


def _selection_metrics(
    selector: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, Any]:
    ids = tuple(candidate["candidate_id"] for candidate in PANEL_CANDIDATES)
    scores = np.asarray([selector[candidate_id] for candidate_id in ids])
    outcomes = np.asarray([reference[candidate_id] for candidate_id in ids])
    best_score = float(np.max(scores))
    selected = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1.0e-12))
    best_outcome = float(np.max(outcomes))
    selected_outcome = float(np.mean(outcomes[selected]))
    correlation = spearmanr(scores, outcomes).statistic
    return {
        "regret_pp": 100.0 * (best_outcome - selected_outcome),
        "exact_best": bool(np.any(np.isclose(outcomes[selected], best_outcome))),
        "within_one_pp": bool(best_outcome - selected_outcome <= 0.01 + 1.0e-12),
        "spearman": None if not np.isfinite(correlation) else float(correlation),
        "selected_candidates": [ids[int(index)] for index in selected],
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["scenario"]), int(row["budget_per_class"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    panel_metrics: list[dict[str, Any]] = []
    for (scenario, budget, seed), panel_rows in sorted(grouped.items()):
        references = {
            str(row["candidate_id"]): float(row["reference_accuracy"])
            for row in panel_rows
        }
        for method in METHODS:
            method_rows = [row for row in panel_rows if row["method"] == method]
            if len(method_rows) != len(PANEL_CANDIDATES):
                raise ValueError("incomplete method panel")
            selector = {
                str(row["candidate_id"]): float(row["selector_score"])
                for row in method_rows
            }
            metric = _selection_metrics(selector, references)
            metric.update(
                {
                    "scenario": scenario,
                    "budget_per_class": budget,
                    "seed": seed,
                    "method": method,
                }
            )
            panel_metrics.append(metric)
    aggregate: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for budget in BUDGETS_PER_CLASS:
            for method in METHODS:
                metrics = [
                    row
                    for row in panel_metrics
                    if row["scenario"] == scenario
                    and row["budget_per_class"] == budget
                    and row["method"] == method
                ]
                method_runtime_rows = [
                    row
                    for row in rows
                    if row["scenario"] == scenario
                    and row["budget_per_class"] == budget
                    and row["method"] == method
                ]
                probe_lookup = {
                    (int(row["seed"]), str(row["candidate_id"])): float(
                        row["wall_seconds"]
                    )
                    for row in rows
                    if row["scenario"] == scenario
                    and row["budget_per_class"] == budget
                    and row["method"] == "linear_probe"
                }
                ratios = [
                    float(row["wall_seconds"])
                    / probe_lookup[(int(row["seed"]), str(row["candidate_id"]))]
                    for row in method_runtime_rows
                ]
                correlations = [
                    float(row["spearman"])
                    for row in metrics
                    if row["spearman"] is not None
                ]
                aggregate.append(
                    {
                        "scenario": scenario,
                        "budget_per_class": budget,
                        "method": method,
                        "mean_regret_pp": float(
                            np.mean([row["regret_pp"] for row in metrics])
                        ),
                        "exact_best_rate": float(
                            np.mean([row["exact_best"] for row in metrics])
                        ),
                        "within_one_pp_rate": float(
                            np.mean([row["within_one_pp"] for row in metrics])
                        ),
                        "median_spearman": (
                            float(statistics.median(correlations))
                            if correlations
                            else None
                        ),
                        "median_wall_seconds": float(
                            statistics.median(
                                float(row["wall_seconds"])
                                for row in method_runtime_rows
                            )
                        ),
                        "median_wall_ratio_vs_probe": float(statistics.median(ratios)),
                    }
                )
    return {"panel_metrics": panel_metrics, "aggregate": aggregate}


def _protocol_identity(protocol_path: Path) -> tuple[str, dict[str, Any]]:
    raw = protocol_path.read_bytes()
    parsed = json.loads(raw)
    return _sha256_bytes(raw), parsed


def run_experiment(protocol_path: Path) -> dict[str, Any]:
    protocol_sha256, protocol = _protocol_identity(protocol_path)
    expected = protocol["frozen_grid"]
    if tuple(expected["scenarios"]) != SCENARIOS:
        raise ValueError("protocol scenario grid does not match implementation")
    if tuple(expected["budgets_per_class"]) != BUDGETS_PER_CLASS:
        raise ValueError("protocol budget grid does not match implementation")
    if tuple(expected["seeds"]) != SEEDS:
        raise ValueError("protocol seed grid does not match implementation")
    first_X, first_y = generate_embedding(
        scenario=SCENARIOS[0],
        seed=SEEDS[0],
        split="selector",
        n_per_class=BUDGETS_PER_CLASS[0],
        candidate=PANEL_CANDIDATES[0],
    )
    for method in METHODS:
        _crossfit_selector(first_X, first_y, method=method, seed=SEEDS[0])

    rows: list[dict[str, Any]] = []
    case_index = 0
    for scenario in SCENARIOS:
        for budget in BUDGETS_PER_CLASS:
            for seed in SEEDS:
                references = {
                    str(candidate["candidate_id"]): _reference_accuracy(
                        scenario=scenario, seed=seed, candidate=candidate
                    )
                    for candidate in PANEL_CANDIDATES
                }
                for candidate in PANEL_CANDIDATES:
                    X, y = generate_embedding(
                        scenario=scenario,
                        seed=seed,
                        split="selector",
                        n_per_class=budget,
                        candidate=candidate,
                    )
                    order = _execution_order(case_index)
                    for position, method in enumerate(order):
                        started = time.perf_counter()
                        score, diagnostics = _crossfit_selector(
                            X, y, method=method, seed=seed
                        )
                        elapsed = time.perf_counter() - started
                        rows.append(
                            {
                                "scenario": scenario,
                                "budget_per_class": budget,
                                "seed": seed,
                                "candidate_id": str(candidate["candidate_id"]),
                                "signal_strength": float(candidate["signal"]),
                                "nuisance_strength": float(candidate["nuisance"]),
                                "method": method,
                                "execution_position": position,
                                "selector_score": score,
                                "reference_head": (
                                    "linear" if scenario.startswith("linear") else "knn"
                                ),
                                "reference_accuracy": references[
                                    str(candidate["candidate_id"])
                                ],
                                "wall_seconds": float(elapsed),
                                **diagnostics,
                            }
                        )
                    case_index += 1
    expected_rows = (
        len(SCENARIOS)
        * len(BUDGETS_PER_CLASS)
        * len(SEEDS)
        * len(PANEL_CANDIDATES)
        * len(METHODS)
    )
    if len(rows) != expected_rows:
        raise AssertionError("toy experiment emitted an incomplete row grid")
    return {
        "schema_version": 1,
        "study": "supervised_diagonal_metric_toy",
        "artifact_status": "completed",
        "development_only": True,
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "grid": {
            "scenarios": list(SCENARIOS),
            "budgets_per_class": list(BUDGETS_PER_CLASS),
            "seeds": list(SEEDS),
            "candidate_ids": [row["candidate_id"] for row in PANEL_CANDIDATES],
            "methods": list(METHODS),
            "row_count": expected_rows,
            "class_count": CLASS_COUNT,
            "feature_count": FEATURE_COUNT,
            "fold_count": FOLDS,
        },
        "metric": {
            "formula": "normalized[floor+(1-floor)*between/(between+within+epsilon)]",
            "weight_floor": WEIGHT_FLOOR,
            "fit_scope": "selector training fold only",
        },
        "warmup_excluded": True,
        "rows": rows,
        "summary": summarize(rows),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Supervised diagonal metric toy",
        "",
        "Development-only paired synthetic screen; not promotion or confirmation evidence.",
        "",
        "| Scenario | Budget/class | Method | Regret pp | Spearman | Within 1 pp | Wall/probe |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["aggregate"]:
        spearman = "—" if row["median_spearman"] is None else f"{row['median_spearman']:.3f}"
        lines.append(
            "| {scenario} | {budget_per_class} | {method} | {mean_regret_pp:.3f} | "
            "{spearman} | {within_one_pp_rate:.2f} | {median_wall_ratio_vs_probe:.2f} |".format(
                spearman=spearman, **row
            )
        )
    lines.extend(
        [
            "",
            "Runtime ratios are paired per candidate against the same cross-fitted linear probe. "
            "The toy dimensions and class count are not a substitute for a real-backbone runtime gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/diagonal_metric_toy/protocol.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output
    if output.exists():
        raise FileExistsError(f"refuse to overwrite {output}")
    output.mkdir(parents=True)
    payload = run_experiment(args.protocol)
    raw_path = output / "raw_results.json"
    report_path = output / "report.md"
    raw_tmp = output / ".raw_results.json.tmp"
    report_tmp = output / ".report.md.tmp"
    raw_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_tmp.write_text(_markdown(payload), encoding="utf-8")
    os.replace(raw_tmp, raw_path)
    os.replace(report_tmp, report_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DiagonalMetricOverlapIndex",
    "FittedDiagonalMetric",
    "fit_diagonal_metric",
    "generate_embedding",
    "run_experiment",
    "summarize",
]
