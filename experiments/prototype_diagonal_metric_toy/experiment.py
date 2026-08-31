"""Prototype-aware diagonal metric toy comparison.

The metric is estimated entirely inside each selector training fold.  A
stratified internal half fits provisional class-owned prototypes; the other
half estimates per-feature correct-versus-impostor prototype margins.  The
resulting bounded diagonal metric is then used consistently for final OI fit
and held-out scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

from overlapindex import OverlapIndex

from experiments.diagonal_metric_toy import experiment as base


METHODS = (
    "raw_unrefined",
    "raw_refined",
    "class_diagonal_refined",
    "prototype_diagonal_unrefined",
    "prototype_diagonal_refined",
    "linear_probe",
)
INTERNAL_PROTOTYPE_FRACTION = 0.5


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _internal_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prototype_rows: list[np.ndarray] = []
    metric_rows: list[np.ndarray] = []
    for label in dict.fromkeys(y.tolist()):
        indices = np.flatnonzero(y == label)
        shuffled = np.asarray(indices[rng.permutation(indices.size)], dtype=np.int64)
        cut = max(base.K_PER_CLASS, int(np.floor(indices.size * INTERNAL_PROTOTYPE_FRACTION)))
        cut = min(cut, indices.size - 1)
        if cut < base.K_PER_CLASS or indices.size - cut < 1:
            raise ValueError("each class needs enough rows for the internal metric split")
        prototype_rows.append(shuffled[:cut])
        metric_rows.append(shuffled[cut:])
    return np.sort(np.concatenate(prototype_rows)), np.sort(np.concatenate(metric_rows))


def fit_prototype_margin_metric(
    X: Any,
    y: Any,
    *,
    seed: int,
    weight_floor: float = base.WEIGHT_FLOOR,
) -> base.FittedDiagonalMetric:
    """Fit bounded weights from correct-versus-impostor prototype margins."""

    if type(seed) is not int:
        raise TypeError("seed must be an int")
    if not isinstance(weight_floor, float) or not 0.0 < weight_floor <= 1.0:
        raise ValueError("weight_floor must be a float in (0, 1]")
    values, target = base._validate_X_y(X, y)
    prototype_indices, metric_indices = _internal_split(target, seed)
    centers: list[np.ndarray] = []
    owners: list[Any] = []
    for class_position, label in enumerate(dict.fromkeys(target.tolist())):
        class_rows = prototype_indices[target[prototype_indices] == label]
        if class_rows.size < base.K_PER_CLASS:
            raise ValueError("internal prototype split has too few rows for a class")
        model = MiniBatchKMeans(
            n_clusters=base.K_PER_CLASS,
            batch_size=256,
            max_no_improvement=5,
            compute_labels=False,
            n_init=1,
            init="random",
            random_state=seed + class_position,
        )
        model.fit(values[class_rows])
        centers.append(np.asarray(model.cluster_centers_, dtype=np.float64))
        owners.extend([label] * base.K_PER_CLASS)
    prototype_matrix = np.vstack(centers)
    owner_array = np.asarray(owners)
    evaluation = values[metric_indices]
    evaluation_y = target[metric_indices]
    difference = evaluation[:, None, :] - prototype_matrix[None, :, :]
    squared = difference * difference
    distances = np.sum(squared, axis=2)
    own_mask = evaluation_y[:, None] == owner_array[None, :]
    own_distance = np.where(own_mask, distances, np.inf)
    wrong_distance = np.where(~own_mask, distances, np.inf)
    own_id = np.argmin(own_distance, axis=1)
    wrong_id = np.argmin(wrong_distance, axis=1)
    rows = np.arange(evaluation.shape[0])
    own_feature_distance = squared[rows, own_id]
    wrong_feature_distance = squared[rows, wrong_id]
    margin = np.mean(wrong_feature_distance - own_feature_distance, axis=0)
    positive_margin = np.maximum(margin, 0.0)
    residual = np.mean(own_feature_distance, axis=0)
    epsilon = max(float(np.mean(residual)) * 1.0e-6, 1.0e-12)
    relevance = positive_margin / (positive_margin + residual + epsilon)
    raw_weights = weight_floor + (1.0 - weight_floor) * relevance
    weights = np.asarray(raw_weights / float(np.mean(raw_weights)), dtype=np.float64)
    scale = np.asarray(np.sqrt(weights), dtype=np.float64)
    mean = np.asarray(np.mean(values, axis=0), dtype=np.float64)
    for array in (mean, weights, scale):
        array.setflags(write=False)
    state = {
        "seed": seed,
        "weight_floor": weight_floor,
        "mean": mean.tolist(),
        "weights": weights.tolist(),
    }
    state_sha256 = _sha256_bytes(_json_bytes(state))
    diagnostics = {
        "weight_floor": weight_floor,
        "n_rows_fit": int(values.shape[0]),
        "n_features_fit": int(values.shape[1]),
        "n_classes_fit": int(len(dict.fromkeys(target.tolist()))),
        "prototype_fit_row_count": int(prototype_indices.size),
        "margin_fit_row_count": int(metric_indices.size),
        "internal_split_disjoint": bool(
            np.intersect1d(prototype_indices, metric_indices).size == 0
        ),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_condition": float(np.max(weights) / np.min(weights)),
        "positive_margin_feature_count": int(np.count_nonzero(positive_margin)),
        "state_sha256": state_sha256,
    }
    return base.FittedDiagonalMetric(mean, weights, scale, state_sha256, diagnostics)


class PrototypeDiagonalOverlapIndex:
    """Two-stage experiment-local prototype-margin OI wrapper."""

    def __init__(
        self,
        *,
        prototype_refinement: bool,
        overlap_index_kwargs: Mapping[str, Any],
        metric_seed: int,
        weight_floor: float = base.WEIGHT_FLOOR,
    ) -> None:
        if type(prototype_refinement) is not bool:
            raise TypeError("prototype_refinement must be a strict bool")
        if type(metric_seed) is not int:
            raise TypeError("metric_seed must be an int")
        if not isinstance(overlap_index_kwargs, Mapping):
            raise TypeError("overlap_index_kwargs must be a mapping")
        if "prototype_refinement" in overlap_index_kwargs:
            raise ValueError("prototype_refinement must not be duplicated")
        self.prototype_refinement = prototype_refinement
        self.overlap_index_kwargs = deepcopy(dict(overlap_index_kwargs))
        self.metric_seed = metric_seed
        self.weight_floor = weight_floor
        self.metric_: base.FittedDiagonalMetric | None = None
        self.estimator_: OverlapIndex | None = None

    def fit(self, X: Any, y: Any) -> "PrototypeDiagonalOverlapIndex":
        values, target = base._validate_X_y(X, y)
        metric = fit_prototype_margin_metric(
            values, target, seed=self.metric_seed, weight_floor=self.weight_floor
        )
        estimator = OverlapIndex(
            prototype_refinement=self.prototype_refinement,
            **deepcopy(self.overlap_index_kwargs),
        )
        estimator.fit(metric.transform(values), target)
        self.metric_ = metric
        self.estimator_ = estimator
        return self

    def _require_fit(self) -> tuple[base.FittedDiagonalMetric, OverlapIndex]:
        if self.metric_ is None or self.estimator_ is None:
            raise ValueError("PrototypeDiagonalOverlapIndex is not fitted")
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


def _crossfit(
    X: np.ndarray, y: np.ndarray, *, method: str, seed: int
) -> tuple[float, dict[str, Any]]:
    mapped = {
        "raw_unrefined": "raw_unrefined",
        "raw_refined": "raw_refined",
        "class_diagonal_refined": "diagonal_refined",
        "linear_probe": "linear_probe",
    }
    if method in mapped:
        return base._crossfit_selector(X, y, method=mapped[method], seed=seed)
    refined = method == "prototype_diagonal_refined"
    if method not in {"prototype_diagonal_unrefined", "prototype_diagonal_refined"}:
        raise ValueError(f"unknown method {method!r}")
    scores: list[float] = []
    conditions: list[float] = []
    applied = 0
    before = 0
    for fold, (train, heldout) in enumerate(base._folds(y, seed)):
        fold_seed = seed + fold
        selector = PrototypeDiagonalOverlapIndex(
            prototype_refinement=refined,
            overlap_index_kwargs=base._oi_kwargs(fold_seed),
            metric_seed=fold_seed + 10_000,
        ).fit(X[train], y[train])
        scores.append(float(selector.score_fixed(X[heldout], y[heldout])))
        conditions.append(float(selector.metric_diagnostics_["weight_condition"]))
        summary = selector.prototype_refinement_
        applied += int(summary.get("applied_count", 0))
        before += int(summary.get("prototype_count_before", 0))
    return float(np.mean(scores)), {
        "metric_condition_mean": float(np.mean(conditions)),
        "refinement_applied_rate": float(applied / before) if before else 0.0,
    }


def _selection_metrics(
    selector: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, Any]:
    ids = tuple(candidate["candidate_id"] for candidate in base.PANEL_CANDIDATES)
    scores = np.asarray([selector[candidate_id] for candidate_id in ids])
    outcomes = np.asarray([reference[candidate_id] for candidate_id in ids])
    selected = np.flatnonzero(
        np.isclose(scores, float(np.max(scores)), rtol=0.0, atol=1.0e-12)
    )
    best = float(np.max(outcomes))
    selected_outcome = float(np.mean(outcomes[selected]))
    correlation = spearmanr(scores, outcomes).statistic
    return {
        "regret_pp": 100.0 * (best - selected_outcome),
        "exact_best": bool(np.any(np.isclose(outcomes[selected], best))),
        "within_one_pp": bool(best - selected_outcome <= 0.01 + 1.0e-12),
        "spearman": None if not np.isfinite(correlation) else float(correlation),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    panel_metrics: list[dict[str, Any]] = []
    for scenario in base.SCENARIOS:
        for budget in base.BUDGETS_PER_CLASS:
            for seed in base.SEEDS:
                panel = [
                    row
                    for row in rows
                    if row["scenario"] == scenario
                    and row["budget_per_class"] == budget
                    and row["seed"] == seed
                ]
                reference = {
                    str(row["candidate_id"]): float(row["reference_accuracy"])
                    for row in panel
                }
                for method in METHODS:
                    method_rows = [row for row in panel if row["method"] == method]
                    if len(method_rows) != len(base.PANEL_CANDIDATES):
                        raise ValueError("incomplete prototype metric panel")
                    metric = _selection_metrics(
                        {
                            str(row["candidate_id"]): float(row["selector_score"])
                            for row in method_rows
                        },
                        reference,
                    )
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
    for scenario in base.SCENARIOS:
        for budget in base.BUDGETS_PER_CLASS:
            probe = {
                (int(row["seed"]), str(row["candidate_id"])): float(row["wall_seconds"])
                for row in rows
                if row["scenario"] == scenario
                and row["budget_per_class"] == budget
                and row["method"] == "linear_probe"
            }
            for method in METHODS:
                metrics = [
                    row
                    for row in panel_metrics
                    if row["scenario"] == scenario
                    and row["budget_per_class"] == budget
                    and row["method"] == method
                ]
                timings = [
                    row
                    for row in rows
                    if row["scenario"] == scenario
                    and row["budget_per_class"] == budget
                    and row["method"] == method
                ]
                correlations = [
                    float(row["spearman"])
                    for row in metrics
                    if row["spearman"] is not None
                ]
                ratios = [
                    float(row["wall_seconds"])
                    / probe[(int(row["seed"]), str(row["candidate_id"]))]
                    for row in timings
                ]
                aggregate.append(
                    {
                        "scenario": scenario,
                        "budget_per_class": budget,
                        "method": method,
                        "mean_regret_pp": float(
                            np.mean([row["regret_pp"] for row in metrics])
                        ),
                        "median_spearman": (
                            float(statistics.median(correlations))
                            if correlations
                            else None
                        ),
                        "within_one_pp_rate": float(
                            np.mean([row["within_one_pp"] for row in metrics])
                        ),
                        "median_wall_seconds": float(
                            statistics.median(float(row["wall_seconds"]) for row in timings)
                        ),
                        "median_wall_ratio_vs_probe": float(statistics.median(ratios)),
                    }
                )
    return {"panel_metrics": panel_metrics, "aggregate": aggregate}


def run_experiment(protocol_path: Path) -> dict[str, Any]:
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    if tuple(protocol["frozen_grid"]["scenarios"]) != base.SCENARIOS:
        raise ValueError("protocol scenarios do not match implementation")
    if tuple(protocol["frozen_grid"]["budgets_per_class"]) != base.BUDGETS_PER_CLASS:
        raise ValueError("protocol budgets do not match implementation")
    if tuple(protocol["frozen_grid"]["seeds"]) != base.SEEDS:
        raise ValueError("protocol seeds do not match implementation")
    first_X, first_y = base.generate_embedding(
        scenario=base.SCENARIOS[0],
        seed=base.SEEDS[0],
        split="selector",
        n_per_class=base.BUDGETS_PER_CLASS[0],
        candidate=base.PANEL_CANDIDATES[0],
    )
    for method in METHODS:
        _crossfit(first_X, first_y, method=method, seed=base.SEEDS[0])
    rows: list[dict[str, Any]] = []
    case_index = 0
    for scenario in base.SCENARIOS:
        for budget in base.BUDGETS_PER_CLASS:
            for seed in base.SEEDS:
                references = {
                    str(candidate["candidate_id"]): base._reference_accuracy(
                        scenario=scenario, seed=seed, candidate=candidate
                    )
                    for candidate in base.PANEL_CANDIDATES
                }
                for candidate in base.PANEL_CANDIDATES:
                    X, y = base.generate_embedding(
                        scenario=scenario,
                        seed=seed,
                        split="selector",
                        n_per_class=budget,
                        candidate=candidate,
                    )
                    shift = case_index % len(METHODS)
                    order = METHODS[shift:] + METHODS[:shift]
                    for position, method in enumerate(order):
                        started = time.perf_counter()
                        score, diagnostics = _crossfit(X, y, method=method, seed=seed)
                        rows.append(
                            {
                                "scenario": scenario,
                                "budget_per_class": budget,
                                "seed": seed,
                                "candidate_id": str(candidate["candidate_id"]),
                                "method": method,
                                "execution_position": position,
                                "selector_score": score,
                                "reference_head": (
                                    "linear" if scenario.startswith("linear") else "knn"
                                ),
                                "reference_accuracy": references[str(candidate["candidate_id"])],
                                "wall_seconds": float(time.perf_counter() - started),
                                **diagnostics,
                            }
                        )
                    case_index += 1
    expected = (
        len(base.SCENARIOS)
        * len(base.BUDGETS_PER_CLASS)
        * len(base.SEEDS)
        * len(base.PANEL_CANDIDATES)
        * len(METHODS)
    )
    if len(rows) != expected:
        raise AssertionError("prototype metric toy emitted an incomplete row grid")
    dependency = Path(base.__file__)
    return {
        "schema_version": 1,
        "study": "prototype_margin_diagonal_metric_toy",
        "artifact_status": "completed",
        "development_only": True,
        "protocol_sha256": _sha256_bytes(protocol_raw),
        "implementation_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "base_experiment_dependency": {
            "path": str(dependency),
            "sha256": _sha256_bytes(dependency.read_bytes()),
        },
        "grid": {
            "scenarios": list(base.SCENARIOS),
            "budgets_per_class": list(base.BUDGETS_PER_CLASS),
            "seeds": list(base.SEEDS),
            "candidate_ids": [row["candidate_id"] for row in base.PANEL_CANDIDATES],
            "methods": list(METHODS),
            "row_count": expected,
        },
        "metric": protocol["prototype_margin_metric"],
        "warmup_excluded": True,
        "rows": rows,
        "summary": summarize(rows),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Prototype-margin diagonal metric toy",
        "",
        "Development-only paired synthetic screen; not promotion or confirmation evidence.",
        "",
        "| Scenario | Budget/class | Method | Regret pp | Spearman | Within 1 pp | Wall/probe |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["aggregate"]:
        corr = "—" if row["median_spearman"] is None else f"{row['median_spearman']:.3f}"
        lines.append(
            "| {scenario} | {budget_per_class} | {method} | {mean_regret_pp:.3f} | "
            "{corr} | {within_one_pp_rate:.2f} | {median_wall_ratio_vs_probe:.2f} |".format(
                corr=corr, **row
            )
        )
    lines.extend(
        [
            "",
            "Prototype-margin runtime includes the provisional prototype fit, margin estimation, "
            "final OI fit, and held-out score. Toy ratios are not real-backbone runtime gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/prototype_diagonal_metric_toy/protocol.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refuse to overwrite {args.output}")
    args.output.mkdir(parents=True)
    payload = run_experiment(args.protocol)
    raw_tmp = args.output / ".raw_results.json.tmp"
    report_tmp = args.output / ".report.md.tmp"
    raw_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_tmp.write_text(_markdown(payload), encoding="utf-8")
    os.replace(raw_tmp, args.output / "raw_results.json")
    os.replace(report_tmp, args.output / "report.md")
    print(args.output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PrototypeDiagonalOverlapIndex",
    "fit_prototype_margin_metric",
    "run_experiment",
    "summarize",
]
