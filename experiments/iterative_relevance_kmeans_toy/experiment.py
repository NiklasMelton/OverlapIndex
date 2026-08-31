"""One-update relevance-weighted K-means/OI toy comparison.

The experiment performs one initial class-owned MiniBatchKMeans fit, learns a
bounded diagonal metric from correct-versus-impostor prototype margins, then
executes exactly one weighted Lloyd assignment/centroid update.  It does not
run a second K-means fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

from experiments.diagonal_metric_toy import experiment as base
from experiments.prototype_diagonal_metric_toy import experiment as two_stage


METHODS = (
    "raw_unrefined",
    "raw_refined",
    "class_diagonal_refined",
    "two_stage_prototype_refined",
    "one_update_relevance",
    "linear_probe",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _distances(X: np.ndarray, centers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    difference = X[:, None, :] - centers[None, :, :]
    return np.sum((difference * difference) * weights[None, None, :], axis=2)


def oi_score_from_centers(
    X: Any,
    y: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
) -> float:
    """Compute the exact single-label OI event definition for fixed centers."""

    values, target = base._validate_X_y(X, y)
    prototype_matrix = np.asarray(centers, dtype=np.float64)
    owner_array = np.asarray(owners)
    feature_weights = np.asarray(weights, dtype=np.float64)
    if prototype_matrix.ndim != 2 or prototype_matrix.shape[1] != values.shape[1]:
        raise ValueError("centers have the wrong shape")
    if owner_array.ndim != 1 or owner_array.size != prototype_matrix.shape[0]:
        raise ValueError("owners must align with centers")
    if feature_weights.shape != (values.shape[1],) or np.any(feature_weights <= 0.0):
        raise ValueError("weights must be a positive feature vector")
    classes = list(dict.fromkeys(target.tolist()))
    if set(classes) != set(dict.fromkeys(owner_array.tolist())):
        raise ValueError("evaluation and prototype class sets must match")
    singleton: list[float] = []
    for source in classes:
        source_rows = values[target == source]
        own_centers = prototype_matrix[owner_array == source]
        if own_centers.shape[0] < 2:
            threshold = np.full(source_rows.shape[0], np.inf, dtype=np.float64)
        else:
            own_distances = _distances(
                source_rows, own_centers, feature_weights
            )
            threshold = np.partition(own_distances, kth=1, axis=1)[:, 1]
        pair_scores: list[float] = []
        for target_class in classes:
            if target_class == source:
                continue
            target_centers = prototype_matrix[owner_array == target_class]
            target_best = np.min(
                _distances(source_rows, target_centers, feature_weights), axis=1
            )
            hits = int(np.count_nonzero(target_best < threshold))
            pair_scores.append(1.0 - float(hits) / float(source_rows.shape[0]))
        singleton.append(min(pair_scores) if pair_scores else 1.0)
    return float(np.mean(singleton))


class OneUpdateRelevanceKMeans:
    """One initial K-means plus one relevance-weighted Lloyd update."""

    def __init__(self, *, seed: int, weight_floor: float = base.WEIGHT_FLOOR) -> None:
        if type(seed) is not int:
            raise TypeError("seed must be an int")
        if not isinstance(weight_floor, float) or not 0.0 < weight_floor <= 1.0:
            raise ValueError("weight_floor must be a float in (0, 1]")
        self.seed = seed
        self.weight_floor = weight_floor
        self.centers_: np.ndarray | None = None
        self.owners_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] | None = None

    def fit(self, X: Any, y: Any) -> "OneUpdateRelevanceKMeans":
        values, target = base._validate_X_y(X, y)
        labels = list(dict.fromkeys(target.tolist()))
        center_blocks: list[np.ndarray] = []
        owners: list[Any] = []
        for position, label in enumerate(labels):
            rows = values[target == label]
            estimator = MiniBatchKMeans(
                n_clusters=base.K_PER_CLASS,
                batch_size=256,
                max_no_improvement=5,
                compute_labels=False,
                n_init=1,
                init="random",
                random_state=self.seed + position,
            )
            estimator.fit(rows)
            center_blocks.append(np.asarray(estimator.cluster_centers_, dtype=np.float64))
            owners.extend([label] * base.K_PER_CLASS)
        initial_centers = np.vstack(center_blocks)
        owner_array = np.asarray(owners)
        unweighted = np.ones(values.shape[1], dtype=np.float64)
        all_distances = _distances(values, initial_centers, unweighted)
        own_mask = target[:, None] == owner_array[None, :]
        own_id = np.argmin(np.where(own_mask, all_distances, np.inf), axis=1)
        wrong_id = np.argmin(np.where(~own_mask, all_distances, np.inf), axis=1)
        rows = np.arange(values.shape[0])
        own_feature = (values - initial_centers[own_id]) ** 2
        wrong_feature = (values - initial_centers[wrong_id]) ** 2
        margin = np.mean(wrong_feature - own_feature, axis=0)
        positive_margin = np.maximum(margin, 0.0)
        residual = np.mean(own_feature, axis=0)
        epsilon = max(float(np.mean(residual)) * 1.0e-6, 1.0e-12)
        relevance = positive_margin / (positive_margin + residual + epsilon)
        raw_weights = self.weight_floor + (1.0 - self.weight_floor) * relevance
        weights = np.asarray(raw_weights / float(np.mean(raw_weights)), dtype=np.float64)

        weighted_distances = _distances(values, initial_centers, weights)
        assignments = np.argmin(
            np.where(own_mask, weighted_distances, np.inf), axis=1
        )
        updated_centers = np.asarray(initial_centers, dtype=np.float64).copy()
        empty_count = 0
        for prototype_id in range(initial_centers.shape[0]):
            assigned = values[assignments == prototype_id]
            if assigned.shape[0]:
                updated_centers[prototype_id] = np.mean(assigned, axis=0)
            else:
                empty_count += 1
        for array in (updated_centers, owner_array, weights):
            array.setflags(write=False)
        state = {
            "seed": self.seed,
            "weight_floor": self.weight_floor,
            "owners": owner_array.tolist(),
            "centers": updated_centers.tolist(),
            "weights": weights.tolist(),
        }
        self.centers_ = updated_centers
        self.owners_ = owner_array
        self.weights_ = weights
        self.diagnostics_ = {
            "n_rows_fit": int(values.shape[0]),
            "n_features_fit": int(values.shape[1]),
            "n_classes_fit": len(labels),
            "prototype_count": int(updated_centers.shape[0]),
            "positive_margin_feature_count": int(np.count_nonzero(positive_margin)),
            "weight_condition": float(np.max(weights) / np.min(weights)),
            "empty_prototype_count_after_update": empty_count,
            "kmeans_fit_count": 1,
            "weighted_lloyd_update_count": 1,
            "state_sha256": _sha256_bytes(_json_bytes(state)),
        }
        return self

    def _require_fit(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.centers_ is None or self.owners_ is None or self.weights_ is None:
            raise ValueError("OneUpdateRelevanceKMeans is not fitted")
        return self.centers_, self.owners_, self.weights_

    def score_fixed(self, X: Any, y: Any) -> float:
        centers, owners, weights = self._require_fit()
        return oi_score_from_centers(
            X, y, centers=centers, owners=owners, weights=weights
        )


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
    if method == "two_stage_prototype_refined":
        return two_stage._crossfit(
            X, y, method="prototype_diagonal_refined", seed=seed
        )
    if method != "one_update_relevance":
        raise ValueError(f"unknown method {method!r}")
    scores: list[float] = []
    conditions: list[float] = []
    for fold, (train, heldout) in enumerate(base._folds(y, seed)):
        selector = OneUpdateRelevanceKMeans(seed=seed + fold).fit(
            X[train], y[train]
        )
        scores.append(float(selector.score_fixed(X[heldout], y[heldout])))
        if selector.diagnostics_ is None:
            raise AssertionError("one-update selector has no diagnostics")
        conditions.append(float(selector.diagnostics_["weight_condition"]))
    return float(np.mean(scores)), {
        "metric_condition_mean": float(np.mean(conditions)),
        "refinement_applied_rate": 0.0,
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
        "spearman": None if not np.isfinite(correlation) else float(correlation),
        "within_one_pp": bool(best - selected_outcome <= 0.01 + 1.0e-12),
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
                        raise ValueError("incomplete one-update panel")
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
    grid = protocol["frozen_grid"]
    if tuple(grid["scenarios"]) != base.SCENARIOS:
        raise ValueError("protocol scenarios do not match implementation")
    if tuple(grid["budgets_per_class"]) != base.BUDGETS_PER_CLASS:
        raise ValueError("protocol budgets do not match implementation")
    if tuple(grid["seeds"]) != base.SEEDS:
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
        raise AssertionError("one-update toy emitted an incomplete row grid")
    dependency_paths = (Path(base.__file__), Path(two_stage.__file__))
    return {
        "schema_version": 1,
        "study": "one_update_relevance_weighted_kmeans_toy",
        "artifact_status": "completed",
        "development_only": True,
        "protocol_sha256": _sha256_bytes(protocol_raw),
        "implementation_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "dependency_hashes": {
            str(path): _sha256_bytes(path.read_bytes()) for path in dependency_paths
        },
        "grid": {
            "scenarios": list(base.SCENARIOS),
            "budgets_per_class": list(base.BUDGETS_PER_CLASS),
            "seeds": list(base.SEEDS),
            "candidate_ids": [row["candidate_id"] for row in base.PANEL_CANDIDATES],
            "methods": list(METHODS),
            "row_count": expected,
        },
        "formulation": protocol["one_update_formulation"],
        "warmup_excluded": True,
        "rows": rows,
        "summary": summarize(rows),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# One-update relevance-weighted K-means toy",
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
            "The one-update runtime includes one initial K-means fit, one margin/weight pass, "
            "one weighted Lloyd update, and held-out OI event scoring.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/iterative_relevance_kmeans_toy/protocol.json"),
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
    "OneUpdateRelevanceKMeans",
    "oi_score_from_centers",
    "run_experiment",
    "summarize",
]
