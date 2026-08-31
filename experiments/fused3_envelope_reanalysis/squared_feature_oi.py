"""Compact squared-feature FUSED3 diagnostic on frozen Aircraft panels.

This is a post-outcome mechanism experiment.  It does not modify the frozen
FUSED3 confirmation estimand, candidate, gates, or decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from scipy.stats import rankdata

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen


STUDY = "fused3_confirmation_v2_aircraft_squared_feature_diagnostic"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
DATASET_ID = "torchvision_fgvc_aircraft"
BASELINE = "FUSED3"
SQUARED_SPECS: tuple[tuple[str, int | None, str], ...] = (
    ("FUSED3-SQ32", 32, "compact_candidate"),
    ("FUSED3-SQ64", 64, "compact_candidate"),
    ("FUSED3-SQALL", None, "mechanism_upper_bound"),
)
METHODS = (BASELINE, *(candidate for candidate, _count, _role in SQUARED_SPECS))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_matrix_target(values: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values)
    labels = np.asarray(target)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must be a nonempty two-dimensional matrix")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("values must be numeric")
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("values must be finite")
    if labels.ndim != 1 or len(labels) != len(matrix):
        raise ValueError("target must be a one-dimensional aligned vector")
    if len(dict.fromkeys(labels.tolist())) < 2:
        raise ValueError("at least two classes are required")
    return matrix, labels


def _array_hash(name: str, value: np.ndarray, digest: Any) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


@dataclass(frozen=True)
class FittedSquaredFeatureLift:
    """Immutable train-fold-fitted map from x to [x, selected centered x^2]."""

    input_feature_count: int
    selected_indices: np.ndarray
    square_means: np.ndarray
    square_scales: np.ndarray
    selected_relevance: np.ndarray
    base_feature_rms: float
    state_sha256: str

    def transform(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.shape[1] != self.input_feature_count:
            raise ValueError("values have the wrong shape for this squared lift")
        if not np.issubdtype(matrix.dtype, np.number):
            raise TypeError("values must be numeric")
        matrix = np.asarray(matrix, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("values must be finite")
        normalized = food101._row_l2(matrix)
        selected = normalized[:, self.selected_indices]
        square_features = (
            np.square(selected, dtype=np.float32) - self.square_means[None, :]
        ) * self.square_scales[None, :]
        lifted = np.concatenate((normalized, square_features), axis=1)
        return np.asarray(food101._row_l2(lifted), dtype=np.float32)

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "input_feature_count": int(self.input_feature_count),
            "selected_square_feature_count": int(len(self.selected_indices)),
            "output_feature_count": int(
                self.input_feature_count + len(self.selected_indices)
            ),
            "selected_indices": self.selected_indices.tolist(),
            "selected_relevance_min": float(np.min(self.selected_relevance)),
            "selected_relevance_median": float(
                np.median(self.selected_relevance)
            ),
            "selected_relevance_max": float(np.max(self.selected_relevance)),
            "base_feature_rms": float(self.base_feature_rms),
            "state_sha256": self.state_sha256,
            "selection_uses_training_labels": True,
            "transform_uses_labels": False,
            "final_row_l2_normalization": True,
        }


def fit_squared_feature_lift(
    values: Any,
    target: Any,
    *,
    max_square_features: int | None,
) -> FittedSquaredFeatureLift:
    """Select squared coordinates by a train-only class-separation ratio."""

    matrix, labels = _validate_matrix_target(values, target)
    if max_square_features is not None and (
        type(max_square_features) is not int or max_square_features < 1
    ):
        raise ValueError("max_square_features must be None or a positive int")
    normalized = food101._row_l2(matrix)
    squared = np.square(normalized, dtype=np.float32).astype(np.float64)
    square_mean = np.mean(squared, axis=0, dtype=np.float64)
    within_sum = np.zeros(matrix.shape[1], dtype=np.float64)
    between_sum = np.zeros(matrix.shape[1], dtype=np.float64)
    for label in dict.fromkeys(labels.tolist()):
        rows = squared[labels == label]
        class_mean = np.mean(rows, axis=0, dtype=np.float64)
        within_sum += np.sum(np.square(rows - class_mean), axis=0, dtype=np.float64)
        between_sum += len(rows) * np.square(class_mean - square_mean)
    denominator = float(len(matrix))
    within = within_sum / denominator
    between = between_sum / denominator
    epsilon = max(float(np.mean(within + between)) * 1.0e-9, 1.0e-18)
    relevance = between / (between + within + epsilon)
    count = (
        matrix.shape[1]
        if max_square_features is None
        else min(int(max_square_features), matrix.shape[1])
    )
    feature_ids = np.arange(matrix.shape[1], dtype=np.int64)
    order = np.lexsort((feature_ids, -relevance))
    selected = np.asarray(order[:count], dtype=np.int64)
    variance = between + within
    square_std = np.sqrt(np.maximum(variance[selected], epsilon))
    base_variance = np.var(normalized.astype(np.float64), axis=0, dtype=np.float64)
    base_feature_rms = float(np.sqrt(np.mean(base_variance)))
    scales = np.asarray(base_feature_rms / square_std, dtype=np.float32)
    means = np.asarray(square_mean[selected], dtype=np.float32)
    selected_relevance = np.asarray(relevance[selected], dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "algorithm": "train_only_selected_centered_square_features_v1",
                "input_feature_count": int(matrix.shape[1]),
                "max_square_features": max_square_features,
                "base_feature_rms": base_feature_rms,
                "selection_score": "between/(between+within+epsilon)",
                "square_scale": "base_feature_rms/sqrt(total_square_variance)",
                "final_row_l2_normalization": True,
            }
        ).encode("utf-8")
    )
    _array_hash("selected_indices", selected, digest)
    _array_hash("square_means", means, digest)
    _array_hash("square_scales", scales, digest)
    _array_hash("selected_relevance", selected_relevance, digest)
    for array in (selected, means, scales, selected_relevance):
        array.setflags(write=False)
    return FittedSquaredFeatureLift(
        input_feature_count=int(matrix.shape[1]),
        selected_indices=selected,
        square_means=means,
        square_scales=scales,
        selected_relevance=selected_relevance,
        base_feature_rms=base_feature_rms,
        state_sha256=digest.hexdigest(),
    )


def _crossfit_baseline(values: Any, target: Any, *, seed: int) -> dict[str, Any]:
    matrix, labels = _validate_matrix_target(values, target)
    normalized = food101._row_l2(matrix)
    folds = food101._stratified_folds(labels, n_splits=5, seed=int(seed))
    started = time.perf_counter()
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        k_per_class = fused_food_screen._k_per_class(labels, train)
        fit_started = time.perf_counter()
        selector = fused_food_screen._build_selector(
            "FUSED3", k_per_class, fold_seed
        ).fit(normalized[train], labels[train])
        fit_wall = time.perf_counter() - fit_started
        score_started = time.perf_counter()
        score = selector.score_fixed(normalized[holdout], labels[holdout])
        score_wall = time.perf_counter() - score_started
        fold_rows.append(
            {
                "fold": int(fold),
                "fold_seed": fold_seed,
                "score": float(score),
                "fit_wall_seconds": float(fit_wall),
                "score_wall_seconds": float(score_wall),
                "input_feature_count": int(matrix.shape[1]),
                "output_feature_count": int(matrix.shape[1]),
                "selected_square_feature_count": 0,
                "lift_state_sha256": None,
            }
        )
    return {
        "score": float(np.mean([row["score"] for row in fold_rows])),
        "total_wall_seconds": float(time.perf_counter() - started),
        "folds": fold_rows,
    }


def crossfit_squared_fused3(
    values: Any,
    target: Any,
    *,
    seed: int,
    max_square_features: int | None,
) -> dict[str, Any]:
    """Fit each squared lift and FUSED3 strictly inside its training fold."""

    matrix, labels = _validate_matrix_target(values, target)
    folds = food101._stratified_folds(labels, n_splits=5, seed=int(seed))
    started = time.perf_counter()
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        lift_started = time.perf_counter()
        lift = fit_squared_feature_lift(
            matrix[train], labels[train], max_square_features=max_square_features
        )
        transformed_train = lift.transform(matrix[train])
        transformed_holdout = lift.transform(matrix[holdout])
        lift_wall = time.perf_counter() - lift_started
        k_per_class = fused_food_screen._k_per_class(labels, train)
        fit_started = time.perf_counter()
        selector = fused_food_screen._build_selector(
            "FUSED3", k_per_class, fold_seed
        ).fit(transformed_train, labels[train])
        fit_wall = time.perf_counter() - fit_started
        score_started = time.perf_counter()
        score = selector.score_fixed(transformed_holdout, labels[holdout])
        score_wall = time.perf_counter() - score_started
        fold_rows.append(
            {
                "fold": int(fold),
                "fold_seed": fold_seed,
                "score": float(score),
                "lift_wall_seconds": float(lift_wall),
                "fit_wall_seconds": float(fit_wall),
                "score_wall_seconds": float(score_wall),
                **lift.diagnostics,
            }
        )
    return {
        "score": float(np.mean([row["score"] for row in fold_rows])),
        "total_wall_seconds": float(time.perf_counter() - started),
        "folds": fold_rows,
    }


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("Spearman inputs must be aligned vectors")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    value = float(
        np.corrcoef(rankdata(x, method="average"), rankdata(y, method="average"))[0, 1]
    )
    return value if np.isfinite(value) else None


def _lookups(
    selector_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int, int], float],
    dict[tuple[str, int, int, str], float],
]:
    frozen: dict[tuple[str, int, int], float] = {}
    reference: dict[tuple[str, int, int, str], float] = {}
    for row in selector_rows:
        if row["dataset_id"] == DATASET_ID and row["candidate_id"] == BASELINE:
            key = (str(row["backbone"]), int(row["replicate_seed"]), int(row["budget"]))
            frozen[key] = float(row["score"])
    for row in reference_rows:
        if row["dataset_id"] == DATASET_ID:
            key = (
                str(row["backbone"]),
                int(row["replicate_seed"]),
                int(row["budget"]),
                str(row["head"]),
            )
            reference[key] = float(row["test_accuracy"])
    if len(frozen) != 100 or len(reference) != 400:
        raise RuntimeError("frozen Aircraft tables are incomplete")
    return frozen, reference


def _execution_order(backbone: str, seed: int, budget: int) -> tuple[str, ...]:
    digest = hashlib.sha256(f"{backbone}:{seed}:{budget}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little") % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def analyze_squared_feature_diagnostic(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
) -> dict[str, Any]:
    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    frozen_scores, reference = _lookups(selectors, references)
    root = Path(registry_root)
    registry = json.loads((root / "audited_registry.json").read_text(encoding="utf-8"))
    aircraft = next(
        row for row in registry["datasets"] if row["dataset_id"] == DATASET_ID
    )
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for backbone in statistics.BACKBONES:
        for replicate, seed in enumerate(statistics.REPLICATE_SEEDS):
            for budget in statistics.BUDGETS:
                panel = datasets.load_panel(
                    aircraft, backbone, int(seed), int(budget), registry_root=root
                )
                identity = (backbone, int(seed), int(budget))
                heads = {
                    head: reference[identity + (head,)] for head in statistics.HEADS
                }
                envelope = max(heads.values())
                order = _execution_order(backbone, int(seed), int(budget))
                results: dict[str, dict[str, Any]] = {}
                for candidate in order:
                    if candidate == BASELINE:
                        result = _crossfit_baseline(
                            panel["training_values"], panel["training_labels"], seed=int(seed)
                        )
                    else:
                        count = next(
                            count
                            for name, count, _role in SQUARED_SPECS
                            if name == candidate
                        )
                        result = crossfit_squared_fused3(
                            panel["training_values"],
                            panel["training_labels"],
                            seed=int(seed),
                            max_square_features=count,
                        )
                    results[candidate] = result
                if abs(results[BASELINE]["score"] - frozen_scores[identity]) > 1.0e-12:
                    raise RuntimeError("recomputed FUSED3 differs from frozen score")
                for position, candidate in enumerate(order):
                    result = results[candidate]
                    rows.append(
                        {
                            "dataset_id": DATASET_ID,
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "candidate_id": candidate,
                            "execution_position": int(position),
                            "execution_order": list(order),
                            "score": float(result["score"]),
                            "frozen_baseline_score": float(frozen_scores[identity]),
                            "best_head_envelope_accuracy": float(envelope),
                            "best_head": statistics.HEADS[
                                int(np.argmax([heads[head] for head in statistics.HEADS]))
                            ],
                            "total_wall_seconds": float(result["total_wall_seconds"]),
                        }
                    )
                    for fold in result["folds"]:
                        fold_rows.append(
                            {
                                "backbone": backbone,
                                "replicate": int(replicate),
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                "candidate_id": candidate,
                                **fold,
                            }
                        )
    if len(rows) != 400 or len(fold_rows) != 2000:
        raise RuntimeError("squared-feature diagnostic grid is incomplete")

    selection_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for seed in statistics.REPLICATE_SEEDS:
            panel = [
                row
                for row in rows
                if row["budget"] == budget and row["replicate_seed"] == seed
            ]
            targets = [
                next(
                    float(row["best_head_envelope_accuracy"])
                    for row in panel
                    if row["backbone"] == backbone
                )
                for backbone in statistics.BACKBONES
            ]
            oracle = max(targets)
            for method in METHODS:
                method_rows = [row for row in panel if row["candidate_id"] == method]
                scores = [
                    next(float(row["score"]) for row in method_rows if row["backbone"] == backbone)
                    for backbone in statistics.BACKBONES
                ]
                maximum = max(scores)
                selected = [
                    row
                    for row in method_rows
                    if abs(float(row["score"]) - maximum)
                    <= statistics.SELECTION_TIE_ATOL
                ]
                selected_accuracy = float(
                    np.mean([row["best_head_envelope_accuracy"] for row in selected])
                )
                selection_rows.append(
                    {
                        "budget": int(budget),
                        "replicate_seed": int(seed),
                        "candidate_id": method,
                        "selected_backbones": sorted(row["backbone"] for row in selected),
                        "selected_envelope_accuracy": selected_accuracy,
                        "oracle_envelope_accuracy": float(oracle),
                        "regret_pp": float(100.0 * (oracle - selected_accuracy)),
                    }
                )
                rank_rows.append(
                    {
                        "budget": int(budget),
                        "replicate_seed": int(seed),
                        "candidate_id": method,
                        "spearman_with_best_head_envelope": _spearman(scores, targets),
                    }
                )

    aggregate: dict[str, dict[str, Any]] = {}
    selection_counts: dict[str, dict[str, int]] = {}
    for method in METHODS:
        selected = [row for row in selection_rows if row["candidate_id"] == method]
        ranked = [row for row in rank_rows if row["candidate_id"] == method]
        method_rows = [row for row in rows if row["candidate_id"] == method]
        counts: Counter[str] = Counter(
            backbone for row in selected for backbone in row["selected_backbones"]
        )
        selection_counts[method] = dict(sorted(counts.items()))
        aggregate[method] = {
            "mean_regret_pp": float(np.mean([row["regret_pp"] for row in selected])),
            "mean_spearman": float(
                np.mean([row["spearman_with_best_head_envelope"] for row in ranked])
            ),
            "median_panel_wall_seconds": float(
                np.median([row["total_wall_seconds"] for row in method_rows])
            ),
        }
    baseline_median = aggregate[BASELINE]["median_panel_wall_seconds"]
    for method in METHODS:
        aggregate[method]["median_wall_ratio_to_recomputed_fused3"] = float(
            aggregate[method]["median_panel_wall_seconds"] / baseline_median
        )
    return {
        "study": STUDY,
        "stage": STAGE,
        "diagnostic_status": STATUS,
        "dataset_id": DATASET_ID,
        "methods": list(METHODS),
        "candidate_contracts": {
            BASELINE: {"role": "exact_frozen_control", "square_feature_count": 0},
            **{
                candidate: {
                    "role": role,
                    "square_feature_count": count if count is not None else "all",
                }
                for candidate, count, role in SQUARED_SPECS
            },
        },
        "lift_contract": {
            "input": "frozen row-L2-normalized embedding",
            "selection_scope": "training fold only",
            "selection_score": "between_class_square_variance/(between+within)",
            "square_centering": "training-fold global mean",
            "square_scaling": "unit variance then energy match to mean raw-coordinate RMS",
            "output": "row-L2([x, selected centered/scaled x^2])",
            "selector_after_lift": "exact FUSED3 implementation",
            "oi_event_rule_changed": False,
        },
        "rows": rows,
        "fold_rows": fold_rows,
        "selection_rows": selection_rows,
        "rank_rows": rank_rows,
        "selection_counts": selection_counts,
        "aggregate": aggregate,
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "timing_status": "descriptive_same_process_hash_cyclic_order_no_gate",
        "input_hashes": dict(verified["input_hashes"]),
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize empty CSV")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: _canonical_json(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    return output.getvalue().encode("utf-8")


def _report(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Aircraft squared-feature FUSED3 diagnostic",
        "",
        "Status: **post-outcome descriptive diagnostic only**. The frozen confirmation decision is unchanged.",
        "",
        "## Best-head-envelope ranking",
        "",
        "| Method | Mean regret (pp) | Mean Spearman | Median panel time | Time / FUSED3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate[method]
        lines.append(
            f"| {method} | {row['mean_regret_pp']:.3f} | {row['mean_spearman']:.3f} | "
            f"{row['median_panel_wall_seconds']:.3f}s | {row['median_wall_ratio_to_recomputed_fused3']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## Selection counts across 10 seed/budget panels",
            "",
        ]
    )
    for method in METHODS:
        lines.append(f"- {method}: `{_canonical_json(summary['selection_counts'][method])}`")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "SQ32/SQ64 test whether a compact axis-aligned quadratic lift supplies the missing ranking signal. SQALL is only a mechanism upper bound. This post-outcome surface cannot authorize promotion, tuning, or reselection.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_squared_feature_diagnostic(full_artifact, registry_root)
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "panel_rows.csv": _csv_bytes(summary["rows"]),
        "fold_rows.csv": _csv_bytes(summary["fold_rows"]),
        "selection_rows.csv": _csv_bytes(summary["selection_rows"]),
        "rank_rows.csv": _csv_bytes(summary["rank_rows"]),
        "report.md": _report(summary).encode("utf-8"),
    }
    manifest = {
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "files": {
            name: {
                "sha256": _sha256_bytes(content),
                "size_bytes": len(content),
            }
            for name, content in payloads.items()
        },
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
    }
    payloads["diagnostic_manifest.json"] = (
        _canonical_json(manifest) + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for name, content in payloads.items():
            (staging / name).write_bytes(content)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for name, descriptor in manifest["files"].items():
        if _sha256(destination / name) != descriptor["sha256"]:
            raise RuntimeError("written squared-feature artifact hash mismatch")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_bundle(args.input, args.registry_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
