"""Aircraft diagnostic for class-subspace and covariance-aware heads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics
from experiments.fused3_envelope_reanalysis.knn_tuning import predict_k_grid


STUDY = "fused3_confirmation_v2_aircraft_structure_head_diagnostic"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
DATASET_ID = "torchvision_fgvc_aircraft"
FOCAL_BACKBONES = ("dinov2-small", "openclip-vit-b-32")
METHODS = ("NEAREST_CLASS_SUBSPACE", "LDA_KNN", "DIAGONAL_GAUSSIAN")
INNER_FOLDS = 3
SUBSPACE_RANK_GRID = (0, 1, 2, 4, 8, 16)
LDA_DIM_GRID = (16, 32, 64, 96)
K_GRID = (1, 3, 7, 15, 31)
VAR_SMOOTHING_GRID = (1.0e-11, 1.0e-9, 1.0e-7, 1.0e-5, 1.0e-3, 1.0e-1)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(
    training_values: Any, training_labels: Any, query_values: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(training_values, dtype=np.float32)
    labels = np.asarray(training_labels)
    query = np.asarray(query_values, dtype=np.float32)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]:
        raise ValueError("training/query matrices must be aligned and 2-D")
    if labels.ndim != 1 or len(labels) != len(train):
        raise ValueError("training labels must align with training rows")
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError("head inputs must be finite")
    return train, labels, query


def _folds(values: np.ndarray, labels: np.ndarray, seed: int):
    _classes, encoded = np.unique(labels, return_inverse=True)
    return tuple(
        StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=int(seed)
        ).split(values, encoded)
    )


def _select(scores: Mapping[Any, int], order: Sequence[Any]) -> Any:
    best = max(scores.values())
    return next(item for item in order if scores[item] == best)


def nearest_subspace_predictions(
    training_values: Any,
    training_labels: Any,
    query_values: Any,
    *,
    ranks: Sequence[int] = SUBSPACE_RANK_GRID,
) -> dict[int, np.ndarray]:
    """Predict by distance to a class-specific affine PCA subspace."""

    train, labels, query = _arrays(
        training_values, training_labels, query_values
    )
    train = normalize(train, copy=True)
    query = normalize(query, copy=True)
    requested = tuple(int(rank) for rank in ranks)
    if not requested or requested != tuple(sorted(set(requested))) or requested[0] < 0:
        raise ValueError("ranks must be sorted, unique, and nonnegative")
    classes = np.unique(labels)
    maximum = requested[-1]
    distances = {
        rank: np.empty((len(query), len(classes)), dtype=np.float64)
        for rank in requested
    }
    for class_position, label in enumerate(classes):
        selected = train[labels == label]
        mean = np.mean(selected, axis=0, dtype=np.float64)
        centered = np.asarray(selected - mean, dtype=np.float32)
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        usable = min(maximum, len(selected) - 1, train.shape[1])
        basis = np.asarray(vt[:usable].T, dtype=np.float32)
        delta = np.asarray(query - mean, dtype=np.float32)
        base = np.sum(delta * delta, axis=1, dtype=np.float64)
        if usable:
            coordinates = np.asarray(delta @ basis, dtype=np.float32)
            cumulative = np.cumsum(
                coordinates * coordinates, axis=1, dtype=np.float64
            )
        for rank in requested:
            used = min(rank, usable)
            explained = 0.0 if used == 0 else cumulative[:, used - 1]
            distances[rank][:, class_position] = np.maximum(base - explained, 0.0)
    return {
        rank: np.asarray(classes[np.argmin(matrix, axis=1)])
        for rank, matrix in distances.items()
    }


def tune_nearest_subspace(
    values: Any, labels: Any, *, seed: int
) -> tuple[int, dict[int, float]]:
    matrix = np.asarray(values, dtype=np.float32)
    target = np.asarray(labels)
    correct = {rank: 0 for rank in SUBSPACE_RANK_GRID}
    total = 0
    for train_ids, validation_ids in _folds(matrix, target, seed):
        predictions = nearest_subspace_predictions(
            matrix[train_ids],
            target[train_ids],
            matrix[validation_ids],
            ranks=SUBSPACE_RANK_GRID,
        )
        for rank, predicted in predictions.items():
            correct[rank] += int(
                np.count_nonzero(predicted == target[validation_ids])
            )
        total += len(validation_ids)
    selected = int(_select(correct, SUBSPACE_RANK_GRID))
    return selected, {rank: float(value / total) for rank, value in correct.items()}


def _lda_dims(feature_count: int, class_count: int) -> tuple[int, ...]:
    maximum = min(feature_count, class_count - 1)
    values = tuple(dim for dim in LDA_DIM_GRID if dim <= maximum)
    if not values or values[-1] != maximum:
        values = (*values, maximum)
    return tuple(sorted(set(values)))


def tune_lda_knn(
    values: Any, labels: Any, *, seed: int
) -> tuple[tuple[int, int], dict[tuple[int, int], float]]:
    matrix = normalize(np.asarray(values, dtype=np.float32), copy=True)
    target = np.asarray(labels)
    dims = _lda_dims(matrix.shape[1], len(np.unique(target)))
    configs = tuple((dim, k) for dim in dims for k in K_GRID)
    correct = {config: 0 for config in configs}
    total = 0
    for train_ids, validation_ids in _folds(matrix, target, seed):
        estimator = LinearDiscriminantAnalysis(
            solver="svd", n_components=max(dims)
        )
        projected_train = np.asarray(
            estimator.fit_transform(matrix[train_ids], target[train_ids]),
            dtype=np.float32,
        )
        projected_validation = np.asarray(
            estimator.transform(matrix[validation_ids]), dtype=np.float32
        )
        for dim in dims:
            predictions = predict_k_grid(
                projected_train[:, :dim],
                target[train_ids],
                projected_validation[:, :dim],
                k_grid=K_GRID,
            )
            for k, predicted in predictions.items():
                correct[(dim, k)] += int(
                    np.count_nonzero(predicted == target[validation_ids])
                )
        total += len(validation_ids)
    selected = tuple(_select(correct, configs))
    return selected, {config: float(value / total) for config, value in correct.items()}


def _fit_lda_knn_predict(
    train: np.ndarray,
    labels: np.ndarray,
    query: np.ndarray,
    *,
    dim: int,
    k: int,
) -> np.ndarray:
    normalized_train = normalize(train, copy=True)
    normalized_query = normalize(query, copy=True)
    estimator = LinearDiscriminantAnalysis(solver="svd", n_components=dim)
    projected_train = np.asarray(
        estimator.fit_transform(normalized_train, labels), dtype=np.float32
    )
    projected_query = np.asarray(estimator.transform(normalized_query), dtype=np.float32)
    return predict_k_grid(
        projected_train, labels, projected_query, k_grid=(k,)
    )[k]


def tune_diagonal_gaussian(
    values: Any, labels: Any, *, seed: int
) -> tuple[float, dict[float, float]]:
    matrix = normalize(np.asarray(values, dtype=np.float32), copy=True)
    target = np.asarray(labels)
    correct = {value: 0 for value in VAR_SMOOTHING_GRID}
    total = 0
    for train_ids, validation_ids in _folds(matrix, target, seed):
        for smoothing in VAR_SMOOTHING_GRID:
            estimator = GaussianNB(var_smoothing=smoothing).fit(
                matrix[train_ids], target[train_ids]
            )
            predicted = estimator.predict(matrix[validation_ids])
            correct[smoothing] += int(
                np.count_nonzero(predicted == target[validation_ids])
            )
        total += len(validation_ids)
    selected = float(_select(correct, VAR_SMOOTHING_GRID))
    return selected, {
        smoothing: float(value / total) for smoothing, value in correct.items()
    }


def fit_evaluate_heads(
    training_values: Any,
    training_labels: Any,
    evaluation_values: Any,
    evaluation_labels: Any,
    *,
    seed: int,
    methods: Sequence[str] = METHODS,
) -> list[dict[str, Any]]:
    train, labels, query = _arrays(
        training_values, training_labels, evaluation_values
    )
    query_labels = np.asarray(evaluation_labels)
    if query_labels.ndim != 1 or len(query_labels) != len(query):
        raise ValueError("evaluation labels must align with evaluation rows")
    requested = tuple(methods)
    if not requested or len(set(requested)) != len(requested) or not set(requested).issubset(METHODS):
        raise ValueError(f"methods must be unique values from {METHODS!r}")
    rows: list[dict[str, Any]] = []
    for method in requested:
        started = time.perf_counter()
        if method == "NEAREST_CLASS_SUBSPACE":
            rank, inner = tune_nearest_subspace(train, labels, seed=seed)
            predicted = nearest_subspace_predictions(
                train, labels, query, ranks=(rank,)
            )[rank]
            config = {"rank": rank}
            inner_accuracy = inner[rank]
        elif method == "LDA_KNN":
            (dim, k), inner = tune_lda_knn(train, labels, seed=seed)
            predicted = _fit_lda_knn_predict(
                train, labels, query, dim=dim, k=k
            )
            config = {"dimension": dim, "k": k}
            inner_accuracy = inner[(dim, k)]
        elif method == "DIAGONAL_GAUSSIAN":
            smoothing, inner = tune_diagonal_gaussian(train, labels, seed=seed)
            normalized_train = normalize(train, copy=True)
            predicted = GaussianNB(var_smoothing=smoothing).fit(
                normalized_train, labels
            ).predict(normalize(query, copy=True))
            config = {"var_smoothing": smoothing}
            inner_accuracy = inner[smoothing]
        else:  # pragma: no cover
            raise RuntimeError("unreachable structure head")
        rows.append(
            {
                "method": method,
                "selected_config": config,
                "inner_cv_accuracy": float(inner_accuracy),
                "test_accuracy": float(np.mean(predicted == query_labels)),
                "wall_seconds": float(time.perf_counter() - started),
                "training_row_count": int(len(train)),
                "evaluation_row_count": int(len(query)),
            }
        )
    return rows


def _reference_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, int, str], float]:
    output = {
        (
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["head"]),
        ): float(row["test_accuracy"])
        for row in rows
        if row["dataset_id"] == DATASET_ID
    }
    if len(output) != 400:
        raise ValueError("Aircraft structure diagnostic requires 400 reference rows")
    return output


def analyze_structure_heads(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    *,
    methods: Sequence[str] = METHODS,
) -> dict[str, Any]:
    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    _selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    reference = _reference_lookup(references)
    root = Path(registry_root)
    registry = json.loads((root / "audited_registry.json").read_text(encoding="utf-8"))
    aircraft = next(
        row for row in registry["datasets"] if row["dataset_id"] == DATASET_ID
    )
    panel_rows: list[dict[str, Any]] = []
    for backbone in FOCAL_BACKBONES:
        for replicate, seed in enumerate(statistics.REPLICATE_SEEDS):
            for budget in statistics.BUDGETS:
                panel = datasets.load_panel(
                    aircraft, backbone, int(seed), int(budget), registry_root=root
                )
                for row in fit_evaluate_heads(
                    panel["training_values"],
                    panel["training_labels"],
                    panel["evaluation_values"],
                    panel["evaluation_labels"],
                    seed=int(seed),
                    methods=methods,
                ):
                    panel_rows.append(
                        {
                            "dataset_id": DATASET_ID,
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            **row,
                            "frozen_quadratic_accuracy": reference[(backbone, int(seed), int(budget), "quadratic")],
                        }
                    )
    expected = len(FOCAL_BACKBONES) * len(statistics.REPLICATE_SEEDS) * len(statistics.BUDGETS) * len(methods)
    if len(panel_rows) != expected:
        raise RuntimeError("Aircraft structure-head grid is incomplete")

    aggregate_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for backbone in FOCAL_BACKBONES:
            for method in methods:
                rows = [
                    row for row in panel_rows
                    if row["budget"] == budget and row["backbone"] == backbone and row["method"] == method
                ]
                aggregate_rows.append(
                    {
                        "budget": int(budget),
                        "backbone": backbone,
                        "method": method,
                        "mean_test_accuracy": float(np.mean([row["test_accuracy"] for row in rows])),
                        "mean_inner_cv_accuracy": float(np.mean([row["inner_cv_accuracy"] for row in rows])),
                        "mean_quadratic_accuracy": float(np.mean([row["frozen_quadratic_accuracy"] for row in rows])),
                        "new_minus_own_quadratic_pp": float(100.0 * np.mean([row["test_accuracy"] - row["frozen_quadratic_accuracy"] for row in rows])),
                        "median_wall_seconds": float(np.median([row["wall_seconds"] for row in rows])),
                        "selected_config_counts": _canonical_json(Counter(_canonical_json(row["selected_config"]) for row in rows)),
                    }
                )

    focal_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        dino_quadratic = {
            int(seed): reference[("dinov2-small", int(seed), int(budget), "quadratic")]
            for seed in statistics.REPLICATE_SEEDS
        }
        for method in methods:
            rows = [
                row for row in panel_rows
                if row["budget"] == budget and row["backbone"] == "openclip-vit-b-32" and row["method"] == method
            ]
            deltas = [row["test_accuracy"] - dino_quadratic[row["replicate_seed"]] for row in rows]
            focal_rows.append(
                {
                    "budget": int(budget),
                    "method": method,
                    "openclip_mean_accuracy": float(np.mean([row["test_accuracy"] for row in rows])),
                    "dino_frozen_quadratic_mean_accuracy": float(np.mean(list(dino_quadratic.values()))),
                    "openclip_minus_dino_quadratic_pp": float(100.0 * np.mean(deltas)),
                    "openclip_better_panel_count": int(sum(delta > 0 for delta in deltas)),
                }
            )
    best = max(focal_rows, key=lambda row: row["openclip_minus_dino_quadratic_pp"])
    return {
        "study": STUDY,
        "stage": STAGE,
        "diagnostic_status": STATUS,
        "dataset_id": DATASET_ID,
        "focal_backbones": list(FOCAL_BACKBONES),
        "methods": list(methods),
        "inner_cv_folds": INNER_FOLDS,
        "selection_protocol": "all parameters selected by three-fold stratified CV on training rows only; evaluation labels used once",
        "panel_rows": panel_rows,
        "aggregate_rows": aggregate_rows,
        "openclip_vs_dino_quadratic_rows": focal_rows,
        "best_openclip_capacity_result": dict(best),
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "input_hashes": dict(verified["input_hashes"]),
        "input_identity": {
            key: verified["raw"][key]
            for key in (
                "protocol_sha256", "code_identity_sha256", "run_identity_sha256",
                "audited_registry_sha256", "lineage_lock_sha256",
            )
        },
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Aircraft structure-aware head capacity diagnostic", "",
        "Status: **post-outcome descriptive diagnostic only**. The frozen confirmation decision is unchanged.", "",
        "All hyperparameters are selected inside the training cohort. Evaluation labels are used only after selection.", "",
        "## Accuracy", "",
        "| Budget | Backbone | Head | Accuracy | Own quadratic | Delta | Median wall |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate_rows"]:
        lines.append(
            f"| {row['budget']} | {row['backbone']} | {row['method']} | {100*row['mean_test_accuracy']:.3f}% | {100*row['mean_quadratic_accuracy']:.3f}% | {row['new_minus_own_quadratic_pp']:+.3f} pp | {row['median_wall_seconds']:.2f}s |"
        )
    lines.extend(["", "## OpenCLIP versus DINO quadratic", "", "| Budget | OpenCLIP head | OpenCLIP accuracy | DINO quadratic | Delta | Panels better |", "|---:|---|---:|---:|---:|---:|"])
    for row in summary["openclip_vs_dino_quadratic_rows"]:
        lines.append(
            f"| {row['budget']} | {row['method']} | {100*row['openclip_mean_accuracy']:.3f}% | {100*row['dino_frozen_quadratic_mean_accuracy']:.3f}% | {row['openclip_minus_dino_quadratic_pp']:+.3f} pp | {row['openclip_better_panel_count']}/5 |"
        )
    best = summary["best_openclip_capacity_result"]
    lines.extend(["", "## Capacity conclusion", "", f"Best tested OpenCLIP result: `{best['method']}` at budget {best['budget']}; OpenCLIP minus DINO quadratic = **{best['openclip_minus_dino_quadratic_pp']:+.3f} pp**.", "", "No promotion, reselection, or confirmation reinterpretation is made.", ""])
    return "\n".join(lines)


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
    *, methods: Sequence[str] = METHODS,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_structure_heads(full_artifact, registry_root, methods=methods)
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode(),
        "panel_rows.csv": _csv_bytes(summary["panel_rows"]),
        "aggregate_rows.csv": _csv_bytes(summary["aggregate_rows"]),
        "openclip_vs_dino_quadratic.csv": _csv_bytes(summary["openclip_vs_dino_quadratic_rows"]),
        "report.md": _report(summary).encode(),
    }
    manifest = {
        "study": STUDY, "stage": STAGE, "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)} for name, content in payloads.items()},
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
    }
    payloads["diagnostic_manifest.json"] = (_canonical_json(manifest) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name, content in payloads.items():
            (staging / name).write_bytes(content)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for name, descriptor in manifest["files"].items():
        if _sha256(destination / name) != descriptor["sha256"]:
            raise RuntimeError("written structure diagnostic hash mismatch")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args(argv)
    write_bundle(args.input, args.registry_root, args.output, methods=args.methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
