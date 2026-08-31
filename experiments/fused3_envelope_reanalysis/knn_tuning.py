"""Post-outcome Aircraft diagnostic for the frozen fixed-k KNN reference head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics


STUDY = "fused3_confirmation_v2_aircraft_knn_tuning_diagnostic"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
DATASET_ID = "torchvision_fgvc_aircraft"
K_GRID = (1, 3, 5, 7, 9, 11, 15, 21, 31, 47, 63)
INNER_FOLDS = 3
CURRENT_K = 15


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("Spearman inputs must be aligned vectors")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    value = float(
        np.corrcoef(
            rankdata(x, method="average"), rankdata(y, method="average")
        )[0, 1]
    )
    return value if np.isfinite(value) else None


def _validate_k_grid(k_grid: Sequence[int], train_count: int) -> tuple[int, ...]:
    values = tuple(k_grid)
    if not values or any(type(k) is not int or k < 1 for k in values):
        raise ValueError("k_grid must contain positive integers")
    if values != tuple(sorted(set(values))):
        raise ValueError("k_grid must be strictly increasing and unique")
    if values[-1] > train_count:
        raise ValueError("k_grid exceeds the available training rows")
    return values


def _distance_weights(distances: np.ndarray) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    with np.errstate(divide="ignore"):
        weights = 1.0 / values
    zero_rows = np.isinf(weights).any(axis=1)
    if np.any(zero_rows):
        weights[zero_rows] = np.isinf(weights[zero_rows]).astype(np.float64)
    return weights


def predict_k_grid(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    query_values: np.ndarray,
    *,
    k_grid: Sequence[int] = K_GRID,
) -> dict[int, np.ndarray]:
    """Predict every K using one exact cosine-neighbor search."""

    train = normalize(np.asarray(train_values, dtype=np.float32), copy=True)
    query = normalize(np.asarray(query_values, dtype=np.float32), copy=True)
    labels = np.asarray(train_labels)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]:
        raise ValueError("train/query matrices must be aligned and two-dimensional")
    if labels.ndim != 1 or len(labels) != len(train):
        raise ValueError("training labels must align with training rows")
    ks = _validate_k_grid(k_grid, len(train))
    classes, encoded = np.unique(labels, return_inverse=True)
    neighbors = NearestNeighbors(
        n_neighbors=ks[-1], metric="cosine", n_jobs=1
    ).fit(train)
    distances, indices = neighbors.kneighbors(query, return_distance=True)
    neighbor_classes = encoded[indices]
    weights = _distance_weights(distances)
    votes = np.zeros((len(query), len(classes)), dtype=np.float64)
    row_ids = np.arange(len(query), dtype=np.intp)
    output: dict[int, np.ndarray] = {}
    wanted = set(ks)
    for position in range(ks[-1]):
        np.add.at(
            votes,
            (row_ids, neighbor_classes[:, position]),
            weights[:, position],
        )
        k = position + 1
        if k in wanted:
            output[k] = np.asarray(classes[np.argmax(votes, axis=1)])
    return output


def tune_k_inner_cv(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    k_grid: Sequence[int] = K_GRID,
) -> tuple[int, dict[int, float]]:
    """Choose K only from the training cohort using deterministic inner CV."""

    X = np.asarray(values, dtype=np.float32)
    y = np.asarray(labels)
    folds = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=int(seed)
    )
    correct = {k: 0 for k in k_grid}
    total = 0
    for train_ids, validation_ids in folds.split(X, y):
        predictions = predict_k_grid(
            X[train_ids], y[train_ids], X[validation_ids], k_grid=k_grid
        )
        for k, predicted in predictions.items():
            correct[k] += int(np.count_nonzero(predicted == y[validation_ids]))
        total += len(validation_ids)
    accuracies = {k: float(correct[k] / total) for k in k_grid}
    best = max(accuracies.values())
    selected = min(
        k for k, accuracy in accuracies.items() if abs(accuracy - best) <= 1.0e-12
    )
    return selected, accuracies


def _reference_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str], float]:
    lookup: dict[tuple[str, int, int, str], float] = {}
    for row in rows:
        if row["dataset_id"] != DATASET_ID:
            continue
        key = (
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["head"]),
        )
        if key in lookup:
            raise ValueError("duplicate Aircraft reference row")
        lookup[key] = float(row["test_accuracy"])
    if len(lookup) != 400:
        raise ValueError("Aircraft KNN tuning requires the exact 400 reference rows")
    return lookup


def analyze_knn_tuning(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    *,
    k_grid: Sequence[int] = K_GRID,
) -> dict[str, Any]:
    """Run a leakage-free inner-CV K choice plus a labeled oracle upper bound."""

    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    _selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    reference = _reference_lookup(references)
    registry_path = Path(registry_root) / "audited_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    aircraft_record = next(
        row for row in registry["datasets"] if row["dataset_id"] == DATASET_ID
    )
    ks = tuple(k_grid)
    panel_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    for backbone in statistics.BACKBONES:
        for replicate, seed in enumerate(statistics.REPLICATE_SEEDS):
            for budget in statistics.BUDGETS:
                panel = datasets.load_panel(
                    aircraft_record,
                    backbone,
                    int(seed),
                    int(budget),
                    registry_root=registry_root,
                )
                train = np.asarray(panel["training_values"], dtype=np.float32)
                train_labels = np.asarray(panel["training_labels"])
                evaluation = np.asarray(panel["evaluation_values"], dtype=np.float32)
                evaluation_labels = np.asarray(panel["evaluation_labels"])
                selected_k, inner_scores = tune_k_inner_cv(
                    train, train_labels, seed=int(seed), k_grid=ks
                )
                predictions = predict_k_grid(
                    train, train_labels, evaluation, k_grid=ks
                )
                test_scores = {
                    k: float(np.mean(predicted == evaluation_labels))
                    for k, predicted in predictions.items()
                }
                frozen_k15 = reference[(backbone, int(seed), int(budget), "knn")]
                if abs(test_scores[CURRENT_K] - frozen_k15) > 1.0e-12:
                    raise RuntimeError("recomputed K=15 accuracy differs from frozen result")
                quadratic = reference[
                    (backbone, int(seed), int(budget), "quadratic")
                ]
                oracle_accuracy = max(test_scores.values())
                oracle_k = min(
                    k
                    for k, accuracy in test_scores.items()
                    if abs(accuracy - oracle_accuracy) <= 1.0e-12
                )
                tuned_accuracy = test_scores[selected_k]
                panel_rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "backbone": backbone,
                        "replicate": int(replicate),
                        "replicate_seed": int(seed),
                        "budget": int(budget),
                        "selected_k_inner_cv": int(selected_k),
                        "inner_cv_accuracy": float(inner_scores[selected_k]),
                        "tuned_knn_test_accuracy": tuned_accuracy,
                        "fixed_k15_test_accuracy": frozen_k15,
                        "oracle_test_selected_k": int(oracle_k),
                        "oracle_knn_test_accuracy": oracle_accuracy,
                        "quadratic_test_accuracy": quadratic,
                        "tuned_minus_k15_pp": 100.0 * (tuned_accuracy - frozen_k15),
                        "tuned_minus_quadratic_pp": 100.0 * (tuned_accuracy - quadratic),
                        "oracle_minus_quadratic_pp": 100.0 * (oracle_accuracy - quadratic),
                    }
                )
                for k in ks:
                    accuracy_rows.append(
                        {
                            "dataset_id": DATASET_ID,
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "k": int(k),
                            "inner_cv_accuracy": float(inner_scores[k]),
                            "test_accuracy": float(test_scores[k]),
                        }
                    )

    if len(panel_rows) != 100 or len(accuracy_rows) != 100 * len(ks):
        raise RuntimeError("Aircraft KNN tuning grid is incomplete")

    backbone_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for backbone in statistics.BACKBONES:
            rows = [
                row
                for row in panel_rows
                if row["budget"] == budget and row["backbone"] == backbone
            ]
            backbone_rows.append(
                {
                    "budget": int(budget),
                    "backbone": backbone,
                    **{
                        field: float(np.mean([row[field] for row in rows]))
                        for field in (
                            "tuned_knn_test_accuracy",
                            "fixed_k15_test_accuracy",
                            "oracle_knn_test_accuracy",
                            "quadratic_test_accuracy",
                            "tuned_minus_k15_pp",
                            "tuned_minus_quadratic_pp",
                            "oracle_minus_quadratic_pp",
                        )
                    },
                }
            )

    rank_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for seed in statistics.REPLICATE_SEEDS:
            rows = [
                row
                for row in panel_rows
                if row["budget"] == budget and row["replicate_seed"] == seed
            ]
            quadratic = [row["quadratic_test_accuracy"] for row in rows]
            for method, field in (
                ("KNN_FIXED_15", "fixed_k15_test_accuracy"),
                ("KNN_INNER_CV_TUNED", "tuned_knn_test_accuracy"),
                ("KNN_TEST_ORACLE", "oracle_knn_test_accuracy"),
            ):
                rank_rows.append(
                    {
                        "budget": int(budget),
                        "replicate_seed": int(seed),
                        "method": method,
                        "target": "quadratic_accuracy",
                        "spearman": _spearman(
                            [row[field] for row in rows], quadratic
                        ),
                    }
                )

    k_counts = Counter(int(row["selected_k_inner_cv"]) for row in panel_rows)
    oracle_counts = Counter(int(row["oracle_test_selected_k"]) for row in panel_rows)
    tuned_better = sum(
        row["tuned_knn_test_accuracy"] > row["quadratic_test_accuracy"]
        for row in panel_rows
    )
    tuned_tied = sum(
        abs(row["tuned_knn_test_accuracy"] - row["quadratic_test_accuracy"])
        <= 1.0e-12
        for row in panel_rows
    )
    oracle_better = sum(
        row["oracle_knn_test_accuracy"] > row["quadratic_test_accuracy"]
        for row in panel_rows
    )
    oracle_tied = sum(
        abs(row["oracle_knn_test_accuracy"] - row["quadratic_test_accuracy"])
        <= 1.0e-12
        for row in panel_rows
    )
    return {
        "study": STUDY,
        "stage": STAGE,
        "diagnostic_status": STATUS,
        "dataset_id": DATASET_ID,
        "k_grid": list(ks),
        "fixed_recipe_k": CURRENT_K,
        "inner_cv_folds": INNER_FOLDS,
        "selection_protocol": (
            "per-backbone, per-replicate, per-budget K selected by three-fold "
            "stratified CV on training rows only; smallest K wins exact ties"
        ),
        "oracle_warning": (
            "oracle_test_selected_k uses evaluation outcomes and is only an "
            "optimistic upper bound, never a valid tuning procedure"
        ),
        "panel_rows": panel_rows,
        "accuracy_rows": accuracy_rows,
        "backbone_rows": backbone_rows,
        "rank_rows": rank_rows,
        "selected_k_counts": {str(k): count for k, count in sorted(k_counts.items())},
        "oracle_k_counts": {str(k): count for k, count in sorted(oracle_counts.items())},
        "aggregate": {
            field: float(np.mean([row[field] for row in panel_rows]))
            for field in (
                "fixed_k15_test_accuracy",
                "tuned_knn_test_accuracy",
                "oracle_knn_test_accuracy",
                "quadratic_test_accuracy",
                "tuned_minus_k15_pp",
                "tuned_minus_quadratic_pp",
                "oracle_minus_quadratic_pp",
            )
        },
        "tuned_vs_quadratic_panel_counts": {
            "better": int(tuned_better),
            "tied": int(tuned_tied),
            "worse": int(100 - tuned_better - tuned_tied),
        },
        "oracle_vs_quadratic_panel_counts": {
            "better": int(oracle_better),
            "tied": int(oracle_tied),
            "worse": int(100 - oracle_better - oracle_tied),
        },
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "input_hashes": dict(verified["input_hashes"]),
        "input_identity": {
            key: verified["raw"][key]
            for key in (
                "protocol_sha256",
                "code_identity_sha256",
                "run_identity_sha256",
                "audited_registry_sha256",
                "lineage_lock_sha256",
            )
        },
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty CSV")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _report(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    tuned_counts = summary["tuned_vs_quadratic_panel_counts"]
    oracle_counts = summary["oracle_vs_quadratic_panel_counts"]
    rank_by_method: dict[str, list[float]] = defaultdict(list)
    for row in summary["rank_rows"]:
        if row["spearman"] is not None:
            rank_by_method[str(row["method"])].append(float(row["spearman"]))
    return "\n".join(
        [
            "# Aircraft KNN K-tuning diagnostic",
            "",
            "Status: **post-outcome descriptive diagnostic only**. The frozen confirmation decision is unchanged.",
            "",
            "## Frozen recipe",
            "",
            "The prior Food-101 and Confirmation V2 head used cosine, distance-weighted KNN with fixed `k=15`; it was not tuned.",
            "",
            "## Leakage-free inner-CV tuning",
            "",
            f"- Fixed k=15 mean accuracy: **{100.0 * aggregate['fixed_k15_test_accuracy']:.3f}%**",
            f"- Inner-CV-tuned KNN mean accuracy: **{100.0 * aggregate['tuned_knn_test_accuracy']:.3f}%**",
            f"- Mean improvement from tuning: **{aggregate['tuned_minus_k15_pp']:+.3f} pp**",
            f"- Quadratic mean accuracy: **{100.0 * aggregate['quadratic_test_accuracy']:.3f}%**",
            f"- Tuned KNN minus quadratic: **{aggregate['tuned_minus_quadratic_pp']:+.3f} pp**",
            f"- Tuned KNN vs quadratic across 100 panels: {tuned_counts['better']} better / {tuned_counts['tied']} tied / {tuned_counts['worse']} worse",
            f"- Mean rank Spearman with quadratic: fixed k=15 **{np.mean(rank_by_method['KNN_FIXED_15']):.3f}**, tuned **{np.mean(rank_by_method['KNN_INNER_CV_TUNED']):.3f}**",
            "",
            "## Optimistic upper bound",
            "",
            "The oracle chooses K using the evaluation outcome and is therefore not a valid procedure; it only asks whether this K grid contains enough capacity.",
            "",
            f"- Oracle KNN minus quadratic: **{aggregate['oracle_minus_quadratic_pp']:+.3f} pp**",
            f"- Oracle KNN vs quadratic: {oracle_counts['better']} better / {oracle_counts['tied']} tied / {oracle_counts['worse']} worse",
            "",
            "## Interpretation",
            "",
            "If legitimate inner-CV tuning closes only a small fraction of the gap—and even the test-oracle remains behind—then k=15 was not the Aircraft explanation. If the oracle closes the gap but inner CV does not, the issue is K-selection reliability at these sample sizes rather than KNN capacity.",
            "",
        ]
    )


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_knn_tuning(full_artifact, registry_root)
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "panel_rows.csv": _csv_bytes(summary["panel_rows"]),
        "accuracy_by_k.csv": _csv_bytes(summary["accuracy_rows"]),
        "backbone_summary.csv": _csv_bytes(summary["backbone_rows"]),
        "rank_correlations.csv": _csv_bytes(summary["rank_rows"]),
        "report.md": _report(summary).encode("utf-8"),
    }
    manifest = {
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "files": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
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
            raise RuntimeError("written diagnostic hash mismatch")
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
