"""Post-outcome Aircraft capacity diagnostic for neighborhood-style heads.

This module deliberately does not alter the frozen FUSED3 confirmation.  It
uses only training-cohort inner cross-validation to choose head parameters and
uses the fixed evaluation cohort once for the reported accuracy.
"""

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
import time
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics
from experiments.fused3_envelope_reanalysis.knn_tuning import predict_k_grid


STUDY = "fused3_confirmation_v2_aircraft_neighborhood_head_diagnostic"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
DATASET_ID = "torchvision_fgvc_aircraft"
FOCAL_BACKBONES = ("dinov2-small", "openclip-vit-b-32")
METHODS = (
    "SOFT_COSINE_PARZEN",
    "CENTROID_SUBSPACE_KNN",
    "PCA_WHITENED_KNN",
    "RBF_SVC_TUNED",
)
INNER_FOLDS = 3
PARZEN_BETA_GRID = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
SUBSPACE_DIM_GRID = (16, 32, 64, 96)
SUBSPACE_K_GRID = (1, 3, 7, 15, 31)
RBF_GRID = (
    (1.0, 1.0),
    (1.0, 0.25),
    (1.0, 4.0),
    (0.3, 1.0),
    (3.0, 1.0),
    (0.3, 0.25),
    (0.3, 4.0),
    (3.0, 0.25),
    (3.0, 4.0),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_arrays(
    training_values: Any,
    training_labels: Any,
    query_values: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(training_values, dtype=np.float32)
    labels = np.asarray(training_labels)
    query = np.asarray(query_values, dtype=np.float32)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]:
        raise ValueError("training/query values must be aligned 2-D matrices")
    if labels.ndim != 1 or len(labels) != len(train):
        raise ValueError("training labels must align with training rows")
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError("head inputs must be finite")
    return train, labels, query


def soft_cosine_predictions(
    training_values: Any,
    training_labels: Any,
    query_values: Any,
    *,
    beta_grid: Sequence[float] = PARZEN_BETA_GRID,
) -> dict[float, np.ndarray]:
    """Classify with a cosine Parzen window over every training neighbor."""

    train, labels, query = _validate_arrays(
        training_values, training_labels, query_values
    )
    betas = tuple(float(value) for value in beta_grid)
    if not betas or any(not np.isfinite(value) or value <= 0 for value in betas):
        raise ValueError("beta_grid must contain positive finite values")
    if betas != tuple(sorted(set(betas))):
        raise ValueError("beta_grid must be strictly increasing and unique")
    train = normalize(train, copy=True)
    query = normalize(query, copy=True)
    similarity = np.asarray(query @ train.T, dtype=np.float32)
    classes, encoded = np.unique(labels, return_inverse=True)
    output: dict[float, np.ndarray] = {}
    for beta in betas:
        scaled = np.asarray(similarity * np.float32(beta), dtype=np.float32)
        scaled -= np.max(scaled, axis=1, keepdims=True)
        kernels = np.exp(scaled, dtype=np.float32)
        votes = np.zeros((len(query), len(classes)), dtype=np.float64)
        for class_id in range(len(classes)):
            votes[:, class_id] = np.sum(
                kernels[:, encoded == class_id], axis=1, dtype=np.float64
            )
        output[beta] = np.asarray(classes[np.argmax(votes, axis=1)])
    return output


def _centroid_basis(values: np.ndarray, labels: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = normalize(np.asarray(values, dtype=np.float32), copy=True)
    classes = np.unique(labels)
    mean = np.mean(matrix, axis=0, dtype=np.float64)
    centroids = np.stack(
        [np.mean(matrix[labels == label], axis=0, dtype=np.float64) for label in classes]
    )
    _u, _s, vt = np.linalg.svd(centroids - mean, full_matrices=False)
    usable = min(int(rank), len(classes) - 1, matrix.shape[1])
    if usable < 1:
        raise ValueError("centroid subspace has zero usable rank")
    return np.asarray(mean, dtype=np.float32), np.asarray(vt[:usable].T, dtype=np.float32)


def _project(values: np.ndarray, mean: np.ndarray, basis: np.ndarray, rank: int) -> np.ndarray:
    matrix = normalize(np.asarray(values, dtype=np.float32), copy=True)
    return np.asarray((matrix - mean) @ basis[:, :rank], dtype=np.float32)


def _valid_dims(dimension: int, class_count: int) -> tuple[int, ...]:
    maximum = min(dimension, class_count - 1)
    dims = tuple(value for value in SUBSPACE_DIM_GRID if value <= maximum)
    if not dims:
        dims = (maximum,)
    elif dims[-1] != maximum:
        dims = (*dims, maximum)
    return tuple(sorted(set(dims)))


def _folds(values: np.ndarray, labels: np.ndarray, seed: int):
    encoded_classes, encoded = np.unique(labels, return_inverse=True)
    if len(encoded_classes) < 2:
        raise ValueError("at least two classes are required")
    splitter = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=int(seed)
    )
    return tuple(splitter.split(values, encoded))


def _select_max(scores: Mapping[Any, int], order: Sequence[Any]) -> Any:
    best = max(scores.values())
    return next(item for item in order if scores[item] == best)


def tune_parzen(
    values: Any,
    labels: Any,
    *,
    seed: int,
    beta_grid: Sequence[float] = PARZEN_BETA_GRID,
) -> tuple[float, dict[float, float]]:
    matrix = np.asarray(values, dtype=np.float32)
    target = np.asarray(labels)
    correct = {float(beta): 0 for beta in beta_grid}
    total = 0
    for train_ids, validation_ids in _folds(matrix, target, seed):
        predictions = soft_cosine_predictions(
            matrix[train_ids], target[train_ids], matrix[validation_ids], beta_grid=beta_grid
        )
        for beta, predicted in predictions.items():
            correct[beta] += int(np.count_nonzero(predicted == target[validation_ids]))
        total += len(validation_ids)
    order = tuple(float(beta) for beta in beta_grid)
    selected = float(_select_max(correct, order))
    return selected, {beta: float(value / total) for beta, value in correct.items()}


def tune_subspace_knn(
    values: Any,
    labels: Any,
    *,
    seed: int,
    kind: str,
) -> tuple[tuple[int, int], dict[tuple[int, int], float]]:
    matrix = np.asarray(values, dtype=np.float32)
    target = np.asarray(labels)
    dims = _valid_dims(matrix.shape[1], len(np.unique(target)))
    configs = tuple((dim, k) for dim in dims for k in SUBSPACE_K_GRID)
    correct = {config: 0 for config in configs}
    total = 0
    for fold_index, (train_ids, validation_ids) in enumerate(
        _folds(matrix, target, seed)
    ):
        train = matrix[train_ids]
        validation = matrix[validation_ids]
        if kind == "centroid":
            mean, basis = _centroid_basis(train, target[train_ids], max(dims))
            projected_train = _project(train, mean, basis, max(dims))
            projected_validation = _project(validation, mean, basis, max(dims))
        elif kind == "pca_whitened":
            normalized_train = normalize(train, copy=True)
            normalized_validation = normalize(validation, copy=True)
            estimator = PCA(
                n_components=max(dims),
                whiten=True,
                svd_solver="randomized",
                random_state=int(seed) + int(fold_index),
            )
            projected_train = np.asarray(
                estimator.fit_transform(normalized_train), dtype=np.float32
            )
            projected_validation = np.asarray(
                estimator.transform(normalized_validation), dtype=np.float32
            )
        else:
            raise ValueError("kind must be 'centroid' or 'pca_whitened'")
        for dim in dims:
            predictions = predict_k_grid(
                projected_train[:, :dim],
                target[train_ids],
                projected_validation[:, :dim],
                k_grid=SUBSPACE_K_GRID,
            )
            for k, predicted in predictions.items():
                correct[(dim, k)] += int(
                    np.count_nonzero(predicted == target[validation_ids])
                )
        total += len(validation_ids)
    selected = tuple(_select_max(correct, configs))
    return selected, {config: float(value / total) for config, value in correct.items()}


def _fit_subspace_predict(
    training_values: np.ndarray,
    training_labels: np.ndarray,
    query_values: np.ndarray,
    *,
    seed: int,
    kind: str,
    dim: int,
    k: int,
) -> np.ndarray:
    if kind == "centroid":
        mean, basis = _centroid_basis(training_values, training_labels, dim)
        train = _project(training_values, mean, basis, dim)
        query = _project(query_values, mean, basis, dim)
    elif kind == "pca_whitened":
        normalized_train = normalize(training_values, copy=True)
        normalized_query = normalize(query_values, copy=True)
        estimator = PCA(
            n_components=dim,
            whiten=True,
            svd_solver="randomized",
            random_state=int(seed),
        )
        train = np.asarray(estimator.fit_transform(normalized_train), dtype=np.float32)
        query = np.asarray(estimator.transform(normalized_query), dtype=np.float32)
    else:
        raise ValueError("unknown subspace kind")
    return predict_k_grid(train, training_labels, query, k_grid=(k,))[k]


def _gamma_scale(values: np.ndarray) -> float:
    variance = float(np.var(values))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("RBF training variance must be positive and finite")
    return float(1.0 / (values.shape[1] * variance))


def tune_rbf_svc(
    values: Any,
    labels: Any,
    *,
    seed: int,
    grid: Sequence[tuple[float, float]] = RBF_GRID,
) -> tuple[tuple[float, float], dict[tuple[float, float], float]]:
    matrix = normalize(np.asarray(values, dtype=np.float32), copy=True)
    target = np.asarray(labels)
    configs = tuple((float(c), float(multiplier)) for c, multiplier in grid)
    correct = {config: 0 for config in configs}
    total = 0
    for train_ids, validation_ids in _folds(matrix, target, seed):
        gamma = _gamma_scale(matrix[train_ids])
        for config in configs:
            c, multiplier = config
            estimator = SVC(
                kernel="rbf",
                C=c,
                gamma=gamma * multiplier,
                cache_size=1024,
            ).fit(matrix[train_ids], target[train_ids])
            predicted = estimator.predict(matrix[validation_ids])
            correct[config] += int(
                np.count_nonzero(predicted == target[validation_ids])
            )
        total += len(validation_ids)
    selected = tuple(_select_max(correct, configs))
    return selected, {config: float(value / total) for config, value in correct.items()}


def fit_evaluate_heads(
    training_values: Any,
    training_labels: Any,
    evaluation_values: Any,
    evaluation_labels: Any,
    *,
    seed: int,
    methods: Sequence[str] = METHODS,
) -> list[dict[str, Any]]:
    """Tune only on training rows, then evaluate each selected head once."""

    train, labels, evaluation = _validate_arrays(
        training_values, training_labels, evaluation_values
    )
    evaluation_target = np.asarray(evaluation_labels)
    if evaluation_target.ndim != 1 or len(evaluation_target) != len(evaluation):
        raise ValueError("evaluation labels must align with evaluation rows")
    requested = tuple(methods)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("methods must be nonempty and unique")
    if not set(requested).issubset(METHODS):
        raise ValueError(f"methods must be drawn from {METHODS!r}")
    rows: list[dict[str, Any]] = []
    for method in requested:
        started = time.perf_counter()
        if method == "SOFT_COSINE_PARZEN":
            beta, inner = tune_parzen(train, labels, seed=seed)
            predicted = soft_cosine_predictions(
                train, labels, evaluation, beta_grid=(beta,)
            )[beta]
            config = {"beta": beta}
            inner_accuracy = inner[beta]
        elif method in ("CENTROID_SUBSPACE_KNN", "PCA_WHITENED_KNN"):
            kind = "centroid" if method == "CENTROID_SUBSPACE_KNN" else "pca_whitened"
            (dim, k), inner = tune_subspace_knn(
                train, labels, seed=seed, kind=kind
            )
            predicted = _fit_subspace_predict(
                train,
                labels,
                evaluation,
                seed=seed,
                kind=kind,
                dim=dim,
                k=k,
            )
            config = {"dimension": int(dim), "k": int(k)}
            inner_accuracy = inner[(dim, k)]
        elif method == "RBF_SVC_TUNED":
            (c, multiplier), inner = tune_rbf_svc(train, labels, seed=seed)
            normalized_train = normalize(train, copy=True)
            normalized_evaluation = normalize(evaluation, copy=True)
            gamma = _gamma_scale(normalized_train)
            estimator = SVC(
                kernel="rbf",
                C=c,
                gamma=gamma * multiplier,
                cache_size=1024,
            ).fit(normalized_train, labels)
            predicted = estimator.predict(normalized_evaluation)
            config = {
                "C": c,
                "gamma_scale_multiplier": multiplier,
                "resolved_gamma": gamma * multiplier,
            }
            inner_accuracy = inner[(c, multiplier)]
        else:  # pragma: no cover - guarded above
            raise RuntimeError("unreachable neighborhood method")
        rows.append(
            {
                "method": method,
                "selected_config": config,
                "inner_cv_accuracy": float(inner_accuracy),
                "test_accuracy": float(np.mean(predicted == evaluation_target)),
                "wall_seconds": float(time.perf_counter() - started),
                "training_row_count": int(len(train)),
                "evaluation_row_count": int(len(evaluation)),
            }
        )
    return rows


def _reference_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, int, str], float]:
    output: dict[tuple[str, int, int, str], float] = {}
    for row in rows:
        if row["dataset_id"] != DATASET_ID:
            continue
        key = (
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["head"]),
        )
        output[key] = float(row["test_accuracy"])
    if len(output) != 400:
        raise ValueError("Aircraft neighborhood diagnostic requires 400 reference rows")
    return output


def analyze_neighborhood_heads(
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
                    aircraft,
                    backbone,
                    int(seed),
                    int(budget),
                    registry_root=root,
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
                            "frozen_linear_accuracy": reference[(backbone, int(seed), int(budget), "linear")],
                            "frozen_quadratic_accuracy": reference[(backbone, int(seed), int(budget), "quadratic")],
                            "frozen_knn_accuracy": reference[(backbone, int(seed), int(budget), "knn")],
                            "frozen_rbf_accuracy": reference[(backbone, int(seed), int(budget), "rbf")],
                        }
                    )
    expected = len(FOCAL_BACKBONES) * len(statistics.REPLICATE_SEEDS) * len(statistics.BUDGETS) * len(methods)
    if len(panel_rows) != expected:
        raise RuntimeError("Aircraft neighborhood-head grid is incomplete")

    aggregate_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for backbone in FOCAL_BACKBONES:
            for method in methods:
                rows = [
                    row
                    for row in panel_rows
                    if row["budget"] == budget
                    and row["backbone"] == backbone
                    and row["method"] == method
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
        for method in methods:
            openclip = [
                row
                for row in panel_rows
                if row["budget"] == budget
                and row["backbone"] == "openclip-vit-b-32"
                and row["method"] == method
            ]
            dino_quadratic = [
                reference[("dinov2-small", int(seed), int(budget), "quadratic")]
                for seed in statistics.REPLICATE_SEEDS
            ]
            deltas = [
                float(row["test_accuracy"] - quadratic)
                for row, quadratic in zip(
                    sorted(openclip, key=lambda row: row["replicate_seed"]),
                    dino_quadratic,
                )
            ]
            focal_rows.append(
                {
                    "budget": int(budget),
                    "method": method,
                    "openclip_mean_accuracy": float(np.mean([row["test_accuracy"] for row in openclip])),
                    "dino_frozen_quadratic_mean_accuracy": float(np.mean(dino_quadratic)),
                    "openclip_minus_dino_quadratic_pp": float(100.0 * np.mean(deltas)),
                    "openclip_better_panel_count": int(sum(delta > 0 for delta in deltas)),
                    "openclip_tied_panel_count": int(sum(abs(delta) <= 1.0e-12 for delta in deltas)),
                }
            )

    best_openclip = max(
        focal_rows,
        key=lambda row: (row["openclip_minus_dino_quadratic_pp"], -row["budget"]),
    )
    return {
        "study": STUDY,
        "stage": STAGE,
        "diagnostic_status": STATUS,
        "dataset_id": DATASET_ID,
        "focal_backbones": list(FOCAL_BACKBONES),
        "methods": list(methods),
        "inner_cv_folds": INNER_FOLDS,
        "selection_protocol": "all head parameters selected by three-fold stratified CV on training-cohort rows only; evaluation labels are used once after selection",
        "panel_rows": panel_rows,
        "aggregate_rows": aggregate_rows,
        "openclip_vs_dino_quadratic_rows": focal_rows,
        "best_openclip_capacity_result": dict(best_openclip),
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
    lines = [
        "# Aircraft neighborhood-head capacity diagnostic",
        "",
        "Status: **post-outcome descriptive diagnostic only**. The frozen confirmation decision is unchanged.",
        "",
        "Every configuration is selected by three-fold stratified inner CV on the training cohort. Evaluation labels are used only for the final reported accuracy.",
        "",
        "## Mean accuracy by backbone and budget",
        "",
        "| Budget | Backbone | Head | Accuracy | Frozen quadratic | Delta | Median wall |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate_rows"]:
        lines.append(
            f"| {row['budget']} | {row['backbone']} | {row['method']} | {100*row['mean_test_accuracy']:.3f}% | {100*row['mean_quadratic_accuracy']:.3f}% | {row['new_minus_own_quadratic_pp']:+.3f} pp | {row['median_wall_seconds']:.2f}s |"
        )
    lines.extend(
        [
            "",
            "## Can OpenCLIP exceed DINO's frozen quadratic head?",
            "",
            "| Budget | OpenCLIP head | OpenCLIP accuracy | DINO quadratic | Delta | Panels better |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["openclip_vs_dino_quadratic_rows"]:
        lines.append(
            f"| {row['budget']} | {row['method']} | {100*row['openclip_mean_accuracy']:.3f}% | {100*row['dino_frozen_quadratic_mean_accuracy']:.3f}% | {row['openclip_minus_dino_quadratic_pp']:+.3f} pp | {row['openclip_better_panel_count']}/5 |"
        )
    best = summary["best_openclip_capacity_result"]
    lines.extend(
        [
            "",
            "## Capacity conclusion",
            "",
            f"The strongest tested OpenCLIP result was `{best['method']}` at budget {best['budget']}, with OpenCLIP minus DINO quadratic = **{best['openclip_minus_dino_quadratic_pp']:+.3f} pp**.",
            "",
            "These heads test neighborhood capacity on the already-inspected Aircraft panel. They cannot authorize candidate promotion, recipe tuning, or reinterpretation of the frozen confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
    *,
    methods: Sequence[str] = METHODS,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_neighborhood_heads(
        full_artifact, registry_root, methods=methods
    )
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "panel_rows.csv": _csv_bytes(summary["panel_rows"]),
        "aggregate_rows.csv": _csv_bytes(summary["aggregate_rows"]),
        "openclip_vs_dino_quadratic.csv": _csv_bytes(
            summary["openclip_vs_dino_quadratic_rows"]
        ),
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
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args(argv)
    write_bundle(args.input, args.registry_root, args.output, methods=args.methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
