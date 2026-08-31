"""Decompose FUSED3/SQ OI ranking into best-, second-, and worst-rival stages."""

from __future__ import annotations

import argparse
import hashlib
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
from scipy.stats import rankdata

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics
from experiments.fused3_envelope_reanalysis.squared_feature_oi import (
    _canonical_json,
    _csv_bytes,
    _sha256,
    _sha256_bytes,
    _validate_matrix_target,
    fit_squared_feature_lift,
)
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen
from experiments.scalable_relevance_kmeans.scalable import _class_ids, _score_kernel
from experiments.scalable_relevance_kmeans.scalable_v2 import fast_tiled_oi_score


STUDY = "fused3_confirmation_v2_aircraft_score_component_diagnostic"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
DATASET_ID = "torchvision_fgvc_aircraft"
STATE_SPECS: tuple[tuple[str, int | None], ...] = (
    ("FUSED3", 0),
    ("FUSED3-SQ64", 64),
    ("FUSED3-SQALL", None),
)
COMPONENTS = (
    "best_own_relative_margin",
    "best_own_win_rate",
    "best_own_worst_rival_oi",
    "second_own_relative_margin",
    "second_own_win_rate",
    "mean_pair_second_own_oi",
    "exact_oi",
)


def component_geometry(
    values: Any,
    target: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
) -> dict[str, float]:
    """Compute nested OI component scores from one immutable fitted state."""

    matrix, labels = _validate_matrix_target(values, target)
    prototypes = np.asarray(centers, dtype=np.float32)
    owner_array = np.asarray(owners)
    feature_weights = np.asarray(weights, dtype=np.float64)
    if prototypes.ndim != 2 or prototypes.shape[1] != matrix.shape[1]:
        raise ValueError("centers have the wrong feature shape")
    if owner_array.shape != (len(prototypes),):
        raise ValueError("owners do not align with centers")
    if feature_weights.shape != (matrix.shape[1],) or np.any(feature_weights <= 0):
        raise ValueError("weights must be a positive feature vector")
    ids_by_class = _class_ids(owner_array)
    if set(dict.fromkeys(labels.tolist())) != set(ids_by_class):
        raise ValueError("evaluation and prototype class sets must match")
    if any(len(ids) < 2 for ids in ids_by_class.values()):
        raise ValueError("component decomposition requires two prototypes per class")

    scores = _score_kernel(matrix, prototypes, feature_weights)
    weighted_row_norm = np.sum(
        np.square(matrix.astype(np.float64)) * feature_weights[None, :], axis=1
    )
    best_margin_by_class: list[float] = []
    second_margin_by_class: list[float] = []
    best_win_by_class: list[float] = []
    second_win_by_class: list[float] = []
    best_worst_by_class: list[float] = []
    second_worst_by_class: list[float] = []
    second_pair_scores: list[float] = []
    for source, own_ids in ids_by_class.items():
        rows = np.flatnonzero(labels == source)
        own_scores = scores[rows][:, own_ids]
        best_own = np.max(own_scores, axis=1)
        second_own = np.partition(own_scores, kth=-2, axis=1)[:, -2]
        wrong_ids = np.concatenate(
            [ids for other, ids in ids_by_class.items() if other != source]
        )
        best_wrong = np.max(scores[rows][:, wrong_ids], axis=1)
        row_norm = weighted_row_norm[rows]
        best_distance = np.maximum(row_norm - 2.0 * best_own, 0.0)
        second_distance = np.maximum(row_norm - 2.0 * second_own, 0.0)
        wrong_distance = np.maximum(row_norm - 2.0 * best_wrong, 0.0)
        epsilon = np.finfo(np.float64).eps
        best_margin_by_class.append(
            float(
                np.mean(
                    (wrong_distance - best_distance)
                    / (wrong_distance + best_distance + epsilon)
                )
            )
        )
        second_margin_by_class.append(
            float(
                np.mean(
                    (wrong_distance - second_distance)
                    / (wrong_distance + second_distance + epsilon)
                )
            )
        )
        best_win_by_class.append(float(np.mean(best_wrong <= best_own)))
        second_win_by_class.append(float(np.mean(best_wrong <= second_own)))
        best_pair: list[float] = []
        second_pair: list[float] = []
        for other, other_ids in ids_by_class.items():
            if other == source:
                continue
            target_best = np.max(scores[rows][:, other_ids], axis=1)
            best_score = 1.0 - float(np.mean(target_best > best_own))
            second_score = 1.0 - float(np.mean(target_best > second_own))
            best_pair.append(best_score)
            second_pair.append(second_score)
            second_pair_scores.append(second_score)
        best_worst_by_class.append(min(best_pair))
        second_worst_by_class.append(min(second_pair))

    exact = float(np.mean(second_worst_by_class))
    tiled, _diagnostics = fast_tiled_oi_score(
        matrix,
        labels,
        centers=prototypes,
        owners=owner_array,
        weights=feature_weights,
        memory_budget_mb=64,
    )
    if abs(exact - tiled) > 1.0e-12:
        raise RuntimeError("component decomposition did not reproduce exact OI")
    return {
        "best_own_relative_margin": float(np.mean(best_margin_by_class)),
        "best_own_win_rate": float(np.mean(best_win_by_class)),
        "best_own_worst_rival_oi": float(np.mean(best_worst_by_class)),
        "second_own_relative_margin": float(np.mean(second_margin_by_class)),
        "second_own_win_rate": float(np.mean(second_win_by_class)),
        "mean_pair_second_own_oi": float(np.mean(second_pair_scores)),
        "exact_oi": exact,
    }


def crossfit_component_state(
    values: Any,
    target: Any,
    *,
    seed: int,
    square_feature_count: int | None,
) -> dict[str, Any]:
    """Fit one state in frozen folds and aggregate all score components."""

    matrix, labels = _validate_matrix_target(values, target)
    normalized = food101._row_l2(matrix)
    folds = food101._stratified_folds(labels, n_splits=5, seed=int(seed))
    started = time.perf_counter()
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        if square_feature_count == 0:
            transformed_train = normalized[train]
            transformed_holdout = normalized[holdout]
            lift_diagnostics = {
                "selected_square_feature_count": 0,
                "output_feature_count": int(matrix.shape[1]),
                "state_sha256": None,
            }
        else:
            lift = fit_squared_feature_lift(
                matrix[train],
                labels[train],
                max_square_features=square_feature_count,
            )
            transformed_train = lift.transform(matrix[train])
            transformed_holdout = lift.transform(matrix[holdout])
            lift_diagnostics = lift.diagnostics
        k_per_class = fused_food_screen._k_per_class(labels, train)
        selector = fused_food_screen._build_selector(
            "FUSED3", k_per_class, fold_seed
        ).fit(transformed_train, labels[train])
        centers_before = np.array(selector.centers_, copy=True)
        owners_before = np.array(selector.owners_, copy=True)
        weights_before = np.array(selector.weights_, copy=True)
        components = component_geometry(
            transformed_holdout,
            labels[holdout],
            centers=selector.centers_,
            owners=selector.owners_,
            weights=selector.weights_,
        )
        if not np.array_equal(selector.centers_, centers_before):
            raise RuntimeError("component diagnostic mutated prototype centers")
        if not np.array_equal(selector.owners_, owners_before):
            raise RuntimeError("component diagnostic mutated prototype owners")
        if not np.array_equal(selector.weights_, weights_before):
            raise RuntimeError("component diagnostic mutated relevance weights")
        raw_count = matrix.shape[1]
        square_weights = np.asarray(selector.weights_[raw_count:], dtype=np.float64)
        raw_weights = np.asarray(selector.weights_[:raw_count], dtype=np.float64)
        fold_rows.append(
            {
                "fold": int(fold),
                "fold_seed": fold_seed,
                **components,
                "selected_square_feature_count": int(
                    lift_diagnostics["selected_square_feature_count"]
                ),
                "output_feature_count": int(lift_diagnostics["output_feature_count"]),
                "lift_state_sha256": lift_diagnostics["state_sha256"],
                "square_to_raw_mean_weight_ratio": (
                    None
                    if square_weights.size == 0
                    else float(np.mean(square_weights) / np.mean(raw_weights))
                ),
            }
        )
    return {
        **{
            component: float(np.mean([row[component] for row in fold_rows]))
            for component in COMPONENTS
        },
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
    selectors: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]
) -> tuple[dict[tuple[str, int, int], float], dict[tuple[str, int, int, str], float]]:
    frozen = {
        (str(row["backbone"]), int(row["replicate_seed"]), int(row["budget"])): float(
            row["score"]
        )
        for row in selectors
        if row["dataset_id"] == DATASET_ID and row["candidate_id"] == "FUSED3"
    }
    reference = {
        (
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["head"]),
        ): float(row["test_accuracy"])
        for row in references
        if row["dataset_id"] == DATASET_ID
    }
    if len(frozen) != 100 or len(reference) != 400:
        raise RuntimeError("frozen Aircraft tables are incomplete")
    return frozen, reference


def analyze_score_components(
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
    state_rows: list[dict[str, Any]] = []
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
                for state_id, square_count in STATE_SPECS:
                    result = crossfit_component_state(
                        panel["training_values"],
                        panel["training_labels"],
                        seed=int(seed),
                        square_feature_count=square_count,
                    )
                    if state_id == "FUSED3" and abs(
                        result["exact_oi"] - frozen_scores[identity]
                    ) > 1.0e-12:
                        raise RuntimeError("component baseline differs from frozen FUSED3")
                    state_rows.append(
                        {
                            "dataset_id": DATASET_ID,
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "state_id": state_id,
                            **{component: result[component] for component in COMPONENTS},
                            "best_head_envelope_accuracy": float(envelope),
                            "best_head": statistics.HEADS[
                                int(np.argmax([heads[head] for head in statistics.HEADS]))
                            ],
                            "total_wall_seconds": float(result["total_wall_seconds"]),
                        }
                    )
                    for fold_row in result["folds"]:
                        fold_rows.append(
                            {
                                "backbone": backbone,
                                "replicate": int(replicate),
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                "state_id": state_id,
                                **fold_row,
                            }
                        )
    if len(state_rows) != 300 or len(fold_rows) != 1500:
        raise RuntimeError("score-component state grid is incomplete")

    ranking_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for seed in statistics.REPLICATE_SEEDS:
            panel = [
                row
                for row in state_rows
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
            for state_id, _square_count in STATE_SPECS:
                states = [row for row in panel if row["state_id"] == state_id]
                for component in COMPONENTS:
                    scores = [
                        next(
                            float(row[component])
                            for row in states
                            if row["backbone"] == backbone
                        )
                        for backbone in statistics.BACKBONES
                    ]
                    maximum = max(scores)
                    selected = [
                        row
                        for row in states
                        if abs(float(row[component]) - maximum)
                        <= statistics.SELECTION_TIE_ATOL
                    ]
                    selected_accuracy = float(
                        np.mean([row["best_head_envelope_accuracy"] for row in selected])
                    )
                    ranking_rows.append(
                        {
                            "budget": int(budget),
                            "replicate_seed": int(seed),
                            "state_id": state_id,
                            "component": component,
                            "selected_backbones": sorted(
                                row["backbone"] for row in selected
                            ),
                            "regret_pp": float(100.0 * (oracle - selected_accuracy)),
                            "spearman_with_best_head_envelope": _spearman(scores, targets),
                        }
                    )

    aggregate_rows: list[dict[str, Any]] = []
    for state_id, _square_count in STATE_SPECS:
        for component in COMPONENTS:
            rows = [
                row
                for row in ranking_rows
                if row["state_id"] == state_id and row["component"] == component
            ]
            counts: Counter[str] = Counter(
                backbone for row in rows for backbone in row["selected_backbones"]
            )
            aggregate_rows.append(
                {
                    "state_id": state_id,
                    "component": component,
                    "mean_regret_pp": float(np.mean([row["regret_pp"] for row in rows])),
                    "mean_spearman": float(
                        np.mean([row["spearman_with_best_head_envelope"] for row in rows])
                    ),
                    "selection_counts": dict(sorted(counts.items())),
                }
            )
    return {
        "study": STUDY,
        "stage": STAGE,
        "diagnostic_status": STATUS,
        "dataset_id": DATASET_ID,
        "state_specs": [
            {
                "state_id": state_id,
                "square_feature_count": (
                    "all" if square_count is None else int(square_count)
                ),
            }
            for state_id, square_count in STATE_SPECS
        ],
        "component_contracts": {
            "best_own_relative_margin": "macro mean (d_wrong-d_best_own)/(d_wrong+d_best_own)",
            "best_own_win_rate": "macro rate best-own distance <= nearest-rival distance",
            "best_own_worst_rival_oi": "exact OI pair/min/macro aggregation with best-own threshold",
            "second_own_relative_margin": "macro mean (d_wrong-d_second_own)/(d_wrong+d_second_own)",
            "second_own_win_rate": "macro rate second-own distance <= nearest-rival distance",
            "mean_pair_second_own_oi": "mean of every directed pair score using second-own threshold",
            "exact_oi": "mean over source classes of minimum directed-pair score using second-own threshold",
        },
        "state_rows": state_rows,
        "fold_rows": fold_rows,
        "ranking_rows": ranking_rows,
        "aggregate_rows": aggregate_rows,
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "input_hashes": dict(verified["input_hashes"]),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lookup = {
        (row["state_id"], row["component"]): row
        for row in summary["aggregate_rows"]
    }
    lines = [
        "# Aircraft FUSED3 score-component diagnostic",
        "",
        "Status: **post-outcome descriptive diagnostic only**. The frozen confirmation decision is unchanged.",
        "",
        "| State | Component | Mean regret (pp) | Mean Spearman | Selections |",
        "|---|---|---:|---:|---|",
    ]
    for state_id, _square_count in STATE_SPECS:
        for component in COMPONENTS:
            row = lookup[(state_id, component)]
            lines.append(
                f"| {state_id} | {component} | {row['mean_regret_pp']:.3f} | "
                f"{row['mean_spearman']:.3f} | `{_canonical_json(row['selection_counts'])}` |"
            )
    lines.extend(
        [
            "",
            "The rows successively expose the effect of replacing the second-own threshold with best-own geometry and replacing exact worst-rival aggregation with smoother margin or mean-pair summaries. No component is a promoted OI definition.",
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
    summary = analyze_score_components(full_artifact, registry_root)
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "state_rows.csv": _csv_bytes(summary["state_rows"]),
        "fold_rows.csv": _csv_bytes(summary["fold_rows"]),
        "ranking_rows.csv": _csv_bytes(summary["ranking_rows"]),
        "aggregate_rows.csv": _csv_bytes(summary["aggregate_rows"]),
        "report.md": _report(summary).encode("utf-8"),
    }
    manifest = {
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "files": {
            name: {"sha256": _sha256_bytes(content), "size_bytes": len(content)}
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
            raise RuntimeError("written component artifact hash mismatch")
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
