"""Contrastive prototype diagnostic for squared-feature FUSED3 states."""

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
from experiments.fused3_envelope_reanalysis.score_components import (
    COMPONENTS,
    component_geometry,
)
from experiments.fused3_envelope_reanalysis.squared_feature_oi import (
    STUDY as SQUARED_STUDY,
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
from experiments.scalable_relevance_kmeans.scalable_v2 import _margin_indices


STUDY = "fused3_confirmation_v2_aircraft_discriminative_prototype_diagnostic"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
DATASET_ID = "torchvision_fgvc_aircraft"
LEARNING_RATE = 0.1
UPDATE_ROWS_PER_CLASS = 32
EPOCHS = (0, 1, 3)
BASE_STATES: tuple[tuple[str, int | None], ...] = (
    ("FUSED3-SQ64", 64),
    ("FUSED3-SQALL", None),
)
RANK_COMPONENTS = ("best_own_win_rate", "exact_oi")


def _stage_id(base_state: str, epoch: int) -> str:
    if epoch == 0:
        return base_state
    return f"{base_state}-LVQ{epoch}"


STATES = tuple(
    _stage_id(base_state, epoch)
    for base_state, _square_count in BASE_STATES
    for epoch in EPOCHS
)


def contrastive_lvq_epoch(
    values: Any,
    target: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
    learning_rate: float = LEARNING_RATE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one batch own-attract/rival-repel update to misclassified rows."""

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
    if not isinstance(learning_rate, float) or not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be a float in (0, 1]")
    ids_by_class = _class_ids(owner_array)
    if set(dict.fromkeys(labels.tolist())) != set(ids_by_class):
        raise ValueError("training and prototype class sets must match")
    scores = _score_kernel(matrix, prototypes, feature_weights)
    own_choice = np.empty(len(matrix), dtype=np.int64)
    wrong_choice = np.empty(len(matrix), dtype=np.int64)
    own_score = np.empty(len(matrix), dtype=np.float32)
    wrong_score = np.empty(len(matrix), dtype=np.float32)
    for label in dict.fromkeys(labels.tolist()):
        rows = np.flatnonzero(labels == label)
        own_ids = ids_by_class[label]
        wrong_ids = np.flatnonzero(owner_array != label)
        local_own = np.argmax(scores[rows][:, own_ids], axis=1)
        local_wrong = np.argmax(scores[rows][:, wrong_ids], axis=1)
        own_choice[rows] = own_ids[local_own]
        wrong_choice[rows] = wrong_ids[local_wrong]
        own_score[rows] = scores[rows, own_choice[rows]]
        wrong_score[rows] = scores[rows, wrong_choice[rows]]
    active = wrong_score > own_score
    updated = prototypes.astype(np.float64, copy=True)
    delta = np.zeros_like(updated)
    counts = np.zeros(len(prototypes), dtype=np.int64)
    if np.any(active):
        active_values = matrix[active].astype(np.float64)
        active_own = own_choice[active]
        active_wrong = wrong_choice[active]
        np.add.at(
            delta,
            active_own,
            active_values - prototypes[active_own].astype(np.float64),
        )
        np.add.at(
            delta,
            active_wrong,
            prototypes[active_wrong].astype(np.float64) - active_values,
        )
        np.add.at(counts, active_own, 1)
        np.add.at(counts, active_wrong, 1)
        changed = counts > 0
        updated[changed] += learning_rate * delta[changed] / counts[changed, None]
    output = np.asarray(updated, dtype=np.float32)
    output.setflags(write=False)
    displacement = np.linalg.norm(
        output.astype(np.float64) - prototypes.astype(np.float64), axis=1
    )
    return output, {
        "algorithm": "batch_lvq_misclassified_own_attract_rival_repel_v1",
        "learning_rate": learning_rate,
        "training_row_count": int(len(matrix)),
        "active_violation_count": int(np.count_nonzero(active)),
        "active_violation_rate": float(np.mean(active)),
        "changed_prototype_count": int(np.count_nonzero(counts)),
        "mean_changed_prototype_displacement": (
            0.0
            if not np.any(counts)
            else float(np.mean(displacement[counts > 0]))
        ),
        "max_prototype_displacement": float(np.max(displacement, initial=0.0)),
        "weights_relearned": False,
        "prototype_owners_changed": False,
        "prototype_count_changed": False,
    }


def crossfit_discriminative_state(
    values: Any,
    target: Any,
    *,
    seed: int,
    max_square_features: int | None,
) -> dict[int, dict[str, Any]]:
    """Derive epoch 0/1/3 states from the same fitted squared FUSED3 fold."""

    matrix, labels = _validate_matrix_target(values, target)
    folds = food101._stratified_folds(labels, n_splits=5, seed=int(seed))
    by_epoch: dict[int, list[dict[str, Any]]] = {epoch: [] for epoch in EPOCHS}
    started = time.perf_counter()
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        lift = fit_squared_feature_lift(
            matrix[train],
            labels[train],
            max_square_features=max_square_features,
        )
        transformed_train = lift.transform(matrix[train])
        transformed_holdout = lift.transform(matrix[holdout])
        k_per_class = fused_food_screen._k_per_class(labels, train)
        selector = fused_food_screen._build_selector(
            "FUSED3", k_per_class, fold_seed
        ).fit(transformed_train, labels[train])
        owners_before = np.array(selector.owners_, copy=True)
        weights_before = np.array(selector.weights_, copy=True)
        centers = np.asarray(selector.centers_, dtype=np.float32)
        chosen = _margin_indices(
            labels[train],
            list(dict.fromkeys(labels[train].tolist())),
            UPDATE_ROWS_PER_CLASS,
            fold_seed,
        )
        update_values = transformed_train[chosen]
        update_labels = labels[train][chosen]
        for epoch in range(0, max(EPOCHS) + 1):
            if epoch in EPOCHS:
                components = component_geometry(
                    transformed_holdout,
                    labels[holdout],
                    centers=centers,
                    owners=selector.owners_,
                    weights=selector.weights_,
                )
                by_epoch[epoch].append(
                    {
                        "fold": int(fold),
                        "fold_seed": fold_seed,
                        **components,
                        "selected_square_feature_count": int(
                            lift.diagnostics["selected_square_feature_count"]
                        ),
                        "lift_state_sha256": lift.state_sha256,
                        "update_row_count": int(len(chosen)),
                        "last_update_diagnostics": (
                            None if epoch == 0 else dict(last_update)
                        ),
                    }
                )
            if epoch == max(EPOCHS):
                break
            centers, last_update = contrastive_lvq_epoch(
                update_values,
                update_labels,
                centers=centers,
                owners=selector.owners_,
                weights=selector.weights_,
                learning_rate=LEARNING_RATE,
            )
        if not np.array_equal(selector.owners_, owners_before):
            raise RuntimeError("LVQ diagnostic mutated prototype owners")
        if not np.array_equal(selector.weights_, weights_before):
            raise RuntimeError("LVQ diagnostic mutated relevance weights")
    elapsed = time.perf_counter() - started
    return {
        epoch: {
            **{
                component: float(
                    np.mean([row[component] for row in by_epoch[epoch]])
                )
                for component in COMPONENTS
            },
            "folds": by_epoch[epoch],
            "shared_total_wall_seconds": float(elapsed),
        }
        for epoch in EPOCHS
    }


def _load_squared_artifact(
    path: os.PathLike[str] | str,
) -> tuple[dict[tuple[str, int, int, str], float], dict[str, str]]:
    root = Path(path)
    manifest = json.loads((root / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_status") != "completed":
        raise RuntimeError("squared-feature artifact is not completed")
    for name, descriptor in manifest.get("files", {}).items():
        if _sha256(root / name) != descriptor.get("sha256"):
            raise RuntimeError("squared-feature artifact hash mismatch")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("study") != SQUARED_STUDY:
        raise RuntimeError("unexpected squared-feature source study")
    if summary.get("original_confirmation_decision_unchanged") is not True:
        raise RuntimeError("squared-feature source changed the frozen decision")
    lookup = {
        (
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        ): float(row["score"])
        for row in summary["rows"]
        if row["candidate_id"] in {base for base, _count in BASE_STATES}
    }
    if len(lookup) != 200:
        raise RuntimeError("squared-feature source grid is incomplete")
    return lookup, {
        "diagnostic_manifest_sha256": _sha256(root / "diagnostic_manifest.json"),
        "summary_sha256": _sha256(root / "summary.json"),
    }


def _lookups(
    selector_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str], float]:
    reference = {
        (
            str(row["backbone"]),
            int(row["replicate_seed"]),
            int(row["budget"]),
            str(row["head"]),
        ): float(row["test_accuracy"])
        for row in reference_rows
        if row["dataset_id"] == DATASET_ID
    }
    if len(reference) != 400:
        raise RuntimeError("frozen Aircraft reference table is incomplete")
    return reference


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


def analyze_discriminative_prototypes(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    squared_artifact: os.PathLike[str] | str,
) -> dict[str, Any]:
    verified = frozen_analysis.verify_completed_artifact(full_artifact)
    selectors, references = statistics.validate_confirmation_inputs(
        verified["selector_rows"], verified["reference_rows"]
    )
    reference = _lookups(selectors, references)
    prior_scores, prior_hashes = _load_squared_artifact(squared_artifact)
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
                for base_state, square_count in BASE_STATES:
                    result = crossfit_discriminative_state(
                        panel["training_values"],
                        panel["training_labels"],
                        seed=int(seed),
                        max_square_features=square_count,
                    )
                    if abs(result[0]["exact_oi"] - prior_scores[identity + (base_state,)]) > 1.0e-12:
                        raise RuntimeError("epoch-zero state differs from squared artifact")
                    for epoch in EPOCHS:
                        state_id = _stage_id(base_state, epoch)
                        state_rows.append(
                            {
                                "dataset_id": DATASET_ID,
                                "backbone": backbone,
                                "replicate": int(replicate),
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                "base_state": base_state,
                                "epoch": int(epoch),
                                "state_id": state_id,
                                **{
                                    component: result[epoch][component]
                                    for component in COMPONENTS
                                },
                                "best_head_envelope_accuracy": float(envelope),
                                "best_head": statistics.HEADS[
                                    int(
                                        np.argmax(
                                            [heads[head] for head in statistics.HEADS]
                                        )
                                    )
                                ],
                                "shared_total_wall_seconds": float(
                                    result[epoch]["shared_total_wall_seconds"]
                                ),
                            }
                        )
                        for fold_row in result[epoch]["folds"]:
                            fold_rows.append(
                                {
                                    "backbone": backbone,
                                    "replicate": int(replicate),
                                    "replicate_seed": int(seed),
                                    "budget": int(budget),
                                    "base_state": base_state,
                                    "epoch": int(epoch),
                                    "state_id": state_id,
                                    **fold_row,
                                }
                            )
    if len(state_rows) != 600 or len(fold_rows) != 3000:
        raise RuntimeError("discriminative-prototype grid is incomplete")

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
            for state_id in STATES:
                states = [row for row in panel if row["state_id"] == state_id]
                for component in RANK_COMPONENTS:
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
    for state_id in STATES:
        for component in RANK_COMPONENTS:
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
        "update_contract": {
            "initial_states": [base for base, _count in BASE_STATES],
            "epochs": list(EPOCHS),
            "learning_rate": LEARNING_RATE,
            "update_rows_per_class": UPDATE_ROWS_PER_CLASS,
            "row_selection": "deterministic capped training-fold bank reused across epochs",
            "active_rows": "nearest rival score strictly greater than nearest own score",
            "own_update": "move nearest own prototype toward active row",
            "rival_update": "move nearest rival prototype away from active row",
            "batch_reduction": "mean signed displacement per touched prototype",
            "weights_relearned": False,
            "owners_changed": False,
            "prototype_count_changed": False,
            "oi_event_rule_changed": False,
        },
        "state_rows": state_rows,
        "fold_rows": fold_rows,
        "ranking_rows": ranking_rows,
        "aggregate_rows": aggregate_rows,
        "squared_source_hashes": prior_hashes,
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "input_hashes": dict(verified["input_hashes"]),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Aircraft discriminative-prototype diagnostic",
        "",
        "Status: **post-outcome descriptive diagnostic only**. The frozen confirmation decision is unchanged.",
        "",
        "| State | Ranking score | Mean regret (pp) | Mean Spearman | Selections |",
        "|---|---|---:|---:|---|",
    ]
    for row in summary["aggregate_rows"]:
        lines.append(
            f"| {row['state_id']} | {row['component']} | {row['mean_regret_pp']:.3f} | "
            f"{row['mean_spearman']:.3f} | `{_canonical_json(row['selection_counts'])}` |"
        )
    lines.extend(
        [
            "",
            "LVQ1/LVQ3 change only prototype centers using training-fold violations. Feature weights, prototype ownership/counts, and exact OI scoring remain fixed. No row authorizes promotion or reselection.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
    squared_artifact: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    summary = analyze_discriminative_prototypes(
        full_artifact, registry_root, squared_artifact
    )
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
        "squared_source_hashes": summary["squared_source_hashes"],
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
            raise RuntimeError("written LVQ artifact hash mismatch")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--squared-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_bundle(
        args.input, args.registry_root, args.squared_artifact, args.output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
