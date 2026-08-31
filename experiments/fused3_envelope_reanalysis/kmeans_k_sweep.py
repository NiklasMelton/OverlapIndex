"""Post-outcome Aircraft diagnostic for FUSED3 prototypes per class.

This diagnostic varies only the class-owned prototype count.  It preserves the
frozen Aircraft cohorts, folds, normalization, fused three-step prototype
fitter, relevance weights, and exact held-out OI scorer.  It cannot promote,
reselect, or modify the failed Confirmation V2 decision.
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
from experiments.scalable_relevance_kmeans.scalable import _class_ids, _score_kernel


STUDY = "fused3_confirmation_v2_aircraft_kmeans_k_sweep"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
DATASET_ID = "torchvision_fgvc_aircraft"
FOCAL_BACKBONES = ("dinov2-small", "openclip-vit-b-32")
K_GRID_BY_BUDGET: Mapping[int, tuple[int, ...]] = {
    32: (2, 3, 5, 8, 10),
    64: (2, 3, 5, 8, 10, 16, 20),
}
CURRENT_K_BY_BUDGET: Mapping[int, int] = {32: 5, 64: 10}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty diagnostic table")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("Spearman inputs must be aligned one-dimensional vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Spearman inputs must be finite")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    value = float(
        np.corrcoef(
            rankdata(x, method="average"), rankdata(y, method="average")
        )[0, 1]
    )
    return value if np.isfinite(value) else None


def validate_k_grid(
    grid: Mapping[int, Sequence[int]] = K_GRID_BY_BUDGET,
) -> dict[int, tuple[int, ...]]:
    if set(grid) != set(statistics.BUDGETS):
        raise ValueError("k grid must have exactly the frozen budget keys")
    validated: dict[int, tuple[int, ...]] = {}
    for budget in statistics.BUDGETS:
        values = tuple(grid[budget])
        if (
            not values
            or any(type(value) is not int or value < 2 for value in values)
            or values != tuple(sorted(set(values)))
        ):
            raise ValueError("each k grid must be sorted unique integers >= 2")
        if CURRENT_K_BY_BUDGET[budget] not in values:
            raise ValueError("each k grid must contain its frozen recipe value")
        validated[int(budget)] = values
    return validated


def constant_k_per_class(
    labels: Any, train: Any, *, k: int
) -> dict[Any, int]:
    """Return one validated K for every first-observed training-fold class."""

    if type(k) is not int or k < 2:
        raise ValueError("k must be an integer >= 2")
    target = np.asarray(labels)
    train_ids = np.asarray(train, dtype=np.int64)
    encoded, classes = food101._stratification_encoding(target)
    train_encoded = encoded[train_ids]
    counts = [
        int(np.count_nonzero(train_encoded == position))
        for position in range(len(classes))
    ]
    if not counts or min(counts) < k:
        raise ValueError("k exceeds the smallest training-fold class")
    return {label: int(k) for label in classes}


def own_prototype_win_rates(
    values: Any,
    target: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
) -> dict[str, float]:
    """Measure best- and second-own wins without changing fitted state."""

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    prototypes = np.asarray(centers, dtype=np.float32)
    owner_array = np.asarray(owners)
    feature_weights = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or labels.shape != (len(matrix),):
        raise ValueError("evaluation values and targets must align")
    if prototypes.ndim != 2 or prototypes.shape[1] != matrix.shape[1]:
        raise ValueError("prototype feature shape mismatch")
    if owner_array.shape != (len(prototypes),):
        raise ValueError("prototype owners must align with prototypes")
    if feature_weights.shape != (matrix.shape[1],) or np.any(feature_weights <= 0):
        raise ValueError("weights must be a positive feature vector")
    ids_by_class = _class_ids(owner_array)
    classes = list(dict.fromkeys(labels.tolist()))
    if set(classes) != set(ids_by_class):
        raise ValueError("evaluation and prototype class sets must match")
    if any(len(ids_by_class[label]) < 2 for label in classes):
        raise ValueError("own-win diagnostics require at least two prototypes per class")

    scores = _score_kernel(matrix, prototypes, feature_weights)
    best_rates: list[float] = []
    second_rates: list[float] = []
    for label in classes:
        rows = np.flatnonzero(labels == label)
        own_ids = ids_by_class[label]
        own_scores = scores[rows][:, own_ids]
        best_own = np.max(own_scores, axis=1)
        second_own = np.partition(own_scores, kth=-2, axis=1)[:, -2]
        wrong_ids = np.concatenate(
            [ids_by_class[other] for other in classes if other != label]
        )
        best_wrong = np.max(scores[rows][:, wrong_ids], axis=1)
        best_rates.append(float(np.mean(best_wrong <= best_own)))
        second_rates.append(float(np.mean(best_wrong <= second_own)))
    return {
        "best_own_win_rate": float(np.mean(best_rates)),
        "second_own_win_rate": float(np.mean(second_rates)),
    }


def crossfit_fused3_at_k(
    values: Any,
    target: Any,
    *,
    seed: int,
    k: int,
    collect_own_win_rates: bool,
) -> dict[str, Any]:
    """Run exact frozen FUSED3 folds with only prototypes-per-class changed."""

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    if matrix.ndim != 2 or labels.shape != (len(matrix),):
        raise ValueError("matrix and scalar targets must align")
    normalized = food101._row_l2(matrix)
    folds = food101._stratified_folds(labels, n_splits=5, seed=int(seed))
    started = time.perf_counter()
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        k_per_class = constant_k_per_class(labels, train, k=k)
        selector = fused_food_screen._build_selector(
            "FUSED3", k_per_class, fold_seed
        )
        fit_started = time.perf_counter()
        selector.fit(normalized[train], labels[train])
        fit_wall = time.perf_counter() - fit_started
        state_before = fused_food_screen._numerical_state_sha256(selector)
        score_started = time.perf_counter()
        score, score_diagnostics = selector.score_fixed_with_diagnostics(
            normalized[holdout], labels[holdout]
        )
        score_wall = time.perf_counter() - score_started
        state_after = fused_food_screen._numerical_state_sha256(selector)
        if state_before != state_after:
            raise RuntimeError("k-sweep score_fixed mutated FUSED3 fitted state")
        if not np.isfinite(score):
            raise RuntimeError("k-sweep emitted a non-finite OI score")
        own_rates = (
            own_prototype_win_rates(
                normalized[holdout],
                labels[holdout],
                centers=selector.centers_,
                owners=selector.owners_,
                weights=selector.weights_,
            )
            if collect_own_win_rates
            else {"best_own_win_rate": None, "second_own_win_rate": None}
        )
        diagnostics = dict(selector.diagnostics_ or {})
        fold_rows.append(
            {
                "fold": int(fold),
                "fold_seed": int(fold_seed),
                "k_per_class": int(k),
                "train_size": int(len(train)),
                "holdout_size": int(len(holdout)),
                "score": float(score),
                "best_own_win_rate": own_rates["best_own_win_rate"],
                "second_own_win_rate": own_rates["second_own_win_rate"],
                "fit_wall_seconds": float(fit_wall),
                "score_fixed_wall_seconds": float(score_wall),
                "prototype_count": int(len(selector.centers_)),
                "empty_prototype_count": int(
                    diagnostics["prototype_fit"]["empty_prototype_count"]
                ),
                "weight_condition": float(diagnostics["weight_condition"]),
                "positive_margin_feature_count": int(
                    diagnostics["positive_margin_feature_count"]
                ),
                "state_sha256": str(diagnostics["state_sha256"]),
                "score_tile_rows": int(score_diagnostics["max_score_tile_rows"]),
                "state_unchanged_after_score_fixed": True,
            }
        )
    result: dict[str, Any] = {
        "score": float(np.mean([row["score"] for row in fold_rows])),
        "total_wall_seconds": float(time.perf_counter() - started),
        "folds": fold_rows,
    }
    for field in ("best_own_win_rate", "second_own_win_rate"):
        values_for_field = [row[field] for row in fold_rows if row[field] is not None]
        result[field] = (
            None
            if not values_for_field
            else float(np.mean(np.asarray(values_for_field, dtype=np.float64)))
        )
    return result


def _lookups(
    selector_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int, int], float],
    dict[tuple[str, int, int, str], float],
]:
    frozen: dict[tuple[str, int, int], float] = {}
    references: dict[tuple[str, int, int, str], float] = {}
    for row in selector_rows:
        if row["dataset_id"] == DATASET_ID and row["candidate_id"] == "FUSED3":
            key = (
                str(row["backbone"]),
                int(row["replicate_seed"]),
                int(row["budget"]),
            )
            if key in frozen:
                raise ValueError("duplicate frozen Aircraft FUSED3 row")
            frozen[key] = float(row["score"])
    for row in reference_rows:
        if row["dataset_id"] == DATASET_ID:
            key = (
                str(row["backbone"]),
                int(row["replicate_seed"]),
                int(row["budget"]),
                str(row["head"]),
            )
            if key in references:
                raise ValueError("duplicate frozen Aircraft reference row")
            references[key] = float(row["test_accuracy"])
    if len(frozen) != 100 or len(references) != 400:
        raise ValueError("Aircraft k sweep requires exact frozen 100/400 rows")
    return frozen, references


def summarize_k_sweep(
    state_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create complete-panel ranking, aggregate, and focal contrast tables."""

    rows = [dict(row) for row in state_rows]
    expected = sum(
        len(K_GRID_BY_BUDGET[budget])
        * len(statistics.BACKBONES)
        * len(statistics.REPLICATE_SEEDS)
        for budget in statistics.BUDGETS
    )
    if len(rows) != expected:
        raise ValueError("k-sweep state grid is incomplete")
    identities = {
        (
            int(row["budget"]),
            int(row["replicate_seed"]),
            int(row["k_per_class"]),
            str(row["backbone"]),
        )
        for row in rows
    }
    if len(identities) != expected:
        raise ValueError("k-sweep state grid contains duplicate identities")

    ranking_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for seed in statistics.REPLICATE_SEEDS:
            for k in K_GRID_BY_BUDGET[budget]:
                panel = [
                    row
                    for row in rows
                    if int(row["budget"]) == budget
                    and int(row["replicate_seed"]) == seed
                    and int(row["k_per_class"]) == k
                ]
                if {str(row["backbone"]) for row in panel} != set(
                    statistics.BACKBONES
                ):
                    raise ValueError("k-sweep backbone panel is incomplete")
                maximum = max(float(row["score"]) for row in panel)
                selected = [
                    row
                    for row in panel
                    if abs(float(row["score"]) - maximum)
                    <= statistics.SELECTION_TIE_ATOL
                ]
                selected_outcome = float(
                    np.mean(
                        [
                            float(row["best_head_envelope_accuracy"])
                            for row in selected
                        ]
                    )
                )
                best_outcome = max(
                    float(row["best_head_envelope_accuracy"]) for row in panel
                )
                ranking_rows.append(
                    {
                        "budget": int(budget),
                        "replicate_seed": int(seed),
                        "k_per_class": int(k),
                        "current_recipe": bool(k == CURRENT_K_BY_BUDGET[budget]),
                        "selected_backbones": json.dumps(
                            sorted(str(row["backbone"]) for row in selected),
                            separators=(",", ":"),
                        ),
                        "selected_envelope_accuracy": selected_outcome,
                        "best_envelope_accuracy": best_outcome,
                        "regret_pp": float(100.0 * (best_outcome - selected_outcome)),
                        "spearman": _spearman(
                            [float(row["score"]) for row in panel],
                            [
                                float(row["best_head_envelope_accuracy"])
                                for row in panel
                            ],
                        ),
                    }
                )

    focal_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for k in K_GRID_BY_BUDGET[budget]:
            selected = [
                row
                for row in rows
                if int(row["budget"]) == budget
                and int(row["k_per_class"]) == k
                and str(row["backbone"]) in FOCAL_BACKBONES
            ]
            by_backbone = {
                backbone: [
                    row for row in selected if str(row["backbone"]) == backbone
                ]
                for backbone in FOCAL_BACKBONES
            }
            focal_rows.append(
                {
                    "budget": int(budget),
                    "k_per_class": int(k),
                    "current_recipe": bool(k == CURRENT_K_BY_BUDGET[budget]),
                    "DINO_mean_score": float(
                        np.mean([row["score"] for row in by_backbone[FOCAL_BACKBONES[0]]])
                    ),
                    "OpenCLIP_mean_score": float(
                        np.mean([row["score"] for row in by_backbone[FOCAL_BACKBONES[1]]])
                    ),
                    "OpenCLIP_minus_DINO_score": float(
                        np.mean([row["score"] for row in by_backbone[FOCAL_BACKBONES[1]]])
                        - np.mean([row["score"] for row in by_backbone[FOCAL_BACKBONES[0]]])
                    ),
                    **{
                        f"{prefix}_{field}": float(
                            np.mean(
                                [
                                    row[field]
                                    for row in by_backbone[backbone]
                                    if row[field] is not None
                                ]
                            )
                        )
                        for prefix, backbone in (
                            ("DINO", FOCAL_BACKBONES[0]),
                            ("OpenCLIP", FOCAL_BACKBONES[1]),
                        )
                        for field in (
                            "best_own_win_rate",
                            "second_own_win_rate",
                        )
                    },
                }
            )

    aggregate_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        current_times = [
            float(row["total_wall_seconds"])
            for row in rows
            if int(row["budget"]) == budget
            and int(row["k_per_class"]) == CURRENT_K_BY_BUDGET[budget]
        ]
        current_median = float(np.median(current_times))
        for k in K_GRID_BY_BUDGET[budget]:
            ranks = [
                row
                for row in ranking_rows
                if int(row["budget"]) == budget
                and int(row["k_per_class"]) == k
            ]
            states = [
                row
                for row in rows
                if int(row["budget"]) == budget
                and int(row["k_per_class"]) == k
            ]
            counts: Counter[str] = Counter()
            for row in ranks:
                for backbone in json.loads(str(row["selected_backbones"])):
                    counts[str(backbone)] += 1
            focal = next(
                row
                for row in focal_rows
                if int(row["budget"]) == budget
                and int(row["k_per_class"]) == k
            )
            median_time = float(
                np.median([float(row["total_wall_seconds"]) for row in states])
            )
            aggregate_rows.append(
                {
                    "budget": int(budget),
                    "k_per_class": int(k),
                    "current_recipe": bool(k == CURRENT_K_BY_BUDGET[budget]),
                    "mean_regret_pp": float(np.mean([row["regret_pp"] for row in ranks])),
                    "mean_spearman": float(
                        np.mean([row["spearman"] for row in ranks if row["spearman"] is not None])
                    ),
                    "selection_counts": json.dumps(
                        dict(sorted(counts.items())), separators=(",", ":")
                    ),
                    "OpenCLIP_minus_DINO_score": float(
                        focal["OpenCLIP_minus_DINO_score"]
                    ),
                    "median_panel_wall_seconds": median_time,
                    "time_ratio_to_current": float(median_time / current_median),
                }
            )
    return ranking_rows, aggregate_rows, focal_rows


def analyze_k_sweep(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Run the complete frozen Aircraft panel over the closed K grid."""

    grid = validate_k_grid()
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
                    aircraft,
                    backbone,
                    int(seed),
                    int(budget),
                    registry_root=root,
                )
                identity = (backbone, int(seed), int(budget))
                heads = {
                    head: reference[identity + (head,)] for head in statistics.HEADS
                }
                envelope = max(heads.values())
                best_heads = [
                    head
                    for head in statistics.HEADS
                    if abs(heads[head] - envelope) <= statistics.SELECTION_TIE_ATOL
                ]
                for k in grid[int(budget)]:
                    result = crossfit_fused3_at_k(
                        panel["training_values"],
                        panel["training_labels"],
                        seed=int(seed),
                        k=int(k),
                        collect_own_win_rates=backbone in FOCAL_BACKBONES,
                    )
                    if k == CURRENT_K_BY_BUDGET[budget] and abs(
                        result["score"] - frozen_scores[identity]
                    ) > 1.0e-12:
                        raise RuntimeError(
                            "current-k recomputation differs from frozen FUSED3"
                        )
                    state_rows.append(
                        {
                            "dataset_id": DATASET_ID,
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "k_per_class": int(k),
                            "current_recipe": bool(
                                k == CURRENT_K_BY_BUDGET[budget]
                            ),
                            "score": float(result["score"]),
                            "best_own_win_rate": result["best_own_win_rate"],
                            "second_own_win_rate": result["second_own_win_rate"],
                            "best_head_envelope_accuracy": float(envelope),
                            "best_head_families": json.dumps(
                                best_heads, separators=(",", ":")
                            ),
                            "total_wall_seconds": float(
                                result["total_wall_seconds"]
                            ),
                        }
                    )
                    for row in result["folds"]:
                        fold_rows.append(
                            {
                                "backbone": backbone,
                                "replicate": int(replicate),
                                "replicate_seed": int(seed),
                                "budget": int(budget),
                                **row,
                            }
                        )

    expected_states = sum(
        len(grid[budget])
        * len(statistics.BACKBONES)
        * len(statistics.REPLICATE_SEEDS)
        for budget in statistics.BUDGETS
    )
    if len(state_rows) != expected_states or len(fold_rows) != 5 * expected_states:
        raise RuntimeError("completed k-sweep grid has the wrong row count")
    ranking_rows, aggregate_rows, focal_rows = summarize_k_sweep(state_rows)
    best_by_regret = {
        str(budget): min(
            [row for row in aggregate_rows if row["budget"] == budget],
            key=lambda row: (row["mean_regret_pp"], -row["mean_spearman"], row["k_per_class"]),
        )["k_per_class"]
        for budget in statistics.BUDGETS
    }
    best_by_spearman = {
        str(budget): max(
            [row for row in aggregate_rows if row["budget"] == budget],
            key=lambda row: (row["mean_spearman"], -row["mean_regret_pp"], -row["k_per_class"]),
        )["k_per_class"]
        for budget in statistics.BUDGETS
    }
    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "dataset_id": DATASET_ID,
        "k_grid_by_budget": {str(key): list(value) for key, value in grid.items()},
        "current_k_by_budget": {
            str(key): int(value) for key, value in CURRENT_K_BY_BUDGET.items()
        },
        "changed_parameter_only": "class_owned_prototypes_per_class",
        "state_rows": state_rows,
        "fold_rows": fold_rows,
        "ranking_rows": ranking_rows,
        "aggregate_rows": aggregate_rows,
        "focal_rows": focal_rows,
        "post_outcome_best_k_by_regret": best_by_regret,
        "post_outcome_best_k_by_spearman": best_by_spearman,
        "post_outcome_warning": (
            "best-k summaries use observed Aircraft outcomes and are mechanism "
            "diagnostics, not a valid tuning or selection rule"
        ),
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


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Aircraft FUSED3 prototypes-per-class sweep",
        "",
        "Status: **post-outcome descriptive diagnostic only**. The frozen confirmation decision is unchanged.",
        "",
        "Only the class-owned prototype count changes. Cohorts, folds, normalization, fused three-step fitting, relevance weighting, and exact fixed OI scoring remain unchanged.",
        "",
        "| Budget | K/class | Frozen recipe | Mean regret (pp) | Mean Spearman | OpenCLIP − DINO score | Median time ratio | Selection counts |",
        "|---:|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["aggregate_rows"]:
        lines.append(
            "| {budget} | {k_per_class} | {current} | {mean_regret_pp:.3f} | "
            "{mean_spearman:.3f} | {OpenCLIP_minus_DINO_score:+.4f} | "
            "{time_ratio_to_current:.2f}× | `{selection_counts}` |".format(
                **row, current="yes" if row["current_recipe"] else ""
            )
        )
    lines.extend(
        [
            "",
            "## Post-outcome capacity check",
            "",
            f"- Lowest-regret K by budget: `{_canonical_json(summary['post_outcome_best_k_by_regret'])}`",
            f"- Highest-Spearman K by budget: `{_canonical_json(summary['post_outcome_best_k_by_spearman'])}`",
            "",
            "These best-K labels use observed Aircraft outcomes. They can diagnose whether the frozen K was capacity-limited, but cannot authorize retuning or promotion.",
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
    summary = analyze_k_sweep(full_artifact, registry_root)
    payloads = {
        "summary.json": (_canonical_json(summary) + "\n").encode("utf-8"),
        "state_rows.csv": _csv_bytes(summary["state_rows"]),
        "fold_rows.csv": _csv_bytes(summary["fold_rows"]),
        "ranking_rows.csv": _csv_bytes(summary["ranking_rows"]),
        "aggregate_rows.csv": _csv_bytes(summary["aggregate_rows"]),
        "focal_rows.csv": _csv_bytes(summary["focal_rows"]),
        "report.md": render_report(summary).encode("utf-8"),
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
            raise RuntimeError("written k-sweep artifact hash mismatch")
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
