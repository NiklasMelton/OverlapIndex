"""Post-outcome Aircraft diagnostic for FUSED3 plus prototype refinement.

The diagnostic preserves the frozen FUSED3 fit, learned relevance weights,
folds, cohorts, and fixed OI scoring rule.  It then applies the existing
one-pass balanced-median prototype refinement in FUSED3's learned weighted
Euclidean metric.  It is descriptive only and cannot change the failed
Confirmation V2 decision.
"""

from __future__ import annotations

import argparse
import hashlib
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

from experiments.fused3_confirmation_v2 import analysis as frozen_analysis
from experiments.fused3_confirmation_v2 import datasets, statistics
from experiments.fused3_envelope_reanalysis.kmeans_k_sweep import (
    DATASET_ID,
    FOCAL_BACKBONES,
    _canonical_json,
    _csv_bytes,
    _lookups,
    _sha256,
    _spearman,
    constant_k_per_class,
    own_prototype_win_rates,
)
from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen
from experiments.scalable_relevance_kmeans.scalable_v2 import fast_tiled_oi_score
from overlapindex._prototype_refinement import apply_balanced_median_refinement


STUDY = "fused3_confirmation_v2_aircraft_balanced_median_refinement"
STAGE = "post_outcome_exploratory_diagnostic"
STATUS = "descriptive_only_no_promotion"
BASELINE = "FUSED3"
REFINED = "FUSED3-REFINED"
METHODS = (BASELINE, REFINED)


def _array_digest(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _state_sha256(centers: Any, owners: Any, weights: Any) -> str:
    digest = hashlib.sha256()
    _array_digest(digest, "centers", centers)
    _array_digest(digest, "owners", np.asarray(owners, dtype=str))
    _array_digest(digest, "weights", weights)
    return digest.hexdigest()


class _MetricCentroidBackend:
    """Minimal centroid-backend surface used by the canonical refiner."""

    def __init__(self, centers: Any, owners: Any) -> None:
        matrix = np.asarray(centers, dtype=np.float32)
        owner_array = np.asarray(owners)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("metric-space centers must be a finite 2D matrix")
        if owner_array.shape != (len(matrix),):
            raise ValueError("prototype owners must align with centers")
        self._dtype = np.float32
        self._centers = np.array(matrix, dtype=np.float32, copy=True)
        self._center_norms = np.einsum("ij,ij->i", self._centers, self._centers)
        self._cluster_to_class = np.array(owner_array, copy=True)
        self._class_center_ids: dict[Any, list[int]] = {}
        for prototype_id, label in enumerate(self._cluster_to_class.tolist()):
            self._class_center_ids.setdefault(label, []).append(int(prototype_id))
        self._class_center_id_arrays = {
            label: np.asarray(ids, dtype=int)
            for label, ids in self._class_center_ids.items()
        }
        self._class_to_clusters = defaultdict(
            set,
            {label: set(ids) for label, ids in self._class_center_ids.items()},
        )

    @property
    def centers(self) -> np.ndarray:
        return self._centers

    @property
    def cluster_to_class(self) -> np.ndarray:
        return self._cluster_to_class

    @property
    def class_center_id_arrays(self) -> dict[Any, np.ndarray]:
        return self._class_center_id_arrays

    def prepare_score_input(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("refinement score input must be two-dimensional")
        return values

    def score_block_prepared(
        self, X_prepared: Any, ids: Any = None
    ) -> np.ndarray:
        values = self.prepare_score_input(X_prepared)
        if ids is None:
            centers = self._centers
            norms = self._center_norms
        elif isinstance(ids, slice):
            centers = self._centers[ids]
            norms = self._center_norms[ids]
        else:
            selected = np.atleast_1d(np.asarray(ids, dtype=int))
            centers = self._centers[selected]
            norms = self._center_norms[selected]
        scores = np.asarray(values @ centers.T, dtype=np.float32)
        scores -= np.asarray(norms, dtype=np.float32)[None, :] * np.float32(0.5)
        return scores


def refine_fused3_state(
    values: Any,
    target: Any,
    *,
    centers: Any,
    owners: Any,
    weights: Any,
    memory_budget_mb: int = 64,
) -> dict[str, Any]:
    """Apply balanced-median refinement in FUSED3's learned metric space."""

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    prototype_matrix = np.asarray(centers, dtype=np.float32)
    owner_array = np.asarray(owners)
    feature_weights = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or labels.shape != (len(matrix),):
        raise ValueError("training values and scalar labels must align")
    if prototype_matrix.ndim != 2 or prototype_matrix.shape[1] != matrix.shape[1]:
        raise ValueError("centers have the wrong feature shape")
    if owner_array.shape != (len(prototype_matrix),):
        raise ValueError("owners must align with centers")
    if feature_weights.shape != (matrix.shape[1],) or np.any(feature_weights <= 0):
        raise ValueError("weights must be a positive feature vector")
    if not np.isfinite(matrix).all() or not np.isfinite(prototype_matrix).all():
        raise ValueError("values and centers must be finite")

    # FUSED3's scorer is exactly ordinary Euclidean prototype ranking after
    # this float32 coordinate scaling.  Refining in this space therefore uses
    # the same metric that determines its OI events.
    scale = np.asarray(np.sqrt(feature_weights), dtype=np.float32)
    metric_values = np.asarray(matrix * scale, dtype=np.float32)
    metric_centers = np.asarray(prototype_matrix * scale, dtype=np.float32)
    backend = _MetricCentroidBackend(metric_centers, owner_array)
    started = time.perf_counter()
    summary = apply_balanced_median_refinement(
        backend,
        metric_values,
        labels,
        memory_budget_mb=int(memory_budget_mb),
    )
    elapsed = time.perf_counter() - started
    refined_centers = np.array(backend.centers, dtype=np.float32, copy=True)
    refined_owners = np.array(backend.cluster_to_class, copy=True)
    unit_weights = np.ones(matrix.shape[1], dtype=np.float64)
    for array in (metric_values, refined_centers, refined_owners, unit_weights):
        array.setflags(write=False)
    return {
        "metric_values": metric_values,
        "centers": refined_centers,
        "owners": refined_owners,
        "weights": unit_weights,
        "summary": dict(summary),
        "refinement_wall_seconds": float(elapsed),
        "state_sha256": _state_sha256(
            refined_centers, refined_owners, unit_weights
        ),
    }


def crossfit_fused3_refinement(
    values: Any,
    target: Any,
    *,
    seed: int,
    collect_own_win_rates: bool,
    k: int | None = None,
) -> dict[str, Any]:
    """Run paired frozen-FUSED3 and weighted-refinement scores in each fold."""

    matrix = np.asarray(values, dtype=np.float32)
    labels = np.asarray(target)
    if matrix.ndim != 2 or labels.shape != (len(matrix),):
        raise ValueError("matrix and scalar targets must align")
    normalized = food101._row_l2(matrix)
    folds = food101._stratified_folds(labels, n_splits=5, seed=int(seed))
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, holdout) in enumerate(folds):
        fold_seed = int(seed) + int(fold)
        k_per_class = (
            fused_food_screen._k_per_class(labels, train)
            if k is None
            else constant_k_per_class(labels, train, k=int(k))
        )
        selector = fused_food_screen._build_selector(
            BASELINE, k_per_class, fold_seed
        )
        fit_started = time.perf_counter()
        selector.fit(normalized[train], labels[train])
        fit_wall = time.perf_counter() - fit_started
        original_state = fused_food_screen._numerical_state_sha256(selector)

        baseline_started = time.perf_counter()
        baseline_score, baseline_score_diagnostics = (
            selector.score_fixed_with_diagnostics(
                normalized[holdout], labels[holdout]
            )
        )
        baseline_score_wall = time.perf_counter() - baseline_started
        if fused_food_screen._numerical_state_sha256(selector) != original_state:
            raise RuntimeError("baseline score_fixed mutated FUSED3 state")

        refined = refine_fused3_state(
            normalized[train],
            labels[train],
            centers=selector.centers_,
            owners=selector.owners_,
            weights=selector.weights_,
            memory_budget_mb=int(selector.memory_budget_mb),
        )
        # The refiner operates on a detached proxy.  Learned FUSED3 centers,
        # owners, and relevance weights must remain byte-identical.
        if fused_food_screen._numerical_state_sha256(selector) != original_state:
            raise RuntimeError("prototype refinement mutated the fitted FUSED3 state")
        scale = np.asarray(np.sqrt(selector.weights_), dtype=np.float32)
        metric_holdout = np.asarray(normalized[holdout] * scale, dtype=np.float32)
        refined_started = time.perf_counter()
        refined_score, refined_score_diagnostics = fast_tiled_oi_score(
            metric_holdout,
            labels[holdout],
            centers=refined["centers"],
            owners=refined["owners"],
            weights=refined["weights"],
            memory_budget_mb=int(selector.memory_budget_mb),
            row_cap=selector.row_cap,
        )
        refined_score_wall = time.perf_counter() - refined_started
        if fused_food_screen._numerical_state_sha256(selector) != original_state:
            raise RuntimeError("refined score mutated the fitted FUSED3 state")
        if not np.isfinite(baseline_score) or not np.isfinite(refined_score):
            raise RuntimeError("refinement diagnostic emitted a non-finite score")

        baseline_own = (
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
        refined_own = (
            own_prototype_win_rates(
                metric_holdout,
                labels[holdout],
                centers=refined["centers"],
                owners=refined["owners"],
                weights=refined["weights"],
            )
            if collect_own_win_rates
            else {"best_own_win_rate": None, "second_own_win_rate": None}
        )
        summary = refined["summary"]
        fold_rows.append(
            {
                "fold": int(fold),
                "fold_seed": int(fold_seed),
                "train_size": int(len(train)),
                "holdout_size": int(len(holdout)),
                "k_min": int(min(k_per_class.values())),
                "k_max": int(max(k_per_class.values())),
                "baseline_score": float(baseline_score),
                "refined_score": float(refined_score),
                "score_delta": float(refined_score - baseline_score),
                "baseline_best_own_win_rate": baseline_own["best_own_win_rate"],
                "baseline_second_own_win_rate": baseline_own["second_own_win_rate"],
                "refined_best_own_win_rate": refined_own["best_own_win_rate"],
                "refined_second_own_win_rate": refined_own["second_own_win_rate"],
                "fit_wall_seconds": float(fit_wall),
                "baseline_score_wall_seconds": float(baseline_score_wall),
                "refinement_wall_seconds": float(
                    refined["refinement_wall_seconds"]
                ),
                "refined_score_wall_seconds": float(refined_score_wall),
                "baseline_total_wall_seconds": float(fit_wall + baseline_score_wall),
                "refined_total_wall_seconds": float(
                    fit_wall
                    + refined["refinement_wall_seconds"]
                    + refined_score_wall
                ),
                "prototype_count_before": int(summary["prototype_count_before"]),
                "prototype_count_after": int(summary["prototype_count_after"]),
                "eligible_count": int(summary["eligible_count"]),
                "applied_count": int(summary["applied_count"]),
                "skipped_count": int(summary["skipped_count"]),
                "baseline_state_sha256": original_state,
                "refined_state_sha256": str(refined["state_sha256"]),
                "baseline_score_tile_rows": int(
                    baseline_score_diagnostics["max_score_tile_rows"]
                ),
                "refined_score_tile_rows": int(
                    refined_score_diagnostics["max_score_tile_rows"]
                ),
                "fitted_fused3_state_unchanged": True,
            }
        )

    output: dict[str, Any] = {
        "scores": {
            BASELINE: float(np.mean([row["baseline_score"] for row in fold_rows])),
            REFINED: float(np.mean([row["refined_score"] for row in fold_rows])),
        },
        "folds": fold_rows,
    }
    for prefix in ("baseline", "refined"):
        for component in ("best_own_win_rate", "second_own_win_rate"):
            field = f"{prefix}_{component}"
            observed = [row[field] for row in fold_rows if row[field] is not None]
            output[field] = (
                None
                if not observed
                else float(np.mean(np.asarray(observed, dtype=np.float64)))
            )
    output["prototype_count_before"] = float(
        np.mean([row["prototype_count_before"] for row in fold_rows])
    )
    output["prototype_count_after"] = float(
        np.mean([row["prototype_count_after"] for row in fold_rows])
    )
    output["applied_count"] = float(
        np.mean([row["applied_count"] for row in fold_rows])
    )
    output["runtime_ratio"] = float(
        sum(row["refined_total_wall_seconds"] for row in fold_rows)
        / sum(row["baseline_total_wall_seconds"] for row in fold_rows)
    )
    return output


def summarize_refinement(
    state_rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = METHODS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build complete-panel ranking, aggregate, and focal contrast rows."""

    ranking_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for seed in statistics.REPLICATE_SEEDS:
            for method in methods:
                panel = [
                    row
                    for row in state_rows
                    if int(row["budget"]) == budget
                    and int(row["replicate_seed"]) == seed
                    and str(row["candidate_id"]) == method
                ]
                if {str(row["backbone"]) for row in panel} != set(
                    statistics.BACKBONES
                ):
                    raise ValueError("refinement ranking panel is incomplete")
                maximum = max(float(row["score"]) for row in panel)
                selected = [
                    row
                    for row in panel
                    if abs(float(row["score"]) - maximum)
                    <= statistics.SELECTION_TIE_ATOL
                ]
                selected_outcome = float(
                    np.mean(
                        [float(row["best_head_envelope_accuracy"]) for row in selected]
                    )
                )
                best_outcome = max(
                    float(row["best_head_envelope_accuracy"]) for row in panel
                )
                ranking_rows.append(
                    {
                        "budget": int(budget),
                        "replicate_seed": int(seed),
                        "candidate_id": method,
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

    aggregate_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for method in methods:
            ranks = [
                row
                for row in ranking_rows
                if int(row["budget"]) == budget
                and str(row["candidate_id"]) == method
            ]
            states = [
                row
                for row in state_rows
                if int(row["budget"]) == budget
                and str(row["candidate_id"]) == method
            ]
            counts: Counter[str] = Counter()
            for row in ranks:
                for backbone in json.loads(str(row["selected_backbones"])):
                    counts[str(backbone)] += 1
            aggregate_rows.append(
                {
                    "budget": int(budget),
                    "candidate_id": method,
                    "mean_regret_pp": float(np.mean([row["regret_pp"] for row in ranks])),
                    "mean_spearman": float(
                        np.mean(
                            [row["spearman"] for row in ranks if row["spearman"] is not None]
                        )
                    ),
                    "selection_counts": json.dumps(
                        dict(sorted(counts.items())), separators=(",", ":")
                    ),
                    "median_panel_wall_seconds": float(
                        np.median([float(row["total_wall_seconds"]) for row in states])
                    ),
                    "mean_prototype_count": float(
                        np.mean([float(row["prototype_count"]) for row in states])
                    ),
                    "mean_applied_count": float(
                        np.mean([float(row["applied_count"]) for row in states])
                    ),
                }
            )

    focal_rows: list[dict[str, Any]] = []
    for budget in statistics.BUDGETS:
        for method in methods:
            selected = [
                row
                for row in state_rows
                if int(row["budget"]) == budget
                and str(row["candidate_id"]) == method
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
                    "candidate_id": method,
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
                }
            )
    return ranking_rows, aggregate_rows, focal_rows


def analyze_refinement(
    full_artifact: os.PathLike[str] | str,
    registry_root: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Run the paired refinement diagnostic over every Aircraft panel."""

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
                result = crossfit_fused3_refinement(
                    panel["training_values"],
                    panel["training_labels"],
                    seed=int(seed),
                    collect_own_win_rates=backbone in FOCAL_BACKBONES,
                )
                if abs(result["scores"][BASELINE] - frozen_scores[identity]) > 1.0e-12:
                    raise RuntimeError(
                        "baseline recomputation differs from frozen FUSED3"
                    )
                for method in METHODS:
                    refined = method == REFINED
                    state_rows.append(
                        {
                            "dataset_id": DATASET_ID,
                            "backbone": backbone,
                            "replicate": int(replicate),
                            "replicate_seed": int(seed),
                            "budget": int(budget),
                            "candidate_id": method,
                            "score": float(result["scores"][method]),
                            "best_own_win_rate": result[
                                f"{'refined' if refined else 'baseline'}_best_own_win_rate"
                            ],
                            "second_own_win_rate": result[
                                f"{'refined' if refined else 'baseline'}_second_own_win_rate"
                            ],
                            "best_head_envelope_accuracy": float(envelope),
                            "best_head_families": json.dumps(
                                best_heads, separators=(",", ":")
                            ),
                            "total_wall_seconds": float(
                                sum(
                                    row[
                                        "refined_total_wall_seconds"
                                        if refined
                                        else "baseline_total_wall_seconds"
                                    ]
                                    for row in result["folds"]
                                )
                            ),
                            "prototype_count": float(
                                result[
                                    "prototype_count_after"
                                    if refined
                                    else "prototype_count_before"
                                ]
                            ),
                            "applied_count": float(
                                result["applied_count"] if refined else 0.0
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

    if len(state_rows) != 200 or len(fold_rows) != 500:
        raise RuntimeError("completed refinement diagnostic has the wrong row count")
    ranking_rows, aggregate_rows, focal_rows = summarize_refinement(state_rows)
    return {
        "schema_version": 1,
        "study": STUDY,
        "stage": STAGE,
        "artifact_status": "completed",
        "diagnostic_status": STATUS,
        "dataset_id": DATASET_ID,
        "methods": list(METHODS),
        "changed_component_only": (
            "one canonical balanced-median refinement pass in the learned "
            "FUSED3 weighted Euclidean metric"
        ),
        "refinement_order": [
            "fit frozen FUSED3 prototypes",
            "learn frozen FUSED3 relevance weights",
            "scale fit rows and prototypes by sqrt(weight)",
            "apply existing one-pass balanced-median refinement",
            "score heldout rows in the same weighted metric without refitting weights",
        ],
        "state_rows": state_rows,
        "fold_rows": fold_rows,
        "ranking_rows": ranking_rows,
        "aggregate_rows": aggregate_rows,
        "focal_rows": focal_rows,
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
        "# Aircraft FUSED3 balanced-median refinement diagnostic",
        "",
        "Status: **post-outcome descriptive diagnostic only**. The frozen confirmation decision is unchanged.",
        "",
        "The refined candidate starts from the exact frozen FUSED3 state, applies the existing one-pass balanced-median refinement in FUSED3's learned weighted metric, and does not relearn weights.",
        "",
        "| Budget | Candidate | Mean regret (pp) | Mean Spearman | Mean prototypes | Mean splits | Median panel wall (s) | Selections |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["aggregate_rows"]:
        lines.append(
            "| {budget} | {candidate_id} | {mean_regret_pp:.3f} | "
            "{mean_spearman:.3f} | {mean_prototype_count:.1f} | "
            "{mean_applied_count:.1f} | {median_panel_wall_seconds:.3f} | "
            "`{selection_counts}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## DINO versus OpenCLIP",
            "",
            "| Budget | Candidate | OpenCLIP − DINO score |",
            "|---:|---|---:|",
        ]
    )
    for row in summary["focal_rows"]:
        lines.append(
            "| {budget} | {candidate_id} | {OpenCLIP_minus_DINO_score:+.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "A positive OpenCLIP − DINO score means the OI variant still prefers OpenCLIP. Runtime is a paired diagnostic from this run, not a counterbalanced product benchmark.",
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
    summary = analyze_refinement(full_artifact, registry_root)
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
            raise RuntimeError("written refinement artifact hash mismatch")
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
