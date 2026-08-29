"""Paired full-grid analysis for the locked FUSED3 Food-101 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_analysis as base_analysis
from experiments.scalable_relevance_kmeans import fused_food_full as full


HEADS = ("linear", "quadratic", "knn", "rbf")
NONLINEAR_HEADS = ("quadratic", "knn", "rbf")
METHODS = ("FUSED3", "REFINED-OI", "LP-FULL", "REFINED-OI-ARCHIVED")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_829


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, full.canonical_json(value) + "\n")


def _is_finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(value) and float(value) >= 0.0


def _validate_fold(
    fold: Mapping[str, Any], *, candidate: str, replicate: int, fold_index: int
) -> None:
    if int(fold.get("fold", -1)) != fold_index:
        raise RuntimeError("full Food fold index mismatch")
    if not _is_finite_nonnegative(fold.get("score")):
        raise RuntimeError("full Food fold score is invalid")
    identity = fold.get("candidate_config_identity")
    digest = fold.get("candidate_config_sha256")
    if not isinstance(identity, Mapping) or full._hash_payload(identity) != digest:
        raise RuntimeError("full Food fold configuration hash mismatch")
    if candidate == "FUSED3":
        if identity.get("candidate_id") != "FUSED3":
            raise RuntimeError("FUSED3 fold identity mismatch")
        if identity.get("spec") != full.screen._candidate_specs()["FUSED3"]:
            raise RuntimeError("FUSED3 fold spec drift")
        if int(identity.get("fold_seed", -1)) != full.SEED + int(replicate) + fold_index:
            raise RuntimeError("FUSED3 fold seed drift")
        k = identity.get("k_per_class")
        if not isinstance(k, Mapping) or len(k) != 40 or set(k.values()) != {10}:
            raise RuntimeError("FUSED3 k-per-class drift")
        if not base_analysis._is_sha256(fold.get("numerical_state_sha256")):
            raise RuntimeError("FUSED3 numerical state hash missing")
        if fold.get("state_unchanged_after_score_fixed") is not True:
            raise RuntimeError("FUSED3 score_fixed state invariant failed")
        for field in (
            "fit_wall_seconds",
            "fit_cpu_seconds",
            "score_fixed_wall_seconds",
            "score_fixed_cpu_seconds",
        ):
            if not _is_finite_nonnegative(fold.get(field)):
                raise RuntimeError(f"FUSED3 fold has invalid {field}")
        return
    upstream_id = "B" if candidate == "REFINED-OI" else candidate
    food101._validate_fold_recipe(upstream_id, fold, error_prefix="full Food fold")
    expected_panel_seed = full.SEED + int(replicate)
    if candidate == "REFINED-OI":
        oi_kwargs = identity.get("overlap_index_kwargs")
        kmeans_kwargs = (
            oi_kwargs.get("kmeans_kwargs") if isinstance(oi_kwargs, Mapping) else None
        )
        if (
            int(fold.get("seed", -1)) != expected_panel_seed + fold_index
            or not isinstance(kmeans_kwargs, Mapping)
            or int(kmeans_kwargs.get("random_state", -1))
            != expected_panel_seed + fold_index
            or identity.get("candidate_id") != "B"
            or identity.get("prototype_refinement") is not True
        ):
            raise RuntimeError("fresh refined-OI frozen seed/recipe drift")
    elif (
        int(fold.get("split_seed", -1)) != expected_panel_seed
        or int(fold.get("model_random_state", -1)) != expected_panel_seed
        or identity.get("candidate_id") != "LP-FULL"
    ):
        raise RuntimeError("LP-FULL frozen seed/recipe drift")
    timing_fields = (
        ("fit_wall_seconds", "fit_cpu_seconds", "predict_wall_seconds", "predict_cpu_seconds")
        if candidate == "LP-FULL"
        else (
            "fit_wall_seconds",
            "fit_cpu_seconds",
            "score_fixed_wall_seconds",
            "score_fixed_cpu_seconds",
        )
    )
    for field in timing_fields:
        if not _is_finite_nonnegative(fold.get(field)):
            raise RuntimeError(f"{candidate} fold has invalid {field}")


def verify_artifact(input_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    manifest = _read_json(input_dir / "manifest.json")
    raw_path = input_dir / "raw_results.json"
    raw = _read_json(raw_path)
    if manifest.get("artifact_status") != "completed" or raw.get("artifact_status") != "completed":
        raise RuntimeError("full Food artifact is not completed")
    if manifest.get("stage") != "full" or raw.get("stage") != "full":
        raise RuntimeError("full Food artifact stage mismatch")
    if manifest.get("raw_results_sha256") != full.sha256_path(raw_path):
        raise RuntimeError("full Food raw-results hash mismatch")
    if manifest.get("run_identity") != raw.get("run_identity"):
        raise RuntimeError("full Food manifest/raw identity mismatch")
    identity = raw.get("run_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("full Food run identity is malformed")
    identity_sha = full._hash_payload(identity)
    if (
        raw.get("run_identity_sha256") != identity_sha
        or manifest.get("run_identity_sha256") != identity_sha
    ):
        raise RuntimeError("full Food run identity hash mismatch")
    protocol_sha, _protocol = full.validate_protocol()
    if identity.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("full Food protocol identity mismatch")
    if identity.get("source_identity") != full.source_identity():
        raise RuntimeError("full Food executed source identity no longer matches")
    if identity.get("environment") != full.EXPECTED_ENVIRONMENT:
        raise RuntimeError("full Food executed environment mismatch")
    expected_counts = {
        "panels": 600,
        "selector_rows": 1800,
        "archived_refined_rows": 600,
        "reference_rows": 600,
        "lp_parity_rows": 600,
        "refined_historical_delta_rows": 600,
        "screen_parity_rows": 30,
        "checkpoints": 600,
    }
    if manifest.get("counts") != expected_counts:
        raise RuntimeError("full Food terminal counts mismatch")

    selector_rows = raw.get("selector_rows")
    expected_keys = {
        (model, replicate, arm, budget, method)
        for model in full.MODELS
        for replicate in full.REPLICATES
        for arm in full.ARMS
        for budget in full.BUDGETS
        for method in full.METHODS
    }
    if not isinstance(selector_rows, list) or len(selector_rows) != 1800:
        raise RuntimeError("full Food executed selector grid is incomplete")
    observed = set()
    panel_positions = {
        (model, replicate, arm, budget): position
        for position, (model, replicate, arm, budget) in enumerate(
            (model, replicate, arm, budget)
            for model in full.MODELS
            for replicate in full.REPLICATES
            for arm in full.ARMS
            for budget in full.BUDGETS
        )
    }
    for row in selector_rows:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            raise RuntimeError("full Food contains malformed/non-ok selector row")
        key = (
            str(row.get("model")),
            int(row.get("replicate", -1)),
            str(row.get("arm")),
            int(row.get("budget", -1)),
            str(row.get("candidate_id")),
        )
        if key in observed:
            raise RuntimeError("full Food duplicates an executed selector row")
        observed.add(key)
        order = full._method_order(panel_positions[key[:4]])
        if (
            row.get("execution_order") != list(order)
            or int(row.get("execution_position", -1)) != order.index(key[-1])
        ):
            raise RuntimeError("full Food execution schedule drift")
        expected_refinement = key[-1] == "REFINED-OI"
        if row.get("prototype_refinement_enabled") is not expected_refinement:
            raise RuntimeError("full Food strict refinement identity mismatch")
        if row.get("warmup_excluded") is not True:
            raise RuntimeError("full Food row includes warmup")
        for field in (
            "score",
            "fit_wall_seconds",
            "fit_cpu_seconds",
            "score_fixed_wall_seconds",
            "score_fixed_cpu_seconds",
            "total_wall_seconds",
            "total_cpu_seconds",
        ):
            if not _is_finite_nonnegative(row.get(field)):
                raise RuntimeError(f"full Food row has invalid {field}")
        if float(row["total_wall_seconds"]) <= 0.0:
            raise RuntimeError("full Food row has non-positive outer wall time")
        folds = row.get("folds")
        if not isinstance(folds, list) or len(folds) != full.FOLDS:
            raise RuntimeError("full Food fold grid is incomplete")
        for fold_index, fold in enumerate(folds):
            if not isinstance(fold, Mapping):
                raise RuntimeError("full Food fold row is malformed")
            _validate_fold(
                fold, candidate=key[-1], replicate=key[1], fold_index=fold_index
            )
    if observed != expected_keys:
        raise RuntimeError("full Food executed selector identities mismatch")

    archived = raw.get("archived_refined_rows")
    archived_keys = {
        (model, replicate, arm, budget, "REFINED-OI-ARCHIVED")
        for model in full.MODELS
        for replicate in full.REPLICATES
        for arm in full.ARMS
        for budget in full.BUDGETS
    }
    if not isinstance(archived, list) or len(archived) != 600:
        raise RuntimeError("full Food archived refined grid is incomplete")
    observed_archived = set()
    for row in archived:
        key = (
            str(row.get("model")),
            int(row.get("replicate", -1)),
            str(row.get("arm")),
            int(row.get("budget", -1)),
            str(row.get("candidate_id")),
        )
        if key in observed_archived or not _is_finite_nonnegative(row.get("score")):
            raise RuntimeError("full Food archived refined row is malformed/duplicated")
        observed_archived.add(key)
    if observed_archived != archived_keys:
        raise RuntimeError("full Food archived refined identities mismatch")

    references = raw.get("reference_rows")
    expected_reference = {
        (model, replicate, arm, head)
        for model in full.MODELS
        for replicate in full.REPLICATES
        for arm in full.ARMS
        for head in HEADS
    }
    if not isinstance(references, list) or len(references) != 600:
        raise RuntimeError("full Food reference grid is incomplete")
    reference_keys = set()
    for row in references:
        key = (
            str(row.get("backbone")),
            int(row.get("replicate", -1)),
            str(row.get("arm")),
            str(row.get("head")),
        )
        if key in reference_keys or not _is_finite_nonnegative(row.get("test_accuracy")):
            raise RuntimeError("full Food reference row is malformed/duplicated")
        reference_keys.add(key)
    if reference_keys != expected_reference:
        raise RuntimeError("full Food reference identities mismatch")

    lp = raw.get("lp_parity")
    screen_parity = raw.get("screen_parity")
    refined = raw.get("refined_historical_deltas")
    if not isinstance(lp, list) or len(lp) != 600 or any(
        not isinstance(row, Mapping)
        or row.get("exact") is not True
        or float(row.get("delta", np.nan)) != 0.0
        for row in lp
    ):
        raise RuntimeError("full Food LP archived parity failed")
    if not isinstance(screen_parity, list) or len(screen_parity) != 30 or any(
        not isinstance(row, Mapping)
        or row.get("exact") is not True
        or float(row.get("delta", np.nan)) != 0.0
        for row in screen_parity
    ):
        raise RuntimeError("full Food locked-screen FUSED3 parity failed")
    if not isinstance(refined, list) or len(refined) != 600 or any(
        not isinstance(row, Mapping) or not np.isfinite(float(row.get("delta", np.nan)))
        for row in refined
    ):
        raise RuntimeError("full Food refined historical diagnostics malformed")
    determinism = raw.get("determinism")
    if not isinstance(determinism, Mapping) or set(determinism) != set(full.METHODS):
        raise RuntimeError("full Food determinism evidence incomplete")
    for method, record in determinism.items():
        if (
            not isinstance(record, Mapping)
            or record.get("candidate_id") != method
            or record.get("exact") is not True
            or record.get("runtime_fields_excluded") is not True
            or record.get("first_signature_sha256") != record.get("second_signature_sha256")
        ):
            raise RuntimeError("full Food determinism descriptor malformed")
    return raw


def _metric_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    selectors = {
        (
            str(row["model"]),
            int(row["replicate"]),
            str(row["arm"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        ): float(row["score"])
        for row in [*raw["selector_rows"], *raw["archived_refined_rows"]]
    }
    references = {
        (
            str(row["backbone"]),
            int(row["replicate"]),
            str(row["arm"]),
            str(row["head"]),
        ): float(row["test_accuracy"])
        for row in raw["reference_rows"]
    }
    output = []
    for replicate in full.REPLICATES:
        for arm in full.ARMS:
            for budget in full.BUDGETS:
                for head in HEADS:
                    outcomes = np.asarray(
                        [references[(model, replicate, arm, head)] for model in full.MODELS]
                    )
                    for method in METHODS:
                        scores = np.asarray(
                            [
                                selectors[(model, replicate, arm, budget, method)]
                                for model in full.MODELS
                            ]
                        )
                        output.append(
                            {
                                "replicate": int(replicate),
                                "arm": arm,
                                "budget": int(budget),
                                "head": head,
                                "candidate_id": method,
                                **base_analysis._selection_metrics(scores, outcomes),
                            }
                        )
    return output


def _rank_auc_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Integrate each complete four-budget Spearman curve in log2 space."""

    log_budgets = np.log2(np.asarray(full.BUDGETS, dtype=np.float64))
    normalized = (log_budgets - log_budgets[0]) / (log_budgets[-1] - log_budgets[0])
    output = []
    for replicate in full.REPLICATES:
        for arm in full.ARMS:
            for head in HEADS:
                for method in METHODS:
                    rows = sorted(
                        (
                            row
                            for row in metrics
                            if int(row["replicate"]) == replicate
                            and row["arm"] == arm
                            and row["head"] == head
                            and row["candidate_id"] == method
                        ),
                        key=lambda row: int(row["budget"]),
                    )
                    if [int(row["budget"]) for row in rows] != list(full.BUDGETS):
                        raise RuntimeError("rank AUC requires the complete frozen budget curve")
                    spearman = [row["spearman"] for row in rows]
                    if any(value is None for value in spearman):
                        value = None
                        status = "undefined"
                    else:
                        y = np.asarray(spearman, dtype=np.float64)
                        value = float(
                            np.sum(
                                np.diff(normalized)
                                * (y[:-1] + y[1:])
                                * np.float64(0.5)
                            )
                        )
                        status = "defined"
                    output.append(
                        {
                            "replicate": int(replicate),
                            "arm": arm,
                            "head": head,
                            "candidate_id": method,
                            "rank_auc": value,
                            "status": status,
                            "budgets": list(full.BUDGETS),
                        }
                    )
    return output


def _paired_interval(
    values: Sequence[float], *, key: str, statistic: Callable[[np.ndarray], float]
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (5,) or not np.all(np.isfinite(array)):
        raise RuntimeError(f"paired interval {key} requires five finite replicate blocks")
    offset = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, offset]))
    draws = rng.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, array.size))
    distribution = np.asarray([statistic(array[index]) for index in draws], dtype=np.float64)
    return {
        "estimate": float(statistic(array)),
        "lower95": float(np.quantile(distribution, 0.025)),
        "upper95": float(np.quantile(distribution, 0.975)),
        "replicate_values": array.tolist(),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def _ranking_contrast(
    metrics: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    head: str,
    candidate: str,
    comparator: str,
    metric: str,
) -> dict[str, Any]:
    lookup = {
        (int(row["replicate"]), int(row["budget"]), str(row["candidate_id"])): row.get(
            metric
        )
        for row in metrics
        if row["arm"] == arm and row["head"] == head
    }
    blocks = []
    for replicate in full.REPLICATES:
        differences = []
        for budget in full.BUDGETS:
            left = lookup[(replicate, budget, candidate)]
            right = lookup[(replicate, budget, comparator)]
            if left is None or right is None:
                raise RuntimeError("ranking contrast contains undefined metric")
            differences.append(float(left) - float(right))
        blocks.append(float(np.mean(differences)))
    return {
        "arm": arm,
        "head": head,
        "candidate_id": candidate,
        "comparator": comparator,
        "metric": metric,
        **_paired_interval(
            blocks,
            key=f"ranking:{metric}:{arm}:{head}:{candidate}:{comparator}",
            statistic=lambda array: float(np.mean(array)),
        ),
    }


def _rank_auc_contrast(
    rank_auc: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    head: str,
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    lookup = {
        (int(row["replicate"]), str(row["candidate_id"])): row.get("rank_auc")
        for row in rank_auc
        if row["arm"] == arm and row["head"] == head
    }
    blocks = []
    for replicate in full.REPLICATES:
        left = lookup[(replicate, candidate)]
        right = lookup[(replicate, comparator)]
        if left is None or right is None:
            return {
                "arm": arm,
                "head": head,
                "candidate_id": candidate,
                "comparator": comparator,
                "metric": "rank_auc",
                "estimate": None,
                "lower95": None,
                "upper95": None,
                "replicate_values": [],
                "resamples": 0,
                "status": "undefined",
            }
        blocks.append(float(left) - float(right))
    return {
        "arm": arm,
        "head": head,
        "candidate_id": candidate,
        "comparator": comparator,
        "metric": "rank_auc",
        "status": "defined",
        **_paired_interval(
            blocks,
            key=f"rank_auc:{arm}:{head}:{candidate}:{comparator}",
            statistic=lambda array: float(np.mean(array)),
        ),
    }
def _runtime_contrast(
    raw: Mapping[str, Any], *, arm: str, comparator: str, full_panel: bool
) -> dict[str, Any]:
    lookup = {
        (
            str(row["model"]),
            int(row["replicate"]),
            str(row["arm"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        ): float(row["total_wall_seconds"])
        for row in raw["selector_rows"]
    }
    blocks = []
    for replicate in full.REPLICATES:
        if full_panel:
            budget_ratios = []
            for budget in full.BUDGETS:
                candidate_total = sum(
                    lookup[(model, replicate, arm, budget, "FUSED3")]
                    for model in full.MODELS
                )
                comparator_total = sum(
                    lookup[(model, replicate, arm, budget, comparator)]
                    for model in full.MODELS
                )
                budget_ratios.append(candidate_total / comparator_total)
            blocks.append(float(np.mean(budget_ratios)))
        else:
            ratios = [
                lookup[(model, replicate, arm, budget, "FUSED3")]
                / lookup[(model, replicate, arm, budget, comparator)]
                for model in full.MODELS
                for budget in full.BUDGETS
            ]
            blocks.append(float(np.median(ratios)))
    kind = "full_panel" if full_panel else "per_call"
    return {
        "arm": arm,
        "candidate_id": "FUSED3",
        "comparator": comparator,
        "kind": kind,
        **_paired_interval(
            blocks,
            key=f"runtime:{kind}:{arm}:{comparator}",
            statistic=lambda array: float(np.median(array)),
        ),
    }


def _aggregate(
    metrics: Sequence[Mapping[str, Any]], rank_auc: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for arm in full.ARMS:
        for head in HEADS:
            for method in METHODS:
                rows = [
                    row
                    for row in metrics
                    if row["arm"] == arm
                    and row["head"] == head
                    and row["candidate_id"] == method
                ]
                rho = [float(row["spearman"]) for row in rows if row["spearman"] is not None]
                rank_rows = [
                    row
                    for row in rank_auc
                    if row["arm"] == arm
                    and row["head"] == head
                    and row["candidate_id"] == method
                ]
                defined_auc = [
                    float(row["rank_auc"])
                    for row in rank_rows
                    if row["rank_auc"] is not None
                ]
                output.append(
                    {
                        "arm": arm,
                        "head": head,
                        "candidate_id": method,
                        "mean_regret_pp": float(np.mean([row["regret_pp"] for row in rows])),
                        "mean_spearman": float(np.mean(rho)) if rho else None,
                        "mean_rank_auc": (
                            float(np.mean(defined_auc))
                            if len(defined_auc) == len(full.REPLICATES)
                            else None
                        ),
                        "exact_best_rate": float(np.mean([row["exact_best"] for row in rows])),
                        "within_one_pp_rate": float(
                            np.mean([row["within_one_pp"] for row in rows])
                        ),
                    }
                )
    return output


def analyze(raw: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metric_rows(raw)
    rank_auc = _rank_auc_rows(metrics)
    regret_contrasts = []
    rank_contrasts = []
    for arm, head, comparator in (
        ("baseline", "linear", "LP-FULL"),
        ("nuisance_full", "linear", "LP-FULL"),
    ):
        regret_contrasts.append(
            _ranking_contrast(
                metrics,
                arm=arm,
                head=head,
                candidate="FUSED3",
                comparator=comparator,
                metric="regret_pp",
            )
        )
    for head in NONLINEAR_HEADS:
        for comparator in ("LP-FULL", "REFINED-OI", "REFINED-OI-ARCHIVED"):
            regret_contrasts.append(
                _ranking_contrast(
                    metrics,
                    arm="nonlinearity_full",
                    head=head,
                    candidate="FUSED3",
                    comparator=comparator,
                    metric="regret_pp",
                )
            )
        regret_contrasts.append(
            _ranking_contrast(
                metrics,
                arm="nonlinearity_full",
                head=head,
                candidate="REFINED-OI",
                comparator="REFINED-OI-ARCHIVED",
                metric="regret_pp",
            )
        )
        rank_contrasts.append(
            _rank_auc_contrast(
                rank_auc,
                arm="nonlinearity_full",
                head=head,
                candidate="FUSED3",
                comparator="REFINED-OI",
            )
        )
    runtime = [
        _runtime_contrast(raw, arm=arm, comparator=comparator, full_panel=full_panel)
        for arm in full.ARMS
        for comparator in ("LP-FULL", "REFINED-OI")
        for full_panel in (False, True)
    ]
    gates = []
    for row in regret_contrasts:
        candidate = row["candidate_id"]
        comparator = row["comparator"]
        if candidate == "REFINED-OI":
            gate = "fresh_refined_historical_stability"
        elif row["arm"] == "baseline":
            gate = "clean_linear_vs_probe"
        elif row["arm"] == "nuisance_full":
            gate = "nuisance_linear_vs_probe"
        elif comparator == "LP-FULL":
            gate = "nonlinear_vs_probe"
        elif comparator == "REFINED-OI":
            gate = "nonlinear_retention_fresh_refined"
        else:
            gate = "nonlinear_retention_archived_refined"
        gates.append(
            {
                "gate": gate,
                "arm": row["arm"],
                "head": row["head"],
                "candidate_id": candidate,
                "comparator": comparator,
                "estimate": row["estimate"],
                "upper95": row["upper95"],
                "threshold": 1.0,
                "status": "pass" if float(row["upper95"]) <= 1.0 else "fail",
                "required": True,
            }
        )
    for row in rank_contrasts:
        rank_defined = row.get("status") == "defined" and row.get("lower95") is not None
        gates.append(
            {
                "gate": "nonlinear_rank_retention",
                "arm": row["arm"],
                "head": row["head"],
                "candidate_id": "FUSED3",
                "comparator": "REFINED-OI",
                "estimate": row.get("estimate"),
                "lower95": row.get("lower95"),
                "threshold": -0.10,
                "status": (
                    "pass"
                    if rank_defined and float(row["lower95"]) >= -0.10
                    else "fail"
                ),
                "required": True,
            }
        )
    for row in runtime:
        gates.append(
            {
                "gate": "runtime_full_panel" if row["kind"] == "full_panel" else "runtime_per_call",
                "arm": row["arm"],
                "head": None,
                "candidate_id": "FUSED3",
                "comparator": row["comparator"],
                "estimate": row["estimate"],
                "upper95": row["upper95"],
                "threshold_median": 1.0,
                "threshold_upper95": 1.10,
                "status": (
                    "pass"
                    if float(row["estimate"]) <= 1.0 and float(row["upper95"]) <= 1.10
                    else "fail"
                ),
                "required": True,
            }
        )
    gates.extend(
        [
            {"gate": "screen_overlap_parity", "status": "pass", "required": True},
            {"gate": "linear_probe_archived_parity", "status": "pass", "required": True},
            {"gate": "loop_fused_design_parity", "status": "pass", "required": True},
            {"gate": "determinism", "status": "pass", "required": True},
        ]
    )
    required_failures = [row for row in gates if row["required"] and row["status"] != "pass"]
    superiority = []
    for row in regret_contrasts:
        if row["candidate_id"] == "FUSED3" and row["comparator"] == "LP-FULL" and row["head"] in NONLINEAR_HEADS:
            superiority.append(
                {
                    "head": row["head"],
                    "estimate": row["estimate"],
                    "upper95": row["upper95"],
                    "status": "established" if float(row["upper95"]) < 0.0 else "not_established",
                }
            )
    faster = []
    for arm in full.ARMS:
        for comparator in ("LP-FULL", "REFINED-OI"):
            pair = [row for row in runtime if row["arm"] == arm and row["comparator"] == comparator]
            faster.append(
                {
                    "arm": arm,
                    "comparator": comparator,
                    "status": (
                        "established"
                        if len(pair) == 2 and all(float(row["upper95"]) < 1.0 for row in pair)
                        else "not_established"
                    ),
                    "promotion_veto": False,
                }
            )
    refined_deltas = np.asarray(
        [float(row["delta"]) for row in raw["refined_historical_deltas"]], dtype=np.float64
    )
    decision = {
        "status": "pass_full_retrospective" if not required_failures else "fail_locked_candidate",
        "selected_candidate": "FUSED3",
        "runner_up_reselection_allowed": False,
        "failed_required_gates": [
            {
                "gate": row["gate"],
                "arm": row.get("arm"),
                "head": row.get("head"),
                "comparator": row.get("comparator"),
            }
            for row in required_failures
        ],
        "confirmation_status": "not_untouched_not_run",
        "retrospective_development_only": True,
    }
    return {
        "schema_version": 1,
        "study": "fused_class_prototypes_food101_full_analysis",
        "artifact_status": "completed",
        "selector_metrics": metrics,
        "rank_auc": rank_auc,
        "aggregate": _aggregate(metrics, rank_auc),
        "regret_contrasts": regret_contrasts,
        "rank_contrasts": rank_contrasts,
        "runtime_contrasts": runtime,
        "gates": gates,
        "nonlinear_superiority_claims": superiority,
        "faster_claims": faster,
        "refined_score_parity_diagnostic": {
            "exact_rate": float(np.mean(refined_deltas == 0.0)),
            "max_abs_delta": float(np.max(np.abs(refined_deltas))),
            "mean_abs_delta": float(np.mean(np.abs(refined_deltas))),
        },
        "decision": decision,
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# FUSED3 full Food-101 retrospective replay",
        "",
        "This is the complete frozen Food-101 grid, but remains retrospective development evidence rather than untouched confirmation.",
        "",
        f"Decision: **{summary['decision']['status']}** for locked `FUSED3`.",
        "",
        "## Ranking",
        "",
        "| Arm | Head | Method | Mean regret pp | Mean Spearman | Rank AUC | Exact best | Within 1 pp |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    primary = {
        ("baseline", "linear"),
        ("nuisance_full", "linear"),
        *(("nonlinearity_full", head) for head in NONLINEAR_HEADS),
    }
    for row in summary["aggregate"]:
        if (row["arm"], row["head"]) not in primary:
            continue
        rho = "—" if row["mean_spearman"] is None else f"{row['mean_spearman']:.3f}"
        rank_auc = "—" if row["mean_rank_auc"] is None else f"{row['mean_rank_auc']:.3f}"
        lines.append(
            f"| {row['arm']} | {row['head']} | {row['candidate_id']} | "
            f"{row['mean_regret_pp']:.3f} | {rho} | {rank_auc} | {row['exact_best_rate']:.2f} | "
            f"{row['within_one_pp_rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired regret contrasts (FUSED3 minus comparator)",
            "",
            "| Arm | Head | Comparator | Estimate pp | Lower 95 | Upper 95 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["regret_contrasts"]:
        if row["candidate_id"] != "FUSED3":
            continue
        lines.append(
            f"| {row['arm']} | {row['head']} | {row['comparator']} | "
            f"{row['estimate']:.3f} | {row['lower95']:.3f} | {row['upper95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Runtime ratios",
            "",
            "| Arm | Comparator | Estimand | Median ratio | Lower 95 | Upper 95 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["runtime_contrasts"]:
        lines.append(
            f"| {row['arm']} | {row['comparator']} | {row['kind']} | "
            f"{row['estimate']:.3f} | {row['lower95']:.3f} | {row['upper95']:.3f} |"
        )
    lines.extend(["", "## Claims", ""])
    for row in summary["nonlinear_superiority_claims"]:
        lines.append(f"- Nonlinear superiority over LP-FULL / {row['head']}: **{row['status']}**")
    for row in summary["faster_claims"]:
        lines.append(
            f"- Faster than {row['comparator']} / {row['arm']}: **{row['status']}**"
        )
    diagnostic = summary["refined_score_parity_diagnostic"]
    lines.extend(
        [
            "",
            "## Historical refined-OI diagnostic",
            "",
            f"Fresh/archived exact score match rate: {diagnostic['exact_rate']:.3f}; "
            f"maximum absolute score delta: {diagnostic['max_abs_delta']:.6g}.",
            "",
            "## Takeaway",
            "",
            (
                "Locked FUSED3 passed every full retrospective ranking, rank-retention, runtime, parity, and determinism gate. Untouched confirmation would still be required for a product claim."
                if summary["decision"]["status"] == "pass_full_retrospective"
                else "Locked FUSED3 failed at least one required full-grid gate and is not eligible for substitution by another candidate."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite non-empty full Food analysis output")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = verify_artifact(Path(input_dir))
    summary = analyze(raw)
    summary["input_raw_results_sha256"] = full.sha256_path(
        Path(input_dir) / "raw_results.json"
    )
    summary["protocol_sha256"] = raw["run_identity"]["protocol_sha256"]
    summary_path = output_dir / "summary.json"
    decision_path = output_dir / "decision.json"
    report_path = output_dir / "report.md"
    _atomic_json(summary_path, summary)
    _atomic_json(decision_path, summary["decision"])
    _atomic_text(report_path, _report(summary))
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": summary["study"],
        "input_raw_results_sha256": summary["input_raw_results_sha256"],
        "protocol_sha256": summary["protocol_sha256"],
        "summary_sha256": full.sha256_path(summary_path),
        "decision_sha256": full.sha256_path(decision_path),
        "report_sha256": full.sha256_path(report_path),
        "decision": summary["decision"],
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = write_analysis(args.input, args.output)
    print(full.canonical_json(summary["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
