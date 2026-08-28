"""Strict paired statistics for the M50 backbone-ranking experiment.

Food-101 is retrospective development evidence.  The confirmation helpers in
this module consume a previously locked selector and never choose a different
candidate from confirmation outcomes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .fusion import FULL_PROBE_CANDIDATE_ID, fractional_midranks


FOOD_BACKBONES = (
    "dinov2-small",
    "deit-tiny",
    "convnext-tiny",
    "mobilenetv3-large",
    "openclip-vit-b-32",
    "resnet50",
    "efficientnet-b0",
    "swin-tiny",
    "vit-small-16",
    "densenet121",
)
FOOD_REPLICATES = tuple(range(5))
FOOD_ARMS = ("baseline", "nonlinearity_full", "nuisance_full")
FOOD_BUDGETS = (64, 68, 72, 80)
FOOD_RUNTIME_BUDGETS = (64, 128, 256, 512, 640)
REFERENCE_HEADS = ("linear", "quadratic", "knn", "rbf")
FACTORIAL_CANDIDATES = ("A", "B", "M0-SW", "M1-SW")
SW_CANDIDATES = ("M0-SW", "M1-SW")
CB_CANDIDATES = ("M0-CB", "M1-CB")
RUNTIME_PRIMITIVE_IDS = (
    "A",
    "B",
    "M0-SW",
    "M1-SW",
    "LP-FULL",
    "LP-CAPPED-2048",
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2_026_082_701
CONFIRMATION_BOOTSTRAP_SEED = 2_026_082_702

LINEAR_REGRET_MARGIN_PP = 1.0
REFINEMENT_NONINFERIORITY_MARGIN_PP = 0.25
REFINEMENT_SPEED_RATIO_MAX = 0.95

PRIMARY_CELLS = (
    ("baseline", "linear"),
    ("nuisance_full", "linear"),
    ("nonlinearity_full", "quadratic"),
    ("nonlinearity_full", "knn"),
    ("nonlinearity_full", "rbf"),
)


@dataclass(frozen=True)
class FoodDesign:
    """Exact complete Food development identities."""

    backbones: tuple[str, ...] = FOOD_BACKBONES
    replicates: tuple[int, ...] = FOOD_REPLICATES
    arms: tuple[str, ...] = FOOD_ARMS
    budgets: tuple[int, ...] = FOOD_BUDGETS
    heads: tuple[str, ...] = REFERENCE_HEADS


def _materialize(rows: Iterable[Mapping[str, Any]], *, name: str) -> list[dict[str, Any]]:
    try:
        materialized = [dict(row) for row in rows]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an iterable of mappings") from exc
    if not materialized:
        raise ValueError(f"{name} must not be empty")
    return materialized


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be a finite real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a finite real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _strict_int(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    return int(value)


def _validate_closed(value: Any, allowed: Sequence[Any], *, field: str) -> Any:
    if value not in allowed:
        raise ValueError(f"{field} must be one of {tuple(allowed)!r}; got {value!r}")
    return value


def _spearman(scores: Sequence[float], outcomes: Sequence[float]) -> tuple[float | None, str]:
    if len(scores) != len(outcomes) or len(scores) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    order = tuple(str(position) for position in range(len(scores)))
    score_ranks = np.asarray(
        list(fractional_midranks(dict(zip(order, scores)), item_order=order).values()),
        dtype=np.float64,
    )
    outcome_ranks = np.asarray(
        list(fractional_midranks(dict(zip(order, outcomes)), item_order=order).values()),
        dtype=np.float64,
    )
    score_centered = score_ranks - np.mean(score_ranks)
    outcome_centered = outcome_ranks - np.mean(outcome_ranks)
    denominator = float(
        np.sqrt(np.sum(score_centered**2) * np.sum(outcome_centered**2))
    )
    if denominator == 0.0:
        return None, "undefined_constant_input"
    return float(np.sum(score_centered * outcome_centered) / denominator), "defined"


def validate_food_inputs(
    selector_rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
    *,
    candidate_ids: Sequence[str],
    design: FoodDesign = FoodDesign(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize once and require the exact complete frozen Food identities."""

    selectors = _materialize(selector_rows, name="selector_rows")
    references = _materialize(reference_rows, name="reference_rows")
    candidates = tuple(str(value) for value in candidate_ids)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate_ids must contain unique values")

    expected_selector = {
        (candidate, backbone, replicate, arm, budget)
        for candidate in candidates
        for backbone in design.backbones
        for replicate in design.replicates
        for arm in design.arms
        for budget in design.budgets
    }
    observed_selector: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in selectors:
        identity = (
            str(row.get("candidate_id")),
            str(row.get("backbone")),
            _strict_int(row.get("replicate"), field="replicate"),
            str(row.get("arm")),
            _strict_int(row.get("budget"), field="budget"),
        )
        _validate_closed(identity[0], candidates, field="candidate_id")
        _validate_closed(identity[1], design.backbones, field="backbone")
        _validate_closed(identity[2], design.replicates, field="replicate")
        _validate_closed(identity[3], design.arms, field="arm")
        _validate_closed(identity[4], design.budgets, field="budget")
        _finite_float(row.get("score"), field="score")
        if identity in observed_selector:
            raise ValueError(f"duplicate selector identity {identity!r}")
        observed_selector[identity] = row
    if set(observed_selector) != expected_selector:
        missing = len(expected_selector - set(observed_selector))
        extra = len(set(observed_selector) - expected_selector)
        raise ValueError(
            f"selector rows are incomplete or malformed; missing={missing}, extra={extra}"
        )

    expected_reference = {
        (backbone, replicate, arm, head)
        for backbone in design.backbones
        for replicate in design.replicates
        for arm in design.arms
        for head in design.heads
    }
    observed_reference: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in references:
        identity = (
            str(row.get("backbone")),
            _strict_int(row.get("replicate"), field="replicate"),
            str(row.get("arm")),
            str(row.get("head")),
        )
        _validate_closed(identity[0], design.backbones, field="backbone")
        _validate_closed(identity[1], design.replicates, field="replicate")
        _validate_closed(identity[2], design.arms, field="arm")
        _validate_closed(identity[3], design.heads, field="head")
        _finite_float(row.get("test_accuracy"), field="test_accuracy")
        if identity in observed_reference:
            raise ValueError(f"duplicate reference identity {identity!r}")
        observed_reference[identity] = row
    if set(observed_reference) != expected_reference:
        missing = len(expected_reference - set(observed_reference))
        extra = len(set(observed_reference) - expected_reference)
        raise ValueError(
            f"reference rows are incomplete or malformed; missing={missing}, extra={extra}"
        )
    return selectors, references


def food_panel_metrics(
    selector_rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
    *,
    candidate_ids: Sequence[str],
    design: FoodDesign = FoodDesign(),
) -> list[dict[str, Any]]:
    """Return regret, exact-best, within-one-point, and tie-safe rank rows."""

    selectors, references = validate_food_inputs(
        selector_rows, reference_rows, candidate_ids=candidate_ids, design=design
    )
    selector_lookup = {
        (
            str(row["candidate_id"]),
            str(row["backbone"]),
            int(row["replicate"]),
            str(row["arm"]),
            int(row["budget"]),
        ): row
        for row in selectors
    }
    reference_lookup = {
        (
            str(row["backbone"]),
            int(row["replicate"]),
            str(row["arm"]),
            str(row["head"]),
        ): row
        for row in references
    }
    result: list[dict[str, Any]] = []
    for candidate in tuple(str(value) for value in candidate_ids):
        for replicate in design.replicates:
            for arm in design.arms:
                for budget in design.budgets:
                    scores = {
                        backbone: _finite_float(
                            selector_lookup[
                                (candidate, backbone, replicate, arm, budget)
                            ]["score"],
                            field="score",
                        )
                        for backbone in design.backbones
                    }
                    if candidate == "F":
                        selected_markers = []
                        for backbone in design.backbones:
                            marker = selector_lookup[
                                (candidate, backbone, replicate, arm, budget)
                            ].get("selected")
                            if type(marker) is not bool:
                                raise ValueError(
                                    f"derived candidate {candidate} requires strict selected markers"
                                )
                            if marker:
                                selected_markers.append(backbone)
                        if len(selected_markers) != 1:
                            raise ValueError(
                                f"derived candidate {candidate} requires exactly one selected backbone"
                            )
                        tied = tuple(selected_markers)
                        selection_semantics = "derived_frozen_order_argmax"
                    elif candidate == "G":
                        semantics = {
                            selector_lookup[
                                (candidate, backbone, replicate, arm, budget)
                            ].get("selection_semantics")
                            for backbone in design.backbones
                        }
                        if semantics not in (
                            {"derived_frozen_order_argmax"},
                            {"archived_atol_1e-12_tie_average"},
                        ):
                            raise ValueError(
                                "G requires one canonical source selection_semantics per panel"
                            )
                        selection_semantics = str(next(iter(semantics)))
                        selected_markers = []
                        for backbone in design.backbones:
                            marker = selector_lookup[
                                (candidate, backbone, replicate, arm, budget)
                            ].get("selected")
                            if type(marker) is not bool:
                                raise ValueError("G requires strict selected markers")
                            if marker:
                                selected_markers.append(backbone)
                        if selection_semantics == "derived_frozen_order_argmax":
                            if len(selected_markers) != 1:
                                raise ValueError(
                                    "frozen-order G source requires exactly one selected backbone"
                                )
                        else:
                            maximum_score = max(scores.values())
                            expected_ties = tuple(
                                backbone
                                for backbone in design.backbones
                                if abs(scores[backbone] - maximum_score) <= 1e-12
                            )
                            if tuple(selected_markers) != expected_ties:
                                raise ValueError(
                                    "G archived selected markers do not match the atol=1e-12 tie set"
                                )
                        tied = tuple(selected_markers)
                    else:
                        maximum_score = max(scores.values())
                        tied = tuple(
                            backbone
                            for backbone in design.backbones
                            if abs(scores[backbone] - maximum_score) <= 1e-12
                        )
                        selection_semantics = "archived_atol_1e-12_tie_average"
                    for head in design.heads:
                        accuracy = {
                            backbone: _finite_float(
                                reference_lookup[(backbone, replicate, arm, head)][
                                    "test_accuracy"
                                ],
                                field="test_accuracy",
                            )
                            for backbone in design.backbones
                        }
                        best = max(accuracy.values())
                        selected_accuracy = float(
                            np.mean([accuracy[backbone] for backbone in tied])
                        )
                        regret_pp = float(100.0 * (best - selected_accuracy))
                        exact_best_rate = float(
                            np.mean(
                                [
                                    abs(best - accuracy[backbone]) <= 1e-12
                                    for backbone in tied
                                ]
                            )
                        )
                        within_one_rate = float(
                            np.mean(
                                [
                                    100.0 * (best - accuracy[backbone])
                                    <= 1.0 + 1e-12
                                    for backbone in tied
                                ]
                            )
                        )
                        correlation, correlation_status = _spearman(
                            [scores[value] for value in design.backbones],
                            [accuracy[value] for value in design.backbones],
                        )
                        result.append(
                            {
                                "candidate_id": candidate,
                                "replicate": int(replicate),
                                "arm": arm,
                                "budget": int(budget),
                                "head": head,
                                "selected_backbone": tied[0] if len(tied) == 1 else None,
                                "selected_backbones": list(tied),
                                "selection_tie_count": len(tied),
                                "selection_semantics": selection_semantics,
                                "selected_accuracy": selected_accuracy,
                                "best_accuracy": best,
                                "regret_pp": regret_pp,
                                "exact_best_rate": exact_best_rate,
                                "within_one_pp_rate": within_one_rate,
                                "spearman": correlation,
                                "spearman_status": correlation_status,
                                "panel_size": len(design.backbones),
                            }
                        )
    return result


def rank_auc_rows(
    panel_metrics: Iterable[Mapping[str, Any]],
    *,
    budgets: Sequence[int] = FOOD_BUDGETS,
) -> list[dict[str, Any]]:
    """Require complete budget sets and integrate Spearman over log2 budget."""

    rows = _materialize(panel_metrics, name="panel_metrics")
    grouped: dict[tuple[str, int, str, str], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("candidate_id")),
            _strict_int(row.get("replicate"), field="replicate"),
            str(row.get("arm")),
            str(row.get("head")),
        )
        budget = _strict_int(row.get("budget"), field="budget")
        slot = grouped.setdefault(key, {})
        if budget in slot:
            raise ValueError(f"duplicate rank-AUC budget for {key!r}: {budget}")
        slot[budget] = row
    expected = tuple(int(value) for value in budgets)
    x = np.log2(np.asarray(expected, dtype=np.float64))
    denominator = float(x[-1] - x[0])
    if denominator <= 0.0:
        raise ValueError("rank-AUC budgets must be strictly increasing")
    result: list[dict[str, Any]] = []
    for key in sorted(grouped, key=repr):
        observed = grouped[key]
        if set(observed) != set(expected):
            raise ValueError(f"rank AUC requires the complete budget set for {key!r}")
        values: list[float] = []
        undefined = False
        for budget in expected:
            value = observed[budget].get("spearman")
            if value is None:
                undefined = True
                break
            values.append(_finite_float(value, field="spearman"))
        if undefined:
            auc, status = None, "undefined_constant_input"
        else:
            y = np.asarray(values, dtype=np.float64)
            auc = float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)) / denominator)
            status = "defined"
        result.append(
            {
                "candidate_id": key[0],
                "replicate": key[1],
                "arm": key[2],
                "head": key[3],
                "rank_auc": auc,
                "rank_auc_status": status,
                "budget_count": len(expected),
            }
        )
    return result


def selection_rate_summary(
    panel_metrics: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return descriptive selector quality by complete candidate/arm/head cells."""

    rows = _materialize(panel_metrics, name="panel_metrics")
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("candidate_id")),
                str(row.get("arm")),
                str(row.get("head")),
            )
        ].append(row)
    result = []
    expected_count = len(FOOD_REPLICATES) * len(FOOD_BUDGETS)
    for key in sorted(grouped, key=repr):
        values = grouped[key]
        if len(values) != expected_count:
            raise ValueError(f"selection-rate summary cell {key!r} is incomplete")
        result.append(
            {
                "candidate_id": key[0],
                "arm": key[1],
                "head": key[2],
                "mean_regret_pp": float(
                    np.mean([_finite_float(row.get("regret_pp"), field="regret_pp") for row in values])
                ),
                "exact_best_rate": float(
                    np.mean(
                        [
                            _finite_float(row.get("exact_best_rate"), field="exact_best_rate")
                            for row in values
                        ]
                    )
                ),
                "within_one_pp_rate": float(
                    np.mean(
                        [
                            _finite_float(row.get("within_one_pp_rate"), field="within_one_pp_rate")
                            for row in values
                        ]
                    )
                ),
                "panel_count": len(values),
            }
        )
    return result


def rank_auc_summary(
    rank_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize complete rank-AUC cells without fabricating undefined values."""

    rows = _materialize(rank_rows, name="rank_auc")
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("candidate_id")),
                str(row.get("arm")),
                str(row.get("head")),
            )
        ].append(row)
    result = []
    for key in sorted(grouped, key=repr):
        values = grouped[key]
        if len(values) != len(FOOD_REPLICATES):
            raise ValueError(f"rank-AUC summary cell {key!r} is incomplete")
        defined = [
            _finite_float(row.get("rank_auc"), field="rank_auc")
            for row in values
            if row.get("rank_auc_status") == "defined"
        ]
        undefined = len(values) - len(defined)
        result.append(
            {
                "candidate_id": key[0],
                "arm": key[1],
                "head": key[2],
                "mean_rank_auc": float(np.mean(defined)) if undefined == 0 else None,
                "status": "defined" if undefined == 0 else "undefined_constant_input",
                "undefined_replicate_count": undefined,
                "replicate_count": len(values),
            }
        )
    return result


def paired_replicate_interval(
    replicate_values: Mapping[int, Any] | Iterable[tuple[int, Any]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile interval for an equally weighted complete-replicate mean."""

    items = list(replicate_values.items()) if isinstance(replicate_values, Mapping) else list(replicate_values)
    if not items:
        raise ValueError("replicate_values must not be empty")
    lookup: dict[int, float] = {}
    for replicate, value in items:
        key = _strict_int(replicate, field="replicate")
        if key in lookup:
            raise ValueError(f"duplicate replicate {key}")
        lookup[key] = _finite_float(value, field="replicate contrast")
    keys = tuple(sorted(lookup))
    values = np.asarray([lookup[key] for key in keys], dtype=np.float64)
    if not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, values.size, size=(int(n_resamples), values.size))
    estimates = np.mean(values[draws], axis=1)
    return {
        "estimate": float(np.mean(values)),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "n_replicates": int(values.size),
        "n_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": "complete_replicate",
    }


def _regret_lookup(
    panel_metrics: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str, int, str], float]:
    result: dict[tuple[str, int, str, int, str], float] = {}
    for row in panel_metrics:
        key = (
            str(row.get("candidate_id")),
            _strict_int(row.get("replicate"), field="replicate"),
            str(row.get("arm")),
            _strict_int(row.get("budget"), field="budget"),
            str(row.get("head")),
        )
        if key in result:
            raise ValueError(f"duplicate panel metric {key!r}")
        result[key] = _finite_float(row.get("regret_pp"), field="regret_pp")
    return result


def regret_contrast(
    panel_metrics: Iterable[Mapping[str, Any]],
    *,
    candidate_id: str,
    comparator_id: str,
    arm: str,
    head: str,
    replicates: Sequence[int] = FOOD_REPLICATES,
    budgets: Sequence[int] = FOOD_BUDGETS,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap candidate-minus-comparator regret with whole replicate blocks."""

    rows = _materialize(panel_metrics, name="panel_metrics")
    lookup = _regret_lookup(rows)
    replicate_values: dict[int, float] = {}
    for replicate in replicates:
        differences = []
        for budget in budgets:
            candidate_key = (str(candidate_id), int(replicate), str(arm), int(budget), str(head))
            comparator_key = (str(comparator_id), int(replicate), str(arm), int(budget), str(head))
            if candidate_key not in lookup or comparator_key not in lookup:
                raise ValueError(
                    f"incomplete paired regret contrast for replicate={replicate}, "
                    f"arm={arm!r}, head={head!r}, budget={budget}"
                )
            differences.append(lookup[candidate_key] - lookup[comparator_key])
        replicate_values[int(replicate)] = float(np.mean(differences))
    return {
        "candidate_id": str(candidate_id),
        "comparator_id": str(comparator_id),
        "arm": str(arm),
        "head": str(head),
        "unit": "accuracy_percentage_points",
        **paired_replicate_interval(
            replicate_values, n_resamples=n_resamples, seed=seed
        ),
    }


def candidate_product_gates(
    panel_metrics: Iterable[Mapping[str, Any]],
    *,
    candidate_id: str,
    probe_id: str = FULL_PROBE_CANDIDATE_ID,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Apply the exact development regret gates for one pure SW candidate."""

    rows = _materialize(panel_metrics, name="panel_metrics")
    gates: dict[str, dict[str, Any]] = {}
    for arm, head in PRIMARY_CELLS:
        interval = regret_contrast(
            rows,
            candidate_id=candidate_id,
            comparator_id=probe_id,
            arm=arm,
            head=head,
            n_resamples=n_resamples,
            seed=seed,
        )
        if head == "linear":
            status = "pass" if interval["upper_95"] <= LINEAR_REGRET_MARGIN_PP else "fail"
            rule = "upper_95 <= 1.0 pp"
        elif head in {"quadratic", "knn"}:
            status = "pass" if interval["upper_95"] < 0.0 else "fail"
            rule = "upper_95 < 0 pp"
        else:
            if interval["upper_95"] < 0.0:
                status = "pass"
            elif (
                interval["lower_95"] <= 0.0 <= interval["upper_95"]
                and interval["upper_95"] <= LINEAR_REGRET_MARGIN_PP
            ):
                status = "inconclusive_fusion_required"
            else:
                status = "fail"
            rule = "upper_95 < 0; crossing zero with upper_95 <= 1.0 activates F"
        gates[f"{arm}:{head}"] = {**interval, "status": status, "rule": rule}

    linear_pass = all(
        gates[key]["status"] == "pass"
        for key in ("baseline:linear", "nuisance_full:linear")
    )
    quadratic_knn_pass = all(
        gates[key]["status"] == "pass"
        for key in ("nonlinearity_full:quadratic", "nonlinearity_full:knn")
    )
    rbf_status = gates["nonlinearity_full:rbf"]["status"]
    nonlinear_pass = quadratic_knn_pass and rbf_status == "pass"
    guardrail_eligible = nonlinear_pass and not linear_pass
    if linear_pass and nonlinear_pass:
        status = "pass"
    elif linear_pass and quadratic_knn_pass and rbf_status == "inconclusive_fusion_required":
        status = "fusion_required"
    elif guardrail_eligible:
        status = "guardrail_eligible"
    else:
        status = "fail"
    return {
        "candidate_id": str(candidate_id),
        "status": status,
        "algorithmically_eligible": status == "pass",
        "base_eligible": status in {"pass", "fusion_required"},
        "requires_fusion": status == "fusion_required",
        "linear_gates_pass": linear_pass,
        "nonlinear_gates_pass": nonlinear_pass,
        "guardrail_eligible": guardrail_eligible,
        "gates": gates,
    }


def factorial_effects(
    panel_metrics: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Report the SW factorial and the non-promotable CB refinement diagnostic."""

    rows = _materialize(panel_metrics, name="panel_metrics")
    effects: dict[str, Any] = {}
    for arm, head in PRIMARY_CELLS:
        raw = regret_contrast(
            rows,
            candidate_id="B",
            comparator_id="A",
            arm=arm,
            head=head,
            n_resamples=n_resamples,
            seed=seed,
        )
        conditioned = regret_contrast(
            rows,
            candidate_id="M1-SW",
            comparator_id="M0-SW",
            arm=arm,
            head=head,
            n_resamples=n_resamples,
            seed=seed,
        )
        # Interaction is computed from panel-level paired rows, not from two
        # independently resampled interval endpoints.
        lookup = _regret_lookup(rows)
        interaction_values: dict[int, float] = {}
        for replicate in FOOD_REPLICATES:
            values = []
            for budget in FOOD_BUDGETS:
                def value(candidate: str) -> float:
                    key = (candidate, replicate, arm, budget, head)
                    if key not in lookup:
                        raise ValueError(f"incomplete factorial cell {key!r}")
                    return lookup[key]

                values.append(
                    (value("M1-SW") - value("M0-SW"))
                    - (value("B") - value("A"))
                )
            interaction_values[replicate] = float(np.mean(values))
        interaction = paired_replicate_interval(
            interaction_values, n_resamples=n_resamples, seed=seed
        )
        cb_diagnostic = regret_contrast(
            rows,
            candidate_id="M1-CB",
            comparator_id="M0-CB",
            arm=arm,
            head=head,
            n_resamples=n_resamples,
            seed=seed,
        )
        effects[f"{arm}:{head}"] = {
            "raw_refinement_effect": raw,
            "conditioned_refinement_effect": conditioned,
            "interaction": {
                **interaction,
                "definition": "(M1-SW - M0-SW) - (B - A)",
                "unit": "accuracy_percentage_points",
            },
            "cb_refinement_diagnostic": {
                **cb_diagnostic,
                "promotable": False,
                "definition": "M1-CB - M0-CB",
            },
        }
    return {
        "candidate_cells": list(FACTORIAL_CANDIDATES),
        "diagnostic_cells": list(CB_CANDIDATES),
        "effects": effects,
    }


def refinement_noninferiority(
    panel_metrics: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Test whether unrefined M50 is within 0.25 pp of refined M50."""

    rows = _materialize(panel_metrics, name="panel_metrics")
    gates: dict[str, Any] = {}
    for arm, head in PRIMARY_CELLS:
        interval = regret_contrast(
            rows,
            candidate_id="M0-SW",
            comparator_id="M1-SW",
            arm=arm,
            head=head,
            n_resamples=n_resamples,
            seed=seed,
        )
        gates[f"{arm}:{head}"] = {
            **interval,
            "margin_pp": REFINEMENT_NONINFERIORITY_MARGIN_PP,
            "status": "pass"
            if interval["upper_95"] <= REFINEMENT_NONINFERIORITY_MARGIN_PP
            else "fail",
        }
    return {
        "status": "pass"
        if all(value["status"] == "pass" for value in gates.values())
        else "fail",
        "gates": gates,
    }


def provisional_development_lock(
    panel_metrics: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Freeze the SW eligibility tier and authorize only primitive runtime work.

    Runtime settles the predeclared M0/M1 refinement rule before F or G is
    derived.  Consequently this provisional decision never names a derived
    policy as a measured estimator and never evaluates a derived policy early.
    """

    rows = _materialize(panel_metrics, name="panel_metrics")
    candidate_gates = {
        candidate: candidate_product_gates(
            rows,
            candidate_id=candidate,
            n_resamples=n_resamples,
            seed=seed,
        )
        for candidate in SW_CANDIDATES
    }
    pure = [
        candidate for candidate in SW_CANDIDATES if candidate_gates[candidate]["status"] == "pass"
    ]
    fusion = [
        candidate
        for candidate in SW_CANDIDATES
        if candidate_gates[candidate]["status"] == "fusion_required"
    ]
    guardrail = [
        candidate
        for candidate in SW_CANDIDATES
        if candidate_gates[candidate]["status"] == "guardrail_eligible"
    ]
    if pure:
        eligible, chain_kind = pure, "pure"
    elif fusion:
        eligible, chain_kind = fusion, "fusion"
    elif guardrail:
        eligible, chain_kind = guardrail, "guardrail"
    else:
        eligible, chain_kind = [], "none"
    noninferiority = refinement_noninferiority(
        rows, n_resamples=n_resamples, seed=seed
    )
    if not eligible:
        return {
            "status": "stopped_no_eligible_sw_candidate",
            "selected_candidate": None,
            "selected_chain": None,
            "candidate_gates": candidate_gates,
            "refinement_noninferiority": noninferiority,
            "runtime_pending": False,
            "runtime_candidate_ids": [],
            "runner_up_allowed": False,
            "cb_rescue_allowed": False,
        }
    if len(eligible) == 1:
        selected = eligible[0]
        runtime_choice = False
    elif noninferiority["status"] == "pass":
        selected = "M0-SW"
        runtime_choice = True
    else:
        selected = "M1-SW"
        runtime_choice = False
    needs_fusion = chain_kind == "fusion"
    needs_guardrail = chain_kind == "guardrail"
    return {
        "status": "provisional_runtime_pending",
        "selected_candidate": selected,
        "selected_chain": None,
        "eligibility_tier": chain_kind,
        "eligible_candidates": list(eligible),
        "requires_fusion": needs_fusion,
        "requires_guardrail": needs_guardrail,
        "candidate_gates": candidate_gates,
        "refinement_noninferiority": noninferiority,
        "runtime_pending": True,
        "runtime_candidate_ids": list(RUNTIME_PRIMITIVE_IDS),
        "unrefined_requires_speed_ratio": runtime_choice,
        "unrefined_speed_ratio_max": REFINEMENT_SPEED_RATIO_MAX if runtime_choice else None,
        "runner_up_allowed": False,
        "cb_rescue_allowed": False,
    }


def _bootstrap_median_by_repeat(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    n_resamples: int,
    seed: int,
    bootstrap_unit: str = "complete_runtime_repeat_block_preserving_arm_backbone_budget",
) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[_strict_int(row.get("repeat"), field="repeat")].append(
            _finite_float(row.get(value_field), field=value_field)
        )
    if not grouped:
        raise ValueError("runtime ratio rows must not be empty")
    repeats = tuple(sorted(grouped))
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, len(repeats), size=(int(n_resamples), len(repeats)))
    samples = np.empty(int(n_resamples), dtype=np.float64)
    for position, draw in enumerate(draws):
        pooled = [value for index in draw for value in grouped[repeats[index]]]
        samples[position] = float(np.median(np.asarray(pooled, dtype=np.float64)))
    observed = np.asarray(
        [value for repeat in repeats for value in grouped[repeat]], dtype=np.float64
    )
    return {
        "median": float(np.median(observed)),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "n_repeats": len(repeats),
        "n_values": int(observed.size),
        "n_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": str(bootstrap_unit),
    }


def resource_gates(
    runtime_rows: Iterable[Mapping[str, Any]],
    *,
    candidate_id: str,
    probe_id: str = FULL_PROBE_CANDIDATE_ID,
    baseline_id: str = "B",
    backbones: Sequence[str] = FOOD_BACKBONES,
    arms: Sequence[str] = FOOD_ARMS,
    repeats: Sequence[int] = FOOD_REPLICATES,
    budgets: Sequence[int] = FOOD_RUNTIME_BUDGETS,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Apply direct LP-full time and B-relative time/memory gates."""

    rows = _materialize(runtime_rows, name="runtime_rows")
    required_candidates = {str(candidate_id), str(probe_id), str(baseline_id)}
    expected = {
        (method, str(backbone), str(arm), int(budget), int(repeat))
        for method in required_candidates
        for backbone in backbones
        for arm in arms
        for budget in budgets
        for repeat in repeats
    }
    lookup: dict[tuple[str, str, str, int, int], Mapping[str, Any]] = {}
    stage_fields = (
        "fit_wall_seconds",
        "fit_cpu_seconds",
        "conditioning_fit_wall_seconds",
        "conditioning_fit_cpu_seconds",
        "oi_fit_wall_seconds",
        "oi_fit_cpu_seconds",
        "score_fixed_wall_seconds",
        "score_fixed_cpu_seconds",
    )
    for row in rows:
        method = str(row.get("candidate_id"))
        if method not in required_candidates:
            continue
        key = (
            method,
            str(row.get("backbone")),
            str(row.get("arm")),
            _strict_int(row.get("budget"), field="budget"),
            _strict_int(row.get("repeat"), field="repeat"),
        )
        if key in lookup:
            raise ValueError(f"duplicate runtime identity {key!r}")
        _finite_float(row.get("total_wall_seconds"), field="total_wall_seconds")
        _finite_float(row.get("total_cpu_seconds"), field="total_cpu_seconds")
        _finite_float(row.get("peak_memory_mb"), field="peak_memory_mb")
        for field in stage_fields:
            value = row.get(field)
            if value is None and field.startswith("oi_fit_") and method in {
                "LP-FULL",
                "LP-CAPPED-2048",
            }:
                continue
            stage_value = _finite_float(value, field=field)
            if stage_value < 0.0:
                raise ValueError(f"runtime stage field {field!r} must be nonnegative")
        lookup[key] = row
    if set(lookup) != expected:
        raise ValueError(
            "runtime evidence must contain exact complete candidate, LP-full, and B rows"
        )

    ratios: list[dict[str, Any]] = []
    for backbone in backbones:
        for arm in arms:
            for budget in budgets:
                for repeat in repeats:
                    identity = (str(backbone), str(arm), int(budget), int(repeat))
                    candidate = lookup[(str(candidate_id), *identity)]
                    probe = lookup[(str(probe_id), *identity)]
                    baseline = lookup[(str(baseline_id), *identity)]
                    candidate_wall = _finite_float(
                        candidate["total_wall_seconds"], field="total_wall_seconds"
                    )
                    probe_wall = _finite_float(
                        probe["total_wall_seconds"], field="total_wall_seconds"
                    )
                    baseline_wall = _finite_float(
                        baseline["total_wall_seconds"], field="total_wall_seconds"
                    )
                    candidate_memory = _finite_float(
                        candidate["peak_memory_mb"], field="peak_memory_mb"
                    )
                    baseline_memory = _finite_float(
                        baseline["peak_memory_mb"], field="peak_memory_mb"
                    )
                    candidate_cpu = _finite_float(
                        candidate["total_cpu_seconds"], field="total_cpu_seconds"
                    )
                    probe_cpu = _finite_float(
                        probe["total_cpu_seconds"], field="total_cpu_seconds"
                    )
                    baseline_cpu = _finite_float(
                        baseline["total_cpu_seconds"], field="total_cpu_seconds"
                    )
                    if min(
                        probe_wall,
                        baseline_wall,
                        baseline_memory,
                        probe_cpu,
                        baseline_cpu,
                    ) <= 0.0:
                        raise ValueError("runtime and memory denominators must be positive")
                    stage_values: dict[str, float] = {}
                    for prefix, source in (
                        ("candidate", candidate),
                        ("probe", probe),
                        ("baseline", baseline),
                    ):
                        for field in stage_fields:
                            stage_values[f"{prefix}_{field}"] = _finite_float(
                                source.get(field, 0.0), field=field
                            )
                    ratios.append(
                        {
                            "backbone": str(backbone),
                            "arm": str(arm),
                            "budget": int(budget),
                            "repeat": int(repeat),
                            "candidate_probe_wall_ratio": candidate_wall / probe_wall,
                            "candidate_baseline_wall_ratio": candidate_wall / baseline_wall,
                            "candidate_baseline_memory_ratio": candidate_memory
                            / baseline_memory,
                            "candidate_probe_cpu_ratio": candidate_cpu / probe_cpu,
                            "candidate_baseline_cpu_ratio": candidate_cpu
                            / baseline_cpu,
                            **stage_values,
                        }
                    )

    def gate_values(arm_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        gated = [row for row in arm_rows if int(row["budget"]) >= 128]
        probe_interval = _bootstrap_median_by_repeat(
            gated,
            value_field="candidate_probe_wall_ratio",
            n_resamples=n_resamples,
            seed=seed,
        )
        baseline_wall_values = np.asarray(
            [row["candidate_baseline_wall_ratio"] for row in gated], dtype=np.float64
        )
        memory_values = np.asarray(
            [row["candidate_baseline_memory_ratio"] for row in gated], dtype=np.float64
        )
        values = {
            "candidate_vs_probe_wall": {
                **probe_interval,
                "median_max": 1.0,
                "upper_95_max": 1.10,
                "status": "pass"
                if probe_interval["median"] <= 1.0
                and probe_interval["upper_95"] <= 1.10
                else "fail",
            },
            "candidate_vs_B_wall": {
                "median": float(np.median(baseline_wall_values)),
                "p95": float(np.quantile(baseline_wall_values, 0.95)),
                "median_max": 1.10,
                "p95_max": 1.25,
                "status": "pass"
                if np.median(baseline_wall_values) <= 1.10
                and np.quantile(baseline_wall_values, 0.95) <= 1.25
                else "fail",
            },
            "candidate_vs_B_memory": {
                "median": float(np.median(memory_values)),
                "p95": float(np.quantile(memory_values, 0.95)),
                "maximum": float(np.max(memory_values)),
                "median_max": 1.15,
                "p95_max": 1.25,
                "maximum_max": 1.50,
                "status": "pass"
                if np.median(memory_values) <= 1.15
                and np.quantile(memory_values, 0.95) <= 1.25
                and np.max(memory_values) <= 1.50
                else "fail",
            },
        }
        return {
            "status": "pass"
            if all(value["status"] == "pass" for value in values.values())
            else "fail",
            "gates": values,
            "per_backbone_faster_status": "established"
            if probe_interval["upper_95"] < 1.0
            else "not_established",
            "per_backbone_faster_claim": {
                "comparator_id": str(probe_id),
                "metric": "candidate_probe_wall_ratio",
                "upper_95": probe_interval["upper_95"],
                "rule": "upper_95 < 1.0",
                "promotion_veto": False,
            },
            "bootstrap_block": "complete_repeat_preserving_arm_backbone_budget",
        }

    arm_gates = {
        str(arm): gate_values(
            [row for row in ratios if str(row["arm"]) == str(arm)]
        )
        for arm in arms
    }

    # Product-facing timing is paid for the complete 10-backbone ranking
    # panel. Aggregate clocks before taking ratios so a fast/slow backbone
    # cannot be implicitly reweighted by its comparator's call duration.
    full_panel_ratios: list[dict[str, Any]] = []
    for arm in arms:
        for budget in budgets:
            for repeat in repeats:
                candidate_wall = sum(
                    _finite_float(
                        lookup[
                            (
                                str(candidate_id),
                                str(backbone),
                                str(arm),
                                int(budget),
                                int(repeat),
                            )
                        ]["total_wall_seconds"],
                        field="total_wall_seconds",
                    )
                    for backbone in backbones
                )
                probe_wall = sum(
                    _finite_float(
                        lookup[
                            (
                                str(probe_id),
                                str(backbone),
                                str(arm),
                                int(budget),
                                int(repeat),
                            )
                        ]["total_wall_seconds"],
                        field="total_wall_seconds",
                    )
                    for backbone in backbones
                )
                baseline_wall = sum(
                    _finite_float(
                        lookup[
                            (
                                str(baseline_id),
                                str(backbone),
                                str(arm),
                                int(budget),
                                int(repeat),
                            )
                        ]["total_wall_seconds"],
                        field="total_wall_seconds",
                    )
                    for backbone in backbones
                )
                candidate_cpu = sum(
                    _finite_float(
                        lookup[
                            (
                                str(candidate_id),
                                str(backbone),
                                str(arm),
                                int(budget),
                                int(repeat),
                            )
                        ]["total_cpu_seconds"],
                        field="total_cpu_seconds",
                    )
                    for backbone in backbones
                )
                probe_cpu = sum(
                    _finite_float(
                        lookup[
                            (
                                str(probe_id),
                                str(backbone),
                                str(arm),
                                int(budget),
                                int(repeat),
                            )
                        ]["total_cpu_seconds"],
                        field="total_cpu_seconds",
                    )
                    for backbone in backbones
                )
                baseline_cpu = sum(
                    _finite_float(
                        lookup[
                            (
                                str(baseline_id),
                                str(backbone),
                                str(arm),
                                int(budget),
                                int(repeat),
                            )
                        ]["total_cpu_seconds"],
                        field="total_cpu_seconds",
                    )
                    for backbone in backbones
                )
                if min(probe_wall, baseline_wall, probe_cpu, baseline_cpu) <= 0.0:
                    raise ValueError("full-panel runtime denominators must be positive")
                full_panel_ratios.append(
                    {
                        "arm": str(arm),
                        "budget": int(budget),
                        "repeat": int(repeat),
                        "backbone_count": len(tuple(backbones)),
                        "candidate_wall_seconds": float(candidate_wall),
                        "probe_wall_seconds": float(probe_wall),
                        "baseline_wall_seconds": float(baseline_wall),
                        "candidate_probe_wall_ratio": float(candidate_wall / probe_wall),
                        "candidate_baseline_wall_ratio": float(
                            candidate_wall / baseline_wall
                        ),
                        "candidate_cpu_seconds": float(candidate_cpu),
                        "probe_cpu_seconds": float(probe_cpu),
                        "baseline_cpu_seconds": float(baseline_cpu),
                        "candidate_probe_cpu_ratio": float(candidate_cpu / probe_cpu),
                        "candidate_baseline_cpu_ratio": float(
                            candidate_cpu / baseline_cpu
                        ),
                    }
                )

    def full_panel_gate_values(
        arm_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        gated = [row for row in arm_rows if int(row["budget"]) >= 128]
        probe_interval = _bootstrap_median_by_repeat(
            gated,
            value_field="candidate_probe_wall_ratio",
            n_resamples=n_resamples,
            seed=seed,
            bootstrap_unit="complete_runtime_repeat_block_preserving_arm_budget_full_panel",
        )
        baseline_values = np.asarray(
            [row["candidate_baseline_wall_ratio"] for row in gated],
            dtype=np.float64,
        )
        gates = {
            "candidate_vs_probe_wall": {
                **probe_interval,
                "median_max": 1.0,
                "upper_95_max": 1.10,
                "status": "pass"
                if probe_interval["median"] <= 1.0
                and probe_interval["upper_95"] <= 1.10
                else "fail",
            },
            "candidate_vs_B_wall": {
                "median": float(np.median(baseline_values)),
                "p95": float(np.quantile(baseline_values, 0.95)),
                "median_max": 1.10,
                "p95_max": 1.25,
                "status": "pass"
                if np.median(baseline_values) <= 1.10
                and np.quantile(baseline_values, 0.95) <= 1.25
                else "fail",
            },
        }
        return {
            "status": "pass"
            if all(value["status"] == "pass" for value in gates.values())
            else "fail",
            "gates": gates,
            "bootstrap_block": "complete_repeat_preserving_arm_budget_full_panel",
            "backbone_count": len(tuple(backbones)),
        }

    full_panel_arm_gates = {
        str(arm): full_panel_gate_values(
            [row for row in full_panel_ratios if str(row["arm"]) == str(arm)]
        )
        for arm in arms
    }
    for arm in arms:
        arm_key = str(arm)
        per_backbone = arm_gates[arm_key]
        full_panel = full_panel_arm_gates[arm_key]
        panel_probe = full_panel["gates"]["candidate_vs_probe_wall"]
        panel_faster = (
            "established" if panel_probe["upper_95"] < 1.0 else "not_established"
        )
        per_backbone["full_panel_faster_status"] = panel_faster
        per_backbone["full_panel_faster_claim"] = {
            "comparator_id": str(probe_id),
            "metric": "full_panel_candidate_probe_wall_ratio",
            "upper_95": panel_probe["upper_95"],
            "rule": "upper_95 < 1.0",
            "promotion_veto": False,
        }
        per_backbone["faster_claim_status"] = (
            "established"
            if per_backbone["per_backbone_faster_status"] == "established"
            and panel_faster == "established"
            else "not_established"
        )
        per_backbone["faster_claim"] = {
            "rule": "per_backbone_upper_95 < 1.0 and full_panel_upper_95 < 1.0",
            "per_backbone_upper_95": per_backbone["per_backbone_faster_claim"][
                "upper_95"
            ],
            "full_panel_upper_95": panel_probe["upper_95"],
            "promotion_veto": False,
        }

    def descriptive(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)),
        }

    secondary_arm_summary: dict[str, Any] = {}
    for arm in arms:
        arm_key = str(arm)
        per_call_rows = [
            row
            for row in ratios
            if row["arm"] == arm_key and int(row["budget"]) >= 128
        ]
        panel_rows = [
            row
            for row in full_panel_ratios
            if row["arm"] == arm_key and int(row["budget"]) >= 128
        ]
        secondary_arm_summary[arm_key] = {
            "inferential": False,
            "promotion_gate": False,
            "per_backbone_cpu_ratios": {
                "candidate_vs_probe": descriptive(
                    [row["candidate_probe_cpu_ratio"] for row in per_call_rows]
                ),
                "candidate_vs_B": descriptive(
                    [row["candidate_baseline_cpu_ratio"] for row in per_call_rows]
                ),
            },
            "full_panel_cpu_ratios": {
                "candidate_vs_probe": descriptive(
                    [row["candidate_probe_cpu_ratio"] for row in panel_rows]
                ),
                "candidate_vs_B": descriptive(
                    [row["candidate_baseline_cpu_ratio"] for row in panel_rows]
                ),
            },
            "stage_seconds": {
                field: descriptive([row[field] for row in per_call_rows])
                for field in (
                    "candidate_conditioning_fit_wall_seconds",
                    "candidate_conditioning_fit_cpu_seconds",
                    "candidate_oi_fit_wall_seconds",
                    "candidate_oi_fit_cpu_seconds",
                    "candidate_score_fixed_wall_seconds",
                    "candidate_score_fixed_cpu_seconds",
                    "probe_fit_wall_seconds",
                    "probe_fit_cpu_seconds",
                    "probe_score_fixed_wall_seconds",
                    "probe_score_fixed_cpu_seconds",
                )
            },
        }

    secondary_scaling = []
    for arm in arms:
        for budget in budgets:
            budget_rows = [
                row
                for row in ratios
                if row["arm"] == str(arm) and int(row["budget"]) == int(budget)
            ]
            secondary_scaling.append(
                {
                    "arm": str(arm),
                    "budget": int(budget),
                    "inferential": False,
                    "promotion_gate": False,
                    "candidate_probe_cpu_ratio_median": descriptive(
                        [row["candidate_probe_cpu_ratio"] for row in budget_rows]
                    )["median"],
                    "candidate_B_cpu_ratio_median": descriptive(
                        [row["candidate_baseline_cpu_ratio"] for row in budget_rows]
                    )["median"],
                    **{
                        f"{field}_median": descriptive(
                            [row[field] for row in budget_rows]
                        )["median"]
                        for field in (
                            "candidate_conditioning_fit_wall_seconds",
                            "candidate_oi_fit_wall_seconds",
                            "candidate_score_fixed_wall_seconds",
                            "probe_fit_wall_seconds",
                            "probe_score_fixed_wall_seconds",
                        )
                    },
                }
            )
    pooled_gated = [row for row in ratios if int(row["budget"]) >= 128]
    pooled_summary = {
        "inferential": False,
        "candidate_probe_wall_median": float(
            np.median([row["candidate_probe_wall_ratio"] for row in pooled_gated])
        ),
        "candidate_B_wall_median": float(
            np.median([row["candidate_baseline_wall_ratio"] for row in pooled_gated])
        ),
        "candidate_B_memory_median": float(
            np.median([row["candidate_baseline_memory_ratio"] for row in pooled_gated])
        ),
    }
    scaling_by_budget = []
    for budget in budgets:
        budget_rows = [row for row in ratios if int(row["budget"]) == int(budget)]
        scaling_by_budget.append(
            {
                "budget": int(budget),
                "inferential": False,
                "candidate_probe_wall_median": float(
                    np.median(
                        np.asarray(
                            [row["candidate_probe_wall_ratio"] for row in budget_rows],
                            dtype=np.float64,
                        )
                    )
                ),
                "candidate_B_wall_median": float(
                    np.median(
                        np.asarray(
                            [row["candidate_baseline_wall_ratio"] for row in budget_rows],
                            dtype=np.float64,
                        )
                    )
                ),
                "candidate_B_memory_median": float(
                    np.median(
                        np.asarray(
                            [row["candidate_baseline_memory_ratio"] for row in budget_rows],
                            dtype=np.float64,
                        )
                    )
                ),
            }
        )
    return {
        "candidate_id": str(candidate_id),
        "status": "pass"
        if all(value["status"] == "pass" for value in arm_gates.values())
        and all(value["status"] == "pass" for value in full_panel_arm_gates.values())
        else "fail",
        "gate_budgets": [int(value) for value in budgets if int(value) >= 128],
        "arm_gates": arm_gates,
        "full_panel_arm_gates": full_panel_arm_gates,
        "secondary_runtime": {
            "inferential": False,
            "promotion_gate": False,
            "arm_summary": secondary_arm_summary,
            "scaling_by_arm_budget": secondary_scaling,
        },
        "pooled_summary": pooled_summary,
        "scaling_by_budget": scaling_by_budget,
        "ratio_rows": ratios,
        "full_panel_ratio_rows": full_panel_ratios,
    }


def finalize_development_lock(
    provisional: Mapping[str, Any],
    *,
    resource_summary: Mapping[str, Any],
    unrefined_vs_refined_runtime_ratio_by_arm: Mapping[str, Any] | None = None,
    fusion_gate_summary: Mapping[str, Any] | None = None,
    guardrail_gate_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalize one chain; any runtime/fusion failure stops with no runner-up."""

    if provisional.get("status") != "provisional_runtime_pending":
        raise ValueError("finalization requires a provisional_runtime_pending decision")
    if provisional.get("selected_chain") is not None:
        raise ValueError("finalization requires provisional selected_chain=null")
    if provisional.get("runtime_pending") is not True:
        raise ValueError("finalization requires runtime_pending=true")
    if provisional.get("runner_up_allowed") is not False:
        raise ValueError("finalization requires runner_up_allowed=false")
    if "locked_candidate" in provisional:
        raise ValueError("finalization rejects locked_candidate compatibility aliases")
    selected = provisional.get("selected_candidate")
    if selected not in SW_CANDIDATES:
        raise ValueError("finalization requires selected_candidate M0-SW or M1-SW")
    selected = str(selected)
    if provisional.get("unrefined_requires_speed_ratio"):
        if not isinstance(unrefined_vs_refined_runtime_ratio_by_arm, Mapping) or set(
            unrefined_vs_refined_runtime_ratio_by_arm
        ) != set(FOOD_ARMS):
            raise ValueError(
                "unrefined refinement lock requires exact per-arm M0/M1 runtime ratios"
            )
        ratio_by_arm = {
            arm: _finite_float(
                unrefined_vs_refined_runtime_ratio_by_arm[arm],
                field=f"unrefined_vs_refined_runtime_ratio_by_arm[{arm!r}]",
            )
            for arm in FOOD_ARMS
        }
        if any(value > REFINEMENT_SPEED_RATIO_MAX for value in ratio_by_arm.values()):
            selected = "M1-SW"
    else:
        ratio_by_arm = None
    needs_fusion = bool(
        provisional.get("candidate_gates", {})
        .get(selected, {})
        .get("requires_fusion")
    )
    needs_guardrail = bool(
        provisional.get("candidate_gates", {})
        .get(selected, {})
        .get("guardrail_eligible")
    )
    chain = "F" if needs_fusion else ("G" if needs_guardrail else selected)
    if needs_fusion:
        fusion_status = None if fusion_gate_summary is None else fusion_gate_summary.get("status")
        if fusion_status == "pass":
            chain = "F"
        elif fusion_status == "guardrail_eligible":
            chain, needs_guardrail = "G", True
        else:
            return {
                **dict(provisional),
                "status": "stopped_fusion_failure",
                "selected_candidate": selected,
                "selected_chain": None,
                "chain_kind": None,
                "algorithmic_oi_status": "fail",
                "product_policy_status": "not_evaluated",
                "runner_up_allowed": False,
            }
    if needs_guardrail:
        if guardrail_gate_summary is None or guardrail_gate_summary.get("status") != "pass":
            return {
                **dict(provisional),
                "status": "stopped_guardrail_failure",
                "selected_candidate": selected,
                "selected_chain": None,
                "chain_kind": None,
                "algorithmic_oi_status": "fail_linear_claim",
                "product_policy_status": "fail",
                "runner_up_allowed": False,
            }
    if resource_summary.get("candidate_id") != chain:
        raise ValueError("resource summary candidate does not match the provisional chain")
    if resource_summary.get("status") != "pass":
        return {
            **dict(provisional),
            "status": "stopped_resource_failure",
            "selected_candidate": selected,
            "selected_chain": None,
            "chain_kind": None,
            "algorithmic_oi_status": "fail_resource"
            if chain != "G"
            else "fail_linear_claim",
            "product_policy_status": "fail_resource"
            if chain == "G"
            else "not_applicable",
            "runner_up_allowed": False,
        }
    chain_kind = (
        "pure_oi"
        if chain in SW_CANDIDATES
        else ("oi_fusion" if chain == "F" else "product_guardrail")
    )
    return {
        **dict(provisional),
        "status": "locked_for_future_confirmation",
        "selected_candidate": selected,
        "selected_chain": chain,
        "chain_kind": chain_kind,
        "algorithmic_oi_status": "pass" if chain != "G" else "fail_linear_claim",
        "product_policy_status": "pass" if chain == "G" else "not_applicable",
        "runtime_pending": False,
        "resource_status": "pass",
        "runner_up_allowed": False,
        "confirmation_may_reselect": False,
        "refinement_runtime_ratio_by_arm": ratio_by_arm,
    }


def hierarchical_dataset_replicate_interval(
    values: Iterable[Mapping[str, Any]],
    *,
    expected_datasets: Sequence[str],
    expected_replicates: Sequence[int],
    expected_budgets: Sequence[int] = (32, 64),
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = CONFIRMATION_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Equal-dataset/equal-budget hierarchical confirmation interval."""

    rows = _materialize(values, name="confirmation contrast rows")
    expected_dataset_set = set(str(value) for value in expected_datasets)
    if len(expected_dataset_set) != 5:
        raise ValueError("untouched confirmation requires exactly five datasets")
    budgets = tuple(int(value) for value in expected_budgets)
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("expected_budgets must contain unique values")
    grouped: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        dataset = str(row.get("dataset"))
        replicate = _strict_int(row.get("replicate"), field="replicate")
        budget = _strict_int(row.get("budget"), field="budget")
        if dataset not in expected_dataset_set:
            raise ValueError(f"unexpected confirmation dataset {dataset!r}")
        if replicate not in expected_replicates:
            raise ValueError(f"unexpected confirmation replicate {replicate!r}")
        if budget not in budgets:
            raise ValueError(f"unexpected confirmation budget {budget!r}")
        if budget in grouped[dataset][replicate]:
            raise ValueError(
                f"duplicate confirmation identity {(dataset, replicate, budget)!r}"
            )
        grouped[dataset][replicate][budget] = _finite_float(
            row.get("contrast_pp"), field="contrast_pp"
        )
    for dataset in expected_dataset_set:
        if set(grouped.get(dataset, {})) != set(int(value) for value in expected_replicates):
            raise ValueError(f"confirmation dataset {dataset!r} has incomplete replicate coverage")
        for replicate in expected_replicates:
            if set(grouped[dataset][int(replicate)]) != set(budgets):
                raise ValueError(
                    f"confirmation dataset {dataset!r} replicate {replicate!r} "
                    "has incomplete budget coverage"
                )

    dataset_names = tuple(sorted(expected_dataset_set))
    replicate_names = tuple(int(value) for value in expected_replicates)
    per_dataset = {
        dataset: float(
            np.mean(
                [
                    np.mean(list(grouped[dataset][replicate].values()))
                    for replicate in replicate_names
                ]
            )
        )
        for dataset in dataset_names
    }
    generator = np.random.default_rng(int(seed))
    boot = np.empty(int(n_resamples), dtype=np.float64)
    for draw in range(int(n_resamples)):
        sampled_datasets = generator.choice(dataset_names, size=len(dataset_names), replace=True)
        dataset_means = []
        for dataset in sampled_datasets:
            sampled_replicates = generator.choice(
                replicate_names, size=len(replicate_names), replace=True
            )
            dataset_means.append(
                float(
                    np.mean(
                        [
                            np.mean(
                                list(grouped[str(dataset)][int(replicate)].values())
                            )
                            for replicate in sampled_replicates
                        ]
                    )
                )
            )
        boot[draw] = float(np.mean(dataset_means))
    return {
        "estimate": float(np.mean(list(per_dataset.values()))),
        "lower_95": float(np.quantile(boot, 0.025)),
        "upper_95": float(np.quantile(boot, 0.975)),
        "per_dataset": per_dataset,
        "n_datasets": len(dataset_names),
        "n_replicates": len(replicate_names),
        "budgets": list(budgets),
        "n_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": "dataset_then_complete_replicate",
        "aggregation": "equal_dataset_then_equal_budget",
    }


def confirmation_gates(
    contrast_rows: Iterable[Mapping[str, Any]],
    *,
    development_lock: Mapping[str, Any],
    expected_datasets: Sequence[str],
    expected_replicates: Sequence[int],
    expected_budgets: Sequence[int] = (32, 64),
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = CONFIRMATION_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Confirm exactly the locked chain; never rank or replace candidates."""

    if development_lock.get("status") != "locked_for_future_confirmation":
        raise ValueError(
            "confirmation requires the exact locked_for_future_confirmation status"
        )
    base = development_lock.get("selected_candidate")
    locked = development_lock.get("selected_chain")
    if base not in SW_CANDIDATES or locked not in {*SW_CANDIDATES, "F", "G"}:
        raise ValueError("confirmation lock lacks exact selected_candidate/selected_chain")
    expected_kind = (
        "pure_oi"
        if locked in SW_CANDIDATES
        else ("oi_fusion" if locked == "F" else "product_guardrail")
    )
    if development_lock.get("chain_kind") != expected_kind:
        raise ValueError("confirmation lock chain_kind does not match selected_chain")
    expected_algorithmic = "pass" if locked != "G" else "fail_linear_claim"
    expected_policy = "pass" if locked == "G" else "not_applicable"
    if (
        development_lock.get("algorithmic_oi_status") != expected_algorithmic
        or development_lock.get("product_policy_status") != expected_policy
    ):
        raise ValueError("confirmation lock has inconsistent algorithmic/policy status")
    locked = str(locked)
    rows = _materialize(contrast_rows, name="contrast_rows")
    if {str(row.get("candidate_id")) for row in rows} != {locked}:
        raise ValueError("confirmation rows must contain exactly the locked chain")
    heads = {str(row.get("head")) for row in rows}
    if heads != set(REFERENCE_HEADS):
        raise ValueError("confirmation requires all four frozen reference heads")
    expected_identities = {
        (str(dataset), int(replicate), int(budget), str(head))
        for dataset in expected_datasets
        for replicate in expected_replicates
        for budget in expected_budgets
        for head in REFERENCE_HEADS
    }
    observed_identities: set[tuple[str, int, int, str]] = set()
    for row in rows:
        identity = (
            str(row.get("dataset")),
            _strict_int(row.get("replicate"), field="replicate"),
            _strict_int(row.get("budget"), field="budget"),
            str(row.get("head")),
        )
        if identity in observed_identities:
            raise ValueError(f"duplicate confirmation identity {identity!r}")
        observed_identities.add(identity)
    if observed_identities != expected_identities:
        raise ValueError("confirmation requires exact complete dataset/replicate/budget/head identities")
    gates: dict[str, Any] = {}
    for head in REFERENCE_HEADS:
        interval = hierarchical_dataset_replicate_interval(
            (row for row in rows if str(row.get("head")) == head),
            expected_datasets=expected_datasets,
            expected_replicates=expected_replicates,
            expected_budgets=expected_budgets,
            n_resamples=n_resamples,
            seed=seed,
        )
        if head == "linear":
            passed = interval["upper_95"] <= 1.0 and max(interval["per_dataset"].values()) <= 2.0
            detail = "pooled upper_95 <= 1 pp and no dataset point estimate > 2 pp"
        else:
            favorable = sum(value < 0.0 for value in interval["per_dataset"].values())
            passed = interval["upper_95"] < 0.0 and favorable >= 4
            interval["favorable_dataset_count"] = favorable
            detail = "pooled upper_95 < 0 and at least four datasets favor locked selector"
        gates[head] = {**interval, "status": "pass" if passed else "fail", "rule": detail}
    return {
        "candidate_id": locked,
        "selected_candidate": str(base),
        "selected_chain": locked,
        "chain_kind": expected_kind,
        "algorithmic_oi_status": development_lock.get("algorithmic_oi_status"),
        "product_policy_status": development_lock.get("product_policy_status"),
        "status": "pass" if all(value["status"] == "pass" for value in gates.values()) else "fail",
        "gates": gates,
        "runner_up_allowed": False,
        "reselection_performed": False,
        "dataset_count": len(set(str(value) for value in expected_datasets)),
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CB_CANDIDATES",
    "CONFIRMATION_BOOTSTRAP_SEED",
    "FACTORIAL_CANDIDATES",
    "FOOD_ARMS",
    "FOOD_BACKBONES",
    "FOOD_BUDGETS",
    "FOOD_REPLICATES",
    "FOOD_RUNTIME_BUDGETS",
    "FoodDesign",
    "PRIMARY_CELLS",
    "REFERENCE_HEADS",
    "RUNTIME_PRIMITIVE_IDS",
    "SW_CANDIDATES",
    "candidate_product_gates",
    "confirmation_gates",
    "factorial_effects",
    "finalize_development_lock",
    "food_panel_metrics",
    "hierarchical_dataset_replicate_interval",
    "paired_replicate_interval",
    "provisional_development_lock",
    "rank_auc_rows",
    "rank_auc_summary",
    "refinement_noninferiority",
    "regret_contrast",
    "resource_gates",
    "selection_rate_summary",
    "validate_food_inputs",
]
