"""Outcome-blind panel policies for the M50 backbone-ranking experiment.

The functions in this module deliberately operate on selector scores and
training-only diagnostics.  They never accept reference-head outcomes, so the
fusion and guardrail policies cannot accidentally trigger on held-out accuracy
or regret.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


FUSION_CANDIDATE_ID = "F"
GUARDRAIL_CANDIDATE_ID = "G"
FULL_PROBE_CANDIDATE_ID = "LP-FULL"
CAPPED_PROBE_CANDIDATE_ID = "LP-CAPPED-2048"

PANEL_IDENTITY_FIELDS = ("replicate", "arm", "budget")
FORBIDDEN_TRIGGER_FIELDS = frozenset(
    {
        "accuracy",
        "best_accuracy",
        "exact_best",
        "head",
        "reference_accuracy",
        "regret",
        "regret_pp",
        "test_accuracy",
        "within_one_pp",
    }
)


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


def fractional_midranks(
    scores: Mapping[str, Any],
    *,
    item_order: Sequence[str],
) -> dict[str, float]:
    """Return ascending fractional midranks in ``[0, 1]``.

    Higher input scores receive higher ranks.  Tied scores receive their
    average rank.  ``item_order`` defines the exact required item set, but is
    not used to break score ties; final selection tie-breaking is separate.
    """

    order = tuple(str(item) for item in item_order)
    if len(order) < 2 or len(set(order)) != len(order):
        raise ValueError("item_order must contain at least two unique items")
    if set(scores) != set(order):
        missing = sorted(set(order) - set(scores))
        extra = sorted(set(scores) - set(order))
        raise ValueError(
            f"scores must match item_order exactly; missing={missing!r}, extra={extra!r}"
        )
    values = np.asarray(
        [_finite_float(scores[item], field=f"score[{item!r}]") for item in order],
        dtype=np.float64,
    )
    sort_order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[sort_order[end]] == values[sort_order[start]]:
            end += 1
        ranks[sort_order[start:end]] = 0.5 * (start + end - 1)
        start = end
    ranks /= float(values.size - 1)
    return {item: float(ranks[position]) for position, item in enumerate(order)}


def fuse_panel_scores(
    raw_scores: Mapping[str, Any],
    m50_scores: Mapping[str, Any],
    *,
    item_order: Sequence[str],
) -> dict[str, float]:
    """Apply the frozen 50/50 raw-plus-M50 fractional-midrank fusion."""

    raw_ranks = fractional_midranks(raw_scores, item_order=item_order)
    m50_ranks = fractional_midranks(m50_scores, item_order=item_order)
    return {
        item: float(0.5 * raw_ranks[item] + 0.5 * m50_ranks[item])
        for item in item_order
    }


def select_highest_score(
    scores: Mapping[str, Any],
    *,
    item_order: Sequence[str],
) -> str:
    """Select the largest score, using frozen item order only for final ties."""

    order = tuple(str(item) for item in item_order)
    if set(scores) != set(order):
        raise ValueError("scores must contain the exact frozen item set")
    values = {item: _finite_float(scores[item], field="score") for item in order}
    maximum = max(values.values())
    return next(item for item in order if values[item] == maximum)


def _panel_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return tuple(row[field] for field in PANEL_IDENTITY_FIELDS)
    except KeyError as exc:
        raise ValueError(f"selector row is missing panel identity field {exc.args[0]!r}") from exc


def _primitive_timings(row: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Return validated primitive timings, constructing them for primitive rows."""

    value = row.get("primitive_timings")
    if value is None:
        candidate = str(row.get("candidate_id"))
        if candidate in {FUSION_CANDIDATE_ID, GUARDRAIL_CANDIDATE_ID}:
            raise ValueError(f"derived candidate {candidate!r} requires primitive_timings")
        return {
            candidate: {
                "wall_seconds": _finite_float(
                    row.get("total_wall_seconds"), field="total_wall_seconds"
                ),
                "cpu_seconds": _finite_float(
                    row.get("total_cpu_seconds"), field="total_cpu_seconds"
                ),
            }
        }
    if not isinstance(value, Mapping) or not value:
        raise ValueError("primitive_timings must be a non-empty mapping")
    result: dict[str, dict[str, float]] = {}
    for component, timing in value.items():
        if not isinstance(timing, Mapping):
            raise ValueError("each primitive timing must be a mapping")
        result[str(component)] = {
            "wall_seconds": _finite_float(
                timing.get("wall_seconds"), field="primitive wall_seconds"
            ),
            "cpu_seconds": _finite_float(
                timing.get("cpu_seconds"), field="primitive cpu_seconds"
            ),
        }
    return result


def _union_primitive_timings(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        for component, timing in _primitive_timings(row).items():
            if component in result and result[component] != timing:
                raise ValueError(
                    f"conflicting timing for shared primitive component {component!r}"
                )
            result[component] = timing
    return result


def derive_fusion_rows(
    selector_rows: Iterable[Mapping[str, Any]],
    *,
    raw_candidate_id: str,
    m50_candidate_id: str,
    backbones: Sequence[str],
) -> list[dict[str, Any]]:
    """Derive complete per-backbone fusion rows from selector-only scores."""

    materialized = [dict(row) for row in selector_rows]
    allowed = {str(raw_candidate_id), str(m50_candidate_id)}
    by_panel: dict[tuple[Any, ...], dict[str, dict[str, Mapping[str, Any]]]] = {}
    for row in materialized:
        candidate = str(row.get("candidate_id"))
        if candidate not in allowed:
            continue
        backbone = str(row.get("backbone"))
        panel = _panel_key(row)
        slot = by_panel.setdefault(panel, {}).setdefault(candidate, {})
        if backbone in slot:
            raise ValueError(
                f"duplicate fusion source row for panel={panel!r}, "
                f"candidate={candidate!r}, backbone={backbone!r}"
            )
        slot[backbone] = row

    expected = tuple(str(value) for value in backbones)
    rows: list[dict[str, Any]] = []
    for panel in sorted(by_panel, key=repr):
        candidates = by_panel[panel]
        if set(candidates) != allowed:
            raise ValueError(f"fusion panel {panel!r} is missing a required view")
        view_scores: dict[str, dict[str, float]] = {}
        for candidate in (str(raw_candidate_id), str(m50_candidate_id)):
            observed = candidates[candidate]
            if set(observed) != set(expected):
                raise ValueError(
                    f"fusion panel {panel!r} candidate {candidate!r} does not "
                    "contain the exact frozen backbone set"
                )
            view_scores[candidate] = {
                backbone: _finite_float(observed[backbone].get("score"), field="score")
                for backbone in expected
            }
        fused = fuse_panel_scores(
            view_scores[str(raw_candidate_id)],
            view_scores[str(m50_candidate_id)],
            item_order=expected,
        )
        selected = select_highest_score(fused, item_order=expected)
        for backbone in expected:
            primitives = _union_primitive_timings(
                (
                    candidates[str(raw_candidate_id)][backbone],
                    candidates[str(m50_candidate_id)][backbone],
                )
            )
            rows.append(
                {
                    **dict(zip(PANEL_IDENTITY_FIELDS, panel)),
                    "backbone": backbone,
                    "candidate_id": FUSION_CANDIDATE_ID,
                    "score": fused[backbone],
                    "selected": backbone == selected,
                    "raw_candidate_id": str(raw_candidate_id),
                    "m50_candidate_id": str(m50_candidate_id),
                    "aggregation": "equal_fractional_midrank",
                    "primitive_timings": primitives,
                    "total_wall_seconds": float(
                        sum(value["wall_seconds"] for value in primitives.values())
                    ),
                    "total_cpu_seconds": float(
                        sum(value["cpu_seconds"] for value in primitives.values())
                    ),
                }
            )
    return rows


def _assert_trigger_safe(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject outcome-bearing keys at any nesting depth."""

    forbidden: set[str] = set()

    def inspect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).lower()
                if normalized in FORBIDDEN_TRIGGER_FIELDS:
                    forbidden.add(str(key))
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)

    inspect(rows)
    found = sorted(forbidden)
    if found:
        raise ValueError(
            "guardrail trigger rows must not contain reference/outcome fields; "
            f"found {found!r}"
        )


def guardrail_trigger(
    diagnostic_rows: Iterable[Mapping[str, Any]],
    *,
    backbones: Sequence[str],
    refined_candidate_id: str = "B",
    conditioned_candidate_id: str = "M1-SW",
    refinement_threshold: float = 0.50,
    disagreement_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate the frozen training-diagnostic-only panel trigger.

    The input is exactly one panel containing B and M1-SW rows for all
    backbones.  Triggering is panel-wide: one flagged backbone sends every
    backbone to the capped probe.
    """

    if refined_candidate_id != "B" or conditioned_candidate_id != "M1-SW":
        raise ValueError("guardrail diagnostics are frozen to exact B and M1-SW inputs")
    rows = [dict(row) for row in diagnostic_rows]
    _assert_trigger_safe(rows)
    expected = tuple(str(value) for value in backbones)
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        candidate = str(row.get("candidate_id"))
        allowed_fields = (
            {"candidate_id", "backbone", "score", "prototype_refinement"}
            if candidate == "B"
            else {"candidate_id", "backbone", "score"}
        )
        if candidate not in {"B", "M1-SW"} or set(row) != allowed_fields:
            raise ValueError(
                "guardrail trigger rows must use the exact closed B/M1-SW schema"
            )
        key = (str(row.get("candidate_id")), str(row.get("backbone")))
        if key in lookup:
            raise ValueError(f"duplicate guardrail diagnostic row {key!r}")
        lookup[key] = row
    required = {
        (candidate, backbone)
        for candidate in (str(refined_candidate_id), str(conditioned_candidate_id))
        for backbone in expected
    }
    if set(lookup) != required:
        raise ValueError("guardrail diagnostics must contain exactly two views per backbone")

    flagged: list[dict[str, Any]] = []
    for backbone in expected:
        raw = lookup[(str(refined_candidate_id), backbone)]
        conditioned = lookup[(str(conditioned_candidate_id), backbone)]
        refinement = raw.get("prototype_refinement")
        if not isinstance(refinement, Mapping) or set(refinement) != {"applied_rate"}:
            raise ValueError("B guardrail rows require prototype_refinement diagnostics")
        activity = _finite_float(refinement.get("applied_rate"), field="applied_rate")
        raw_score = _finite_float(raw.get("score"), field="B score")
        conditioned_score = _finite_float(
            conditioned.get("score"), field="M1-SW score"
        )
        disagreement = abs(raw_score - conditioned_score)
        if activity >= float(refinement_threshold) or disagreement >= float(
            disagreement_threshold
        ):
            flagged.append(
                {
                    "backbone": backbone,
                    "refinement_applied_rate": activity,
                    "raw_conditioned_disagreement": disagreement,
                    "activity_triggered": activity >= float(refinement_threshold),
                    "disagreement_triggered": disagreement
                    >= float(disagreement_threshold),
                }
            )
    return {
        "triggered": bool(flagged),
        "triggered_backbones": flagged,
        "refinement_threshold": float(refinement_threshold),
        "disagreement_threshold": float(disagreement_threshold),
        "trigger_inputs": "training_diagnostics_only",
    }


def derive_guardrail_rows(
    *,
    diagnostic_rows: Iterable[Mapping[str, Any]],
    base_rows: Iterable[Mapping[str, Any]],
    capped_probe_rows: Iterable[Mapping[str, Any]],
    backbones: Sequence[str],
    base_candidate_id: str,
) -> list[dict[str, Any]]:
    """Derive one guardrail panel and charge every uniquely computed component."""

    diagnostics = [dict(row) for row in diagnostic_rows]
    bases = [dict(row) for row in base_rows]
    capped = [dict(row) for row in capped_probe_rows]
    _assert_trigger_safe(diagnostics)
    expected = tuple(str(value) for value in backbones)
    if base_candidate_id not in {"M0-SW", "M1-SW", "F"}:
        raise ValueError("base_candidate_id must be the exact settled M0-SW, M1-SW, or F")

    all_rows = [*diagnostics, *bases, *capped]
    if not all_rows:
        raise ValueError("guardrail inputs must not be empty")
    panel_keys = {_panel_key(row) for row in all_rows}
    if len(panel_keys) != 1:
        raise ValueError("guardrail inputs must contain exactly one aligned panel")

    def one_per_backbone(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            backbone = str(row.get("backbone"))
            if backbone in result:
                raise ValueError(f"duplicate {label} row for backbone {backbone!r}")
            result[backbone] = row
        if set(result) != set(expected):
            raise ValueError(f"{label} rows must contain the exact frozen backbone set")
        return result

    base_lookup = one_per_backbone(bases, "base")
    cap_lookup = one_per_backbone(capped, "capped probe")
    if {str(row.get("candidate_id")) for row in bases} != {base_candidate_id}:
        raise ValueError("guardrail base rows do not match exact base_candidate_id")
    if {str(row.get("candidate_id")) for row in capped} != {
        CAPPED_PROBE_CANDIDATE_ID
    }:
        raise ValueError("guardrail capped rows must be exactly LP-CAPPED-2048")
    if base_candidate_id == FUSION_CANDIDATE_ID:
        selected = [row for row in bases if row.get("selected") is True]
        if len(selected) != 1 or any(type(row.get("selected")) is not bool for row in bases):
            raise ValueError("F base rows require exactly one strict selected marker")
    diagnostic_lookup: dict[str, list[Mapping[str, Any]]] = {value: [] for value in expected}
    for row in diagnostics:
        backbone = str(row.get("backbone"))
        if backbone not in diagnostic_lookup:
            raise ValueError(f"unknown diagnostic backbone {backbone!r}")
        diagnostic_lookup[backbone].append(row)

    trigger_rows: list[dict[str, Any]] = []
    for backbone in expected:
        rows_for_backbone = diagnostic_lookup[backbone]
        by_candidate = {str(row.get("candidate_id")): row for row in rows_for_backbone}
        if len(rows_for_backbone) != 2 or set(by_candidate) != {"B", "M1-SW"}:
            raise ValueError("guardrail requires exact B and M1-SW diagnostics per backbone")
        b_row = by_candidate["B"]
        m1_row = by_candidate["M1-SW"]
        refinement = b_row.get("prototype_refinement")
        if not isinstance(refinement, Mapping):
            raise ValueError("B guardrail rows require prototype_refinement diagnostics")
        trigger_rows.extend(
            [
                {
                    "candidate_id": "B",
                    "backbone": backbone,
                    "score": b_row.get("score"),
                    "prototype_refinement": {
                        "applied_rate": refinement.get("applied_rate")
                    },
                },
                {
                    "candidate_id": "M1-SW",
                    "backbone": backbone,
                    "score": m1_row.get("score"),
                },
            ]
        )
    decision = guardrail_trigger(trigger_rows, backbones=backbones)

    source_lookup = cap_lookup if decision["triggered"] else base_lookup
    scores = {
        backbone: _finite_float(source_lookup[backbone].get("score"), field="policy score")
        for backbone in expected
    }
    if not decision["triggered"] and base_candidate_id == FUSION_CANDIDATE_ID:
        selected_backbones = {
            backbone
            for backbone in expected
            if base_lookup[backbone].get("selected") is True
        }
        selection_semantics = "derived_frozen_order_argmax"
    else:
        maximum = max(scores.values())
        selected_backbones = {
            backbone
            for backbone in expected
            if abs(scores[backbone] - maximum) <= 1e-12
        }
        selection_semantics = "archived_atol_1e-12_tie_average"
    result: list[dict[str, Any]] = []
    for backbone in expected:
        charged_rows = [*diagnostic_lookup[backbone], base_lookup[backbone]]
        if decision["triggered"]:
            charged_rows.append(cap_lookup[backbone])
        primitives = _union_primitive_timings(charged_rows)
        wall = sum(value["wall_seconds"] for value in primitives.values())
        cpu = sum(value["cpu_seconds"] for value in primitives.values())
        result.append(
            {
                **{field: base_lookup[backbone].get(field) for field in PANEL_IDENTITY_FIELDS},
                "backbone": backbone,
                "candidate_id": GUARDRAIL_CANDIDATE_ID,
                "score": scores[backbone],
                "selected": backbone in selected_backbones,
                "selection_semantics": selection_semantics,
                "triggered": bool(decision["triggered"]),
                "score_source": CAPPED_PROBE_CANDIDATE_ID
                if decision["triggered"]
                else str(base_lookup[backbone].get("candidate_id")),
                "charged_component_ids": sorted(primitives),
                "primitive_timings": primitives,
                "total_wall_seconds": float(wall),
                "total_cpu_seconds": float(cpu),
                "trigger": decision,
                "base_candidate_id": base_candidate_id,
            }
        )
    return result


__all__ = [
    "CAPPED_PROBE_CANDIDATE_ID",
    "FULL_PROBE_CANDIDATE_ID",
    "FUSION_CANDIDATE_ID",
    "GUARDRAIL_CANDIDATE_ID",
    "derive_fusion_rows",
    "derive_guardrail_rows",
    "fractional_midranks",
    "fuse_panel_scores",
    "guardrail_trigger",
    "select_highest_score",
]
