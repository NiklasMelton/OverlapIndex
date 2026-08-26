"""Deterministic Markdown, table, and SVG reporting for nuisance analysis.

The analysis module owns estimands and gate calculations.  This module is a
thin presentation layer: it reads fields emitted by ``analysis.py``, keeps
missing values explicit, and never replaces a selector or robustness surface
with a raw score proxy.
"""

from __future__ import annotations

import csv
import html
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .analysis import CANDIDATE_ORDER, canonical_json


_ROLE_INFO: dict[str, dict[str, str]] = {
    "A": {"role": "exact current OI baseline", "scope": "control", "definition": "unrefined raw OI; exact current baseline"},
    "B": {"role": "exact current refined OI baseline", "scope": "control", "definition": "refined raw OI; exact current refined baseline"},
    "C": {"role": "algorithmic OI", "scope": "promotable C–E", "definition": "refined OI with global-isotropy conditioning"},
    "D": {"role": "algorithmic OI", "scope": "promotable C–E", "definition": "refined OI with pooled diagonal conditioning"},
    "E": {"role": "algorithmic OI", "scope": "promotable C–E", "definition": "refined OI with pooled full conditioning"},
    "F": {"role": "diagnostic only", "scope": "not promotable", "definition": "raw conditioned disagreement (absolute B–E score difference)"},
    "G": {"role": "product guardrail", "scope": "not algorithmic OI; not promotable", "definition": "panel-selective capped linear-probe guardrail"},
}
_HEAD_ORDER = ("linear", "quadratic", "knn", "rbf")
_PROBE_CANDIDATE_KEYS = {"g_probe_component", "capped_probe", "full_probe", "prior_full_probe"}
_DETECTION_FIELDS = ("auroc", "auprc", "fpr", "fnr", "brier", "ece")
_RUNTIME_STAGES = ("fit", "score_fixed", "total", "fit_cpu", "score_fixed_cpu", "total_cpu", "peak_memory")
_RATIO_FIELDS = (
    "median_total_ratio_vs_B", "p95_total_ratio_vs_B", "median_score_fixed_ratio_vs_B",
    "median_peak_memory_ratio_vs_B", "p95_peak_memory_ratio_vs_B", "individual_peak_memory_ratio_vs_B",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _lookup(value: Any, *paths: str) -> Any:
    """Read exact or dotted paths without assuming one summary layout."""

    if not isinstance(value, Mapping):
        return None
    for path in paths:
        if path in value:
            return value[path]
        current: Any = value
        for part in str(path).split("."):
            if not isinstance(current, Mapping) or part not in current:
                break
            current = current[part]
        else:
            return current
    return None


def _metric(metrics: Mapping[str, Any], *path: str) -> Any:
    """Compatibility helper retained for old local callers."""

    if len(path) > 1:
        nested = _lookup(metrics, ".".join(path))
        if nested is not None:
            return nested
    return _lookup(metrics, *path)


def _estimate(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("estimate", "value", "mean", "median"):
            if key in value:
                return _number(value[key])
        return None
    return _number(value)


def _ci(value: Any, bound: str) -> float | None:
    if not isinstance(value, Mapping):
        return None
    aliases = (bound, "high" if bound == "upper" else "low", f"{bound}_bound")
    return next((_number(value[key]) for key in aliases if key in value), None)


def _status(value: Any) -> str:
    if isinstance(value, Mapping) and value.get("status") is not None:
        return str(value["status"])
    return "defined" if _estimate(value) is not None else "undefined"


def _fmt(value: Any) -> str:
    number = _number(value)
    if number is not None:
        return f"{number:.6g}"
    if value is None or value == "":
        return "—"
    return str(value)


def _cell(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def _ordered_candidates(summary: Mapping[str, Any]) -> list[str]:
    candidates = _lookup(summary, "candidates")
    if not isinstance(candidates, Mapping):
        return []
    names = [str(key) for key in candidates if str(key).lower() not in _PROBE_CANDIDATE_KEYS]
    ordered = [candidate for candidate in CANDIDATE_ORDER if candidate in names]
    ordered.extend(sorted(candidate for candidate in names if candidate not in ordered))
    return ordered


def _candidate_metrics(summary: Mapping[str, Any], candidate: str) -> Mapping[str, Any]:
    candidates = _lookup(summary, "candidates")
    if not isinstance(candidates, Mapping):
        return {}
    value = candidates.get(candidate, {})
    return value if isinstance(value, Mapping) else {}


def _role(candidate: str, metrics: Mapping[str, Any] | None = None) -> dict[str, str]:
    result = dict(_ROLE_INFO.get(candidate, {"role": "candidate", "scope": "unclassified", "definition": candidate}))
    if isinstance(metrics, Mapping):
        for key in result:
            if metrics.get(key) is not None:
                result[key] = str(metrics[key])
    return result


def _selection_heads(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = _lookup(metrics, "selection_metrics", "selector", "selection")
    if not isinstance(selection, Mapping):
        return {}
    heads = selection.get("heads", selection)
    return heads if isinstance(heads, Mapping) else {}


def _selection_value(head: Mapping[str, Any], key: str) -> Any:
    aliases = {
        "regret": ("regret", "selector_regret"),
        "rank_auc": ("rank_auc", "rankAUC", "rank_auc_score"),
        "exact_best": ("exact_best", "exact-best", "exact_best_rate"),
        "within_one_point": ("within_one_point", "within_1pp", "within_one_pp", "within_0.01"),
        "spearman": ("spearman", "rank_spearman"),
    }
    return _lookup(head, *aliases.get(key, (key,)))


def selector_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten selector regret/rank outcomes with undefined cells retained."""

    rows: list[dict[str, Any]] = []
    for candidate in _ordered_candidates(summary):
        heads = _selection_heads(_candidate_metrics(summary, candidate))
        names = [head for head in _HEAD_ORDER if head in heads]
        names.extend(sorted(str(head) for head in heads if str(head) not in names))
        for head_name in (_HEAD_ORDER if not names else names):
            head = heads.get(head_name, {})
            if not isinstance(head, Mapping):
                head = {}
            regret = _selection_value(head, "regret")
            rank_auc = _selection_value(head, "rank_auc")
            exact = _selection_value(head, "exact_best")
            within = _selection_value(head, "within_one_point")
            rows.append({
                "candidate": candidate,
                "head": str(head_name),
                "regret": _estimate(regret),
                "regret_lower": _ci(regret, "lower"),
                "regret_upper": _ci(regret, "upper"),
                "rank_auc": _estimate(rank_auc),
                "exact_best": _estimate(exact),
                "within_one_point": _estimate(within),
                "within_1pp": _estimate(within),
                "spearman": _estimate(_selection_value(head, "spearman")),
                "n": _lookup(head, "n") or _lookup(regret, "n") or _lookup(rank_auc, "n"),
                "status": "defined" if any(_estimate(value) is not None for value in (regret, rank_auc, exact, within)) else str(head.get("status", "undefined")),
            })
    return rows


def _curve_points(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _lookup(metrics, "nuisance_curve", "nuisance.curve", "nuisance_curves", "nuisance.curves")
    if isinstance(raw, Mapping):
        for key in ("points", "curve", "values", "rows"):
            if isinstance(raw.get(key), (list, tuple, Mapping)):
                raw = raw[key]
                break
        else:
            strengths = raw.get("strength", raw.get("x"))
            evidences = raw.get("evidence", raw.get("y", raw.get("value")))
            if isinstance(strengths, (list, tuple)) and isinstance(evidences, (list, tuple)):
                raw = [{"strength": strength, "evidence": evidence} for strength, evidence in zip(strengths, evidences)]
    if isinstance(raw, Mapping):
        iterable: Iterable[Any] = [{"strength": key, "evidence": value} for key, value in raw.items()]
    elif isinstance(raw, (list, tuple)):
        iterable = raw
    else:
        iterable = ()
    points: list[dict[str, Any]] = []
    for item in iterable:
        if isinstance(item, Mapping):
            strength = _lookup(item, "strength", "x", "nuisance_strength")
            evidence = _lookup(item, "evidence", "y", "value", "overlap_evidence", "designated_pair_evidence")
            count = _lookup(item, "n", "count", "n_rows")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            strength, evidence = item[0], item[1]
            count = item[2] if len(item) > 2 else None
        else:
            continue
        x, y = _estimate(strength), _estimate(evidence)
        if x is not None and y is not None:
            points.append({"strength": x, "evidence": y, "n": _estimate(count)})
    points.sort(key=lambda item: (float(item["strength"]), float(item["evidence"])))
    return points


def _robustness_auc(metrics: Mapping[str, Any]) -> Any:
    return _lookup(metrics, "nuisance_robustness_auc", "robustness_auc", "nuisance.robustness_auc")


def nuisance_curve_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in _ordered_candidates(summary):
        metrics = _candidate_metrics(summary, candidate)
        auc = _robustness_auc(metrics)
        for point in _curve_points(metrics):
            rows.append({"candidate": candidate, "strength": point["strength"], "evidence": point["evidence"], "n": point.get("n"), "robustness_auc": _estimate(auc), "robustness_status": _status(auc)})
    return rows


def robustness_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in _ordered_candidates(summary):
        value = _robustness_auc(_candidate_metrics(summary, candidate))
        rows.append({"candidate": candidate, "robustness_auc": _estimate(value), "status": _status(value)})
    return rows


def genuine_overlap_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    aliases = {
        "auroc": ("auroc", "AUROC"), "auprc": ("auprc", "AUPRC"),
        "fpr": ("fpr", "false_positive_rate"), "fnr": ("fnr", "false_negative_rate"),
        "brier": ("brier", "brier_score", "calibration_brier"),
        "ece": ("ece", "expected_calibration_error", "calibration_ece"),
    }
    rows: list[dict[str, Any]] = []
    for candidate in _ordered_candidates(summary):
        metrics = _candidate_metrics(summary, candidate)
        detection = _lookup(metrics, "genuine_overlap", "detection")
        if not isinstance(detection, Mapping):
            detection = metrics
        calibration = _lookup(detection, "calibration")
        row: dict[str, Any] = {"candidate": candidate}
        for field in _DETECTION_FIELDS:
            value = _lookup(detection, *aliases[field])
            if value is None and field in {"brier", "ece"} and isinstance(calibration, Mapping):
                value = _lookup(calibration, *aliases[field])
            row[field] = _estimate(value)
            row[f"{field}_status"] = _status(value)
        row["status"] = "defined" if any(row[field] is not None for field in _DETECTION_FIELDS) else "undefined"
        row["calibration_status"] = "defined" if row["brier"] is not None or row["ece"] is not None else "undefined"
        row["n"] = _lookup(detection, "n", "n_pairs", "n_rows")
        rows.append(row)
    return rows


def _stage_metric(metrics: Mapping[str, Any], stage: str) -> Any:
    timing = _lookup(metrics, "timing", "runtime", "timings")
    if isinstance(timing, Mapping) and _lookup(timing, stage) is not None:
        return _lookup(timing, stage)
    aliases = {
        "fit": ("fit", "fit_seconds", "fit_time", "fit_wall_seconds"),
        "score_fixed": ("score_fixed", "score_fixed_seconds", "score_fixed_time", "score_fixed_wall_seconds"),
        "total": ("total", "total_seconds", "runtime_seconds", "elapsed_seconds"),
        "fit_cpu": ("fit_cpu", "fit_cpu_seconds"),
        "score_fixed_cpu": ("score_fixed_cpu", "score_fixed_cpu_seconds"),
        "total_cpu": ("total_cpu", "total_cpu_seconds", "outer_cpu_seconds", "policy_cpu_seconds", "cpu_seconds"),
        "peak_memory": ("peak_memory", "peak_memory_per_cell_mb", "cell_peak_memory_mb", "individual_peak_memory_mb"),
    }
    return _lookup(metrics, *aliases[stage])


def runtime_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in _ordered_candidates(summary):
        metrics = _candidate_metrics(summary, candidate)
        for stage in _RUNTIME_STAGES:
            value = _stage_metric(metrics, stage)
            if value is None and stage == "peak_memory":
                timing = _lookup(metrics, "timing", "runtime", "timings")
                memory_status = _lookup(timing, "memory_status", "peak_memory_status") if isinstance(timing, Mapping) else None
                if memory_status is None:
                    memory_status = _lookup(metrics, "memory_status", "peak_memory_status")
                if memory_status is not None:
                    value = {"status": memory_status}
            rows.append({"candidate": candidate, "stage": stage, "median": _estimate(value), "p95": _number(_lookup(value, "p95", "q95")), "n": _lookup(value, "n") if isinstance(value, Mapping) else None, "status": _status(value), "unit": "MB" if stage == "peak_memory" else "seconds"})
    return rows


def _ratio(metrics: Mapping[str, Any], field: str) -> float | None:
    value = _lookup(metrics, field)
    timing = _lookup(metrics, "timing", "runtime", "timings")
    if value is None and isinstance(timing, Mapping):
        value = _lookup(timing, field)
    return _estimate(value)


def _walk_named(value: Any, names: set[str], path: tuple[str, ...] = (), depth: int = 0) -> list[tuple[tuple[str, ...], Any]]:
    """Find small named artifacts without traversing row lists indefinitely."""

    if depth > 5 or not isinstance(value, Mapping):
        return []
    found: list[tuple[tuple[str, ...], Any]] = []
    for key in sorted(value, key=lambda item: str(item)):
        child = value[key]
        text = str(key).lower().replace("-", "_")
        child_path = (*path, str(key))
        # Do not mistake a configured ``maximum_rows`` scalar for an actual
        # probe artifact.  Only named probe rows/runtime/summary containers
        # are data surfaces; the cap is handled separately below.
        if text in names or text.endswith(("_probe_rows", "_probe_runtime", "_probe_summary")):
            found.append((child_path, child))
        if isinstance(child, Mapping) and text not in {"candidates", "rows", "records"}:
            found.extend(_walk_named(child, names, child_path, depth + 1))
    return found


def _probe_stats(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        for key in ("rows", "records", "probe_rows", "components"):
            if isinstance(value.get(key), (list, tuple)):
                nested = _probe_stats(value[key])
                value = {**value, **{name: value.get(name, nested.get(name)) for name in ("status", "n", "median_seconds", "p95_seconds", "score", "max_rows", "triggered_panels")}}
                break
        status_value = value.get("status")
        wall = _number(_lookup(value, "wall_seconds", "runtime_seconds", "total_seconds", "policy_wall_seconds"))
        if wall is None:
            wall = _estimate(_lookup(value, "median_seconds", "runtime", "runtime.total", "timing.total", "timing.total_seconds"))
        p95 = _number(_lookup(value, "p95_seconds", "p95", "q95"))
        if p95 is None:
            p95 = _number(_lookup(value, "timing.total.p95", "timing.total.q95"))
        if p95 is None:
            p95 = _number(value.get("p95_seconds"))
        if p95 is None:
            p95 = _number(_lookup(value, "runtime.total.p95", "runtime.total.q95"))
        score = _estimate(_lookup(value, "score", "accuracy"))
        status = "fail" if str(status_value).lower() in {"error", "failed", "fail"} else str(status_value) if status_value is not None else "defined" if any(item is not None for item in (wall, score, _lookup(value, "n_rows", "n"))) else "undefined"
        return {"status": status, "n": _lookup(value, "n", "count", "n_rows"), "median_seconds": wall, "p95_seconds": p95, "score": score, "max_rows": _lookup(value, "maximum_rows", "max_rows", "n_rows"), "triggered_panels": _lookup(value, "triggered_panels", "n_triggered", "panel_triggered")}
    if isinstance(value, (list, tuple)):
        entries = [item for item in value if isinstance(item, Mapping)]
        walls = sorted(item for item in (_number(_lookup(item, "wall_seconds", "runtime_seconds", "total_seconds", "policy_wall_seconds")) for item in entries) if item is not None)
        scores = [item for item in (_number(_lookup(item, "score", "accuracy")) for item in entries) if item is not None]
        statuses = [str(item.get("status")).lower() for item in entries if item.get("status") is not None]
        triggered = [_lookup(item, "panel_triggered") for item in entries]
        status = "fail" if any(item in {"error", "failed", "fail"} for item in statuses) else "ok" if entries and statuses and all(item == "ok" for item in statuses) else "defined" if entries else "undefined"
        return {"status": status, "n": len(entries), "median_seconds": walls[len(walls) // 2] if walls else None, "p95_seconds": walls[min(len(walls) - 1, int(math.ceil(0.95 * len(walls)) - 1))] if walls else None, "score": sum(scores) / len(scores) if scores else None, "max_rows": max((_number(_lookup(item, "n_rows", "maximum_rows", "max_rows")) or 0 for item in entries), default=None), "triggered_panels": sum(bool(item) for item in triggered) if triggered else None}
    return {"status": "undefined", "n": None, "median_seconds": None, "p95_seconds": None, "score": None, "max_rows": None, "triggered_panels": None}


def probe_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = {"capped_probe_rows", "prior_full_probe_rows", "full_probe_rows", "capped_probe", "full_probe", "capped_probe_runtime", "full_probe_runtime", "guardrail_rows"}
    labels = {"capped_probe_rows": "capped probe", "capped_probe": "capped probe", "capped_probe_runtime": "capped probe", "prior_full_probe_rows": "full probe (prior)", "full_probe_rows": "full probe", "full_probe": "full probe", "full_probe_runtime": "full probe", "guardrail_rows": "G guardrail policy"}
    rows: list[dict[str, Any]] = []
    for path, value in _walk_named(summary, names):
        key = str(path[-1]).lower().replace("-", "_")
        rows.append({"probe": labels.get(key, key), "path": ".".join(path), **_probe_stats(value)})
    # analysis.py may normalize Food-101 probe streams as pseudo-candidates
    # (``full_probe``/``G_probe_component``) before building its summary.  They
    # are data artifacts, not candidate arms, so extract them here and keep
    # them out of the C–E/F/G decision table.
    candidate_map = _lookup(summary, "candidates")
    existing_labels = {row["probe"] for row in rows}
    if isinstance(candidate_map, Mapping):
        candidate_labels = {
            "g_probe_component": "capped probe",
            "capped_probe": "capped probe",
            "full_probe": "full probe",
            "prior_full_probe": "full probe (prior)",
            "g": "G guardrail policy",
        }
        for key in sorted(candidate_map, key=lambda item: str(item)):
            normalized = str(key).lower().replace("-", "_")
            label = candidate_labels.get(normalized)
            if label is None or label in existing_labels:
                continue
            value = candidate_map[key]
            if isinstance(value, Mapping):
                stats = _probe_stats(value)
                if label == "capped probe" and stats.get("max_rows") in {None, 0, 1}:
                    configuration = _lookup(summary, "manifest.configuration", "configuration")
                    cap = _lookup(configuration, "capped_probe_maximum_rows", "maximum_rows") if isinstance(configuration, Mapping) else None
                    if cap is not None:
                        stats["max_rows"] = cap
                if stats.get("status") != "undefined":
                    rows.append({"probe": label, "path": f"candidates.{key}", **stats})
                    existing_labels.add(label)
    if not rows:
        configuration = _lookup(summary, "manifest.configuration", "configuration")
        cap = _lookup(configuration, "capped_probe_maximum_rows", "maximum_rows") if isinstance(configuration, Mapping) else None
        if cap is not None:
            rows.append({"probe": "capped probe", "path": "configuration.capped_probe_maximum_rows", "status": "configured", "n": None, "median_seconds": None, "p95_seconds": None, "score": None, "max_rows": cap, "triggered_panels": None})
    return rows


def conditioning_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    aliases = {
        "refinement_rate": ("refinement_rate", "refinement.applied_rate", "applied_rate"),
        "refinement_applied": ("refinement_applied", "refinement.applied_count", "applied_count"),
        "refinement_eligible": ("refinement_eligible", "refinement.eligible_count", "eligible_count"),
        "condition_number": ("condition_number", "conditioning.condition_number", "conditioning.condition_number_after"),
        "conditioning_strength_spearman": ("conditioning_strength_spearman", "strength_spearman"),
        "condition_number_before": ("condition_number_before", "conditioning.condition_number_before"),
        "condition_number_after": ("condition_number_after", "conditioning.condition_number_after"),
        "cap_or_floor_active": ("cap_or_floor_active", "conditioning.cap_or_floor_active"),
        "mode": ("mode", "conditioning.mode"),
    }
    rows: list[dict[str, Any]] = []
    for candidate in _ordered_candidates(summary):
        metrics = _candidate_metrics(summary, candidate)
        containers = [_lookup(metrics, "conditioning_refinement", "diagnostics"), _lookup(metrics, "refinement"), _lookup(metrics, "conditioning"), metrics]
        containers = [item for item in containers if isinstance(item, Mapping)]
        row: dict[str, Any] = {"candidate": candidate}
        for name, paths in aliases.items():
            value = next((candidate_value for container in containers if (candidate_value := _lookup(container, *paths)) is not None), None)
            row[name] = value if name in {"mode", "cap_or_floor_active"} else _estimate(value)
        row["status"] = "defined" if any(row.get(key) is not None for key in aliases) else "undefined"
        rows.append(row)
    return rows


def _gate_map(summary: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _lookup(summary, key)
    return value if isinstance(value, Mapping) else {}


def _gate_status(gates: Mapping[str, Any], candidate: str) -> str | None:
    value = gates.get(candidate)
    return str(value["status"]) if isinstance(value, Mapping) and value.get("status") is not None else None


def _decision_status(candidate: str, historical: str | None, product: str | None) -> str:
    if candidate not in {"C", "D", "E"}:
        return "inconclusive"
    statuses = [value for value in (historical, product) if value in {"pass", "fail", "inconclusive"}]
    if "fail" in statuses:
        return "fail"
    if len(statuses) == 2 and all(value == "pass" for value in statuses):
        return "pass"
    return "inconclusive"


def _promotion_candidate(promotion: Mapping[str, Any] | None) -> str | None:
    """Return the immutable promotion lock, preferring the explicit field.

    Older screen artifacts called this field ``selected_candidate``.  Newer
    promotion artifacts may additionally expose ``locked_candidate``; that
    field is authoritative for the final Food-101 decision and must not be
    replaced by a runner-up selected anywhere else in the artifact.
    """

    if not isinstance(promotion, Mapping):
        return None
    for key in ("locked_candidate", "selected_candidate"):
        value = _lookup(promotion, key)
        if value is not None and value != "":
            return str(value)
    return None


def decision_rows(summary: Mapping[str, Any], promotion: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Flatten every candidate into one stable decision/metrics row."""

    historical, product = _gate_map(summary, "historical_gates"), _gate_map(summary, "product_gates")
    selected = _promotion_candidate(promotion)
    selector = {row["candidate"]: row for row in selector_rows(summary) if row["head"] == "linear"}
    detection = {row["candidate"]: row for row in genuine_overlap_rows(summary)}
    robust = {row["candidate"]: row for row in robustness_rows(summary)}
    diagnostic = {row["candidate"]: row for row in conditioning_rows(summary)}
    timing = runtime_rows(summary)
    out: list[dict[str, Any]] = []
    for candidate in _ordered_candidates(summary):
        metrics = _candidate_metrics(summary, candidate)
        linear, detect, diag = selector.get(candidate, {}), detection.get(candidate, {}), diagnostic.get(candidate, {})
        candidate_timing = {row["stage"]: row for row in timing if row["candidate"] == candidate}
        historical_status, product_status = _gate_status(historical, candidate), _gate_status(product, candidate)
        role = _role(candidate, metrics)
        clean_regret_source = _lookup(metrics, "clean_linear_regret_delta", "clean_linear_regret")
        nuisance_regret_source = _lookup(metrics, "nuisance_linear_regret_delta", "nuisance_linear_regret")
        out.append({
            "candidate": candidate, "name": metrics.get("name", metrics.get("candidate_name", candidate)), "role": role["role"], "scope": role["scope"], "definition": role["definition"], "n_rows": _lookup(metrics, "n_rows"),
            "score": _estimate(_lookup(metrics, "score")), "overlap_evidence": _estimate(_lookup(metrics, "overlap_evidence")), "pair_overlap_evidence": _estimate(_lookup(metrics, "pair_overlap_evidence")),
            "linear_regret": linear.get("regret"), "linear_regret_lower": linear.get("regret_lower"), "linear_regret_upper": linear.get("regret_upper"), "linear_rank_auc": linear.get("rank_auc"), "linear_exact_best": linear.get("exact_best"), "linear_within_one_point": linear.get("within_one_point"), "linear_within_1pp": linear.get("within_1pp"),
            "fpr": detect.get("fpr"), "fnr": detect.get("fnr"), "auroc": detect.get("auroc"), "auprc": detect.get("auprc"), "brier": detect.get("brier"), "ece": detect.get("ece"),
            "clean_mae": _estimate(_lookup(metrics, "clean_mae")), "clean_linear_regret_upper": _ci(clean_regret_source, "upper") if _ci(clean_regret_source, "upper") is not None else _estimate(_lookup(metrics, "clean_linear_regret_upper")), "nuisance_linear_regret_upper": _ci(nuisance_regret_source, "upper") if _ci(nuisance_regret_source, "upper") is not None else _estimate(_lookup(metrics, "nuisance_linear_regret_upper")), "robustness_auc": robust.get(candidate, {}).get("robustness_auc"), "nuisance_spearman": _estimate(_lookup(metrics, "nuisance_spearman")), "nuisance_ordering_rate": _estimate(_lookup(metrics, "nuisance_ordering_rate")),
            "refinement_rate": diag.get("refinement_rate"), "condition_number": diag.get("condition_number"), "median_total_seconds": candidate_timing.get("total", {}).get("median"), "p95_total_seconds": candidate_timing.get("total", {}).get("p95"), "median_score_fixed_seconds": candidate_timing.get("score_fixed", {}).get("median"),
            **{field: _ratio(metrics, field) for field in _RATIO_FIELDS},
            "historical_gate": historical_status, "product_gate": product_status, "decision_status": _decision_status(candidate, historical_status, product_status), "promoted": candidate == selected,
        })
    return out


_DECISION_FIELDS = ("candidate", "name", "role", "scope", "definition", "n_rows", "linear_regret", "linear_regret_lower", "linear_regret_upper", "linear_rank_auc", "linear_exact_best", "linear_within_one_point", "linear_within_1pp", "auroc", "auprc", "fpr", "fnr", "brier", "ece", "robustness_auc", "refinement_rate", "condition_number", "median_total_seconds", "p95_total_seconds", "median_score_fixed_seconds", *_RATIO_FIELDS, "historical_gate", "product_gate", "decision_status", "promoted")


def write_decision_table(destination: Path, summary: Mapping[str, Any], promotion: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = decision_rows(summary, promotion)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "decision_table.json").write_text(canonical_json(rows) + "\n", encoding="utf-8")
    with (destination / "decision_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_DECISION_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


# The small hand-written SVG renderer avoids matplotlib version metadata and
# remains deterministic in minimal environments.  A placeholder is emitted
# only when the corresponding summary surface has no finite values.
def _placeholder_svg(path: Path, message: str) -> None:
    path.write_text("<?xml version='1.0' encoding='UTF-8'?>\n<svg xmlns='http://www.w3.org/2000/svg' width='960' height='320' viewBox='0 0 960 320'><rect width='100%' height='100%' fill='white'/><text x='24' y='48' font-family='sans-serif' font-size='18' fill='#444'>" + html.escape(message) + "</text></svg>\n", encoding="utf-8")


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _svg_document(body: str, title: str, width: int = 960, height: int = 520) -> str:
    return "<?xml version='1.0' encoding='UTF-8'?>\n" + f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'><rect width='100%' height='100%' fill='white'/><title>{_text(title)}</title>{body}</svg>\n"


def _finite_series(series: Sequence[tuple[str, Sequence[Any], str]]) -> list[tuple[str, list[float | None], str]]:
    return [(name, [_number(value) for value in values], color) for name, values, color in series if any(_number(value) is not None for value in values)]


def _bar_panel(labels: Sequence[str], series: Sequence[tuple[str, Sequence[Any], str]], *, x: float, y: float, width: float, height: float, title: str, ylabel: str) -> tuple[str, bool]:
    selected = _finite_series(series)
    if not selected:
        return "", False
    values = [value for _, numbers, _ in selected for value in numbers if value is not None]
    low, high = min(0.0, min(values)), max(0.0, max(values))
    if math.isclose(low, high):
        low, high = low - 1.0, high + 1.0
    ix, iy, iw, ih = x + 58, y + 38, width - 78, height - 82
    def ycoord(value: float) -> float:
        return iy + ih * (high - value) / (high - low)
    baseline = ycoord(0.0)
    parts = [f"<text x='{x + 4:g}' y='{y + 18:g}' font-family='sans-serif' font-size='16' font-weight='bold'>{_text(title)}</text><text x='{x + 4:g}' y='{y + 34:g}' font-family='sans-serif' font-size='11' fill='#555'>{_text(ylabel)}</text><rect x='{ix:g}' y='{iy:g}' width='{iw:g}' height='{ih:g}' fill='none' stroke='#777'/><line x1='{ix:g}' y1='{baseline:g}' x2='{ix + iw:g}' y2='{baseline:g}' stroke='#555'/><text x='{x + 4:g}' y='{iy + 4:g}' font-family='sans-serif' font-size='10' fill='#555'>{_text(_fmt(high))}</text><text x='{x + 4:g}' y='{iy + ih:g}' font-family='sans-serif' font-size='10' fill='#555'>{_text(_fmt(low))}</text>"]
    n_labels, n_series = max(1, len(labels)), len(selected)
    group_w, bar_w = iw / n_labels, min(30.0, (iw / n_labels) * 0.78 / max(1, n_series))
    for index, label in enumerate(labels):
        center = ix + (index + 0.5) * group_w
        parts.append(f"<text x='{center:g}' y='{iy + ih + 18:g}' text-anchor='middle' font-family='sans-serif' font-size='11'>{_text(label)}</text>")
        for offset, (_name, numbers, color) in enumerate(selected):
            value = numbers[index] if index < len(numbers) else None
            if value is None:
                continue
            left = center + (offset - (n_series - 1) / 2) * bar_w - bar_w / 2
            top, bar_h = min(ycoord(value), baseline), max(1.0, abs(ycoord(value) - baseline))
            parts.append(f"<rect x='{left:g}' y='{top:g}' width='{bar_w:g}' height='{bar_h:g}' fill='{_text(color)}'><title>{_text(label)} {_text(_fmt(value))}</title></rect>")
    legend_y = y + height - 10
    for index, (name, _numbers, color) in enumerate(selected):
        lx = ix + index * max(110.0, width / max(1, len(selected)))
        parts.append(f"<rect x='{lx:g}' y='{legend_y - 10:g}' width='10' height='10' fill='{_text(color)}'/><text x='{lx + 14:g}' y='{legend_y:g}' font-family='sans-serif' font-size='11'>{_text(name)}</text>")
    return "".join(parts), True


def _line_panel(curves: Mapping[str, Sequence[tuple[float, float]]], aucs: Mapping[str, float | None], *, x: float, y: float, width: float, height: float, title: str) -> tuple[str, bool]:
    finite_curves = {name: sorted((float(px), float(py)) for px, py in points if _number(px) is not None and _number(py) is not None) for name, points in curves.items()}
    finite_curves = {name: points for name, points in finite_curves.items() if points}
    if not finite_curves:
        return "", False
    all_points = [point for points in finite_curves.values() for point in points]
    x_values, y_values = [point[0] for point in all_points], [point[1] for point in all_points]
    x_low, x_high, y_low, y_high = min(x_values), max(x_values), min(y_values), max(y_values)
    if math.isclose(x_low, x_high): x_low, x_high = x_low - 1.0, x_high + 1.0
    if math.isclose(y_low, y_high): y_low, y_high = y_low - 1.0, y_high + 1.0
    ix, iy, iw, ih = x + 64, y + 42, width - 98, height - 88
    def point(px: float, py: float) -> tuple[float, float]:
        return ix + iw * (px - x_low) / (x_high - x_low), iy + ih * (y_high - py) / (y_high - y_low)
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2", "#ff9da6")
    parts = [f"<text x='{x + 4:g}' y='{y + 20:g}' font-family='sans-serif' font-size='16' font-weight='bold'>{_text(title)}</text><rect x='{ix:g}' y='{iy:g}' width='{iw:g}' height='{ih:g}' fill='none' stroke='#777'/><text x='{ix:g}' y='{iy + ih + 20:g}' font-family='sans-serif' font-size='11'>nuisance strength</text><text x='{x + 4:g}' y='{iy + 4:g}' font-family='sans-serif' font-size='10' fill='#555'>{_text(_fmt(y_high))}</text><text x='{x + 4:g}' y='{iy + ih:g}' font-family='sans-serif' font-size='10' fill='#555'>{_text(_fmt(y_low))}</text>"]
    for index, (name, points) in enumerate(sorted(finite_curves.items())):
        color = colors[index % len(colors)]
        coords = [point(px, py) for px, py in points]
        d = " ".join(("M" if offset == 0 else "L") + f" {cx:g},{cy:g}" for offset, (cx, cy) in enumerate(coords))
        parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>")
        parts.extend(f"<circle cx='{cx:g}' cy='{cy:g}' r='3' fill='{color}'/>" for cx, cy in coords)
        label = name if aucs.get(name) is None else f"{name} (robustness AUC={_fmt(aucs[name])})"
        lx, ly = ix + iw + 8, iy + index * 20 + 12
        parts.append(f"<line x1='{lx:g}' y1='{ly - 4:g}' x2='{lx + 16:g}' y2='{ly - 4:g}' stroke='{color}' stroke-width='2'/><text x='{lx + 20:g}' y='{ly:g}' font-family='sans-serif' font-size='10'>{_text(label)}</text>")
    return "".join(parts), True


def _write_selector_plot(path: Path, summary: Mapping[str, Any]) -> None:
    rows, labels = selector_rows(summary), _ordered_candidates(summary)
    linear = {row["candidate"]: row for row in rows if row["head"] == "linear"}
    body1, okay1 = _bar_panel(labels, [("linear regret", [linear.get(label, {}).get("regret") for label in labels], "#e45756")], x=0, y=0, width=480, height=330, title="Selector regret", ylabel="accuracy points; lower is better")
    body2, okay2 = _bar_panel(labels, [("linear rank AUC", [linear.get(label, {}).get("rank_auc") for label in labels], "#4c78a8")], x=480, y=0, width=480, height=330, title="Selector rank AUC", ylabel="rank association; higher is better")
    if not okay1 and not okay2:
        _placeholder_svg(path, "selector regret and rank AUC unavailable")
        return
    path.write_text(_svg_document(body1 + body2 + "<text x='24' y='370' font-family='sans-serif' font-size='11' fill='#555'>Exact-best and within-1pp are tabulated; undefined cells remain unavailable.</text>", "Selector regret and rank AUC", height=410), encoding="utf-8")


def _write_nuisance_plot(path: Path, summary: Mapping[str, Any]) -> None:
    rows, curves, aucs = nuisance_curve_rows(summary), {}, {}
    for row in rows:
        curves.setdefault(str(row["candidate"]), []).append((float(row["strength"]), float(row["evidence"])))
        aucs[str(row["candidate"])] = row.get("robustness_auc")
    body, okay = _line_panel(curves, aucs, x=0, y=0, width=960, height=360, title="Nuisance summary curves")
    if not okay:
        robust = robustness_rows(summary)
        body, okay = _bar_panel([row["candidate"] for row in robust], [("robustness AUC", [row["robustness_auc"] for row in robust], "#f58518")], x=0, y=0, width=960, height=360, title="Nuisance robustness AUC", ylabel="candidate-minus-B designated-pair evidence; lower is better")
    if not okay:
        _placeholder_svg(path, "nuisance curves and robustness AUC unavailable")
        return
    note = "<text x='24' y='390' font-family='sans-serif' font-size='11' fill='#555'>Curves use the stored nuisance summary surface; robustness AUC uses its stored designated-pair estimand.</text>"
    path.write_text(_svg_document(body + note, "Nuisance curves and robustness AUC", height=425), encoding="utf-8")


def _write_detection_plot(path: Path, summary: Mapping[str, Any]) -> None:
    rows = genuine_overlap_rows(summary)
    series = [(name.upper() if name in {"auroc", "auprc"} else name.upper(), [row[name] for row in rows], color) for name, color in (("auroc", "#54a24b"), ("auprc", "#4c78a8"), ("fpr", "#e45756"), ("fnr", "#f58518"), ("brier", "#b279a2"), ("ece", "#72b7b2"))]
    body, okay = _bar_panel([row["candidate"] for row in rows], series, x=0, y=0, width=960, height=390, title="Genuine-overlap detection and calibration", ylabel="stored metric")
    if not okay:
        _placeholder_svg(path, "genuine-overlap metrics unavailable")
        return
    note = "<text x='24' y='420' font-family='sans-serif' font-size='11' fill='#555'>AUROC/AUPRC are ranking metrics; FPR/FNR/Brier/ECE are shown without proxy substitution.</text>"
    path.write_text(_svg_document(body + note, "Genuine-overlap detection and calibration", height=450), encoding="utf-8")


def _write_runtime_plot(path: Path, summary: Mapping[str, Any]) -> None:
    rows, labels = runtime_rows(summary), _ordered_candidates(summary)
    by_stage = {(row["candidate"], row["stage"]): row for row in rows}
    series = [(stage, [by_stage.get((candidate, stage), {}).get("median") for candidate in labels], color) for stage, color in (("fit", "#4c78a8"), ("score_fixed", "#54a24b"), ("total", "#f58518"))]
    body, okay = _bar_panel(labels, series, x=0, y=0, width=960, height=340, title="Runtime stages", ylabel="median seconds")
    if not okay:
        ratio_series = [(field.replace("_ratio_vs_B", ""), [_ratio(_candidate_metrics(summary, candidate), field) for candidate in labels], color) for field, color in zip(_RATIO_FIELDS, ("#4c78a8", "#54a24b", "#f58518", "#b279a2", "#e45756", "#72b7b2"))]
        body, okay = _bar_panel(labels, ratio_series, x=0, y=0, width=960, height=340, title="Runtime and memory ratios vs B", ylabel="ratio")
    if not okay:
        memory = [by_stage.get((candidate, "peak_memory"), {}).get("median") for candidate in labels]
        body, okay = _bar_panel(labels, [("peak memory", memory, "#9d755d")], x=0, y=0, width=960, height=340, title="Peak memory per cell", ylabel="MB")
    if not okay:
        probes = [row for row in probe_rows(summary) if row.get("median_seconds") is not None]
        body, okay = _bar_panel([str(row["probe"]) for row in probes], [("probe runtime", [row["median_seconds"] for row in probes], "#b279a2")], x=0, y=0, width=960, height=340, title="Full/capped probe runtime", ylabel="seconds")
    if not okay:
        _placeholder_svg(path, "runtime stages, ratios, and memory unavailable")
        return
    notes = []
    for candidate in labels:
        values = [f"{field}={_fmt(_ratio(_candidate_metrics(summary, candidate), field))}" for field in _RATIO_FIELDS if _ratio(_candidate_metrics(summary, candidate), field) is not None]
        if values: notes.append(f"{candidate}: " + ", ".join(values))
        memory_status = by_stage.get((candidate, "peak_memory"), {}).get("status")
        if memory_status not in {None, "undefined"}:
            notes.append(f"{candidate}: peak memory status={memory_status}")
    for row in probe_rows(summary):
        if row.get("status") not in {"undefined", None}: notes.append(f"{row['probe']}: status={row['status']}, n={_fmt(row.get('n'))}, seconds={_fmt(row.get('median_seconds'))}, max_rows={_fmt(row.get('max_rows'))}")
    note_lines = notes or ["No runtime ratio or probe rows were supplied."]
    note = "".join(f"<text x='24' y='{372 + index * 14:g}' font-family='sans-serif' font-size='10' fill='#555'>{_text(item)}</text>" for index, item in enumerate(note_lines))
    path.write_text(_svg_document(body + note, "Runtime stages and ratios", height=430), encoding="utf-8")


def _write_diagnostic_plot(path: Path, summary: Mapping[str, Any]) -> None:
    rows = conditioning_rows(summary)
    body1, okay1 = _bar_panel([row["candidate"] for row in rows], [("refinement rate", [row["refinement_rate"] for row in rows], "#ff9da6")], x=0, y=0, width=480, height=330, title="Prototype refinement", ylabel="applied / eligible")
    body2, okay2 = _bar_panel([row["candidate"] for row in rows], [("condition number", [row["condition_number"] for row in rows], "#9d755d")], x=480, y=0, width=480, height=330, title="Conditioning", ylabel="stored condition number")
    if not okay1 and not okay2:
        _placeholder_svg(path, "conditioning and refinement diagnostics unavailable")
        return
    spearman = ", ".join(f"{row['candidate']} conditioning-strength Spearman={_fmt(row.get('conditioning_strength_spearman'))}" for row in rows if row.get("conditioning_strength_spearman") is not None)
    note = f"<text x='24' y='370' font-family='sans-serif' font-size='11' fill='#555'>{_text(spearman or 'Cap/floor and conditioning-strength diagnostics are tabulated when supplied.')}</text>"
    path.write_text(_svg_document(body1 + body2 + note, "Conditioning and refinement diagnostics", height=410), encoding="utf-8")


def write_plots(destination: Path, summary: Mapping[str, Any]) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    paths = ("score_regret.svg", "nuisance_curves.svg", "genuine_overlap.svg", "runtime.svg", "conditioning_refinement.svg")
    _write_selector_plot(destination / paths[0], summary)
    _write_nuisance_plot(destination / paths[1], summary)
    _write_detection_plot(destination / paths[2], summary)
    _write_runtime_plot(destination / paths[3], summary)
    _write_diagnostic_plot(destination / paths[4], summary)
    return list(paths)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return lines


def _source_rows(summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    sources: dict[str, str] = {}
    containers = (("", summary), ("food101:", _lookup(summary, "food101")))
    for prefix, container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in ("source_hashes", "input_hashes"):
            value = _lookup(container, key)
            if isinstance(value, Mapping):
                for path, digest in value.items():
                    if digest is not None: sources.setdefault(prefix + str(path), str(digest))
        provenance = _lookup(container, "provenance")
        provenance_hashes = _lookup(provenance, "source_hashes") if isinstance(provenance, Mapping) else None
        if isinstance(provenance_hashes, Mapping):
            for path, digest in provenance_hashes.items():
                if digest is not None: sources.setdefault(prefix + str(path), str(digest))
        manifest = _lookup(container, "manifest")
        raw_sources = _lookup(manifest, "sources") if isinstance(manifest, Mapping) else None
        if isinstance(raw_sources, Mapping):
            for source in raw_sources.values():
                if isinstance(source, Mapping) and source.get("path") is not None:
                    sources.setdefault(prefix + str(source["path"]), str(source.get("sha256", source.get("hash", "—"))))
        elif isinstance(raw_sources, (list, tuple)):
            for source in raw_sources:
                if isinstance(source, Mapping) and source.get("path") is not None:
                    sources.setdefault(prefix + str(source["path"]), str(source.get("sha256", source.get("hash", "—"))))
        evidence = _lookup(container, "source_evidence") or _lookup(container, "protocol.source_evidence")
        if isinstance(evidence, (list, tuple)):
            for source in evidence:
                if isinstance(source, Mapping) and source.get("path") is not None:
                    sources.setdefault(prefix + str(source["path"]), str(source.get("sha256", source.get("hash", "—"))))
    return sorted(sources.items(), key=lambda item: item[0])


def _find_scalar(summary: Mapping[str, Any], names: set[str]) -> Any:
    found = [
        item for item in _walk_named(summary, names)
        if str(item[0][-1]).lower().replace("-", "_") in names
    ]
    return found[0][1] if found else None


def _retrospective_lines(summary: Mapping[str, Any]) -> list[str]:
    retrospective = _find_scalar(summary, {"retrospective", "retrospective_evidence"})
    confirmation = _find_scalar(summary, {"confirmation_status", "untouched_confirmation", "untouched_confirmed", "untouched_embedding_panel"})
    retro_text = ("yes" if retrospective else "no") if isinstance(retrospective, bool) else str(retrospective or "not recorded")
    if isinstance(confirmation, bool):
        confirmation_text = "recorded" if confirmation else "No untouched confirmation is recorded"
    else:
        confirmation_text = str(confirmation or "No untouched confirmation is recorded; this report makes no claim")
    return [f"- Retrospective evidence: **{_cell(retro_text)}**.", f"- Untouched confirmation: **{_cell(confirmation_text)}**."]


def _deviations(summary: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for container in (summary, _lookup(summary, "manifest"), _lookup(summary, "food101")):
        if not isinstance(container, Mapping): continue
        for deviation_key in ("deviations", "recovered_deviations"):
            value = container.get(deviation_key)
            if isinstance(value, (list, tuple)): values.extend(str(item) for item in value)
            elif value: values.append(str(value))
    return list(dict.fromkeys(values))


def _gate_rows(summary: Mapping[str, Any]) -> list[tuple[str, str, str, str, str]]:
    """Flatten gate names/details without making a missing gate a pass."""

    rows: list[tuple[str, str, str, str, str]] = []
    for family, key in (("historical", "historical_gates"), ("product", "product_gates")):
        gates = _gate_map(summary, key)
        for candidate in sorted(gates, key=lambda value: (str(value))):
            value = gates.get(candidate)
            if not isinstance(value, Mapping):
                continue
            gate_list = value.get("gates", ()) if isinstance(value.get("gates"), (list, tuple)) else ()
            emitted = False
            for gate in gate_list:
                if not isinstance(gate, Mapping):
                    continue
                emitted = True
                rows.append((family, str(candidate), str(gate.get("name", "unnamed")), str(gate.get("status", "inconclusive")), str(gate.get("detail", gate.get("reason", "")))))
            if not emitted and (value.get("status") is not None or value.get("reason") is not None):
                rows.append((family, str(candidate), "overall", str(value.get("status", "inconclusive")), str(value.get("reason", ""))))
    return rows


def _food101_lines(summary: Mapping[str, Any]) -> list[str]:
    """Render Food-101 as a separate retrospective evidence section."""

    food = _lookup(summary, "food101")
    if not isinstance(food, Mapping):
        return []
    rows = selector_rows(food)
    stats = _lookup(food, "statistics")
    regret_seed = _lookup(stats, "food101_regret_bootstrap_seed", "regret_bootstrap_seed") if isinstance(stats, Mapping) else None
    rank_seed = _lookup(stats, "food101_rank_auc_bootstrap_seed", "rank_auc_bootstrap_seed") if isinstance(stats, Mapping) else None
    if regret_seed is None:
        regret_seed = _lookup(summary, "food101_regret_bootstrap_seed")
    if rank_seed is None:
        rank_seed = _lookup(summary, "food101_rank_auc_bootstrap_seed")
    # These are frozen protocol seeds used by analysis.py's Food-101 helper;
    # retain the defaults when the compact summary omits the duplicated fields.
    regret_seed = regret_seed if regret_seed is not None else 145
    rank_seed = rank_seed if rank_seed is not None else 143
    lines = [
        "## Food-101 retrospective panel (separate evidence)",
        "",
        "Synthetic screen ranking locks one C–E candidate. Food-101 selector results are reported separately; mandatory Food-101 product gates decide the final locked candidate and never trigger reranking. Frozen paired seeds: regret **" + _cell(regret_seed) + "**; rank AUC **" + _cell(rank_seed) + "**.",
        "",
    ]
    lines.extend(_markdown_table(("Candidate", "Head", "Regret", "Rank AUC", "Exact-best", "Within-1pp", "Status"), ((row["candidate"], row["head"], row["regret"], row["rank_auc"], row["exact_best"], row["within_1pp"], row["status"]) for row in rows)))
    if not rows:
        lines.append("No Food-101 selector rows were supplied; cells remain unavailable.")
    return lines


def _promotion_lines(promotion: Mapping[str, Any]) -> list[str]:
    """Describe screen locking and final Food-101 gate outcomes precisely."""

    candidate = _promotion_candidate(promotion) or "none"
    locked_value = _lookup(promotion, "locked_candidate")
    has_explicit_lock = locked_value is not None and locked_value != ""
    raw_status = _lookup(promotion, "status")
    display_status = str(raw_status or "inconclusive")
    normalized = display_status.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"fail", "rejected", "reject", "failed", "failure", "not_promoted", "not_promotable", "final_fail", "final_rejected"}:
        final_status = "fail"
    elif normalized in {"pass", "passed", "approved", "final_pass", "promoted"}:
        final_status = "pass"
    elif normalized in {"promoted_for_full_evaluation", "screen_locked", "screen_pass", "locked"}:
        final_status = "screen_locked"
    else:
        final_status = "inconclusive"
    reason = _lookup(promotion, "reason", "failure_reason", "decision_reason")
    reason_text = f" Reason: {_cell(reason)}." if reason not in (None, "") else ""
    if has_explicit_lock and final_status == "fail":
        line = f"Final locked-candidate decision: **rejected** (status **{_cell(display_status)}**) for locked candidate **{_cell(candidate)}**; mandatory Food-101 product gates failed. No reranking was performed."
    elif has_explicit_lock and final_status == "pass":
        line = f"Final locked-candidate decision: **pass** for locked candidate **{_cell(candidate)}** after mandatory Food-101 product gates. No reranking was performed."
    elif has_explicit_lock and final_status == "screen_locked":
        line = f"Synthetic screen locked candidate **{_cell(candidate)}** for Food-101 evaluation (status **{_cell(display_status)}**); mandatory Food-101 product gates remain pending for the final locked-candidate decision. They never trigger reranking."
    elif has_explicit_lock:
        line = f"Final locked-candidate decision: **inconclusive** (status **{_cell(display_status)}**) for locked candidate **{_cell(candidate)}**; mandatory Food-101 product gates are inconclusive. No reranking was performed."
    elif final_status == "screen_locked":
        line = f"Synthetic screen locked candidate **{_cell(candidate)}** for Food-101 evaluation (status **{_cell(display_status)}**); mandatory Food-101 product gates remain pending for the final locked-candidate decision. They never trigger reranking."
    else:
        line = f"Promotion artifact: **{_cell(candidate)}**; status **{_cell(display_status)}**; {_cell(reason or '')}."
    lines = [line + reason_text]
    deferred = _lookup(promotion, "deferred_gates")
    if deferred:
        deferred_values = deferred if isinstance(deferred, (list, tuple)) else [deferred]
        lines.append("Deferred gates: " + ", ".join(_cell(value) for value in deferred_values) + ".")
    return lines


def render_report(summary: Mapping[str, Any], promotion: Mapping[str, Any] | None = None) -> str:
    """Return a concise deterministic Markdown report with explicit gaps."""

    rows, selector = decision_rows(summary, promotion), selector_rows(summary)
    nuisance, robust = nuisance_curve_rows(summary), robustness_rows(summary)
    detection, runtime, diagnostics, probes = genuine_overlap_rows(summary), runtime_rows(summary), conditioning_rows(summary), probe_rows(summary)
    stage = _lookup(summary, "stage") or _lookup(summary, "manifest.stage") or "not recorded"
    complete = _lookup(summary, "artifact_completeness")
    complete_status = _lookup(complete, "status") if isinstance(complete, Mapping) else None
    stats, stats_text = _lookup(summary, "statistics"), []
    if isinstance(stats, Mapping):
        if stats.get("bootstrap_resamples") is not None: stats_text.append(f"{stats['bootstrap_resamples']} bootstrap resamples")
        if stats.get("bootstrap_seed") is not None: stats_text.append(f"seed {stats['bootstrap_seed']}")
        if stats.get("interval"): stats_text.append(str(stats["interval"]))
    lines = ["# Nuisance-conditioned distance analysis", "", f"Stage: **{_cell(stage)}**; Rows analyzed: **{_cell(_lookup(summary, 'n_rows'))}**; artifact completeness: **{_cell(complete_status or 'inconclusive')}**.", ("Statistics: " + "; ".join(stats_text) + ".") if stats_text else "Statistics: not recorded in the summary.", "", "## Candidate roles and scope", "", "C–E (C-E) are the algorithmic OI candidates. F is an explicit B–E disagreement diagnostic, not a selector. G is the panel-selective capped-probe product guardrail, not an algorithmic OI candidate and never promotable; its frozen trigger is B refinement rate ≥ 0.50 or |B−E score| ≥ 0.05.", ""]
    if isinstance(complete, Mapping) and complete.get("failures"):
        lines.extend(["Artifact completeness details:"] + [f"- {item}" for item in complete["failures"]] + [""])
    lines.extend(_markdown_table(("Candidate", "Name", "Role", "Scope", "Definition"), ((row["candidate"], row["name"], row["role"], row["scope"], row["definition"]) for row in rows)))
    lines.extend(["", "## Selector regret and rank AUC", "", "Regret is the stored selector outcome (lower is better); rank AUC, exact-best, and within-1pp are shown independently. A dash is undefined, not zero.", ""])
    lines.extend(_markdown_table(("Candidate", "Head", "Regret", "Rank AUC", "Exact-best", "Within-1pp", "Status"), ((row["candidate"], row["head"], row["regret"], row["rank_auc"], row["exact_best"], row["within_1pp"], row["status"]) for row in selector)))
    lines.extend(["", "## Nuisance curves and robustness AUC", ""])
    lines.extend(_markdown_table(("Candidate", "Strength", "Summary evidence", "n", "Robustness AUC", "Status"), ((row["candidate"], row["strength"], row["evidence"], row["n"], row["robustness_auc"], row["robustness_status"]) for row in nuisance)))
    if not nuisance: lines.append("No nuisance curve points were supplied; robustness AUC cells remain explicit below.")
    lines.extend(["", *(_markdown_table(("Candidate", "Candidate-minus-B designated-pair robustness AUC", "Status"), ((row["candidate"], row["robustness_auc"], row["status"]) for row in robust))), "", "## Genuine-overlap detection and calibration", ""])
    lines.extend(_markdown_table(("Candidate", "AUROC", "AUPRC", "FPR", "FNR", "Brier", "ECE", "Status"), ((row["candidate"], row["auroc"], row["auprc"], row["fpr"], row["fnr"], row["brier"], row["ece"], row["status"]) for row in detection)))
    lines.extend(["", "## Runtime stages, ratios, memory, and probes", ""])
    lines.extend(_markdown_table(("Candidate", "Stage", "Median", "P95", "Unit", "Status"), ((row["candidate"], row["stage"], row["median"], row["p95"], row["unit"], row["status"]) for row in runtime)))
    lines.extend(["", "Runtime ratios are paired against B when present; memory ratios are not inferred from process-wide RSS.", ""])
    ratio_rows = [(row["candidate"], *[_ratio(_candidate_metrics(summary, row["candidate"]), field) for field in _RATIO_FIELDS]) for row in rows]
    lines.extend(_markdown_table(("Candidate", "Median total/B", "P95 total/B", "Median score-fixed/B", "Median memory/B", "P95 memory/B", "Max memory/B"), ratio_rows))
    lines.extend(["", "Probe and guardrail artifacts (when present):", ""])
    lines.extend(_markdown_table(("Probe", "Source path", "Status", "n", "Median seconds", "P95 seconds", "Score", "Max rows", "Triggered panels"), ((row["probe"], row["path"], row["status"], row["n"], row["median_seconds"], row["p95_seconds"], row["score"], row["max_rows"], row["triggered_panels"]) for row in probes)))
    if not probes: lines.append("| — | — | unavailable | — | — | — | — | — | — |")
    lines.extend(["", "## Conditioning and refinement diagnostics", ""])
    lines.extend(_markdown_table(("Candidate", "Mode", "Refinement rate", "Applied", "Eligible", "Condition number", "Strength Spearman", "Cap/floor active", "Status"), ((row["candidate"], row["mode"], row["refinement_rate"], row["refinement_applied"], row["refinement_eligible"], row["condition_number"], row["conditioning_strength_spearman"], row["cap_or_floor_active"], row["status"]) for row in diagnostics)))
    lines.extend(["", "## Candidate decision table", "", "Pass/fail/inconclusive gate statuses are retained; controls, F, and G are never silently treated as promotable. F and G remain non-promotable by protocol.", ""])
    lines.extend(_markdown_table(("Candidate", "Historical", "Product", "Decision", "Promoted", "Role"), ((row["candidate"], row["historical_gate"] or "inconclusive", row["product_gate"] or "inconclusive", row["decision_status"], row["promoted"], row["role"]) for row in rows)))
    gate_rows = _gate_rows(summary)
    if gate_rows:
        lines.extend(["", "Gate details / reasons:", ""])
        lines.extend(_markdown_table(("Family", "Candidate", "Gate", "Status", "Detail"), gate_rows))
    if promotion:
        lines.extend(["", *_promotion_lines(promotion)])
    lines.extend(["", *_food101_lines(summary), "", "## Retrospective status and untouched confirmation", "", *_retrospective_lines(summary), "", "## Provenance, source hashes, and deviations", ""])
    provenance = _lookup(summary, "provenance")
    if isinstance(provenance, Mapping):
        lines.extend([f"Commit: `{_cell(provenance.get('commit') or '—')}`; dirty: `{_cell(bool(provenance.get('dirty')))}`; code identity SHA-256: `{_cell(provenance.get('code_identity_sha256') or '—')}`.", ""])
    lines.extend(_markdown_table(("Source path", "SHA-256"), _source_rows(summary) or (("—", "not recorded"),)))
    deviations = _deviations(summary)
    lines.extend(["", "Deviations / unavailable evidence:", *[f"- {item}" for item in deviations or ["None recorded in the summary."]], "", "Undefined metrics are shown as `—` and remain inconclusive; no threshold is tuned from outcomes.", "", "Plots: `score_regret.svg`, `nuisance_curves.svg`, `genuine_overlap.svg`, `runtime.svg`, and `conditioning_refinement.svg`."])
    return "\n".join(lines) + "\n"


def write_reports(destination: Path, summary: Mapping[str, Any], promotion: Mapping[str, Any] | None = None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    write_decision_table(destination, summary, promotion)
    (destination / "report.md").write_text(render_report(summary, promotion), encoding="utf-8")
    write_plots(destination, summary)


__all__ = ["conditioning_rows", "decision_rows", "genuine_overlap_rows", "nuisance_curve_rows", "probe_rows", "render_report", "robustness_rows", "runtime_rows", "selector_rows", "write_decision_table", "write_plots", "write_reports"]
