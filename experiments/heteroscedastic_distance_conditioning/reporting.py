"""Deterministic presentation for heteroscedastic statistics artifacts.

Reporting is deliberately downstream of :mod:`statistics`: it does not infer
metrics from raw scores, run a selector, or choose a candidate.  Missing and
inconclusive surfaces stay visible.  The wording distinguishes ordered
mechanistic concordance from causal mediation and never implies Food-101
evidence (that study is not authorized by the frozen protocol).
"""

from __future__ import annotations

import csv
import html
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .statistics import (
    CANDIDATE_ORDER,
    PROMOTABLE_CANDIDATES,
    REFERENCE_HEADS,
    canonical_json,
    json_safe,
)


ROLE_INFO: Dict[str, Dict[str, str]] = {
    "A": {"role": "exact OI control", "scope": "control", "definition": "unrefined raw OI"},
    "B": {"role": "exact refined OI control", "scope": "control", "definition": "balanced-median refined raw OI"},
    "L": {"role": "legacy reference", "scope": "non-promotable", "definition": "legacy full whitening pooled diagonal OAS"},
    "P25": {"role": "algorithmic OI", "scope": "promotable", "definition": "partial whitening gamma 0.25, sample-weighted"},
    "P50-SW": {"role": "algorithmic OI", "scope": "promotable", "definition": "partial whitening gamma 0.50, sample-weighted"},
    "P50-CB": {"role": "algorithmic OI", "scope": "promotable", "definition": "partial whitening gamma 0.50, class-balanced"},
    "W50-SW": {"role": "algorithmic OI", "scope": "promotable", "definition": "winsorized partial whitening gamma 0.50, sample-weighted"},
    "W50-CB": {"role": "algorithmic OI", "scope": "promotable", "definition": "winsorized partial whitening gamma 0.50, class-balanced"},
    "M50-SW": {"role": "algorithmic OI", "scope": "promotable", "definition": "MAD partial whitening gamma 0.50, sample-weighted"},
    "M50-CB": {"role": "algorithmic OI", "scope": "promotable", "definition": "MAD partial whitening gamma 0.50, class-balanced"},
}


def _lookup(value: Any, *paths: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    for path in paths:
        current: Any = value
        ok = True
        for part in str(path).split("."):
            if not isinstance(current, Mapping):
                ok = False
                break
            if part in current:
                current = current[part]
            else:
                key = next((key for key in current if str(key).lower() == part.lower()), None)
                if key is None:
                    ok = False
                    break
                current = current[key]
        if ok:
            return current
    return None


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _atomic_write_bytes(path: Path | str, payload: bytes) -> None:
    """Replace one presentation file atomically after it is fully rendered."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path | str, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _estimate(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        value = _lookup(value, "estimate", "value", "mean", "median")
    return _number(value)


def _status(value: Any) -> str:
    if isinstance(value, Mapping) and value.get("status") is not None:
        return str(value["status"])
    return "defined" if _estimate(value) is not None else "undefined"


def _candidate_names(summary: Mapping[str, Any]) -> List[str]:
    candidates = summary.get("candidates", {})
    if not isinstance(candidates, Mapping):
        candidates = {}
    names = [str(name) for name in candidates]
    ordered = [name for name in CANDIDATE_ORDER if name in names]
    ordered.extend(sorted(name for name in names if name not in ordered))
    return ordered


def _candidate(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = _lookup(summary, "candidates")
    if isinstance(value, Mapping) and isinstance(value.get(name), Mapping):
        return value[name]
    return {}


def _rows_for_mapping(mapping: Any, *, candidate_field: str = "candidate") -> List[Dict[str, Any]]:
    if not isinstance(mapping, Mapping):
        return []
    rows = []
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            row = dict(value)
        else:
            row = {"value": value}
        row.setdefault(candidate_field, str(key))
        rows.append(row)
    return rows


def selector_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    selection = _lookup(summary, "selection", "selector", "selection_metrics")
    result: List[Dict[str, Any]] = []
    if isinstance(selection, Mapping) and isinstance(selection.get("candidates"), Mapping):
        for candidate in _candidate_names({"candidates": selection["candidates"]}):
            value = selection["candidates"].get(candidate, {})
            heads = value.get("heads", {}) if isinstance(value, Mapping) else {}
            for head in REFERENCE_HEADS:
                data = heads.get(head, {}) if isinstance(heads, Mapping) else {}
                pooled = data.get("pooled", {}) if isinstance(data, Mapping) else {}
                result.append({
                    "candidate": candidate, "head": head,
                    "regret": _estimate(pooled), "regret_lower": _lookup(pooled, "lower"), "regret_upper": _lookup(pooled, "upper"),
                    "rank_spearman": _estimate(data.get("pooled_rank_spearman")) if isinstance(data, Mapping) else None,
                    "rank_auc": _estimate(data.get("rank_auc")) if isinstance(data, Mapping) else None,
                    "exact_best": _estimate(data.get("pooled_exact_best")) if isinstance(data, Mapping) else None,
                    "within_one_point": _estimate(data.get("pooled_within_one_point")) if isinstance(data, Mapping) else None,
                    "status": _status(pooled),
                })
        return result
    # Support a compact summary where candidate/head metrics live directly
    # under ``candidates``.
    for candidate in _candidate_names(summary):
        metrics = _candidate(summary, candidate)
        heads = _lookup(metrics, "heads", "selection_metrics.heads", "selector.heads")
        if not isinstance(heads, Mapping):
            heads = {}
        for head in REFERENCE_HEADS:
            data = heads.get(head, {}) if isinstance(heads, Mapping) else {}
            regret = _lookup(data, "regret", "pooled_regret")
            result.append({"candidate": candidate, "head": head, "regret": _estimate(regret), "regret_lower": _lookup(regret, "lower"), "regret_upper": _lookup(regret, "upper"), "rank_spearman": _estimate(_lookup(data, "pooled_rank_spearman")), "rank_auc": _estimate(_lookup(data, "rank_auc")), "exact_best": _estimate(_lookup(data, "pooled_exact_best")), "within_one_point": _estimate(_lookup(data, "pooled_within_one_point")), "status": _status(regret)})
    return result


def geometry_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    claims = _lookup(summary, "primary_claims")
    rows: List[Dict[str, Any]] = []
    if isinstance(claims, Mapping):
        for name in ("q1", "q2", "q3", "q4"):
            value = claims.get(name, {})
            if isinstance(value, Mapping):
                rows.append({"claim": name, "estimate": _estimate(value), "lower": _lookup(value, "lower"), "upper": _lookup(value, "upper"), "status": value.get("status", "undefined"), "direction": value.get("direction")})
    return rows


def claim_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return geometry_rows(summary)


def overlap_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    pooled = _lookup(summary, "genuine_overlap", "overlap")
    result: List[Dict[str, Any]] = []
    if isinstance(pooled, Mapping):
        pooled = pooled.get("pooled", pooled.get("candidates", pooled))
    if not isinstance(pooled, Mapping):
        return result
    for candidate in pooled:
        metrics = pooled[candidate]
        if not isinstance(metrics, Mapping):
            continue
        row: Dict[str, Any] = {"candidate": str(candidate), "status": metrics.get("status", "defined")}
        for field in ("auroc", "auprc", "fpr", "fnr", "brier", "ece"):
            row[field] = _estimate(metrics.get(field))
        result.append(row)
    return result


def family_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    family = _lookup(summary, "family_drift", "families")
    rows: List[Dict[str, Any]] = []
    if not isinstance(family, Mapping):
        return rows
    for candidate in sorted(family, key=lambda value: (CANDIDATE_ORDER.index(str(value)) if str(value) in CANDIDATE_ORDER else len(CANDIDATE_ORDER), str(value))):
        values = family[candidate]
        if not isinstance(values, Mapping):
            continue
        for block, value in sorted(values.items(), key=lambda item: str(item[0])):
            gate = value.get("gate", {}) if isinstance(value, Mapping) else {}
            rows.append({"candidate": str(candidate), "family": str(block), "estimate": _estimate(value), "upper": _lookup(value, "upper"), "status": _lookup(gate, "status") or _status(value)})
    return rows


def shift_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    shift = _lookup(summary, "stable_shift_gates", "stable_shift")
    rows: List[Dict[str, Any]] = []
    if not isinstance(shift, Mapping):
        return rows
    for candidate in sorted(shift, key=lambda value: str(value)):
        balances = shift[candidate]
        if not isinstance(balances, Mapping):
            continue
        for balance, values in balances.items():
            if not isinstance(values, Mapping):
                continue
            for contrast, value in values.items():
                rows.append({"candidate": str(candidate), "balance": str(balance), "contrast": str(contrast), "estimate": _estimate(value), "upper": _lookup(value, "upper"), "status": _lookup(value, "gate.status") or _status(value)})
    return rows


def conditioning_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    diagnostics = _lookup(summary, "diagnostics.conditioning", "conditioning", "conditioning_refinement")
    rows: List[Dict[str, Any]] = []
    if isinstance(diagnostics, Mapping) and any(key in diagnostics for key in ("condition_after", "condition_number")):
        rows.append({
            "candidate": "all", "mode": _lookup(diagnostics, "mode"),
            "condition_before": _estimate(_lookup(diagnostics, "condition_before")),
            "condition_number": _estimate(_lookup(diagnostics, "condition_after", "condition_number")),
            "prototype_activity": _estimate(_lookup(diagnostics, "prototype_activity", "refinement_activity")),
            "regularized_eigenvalue_count": _lookup(diagnostics, "regularized_eigenvalue_count", "regularized_variance_count"),
            "cap_or_floor_active": _lookup(diagnostics, "cap_or_floor_active"),
            "state": "summary",
        })
        return rows
    # build_summary exposes a compact per-candidate diagnostic surface.  Keep
    # conditioning and refinement separate so report consumers can relate
    # activity to later score/regret values without treating it as mediation.
    compact = _lookup(summary, "diagnostics")
    if isinstance(compact, Mapping):
        for candidate in _candidate_names({"candidates": compact}):
            value = compact.get(candidate, {})
            if not isinstance(value, Mapping):
                continue
            cond = value.get("conditioning", {}) if isinstance(value.get("conditioning"), Mapping) else {}
            ref = value.get("refinement", {}) if isinstance(value.get("refinement"), Mapping) else {}
            rows.append({
                "candidate": candidate,
                "mode": cond.get("mode"),
                "condition_before": _estimate(cond.get("condition_before")),
                "condition_number": _estimate(cond.get("condition_after")),
                "prototype_activity": _estimate(ref.get("activity_rate")),
                "regularized_eigenvalue_count": cond.get("regularized_eigenvalue_count"),
                "cap_or_floor_active": cond.get("cap_or_floor_active_rate"),
                "applied_count": _estimate(ref.get("applied_count")),
                "eligible_count": _estimate(ref.get("eligible_count")),
            })
        if rows:
            return rows
    for candidate in _candidate_names(summary):
        value = _candidate(summary, candidate)
        data = _lookup(value, "diagnostics", "conditioning", "conditioning_refinement")
        if not isinstance(data, Mapping):
            data = value
        rows.append({"candidate": candidate, "mode": _lookup(data, "mode"), "condition_before": _estimate(_lookup(data, "condition_before")), "condition_number": _estimate(_lookup(data, "condition_after", "condition_number")), "prototype_activity": _estimate(_lookup(data, "prototype_activity", "refinement_activity")), "clipped_row_fraction": _estimate(_lookup(data, "clipped_row_fraction")), "regularized_eigenvalue_count": _lookup(data, "regularized_eigenvalue_count", "regularized_variance_count"), "cap_or_floor_active": _lookup(data, "cap_or_floor_active")})
    return rows


def geometry_chain_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return descriptive geometry→prototype→score concordance rows."""

    diagnostics = _lookup(summary, "diagnostics")
    result: List[Dict[str, Any]] = []
    if isinstance(diagnostics, Mapping):
        for candidate in _candidate_names({"candidates": diagnostics}):
            value = diagnostics.get(candidate, {})
            if not isinstance(value, Mapping):
                continue
            geometry = value.get("geometry", {}) if isinstance(value.get("geometry"), Mapping) else {}
            prototype = value.get("prototype", {}) if isinstance(value.get("prototype"), Mapping) else {}
            score_delta = value.get("score_delta_vs_B", {}) if isinstance(value.get("score_delta_vs_B"), Mapping) else {}
            refinement_delta = value.get("refinement_B_minus_A", {}) if isinstance(value.get("refinement_B_minus_A"), Mapping) else {}
            concordance = value.get("concordance", {}) if isinstance(value.get("concordance"), Mapping) else {}
            neighbor_link = concordance.get("neighbor_impurity_vs_score_movement", {}) if isinstance(concordance.get("neighbor_impurity_vs_score_movement"), Mapping) else {}
            prototype_link = concordance.get("prototype_purity_vs_score_movement", {}) if isinstance(concordance.get("prototype_purity_vs_score_movement"), Mapping) else {}
            result.append({
                "candidate": candidate,
                "pair_distance_spearman": _estimate(geometry.get("pair_distance_spearman")),
                "neighbor_impurity": _estimate(geometry.get("neighbor_impurity")),
                "oracle_neighbor_jaccard": _estimate(geometry.get("oracle_neighbor_jaccard")),
                "prototype_purity": _estimate(prototype.get("weighted_majority_assignment_purity")),
                "owner_agreement": _estimate(prototype.get("owner_label_agreement")),
                "score": _estimate(value.get("score", {}).get("mean") if isinstance(value.get("score"), Mapping) else None),
                "score_delta_vs_B": _estimate(score_delta.get("mean")),
                "score_delta_vs_B_n": score_delta.get("n"),
                "refinement_B_minus_A": _estimate(refinement_delta.get("mean")),
                "refinement_B_minus_A_n": refinement_delta.get("n"),
                "neighbor_score_movement_spearman": _estimate(neighbor_link.get("spearman")),
                "prototype_score_movement_spearman": _estimate(prototype_link.get("spearman")),
            })
    return result


def resource_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    resources = _lookup(summary, "resource_gates", "resources", "runtime")
    result: List[Dict[str, Any]] = []
    if not isinstance(resources, Mapping):
        return result
    for candidate, value in resources.items():
        if not isinstance(value, Mapping):
            continue
        row: Dict[str, Any] = {"candidate": str(candidate), "status": value.get("status", "undefined")}
        summaries = value.get("summaries", value)
        if isinstance(summaries, Mapping):
            for key in ("median_total", "p95_total", "median_score_fixed", "median_peak_memory", "p95_peak_memory", "individual_peak_memory"):
                row[key] = _estimate(summaries.get(key))
        result.append(row)
    return result


runtime_rows = resource_rows


def decision_rows(summary: Mapping[str, Any], decision: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    eligibility = _lookup(summary, "eligibility")
    decision = decision or {}
    rows: List[Dict[str, Any]] = []
    names = list(CANDIDATE_ORDER)
    supplied_candidates = _lookup(summary, "candidates")
    if isinstance(supplied_candidates, Mapping):
        names.extend(str(name) for name in supplied_candidates if str(name) not in names)
    if isinstance(eligibility, Mapping):
        names = sorted(set(names) | {str(name) for name in eligibility}, key=lambda value: (CANDIDATE_ORDER.index(value) if value in CANDIDATE_ORDER else len(CANDIDATE_ORDER), value))
    for candidate in names:
        info = ROLE_INFO.get(candidate, {"role": "candidate", "scope": "unclassified", "definition": candidate})
        eligibility_value = eligibility.get(candidate, {}) if isinstance(eligibility, Mapping) else {}
        if isinstance(eligibility_value, Mapping):
            status = eligibility_value.get("status", "undefined")
            reasons = eligibility_value.get("reasons", [])
        else:
            status, reasons = "undefined", []
        # A confirmation artifact may carry the locked candidate for audit,
        # but it is not a new promotion.  Mark it selected only after the
        # final confirmation algorithm status passes.
        decision_stage = str(decision.get("stage", decision.get("decision_stage", ""))).lower()
        decision_status = str(decision.get("status", "")).lower()
        algorithm = decision.get("algorithm") if isinstance(decision.get("algorithm"), Mapping) else {}
        algorithm_status = str(algorithm.get("status", "")).lower()
        is_final_confirmation_pass = decision_stage == "confirmation" and (
            algorithm_status == "pass" or (not algorithm and decision_status == "pass")
        )
        selected = candidate == decision.get("selected_candidate", decision.get("locked_candidate")) and (
            decision_stage != "confirmation" or is_final_confirmation_pass
        )
        rows.append({"candidate": candidate, "role": info["role"], "scope": info["scope"], "definition": info["definition"], "eligibility": status, "decision_status": "locked" if selected else status, "selected": selected, "reasons": "; ".join(str(item) for item in reasons)})
    return rows


def _fmt(value: Any) -> str:
    number = _number(value)
    if number is not None:
        return f"{number:.6g}"
    return "—" if value is None or value == "" else str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(item).replace("|", "\\|") for item in row) + " |" for row in rows)
    return "\n".join(lines)


def _svg(title: str, series: Mapping[str, Sequence[float]], *, y_label: str = "value") -> str:
    """Small deterministic point plot; no arbitrary category polylines."""

    width, height = 760, 360
    values = [float(item) for points in series.values() for item in points if _number(item) is not None]
    low, high = (min(values), max(values)) if values else (0.0, 1.0)
    if low == high:
        low -= 0.5; high += 0.5
    colors = ("#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6", "#dd4477")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', f'<title>{html.escape(title)}</title>', '<rect width="100%" height="100%" fill="white"/>', f'<text x="20" y="24" font-family="sans-serif" font-size="16">{html.escape(title)}</text>', f'<text x="12" y="180" transform="rotate(-90 12 180)" font-family="sans-serif" font-size="12">{html.escape(y_label)}</text>', '<line x1="60" y1="40" x2="60" y2="320" stroke="#555"/><line x1="60" y1="320" x2="730" y2="320" stroke="#555"/>']
    for offset, (name, points) in enumerate(sorted(series.items(), key=lambda item: str(item[0]))):
        finite = [float(item) for item in points if _number(item) is not None]
        if not finite:
            continue
        coords = []
        for index, value in enumerate(finite):
            x = 60.0 + 670.0 * (index / max(1, len(finite) - 1))
            y = 320.0 - 260.0 * ((value - low) / (high - low))
            coords.append((x, y))
        color = colors[offset % len(colors)]
        for x, y in coords:
            parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3.5" fill="{color}"/>')
        parts.append(f'<text x="{70 + offset * 105}" y="345" font-family="sans-serif" font-size="11" fill="{color}">{html.escape(str(name))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_report(summary: Mapping[str, Any], decision: Optional[Mapping[str, Any]] = None) -> str:
    decision = decision or {}
    stage = summary.get("stage", "unknown")
    lines = [
        "# Heteroscedastic distance-conditioning experiment",
        "",
        f"Stage: **{stage}**. This report is generated from the normalized artifact and frozen protocol statistics.",
        "",
        "## Scope and decision",
        "",
        "The experiment tests ordered mechanistic concordance along the geometry chain; it does not establish formal causal mediation. Smoke is structural only. Food-101 is not authorized here, so no Food evidence is claimed.",
        "",
        "Algorithmic candidate eligibility and mechanism claims are reported separately; neither surface is a causal mediation claim.",
        "",
        "A and B are exact current OI controls; L is a non-promotable legacy reference. P25, P50-SW, P50-CB, W50-SW, W50-CB, M50-SW, and M50-CB are the algorithmic OI candidates.",
        "",
        f"Lock status: **{decision.get('status', 'not evaluated')}**; selected candidate: **{decision.get('selected_candidate', decision.get('locked_candidate')) or 'none'}**.",
        "",
        "## Geometry chain and primary claims",
        "",
        "The primary endpoint is held-out separated-state cross-class neighbor impurity. The ordered chain is distance ordering → neighbor purity → prototype purity → adjacency → refinement → score → selector regret. Correlations are descriptive and do not imply causality.",
        "",
    ]
    claims = geometry_rows(summary)
    if claims:
        lines.append(_table(("Claim", "Estimate", "Lower", "Upper", "Direction", "Status"), [[row["claim"], _fmt(row["estimate"]), _fmt(row["lower"]), _fmt(row["upper"]), _fmt(row["direction"]), _fmt(row["status"])] for row in claims]))
    else:
        lines.append("No outcome claims are defined for this artifact.")
    lines.extend(["", "## Selector regret and rank AUC", ""])
    selector = selector_rows(summary)
    lines.append(_table(("Candidate", "Head", "Regret", "Rank AUC", "Exact best", "Within 1 point", "Status"), [[row["candidate"], row["head"], _fmt(row["regret"]), _fmt(row["rank_auc"]), _fmt(row["exact_best"]), _fmt(row["within_one_point"]), row["status"]] for row in selector]) if selector else "No complete selector panels are available.")
    lines.extend(["", "## Genuine-overlap detection and calibration", ""])
    lines.append(_table(("Candidate", "AUROC", "AUPRC", "FPR", "FNR", "Brier", "ECE"), [[row["candidate"], _fmt(row["auroc"]), _fmt(row["auprc"]), _fmt(row["fpr"]), _fmt(row["fnr"]), _fmt(row["brier"]), _fmt(row["ece"])] for row in overlap_rows(summary)]) if overlap_rows(summary) else "No pooled directed-pair overlap surface is available.")
    lines.extend(["", "## Family drift and stable-shift gates", ""])
    families = family_rows(summary); shifts = shift_rows(summary)
    lines.append(_table(("Candidate", "Family", "Estimate", "Upper", "Status"), [[row["candidate"], row["family"], _fmt(row["estimate"]), _fmt(row["upper"]), row["status"]] for row in families]) if families else "No complete family blocks are available.")
    if shifts:
        lines.extend(["", _table(("Candidate", "Balance", "Contrast", "Estimate", "Upper", "Status"), [[row["candidate"], row["balance"], row["contrast"], _fmt(row["estimate"]), _fmt(row["upper"]), row["status"]] for row in shifts])])
    lines.extend(["", "## Conditioning and refinement diagnostics", ""])
    conditioning = conditioning_rows(summary)
    lines.append(_table(("Candidate", "Mode", "Condition before", "Condition after", "Prototype activity", "Regularized eigenvalues", "Cap/floor active"), [[row["candidate"], _fmt(row.get("mode")), _fmt(row.get("condition_before")), _fmt(row.get("condition_number")), _fmt(row.get("prototype_activity")), _fmt(row.get("regularized_eigenvalue_count")), _fmt(row.get("cap_or_floor_active"))] for row in conditioning]) if conditioning else "No conditioning diagnostics are available.")
    chain = geometry_chain_rows(summary)
    lines.extend(["", "## Geometry-to-score concordance (descriptive)", ""])
    lines.append(_table(("Candidate", "Pair-distance Spearman", "Neighbor impurity", "Oracle Jaccard", "Prototype purity", "Owner agreement", "Score", "Score Δ vs B", "B−A refinement", "Neighbor/link ρ", "Prototype/link ρ"), [[row["candidate"], _fmt(row.get("pair_distance_spearman")), _fmt(row.get("neighbor_impurity")), _fmt(row.get("oracle_neighbor_jaccard")), _fmt(row.get("prototype_purity")), _fmt(row.get("owner_agreement")), _fmt(row.get("score")), _fmt(row.get("score_delta_vs_B")), _fmt(row.get("refinement_B_minus_A")), _fmt(row.get("neighbor_score_movement_spearman")), _fmt(row.get("prototype_score_movement_spearman"))] for row in chain]) if chain else "No geometry/prototype concordance rows are available.")
    lines.extend(["", "## Runtime and memory", ""])
    resources = resource_rows(summary)
    lines.append(_table(("Candidate", "Median total", "P95 total", "Median score_fixed", "Median memory", "P95 memory", "Individual memory", "Status"), [[row["candidate"], _fmt(row["median_total"]), _fmt(row["p95_total"]), _fmt(row["median_score_fixed"]), _fmt(row["median_peak_memory"]), _fmt(row["p95_peak_memory"]), _fmt(row["individual_peak_memory"]), row["status"]] for row in resources]) if resources else "No complete resource surface is available.")
    lines.extend(["", "## Candidate decisions", "", _table(("Candidate", "Role", "Scope", "Eligibility", "Decision", "Reasons"), [[row["candidate"], row["role"], row["scope"], row["eligibility"], row["decision_status"], row["reasons"]] for row in decision_rows(summary, decision)]), "", "All intervals and gates use the frozen complete-seed paired bootstrap. Missing or incomplete strata are inconclusive; they are not silently treated as passes. Results are evidence for this protocol and are not universal claims."])
    return "\n".join(lines) + "\n"


def write_decision_table(path: Path | str, summary: Mapping[str, Any], decision: Optional[Mapping[str, Any]] = None) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    rows = decision_rows(summary, decision)
    fieldnames = ("candidate", "role", "scope", "definition", "eligibility", "decision_status", "selected", "reasons")
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
    _atomic_write_text(target, buffer.getvalue())


def write_reports(output_path: Path | str, summary: Mapping[str, Any], decision: Optional[Mapping[str, Any]] = None) -> None:
    destination = Path(output_path); destination.mkdir(parents=True, exist_ok=True)
    decision = decision or {}
    _atomic_write_text(destination / "report.md", render_report(summary, decision))
    write_decision_table(destination / "decision_table.csv", summary, decision)
    _atomic_write_text(destination / "decision_table.json", canonical_json(decision_rows(summary, decision)) + "\n")
    selector = selector_rows(summary)
    _atomic_write_text(destination / "geometry_chain.svg", _svg("Geometry chain", {row["claim"]: [value for value in (row["estimate"], row["lower"], row["upper"]) if _number(value) is not None] for row in geometry_rows(summary)}))
    chain = geometry_chain_rows(summary)
    _atomic_write_text(destination / "claims.svg", _svg("Primary claims", {row["candidate"]: [value for value in (row["neighbor_impurity"], row["prototype_purity"], row["score"], row.get("score_delta_vs_B"), row.get("refinement_B_minus_A")) if _number(value) is not None] for row in chain}))
    _atomic_write_text(destination / "selector.svg", _svg("Selector regret and rank AUC", {f"{row['candidate']} {row['head']}": [value for value in (row["regret"], row["rank_auc"]) if _number(value) is not None] for row in selector}))
    _atomic_write_text(destination / "overlap.svg", _svg("Genuine-overlap detection", {str(row["candidate"]): [row[field] for field in ("auroc", "auprc", "fpr", "fnr", "brier") if _number(row.get(field)) is not None] for row in overlap_rows(summary)}))
    _atomic_write_text(destination / "family_shift.svg", _svg("Family drift and stable shift", {f"{row['candidate']} {row['family']}": [value for value in (row["estimate"], row["upper"]) if _number(value) is not None] for row in family_rows(summary)}))
    _atomic_write_text(destination / "refinement_conditioning.svg", _svg("Refinement and conditioning", {str(row["candidate"]): [value for value in (row["condition_number"], row["prototype_activity"]) if _number(value) is not None] for row in conditioning_rows(summary)}))
    _atomic_write_text(destination / "runtime_memory.svg", _svg("Runtime and memory", {str(row["candidate"]): [value for value in (row["median_total"], row["median_peak_memory"]) if _number(value) is not None] for row in resource_rows(summary)}))
    _atomic_write_text(destination / "candidate_decisions.svg", _svg("Candidate decisions", {row["candidate"]: [1.0 if row["selected"] else 0.0] for row in decision_rows(summary, decision)}))


__all__ = [
    "claim_rows", "conditioning_rows", "decision_rows", "family_rows", "geometry_chain_rows", "geometry_rows", "overlap_rows", "render_report", "resource_rows", "runtime_rows", "selector_rows", "shift_rows", "write_decision_table", "write_reports",
]
