"""Machine-readable and human-readable reporting for M50 backbone ranking."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("reports must not serialize NaN or infinity")
    return value


def _json_text(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _candidate_gate_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    candidate_gates = summary.get("candidate_gates", {})
    if not isinstance(candidate_gates, Mapping):
        return result
    for candidate, payload in candidate_gates.items():
        if not isinstance(payload, Mapping):
            continue
        gates = payload.get("gates", {})
        if not isinstance(gates, Mapping):
            continue
        for identity, gate in gates.items():
            if not isinstance(gate, Mapping):
                continue
            result.append(
                {
                    "candidate_id": str(candidate),
                    "cell": str(identity),
                    "estimate_pp": gate.get("estimate"),
                    "lower_95_pp": gate.get("lower_95"),
                    "upper_95_pp": gate.get("upper_95"),
                    "status": gate.get("status"),
                    "rule": gate.get("rule"),
                }
            )
    for summary_key, fallback_candidate in (
        ("fusion_gates", "F"),
        ("guardrail_gates", "G"),
    ):
        payload = summary.get(summary_key)
        if not isinstance(payload, Mapping):
            continue
        candidate = str(payload.get("candidate_id", fallback_candidate))
        gates = payload.get("gates")
        if not isinstance(gates, Mapping):
            continue
        for identity, gate in gates.items():
            if isinstance(gate, Mapping):
                result.append(
                    {
                        "candidate_id": candidate,
                        "cell": str(identity),
                        "estimate_pp": gate.get("estimate"),
                        "lower_95_pp": gate.get("lower_95"),
                        "upper_95_pp": gate.get("upper_95"),
                        "status": gate.get("status"),
                        "rule": gate.get("rule"),
                    }
                )
    return sorted(result, key=lambda row: (row["candidate_id"], row["cell"]))


def render_markdown_report(
    summary: Mapping[str, Any],
    development_lock: Mapping[str, Any],
) -> str:
    """Render a concise report with the required claim-scope distinctions."""

    lines = [
        "# M50 backbone-ranking experiment",
        "",
        "## Scope",
        "",
        "Food-101 is retrospective development evidence, not untouched confirmation. "
        "All conclusions concern ranking backbones under class-stratified "
        "same-distribution sampling; they do not "
        "generalize to unstratified imbalance. Distribution-shift stress tests and generic overlap calibration "
        "are diagnostic only unless they are linked to a ranking failure.",
        "",
        "## Development decision",
        "",
        f"Status: **{development_lock.get('status', 'unknown')}**. ",
        f"Conditioned OI base candidate: **{development_lock.get('selected_candidate') or 'none'}**. ",
        f"Locked selector chain: **{development_lock.get('selected_chain') or 'none'}**.",
        f"Chain kind: **{development_lock.get('chain_kind') or 'pending'}**. ",
        f"Algorithmic OI status: **{development_lock.get('algorithmic_oi_status') or 'pending'}**. ",
        f"Product-policy status: **{development_lock.get('product_policy_status') or 'pending'}**.",
        "",
        "This V1 status is a retrospective development/resource lock only. It is not "
        "a confirmed product claim; a separately frozen future V2 confirmation is required.",
        "",
        "Class-balanced M50 arms are diagnostic and cannot rescue or replace the "
        "sample-weighted candidate on the balanced Food panel. Runtime failure rejects "
        "the provisional chain; it does not promote a runner-up.",
        "",
    ]
    recipe_bindings = development_lock.get("selected_chain_recipe_bindings")
    if isinstance(recipe_bindings, Mapping):
        lines.extend(
            [
                "## Hash-bound executable recipe",
                "",
                f"Binding SHA-256: `{development_lock.get('selected_chain_recipe_bindings_sha256')}`.",
                "",
                "| Primitive component | Food recipe family | Runtime recipe family |",
                "|---|---|---|",
            ]
        )
        component_hashes = recipe_bindings.get("component_recipe_family_sha256")
        if isinstance(component_hashes, Mapping):
            for component, value in sorted(component_hashes.items()):
                if isinstance(value, Mapping):
                    lines.append(
                        f"| {component} | `{value.get('food_recipe_family_sha256')}` | "
                        f"`{value.get('runtime_recipe_family_sha256')}` |"
                    )
        derived_hashes = recipe_bindings.get("derived_recipe_sha256")
        if isinstance(derived_hashes, Mapping) and derived_hashes:
            lines.extend(["", "Derived-chain recipe hashes:", ""])
            for candidate_id, digest in sorted(derived_hashes.items()):
                lines.append(f"- {candidate_id}: `{digest}`")
        seed_rules = recipe_bindings.get("deterministic_seed_rule")
        if isinstance(seed_rules, Mapping):
            lines.extend(["", "Deterministic seed rules:", ""])
            for stage, rule in sorted(seed_rules.items()):
                lines.append(
                    f"- {stage}: `"
                    + json.dumps(
                        _json_safe(rule),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "`"
                )
        lines.extend(
            [
                "",
                "The binding also freezes the Food and runtime deterministic seed rules; "
                "future V2 must hash-link these exact bytes.",
                "",
            ]
        )
    lines.extend(
        [
        "## Regret gates versus the directly measured full linear probe",
        "",
        "| Candidate | Cell | Estimate (pp) | 95% interval (pp) | Status |",
        "|---|---|---:|---:|---|",
        ]
    )
    for row in _candidate_gate_rows(summary):
        interval = f"[{_format_number(row['lower_95_pp'])}, {_format_number(row['upper_95_pp'])}]"
        lines.append(
            f"| {row['candidate_id']} | {row['cell']} | "
            f"{_format_number(row['estimate_pp'])} | {interval} | {row['status']} |"
        )
    if not _candidate_gate_rows(summary):
        lines.append("| — | — | — | — | unavailable |")

    lines.extend(
        [
            "",
            "## Candidate decision table",
            "",
            "| Candidate | Role | Decision status | OI-promotable | Selected chain |",
            "|---|---|---|---|---|",
        ]
    )
    for candidate, role, status in (
        ("A", "raw unrefined parity control", "control only"),
        ("B", "raw refined parity control", "control only"),
        ("M0-CB", "class-balanced conditioning diagnostic", "diagnostic only"),
        ("M1-CB", "class-balanced conditioning diagnostic", "diagnostic only"),
        ("LP-FULL", "direct full linear-probe comparator", "comparator only"),
    ):
        lines.append(f"| {candidate} | {role} | {status} | no | no |")
    candidates = summary.get("candidate_gates")
    if isinstance(candidates, Mapping):
        for candidate, payload in sorted(candidates.items()):
            status = payload.get("status") if isinstance(payload, Mapping) else "malformed"
            lines.append(
                f"| {candidate} | sample-weighted OI | {status} | yes | "
                f"{'yes' if development_lock.get('selected_chain') == candidate else 'no'} |"
            )
    for candidate, key, role, promotable in (
        ("F", "fusion_gates", "fixed raw+M50 OI fusion", "yes"),
        ("G", "guardrail_gates", "product-only capped-probe guardrail", "no (product only)"),
    ):
        payload = summary.get(key)
        status = payload.get("status") if isinstance(payload, Mapping) else "not evaluated"
        lines.append(
            f"| {candidate} | {role} | {status} | {promotable} | "
            f"{'yes' if development_lock.get('selected_chain') == candidate else 'no'} |"
        )

    selection_rates = summary.get("selection_rates")
    lines.extend(["", "## Selection rates and average regret", ""])
    if isinstance(selection_rates, Sequence) and not isinstance(selection_rates, (str, bytes)):
        lines.extend(
            [
                "| Candidate | Arm | Head | Mean regret (pp) | Exact best | Within 1 pp |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in selection_rates:
            if isinstance(row, Mapping):
                lines.append(
                    f"| {row.get('candidate_id')} | {row.get('arm')} | {row.get('head')} | "
                    f"{_format_number(row.get('mean_regret_pp'))} | "
                    f"{_format_number(row.get('exact_best_rate'))} | "
                    f"{_format_number(row.get('within_one_pp_rate'))} |"
                )
    else:
        lines.append("Selection-rate evidence is unavailable.")

    rank_summary = summary.get("rank_auc_summary")
    lines.extend(["", "## Rank AUC", ""])
    if isinstance(rank_summary, Sequence) and not isinstance(rank_summary, (str, bytes)):
        lines.extend(
            [
                "| Candidate | Arm | Head | Mean rank AUC | Status | Undefined replicates |",
                "|---|---|---|---:|---|---:|",
            ]
        )
        for row in rank_summary:
            if isinstance(row, Mapping):
                lines.append(
                    f"| {row.get('candidate_id')} | {row.get('arm')} | {row.get('head')} | "
                    f"{_format_number(row.get('mean_rank_auc'))} | {row.get('status')} | "
                    f"{_format_number(row.get('undefined_replicate_count'))} |"
                )
    else:
        lines.append("Rank-AUC evidence is unavailable.")

    factorial = summary.get("factorial", {})
    lines.extend(
        [
            "",
            "## Conditioning × refinement factorial",
            "",
            "The factorial compares A/B with M0-SW/M1-SW. "
            "The interaction is `(M50 refined − M50 unrefined) − "
            "(raw refined − raw unrefined)` in regret percentage points.",
            "",
        ]
    )
    if isinstance(factorial, Mapping) and factorial.get("effects"):
        lines.extend(
            [
                "| Cell | Raw refinement | M50 refinement | Interaction | CB diagnostic |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for cell, payload in sorted(factorial["effects"].items()):
            lines.append(
                f"| {cell} | "
                f"{_format_number(payload['raw_refinement_effect'].get('estimate'))} | "
                f"{_format_number(payload['conditioned_refinement_effect'].get('estimate'))} | "
                f"{_format_number(payload['interaction'].get('estimate'))} | "
                f"{_format_number(payload.get('cb_refinement_diagnostic', {}).get('estimate'))} |"
            )
    else:
        lines.append("Factorial evidence is unavailable.")

    resource = summary.get("resources", {})
    lines.extend(
        [
            "",
            "## Runtime and memory",
            "",
            "Each arm is gated independently and all three arms must pass. Pooled and "
            "budget-scaling summaries are descriptive only.",
            "Warm-up is excluded from selector scores and wall/CPU clocks. Fresh-process "
            "peak RSS conservatively includes warm-up.",
            "",
        ]
    )
    if isinstance(resource, Mapping) and resource.get("arm_gates"):
        full_panel_gates = resource.get("full_panel_arm_gates")
        if isinstance(full_panel_gates, Mapping):
            lines.extend(
                [
                    "### Full 10-backbone product-panel timing gates",
                    "",
                    "These ratios sum outer wall time across all ten backbones within "
                    "each arm, budget, and repeat before comparison. They are promotion "
                    "gates in addition to the per-backbone-call gates below.",
                    "",
                    "| Arm | Full-panel gate | Median | P95 / upper 95% | Status |",
                    "|---|---|---:|---:|---|",
                ]
            )
            for arm, arm_payload in sorted(full_panel_gates.items()):
                if not isinstance(arm_payload, Mapping):
                    continue
                gates = arm_payload.get("gates")
                if not isinstance(gates, Mapping):
                    continue
                for name, gate in sorted(gates.items()):
                    if not isinstance(gate, Mapping):
                        continue
                    tail = gate.get("p95", gate.get("upper_95"))
                    lines.append(
                        f"| {arm} | {name} | {_format_number(gate.get('median'))} | "
                        f"{_format_number(tail)} | {gate.get('status')} |"
                    )
            lines.append("")
        lines.extend(
            [
                "### Per-backbone-call timing and memory gates",
                "",
                "| Arm | Per-call gate | Median | P95 / upper 95% | Status |",
                "|---|---|---:|---:|---|",
            ]
        )
        for arm, arm_payload in sorted(resource["arm_gates"].items()):
            if not isinstance(arm_payload, Mapping):
                continue
            gates = arm_payload.get("gates")
            if not isinstance(gates, Mapping):
                continue
            for name, gate in sorted(gates.items()):
                if not isinstance(gate, Mapping):
                    continue
                tail = gate.get("p95", gate.get("upper_95"))
                lines.append(
                    f"| {arm} | {name} | {_format_number(gate.get('median'))} | "
                    f"{_format_number(tail)} | {gate.get('status')} |"
                )
        lines.extend(
            [
                "",
                "### Faster than LP-FULL claim (non-promotion)",
                "",
                "This claim is established per arm only when both the paired per-backbone "
                "and full 10-backbone product-panel candidate/LP-FULL wall-time ratios "
                "have upper 95% bounds below 1.0. It is not an additional promotion veto.",
                "",
                "| Arm | Per-backbone upper 95% | Full-panel upper 95% | Claim status |",
                "|---|---:|---:|---|",
            ]
        )
        for arm, arm_payload in sorted(resource["arm_gates"].items()):
            if isinstance(arm_payload, Mapping):
                claim = arm_payload.get("faster_claim")
                per_backbone_upper = (
                    claim.get("per_backbone_upper_95")
                    if isinstance(claim, Mapping)
                    else None
                )
                full_panel_upper = (
                    claim.get("full_panel_upper_95")
                    if isinstance(claim, Mapping)
                    else None
                )
                lines.append(
                    f"| {arm} | {_format_number(per_backbone_upper)} | "
                    f"{_format_number(full_panel_upper)} | "
                    f"{arm_payload.get('faster_claim_status', 'inconclusive')} |"
                )

        secondary = resource.get("secondary_runtime")
        if isinstance(secondary, Mapping):
            lines.extend(
                [
                    "",
                    "### Descriptive CPU and stage clocks (noninferential, nonpromotion)",
                    "",
                    "CPU ratios and directly measured stage clocks are secondary diagnostics. "
                    "No stage clock is derived by subtracting noisy totals, and none changes "
                    "the frozen wall/memory promotion gates.",
                    "",
                    "| Arm | CPU estimand | Candidate / LP median (P95) | Candidate / B median (P95) |",
                    "|---|---|---:|---:|",
                ]
            )
            arm_summary = secondary.get("arm_summary")
            if isinstance(arm_summary, Mapping):
                for arm, payload in sorted(arm_summary.items()):
                    if not isinstance(payload, Mapping):
                        continue
                    for label, field in (
                        ("per-backbone call", "per_backbone_cpu_ratios"),
                        ("full 10-backbone panel", "full_panel_cpu_ratios"),
                    ):
                        ratios = payload.get(field)
                        if not isinstance(ratios, Mapping):
                            continue
                        probe = ratios.get("candidate_vs_probe")
                        baseline = ratios.get("candidate_vs_B")
                        if isinstance(probe, Mapping) and isinstance(baseline, Mapping):
                            lines.append(
                                f"| {arm} | {label} | "
                                f"{_format_number(probe.get('median'))} "
                                f"({_format_number(probe.get('p95'))}) | "
                                f"{_format_number(baseline.get('median'))} "
                                f"({_format_number(baseline.get('p95'))}) |"
                            )
                lines.extend(
                    [
                        "",
                        "| Arm | Direct stage | Median seconds | P95 seconds |",
                        "|---|---|---:|---:|",
                    ]
                )
                for arm, payload in sorted(arm_summary.items()):
                    stages = payload.get("stage_seconds") if isinstance(payload, Mapping) else None
                    if not isinstance(stages, Mapping):
                        continue
                    for stage, values in sorted(stages.items()):
                        if isinstance(values, Mapping):
                            lines.append(
                                f"| {arm} | {stage} | "
                                f"{_format_number(values.get('median'))} | "
                                f"{_format_number(values.get('p95'))} |"
                            )
            secondary_scaling = secondary.get("scaling_by_arm_budget")
            if (
                isinstance(secondary_scaling, Sequence)
                and not isinstance(secondary_scaling, (str, bytes))
                and secondary_scaling
            ):
                lines.extend(
                    [
                        "",
                        "#### Descriptive CPU/stage scaling by arm and budget",
                        "",
                        "| Arm | Budget | Candidate/LP CPU | Conditioning wall | OI fit/refinement wall | Score-fixed wall | Probe fit wall | Probe predict wall |",
                        "|---|---:|---:|---:|---:|---:|---:|---:|",
                    ]
                )
                for row in secondary_scaling:
                    if isinstance(row, Mapping):
                        lines.append(
                            f"| {row.get('arm')} | {row.get('budget')} | "
                            f"{_format_number(row.get('candidate_probe_cpu_ratio_median'))} | "
                            f"{_format_number(row.get('candidate_conditioning_fit_wall_seconds_median'))} | "
                            f"{_format_number(row.get('candidate_oi_fit_wall_seconds_median'))} | "
                            f"{_format_number(row.get('candidate_score_fixed_wall_seconds_median'))} | "
                            f"{_format_number(row.get('probe_fit_wall_seconds_median'))} | "
                            f"{_format_number(row.get('probe_score_fixed_wall_seconds_median'))} |"
                        )
    else:
        lines.append("Direct paired runtime evidence is unavailable or pending.")
    scaling = resource.get("scaling_by_budget") if isinstance(resource, Mapping) else None
    if isinstance(scaling, Sequence) and not isinstance(scaling, (str, bytes)) and scaling:
        lines.extend(
            [
                "",
                "### Descriptive resource scaling by budget (pooled across arms)",
                "",
                "| Budget | Candidate / LP wall | Candidate / B wall | Candidate / B memory |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in scaling:
            if isinstance(row, Mapping):
                lines.append(
                    f"| {row.get('budget')} | {_format_number(row.get('candidate_probe_wall_median'))} | "
                    f"{_format_number(row.get('candidate_B_wall_median'))} | "
                    f"{_format_number(row.get('candidate_B_memory_median'))} |"
                )

    lines.extend(
        [
            "",
            "## Algorithmic selector versus product guardrail",
            "",
            "M50 and the fixed 50/50 raw+M50 rank fusion F are algorithmic selector "
            "candidates. G is a separate selective capped-probe product policy. G may "
            "protect linear ranking, but it never rescues a failed nonlinear superiority "
            "claim and must be charged for every diagnostic and probe component it runs.",
            "",
            "## Confirmation boundary",
            "",
            "No Food or previously inspected synthetic panel is untouched confirmation. "
            "V1 ends at `locked_for_future_confirmation` and makes no confirmed product "
            "claim. A future V2 protocol must consume the exact locked chain, register its "
            "new datasets before embeddings or outcomes are generated, and forbid candidate "
            "reselection.",
            "",
        ]
    )
    confirmation = summary.get("confirmation")
    if isinstance(confirmation, Mapping):
        lines.extend(
            [
                "## Future V2 confirmation design evidence",
                "",
                "This optional structure exercises predeclared gate reporting only; V1 "
                "cannot execute or certify confirmation outcomes.",
                "",
                f"Design-evidence status: **{confirmation.get('status', 'unknown')}**.",
                "",
                "| Head | Estimate (pp) | 95% interval (pp) | Status |",
                "|---|---:|---:|---|",
            ]
        )
        gates = confirmation.get("gates")
        if isinstance(gates, Mapping):
            for head, gate in sorted(gates.items()):
                if isinstance(gate, Mapping):
                    lines.append(
                        f"| {head} | {_format_number(gate.get('estimate'))} | "
                        f"[{_format_number(gate.get('lower_95'))}, "
                        f"{_format_number(gate.get('upper_95'))}] | {gate.get('status')} |"
                    )
        lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialized = sorted(
        (dict(row) for row in rows),
        key=lambda row: json.dumps(
            _json_safe(row), sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    )
    fields = sorted({str(key) for row in materialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    field: json.dumps(_json_safe(row.get(field)), sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else _json_safe(row.get(field))
                    for field in fields
                }
            )


def _bar_svg(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_field: str,
    value_field: str,
    title: str,
) -> str:
    values = [
        (str(row.get(label_field)), float(row[value_field]))
        for row in rows
        if row.get(value_field) is not None and np.isfinite(float(row[value_field]))
    ]
    width, left, row_height = 760, 230, 28
    height = max(100, 70 + row_height * len(values))
    maximum = max([abs(value) for _label, value in values] + [1.0])
    plot_width = width - left - 40
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    for position, (label, value) in enumerate(values):
        y = 50 + position * row_height
        bar_width = abs(value) / maximum * plot_width
        color = "#2f6f9f" if value <= 0.0 else "#c75b39"
        pieces.extend(
            [
                f'<text x="20" y="{y + 15}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.3f}" height="18" fill="{color}"/>',
                f'<text x="{left + bar_width + 6:.3f}" y="{y + 14}" font-family="sans-serif" font-size="11">{value:.3f}</text>',
            ]
        )
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def write_report_bundle(
    output: Path | str,
    *,
    summary: Mapping[str, Any],
    development_lock: Mapping[str, Any],
    panel_metrics: Sequence[Mapping[str, Any]],
    rank_auc: Sequence[Mapping[str, Any]],
    factorial_rows: Sequence[Mapping[str, Any]] = (),
    resource_rows: Sequence[Mapping[str, Any]] = (),
    full_panel_resource_rows: Sequence[Mapping[str, Any]] = (),
    selected_chain_recipe_bindings: Mapping[str, Any] | None = None,
    linked_development_files: Mapping[str, Path | str] | None = None,
    lineage: Mapping[str, Any],
) -> Path:
    """Atomically create one non-overwriting analysis directory."""

    required_lineage = {
        "stage",
        "status",
        "protocol_sha256",
        "code_identity_sha256",
        "input_manifest_sha256",
    }
    if set(lineage) != required_lineage or any(
        not isinstance(lineage[key], str) or not lineage[key]
        for key in required_lineage
    ):
        raise ValueError(
            "lineage must contain exactly non-empty stage, status, protocol_sha256, "
            "code_identity_sha256, and input_manifest_sha256 strings"
        )
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite analysis output {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        (temporary / "summary.json").write_text(_json_text(summary), encoding="utf-8")
        (temporary / "development_lock.json").write_text(
            _json_text(development_lock), encoding="utf-8"
        )
        recipe_binding_hash = None
        if selected_chain_recipe_bindings is not None:
            normalized_recipe_bindings = dict(selected_chain_recipe_bindings)
            recipe_binding_hash = hashlib.sha256(
                json.dumps(
                    _json_safe(normalized_recipe_bindings),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            if (
                development_lock.get("selected_chain_recipe_bindings")
                != normalized_recipe_bindings
                or development_lock.get("selected_chain_recipe_bindings_sha256")
                != recipe_binding_hash
            ):
                raise ValueError(
                    "selected-chain recipe bindings do not match the final development lock"
                )
            (temporary / "selected_chain_recipe_bindings.json").write_text(
                _json_text(normalized_recipe_bindings), encoding="utf-8"
            )
        _write_csv(temporary / "panel_metrics.csv", panel_metrics)
        _write_csv(temporary / "rank_auc.csv", rank_auc)
        _write_csv(temporary / "factorial.csv", factorial_rows)
        _write_csv(temporary / "resource.csv", resource_rows)
        _write_csv(
            temporary / "full_panel_resource.csv", full_panel_resource_rows
        )
        (temporary / "report.md").write_text(
            render_markdown_report(summary, development_lock), encoding="utf-8"
        )
        gate_rows = _candidate_gate_rows(summary)
        gate_plot_rows = [
            {
                **row,
                "candidate_cell": f"{row['candidate_id']}:{row['cell']}",
            }
            for row in gate_rows
        ]
        (temporary / "selector_regret.svg").write_text(
            _bar_svg(
                gate_plot_rows,
                label_field="candidate_cell",
                value_field="estimate_pp",
                title="Candidate-minus-probe regret (pp)",
            ),
            encoding="utf-8",
        )
        resource_plot_rows = []
        arm_gates = summary.get("resources", {}).get("arm_gates", {}) if isinstance(summary.get("resources"), Mapping) else {}
        if isinstance(arm_gates, Mapping):
            for arm, arm_payload in sorted(arm_gates.items()):
                if not isinstance(arm_payload, Mapping):
                    continue
                gates = arm_payload.get("gates")
                if not isinstance(gates, Mapping):
                    continue
                for name, payload in sorted(gates.items()):
                    if isinstance(payload, Mapping) and payload.get("median") is not None:
                        resource_plot_rows.append(
                            {"gate": f"{arm}:{name}", "ratio": payload["median"]}
                        )
        (temporary / "runtime_ratio.svg").write_text(
            _bar_svg(
                resource_plot_rows,
                label_field="gate",
                value_field="ratio",
                title="Paired median runtime/memory ratios",
            ),
            encoding="utf-8",
        )
        linked_names: list[str] = []
        if linked_development_files is not None:
            for name, source_value in sorted(linked_development_files.items()):
                if Path(name).name != name or not name:
                    raise ValueError("linked development file names must be plain basenames")
                source = Path(source_value).resolve()
                if not source.is_file():
                    raise ValueError(f"linked development file is missing: {source}")
                destination_name = f"development_{name}"
                if (temporary / destination_name).exists():
                    raise ValueError(f"duplicate linked development file {destination_name!r}")
                (temporary / destination_name).write_bytes(source.read_bytes())
                linked_names.append(destination_name)
        artifact_files = sorted(
            path for path in temporary.iterdir() if path.name != "analysis_manifest.json"
        )

        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            return digest.hexdigest()

        row_counts = {
            "panel_metrics.csv": len(panel_metrics),
            "rank_auc.csv": len(rank_auc),
            "factorial.csv": len(factorial_rows),
            "resource.csv": len(resource_rows),
            "full_panel_resource.csv": len(full_panel_resource_rows),
        }
        for name in linked_names:
            if Path(name).suffix == ".csv":
                with (temporary / name).open("r", encoding="utf-8", newline="") as handle:
                    csv_rows = list(csv.reader(handle))
                row_counts[name] = max(0, len(csv_rows) - 1) if csv_rows else 0
        lock_hash = sha256(temporary / "development_lock.json")
        manifest = {
            "schema_version": 1,
            **dict(lineage),
            "development_lock_sha256": lock_hash,
            **(
                {"selected_chain_recipe_bindings_sha256": recipe_binding_hash}
                if recipe_binding_hash is not None
                else {}
            ),
            "files": {
                path.name: {
                    "sha256": sha256(path),
                    "row_count": row_counts.get(path.name),
                }
                for path in artifact_files
            },
        }
        (temporary / "analysis_manifest.json").write_text(
            _json_text(manifest), encoding="utf-8"
        )
        temporary.replace(destination)
    except Exception:
        # Keep cleanup narrowly scoped to the private temporary directory.
        for child in temporary.iterdir():
            if child.is_file():
                child.unlink()
        temporary.rmdir()
        raise
    return destination


__all__ = [
    "render_markdown_report",
    "write_report_bundle",
]
