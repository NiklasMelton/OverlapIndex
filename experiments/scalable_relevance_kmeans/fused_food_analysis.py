"""Analysis and immutable decision for the fused-prototype Food-101 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from experiments.scalable_relevance_kmeans import fused_food_screen as screen


HEADS = ("linear", "quadratic", "knn", "rbf")
NONLINEAR_HEADS = ("quadratic", "knn", "rbf")
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
    _atomic_text(path, screen.canonical_json(value) + "\n")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_fold(row: Mapping[str, Any], candidate_id: str, fold_index: int) -> None:
    if int(row.get("fold", -1)) != fold_index:
        raise RuntimeError("fused Food fold order/index mismatch")
    for field in (
        "score",
        "fit_wall_seconds",
        "fit_cpu_seconds",
        "score_fixed_wall_seconds",
        "score_fixed_cpu_seconds",
    ):
        value = row.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or float(value) < 0.0:
            raise RuntimeError(f"fused Food fold has invalid {field}")
    identity = row.get("candidate_config_identity")
    digest = row.get("candidate_config_sha256")
    if not isinstance(identity, Mapping) or screen._hash_payload(identity) != digest:
        raise RuntimeError("fused Food fold configuration hash mismatch")
    if str(identity.get("candidate_id")) != candidate_id:
        raise RuntimeError("fused Food fold candidate configuration identity mismatch")
    if candidate_id in {"NOLLOYD", "LOOP3", "FUSED3"}:
        if identity.get("spec") != screen._candidate_specs()[candidate_id]:
            raise RuntimeError("fused Food fold candidate spec drift")
        if int(identity.get("fold_seed", -1)) != screen.SEED + fold_index:
            raise RuntimeError("fused Food fold seed drift")
        k = identity.get("k_per_class")
        if not isinstance(k, Mapping) or len(k) != 40 or set(k.values()) != {10}:
            raise RuntimeError("fused Food fold k-per-class drift")
        if not _is_sha256(row.get("numerical_state_sha256")):
            raise RuntimeError("fused Food fold lacks numerical fitted-state identity")
        if row.get("state_unchanged_after_score_fixed") is not True:
            raise RuntimeError("fused Food score_fixed state invariant failed")
    elif candidate_id != "LP-FULL":
        raise RuntimeError(f"unknown fused Food candidate {candidate_id!r}")


def verify_artifact(input_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    manifest = _read_json(input_dir / "manifest.json")
    raw_path = input_dir / "raw_results.json"
    raw = _read_json(raw_path)
    if manifest.get("artifact_status") != "completed" or raw.get("artifact_status") != "completed":
        raise RuntimeError("fused Food artifact is not completed")
    if manifest.get("stage") != "development" or raw.get("stage") != "development":
        raise RuntimeError("fused Food analysis requires the development stage")
    if manifest.get("raw_results_sha256") != screen.sha256_path(raw_path):
        raise RuntimeError("fused Food raw-results hash mismatch")
    if manifest.get("run_identity") != raw.get("run_identity"):
        raise RuntimeError("fused Food manifest/raw run identity mismatch")
    identity = raw.get("run_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("fused Food run identity is malformed")
    identity_sha = screen._hash_payload(identity)
    if (
        raw.get("run_identity_sha256") != identity_sha
        or manifest.get("run_identity_sha256") != identity_sha
    ):
        raise RuntimeError("fused Food run-identity hash mismatch")
    protocol_sha, _protocol = screen.validate_protocol()
    if identity.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("fused Food protocol identity mismatch")
    if identity.get("source_identity") != screen.source_identity():
        raise RuntimeError("fused Food executed source identity no longer matches")
    expected_counts = {
        "panels": 30,
        "selector_rows": 120,
        "prior_parity_rows": 60,
        "implementation_parity_rows": 30,
        "reference_rows": 120,
        "checkpoints": 30,
    }
    if manifest.get("counts") != expected_counts:
        raise RuntimeError("fused Food terminal counts mismatch")

    selector_rows = raw.get("selector_rows")
    if not isinstance(selector_rows, list) or len(selector_rows) != 120:
        raise RuntimeError("fused Food selector grid is incomplete")
    expected_keys = {
        (model, screen.REPLICATE, arm, screen.BUDGET, method)
        for model in screen.MODELS
        for arm in screen.ARMS
        for method in screen.METHODS
    }
    observed_keys: set[tuple[str, int, str, int, str]] = set()
    panel_positions = {
        (model, arm): panel_index
        for panel_index, (model, arm) in enumerate(
            (model, arm) for model in screen.MODELS for arm in screen.ARMS
        )
    }
    for row in selector_rows:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            raise RuntimeError("fused Food contains a malformed/non-ok selector row")
        key = (
            str(row.get("model")),
            int(row.get("replicate", -1)),
            str(row.get("arm")),
            int(row.get("budget", -1)),
            str(row.get("candidate_id")),
        )
        if key in observed_keys:
            raise RuntimeError("fused Food duplicates a selector row")
        observed_keys.add(key)
        if row.get("warmup_excluded") is not True:
            raise RuntimeError("fused Food selector row includes warmup")
        expected_order = screen._method_order(panel_positions[(key[0], key[2])])
        if (
            row.get("execution_order") != list(expected_order)
            or int(row.get("execution_position", -1)) != expected_order.index(key[-1])
        ):
            raise RuntimeError("fused Food execution schedule drift")
        if type(row.get("prototype_refinement_enabled")) is not bool or row.get(
            "prototype_refinement_enabled"
        ):
            raise RuntimeError("fused Food refinement flag drift")
        for field in (
            "score",
            "fit_wall_seconds",
            "fit_cpu_seconds",
            "score_fixed_wall_seconds",
            "score_fixed_cpu_seconds",
            "total_wall_seconds",
            "total_cpu_seconds",
        ):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or float(value) < 0.0:
                raise RuntimeError(f"fused Food selector row has invalid {field}")
        if float(row["total_wall_seconds"]) <= 0.0:
            raise RuntimeError("fused Food selector row has non-positive outer wall time")
        folds = row.get("folds")
        if not isinstance(folds, list) or len(folds) != screen.FOLDS:
            raise RuntimeError("fused Food selector fold grid is incomplete")
        for index, fold in enumerate(folds):
            if not isinstance(fold, Mapping):
                raise RuntimeError("fused Food selector fold row is malformed")
            _validate_fold(fold, key[-1], index)
    if observed_keys != expected_keys:
        raise RuntimeError("fused Food selector identities do not match frozen grid")

    references = raw.get("reference_rows")
    expected_reference_keys = {
        (model, screen.REPLICATE, arm, head)
        for model in screen.MODELS
        for arm in screen.ARMS
        for head in HEADS
    }
    reference_keys: set[tuple[str, int, str, str]] = set()
    if not isinstance(references, list) or len(references) != 120:
        raise RuntimeError("fused Food reference grid is incomplete")
    for row in references:
        key = (
            str(row.get("backbone")),
            int(row.get("replicate", -1)),
            str(row.get("arm")),
            str(row.get("head")),
        )
        if key in reference_keys:
            raise RuntimeError("fused Food duplicates a reference row")
        reference_keys.add(key)
        value = row.get("test_accuracy")
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise RuntimeError("fused Food reference outcome is non-finite")
    if reference_keys != expected_reference_keys:
        raise RuntimeError("fused Food reference identities do not match frozen grid")

    prior = raw.get("prior_parity")
    if not isinstance(prior, list) or len(prior) != 60 or any(
        not isinstance(row, Mapping)
        or row.get("exact") is not True
        or float(row.get("delta", np.nan)) != 0.0
        for row in prior
    ):
        raise RuntimeError("fused Food prior parity evidence failed")
    implementation = raw.get("implementation_parity")
    if not isinstance(implementation, list) or len(implementation) != 30 or any(
        not isinstance(row, Mapping)
        or row.get("exact") is not True
        or row.get("exact_panel_score") is not True
        or row.get("exact_folds") is not True
        or not isinstance(row.get("folds"), list)
        or len(row["folds"]) != screen.FOLDS
        or any(
            not isinstance(fold, Mapping)
            or fold.get("exact") is not True
            or not _is_sha256(fold.get("looped_numerical_state_sha256"))
            or fold.get("looped_numerical_state_sha256")
            != fold.get("fused_numerical_state_sha256")
            for fold in row["folds"]
        )
        for row in implementation
    ):
        raise RuntimeError("fused Food LOOP3/FUSED3 parity evidence failed")
    determinism = raw.get("determinism")
    if not isinstance(determinism, Mapping) or set(determinism) != set(screen.METHODS):
        raise RuntimeError("fused Food determinism evidence is incomplete")
    for method, record in determinism.items():
        if (
            not isinstance(record, Mapping)
            or record.get("candidate_id") != method
            or record.get("exact") is not True
            or record.get("runtime_fields_excluded") is not True
            or not _is_sha256(record.get("first_signature_sha256"))
            or record.get("first_signature_sha256") != record.get("second_signature_sha256")
        ):
            raise RuntimeError("fused Food determinism descriptor is malformed")
    return raw


def _selection_metrics(scores: np.ndarray, outcomes: np.ndarray) -> dict[str, Any]:
    if scores.shape != (10,) or outcomes.shape != (10,):
        raise RuntimeError("selection metrics require one complete ten-backbone panel")
    selected = np.flatnonzero(
        np.isclose(scores, float(np.max(scores)), rtol=0.0, atol=1.0e-12)
    )
    best = float(np.max(outcomes))
    selected_outcome = float(np.mean(outcomes[selected]))
    spearman = float(spearmanr(scores, outcomes).statistic)
    kendall = float(kendalltau(scores, outcomes).statistic)
    return {
        "regret_pp": 100.0 * (best - selected_outcome),
        "exact_best": bool(
            np.any(np.isclose(outcomes[selected], best, rtol=0.0, atol=1.0e-12))
        ),
        "within_one_pp": bool(best - selected_outcome <= 0.01 + 1.0e-12),
        "spearman": spearman if np.isfinite(spearman) else None,
        "kendall": kendall if np.isfinite(kendall) else None,
        "selected_backbones": [screen.MODELS[int(index)] for index in selected],
    }


def _metric_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    selectors = {
        (str(row["model"]), str(row["arm"]), str(row["candidate_id"])): float(
            row["score"]
        )
        for row in raw["selector_rows"]
    }
    references = {
        (str(row["backbone"]), str(row["arm"]), str(row["head"])): float(
            row["test_accuracy"]
        )
        for row in raw["reference_rows"]
    }
    output = []
    for arm in screen.ARMS:
        for head in HEADS:
            outcomes = np.asarray(
                [references[(model, arm, head)] for model in screen.MODELS],
                dtype=np.float64,
            )
            for method in screen.METHODS:
                scores = np.asarray(
                    [selectors[(model, arm, method)] for model in screen.MODELS],
                    dtype=np.float64,
                )
                output.append(
                    {
                        "arm": arm,
                        "head": head,
                        "candidate_id": method,
                        **_selection_metrics(scores, outcomes),
                    }
                )
    return output


def _bootstrap_runtime(values: Sequence[float], *, key: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (10,) or not np.all(np.isfinite(array)):
        raise RuntimeError("runtime bootstrap requires ten finite backbone blocks")
    offset = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, offset]))
    draws = rng.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, array.size))
    distribution = np.median(array[draws], axis=1)
    return {
        "estimate": float(np.median(array)),
        "lower95": float(np.quantile(distribution, 0.025)),
        "upper95": float(np.quantile(distribution, 0.975)),
        "backbone_values": array.tolist(),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def _runtime_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["model"]), str(row["arm"]), str(row["candidate_id"])): float(
            row["total_wall_seconds"]
        )
        for row in raw["selector_rows"]
    }
    output = []
    for candidate in ("NOLLOYD", "LOOP3", "FUSED3"):
        for arm in screen.ARMS:
            ratios = [
                lookup[(model, arm, candidate)] / lookup[(model, arm, "LP-FULL")]
                for model in screen.MODELS
            ]
            output.append(
                {
                    "candidate_id": candidate,
                    "arm": arm,
                    "comparator": "LP-FULL",
                    "unit": "paired_backbone_wall_ratio",
                    **_bootstrap_runtime(ratios, key=f"runtime:{candidate}:{arm}"),
                }
            )
    return output


def _stage_summary(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for method in screen.METHODS:
        rows = [row for row in raw["selector_rows"] if row["candidate_id"] == method]
        output.append(
            {
                "candidate_id": method,
                "median_total_wall_seconds": float(
                    np.median([row["total_wall_seconds"] for row in rows])
                ),
                "median_fit_wall_seconds": float(
                    np.median([row["fit_wall_seconds"] for row in rows])
                ),
                "median_score_fixed_wall_seconds": float(
                    np.median([row["score_fixed_wall_seconds"] for row in rows])
                ),
            }
        )
    return output


def _metric_lookup(metrics: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(row["arm"]), str(row["head"]), str(row["candidate_id"])): row
        for row in metrics
    }


def _candidate_gates(
    candidate: str,
    metrics: Sequence[Mapping[str, Any]],
    runtime: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = _metric_lookup(metrics)
    output: list[dict[str, Any]] = []

    def regret_gate(gate: str, arm: str, head: str, comparator: str) -> None:
        candidate_value = float(lookup[(arm, head, candidate)]["regret_pp"])
        comparator_value = float(lookup[(arm, head, comparator)]["regret_pp"])
        delta = candidate_value - comparator_value
        output.append(
            {
                "candidate_id": candidate,
                "gate": gate,
                "arm": arm,
                "head": head,
                "comparator": comparator,
                "estimate": delta,
                "threshold": 1.0,
                "direction": "max",
                "status": "pass" if delta <= 1.0 else "fail",
                "required": True,
            }
        )

    regret_gate("clean_linear_vs_probe", "baseline", "linear", "LP-FULL")
    regret_gate("nuisance_linear_vs_probe", "nuisance_full", "linear", "LP-FULL")
    for head in NONLINEAR_HEADS:
        regret_gate("nonlinear_vs_probe", "nonlinearity_full", head, "LP-FULL")
        regret_gate("nonlinear_retention", "nonlinearity_full", head, "NOLLOYD")
        candidate_rho = lookup[("nonlinearity_full", head, candidate)]["spearman"]
        reference_rho = lookup[("nonlinearity_full", head, "NOLLOYD")]["spearman"]
        if candidate_rho is None or reference_rho is None:
            status, delta = "fail", None
        else:
            delta = float(candidate_rho) - float(reference_rho)
            status = "pass" if delta >= -0.10 else "fail"
        output.append(
            {
                "candidate_id": candidate,
                "gate": "nonlinear_rank_retention",
                "arm": "nonlinearity_full",
                "head": head,
                "comparator": "NOLLOYD",
                "estimate": delta,
                "threshold": -0.10,
                "direction": "min",
                "status": status,
                "required": True,
            }
        )
    for arm in screen.ARMS:
        row = next(
            item
            for item in runtime
            if item["candidate_id"] == candidate and item["arm"] == arm
        )
        output.append(
            {
                "candidate_id": candidate,
                "gate": "runtime_each_arm",
                "arm": arm,
                "head": None,
                "comparator": "LP-FULL",
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
    output.extend(
        [
            {
                "candidate_id": candidate,
                "gate": "exact_parity",
                "status": "pass",
                "required": True,
            },
            {
                "candidate_id": candidate,
                "gate": "determinism",
                "status": "pass",
                "required": True,
            },
        ]
    )
    return output


def analyze(raw: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metric_rows(raw)
    runtime = _runtime_rows(raw)
    gates = [
        row
        for candidate in screen.CANDIDATES
        for row in _candidate_gates(candidate, metrics, runtime)
    ]
    eligible = [
        candidate
        for candidate in screen.CANDIDATES
        if all(
            row["status"] == "pass"
            for row in gates
            if row["candidate_id"] == candidate and row.get("required") is True
        )
    ]
    pooled_wall = {
        candidate: float(
            np.median(
                [
                    row["total_wall_seconds"]
                    for row in raw["selector_rows"]
                    if row["candidate_id"] == candidate
                ]
            )
        )
        for candidate in screen.CANDIDATES
    }
    selected = None
    selection_reason = "no candidate passed every required gate"
    if len(eligible) == 1:
        selected = eligible[0]
        selection_reason = "only eligible candidate"
    elif len(eligible) == 2:
        if pooled_wall["LOOP3"] <= 0.95 * pooled_wall["FUSED3"]:
            selected = "LOOP3"
            selection_reason = "LOOP3 was at least five percent faster by pooled median"
        else:
            selected = "FUSED3"
            selection_reason = "FUSED3 retained the frozen preference within the five-percent tie band"
    decision = {
        "status": "pass_for_full_food_design" if selected is not None else "stopped_no_eligible",
        "selected_candidate": selected,
        "eligible_candidates": eligible,
        "selection_reason": selection_reason,
        "pooled_median_wall_seconds": pooled_wall,
        "full_replay_status": "not_run_requires_separate_frozen_protocol",
        "retrospective_development_only": True,
    }
    return {
        "schema_version": 1,
        "study": "fused_class_prototypes_food101_screen_analysis",
        "artifact_status": "completed",
        "selector_metrics": metrics,
        "runtime_contrasts": runtime,
        "stage_summary": _stage_summary(raw),
        "gates": gates,
        "decision": decision,
    }


def _report(summary: Mapping[str, Any]) -> str:
    decision = summary["decision"]
    lines = [
        "# Fused class-prototype Food-101 screen",
        "",
        "Retrospective development-only evidence. This screen does not authorize a product claim or untouched confirmation.",
        "",
        f"Decision: **{decision['status']}**.",
        f"Selected candidate: **{decision['selected_candidate'] or 'none'}**.",
        f"Reason: {decision['selection_reason']}.",
        "",
        "## Ranking",
        "",
        "| Arm | Head | Method | Regret pp | Spearman | Exact best | Within 1 pp |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    primary = {
        ("baseline", "linear"),
        ("nuisance_full", "linear"),
        *(("nonlinearity_full", head) for head in NONLINEAR_HEADS),
    }
    for row in summary["selector_metrics"]:
        if (row["arm"], row["head"]) not in primary:
            continue
        rho = "—" if row["spearman"] is None else f"{row['spearman']:.3f}"
        lines.append(
            f"| {row['arm']} | {row['head']} | {row['candidate_id']} | "
            f"{row['regret_pp']:.3f} | {rho} | {row['exact_best']} | {row['within_one_pp']} |"
        )
    lines.extend(
        [
            "",
            "## Runtime against LP-FULL",
            "",
            "| Arm | Candidate | Median ratio | Lower 95 | Upper 95 | Gate |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["runtime_contrasts"]:
        gate = next(
            (
                item["status"]
                for item in summary["gates"]
                if item["candidate_id"] == row["candidate_id"]
                and item["gate"] == "runtime_each_arm"
                and item["arm"] == row["arm"]
            ),
            "descriptive",
        )
        lines.append(
            f"| {row['arm']} | {row['candidate_id']} | {row['estimate']:.3f} | "
            f"{row['lower95']:.3f} | {row['upper95']:.3f} | {gate} |"
        )
    lines.extend(
        [
            "",
            "## Absolute stage timing",
            "",
            "| Method | Median total s | Median fit s | Median fixed-score s |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["stage_summary"]:
        lines.append(
            f"| {row['candidate_id']} | {row['median_total_wall_seconds']:.4f} | "
            f"{row['median_fit_wall_seconds']:.4f} | {row['median_score_fixed_wall_seconds']:.4f} |"
        )
    failed = [
        row
        for row in summary["gates"]
        if row.get("required") is True and row["status"] != "pass"
    ]
    lines.extend(["", "## Required gates", ""])
    if failed:
        for row in failed:
            lines.append(
                f"- **FAIL** {row['candidate_id']} / {row['gate']} / "
                f"{row.get('arm') or 'all'} / {row.get('head') or 'all'}"
            )
    else:
        lines.append("- All required gates passed for both fixed-Lloyd candidates.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The screen supports designing a separately frozen full Food replay for the selected candidate. It does not reopen candidate definitions or permit a backbone/dimension-specific hybrid."
                if decision["selected_candidate"] is not None
                else "Neither fixed-Lloyd candidate met the complete ranking and runtime contract, so this line should stop without a full replay."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite non-empty fused Food analysis output")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = verify_artifact(Path(input_dir))
    summary = analyze(raw)
    summary["input_raw_results_sha256"] = screen.sha256_path(
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
        "summary_sha256": screen.sha256_path(summary_path),
        "decision_sha256": screen.sha256_path(decision_path),
        "report_sha256": screen.sha256_path(report_path),
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
    print(screen.canonical_json(summary["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
