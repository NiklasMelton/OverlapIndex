"""Analysis and predeclared gates for the scalable Food-101 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from experiments.scalable_relevance_kmeans import food101_screen as screen


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_829
HEADS = ("linear", "quadratic", "knn", "rbf")
NONLINEAR_HEADS = ("quadratic", "knn", "rbf")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, screen.canonical_json(payload) + "\n")


def _verify_artifact(input_dir: Path) -> dict[str, Any]:
    manifest = _read_json(input_dir / "manifest.json")
    raw_path = input_dir / "raw_results.json"
    raw = _read_json(raw_path)
    if manifest.get("artifact_status") != "completed" or raw.get("artifact_status") != "completed":
        raise RuntimeError("Food screen artifact is not completed")
    if manifest.get("stage") != "development" or raw.get("stage") != "development":
        raise RuntimeError("analysis requires the complete development artifact, not smoke")
    if manifest.get("raw_results_sha256") != screen.sha256_path(raw_path):
        raise RuntimeError("Food screen raw-results hash mismatch")
    if manifest.get("run_identity") != raw.get("run_identity"):
        raise RuntimeError("Food screen manifest/raw identity mismatch")
    identity = raw.get("run_identity", {})
    if not isinstance(identity, Mapping):
        raise RuntimeError("Food screen run identity is malformed")
    identity_sha = hashlib.sha256(screen.canonical_json(identity).encode("utf-8")).hexdigest()
    if (
        manifest.get("run_identity_sha256") != identity_sha
        or raw.get("run_identity_sha256") != identity_sha
    ):
        raise RuntimeError("Food screen run-identity hash mismatch")
    if identity.get("sources") != screen._source_identity():
        raise RuntimeError("Food screen executed-source identity no longer matches")
    protocol_sha, _protocol = screen._validate_protocol()
    if identity.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Food screen protocol identity mismatch")
    expected_counts = {
        "panels": 600,
        "selector_rows": 2400,
        "parity_rows": 1800,
        "reference_rows": 600,
        "checkpoints": 600,
    }
    if manifest.get("counts") != expected_counts:
        raise RuntimeError("Food screen terminal counts are incomplete")
    rows = raw.get("selector_rows")
    references = raw.get("reference_rows")
    parity = raw.get("parity_rows")
    if not isinstance(rows, list) or len(rows) != 2400:
        raise RuntimeError("Food screen selector rows are incomplete")
    if not isinstance(references, list) or len(references) != 600:
        raise RuntimeError("Food screen reference rows are incomplete")
    if not isinstance(parity, list) or len(parity) != 1800:
        raise RuntimeError("Food screen parity rows are incomplete")
    expected_keys = {
        (model, replicate, arm, budget, method)
        for model in screen.MODELS
        for replicate in screen.REPLICATES
        for arm in screen.ARMS
        for budget in screen.BUDGETS
        for method in screen.METHODS
    }
    observed_keys: set[tuple[str, int, str, int, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            raise RuntimeError("Food screen contains a malformed/non-ok selector row")
        key = (
            str(row.get("model")),
            int(row.get("replicate", -1)),
            str(row.get("arm")),
            int(row.get("budget", -1)),
            str(row.get("candidate_id")),
        )
        if key in observed_keys:
            raise RuntimeError("Food screen duplicates a selector row")
        observed_keys.add(key)
        for field in ("score", "total_wall_seconds", "total_cpu_seconds"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise RuntimeError(f"Food screen row has invalid {field}")
        if float(row["total_wall_seconds"]) < 0.0:
            raise RuntimeError("Food screen row has negative wall time")
    if observed_keys != expected_keys:
        raise RuntimeError("Food screen selector identities do not match the frozen grid")
    if any(row.get("exact") is not True or float(row.get("delta", np.nan)) != 0.0 for row in parity):
        raise RuntimeError("Food screen archived parity is not exact")
    determinism = raw.get("determinism")
    if not isinstance(determinism, Mapping) or set(determinism) != set(screen.METHODS):
        raise RuntimeError("Food screen determinism evidence is incomplete")
    for method, record in determinism.items():
        if (
            not isinstance(record, Mapping)
            or record.get("candidate_id") != method
            or record.get("exact") is not True
            or record.get("runtime_fields_excluded") is not True
            or record.get("first_signature_sha256") != record.get("second_signature_sha256")
        ):
            raise RuntimeError("Food screen determinism record is malformed")
    return raw


def _selection_metrics(scores: np.ndarray, outcomes: np.ndarray) -> dict[str, Any]:
    if scores.shape != (10,) or outcomes.shape != (10,):
        raise RuntimeError("selection metrics require one complete 10-backbone panel")
    selected = np.flatnonzero(
        np.isclose(scores, float(np.max(scores)), rtol=0.0, atol=1.0e-12)
    )
    best = float(np.max(outcomes))
    selected_outcome = float(np.mean(outcomes[selected]))
    spearman = float(spearmanr(scores, outcomes).statistic)
    kendall = float(kendalltau(scores, outcomes).statistic)
    return {
        "regret": best - selected_outcome,
        "regret_pp": 100.0 * (best - selected_outcome),
        "exact_best": bool(np.any(np.isclose(outcomes[selected], best, rtol=0.0, atol=1e-12))),
        "within_one_pp": bool(best - selected_outcome <= 0.01 + 1e-12),
        "spearman": spearman if np.isfinite(spearman) else None,
        "kendall": kendall if np.isfinite(kendall) else None,
        "selected_backbones": [screen.MODELS[int(index)] for index in selected],
    }


def _metric_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    selector = {
        (
            str(row["model"]),
            int(row["replicate"]),
            str(row["arm"]),
            int(row["budget"]),
            str(row["candidate_id"]),
        ): float(row["score"])
        for row in raw["selector_rows"]
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
    output: list[dict[str, Any]] = []
    for replicate in screen.REPLICATES:
        for arm in screen.ARMS:
            for budget in screen.BUDGETS:
                for head in HEADS:
                    outcomes = np.asarray(
                        [references[(model, replicate, arm, head)] for model in screen.MODELS]
                    )
                    for method in screen.METHODS:
                        scores = np.asarray(
                            [
                                selector[(model, replicate, arm, budget, method)]
                                for model in screen.MODELS
                            ]
                        )
                        output.append(
                            {
                                "replicate": int(replicate),
                                "arm": arm,
                                "budget": int(budget),
                                "head": head,
                                "candidate_id": method,
                                **_selection_metrics(scores, outcomes),
                            }
                        )
    return output


def _rank_auc_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    xx = np.asarray(screen.BUDGETS, dtype=np.float64)
    width = float(xx[-1] - xx[0])
    for replicate in screen.REPLICATES:
        for arm in screen.ARMS:
            for head in HEADS:
                for method in screen.METHODS:
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
                    if len(rows) != len(screen.BUDGETS) or any(row["spearman"] is None for row in rows):
                        auc = None
                    else:
                        yy = np.asarray([float(row["spearman"]) for row in rows])
                        auc = float(np.sum((yy[:-1] + yy[1:]) * np.diff(xx) * 0.5) / width)
                    output.append(
                        {
                            "replicate": int(replicate),
                            "arm": arm,
                            "head": head,
                            "candidate_id": method,
                            "rank_auc": auc,
                        }
                    )
    return output


def _bootstrap(values: Sequence[float], *, key: str, statistic: str = "mean") -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (5,) or not np.all(np.isfinite(array)):
        raise RuntimeError(f"bootstrap {key} requires five finite replicate blocks")
    seed_offset = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, seed_offset]))
    draws = rng.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, array.size))
    sampled = array[draws]
    if statistic == "median":
        distribution = np.median(sampled, axis=1)
        estimate = float(np.median(array))
    elif statistic == "mean":
        distribution = np.mean(sampled, axis=1)
        estimate = float(np.mean(array))
    else:
        raise ValueError("unknown bootstrap statistic")
    return {
        "estimate": estimate,
        "lower95": float(np.quantile(distribution, 0.025)),
        "upper95": float(np.quantile(distribution, 0.975)),
        "replicate_values": array.tolist(),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def _regret_contrast(
    metrics: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    head: str,
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    lookup = {
        (int(row["replicate"]), int(row["budget"]), str(row["candidate_id"])): float(
            row["regret_pp"]
        )
        for row in metrics
        if row["arm"] == arm and row["head"] == head
    }
    values = []
    for replicate in screen.REPLICATES:
        differences = [
            lookup[(replicate, budget, candidate)] - lookup[(replicate, budget, comparator)]
            for budget in screen.BUDGETS
        ]
        values.append(float(np.mean(differences)))
    return {
        "arm": arm,
        "head": head,
        "candidate_id": candidate,
        "comparator": comparator,
        "unit": "accuracy_points",
        **_bootstrap(values, key=f"regret:{arm}:{head}:{candidate}:{comparator}"),
    }


def _runtime_contrast(
    raw: Mapping[str, Any], *, arm: str, comparator: str
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
    replicate_values: list[float] = []
    for replicate in screen.REPLICATES:
        ratios = [
            lookup[(model, replicate, arm, budget, "S")]
            / lookup[(model, replicate, arm, budget, comparator)]
            for model in screen.MODELS
            for budget in screen.BUDGETS
        ]
        replicate_values.append(float(np.median(ratios)))
    return {
        "arm": arm,
        "candidate_id": "S",
        "comparator": comparator,
        "unit": "wall_ratio",
        **_bootstrap(
            replicate_values,
            key=f"runtime:{arm}:S:{comparator}",
            statistic="median",
        ),
    }


def _aggregate_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in screen.ARMS:
        for head in HEADS:
            for method in screen.METHODS:
                rows = [
                    row
                    for row in metrics
                    if row["arm"] == arm and row["head"] == head and row["candidate_id"] == method
                ]
                spearman = [float(row["spearman"]) for row in rows if row["spearman"] is not None]
                output.append(
                    {
                        "arm": arm,
                        "head": head,
                        "candidate_id": method,
                        "mean_regret_pp": float(np.mean([row["regret_pp"] for row in rows])),
                        "mean_spearman": float(np.mean(spearman)) if spearman else None,
                        "exact_best_rate": float(np.mean([row["exact_best"] for row in rows])),
                        "within_one_pp_rate": float(np.mean([row["within_one_pp"] for row in rows])),
                    }
                )
    return output


def analyze(raw: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metric_rows(raw)
    auc_rows = _rank_auc_rows(metrics)
    contrasts: list[dict[str, Any]] = []
    contrasts.append(
        _regret_contrast(
            metrics, arm="baseline", head="linear", candidate="S", comparator="LP-FULL"
        )
    )
    contrasts.append(
        _regret_contrast(
            metrics, arm="nuisance_full", head="linear", candidate="S", comparator="LP-FULL"
        )
    )
    for head in NONLINEAR_HEADS:
        contrasts.append(
            _regret_contrast(
                metrics,
                arm="nonlinearity_full",
                head=head,
                candidate="S",
                comparator="LP-FULL",
            )
        )
        contrasts.append(
            _regret_contrast(
                metrics,
                arm="nonlinearity_full",
                head=head,
                candidate="S",
                comparator="B",
            )
        )
    runtime = [
        _runtime_contrast(raw, arm=arm, comparator=comparator)
        for arm in screen.ARMS
        for comparator in ("LP-FULL", "B")
    ]
    gate_rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        comparator = str(contrast["comparator"])
        if contrast["arm"] == "baseline" and contrast["head"] == "linear":
            gate = "clean_linear_vs_probe"
            required = True
        elif contrast["arm"] == "nuisance_full" and contrast["head"] == "linear":
            gate = "nuisance_linear_vs_probe"
            required = True
        elif comparator == "B":
            gate = "nonlinear_retention"
            required = True
        else:
            gate = "nonlinear_vs_probe"
            required = True
        gate_rows.append(
            {
                "gate": gate,
                "arm": contrast["arm"],
                "head": contrast["head"],
                "comparator": comparator,
                "threshold_upper95_pp": 1.0,
                "estimate_pp": contrast["estimate"],
                "upper95_pp": contrast["upper95"],
                "status": "pass" if float(contrast["upper95"]) <= 1.0 else "fail",
                "required": required,
            }
        )
        if contrast["arm"] == "nonlinearity_full" and comparator == "LP-FULL":
            gate_rows.append(
                {
                    "gate": "nonlinear_superiority_claim",
                    "arm": contrast["arm"],
                    "head": contrast["head"],
                    "comparator": comparator,
                    "threshold_upper95_pp": 0.0,
                    "estimate_pp": contrast["estimate"],
                    "upper95_pp": contrast["upper95"],
                    "status": "established" if float(contrast["upper95"]) < 0.0 else "not_established",
                    "required": False,
                }
            )
    for row in runtime:
        if row["comparator"] != "LP-FULL":
            continue
        gate_rows.append(
            {
                "gate": "runtime",
                "arm": row["arm"],
                "head": None,
                "comparator": "LP-FULL",
                "threshold_median": 1.0,
                "threshold_upper95": 1.10,
                "estimate": row["estimate"],
                "upper95": row["upper95"],
                "status": (
                    "pass"
                    if float(row["estimate"]) <= 1.0 and float(row["upper95"]) <= 1.10
                    else "fail"
                ),
                "required": True,
            }
        )
    gate_rows.extend(
        [
            {
                "gate": "parity",
                "status": "pass",
                "required": True,
                "exact_rows": len(raw["parity_rows"]),
            },
            {
                "gate": "determinism",
                "status": "pass",
                "required": True,
                "exact_methods": sorted(raw["determinism"]),
            },
        ]
    )
    required = [row for row in gate_rows if row.get("required") is True]
    decision = {
        "candidate_id": "S",
        "status": "pass" if all(row["status"] == "pass" for row in required) else "fail",
        "required_gate_count": len(required),
        "failed_required_gates": [
            {
                "gate": row["gate"],
                "arm": row.get("arm"),
                "head": row.get("head"),
            }
            for row in required
            if row["status"] != "pass"
        ],
        "retrospective_development_only": True,
        "confirmation_status": "not_run_not_untouched",
    }
    return {
        "schema_version": 1,
        "study": "scalable_one_update_food101_screen_analysis",
        "artifact_status": "completed",
        "selector_metrics": metrics,
        "rank_auc_rows": auc_rows,
        "aggregate": _aggregate_rows(metrics),
        "regret_contrasts": contrasts,
        "runtime_contrasts": runtime,
        "gates": gate_rows,
        "decision": decision,
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Scalable one-update Food-101 screen",
        "",
        "Retrospective development-only evidence; no untouched confirmation or public-product claim.",
        "",
        f"Decision: **{str(summary['decision']['status']).upper()}** for the frozen development gates.",
        "",
        "## Primary selection outcomes",
        "",
        "| Arm | Head | Method | Mean regret pp | Spearman | Exact best | Within 1 pp |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    primary = {
        ("baseline", "linear"),
        ("nuisance_full", "linear"),
        *(('nonlinearity_full', head) for head in NONLINEAR_HEADS),
    }
    for row in summary["aggregate"]:
        if (row["arm"], row["head"]) not in primary:
            continue
        corr = "—" if row["mean_spearman"] is None else f"{row['mean_spearman']:.3f}"
        lines.append(
            f"| {row['arm']} | {row['head']} | {row['candidate_id']} | "
            f"{row['mean_regret_pp']:.3f} | {corr} | {row['exact_best_rate']:.2f} | "
            f"{row['within_one_pp_rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired regret gates (S minus comparator)",
            "",
            "| Arm | Head | Comparator | Estimate pp | Upper 95 pp | Status |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in summary["regret_contrasts"]:
        status_rows = [
            gate
            for gate in summary["gates"]
            if gate.get("arm") == row["arm"]
            and gate.get("head") == row["head"]
            and gate.get("comparator") == row["comparator"]
            and gate.get("gate") != "nonlinear_superiority_claim"
        ]
        status = status_rows[0]["status"] if status_rows else "descriptive"
        lines.append(
            f"| {row['arm']} | {row['head']} | {row['comparator']} | "
            f"{row['estimate']:.3f} | {row['upper95']:.3f} | {status} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| Arm | Comparator | Median S/comparator | Upper 95 |",
            "|---|---|---:|---:|",
        ]
    )
    for row in summary["runtime_contrasts"]:
        lines.append(
            f"| {row['arm']} | {row['comparator']} | {row['estimate']:.3f} | {row['upper95']:.3f} |"
        )
    failed = summary["decision"]["failed_required_gates"]
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            (
                "All predeclared development gates passed. A future untouched confirmation would still be required."
                if not failed
                else "The scalable method did not pass every predeclared development gate; see the failed gate list in summary.json."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite non-empty Food analysis output")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _verify_artifact(Path(input_dir))
    summary = analyze(raw)
    summary["input_raw_results_sha256"] = screen.sha256_path(Path(input_dir) / "raw_results.json")
    summary["protocol_sha256"] = raw["run_identity"]["protocol_sha256"]
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    _atomic_json(summary_path, summary)
    _atomic_text(report_path, _report(summary))
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": summary["study"],
        "input_raw_results_sha256": summary["input_raw_results_sha256"],
        "protocol_sha256": summary["protocol_sha256"],
        "summary_sha256": screen.sha256_path(summary_path),
        "report_sha256": screen.sha256_path(report_path),
        "decision": summary["decision"],
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = write_analysis(args.input, args.output)
    print(screen.canonical_json(summary["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
