"""Frozen synthetic regression check for the scalable one-update selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.diagonal_metric_toy import experiment as base
from experiments.scalable_relevance_kmeans.scalable import (
    ScalableOneUpdateRelevanceKMeans,
)


SCENARIO = "nonlinear_nuisance"
BUDGET_PER_CLASS = 256
SEED = 43
METHODS = ("raw_unrefined", "raw_refined", "scalable_one_update", "linear_probe")
PROTOCOL_PATH = Path(__file__).resolve().parent / "protocol.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _scalable_crossfit(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, dict[str, Any]]:
    scores: list[float] = []
    conditions: list[float] = []
    tile_bytes: list[int] = []
    for fold, (train, heldout) in enumerate(base._folds(y, seed)):
        selector = ScalableOneUpdateRelevanceKMeans(
            k_per_class=base.K_PER_CLASS,
            kmeans_kwargs={
                "batch_size": 256,
                "max_no_improvement": 5,
                "compute_labels": False,
                "n_init": 1,
                "init": "random",
                "random_state": int(seed + fold),
            },
            memory_budget_mb=64,
        ).fit(X[train], y[train])
        score, score_diagnostics = selector.score_fixed_with_diagnostics(
            X[heldout], y[heldout]
        )
        scores.append(float(score))
        diagnostics = dict(selector.diagnostics_ or {})
        conditions.append(float(diagnostics["weight_condition"]))
        tile_bytes.extend(
            [
                int(diagnostics["max_score_tile_bytes"]),
                int(score_diagnostics["max_score_tile_bytes"]),
            ]
        )
    return float(np.mean(scores)), {
        "weight_condition_mean": float(np.mean(conditions)),
        "max_score_tile_bytes": int(max(tile_bytes)),
    }


def run_check() -> dict[str, Any]:
    protocol_raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_raw)
    frozen = protocol["synthetic_regression"]
    expected = {
        "scenario": SCENARIO,
        "budget_per_class": BUDGET_PER_CLASS,
        "seed": SEED,
        "candidate_ids": [row["candidate_id"] for row in base.PANEL_CANDIDATES],
        "methods": list(METHODS),
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"synthetic protocol mismatch for {key}")
    references = {
        str(candidate["candidate_id"]): base._reference_accuracy(
            scenario=SCENARIO, seed=SEED, candidate=candidate
        )
        for candidate in base.PANEL_CANDIDATES
    }
    rows: list[dict[str, Any]] = []
    for candidate in base.PANEL_CANDIDATES:
        candidate_id = str(candidate["candidate_id"])
        X, y = base.generate_embedding(
            scenario=SCENARIO,
            seed=SEED,
            split="selector",
            n_per_class=BUDGET_PER_CLASS,
            candidate=candidate,
        )
        for method in METHODS:
            started = time.perf_counter()
            if method == "scalable_one_update":
                score, diagnostics = _scalable_crossfit(X, y, SEED)
            else:
                score, diagnostics = base._crossfit_selector(X, y, method=method, seed=SEED)
            rows.append(
                {
                    "scenario": SCENARIO,
                    "budget_per_class": BUDGET_PER_CLASS,
                    "seed": SEED,
                    "candidate_id": candidate_id,
                    "method": method,
                    "selector_score": float(score),
                    "reference_head": "knn",
                    "reference_accuracy": float(references[candidate_id]),
                    "wall_seconds": float(time.perf_counter() - started),
                    "diagnostics": diagnostics,
                }
            )
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        selected = {
            str(row["candidate_id"]): float(row["selector_score"])
            for row in rows
            if row["method"] == method
        }
        metrics = base._selection_metrics(selected, references)
        summary.append({"method": method, **metrics})
    return {
        "schema_version": 1,
        "study": "scalable_one_update_synthetic_regression",
        "artifact_status": "completed",
        "development_only": True,
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "rows": rows,
        "summary": summary,
    }


def _report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Scalable one-update synthetic regression",
        "",
        "Development-only frozen regression panel; not promotion or confirmation evidence.",
        "",
        "| Method | Regret pp | Spearman | Within 1 pp | Selected |",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload["summary"]:
        correlation = "—" if row["spearman"] is None else f"{row['spearman']:.3f}"
        lines.append(
            f"| {row['method']} | {row['regret_pp']:.3f} | {correlation} | "
            f"{str(row['within_one_pp']).lower()} | {', '.join(row['selected_candidates'])} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    payload = run_check()
    raw = args.output / ".raw_results.json.tmp"
    report = args.output / ".report.md.tmp"
    raw.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    report.write_text(_report(payload), encoding="utf-8")
    os.replace(raw, args.output / "raw_results.json")
    os.replace(report, args.output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
