"""Source-bound post-outcome analysis for the matched-geometry diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from . import run


def ranking_metrics(scores: Mapping[str, float], truth: Mapping[str, float]) -> dict[str, Any]:
    if set(scores) != set(run.CANDIDATES) or set(truth) != set(run.CANDIDATES):
        raise ValueError("ranking vectors must contain the frozen candidate set")
    score = np.asarray([scores[name] for name in run.CANDIDATES], dtype=np.float64)
    target = np.asarray([truth[name] for name in run.CANDIDATES], dtype=np.float64)
    if not np.isfinite(score).all() or not np.isfinite(target).all():
        raise ValueError("ranking vectors must be finite")
    selected = np.flatnonzero(np.isclose(score, score.max(), rtol=0.0, atol=1e-12))
    correlation = spearmanr(score, target).statistic
    regret = float(target.max() - np.max(target[selected]))
    return {
        "spearman": None if not np.isfinite(correlation) else float(correlation),
        "top1_regret_r2": regret,
        "exact_best": bool(np.any(np.isclose(target[selected], target.max(), rtol=0.0, atol=1e-12))),
        "within_point01": bool(regret <= 0.01 + 1e-12),
        "selected_candidate_ids": [run.CANDIDATES[int(position)] for position in selected],
    }


def analyze(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("artifact_status") != "completed" or len(raw.get("panels", [])) != 12:
        raise ValueError("matched-geometry raw artifact is incomplete")
    panel_rows = []
    for panel in raw["panels"]:
        if not panel["reference_rows"]:
            continue
        envelope = {row["candidate_id"]: float(row["envelope_r2"]) for row in panel["reference_rows"]}
        linear_cv = {
            row["candidate_id"]: float(row["heads"]["linear_ridge"]["inner_best_r2"])
            for row in panel["reference_rows"]
        }
        score_maps: dict[str, dict[str, float]] = {"linear_cv": linear_cv}
        for method in run.METHODS:
            rows = [row for row in panel["selector_rows"] if row["method_id"] == method]
            score_maps[method] = {
                candidate: float(np.mean([row["score"] for row in rows if row["candidate_id"] == candidate]))
                for candidate in run.CANDIDATES
            }
        for method, scores in score_maps.items():
            panel_rows.append({
                "dimension": int(panel["dimension"]),
                "train_rows": int(panel["train_rows"]),
                "training_seed": int(panel["training_seed"]),
                "method_id": method,
                **ranking_metrics(scores, envelope),
            })
    aggregate = []
    for method in ("linear_cv", *run.METHODS):
        rows = [row for row in panel_rows if row["method_id"] == method]
        aggregate.append({
            "method_id": method,
            "mean_top1_regret_r2": float(np.mean([row["top1_regret_r2"] for row in rows])),
            "mean_spearman": float(np.mean([row["spearman"] for row in rows if row["spearman"] is not None])),
            "exact_best_rate": float(np.mean([row["exact_best"] for row in rows])),
            "within_point01_rate": float(np.mean([row["within_point01"] for row in rows])),
        })
    sampling_rows = []
    for dimension in (16, 512):
        for train_rows in (300, 1200):
            panels = [row for row in raw["panels"] if row["dimension"] == dimension and row["train_rows"] == train_rows]
            for method in run.METHODS:
                for candidate in run.CANDIDATES:
                    values = []
                    for panel in panels:
                        rows = [row for row in panel["selector_rows"] if row["method_id"] == method and row["candidate_id"] == candidate]
                        values.append(float(np.mean([row["score"] for row in rows])))
                    sampling_rows.append({"dimension": dimension, "train_rows": train_rows,
                                          "method_id": method, "candidate_id": candidate,
                                          "score_sd": float(np.std(values, ddof=0))})
    return {"schema_version": 1, "head_envelope_panel_rows": panel_rows,
            "head_envelope_aggregate": aggregate, "sampling_variance": sampling_rows}


def write_analysis(results: Path) -> dict[str, Any]:
    results = results.resolve()
    raw_path = results / "raw_results.json"
    terminal_path = results / "manifest.json"
    output = results / "analysis.json"
    manifest_path = results / "analysis_manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite matched-geometry analysis")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("raw_results_sha256") != run._sha256(raw_path):
        raise ValueError("raw artifact does not match terminal manifest")
    result = analyze(json.loads(raw_path.read_text(encoding="utf-8")))
    run._atomic_json(output, result)
    manifest = {"schema_version": 1, "artifact_status": "completed",
                "raw_results_sha256": run._sha256(raw_path),
                "terminal_manifest_sha256": run._sha256(terminal_path),
                "analyzer_sha256": run._sha256(Path(__file__)),
                "analysis_sha256": run._sha256(output)}
    run._atomic_json(manifest_path, manifest)
    return {"analysis": result, "manifest": manifest}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=run.DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(write_analysis(args.results)["analysis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
