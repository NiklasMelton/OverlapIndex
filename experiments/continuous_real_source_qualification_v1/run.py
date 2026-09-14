"""Run the frozen real-source reference-only qualification screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn

from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_reference_historical_screen import run as historical
from experiments.continuous_reference_library_qualification import run as q1
from experiments.continuous_reference_library_qualification_v3 import run as q3
from experiments.continuous_real_source_qualification_v1 import extract, prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_source_qualification_v1" / "screen"
Q3_ROOT = ROOT / "artifacts" / "continuous_reference_library_qualification_v3" / "pilot"
BACKBONES = q1.BACKBONES
HEADS = q1.HEADS
TIE_ATOL = q1.TIE_ATOL


def load_protocol() -> dict[str, Any]:
    protocol = prepare.load_protocol()
    if tuple(protocol["backbones"]) != BACKBONES:
        raise ValueError("real-source scout backbone roster changed")
    if tuple(protocol["reference_library"]["families"]) != HEADS:
        raise ValueError("real-source reference head roster changed")
    if len({row["target_key"] for row in protocol["targets"]}) != len(protocol["targets"]):
        raise ValueError("real-source target keys repeat")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "protocol": prepare.PROTOCOL_PATH,
        "protocol_sidecar": prepare.PROTOCOL_SHA256_PATH,
        "prepare": Path(prepare.__file__).resolve(),
        "extract": Path(extract.__file__).resolve(),
        "runner": Path(__file__).resolve(),
        "qualified_evaluator": q1.PACKAGE_DIR / "evaluator.py",
        "qualification_v3_protocol": q3.PROTOCOL_PATH,
        "qualification_v3_raw_results": Q3_ROOT / "raw_results.json",
        "qualification_v3_manifest": Q3_ROOT / "manifest.json",
        "input_cohort": prepare.DEFAULT_PREPARED_ROOT / "cohort.json",
        "input_manifest": prepare.DEFAULT_PREPARED_ROOT / "manifest.json",
        "embedding_manifest": extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
    }


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"real-source provenance input is missing: {missing}")
    observed = {name: base.sha256_path(path) for name, path in paths.items()}
    ancestry = protocol["ancestry"]
    expected = {
        "qualification_v3_protocol": ancestry["qualification_v3_protocol_sha256"],
        "qualification_v3_raw_results": ancestry["qualification_v3_raw_results_sha256"],
        "qualification_v3_manifest": ancestry["qualification_v3_manifest_sha256"],
    }
    if any(observed[key] != value for key, value in expected.items()):
        raise ValueError("real-source qualified-library ancestry changed")
    terminal = q3.validate_artifacts(Q3_ROOT)
    if not terminal["decision"]["qualification_pass"]:
        raise ValueError("real-source screen requires a passing reference-library qualification")
    return observed


def _target_draws(cohort: Mapping[str, Any], config: Mapping[str, Any]):
    dataset_id, target_id = str(config["target_key"]).split("::", 1)
    rows = cohort["datasets"][dataset_id]["target_cohorts"][target_id]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    if len(evaluation) != int(config["evaluation_rows"]):
        raise ValueError(f"real-source evaluation count changed: {config['target_key']}")
    evaluation_ids = {str(row["sample_id"]) for row in evaluation}
    if len(evaluation_ids) != len(evaluation):
        raise ValueError(f"real-source evaluation identities repeat: {config['target_key']}")
    draws = [
        base.select_unique_draw(
            rows, pool_rows=int(config["training_pool_rows"]),
            draw_rows=int(config["training_rows_per_draw"]), seed=int(seed),
        )
        for seed in config["draw_seeds"]
    ]
    if any(evaluation_ids.intersection(str(row["sample_id"]) for row in draw) for draw in draws):
        raise ValueError(f"real-source training/evaluation overlap: {config['target_key']}")
    return dataset_id, evaluation, draws


def build_panel(
    screen_protocol: Mapping[str, Any], library_protocol: Mapping[str, Any],
    config: Mapping[str, Any], cohort: Mapping[str, Any],
    matrices: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    target_key = str(config["target_key"])
    dataset_id, evaluation, draws = _target_draws(cohort, config)
    y_evaluation = np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64)
    evaluation_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    candidates = []
    for backbone in BACKBONES:
        matrix = matrices[dataset_id][backbone]
        candidate_draws = []
        for draw_position, (draw_seed, draw) in enumerate(zip(config["draw_seeds"], draws)):
            train_indices = np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)
            y_train = np.asarray([float(row["target"]) for row in draw], dtype=np.float64)
            result = q1.evaluate_heads(
                matrix[train_indices], y_train,
                np.asarray([int(row["target_bin"]) for row in draw], dtype=np.int64),
                [str(row["sample_id"]) for row in draw], matrix[evaluation_indices], y_evaluation,
                split_seed=base._seed("real-source-screen", target_key, backbone, draw_seed),
                protocol=library_protocol,
            )
            candidate_draws.append({
                "draw": draw_position, "draw_seed": int(draw_seed),
                "sample_ids": [str(row["sample_id"]) for row in draw],
                "sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in draw]),
                "unique_training_rows": len({str(row["sample_id"]) for row in draw}), **result,
            })
        head_means = {
            head: float(np.mean([draw["heads"][head]["evaluation_r2"] for draw in candidate_draws]))
            for head in HEADS
        }
        envelope = max(head_means.values())
        candidates.append({
            "candidate_id": backbone, "draws": candidate_draws,
            "head_mean_evaluation_r2": head_means, "envelope_r2": float(envelope),
            "best_head_ids": [head for head in HEADS if head_means[head] >= envelope - TIE_ATOL],
        })
    panel = {
        "target_key": target_key, "dataset_id": dataset_id,
        "training_pool_rows": int(config["training_pool_rows"]),
        "training_rows_per_draw": int(config["training_rows_per_draw"]),
        "draw_count": len(draws), "evaluation_rows": len(evaluation),
        "evaluation_sample_ids": [str(row["sample_id"]) for row in evaluation],
        "evaluation_target": y_evaluation.tolist(), "candidates": candidates,
    }
    panel["screen"] = historical.screen_panel(panel, screen_protocol)
    return panel


def summarize(protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        {"target_key": panel["target_key"], "dataset_id": panel["dataset_id"], "draw_count": panel["draw_count"], **panel["screen"]}
        for panel in panels
    ]
    nonlinear = [row for row in rows if row["classification"] == "nonlinear_challenge_lead"]
    sources = {row["dataset_id"] for row in nonlinear}
    sufficient = len(nonlinear) >= 3 and len(sources) >= 3
    return {
        "schema_version": 1, "status": "completed_continuous_real_source_qualification_v1",
        "target_rows": rows,
        "classification_counts": {label: sum(row["classification"] == label for row in rows) for label in protocol["classification"]["labels"]},
        "decision": {
            "sufficient_real_nonlinear_diversity": sufficient,
            "nonlinear_challenge_lead_count": len(nonlinear),
            "nonlinear_challenge_source_count": len(sources),
            "next_stage": "freeze_sources_for_oi_development" if sufficient else "expand_real_sources_before_oi",
            "method_promotion": "none_reference_only",
        },
        "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous real-source qualification V1", "",
        "| Target | Class | Envelope best | Linear best | Linear rho | Point regret | Conservative regret | Nonlinear gain | Top-1 probability |",
        "|---|---|---|---|---:|---:|---|---:|---:|",
    ]
    for row in summary["target_rows"]:
        bounds = row["linear_top1_regret_conservative_bounds_r2"]
        lines.append(
            f"| {row['target_key']} | {row['classification']} | {', '.join(row['envelope_best_candidate_ids'])} | "
            f"{', '.join(row['linear_best_candidate_ids'])} | {row['linear_to_envelope_spearman']:.4f} | "
            f"{row['linear_top1_regret_point_r2']:.4f} | [{bounds[0]:.4f}, {bounds[1]:.4f}] | "
            f"{row['maximum_nonlinear_gain_point_r2']:.4f} | {row['winner_top1_probability']:.4f} |"
        )
    decision = summary["decision"]
    lines.extend([
        "", "## Decision", "",
        f"- Classification counts: `{json.dumps(summary['classification_counts'], sort_keys=True)}`.",
        f"- Nonlinear challenge leads: **{decision['nonlinear_challenge_lead_count']}** from **{decision['nonlinear_challenge_source_count']}** sources.",
        f"- Real nonlinear diversity sufficient: **{str(decision['sufficient_real_nonlinear_diversity']).lower()}**.",
        f"- Next stage: **{decision['next_stage']}**.",
        "- This reference-only screen promotes no OverlapIndex method.", "",
    ])
    return "\n".join(lines)


def _identity(protocol: Mapping[str, Any], hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "protocol_sha256": base.sha256_path(prepare.PROTOCOL_PATH),
        "qualified_library_protocol_sha256": base.sha256_path(q3.PROTOCOL_PATH),
        "source_hashes": dict(hashes), "target_keys": [row["target_key"] for row in protocol["targets"]],
    }


def validate_grid(protocol: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    expected, panels = protocol["expected_grid"], checkpoint["panels"]
    counts = (
        len(panels),
        sum(len(panel["candidates"]) for panel in panels),
        sum(len(candidate["draws"]) for panel in panels for candidate in panel["candidates"]),
    )
    wanted = (int(expected["target_count"]), int(expected["candidate_count"]), int(expected["candidate_draw_count"]))
    if counts != wanted:
        raise ValueError("real-source screen executed grid changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    output_root = output_root.resolve()
    protocol, library = load_protocol(), q3.load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("real-source outcome execution requires a clean committed source tree")
    hashes = _source_hashes(protocol)
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("real-source screen terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    checkpoint_path, identity = output_root / "checkpoint.json", _identity(protocol, hashes)
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError("real-source screen checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("real-source screen checkpoint identity changed")
    else:
        checkpoint = {"schema_version": 1, "identity": identity, "panels": []}
    output_root.mkdir(parents=True, exist_ok=True)
    cohort, matrices = extract.load_bank()
    complete = {row["target_key"] for row in checkpoint["panels"]}
    for config in protocol["targets"]:
        if config["target_key"] not in complete:
            checkpoint["panels"].append(build_panel(protocol, library, config, cohort, matrices))
            base._finite_json(checkpoint)
            base._atomic_json(checkpoint_path, checkpoint)
            print(f"target {len(checkpoint['panels'])}/{len(protocol['targets'])}: {config['target_key']}", flush=True)
    validate_grid(protocol, checkpoint)
    summary = summarize(protocol, checkpoint["panels"])
    raw = {
        "schema_version": 1, "status": "completed_continuous_real_source_qualification_v1",
        "protocol": protocol, "qualified_library_protocol": library,
        "protocol_sha256": base.sha256_path(prepare.PROTOCOL_PATH), "source_hashes": hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "thread_environment": {key: os.environ.get(key) for key in base.THREAD_KEYS}},
        "panels": checkpoint["panels"], "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }
    base._finite_json(raw); base._finite_json(summary)
    raw_path, summary_path, report_path = output_root / "raw_results.json", output_root / "summary.json", output_root / "report.md"
    base._atomic_json(raw_path, raw); base._atomic_json(summary_path, summary); base._atomic_bytes(report_path, render_report(summary).encode("utf-8"))
    manifest = {
        "schema_version": 1, "status": "completed_continuous_real_source_qualification_v1",
        "protocol_sha256": base.sha256_path(prepare.PROTOCOL_PATH),
        "qualified_library_protocol_sha256": base.sha256_path(q3.PROTOCOL_PATH),
        "source_hashes": hashes, "checkpoint_identity": identity,
        "artifacts": {name: base.sha256_path(output_root / name) for name in ("checkpoint.json", "raw_results.json", "summary.json", "report.md")},
        "counts": {"targets": len(checkpoint["panels"]), "candidates": sum(len(panel["candidates"]) for panel in checkpoint["panels"]), "candidate_draws": sum(len(candidate["draws"]) for panel in checkpoint["panels"] for candidate in panel["candidates"]), "overlapindex_fits": 0, "reserved_backbones": 0},
        "git": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "dirty": False},
        "decision": summary["decision"],
    }
    base._atomic_json(output_root / "manifest.json", manifest)
    terminal = {"schema_version": 1, "status": "completed_continuous_real_source_qualification_v1", "manifest_sha256": base.sha256_path(output_root / "manifest.json"), "summary_sha256": base.sha256_path(summary_path), "decision": summary["decision"]}
    base._atomic_json(terminal_path, terminal)
    return validate_artifacts(output_root)


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve(); protocol = load_protocol()
    terminal = json.loads((output_root / "terminal.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output_root / "checkpoint.json").read_text(encoding="utf-8"))
    if terminal.get("status") != "completed_continuous_real_source_qualification_v1" or terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json") or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json") or manifest.get("source_hashes") != _source_hashes(protocol) or terminal.get("decision") != summary.get("decision"):
        raise ValueError("real-source terminal identity changed")
    for name, expected in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != expected:
            raise ValueError(f"real-source artifact changed: {name}")
    validate_grid(protocol, checkpoint)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume)
    print(base.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
