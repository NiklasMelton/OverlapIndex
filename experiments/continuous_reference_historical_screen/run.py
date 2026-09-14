"""Run the eight-target historical reference screen."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.metrics import r2_score

from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_reference_library_qualification import run as q1
from experiments.continuous_reference_library_qualification_v3 import run as q3


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_reference_historical_screen" / "screen"
Q3_ROOT = ROOT / "artifacts" / "continuous_reference_library_qualification_v3" / "pilot"
BACKBONES = q1.BACKBONES
HEADS = q1.HEADS
TIE_ATOL = q1.TIE_ATOL


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or base.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("historical screen protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_reference_historical_screen_before_outcomes":
        raise ValueError("historical screen protocol is not frozen")
    if tuple(protocol["backbones"]) != BACKBONES or tuple(protocol["reference_library"]["families"]) != HEADS:
        raise ValueError("historical screen roster changed")
    if len({row["target_key"] for row in protocol["targets"]}) != len(protocol["targets"]):
        raise ValueError("historical screen target keys repeat")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL_PATH,
        "protocol_sidecar": PROTOCOL_SHA256_PATH,
        "runner": Path(__file__).resolve(),
        "qualified_evaluator": q1.PACKAGE_DIR / "evaluator.py",
        "qualification_v3_protocol": q3.PROTOCOL_PATH,
        "qualification_v3_raw_results": Q3_ROOT / "raw_results.json",
        "qualification_v3_manifest": Q3_ROOT / "manifest.json",
        "input_cohort": q1.v1_prepare.DEFAULT_OUTPUT_ROOT / "cohort.json",
        "embedding_manifest": q1.v1_extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
    }


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"historical screen provenance input is missing: {missing}")
    observed = {name: base.sha256_path(path) for name, path in paths.items()}
    ancestry = protocol["ancestry"]
    expected = {
        "qualification_v3_protocol": ancestry["qualification_v3_protocol_sha256"],
        "qualification_v3_raw_results": ancestry["qualification_v3_raw_results_sha256"],
        "qualification_v3_manifest": ancestry["qualification_v3_manifest_sha256"],
    }
    if any(observed[key] != value for key, value in expected.items()):
        raise ValueError("historical screen qualification ancestry changed")
    terminal = q3.validate_artifacts(Q3_ROOT)
    if not terminal["decision"]["qualification_pass"]:
        raise ValueError("historical screen requires a passing reference-library qualification")
    return observed


def _target_draws(cohort, config):
    dataset_id, target_id = str(config["target_key"]).split("::", 1)
    rows = cohort["datasets"][dataset_id]["target_cohorts"][target_id]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    if len(evaluation) != int(config["evaluation_rows"]):
        raise ValueError(f"historical evaluation count changed: {config['target_key']}")
    evaluation_ids = [str(row["sample_id"]) for row in evaluation]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError(f"historical evaluation identities repeat: {config['target_key']}")
    draws = [
        base.select_unique_draw(
            rows, pool_rows=int(config["training_pool_rows"]),
            draw_rows=int(config["training_rows_per_draw"]), seed=int(seed)
        )
        for seed in config["draw_seeds"]
    ]
    if any(set(evaluation_ids).intersection(str(row["sample_id"]) for row in draw) for draw in draws):
        raise ValueError(f"historical training/evaluation overlap: {config['target_key']}")
    return dataset_id, evaluation, draws


def build_panel(screen_protocol, library_protocol, config, cohort, matrices):
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
                split_seed=base._seed("historical-screen", target_key, backbone, draw_seed),
                protocol=library_protocol,
            )
            candidate_draws.append({
                "draw": draw_position, "draw_seed": int(draw_seed),
                "sample_ids": [str(row["sample_id"]) for row in draw],
                "sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in draw]),
                "unique_training_rows": len({str(row["sample_id"]) for row in draw}), **result,
            })
        head_means = {head: float(np.mean([draw["heads"][head]["evaluation_r2"] for draw in candidate_draws])) for head in HEADS}
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
        "draw_count": len(draws), "draw_limitation": config["draw_limitation"],
        "evaluation_rows": len(evaluation),
        "evaluation_sample_ids": [str(row["sample_id"]) for row in evaluation],
        "evaluation_target": y_evaluation.tolist(), "candidates": candidates,
    }
    panel["screen"] = screen_panel(panel, screen_protocol)
    return panel


def _bootstrap_utilities(panel, protocol):
    candidates = panel["candidates"]
    y = np.asarray(panel["evaluation_target"], dtype=np.float64)
    predictions = np.asarray([
        [[draw["heads"][head]["prediction"] for head in HEADS] for draw in candidate["draws"]]
        for candidate in candidates
    ], dtype=np.float64)
    draw_count = predictions.shape[1]
    resamples = int(protocol["uncertainty"]["resamples"])
    rng = np.random.default_rng(base._seed(protocol["uncertainty"]["seed"], panel["target_key"]))
    boot = np.empty((resamples, len(BACKBONES), len(HEADS)), dtype=np.float64)
    for sample in range(resamples):
        draw_ids = rng.integers(0, draw_count, size=draw_count)
        row_ids = rng.integers(0, len(y), size=len(y))
        for candidate in range(len(BACKBONES)):
            for head in range(len(HEADS)):
                boot[sample, candidate, head] = float(np.mean([
                    r2_score(y[row_ids], predictions[candidate, draw, head, row_ids])
                    for draw in draw_ids
                ]))
    return boot


def screen_panel(panel, protocol):
    point_heads = np.asarray([
        [candidate["head_mean_evaluation_r2"][head] for head in HEADS]
        for candidate in panel["candidates"]
    ], dtype=np.float64)
    point_envelope = np.max(point_heads, axis=1)
    point_linear = point_heads[:, HEADS.index("linear_ridge")]
    point_nonlinear = np.max(point_heads[:, 1:], axis=1)
    boot_heads = _bootstrap_utilities(panel, protocol)
    boot_envelope = np.max(boot_heads, axis=2)
    alpha = float(protocol["uncertainty"]["familywise_alpha"])
    utilities = int(protocol["uncertainty"]["utility_count_per_target"])
    lower_q, upper_q = alpha / (2.0 * utilities), 1.0 - alpha / (2.0 * utilities)
    head_lower = np.quantile(boot_heads, lower_q, axis=0)
    head_upper = np.quantile(boot_heads, upper_q, axis=0)
    envelope_lower = np.max(head_lower, axis=1)
    envelope_upper = np.max(head_upper, axis=1)
    linear_position = int(np.flatnonzero(point_linear >= np.max(point_linear) - TIE_ATOL)[0])
    winner_position = int(np.flatnonzero(point_envelope >= np.max(point_envelope) - TIE_ATOL)[0])
    regret_bounds = [
        float(max(0.0, np.max(envelope_lower) - envelope_upper[linear_position])),
        float(max(0.0, np.max(envelope_upper) - envelope_lower[linear_position])),
    ]
    gain_lower = np.max(head_lower[:, 1:], axis=1) - head_upper[:, 0]
    gain_upper = np.max(head_upper[:, 1:], axis=1) - head_lower[:, 0]
    material = float(protocol["uncertainty"]["material_r2"])
    reversal_rows = []
    for left in range(len(BACKBONES)):
        for right in range(left + 1, len(BACKBONES)):
            envelope_delta = float(point_envelope[left] - point_envelope[right])
            linear_delta = float(point_linear[left] - point_linear[right])
            if abs(envelope_delta) < material or abs(linear_delta) < material or np.sign(envelope_delta) == np.sign(linear_delta):
                continue
            probability = float(np.mean(
                np.sign(boot_envelope[:, left] - boot_envelope[:, right]) == np.sign(envelope_delta)
            ))
            if probability >= float(protocol["uncertainty"]["reversal_probability_threshold"]):
                reversal_rows.append({"left": BACKBONES[left], "right": BACKBONES[right], "envelope_delta_r2": envelope_delta, "linear_delta_r2": linear_delta, "envelope_sign_probability": probability})
    top_probability = float(np.mean(
        boot_envelope[:, winner_position] >= np.max(boot_envelope, axis=1) - TIE_ATOL
    ))
    config = protocol["classification"]
    best = float(np.max(point_envelope))
    near = point_envelope >= best - float(config["near_leading_gap_r2"])
    learnable = bool(best >= float(config["learnable_best_envelope_point_r2"]) and envelope_lower[winner_position] > float(config["learnable_best_envelope_lower_bound_r2"]))
    nonlinear_opportunity = bool(np.any(near & (gain_lower > float(config["nonlinear_gain_lower_bound_r2"]))))
    ranking_conflict = bool(regret_bounds[0] > float(config["linear_regret_lower_bound_r2"]) or len(reversal_rows) >= int(config["minimum_material_pair_reversals"]))
    identifiable = top_probability >= float(protocol["uncertainty"]["top1_probability_threshold"])
    maximum_gain_upper = float(np.max(gain_upper))
    if learnable and nonlinear_opportunity and ranking_conflict and identifiable:
        label = "nonlinear_challenge_lead"
    elif learnable and regret_bounds[1] <= float(config["linear_control_regret_upper_bound_r2"]) and maximum_gain_upper <= float(config["linear_control_nonlinear_gain_upper_bound_r2"]):
        label = "linear_control"
    elif learnable and (nonlinear_opportunity or float(np.max(point_nonlinear - point_linear)) > float(config["nonlinear_gain_lower_bound_r2"])):
        label = "mixed"
    else:
        label = "unresolved"
    return {
        "classification": label, "learnable": learnable,
        "nonlinear_opportunity": nonlinear_opportunity, "ranking_conflict": ranking_conflict,
        "winner_identifiable": identifiable, "envelope_best_candidate_ids": [BACKBONES[index] for index in np.flatnonzero(point_envelope >= best - TIE_ATOL)],
        "linear_best_candidate_ids": [BACKBONES[index] for index in np.flatnonzero(point_linear >= np.max(point_linear) - TIE_ATOL)],
        "linear_to_envelope_spearman": base.ranking_metrics(point_envelope, point_linear, BACKBONES)["spearman"],
        "linear_top1_regret_point_r2": base.ranking_metrics(point_envelope, point_linear, BACKBONES)["top1_regret_r2"],
        "linear_top1_regret_conservative_bounds_r2": regret_bounds,
        "best_envelope_r2": best, "best_envelope_simultaneous_bounds_r2": [float(envelope_lower[winner_position]), float(envelope_upper[winner_position])],
        "winner_top1_probability": top_probability,
        "maximum_nonlinear_gain_point_r2": float(np.max(point_nonlinear - point_linear)),
        "maximum_nonlinear_gain_upper_bound_r2": maximum_gain_upper,
        "candidate_nonlinear_gain_simultaneous_bounds_r2": {BACKBONES[index]: [float(gain_lower[index]), float(gain_upper[index])] for index in range(len(BACKBONES))},
        "material_pair_reversals": reversal_rows,
        "bootstrap_resamples": int(protocol["uncertainty"]["resamples"]),
        "simultaneous_quantiles": [lower_q, upper_q],
    }


def summarize(protocol, panels):
    rows = [{"target_key": panel["target_key"], "dataset_id": panel["dataset_id"], "draw_count": panel["draw_count"], "draw_limitation": panel["draw_limitation"], **panel["screen"]} for panel in panels]
    nonlinear = [row for row in rows if row["classification"] == "nonlinear_challenge_lead"]
    sources = {row["dataset_id"] for row in nonlinear}
    sufficient = len(nonlinear) >= 2 and len(sources) >= 2
    return {
        "schema_version": 1, "status": "completed_continuous_reference_historical_screen",
        "target_rows": rows,
        "classification_counts": {label: sum(row["classification"] == label for row in rows) for label in protocol["classification"]["labels"]},
        "decision": {
            "sufficient_historical_nonlinear_diversity": sufficient,
            "nonlinear_challenge_lead_count": len(nonlinear),
            "nonlinear_challenge_source_count": len(sources),
            "next_stage": "larger_new_data_precision_panel" if sufficient else "expand_real_sources_before_oi",
            "method_promotion": "none_reference_only",
        },
        "overlapindex_fit_count": 0, "reserved_backbones_evaluated": [],
    }


def render_report(summary):
    lines = ["# Continuous reference historical screen", "", "| Target | Draws | Class | Envelope best | Linear best | Linear rho | Point regret | Conservative regret | Nonlinear gain | Top-1 probability |", "|---|---:|---|---|---|---:|---:|---|---:|---:|"]
    for row in summary["target_rows"]:
        bounds = row["linear_top1_regret_conservative_bounds_r2"]
        lines.append(f"| {row['target_key']} | {row['draw_count']} | {row['classification']} | {', '.join(row['envelope_best_candidate_ids'])} | {', '.join(row['linear_best_candidate_ids'])} | {row['linear_to_envelope_spearman']:.4f} | {row['linear_top1_regret_point_r2']:.4f} | [{bounds[0]:.4f}, {bounds[1]:.4f}] | {row['maximum_nonlinear_gain_point_r2']:.4f} | {row['winner_top1_probability']:.4f} |")
    decision = summary["decision"]
    lines.extend(["", "## Decision", "", f"- Classification counts: `{json.dumps(summary['classification_counts'], sort_keys=True)}`.", f"- Nonlinear challenge leads: **{decision['nonlinear_challenge_lead_count']}** from **{decision['nonlinear_challenge_source_count']}** sources.", f"- Historical nonlinear diversity sufficient: **{str(decision['sufficient_historical_nonlinear_diversity']).lower()}**.", f"- Next stage: **{decision['next_stage']}**.", "- This historical development screen promotes no OverlapIndex method.", ""])
    return "\n".join(lines)


def _identity(protocol, hashes):
    return {"protocol_sha256": base.sha256_path(PROTOCOL_PATH), "qualified_library_protocol_sha256": base.sha256_path(q3.PROTOCOL_PATH), "source_hashes": dict(hashes), "target_keys": [row["target_key"] for row in protocol["targets"]]}


def validate_grid(protocol, checkpoint):
    expected = protocol["expected_grid"]; panels = checkpoint["panels"]
    if len(panels) != int(expected["target_count"]) or sum(len(panel["candidates"]) for panel in panels) != int(expected["candidate_count"]) or sum(len(candidate["draws"]) for panel in panels for candidate in panel["candidates"]) != int(expected["candidate_draw_count"]):
        raise ValueError("historical screen executed grid changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False):
    output_root = output_root.resolve(); protocol = load_protocol(); library = q3.load_protocol(); hashes = _source_hashes(protocol); terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume: raise FileExistsError("historical screen terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    checkpoint_path = output_root / "checkpoint.json"; identity = _identity(protocol, hashes)
    if checkpoint_path.exists():
        if not resume: raise FileExistsError("historical screen checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("identity") != identity: raise ValueError("historical screen checkpoint identity changed")
    else: checkpoint = {"schema_version": 1, "identity": identity, "panels": []}
    cohort, matrices = q1.v1_extract.load_bank(); complete = {row["target_key"] for row in checkpoint["panels"]}
    for config in protocol["targets"]:
        if config["target_key"] not in complete:
            checkpoint["panels"].append(build_panel(protocol, library, config, cohort, matrices)); base._finite_json(checkpoint); base._atomic_json(checkpoint_path, checkpoint); print(f"target {len(checkpoint['panels'])}/{len(protocol['targets'])}: {config['target_key']}", flush=True)
    validate_grid(protocol, checkpoint); summary = summarize(protocol, checkpoint["panels"])
    raw = {"schema_version": 1, "status": "completed_continuous_reference_historical_screen", "protocol": protocol, "qualified_library_protocol": library, "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": hashes, "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "thread_environment": {key: os.environ.get(key) for key in base.THREAD_KEYS}}, "panels": checkpoint["panels"], "overlapindex_fit_count": 0, "reserved_backbones_evaluated": []}
    base._finite_json(raw); base._finite_json(summary); raw_path, summary_path, report_path = output_root / "raw_results.json", output_root / "summary.json", output_root / "report.md"; base._atomic_json(raw_path, raw); base._atomic_json(summary_path, summary); base._atomic_bytes(report_path, render_report(summary).encode())
    manifest = {"schema_version": 1, "status": "completed_continuous_reference_historical_screen", "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "qualified_library_protocol_sha256": base.sha256_path(q3.PROTOCOL_PATH), "source_hashes": hashes, "checkpoint_identity": identity, "artifacts": {name: base.sha256_path(output_root / name) for name in ("checkpoint.json", "raw_results.json", "summary.json", "report.md")}, "counts": {"targets": len(checkpoint["panels"]), "candidates": sum(len(panel["candidates"]) for panel in checkpoint["panels"]), "candidate_draws": sum(len(candidate["draws"]) for panel in checkpoint["panels"] for candidate in panel["candidates"]), "overlapindex_fits": 0, "reserved_backbones": 0}, "git": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())}, "decision": summary["decision"]}
    base._atomic_json(output_root / "manifest.json", manifest); terminal = {"schema_version": 1, "status": "completed_continuous_reference_historical_screen", "manifest_sha256": base.sha256_path(output_root / "manifest.json"), "summary_sha256": base.sha256_path(summary_path), "decision": summary["decision"]}; base._atomic_json(terminal_path, terminal); return validate_artifacts(output_root)


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT):
    output_root = output_root.resolve(); protocol = load_protocol(); terminal = json.loads((output_root / "terminal.json").read_text()); manifest = json.loads((output_root / "manifest.json").read_text()); summary = json.loads((output_root / "summary.json").read_text()); checkpoint = json.loads((output_root / "checkpoint.json").read_text())
    if terminal.get("status") != "completed_continuous_reference_historical_screen" or terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json") or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json") or manifest.get("source_hashes") != _source_hashes(protocol) or terminal.get("decision") != summary.get("decision"): raise ValueError("historical screen terminal identity changed")
    for name, expected in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != expected: raise ValueError(f"historical screen artifact changed: {name}")
    validate_grid(protocol, checkpoint); return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT); parser.add_argument("--resume", action="store_true"); parser.add_argument("--validate-only", action="store_true"); args = parser.parse_args(argv); result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume); print(base.canonical_json(result)); return 0


if __name__ == "__main__": raise SystemExit(main())

