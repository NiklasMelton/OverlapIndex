"""Run the stronger continuous reference-library qualification experiment."""

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

from experiments.continuous_ranking_benchmark_v2 import run as v2
from experiments.continuous_ranking_reliability_recoverability import extract as v1_extract
from experiments.continuous_ranking_reliability_recoverability import prepare as v1_prepare
from experiments.continuous_reference_library_qualification.evaluator import HEADS, TIE_ATOL, evaluate_heads


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_reference_library_qualification" / "pilot"
V2_ROOT = ROOT / "artifacts" / "continuous_ranking_benchmark_v2" / "evaluator_pilot"
BACKBONES = tuple(v1_extract.BACKBONES)
SYNTHETIC_CANDIDATES = v2.SYNTHETIC_CANDIDATES
THREAD_KEYS = v2.THREAD_KEYS


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or v2.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("reference-library qualification protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_reference_library_qualification_before_outcomes":
        raise ValueError("reference-library qualification protocol is not frozen")
    if tuple(protocol["backbones"]) != BACKBONES:
        raise ValueError("reference-library qualification backbone roster changed")
    if tuple(protocol["heads"]["families"]) != HEADS:
        raise ValueError("reference-library qualification head roster changed")
    if int(protocol["heads"]["inner_folds"]) != v2.INNER_FOLDS:
        raise ValueError("reference-library qualification fold count changed")
    if int(protocol["preprocessing"]["pca_rank"]) != v2.PCA_RANK:
        raise ValueError("reference-library qualification PCA rank changed")
    if tuple(protocol["synthetic_controls"]["candidate_ids"]) != SYNTHETIC_CANDIDATES:
        raise ValueError("reference-library qualification synthetic roster changed")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL_PATH,
        "protocol_sidecar": PROTOCOL_SHA256_PATH,
        "evaluator": PACKAGE_DIR / "evaluator.py",
        "runner": Path(__file__).resolve(),
        "v2_protocol": v2.PROTOCOL_PATH,
        "v2_raw_results": V2_ROOT / "raw_results.json",
        "v2_manifest": V2_ROOT / "manifest.json",
        "v1_input_cohort": v1_prepare.DEFAULT_OUTPUT_ROOT / "cohort.json",
        "v1_embedding_manifest": v1_extract.DEFAULT_OUTPUT_ROOT / "manifest.json",
    }


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"reference-library provenance input is missing: {missing}")
    observed = {name: v2.sha256_path(path) for name, path in paths.items()}
    ancestry = protocol["ancestry"]
    expected = {
        "v2_protocol": ancestry["v2_protocol_sha256"],
        "v2_raw_results": ancestry["v2_raw_results_sha256"],
        "v2_manifest": ancestry["v2_manifest_sha256"],
        "v1_input_cohort": ancestry["v1_input_cohort_sha256"],
        "v1_embedding_manifest": ancestry["v1_embedding_manifest_sha256"],
    }
    if any(observed[key] != value for key, value in expected.items()):
        raise ValueError("reference-library qualification ancestry identity changed")
    v2.validate_artifacts(V2_ROOT)
    return observed


def _checkpoint_identity(protocol: Mapping[str, Any], source_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "protocol_sha256": v2.sha256_path(PROTOCOL_PATH),
        "source_hashes": dict(source_hashes),
        "backbones": list(BACKBONES),
        "targets": [row["target_key"] for row in protocol["targets"]],
        "synthetic_families": list(protocol["synthetic_controls"]["families"]),
        "synthetic_seeds": list(protocol["synthetic_controls"]["seeds"]),
    }


def _load_checkpoint(path: Path, identity: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError("reference-library checkpoint exists; use --resume")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("identity") != identity:
            raise ValueError("reference-library checkpoint identity changed")
        return value
    return {"schema_version": 1, "identity": dict(identity), "real_panels": [], "synthetic_panels": []}


def _evaluation_rows(
    cohort: Mapping[str, Any], target_config: Mapping[str, Any]
) -> tuple[str, str, list[Mapping[str, Any]]]:
    target_key = str(target_config["target_key"])
    dataset_id, target_id = target_key.split("::", 1)
    rows = cohort["datasets"][dataset_id]["target_cohorts"][target_id]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    if len(evaluation) != int(target_config["evaluation_rows"]):
        raise ValueError(f"reference-library evaluation count changed: {target_key}")
    ids = [str(row["sample_id"]) for row in evaluation]
    if len(ids) != len(set(ids)):
        raise ValueError(f"reference-library evaluation identities repeat: {target_key}")
    return dataset_id, target_id, evaluation


def build_real_panel(
    protocol: Mapping[str, Any],
    target_config: Mapping[str, Any],
    cohort: Mapping[str, Any],
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    v2_raw: Mapping[str, Any],
) -> dict[str, Any]:
    target_key = str(target_config["target_key"])
    dataset_id, target_id, evaluation = _evaluation_rows(cohort, target_config)
    rows = cohort["datasets"][dataset_id]["target_cohorts"][target_id]
    evaluation_ids = [str(row["sample_id"]) for row in evaluation]
    y_evaluation = np.asarray([float(row["target"]) for row in evaluation], dtype=np.float64)
    evaluation_indices = np.asarray([int(row["matrix_index"]) for row in evaluation], dtype=np.int64)
    draws = [
        v2.select_unique_draw(
            rows,
            pool_rows=int(target_config["training_pool_rows"]),
            draw_rows=int(target_config["training_rows_per_draw"]),
            seed=int(seed),
        )
        for seed in target_config["draw_seeds"]
    ]
    if any(set(evaluation_ids).intersection(str(row["sample_id"]) for row in draw) for draw in draws):
        raise ValueError(f"reference-library training/evaluation identities overlap: {target_key}")

    prior_panel = next(row for row in v2_raw["real_panels"] if row["target_key"] == target_key)
    prior_by_candidate = {row["candidate_id"]: row for row in prior_panel["candidates"]}
    candidates = []
    for backbone in BACKBONES:
        matrix = matrices[dataset_id][backbone]
        candidate_draws = []
        for draw_position, (draw_seed, draw) in enumerate(zip(target_config["draw_seeds"], draws)):
            train_indices = np.asarray([int(row["matrix_index"]) for row in draw], dtype=np.int64)
            y_train = np.asarray([float(row["target"]) for row in draw], dtype=np.float64)
            result = evaluate_heads(
                matrix[train_indices],
                y_train,
                np.asarray([int(row["target_bin"]) for row in draw], dtype=np.int64),
                [str(row["sample_id"]) for row in draw],
                matrix[evaluation_indices],
                y_evaluation,
                split_seed=v2._seed("real-inner", target_key, backbone, draw_seed),
                protocol=protocol,
            )
            candidate_draws.append(
                {
                    "draw": draw_position,
                    "draw_seed": int(draw_seed),
                    "sample_ids": [str(row["sample_id"]) for row in draw],
                    "sample_ids_sha256": v2.sha256_payload([str(row["sample_id"]) for row in draw]),
                    "unique_training_rows": len({str(row["sample_id"]) for row in draw}),
                    **result,
                }
            )
        head_means = {
            head: float(np.mean([draw["heads"][head]["evaluation_r2"] for draw in candidate_draws]))
            for head in HEADS
        }
        envelope = max(head_means.values())
        prior_draws = [
            draw for draw in prior_by_candidate[backbone]["draws"]
            if int(draw["draw_seed"]) in {int(value) for value in target_config["draw_seeds"]}
        ]
        if len(prior_draws) != len(candidate_draws):
            raise ValueError(f"reference-library V2 comparison draw set changed: {target_key}/{backbone}")
        prior_head_means = {
            head: float(np.mean([draw["heads"][head]["evaluation_r2"] for draw in prior_draws]))
            for head in HEADS
        }
        candidates.append(
            {
                "candidate_id": backbone,
                "draws": candidate_draws,
                "head_mean_evaluation_r2": head_means,
                "envelope_r2": float(envelope),
                "best_head_ids": [head for head in HEADS if head_means[head] >= envelope - TIE_ATOL],
                "v2_same_draw_head_mean_evaluation_r2": prior_head_means,
                "v2_same_draw_envelope_r2": float(max(prior_head_means.values())),
            }
        )
    envelope = [row["envelope_r2"] for row in candidates]
    linear = [row["head_mean_evaluation_r2"]["linear_ridge"] for row in candidates]
    return {
        "target_key": target_key,
        "role": str(target_config["role"]),
        "training_pool_rows": int(target_config["training_pool_rows"]),
        "training_rows_per_draw": int(target_config["training_rows_per_draw"]),
        "draw_count": len(draws),
        "evaluation_rows": len(evaluation),
        "evaluation_sample_ids": evaluation_ids,
        "evaluation_target": y_evaluation.tolist(),
        "candidates": candidates,
        "linear_ranking_metrics": v2.ranking_metrics(envelope, linear, BACKBONES),
    }


def build_synthetic_panel(family: str, seed: int, protocol: Mapping[str, Any]) -> dict[str, Any]:
    data = v2.synthetic_data(family, seed, protocol)
    training_rows = int(protocol["synthetic_controls"]["training_rows"])
    y = np.asarray(data["y"], dtype=np.float64)
    y_train, y_evaluation = y[:training_rows], y[training_rows:]
    bins = v2._target_bins(y_train)
    sample_ids = [f"synthetic/{family}/{seed}/{position}" for position in range(training_rows)]
    candidates = []
    for candidate_id in SYNTHETIC_CANDIDATES:
        matrix = data["matrices"][candidate_id]
        result = evaluate_heads(
            matrix[:training_rows], y_train, bins, sample_ids,
            matrix[training_rows:], y_evaluation,
            split_seed=v2._seed("synthetic-inner", family, seed, candidate_id),
            protocol=protocol,
        )
        head_scores = {head: result["heads"][head]["evaluation_r2"] for head in HEADS}
        envelope = max(head_scores.values())
        candidates.append(
            {
                "candidate_id": candidate_id,
                "heads": result["heads"],
                "head_evaluation_r2": head_scores,
                "envelope_r2": float(envelope),
                "best_head_ids": [head for head in HEADS if head_scores[head] >= envelope - TIE_ATOL],
                "fold_audit": result["fold_audit"],
                "full_preprocessing": result["full_preprocessing"],
                "full_rbf_median_distance": result["full_rbf_median_distance"],
                "shared_preprocessing_wall_seconds": result["shared_preprocessing_wall_seconds"],
            }
        )
    envelope = [row["envelope_r2"] for row in candidates]
    linear = [row["head_evaluation_r2"]["linear_ridge"] for row in candidates]
    intended = {"linear": "linear_ridge", "quadratic": "quadratic_ridge", "smooth_periodic": "rbf_svr"}[family]
    return {
        "family": family,
        "seed": int(seed),
        "training_rows": training_rows,
        "unique_training_rows": len(set(sample_ids)),
        "evaluation_rows": len(y_evaluation),
        "candidates": candidates,
        "linear_ranking_metrics": v2.ranking_metrics(envelope, linear, SYNTHETIC_CANDIDATES),
        "clean_latent_is_best": bool(envelope[0] >= max(envelope) - TIE_ATOL),
        "clean_intended_head": intended,
        "clean_intended_head_r2": float(candidates[0]["head_evaluation_r2"][intended]),
    }


def _all_head_results(
    real_panels: Sequence[Mapping[str, Any]], synthetic_panels: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for panel in real_panels:
        for candidate in panel["candidates"]:
            for draw in candidate["draws"]:
                for head, result in draw["heads"].items():
                    rows.append({"domain": "real", "panel": panel["target_key"], "candidate": candidate["candidate_id"], "head": head, **result})
    for panel in synthetic_panels:
        for candidate in panel["candidates"]:
            for head, result in candidate["heads"].items():
                rows.append({"domain": "synthetic", "panel": f"{panel['family']}/{panel['seed']}", "candidate": candidate["candidate_id"], "head": head, **result})
    return rows


def summarize(
    protocol: Mapping[str, Any],
    real_panels: Sequence[Mapping[str, Any]],
    synthetic_panels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_heads = _all_head_results(real_panels, synthetic_panels)
    endpoint_rows = [
        {"domain": row["domain"], "panel": row["panel"], "candidate": row["candidate"], "head": row["head"], **item}
        for row in all_heads for item in row["endpoint_diagnostics"]
    ]
    material_endpoints = [row for row in endpoint_rows if row["material"]]
    warnings_rows = [
        {"domain": row["domain"], "panel": row["panel"], "candidate": row["candidate"], "head": row["head"], "warning_types": warning_types}
        for row in all_heads
        for warning_types in [sorted(set(row["final_warning_types"]).union(*(set(item["warning_types"]) for item in row["parameter_rows"]))) ]
        if warning_types
    ]

    real_rows = []
    for panel in real_panels:
        envelopes = np.asarray([row["envelope_r2"] for row in panel["candidates"]], dtype=np.float64)
        linear = np.asarray([row["head_mean_evaluation_r2"]["linear_ridge"] for row in panel["candidates"]], dtype=np.float64)
        prior = np.asarray([row["v2_same_draw_envelope_r2"] for row in panel["candidates"]], dtype=np.float64)
        real_rows.append(
            {
                "target_key": panel["target_key"],
                "role": panel["role"],
                "draw_count": panel["draw_count"],
                "envelope_best_candidate_ids": [BACKBONES[index] for index in np.flatnonzero(envelopes >= np.max(envelopes) - TIE_ATOL)],
                "linear_best_candidate_ids": [BACKBONES[index] for index in np.flatnonzero(linear >= np.max(linear) - TIE_ATOL)],
                "linear_to_envelope_spearman": panel["linear_ranking_metrics"]["spearman"],
                "linear_top1_regret_r2": panel["linear_ranking_metrics"]["top1_regret_r2"],
                "maximum_nonlinear_gain_r2": float(np.max(envelopes - linear)),
                "mean_envelope_change_from_v2_r2": float(np.mean(envelopes - prior)),
                "maximum_envelope_change_from_v2_r2": float(np.max(envelopes - prior)),
            }
        )

    synth_config = protocol["synthetic_controls"]
    synthetic_rows = [
        {
            "family": panel["family"],
            "seed": panel["seed"],
            "clean_latent_is_best": panel["clean_latent_is_best"],
            "clean_intended_head": panel["clean_intended_head"],
            "clean_intended_head_r2": panel["clean_intended_head_r2"],
            "linear_top1_regret_r2": panel["linear_ranking_metrics"]["top1_regret_r2"],
        }
        for panel in synthetic_panels
    ]
    family_rows = []
    synthetic_pass = True
    for family in synth_config["families"]:
        rows = [row for row in synthetic_rows if row["family"] == family]
        clean_probability = float(np.mean([row["clean_latent_is_best"] for row in rows]))
        minimum_intended = float(min(row["clean_intended_head_r2"] for row in rows))
        mean_regret = float(np.mean([row["linear_top1_regret_r2"] for row in rows]))
        passed = bool(
            clean_probability >= float(synth_config["minimum_clean_latent_top_probability"])
            and minimum_intended >= float(synth_config["minimum_intended_clean_head_r2"])
        )
        if family == "linear":
            passed &= mean_regret <= float(synth_config["maximum_linear_control_regret_r2"])
        else:
            passed &= mean_regret >= float(synth_config["minimum_nonlinear_linear_probe_regret_r2"])
        synthetic_pass &= passed
        family_rows.append(
            {
                "family": family,
                "panel_count": len(rows),
                "clean_latent_top_probability": clean_probability,
                "minimum_clean_intended_head_r2": minimum_intended,
                "mean_linear_top1_regret_r2": mean_regret,
                "passed": bool(passed),
            }
        )

    by_target = {row["target_key"]: row for row in real_rows}
    cub_pass = all(
        row["linear_top1_regret_r2"] <= float(protocol["real_controls"]["maximum_cub_linear_top1_regret_r2"])
        for row in real_rows if row["role"] == "linear_control"
    )
    lsp = next(row for row in real_rows if row["role"] == "nonlinear_head_control")
    lsp_pass = lsp["maximum_nonlinear_gain_r2"] >= float(protocol["real_controls"]["minimum_lsp_maximum_nonlinear_gain_r2"])
    endpoint_pass = len(material_endpoints) <= int(protocol["grid_adequacy"]["maximum_material_endpoint_selections"])
    warning_pass = not warnings_rows
    decision = {
        "grid_endpoints_adequate": bool(endpoint_pass),
        "synthetic_controls_pass": bool(synthetic_pass),
        "cub_linear_controls_pass": bool(cub_pass),
        "lsp_nonlinear_head_gain_pass": bool(lsp_pass),
        "warning_gate_pass": bool(warning_pass),
        "qualification_pass": bool(endpoint_pass and synthetic_pass and cub_pass and lsp_pass and warning_pass),
        "next_stage": "eight_target_historical_screen" if endpoint_pass and synthetic_pass and cub_pass and lsp_pass and warning_pass else "blocked_new_protocol_required",
        "method_promotion": "none_reference_only",
    }
    return {
        "schema_version": 1,
        "status": "completed_continuous_reference_library_qualification",
        "real_rows": real_rows,
        "synthetic_rows": synthetic_rows,
        "synthetic_family_rows": family_rows,
        "endpoint_selection_count": len(endpoint_rows),
        "material_endpoint_selection_count": len(material_endpoints),
        "material_endpoint_rows": material_endpoints,
        "warning_rows": warnings_rows,
        "decision": decision,
        "overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous reference-library qualification", "",
        "## Real controls", "",
        "| Target | Envelope best | Linear best | Linear rho | Linear regret | Nonlinear gain | Mean change vs V2 |", 
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["real_rows"]:
        lines.append(
            f"| {row['target_key']} | {', '.join(row['envelope_best_candidate_ids'])} | "
            f"{', '.join(row['linear_best_candidate_ids'])} | {row['linear_to_envelope_spearman']:.4f} | "
            f"{row['linear_top1_regret_r2']:.4f} | {row['maximum_nonlinear_gain_r2']:.4f} | "
            f"{row['mean_envelope_change_from_v2_r2']:.4f} |"
        )
    lines.extend(["", "## Synthetic controls", "", "| Family | Panels | Clean top | Minimum intended-head R2 | Mean linear regret | Pass |", "|---|---:|---:|---:|---:|---:|"])
    for row in summary["synthetic_family_rows"]:
        lines.append(
            f"| {row['family']} | {row['panel_count']} | {row['clean_latent_top_probability']:.4f} | "
            f"{row['minimum_clean_intended_head_r2']:.4f} | {row['mean_linear_top1_regret_r2']:.4f} | "
            f"{str(row['passed']).lower()} |"
        )
    decision = summary["decision"]
    lines.extend([
        "", "## Frozen decision", "",
        f"- Material endpoint selections: **{summary['material_endpoint_selection_count']}**.",
        f"- Endpoint gate: **{str(decision['grid_endpoints_adequate']).lower()}**.",
        f"- Synthetic gate: **{str(decision['synthetic_controls_pass']).lower()}**.",
        f"- CUB control gate: **{str(decision['cub_linear_controls_pass']).lower()}**.",
        f"- LSP nonlinear-head gate: **{str(decision['lsp_nonlinear_head_gain_pass']).lower()}**.",
        f"- Warning gate: **{str(decision['warning_gate_pass']).lower()}**.",
        f"- Qualification pass: **{str(decision['qualification_pass']).lower()}**.",
        f"- Next stage: **{decision['next_stage']}**.",
        "- No OverlapIndex method was evaluated or promoted.", "",
    ])
    if summary["material_endpoint_rows"]:
        lines.extend(["## Material endpoint failures", "", "| Domain | Panel | Candidate | Head | Parameter | Endpoint | Advantage |", "|---|---|---|---|---|---|---:|"])
        for row in summary["material_endpoint_rows"]:
            lines.append(f"| {row['domain']} | {row['panel']} | {row['candidate']} | {row['head']} | {row['parameter']} | {row['endpoint']} | {row['inner_cv_advantage_r2']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def validate_grid(protocol: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    expected = protocol["expected_grid"]
    real = checkpoint["real_panels"]
    synthetic = checkpoint["synthetic_panels"]
    real_draws = sum(len(candidate["draws"]) for panel in real for candidate in panel["candidates"])
    synthetic_candidates = sum(len(panel["candidates"]) for panel in synthetic)
    if (
        len(real) != int(expected["real_target_count"])
        or real_draws != int(expected["real_candidate_draw_count"])
        or len(synthetic) != int(expected["synthetic_panel_count"])
        or synthetic_candidates != int(expected["synthetic_candidate_count"])
    ):
        raise ValueError("reference-library executed grid changed")


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    output_root = output_root.resolve()
    protocol = load_protocol()
    source_hashes = _source_hashes(protocol)
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("reference-library terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    identity = _checkpoint_identity(protocol, source_hashes)
    checkpoint_path = output_root / "checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path, identity, resume=resume)
    cohort, matrices, _ = v1_extract.load_bank()
    v2_raw = json.loads((V2_ROOT / "raw_results.json").read_text(encoding="utf-8"))

    completed_real = {row["target_key"] for row in checkpoint["real_panels"]}
    for target_config in protocol["targets"]:
        target_key = str(target_config["target_key"])
        if target_key in completed_real:
            continue
        checkpoint["real_panels"].append(build_real_panel(protocol, target_config, cohort, matrices, v2_raw))
        v2._finite_json(checkpoint)
        v2._atomic_json(checkpoint_path, checkpoint)
        print(f"real {len(checkpoint['real_panels'])}/{len(protocol['targets'])}: {target_key}", flush=True)

    completed_synthetic = {(row["family"], int(row["seed"])) for row in checkpoint["synthetic_panels"]}
    total_synthetic = int(protocol["expected_grid"]["synthetic_panel_count"])
    for family in protocol["synthetic_controls"]["families"]:
        for seed in protocol["synthetic_controls"]["seeds"]:
            key = (str(family), int(seed))
            if key in completed_synthetic:
                continue
            checkpoint["synthetic_panels"].append(build_synthetic_panel(*key, protocol))
            v2._finite_json(checkpoint)
            v2._atomic_json(checkpoint_path, checkpoint)
            print(f"synthetic {len(checkpoint['synthetic_panels'])}/{total_synthetic}: {key[0]}/{key[1]}", flush=True)

    validate_grid(protocol, checkpoint)
    summary = summarize(protocol, checkpoint["real_panels"], checkpoint["synthetic_panels"])
    raw = {
        "schema_version": 1,
        "status": "completed_continuous_reference_library_qualification",
        "protocol": protocol,
        "protocol_sha256": v2.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
        },
        "real_panels": checkpoint["real_panels"],
        "synthetic_panels": checkpoint["synthetic_panels"],
        "overlapindex_fit_count": 0,
        "reserved_backbones_evaluated": [],
    }
    v2._finite_json(raw)
    v2._finite_json(summary)
    raw_path = output_root / "raw_results.json"
    summary_path = output_root / "summary.json"
    report_path = output_root / "report.md"
    v2._atomic_json(raw_path, raw)
    v2._atomic_json(summary_path, summary)
    v2._atomic_bytes(report_path, render_report(summary).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_reference_library_qualification",
        "protocol_sha256": v2.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "checkpoint_identity": identity,
        "artifacts": {
            "checkpoint.json": v2.sha256_path(checkpoint_path),
            "raw_results.json": v2.sha256_path(raw_path),
            "summary.json": v2.sha256_path(summary_path),
            "report.md": v2.sha256_path(report_path),
        },
        "counts": {
            "real_panels": len(checkpoint["real_panels"]),
            "real_candidate_draws": sum(len(candidate["draws"]) for panel in checkpoint["real_panels"] for candidate in panel["candidates"]),
            "synthetic_panels": len(checkpoint["synthetic_panels"]),
            "synthetic_candidates": sum(len(panel["candidates"]) for panel in checkpoint["synthetic_panels"]),
            "overlapindex_fits": 0,
            "reserved_backbones": 0,
        },
        "git": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        },
        "decision": summary["decision"],
    }
    v2._atomic_json(output_root / "manifest.json", manifest)
    terminal = {
        "schema_version": 1,
        "status": "completed_continuous_reference_library_qualification",
        "manifest_sha256": v2.sha256_path(output_root / "manifest.json"),
        "summary_sha256": v2.sha256_path(summary_path),
        "decision": summary["decision"],
    }
    v2._atomic_json(terminal_path, terminal)
    return validate_artifacts(output_root)


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    protocol = load_protocol()
    terminal = json.loads((output_root / "terminal.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output_root / "checkpoint.json").read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "completed_continuous_reference_library_qualification"
        or terminal.get("manifest_sha256") != v2.sha256_path(output_root / "manifest.json")
        or terminal.get("summary_sha256") != v2.sha256_path(output_root / "summary.json")
        or manifest.get("protocol_sha256") != v2.sha256_path(PROTOCOL_PATH)
        or manifest.get("source_hashes") != _source_hashes(protocol)
        or terminal.get("decision") != summary.get("decision")
    ):
        raise ValueError("reference-library terminal identity changed")
    for name, expected in manifest["artifacts"].items():
        if v2.sha256_path(output_root / name) != expected:
            raise ValueError(f"reference-library artifact changed: {name}")
    validate_grid(protocol, checkpoint)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume)
    print(v2.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

