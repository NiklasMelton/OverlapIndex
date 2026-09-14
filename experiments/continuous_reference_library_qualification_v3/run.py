"""Run continuous reference-library qualification V3."""

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

from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_reference_library_qualification import run as q1
from experiments.continuous_reference_library_qualification_v2 import run as q2


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_reference_library_qualification_v3" / "pilot"
Q2_ROOT = ROOT / "artifacts" / "continuous_reference_library_qualification_v2" / "pilot"
BACKBONES = q1.BACKBONES
HEADS = q1.HEADS
SYNTHETIC_CANDIDATES = q1.SYNTHETIC_CANDIDATES


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or base.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("reference-library qualification V3 protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_reference_library_qualification_v3_before_outcomes":
        raise ValueError("reference-library qualification V3 is not frozen")
    if tuple(protocol["backbones"]) != BACKBONES or tuple(protocol["heads"]["families"]) != HEADS:
        raise ValueError("reference-library qualification V3 roster changed")
    if int(protocol["heads"]["knn"]["structural_maximum"]) > min(
        int(row["training_rows_per_draw"]) * 4 // 5 for row in protocol["targets"]
    ):
        raise ValueError("reference-library V3 structural k exceeds an inner fold")
    return protocol


def _source_paths() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL_PATH,
        "protocol_sidecar": PROTOCOL_SHA256_PATH,
        "runner": Path(__file__).resolve(),
        "shared_evaluator": q1.PACKAGE_DIR / "evaluator.py",
        "q1_runner": q1.PACKAGE_DIR / "run.py",
        "q2_runner": q2.PACKAGE_DIR / "run.py",
        "q2_protocol": q2.PROTOCOL_PATH,
        "q2_raw_results": Q2_ROOT / "raw_results.json",
        "q2_manifest": Q2_ROOT / "manifest.json",
    }


def _source_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"reference-library V3 provenance input is missing: {missing}")
    observed = {name: base.sha256_path(path) for name, path in paths.items()}
    ancestry = protocol["ancestry"]
    expected = {
        "q2_protocol": ancestry["qualification_v2_protocol_sha256"],
        "q2_raw_results": ancestry["qualification_v2_raw_results_sha256"],
        "q2_manifest": ancestry["qualification_v2_manifest_sha256"],
    }
    if any(observed[key] != value for key, value in expected.items()):
        raise ValueError("reference-library V3 ancestry identity changed")
    q2.validate_artifacts(Q2_ROOT)
    return observed


def collect_material_endpoints(
    real_panels: Sequence[Mapping[str, Any]],
    synthetic_panels: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for panel in real_panels:
        for candidate in panel["candidates"]:
            for draw in candidate["draws"]:
                for head, result in draw["heads"].items():
                    for item in result["endpoint_diagnostics"]:
                        if item["material"]:
                            rows.append({
                                "domain": "real", "panel": panel["target_key"],
                                "candidate": candidate["candidate_id"], "draw_seed": int(draw["draw_seed"]),
                                "head": head, "head_inner_best_r2": float(result["inner_best_r2"]), **item,
                            })
    for panel in synthetic_panels:
        for candidate in panel["candidates"]:
            for head, result in candidate["heads"].items():
                for item in result["endpoint_diagnostics"]:
                    if item["material"]:
                        rows.append({
                            "domain": "synthetic", "panel": f"{panel['family']}/{panel['seed']}",
                            "family": panel["family"], "candidate": candidate["candidate_id"],
                            "seed": int(panel["seed"]), "head": head,
                            "head_inner_best_r2": float(result["inner_best_r2"]), **item,
                        })
    return rows


def classify_material_endpoints(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = protocol["grid_adequacy"]
    nulls = set(config["diagnostic_only_synthetic_candidates"])
    always = {(row["family"], row["candidate"], row["head"]) for row in config["always_actionable_synthetic_cells"]}
    structural = config["structural_endpoints"]
    threshold = float(config["minimum_actionable_inner_cv_r2"])
    actionable, diagnostic = [], []
    for original in rows:
        row = dict(original)
        required = row["domain"] == "synthetic" and (row.get("family"), row["candidate"], row["head"]) in always
        reason = None
        if row["domain"] == "synthetic" and row["candidate"] in nulls and not required:
            reason = "predeclared_known_null_candidate"
        elif any(
            row["head"] == item["head"] and row["parameter"] == item["parameter"]
            and row["endpoint"] == item["endpoint"] and float(row["selected_value"]) == float(item["value"])
            for item in structural
        ) and not required:
            reason = "predeclared_structural_endpoint"
        elif float(row["head_inner_best_r2"]) < threshold and not required:
            reason = "predeclared_unlearnable_head_cell"
        if reason is None:
            actionable.append(row)
        else:
            row["non_actionable_reason"] = reason
            diagnostic.append(row)
    return actionable, diagnostic


def summarize(protocol, real_panels, synthetic_panels):
    summary = q1.summarize(protocol, real_panels, synthetic_panels)
    all_material = collect_material_endpoints(real_panels, synthetic_panels)
    actionable, diagnostic = classify_material_endpoints(all_material, protocol)
    summary.update({
        "reported_material_endpoint_selection_count": len(all_material),
        "actionable_material_endpoint_selection_count": len(actionable),
        "non_actionable_material_endpoint_selection_count": len(diagnostic),
        "material_endpoint_selection_count": len(actionable),
        "material_endpoint_rows": actionable,
        "non_actionable_material_endpoint_rows": diagnostic,
        "status": "completed_continuous_reference_library_qualification_v3",
    })
    decision = summary["decision"]
    endpoint_pass = len(actionable) <= int(protocol["grid_adequacy"]["maximum_actionable_material_endpoint_selections"])
    decision["grid_endpoints_adequate"] = endpoint_pass
    passed = bool(endpoint_pass and decision["synthetic_controls_pass"] and decision["cub_linear_controls_pass"] and decision["lsp_nonlinear_head_gain_pass"] and decision["warning_gate_pass"])
    decision["qualification_pass"] = passed
    decision["next_stage"] = "eight_target_historical_screen" if passed else "blocked_revisit_head_formulation"
    return summary


def render_report(summary):
    report = q2.render_report(summary)
    if summary["non_actionable_material_endpoint_rows"]:
        report += "\n## Diagnostic-only material endpoints\n\n| Domain | Panel | Candidate | Head | Parameter | Inner R2 | Reason |\n|---|---|---|---|---|---:|---|\n"
        for row in summary["non_actionable_material_endpoint_rows"]:
            report += f"| {row['domain']} | {row['panel']} | {row['candidate']} | {row['head']} | {row['parameter']} | {row['head_inner_best_r2']:.4f} | {row['non_actionable_reason']} |\n"
    return report


def _identity(protocol, hashes):
    return {"protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": dict(hashes), "targets": [row["target_key"] for row in protocol["targets"]], "synthetic_families": list(protocol["synthetic_controls"]["families"]), "synthetic_seeds": list(protocol["synthetic_controls"]["seeds"])}


def _checkpoint(path, identity, resume):
    if path.exists():
        if not resume:
            raise FileExistsError("reference-library V3 checkpoint exists; use --resume")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("identity") != identity:
            raise ValueError("reference-library V3 checkpoint identity changed")
        return value
    return {"schema_version": 1, "identity": dict(identity), "real_panels": [], "synthetic_panels": []}


def validate_grid(protocol, checkpoint):
    q1.validate_grid(protocol, checkpoint)


def run(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False):
    output_root = output_root.resolve()
    protocol = load_protocol()
    hashes = _source_hashes(protocol)
    terminal_path = output_root / "terminal.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("reference-library V3 terminal exists; use --resume or --validate-only")
        return validate_artifacts(output_root)
    checkpoint_path = output_root / "checkpoint.json"
    checkpoint = _checkpoint(checkpoint_path, _identity(protocol, hashes), resume)
    cohort, matrices = q1.v1_extract.load_bank()
    corrected_raw = json.loads((q1.V2_ROOT / "raw_results.json").read_text(encoding="utf-8"))
    complete = {row["target_key"] for row in checkpoint["real_panels"]}
    for config in protocol["targets"]:
        if config["target_key"] not in complete:
            checkpoint["real_panels"].append(q1.build_real_panel(protocol, config, cohort, matrices, corrected_raw))
            base._finite_json(checkpoint); base._atomic_json(checkpoint_path, checkpoint)
            print(f"real {len(checkpoint['real_panels'])}/{len(protocol['targets'])}: {config['target_key']}", flush=True)
    complete_synth = {(row["family"], int(row["seed"])) for row in checkpoint["synthetic_panels"]}
    total = int(protocol["expected_grid"]["synthetic_panel_count"])
    for family in protocol["synthetic_controls"]["families"]:
        for seed in protocol["synthetic_controls"]["seeds"]:
            key = (str(family), int(seed))
            if key not in complete_synth:
                checkpoint["synthetic_panels"].append(q1.build_synthetic_panel(*key, protocol))
                base._finite_json(checkpoint); base._atomic_json(checkpoint_path, checkpoint)
                print(f"synthetic {len(checkpoint['synthetic_panels'])}/{total}: {key[0]}/{key[1]}", flush=True)
    validate_grid(protocol, checkpoint)
    summary = summarize(protocol, checkpoint["real_panels"], checkpoint["synthetic_panels"])
    raw = {"schema_version": 1, "status": "completed_continuous_reference_library_qualification_v3", "protocol": protocol, "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": hashes, "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "thread_environment": {key: os.environ.get(key) for key in base.THREAD_KEYS}}, "real_panels": checkpoint["real_panels"], "synthetic_panels": checkpoint["synthetic_panels"], "overlapindex_fit_count": 0, "reserved_backbones_evaluated": []}
    base._finite_json(raw); base._finite_json(summary)
    raw_path, summary_path, report_path = output_root / "raw_results.json", output_root / "summary.json", output_root / "report.md"
    base._atomic_json(raw_path, raw); base._atomic_json(summary_path, summary); base._atomic_bytes(report_path, render_report(summary).encode("utf-8"))
    manifest = {"schema_version": 1, "status": "completed_continuous_reference_library_qualification_v3", "protocol_sha256": base.sha256_path(PROTOCOL_PATH), "source_hashes": hashes, "checkpoint_identity": _identity(protocol, hashes), "artifacts": {name: base.sha256_path(output_root / name) for name in ("checkpoint.json", "raw_results.json", "summary.json", "report.md")}, "counts": {"real_panels": len(checkpoint["real_panels"]), "real_candidate_draws": sum(len(candidate["draws"]) for panel in checkpoint["real_panels"] for candidate in panel["candidates"]), "synthetic_panels": len(checkpoint["synthetic_panels"]), "synthetic_candidates": sum(len(panel["candidates"]) for panel in checkpoint["synthetic_panels"]), "overlapindex_fits": 0, "reserved_backbones": 0}, "git": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())}, "decision": summary["decision"]}
    base._atomic_json(output_root / "manifest.json", manifest)
    terminal = {"schema_version": 1, "status": "completed_continuous_reference_library_qualification_v3", "manifest_sha256": base.sha256_path(output_root / "manifest.json"), "summary_sha256": base.sha256_path(summary_path), "decision": summary["decision"]}
    base._atomic_json(terminal_path, terminal)
    return validate_artifacts(output_root)


def validate_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT):
    output_root = output_root.resolve(); protocol = load_protocol()
    terminal = json.loads((output_root / "terminal.json").read_text()); manifest = json.loads((output_root / "manifest.json").read_text()); summary = json.loads((output_root / "summary.json").read_text()); checkpoint = json.loads((output_root / "checkpoint.json").read_text())
    if terminal.get("status") != "completed_continuous_reference_library_qualification_v3" or terminal.get("manifest_sha256") != base.sha256_path(output_root / "manifest.json") or terminal.get("summary_sha256") != base.sha256_path(output_root / "summary.json") or manifest.get("protocol_sha256") != base.sha256_path(PROTOCOL_PATH) or manifest.get("source_hashes") != _source_hashes(protocol) or terminal.get("decision") != summary.get("decision"):
        raise ValueError("reference-library V3 terminal identity changed")
    for name, expected in manifest["artifacts"].items():
        if base.sha256_path(output_root / name) != expected:
            raise ValueError(f"reference-library V3 artifact changed: {name}")
    validate_grid(protocol, checkpoint)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT); parser.add_argument("--resume", action="store_true"); parser.add_argument("--validate-only", action="store_true"); args = parser.parse_args(argv)
    result = validate_artifacts(args.output_root) if args.validate_only else run(args.output_root, resume=args.resume)
    print(base.canonical_json(result)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

