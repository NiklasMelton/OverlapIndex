"""Versioned analysis-only correction for the fused Food screen.

The frozen runner stores LP fold scoring clocks as ``predict_*``.  The original
analysis verifier incorrectly required the OI-only ``score_fixed_*`` names for
all methods.  This module adds those two LP aliases in memory solely for the
original verifier, preserving the raw artifact, estimands, gates, and decision
logic byte-for-byte.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.scalable_relevance_kmeans import fused_food_analysis as original
from experiments.scalable_relevance_kmeans import fused_food_screen as screen


CORRECTION_ID = "lp_predict_clock_schema_alias_only_v1"


def _with_lp_timing_aliases(raw: Mapping[str, Any]) -> dict[str, Any]:
    corrected = copy.deepcopy(dict(raw))
    for row in corrected.get("selector_rows", ()):
        if not isinstance(row, dict) or row.get("candidate_id") != "LP-FULL":
            continue
        for fold in row.get("folds", ()):
            if not isinstance(fold, dict):
                continue
            if "score_fixed_wall_seconds" in fold or "score_fixed_cpu_seconds" in fold:
                raise RuntimeError("LP fold unexpectedly already carries OI timing aliases")
            fold["score_fixed_wall_seconds"] = fold.get("predict_wall_seconds")
            fold["score_fixed_cpu_seconds"] = fold.get("predict_cpu_seconds")
    return corrected


def _verify_recorded_sources(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = raw.get("run_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("fused Food run identity is malformed")
    sources = identity.get("source_identity")
    if not isinstance(sources, Mapping):
        raise RuntimeError("fused Food recorded source identity is malformed")
    local = sources.get("local_source_hashes")
    if not isinstance(local, Mapping) or not local:
        raise RuntimeError("fused Food recorded local source hashes are missing")
    observed = {str(path): screen.sha256_path(Path(path)) for path in local}
    if observed != dict(local):
        raise RuntimeError("fused Food executed source bytes changed after execution")
    if sources.get("code_identity_sha256") != screen._hash_payload(local):
        raise RuntimeError("fused Food recorded code identity is malformed")
    if sources.get("development_evidence_hashes") != screen._external_evidence():
        raise RuntimeError("fused Food development evidence changed after execution")
    archived = {
        "driver": screen.sha256_path(Path(screen.food101.DEFAULT_DRIVER)),
        "source_result": screen.sha256_path(Path(screen.food101.DEFAULT_RESULT)),
        "source_cohort": screen.sha256_path(Path(screen.food101.DEFAULT_COHORT)),
        "prior_replay": screen.sha256_path(Path(screen.food101.DEFAULT_PRIOR)),
    }
    if sources.get("archived_source_hashes") != archived:
        raise RuntimeError("fused Food archived sources changed after execution")
    return sources


def verify_artifact(input_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    manifest = original._read_json(input_dir / "manifest.json")
    raw_path = input_dir / "raw_results.json"
    raw = original._read_json(raw_path)
    if manifest.get("raw_results_sha256") != screen.sha256_path(raw_path):
        raise RuntimeError("fused Food raw-results hash mismatch before correction")
    sources = _verify_recorded_sources(raw)
    corrected = _with_lp_timing_aliases(raw)

    original_reader = original._read_json
    original_source_identity = screen.source_identity

    def read_for_validation(path: Path) -> dict[str, Any]:
        candidate = Path(path)
        if candidate == raw_path:
            return copy.deepcopy(corrected)
        return original_reader(candidate)

    try:
        original._read_json = read_for_validation
        screen.source_identity = lambda: dict(sources)
        return original.verify_artifact(input_dir)
    finally:
        original._read_json = original_reader
        screen.source_identity = original_source_identity


def write_analysis(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite non-empty corrected analysis output")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = verify_artifact(Path(input_dir))
    summary = original.analyze(raw)
    summary["input_raw_results_sha256"] = screen.sha256_path(
        Path(input_dir) / "raw_results.json"
    )
    summary["protocol_sha256"] = raw["run_identity"]["protocol_sha256"]
    summary["analysis_correction"] = {
        "correction_id": CORRECTION_ID,
        "scope": "LP fold timing field validation only",
        "outcomes_changed": False,
        "estimands_gates_and_selection_changed": False,
        "original_analysis_sha256": screen.sha256_path(Path(original.__file__)),
        "corrective_analysis_sha256": screen.sha256_path(Path(__file__)),
    }
    summary_path = output_dir / "summary.json"
    decision_path = output_dir / "decision.json"
    report_path = output_dir / "report.md"
    original._atomic_json(summary_path, summary)
    original._atomic_json(decision_path, summary["decision"])
    report = original._report(summary)
    report += (
        "\n## Analysis correction\n\n"
        "The frozen raw artifact is unchanged. A versioned verifier correction maps "
        "LP fold `predict_wall_seconds`/`predict_cpu_seconds` to the verifier's "
        "OI-specific timing names in memory only. Scores, gates, and selection logic "
        "are unchanged.\n"
    )
    original._atomic_text(report_path, report)
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "study": summary["study"],
        "correction_id": CORRECTION_ID,
        "input_raw_results_sha256": summary["input_raw_results_sha256"],
        "protocol_sha256": summary["protocol_sha256"],
        "original_analysis_sha256": summary["analysis_correction"][
            "original_analysis_sha256"
        ],
        "corrective_analysis_sha256": summary["analysis_correction"][
            "corrective_analysis_sha256"
        ],
        "summary_sha256": screen.sha256_path(summary_path),
        "decision_sha256": screen.sha256_path(decision_path),
        "report_sha256": screen.sha256_path(report_path),
        "decision": summary["decision"],
    }
    original._atomic_json(output_dir / "manifest.json", manifest)
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
