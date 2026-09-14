"""Prepare the frozen LSP wrist-span fresh evaluation cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from PIL import Image

from experiments.continuous_fresh_structured_opportunity_discovery import prepare as lsp_parent
from experiments.continuous_large_structured_opportunity_benchmark import prepare as source
from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_ranking_reliability_recoverability import prepare as prior


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_early_mean_lsp_holdout_v1" / "inputs"
PRIOR_COHORT_PATH = prior.DEFAULT_OUTPUT_ROOT / "cohort.json"
PRIOR_MANIFEST_PATH = prior.DEFAULT_OUTPUT_ROOT / "manifest.json"
PRIOR_RESULTS_ROOT = ROOT / "artifacts" / "continuous_ranking_reliability_recoverability" / "results"
FACTORIAL_ROOT = ROOT / "artifacts" / "continuous_intermediate_pooling_factorial_v2" / "audit"
TARGET_ID = "wrist_span_over_torso_length"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _source_paths() -> dict[str, Path]:
    return {
        "factorial_protocol": ROOT / "experiments" / "continuous_intermediate_pooling_factorial_v2" / "protocol.json",
        "factorial_terminal": FACTORIAL_ROOT / "terminal.json",
        "factorial_summary": FACTORIAL_ROOT / "summary.json",
        "prior_lsp_input_cohort": PRIOR_COHORT_PATH,
        "prior_lsp_input_manifest": PRIOR_MANIFEST_PATH,
        "prior_lsp_result_summary": PRIOR_RESULTS_ROOT / "summary.json",
        "prior_lsp_result_terminal": PRIOR_RESULTS_ROOT / "terminal.json",
        "lsp_joints": source.LSP_JOINTS,
        "lsp_parent_manifest": lsp_parent.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "source_pose_recipe": Path(source.__file__).resolve(),
        "shared_selector": ROOT / "experiments" / "continuous_bounded_shared_evaluator" / "shared.py",
        "continuous_overlap_index": ROOT / "overlapindex" / "ContinuousOverlapIndex.py",
    }


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or base.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("early-mean LSP holdout protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_early_mean_lsp_holdout_v1_before_outcomes":
        raise ValueError("early-mean LSP holdout protocol is not frozen")
    if protocol["primary_view"] != "early_mean" or tuple(protocol["views"]) != (
        "early_mean", "early_mean_std", "middle_mean", "middle_mean_std", "late_mean", "late_mean_std",
    ):
        raise ValueError("early-mean LSP holdout view contract changed")
    observed = {name: base.sha256_path(path) for name, path in _source_paths().items()}
    for name, digest in observed.items():
        if digest != protocol["ancestry"][f"{name}_sha256"]:
            raise ValueError(f"early-mean LSP holdout ancestry changed: {name}")
    return protocol


def _selection_key(seed: int, split: str, sample_id: str) -> tuple[str, str]:
    payload = f"{seed}|lsp_2000_pose_geometry|{TARGET_ID}|{split}|{sample_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), sample_id


def _select_evaluation(
    records: Sequence[Mapping[str, Any]], used_ids: set[str], *, count: int, bins: int, seed: int,
) -> tuple[list[dict[str, Any]], int]:
    eligible = [
        row for row in records
        if row["source_split"] == "evaluation"
        and TARGET_ID in row["targets"]
        and math.isfinite(float(row["targets"][TARGET_ID]))
        and str(row["sample_id"]) not in used_ids
    ]
    if count % bins or len(eligible) < count:
        raise ValueError(f"fresh LSP evaluation capacity changed: {len(eligible)} available for {count}")
    ordered = sorted(eligible, key=lambda row: (float(row["targets"][TARGET_ID]), str(row["sample_id"])))
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for position, row in enumerate(ordered):
        grouped[min(bins - 1, position * bins // len(ordered))].append(row)
    selected = []
    per_bin = count // bins
    for target_bin in range(bins):
        candidates = sorted(
            grouped[target_bin],
            key=lambda row: _selection_key(seed, "evaluation", str(row["sample_id"])),
        )
        if len(candidates) < per_bin:
            raise ValueError(f"fresh LSP evaluation target bin {target_bin} is too small")
        selected.extend(
            {
                "sample_id": str(row["sample_id"]),
                "source_index": int(row["source_index"]),
                "split": "evaluation",
                "target_bin": target_bin,
                "target": float(row["targets"][TARGET_ID]),
            }
            for row in candidates[:per_bin]
        )
    return selected, len(eligible)


def prepare_inputs(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("early-mean LSP holdout inputs already exist")
    previous = prior.load_cohort()
    prior_rows = previous["datasets"][source.LSP_ID]["target_cohorts"][TARGET_ID]
    training = sorted(
        (row for row in prior_rows if row["split"] == "training"),
        key=lambda row: int(row["budget_rank"]),
    )
    previous_evaluation = [row for row in prior_rows if row["split"] == "evaluation"]
    target = protocol["target"]
    if len(training) != int(target["training_pool_rows"]) or len(previous_evaluation) != 200:
        raise ValueError("prior LSP wrist cohort size changed")
    source_records = source.lsp_records()
    source_by_id = {str(row["sample_id"]): row for row in source_records}
    training_rows = []
    for row in training:
        original = source_by_id[str(row["sample_id"])]
        value = float(original["targets"][TARGET_ID])
        if abs(value - float(row["target"])) > 1.0e-12:
            raise ValueError("prior LSP wrist target value changed")
        training_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "source_index": int(row["source_index"]),
                "split": "training",
                "budget_rank": int(row["budget_rank"]),
                "target_bin": int(row["target_bin"]),
                "target": value,
            }
        )
    used_evaluation_ids = {str(row["sample_id"]) for row in previous_evaluation}
    evaluation_rows, available = _select_evaluation(
        source_records,
        used_evaluation_ids,
        count=int(target["evaluation_rows"]),
        bins=int(target["target_bins"]),
        seed=int(target["evaluation_selection_seed"]),
    )
    if used_evaluation_ids.intersection(str(row["sample_id"]) for row in evaluation_rows):
        raise ValueError("fresh LSP evaluation overlaps the prior wrist evaluation")
    all_rows = training_rows + evaluation_rows
    parent = lsp_parent.load_cohort(lsp_parent.DEFAULT_INPUT_ROOT)["datasets"]["lsp_2000_pose"]
    image_by_id = {
        str(row["sample_id"]): lsp_parent.DEFAULT_INPUT_ROOT / str(row["image_path"])
        for row in parent["rows"]
    }
    rows = []
    for matrix_index, row in enumerate(all_rows):
        image_path = image_by_id[str(row["sample_id"])]
        if not image_path.is_file():
            raise FileNotFoundError(f"LSP image is missing: {row['sample_id']}")
        with Image.open(image_path) as image:
            image.verify()
        rows.append(
            {
                **row,
                "matrix_index": matrix_index,
                "image_path": str(image_path.resolve().relative_to(ROOT)),
                "image_sha256": base.sha256_path(image_path),
            }
        )
    cohort = {
        "schema_version": 1,
        "status": "prepared_continuous_early_mean_lsp_holdout_v1",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "target_key": target["target_key"],
        "rows": rows,
        "prior_wrist_evaluation_sample_ids_sha256": base.sha256_payload(sorted(used_evaluation_ids)),
        "selected_evaluation_sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in evaluation_rows]),
    }
    output_root.mkdir(parents=True, exist_ok=False)
    cohort_path = output_root / "cohort.json"
    _atomic_json(cohort_path, cohort)
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_lsp_holdout_v1_preparation",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "cohort_sha256": base.sha256_path(cohort_path),
        "cohort_identity_sha256": base.sha256_payload(cohort),
        "source_hashes": {name: base.sha256_path(path) for name, path in _source_paths().items()},
        "counts": {
            "training_pool_rows": len(training_rows),
            "prior_evaluation_rows_excluded": len(previous_evaluation),
            "unused_valid_evaluation_rows_available": available,
            "fresh_evaluation_rows": len(evaluation_rows),
            "total_rows": len(rows),
            "prior_evaluation_overlap": 0,
        },
        "git": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        },
        "predictive_outcomes_inspected": False,
        "overlapindex_fit_count": 0,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    return manifest


def load_cohort(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    output_root = output_root.resolve()
    cohort_path, manifest_path = output_root / "cohort.json", output_root / "manifest.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_rows = int(protocol["expected_grid"]["cohort_rows"])
    if (
        cohort.get("status") != "prepared_continuous_early_mean_lsp_holdout_v1"
        or cohort.get("protocol_sha256") != base.sha256_path(PROTOCOL_PATH)
        or manifest.get("status") != "completed_continuous_early_mean_lsp_holdout_v1_preparation"
        or manifest.get("protocol_sha256") != base.sha256_path(PROTOCOL_PATH)
        or manifest.get("cohort_sha256") != base.sha256_path(cohort_path)
        or manifest.get("cohort_identity_sha256") != base.sha256_payload(cohort)
        or manifest.get("source_hashes") != {name: base.sha256_path(path) for name, path in _source_paths().items()}
        or len(cohort.get("rows", [])) != expected_rows
        or int(manifest.get("counts", {}).get("prior_evaluation_overlap", -1)) != 0
        or int(manifest.get("overlapindex_fit_count", -1)) != 0
    ):
        raise ValueError("early-mean LSP holdout prepared identity changed")
    rows = cohort["rows"]
    training = [row for row in rows if row["split"] == "training"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    if (
        rows != training + evaluation
        or len(training) != int(protocol["target"]["training_pool_rows"])
        or len(evaluation) != int(protocol["target"]["evaluation_rows"])
        or [int(row["matrix_index"]) for row in rows] != list(range(expected_rows))
        or len({str(row["sample_id"]) for row in rows}) != expected_rows
    ):
        raise ValueError("early-mean LSP holdout row grid changed")
    return cohort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = load_cohort(args.output_root) if args.validate_only else prepare_inputs(args.output_root)
    print(canonical_json(result if not args.validate_only else {"status": result["status"], "rows": len(result["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
