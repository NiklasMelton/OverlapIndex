"""Reconstruct the frozen COCO input cohort without outcome dependencies.

The original COCO planner unnecessarily requires an ignored prior-results
summary before it will build an outcome-free image plan.  That summary was
lost with the temporary experiment checkout.  This recovery helper validates
the frozen COCO protocol, annotation archive, constants, and exact planning
function, but deliberately does not recreate or spoof the missing outcome
artifact.  It writes the standard parent cohort schema consumed by this
diagnostic and records the recovery condition in its manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from experiments.continuous_coco_pose_opportunity_replication import download as coco_download
from experiments.continuous_coco_pose_opportunity_replication import prepare as coco_prepare


def _protocol() -> dict[str, Any]:
    expected = coco_prepare.PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or coco_prepare.sha256_path(coco_prepare.PROTOCOL_PATH) != expected:
        raise ValueError("COCO recovery protocol SHA-256 mismatch")
    protocol = json.loads(coco_prepare.PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "frozen_coco_pose_opportunity_replication_before_outcomes"
        or tuple(protocol["dataset"]["targets"]) != coco_prepare.TARGETS
        or int(protocol["cohort"]["selection_seed"]) != coco_prepare.SELECTION_SEED
        or tuple(protocol["cohort"]["replicate_seeds"]) != coco_prepare.REPLICATE_SEEDS
        or int(protocol["cohort"]["training_rows_per_replicate"]) != coco_prepare.TRAIN_ROWS
        or int(protocol["cohort"]["evaluation_rows"]) != coco_prepare.EVALUATION_ROWS
    ):
        raise ValueError("COCO recovery protocol contract changed")
    return protocol


def acquire(
    input_root: Path = coco_prepare.DEFAULT_INPUT_ROOT,
    output_root: Path = coco_prepare.DEFAULT_OUTPUT_ROOT,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer from 1 through 16")
    protocol = _protocol()
    input_root, output_root = input_root.resolve(), output_root.resolve()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        return json.loads(terminal.read_text(encoding="utf-8"))
    archive = coco_prepare.DEFAULT_RAW_ROOT / str(protocol["dataset"]["annotation_archive"])
    if not archive.is_file() or coco_prepare.sha256_path(archive) != protocol["dataset"]["annotation_archive_sha256"]:
        raise ValueError("COCO recovery annotation identity changed")
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path, plan_manifest_path = output_root / "plan.json", output_root / "plan_manifest.json"
    if plan_path.exists() != plan_manifest_path.exists():
        raise ValueError("COCO recovery plan is partial")
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
        if (
            plan_manifest.get("plan_sha256") != coco_prepare.sha256_path(plan_path)
            or plan_manifest.get("plan_identity_sha256") != coco_prepare.sha256_payload(plan)
        ):
            raise ValueError("COCO recovery plan identity changed")
    else:
        plan = coco_prepare.build_plan(archive)
        coco_prepare._atomic_json(plan_path, plan)
        plan_manifest = {
            "schema_version": 1,
            "status": "completed_coco_pose_cohort_planning",
            "protocol_sha256": coco_prepare.sha256_path(coco_prepare.PROTOCOL_PATH),
            "annotation_archive_sha256": coco_prepare.sha256_path(archive),
            "plan_sha256": coco_prepare.sha256_path(plan_path),
            "plan_identity_sha256": coco_prepare.sha256_payload(plan),
            "image_count": len(plan["rows"]),
            "target_cohort_rows": {target: len(rows) for target, rows in plan["target_cohorts"].items()},
            "predictive_outcomes_inspected": False,
            "recovery_note": "prior outcome summary unavailable; frozen input planner validated directly",
        }
        coco_prepare._atomic_json(plan_manifest_path, plan_manifest)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(coco_download._download_one, input_root, protocol, row): row
            for row in plan["rows"]
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(plan["rows"]):
                print(f"downloaded/validated {completed}/{len(plan['rows'])}", flush=True)
    hashes = {}
    for row in plan["rows"]:
        path = input_root / row["image_path"]
        if not path.is_file():
            raise FileNotFoundError(f"COCO recovery image is missing: {row['sample_id']}")
        with Image.open(path) as image:
            if image.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"COCO recovery image dimensions changed: {row['sample_id']}")
            image.verify()
        hashes[row["sample_id"]] = coco_prepare.sha256_path(path)
    cohort = dict(plan)
    cohort["status"] = "prepared_coco_pose_opportunity_cohort"
    cohort["image_content_inventory_sha256"] = coco_prepare.sha256_payload(hashes)
    cohort_path = output_root / "cohort.json"
    coco_prepare._atomic_json(cohort_path, cohort)
    manifest = {
        "schema_version": 1,
        "status": "completed_coco_pose_opportunity_preparation",
        "protocol_sha256": coco_prepare.sha256_path(coco_prepare.PROTOCOL_PATH),
        "plan_sha256": coco_prepare.sha256_path(plan_path),
        "cohort_sha256": coco_prepare.sha256_path(cohort_path),
        "cohort_identity_sha256": coco_prepare.sha256_payload(cohort),
        "image_count": len(cohort["rows"]),
        "image_content_inventory_sha256": cohort["image_content_inventory_sha256"],
        "source_hashes": {"recovery_helper": coco_prepare.sha256_path(Path(__file__).resolve())},
        "predictive_outcomes_inspected": False,
        "recovery_note": "prior outcome summary unavailable; no outcome artifact was fabricated",
    }
    coco_prepare._atomic_json(terminal, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=coco_prepare.DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=coco_prepare.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    print(coco_prepare.canonical_json(acquire(args.input_root, args.output_root, workers=args.workers)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
