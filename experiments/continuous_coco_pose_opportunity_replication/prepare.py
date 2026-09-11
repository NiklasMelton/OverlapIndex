"""Plan and finalize the frozen COCO pose cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
PRIOR_SUMMARY = (
    ROOT
    / "artifacts"
    / "continuous_fresh_structured_opportunity_discovery"
    / "results"
    / "summary.json"
)
DEFAULT_INPUT_ROOT = (
    ROOT / "artifacts" / "continuous_coco_pose_opportunity_replication" / "inputs"
)
DEFAULT_RAW_ROOT = DEFAULT_INPUT_ROOT / "raw"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "prepared"
DATASET_ID = "coco_2017_person_pose"
TARGETS = (
    "torso_axis_sine",
    "torso_axis_cosine",
    "bilateral_knee_extension_cosine",
)
REPLICATE_SEEDS = (4103, 4201, 4303)
SELECTION_SEED = 2026092501
TRAIN_ROWS = 200
EVALUATION_ROWS = 500


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("COCO pose protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_coco_pose_opportunity_replication_before_outcomes":
        raise ValueError("COCO pose protocol is not frozen")
    cohort = protocol["cohort"]
    if (
        int(cohort["selection_seed"]) != SELECTION_SEED
        or tuple(cohort["replicate_seeds"]) != REPLICATE_SEEDS
        or int(cohort["training_rows_per_replicate"]) != TRAIN_ROWS
        or int(cohort["evaluation_rows"]) != EVALUATION_ROWS
    ):
        raise ValueError("COCO pose cohort contract changed")
    if tuple(protocol["dataset"]["targets"]) != TARGETS:
        raise ValueError("COCO pose target grid changed")
    prior = protocol["prior_evidence"]
    if not PRIOR_SUMMARY.is_file() or sha256_path(PRIOR_SUMMARY) != prior["summary_sha256"]:
        raise ValueError("COCO pose prior LSP evidence identity changed")
    summary = json.loads(PRIOR_SUMMARY.read_text(encoding="utf-8"))
    admitted = {
        (row["dataset_id"], target)
        for row in summary["dataset_summaries"]
        for target in row["admitted_targets"]
    }
    if (prior["admitted_dataset"], prior["admitted_target"]) not in admitted:
        raise ValueError("COCO pose prior LSP opportunity is not admitted")
    return protocol


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.shape != (2,) or not np.isfinite(values).all() or norm <= 0.0:
        raise ValueError(f"COCO {label} vector is invalid")
    return values / norm


def pose_targets(keypoints: np.ndarray) -> dict[str, float]:
    points = np.asarray(keypoints, dtype=np.float64)
    if points.shape != (17, 3) or not np.isfinite(points).all():
        raise ValueError("COCO keypoint matrix is invalid")
    hip_midpoint = 0.5 * (points[11, :2] + points[12, :2])
    shoulder_midpoint = 0.5 * (points[5, :2] + points[6, :2])
    torso = _unit(shoulder_midpoint - hip_midpoint, "torso")
    left = float(
        np.dot(
            _unit(points[11, :2] - points[13, :2], "left thigh"),
            _unit(points[15, :2] - points[13, :2], "left shin"),
        )
    )
    right = float(
        np.dot(
            _unit(points[12, :2] - points[14, :2], "right thigh"),
            _unit(points[16, :2] - points[14, :2], "right shin"),
        )
    )
    return {
        "torso_axis_sine": float(torso[1]),
        "torso_axis_cosine": float(torso[0]),
        "bilateral_knee_extension_cosine": float(-0.5 * (left + right)),
    }


def _torso_targets(points: np.ndarray) -> tuple[float, float]:
    hip_midpoint = 0.5 * (points[11, :2] + points[12, :2])
    shoulder_midpoint = 0.5 * (points[5, :2] + points[6, :2])
    torso = _unit(shoulder_midpoint - hip_midpoint, "torso")
    return float(torso[1]), float(torso[0])


def _knee_target(points: np.ndarray) -> float:
    left = float(
        np.dot(
            _unit(points[11, :2] - points[13, :2], "left thigh"),
            _unit(points[15, :2] - points[13, :2], "left shin"),
        )
    )
    right = float(
        np.dot(
            _unit(points[12, :2] - points[14, :2], "right thigh"),
            _unit(points[16, :2] - points[14, :2], "right shin"),
        )
    )
    return float(-0.5 * (left + right))


def _read_annotations(archive_path: Path, split: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        name = f"annotations/person_keypoints_{split}2017.json"
        try:
            payload = json.loads(archive.read(name))
        except KeyError as error:
            raise ValueError(f"COCO archive is missing {name}") from error
    expected_images = 118287 if split == "train" else 5000
    if len(payload.get("images", [])) != expected_images:
        raise ValueError(f"COCO {split} image grid changed")
    if not isinstance(payload.get("annotations"), list):
        raise ValueError(f"COCO {split} annotations are malformed")
    return payload


def _eligible_record(
    annotation: Mapping[str, Any], group: str, split: str, image: Mapping[str, Any]
) -> dict[str, Any] | None:
    if int(annotation.get("category_id", -1)) != 1 or bool(annotation.get("iscrowd")):
        return None
    bbox = np.asarray(annotation.get("bbox", ()), dtype=np.float64)
    keypoints = np.asarray(annotation.get("keypoints", ()), dtype=np.float64)
    if (
        bbox.shape != (4,)
        or not np.isfinite(bbox).all()
        or bbox[2] <= 0.0
        or bbox[3] <= 0.0
        or keypoints.shape != (51,)
        or not np.isfinite(keypoints).all()
    ):
        return None
    keypoints = keypoints.reshape(17, 3)
    required = (5, 6, 11, 12) if group == "torso" else (11, 12, 13, 14, 15, 16)
    if not np.all(keypoints[list(required), 2] == 2):
        return None
    try:
        if group == "torso":
            sine, cosine = _torso_targets(keypoints)
            values = {
                "torso_axis_sine": sine,
                "torso_axis_cosine": cosine,
            }
        else:
            values = {
                "bilateral_knee_extension_cosine": _knee_target(keypoints)
            }
    except ValueError:
        return None
    image_id = int(annotation["image_id"])
    if int(image["id"]) != image_id:
        raise ValueError("COCO annotation/image identity is misaligned")
    return {
        "image_id": image_id,
        "annotation_id": int(annotation["id"]),
        "source_split": split,
        "file_name": str(image["file_name"]),
        "width": int(image["width"]),
        "height": int(image["height"]),
        "bbox_area": float(bbox[2] * bbox[3]),
        "targets": values,
    }


def candidate_records(archive_path: Path) -> dict[str, list[dict[str, Any]]]:
    output = {"torso": [], "knees": []}
    for split in ("train", "val"):
        payload = _read_annotations(archive_path, split)
        images = {int(row["id"]): row for row in payload["images"]}
        if len(images) != len(payload["images"]):
            raise ValueError(f"COCO {split} image ids are duplicated")
        best: dict[str, dict[int, dict[str, Any]]] = {
            "torso": {}, "knees": {}
        }
        for annotation in payload["annotations"]:
            image_id = int(annotation.get("image_id", -1))
            image = images.get(image_id)
            if image is None:
                raise ValueError("COCO annotation refers to an unknown image")
            for group in ("torso", "knees"):
                record = _eligible_record(annotation, group, split, image)
                if record is None:
                    continue
                existing = best[group].get(image_id)
                rank = (-record["bbox_area"], record["annotation_id"])
                old_rank = None if existing is None else (-existing["bbox_area"], existing["annotation_id"])
                if old_rank is None or rank < old_rank:
                    best[group][image_id] = record
        for group in ("torso", "knees"):
            output[group].extend(best[group].values())
    return output


def _order_key(group: str, split: str, bin_id: int, record: Mapping[str, Any]) -> tuple[str, int]:
    payload = (
        f"{SELECTION_SEED}|{DATASET_ID}|{group}|{split}|{bin_id}|"
        f"{record['image_id']}|{record['annotation_id']}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), int(record["image_id"])


def _quantile_bins(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, 11))
    if not np.isfinite(edges).all() or len(np.unique(edges)) != 11:
        raise ValueError("COCO values cannot form ten distinct quantile strata")
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int64)


def _select_group(
    records: Sequence[Mapping[str, Any]], group: str, targets: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    output = {target: [] for target in targets}
    for split in ("train", "val"):
        source = [row for row in records if row["source_split"] == split]
        if group == "torso":
            values = np.asarray(
                [
                    math.atan2(
                        row["targets"]["torso_axis_sine"],
                        row["targets"]["torso_axis_cosine"],
                    )
                    for row in source
                ]
            )
        else:
            values = np.asarray(
                [row["targets"]["bilateral_knee_extension_cosine"] for row in source]
            )
        bins = _quantile_bins(values)
        for bin_id in range(10):
            pool = sorted(
                [source[index] for index in np.flatnonzero(bins == bin_id)],
                key=lambda row: _order_key(group, split, bin_id, row),
            )
            required = 60 if split == "train" else 50
            if len(pool) < required:
                raise ValueError(f"COCO {group}/{split}/bin-{bin_id} cannot support the frozen cohort")
            if split == "val":
                assignments = [(row, None) for row in pool[:50]]
            else:
                assignments = []
                for seed_position, seed in enumerate(REPLICATE_SEEDS):
                    assignments.extend(
                        (row, seed)
                        for row in pool[20 * seed_position : 20 * seed_position + 20]
                    )
            for target in targets:
                output[target].extend(
                    {
                        "image_id": int(row["image_id"]),
                        "annotation_id": int(row["annotation_id"]),
                        "source_split": split,
                        "replicate_seed": seed,
                        "target": float(row["targets"][target]),
                    }
                    for row, seed in assignments
                )
    for target, rows in output.items():
        if len(rows) != EVALUATION_ROWS + TRAIN_ROWS * len(REPLICATE_SEEDS):
            raise RuntimeError(f"COCO {target} selected cohort size changed")
        if len({(row["source_split"], row["image_id"]) for row in rows}) != len(rows):
            raise RuntimeError(f"COCO {target} selected cohort is not disjoint")
    return output


def build_plan(archive_path: Path) -> dict[str, Any]:
    candidates = candidate_records(archive_path)
    cohorts = _select_group(
        candidates["torso"], "torso", ("torso_axis_sine", "torso_axis_cosine")
    )
    cohorts.update(
        _select_group(
            candidates["knees"], "knees", ("bilateral_knee_extension_cosine",)
        )
    )
    lookup = {
        (row["source_split"], int(row["image_id"])): row
        for rows in candidates.values()
        for row in rows
    }
    keys = sorted(
        {
            (row["source_split"], int(row["image_id"]))
            for rows in cohorts.values()
            for row in rows
        }
    )
    rows = []
    positions = {}
    for position, key in enumerate(keys):
        record = lookup[key]
        positions[key] = position
        split, image_id = key
        rows.append(
            {
                "sample_id": f"coco/{split}2017/{image_id:012d}",
                "source_split": split,
                "image_id": image_id,
                "file_name": record["file_name"],
                "width": int(record["width"]),
                "height": int(record["height"]),
                "image_path": f"raw/images/{split}2017/{record['file_name']}",
            }
        )
    for target_rows in cohorts.values():
        for row in target_rows:
            row["matrix_index"] = positions[(row["source_split"], row["image_id"])]
    return {
        "schema_version": 1,
        "status": "planned_coco_pose_opportunity_cohort_before_image_download",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "dataset_id": DATASET_ID,
        "candidate_support": {
            group: {
                split: sum(row["source_split"] == split for row in rows)
                for split in ("train", "val")
            }
            for group, rows in candidates.items()
        },
        "rows": rows,
        "target_cohorts": cohorts,
    }


def write_plan(
    raw_root: Path = DEFAULT_RAW_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    protocol = load_protocol()
    raw_root, output_root = raw_root.resolve(), output_root.resolve()
    archive = raw_root / protocol["dataset"]["annotation_archive"]
    if not archive.is_file() or sha256_path(archive) != protocol["dataset"]["annotation_archive_sha256"]:
        raise ValueError("COCO annotation archive identity changed")
    plan_path, manifest_path = output_root / "plan.json", output_root / "plan_manifest.json"
    if plan_path.exists() or manifest_path.exists():
        raise FileExistsError("COCO pose plan already exists")
    plan = build_plan(archive)
    _atomic_json(plan_path, plan)
    manifest = {
        "schema_version": 1,
        "status": "completed_coco_pose_cohort_planning",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "annotation_archive_sha256": sha256_path(archive),
        "plan_sha256": sha256_path(plan_path),
        "plan_identity_sha256": sha256_payload(plan),
        "image_count": len(plan["rows"]),
        "target_cohort_rows": {
            target: len(rows) for target, rows in plan["target_cohorts"].items()
        },
        "predictive_outcomes_inspected": False,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def load_plan(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    plan_path, manifest_path = output_root / "plan.json", output_root / "plan_manifest.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed_coco_pose_cohort_planning"
        or manifest.get("protocol_sha256") != sha256_path(PROTOCOL_PATH)
        or manifest.get("plan_sha256") != sha256_path(plan_path)
        or manifest.get("plan_identity_sha256") != sha256_payload(plan)
    ):
        raise ValueError("COCO pose plan identity changed")
    return plan


def finalize(
    input_root: Path = DEFAULT_INPUT_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    load_protocol()
    input_root, output_root = input_root.resolve(), output_root.resolve()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        raise FileExistsError("COCO pose terminal input manifest already exists")
    plan = load_plan(output_root)
    hashes = {}
    for row in plan["rows"]:
        path = input_root / row["image_path"]
        if not path.is_file():
            raise ValueError(f"COCO selected image is missing: {row['sample_id']}")
        with Image.open(path) as image:
            if image.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"COCO image dimensions changed: {row['sample_id']}")
            image.verify()
        hashes[row["sample_id"]] = sha256_path(path)
    cohort = dict(plan)
    cohort["status"] = "prepared_coco_pose_opportunity_cohort"
    cohort["image_content_inventory_sha256"] = sha256_payload(hashes)
    cohort_path = output_root / "cohort.json"
    _atomic_json(cohort_path, cohort)
    manifest = {
        "schema_version": 1,
        "status": "completed_coco_pose_opportunity_preparation",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "plan_sha256": sha256_path(output_root / "plan.json"),
        "cohort_sha256": sha256_path(cohort_path),
        "cohort_identity_sha256": sha256_payload(cohort),
        "image_count": len(cohort["rows"]),
        "image_content_inventory_sha256": cohort["image_content_inventory_sha256"],
        "source_hashes": {"prepare": sha256_path(__file__)},
        "predictive_outcomes_inspected": False,
    }
    _atomic_json(terminal, manifest)
    return manifest


def load_cohort(input_root: Path = DEFAULT_INPUT_ROOT) -> dict[str, Any]:
    output_root = input_root / "prepared"
    cohort_path, manifest_path = output_root / "cohort.json", output_root / "manifest.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed_coco_pose_opportunity_preparation"
        or manifest.get("protocol_sha256") != sha256_path(PROTOCOL_PATH)
        or manifest.get("cohort_sha256") != sha256_path(cohort_path)
        or manifest.get("cohort_identity_sha256") != sha256_payload(cohort)
    ):
        raise ValueError("COCO pose prepared cohort identity changed")
    return cohort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "finalize"))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = (
        write_plan(args.raw_root, args.output_root)
        if args.stage == "plan"
        else finalize(args.input_root, args.output_root)
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
