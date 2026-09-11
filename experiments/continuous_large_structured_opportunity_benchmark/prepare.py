"""Prepare the frozen 20-target, five-domain opportunity benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_INPUT_ROOT = (
    ROOT / "artifacts" / "continuous_large_structured_opportunity_benchmark" / "inputs"
)
DEFAULT_RAW_ROOT = DEFAULT_INPUT_ROOT / "raw"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "prepared"

CUB_ID = "cub_200_2011_part_geometry"
LSP_ID = "lsp_2000_pose_geometry"
COCO_ID = "coco_2017_person_pose_geometry"
PET_ID = "oxford_iiit_pet_mask_geometry"
VOC_ID = "pascal_voc_2007_largest_box_geometry"
DATASETS = (CUB_ID, LSP_ID, COCO_ID, PET_ID, VOC_ID)

CUB_TARGETS = (
    "wing_axis_sine",
    "wing_axis_cosine",
    "wing_span_over_bbox_diagonal",
    "leg_span_over_bbox_diagonal",
)
POSE_TARGETS = (
    "bilateral_elbow_extension_cosine",
    "wrist_span_over_torso_length",
    "ankle_span_over_torso_length",
    "shoulder_to_hip_width_ratio",
)
PET_TARGETS = (
    "foreground_compactness",
    "foreground_bbox_fill",
    "foreground_eccentricity",
    "foreground_boundary_roughness",
)
VOC_TARGETS = (
    "largest_box_area_fraction",
    "largest_box_center_x",
    "largest_box_center_y",
    "largest_box_log_aspect_ratio",
)
TARGETS_BY_DATASET = {
    CUB_ID: CUB_TARGETS,
    LSP_ID: POSE_TARGETS,
    COCO_ID: POSE_TARGETS,
    PET_ID: PET_TARGETS,
    VOC_ID: VOC_TARGETS,
}

REPLICATE_SEEDS = (4103, 4201, 4303)
SELECTION_SEED = 2026090401
TRAIN_ROWS = 100
EVALUATION_ROWS = 200
TARGET_BINS = 10

CUB_ARCHIVE = (
    ROOT
    / "artifacts"
    / "continuous_fresh_structured_opportunity_discovery"
    / "inputs"
    / "raw"
    / "CUB_200_2011.tgz"
)
LSP_JOINTS = (
    ROOT
    / "artifacts"
    / "continuous_fresh_structured_opportunity_discovery"
    / "inputs"
    / "raw"
    / "lsp"
    / "joints.mat"
)
LSP_PARENT_COHORT = (
    ROOT
    / "artifacts"
    / "continuous_fresh_structured_opportunity_discovery"
    / "inputs"
    / "prepared"
    / "cohort.json"
)
COCO_ANNOTATIONS = (
    ROOT
    / "artifacts"
    / "continuous_coco_pose_opportunity_replication"
    / "inputs"
    / "raw"
    / "annotations_trainval2017.zip"
)
COCO_PARENT_COHORT = (
    ROOT
    / "artifacts"
    / "continuous_coco_pose_opportunity_replication"
    / "inputs"
    / "prepared"
    / "cohort.json"
)
PET_INPUT_ROOT = ROOT / "artifacts" / "continuous_real_pet_area_panel" / "inputs"
PET_PARENT_COHORT = PET_INPUT_ROOT / "cohort.json"
VOC_TRAINVAL = DEFAULT_RAW_ROOT / "voc" / "VOCtrainval_06-Nov-2007.tar"
VOC_TEST = DEFAULT_RAW_ROOT / "voc" / "VOCtest_06-Nov-2007.tar"

CUB_MD5 = "97eceeb196236b17998738112f37df78"
VOC_TRAINVAL_MD5 = "c52e279531787c972589f7e41ab4ae64"
VOC_TEST_MD5 = "b6e924de25625d8de591ea690078ad9f"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_path(path: os.PathLike[str] | str) -> str:
    digest = hashlib.md5()  # noqa: S324 - publisher identity verification
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
        raise ValueError("large structured protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_large_structured_opportunity_benchmark_before_outcomes":
        raise ValueError("large structured opportunity protocol is not frozen")
    if tuple(protocol["datasets"]) != DATASETS:
        raise ValueError("large structured dataset order changed")
    for dataset_id in DATASETS:
        if tuple(protocol["datasets"][dataset_id]["targets"]) != TARGETS_BY_DATASET[dataset_id]:
            raise ValueError(f"large structured target grid changed: {dataset_id}")
    cohort = protocol["cohort"]
    if (
        int(cohort["selection_seed"]) != SELECTION_SEED
        or tuple(cohort["replicate_seeds"]) != REPLICATE_SEEDS
        or int(cohort["training_rows_per_replicate"]) != TRAIN_ROWS
        or int(cohort["evaluation_rows"]) != EVALUATION_ROWS
        or int(cohort["target_bins"]) != TARGET_BINS
    ):
        raise ValueError("large structured cohort contract changed")
    return protocol


def _hash_order(dataset_id: str, target_id: str, split: str, sample_id: str) -> tuple[str, str]:
    payload = f"{SELECTION_SEED}|{dataset_id}|{target_id}|{split}|{sample_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), sample_id


def _select_target(
    dataset_id: str,
    target_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    training_label: str = "training",
    evaluation_label: str = "evaluation",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_split, total in (
        (training_label, TRAIN_ROWS * len(REPLICATE_SEEDS)),
        (evaluation_label, EVALUATION_ROWS),
    ):
        eligible = [
            row
            for row in records
            if row["source_split"] == source_split
            and target_id in row["targets"]
            and math.isfinite(float(row["targets"][target_id]))
        ]
        eligible.sort(key=lambda row: (float(row["targets"][target_id]), str(row["sample_id"])))
        if len(eligible) < total:
            raise ValueError(
                f"{dataset_id}/{target_id}/{source_split} has {len(eligible)} eligible rows; {total} required"
            )
        bins: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for position, row in enumerate(eligible):
            bins[min(TARGET_BINS - 1, position * TARGET_BINS // len(eligible))].append(row)
        per_bin = total // TARGET_BINS
        if per_bin * TARGET_BINS != total:
            raise RuntimeError("frozen selection size must divide target bins")
        selected: list[Mapping[str, Any]] = []
        for target_bin in range(TARGET_BINS):
            ordered = sorted(
                bins[target_bin],
                key=lambda row: _hash_order(
                    dataset_id, target_id, source_split, str(row["sample_id"])
                ),
            )
            if len(ordered) < per_bin:
                raise ValueError(f"{dataset_id}/{target_id} target bin {target_bin} is too small")
            selected.extend(ordered[:per_bin])
        for position, row in enumerate(selected):
            if source_split == evaluation_label:
                split, seed = "evaluation", None
            else:
                split = "training"
                rows_per_seed_per_bin = TRAIN_ROWS // TARGET_BINS
                within_bin = position % per_bin
                seed = REPLICATE_SEEDS[within_bin // rows_per_seed_per_bin]
            output.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "source_index": int(row["source_index"]),
                    "split": split,
                    "replicate_seed": seed,
                    "target": float(row["targets"][target_id]),
                }
            )
    if len(output) != TRAIN_ROWS * len(REPLICATE_SEEDS) + EVALUATION_ROWS:
        raise RuntimeError("selected target cohort size changed")
    if len({row["sample_id"] for row in output}) != len(output):
        raise RuntimeError("selected target cohort overlaps between train and evaluation")
    return output


def _select_assigned_target(
    dataset_id: str,
    target_id: str,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    groups = [("evaluation", None, EVALUATION_ROWS)] + [
        ("training", seed, TRAIN_ROWS) for seed in REPLICATE_SEEDS
    ]
    for split, seed, count in groups:
        eligible = [
            row
            for row in records
            if row["source_split"] == split
            and row.get("source_replicate_seed") == seed
            and math.isfinite(float(row["targets"][target_id]))
        ]
        eligible.sort(key=lambda row: (float(row["targets"][target_id]), str(row["sample_id"])))
        if len(eligible) < count:
            raise ValueError(f"{dataset_id}/{target_id}/{split}/{seed} is too small")
        per_bin = count // TARGET_BINS
        selected = []
        for target_bin in range(TARGET_BINS):
            start = target_bin * len(eligible) // TARGET_BINS
            stop = (target_bin + 1) * len(eligible) // TARGET_BINS
            ordered = sorted(
                eligible[start:stop],
                key=lambda row: _hash_order(
                    dataset_id, target_id, f"{split}/{seed}", str(row["sample_id"])
                ),
            )
            if len(ordered) < per_bin:
                raise ValueError(f"{dataset_id}/{target_id} assigned target bin is too small")
            selected.extend(ordered[:per_bin])
        output.extend(
            {
                "sample_id": str(row["sample_id"]),
                "source_index": int(row["source_index"]),
                "split": split,
                "replicate_seed": seed,
                "target": float(row["targets"][target_id]),
            }
            for row in selected
        )
    if len(output) != TRAIN_ROWS * len(REPLICATE_SEEDS) + EVALUATION_ROWS:
        raise RuntimeError("assigned target cohort size changed")
    return output


def _tar_text(archive: tarfile.TarFile, relative: str) -> str:
    handle = archive.extractfile(f"CUB_200_2011/{relative}")
    if handle is None:
        raise ValueError(f"CUB archive is missing {relative}")
    return handle.read().decode("utf-8")


def _indexed_rows(text: str, width: int, label: str) -> dict[int, list[str]]:
    rows = {}
    for raw in text.splitlines():
        fields = raw.split()
        if len(fields) != width:
            raise ValueError(f"CUB {label} row width changed")
        index = int(fields[0])
        if index in rows:
            raise ValueError(f"CUB {label} has duplicate ids")
        rows[index] = fields[1:]
    return rows


def _pair_metrics(
    left: Sequence[float], right: Sequence[float], bbox_wh: Sequence[float]
) -> tuple[float, float, float]:
    delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    width, height = (float(value) for value in bbox_wh)
    distance = float(np.linalg.norm(delta))
    diagonal = float(math.hypot(width, height))
    if not np.isfinite(delta).all() or distance <= 0.0 or diagonal <= 0.0:
        raise ValueError("CUB paired-part geometry is invalid")
    return float(delta[1] / distance), float(delta[0] / distance), float(distance / diagonal)


def cub_records() -> list[dict[str, Any]]:
    if md5_path(CUB_ARCHIVE) != CUB_MD5:
        raise ValueError("CUB archive identity changed")
    with tarfile.open(CUB_ARCHIVE, "r:gz") as archive:
        images = _indexed_rows(_tar_text(archive, "images.txt"), 2, "images")
        splits = _indexed_rows(_tar_text(archive, "train_test_split.txt"), 2, "splits")
        boxes = _indexed_rows(_tar_text(archive, "bounding_boxes.txt"), 5, "boxes")
        parts = {}
        for raw in _tar_text(archive, "parts/part_locs.txt").splitlines():
            fields = raw.split()
            if len(fields) != 5:
                raise ValueError("CUB part row width changed")
            parts[(int(fields[0]), int(fields[1]))] = (
                float(fields[2]), float(fields[3]), int(fields[4])
            )
    if set(images) != set(range(1, 11789)) or len(parts) != 11788 * 15:
        raise ValueError("CUB metadata grid changed")
    records = []
    for image_id in sorted(images):
        _, _, width, height = (float(value) for value in boxes[image_id])
        targets = {}
        left_wing, right_wing = parts[(image_id, 9)], parts[(image_id, 13)]
        if left_wing[2] == right_wing[2] == 1:
            try:
                sine, cosine, span = _pair_metrics(left_wing[:2], right_wing[:2], (width, height))
                targets.update(
                    wing_axis_sine=sine,
                    wing_axis_cosine=cosine,
                    wing_span_over_bbox_diagonal=span,
                )
            except ValueError:
                pass
        left_leg, right_leg = parts[(image_id, 8)], parts[(image_id, 12)]
        if left_leg[2] == right_leg[2] == 1:
            try:
                _, _, span = _pair_metrics(left_leg[:2], right_leg[:2], (width, height))
                targets["leg_span_over_bbox_diagonal"] = span
            except ValueError:
                pass
        records.append(
            {
                "sample_id": f"cub/{image_id:05d}",
                "source_index": image_id,
                "source_split": "training" if int(splits[image_id][0]) == 1 else "evaluation",
                "image_member": f"CUB_200_2011/images/{images[image_id][0]}",
                "targets": targets,
            }
        )
    return records


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (2,) or not np.isfinite(vector).all() or norm <= 0.0:
        raise ValueError(f"{label} vector is invalid")
    return vector / norm


def pose_targets(points_xy: np.ndarray, visibility: np.ndarray, *, visible_code: int) -> dict[str, float]:
    points = np.asarray(points_xy, dtype=np.float64)
    flags = np.asarray(visibility)
    if points.shape[1:] != (2,) or flags.ndim != 1 or len(points) != len(flags):
        raise ValueError("pose geometry shape is invalid")
    if not np.isfinite(points).all():
        raise ValueError("pose geometry contains nonfinite coordinates")
    targets = {}
    groups = {
        "elbow": (5, 6, 7, 8, 9, 10),
        "wrist": (5, 6, 9, 10, 11, 12),
        "ankle": (5, 6, 11, 12, 15, 16),
        "width": (5, 6, 11, 12),
    }
    eligible = {key: bool(np.all(flags[list(indices)] == visible_code)) for key, indices in groups.items()}
    if eligible["elbow"]:
        try:
            left = float(np.dot(_unit(points[5] - points[7], "left upper arm"), _unit(points[9] - points[7], "left forearm")))
            right = float(np.dot(_unit(points[6] - points[8], "right upper arm"), _unit(points[10] - points[8], "right forearm")))
            targets["bilateral_elbow_extension_cosine"] = float(-0.5 * (left + right))
        except ValueError:
            pass
    shoulder_mid = 0.5 * (points[5] + points[6])
    hip_mid = 0.5 * (points[11] + points[12])
    torso = float(np.linalg.norm(shoulder_mid - hip_mid))
    if eligible["wrist"] and torso > 0.0:
        targets["wrist_span_over_torso_length"] = float(np.linalg.norm(points[9] - points[10]) / torso)
    if eligible["ankle"] and torso > 0.0:
        targets["ankle_span_over_torso_length"] = float(np.linalg.norm(points[15] - points[16]) / torso)
    hip_width = float(np.linalg.norm(points[11] - points[12]))
    if eligible["width"] and hip_width > 0.0:
        targets["shoulder_to_hip_width_ratio"] = float(np.linalg.norm(points[5] - points[6]) / hip_width)
    return {key: value for key, value in targets.items() if math.isfinite(value)}


def lsp_records() -> list[dict[str, Any]]:
    parent = json.loads(LSP_PARENT_COHORT.read_text(encoding="utf-8"))["datasets"]["lsp_2000_pose"]
    joints = np.asarray(loadmat(LSP_JOINTS)["joints"], dtype=np.float64)
    if joints.shape != (3, 14, 2000) or len(parent["rows"]) != 2000:
        raise ValueError("LSP parent identity changed")
    output = []
    for matrix_index, row in enumerate(parent["rows"]):
        source_index = int(row["source_index"])
        if matrix_index != source_index:
            raise ValueError("LSP matrix/source index alignment changed")
        raw = joints[:, :, source_index]
        # Map LSP order into the shared COCO-like 17-position geometry index.
        points = np.zeros((17, 2), dtype=np.float64)
        flags = np.ones(17, dtype=np.int64)
        mapping = {16: 0, 14: 1, 12: 2, 11: 3, 13: 4, 15: 5, 10: 6, 8: 7, 6: 8, 5: 9, 7: 10, 9: 11}
        for destination, source in mapping.items():
            points[destination] = raw[:2, source]
            flags[destination] = int(raw[2, source])
        output.append(
            {
                "sample_id": str(row["sample_id"]),
                "source_index": matrix_index,
                "source_split": str(row["source_split"]),
                "targets": pose_targets(points, flags, visible_code=0),
            }
        )
    return output


def coco_records() -> list[dict[str, Any]]:
    parent = json.loads(COCO_PARENT_COHORT.read_text(encoding="utf-8"))
    rows = parent["rows"]
    selected = {(str(row["source_split"]), int(row["image_id"])): index for index, row in enumerate(rows)}
    best: dict[tuple[str, int], tuple[tuple[float, int], np.ndarray]] = {}
    with zipfile.ZipFile(COCO_ANNOTATIONS) as archive:
        for split in ("train", "val"):
            payload = json.loads(archive.read(f"annotations/person_keypoints_{split}2017.json"))
            for annotation in payload["annotations"]:
                key = (split, int(annotation.get("image_id", -1)))
                if key not in selected or int(annotation.get("category_id", -1)) != 1 or bool(annotation.get("iscrowd")):
                    continue
                bbox = np.asarray(annotation.get("bbox", ()), dtype=np.float64)
                keypoints = np.asarray(annotation.get("keypoints", ()), dtype=np.float64)
                if bbox.shape != (4,) or keypoints.shape != (51,) or not np.isfinite(bbox).all() or not np.isfinite(keypoints).all() or min(bbox[2:]) <= 0:
                    continue
                rank = (-float(bbox[2] * bbox[3]), int(annotation["id"]))
                if key not in best or rank < best[key][0]:
                    best[key] = (rank, keypoints.reshape(17, 3))
    output = []
    for key, matrix_index in sorted(selected.items(), key=lambda item: item[1]):
        row = rows[matrix_index]
        points = best.get(key)
        targets = {} if points is None else pose_targets(points[1][:, :2], points[1][:, 2], visible_code=2)
        output.append(
            {
                "sample_id": str(row["sample_id"]),
                "source_index": matrix_index,
                "source_split": "training" if key[0] == "train" else "evaluation",
                "targets": targets,
            }
        )
    return output


def mask_targets(mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(mask)
    if values.ndim != 2 or values.size == 0 or not set(np.unique(values)).issubset({1, 2, 3}):
        raise ValueError("Oxford-IIIT Pet trimap is malformed")
    foreground = values != 2
    ys, xs = np.nonzero(foreground)
    area = int(len(xs))
    if area < 2:
        raise ValueError("Oxford-IIIT Pet foreground is degenerate")
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    padded = np.pad(foreground, 1, mode="constant", constant_values=False)
    perimeter = int(
        np.count_nonzero(padded[1:, :] != padded[:-1, :])
        + np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    )
    coords = np.column_stack((xs, ys)).astype(np.float64)
    covariance = np.cov(coords, rowvar=False, ddof=0)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if perimeter <= 0 or eigenvalues[-1] <= 0.0:
        raise ValueError("Oxford-IIIT Pet mask geometry is degenerate")
    return {
        "foreground_compactness": float(4.0 * math.pi * area / (perimeter * perimeter)),
        "foreground_bbox_fill": float(area / (width * height)),
        "foreground_eccentricity": float(math.sqrt(max(0.0, 1.0 - eigenvalues[0] / eigenvalues[-1]))),
        "foreground_boundary_roughness": float(perimeter / (2.0 * (width + height))),
    }


def pet_records() -> list[dict[str, Any]]:
    parent = json.loads(PET_PARENT_COHORT.read_text(encoding="utf-8"))
    output = []
    for matrix_index, row in enumerate(parent["rows"]):
        with Image.open(PET_INPUT_ROOT / "trimaps" / f"{row['sample_id']}.png") as source:
            targets = mask_targets(np.asarray(source))
        source_seed = row.get("replicate_seed")
        if row["split"] == "training" and source_seed not in REPLICATE_SEEDS:
            continue
        output.append(
            {
                "sample_id": f"pet/{row['sample_id']}",
                "source_index": matrix_index,
                "source_split": str(row["split"]),
                "source_replicate_seed": source_seed,
                "targets": targets,
            }
        )
    return output


def _xml_text(element: ET.Element, path: str) -> str:
    value = element.findtext(path)
    if value is None:
        raise ValueError(f"VOC annotation is missing {path}")
    return value.strip()


def _voc_record(xml_bytes: bytes, split: str, image_set_id: str) -> dict[str, Any] | None:
    root = ET.fromstring(xml_bytes)
    width, height = int(_xml_text(root, "size/width")), int(_xml_text(root, "size/height"))
    if width <= 1 or height <= 1:
        raise ValueError("VOC image size is invalid")
    candidates = []
    for position, obj in enumerate(root.findall("object")):
        if int(_xml_text(obj, "difficult")) != 0:
            continue
        xmin = int(float(_xml_text(obj, "bndbox/xmin")))
        ymin = int(float(_xml_text(obj, "bndbox/ymin")))
        xmax = int(float(_xml_text(obj, "bndbox/xmax")))
        ymax = int(float(_xml_text(obj, "bndbox/ymax")))
        if not (1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height):
            raise ValueError("VOC bounding box is outside its image")
        box_width, box_height = xmax - xmin + 1, ymax - ymin + 1
        candidates.append((-(box_width * box_height), position, _xml_text(obj, "name"), xmin, ymin, xmax, ymax))
    if not candidates:
        return None
    _, position, class_name, xmin, ymin, xmax, ymax = min(candidates)
    box_width, box_height = xmax - xmin + 1, ymax - ymin + 1
    targets = {
        "largest_box_area_fraction": float(box_width * box_height / (width * height)),
        "largest_box_center_x": float((xmin + xmax - 2.0) / (2.0 * (width - 1))),
        "largest_box_center_y": float((ymin + ymax - 2.0) / (2.0 * (height - 1))),
        "largest_box_log_aspect_ratio": float(math.log(box_width / box_height)),
    }
    return {
        "sample_id": f"voc/{split}/{image_set_id}",
        "source_index": int(image_set_id),
        "source_split": split,
        "image_member": f"VOCdevkit/VOC2007/JPEGImages/{image_set_id}.jpg",
        "selected_object_position": position,
        "selected_object_class": class_name,
        "targets": targets,
    }


def voc_records() -> list[dict[str, Any]]:
    if md5_path(VOC_TRAINVAL) != VOC_TRAINVAL_MD5 or md5_path(VOC_TEST) != VOC_TEST_MD5:
        raise ValueError("Pascal VOC 2007 archive identity changed")
    output = []
    for path, split in ((VOC_TRAINVAL, "training"), (VOC_TEST, "evaluation")):
        with tarfile.open(path, "r:") as archive:
            set_handle = archive.extractfile("VOCdevkit/VOC2007/ImageSets/Main/trainval.txt" if split == "training" else "VOCdevkit/VOC2007/ImageSets/Main/test.txt")
            if set_handle is None:
                raise ValueError("VOC image-set file is missing")
            image_ids = [line.strip() for line in set_handle.read().decode("ascii").splitlines() if line.strip()]
            expected = 5011 if split == "training" else 4952
            if len(image_ids) != expected or len(set(image_ids)) != expected:
                raise ValueError("VOC image-set identity changed")
            for image_id in image_ids:
                handle = archive.extractfile(f"VOCdevkit/VOC2007/Annotations/{image_id}.xml")
                if handle is None:
                    raise ValueError(f"VOC annotation is missing: {image_id}")
                record = _voc_record(handle.read(), split, image_id)
                if record is not None:
                    output.append(record)
    return output


def _extract_selected_images(
    archive_path: Path,
    selected: Mapping[str, Mapping[str, Any]],
    image_root: Path,
) -> None:
    remaining = {str(row["image_member"]): sample_id for sample_id, row in selected.items()}
    mode = "r:gz" if archive_path.suffix == ".tgz" else "r:"
    with tarfile.open(archive_path, mode) as archive:
        for member in archive:
            sample_id = remaining.get(member.name)
            if sample_id is None:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"selected image cannot be read: {sample_id}")
            destination = image_root / (sample_id.replace("/", "__") + ".jpg")
            destination.write_bytes(handle.read())
            del remaining[member.name]
    if remaining:
        raise ValueError(f"selected image archive is incomplete: {len(remaining)} missing")


def _finalize_external_dataset(
    dataset_id: str,
    records: Sequence[Mapping[str, Any]],
    target_cohorts: Mapping[str, Sequence[Mapping[str, Any]]],
    input_root: Path,
    archives: Sequence[Path],
) -> dict[str, Any]:
    selected_ids = {str(row["sample_id"]) for rows in target_cohorts.values() for row in rows}
    selected = {str(row["sample_id"]): row for row in records if str(row["sample_id"]) in selected_ids}
    if set(selected) != selected_ids:
        raise ValueError(f"{dataset_id} selected rows are missing from source records")
    rows = []
    for matrix_index, sample_id in enumerate(sorted(selected)):
        row = selected[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "source_index": int(row["source_index"]),
                "image_path": f"images/{dataset_id}/{sample_id.replace('/', '__')}.jpg",
                "targets": dict(row["targets"]),
            }
        )
    matrix_index_by_id = {row["sample_id"]: index for index, row in enumerate(rows)}
    normalized_cohorts = {
        target_id: [
            {**dict(row), "matrix_index": matrix_index_by_id[str(row["sample_id"])]}
            for row in cohort
        ]
        for target_id, cohort in target_cohorts.items()
    }
    image_root = input_root / "images" / dataset_id
    image_root.mkdir(parents=True, exist_ok=False)
    for archive_path in archives:
        archive_selected = {
            sample_id: row
            for sample_id, row in selected.items()
            if (archive_path == CUB_ARCHIVE and sample_id.startswith("cub/"))
            or (archive_path == VOC_TRAINVAL and sample_id.startswith("voc/training/"))
            or (archive_path == VOC_TEST and sample_id.startswith("voc/evaluation/"))
        }
        _extract_selected_images(archive_path, archive_selected, image_root)
    for row in rows:
        path = input_root / row["image_path"]
        if not path.is_file():
            raise ValueError(f"prepared image is missing: {row['sample_id']}")
        with Image.open(path) as source:
            source.verify()
        row["image_sha256"] = sha256_path(path)
    return {"rows": rows, "target_cohorts": normalized_cohorts}


def prepare_inputs(input_root: Path = DEFAULT_INPUT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    input_root = input_root.resolve()
    output_root = input_root / "prepared"
    if output_root.exists():
        raise FileExistsError("large structured prepared inputs already exist")
    records_by_dataset = {
        CUB_ID: cub_records(),
        LSP_ID: lsp_records(),
        COCO_ID: coco_records(),
        PET_ID: pet_records(),
        VOC_ID: voc_records(),
    }
    target_cohorts = {}
    for dataset_id in DATASETS:
        records = records_by_dataset[dataset_id]
        selector = _select_assigned_target if dataset_id == PET_ID else _select_target
        target_cohorts[dataset_id] = {
            target_id: selector(dataset_id, target_id, records)
            for target_id in TARGETS_BY_DATASET[dataset_id]
        }
    output_root.mkdir(parents=True, exist_ok=False)
    datasets = {
        CUB_ID: _finalize_external_dataset(
            CUB_ID, records_by_dataset[CUB_ID], target_cohorts[CUB_ID], output_root, (CUB_ARCHIVE,)
        ),
        VOC_ID: _finalize_external_dataset(
            VOC_ID, records_by_dataset[VOC_ID], target_cohorts[VOC_ID], output_root, (VOC_TRAINVAL, VOC_TEST)
        ),
    }
    for dataset_id, parent_path in (
        (LSP_ID, LSP_PARENT_COHORT),
        (COCO_ID, COCO_PARENT_COHORT),
        (PET_ID, PET_PARENT_COHORT),
    ):
        rows = records_by_dataset[dataset_id]
        datasets[dataset_id] = {
            "rows": [
                {
                    "sample_id": str(row["sample_id"]),
                    "source_index": int(row["source_index"]),
                    "targets": dict(row["targets"]),
                }
                for row in rows
            ],
            "target_cohorts": {
                target_id: [
                    {**dict(entry), "matrix_index": int(entry["source_index"])}
                    for entry in cohort
                ]
                for target_id, cohort in target_cohorts[dataset_id].items()
            },
            "parent_cohort_path": str(parent_path.relative_to(ROOT)),
            "parent_cohort_sha256": sha256_path(parent_path),
        }
    datasets = {dataset_id: datasets[dataset_id] for dataset_id in DATASETS}
    cohort = {
        "schema_version": 1,
        "status": "prepared_large_structured_opportunity_inputs",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "selection_seed": SELECTION_SEED,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "datasets": datasets,
    }
    cohort_path = output_root / "cohort.json"
    _atomic_json(cohort_path, cohort)
    manifest = {
        "schema_version": 1,
        "status": "completed_large_structured_opportunity_input_preparation",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "cohort_sha256": sha256_path(cohort_path),
        "cohort_identity_sha256": sha256_payload(cohort),
        "dataset_ids": list(DATASETS),
        "target_count": sum(len(value) for value in TARGETS_BY_DATASET.values()),
        "source_hashes": {
            "cub_archive_sha256": sha256_path(CUB_ARCHIVE),
            "lsp_joints_sha256": sha256_path(LSP_JOINTS),
            "coco_annotations_sha256": sha256_path(COCO_ANNOTATIONS),
            "pet_parent_cohort_sha256": sha256_path(PET_PARENT_COHORT),
            "voc_trainval_sha256": sha256_path(VOC_TRAINVAL),
            "voc_test_sha256": sha256_path(VOC_TEST),
        },
        "predictive_outcomes_inspected": False,
        "overlapindex_fit_count": 0,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    return manifest


def load_cohort(input_root: Path = DEFAULT_INPUT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    prepared = input_root.resolve() / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    cohort = json.loads((prepared / "cohort.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed_large_structured_opportunity_input_preparation"
        or manifest.get("protocol_sha256") != sha256_path(PROTOCOL_PATH)
        or manifest.get("cohort_sha256") != sha256_path(prepared / "cohort.json")
        or manifest.get("cohort_identity_sha256") != sha256_payload(cohort)
        or cohort.get("protocol_sha256") != sha256_path(PROTOCOL_PATH)
        or set(cohort.get("datasets", ())) != set(DATASETS)
    ):
        raise ValueError("large structured prepared input identity changed")
    if protocol["expected_grid"]["target_count"] != manifest["target_count"]:
        raise ValueError("large structured prepared target count changed")
    return cohort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    args = parser.parse_args(argv)
    print(canonical_json(prepare_inputs(args.input_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
