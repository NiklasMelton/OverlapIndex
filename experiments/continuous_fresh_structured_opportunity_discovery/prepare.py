"""Validate and prepare the frozen CUB/LSP structured-target cohorts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_INPUT_ROOT = (
    ROOT / "artifacts" / "continuous_fresh_structured_opportunity_discovery" / "inputs"
)
DEFAULT_RAW_ROOT = DEFAULT_INPUT_ROOT / "raw"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "prepared"
REPLICATE_SEEDS = (4103, 4201, 4303)
SELECTION_SEED = 2026092401
CUB_TRAIN_PER_SEED = 400
CUB_EVALUATION = 800
LSP_TRAIN_PER_SEED = 200
LSP_EVALUATION = 500
CUB_ID = "cub_200_2011_beak_tail"
LSP_ID = "lsp_2000_pose"
CUB_TARGETS = (
    "beak_tail_axis_sine",
    "beak_tail_axis_cosine",
    "beak_tail_length_over_bbox_diagonal",
)
LSP_TARGETS = (
    "torso_axis_sine",
    "torso_axis_cosine",
    "bilateral_knee_extension_cosine",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_path(path: os.PathLike[str] | str) -> str:
    digest = hashlib.md5()  # noqa: S324 - validating the publisher's identity hash
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
        raise ValueError("fresh structured protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_fresh_structured_opportunity_discovery_before_outcomes":
        raise ValueError("fresh structured protocol is not frozen")
    if tuple(protocol["cohort"]["replicate_seeds"]) != REPLICATE_SEEDS:
        raise ValueError("fresh structured replicate seed grid changed")
    if int(protocol["cohort"]["selection_seed"]) != SELECTION_SEED:
        raise ValueError("fresh structured selection seed changed")
    datasets = protocol["datasets"]
    if tuple(datasets[CUB_ID]["targets"]) != CUB_TARGETS:
        raise ValueError("CUB target grid changed")
    if tuple(datasets[LSP_ID]["targets"]) != LSP_TARGETS:
        raise ValueError("LSP target grid changed")
    return protocol


def _tar_text(archive: tarfile.TarFile, relative: str) -> str:
    name = f"CUB_200_2011/{relative}"
    try:
        member = archive.getmember(name)
    except KeyError as error:
        raise ValueError(f"CUB archive is missing {relative}") from error
    if not member.isfile():
        raise ValueError(f"CUB archive member is not a file: {relative}")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"CUB archive member cannot be read: {relative}")
    return handle.read().decode("utf-8")


def _indexed_rows(text: str, width: int, label: str) -> dict[int, list[str]]:
    rows: dict[int, list[str]] = {}
    for raw in text.splitlines():
        fields = raw.strip().split()
        if len(fields) != width:
            raise ValueError(f"CUB {label} row width changed")
        index = int(fields[0])
        if index in rows:
            raise ValueError(f"CUB {label} contains duplicate image id")
        rows[index] = fields[1:]
    return rows


def cub_targets(
    beak_xy: Sequence[float], tail_xy: Sequence[float], bbox_wh: Sequence[float]
) -> dict[str, float]:
    beak = np.asarray(beak_xy, dtype=np.float64)
    tail = np.asarray(tail_xy, dtype=np.float64)
    width, height = (float(value) for value in bbox_wh)
    delta = beak - tail
    distance = float(np.linalg.norm(delta))
    diagonal = float(math.hypot(width, height))
    if (
        beak.shape != (2,)
        or tail.shape != (2,)
        or not np.isfinite(beak).all()
        or not np.isfinite(tail).all()
        or not math.isfinite(distance)
        or distance <= 0.0
        or not math.isfinite(diagonal)
        or diagonal <= 0.0
    ):
        raise ValueError("CUB beak/tail geometry is invalid")
    return {
        "beak_tail_axis_sine": float(delta[1] / distance),
        "beak_tail_axis_cosine": float(delta[0] / distance),
        "beak_tail_length_over_bbox_diagonal": float(distance / diagonal),
    }


def read_cub_metadata(archive_path: Path) -> dict[int, dict[str, Any]]:
    with tarfile.open(archive_path, "r:gz") as archive:
        images = _indexed_rows(_tar_text(archive, "images.txt"), 2, "images")
        classes = _indexed_rows(
            _tar_text(archive, "image_class_labels.txt"), 2, "classes"
        )
        splits = _indexed_rows(_tar_text(archive, "train_test_split.txt"), 2, "split")
        boxes = _indexed_rows(_tar_text(archive, "bounding_boxes.txt"), 5, "boxes")
        parts: dict[tuple[int, int], tuple[float, float, int]] = {}
        for raw in _tar_text(archive, "parts/part_locs.txt").splitlines():
            fields = raw.strip().split()
            if len(fields) != 5:
                raise ValueError("CUB part-location row width changed")
            image_id, part_id = int(fields[0]), int(fields[1])
            key = (image_id, part_id)
            if key in parts:
                raise ValueError("CUB part locations contain a duplicate key")
            parts[key] = (float(fields[2]), float(fields[3]), int(fields[4]))
    expected_ids = set(range(1, 11789))
    if not all(set(table) == expected_ids for table in (images, classes, splits, boxes)):
        raise ValueError("CUB image metadata id grid changed")
    if len(parts) != 11788 * 15:
        raise ValueError("CUB part-location grid changed")
    records: dict[int, dict[str, Any]] = {}
    for image_id in sorted(expected_ids):
        class_id = int(classes[image_id][0])
        is_train = int(splits[image_id][0])
        x, y, width, height = (float(value) for value in boxes[image_id])
        beak = parts[(image_id, 2)]
        tail = parts[(image_id, 14)]
        eligible = (
            beak[2] == 1
            and tail[2] == 1
            and width > 0.0
            and height > 0.0
            and np.isfinite((x, y, width, height, *beak[:2], *tail[:2])).all()
            and (beak[0] != tail[0] or beak[1] != tail[1])
        )
        records[image_id] = {
            "image_id": image_id,
            "image_member": f"CUB_200_2011/images/{images[image_id][0]}",
            "class_id": class_id,
            "official_split": "training" if is_train == 1 else "evaluation",
            "eligible": bool(eligible),
            "targets": cub_targets(beak[:2], tail[:2], (width, height)) if eligible else None,
        }
    if set(record["class_id"] for record in records.values()) != set(range(1, 201)):
        raise ValueError("CUB class id grid changed")
    if sum(record["official_split"] == "training" for record in records.values()) != 5994:
        raise ValueError("CUB official training size changed")
    if sum(record["official_split"] == "evaluation" for record in records.values()) != 5794:
        raise ValueError("CUB official evaluation size changed")
    return records


def _order_key(dataset_id: str, group: str, source_index: int) -> tuple[str, int]:
    payload = f"{SELECTION_SEED}|{dataset_id}|{group}|{source_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), int(source_index)


def select_cub_rows(records: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for image_id, record in records.items():
        if record["eligible"]:
            by_class[int(record["class_id"])][str(record["official_split"])].append(image_id)
    output = []
    for class_id in range(1, 201):
        train = sorted(
            by_class[class_id]["training"],
            key=lambda value: _order_key(CUB_ID, f"train/{class_id}", value),
        )
        evaluation = sorted(
            by_class[class_id]["evaluation"],
            key=lambda value: _order_key(CUB_ID, f"eval/{class_id}", value),
        )
        if len(train) < 6 or len(evaluation) < 4:
            raise ValueError(
                f"CUB class {class_id} cannot support the frozen visible-part cohort"
            )
        for seed_position, seed in enumerate(REPLICATE_SEEDS):
            for image_id in train[2 * seed_position : 2 * seed_position + 2]:
                output.append(_cub_row(records[image_id], "training", seed))
        for image_id in evaluation[:4]:
            output.append(_cub_row(records[image_id], "evaluation", None))
    if len(output) != CUB_EVALUATION + CUB_TRAIN_PER_SEED * len(REPLICATE_SEEDS):
        raise RuntimeError("CUB selected cohort size changed")
    if len({row["sample_id"] for row in output}) != len(output):
        raise RuntimeError("CUB selected cohort is not disjoint")
    return sorted(output, key=lambda row: row["sample_id"])


def _cub_row(record: Mapping[str, Any], split: str, seed: int | None) -> dict[str, Any]:
    image_id = int(record["image_id"])
    return {
        "sample_id": f"cub/{image_id:05d}",
        "source_index": image_id,
        "class_id": int(record["class_id"]),
        "split": split,
        "replicate_seed": seed,
        "image_path": f"prepared/images/cub_{image_id:05d}.jpg",
        "targets": dict(record["targets"]),
    }


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (2,) or not np.isfinite(vector).all() or norm <= 0.0:
        raise ValueError(f"LSP {label} vector is invalid")
    return vector / norm


def lsp_targets(joints_xy: np.ndarray) -> dict[str, float]:
    joints = np.asarray(joints_xy, dtype=np.float64)
    if joints.shape != (14, 2) or not np.isfinite(joints).all():
        raise ValueError("LSP joint coordinate matrix is invalid")
    hip_midpoint = 0.5 * (joints[2] + joints[3])
    shoulder_midpoint = 0.5 * (joints[8] + joints[9])
    torso = _unit(shoulder_midpoint - hip_midpoint, "torso")
    right = float(np.dot(_unit(joints[2] - joints[1], "right thigh"), _unit(joints[0] - joints[1], "right shin")))
    left = float(np.dot(_unit(joints[3] - joints[4], "left thigh"), _unit(joints[5] - joints[4], "left shin")))
    return {
        "torso_axis_sine": float(torso[1]),
        "torso_axis_cosine": float(torso[0]),
        "bilateral_knee_extension_cosine": float(-0.5 * (right + left)),
    }


def _lsp_torso_targets(joints: np.ndarray) -> tuple[float, float]:
    hip_midpoint = 0.5 * (joints[2] + joints[3])
    shoulder_midpoint = 0.5 * (joints[8] + joints[9])
    torso = _unit(shoulder_midpoint - hip_midpoint, "torso")
    return float(torso[1]), float(torso[0])


def _lsp_knee_target(joints: np.ndarray) -> float:
    right = float(
        np.dot(
            _unit(joints[2] - joints[1], "right thigh"),
            _unit(joints[0] - joints[1], "right shin"),
        )
    )
    left = float(
        np.dot(
            _unit(joints[3] - joints[4], "left thigh"),
            _unit(joints[5] - joints[4], "left shin"),
        )
    )
    return float(-0.5 * (right + left))


def read_lsp_metadata(raw_root: Path) -> list[dict[str, Any]]:
    joints_path = raw_root / "joints.mat"
    joints = np.asarray(loadmat(joints_path)["joints"], dtype=np.float64)
    if joints.shape != (3, 14, 2000) or not np.isfinite(joints).all():
        raise ValueError("LSP joints matrix identity changed")
    if set(np.unique(joints[2])) != {0.0, 1.0}:
        raise ValueError("LSP occlusion-flag domain changed")
    records = []
    for source_index in range(2000):
        coordinates = joints[:2, :, source_index].T
        occluded = joints[2, :, source_index].astype(np.int64)
        torso_eligible = bool((occluded[[2, 3, 8, 9]] == 0).all())
        knee_eligible = bool((occluded[[0, 1, 2, 3, 4, 5]] == 0).all())
        torso_sine = torso_cosine = knee_extension = None
        if torso_eligible:
            try:
                torso_sine, torso_cosine = _lsp_torso_targets(coordinates)
            except ValueError:
                torso_eligible = False
        if knee_eligible:
            try:
                knee_extension = _lsp_knee_target(coordinates)
            except ValueError:
                knee_eligible = False
        values = {
            "torso_axis_sine": torso_sine,
            "torso_axis_cosine": torso_cosine,
            "bilateral_knee_extension_cosine": knee_extension,
        }
        records.append(
            {
                "sample_id": f"lsp/{source_index + 1:04d}",
                "source_index": source_index,
                "source_split": "training" if source_index < 1000 else "evaluation",
                "image_path": f"raw/lsp/images/im{source_index + 1:04d}.jpg",
                "targets": values,
                "eligible": {
                    "torso_axis_sine": torso_eligible,
                    "torso_axis_cosine": torso_eligible,
                    "bilateral_knee_extension_cosine": knee_eligible,
                },
            }
        )
    return records


def _quantile_bins(values: np.ndarray, count: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < count or not np.isfinite(values).all():
        raise ValueError("quantile source values are invalid")
    edges = np.quantile(values, np.linspace(0.0, 1.0, count + 1))
    if len(np.unique(edges)) != len(edges):
        raise ValueError("quantile source values cannot form ten distinct strata")
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int64)


def _select_lsp_group(
    records: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    stratifier: np.ndarray,
    group_name: str,
) -> dict[str, list[dict[str, Any]]]:
    eligible = np.asarray(
        [all(bool(row["eligible"][target]) for target in targets) for row in records],
        dtype=bool,
    )
    result = {target: [] for target in targets}
    for source_split in ("training", "evaluation"):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(records)
                if row["source_split"] == source_split and eligible[index]
            ],
            dtype=np.int64,
        )
        bins = _quantile_bins(stratifier[indices])
        for bin_id in range(10):
            pool = sorted(
                (int(value) for value in indices[bins == bin_id]),
                key=lambda value: _order_key(
                    LSP_ID, f"{group_name}/{source_split}/{bin_id}", value
                ),
            )
            required = 60 if source_split == "training" else 50
            if len(pool) < required:
                raise ValueError(
                    f"LSP {group_name} stratum {bin_id} {source_split} cannot support the frozen cohort"
                )
            if source_split == "evaluation":
                assignments = [(value, None) for value in pool[:50]]
            else:
                assignments = []
                for seed_position, seed in enumerate(REPLICATE_SEEDS):
                    assignments.extend(
                        (value, seed)
                        for value in pool[20 * seed_position : 20 * seed_position + 20]
                    )
            for target in targets:
                result[target].extend(
                    {
                        "source_index": value,
                        "split": source_split,
                        "replicate_seed": seed,
                    }
                    for value, seed in assignments
                )
    for target, rows in result.items():
        if len(rows) != LSP_EVALUATION + LSP_TRAIN_PER_SEED * len(REPLICATE_SEEDS):
            raise RuntimeError(f"LSP {target} selected cohort size changed")
        if len({int(row["source_index"]) for row in rows}) != len(rows):
            raise RuntimeError(f"LSP {target} selected cohort is not disjoint")
    return result


def select_lsp_cohorts(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    torso_angle = np.asarray(
        [
            math.atan2(
                row["targets"]["torso_axis_sine"],
                row["targets"]["torso_axis_cosine"],
            )
            if row["eligible"]["torso_axis_sine"]
            else np.nan
            for row in records
        ],
        dtype=np.float64,
    )
    knee = np.asarray(
        [
            row["targets"]["bilateral_knee_extension_cosine"]
            if row["eligible"]["bilateral_knee_extension_cosine"]
            else np.nan
            for row in records
        ],
        dtype=np.float64,
    )
    cohorts = _select_lsp_group(
        records,
        ("torso_axis_sine", "torso_axis_cosine"),
        torso_angle,
        "torso_angle",
    )
    cohorts.update(
        _select_lsp_group(
            records,
            ("bilateral_knee_extension_cosine",),
            knee,
            "knee_extension",
        )
    )
    return cohorts


def validate_raw(protocol: Mapping[str, Any], raw_root: Path) -> dict[str, Any]:
    cub_source = protocol["datasets"][CUB_ID]["source"]
    cub_path = raw_root / str(cub_source["archive_name"])
    if not cub_path.is_file():
        raise ValueError("CUB archive is missing")
    if md5_path(cub_path) != cub_source["official_md5"]:
        raise ValueError("CUB publisher MD5 mismatch")
    if sha256_path(cub_path) != cub_source["archive_sha256"]:
        raise ValueError("CUB archive SHA-256 mismatch")
    lsp_root = raw_root / "lsp"
    images = sorted((lsp_root / "images").glob("im*.jpg"))
    expected_names = [f"im{value:04d}.jpg" for value in range(1, 2001)]
    if [path.name for path in images] != expected_names:
        raise ValueError("LSP image file grid is incomplete or changed")
    lsp_source = protocol["datasets"][LSP_ID]["source"]
    if sha256_path(lsp_root / "joints.mat") != lsp_source["joints_sha256"]:
        raise ValueError("LSP joints identity changed")
    if sha256_path(lsp_root / "README.md") != lsp_source["readme_sha256"]:
        raise ValueError("LSP README identity changed")
    image_hashes = {path.name: sha256_path(path) for path in images}
    return {
        "cub": {
            "archive_name": cub_path.name,
            "bytes": cub_path.stat().st_size,
            "md5": md5_path(cub_path),
            "sha256": sha256_path(cub_path),
        },
        "lsp": {
            "revision": lsp_source["revision"],
            "image_count": len(images),
            "image_content_inventory_sha256": sha256_payload(image_hashes),
            "joints_sha256": sha256_path(lsp_root / "joints.mat"),
            "readme_sha256": sha256_path(lsp_root / "README.md"),
        },
    }


def _extract_cub_images(
    archive_path: Path,
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[int, Mapping[str, Any]],
    output_root: Path,
) -> None:
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    wanted = {
        str(records[int(row["source_index"])]["image_member"]): int(row["source_index"])
        for row in rows
    }
    extracted = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            image_id = wanted[member.name]
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"CUB image cannot be read: {image_id}")
            _atomic_bytes(image_root / f"cub_{image_id:05d}.jpg", handle.read())
            extracted.add(image_id)
    if len(extracted) != len(wanted):
        raise ValueError("CUB archive does not contain every selected image")


def run_preparation(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    protocol = load_protocol()
    raw_root, output_root = raw_root.resolve(), output_root.resolve()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        raise FileExistsError("fresh structured terminal input manifest already exists")
    raw_identity = validate_raw(protocol, raw_root)
    cub_archive = raw_root / protocol["datasets"][CUB_ID]["source"]["archive_name"]
    cub_records = read_cub_metadata(cub_archive)
    cub_rows = select_cub_rows(cub_records)
    lsp_records = read_lsp_metadata(raw_root / "lsp")
    lsp_cohorts = select_lsp_cohorts(lsp_records)
    _extract_cub_images(cub_archive, cub_rows, cub_records, output_root)
    cohort = {
        "schema_version": 1,
        "status": "prepared_fresh_structured_opportunity_cohort",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "datasets": {
            CUB_ID: {"rows": cub_rows, "target_cohorts": {target: None for target in CUB_TARGETS}},
            LSP_ID: {"rows": lsp_records, "target_cohorts": lsp_cohorts},
        },
    }
    cohort_path = output_root / "cohort.json"
    _atomic_json(cohort_path, cohort)
    extracted = sorted((output_root / "images").glob("cub_*.jpg"))
    if len(extracted) != len(cub_rows):
        raise RuntimeError("CUB extracted image count changed")
    manifest = {
        "schema_version": 1,
        "status": "completed_fresh_structured_opportunity_preparation",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "raw_identity": raw_identity,
        "cohort_sha256": sha256_path(cohort_path),
        "cohort_identity_sha256": sha256_payload(cohort),
        "dataset_rows": {CUB_ID: len(cub_rows), LSP_ID: len(lsp_records)},
        "target_cohort_rows": {
            CUB_ID: {target: len(cub_rows) for target in CUB_TARGETS},
            LSP_ID: {target: len(rows) for target, rows in lsp_cohorts.items()},
        },
        "source_hashes": {"prepare": sha256_path(__file__)},
        "predictive_outcomes_inspected": False,
    }
    _atomic_json(terminal, manifest)
    return manifest


def load_cohort(input_root: Path = DEFAULT_INPUT_ROOT) -> dict[str, Any]:
    prepared = input_root / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    cohort = json.loads((prepared / "cohort.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_fresh_structured_opportunity_preparation":
        raise ValueError("fresh structured preparation is incomplete")
    if manifest.get("protocol_sha256") != sha256_path(PROTOCOL_PATH):
        raise ValueError("fresh structured prepared protocol identity changed")
    if manifest.get("cohort_sha256") != sha256_path(prepared / "cohort.json"):
        raise ValueError("fresh structured cohort file identity changed")
    if manifest.get("cohort_identity_sha256") != sha256_payload(cohort):
        raise ValueError("fresh structured cohort payload identity changed")
    return cohort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(canonical_json(run_preparation(args.raw_root, args.output_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
