"""Prepare the frozen SmallNORB left-camera structured-pose cohort."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_INPUT_ROOT = ROOT / "artifacts" / "continuous_smallnorb_head_discovery" / "inputs"
DEFAULT_RAW_ROOT = DEFAULT_INPUT_ROOT / "raw"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "prepared"
REPLICATE_SEEDS = (4103, 4201, 4303)
SELECTION_SEED = 2026091601
TRAIN_ROWS = 500
EVALUATION_ROWS = 1000
EXPECTED_ROWS = 24_300
IMAGE_SHAPE = (96, 96)
BYTE_MAGIC = 0x1E3D4C55
INTEGER_MAGIC = 0x1E3D4C54


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
        raise ValueError("SmallNORB protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_smallnorb_structured_pose_discovery_before_outcomes":
        raise ValueError("SmallNORB protocol is not frozen")
    cohort = protocol["cohort"]
    if (
        int(cohort["selection_seed"]) != SELECTION_SEED
        or tuple(cohort["replicate_seeds"]) != REPLICATE_SEEDS
        or int(cohort["training_rows_per_replicate"]) != TRAIN_ROWS
        or int(cohort["evaluation_rows"]) != EVALUATION_ROWS
    ):
        raise ValueError("SmallNORB cohort protocol differs from implementation")
    return protocol


def _read_exact(handle: BinaryIO, count: int) -> bytes:
    value = handle.read(count)
    if len(value) != count:
        raise ValueError("SmallNORB matrix file ended early")
    return value


def read_header(handle: BinaryIO) -> tuple[int, tuple[int, ...]]:
    magic = struct.unpack("<I", _read_exact(handle, 4))[0]
    ndim = struct.unpack("<i", _read_exact(handle, 4))[0]
    if magic not in {BYTE_MAGIC, INTEGER_MAGIC} or not 1 <= ndim <= 4:
        raise ValueError("SmallNORB matrix header has unknown magic or dimension count")
    padded = max(3, ndim)
    dimensions = struct.unpack(f"<{padded}i", _read_exact(handle, 4 * padded))[:ndim]
    if any(value <= 0 for value in dimensions):
        raise ValueError("SmallNORB matrix header contains a nonpositive dimension")
    return magic, tuple(int(value) for value in dimensions)


def read_integer_matrix(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, shape = read_header(handle)
        if magic != INTEGER_MAGIC or shape != expected_shape:
            raise ValueError(f"SmallNORB integer matrix shape changed: {path.name}")
        count = int(np.prod(shape))
        values = np.frombuffer(_read_exact(handle, count * 4), dtype="<i4").copy()
        if handle.read(1):
            raise ValueError("SmallNORB integer matrix has trailing bytes")
    return values.reshape(shape)


def _split_names(split: str) -> dict[str, str]:
    if split == "training":
        stem = "smallnorb-5x46789x9x18x6x2x96x96-training"
    elif split == "testing":
        stem = "smallnorb-5x01235x9x18x6x2x96x96-testing"
    else:
        raise ValueError(f"unknown SmallNORB split {split!r}")
    return {kind: f"{stem}-{kind}.mat.gz" for kind in ("dat", "cat", "info")}


def validate_raw(protocol: Mapping[str, Any], raw_root: Path) -> dict[str, str]:
    expected = protocol["dataset"]["raw_files"]
    observed = {}
    for name, digest in expected.items():
        path = raw_root / name
        if not path.is_file() or sha256_path(path) != digest:
            raise ValueError(f"SmallNORB raw archive identity changed: {name}")
        observed[name] = digest
    if set(observed) != set(expected):
        raise ValueError("SmallNORB raw archive set changed")
    return observed


def load_metadata(raw_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    names = _split_names(split)
    categories = read_integer_matrix(raw_root / names["cat"], (EXPECTED_ROWS,))
    info = read_integer_matrix(raw_root / names["info"], (EXPECTED_ROWS, 4))
    if set(np.unique(categories)) != set(range(5)):
        raise ValueError("SmallNORB category grid changed")
    expected_instances = {4, 6, 7, 8, 9} if split == "training" else {0, 1, 2, 3, 5}
    expected_domains = (
        expected_instances,
        set(range(9)),
        set(range(0, 36, 2)),
        set(range(6)),
    )
    for column, expected in enumerate(expected_domains):
        if set(np.unique(info[:, column])) != expected:
            raise ValueError(f"SmallNORB info domain changed at column {column}")
    if set(np.unique(info[:, 0])) != expected_instances:
        raise ValueError("SmallNORB official instance split changed")
    return categories, info


def _quota(total: int, position: int, count: int) -> int:
    base, remainder = divmod(total, count)
    return base + int(position < remainder)


def _order_key(split: str, source_index: int) -> tuple[str, int]:
    digest = hashlib.sha256(
        f"{SELECTION_SEED}|{split}|{source_index}".encode("utf-8")
    ).hexdigest()
    return digest, int(source_index)


def target_values(info_row: Sequence[int]) -> dict[str, float]:
    if len(info_row) != 4:
        raise ValueError("SmallNORB info row must contain four fields")
    _instance, elevation, azimuth, _lighting = (int(value) for value in info_row)
    angle = np.deg2rad(10.0 * azimuth)
    return {
        "elevation_degrees": float(30 + 5 * elevation),
        "azimuth_sine": float(np.sin(angle)),
        "azimuth_cosine": float(np.cos(angle)),
    }


def select_rows(
    train_categories: np.ndarray,
    train_info: np.ndarray,
    test_categories: np.ndarray,
    test_info: np.ndarray,
) -> list[dict[str, Any]]:
    sources = {
        "training": (np.asarray(train_categories), np.asarray(train_info)),
        "testing": (np.asarray(test_categories), np.asarray(test_info)),
    }
    output: list[dict[str, Any]] = []
    strata = [(category, elevation) for category in range(5) for elevation in range(9)]
    for stratum_position, (category, elevation) in enumerate(strata):
        test_cat, test_meta = sources["testing"]
        test_pool = np.flatnonzero((test_cat == category) & (test_meta[:, 1] == elevation))
        test_order = sorted((int(value) for value in test_pool), key=lambda value: _order_key("testing", value))
        eval_quota = _quota(EVALUATION_ROWS, stratum_position, len(strata))
        for source_index in test_order[:eval_quota]:
            output.append(
                _row("testing", source_index, category, test_meta[source_index], "evaluation", None)
            )

        train_cat, train_meta = sources["training"]
        train_pool = np.flatnonzero((train_cat == category) & (train_meta[:, 1] == elevation))
        train_order = sorted((int(value) for value in train_pool), key=lambda value: _order_key("training", value))
        train_quota = _quota(TRAIN_ROWS, stratum_position, len(strata))
        required = train_quota * len(REPLICATE_SEEDS)
        if len(train_order) < required:
            raise ValueError("SmallNORB training stratum cannot support disjoint cohorts")
        for replicate_position, replicate_seed in enumerate(REPLICATE_SEEDS):
            start = replicate_position * train_quota
            for source_index in train_order[start : start + train_quota]:
                output.append(
                    _row(
                        "training",
                        source_index,
                        category,
                        train_meta[source_index],
                        "training",
                        replicate_seed,
                    )
                )
    ids = [row["sample_id"] for row in output]
    if len(output) != EVALUATION_ROWS + TRAIN_ROWS * len(REPLICATE_SEEDS) or len(ids) != len(set(ids)):
        raise RuntimeError("SmallNORB selected cohort is incomplete or duplicated")
    if sum(row["split"] == "evaluation" for row in output) != EVALUATION_ROWS:
        raise RuntimeError("SmallNORB evaluation cohort size changed")
    for seed in REPLICATE_SEEDS:
        if sum(row["replicate_seed"] == seed for row in output) != TRAIN_ROWS:
            raise RuntimeError("SmallNORB training cohort size changed")
    return sorted(
        output,
        key=lambda row: (
            row["split"],
            row["replicate_seed"] or 0,
            row["category"],
            row["elevation_code"],
            row["sample_id"],
        ),
    )


def _row(
    source_split: str,
    source_index: int,
    category: int,
    info: Sequence[int],
    split: str,
    replicate_seed: int | None,
) -> dict[str, Any]:
    instance, elevation, azimuth, lighting = (int(value) for value in info)
    sample_id = f"smallnorb/{source_split}/{source_index:05d}/left"
    return {
        "sample_id": sample_id,
        "source_split": source_split,
        "source_index": int(source_index),
        "camera": "left",
        "category": int(category),
        "instance": instance,
        "elevation_code": elevation,
        "azimuth_code": azimuth,
        "lighting_code": lighting,
        "split": split,
        "replicate_seed": replicate_seed,
        "targets": target_values(info),
    }


def _write_selected_images(
    dat_path: Path,
    split: str,
    selected: Mapping[int, Mapping[str, Any]],
    image_root: Path,
) -> dict[int, str]:
    output: dict[int, str] = {}
    row_bytes = 2 * IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    with gzip.open(dat_path, "rb") as handle:
        magic, shape = read_header(handle)
        if magic != BYTE_MAGIC or shape != (EXPECTED_ROWS, 2, *IMAGE_SHAPE):
            raise ValueError(f"SmallNORB image matrix shape changed: {dat_path.name}")
        for source_index in range(EXPECTED_ROWS):
            payload = _read_exact(handle, row_bytes)
            if source_index not in selected:
                continue
            left = np.frombuffer(payload[: IMAGE_SHAPE[0] * IMAGE_SHAPE[1]], dtype=np.uint8)
            left = np.ascontiguousarray(left.reshape(IMAGE_SHAPE).T)
            filename = f"{split}_{source_index:05d}_left.png"
            path = image_root / filename
            Image.fromarray(left).save(path, format="PNG", optimize=False)
            output[source_index] = filename
        if handle.read(1):
            raise ValueError("SmallNORB image matrix has trailing bytes")
    if set(output) != set(selected):
        raise ValueError(f"SmallNORB selected images are incomplete for {split}")
    return output


def prepare_inputs(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    protocol = load_protocol()
    raw_root, output_root = raw_root.resolve(), output_root.resolve()
    if output_root.exists():
        raise FileExistsError("SmallNORB prepared input root already exists")
    raw_hashes = validate_raw(protocol, raw_root)
    train_cat, train_info = load_metadata(raw_root, "training")
    test_cat, test_info = load_metadata(raw_root, "testing")
    rows = select_rows(train_cat, train_info, test_cat, test_info)
    output_root.mkdir(parents=True, exist_ok=False)
    image_root = output_root / "images"
    image_root.mkdir()
    by_source = {
        split: {row["source_index"]: row for row in rows if row["source_split"] == split}
        for split in ("training", "testing")
    }
    image_names = {}
    for split in ("training", "testing"):
        dat_path = raw_root / _split_names(split)["dat"]
        image_names[split] = _write_selected_images(dat_path, split, by_source[split], image_root)
    records = []
    for row in rows:
        filename = image_names[row["source_split"]][row["source_index"]]
        image_path = image_root / filename
        with Image.open(image_path) as image:
            if image.mode != "L" or image.size != IMAGE_SHAPE:
                raise ValueError("SmallNORB prepared image mode or shape changed")
        records.append(
            {
                **row,
                "image_path": f"images/{filename}",
                "image_sha256": sha256_path(image_path),
            }
        )
    cohort = {
        "schema_version": 1,
        "status": "prepared_smallnorb_structured_pose_cohort",
        "dataset_id": protocol["dataset"]["id"],
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "raw_hashes": raw_hashes,
        "selection_seed": SELECTION_SEED,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "training_rows_per_replicate": TRAIN_ROWS,
        "evaluation_rows": EVALUATION_ROWS,
        "union_rows": len(records),
        "targets": list(protocol["dataset"]["targets"]),
        "rows": records,
    }
    cohort_path = output_root / "cohort.json"
    _atomic_json(cohort_path, cohort)
    manifest = {
        "schema_version": 1,
        "status": "completed_smallnorb_input_preparation",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "raw_root": str(raw_root),
        "raw_hashes": raw_hashes,
        "cohort_sha256": sha256_path(cohort_path),
        "cohort_identity_sha256": sha256_payload(cohort),
        "image_count": len(records),
        "outcomes_inspected": False,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    return manifest


def load_cohort(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    output_root = output_root.resolve()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    cohort_path = output_root / "cohort.json"
    if manifest.get("status") != "completed_smallnorb_input_preparation":
        raise ValueError("SmallNORB input manifest is incomplete")
    if manifest.get("protocol_sha256") != sha256_path(PROTOCOL_PATH):
        raise ValueError("SmallNORB input protocol identity changed")
    if sha256_path(cohort_path) != manifest.get("cohort_sha256"):
        raise ValueError("SmallNORB cohort hash changed")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if sha256_payload(cohort) != manifest.get("cohort_identity_sha256"):
        raise ValueError("SmallNORB cohort payload identity changed")
    if len(cohort.get("rows", ())) != 2500:
        raise ValueError("SmallNORB cohort row count changed")
    for row in cohort["rows"]:
        image_path = output_root / row["image_path"]
        if sha256_path(image_path) != row["image_sha256"]:
            raise ValueError(f"SmallNORB prepared image identity changed: {row['sample_id']}")
    return cohort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(canonical_json(prepare_inputs(args.raw_root, args.output_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
