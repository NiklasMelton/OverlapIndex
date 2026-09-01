"""Prepare the outcome-free Oxford-IIIT Pet foreground-occupancy cohorts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "artifacts" / "continuous_real_pet_area_panel" / "inputs"
IMAGES_ARCHIVE = "images.tar.gz"
ANNOTATIONS_ARCHIVE = "annotations.tar.gz"
IMAGES_SHA256 = "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d"
ANNOTATIONS_SHA256 = "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91"
OFFICIAL_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
OFFICIAL_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
OFFICIAL_ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
SELECTION_SEED = 2026090301
REPLICATE_SEEDS = (4103, 4201, 4303, 4409, 4507)
TARGET_BINS = tuple(range(10))
ROWS_PER_BIN = 20
MIN_TARGET = 0.01
MAX_TARGET = 0.99


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def foreground_fraction(mask: np.ndarray) -> tuple[int, int, float]:
    values = np.asarray(mask)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Oxford-IIIT Pet trimap must be a nonempty 2-D matrix")
    if not set(np.unique(values)).issubset({1, 2, 3}):
        raise ValueError("Oxford-IIIT Pet trimap contains an unknown pixel label")
    foreground = int(np.count_nonzero(values != 2))
    pixels = int(values.size)
    return foreground, pixels, float(foreground / pixels)


def _official_rows(annotations: tarfile.TarFile) -> list[dict[str, Any]]:
    handle = annotations.extractfile("annotations/list.txt")
    if handle is None:
        raise ValueError("official Oxford-IIIT Pet list is missing")
    rows = []
    for line in handle.read().decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("official Oxford-IIIT Pet list row is malformed")
        sample_id, class_id, species, breed_id = fields
        parsed = tuple(int(value) for value in (class_id, species, breed_id))
        if parsed[0] not in range(1, 38) or parsed[1] not in (1, 2):
            raise ValueError("official Oxford-IIIT Pet class metadata is invalid")
        rows.append(
            {
                "sample_id": sample_id,
                "class_id": parsed[0],
                "species": parsed[1],
                "breed_id": parsed[2],
            }
        )
    if len(rows) != 7349 or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("official Oxford-IIIT Pet list identity changed")
    return rows


def selected_rows(images_path: Path, annotations_path: Path) -> list[dict[str, Any]]:
    if sha256_path(images_path) != IMAGES_SHA256:
        raise ValueError("Oxford-IIIT Pet image archive SHA-256 mismatch")
    if sha256_path(annotations_path) != ANNOTATIONS_SHA256:
        raise ValueError("Oxford-IIIT Pet annotation archive SHA-256 mismatch")
    with tarfile.open(annotations_path, "r:gz") as annotations:
        official = _official_rows(annotations)
    official_by_id = {str(row["sample_id"]): row for row in official}
    with tarfile.open(images_path, "r:gz") as images:
        image_ids = {
            Path(member.name).stem
            for member in images.getmembers()
            if member.isfile() and member.name.startswith("images/") and member.name.endswith(".jpg")
        }
    if not set(official_by_id).issubset(image_ids):
        raise ValueError("official Oxford-IIIT Pet image set is incomplete")
    targets: dict[str, tuple[int, int, float]] = {}
    with tarfile.open(annotations_path, "r|gz") as annotations:
        for member in annotations:
            if not (
                member.isfile()
                and member.name.startswith("annotations/trimaps/")
                and member.name.endswith(".png")
            ):
                continue
            sample_id = Path(member.name).stem
            if sample_id not in official_by_id:
                continue
            mask_handle = annotations.extractfile(member)
            if mask_handle is None:
                raise ValueError(f"official trimap cannot be read: {sample_id}")
            with Image.open(io.BytesIO(mask_handle.read())) as mask_image:
                targets[sample_id] = foreground_fraction(np.asarray(mask_image))
    if set(targets) != set(official_by_id):
        raise ValueError("official Oxford-IIIT Pet trimap set is incomplete")
    eligible = []
    for sample_id, row in official_by_id.items():
        foreground, pixels, target = targets[sample_id]
        if not MIN_TARGET <= target <= MAX_TARGET:
            continue
        eligible.append(
            {
                **row,
                "foreground_and_boundary_pixels": foreground,
                "pixel_count": pixels,
                "foreground_fraction": target,
            }
        )
    eligible.sort(key=lambda row: (row["foreground_fraction"], row["sample_id"]))
    if len(eligible) != 7334:
        raise ValueError("Oxford-IIIT Pet valid target population changed")
    by_bin: dict[int, list[dict[str, Any]]] = {value: [] for value in TARGET_BINS}
    for position, row in enumerate(eligible):
        target_bin = min(9, position * len(TARGET_BINS) // len(eligible))
        selection_key = hashlib.sha256(
            f"{SELECTION_SEED}|{target_bin}|{row['sample_id']}".encode("utf-8")
        ).hexdigest()
        by_bin[target_bin].append(
            {**row, "target_bin": target_bin, "selection_key": selection_key}
        )
    required = ROWS_PER_BIN * (1 + len(REPLICATE_SEEDS))
    output = []
    for target_bin in TARGET_BINS:
        available = sorted(
            by_bin[target_bin], key=lambda row: (row["selection_key"], row["sample_id"])
        )
        if len(available) < required:
            raise ValueError(f"target bin {target_bin} has insufficient samples")
        for position, row in enumerate(available[:required]):
            if position < ROWS_PER_BIN:
                split, replicate_seed = "evaluation", None
            else:
                split = "training"
                replicate_seed = REPLICATE_SEEDS[
                    (position - ROWS_PER_BIN) // ROWS_PER_BIN
                ]
            output.append({**row, "split": split, "replicate_seed": replicate_seed})
    if len(output) != 1200 or len({row["sample_id"] for row in output}) != 1200:
        raise RuntimeError("frozen Oxford-IIIT Pet cohort is incomplete or duplicated")
    return sorted(
        output,
        key=lambda row: (
            row["split"],
            row["replicate_seed"] or 0,
            row["target_bin"],
            row["sample_id"],
        ),
    )


def prepare_inputs(input_root: Path = DEFAULT_INPUT_ROOT) -> dict[str, Any]:
    input_root = input_root.resolve()
    images_archive = input_root / IMAGES_ARCHIVE
    annotations_archive = input_root / ANNOTATIONS_ARCHIVE
    cohort_path = input_root / "cohort.json"
    manifest_path = input_root / "manifest.json"
    image_root = input_root / "images"
    mask_root = input_root / "trimaps"
    if cohort_path.exists() or manifest_path.exists() or image_root.exists() or mask_root.exists():
        raise FileExistsError("prepared Oxford-IIIT Pet inputs already exist")
    rows = selected_rows(images_archive, annotations_archive)
    image_root.mkdir(parents=True, exist_ok=False)
    mask_root.mkdir(parents=True, exist_ok=False)
    selected_ids = {str(row["sample_id"]) for row in rows}
    with tarfile.open(images_archive, "r|gz") as images:
        for member in images:
            if not (member.isfile() and member.name.startswith("images/") and member.name.endswith(".jpg")):
                continue
            sample_id = Path(member.name).stem
            if sample_id not in selected_ids:
                continue
            handle = images.extractfile(member)
            if handle is None:
                raise ValueError(f"selected image cannot be read: {sample_id}")
            (image_root / f"{sample_id}.jpg").write_bytes(handle.read())
    with tarfile.open(annotations_archive, "r|gz") as annotations:
        for member in annotations:
            if not (
                member.isfile()
                and member.name.startswith("annotations/trimaps/")
                and member.name.endswith(".png")
            ):
                continue
            sample_id = Path(member.name).stem
            if sample_id not in selected_ids:
                continue
            handle = annotations.extractfile(member)
            if handle is None:
                raise ValueError(f"selected trimap cannot be read: {sample_id}")
            (mask_root / f"{sample_id}.png").write_bytes(handle.read())
    records = []
    for row in rows:
        sample_id = str(row["sample_id"])
        image_path, mask_path = image_root / f"{sample_id}.jpg", mask_root / f"{sample_id}.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise ValueError(f"selected sample cannot be extracted: {sample_id}")
        with Image.open(image_path) as source:
            source.verify()
        with Image.open(image_path) as source, Image.open(mask_path) as mask_source:
            image_size = tuple(source.size)
            mask = np.asarray(mask_source)
        foreground, pixels, target = foreground_fraction(mask)
        if image_size != (int(mask.shape[1]), int(mask.shape[0])):
            raise ValueError(f"image/trimap dimensions differ: {sample_id}")
        if (
            foreground != row["foreground_and_boundary_pixels"]
            or pixels != row["pixel_count"]
            or target != row["foreground_fraction"]
        ):
            raise ValueError(f"selected target changed during extraction: {sample_id}")
        records.append(
            {
                **row,
                "width": image_size[0],
                "height": image_size[1],
                "image_sha256": sha256_path(image_path),
                "trimap_sha256": sha256_path(mask_path),
            }
        )
    cohort = {
        "schema_version": 1,
        "dataset_id": "oxford_iiit_pet_foreground_occupancy",
        "target": "fraction_of_pixels_labeled_foreground_or_boundary_in_official_trimap",
        "target_valid_range": [MIN_TARGET, MAX_TARGET],
        "selection_seed": SELECTION_SEED,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "target_bins": len(TARGET_BINS),
        "binning": "equal-count rank bins over all 7334 valid official-list samples",
        "rows_per_bin_per_split": ROWS_PER_BIN,
        "evaluation_rows": ROWS_PER_BIN * len(TARGET_BINS),
        "training_rows_per_replicate": ROWS_PER_BIN * len(TARGET_BINS),
        "union_rows": len(records),
        "rows": records,
    }
    cohort_path.write_text(canonical_json(cohort) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "prepared_outcome_free_inputs",
        "official_page": OFFICIAL_PAGE,
        "license_scope": "research_use_per_archive_README; official_page_also_links_CC_BY_SA_4_0",
        "archives": {
            IMAGES_ARCHIVE: {
                "url": OFFICIAL_IMAGES_URL,
                "size_bytes": images_archive.stat().st_size,
                "sha256": IMAGES_SHA256,
            },
            ANNOTATIONS_ARCHIVE: {
                "url": OFFICIAL_ANNOTATIONS_URL,
                "size_bytes": annotations_archive.stat().st_size,
                "sha256": ANNOTATIONS_SHA256,
            },
        },
        "cohort": {
            "path": cohort_path.name,
            "sha256": sha256_path(cohort_path),
            "identity_sha256": sha256_payload(cohort),
        },
        "outcomes_inspected": False,
    }
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def load_cohort(input_root: Path = DEFAULT_INPUT_ROOT) -> dict[str, Any]:
    input_root = input_root.resolve()
    manifest_path, cohort_path = input_root / "manifest.json", input_root / "cohort.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_outcome_free_inputs":
        raise ValueError("Oxford-IIIT Pet input manifest status is invalid")
    if sha256_path(input_root / IMAGES_ARCHIVE) != IMAGES_SHA256:
        raise ValueError("Oxford-IIIT Pet image archive identity changed")
    if sha256_path(input_root / ANNOTATIONS_ARCHIVE) != ANNOTATIONS_SHA256:
        raise ValueError("Oxford-IIIT Pet annotation archive identity changed")
    if sha256_path(cohort_path) != manifest["cohort"]["sha256"]:
        raise ValueError("Oxford-IIIT Pet cohort file identity changed")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if sha256_payload(cohort) != manifest["cohort"]["identity_sha256"]:
        raise ValueError("Oxford-IIIT Pet cohort payload identity changed")
    if len(cohort.get("rows", ())) != 1200:
        raise ValueError("Oxford-IIIT Pet cohort row count changed")
    for row in cohort["rows"]:
        image_path = input_root / "images" / f"{row['sample_id']}.jpg"
        mask_path = input_root / "trimaps" / f"{row['sample_id']}.png"
        if sha256_path(image_path) != row["image_sha256"]:
            raise ValueError(f"cohort image identity changed: {row['sample_id']}")
        if sha256_path(mask_path) != row["trimap_sha256"]:
            raise ValueError(f"cohort trimap identity changed: {row['sample_id']}")
    return cohort


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(canonical_json(prepare_inputs(args.input_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
