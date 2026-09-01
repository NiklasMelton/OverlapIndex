"""Prepare the outcome-free, age-stratified UTKFace cohort bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping, Sequence

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = ROOT / "artifacts" / "continuous_real_age_panel" / "inputs"
ARCHIVE_NAME = "UTKFace.tar.gz"
ARCHIVE_SHA256 = "2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7"
MIRROR_REPOSITORY = "MJZU/UTKFace-Cropped"
MIRROR_REVISION = "9ed55e9d06440324163069a9053bbb7e7d8e4017"
OFFICIAL_PAGE = "https://susanqq.github.io/UTKFace/"
SELECTION_SEED = 2026090101
REPLICATE_SEEDS = (4103, 4201, 4303, 4409, 4507)
AGE_BINS = tuple(range(9))
ROWS_PER_BIN = 25


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


def _member_label(name: str) -> tuple[str, int, int, int] | None:
    path = PurePosixPath(name)
    if len(path.parts) != 2 or path.parts[0] != "UTKFace":
        return None
    filename = path.name
    pieces = filename.split("_")
    if len(pieces) < 4:
        return None
    try:
        age, gender, race = (int(pieces[index]) for index in range(3))
    except ValueError:
        return None
    if not (0 <= age <= 116 and gender in (0, 1) and race in range(5)):
        return None
    return filename, age, gender, race


def selected_rows(archive: Path) -> list[dict[str, Any]]:
    if sha256_path(archive) != ARCHIVE_SHA256:
        raise ValueError("UTKFace archive SHA-256 mismatch")
    by_bin: dict[int, list[dict[str, Any]]] = {value: [] for value in AGE_BINS}
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            if not member.isfile():
                if member.isdir() and member.name.rstrip("/") == "UTKFace":
                    continue
                raise ValueError(f"UTKFace archive contains a non-file member: {member.name}")
            parsed = _member_label(member.name)
            if parsed is None:
                # The published archive contains a small number of malformed
                # filename annotations. They are excluded by the closed parser
                # before cohort selection and can never enter a target array.
                continue
            filename, age, gender, race = parsed
            age_bin = age // 10
            if age_bin not in by_bin:
                continue
            key = hashlib.sha256(
                f"{SELECTION_SEED}|{member.name}".encode("utf-8")
            ).hexdigest()
            by_bin[age_bin].append(
                {
                    "member": member.name,
                    "sample_id": filename,
                    "age": age,
                    "age_bin": age_bin,
                    "gender": gender,
                    "race": race,
                    "selection_key": key,
                }
            )
    required = ROWS_PER_BIN * (1 + len(REPLICATE_SEEDS))
    output: list[dict[str, Any]] = []
    for age_bin in AGE_BINS:
        available = sorted(by_bin[age_bin], key=lambda row: (row["selection_key"], row["sample_id"]))
        if len(available) < required:
            raise ValueError(f"age bin {age_bin} has fewer than {required} valid images")
        chosen = available[:required]
        for position, row in enumerate(chosen):
            if position < ROWS_PER_BIN:
                split, replicate_seed = "evaluation", None
            else:
                replicate_position = (position - ROWS_PER_BIN) // ROWS_PER_BIN
                split = "training"
                replicate_seed = REPLICATE_SEEDS[replicate_position]
            output.append(
                {
                    **row,
                    "split": split,
                    "replicate_seed": replicate_seed,
                }
            )
    if len(output) != 1350 or len({row["sample_id"] for row in output}) != len(output):
        raise RuntimeError("frozen UTKFace cohort bank is incomplete or duplicated")
    return sorted(output, key=lambda row: (row["split"], row["replicate_seed"] or 0, row["age_bin"], row["sample_id"]))


def prepare_inputs(input_root: Path = DEFAULT_INPUT_ROOT) -> dict[str, Any]:
    input_root = input_root.resolve()
    archive = input_root / ARCHIVE_NAME
    cohort_path = input_root / "cohort.json"
    manifest_path = input_root / "manifest.json"
    images_root = input_root / "images"
    if cohort_path.exists() or manifest_path.exists() or images_root.exists():
        raise FileExistsError("prepared UTKFace inputs already exist")
    rows = selected_rows(archive)
    images_root.mkdir(parents=True, exist_ok=False)
    members = {row["member"]: row for row in rows}
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            if member.name not in members:
                continue
            handle = source.extractfile(member)
            if handle is None:
                raise ValueError(f"selected image cannot be read: {member.name}")
            data = handle.read()
            destination = images_root / members[member.name]["sample_id"]
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_bytes(data)
            temporary.replace(destination)
    records = []
    for row in rows:
        image_path = images_root / row["sample_id"]
        if not image_path.is_file():
            raise RuntimeError(f"selected image is missing: {row['sample_id']}")
        with Image.open(image_path) as image:
            image.verify()
        records.append({**row, "image_sha256": sha256_path(image_path)})
    cohort = {
        "schema_version": 1,
        "dataset_id": "utkface_aligned_age_0_89_balanced_decades",
        "target": "integer_age_from_filename_as_continuous_scalar",
        "selection_seed": SELECTION_SEED,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "age_bins": [[10 * value, 10 * value + 9] for value in AGE_BINS],
        "rows_per_bin_per_split": ROWS_PER_BIN,
        "evaluation_rows": ROWS_PER_BIN * len(AGE_BINS),
        "training_rows_per_replicate": ROWS_PER_BIN * len(AGE_BINS),
        "union_rows": len(records),
        "rows": records,
    }
    cohort_path.write_text(canonical_json(cohort) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "prepared_outcome_free_inputs",
        "official_page": OFFICIAL_PAGE,
        "license_scope": "non_commercial_research_only_per_official_page",
        "mirror_repository": MIRROR_REPOSITORY,
        "mirror_revision": MIRROR_REVISION,
        "archive": {
            "path": ARCHIVE_NAME,
            "size_bytes": archive.stat().st_size,
            "sha256": ARCHIVE_SHA256,
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
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    cohort_path = input_root / str(manifest["cohort"]["path"])
    if manifest.get("status") != "prepared_outcome_free_inputs":
        raise ValueError("input manifest status is invalid")
    if sha256_path(input_root / ARCHIVE_NAME) != ARCHIVE_SHA256:
        raise ValueError("input archive identity changed")
    if sha256_path(cohort_path) != manifest["cohort"]["sha256"]:
        raise ValueError("cohort manifest identity changed")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if sha256_payload(cohort) != manifest["cohort"]["identity_sha256"]:
        raise ValueError("cohort payload identity changed")
    if len(cohort.get("rows", ())) != 1350:
        raise ValueError("cohort row count changed")
    for row in cohort["rows"]:
        image = input_root / "images" / row["sample_id"]
        if sha256_path(image) != row["image_sha256"]:
            raise ValueError(f"cohort image identity changed: {row['sample_id']}")
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
