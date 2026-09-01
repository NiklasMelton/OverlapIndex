"""Prepare outcome-free Aircraft and GTSRB box-occupancy cohorts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FUSED_INPUT_ROOT = Path(
    "/Users/niklasmelton/.codex/worktrees/b2ea/OverlapIndex/"
    "artifacts/fused3_confirmation_v2/inputs"
)
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_bbox_panels" / "inputs"
REGISTRY_SHA256 = "a6fd86518562cca32851e8ddf2ae0d874b61c32433614ba3ee91a8bbc86861b9"
DATASETS = ("torchvision_gtsrb", "torchvision_fgvc_aircraft")
REPLICATE_SEEDS = (4103, 4201, 4303, 4409, 4507)
TARGET_BINS = 10
TRAIN_ROWS = 200
EVALUATION_ROWS = 200
TRAIN_BANK_SEED = 2026091101
EVALUATION_SEED = 2026091102


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


def _load_registry(input_root: Path) -> dict[str, Any]:
    path = input_root / "audited_registry.json"
    if sha256_path(path) != REGISTRY_SHA256:
        raise ValueError("frozen FUSED3 registry SHA-256 mismatch")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("status") != "audited_complete" or registry.get("outcomes_inspected") is not False:
        raise ValueError("FUSED3 registry is not an outcome-free audited input")
    records = registry.get("datasets")
    if not isinstance(records, list) or not set(DATASETS).issubset(
        {row.get("dataset_id") for row in records if isinstance(row, Mapping)}
    ):
        raise ValueError("FUSED3 registry is missing a required dataset")
    return registry


def _dataset_record(registry: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    rows = [row for row in registry["datasets"] if row.get("dataset_id") == dataset_id]
    if len(rows) != 1:
        raise ValueError(f"registry dataset identity is not unique: {dataset_id}")
    return rows[0]


def _provider_datasets(dataset_id: str, raw_root: Path) -> tuple[Any, Any]:
    import torchvision
    from torchvision import datasets

    if str(torchvision.__version__) != "0.23.0":
        raise ValueError("box-cohort preparation requires torchvision 0.23.0")
    common = {"root": str(raw_root), "download": False, "transform": None}
    if dataset_id == "torchvision_gtsrb":
        return datasets.GTSRB(split="train", **common), datasets.GTSRB(split="test", **common)
    if dataset_id == "torchvision_fgvc_aircraft":
        return (
            datasets.FGVCAircraft(split="trainval", annotation_level="variant", **common),
            datasets.FGVCAircraft(split="test", annotation_level="variant", **common),
        )
    raise ValueError(f"unknown box dataset {dataset_id!r}")


def box_fraction(width: int, height: int, x1: int, y1: int, x2: int, y2: int) -> float:
    values = (width, height, x1, y1, x2, y2)
    if any(type(value) is not int for value in values):
        raise TypeError("box coordinates and dimensions must be strict integers")
    if width <= 0 or height <= 0 or not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        raise ValueError("box coordinates are outside the image")
    result = ((x2 - x1 + 1) * (y2 - y1 + 1)) / float(width * height)
    if not (0.0 < result <= 1.0):
        raise ValueError("box fraction is outside (0, 1]")
    return float(result)


def _read_gtsrb_annotations(raw_root: Path) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]], list[dict[str, str]]]:
    training: dict[tuple[str, str], dict[str, str]] = {}
    source_rows: list[dict[str, str]] = []
    training_root = raw_root / "gtsrb" / "GTSRB" / "Training"
    for path in sorted(training_root.glob("*/GT-*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                key = (path.parent.name, row["Filename"])
                if key in training:
                    raise ValueError("duplicate GTSRB training annotation")
                training[key] = dict(row)
        source_rows.append({"path": str(path), "sha256": sha256_path(path)})
    test_path = raw_root / "gtsrb" / "GT-final_test.csv"
    with test_path.open(newline="", encoding="utf-8-sig") as handle:
        test = {row["Filename"]: dict(row) for row in csv.DictReader(handle, delimiter=";")}
    source_rows.append({"path": str(test_path), "sha256": sha256_path(test_path)})
    if len(training) != 26640 or len(test) != 12630:
        raise ValueError("GTSRB annotation grid is incomplete")
    return training, test, source_rows


def _gtsrb_targets(record: Mapping[str, Any], raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    train, test = _provider_datasets("torchvision_gtsrb", raw_root)
    train_annotations, test_annotations, sources = _read_gtsrb_annotations(raw_root)
    output = []
    for sample_id in record["cohort_manifest"]["union"]["sample_ids"]:
        _, split, position_text = sample_id.rsplit("/", 2)
        position = int(position_text)
        provider = train if split == "train" else test
        path = Path(provider._samples[position][0])
        annotation = (
            train_annotations[(path.parent.name, path.name)]
            if split == "train"
            else test_annotations[path.name]
        )
        width, height, x1, y1, x2, y2 = (
            int(annotation[name])
            for name in ("Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
        )
        if Image.open(path).size != (width, height):
            raise ValueError("GTSRB annotation dimensions differ from image bytes")
        output.append(
            {
                "sample_id": sample_id,
                "source_split": split,
                "target": box_fraction(width, height, x1, y1, x2, y2),
            }
        )
    return output, sources


def _aircraft_targets(record: Mapping[str, Any], raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    train, test = _provider_datasets("torchvision_fgvc_aircraft", raw_root)
    box_path = raw_root / "fgvc-aircraft-2013b" / "data" / "images_box.txt"
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for line in box_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] in boxes:
            raise ValueError("Aircraft box annotations are malformed or duplicated")
        boxes[parts[0]] = tuple(int(value) for value in parts[1:])
    if len(boxes) != 10000:
        raise ValueError("Aircraft box annotation grid is incomplete")
    output = []
    for sample_id in record["cohort_manifest"]["union"]["sample_ids"]:
        _, split, position_text = sample_id.rsplit("/", 2)
        position = int(position_text)
        provider = train if split == "trainval" else test
        path = Path(provider._image_files[position])
        width, height = Image.open(path).size
        x1, y1, x2, y2 = boxes[path.stem]
        output.append(
            {
                "sample_id": sample_id,
                "source_split": split,
                "target": box_fraction(width, height, x1, y1, x2, y2),
            }
        )
    return output, [{"path": str(box_path), "sha256": sha256_path(box_path)}]


def _hash_order(sample_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    return digest, sample_id


def _target_bins(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (float(row["target"]), str(row["sample_id"])))
    bins = [list(chunk) for chunk in np.array_split(np.asarray(ordered, dtype=object), TARGET_BINS)]
    if len(bins) != TARGET_BINS or any(not chunk for chunk in bins):
        raise ValueError("target-rank bins are incomplete")
    return bins


def select_cohorts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    training = [row for row in rows if row["source_split"] in {"train", "trainval"}]
    evaluation = [row for row in rows if row["source_split"] == "test"]
    train_per_bin = TRAIN_ROWS // TARGET_BINS
    eval_per_bin = EVALUATION_ROWS // TARGET_BINS
    train_bins = _target_bins(training)
    eval_bins = _target_bins(evaluation)
    if any(len(chunk) < train_per_bin * len(REPLICATE_SEEDS) for chunk in train_bins):
        raise ValueError("not enough cached training rows for disjoint cohorts")
    if any(len(chunk) < eval_per_bin for chunk in eval_bins):
        raise ValueError("not enough cached evaluation rows")
    output: list[dict[str, Any]] = []
    for bin_index, chunk in enumerate(eval_bins):
        ordered = sorted(chunk, key=lambda row: _hash_order(str(row["sample_id"]), EVALUATION_SEED + bin_index))
        for row in ordered[:eval_per_bin]:
            output.append({**row, "split": "evaluation", "replicate_seed": None, "target_bin": bin_index})
    for replicate_index, replicate_seed in enumerate(REPLICATE_SEEDS):
        for bin_index, chunk in enumerate(train_bins):
            ordered = sorted(chunk, key=lambda row: _hash_order(str(row["sample_id"]), TRAIN_BANK_SEED + bin_index))
            start = replicate_index * train_per_bin
            for row in ordered[start : start + train_per_bin]:
                output.append(
                    {
                        **row,
                        "split": "training",
                        "replicate_seed": replicate_seed,
                        "target_bin": bin_index,
                    }
                )
    if len(output) != EVALUATION_ROWS + TRAIN_ROWS * len(REPLICATE_SEEDS):
        raise RuntimeError("box cohort has the wrong row count")
    ids = [str(row["sample_id"]) for row in output]
    if len(ids) != len(set(ids)):
        raise RuntimeError("box cohorts are not disjoint")
    return output


def build_cohort(input_root: Path) -> dict[str, Any]:
    registry = _load_registry(input_root)
    datasets = []
    for dataset_id in DATASETS:
        record = _dataset_record(registry, dataset_id)
        raw_root = Path(record["source"]["raw_root"])
        if dataset_id == "torchvision_gtsrb":
            targets, annotation_sources = _gtsrb_targets(record, raw_root)
            family = "gtsrb_sign_roi_occupancy"
        else:
            targets, annotation_sources = _aircraft_targets(record, raw_root)
            family = "fgvc_aircraft_bbox_occupancy"
        rows = select_cohorts(targets)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "dataset_group": dataset_id,
                "family": family,
                "target_recipe": "inclusive_annotation_box_area_divided_by_raw_image_area",
                "target_sha256": sha256_payload(
                    [[row["sample_id"], row["target"]] for row in targets]
                ),
                "annotation_sources": annotation_sources,
                "rows": rows,
                "rows_sha256": sha256_payload(rows),
            }
        )
    return {
        "schema_version": 1,
        "status": "prepared_outcome_free_bbox_cohorts",
        "registry_sha256": REGISTRY_SHA256,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "target_bins": TARGET_BINS,
        "training_rows_per_replicate": TRAIN_ROWS,
        "evaluation_rows": EVALUATION_ROWS,
        "training_bank_seed": TRAIN_BANK_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "datasets": datasets,
    }


def prepare(input_root: Path = DEFAULT_FUSED_INPUT_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("box-panel input root already exists")
    cohort = build_cohort(input_root.resolve())
    output_root.mkdir(parents=True, exist_ok=False)
    cohort_path = output_root / "cohort.json"
    cohort_path.write_text(canonical_json(cohort) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed_outcome_free_bbox_cohort_preparation",
        "registry_path": str((input_root / "audited_registry.json").resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "cohort_sha256": sha256_path(cohort_path),
        "cohort_bytes": cohort_path.stat().st_size,
    }
    (output_root / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def load_cohort(input_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    manifest_path = input_root / "manifest.json"
    cohort_path = input_root / "cohort.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_outcome_free_bbox_cohort_preparation":
        raise ValueError("box-panel input manifest is not complete")
    if manifest.get("registry_sha256") != REGISTRY_SHA256:
        raise ValueError("box-panel registry identity changed")
    if sha256_path(cohort_path) != manifest.get("cohort_sha256"):
        raise ValueError("box-panel cohort hash mismatch")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("status") != "prepared_outcome_free_bbox_cohorts":
        raise ValueError("box-panel cohort is not prepared")
    return cohort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-input-root", type=Path, default=DEFAULT_FUSED_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(canonical_json(prepare(args.fused_input_root, args.output_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
