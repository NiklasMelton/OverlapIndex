"""Prepare a deterministic continuous DTD texture-gradient target and cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from experiments.continuous_real_bbox_panels import prepare as bbox_prepare


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FUSED_INPUT_ROOT = bbox_prepare.DEFAULT_FUSED_INPUT_ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_dtd_texture_confirmation" / "inputs"
REGISTRY_SHA256 = bbox_prepare.REGISTRY_SHA256
DATASET_ID = "torchvision_dtd"
FAMILY = "dtd_texture_gradient_energy"
REPLICATE_SEEDS = bbox_prepare.REPLICATE_SEEDS
TARGET_BINS = bbox_prepare.TARGET_BINS
TRAIN_ROWS = bbox_prepare.TRAIN_ROWS
EVALUATION_ROWS = bbox_prepare.EVALUATION_ROWS
TRAIN_BANK_SEED = bbox_prepare.TRAIN_BANK_SEED
EVALUATION_SEED = bbox_prepare.EVALUATION_SEED
TARGET_IMAGE_SIZE = 224
TARGET_RECIPE = (
    "PIL RGB source converted to 8-bit L, bilinear-resized to 224x224, divided by 255, "
    "then sqrt(0.5*(mean(horizontal adjacent difference squared)+"
    "mean(vertical adjacent difference squared)))"
)


def texture_gradient_energy(image: Image.Image) -> float:
    """Return the frozen resolution-normalized grayscale RMS adjacent gradient."""
    if not isinstance(image, Image.Image):
        raise TypeError("texture target requires a PIL image")
    grayscale = image.convert("L").resize(
        (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    values = np.asarray(grayscale, dtype=np.float64) / 255.0
    if values.shape != (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE) or not np.isfinite(values).all():
        raise ValueError("texture target image conversion is invalid")
    horizontal = np.diff(values, axis=1)
    vertical = np.diff(values, axis=0)
    result = np.sqrt(
        0.5 * (float(np.mean(horizontal * horizontal)) + float(np.mean(vertical * vertical)))
    )
    if not np.isfinite(result) or not (0.0 <= result <= 1.0):
        raise ValueError("texture-gradient target is outside [0, 1]")
    return float(result)


def _load_registry(input_root: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    registry = bbox_prepare._load_registry(input_root)
    record = bbox_prepare._dataset_record(registry, DATASET_ID)
    return registry, record


def _provider_by_split(raw_root: Path) -> dict[str, Any]:
    import torchvision
    from torchvision import datasets

    if str(torchvision.__version__) != "0.23.0":
        raise ValueError("DTD target preparation requires torchvision 0.23.0")
    common = {"root": str(raw_root), "partition": 1, "download": False, "transform": None}
    return {
        split: datasets.DTD(split=split, **common)
        for split in ("train", "val", "test")
    }


def _split_from_id(sample_id: str) -> tuple[str, int]:
    parts = sample_id.split("/")
    if len(parts) != 3 or parts[0] != DATASET_ID:
        raise ValueError("DTD union sample ID is malformed")
    split_token = parts[1]
    expected = {
        "train_partition-1": "train",
        "val_partition-1": "val",
        "test_partition-1": "test",
    }
    if split_token not in expected or not parts[2].isdigit():
        raise ValueError("DTD union sample split or position is malformed")
    return expected[split_token], int(parts[2])


def build_targets(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_root = Path(record["source"]["raw_root"])
    providers = _provider_by_split(raw_root)
    output: list[dict[str, Any]] = []
    union_ids = record["cohort_manifest"]["union"]["sample_ids"]
    if len(union_ids) != len(set(union_ids)):
        raise ValueError("DTD union sample IDs are duplicated")
    for sample_id in union_ids:
        split, position = _split_from_id(str(sample_id))
        provider = providers[split]
        if position >= len(provider):
            raise ValueError("DTD union position is outside its provider split")
        path = Path(provider._image_files[position])
        with Image.open(path) as image:
            target = texture_gradient_energy(image)
        output.append(
            {
                "sample_id": str(sample_id),
                "source_split": "test" if split == "test" else "train",
                "provider_split": split,
                "target": target,
            }
        )
    return output


def build_cohort(input_root: Path) -> dict[str, Any]:
    _, record = _load_registry(input_root)
    targets = build_targets(record)
    rows = bbox_prepare.select_cohorts(targets)
    return {
        "schema_version": 1,
        "status": "prepared_outcome_free_dtd_texture_target",
        "registry_sha256": REGISTRY_SHA256,
        "dataset_id": DATASET_ID,
        "dataset_group": DATASET_ID,
        "family": FAMILY,
        "target_recipe": TARGET_RECIPE,
        "target_image_size": TARGET_IMAGE_SIZE,
        "target_sha256": bbox_prepare.sha256_payload(
            [[row["sample_id"], row["target"]] for row in targets]
        ),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "target_bins": TARGET_BINS,
        "training_rows_per_replicate": TRAIN_ROWS,
        "evaluation_rows": EVALUATION_ROWS,
        "training_bank_seed": TRAIN_BANK_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "rows": rows,
        "rows_sha256": bbox_prepare.sha256_payload(rows),
        "source_manifest_sha256": record["source_manifest_sha256"],
        "embedding_union_sample_ids_sha256": record["sample_ids_sha256"],
    }


def prepare(
    input_root: Path = DEFAULT_FUSED_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("DTD texture input root already exists")
    cohort = build_cohort(input_root.resolve())
    output_root.mkdir(parents=True, exist_ok=False)
    cohort_path = output_root / "cohort.json"
    cohort_path.write_text(bbox_prepare.canonical_json(cohort) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed_outcome_free_dtd_texture_preparation",
        "registry_path": str((input_root / "audited_registry.json").resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "cohort_sha256": bbox_prepare.sha256_path(cohort_path),
        "cohort_bytes": cohort_path.stat().st_size,
    }
    (output_root / "manifest.json").write_text(
        bbox_prepare.canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def load_cohort(input_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    manifest_path = input_root / "manifest.json"
    cohort_path = input_root / "cohort.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_outcome_free_dtd_texture_preparation":
        raise ValueError("DTD texture input manifest is incomplete")
    if manifest.get("registry_sha256") != REGISTRY_SHA256:
        raise ValueError("DTD texture registry identity changed")
    if bbox_prepare.sha256_path(cohort_path) != manifest.get("cohort_sha256"):
        raise ValueError("DTD texture cohort hash mismatch")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("status") != "prepared_outcome_free_dtd_texture_target":
        raise ValueError("DTD texture cohort is not prepared")
    return cohort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-input-root", type=Path, default=DEFAULT_FUSED_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(bbox_prepare.canonical_json(prepare(args.fused_input_root, args.output_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
