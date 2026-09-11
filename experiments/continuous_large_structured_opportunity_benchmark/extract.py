"""Extract new CUB/VOC scout embeddings and bind reused embedding banks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from experiments.continuous_fresh_structured_opportunity_discovery import extract as fresh_extract
from experiments.continuous_coco_pose_opportunity_replication import extract as coco_extract
from experiments.continuous_large_structured_opportunity_benchmark import prepare
from experiments.continuous_real_age_panel import extract as age_extract
from experiments.continuous_real_pet_area_panel import extract as pet_extract
from experiments.continuous_real_pet_area_panel import prepare as pet_prepare


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = prepare.DEFAULT_INPUT_ROOT
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts" / "continuous_large_structured_opportunity_benchmark" / "embeddings"
)
BACKBONES = fresh_extract.BACKBONES
NEW_DATASETS = (prepare.CUB_ID, prepare.VOC_ID)
REUSED_DATASETS = (prepare.LSP_ID, prepare.COCO_ID, prepare.PET_ID)
SPEC_BY_ID = fresh_extract.SPEC_BY_ID


def _values_sha256(matrix: np.ndarray) -> str:
    return prepare.sha256_payload(
        {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "bytes_sha256": __import__("hashlib").sha256(
                np.ascontiguousarray(matrix).tobytes(order="C")
            ).hexdigest(),
        }
    )


def _paths(output_root: Path, dataset_id: str, backbone: str) -> tuple[Path, Path]:
    root = output_root / dataset_id
    return root / f"{backbone}.npy", root / f"{backbone}.json"


def _images(input_root: Path, rows: Sequence[Mapping[str, Any]]) -> list[Image.Image]:
    output = []
    for row in rows:
        with Image.open(input_root / "prepared" / str(row["image_path"])) as source:
            output.append(source.convert("RGB").copy())
    return output


def _validate_new_cache(
    output_root: Path,
    dataset_id: str,
    backbone: str,
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    matrix_path, manifest_path = _paths(output_root, dataset_id, backbone)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    rows = cohort["datasets"][dataset_id]["rows"]
    spec = SPEC_BY_ID[backbone]
    checks = (
        manifest.get("dataset_id") == dataset_id,
        manifest.get("backbone_id") == backbone,
        manifest.get("spec_sha256") == prepare.sha256_payload(spec),
        manifest.get("cohort_sha256") == prepare.sha256_payload(rows),
        manifest.get("sample_ids_sha256")
        == prepare.sha256_payload([row["sample_id"] for row in rows]),
        manifest.get("matrix_sha256") == prepare.sha256_path(matrix_path),
        manifest.get("values_sha256") == _values_sha256(np.asarray(matrix)),
        tuple(matrix.shape) == (len(rows), int(spec["dimension"])),
        matrix.dtype == np.dtype("float32"),
        bool(np.isfinite(np.asarray(matrix)).all()),
    )
    if not all(checks):
        raise ValueError(f"large structured embedding cache failed: {dataset_id}/{backbone}")
    return manifest


def _reused_sources() -> dict[str, dict[str, Path]]:
    return {
        prepare.LSP_ID: {
            backbone: fresh_extract.DEFAULT_OUTPUT_ROOT / "lsp_2000_pose" / f"{backbone}.npy"
            for backbone in BACKBONES
        },
        prepare.COCO_ID: {
            backbone: coco_extract.DEFAULT_OUTPUT_ROOT / f"{backbone}.npy"
            for backbone in BACKBONES
        },
        prepare.PET_ID: {
            backbone: pet_extract.DEFAULT_OUTPUT_ROOT / f"{backbone}.npy"
            for backbone in BACKBONES
        },
    }


def validate_reused_banks() -> list[dict[str, Any]]:
    fresh_cohort, fresh_bank = fresh_extract.load_bank()
    del fresh_cohort, fresh_bank
    coco_cohort, coco_bank = coco_extract.load_bank()
    del coco_cohort, coco_bank
    pet_cohort = pet_prepare.load_cohort(pet_prepare.DEFAULT_INPUT_ROOT)
    pet_specs = {str(spec["backbone_id"]): spec for spec in pet_extract.BACKBONE_SPECS}
    for backbone in BACKBONES:
        pet_extract._validate_cache(pet_extract.DEFAULT_OUTPUT_ROOT, pet_specs[backbone], pet_cohort)
    records = []
    for dataset_id, paths in _reused_sources().items():
        for backbone, path in paths.items():
            if not path.is_file():
                raise ValueError(f"reused embedding matrix is missing: {dataset_id}/{backbone}")
            matrix = np.load(path, mmap_mode="r", allow_pickle=False)
            if matrix.dtype != np.dtype("float32") or matrix.ndim != 2 or not np.isfinite(np.asarray(matrix)).all():
                raise ValueError(f"reused embedding matrix is invalid: {dataset_id}/{backbone}")
            records.append(
                {
                    "dataset_id": dataset_id,
                    "backbone_id": backbone,
                    "path": str(path.relative_to(ROOT)),
                    "rows": int(matrix.shape[0]),
                    "dimension": int(matrix.shape[1]),
                    "matrix_sha256": prepare.sha256_path(path),
                }
            )
    return records


def run_extraction(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    protocol = prepare.load_protocol()
    input_root, output_root = input_root.resolve(), output_root.resolve()
    cohort = prepare.load_cohort(input_root)
    model_inputs = fresh_extract.validate_model_inputs()
    reused = validate_reused_banks()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        if not resume:
            raise FileExistsError("large structured terminal embedding manifest exists")
        manifest = json.loads(terminal.read_text(encoding="utf-8"))
        for dataset_id in NEW_DATASETS:
            for backbone in BACKBONES:
                _validate_new_cache(output_root, dataset_id, backbone, cohort)
        if manifest.get("reused_matrices") != reused:
            raise ValueError("large structured reused embedding identities changed")
        return manifest
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    total = len(NEW_DATASETS) * len(BACKBONES)
    completed = 0
    for dataset_id in NEW_DATASETS:
        rows = cohort["datasets"][dataset_id]["rows"]
        images = _images(input_root, rows)
        for backbone in BACKBONES:
            completed += 1
            matrix_path, manifest_path = _paths(output_root, dataset_id, backbone)
            if matrix_path.exists() or manifest_path.exists():
                if not (resume and matrix_path.exists() and manifest_path.exists()):
                    raise FileExistsError(f"partial large structured cache exists: {dataset_id}/{backbone}")
                records.append(_validate_new_cache(output_root, dataset_id, backbone, cohort))
                print(f"validated {completed}/{total}: {dataset_id}/{backbone}", flush=True)
                continue
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            spec = SPEC_BY_ID[backbone]
            matrix = age_extract._extract(spec, images)
            temporary = matrix_path.with_name(f".{matrix_path.name}.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, matrix, allow_pickle=False)
            temporary.replace(matrix_path)
            record = {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "backbone_id": backbone,
                "rows": len(rows),
                "dimension": int(matrix.shape[1]),
                "dtype": "float32",
                "cohort_sha256": prepare.sha256_payload(rows),
                "sample_ids_sha256": prepare.sha256_payload([row["sample_id"] for row in rows]),
                "spec": spec,
                "spec_sha256": prepare.sha256_payload(spec),
                "matrix_sha256": prepare.sha256_path(matrix_path),
                "values_sha256": _values_sha256(matrix),
            }
            prepare._atomic_json(manifest_path, record)
            records.append(record)
            print(f"extracted {completed}/{total}: {dataset_id}/{backbone}", flush=True)
        del images
    manifest = {
        "schema_version": 1,
        "status": "completed_large_structured_opportunity_embedding_extraction",
        "protocol_sha256": prepare.sha256_path(prepare.PROTOCOL_PATH),
        "input_manifest_sha256": prepare.sha256_path(input_root / "prepared" / "manifest.json"),
        "cohort_sha256": prepare.sha256_path(input_root / "prepared" / "cohort.json"),
        "new_dataset_ids": list(NEW_DATASETS),
        "reused_dataset_ids": list(REUSED_DATASETS),
        "backbone_ids": list(BACKBONES),
        "model_inputs": model_inputs,
        "new_records": records,
        "reused_matrices": reused,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "Pillow": importlib.metadata.version("Pillow"),
            "torch": importlib.metadata.version("torch"),
            "torchvision": importlib.metadata.version("torchvision"),
            "timm": importlib.metadata.version("timm"),
            "transformers": importlib.metadata.version("transformers"),
            "open_clip_torch": importlib.metadata.version("open_clip_torch"),
            "vertebrae": importlib.metadata.version("vertebrae"),
            "device": pet_extract.DEVICE,
            "batch_size": pet_extract.BATCH_SIZE,
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "predictive_outcomes_inspected": False,
        "overlapindex_fit_count": 0,
    }
    prepare._atomic_json(terminal, manifest)
    return manifest


def load_bank(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    cohort = prepare.load_cohort(input_root.resolve())
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed_large_structured_opportunity_embedding_extraction"
        or manifest.get("protocol_sha256") != prepare.sha256_path(prepare.PROTOCOL_PATH)
        or tuple(manifest.get("backbone_ids", ())) != BACKBONES
    ):
        raise ValueError("large structured embedding terminal identity changed")
    reused = validate_reused_banks()
    if manifest.get("reused_matrices") != reused:
        raise ValueError("large structured reused embedding identities changed")
    sources = _reused_sources()
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for dataset_id in prepare.DATASETS:
        matrices[dataset_id] = {}
        for backbone in BACKBONES:
            if dataset_id in NEW_DATASETS:
                _validate_new_cache(output_root, dataset_id, backbone, cohort)
                path, _ = _paths(output_root, dataset_id, backbone)
            else:
                path = sources[dataset_id][backbone]
            matrices[dataset_id][backbone] = np.load(path, mmap_mode="r", allow_pickle=False)
    return cohort, matrices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(prepare.canonical_json(run_extraction(args.input_root, args.output_root, resume=args.resume)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
