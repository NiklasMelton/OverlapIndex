"""Extract the five frozen scout backbones for the nested diagnostic cohorts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
from PIL import Image

from experiments.continuous_real_age_panel import extract as age_extract
from experiments.continuous_ranking_reliability_recoverability import prepare


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = prepare.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_ranking_reliability_recoverability" / "embeddings"
BACKBONES = (
    "dinov2-small", "deit-tiny", "mobilenetv3-large", "resnet50", "convnext-tiny",
)
SPEC_BY_ID = {str(spec["backbone_id"]): spec for spec in age_extract.BACKBONE_SPECS}
SPECS = tuple(SPEC_BY_ID[backbone] for backbone in BACKBONES)
EXTRACTION_CHUNK_ROWS = 256


def _values_sha256(matrix: np.ndarray) -> str:
    return __import__("hashlib").sha256(np.ascontiguousarray(matrix).tobytes(order="C")).hexdigest()


def _paths(output_root: Path, dataset_id: str, backbone: str) -> tuple[Path, Path]:
    root = output_root / dataset_id
    return root / f"{backbone}.npy", root / f"{backbone}.json"


def _load_images(rows: Sequence[Mapping[str, Any]]) -> list[Image.Image]:
    images = []
    for row in rows:
        path = ROOT / str(row["image_path"])
        if prepare.sha256_path(path) != row["image_sha256"]:
            raise ValueError(f"diagnostic image identity changed: {row['sample_id']}")
        with Image.open(path) as source:
            images.append(source.convert("RGB").copy())
    return images


def _extract_chunked(spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    extractor = age_extract._build_extractor(spec)
    blocks = []
    try:
        for start in range(0, len(rows), EXTRACTION_CHUNK_ROWS):
            images = _load_images(rows[start : start + EXTRACTION_CHUNK_ROWS])
            values = {"image": images} if spec["provider"] == "openclip" else images
            outputs = extractor.transform_many(values)
            selected = [row for row in outputs if row.name == spec["output"]]
            if len(selected) != 1:
                raise ValueError("extractor did not emit exactly one frozen output")
            block = np.asarray(selected[0].embeddings, dtype=np.float32)
            expected_shape = (len(images), int(spec["dimension"]))
            if block.shape != expected_shape or not np.isfinite(block).all():
                raise ValueError(f"invalid diagnostic embedding block: {block.shape}")
            blocks.append(np.ascontiguousarray(block))
            del images, values, outputs, selected, block
    finally:
        for name in ("_model", "_processor", "_torch", "_preprocess", "_tokenizer"):
            if hasattr(extractor, name):
                setattr(extractor, name, None)
        gc.collect()
    matrix = np.concatenate(blocks, axis=0)
    if matrix.shape != (len(rows), int(spec["dimension"])):
        raise ValueError("chunked diagnostic matrix shape changed")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _validate_cache(
    output_root: Path,
    dataset_id: str,
    backbone: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matrix_path, manifest_path = _paths(output_root, dataset_id, backbone)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    spec = SPEC_BY_ID[backbone]
    checks = (
        manifest.get("dataset_id") == dataset_id,
        manifest.get("backbone_id") == backbone,
        manifest.get("sample_ids_sha256") == prepare.sha256_payload([row["sample_id"] for row in rows]),
        manifest.get("spec_sha256") == prepare.sha256_payload(spec),
        manifest.get("matrix_sha256") == prepare.sha256_path(matrix_path),
        manifest.get("values_sha256") == _values_sha256(np.asarray(matrix)),
        tuple(matrix.shape) == (len(rows), int(spec["dimension"])),
        matrix.dtype == np.dtype("float32"),
        bool(np.isfinite(np.asarray(matrix)).all()),
    )
    if not all(checks):
        raise ValueError(f"diagnostic embedding cache failed: {dataset_id}/{backbone}")
    return manifest


def run_extraction(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    cohort = prepare.load_cohort(input_root.resolve())
    model_records = []
    for spec in SPECS:
        checkpoint = age_extract._checkpoint_path(spec)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"frozen model checkpoint is missing: {checkpoint}")
        observed = prepare.sha256_path(checkpoint)
        if observed != spec["checkpoint_sha256"]:
            raise ValueError(f"frozen checkpoint SHA-256 mismatch for {spec['backbone_id']}")
        model_records.append({"backbone_id": spec["backbone_id"], "checkpoint_sha256": observed})
    model_inputs = {"backbone_count": len(model_records), "records": model_records}
    output_root = output_root.resolve()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        if not resume:
            raise FileExistsError("diagnostic terminal embedding manifest exists")
        manifest = json.loads(terminal.read_text(encoding="utf-8"))
        for dataset_id, dataset in cohort["datasets"].items():
            for backbone in BACKBONES:
                _validate_cache(output_root, dataset_id, backbone, dataset["rows"])
        return manifest
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    total = len(cohort["datasets"]) * len(BACKBONES)
    completed = 0
    for dataset_id, dataset in cohort["datasets"].items():
        rows = dataset["rows"]
        for backbone in BACKBONES:
            completed += 1
            matrix_path, manifest_path = _paths(output_root, dataset_id, backbone)
            if matrix_path.exists() or manifest_path.exists():
                if not (resume and matrix_path.exists() and manifest_path.exists()):
                    raise FileExistsError(f"partial diagnostic cache exists: {dataset_id}/{backbone}")
                records.append(_validate_cache(output_root, dataset_id, backbone, rows))
                print(f"validated {completed}/{total}: {dataset_id}/{backbone}", flush=True)
                continue
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            spec = SPEC_BY_ID[backbone]
            matrix = _extract_chunked(spec, rows)
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
                "sample_ids_sha256": prepare.sha256_payload([row["sample_id"] for row in rows]),
                "spec": dict(spec),
                "spec_sha256": prepare.sha256_payload(spec),
                "matrix_sha256": prepare.sha256_path(matrix_path),
                "values_sha256": _values_sha256(matrix),
            }
            prepare._atomic_json(manifest_path, record)
            records.append(record)
            print(f"extracted {completed}/{total}: {dataset_id}/{backbone}", flush=True)
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_ranking_reliability_recoverability_embedding_extraction",
        "protocol_sha256": prepare.sha256_path(prepare.PROTOCOL_PATH),
        "input_manifest_sha256": prepare.sha256_path(input_root / "manifest.json"),
        "cohort_sha256": prepare.sha256_path(input_root / "cohort.json"),
        "dataset_ids": list(cohort["datasets"]),
        "backbone_ids": list(BACKBONES),
        "records": records,
        "model_inputs": model_inputs,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "Pillow": importlib.metadata.version("Pillow"),
            "torch": importlib.metadata.version("torch"),
            "timm": importlib.metadata.version("timm"),
            "transformers": importlib.metadata.version("transformers"),
            "vertebrae": importlib.metadata.version("vertebrae"),
            "device": age_extract.DEVICE,
            "batch_size": age_extract.BATCH_SIZE,
            "extraction_chunk_rows": EXTRACTION_CHUNK_ROWS,
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "predictive_outcomes_inspected": False,
        "overlapindex_fit_count": 0,
        "reserved_backbone_count": 0,
    }
    prepare._atomic_json(terminal, manifest)
    return manifest


def load_bank(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    cohort = prepare.load_cohort(input_root.resolve())
    output_root = output_root.resolve()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed_continuous_ranking_reliability_recoverability_embedding_extraction"
        or manifest.get("protocol_sha256") != prepare.sha256_path(prepare.PROTOCOL_PATH)
        or tuple(manifest.get("dataset_ids", ())) != tuple(cohort["datasets"])
        or tuple(manifest.get("backbone_ids", ())) != BACKBONES
        or int(manifest.get("overlapindex_fit_count", -1)) != 0
        or int(manifest.get("reserved_backbone_count", -1)) != 0
    ):
        raise ValueError("diagnostic embedding terminal identity changed")
    matrices = {}
    for dataset_id, dataset in cohort["datasets"].items():
        matrices[dataset_id] = {}
        for backbone in BACKBONES:
            _validate_cache(output_root, dataset_id, backbone, dataset["rows"])
            path, _ = _paths(output_root, dataset_id, backbone)
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
