"""Extract the frozen 16-backbone Oxford-IIIT Pet embedding bank."""

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

from experiments.continuous_real_age_panel import extract as parent
from experiments.continuous_real_pet_area_panel import prepare


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_pet_area_panel" / "embeddings"
BACKBONE_SPECS = parent.BACKBONE_SPECS
BACKBONE_IDS = parent.BACKBONE_IDS
BATCH_SIZE = parent.BATCH_SIZE
DEVICE = parent.DEVICE
validate_model_inputs = parent.validate_model_inputs


def _load_images(input_root: Path, cohort: Mapping[str, Any]) -> list[Image.Image]:
    images = []
    for row in cohort["rows"]:
        path = input_root / "images" / f"{row['sample_id']}.jpg"
        with Image.open(path) as source:
            images.append(source.convert("RGB").copy())
    return images


def _label_sha(cohort: Mapping[str, Any]) -> str:
    return prepare.sha256_payload([row["foreground_fraction"] for row in cohort["rows"]])


def _validate_cache(
    output_root: Path, spec: Mapping[str, Any], cohort: Mapping[str, Any]
) -> dict[str, Any]:
    matrix_path = output_root / f"{spec['backbone_id']}.npy"
    manifest_path = output_root / f"{spec['backbone_id']}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    checks = (
        manifest.get("backbone_id") == spec["backbone_id"],
        manifest.get("spec_sha256") == prepare.sha256_payload(spec),
        manifest.get("sample_ids_sha256")
        == prepare.sha256_payload([row["sample_id"] for row in cohort["rows"]]),
        manifest.get("labels_sha256") == _label_sha(cohort),
        manifest.get("matrix_sha256") == prepare.sha256_path(matrix_path),
        manifest.get("values_sha256") == parent._values_sha(np.asarray(matrix)),
        tuple(matrix.shape) == (len(cohort["rows"]), int(spec["dimension"])),
        matrix.dtype == np.dtype("float32"),
        bool(np.isfinite(np.asarray(matrix)).all()),
    )
    if not all(checks):
        raise ValueError(f"pet-area embedding cache validation failed for {spec['backbone_id']}")
    return manifest


def run_extraction(
    input_root: Path = prepare.DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    input_root, output_root = input_root.resolve(), output_root.resolve()
    cohort = prepare.load_cohort(input_root)
    validate_model_inputs()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        if not resume:
            raise FileExistsError("terminal pet-area embedding manifest exists")
        manifest = json.loads(terminal.read_text(encoding="utf-8"))
        for spec in BACKBONE_SPECS:
            _validate_cache(output_root, spec, cohort)
        return manifest
    output_root.mkdir(parents=True, exist_ok=True)
    images = _load_images(input_root, cohort)
    records = []
    for spec in BACKBONE_SPECS:
        matrix_path = output_root / f"{spec['backbone_id']}.npy"
        manifest_path = output_root / f"{spec['backbone_id']}.json"
        if matrix_path.exists() or manifest_path.exists():
            if not (resume and matrix_path.exists() and manifest_path.exists()):
                raise FileExistsError(f"partial embedding cache exists for {spec['backbone_id']}")
            records.append(_validate_cache(output_root, spec, cohort))
            continue
        matrix = parent._extract(spec, images)
        temporary = matrix_path.with_name(f".{matrix_path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
        temporary.replace(matrix_path)
        record = {
            "backbone_id": spec["backbone_id"],
            "rows": len(matrix),
            "dimension": int(matrix.shape[1]),
            "dtype": "float32",
            "sample_ids_sha256": prepare.sha256_payload(
                [row["sample_id"] for row in cohort["rows"]]
            ),
            "labels_sha256": _label_sha(cohort),
            "spec": dict(spec),
            "spec_sha256": prepare.sha256_payload(spec),
            "matrix_sha256": prepare.sha256_path(matrix_path),
            "values_sha256": parent._values_sha(matrix),
        }
        manifest_path.write_text(prepare.canonical_json(record) + "\n", encoding="utf-8")
        records.append(record)
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "Pillow": importlib.metadata.version("Pillow"),
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
        "timm": importlib.metadata.version("timm"),
        "transformers": importlib.metadata.version("transformers"),
        "open_clip_torch": importlib.metadata.version("open_clip_torch"),
        "vertebrae": importlib.metadata.version("vertebrae"),
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    manifest = {
        "schema_version": 1,
        "status": "completed_embedding_extraction",
        "input_manifest_sha256": prepare.sha256_path(input_root / "manifest.json"),
        "cohort_sha256": prepare.sha256_path(input_root / "cohort.json"),
        "backbone_ids": list(BACKBONE_IDS),
        "backbone_count": len(BACKBONE_IDS),
        "records": records,
        "environment": environment,
        "outcomes_inspected": False,
    }
    terminal.write_text(prepare.canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=prepare.DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        prepare.canonical_json(
            run_extraction(args.input_root, args.output_root, resume=args.resume)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
