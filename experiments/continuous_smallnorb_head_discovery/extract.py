"""Extract the five frozen scout-backbone SmallNORB embedding matrices."""

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

from experiments.continuous_real_age_panel import extract as age_extract
from experiments.continuous_real_pet_area_panel import extract as parent_extract
from experiments.continuous_smallnorb_head_discovery import prepare


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = prepare.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_smallnorb_head_discovery" / "embeddings"
BACKBONES = (
    "dinov2-small",
    "deit-tiny",
    "mobilenetv3-large",
    "resnet50",
    "convnext-tiny",
)
SPEC_BY_ID = {
    str(spec["backbone_id"]): dict(spec)
    for spec in parent_extract.BACKBONE_SPECS
    if str(spec["backbone_id"]) in BACKBONES
}


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


def validate_model_inputs() -> list[dict[str, Any]]:
    if set(SPEC_BY_ID) != set(BACKBONES):
        raise ValueError("SmallNORB scout model specification grid is incomplete")
    records = []
    for backbone in BACKBONES:
        spec = SPEC_BY_ID[backbone]
        checkpoint = age_extract._checkpoint_path(spec)
        if not checkpoint.is_file() or prepare.sha256_path(checkpoint) != spec["checkpoint_sha256"]:
            raise ValueError(f"SmallNORB frozen model checkpoint changed: {backbone}")
        records.append(
            {
                "backbone_id": backbone,
                "spec_sha256": prepare.sha256_payload(spec),
                "checkpoint_sha256": spec["checkpoint_sha256"],
            }
        )
    return records


def _images(input_root: Path, cohort: Mapping[str, Any]) -> list[Image.Image]:
    output = []
    for row in cohort["rows"]:
        with Image.open(input_root / row["image_path"]) as source:
            output.append(source.convert("RGB").copy())
    return output


def _validate_cache(
    output_root: Path,
    backbone: str,
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    matrix_path = output_root / f"{backbone}.npy"
    manifest_path = output_root / f"{backbone}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    spec = SPEC_BY_ID[backbone]
    checks = (
        manifest.get("backbone_id") == backbone,
        manifest.get("spec_sha256") == prepare.sha256_payload(spec),
        manifest.get("cohort_sha256") == prepare.sha256_payload(cohort),
        manifest.get("sample_ids_sha256")
        == prepare.sha256_payload([row["sample_id"] for row in cohort["rows"]]),
        manifest.get("matrix_sha256") == prepare.sha256_path(matrix_path),
        manifest.get("values_sha256") == _values_sha256(np.asarray(matrix)),
        tuple(matrix.shape) == (len(cohort["rows"]), int(spec["dimension"])),
        matrix.dtype == np.dtype("float32"),
        bool(np.isfinite(np.asarray(matrix)).all()),
    )
    if not all(checks):
        raise ValueError(f"SmallNORB embedding cache validation failed: {backbone}")
    return manifest


def run_extraction(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    input_root, output_root = input_root.resolve(), output_root.resolve()
    cohort = prepare.load_cohort(input_root)
    model_inputs = validate_model_inputs()
    terminal = output_root / "manifest.json"
    if terminal.exists():
        if not resume:
            raise FileExistsError("SmallNORB terminal embedding manifest already exists")
        manifest = json.loads(terminal.read_text(encoding="utf-8"))
        for backbone in BACKBONES:
            _validate_cache(output_root, backbone, cohort)
        return manifest
    output_root.mkdir(parents=True, exist_ok=True)
    images = _images(input_root, cohort)
    records = []
    for position, backbone in enumerate(BACKBONES, start=1):
        matrix_path = output_root / f"{backbone}.npy"
        manifest_path = output_root / f"{backbone}.json"
        if matrix_path.exists() or manifest_path.exists():
            if not (resume and matrix_path.exists() and manifest_path.exists()):
                raise FileExistsError(f"partial SmallNORB embedding cache exists: {backbone}")
            records.append(_validate_cache(output_root, backbone, cohort))
            print(f"validated {position}/{len(BACKBONES)}: {backbone}", flush=True)
            continue
        spec = SPEC_BY_ID[backbone]
        matrix = age_extract._extract(spec, images)
        temporary = matrix_path.with_name(f".{matrix_path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
        temporary.replace(matrix_path)
        record = {
            "schema_version": 1,
            "backbone_id": backbone,
            "rows": int(len(matrix)),
            "dimension": int(matrix.shape[1]),
            "dtype": "float32",
            "cohort_sha256": prepare.sha256_payload(cohort),
            "sample_ids_sha256": prepare.sha256_payload(
                [row["sample_id"] for row in cohort["rows"]]
            ),
            "spec": spec,
            "spec_sha256": prepare.sha256_payload(spec),
            "matrix_sha256": prepare.sha256_path(matrix_path),
            "values_sha256": _values_sha256(matrix),
        }
        prepare._atomic_json(manifest_path, record)
        records.append(record)
        print(f"extracted {position}/{len(BACKBONES)}: {backbone}", flush=True)
    manifest = {
        "schema_version": 1,
        "status": "completed_smallnorb_embedding_extraction",
        "protocol_sha256": prepare.sha256_path(prepare.PROTOCOL_PATH),
        "input_manifest_sha256": prepare.sha256_path(input_root / "manifest.json"),
        "cohort_sha256": prepare.sha256_path(input_root / "cohort.json"),
        "cohort_identity_sha256": prepare.sha256_payload(cohort),
        "backbone_ids": list(BACKBONES),
        "model_inputs": model_inputs,
        "records": records,
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
            "device": parent_extract.DEVICE,
            "batch_size": parent_extract.BATCH_SIZE,
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "outcomes_inspected": False,
    }
    prepare._atomic_json(terminal, manifest)
    return manifest


def load_bank(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cohort = prepare.load_cohort(input_root.resolve())
    terminal = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if terminal.get("status") != "completed_smallnorb_embedding_extraction":
        raise ValueError("SmallNORB embedding extraction is incomplete")
    if terminal.get("protocol_sha256") != prepare.sha256_path(prepare.PROTOCOL_PATH):
        raise ValueError("SmallNORB embedding protocol identity changed")
    if tuple(terminal.get("backbone_ids", ())) != BACKBONES:
        raise ValueError("SmallNORB embedding scout roster changed")
    matrices = {}
    for backbone in BACKBONES:
        _validate_cache(output_root, backbone, cohort)
        matrices[backbone] = np.load(
            output_root / f"{backbone}.npy", mmap_mode="r", allow_pickle=False
        )
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
