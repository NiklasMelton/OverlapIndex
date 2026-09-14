"""Extract the six frozen layer/pooling views for the LSP holdout."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from experiments.continuous_early_mean_lsp_holdout_v1 import prepare
from experiments.continuous_early_mean_replication_v1 import extract as matrix_io
from experiments.continuous_intermediate_pooling_factorial_v2 import extract as factorial
from experiments.continuous_ranking_benchmark_v2 import run as base
from experiments.continuous_real_age_panel import extract as model_source


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_early_mean_lsp_holdout_v1" / "embeddings"
SPECS = {str(row["backbone_id"]): row for row in model_source.BACKBONE_SPECS}


def load_protocol() -> dict[str, Any]:
    return prepare.load_protocol()


def _source_hashes() -> dict[str, str]:
    paths = {
        "input_cohort": prepare.DEFAULT_OUTPUT_ROOT / "cohort.json",
        "input_manifest": prepare.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "preparer": Path(prepare.__file__).resolve(),
        "factorial_extractor": Path(factorial.__file__).resolve(),
        "model_spec_source": Path(model_source.__file__).resolve(),
    }
    return {name: base.sha256_path(path) for name, path in paths.items()}


def _images(rows: Sequence[Mapping[str, Any]]) -> list[Image.Image]:
    output = []
    for row in rows:
        with Image.open(ROOT / str(row["image_path"])) as image:
            output.append(image.convert("RGB").copy())
    return output


def _extract_huggingface(
    config: Mapping[str, Any], spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], batch_size: int,
) -> dict[str, np.ndarray]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        str(spec["model_id"]), revision=str(spec["revision"]), local_files_only=True, use_fast=False,
    )
    kwargs = {"add_pooling_layer": False} if spec["backbone_id"] == "deit-tiny" else {}
    model = AutoModel.from_pretrained(
        str(spec["model_id"]), revision=str(spec["revision"]), local_files_only=True, **kwargs,
    ).to("cpu").eval()
    output = {
        f"{depth['depth_id']}_{pooling}": []
        for depth in config["relative_depths"] for pooling in ("mean", "mean_std")
    }
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                encoded = processor(images=_images(rows[start : start + batch_size]), return_tensors="pt")
                result = model(**encoded, output_hidden_states=True)
                for depth in config["relative_depths"]:
                    tokens = result.hidden_states[int(depth["hidden_layer"])][
                        :, int(config["special_tokens"]) :, :
                    ]
                    for pooling, matrix in factorial._pool_tokens(tokens, torch).items():
                        output[f"{depth['depth_id']}_{pooling}"].append(np.ascontiguousarray(matrix))
    finally:
        del model, processor
        gc.collect()
    return {view: np.ascontiguousarray(np.vstack(chunks)) for view, chunks in output.items()}


def _extract_timm(
    config: Mapping[str, Any], spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], batch_size: int,
) -> dict[str, np.ndarray]:
    import timm
    import torch

    model = timm.create_model(str(spec["model_id"]), pretrained=True, num_classes=0).to("cpu").eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    indices = [int(depth["feature_index"]) for depth in config["relative_depths"]]
    output = {
        f"{depth['depth_id']}_{pooling}": []
        for depth in config["relative_depths"] for pooling in ("mean", "mean_std")
    }
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch = torch.stack([transform(image) for image in _images(rows[start : start + batch_size])])
                _final, stages = model.forward_intermediates(
                    batch, indices=indices, norm=False, output_fmt="NCHW",
                )
                if len(stages) != len(indices):
                    raise ValueError("LSP holdout extraction returned the wrong timm stage count")
                for depth, stage in zip(config["relative_depths"], stages):
                    for pooling, matrix in factorial._pool_spatial(stage, torch).items():
                        output[f"{depth['depth_id']}_{pooling}"].append(np.ascontiguousarray(matrix))
    finally:
        del model
        gc.collect()
    return {view: np.ascontiguousarray(np.vstack(chunks)) for view, chunks in output.items()}


def _validate_record(output_root: Path, record: Mapping[str, Any]) -> None:
    for output in record["outputs"]:
        path = output_root / str(record["backbone_id"]) / f"{output['view_id']}.npy"
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            base.sha256_path(path) != output["matrix_sha256"]
            or tuple(matrix.shape) != (int(record["rows"]), int(output["dimension"]))
            or matrix.dtype != np.dtype("float32")
            or not np.isfinite(np.asarray(matrix)).all()
        ):
            raise ValueError(f"LSP holdout matrix identity changed: {record['backbone_id']}/{output['view_id']}")


def run_extraction(output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False) -> dict[str, Any]:
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    output_root = output_root.resolve()
    protocol = load_protocol()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("LSP holdout extraction requires a clean committed source tree")
    cohort = prepare.load_cohort()
    source_hashes = _source_hashes()
    terminal_path = output_root / "manifest.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("LSP holdout embedding manifest exists; use --resume or --validate-only")
        return validate_bank(output_root)
    rows = cohort["rows"]
    for config in protocol["backbones"]:
        backbone = str(config["backbone_id"])
        spec = SPECS[backbone]
        checkpoint = model_source._checkpoint_path(spec)
        if not checkpoint.is_file() or base.sha256_path(checkpoint) != spec["checkpoint_sha256"]:
            raise ValueError(f"LSP holdout frozen checkpoint identity changed: {backbone}")
    output_root.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(int(protocol["extraction"]["torch_intraop_threads"]))
    torch.set_num_interop_threads(int(protocol["extraction"]["torch_interop_threads"]))
    records = []
    for config in protocol["backbones"]:
        backbone = str(config["backbone_id"])
        record_path = output_root / f"{backbone}.json"
        if record_path.exists():
            if not resume:
                raise FileExistsError(f"partial LSP holdout cache exists: {backbone}")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            _validate_record(output_root, record)
            records.append(record)
            continue
        root = output_root / backbone
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"uncommitted LSP holdout matrices exist: {backbone}")
        spec = SPECS[backbone]
        matrices = (
            _extract_huggingface(config, spec, rows, int(protocol["extraction"]["batch_size"]))
            if config["provider"] == "huggingface"
            else _extract_timm(config, spec, rows, int(protocol["extraction"]["batch_size"]))
        )
        dimensions = factorial._expected_dimensions(protocol, config)
        if set(matrices) != set(protocol["views"]) or dimensions.keys() != matrices.keys():
            raise ValueError(f"LSP holdout output roster changed: {backbone}")
        root.mkdir(parents=True, exist_ok=True)
        outputs = []
        for view, matrix in matrices.items():
            if (
                matrix.shape != (len(rows), dimensions[view])
                or matrix.dtype != np.dtype("float32")
                or not np.isfinite(matrix).all()
            ):
                raise ValueError(f"invalid LSP holdout matrix: {backbone}/{view}")
            path = root / f"{view}.npy"
            matrix_io._write_matrix(path, matrix)
            outputs.append(
                {
                    "view_id": view,
                    "dimension": int(matrix.shape[1]),
                    "matrix_sha256": base.sha256_path(path),
                    "values_sha256": matrix_io._values_sha(matrix),
                }
            )
        record = {
            "backbone_id": backbone,
            "rows": len(rows),
            "sample_ids_sha256": base.sha256_payload([str(row["sample_id"]) for row in rows]),
            "model_spec_sha256": base.sha256_payload(spec),
            "outputs": outputs,
        }
        base._atomic_json(record_path, record)
        records.append(record)
        print(f"LSP holdout embedding {len(records)}/{len(protocol['backbones'])}: {backbone}", flush=True)
    if (
        len(records) != len(protocol["backbones"])
        or sum(len(row["outputs"]) for row in records) != int(protocol["expected_grid"]["embedding_matrix_count"])
    ):
        raise ValueError("LSP holdout embedding grid changed")
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_early_mean_lsp_holdout_v1_extraction",
        "protocol_sha256": base.sha256_path(PROTOCOL_PATH),
        "source_hashes": source_hashes,
        "records": records,
        "counts": {
            "rows": len(rows),
            "backbones": len(records),
            "matrices": sum(len(row["outputs"]) for row in records),
            "image_passes": len(rows) * len(records),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "Pillow": importlib.metadata.version("Pillow"),
            "torch": importlib.metadata.version("torch"),
            "timm": importlib.metadata.version("timm"),
            "transformers": importlib.metadata.version("transformers"),
            "device": "cpu",
            "torch_intraop_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
        },
        "git": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": False,
        },
        "outcomes_inspected": False,
        "overlapindex_fit_count": 0,
    }
    base._atomic_json(terminal_path, manifest)
    return validate_bank(output_root)


def validate_bank(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    protocol = load_protocol()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed_continuous_early_mean_lsp_holdout_v1_extraction"
        or manifest.get("protocol_sha256") != base.sha256_path(PROTOCOL_PATH)
        or manifest.get("source_hashes") != _source_hashes()
        or int(manifest.get("counts", {}).get("rows", -1)) != int(protocol["expected_grid"]["cohort_rows"])
        or int(manifest.get("counts", {}).get("matrices", -1)) != int(protocol["expected_grid"]["embedding_matrix_count"])
        or int(manifest.get("counts", {}).get("image_passes", -1)) != 4450
        or int(manifest.get("overlapindex_fit_count", -1)) != 0
    ):
        raise ValueError("LSP holdout embedding manifest changed")
    for record in manifest["records"]:
        _validate_record(output_root, record)
    return manifest


def load_bank(output_root: Path = DEFAULT_OUTPUT_ROOT) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    manifest = validate_bank(output_root)
    matrices = {
        str(record["backbone_id"]): {
            str(output["view_id"]): np.load(
                output_root / str(record["backbone_id"]) / f"{output['view_id']}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for output in record["outputs"]
        }
        for record in manifest["records"]
    }
    return manifest, matrices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = validate_bank(args.output_root) if args.validate_only else run_extraction(args.output_root, resume=args.resume)
    print(prepare.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
