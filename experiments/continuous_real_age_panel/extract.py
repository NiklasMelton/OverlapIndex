"""Extract the frozen 16-backbone UTKFace embedding bank."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from experiments.continuous_real_age_panel import prepare


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_age_panel" / "embeddings"
BATCH_SIZE = 16
DEVICE = "cpu"


def _spec(
    backbone_id: str,
    *,
    provider: str,
    model_id: str,
    revision: str,
    repository: str,
    checkpoint: str,
    checkpoint_sha256: str,
    output: str,
    dimension: int,
) -> dict[str, Any]:
    return {
        "backbone_id": backbone_id,
        "provider": provider,
        "model_id": model_id,
        "revision": revision,
        "repository": repository,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "output": output,
        "dimension": dimension,
        "image_mode": "rgb",
        "pretrained": True,
        "model_kwargs": {"num_classes": 0} if provider == "timm" else {},
    }


BACKBONE_SPECS = (
    _spec("dinov2-small", provider="hf_cls", model_id="facebook/dinov2-small", revision="ed25f3a31f01632728cabb09d1542f84ab7b0056", repository="models--facebook--dinov2-small", checkpoint="model.safetensors", checkpoint_sha256="ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1", output="final_cls", dimension=384),
    _spec("deit-tiny", provider="hf_cls", model_id="facebook/deit-tiny-patch16-224", revision="b3428f18dcc7b543470d07f14b4a4157815d1880", repository="models--facebook--deit-tiny-patch16-224", checkpoint="pytorch_model.bin", checkpoint_sha256="e1a51b0c81ff812e079d2189352747947879fee56429e4480064a6695cb34b2c", output="final_cls", dimension=192),
    _spec("convnext-tiny", provider="timm", model_id="convnext_tiny", revision="aa096f03029c7f0ec052013f64c819b34f8ad790", repository="models--timm--convnext_tiny.in12k_ft_in1k", checkpoint="model.safetensors", checkpoint_sha256="a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95", output="final", dimension=768),
    _spec("mobilenetv3-large", provider="timm", model_id="mobilenetv3_large_100", revision="96f46a1c52932f27492dff66c72378eb99b443a7", repository="models--timm--mobilenetv3_large_100.ra_in1k", checkpoint="model.safetensors", checkpoint_sha256="f425af34cc1cead2b5d6211f789a1f30b94835dc32f9c0fcc5a916e4fd2dde85", output="final", dimension=1280),
    _spec("openclip-vit-b-32", provider="openclip", model_id="ViT-B-32", revision="1a25a446712ba5ee05982a381eed697ef9b435cf", repository="models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K", checkpoint="open_clip_model.safetensors", checkpoint_sha256="ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6", output="final_image", dimension=512),
    _spec("resnet50", provider="timm", model_id="resnet50", revision="767268603ca0cb0bfe326fa87277f19c419566ef", repository="models--timm--resnet50.a1_in1k", checkpoint="model.safetensors", checkpoint_sha256="773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb", output="final", dimension=2048),
    _spec("efficientnet-b0", provider="timm", model_id="efficientnet_b0", revision="1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a", repository="models--timm--efficientnet_b0.ra_in1k", checkpoint="model.safetensors", checkpoint_sha256="d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd", output="final", dimension=1280),
    _spec("swin-tiny", provider="timm", model_id="swin_tiny_patch4_window7_224", revision="4cc3a7275b50b53a7bec45f32c236ebe64227cff", repository="models--timm--swin_tiny_patch4_window7_224.ms_in1k", checkpoint="model.safetensors", checkpoint_sha256="fb01861f793143135fa0d6cd97b1631e4b33eaa3ee162bbea9e62de1c76ebac1", output="final", dimension=768),
    _spec("vit-small-16", provider="timm", model_id="vit_small_patch16_224", revision="7e2c55630205e1266030f18370f4c6ed1a514b52", repository="models--timm--vit_small_patch16_224.augreg_in21k_ft_in1k", checkpoint="model.safetensors", checkpoint_sha256="79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa", output="final", dimension=384),
    _spec("densenet121", provider="timm", model_id="densenet121", revision="92007b6200e0b4a4fe68cb4e3947022a928aaaae", repository="models--timm--densenet121.ra_in1k", checkpoint="model.safetensors", checkpoint_sha256="eae900d8ae5b3716efa7f443fcc692aeb5a7f01f72581835c03c8c495f9eda10", output="final", dimension=1024),
    _spec("dino-vit-small-16", provider="hf_cls", model_id="facebook/dino-vits16", revision="abe3b354cb6a9b6f146096b14a4a9d7eecbcb4bd", repository="models--facebook--dino-vits16", checkpoint="pytorch_model.bin", checkpoint_sha256="a5bb4ddaf1b3abbdbba9962d7432bbdc633527c1c3e533e0bbc4d9d1b7e08d99", output="final_cls", dimension=384),
    _spec("resnet18", provider="hf_pooler", model_id="microsoft/resnet-18", revision="65a5785d9156231087c481e0c7dd33a5ff6f7e3e", repository="models--microsoft--resnet-18", checkpoint="model.safetensors", checkpoint_sha256="1cf00ee468998c23d084361b02cffadabacb5074105564c596b8079f77ecb126", output="final_pooler", dimension=512),
    _spec("resnet34", provider="timm", model_id="resnet34.a1_in1k", revision="0e8757122c4c09b15fb9300c0fd09b3fe89e28bf", repository="models--timm--resnet34.a1_in1k", checkpoint="pytorch_model.bin", checkpoint_sha256="f6c1d7785b5d0e796a83f9643fe2cb952ac95740daa9fb2d5c8c6514fdadaeb3", output="final", dimension=512),
    _spec("efficientnet-b2", provider="timm", model_id="efficientnet_b2.ra_in1k", revision="3577c4a7d84723645311bb5a9e5086f1b62ec8e2", repository="models--timm--efficientnet_b2.ra_in1k", checkpoint="pytorch_model.bin", checkpoint_sha256="27f82b064eab49b87e272a0fbf5aa95ce69a3eb9f1bf7685b968d2d1ae5fa532", output="final", dimension=1408),
    _spec("mobilenetv3-small", provider="timm", model_id="mobilenetv3_small_100.lamb_in1k", revision="1824797e7887cbec1990e4adbd6675960a36c589", repository="models--timm--mobilenetv3_small_100.lamb_in1k", checkpoint="pytorch_model.bin", checkpoint_sha256="c66054da8dbd3453fecb5c3c842faef8e4e1fcdc71d68208cb577388ae191c5f", output="final", dimension=1024),
    _spec("convnext-small", provider="timm", model_id="convnext_small.in12k_ft_in1k", revision="45db0bc46e10f927840106d2763c36e6b8a2fc53", repository="models--timm--convnext_small.in12k_ft_in1k", checkpoint="pytorch_model.bin", checkpoint_sha256="d859df441c0e01e266678ebeacbcd95c42b3804ce11d421884b9000c5eaf82e3", output="final", dimension=768),
)
BACKBONE_IDS = tuple(spec["backbone_id"] for spec in BACKBONE_SPECS)


def _checkpoint_path(spec: Mapping[str, Any]) -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / str(spec["repository"])
        / "snapshots"
        / str(spec["revision"])
        / str(spec["checkpoint"])
    )


def validate_model_inputs() -> dict[str, Any]:
    records = []
    for spec in BACKBONE_SPECS:
        checkpoint = _checkpoint_path(spec)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"frozen model checkpoint is missing: {checkpoint}")
        observed = prepare.sha256_path(checkpoint)
        if observed != spec["checkpoint_sha256"]:
            raise ValueError(f"frozen checkpoint SHA-256 mismatch for {spec['backbone_id']}")
        records.append({"backbone_id": spec["backbone_id"], "checkpoint_sha256": observed})
    return {"backbone_count": len(records), "records": records}


def _build_extractor(spec: Mapping[str, Any]) -> Any:
    provider = str(spec["provider"])
    if provider.startswith("hf_"):
        from vertebrae.extractors.huggingface_vision import HFVisionExtractor

        pooling = "pooler" if provider == "hf_pooler" else "cls"
        return HFVisionExtractor(
            name=str(spec["backbone_id"]),
            model_id=str(spec["model_id"]),
            revision=str(spec["revision"]),
            outputs=[{"name": spec["output"], "hidden_layer": None if pooling == "pooler" else -1, "pooling": pooling}],
            batch_size=BATCH_SIZE,
            image_mode="rgb",
            device=DEVICE,
            processor_kwargs={"use_fast": False},
            model_kwargs={"add_pooling_layer": False} if spec["backbone_id"] == "deit-tiny" else {},
        )
    if provider == "openclip":
        from vertebrae.extractors.openclip import OpenCLIPExtractor

        return OpenCLIPExtractor(
            name=str(spec["backbone_id"]), model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k", input_modalities={"image": "image"},
            outputs=[{"name": spec["output"], "source": "image"}], batch_size=BATCH_SIZE,
            image_mode="rgb", device=DEVICE,
        )
    from vertebrae.extractors.timm import TimmVisionExtractor

    return TimmVisionExtractor(
        name=str(spec["backbone_id"]), model_name=str(spec["model_id"]), pretrained=True,
        outputs=[{"name": spec["output"]}], model_kwargs={"num_classes": 0},
        batch_size=BATCH_SIZE, image_mode="rgb", device=DEVICE,
    )


def _extract(spec: Mapping[str, Any], images: Sequence[Image.Image]) -> np.ndarray:
    extractor = _build_extractor(spec)
    try:
        values = {"image": list(images)} if spec["provider"] == "openclip" else list(images)
        outputs = extractor.transform_many(values)
        selected = [row for row in outputs if row.name == spec["output"]]
        if len(selected) != 1:
            raise ValueError("extractor did not emit exactly one frozen output")
        matrix = np.asarray(selected[0].embeddings, dtype=np.float32)
    finally:
        for name in ("_model", "_processor", "_torch", "_preprocess", "_tokenizer"):
            if hasattr(extractor, name):
                setattr(extractor, name, None)
        gc.collect()
    if matrix.shape != (len(images), int(spec["dimension"])) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid extracted matrix for {spec['backbone_id']}: {matrix.shape}")
    return np.ascontiguousarray(matrix)


def _values_sha(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes(order="C")).hexdigest()


def _load_images(input_root: Path, cohort: Mapping[str, Any]) -> list[Image.Image]:
    images = []
    for row in cohort["rows"]:
        with Image.open(input_root / "images" / row["sample_id"]) as source:
            images.append(source.convert("RGB").copy())
    return images


def _validate_cache(output_root: Path, spec: Mapping[str, Any], cohort: Mapping[str, Any]) -> dict[str, Any]:
    matrix_path = output_root / f"{spec['backbone_id']}.npy"
    manifest_path = output_root / f"{spec['backbone_id']}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    expected_ids = prepare.sha256_payload([row["sample_id"] for row in cohort["rows"]])
    expected_labels = prepare.sha256_payload([row["age"] for row in cohort["rows"]])
    checks = (
        manifest.get("backbone_id") == spec["backbone_id"],
        manifest.get("spec_sha256") == prepare.sha256_payload(spec),
        manifest.get("sample_ids_sha256") == expected_ids,
        manifest.get("labels_sha256") == expected_labels,
        manifest.get("matrix_sha256") == prepare.sha256_path(matrix_path),
        manifest.get("values_sha256") == _values_sha(np.asarray(matrix)),
        tuple(matrix.shape) == (len(cohort["rows"]), int(spec["dimension"])),
        matrix.dtype == np.dtype("float32"),
        bool(np.isfinite(np.asarray(matrix)).all()),
    )
    if not all(checks):
        raise ValueError(f"embedding cache validation failed for {spec['backbone_id']}")
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
            raise FileExistsError("terminal embedding manifest exists")
        return json.loads(terminal.read_text(encoding="utf-8"))
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
        matrix = _extract(spec, images)
        temporary = matrix_path.with_name(f".{matrix_path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
        temporary.replace(matrix_path)
        record = {
            "backbone_id": spec["backbone_id"],
            "rows": len(matrix),
            "dimension": int(matrix.shape[1]),
            "dtype": "float32",
            "sample_ids_sha256": prepare.sha256_payload([row["sample_id"] for row in cohort["rows"]]),
            "labels_sha256": prepare.sha256_payload([row["age"] for row in cohort["rows"]]),
            "spec": dict(spec),
            "spec_sha256": prepare.sha256_payload(spec),
            "matrix_sha256": prepare.sha256_path(matrix_path),
            "values_sha256": _values_sha(matrix),
        }
        manifest_path.write_text(prepare.canonical_json(record) + "\n", encoding="utf-8")
        records.append(record)
    environment = {
        "python": platform.python_version(), "numpy": np.__version__,
        "Pillow": importlib.metadata.version("Pillow"), "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"), "timm": importlib.metadata.version("timm"),
        "transformers": importlib.metadata.version("transformers"), "open_clip_torch": importlib.metadata.version("open_clip_torch"),
        "vertebrae": importlib.metadata.version("vertebrae"), "device": DEVICE, "batch_size": BATCH_SIZE,
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
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
    print(prepare.canonical_json(run_extraction(args.input_root, args.output_root, resume=args.resume)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
