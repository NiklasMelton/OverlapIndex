"""Verified Food-101 backbone recipes and hash-bound float32 caches.

The only provider construction path in this module is the archived Food-101
factory.  The source file is hashed before it is imported, and model IDs are
closed to the ten protocol backbones.  Cache writers accept already selected
cohort rows only; selection and extraction are separate stages so a cache
cannot quietly introduce an evaluation row or a second sampling rule.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from . import datasets
from . import lineage
from . import recipes


FOOD_DRIVER_PATH = Path(
    "/Users/niklasmelton/code/vertabrae/examples/food101_nonlinear_backbone_bridge.py"
)
FOOD_DRIVER_SHA256 = "63f906f12070aaf697d88c193dd11dc2f430bc9fa6b0389756d556636ad85eae"
FOOD_BACKBONE_IDS = (
    "dinov2-small",
    "deit-tiny",
    "convnext-tiny",
    "mobilenetv3-large",
    "openclip-vit-b-32",
    "resnet50",
    "efficientnet-b0",
    "swin-tiny",
    "vit-small-16",
    "densenet121",
)


def _model_spec(
    backbone_id: str,
    *,
    provider: str,
    provider_version: str,
    model_id: str,
    snapshot_revision: str,
    output: str,
    embedding_dimension: int,
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    static_recipe = deepcopy(dict(recipe))
    static_recipe["output"] = output
    return {
        "backbone_id": backbone_id,
        "provider": provider,
        "provider_version": provider_version,
        "model_id": model_id,
        "snapshot_revision": snapshot_revision,
        # A planning descriptor is explicit about unavailable local bytes.
        # resolve_backbone_spec() is required before an extractor/cache run.
        "weights_sha256": None,
        "recipe": static_recipe,
        "recipe_sha256": lineage.sha256_payload(static_recipe),
        "embedding_dimension": int(embedding_dimension),
        "dtype": "float32",
        "normalization": "none_at_extraction_row_l2_in_selector_or_head",
        "device_policy": "archived_food_factory_device_argument",
    }


FOOD_BACKBONE_SPECS = {
    "dinov2-small": _model_spec(
        "dinov2-small",
        provider="huggingface",
        provider_version="transformers_4.57.6",
        model_id="facebook/dinov2-small",
        snapshot_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        output="final_cls",
        embedding_dimension=384,
        recipe={
            "extractor_type": "HFVisionExtractor",
            "outputs": [{"name": "final_cls", "hidden_layer": -1, "pooling": "cls"}],
            "image_mode": "rgb",
            "processor_kwargs": {"use_fast": False},
            "model_kwargs": {},
        },
    ),
    "deit-tiny": _model_spec(
        "deit-tiny",
        provider="huggingface",
        provider_version="transformers_4.57.6",
        model_id="facebook/deit-tiny-patch16-224",
        snapshot_revision="b3428f18dcc7b543470d07f14b4a4157815d1880",
        output="final_cls",
        embedding_dimension=192,
        recipe={
            "extractor_type": "HFVisionExtractor",
            "outputs": [{"name": "final_cls", "hidden_layer": -1, "pooling": "cls"}],
            "image_mode": "rgb",
            "processor_kwargs": {"use_fast": False},
            "model_kwargs": {"add_pooling_layer": False},
        },
    ),
    "convnext-tiny": _model_spec(
        "convnext-tiny",
        provider="timm",
        provider_version="timm_1.0.27",
        model_id="convnext_tiny",
        snapshot_revision="aa096f03029c7f0ec052013f64c819b34f8ad790",
        output="final",
        embedding_dimension=768,
        recipe={
            "extractor_type": "TimmVisionExtractor",
            "pretrained": True,
            "outputs": [{"name": "final"}],
            "model_kwargs": {"num_classes": 0},
            "image_mode": "rgb",
        },
    ),
    "mobilenetv3-large": _model_spec(
        "mobilenetv3-large",
        provider="timm",
        provider_version="timm_1.0.27",
        model_id="mobilenetv3_large_100",
        snapshot_revision="96f46a1c52932f27492dff66c72378eb99b443a7",
        output="final",
        embedding_dimension=1280,
        recipe={
            "extractor_type": "TimmVisionExtractor",
            "pretrained": True,
            "outputs": [{"name": "final"}],
            "model_kwargs": {"num_classes": 0},
            "image_mode": "rgb",
        },
    ),
    "openclip-vit-b-32": _model_spec(
        "openclip-vit-b-32",
        provider="open_clip",
        provider_version="open_clip_2.32.0",
        model_id="ViT-B-32",
        snapshot_revision="1a25a446712ba5ee05982a381eed697ef9b435cf",
        output="final_image",
        embedding_dimension=512,
        recipe={
            "extractor_type": "OpenCLIPExtractor",
            "pretrained": "laion2b_s34b_b79k",
            "input_modalities": {"image": "image"},
            "outputs": [{"name": "final_image", "source": "image"}],
            "image_mode": "rgb",
        },
    ),
    "resnet50": _model_spec(
        "resnet50",
        provider="timm",
        provider_version="timm_1.0.27",
        model_id="resnet50",
        snapshot_revision="767268603ca0cb0bfe326fa87277f19c419566ef",
        output="final",
        embedding_dimension=2048,
        recipe={
            "extractor_type": "TimmVisionExtractor",
            "pretrained": True,
            "outputs": [{"name": "final"}],
            "model_kwargs": {"num_classes": 0},
            "image_mode": "rgb",
        },
    ),
    "efficientnet-b0": _model_spec(
        "efficientnet-b0",
        provider="timm",
        provider_version="timm_1.0.27",
        model_id="efficientnet_b0",
        snapshot_revision="1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a",
        output="final",
        embedding_dimension=1280,
        recipe={
            "extractor_type": "TimmVisionExtractor",
            "pretrained": True,
            "outputs": [{"name": "final"}],
            "model_kwargs": {"num_classes": 0},
            "image_mode": "rgb",
        },
    ),
    "swin-tiny": _model_spec(
        "swin-tiny",
        provider="timm",
        provider_version="timm_1.0.27",
        model_id="swin_tiny_patch4_window7_224",
        snapshot_revision="4cc3a7275b50b53a7bec45f32c236ebe64227cff",
        output="final",
        embedding_dimension=768,
        recipe={
            "extractor_type": "TimmVisionExtractor",
            "pretrained": True,
            "outputs": [{"name": "final"}],
            "model_kwargs": {"num_classes": 0},
            "image_mode": "rgb",
        },
    ),
    "vit-small-16": _model_spec(
        "vit-small-16",
        provider="timm",
        provider_version="timm_1.0.27",
        # The archived Food driver passes this alias to timm; timm resolves it
        # to the hash-locked augreg.in21k_ft_in1k weight snapshot below.
        model_id="vit_small_patch16_224",
        snapshot_revision="7e2c55630205e1266030f18370f4c6ed1a514b52",
        output="final",
        embedding_dimension=384,
        recipe={
            "extractor_type": "TimmVisionExtractor",
            "pretrained": True,
            "outputs": [{"name": "final"}],
            "model_kwargs": {"num_classes": 0},
            "image_mode": "rgb",
        },
    ),
    "densenet121": _model_spec(
        "densenet121",
        provider="timm",
        provider_version="timm_1.0.27",
        model_id="densenet121",
        snapshot_revision="92007b6200e0b4a4fe68cb4e3947022a928aaaae",
        output="final",
        embedding_dimension=1024,
        recipe={
            "extractor_type": "TimmVisionExtractor",
            "pretrained": True,
            "outputs": [{"name": "final"}],
            "model_kwargs": {"num_classes": 0},
            "image_mode": "rgb",
        },
    ),
}


def _validate_backbone_id(backbone_id: Any) -> str:
    if not isinstance(backbone_id, str) or backbone_id not in FOOD_BACKBONE_IDS:
        raise ValueError(
            f"backbone_id must be one of {FOOD_BACKBONE_IDS!r}; got {backbone_id!r}"
        )
    return backbone_id


def backbone_spec(backbone_id: str) -> dict[str, Any]:
    """Return a detached planning spec for one frozen backbone."""

    _validate_backbone_id(backbone_id)
    return deepcopy(FOOD_BACKBONE_SPECS[backbone_id])


def backbone_specs() -> tuple[dict[str, Any], ...]:
    return tuple(backbone_spec(backbone_id) for backbone_id in FOOD_BACKBONE_IDS)


def _sha256_file(path: os.PathLike[str] | str) -> str:
    return lineage.sha256_path(path)


def _sha256_directory(path: os.PathLike[str] | str) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"snapshot path is not a directory: {root}")
    entries: list[dict[str, str]] = []
    for child in sorted(root.rglob("*")):
        if child.is_file():
            relative = child.relative_to(root).as_posix()
            entries.append({"path": relative, "sha256": _sha256_file(child)})
        elif child.is_symlink():
            raise ValueError(f"snapshot contains unsupported symlink: {child}")
    if not entries:
        raise ValueError(f"snapshot directory is empty: {root}")
    return lineage.sha256_payload(entries)


def _sha256_paths(paths: Sequence[os.PathLike[str] | str]) -> str:
    if not paths:
        raise ValueError("at least one preprocessor file is required")
    entries = []
    for path in paths:
        candidate = Path(path)
        digest = (
            _sha256_directory(candidate)
            if candidate.is_dir()
            else _sha256_file(candidate)
        )
        entries.append({"path": str(candidate), "sha256": digest})
    return lineage.sha256_payload(entries)


def resolved_local_backbone_specs(
    *,
    cache_root: os.PathLike[str] | str = Path.home() / ".cache" / "huggingface" / "hub",
) -> dict[str, dict[str, Any]]:
    """Resolve the exact already-cached Food weights without network access."""

    root = Path(cache_root)
    repositories = {
        "dinov2-small": ("models--facebook--dinov2-small", "model.safetensors", "huggingface"),
        "deit-tiny": ("models--facebook--deit-tiny-patch16-224", "pytorch_model.bin", "huggingface"),
        "convnext-tiny": ("models--timm--convnext_tiny.in12k_ft_in1k", "model.safetensors", "timm"),
        "mobilenetv3-large": ("models--timm--mobilenetv3_large_100.ra_in1k", "model.safetensors", "timm"),
        "openclip-vit-b-32": ("models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K", "open_clip_model.safetensors", "open_clip"),
        "resnet50": ("models--timm--resnet50.a1_in1k", "model.safetensors", "timm"),
        "efficientnet-b0": ("models--timm--efficientnet_b0.ra_in1k", "model.safetensors", "timm"),
        "swin-tiny": ("models--timm--swin_tiny_patch4_window7_224.ms_in1k", "model.safetensors", "timm"),
        "vit-small-16": ("models--timm--vit_small_patch16_224.augreg_in21k_ft_in1k", "model.safetensors", "timm"),
        "densenet121": ("models--timm--densenet121.ra_in1k", "model.safetensors", "timm"),
    }
    import open_clip
    import timm

    package_roots = {
        "timm": Path(timm.__file__).resolve().parent,
        "open_clip": Path(open_clip.__file__).resolve().parent,
    }
    output: dict[str, dict[str, Any]] = {}
    for backbone in FOOD_BACKBONE_IDS:
        repository, checkpoint_name, preprocessing = repositories[backbone]
        snapshot = (
            root
            / repository
            / "snapshots"
            / str(FOOD_BACKBONE_SPECS[backbone]["snapshot_revision"])
        )
        checkpoint = snapshot / checkpoint_name
        if not snapshot.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError(
                f"frozen cached snapshot/checkpoint is missing for {backbone}: {snapshot}"
            )
        if preprocessing == "huggingface":
            preprocessor_paths = tuple(
                path
                for path in (snapshot / "preprocessor_config.json", snapshot / "config.json")
                if path.is_file()
            )
            if not preprocessor_paths:
                raise FileNotFoundError(f"cached processor configuration is missing for {backbone}")
        else:
            preprocessor_paths = (package_roots[preprocessing],)
        output[backbone] = resolve_backbone_spec(
            backbone,
            checkpoint_path=checkpoint,
            preprocessor_paths=preprocessor_paths,
            snapshot_path=snapshot,
        )
    return output


def observed_extraction_environment(
    *, device: str = "cpu", batch_size: int = 16
) -> dict[str, Any]:
    """Record and require the exact frozen extraction runtime."""

    if device != "cpu" or type(batch_size) is not int or batch_size != 16:
        raise ValueError("extraction is frozen to device='cpu' and batch_size=16")
    import platform
    import PIL
    import open_clip
    import timm
    import torch
    import torchvision
    import transformers

    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "timm": str(timm.__version__),
        "transformers": str(transformers.__version__),
        "open_clip": str(open_clip.__version__),
        "Pillow": str(PIL.__version__),
        "vertebrae": importlib.metadata.version("vertebrae"),
        "device": device,
        "batch_size": batch_size,
        "network_policy": "offline_cached_weights_only",
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    if observed != recipes.EXTRACTION_ENVIRONMENT:
        raise RuntimeError(
            f"extraction environment mismatch: expected {recipes.EXTRACTION_ENVIRONMENT!r}, got {observed!r}"
        )
    return observed


def resolve_backbone_spec(
    backbone_id: str,
    *,
    checkpoint_path: os.PathLike[str] | str,
    preprocessor_paths: Sequence[os.PathLike[str] | str],
    snapshot_path: os.PathLike[str] | str,
    provider_version: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve required checkpoint/preprocessor/snapshot bytes into a ready spec."""

    resolved = backbone_spec(backbone_id)
    if provider_version is not None:
        if not isinstance(provider_version, str) or not provider_version:
            raise ValueError("provider_version must be a nonempty string")
        resolved["provider_version"] = provider_version
    checkpoint_hash = _sha256_file(checkpoint_path)
    preprocessor_hash = _sha256_paths(preprocessor_paths)
    snapshot_hash = (
        _sha256_directory(snapshot_path)
        if Path(snapshot_path).is_dir()
        else _sha256_file(snapshot_path)
    )
    artifacts = {
        "checkpoint_file": str(Path(checkpoint_path)),
        "checkpoint_file_sha256": checkpoint_hash,
        "preprocessor_files": [str(Path(path)) for path in preprocessor_paths],
        "preprocessor_sha256": preprocessor_hash,
        "snapshot": str(Path(snapshot_path)),
        "snapshot_sha256": snapshot_hash,
    }
    resolved["weights_sha256"] = checkpoint_hash
    resolved["recipe"]["artifacts"] = artifacts
    resolved["recipe_sha256"] = lineage.sha256_payload(resolved["recipe"])
    return resolved


def require_ready_backbone_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a resolved spec before model construction or cache writing."""

    required = {
        "backbone_id",
        "provider",
        "provider_version",
        "model_id",
        "snapshot_revision",
        "weights_sha256",
        "recipe",
        "recipe_sha256",
        "embedding_dimension",
        "dtype",
        "normalization",
        "device_policy",
    }
    if not isinstance(spec, Mapping) or set(spec) != required:
        raise ValueError("backbone spec has unknown or missing keys")
    backbone_id = _validate_backbone_id(spec["backbone_id"])
    if spec["dtype"] != "float32" or type(spec["embedding_dimension"]) is not int:
        raise ValueError("backbone spec dtype/dimension is invalid")
    if spec["embedding_dimension"] <= 0:
        raise ValueError("backbone embedding dimension must be positive")
    if not isinstance(spec["recipe"], Mapping):
        raise ValueError("backbone recipe must be a mapping")
    if lineage.sha256_payload(spec["recipe"]) != spec["recipe_sha256"]:
        raise ValueError("backbone recipe hash mismatch")
    lineage.require_hash(spec["weights_sha256"], field="weights_sha256")
    artifacts = spec["recipe"].get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("resolved backbone recipe lacks artifact hashes")
    for field in ("checkpoint_file_sha256", "preprocessor_sha256", "snapshot_sha256"):
        value = artifacts.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"resolved backbone recipe lacks valid {field}")
    return deepcopy(dict(spec))


def model_weight_lock_from_specs(
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the one closed model/recipe identity used by the registry."""

    if not isinstance(specs, Mapping) or set(specs) != set(FOOD_BACKBONE_IDS):
        raise ValueError("backbone specs must contain exactly the frozen ten-model panel")
    output: dict[str, dict[str, Any]] = {}
    for backbone in FOOD_BACKBONE_IDS:
        resolved = require_ready_backbone_spec(specs[backbone])
        artifacts = resolved["recipe"]["artifacts"]
        output[backbone] = {
            "backbone_id": backbone,
            "provider": resolved["provider"],
            "provider_version": resolved["provider_version"],
            "model_id": resolved["model_id"],
            "snapshot_revision": resolved["snapshot_revision"],
            "weights_sha256": resolved["weights_sha256"],
            "snapshot_sha256": artifacts["snapshot_sha256"],
            "recipe_sha256": resolved["recipe_sha256"],
            "embedding_dimension": resolved["embedding_dimension"],
            "dtype": resolved["dtype"],
        }
    return output


def import_verified_food_driver(
    driver_path: os.PathLike[str] | str = FOOD_DRIVER_PATH,
    *,
    expected_sha256: str = FOOD_DRIVER_SHA256,
) -> ModuleType:
    """Hash the archived driver before import and return its isolated module."""

    observed = lineage.verify_file_hash(driver_path, expected_sha256, field="Food driver")
    module_name = f"_fused3_verified_food_driver_{observed[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, str(Path(driver_path)))
    if spec is None or spec.loader is None:
        raise ImportError("could not create an import spec for the verified Food driver")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and a few provider helpers inspect sys.modules during class
    # creation under Python 3.9; register only after the hash check succeeded.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    factory = getattr(module, "_build_final_output_extractors", None)
    if not callable(factory):
        sys.modules.pop(module_name, None)
        raise AttributeError("verified Food driver lacks _build_final_output_extractors")
    declared = tuple(getattr(module, "_DEFAULT_MODELS", ()))
    if declared != FOOD_BACKBONE_IDS:
        raise ValueError("verified Food driver frozen backbone panel differs")
    return module


def verified_food_extractor_factory(
    driver_path: os.PathLike[str] | str = FOOD_DRIVER_PATH,
    *,
    expected_sha256: str = FOOD_DRIVER_SHA256,
) -> Callable[..., list[Any]]:
    """Return the exact archived factory only after source verification."""

    module = import_verified_food_driver(driver_path, expected_sha256=expected_sha256)
    archived_factory = getattr(module, "_build_final_output_extractors")

    def factory(
        models: Sequence[str] = FOOD_BACKBONE_IDS,
        *,
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> list[Any]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive strict int")
        requested = tuple(models)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("models must be a nonempty sequence without duplicates")
        if any(model not in FOOD_BACKBONE_IDS for model in requested):
            raise ValueError("models must be drawn from the frozen Food backbone panel")
        # Preserve the frozen panel order even when the caller requests a
        # subset, so method order cannot become a hidden recipe choice.
        if requested != tuple(model for model in FOOD_BACKBONE_IDS if model in requested):
            raise ValueError("models must use the frozen Food backbone order")
        return list(
            archived_factory(
                list(requested), batch_size=int(batch_size), device=device
            )
        )

    setattr(factory, "driver_sha256", expected_sha256)
    setattr(factory, "backbone_ids", FOOD_BACKBONE_IDS)
    return factory


def _hash_npy_file(path: Path) -> str:
    return lineage.sha256_path(path)


def _matrix_identity(matrix: np.ndarray) -> str:
    return lineage.sha256_bytes(np.ascontiguousarray(matrix).tobytes(order="C"))


def _cache_record(
    matrix_path: Path,
    manifest_path: Path,
    *,
    matrix: np.ndarray,
    dataset_id: str,
    sample_ids: Sequence[str],
    labels: Sequence[Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = require_ready_backbone_spec(spec)
    if len(sample_ids) != len(matrix) or len(labels) != len(matrix):
        raise ValueError("cache rows and sample identity are not aligned")
    sample_hash = datasets.sequence_sha256(sample_ids)
    label_hash = datasets.sequence_sha256(labels)
    # The manifest's matrix hash is the on-disk .npy hash, matching the prior
    # cache convention.  The raw values hash is carried inside the recipe so a
    # file-format rewrite cannot silently change numerical bytes.
    recipe = deepcopy(dict(resolved["recipe"]))
    recipe["matrix_values_sha256"] = _matrix_identity(matrix)
    recipe["sample_id_sha256"] = sample_hash
    recipe["label_sha256"] = label_hash
    recipe_sha = lineage.sha256_payload(recipe)
    return {
        "matrix_path": str(matrix_path),
        "manifest_path": str(manifest_path),
        # Filled into the external record after the manifest bytes exist.
        "manifest_sha256": "",
        "matrix_sha256": _hash_npy_file(matrix_path),
        "shape": [int(value) for value in matrix.shape],
        "dtype": "float32",
        "sample_ids_sha256": sample_hash,
        "labels_sha256": label_hash,
        "extractor_recipe": recipe,
        "extractor_recipe_sha256": recipe_sha,
        "model_weight_identity": {
            "model_id": resolved["model_id"],
            "snapshot_revision": resolved["snapshot_revision"],
            "weights_sha256": resolved["weights_sha256"],
            "snapshot_sha256": resolved["recipe"]["artifacts"]["snapshot_sha256"],
        },
        "model_weight_sha256": resolved["weights_sha256"],
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"embedding cache manifest is not readable: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("embedding cache manifest must be a JSON object")
    return dict(value)


def _manifest_without_external_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload.pop("manifest_sha256", None)
    return payload


def write_embedding_cache(
    matrix_path: os.PathLike[str] | str,
    matrix: Any,
    *,
    manifest_path: Optional[os.PathLike[str] | str] = None,
    dataset_id: str,
    sample_ids: Sequence[str],
    labels: Sequence[Any],
    backbone_spec: Mapping[str, Any],
    resume: bool = False,
) -> dict[str, Any]:
    """Atomically write one float32 cache or validate an existing one on resume."""

    path = Path(matrix_path)
    manifest = Path(manifest_path) if manifest_path is not None else path.with_suffix(".json")
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    resolved = require_ready_backbone_spec(backbone_spec)
    if dataset_id not in datasets.DATASET_IDS:
        raise ValueError("dataset_id is outside the frozen confirmation panel")
    values = np.asarray(matrix)
    if values.ndim != 2 or values.dtype != np.dtype("float32"):
        raise ValueError("embedding cache values must be a two-dimensional float32 matrix")
    if not np.isfinite(values).all():
        raise ValueError("embedding cache values must be finite")
    if values.shape[1] != int(resolved["embedding_dimension"]):
        raise ValueError("embedding dimension does not match resolved backbone spec")
    if len(sample_ids) != len(values) or len(labels) != len(values):
        raise ValueError("cache sample IDs/labels are not aligned with values")
    if len(set(str(value) for value in sample_ids)) != len(sample_ids):
        raise ValueError("cache sample IDs must be unique")
    if path.exists() or manifest.exists():
        if not resume:
            raise FileExistsError("embedding cache exists; pass resume=True to validate it")
        loaded_values, loaded_record = load_embedding_cache(
            path,
            manifest_path=manifest,
            dataset_id=dataset_id,
            sample_ids=sample_ids,
            labels=labels,
            backbone_spec=resolved,
        )
        del loaded_values
        return loaded_record
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, contiguous, allow_pickle=False)
    temporary.replace(path)
    record = _cache_record(
        path,
        manifest,
        matrix=contiguous,
        dataset_id=dataset_id,
        sample_ids=sample_ids,
        labels=labels,
        spec=resolved,
    )
    record["manifest_sha256"] = ""
    _atomic_write_bytes(manifest, (lineage.canonical_json(_manifest_without_external_hash(record)) + "\n").encode("utf-8"))
    record["manifest_sha256"] = lineage.sha256_path(manifest)
    return record


def load_embedding_cache(
    matrix_path: os.PathLike[str] | str,
    *,
    manifest_path: Optional[os.PathLike[str] | str] = None,
    dataset_id: Optional[str] = None,
    sample_ids: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[Any]] = None,
    backbone_spec: Optional[Mapping[str, Any]] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate cache bytes, recipe, and cohort identity before exposing rows."""

    path = Path(matrix_path)
    manifest = Path(manifest_path) if manifest_path is not None else path.with_suffix(".json")
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError("embedding cache matrix and manifest must both exist")
    record = _read_manifest(manifest)
    required = {
        "matrix_path",
        "manifest_path",
        "matrix_sha256",
        "shape",
        "dtype",
        "sample_ids_sha256",
        "labels_sha256",
        "extractor_recipe",
        "extractor_recipe_sha256",
        "model_weight_identity",
        "model_weight_sha256",
    }
    if set(record) != required:
        raise ValueError("embedding cache manifest has unknown or missing keys")
    if Path(record["matrix_path"]).resolve() != path.resolve() or Path(record["manifest_path"]).resolve() != manifest.resolve():
        raise ValueError("embedding cache manifest path identity mismatch")
    if record["dtype"] != "float32" or not isinstance(record["shape"], list) or len(record["shape"]) != 2:
        raise ValueError("embedding cache manifest shape/dtype is invalid")
    for value in record["shape"]:
        if type(value) is not int or value <= 0:
            raise ValueError("embedding cache shape must contain positive strict ints")
    for field in (
        "matrix_sha256",
        "sample_ids_sha256",
        "labels_sha256",
        "extractor_recipe_sha256",
        "model_weight_sha256",
    ):
        lineage.require_hash(record[field], field=field)
    if lineage.sha256_path(path) != record["matrix_sha256"]:
        raise ValueError("embedding cache matrix hash mismatch")
    if lineage.sha256_payload(record["extractor_recipe"]) != record["extractor_recipe_sha256"]:
        raise ValueError("embedding cache recipe hash mismatch")
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("embedding cache matrix cannot be loaded without pickle") from exc
    if values.ndim != 2 or values.dtype != np.dtype("float32") or tuple(values.shape) != tuple(record["shape"]):
        raise ValueError("embedding cache matrix shape/dtype mismatch")
    if not np.isfinite(np.asarray(values)).all():
        raise ValueError("embedding cache matrix contains non-finite values")
    if dataset_id is not None and dataset_id not in datasets.DATASET_IDS:
        raise ValueError("dataset_id is outside the frozen confirmation panel")
    if sample_ids is not None:
        if record["sample_ids_sha256"] != datasets.sequence_sha256(sample_ids):
            raise ValueError("embedding cache sample ID identity mismatch")
    if labels is not None:
        if record["labels_sha256"] != datasets.sequence_sha256(labels):
            raise ValueError("embedding cache label identity mismatch")
    if backbone_spec is not None:
        resolved = require_ready_backbone_spec(backbone_spec)
        if record["model_weight_sha256"] != resolved["weights_sha256"]:
            raise ValueError("embedding cache model-weight identity mismatch")
        expected_recipe = deepcopy(dict(resolved["recipe"]))
        expected_recipe["matrix_values_sha256"] = _matrix_identity(np.asarray(values))
        expected_recipe["sample_id_sha256"] = record["sample_ids_sha256"]
        expected_recipe["label_sha256"] = record["labels_sha256"]
        if lineage.sha256_payload(expected_recipe) != record["extractor_recipe_sha256"]:
            raise ValueError("embedding cache resolved recipe mismatch")
    external_record = dict(record)
    external_record["manifest_sha256"] = lineage.sha256_path(manifest)
    return values, external_record


def extract_union_embeddings(
    images: Sequence[Any],
    union: datasets.CohortUnion,
    *,
    backbone_id: str,
    extractor_factory: Callable[..., Sequence[Any]],
    batch_size: int = 16,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract one backbone exactly once over the precomputed cohort union."""

    _validate_backbone_id(backbone_id)
    if len(images) != union.row_count:
        raise ValueError("image rows must exactly match the precomputed cohort union")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive strict int")
    extractors = list(
        extractor_factory((backbone_id,), batch_size=batch_size, device=device)
    )
    if len(extractors) != 1:
        raise RuntimeError("verified factory must return exactly one requested extractor")
    extractor = extractors[0]
    try:
        if getattr(extractor, "extractor_type", None) == "openclip":
            transformed = extractor.transform_many({"image": list(images)})
        else:
            transformed = extractor.transform_many(list(images))
        outputs = [
            output
            for output in transformed
            if str(getattr(output, "name", ""))
            == str(FOOD_BACKBONE_SPECS[backbone_id]["recipe"]["output"])
        ]
        if len(outputs) != 1:
            raise ValueError("verified backbone must emit exactly one frozen final output")
        matrix = np.asarray(getattr(outputs[0], "embeddings"), dtype=np.float32)
    finally:
        release = getattr(extractor, "close", None)
        if callable(release):
            release()
    if matrix.ndim != 2 or len(matrix) != union.row_count:
        raise ValueError("extracted embeddings are not aligned with the cohort union")
    if matrix.shape[1] != int(FOOD_BACKBONE_SPECS[backbone_id]["embedding_dimension"]):
        raise ValueError("extracted embedding dimension differs from frozen recipe")
    if not np.isfinite(matrix).all():
        raise ValueError("extracted embeddings must be finite")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _cohort_registry_manifest(
    training_cohorts: Sequence[datasets.Cohort],
    evaluation_cohort: datasets.Cohort,
    union: datasets.CohortUnion,
) -> dict[str, Any]:
    """Serialize one dataset's complete pre-extraction cohort identity."""

    if tuple(cohort.seed for cohort in training_cohorts) != tuple(
        seed for seed in datasets.REPLICATE_SEEDS for _budget in datasets.TRAINING_BUDGETS
    ):
        raise ValueError("training cohorts are not in the frozen seed/budget order")
    replicates: dict[str, dict[str, dict[str, Any]]] = {}
    position = 0
    for seed in datasets.REPLICATE_SEEDS:
        budgets: dict[str, dict[str, Any]] = {}
        for budget in datasets.TRAINING_BUDGETS:
            cohort = training_cohorts[position]
            if cohort.seed != seed or cohort.budget != budget:
                raise ValueError("training cohort order does not match the frozen grid")
            budgets[str(budget)] = cohort.to_manifest()
            position += 1
        replicates[str(seed)] = budgets
    return {
        "replicates": replicates,
        "evaluation": evaluation_cohort.to_manifest(),
        "union": union.to_manifest(),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes((lineage.canonical_json(payload) + "\n").encode("utf-8"))
    temporary.replace(path)


def source_metadata_from_roots(
    raw_roots: Mapping[str, os.PathLike[str] | str],
    *,
    manifest_dir: os.PathLike[str] | str,
) -> dict[str, dict[str, Any]]:
    """Hash every frozen raw-dataset byte into external source manifests."""

    roots = {key: Path(value).resolve() for key, value in raw_roots.items()}
    if set(roots) != set(datasets.DATASET_IDS):
        raise ValueError("raw_roots must contain exactly the five frozen datasets")
    output_root = Path(manifest_dir)
    metadata: dict[str, dict[str, Any]] = {}
    for dataset_id in datasets.DATASET_IDS:
        root = roots[dataset_id]
        if not root.is_dir():
            raise FileNotFoundError(f"raw dataset root is missing: {root}")
        files: list[dict[str, Any]] = []
        total_bytes = 0
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"raw dataset root contains a symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            size = int(path.stat().st_size)
            files.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "sha256": lineage.sha256_path(path),
                }
            )
            total_bytes += size
        if not files:
            raise ValueError(f"raw dataset root has no files: {root}")
        archive_identity = [
            {"path": row["path"], "sha256": row["sha256"]} for row in files
        ]
        manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "root": str(root),
            "hash_recipe": "sorted_relative_path_size_and_file_sha256_v1",
            "file_count": len(files),
            "total_bytes": total_bytes,
            "archive_sha256": lineage.sha256_payload(archive_identity),
            "files": files,
        }
        manifest_path = output_root / f"{dataset_id}.source_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError("existing source manifest differs from raw dataset bytes")
        else:
            _atomic_json(manifest_path, manifest)
        metadata[dataset_id] = {
            "source": {
                "provider": "torchvision",
                "dataset_id": dataset_id,
                "raw_root": str(root),
                "manifest_path": str(manifest_path.resolve()),
                "file_count": len(files),
                "total_bytes": total_bytes,
                "hash_recipe": manifest["hash_recipe"],
            },
            "version": f"torchvision_{recipes.EXTRACTION_ENVIRONMENT['torchvision']}",
            "archive_sha256": manifest["archive_sha256"],
            "source_manifest_sha256": lineage.sha256_path(manifest_path),
        }
    return metadata


def download_frozen_datasets(
    raw_roots: Mapping[str, os.PathLike[str] | str],
) -> None:
    """Explicitly download and validate all five raw provider datasets."""

    if set(raw_roots) != set(datasets.DATASET_IDS):
        raise ValueError("raw_roots must contain exactly the five frozen datasets")
    for dataset_id in datasets.DATASET_IDS:
        datasets.load_confirmation_dataset(
            dataset_id,
            raw_root=raw_roots[dataset_id],
            download=True,
        )


def prepare_inputs(
    *,
    raw_roots: Mapping[str, os.PathLike[str] | str],
    output_dir: os.PathLike[str] | str,
    registry_path: os.PathLike[str] | str,
    protocol_sha256: str,
    lineage_lock_sha256: str,
    parent_registry_sha256: str,
    extraction_environment: Mapping[str, Any],
    model_weight_lock: Mapping[str, Any],
    backbone_specs: Mapping[str, Mapping[str, Any]],
    source_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    driver_path: os.PathLike[str] | str = FOOD_DRIVER_PATH,
    driver_sha256: str = FOOD_DRIVER_SHA256,
    download: bool = False,
    resume: bool = False,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Prepare and hash the complete five-dataset input registry.

    This is the only orchestration helper in the input slice.  It validates
    every raw-root/spec/weight/source identity before opening a model, builds
    each dataset's full train/evaluation cohort union exactly once, then
    invokes the caller's image loader once per dataset and the verified Food
    factory once per backbone.  It emits no downstream outcomes.  Audited raw
    source metadata is mandatory; omitting it fails before any dataset or model
    is opened.

    Images are materialized only through :func:`load_union_images`, the
    code-owned implementation that validates sample IDs and labels against the
    precomputed cohort union before any model is constructed.
    """

    if source_metadata is None:
        raise ValueError("source_metadata is required before input preparation")
    if download is not False:
        raise ValueError(
            "prepare_inputs never downloads; call download_frozen_datasets before hashing sources"
        )
    return _prepare_inputs_with_source_metadata(
        source_metadata=source_metadata,
        raw_roots=raw_roots,
        output_dir=output_dir,
        registry_path=registry_path,
        protocol_sha256=protocol_sha256,
        lineage_lock_sha256=lineage_lock_sha256,
        parent_registry_sha256=parent_registry_sha256,
        extraction_environment=extraction_environment,
        model_weight_lock=model_weight_lock,
        backbone_specs=backbone_specs,
        driver_path=driver_path,
        driver_sha256=driver_sha256,
        download=download,
        resume=resume,
        batch_size=batch_size,
        device=device,
    )


def _prepare_inputs_with_source_metadata(
    *,
    source_metadata: Mapping[str, Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Prepare inputs with explicit audited raw-source metadata.

    This named entry point keeps the default preparation path fail closed.  It
    is not a permissive alternate recipe: source metadata is validated against
    the exact five IDs and the resulting registry uses one canonical schema.
    """

    required_source_keys = {
        "source",
        "version",
        "archive_sha256",
        "source_manifest_sha256",
    }
    if not isinstance(source_metadata, Mapping) or set(source_metadata) != set(datasets.DATASET_IDS):
        raise ValueError("source_metadata must contain exactly the five frozen dataset IDs")
    normalized_source: dict[str, dict[str, Any]] = {}
    source_keys = {
        "provider",
        "dataset_id",
        "raw_root",
        "manifest_path",
        "file_count",
        "total_bytes",
        "hash_recipe",
    }
    for dataset_id in datasets.DATASET_IDS:
        record = source_metadata[dataset_id]
        if not isinstance(record, Mapping) or set(record) != required_source_keys:
            raise ValueError("source metadata has unknown or missing keys")
        if not isinstance(record["source"], Mapping) or set(record["source"]) != source_keys:
            raise ValueError("source metadata source has unknown or missing keys")
        source = record["source"]
        if source["provider"] != "torchvision" or source["dataset_id"] != dataset_id:
            raise ValueError("source metadata provider/dataset mismatch")
        if source["hash_recipe"] != "sorted_relative_path_size_and_file_sha256_v1":
            raise ValueError("source metadata hash recipe mismatch")
        for field in ("raw_root", "manifest_path"):
            if not isinstance(source[field], str) or not source[field]:
                raise ValueError(f"source metadata {field} must be nonempty")
        for field in ("file_count", "total_bytes"):
            if type(source[field]) is not int or source[field] <= 0:
                raise ValueError(f"source metadata {field} must be a positive strict int")
        if not isinstance(record["version"], str) or not record["version"]:
            raise ValueError("source metadata version is required")
        for field in ("archive_sha256", "source_manifest_sha256"):
            lineage.require_hash(record[field], field=field)
        normalized_source[dataset_id] = dict(record)

    manifest_roots = {
        dataset_id: Path(str(normalized_source[dataset_id]["source"]["manifest_path"])).parent
        for dataset_id in datasets.DATASET_IDS
    }
    if len(set(manifest_roots.values())) != 1:
        raise ValueError("source manifests must share one external manifest directory")
    raw_roots = kwargs["raw_roots"]
    recomputed_source = source_metadata_from_roots(
        raw_roots,
        manifest_dir=next(iter(manifest_roots.values())),
    )
    if normalized_source != recomputed_source:
        raise ValueError("source_metadata differs from the current raw dataset bytes")

    # Avoid duplicating the extraction loop above: perform the same validated
    # work with source records and publish only after all caches are complete.
    # Registry cache paths are absolute so copying the registry beside a run
    # cannot accidentally prepend its new parent to an already workspace-
    # relative path.
    output_root = Path(kwargs["output_dir"]).resolve()
    registry_file = Path(kwargs["registry_path"]).resolve()
    resume = kwargs.get("resume", False)
    if type(resume) is not bool:
        raise TypeError("resume must be a strict bool")
    if kwargs.get("batch_size", 16) != 16 or kwargs.get("device") != "cpu":
        raise ValueError("input extraction is frozen to batch_size=16 and device='cpu'")
    if registry_file.exists():
        if not resume:
            raise FileExistsError("audited input registry exists; pass resume=True to validate it")
        return datasets.load_and_validate_registry(
            registry_file,
            expected_protocol_sha=kwargs["protocol_sha256"],
            expected_lock_sha=kwargs["lineage_lock_sha256"],
            expected_parent_registry_sha=kwargs["parent_registry_sha256"],
        )
    # Re-run all fail-closed argument checks by invoking the main validation
    # surface's private-free portions explicitly, then construct the verified
    # factory once.
    for field in ("protocol_sha256", "lineage_lock_sha256", "parent_registry_sha256"):
        lineage.require_hash(kwargs[field], field=field)
    raw_map = dict(raw_roots)
    if set(raw_map) != set(datasets.DATASET_IDS):
        raise ValueError("raw_roots must contain exactly the five frozen datasets")
    specs = kwargs["backbone_specs"]
    if not isinstance(specs, Mapping) or set(specs) != set(FOOD_BACKBONE_IDS):
        raise ValueError("backbone_specs must contain exactly the ten frozen backbones")
    resolved_specs = {
        backbone: require_ready_backbone_spec(specs[backbone])
        for backbone in FOOD_BACKBONE_IDS
    }
    model_lock = kwargs["model_weight_lock"]
    if not isinstance(model_lock, Mapping) or set(model_lock) != set(FOOD_BACKBONE_IDS):
        raise ValueError("model_weight_lock must contain exactly the ten frozen backbones")
    if dict(kwargs["extraction_environment"]) != recipes.EXTRACTION_ENVIRONMENT:
        raise ValueError("extraction_environment differs from the frozen CPU recipe")
    expected_model_lock = model_weight_lock_from_specs(resolved_specs)
    if dict(model_lock) != expected_model_lock:
        raise ValueError("model_weight_lock differs from the resolved backbone specs")
    factory = verified_food_extractor_factory(
        kwargs.get("driver_path", FOOD_DRIVER_PATH),
        expected_sha256=kwargs.get("driver_sha256", FOOD_DRIVER_SHA256),
    )
    dataset_records: list[dict[str, Any]] = []
    for dataset_id in datasets.DATASET_IDS:
        bundle = datasets.load_confirmation_dataset(
            dataset_id,
            raw_root=raw_map[dataset_id],
            download=kwargs.get("download", False),
        )
        training = datasets.build_training_cohorts(bundle.training)
        evaluation = datasets.build_evaluation_cohort(
            bundle.evaluation,
            seed=datasets.EVALUATION_SEED,
        )
        union = datasets.union_cohort_samples(
            tuple(training) + (evaluation,),
            source_split_ids=(bundle.training.split, bundle.evaluation.split),
        )
        images: Optional[list[Any]] = None
        records: dict[str, dict[str, Any]] = {}
        for backbone in FOOD_BACKBONE_IDS:
            cache_dir = output_root / dataset_id
            matrix_path = cache_dir / f"{backbone}.npy"
            manifest_path = cache_dir / f"{backbone}.json"
            if resume and (matrix_path.exists() or manifest_path.exists()):
                loaded, record = load_embedding_cache(
                    matrix_path,
                    manifest_path=manifest_path,
                    dataset_id=dataset_id,
                    sample_ids=union.sample_ids,
                    labels=union.labels,
                    backbone_spec=resolved_specs[backbone],
                )
                del loaded
                records[backbone] = record
                continue
            if images is None:
                images = load_union_images(bundle, union)
                if len(images) != union.row_count:
                    raise ValueError(
                        "code-owned image materialization differs from the frozen union"
                    )
            matrix = extract_union_embeddings(
                images,
                union,
                backbone_id=backbone,
                extractor_factory=factory,
                batch_size=16,
                device="cpu",
            )
            records[backbone] = write_embedding_cache(
                matrix_path,
                matrix,
                manifest_path=manifest_path,
                dataset_id=dataset_id,
                sample_ids=union.sample_ids,
                labels=union.labels,
                backbone_spec=resolved_specs[backbone],
                resume=False,
            )
        source = normalized_source[dataset_id]
        dataset_records.append(
            {
                "dataset_id": dataset_id,
                "class_count": int(datasets.CLASS_COUNTS[dataset_id]),
                "source": dict(source["source"]),
                "split_recipe": dict(datasets.DATASET_SPLIT_RECIPES[dataset_id]),
                "version": source["version"],
                "archive_sha256": source["archive_sha256"],
                "source_manifest_sha256": source["source_manifest_sha256"],
                "sample_ids_sha256": union.sample_id_sha256,
                "label_sha256": union.label_sha256,
                "cohort_manifest": _cohort_registry_manifest(training, evaluation, union),
                "embeddings": records,
            }
        )
    registry = {
        "schema_version": 1,
        "registry_id": "fused3_confirmation_v2_inputs",
        "status": "audited_complete",
        "outcomes_inspected": False,
        "parent_registry_sha256": kwargs["parent_registry_sha256"],
        "protocol_sha256": kwargs["protocol_sha256"],
        "lineage_lock_sha256": kwargs["lineage_lock_sha256"],
        "extraction_environment": dict(kwargs["extraction_environment"]),
        "model_weight_lock": dict(model_lock),
        "datasets": dataset_records,
    }
    validated = datasets._validate_registry_shape(  # type: ignore[attr-defined]
        registry,
        expected_protocol_sha=kwargs["protocol_sha256"],
        expected_lock_sha=kwargs["lineage_lock_sha256"],
        expected_parent_registry_sha=kwargs["parent_registry_sha256"],
    )
    _atomic_json(registry_file, validated)
    return validated


def load_union_images(
    bundle: datasets.DatasetBundle,
    union: datasets.CohortUnion,
) -> list[Any]:
    """Materialize the predeclared union in exact sample-ID order."""

    if bundle.dataset_id != union.dataset_id:
        raise ValueError("dataset bundle and cohort union IDs differ")
    lookup: dict[str, tuple[Any, int, Any]] = {}
    for dataset, identity in (
        (bundle.training_dataset, bundle.training),
        (bundle.evaluation_dataset, bundle.evaluation),
    ):
        for index, (sample_id, label) in enumerate(
            zip(identity.sample_ids, identity.labels)
        ):
            if sample_id in lookup:
                raise ValueError("dataset sample IDs are not unique across splits")
            lookup[sample_id] = (dataset, index, label)
    images: list[Any] = []
    for sample_id, expected_label in zip(union.sample_ids, union.labels):
        if sample_id not in lookup:
            raise ValueError("cohort union refers to an unknown dataset row")
        dataset, index, identity_label = lookup[sample_id]
        item = dataset[index]
        if not isinstance(item, Sequence) or len(item) < 2:
            raise ValueError("torchvision dataset row must contain image and label")
        image, observed_label = item[0], item[1]
        if observed_label != identity_label or observed_label != expected_label:
            raise ValueError("dataset row label differs from the frozen cohort identity")
        images.append(image)
    if len(images) != union.row_count:
        raise RuntimeError("materialized image count differs from the frozen union")
    return images


__all__ = [
    "FOOD_BACKBONE_IDS",
    "FOOD_BACKBONE_SPECS",
    "FOOD_DRIVER_PATH",
    "FOOD_DRIVER_SHA256",
    "backbone_spec",
    "backbone_specs",
    "download_frozen_datasets",
    "extract_union_embeddings",
    "import_verified_food_driver",
    "load_embedding_cache",
    "load_union_images",
    "model_weight_lock_from_specs",
    "observed_extraction_environment",
    "prepare_inputs",
    "require_ready_backbone_spec",
    "resolve_backbone_spec",
    "resolved_local_backbone_specs",
    "source_metadata_from_roots",
    "verified_food_extractor_factory",
    "write_embedding_cache",
]
