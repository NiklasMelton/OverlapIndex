"""Closed torchvision inputs and deterministic cohorts for confirmation V2.

This module owns identities, sampling, and cache validation only.  It never
downloads data implicitly and never reads a model embedding to choose a
cohort.  A caller must explicitly request ``download=True`` to let torchvision
obtain missing raw files; the confirmation runner should normally perform its
own audited download step before invoking the loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from . import lineage
from . import recipes


DATASET_IDS = tuple(recipes.DATASET_IDS)
TRAINING_BUDGETS = (32, 64)
REPLICATE_SEEDS = tuple(recipes.REPLICATE_SEEDS)
EVALUATION_SEED = int(recipes.EVALUATION_SEED)
EVALUATION_PER_CLASS = 20
TRAINING_CLASS_MINIMUM = 64

CLASS_COUNTS = {
    "torchvision_cifar10": 10,
    "torchvision_stl10": 10,
    "torchvision_gtsrb": 43,
    "torchvision_fgvc_aircraft": 100,
    "torchvision_dtd": 47,
}

# These are the official torchvision row counts for the exact split recipe.
# DTD partition 1 combines its train and validation partitions before any
# cohort is sampled, as required by the v2 protocol.
EXPECTED_ROW_COUNTS = {
    "torchvision_cifar10": {"train": 50_000, "test": 10_000},
    "torchvision_stl10": {"train": 5_000, "test": 8_000},
    "torchvision_gtsrb": {"train": 26_640, "test": 12_630},
    "torchvision_fgvc_aircraft": {"trainval": 6_667, "test": 3_333},
    "torchvision_dtd": {
        "train_plus_val_partition_1": 3_760,
        "test_partition_1": 1_880,
    },
}

DATASET_SPLIT_RECIPES = {
    "torchvision_cifar10": {"training": "train", "evaluation": "test"},
    "torchvision_stl10": {"training": "train", "evaluation": "test"},
    "torchvision_gtsrb": {"training": "train", "evaluation": "test"},
    "torchvision_fgvc_aircraft": {
        "training": "trainval",
        "evaluation": "test",
    },
    "torchvision_dtd": {
        "training": "train_plus_val_partition_1",
        "evaluation": "test_partition_1",
    },
}
DATASET_PARTITIONS = {
    "torchvision_cifar10": None,
    "torchvision_stl10": None,
    "torchvision_gtsrb": None,
    "torchvision_fgvc_aircraft": "variant",
    "torchvision_dtd": 1,
}
DATASET_TERMS = {
    "torchvision_cifar10": "CIFAR-10 terms and torchvision attribution must be recorded at extraction",
    "torchvision_stl10": "STL-10 terms and torchvision attribution must be recorded at extraction",
    "torchvision_gtsrb": "GTSRB terms and torchvision attribution must be recorded at extraction",
    "torchvision_fgvc_aircraft": "FGVC Aircraft terms and torchvision attribution must be recorded at extraction",
    "torchvision_dtd": "DTD terms and torchvision attribution must be recorded at extraction",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return _json_safe(converted)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("identity labels cannot contain non-finite values")
        return value
    raise TypeError(f"label {value!r} is not a JSON-safe scalar")


def canonical_json(value: Any) -> str:
    return lineage.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sequence_sha256(values: Sequence[Any]) -> str:
    """Hash an ordered identity sequence, preserving labels exactly."""

    return sha256_bytes(canonical_json(list(values)).encode("utf-8"))


def indices_sha256(values: Sequence[int]) -> str:
    normalized = []
    for value in values:
        if type(value) is not int or value < 0:
            raise TypeError("row indices must be nonnegative strict ints")
        normalized.append(int(value))
    return sequence_sha256(normalized)


def _strict_seed(value: Any, *, field: str = "seed") -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a strict int")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return int(value)


def _strict_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a strict bool")
    return bool(value)


def _strict_hash(value: Any, *, field: str, allow_none: bool = False) -> Optional[str]:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lower-case 64-hex SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lower-case 64-hex SHA-256 digest")
    return value


def _strict_sequence(values: Any, *, field: str) -> list[Any]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return list(values)


@dataclass(frozen=True)
class SplitIdentity:
    """Stable row identity for one official dataset split."""

    dataset_id: str
    split: str
    sample_ids: tuple[str, ...]
    labels: tuple[Any, ...]
    class_count: int

    @property
    def row_count(self) -> int:
        return len(self.sample_ids)

    @property
    def sample_id_sha256(self) -> str:
        return sequence_sha256(self.sample_ids)

    @property
    def label_sha256(self) -> str:
        return sequence_sha256(self.labels)


def make_split_identity(
    dataset_id: str,
    split: str,
    sample_ids: Sequence[str],
    labels: Sequence[Any],
    *,
    class_count: Optional[int] = None,
) -> SplitIdentity:
    """Construct and validate one stable split identity.

    The helper is intentionally the only public constructor for identities
    created by a provider adapter.  It does not infer or reorder rows: the
    supplied sample-ID/label order is the order later used for cohort indices
    and extraction.  ``class_count`` is optional only to make the helper
    convenient for provider tests; when supplied it must equal the frozen
    count for the dataset.
    """

    dataset_value = _validate_dataset_id(dataset_id)
    if not isinstance(split, str) or not split:
        raise TypeError("split must be a nonempty string")
    ids = _strict_sequence(sample_ids, field="sample_ids")
    target = [_json_safe(value) for value in _strict_sequence(labels, field="labels")]
    if len(ids) != len(target):
        raise ValueError("sample IDs and labels must have equal length")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("sample IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("sample IDs must be unique")
    expected_classes = CLASS_COUNTS[dataset_value]
    if class_count is None:
        class_value = expected_classes
    elif type(class_count) is not int:
        raise TypeError("class_count must be a strict int")
    else:
        class_value = int(class_count)
        if class_value != expected_classes:
            raise ValueError("class_count does not match the frozen dataset")
    if any(type(value) is not int for value in target):
        raise ValueError("dataset labels must be strict integer class IDs")
    if set(target) != set(range(class_value)):
        raise ValueError("dataset labels must cover the exact frozen class set")
    return SplitIdentity(
        dataset_value,
        split,
        tuple(ids),
        tuple(target),
        class_value,
    )


@dataclass(frozen=True)
class Cohort:
    """A deterministic row subset, retaining source indices and labels."""

    dataset_id: str
    split: str
    cohort_type: str
    seed: int
    budget: Optional[int]
    per_class: int
    indices: tuple[int, ...]
    sample_ids: tuple[str, ...]
    labels: tuple[Any, ...]

    @property
    def row_count(self) -> int:
        return len(self.indices)

    @property
    def sample_id_sha256(self) -> str:
        return sequence_sha256(self.sample_ids)

    @property
    def label_sha256(self) -> str:
        return sequence_sha256(self.labels)

    @property
    def indices_sha256(self) -> str:
        return indices_sha256(self.indices)

    def to_manifest(self) -> dict[str, Any]:
        """Return the one JSON-safe cohort descriptor used by the registry."""

        payload: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "cohort_type": self.cohort_type,
            "seed": int(self.seed),
            "budget": None if self.budget is None else int(self.budget),
            "per_class": int(self.per_class),
            "row_count": int(self.row_count),
            "row_indices": [int(value) for value in self.indices],
            "sample_ids": list(self.sample_ids),
            "labels": [_json_safe(value) for value in self.labels],
            "sample_id_sha256": self.sample_id_sha256,
            "label_sha256": self.label_sha256,
            "row_indices_sha256": self.indices_sha256,
        }
        return payload


@dataclass(frozen=True)
class CohortUnion:
    """The one extraction input formed before any model is instantiated."""

    dataset_id: str
    source_split_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    labels: tuple[Any, ...]
    memberships: tuple[tuple[str, ...], ...]

    @property
    def row_count(self) -> int:
        return len(self.sample_ids)

    @property
    def sample_id_sha256(self) -> str:
        return sequence_sha256(self.sample_ids)

    @property
    def label_sha256(self) -> str:
        return sequence_sha256(self.labels)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_split_ids": list(self.source_split_ids),
            "row_count": int(self.row_count),
            "sample_ids": list(self.sample_ids),
            "labels": [_json_safe(value) for value in self.labels],
            "sample_id_sha256": self.sample_id_sha256,
            "label_sha256": self.label_sha256,
            "memberships": [list(value) for value in self.memberships],
        }


class _ConcatenatedDataset:
    """Tiny torchvision-compatible concatenation without importing torch."""

    def __init__(self, datasets: Sequence[Any]) -> None:
        self._datasets = tuple(datasets)
        self._lengths = tuple(len(dataset) for dataset in self._datasets)
        cumulative = []
        total = 0
        for length in self._lengths:
            total += int(length)
            cumulative.append(total)
        self._cumulative = tuple(cumulative)

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, index: int) -> Any:
        if type(index) is not int:
            raise TypeError("dataset index must be a strict int")
        if index < 0 or index >= len(self):
            raise IndexError(index)
        previous = 0
        for cumulative, dataset in zip(self._cumulative, self._datasets):
            if index < cumulative:
                return dataset[index - previous]
            previous = cumulative
        raise IndexError(index)


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded raw datasets plus independently validated split identities."""

    dataset_id: str
    provider_version: str
    training_dataset: Any
    evaluation_dataset: Any
    training: SplitIdentity
    evaluation: SplitIdentity
    spec: Mapping[str, Any]

    @property
    def class_count(self) -> int:
        return int(self.training.class_count)


def dataset_spec(
    dataset_id: str,
    *,
    raw_root: os.PathLike[str] | str,
    provider_version: Optional[str] = None,
    raw_manifest_sha256: Optional[str] = None,
    sample_id_sha256: Optional[str] = None,
    label_sha256: Optional[str] = None,
    download_required: Optional[bool] = None,
) -> dict[str, Any]:
    """Build the exact closed dataset specification surface.

    Hashes may be ``None`` only for a planning descriptor.  A ready registry
    must contain all hashes; :func:`load_and_validate_registry` enforces that.
    """

    _validate_dataset_id(dataset_id)
    root = Path(raw_root)
    if download_required is None:
        download_required = not root.exists()
    _strict_bool(download_required, field="download_required")
    return {
        "dataset_id": dataset_id,
        "provider": "torchvision",
        "provider_version": (
            str(provider_version)
            if provider_version is not None
            else "torchvision_dataset_definition_pinned_at_extraction"
        ),
        "split_recipe": dict(DATASET_SPLIT_RECIPES[dataset_id]),
        "partition": DATASET_PARTITIONS[dataset_id],
        "class_count": int(CLASS_COUNTS[dataset_id]),
        "expected_row_count": dict(EXPECTED_ROW_COUNTS[dataset_id]),
        "download_required": bool(download_required),
        "raw_root": str(root),
        "raw_manifest_sha256": _strict_hash(
            raw_manifest_sha256, field="raw_manifest_sha256", allow_none=True
        ),
        "sample_id_sha256": _strict_hash(
            sample_id_sha256, field="sample_id_sha256", allow_none=True
        ),
        "label_sha256": _strict_hash(
            label_sha256, field="label_sha256", allow_none=True
        ),
        "license_or_terms": DATASET_TERMS[dataset_id],
    }


def dataset_specs(
    raw_roots: Mapping[str, os.PathLike[str] | str],
    *,
    provider_version: Optional[str] = None,
) -> tuple[dict[str, Any], ...]:
    """Return the frozen five specs in protocol order; reject additions."""

    if set(raw_roots) != set(DATASET_IDS):
        raise ValueError("raw_roots must contain exactly the five frozen dataset IDs")
    return tuple(
        dataset_spec(dataset_id, raw_root=raw_roots[dataset_id], provider_version=provider_version)
        for dataset_id in DATASET_IDS
    )


def _validate_dataset_id(dataset_id: Any) -> str:
    if not isinstance(dataset_id, str) or dataset_id not in DATASET_IDS:
        raise ValueError(f"dataset_id must be one of {DATASET_IDS!r}; got {dataset_id!r}")
    return dataset_id


def _torchvision_version_and_datasets() -> tuple[str, Any]:
    try:
        import torchvision
        from torchvision import datasets as tv_datasets
    except Exception as exc:  # pragma: no cover - provider environment dependent
        raise RuntimeError("torchvision is required to load confirmation datasets") from exc
    return str(getattr(torchvision, "__version__", "unknown")), tv_datasets


def _dataset_targets(dataset: Any) -> list[Any]:
    for name in ("targets", "labels", "_labels"):
        if hasattr(dataset, name):
            values = getattr(dataset, name)
            if values is not None:
                if hasattr(values, "tolist"):
                    values = values.tolist()
                return list(values)
    # torchvision 0.23's GTSRB keeps its stable, deterministic row order and
    # scalar class labels together in ``_samples`` rather than exposing a
    # separate targets/labels attribute.  Read only the labels here; the
    # sample paths remain provider-owned inputs and are hashed separately.
    if hasattr(dataset, "_samples"):
        samples = getattr(dataset, "_samples")
        if samples is not None:
            rows = list(samples)
            if all(isinstance(row, (tuple, list)) and len(row) == 2 for row in rows):
                return [row[1] for row in rows]
    raise ValueError(f"dataset {type(dataset).__name__} exposes no stable labels")


def _split_identity(
    dataset_id: str,
    split: str,
    dataset: Any,
    *,
    expected_rows: int,
    class_count: int,
    id_prefix: str,
) -> SplitIdentity:
    labels = [_json_safe(value) for value in _dataset_targets(dataset)]
    if len(dataset) != expected_rows or len(labels) != expected_rows:
        raise ValueError(
            f"{dataset_id} {split} expected {expected_rows} rows, got {len(dataset)}"
        )
    if any(isinstance(value, (list, dict, set, tuple)) for value in labels):
        raise ValueError("dataset labels must be scalar values")
    observed = set(labels)
    expected = set(range(class_count))
    if observed != expected:
        raise ValueError(
            f"{dataset_id} {split} labels do not cover the exact class set"
        )
    counts = {label: labels.count(label) for label in expected}
    if any(count <= 0 for count in counts.values()):
        raise ValueError(f"{dataset_id} {split} has an empty class")
    sample_ids = tuple(f"{id_prefix}/{index:06d}" for index in range(expected_rows))
    if len(set(sample_ids)) != expected_rows:
        raise RuntimeError("generated dataset sample IDs are not unique")
    return make_split_identity(
        dataset_id,
        split,
        sample_ids,
        labels,
        class_count=class_count,
    )


def load_confirmation_dataset(
    dataset_id: str,
    *,
    raw_root: os.PathLike[str] | str,
    download: bool = False,
    annotation_level: str = "variant",
) -> DatasetBundle:
    """Load one exact torchvision dataset recipe without implicit downloads."""

    _validate_dataset_id(dataset_id)
    _strict_bool(download, field="download")
    if not isinstance(annotation_level, str) or annotation_level != "variant":
        raise ValueError("annotation_level is frozen to 'variant'")
    root = Path(raw_root)
    if not root.exists() and not download:
        raise FileNotFoundError(
            f"raw root {root} is missing; pass download=True only in an explicit audited preparation step"
        )
    provider_version, tv = _torchvision_version_and_datasets()
    common = {"root": str(root), "download": bool(download), "transform": None}
    if dataset_id == "torchvision_cifar10":
        train_dataset = tv.CIFAR10(train=True, **common)
        eval_dataset = tv.CIFAR10(train=False, **common)
        train_split, eval_split = "train", "test"
        train_prefix, eval_prefix = f"{dataset_id}/train", f"{dataset_id}/test"
    elif dataset_id == "torchvision_stl10":
        train_dataset = tv.STL10(split="train", **common)
        eval_dataset = tv.STL10(split="test", **common)
        train_split, eval_split = "train", "test"
        train_prefix, eval_prefix = f"{dataset_id}/train", f"{dataset_id}/test"
    elif dataset_id == "torchvision_gtsrb":
        train_dataset = tv.GTSRB(split="train", **common)
        eval_dataset = tv.GTSRB(split="test", **common)
        train_split, eval_split = "train", "test"
        train_prefix, eval_prefix = f"{dataset_id}/train", f"{dataset_id}/test"
    elif dataset_id == "torchvision_fgvc_aircraft":
        aircraft_common = dict(common)
        aircraft_common["annotation_level"] = annotation_level
        train_dataset = tv.FGVCAircraft(split="trainval", **aircraft_common)
        eval_dataset = tv.FGVCAircraft(split="test", **aircraft_common)
        train_split, eval_split = "trainval", "test"
        train_prefix, eval_prefix = f"{dataset_id}/trainval", f"{dataset_id}/test"
    else:
        dtd_common = dict(common)
        dtd_common["partition"] = 1
        train_part = tv.DTD(split="train", **dtd_common)
        val_part = tv.DTD(split="val", **dtd_common)
        eval_dataset = tv.DTD(split="test", **dtd_common)
        train_dataset = _ConcatenatedDataset((train_part, val_part))
        train_split, eval_split = (
            "train_plus_val_partition_1",
            "test_partition_1",
        )
        # The split identity includes the original partition name so that a
        # train/val row cannot collide even when both contain the same index.
        train_ids = tuple(
            f"{dataset_id}/train_partition-1/{index:06d}"
            for index in range(len(train_part))
        ) + tuple(
            f"{dataset_id}/val_partition-1/{index:06d}"
            for index in range(len(val_part))
        )
        train_labels = tuple(
            [_json_safe(value) for value in _dataset_targets(train_part)]
            + [_json_safe(value) for value in _dataset_targets(val_part)]
        )
        if len(train_dataset) != EXPECTED_ROW_COUNTS[dataset_id][train_split]:
            raise ValueError("DTD train+val partition 1 row count mismatch")
        if len(train_labels) != len(train_ids):
            raise ValueError("DTD train+val labels and sample IDs are misaligned")
        if set(train_labels) != set(range(CLASS_COUNTS[dataset_id])):
            raise ValueError("DTD train+val labels do not cover the exact class set")
        train_identity = make_split_identity(
            dataset_id,
            train_split,
            train_ids,
            train_labels,
            class_count=CLASS_COUNTS[dataset_id],
        )
        eval_identity = _split_identity(
            dataset_id,
            eval_split,
            eval_dataset,
            expected_rows=EXPECTED_ROW_COUNTS[dataset_id][eval_split],
            class_count=CLASS_COUNTS[dataset_id],
            id_prefix=f"{dataset_id}/test_partition-1",
        )
        spec = dataset_spec(
            dataset_id,
            raw_root=root,
            provider_version=provider_version,
            sample_id_sha256=sequence_sha256(
                train_identity.sample_ids + eval_identity.sample_ids
            ),
            label_sha256=sequence_sha256(
                train_identity.labels + eval_identity.labels
            ),
            download_required=False,
        )
        return DatasetBundle(
            dataset_id,
            provider_version,
            train_dataset,
            eval_dataset,
            train_identity,
            eval_identity,
            spec,
        )
    train_identity = _split_identity(
        dataset_id,
        train_split,
        train_dataset,
        expected_rows=EXPECTED_ROW_COUNTS[dataset_id][train_split],
        class_count=CLASS_COUNTS[dataset_id],
        id_prefix=train_prefix,
    )
    eval_identity = _split_identity(
        dataset_id,
        eval_split,
        eval_dataset,
        expected_rows=EXPECTED_ROW_COUNTS[dataset_id][eval_split],
        class_count=CLASS_COUNTS[dataset_id],
        id_prefix=eval_prefix,
    )
    spec = dataset_spec(
        dataset_id,
        raw_root=root,
        provider_version=provider_version,
        sample_id_sha256=sequence_sha256(
            train_identity.sample_ids + eval_identity.sample_ids
        ),
        label_sha256=sequence_sha256(train_identity.labels + eval_identity.labels),
        download_required=False,
    )
    return DatasetBundle(
        dataset_id,
        provider_version,
        train_dataset,
        eval_dataset,
        train_identity,
        eval_identity,
        spec,
    )


def _validate_split_for_cohort(split: SplitIdentity) -> None:
    _validate_dataset_id(split.dataset_id)
    if len(split.sample_ids) != len(split.labels):
        raise ValueError("split sample IDs and labels must be aligned")
    if len(set(split.sample_ids)) != len(split.sample_ids):
        raise ValueError("split sample IDs must be unique")
    if split.class_count != CLASS_COUNTS[split.dataset_id]:
        raise ValueError("split class count does not match frozen dataset")
    if any(not isinstance(value, (str, bool, int, float)) or isinstance(value, (bytes, bytearray)) for value in split.labels):
        raise ValueError("split labels must be scalar values")


def _class_positions(split: SplitIdentity) -> tuple[Any, ...]:
    labels = []
    for label in split.labels:
        if label not in labels:
            labels.append(label)
    expected = set(range(split.class_count))
    if set(labels) != expected:
        raise ValueError("split labels must be exactly the frozen integer class IDs")
    return tuple(labels)


def _cohort_from_indices(
    split: SplitIdentity,
    indices: Sequence[int],
    *,
    cohort_type: str,
    seed: int,
    budget: Optional[int],
    per_class: int,
) -> Cohort:
    ordered = tuple(int(value) for value in indices)
    if len(set(ordered)) != len(ordered):
        raise ValueError("cohort row indices must be unique")
    if any(value < 0 or value >= split.row_count for value in ordered):
        raise ValueError("cohort row index is outside its source split")
    sample_ids = tuple(split.sample_ids[index] for index in ordered)
    labels = tuple(split.labels[index] for index in ordered)
    counts = {label: labels.count(label) for label in _class_positions(split)}
    if any(count != per_class for count in counts.values()):
        raise ValueError("cohort does not contain the exact requested rows per class")
    return Cohort(
        split.dataset_id,
        split.split,
        cohort_type,
        _strict_seed(seed),
        budget,
        int(per_class),
        ordered,
        sample_ids,
        labels,
    )


def build_training_cohorts(
    split: SplitIdentity,
    *,
    seeds: Sequence[int] = REPLICATE_SEEDS,
    budgets: Sequence[int] = TRAINING_BUDGETS,
) -> tuple[Cohort, ...]:
    """Build the exact nested 32/64-per-class cohorts for five seeds."""

    _validate_split_for_cohort(split)
    seed_values = tuple(_strict_seed(value, field="replicate seed") for value in seeds)
    if seed_values != REPLICATE_SEEDS:
        raise ValueError("training seeds must equal the five frozen replicate seeds")
    budget_values = tuple(budgets)
    if budget_values != TRAINING_BUDGETS:
        raise ValueError("training budgets must equal frozen (32, 64)")
    classes = _class_positions(split)
    per_class_indices = {
        label: np.asarray(
            [index for index, observed in enumerate(split.labels) if observed == label],
            dtype=np.int64,
        )
        for label in classes
    }
    if any(len(indices) < max(budget_values) for indices in per_class_indices.values()):
        raise ValueError("training split lacks 64 rows for at least one class")
    output: list[Cohort] = []
    for seed in seed_values:
        # One independent permutation per class, with the class position in
        # the SeedSequence.  Prefixes therefore remain nested while labels
        # with different representations cannot affect RNG state.
        permuted: dict[Any, np.ndarray] = {}
        for class_position, label in enumerate(classes):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(class_position)])
            )
            permuted[label] = per_class_indices[label][rng.permutation(len(per_class_indices[label]))]
        for budget in budget_values:
            selected: list[int] = []
            for label in classes:
                selected.extend(int(value) for value in permuted[label][: int(budget)])
            output.append(
                _cohort_from_indices(
                    split,
                    selected,
                    cohort_type="training",
                    seed=seed,
                    budget=int(budget),
                    per_class=int(budget),
                )
            )
    return tuple(output)


def build_evaluation_cohort(
    split: SplitIdentity,
    *,
    seed: int,
    per_class: int = EVALUATION_PER_CLASS,
) -> Cohort:
    """Build one fixed evaluation cohort; the seed is intentionally required."""

    _validate_split_for_cohort(split)
    seed_value = _strict_seed(seed, field="evaluation seed")
    if seed_value != EVALUATION_SEED:
        raise ValueError("evaluation seed must equal the predeclared frozen seed")
    if type(per_class) is not int or per_class != EVALUATION_PER_CLASS:
        raise ValueError("evaluation per_class must equal the frozen value 20")
    classes = _class_positions(split)
    selected: list[int] = []
    for class_position, label in enumerate(classes):
        positions = np.asarray(
            [index for index, observed in enumerate(split.labels) if observed == label],
            dtype=np.int64,
        )
        if len(positions) < per_class:
            raise ValueError("evaluation split lacks 20 rows for at least one class")
        rng = np.random.default_rng(
            np.random.SeedSequence([seed_value, int(class_position), 20_20_20])
        )
        selected.extend(int(value) for value in positions[rng.permutation(len(positions))[:per_class]])
    return _cohort_from_indices(
        split,
        selected,
        cohort_type="evaluation",
        seed=seed_value,
        budget=None,
        per_class=per_class,
    )


def union_cohort_samples(
    cohorts: Sequence[Cohort],
    *,
    source_split_ids: Optional[Sequence[str]] = None,
) -> CohortUnion:
    """Form the deduplicated extraction cohort before any model is built."""

    if not isinstance(cohorts, Sequence) or not cohorts:
        raise ValueError("cohorts must be a non-empty sequence")
    dataset_id = cohorts[0].dataset_id
    for cohort in cohorts:
        if not isinstance(cohort, Cohort):
            raise TypeError("cohorts must contain Cohort objects")
        if cohort.dataset_id != dataset_id:
            raise ValueError("all cohorts must belong to one dataset")
    expected_splits = tuple(
        source_split_ids
        if source_split_ids is not None
        else dict.fromkeys(cohort.split for cohort in cohorts)
    )
    if not expected_splits:
        raise ValueError("source_split_ids must not be empty")
    by_id: dict[str, tuple[Any, list[str]]] = {}
    for cohort in cohorts:
        membership = f"{cohort.cohort_type}:{cohort.seed}:{cohort.budget}"
        for sample_id, label in zip(cohort.sample_ids, cohort.labels):
            if sample_id in by_id:
                old_label, memberships = by_id[sample_id]
                if old_label != label:
                    raise ValueError(f"sample {sample_id!r} has conflicting labels")
                memberships.append(membership)
            else:
                by_id[sample_id] = (label, [membership])
    sample_ids = tuple(by_id)
    labels = tuple(by_id[sample_id][0] for sample_id in sample_ids)
    memberships = tuple(tuple(by_id[sample_id][1]) for sample_id in sample_ids)
    return CohortUnion(
        dataset_id,
        tuple(str(value) for value in expected_splits),
        sample_ids,
        labels,
        memberships,
    )


def _registry_keys() -> tuple[str, ...]:
    return (
        "schema_version",
        "registry_id",
        "status",
        "outcomes_inspected",
        "parent_registry_sha256",
        "protocol_sha256",
        "lineage_lock_sha256",
        "extraction_environment",
        "model_weight_lock",
        "datasets",
    )


def _dataset_record_keys() -> tuple[str, ...]:
    return (
        "dataset_id",
        "class_count",
        "source",
        "split_recipe",
        "version",
        "archive_sha256",
        "source_manifest_sha256",
        "sample_ids_sha256",
        "label_sha256",
        "cohort_manifest",
        "embeddings",
    )


def _require_exact_mapping(value: Any, keys: Sequence[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{field} must contain exactly {tuple(keys)!r}")
    return value


def _validate_cohort_descriptor(
    descriptor: Mapping[str, Any],
    *,
    dataset_id: str,
    expected_type: str,
    expected_seed: int,
    expected_budget: Optional[int],
    expected_per_class: int,
    union_sample_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    expected_keys = {
        "dataset_id",
        "split",
        "cohort_type",
        "seed",
        "budget",
        "per_class",
        "row_count",
        "row_indices",
        "sample_ids",
        "labels",
        "sample_id_sha256",
        "label_sha256",
        "row_indices_sha256",
    }
    if not isinstance(descriptor, Mapping) or set(descriptor) != expected_keys:
        raise ValueError("cohort descriptor has unknown or missing keys")
    if descriptor["dataset_id"] != dataset_id or descriptor["cohort_type"] != expected_type:
        raise ValueError("cohort descriptor dataset/type mismatch")
    if type(descriptor["seed"]) is not int or int(descriptor["seed"]) != expected_seed:
        raise ValueError("cohort descriptor seed mismatch")
    if descriptor["budget"] != expected_budget:
        raise ValueError("cohort descriptor budget mismatch")
    if type(descriptor["per_class"]) is not int or int(descriptor["per_class"]) != expected_per_class:
        raise ValueError("cohort descriptor per_class mismatch")
    indices = _strict_sequence(descriptor["row_indices"], field="row_indices")
    sample_ids = _strict_sequence(descriptor["sample_ids"], field="sample_ids")
    labels = _strict_sequence(descriptor["labels"], field="labels")
    if len(indices) != len(sample_ids) or len(labels) != len(sample_ids):
        raise ValueError("cohort descriptor rows are not aligned")
    if type(descriptor["row_count"]) is not int or descriptor["row_count"] != len(sample_ids):
        raise ValueError("cohort descriptor row_count mismatch")
    if sequence_sha256(sample_ids) != descriptor["sample_id_sha256"]:
        raise ValueError("cohort descriptor sample ID hash mismatch")
    if sequence_sha256(labels) != descriptor["label_sha256"]:
        raise ValueError("cohort descriptor label hash mismatch")
    if indices_sha256(indices) != descriptor["row_indices_sha256"]:
        raise ValueError("cohort descriptor row-index hash mismatch")
    counts = {label: labels.count(label) for label in range(CLASS_COUNTS[dataset_id])}
    if any(count != expected_per_class for count in counts.values()):
        raise ValueError("cohort descriptor class balance mismatch")
    if union_sample_ids is not None and any(sample_id not in union_sample_ids for sample_id in sample_ids):
        raise ValueError("cohort descriptor refers to a sample outside the union")
    return dict(descriptor)


def _validate_embedding_record(
    record: Mapping[str, Any], *, backbone_id: str
) -> dict[str, Any]:
    expected_keys = {
        "matrix_path",
        "manifest_path",
        "manifest_sha256",
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
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise ValueError(f"embedding record for {backbone_id!r} has unknown or missing keys")
    for field in (
        "manifest_sha256",
        "matrix_sha256",
        "sample_ids_sha256",
        "labels_sha256",
        "extractor_recipe_sha256",
        "model_weight_sha256",
    ):
        _strict_hash(record[field], field=field)
    if not isinstance(record["matrix_path"], str) or not record["matrix_path"]:
        raise ValueError("embedding matrix_path must be a nonempty string")
    if not isinstance(record["manifest_path"], str) or not record["manifest_path"]:
        raise ValueError("embedding manifest_path must be a nonempty string")
    if record["dtype"] != "float32":
        raise ValueError("confirmation embedding caches must be float32")
    shape = _strict_sequence(record["shape"], field="embedding shape")
    if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("embedding shape must contain two positive strict ints")
    if not isinstance(record["extractor_recipe"], Mapping):
        raise ValueError("extractor_recipe must be a mapping")
    if not isinstance(record["model_weight_identity"], Mapping):
        raise ValueError("model_weight_identity must be a mapping")
    return dict(record)


def _validate_registry_shape(
    registry: Mapping[str, Any],
    *,
    expected_protocol_sha: Optional[str] = None,
    expected_lock_sha: Optional[str] = None,
    expected_parent_registry_sha: Optional[str] = None,
) -> dict[str, Any]:
    _require_exact_mapping(registry, _registry_keys(), field="registry")
    if registry["schema_version"] != 1 or registry["registry_id"] != "fused3_confirmation_v2_inputs":
        raise ValueError("registry schema or registry_id mismatch")
    if registry["status"] != "audited_complete":
        raise ValueError("registry must be audited_complete before outcomes")
    if type(registry["outcomes_inspected"]) is not bool or registry["outcomes_inspected"]:
        raise ValueError("registry outcomes_inspected must be exactly false")
    for field in ("parent_registry_sha256", "protocol_sha256", "lineage_lock_sha256"):
        _strict_hash(registry[field], field=field)
    if expected_protocol_sha is not None:
        if _strict_hash(expected_protocol_sha, field="expected_protocol_sha256") != registry["protocol_sha256"]:
            raise ValueError("registry protocol hash mismatch")
    if expected_lock_sha is not None:
        if _strict_hash(expected_lock_sha, field="expected_lineage_lock_sha256") != registry["lineage_lock_sha256"]:
            raise ValueError("registry lineage lock hash mismatch")
    if expected_parent_registry_sha is not None:
        if (
            _strict_hash(
                expected_parent_registry_sha,
                field="expected_parent_registry_sha256",
            )
            != registry["parent_registry_sha256"]
        ):
            raise ValueError("registry parent v1 registry hash mismatch")
    if registry["extraction_environment"] != recipes.EXTRACTION_ENVIRONMENT:
        raise ValueError("registry extraction_environment differs from the frozen recipe")
    model_lock = registry["model_weight_lock"]
    if not isinstance(model_lock, Mapping) or set(model_lock) != set(recipes.BACKBONES):
        raise ValueError("model_weight_lock must contain exactly the frozen backbones")
    model_lock_keys = {
        "backbone_id",
        "provider",
        "provider_version",
        "model_id",
        "snapshot_revision",
        "weights_sha256",
        "snapshot_sha256",
        "recipe_sha256",
        "embedding_dimension",
        "dtype",
    }
    for backbone in recipes.BACKBONES:
        locked = model_lock[backbone]
        if not isinstance(locked, Mapping) or set(locked) != model_lock_keys:
            raise ValueError("model_weight_lock record has unknown or missing keys")
        if locked["backbone_id"] != backbone or locked["dtype"] != "float32":
            raise ValueError("model_weight_lock backbone/dtype mismatch")
        if type(locked["embedding_dimension"]) is not int or locked["embedding_dimension"] <= 0:
            raise ValueError("model_weight_lock embedding dimension is invalid")
        for field in ("weights_sha256", "snapshot_sha256", "recipe_sha256"):
            _strict_hash(locked[field], field=field)
        for field in ("provider", "provider_version", "model_id", "snapshot_revision"):
            if not isinstance(locked[field], str) or not locked[field]:
                raise ValueError(f"model_weight_lock {field} must be nonempty")
    rows = registry["datasets"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) != len(DATASET_IDS):
        raise ValueError("registry must contain exactly five dataset records")
    if [row.get("dataset_id") if isinstance(row, Mapping) else None for row in rows] != list(DATASET_IDS):
        raise ValueError("registry dataset order/IDs do not match the frozen panel")
    validated_rows: list[dict[str, Any]] = []
    for row in rows:
        _require_exact_mapping(row, _dataset_record_keys(), field="dataset record")
        dataset_id = _validate_dataset_id(row["dataset_id"])
        if type(row["class_count"]) is not int or row["class_count"] != CLASS_COUNTS[dataset_id]:
            raise ValueError("dataset class_count mismatch")
        source = row["source"]
        source_keys = {
            "provider",
            "dataset_id",
            "raw_root",
            "manifest_path",
            "file_count",
            "total_bytes",
            "hash_recipe",
        }
        if not isinstance(source, Mapping) or set(source) != source_keys:
            raise ValueError("dataset source has unknown or missing keys")
        if source["provider"] != "torchvision" or source["dataset_id"] != dataset_id:
            raise ValueError("dataset source provider/ID mismatch")
        if source["hash_recipe"] != "sorted_relative_path_size_and_file_sha256_v1":
            raise ValueError("dataset source hash recipe mismatch")
        for field in ("raw_root", "manifest_path"):
            if not isinstance(source[field], str) or not source[field]:
                raise ValueError(f"dataset source {field} must be nonempty")
        for field in ("file_count", "total_bytes"):
            if type(source[field]) is not int or source[field] <= 0:
                raise ValueError(f"dataset source {field} must be a positive strict int")
        if not isinstance(row["split_recipe"], Mapping) or dict(row["split_recipe"]) != DATASET_SPLIT_RECIPES[dataset_id]:
            raise ValueError("dataset split_recipe mismatch")
        if not isinstance(row["version"], str) or not row["version"]:
            raise ValueError("dataset version is required")
        for field in (
            "archive_sha256",
            "source_manifest_sha256",
            "sample_ids_sha256",
            "label_sha256",
        ):
            _strict_hash(row[field], field=field)
        source_manifest_path = Path(source["manifest_path"])
        lineage.verify_file_hash(
            source_manifest_path,
            row["source_manifest_sha256"],
            field="dataset source manifest",
        )
        try:
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("dataset source manifest is not readable JSON") from exc
        if not isinstance(source_manifest, Mapping):
            raise ValueError("dataset source manifest must be an object")
        expected_source_manifest_keys = {
            "schema_version",
            "dataset_id",
            "root",
            "hash_recipe",
            "file_count",
            "total_bytes",
            "archive_sha256",
            "files",
        }
        if set(source_manifest) != expected_source_manifest_keys:
            raise ValueError("dataset source manifest schema mismatch")
        if (
            source_manifest["schema_version"] != 1
            or source_manifest["dataset_id"] != dataset_id
            or source_manifest["root"] != source["raw_root"]
            or source_manifest["hash_recipe"] != source["hash_recipe"]
            or source_manifest["file_count"] != source["file_count"]
            or source_manifest["total_bytes"] != source["total_bytes"]
            or source_manifest["archive_sha256"] != row["archive_sha256"]
        ):
            raise ValueError("dataset source manifest differs from registry identity")
        cohort_manifest = row["cohort_manifest"]
        if not isinstance(cohort_manifest, Mapping):
            raise ValueError("cohort_manifest must be a mapping")
        expected_cohort_keys = {"replicates", "evaluation", "union"}
        if set(cohort_manifest) != expected_cohort_keys:
            raise ValueError("cohort_manifest has unknown or missing keys")
        replicates = cohort_manifest["replicates"]
        if not isinstance(replicates, Mapping) or set(replicates) != {str(seed) for seed in REPLICATE_SEEDS}:
            raise ValueError("cohort_manifest replicate seed set mismatch")
        union_descriptor = cohort_manifest["union"]
        if not isinstance(union_descriptor, Mapping):
            raise ValueError("cohort_manifest union must be a mapping")
        union_keys = {"dataset_id", "source_split_ids", "row_count", "sample_ids", "labels", "sample_id_sha256", "label_sha256", "memberships"}
        if set(union_descriptor) != union_keys:
            raise ValueError("cohort_manifest union has unknown or missing keys")
        union_ids = _strict_sequence(union_descriptor["sample_ids"], field="union sample_ids")
        union_labels = _strict_sequence(union_descriptor["labels"], field="union labels")
        if len(union_ids) != len(union_labels) or int(union_descriptor["row_count"]) != len(union_ids):
            raise ValueError("cohort union rows are not aligned")
        if union_descriptor["dataset_id"] != dataset_id:
            raise ValueError("cohort union dataset mismatch")
        if sequence_sha256(union_ids) != union_descriptor["sample_id_sha256"] or sequence_sha256(union_labels) != union_descriptor["label_sha256"]:
            raise ValueError("cohort union hash mismatch")
        if row["sample_ids_sha256"] != sequence_sha256(union_ids):
            raise ValueError("dataset sample_ids_sha256 does not bind the cohort union")
        if row["label_sha256"] != sequence_sha256(union_labels):
            raise ValueError("dataset label_sha256 does not bind the cohort union")
        for seed in REPLICATE_SEEDS:
            per_seed = replicates[str(seed)]
            if not isinstance(per_seed, Mapping) or set(per_seed) != {"32", "64"}:
                raise ValueError("replicate cohort budgets must be exactly 32 and 64")
            for budget in TRAINING_BUDGETS:
                _validate_cohort_descriptor(
                    per_seed[str(budget)],
                    dataset_id=dataset_id,
                    expected_type="training",
                    expected_seed=seed,
                    expected_budget=budget,
                    expected_per_class=budget,
                    union_sample_ids=union_ids,
                )
        _validate_cohort_descriptor(
            cohort_manifest["evaluation"],
            dataset_id=dataset_id,
            expected_type="evaluation",
            expected_seed=EVALUATION_SEED,
            expected_budget=None,
            expected_per_class=EVALUATION_PER_CLASS,
            union_sample_ids=union_ids,
        )
        embeddings = row["embeddings"]
        if not isinstance(embeddings, Mapping) or set(embeddings) != set(recipes.BACKBONES):
            raise ValueError("embedding backbone set does not match the frozen panel")
        validated_embeddings = {
            backbone: _validate_embedding_record(embeddings[backbone], backbone_id=backbone)
            for backbone in recipes.BACKBONES
        }
        for backbone, embedding in validated_embeddings.items():
            locked = model_lock[backbone]
            identity = embedding["model_weight_identity"]
            expected_identity = {
                "model_id": locked["model_id"],
                "snapshot_revision": locked["snapshot_revision"],
                "weights_sha256": locked["weights_sha256"],
                "snapshot_sha256": locked["snapshot_sha256"],
            }
            if identity != expected_identity:
                raise ValueError("embedding model identity differs from model_weight_lock")
            if embedding["model_weight_sha256"] != locked["weights_sha256"]:
                raise ValueError("embedding weight hash differs from model_weight_lock")
            if embedding["shape"][1] != locked["embedding_dimension"]:
                raise ValueError("embedding dimension differs from model_weight_lock")
            base_recipe = dict(embedding["extractor_recipe"])
            for field in (
                "matrix_values_sha256",
                "sample_id_sha256",
                "label_sha256",
            ):
                base_recipe.pop(field, None)
            if lineage.sha256_payload(base_recipe) != locked["recipe_sha256"]:
                raise ValueError("embedding recipe differs from model_weight_lock")
        validated_row = dict(row)
        validated_cohort = dict(cohort_manifest)
        validated_cohort["union"] = dict(union_descriptor)
        validated_cohort["replicates"] = {
            str(seed): dict(replicates[str(seed)]) for seed in REPLICATE_SEEDS
        }
        validated_row["cohort_manifest"] = validated_cohort
        validated_row["embeddings"] = validated_embeddings
        validated_rows.append(validated_row)
    output = dict(registry)
    output["datasets"] = validated_rows
    return output


def load_and_validate_registry(
    path: os.PathLike[str] | str,
    *,
    expected_protocol_sha: str,
    expected_lock_sha: str,
    expected_parent_registry_sha: str,
) -> dict[str, Any]:
    """Read the exact audited registry and fail closed before panel access."""

    candidate = Path(path)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read audited registry at {candidate}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("audited registry must be a JSON object")
    validated = _validate_registry_shape(
        value,
        expected_protocol_sha=expected_protocol_sha,
        expected_lock_sha=expected_lock_sha,
        expected_parent_registry_sha=expected_parent_registry_sha,
    )
    return validated


def _resolve_registry_path(value: str, *, registry_root: Optional[os.PathLike[str] | str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        if registry_root is None:
            raise ValueError("relative cache paths require registry_root")
        path = Path(registry_root) / path
    return path


def _hash_matrix(matrix: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(matrix, dtype=np.float32)
    return sha256_bytes(contiguous.tobytes(order="C"))


def load_panel(
    dataset_record: Mapping[str, Any],
    backbone: str,
    seed: int,
    budget: int,
    *,
    registry_root: Optional[os.PathLike[str] | str] = None,
) -> dict[str, Any]:
    """Load one panel from a validated registry without refitting or sampling."""

    if not isinstance(dataset_record, Mapping):
        raise TypeError("dataset_record must be a mapping")
    dataset_id = _validate_dataset_id(dataset_record.get("dataset_id"))
    if backbone not in recipes.BACKBONES:
        raise ValueError(f"backbone must be one of {recipes.BACKBONES!r}; got {backbone!r}")
    seed_value = _strict_seed(seed, field="replicate seed")
    if seed_value not in REPLICATE_SEEDS:
        raise ValueError("seed is outside the frozen confirmation replicates")
    if type(budget) is not int or budget not in TRAINING_BUDGETS:
        raise ValueError("budget is outside the frozen training budgets")
    # This call also rejects malformed records, but does not require a parent
    # protocol/lock expectation because the caller already validated the full
    # registry.  It protects the private seam used by focused tests.
    _require_exact_mapping(dataset_record, _dataset_record_keys(), field="dataset record")
    cohort_manifest = dataset_record["cohort_manifest"]
    union = cohort_manifest["union"]
    training = _validate_cohort_descriptor(
        cohort_manifest["replicates"][str(seed_value)][str(budget)],
        dataset_id=dataset_id,
        expected_type="training",
        expected_seed=seed_value,
        expected_budget=budget,
        expected_per_class=budget,
        union_sample_ids=union["sample_ids"],
    )
    evaluation = _validate_cohort_descriptor(
        cohort_manifest["evaluation"],
        dataset_id=dataset_id,
        expected_type="evaluation",
        expected_seed=EVALUATION_SEED,
        expected_budget=None,
        expected_per_class=EVALUATION_PER_CLASS,
        union_sample_ids=union["sample_ids"],
    )
    embedding_record = _validate_embedding_record(
        dataset_record["embeddings"][backbone], backbone_id=backbone
    )
    manifest_path = _resolve_registry_path(
        embedding_record["manifest_path"], registry_root=registry_root
    )
    matrix_path = _resolve_registry_path(
        embedding_record["matrix_path"], registry_root=registry_root
    )
    lineage.verify_file_hash(manifest_path, embedding_record["manifest_sha256"], field="embedding manifest")
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("embedding manifest is not readable JSON") from exc
    if not isinstance(manifest_value, Mapping):
        raise ValueError("embedding manifest must be a JSON object")
    # ``manifest_sha256`` is an external file identity and therefore cannot be
    # embedded in the bytes whose hash it records.  The cache manifest carries
    # the remaining canonical record; the registry carries the file hash.
    expected_manifest_keys = set(embedding_record) - {"manifest_sha256"}
    if set(manifest_value) != expected_manifest_keys:
        raise ValueError("embedding manifest schema does not match registry record")
    expected_manifest = {
        key: value for key, value in embedding_record.items() if key != "manifest_sha256"
    }
    if dict(manifest_value) != expected_manifest:
        raise ValueError("embedding manifest differs from registry embedding identity")
    if not matrix_path.is_file():
        raise FileNotFoundError(f"embedding matrix is missing: {matrix_path}")
    try:
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("embedding matrix cannot be loaded without pickle") from exc
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("embedding matrix must be two-dimensional")
    if matrix.dtype != np.dtype("float32"):
        raise ValueError("embedding matrix dtype must be float32")
    if tuple(int(value) for value in matrix.shape) != tuple(embedding_record["shape"]):
        raise ValueError("embedding matrix shape mismatch")
    if not np.isfinite(np.asarray(matrix)).all():
        raise ValueError("embedding matrix contains non-finite values")
    if lineage.sha256_path(matrix_path) != embedding_record["matrix_sha256"]:
        raise ValueError("embedding matrix hash mismatch")
    union_ids = list(union["sample_ids"])
    id_to_row = {sample_id: position for position, sample_id in enumerate(union_ids)}
    train_rows = [id_to_row[sample_id] for sample_id in training["sample_ids"]]
    eval_rows = [id_to_row[sample_id] for sample_id in evaluation["sample_ids"]]
    training_values = np.asarray(matrix[train_rows], dtype=np.float32)
    evaluation_values = np.asarray(matrix[eval_rows], dtype=np.float32)
    return {
        "dataset_id": dataset_id,
        "backbone": backbone,
        "replicate_seed": seed_value,
        "budget": int(budget),
        "training_values": training_values,
        "training_labels": np.asarray(training["labels"]),
        "evaluation_values": evaluation_values,
        "evaluation_labels": np.asarray(evaluation["labels"]),
        "cohort_hashes": {
            "union_sample_ids_sha256": union["sample_id_sha256"],
            "union_label_sha256": union["label_sha256"],
            "training_sample_ids_sha256": training["sample_id_sha256"],
            "training_label_sha256": training["label_sha256"],
            "training_row_indices_sha256": training["row_indices_sha256"],
            "evaluation_sample_ids_sha256": evaluation["sample_id_sha256"],
            "evaluation_label_sha256": evaluation["label_sha256"],
            "evaluation_row_indices_sha256": evaluation["row_indices_sha256"],
        },
        "embedding_manifest": dict(embedding_record),
    }


__all__ = [
    "CLASS_COUNTS",
    "Cohort",
    "CohortUnion",
    "DATASET_IDS",
    "DATASET_PARTITIONS",
    "DATASET_SPLIT_RECIPES",
    "DatasetBundle",
    "EVALUATION_PER_CLASS",
    "EVALUATION_SEED",
    "EXPECTED_ROW_COUNTS",
    "REPLICATE_SEEDS",
    "SplitIdentity",
    "TRAINING_BUDGETS",
    "build_evaluation_cohort",
    "build_training_cohorts",
    "canonical_json",
    "dataset_spec",
    "dataset_specs",
    "indices_sha256",
    "load_and_validate_registry",
    "load_confirmation_dataset",
    "load_panel",
    "make_split_identity",
    "sequence_sha256",
    "sha256_bytes",
    "union_cohort_samples",
]
