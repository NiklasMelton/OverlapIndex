from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.fused3_confirmation_v2 import datasets
from experiments.fused3_confirmation_v2 import extractors


def _split(dataset_id: str, split: str, *, rows_per_class: int) -> datasets.SplitIdentity:
    class_count = datasets.CLASS_COUNTS[dataset_id]
    labels = tuple(
        label for label in range(class_count) for _ in range(rows_per_class)
    )
    sample_ids = tuple(f"{dataset_id}/{split}/{index:06d}" for index in range(len(labels)))
    return datasets.make_split_identity(
        dataset_id, split, sample_ids, labels, class_count=class_count
    )


def test_exact_five_dataset_specs_and_frozen_splits() -> None:
    assert datasets.DATASET_IDS == (
        "torchvision_cifar10",
        "torchvision_stl10",
        "torchvision_gtsrb",
        "torchvision_fgvc_aircraft",
        "torchvision_dtd",
    )
    assert datasets.DATASET_SPLIT_RECIPES["torchvision_fgvc_aircraft"] == {
        "training": "trainval",
        "evaluation": "test",
    }
    assert datasets.DATASET_SPLIT_RECIPES["torchvision_dtd"] == {
        "training": "train_plus_val_partition_1",
        "evaluation": "test_partition_1",
    }
    specs = datasets.dataset_specs(
        {dataset_id: Path("/raw") / dataset_id for dataset_id in datasets.DATASET_IDS}
    )
    assert tuple(spec["dataset_id"] for spec in specs) == datasets.DATASET_IDS
    assert all(set(spec) == {
        "dataset_id", "provider", "provider_version", "split_recipe", "partition",
        "class_count", "expected_row_count", "download_required", "raw_root",
        "raw_manifest_sha256", "sample_id_sha256", "label_sha256", "license_or_terms",
    } for spec in specs)
    with pytest.raises(ValueError, match="exactly"):
        datasets.dataset_specs({"torchvision_cifar10": "/raw/cifar"})


def test_gtsrb_sample_rows_expose_stable_labels_without_loading_images() -> None:
    class GTSRBLike:
        _samples = (
            ("/raw/00000.ppm", 0),
            ("/raw/00001.ppm", 2),
            ("/raw/00002.ppm", 1),
        )

    assert datasets._dataset_targets(GTSRBLike()) == [0, 2, 1]


def test_training_cohorts_are_nested_and_seed_deterministic() -> None:
    split = _split("torchvision_cifar10", "train", rows_per_class=64)
    first = datasets.build_training_cohorts(split)
    second = datasets.build_training_cohorts(split)
    assert len(first) == 10
    assert [cohort.to_manifest() for cohort in first] == [
        cohort.to_manifest() for cohort in second
    ]
    for offset, seed in enumerate(datasets.REPLICATE_SEEDS):
        small = first[offset * 2]
        large = first[offset * 2 + 1]
        assert small.seed == large.seed == seed
        assert small.budget == 32
        assert large.budget == 64
        assert set(small.sample_ids).issubset(set(large.sample_ids))
        assert small.row_count == 32 * datasets.CLASS_COUNTS["torchvision_cifar10"]
        assert large.row_count == 64 * datasets.CLASS_COUNTS["torchvision_cifar10"]
    with pytest.raises(ValueError, match="frozen"):
        datasets.build_training_cohorts(split, budgets=(16, 32))


def test_fixed_evaluation_cohort_requires_predeclared_seed_and_is_disjoint() -> None:
    train = _split("torchvision_cifar10", "train", rows_per_class=64)
    evaluation = _split("torchvision_cifar10", "test", rows_per_class=20)
    train_cohort = datasets.build_training_cohorts(train)[0]
    eval_cohort = datasets.build_evaluation_cohort(
        evaluation, seed=datasets.EVALUATION_SEED
    )
    assert eval_cohort.row_count == 20 * datasets.CLASS_COUNTS["torchvision_cifar10"]
    assert set(train_cohort.sample_ids).isdisjoint(eval_cohort.sample_ids)
    with pytest.raises(ValueError, match="predeclared"):
        datasets.build_evaluation_cohort(evaluation, seed=7)
    with pytest.raises(TypeError, match="strict int"):
        datasets.build_evaluation_cohort(evaluation, seed=True)


def test_union_is_deduplicated_before_extraction_and_preserves_labels() -> None:
    train = _split("torchvision_cifar10", "train", rows_per_class=64)
    evaluation = _split("torchvision_cifar10", "test", rows_per_class=20)
    training = datasets.build_training_cohorts(train)
    fixed_eval = datasets.build_evaluation_cohort(
        evaluation, seed=datasets.EVALUATION_SEED
    )
    union = datasets.union_cohort_samples(
        tuple(training) + (fixed_eval,),
        source_split_ids=(train.split, evaluation.split),
    )
    expected_ids = set().union(*(set(cohort.sample_ids) for cohort in training), set(fixed_eval.sample_ids))
    assert union.row_count == len(expected_ids)
    assert set(union.sample_ids) == expected_ids
    assert len(union.sample_ids) == len(set(union.sample_ids))
    assert union.sample_id_sha256 == datasets.sequence_sha256(union.sample_ids)
    assert union.label_sha256 == datasets.sequence_sha256(union.labels)
    assert all(memberships for memberships in union.memberships)


def test_float32_cache_manifest_is_atomic_and_resume_validated(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    preprocessor = tmp_path / "preprocessor.json"
    preprocessor.write_text('{"size":224}\n', encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model":"toy"}\n', encoding="utf-8")
    spec = extractors.resolve_backbone_spec(
        "dinov2-small",
        checkpoint_path=checkpoint,
        preprocessor_paths=(preprocessor,),
        snapshot_path=snapshot,
    )
    sample_ids = ("torchvision_cifar10/train/000000", "torchvision_cifar10/train/000001")
    labels = (0, 0)
    matrix = np.arange(768, dtype="float32").reshape(2, 384)
    path = tmp_path / "cache.npy"
    manifest = tmp_path / "cache.json"
    record = extractors.write_embedding_cache(
        path,
        matrix,
        manifest_path=manifest,
        dataset_id="torchvision_cifar10",
        sample_ids=sample_ids,
        labels=labels,
        backbone_spec=spec,
    )
    assert record["dtype"] == "float32"
    assert len(record["manifest_sha256"]) == 64
    loaded, resumed = extractors.load_embedding_cache(
        path,
        manifest_path=manifest,
        dataset_id="torchvision_cifar10",
        sample_ids=sample_ids,
        labels=labels,
        backbone_spec=spec,
    )
    assert loaded.dtype == np.dtype("float32")
    assert resumed["manifest_sha256"] == record["manifest_sha256"]
    with pytest.raises(ValueError, match="sample ID"):
        extractors.load_embedding_cache(
            path,
            manifest_path=manifest,
            dataset_id="torchvision_cifar10",
            sample_ids=(sample_ids[0], "changed"),
            labels=labels,
            backbone_spec=spec,
        )
    with pytest.raises(FileExistsError, match="exists"):
        extractors.write_embedding_cache(
            path,
            matrix,
            manifest_path=manifest,
            dataset_id="torchvision_cifar10",
            sample_ids=sample_ids,
            labels=labels,
            backbone_spec=spec,
        )


def test_prepare_inputs_requires_audited_source_metadata_before_provider_access(
    tmp_path: Path,
) -> None:
    valid_hash = "0" * 64
    with pytest.raises(ValueError, match="source_metadata"):
        extractors.prepare_inputs(
            raw_roots={dataset_id: tmp_path for dataset_id in datasets.DATASET_IDS},
            output_dir=tmp_path / "out",
            registry_path=tmp_path / "registry.json",
            protocol_sha256=valid_hash,
            lineage_lock_sha256=valid_hash,
            parent_registry_sha256=valid_hash,
            extraction_environment={},
            model_weight_lock={backbone: {} for backbone in extractors.FOOD_BACKBONE_IDS},
            backbone_specs={
                backbone: extractors.backbone_spec(backbone)
                for backbone in extractors.FOOD_BACKBONE_IDS
            },
        )


def test_source_manifests_hash_every_raw_byte_and_detect_mutation(tmp_path: Path) -> None:
    roots = {}
    for position, dataset_id in enumerate(datasets.DATASET_IDS):
        root = tmp_path / "raw" / dataset_id
        root.mkdir(parents=True)
        (root / "payload.bin").write_bytes(f"dataset-{position}".encode())
        roots[dataset_id] = root
    manifest_dir = tmp_path / "manifests"
    first = extractors.source_metadata_from_roots(roots, manifest_dir=manifest_dir)
    second = extractors.source_metadata_from_roots(roots, manifest_dir=manifest_dir)
    assert first == second
    for dataset_id in datasets.DATASET_IDS:
        source = first[dataset_id]["source"]
        assert source["file_count"] == 1
        assert Path(source["manifest_path"]).is_file()
    (roots[datasets.DATASET_IDS[0]] / "payload.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        extractors.source_metadata_from_roots(roots, manifest_dir=manifest_dir)


def test_model_weight_lock_is_closed_and_derived_from_resolved_specs(tmp_path: Path) -> None:
    specs = {}
    for backbone in extractors.FOOD_BACKBONE_IDS:
        root = tmp_path / backbone
        root.mkdir()
        checkpoint = root / "weights.bin"
        checkpoint.write_bytes(backbone.encode())
        preprocessor = root / "preprocessor.json"
        preprocessor.write_text("{}\n", encoding="utf-8")
        specs[backbone] = extractors.resolve_backbone_spec(
            backbone,
            checkpoint_path=checkpoint,
            preprocessor_paths=(preprocessor,),
            snapshot_path=root,
        )
    locked = extractors.model_weight_lock_from_specs(specs)
    assert tuple(locked) == extractors.FOOD_BACKBONE_IDS
    assert all(record["weights_sha256"] for record in locked.values())
    changed = dict(specs)
    changed.pop(extractors.FOOD_BACKBONE_IDS[-1])
    with pytest.raises(ValueError, match="exactly"):
        extractors.model_weight_lock_from_specs(changed)


def test_code_owned_union_image_loader_preserves_id_and_label_alignment() -> None:
    class ToyDataset:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, index):
            return self.rows[index]

        def __len__(self):
            return len(self.rows)

    training = datasets.SplitIdentity(
        "torchvision_cifar10", "train", ("train/0", "train/1"), (0, 1), 2
    )
    evaluation = datasets.SplitIdentity(
        "torchvision_cifar10", "test", ("test/0",), (0,), 2
    )
    bundle = datasets.DatasetBundle(
        "torchvision_cifar10",
        "0.23.0",
        ToyDataset((("image-0", 0), ("image-1", 1))),
        ToyDataset((("image-2", 0),)),
        training,
        evaluation,
        {},
    )
    union = datasets.CohortUnion(
        "torchvision_cifar10",
        ("train", "test"),
        ("test/0", "train/1"),
        (0, 1),
        (("evaluation",), ("training",)),
    )
    assert extractors.load_union_images(bundle, union) == ["image-2", "image-1"]
    wrong = datasets.CohortUnion(
        union.dataset_id,
        union.source_split_ids,
        union.sample_ids,
        (1, 1),
        union.memberships,
    )
    with pytest.raises(ValueError, match="label"):
        extractors.load_union_images(bundle, wrong)
