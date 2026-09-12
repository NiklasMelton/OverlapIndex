"""Prepare nested real-data cohorts for the reliability/recoverability diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image

from experiments.continuous_coco_pose_opportunity_replication import prepare as coco_prepare
from experiments.continuous_fresh_structured_opportunity_discovery import prepare as fresh_prepare
from experiments.continuous_large_structured_opportunity_benchmark import prepare as source_prepare
from experiments.continuous_real_pet_area_panel import prepare as pet_prepare


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_ranking_reliability_recoverability" / "inputs"

TARGETS_BY_DATASET = {
    source_prepare.CUB_ID: (
        "leg_span_over_bbox_diagonal",
        "wing_axis_cosine",
    ),
    source_prepare.LSP_ID: (
        "ankle_span_over_torso_length",
        "wrist_span_over_torso_length",
    ),
    source_prepare.COCO_ID: ("wrist_span_over_torso_length",),
    source_prepare.PET_ID: ("foreground_compactness",),
    source_prepare.VOC_ID: (
        "largest_box_area_fraction",
        "largest_box_log_aspect_ratio",
    ),
}
DATASETS = tuple(TARGETS_BY_DATASET)
TARGET_KEYS = tuple(
    f"{dataset_id}::{target_id}"
    for dataset_id, targets in TARGETS_BY_DATASET.items()
    for target_id in targets
)
MAX_TRAIN_BY_TARGET = {
    f"{source_prepare.CUB_ID}::leg_span_over_bbox_diagonal": 1200,
    f"{source_prepare.CUB_ID}::wing_axis_cosine": 900,
    f"{source_prepare.LSP_ID}::ankle_span_over_torso_length": 600,
    f"{source_prepare.LSP_ID}::wrist_span_over_torso_length": 590,
    f"{source_prepare.COCO_ID}::wrist_span_over_torso_length": 600,
    f"{source_prepare.PET_ID}::foreground_compactness": 600,
    f"{source_prepare.VOC_ID}::largest_box_area_fraction": 1200,
    f"{source_prepare.VOC_ID}::largest_box_log_aspect_ratio": 1200,
}
EVALUATION_ROWS = 200
TARGET_BINS = 10
SELECTION_SEED = 2026092101


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("ranking reliability/recoverability protocol SHA-256 mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_continuous_ranking_reliability_recoverability_before_outcomes":
        raise ValueError("ranking reliability/recoverability protocol is not frozen")
    frozen_targets = tuple(
        protocol["targets"][group][position]
        for group in ("recoverable_lp_dominated_control", "rank_changing_nonlinear_opportunity")
        for position in range(len(protocol["targets"][group]))
    )
    if set(frozen_targets) != set(TARGET_KEYS) or len(frozen_targets) != len(TARGET_KEYS):
        raise ValueError("ranking reliability/recoverability target roster changed")
    cohort = protocol["cohort"]
    if (
        int(cohort["selection_seed"]) != SELECTION_SEED
        or int(cohort["target_bins"]) != TARGET_BINS
        or int(cohort["evaluation_rows"]) != EVALUATION_ROWS
        or {key: int(value) for key, value in cohort["maximum_training_rows_by_target"].items()}
        != MAX_TRAIN_BY_TARGET
    ):
        raise ValueError("ranking reliability/recoverability cohort contract changed")
    return protocol


def _order_key(dataset_id: str, target_id: str, split: str, row: Mapping[str, Any]) -> tuple[str, str]:
    sample_id = str(row["sample_id"])
    payload = f"{SELECTION_SEED}|{dataset_id}|{target_id}|{split}|{sample_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), sample_id


def select_nested_target(
    dataset_id: str,
    target_id: str,
    records: Sequence[Mapping[str, Any]],
    maximum_training_rows: int,
    evaluation_rows: int = EVALUATION_ROWS,
) -> list[dict[str, Any]]:
    """Select quantile-balanced rows whose training order has balanced prefixes."""
    if maximum_training_rows % TARGET_BINS or evaluation_rows % TARGET_BINS:
        raise ValueError("nested cohort sizes must divide the target-bin count")
    output: list[dict[str, Any]] = []
    for split, count in (("training", maximum_training_rows), ("evaluation", evaluation_rows)):
        eligible = [
            row
            for row in records
            if str(row.get("source_split")) == split
            and target_id in row.get("targets", {})
            and math.isfinite(float(row["targets"][target_id]))
        ]
        eligible.sort(key=lambda row: (float(row["targets"][target_id]), str(row["sample_id"])))
        if len(eligible) < count:
            raise ValueError(
                f"{dataset_id}/{target_id}/{split} has {len(eligible)} eligible rows; {count} required"
            )
        bins: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for position, row in enumerate(eligible):
            bins[min(TARGET_BINS - 1, position * TARGET_BINS // len(eligible))].append(row)
        per_bin = count // TARGET_BINS
        selected_by_bin = []
        for bin_id in range(TARGET_BINS):
            ordered = sorted(
                bins[bin_id],
                key=lambda row: _order_key(dataset_id, target_id, split, row),
            )
            if len(ordered) < per_bin:
                raise ValueError(f"{dataset_id}/{target_id}/{split}/bin-{bin_id} is too small")
            selected_by_bin.append(ordered[:per_bin])
        # Interleaving bins makes every prefix divisible by ten exactly balanced.
        selected = [selected_by_bin[bin_id][rank] for rank in range(per_bin) for bin_id in range(TARGET_BINS)]
        for position, row in enumerate(selected):
            output.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "source_index": int(row["source_index"]),
                    "split": split,
                    "budget_rank": position if split == "training" else None,
                    "target_bin": position % TARGET_BINS,
                    "target": float(row["targets"][target_id]),
                }
            )
    if len({row["sample_id"] for row in output}) != len(output):
        raise ValueError(f"{dataset_id}/{target_id} training and evaluation cohorts overlap")
    return output


def _source_records() -> dict[str, list[dict[str, Any]]]:
    # Parent loaders verify the exact prepared image cohorts before their row
    # identities are used below.
    fresh_prepare.load_cohort(fresh_prepare.DEFAULT_INPUT_ROOT)
    coco_prepare.load_cohort(coco_prepare.DEFAULT_INPUT_ROOT)
    pet_prepare.load_cohort(pet_prepare.DEFAULT_INPUT_ROOT)
    return {
        source_prepare.CUB_ID: source_prepare.cub_records(),
        source_prepare.LSP_ID: source_prepare.lsp_records(),
        source_prepare.COCO_ID: source_prepare.coco_records(),
        source_prepare.PET_ID: source_prepare.pet_records(),
        source_prepare.VOC_ID: source_prepare.voc_records(),
    }


def _image_sources() -> dict[str, dict[str, Path]]:
    fresh = fresh_prepare.load_cohort(fresh_prepare.DEFAULT_INPUT_ROOT)
    coco = coco_prepare.load_cohort(coco_prepare.DEFAULT_INPUT_ROOT)
    pet = pet_prepare.load_cohort(pet_prepare.DEFAULT_INPUT_ROOT)
    return {
        source_prepare.LSP_ID: {
            str(row["sample_id"]): fresh_prepare.DEFAULT_INPUT_ROOT / str(row["image_path"])
            for row in fresh["datasets"][fresh_prepare.LSP_ID]["rows"]
        },
        source_prepare.COCO_ID: {
            str(row["sample_id"]): coco_prepare.DEFAULT_INPUT_ROOT / str(row["image_path"])
            for row in coco["rows"]
        },
        source_prepare.PET_ID: {
            f"pet/{row['sample_id']}": pet_prepare.DEFAULT_INPUT_ROOT / "images" / f"{row['sample_id']}.jpg"
            for row in pet["rows"]
        },
    }


def _extract_external_images(
    output_root: Path,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    cohorts: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[str, Path]]:
    image_sources: dict[str, dict[str, Path]] = {}
    for dataset_id, archives in (
        (source_prepare.CUB_ID, (source_prepare.CUB_ARCHIVE,)),
        (source_prepare.VOC_ID, (source_prepare.VOC_TRAINVAL, source_prepare.VOC_TEST)),
    ):
        selected_ids = {
            str(row["sample_id"])
            for target_rows in cohorts[dataset_id].values()
            for row in target_rows
        }
        lookup = {str(row["sample_id"]): row for row in records[dataset_id] if str(row["sample_id"]) in selected_ids}
        if set(lookup) != selected_ids:
            raise ValueError(f"{dataset_id} selected source record is missing")
        image_root = output_root / "images" / dataset_id
        image_root.mkdir(parents=True, exist_ok=False)
        for archive in archives:
            archive_rows = {
                sample_id: row
                for sample_id, row in lookup.items()
                if (archive == source_prepare.CUB_ARCHIVE and sample_id.startswith("cub/"))
                or (archive == source_prepare.VOC_TRAINVAL and sample_id.startswith("voc/training/"))
                or (archive == source_prepare.VOC_TEST and sample_id.startswith("voc/evaluation/"))
            }
            source_prepare._extract_selected_images(archive, archive_rows, image_root)
        image_sources[dataset_id] = {
            sample_id: image_root / (sample_id.replace("/", "__") + ".jpg")
            for sample_id in selected_ids
        }
    return image_sources


def prepare_inputs(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    output_root = output_root.resolve()
    terminal = output_root / "manifest.json"
    if output_root.exists():
        raise FileExistsError("ranking reliability/recoverability prepared inputs already exist")
    records = _source_records()
    cohorts = {
        dataset_id: {
            target_id: select_nested_target(
                dataset_id,
                target_id,
                records[dataset_id],
                MAX_TRAIN_BY_TARGET[f"{dataset_id}::{target_id}"],
            )
            for target_id in TARGETS_BY_DATASET[dataset_id]
        }
        for dataset_id in DATASETS
    }
    external_sources = _extract_external_images(output_root, records, cohorts)
    image_sources = _image_sources()
    image_sources.update(external_sources)
    datasets: dict[str, Any] = {}
    for dataset_id in DATASETS:
        sample_ids = sorted(
            {
                str(row["sample_id"])
                for target_rows in cohorts[dataset_id].values()
                for row in target_rows
            }
        )
        matrix_index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
        rows = []
        for sample_id in sample_ids:
            path = image_sources[dataset_id].get(sample_id)
            if path is None or not path.is_file():
                raise FileNotFoundError(f"prepared image is missing: {dataset_id}/{sample_id}")
            with Image.open(path) as image:
                image.verify()
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(path.resolve().relative_to(ROOT)),
                    "image_sha256": sha256_path(path),
                }
            )
        target_cohorts = {
            target_id: [
                {**row, "matrix_index": matrix_index[str(row["sample_id"])]}
                for row in target_rows
            ]
            for target_id, target_rows in cohorts[dataset_id].items()
        }
        datasets[dataset_id] = {"rows": rows, "target_cohorts": target_cohorts}
    cohort = {
        "schema_version": 1,
        "status": "prepared_continuous_ranking_reliability_recoverability_inputs",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "selection_seed": SELECTION_SEED,
        "datasets": datasets,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    cohort_path = output_root / "cohort.json"
    _atomic_json(cohort_path, cohort)
    manifest = {
        "schema_version": 1,
        "status": "completed_continuous_ranking_reliability_recoverability_preparation",
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "cohort_sha256": sha256_path(cohort_path),
        "cohort_identity_sha256": sha256_payload(cohort),
        "dataset_ids": list(DATASETS),
        "target_keys": list(TARGET_KEYS),
        "dataset_row_counts": {key: len(value["rows"]) for key, value in datasets.items()},
        "source_hashes": {
            "cub_archive": sha256_path(source_prepare.CUB_ARCHIVE),
            "lsp_parent_manifest": sha256_path(fresh_prepare.DEFAULT_OUTPUT_ROOT / "manifest.json"),
            "coco_parent_manifest": sha256_path(coco_prepare.DEFAULT_OUTPUT_ROOT / "manifest.json"),
            "pet_parent_manifest": sha256_path(pet_prepare.DEFAULT_INPUT_ROOT / "manifest.json"),
            "voc_trainval": sha256_path(source_prepare.VOC_TRAINVAL),
            "voc_test": sha256_path(source_prepare.VOC_TEST),
            "prepare": sha256_path(Path(__file__).resolve()),
        },
        "predictive_outcomes_inspected": False,
        "overlapindex_fit_count": 0,
        "reserved_backbone_count": 0,
    }
    _atomic_json(terminal, manifest)
    return manifest


def load_cohort(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    output_root = output_root.resolve()
    cohort_path, manifest_path = output_root / "cohort.json", output_root / "manifest.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed_continuous_ranking_reliability_recoverability_preparation"
        or manifest.get("protocol_sha256") != sha256_path(PROTOCOL_PATH)
        or manifest.get("cohort_sha256") != sha256_path(cohort_path)
        or manifest.get("cohort_identity_sha256") != sha256_payload(cohort)
        or cohort.get("protocol_sha256") != sha256_path(PROTOCOL_PATH)
        or set(cohort.get("datasets", {})) != set(DATASETS)
        or len(cohort.get("datasets", {})) != len(DATASETS)
        or int(manifest.get("overlapindex_fit_count", -1)) != 0
        or int(manifest.get("reserved_backbone_count", -1)) != 0
    ):
        raise ValueError("ranking reliability/recoverability prepared input identity changed")
    for dataset_id, targets in TARGETS_BY_DATASET.items():
        dataset = cohort["datasets"][dataset_id]
        if tuple(dataset["target_cohorts"]) != targets:
            raise ValueError(f"prepared target roster changed: {dataset_id}")
        for target_id, rows in dataset["target_cohorts"].items():
            training = [row for row in rows if row["split"] == "training"]
            evaluation = [row for row in rows if row["split"] == "evaluation"]
            expected = MAX_TRAIN_BY_TARGET[f"{dataset_id}::{target_id}"]
            if (
                len(training) != expected
                or [int(row["budget_rank"]) for row in training] != list(range(expected))
                or len(evaluation) != EVALUATION_ROWS
                or len({row["sample_id"] for row in rows}) != len(rows)
            ):
                raise ValueError(f"prepared cohort grid changed: {dataset_id}/{target_id}")
    del protocol
    return cohort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(canonical_json(prepare_inputs(args.output_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
