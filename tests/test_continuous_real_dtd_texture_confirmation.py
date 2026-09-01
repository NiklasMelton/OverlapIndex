import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.continuous_real_dtd_texture_confirmation import prepare


def test_texture_gradient_energy_is_zero_for_constant_image():
    image = Image.fromarray(np.full((19, 31, 3), 127, dtype=np.uint8))
    assert prepare.texture_gradient_energy(image) == 0.0


def test_texture_gradient_energy_detects_alternating_texture():
    flat = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
    checker = ((np.indices((224, 224)).sum(axis=0) % 2) * 255).astype(np.uint8)
    textured = Image.fromarray(checker)
    assert prepare.texture_gradient_energy(textured) > prepare.texture_gradient_energy(flat)


def test_split_parser_is_closed():
    assert prepare._split_from_id("torchvision_dtd/train_partition-1/000007") == ("train", 7)
    with pytest.raises(ValueError):
        prepare._split_from_id("torchvision_dtd/train_partition-2/000007")


def test_load_cohort_rejects_hash_change(tmp_path: Path):
    cohort = {"status": "prepared_outcome_free_dtd_texture_target"}
    (tmp_path / "cohort.json").write_text(
        json.dumps(cohort, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed_outcome_free_dtd_texture_preparation",
                "registry_sha256": prepare.REGISTRY_SHA256,
                "cohort_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cohort hash"):
        prepare.load_cohort(tmp_path)
