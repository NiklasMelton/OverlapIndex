from __future__ import annotations

import numpy as np
import pytest

from experiments.continuous_real_pet_area_panel import extract, prepare, run


def test_foreground_fraction_has_closed_trimap_semantics() -> None:
    mask = np.asarray([[1, 2], [3, 2]], dtype=np.uint8)
    assert prepare.foreground_fraction(mask) == (2, 4, 0.5)
    with pytest.raises(ValueError):
        prepare.foreground_fraction(np.asarray([1, 2, 3]))
    with pytest.raises(ValueError):
        prepare.foreground_fraction(np.asarray([[1, 4]], dtype=np.uint8))


def test_protocol_roster_and_input_lineage_match_code() -> None:
    protocol = run._load_protocol()
    assert tuple(protocol["backbones"]) == extract.BACKBONE_IDS
    assert tuple(protocol["replicate_seeds"]) == prepare.REPLICATE_SEEDS
    assert protocol["dataset"]["target_pixel_rule"].startswith("mask_value != 2")
    assert protocol["continuous_oi"]["shortlist_depth"] == 5


def test_prepared_cohorts_are_exact_balanced_and_disjoint() -> None:
    cohort = prepare.load_cohort()
    assert len(cohort["rows"]) == 1200
    ids = [row["sample_id"] for row in cohort["rows"]]
    assert len(ids) == len(set(ids))
    evaluation = [row for row in cohort["rows"] if row["split"] == "evaluation"]
    assert len(evaluation) == 200
    assert {value: sum(row["target_bin"] == value for row in evaluation) for value in range(10)} == {value: 20 for value in range(10)}
    for seed in prepare.REPLICATE_SEEDS:
        selected = [row for row in cohort["rows"] if row["replicate_seed"] == seed]
        assert len(selected) == 200
        assert {value: sum(row["target_bin"] == value for row in selected) for value in range(10)} == {value: 20 for value in range(10)}


def test_training_only_pca_is_finite_and_deterministic() -> None:
    rng = np.random.default_rng(17)
    train = rng.normal(size=(80, 48)).astype(np.float32)
    test = rng.normal(size=(20, 48)).astype(np.float32)
    left_train, left_test, left_diagnostics = run.parent._pca_geometry(train, test, 71)
    right_train, right_test, right_diagnostics = run.parent._pca_geometry(train, test, 71)
    np.testing.assert_array_equal(left_train, right_train)
    np.testing.assert_array_equal(left_test, right_test)
    assert left_train.shape == (80, 32)
    assert left_test.shape == (20, 32)
    assert left_diagnostics == right_diagnostics
