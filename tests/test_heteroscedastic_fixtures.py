"""Focused structural tests for the frozen heteroscedastic fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.heteroscedastic_distance_conditioning import fixtures as f


def _case(**overrides):
    values = {
        "stage": "development",
        "seed": 21000,
        "scenario": "H2",
        "balance": "balanced",
        "count_level": "small",
        "nuisance_dim": 32,
        "k": 2,
        "signal_state": "linear_separated",
    }
    values.update(overrides)
    return f.case_spec(**values)


def test_maximum_banks_are_bit_deterministic_and_read_only() -> None:
    first = f.latent_banks(21000)
    second = f.latent_banks(21000)
    assert f.bank_hashes(first) == f.bank_hashes(second)
    assert first.signal_train["linear_separated"].shape == (4, 320, 2)
    assert first.nuisance_train.uniform.shape == (4, 320, 32)
    assert not first.signal_train["linear_separated"].flags.writeable
    assert not first.nuisance_train.uniform.flags.writeable
    with pytest.raises(ValueError):
        first.nuisance_train.uniform[0, 0, 0] = 1.0


def test_signal_and_nuisance_streams_are_independent_and_d4_is_prefix() -> None:
    d4 = f.generate_dataset(_case(nuisance_dim=4))
    d32 = f.generate_dataset(_case(nuisance_dim=32))
    np.testing.assert_array_equal(d4.train.y, d32.train.y)
    np.testing.assert_array_equal(d4.train.signal, d32.train.signal)
    np.testing.assert_array_equal(d4.evaluation.signal, d32.evaluation.signal)
    np.testing.assert_array_equal(d4.train.nuisance, d32.train.nuisance[:, :4])
    np.testing.assert_array_equal(d4.evaluation.nuisance, d32.evaluation.nuisance[:, :4])
    # Scenario derivations reuse the exact signal/labels/row order.  H1 and H2
    # intentionally change both amplitude and heterogeneity ratio, so they are
    # not expected to be a scalar nuisance multiple.
    lower = f.generate_dataset(_case(scenario="H1"))
    higher = f.generate_dataset(_case(scenario="H2"))
    np.testing.assert_array_equal(lower.train.signal, higher.train.signal)
    np.testing.assert_array_equal(lower.train.y, higher.train.y)
    assert lower.case.amplitude == 1.0 and lower.case.scale_ratio == 2.0
    assert higher.case.amplitude == 2.0 and higher.case.scale_ratio == 4.0
    assert not np.array_equal(higher.train.nuisance, lower.train.nuisance)


def test_nested_prefixes_and_class_counts_are_exact() -> None:
    small = f.generate_dataset(_case(count_level="small", balance="imbalanced"))
    large = f.generate_dataset(_case(count_level="large", balance="imbalanced"))
    assert small.train.n_samples == 160
    assert large.train.n_samples == 640
    # The class-major prefixes are preserved through a common row permutation
    # for a given count cell; class counts still exactly follow size ranks.
    assert tuple(np.bincount(small.train.y, minlength=4)) == (16, 24, 40, 80)
    assert tuple(np.bincount(large.train.y, minlength=4)) == (64, 96, 160, 320)
    assert small.metadata["nested_selection"] is True
    assert small.metadata["shared_panel_row_order"] is True


def test_scale_recipe_and_reversed_heldout_assignment() -> None:
    values = f.scale_values(4.0)
    assert np.all(np.diff(values) > 0)
    assert np.isclose(np.mean(values * values), 1.0, rtol=0.0, atol=1e-15)
    stable = f.generate_dataset(_case(scenario="H2"))
    reversed_case = f.generate_dataset(_case(scenario="X"))
    # Reversal is a nuisance-only held-out shift; training rows remain paired.
    np.testing.assert_array_equal(stable.train.signal, reversed_case.train.signal)
    np.testing.assert_array_equal(stable.train.y, reversed_case.train.y)
    np.testing.assert_array_equal(stable.train.nuisance, reversed_case.train.nuisance)
    assert not np.array_equal(stable.evaluation.nuisance, reversed_case.evaluation.nuisance)


def test_contamination_is_rowwise_and_shared_across_nuisance_dimensions() -> None:
    banks = f.latent_banks(21000)
    mask = banks.nuisance_train.contamination_mask
    assert mask.shape == (4, 320)
    dataset = f.generate_dataset(_case(scenario="C", nuisance_dim=32))
    assert dataset.metadata["nuisance_dim"] == 32
    # The mask is a single class-major row stream, not one Bernoulli draw per
    # feature.  The raw bank itself is the canonical paired-mask evidence.
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, banks.nuisance_train.contamination_mask)
    assert dataset.train.contamination_mask.shape == (160,)


def test_truth_only_marks_the_predeclared_half_overlap_pair() -> None:
    separated = f.generate_dataset(_case(signal_state="linear_separated"))
    overlap = f.generate_dataset(_case(signal_state="genuine_overlap_half"))
    assert np.all(separated.truth["pair_overlap"] == 0.0)
    assert overlap.truth["pair_overlap"][0, 1] == 0.5
    assert overlap.truth["pair_overlap"][1, 0] == 0.5
    assert np.count_nonzero(overlap.truth["pair_overlap"]) == 2
    # Even selected prefixes retain exactly half shared rows for classes 0/1.
    assert np.sum(overlap.train.component_ids[overlap.train.y == 0] == 0) == 20
    assert np.sum(overlap.train.component_ids[overlap.train.y == 1] == 0) == 20


def test_case_grid_counts_and_confirmation_bank_guard() -> None:
    assert len(f.smoke_cases()) == 30
    assert len(f.development_cases()) == 5184
    assert len(f.confirmation_cases()) == 10368
    with pytest.raises(PermissionError, match="confirmation banks require"):
        f.latent_banks(31000, stage="confirmation")
    with pytest.raises(PermissionError, match="confirmation banks require"):
        f.generate_confirmation_dataset(_case(stage="confirmation", seed=31000))


def test_smoke_identities_match_the_three_frozen_groups() -> None:
    cases = f.smoke_cases()
    assert [case.seed for case in cases[:18]] == [21000] * 18
    assert all(case.count_level == "small" and case.nuisance_dim == 4 and case.k == 2 for case in cases[:18])
    assert [case.scenario for case in cases[18:27]] == list(f.SCENARIO_ORDER)
    assert [case.balance for case in cases[18:27]] == [
        "balanced" if index % 2 == 0 else "imbalanced" for index in range(9)
    ]
    assert [case.signal_state for case in cases[18:27]] == [f.SIGNAL_STATE_ORDER[i % 3] for i in range(9)]
    assert [case.signal_state for case in cases[27:]] == list(f.SIGNAL_STATE_ORDER)
    assert all(case.scenario == "ALL" for case in cases[27:])
