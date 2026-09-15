"""Focused contracts for the representation-by-scorer diagnostic."""

from __future__ import annotations

import numpy as np

from experiments.continuous_representation_scorer_diagnostic_v1 import run


def test_protocol_closes_grid_and_exact_row_counts():
    protocol = run.load_protocol()
    assert len(run.panel_keys(protocol)) == 36
    assert tuple(protocol["ranking_methods"]) == run.METHODS
    assert protocol["execution"]["expected_selector_rows"] == 4536
    assert protocol["execution"]["expected_reference_rows"] == 864


def test_all_relations_recover_the_same_latent_measurement():
    protocol = run.load_protocol()
    for orientation in run.ORIENTATIONS:
        panel = run.generate_panel(protocol, "product_wins", orientation, 16, 2801)
        assert panel["expected_winner"] == "product_candidate"
        for candidate in panel["candidates"]:
            recovered = run.recover_signal(
                candidate["views_train"]["signal"],
                candidate["relation"],
                orientation,
                candidate["shift"],
                protocol["data"]["radial_weight"],
            )
            assert recovered.shape == (protocol["data"]["training_rows"],)
            assert np.isfinite(recovered).all()


def test_orientation_changes_only_signal_coordinates():
    protocol = run.load_protocol()
    axis = run.generate_panel(protocol, "radial_wins", "axis", 16, 3109)
    rotated = run.generate_panel(protocol, "radial_wins", "rotated_45", 16, 3109)
    np.testing.assert_array_equal(axis["y_train"], rotated["y_train"])
    for left, right in zip(axis["candidates"], rotated["candidates"]):
        assert left["candidate_id"] == right["candidate_id"]
        np.testing.assert_allclose(
            left["views_train"]["observed"][:, 2:],
            right["views_train"]["observed"][:, 2:],
        )
        np.testing.assert_allclose(
            left["views_train"]["signal"] @ run._rotation("rotated_45"),
            right["views_train"]["signal"],
        )


def test_product_rotation_requires_diagonal_quadratic_terms():
    rng = np.random.default_rng(7)
    z = rng.normal(size=200)
    sign = np.where(rng.integers(0, 2, size=200) == 0, -1.0, 1.0)
    primitive = np.column_stack([sign, sign * z])
    rotated = primitive @ run._rotation("rotated_45")
    recovered = 0.5 * (np.square(rotated[:, 0]) - np.square(rotated[:, 1]))
    np.testing.assert_allclose(recovered, z, rtol=1e-12, atol=1e-12)


def test_expanded_quadratic_screen_includes_squares_and_is_deterministic():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 12))
    y = np.square(X[:, 0]) - np.square(X[:, 1]) + 0.05 * rng.normal(size=300)
    first = run._fit_head_features("screened_quadratic_closure8", X, y)
    second = run._fit_head_features("screened_quadratic_closure8", X, y)
    np.testing.assert_array_equal(first["selected_pairs"], second["selected_pairs"])
    selected = {tuple(pair) for pair in first["selected_pairs"]}
    assert (0, 0) in selected
    assert (1, 1) in selected


def test_generated_view_shapes_and_analytic_winner():
    protocol = run.load_protocol()
    for scenario, spec in protocol["scenarios"].items():
        panel = run.generate_panel(protocol, scenario, "axis", 512, 3413)
        truth = {row["candidate_id"]: row["analytic_r2"] for row in panel["candidates"]}
        assert max(truth, key=truth.get) == spec["expected_winner"]
        for candidate in panel["candidates"]:
            assert candidate["views_train"]["observed"].shape == (1200, 512)
            assert candidate["views_train"]["signal"].shape == (1200, 2)
            assert candidate["views_train"]["lift"].shape == (1200, 5)
