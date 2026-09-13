import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.continuous_ranking_benchmark_v2 import run


def _small_protocol():
    protocol = run.load_protocol()
    protocol = copy.deepcopy(protocol)
    protocol["heads"]["linear_ridge"]["alpha"] = [1.0]
    protocol["heads"]["quadratic_ridge"]["alpha"] = [1.0]
    protocol["heads"]["knn"]["n_neighbors"] = [5]
    protocol["heads"]["rbf_svr"]["C"] = [1.0]
    protocol["heads"]["rbf_svr"]["gamma"] = ["scale"]
    protocol["heads"]["rbf_svr"]["epsilon"] = [0.05]
    return protocol


def _rows(count=1200):
    return [
        {
            "sample_id": f"training/{position:04d}",
            "split": "training",
            "budget_rank": position,
            "target_bin": position % 10,
            "target": float(position),
            "matrix_index": position,
        }
        for position in range(count)
    ]


def test_protocol_is_frozen_and_content_addressed():
    assert hashlib.sha256(run.PROTOCOL_PATH.read_bytes()).hexdigest() == run.PROTOCOL_SHA256_PATH.read_text().strip()
    protocol = run.load_protocol()
    assert protocol["scope"]["overlapindex_evaluated"] is False
    assert protocol["scope"]["method_selection_allowed"] is False
    assert protocol["expected_grid"]["real_candidate_draw_count"] == 105
    assert protocol["expected_grid"]["synthetic_panel_count"] == 15


def test_unique_draw_is_deterministic_balanced_and_has_no_duplicates():
    first = run.select_unique_draw(_rows(), pool_rows=1200, draw_rows=600, seed=17)
    second = run.select_unique_draw(list(reversed(_rows())), pool_rows=1200, draw_rows=600, seed=17)
    other = run.select_unique_draw(_rows(), pool_rows=1200, draw_rows=600, seed=19)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert [row["sample_id"] for row in first] != [row["sample_id"] for row in other]
    assert len(first) == len({row["sample_id"] for row in first}) == 600
    assert [sum(row["target_bin"] == target_bin for row in first) for target_bin in range(10)] == [60] * 10


def test_unique_draw_fails_closed_on_duplicate_pool_identity():
    rows = _rows(600)
    rows[-1] = {**rows[-1], "sample_id": rows[0]["sample_id"]}
    with pytest.raises(ValueError, match="duplicate sample identities"):
        run.select_unique_draw(rows, pool_rows=600, draw_rows=600, seed=17)


def test_corrected_evaluator_has_disjoint_folds_and_fold_local_preprocessing():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(120, 6))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.05 * rng.normal(size=120)
    result = run.evaluate_heads_corrected(
        X[:100], y[:100], run._target_bins(y[:100]),
        [f"row/{position}" for position in range(100)],
        X[100:], y[100:], split_seed=29, protocol=_small_protocol(),
    )
    assert set(result["heads"]) == set(run.HEADS)
    assert len(result["fold_audit"]) == 5
    assert all(row["identity_overlap_count"] == 0 for row in result["fold_audit"])
    assert all(row["preprocessing"]["fit_rows"] == 80 for row in result["fold_audit"])
    assert result["full_preprocessing"]["fit_rows"] == 100
    assert result["heads"]["linear_ridge"]["evaluation_r2"] > 0.9


def test_corrected_evaluator_rejects_duplicate_sample_ids_before_splitting():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(120, 4))
    y = rng.normal(size=120)
    ids = [f"row/{position}" for position in range(100)]
    ids[-1] = ids[0]
    with pytest.raises(ValueError, match="duplicate sample identities"):
        run.evaluate_heads_corrected(
            X[:100], y[:100], run._target_bins(y[:100]), ids,
            X[100:], y[100:], split_seed=37, protocol=_small_protocol(),
        )


def test_synthetic_generator_is_deterministic_aligned_and_unit_norm():
    protocol = run.load_protocol()
    first = run.synthetic_data("smooth_periodic", 41, protocol)
    second = run.synthetic_data("smooth_periodic", 41, protocol)
    assert np.array_equal(first["y"], second["y"])
    assert tuple(first["matrices"]) == run.SYNTHETIC_CANDIDATES
    assert all(matrix.shape == (5600, 2) for matrix in first["matrices"].values())
    assert all(np.allclose(np.linalg.norm(matrix, axis=1), 1.0) for matrix in first["matrices"].values())
    with pytest.raises(ValueError, match="unknown evaluator-pilot synthetic family"):
        run.synthetic_data("unknown", 41, protocol)


def test_ranking_metrics_preserve_selection_regret_and_null_correlation():
    ids = [f"candidate-{position}" for position in range(5)]
    perfect = run.ranking_metrics([0.5, 0.4, 0.3, 0.2, 0.1], [5, 4, 3, 2, 1], ids)
    reversed_row = run.ranking_metrics([0.5, 0.4, 0.3, 0.2, 0.1], [1, 2, 3, 4, 5], ids)
    constant = run.ranking_metrics([0.5, 0.4, 0.3, 0.2, 0.1], [1] * 5, ids)
    assert perfect["spearman"] == pytest.approx(1.0)
    assert perfect["top1_regret_r2"] == 0.0
    assert reversed_row["spearman"] == pytest.approx(-1.0)
    assert reversed_row["top1_regret_r2"] == pytest.approx(0.4)
    assert constant["spearman"] is None


def test_pilot_package_does_not_modify_public_sources():
    package_files = {path.name for path in Path(run.PACKAGE_DIR).glob("*.py")}
    assert package_files == {"__init__.py", "run.py"}
