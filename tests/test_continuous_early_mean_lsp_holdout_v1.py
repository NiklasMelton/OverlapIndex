from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.metrics import r2_score

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_early_mean_lsp_holdout_v1 import prepare, run


def test_protocol_hash_and_locked_contract():
    protocol = prepare.load_protocol()
    assert hashlib.sha256(prepare.PROTOCOL_PATH.read_bytes()).hexdigest() == prepare.PROTOCOL_SHA256_PATH.read_text().strip()
    assert protocol["primary_view"] == "early_mean"
    assert protocol["continuous_ranking"]["locked_primary_method"] == "broad_rff_full"
    assert protocol["expected_grid"]["candidate_cell_count"] == 150
    assert protocol["expected_grid"]["ranking_row_count"] == 150


def test_panel_seed_is_deterministic_and_safe_for_inherited_arithmetic():
    seed = run._panel_seed("panel", 2026091401, "early_mean")
    assert seed == run._panel_seed("panel", 2026091401, "early_mean")
    assert 0 < seed <= run.MAX_PANEL_SEED
    assert shared.null_parent._candidate_fit_seed(seed, 4, 3) < 2**32
    assert shared.null_parent._common_bank_seed(seed, 3) < 2**32


def test_fresh_evaluation_selection_is_balanced_deterministic_and_excludes_used():
    records = [
        {
            "sample_id": f"lsp/{index:04d}",
            "source_index": index,
            "source_split": "evaluation",
            "targets": {prepare.TARGET_ID: float(index)},
        }
        for index in range(120)
    ]
    used = {"lsp/0000", "lsp/0001", "lsp/0119"}
    first, available = prepare._select_evaluation(records, used, count=50, bins=10, seed=17)
    second, repeated_available = prepare._select_evaluation(records, used, count=50, bins=10, seed=17)
    assert first == second
    assert available == repeated_available == 117
    assert len(first) == len({row["sample_id"] for row in first}) == 50
    assert not used.intersection(row["sample_id"] for row in first)
    assert {row["target_bin"] for row in first} == set(range(10))
    assert all(sum(row["target_bin"] == target_bin for row in first) == 5 for target_bin in range(10))


def test_draw_selection_is_balanced_deterministic_and_disjoint():
    protocol = json.loads(prepare.PROTOCOL_PATH.read_text())
    training = [
        {
            "sample_id": f"train/{target_bin}/{position}",
            "split": "training",
            "target_bin": target_bin,
            "matrix_index": target_bin * 59 + position,
        }
        for target_bin in range(10) for position in range(59)
    ]
    evaluation = [
        {"sample_id": f"eval/{position}", "split": "evaluation", "matrix_index": 590 + position}
        for position in range(300)
    ]
    cohort = {"rows": training + evaluation}
    first, heldout = run._draw_rows(protocol, cohort, 101)
    repeated, _ = run._draw_rows(protocol, cohort, 101)
    other, _ = run._draw_rows(protocol, cohort, 102)
    assert first == repeated
    assert [row["sample_id"] for row in first] != [row["sample_id"] for row in other]
    assert len(first) == 300 and len(heldout) == 300
    assert all(sum(row["target_bin"] == target_bin for row in first) == 30 for target_bin in range(10))
    assert not {row["sample_id"] for row in first}.intersection(row["sample_id"] for row in heldout)


def test_quadratic_kernel_ridge_recovers_quadratic_signal():
    generator = np.random.default_rng(19)
    train = generator.normal(size=(120, 4))
    test = generator.normal(size=(80, 4))
    y_train = train[:, 0] ** 2 - 0.7 * train[:, 1] * train[:, 2]
    y_test = test[:, 0] ** 2 - 0.7 * test[:, 1] * test[:, 2]
    train_kernel, test_kernel = run._quadratic_kernels(train, test)
    prediction = run._kernel_ridge_predictions(train_kernel, test_kernel, y_train, [1.0e-6])[0]
    assert r2_score(y_test, prediction) > 0.98


def test_prototype_free_broad_score_matches_shared_path_exactly():
    generator = np.random.default_rng(23)
    X = generator.normal(size=(128, 12)).astype(np.float32)
    y = (0.7 * X[:, 0] - 0.4 * X[:, 1] ** 2 + generator.normal(scale=0.05, size=128)).astype(np.float64)
    primitive = np.zeros((len(X), 2), dtype=np.float32)
    broad = run.evaluate_broad_only(X, y, panel_seed=5101, candidate_position=0)
    combined = shared.evaluate_shared_pair(X, primitive, y, panel_seed=5101, candidate_position=0)
    assert broad["fitted_selector_count"] == 0
    assert abs(broad["score"] - combined["broad_rff"]["score"]) <= 1.0e-12


def test_summary_applies_locked_primary_gate_without_reselection():
    protocol = json.loads(prepare.PROTOCOL_PATH.read_text())
    truth = [0.5, 0.4, 0.3, 0.2, 0.1]
    linear = [0.4, 0.5, 0.3, 0.2, 0.1]
    cells = []
    for draw_seed in protocol["target"]["draw_seeds"]:
        for view in protocol["views"]:
            for position, backbone in enumerate(row["backbone_id"] for row in protocol["backbones"]):
                cells.append(
                    {
                        "cell_id": f"{draw_seed}::{view}::{backbone}",
                        "draw_seed": draw_seed,
                        "view_id": view,
                        "candidate_id": backbone,
                        "truth": {
                            "heldout_best_head_envelope_r2": truth[position],
                            "heldout_linear_r2": linear[position],
                        },
                        "scores": {
                            "linear_cv_full": linear[position],
                            "raw_cv_max_full": truth[position],
                            "observed_oi_full": truth[position],
                            "broad_rff_full": truth[position],
                            "save_rff_full": truth[position],
                        },
                        "runtime_seconds": {
                            "linear_cv_full": 1.0,
                            "raw_cv_max_full": 4.0,
                            "observed_oi_full": 1.0,
                            "broad_rff_full": 1.0,
                            "save_rff_full": 1.0,
                        },
                        "heads": {"heads": {head: {"parameter_rows": []} for head in run.HEADS}},
                    }
                )
    summary = run.summarize(protocol, cells)
    assert summary["reference_opportunity"] == {
        "material_rank_conflict_draws": 5,
        "material_nonlinear_gain_draws": 5,
        "pass": True,
    }
    assert summary["decision"]["locked_primary_method"] == "broad_rff_full"
    assert summary["decision"]["primary_gate_pass"] is True
    assert summary["overlapindex_fit_count"] == 1200
