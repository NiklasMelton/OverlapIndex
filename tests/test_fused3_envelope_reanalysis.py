from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fused3_envelope_reanalysis import analysis as envelope
from experiments.fused3_confirmation_v2 import statistics


def _rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selectors: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    for dataset in statistics.DATASET_IDS:
        for backbone_index, backbone in enumerate(statistics.BACKBONES):
            for replicate, replicate_seed in enumerate(statistics.REPLICATE_SEEDS):
                for budget in statistics.BUDGETS:
                    selectors.extend(
                        [
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate": replicate,
                                "replicate_seed": replicate_seed,
                                "budget": budget,
                                "candidate_id": "FUSED3",
                                "score": float(backbone_index),
                                "status": "ok",
                            },
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate": replicate,
                                "replicate_seed": replicate_seed,
                                "budget": budget,
                                "candidate_id": "LP-FULL",
                                "score": float(-backbone_index),
                                "status": "ok",
                            },
                        ]
                    )
                    accuracies = {
                        "linear": 0.55 - 0.005 * backbone_index,
                        "quadratic": 0.50 + 0.01 * backbone_index,
                        "knn": 0.40 + 0.005 * backbone_index,
                        "rbf": 0.45,
                    }
                    for head in statistics.HEADS:
                        references.append(
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate": replicate,
                                "replicate_seed": replicate_seed,
                                "budget": budget,
                                "head": head,
                                "test_accuracy": accuracies[head],
                                "status": "ok",
                            }
                        )
    return selectors, references


def test_envelope_uses_best_head_per_backbone_and_supports_iterators() -> None:
    selectors, references = _rows()
    metrics = envelope.envelope_panel_metrics(
        (row for row in selectors), (row for row in references)
    )
    assert len(metrics) == 100
    fused = next(
        row
        for row in metrics
        if row["dataset_id"] == statistics.DATASET_IDS[0]
        and row["replicate_seed"] == statistics.REPLICATE_SEEDS[0]
        and row["budget"] == 32
        and row["candidate_id"] == "FUSED3"
    )
    probe = next(
        row
        for row in metrics
        if row["dataset_id"] == statistics.DATASET_IDS[0]
        and row["replicate_seed"] == statistics.REPLICATE_SEEDS[0]
        and row["budget"] == 32
        and row["candidate_id"] == "LP-FULL"
    )
    assert fused["selected_backbones"] == [statistics.BACKBONES[-1]]
    assert fused["selected_backbone_best_heads"] == {
        statistics.BACKBONES[-1]: ["quadratic"]
    }
    assert fused["oracle_best_head_families"] == ["quadratic"]
    assert fused["regret_pp"] == 0.0
    assert np.isclose(probe["regret_pp"], 4.0)
    assert fused["spearman_status"] == "defined"
    assert 0.7 < fused["spearman"] < 0.8


def test_selector_ties_average_best_head_envelope_outcomes() -> None:
    selectors, references = _rows()
    target = next(
        row
        for row in selectors
        if row["dataset_id"] == statistics.DATASET_IDS[0]
        and row["replicate_seed"] == statistics.REPLICATE_SEEDS[0]
        and row["budget"] == 32
        and row["candidate_id"] == "FUSED3"
        and row["backbone"] == statistics.BACKBONES[-2]
    )
    target["score"] = 9.0 - 5.0e-13
    metric = next(
        row
        for row in envelope.envelope_panel_metrics(selectors, references)
        if row["dataset_id"] == statistics.DATASET_IDS[0]
        and row["replicate_seed"] == statistics.REPLICATE_SEEDS[0]
        and row["budget"] == 32
        and row["candidate_id"] == "FUSED3"
    )
    assert metric["selected_backbones"] == list(statistics.BACKBONES[-2:])
    assert np.isclose(metric["selected_envelope_accuracy"], 0.585)
    assert np.isclose(metric["regret_pp"], 0.5)
    assert metric["exact_best"] == 0.5


def test_summary_is_descriptive_complete_and_deterministic() -> None:
    selectors, references = _rows()
    first = envelope.summarize_envelope(
        (row for row in selectors),
        (row for row in references),
        n_resamples=512,
    )
    second = envelope.summarize_envelope(
        selectors, references, n_resamples=512
    )
    assert first == second
    assert first["original_confirmation_decision"] == "fail_locked_candidate"
    assert first["original_confirmation_decision_unchanged"] is True
    assert first["promotion_or_reselection_performed"] is False
    assert np.isclose(first["comparison"]["estimate"], -4.0)
    assert first["comparison"]["inferential_role"].startswith("descriptive")
    assert len(first["regret_contrasts"]) == 50
    assert len(first["rank_auc"]) == 50
    assert first["head_prevalence"]["backbone_panel_best_head_memberships"] == {
        "linear": 200,
        "quadratic": 300,
        "knn": 0,
        "rbf": 0,
    }


def test_incomplete_grid_and_constant_rank_fail_closed_or_remain_null() -> None:
    selectors, references = _rows()
    with pytest.raises(ValueError, match="exact confirmation grid"):
        envelope.envelope_panel_metrics(selectors[:-1], references)

    constant = copy.deepcopy(selectors)
    for row in constant:
        if row["candidate_id"] == "FUSED3":
            row["score"] = 1.0
    metrics = envelope.envelope_panel_metrics(constant, references)
    fused = [row for row in metrics if row["candidate_id"] == "FUSED3"]
    assert all(row["spearman"] is None for row in fused)
    auc = envelope.envelope_rank_auc_rows(metrics)
    fused_auc = [row for row in auc if row["candidate_id"] == "FUSED3"]
    assert all(row["rank_auc"] is None for row in fused_auc)
    assert all(row["supporting_only"] for row in fused_auc)


def test_writer_is_nonoverwriting_hash_bound_and_has_no_decision_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selectors, references = _rows()
    loaded = {
        "selector_rows": selectors,
        "reference_rows": references,
        "raw": {
            "protocol_sha256": "1" * 64,
            "code_identity_sha256": "2" * 64,
            "run_identity_sha256": "3" * 64,
            "audited_registry_sha256": "4" * 64,
            "lineage_lock_sha256": "5" * 64,
        },
        "input_hashes": {
            "manifest_sha256": "6" * 64,
            "raw_results_sha256": "7" * 64,
            "selector_rows_sha256": "8" * 64,
            "reference_rows_sha256": "9" * 64,
            "lineage_lock_sha256": "5" * 64,
            "audited_registry_sha256": "4" * 64,
        },
    }
    monkeypatch.setattr(
        envelope.frozen_analysis,
        "verify_completed_artifact",
        lambda _path: loaded,
    )
    output = tmp_path / "envelope"
    summary = envelope.write_reanalysis_bundle(
        tmp_path / "source", output, n_resamples=128
    )
    expected = {
        "summary.json",
        "envelope_panel_metrics.csv",
        "envelope_regret_contrasts.csv",
        "envelope_rank_auc.csv",
        "report.md",
        "reanalysis_manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert "decision.json" not in expected
    assert summary["reanalysis_status"] == "descriptive_only_no_promotion"
    manifest = json.loads((output / "reanalysis_manifest.json").read_text())
    assert manifest["original_confirmation_decision_unchanged"] is True
    for name, descriptor in manifest["files"].items():
        assert descriptor["sha256"] == hashlib.sha256(
            (output / name).read_bytes()
        ).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        envelope.write_reanalysis_bundle(tmp_path / "source", output)
