from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fused3_confirmation_v2 import statistics
from experiments.fused3_envelope_reanalysis import aircraft


def _rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selectors: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    dino = "dinov2-small"
    clip = "openclip-vit-b-32"
    for dataset in statistics.DATASET_IDS:
        for backbone_index, backbone in enumerate(statistics.BACKBONES):
            for replicate, seed in enumerate(statistics.REPLICATE_SEEDS):
                for budget in statistics.BUDGETS:
                    for method in statistics.METHODS:
                        score = float(backbone_index)
                        if dataset == aircraft.DATASET_ID:
                            if method == "FUSED3":
                                score = 10.0 if backbone == clip else (
                                    5.0 if backbone == dino else 0.0
                                )
                            elif budget == 32:
                                score = 10.0 if backbone == clip else (
                                    9.0 if backbone == dino else 0.0
                                )
                            else:
                                score = 10.0 if backbone == dino else (
                                    9.0 if backbone == clip else 0.0
                                )
                        row: dict[str, object] = {
                            "dataset_id": dataset,
                            "backbone": backbone,
                            "replicate": replicate,
                            "replicate_seed": seed,
                            "budget": budget,
                            "candidate_id": method,
                            "score": score,
                            "status": "ok",
                        }
                        if dataset == aircraft.DATASET_ID and method == "FUSED3":
                            feature_count = 512 if backbone == clip else 384
                            row["folds"] = [
                                {
                                    "score": score / 100.0,
                                    "structural_diagnostics": {
                                        "n_features_fit": feature_count,
                                        "weight_condition": (
                                            2.0 if backbone == clip else 9.0
                                        ),
                                        "positive_margin_feature_count": feature_count,
                                        "margin_row_count": 3200,
                                    },
                                }
                            ]
                        selectors.append(row)

                    for head in statistics.HEADS:
                        accuracy = 0.4 + 0.001 * backbone_index
                        if dataset == aircraft.DATASET_ID and backbone == dino:
                            accuracy = {
                                "linear": 0.60,
                                "quadratic": 0.80,
                                "knn": 0.50,
                                "rbf": 0.65,
                            }[head]
                        elif dataset == aircraft.DATASET_ID and backbone == clip:
                            accuracy = {
                                "linear": 0.58,
                                "quadratic": 0.70,
                                "knn": 0.65,
                                "rbf": 0.62,
                            }[head]
                        references.append(
                            {
                                "dataset_id": dataset,
                                "backbone": backbone,
                                "replicate": replicate,
                                "replicate_seed": seed,
                                "budget": budget,
                                "head": head,
                                "test_accuracy": accuracy,
                                "status": "ok",
                            }
                        )
    return selectors, references


def test_aircraft_diagnostic_exposes_budget_switch_and_head_conflict() -> None:
    selectors, references = _rows()
    summary = aircraft.analyze_aircraft(selectors, references)
    selection = {
        (row["budget"], row["candidate_id"]): row["backbone"]
        for row in summary["selection_rows"]
    }
    assert selection[(32, "FUSED3")] == "openclip-vit-b-32"
    assert selection[(64, "FUSED3")] == "openclip-vit-b-32"
    assert selection[(32, "LP-FULL")] == "openclip-vit-b-32"
    assert selection[(64, "LP-FULL")] == "dinov2-small"

    focal = {row["budget"]: row for row in summary["focal_contrasts"]}
    assert focal[32]["FUSED3_score_delta"] > 0.0
    assert focal[64]["LP_FULL_score_delta"] < 0.0
    assert focal[64]["knn_accuracy_delta_pp"] > 0.0
    assert focal[64]["quadratic_accuracy_delta_pp"] < 0.0
    assert focal[64]["envelope_accuracy_delta_pp"] < 0.0
    assert summary["original_confirmation_decision_unchanged"] is True
    assert summary["promotion_or_reselection_performed"] is False

    structure = {
        (row["backbone"], row["budget"]): row
        for row in summary["fused3_structural_diagnostics"]
    }
    assert structure[("dinov2-small", 64)]["mean_weight_condition"] == 9.0
    assert structure[("openclip-vit-b-32", 64)]["mean_weight_condition"] == 2.0


def test_writer_is_hash_bound_and_nonoverwriting(
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
        aircraft.frozen_analysis,
        "verify_completed_artifact",
        lambda _path: loaded,
    )
    output = tmp_path / "aircraft"
    summary = aircraft.write_diagnostic_bundle(tmp_path / "source", output)
    assert summary["diagnostic_status"] == "descriptive_only_no_promotion"
    assert "decision.json" not in {path.name for path in output.iterdir()}
    manifest = json.loads((output / "diagnostic_manifest.json").read_text())
    for name, descriptor in manifest["files"].items():
        assert descriptor["sha256"] == hashlib.sha256(
            (output / name).read_bytes()
        ).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        aircraft.write_diagnostic_bundle(tmp_path / "source", output)
