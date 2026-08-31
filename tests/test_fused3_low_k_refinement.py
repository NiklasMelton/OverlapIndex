from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fused3_envelope_reanalysis import low_k_refinement as low
from experiments.fused3_envelope_reanalysis import refined_fused3 as refined


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(109)
    values = np.vstack(
        [rng.normal(loc=position * 2.0, scale=0.4, size=(20, 7)) for position in range(3)]
    ).astype(np.float32)
    labels = np.repeat(np.asarray(["a", "b", "c"], dtype=object), 20)
    return values, labels


def test_low_k_design_is_closed() -> None:
    assert low.validate_low_k_design() == {32: 3, 64: 2}
    with pytest.raises(ValueError, match="exactly"):
        low.validate_low_k_design({32: 2, 64: 2})


def test_crossfit_fixed_low_k_changes_only_capacity_and_is_deterministic() -> None:
    values, labels = _toy_data()
    first = refined.crossfit_fused3_refinement(
        values, labels, seed=143, collect_own_win_rates=True, k=2
    )
    second = refined.crossfit_fused3_refinement(
        values, labels, seed=143, collect_own_win_rates=True, k=2
    )
    assert first["scores"] == second["scores"]
    assert all(row["k_min"] == row["k_max"] == 2 for row in first["folds"])
    assert all(row["prototype_count_before"] == 6 for row in first["folds"])
    assert [row["refined_state_sha256"] for row in first["folds"]] == [
        row["refined_state_sha256"] for row in second["folds"]
    ]


def test_source_diagnostic_loader_validates_every_file(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    summary = {
        "study": low.K_SWEEP_STUDY,
        "artifact_status": "completed",
        "diagnostic_status": low.STATUS,
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "state_rows": [],
    }
    summary_bytes = (json.dumps(summary, sort_keys=True) + "\n").encode()
    (root / "summary.json").write_bytes(summary_bytes)
    manifest = {
        "study": low.K_SWEEP_STUDY,
        "artifact_status": "completed",
        "diagnostic_status": low.STATUS,
        "original_confirmation_decision_unchanged": True,
        "promotion_or_reselection_performed": False,
        "files": {
            "summary.json": {
                "sha256": hashlib.sha256(summary_bytes).hexdigest(),
                "size_bytes": len(summary_bytes),
            }
        },
    }
    (root / "diagnostic_manifest.json").write_text(json.dumps(manifest))
    observed, hashes = low._load_diagnostic_summary(
        root, expected_study=low.K_SWEEP_STUDY
    )
    assert observed == summary
    assert hashes["summary.json"] == manifest["files"]["summary.json"]["sha256"]
    (root / "summary.json").write_text("{}")
    with pytest.raises(RuntimeError, match="hash/size"):
        low._load_diagnostic_summary(root, expected_study=low.K_SWEEP_STUDY)
