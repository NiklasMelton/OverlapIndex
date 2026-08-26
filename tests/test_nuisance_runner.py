"""Focused correctness tests for the private nuisance experiment harness."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.nuisance_conditioned_distance import fixtures
from experiments.nuisance_conditioned_distance import runner
from overlapindex import OverlapIndex


def test_archived_fixture_is_deterministic_and_has_independent_pools() -> None:
    config = fixtures.ConfirmConfig(
        family="shared_low_rank",
        overlap_severity=0.5,
        nuisance_strength=2.0,
        train_seed=1003,
        eval_seed=11003,
    )
    first = fixtures.generate_confirm_dataset(config)
    second = fixtures.generate_confirm_dataset(config)
    np.testing.assert_array_equal(first.train.X, second.train.X)
    np.testing.assert_array_equal(first.evaluation.X, second.evaluation.X)
    np.testing.assert_array_equal(first.train.y, second.train.y)
    np.testing.assert_array_equal(first.evaluation.y, second.evaluation.y)
    assert not np.shares_memory(first.train.X, first.evaluation.X)
    assert not np.shares_memory(first.train.signal, first.evaluation.signal)
    assert not np.shares_memory(first.train.nuisance, first.evaluation.nuisance)
    assert first.train.y.ndim == first.evaluation.y.ndim == 1
    assert first.train.y.dtype.kind in "iu"
    assert first.metadata["train_eval_independent"] is True
    assert first.metadata["nuisance_label_independent"] is True


def test_archived_stage2_grid_constants_are_frozen_without_running_it() -> None:
    assert fixtures.ARCHIVED_STAGE2_SEEDS == tuple(range(1000, 1020))
    assert fixtures.ARCHIVED_STAGE2_NUISANCE_STRENGTHS == (0.0, 1.0, 2.0)
    assert fixtures.ARCHIVED_STAGE2_FAMILIES == (
        "shared_low_rank",
        "clustered_multimodal",
        "heteroscedastic_multiplicative",
    )
    assert fixtures.ARCHIVED_STAGE2_CONDITIONS == (
        {
            "name": "separated_balanced_stable",
            "overlap_severity": 0.0,
            "signal_gap": 0.75,
            "balance": "balanced",
            "nuisance_shift": False,
        },
        {
            "name": "separated_balanced_shift",
            "overlap_severity": 0.0,
            "signal_gap": 0.75,
            "balance": "balanced",
            "nuisance_shift": True,
        },
        {
            "name": "overlap_half_balanced_stable",
            "overlap_severity": 0.5,
            "signal_gap": 0.75,
            "balance": "balanced",
            "nuisance_shift": False,
        },
        {
            "name": "overlap_half_balanced_shift",
            "overlap_severity": 0.5,
            "signal_gap": 0.75,
            "balance": "balanced",
            "nuisance_shift": True,
        },
        {
            "name": "overlap_quarter_imbalanced_stable",
            "overlap_severity": 0.25,
            "signal_gap": 0.75,
            "balance": "imbalanced",
            "nuisance_shift": False,
        },
        {
            "name": "overlap_full_imbalanced_stable",
            "overlap_severity": 1.0,
            "signal_gap": 0.75,
            "balance": "imbalanced",
            "nuisance_shift": False,
        },
    )
    grid = fixtures.archived_stage2_case_grid()
    assert len(grid) == 3 * 6 * 3 * 20 * 2
    assert {row["k"] for row in grid} == {2, 8}
    assert {row["seed"] for row in grid} == set(range(1000, 1020))
    assert {row["eval_seed"] - row["train_seed"] for row in grid} == {10000}


def test_mechanistic_smoke_spans_requested_geometries_and_nuisance() -> None:
    for geometry in ("linear", "rings", "xor"):
        for nuisance_kind in ("isotropic", "anisotropic", "correlated_low_rank"):
            dataset = fixtures.generate_mechanistic_dataset(
                fixtures.MechanisticConfig(
                    geometry=geometry,
                    nuisance_kind=nuisance_kind,
                    nuisance_strength=1.0,
                    overlap_severity=0.5 if geometry == "linear" else 0.0,
                )
            )
            assert dataset.train.X.shape[1] == 6
            assert dataset.evaluation.X.shape[1] == 6
            assert dataset.train.signal.shape[1] == 2
            assert dataset.train.nuisance.shape[1] == 4
            assert dataset.ground_truth["pair_overlap"].shape == (2, 2)
            assert np.isfinite(dataset.train.X).all()
            assert np.isfinite(dataset.evaluation.X).all()


def test_stage_validation_and_case_sizes_do_not_run_full() -> None:
    assert runner.validate_stage("smoke") == "smoke"
    assert runner.validate_stage("SCREEN") == "screen"
    with pytest.raises(ValueError):
        runner.validate_stage("analysis")
    assert len(runner.stage_cases("smoke")) > 0
    assert len(runner.stage_cases("screen")) == 3 * 4 * 3 * 10 * 2
    assert len(runner.stage_cases("full")) == runner.FULL_STAGE_CASE_COUNT
    full_first = runner.stage_cases("full")[0]
    assert full_first.config.n_per_class == 40
    assert full_first.config.n_classes == 4
    assert full_first.config.signal_dim == 2
    assert full_first.config.nuisance_dim == 4


@pytest.mark.parametrize("stage", ["smoke", "screen", "full"])
def test_execution_order_is_position_counterbalanced(stage: str) -> None:
    counts = {candidate: [0] * 5 for candidate in ("A", "B", "C", "D", "E")}
    for case in runner.stage_cases(stage):
        for position, candidate in enumerate(runner._execution_order(case)):
            counts[candidate][position] += 1
    for positions in counts.values():
        assert max(positions) - min(positions) <= 1


def test_provenance_starting_commit_matches_develop_merge_base() -> None:
    assert runner.EXPERIMENT_STARTING_COMMIT == "165fa344653a72dd2abc04a20387d90b891b9acc"
    assert runner.verify_starting_commit() == runner.EXPERIMENT_STARTING_COMMIT
    source_hashes = runner._source_hashes()
    for relative in (
        "experiments/nuisance_conditioned_distance/fixtures.py",
        "experiments/nuisance_conditioned_distance/runner.py",
        "experiments/nuisance_conditioned_distance/conditioning_adapter.py",
        "experiments/nuisance_conditioned_distance/protocol.json",
        "experiments/nuisance_conditioned_distance/protocol.sha256",
        "experiments/nuisance_conditioned_distance/analysis.py",
        "experiments/nuisance_conditioned_distance/reporting.py",
    ):
        assert source_hashes[relative]
    assert runner._code_identity_hash(source_hashes)


def test_code_identity_ignores_generated_output_artifacts(tmp_path) -> None:
    source_hashes = runner._source_hashes()
    before = runner._code_identity_hash(source_hashes)
    output = tmp_path / "artifacts" / "nuisance_conditioned_distance" / "smoke"
    output.mkdir(parents=True)
    (output / "raw_results.json").write_text('{"generated": true}\n', encoding="utf-8")
    (output / "checkpoint.json").write_text('{"resume": true}\n', encoding="utf-8")
    after = runner._code_identity_hash(runner._source_hashes())
    assert after == before


def test_a_and_b_use_exact_public_oi_constructor_and_fresh_kwargs() -> None:
    case = next(case for case in runner.stage_cases("smoke") if case.family == "mechanistic")
    spec_a = runner.CANDIDATE_BY_ID["A"]
    spec_b = runner.CANDIDATE_BY_ID["B"]
    first = runner._estimator(spec_a, case.k, case.seed)
    second = runner._estimator(spec_a, case.k, case.seed)
    assert isinstance(first, OverlapIndex)
    assert first.prototype_refinement is False
    assert second.kmeans_kwargs is not first.kmeans_kwargs
    assert first.kmeans_kwargs == second.kmeans_kwargs
    dataset = fixtures.generate_mechanistic_dataset(case.config)
    first.fit(dataset.train.X, dataset.train.y)
    # B is still the exact public estimator; only the explicit frozen
    # prototype_refinement flag differs.
    refined = runner._estimator(spec_b, case.k, case.seed)
    assert isinstance(refined, OverlapIndex)
    assert refined.prototype_refinement is True


def test_score_fixed_does_not_refit_fitted_backend() -> None:
    dataset = fixtures.generate_mechanistic_dataset(
        fixtures.MechanisticConfig(
            geometry="linear",
            nuisance_kind="isotropic",
            n_per_class=10,
            nuisance_dim=2,
            nuisance_strength=0.0,
            signal_gap=1.5,
        )
    )
    model = OverlapIndex(
        model_type="MiniBatchKMeans",
        kmeans_k=2,
        kmeans_kwargs=runner.oi_kwargs(2, 99)["kmeans_kwargs"],
        prototype_refinement=False,
    ).fit(dataset.train.X, dataset.train.y)
    before = np.asarray(model._model.centers).copy()

    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("score_fixed must not call backend fit_offline")

    model._model.fit_offline = fail_if_called  # type: ignore[method-assign]
    score = model.score_fixed(dataset.evaluation.X, dataset.evaluation.y)
    assert np.isfinite(score)
    np.testing.assert_array_equal(before, np.asarray(model._model.centers))


def test_run_case_has_paired_rows_and_no_selector_score_for_f(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConditionedOverlapIndex:
        def __init__(
            self,
            mode: str = "none",
            *,
            prototype_refinement: bool = False,
            conditioning_kwargs: object = None,
            overlap_index_kwargs: dict[str, object] | None = None,
        ) -> None:
            assert mode in {"global_isotropy", "pooled_diagonal", "pooled_full"}
            assert conditioning_kwargs is None
            self._inner = OverlapIndex(
                **copy.deepcopy(overlap_index_kwargs or {}),
                prototype_refinement=prototype_refinement,
            )

        def fit(self, X: np.ndarray, y: np.ndarray) -> "FakeConditionedOverlapIndex":
            self._inner.fit(X, y)
            self.index = self._inner.index
            return self

        def score_fixed(self, X: np.ndarray, y: np.ndarray) -> float:
            return self._inner.score_fixed(X, y)

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self._inner.predict(X)

        @property
        def _model(self) -> object:
            return self._inner._model

        @property
        def pairwise_index(self) -> object:
            return self._inner.pairwise_index

        @property
        def prototype_refinement_(self) -> object:
            return self._inner.prototype_refinement_

    monkeypatch.setattr(runner, "_conditioning_adapter", lambda: FakeConditionedOverlapIndex)
    case = runner.stage_cases("smoke")[-1]
    rows = runner.run_case(case)
    assert [row["candidate_id"] for row in rows] == ["A", "B", "C", "D", "E", "F"]
    assert {row["case_id"] for row in rows} == {case.case_id}
    assert all(row["execution_order"] == rows[0]["execution_order"] for row in rows)
    assert all(row["fit_split"] == "train" for row in rows)
    assert all(row["evaluation_split"] == "evaluation" for row in rows)
    assert rows[-1]["candidate_score"] is None
    assert rows[-1]["diagnostic_only"] is True
    assert rows[-1]["raw_candidate_id"] == "B"
    assert rows[-1]["conditioned_candidate_id"] == "E"
    assert rows[-1]["raw_vs_conditioned_score_disagreement"] >= 0.0
    assert "raw_vs_conditioned_score_delta" not in rows[-1]
    assert all(row["status"] == "ok" for row in rows)


def test_runner_does_not_call_score_refit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConditionedOverlapIndex:
        def __init__(self, mode: str = "none", *, prototype_refinement: bool = False, conditioning_kwargs: object = None, overlap_index_kwargs: dict[str, object] | None = None) -> None:
            self._inner = OverlapIndex(**copy.deepcopy(overlap_index_kwargs or {}), prototype_refinement=prototype_refinement)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "FakeConditionedOverlapIndex":
            self._inner.fit(X, y)
            self.index = self._inner.index
            return self

        def score_fixed(self, X: np.ndarray, y: np.ndarray) -> float:
            return self._inner.score_fixed(X, y)

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self._inner.predict(X)

        @property
        def _model(self) -> object:
            return self._inner._model

        @property
        def pairwise_index(self) -> object:
            return self._inner.pairwise_index

        @property
        def prototype_refinement_(self) -> object:
            return self._inner.prototype_refinement_

    monkeypatch.setattr(runner, "_conditioning_adapter", lambda: FakeConditionedOverlapIndex)
    original_score = OverlapIndex.score

    def fail_score(self: OverlapIndex, *args: object, **kwargs: object) -> float:
        raise AssertionError("runner must use fit-time index and score_fixed, not score")

    monkeypatch.setattr(OverlapIndex, "score", fail_score)
    rows = runner.run_case(runner.stage_cases("smoke")[-1])
    assert any(row["status"] == "ok" for row in rows)
    assert original_score is not OverlapIndex.score


def test_conditioned_prediction_cannot_fall_through_to_inner_raw_estimator() -> None:
    class InnerOnly:
        estimator_ = object()

    with pytest.raises(RuntimeError, match="public selector geometry"):
        runner._predict(InnerOnly(), np.zeros((1, 1)))


def test_determinism_verification_excludes_only_runtime_fields() -> None:
    first: list[dict[str, object]] = []
    second: list[dict[str, object]] = []
    for candidate_id in ("A", "B", "C", "D", "E"):
        structural = {
            "candidate_id": candidate_id,
            "status": "ok",
            "fit_score": 0.8,
            "candidate_score": 0.75,
            "fit_state": {"centers_sha256": f"fit-{candidate_id}"},
            "score_fixed_state": {"centers_sha256": f"fit-{candidate_id}"},
            "pairwise": {"0->1": {"pairwise_index": 0.9}},
            "refinement": {"applied_count": 1},
            "refinement_after_score_fixed": {"applied_count": 1},
            "conditioning": {"state_sha256": f"state-{candidate_id}"},
            "conditioning_after_score_fixed": {
                "state_sha256": f"state-{candidate_id}"
            },
            "evaluation_prediction_accuracy": 0.7,
        }
        first.append({**structural, "fit_wall_seconds": 1.0, "peak_rss_bytes": 10})
        second.append({**structural, "fit_wall_seconds": 9.0, "peak_rss_bytes": 99})

    verification = runner._determinism_verification(first, second)
    assert verification["status"] == "pass"
    assert verification["exact"] is True
    assert verification["runtime_fields_excluded"] is True
    assert all(
        value["warmup_signature_sha256"] == value["measured_signature_sha256"]
        for value in verification["candidates"].values()
    )

    second[2]["candidate_score"] = 0.74
    failed = runner._determinism_verification(first, second)
    assert failed["status"] == "fail"
    assert failed["candidates"]["C"]["exact"] is False
