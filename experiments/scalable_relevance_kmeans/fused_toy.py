"""Toy ranking and Food-shaped timing trials for the fused prototype fitter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from experiments.diagonal_metric_toy import experiment as base
from experiments.scalable_relevance_kmeans.fused_prototypes import (
    FusedFastRelevanceKMeans,
    fit_fused_class_prototypes,
    fit_looped_class_prototypes,
)
from experiments.scalable_relevance_kmeans.scalable_v2 import (
    FastScalableRelevanceKMeans,
)


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "fused_toy_protocol.json"
PROTOCOL_SIDECAR = PACKAGE_DIR / "fused_toy_protocol.sha256"
METHODS = ("current_nolloyd", "looped_fixed3", "fused_fixed3", "linear_probe")
SCENARIOS = ("linear_nuisance", "nonlinear_nuisance")
BUDGET_PER_CLASS = 128
SEED = 43
PROTOTYPE_ITERATIONS = 3
PROFILE_FEATURES = (384, 768, 2048)
PROFILE_REPEATS = 7


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, _canonical_json(value) + "\n")


def _validate_protocol() -> str:
    observed = _sha256(PROTOCOL_PATH)
    expected = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip()
    if observed != expected or len(expected) != 64:
        raise RuntimeError("fused toy protocol sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_ranking_outcomes":
        raise RuntimeError("fused toy protocol is not frozen")
    ranking = protocol.get("ranking_toy", {})
    profile = protocol.get("food_shaped_prototype_benchmark", {})
    if (
        tuple(protocol.get("methods", {})) != METHODS
        or tuple(ranking.get("scenarios", ())) != SCENARIOS
        or ranking.get("budget_per_class") != BUDGET_PER_CLASS
        or ranking.get("seed") != SEED
        or tuple(profile.get("feature_counts", ())) != PROFILE_FEATURES
        or profile.get("measured_repeats") != PROFILE_REPEATS
        or protocol.get("fixed_lloyd_iterations") != PROTOTYPE_ITERATIONS
    ):
        raise RuntimeError("fused toy protocol does not match implementation")
    return observed


def _current_selector(seed: int) -> FastScalableRelevanceKMeans:
    return FastScalableRelevanceKMeans(
        k_per_class=base.K_PER_CLASS,
        kmeans_kwargs={
            "batch_size": 256,
            "max_no_improvement": 5,
            "compute_labels": False,
            "n_init": 1,
            "init": "random",
            "random_state": int(seed),
        },
        fast_scoring=True,
        margin_rows_per_class=32,
        lloyd_update=False,
    )


def _fixed_selector(seed: int, backend: str) -> FusedFastRelevanceKMeans:
    return FusedFastRelevanceKMeans(
        k_per_class=base.K_PER_CLASS,
        seed=int(seed),
        prototype_iterations=PROTOTYPE_ITERATIONS,
        prototype_backend=backend,
        margin_rows_per_class=32,
    )


def _crossfit(X: np.ndarray, y: np.ndarray, *, method: str, seed: int) -> tuple[float, dict[str, Any]]:
    if method == "linear_probe":
        return base._crossfit_selector(X, y, method="linear_probe", seed=seed)
    scores: list[float] = []
    prototype_seconds: list[float] = []
    state_hashes: list[str] = []
    for fold, (train, heldout) in enumerate(base._folds(y, seed)):
        fold_seed = int(seed) + int(fold)
        if method == "current_nolloyd":
            selector: Any = _current_selector(fold_seed)
        elif method == "looped_fixed3":
            selector = _fixed_selector(fold_seed, "looped")
        elif method == "fused_fixed3":
            selector = _fixed_selector(fold_seed, "fused")
        else:
            raise ValueError(f"unknown fused toy method {method!r}")
        selector.fit(X[train], y[train])
        scores.append(float(selector.score_fixed(X[heldout], y[heldout])))
        diagnostics = dict(selector.diagnostics_ or {})
        if method == "current_nolloyd":
            prototype_seconds.append(float(diagnostics["initial_kmeans_wall_seconds"]))
        else:
            prototype_seconds.append(float(diagnostics["prototype_fit_wall_seconds"]))
        state_hashes.append(str(diagnostics["state_sha256"]))
    return float(np.mean(scores)), {
        "prototype_wall_seconds": float(sum(prototype_seconds)),
        "state_hashes": state_hashes,
    }


def _method_order(case_index: int) -> tuple[str, ...]:
    offset = int(case_index) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def _ranking_trial() -> dict[str, Any]:
    # Excluded warmups use the first frozen panel and are never serialized.
    warm_X, warm_y = base.generate_embedding(
        scenario=SCENARIOS[0],
        seed=SEED,
        split="selector",
        n_per_class=BUDGET_PER_CLASS,
        candidate=base.PANEL_CANDIDATES[0],
    )
    for method in METHODS:
        _crossfit(warm_X, warm_y, method=method, seed=SEED)

    rows: list[dict[str, Any]] = []
    references: dict[str, dict[str, float]] = {}
    case_index = 0
    for scenario in SCENARIOS:
        references[scenario] = {
            str(candidate["candidate_id"]): base._reference_accuracy(
                scenario=scenario, seed=SEED, candidate=candidate
            )
            for candidate in base.PANEL_CANDIDATES
        }
        for candidate in base.PANEL_CANDIDATES:
            candidate_id = str(candidate["candidate_id"])
            X, y = base.generate_embedding(
                scenario=scenario,
                seed=SEED,
                split="selector",
                n_per_class=BUDGET_PER_CLASS,
                candidate=candidate,
            )
            order = _method_order(case_index)
            for position, method in enumerate(order):
                started = time.perf_counter()
                score, diagnostics = _crossfit(X, y, method=method, seed=SEED)
                wall = time.perf_counter() - started
                rows.append(
                    {
                        "scenario": scenario,
                        "candidate_id": candidate_id,
                        "method": method,
                        "score": float(score),
                        "reference_accuracy": float(references[scenario][candidate_id]),
                        "wall_seconds": float(wall),
                        "execution_position": int(position),
                        "execution_order": list(order),
                        "warmup_excluded": True,
                        "diagnostics": diagnostics,
                    }
                )
            case_index += 1
    summary: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for method in METHODS:
            selected = {
                str(row["candidate_id"]): float(row["score"])
                for row in rows
                if row["scenario"] == scenario and row["method"] == method
            }
            metrics = base._selection_metrics(selected, references[scenario])
            method_rows = [
                row for row in rows if row["scenario"] == scenario and row["method"] == method
            ]
            summary.append(
                {
                    "scenario": scenario,
                    "method": method,
                    **metrics,
                    "median_wall_seconds": float(
                        np.median([row["wall_seconds"] for row in method_rows])
                    ),
                }
            )
    parity = []
    for scenario in SCENARIOS:
        looped = next(
            row for row in summary if row["scenario"] == scenario and row["method"] == "looped_fixed3"
        )
        fused = next(
            row for row in summary if row["scenario"] == scenario and row["method"] == "fused_fixed3"
        )
        scores_looped = [
            row["score"] for row in rows if row["scenario"] == scenario and row["method"] == "looped_fixed3"
        ]
        scores_fused = [
            row["score"] for row in rows if row["scenario"] == scenario and row["method"] == "fused_fixed3"
        ]
        parity.append(
            {
                "scenario": scenario,
                "exact_scores": scores_looped == scores_fused,
                "exact_regret": looped["regret_pp"] == fused["regret_pp"],
            }
        )
    return {"rows": rows, "summary": summary, "looped_fused_parity": parity}


def _profile_values(feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, feature_count]))
    labels = np.repeat(np.arange(40, dtype=np.int64), 64)
    values = rng.normal(size=(labels.size, feature_count)).astype(np.float32)
    signal_width = min(40, feature_count)
    values[:, :signal_width] += np.eye(signal_width, dtype=np.float32)[
        labels % signal_width
    ]
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)
    return values, labels


def _current_sklearn_prototypes(X: np.ndarray, y: np.ndarray, seed: int) -> None:
    for label in dict.fromkeys(y.tolist()):
        estimator = MiniBatchKMeans(
            n_clusters=10,
            batch_size=256,
            max_no_improvement=5,
            compute_labels=False,
            n_init=1,
            init="random",
            random_state=int(seed),
        )
        estimator.fit(X[y == label])


def _prototype_benchmark() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = ("current_sklearn", "looped_fixed3", "fused_fixed3")
    for feature_count in PROFILE_FEATURES:
        X, y = _profile_values(feature_count)
        runners = {
            "current_sklearn": lambda: _current_sklearn_prototypes(X, y, SEED),
            "looped_fixed3": lambda: fit_looped_class_prototypes(
                X, y, k_per_class=10, seed=SEED, iterations=PROTOTYPE_ITERATIONS
            ),
            "fused_fixed3": lambda: fit_fused_class_prototypes(
                X, y, k_per_class=10, seed=SEED, iterations=PROTOTYPE_ITERATIONS
            ),
        }
        for runner in runners.values():
            runner()
        observations = {method: [] for method in methods}
        for repeat in range(PROFILE_REPEATS):
            order = methods[repeat % len(methods) :] + methods[: repeat % len(methods)]
            for method in order:
                started = time.perf_counter()
                runners[method]()
                observations[method].append(float(time.perf_counter() - started))
        medians = {method: float(np.median(observations[method])) for method in methods}
        looped = fit_looped_class_prototypes(
            X, y, k_per_class=10, seed=SEED, iterations=PROTOTYPE_ITERATIONS
        )
        fused = fit_fused_class_prototypes(
            X, y, k_per_class=10, seed=SEED, iterations=PROTOTYPE_ITERATIONS
        )
        output.append(
            {
                "feature_count": int(feature_count),
                "class_count": 40,
                "rows_per_class": 64,
                "k_per_class": 10,
                "iterations": PROTOTYPE_ITERATIONS,
                "median_wall_seconds": medians,
                "fused_over_current": medians["fused_fixed3"] / medians["current_sklearn"],
                "looped_over_current": medians["looped_fixed3"] / medians["current_sklearn"],
                "fused_over_looped": medians["fused_fixed3"] / medians["looped_fixed3"],
                "max_center_abs_difference": float(
                    np.max(np.abs(fused.centers - looped.centers))
                ),
                "measured_repeats": PROFILE_REPEATS,
                "warmup_excluded": True,
            }
        )
    return output


def run_trials() -> dict[str, Any]:
    protocol_sha = _validate_protocol()
    ranking = _ranking_trial()
    profile = _prototype_benchmark()
    current = {
        (row["scenario"]): row
        for row in ranking["summary"]
        if row["method"] == "current_nolloyd"
    }
    fused = {
        (row["scenario"]): row
        for row in ranking["summary"]
        if row["method"] == "fused_fixed3"
    }
    gates = {
        "prototype_parity": all(
            row["max_center_abs_difference"] == 0.0 for row in profile
        ),
        "selector_parity": all(
            row["exact_scores"] is True for row in ranking["looped_fused_parity"]
        ),
        "ranking": all(
            float(fused[scenario]["regret_pp"])
            - float(current[scenario]["regret_pp"])
            <= 1.0
            for scenario in SCENARIOS
        ),
        "prototype_speed": all(row["fused_over_current"] < 1.0 for row in profile),
    }
    return {
        "schema_version": 1,
        "study": "fused_class_batched_prototype_toy",
        "artifact_status": "completed",
        "development_only": True,
        "protocol_sha256": protocol_sha,
        "implementation_sha256": _sha256(Path(__file__)),
        "fitter_sha256": _sha256(PACKAGE_DIR / "fused_prototypes.py"),
        "generator_sha256": _sha256(Path(base.__file__)),
        "ranking": ranking,
        "prototype_benchmark": profile,
        "gates": gates,
        "decision": "pass_for_food_screen_design" if all(gates.values()) else "fail",
    }


def _report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Fused class-batched prototype toy",
        "",
        "Development-only evidence; not a Food-101 promotion or public API result.",
        "",
        f"Decision: **{payload['decision']}**.",
        "",
        "## Ranking",
        "",
        "| Scenario | Method | Regret pp | Spearman | Median wall s |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["ranking"]["summary"]:
        rho = "—" if row["spearman"] is None else f"{row['spearman']:.3f}"
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['regret_pp']:.3f} | "
            f"{rho} | {row['median_wall_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Food-shaped prototype fitting",
            "",
            "| Features | Current sklearn s | Looped fixed3 s | Fused fixed3 s | Fused/current | Fused/looped | Center max delta |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["prototype_benchmark"]:
        medians = row["median_wall_seconds"]
        lines.append(
            f"| {row['feature_count']} | {medians['current_sklearn']:.5f} | "
            f"{medians['looped_fixed3']:.5f} | {medians['fused_fixed3']:.5f} | "
            f"{row['fused_over_current']:.3f} | {row['fused_over_looped']:.3f} | "
            f"{row['max_center_abs_difference']:.1e} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, status in payload["gates"].items():
        lines.append(f"- {name}: **{'pass' if status else 'fail'}**")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    payload = run_trials()
    _atomic_json(args.output / "raw_results.json", payload)
    _atomic_text(args.output / "report.md", _report(payload))
    _atomic_json(
        args.output / "manifest.json",
        {
            "artifact_status": "completed",
            "protocol_sha256": payload["protocol_sha256"],
            "raw_results_sha256": _sha256(args.output / "raw_results.json"),
            "report_sha256": _sha256(args.output / "report.md"),
            "decision": payload["decision"],
        },
    )
    print(_canonical_json({"decision": payload["decision"], "gates": payload["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
