"""Frozen exactness and speed benchmark for the shared bounded evaluator."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import scipy
import sklearn

from experiments.continuous_bounded_shared_evaluator import shared
from experiments.continuous_fisher_ranking import run as common
from experiments.continuous_kernel_alignment import run as kernel
from experiments.continuous_prototype_capacity import run as capacity
from experiments.continuous_shortlist_depth_audit import run as audit_parent
from experiments.continuous_top5_bounded_confirmation import run as confirmation_parent
from experiments.continuous_winner_fill_shortlist import run as winner_parent


PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
RUNNER_PATH = Path(__file__).resolve()
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIRMATION_ROOT = ROOT / "artifacts" / "continuous_top5_bounded_confirmation"
FAMILIES = confirmation_parent.FAMILIES
BENCHMARK_SEED = 3907
SCORE_ATOL = 1.0e-12
SPEED_RATIO_MAXIMUM = 0.75


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or common._sha256(PROTOCOL_PATH) != expected:
        raise ValueError("shared evaluator protocol SHA-256 sidecar does not match")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_shared_evaluator_benchmark_before_execution":
        raise ValueError("shared evaluator protocol is not frozen")
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if int(protocol["benchmark_seed"]) != BENCHMARK_SEED:
        raise ValueError("shared evaluator seed does not match code")
    if tuple(protocol["families"]) != FAMILIES:
        raise ValueError("shared evaluator family set does not match code")
    if int(protocol["candidate_count_per_panel"]) != 16:
        raise ValueError("shared evaluator candidate count does not match code")
    if float(protocol["verification"]["score_absolute_tolerance"]) != SCORE_ATOL:
        raise ValueError("shared evaluator score tolerance does not match code")
    if float(protocol["speed_gate"]["maximum"]) != SPEED_RATIO_MAXIMUM:
        raise ValueError("shared evaluator speed gate does not match code")


def _validate_confirmation(protocol: Mapping[str, Any], root: Path) -> dict[str, Any]:
    root = root.resolve()
    ancestry = protocol["ancestry"]
    paths = {
        "confirmation_raw_results_sha256": root / "raw_results.json",
        "confirmation_summary_sha256": root / "summary.json",
        "confirmation_report_sha256": root / "report.md",
        "confirmation_manifest_sha256": root / "manifest.json",
    }
    for key, path in paths.items():
        if common._sha256(path) != ancestry[key]:
            raise ValueError(f"shared evaluator confirmation mismatch for {key}")
    if common._sha256(confirmation_parent.PROTOCOL_PATH) != ancestry[
        "confirmation_protocol_sha256"
    ]:
        raise ValueError("shared evaluator confirmation protocol changed")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if summary.get("decision", {}).get("status") != ancestry[
        "required_confirmation_status"
    ]:
        raise ValueError("shared evaluator is not authorized by confirmation")
    if manifest.get("implementation_commit") != ancestry[
        "confirmation_implementation_commit"
    ]:
        raise ValueError("shared evaluator confirmation commit mismatch")
    return json.loads((root / "raw_results.json").read_text(encoding="utf-8"))["protocol"]


def _legacy_pair(
    X: np.ndarray,
    primitive: np.ndarray,
    y: np.ndarray,
    panel_seed: int,
    candidate_position: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    broad = kernel.evaluate_selector(
        {"geometry": "rff_mean", "rff_components": 64},
        X,
        primitive,
        y,
        panel_seed,
        candidate_position,
    )
    structured = kernel.evaluate_selector(
        {"geometry": "multiview", "blocks": ["save", "rff_mean"]},
        X,
        primitive,
        y,
        panel_seed,
        candidate_position,
    )
    return {
        "broad_rff_score": float(
            broad["kernel_alignment"]["feature_bandwidths"]["4.0"]["score"]
        ),
        "save_rff_current_score": float(structured["current_common32"]["score"]),
        "wall_seconds": float(time.perf_counter() - started),
    }


def _shared_pair(
    X: np.ndarray,
    primitive: np.ndarray,
    y: np.ndarray,
    panel_seed: int,
    candidate_position: int,
) -> dict[str, Any]:
    result = shared.evaluate_shared_pair(
        X, primitive, y, panel_seed, candidate_position
    )
    return {
        "broad_rff_score": float(result["broad_rff"]["score"]),
        "save_rff_current_score": float(result["save_rff_current"]["score"]),
        "wall_seconds": float(result["wall_seconds"]),
        "fitted_selector_count": int(result["fitted_selector_count"]),
    }


def _score_panel(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "selectors": {
                    "rff64_mean8_k4": {
                        "kernel_alignment": {
                            "feature_bandwidths": {
                                "4.0": {"score": row[f"{prefix}_broad_rff_score"]}
                            }
                        }
                    },
                    "save_rffmean_k4": {
                        "current_common32": {
                            "score": row[f"{prefix}_save_rff_current_score"]
                        }
                    },
                },
            }
            for row in rows
        ]
    }


def evaluate_family(
    generation_protocol: Mapping[str, Any], family: str
) -> dict[str, Any]:
    panel = capacity.generate_panel(generation_protocol, BENCHMARK_SEED, family)
    first = panel["candidates"][0]
    _legacy_pair(
        first["X_train"], first["primitive_train"], panel["y_train"], BENCHMARK_SEED, 0
    )
    _shared_pair(
        first["X_train"], first["primitive_train"], panel["y_train"], BENCHMARK_SEED, 0
    )
    rows = []
    for position, candidate in enumerate(panel["candidates"]):
        arguments = (
            candidate["X_train"],
            candidate["primitive_train"],
            panel["y_train"],
            BENCHMARK_SEED,
            position,
        )
        if (FAMILIES.index(family) + position) % 2 == 0:
            legacy = _legacy_pair(*arguments)
            shared_result = _shared_pair(*arguments)
            order = "legacy_then_shared"
        else:
            shared_result = _shared_pair(*arguments)
            legacy = _legacy_pair(*arguments)
            order = "shared_then_legacy"
        rows.append(
            {
                "family": family,
                "candidate_id": candidate["candidate_id"],
                "candidate_position": int(position),
                "execution_order": order,
                "legacy_broad_rff_score": legacy["broad_rff_score"],
                "shared_broad_rff_score": shared_result["broad_rff_score"],
                "legacy_save_rff_current_score": legacy["save_rff_current_score"],
                "shared_save_rff_current_score": shared_result[
                    "save_rff_current_score"
                ],
                "legacy_wall_seconds": legacy["wall_seconds"],
                "shared_wall_seconds": shared_result["wall_seconds"],
                "shared_fitted_selector_count": shared_result[
                    "fitted_selector_count"
                ],
            }
        )
    legacy_panel = _score_panel(rows, "legacy")
    shared_panel = _score_panel(rows, "shared")
    rank_equal = {
        "broad_rff": winner_parent.ranking_groups(legacy_panel, "broad_rff")
        == winner_parent.ranking_groups(shared_panel, "broad_rff"),
        "save_rff_current": winner_parent.ranking_groups(
            legacy_panel, "save_rff_current"
        )
        == winner_parent.ranking_groups(shared_panel, "save_rff_current"),
        "bounded_2k": audit_parent.ranking_groups(legacy_panel, "bounded_2k", 5)
        == audit_parent.ranking_groups(shared_panel, "bounded_2k", 5),
    }
    legacy_wall = float(sum(row["legacy_wall_seconds"] for row in rows))
    shared_wall = float(sum(row["shared_wall_seconds"] for row in rows))
    return {
        "family": family,
        "candidate_count": len(rows),
        "candidate_rows": rows,
        "maximum_broad_score_absolute_difference": float(
            max(
                abs(row["legacy_broad_rff_score"] - row["shared_broad_rff_score"])
                for row in rows
            )
        ),
        "maximum_save_score_absolute_difference": float(
            max(
                abs(
                    row["legacy_save_rff_current_score"]
                    - row["shared_save_rff_current_score"]
                )
                for row in rows
            )
        ),
        "rank_groups_exact": rank_equal,
        "legacy_panel_wall_seconds": legacy_wall,
        "shared_panel_wall_seconds": shared_wall,
        "shared_over_legacy_wall_ratio": float(shared_wall / legacy_wall),
    }


def summarize(protocol: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(panels) != 6 or {row["family"] for row in panels} != set(FAMILIES):
        raise ValueError("shared evaluator requires exactly six family panels")
    ratios = [float(row["shared_over_legacy_wall_ratio"]) for row in panels]
    complete = all(
        int(row["candidate_count"]) == 16
        and len(row["candidate_rows"]) == 16
        and np.isfinite(
            [
                candidate[key]
                for candidate in row["candidate_rows"]
                for key in (
                    "legacy_broad_rff_score",
                    "shared_broad_rff_score",
                    "legacy_save_rff_current_score",
                    "shared_save_rff_current_score",
                    "legacy_wall_seconds",
                    "shared_wall_seconds",
                )
            ]
        ).all()
        for row in panels
    )
    checks = [
        {
            "criterion": "all_component_scores_exact",
            "maximum_broad_difference": max(
                row["maximum_broad_score_absolute_difference"] for row in panels
            ),
            "maximum_save_difference": max(
                row["maximum_save_score_absolute_difference"] for row in panels
            ),
            "maximum": SCORE_ATOL,
            "pass": bool(
                max(row["maximum_broad_score_absolute_difference"] for row in panels)
                <= SCORE_ATOL
                and max(row["maximum_save_score_absolute_difference"] for row in panels)
                <= SCORE_ATOL
            ),
        },
        {
            "criterion": "all_rank_groups_exact",
            "observed": {
                row["family"]: row["rank_groups_exact"] for row in panels
            },
            "pass": bool(
                all(all(row["rank_groups_exact"].values()) for row in panels)
            ),
        },
        {
            "criterion": "median_panel_runtime_ratio",
            "observed": float(np.median(ratios)),
            "maximum": SPEED_RATIO_MAXIMUM,
            "pass": bool(float(np.median(ratios)) <= SPEED_RATIO_MAXIMUM),
        },
        {
            "criterion": "all_rows_complete_and_finite",
            "observed_candidate_row_count": sum(
                len(row["candidate_rows"]) for row in panels
            ),
            "required_candidate_row_count": 96,
            "pass": bool(complete),
        },
    ]
    return {
        "schema_version": 1,
        "status": "completed_shared_evaluator_benchmark",
        "family_panel_count": len(panels),
        "candidate_row_count": sum(len(row["candidate_rows"]) for row in panels),
        "panel_summaries": list(panels),
        "median_shared_over_legacy_wall_ratio": float(np.median(ratios)),
        "decision": {
            "status": "pass" if all(row["pass"] for row in checks) else "fail",
            "checks": checks,
            "scope": "experimental_shared_evaluator_only",
        },
        "protocol": dict(protocol),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuous bounded shared-evaluator benchmark",
        "",
        f"Decision: **{summary['decision']['status'].upper()}**.",
        "",
        "| Family | Legacy seconds | Shared seconds | Shared / legacy | Broad max diff | SAVE max diff | Bounded rank exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["panel_summaries"]:
        lines.append(
            f"| {row['family']} | {row['legacy_panel_wall_seconds']:.3f} | "
            f"{row['shared_panel_wall_seconds']:.3f} | "
            f"{row['shared_over_legacy_wall_ratio']:.3f} | "
            f"{row['maximum_broad_score_absolute_difference']:.3g} | "
            f"{row['maximum_save_score_absolute_difference']:.3g} | "
            f"{row['rank_groups_exact']['bounded_2k']} |"
        )
    lines.extend(["", "## Frozen checks", ""])
    for check in summary["decision"]["checks"]:
        lines.append(f"- {'PASS' if check['pass'] else 'FAIL'}: `{check['criterion']}`")
    lines.extend(
        [
            "",
            "This benchmark changes computation only. It does not change the confirmed ranking formula or public API.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _repository_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_paths(confirmation_root: Path) -> dict[str, Path]:
    result = {
        "protocol": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256_PATH,
        "runner": RUNNER_PATH,
        "shared_evaluator": Path(shared.__file__).resolve(),
        "readme": PACKAGE_DIR / "README.md",
        "focused_test": ROOT / "tests" / "test_continuous_bounded_shared_evaluator.py",
        "confirmation_protocol": confirmation_parent.PROTOCOL_PATH,
        "confirmation_runner": confirmation_parent.RUNNER_PATH,
        "kernel_runner": kernel.RUNNER_PATH,
        "capacity_runner": capacity.RUNNER_PATH,
        "audit_runner": audit_parent.RUNNER_PATH,
        "winner_runner": winner_parent.RUNNER_PATH,
    }
    for name in ("raw_results.json", "summary.json", "report.md", "manifest.json"):
        result[f"confirmation_{name}"] = confirmation_root.resolve() / name
    return result


def run(output: Path, confirmation_root: Path = DEFAULT_CONFIRMATION_ROOT) -> dict[str, Any]:
    protocol = _load_protocol()
    _validate_protocol(protocol)
    generation_protocol = _validate_confirmation(protocol, confirmation_root)
    implementation_commit = os.environ.get("CONTINUOUS_OI_COMMIT")
    if implementation_commit != _repository_head():
        raise ValueError("CONTINUOUS_OI_COMMIT must equal the frozen Git HEAD")
    source_hashes = {
        key: common._sha256(path)
        for key, path in _source_paths(confirmation_root).items()
    }
    output = output.resolve()
    for name in ("raw_results.json", "summary.json", "report.md", "manifest.json"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite {output / name}")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    for index, family in enumerate(FAMILIES, start=1):
        path = checkpoints / f"seed_{BENCHMARK_SEED}__{family}.json"
        if path.exists():
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            if checkpoint.get("source_hashes") != source_hashes:
                raise ValueError(f"stale shared-evaluator checkpoint: {path}")
            panel = checkpoint["panel"]
        else:
            panel = evaluate_family(generation_protocol, family)
            common._atomic_json(
                path,
                {
                    "schema_version": 1,
                    "source_hashes": source_hashes,
                    "panel": panel,
                },
            )
        panels.append(panel)
        print(f"completed {index}/6: family={family}", flush=True)
    raw = {
        "schema_version": 1,
        "artifact_status": "completed",
        "implementation_commit": implementation_commit,
        "source_hashes": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "protocol": protocol,
        "panel_summaries": panels,
    }
    summary = summarize(protocol, panels)
    common._atomic_json(output / "raw_results.json", raw)
    common._atomic_json(output / "summary.json", summary)
    _atomic_text(output / "report.md", _report(summary))
    manifest = {
        "schema_version": 1,
        "artifact_status": "completed",
        "implementation_commit": implementation_commit,
        "source_hashes": source_hashes,
        "checkpoint_count": len(panels),
        "files": {
            name: {
                "sha256": common._sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in ("raw_results.json", "summary.json", "report.md")
        },
    }
    common._atomic_json(output / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirmation-root", type=Path, default=DEFAULT_CONFIRMATION_ROOT
    )
    args = parser.parse_args()
    summary = run(args.output, args.confirmation_root)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
