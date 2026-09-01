"""Evaluate the frozen winner-preserving gate on the new DTD texture target."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import scipy
import sklearn

from experiments.continuous_real_bbox_panels import prepare as common_prepare
from experiments.continuous_real_bbox_panels import run as bbox_parent
from experiments.continuous_real_dtd_texture_confirmation import prepare
from experiments.continuous_real_reliability_gate import run as gate_parent
from experiments.continuous_real_reliability_gate_expanded import run as expanded_parent
from experiments.continuous_real_winner_preserved_gate import run as winner_parent


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
PROTOCOL_SHA256_PATH = PACKAGE_DIR / "protocol.sha256"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "continuous_real_dtd_texture_confirmation" / "results"
BACKBONES = bbox_parent.BACKBONES
METHODS = (
    "broad_rff",
    "winner_preserved_borda_a050",
    "fixed_winner_preserved_gate",
    "linear_probe",
)
FEATURE_NAMES = gate_parent.FEATURE_NAMES
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026091301
TIE_ATOL = 1.0e-12


def _load_protocol() -> dict[str, Any]:
    expected = PROTOCOL_SHA256_PATH.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or common_prepare.sha256_path(PROTOCOL_PATH) != expected:
        raise ValueError("DTD texture confirmation protocol SHA-256 sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_dtd_texture_new_target_before_outcomes":
        raise ValueError("DTD texture confirmation protocol is not frozen")
    if tuple(protocol["backbones"]) != BACKBONES:
        raise ValueError("DTD texture confirmation backbone roster changed")
    if tuple(protocol["methods"]) != METHODS:
        raise ValueError("DTD texture confirmation method set changed")
    if tuple(protocol["replicate_seeds"]) != prepare.REPLICATE_SEEDS:
        raise ValueError("DTD texture confirmation replicate seeds changed")
    if tuple(protocol["gate_model"]["feature_names"]) != FEATURE_NAMES:
        raise ValueError("DTD texture confirmation gate features changed")
    ancestry = protocol["ancestry"]
    paths = {
        "cohort_sha256": prepare.DEFAULT_OUTPUT_ROOT / "cohort.json",
        "input_manifest_sha256": prepare.DEFAULT_OUTPUT_ROOT / "manifest.json",
        "development_protocol_sha256": winner_parent.PROTOCOL_PATH,
        "development_summary_sha256": winner_parent.DEFAULT_OUTPUT_ROOT / "summary.json",
        "development_manifest_sha256": winner_parent.DEFAULT_OUTPUT_ROOT / "manifest.json",
    }
    for field, path in paths.items():
        if common_prepare.sha256_path(path) != ancestry[field]:
            raise ValueError(f"DTD texture confirmation ancestry changed: {field}")
    if ancestry["registry_sha256"] != prepare.REGISTRY_SHA256:
        raise ValueError("DTD texture confirmation registry identity changed")
    return protocol


def recompute_development_gate_model() -> dict[str, Any]:
    """Recompute the frozen final model from all 25 development panels."""
    panels = expanded_parent.load_panels(expanded_parent._load_protocol())
    rows = [
        winner_parent.panel_metric(panel, method)
        for panel in panels
        for method in ("broad_rff", "winner_preserved_borda_a050")
    ]
    lookup = expanded_parent._lookup(rows)
    features, benefits = [], []
    for panel in panels:
        panel_id = f"{panel['family']}__s{int(panel['seed'])}"
        features.append(gate_parent.feature_vector(panel))
        benefits.append(
            float(lookup[("broad_rff", panel_id)]["top_5_mean_utility_regret_r2"])
            - float(
                lookup[("winner_preserved_borda_a050", panel_id)][
                    "top_5_mean_utility_regret_r2"
                ]
            )
        )
    model = gate_parent.fit_ridge_gate(np.asarray(features), np.asarray(benefits))
    model.update(
        {
            "feature_names": list(FEATURE_NAMES),
            "training_target": "broad_minus_winner_preserved_borda_top5_mean_regret",
            "training_dataset_groups": list(expanded_parent.GROUPS),
        }
    )
    return model


def _assert_model_exact(protocol: Mapping[str, Any]) -> None:
    expected = json.dumps(protocol["gate_model"], sort_keys=True, separators=(",", ":"))
    observed = json.dumps(recompute_development_gate_model(), sort_keys=True, separators=(",", ":"))
    if observed != expected:
        raise ValueError("frozen all-development gate model does not recompute exactly")


def _validate_cohort(cohort: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    expected = {
        "dataset_id": prepare.DATASET_ID,
        "dataset_group": prepare.DATASET_ID,
        "family": prepare.FAMILY,
        "target_recipe": prepare.TARGET_RECIPE,
        "target_image_size": prepare.TARGET_IMAGE_SIZE,
        "target_sha256": protocol["target"]["target_sha256"],
        "rows_sha256": protocol["target"]["cohort_rows_sha256"],
        "source_manifest_sha256": protocol["target"]["source_manifest_sha256"],
        "embedding_union_sample_ids_sha256": protocol["target"][
            "embedding_union_sample_ids_sha256"
        ],
        "replicate_seeds": list(prepare.REPLICATE_SEEDS),
        "target_bins": prepare.TARGET_BINS,
        "training_rows_per_replicate": prepare.TRAIN_ROWS,
        "evaluation_rows": prepare.EVALUATION_ROWS,
        "training_bank_seed": prepare.TRAIN_BANK_SEED,
        "evaluation_seed": prepare.EVALUATION_SEED,
    }
    for field, value in expected.items():
        if cohort.get(field) != value:
            raise ValueError(f"DTD texture cohort field changed: {field}")
    if common_prepare.sha256_payload(cohort.get("rows")) != cohort.get("rows_sha256"):
        raise ValueError("DTD texture cohort rows are malformed")


def _panel_rows(cohort: Mapping[str, Any], seed: int):
    training = [
        row
        for row in cohort["rows"]
        if row["split"] == "training" and int(row["replicate_seed"]) == int(seed)
    ]
    evaluation = [row for row in cohort["rows"] if row["split"] == "evaluation"]
    if len(training) != prepare.TRAIN_ROWS or len(evaluation) != prepare.EVALUATION_ROWS:
        raise ValueError("DTD texture panel cohort is incomplete")
    if {row["sample_id"] for row in training}.intersection(
        row["sample_id"] for row in evaluation
    ):
        raise ValueError("DTD texture training and evaluation cohorts overlap")
    return training, evaluation


def run_panel(
    cohort: Mapping[str, Any],
    union_ids: Sequence[str],
    bank: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    training, evaluation = _panel_rows(cohort, seed)
    return {
        "seed": int(seed),
        "family": prepare.FAMILY,
        "dataset_group": prepare.DATASET_ID,
        "candidate_count": len(BACKBONES),
        "candidates": [
            bbox_parent.evaluate_candidate(
                backbone, bank[backbone], union_ids, training, evaluation, seed, position
            )
            for position, backbone in enumerate(BACKBONES)
        ],
    }


def gate_selection(panel: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    features = gate_parent.feature_mapping(panel)
    predicted = gate_parent.predict_benefit(
        model,
        np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64),
    )
    selected = "winner_preserved_borda_a050" if predicted > 0.0 else "broad_rff"
    return {
        "features": features,
        "predicted_alternative_benefit": float(predicted),
        "selected_method": selected,
    }


def panel_metric(
    panel: Mapping[str, Any], method: str, model: Mapping[str, Any]
) -> dict[str, Any]:
    if method in {"broad_rff", "winner_preserved_borda_a050", "linear_probe"}:
        return winner_parent.panel_metric(panel, method)
    if method != "fixed_winner_preserved_gate":
        raise ValueError(f"unknown DTD texture confirmation method {method!r}")
    selection = gate_selection(panel, model)
    groups = (
        winner_parent.winner_preserved_borda_groups(panel)
        if selection["selected_method"] == "winner_preserved_borda_a050"
        else bbox_parent.ranking_groups(panel, "broad_rff")
    )
    row = expanded_parent._custom_panel_metric(panel, method, groups)
    row.update(selection)
    return row


def _paired_interval(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, metric: str, offset: int
) -> dict[str, Any]:
    lookup = expanded_parent._lookup(rows)
    values = np.asarray(
        [
            float(lookup[(left, f"{prepare.FAMILY}__s{seed}")][metric])
            - float(lookup[(right, f"{prepare.FAMILY}__s{seed}")][metric])
            for seed in prepare.REPLICATE_SEEDS
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    draws = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "estimate": float(np.mean(values)),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "seed_block_values": values.tolist(),
    }


def summarize(
    panels: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if [int(panel["seed"]) for panel in panels] != list(prepare.REPLICATE_SEEDS):
        raise ValueError("DTD texture confirmation panel grid changed")
    model = protocol["gate_model"]
    rows = [panel_metric(panel, method, model) for panel in panels for method in METHODS]
    summaries = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != len(prepare.REPLICATE_SEEDS):
            raise ValueError("DTD texture confirmation method grid is incomplete")
        summaries.append(
            {
                "method": method,
                "panel_count": len(selected),
                "mean_spearman": float(np.mean([row["spearman"] for row in selected])),
                "mean_top_1_opportunity_regret_r2": float(
                    np.mean([row["top_1_opportunity_regret_r2"] for row in selected])
                ),
                "mean_top_1_exact_best_coverage": float(
                    np.mean([row["top_1_exact_best_coverage"] for row in selected])
                ),
                "mean_top_5_opportunity_regret_r2": float(
                    np.mean([row["top_5_opportunity_regret_r2"] for row in selected])
                ),
                "mean_top_5_mean_utility_regret_r2": float(
                    np.mean([row["top_5_mean_utility_regret_r2"] for row in selected])
                ),
                "mean_top_5_material_precision": float(
                    np.mean([row["top_5_material_precision"] for row in selected])
                ),
            }
        )
    comparisons = (
        ("broad_rff", "top_5_mean_utility_regret_r2"),
        ("winner_preserved_borda_a050", "top_5_mean_utility_regret_r2"),
        ("linear_probe", "top_5_mean_utility_regret_r2"),
        ("broad_rff", "top_5_opportunity_regret_r2"),
        ("broad_rff", "top_1_opportunity_regret_r2"),
    )
    contrasts = [
        _paired_interval(rows, "fixed_winner_preserved_gate", right, metric, index * 100)
        for index, (right, metric) in enumerate(comparisons)
    ]
    lookup = {(row["right"], row["metric"]): row for row in contrasts}
    primary = lookup[("broad_rff", "top_5_mean_utility_regret_r2")]
    top5 = lookup[("broad_rff", "top_5_opportunity_regret_r2")]
    top1 = lookup[("broad_rff", "top_1_opportunity_regret_r2")]
    checks = [
        {
            "criterion": "fixed_gate_improves_broad_top5_mean_regret",
            "observed_upper_95": primary["upper_95"],
            "required": "<0",
            "pass": bool(primary["upper_95"] < 0.0),
        },
        {
            "criterion": "fixed_gate_top1_exactly_matches_broad",
            "observed_estimate": top1["estimate"],
            "pass": bool(
                top1["estimate"] == 0.0
                and top1["lower_95"] == 0.0
                and top1["upper_95"] == 0.0
            ),
        },
        {
            "criterion": "fixed_gate_top5_opportunity_noninferior_to_broad",
            "observed_upper_95": top5["upper_95"],
            "maximum": 0.01,
            "pass": bool(top5["upper_95"] <= 0.01),
        },
        {
            "criterion": "complete_finite_five_seed_grid",
            "observed_rows": len(rows),
            "pass": bool(
                len(rows) == len(METHODS) * len(prepare.REPLICATE_SEEDS)
                and all(np.isfinite(row["spearman"]) for row in rows)
            ),
        },
    ]
    return {
        "schema_version": 1,
        "status": "completed_dtd_texture_new_target_confirmation",
        "scope": "new_continuous_target_holdout_on_previously_used_dtd_images",
        "panel_count": len(panels),
        "gate_model": dict(model),
        "gate_predictions": [
            {"seed": int(panel["seed"]), **gate_selection(panel, model)} for panel in panels
        ],
        "panel_metrics": rows,
        "method_summaries": summaries,
        "contrasts": contrasts,
        "best_head_counts": {
            head: sum(
                head in candidate["best_head_ids"]
                for panel in panels
                for candidate in panel["candidates"]
            )
            for head in ("linear_ridge", "quadratic_ridge", "knn", "rbf_svr")
        },
        "decision": {"status": "pass" if all(row["pass"] for row in checks) else "fail", "checks": checks},
        "protocol": dict(protocol),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# DTD texture-gradient new-target confirmation",
        "",
        f"Decision: **{summary['decision']['status'].upper()}**.",
        "",
        "This is a new continuous target on previously used DTD images, not an untouched-dataset confirmation.",
        "The fixed gate was fit only on the prior 25 development panels and was not refit here.",
        "",
        "| Method | Spearman | Top-1 regret | Top-1 exact | Top-5 regret | Top-5 precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["method_summaries"]:
        lines.append(
            f"| {row['method']} | {row['mean_spearman']:.4f} | "
            f"{row['mean_top_1_opportunity_regret_r2']:.5f} | "
            f"{row['mean_top_1_exact_best_coverage']:.3f} | "
            f"{row['mean_top_5_mean_utility_regret_r2']:.5f} | "
            f"{row['mean_top_5_material_precision']:.3f} |"
        )
    lines.extend(["", "## Frozen gate choices", ""])
    for row in summary["gate_predictions"]:
        lines.append(
            f"- seed {row['seed']}: `{row['selected_method']}` "
            f"(predicted benefit {row['predicted_alternative_benefit']:.6f})"
        )
    lines.extend(["", "## Best reference-head counts", ""])
    lines.extend(f"- `{head}`: {count}" for head, count in summary["best_head_counts"].items())
    return "\n".join(lines) + "\n"


def run(
    input_root: Path = prepare.DEFAULT_OUTPUT_ROOT,
    fused_input_root: Path = prepare.DEFAULT_FUSED_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    protocol = _load_protocol()
    _assert_model_exact(protocol)
    cohort = prepare.load_cohort(input_root.resolve())
    _validate_cohort(cohort, protocol)
    registry = bbox_parent._load_registry(fused_input_root.resolve())
    record = bbox_parent._record(registry, prepare.DATASET_ID)
    union_ids, bank = bbox_parent._load_bank(record)
    output_root = output_root.resolve()
    if (output_root / "manifest.json").exists():
        raise FileExistsError("DTD texture confirmation terminal artifact already exists")
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    panels = []
    for seed in prepare.REPLICATE_SEEDS:
        path = checkpoints / f"{prepare.DATASET_ID}__seed_{seed}.json"
        if path.exists():
            if not resume:
                raise FileExistsError(f"DTD texture checkpoint exists: {path.name}")
            panel = json.loads(path.read_text(encoding="utf-8"))
        else:
            panel = run_panel(cohort, union_ids, bank, seed)
            path.write_text(common_prepare.canonical_json(panel) + "\n", encoding="utf-8")
        panels.append(panel)
    raw = {"schema_version": 1, "status": "completed", "protocol": protocol, "panels": panels}
    summary = summarize(panels, protocol)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, payload in (("raw_results.json", raw), ("summary.json", summary)):
        (output_root / name).write_text(common_prepare.canonical_json(payload) + "\n", encoding="utf-8")
    (output_root / "report.md").write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed_dtd_texture_new_target_confirmation",
        "implementation_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "protocol_sha256": common_prepare.sha256_path(PROTOCOL_PATH),
        "registry_sha256": prepare.REGISTRY_SHA256,
        "cohort_sha256": common_prepare.sha256_path(input_root / "cohort.json"),
        "files": {
            name: {
                "sha256": common_prepare.sha256_path(output_root / name),
                "bytes": (output_root / name).stat().st_size,
            }
            for name in ("raw_results.json", "summary.json", "report.md")
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    (output_root / "manifest.json").write_text(
        common_prepare.canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=prepare.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fused-input-root", type=Path, default=prepare.DEFAULT_FUSED_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.input_root, args.fused_input_root, args.output_root, resume=args.resume)
    print(common_prepare.canonical_json({"status": summary["status"], "decision": summary["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
