"""
Visualize how ContinuousOverlapIndex responds to synthetic separation.

The feature geometries match ``visualize_discrete_overlap_sweeps.py``, but each
sample has a noisy continuous target drawn from one of two target regimes. The
figure sweeps the separation between those regimes in feature space and writes
``img/continuous_overlap_sweeps.png`` by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-overlapindex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from overlapindex import ContinuousOverlapIndex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "img" / "continuous_overlap_sweeps.png"
N_PER_REGIME = 280
N_REPETITIONS = 3

DatasetFactory = Callable[[float, int], tuple[np.ndarray, np.ndarray]]


def continuous_targets(rng: np.random.Generator) -> np.ndarray:
    """Return two narrow but genuinely continuous target regimes."""
    low = rng.normal(loc=-1.0, scale=0.13, size=N_PER_REGIME)
    high = rng.normal(loc=1.0, scale=0.13, size=N_PER_REGIME)
    return np.concatenate((low, high))


def gaussian_clouds(distance: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gaussian feature clouds associated with two target regimes."""
    rng = np.random.default_rng(seed)
    left = rng.normal(scale=0.72, size=(N_PER_REGIME, 2))
    right = rng.normal(scale=0.72, size=(N_PER_REGIME, 2))
    left[:, 0] -= distance / 2.0
    right[:, 0] += distance / 2.0
    return np.vstack((left, right)), continuous_targets(rng)


def vertical_bars(distance: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return vertical feature bars associated with two target regimes."""
    rng = np.random.default_rng(seed)
    left = np.column_stack(
        (
            rng.uniform(-0.62, 0.62, N_PER_REGIME) - distance / 2.0,
            rng.uniform(-2.0, 2.0, N_PER_REGIME),
        )
    )
    right = np.column_stack(
        (
            rng.uniform(-0.62, 0.62, N_PER_REGIME) + distance / 2.0,
            rng.uniform(-2.0, 2.0, N_PER_REGIME),
        )
    )
    return np.vstack((left, right)), continuous_targets(rng)


def concentric_rings(radius_difference: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two concentric feature rings associated with target regimes."""
    rng = np.random.default_rng(seed)
    inner_angle = rng.uniform(0.0, 2.0 * np.pi, N_PER_REGIME)
    outer_angle = rng.uniform(0.0, 2.0 * np.pi, N_PER_REGIME)
    inner_radius = 1.25 + rng.normal(scale=0.11, size=N_PER_REGIME)
    outer_radius = (
        1.25
        + radius_difference
        + rng.normal(scale=0.11, size=N_PER_REGIME)
    )
    inner = np.column_stack(
        (inner_radius * np.cos(inner_angle), inner_radius * np.sin(inner_angle))
    )
    outer = np.column_stack(
        (outer_radius * np.cos(outer_angle), outer_radius * np.sin(outer_angle))
    )
    return np.vstack((inner, outer)), continuous_targets(rng)


def make_model(seed: int) -> ContinuousOverlapIndex:
    """Return the deterministic offline estimator used in the sweep."""
    return ContinuousOverlapIndex(
        model_type="KMeans",
        kmeans_k=8,
        kmeans_kwargs={"random_state": seed, "n_init": 3},
        n_target_cells=2,
        adjacency_mode="soft_topk",
        top_k=5,
        feature_temperature=0.25,
        null_mode="refit_permutation",
        n_null_permutations=6,
        random_state=seed,
    )


def score_sweep(
    factory: DatasetFactory,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and standard deviation across repeated synthetic draws."""
    scores = np.empty((N_REPETITIONS, parameters.size), dtype=float)
    for repetition in range(N_REPETITIONS):
        seed = 200 + repetition
        for index, parameter in enumerate(parameters):
            X, y = factory(float(parameter), seed)
            scores[repetition, index] = make_model(seed).fit(X, y).index
    return scores.mean(axis=0), scores.std(axis=0)


def plot_row(
    axes: np.ndarray,
    name: str,
    parameter_name: str,
    factory: DatasetFactory,
    parameters: np.ndarray,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """Plot three exemplars and the response curve for one geometry."""
    mean, std = score_sweep(factory, parameters)
    exemplar_indices = (0, parameters.size // 2, parameters.size - 1)

    for column, parameter_index in enumerate(exemplar_indices):
        parameter = float(parameters[parameter_index])
        X, y = factory(parameter, seed=200)
        score = make_model(seed=200).fit(X, y).index
        ax = axes[column]
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="coolwarm",
            vmin=-1.4,
            vmax=1.4,
            s=7,
            alpha=0.68,
            linewidths=0,
            rasterized=True,
        )
        ax.set(
            xlim=limits[0],
            ylim=limits[1],
            aspect="equal",
            title=f"{parameter_name} = {parameter:.2f}\nCOI = {score:.3f}",
        )
        ax.grid(alpha=0.14)
        if column == 0:
            ax.set_ylabel(f"{name}\nFeature 2")
        ax.set_xlabel("Feature 1")

    curve_ax = axes[3]
    curve_ax.plot(parameters, mean, color="#138a72", linewidth=2.2)
    curve_ax.fill_between(
        parameters,
        np.clip(mean - std, 0.0, 1.0),
        np.clip(mean + std, 0.0, 1.0),
        color="#138a72",
        alpha=0.18,
        linewidth=0,
    )
    curve_ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1.0)
    curve_ax.set(
        xlim=(float(parameters[0]), float(parameters[-1])),
        ylim=(-0.02, 1.02),
        xlabel=parameter_name,
        ylabel="Continuous Overlap Index",
        title="Score response",
    )
    curve_ax.grid(alpha=0.2)


def build_figure(output_path: Path) -> None:
    """Build and save the full continuous-overlap gallery."""
    experiments = (
        (
            "Gaussian clouds",
            "Center distance",
            gaussian_clouds,
            np.linspace(0.0, 4.0, 17),
            ((-3.2, 3.2), (-2.5, 2.5)),
        ),
        (
            "Vertical bars",
            "Center distance",
            vertical_bars,
            np.linspace(0.0, 2.8, 17),
            ((-2.2, 2.2), (-2.2, 2.2)),
        ),
        (
            "Concentric rings",
            "Radius difference",
            concentric_rings,
            np.linspace(0.0, 3.0, 17),
            ((-4.6, 4.6), (-4.6, 4.6)),
        ),
    )

    fig, axes = plt.subplots(3, 4, figsize=(14.2, 10.4), constrained_layout=True)
    for row, experiment in enumerate(experiments):
        plot_row(axes[row], *experiment)

    fig.suptitle(
        "ContinuousOverlapIndex: synthetic separation sweeps",
        fontsize=16,
        fontweight="bold",
    )
    scalar_mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=-1.4, vmax=1.4),
        cmap="coolwarm",
    )
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=axes[:, :3].ravel().tolist(),
        location="bottom",
        shrink=0.55,
        aspect=35,
        pad=0.06,
    )
    colorbar.set_label("Continuous target")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved continuous visualization to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_figure(parse_args().output.resolve())
