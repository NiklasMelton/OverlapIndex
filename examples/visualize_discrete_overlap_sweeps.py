"""
Visualize how the discrete OverlapIndex responds to synthetic separation.

Three two-class geometries are swept from overlapping to separated:

* Gaussian clouds, parameterized by center distance.
* Vertical bars, parameterized by center distance.
* Concentric rings, parameterized by radius difference.

The figure combines representative datasets with the mean OverlapIndex curve
across deterministic repetitions. By default it is written to
``img/discrete_overlap_sweeps.png``.
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

from overlapindex import OverlapIndex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "img" / "discrete_overlap_sweeps.png"
N_PER_CLASS = 320
N_REPETITIONS = 4

DatasetFactory = Callable[[float, int], tuple[np.ndarray, np.ndarray]]


def gaussian_clouds(distance: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two isotropic Gaussian clouds separated along the x axis."""
    rng = np.random.default_rng(seed)
    left = rng.normal(scale=0.72, size=(N_PER_CLASS, 2))
    right = rng.normal(scale=0.72, size=(N_PER_CLASS, 2))
    left[:, 0] -= distance / 2.0
    right[:, 0] += distance / 2.0
    return np.vstack((left, right)), np.repeat((0, 1), N_PER_CLASS)


def vertical_bars(distance: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two uniform vertical bars separated along the x axis."""
    rng = np.random.default_rng(seed)
    left = np.column_stack(
        (
            rng.uniform(-0.62, 0.62, N_PER_CLASS) - distance / 2.0,
            rng.uniform(-2.0, 2.0, N_PER_CLASS),
        )
    )
    right = np.column_stack(
        (
            rng.uniform(-0.62, 0.62, N_PER_CLASS) + distance / 2.0,
            rng.uniform(-2.0, 2.0, N_PER_CLASS),
        )
    )
    return np.vstack((left, right)), np.repeat((0, 1), N_PER_CLASS)


def concentric_rings(radius_difference: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a noisy inner ring and a larger concentric ring."""
    rng = np.random.default_rng(seed)
    inner_angle = rng.uniform(0.0, 2.0 * np.pi, N_PER_CLASS)
    outer_angle = rng.uniform(0.0, 2.0 * np.pi, N_PER_CLASS)
    inner_radius = 1.25 + rng.normal(scale=0.11, size=N_PER_CLASS)
    outer_radius = (
        1.25
        + radius_difference
        + rng.normal(scale=0.11, size=N_PER_CLASS)
    )
    inner = np.column_stack(
        (inner_radius * np.cos(inner_angle), inner_radius * np.sin(inner_angle))
    )
    outer = np.column_stack(
        (outer_radius * np.cos(outer_angle), outer_radius * np.sin(outer_angle))
    )
    return np.vstack((inner, outer)), np.repeat((0, 1), N_PER_CLASS)


def make_model(seed: int) -> OverlapIndex:
    """Return the deterministic offline estimator used in the sweep."""
    return OverlapIndex(
        model_type="KMeans",
        kmeans_k=8,
        kmeans_kwargs={"random_state": seed, "n_init": 5},
    )


def score_sweep(
    factory: DatasetFactory,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and standard deviation across repeated synthetic draws."""
    scores = np.empty((N_REPETITIONS, parameters.size), dtype=float)
    for repetition in range(N_REPETITIONS):
        seed = 100 + repetition
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
        X, y = factory(parameter, seed=100)
        score = make_model(seed=100).fit(X, y).index
        ax = axes[column]
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=np.where(y == 0, "#2979a8", "#e76f51"),
            s=7,
            alpha=0.62,
            linewidths=0,
            rasterized=True,
        )
        ax.set(
            xlim=limits[0],
            ylim=limits[1],
            aspect="equal",
            title=f"{parameter_name} = {parameter:.2f}\nOI = {score:.3f}",
        )
        ax.grid(alpha=0.14)
        if column == 0:
            ax.set_ylabel(f"{name}\nFeature 2")
        ax.set_xlabel("Feature 1")

    curve_ax = axes[3]
    curve_ax.plot(parameters, mean, color="#5b4b8a", linewidth=2.2)
    curve_ax.fill_between(
        parameters,
        np.clip(mean - std, 0.0, 1.0),
        np.clip(mean + std, 0.0, 1.0),
        color="#5b4b8a",
        alpha=0.18,
        linewidth=0,
    )
    curve_ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1.0)
    curve_ax.set(
        xlim=(float(parameters[0]), float(parameters[-1])),
        ylim=(-0.02, 1.02),
        xlabel=parameter_name,
        ylabel="Overlap Index",
        title="Score response",
    )
    curve_ax.grid(alpha=0.2)


def build_figure(output_path: Path) -> None:
    """Build and save the full discrete-overlap gallery."""
    experiments = (
        (
            "Gaussian clouds",
            "Center distance",
            gaussian_clouds,
            np.linspace(0.0, 4.0, 21),
            ((-3.2, 3.2), (-2.5, 2.5)),
        ),
        (
            "Vertical bars",
            "Center distance",
            vertical_bars,
            np.linspace(0.0, 2.8, 21),
            ((-2.2, 2.2), (-2.2, 2.2)),
        ),
        (
            "Concentric rings",
            "Radius difference",
            concentric_rings,
            np.linspace(0.0, 3.0, 21),
            ((-4.6, 4.6), (-4.6, 4.6)),
        ),
    )

    fig, axes = plt.subplots(3, 4, figsize=(14.2, 10.4), constrained_layout=True)
    for row, experiment in enumerate(experiments):
        plot_row(axes[row], *experiment)

    fig.suptitle(
        "Discrete OverlapIndex: synthetic separation sweeps",
        fontsize=16,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="", color="#2979a8", label="Class 0"),
            plt.Line2D([], [], marker="o", linestyle="", color="#e76f51", label="Class 1"),
        ],
        loc="outside upper right",
        frameon=False,
        ncols=2,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved discrete visualization to {output_path}")


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
