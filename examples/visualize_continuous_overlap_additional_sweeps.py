"""
Visualize two additional ContinuousOverlapIndex regression experiments.

The first row progressively removes heteroscedastic target noise from a smooth
regression field. The second row improves a two-dimensional observation of a
continuous oscillator state. Each row shows three representative datasets and
the mean score curve across deterministic repetitions. By default the figure
is written to ``img/continuous_overlap_additional_sweeps.png``.
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
DEFAULT_OUTPUT = ROOT / "img" / "continuous_overlap_additional_sweeps.png"
N_SAMPLES = 560
N_REPETITIONS = 3

Target = np.ndarray
DatasetFactory = Callable[[float, int], tuple[np.ndarray, Target]]
ColorFactory = Callable[[Target], np.ndarray]


def heteroscedastic_field(
    clean_field_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth field with a progressively shrinking noisy band."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(N_SAMPLES, 2))
    signal = 1.35 * X[:, 0] + 0.45 * np.sin(np.pi * X[:, 1])
    noisy_band_width = 1.0 - clean_field_fraction
    if noisy_band_width == 0.0:
        band_strength = np.zeros(N_SAMPLES)
    else:
        scaled_distance = np.abs(X[:, 1]) / noisy_band_width
        band_strength = np.exp(-(scaled_distance**4))
    target_scale = 0.07 + 1.05 * band_strength
    y = signal + target_scale * rng.normal(size=N_SAMPLES)
    return X, y


def multivariate_oscillator(
    observation_fidelity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return noisy feature observations of a continuous two-target state."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=N_SAMPLES)
    target = np.column_stack([np.cos(phase), np.sin(phase)])
    target += rng.normal(scale=0.025, size=target.shape)

    nuisance_phase = rng.uniform(-np.pi, np.pi, size=N_SAMPLES)
    nuisance_radius = np.sqrt(rng.uniform(0.0, 1.0, size=N_SAMPLES))
    nuisance = np.column_stack(
        [
            nuisance_radius * np.cos(nuisance_phase),
            nuisance_radius * np.sin(nuisance_phase),
        ]
    )
    nuisance_fraction = np.sqrt(max(0.0, 1.0 - observation_fidelity**2))
    X = observation_fidelity * target + nuisance_fraction * nuisance
    return X, target


def scalar_target_color(target: Target) -> np.ndarray:
    """Return scalar targets unchanged for point coloring."""
    return np.asarray(target, dtype=float)


def oscillator_phase(target: Target) -> np.ndarray:
    """Return the phase of a two-dimensional oscillator target."""
    target = np.asarray(target, dtype=float)
    return np.arctan2(target[:, 1], target[:, 0])


def make_model(seed: int) -> ContinuousOverlapIndex:
    """Return the deterministic offline estimator used in both sweeps."""
    return ContinuousOverlapIndex(
        model_type="KMeans",
        kmeans_k=8,
        kmeans_kwargs={"random_state": seed, "n_init": 3},
        target_cover="auto",
        n_target_cells=8,
        target_cover_kwargs={"n_init": 3},
        target_distance="auto",
        adjacency_mode="soft_topk",
        top_k=5,
        feature_temperature=0.25,
        null_mode="refit_permutation",
        n_null_permutations=6,
        n_projections=32,
        random_state=seed,
    )


def score_sweep(
    factory: DatasetFactory,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and standard deviation across repeated synthetic draws."""
    scores = np.empty((N_REPETITIONS, parameters.size), dtype=float)
    for repetition in range(N_REPETITIONS):
        seed = 300 + repetition
        for index, parameter in enumerate(parameters):
            X, target = factory(float(parameter), seed)
            scores[repetition, index] = make_model(seed).fit(X, target).index
    return scores.mean(axis=0), scores.std(axis=0)


def plot_row(
    axes: np.ndarray,
    name: str,
    parameter_name: str,
    factory: DatasetFactory,
    color_factory: ColorFactory,
    parameters: np.ndarray,
    limits: tuple[tuple[float, float], tuple[float, float]],
    cmap: str,
    color_limits: tuple[float, float],
) -> None:
    """Plot three exemplars and the response curve for one experiment."""
    mean, std = score_sweep(factory, parameters)
    exemplar_indices = (0, parameters.size // 2, parameters.size - 1)

    for column, parameter_index in enumerate(exemplar_indices):
        parameter = float(parameters[parameter_index])
        X, target = factory(parameter, seed=300)
        score = make_model(seed=300).fit(X, target).index
        ax = axes[column]
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=color_factory(target),
            cmap=cmap,
            vmin=color_limits[0],
            vmax=color_limits[1],
            s=7,
            alpha=0.72,
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
    curve_ax.plot(parameters, mean, color="#b45309", linewidth=2.2)
    curve_ax.fill_between(
        parameters,
        np.clip(mean - std, 0.0, 1.0),
        np.clip(mean + std, 0.0, 1.0),
        color="#b45309",
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
    """Build and save the additional continuous-overlap gallery."""
    experiments = (
        (
            "Heteroscedastic recovery",
            "Clean-field fraction",
            heteroscedastic_field,
            scalar_target_color,
            np.linspace(0.0, 1.0, 17),
            ((-1.08, 1.08), (-1.08, 1.08)),
            "viridis",
            (-2.5, 2.5),
        ),
        (
            "Multivariate oscillator",
            "Observation fidelity",
            multivariate_oscillator,
            oscillator_phase,
            np.linspace(0.0, 1.0, 17),
            ((-1.48, 1.48), (-1.48, 1.48)),
            "twilight",
            (-np.pi, np.pi),
        ),
    )

    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.2), constrained_layout=True)
    for row, experiment in enumerate(experiments):
        plot_row(axes[row], *experiment)

    fig.suptitle(
        "ContinuousOverlapIndex: distribution and multivariate sweeps",
        fontsize=16,
        fontweight="bold",
    )

    target_mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=-2.5, vmax=2.5),
        cmap="viridis",
    )
    target_colorbar = fig.colorbar(
        target_mappable,
        ax=axes[0, :3].ravel().tolist(),
        location="bottom",
        shrink=0.55,
        aspect=35,
        pad=0.06,
    )
    target_colorbar.set_label("Continuous target y")

    phase_mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=-np.pi, vmax=np.pi),
        cmap="twilight",
    )
    phase_colorbar = fig.colorbar(
        phase_mappable,
        ax=axes[1, :3].ravel().tolist(),
        location="bottom",
        shrink=0.55,
        aspect=35,
        pad=0.06,
        ticks=(-np.pi, 0.0, np.pi),
    )
    phase_colorbar.set_ticklabels(("-π", "0", "π"))
    phase_colorbar.set_label("Oscillator target phase")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved additional continuous visualization to {output_path}")


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
