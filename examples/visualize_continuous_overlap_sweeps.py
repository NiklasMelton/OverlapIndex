"""
Visualize ContinuousOverlapIndex on naturally continuous regression problems.

Three experiments sweep from an ambiguous representation to an informative
one:

* A smooth latent signal with increasing observation fidelity.
* A folded latent trajectory with increasing signed-coordinate retention.
* A regression problem with an increasingly observable continuous covariate.

Each row combines three representative target-colored datasets with the mean
ContinuousOverlapIndex curve across deterministic repetitions. By default the
figure is written to ``img/continuous_overlap_sweeps.png``.
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
N_SAMPLES = 560
N_REPETITIONS = 3
TARGET_LIMIT = 2.5

DatasetFactory = Callable[[float, int], tuple[np.ndarray, np.ndarray]]


def smooth_signal_recovery(
    observation_fidelity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a continuous signal observed through a progressively clean manifold."""
    rng = np.random.default_rng(seed)
    latent = rng.uniform(-1.0, 1.0, size=N_SAMPLES)
    feature_noise_1 = rng.uniform(-1.0, 1.0, size=N_SAMPLES)
    feature_noise_2 = rng.uniform(-1.0, 1.0, size=N_SAMPLES)
    noise_fraction = np.sqrt(max(0.0, 1.0 - observation_fidelity**2))
    observed_signal = (
        observation_fidelity * latent + noise_fraction * feature_noise_1
    )
    X = np.column_stack(
        [
            observed_signal,
            0.75 * observation_fidelity * np.sin(np.pi * latent)
            + (1.0 - observation_fidelity) * feature_noise_2,
        ]
    )
    y = 2.0 * latent
    return X, y


def folded_trajectory(
    signed_coordinate_retention: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a continuous latent trajectory that unfolds in feature space."""
    rng = np.random.default_rng(seed)
    latent = rng.uniform(-1.0, 1.0, size=N_SAMPLES)
    nuisance = rng.uniform(-1.0, 1.0, size=N_SAMPLES)
    X = np.column_stack(
        [
            (1.0 - signed_coordinate_retention) * latent**2
            + signed_coordinate_retention * latent,
            0.65 * signed_coordinate_retention * latent**3
            + 0.35 * (1.0 - signed_coordinate_retention) * nuisance,
        ]
    )
    y = 2.0 * latent
    return X, y


def observable_covariate(
    recovery_progress: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a regression problem with a progressively revealed covariate."""
    rng = np.random.default_rng(seed)
    observed = rng.uniform(-1.0, 1.0, size=N_SAMPLES)
    hidden = rng.uniform(-1.0, 1.0, size=N_SAMPLES)
    recovered_fraction = recovery_progress**2
    partial_signal = (
        observed
        + recovered_fraction * hidden
        + 0.18 * np.sin(3.0 * observed)
    )
    X = np.column_stack(
        [
            0.5 * partial_signal,
            0.6 * recovered_fraction * np.sin(1.5 * partial_signal)
            + 0.35 * (1.0 - recovered_fraction) * observed,
        ]
    )
    y = (
        observed
        + hidden
        + 0.18 * np.sin(3.0 * observed)
    )
    return X, y


def make_model(seed: int) -> ContinuousOverlapIndex:
    """Return the deterministic offline estimator used in the sweep."""
    return ContinuousOverlapIndex(
        model_type="KMeans",
        kmeans_k=8,
        kmeans_kwargs={"random_state": seed, "n_init": 3},
        n_target_cells=8,
        adjacency_mode="hard_top1",
        top_k=1,
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
            cmap="viridis",
            vmin=-TARGET_LIMIT,
            vmax=TARGET_LIMIT,
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
            "Smooth signal recovery",
            "Observation fidelity",
            smooth_signal_recovery,
            np.linspace(0.0, 1.0, 17),
            ((-1.08, 1.08), (-1.08, 1.08)),
        ),
        (
            "Folded trajectory",
            "Signed-coordinate retention",
            folded_trajectory,
            np.linspace(0.0, 1.0, 17),
            ((-1.08, 1.08), (-0.72, 0.72)),
        ),
        (
            "Observable covariate",
            "Recovery progress",
            observable_covariate,
            np.linspace(0.0, 1.0, 17),
            ((-1.15, 1.15), (-0.68, 0.68)),
        ),
    )

    fig, axes = plt.subplots(3, 4, figsize=(14.2, 10.4), constrained_layout=True)
    for row, experiment in enumerate(experiments):
        plot_row(axes[row], *experiment)

    fig.suptitle(
        "ContinuousOverlapIndex: continuous regression structure sweeps",
        fontsize=16,
        fontweight="bold",
    )
    scalar_mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=-TARGET_LIMIT, vmax=TARGET_LIMIT),
        cmap="viridis",
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
