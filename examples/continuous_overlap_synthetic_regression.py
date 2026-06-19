"""
Visual ContinuousOverlapIndex example on synthetic univariate regression data.

This script builds four 2D synthetic datasets with progressively less useful
feature-target alignment, fits ContinuousOverlapIndex on each one, and plots
the data together with the corresponding COI score on a single figure.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-overlapindex"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from overlapindex import ContinuousOverlapIndex


OUTPUT_PATH = Path(__file__).with_name("continuous_overlap_synthetic_regression.png")


def _base_feature_blocks(rng: np.random.Generator, points_per_block: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Return four compact 2D feature regions and their block ids."""
    centers = np.asarray(
        [
            [-2.4, -1.0],
            [-0.8, 1.1],
            [0.8, -1.1],
            [2.4, 1.0],
        ],
        dtype=float,
    )

    blocks = []
    block_ids = []
    for idx, center in enumerate(centers):
        cov = np.asarray([[0.22, 0.04], [0.04, 0.18]], dtype=float)
        block = rng.multivariate_normal(center, cov, size=points_per_block)
        blocks.append(block)
        block_ids.append(np.full(points_per_block, idx, dtype=int))

    X = np.vstack(blocks)
    block_ids_arr = np.concatenate(block_ids)
    X = MinMaxScaler().fit_transform(X)
    return X.astype(float), block_ids_arr


def _aligned_targets(block_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """High-COI monotonic target aligned with feature regions."""
    means = np.asarray([-2.2, -0.7, 0.8, 2.3], dtype=float)
    return means[block_ids] + rng.normal(scale=0.18, size=block_ids.shape[0])


def _moderate_overlap_targets(block_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mostly aligned target with partial swapping across neighboring regions."""
    y = _aligned_targets(block_ids, rng)
    swap_mask = rng.random(block_ids.shape[0]) < 0.28
    shifted_blocks = np.clip(
        block_ids[swap_mask] + rng.choice([-1, 1], size=swap_mask.sum()),
        0,
        3,
    )
    means = np.asarray([-2.2, -0.7, 0.8, 2.3], dtype=float)
    y[swap_mask] = means[shifted_blocks] + rng.normal(scale=0.28, size=swap_mask.sum())
    return y


def _null_like_targets(block_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Target independent of feature region, intended to land near the null."""
    del block_ids
    component_means = np.asarray([-2.0, -0.4, 0.5, 2.0], dtype=float)
    draws = rng.integers(0, component_means.size, size=320)
    return component_means[draws] + rng.normal(scale=0.35, size=draws.shape[0])


def _pathological_targets(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Adversarial target assignment that flips rapidly within local neighborhoods."""
    order = np.argsort(X[:, 1])
    alternating = np.tile(np.asarray([-2.5, 2.5], dtype=float), X.shape[0] // 2)
    if alternating.shape[0] < X.shape[0]:
        alternating = np.concatenate([alternating, np.asarray([-2.5], dtype=float)])

    y = np.empty(X.shape[0], dtype=float)
    y[order] = alternating[: X.shape[0]] + rng.normal(scale=0.02, size=X.shape[0])
    return y


def _make_examples() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build all synthetic datasets with deterministic randomness."""
    base_rng = np.random.default_rng(7)
    X, block_ids = _base_feature_blocks(base_rng)

    datasets = [
        ("Strong Separation", X, _aligned_targets(block_ids, np.random.default_rng(11))),
        ("Moderate Overlap", X, _moderate_overlap_targets(block_ids, np.random.default_rng(13))),
        ("Null-Like Assignment", X, _null_like_targets(block_ids, np.random.default_rng(17))),
        ("Pathological Overlap", X, _pathological_targets(X, np.random.default_rng(19))),
    ]
    return datasets


def _make_model() -> ContinuousOverlapIndex:
    """Return a deterministic COI configuration for the visual example."""
    return ContinuousOverlapIndex(
        model_type="KMeans",
        kmeans_k=6,
        kmeans_kwargs={"random_state": 0, "n_init": 10},
        n_target_cells=4,
        n_null_permutations=24,
        target_cover="quantile",
        target_distance="wasserstein",
        random_state=0,
    )


def main() -> None:
    datasets = _make_examples()
    all_y = np.concatenate([y for _, _, y in datasets])
    color_limits = (float(all_y.min()), float(all_y.max()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax, (name, X, y) in zip(axes, datasets):
        coi = _make_model()
        coi.fit(X, y)

        scatter = ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="coolwarm",
            vmin=color_limits[0],
            vmax=color_limits[1],
            s=28,
            linewidths=0.0,
            alpha=0.9,
        )
        ax.set_title(f"{name}\nCOI = {coi.index:.3f}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.18)

        ax.text(
            0.02,
            0.03,
            f"actual/null = {coi.actual_loss_:.3f}/{coi.null_loss_:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )

        print(
            f"{name:20s} COI={coi.index:.6f}  raw={coi.raw_index_:.6f}  "
            f"actual={coi.actual_loss_:.6f}  null={coi.null_loss_:.6f}"
        )

    cbar = fig.colorbar(scatter, ax=axes.tolist(), shrink=0.92)
    cbar.set_label("Continuous target y")
    fig.suptitle("Synthetic Univariate Regression Datasets and ContinuousOverlapIndex", fontsize=15)
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)

    print(f"\nSaved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
