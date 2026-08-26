"""Deterministic, private fixtures for nuisance-conditioned-distance runs.

This module is intentionally independent of the public :mod:`overlapindex`
API.  It contains two fixture layers:

* :class:`ConfirmConfig` / :func:`generate_confirm_dataset` reproduce the
  frozen Stage-2 nuisance battery recovered from the archived
  ``confirm_generators.py`` harness; and
* :class:`MechanisticConfig` / :func:`generate_mechanistic_dataset` provide
  small, explicit signal/nuisance cases for smoke tests.

Every returned dataset has independent labelled ``train`` and ``evaluation``
pools.  ``signal`` and ``nuisance`` are retained separately so a run can
inspect the true geometry without fitting a conditioner on held-out rows.
Labels are scalar integer values.  Pair truth is represented by a symmetric
``pair_overlap`` coefficient matrix and a binary ``pair_overlap_label``
matrix, with diagonal entries set to one and zero respectively.

The Stage-2 constants below are immutable protocol data, not tuning knobs.
The archived source/config identities are recorded in metadata and manifests
by the runner.  The source was readable at:

``/Users/niklasmelton/.codex/worktrees/cb8b/OverlapIndex/experiments/
synthetic_nuisance/confirm_generators.py``

with config ``configs/stage2_confirm.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


Array = np.ndarray

# Recovered archived identities.  Keeping these in the fixture makes an
# artifact auditable even if the old worktree is later unavailable.
ARCHIVED_STAGE2_GENERATOR_PATH = (
    "/Users/niklasmelton/.codex/worktrees/cb8b/OverlapIndex/experiments/"
    "synthetic_nuisance/confirm_generators.py"
)
ARCHIVED_STAGE2_CONFIG_PATH = (
    "/Users/niklasmelton/.codex/worktrees/cb8b/OverlapIndex/experiments/"
    "synthetic_nuisance/configs/stage2_confirm.json"
)
ARCHIVED_STAGE2_GENERATOR_SHA256 = (
    "277af6d6a6fd8fb8d613faeb4bf0bce845145b57c6a0862de05167efd5f0efb3"
)
ARCHIVED_STAGE2_CONFIG_SHA256 = (
    "66719e8f0a053a618e0c946dd34a346d34d31ef639287fba578dfbed599b6586"
)
ARCHIVED_STAGE2_SEEDS = tuple(range(1000, 1020))
ARCHIVED_STAGE2_NUISANCE_STRENGTHS = (0.0, 1.0, 2.0)
ARCHIVED_STAGE2_FAMILIES = (
    "shared_low_rank",
    "clustered_multimodal",
    "heteroscedastic_multiplicative",
)
ARCHIVED_STAGE2_K_VALUES = (2, 8)
ARCHIVED_STAGE2_N_PER_CLASS = 40
ARCHIVED_STAGE2_N_CLASSES = 4
ARCHIVED_STAGE2_SIGNAL_DIM = 2
ARCHIVED_STAGE2_NUISANCE_DIM = 4
ARCHIVED_STAGE2_OVERLAP_PAIRS = ((0, 1),)
ARCHIVED_STAGE2_SEED_ROLE = "stage2_confirm_untouched_1000_1019"

ARCHIVED_STAGE2_CONDITIONS: tuple[dict[str, Any], ...] = (
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


@dataclass(frozen=True)
class ConfirmConfig:
    """One Stage-2 confirmation generator configuration.

    This is a line-for-line semantic reproduction of the archived Stage-2
    generator configuration.  ``eval_seed`` is intentionally explicit: the
    full frozen suite derives it as ``train_seed + 10000``.
    """

    family: str
    signal_design: str = "selective_multiclass"
    n_per_class: int = ARCHIVED_STAGE2_N_PER_CLASS
    n_classes: int = ARCHIVED_STAGE2_N_CLASSES
    signal_dim: int = ARCHIVED_STAGE2_SIGNAL_DIM
    nuisance_dim: int = ARCHIVED_STAGE2_NUISANCE_DIM
    overlap_severity: float = 0.5
    signal_gap: float = 0.75
    nuisance_strength: float = 1.0
    balance: str = "balanced"
    nuisance_shift: bool = False
    train_seed: int = 1000
    # Matches the archived class default; the frozen full runner explicitly
    # derives this as ``train_seed + 10000`` for each seed row.
    eval_seed: int = 2000
    overlap_pairs: tuple[tuple[int, int], ...] = ARCHIVED_STAGE2_OVERLAP_PAIRS

    def __post_init__(self) -> None:
        if self.family not in ARCHIVED_STAGE2_FAMILIES:
            raise ValueError(f"unknown confirm nuisance family: {self.family!r}")
        if self.signal_design not in {"binary_overlap", "selective_multiclass"}:
            raise ValueError(f"unknown signal design: {self.signal_design!r}")
        if int(self.n_per_class) != self.n_per_class or self.n_per_class <= 0:
            raise ValueError("n_per_class must be a positive integer")
        if int(self.n_classes) != self.n_classes or self.n_classes < 2:
            raise ValueError("n_classes must be at least two")
        if self.signal_design == "selective_multiclass" and self.n_classes < 4:
            raise ValueError("selective_multiclass requires n_classes >= 4")
        if int(self.signal_dim) != self.signal_dim or self.signal_dim < 1:
            raise ValueError("signal_dim must be positive")
        if int(self.nuisance_dim) != self.nuisance_dim or self.nuisance_dim < 1:
            raise ValueError("nuisance_dim must be positive")
        if not np.isfinite(self.overlap_severity) or not 0.0 <= self.overlap_severity <= 1.0:
            raise ValueError("overlap_severity must lie in [0, 1]")
        if not np.isfinite(self.signal_gap) or self.signal_gap <= 0.0:
            raise ValueError("signal_gap must be finite and positive")
        if not np.isfinite(self.nuisance_strength) or self.nuisance_strength < 0.0:
            raise ValueError("nuisance_strength must be finite and non-negative")
        if self.balance not in {"balanced", "imbalanced"}:
            raise ValueError("balance must be 'balanced' or 'imbalanced'")
        pairs = tuple((int(a), int(b)) for a, b in self.overlap_pairs)
        if any(
            a == b or a < 0 or b < 0 or a >= self.n_classes or b >= self.n_classes
            for a, b in pairs
        ):
            raise ValueError("overlap_pairs contain invalid class ids")
        if len({tuple(sorted(pair)) for pair in pairs}) != len(pairs):
            raise ValueError("overlap_pairs must be unique")
        object.__setattr__(self, "overlap_pairs", pairs)


@dataclass
class DatasetSplit:
    """One labelled pool with observed and clean components retained."""

    X: Array
    y: Array
    signal: Array
    nuisance: Array
    clean_signal: Array | None = None
    nuisance_multiplier: Array | None = None

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


@dataclass
class SyntheticDataset:
    """Independent train/evaluation pools and pair-level generator truth."""

    family: str
    train: DatasetSplit
    evaluation: DatasetSplit
    ground_truth: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def eval(self) -> DatasetSplit:
        return self.evaluation

    @property
    def truth(self) -> dict[str, Any]:
        return self.ground_truth


# Historical Stage-2 callers used these names.  The aliases preserve the
# recovered fixture identity without maintaining a second implementation.
ConfirmSplit = DatasetSplit
ConfirmDataset = SyntheticDataset


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def _counts(config: ConfirmConfig) -> np.ndarray:
    weights = np.ones(config.n_classes, dtype=float)
    if config.balance == "imbalanced":
        weights[0] = 2.0
    return np.asarray(np.rint(config.n_per_class * weights), dtype=int)


def _centers(config: ConfirmConfig) -> Array:
    centers = np.zeros((config.n_classes, config.signal_dim), dtype=float)
    base = 0.40 + float(config.signal_gap)
    centers[0, 0] = -base
    centers[1, 0] = base
    if config.signal_design == "selective_multiclass":
        for class_id in range(2, config.n_classes):
            centers[class_id, 0] = (class_id + 1.5) * base
    return centers


def _component_signal(
    rng: np.random.Generator,
    class_id: int,
    count: int,
    config: ConfirmConfig,
    centers: Array,
) -> tuple[Array, Array]:
    half_width = 0.20
    values = np.empty((count, config.signal_dim), dtype=float)
    component = np.ones(count, dtype=np.int8)
    pair = tuple(sorted((0, 1)))
    shared = class_id in pair and pair in {tuple(sorted(p)) for p in config.overlap_pairs}
    n_shared = int(np.rint(config.overlap_severity * count)) if shared else 0
    if n_shared:
        component[:n_shared] = 0
        values[:n_shared] = rng.uniform(-half_width, half_width, size=(n_shared, config.signal_dim))
    if n_shared < count:
        values[n_shared:] = centers[class_id] + rng.uniform(
            -half_width, half_width, size=(count - n_shared, config.signal_dim)
        )
    order = rng.permutation(count)
    return values[order], component[order]


def _signal_pool(
    rng: np.random.Generator,
    config: ConfirmConfig,
    centers: Array,
) -> tuple[Array, Array, Array, Array]:
    counts = _counts(config)
    signals: list[Array] = []
    labels: list[Array] = []
    components: list[Array] = []
    for class_id, count in enumerate(counts.tolist()):
        values, component = _component_signal(rng, class_id, int(count), config, centers)
        signals.append(values)
        components.append(component)
        labels.append(np.full(int(count), class_id, dtype=np.int64))
    signal = np.concatenate(signals, axis=0)
    y = np.concatenate(labels, axis=0)
    component_ids = np.concatenate(components, axis=0)
    order = rng.permutation(y.size)
    return signal[order], y[order], component_ids[order], counts


def _shared_low_rank(
    rng: np.random.Generator,
    n: int,
    dim: int,
    strength: float,
    shifted: bool,
) -> Array:
    if float(strength) == 0.0:
        return np.zeros((n, dim), dtype=float)
    rank = min(2, dim)
    loadings = np.zeros((rank, dim), dtype=float)
    loadings[:, :rank] = np.eye(rank)
    if dim > rank:
        loadings[:, rank:] = rng.normal(scale=0.35, size=(rank, dim - rank))
    latent_scale = float(strength) * (1.75 if shifted else 1.0)
    latent_mean = 0.30 * float(strength) if shifted else 0.0
    latent = rng.normal(loc=latent_mean, scale=latent_scale, size=(n, rank))
    noise = rng.normal(scale=0.05 * float(strength), size=(n, dim))
    return latent @ loadings + noise


def _clustered_multimodal(
    rng: np.random.Generator,
    n: int,
    dim: int,
    strength: float,
    shifted: bool,
) -> Array:
    if float(strength) == 0.0:
        return np.zeros((n, dim), dtype=float)
    clusters = 3
    ids = rng.integers(0, clusters, size=n)
    centers = np.zeros((clusters, dim), dtype=float)
    centers[:, 0] = (np.arange(clusters, dtype=float) - 1.0) * 2.0 * float(strength)
    if shifted:
        centers[:, 0] += 0.7 * float(strength)
    return centers[ids] + rng.normal(scale=0.08 * float(strength), size=(n, dim))


def _heteroscedastic_multiplicative(
    rng: np.random.Generator,
    n: int,
    dim: int,
    strength: float,
    shifted: bool,
) -> tuple[Array, Array]:
    if float(strength) == 0.0:
        return np.zeros((n, dim), dtype=float), np.ones(n, dtype=float)
    log_scale = 0.22 * float(strength) * (1.8 if shifted else 1.0)
    multipliers = np.exp(rng.normal(loc=0.0, scale=log_scale, size=n))
    nuisance = rng.normal(scale=0.20 * float(strength), size=(n, dim))
    return nuisance, multipliers


def _nuisance_pool(
    rng: np.random.Generator,
    config: ConfirmConfig,
    n: int,
    shifted: bool,
) -> tuple[Array, Array | None]:
    if config.family == "shared_low_rank":
        return _shared_low_rank(rng, n, config.nuisance_dim, config.nuisance_strength, shifted), None
    if config.family == "clustered_multimodal":
        return _clustered_multimodal(rng, n, config.nuisance_dim, config.nuisance_strength, shifted), None
    return _heteroscedastic_multiplicative(rng, n, config.nuisance_dim, config.nuisance_strength, shifted)


def _confirm_truth(config: ConfirmConfig, centers: Array) -> dict[str, Any]:
    overlap = np.zeros((config.n_classes, config.n_classes), dtype=float)
    np.fill_diagonal(overlap, 1.0)
    for source, target in config.overlap_pairs:
        overlap[source, target] = overlap[target, source] = float(config.overlap_severity)
    distance = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    labels = overlap > 0.0
    np.fill_diagonal(labels, False)
    positive_gap = np.ones_like(labels, dtype=bool)
    np.fill_diagonal(positive_gap, False)
    for source, target in config.overlap_pairs:
        if config.overlap_severity > 0.0:
            positive_gap[source, target] = positive_gap[target, source] = False
    return {
        "pair_overlap": overlap,
        "truth_pair_overlap": overlap,
        "pair_overlap_label": labels.astype(int),
        "truth_overlap_label": labels.astype(int),
        "pair_positive_gap": positive_gap.astype(int),
        "pair_signal_distance": distance,
        "overlap_pairs": [list(pair) for pair in config.overlap_pairs],
        "overlap_severity": float(config.overlap_severity),
        "signal_gap": float(config.signal_gap),
        "global_overlap": float(np.mean(overlap[~np.eye(config.n_classes, dtype=bool)])),
        "overlap_definition": (
            "shared compact component mass for predeclared pairs; all other "
            "pairs have strict positive support gaps"
        ),
    }


def generate_confirm_dataset(config: ConfirmConfig | Mapping[str, Any]) -> SyntheticDataset:
    """Generate one exact archived Stage-2 style dataset."""

    if not isinstance(config, ConfirmConfig):
        config = ConfirmConfig(**dict(config))
    centers = _centers(config)
    train_rng = _rng(config.train_seed, 0)
    eval_rng = _rng(config.eval_seed, 1)
    train_signal, train_y, train_component, counts = _signal_pool(train_rng, config, centers)
    eval_signal, eval_y, eval_component, _ = _signal_pool(eval_rng, config, centers)
    train_nuisance, train_multiplier = _nuisance_pool(train_rng, config, train_signal.shape[0], False)
    eval_nuisance, eval_multiplier = _nuisance_pool(eval_rng, config, eval_signal.shape[0], config.nuisance_shift)
    observed_train = train_signal if train_multiplier is None else train_signal * train_multiplier[:, None]
    observed_eval = eval_signal if eval_multiplier is None else eval_signal * eval_multiplier[:, None]
    train = DatasetSplit(
        np.concatenate((observed_train, train_nuisance), axis=1),
        train_y,
        observed_train,
        train_nuisance,
        train_signal,
        train_multiplier,
    )
    evaluation = DatasetSplit(
        np.concatenate((observed_eval, eval_nuisance), axis=1),
        eval_y,
        observed_eval,
        eval_nuisance,
        eval_signal,
        eval_multiplier,
    )
    truth = _confirm_truth(config, centers)
    metadata: dict[str, Any] = {
        "family": config.family,
        "signal_design": config.signal_design,
        "n_classes": int(config.n_classes),
        "n_features": int(config.signal_dim + config.nuisance_dim),
        "class_counts": counts.tolist(),
        "balance": config.balance,
        "nuisance_shift": bool(config.nuisance_shift),
        "nuisance_strength": float(config.nuisance_strength),
        "nuisance_label_independent": True,
        "train_eval_independent": True,
        "train_seed": int(config.train_seed),
        "eval_seed": int(config.eval_seed),
        "signal_centers": centers.tolist(),
        "overlap_pairs": [list(pair) for pair in config.overlap_pairs],
        "signal_component_ids_train": train_component,
        "signal_component_ids_eval": eval_component,
        "archived_generator_path": ARCHIVED_STAGE2_GENERATOR_PATH,
        "archived_generator_sha256": ARCHIVED_STAGE2_GENERATOR_SHA256,
        "archived_config_path": ARCHIVED_STAGE2_CONFIG_PATH,
        "archived_config_sha256": ARCHIVED_STAGE2_CONFIG_SHA256,
        "archived_seed_role": ARCHIVED_STAGE2_SEED_ROLE,
    }
    return SyntheticDataset(config.family, train, evaluation, truth, metadata)


def archived_stage2_configurations() -> tuple[ConfirmConfig, ...]:
    """Return the immutable 2160-case Stage-2 configuration grid."""

    configurations: list[ConfirmConfig] = []
    for family in ARCHIVED_STAGE2_FAMILIES:
        for condition in ARCHIVED_STAGE2_CONDITIONS:
            for nuisance_strength in ARCHIVED_STAGE2_NUISANCE_STRENGTHS:
                for seed in ARCHIVED_STAGE2_SEEDS:
                    for _k in ARCHIVED_STAGE2_K_VALUES:
                        # ``k`` is a runner parameter, but returning one config
                        # per k keeps this helper a complete case manifest.
                        configurations.append(
                            ConfirmConfig(
                                family=family,
                                n_per_class=ARCHIVED_STAGE2_N_PER_CLASS,
                                n_classes=ARCHIVED_STAGE2_N_CLASSES,
                                signal_dim=ARCHIVED_STAGE2_SIGNAL_DIM,
                                nuisance_dim=ARCHIVED_STAGE2_NUISANCE_DIM,
                                overlap_severity=float(condition["overlap_severity"]),
                                signal_gap=float(condition["signal_gap"]),
                                nuisance_strength=float(nuisance_strength),
                                balance=str(condition["balance"]),
                                nuisance_shift=bool(condition["nuisance_shift"]),
                                train_seed=int(seed),
                                eval_seed=int(seed) + 10000,
                            )
                        )
    return tuple(configurations)


def archived_stage2_case_grid() -> tuple[dict[str, Any], ...]:
    """Return case dictionaries including ``family``, condition, and ``k``."""

    rows: list[dict[str, Any]] = []
    for family in ARCHIVED_STAGE2_FAMILIES:
        for condition in ARCHIVED_STAGE2_CONDITIONS:
            for nuisance_strength in ARCHIVED_STAGE2_NUISANCE_STRENGTHS:
                for seed in ARCHIVED_STAGE2_SEEDS:
                    for k in ARCHIVED_STAGE2_K_VALUES:
                        row = dict(condition)
                        row.update(
                            {
                                "family": family,
                                "nuisance_strength": float(nuisance_strength),
                                "seed": int(seed),
                                "train_seed": int(seed),
                                "eval_seed": int(seed) + 10000,
                                "k": int(k),
                            }
                        )
                        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class MechanisticConfig:
    """Compact smoke fixture configuration.

    ``nuisance_kind`` covers appended isotropic, anisotropic, and correlated
    low-rank nuisance.  ``geometry`` covers linearly separated/overlapping
    classes, concentric rings, and XOR-style nonlinear separation.  The
    parameters deliberately expose dimensionality, imbalance, sample count,
    and overlap so a smoke matrix can vary them without hidden defaults.
    """

    geometry: str = "linear"
    nuisance_kind: str = "isotropic"
    n_per_class: int = 24
    n_classes: int = 2
    signal_dim: int = 2
    nuisance_dim: int = 4
    signal_gap: float = 1.0
    overlap_severity: float = 0.0
    nuisance_strength: float = 0.0
    imbalance: bool = False
    train_seed: int = 7
    eval_seed: int = 10007
    nuisance_shift: bool = False

    def __post_init__(self) -> None:
        if self.geometry not in {"linear", "rings", "xor"}:
            raise ValueError("geometry must be one of {'linear', 'rings', 'xor'}")
        if self.nuisance_kind not in {"none", "isotropic", "anisotropic", "correlated_low_rank"}:
            raise ValueError("unknown nuisance_kind")
        if self.geometry in {"rings", "xor"} and self.n_classes != 2:
            raise ValueError("rings and xor smoke geometries require n_classes=2")
        if self.geometry in {"rings", "xor"} and self.signal_dim < 2:
            raise ValueError("rings and xor require signal_dim >= 2")
        if int(self.n_per_class) != self.n_per_class or self.n_per_class <= 0:
            raise ValueError("n_per_class must be a positive integer")
        if int(self.n_classes) != self.n_classes or self.n_classes < 2:
            raise ValueError("n_classes must be at least two")
        if int(self.signal_dim) != self.signal_dim or self.signal_dim < 1:
            raise ValueError("signal_dim must be positive")
        if int(self.nuisance_dim) != self.nuisance_dim or self.nuisance_dim < 0:
            raise ValueError("nuisance_dim must be non-negative")
        if not np.isfinite(self.signal_gap) or self.signal_gap <= 0.0:
            raise ValueError("signal_gap must be positive")
        if not np.isfinite(self.overlap_severity) or not 0.0 <= self.overlap_severity <= 1.0:
            raise ValueError("overlap_severity must lie in [0, 1]")
        if not np.isfinite(self.nuisance_strength) or self.nuisance_strength < 0.0:
            raise ValueError("nuisance_strength must be non-negative")


def _mechanistic_counts(config: MechanisticConfig) -> np.ndarray:
    values = np.full(config.n_classes, int(config.n_per_class), dtype=int)
    if config.imbalance:
        values[0] *= 2
    return values


def _linear_signal(
    rng: np.random.Generator,
    config: MechanisticConfig,
    counts: np.ndarray,
) -> tuple[Array, Array, Array]:
    centers = np.zeros((config.n_classes, config.signal_dim), dtype=float)
    positions = (np.arange(config.n_classes) - (config.n_classes - 1) / 2.0) * (
        0.5 + float(config.signal_gap)
    )
    centers[:, 0] = positions
    signals: list[Array] = []
    labels: list[Array] = []
    component_ids: list[Array] = []
    for class_id, count in enumerate(counts.tolist()):
        private_count = int(round(count * (1.0 - config.overlap_severity)))
        shared_count = int(count - private_count)
        values: list[Array] = []
        ids: list[Array] = []
        if shared_count:
            values.append(rng.uniform(-0.20, 0.20, size=(shared_count, config.signal_dim)))
            ids.append(np.zeros(shared_count, dtype=np.int8))
        if private_count:
            values.append(
                centers[class_id] + rng.uniform(-0.20, 0.20, size=(private_count, config.signal_dim))
            )
            ids.append(np.ones(private_count, dtype=np.int8))
        class_values = np.concatenate(values, axis=0)
        class_ids = np.concatenate(ids, axis=0)
        order = rng.permutation(count)
        signals.append(class_values[order])
        component_ids.append(class_ids[order])
        labels.append(np.full(count, class_id, dtype=np.int64))
    signal = np.concatenate(signals, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    components = np.concatenate(component_ids, axis=0)
    order = rng.permutation(labels_array.size)
    return signal[order], labels_array[order], components[order]


def _nonlinear_signal(
    rng: np.random.Generator,
    config: MechanisticConfig,
    counts: np.ndarray,
) -> tuple[Array, Array, Array, str]:
    n = int(np.sum(counts))
    if config.geometry == "xor":
        values = rng.uniform(-1.0, 1.0, size=(n, 2))
        labels = ((values[:, 0] > 0.0) ^ (values[:, 1] > 0.0)).astype(np.int64)
        # Deterministically balance/truncate to requested class counts.
        selected: list[int] = []
        for class_id, count in enumerate(counts.tolist()):
            ids = np.flatnonzero(labels == class_id)
            if ids.size < count:
                extra = rng.choice(ids, size=count - ids.size, replace=True)
                ids = np.concatenate((ids, extra))
            selected.extend(ids[:count].tolist())
        selected_array = np.asarray(selected, dtype=int)
        labels = labels[selected_array]
        values = values[selected_array]
        return values, labels, np.ones(labels.size, dtype=np.int8), "xor"
    # Concentric radial bands: the class is the radius, not a linear axis.
    values: list[Array] = []
    labels: list[Array] = []
    components: list[Array] = []
    for class_id, count in enumerate(counts.tolist()):
        private = int(round(count * (1.0 - config.overlap_severity)))
        shared = int(count - private)
        radii: list[Array] = []
        ids: list[Array] = []
        if shared:
            radii.append(rng.uniform(0.85, 1.15, size=shared))
            ids.append(np.zeros(shared, dtype=np.int8))
        if private:
            base = 0.55 if class_id == 0 else 1.65
            radii.append(rng.uniform(base - 0.12, base + 0.12, size=private))
            ids.append(np.ones(private, dtype=np.int8))
        radius = np.concatenate(radii)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
        points = np.zeros((count, config.signal_dim), dtype=float)
        points[:, 0] = radius * np.cos(angles)
        points[:, 1] = radius * np.sin(angles)
        order = rng.permutation(count)
        values.append(points[order])
        labels.append(np.full(count, class_id, dtype=np.int64))
        components.append(np.concatenate(ids)[order])
    signal = np.concatenate(values, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    components_array = np.concatenate(components, axis=0)
    order = rng.permutation(labels_array.size)
    return signal[order], labels_array[order], components_array[order], "rings"


def _mechanistic_nuisance(
    rng: np.random.Generator,
    config: MechanisticConfig,
    n: int,
    shifted: bool,
) -> Array:
    if config.nuisance_dim == 0 or config.nuisance_kind == "none" or config.nuisance_strength == 0.0:
        return np.zeros((n, config.nuisance_dim), dtype=float)
    strength = float(config.nuisance_strength)
    shift = 0.5 * strength if shifted else 0.0
    if config.nuisance_kind == "isotropic":
        return rng.normal(loc=shift, scale=strength, size=(n, config.nuisance_dim))
    if config.nuisance_kind == "anisotropic":
        scales = np.geomspace(0.25, 4.0, config.nuisance_dim) * strength
        return rng.normal(loc=shift, scale=scales, size=(n, config.nuisance_dim))
    rank = min(2, config.nuisance_dim)
    latent = rng.normal(loc=shift, scale=strength, size=(n, rank))
    loadings = np.zeros((rank, config.nuisance_dim), dtype=float)
    loadings[:, :rank] = np.eye(rank)
    if config.nuisance_dim > rank:
        loadings[:, rank:] = np.linspace(0.5, 1.0, config.nuisance_dim - rank)[None, :]
    return latent @ loadings + rng.normal(scale=0.05 * strength, size=(n, config.nuisance_dim))


def generate_mechanistic_dataset(config: MechanisticConfig | Mapping[str, Any] = MechanisticConfig()) -> SyntheticDataset:
    """Generate a compact deterministic smoke fixture with explicit truth."""

    if not isinstance(config, MechanisticConfig):
        config = MechanisticConfig(**dict(config))
    counts = _mechanistic_counts(config)
    train_rng = _rng(config.train_seed, 10)
    eval_rng = _rng(config.eval_seed, 11)
    if config.geometry == "linear":
        train_signal, train_y, train_component = _linear_signal(train_rng, config, counts)
        eval_signal, eval_y, eval_component = _linear_signal(eval_rng, config, counts)
    else:
        train_signal, train_y, train_component, _ = _nonlinear_signal(train_rng, config, counts)
        eval_signal, eval_y, eval_component, _ = _nonlinear_signal(eval_rng, config, counts)
    train_nuisance = _mechanistic_nuisance(train_rng, config, train_signal.shape[0], False)
    eval_nuisance = _mechanistic_nuisance(eval_rng, config, eval_signal.shape[0], config.nuisance_shift)
    train = DatasetSplit(
        np.concatenate((train_signal, train_nuisance), axis=1),
        train_y,
        train_signal,
        train_nuisance,
        train_signal.copy(),
    )
    evaluation = DatasetSplit(
        np.concatenate((eval_signal, eval_nuisance), axis=1),
        eval_y,
        eval_signal,
        eval_nuisance,
        eval_signal.copy(),
    )
    n_classes = int(config.n_classes)
    overlap = np.zeros((n_classes, n_classes), dtype=float)
    np.fill_diagonal(overlap, 1.0)
    if config.geometry == "linear":
        overlap[0, 1] = overlap[1, 0] = float(config.overlap_severity)
    elif config.overlap_severity:
        overlap[0, 1] = overlap[1, 0] = float(config.overlap_severity)
    overlap_label = (overlap > 0.0).astype(int)
    np.fill_diagonal(overlap_label, 0)
    truth = {
        "pair_overlap": overlap,
        "truth_pair_overlap": overlap,
        "pair_overlap_label": overlap_label,
        "truth_overlap_label": overlap_label,
        "overlap_severity": float(config.overlap_severity),
        "global_overlap": float(np.mean(overlap[~np.eye(n_classes, dtype=bool)])),
        "overlap_definition": "predeclared shared signal component mass; smoke geometry truth",
    }
    metadata = {
        "geometry": config.geometry,
        "nuisance_kind": config.nuisance_kind,
        "nuisance_strength": float(config.nuisance_strength),
        "nuisance_shift": bool(config.nuisance_shift),
        "nuisance_label_independent": True,
        "train_eval_independent": True,
        "class_counts": counts.tolist(),
        "signal_dim": int(config.signal_dim),
        "nuisance_dim": int(config.nuisance_dim),
        "train_seed": int(config.train_seed),
        "eval_seed": int(config.eval_seed),
        "signal_component_ids_train": train_component,
        "signal_component_ids_eval": eval_component,
    }
    return SyntheticDataset("mechanistic", train, evaluation, truth, metadata)


# Explicit aliases make the fixture convenient from notebooks and keep the
# runner independent of any historical generator module names.
make_mechanistic_fixture = generate_mechanistic_dataset


def generate_shared_low_rank(**kwargs: Any) -> SyntheticDataset:
    return generate_confirm_dataset(ConfirmConfig(family="shared_low_rank", **kwargs))


def generate_clustered_multimodal(**kwargs: Any) -> SyntheticDataset:
    return generate_confirm_dataset(ConfirmConfig(family="clustered_multimodal", **kwargs))


def generate_heteroscedastic_multiplicative(**kwargs: Any) -> SyntheticDataset:
    return generate_confirm_dataset(
        ConfirmConfig(family="heteroscedastic_multiplicative", **kwargs)
    )

GENERATOR_REGISTRY = {
    "shared_low_rank": generate_shared_low_rank,
    "clustered_multimodal": generate_clustered_multimodal,
    "heteroscedastic_multiplicative": generate_heteroscedastic_multiplicative,
}


__all__ = [
    "ARCHIVED_STAGE2_CONFIG_PATH",
    "ARCHIVED_STAGE2_CONFIG_SHA256",
    "ARCHIVED_STAGE2_CONDITIONS",
    "ARCHIVED_STAGE2_FAMILIES",
    "ARCHIVED_STAGE2_GENERATOR_PATH",
    "ARCHIVED_STAGE2_GENERATOR_SHA256",
    "ARCHIVED_STAGE2_K_VALUES",
    "ARCHIVED_STAGE2_N_CLASSES",
    "ARCHIVED_STAGE2_N_PER_CLASS",
    "ARCHIVED_STAGE2_NUISANCE_DIM",
    "ARCHIVED_STAGE2_NUISANCE_STRENGTHS",
    "ARCHIVED_STAGE2_SEEDS",
    "ARCHIVED_STAGE2_SEED_ROLE",
    "ARCHIVED_STAGE2_SIGNAL_DIM",
    "ConfirmConfig",
    "ConfirmDataset",
    "ConfirmSplit",
    "DatasetSplit",
    "GENERATOR_REGISTRY",
    "MechanisticConfig",
    "SyntheticDataset",
    "archived_stage2_case_grid",
    "archived_stage2_configurations",
    "generate_clustered_multimodal",
    "generate_confirm_dataset",
    "generate_heteroscedastic_multiplicative",
    "generate_mechanistic_dataset",
    "generate_shared_low_rank",
    "make_mechanistic_fixture",
]
