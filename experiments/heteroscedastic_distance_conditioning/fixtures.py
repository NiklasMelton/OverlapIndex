"""Deterministic latent banks and case fixtures for the follow-up experiment.

The fixture layer is deliberately independent of the selector and of
``OverlapIndex``.  It owns only the frozen data-generating recipe from
``protocol.json``.  In particular, training and held-out banks are separate,
small cells are prefixes of the maximum banks, and every scenario in a case
panel reuses the same latent rows and row order.

Confirmation *banks* are protected by an explicit authorization object.  Case
identity planning is harmless and is available before confirmation, but no
confirmation array is allocated until a caller has validated the prerequisite
decision artifacts through :mod:`manifest`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import ndtri
from scipy.stats import t as student_t


Array = np.ndarray


# ---------------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------------

PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
PROTOCOL_SHA256 = "1e75e24b626531bafc65d597d6f7bc1d14f05030a6ffe7457665d5afd739d15e"

SCENARIO_ORDER: Tuple[str, ...] = ("S0", "S1", "S2", "H1", "H2", "T", "C", "X", "ALL")
SIGNAL_STATE_ORDER: Tuple[str, ...] = (
    "linear_separated",
    "nonlinear_separated",
    "genuine_overlap_half",
)
BALANCE_ORDER: Tuple[str, ...] = ("balanced", "imbalanced")
COUNT_LEVEL_ORDER: Tuple[str, ...] = ("small", "large")
NUISANCE_DIM_ORDER: Tuple[int, ...] = (4, 32)
K_ORDER: Tuple[int, ...] = (2, 8)

# Readable immutable views used by manifest/runner code.  They are values, not
# mutable registries: changing a local copy cannot alter the recipe.
SCENARIOS = SCENARIO_ORDER
SIGNAL_STATES = SIGNAL_STATE_ORDER

N_CLASSES = 4
SIGNAL_DIM = 2
MAX_ROWS_PER_CLASS = 320
MAX_ROWS = MAX_ROWS_PER_CLASS
DEVELOPMENT_SEEDS: Tuple[int, ...] = tuple(range(21000, 21012))
CONFIRMATION_SEEDS: Tuple[int, ...] = tuple(range(31000, 31024))

STREAM_CODES: Mapping[str, int] = {
    "signal_train": 100,
    "signal_evaluation": 101,
    "uniform_nuisance_train": 200,
    "uniform_nuisance_evaluation": 201,
    "contamination_mask_train": 300,
    "contamination_mask_evaluation": 301,
    "outlier_gaussian_train": 400,
    "outlier_gaussian_evaluation": 401,
    "row_order_train": 500,
    "row_order_evaluation": 501,
}

SIGNAL_STATE_INDEX: Mapping[str, int] = {
    "linear_separated": 0,
    "nonlinear_separated": 1,
    "genuine_overlap_half": 2,
}
BALANCE_INDEX: Mapping[str, int] = {"balanced": 0, "imbalanced": 1}
COUNT_LEVEL_INDEX: Mapping[str, int] = {"small": 0, "large": 1}

# Development scale and size assignments are part of the protocol, not
# generated from Python's hash function.
DEVELOPMENT_SCALE_RANKS: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (0, 3, 2, 1),
    (1, 0, 3, 2),
    (2, 1, 0, 3),
    (3, 2, 1, 0),
    (0, 2, 1, 3),
    (1, 3, 2, 0),
    (2, 0, 3, 1),
    (3, 1, 0, 2),
)
DEVELOPMENT_SIZE_RANKS: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (0, 1, 2, 3),
    (0, 1, 2, 3),
    (1, 0, 3, 2),
    (2, 0, 3, 1),
    (3, 2, 1, 0),
    (3, 2, 1, 0),
    (3, 2, 1, 0),
    (1, 3, 0, 2),
    (2, 3, 0, 1),
    (1, 3, 0, 2),
    (2, 0, 3, 1),
)
CONFIRMATION_SCALE_RANKS: Tuple[Tuple[int, ...], ...] = tuple(itertools.permutations(range(4)))
CONFIRMATION_SIZE_SHIFTS: Tuple[int, ...] = (
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    2,
    3,
    3,
    1,
    1,
    3,
    1,
    3,
    2,
    3,
    3,
    2,
    2,
)

SMALL_COUNTS = {"balanced": (40, 40, 40, 40), "imbalanced": (16, 24, 40, 80)}
LARGE_COUNTS = {"balanced": (160, 160, 160, 160), "imbalanced": (64, 96, 160, 320)}

# (amplitude, ratio, base distribution, held-out assignment)
SCENARIO_BLOCKS: Mapping[str, Mapping[str, Any]] = {
    "S0": {"amplitude": 0.0, "scale_ratio": 1.0, "distribution": "gaussian", "shift": "stable"},
    "S1": {"amplitude": 1.0, "scale_ratio": 1.0, "distribution": "gaussian", "shift": "stable"},
    "S2": {"amplitude": 2.0, "scale_ratio": 1.0, "distribution": "gaussian", "shift": "stable"},
    "H1": {"amplitude": 1.0, "scale_ratio": 2.0, "distribution": "gaussian", "shift": "stable"},
    "H2": {"amplitude": 2.0, "scale_ratio": 4.0, "distribution": "gaussian", "shift": "stable"},
    "T": {"amplitude": 2.0, "scale_ratio": 4.0, "distribution": "student_t5", "shift": "stable"},
    "C": {"amplitude": 2.0, "scale_ratio": 4.0, "distribution": "contaminated", "shift": "stable"},
    "X": {"amplitude": 2.0, "scale_ratio": 4.0, "distribution": "gaussian", "shift": "reversed"},
    "ALL": {"amplitude": 2.0, "scale_ratio": 4.0, "distribution": "contaminated", "shift": "reversed"},
}

METHOD_IDS: Tuple[str, ...] = (
    "A",
    "B",
    "L",
    "P25",
    "P50-SW",
    "P50-CB",
    "W50-SW",
    "W50-CB",
    "M50-SW",
    "M50-CB",
)
DEVELOPMENT_METHODS = METHOD_IDS
SMOKE_METHODS = METHOD_IDS
CONFIRMATION_BASE_METHODS: Tuple[str, ...] = ("A", "B", "L", "P50-SW", "W50-SW", "W50-CB")


# ---------------------------------------------------------------------------
# Small immutable data records
# ---------------------------------------------------------------------------


def _readonly(array: Array) -> Array:
    result = np.asarray(array)
    result.setflags(write=False)
    return result


def _readonly_mapping(mapping: Mapping[str, Array]) -> Dict[str, Array]:
    return {str(key): _readonly(value) for key, value in mapping.items()}


@dataclass(frozen=True)
class DatasetSplit:
    """Observed data plus latent components for one independent split."""

    X: Array
    y: Array
    signal: Array
    nuisance: Array
    clean_signal: Array
    component_ids: Array
    contamination_mask: Optional[Array] = None

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def features(self) -> Array:
        return self.X


@dataclass(frozen=True)
class SyntheticDataset:
    """A generated case with independent train/evaluation pools and truth."""

    case: "CaseSpec"
    train: DatasetSplit
    evaluation: DatasetSplit
    truth: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def eval(self) -> DatasetSplit:
        return self.evaluation

    @property
    def ground_truth(self) -> Mapping[str, Any]:
        return self.truth


@dataclass(frozen=True)
class NuisanceBank:
    """Maximum-size nuisance bank for one split."""

    uniform: Array
    contamination_mask: Array
    outlier_gaussian: Array


@dataclass(frozen=True)
class LatentBanks:
    """All maximum-size signal and nuisance banks for one seed."""

    seed: int
    signal_train: Mapping[str, Array]
    signal_evaluation: Mapping[str, Array]
    component_train: Mapping[str, Array]
    component_evaluation: Mapping[str, Array]
    nuisance_train: NuisanceBank
    nuisance_evaluation: NuisanceBank


@dataclass(frozen=True)
class CaseSpec:
    """One exact case identity in smoke, development, or confirmation."""

    stage: str
    seed: int
    scenario: str
    balance: str
    count_level: str
    nuisance_dim: int
    k: int
    signal_state: str
    scale_ranks: Tuple[int, ...]
    size_ranks: Tuple[int, ...]

    def __post_init__(self) -> None:
        stage = str(self.stage).lower()
        if stage not in {"smoke", "development", "confirmation"}:
            raise ValueError("stage must be one of {'smoke', 'development', 'confirmation'}")
        if self.scenario not in SCENARIO_BLOCKS:
            raise ValueError("unknown scenario block")
        if self.balance not in BALANCE_ORDER:
            raise ValueError("balance must be 'balanced' or 'imbalanced'")
        if self.count_level not in COUNT_LEVEL_ORDER:
            raise ValueError("count_level must be 'small' or 'large'")
        if int(self.nuisance_dim) != self.nuisance_dim or int(self.nuisance_dim) not in NUISANCE_DIM_ORDER:
            raise ValueError("nuisance_dim must be 4 or 32")
        if int(self.k) != self.k or int(self.k) not in K_ORDER:
            raise ValueError("k must be 2 or 8")
        if self.signal_state not in SIGNAL_STATE_INDEX:
            raise ValueError("unknown signal_state")
        scale = tuple(int(v) for v in self.scale_ranks)
        size = tuple(int(v) for v in self.size_ranks)
        if sorted(scale) != [0, 1, 2, 3] or sorted(size) != [0, 1, 2, 3]:
            raise ValueError("scale_ranks and size_ranks must each be permutations of 0..3")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "nuisance_dim", int(self.nuisance_dim))
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "scale_ranks", scale)
        object.__setattr__(self, "size_ranks", size)

    @property
    def case_id(self) -> str:
        return (
            f"{self.stage}__seed-{self.seed}__scenario-{self.scenario}__"
            f"balance-{self.balance}__count-{self.count_level}__d-{self.nuisance_dim}__"
            f"k-{self.k}__signal-{self.signal_state}"
        )

    @property
    def scenario_spec(self) -> Mapping[str, Any]:
        return SCENARIO_BLOCKS[self.scenario]

    @property
    def amplitude(self) -> float:
        return float(self.scenario_spec["amplitude"])

    @property
    def scale_ratio(self) -> float:
        return float(self.scenario_spec["scale_ratio"])

    @property
    def nuisance_distribution(self) -> str:
        return str(self.scenario_spec["distribution"])

    @property
    def nuisance_shift(self) -> str:
        return str(self.scenario_spec["shift"])

    @property
    def nuisance_strength(self) -> float:
        """Alias for the protocol's scenario amplitude in row-generation code."""

        return self.amplitude

    @property
    def overlap_severity(self) -> float:
        return 0.5 if self.signal_state == "genuine_overlap_half" else 0.0

    @property
    def geometry(self) -> str:
        return {
            "linear_separated": "linear",
            "nonlinear_separated": "rings",
            "genuine_overlap_half": "overlap",
        }[self.signal_state]

    @property
    def condition(self) -> str:
        return self.scenario

    @property
    def train_seed(self) -> int:
        return self.seed

    @property
    def evaluation_seed(self) -> int:
        return self.seed

    @property
    def counts(self) -> Tuple[int, ...]:
        table = SMALL_COUNTS if self.count_level == "small" else LARGE_COUNTS
        if self.balance == "balanced":
            return tuple(int(v) for v in table["balanced"])
        values = table["imbalanced"]
        return tuple(int(values[rank]) for rank in self.size_ranks)

    @property
    def total_rows(self) -> int:
        return int(sum(self.counts))

    def identity(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "seed": self.seed,
            "scenario": self.scenario,
            "balance": self.balance,
            "count_level": self.count_level,
            "nuisance_dim": self.nuisance_dim,
            "k": self.k,
            "signal_state": self.signal_state,
            "scale_ranks": list(self.scale_ranks),
            "size_ranks": list(self.size_ranks),
            "counts": list(self.counts),
            "amplitude": self.amplitude,
            "scale_ratio": self.scale_ratio,
            "distribution": self.nuisance_distribution,
            "heldout_scale_assignment": self.nuisance_shift,
        }


# Alias used by callers that call the records "cases" rather than "specs".
Case = CaseSpec
CaseConfig = CaseSpec


@dataclass(frozen=True)
class ConfirmationAuthorization:
    """Opaque authorization returned by validated manifest prerequisites."""

    promotion_decision_sha256: str
    prior_regression_decision_sha256: str
    protocol_sha256: str
    _marker: object


_CONFIRMATION_MARKER = object()


def _make_confirmation_authorization(
    promotion_decision_sha256: str,
    prior_regression_decision_sha256: str,
    protocol_sha256: str = PROTOCOL_SHA256,
) -> ConfirmationAuthorization:
    """Private seam used by :func:`manifest.authorize_confirmation` and tests."""

    return ConfirmationAuthorization(
        str(promotion_decision_sha256),
        str(prior_regression_decision_sha256),
        str(protocol_sha256),
        _CONFIRMATION_MARKER,
    )


def _check_confirmation_authorization(authorization: Any) -> None:
    if not isinstance(authorization, ConfirmationAuthorization) or authorization._marker is not _CONFIRMATION_MARKER:
        raise PermissionError(
            "confirmation banks require a validated ConfirmationAuthorization; "
            "call manifest.authorize_confirmation after the prerequisite decisions pass"
        )
    if (
        not authorization.promotion_decision_sha256
        or not authorization.prior_regression_decision_sha256
        or authorization.protocol_sha256 != PROTOCOL_SHA256
    ):
        raise PermissionError("confirmation authorization does not match the frozen protocol")


# ---------------------------------------------------------------------------
# RNG and maximum-size bank construction
# ---------------------------------------------------------------------------


def _rng(seed_components: Sequence[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(v) for v in seed_components]))


def stream_code(name: str) -> int:
    try:
        return int(STREAM_CODES[str(name)])
    except KeyError as exc:
        raise ValueError(f"unknown fixture stream: {name!r}") from exc


def signal_state_index(state: str) -> int:
    try:
        return int(SIGNAL_STATE_INDEX[str(state)])
    except KeyError as exc:
        raise ValueError(f"unknown signal state: {state!r}") from exc


def balance_index(balance: str) -> int:
    try:
        return int(BALANCE_INDEX[str(balance)])
    except KeyError as exc:
        raise ValueError(f"unknown balance: {balance!r}") from exc


def count_level_index(count_level: str) -> int:
    try:
        return int(COUNT_LEVEL_INDEX[str(count_level)])
    except KeyError as exc:
        raise ValueError(f"unknown count level: {count_level!r}") from exc


def _uniform_open(rng: np.random.Generator, shape: Tuple[int, ...]) -> Array:
    values = rng.random(shape, dtype=np.float64)
    return np.clip(values, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))


def _signal_bank(seed: int, state: str, stream_code: int) -> Tuple[Array, Array]:
    """Generate one class-major maximum signal bank exactly as frozen."""

    state_index = SIGNAL_STATE_INDEX[state]
    rng = _rng((seed, stream_code, state_index))
    if state == "linear_separated":
        # One class-major uniform draw is intentional and part of provenance.
        uniform = rng.random((N_CLASSES, MAX_ROWS_PER_CLASS, SIGNAL_DIM), dtype=np.float64)
        centers = np.asarray((-2.25, -0.75, 0.75, 2.25), dtype=np.float64)
        values = uniform * 0.4 - 0.2
        values[:, :, 0] += centers[:, None]
        components = np.ones((N_CLASSES, MAX_ROWS_PER_CLASS), dtype=np.int8)
        return values, components

    if state == "nonlinear_separated":
        # The draw order is all class-major angles, then all radial noise.
        angles = rng.random((N_CLASSES, MAX_ROWS_PER_CLASS), dtype=np.float64) * (2.0 * np.pi)
        radial = rng.random((N_CLASSES, MAX_ROWS_PER_CLASS), dtype=np.float64) * 0.16 - 0.08
        radii = np.asarray((0.5, 1.0, 1.5, 2.0), dtype=np.float64)[:, None] + radial
        values = np.zeros((N_CLASSES, MAX_ROWS_PER_CLASS, SIGNAL_DIM), dtype=np.float64)
        values[:, :, 0] = radii * np.cos(angles)
        values[:, :, 1] = radii * np.sin(angles)
        components = np.ones((N_CLASSES, MAX_ROWS_PER_CLASS), dtype=np.int8)
        return values, components

    # Half-overlap draw order: class 0 shared/private, class 1 shared/private,
    # then class 2 private and class 3 private.  Interleave shared/private
    # before any nested selection so every even prefix has exactly half shared.
    half = MAX_ROWS_PER_CLASS // 2
    values = np.empty((N_CLASSES, MAX_ROWS_PER_CLASS, SIGNAL_DIM), dtype=np.float64)
    components = np.ones((N_CLASSES, MAX_ROWS_PER_CLASS), dtype=np.int8)
    for class_id in (0, 1):
        shared = rng.random((half, SIGNAL_DIM), dtype=np.float64) * 0.4 - 0.2
        private = rng.random((half, SIGNAL_DIM), dtype=np.float64) * 0.4 - 0.2
        private[:, 0] += (-1.15 if class_id == 0 else 1.15)
        values[class_id, 0::2] = shared
        values[class_id, 1::2] = private
        components[class_id, 0::2] = 0
    for class_id, center in ((2, 3.5), (3, 5.0)):
        private = rng.random((MAX_ROWS_PER_CLASS, SIGNAL_DIM), dtype=np.float64) * 0.4 - 0.2
        private[:, 0] += center
        values[class_id] = private
    return values, components


def _nuisance_bank(seed: int, pool: str) -> NuisanceBank:
    if pool == "train":
        uniform_code = STREAM_CODES["uniform_nuisance_train"]
        mask_code = STREAM_CODES["contamination_mask_train"]
        outlier_code = STREAM_CODES["outlier_gaussian_train"]
    elif pool == "evaluation":
        uniform_code = STREAM_CODES["uniform_nuisance_evaluation"]
        mask_code = STREAM_CODES["contamination_mask_evaluation"]
        outlier_code = STREAM_CODES["outlier_gaussian_evaluation"]
    else:
        raise ValueError("pool must be 'train' or 'evaluation'")
    uniform = _rng((seed, uniform_code)).random(
        (N_CLASSES, MAX_ROWS_PER_CLASS, 32), dtype=np.float64
    )
    # The mask and outlier streams are deliberately independent from the
    # common uniform stream and from one another.
    mask = _rng((seed, mask_code)).random((N_CLASSES, MAX_ROWS_PER_CLASS), dtype=np.float64) < 0.05
    outlier = _rng((seed, outlier_code)).standard_normal(
        (N_CLASSES, MAX_ROWS_PER_CLASS, 32), dtype=np.float64
    )
    return NuisanceBank(_readonly(uniform), _readonly(mask), _readonly(outlier))


def _build_banks(seed: int) -> LatentBanks:
    train_signal: Dict[str, Array] = {}
    eval_signal: Dict[str, Array] = {}
    train_components: Dict[str, Array] = {}
    eval_components: Dict[str, Array] = {}
    for state in SIGNAL_STATE_ORDER:
        train_signal[state], train_components[state] = _signal_bank(
            seed, state, STREAM_CODES["signal_train"]
        )
        eval_signal[state], eval_components[state] = _signal_bank(
            seed, state, STREAM_CODES["signal_evaluation"]
        )
    return LatentBanks(
        int(seed),
        _readonly_mapping(train_signal),
        _readonly_mapping(eval_signal),
        _readonly_mapping(train_components),
        _readonly_mapping(eval_components),
        _nuisance_bank(seed, "train"),
        _nuisance_bank(seed, "evaluation"),
    )


_DEVELOPMENT_BANK_CACHE: Dict[int, LatentBanks] = {}
_CONFIRMATION_BANK_CACHE: Dict[int, LatentBanks] = {}
_BASE_NOISE_CACHE: Dict[Tuple[int, str, str], Array] = {}
_BANK_HASH_CACHE: Dict[int, Dict[str, str]] = {}


def latent_banks(
    seed: int,
    *,
    stage: str = "development",
    authorization: Optional[ConfirmationAuthorization] = None,
) -> LatentBanks:
    """Return maximum-size latent banks for ``seed``.

    Confirmation bank allocation is guarded even when a caller asks for a seed
    that happens to be present in the development range.  The cache is only an
    optimization; all returned arrays are read-only.
    """

    stage_value = str(stage).lower()
    if stage_value not in {"smoke", "development", "confirmation"}:
        raise ValueError("stage must be one of {'smoke', 'development', 'confirmation'}")
    if stage_value == "confirmation":
        _check_confirmation_authorization(authorization)
        cache = _CONFIRMATION_BANK_CACHE
    else:
        cache = _DEVELOPMENT_BANK_CACHE
    seed_value = int(seed)
    if seed_value not in cache:
        cache[seed_value] = _build_banks(seed_value)
    return cache[seed_value]


def bank_hashes(banks: LatentBanks) -> Dict[str, str]:
    """Return byte hashes for all maximum banks in deterministic key order."""

    cache_key = id(banks)
    cached = _BANK_HASH_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    result: Dict[str, str] = {}
    arrays: Dict[str, Array] = {}
    for state in SIGNAL_STATE_ORDER:
        arrays[f"signal_train.{state}"] = banks.signal_train[state]
        arrays[f"signal_evaluation.{state}"] = banks.signal_evaluation[state]
        arrays[f"component_train.{state}"] = banks.component_train[state]
        arrays[f"component_evaluation.{state}"] = banks.component_evaluation[state]
    arrays.update(
        {
            "nuisance_train.uniform": banks.nuisance_train.uniform,
            "nuisance_train.contamination_mask": banks.nuisance_train.contamination_mask,
            "nuisance_train.outlier_gaussian": banks.nuisance_train.outlier_gaussian,
            "nuisance_evaluation.uniform": banks.nuisance_evaluation.uniform,
            "nuisance_evaluation.contamination_mask": banks.nuisance_evaluation.contamination_mask,
            "nuisance_evaluation.outlier_gaussian": banks.nuisance_evaluation.outlier_gaussian,
        }
    )
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        result[name] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    _BANK_HASH_CACHE[cache_key] = dict(result)
    return result


# ---------------------------------------------------------------------------
# Case planning and exact data derivation
# ---------------------------------------------------------------------------


def scale_values(scale_ratio: float) -> Array:
    ratio = float(scale_ratio)
    if not np.isfinite(ratio) or ratio < 1.0:
        raise ValueError("scale_ratio must be finite and at least one")
    values = np.geomspace(ratio ** -0.5, ratio ** 0.5, 4).astype(np.float64)
    values /= np.sqrt(np.mean(values * values, dtype=np.float64))
    return values


def counts_for(balance: str, count_level: str, size_ranks: Sequence[int]) -> Tuple[int, ...]:
    if balance not in BALANCE_ORDER or count_level not in COUNT_LEVEL_ORDER:
        raise ValueError("unknown balance or count_level")
    table = SMALL_COUNTS if count_level == "small" else LARGE_COUNTS
    if balance == "balanced":
        return tuple(int(v) for v in table["balanced"])
    ranks = tuple(int(v) for v in size_ranks)
    if sorted(ranks) != [0, 1, 2, 3]:
        raise ValueError("size_ranks must be a permutation of 0..3")
    return tuple(int(table["imbalanced"][rank]) for rank in ranks)


def nested_selection_indices(
    balance: str,
    count_level: str,
    size_ranks: Sequence[int],
) -> Tuple[Array, ...]:
    """Return class-major prefix indices before the shared row permutation."""

    counts = counts_for(balance, count_level, size_ranks)
    return tuple(np.arange(count, dtype=np.int64) for count in counts)


def scenario_block(name: str) -> Mapping[str, Any]:
    try:
        return dict(SCENARIO_BLOCKS[str(name)])
    except KeyError as exc:
        raise ValueError(f"unknown scenario block: {name!r}") from exc


def _development_permutation(seed: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    try:
        offset = DEVELOPMENT_SEEDS.index(int(seed))
    except ValueError as exc:
        raise ValueError("development seed is not in the frozen seed list") from exc
    return DEVELOPMENT_SCALE_RANKS[offset], DEVELOPMENT_SIZE_RANKS[offset]


def _confirmation_permutation(seed: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    try:
        offset = CONFIRMATION_SEEDS.index(int(seed))
    except ValueError as exc:
        raise ValueError("confirmation seed is not in the frozen seed list") from exc
    scale = CONFIRMATION_SCALE_RANKS[offset]
    shift = CONFIRMATION_SIZE_SHIFTS[offset]
    size = tuple((rank + shift) % 4 for rank in scale)
    return scale, size


def case_spec(
    *,
    stage: str,
    seed: int,
    scenario: str,
    balance: str,
    count_level: str,
    nuisance_dim: int,
    k: int,
    signal_state: str,
) -> CaseSpec:
    stage_value = str(stage).lower()
    if stage_value in {"smoke", "development"}:
        scale, size = _development_permutation(seed)
    elif stage_value == "confirmation":
        scale, size = _confirmation_permutation(seed)
    else:
        raise ValueError("stage must be smoke, development, or confirmation")
    return CaseSpec(
        stage_value,
        int(seed),
        str(scenario),
        str(balance),
        str(count_level),
        int(nuisance_dim),
        int(k),
        str(signal_state),
        tuple(scale),
        tuple(size),
    )


def development_cases() -> Tuple[CaseSpec, ...]:
    cases = []
    for seed in DEVELOPMENT_SEEDS:
        scale, size = _development_permutation(seed)
        for scenario in SCENARIO_ORDER:
            for balance in BALANCE_ORDER:
                for count_level in COUNT_LEVEL_ORDER:
                    for nuisance_dim in NUISANCE_DIM_ORDER:
                        for k in K_ORDER:
                            for signal_state in SIGNAL_STATE_ORDER:
                                cases.append(
                                    CaseSpec(
                                        "development",
                                        seed,
                                        scenario,
                                        balance,
                                        count_level,
                                        nuisance_dim,
                                        k,
                                        signal_state,
                                        tuple(scale),
                                        tuple(size),
                                    )
                                )
    return tuple(cases)


def confirmation_cases(
    authorization: Optional[ConfirmationAuthorization] = None,
) -> Tuple[CaseSpec, ...]:
    """Plan confirmation identities, requiring authorization before banks.

    Planning identities does not allocate arrays and is intentionally available
    before the one-shot confirmation lock.  The token is consumed by
    :func:`latent_banks`/``generate_confirmation_dataset``; accepting it here
    as an optional argument keeps callers that already hold authorization
    source-compatible without weakening the bank-generation guard.
    """

    cases = []
    for seed in CONFIRMATION_SEEDS:
        scale, size = _confirmation_permutation(seed)
        for scenario in SCENARIO_ORDER:
            for balance in BALANCE_ORDER:
                for count_level in COUNT_LEVEL_ORDER:
                    for nuisance_dim in NUISANCE_DIM_ORDER:
                        for k in K_ORDER:
                            for signal_state in SIGNAL_STATE_ORDER:
                                cases.append(
                                    CaseSpec(
                                        "confirmation",
                                        seed,
                                        scenario,
                                        balance,
                                        count_level,
                                        nuisance_dim,
                                        k,
                                        signal_state,
                                        tuple(scale),
                                        tuple(size),
                                    )
                                )
    return tuple(cases)


def smoke_cases() -> Tuple[CaseSpec, ...]:
    cases = []
    # Group 1: all scenario/balance cells at the small d4/k2 linear cell.
    scale, size = _development_permutation(21000)
    for scenario in SCENARIO_ORDER:
        for balance in BALANCE_ORDER:
            cases.append(
                CaseSpec("smoke", 21000, scenario, balance, "small", 4, 2, "linear_separated", scale, size)
            )
    # Group 2: the exact nine alternating tuples in the protocol.
    group_2_states = SIGNAL_STATE_ORDER
    for index, scenario in enumerate(SCENARIO_ORDER):
        balance = "balanced" if index % 2 == 0 else "imbalanced"
        signal_state = group_2_states[index % len(group_2_states)]
        scale, size = _development_permutation(21001)
        cases.append(
            CaseSpec("smoke", 21001, scenario, balance, "large", 32, 8, signal_state, scale, size)
        )
    # Group 3: ALL at the large d32/k8 cell for all signal states.
    scale, size = _development_permutation(21002)
    for signal_state in SIGNAL_STATE_ORDER:
        cases.append(
            CaseSpec("smoke", 21002, "ALL", "imbalanced", "large", 32, 8, signal_state, scale, size)
        )
    return tuple(cases)


def development_case_identities() -> Tuple[Dict[str, Any], ...]:
    return tuple(case.identity() for case in development_cases())


def confirmation_case_identities(
    authorization: Optional[ConfirmationAuthorization] = None,
) -> Tuple[Dict[str, Any], ...]:
    return tuple(case.identity() for case in confirmation_cases(authorization))


def smoke_case_identities() -> Tuple[Dict[str, Any], ...]:
    return tuple(case.identity() for case in smoke_cases())


def _row_order(case: CaseSpec, pool: str) -> Array:
    stream = STREAM_CODES["row_order_train" if pool == "train" else "row_order_evaluation"]
    rng = _rng(
        (
            case.seed,
            stream,
            SIGNAL_STATE_INDEX[case.signal_state],
            BALANCE_INDEX[case.balance],
            COUNT_LEVEL_INDEX[case.count_level],
        )
    )
    return rng.permutation(case.total_rows)


def _base_noise(bank: NuisanceBank, distribution: str) -> Array:
    cache_key = (id(bank), str(distribution), "v1")
    cached = _BASE_NOISE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    clipped = np.clip(bank.uniform, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    gaussian = ndtri(clipped)
    if distribution == "gaussian":
        result = gaussian
    elif distribution == "student_t5":
        result = student_t.ppf(clipped, df=5.0) * np.sqrt(3.0 / 5.0)
    elif distribution == "contaminated":
        result = np.where(bank.contamination_mask[:, :, None], 8.0 * bank.outlier_gaussian, gaussian) / np.sqrt(4.15)
    else:
        raise ValueError("unknown nuisance distribution")
    result = _readonly(result)
    _BASE_NOISE_CACHE[cache_key] = result
    return result


def _derive_split(
    case: CaseSpec,
    banks: LatentBanks,
    pool: str,
) -> DatasetSplit:
    if pool == "train":
        signal_bank = banks.signal_train[case.signal_state]
        components_bank = banks.component_train[case.signal_state]
        nuisance_bank = banks.nuisance_train
        scale_ranks = case.scale_ranks
    elif pool == "evaluation":
        signal_bank = banks.signal_evaluation[case.signal_state]
        components_bank = banks.component_evaluation[case.signal_state]
        nuisance_bank = banks.nuisance_evaluation
        scale_ranks = (
            tuple(3 - rank for rank in case.scale_ranks)
            if case.nuisance_shift == "reversed"
            else case.scale_ranks
        )
    else:
        raise ValueError("pool must be 'train' or 'evaluation'")

    counts = case.counts
    base_noise = _base_noise(nuisance_bank, case.nuisance_distribution)
    scales = scale_values(case.scale_ratio)
    signal_rows = []
    nuisance_rows = []
    labels = []
    components = []
    contamination = []
    for class_id, count in enumerate(counts):
        signal_rows.append(np.asarray(signal_bank[class_id, :count], dtype=np.float64))
        components.append(np.asarray(components_bank[class_id, :count], dtype=np.int8))
        contamination.append(np.asarray(nuisance_bank.contamination_mask[class_id, :count], dtype=bool))
        labels.append(np.full(count, class_id, dtype=np.int64))
        transformed = (
            float(case.amplitude)
            * float(scales[scale_ranks[class_id]])
            * np.asarray(base_noise[class_id, :count, : case.nuisance_dim], dtype=np.float64)
        )
        nuisance_rows.append(transformed)
    signal = np.concatenate(signal_rows, axis=0)
    nuisance = np.concatenate(nuisance_rows, axis=0)
    y = np.concatenate(labels, axis=0)
    component_ids = np.concatenate(components, axis=0)
    order = _row_order(case, pool)
    signal = signal[order]
    nuisance = nuisance[order]
    y = y[order]
    component_ids = component_ids[order]
    contamination_mask = np.concatenate(contamination, axis=0)[order]
    observed = np.concatenate((signal, nuisance), axis=1)
    return DatasetSplit(
        _readonly(observed),
        _readonly(y),
        _readonly(signal),
        _readonly(nuisance),
        _readonly(signal.copy()),
        _readonly(component_ids),
        _readonly(contamination_mask),
    )


def _truth(case: CaseSpec) -> Dict[str, Any]:
    overlap = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    if case.signal_state == "genuine_overlap_half":
        overlap[0, 1] = overlap[1, 0] = 0.5
    labels = (overlap > 0.0).astype(np.int8)
    np.fill_diagonal(labels, 0)
    signal_centers = np.zeros((N_CLASSES, SIGNAL_DIM), dtype=np.float64)
    signal_centers[:, 0] = (-2.25, -0.75, 0.75, 2.25)
    distances = np.linalg.norm(signal_centers[:, None] - signal_centers[None, :], axis=2)
    return {
        "pair_overlap": overlap,
        "truth_pair_overlap": overlap.copy(),
        "pair_overlap_label": labels,
        "truth_overlap_label": labels.copy(),
        "overlap_pairs": [[0, 1]],
        "overlap_severity": 0.5 if case.signal_state == "genuine_overlap_half" else 0.0,
        "signal_state": case.signal_state,
        "signal_centers": signal_centers,
        "pair_signal_distance": distances,
        "overlap_definition": "only class pair (0,1) shares exactly half of its signal bank",
    }


def generate_dataset(
    case: Union[CaseSpec, Mapping[str, Any]],
    *,
    authorization: Optional[ConfirmationAuthorization] = None,
) -> SyntheticDataset:
    """Generate one case, applying the confirmation authorization guard."""

    if not isinstance(case, CaseSpec):
        values = dict(case)
        case = case_spec(**values)
    banks = latent_banks(case.seed, stage=case.stage, authorization=authorization)
    train = _derive_split(case, banks, "train")
    evaluation = _derive_split(case, banks, "evaluation")
    metadata: Dict[str, Any] = {
        **case.identity(),
        "n_classes": N_CLASSES,
        "signal_dim": SIGNAL_DIM,
        "nuisance_dim": case.nuisance_dim,
        "maximum_rows_per_class": MAX_ROWS_PER_CLASS,
        "train_evaluation_independent": True,
        "nested_selection": True,
        "shared_panel_row_order": True,
        "train_seed_components": [case.seed, STREAM_CODES["signal_train"], SIGNAL_STATE_INDEX[case.signal_state]],
        "evaluation_seed_components": [case.seed, STREAM_CODES["signal_evaluation"], SIGNAL_STATE_INDEX[case.signal_state]],
        "row_order_seed_components_train": [
            case.seed,
            STREAM_CODES["row_order_train"],
            SIGNAL_STATE_INDEX[case.signal_state],
            BALANCE_INDEX[case.balance],
            COUNT_LEVEL_INDEX[case.count_level],
        ],
        "row_order_seed_components_evaluation": [
            case.seed,
            STREAM_CODES["row_order_evaluation"],
            SIGNAL_STATE_INDEX[case.signal_state],
            BALANCE_INDEX[case.balance],
            COUNT_LEVEL_INDEX[case.count_level],
        ],
        "bank_hashes": bank_hashes(banks),
        "protocol_sha256": PROTOCOL_SHA256,
    }
    # Keep metadata JSON-safe.  Arrays remain available via the split/truth
    # records, not as mutable metadata side channels.
    return SyntheticDataset(case, train, evaluation, _truth(case), metadata)


def generate_development_dataset(case: Union[CaseSpec, Mapping[str, Any]]) -> SyntheticDataset:
    if isinstance(case, Mapping):
        values = dict(case)
        values["stage"] = "development"
        case = case_spec(**values)
    elif case.stage not in {"smoke", "development"}:
        raise ValueError("development dataset requires a smoke/development case")
    return generate_dataset(case)


def generate_confirmation_dataset(
    case: Union[CaseSpec, Mapping[str, Any]],
    *,
    authorization: Optional[ConfirmationAuthorization] = None,
) -> SyntheticDataset:
    if isinstance(case, Mapping):
        values = dict(case)
        values["stage"] = "confirmation"
        case = case_spec(**values)
    if case.stage != "confirmation":
        raise ValueError("confirmation dataset requires a confirmation case")
    return generate_dataset(case, authorization=authorization)


# Natural descriptive names used by runner code.
generate_case = generate_dataset
plan_development_cases = development_cases
plan_smoke_cases = smoke_cases
generate_latent_banks = latent_banks


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def case_identity_sha256(case: Union[CaseSpec, Mapping[str, Any]]) -> str:
    identity = case.identity() if isinstance(case, CaseSpec) else dict(case)
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


__all__ = [
    "BALANCE_INDEX",
    "BALANCE_ORDER",
    "Case",
    "CaseConfig",
    "CaseSpec",
    "CONFIRMATION_BASE_METHODS",
    "CONFIRMATION_SCALE_RANKS",
    "CONFIRMATION_SEEDS",
    "CONFIRMATION_SIZE_SHIFTS",
    "ConfirmationAuthorization",
    "COUNT_LEVEL_INDEX",
    "COUNT_LEVEL_ORDER",
    "DEVELOPMENT_METHODS",
    "DEVELOPMENT_SCALE_RANKS",
    "DEVELOPMENT_SEEDS",
    "DEVELOPMENT_SIZE_RANKS",
    "DatasetSplit",
    "K_ORDER",
    "LatentBanks",
    "MAX_ROWS",
    "MAX_ROWS_PER_CLASS",
    "METHOD_IDS",
    "N_CLASSES",
    "NuisanceBank",
    "NUISANCE_DIM_ORDER",
    "PROTOCOL_PATH",
    "PROTOCOL_SHA256",
    "SCENARIO_BLOCKS",
    "SCENARIO_ORDER",
    "SCENARIOS",
    "SIGNAL_DIM",
    "SIGNAL_STATE_INDEX",
    "SIGNAL_STATE_ORDER",
    "SIGNAL_STATES",
    "SMOKE_METHODS",
    "SyntheticDataset",
    "balance_index",
    "bank_hashes",
    "canonical_json",
    "case_identity_sha256",
    "case_spec",
    "count_level_index",
    "confirmation_case_identities",
    "confirmation_cases",
    "counts_for",
    "nested_selection_indices",
    "development_case_identities",
    "development_cases",
    "generate_case",
    "generate_confirmation_dataset",
    "generate_dataset",
    "generate_development_dataset",
    "generate_latent_banks",
    "latent_banks",
    "plan_development_cases",
    "plan_smoke_cases",
    "scale_values",
    "scenario_block",
    "signal_state_index",
    "smoke_case_identities",
    "smoke_cases",
    "stream_code",
    "_make_confirmation_authorization",
]
