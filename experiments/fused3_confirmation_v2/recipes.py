"""Closed method and downstream-head recipes for FUSED3 confirmation V2."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from experiments.m50_backbone_ranking import food101
from experiments.scalable_relevance_kmeans import fused_food_screen


DATASET_IDS = (
    "torchvision_cifar10",
    "torchvision_stl10",
    "torchvision_gtsrb",
    "torchvision_fgvc_aircraft",
    "torchvision_dtd",
)
BACKBONES = tuple(food101.MODELS)
BUDGETS = (32, 64)
REPLICATE_SEEDS = tuple(2026082710 + offset for offset in range(5))
EVALUATION_SEED = 2026082709
HEADS = ("linear", "quadratic", "knn", "rbf")
SELECTORS = ("FUSED3", "LP-FULL")
FOLDS = 5
EXTRACTION_ENVIRONMENT = {
    "python": "3.9.6",
    "numpy": "2.0.2",
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "timm": "1.0.27",
    "transformers": "4.57.6",
    "open_clip": "2.32.0",
    "Pillow": "11.3.0",
    "vertebrae": "1.0.0",
    "device": "cpu",
    "batch_size": 16,
    "network_policy": "offline_cached_weights_only",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def candidate_recipe(candidate_id: str) -> dict[str, Any]:
    """Return the exact, JSON-safe frozen recipe for one selector."""

    if candidate_id == "FUSED3":
        return {
            "candidate_id": "FUSED3",
            "input_dtype": "float32",
            "preprocessing": "row_l2_once_before_folds",
            "folds": {
                "class": "StratifiedKFold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": "replicate_seed",
            },
            "fold_seed": "replicate_seed + fold_index",
            "k_per_class": "min(10,max(1,train_class_count//5),train_class_count)",
            "prototype_iterations": 3,
            "prototype_backend": "fused",
            "margin_rows_per_class": 32,
            "weight_floor": 0.05,
            "memory_budget_mb": 64,
            "row_cap": None,
            "score": "mean_of_five_score_fixed_holdout_scores",
            "fitted_state": "centers_owners_weights_sha256_must_not_change_on_score_fixed",
        }
    if candidate_id == "LP-FULL":
        return {
            "candidate_id": "LP-FULL",
            "input_dtype": "float32",
            "folds": {
                "class": "StratifiedKFold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": "replicate_seed",
            },
            "pipeline": [
                {"class": "Normalizer", "norm": "l2"},
                {
                    "class": "LogisticRegression",
                    "C": 1.0,
                    "max_iter": 2000,
                    "random_state": "replicate_seed",
                    "n_jobs": 1,
                },
            ],
            "score": "five_fold_out_of_fold_accuracy",
        }
    raise ValueError(f"candidate_id must be one of {SELECTORS!r}; got {candidate_id!r}")


def head_recipe(head: str) -> dict[str, Any]:
    """Return the exact downstream reference-head recipe."""

    recipes = {
        "linear": {
            "preprocessing": {"class": "Normalizer", "norm": "l2"},
            "estimator": {
                "class": "LogisticRegression",
                "C": 1.0,
                "max_iter": 2000,
                "random_state": "replicate_seed + head_index",
                "n_jobs": 1,
            },
        },
        "quadratic": {
            "preprocessing": {"class": "Normalizer", "norm": "l2"},
            "estimator": {
                "class": "SVC",
                "kernel": "poly",
                "degree": 2,
                "C": 1.0,
                "gamma": "scale",
                "coef0": 1.0,
            },
        },
        "knn": {
            "preprocessing": {"class": "Normalizer", "norm": "l2"},
            "estimator": {
                "class": "KNeighborsClassifier",
                "n_neighbors": 15,
                "weights": "distance",
                "metric": "cosine",
                "n_jobs": 1,
            },
        },
        "rbf": {
            "preprocessing": {"class": "Normalizer", "norm": "l2"},
            "estimator": {
                "class": "SVC",
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
            },
        },
    }
    try:
        return {"head": head, **recipes[head]}
    except KeyError as exc:
        raise ValueError(f"head must be one of {HEADS!r}; got {head!r}") from exc


def build_head(head: str, *, seed: int) -> Any:
    """Build one frozen downstream estimator without tuning."""

    if type(seed) is not int:
        raise TypeError("seed must be an int")
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.svm import SVC

    if head == "linear":
        estimator: Any = LogisticRegression(
            C=1.0, max_iter=2000, random_state=seed, n_jobs=1
        )
    elif head == "quadratic":
        estimator = SVC(
            kernel="poly", degree=2, C=1.0, gamma="scale", coef0=1.0
        )
    elif head == "knn":
        estimator = KNeighborsClassifier(
            n_neighbors=15, weights="distance", metric="cosine", n_jobs=1
        )
    elif head == "rbf":
        estimator = SVC(kernel="rbf", C=1.0, gamma="scale")
    else:
        raise ValueError(f"head must be one of {HEADS!r}; got {head!r}")
    return make_pipeline(Normalizer(norm="l2"), estimator)


def execute_selector(
    candidate_id: str,
    values: Any,
    labels: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    """Execute exactly the locked FUSED3 or full-probe adapter."""

    if type(seed) is not int:
        raise TypeError("seed must be an int")
    matrix = np.asarray(values, dtype=np.float32)
    target = np.asarray(labels)
    if matrix.ndim != 2 or target.ndim != 1 or len(matrix) != len(target):
        raise ValueError("values/labels must be aligned 2-D/1-D arrays")
    if not np.isfinite(matrix).all():
        raise ValueError("values must be finite")
    if candidate_id == "FUSED3":
        result = fused_food_screen.cross_fitted_candidate(
            matrix, target, candidate_id="FUSED3", seed=seed
        )
    elif candidate_id == "LP-FULL":
        result = food101._cross_fitted_score(
            matrix, target, candidate="LP-FULL", seed=seed
        )
    else:
        raise ValueError(f"candidate_id must be one of {SELECTORS!r}; got {candidate_id!r}")
    score = float(result.get("score", float("nan")))
    if not np.isfinite(score):
        raise RuntimeError(f"{candidate_id} produced a non-finite score")
    output = dict(result)
    output["candidate_recipe"] = candidate_recipe(candidate_id)
    output["candidate_recipe_sha256"] = payload_sha256(output["candidate_recipe"])
    return output


def execute_reference_head(
    head: str,
    training_values: Any,
    training_labels: Any,
    evaluation_values: Any,
    evaluation_labels: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit on the sampled target-training cohort and evaluate on test only."""

    from sklearn.metrics import accuracy_score

    train = np.asarray(training_values, dtype=np.float32)
    train_y = np.asarray(training_labels)
    test = np.asarray(evaluation_values, dtype=np.float32)
    test_y = np.asarray(evaluation_labels)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("training/evaluation values must have matching feature dimensions")
    if train_y.ndim != 1 or test_y.ndim != 1:
        raise ValueError("training/evaluation labels must be one-dimensional")
    if len(train) != len(train_y) or len(test) != len(test_y):
        raise ValueError("values and labels must be aligned")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("reference-head values must be finite")
    estimator = build_head(head, seed=seed)
    estimator.fit(train, train_y)
    predicted = estimator.predict(test)
    accuracy = float(accuracy_score(test_y, predicted))
    if not np.isfinite(accuracy):
        raise RuntimeError(f"{head} produced a non-finite accuracy")
    recipe = head_recipe(head)
    return {
        "head": head,
        "test_accuracy": accuracy,
        "training_row_count": int(len(train)),
        "evaluation_row_count": int(len(test)),
        "recipe": recipe,
        "recipe_sha256": payload_sha256(recipe),
    }


def method_order(panel_index: int) -> tuple[str, str]:
    """Two-method exact cyclic counterbalance over the 500 frozen panels."""

    if type(panel_index) is not int or panel_index < 0:
        raise ValueError("panel_index must be a nonnegative int")
    return SELECTORS if panel_index % 2 == 0 else tuple(reversed(SELECTORS))


__all__ = [
    "BACKBONES",
    "BUDGETS",
    "DATASET_IDS",
    "EVALUATION_SEED",
    "EXTRACTION_ENVIRONMENT",
    "FOLDS",
    "HEADS",
    "REPLICATE_SEEDS",
    "SELECTORS",
    "build_head",
    "candidate_recipe",
    "execute_reference_head",
    "execute_selector",
    "head_recipe",
    "method_order",
    "payload_sha256",
]
