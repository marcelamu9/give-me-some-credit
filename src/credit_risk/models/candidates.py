"""Candidate models, hyperparameter search, calibration and latency measurement."""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from credit_risk.features.treatment import build_preprocessor

COMPLEXITY_ORDER = ("logistic", "random_forest", "lightgbm")


@dataclass
class SearchResult:
    name: str
    estimator: Pipeline
    best_params: dict
    cv_auc_mean: float
    cv_auc_std: float
    search_seconds: float


def candidate_specs(seed: int) -> dict[str, tuple[Pipeline, dict]]:
    """Pipeline (preprocessing + model) and search distributions for each candidate."""
    logistic = Pipeline(
        [
            ("preprocess", build_preprocessor("logistic")),
            (
                "model",
                LogisticRegression(solver="saga", max_iter=2000, random_state=seed),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("preprocess", build_preprocessor("tree")),
            ("model", RandomForestClassifier(n_jobs=-1, random_state=seed)),
        ]
    )
    boosting = Pipeline(
        [
            ("preprocess", build_preprocessor("tree")),
            (
                "model",
                LGBMClassifier(
                    subsample_freq=1,
                    random_state=seed,
                    deterministic=True,
                    force_row_wise=True,
                    verbose=-1,
                ),
            ),
        ]
    )
    return {
        "logistic": (
            logistic,
            {
                "model__C": loguniform(1e-3, 1e2),
                # penalty is deprecated in scikit-learn 1.9: l1_ratio 0 = l2, 1 = l1.
                "model__l1_ratio": [0.0, 1.0],
            },
        ),
        "random_forest": (
            forest,
            {
                "model__n_estimators": randint(100, 301),
                "model__max_depth": randint(4, 13),
                "model__min_samples_leaf": randint(20, 201),
                "model__max_features": ["sqrt", 0.5],
            },
        ),
        "lightgbm": (
            boosting,
            {
                "model__n_estimators": randint(200, 801),
                "model__learning_rate": loguniform(0.01, 0.1),
                "model__num_leaves": randint(15, 64),
                "model__min_child_samples": randint(20, 201),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__reg_lambda": uniform(0, 5),
            },
        ),
    }


def search_candidate(
    name: str,
    pipeline: Pipeline,
    distributions: dict,
    X: pd.DataFrame,
    y,
    n_iter: int,
    cv_folds: int,
    seed: int,
) -> SearchResult:
    """Randomized search by ROC-AUC on stratified folds; refits the best model on all of X."""
    search = RandomizedSearchCV(
        pipeline,
        distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        random_state=seed,
        refit=True,
    )
    start = time.perf_counter()
    search.fit(X, y)
    seconds = time.perf_counter() - start
    best = search.best_index_
    return SearchResult(
        name=name,
        estimator=search.best_estimator_,
        best_params=search.best_params_,
        cv_auc_mean=float(search.cv_results_["mean_test_score"][best]),
        cv_auc_std=float(search.cv_results_["std_test_score"][best]),
        search_seconds=seconds,
    )


def calibrate(
    estimator: Pipeline, X: pd.DataFrame, y, cv_folds: int
) -> tuple[CalibratedClassifierCV, float]:
    """Isotonic calibration with cross-validation; returns the model and the fit seconds."""
    model = CalibratedClassifierCV(clone(estimator), method="isotonic", cv=cv_folds)
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start


def prediction_latency_ms(model, X: pd.DataFrame, n_rows: int = 1000, repeats: int = 5) -> float:
    """Median per-client latency in ms, predicting one row at a time (as the API will)."""
    sample = X.iloc[: min(n_rows, len(X))]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for i in range(len(sample)):
            model.predict_proba(sample.iloc[[i]])
        timings.append((time.perf_counter() - start) / len(sample) * 1000)
    return float(np.median(timings))
