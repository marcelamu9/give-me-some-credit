import numpy as np
import pandas as pd
import pytest

from credit_risk.data.schema import FEATURES
from credit_risk.models.candidates import (
    CANDIDATE_NAMES,
    calibrate,
    candidate_specs,
    prediction_latency_ms,
    search_candidate,
)


def make_xy(n: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    revolving = rng.uniform(0, 1.2, n)
    late = rng.poisson(0.3, n)
    age = rng.integers(20, 80, n)
    income = rng.lognormal(8.5, 0.5, n)
    income[rng.random(n) < 0.15] = np.nan
    logit = -3.5 + 3.0 * revolving + 0.8 * late - 0.02 * (age - 50)
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))), name="y")
    X = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": revolving,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late,
            "DebtRatio": rng.uniform(0, 1.5, n),
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, n),
            "NumberOfTimes90DaysLate": rng.poisson(0.1, n),
            "NumberRealEstateLoansOrLines": rng.poisson(1, n),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.1, n),
            "NumberOfDependents": rng.integers(0, 4, n).astype(float),
        }
    )[FEATURES]
    return X, y


def test_candidate_specs_names_and_steps():
    specs = candidate_specs(seed=0)
    assert tuple(specs) == CANDIDATE_NAMES
    for pipeline, distributions in specs.values():
        assert pipeline.steps[-1][0] == "model"
        assert distributions
        assert all(key.startswith("model__") for key in distributions)


@pytest.mark.parametrize("name", CANDIDATE_NAMES)
def test_search_candidate_runs_on_small_data(name):
    X, y = make_xy()
    pipeline, distributions = candidate_specs(seed=0)[name]
    result = search_candidate(name, pipeline, distributions, X, y, n_iter=2, cv_folds=2, seed=0)
    assert result.name == name
    assert result.cv_auc_mean > 0.6
    assert result.cv_auc_std >= 0
    assert result.search_seconds > 0
    assert set(result.best_params) == set(distributions)
    assert result.estimator.predict_proba(X.iloc[:5]).shape == (5, 2)


def test_calibrate_returns_probabilities():
    X, y = make_xy()
    pipeline, _ = candidate_specs(seed=0)["logistic"]
    model, seconds = calibrate(pipeline, X, y, cv_folds=2)
    proba = model.predict_proba(X.iloc[:50])
    assert proba.min() >= 0 and proba.max() <= 1
    assert proba.sum(axis=1) == pytest.approx(1.0)
    assert seconds > 0


def test_prediction_latency_positive():
    X, y = make_xy(400)
    pipeline, _ = candidate_specs(seed=0)["logistic"]
    model, _ = calibrate(pipeline, X, y, cv_folds=2)
    assert prediction_latency_ms(model, X, n_rows=5, repeats=2) > 0
