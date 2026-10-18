"""Missing-value and outlier treatment as scikit-learn transformers.

Everything that is learned (caps, cut points, medians) is learned in ``fit`` from the training
data only and reused as-is in ``transform``, so the same objects serve valid, test and the API.
"""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data.schema import FEATURES, PAST_DUE_COLUMNS, SPECIAL_PAST_DUE_CODES

REVOLVING = "RevolvingUtilizationOfUnsecuredLines"
SPECIAL_FLAG = "has_special_past_due_code"
INCOME_FLAG = "income_unavailable"

MODEL_FEATURES = [*FEATURES, SPECIAL_FLAG, INCOME_FLAG]
COUNT_COLUMNS = [
    *PAST_DUE_COLUMNS,
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
]

CAP_QUANTILE = 0.99
AGE_QUANTILES = np.linspace(0.1, 0.9, 9)
MAX_DEPENDENTS_GROUP = 4


def drop_underage(df: pd.DataFrame, min_age: int = 18) -> pd.DataFrame:
    """Return the rows with ``age >= min_age`` (the input is not modified)."""
    return df.loc[df["age"] >= min_age].copy()


class SpecialPastDueRecoder(BaseEstimator, TransformerMixin):
    """Select FEATURES, flag the 96/98 past-due codes and reset them to 0."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X[FEATURES].copy()
        special = out[PAST_DUE_COLUMNS].isin(SPECIAL_PAST_DUE_CODES)
        out[SPECIAL_FLAG] = special.any(axis=1).astype(int)
        out[PAST_DUE_COLUMNS] = out[PAST_DUE_COLUMNS].mask(special, 0)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([*FEATURES, SPECIAL_FLAG], dtype=object)


class IncomeImputer(BaseEstimator, TransformerMixin):
    """Impute unavailable income (NaN or 0) by age decile x dependents group medians."""

    def __init__(self, min_group_size: int = 30):
        self.min_group_size = min_group_size

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        income = X["MonthlyIncome"]
        observed = income[income > 0]
        self.income_cap_ = float(observed.quantile(CAP_QUANTILE))
        capped = observed.clip(upper=self.income_cap_)

        inner = np.unique(X["age"].quantile(AGE_QUANTILES).to_numpy())
        self.age_edges_ = np.concatenate([[-np.inf], inner, [np.inf]])

        groups = pd.DataFrame(
            {
                "age_bin": self._age_bins(X.loc[observed.index, "age"]),
                "dependents": self._dependent_groups(X.loc[observed.index, "NumberOfDependents"]),
                "income": capped.to_numpy(),
            }
        )
        keys = ["age_bin", "dependents"]
        grouped = groups.groupby(keys)["income"]
        medians, sizes = grouped.median(), grouped.size()
        self.group_medians_ = {
            (int(a), int(d)): float(medians[(a, d)])
            for (a, d), size in sizes.items()
            if size >= self.min_group_size
        }
        self.age_medians_ = {
            int(b): float(m) for b, m in groups.groupby("age_bin")["income"].median().items()
        }
        self.global_median_ = float(capped.median())
        return self

    def _age_bins(self, age: pd.Series) -> np.ndarray:
        return np.digitize(age.to_numpy(dtype=float), self.age_edges_[1:-1])

    @staticmethod
    def _dependent_groups(dependents: pd.Series) -> np.ndarray:
        return dependents.fillna(0).clip(0, MAX_DEPENDENTS_GROUP).to_numpy().astype(int)

    def transform(self, X):
        out = X.copy()
        income = out["MonthlyIncome"]
        unavailable = income.isna() | (income == 0)
        out[INCOME_FLAG] = unavailable.astype(int)

        dependents = out["NumberOfDependents"].fillna(0)
        out["NumberOfDependents"] = dependents

        bins = self._age_bins(out["age"])
        groups = self._dependent_groups(dependents)
        fill = np.array(
            [
                self.group_medians_.get(
                    (int(b), int(g)), self.age_medians_.get(int(b), self.global_median_)
                )
                for b, g in zip(bins, groups, strict=True)
            ],
            dtype=float,
        )
        capped = income.clip(upper=self.income_cap_)
        out["MonthlyIncome"] = np.where(unavailable, fill, capped)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([*self.feature_names_in_, INCOME_FLAG], dtype=object)


class RatioTreatment(BaseEstimator, TransformerMixin):
    """Recompute DebtRatio where income was unavailable and cap the ratio columns."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        available = X.loc[X[INCOME_FLAG] == 0, "DebtRatio"]
        self.debt_ratio_cap_ = float(available.quantile(CAP_QUANTILE))
        self.revolving_cap_ = float(X[REVOLVING].quantile(CAP_QUANTILE))
        return self

    def transform(self, X):
        out = X.copy()
        amount = out[INCOME_FLAG] == 1
        ratio = out["DebtRatio"].where(~amount, out["DebtRatio"] / out["MonthlyIncome"])
        out["DebtRatio"] = ratio.clip(upper=self.debt_ratio_cap_)
        out[REVOLVING] = out[REVOLVING].clip(upper=self.revolving_cap_)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_, dtype=object)


class LogisticScaling(BaseEstimator, TransformerMixin):
    """Logistic-branch only: cap the counts at their p99 and log-transform skewed columns."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.count_caps_ = {c: float(X[c].quantile(CAP_QUANTILE)) for c in COUNT_COLUMNS}
        return self

    def transform(self, X):
        out = X.copy()
        for column, cap in self.count_caps_.items():
            out[column] = out[column].clip(upper=cap)
        for column in ("MonthlyIncome", REVOLVING, "DebtRatio"):
            out[column] = np.log1p(out[column])
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_, dtype=object)


def build_preprocessor(kind: Literal["logistic", "tree"]) -> Pipeline:
    """Common treatment, plus capping/log/standardisation for the logistic branch."""
    steps: list[tuple[str, object]] = [
        ("recode", SpecialPastDueRecoder()),
        ("impute_income", IncomeImputer()),
        ("ratios", RatioTreatment()),
    ]
    if kind == "logistic":
        steps += [("scaling", LogisticScaling()), ("standardize", StandardScaler())]
    elif kind != "tree":
        raise ValueError(f"Unknown preprocessor kind: {kind!r}")
    return Pipeline(steps).set_output(transform="pandas")
