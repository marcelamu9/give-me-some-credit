"""Pydantic contracts of the inference API (see the API spec, section 3)."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from credit_risk.data.schema import FEATURES

# Counts accept 96/98 (the source's "unknown" codes); the pipeline handles them.
_COUNT_MAX = 98


class ClientFeatures(BaseModel):
    """One client, with the original Kaggle names as JSON keys (aliases)."""

    # strict: numbers sent as text ("30") or booleans are rejected instead of coerced.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    revolving_utilization: float = Field(alias="RevolvingUtilizationOfUnsecuredLines", ge=0)
    age: int = Field(alias="age", ge=18, le=120)
    past_due_30_59: int = Field(alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0, le=_COUNT_MAX)
    debt_ratio: float = Field(alias="DebtRatio", ge=0)
    monthly_income: float | None = Field(default=None, alias="MonthlyIncome", ge=0)
    open_credit_lines: int = Field(alias="NumberOfOpenCreditLinesAndLoans", ge=0, le=_COUNT_MAX)
    times_90_days_late: int = Field(alias="NumberOfTimes90DaysLate", ge=0, le=_COUNT_MAX)
    real_estate_loans: int = Field(alias="NumberRealEstateLoansOrLines", ge=0, le=_COUNT_MAX)
    past_due_60_89: int = Field(alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0, le=_COUNT_MAX)
    dependents: int | None = Field(default=None, alias="NumberOfDependents", ge=0, le=20)

    def to_frame(self) -> pd.DataFrame:
        """One-row frame with the model's columns in training order (None -> NaN)."""
        row = self.model_dump(by_alias=True)
        return pd.DataFrame([row], columns=FEATURES).astype("float64")


class Reason(BaseModel):
    feature: str
    description: str


class PredictionResponse(BaseModel):
    probability: float
    risk_level: Literal["low", "medium", "high"]
    decision: Literal["approve", "review", "reject"]
    top_reasons: list[Reason] = Field(min_length=3, max_length=3)
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None


class ModelInfoResponse(BaseModel):
    model_version: str
    champion: str
    thresholds: dict[str, float]
    costs: dict[str, float]
    review_capacity: float
    test_metrics: dict[str, float]
    features: list[str]
