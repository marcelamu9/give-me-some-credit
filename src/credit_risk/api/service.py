"""Model loading, prediction and per-client reasons (pure logic, no FastAPI)."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from credit_risk.decision.policy import RISK_LEVEL, assign_band
from credit_risk.features.treatment import MODEL_FEATURES

ARTIFACT_NAME = "champion.joblib"
METADATA_NAME = "champion_metadata.json"

# Model variable -> (feature shown to the client, description in Spanish).
REASONS: dict[str, tuple[str, str]] = {
    "RevolvingUtilizationOfUnsecuredLines": (
        "RevolvingUtilizationOfUnsecuredLines",
        "Uso alto de tarjetas y líneas de crédito",
    ),
    "age": ("age", "Edad"),
    "NumberOfTime30-59DaysPastDueNotWorse": (
        "NumberOfTime30-59DaysPastDueNotWorse",
        "Atrasos de 30 a 59 días",
    ),
    "NumberOfTime60-89DaysPastDueNotWorse": (
        "NumberOfTime60-89DaysPastDueNotWorse",
        "Atrasos de 60 a 89 días",
    ),
    "NumberOfTimes90DaysLate": ("NumberOfTimes90DaysLate", "Atrasos de 90 días o más"),
    "DebtRatio": ("DebtRatio", "Nivel de endeudamiento frente al ingreso"),
    "MonthlyIncome": ("MonthlyIncome", "Ingreso mensual"),
    "NumberOfOpenCreditLinesAndLoans": (
        "NumberOfOpenCreditLinesAndLoans",
        "Número de créditos abiertos",
    ),
    "NumberRealEstateLoansOrLines": ("NumberRealEstateLoansOrLines", "Créditos inmobiliarios"),
    "NumberOfDependents": ("NumberOfDependents", "Número de dependientes"),
    "income_unavailable": ("MonthlyIncome", "Ingreso no informado"),
    "has_special_past_due_code": (
        "NumberOfTimes90DaysLate",
        "Historial de atrasos con código especial (96/98)",
    ),
}


@dataclass(frozen=True)
class LoadedModel:
    model: object
    features: list[str]
    t_low: float
    t_star: float
    metadata: dict

    @property
    def version(self) -> str:
        return self.metadata["trained_at"]


def load_model(model_dir: Path) -> LoadedModel | None:
    """Load the champion artifact and its metadata; ``None`` if either is missing."""
    artifact_path = model_dir / ARTIFACT_NAME
    metadata_path = model_dir / METADATA_NAME
    if not artifact_path.is_file() or not metadata_path.is_file():
        return None
    artifact = joblib.load(artifact_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return LoadedModel(
        model=artifact["model"],
        features=artifact["features"],
        t_low=artifact["t_low"],
        t_star=artifact["t_star"],
        metadata=metadata,
    )


def _copy_contributions(pipeline, row: pd.DataFrame) -> np.ndarray:
    """Contribution of each model variable to the log-odds, for one calibrated copy."""
    transformed = pipeline[:-1].transform(row)
    estimator = pipeline[-1]
    if hasattr(estimator, "coef_"):
        return estimator.coef_[0] * transformed.to_numpy(dtype=float)[0]
    values = shap.TreeExplainer(estimator).shap_values(transformed)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values[0]


def feature_contributions(model, row: pd.DataFrame) -> pd.Series:
    """Per-variable contribution averaged over the calibrated copies (as the probability is)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_copy = [_copy_contributions(c.estimator, row) for c in model.calibrated_classifiers_]
    return pd.Series(np.mean(per_copy, axis=0), index=MODEL_FEATURES)


def top_reasons(contributions: pd.Series, n: int = 3) -> list[dict[str, str]]:
    """Variables that raise the risk the most, mapped to client-facing names, without repeats."""
    positive = contributions[contributions > 0].sort_values(ascending=False)
    rest = contributions[contributions <= 0]
    rest = rest.reindex(rest.abs().sort_values(ascending=False).index)
    reasons: list[dict[str, str]] = []
    seen: set[str] = set()
    for variable in [*positive.index, *rest.index]:
        feature, description = REASONS[variable]
        if feature in seen:
            continue
        seen.add(feature)
        reasons.append({"feature": feature, "description": description})
        if len(reasons) == n:
            break
    return reasons


def predict(loaded: LoadedModel, row: pd.DataFrame) -> dict:
    probability = float(loaded.model.predict_proba(row)[0, 1])
    decision = str(assign_band(probability, loaded.t_low, loaded.t_star))
    return {
        "probability": round(probability, 4),
        "decision": decision,
        "risk_level": RISK_LEVEL[decision],
        "top_reasons": top_reasons(feature_contributions(loaded.model, row)),
        "model_version": loaded.version,
    }
