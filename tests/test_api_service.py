import json

import numpy as np
import pandas as pd
import pytest
import shap
from api_support import build_model_dir
from test_api_schemas import VALID

from credit_risk.api.schemas import ClientFeatures
from credit_risk.api.service import (
    REASONS,
    feature_contributions,
    load_model,
    predict,
    top_reasons,
)
from credit_risk.decision.policy import RISK_LEVEL, assign_band
from credit_risk.features.treatment import MODEL_FEATURES


@pytest.fixture(scope="module")
def lightgbm_dir(tmp_path_factory):
    return build_model_dir(tmp_path_factory.mktemp("lgbm"), "lightgbm")


@pytest.fixture(scope="module")
def logistic_dir(tmp_path_factory):
    return build_model_dir(tmp_path_factory.mktemp("logit"), "logistic")


@pytest.fixture(scope="module")
def loaded(lightgbm_dir):
    return load_model(lightgbm_dir)


def row(**overrides) -> pd.DataFrame:
    return ClientFeatures.model_validate({**VALID, **overrides}).to_frame()


def test_load_model_missing_returns_none(tmp_path):
    assert load_model(tmp_path) is None


def test_load_model_without_metadata_returns_none(lightgbm_dir, tmp_path):
    (tmp_path / "champion.joblib").write_bytes((lightgbm_dir / "champion.joblib").read_bytes())
    assert load_model(tmp_path) is None


def test_load_model_reads_artifact(loaded, lightgbm_dir):
    metadata = json.loads((lightgbm_dir / "champion_metadata.json").read_text())
    assert loaded.version == metadata["trained_at"]
    assert loaded.t_low < loaded.t_star


def test_contributions_average_the_five_copies(loaded):
    model = loaded.model
    assert len(model.calibrated_classifiers_) == 5
    client = row()
    per_copy = []
    for copy in model.calibrated_classifiers_:
        pipeline = copy.estimator
        transformed = pipeline[:-1].transform(client)
        values = shap.TreeExplainer(pipeline["model"]).shap_values(transformed)
        values = np.asarray(values[1] if isinstance(values, list) else values)
        if values.ndim == 3:
            values = values[:, :, 1]
        per_copy.append(values[0])
    expected = np.mean(per_copy, axis=0)

    result = feature_contributions(model, client)
    assert list(result.index) == MODEL_FEATURES
    assert np.allclose(result.to_numpy(), expected)


def test_contributions_for_logistic(logistic_dir):
    logistic = load_model(logistic_dir)
    result = feature_contributions(logistic.model, row())
    assert list(result.index) == MODEL_FEATURES
    assert not result.isna().any()


def test_top_reasons_positive_first():
    contributions = pd.Series(
        {"age": 0.1, "DebtRatio": 0.4, "MonthlyIncome": 0.3, "NumberOfDependents": 0.2,
         "NumberOfTimes90DaysLate": -0.5}
    )  # fmt: skip
    reasons = top_reasons(contributions)
    assert [r["feature"] for r in reasons] == ["DebtRatio", "MonthlyIncome", "NumberOfDependents"]


def test_top_reasons_fills_with_absolute():
    contributions = pd.Series(
        {"age": 0.1, "DebtRatio": -0.4, "MonthlyIncome": -0.3, "NumberOfDependents": -0.05}
    )
    reasons = top_reasons(contributions)
    assert len(reasons) == 3
    assert [r["feature"] for r in reasons] == ["age", "DebtRatio", "MonthlyIncome"]


def test_top_reasons_maps_and_deduplicates():
    contributions = pd.Series(
        {
            "income_unavailable": 0.5,
            "MonthlyIncome": 0.4,
            "has_special_past_due_code": 0.3,
            "age": 0.2,
        }
    )
    reasons = top_reasons(contributions)
    assert [r["feature"] for r in reasons] == ["MonthlyIncome", "NumberOfTimes90DaysLate", "age"]
    assert reasons[0]["description"] == REASONS["income_unavailable"][1]
    assert reasons[1]["description"] == REASONS["has_special_past_due_code"][1]


def test_predict_matches_artifact_probability(loaded):
    client = row()
    result = predict(loaded, client)
    p = loaded.model.predict_proba(client)[0, 1]
    assert result["probability"] == round(p, 4)
    assert result["decision"] == str(assign_band(p, loaded.t_low, loaded.t_star))
    assert result["risk_level"] == RISK_LEVEL[result["decision"]]
    assert len(result["top_reasons"]) == 3
    assert result["model_version"] == loaded.version


def test_predict_single_row_with_special_codes_and_nulls(loaded):
    client = row(
        **{
            "NumberOfTime30-59DaysPastDueNotWorse": 98,
            "NumberOfTimes90DaysLate": 98,
            "NumberOfTime60-89DaysPastDueNotWorse": 98,
            "MonthlyIncome": None,
            "NumberOfDependents": None,
        }
    )
    assert 0 <= predict(loaded, client)["probability"] <= 1


def test_load_model_corrupt_artifact_returns_none(lightgbm_dir, tmp_path):
    (tmp_path / "champion.joblib").write_bytes(b"not a joblib file")
    (tmp_path / "champion_metadata.json").write_text("{}", encoding="utf-8")
    assert load_model(tmp_path) is None


def test_load_model_incomplete_metadata_returns_none(lightgbm_dir, tmp_path):
    (tmp_path / "champion.joblib").write_bytes((lightgbm_dir / "champion.joblib").read_bytes())
    (tmp_path / "champion_metadata.json").write_text("{}", encoding="utf-8")
    assert load_model(tmp_path) is None


def test_top_reasons_skips_excluded_variables():
    contributions = pd.Series(
        {"income_unavailable": 0.5, "age": 0.3, "DebtRatio": 0.2, "NumberOfDependents": 0.1}
    )
    reasons = top_reasons(contributions, exclude={"income_unavailable"})
    assert [r["feature"] for r in reasons] == ["age", "DebtRatio", "NumberOfDependents"]


def test_predict_does_not_claim_missing_income_when_given(loaded, monkeypatch):
    from credit_risk.api import service

    flagged = pd.Series(0.0, index=MODEL_FEATURES)
    flagged["income_unavailable"] = 0.9
    flagged["has_special_past_due_code"] = 0.8
    monkeypatch.setattr(service, "feature_contributions", lambda model, row: flagged)
    result = predict(loaded, row())  # income given, no 96/98 codes
    descriptions = [r["description"] for r in result["top_reasons"]]
    assert REASONS["income_unavailable"][1] not in descriptions
    assert REASONS["has_special_past_due_code"][1] not in descriptions


def test_predict_reports_missing_income_and_special_codes(loaded, monkeypatch):
    from credit_risk.api import service

    flagged = pd.Series(0.0, index=MODEL_FEATURES)
    flagged["income_unavailable"] = 0.9
    flagged["has_special_past_due_code"] = 0.8
    monkeypatch.setattr(service, "feature_contributions", lambda model, row: flagged)
    client = row(**{"MonthlyIncome": None, "NumberOfTimes90DaysLate": 98})
    descriptions = [r["description"] for r in predict(loaded, client)["top_reasons"]]
    assert REASONS["income_unavailable"][1] in descriptions
    assert REASONS["has_special_past_due_code"][1] in descriptions


def test_reason_labels_do_not_claim_high_risk():
    assert "alto" not in REASONS["RevolvingUtilizationOfUnsecuredLines"][1].lower()
