import math

import pytest
from pydantic import ValidationError

from credit_risk.api.schemas import ClientFeatures
from credit_risk.data.schema import FEATURES

VALID = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.25,
    "MonthlyIncome": 6500.0,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}


def test_valid_payload_parses_by_kaggle_names():
    frame = ClientFeatures.model_validate(VALID).to_frame()
    assert list(frame.columns) == FEATURES
    assert len(frame) == 1


def test_nullable_fields():
    payload = {**VALID, "MonthlyIncome": None, "NumberOfDependents": None}
    frame = ClientFeatures.model_validate(payload).to_frame()
    assert math.isnan(frame.loc[0, "MonthlyIncome"])
    assert math.isnan(frame.loc[0, "NumberOfDependents"])


def _without(key):
    return {k: v for k, v in VALID.items() if k != key}


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID, "age": 17},
        {**VALID, "age": 121},
        {**VALID, "age": 30.5},
        {**VALID, "age": "abc"},
        {**VALID, "NumberOfTimes90DaysLate": -1},
        {**VALID, "NumberOfTimes90DaysLate": 99},
        {**VALID, "NumberOfDependents": 21},
        {**VALID, "DebtRatio": -0.1},
        {**VALID, "MonthlyIncome": -1},
        _without("age"),
        {**VALID, "foo": 1},
        {**_without("NumberOfTime30-59DaysPastDueNotWorse"), "past_due_30_59": 0},
    ],
)
def test_invalid_payload_is_rejected(payload):
    with pytest.raises(ValidationError):
        ClientFeatures.model_validate(payload)


def test_special_codes_accepted():
    payload = {
        **VALID,
        "NumberOfTime30-59DaysPastDueNotWorse": 98,
        "NumberOfTimes90DaysLate": 98,
        "NumberOfTime60-89DaysPastDueNotWorse": 96,
    }
    assert ClientFeatures.model_validate(payload)


@pytest.mark.parametrize("key", ["MonthlyIncome", "NumberOfDependents"])
def test_optional_fields_can_be_omitted(key):
    frame = ClientFeatures.model_validate(_without(key)).to_frame()
    assert math.isnan(frame[key].iloc[0])


@pytest.mark.parametrize(
    ("key", "value"),
    [("age", "30"), ("DebtRatio", "0.3"), ("NumberOfTimes90DaysLate", "1"), ("age", True)],
)
def test_strings_and_booleans_are_rejected_in_numeric_fields(key, value):
    with pytest.raises(ValidationError):
        ClientFeatures.model_validate({**VALID, key: value})


def test_integers_are_accepted_in_float_fields():
    frame = ClientFeatures.model_validate(
        {**VALID, "DebtRatio": 1, "MonthlyIncome": 5000}
    ).to_frame()
    assert frame["DebtRatio"].iloc[0] == 1.0
