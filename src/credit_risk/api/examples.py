"""Demo payloads preloaded in /docs (POST /predict)."""

LOW_RISK = {
    "RevolvingUtilizationOfUnsecuredLines": 0.10,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.25,
    "MonthlyIncome": 6500,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}

SAME_CLIENT_WITH_DELAYS = {
    **LOW_RISK,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "RevolvingUtilizationOfUnsecuredLines": 0.95,
}

INVALID_AGE = {**LOW_RISK, "age": 10}

OPENAPI_EXAMPLES = {
    "low_risk": {
        "summary": "Cliente de bajo riesgo",
        "value": LOW_RISK,
    },
    "same_client_with_delays": {
        "summary": "El mismo cliente con atrasos y la tarjeta casi al límite",
        "value": SAME_CLIENT_WITH_DELAYS,
    },
    "invalid_age": {
        "summary": "Cliente inválido (edad 10): devuelve 422",
        "value": INVALID_AGE,
    },
}
