"""Expected structure of cs-training.csv (see docs/data_dictionary.md)."""

ID_NAME = "id"
TARGET = "SeriousDlqin2yrs"

PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
RATIO_COLUMNS = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
COLUMNS = [TARGET, *FEATURES]

FLOAT_COLUMNS = [*RATIO_COLUMNS, "MonthlyIncome"]
INTEGER_COLUMNS = [column for column in COLUMNS if column not in FLOAT_COLUMNS]

# Codes the source uses for "unknown" in the past-due counters.
SPECIAL_PAST_DUE_CODES = (96, 98)
