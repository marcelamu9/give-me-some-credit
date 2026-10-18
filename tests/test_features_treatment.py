import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from credit_risk.data.schema import FEATURES, PAST_DUE_COLUMNS, TARGET
from credit_risk.features.treatment import (
    COUNT_COLUMNS,
    MODEL_FEATURES,
    IncomeImputer,
    LogisticScaling,
    RatioTreatment,
    SpecialPastDueRecoder,
    build_preprocessor,
    drop_underage,
)

REVOLVING = "RevolvingUtilizationOfUnsecuredLines"


def make_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(8.5, 0.5, n)
    income[:30] = np.nan
    income[30:40] = 0.0
    unavailable = np.isnan(income) | (income == 0)
    debt_ratio = np.where(unavailable, rng.uniform(500, 5000, n), rng.uniform(0, 2, n))
    dependents = rng.integers(0, 7, n).astype(float)
    dependents[:25] = np.nan
    late_30 = rng.poisson(0.3, n)
    late_90 = rng.poisson(0.1, n)
    late_60 = rng.poisson(0.1, n)
    late_30[50:53] = 98
    late_90[50:53] = 98
    late_60[50:53] = 98
    late_90[60] = 96
    return pd.DataFrame(
        {
            TARGET: rng.integers(0, 2, n),
            REVOLVING: np.append(rng.uniform(0, 1.5, n - 1), 5000.0),
            "age": rng.integers(20, 81, n),
            "NumberOfTime30-59DaysPastDueNotWorse": late_30,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, n),
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": rng.poisson(1, n),
            "NumberOfTime60-89DaysPastDueNotWorse": late_60,
            "NumberOfDependents": dependents,
        }
    )


def chain(df: pd.DataFrame) -> tuple[SpecialPastDueRecoder, IncomeImputer, pd.DataFrame]:
    recoder = SpecialPastDueRecoder()
    imputer = IncomeImputer()
    return recoder, imputer, imputer.fit_transform(recoder.fit_transform(df))


def row(**values) -> pd.DataFrame:
    base = {
        REVOLVING: 0.3,
        "age": 40,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.4,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0,
    }
    base.update(values)
    return pd.DataFrame([base])


def test_drop_underage():
    df = pd.DataFrame({"age": [17, 18, 40], "x": [1, 2, 3]})
    result = drop_underage(df)
    assert result["age"].tolist() == [18, 40]
    assert len(df) == 3


def test_recoder_flags_and_zeroes_codes():
    df = pd.concat(
        [
            row(**{c: 98 for c in PAST_DUE_COLUMNS}),
            row(**{PAST_DUE_COLUMNS[0]: 2}),
        ],
        ignore_index=True,
    )
    out = SpecialPastDueRecoder().fit_transform(df)
    assert out["has_special_past_due_code"].tolist() == [1, 0]
    assert out.loc[0, PAST_DUE_COLUMNS].tolist() == [0, 0, 0]
    assert out.loc[1, PAST_DUE_COLUMNS[0]] == 2


def test_recoder_selects_features_in_order():
    df = make_frame(70)
    shuffled = df[list(reversed(df.columns))]
    out = SpecialPastDueRecoder().fit_transform(shuffled)
    assert list(out.columns) == [*FEATURES, "has_special_past_due_code"]


def test_imputer_flags_unavailable_income():
    df = pd.concat(
        [row(MonthlyIncome=np.nan), row(MonthlyIncome=0.0), row(MonthlyIncome=5000.0)],
        ignore_index=True,
    )
    out = IncomeImputer().fit_transform(df)
    assert out["income_unavailable"].tolist() == [1, 1, 0]


def test_imputer_leaves_no_missing():
    _, _, out = chain(make_frame())
    assert out["MonthlyIncome"].notna().all()
    assert out["NumberOfDependents"].notna().all()
    assert (out["MonthlyIncome"] > 0).all()


def _group_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    target_group = pd.concat([row(age=30, NumberOfDependents=2.0, MonthlyIncome=3000.0)] * 40)
    others = pd.concat(
        [
            row(age=int(a), NumberOfDependents=0.0, MonthlyIncome=9000.0)
            for a in rng.integers(20, 81, 360)
        ]
    )
    return pd.concat([target_group, others], ignore_index=True)


def test_imputer_uses_group_median():
    imputer = IncomeImputer().fit(_group_frame())
    out = imputer.transform(row(age=30, NumberOfDependents=2.0, MonthlyIncome=np.nan))
    assert out.loc[0, "MonthlyIncome"] == 3000.0


def test_imputer_falls_back_for_unseen_group():
    imputer = IncomeImputer().fit(_group_frame())
    out = imputer.transform(row(age=95, NumberOfDependents=9.0, MonthlyIncome=np.nan))
    top_bin = len(imputer.age_edges_) - 2
    assert out.loc[0, "MonthlyIncome"] == imputer.age_medians_[top_bin]
    assert out["MonthlyIncome"].notna().all()


def test_imputer_learns_only_from_fit_data():
    fit_frame = make_frame(seed=0)
    imputer = IncomeImputer().fit(fit_frame)
    cap = imputer.income_cap_
    medians = dict(imputer.group_medians_)
    other = make_frame(seed=5)
    other["MonthlyIncome"] = other["MonthlyIncome"].where(
        other["MonthlyIncome"].isna(), 1_000_000.0
    )
    out = imputer.transform(other)
    assert imputer.income_cap_ == cap
    assert imputer.group_medians_ == medians
    observed = out.loc[out["income_unavailable"] == 0, "MonthlyIncome"]
    assert (observed == cap).all()


def test_ratio_recomputed_when_income_unavailable():
    recoder, imputer, imputed = chain(make_frame())
    ratio = RatioTreatment().fit(imputed)
    frame = imputed.iloc[:1].copy()
    frame["income_unavailable"] = 1
    frame["MonthlyIncome"] = 3000.0
    frame["DebtRatio"] = 1500.0
    out = ratio.transform(frame)
    assert out.loc[out.index[0], "DebtRatio"] == 0.5
    assert np.isfinite(out["DebtRatio"]).all()


def test_ratio_caps():
    _, _, imputed = chain(make_frame())
    ratio = RatioTreatment().fit(imputed)
    frame = imputed.iloc[:2].copy()
    frame["income_unavailable"] = 0
    frame["DebtRatio"] = 1e9
    frame[REVOLVING] = 1e9
    out = ratio.transform(frame)
    assert (out["DebtRatio"] == ratio.debt_ratio_cap_).all()
    assert (out[REVOLVING] == ratio.revolving_cap_).all()


@pytest.mark.parametrize("kind", ["tree", "logistic"])
def test_single_row_transform(kind):
    df = make_frame()
    pipe = build_preprocessor(kind).fit(df)
    out = pipe.transform(df.iloc[[3]])
    assert out.shape == (1, len(MODEL_FEATURES))
    assert not out.isna().any().any()


def test_single_row_reuses_learned_cuts():
    df = make_frame()
    pipe = build_preprocessor("tree").fit(df)
    full = pipe.transform(df)
    one_by_one = pd.concat([pipe.transform(df.iloc[[i]]) for i in range(10)])
    assert_frame_equal(full.iloc[:10], one_by_one)


def test_unseen_group_in_pipeline_never_nan():
    df = make_frame()
    pipe = build_preprocessor("tree").fit(df)
    out = pipe.transform(row(age=99, NumberOfDependents=12.0, MonthlyIncome=np.nan))
    assert not out.isna().any().any()


def test_zero_income_with_amount_ratio_is_finite():
    df = make_frame()
    pipe = build_preprocessor("tree").fit(df)
    out = pipe.transform(row(MonthlyIncome=0.0, DebtRatio=2500.0))
    assert np.isfinite(out.to_numpy(dtype=float)).all()


@pytest.mark.parametrize("kind", ["tree", "logistic"])
def test_preprocessor_output_columns(kind):
    df = make_frame()
    out = build_preprocessor(kind).fit_transform(df)
    assert list(out.columns) == MODEL_FEATURES


def test_logistic_scaling_clips_counts_and_logs():
    _, _, imputed = chain(make_frame())
    scaling = LogisticScaling().fit(imputed)
    frame = imputed.iloc[:1].copy()
    column = COUNT_COLUMNS[3]
    frame[column] = 10_000
    out = scaling.transform(frame)
    assert out.loc[out.index[0], column] == scaling.count_caps_[column]
    expected = np.log1p(frame["MonthlyIncome"])
    assert out["MonthlyIncome"].iloc[0] == pytest.approx(expected.iloc[0])


def test_transformers_do_not_mutate_input():
    df = make_frame()
    before = df.copy()
    recoder, imputer, imputed = chain(df)
    assert_frame_equal(df, before)
    imputed_before = imputed.copy()
    ratio = RatioTreatment().fit(imputed)
    ratio.transform(imputed)
    scaling = LogisticScaling().fit(imputed)
    scaling.transform(imputed)
    assert_frame_equal(imputed, imputed_before)
