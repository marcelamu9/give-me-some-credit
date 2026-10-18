import numpy as np
import pandas as pd
import pytest

from credit_risk.eda.profile import (
    debt_ratio_by_income_status,
    debt_ratio_profile_with_income,
    default_rate_by_decile,
    median_by,
    missing_overlap,
    missing_summary,
    outlier_counts,
    percentile_table,
    spearman_matrix,
    target_summary,
)


def test_target_summary():
    df = pd.DataFrame({"y": [0, 0, 0, 1]})
    assert target_summary(df, "y") == {"rows": 4, "positives": 1, "default_rate": 0.25}


def test_missing_summary_informative_missingness():
    df = pd.DataFrame(
        {"MonthlyIncome": [np.nan, np.nan, 1, 2], "full": [1, 2, 3, 4], "y": [1, 1, 0, 0]}
    )
    summary = missing_summary(df, "y")
    assert summary.index.tolist() == ["MonthlyIncome", "full"]
    income = summary.loc["MonthlyIncome"]
    assert income["missing_count"] == 2
    assert income["missing_pct"] == 50.0
    assert income["default_rate_missing"] == 1.0
    assert income["default_rate_present"] == 0.0
    assert summary.loc["full", "missing_count"] == 0
    assert np.isnan(summary.loc["full", "default_rate_missing"])


def test_percentile_table():
    df = pd.DataFrame({"x": range(1, 101)})
    table = percentile_table(df, ["x"])
    assert table.columns.tolist() == ["p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"]
    assert table.loc["x", "p50"] == 50.5
    assert table.loc["x", "max"] == 100


def test_outlier_counts():
    df = pd.DataFrame({"x": [*range(1, 100), 1000]})
    counts = outlier_counts(df, ["x"])
    assert counts.loc["x", "iqr_outliers"] == 1
    assert counts.loc["x", "above_p99"] == 1


def test_spearman_matrix_is_rank_based():
    a = np.arange(1, 51)
    df = pd.DataFrame({"a": a, "b": a**3})
    corr = spearman_matrix(df, ["a", "b"])
    assert corr.loc["a", "b"] == pytest.approx(1.0)
    assert corr.loc["a", "a"] == pytest.approx(1.0)


def test_default_rate_by_decile_monotone():
    x = np.arange(1, 101)
    df = pd.DataFrame({"x": x, "y": (x > 90).astype(int)})
    table = default_rate_by_decile(df, "x", "y")
    assert len(table) == 10
    assert (table["rows"] == 10).all()
    assert table["default_rate"].iloc[0] == 0.0
    assert table["default_rate"].iloc[-1] == 1.0


def test_default_rate_by_decile_handles_ties():
    df = pd.DataFrame({"x": [0] * 90 + list(range(1, 11)), "y": [0] * 100})
    table = default_rate_by_decile(df, "x", "y")
    assert len(table) < 10
    assert table["rows"].sum() == 100


def test_default_rate_by_decile_keeps_missing_bin():
    df = pd.DataFrame({"x": [*range(1, 21), *[np.nan] * 5], "y": [0] * 20 + [1] * 5})
    table = default_rate_by_decile(df, "x", "y")
    last = table.iloc[-1]
    assert last["bin"] == "missing"
    assert last["rows"] == 5
    assert last["default_rate"] == 1.0
    assert table["rows"].sum() == 25


def test_profile_functions_do_not_mutate_input():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, np.nan, 4.0, 100.0], "z": [1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 0]}
    )
    before = df.copy()
    target_summary(df, "y")
    missing_summary(df, "y")
    percentile_table(df, ["x", "z"])
    outlier_counts(df, ["x", "z"])
    spearman_matrix(df, ["x", "z"])
    default_rate_by_decile(df, "x", "y", q=2)
    pd.testing.assert_frame_equal(df, before)


def test_debt_ratio_by_income_status():
    df = pd.DataFrame(
        {
            "MonthlyIncome": [np.nan, np.nan, 0.0, 1000.0, 2000.0, 3000.0],
            "DebtRatio": [1500.0, 800.0, 900.0, 0.25, 0.5, 1.5],
            "y": [1, 0, 0, 0, 1, 0],
        }
    )
    result = debt_ratio_by_income_status(df, "y")
    assert result.index.tolist() == ["missing", "zero", "positive"]
    assert result["rows"].tolist() == [2, 1, 3]
    assert result.loc["missing", "debt_ratio_median"] == pytest.approx(1150.0)
    assert result.loc["missing", "pct_debt_ratio_gt_1"] == pytest.approx(100.0)
    assert result.loc["missing", "pct_debt_ratio_integer"] == pytest.approx(100.0)
    assert result.loc["positive", "pct_debt_ratio_gt_1"] == pytest.approx(100 / 3)
    assert result.loc["positive", "pct_debt_ratio_integer"] == pytest.approx(0.0)
    assert result.loc["positive", "default_rate"] == pytest.approx(1 / 3)


def test_median_by_exact_values():
    df = pd.DataFrame({"g": [0, 0, 1, 1, 1], "v": [10.0, 20.0, 5.0, np.nan, 7.0]})
    result = median_by(df, "v", "g")
    assert result["bin"].tolist() == ["0", "1"]
    assert result["rows"].tolist() == [2, 3]
    assert result["median"].tolist() == [15.0, 6.0]
    assert result["missing_pct"].tolist() == pytest.approx([0.0, 100 / 3])


def test_median_by_quantile_bins_and_missing_group():
    df = pd.DataFrame({"g": [*range(1, 21), np.nan], "v": [*range(1, 21), 99.0]})
    result = median_by(df, "v", "g", q=4)
    assert len(result) == 5
    assert result["bin"].iloc[-1] == "missing"
    assert result["median"].iloc[-1] == 99.0
    assert result["rows"].sum() == 21


def test_missing_overlap():
    df = pd.DataFrame({"a": [np.nan, np.nan, 1.0, 2.0], "b": [np.nan, 1.0, 1.0, np.nan]})
    assert missing_overlap(df, "a", "b") == {"a_missing": 2, "b_missing": 2, "both": 1}


def test_debt_ratio_profile_with_income():
    df = pd.DataFrame(
        {
            "MonthlyIncome": [np.nan, 0.0, *[1000.0] * 10],
            "DebtRatio": [5000.0, 900.0, *[0.1 * i for i in range(1, 9)], 1.5, 3.0],
            "y": [1, 1, *[0] * 8, 1, 0],
        }
    )
    result = debt_ratio_profile_with_income(df, "y")
    assert result["rows"] == 10
    assert result["rows_gt_1"] == 2
    assert result["default_rate_gt_1"] == pytest.approx(0.5)
    assert result["default_rate_le_1"] == pytest.approx(0.0)
    assert result["max"] == 3.0
    assert set(result) == {
        "rows",
        "rows_gt_1",
        "default_rate_gt_1",
        "default_rate_le_1",
        "p95",
        "p99",
        "p99_9",
        "max",
    }
