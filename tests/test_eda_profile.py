import numpy as np
import pandas as pd
import pytest

from credit_risk.eda.profile import (
    default_rate_by_decile,
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
