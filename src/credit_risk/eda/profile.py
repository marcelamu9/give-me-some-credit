"""Statistical profiling functions for the EDA. None of them modifies its input."""

import pandas as pd

PERCENTILES = {
    "p1": 0.01,
    "p5": 0.05,
    "p25": 0.25,
    "p50": 0.5,
    "p75": 0.75,
    "p95": 0.95,
    "p99": 0.99,
}


def target_summary(df: pd.DataFrame, target: str) -> dict:
    positives = int(df[target].sum())
    return {"rows": len(df), "positives": positives, "default_rate": positives / len(df)}


def missing_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = {}
    for column in df.columns.drop(target):
        is_missing = df[column].isna()
        rows[column] = {
            "missing_count": int(is_missing.sum()),
            "missing_pct": float(is_missing.mean() * 100),
            "default_rate_missing": float(df.loc[is_missing, target].mean()),
            "default_rate_present": float(df.loc[~is_missing, target].mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def percentile_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(
        {name: df[columns].quantile(q) for name, q in PERCENTILES.items()}, index=columns
    )
    table["max"] = df[columns].max()
    return table


def outlier_counts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = {}
    for column in columns:
        series = df[column]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        outside = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        rows[column] = {
            "iqr_outliers": int(outside.sum()),
            "above_p99": int((series > series.quantile(0.99)).sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def spearman_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].corr(method="spearman")


def default_rate_by_decile(df: pd.DataFrame, column: str, target: str, q: int = 10) -> pd.DataFrame:
    """Default rate per quantile bin; heavy ties merge bins, and NaN forms a final 'missing' bin."""
    bins = pd.qcut(df[column], q=q, duplicates="drop")
    grouped = df.groupby(bins, observed=True)[target].agg(rows="size", default_rate="mean")
    table = grouped.reset_index(drop=True).assign(bin=[str(interval) for interval in grouped.index])
    missing = df[column].isna()
    if missing.any():
        extra = pd.DataFrame(
            {
                "rows": [int(missing.sum())],
                "default_rate": [float(df.loc[missing, target].mean())],
                "bin": ["missing"],
            }
        )
        table = pd.concat([table, extra], ignore_index=True)
    return table[["bin", "rows", "default_rate"]]


def debt_ratio_by_income_status(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """DebtRatio behaviour by MonthlyIncome status (missing / zero / positive).

    Without income the ratio cannot be computed; an integer-valued DebtRatio there suggests the
    column holds a monthly amount instead of a ratio.
    """
    income = df["MonthlyIncome"]
    statuses = {"missing": income.isna(), "zero": income == 0, "positive": income > 0}
    rows = {}
    for status, mask in statuses.items():
        ratio = df.loc[mask, "DebtRatio"]
        rows[status] = {
            "rows": int(mask.sum()),
            "debt_ratio_median": float(ratio.median()),
            "pct_debt_ratio_gt_1": 100 * float((ratio > 1).mean()),
            "pct_debt_ratio_integer": 100 * float((ratio % 1 == 0).mean()),
            "default_rate": float(df.loc[mask, target].mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _bin_label(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def median_by(df: pd.DataFrame, value: str, by: str, q: int | None = None) -> pd.DataFrame:
    """Median of `value` per group of `by` (exact values, or `q` quantile bins); NaN group last."""
    has_group = df[by].notna()
    keys = pd.qcut(df.loc[has_group, by], q=q, duplicates="drop") if q else df.loc[has_group, by]
    rows = []
    for key, group in df.loc[has_group, value].groupby(keys, observed=True, sort=True):
        rows.append((_bin_label(key), group))
    if (~has_group).any():
        rows.append(("missing", df.loc[~has_group, value]))
    return pd.DataFrame(
        [
            {
                "bin": label,
                "rows": len(values),
                "median": float(values.median()),
                "missing_pct": 100 * float(values.isna().mean()),
            }
            for label, values in rows
        ]
    )


def missing_overlap(df: pd.DataFrame, a: str, b: str) -> dict[str, int]:
    """How many rows miss `a`, miss `b`, and miss both."""
    a_missing, b_missing = df[a].isna(), df[b].isna()
    return {
        "a_missing": int(a_missing.sum()),
        "b_missing": int(b_missing.sum()),
        "both": int((a_missing & b_missing).sum()),
    }


def debt_ratio_profile_with_income(df: pd.DataFrame, target: str) -> dict[str, float | int]:
    """DebtRatio among rows with positive income: how many exceed 1, their risk, and the tail."""
    with_income = df[df["MonthlyIncome"] > 0]
    ratio = with_income["DebtRatio"]
    over_one = ratio > 1
    return {
        "rows": len(with_income),
        "rows_gt_1": int(over_one.sum()),
        "default_rate_gt_1": float(with_income.loc[over_one, target].mean()),
        "default_rate_le_1": float(with_income.loc[~over_one, target].mean()),
        "p95": float(ratio.quantile(0.95)),
        "p99": float(ratio.quantile(0.99)),
        "p99_9": float(ratio.quantile(0.999)),
        "max": float(ratio.max()),
    }
