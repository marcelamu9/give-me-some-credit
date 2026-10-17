"""Structural data-quality checks (Phase 1): they report, only exact duplicates are dropped."""

import pandas as pd

from credit_risk.data.schema import (
    COLUMNS,
    FEATURES,
    INTEGER_COLUMNS,
    PAST_DUE_COLUMNS,
    RATIO_COLUMNS,
    SPECIAL_PAST_DUE_CODES,
)


def check_schema(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "missing": [c for c in COLUMNS if c not in df.columns],
        "unexpected": [c for c in df.columns if c not in COLUMNS],
    }


def dtype_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for column in df.columns.intersection(COLUMNS):
        series = df[column]
        expected = "integer" if column in INTEGER_COLUMNS else "float"
        is_integer = pd.api.types.is_integer_dtype(series)
        is_float = pd.api.types.is_float_dtype(series)
        non_null = series.dropna()
        rows[column] = {
            "dtype": str(series.dtype),
            "expected": expected,
            "matches": is_integer if expected == "integer" else (is_float or is_integer),
            "integer_stored_as_float": expected == "integer"
            and is_float
            and bool((non_null % 1 == 0).all()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def non_numeric_counts(df: pd.DataFrame) -> pd.Series:
    counts = {}
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        counts[column] = int((converted.isna() & df[column].notna()).sum())
    return pd.Series(counts)


def id_report(df: pd.DataFrame) -> dict[str, int | bool]:
    ids = pd.to_numeric(df.index.to_series(), errors="coerce")
    return {
        "unique": bool(df.index.is_unique),
        "nulls": int(ids.isna().sum()),
        "non_positive": int((ids <= 0).sum()),
    }


def empty_or_constant_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    empty = [c for c in df.columns if df[c].isna().all()]
    constant = [c for c in df.columns if c not in empty and df[c].nunique(dropna=True) == 1]
    return {"empty": empty, "constant": constant}


def target_report(df: pd.DataFrame, target: str) -> dict:
    return {
        "values": sorted(df[target].dropna().unique().tolist()),
        "nulls": int(df[target].isna().sum()),
    }


def negative_counts(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes("number")
    return (numeric < 0).sum()


def _domain_masks(df: pd.DataFrame) -> list[tuple[str, str, str, pd.Series]]:
    """(check, column, kind, boolean mask) for every domain check."""
    masks = [("age < 18", "age", "impossible", df["age"] < 18)]
    for column in PAST_DUE_COLUMNS:
        masks.append(("code 96/98", column, "impossible", df[column].isin(SPECIAL_PAST_DUE_CODES)))
    masks.append(("== 0", "MonthlyIncome", "suspicious", df["MonthlyIncome"] == 0))
    for column in RATIO_COLUMNS:
        masks.append(("> 1", column, "suspicious", df[column] > 1))
    return masks


def domain_issues(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [(check, column, kind, int(mask.sum())) for check, column, kind, mask in _domain_masks(df)],
        columns=["check", "column", "kind", "count"],
    )


def domain_issue_profile(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Per check: affected rows, their default rate vs. the base rate, and positives at stake."""
    base_rate = float(df[target].mean())
    rows = []
    for check, column, kind, mask in _domain_masks(df):
        flagged = df.loc[mask, target]
        rows.append(
            {
                "check": check,
                "column": column,
                "kind": kind,
                "rows": int(mask.sum()),
                "pct_rows": 100 * float(mask.mean()),
                "default_rate": float(flagged.mean()) if len(flagged) else float("nan"),
                "base_default_rate": base_rate,
                "positives": int(flagged.sum()),
            }
        )
    return pd.DataFrame(rows)


def past_due_code_overlap(df: pd.DataFrame) -> dict[str, int]:
    """Rows with a 96/98 code in any vs. in all of the past-due columns."""
    coded = df[PAST_DUE_COLUMNS].isin(SPECIAL_PAST_DUE_CODES)
    return {"any": int(coded.any(axis=1).sum()), "all": int(coded.all(axis=1).sum())}


def count_exact_duplicates(df: pd.DataFrame) -> int:
    """Rows identical in every column (ID ignored, NaN equal), beyond their first occurrence."""
    return int(df.duplicated(keep="first").sum())


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.duplicated(keep="first")]


def count_conflicting_duplicate_groups(df: pd.DataFrame, target: str) -> int:
    """Groups of rows with the same features (NaN equal) but more than one target value."""
    return int((df.groupby(FEATURES, dropna=False)[target].nunique() > 1).sum())
