import numpy as np
import pytest

from credit_risk.data.quality import (
    check_schema,
    count_conflicting_duplicate_groups,
    count_exact_duplicates,
    domain_issue_profile,
    domain_issues,
    drop_exact_duplicates,
    dtype_report,
    empty_or_constant_columns,
    id_report,
    negative_counts,
    non_numeric_counts,
    past_due_code_overlap,
    target_report,
)
from credit_risk.data.schema import PAST_DUE_COLUMNS, RATIO_COLUMNS, TARGET


def test_check_schema_clean_and_with_differences(raw_frame):
    assert check_schema(raw_frame) == {"missing": [], "unexpected": []}
    altered = raw_frame.drop(columns="age").assign(extra=1)
    assert check_schema(altered) == {"missing": ["age"], "unexpected": ["extra"]}


def test_dtype_report_flags_integer_stored_as_float(raw_frame):
    report = dtype_report(raw_frame)
    assert bool(report.loc["NumberOfDependents", "integer_stored_as_float"]) is True
    assert bool(report.loc["age", "matches"]) is True
    assert bool(report.loc["age", "integer_stored_as_float"]) is False


def test_non_numeric_counts(raw_frame):
    df = raw_frame.astype({"MonthlyIncome": object})
    df.loc[1, "MonthlyIncome"] = "abc"
    counts = non_numeric_counts(df)
    assert counts["MonthlyIncome"] == 1
    assert counts.drop("MonthlyIncome").sum() == 0


def test_id_report(raw_frame):
    assert id_report(raw_frame) == {"unique": True, "nulls": 0, "non_positive": 0}
    zero = raw_frame.rename(index={1: 0})
    assert id_report(zero)["non_positive"] == 1
    duplicated = raw_frame.rename(index={2: 1})
    assert id_report(duplicated)["unique"] is False


def test_empty_or_constant_columns(raw_frame):
    assert empty_or_constant_columns(raw_frame) == {"empty": [], "constant": []}
    df = raw_frame.assign(all_nan=np.nan, same=7)
    assert empty_or_constant_columns(df) == {"empty": ["all_nan"], "constant": ["same"]}


def test_target_report(raw_frame):
    assert target_report(raw_frame, TARGET) == {"values": [0, 1], "nulls": 0}


def test_negative_counts(raw_frame):
    assert negative_counts(raw_frame).sum() == 0
    df = raw_frame.copy()
    df.loc[1, "age"] = -1
    assert negative_counts(df)["age"] == 1


def test_domain_issues(raw_frame):
    issues = domain_issues(raw_frame)
    counts = {(r.check, r.column): (r.kind, r.count) for r in issues.itertuples()}
    assert counts[("age < 18", "age")] == ("impossible", 1)
    for column in PAST_DUE_COLUMNS:
        assert counts[("code 96/98", column)] == ("impossible", 1)
    assert counts[("== 0", "MonthlyIncome")] == ("suspicious", 1)
    for column in RATIO_COLUMNS:
        assert counts[("> 1", column)] == ("suspicious", 1)
    assert list(issues.columns) == ["check", "column", "kind", "count"]


def test_exact_duplicates(raw_frame):
    assert count_exact_duplicates(raw_frame) == 2
    assert drop_exact_duplicates(raw_frame).index.tolist() == [1, 3, 4, 5, 6, 8]


def test_conflicting_duplicate_groups_with_nan(raw_frame):
    assert count_conflicting_duplicate_groups(raw_frame, TARGET) == 2
    deduped = drop_exact_duplicates(raw_frame)
    assert count_conflicting_duplicate_groups(deduped, TARGET) == 2


@pytest.mark.parametrize("function", [count_exact_duplicates, drop_exact_duplicates])
def test_duplicate_functions_do_not_mutate(raw_frame, function):
    before = raw_frame.copy()
    function(raw_frame)
    assert raw_frame.equals(before)


def test_domain_issue_profile(raw_frame):

    profile = domain_issue_profile(raw_frame, TARGET)
    age = profile[profile["check"] == "age < 18"].iloc[0]
    assert age["rows"] == 1
    assert age["pct_rows"] == pytest.approx(12.5)
    assert age["default_rate"] == pytest.approx(1.0)
    assert age["base_default_rate"] == pytest.approx(3 / 8)
    assert age["positives"] == 1
    assert list(profile.columns) == [
        "check",
        "column",
        "kind",
        "rows",
        "pct_rows",
        "default_rate",
        "base_default_rate",
        "positives",
    ]


def test_past_due_code_overlap(raw_frame):

    assert past_due_code_overlap(raw_frame) == {"any": 1, "all": 1}
