import numpy as np
import pytest

from credit_risk.decision.policy import (
    RISK_LEVEL,
    assign_band,
    band_summary,
    capacity_sensitivity,
    cost_sensitivity,
    optimal_threshold,
    review_threshold,
)


@pytest.fixture
def calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 10_000)
    y = rng.binomial(1, p)
    return y, p


def test_optimal_threshold_matches_bayes_rule(calibrated):
    y, p = calibrated
    t_star, cost = optimal_threshold(y, p, fn_cost=5, fp_cost=1)
    assert t_star == pytest.approx(1 / 6, abs=0.03)
    assert cost > 0


def test_optimal_threshold_tie_takes_lowest():
    y = np.array([0, 0, 0])
    p = np.array([0.001, 0.002, 0.003])
    t_star, cost = optimal_threshold(y, p, fn_cost=5, fp_cost=1)
    assert t_star == 0.005
    assert cost == 0


def test_review_threshold_capacity():
    p = np.linspace(0, 1, 1001)
    t_low = review_threshold(p, t_star=0.5, capacity=0.10)
    assert t_low == pytest.approx(0.4, abs=0.01)
    in_band = np.mean((p >= t_low) & (p < 0.5))
    assert in_band == pytest.approx(0.10, abs=0.005)


def test_review_threshold_clips_at_zero():
    p = np.linspace(0, 1, 1001)
    assert review_threshold(p, t_star=0.05, capacity=0.20) == 0.0


def test_review_threshold_clips_at_zero_even_if_min_score_is_positive():
    p = np.linspace(0.1, 1, 1001)
    assert review_threshold(p, t_star=0.2, capacity=0.5) == 0.0


def test_assign_band_boundaries():
    bands = assign_band(np.array([0.1, 0.4, 0.5]), t_low=0.4, t_star=0.5)
    assert bands.tolist() == ["approve", "review", "reject"]
    assert RISK_LEVEL == {"approve": "low", "review": "medium", "reject": "high"}


def test_band_summary_shares_sum_to_one(calibrated):
    y, p = calibrated
    summary = band_summary(y, p, t_low=0.1, t_star=0.2)
    assert list(summary.index) == ["approve", "review", "reject"]
    assert list(summary.columns) == ["rows", "share", "default_rate"]
    assert summary["share"].sum() == pytest.approx(1.0)
    assert summary["rows"].sum() == len(y)
    assert summary["default_rate"].is_monotonic_increasing


def test_band_summary_empty_band_has_nan_default_rate():
    summary = band_summary([0, 1], [0.1, 0.9], t_low=0.5, t_star=0.5)
    assert summary.loc["review", "rows"] == 0
    assert np.isnan(summary.loc["review", "default_rate"])


def test_cost_sensitivity_threshold_falls_as_fn_cost_rises(calibrated):
    y, p = calibrated
    table = cost_sensitivity(y, p, ratios=(3, 5, 10, 20), fp_cost=1, capacity=0.10)
    assert list(table.columns) == [
        "fn_fp_ratio",
        "t_star",
        "t_low",
        "approval_rate",
        "default_rate_approved",
        "expected_cost",
    ]
    assert table["fn_fp_ratio"].tolist() == [3, 5, 10, 20]
    assert table["t_star"].is_monotonic_decreasing
    assert (table["t_low"] <= table["t_star"]).all()


def test_capacity_sensitivity_review_grows_with_capacity(calibrated):
    y, p = calibrated
    table = capacity_sensitivity(y, p, t_star=0.2, capacities=(0.05, 0.10, 0.20))
    assert table["review_rows"].is_monotonic_increasing
    assert table["t_low"].is_monotonic_decreasing
    for column in ("approve_rows", "approve_default_rate", "reject_rows", "reject_default_rate"):
        assert column in table.columns
