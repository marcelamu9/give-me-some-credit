import pandas as pd

from credit_risk.models.selection import select_champion


def table(costs, briers=(0.05, 0.05, 0.05), latencies=(5.0, 180.0, 38.0)):
    return pd.DataFrame(
        {
            "model": ["logistic", "random_forest", "lightgbm"],
            "expected_cost": list(costs),
            "brier": list(briers),
            "latency_ms": list(latencies),
        }
    )


def test_select_champion_lowest_cost():
    assert select_champion(table((1.00, 0.90, 0.80))) == "lightgbm"


def test_select_champion_prefers_logistic_within_tolerance():
    assert select_champion(table((0.81, 0.90, 0.80))) == "logistic"


def test_select_champion_rejects_logistic_with_bad_brier():
    assert select_champion(table((0.81, 0.90, 0.80), briers=(0.06, 0.05, 0.05))) == "lightgbm"


def test_select_champion_prefers_faster_tree_within_tolerance():
    # Random Forest is cheapest, but LightGBM is within 2 % and much faster.
    assert select_champion(table((0.90, 0.80, 0.805))) == "lightgbm"


def test_select_champion_faster_tree_needs_acceptable_brier():
    costs, briers = (0.90, 0.80, 0.805), (0.05, 0.05, 0.06)
    assert select_champion(table(costs, briers=briers)) == "random_forest"


def test_select_champion_real_run_picks_lightgbm():
    # Valid results of the first training run (docs/model_card.md, ADR-002).
    costs, briers = (0.2219, 0.2160, 0.2141), (0.0498, 0.0492, 0.0490)
    assert select_champion(table(costs, briers, latencies=(49.0, 179.6, 37.7))) == "lightgbm"


def test_select_champion_is_independent_of_row_order():
    shuffled = table((0.81, 0.90, 0.80)).iloc[::-1]
    assert select_champion(shuffled) == "logistic"
