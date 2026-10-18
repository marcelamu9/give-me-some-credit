import pandas as pd

from credit_risk.models.selection import select_champion


def table(costs, briers=(0.05, 0.05, 0.05)):
    return pd.DataFrame(
        {
            "model": ["logistic", "random_forest", "lightgbm"],
            "expected_cost": list(costs),
            "brier": list(briers),
        }
    )


def test_select_champion_lowest_cost():
    assert select_champion(table((1.00, 0.90, 0.80))) == "lightgbm"


def test_select_champion_prefers_simpler_within_tolerance():
    assert select_champion(table((0.81, 0.90, 0.80))) == "logistic"


def test_select_champion_rejects_simpler_with_bad_brier():
    assert select_champion(table((0.81, 0.90, 0.80), briers=(0.06, 0.05, 0.05))) == "lightgbm"


def test_select_champion_skips_to_next_simplest_within_tolerance():
    costs = (0.82, 0.805, 0.80)
    assert select_champion(table(costs, briers=(0.06, 0.05, 0.05))) == "random_forest"


def test_select_champion_is_independent_of_row_order():
    shuffled = table((0.81, 0.90, 0.80)).iloc[::-1]
    assert select_champion(shuffled) == "logistic"
