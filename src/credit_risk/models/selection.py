"""Champion selection: lowest expected cost, simplest model within tolerance."""

import pandas as pd

from credit_risk.models.candidates import COMPLEXITY_ORDER


def select_champion(
    table: pd.DataFrame, tolerance: float = 0.02, brier_tolerance: float = 0.10
) -> str:
    """Pick the champion from a table with columns ``model``, ``expected_cost`` and ``brier``.

    The model with the lowest cost wins; among those within ``tolerance`` (relative) of that
    cost, the simplest one wins unless its Brier score is more than ``brier_tolerance``
    (relative) worse than the lowest-cost model's.
    """
    best = table.loc[table["expected_cost"].idxmin()]
    close = table[table["expected_cost"] <= best["expected_cost"] * (1 + tolerance)]
    for name in COMPLEXITY_ORDER:
        match = close[close["model"] == name]
        if match.empty:
            continue
        if match.iloc[0]["brier"] <= best["brier"] * (1 + brier_tolerance):
            return name
    return str(best["model"])
