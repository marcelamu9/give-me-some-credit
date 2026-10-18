"""Champion selection: lowest expected cost, then the cheapest model to operate and explain."""

import pandas as pd

INTERPRETABLE_MODEL = "logistic"


def select_champion(
    table: pd.DataFrame, tolerance: float = 0.02, brier_tolerance: float = 0.10
) -> str:
    """Pick the champion from a table with ``model``, ``expected_cost``, ``brier``, ``latency_ms``.

    Eligible models are those within ``tolerance`` (relative) of the lowest cost whose Brier
    score is at most ``brier_tolerance`` (relative) worse than the lowest-cost model's. Among
    them, the interpretable logistic regression wins; otherwise the fastest model (lowest
    prediction latency) wins, with cost as the tie-breaker.
    """
    best = table.loc[table["expected_cost"].idxmin()]
    eligible = table[
        (table["expected_cost"] <= best["expected_cost"] * (1 + tolerance))
        & (table["brier"] <= best["brier"] * (1 + brier_tolerance))
    ]
    if (eligible["model"] == INTERPRETABLE_MODEL).any():
        return INTERPRETABLE_MODEL
    return str(eligible.sort_values(["latency_ms", "expected_cost"]).iloc[0]["model"])
