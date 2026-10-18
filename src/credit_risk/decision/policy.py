"""Cost-based threshold, three-band policy and sensitivity analyses (pure functions on y / p)."""

import numpy as np
import pandas as pd

from credit_risk.evaluation.metrics import expected_cost

THRESHOLD_GRID = np.round(np.arange(0.005, 0.996, 0.005), 3)
BANDS = ("approve", "review", "reject")
RISK_LEVEL = {"approve": "low", "review": "medium", "reject": "high"}


def optimal_threshold(y_true, p, fn_cost, fp_cost, grid=THRESHOLD_GRID) -> tuple[float, float]:
    """Threshold on ``grid`` with the lowest expected cost (the lowest one on ties)."""
    costs = np.array([expected_cost(y_true, p, t, fn_cost, fp_cost) for t in grid])
    best = int(np.argmin(costs))
    return float(grid[best]), float(costs[best])


def review_threshold(p, t_star: float, capacity: float) -> float:
    """Lower threshold so that about ``capacity`` of the clients fall in ``[t_low, t_star)``."""
    p = np.asarray(p, dtype=float)
    below = float(np.mean(p < t_star))
    if below <= capacity:
        return 0.0
    t_low = float(np.quantile(p, below - capacity))
    return min(max(t_low, 0.0), t_star)


def assign_band(p, t_low: float, t_star: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.where(p < t_low, "approve", np.where(p < t_star, "review", "reject"))


def band_summary(y_true, p, t_low: float, t_star: float) -> pd.DataFrame:
    """Volume and observed default rate per band."""
    y = np.asarray(y_true)
    bands = assign_band(p, t_low, t_star)
    rows = []
    for band in BANDS:
        mask = bands == band
        rows.append(
            {
                "rows": int(mask.sum()),
                "share": float(mask.mean()),
                "default_rate": float(y[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(BANDS))


def cost_sensitivity(y_true, p, ratios, fp_cost, capacity) -> pd.DataFrame:
    """Threshold, approval and cost for each FN:FP cost ratio."""
    y = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    rows = []
    for ratio in ratios:
        t_star, cost = optimal_threshold(y, p, ratio * fp_cost, fp_cost)
        approved = p < t_star
        rows.append(
            {
                "fn_fp_ratio": ratio,
                "t_star": t_star,
                "t_low": review_threshold(p, t_star, capacity),
                "approval_rate": float(approved.mean()),
                "default_rate_approved": float(y[approved].mean()) if approved.any() else np.nan,
                "expected_cost": cost,
            }
        )
    return pd.DataFrame(rows)


def capacity_sensitivity(y_true, p, t_star: float, capacities) -> pd.DataFrame:
    """Band volumes and default rates for each review capacity."""
    rows = []
    for capacity in capacities:
        t_low = review_threshold(p, t_star, capacity)
        summary = band_summary(y_true, p, t_low, t_star)
        row = {"capacity": capacity, "t_low": t_low}
        for band in BANDS:
            row[f"{band}_rows"] = int(summary.loc[band, "rows"])
            row[f"{band}_default_rate"] = float(summary.loc[band, "default_rate"])
        rows.append(row)
    return pd.DataFrame(rows)
