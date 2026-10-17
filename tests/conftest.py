import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from credit_risk.data.schema import COLUMNS, FLOAT_COLUMNS, ID_NAME  # noqa: E402

NAN = np.nan


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Synthetic stand-in for cs-training.csv with known duplicates and domain issues."""
    rows = [
        # columns follow COLUMNS: target first, then FEATURES in CSV order
        [0, 0.5, 40, 0, 0.3, 5000, 5, 0, 1, 0, 2],
        [0, 0.5, 40, 0, 0.3, 5000, 5, 0, 1, 0, 2],
        [1, 0.9, 30, 1, 0.8, 3000, 3, 0, 0, 0, 1],
        [0, 0.9, 30, 1, 0.8, 3000, 3, 0, 0, 0, 1],
        [1, 1.2, 15, 98, 3.5, 0, 2, 98, 0, 98, 0],
        [0, 0.1, 60, 0, 0.2, NAN, 8, 0, 2, 0, NAN],
        [0, 0.1, 60, 0, 0.2, NAN, 8, 0, 2, 0, NAN],
        [1, 0.1, 60, 0, 0.2, NAN, 8, 0, 2, 0, NAN],
    ]
    df = pd.DataFrame(rows, columns=COLUMNS, index=pd.RangeIndex(1, 9, name=ID_NAME))
    float_columns = [*FLOAT_COLUMNS, "NumberOfDependents"]
    return df.astype({c: "int64" for c in COLUMNS if c not in float_columns})
