"""Loading of the raw Kaggle training file."""

from pathlib import Path

import pandas as pd

from credit_risk.data.schema import ID_NAME


def load_raw_training(path: Path) -> pd.DataFrame:
    """Read cs-training.csv, using its unnamed first column as the index named ``id``."""
    df = pd.read_csv(path, index_col=0)
    df.index.name = ID_NAME
    return df
