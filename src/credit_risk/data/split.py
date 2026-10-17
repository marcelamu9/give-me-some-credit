"""Stratified train/valid/test split and persistence of the partitions."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.data.schema import ID_NAME

PARTITIONS = ("train", "valid", "test")


def split_train_valid_test(
    df: pd.DataFrame, target: str, test_size: float, valid_size: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Two stratified steps: test_size of the total to test, valid_size of the rest to valid."""
    rest, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
    train, valid = train_test_split(
        rest, test_size=valid_size, stratify=rest[target], random_state=seed
    )
    return {"train": train, "valid": valid, "test": test}


def partition_summary(parts: dict[str, pd.DataFrame], target: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rows": {name: len(parts[name]) for name in PARTITIONS},
            "default_rate": {name: float(parts[name][target].mean()) for name in PARTITIONS},
        }
    )


def cross_partition_feature_overlap(parts: dict[str, pd.DataFrame], features: list[str]) -> int:
    """Distinct feature combinations (NaN equal) that appear in more than one partition."""
    combos = [parts[name][features].drop_duplicates().assign(_partition=name) for name in parts]
    stacked = pd.concat(combos, ignore_index=True)
    partitions_per_combo = stacked.groupby(features, dropna=False)["_partition"].nunique()
    return int((partitions_per_combo > 1).sum())


def save_partitions(parts: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in PARTITIONS:
        paths[name] = out_dir / f"{name}.csv"
        parts[name].to_csv(paths[name])
    return paths


def load_partition(name: str, directory: Path) -> pd.DataFrame:
    if name not in PARTITIONS:
        raise ValueError(f"Unknown partition {name!r}; expected one of {PARTITIONS}")
    return pd.read_csv(directory / f"{name}.csv", index_col=ID_NAME, float_precision="round_trip")
