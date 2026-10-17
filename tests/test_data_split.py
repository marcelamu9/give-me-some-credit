import numpy as np
import pandas as pd
import pytest

from credit_risk.data.schema import COLUMNS, ID_NAME, TARGET
from credit_risk.data.split import (
    PARTITIONS,
    cross_partition_feature_overlap,
    load_partition,
    partition_summary,
    save_partitions,
    split_train_valid_test,
)


@pytest.fixture
def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame(rng.random((n, len(COLUMNS))), columns=COLUMNS)
    df[TARGET] = 0
    df.loc[rng.choice(n, size=100, replace=False), TARGET] = 1
    df["age"] = rng.integers(18, 90, size=n)
    df.loc[rng.choice(n, size=50, replace=False), "MonthlyIncome"] = np.nan
    df.index = pd.RangeIndex(1, n + 1, name=ID_NAME)
    return df


def split(df, seed=42):
    return split_train_valid_test(df, TARGET, test_size=0.2, valid_size=0.2, seed=seed)


def test_partition_sizes(synthetic):
    parts = split(synthetic)
    assert (len(parts["test"]), len(parts["valid"]), len(parts["train"])) == (200, 160, 640)


def test_partitions_are_disjoint_and_complete(synthetic):
    parts = split(synthetic)
    train, valid, test = (set(parts[name].index) for name in PARTITIONS)
    assert not (train & valid) and not (train & test) and not (valid & test)
    assert train | valid | test == set(synthetic.index)


def test_stratification_keeps_default_rate(synthetic):
    for part in split(synthetic).values():
        assert abs(part[TARGET].mean() - 0.10) < 0.01


def test_split_is_reproducible(synthetic):
    first, second, other = split(synthetic), split(synthetic), split(synthetic, seed=1)
    for name in PARTITIONS:
        assert first[name].index.equals(second[name].index)
    assert not first["train"].index.equals(other["train"].index)


def test_partition_summary_columns(synthetic):
    summary = partition_summary(split(synthetic), TARGET)
    assert summary.index.tolist() == ["train", "valid", "test"]
    assert summary.columns.tolist() == ["rows", "default_rate"]


def test_cross_partition_feature_overlap():
    def frame(ids, value):
        return pd.DataFrame(
            {"a": [value] * len(ids), "b": [np.nan] * len(ids)},
            index=pd.Index(ids, name=ID_NAME),
        )

    shared = {"train": frame([1], 1.0), "valid": frame([2], 2.0), "test": frame([3], 1.0)}
    assert cross_partition_feature_overlap(shared, ["a", "b"]) == 1
    distinct = {"train": frame([1], 1.0), "valid": frame([2], 2.0), "test": frame([3], 3.0)}
    assert cross_partition_feature_overlap(distinct, ["a", "b"]) == 0


def test_save_and_load_round_trip(tmp_path, synthetic):
    parts = split(synthetic)
    paths = save_partitions(parts, tmp_path / "interim")
    assert all(path.is_file() for path in paths.values())
    for name in PARTITIONS:
        pd.testing.assert_frame_equal(load_partition(name, tmp_path / "interim"), parts[name])


def test_load_partition_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="holdout"):
        load_partition("holdout", tmp_path)
