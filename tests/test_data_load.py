from credit_risk.data.load import load_raw_training
from credit_risk.data.schema import COLUMNS


def test_load_raw_training_uses_unnamed_first_column_as_id(tmp_path, raw_frame):
    path = tmp_path / "cs-training.csv"
    raw_frame.to_csv(path, index_label="")

    df = load_raw_training(path)

    assert df.index.name == "id"
    assert df.shape == (8, 11)
    assert list(df.columns) == COLUMNS
