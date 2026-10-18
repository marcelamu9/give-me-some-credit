import dataclasses
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from credit_risk.config import Search, load_config
from credit_risk.data.schema import COLUMNS, FEATURES, ID_NAME
from credit_risk.models import train as train_module
from credit_risk.models.candidates import COMPLEXITY_ORDER
from credit_risk.models.report import markdown_table
from credit_risk.models.train import run_training

N_ROWS = {"train": 600, "valid": 200, "test": 200}


def make_partition(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    revolving = rng.uniform(0, 1.2, n)
    late = rng.poisson(0.4, n)
    age = rng.integers(20, 80, n)
    income = rng.lognormal(8.5, 0.5, n)
    income[rng.random(n) < 0.15] = np.nan
    logit = -2.5 + 3.0 * revolving + 0.9 * late - 0.02 * (age - 50)
    df = pd.DataFrame(
        {
            "SeriousDlqin2yrs": rng.binomial(1, 1 / (1 + np.exp(-logit))),
            "RevolvingUtilizationOfUnsecuredLines": revolving,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late,
            "DebtRatio": rng.uniform(0, 1.5, n),
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, n),
            "NumberOfTimes90DaysLate": rng.poisson(0.1, n),
            "NumberRealEstateLoansOrLines": rng.poisson(1, n),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.1, n),
            "NumberOfDependents": rng.integers(0, 4, n).astype(float),
        }
    )[COLUMNS]
    df.index = pd.RangeIndex(1, n + 1, name=ID_NAME)
    return df


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    # Row-by-row latency over 200 rows dominates the runtime; a handful is enough here.
    monkeypatch.setattr(train_module, "LATENCY_ROWS", 5)
    monkeypatch.setattr(train_module, "LATENCY_REPEATS", 1)
    interim = tmp_path / "interim"
    interim.mkdir()
    for seed, (name, n) in enumerate(N_ROWS.items()):
        df = make_partition(n, seed)
        if name == "train":
            df.loc[df.index[0], "age"] = 10
        df.to_csv(interim / f"{name}.csv")
    base = load_config()
    paths = dataclasses.replace(
        base.paths,
        interim=interim,
        models=tmp_path / "models",
        reports=tmp_path / "reports",
    )
    return dataclasses.replace(base, paths=paths, search=Search(n_iter=2, cv_folds=2))


def test_run_training_end_to_end(cfg, tmp_path):
    metadata = run_training(cfg, tracking_uri=(tmp_path / "mlruns").as_uri())

    assert (cfg.paths.models / "champion.joblib").is_file()
    assert (cfg.paths.models / "candidates.joblib").is_file()
    assert (cfg.paths.models / "champion_metadata.json").is_file()
    assert (cfg.paths.reports / "model_comparison.md").is_file()

    assert metadata["champion"] in COMPLEXITY_ORDER
    thresholds = metadata["thresholds"]
    assert 0 <= thresholds["t_low"] <= thresholds["t_star"]
    assert thresholds["bayes_threshold"] == pytest.approx(1 / 6)
    assert set(metadata["candidates"]) == set(COMPLEXITY_ORDER)
    assert set(metadata["test_metrics"]) == {"roc_auc", "gini", "ks", "pr_auc", "brier"}
    assert metadata["rows"]["train"] == N_ROWS["train"] - 1
    assert set(metadata["data_sha256"]) == {"train", "valid", "test"}

    on_disk = json.loads((cfg.paths.models / "champion_metadata.json").read_text(encoding="utf-8"))
    assert on_disk["champion"] == metadata["champion"]

    artifact = joblib.load(cfg.paths.models / "champion.joblib")
    assert artifact["features"] == FEATURES
    assert artifact["t_star"] == thresholds["t_star"]
    proba = artifact["model"].predict_proba(make_partition(3, 9)[FEATURES])
    assert proba.shape == (3, 2)

    assert (tmp_path / "mlruns").is_dir()


def test_test_partition_loaded_once(cfg, tmp_path, monkeypatch):
    calls = []
    original = train_module.load_partition

    def counting(name, directory):
        calls.append(name)
        return original(name, directory)

    monkeypatch.setattr(train_module, "load_partition", counting)
    run_training(cfg, tracking_uri=(tmp_path / "mlruns").as_uri())
    assert calls.count("test") == 1


def test_markdown_table():
    table = markdown_table(pd.DataFrame({"a": [1, 2], "b": [0.12345, 0.5]}))
    lines = table.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1 | 0.1235 |"
    assert len(lines) == 4
