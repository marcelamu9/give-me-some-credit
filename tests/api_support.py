"""Helper for the API tests: a small synthetic model with the real artifact's structure."""

import json
from pathlib import Path

import joblib
from test_models_train import make_partition

from credit_risk.data.schema import FEATURES, TARGET
from credit_risk.models.candidates import calibrate, candidate_specs

T_LOW = 0.05
T_STAR = 0.20
TRAINED_AT = "2026-01-01T00:00:00+00:00"


def build_model_dir(directory: Path, kind: str = "lightgbm", seed: int = 0) -> Path:
    """Train a tiny calibrated model on synthetic data and save artifact + metadata."""
    data = make_partition(600, seed)
    pipeline, _ = candidate_specs(seed)[kind]
    if kind != "logistic":
        pipeline.set_params(model__n_estimators=30)
    model, _ = calibrate(pipeline, data[FEATURES], data[TARGET], cv_folds=5)

    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES, "t_low": T_LOW, "t_star": T_STAR},
        directory / "champion.joblib",
    )
    metadata = {
        "trained_at": TRAINED_AT,
        "champion": kind,
        "thresholds": {"t_star": T_STAR, "t_low": T_LOW, "bayes_threshold": 1 / 6},
        "costs": {"false_negative": 5.0, "false_positive": 1.0},
        "review_capacity": 0.1,
        "test_metrics": {
            "roc_auc": 0.8,
            "pr_auc": 0.3,
            "ks": 0.5,
            "brier": 0.05,
            "log_loss": 0.2,
        },
        "features": FEATURES,
    }
    (directory / "champion_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory
