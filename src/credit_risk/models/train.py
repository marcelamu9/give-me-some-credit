"""Reproducible training: search, calibrate, select, fix thresholds, evaluate once on test."""

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from importlib.metadata import version

import joblib
import mlflow
import numpy as np
import pandas as pd

from credit_risk.config import PROJECT_ROOT, Config, load_config
from credit_risk.data.schema import FEATURES
from credit_risk.data.split import PARTITIONS, load_partition
from credit_risk.decision.policy import (
    band_summary,
    capacity_sensitivity,
    cost_sensitivity,
    optimal_threshold,
    review_threshold,
)
from credit_risk.evaluation.metrics import classification_metrics, expected_cost
from credit_risk.features.treatment import MODEL_FEATURES, drop_underage
from credit_risk.models.candidates import (
    calibrate,
    candidate_specs,
    prediction_latency_ms,
    search_candidate,
)
from credit_risk.models.report import render_report
from credit_risk.models.selection import select_champion

# Recent MLflow refuses the local file store unless explicitly allowed; the spec keeps runs in
# a local, git-ignored mlruns/ directory.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

EXPERIMENT = "credit-risk-modeling"
LATENCY_ROWS = 200
LATENCY_REPEATS = 3
LIBRARIES = ("scikit-learn", "lightgbm", "pandas", "numpy", "mlflow", "shap")
SELECTION_RULE = (
    "menor costo esperado por cliente en validación; son elegibles los modelos a menos de 2 % de "
    "ese costo cuyo Brier no sea más de 10 % peor que el del modelo de menor costo; entre ellos "
    "gana la regresión logística (interpretable) y, si no es elegible, el de menor latencia de "
    "predicción (más barato de operar)."
)


def _jsonable(value):
    """Recursively convert numpy / pandas values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _bands(frame: pd.DataFrame) -> dict:
    return _jsonable(frame.to_dict(orient="index"))


def _load(cfg: Config, name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = drop_underage(load_partition(name, cfg.paths.interim))
    return df[FEATURES], df[cfg.target]


def run_training(cfg: Config, tracking_uri: str | None = None) -> dict:
    """Train, select and evaluate; write the artifacts and return the metadata."""
    started = time.perf_counter()
    fn_cost, fp_cost = cfg.costs.false_negative, cfg.costs.false_positive
    mlflow.set_tracking_uri(tracking_uri or (PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)

    X_train, y_train = _load(cfg, "train")
    X_valid, y_valid = _load(cfg, "valid")

    fitted, probabilities, candidates = {}, {}, {}
    for name, (pipeline, distributions) in candidate_specs(cfg.seed).items():
        with mlflow.start_run(run_name=name):
            result = search_candidate(
                name,
                pipeline,
                distributions,
                X_train,
                y_train,
                n_iter=cfg.search.n_iter,
                cv_folds=cfg.search.cv_folds,
                seed=cfg.seed,
            )
            model, calibration_seconds = calibrate(
                result.estimator, X_train, y_train, cfg.search.cv_folds
            )
            p_valid = model.predict_proba(X_valid)[:, 1]
            p_raw = result.estimator.predict_proba(X_valid)[:, 1]
            metrics = classification_metrics(y_valid, p_valid)
            t_star, cost = optimal_threshold(y_valid, p_valid, fn_cost, fp_cost)
            latency = prediction_latency_ms(model, X_valid, LATENCY_ROWS, LATENCY_REPEATS)

            candidates[name] = {
                "best_params": result.best_params,
                "cv_auc_mean": result.cv_auc_mean,
                "cv_auc_std": result.cv_auc_std,
                "search_seconds": result.search_seconds,
                "calibration_seconds": calibration_seconds,
                "latency_ms": latency,
                "valid_metrics": metrics,
                "valid_metrics_uncalibrated": classification_metrics(y_valid, p_raw),
                "valid_expected_cost": cost,
                "valid_t_star": t_star,
            }
            mlflow.log_params({k: str(v) for k, v in result.best_params.items()})
            mlflow.log_metrics(
                {
                    "cv_auc_mean": result.cv_auc_mean,
                    "cv_auc_std": result.cv_auc_std,
                    **{f"valid_{k}": v for k, v in metrics.items()},
                    "valid_expected_cost": cost,
                    "valid_t_star": t_star,
                    "search_seconds": result.search_seconds,
                    "calibration_seconds": calibration_seconds,
                    "latency_ms": latency,
                }
            )
            fitted[name], probabilities[name] = model, p_valid

    comparison = pd.DataFrame(
        {
            "model": list(candidates),
            "expected_cost": [c["valid_expected_cost"] for c in candidates.values()],
            "brier": [c["valid_metrics"]["brier"] for c in candidates.values()],
            "latency_ms": [c["latency_ms"] for c in candidates.values()],
        }
    )
    champion = select_champion(comparison)

    p_valid = probabilities[champion]
    t_star = candidates[champion]["valid_t_star"]
    t_low = review_threshold(p_valid, t_star, cfg.decision.review_capacity)
    valid_bands = band_summary(y_valid, p_valid, t_low, t_star)
    cost_table = cost_sensitivity(
        y_valid,
        p_valid,
        cfg.decision.cost_sensitivity_ratios,
        fp_cost,
        cfg.decision.review_capacity,
    )
    capacity_table = capacity_sensitivity(
        y_valid, p_valid, t_star, cfg.decision.review_capacity_sensitivity
    )

    # The test partition is read and evaluated exactly once, with the final model and thresholds.
    X_test, y_test = _load(cfg, "test")
    p_test = fitted[champion].predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, p_test)
    test_cost = expected_cost(y_test, p_test, t_star, fn_cost, fp_cost)
    test_bands = band_summary(y_test, p_test, t_low, t_star)

    metadata = _jsonable(
        {
            "champion": champion,
            "selection_rule": SELECTION_RULE,
            "features": FEATURES,
            "model_features": MODEL_FEATURES,
            "thresholds": {
                "t_star": t_star,
                "t_low": t_low,
                "bayes_threshold": fp_cost / (fp_cost + fn_cost),
            },
            "costs": {"false_negative": fn_cost, "false_positive": fp_cost},
            "review_capacity": cfg.decision.review_capacity,
            "candidates": candidates,
            "valid_bands": _bands(valid_bands),
            "cost_sensitivity": cost_table,
            "capacity_sensitivity": capacity_table,
            "test_metrics": test_metrics,
            "test_expected_cost": test_cost,
            "test_bands": _bands(test_bands),
            "rows": {"train": len(X_train), "valid": len(X_valid), "test": len(X_test)},
            "data_sha256": {
                name: hashlib.sha256((cfg.paths.interim / f"{name}.csv").read_bytes()).hexdigest()
                for name in PARTITIONS
            },
            "seed": cfg.seed,
            "latency_rows": min(LATENCY_ROWS, len(X_valid)),
            "library_versions": {lib: version(lib) for lib in LIBRARIES},
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "training_seconds": time.perf_counter() - started,
        }
    )

    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": fitted[champion], "features": FEATURES, "t_low": t_low, "t_star": t_star},
        cfg.paths.models / "champion.joblib",
    )
    joblib.dump(fitted, cfg.paths.models / "candidates.joblib")
    (cfg.paths.models / "champion_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (cfg.paths.reports / "model_comparison.md").write_text(
        render_report(metadata), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    run_training(load_config())
