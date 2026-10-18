import pytest
from sklearn.metrics import roc_auc_score

from credit_risk.evaluation.metrics import (
    classification_metrics,
    expected_cost,
    gini,
    ks_statistic,
)

Y = [0, 0, 1, 1]
P_PERFECT = [0.1, 0.2, 0.8, 0.9]


def test_perfect_separation():
    assert roc_auc_score(Y, P_PERFECT) == 1
    assert gini(Y, P_PERFECT) == 1
    assert ks_statistic(Y, P_PERFECT) == 1


def test_constant_prediction_has_no_discrimination():
    p = [0.5] * 4
    assert gini(Y, p) == 0
    assert ks_statistic(Y, p) == 0


def test_expected_cost_counts_both_error_types():
    y = [1, 0, 0, 1]
    p = [0.9, 0.6, 0.1, 0.2]
    assert expected_cost(y, p, threshold=0.5, fn_cost=5, fp_cost=1) == pytest.approx(1.5)


def test_expected_cost_predicts_positive_at_threshold():
    assert expected_cost([0], [0.5], threshold=0.5, fn_cost=5, fp_cost=1) == 1.0


def test_classification_metrics_keys_and_brier():
    metrics = classification_metrics([0, 1], [0.0, 1.0])
    assert set(metrics) == {"roc_auc", "gini", "ks", "pr_auc", "brier"}
    assert metrics["brier"] == 0
    assert classification_metrics(Y, P_PERFECT)["roc_auc"] == 1
