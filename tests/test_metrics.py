import numpy as np

from src.evaluation.metrics import evaluate, mae, mdae, mmre, pred_at, r2, sa


def test_perfect_prediction_scores():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = evaluate(y, y, y)
    assert out["mae"] == 0.0
    assert out["mdae"] == 0.0
    assert out["mmre"] == 0.0
    assert out["pred_25"] == 1.0
    assert out["pred_50"] == 1.0
    assert out["r2"] == 1.0
    assert out["sa"] == 1.0


def test_pred_at_level_threshold():
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([10.0, 12.0, 13.0, 20.0])  # MRE = 0, .2, .3, 1.0
    assert pred_at(y_true, y_pred, 0.25) == 0.5  # first two within 25%
    assert pred_at(y_true, y_pred, 0.50) == 0.75


def test_sa_against_mean_baseline():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([2.0, 4.0, 6.0])
    # Perfect predictor beats the train-mean baseline → SA approaches 1.
    assert sa(y_true, y_true, y_train) == 1.0
    # Mean-baseline predictor → SA == 0 by construction.
    mean_pred = np.full_like(y_true, y_train.mean())
    assert sa(y_true, mean_pred, y_train) == 0.0


def test_individual_metric_signatures():
    y_true = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([1.5, 2.0, 3.0])
    assert mae(y_true, y_pred) > 0.0
    assert mdae(y_true, y_pred) >= 0.0
    assert mmre(y_true, y_pred) > 0.0
    assert r2(y_true, y_pred) <= 1.0
