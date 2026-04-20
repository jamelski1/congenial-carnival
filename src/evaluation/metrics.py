"""Effort-estimation metrics, matching btoe's trainer.

All functions expect numpy arrays of *real-scale* effort (hours), not
log-transformed. The pipeline inverse-transforms before scoring.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


_EPS = 1e-9


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mdae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_true - y_pred)))


def mmre(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean magnitude of relative error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < _EPS, _EPS, y_true)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(denom)))


def pred_at(y_true: np.ndarray, y_pred: np.ndarray, level: float) -> float:
    """Fraction of predictions within `level` (e.g. 0.25) relative error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < _EPS, _EPS, y_true)
    mre = np.abs(y_true - y_pred) / np.abs(denom)
    return float(np.mean(mre <= level))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def sa(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Standardised accuracy vs. a random-guess baseline (Shepperd & MacDonell).

    SA = 1 - MAE(model) / MAE(random_guess), where the random guess predicts
    the training-set mean. Positive means the model beats random guessing.
    """
    y_train = np.asarray(y_train, dtype=float)
    mean_pred = np.full_like(np.asarray(y_true, dtype=float), fill_value=float(np.mean(y_train)))
    denom = mean_absolute_error(y_true, mean_pred)
    if denom < _EPS:
        return 0.0
    return float(1.0 - mae(y_true, y_pred) / denom)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "mdae": mdae(y_true, y_pred),
        "mmre": mmre(y_true, y_pred),
        "pred_25": pred_at(y_true, y_pred, 0.25),
        "pred_50": pred_at(y_true, y_pred, 0.50),
        "r2": r2(y_true, y_pred),
        "sa": sa(y_true, y_pred, y_train),
    }
