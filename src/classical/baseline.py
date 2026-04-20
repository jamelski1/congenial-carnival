from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor


@dataclass
class XGBConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_seed: int = 42


class XGBBaseline(BaseEstimator, RegressorMixin):
    """XGBoost baseline on the *same* reduced features the quantum models see.

    This isolates the "quantum vs. classical" axis from the "50-d vs. 8-d"
    axis — the original btoe XGBoost saw the full 50 PCA components, so
    this is a deliberately handicapped baseline. Its job is fairness,
    not winning.
    """

    def __init__(self, config: XGBConfig):
        self.config = config
        self._model: XGBRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBBaseline":
        c = self.config
        self._model = XGBRegressor(
            n_estimators=c.n_estimators,
            max_depth=c.max_depth,
            learning_rate=c.learning_rate,
            subsample=c.subsample,
            colsample_bytree=c.colsample_bytree,
            reg_alpha=c.reg_alpha,
            reg_lambda=c.reg_lambda,
            random_state=c.random_seed,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
        )
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBBaseline.fit must be called before predict")
        return np.asarray(self._model.predict(np.asarray(X, dtype=float))).ravel()
