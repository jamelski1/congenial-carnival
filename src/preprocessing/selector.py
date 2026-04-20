from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class XGBTopKSelector(BaseEstimator, TransformerMixin):
    """Pick the top-K most important features via a cheap XGBoost pre-fit.

    Used to compress the raw 774-d btoe feature vector down to the qubit
    budget without going through PCA. Where PCA picks high-*variance*
    directions (often dominated by raw scale, e.g. text_length), this
    picks the K features that a quick gradient-boosted regressor finds
    most predictive of `duration_hours` — so the quantum kernel encodes
    informative inputs rather than noisy ones.

    Both the classical baseline and the quantum models then train on the
    same K features, so the comparison stays apples-to-apples.
    """

    def __init__(self, k: int, n_estimators: int = 200, max_depth: int = 4, random_seed: int = 42):
        self.k = k
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_seed = random_seed
        self._idx: np.ndarray | None = None
        self._importances: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBTopKSelector":
        from xgboost import XGBRegressor

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y is None:
            raise ValueError("XGBTopKSelector requires y at fit time")
        if self.k > X.shape[1]:
            raise ValueError(f"k={self.k} exceeds feature count {X.shape[1]}")

        model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_seed,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
        )
        model.fit(X, y)
        importances = np.asarray(model.feature_importances_, dtype=float)

        # Stable ordering: highest importance first; ties broken by index ascending
        order = np.lexsort((np.arange(len(importances)), -importances))
        self._idx = np.sort(order[: self.k])
        self._importances = importances
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._idx is None:
            raise RuntimeError("XGBTopKSelector.fit must be called before transform")
        return np.asarray(X, dtype=float)[:, self._idx]
