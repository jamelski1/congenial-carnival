from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ReducerConfig:
    n_qubits: int
    scale_features: bool = True
    log_transform_target: bool = True


def build_feature_pipeline(cfg: ReducerConfig) -> Pipeline:
    steps: list = [("pca", PCA(n_components=cfg.n_qubits, random_state=0))]
    if cfg.scale_features:
        # Angle-encoded circuits expect [0, pi]; MinMax to that range.
        steps.append(("scale", MinMaxScaler(feature_range=(0.0, float(np.pi)))))
    return Pipeline(steps)


def transform_target(y: np.ndarray, cfg: ReducerConfig) -> np.ndarray:
    return np.log1p(y) if cfg.log_transform_target else y.astype(float)


def inverse_transform_target(y: np.ndarray, cfg: ReducerConfig) -> np.ndarray:
    return np.expm1(y) if cfg.log_transform_target else y
