from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .selector import XGBTopKSelector


@dataclass
class ReducerConfig:
    n_qubits: int
    mode: str = "pca"               # "pca" | "topk"
    scale_features: bool = True
    log_transform_target: bool = True
    random_seed: int = 42


def build_feature_pipeline(cfg: ReducerConfig) -> Pipeline:
    """Build the feature-reduction pipeline.

    Two modes:
      - "pca": project to n_qubits principal components (variance-driven).
      - "topk": keep the n_qubits features whose XGBoost-importance is
        highest on the training set (signal-driven).

    Both end with a MinMax scale to [0, pi] so the resulting columns are
    valid angle inputs to the quantum feature map.
    """
    if cfg.mode == "pca":
        steps: list = [("pca", PCA(n_components=cfg.n_qubits, random_state=0))]
    elif cfg.mode == "topk":
        steps = [("topk", XGBTopKSelector(k=cfg.n_qubits, random_seed=cfg.random_seed))]
    else:
        raise ValueError(f"unknown reducer mode: {cfg.mode!r} (expected 'pca' or 'topk')")

    if cfg.scale_features:
        steps.append(("scale", MinMaxScaler(feature_range=(0.0, float(np.pi)))))
    return Pipeline(steps)


def transform_target(y: np.ndarray, cfg: ReducerConfig) -> np.ndarray:
    return np.log1p(y) if cfg.log_transform_target else y.astype(float)


def inverse_transform_target(y: np.ndarray, cfg: ReducerConfig) -> np.ndarray:
    return np.expm1(y) if cfg.log_transform_target else y
