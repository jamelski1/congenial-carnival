from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .backends import BackendHandles
from .circuits import build_ansatz, build_feature_map


@dataclass
class VQRConfig:
    n_qubits: int
    feature_map: dict = field(default_factory=lambda: {"kind": "zz", "reps": 2, "entanglement": "linear"})
    ansatz: dict = field(default_factory=lambda: {"kind": "real_amplitudes", "reps": 3, "entanglement": "linear"})
    optimizer: str = "cobyla"
    maxiter: int = 150
    initial_point: np.ndarray | None = None
    random_seed: int = 42


class VQRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible Variational Quantum Regressor.

    Wraps qiskit-machine-learning's EstimatorQNN + NeuralNetworkRegressor so it
    participates in the same train/predict flow as the classical baseline.
    """

    def __init__(self, config: VQRConfig, handles: BackendHandles):
        self.config = config
        self.handles = handles
        self._model: Any = None

    def _build_optimizer(self):
        from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, SPSA

        name = self.config.optimizer.lower()
        if name == "cobyla":
            return COBYLA(maxiter=self.config.maxiter)
        if name == "spsa":
            return SPSA(maxiter=self.config.maxiter)
        if name == "l_bfgs_b":
            return L_BFGS_B(maxiter=self.config.maxiter)
        raise ValueError(f"unknown optimizer: {self.config.optimizer}")

    def _build_model(self):
        from qiskit_machine_learning.algorithms.regressors import VQR
        from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient

        n = self.config.n_qubits
        fmap = build_feature_map(n, **self.config.feature_map)
        ansatz = build_ansatz(n, **self.config.ansatz)
        optimizer = self._build_optimizer()
        gradient = ParamShiftEstimatorGradient(self.handles.estimator)

        initial_point = self.config.initial_point
        if initial_point is None:
            rng = np.random.default_rng(self.config.random_seed)
            initial_point = rng.uniform(-np.pi, np.pi, size=ansatz.num_parameters)

        return VQR(
            feature_map=fmap,
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=self.handles.estimator,
            gradient=gradient,
            initial_point=np.asarray(initial_point, dtype=float),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQRegressor":
        self._model = self._build_model()
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("VQRegressor.fit must be called before predict")
        return np.asarray(self._model.predict(np.asarray(X, dtype=float))).ravel()
