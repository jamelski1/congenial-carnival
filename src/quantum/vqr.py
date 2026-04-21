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
    # Number of classical features fed into the feature map. When larger
    # than n_qubits the feature map chunks the input across blocks on the
    # same qubit register (data re-uploading). None => matches n_qubits.
    n_features: int | None = None
    # Bandwidth γ: inputs are scaled by γ before angle-encoding. Keeps the
    # encoding rotations small so gradients don't vanish (the variational
    # analogue of kernel concentration).
    bandwidth: float = 1.0
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
        # VQR's default observable is <Z>, which outputs values in [-1, 1].
        # We standardise y to zero mean / unit variance during fit so it
        # lives in a range the network can actually reach, and invert at
        # predict time.
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

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
        # qml 0.9's VQR uses V2 primitives internally and picks its own
        # gradient calculator — the earlier `gradient=` kwarg was removed.
        from qiskit_machine_learning.algorithms.regressors import VQR

        n = self.config.n_qubits
        fmap = build_feature_map(n, n_features=self.config.n_features, **self.config.feature_map)
        ansatz = build_ansatz(n, **self.config.ansatz)
        optimizer = self._build_optimizer()

        initial_point = self.config.initial_point
        if initial_point is None:
            rng = np.random.default_rng(self.config.random_seed)
            initial_point = rng.uniform(-np.pi, np.pi, size=ansatz.num_parameters)

        return VQR(
            feature_map=fmap,
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=self.handles.estimator,
            initial_point=np.asarray(initial_point, dtype=float),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQRegressor":
        self._model = self._build_model()
        Xs = np.asarray(X, dtype=float) * self.config.bandwidth
        y = np.asarray(y, dtype=float)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std())
        if self._y_std < 1e-9:
            self._y_std = 1.0
        y_scaled = (y - self._y_mean) / self._y_std
        self._model.fit(Xs, y_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("VQRegressor.fit must be called before predict")
        Xs = np.asarray(X, dtype=float) * self.config.bandwidth
        y_scaled = np.asarray(self._model.predict(Xs)).ravel()
        return y_scaled * self._y_std + self._y_mean
