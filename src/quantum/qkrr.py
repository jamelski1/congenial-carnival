from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge

from .backends import BackendHandles
from .circuits import build_feature_map


@dataclass
class QKRRConfig:
    n_qubits: int
    alpha: float = 1.0
    feature_map: dict = field(default_factory=lambda: {"kind": "zz", "reps": 2, "entanglement": "linear"})


class QuantumKernelRidge(BaseEstimator, RegressorMixin):
    """Kernel ridge regression with a quantum fidelity kernel.

    Uses qiskit-machine-learning's FidelityQuantumKernel against the configured
    feature map. The kernel matrices are precomputed and handed to sklearn's
    KernelRidge, which keeps the classical bit drop-in-swappable.
    """

    def __init__(self, config: QKRRConfig, handles: BackendHandles):
        self.config = config
        self.handles = handles
        self._kernel = None
        self._model: KernelRidge | None = None
        self._X_train: np.ndarray | None = None

    def _build_kernel(self):
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute

        fmap = build_feature_map(self.config.n_qubits, **self.config.feature_map)
        fidelity = ComputeUncompute(sampler=self.handles.sampler)
        return FidelityQuantumKernel(feature_map=fmap, fidelity=fidelity)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelRidge":
        X = np.asarray(X, dtype=float)
        self._kernel = self._build_kernel()
        gram = self._kernel.evaluate(x_vec=X)
        self._model = KernelRidge(alpha=self.config.alpha, kernel="precomputed")
        self._model.fit(gram, np.asarray(y, dtype=float))
        self._X_train = X
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._kernel is None or self._X_train is None:
            raise RuntimeError("QuantumKernelRidge.fit must be called before predict")
        X = np.asarray(X, dtype=float)
        gram = self._kernel.evaluate(x_vec=X, y_vec=self._X_train)
        return np.asarray(self._model.predict(gram)).ravel()
