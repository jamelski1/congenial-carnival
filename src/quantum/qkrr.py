from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

from .backends import BackendHandles
from .circuits import build_feature_map


@dataclass
class QKRRConfig:
    n_qubits: int
    alpha: float = 1.0
    # Bandwidth γ: features are scaled by γ before angle-encoding. Single
    # float = fixed bandwidth; sequence = grid-search the best γ via CV
    # on the training fold and keep that γ for the final fit.
    bandwidth: float | Sequence[float] = 1.0
    cv_folds: int = 5
    # Cap CV training folds to this many rows for tractability when the
    # full kernel is expensive (Gram is O(N^2 * 2^n_qubits)).
    cv_subsample: int | None = None
    cv_random_seed: int = 42
    feature_map: dict = field(
        default_factory=lambda: {"kind": "zz", "reps": 2, "entanglement": "linear"}
    )


class QuantumKernelRidge(BaseEstimator, RegressorMixin):
    """Kernel ridge regression with a quantum fidelity kernel.

    Two kernel paths depending on the backend:

    - On `aer` (local simulator) we use `FidelityStatevectorKernel`, which
      computes |<psi(x)|psi(y)>|^2 analytically from the statevector. No
      shots, no SWAP-test circuits — orders of magnitude faster and still
      mathematically identical to the noiseless limit of the shot-based
      kernel.
    - On `ibm` (real hardware) we fall back to `FidelityQuantumKernel` +
      `ComputeUncompute`, which runs SWAP-test-style circuits through the
      configured V2 sampler.

    Bandwidth γ is the standard fix for kernel concentration above ~12
    qubits (Thanasilp et al. 2022): inputs are scaled to γ·x before
    encoding so the rotation angles stay small and the fidelity kernel
    does not collapse to the identity matrix.
    """

    def __init__(self, config: QKRRConfig, handles: BackendHandles):
        self.config = config
        self.handles = handles
        self._kernel = None
        self._model: KernelRidge | None = None
        self._X_train: np.ndarray | None = None
        self._best_gamma: float | None = None
        self._cv_scores: dict[float, float] | None = None

    def _build_kernel(self):
        fmap = build_feature_map(self.config.n_qubits, **self.config.feature_map)

        if self.handles.name.startswith("aer"):
            from qiskit_machine_learning.kernels import FidelityStatevectorKernel
            return FidelityStatevectorKernel(feature_map=fmap)

        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
        fidelity = ComputeUncompute(sampler=self.handles.sampler)
        return FidelityQuantumKernel(feature_map=fmap, fidelity=fidelity)

    def _bandwidth_grid(self) -> list[float]:
        b = self.config.bandwidth
        if isinstance(b, (int, float)):
            return [float(b)]
        grid = [float(x) for x in b]
        if not grid:
            raise ValueError("bandwidth grid is empty")
        return grid

    def _cv_select_bandwidth(self, X: np.ndarray, y: np.ndarray, gammas: list[float]) -> float:
        if self.config.cv_subsample and self.config.cv_subsample < len(X):
            rng = np.random.default_rng(self.config.cv_random_seed)
            idx = rng.choice(len(X), size=self.config.cv_subsample, replace=False)
            X = X[idx]
            y = y[idx]

        kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.cv_random_seed)
        kernel = self._build_kernel()
        scores: dict[float, float] = {}

        for g in gammas:
            Xs = X * g
            fold_maes: list[float] = []
            for tr_idx, va_idx in kf.split(Xs):
                Xtr, Xva = Xs[tr_idx], Xs[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]
                gram_tr = kernel.evaluate(x_vec=Xtr)
                gram_va = kernel.evaluate(x_vec=Xva, y_vec=Xtr)
                m = KernelRidge(alpha=self.config.alpha, kernel="precomputed").fit(gram_tr, ytr)
                preds = np.asarray(m.predict(gram_va)).ravel()
                fold_maes.append(float(np.mean(np.abs(yva - preds))))
            scores[g] = float(np.mean(fold_maes))

        self._cv_scores = scores
        return min(scores, key=scores.get)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelRidge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        gammas = self._bandwidth_grid()

        self._best_gamma = (
            gammas[0] if len(gammas) == 1 else self._cv_select_bandwidth(X, y, gammas)
        )
        if len(gammas) > 1:
            print(f"[qkrr] CV selected bandwidth γ={self._best_gamma}; scores={self._cv_scores}")

        self._kernel = self._build_kernel()
        Xs = X * self._best_gamma
        gram = self._kernel.evaluate(x_vec=Xs)
        self._model = KernelRidge(alpha=self.config.alpha, kernel="precomputed").fit(gram, y)
        self._X_train = Xs
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._kernel is None or self._X_train is None or self._best_gamma is None:
            raise RuntimeError("QuantumKernelRidge.fit must be called before predict")
        Xs = np.asarray(X, dtype=float) * self._best_gamma
        gram = self._kernel.evaluate(x_vec=Xs, y_vec=self._X_train)
        return np.asarray(self._model.predict(gram)).ravel()
