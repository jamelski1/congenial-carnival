"""Tests for QKRRConfig + bandwidth grid handling.

We do not exercise the quantum backend here (qiskit_aer is heavy and the
real kernel is checked end-to-end via examples/synthetic_demo.py). These
tests pin down the config surface and the bandwidth-grid logic that
drives CV selection.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.quantum.qkrr import QKRRConfig, QuantumKernelRidge


def _make(bandwidth, cv_subsample=None):
    cfg = QKRRConfig(
        n_qubits=4,
        alpha=1.0,
        bandwidth=bandwidth,
        cv_folds=3,
        cv_subsample=cv_subsample,
    )
    # handles is unused by _bandwidth_grid; pass a stub
    return QuantumKernelRidge(cfg, handles=None)  # type: ignore[arg-type]


def test_scalar_bandwidth_yields_single_element_grid():
    qk = _make(0.3)
    assert qk._bandwidth_grid() == [0.3]


def test_list_bandwidth_yields_grid():
    qk = _make([1.0, 0.1, 0.01])
    assert qk._bandwidth_grid() == [1.0, 0.1, 0.01]


def test_empty_bandwidth_grid_raises():
    qk = _make([])
    with pytest.raises(ValueError, match="bandwidth grid is empty"):
        qk._bandwidth_grid()


def test_predict_before_fit_raises():
    qk = _make(1.0)
    with pytest.raises(RuntimeError, match="fit must be called"):
        qk.predict(np.zeros((1, 4)))


def test_config_default_bandwidth_is_one():
    cfg = QKRRConfig(n_qubits=4)
    assert cfg.bandwidth == 1.0
    assert cfg.cv_folds == 5
