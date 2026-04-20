"""Tests for the feature map builder — standard and chunked re-upload modes."""
from __future__ import annotations

import pytest

from src.quantum.circuits import build_ansatz, build_feature_map


def test_single_block_when_features_match_qubits():
    fm = build_feature_map(n_qubits=4, kind="zz", reps=1, n_features=4)
    assert fm.num_qubits == 4
    assert fm.num_parameters == 4


def test_single_block_when_features_below_qubits():
    # Single-block fallback uses the smaller width (3 qubits, 3 params).
    # Downstream kernels simulate on whatever width the circuit returns.
    fm = build_feature_map(n_qubits=6, kind="zz", reps=1, n_features=3)
    assert fm.num_qubits == 3
    assert fm.num_parameters == 3


def test_default_n_features_equals_n_qubits():
    fm = build_feature_map(n_qubits=5, kind="zz", reps=1)
    assert fm.num_parameters == 5


def test_chunked_reupload_parameter_count_equals_n_features():
    # 10 features on 4 qubits -> ceil(10/4) = 3 chunks, widths 4+4+2
    fm = build_feature_map(n_qubits=4, kind="zz", reps=1, n_features=10)
    assert fm.num_qubits == 4
    assert fm.num_parameters == 10


def test_chunked_with_exact_multiple():
    fm = build_feature_map(n_qubits=4, kind="zz", reps=1, n_features=12)
    assert fm.num_parameters == 12


def test_chunked_with_pauli_variant():
    # 6 features on 3 qubits -> 2 chunks of width 3 each (no 1-wide leftover
    # that PauliFeatureMap's linear entanglement can't handle).
    fm = build_feature_map(n_qubits=3, kind="pauli", reps=1, n_features=6)
    assert fm.num_parameters == 6


def test_invalid_n_features_raises():
    with pytest.raises(ValueError, match="must be positive"):
        build_feature_map(n_qubits=4, kind="zz", reps=1, n_features=0)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown feature map kind"):
        build_feature_map(n_qubits=4, kind="bogus", reps=1, n_features=4)


def test_ansatz_builder_unchanged():
    a = build_ansatz(n_qubits=4, kind="real_amplitudes", reps=2)
    assert a.num_qubits == 4
    assert a.num_parameters > 0
