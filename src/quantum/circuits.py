from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    EfficientSU2,
    PauliFeatureMap,
    RealAmplitudes,
    ZZFeatureMap,
)


def build_feature_map(
    n_qubits: int, kind: str = "zz", reps: int = 2, entanglement: str = "linear"
) -> QuantumCircuit:
    if kind == "zz":
        return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
    if kind == "pauli":
        return PauliFeatureMap(
            feature_dimension=n_qubits, reps=reps, entanglement=entanglement
        )
    raise ValueError(f"unknown feature map kind: {kind}")


def build_ansatz(
    n_qubits: int,
    kind: str = "real_amplitudes",
    reps: int = 3,
    entanglement: str = "linear",
) -> QuantumCircuit:
    if kind == "real_amplitudes":
        return RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
    if kind == "efficient_su2":
        return EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
    raise ValueError(f"unknown ansatz kind: {kind}")
