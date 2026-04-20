from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    EfficientSU2,
    PauliFeatureMap,
    RealAmplitudes,
    ZZFeatureMap,
)


def _decompose(circuit: QuantumCircuit) -> QuantumCircuit:
    """Flatten a library circuit to basic gates.

    Qiskit 2.x + Aer V2 primitives do not recognise high-level library
    names like `ZZFeatureMap` as Aer instructions, so we decompose once
    here into standard single- and two-qubit gates before handing the
    circuit to any primitive.
    """
    return circuit.decompose(reps=2)


def build_feature_map(
    n_qubits: int, kind: str = "zz", reps: int = 2, entanglement: str = "linear"
) -> QuantumCircuit:
    if kind == "zz":
        fm = ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
    elif kind == "pauli":
        fm = PauliFeatureMap(
            feature_dimension=n_qubits, reps=reps, entanglement=entanglement
        )
    else:
        raise ValueError(f"unknown feature map kind: {kind}")
    return _decompose(fm)


def build_ansatz(
    n_qubits: int,
    kind: str = "real_amplitudes",
    reps: int = 3,
    entanglement: str = "linear",
) -> QuantumCircuit:
    if kind == "real_amplitudes":
        a = RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
    elif kind == "efficient_su2":
        a = EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
    else:
        raise ValueError(f"unknown ansatz kind: {kind}")
    return _decompose(a)
