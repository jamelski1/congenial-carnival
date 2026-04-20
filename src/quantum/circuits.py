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


def _single_feature_map(
    n_qubits: int, kind: str, reps: int, entanglement: str, parameter_prefix: str = "x"
) -> QuantumCircuit:
    if kind == "zz":
        return ZZFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
        )
    if kind == "pauli":
        return PauliFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
        )
    raise ValueError(f"unknown feature map kind: {kind}")


def build_feature_map(
    n_qubits: int,
    kind: str = "zz",
    reps: int = 2,
    entanglement: str = "linear",
    n_features: int | None = None,
) -> QuantumCircuit:
    """Build a feature map that encodes `n_features` values into `n_qubits` qubits.

    When n_features <= n_qubits (or None) a single ZZ/Pauli feature map is
    returned. When n_features > n_qubits we chunk the features into
    ceil(n_features / n_qubits) blocks and apply a ZZ/Pauli feature map per
    block on the same qubit register, re-uploading classical data between
    chunks. This trades circuit depth for representational coverage without
    growing the qubit count — the workaround for going past the ~20-qubit
    wall of dense statevector simulation.

    The total number of parameters in the returned circuit equals n_features,
    ordered so that parameter i binds to feature i (chunks are concatenated
    in feature-index order).
    """
    n = int(n_qubits)
    k = int(n_features) if n_features is not None else n
    if k <= 0:
        raise ValueError("n_features must be positive")

    if k <= n:
        return _decompose(_single_feature_map(k, kind, reps, entanglement))

    circuit = QuantumCircuit(n)
    for block_idx, start in enumerate(range(0, k, n)):
        width = min(n, k - start)
        chunk = _single_feature_map(
            width, kind, reps, entanglement, parameter_prefix=f"x{block_idx}"
        )
        circuit.compose(chunk, qubits=range(width), inplace=True)
    return _decompose(circuit)


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
