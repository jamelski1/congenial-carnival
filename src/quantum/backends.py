from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class BackendHandles:
    """Runtime objects passed to Qiskit ML primitives.

    `estimator` and `sampler` are V2 primitive instances configured for the
    requested backend. `name` is informational.
    """

    estimator: Any
    sampler: Any
    name: str


def get_backend(backend: str = "aer", shots: int = 1024) -> BackendHandles:
    backend = backend.lower()
    if backend == "aer":
        return _aer(shots)
    if backend == "ibm":
        return _ibm(shots)
    raise ValueError(f"unknown backend: {backend}")


def _aer(shots: int) -> BackendHandles:
    from qiskit_aer.primitives import EstimatorV2, SamplerV2

    estimator = EstimatorV2()
    sampler = SamplerV2()
    estimator.options.default_shots = shots
    sampler.options.default_shots = shots
    return BackendHandles(estimator=estimator, sampler=sampler, name="aer-simulator")


def _ibm(shots: int) -> BackendHandles:
    from qiskit_ibm_runtime import (
        EstimatorV2,
        QiskitRuntimeService,
        SamplerV2,
    )

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError(
            "IBM_QUANTUM_TOKEN not set. Copy .env.example to .env and paste your "
            "IBM Quantum Platform token, or switch to --backend aer."
        )

    instance = os.environ.get("IBM_QUANTUM_INSTANCE") or None
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)

    requested = os.environ.get("IBM_QUANTUM_BACKEND")
    if requested:
        backend_obj = service.backend(requested)
    else:
        backend_obj = service.least_busy(operational=True, simulator=False)

    estimator = EstimatorV2(mode=backend_obj)
    sampler = SamplerV2(mode=backend_obj)
    estimator.options.default_shots = shots
    sampler.options.default_shots = shots
    return BackendHandles(estimator=estimator, sampler=sampler, name=backend_obj.name)
