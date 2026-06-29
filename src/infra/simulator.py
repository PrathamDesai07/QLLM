"""Uniform simulation wrapper over Qiskit backends (Qiskit ≥ 1.x)."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

from config import BACKEND


def _estimator(seed: int | None = None) -> StatevectorEstimator:
    return StatevectorEstimator(seed=seed or BACKEND.get("seed", 42))


def _sampler(seed: int | None = None) -> StatevectorSampler:
    return StatevectorSampler(seed=seed or BACKEND.get("seed", 42))


def _bind_parameters(
    circuit: QuantumCircuit, parameter_values: list[float] | None
) -> QuantumCircuit:
    """Return a bound copy of the circuit, or the original if unparameterized."""
    if parameter_values is not None:
        return circuit.assign_parameters(parameter_values)
    return circuit


def compute_expectation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    parameter_values: list[float] | None = None,
    shots: int | None = None,
) -> float:
    """Compute ⟨ψ|H|ψ⟩ using Qiskit's StatevectorEstimator."""
    est = _estimator()
    bound = _bind_parameters(circuit, parameter_values)
    pubs = [(bound, [observable])]
    job = est.run(pubs)
    return float(job.result()[0].data.evs)


def compute_expectation_batch(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    parameter_sets: list[list[float]],
    shots: int | None = None,
) -> list[float]:
    """Batch compute ⟨ψ|H|ψ⟩ for multiple parameter sets."""
    est = _estimator()
    pubs = [(circuit.assign_parameters(params), [observable])
            for params in parameter_sets]
    job = est.run(pubs)
    return [float(batch.data.evs) for batch in job.result()]


def sample_distribution(
    circuit: QuantumCircuit,
    parameter_values: list[float] | None = None,
    shots: int | None = None,
) -> dict[str, float]:
    """Return measurement outcome distribution via StatevectorSampler."""
    sampler = _sampler()
    bound = _bind_parameters(circuit, parameter_values)
    bound.measure_all()
    pubs = [(bound,)]
    job = sampler.run(pubs, shots=shots or BACKEND["shots"])
    counts = job.result()[0].data.meas.get_counts()
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def estimate_gradient(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    params: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Finite-difference gradient of ⟨ψ|H|ψ⟩ w.r.t. variational parameters."""
    grad = np.empty_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps

        e_plus = compute_expectation(circuit, observable, params_plus.tolist())
        e_minus = compute_expectation(circuit, observable, params_minus.tolist())
        grad[i] = (e_plus - e_minus) / (2 * eps)
    return grad
