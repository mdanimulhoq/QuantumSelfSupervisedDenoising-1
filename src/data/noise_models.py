"""
Noise models for QEM (TDD §5.3).
"""

from typing import Dict, Optional, Union

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import (
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
    readout_error,
)
from qiskit.providers.fake_provider import FakeManila, FakeJakarta, FakeLima, FakeVigo


def depolarizing(p: float, n_qubits: int = 1) -> NoiseModel:
    """
    Create a depolarizing noise model.

    Args:
        p: Depolarizing error probability per gate
        n_qubits: Number of qubits the error acts on (1 or 2)

    Returns:
        NoiseModel: Depolarizing noise model
    """
    if p <= 0:
        return NoiseModel()

    noise_model = NoiseModel()
    if n_qubits == 1:
        error = depolarizing_error(p, 1)
        noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
    elif n_qubits == 2:
        error = depolarizing_error(p, 2)
        noise_model.add_all_qubit_quantum_error(error, ["cx", "cz", "swap"])
    else:
        raise ValueError(f"n_qubits must be 1 or 2, got {n_qubits}")

    return noise_model


def amplitude_damping(gamma: float) -> NoiseModel:
    """
    Create an amplitude damping noise model.

    Args:
        gamma: Amplitude damping probability (T1 decay)

    Returns:
        NoiseModel: Amplitude damping noise model
    """
    if gamma <= 0:
        return NoiseModel()

    noise_model = NoiseModel()
    error = amplitude_damping_error(gamma)
    noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
    return noise_model


def phase_damping(lam: float) -> NoiseModel:
    """
    Create a phase damping noise model.

    Args:
        lam: Phase damping probability (T2 decay)

    Returns:
        NoiseModel: Phase damping noise model
    """
    if lam <= 0:
        return NoiseModel()

    noise_model = NoiseModel()
    error = phase_damping_error(lam)
    noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
    return noise_model


def thermal_relaxation(
    t1: float = 50e-6,
    t2: float = 30e-6,
    gate_time: float = 1e-6,
    temperature: float = 0.0,
) -> NoiseModel:
    """
    Create a thermal relaxation noise model.

    Args:
        t1: T1 relaxation time in seconds
        t2: T2 dephasing time in seconds
        gate_time: Gate duration in seconds
        temperature: Temperature for excited state population

    Returns:
        NoiseModel: Thermal relaxation noise model
    """
    noise_model = NoiseModel()
    error = thermal_relaxation_error(t1, t2, gate_time, temperature)
    noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
    return noise_model


def ibmq_realistic(backend_name: str = "fake_manila") -> NoiseModel:
    """
    Create an IBMQ-realistic noise model from a fake backend.

    Args:
        backend_name: Name of the fake backend.
            Options: "fake_manila", "fake_jakarta", "fake_lima", "fake_vigo"

    Returns:
        NoiseModel: Noise model from backend calibration

    Raises:
        ValueError: If backend_name is unknown
    """
    backend_map = {
        "fake_manila": FakeManila,
        "fake_jakarta": FakeJakarta,
        "fake_lima": FakeLima,
        "fake_vigo": FakeVigo,
    }

    if backend_name not in backend_map:
        available = list(backend_map.keys())
        raise ValueError(
            f"Unknown backend: {backend_name}. Available: {available}"
        )

    backend = backend_map[backend_name]()
    noise_model = NoiseModel.from_backend(backend)

    # Add readout errors if not already present
    # (Some fake backends already include readout errors)
    if not noise_model.has_readout_error:
        # Add a simple readout error model
        readout_error_prob = 0.02
        error = readout_error(readout_error_prob, readout_error_prob)
        for qubit in range(backend.num_qubits):
            noise_model.add_readout_error(error, [qubit])

    return noise_model


def combined_noise(
    p_depol: float = 0.001,
    gamma_amp: float = 0.0,
    lam_phase: float = 0.0,
    readout_error_prob: float = 0.02,
    backend_name: Optional[str] = None,
) -> NoiseModel:
    """
    Create a combined noise model with multiple error sources.

    Args:
        p_depol: Depolarizing error probability
        gamma_amp: Amplitude damping probability
        lam_phase: Phase damping probability
        readout_error_prob: Readout error probability (0->1 and 1->0)
        backend_name: If provided, use IBMQ-realistic backend as base

    Returns:
        NoiseModel: Combined noise model
    """
    if backend_name:
        noise_model = ibmq_realistic(backend_name)
    else:
        noise_model = NoiseModel()

    # Add depolarizing
    if p_depol > 0:
        depol_error = depolarizing_error(p_depol, 1)
        noise_model.add_all_qubit_quantum_error(depol_error, ["u1", "u2", "u3"])
        depol_error_2q = depolarizing_error(p_depol, 2)
        noise_model.add_all_qubit_quantum_error(depol_error_2q, ["cx", "cz", "swap"])

    # Add amplitude damping
    if gamma_amp > 0:
        amp_error = amplitude_damping_error(gamma_amp)
        noise_model.add_all_qubit_quantum_error(amp_error, ["u1", "u2", "u3"])

    # Add phase damping
    if lam_phase > 0:
        phase_error = phase_damping_error(lam_phase)
        noise_model.add_all_qubit_quantum_error(phase_error, ["u1", "u2", "u3"])

    # Add readout errors
    if readout_error_prob > 0 and not noise_model.has_readout_error:
        ro_error = readout_error(readout_error_prob, readout_error_prob)
        # Add to all qubits (max 10 qubits for compatibility)
        for q in range(10):
            noise_model.add_readout_error(ro_error, [q])

    return noise_model
