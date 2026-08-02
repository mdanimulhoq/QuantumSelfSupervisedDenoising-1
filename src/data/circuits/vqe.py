"""
VQE ansatz circuit generator (TDD §5.1).
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
import numpy as np


def generate_vqe(
    n_qubits: int,
    depth: int = 1,
    seed: int = None,
    ansatz_type: str = "efficient_su2",
) -> QuantumCircuit:
    """
    Generate a VQE-style ansatz circuit.

    Args:
        n_qubits: Number of qubits
        depth: Number of repetitions (reps)
        seed: Random seed
        ansatz_type: "efficient_su2" or "real_amplitudes"

    Returns:
        QuantumCircuit: VQE ansatz with measurements
    """
    if seed is not None:
        np.random.seed(seed)

    if ansatz_type == "efficient_su2":
        circuit = EfficientSU2(n_qubits, reps=depth, entanglement="linear")
    elif ansatz_type == "real_amplitudes":
        circuit = RealAmplitudes(n_qubits, reps=depth, entanglement="linear")
    else:
        raise ValueError(f"Unknown ansatz_type: {ansatz_type}")

    # Bind random parameters
    params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)
    circuit = circuit.assign_parameters(params)
    circuit.measure_all()
    circuit.name = f"vqe_{ansatz_type}_n{n_qubits}_d{depth}"
    return circuit

# Alias for compatibility
generate_vqe_ansatz = generate_vqe
