"""
Clifford circuit generator (TDD §5.1).
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford
import numpy as np


def generate_clifford(n_qubits: int, depth: int = 2, seed: int = None) -> QuantumCircuit:
    """
    Generate a random Clifford circuit.

    Args:
        n_qubits: Number of qubits
        depth: Not used for Clifford (always depth 1), kept for API consistency
        seed: Random seed

    Returns:
        QuantumCircuit: Clifford circuit with measurements
    """
    if seed is not None:
        np.random.seed(seed)
    cliff = random_clifford(n_qubits, seed=seed)
    circuit = cliff.to_circuit()
    circuit.measure_all()
    circuit.name = f"clifford_n{n_qubits}"
    return circuit
