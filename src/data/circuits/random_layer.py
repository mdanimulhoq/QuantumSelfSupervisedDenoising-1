"""
Random circuit generator (TDD §5.1).
"""

from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
import numpy as np


def generate_random(n_qubits: int, depth: int = 2, seed: int = None) -> QuantumCircuit:
    """
    Generate a random circuit with single-qubit and CNOT gates.

    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
        seed: Random seed

    Returns:
        QuantumCircuit: Random circuit with measurements
    """
    if seed is not None:
        np.random.seed(seed)
    circuit = random_circuit(n_qubits, depth, measure=False, seed=seed)
    circuit.measure_all()
    circuit.name = f"random_n{n_qubits}_d{depth}"
    return circuit
