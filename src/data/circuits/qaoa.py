"""
QAOA circuit generator (TDD §5.1).
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def generate_qaoa(
    n_qubits: int,
    depth: int = 1,
    seed: int = None,
    problem_type: str = "maxcut",
) -> QuantumCircuit:
    """
    Generate a QAOA-style circuit.

    Args:
        n_qubits: Number of qubits
        depth: Number of QAOA layers (p)
        seed: Random seed
        problem_type: "maxcut" or "ising"

    Returns:
        QuantumCircuit: QAOA circuit with measurements
    """
    if seed is not None:
        np.random.seed(seed)

    # Create a simple Max-Cut operator on a ring graph
    # For n_qubits, we create a ring graph: (0-1), (1-2), ..., (n-2, n-1), (n-1, 0)
    pauli_list = []
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        # Create Pauli string: Z on qubits i and j
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        pauli[j] = 'Z'
        pauli_str = ''.join(pauli)
        pauli_list.append((pauli_str, 1.0))

    operator = SparsePauliOp.from_list(pauli_list)
    circuit = QAOAAnsatz(operator, reps=depth)

    # Bind random parameters
    params = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)
    circuit = circuit.assign_parameters(params)
    circuit.measure_all()
    circuit.name = f"qaoa_n{n_qubits}_p{depth}"
    return circuit

# Alias for compatibility
generate_qaoa_ansatz = generate_qaoa
