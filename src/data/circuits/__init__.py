"""
Circuit generators (TDD §5.1).
"""

from src.data.circuits.clifford import generate_clifford
from src.data.circuits.random_layer import generate_random
from src.data.circuits.vqe import generate_vqe, generate_vqe_ansatz
from src.data.circuits.qaoa import generate_qaoa, generate_qaoa_ansatz


def get_circuit_generators(circuit_type: str):
    """
    Get a circuit generator function by type.

    Args:
        circuit_type: "clifford", "random", "vqe", "qaoa"

    Returns:
        Callable: generator function with signature (n_qubits, depth, seed)
    """
    generators = {
        "clifford": generate_clifford,
        "random": generate_random,
        "vqe": generate_vqe,
        "qaoa": generate_qaoa,
    }
    return generators.get(circuit_type)
