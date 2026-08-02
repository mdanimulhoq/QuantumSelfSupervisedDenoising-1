"""
MPS (Matrix Product State) simulator wrapper for N2LN-QEM.
Provides the same interface as Qiskit Aer for up to 30 qubits.

Implements TDD §7.2 Phase B.
"""

import numpy as np
from typing import Dict, Optional, List, Union, Tuple
from qiskit import QuantumCircuit
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit import transpile


class MPSSimulator:
    """
    Wrapper for Qiskit Aer MPS simulator using Aer.get_backend().
    
    Uses matrix product state representation for efficient simulation
    of low-entanglement circuits up to ~30 qubits.
    """
    
    def __init__(
        self,
        max_bond_dimension: int = 256,
        shots: int = 1000,
        noise_model: Optional[NoiseModel] = None,
        seed: Optional[int] = None,
    ):
        self.max_bond_dimension = max_bond_dimension
        self.shots = shots
        self.noise_model = noise_model
        self.seed = seed
        
        # Use Aer.get_backend with method='matrix_product_state'
        self._backend = Aer.get_backend('qasm_simulator')
        self._backend.set_options(
            method='matrix_product_state',
            max_bond_dimension=max_bond_dimension,
            shots=shots,
            seed_simulator=seed,
        )
    
    def run(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> Dict[str, int]:
        """Run circuit and return counts."""
        shots = shots or self.shots
        noise = noise_model or self.noise_model
        
        backend = Aer.get_backend('qasm_simulator')
        backend.set_options(
            method='matrix_product_state',
            max_bond_dimension=self.max_bond_dimension,
            shots=shots,
            seed_simulator=self.seed,
        )
        if noise is not None:
            backend.set_options(noise_model=noise)
        
        # Transpile for better performance
        transpiled = transpile(circuit, backend)
        result = backend.run(transpiled).result()
        counts = result.get_counts()
        return dict(counts)
    
    def get_probabilities(
        self,
        circuit: QuantumCircuit,
        noise_model: Optional[NoiseModel] = None,
    ) -> Dict[str, float]:
        """Get exact probabilities (no sampling)."""
        backend = Aer.get_backend('statevector_simulator')
        backend.set_options(
            method='matrix_product_state',
            max_bond_dimension=self.max_bond_dimension,
            seed_simulator=self.seed,
        )
        if noise_model is not None:
            backend.set_options(noise_model=noise_model)
        
        transpiled = transpile(circuit, backend)
        result = backend.run(transpiled).result()
        statevector = result.get_statevector()
        
        # Convert to probabilities
        probs = np.abs(statevector) ** 2
        n_qubits = circuit.num_qubits
        
        prob_dict = {}
        for i, p in enumerate(probs):
            if p > 1e-12:
                bitstring = format(i, f'0{n_qubits}b')
                prob_dict[bitstring] = float(p)
        return prob_dict
    
    def sample(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> np.ndarray:
        """Sample bitstrings."""
        counts = self.run(circuit, shots=shots, noise_model=noise_model)
        samples = []
        for bitstring, count in counts.items():
            samples.extend([int(bitstring, 2)] * count)
        return np.array(samples)
    
    def get_state_vector(self, circuit: QuantumCircuit) -> np.ndarray:
        """Get state vector."""
        backend = Aer.get_backend('statevector_simulator')
        backend.set_options(
            method='matrix_product_state',
            max_bond_dimension=self.max_bond_dimension,
        )
        transpiled = transpile(circuit, backend)
        result = backend.run(transpiled).result()
        return np.array(result.get_statevector())


def create_mps_simulator(
    max_bond_dimension: int = 256,
    shots: int = 1000,
    seed: Optional[int] = None,
) -> MPSSimulator:
    """Convenience function to create an MPS simulator."""
    return MPSSimulator(
        max_bond_dimension=max_bond_dimension,
        shots=shots,
        seed=seed,
    )


def compare_aer_vs_mps(
    circuit: QuantumCircuit,
    n_shots: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compare Aer (exact) and MPS simulator results."""
    from src.losses.distribution import total_variation_distance
    import torch
    
    # Aer exact
    aer_sim = Aer.get_backend('statevector_simulator')
    transpiled = transpile(circuit, aer_sim)
    result = aer_sim.run(transpiled).result()
    aer_probs = np.abs(result.get_statevector()) ** 2
    
    # MPS probabilities
    mps_sim = MPSSimulator(max_bond_dimension=256, seed=seed)
    mps_probs_dict = mps_sim.get_probabilities(circuit)
    
    n_qubits = circuit.num_qubits
    mps_probs = np.zeros(2 ** n_qubits)
    for bs, p in mps_probs_dict.items():
        idx = int(bs, 2)
        mps_probs[idx] = p
    
    aer_tensor = torch.tensor(aer_probs, dtype=torch.float32)
    mps_tensor = torch.tensor(mps_probs, dtype=torch.float32)
    
    tvd = total_variation_distance(mps_tensor, aer_tensor).item()
    fidelity = (torch.sqrt(aer_tensor * mps_tensor).sum() ** 2).item()
    
    return {
        'tvd': tvd,
        'fidelity': fidelity,
        'max_bond_dimension': 256,
        'n_qubits': n_qubits,
    }


def estimate_mps_entropy(
    circuit: QuantumCircuit,
    max_bond_dimension: int = 256,
) -> Tuple[float, bool]:
    """Estimate entanglement entropy."""
    backend = Aer.get_backend('statevector_simulator')
    backend.set_options(
        method='matrix_product_state',
        max_bond_dimension=max_bond_dimension,
    )
    try:
        transpiled = transpile(circuit, backend)
        result = backend.run(transpiled).result()
        if result.success:
            bond_dim = min(max_bond_dimension, 2 ** (circuit.num_qubits // 2))
            entropy = np.log2(bond_dim)
            return entropy, True
        else:
            return 0.0, False
    except Exception as e:
        return 0.0, False
