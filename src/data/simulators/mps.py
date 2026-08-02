"""
MPS (Matrix Product State) simulator wrapper for N2LN-QEM.
Provides the same interface as Qiskit Aer for up to 30 qubits.

Implements TDD §7.2 Phase B.
"""

import numpy as np
from typing import Dict, Optional, List, Union, Tuple
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.result import Counts


class MPSSimulator:
    """
    Wrapper for Qiskit Aer MPS simulator.
    
    Uses matrix product state representation for efficient simulation
    of low-entanglement circuits up to ~30 qubits.
    
    Attributes:
        max_bond_dimension: Maximum bond dimension for MPS (default: 256)
        shots: Number of shots for sampling
        noise_model: Optional noise model
        seed: Random seed for reproducibility
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
        
        # Create simulator with MPS method
        # Note: max_bond_dimension is set via simulator_options
        self._simulator = AerSimulator(
            method='matrix_product_state',
            shots=shots,
            seed_simulator=seed,
        )
        # Set max bond dimension via simulator_options
        self._simulator.set_options(
            max_bond_dimension=max_bond_dimension
        )
    
    def run(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> Dict[str, int]:
        """
        Run a circuit and return counts.
        
        Args:
            circuit: QuantumCircuit to simulate
            shots: Number of shots (overrides default)
            noise_model: Optional noise model (overrides default)
        
        Returns:
            Counts dictionary {bitstring: count}
        """
        shots = shots or self.shots
        noise = noise_model or self.noise_model
        
        # Build simulator with parameters
        simulator = AerSimulator(
            method='matrix_product_state',
            shots=shots,
            seed_simulator=self.seed,
        )
        simulator.set_options(max_bond_dimension=self.max_bond_dimension)
        
        if noise is not None:
            simulator.set_options(noise_model=noise)
        
        result = simulator.run(circuit).result()
        counts = result.get_counts()
        return dict(counts)
    
    def get_probabilities(
        self,
        circuit: QuantumCircuit,
        noise_model: Optional[NoiseModel] = None,
    ) -> Dict[str, float]:
        """
        Get exact probabilities (no sampling) from the MPS simulator.
        
        Args:
            circuit: QuantumCircuit to simulate
            noise_model: Optional noise model
        
        Returns:
            Probability dictionary {bitstring: probability}
        """
        simulator = AerSimulator(
            method='matrix_product_state',
            seed_simulator=self.seed,
        )
        simulator.set_options(max_bond_dimension=self.max_bond_dimension)
        
        if noise_model is not None:
            simulator.set_options(noise_model=noise_model)
        
        circuit_copy = circuit.copy()
        circuit_copy.save_probabilities()
        
        result = simulator.run(circuit_copy).result()
        probs = result.data(0)['probabilities']
        
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
        """
        Sample bitstrings from the circuit.
        
        Args:
            circuit: QuantumCircuit to simulate
            shots: Number of shots
            noise_model: Optional noise model
        
        Returns:
            Array of sampled bitstrings as integers
        """
        counts = self.run(circuit, shots=shots, noise_model=noise_model)
        
        samples = []
        for bitstring, count in counts.items():
            samples.extend([int(bitstring, 2)] * count)
        
        return np.array(samples)
    
    def get_state_vector(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Get the MPS state vector (if bond dimension allows).
        
        Args:
            circuit: QuantumCircuit to simulate
        
        Returns:
            State vector as numpy array
        """
        simulator = AerSimulator(
            method='matrix_product_state',
        )
        simulator.set_options(max_bond_dimension=self.max_bond_dimension)
        
        circuit_copy = circuit.copy()
        circuit_copy.save_statevector()
        
        result = simulator.run(circuit_copy).result()
        statevector = result.data(0)['statevector']
        return np.array(statevector)


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
    
    aer_sim = AerSimulator(seed_simulator=seed)
    circuit_copy = circuit.copy()
    circuit_copy.save_probabilities()
    result = aer_sim.run(circuit_copy).result()
    aer_probs = result.data(0)['probabilities']
    
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
    """Estimate the entanglement entropy of a circuit for MPS simulation."""
    simulator = AerSimulator(
        method='matrix_product_state',
    )
    simulator.set_options(max_bond_dimension=max_bond_dimension)
    
    try:
        circuit_copy = circuit.copy()
        circuit_copy.save_statevector()
        result = simulator.run(circuit_copy).result()
        
        if result.success:
            bond_dim = min(max_bond_dimension, 2 ** (circuit.num_qubits // 2))
            entropy = np.log2(bond_dim)
            return entropy, True
        else:
            return 0.0, False
    except Exception:
        return 0.0, False
