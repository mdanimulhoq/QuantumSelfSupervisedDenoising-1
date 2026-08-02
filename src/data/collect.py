"""
Data collection protocol (TDD §5.2).
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import h5py
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.providers.aer.noise import NoiseModel

from src.data.encoding import counts_to_tensor, tensor_to_counts
from src.utils.seeding import set_seed


# ------------------------------------------------------------
# 1. Core simulation function
# ------------------------------------------------------------

def run_circuit(
    circuit: QuantumCircuit,
    shots: int = 1000,
    noise_model: Optional[NoiseModel] = None,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Run a circuit and return counts.

    Args:
        circuit: QuantumCircuit with measurements
        shots: Number of shots
        noise_model: NoiseModel to apply
        seed: Random seed for simulator

    Returns:
        CountsDict: {bitstring: count}
    """
    backend = Aer.get_backend("qasm_simulator")
    if noise_model is None:
        noise_model = NoiseModel()

    # Transpile for simulator
    transpiled = transpile(circuit, backend)

    job = execute(
        transpiled,
        backend,
        shots=shots,
        noise_model=noise_model,
        seed_simulator=seed,
    )
    result = job.result()
    return result.get_counts()


def run_circuit_with_caching(
    circuit: QuantumCircuit,
    shots: int = 1000,
    noise_model: Optional[NoiseModel] = None,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, int]:
    """
    Run a circuit with optional caching to disk.

    Args:
        circuit: QuantumCircuit with measurements
        shots: Number of shots
        noise_model: NoiseModel to apply
        seed: Random seed for simulator
        cache_dir: Directory to cache results (if None, no caching)

    Returns:
        CountsDict: {bitstring: count}
    """
    if cache_dir is None:
        return run_circuit(circuit, shots, noise_model, seed)

    # Generate cache key from circuit QASM + shots + seed
    import hashlib
    qasm = circuit.qasm()
    key_str = f"{qasm}_{shots}_{seed}_{str(noise_model)}"
    key = hashlib.md5(key_str.encode()).hexdigest()

    cache_path = os.path.join(cache_dir, f"{key}.npy")
    if os.path.exists(cache_path):
        counts = np.load(cache_path, allow_pickle=True).item()
        return counts

    counts = run_circuit(circuit, shots, noise_model, seed)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, counts)
    return counts


# ------------------------------------------------------------
# 2. Shot-pair collection (SN-D)
# ------------------------------------------------------------

def collect_shot_pairs(
    circuit: QuantumCircuit,
    low_shots: int = 100,
    high_shots: int = 10000,
    noise_model: Optional[NoiseModel] = None,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Collect low-shot and high-shot pairs for SN-D training.

    Args:
        circuit: QuantumCircuit with measurements
        low_shots: Number of shots for low-shot measurement
        high_shots: Number of shots for high-shot measurement
        noise_model: NoiseModel to apply
        seed: Random seed
        cache_dir: Directory for caching

    Returns:
        Dict with keys "low" and "high", each containing CountsDict
    """
    if seed is not None:
        set_seed(seed)

    # Run low-shot
    low_counts = run_circuit_with_caching(
        circuit, low_shots, noise_model, seed, cache_dir
    )

    # Run high-shot (use different seed for independence)
    high_seed = seed + 1000 if seed is not None else None
    high_counts = run_circuit_with_caching(
        circuit, high_shots, noise_model, high_seed, cache_dir
    )

    return {
        "low": low_counts,
        "high": high_counts,
    }


def collect_shot_pairs_batch(
    circuits: List[QuantumCircuit],
    low_shots: int = 100,
    high_shots: int = 10000,
    noise_model: Optional[NoiseModel] = None,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Dict[str, int]]]:
    """
    Collect shot pairs for a batch of circuits.

    Args:
        circuits: List of QuantumCircuits
        low_shots: Number of shots for low-shot measurement
        high_shots: Number of shots for high-shot measurement
        noise_model: NoiseModel to apply
        seed: Random seed
        cache_dir: Directory for caching

    Returns:
        List of Dicts with "low" and "high" counts
    """
    results = []
    for i, circuit in enumerate(circuits):
        circuit_seed = seed + i * 100 if seed is not None else None
        result = collect_shot_pairs(
            circuit, low_shots, high_shots, noise_model, circuit_seed, cache_dir
        )
        results.append(result)
    return results


# ------------------------------------------------------------
# 3. Noise-scaled collection (HN-E)
# ------------------------------------------------------------

def gate_fold(circuit: QuantumCircuit, factor: float) -> QuantumCircuit:
    """
    Apply gate folding for noise scaling.

    Args:
        circuit: QuantumCircuit
        factor: Noise scale factor (>1 for amplification)

    Returns:
        QuantumCircuit: Folded circuit
    """
    if factor <= 1.0:
        return circuit

    # Simple folding: repeat the circuit (factor - 1) times
    # More sophisticated folding would fold individual gates
    folded = circuit.copy()
    for _ in range(int(factor) - 1):
        folded = folded.compose(circuit)
    return folded


def add_dynamical_decoupling(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Add dynamical decoupling sequences to reduce noise.

    Args:
        circuit: QuantumCircuit

    Returns:
        QuantumCircuit: Circuit with DD sequences (currently placeholder)
    """
    # Simplified: no DD implemented for now
    # Full implementation would insert XY4 sequences
    return circuit


def collect_noise_scaled(
    circuit: QuantumCircuit,
    scale_factors: List[float] = None,
    shots: int = 1000,
    noise_model: Optional[NoiseModel] = None,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Collect noise-scaled data for HN-E training.

    Args:
        circuit: QuantumCircuit with measurements
        scale_factors: List of noise scale factors (e.g., [1.0, 1.5, 2.0, 2.5, 3.0])
        shots: Number of shots per scale factor
        noise_model: NoiseModel to apply
        seed: Random seed
        cache_dir: Directory for caching
        output_dir: Directory to save HDF5 files

    Returns:
        Dict mapping scale_factor -> CountsDict
    """
    if scale_factors is None:
        scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    if seed is not None:
        set_seed(seed)

    results = {}
    n_qubits = circuit.num_qubits

    for i, factor in enumerate(scale_factors):
        # Apply gate folding for noise scaling
        if factor > 1.0:
            folded_circuit = gate_fold(circuit, factor)
        else:
            folded_circuit = circuit

        # Add dynamical decoupling for factor < 1 (reduced noise)
        if factor < 1.0:
            folded_circuit = add_dynamical_decoupling(folded_circuit)

        # Run circuit
        circuit_seed = seed + i * 1000 if seed is not None else None
        counts = run_circuit_with_caching(
            folded_circuit, shots, noise_model, circuit_seed, cache_dir
        )
        results[str(factor)] = counts

    # Save to HDF5 if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        circuit_name = getattr(circuit, "name", "circuit")
        timestamp = int(time.time())
        filename = f"{circuit_name}_n{n_qubits}_seed{seed}_{timestamp}.h5"
        filepath = os.path.join(output_dir, filename)

        with h5py.File(filepath, "w") as f:
            f.attrs["n_qubits"] = n_qubits
            f.attrs["shots"] = shots
            f.attrs["seed"] = seed
            f.attrs["scale_factors"] = scale_factors
            f.attrs["timestamp"] = timestamp
            f.attrs["circuit_name"] = circuit_name

            for factor, counts in results.items():
                group = f.create_group(f"scale_{factor}")
                group.attrs["scale_factor"] = float(factor)

                # Convert counts to arrays
                bitstrings = []
                count_values = []
                for bs, cnt in counts.items():
                    # Pad bitstring to n_qubits
                    padded = bs.zfill(n_qubits)
                    bits = [int(b) for b in padded[::-1]]  # Reverse for qubit order
                    bitstrings.append(bits)
                    count_values.append(cnt)

                group.create_dataset("bitstrings", data=np.array(bitstrings, dtype=np.int8))
                group.create_dataset("counts", data=np.array(count_values, dtype=np.int32))

    return results


def collect_noise_scaled_batch(
    circuits: List[QuantumCircuit],
    scale_factors: List[float] = None,
    shots: int = 1000,
    noise_model: Optional[NoiseModel] = None,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> List[Dict[str, Dict[str, int]]]:
    """
    Collect noise-scaled data for a batch of circuits.

    Args:
        circuits: List of QuantumCircuits
        scale_factors: List of noise scale factors
        shots: Number of shots per scale factor
        noise_model: NoiseModel to apply
        seed: Random seed
        cache_dir: Directory for caching
        output_dir: Directory to save HDF5 files

    Returns:
        List of Dicts mapping scale_factor -> CountsDict
    """
    results = []
    for i, circuit in enumerate(circuits):
        circuit_seed = seed + i * 1000 if seed is not None else None
        circuit_output = os.path.join(output_dir, f"circuit_{i}") if output_dir else None
        result = collect_noise_scaled(
            circuit, scale_factors, shots, noise_model, circuit_seed, cache_dir, circuit_output
        )
        results.append(result)
    return results


# ------------------------------------------------------------
# 4. Convenience functions for full dataset generation
# ------------------------------------------------------------

def generate_snd_dataset(
    circuits: List[QuantumCircuit],
    output_dir: str,
    low_shots: int = 100,
    high_shots: int = 10000,
    noise_model: Optional[NoiseModel] = None,
    seed: int = 42,
    num_workers: int = 1,
) -> str:
    """
    Generate a complete SN-D dataset from a list of circuits.

    Args:
        circuits: List of QuantumCircuits
        output_dir: Directory to save HDF5 files
        low_shots: Number of shots for low-shot measurement
        high_shots: Number of shots for high-shot measurement
        noise_model: NoiseModel to apply
        seed: Random seed
        num_workers: Number of parallel workers (not yet implemented)

    Returns:
        Path to the generated dataset directory
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, circuit in enumerate(circuits):
        circuit_seed = seed + i * 100
        data = collect_shot_pairs(
            circuit, low_shots, high_shots, noise_model, circuit_seed
        )

        # Save to HDF5
        n_qubits = circuit.num_qubits
        circuit_name = getattr(circuit, "name", f"circuit_{i}")
        filename = f"{circuit_name}_n{n_qubits}_seed{circuit_seed}.h5"
        filepath = os.path.join(output_dir, filename)

        with h5py.File(filepath, "w") as f:
            f.attrs["n_qubits"] = n_qubits
            f.attrs["low_shots"] = low_shots
            f.attrs["high_shots"] = high_shots
            f.attrs["seed"] = circuit_seed
            f.attrs["circuit_name"] = circuit_name
            f.attrs["index"] = i

            # Low-shot data
            low_group = f.create_group("low")
            low_bits, low_counts = counts_to_tensor(data["low"], n_qubits, return_counts=True)
            low_group.create_dataset("bitstrings", data=low_bits.numpy())
            low_group.create_dataset("counts", data=low_counts.numpy())

            # High-shot data
            high_group = f.create_group("high")
            high_bits, high_counts = counts_to_tensor(data["high"], n_qubits, return_counts=True)
            high_group.create_dataset("bitstrings", data=high_bits.numpy())
            high_group.create_dataset("counts", data=high_counts.numpy())

    return output_dir


def generate_hne_dataset(
    circuits: List[QuantumCircuit],
    output_dir: str,
    scale_factors: List[float] = None,
    shots: int = 1000,
    noise_model: Optional[NoiseModel] = None,
    seed: int = 42,
) -> str:
    """
    Generate a complete HN-E dataset from a list of circuits.

    Args:
        circuits: List of QuantumCircuits
        output_dir: Directory to save HDF5 files
        scale_factors: List of noise scale factors
        shots: Number of shots per scale factor
        noise_model: NoiseModel to apply
        seed: Random seed

    Returns:
        Path to the generated dataset directory
    """
    if scale_factors is None:
        scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    os.makedirs(output_dir, exist_ok=True)

    for i, circuit in enumerate(circuits):
        circuit_seed = seed + i * 1000
        collect_noise_scaled(
            circuit, scale_factors, shots, noise_model, circuit_seed,
            output_dir=output_dir
        )

    return output_dir
