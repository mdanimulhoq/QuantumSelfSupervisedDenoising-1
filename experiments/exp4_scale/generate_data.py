#!/usr/bin/env python
"""
Generate large-scale datasets for qubit-count generalization.
Qubits: 10, 15, 20, 25, 30
Uses MPS simulator with fewer shots.
Streamed to disk in HDF5 format.
"""

import os
import sys
import json
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import torch
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.simulators.mps import MPSSimulator
from src.data.circuit_generator import random_circuit, clifford_circuit


# ============================================================
# Configuration
# ============================================================

QUBITS_LIST = [10, 15, 20, 25, 30]
NUM_CIRCUITS_PER_QUBIT = 100  # Small for testing, use 500 for full
SHOTS = 100  # Fewer shots for large scale
MAX_BOND_DIMENSION = 256
SEED = 42

# ============================================================
# Circuit Generation
# ============================================================

def generate_circuit(n_qubits: int, depth: int = 3, seed: int = None) -> QuantumCircuit:
    """Generate a random circuit with low entanglement."""
    if seed is not None:
        np.random.seed(seed)
    
    # Use 1D nearest-neighbor connectivity for low entanglement
    qc = QuantumCircuit(n_qubits)
    
    # Random single-qubit gates
    for qubit in range(n_qubits):
        qc.u(np.random.rand() * 2 * np.pi, 
             np.random.rand() * 2 * np.pi,
             np.random.rand() * 2 * np.pi, qubit)
    
    # CNOT gates on nearest neighbors (low entanglement)
    for layer in range(depth):
        for qubit in range(0, n_qubits - 1, 2):
            qc.cx(qubit, qubit + 1)
        for qubit in range(1, n_qubits - 1, 2):
            qc.cx(qubit, qubit + 1)
    
    return qc

# ============================================================
# Data Collection
# ============================================================

def collect_data_for_qubits(
    n_qubits: int,
    num_circuits: int = NUM_CIRCUITS_PER_QUBIT,
    shots: int = SHOTS,
    max_bond_dimension: int = MAX_BOND_DIMENSION,
    seed: int = SEED,
) -> dict:
    """
    Collect data for a specific qubit count.
    
    Returns:
        Dictionary with bitstrings and counts
    """
    np.random.seed(seed)
    
    mps_sim = MPSSimulator(
        max_bond_dimension=max_bond_dimension,
        shots=shots,
        seed=seed,
    )
    
    all_bitstrings = []
    all_counts = []
    
    for circ_idx in tqdm(range(num_circuits), desc=f"n={n_qubits}"):
        # Generate circuit
        circuit = generate_circuit(n_qubits, depth=3, seed=seed + circ_idx)
        
        # Run simulation
        counts = mps_sim.run(circuit, shots=shots)
        
        # Convert to arrays
        bitstrings = []
        counts_array = []
        for bs, c in counts.items():
            bitstrings.append(bs)
            counts_array.append(c)
        
        # Pad to fixed length for HDF5
        max_bitstrings = min(2 ** n_qubits, 128)  # Cap for large n
        if len(bitstrings) > max_bitstrings:
            # Keep top bitstrings
            sorted_items = sorted(zip(bitstrings, counts_array), key=lambda x: -x[1])
            bitstrings = [bs for bs, _ in sorted_items[:max_bitstrings]]
            counts_array = [c for _, c in sorted_items[:max_bitstrings]]
        
        all_bitstrings.append(bitstrings)
        all_counts.append(counts_array)
    
    return {
        'bitstrings': all_bitstrings,
        'counts': all_counts,
        'n_qubits': n_qubits,
        'shots': shots,
        'num_circuits': num_circuits,
    }

# ============================================================
# Save to HDF5
# ============================================================

def save_to_hdf5(data: dict, output_path: Path) -> None:
    """Save collected data to HDF5 file."""
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['n_qubits'] = data['n_qubits']
        f.attrs['shots'] = data['shots']
        f.attrs['num_circuits'] = data['num_circuits']
        f.attrs['created_at'] = '2026-08-02'
        
        # Store bitstrings as strings
        bitstrings = data['bitstrings']
        counts = data['counts']
        
        # Convert to fixed-length arrays
        max_len = max(len(bs) for bs in bitstrings)
        
        # Store bitstrings
        bs_dataset = f.create_dataset('bitstrings', 
                                       shape=(len(bitstrings), max_len),
                                       dtype=h5py.string_dtype())
        for i, bs_list in enumerate(bitstrings):
            # Pad with empty strings
            padded = bs_list + [''] * (max_len - len(bs_list))
            bs_dataset[i] = padded
        
        # Store counts
        counts_dataset = f.create_dataset('counts',
                                          shape=(len(counts), max_len),
                                          dtype=np.int32)
        for i, c_list in enumerate(counts):
            padded = c_list + [0] * (max_len - len(c_list))
            counts_dataset[i] = padded
        
        # Store actual lengths
        lengths = [len(bs) for bs in bitstrings]
        f.create_dataset('lengths', data=lengths)

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Experiment 4: Qubit Scaling - Large-scale Dataset")
    print("=" * 60)
    
    output_dir = Path("data/raw/exp4_scale")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data for each qubit count
    for n_qubits in QUBITS_LIST:
        print(f"\n📊 Generating data for n={n_qubits}...")
        
        data = collect_data_for_qubits(
            n_qubits=n_qubits,
            num_circuits=NUM_CIRCUITS_PER_QUBIT,
            shots=SHOTS,
        )
        
        # Save
        output_path = output_dir / f"exp4_scale_{n_qubits}.h5"
        save_to_hdf5(data, output_path)
        print(f"   ✅ Saved: {output_path}")
    
    print("\n" + "=" * 60)
    print("🎉 Large-scale dataset generation complete!")
    print("=" * 60)
    
    # Print summary
    print("\n📊 Dataset Summary:")
    for n in QUBITS_LIST:
        path = output_dir / f"exp4_scale_{n}.h5"
        if path.exists():
            size = path.stat().st_size
            print(f"   n={n}: {size / 1024:.1f} KB")

if __name__ == '__main__':
    main()
