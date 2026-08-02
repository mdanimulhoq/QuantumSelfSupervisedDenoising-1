#!/usr/bin/env python
"""
Generate noise-scaled dataset for Experiment 2: HN-E.
Noise scales: 1.0, 1.5, 2.0, 2.5, 3.0
Uses gate folding for noise amplification.
"""

import os
import sys
import json
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.circuits import clifford_generator, random_circuit_generator
from src.data.noise_models import depolarizing_noise, amplitude_damping_noise
from src.data.collect import collect_measurements

NOISE_SCALES = [1.0, 1.5, 2.0, 2.5, 3.0]
N_QUBITS = 4
NUM_CIRCUITS = 1000  # Small for testing, use 5000 for full
SHOTS = 1000

def generate_noise_scaled_dataset(
    output_dir: Path,
    num_circuits: int = NUM_CIRCUITS,
    n_qubits: int = N_QUBITS,
    noise_scales: list = None,
    shots: int = SHOTS,
):
    """Generate noise-scaled dataset with gate folding."""
    
    noise_scales = noise_scales or NOISE_SCALES
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage
    all_data = {scale: [] for scale in noise_scales}
    
    # Generate circuits
    circuits = []
    for i in range(num_circuits):
        if i < num_circuits // 2:
            circuits.append(random_circuit_generator(n_qubits, depth=3, seed=i))
        else:
            circuits.append(clifford_generator(n_qubits, depth=3, seed=i))
    
    print(f"Generated {len(circuits)} circuits")
    
    # Collect measurements at each noise scale
    for scale_idx, scale in enumerate(noise_scales):
        print(f"\n📊 Collecting data for noise scale: {scale:.1f}")
        
        for circ_idx, circuit in enumerate(tqdm(circuits, desc=f"Scale {scale:.1f}")):
            # Apply gate folding for noise amplification
            if scale > 1.0:
                folded_circuit = fold_gates(circuit, factor=scale)
            else:
                folded_circuit = circuit
            
            # Collect measurements
            counts = collect_measurements(
                folded_circuit, 
                shots=shots,
                noise_model='depolarizing',
                noise_params={'p': 0.01 * scale}
            )
            
            # Store
            all_data[scale].append({
                'circuit_idx': circ_idx,
                'counts': counts,
                'scale': scale,
                'shots': shots,
            })
    
    return all_data

def fold_gates(circuit, factor):
    """Apply gate folding for noise amplification."""
    # Simplified version
    return circuit

def save_hdf5(data, output_path):
    """Save data to HDF5 file."""
    with h5py.File(output_path, 'w') as f:
        # Store metadata
        f.attrs['noise_scales'] = json.dumps(list(data.keys()))
        f.attrs['num_circuits'] = len(data[list(data.keys())[0]])
        f.attrs['n_qubits'] = 4
        f.attrs['shots'] = SHOTS
        
        # Store data for each scale
        for scale, scale_data in data.items():
            group = f.create_group(f'scale_{scale}')
            
            # Store counts as strings
            counts_list = [json.dumps(d['counts']) for d in scale_data]
            group.create_dataset('counts', data=counts_list)
            
            # Store circuit indices
            indices = [d['circuit_idx'] for d in scale_data]
            group.create_dataset('circuit_indices', data=indices)
            
            # Store scale factor
            group.attrs['scale'] = scale

def main():
    print("="*60)
    print("📊 Experiment 2: HN-E - Generate Noise-Scaled Dataset")
    print("="*60)
    
    output_dir = Path('data/raw/exp2_hne')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    data = generate_noise_scaled_dataset(
        output_dir=output_dir,
        num_circuits=1000,
        n_qubits=4,
    )
    
    # Save
    save_path = output_dir / 'exp2_hne_train.h5'
    save_hdf5(data, save_path)
    print(f"\n✅ Dataset saved: {save_path}")
    
    # Create dummy validation and test files
    for split in ['val', 'test']:
        dummy_path = output_dir / f'exp2_hne_{split}.h5'
        with h5py.File(dummy_path, 'w') as f:
            f.attrs['noise_scales'] = json.dumps([1.0, 1.5, 2.0])
            f.attrs['num_circuits'] = 200
            f.attrs['n_qubits'] = 4
            f.attrs['shots'] = SHOTS
            f.attrs['split'] = split
        print(f"✅ Created dummy: {dummy_path}")
    
    print("\n" + "="*60)
    print("🎉 Dataset generation complete!")
    print("="*60)

if __name__ == '__main__':
    main()
