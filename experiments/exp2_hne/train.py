#!/usr/bin/env python
"""
Train HN-E (Hardware-Noise Extrapolation) from scratch.
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, '/content/QuantumSelfSupervisedDenoising-1')

from src.models.n2ln import N2LNQEM
from src.losses.distribution import DistributionLoss
from src.losses.physicality import PhysicalityLoss
from src.training.trainer import Trainer
from src.utils.seeding import set_seed
from src.utils.device import get_device


class HNEDataset(Dataset):
    def __init__(self, data_path, n_qubits=4, max_bitstrings=128):
        self.data_path = Path(data_path)
        self.n_qubits = n_qubits
        self.max_bitstrings = max_bitstrings
        self.data = []
        self._load_data()
    
    def _load_data(self):
        with h5py.File(self.data_path, 'r') as f:
            print(f"   HDF5 keys: {list(f.keys())}")
            
            if 'n_qubits' in f:
                n_qubits_arr = f['n_qubits'][:]
                if len(n_qubits_arr) > 0:
                    self.n_qubits = int(n_qubits_arr[0])
                    print(f"   n_qubits from file: {self.n_qubits}")
            
            # Find all noise scales
            scales = []
            for key in f.keys():
                if key.startswith('bitstrings_'):
                    scale_str = key.replace('bitstrings_', '')
                    try:
                        scale = float(scale_str)
                        scales.append(scale)
                    except:
                        continue
            
            print(f"   Found noise scales: {scales}")
            
            if not scales:
                print("   Warning: No scale data found! Using dummy data...")
                self._create_dummy()
                return
            
            # Load data for each scale — bypass shapes!
            for scale in scales:
                bitstrings_key = f'bitstrings_{scale}'
                probs_key = f'probs_{scale}'
                
                if bitstrings_key not in f or probs_key not in f:
                    continue
                
                bitstrings_flat = f[bitstrings_key][:]  # list of arrays
                probs_flat = f[probs_key][:]            # list of arrays
                
                num_circuits = len(bitstrings_flat)
                print(f"   Scale {scale}: {num_circuits} circuits")
                
                for i in range(num_circuits):
                    bits = bitstrings_flat[i]   # flat bits for circuit i
                    probs = probs_flat[i]       # flat probs for circuit i
                    
                    # Skip if no probabilities
                    if len(probs) == 0:
                        continue
                    
                    num_bitstrings = len(probs)
                    
                    # Reshape bits: should be num_bitstrings * n_qubits
                    if len(bits) != num_bitstrings * self.n_qubits:
                        # If bits is already 2D, use as is
                        if bits.ndim == 2 and bits.shape[0] == num_bitstrings:
                            bits_reshaped = bits
                        else:
                            # Skip this circuit if shape mismatch
                            continue
                    else:
                        bits_reshaped = bits.reshape(num_bitstrings, self.n_qubits)
                    
                    low = {}
                    high = {}
                    for j in range(num_bitstrings):
                        bits_row = bits_reshaped[j]
                        bs_int = 0
                        for bit in bits_row:
                            bs_int = (bs_int << 1) | int(bit)
                        bs_str = format(bs_int, f'0{self.n_qubits}b')
                        low[bs_str] = float(probs[j])
                        high[bs_str] = float(probs[j])
                    
                    if low and high:
                        self.data.append({'low': low, 'high': high, 'scale': scale})
            
            if not self.data:
                print("   Warning: No valid circuits found! Using dummy data...")
                self._create_dummy()
            else:
                print(f"   Loaded {len(self.data)} circuits from real data")
    
    def _create_dummy(self):
        for i in range(10):
            counts = {f"{j:04b}": np.random.randint(1, 20) for j in range(4)}
            total = sum(counts.values())
            low = {bs: c/total for bs, c in counts.items()}
            high = {bs: c/total for bs, c in counts.items()}
            self.data.append({'low': low, 'high': high, 'scale': 1.0})
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        low_bitstrings, low_counts = self._to_tensors(sample['low'])
        high_counts = self._align_high_to_low(sample['high'], low_bitstrings)
        
        num_real = low_bitstrings.shape[0]
        if num_real < self.max_bitstrings:
            pad_bs = torch.zeros(self.max_bitstrings - num_real, self.n_qubits, dtype=torch.long)
            low_bitstrings = torch.cat([low_bitstrings, pad_bs], dim=0)
            pad_cnt = torch.zeros(self.max_bitstrings - num_real, 1, dtype=torch.float32)
            low_counts = torch.cat([low_counts, pad_cnt], dim=0)
            pad_tgt = torch.zeros(self.max_bitstrings - num_real, dtype=torch.float32)
            high_counts = torch.cat([high_counts, pad_tgt], dim=0)
        else:
            low_bitstrings = low_bitstrings[:self.max_bitstrings]
            low_counts = low_counts[:self.max_bitstrings]
            high_counts = high_counts[:self.max_bitstrings]
        
        mask = torch.zeros(self.max_bitstrings, dtype=torch.bool)
        mask[:num_real] = True
        
        return {
            'bitstrings': low_bitstrings,
            'counts': low_counts,
            'sn_target': high_counts,
            'hn_target': high_counts,
            'mask': mask,
            'scale': torch.tensor(sample.get('scale', 1.0), dtype=torch.float32),
        }
    
    def _to_tensors(self, counts_dict):
        items = sorted(counts_dict.items(), key=lambda x: -x[1])[:self.max_bitstrings]
        bitstrings = []
        counts = []
        total = sum(c for _, c in items)
        if total == 0:
            return torch.zeros(1, self.n_qubits, dtype=torch.long), torch.zeros(1, 1)
        for bs, c in items:
            bs_padded = bs.zfill(self.n_qubits)
            bitstrings.append([int(b) for b in bs_padded])
            counts.append(c / total)
        return torch.tensor(bitstrings, dtype=torch.long), torch.tensor(counts, dtype=torch.float32).unsqueeze(1)
    
    def _align_high_to_low(self, high_dict, low_bitstrings):
        probs = []
        for i in range(low_bitstrings.shape[0]):
            bs = ''.join([str(int(b)) for b in low_bitstrings[i]])
            probs.append(high_dict.get(bs, 0.0))
        return torch.tensor(probs, dtype=torch.float32)


def main():
    print("=" * 60)
    print("HN-E Training (from scratch)")
    print("=" * 60)
    
    device = get_device()
    set_seed(42)
    print(f"Device: {device}")
    
    data_dir = Path("data/raw/exp2_hne")
    train_dataset = HNEDataset(data_dir / 'exp2_hne_train.h5', n_qubits=4)
    val_dataset = HNEDataset(data_dir / 'exp2_hne_val.h5', n_qubits=4)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    n_qubits = train_dataset.n_qubits
    print(f"   Using n_qubits: {n_qubits}")
    
    model = N2LNQEM(
        d_model=64,
        n_heads=4,
        n_isab=2,
        n_sab=1,
        d_ff=256,
        m=16,
        decoder_hidden=128,
        dropout=0.1,
        max_qubits=n_qubits,
        n_qubits=n_qubits,
    )
    model = model.to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    loss_fns = {
        'snd': DistributionLoss(alpha=1.0, beta=0.5, gamma=0.1),
        'hne': DistributionLoss(alpha=1.0, beta=0.5, gamma=0.1),
        'physicality': PhysicalityLoss(),
        'consistency': None,
    }
    
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 8,
        'gradient_clip': 1.0,
        'log_interval': 10,
        'seed': 42,
    }
    
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fns=loss_fns,
        device=device,
        log_dir=Path("experiments/exp2_hne/logs"),
        use_wandb=False,
    )
    
    print("\nTraining HN-E...")
    trainer.train(num_epochs=30, phase='phase1', save_every=5, early_stopping_patience=10)
    
    checkpoint_dir = Path("checkpoints/exp2_hne")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(checkpoint_dir / 'best_model.pt'))
    print(f"Checkpoint saved: {checkpoint_dir / 'best_model.pt'}")
    
    print("\nHN-E Training Complete!")

if __name__ == '__main__':
    main()
