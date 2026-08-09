#!/usr/bin/env python
"""
Train SN-D (Shot-Noise Denoising) from scratch.
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


class SNDDataset(Dataset):
    def __init__(self, data_path, max_bitstrings=128):
        self.data_path = Path(data_path)
        self.max_bitstrings = max_bitstrings
        self.data = []
        self.n_qubits = 4
        self._load_data()
    
    def _load_data(self):
        with h5py.File(self.data_path, 'r') as f:
            print(f"   HDF5 keys: {list(f.keys())}")
            
            if 'n_qubits' in f:
                n_qubits_arr = f['n_qubits'][:]
                if len(n_qubits_arr) > 0:
                    self.n_qubits = int(n_qubits_arr[0])
                    print(f"   n_qubits from file: {self.n_qubits}")
            
            num_circuits = len(f['n_qubits'])
            print(f"   Number of circuits: {num_circuits}")
            
            for i in range(num_circuits):
                try:
                    low_bs_flat = f['low_bitstrings'][i]
                    low_p_flat = f['low_probs'][i]
                    high_bs_flat = f['high_bitstrings'][i]
                    high_p_flat = f['high_probs'][i]
                    
                    num_bitstrings = len(low_p_flat)
                    if num_bitstrings == 0:
                        continue
                    
                    low_bs_reshaped = low_bs_flat.reshape(num_bitstrings, self.n_qubits)
                    high_bs_reshaped = high_bs_flat.reshape(num_bitstrings, self.n_qubits)
                    
                    low = {}
                    high = {}
                    for j in range(num_bitstrings):
                        # Low
                        bs_int = 0
                        for bit in low_bs_reshaped[j]:
                            bs_int = (bs_int << 1) | int(bit)
                        bs_str = format(bs_int, f'0{self.n_qubits}b')
                        low[bs_str] = float(low_p_flat[j])
                        
                        # High
                        bs_int = 0
                        for bit in high_bs_reshaped[j]:
                            bs_int = (bs_int << 1) | int(bit)
                        bs_str = format(bs_int, f'0{self.n_qubits}b')
                        high[bs_str] = float(high_p_flat[j])
                    
                    if low and high:
                        self.data.append({'low': low, 'high': high})
                except Exception as e:
                    continue
        
        if not self.data:
            print("   Warning: No data found! Using dummy data...")
            for i in range(10):
                low = {f"{j:04b}": np.random.randint(1, 20) for j in range(4)}
                high = {f"{j:04b}": np.random.randint(1, 100) for j in range(4)}
                self.data.append({'low': low, 'high': high})
        
        print(f"   Loaded {len(self.data)} circuits")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert low to tensors and get bitstrings in order
        low_bitstrings, low_counts = self._to_tensors(sample['low'])
        # Align high counts to low bitstrings order
        high_counts = self._align_high_to_low(sample['high'], low_bitstrings)
        
        # Pad everything to max_bitstrings
        num_real = low_bitstrings.shape[0]
        if num_real < self.max_bitstrings:
            # Pad bitstrings with zeros
            pad_bs = torch.zeros(self.max_bitstrings - num_real, self.n_qubits, dtype=torch.long)
            low_bitstrings = torch.cat([low_bitstrings, pad_bs], dim=0)
            # Pad counts with zeros
            pad_cnt = torch.zeros(self.max_bitstrings - num_real, 1, dtype=torch.float32)
            low_counts = torch.cat([low_counts, pad_cnt], dim=0)
            # Pad targets with zeros
            pad_tgt = torch.zeros(self.max_bitstrings - num_real, dtype=torch.float32)
            high_counts = torch.cat([high_counts, pad_tgt], dim=0)
        else:
            # Truncate if more than max
            low_bitstrings = low_bitstrings[:self.max_bitstrings]
            low_counts = low_counts[:self.max_bitstrings]
            high_counts = high_counts[:self.max_bitstrings]
        
        # Create mask: 1 for real, 0 for padded
        mask = torch.zeros(self.max_bitstrings, dtype=torch.bool)
        mask[:num_real] = True
        
        return {
            'bitstrings': low_bitstrings,
            'counts': low_counts,
            'sn_target': high_counts,
            'hn_target': high_counts,
            'mask': mask,
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
    print("SN-D Training (from scratch)")
    print("=" * 60)
    
    device = get_device()
    set_seed(42)
    print(f"Device: {device}")
    
    data_dir = Path("data/raw/exp1_snd")
    train_dataset = SNDDataset(data_dir / 'exp1_snd_train.h5')
    val_dataset = SNDDataset(data_dir / 'exp1_snd_val.h5')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
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
        'batch_size': 16,
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
        log_dir=Path("experiments/exp1_snd/logs"),
        use_wandb=False,
    )
    
    print("\nTraining SN-D...")
    trainer.train(num_epochs=30, phase='phase1', save_every=5, early_stopping_patience=10)
    
    checkpoint_dir = Path("checkpoints/exp1_snd")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(checkpoint_dir / 'best_model.pt'))
    print(f"Checkpoint saved: {checkpoint_dir / 'best_model.pt'}")
    
    print("\nSN-D Training Complete!")

if __name__ == '__main__':
    main()
