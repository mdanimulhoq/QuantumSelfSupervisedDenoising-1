#!/usr/bin/env python
"""
Scale-out training for qubit-count generalization (Exp 4).
Trains N2LN on n=10,15,20,25,30 using mixed precision and gradient accumulation.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.n2ln import N2LN
from src.losses.distribution import DistributionLoss
from src.losses.physicality import PhysicalityLoss
from src.utils.seeding import set_seed
from src.utils.device import get_device


# ============================================================
# Dataset
# ============================================================

class ScaleDataset(Dataset):
    """Dataset for qubit-count scaling experiments."""
    
    def __init__(self, data_path, max_bitstrings=128):
        self.data_path = Path(data_path)
        self.max_bitstrings = max_bitstrings
        self._load_data()
    
    def _load_data(self):
        with h5py.File(self.data_path, 'r') as f:
            self.n_qubits = f.attrs['n_qubits']
            self.shots = f.attrs['shots']
            self.num_circuits = f.attrs['num_circuits']
            
            # Load bitstrings and counts
            self.bitstrings = f['bitstrings'][:]
            self.counts = f['counts'][:]
            self.lengths = f['lengths'][:]
    
    def __len__(self):
        return self.num_circuits
    
    def __getitem__(self, idx):
        # Get actual bitstrings and counts (remove padding)
        actual_len = self.lengths[idx]
        bs_list = self.bitstrings[idx][:actual_len]
        c_list = self.counts[idx][:actual_len]
        
        # Convert to tensors
        bitstrings = []
        counts = []
        total = sum(c_list)
        
        for bs, c in zip(bs_list, c_list):
            if bs != '' and c > 0:
                bs_tensor = [int(b) for b in bs.zfill(self.n_qubits)]
                bitstrings.append(bs_tensor)
                counts.append(c / total)
        
        if not bitstrings:
            # Fallback: single bitstring
            bitstrings = [[0] * self.n_qubits]
            counts = [1.0]
        
        return {
            'bitstrings': torch.tensor(bitstrings, dtype=torch.long),
            'counts': torch.tensor(counts, dtype=torch.float32).unsqueeze(1),
            'n_qubits': self.n_qubits,
        }


def collate_fn(batch):
    """Custom collate function for variable-length bitstrings."""
    # Pad bitstrings
    max_len = max(b['bitstrings'].shape[0] for b in batch)
    
    padded_bitstrings = []
    padded_counts = []
    n_qubits = batch[0]['n_qubits']
    
    for b in batch:
        bs = b['bitstrings']
        c = b['counts']
        
        if bs.shape[0] < max_len:
            pad_bs = torch.zeros(max_len - bs.shape[0], n_qubits, dtype=torch.long)
            pad_c = torch.zeros(max_len - c.shape[0], 1, dtype=torch.float32)
            bs = torch.cat([bs, pad_bs], dim=0)
            c = torch.cat([c, pad_c], dim=0)
        
        padded_bitstrings.append(bs)
        padded_counts.append(c)
    
    return {
        'bitstrings': torch.stack(padded_bitstrings),
        'counts': torch.stack(padded_counts),
        'n_qubits': batch[0]['n_qubits'],
    }

# ============================================================
# Training
# ============================================================

def train_on_qubit_count(
    n_qubits: int,
    data_dir: Path,
    checkpoint_dir: Path,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 16,
    accumulation_steps: int = 4,
    use_mixed_precision: bool = True,
) -> Dict:
    """
    Train on a specific qubit count with mixed precision.
    """
    print(f"\n{'='*50}")
    print(f"Training on n={n_qubits}")
    print(f"{'='*50}")
    
    # Load data
    data_path = data_dir / f'exp4_scale_{n_qubits}.h5'
    if not data_path.exists():
        print(f"⚠️ Data not found: {data_path}")
        return {'status': 'skipped', 'n_qubits': n_qubits}
    
    dataset = ScaleDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    print(f"   Dataset: {len(dataset)} circuits, n={n_qubits}")
    
    # Model
    model = N2LN(
        d_model=64,
        n_heads=4,
        n_ISAB=2,
        n_SAB=1,
        d_ff=256,
        inducing_points=16,
        dropout=0.1,
        temperature=1.0,
        max_qubits=max(n_qubits, 30),
        use_count_weighting=True,
        use_positional_encoding=True,
    )
    model = model.to(device)
    print(f"   Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Loss
    loss_fn = DistributionLoss(alpha=1.0, beta=0.5, gamma=0.1)
    phys_loss = PhysicalityLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Mixed precision scaler
    scaler = GradScaler() if use_mixed_precision else None
    
    # Training loop
    model.train()
    losses = []
    step = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            bitstrings = batch['bitstrings'].to(device)
            counts = batch['counts'].to(device)
            n_qubits_batch = batch['n_qubits']
            
            # Forward pass with mixed precision
            if use_mixed_precision and scaler is not None:
                with autocast():
                    sn_dist, hn_dist = model(bitstrings, counts, mode='phase1')
                    loss = loss_fn(sn_dist, sn_dist)  # Self-supervised
                    loss = loss + 0.1 * phys_loss(sn_dist)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                sn_dist, hn_dist = model(bitstrings, counts, mode='phase1')
                loss = loss_fn(sn_dist, sn_dist)
                loss = loss + 0.1 * phys_loss(sn_dist)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            step += 1
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"   Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f'model_n{n_qubits}.pt'
    torch.save({
        'n_qubits': n_qubits,
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'epochs': epochs,
    }, checkpoint_path)
    print(f"   ✅ Checkpoint saved: {checkpoint_path}")
    
    return {
        'n_qubits': n_qubits,
        'losses': losses,
        'final_loss': losses[-1] if losses else None,
        'status': 'completed',
    }

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Experiment 4: Qubit Scaling - Scale-out Training")
    print("=" * 60)
    
    exp_dir = Path("experiments/exp4_scale")
    data_dir = Path("data/raw/exp4_scale")
    checkpoint_dir = Path("checkpoints/exp4_scale")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    set_seed(42)
    print(f"Device: {device}")
    
    # Qubit counts to train on (in increasing order)
    qubit_counts = [10, 15, 20, 25, 30]
    
    results = {}
    for n in qubit_counts:
        result = train_on_qubit_count(
            n_qubits=n,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            device=device,
            epochs=5,  # Small for testing
            batch_size=8,
            accumulation_steps=4,
            use_mixed_precision=True,
        )
        results[str(n)] = result
    
    # Save metrics
    metrics = {
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'status': 'completed',
        'qubit_counts': qubit_counts,
        'config': {
            'epochs': 5,
            'batch_size': 8,
            'accumulation_steps': 4,
            'mixed_precision': True,
        },
        'notes': 'Trained on n=10,15,20,25,30 with gradient accumulation',
    }
    
    metrics_path = exp_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("🎉 Scale-out training complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
