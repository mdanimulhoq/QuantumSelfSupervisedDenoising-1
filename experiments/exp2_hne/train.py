#!/usr/bin/env python
"""
Training script for Experiment 2: HN-E (Hardware-Noise Extrapolation).
Loads Phase 4 checkpoint, freezes SN-D, trains HN-E only.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.n2ln import N2LN
from src.losses.distribution import DistributionLoss
from src.losses.physicality import PhysicalityLoss
from src.training.trainer import Trainer
from src.utils.seeding import set_seed
from src.utils.device import get_device
from src.utils.logging import setup_logger

# ============================================================
# Dataset
# ============================================================

class HNEDataset(Dataset):
    """Dataset for HN-E training with noise scales."""
    
    def __init__(self, data_path, n_qubits=4, max_bitstrings=256):
        self.data_path = Path(data_path)
        self.n_qubits = n_qubits
        self.max_bitstrings = max_bitstrings
        self._load_data()
    
    def _load_data(self):
        with h5py.File(self.data_path, 'r') as f:
            # Get noise scales
            self.noise_scales = []
            self.data = []
            
            for key in f.keys():
                if key.startswith('scale_'):
                    scale = f[key].attrs['scale']
                    self.noise_scales.append(scale)
                    
                    counts_json = f[key]['counts'][:]
                    circuit_ids = f[key]['circuit_ids'][:]
                    
                    for idx, (counts_str, cid) in enumerate(zip(counts_json, circuit_ids)):
                        counts = json.loads(counts_str)
                        self.data.append({
                            'counts': counts,
                            'circuit_id': int(cid),
                            'scale': float(scale),
                            'idx': idx,
                        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Convert counts to tensors
        bitstrings, counts = self._dict_to_tensors(sample['counts'])
        
        # Target is the same distribution (HN-E learns to denoise)
        # For now, use same as input (will be refined)
        target = counts.clone()
        
        return {
            'bitstrings': bitstrings,
            'counts': counts,
            'hn_target': target,
            'sn_target': target,  # Not used for HN-E
            'scale': torch.tensor(sample['scale'], dtype=torch.float32),
        }
    
    def _dict_to_tensors(self, counts_dict):
        if not counts_dict:
            return torch.zeros(1, self.n_qubits, dtype=torch.long), torch.zeros(1, 1)
        
        items = sorted(counts_dict.items(), key=lambda x: -x[1])
        items = items[:self.max_bitstrings]
        
        bitstrings = []
        counts = []
        total = sum(c for _, c in items)
        
        for bs, c in items:
            bs_tensor = [int(b) for b in bs.zfill(self.n_qubits)]
            bitstrings.append(bs_tensor)
            counts.append(c / total)
        
        return torch.tensor(bitstrings, dtype=torch.long), torch.tensor(counts, dtype=torch.float32).unsqueeze(1)

# ============================================================
# Main Training
# ============================================================

def main():
    print("="*60)
    print("📊 Experiment 2: HN-E Head Training")
    print("="*60)
    
    # Paths
    exp_dir = Path("experiments/exp2_hne")
    data_dir = Path("data/raw/exp2_hne")
    checkpoint_dir = Path("checkpoints/exp2_hne")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup
    device = get_device()
    set_seed(42)
    print(f"Device: {device}")
    
    # Load Phase 4 checkpoint
    phase4_checkpoint = Path("checkpoints/exp1_snd/best_model.pt")
    if not phase4_checkpoint.exists():
        print(f"⚠️ Phase 4 checkpoint not found: {phase4_checkpoint}")
        print("   Creating new model...")
        model = N2LN(
            d_model=64,
            n_heads=4,
            n_ISAB=2,
            n_SAB=1,
            d_ff=256,
            inducing_points=16,
            dropout=0.1,
            temperature=1.0,
            max_qubits=20,
        )
    else:
        print(f"✅ Loading Phase 4 checkpoint: {phase4_checkpoint}")
        checkpoint = torch.load(phase4_checkpoint, map_location='cpu')
        model = N2LN(
            d_model=64,
            n_heads=4,
            n_ISAB=2,
            n_SAB=1,
            d_ff=256,
            inducing_points=16,
            dropout=0.1,
            temperature=1.0,
            max_qubits=20,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze SN-D head
        for param in model.sn_head.parameters():
            param.requires_grad = False
        print("✅ SN-D head frozen")
    
    model = model.to(device)
    
    # Load data
    train_dataset = HNEDataset(data_dir / 'exp2_hne_train.h5')
    val_dataset = HNEDataset(data_dir / 'exp2_hne_val.h5')
    test_dataset = HNEDataset(data_dir / 'exp2_hne_test.h5')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Loss functions
    loss_fns = {
        'snd': DistributionLoss(alpha=1.0, beta=0.5, gamma=0.1),
        'hne': DistributionLoss(alpha=1.0, beta=0.5, gamma=0.1),
        'physicality': PhysicalityLoss(),
        'consistency': None,  # Not used in Phase 2
    }
    
    # Trainer
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 32,
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
        log_dir=exp_dir / 'logs',
        use_wandb=False,
    )
    
    # Train HN-E (Phase 2)
    print("\n" + "="*50)
    print("Phase 2: HN-E Training")
    print("="*50)
    
    trainer.train(
        num_epochs=50,
        phase='phase2',
        save_every=10,
        early_stopping_patience=10,
    )
    
    # Save final checkpoint
    final_checkpoint = checkpoint_dir / 'best_model.pt'
    trainer.save_checkpoint(str(final_checkpoint))
    print(f"✅ Model saved: {final_checkpoint}")
    
    # Save metrics
    metrics = {
        'train_loss': trainer.best_val_loss,
        'epochs': trainer.current_epoch,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'status': 'completed',
    }
    
    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved: {exp_dir / 'metrics.json'}")
    
    print("\n🎉 HN-E Training Complete!")

if __name__ == '__main__':
    main()
