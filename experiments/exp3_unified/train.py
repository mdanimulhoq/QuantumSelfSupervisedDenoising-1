#!/usr/bin/env python
"""
Joint fine-tuning script for Experiment 3: Unified N2LN.
Loads Phase 4 (SN-D) and Phase 5 (HN-E) checkpoints,
unfreezes both heads, and fine-tunes with consistency loss.
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
from src.losses.consistency import CrossStageConsistency
from src.training.trainer import Trainer
from src.training.curriculum import CurriculumController
from src.utils.seeding import set_seed
from src.utils.device import get_device
from src.utils.logging import setup_logger


class UnifiedDataset(Dataset):
    """Combined dataset for unified N2LN training."""
    
    def __init__(self, snd_data_path, hne_data_path, n_qubits=4, max_bitstrings=256):
        self.n_qubits = n_qubits
        self.max_bitstrings = max_bitstrings
        self.snd_data = []
        self.hne_data = []
        self._load_snd_data(snd_data_path)
        self._load_hne_data(hne_data_path)
    
    def _load_snd_data(self, data_path):
        if Path(data_path).exists():
            with h5py.File(data_path, 'r') as f:
                bitstrings = f['bitstrings'][:]
                low_counts = f['low_counts'][:]
                high_counts = f['high_counts'][:]
                for i in range(len(bitstrings)):
                    low_dict = self._counts_to_dict(low_counts[i], bitstrings[i])
                    high_dict = self._counts_to_dict(high_counts[i], bitstrings[i])
                    self.snd_data.append({
                        'type': 'snd',
                        'low_counts': low_dict,
                        'high_counts': high_dict,
                        'bitstrings': bitstrings[i],
                    })
    
    def _load_hne_data(self, data_path):
        if Path(data_path).exists():
            with h5py.File(data_path, 'r') as f:
                for key in f.keys():
                    if key.startswith('scale_'):
                        scale = f[key].attrs['scale']
                        counts_json = f[key]['counts'][:]
                        for counts_str in counts_json:
                            counts = json.loads(counts_str)
                            self.hne_data.append({
                                'type': 'hne',
                                'counts': counts,
                                'scale': float(scale),
                            })
    
    def _counts_to_dict(self, counts, bitstrings):
        mask = counts > 0
        if not mask.any():
            return {}
        return {str(bs): int(c) for bs, c in zip(bitstrings[mask], counts[mask]) if c > 0}
    
    def __len__(self):
        return len(self.snd_data) + len(self.hne_data)
    
    def __getitem__(self, idx):
        if idx < len(self.snd_data):
            sample = self.snd_data[idx]
            bitstrings, counts = self._dict_to_tensors(sample['low_counts'])
            target_bitstrings, target_counts = self._dict_to_tensors(sample['high_counts'])
            return {
                'bitstrings': bitstrings,
                'counts': counts,
                'sn_target': target_counts,
                'hn_target': target_counts,
                'data_type': 'snd',
                'scale': torch.tensor(1.0, dtype=torch.float32),
            }
        else:
            idx2 = idx - len(self.snd_data)
            sample = self.hne_data[idx2]
            bitstrings, counts = self._dict_to_tensors(sample['counts'])
            return {
                'bitstrings': bitstrings,
                'counts': counts,
                'sn_target': counts,
                'hn_target': counts,
                'data_type': 'hne',
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


def main():
    print("=" * 60)
    print("Experiment 3: Unified N2LN - Joint Fine-Tuning")
    print("=" * 60)
    
    exp_dir = Path("experiments/exp3_unified")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path("data/raw")
    checkpoint_dir = Path("checkpoints/exp3_unified")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    set_seed(42)
    print(f"Device: {device}")
    
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
    
    phase4_path = Path("checkpoints/exp1_snd/best_model.pt")
    if phase4_path.exists():
        print(f"Loading Phase 4 checkpoint: {phase4_path}")
        checkpoint = torch.load(phase4_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print(f"Phase 4 checkpoint not found: {phase4_path}")
    
    phase5_path = Path("checkpoints/exp2_hne/best_model.pt")
    if phase5_path.exists():
        print(f"Loading Phase 5 checkpoint: {phase5_path}")
        checkpoint = torch.load(phase5_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print(f"Phase 5 checkpoint not found: {phase5_path}")
    
    for param in model.parameters():
        param.requires_grad = True
    print("All heads unfrozen for joint fine-tuning")
    
    model = model.to(device)
    
    dataset = UnifiedDataset(
        data_dir / 'exp1_snd' / 'exp1_snd_train.h5',
        data_dir / 'exp2_hne' / 'exp2_hne_train.h5',
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Dataset: {len(dataset)} samples")
    
    loss_fns = {
        'snd': DistributionLoss(alpha=1.0, beta=0.5, gamma=0.1),
        'hne': DistributionLoss(alpha=1.0, beta=0.5, gamma=0.1),
        'physicality': PhysicalityLoss(),
        'consistency': CrossStageConsistency(
            tvd_weight=0.5,
            kl_weight=0.5,
            consistency_weight=0.3,
        ),
    }
    
    config = {
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
        'batch_size': 32,
        'gradient_clip': 1.0,
        'log_interval': 10,
        'seed': 42,
        'wandb_project': 'n2ln-qem-unified',
        'use_wandb': False,
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
    
    print("\n" + "=" * 50)
    print("Phase 3: Joint Fine-Tuning with Consistency Loss")
    print("=" * 50)
    
    trainer.train(
        num_epochs=50,
        phase='phase3',
        save_every=10,
        early_stopping_patience=15,
    )
    
    final_checkpoint = checkpoint_dir / 'best_model.pt'
    trainer.save_checkpoint(str(final_checkpoint))
    print(f"Model saved: {final_checkpoint}")
    
    metrics = {
        'train_loss': trainer.best_val_loss,
        'epochs': trainer.current_epoch,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'phase': 'phase3',
        'consistency_enabled': True,
        'status': 'completed',
    }
    
    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {exp_dir / 'metrics.json'}")
    
    print("\nUnified N2LN Fine-Tuning Complete!")

if __name__ == '__main__':
    main()
