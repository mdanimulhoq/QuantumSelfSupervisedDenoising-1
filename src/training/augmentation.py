"""
Noise-aware data augmentation (TDD §4.2).
Bootstrap resampling, noise level interpolation, circuit perturbation.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    bootstrap_resampling: bool = True
    noise_level_interpolation: bool = True
    circuit_perturbation: bool = False
    num_augmentations: int = 1
    noise_levels: List[float] = None
    perturbation_scale: float = 0.1


class NoiseAwareAugmentation:
    """
    Noise-aware data augmentation for N2LN-QEM training.
    
    Methods:
    1. Bootstrap resampling: Generate multiple low-shot samples from high-shot
    2. Noise level interpolation: Interpolate between noise levels
    3. Circuit parameter perturbation: Perturb circuit parameters slightly
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            config: Augmentation configuration
            seed: Random seed for reproducibility
        """
        self.config = config or AugmentationConfig()
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def bootstrap_resample(
        self,
        counts: Dict[str, int],
        n_qubits: int,
        low_shots: int = 100,
        num_samples: int = 1,
    ) -> List[Dict[str, int]]:
        """
        Generate bootstrap resampled low-shot counts.
        
        Args:
            counts: Original counts dictionary
            n_qubits: Number of qubits
            low_shots: Number of shots for resampled data
            num_samples: Number of bootstrap samples
        
        Returns:
            List of resampled counts dictionaries
        """
        if not self.config.bootstrap_resampling:
            return [counts.copy()]
        
        # Convert to probability vector
        total_shots = sum(counts.values())
        probs = {bs: c / total_shots for bs, c in counts.items()}
        
        # Prepare for sampling
        bitstrings = list(probs.keys())
        probs_list = [probs[bs] for bs in bitstrings]
        
        samples = []
        for _ in range(num_samples):
            # Sample with replacement
            sampled_indices = np.random.choice(
                len(bitstrings),
                size=low_shots,
                p=probs_list,
                replace=True,
            )
            
            # Count occurrences
            new_counts = {}
            for idx in sampled_indices:
                bs = bitstrings[idx]
                new_counts[bs] = new_counts.get(bs, 0) + 1
            
            samples.append(new_counts)
        
        return samples
    
    def interpolate_noise_levels(
        self,
        counts_low: Dict[str, int],
        counts_high: Dict[str, int],
        alpha: float = 0.5,
    ) -> Dict[str, float]:
        """
        Interpolate between two noise levels.
        
        Args:
            counts_low: Counts at lower noise level
            counts_high: Counts at higher noise level
            alpha: Interpolation factor (0 = low, 1 = high)
        
        Returns:
            Interpolated probabilities
        """
        if not self.config.noise_level_interpolation:
            alpha = 0.0
        
        # Normalize
        total_low = sum(counts_low.values())
        total_high = sum(counts_high.values())
        
        # Get union of bitstrings
        all_bitstrings = set(counts_low.keys()) | set(counts_high.keys())
        
        # Interpolate probabilities
        interp_counts = {}
        for bs in all_bitstrings:
            p_low = counts_low.get(bs, 0) / total_low if total_low > 0 else 0
            p_high = counts_high.get(bs, 0) / total_high if total_high > 0 else 0
            interp_counts[bs] = (1 - alpha) * p_low + alpha * p_high
        
        return interp_counts
    
    def perturb_circuit_parameters(
        self,
        params: np.ndarray,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        Perturb circuit parameters slightly.
        
        Args:
            params: Original parameters
            scale: Perturbation scale (default: from config)
        
        Returns:
            Perturbed parameters
        """
        if not self.config.circuit_perturbation:
            return params.copy()
        
        scale = scale or self.config.perturbation_scale
        noise = np.random.normal(0, scale, size=params.shape)
        return params + noise
    
    def augment_batch(
        self,
        bitstrings: torch.Tensor,
        counts: torch.Tensor,
        targets: torch.Tensor,
        num_augmentations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Augment a batch of data.
        
        Args:
            bitstrings: (B, M, n) bitstring tensors
            counts: (B, M, 1) count tensors
            targets: (B, M) target tensors
            num_augmentations: Number of augmentations per sample
        
        Returns:
            Tuple of augmented bitstrings, counts, targets
        """
        num_aug = num_augmentations or self.config.num_augmentations
        
        if num_aug <= 1:
            return bitstrings, counts, targets
        
        # Simple augmentation: add noise to counts
        augmented_bitstrings = []
        augmented_counts = []
        augmented_targets = []
        
        for i in range(bitstrings.shape[0]):
            # Original
            augmented_bitstrings.append(bitstrings[i])
            augmented_counts.append(counts[i])
            augmented_targets.append(targets[i])
            
            # Augmentations
            for _ in range(num_aug - 1):
                # Add Poisson noise to counts
                count_noise = torch.poisson(counts[i] * 100) / 100
                augmented_counts.append(count_noise)
                augmented_bitstrings.append(bitstrings[i])
                augmented_targets.append(targets[i])
        
        return (
            torch.stack(augmented_bitstrings),
            torch.stack(augmented_counts),
            torch.stack(augmented_targets),
        )


def get_default_augmentation_config() -> AugmentationConfig:
    """Get default augmentation configuration."""
    return AugmentationConfig(
        bootstrap_resampling=True,
        noise_level_interpolation=True,
        circuit_perturbation=False,
        num_augmentations=3,
        perturbation_scale=0.1,
    )
