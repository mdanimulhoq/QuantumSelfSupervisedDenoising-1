"""
Physicality regularization (TDD §4.1).
Non-negativity + normalization penalty for valid probability distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalityLoss(nn.Module):
    """
    Physicality regularization loss.
    
    Penalizes:
    1. Non-negativity: negative probabilities
    2. Normalization: sum of probabilities deviating from 1
    
    With softmax outputs, these are automatically satisfied, but the
    regularization helps during early training when the temperature
    parameter is unstable.
    """
    
    def __init__(
        self,
        nonneg_weight: float = 1.0,
        norm_weight: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.nonneg_weight = nonneg_weight
        self.norm_weight = norm_weight
        self.eps = eps
    
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Compute physicality loss for a distribution.
        
        Args:
            dist: (B, M) probability distribution (should sum to ~1)
        
        Returns:
            Scalar physicality loss
        """
        # Non-negativity penalty
        neg_penalty = F.relu(-dist).pow(2).sum(dim=-1).mean()
        
        # Normalization penalty
        sum_dist = dist.sum(dim=-1)
        norm_penalty = (sum_dist - 1.0).pow(2).mean()
        
        loss = self.nonneg_weight * neg_penalty + self.norm_weight * norm_penalty
        return loss
    
    def get_individual_losses(self, dist: torch.Tensor) -> dict:
        """Get individual loss components for logging."""
        with torch.no_grad():
            neg_penalty = F.relu(-dist).pow(2).sum(dim=-1).mean().item()
            sum_dist = dist.sum(dim=-1).mean().item()
            norm_penalty = (dist.sum(dim=-1) - 1.0).pow(2).mean().item()
            
            return {
                'neg_penalty': neg_penalty,
                'norm_penalty': norm_penalty,
                'sum_dist': sum_dist,
            }


class EntropyRegularization(nn.Module):
    """Entropy regularization to prevent over-sharpening."""
    
    def __init__(self, target_entropy: float = 0.5, weight: float = 0.01, eps: float = 1e-12):
        super().__init__()
        self.target_entropy = target_entropy
        self.weight = weight
        self.eps = eps
    
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist_clamped = dist.clamp(min=self.eps)
        entropy = -(dist_clamped * dist_clamped.log()).sum(dim=-1).mean()
        loss = F.relu(self.target_entropy - entropy)
        return self.weight * loss
