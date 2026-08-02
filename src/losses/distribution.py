"""
Distribution loss functions (TDD §4.1).
KL divergence, Total Variation Distance, and Chi-squared divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Union


def kl_divergence(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute KL divergence: KL(pred || target)
    
    D_KL(pred || target) = sum(pred * log(pred / target))
    
    Args:
        pred: (B, M) predicted distribution
        target: (B, M) target distribution
        reduction: 'mean', 'sum', or 'none'
        eps: Small constant for numerical stability
    
    Returns:
        KL divergence
    """
    pred_clamped = pred.clamp(min=eps)
    target_clamped = target.clamp(min=eps)
    kl = (pred_clamped * (pred_clamped / target_clamped).log()).sum(dim=-1)
    
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl


def total_variation_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Compute Total Variation Distance: 0.5 * sum(|pred - target|)
    
    Args:
        pred: (B, M) predicted distribution
        target: (B, M) target distribution
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        TVD (in [0, 1])
    """
    tvd = 0.5 * (pred - target).abs().sum(dim=-1)
    
    if reduction == "mean":
        return tvd.mean()
    elif reduction == "sum":
        return tvd.sum()
    else:
        return tvd


def chi2_divergence(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute Chi-squared divergence: sum((pred - target)^2 / target)
    
    Penalizes large relative errors on rare events.
    
    Args:
        pred: (B, M) predicted distribution
        target: (B, M) target distribution
        reduction: 'mean', 'sum', or 'none'
        eps: Small constant for numerical stability
    
    Returns:
        Chi-squared divergence
    """
    target_clamped = target.clamp(min=eps)
    chi2 = ((pred - target_clamped).pow(2) / target_clamped).sum(dim=-1)
    
    if reduction == "mean":
        return chi2.mean()
    elif reduction == "sum":
        return chi2.sum()
    else:
        return chi2


class DistributionLoss(nn.Module):
    """
    Composite distribution loss with configurable weights.
    
    L = alpha * KL + beta * TVD + gamma * Chi2
    
    Args:
        alpha: Weight for KL divergence
        beta: Weight for TVD
        gamma: Weight for Chi-squared divergence
        eps: Small constant for numerical stability
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> torch.Tensor:
        """
        Compute composite distribution loss.
        
        Args:
            pred: (B, M) predicted distribution
            target: (B, M) target distribution
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            Composite loss
        """
        kl_loss = kl_divergence(pred, target, reduction=reduction, eps=self.eps)
        tvd_loss = total_variation_distance(pred, target, reduction=reduction)
        chi2_loss = chi2_divergence(pred, target, reduction=reduction, eps=self.eps)
        
        loss = self.alpha * kl_loss + self.beta * tvd_loss + self.gamma * chi2_loss
        return loss
    
    def get_individual_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        Get individual loss components for logging.
        
        Args:
            pred: (B, M) predicted distribution
            target: (B, M) target distribution
        
        Returns:
            dict: Individual loss values
        """
        with torch.no_grad():
            kl = kl_divergence(pred, target, reduction="mean", eps=self.eps)
            tvd = total_variation_distance(pred, target, reduction="mean")
            chi2 = chi2_divergence(pred, target, reduction="mean", eps=self.eps)
            
            return {
                "kl_loss": kl.item(),
                "tvd_loss": tvd.item(),
                "chi2_loss": chi2.item(),
                "composite_loss": (self.alpha * kl + self.beta * tvd + self.gamma * chi2).item(),
            }


class SNDLoss(DistributionLoss):
    """
    Loss for Shot-Noise Denoising (SN-D) head.
    Uses composite distribution loss with default weights.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        eps: float = 1e-12,
    ):
        super().__init__(alpha=alpha, beta=beta, gamma=gamma, eps=eps)


class HNEELoss(DistributionLoss):
    """
    Loss for Hardware-Noise Extrapolation (HN-E) head.
    Uses composite distribution loss with default weights.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        eps: float = 1e-12,
    ):
        super().__init__(alpha=alpha, beta=beta, gamma=gamma, eps=eps)
