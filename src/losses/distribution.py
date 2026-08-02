"""
Distribution loss functions (TDD §4.1).
KL divergence, Total Variation Distance, Chi-squared divergence,
and composite loss with weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    KL divergence: D_KL(target || pred)
    
    Args:
        pred: (B, M) predicted distribution (softmax)
        target: (B, M) target distribution (probabilities)
        eps: Small constant for numerical stability
        reduction: 'mean' or 'sum' or 'none'
    
    Returns:
        KL divergence
    """
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)
    kl = (target * (target / pred).log()).sum(dim=-1)
    
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl


def total_variation_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Total Variation Distance: TVD(pred, target) = 0.5 * sum(|pred - target|)
    
    Args:
        pred: (B, M) predicted distribution
        target: (B, M) target distribution
        reduction: 'mean' or 'sum' or 'none'
    
    Returns:
        TVD
    """
    tvd = 0.5 * (pred - target).abs().sum(dim=-1)
    
    if reduction == 'mean':
        return tvd.mean()
    elif reduction == 'sum':
        return tvd.sum()
    else:
        return tvd


def chi_squared_divergence(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Chi-squared divergence: chi2(pred, target) = sum((pred - target)^2 / target)
    
    Args:
        pred: (B, M) predicted distribution
        target: (B, M) target distribution
        eps: Small constant for numerical stability
        reduction: 'mean' or 'sum' or 'none'
    
    Returns:
        Chi-squared divergence
    """
    target = target.clamp(min=eps)
    chi2 = ((pred - target) ** 2 / (target + eps)).sum(dim=-1)
    
    if reduction == 'mean':
        return chi2.mean()
    elif reduction == 'sum':
        return chi2.sum()
    else:
        return chi2


class CompositeDistributionLoss(nn.Module):
    """
    Composite distribution loss: alpha * KL + beta * TVD + gamma * Chi2
    
    TDD §4.1: Default coefficients: alpha=1.0, beta=0.5, gamma=0.1
    
    Additional optional terms:
    - sharpness: Encourages sharp distributions (negative entropy)
    - entropy_floor: Minimum entropy threshold
    - sharpness_margin: Margin for sharpness loss
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        sharpness: float = 0.0,
        entropy_floor: float = 0.0,
        sharpness_margin: float = 0.02,
        eps: float = 1e-12,
    ):
        """
        Args:
            alpha: Weight for KL divergence
            beta: Weight for TVD
            gamma: Weight for Chi-squared divergence
            sharpness: Weight for sharpness (negative entropy) regularization
            entropy_floor: Minimum entropy threshold
            sharpness_margin: Margin for sharpness loss
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sharpness = sharpness
        self.entropy_floor = entropy_floor
        self.sharpness_margin = sharpness_margin
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Compute the composite loss.
        
        Args:
            pred: (B, M) predicted distribution
            target: (B, M) target distribution
            reduction: 'mean' or 'sum'
        
        Returns:
            Composite loss
        """
        # Primary losses
        kl_loss = kl_divergence(pred, target, eps=self.eps, reduction=reduction)
        tvd_loss = total_variation_distance(pred, target, reduction=reduction)
        chi2_loss = chi_squared_divergence(pred, target, eps=self.eps, reduction=reduction)
        
        loss = self.alpha * kl_loss + self.beta * tvd_loss + self.gamma * chi2_loss
        
        # Sharpness regularization (negative entropy)
        if self.sharpness > 0:
            entropy = -(pred * (pred + self.eps).log()).sum(dim=-1)
            if reduction == 'mean':
                entropy = entropy.mean()
            elif reduction == 'sum':
                entropy = entropy.sum()
            sharpness_loss = F.relu(self.entropy_floor - entropy + self.sharpness_margin)
            loss = loss + self.sharpness * sharpness_loss
        
        return loss
    
    def get_individual_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        Get the individual loss components for logging.
        
        Args:
            pred: (B, M) predicted distribution
            target: (B, M) target distribution
        
        Returns:
            dict: Individual loss values
        """
        with torch.no_grad():
            kl = kl_divergence(pred, target, eps=self.eps, reduction='mean')
            tvd = total_variation_distance(pred, target, reduction='mean')
            chi2 = chi_squared_divergence(pred, target, eps=self.eps, reduction='mean')
            
            entropy = -(pred * (pred + self.eps).log()).sum(dim=-1).mean()
            
            return {
                'kl': kl.item(),
                'tvd': tvd.item(),
                'chi2': chi2.item(),
                'entropy': entropy.item(),
            }


class WeightedCompositeLoss(nn.Module):
    """
    Weighted composite loss with separate weights for SN-D and HN-E heads.
    
    Useful when the two heads have different convergence properties.
    """
    
    def __init__(
        self,
        sn_weights: dict = None,
        hn_weights: dict = None,
        shared_loss: CompositeDistributionLoss = None,
    ):
        """
        Args:
            sn_weights: Dictionary with alpha, beta, gamma for SN-D
            hn_weights: Dictionary with alpha, beta, gamma for HN-E
            shared_loss: Shared CompositeDistributionLoss instance
        """
        super().__init__()
        
        if shared_loss is not None:
            self.sn_loss = shared_loss
            self.hn_loss = shared_loss
        else:
            if sn_weights is None:
                sn_weights = {'alpha': 1.0, 'beta': 0.5, 'gamma': 0.1}
            if hn_weights is None:
                hn_weights = {'alpha': 1.0, 'beta': 0.5, 'gamma': 0.1}
            
            self.sn_loss = CompositeDistributionLoss(**sn_weights)
            self.hn_loss = CompositeDistributionLoss(**hn_weights)
    
    def forward(
        self,
        sn_pred: torch.Tensor,
        sn_target: torch.Tensor,
        hn_pred: torch.Tensor,
        hn_target: torch.Tensor,
    ) -> tuple:
        """
        Compute weighted losses for both heads.
        
        Args:
            sn_pred: SN-D predicted distribution
            sn_target: SN-D target distribution
            hn_pred: HN-E predicted distribution
            hn_target: HN-E target distribution
        
        Returns:
            (sn_loss, hn_loss, total_loss)
        """
        sn_loss = self.sn_loss(sn_pred, sn_target)
        hn_loss = self.hn_loss(hn_pred, hn_target)
        total_loss = sn_loss + hn_loss
        
        return sn_loss, hn_loss, total_loss
