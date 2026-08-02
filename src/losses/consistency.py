"""
Cross-stage consistency loss (TDD §4.1).
Encourages SN-D and HN-E heads to produce consistent outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.distribution import total_variation_distance, kl_divergence


class ConsistencyLoss(nn.Module):
    """
    Cross-stage consistency loss.
    
    L_consist = TVD(f_HN(f_SN(x_low)), f_HN(x_high))
    """
    
    def __init__(self, method: str = 'tvd', weight: float = 0.3):
        super().__init__()
        self.method = method
        self.weight = weight
    
    def forward(self, hn_on_sn: torch.Tensor, hn_on_raw: torch.Tensor, 
                reduction: str = 'mean') -> torch.Tensor:
        if self.method == 'tvd':
            loss = total_variation_distance(hn_on_sn, hn_on_raw, reduction=reduction)
        elif self.method == 'kl':
            loss = kl_divergence(hn_on_sn, hn_on_raw, reduction=reduction)
        elif self.method == 'both':
            tvd = total_variation_distance(hn_on_sn, hn_on_raw, reduction=reduction)
            kl = kl_divergence(hn_on_sn, hn_on_raw, reduction=reduction)
            loss = 0.5 * tvd + 0.5 * kl
        else:
            raise ValueError(f"Unknown consistency method: {self.method}")
        return self.weight * loss


class CrossStageConsistency(nn.Module):
    """Full cross-stage consistency with both forward and backward directions."""
    
    def __init__(self, tvd_weight: float = 0.5, kl_weight: float = 0.5, 
                 consistency_weight: float = 0.3):
        super().__init__()
        self.tvd_weight = tvd_weight
        self.kl_weight = kl_weight
        self.consistency_weight = consistency_weight
    
    def forward(self, sn_dist: torch.Tensor, hn_dist: torch.Tensor) -> torch.Tensor:
        # Symmetric TVD
        tvd_loss = total_variation_distance(sn_dist, hn_dist, reduction='mean')
        
        # Symmetric KL divergence
        eps = 1e-12
        sn_clamped = sn_dist.clamp(min=eps)
        hn_clamped = hn_dist.clamp(min=eps)
        
        kl_sn_to_hn = (sn_clamped * (sn_clamped / hn_clamped).log()).sum(dim=-1).mean()
        kl_hn_to_sn = (hn_clamped * (hn_clamped / sn_clamped).log()).sum(dim=-1).mean()
        kl_loss = 0.5 * (kl_sn_to_hn + kl_hn_to_sn)
        
        loss = self.tvd_weight * tvd_loss + self.kl_weight * kl_loss
        loss = self.consistency_weight * loss
        return loss
    
    def get_individual_losses(self, sn_dist: torch.Tensor, hn_dist: torch.Tensor) -> dict:
        with torch.no_grad():
            tvd = total_variation_distance(sn_dist, hn_dist, reduction='mean').item()
            
            eps = 1e-12
            sn_clamped = sn_dist.clamp(min=eps)
            hn_clamped = hn_dist.clamp(min=eps)
            
            kl_sn_to_hn = (sn_clamped * (sn_clamped / hn_clamped).log()).sum(dim=-1).mean().item()
            kl_hn_to_sn = (hn_clamped * (hn_clamped / sn_clamped).log()).sum(dim=-1).mean().item()
            kl_avg = 0.5 * (kl_sn_to_hn + kl_hn_to_sn)
            
            return {
                'tvd_consistency': tvd,
                'kl_consistency': kl_avg,
                'kl_sn_to_hn': kl_sn_to_hn,
                'kl_hn_to_sn': kl_hn_to_sn,
            }


class JointConsistencyLoss(nn.Module):
    """Joint consistency loss combining multiple terms."""
    
    def __init__(self, cross_stage_weight: float = 0.3, temporal_weight: float = 0.0,
                 use_kl: bool = True, use_tvd: bool = True):
        super().__init__()
        self.cross_stage_weight = cross_stage_weight
        self.temporal_weight = temporal_weight
        self.use_kl = use_kl
        self.use_tvd = use_tvd
    
    def forward(self, sn_dist: torch.Tensor, hn_dist: torch.Tensor,
                prev_sn_dist: torch.Tensor = None, prev_hn_dist: torch.Tensor = None) -> torch.Tensor:
        loss = torch.tensor(0.0, device=sn_dist.device)
        
        # Cross-stage consistency
        if self.use_kl or self.use_tvd:
            eps = 1e-12
            sn_clamped = sn_dist.clamp(min=eps)
            hn_clamped = hn_dist.clamp(min=eps)
            
            if self.use_kl and self.use_tvd:
                tvd = total_variation_distance(sn_dist, hn_dist, reduction='mean')
                kl = 0.5 * ((sn_clamped * (sn_clamped / hn_clamped).log()).sum(dim=-1).mean()
                           + (hn_clamped * (hn_clamped / sn_clamped).log()).sum(dim=-1).mean())
                loss = loss + self.cross_stage_weight * (0.5 * tvd + 0.5 * kl)
            elif self.use_tvd:
                loss = loss + self.cross_stage_weight * total_variation_distance(
                    sn_dist, hn_dist, reduction='mean')
            elif self.use_kl:
                loss = loss + self.cross_stage_weight * ((sn_clamped * (sn_clamped / hn_clamped).log())
                                                         .sum(dim=-1).mean())
        
        # Temporal consistency
        if self.temporal_weight > 0 and prev_sn_dist is not None:
            sn_temp = total_variation_distance(sn_dist, prev_sn_dist, reduction='mean')
            loss = loss + self.temporal_weight * sn_temp
        
        if self.temporal_weight > 0 and prev_hn_dist is not None:
            hn_temp = total_variation_distance(hn_dist, prev_hn_dist, reduction='mean')
            loss = loss + self.temporal_weight * hn_temp
        
        return loss
