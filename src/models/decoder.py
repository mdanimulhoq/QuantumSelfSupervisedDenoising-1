"""
Dual-Head Decoder (TDD §3.4)
SN-D head + HN-E head with softmax output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DualHeadDecoder(nn.Module):
    """
    Dual-head decoder with SN-D and HN-E heads.
    """
    
    def __init__(
        self,
        d_model: int = 64,
        max_qubits: int = 20,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        dropout: float = 0.1,
        use_mlp_scorer: bool = True,
        temperature_floor: float = 0.3,
    ):
        super().__init__()
        self.max_qubits = max_qubits
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.temperature_floor = temperature_floor
        self.use_mlp_scorer = use_mlp_scorer
        
        # Shared scoring network
        self.scorer = nn.Sequential(
            nn.Linear(d_model, 1),
        ) if not use_mlp_scorer else nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # SN-D head
        self.sn_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        
        # HN-E head
        self.hn_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        
        # Bitstring encoder reference (set by model wrapper)
        self.bitstring_encoder = None
    
    def set_bitstring_encoder(self, encoder):
        """Set the bitstring encoder reference."""
        self.bitstring_encoder = encoder
    
    def _compute_logits(
        self,
        z: torch.Tensor,
        bitstrings: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logits for both heads.
        
        Args:
            z: (B, d_model) global latent
            bitstrings: (B, M, n_qubits)
            mask: (B, M) boolean mask for valid positions
        
        Returns:
            sn_logits: (B, M) logits for SN-D head
            hn_logits: (B, M) logits for HN-E head
        """
        # Get embeddings from encoder (now returns only embeddings)
        if self.bitstring_encoder is not None:
            embeddings = self.bitstring_encoder(bitstrings)  # (B, M, d_model)
        else:
            embeddings = bitstrings.float()
        
        # If mask not provided, assume all positions are valid
        if mask is None:
            mask = torch.ones(bitstrings.shape[0], bitstrings.shape[1], dtype=torch.bool, device=bitstrings.device)
        
        # SN-D head
        sn_z = self.sn_head(z)  # (B, d_model)
        sn_logits = self.scorer(sn_z.unsqueeze(1) * embeddings).squeeze(-1)  # (B, M)
        sn_logits = sn_logits.masked_fill(~mask, -1e9)
        
        # HN-E head
        hn_z = self.hn_head(z)  # (B, d_model)
        hn_logits = self.scorer(hn_z.unsqueeze(1) * embeddings).squeeze(-1)  # (B, M)
        hn_logits = hn_logits.masked_fill(~mask, -1e9)
        
        return sn_logits, hn_logits
    
    def forward(
        self,
        z: torch.Tensor,
        bitstrings: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            z: (B, d_model) global latent
            bitstrings: (B, M, n_qubits)
            mask: (B, M) boolean mask for valid positions
        
        Returns:
            sn_dist: (B, M) SN-D distribution
            hn_dist: (B, M) HN-E distribution
        """
        # Compute logits
        sn_logits, hn_logits = self._compute_logits(z, bitstrings, mask)
        
        # Temperature scaling
        temp = self.temperature.clamp(min=self.temperature_floor)
        sn_dist = F.softmax(sn_logits / temp, dim=-1)
        hn_dist = F.softmax(hn_logits / temp, dim=-1)
        
        return sn_dist, hn_dist
