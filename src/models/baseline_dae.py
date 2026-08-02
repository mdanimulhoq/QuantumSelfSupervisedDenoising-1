"""
Baseline Denoising Autoencoder (TDD §3.5).
MLP encoder/decoder on full 2^n dimensional distribution vectors.
Not permutation-invariant. Fixed input size = 2^n.
Used for small qubit counts (n ≤ 8) as a baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DistributionAutoencoder(nn.Module):
    """
    Denoising Autoencoder (DAE) baseline.
    
    Architecture:
    1. Encoder: MLP compressing 2^n-dim input to bottleneck
    2. Decoder: Two separate MLP heads (SN-D and HN-E) from bottleneck to 2^n-dim output
    3. Output: Both heads produce distributions via softmax
    
    This is a fixed-size architecture that operates on full distribution vectors.
    It does NOT support variable qubit counts or permutation invariance.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        bottleneck_dim: int = 32,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of the input distribution (2^n)
            hidden_dims: List of hidden layer dimensions
            bottleneck_dim: Dimension of the bottleneck (latent) representation
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # ----- Encoder -----
        encoder_layers = []
        d = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            d = h
        # Final bottleneck layer
        encoder_layers.append(nn.Linear(d, bottleneck_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ----- Decoder (SN-D head) -----
        # Rebuild hidden dimensions in reverse
        decoder_hidden = hidden_dims[::-1]
        
        sn_decoder_layers = []
        d = bottleneck_dim
        for h in decoder_hidden:
            sn_decoder_layers.extend([
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            d = h
        # Final layer to output dimension
        sn_decoder_layers.append(nn.Linear(d, input_dim))
        self.sn_decoder = nn.Sequential(*sn_decoder_layers)
        
        # ----- Decoder (HN-E head) -----
        hn_decoder_layers = []
        d = bottleneck_dim
        for h in decoder_hidden:
            hn_decoder_layers.extend([
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            d = h
        hn_decoder_layers.append(nn.Linear(d, input_dim))
        self.hn_decoder = nn.Sequential(*hn_decoder_layers)
        
        # Temperature parameter for softmax (learnable)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.temperature_floor = 0.3
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: (B, input_dim) distribution vector (normalized, sums to 1)
        
        Returns:
            sn_dist: (B, input_dim) SN-D distribution (softmax)
            hn_dist: (B, input_dim) HN-E distribution (softmax)
        """
        # Encoder
        z = self.encoder(x)  # (B, bottleneck_dim)
        
        # SN-D decoder
        sn_logits = self.sn_decoder(z)  # (B, input_dim)
        
        # HN-E decoder
        hn_logits = self.hn_decoder(z)  # (B, input_dim)
        
        # Temperature for softmax
        temp = torch.clamp(self.temp, min=self.temperature_floor)
        
        # Apply softmax to get valid probability distributions
        sn_dist = F.softmax(sn_logits / temp, dim=-1)
        hn_dist = F.softmax(hn_logits / temp, dim=-1)
        
        return sn_dist, hn_dist
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to bottleneck representation.
        
        Args:
            x: (B, input_dim) distribution vector
        
        Returns:
            (B, bottleneck_dim) latent representation
        """
        return self.encoder(x)
    
    def decode_sn(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to SN-D distribution.
        
        Args:
            z: (B, bottleneck_dim) latent representation
        
        Returns:
            (B, input_dim) SN-D distribution
        """
        temp = torch.clamp(self.temp, min=self.temperature_floor)
        logits = self.sn_decoder(z)
        return F.softmax(logits / temp, dim=-1)
    
    def decode_hn(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to HN-E distribution.
        
        Args:
            z: (B, bottleneck_dim) latent representation
        
        Returns:
            (B, input_dim) HN-E distribution
        """
        temp = torch.clamp(self.temp, min=self.temperature_floor)
        logits = self.hn_decoder(z)
        return F.softmax(logits / temp, dim=-1)


class DistributionAutoencoderWithSupport(nn.Module):
    """
    DAE that operates on a sparse support (observed bitstrings only).
    
    This is a variant that can handle variable-sized supports by projecting
    to a fixed dimension and then back.
    """
    
    def __init__(
        self,
        max_support: int = 256,
        hidden_dims: list = None,
        bottleneck_dim: int = 32,
        dropout: float = 0.1,
    ):
        """
        Args:
            max_support: Maximum support size (M)
            hidden_dims: List of hidden layer dimensions
            bottleneck_dim: Dimension of the bottleneck (latent) representation
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.max_support = max_support
        self.bottleneck_dim = bottleneck_dim
        
        # Encoder: from max_support to bottleneck
        encoder_layers = []
        d = max_support
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            d = h
        encoder_layers.append(nn.Linear(d, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder SN-D: from bottleneck to max_support
        decoder_hidden = hidden_dims[::-1]
        sn_decoder_layers = []
        d = bottleneck_dim
        for h in decoder_hidden:
            sn_decoder_layers.extend([
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            d = h
        sn_decoder_layers.append(nn.Linear(d, max_support))
        self.sn_decoder = nn.Sequential(*sn_decoder_layers)
        
        # Decoder HN-E: from bottleneck to max_support
        hn_decoder_layers = []
        d = bottleneck_dim
        for h in decoder_hidden:
            hn_decoder_layers.extend([
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            d = h
        hn_decoder_layers.append(nn.Linear(d, max_support))
        self.hn_decoder = nn.Sequential(*hn_decoder_layers)
        
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.temperature_floor = 0.3
    
    def forward(
        self,
        x: torch.Tensor,
        support_mask: torch.Tensor = None,
    ) -> tuple:
        """
        Forward pass with optional support mask.
        
        Args:
            x: (B, M) distribution vector (padded to max_support)
            support_mask: (B, M) boolean mask for valid positions
        
        Returns:
            sn_dist: (B, M) SN-D distribution (softmax)
            hn_dist: (B, M) HN-E distribution (softmax)
        """
        temp = torch.clamp(self.temp, min=self.temperature_floor)
        
        z = self.encoder(x)  # (B, bottleneck_dim)
        
        sn_logits = self.sn_decoder(z)  # (B, max_support)
        hn_logits = self.hn_decoder(z)  # (B, max_support)
        
        # Apply support mask if provided
        if support_mask is not None:
            sn_logits = sn_logits.masked_fill(~support_mask, -1e9)
            hn_logits = hn_logits.masked_fill(~support_mask, -1e9)
        
        sn_dist = F.softmax(sn_logits / temp, dim=-1)
        hn_dist = F.softmax(hn_logits / temp, dim=-1)
        
        return sn_dist, hn_dist
