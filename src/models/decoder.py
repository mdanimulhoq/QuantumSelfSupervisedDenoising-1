"""
Dual-Head Decoder (TDD §3.4)
SN-D head + HN-E head, both producing valid probability distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DualHeadDecoder(nn.Module):
    """
    Dual-Head Decoder with SN-D and HN-E heads.
    
    Architecture:
    1. SN-D head: projects global latent to distribution over bitstrings
    2. HN-E head: projects global latent to distribution over bitstrings
    3. Both heads use softmax to produce valid probability distributions
    4. Temperature parameter controls sharpness of the output distribution
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
        """
        Args:
            d_model: Input embedding dimension
            max_qubits: Maximum number of qubits (for output size estimation)
            hidden_dim: Hidden dimension for MLP heads
            temperature: Initial temperature for softmax
            dropout: Dropout rate
            use_mlp_scorer: If True, use MLP scorer; else use dot product
            temperature_floor: Minimum temperature (prevents over-sharpening)
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_qubits = max_qubits
        self.temperature_floor = temperature_floor
        
        # Temperature parameter (learnable, with floor)
        self.temp = nn.Parameter(torch.tensor(temperature))
        
        # SN-D head
        self.sn_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        
        # HN-E head
        self.hn_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        
        # Scorer: computes logits from encoder features and head outputs
        # Two options: dot product (faster) or MLP (more expressive)
        self.use_mlp_scorer = use_mlp_scorer
        if use_mlp_scorer:
            self.scorer = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.scorer = nn.Linear(d_model, 1, bias=False)
        
        # Bitstring encoder reference (set later)
        self.bitstring_encoder = None
    
    def set_bitstring_encoder(self, encoder):
        """
        Set the bitstring encoder for scoring candidate bitstrings.
        
        Args:
            encoder: BitstringEncoder instance
        """
        self.bitstring_encoder = encoder
    
    def _compute_logits(
        self,
        head_output: torch.Tensor,
        bitstrings: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute logits for each bitstring using the head output.
        
        Args:
            head_output: (B, d_model) - output from SN-D or HN-E head
            bitstrings: (B, M, n_qubits) - candidate bitstrings
            mask: (B, M) - boolean mask for valid positions
        
        Returns:
            (B, M) - logits for each bitstring
        """
        B, M, n = bitstrings.shape
        
        if self.bitstring_encoder is None:
            raise RuntimeError("Bitstring encoder not set. Call set_bitstring_encoder() first.")
        
        # Encode bitstrings: (B, M, d_model)
        # Use no counts (all ones) for encoding
        counts = torch.ones(B, M, 1, device=bitstrings.device)
        embeddings, enc_mask = self.bitstring_encoder(bitstrings)
        
        # Combine mask with any provided mask
        if mask is not None:
            mask = mask & enc_mask
        else:
            mask = enc_mask
        
        if self.use_mlp_scorer:
            # MLP scorer: concatenate head_output and embeddings
            # Expand head_output to (B, M, d_model)
            head_expanded = head_output.unsqueeze(1).expand(-1, M, -1)
            combined = torch.cat([head_expanded, embeddings], dim=-1)
            logits = self.scorer(combined).squeeze(-1)  # (B, M)
        else:
            # Dot product scorer: (B, d_model) x (B, M, d_model) -> (B, M)
            logits = torch.bmm(
                head_output.unsqueeze(1),  # (B, 1, d_model)
                embeddings.transpose(1, 2)  # (B, d_model, M)
            ).squeeze(1)  # (B, M)
        
        # Mask invalid positions
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        
        return logits, mask
    
    def forward(
        self,
        z: torch.Tensor,
        bitstrings: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Generate distributions from the global latent representation.
        
        Args:
            z: (B, d_model) - global latent from Set Transformer
            bitstrings: (B, M, n_qubits) - candidate bitstrings
            mask: (B, M) - boolean mask for valid positions
        
        Returns:
            sn_dist: (B, M) - SN-D distribution (softmax over M)
            hn_dist: (B, M) - HN-E distribution (softmax over M)
        """
        # Apply temperature floor
        temp = torch.clamp(self.temp, min=self.temperature_floor)
        
        # SN-D head
        sn_z = self.sn_head(z)  # (B, d_model)
        sn_logits, mask = self._compute_logits(sn_z, bitstrings, mask)
        sn_dist = F.softmax(sn_logits / temp, dim=-1)
        
        # HN-E head
        hn_z = self.hn_head(z)  # (B, d_model)
        hn_logits, mask = self._compute_logits(hn_z, bitstrings, mask)
        hn_dist = F.softmax(hn_logits / temp, dim=-1)
        
        return sn_dist, hn_dist
