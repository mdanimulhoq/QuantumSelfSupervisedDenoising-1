"""
Full N2LN Model Wrapper (TDD §2.6, §3.1)
Assembles Encoder + Set Transformer + Dual-Head Decoder.
Supports mode switching: 'sn_only', 'hn_only', 'unified'.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import BitstringEncoder
from src.models.set_transformer import CountWeightedSetTransformer
from src.models.decoder import DualHeadDecoder


class N2LNQEM(nn.Module):
    """
    N2LN-QEM Model Wrapper.
    
    Architecture:
    1. Bitstring Encoder: per-qubit embedding + positional encoding
    2. Count-Weighted Set Transformer: ISAB + PMA with count weighting
    3. Dual-Head Decoder: SN-D and HN-E heads with softmax output
    
    The model supports three training modes:
    - 'sn_only': Only SN-D head is used (Phase 1)
    - 'hn_only': Only HN-E head is used (Phase 2)
    - 'unified': Both heads are used with consistency loss (Phase 3)
    """
    
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_isab: int = 2,
        n_sab: int = 1,
        d_ff: int = 256,
        m: int = 16,
        decoder_hidden: int = 128,
        dropout: float = 0.1,
        max_qubits: int = 20,
        use_mlp_scorer: bool = True,
        temperature_floor: float = 0.3,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_isab: Number of ISAB layers
            n_sab: Number of SAB layers in decoder
            d_ff: Feed-forward dimension
            m: Number of inducing points for ISAB
            decoder_hidden: Hidden dimension in decoder MLP
            dropout: Dropout rate
            max_qubits: Maximum qubits supported
            use_mlp_scorer: Whether to use MLP scorer in decoder
            temperature_floor: Minimum temperature for softmax
        """
        super().__init__()
        
        # 1. Bitstring Encoder
        self.encoder = BitstringEncoder(
            d_model=d_model,
            n_max_qubits=max_qubits,
            dropout=dropout,
        )
        
        # 2. Count-Weighted Set Transformer
        self.transformer = CountWeightedSetTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_ISAB=n_isab,
            n_SAB=n_sab,
            d_ff=d_ff,
            m=m,
            dropout=dropout,
        )
        
        # 3. Dual-Head Decoder
        self.decoder = DualHeadDecoder(
            d_model=d_model,
            max_qubits=max_qubits,
            hidden_dim=decoder_hidden,
            temperature=1.0,
            dropout=dropout,
            use_mlp_scorer=use_mlp_scorer,
            temperature_floor=temperature_floor,
        )
        
        # Connect decoder to encoder for scoring
        self.decoder.set_bitstring_encoder(self.encoder)
        
        # Mode for training phases
        self.mode = 'unified'  # 'sn_only', 'hn_only', or 'unified'
        
        # Consistency loss weight (set during training)
        self.consistency_weight = 0.3
    
    def forward(
        self,
        bitstrings: torch.Tensor,
        counts: torch.Tensor,
        mode: str = None,
    ) -> tuple:
        """
        Forward pass through the model.
        
        Args:
            bitstrings: (B, M, n_qubits) tensor of 0/1 values
            counts: (B, M, 1) normalized counts (probabilities)
            mode: One of 'sn_only', 'hn_only', or 'unified'
                  If None, uses self.mode
        
        Returns:
            sn_dist: (B, M) SN-D distribution (or None if hn_only)
            hn_dist: (B, M) HN-E distribution (or None if sn_only)
        """
        if mode is None:
            mode = self.mode
        
        # 1. Encode bitstrings
        embeddings, mask = self.encoder(bitstrings)
        
        # 2. Set Transformer -> global latent
        z = self.transformer(embeddings, counts, mask)
        
        # 3. Decoder -> distributions
        sn_dist, hn_dist = self.decoder(z, bitstrings, mask)
        
        # 4. Mode-based output
        if mode == 'sn_only':
            return sn_dist, None
        elif mode == 'hn_only':
            return None, hn_dist
        else:  # unified
            return sn_dist, hn_dist
    
    def set_mode(self, mode: str):
        """
        Set the training mode.
        
        Args:
            mode: 'sn_only', 'hn_only', or 'unified'
        """
        if mode not in ['sn_only', 'hn_only', 'unified']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sn_only', 'hn_only', or 'unified'")
        self.mode = mode
        print(f"🔀 Model mode set to: {mode}")
    
    def set_consistency_weight(self, weight: float):
        """
        Set the consistency loss weight for unified training.
        
        Args:
            weight: Consistency loss weight (recommended: 0.1-0.5)
        """
        self.consistency_weight = weight
        print(f"⚖️ Consistency weight set to: {weight}")
    
    def compute_consistency_loss(
        self,
        sn_dist: torch.Tensor,
        hn_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the cross-stage consistency loss.
        
        The consistency loss encourages the SN-D and HN-E heads to produce
        similar distributions on the same input.
        
        Args:
            sn_dist: (B, M) SN-D distribution
            hn_dist: (B, M) HN-E distribution
        
        Returns:
            (scalar) consistency loss
        """
        # KL divergence between SN-D and HN-E distributions
        eps = 1e-12
        sn_dist = sn_dist.clamp(min=eps)
        hn_dist = hn_dist.clamp(min=eps)
        
        # Symmetric KL divergence
        kl_sn_to_hn = (sn_dist * (sn_dist / hn_dist).log()).sum(dim=-1).mean()
        kl_hn_to_sn = (hn_dist * (hn_dist / sn_dist).log()).sum(dim=-1).mean()
        
        return 0.5 * (kl_sn_to_hn + kl_hn_to_sn)
    
    def get_sn_distribution(self, bitstrings: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Get only the SN-D distribution (shot-noise denoised).
        
        Args:
            bitstrings: (B, M, n_qubits) tensor
            counts: (B, M, 1) normalized counts
        
        Returns:
            (B, M) SN-D distribution
        """
        sn_dist, _ = self.forward(bitstrings, counts, mode='sn_only')
        return sn_dist
    
    def get_hn_distribution(self, bitstrings: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Get only the HN-E distribution (hardware-mitigated).
        
        Args:
            bitstrings: (B, M, n_qubits) tensor
            counts: (B, M, 1) normalized counts
        
        Returns:
            (B, M) HN-E distribution
        """
        _, hn_dist = self.forward(bitstrings, counts, mode='hn_only')
        return hn_dist
