"""
Full N2LN Model Wrapper (TDD §2.6, §3.1)
Assembles Encoder + Set Transformer + Dual-Head Decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import BitstringEncoder
from src.models.set_transformer import CountWeightedSetTransformer
from src.models.decoder import DualHeadDecoder


class N2LNQEM(nn.Module):
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
        n_qubits: int = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits if n_qubits is not None else max_qubits
        self.max_qubits = max_qubits
        
        self.encoder = BitstringEncoder(
            d_model=d_model,
            n_max_qubits=max_qubits,
            dropout=dropout,
        )
        
        self.transformer = CountWeightedSetTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_ISAB=n_isab,
            n_SAB=n_sab,
            d_ff=d_ff,
            m=m,
            dropout=dropout,
        )
        
        self.decoder = DualHeadDecoder(
            d_model=d_model,
            max_qubits=max_qubits,
            hidden_dim=decoder_hidden,
            temperature=1.0,
            dropout=dropout,
            use_mlp_scorer=use_mlp_scorer,
            temperature_floor=temperature_floor,
        )
        self.decoder.set_bitstring_encoder(self.encoder)
        self.mode = 'unified'
        self.consistency_weight = 0.3
    
    def forward(self, bitstrings, counts, mask=None, mode=None):
        if mode is None:
            mode = self.mode
        
        # If mask not provided, assume all are real
        if mask is None:
            mask = torch.ones(bitstrings.shape[0], bitstrings.shape[1], dtype=torch.bool, device=bitstrings.device)
        
        embeddings = self.encoder(bitstrings)
        z = self.transformer(embeddings, counts, mask)
        sn_dist, hn_dist = self.decoder(z, bitstrings, mask)
        
        if mode == 'sn_only':
            return sn_dist, None
        elif mode == 'hn_only':
            return None, hn_dist
        else:
            return sn_dist, hn_dist
    
    def set_mode(self, mode):
        self.mode = mode
    
    def set_consistency_weight(self, weight):
        self.consistency_weight = weight
    
    def compute_consistency_loss(self, sn_dist, hn_dist):
        eps = 1e-12
        sn_dist = sn_dist.clamp(min=eps)
        hn_dist = hn_dist.clamp(min=eps)
        kl_sn_to_hn = (sn_dist * (sn_dist / hn_dist).log()).sum(dim=-1).mean()
        kl_hn_to_sn = (hn_dist * (hn_dist / sn_dist).log()).sum(dim=-1).mean()
        return 0.5 * (kl_sn_to_hn + kl_hn_to_sn)
