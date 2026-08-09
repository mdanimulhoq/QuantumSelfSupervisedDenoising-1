"""
Count-Weighted Set Transformer (TDD §3.3)
ISAB + PMA with count features in attention logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional mask.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        count_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Q, d_model)
            key: (B, K, d_model)
            value: (B, K, d_model)
            mask: (B, Q, K) or (B, K) boolean mask
            count_weights: (B, K, 1) or (B, K) count weights
        
        Returns:
            (B, Q, d_model)
        """
        B, Q, _ = query.shape
        B, K, _ = key.shape
        
        q = self.W_q(query).view(B, Q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(B, K, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(B, K, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Handle count_weights with proper shape
        if count_weights is not None:
            if count_weights.dim() == 3:
                # (B, K, 1) -> (B, 1, 1, K)
                count_weights = count_weights.squeeze(-1).unsqueeze(1).unsqueeze(1)
            elif count_weights.dim() == 2:
                # (B, K) -> (B, 1, 1, K)
                count_weights = count_weights.unsqueeze(1).unsqueeze(1)
            # Add to scores
            scores = scores + count_weights * 0.1
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Q, self.d_model)
        out = self.W_o(out)
        
        return out


class MAB(nn.Module):
    """
    Multihead Attention Block.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        mask: torch.Tensor = None,
        count_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if y is None:
            y = x
        
        attn = self.mha(x, y, y, mask, count_weights)
        x = self.norm1(x + self.dropout(attn))
        
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        
        return x


class SAB(nn.Module):
    """
    Self-Attention Block (SAB).
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mab = MAB(d_model, n_heads, d_ff, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        count_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.mab(x, None, mask, count_weights)


class ISAB(nn.Module):
    """
    Induced Set Attention Block (ISAB).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        m: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.m = m
        self.seed = nn.Parameter(torch.randn(1, m, d_model))
        self.mab1 = MAB(d_model, n_heads, d_ff, dropout)
        self.mab2 = MAB(d_model, n_heads, d_ff, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        count_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        seeds = self.seed.expand(B, -1, -1)
        h = self.mab1(seeds, x, mask, count_weights)
        out = self.mab2(x, h, None, None)
        return out


class PMA(nn.Module):
    """
    Pooling by Multihead Attention (PMA).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_seeds: int = 1,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_seeds = n_seeds
        self.seed = nn.Parameter(torch.randn(1, n_seeds, d_model))
        self.mab = MAB(d_model, n_heads, d_ff, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        count_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        seeds = self.seed.expand(B, -1, -1)
        return self.mab(seeds, x, mask, count_weights)


class CountWeightedSetTransformer(nn.Module):
    """
    Count-Weighted Set Transformer (TDD §3.3).
    """
    
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_ISAB: int = 2,
        n_SAB: int = 1,
        d_ff: int = 256,
        m: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            ISAB(d_model, n_heads, d_ff, m, dropout)
            for _ in range(n_ISAB)
        ])
        
        self.pma = PMA(d_model, n_heads, 1, d_ff, dropout)
        
        self.decoder = nn.ModuleList([
            SAB(d_model, n_heads, d_ff, dropout)
            for _ in range(n_SAB)
        ])
        
        self.count_proj = nn.Linear(1, d_model)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        counts: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        count_emb = self.count_proj(counts)
        x = embeddings + count_emb
        count_weights = counts
        
        if mask is None:
            mask = torch.ones_like(counts).squeeze(-1).bool()
        
        for isab in self.encoder:
            x = isab(x, mask, count_weights)
        
        z = self.pma(x, mask, count_weights)
        
        for sab in self.decoder:
            z = sab(z)
        
        return z.squeeze(1)
