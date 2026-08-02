"""
Bitstring Encoder (TDD §3.2)
Per-qubit embedding + positional encoding, permutation-invariant across shots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for qubit positions.
    
    Uses sinusoidal encoding to make the model aware of qubit indices.
    """
    
    def __init__(self, d_model: int, max_len: int = 32):
        """
        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum number of qubits supported
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, dim: int = 2) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (B, M, n_qubits, d_model//2) or (B, M, n_qubits, d_model)
            dim: Dimension along which to add positional encoding (should be qubit dimension)
        
        Returns:
            Tensor with positional encoding added
        """
        # Ensure we have enough positional encodings
        if x.size(dim) > self.pe.size(0):
            # Extend PE if needed (periodic extension)
            new_len = x.size(dim)
            new_pe = torch.zeros(new_len, self.pe.size(1), device=self.pe.device)
            new_pe[:self.pe.size(0)] = self.pe
            for i in range(self.pe.size(0), new_len):
                new_pe[i] = self.pe[i % self.pe.size(0)]
            self.pe = new_pe
        
        # Add positional encoding along the qubit dimension
        return x + self.pe[:x.size(dim), :]


class BitstringEncoder(nn.Module):
    """
    Encodes bitstrings into learned features with positional encoding.
    
    Architecture:
    1. Per-qubit embedding: each bit (0/1) is embedded into d_model/2 dimensions
    2. Positional encoding: sinusoidal encoding added along qubit dimension
    3. Sum-pooling over qubits: permutation-invariant with respect to qubit ordering
    4. Linear projection to final d_model dimension
    
    The encoder is permutation-invariant across the shot axis (each shot is processed
    independently) and position-aware across qubits.
    """
    
    def __init__(
        self,
        d_model: int = 64,
        n_max_qubits: int = 32,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Output embedding dimension
            n_max_qubits: Maximum number of qubits supported (for positional encoding)
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_max_qubits = n_max_qubits
        
        # Per-qubit embedding: embed each bit (0/1) into d_model//2 dimensions
        self.qubit_embed = nn.Embedding(2, d_model // 2)
        
        # Positional encoding for qubit positions
        self.pos_enc = PositionalEncoding(d_model // 2, n_max_qubits)
        
        # Projection from d_model//2 to d_model
        self.proj = nn.Linear(d_model // 2, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, bitstrings: torch.Tensor) -> tuple:
        """
        Encode bitstrings to embeddings.
        
        Args:
            bitstrings: (B, M, n_qubits) tensor of 0/1 values
                        where -1 indicates padding (masked out)
        
        Returns:
            embeddings: (B, M, d_model) tensor of embeddings
            mask: (B, M) boolean mask where True indicates valid positions
        """
        B, M, n = bitstrings.shape
        
        # Create mask for padding (-1 values)
        mask = (bitstrings != -1).all(dim=-1)  # (B, M)
        
        # Replace -1 with 0 for embedding lookup (they will be masked out later)
        bits = bitstrings.clone()
        bits[bits == -1] = 0
        bits = bits.long()
        
        # Per-qubit embedding: (B, M, n, d_model//2)
        qubit_emb = self.qubit_embed(bits)
        
        # Add positional encoding along qubit dimension
        qubit_emb = self.pos_enc(qubit_emb, dim=2)
        
        # Mask out padding positions
        # Expand mask to qubit_emb shape: (B, M, n, 1)
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # (B, M, 1, 1)
        qubit_emb = qubit_emb * mask_expanded.float()
        
        # Sum-pooling over qubits: (B, M, d_model//2)
        # This makes the representation invariant to qubit ordering
        pooled = qubit_emb.sum(dim=2)
        
        # Project to d_model: (B, M, d_model)
        embeddings = self.proj(pooled)
        
        # Apply dropout and layer norm
        embeddings = self.dropout(embeddings)
        embeddings = self.norm(embeddings)
        
        # Mask out invalid positions in embeddings
        mask_expanded = mask.unsqueeze(-1)  # (B, M, 1)
        embeddings = embeddings * mask_expanded.float()
        
        return embeddings, mask


class BitstringEncoderWithCounts(nn.Module):
    """
    Bitstring encoder that also integrates count information.
    
    This is a wrapper that combines the base encoder with count weighting
    for use in the Set Transformer.
    """
    
    def __init__(self, encoder: BitstringEncoder):
        """
        Args:
            encoder: Base BitstringEncoder instance
        """
        super().__init__()
        self.encoder = encoder
        self.count_proj = nn.Linear(1, encoder.d_model)
    
    def forward(
        self,
        bitstrings: torch.Tensor,
        counts: torch.Tensor,
    ) -> tuple:
        """
        Encode bitstrings with count information.
        
        Args:
            bitstrings: (B, M, n_qubits) tensor of 0/1 values
            counts: (B, M, 1) normalized counts (probabilities)
        
        Returns:
            embeddings: (B, M, d_model) tensor of embeddings with count info
            mask: (B, M) boolean mask
        """
        # Get base embeddings
        embeddings, mask = self.encoder(bitstrings)
        
        # Add count projection
        count_emb = self.count_proj(counts)
        embeddings = embeddings + count_emb
        
        return embeddings, mask
