"""Type definitions for N2LN-QEM (TDD §2.1)."""

from dataclasses import dataclass
from typing import Dict, List, Union
import torch

Bitstring = str
CountsDict = Dict[Bitstring, int]
ProbVec = torch.Tensor  # (M,) normalized probabilities

@dataclass
class Distribution:
    bitstrings: torch.Tensor  # (M, n)
    probs: torch.Tensor       # (M,)
    n_qubits: int
    shots: int

@dataclass
class Batch:
    bitstrings: torch.Tensor  # (B, M, n)
    counts: torch.Tensor      # (B, M, 1)
    target_sn: torch.Tensor   # (B, 2^n)
    target_hn: torch.Tensor   # (B, 2^n)
