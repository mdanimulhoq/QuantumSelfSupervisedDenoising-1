"""
Tests for seeding utilities.
"""
import random
import numpy as np
import torch
import pytest
from src.utils.seeding import set_seed

def test_set_seed_identical():
    """Two runs with same seed produce identical tensors."""
    set_seed(42)
    a1 = torch.randn(10)
    b1 = np.random.randn(10)
    c1 = [random.random() for _ in range(10)]
    
    set_seed(42)  # reset
    a2 = torch.randn(10)
    b2 = np.random.randn(10)
    c2 = [random.random() for _ in range(10)]
    
    assert torch.allclose(a1, a2)
    assert np.allclose(b1, b2)
    assert c1 == c2

def test_set_seed_different():
    """Different seeds produce different outputs."""
    set_seed(42)
    a1 = torch.randn(10)
    
    set_seed(43)
    a2 = torch.randn(10)
    
    # Should be different (high probability)
    assert not torch.allclose(a1, a2)

def test_set_seed_reproducible_sequence():
    """Same seed produces same random sequence."""
    set_seed(123)
    seq1 = [random.randint(0, 100) for _ in range(10)]
    
    set_seed(123)
    seq2 = [random.randint(0, 100) for _ in range(10)]
    
    assert seq1 == seq2
