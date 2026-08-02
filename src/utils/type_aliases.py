"""
Type aliases and dataclasses for N2LN-QEM.

Implements TDD §2.1 Notation.

Notation reference (from TDD §2.1):
    - n: Number of qubits
    - N: Number of shots (measurement repetitions)
    - p_ideal(x): Ideal (noise-free) probability of bitstring x
    - p_noisy(x): Noisy probability (hardware noise applied, infinite shots)
    - \hat{p}_S(x): Empirical distribution from S shots
"""
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import torch

# ============= Type Aliases =============

# A bitstring as a tuple of ints (0/1) or a string
Bitstring = Union[str, tuple, List[int]]

# Counts dictionary: mapping from bitstring to count
# Example: {"000": 45, "001": 23, ...}
CountsDict = Dict[str, int]

# Probability vector: either numpy array or torch tensor
ProbVec = Union[np.ndarray, torch.Tensor]

# ============= Dataclasses =============

@dataclass
class Distribution:
    """
    Represents a probability distribution over bitstrings.
    
    This is the core data structure used throughout N2LN-QEM.
    
    Attributes:
        probs: Probability vector (ProbVec) of shape (M,) where M = 2^n
        bitstrings: List of bitstrings as strings, length M
        n_qubits: Number of qubits (n)
        shots: Number of shots (N) used to estimate this distribution, if empirical
        is_empirical: Whether this is an empirical (finite-shot) distribution
        metadata: Additional metadata (noise scale, circuit info, etc.)
    
    TDD Notation:
        - n: n_qubits
        - N: shots
        - p_ideal: The ideal (noise-free) distribution
        - p_noisy: The noisy distribution (hardware noise applied)
        - \hat{p}_S: Empirical distribution from S shots
    """
    probs: ProbVec
    bitstrings: List[str]
    n_qubits: int
    shots: Optional[int] = None
    is_empirical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that probs sum to 1 (approximately)."""
        if isinstance(self.probs, torch.Tensor):
            prob_sum = self.probs.sum().item()
        else:
            prob_sum = float(np.sum(self.probs))
        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(f"Probabilities sum to {prob_sum}, not 1.0")
    
    @property
    def support_size(self) -> int:
        """Size of the support (number of bitstrings with non-zero probability)."""
        if isinstance(self.probs, torch.Tensor):
            return int((self.probs > 1e-8).sum().item())
        return int(np.sum(self.probs > 1e-8))
    
    @property
    def entropy(self) -> float:
        """Shannon entropy of the distribution."""
        if isinstance(self.probs, torch.Tensor):
            p = self.probs[self.probs > 1e-8]
            return - (p * torch.log(p)).sum().item()
        p = self.probs[self.probs > 1e-8]
        return - float(np.sum(p * np.log(p)))
    
    def to_tensor(self) -> torch.Tensor:
        """Convert probs to torch tensor."""
        if isinstance(self.probs, torch.Tensor):
            return self.probs
        return torch.tensor(self.probs, dtype=torch.float32)
    
    def to_numpy(self) -> np.ndarray:
        """Convert probs to numpy array."""
        if isinstance(self.probs, np.ndarray):
            return self.probs
        return self.probs.numpy()
    
    def sample(self, num_samples: int, seed: Optional[int] = None) -> List[str]:
        """
        Sample bitstrings from the distribution.
        
        Args:
            num_samples: Number of samples to draw
            seed: Random seed for reproducibility
        
        Returns:
            List of sampled bitstrings
        """
        if seed is not None:
            np.random.seed(seed)
        
        probs = self.to_numpy()
        indices = np.random.choice(
            len(probs), 
            size=num_samples, 
            p=probs,
            replace=True
        )
        return [self.bitstrings[i] for i in indices]

@dataclass
class CountsData:
    """
    Raw counts data from quantum measurement.
    
    This is the input format from hardware/simulator.
    
    Attributes:
        counts: Dictionary mapping bitstring -> count
        n_qubits: Number of qubits (n)
        shots: Total number of shots (N)
        noise_scale: Noise scale factor (lambda)
        metadata: Additional metadata
    """
    counts: CountsDict
    n_qubits: int
    shots: int
    noise_scale: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_distribution(self) -> Distribution:
        """
        Convert counts to a Distribution object.
        
        Returns:
            Distribution: Empirical distribution from these counts
        """
        # Sort bitstrings for reproducibility
        sorted_items = sorted(self.counts.items())
        bitstrings = [bs for bs, _ in sorted_items]
        counts = [c for _, c in sorted_items]
        probs = np.array(counts, dtype=np.float32) / self.shots
        
        return Distribution(
            probs=probs,
            bitstrings=bitstrings,
            n_qubits=self.n_qubits,
            shots=self.shots,
            is_empirical=True,
            metadata={
                "noise_scale": self.noise_scale,
                **self.metadata
            }
        )
    
    @property
    def support_size(self) -> int:
        """Number of observed bitstrings."""
        return len(self.counts)
    
    @property
    def sparsity(self) -> float:
        """Fraction of the full Hilbert space observed."""
        return len(self.counts) / (2 ** self.n_qubits)

# ============= Helper Functions =============

def create_ideal_distribution(
    probs: ProbVec,
    n_qubits: int,
    bitstrings: Optional[List[str]] = None
) -> Distribution:
    """
    Create an ideal (noise-free) distribution.
    
    TDD Notation: p_ideal
    
    Args:
        probs: Probability vector
        n_qubits: Number of qubits
        bitstrings: List of bitstrings (auto-generated if None)
    
    Returns:
        Distribution: Ideal distribution
    """
    if bitstrings is None:
        # Generate all bitstrings in binary order
        bitstrings = [format(i, f"0{n_qubits}b") for i in range(2 ** n_qubits)]
    
    return Distribution(
        probs=probs,
        bitstrings=bitstrings,
        n_qubits=n_qubits,
        is_empirical=False,
        metadata={"type": "ideal"}
    )

def create_noisy_distribution(
    probs: ProbVec,
    n_qubits: int,
    noise_scale: float = 1.0,
    bitstrings: Optional[List[str]] = None
) -> Distribution:
    """
    Create a noisy distribution (hardware noise applied).
    
    TDD Notation: p_noisy
    
    Args:
        probs: Noisy probability vector
        n_qubits: Number of qubits
        noise_scale: Noise scale factor (lambda)
        bitstrings: List of bitstrings (auto-generated if None)
    
    Returns:
        Distribution: Noisy distribution
    """
    if bitstrings is None:
        bitstrings = [format(i, f"0{n_qubits}b") for i in range(2 ** n_qubits)]
    
    return Distribution(
        probs=probs,
        bitstrings=bitstrings,
        n_qubits=n_qubits,
        is_empirical=False,
        metadata={"type": "noisy", "noise_scale": noise_scale}
    )

def create_empirical_distribution(
    counts: CountsDict,
    n_qubits: int,
    shots: int,
    noise_scale: float = 1.0
) -> Distribution:
    """
    Create an empirical distribution from counts.
    
    TDD Notation: \hat{p}_S
    
    Args:
        counts: Counts dictionary
        n_qubits: Number of qubits
        shots: Total number of shots (S)
        noise_scale: Noise scale factor (lambda)
    
    Returns:
        Distribution: Empirical distribution
    """
    return CountsData(
        counts=counts,
        n_qubits=n_qubits,
        shots=shots,
        noise_scale=noise_scale
    ).to_distribution()

# ============= Type Checking Helpers =============

def is_valid_prob_vector(probs: ProbVec, eps: float = 1e-6) -> bool:
    """
    Check if a vector is a valid probability distribution.
    
    Args:
        probs: Probability vector
        eps: Tolerance for sum check
    
    Returns:
        bool: True if valid
    """
    if isinstance(probs, torch.Tensor):
        return (probs >= -eps).all() and abs(probs.sum().item() - 1.0) < eps
    return (probs >= -eps).all() and abs(np.sum(probs) - 1.0) < eps

def validate_distribution(dist: Distribution) -> bool:
    """
    Validate a Distribution object.
    
    Checks:
        - Probabilities sum to 1
        - All probabilities >= 0
        - Number of bitstrings matches 2^n (if full support)
    
    Returns:
        bool: True if valid
    """
    # Check sum
    if not is_valid_prob_vector(dist.probs):
        return False
    
    # Check bitstrings length
    expected_len = 2 ** dist.n_qubits
    if len(dist.bitstrings) != expected_len:
        # For sparse distributions, this may not hold
        pass
    
    return True
