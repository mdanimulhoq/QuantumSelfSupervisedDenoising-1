"""
Data encoding: counts → tensor (TDD §5.4)
Vectorised, no Python loops in hot path.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor

from src.utils.types import Bitstring, CountsDict, Distribution

# ------------------------------------------------------------
# 1. Counts → Tensor
# ------------------------------------------------------------

def counts_to_tensor(
    counts: CountsDict,
    n_qubits: int,
    max_bitstrings: Optional[int] = None,
    sort_by_count: bool = True,
    return_counts: bool = False,
    dtype: torch.dtype = torch.long,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Convert Qiskit CountsDict to a tensor of bitstrings.

    Args:
        counts: Dict[str, int], e.g. {'010': 45, '101': 55}
        n_qubits: Number of qubits (for padding bitstrings)
        max_bitstrings: Keep only the top-k most frequent bitstrings
        sort_by_count: Sort by descending count
        return_counts: If True, also return the counts as a tensor
        dtype: Data type for the bitstring tensor

    Returns:
        bitstrings: (M, n) int tensor
        counts_tensor: (M,) int tensor (if return_counts=True)
    """
    # Sort items (faster than Python loop)
    items = sorted(counts.items(), key=lambda x: x[1], reverse=sort_by_count)
    if max_bitstrings is not None:
        items = items[:max_bitstrings]

    if not items:
        bitstrings = torch.zeros((0, n_qubits), dtype=dtype)
        if return_counts:
            return bitstrings, torch.zeros(0, dtype=torch.long)
        return bitstrings

    # Vectorised: build a list of ints for each bitstring
    # Using list comprehension is still Python-level, but fast for typical M <= 2^n
    # For larger M, this is still the best we can do without custom ops.
    bits_list = []
    counts_list = []
    for bs, cnt in items:
        # Pad bitstring to n_qubits (Qiskit order: least significant first?)
        # We store as int bits 0/1 in the order they appear in Qiskit string.
        # '010' -> [0,1,0] for n_qubits=3
        # But we need to handle cases where len(bs) < n_qubits.
        if len(bs) < n_qubits:
            bs_padded = bs.zfill(n_qubits)
        else:
            bs_padded = bs
        # Convert to list of ints (MSB first? We keep Qiskit order)
        # Qiskit string: '010' -> bit 0 is least significant qubit?
        # To match typical quantum convention, we reverse to qubit order.
        # TDD §5.4: "bitstrings are stored in qubit order (MSB first)"
        # So we reverse the string to get MSB first.
        bits = [int(c) for c in bs_padded[::-1]]  # reverse: MSB first
        bits_list.append(bits)
        counts_list.append(cnt)

    bitstrings = torch.tensor(bits_list, dtype=dtype)
    counts_tensor = torch.tensor(counts_list, dtype=torch.long)

    if return_counts:
        return bitstrings, counts_tensor
    return bitstrings


# ------------------------------------------------------------
# 2. Tensor → Counts
# ------------------------------------------------------------

def tensor_to_counts(
    bitstrings: Tensor,
    counts: Optional[Tensor] = None,
) -> CountsDict:
    """
    Convert a tensor of bitstrings back to a Qiskit-style CountsDict.

    Args:
        bitstrings: (M, n) int tensor
        counts: (M,) optional counts tensor (if not provided, all counts = 1)

    Returns:
        CountsDict: {bitstring: count}
    """
    if bitstrings.numel() == 0:
        return {}

    # Convert to list of bitstrings (Python-level, but M is typically small)
    # We need to reverse back to Qiskit order (LSB first)
    if counts is None:
        counts = torch.ones(bitstrings.shape[0], dtype=torch.long)

    counts_dict = {}
    for i in range(bitstrings.shape[0]):
        # Convert bits back to string: MSB -> LSB, so reverse back
        bits = bitstrings[i].tolist()
        # Reverse to get Qiskit order (LSB first)
        bs = ''.join(str(b) for b in reversed(bits))
        cnt = int(counts[i].item()) if isinstance(counts, Tensor) else int(counts[i])
        counts_dict[bs] = cnt

    return counts_dict


# ------------------------------------------------------------
# 3. Batched variants
# ------------------------------------------------------------

def batched_counts_to_tensor(
    counts_list: List[CountsDict],
    n_qubits: int,
    max_bitstrings: Optional[int] = None,
    pad_to_max: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Convert a list of CountsDict to batched tensors.

    Args:
        counts_list: List of CountsDict
        n_qubits: Number of qubits
        max_bitstrings: Max bitstrings per sample (keep top-k)
        pad_to_max: Pad all samples to the same length

    Returns:
        bitstrings: (B, M, n) padded tensor
        counts: (B, M) padded counts tensor
        mask: (B, M) valid mask (1 for real, 0 for padding)
    """
    if not counts_list:
        return torch.zeros((0, 0, n_qubits), dtype=torch.long), torch.zeros((0, 0), dtype=torch.long), torch.zeros((0, 0), dtype=torch.bool)

    batch_data = []
    for counts in counts_list:
        bits, cnts = counts_to_tensor(
            counts, n_qubits, max_bitstrings, return_counts=True
        )
        batch_data.append((bits, cnts))

    if pad_to_max:
        max_M = max(b.shape[0] for b, _ in batch_data)
        batched_bits = []
        batched_counts = []
        masks = []
        for bits, cnts in batch_data:
            M = bits.shape[0]
            pad_len = max_M - M
            if pad_len > 0:
                pad_bits = torch.zeros((pad_len, n_qubits), dtype=bits.dtype)
                pad_cnts = torch.zeros(pad_len, dtype=cnts.dtype)
                bits = torch.cat([bits, pad_bits], dim=0)
                cnts = torch.cat([cnts, pad_cnts], dim=0)
            mask = torch.cat([torch.ones(M, dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)], dim=0)
            batched_bits.append(bits)
            batched_counts.append(cnts)
            masks.append(mask)
        return torch.stack(batched_bits, dim=0), torch.stack(batched_counts, dim=0), torch.stack(masks, dim=0)
    else:
        # Return as list of tensors (variable length)
        batched_bits = [b for b, _ in batch_data]
        batched_counts = [c for _, c in batch_data]
        masks = [torch.ones(b.shape[0], dtype=torch.bool) for b in batched_bits]
        return batched_bits, batched_counts, masks


def batched_tensor_to_counts(
    batched_bits: Union[Tensor, List[Tensor]],
    batched_counts: Optional[Union[Tensor, List[Tensor]]] = None,
    mask: Optional[Union[Tensor, List[Tensor]]] = None,
) -> List[CountsDict]:
    """
    Convert batched tensors back to a list of CountsDict.
    """
    # Convert to list format if batched tensor
    if isinstance(batched_bits, Tensor):
        B = batched_bits.shape[0]
        bits_list = [batched_bits[i] for i in range(B)]
        if batched_counts is not None and isinstance(batched_counts, Tensor):
            counts_list = [batched_counts[i] for i in range(B)]
        else:
            counts_list = [None] * B
        if mask is not None and isinstance(mask, Tensor):
            mask_list = [mask[i] for i in range(B)]
        else:
            mask_list = [None] * B
    else:
        bits_list = batched_bits
        counts_list = batched_counts if batched_counts is not None else [None] * len(bits_list)
        mask_list = mask if mask is not None else [None] * len(bits_list)

    result = []
    for i, bits in enumerate(bits_list):
        cnts = counts_list[i] if counts_list[i] is not None else None
        m = mask_list[i] if mask_list[i] is not None else None
        if m is not None:
            bits = bits[m]
            if cnts is not None:
                cnts = cnts[m]
        result.append(tensor_to_counts(bits, cnts))

    return result


# ------------------------------------------------------------
# 4. Distribution ↔ Tensor
# ------------------------------------------------------------

def distribution_to_tensor(
    dist: Distribution,
) -> Tensor:
    """
    Convert a Distribution dataclass to a tensor (bitstrings + probs).
    """
    # Already has bitstrings and probs
    return dist.bitstrings, dist.probs


def tensor_to_distribution(
    bitstrings: Tensor,
    probs: Tensor,
    n_qubits: int,
    shots: int,
) -> Distribution:
    """
    Convert tensors back to a Distribution dataclass.
    """
    return Distribution(
        bitstrings=bitstrings,
        probs=probs,
        n_qubits=n_qubits,
        shots=shots,
    )
