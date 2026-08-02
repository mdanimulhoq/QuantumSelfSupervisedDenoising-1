"""
Training utilities: metrics tracking, early stopping, gradient accumulation.
"""

import json
from typing import Dict, Any, Optional, List
from collections import defaultdict
import torch
import numpy as np


class MetricTracker:
    """
    Track and log metrics during training.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_epoch_metrics = defaultdict(list)
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics with new values."""
        for k, v in metrics.items():
            self.metrics[k].append(v)
            self.current_epoch_metrics[k].append(v)
    
    def get_average(self, key: str, window: Optional[int] = None) -> float:
        """Get average of a metric (optionally over window)."""
        values = self.metrics[key]
        if window and len(values) > window:
            values = values[-window:]
        return sum(values) / len(values) if values else 0.0
    
    def get_epoch_average(self, key: str) -> float:
        """Get average for the current epoch."""
        values = self.current_epoch_metrics.get(key, [])
        return sum(values) / len(values) if values else 0.0
    
    def reset_epoch(self) -> None:
        """Reset epoch metrics."""
        self.current_epoch_metrics = defaultdict(list)
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get averages of all metrics."""
        return {k: sum(v) / len(v) if v else 0.0 for k, v in self.metrics.items()}


class EarlyStopping:
    """
    Early stopping with patience and min delta.
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to consider improvement
            metric: Metric to monitor
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def update(self, value: float) -> bool:
        """
        Update early stopping state.
        
        Returns:
            bool: True if early stopping triggered
        """
        if self.best_value is None:
            self.best_value = value
            self.counter = 0
            return False
        
        if self.mode == "min":
            is_improvement = (value < self.best_value - self.min_delta)
        else:
            is_improvement = (value > self.best_value + self.min_delta)
        
        if is_improvement:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class GradientAccumulator:
    """
    Gradient accumulation for larger effective batch sizes.
    """
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.loss = None
    
    def should_update(self) -> bool:
        """Check if gradients should be updated."""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False
    
    def reset(self) -> None:
        """Reset accumulator."""
        self.current_step = 0
        self.loss = None


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: List[str] = ["tvd", "kl", "mse"],
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        pred: Predicted distribution
        target: Target distribution
        metrics: List of metrics to compute
    
    Returns:
        Dictionary of metric values
    """
    from src.losses.distribution import total_variation_distance, kl_divergence
    
    results = {}
    eps = 1e-12
    
    if "tvd" in metrics:
        results["tvd"] = total_variation_distance(pred, target, reduction="mean").item()
    
    if "kl" in metrics:
        results["kl"] = kl_divergence(pred, target, reduction="mean", eps=eps).item()
    
    if "mse" in metrics:
        results["mse"] = torch.nn.functional.mse_loss(pred, target).item()
    
    if "fidelity" in metrics:
        # Fidelity = (sum(sqrt(pred * target)))^2
        fidelity = (torch.sqrt(pred * target).sum(dim=-1) ** 2).mean().item()
        results["fidelity"] = fidelity
    
    if "cross_entropy" in metrics:
        # Cross entropy = -sum(target * log(pred))
        pred_clamped = pred.clamp(min=eps)
        ce = -(target * pred_clamped.log()).sum(dim=-1).mean().item()
        results["cross_entropy"] = ce
    
    return results
