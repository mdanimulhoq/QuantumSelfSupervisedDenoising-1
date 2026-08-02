"""
Training loop for N2LN-QEM (TDD §4.2).
AdamW optimizer, LR scheduling, gradient clipping, wandb logging, checkpointing.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import wandb

from src.utils.seeding import set_seed
from src.utils.device import get_device, to_device
from src.utils.logging import setup_logger, log_metrics, save_checkpoint, load_checkpoint


class Trainer:
    """
    Main trainer class for N2LN-QEM.
    
    Handles:
        - Training loop with curriculum phases
        - AdamW optimizer with weight decay
        - Cosine annealing with warmup
        - Gradient clipping
        - Wandb logging
        - Checkpoint saving/loading
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fns: Dict[str, nn.Module],
        device: Optional[torch.device] = None,
        log_dir: Optional[Path] = None,
        use_wandb: bool = True,
    ):
        """
        Args:
            model: N2LN model
            config: Training configuration dictionary
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            loss_fns: Dictionary of loss functions
                Keys: 'snd', 'hne', 'physicality', 'consistency'
            device: Device to use (auto-detected if None)
            log_dir: Directory to save logs and checkpoints
            use_wandb: Whether to use wandb logging
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fns = loss_fns
        self.device = device or get_device()
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.use_wandb = use_wandb
        
        # Set seed for reproducibility
        set_seed(config.get("seed", 42))
        
        # Setup logger
        self.logger = setup_logger(
            name="trainer",
            log_dir=self.log_dir,
            use_wandb=use_wandb,
            wandb_project=config.get("wandb_project", "n2ln-qem"),
            wandb_entity=config.get("wandb_entity", None),
            wandb_config=config,
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Create checkpoint directory
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup AdamW optimizer."""
        lr = self.config.get("learning_rate", 3e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        betas = self.config.get("betas", [0.9, 0.999])
        eps = self.config.get("eps", 1e-8)
        
        # Group parameters (optional: different LR for different parts)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        self.logger.info(f"Optimizer: AdamW (lr={lr}, wd={weight_decay})")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[SequentialLR]:
        """Setup learning rate scheduler with warmup."""
        scheduler_config = self.config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine_warmup")
        
        if scheduler_type == "cosine_warmup":
            warmup_steps = scheduler_config.get("warmup_steps", 1000)
            total_steps = scheduler_config.get("total_steps", 100000)
            min_lr = scheduler_config.get("min_lr", 1e-6)
            
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=min_lr,
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
            self.logger.info(f"Scheduler: Cosine warmup (warmup={warmup_steps}, total={total_steps})")
            return scheduler
        
        self.logger.info("No scheduler configured")
        return None
    
    def _compute_loss(
        self,
        sn_dist: torch.Tensor,
        hn_dist: torch.Tensor,
        sn_target: torch.Tensor,
        hn_target: torch.Tensor,
        phase: str = "phase1",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss for a batch.
        
        Args:
            sn_dist: SN-D head output
            hn_dist: HN-E head output
            sn_target: SN-D target (high-shot distribution)
            hn_target: HN-E target (reduced-noise distribution)
            phase: Current training phase ('phase1', 'phase2', 'phase3')
        
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # SN-D loss (always computed)
        sn_loss = self.loss_fns["snd"](sn_dist, sn_target)
        losses["sn_loss"] = sn_loss
        
        # HN-E loss (if enabled)
        if phase != "phase1":
            hn_loss = self.loss_fns["hne"](hn_dist, hn_target)
            losses["hn_loss"] = hn_loss
        
        # Physicality regularization
        phys_loss_sn = self.loss_fns["physicality"](sn_dist)
        phys_loss_hn = self.loss_fns["physicality"](hn_dist)
        losses["phys_sn"] = phys_loss_sn
        losses["phys_hn"] = phys_loss_hn
        
        # Cross-stage consistency loss (phase3 only)
        if phase == "phase3":
            consist_loss = self.loss_fns["consistency"](sn_dist, hn_dist)
            losses["consist_loss"] = consist_loss
        
        # Total loss
        total_loss = sn_loss + losses.get("hn_loss", 0.0)
        total_loss = total_loss + 0.1 * (phys_loss_sn + phys_loss_hn)
        if phase == "phase3":
            total_loss = total_loss + 0.3 * losses.get("consist_loss", 0.0)
        
        losses["total_loss"] = total_loss
        
        return losses
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        phase: str = "phase1",
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Dictionary containing inputs and targets
            phase: Current training phase
        
        Returns:
            Dictionary of loss values
        """
        # Move batch to device
        batch = to_device(batch, self.device)
        
        # Forward pass
        sn_dist, hn_dist = self.model(
            bitstrings=batch["bitstrings"],
            counts=batch["counts"],
            mode=phase,
        )
        
        # Compute losses
        loss_dict = self._compute_loss(
            sn_dist=sn_dist,
            hn_dist=hn_dist,
            sn_target=batch["sn_target"],
            hn_target=batch["hn_target"],
            phase=phase,
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        
        # Gradient clipping
        grad_clip = self.config.get("gradient_clip", 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Convert to float values
        return {k: v.item() for k, v in loss_dict.items()}
    
    def train_epoch(
        self,
        epoch: int,
        phase: str = "phase1",
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            phase: Current training phase
        
        Returns:
            Average loss values for the epoch
        """
        self.model.train()
        epoch_losses = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            loss_dict = self.train_step(batch, phase)
            
            # Accumulate losses
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v
            
            # Log every N steps
            log_interval = self.config.get("log_interval", 10)
            if self.global_step % log_interval == 0:
                log_metrics(self.logger, loss_dict, step=self.global_step)
            
            self.global_step += 1
        
        # Average losses
        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Average validation losses
        """
        self.model.eval()
        val_losses = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = to_device(batch, self.device)
                
                sn_dist, hn_dist = self.model(
                    bitstrings=batch["bitstrings"],
                    counts=batch["counts"],
                    mode="phase3",  # Use full model for validation
                )
                
                loss_dict = self._compute_loss(
                    sn_dist=sn_dist,
                    hn_dist=hn_dist,
                    sn_target=batch["sn_target"],
                    hn_target=batch["hn_target"],
                    phase="phase3",
                )
                
                for k, v in loss_dict.items():
                    if k not in val_losses:
                        val_losses[k] = 0.0
                    val_losses[k] += v.item()
        
        num_batches = len(self.val_loader)
        return {k: v / num_batches for k, v in val_losses.items()}
    
    def train(
        self,
        num_epochs: int,
        phase: str = "phase1",
        save_every: int = 10,
        early_stopping_patience: Optional[int] = None,
    ) -> None:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            phase: Current training phase
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Patience for early stopping
        """
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch(epoch, phase)
            
            # Validate
            val_losses = self.validate()
            
            # Log
            self.logger.info(f"Epoch {epoch} - Train: {train_losses['total_loss']:.6f}, Val: {val_losses['total_loss']:.6f}")
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_total_loss": train_losses["total_loss"],
                    "val_total_loss": val_losses["total_loss"],
                    "train_sn_loss": train_losses.get("sn_loss", 0),
                    "val_sn_loss": val_losses.get("sn_loss", 0),
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    filename=f"checkpoint_epoch_{epoch+1}.pt",
                    metrics={"train_loss": train_losses["total_loss"], "val_loss": val_losses["total_loss"]},
                )
            
            # Early stopping
            if early_stopping_patience is not None:
                if val_losses["total_loss"] < best_val_loss:
                    best_val_loss = val_losses["total_loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Save final checkpoint
        self.save_checkpoint(
            filename="checkpoint_final.pt",
            metrics={"final_train_loss": train_losses["total_loss"], "final_val_loss": val_losses["total_loss"]},
        )
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict[str, float]] = None) -> None:
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "best_val_loss": self.best_val_loss,
            "metrics": metrics or {},
        }
        
        filepath = self.checkpoint_dir / filename
        save_checkpoint(filepath, checkpoint, self.logger)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load checkpoint."""
        filepath = self.checkpoint_dir / filename
        checkpoint = load_checkpoint(filepath, self.logger)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        self.logger.info(f"Loaded checkpoint from {filename}")
