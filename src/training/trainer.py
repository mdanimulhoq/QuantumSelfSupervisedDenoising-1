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
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fns = loss_fns
        self.device = device or get_device()
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.use_wandb = use_wandb
        
        set_seed(config.get("seed", 42))
        
        self.logger = setup_logger(
            name="trainer",
            log_dir=self.log_dir,
            use_wandb=use_wandb,
            wandb_project=config.get("wandb_project", "n2ln-qem"),
            wandb_entity=config.get("wandb_entity", None),
            wandb_config=config,
        )
        
        self.model = self.model.to(self.device)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        lr = self.config.get("learning_rate", 3e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        betas = self.config.get("betas", [0.9, 0.999])
        eps = self.config.get("eps", 1e-8)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        self.logger.info(f"Optimizer: AdamW (lr={lr}, wd={weight_decay})")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[SequentialLR]:
        scheduler_config = self.config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine_warmup")
        if scheduler_type == "cosine_warmup":
            warmup_steps = scheduler_config.get("warmup_steps", 1000)
            total_steps = scheduler_config.get("total_steps", 100000)
            min_lr = scheduler_config.get("min_lr", 1e-6)
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
            scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
            self.logger.info(f"Scheduler: Cosine warmup (warmup={warmup_steps}, total={total_steps})")
            return scheduler
        self.logger.info("No scheduler configured")
        return None
    
    def _compute_loss(self, sn_dist, hn_dist, sn_target, hn_target, phase):
        losses = {}
        sn_loss = self.loss_fns["snd"](sn_dist, sn_target)
        losses["sn_loss"] = sn_loss
        
        if phase != "phase1":
            hn_loss = self.loss_fns["hne"](hn_dist, hn_target)
            losses["hn_loss"] = hn_loss
        
        phys_sn = self.loss_fns["physicality"](sn_dist)
        phys_hn = self.loss_fns["physicality"](hn_dist)
        losses["phys_sn"] = phys_sn
        losses["phys_hn"] = phys_hn
        
        if phase == "phase3" and self.loss_fns.get("consistency") is not None:
            consist_loss = self.loss_fns["consistency"](sn_dist, hn_dist)
            losses["consist_loss"] = consist_loss
        
        total_loss = sn_loss + losses.get("hn_loss", 0.0)
        total_loss += 0.1 * (phys_sn + phys_hn)
        if phase == "phase3" and self.loss_fns.get("consistency") is not None:
            total_loss += 0.3 * losses.get("consist_loss", 0.0)
        
        losses["total_loss"] = total_loss
        return losses
    
    def train_step(self, batch, phase):
        batch = to_device(batch, self.device)
        sn_dist, hn_dist = self.model(
            bitstrings=batch["bitstrings"],
            counts=batch["counts"],
            mask=batch.get("mask", None),
            mode=phase,
        )
        loss_dict = self._compute_loss(
            sn_dist=sn_dist,
            hn_dist=hn_dist,
            sn_target=batch["sn_target"],
            hn_target=batch["hn_target"],
            phase=phase,
        )
        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        grad_clip = self.config.get("gradient_clip", 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return {k: v.item() for k, v in loss_dict.items()}
    
    def train_epoch(self, epoch, phase):
        self.model.train()
        epoch_losses = {}
        for batch_idx, batch in enumerate(self.train_loader):
            loss_dict = self.train_step(batch, phase)
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v
            if self.global_step % self.config.get("log_interval", 10) == 0:
                log_metrics(self.logger, loss_dict, step=self.global_step)
            self.global_step += 1
        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    def validate(self):
        self.model.eval()
        val_losses = {}
        with torch.no_grad():
            for batch in self.val_loader:
                batch = to_device(batch, self.device)
                sn_dist, hn_dist = self.model(
                    bitstrings=batch["bitstrings"],
                    counts=batch["counts"],
                    mask=batch.get("mask", None),
                    mode="phase3",
                )
                phase = "phase3" if self.loss_fns.get("consistency") is not None else "phase1"
                loss_dict = self._compute_loss(
                    sn_dist=sn_dist,
                    hn_dist=hn_dist,
                    sn_target=batch["sn_target"],
                    hn_target=batch["hn_target"],
                    phase=phase,
                )
                for k, v in loss_dict.items():
                    if k not in val_losses:
                        val_losses[k] = 0.0
                    val_losses[k] += v.item()
        num_batches = len(self.val_loader)
        return {k: v / num_batches for k, v in val_losses.items()}
    
    def train(self, num_epochs: int, phase: str = "phase1", save_every: int = 10, early_stopping_patience: Optional[int] = None):
        """Main training loop."""
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_losses = self.train_epoch(epoch, phase)
            val_losses = self.validate()
            self.logger.info(f"Epoch {epoch} - Train: {train_losses['total_loss']:.6f}, Val: {val_losses['total_loss']:.6f}")
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_total_loss": train_losses["total_loss"],
                    "val_total_loss": val_losses["total_loss"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", {"train_loss": train_losses["total_loss"], "val_loss": val_losses["total_loss"]})
            if early_stopping_patience is not None:
                if val_losses["total_loss"] < best_val_loss:
                    best_val_loss = val_losses["total_loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        self.save_checkpoint("checkpoint_final.pt", {"final_train_loss": train_losses["total_loss"], "final_val_loss": val_losses["total_loss"]})
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict[str, float]] = None):
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
    
    def load_checkpoint(self, filename: str):
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
