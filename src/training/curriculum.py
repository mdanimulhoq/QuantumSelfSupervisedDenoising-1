"""
Curriculum controller for N2LN-QEM (TDD §4.2).
Three-phase training: SN-D only → Joint → Fine-tune.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import torch


class Phase(Enum):
    """Training phases."""
    PHASE1 = "phase1"  # SN-D only
    PHASE2 = "phase2"  # Joint training (add HN-E)
    PHASE3 = "phase3"  # Fine-tune (full model)


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""
    name: str
    epochs: int
    snd_only: bool
    hne_enabled: bool
    consist_enabled: bool
    hne_ramp: bool = False
    lr_factor: float = 1.0
    description: str = ""


@dataclass
class CurriculumConfig:
    """Full curriculum configuration."""
    phases: List[PhaseConfig]
    current_phase: int = 0
    current_epoch: int = 0
    total_epochs: int = 0


class CurriculumController:
    """
    Controls the three-phase curriculum for N2LN-QEM training.
    
    Phase 1: SN-D only (learn shot noise patterns)
    Phase 2: Add HN-E (joint training, ramp HN-E weight)
    Phase 3: Fine-tune (full model with consistency loss)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        total_epochs: Optional[int] = None,
    ):
        """
        Args:
            config: Curriculum configuration from Hydra
            total_epochs: Override total epochs (optional)
        """
        self.config = config
        self.curriculum_config = self._parse_config(config)
        self.total_epochs = total_epochs or sum(p.epochs for p in self.curriculum_config.phases)
        self.current_epoch = 0
        self.current_phase_idx = 0
        self.phase_epoch_counter = 0
        self.hne_weight = 0.0
        self.consist_weight = 0.0
        
        # Get initial phase
        self._update_phase()
    
    def _parse_config(self, config: Dict[str, Any]) -> CurriculumConfig:
        """Parse configuration into phase objects."""
        curriculum = config.get("curriculum", {})
        
        phases = []
        phase_configs = curriculum.get("phases", {})
        
        # Default phase configs if not provided
        default_phases = [
            PhaseConfig(
                name="phase1",
                epochs=100,
                snd_only=True,
                hne_enabled=False,
                consist_enabled=False,
                description="SN-D only: learn shot noise patterns",
            ),
            PhaseConfig(
                name="phase2",
                epochs=150,
                snd_only=False,
                hne_enabled=True,
                consist_enabled=False,
                hne_ramp=True,
                description="Joint training: add HN-E head",
            ),
            PhaseConfig(
                name="phase3",
                epochs=50,
                snd_only=False,
                hne_enabled=True,
                consist_enabled=True,
                lr_factor=0.1,
                description="Fine-tune: full model with consistency",
            ),
        ]
        
        # Use default if no config provided
        if not phase_configs:
            phases = default_phases
        else:
            for phase_name, phase_cfg in phase_configs.items():
                phases.append(PhaseConfig(
                    name=phase_name,
                    epochs=phase_cfg.get("epochs", 100),
                    snd_only=phase_cfg.get("snd_only", False),
                    hne_enabled=phase_cfg.get("hne_enabled", True),
                    consist_enabled=phase_cfg.get("consist_enabled", False),
                    hne_ramp=phase_cfg.get("hne_ramp", False),
                    lr_factor=phase_cfg.get("lr_factor", 1.0),
                    description=phase_cfg.get("description", ""),
                ))
        
        return CurriculumConfig(phases=phases)
    
    def _update_phase(self) -> None:
        """Update current phase based on epoch."""
        total_epochs = 0
        for idx, phase in enumerate(self.curriculum_config.phases):
            if self.current_epoch < total_epochs + phase.epochs:
                self.current_phase_idx = idx
                self.phase_epoch_counter = self.current_epoch - total_epochs
                break
            total_epochs += phase.epochs
        else:
            # Last phase
            self.current_phase_idx = len(self.curriculum_config.phases) - 1
            self.phase_epoch_counter = self.curriculum_config.phases[-1].epochs
    
    def step(self) -> None:
        """Advance to next epoch."""
        self.current_epoch += 1
        self._update_phase()
        self._update_weights()
    
    def _update_weights(self) -> None:
        """Update HN-E and consistency weights based on phase."""
        phase = self.get_current_phase()
        progress = self.get_phase_progress()
        
        if phase.hne_ramp and phase.hne_enabled:
            # Ramp HN-E weight from 0 to 1 over the phase
            self.hne_weight = min(1.0, progress * 2.0)  # Ramp over first 50% of phase
        elif phase.hne_enabled:
            self.hne_weight = 1.0
        else:
            self.hne_weight = 0.0
        
        if phase.consist_enabled:
            # Start consistency loss in phase3
            self.consist_weight = 1.0 * min(1.0, progress * 3.0)  # Gradual ramp
        else:
            self.consist_weight = 0.0
    
    def get_current_phase(self) -> PhaseConfig:
        """Get current phase configuration."""
        return self.curriculum_config.phases[self.current_phase_idx]
    
    def get_phase_name(self) -> str:
        """Get current phase name."""
        return self.get_current_phase().name
    
    def get_phase_progress(self) -> float:
        """Get progress through current phase (0.0 to 1.0)."""
        phase = self.get_current_phase()
        if phase.epochs == 0:
            return 1.0
        return min(1.0, self.phase_epoch_counter / phase.epochs)
    
    def get_overall_progress(self) -> float:
        """Get overall training progress (0.0 to 1.0)."""
        return min(1.0, self.current_epoch / self.total_epochs)
    
    def get_training_mode(self) -> Dict[str, Any]:
        """
        Get training mode configuration for the model.
        
        Returns:
            Dictionary with mode settings:
                - mode: 'phase1', 'phase2', 'phase3'
                - snd_enabled: bool
                - hne_enabled: bool
                - consist_enabled: bool
                - hne_weight: float
                - consist_weight: float
                - lr_factor: float
        """
        phase = self.get_current_phase()
        
        return {
            "mode": phase.name,
            "snd_enabled": True,  # Always enabled
            "hne_enabled": phase.hne_enabled,
            "consist_enabled": phase.consist_enabled,
            "hne_weight": self.hne_weight,
            "consist_weight": self.consist_weight,
            "lr_factor": phase.lr_factor,
            "phase_epoch": self.phase_epoch_counter,
            "phase_total": phase.epochs,
            "overall_progress": self.get_overall_progress(),
        }
    
    def should_enable_hne(self) -> bool:
        """Check if HN-E should be enabled."""
        return self.get_current_phase().hne_enabled
    
    def should_enable_consistency(self) -> bool:
        """Check if consistency loss should be enabled."""
        return self.get_current_phase().consist_enabled
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        Get current loss weights.
        
        Returns:
            Dictionary of loss weights
        """
        return {
            "snd_weight": 1.0,
            "hne_weight": self.hne_weight,
            "consist_weight": self.consist_weight,
            "phys_weight": 0.1,
        }
    
    def get_phase_info(self) -> Dict[str, Any]:
        """
        Get detailed phase information for logging.
        
        Returns:
            Dictionary with phase info
        """
        phase = self.get_current_phase()
        return {
            "phase": phase.name,
            "phase_idx": self.current_phase_idx,
            "phase_epoch": self.phase_epoch_counter,
            "phase_total": phase.epochs,
            "phase_progress": self.get_phase_progress(),
            "overall_progress": self.get_overall_progress(),
            "total_epochs": self.total_epochs,
            "current_epoch": self.current_epoch,
            "snd_only": phase.snd_only,
            "hne_enabled": phase.hne_enabled,
            "hne_weight": self.hne_weight,
            "consist_enabled": phase.consist_enabled,
            "consist_weight": self.consist_weight,
            "lr_factor": phase.lr_factor,
            "description": phase.description,
        }
    
    def reset(self) -> None:
        """Reset curriculum to beginning."""
        self.current_epoch = 0
        self.current_phase_idx = 0
        self.phase_epoch_counter = 0
        self.hne_weight = 0.0
        self.consist_weight = 0.0
        self._update_phase()
        self._update_weights()


class CurriculumScheduler:
    """
    Scheduler that adjusts training parameters based on curriculum phase.
    """
    
    def __init__(
        self,
        curriculum: CurriculumController,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Args:
            curriculum: Curriculum controller
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
        """
        self.curriculum = curriculum
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.phase_history = []
    
    def step(self) -> None:
        """Step the curriculum and update parameters."""
        self.curriculum.step()
        
        # Adjust learning rate based on phase
        mode = self.curriculum.get_training_mode()
        lr_factor = mode.get("lr_factor", 1.0)
        
        # Apply LR factor for fine-tuning phase
        if lr_factor < 1.0:
            for param_group in self.optimizer.param_groups:
                base_lr = param_group.get("base_lr", param_group["lr"])
                if "base_lr" not in param_group:
                    param_group["base_lr"] = param_group["lr"]
                param_group["lr"] = base_lr * lr_factor
        
        # Log phase transition
        if self.phase_history:
            last_phase = self.phase_history[-1]
            current_phase = self.curriculum.get_phase_name()
            if last_phase != current_phase:
                self.phase_history.append(current_phase)
        else:
            self.phase_history.append(self.curriculum.get_phase_name())
    
    def get_info(self) -> Dict[str, Any]:
        """Get scheduler info for logging."""
        return self.curriculum.get_phase_info()
