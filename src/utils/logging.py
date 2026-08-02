"""
Logging utilities: wandb integration and local file logger.
"""
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import wandb

def setup_logger(
    name: str = "n2ln-qem",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    use_wandb: bool = False,
    wandb_project: str = "n2ln-qem",
    wandb_entity: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """
    Setup logger with both file and console handlers.
    Optionally initialize wandb.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        use_wandb: Whether to use wandb
        wandb_project: Wandb project name
        wandb_entity: Wandb entity/username
        wandb_config: Config dict to log to wandb
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"log_{timestamp}.txt"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    # Wandb
    if use_wandb:
        try:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=wandb_config or {},
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info(f"Wandb initialized: {wandb_project}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    return logger

def log_metrics(logger: logging.Logger, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log metrics to both file logger and wandb.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        step: Optional step number
    """
    # File logging
    step_str = f" [step {step}]" if step is not None else ""
    logger.info(f"Metrics{step_str}: {json.dumps(metrics, default=str)}")
    
    # Wandb logging
    if wandb.run is not None:
        wandb.log(metrics, step=step)

def save_checkpoint(
    filepath: Path,
    state: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        filepath: Path to save checkpoint
        state: Dictionary containing model state, optimizer state, etc.
        logger: Optional logger for info
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    if logger:
        logger.info(f"Checkpoint saved: {filepath}")

def load_checkpoint(
    filepath: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        logger: Optional logger for info
    
    Returns:
        dict: Loaded checkpoint state
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    state = torch.load(filepath, map_location="cpu")
    if logger:
        logger.info(f"Checkpoint loaded: {filepath}")
    return state
