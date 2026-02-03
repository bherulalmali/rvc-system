"""Model checkpoint saving and loading."""

import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    checkpoint_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch number
        loss: Current loss value
        checkpoint_path: Path to save checkpoint
        metadata: Optional metadata to save
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "n_mel": model.n_mel if hasattr(model, "n_mel") else 80,
            "n_fft": model.n_fft if hasattr(model, "n_fft") else 512,
            "hop_length": model.hop_length if hasattr(model, "hop_length") else 160,
            "feature_dim": model.feature_dim if hasattr(model, "feature_dim") else 768,
            "hidden_dim": model.hidden_dim if hasattr(model, "hidden_dim") else 256,
        },
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if metadata:
        checkpoint["metadata"] = metadata
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")


def load_model_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    return_optimizer: bool = False
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Optional model to load state into
        device: Device to load checkpoint on
        return_optimizer: Whether to return optimizer state
        
    Returns:
        Dict containing checkpoint data
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        from ...utils.device import get_device
        device = get_device()
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        logger.info("Model state loaded successfully")
    
    result = {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "model_config": checkpoint.get("model_config", {}),
        "metadata": checkpoint.get("metadata", {}),
    }
    
    if return_optimizer and "optimizer_state_dict" in checkpoint:
        result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
    
    return result
