"""Device detection and management for GPU-agnostic execution."""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Automatically detect and return the best available device.
    
    Priority:
    1. CUDA GPU (if available and requested)
    2. Apple MPS (if available and requested)
    3. CPU (fallback)
    
    Args:
        device_preference: "auto", "cuda", "mps", or "cpu"
        
    Returns:
        torch.device: The selected device
    """
    if device_preference == "cpu":
        return torch.device("cpu")
    
    if device_preference == "cuda" or device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
    
    if device_preference == "mps" or device_preference == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    
    return torch.device("cpu")


def log_device_info(device: torch.device) -> None:
    """
    Log detailed information about the selected device.
    
    Args:
        device: The torch device to log info for
    """
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == "mps":
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        logger.info("Using CPU (no GPU acceleration)")


def get_device_name(device: torch.device) -> str:
    """
    Get a human-readable device name.
    
    Args:
        device: The torch device
        
    Returns:
        str: Human-readable device name
    """
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device.type == "mps":
        return "Apple MPS"
    else:
        return "CPU"
