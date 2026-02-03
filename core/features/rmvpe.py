"""RMVPE (Robust Multi-band Pitch Estimation) model loading and F0 extraction."""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global model cache
_rmvpe_model_cache = None


def load_rmvpe_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load RMVPE model for F0 extraction.
    
    Args:
        model_path: Path to RMVPE model file
        device: Device to load model on
        
    Returns:
        Loaded RMVPE model
    """
    global _rmvpe_model_cache
    
    if _rmvpe_model_cache is not None:
        return _rmvpe_model_cache
    
    if device is None:
        from ...utils.device import get_device
        device = get_device()
    
    if model_path is None:
        model_path = "pretrained/rmvpe/rmvpe.pt"
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.warning(
            f"RMVPE model not found at {model_path}. "
            "Falling back to DIO method."
        )
        return None
    
    try:
        logger.info(f"Loading RMVPE model from {model_path}")
        
        # Placeholder: In production, load actual RMVPE model
        # For now, return None to trigger fallback
        logger.warning("RMVPE model loading not fully implemented. Using DIO fallback.")
        
        # In production, you would do:
        # model = torch.load(model_path, map_location=device)
        # model.eval()
        # _rmvpe_model_cache = model
        # return model
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to load RMVPE model: {e}")
        return None


def extract_f0_with_rmvpe(
    audio: np.ndarray,
    model: torch.nn.Module,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Extract F0 using RMVPE model.
    
    Args:
        audio: Audio array
        model: Loaded RMVPE model
        sample_rate: Sample rate
        
    Returns:
        F0 values
    """
    if model is None:
        # Fallback to DIO
        from .f0 import extract_f0_dio
        return extract_f0_dio(audio, sample_rate)
    
    try:
        # Convert audio to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Add batch dimension if needed
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            # Placeholder: In production, run actual RMVPE inference
            # f0 = model(audio_tensor)
            
            # For now, fallback to DIO
            from .f0 import extract_f0_dio
            return extract_f0_dio(audio, sample_rate)
            
    except Exception as e:
        logger.error(f"RMVPE inference failed: {e}")
        from .f0 import extract_f0_dio
        return extract_f0_dio(audio, sample_rate)
