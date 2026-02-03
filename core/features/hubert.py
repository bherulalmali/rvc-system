"""HuBERT feature extraction for RVC."""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_hubert_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load HuBERT model for feature extraction.
    
    Args:
        model_path: Path to HuBERT model file (default: pretrained/hubert/hubert_base.pt)
        device: Device to load model on
        
    Returns:
        Loaded HuBERT model
    """
    if device is None:
        from ...utils.device import get_device
        device = get_device()
    
    if model_path is None:
        model_path = "pretrained/hubert/hubert_base.pt"
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"HuBERT model not found at {model_path}. "
            "Please download it first using scripts/download_pretrained.py"
        )
    
    try:
        # Load HuBERT model
        # Note: This is a placeholder - actual implementation depends on model format
        # For now, we'll create a mock structure
        logger.info(f"Loading HuBERT model from {model_path}")
        
        # In a real implementation, you would load the actual HuBERT model here
        # For example, using fairseq or transformers library
        # model = torch.hub.load('facebookresearch/fairseq', 'hubert_base')
        
        # Placeholder: Return a mock model structure
        # In production, replace this with actual model loading
        logger.warning("Using placeholder HuBERT model. Replace with actual implementation.")
        
        class PlaceholderHuBERT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.device = device
            
            def forward(self, audio):
                # Placeholder: return random features
                # In production, this should extract actual HuBERT features
                batch_size = audio.shape[0] if len(audio.shape) > 1 else 1
                seq_len = audio.shape[-1] // 320  # Approximate downsampling
                return torch.randn(batch_size, 768, seq_len, device=self.device)
        
        model = PlaceholderHuBERT().to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load HuBERT model: {e}")
        raise


def extract_hubert_features(
    audio: np.ndarray,
    model: torch.nn.Module,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Extract HuBERT features from audio.
    
    Args:
        audio: Audio array (1D numpy array)
        model: Loaded HuBERT model
        device: Device to run inference on
        
    Returns:
        HuBERT features (shape: [seq_len, feature_dim])
    """
    if device is None:
        from ...utils.device import get_device
        device = get_device()
    
    # Convert to tensor
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float().to(device)
    else:
        audio_tensor = audio.float().to(device)
    
    # Add batch dimension if needed
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Extract features
    with torch.no_grad():
        features = model(audio_tensor)
    
    # Remove batch dimension and convert to numpy
    if len(features.shape) == 3:
        features = features.squeeze(0)
    
    features_np = features.cpu().numpy()
    
    # Transpose if needed: [feature_dim, seq_len] -> [seq_len, feature_dim]
    if features_np.shape[0] < features_np.shape[1]:
        features_np = features_np.T
    
    return features_np
