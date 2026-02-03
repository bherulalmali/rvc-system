"""Audio utility functions for validation and information extraction."""

import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def validate_audio_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that an audio file is in a supported format and readable.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        # Check file extension
        valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        if path.suffix.lower() not in valid_extensions:
            return False, f"Unsupported format: {path.suffix}. Supported: {valid_extensions}"
        
        # Try to read the file
        data, sr = sf.read(str(path))
        if len(data) == 0:
            return False, "Audio file is empty"
        
        # Check duration (minimum 1 second)
        duration = len(data) / sr
        if duration < 1.0:
            return False, f"Audio too short: {duration:.2f}s (minimum 1s)"
        
        # Check for reasonable sample rate
        if sr < 8000 or sr > 48000:
            logger.warning(f"Unusual sample rate: {sr} Hz")
        
        return True, None
        
    except Exception as e:
        return False, f"Error reading audio file: {str(e)}"


def get_audio_info(file_path: str) -> dict:
    """
    Get information about an audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dict with audio information (sample_rate, duration, channels, etc.)
    """
    try:
        data, sr = sf.read(str(file_path))
        duration = len(data) / sr
        
        # Handle stereo/mono
        if len(data.shape) == 1:
            channels = 1
        else:
            channels = data.shape[1]
        
        return {
            "sample_rate": sr,
            "duration": duration,
            "channels": channels,
            "samples": len(data),
            "dtype": str(data.dtype),
        }
    except Exception as e:
        logger.error(f"Failed to get audio info: {e}")
        return {}


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.
    
    Args:
        audio: Audio array
        
    Returns:
        Normalized audio array
    """
    if audio.max() == 0 and audio.min() == 0:
        return audio
    
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    
    return audio
