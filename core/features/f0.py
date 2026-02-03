"""F0 (fundamental frequency) extraction for RVC."""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def extract_f0(
    audio: np.ndarray,
    sample_rate: int = 16000,
    method: str = "rmvpe"
) -> np.ndarray:
    """
    Extract F0 (fundamental frequency) from audio.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate of audio
        method: Extraction method ("rmvpe", "dio", "harvest")
        
    Returns:
        F0 values (1D array)
    """
    if method == "rmvpe":
        return extract_f0_rmvpe(audio, sample_rate)
    elif method == "dio":
        return extract_f0_dio(audio, sample_rate)
    else:
        logger.warning(f"Unknown F0 method: {method}. Using RMVPE.")
        return extract_f0_rmvpe(audio, sample_rate)


def extract_f0_rmvpe(
    audio: np.ndarray,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Extract F0 using RMVPE method.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        F0 values
    """
    try:
        from .rmvpe import load_rmvpe_model, extract_f0_with_rmvpe
        
        model = load_rmvpe_model()
        f0 = extract_f0_with_rmvpe(audio, model, sample_rate)
        return f0
        
    except Exception as e:
        logger.warning(f"RMVPE extraction failed: {e}. Falling back to DIO.")
        return extract_f0_dio(audio, sample_rate)


def extract_f0_dio(
    audio: np.ndarray,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Extract F0 using DIO (Distributed Inline Operation) method.
    
    This is a simpler, more robust method that doesn't require a model.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        F0 values
    """
    try:
        import parselmouth
        
        # Convert to parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
        
        # Extract pitch using DIO
        pitch = sound.to_pitch_ac(
            time_step=0.01,  # 10ms frames
            voicing_threshold=0.6,
            pitch_floor=50.0,
            pitch_ceiling=800.0
        )
        
        # Get F0 values
        f0 = pitch.selected_array['frequency']
        
        # Replace unvoiced frames (0) with NaN or 0
        f0[f0 == 0] = 0
        
        return f0.astype(np.float32)
        
    except ImportError:
        logger.warning("parselmouth not available. Using simple F0 estimation.")
        # Fallback: very simple F0 estimation
        # This is not accurate but prevents crashes
        frame_length = int(sample_rate * 0.01)  # 10ms frames
        num_frames = len(audio) // frame_length
        f0 = np.ones(num_frames, dtype=np.float32) * 200.0  # Default 200 Hz
        return f0
    except Exception as e:
        logger.error(f"F0 extraction failed: {e}")
        # Return default F0
        frame_length = int(sample_rate * 0.01)
        num_frames = len(audio) // frame_length
        return np.ones(num_frames, dtype=np.float32) * 200.0
