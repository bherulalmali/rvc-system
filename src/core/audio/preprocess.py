"""Audio preprocessing for RVC training and inference."""

import logging
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_audio(
    file_path: str,
    sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (default 16000)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_array, actual_sample_rate)
    """
    try:
        # Use librosa for robust loading and resampling
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=mono)
        
        # Ensure float32 format
        audio = audio.astype(np.float32)
        
        logger.debug(f"Loaded audio: {file_path}, shape: {audio.shape}, sr: {sr}")
        return audio, sr
        
    except Exception as e:
        logger.error(f"Failed to load audio {file_path}: {e}")
        raise


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def preprocess_audio(
    file_path: str,
    output_dir: str,
    sample_rate: int = 16000,
    trim_silence: bool = True,
    normalize: bool = True
) -> str:
    """
    Preprocess audio file for training.
    
    This function:
    1. Loads and resamples audio
    2. Converts to mono
    3. Trims silence (optional)
    4. Normalizes (optional)
    5. Saves preprocessed file
    
    Args:
        file_path: Path to input audio file
        output_dir: Directory to save preprocessed audio
        sample_rate: Target sample rate
        trim_silence: Whether to trim leading/trailing silence
        normalize: Whether to normalize audio
        
    Returns:
        Path to preprocessed audio file
    """
    from ..audio.utils import normalize_audio
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    audio, sr = load_audio(file_path, sample_rate=sample_rate, mono=True)
    
    # Trim silence
    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Normalize
    if normalize:
        audio = normalize_audio(audio)
    
    # Save preprocessed audio
    output_path = output_dir / f"{Path(file_path).stem}_preprocessed.wav"
    sf.write(str(output_path), audio, sample_rate)
    
    logger.info(f"Preprocessed audio saved to: {output_path}")
    return str(output_path)


def prepare_dataset(
    audio_files: List[str],
    output_dir: str,
    person_name: str,
    sample_rate: int = 16000
) -> List[str]:
    """
    Prepare a dataset from multiple audio files for training.
    
    Args:
        audio_files: List of audio file paths
        output_dir: Directory to save preprocessed files
        person_name: Name of the person/speaker
        sample_rate: Target sample rate
        
    Returns:
        List of paths to preprocessed audio files
    """
    dataset_dir = Path(output_dir) / person_name / "audio"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessed_files = []
    
    for i, audio_file in enumerate(audio_files):
        try:
            preprocessed_path = preprocess_audio(
                audio_file,
                str(dataset_dir),
                sample_rate=sample_rate
            )
            preprocessed_files.append(preprocessed_path)
            logger.info(f"Processed {i+1}/{len(audio_files)}: {audio_file}")
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
    
    logger.info(f"Dataset prepared: {len(preprocessed_files)} files in {dataset_dir}")
    return preprocessed_files
