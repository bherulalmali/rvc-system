"""Voice conversion inference."""

import logging
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, Tuple

from ..model import RVCModel, load_model_checkpoint
from ..audio import load_audio
from ..features import extract_hubert_features, extract_f0, load_hubert_model
from ...utils.device import get_device

logger = logging.getLogger(__name__)


class VoiceConverter:
    """Voice converter for inference."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        sample_rate: int = 16000
    ):
        """
        Initialize voice converter.
        
        Args:
            model_path: Path to trained RVC model
            device: Device to run on (auto-detected if None)
            sample_rate: Target sample rate
        """
        if device is None:
            device = get_device()
        
        self.device = device
        self.sample_rate = sample_rate
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = load_model_checkpoint(model_path, device=device)
        model_config = checkpoint["model_config"]
        
        self.model = RVCModel(
            n_mel=model_config.get("n_mel", 80),
            n_fft=model_config.get("n_fft", 512),
            hop_length=model_config.get("hop_length", 160),
            feature_dim=model_config.get("feature_dim", 768),
            hidden_dim=model_config.get("hidden_dim", 256),
        ).to(device)
        
        load_model_checkpoint(model_path, model=self.model, device=device)
        self.model.eval()
        
        # Load HuBERT model
        self.hubert_model = load_hubert_model(device=device)
        
        logger.info("Voice converter initialized")
    
    def convert(
        self,
        source_audio_path: str,
        output_path: Optional[str] = None,
        pitch_shift: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Convert voice from source audio.
        
        Args:
            source_audio_path: Path to source audio file
            output_path: Optional path to save output
            pitch_shift: Pitch shift in semitones (0 = no change)
            
        Returns:
            Tuple of (converted_audio, sample_rate)
        """
        # Load source audio
        audio, sr = load_audio(source_audio_path, sample_rate=self.sample_rate)
        
        # Extract features
        logger.info("Extracting features...")
        features = extract_hubert_features(audio, self.hubert_model, self.device)
        f0 = extract_f0(audio, self.sample_rate, method="rmvpe")
        
        # Apply pitch shift if needed
        if pitch_shift != 0.0:
            f0 = f0 * (2 ** (pitch_shift / 12))
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        f0_tensor = torch.from_numpy(f0).float().unsqueeze(0).to(self.device)
        
        # Run inference
        logger.info("Running voice conversion...")
        with torch.no_grad():
            mel_converted = self.model.inference(features_tensor, f0_tensor)
        
        # Convert mel to audio (vocoder)
        # In production, use a proper vocoder (HiFi-GAN, etc.)
        # For now, use Griffin-Lim as placeholder
        mel_np = mel_converted.squeeze(0).cpu().numpy()
        mel_db = librosa.db_to_power(mel_np)
        
        # Griffin-Lim vocoder
        audio_converted = librosa.feature.inverse.mel_to_stft(
            mel_db,
            sr=self.sample_rate,
            n_fft=512,
            hop_length=160,
        )
        audio_converted = librosa.griffinlim(
            audio_converted,
            n_iter=32,
            hop_length=160,
            length=len(audio),
        )
        
        # Normalize
        audio_converted = audio_converted / np.abs(audio_converted).max()
        audio_converted = audio_converted.astype(np.float32)
        
        # Save if output path provided
        if output_path:
            sf.write(output_path, audio_converted, self.sample_rate)
            logger.info(f"Saved converted audio to {output_path}")
        
        return audio_converted, self.sample_rate


def convert_voice(
    source_audio_path: str,
    model_path: str,
    output_path: str,
    device: Optional[torch.device] = None,
    pitch_shift: float = 0.0
) -> str:
    """
    Convenience function for voice conversion.
    
    Args:
        source_audio_path: Path to source audio
        model_path: Path to trained RVC model
        output_path: Path to save converted audio
        device: Device to run on (auto-detected if None)
        pitch_shift: Pitch shift in semitones
        
    Returns:
        Path to output file
    """
    converter = VoiceConverter(model_path, device=device)
    converter.convert(source_audio_path, output_path, pitch_shift)
    return output_path
