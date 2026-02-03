"""RVC model training pipeline."""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from ..model import RVCModel, save_model_checkpoint
from ..audio import load_audio, prepare_dataset
from ..features import extract_hubert_features, extract_f0, load_hubert_model
from ...utils.device import get_device, log_device_info

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for RVC training."""
    
    def __init__(
        self,
        batch_size: int = 4,
        learning_rate: float = 0.0001,
        epochs: int = 50,
        save_interval: int = 10,
        log_interval: int = 10,
        sample_rate: int = 16000,
        n_mel: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.sample_rate = sample_rate
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.hop_length = hop_length


def create_dataloader(
    audio_files: List[str],
    hubert_model: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device
) -> List[Dict[str, torch.Tensor]]:
    """
    Create dataset from audio files.
    
    Args:
        audio_files: List of audio file paths
        hubert_model: HuBERT model for feature extraction
        config: Training configuration
        device: Device to run on
        
    Returns:
        List of data samples (features, f0, mel)
    """
    dataset = []
    
    logger.info(f"Processing {len(audio_files)} audio files...")
    
    for audio_file in tqdm(audio_files, desc="Extracting features"):
        try:
            # Load audio
            audio, sr = load_audio(audio_file, sample_rate=config.sample_rate)
            
            # Extract HuBERT features
            features = extract_hubert_features(audio, hubert_model, device)
            
            # Extract F0
            f0 = extract_f0(audio, config.sample_rate, method="rmvpe")
            
            # Compute mel spectrogram (target)
            import librosa
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=config.sample_rate,
                n_mels=config.n_mel,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
            )
            mel = librosa.power_to_db(mel, ref=np.max)
            
            # Convert to tensors
            features_tensor = torch.from_numpy(features).float()
            f0_tensor = torch.from_numpy(f0).float()
            mel_tensor = torch.from_numpy(mel).float()
            
            # Ensure compatible lengths
            min_len = min(features_tensor.shape[0], f0_tensor.shape[0], mel_tensor.shape[1])
            features_tensor = features_tensor[:min_len]
            f0_tensor = f0_tensor[:min_len]
            mel_tensor = mel_tensor[:, :min_len]
            
            dataset.append({
                "features": features_tensor,
                "f0": f0_tensor,
                "mel": mel_tensor,
            })
            
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            continue
    
    logger.info(f"Created dataset with {len(dataset)} samples")
    return dataset


def train_rvc_model(
    audio_files: List[str],
    person_name: str,
    output_dir: str = "models",
    config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    progress_callback: Optional[callable] = None
) -> str:
    """
    Train an RVC model on audio files.
    
    Args:
        audio_files: List of audio file paths for training
        person_name: Unique identifier for the voice
        output_dir: Directory to save model
        config: Training configuration (uses defaults if None)
        device: Device to train on (auto-detected if None)
        progress_callback: Optional callback(epoch, loss, status) for progress updates
        
    Returns:
        Path to saved model checkpoint
    """
    if config is None:
        config = TrainingConfig()
    
    if device is None:
        device = get_device()
    
    log_device_info(device)
    
    # Setup output directory
    model_dir = Path(output_dir) / person_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training for voice: {person_name}")
    logger.info(f"Output directory: {model_dir}")
    
    # Load HuBERT model
    logger.info("Loading HuBERT model...")
    hubert_model = load_hubert_model(device=device)
    
    # Create dataset
    if progress_callback:
        progress_callback(0, 0.0, "Preparing dataset...")
    
    dataset = create_dataloader(audio_files, hubert_model, config, device)
    
    if len(dataset) == 0:
        raise ValueError("No valid audio samples found. Please check your audio files.")
    
    # Initialize model
    model = RVCModel(
        n_mel=config.n_mel,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    ).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    best_loss = float("inf")
    
    logger.info(f"Starting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        epoch_losses = []
        
        # Simple batching (in production, use proper DataLoader)
        for sample in dataset:
            features = sample["features"].unsqueeze(0).to(device)
            f0 = sample["f0"].unsqueeze(0).to(device)
            mel_target = sample["mel"].unsqueeze(0).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            mel_pred = model(features, f0)
            
            # Loss
            loss = criterion(mel_pred, mel_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        
        # Logging
        if (epoch + 1) % config.log_interval == 0:
            logger.info(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")
        
        # Progress callback
        if progress_callback:
            progress_callback(epoch + 1, avg_loss, f"Training epoch {epoch+1}/{config.epochs}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0 or avg_loss < best_loss:
            checkpoint_path = model_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_model_checkpoint(
                model,
                optimizer,
                epoch + 1,
                avg_loss,
                str(checkpoint_path),
                metadata={
                    "person_name": person_name,
                    "num_samples": len(dataset),
                }
            )
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save as best model
                best_model_path = model_dir / "model.pth"
                save_model_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    avg_loss,
                    str(best_model_path),
                    metadata={
                        "person_name": person_name,
                        "num_samples": len(dataset),
                        "is_best": True,
                    }
                )
    
    logger.info(f"Training completed! Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to: {model_dir}")
    
    # Return path to best model
    best_model_path = model_dir / "model.pth"
    return str(best_model_path)
