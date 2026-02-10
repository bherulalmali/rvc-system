"""Dynamic voice registry for discovering and managing trained models."""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class VoiceRegistry:
    """Manages discovery and registration of trained voice models."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the voice registry.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.models_dir / "registry.json"
        self._registry: Dict[str, Dict] = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load registry from disk or initialize empty."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}. Starting fresh.")
                return {}
        return {}
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def discover_voices(self) -> List[str]:
        """
        Scan models directory and discover all trained voices.
        
        Returns:
            List of voice names (person names)
        """
        voices = []
        
        # Scan directory structure: models/<person_name>/
        if not self.models_dir.exists():
            return voices
        
        for person_dir in self.models_dir.iterdir():
            if person_dir.is_dir() and not person_dir.name.startswith("."):
                # Check if this directory contains model files
                model_files = list(person_dir.glob("*.pth"))
                if model_files:
                    voices.append(person_dir.name)
        
        # Also check registry for any registered voices
        for voice_name in self._registry.keys():
            if voice_name not in voices:
                voices.append(voice_name)
        
        return sorted(voices)
    
    def register_voice(
        self,
        person_name: str,
        model_path: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Register a trained voice model.
        
        Args:
            person_name: Unique identifier for the voice
            model_path: Path to the model checkpoint
            metadata: Optional metadata (training date, sample rate, etc.)
        """
        self._registry[person_name] = {
            "model_path": str(model_path),
            "metadata": metadata or {},
        }
        self._save_registry()
        logger.info(f"Registered voice: {person_name}")
    
    def get_voice_info(self, person_name: str) -> Optional[Dict]:
        """
        Get information about a registered voice.
        
        Args:
            person_name: Name of the voice
            
        Returns:
            Dict with voice info or None if not found
        """
        return self._registry.get(person_name)
    
    def get_model_path(self, person_name: str) -> Optional[str]:
        """
        Get the model path for a voice.
        
        Args:
            person_name: Name of the voice
            
        Returns:
            Path to model file or None if not found
        """
        info = self.get_voice_info(person_name)
        if info:
            return info.get("model_path")
        
        # Fallback: check if model exists in expected location
        model_dir = self.models_dir / person_name
        model_files = list(model_dir.glob("*.pth"))
        if model_files:
            return str(model_files[0])
        
        return None
    
    def remove_voice(self, person_name: str) -> bool:
        """
        Remove a voice from the registry.
        
        Args:
            person_name: Name of the voice to remove
            
        Returns:
            True if removed, False if not found
        """
        if person_name in self._registry:
            del self._registry[person_name]
            self._save_registry()
            logger.info(f"Removed voice: {person_name}")
            return True
        return False


def discover_voices(models_dir: str = "models") -> List[str]:
    """
    Convenience function to discover all available voices.
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        List of available voice names
    """
    registry = VoiceRegistry(models_dir)
    return registry.discover_voices()
