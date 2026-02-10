"""RVC voice conversion pipeline using rvc-python."""

import logging
from pathlib import Path
from typing import Optional

try:
    from rvc_python.infer import RVCInference
    HAS_RVC = True
except ImportError:
    HAS_RVC = False

logger = logging.getLogger(__name__)

class VoiceConverter:
    """RVC Voice Converter wrapper."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize voice converter.
        
        Args:
            model_path: Path to RVC model (.pth)
            device: Device to use (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device if device else "auto"
        
        if HAS_RVC:
            # Initialize RVC inference
            self.rvc = RVCInference(
                device=self.device,
                is_half=True if self.device != "cpu" else False # Use half precision on GPU
            )
        else:
            logger.warning("RVC library not found. Using mock converter.")
            self.rvc = None
        
    def convert(
        self,
        source_audio_path: str,
        output_path: str,
        pitch_shift: float = 0.0,
        f0_method: str = "rmvpe",
        index_path: Optional[str] = None,
        index_rate: float = 0.75,
    ) -> str:
        """
        Convert voice.
        
        Args:
            source_audio_path: Path to source audio
            output_path: Path to save output
            pitch_shift: Pitch shift in semitones
            f0_method: Method for F0 extraction (rmvpe, pm, harvest)
            index_path: Path to feature index file (optional)
            index_rate: Rate of index influence
            
        Returns:
            Path to output file
        """
        try:
            logger.info(f"Converting {source_audio_path} using model {self.model_path}")
            
            if self.rvc:
                # rvc-python infer method signature might vary slightly by version, 
                # but generally follows this pattern
                self.rvc.load_checkpoint(self.model_path, index_path)
                
                self.rvc.infer_file(
                    input_path=source_audio_path,
                    output_path=output_path,
                    f0_up_key=int(pitch_shift),
                    f0_method=f0_method,
                    index_rate=index_rate
                )
            else:
                # Mock conversion
                import shutil
                shutil.copy(source_audio_path, output_path)
                logger.warning("Mock conversion: Input copied to output (RVC libs missing). Result will sound identical to source.")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Voice conversion failed: {e}")
            raise
