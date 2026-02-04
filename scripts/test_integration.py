"""Integration test to verify project structure."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    print("Testing imports...")
    try:
        from core.training.trainer import train_rvc_model
        from core.inference.converter import VoiceConverter
        from app import app
        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)

def test_mock_training():
    print("\nTesting mock training logic...")
    from core.training.trainer import train_rvc_model
    
    # Create dummy audio file
    import soundfile as sf
    import numpy as np
    
    audio_path = "test_audio.wav"
    sr = 16000
    data = np.random.uniform(-1, 1, sr * 2) # 2 seconds
    sf.write(audio_path, data, sr)
    
    try:
        model_path = train_rvc_model(
            audio_files=[audio_path],
            person_name="test_voice",
            output_dir="models_test"
        )
        print(f"✅ Mock training successful: {model_path}")
        
    except Exception as e:
        print(f"❌ Mock training failed: {e}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
if __name__ == "__main__":
    test_imports()
    test_mock_training()
