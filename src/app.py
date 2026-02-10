import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np

from core.training.trainer import train_rvc_model, TrainingConfig
from core.inference.converter import VoiceConverter
from utils.device import get_device, log_device_info, get_device_name
from utils.registry import VoiceRegistry, discover_voices
from core.audio.utils import validate_audio_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize device
device = get_device()
log_device_info(device)
device_name = get_device_name(device)

# Initialize voice registry
registry = VoiceRegistry(models_dir="models/finetuned_models")


def train_voice_ui(
    audio_files: List[str],
    person_name: str,
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 0.0001,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Gradio interface for training a new voice.
    
    Args:
        audio_files: List of uploaded audio file paths
        person_name: Unique identifier for the voice
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (status_message, logs)
    """
    if not audio_files:
        return "❌ Error: Please upload at least one audio file.", ""
    
    if not person_name or not person_name.strip():
        return "❌ Error: Please enter a person/speaker name.", ""
    
    person_name = person_name.strip()
    
    # Convert Gradio file objects to paths
    audio_paths = []
    for f in audio_files:
        if hasattr(f, "name"):
            audio_paths.append(f.name)
        else:
            audio_paths.append(str(f))
            
    audio_files = audio_paths
    
    # Validate audio files
    invalid_files = []
    for audio_file in audio_files:
        is_valid, error_msg = validate_audio_file(audio_file)
        if not is_valid:
            invalid_files.append(f"{Path(audio_file).name}: {error_msg}")
    
    if invalid_files:
        error_msg = "❌ Invalid audio files:\n" + "\n".join(invalid_files)
        return error_msg, ""
    
    # Progress callback
    logs = []
    
    def progress_callback(epoch: int, loss: float, status: str):
        log_msg = f"Epoch {epoch}: Loss = {loss:.4f} - {status}"
        logs.append(log_msg)
        progress(epoch / epochs, desc=status)
    
    try:
        # Create training config
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        
        # Train model
        logger.info(f"Starting training for: {person_name}")
        model_path = train_rvc_model(
            audio_files=audio_files,
            person_name=person_name,
            output_dir="models/finetuned_models",
            config=config,
            device=device,
            progress_callback=progress_callback,
        )
        
        # Register voice
        registry.register_voice(
            person_name=person_name,
            model_path=model_path,
            metadata={
                "num_files": len(audio_files),
                "epochs": epochs,
                "device": str(device),
            }
        )
        
        success_msg = (
            f"✅ Training completed successfully!\n\n"
            f"Voice: {person_name}\n"
            f"Model saved to: {model_path}\n"
            f"Device used: {device_name}\n\n"
            f"The voice '{person_name}' is now available in the Inference tab."
        )
        
        log_output = "\n".join(logs)
        
        return success_msg, log_output
        
    except Exception as e:
        error_msg = f"❌ Training failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, "\n".join(logs)


def get_available_voices() -> List[str]:
    """Get list of available voices for dropdown."""
    voices = discover_voices(models_dir="models/finetuned_models")
    return voices if voices else ["No voices available - train a voice first"]


def convert_voice_ui(
    source_audio: str,
    target_voice: str,
    pitch_shift: float = 0.0
) -> Tuple[Optional[str], str]:
    """
    Gradio interface for voice conversion.
    
    Args:
        source_audio: Path to source audio file
        target_voice: Name of target voice
        pitch_shift: Pitch shift in semitones
        
    Returns:
        Tuple of (output_audio_path, status_message)
    """
    if not source_audio:
        return None, "❌ Error: Please upload a source audio file."
    
    if not target_voice or target_voice == "No voices available - train a voice first":
        return None, "❌ Error: Please select a target voice (or train one first)."
    
    # Validate source audio
    is_valid, error_msg = validate_audio_file(source_audio)
    if not is_valid:
        return None, f"❌ Error: {error_msg}"
    
    # Get model path
    model_path = registry.get_model_path(target_voice)
    if not model_path or not Path(model_path).exists():
        return None, f"❌ Error: Model not found for voice '{target_voice}'."
    
    try:
        # Create person-specific output folder
        output_dir = Path("data/outputs") / target_voice
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(source_audio).stem}_{target_voice}.wav"
        
        # Auto-detect index file in the same directory as the model
        index_path = None
        model_dir = Path(model_path).parent
        index_files = list(model_dir.glob("*.index"))
        if index_files:
            index_path = str(index_files[0])
            logger.info(f"Auto-detected index file: {index_path}")
        
        # Convert voice
        logger.info(f"Converting voice: {source_audio} -> {target_voice}")
        converter = VoiceConverter(model_path, device=device)
        converter.convert(
            source_audio_path=source_audio,
            output_path=str(output_path),
            pitch_shift=pitch_shift,
            index_path=index_path
        )
        
        success_msg = (
            f"✅ Voice conversion completed!\n\n"
            f"Source: {Path(source_audio).name}\n"
            f"Target voice: {target_voice}\n"
            f"Output saved to: {output_path}\n"
            f"Device used: {device_name}"
        )
        
        return str(output_path), success_msg
        
    except Exception as e:
        error_msg = f"❌ Conversion failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg


def refresh_voices() -> gr.Dropdown:
    """Refresh the voice dropdown with latest voices."""
    voices = get_available_voices()
    return gr.Dropdown(choices=voices, value=voices[0] if voices else None)


# Build Gradio interface
with gr.Blocks(title="RVC Voice Cloning System") as app:
    gr.Markdown(
        f"""
        # 🎤 RVC Voice Cloning System
        
        **Device:** {device_name}
        
        This system allows you to train custom voice models and convert audio to any trained voice.
        All processing happens locally on your device.
        """
    )
    
    with gr.Tabs():
        # Training Tab
        with gr.Tab("🎓 Training"):
            gr.Markdown(
                """
                ## Train a New Voice
                
                Upload audio files of the person you want to clone, enter a unique name, and start training.
                The model will be saved and automatically available for inference.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    audio_input = gr.File(
                        file_count="multiple",
                        label="Upload Audio Files",
                        file_types=["audio"],
                    )
                    person_name_input = gr.Textbox(
                        label="Person/Speaker Name",
                        placeholder="e.g., 'John_Doe' or 'Narrator'",
                        info="Unique identifier for this voice"
                    )
                    
                    epochs_input = gr.Slider(
                        minimum=10,
                        maximum=5000,
                        value=50,
                        step=10,
                        label="Epochs",
                        info="Number of training epochs (10-5000)"
                    )
                    
                    # Mock Mode Warning
                    try:
                        from core.training.trainer import HA_RVC
                        if not HA_RVC:
                            gr.Markdown(
                                """
                                > [!WARNING]
                                > **Running in Mock Mode**
                                > RVC dependencies are missing on this system. Training and inference will be simulated.
                                > Please use Google Colab for real results.
                                """
                            )
                    except ImportError:
                        pass
                    
                    with gr.Accordion("Advanced Training Settings", open=False):
                        batch_size_input = gr.Slider(
                            minimum=1,
                            maximum=16,
                            value=4,
                            step=1,
                            label="Batch Size",
                            info="Training batch size"
                        )
                        learning_rate_input = gr.Number(
                            value=0.0001,
                            label="Learning Rate",
                            info="Training learning rate"
                        )
                    
                    train_button = gr.Button("🚀 Train Model", variant="primary", size="lg")
                
                with gr.Column():
                    training_status = gr.Textbox(
                        label="Training Status",
                        lines=10,
                        interactive=False,
                    )
                    training_logs = gr.Textbox(
                        label="Training Logs",
                        lines=15,
                        interactive=False,
                    )
            
            # Wrapper to update dropdown after training
            def train_and_update(*args):
                status, logs = train_voice_ui(*args)
                new_voices = get_available_voices()
                # Return status, logs, and updated dropdown
                return status, logs, gr.Dropdown.update(choices=new_voices, value=args[1]) # Select the new voice

            # train_button.click moved to end of file to access voice_dropdown

        
        # Inference Tab
        with gr.Tab("🎭 Inference"):
            gr.Markdown(
                """
                ## Voice Conversion
                
                Upload source audio and select a target voice to convert.
                The converted audio will be generated using the selected voice model.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    source_audio_input = gr.Audio(
                        label="Source Audio",
                        type="filepath",
                        # sources=["upload", "microphone"], # Gradio 3 default includes both or use source="upload"
                    )
                    
                    voice_dropdown = gr.Dropdown(
                        choices=get_available_voices(),
                        label="Target Voice",
                        info="Select the voice to convert to",
                    )
                    
                    refresh_button = gr.Button("🔄 Refresh Voices", size="sm")
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        pitch_shift_input = gr.Slider(
                            minimum=-12,
                            maximum=12,
                            value=0,
                            step=1,
                            label="Pitch Shift (semitones)",
                            info="Shift pitch up or down"
                        )
                    
                    convert_button = gr.Button("🎯 Convert Voice", variant="primary", size="lg")
                
                with gr.Column():
                    output_audio = gr.Audio(
                        label="Converted Audio",
                        type="filepath",
                    )
                    conversion_status = gr.Textbox(
                        label="Conversion Status",
                        lines=5,
                        interactive=False,
                    )
            
            refresh_button.click(
                fn=refresh_voices,
                outputs=voice_dropdown,
            )
            
            convert_button.click(
                fn=convert_voice_ui,
                inputs=[
                    source_audio_input,
                    voice_dropdown,
                    pitch_shift_input,
                ],
                outputs=[output_audio, conversion_status],
            )
        
        # About Tab
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ## About RVC Voice Cloning System
                
                This is a production-grade RVC (Retrieval-based Voice Conversion) system that works seamlessly
                on both local GPUs and Google Colab.
                
                ### Features
                - 🎓 **Training**: Train custom voice models from audio samples
                - 🎭 **Inference**: Convert audio to any trained voice
                - 🔄 **Dynamic Discovery**: Trained voices appear automatically
                - 🖥️ **GPU-Agnostic**: Works on CUDA, MPS, or CPU
                
                ### Usage Modes
                - **Local GPU**: Run `python app.py` to launch this Gradio UI
                - **Google Colab**: Open `notebooks/rvc_colab.ipynb` in Colab
                
                ### Architecture
                - **Core Logic**: Pure Python, no UI dependencies
                - **UI Layer**: Thin Gradio wrapper
                - **Colab Integration**: Minimal orchestration
                
                All code lives in this GitHub repository - no duplicate logic!
                """
            )
            
    # Event handlers defined here to ensure all components are in scope
    train_button.click(
        fn=train_and_update,
        inputs=[
            audio_input,
            person_name_input,
            epochs_input,
            batch_size_input,
            learning_rate_input,
        ],
        outputs=[training_status, training_logs, voice_dropdown],
    )


if __name__ == "__main__":
    import os
    
    # Create necessary directories
    Path("models/finetuned_models").mkdir(parents=True, exist_ok=True)
    Path("data/inputs").mkdir(parents=True, exist_ok=True)
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    Path("pretrained").mkdir(parents=True, exist_ok=True)
    
    # Check if running in Colab (enable share mode)
    is_colab = os.environ.get("COLAB_RELEASE_TAG") is not None
    share_mode = os.environ.get("GRADIO_SHARE", "False").lower() == "true" or is_colab
    
    # Launch app
    app.queue().launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=share_mode,  # Enable share in Colab or if GRADIO_SHARE=True
    )
