# RVC Voice Cloning System

A production-grade Retrieval-based Voice Conversion (RVC) system that works seamlessly on both local GPUs (via Gradio UI) and Google Colab (GPU-as-a-Service).

## 🎯 Core Philosophy

**GitHub is the brain. Gradio is the face. Colab is just borrowed muscle.**

- Single source of truth: All logic lives in this repository
- Dual-mode execution: Local GPU or Colab GPU
- Identical results: Same code, same models, same outputs

## 🚀 Quick Start

### Mode A: Local GPU (Gradio UI)

```bash
# Install dependencies
pip install -r requirements.txt

# Download pretrained models (first time only)
python scripts/download_pretrained.py

# Launch Gradio UI
python app.py
```

### Mode B: Google Colab

1. Open `notebooks/rvc_colab.ipynb` in Google Colab
2. Run all cells
3. The notebook will clone this repo and set everything up automatically

## 📁 Repository Structure

```
rvc-system/
├── app.py                 # Gradio UI entry point
├── core/                  # Core ML logic (no UI dependencies)
│   ├── audio/            # Audio preprocessing
│   ├── features/         # Feature extraction (HuBERT, F0, RMVPE)
│   ├── model/            # RVC model definition
│   ├── training/         # Training pipeline
│   └── inference/        # Voice conversion pipeline
├── utils/                # Utilities (device detection, registry)
├── models/               # Trained voice models (auto-discovered)
├── notebooks/            # Colab notebook
└── pretrained/           # Pretrained models (HuBERT, etc.)
```

## 🎨 Features

- **Training UI**: Upload audio, train custom voices
- **Inference UI**: Convert audio to any trained voice
- **Dynamic Voice Registry**: Auto-discovers trained models
- **GPU-Agnostic**: Auto-detects CUDA/MPS/CPU
- **Production-Ready**: Clean, modular, maintainable code

## 📝 Usage

### Training a New Voice

1. Open Gradio UI (local) or Colab notebook
2. Go to "Training" tab
3. Upload audio files (WAV format recommended)
4. Enter a unique person/speaker name
5. Click "Train Model"
6. Wait for training to complete

Trained models are saved to `models/<person_name>/` and automatically appear in the inference dropdown.

### Voice Conversion

1. Go to "Inference" tab
2. Upload source audio
3. Select target voice from dropdown
4. Click "Convert Voice"
5. Download or play the result

## 🔧 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS) or CPU fallback
- See `requirements.txt` for full dependencies

## 📚 Architecture

The system is designed with clear separation of concerns:

- **Core Logic**: Pure Python, no UI dependencies
- **UI Layer**: Thin Gradio wrapper over core functions
- **Colab Integration**: Minimal orchestration, calls core modules

This ensures:
- Same code runs in both modes
- Easy to test and maintain
- Simple to extend (new models, real-time inference, etc.)

## 🤝 Contributing

This is a production system designed for long-term maintenance. Code quality, modularity, and documentation are priorities.

## 📄 License

[Add your license here]
