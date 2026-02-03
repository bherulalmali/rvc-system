# Architecture Documentation

## Overview

This RVC (Retrieval-based Voice Conversion) system is designed with a clear separation of concerns:

- **Core Logic**: Pure Python ML code, no UI dependencies
- **UI Layer**: Thin Gradio wrapper
- **Colab Integration**: Minimal orchestration

## Repository Structure

```
rvc-system/
├── app.py                 # Gradio UI entry point
├── core/                  # Core ML logic (no UI dependencies)
│   ├── audio/            # Audio preprocessing
│   │   ├── preprocess.py # Audio loading, resampling, normalization
│   │   └── utils.py      # Audio validation and info extraction
│   ├── features/         # Feature extraction
│   │   ├── hubert.py     # HuBERT feature extraction
│   │   ├── f0.py         # F0 (pitch) extraction
│   │   └── rmvpe.py      # RMVPE model for F0
│   ├── model/            # RVC model definition
│   │   ├── rvc_model.py  # Model architecture
│   │   └── checkpoint.py # Model save/load
│   ├── training/         # Training pipeline
│   │   └── trainer.py    # Training loop and dataset preparation
│   └── inference/        # Voice conversion pipeline
│       └── converter.py  # Inference and vocoder
├── utils/                # Utilities
│   ├── device.py         # GPU/CPU detection
│   └── registry.py       # Voice model registry
├── models/                # Trained models (auto-discovered)
├── notebooks/            # Colab notebook
│   └── rvc_colab.ipynb   # Colab orchestration
└── scripts/              # Utility scripts
    └── download_pretrained.py
```

## Core Principles

### 1. Single Source of Truth

All ML logic lives in `core/`. Both Gradio UI and Colab notebook call the same functions:

```python
# Used by both app.py and notebooks/rvc_colab.ipynb
from core.training import train_rvc_model
from core.inference import VoiceConverter
```

### 2. No Duplication

- No ML code in Colab notebook
- No ML code in Gradio UI
- All logic in `core/` modules

### 3. Device Agnostic

The system automatically detects and uses:
- CUDA GPU (if available)
- Apple MPS (if available)
- CPU (fallback)

Same code runs on all devices.

### 4. Dynamic Discovery

Trained voices are auto-discovered:
- Scans `models/` directory
- Maintains registry metadata
- Updates UI automatically

## Data Flow

### Training Flow

```
Audio Files → Preprocessing → Feature Extraction → Model Training → Checkpoint
     ↓              ↓                  ↓                ↓              ↓
  Validate      Resample          HuBERT + F0      Training Loop   Save to
  Audio         Normalize         Extract          Optimize        models/
```

### Inference Flow

```
Source Audio → Feature Extraction → Model Inference → Vocoder → Converted Audio
     ↓                ↓                  ↓             ↓            ↓
  Load Audio      HuBERT + F0        Forward Pass   Mel→Audio    Save Output
```

## Module Responsibilities

### `core/audio/`

- **preprocess.py**: Audio loading, resampling, normalization
- **utils.py**: File validation, audio info extraction

### `core/features/`

- **hubert.py**: HuBERT model loading and feature extraction
- **f0.py**: F0 extraction (supports RMVPE, DIO methods)
- **rmvpe.py**: RMVPE model for high-quality F0

### `core/model/`

- **rvc_model.py**: RVC model architecture (encoder-decoder)
- **checkpoint.py**: Model serialization and loading

### `core/training/`

- **trainer.py**: Training loop, dataset preparation, checkpointing

### `core/inference/`

- **converter.py**: Voice conversion pipeline, vocoder integration

### `utils/`

- **device.py**: Device detection and logging
- **registry.py**: Voice model discovery and registration

## Execution Modes

### Mode A: Local GPU (Gradio)

```bash
python app.py
```

- Launches Gradio UI
- Uses local GPU/CPU
- Models saved to `models/`
- UI auto-updates with new voices

### Mode B: Google Colab

1. Open `notebooks/rvc_colab.ipynb`
2. Run cells in order
3. Notebook clones repo and calls `core/` functions
4. Optionally launches Gradio UI

## Extension Points

### Adding New Models

1. Add model class to `core/model/`
2. Update `core/training/trainer.py` to use new model
3. Update `core/inference/converter.py` for inference

### Adding New Features

1. Add feature extractor to `core/features/`
2. Update training/inference to use new features

### Adding New UIs

1. Create new UI file (e.g., `app_streamlit.py`)
2. Import from `core/` modules
3. Call same functions as Gradio UI

## Testing Strategy

- Unit tests for `core/` modules (no UI dependencies)
- Integration tests for training/inference pipelines
- UI tests for Gradio interface
- Colab notebook can be tested manually

## Performance Considerations

- GPU acceleration for training and inference
- Batch processing for multiple files
- Model checkpointing for long training runs
- Efficient feature extraction (caching HuBERT model)

## Future Enhancements

- Real-time inference
- WebSocket API
- Model quantization
- Distributed training
- Additional vocoders (HiFi-GAN, etc.)
