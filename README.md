# RVC Voice Cloning System

A high-performance implementation of the Retrieval-based Voice Conversion (RVC) framework. This system provides a unified interface for model training and inference, optimized for both local deployment and Google Colab environments.

---

## Technical Overview

The architecture follows a modular "Core-Logic" pattern, separating ML pipelines from the interface layer. This ensures consistent execution across different platforms.

*   **Unified Backend**: All inference and training logic is centralized in the `src/core` module.
*   **Dynamic Discovery**: Automatic registration of new models and feature indexes via the `VoiceRegistry`.
*   **Hardware Compatibility**: Native support for CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallbacks.

---

## Quick Start

### Local Setup (Gradio UI)

1.  **Environment Setup**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Pretrained Assets**:
    ```bash
    python scripts/download_pretrained.py
    ```
3.  **Launch**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python src/app.py
    ```
4.  **UI Access**: Navigate to `http://localhost:7860`.

### Google Colab Execution

1.  Open `notebooks/rvc_colab.ipynb`.
2.  Select **T4 GPU** runtime.
3.  Execute cells sequentially to initialize the environment and backend.

---

## User Guides

### Google Colab Workflow

1.  **Mount Google Drive**: Establish persistence for models and converted audio.
2.  **Environment Setup**: Clones the repository and installs system-level dependencies.
3.  **Training**:
    *   Upload clean audio dataset (WAV format recommended).
    *   Configure `PERSON_NAME` and `EPOCHS`.
    *   Trained models are automatically exported to `/RVC_Models/` on your Drive.
4.  **Inference**:
    *   Select target voice and upload source audio.
    *   Adjust pitch shift (+12 for female-to-male, -12 for male-to-female).
5.  **Asset Management**: Use Step 6 to package and download your `.pth`, `.index`, and converted audio files.

### Local CLI/UI Workflow

-   **Training**: Input speaker name and dataset path. Monitor logs via the UI terminal.
-   **Inference**: Real-time conversion with automated `.index` file detection for improved quality.

---

## Repository Structure

```text
rvc-system/
├── data/
│   ├── inputs/               # Raw datasets and source audio
│   └── outputs/
│       └── {person_name}/    # Organized inference exports
├── models/
│   ├── finetuned_models/
│   │   └── {person_name}/    # .pth and .index model pairs
│   └── pretrained/           # Hubert/RMVPE base models
├── src/
│   ├── app.py                # Gradio entry point
│   ├── core/
│   │   ├── training/         # Training pipelines
│   │   ├── inference/        # Conversion logic
│   │   └── audio/            # Signal processing utils
│   └── utils/                # Device and Registry management
├── notebooks/
│   └── rvc_colab.ipynb       # GPU-accelerated notebook
├── requirements.txt          # Mandatory dependencies
└── Dockerfile                # Production container spec
```

---

## Adding Custom Models

To integrate external RVC models:

1.  Place your `.pth` and `.index` files in `models/finetuned_models/{VoiceName}/`.
2.  The system automatically detects the index file for feature retrieval.
3.  Click **Refresh Voices** in the UI to register the new model.

---

## License

This project is maintained for production use cases. Please refer to the [MIT License](LICENSE) for terms of use.
