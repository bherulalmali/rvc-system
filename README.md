# 🎤 RVC Voice Cloning System

A production-grade Retrieval-based Voice Conversion (RVC) system designed for seamless operation on both local GPUs and Google Colab.

---

## 🎯 Core Philosophy

**GitHub is the brain. Gradio is the face. Colab is just borrowed muscle.**

- **Single Source of Truth**: All ML logic lives in the `core/` directory.
- **Dual-Mode Execution**: Run locally with your own GPU or on Google Colab.
- **Identical Results**: Same code, same models, and same quality regardless of where you run it.

---

## 🎨 Key Features

- **Training UI**: Upload audio files and train high-quality custom voice models.
- **Inference UI**: Convert source audio to any of your trained voices with pitch shifting.
- **Dynamic Voice Discovery**: Newly trained models automatically appear in the interface.
- **Device-Agnostic**: Automatically detects and uses CUDA, Apple MPS, or CPU fallbacks.
- **Production Architecture**: Modular "Pure Python" core, separate from UI and orchestration.

---

## 🚀 Getting Started

### Mode A: Local GPU (Gradio UI)

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download Pretrained Models** (Required for first run):
    ```bash
    python scripts/download_pretrained.py
    ```
3.  **Launch the System**:
    ```bash
    python app.py
    ```
4.  **Access UI**: Open `http://localhost:7860` in your browser.

### Mode B: Google Colab (GPU-as-a-Service)

1.  Open `notebooks/rvc_colab.ipynb` in Google Colab.
2.  Set Runtime to **Python 3** and **T4 GPU** (`Runtime` -> `Change runtime type`).
3.  Run the cells sequentially. The notebook will automatically set up the environment and the RVC backend.

---

## 📖 Detailed User Guides

### 🚀 Using Google Colab (Step-by-Step)

This is the recommended way for users without a powerful NVIDIA GPU.

#### 1. Initial Setup
*   **Mount Google Drive**: Run the first cell to connect your Drive. This is where your models (`.pth` and `.index`) will be saved permanently.
*   **Install Backend**: The second cell clones the repository and installs all complex ML dependencies. This takes about 2-3 minutes.
*   **Download Pretrained**: Downloads the base HuBERT and RMVPE models required for RVC to work.

#### 2. Training a Custom Voice
*   **Data Preparation**: Gather 1-10 minutes of clean, dry vocals (no music).
*   **Configure**: In the Training cell, set your `PERSON_NAME` and `EPOCHS` (250-500 for high quality).
*   **Upload**: When you run the cell, a file uploader will appearing. Select your audio files.
*   **Wait**: The notebook will extract features and start training. Once finished, the model is automatically copied to your Google Drive in `/RVC_Models/`.

#### 3. Converting Audio (Inference)
*   **Select Model**: The inference cell automatically lists all models found in the session.
*   **Upload Source**: Upload the audio you want to transform (e.g., your own voice).
*   **Adjust Pitch**: Use `+12` for female voices and `-12` for male voices if needed.
*   **Download**: The output is generated and downloaded automatically as `output_converted.wav`.

#### 4. Google Drive Persistence
Your models are saved to your Drive. To use them in a new session:
*   Mount Drive as usual.
*   Run the **"Load Saved Models from Drive"** cell. It will sync your models back into the temporary Colab workspace.

### Using Local Gradio UI

1.  **Training Tab**: Upload your audio dataset, name the speaker, and click **🚀 Train Model**. Monitor progress via the live logs.
2.  **Inference Tab**: Upload source audio, select the target voice from the dropdown, and click **🎯 Convert Voice**.
3.  **Refresh**: Use the **🔄 Refresh Voices** button if a newly trained model doesn't appear immediately.

---

---

## 📂 Repository Structure

```
RVCVoiceCloning/
├── data/
│   ├── inputs/          # Dataset/Source audio
│   └── outputs/         # Converted audio
├── models/
│   └── finetuned_models/ # Trained voice models
├── src/
│   ├── app.py           # Gradio UI
│   ├── core/            # ML Logic
│   └── utils/           # Utilities
├── Configs/
│   └── config.yaml      # Configuration settings
├── scripts/
│   ├── setup.sh         # Setup script
│   └── ...              # Other scripts
├── notebooks/           # Colab notebooks
├── README.md
├── requirements.txt
├── .gitignore
└── Dockerfile
```

---

## 🛠️ Adding Custom Models

If you have a pretrained RVC model and an index file, follow these steps to add them:

1.  **Create a Folder**: Go to `models/finetuned_models/` and create a subfolder with your voice name (e.g., `MyCustomVoice`).
2.  **Add Files**: Place your `.pth` model file and your `.index` file inside that folder.
3.  **Automatic Detection**: The system will automatically detect the `.index` file for better quality.
4.  **Refresh**: Open the Gradio UI and click **🔄 Refresh Voices** to see your model.

---

## 📄 License

This project is licensed under the [Add License Type] License.
