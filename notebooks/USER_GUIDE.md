# 📖 RVC Voice Cloning - Google Colab User Guide

This guide explains how to use the `rvc_colab.ipynb` notebook to train and use RVC (Retrieval-based Voice Conversion) models on Google Colab. This is ideal for users who do not have a powerful local GPU.

---

## 🛠️ Prerequisites

1.  **Google Account**: Required for Google Colab and Google Drive.
2.  **GPU Runtime**: Google Colab provides free/paid GPUs. Ensure your runtime type is set to **Python 3** and **T4 GPU** (or better).
    -   Go to `Runtime` -> `Change runtime type` -> Select `T4 GPU`.

---

## 🚀 Step-by-Step Workflow

### 1. Mount Google Drive
Run the first cell to mount your Google Drive. This is **critical** because:
-   It stores your trained models persistentely.
-   Colab's local storage is deleted after the session ends.
-   Models will be saved in `/MyDrive/RVC_Models`.

### 2. Setup Environment
This step clones the repository and prepares the backend. 
-   **Note**: You can change the `REPO_URL` if you are using a specific fork.
-   Wait for the setup to complete. It installs necessary system packages and Python libraries.

### 3. Load Saved Models (Optional)
If you have previously trained models in your Google Drive, run this cell to sync them into the current session's workspace.

### 4. Train a New Voice
This is where the magic happens.
1.  **PERSON_NAME**: Enter a unique name for the voice (e.g., `elon_musk`).
2.  **EPOCHS**: Set the training length. 
    -   `50` is a good starting point for testing.
    -   `200-500` is recommended for high quality.
3.  **Upload Audio**: When prompted, upload clean `.wav` files of the target voice.
    -   Avoid background noise or music.
    -   1-5 minutes of audio is usually sufficient.
4.  **Wait**: The notebook will patch dependencies (specifically for Python 3.12 compatibility), extract features, and start training.
5.  **Persistence**: The final model (`.pth`) and index file will be automatically copied to your Google Drive.

### 5. Voice Conversion (Inference)
Use this section to convert any audio file into your trained voice.
1.  **Select Voice**: It will list available models in the `models/` folder.
2.  **Upload Source**: Upload the audio file you want to "transform".
3.  **Download**: The converted audio will be automatically downloaded to your computer as `output_converted.wav`.

---

## 💡 Pro Tips

-   **Runtime Disconnection**: Large training runs can take hours. Colab might disconnect if you are inactive. Use a mouse jiggler or keep the tab active.
-   **Audio Quality**: The quality of the output depends 90% on the quality of the training data. Use high-quality, dry (no reverb) vocals.
-   **Index Files**: Always use the `.index` file during conversion for better accent and tone matching.

---

## ❓ Troubleshooting

-   **"Out of Memory (OOM)"**: If the training crashes with a GPU memory error, try reducing the `batch_size` in the training script or using shorter audio clips.
-   **Drive Mount Failed**: Ensure you approve the Google Drive access pop-up.
-   **Dependency Errors**: If a library fails to install, try restarting the runtime (`Runtime` -> `Restart session`) and running the setup again.

---

# 🖥️ Local Gradio UI User Guide (`app.py`)

The Gradio UI provides a user-friendly web interface for running the RVC Voice Cloning system locally on your machine.

## 🛠️ Prerequisites

1.  **Python Environment**: Ensure you have Python 3.8+ installed.
2.  **Dependencies**: Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  **GPU (Highly Recommended)**: A CUDA-enabled NVIDIA GPU is recommended for reasonable training times. If no GPU is found, the system will run on CPU (very slow) or in "Mock Mode" if RVC dependencies are missing.

---

## 🚀 How to Run

1.  Open your terminal in the project root directory.
2.  Run the application:
    ```bash
    python app.py
    ```
3.  Open the provided URL in your browser (usually `http://127.0.0.1:7860`).

---

## 🏗️ Interface Overview

### 🎓 Training Tab
Use this tab to create a new voice model.
1.  **Upload Audio Files**: Select multiple `.wav` or `.mp3` files of the person you want to clone.
2.  **Person/Speaker Name**: Provide a unique identifier for this voice.
3.  **Epochs**: Set how many times the model should learn from the data (default: 50).
4.  **Advanced Settings**: Adjust `Batch Size` and `Learning Rate` if you are an advanced user.
5.  **Status & Logs**: Monitor the training progress through the text boxes on the right.

### 🎭 Inference Tab
Use this tab to convert audio using a trained voice.
1.  **Source Audio**: Upload the audio file you want to convert (or record from microphone).
2.  **Target Voice**: Select a voice from the dropdown menu.
    -   *If your voice isn't there, click the **🔄 Refresh Voices** button.*
3.  **Pitch Shift**: Adjust the pitch (e.g., set to `12` for female-to-male or `-12` for male-to-female).
4.  **Convert**: Click **🎯 Convert Voice** and wait for the result. Use the audio player to listen and download.

### ℹ️ About Tab
Contains technical details about the system and current device info.

---

## ❓ Local Troubleshooting

-   **"Mock Mode" Warning**: This means the RVC-specific dependencies (like `fairseq` or `librosa` versions) are incompatible or missing. The UI will still work but will simulate training/inference.
-   **Port in Use**: If `7860` is taken, Gradio will automatically try `7861`, `7862`, etc.
-   **Slow Conversion**: If you are running on CPU, conversion can take several minutes for a 30-second clip.
