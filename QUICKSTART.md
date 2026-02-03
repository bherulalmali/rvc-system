# Quick Start Guide

## 🚀 Local GPU Setup (Gradio UI)

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS) or CPU

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rvc-system.git
   cd rvc-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pretrained models** (first time only)
   ```bash
   python scripts/download_pretrained.py
   ```

4. **Launch Gradio UI**
   ```bash
   python app.py
   ```

5. **Open browser**
   - Local: http://localhost:7860
   - The UI will show your device (CUDA/MPS/CPU)

### Using the UI

#### Training Tab
1. Upload audio files (WAV format recommended)
2. Enter a unique person/speaker name
3. Adjust training parameters (optional)
4. Click "Train Model"
5. Wait for training to complete

#### Inference Tab
1. Upload source audio
2. Select target voice from dropdown
3. Adjust pitch shift if needed (optional)
4. Click "Convert Voice"
5. Download or play the result

---

## 🟨 Google Colab Setup

### Steps

1. **Open Colab notebook**
   - Upload `notebooks/rvc_colab.ipynb` to Google Colab
   - Or open directly if repo is public

2. **Update repository URL** (if needed)
   - Edit the first cell to point to your GitHub repo
   - Replace `https://github.com/yourusername/rvc-system.git`

3. **Run all cells**
   - Cell 1: Clone repository
   - Cell 2: Install dependencies
   - Cell 3: Check GPU
   - Cell 4: Download pretrained models
   - Cell 5: Train voice (edit paths)
   - Cell 6: Convert voice (edit paths)
   - Cell 7: Launch Gradio UI (optional)

### Colab Tips

- **Upload audio files**: Use Colab's file upload widget
- **Save models**: Download from `models/` directory
- **Gradio link**: Use the public link from Cell 7
- **GPU**: Colab provides free GPU (T4, V100, etc.)

---

## 📁 Project Structure

```
rvc-system/
├── app.py                    # Gradio UI (local GPU)
├── core/                     # Core ML logic
│   ├── audio/               # Audio preprocessing
│   ├── features/            # Feature extraction
│   ├── model/               # RVC model
│   ├── training/            # Training pipeline
│   └── inference/           # Voice conversion
├── utils/                   # Utilities
├── models/                  # Trained voices (auto-discovered)
├── notebooks/               # Colab notebook
└── scripts/                 # Utility scripts
```

---

## 🔧 Configuration

Edit `config.yaml` to customize:
- Training parameters (epochs, batch size, learning rate)
- Audio processing (sample rate, hop length)
- Feature extraction (F0 method)
- Device settings (auto-detected by default)

---

## 🐛 Troubleshooting

### No GPU detected
- System will fall back to CPU automatically
- Training will be slower but still works

### Audio file errors
- Ensure files are in supported format (WAV, MP3, FLAC)
- Check audio duration (minimum 1 second)
- Verify file is not corrupted

### Model not found
- Ensure training completed successfully
- Check `models/<person_name>/model.pth` exists
- Use "Refresh Voices" button in UI

### Colab connection issues
- Check repository URL is correct
- Ensure repo is public or use authentication
- Verify internet connection in Colab

---

## 📚 Next Steps

- Read `ARCHITECTURE.md` for detailed design
- Read `README.md` for full documentation
- Check `config.yaml` for customization options
- Explore `core/` modules for extension

---

## 💡 Tips

1. **Training**: More audio = better quality (aim for 5-10 minutes)
2. **Quality**: Use clean, high-quality audio recordings
3. **Names**: Use unique, descriptive names for voices
4. **Backup**: Download trained models regularly
5. **GPU**: Colab GPU is free but has time limits

---

## 🆘 Support

- Check GitHub Issues for known problems
- Review `ARCHITECTURE.md` for system design
- Verify all dependencies are installed
- Check logs in Gradio UI for detailed errors
