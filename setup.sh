#!/bin/bash
# Setup script for RVC Voice Cloning System

set -e

echo "🎤 RVC Voice Cloning System - Setup"
echo "===================================="

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p outputs
mkdir -p pretrained/hubert
mkdir -p pretrained/rmvpe
mkdir -p datasets

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download pretrained models (optional)
read -p "Download pretrained models now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Downloading pretrained models..."
    python scripts/download_pretrained.py
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download pretrained models: python scripts/download_pretrained.py"
echo "2. Launch Gradio UI: python app.py"
echo "3. Or use Google Colab: notebooks/rvc_colab.ipynb"
