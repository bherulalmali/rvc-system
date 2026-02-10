# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Create necessary directories
RUN mkdir -p data/inputs data/outputs models/finetuned_models pretrained

# Make port 7860 available
EXPOSE 7860

# Run app.py
CMD ["python", "src/app.py"]
