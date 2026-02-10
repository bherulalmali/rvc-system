import os
import requests
from pathlib import Path

def download_file(url: str, dest_path: Path):
    """Download a file with progress logging."""
    print(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Downloaded: {dest_path.name}")

def main():
    # Model URLs (Placeholders for production)
    MODELS = {
        "hubert": {
            "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
            "path": Path("pretrained/hubert/hubert_base.pt")
        },
        "rmvpe": {
            "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
            "path": Path("pretrained/rmvpe/rmvpe.pt")
        }
    }

    print("🚀 Downloading RVC Pretrained Models...")
    
    for name, info in MODELS.items():
        if not info["path"].exists():
            try:
                download_file(info["url"], info["path"])
            except Exception as e:
                print(f"❌ Failed to download {name}: {e}")
        else:
            print(f"✅ {name} already exists.")

    print("\n🎉 Pretrained models are ready!")

if __name__ == "__main__":
    main()
